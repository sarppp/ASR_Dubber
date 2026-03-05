"""
nemo_audio.py — Audio helpers and subtitle assembly for nemo pipeline.

No top-level nemo.collections imports — see nemo_model.py for the sys.path workaround.
"""

import logging
import subprocess
import wave
from pathlib import Path

import torch

log = logging.getLogger("nemo_local")

# ── Model registry ────────────────────────────────────────────────────────────
# Friendly shortname → full NeMo model ID.
# Used by --asr-model in nemo.py and run_pipeline.py.
ASR_MODELS = {
    # CTC/TDT models — word-level timestamps, no chunk quality limit
    "parakeet-v2":  "nvidia/parakeet-tdt-0.6b-v2",   # English only, fastest
    "parakeet-v3":  "nvidia/parakeet-tdt-0.6b-v3",   # 25 EU langs, same speed
    # NeMo encoder-decoder models — segment timestamps, 60s chunk cap
    "canary":       "nvidia/canary-1b-v2",            # EN/DE/FR/ES + translation
    # Qwen3-ASR (qwen-asr package) — word timestamps via ForcedAligner, 30 langs
    "qwen3-asr":    "Qwen/Qwen3-ASR-1.7B",           # 30 langs, ~5GB VRAM, best quality
    "qwen3-asr-s":  "Qwen/Qwen3-ASR-0.6B",           # 30 langs, ~2GB VRAM, faster
}

MODEL_EN    = ASR_MODELS["parakeet-v3"]   # v3 supports EN + 25 EU langs, same speed as v2
MODEL_MULTI = ASR_MODELS["parakeet-v3"]
                                           # 25 langs incl. FR/DE with word timestamps

# Languages auto-routed to MODEL_MULTI (parakeet-v3 supported languages)
MULTI_LANGS = {
    "fr", "de", "es", "it", "nl", "pl", "pt", "ru", "sv", "da",
    "fi", "cs", "sk", "sl", "hr", "ro", "hu", "bg", "el", "et",
    "lv", "lt", "uk", "mt",
}  # English ("en") is excluded — it stays on the faster parakeet-v2

VIDEO_EXT   = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}
CHUNK_OVERLAP_SEC = 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_ts(s: float) -> str:
    h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s%1)*1000):03d}"

def _fmt_dur(s: float) -> str:
    return f"{int(s//60)}m{int(s%60):02d}s" if s >= 60 else f"{s:.1f}s"

def _vram_gb() -> tuple[float, float]:
    if not torch.cuda.is_available(): return 0.0, 0.0
    free, total = torch.cuda.mem_get_info()
    return free / 1024**3, total / 1024**3

def _audio_duration(path: str) -> float:
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0

def _extract_audio(video_path: str, out_path: str, trim_sec: int = 0) -> None:
    cmd = ["ffmpeg", "-y", "-threads", "0", "-i", video_path]
    if trim_sec > 0:
        cmd += ["-t", str(trim_sec)]
    cmd += ["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out_path]
    subprocess.run(cmd, check=True, capture_output=True)

def _chunk_audio(audio_path: str, work_dir: Path, chunk_sec: int) -> list[tuple[str, float]]:
    dur = _audio_duration(audio_path)
    if dur <= chunk_sec + 5:
        return [(audio_path, 0.0)]
    chunks, step, offset, idx = [], chunk_sec - CHUNK_OVERLAP_SEC, 0.0, 0
    while offset < dur:
        cp = str(work_dir / f"_chunk_{idx:04d}.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-threads", "0", "-ss", str(offset), "-i", audio_path,
             "-t", str(min(chunk_sec, dur - offset)),
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", cp],
            check=True, capture_output=True,
        )
        chunks.append((cp, offset)); offset += step; idx += 1
    return chunks

def _srt_last_timestamp(srt: str) -> float:
    """Return the end timestamp (seconds) of the last SRT block, or 0.0."""
    import re
    pattern = re.compile(r"\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})")
    matches = list(pattern.finditer(srt))
    if not matches:
        return 0.0
    h, m, s, ms = [int(x) for x in matches[-1].groups()]
    return h * 3600 + m * 60 + s + ms / 1000.0


def _cleanup_chunks(manifest: list, keep: str) -> None:
    for e in manifest or []:
        p = e.get("path")
        if p and p != keep:
            Path(p).unlink(missing_ok=True)


def _strip_special_tokens(text: str) -> str:
    """Remove model special tokens and leftover artifacts from ASR output.

    Encoder-decoder models (Canary, Whisper) sometimes emit <|endoftext|> or
    other <|...|> tokens when they run past the actual speech content.  The
    trailing content looks like:
        '...real text.<|endoftext|><|endoftext|>...<|endoftext|>.<|endoftext|>....'
    """
    import re
    # Strip trailing <|endoftext|> storm first (before dot collapse) so the
    # sentence-ending period is not merged into the artifact dot run.
    # Pattern: one or more (<|endoftext|> followed by any whitespace/dots) at end.
    text = re.sub(r"(<\|endoftext\|>[\s.]*)+$", "", text)
    # Remove remaining isolated special tokens (mid-text, e.g. <|startoftranscript|>)
    text = re.sub(r"<\|[^|>]+\|>", "", text)
    # Collapse runs of 4+ dots to ellipsis
    text = re.sub(r"\.{4,}", "...", text)
    # Lone "..." left after token removal → empty
    if text.strip() == "...":
        text = ""
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _strip_asr_repetition(text: str, min_unit_words: int = 5, min_reps: int = 3) -> str:
    """
    Remove hallucinated repetition loops from Canary/Whisper ASR output.

    Encoder-decoder ASR models sometimes get stuck repeating the same phrase.
    Finds any phrase of ≥min_unit_words words that repeats ≥min_reps consecutive
    times anywhere in the text, then truncates to keep only the first copy.

    Example (Canary de hallucination):
        'Das ist gut für die Gesundheit. Das ist gut für die Gesundheit. Das ist gut...'
        → 'Das ist gut für die Gesundheit.'
    """
    words = text.split()
    n = len(words)
    if n < min_unit_words * min_reps:
        return text

    for start in range(n - min_unit_words * min_reps + 1):
        for unit_len in range(min_unit_words, (n - start) // min_reps + 1):
            unit = words[start:start + unit_len]
            reps, pos = 1, start + unit_len
            while pos + unit_len <= n and words[pos:pos + unit_len] == unit:
                reps += 1
                pos += unit_len
            if reps >= min_reps:
                stripped = " ".join(words[:start + unit_len])
                log.info(
                    f"ASR repetition stripped: {reps}× repeat of {unit_len}-word phrase "
                    f"starting at word {start} — removed {len(text) - len(stripped)} chars"
                )
                return stripped
    return text


# ── Subtitle assembly ─────────────────────────────────────────────────────────

def _words_to_segs(words, max_w=10, max_dur=5.0, max_ch=80, diarized=False):
    segs, cur_w, cur_t, cur_s, cur_spk = [], [], "", None, None
    for w in words:
        word = w.get("word", "").strip()
        if not word: continue
        ws, we = w.get("start", 0.0), w.get("end", 0.0)
        spk = w.get("speaker", "unknown") if diarized else None
        if cur_s is None: cur_s, cur_spk = ws, spk
        cand = (cur_t + " " + word).strip() if cur_t else word
        split = (len(cur_w) >= max_w or (we - cur_s) > max_dur or len(cand) > max_ch
                 or (cur_t and cur_t[-1] in ".!?" and len(cur_w) >= 3)
                 or (diarized and spk != cur_spk and cur_w))
        if split and cur_w:
            seg = {"start": cur_s, "end": cur_w[-1].get("end", cur_s), "text": cur_t}
            if diarized: seg["speaker"] = cur_spk
            segs.append(seg)
            cur_w, cur_t, cur_s, cur_spk = [], "", ws, spk
            cand = word  # reset cand — cur_t is now "" so the old cand is stale
        cur_w.append(w); cur_t = cand
    if cur_w and cur_t.strip():
        seg = {"start": cur_s, "end": cur_w[-1].get("end", cur_s), "text": cur_t}
        if diarized: seg["speaker"] = cur_spk
        segs.append(seg)
    return segs

def _split_coarse_segs(segs, max_w=10, max_ch=80):
    """Split Canary's single large segments into subtitle-sized lines."""
    out = []
    for seg in segs:
        words = seg.get("text", "").strip().split()
        if not words: continue
        start, dur = seg.get("start", 0.0), max(0.1, seg.get("end", 0.0) - seg.get("start", 0.0))
        spk = seg.get("speaker")
        lines, cur = [], []
        for word in words:
            cur.append(word)
            if len(cur) >= max_w or len(" ".join(cur)) >= max_ch:
                lines.append(cur); cur = []
        if cur: lines.append(cur)
        total = sum(len(l) for l in lines)
        t = start
        for line in lines:
            frac = len(line) / total if total else 1 / len(lines)
            entry = {"text": " ".join(line), "start": t, "end": t + dur * frac}
            if spk is not None: entry["speaker"] = spk
            out.append(entry); t += dur * frac
    return out

def _segs_to_srt(segs, diarized=False):
    if diarized:
        spk_list = sorted({s.get("speaker", "unknown") for s in segs})
        spk_map = {s: f"Speaker {i+1}" for i, s in enumerate(spk_list)}
    lines, idx, prev_text, prev_spk = [], 0, None, None
    for s in segs:
        t = s["text"].strip()
        if not t:
            continue
        spk = s.get("speaker") if diarized else None
        # Deduplicate consecutive identical segments (both modes).
        # In diarized mode only skip if the same speaker repeats — different speakers
        # saying the same word is rare but valid; same speaker is always hallucination.
        if t == prev_text and (not diarized or spk == prev_spk):
            continue
        idx += 1
        label = f"[{spk_map.get(s.get('speaker', 'unknown'), 'Speaker ?')}] " if diarized else ""
        lines += [str(idx), f"{_fmt_ts(s['start'])} --> {_fmt_ts(s['end'])}", label + t, ""]
        prev_text = t
        prev_spk = spk
    return "\n".join(lines)
