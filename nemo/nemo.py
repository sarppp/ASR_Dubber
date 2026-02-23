"""
nemo3.py — NeMo local ASR/translation runner

Models:
  nvidia/parakeet-tdt-0.6b-v2   — fast English-only (auto-selected for --language en)
  nvidia/canary-1b-v2           — multilingual ASR + translation (de/fr/es ↔ en)

Usage:
  python nemo3.py video.mp4 --language de --precision fp16
  python nemo3.py video.mp4 --language de --translate          # de → en
  python nemo3.py --language en --precision fp16 --all         # batch all pending
  python nemo3.py video.mp4 --language de --diarize
"""

import argparse
import gc
import json
import logging
import os
import subprocess
import time
import wave
from pathlib import Path

import torch
from omegaconf import OmegaConf

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-8s │ %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("nemo_local")

MODEL_EN    = "nvidia/parakeet-tdt-0.6b-v2"
MODEL_MULTI = "nvidia/canary-1b-v2"
MULTI_LANGS = {"fr", "de", "es"}
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

def _cleanup_chunks(manifest: list, keep: str) -> None:
    for e in manifest or []:
        p = e.get("path")
        if p and p != keep:
            Path(p).unlink(missing_ok=True)


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
            continue
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
    lines, idx, prev = [], 0, None
    for s in segs:
        t = s["text"].strip()
        if not t or (not diarized and t == prev): continue
        idx += 1
        label = f"[{spk_map.get(s.get('speaker', 'unknown'), 'Speaker ?')}] " if diarized else ""
        lines += [str(idx), f"{_fmt_ts(s['start'])} --> {_fmt_ts(s['end'])}", label + t, ""]
        prev = t
    return "\n".join(lines)


# ── Import helpers ───────────────────────────────────────────────────────────

def _import_nemo_asr():
    """Load nemo.collections.asr even if this script is named nemo.py."""
    import importlib
    import sys

    script_dir = Path(__file__).resolve().parent
    original_path = list(sys.path)

    def _is_script_dir(entry: str | None) -> bool:
        if entry is None:
            return False
        try:
            resolved = Path(entry or ".").resolve()
        except Exception:
            return False
        return resolved == script_dir

    try:
        sys.path = [entry for entry in original_path if not _is_script_dir(entry)]
        nemo_asr = importlib.import_module("nemo.collections.asr")
    finally:
        sys.path = original_path

    return nemo_asr

# ── Hypothesis helpers (shared with legacy nemo_local_asr) ───────────────────

def _hyp_timestamps(hyp) -> dict | None:
    if hyp is None:
        return None
    for attr in ("timestamp", "timestep", "timestamps"):
        if isinstance(hyp, dict):
            ts = hyp.get(attr)
        else:
            ts = getattr(hyp, attr, None)
        if ts:
            return ts
    return None

def _hyp_field(hyp, attr: str, default=None):
    if hyp is None:
        return default
    if isinstance(hyp, dict):
        return hyp.get(attr, default)
    return getattr(hyp, attr, default)

def _looks_like_hyp(obj) -> bool:
    if obj is None:
        return False
    if isinstance(obj, dict):
        return any(k in obj for k in ("text", "words", "timestamp", "timestep", "timestamps"))
    return any(hasattr(obj, attr) for attr in ("text", "words", "timestamp", "timestep", "timestamps"))

def _extract_first_hypothesis(batch_output):
    if batch_output is None:
        return None
    if _looks_like_hyp(batch_output):
        return batch_output
    if isinstance(batch_output, (list, tuple)):
        for item in batch_output:
            hyp = _extract_first_hypothesis(item)
            if hyp is not None:
                return hyp
    return None


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model(model_name: str, precision: str, device: str):
    nemo_asr = _import_nemo_asr()

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache(); gc.collect()
        free, total = _vram_gb()
        log.info(f"VRAM before load : {free:.2f}/{total:.2f} GB free")
        min_gb = 4.0 if precision == "fp32" else 2.5
        if free < min_gb:
            raise RuntimeError(f"Only {free:.2f} GB VRAM free — need {min_gb:.1f} GB for {precision}")

    t0 = time.perf_counter()
    log.info("Loading model…")
    try:
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=model_name, map_location=None if device == "cuda" else "cpu")
    except Exception as e:
        if device != "cuda": raise
        log.warning(f"Direct GPU load failed ({e}); loading on CPU first")
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, map_location="cpu")

    if device == "cuda":
        torch.cuda.empty_cache(); gc.collect()
        dtype = (torch.bfloat16 if precision == "bf16" and torch.cuda.is_bf16_supported()
                 else torch.float16 if precision == "fp16" else torch.float32)
        log.info(f"Moving to GPU [{dtype}] layer-by-layer…")
        for _, module in model.named_children():
            module.to(dtype).to(device); torch.cuda.empty_cache()
        model = model.to(device)
        log.info(f"Precision : {dtype}")

    model.eval()
    torch.cuda.empty_cache(); gc.collect()
    load_sec = time.perf_counter() - t0
    free_after, _ = _vram_gb()
    log.info(f"Model loaded {load_sec:.1f} s | VRAM used {free - free_after:.2f} GB | free {free_after:.2f} GB")

    if device == "cuda" and free_after > 1.0:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            log.info("torch.compile(reduce-overhead) active — first chunk warms up")
        except Exception: pass

    return model


# ── Transcription: Parakeet ───────────────────────────────────────────────────

def _transcribe_parakeet(model, audio_path: str, offset: float) -> tuple[list, list]:
    with torch.inference_mode():
        out = model.transcribe([audio_path], batch_size=1, timestamps=True)

    hyp = _extract_first_hypothesis(out)
    if hyp is None:
        log.error("Parakeet returned no recognizable hypothesis")
        return [], []

    text = _hyp_field(hyp, "text", "")
    if isinstance(text, (list, tuple)):
        text = " ".join(str(t) for t in text)
    elif not isinstance(text, str):
        text = str(text)

    all_words = []
    ts = _hyp_timestamps(hyp)
    words_str = _hyp_field(hyp, "words", []) or []

    if ts and isinstance(ts, dict) and "word" in ts:
        prev_cum = ""
        for i, td in enumerate(ts.get("word") or []):
            if not isinstance(td, dict):
                continue
            s = td.get("start", 0.0)
            e = td.get("end", 0.0)
            w = td.get("word", None)
            if not w and i < len(words_str):
                w = words_str[i]
            w = str(w or "").strip()
            if w:
                cum = w
                if prev_cum and cum.startswith(prev_cum):
                    delta = cum[len(prev_cum):].strip()
                    if not delta:
                        prev_cum = cum
                        continue
                    w = delta
                prev_cum = cum
            if "\u00a0" in w:
                w = w.replace("\u00a0", " ")
            if " " in w:
                w = w.split()[-1]
            if not w:
                continue
            all_words.append({"word": w, "start": float(s) + offset, "end": float(e) + offset})

    if not all_words and ts and isinstance(ts, dict):
        log.warning("No 'word' key in Parakeet timestamps; scanning all keys")
        for key in ts:
            items = ts[key]
            if isinstance(items, list) and items and isinstance(items[0], dict):
                for item in items:
                    w = (item.get("word") or item.get("char") or item.get("label") or item.get("segment") or "")
                    if w:
                        all_words.append({
                            "word": str(w),
                            "start": float(item.get("start", 0.0)) + offset,
                            "end": float(item.get("end", 0.0)) + offset,
                        })
                if all_words:
                    break

    if not all_words and ts and isinstance(ts, dict) and "segment" in ts:
        segs_out = []
        for seg in ts["segment"]:
            segs_out.append({
                "text": str(seg.get("segment", "")),
                "start": float(seg.get("start", 0.0)) + offset,
                "end": float(seg.get("end", 0.0)) + offset,
            })
        return [], segs_out

    return all_words, []


# ── Transcription: Canary ─────────────────────────────────────────────────────

def _transcribe_canary(model, audio_path: str, offset: float, src_lang: str, tgt_lang: str) -> tuple[list, list]:
    from canary_patch import patch_canary2_eos_assert, patch_manifest_lang, build_transcription_config

    patch_canary2_eos_assert()
    patch_manifest_lang(src_lang, tgt_lang)
    cfg = build_transcription_config(src_lang, tgt_lang)

    with torch.inference_mode():
        try:
            out = (model.transcribe([audio_path], override_config=cfg) if cfg
                   else model.transcribe([audio_path], batch_size=1,
                                         source_lang=src_lang, target_lang=tgt_lang))
        except Exception as exc:
            log.error(f"Canary transcribe failed: {exc}"); raise

    # NeMo 2.1+ returns list[str] directly
    text = ""
    if out:
        first = out[0]
        if isinstance(first, str):
            text = first
        else:
            for attr in ("text", "pred_text", "transcription"):
                v = first.get(attr) if isinstance(first, dict) else getattr(first, attr, None)
                if v and isinstance(v, str): text = v; break
    text = text.strip()
    log.info(f"Canary output: {len(text)} chars | {text[:80]!r}")

    audio_dur = _audio_duration(audio_path)
    seg = {"text": text, "start": offset,
           "end": offset + (audio_dur if audio_dur > 0 else max(1.0, len(text.split()) * 0.4))}
    return [], [seg]


# ── Chunked transcription with OOM retry ─────────────────────────────────────

def _transcribe_chunked(model, audio_path: str, model_name: str,
                         src_lang: str, tgt_lang: str, chunk_sec: int):
    is_canary = "canary" in model_name.lower()
    work_dir = Path(audio_path).parent
    dur = _audio_duration(audio_path)

    while chunk_sec >= 30:
        manifest = []
        try:
            if dur <= chunk_sec:
                manifest = [{"path": audio_path, "offset": 0.0}]
                log.info(f"Single-pass — full {_fmt_dur(dur)} fits in one chunk")
            else:
                raw = _chunk_audio(audio_path, work_dir, chunk_sec)
                manifest = [{"path": p, "offset": off} for p, off in raw]
                log.info(f"Chunk size {_fmt_dur(chunk_sec)} → {len(manifest)} chunk(s)")

            all_words, all_segs = [], []
            for ci, entry in enumerate(manifest):
                path, offset = entry["path"], entry["offset"]
                t1 = time.perf_counter()
                if is_canary:
                    words, segs = _transcribe_canary(model, path, offset, src_lang, tgt_lang)
                else:
                    words, segs = _transcribe_parakeet(model, path, offset)
                elapsed = time.perf_counter() - t1
                if len(manifest) > 1:
                    free, _ = _vram_gb()
                    log.info(f"  chunk {ci+1}/{len(manifest)}: {elapsed:.1f} s | VRAM free {free:.2f} GB")
                all_words.extend(words); all_segs.extend(segs)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache(); gc.collect()

            # Dedup overlapping words from chunk overlap
            if len(manifest) > 1 and all_words:
                out, prev = [all_words[0]], all_words[0]
                for w in all_words[1:]:
                    if not (w["start"] < prev["end"] - 0.05 and w["word"] == prev["word"]):
                        out.append(w); prev = w
                all_words = out

            return all_words, all_segs, manifest

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                _cleanup_chunks(manifest, audio_path); raise
            torch.cuda.empty_cache(); gc.collect()
            old = chunk_sec; chunk_sec = max(30, chunk_sec // 2)
            log.warning(f"OOM at chunk={_fmt_dur(old)} → retrying with {_fmt_dur(chunk_sec)}")
            _cleanup_chunks(manifest, audio_path)

    raise RuntimeError("Could not fit even 30s chunks in VRAM")


def _estimate_chunk_sec(model_name: str, safety: float, reserve_gb: float) -> int:
    free, _ = _vram_gb()
    if free <= 0: return 300
    usable = max(0.0, free - reserve_gb) * safety
    if usable <= 0: return 60
    gb_per_min = 0.28 if "parakeet" in model_name.lower() else 0.50
    secs = max(30, min(int(usable / gb_per_min * 60), 600))
    log.info(f"VRAM {free:.2f} GB free → usable {usable:.2f} GB → chunk target {_fmt_dur(secs)}")
    return secs


# ── Diarization ───────────────────────────────────────────────────────────────

def _run_diarization(audio_path: str, work_dir: Path) -> list:
    import shutil
    from nemo.collections.asr.models import ClusteringDiarizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Running speaker diarization…")
    ddir = work_dir / "_diarize"
    ddir.mkdir(parents=True, exist_ok=True)
    mpath = ddir / "manifest.json"
    mpath.write_text(json.dumps({
        "audio_filepath": str(Path(audio_path).resolve()), "offset": 0,
        "duration": None, "label": "infer", "text": "",
        "num_speakers": None, "rttm_filepath": "", "uem_filepath": "",
    }) + "\n", encoding="utf-8")
    cfg = OmegaConf.create({
        "name": "ClusterDiarizer", "num_workers": 0, "sample_rate": 16000,
        "batch_size": 16, "device": device, "verbose": True,
        "diarizer": {
            "manifest_filepath": str(mpath), "out_dir": str(ddir),
            "oracle_vad": False, "collar": 0.25, "ignore_overlap": True,
            "vad": {"model_path": "vad_multilingual_marblenet", "parameters": {
                "window_length_in_sec": 0.63, "shift_length_in_sec": 0.01,
                "smoothing": False, "overlap": 0.5, "onset": 0.9, "offset": 0.5,
                "pad_onset": 0.0, "pad_offset": 0.0,
                "min_duration_on": 0.0, "min_duration_off": 0.6, "filter_speech_first": True,
            }},
            "speaker_embeddings": {"model_path": "titanet_large", "parameters": {
                "window_length_in_sec": [1.5, 1.0, 0.5], "shift_length_in_sec": [0.75, 0.5, 0.25],
                "multiscale_weights": [1, 1, 1], "save_embeddings": False,
            }},
            "clustering": {"parameters": {
                "oracle_num_speakers": False, "max_num_speakers": 8,
                "enhanced_count_thres": 80, "max_rp_threshold": 0.25,
                "sparse_search_volume": 30, "maj_vote_spk_count": False,
                "chunk_cluster_count": 50, "embeddings_per_chunk": 10000,
            }},
        },
    })
    ClusteringDiarizer(cfg=cfg).to(device).diarize()
    rttm_files = list((ddir / "pred_rttms").glob("*.rttm")) or list(ddir.rglob("*.rttm"))
    turns = []
    if rttm_files:
        for line in rttm_files[0].read_text().splitlines():
            parts = line.split()
            if len(parts) >= 8 and parts[0].upper() == "SPEAKER":
                try:
                    s, d = float(parts[3]), float(parts[4])
                    turns.append({"speaker": parts[7], "start": s, "end": s + d})
                except (ValueError, IndexError): pass
    turns.sort(key=lambda t: t["start"])
    log.info(f"Diarization done — {len({t['speaker'] for t in turns})} speaker(s), {len(turns)} turns")
    shutil.rmtree(ddir, ignore_errors=True)
    return turns

def _assign_speakers(items: list, turns: list) -> list:
    for item in items:
        s, e = item.get("start", 0.0), item.get("end", 0.0)
        best_spk, best_ov = "unknown", 0.0
        for t in turns:
            ov = max(0.0, min(e, t["end"]) - max(s, t["start"]))
            if ov > best_ov: best_ov, best_spk = ov, t["speaker"]
        item["speaker"] = best_spk
    return items


# ── Main pipeline ─────────────────────────────────────────────────────────────

def _run_with_model(model, video_path: str, language: str, model_name: str,
                     translate: bool, diarize: bool, trim_sec: int,
                     safety_factor: float, reserve_gb: float, chunk_override_sec) -> str:
    t0 = time.perf_counter()
    work_dir = Path(video_path).parent
    stem = Path(video_path).stem
    src_lang, tgt_lang = language, ("en" if translate else language)
    is_canary = "canary" in model_name.lower()
    free_before, _ = _vram_gb()

    # Audio
    trim_tag = f"trim{trim_sec}" if trim_sec else "full"
    audio_path = str(work_dir / f"{stem}_nemo_16k_{trim_tag}.wav")
    if Path(audio_path).exists():
        log.info(f"Reusing cached audio: {Path(audio_path).name}")
    else:
        log.info("Extracting 16 kHz mono WAV…")
        _extract_audio(video_path, audio_path, trim_sec)
        log.info(f"Audio extracted {time.perf_counter() - t0:.1f} s")
    audio_dur = _audio_duration(audio_path)
    log.info(f"Audio ready | duration {_fmt_dur(audio_dur)}")

    # Chunk size
    if chunk_override_sec and is_canary:
        chunk_sec = max(30, min(int(chunk_override_sec), 900))
        log.info(f"Manual chunk override: {_fmt_dur(chunk_sec)}")
    else:
        chunk_sec = _estimate_chunk_sec(model_name, safety_factor, reserve_gb)
    log.info(f"Transcribing with {_fmt_dur(chunk_sec)} chunk target…")

    # Transcribe
    t_asr = time.perf_counter()
    words, segs, manifest = _transcribe_chunked(model, audio_path, model_name,
                                                  src_lang, tgt_lang, chunk_sec)
    _cleanup_chunks(manifest, audio_path)
    asr_elapsed = time.perf_counter() - t_asr
    rtf = asr_elapsed / audio_dur if audio_dur > 0 else 0
    log.info(f"Transcription done  {asr_elapsed:.1f} s  (RTF {rtf:.2f}x)")

    if is_canary: words = []
    if not words and not segs:
        raise RuntimeError("NeMo returned no output.")
    log.info(f"Got {len(segs)} segment timestamps" if segs else f"Got {len(words)} word timestamps")

    # Build subtitles
    if diarize:
        turns = _run_diarization(audio_path, work_dir)
        final_segs = (_words_to_segs(_assign_speakers(words, turns), diarized=True) if words
                      else _split_coarse_segs(_assign_speakers(segs, turns)))
        spk_counts: dict = {}
        for seg in final_segs: spk_counts[seg.get("speaker", "?")] = spk_counts.get(seg.get("speaker", "?"), 0) + 1
        log.info(f"Built {len(final_segs)} diarized subtitle segments")
        for spk, n in sorted(spk_counts.items(), key=lambda x: -x[1]):
            log.info(f"  {spk}: {n} segments ({n/len(final_segs)*100:.0f}%)")
        srt = _segs_to_srt(final_segs, diarized=True)
    else:
        final_segs = _words_to_segs(words) if words else _split_coarse_segs(segs)
        log.info(f"Built {len(final_segs)} subtitle segments")
        srt = _segs_to_srt(final_segs)

    wall = time.perf_counter() - t0
    log.info(
        f"{'='*55}\n"
        f"  Total wall time   : {_fmt_dur(wall)}\n"
        f"  Audio duration    : {_fmt_dur(audio_dur)}\n"
        f"  ASR time          : {_fmt_dur(asr_elapsed)}\n"
        f"  Real-time factor  : {rtf:.2f}x  (< 1.0 = faster than real-time)\n"
        f"  Subtitle segments : {len(final_segs)}\n"
        f"{'='*55}"
    )
    return srt


# ── CLI ───────────────────────────────────────────────────────────────────────

def _select_model(language: str, user_model: str) -> str:
    if user_model and user_model != MODEL_EN: return user_model
    return MODEL_MULTI if language in MULTI_LANGS else MODEL_EN

def main():
    p = argparse.ArgumentParser(description="NeMo ASR local GPU transcription.")
    p.add_argument("video", nargs="?", help="Video file (auto-detect if omitted)")
    p.add_argument("--all", action="store_true", help="Process all pending videos")
    p.add_argument("--language", default="en", help="Source language: en/de/fr/es [default: en]")
    p.add_argument("--nemo-model", default=MODEL_EN, help="Override model name")
    p.add_argument("--precision", default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--translate", action="store_true", help="Translate to English (canary only)")
    p.add_argument("--diarize", action="store_true", help="Add [Speaker N] labels")
    p.add_argument("--trim", type=int, default=0, metavar="SEC", help="Trim to first N seconds")
    p.add_argument("--safety-factor", type=float, default=0.85)
    p.add_argument("--reserve-gb", type=float, default=1.5)
    p.add_argument("--chunk-override", type=int, default=None, metavar="SEC")
    args = p.parse_args()

    cwd = Path.cwd()
    if args.video:
        vp = Path(args.video)
        if not vp.is_absolute(): vp = cwd / vp
        if not vp.exists(): log.error(f"Video not found: {vp}"); return 1
        videos = [vp]
    else:
        found = [f for f in cwd.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXT]
        if not found: log.error("No video files found"); return 1
        suffix = ".nemo.en.srt" if args.translate else f".nemo.{args.language}.srt"
        pending = [v for v in found if not (v.parent / (v.stem + suffix)).exists()]
        if not pending: log.info("All videos done!"); return 0
        videos = pending if args.all else [pending[0]]
        log.info(f"Auto-detected {len(videos)} video(s)")

    if args.translate and args.language == "en":
        log.warning("--translate with --language en is a no-op. Ignoring."); args.translate = False

    model_name = MODEL_MULTI if args.translate else _select_model(args.language, args.nemo_model)
    task = "Translation" if args.translate else "Transcription"

    log.info("=" * 60)
    log.info(f"NeMo ASR {task} Pipeline (Local GPU)")
    log.info("=" * 60)
    log.info(f"Model    : {model_name}")
    log.info(f"Language : {args.language}" + (" → en" if args.translate else ""))
    log.info(f"Precision: {args.precision}  |  Diarize: {'yes' if args.diarize else 'no'}  |  Trim: {_fmt_dur(args.trim) if args.trim else 'full'}")
    log.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = _load_model(model_name, args.precision, device)
    except Exception as e:
        log.error(f"Model load failed: {e}", exc_info=True); return 1

    srt_suffix = (".nemo.en.srt" if args.translate
                  else f".nemo.{args.language}.diarize.srt" if args.diarize
                  else f".nemo.{args.language}.srt")
    ok = True
    for i, vp in enumerate(videos):
        if len(videos) > 1:
            log.info(f"\n{'='*60}\nFile {i+1}/{len(videos)}: {vp.name}\n{'='*60}")
        try:
            srt = _run_with_model(model, str(vp), args.language, model_name,
                                   args.translate, args.diarize, args.trim,
                                   args.safety_factor, args.reserve_gb, args.chunk_override)
        except Exception as e:
            ok = False; log.error(f"Failed for {vp.name}: {e}", exc_info=True); continue

        out_path = vp.parent / (vp.stem + srt_suffix)
        out_path.write_text(srt, encoding="utf-8")
        log.info(f"\n{'='*60}")
        log.info(f"✅ {task} complete!")
        log.info(f"📄 SRT saved: {out_path}")
        log.info(f"{'='*60}")

        if len(videos) == 1:
            lines = srt.split("\n")
            log.info("\n📋 Preview (first segments):\n" + "-" * 40)
            for line in lines[:16]: log.info(f"  {line}")
            seg_count = sum(1 for l in lines if l.strip().isdigit())
            if seg_count > 4: log.info(f"  ... ({seg_count} segments total)")

    return 0 if ok else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())