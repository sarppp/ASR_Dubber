'''
NeMo ASR Transcription Pipeline on Modal
=========================================

Runs NeMo ASR in a Modal cloud GPU container. You run this script locally;
it uploads your video, transcribes it on Modal, and saves the .srt next to
your video file — all in one command.

── Prerequisites ─────────────────────────────────────────────────────────────

  1. Modal account + CLI:  pip install modal && modal setup
  2. .env file in this directory with your Modal token (or use modal setup)
  3. Video file(s) in the directory you run the command from

── --language : SOURCE language, not target ──────────────────────────────────

  --language is the spoken language IN the video. You must specify it manually.
  There is NO automatic language detection (no Whisper, no audio sniffing).
  Wrong --language = garbage output.

  --translate LANG takes a target language code (e.g. fr, de, en).
  Canary model is auto-selected. Canary supports en/de/fr/es as both source and target.
  Output SRT is named after the target language: momo.nemo.fr.srt

── Video input & SRT output ──────────────────────────────────────────────────

  Run the command from the folder that contains your video file.
  "Current directory" = wherever your terminal is when you run modal run.

  Without --video-filename:
    Scans the current directory for the first video or audio file (.mp4 .mkv
    .avi .mov .webm .flv .wmv .m4v .wav .mp3 .flac .m4a .ogg) that does NOT
    already have a matching SRT file. Skips already-done files so you can re-run safely.
    WAV files skip the ffmpeg extraction step and are used directly.

  With --video-filename momo.mp4:
    Uses that specific file from the current directory.

  Output SRT is written to the same directory as the video:
    momo.mp4  ->  momo.nemo.de.srt          (transcription)
    momo.mp4  ->  momo.nemo.fr.srt          (--translate fr)
    momo.mp4  ->  momo.nemo.de.diarize.srt  (--diarize)

── Models ────────────────────────────────────────────────────────────────────

  Shortname     HuggingFace ID                  Notes
  ──────────    ──────────────────────────────  ─────────────────────────────
  parakeet-v3   nvidia/parakeet-tdt-0.6b-v3     DEFAULT. 25 EU langs + EN.
                                                Word-level timestamps. ~2 GB.
  parakeet-v2   nvidia/parakeet-tdt-0.6b-v2     English only. Slightly faster.
  canary        nvidia/canary-1b-v2             EN/DE/FR/ES only. Segment
                                                timestamps. Required for
                                                --translate. ~5 GB VRAM.
  qwen3-asr     Qwen/Qwen3-ASR-1.7B            30 langs, best quality. ~5 GB.
  qwen3-asr-s   Qwen/Qwen3-ASR-0.6B            30 langs, lighter. ~2 GB.

  Auto-selected when no --nemo-model given:
    --language en              -> parakeet-v3
    --language de/fr/es/it/... -> parakeet-v3
    --translate LANG           -> canary  (overrides everything; e.g. --translate fr)

  Supported --language codes for parakeet-v3:
    en fr de es it nl pl pt ru sv da fi cs sk sl hr ro hu bg el et lv lt uk mt

  You can also pass a full HuggingFace model ID instead of a shortname.

── GPU ───────────────────────────────────────────────────────────────────────

  GPU_TYPE is set near the top of this file (default: T4).
  Change it before running if you need more VRAM or speed.

  T4    16 GB  ~$0.000164/sec  fine for most videos with parakeet/canary
  A10G  24 GB  ~$0.000306/sec  recommended for long videos or qwen3-asr
  A100  40 GB  ~$0.000890/sec  fastest, overkill unless you batch

── Examples ──────────────────────────────────────────────────────────────────

  # Transcribe a German video (auto-detects first .mp4 in current folder)
  uv run --env-file .env modal run nemo_modal_app.py --language de

  # Transcribe a specific file
  uv run --env-file .env modal run nemo_modal_app.py --video-filename momo.mp4 --language en

  # Translate German audio to English subtitles  (--translate takes the TARGET language)
  modal run nemo_modal_app.py --video-filename talk.mp4 --language de --translate en

  # Translate English audio to French subtitles
  modal run nemo_modal_app.py --video-filename talk.mp4 --language en --translate fr

  # Add speaker labels (who said what)
  modal run nemo_modal_app.py --language en --diarize

  # Use Qwen3-ASR for best quality on a 30-language video
  modal run nemo_modal_app.py --language ru --nemo-model qwen3-asr

  # Test with only the first 5 minutes before processing the full video
  modal run nemo_modal_app.py --language de --trim 300

  # Reduce precision to save VRAM on T4
  modal run nemo_modal_app.py --language en --precision fp16

  # Force chunk size in seconds — works for all models (useful if auto-sizing OOMs)
  modal run nemo_modal_app.py --language de --chunk-override 120
  modal run nemo_modal_app.py --language en --nemo-model qwen3-asr --chunk-override 1800

  # Combine: diarize a 10-min clip in fp16
  modal run nemo_modal_app.py --video-filename talk.mp4 --language en --diarize --trim 600 --precision fp16
'''

import modal
from pathlib import Path
import os
import sys
import gc
import time
import wave
import subprocess

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REMOTE_IO_PATH = Path("/app/nemo_asr")
NEMO_CONDA_ENV = "nemo-env"
CONDA_PYTHON = f"/opt/conda/envs/{NEMO_CONDA_ENV}/bin/python"
GPU_TYPE = "A10G"  # Change to "A10G" for long videos

# Friendly shortname → full NeMo model ID (mirrors nemo_audio.ASR_MODELS)
ASR_MODELS = {
    "parakeet-v2":  "nvidia/parakeet-tdt-0.6b-v2",   # English only, fastest
    "parakeet-v3":  "nvidia/parakeet-tdt-0.6b-v3",   # 25 EU langs, same speed
    "canary":       "nvidia/canary-1b-v2",            # EN/DE/FR/ES + translation
    "qwen3-asr":    "Qwen/Qwen3-ASR-1.7B",           # 30 langs, best quality
    "qwen3-asr-s":  "Qwen/Qwen3-ASR-0.6B",           # 30 langs, faster/lighter
}

MODEL_EN    = ASR_MODELS["parakeet-v3"]   # v3 supports EN + 25 EU langs
MODEL_MULTI = ASR_MODELS["parakeet-v3"]   # same model for all non-EN langs
MODEL_CANARY = ASR_MODELS["canary"]       # only used when --translate

# Languages auto-routed to MODEL_MULTI (parakeet-v3 supported languages)
MULTI_LANGS = {
    "fr", "de", "es", "it", "nl", "pl", "pt", "ru", "sv", "da",
    "fi", "cs", "sk", "sl", "hr", "ro", "hu", "bg", "el", "et",
    "lv", "lt", "uk", "mt",
}  # English ("en") excluded — uses MODEL_EN

DEFAULT_NEMO_MODEL = MODEL_EN

CHUNK_OVERLAP_SEC = 2


def select_nemo_model(language: str, nemo_model: str | None = None) -> str:
    """Resolve the final model ID.

    Priority: --nemo-model shortname/ID > auto (language-based).
    """
    if nemo_model:
        return ASR_MODELS.get(nemo_model, nemo_model)
    return MODEL_MULTI if language in MULTI_LANGS else MODEL_EN


# ---------------------------------------------------------------------------
# Modal image (same proven conda setup as original)
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("ffmpeg", "git", "curl", "ca-certificates", "bash")
    .env({
        "PATH": "/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONPATH": "/opt/conda/envs/nemo-env/lib/python3.12/site-packages",
    })
    .run_commands(
        "curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py312_24.7.1-0-Linux-x86_64.sh -o /tmp/miniconda.sh",
        "bash /tmp/miniconda.sh -b -p /opt/conda",
        "rm -f /tmp/miniconda.sh",
    )
    .run_commands(
        "bash -lc '/opt/conda/bin/conda create -n nemo-env python=3.12 -y'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate nemo-env && pip install -U pip'",
        'bash -lc \'source /opt/conda/etc/profile.d/conda.sh && conda activate nemo-env && pip install "numpy<2.0"\'',
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate nemo-env && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate nemo-env && pip install soundfile librosa nemo_toolkit[asr]'",
        "bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate nemo-env && pip install qwen-asr'",
    )
    .run_commands(
        "bash -lc 'ln -sf /opt/conda/envs/nemo-env/bin/python /usr/local/bin/python'",
        "bash -lc 'ln -sf /opt/conda/envs/nemo-env/bin/pip /usr/local/bin/pip'",
    )
)

app = modal.App(name="nemo-asr-transcriber-v2")

# ---------------------------------------------------------------------------
# Lazy imports for torch / NeMo (so local client doesn't need them)
# ---------------------------------------------------------------------------

torch = None
_torch_mp = None
nemo_asr = None
OmegaConf = None
ClusteringDiarizer = None


NEMO_ENV_SITE_PACKAGES = "/opt/conda/envs/nemo-env/lib/python3.12/site-packages"


def _ensure_remote_imports():
    global torch, _torch_mp, nemo_asr, OmegaConf, ClusteringDiarizer
    if torch is not None:
        return

    if NEMO_ENV_SITE_PACKAGES not in sys.path:
        sys.path.insert(0, NEMO_ENV_SITE_PACKAGES)

    import torch as _torch
    import torch.multiprocessing as _mp
    import nemo.collections.asr as _nemo_asr
    from omegaconf import OmegaConf as _OmegaConf
    from nemo.collections.asr.models import ClusteringDiarizer as _ClusteringDiarizer

    torch = _torch
    _torch_mp = _mp
    nemo_asr = _nemo_asr
    OmegaConf = _OmegaConf
    ClusteringDiarizer = _ClusteringDiarizer

    # PyTorch 2.6 weights_only fix (best-effort)
    try:
        _original_load = torch.load

        def _safe_load_wrapper(*args, **kwargs):
            kwargs["weights_only"] = False
            return _original_load(*args, **kwargs)

        torch.load = _safe_load_wrapper
    except Exception:
        pass

    # Avoid /dev/shm bus errors in containers
    try:
        _torch_mp.set_sharing_strategy("file_system")
    except Exception:
        pass

    # CUDA memory optimizations
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")


# ---------------------------------------------------------------------------
# Helpers — timestamp / SRT formatting
# ---------------------------------------------------------------------------

def _fmt_ts(s: float) -> str:
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"


def _fmt_dur(s: float) -> str:
    return f"{int(s // 60)}m{int(s % 60):02d}s" if s >= 60 else f"{s:.1f}s"


# ---------------------------------------------------------------------------
# Segment builders
# ---------------------------------------------------------------------------

def _words_to_segs(words, max_w=10, max_dur=5.0, max_ch=80, diarized=False):
    """Unified segment builder. Fixed: boundary word is no longer dropped."""
    segs, cur_w, cur_t, cur_s, cur_spk = [], [], "", None, None
    for w in words:
        word = w.get("word", "").strip()
        if not word:
            continue
        ws, we = w.get("start", 0.0), w.get("end", 0.0)
        spk = w.get("speaker", "unknown") if diarized else None
        if cur_s is None:
            cur_s, cur_spk = ws, spk
        cand = (cur_t + " " + word).strip() if cur_t else word
        split = (
            len(cur_w) >= max_w
            or (we - cur_s) > max_dur
            or len(cand) > max_ch
            or (cur_t and cur_t[-1] in ".!?" and len(cur_w) >= 3)
            or (diarized and spk != cur_spk and cur_w)
        )
        if split and cur_w:
            seg = {"start": cur_s, "end": cur_w[-1].get("end", cur_s), "text": cur_t}
            if diarized:
                seg["speaker"] = cur_spk
            segs.append(seg)
            cur_w, cur_t, cur_s, cur_spk = [], "", ws, spk
            cand = word  # boundary word becomes start of next segment
        cur_w.append(w)
        cur_t = cand
    if cur_w and cur_t.strip():
        seg = {"start": cur_s, "end": cur_w[-1].get("end", cur_s), "text": cur_t}
        if diarized:
            seg["speaker"] = cur_spk
        segs.append(seg)
    return segs


def _words_to_segments(words, max_words=10, max_dur=5.0, max_chars=80):
    return _words_to_segs(words, max_w=max_words, max_dur=max_dur, max_ch=max_chars, diarized=False)


def _words_to_segments_diarized(words, max_words=10, max_dur=5.0, max_chars=80):
    return _words_to_segs(words, max_w=max_words, max_dur=max_dur, max_ch=max_chars, diarized=True)


def _strip_asr_repetition(text: str, min_unit_words: int = 5, min_reps: int = 3) -> str:
    """Remove hallucinated repetition loops from Canary/Whisper ASR output."""
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
                return " ".join(words[:start + unit_len])
    return text


def _segs_to_srt(segs, diarized=False):
    if diarized:
        spk_list = sorted({s.get("speaker", "unknown") for s in segs})
        spk_map = {s: f"Speaker {i + 1}" for i, s in enumerate(spk_list)}
    lines, idx, prev_text, prev_spk = [], 0, None, None
    for s in segs:
        t = s["text"].strip()
        if not t:
            continue
        spk = s.get("speaker") if diarized else None
        # Same speaker repeating same text = hallucination; different speakers = valid dialogue
        if t == prev_text and (not diarized or spk == prev_spk):
            continue
        idx += 1
        if diarized:
            label = spk_map.get(s.get("speaker", "unknown"), "Speaker ?")
            lines += [str(idx), f"{_fmt_ts(s['start'])} --> {_fmt_ts(s['end'])}", f"[{label}] {t}", ""]
        else:
            lines += [str(idx), f"{_fmt_ts(s['start'])} --> {_fmt_ts(s['end'])}", t, ""]
        prev_text = t
        prev_spk = spk
    return "\n".join(lines)


def _segs_to_srt_diarized(segs):
    return _segs_to_srt(segs, diarized=True)


# ---------------------------------------------------------------------------
# Token merging (from nemo3.py — handles canary BPE / sentencepiece tokens
# and parakeet cumulative-text output)
# ---------------------------------------------------------------------------

def _merge_canary_tokens(words: list) -> list:
    if not words:
        return words
    has_spm = any("▁" in (w.get("word") or "") for w in words)
    has_g = any((w.get("word") or "").startswith("Ġ") for w in words)
    if not has_spm and not has_g:
        return words
    merged, cur = [], None
    for w in words:
        tok = str(w.get("word") or "")
        if not tok:
            continue
        starts_new = tok.startswith("▁") or tok.startswith("Ġ")
        piece = tok.lstrip("▁")
        if tok.startswith("Ġ"):
            piece = tok[1:]
        if starts_new or cur is None:
            if cur is not None and cur.get("word"):
                merged.append(cur)
            cur = {
                "word": piece,
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", w.get("start", 0.0))),
            }
            if "speaker" in w:
                cur["speaker"] = w["speaker"]
        else:
            cur["word"] = cur.get("word", "") + piece
            cur["end"] = float(w.get("end", cur.get("end", 0.0)))
    if cur is not None and cur.get("word"):
        merged.append(cur)
    return merged


def _merge_cumulative_words(words: list) -> list:
    """Handle parakeet's cumulative text output (each token = sentence so far)."""
    if not words or len(words) < 2:
        return words
    merged, prev_text = [], ""
    for w in words:
        current_text = w.get("word", "").strip()
        if not current_text:
            continue
        if prev_text and current_text.startswith(prev_text):
            delta = current_text[len(prev_text):].strip()
            if delta:
                new_word = {"word": delta, "start": w.get("start", 0.0), "end": w.get("end", 0.0)}
                if "speaker" in w:
                    new_word["speaker"] = w["speaker"]
                merged.append(new_word)
        elif not prev_text:
            merged.append(w.copy())
        else:
            merged.append(w.copy())
        prev_text = current_text
    return merged


def _dedup_words(words: list) -> list:
    if not words:
        return words
    out = [words[0]]
    for w in words[1:]:
        p = out[-1]
        if w["start"] < p["end"] - 0.05 and w["word"] == p["word"]:
            continue
        if w["start"] < p["start"]:
            continue
        out.append(w)
    return out


# ---------------------------------------------------------------------------
# Special token stripping
# ---------------------------------------------------------------------------

def _strip_special_tokens(text: str) -> str:
    import re
    text = re.sub(r"(<\|endoftext\|>[\s.]*)+$", "", text)
    text = re.sub(r"<\|[^|>]+\|>", "", text)
    text = re.sub(r"\.{4,}", "...", text)
    if text.strip() == "...":
        text = ""
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Qwen3-ASR support (separate from NeMo — uses qwen-asr package)
# ---------------------------------------------------------------------------

QWEN3_LANG_MAP = {
    "en": "English",    "de": "German",     "fr": "French",
    "es": "Spanish",    "it": "Italian",    "nl": "Dutch",
    "pt": "Portuguese", "ru": "Russian",    "zh": "Chinese",
    "ja": "Japanese",   "ko": "Korean",     "ar": "Arabic",
    "tr": "Turkish",    "hi": "Hindi",      "vi": "Vietnamese",
    "th": "Thai",       "pl": "Polish",     "cs": "Czech",
    "sv": "Swedish",    "da": "Danish",     "fi": "Finnish",
    "el": "Greek",      "hu": "Hungarian",  "ro": "Romanian",
    "uk": "Ukrainian",  "id": "Indonesian", "ms": "Malay",
    "fa": "Persian",    "fil": "Filipino",  "mk": "Macedonian",
}


def _is_qwen3_asr(model_name: str) -> bool:
    return "Qwen3-ASR" in model_name or "qwen3-asr" in model_name.lower()


def _load_qwen3_asr(model_name: str, device: str, precision: str):
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError:
        raise RuntimeError("qwen-asr package not installed in the Modal image.")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(precision, torch.bfloat16)
    device_map = "cuda:0" if device == "cuda" else "cpu"
    print(f"Loading Qwen3-ASR: {model_name} [{dtype}] on {device_map}")
    model = Qwen3ASRModel.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs=dict(dtype=dtype, device_map=device_map),
    )
    print("Qwen3-ASR model ready")
    return model


def _transcribe_qwen3_asr(model, audio_path: str, offset: float, src_lang: str) -> tuple:
    lang_name = QWEN3_LANG_MAP.get(src_lang)  # None = auto-detect
    try:
        results = model.transcribe(audio=audio_path, language=lang_name, return_time_stamps=True)
    except Exception as exc:
        print(f"Qwen3-ASR transcribe failed: {exc}"); raise

    if not results:
        return [], []

    result = results[0]
    text = _strip_special_tokens(getattr(result, "text", "") or "")

    all_words = []
    for ts in (getattr(result, "time_stamps", None) or []):
        word = _strip_special_tokens(getattr(ts, "text", "") or "").strip()
        if not word:
            continue
        all_words.append({
            "word": word,
            "start": float(getattr(ts, "start_time", 0.0)) + offset,
            "end":   float(getattr(ts, "end_time",   0.0)) + offset,
        })

    if not all_words and text:
        dur = _audio_duration(audio_path)
        seg = {
            "text": text,
            "start": offset,
            "end": offset + (dur if dur > 0 else max(1.0, len(text.split()) * 0.4)),
        }
        return [], [seg]

    return all_words, []


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def _audio_duration(path: str) -> float:
    import wave
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def _extract_audio(video_path: str, audio_path: str, trim_sec: int = 0) -> None:
    cpu_count = os.cpu_count() or 4
    cmd = ["ffmpeg", "-y", "-threads", str(cpu_count), "-i", video_path]
    if trim_sec > 0:
        cmd += ["-t", str(trim_sec)]
    cmd += ["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-threads", str(cpu_count), audio_path]
    subprocess.run(cmd, check=True, capture_output=True)


def _chunk_audio(audio_path: str, work_dir: Path, chunk_sec: int, overlap_sec: int = CHUNK_OVERLAP_SEC) -> list:
    cpu_count = os.cpu_count() or 4
    duration = _audio_duration(audio_path)
    if duration <= chunk_sec + 5:
        return [(audio_path, 0.0)]
    chunks, step, offset, idx = [], chunk_sec - overlap_sec, 0.0, 0
    while offset < duration:
        cp = str(work_dir / f"_chunk_{idx:04d}.wav")
        dur = min(chunk_sec, duration - offset)
        subprocess.run(
            ["ffmpeg", "-y", "-threads", str(cpu_count), "-ss", str(offset), "-i", audio_path,
             "-t", str(dur), "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", cp],
            check=True, capture_output=True,
        )
        chunks.append((cp, offset))
        offset += step
        idx += 1
    return chunks


def _cleanup_chunks(manifest: list, original_audio: str) -> None:
    for entry in manifest or []:
        path = entry.get("path")
        if not path or path == original_audio:
            continue
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------

def _vram_gb() -> tuple:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free_b, total_b = torch.cuda.mem_get_info()
    return free_b / 1024 ** 3, total_b / 1024 ** 3


def _compute_max_chunk_sec(model_name: str, safety_factor: float = 0.85, reserve_gb: float = 1.5) -> int:
    free_gb, _ = _vram_gb()
    if free_gb <= 0:
        return 300
    usable_gb = max(0.0, free_gb - reserve_gb) * safety_factor
    if usable_gb <= 0:
        return 60
    if "canary" in model_name.lower():
        # QUALITY cap — Canary was trained on ≤40s segments; decoder attention
        # tracking collapses above ~60s regardless of VRAM. Not a memory limit.
        return 60
    elif _is_qwen3_asr(model_name):
        # LLM-based ASR: VRAM scales with output token KV cache, not raw audio
        # length. Offline inference handles long context; no quality cap needed.
        gb_per_minute = 0.35
    elif "parakeet" in model_name.lower():
        # CTC/TDT: quality unaffected by chunk length; VRAM scales linearly.
        gb_per_minute = 0.28
    else:
        gb_per_minute = 0.50
    max_minutes = usable_gb / gb_per_minute
    secs = int(max_minutes * 60)
    # No model-specific cap — free VRAM is the only constraint.
    # 7200s (2h) absolute ceiling: a single chunk longer than 2h would never
    # be triggered in practice (calibration runs first for long audio).
    return max(30, min(secs, 7200))


def _calibrate_chunk_size(
    model, audio_path: str, model_name: str, language: str, target_lang: str,
    initial_guess_sec: int, reserve_gb: float, safety_factor: float,
) -> int:
    if not torch.cuda.is_available():
        return initial_guess_sec

    audio_dur = _audio_duration(audio_path)
    test_sec = max(60, min(int(audio_dur * 0.1), int(audio_dur)) if audio_dur > 0 else initial_guess_sec)

    work_dir = Path(audio_path).parent
    test_chunk = work_dir / "_calibration_chunk.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", "0", "-i", audio_path, "-t", str(test_sec),
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(test_chunk)],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError:
        return initial_guess_sec

    try:
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated() if hasattr(torch.cuda, "memory_allocated") else 0
        _transcribe_manifest(model, [{"path": str(test_chunk), "offset": 0.0}], model_name, language, target_lang)
        peak_bytes = torch.cuda.max_memory_allocated() if hasattr(torch.cuda, "max_memory_allocated") else baseline
        delta_bytes = max(0, peak_bytes - baseline)
        vram_used_gb = delta_bytes / 1024 ** 3 if delta_bytes else 0.0
        gb_per_sec = vram_used_gb / test_sec if test_sec > 0 else 0.0
        if gb_per_sec < 0.001:
            return initial_guess_sec
        free_now, _ = _vram_gb()
        usable_gb = max(0.0, free_now - reserve_gb) * safety_factor
        projected_sec = int(usable_gb / gb_per_sec)
        # Cap at audio duration: no benefit in a chunk larger than the full audio.
        # Do NOT cap at initial_guess_sec — on a powerful GPU, calibration should
        # be free to project upward (e.g. 13 GB free → single-pass 30-min audio).
        return max(60, min(projected_sec, int(audio_dur) + 60))
    finally:
        test_chunk.unlink(missing_ok=True)
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Model loading (layer-by-layer, from nemo3.py)
# ---------------------------------------------------------------------------

def _load_model(model_name: str, precision: str, device: str):
    if _is_qwen3_asr(model_name):
        return _load_qwen3_asr(model_name, device, precision)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        gc.collect()
        free_before, total_gb = _vram_gb()
        print(f"VRAM before load: {free_before:.2f}/{total_gb:.2f} GB free")
        min_vram_gb = 4.0 if precision == "fp32" else 2.5
        if free_before < min_vram_gb:
            raise RuntimeError(f"Insufficient VRAM: {free_before:.2f} GB free (need {min_vram_gb:.1f} GB)")

    t0 = time.perf_counter()
    print(f"Loading model: {model_name}…")

    map_loc = "cpu" if device == "cpu" else None
    try:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, map_location=map_loc)
    except Exception as e:
        if device == "cuda":
            print(f"GPU load failed ({e}); retrying on CPU then moving to GPU")
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, map_location="cpu")
        else:
            raise

    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
        target_dtype = torch.float32
        if precision == "bf16" and torch.cuda.is_bf16_supported():
            target_dtype = torch.bfloat16
            print("Moving to GPU with bfloat16 (layer-by-layer)…")
        elif precision == "fp16":
            target_dtype = torch.float16
            print("Moving to GPU with float16 (layer-by-layer)…")
        elif precision == "bf16":
            print("bfloat16 unsupported on this GPU — using fp32")
        else:
            print("Moving to GPU with float32…")

        try:
            for name, module in model.named_children():
                module.to(target_dtype).to(device)
                torch.cuda.empty_cache()
            model = model.to(device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError(f"OOM during model transfer. Try --precision fp16.") from e
            raise
    else:
        print("Precision: float32 (CPU)")

    model.eval()
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    load_sec = time.perf_counter() - t0

    if device == "cuda":
        free_after, _ = _vram_gb()
        used_gb = free_before - free_after
        print(f"Model loaded {load_sec:.1f}s | model VRAM {used_gb:.2f} GB | free VRAM {free_after:.2f} GB")

        free_gb, _ = _vram_gb()
        if free_gb > 1.0:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print("torch.compile(reduce-overhead) active — first chunk warms up")
            except Exception:
                pass
        else:
            print(f"Skipping torch.compile (only {free_gb:.2f} GB VRAM free)")
    else:
        print(f"Model loaded {load_sec:.1f}s (CPU mode)")

    return model


# ---------------------------------------------------------------------------
# Core transcription (from nemo3.py — CUDA graph fix, token merging, etc.)
# ---------------------------------------------------------------------------

def _transcribe_manifest(model, manifest: list, model_name: str, language: str, target_lang: str) -> tuple:
    # Disable CUDA graphs (avoids invalid getCurrentStream errors on some GPUs)
    if hasattr(model, "cfg") and hasattr(model.cfg, "decoding"):
        for attr in ("use_cuda_graphs", "cuda_graphs", "use_cuda_graph"):
            if attr in model.cfg.decoding:
                model.cfg.decoding[attr] = False
    if hasattr(model, "decoding"):
        for attr in ("use_cuda_graphs", "cuda_graphs", "use_cuda_graph"):
            if hasattr(model.decoding, attr):
                setattr(model.decoding, attr, False)
        dc = getattr(model.decoding, "decoding_computer", None)
        if dc:
            for attr in ("use_cuda_graphs", "cuda_graphs", "use_cuda_graph"):
                if hasattr(dc, attr):
                    setattr(dc, attr, False)
            if hasattr(dc, "cuda_graphs_impl"):
                dc.cuda_graphs_impl = None

    all_words, all_segs, text_parts = [], [], []
    n = len(manifest)

    is_qwen3 = _is_qwen3_asr(model_name)

    for ci, entry in enumerate(manifest):
        path, offset = entry["path"], entry["offset"]

        if is_qwen3:
            words, segs = _transcribe_qwen3_asr(model, path, offset, language)
            all_words.extend(words)
            all_segs.extend(segs)
            if n > 1:
                free_gb, _ = _vram_gb()
                print(f"  chunk {ci + 1}/{n} | VRAM free {free_gb:.2f} GB")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            continue

        prev_cum = ""
        with torch.inference_mode():
            kw = {"timestamps": True, "batch_size": 1}
            if "canary" in model_name.lower():
                kw["source_lang"] = language
                kw["target_lang"] = target_lang
            out = model.transcribe([path], **kw)

        if n > 1:
            free_gb, _ = _vram_gb()
            print(f"  chunk {ci + 1}/{n} | VRAM free {free_gb:.2f} GB")

        if hasattr(out, "__len__") and len(out) > 0:
            hyp = out[0]
            text = hyp.text if hasattr(hyp, "text") else (hyp if isinstance(hyp, str) else "")
            text_parts.append(text or "")
            ts = getattr(hyp, "timestamp", None)
            words_str = getattr(hyp, "words", None) or []

            if ts and isinstance(ts, dict) and "word" in ts:
                word_items = ts.get("word") or []
                for i, td in enumerate(word_items):
                    if not isinstance(td, dict):
                        continue
                    s = td.get("start", 0.0)
                    e = td.get("end", 0.0)
                    w = td.get("word", None)
                    if not w and i < len(words_str):
                        w = words_str[i]
                    w = str(w or "").strip()
                    if ("canary" in model_name.lower() or "parakeet" in model_name.lower()) and w:
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

            if ts and isinstance(ts, dict) and "segment" in ts:
                for seg in ts["segment"]:
                    all_segs.append({
                        "text": str(seg.get("segment", "")),
                        "start": float(seg.get("start", 0.0)) + offset,
                        "end": float(seg.get("end", 0.0)) + offset,
                    })

            if not all_words and not all_segs and ts and isinstance(ts, dict):
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    if "canary" in model_name.lower() and all_words:
        all_words = _merge_canary_tokens(all_words)
    elif "parakeet" in model_name.lower() and all_words:
        all_words = _merge_cumulative_words(all_words)

    # Strip hallucination loops from Canary segment text
    if "canary" in model_name.lower() and all_segs:
        for seg in all_segs:
            seg["text"] = _strip_asr_repetition(seg["text"])

    return all_words, all_segs, text_parts


# ---------------------------------------------------------------------------
# OOM-retry transcription with adaptive chunk halving (from nemo3.py)
# ---------------------------------------------------------------------------

def _transcribe_with_retry(model, audio_path: str, offset: float, model_name: str,
                            language: str, target_lang: str, initial_chunk_sec: int):
    chunk_sec = max(30, initial_chunk_sec)
    work_dir = Path(audio_path).parent
    duration = _audio_duration(audio_path)

    while chunk_sec >= 30:
        manifest = []
        try:
            if duration <= chunk_sec:
                manifest = [{"path": audio_path, "offset": offset}]
                print(f"Single-pass — full {_fmt_dur(duration)}")
            else:
                raw = _chunk_audio(audio_path, work_dir, chunk_sec=chunk_sec)
                manifest = [{"path": cp, "offset": off} for cp, off in raw]
                print(f"Chunk size {_fmt_dur(chunk_sec)} → {len(manifest)} chunk(s)")
            words, segs, txt = _transcribe_manifest(model, manifest, model_name, language, target_lang)
            return words, segs, txt, manifest
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            old, chunk_sec = chunk_sec, max(30, chunk_sec // 2)
            print(f"OOM at chunk={_fmt_dur(old)} → retrying with {_fmt_dur(chunk_sec)}")
            _cleanup_chunks(manifest, audio_path)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
                old, chunk_sec = chunk_sec, max(30, chunk_sec // 2)
                print(f"OOM (RuntimeError) → retrying with {_fmt_dur(chunk_sec)}")
                _cleanup_chunks(manifest, audio_path)
            else:
                _cleanup_chunks(manifest, audio_path)
                raise

    raise RuntimeError("Could not fit even 30s chunks in VRAM.")


# ---------------------------------------------------------------------------
# Diarization (inline NeMo ClusteringDiarizer, no subprocess)
# ---------------------------------------------------------------------------

def _run_diarization(audio_path: str, work_dir: Path) -> list:
    import json, shutil
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running speaker diarization…")

    ddir = work_dir / "_diarize"
    ddir.mkdir(parents=True, exist_ok=True)

    try:
        # NeMo uses the WAV stem as the key for internal files (VAD, RTTM, etc.)
        # Spaces/apostrophes in the stem cause silent RTTM lookup failures.
        # Always copy to a clean fixed name.
        safe_wav = ddir / "input_16k_mono.wav"
        shutil.copy2(audio_path, safe_wav)
        print(f"Copied WAV to safe path: {safe_wav.name}")

        mpath = ddir / "manifest.json"
        mpath.write_text(
            json.dumps({
                "audio_filepath": str(safe_wav.resolve()),
                "offset": 0, "duration": None, "label": "infer",
                "text": "", "num_speakers": None, "rttm_filepath": "", "uem_filepath": "",
            }) + "\n",
            encoding="utf-8",
        )

        # Scale batch_size for long audio to avoid STFT VRAM spikes
        audio_dur = _audio_duration(audio_path)
        if audio_dur > 600:
            batch_size = 1
        elif audio_dur > 300:
            batch_size = 4
        else:
            batch_size = 16
        print(f"Audio {_fmt_dur(audio_dur)} → diarization batch_size={batch_size}")

        cfg = OmegaConf.create({
            "name": "ClusterDiarizer",
            "num_workers": 0, "sample_rate": 16000, "batch_size": batch_size,
            "device": device, "verbose": True,
            "diarizer": {
                "manifest_filepath": str(mpath),
                "out_dir": str(ddir),
                "oracle_vad": False, "collar": 0.25, "ignore_overlap": True,
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "parameters": {
                        "window_length_in_sec": 0.63, "shift_length_in_sec": 0.01,
                        "smoothing": False, "overlap": 0.5, "onset": 0.9, "offset": 0.5,
                        "pad_onset": 0.0, "pad_offset": 0.0, "min_duration_on": 0.0,
                        "min_duration_off": 0.6, "filter_speech_first": True,
                    },
                },
                "speaker_embeddings": {
                    "model_path": "titanet_large",
                    "parameters": {
                        "window_length_in_sec": [1.5, 1.0, 0.5],
                        "shift_length_in_sec": [0.75, 0.5, 0.25],
                        "multiscale_weights": [1, 1, 1], "save_embeddings": False,
                    },
                },
                "clustering": {
                    "parameters": {
                        "oracle_num_speakers": False, "max_num_speakers": 8,
                        "enhanced_count_thres": 80, "max_rp_threshold": 0.25,
                        "sparse_search_volume": 30, "maj_vote_spk_count": False,
                        "chunk_cluster_count": 50, "embeddings_per_chunk": 10000,
                    },
                },
            },
        })

        try:
            ClusteringDiarizer(cfg=cfg).to(device).diarize()
        except (RuntimeError, Exception) as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
                raise RuntimeError(
                    f"Diarization OOM on {_fmt_dur(audio_dur)} audio — "
                    "VAD+TitaNet exceeded VRAM. Try a shorter clip or use A10G GPU."
                ) from e
            raise

        rttm_files = list((ddir / "pred_rttms").glob("*.rttm"))
        if not rttm_files:
            rttm_files = list(ddir.rglob("*.rttm"))

        turns = []
        if rttm_files:
            with open(rttm_files[0], "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8 and parts[0].upper() == "SPEAKER":
                        try:
                            start = float(parts[3])
                            dur = float(parts[4])
                            spk = str(parts[7])
                            turns.append({"speaker": spk, "start": start, "end": start + dur})
                        except (ValueError, IndexError) as e:
                            print(f"Skipping malformed RTTM line: {line.strip()} | {e}")

        turns.sort(key=lambda t: t["start"])
        print(f"Diarization done — {len({t['speaker'] for t in turns})} speaker(s), {len(turns)} turns")
        return turns

    finally:
        shutil.rmtree(ddir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Speaker assignment (words and segments)
# ---------------------------------------------------------------------------

def _assign_speakers(words: list, turns: list) -> list:
    for w in words:
        mid = (w["start"] + w["end"]) / 2.0
        best_spk, best_ov = None, 0.0
        for t in turns:
            ov = max(0.0, min(w["end"], t["end"]) - max(w["start"], t["start"]))
            if ov > best_ov:
                best_ov, best_spk = ov, t["speaker"]
            if best_spk is None and t["start"] <= mid <= t["end"]:
                best_spk = t["speaker"]
        w["speaker"] = best_spk or "unknown"
    return words


def _assign_speakers_to_segments(segments: list, turns: list) -> list:
    for seg in segments:
        seg_start, seg_end = seg.get("start", 0.0), seg.get("end", 0.0)
        best_spk, best_score = "unknown", 0.0
        for turn in turns:
            overlap = max(0.0, min(seg_end, turn["end"]) - max(seg_start, turn["start"]))
            seg_dur = seg_end - seg_start
            if seg_dur > 0:
                pct = overlap / seg_dur
                if pct > best_score:
                    best_score, best_spk = pct, turn["speaker"]
        seg["speaker"] = best_spk
    return segments


# ---------------------------------------------------------------------------
# Main pipeline (runs inside Modal container)
# ---------------------------------------------------------------------------

def _run_pipeline(
    video_path: str,
    language: str,
    nemo_model: str,
    precision: str,
    target_lang: str,
    diarize: bool,
    trim_sec: int,
    safety_factor: float,
    reserve_gb: float,
    chunk_override_sec: int | None,
) -> str:
    _ensure_remote_imports()
    wall_t0 = time.perf_counter()
    work_dir = Path(video_path).parent
    stem = Path(video_path).stem
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trim_tag = f"trim{int(trim_sec)}" if trim_sec and trim_sec > 0 else "full"
    is_wav_input = Path(video_path).suffix.lower() == ".wav"

    if is_wav_input and not (trim_sec and trim_sec > 0):
        # Already a WAV — use directly, skip ffmpeg
        audio_path = video_path
        print("Input is WAV — skipping audio extraction.")
    else:
        audio_path = str(work_dir / f"{stem}_nemo_16k_{trim_tag}.wav")
        print("Extracting 16 kHz mono WAV…")
        _extract_audio(video_path, audio_path, trim_sec=trim_sec)

    audio_dur = _audio_duration(audio_path)
    print(f"Audio ready | duration {_fmt_dur(audio_dur)}")

    print(f"Loading model: {nemo_model} [{device}]")
    model = _load_model(nemo_model, precision, device)

    # Determine chunk size
    forced_chunk = None
    if chunk_override_sec and chunk_override_sec >= 30:
        forced_chunk = max(30, int(chunk_override_sec))
        print(f"Manual chunk override: {_fmt_dur(forced_chunk)}")

    if forced_chunk:
        optimal_chunk = forced_chunk
    elif device == "cuda":
        initial_chunk = _compute_max_chunk_sec(nemo_model, safety_factor, reserve_gb)
        print(f"VRAM-estimated chunk: {_fmt_dur(initial_chunk)}")
        if audio_dur > initial_chunk * 1.5:
            try:
                optimal_chunk = _calibrate_chunk_size(
                    model, audio_path, nemo_model, language, target_lang,
                    initial_chunk, reserve_gb, safety_factor,
                )
                print(f"Calibrated chunk: {_fmt_dur(optimal_chunk)}")
            except Exception as exc:
                print(f"Calibration failed ({exc}); using {_fmt_dur(initial_chunk)}")
                optimal_chunk = initial_chunk
        else:
            optimal_chunk = initial_chunk
    else:
        optimal_chunk = 300

    print(f"Transcribing | chunk target: {_fmt_dur(optimal_chunk)}")
    t_asr = time.perf_counter()
    words, segs, _, manifest = _transcribe_with_retry(
        model, audio_path, 0.0, nemo_model, language, target_lang, optimal_chunk,
    )
    _cleanup_chunks(manifest, audio_path)
    asr_elapsed = time.perf_counter() - t_asr
    rtf = asr_elapsed / audio_dur if audio_dur > 0 else 0
    print(f"Transcription done {asr_elapsed:.1f}s (RTF {rtf:.2f}x)")

    if len(manifest) > 1 and words:
        words = _dedup_words(words)
    if "canary" in nemo_model.lower() and segs:
        words = []
    if not words and not segs:
        raise RuntimeError("NeMo returned no timestamps.")

    print(f"Got {len(words) if words else len(segs)} {'word' if words else 'segment'} timestamps")

    if diarize:
        turns = _run_diarization(audio_path, work_dir)
        if words:
            words = _assign_speakers(words, turns)
            final_segs = _words_to_segments_diarized(words)
        elif segs:
            final_segs = _assign_speakers_to_segments(segs, turns)
        else:
            final_segs = []
        srt = _segs_to_srt_diarized(final_segs)
    else:
        final_segs = _words_to_segments(words) if words else segs
        srt = _segs_to_srt(final_segs)

    wall_elapsed = time.perf_counter() - wall_t0
    print(
        f"{'=' * 55}\n"
        f"  Total wall time   : {_fmt_dur(wall_elapsed)}\n"
        f"  Audio duration    : {_fmt_dur(audio_dur)}\n"
        f"  ASR time          : {_fmt_dur(asr_elapsed)}\n"
        f"  Real-time factor  : {rtf:.2f}x\n"
        f"  Subtitle segments : {len(final_segs)}\n"
        f"{'=' * 55}"
    )

    # Cleanup extracted audio (but never delete the original input if it was a WAV)
    if audio_path != video_path:
        try:
            Path(audio_path).unlink(missing_ok=True)
        except Exception:
            pass

    return srt


# ---------------------------------------------------------------------------
# Modal remote function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=3600,
)
def transcribe_nemo_remote(
    video_filename: str,
    video_data: bytes,
    language: str = "en",
    trim_sec: int = 0,
    nemo_model: str = DEFAULT_NEMO_MODEL,
    precision: str = "bf16",
    target_lang: str | None = None,
    diarize: bool = False,
    safety_factor: float = 0.85,
    reserve_gb: float = 1.5,
    chunk_override_sec: int | None = None,
) -> bytes:
    _ensure_remote_imports()
    try:
        gpu_name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True,
        ).strip().splitlines()
        if gpu_name:
            print(f"GPU: {gpu_name[0]}")
    except Exception:
        pass

    REMOTE_IO_PATH.mkdir(parents=True, exist_ok=True)
    video_path = REMOTE_IO_PATH / video_filename

    with open(video_path, "wb") as f:
        f.write(video_data)

    srt_content = _run_pipeline(
        video_path=str(video_path),
        language=language,
        nemo_model=nemo_model,
        precision=precision,
        target_lang=target_lang or language,
        diarize=diarize,
        trim_sec=trim_sec,
        safety_factor=safety_factor,
        reserve_gb=reserve_gb,
        chunk_override_sec=chunk_override_sec,
    )

    if video_path.exists():
        video_path.unlink()

    return srt_content.encode("utf-8")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    video_filename: str = None,
    language: str = "en",
    trim: int = 0,
    nemo_model: str = DEFAULT_NEMO_MODEL,
    precision: str = "bf16",
    translate: str = None,
    diarize: bool = False,
    safety_factor: float = 0.85,
    reserve_gb: float = 1.5,
    chunk_override: int = None,
):
    """
    Transcribe or translate video using NeMo ASR (local-parity engine).

    --translate LANG   target language code (e.g. fr, de, en). Requires canary model.
                       Canary supports: en, de, fr, es (both source and target).

    Examples:
        modal run nemo_modal_app.py --language en --diarize
        modal run nemo_modal_app.py --language de --translate en
        modal run nemo_modal_app.py --language en --translate fr
        modal run nemo_modal_app.py --language en --trim 300 --precision fp16
        modal run nemo_modal_app.py --language de --chunk-override 720          # any model
        modal run nemo_modal_app.py --language en --nemo-model qwen3-asr --chunk-override 1800
    """
    import time as _time

    local_io_path = Path(".")
    VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v",
                 ".wav", ".mp3", ".flac", ".m4a", ".ogg"}

    if translate and translate == language:
        print(f"⚠️  --translate {translate} is the same as --language {language}. Ignoring --translate.")
        translate = None
    # When --nemo-model + --translate are both given, run both jobs:
    #   transcription job  → specified model (e.g. qwen3-asr)
    #   translation job    → canary (only model that supports AST)
    explicit_model = nemo_model and nemo_model != DEFAULT_NEMO_MODEL
    run_both = bool(translate and explicit_model and "canary" not in nemo_model.lower())
    transcribe_model = select_nemo_model(language, nemo_model)
    translate_model  = MODEL_CANARY

    if video_filename is None:
        print(f"🔍 Searching for videos in: {local_io_path.absolute()}")
        found = [f for f in local_io_path.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXT]
        if not found:
            print(f"❌ No video files found in {local_io_path.absolute()}")
            print(f"   Supported formats: {', '.join(sorted(VIDEO_EXT))}")
            print(f"   Tip: run this command from the folder containing your video,")
            print(f"        or use --video-filename /full/path/to/video.mp4")
            return

        if diarize:
            srt_suffix = f".nemo.{language}.diarize.srt"
        elif translate:
            srt_suffix = f".nemo.{translate}.srt"
        else:
            srt_suffix = f".nemo.{language}.srt"

        if run_both:
            transcribe_suffix = f".nemo.{language}.srt"
            pending = [v for v in found if not (
                (v.parent / (v.stem + transcribe_suffix)).exists() and
                (v.parent / (v.stem + srt_suffix)).exists()
            )]
        else:
            pending = [v for v in found if not (v.parent / (v.stem + srt_suffix)).exists()]
        if not pending:
            print(f"✅ All {len(found)} video(s) already done!")
            return
        local_video_path = pending[0]
        print(f"🎯 Auto-detected video: {local_video_path.name}")
    else:
        # Accept both bare filename (in CWD) and full/relative paths
        p = Path(video_filename)
        local_video_path = p if p.is_absolute() else local_io_path / video_filename
        if not local_video_path.exists():
            print(f"❌ Video not found: {local_video_path.absolute()}")
            return

    if run_both:
        task_label = f"Transcription ({transcribe_model}) + Translation → {translate} ({translate_model})"
    elif translate:
        task_label = f"Translation → {translate}"
    else:
        task_label = "Transcription"

    print(f"\n{'=' * 60}")
    print(f"🎙️  NeMo ASR Pipeline")
    print(f"{'=' * 60}")
    print(f"📹 Video     : {local_video_path.absolute()}")
    if run_both:
        print(f"🧠 Transcribe: {transcribe_model}")
        print(f"🔀 Translate : {translate_model}  ({language} → {translate})")
    else:
        print(f"🧠 Model     : {transcribe_model if not translate else translate_model}")
        print(f"🌍 Language  : {language}" + (f" → {translate}" if translate else ""))
    print(f"🎯 Precision : {precision}")
    print(f"🗣️  Diarize   : {'✅' if diarize else '❌'}")
    print(f"✂️  Trim      : {_fmt_dur(trim) if trim > 0 else 'full video'}")
    print(f"⚙️  Safety    : {safety_factor:.0%}  | Reserve: {reserve_gb:.1f} GB")
    if chunk_override:
        print(f"📦 Chunk override: {_fmt_dur(chunk_override)}")
    print(f"🖥️  GPU       : {GPU_TYPE}")
    print(f"{'=' * 60}\n")

    print("📤 Reading video file…")
    with open(local_video_path, "rb") as f:
        video_data = f.read()
    print(f"   Size: {len(video_data) / 1024 / 1024:.1f} MB")

    _common = dict(
        video_filename=local_video_path.name,
        video_data=video_data,
        language=language,
        trim_sec=trim,
        precision=precision,
        diarize=diarize,
        safety_factor=safety_factor,
        reserve_gb=reserve_gb,
        chunk_override_sec=chunk_override,
    )

    wall_t0 = time.time()
    saved_paths = []

    if run_both:
        # ── Job 1: transcription ──────────────────────────────────────────
        print(f"🚀 [1/2] Transcribing with {transcribe_model}…\n")
        t_bytes = transcribe_nemo_remote.remote(**_common, nemo_model=transcribe_model, target_lang=None)
        srt_suffix = f".nemo.{language}.diarize.srt" if diarize else f".nemo.{language}.srt"
        p = local_io_path / (local_video_path.stem + srt_suffix)
        p.write_bytes(t_bytes)
        saved_paths.append(p)

        # ── Job 2: translation ────────────────────────────────────────────
        print(f"🚀 [2/2] Translating {language} → {translate} with {translate_model}…\n")
        tr_bytes = transcribe_nemo_remote.remote(**_common, nemo_model=translate_model, target_lang=translate)
        p2 = local_io_path / (local_video_path.stem + f".nemo.{translate}.srt")
        p2.write_bytes(tr_bytes)
        saved_paths.append(p2)
        srt_bytes = t_bytes  # use transcription for preview

    elif translate:
        print(f"🚀 Sending translation job to Modal…\n")
        srt_bytes = transcribe_nemo_remote.remote(**_common, nemo_model=translate_model, target_lang=translate)
        p = local_io_path / (local_video_path.stem + f".nemo.{translate}.srt")
        p.write_bytes(srt_bytes)
        saved_paths.append(p)

    else:
        print(f"🚀 Sending transcription job to Modal…\n")
        srt_bytes = transcribe_nemo_remote.remote(**_common, nemo_model=transcribe_model, target_lang=None)
        srt_suffix = f".nemo.{language}.diarize.srt" if diarize else f".nemo.{language}.srt"
        p = local_io_path / (local_video_path.stem + srt_suffix)
        p.write_bytes(srt_bytes)
        saved_paths.append(p)

    wall_elapsed = time.time() - wall_t0

    print(f"\n{'=' * 60}")
    print(f"✅ {task_label} complete! (wall time: {_fmt_dur(wall_elapsed)})")
    for p in saved_paths:
        print(f"📄 SRT saved: {p.absolute()}")
    print(f"{'=' * 60}")

    srt_text = srt_bytes.decode("utf-8")
    lines = srt_text.split("\n")
    preview = lines[:min(16, len(lines))]
    print("\n📋 Preview (first segments):")
    print("-" * 40)
    for line in preview:
        print(f"  {line}")
    seg_count = len([l for l in lines if l.strip() and l.strip().isdigit()])
    if len(lines) > 16:
        print(f"  … ({seg_count} segments total)")