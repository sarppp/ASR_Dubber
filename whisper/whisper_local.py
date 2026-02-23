"""
Local Whisper Transcription/Translation Pipeline
=================================================
Whisper-only, GPU-optimised, no cloud required.

Usage examples:

  # Auto-detect video, transcribe French
  python whisper_local.py --language fr --model turbo

  # Translate to English
  python whisper_local.py --language fr --model turbo --translate

  # With Demucs vocal isolation
  python whisper_local.py --language fr --model turbo --use-demucs

  # Trim to first 5 minutes (fast test run)
  python whisper_local.py --language de --model turbo --trim 300

  # Batch-process all videos in folder
  python whisper_local.py --language fr --model turbo --all

  # Specific file
  python whisper_local.py interview.mp4 --language en --model large-v3

  # Manual chunk size (useful when VRAM estimate is off)
  python whisper_local.py --language fr --model turbo --chunk-override 600

  # Full combo
  python whisper_local.py --language fr --model turbo --use-demucs --translate
"""

import argparse
import gc
import importlib
import inspect
import logging
import shutil
import subprocess
import time
import os
import wave

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("whisper_local")

# Cache whether the installed whisper build supports DecodingOptions.progress_callback
_PROGRESS_CALLBACK_SUPPORTED = None
_warned_progress_callback = False


def _supports_progress_callback() -> bool:
    """Return True if whisper.DecodingOptions accepts progress_callback."""
    global _PROGRESS_CALLBACK_SUPPORTED, _warned_progress_callback
    if _PROGRESS_CALLBACK_SUPPORTED is None:
        try:
            decoding_mod = importlib.import_module("whisper.decoding")
            DecodingOptions = getattr(decoding_mod, "DecodingOptions")
            _PROGRESS_CALLBACK_SUPPORTED = (
                "progress_callback" in inspect.signature(DecodingOptions.__init__).parameters
            )
        except Exception:
            _PROGRESS_CALLBACK_SUPPORTED = False
    if not _PROGRESS_CALLBACK_SUPPORTED and not _warned_progress_callback:
        log.debug("progress_callback not supported by this whisper build; skipping logger hook")
        _warned_progress_callback = True
    return bool(_PROGRESS_CALLBACK_SUPPORTED)


# ── Constants ─────────────────────────────────────────────────────────────────

VIDEO_EXT        = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}
DEFAULT_MODEL    = "turbo"
DEFAULT_LANGUAGE = "en"
CHUNK_OVERLAP_SEC = 2  # seconds of overlap between chunks to avoid cut-off words

# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_ts(seconds: float) -> str:
    """seconds → SRT timestamp  HH:MM:SS,mmm"""
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_dur(seconds: float) -> str:
    if seconds >= 60:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{seconds:.1f}s"


# ── VRAM helpers ──────────────────────────────────────────────────────────────

def _vram_gb() -> tuple[float, float]:
    """Return (free_gb, total_gb). Both 0.0 if no CUDA."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free, total = torch.cuda.mem_get_info()
    return free / 1024**3, total / 1024**3


def _log_vram(label: str) -> float:
    free, total = _vram_gb()
    if total > 0:
        log.info(f"   VRAM [{label}]: {free:.2f}/{total:.2f} GB free")
    return free


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _audio_duration(path: str) -> float:
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def _extract_audio(
    video_path: str,
    audio_path: str,
    sample_rate: int = 16000,
    trim_sec: int = 0,
) -> None:
    """
    Extract audio from video via ffmpeg.
    Uses -threads 0 so ffmpeg grabs all available CPU cores — much faster
    than the default single-threaded behaviour, especially for long files.
    """
    cmd = ["ffmpeg", "-y", "-threads", "0", "-i", video_path]
    if trim_sec > 0:
        cmd += ["-t", str(trim_sec)]
    cmd += ["-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", "1", audio_path]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.decode()}")


def _extract_audio_parallel(
    video_path: str,
    outputs: list[tuple[str, int, int]],  # [(out_path, sample_rate, trim_sec), ...]
) -> None:
    """
    When both 16 kHz (Whisper) and 44.1 kHz (Demucs) are needed, extract
    them simultaneously using a ThreadPoolExecutor instead of sequentially.
    Saves the full extraction time of the second format.
    """
    def _do(args):
        out_path, sr, trim = args
        if not Path(out_path).exists():
            _extract_audio(video_path, out_path, sr, trim)
            log.info(f"   ✅ Extracted {Path(out_path).name}  ({sr} Hz)")
        else:
            log.info(f"   ♻️  Reusing cached {Path(out_path).name}")

    with ThreadPoolExecutor(max_workers=len(outputs)) as ex:
        list(ex.map(_do, outputs))


# ── VRAM-aware model loading ──────────────────────────────────────────────────

# Approximate VRAM needed per model at fp16 (GB).
_MODEL_VRAM_GB = {
    "tiny":     0.5,
    "base":     0.8,
    "small":    1.5,
    "medium":   3.0,
    "large":    6.0,
    "large-v2": 6.0,
    "large-v3": 6.0,
    "turbo":    3.0,
}


def _keep_layer_norm_fp32(module) -> None:
    """Ensure LayerNorm parameters stay float32 for numerical stability."""
    for child in module.modules():
        if isinstance(child, torch.nn.LayerNorm):
            child.float()


def _load_model_vram_aware(model_name: str, device: str):
    """
    Load Whisper to CPU first, then move each child module to GPU one at a time
    (with a cache flush between each), rather than one-shot GPU load.
    This avoids OOM when VRAM is tight, because peak allocation during a bulk
    `.to(device)` can be ~2× the final model size.

    Falls back gracefully to CPU if any layer triggers OOM.
    """
    import whisper

    required  = _MODEL_VRAM_GB.get(model_name, 6.0)
    free_gb, total_gb = _vram_gb()

    if device == "cuda":
        log.info(f"   VRAM before model load: {free_gb:.2f}/{total_gb:.2f} GB free")
        log.info(f"   Estimated model need  : ~{required:.1f} GB (fp16)")
        if free_gb < required * 0.75:
            log.warning(
                f"   ⚠️  Low VRAM ({free_gb:.2f} GB). "
                f"Using layer-by-layer transfer to avoid OOM."
            )

    log.info(f"🧠 Loading Whisper '{model_name}' weights to CPU…")
    t0    = time.perf_counter()
    download_root = os.environ.get("WHISPER_CACHE_DIR", str(Path.home() / ".cache" / "whisper"))
    model = whisper.load_model(model_name, device="cpu", download_root=download_root)
    log.info(f"   CPU load done in {time.perf_counter() - t0:.1f}s")

    if device != "cuda":
        log.warning("   No CUDA GPU — running on CPU (expect slow transcription)")
        return model

    log.info("   Moving model to GPU layer-by-layer (fp16)…")
    try:
        for name, module in model.named_children():
            module.half().to(device)      # fp16 halves VRAM vs fp32
            torch.cuda.empty_cache()      # flush allocator after each layer
        model = model.to(device)          # anchor top-level references
        _keep_layer_norm_fp32(model)

        log.info("   ✅ Model on GPU (fp16)")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            log.warning("   OOM during GPU transfer — falling back to CPU inference")
            model = model.float().cpu()
            torch.cuda.empty_cache()
            gc.collect()
        else:
            raise

    model.eval()
    free_after, _ = _vram_gb()
    used = free_gb - free_after
    log.info(f"   VRAM after load: {free_after:.2f} GB free  (model ~{used:.2f} GB)")
    return model


# ── VRAM-based chunk size estimation ─────────────────────────────────────────

def _estimate_chunk_sec(
    model_name: str,
    safety_factor: float = 0.85,
    reserve_gb: float    = 1.0,
) -> int:
    """
    After model load, check remaining free VRAM and estimate the longest
    audio chunk we can process in one Whisper call without OOM.

    Rule of thumb (empirical, fp16): Whisper uses ~0.008 GB per second of audio.
    safety_factor  — shrinks the estimate (e.g. 0.85 = use 85% of free VRAM).
    reserve_gb     — hard headroom kept free for CUDA kernels / overhead.
    """
    free_gb, _ = _vram_gb()
    if free_gb <= 0:
        log.info("   No CUDA — defaulting to 300s chunks")
        return 300

    usable = max(0.0, free_gb - reserve_gb) * safety_factor
    if usable <= 0:
        log.warning(f"   Only {free_gb:.2f} GB free — using 60s chunks")
        return 60

    gb_per_sec = 0.008                         # Whisper fp16 empirical figure
    chunk_sec  = int(usable / gb_per_sec)
    chunk_sec  = max(30, min(chunk_sec, 1800)) # clamp: 30s – 30min

    log.info(
        f"   Free VRAM: {free_gb:.2f} GB  →  usable: {usable:.2f} GB  "
        f"→  auto chunk: {_fmt_dur(chunk_sec)}"
    )
    return chunk_sec


# ── Audio chunking ────────────────────────────────────────────────────────────

def _chunk_audio(
    audio_path: str,
    work_dir: Path,
    chunk_sec: int,
    overlap_sec: int = CHUNK_OVERLAP_SEC,
) -> list[tuple[str, float]]:
    """
    Split a WAV into overlapping chunks using ffmpeg (-threads 0 per chunk).
    Returns list of (chunk_wav_path, time_offset_seconds).
    If the file fits in one chunk, returns [(audio_path, 0.0)] — no splitting.
    """
    duration = _audio_duration(audio_path)
    if duration <= chunk_sec + 5:
        return [(audio_path, 0.0)]

    log.info(
        f"   Audio {_fmt_dur(duration)} > chunk {_fmt_dur(chunk_sec)} "
        f"— splitting with {overlap_sec}s overlap…"
    )

    chunks = []
    step   = chunk_sec - overlap_sec
    offset = 0.0
    idx    = 0

    while offset < duration:
        cp  = str(work_dir / f"_chunk_{idx:04d}.wav")
        dur = min(chunk_sec, duration - offset)
        subprocess.run(
            [
                "ffmpeg", "-y", "-threads", "0",
                "-ss", str(offset), "-i", audio_path,
                "-t", str(dur),
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", cp,
            ],
            check=True, capture_output=True,
        )
        chunks.append((cp, offset))
        offset += step
        idx    += 1

    log.info(f"   Split into {len(chunks)} chunk(s) of ~{_fmt_dur(chunk_sec)} each")
    return chunks


def _cleanup_chunks(chunks: list[tuple[str, float]], original: str) -> None:
    for path, _ in chunks:
        if path != original:
            Path(path).unlink(missing_ok=True)


# ── Overlap deduplication ─────────────────────────────────────────────────────

def _dedup_segments(segments: list) -> list:
    """
    When chunks overlap, Whisper may repeat the last sentence of chunk N
    at the start of chunk N+1. Remove segments whose start time falls
    before the end of the previous segment (overlap artefact).
    """
    if not segments:
        return segments
    out = [segments[0]]
    for seg in segments[1:]:
        if seg["start"] < out[-1]["end"] - 0.1:
            continue   # duplicate from overlap — drop
        out.append(seg)
    return out


# ── Progress logger ───────────────────────────────────────────────────────────

class _ProgressLogger:
    """Logs transcription progress every `interval` audio-seconds processed."""

    def __init__(self, total_duration: float, interval: float = 30.0):
        self._total    = total_duration
        self._interval = interval
        self._last     = 0.0
        self._t0       = time.perf_counter()

    def __call__(self, seek: int, _total: int):
        # Whisper's internal seek unit is centiseconds
        pos = seek / 100.0
        if pos - self._last >= self._interval or pos >= self._total * 0.999:
            elapsed = time.perf_counter() - self._t0
            pct     = min(100.0, 100.0 * pos / self._total) if self._total > 0 else 0
            eta     = (elapsed / pct * (100 - pct)) if pct > 0 else 0
            log.info(
                f"   ⏳ {pct:5.1f}%  "
                f"({_fmt_dur(pos)} / {_fmt_dur(self._total)})  "
                f"elapsed {_fmt_dur(elapsed)}  ETA {_fmt_dur(eta)}"
            )
            self._last = pos


# ── SRT output ────────────────────────────────────────────────────────────────

def _to_srt(segments: list) -> str:
    lines = []
    idx   = 0
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        idx += 1
        lines += [str(idx), f"{_fmt_ts(seg['start'])} --> {_fmt_ts(seg['end'])}", text, ""]
    return "\n".join(lines)


# ── Demucs vocal isolation ────────────────────────────────────────────────────

def _run_demucs(audio_path: str, work_dir: Path, demucs_model: str = "htdemucs") -> str:
    """Isolate vocals. Returns path to vocals.wav, or original audio on failure."""
    log.info(f"🎵 Running Demucs ({demucs_model})…")
    sep_dir = work_dir / "separated"
    result  = subprocess.run(
        ["demucs", "--two-stems=vocals", "-n", demucs_model, "-o", str(sep_dir), audio_path],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        vocals = list(sep_dir.rglob("vocals.wav"))
        if vocals:
            log.info("   ✅ Vocals isolated")
            return str(vocals[0])
    log.warning("   ⚠️  Demucs failed — falling back to original audio")
    return audio_path


# ── Core transcription ────────────────────────────────────────────────────────

def _transcribe_audio(
    audio_path: str,
    model,
    language: str,
    task: str,
    chunk_sec: int,
    work_dir: Path,
) -> list:
    """
    Transcribe (or translate) a WAV file.
    Automatically chunks the audio if it exceeds chunk_sec,
    then merges and deduplicates results.
    """
    import librosa

    chunks   = _chunk_audio(audio_path, work_dir, chunk_sec)
    all_segs = []

    for ci, (chunk_path, offset) in enumerate(chunks):
        if len(chunks) > 1:
            log.info(f"   🔊 Chunk {ci+1}/{len(chunks)}  (offset {_fmt_dur(offset)})")
            _log_vram(f"chunk {ci+1}")

        audio_data, _ = librosa.load(chunk_path, sr=16000)
        chunk_dur     = len(audio_data) / 16000

        params: dict = {
            "condition_on_previous_text": False,
            "compression_ratio_threshold": 2.4,
            "verbose": True,
            "fp16": torch.cuda.is_available(),
        }
        if task == "translate":
            params["task"] = "translate"
        else:
            params["task"]     = "transcribe"
            params["language"] = language

        # Progress hook only on single-chunk runs (multi-chunk has per-chunk logs)
        if len(chunks) == 1 and _supports_progress_callback():
            params["progress_callback"] = _ProgressLogger(chunk_dur)

        with torch.inference_mode():
            result = model.transcribe(audio_data, **params)

        # Shift timestamps by chunk's position in the full audio
        for seg in result.get("segments", []):
            all_segs.append({
                "start": seg["start"] + offset,
                "end":   seg["end"]   + offset,
                "text":  seg["text"],
            })

        # Release CUDA memory between chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    _cleanup_chunks(chunks, audio_path)
    return _dedup_segments(all_segs)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    video_path: str,
    language: str        = DEFAULT_LANGUAGE,
    model_name: str      = DEFAULT_MODEL,
    translate: bool      = False,
    use_demucs: bool     = False,
    demucs_model: str    = "htdemucs",
    trim_sec: int        = 0,
    chunk_override: int  = 0,
    safety_factor: float = 0.85,
    reserve_gb: float    = 1.0,
) -> str:
    """Run the full pipeline and return SRT content as a string."""

    wall_t0    = time.perf_counter()
    video_path = str(Path(video_path).resolve())
    work_dir   = Path(video_path).parent
    stem       = Path(video_path).stem
    task       = "translate" if translate else "transcribe"
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    trim_tag   = f"trim{trim_sec}" if trim_sec > 0 else "full"

    # ── Step 1: Audio extraction ───────────────────────────────────────────
    audio_16k = str(work_dir / f"{stem}_16k_{trim_tag}.wav")

    if use_demucs:
        # Extract both formats at the same time using threads
        audio_44k = str(work_dir / f"{stem}_44k_{trim_tag}.wav")
        log.info("🔄 Extracting audio in parallel (16 kHz + 44.1 kHz for Demucs)…")
        _extract_audio_parallel(video_path, [
            (audio_16k, 16000, trim_sec),
            (audio_44k, 44100, trim_sec),
        ])
    else:
        if not Path(audio_16k).exists():
            log.info("🔄 Extracting 16 kHz mono audio…")
            _extract_audio(video_path, audio_16k, 16000, trim_sec)
        else:
            log.info(f"♻️  Reusing cached {Path(audio_16k).name}")

    audio_dur = _audio_duration(audio_16k)
    log.info(f"⏱️  Audio duration: {_fmt_dur(audio_dur)}")

    # ── Step 2: Optional Demucs vocal isolation ───────────────────────────
    whisper_audio = audio_16k
    demucs_dir    = None
    if use_demucs:
        vocals = _run_demucs(audio_44k, work_dir, demucs_model)
        if vocals != audio_44k:
            demucs_dir    = work_dir / "separated"
            whisper_audio = vocals

    # ── Step 3: Load Whisper (layer-by-layer VRAM-safe) ───────────────────
    free_before = _log_vram("before model load")
    model       = _load_model_vram_aware(model_name, device)

    # ── Step 4: Auto-estimate chunk size from remaining VRAM ──────────────
    if chunk_override > 0:
        chunk_sec = max(30, min(int(chunk_override), 3600))
        log.info(f"   Manual chunk override: {_fmt_dur(chunk_sec)}")
    elif device == "cuda":
        chunk_sec = _estimate_chunk_sec(model_name, safety_factor, reserve_gb)
    else:
        chunk_sec = 300
        log.info(f"   CPU mode — using {_fmt_dur(chunk_sec)} chunks")

    # ── Step 5: Transcribe / translate ───────────────────────────────────
    log.info(
        f"🎧 Starting "
        f"{'translation → English' if translate else f'transcription [{language}]'}…"
    )
    t_asr    = time.perf_counter()
    segments = _transcribe_audio(
        whisper_audio, model,
        language=language, task=task,
        chunk_sec=chunk_sec, work_dir=work_dir,
    )
    asr_elapsed = time.perf_counter() - t_asr
    rtf         = asr_elapsed / audio_dur if audio_dur > 0 else 0

    log.info(
        f"✅ Whisper done  {asr_elapsed:.1f}s  "
        f"RTF {rtf:.2f}x  ({'faster' if rtf < 1 else 'slower'} than real-time)  "
        f"| {len(segments)} segment(s)"
    )

    # Free GPU memory immediately after inference
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
        _log_vram("after inference")

    # ── Step 6: Build SRT ─────────────────────────────────────────────────
    log.info("📝 Generating SRT…")
    srt_content = _to_srt(segments)

    # ── Step 7: Cleanup temp audio ────────────────────────────────────────
    for tmp in [audio_16k, work_dir / f"{stem}_44k_{trim_tag}.wav"]:
        Path(tmp).unlink(missing_ok=True)
    if demucs_dir and demucs_dir.exists():
        shutil.rmtree(demucs_dir, ignore_errors=True)

    # ── Summary ───────────────────────────────────────────────────────────
    wall_elapsed = time.perf_counter() - wall_t0
    log.info("=" * 58)
    log.info(f"  Total wall time  : {_fmt_dur(wall_elapsed)}")
    log.info(f"  Audio duration   : {_fmt_dur(audio_dur)}")
    log.info(f"  ASR time         : {_fmt_dur(asr_elapsed)}")
    log.info(f"  Real-time factor : {rtf:.2f}x")
    log.info(f"  Segments         : {len(segments)}")
    log.info("=" * 58)

    return srt_content


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Local Whisper transcription — GPU-optimised, no cloud needed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("video", nargs="?",
                   help="Video file (auto-detect in cwd if omitted)")
    p.add_argument("--all", action="store_true",
                   help="Process ALL pending videos in current directory")

    p.add_argument("--language", "-l", default=DEFAULT_LANGUAGE,
                   help="Source language code, e.g. en/fr/de/es  [default: en]")
    p.add_argument("--model", "-m", default=DEFAULT_MODEL,
                   choices=["tiny","base","small","medium","large","large-v2","large-v3","turbo"],
                   help=f"Whisper model  [default: {DEFAULT_MODEL}]")
    p.add_argument("--translate", action="store_true",
                   help="Translate to English instead of transcribing")

    p.add_argument("--use-demucs", action="store_true",
                   help="Isolate vocals with Demucs before transcription")
    p.add_argument("--demucs-model", default="htdemucs",
                   choices=["htdemucs","htdemucs_ft","mdx_extra"],
                   help="Demucs model  [default: htdemucs]")

    p.add_argument("--trim", type=int, default=0, metavar="SEC",
                   help="Process only the first N seconds  (0 = full video)")
    p.add_argument("--chunk-override", type=int, default=0, metavar="SEC",
                   help="Force chunk length in seconds  (0 = auto from VRAM)")
    p.add_argument("--safety-factor", type=float, default=0.85,
                   help="VRAM safety margin 0–1  [default: 0.85]")
    p.add_argument("--reserve-gb", type=float, default=1.0,
                   help="VRAM GB to keep free for CUDA overhead  [default: 1.0]")

    args = p.parse_args()

    if args.translate and args.language == "en":
        log.warning("--translate with --language en is a no-op. Ignoring --translate.")
        args.translate = False

    srt_suffix = ".en.srt" if args.translate else f".{args.language}.srt"

    # ── Find video(s) ────────────────────────────────────────────────────
    cwd    = Path.cwd()
    videos: list[Path] = []

    if args.video:
        vp = Path(args.video)
        if not vp.is_absolute():
            vp = cwd / vp
        if not vp.exists():
            log.error(f"Video not found: {vp}")
            return 1
        videos = [vp]
    else:
        found   = sorted(f for f in cwd.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXT)
        if not found:
            log.error(f"❌ No video files found in: {cwd}")
            log.error(f"   Supported formats: {', '.join(sorted(VIDEO_EXT))}")
            log.error(f"   Files seen: {[f.name for f in sorted(cwd.iterdir()) if f.is_file()]}")
            return 1
        pending = [v for v in found if not (v.parent / (v.stem + srt_suffix)).exists()]
        skipped = [v for v in found if (v.parent / (v.stem + srt_suffix)).exists()]
        if skipped:
            log.info(f"⏭️  Skipping {len(skipped)} already-done video(s):")
            for s in skipped:
                log.info(f"   ✓ {s.name}")
        if not pending:
            log.info("✅ All videos already have SRT files — nothing to do.")
            return 0
        videos = pending if args.all else [pending[0]]
        if not args.all and len(pending) > 1:
            log.info(f"   Found {len(pending)} pending — processing first. Use --all for all.")

    # ── Print config ─────────────────────────────────────────────────────
    log.info("=" * 58)
    log.info("  Whisper Local Pipeline")
    log.info("=" * 58)
    log.info(f"  Task          : {'TRANSLATE → English' if args.translate else f'TRANSCRIBE [{args.language}]'}")
    log.info(f"  Model         : {args.model}")
    log.info(f"  Demucs        : {'✅ ' + args.demucs_model if args.use_demucs else '❌'}")
    log.info(f"  Trim          : {_fmt_dur(args.trim) if args.trim else 'full video'}")
    log.info(f"  Chunk         : {'auto (VRAM-based)' if not args.chunk_override else _fmt_dur(args.chunk_override)}")
    log.info(f"  Safety factor : {args.safety_factor:.0%}")
    log.info(f"  Reserve VRAM  : {args.reserve_gb:.1f} GB")
    log.info(f"  Videos        : {len(videos)}")
    log.info("=" * 58)

    # ── Process ──────────────────────────────────────────────────────────
    ok = True
    for i, vp in enumerate(videos):
        if len(videos) > 1:
            log.info(f"\n── [{i+1}/{len(videos)}] {vp.name} ──")

        srt_out = vp.parent / (vp.stem + srt_suffix)
        if srt_out.exists():
            log.info(f"⏭️  Already exists: {srt_out.name} — skipping.")
            continue

        try:
            srt_content = run_pipeline(
                video_path     = str(vp),
                language       = args.language,
                model_name     = args.model,
                translate      = args.translate,
                use_demucs     = args.use_demucs,
                demucs_model   = args.demucs_model,
                trim_sec       = args.trim,
                chunk_override = args.chunk_override,
                safety_factor  = args.safety_factor,
                reserve_gb     = args.reserve_gb,
            )
        except Exception as exc:
            log.error(f"Pipeline failed for {vp.name}: {exc}", exc_info=True)
            ok = False
            continue

        srt_out.write_text(srt_content, encoding="utf-8")
        log.info(f"\n✅ Saved → {srt_out}")

        # Preview first few lines
        lines = srt_content.split("\n")
        for line in lines[:20]:
            log.info(f"  {line}")
        total_segs = srt_content.count("\n\n")
        if total_segs > 5:
            log.info(f"  … ({total_segs} segments total)")

    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())