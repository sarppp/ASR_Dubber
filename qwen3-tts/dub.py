#!/usr/bin/env python3
"""
dub.py — Video dubbing pipeline (NeMo SRT + Supertonic CPU (Qwen optional))
======================================================

Full workflow:
  1. nemo.py --diarize      → video.nemo.de.diarize.srt
  2. translate.py           → video.nemo.de.diarize_fr.srt   (Gemma via Ollama)
  3. dub.py (this script)   → final_dub.mp4

Usage:
  # With background music preservation (demucs):
  uv run python dub.py video.mp4 video.nemo.de.diarize_fr.srt

  # Without demucs (faster, replaces full audio track):
  uv run python dub.py video.mp4 video.nemo.de.diarize_fr.srt --no-demucs

  # Voice cloning mode:
  uv run python dub.py video.mp4 video.nemo.de.diarize_fr.srt --tts-engine qwen --qwen-mode clone

  # Clone but skip demucs (use original video audio for voice refs):
  uv run python dub.py video.mp4 video.nemo.de.diarize_fr.srt --tts-engine qwen --qwen-mode clone --no-demucs
"""

import argparse
import logging
import math
import queue as _queue
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
from dub_cache import file_digest, speech_key, restore as restore_speech, save as save_speech

log = logging.getLogger(__name__)


def _needed_seconds(message: str):
    match = re.search(r"needs ([0-9.]+)s", message or "")
    return float(match.group(1)) if match else None


def _required_tempo(message: str):
    match = re.search(r"required tempo ([0-9.]+)", message or "")
    return float(match.group(1)) if match else None
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from dub_srt import (
    QWEN_FEMALE_VOICES, _qwen_lang, parse_srt, merge_segments,
    build_voice_map, write_dub_srt,
)
from dub_audio import (
    extract_audio,
    separate_audio,
    extract_clone_refs,
    detect_speaker_genders,
    _qwen_python,
    _qwen_worker,
    PersistentTTSWorker,
    SharedTTSManager,
    speed_fit,
    split_tts_proportional,
    stitch_and_mix,
    _audio_duration,
    _save_checkpoint,
    _load_checkpoint,
)


# ---------------------------------------------------------------------------
# VRAM auto-detection
# ---------------------------------------------------------------------------

def _fit_window(start: float, end: float, next_start=None) -> float:
    """Borrow at most 250 ms; never cross the following utterance onset."""
    values = (start, end) if next_start is None else (start, end, next_start)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in values):
        raise ValueError("Speech times must be finite numbers")
    if start < 0 or end <= start:
        raise ValueError("Speech timestamps must have a positive duration")
    deadline = end if next_start is None else min(end + .25, next_start)
    if deadline <= start:
        raise ValueError("Speech onsets must be ordered and distinct")
    return deadline - start


def _auto_workers(device_ids: List[int]) -> int:
    """Return the number of TTS worker threads.
    
    With SharedTTSManager, VRAM is constant (~6GB) regardless of worker count.
    Workers overlap CPU speed_fit with GPU synthesis.
    
    2 workers is the sweet spot: one synthesizes while other does speed_fit.
    More workers add overhead without much benefit.
    
    Multi-GPU: same logic, but each GPU gets its own model (not yet implemented).
    """
    # With shared model, VRAM no longer scales with workers
    # 2 workers = optimal CPU/GPU overlap
    return 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dub a video from a pre-translated diarized SRT using Supertonic CPU or Qwen TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Full pipeline:
  1. uv run --project ../nemo python nemo.py video.mp4 --language de --diarize
  2. python translate.py   (Gemma via Ollama → translated SRT)
  3. uv run python dub.py video.mp4 translated.srt --language fr

Examples:
  uv run python dub.py video.mp4 video.nemo.de.diarize_fr.srt --language fr
  uv run python dub.py video.mp4 video.nemo.de.diarize_fr.srt --language fr --tts-engine qwen --qwen-mode clone
  uv run python dub.py video.mp4 video.nemo.de.diarize_fr.srt --language fr --no-demucs
        """,
    )
    parser.add_argument("video", nargs="?", default=None,
                        help="Input video file (auto-discovered in --search-dir if omitted)")
    parser.add_argument("srt",   nargs="?", default=None,
                        help="Pre-translated diarized SRT (auto-discovered if omitted)")
    parser.add_argument("--tts-engine", default="supertonic", choices=["supertonic", "qwen"],
                        help="TTS backend (default: supertonic on CPU)")
    parser.add_argument("--supertonic-voice", default="M1",
                        choices=[f"{gender}{i}" for gender in ("M", "F") for i in range(1, 6)],
                        help="Supertonic preset voice (default: M1, same as the comparison)")
    parser.add_argument("--language",   default="fr",
                        help="Target language for speech synthesis (default: fr)")
    parser.add_argument("--qwen-mode",  default="clone", choices=["custom", "clone"],
                        help="'custom' = fixed voice | 'clone' = voice cloned from speaker (default: clone)")
    parser.add_argument("--no-demucs",  action="store_true",
                        help="Skip vocal separation — faster but loses background music")
    parser.add_argument("--search-dir", default="../nemo",
                        help="Folder to auto-discover video + SRT from (default: ../nemo)")
    parser.add_argument("--workdir",    default="output/dub",
                        help="Working directory for intermediate files (default: output/dub)")
    parser.add_argument("--qwen-dir",   default=".",
                        help="Path to the qwen3-tts uv project (default: current folder)")
    parser.add_argument("--max-speed",  type=float, default=1.0,
                        help="Maximum tempo; native speed by default, overflow requires review")
    parser.add_argument("--strict-timing", action="store_true",
                        help="Reject native speech ending over 400 ms before its source passage")
    parser.add_argument("--min-speed",  type=float, default=1.0,
                        help="Native minimum tempo; only 1.0 is supported (no artificial slowing)")
    parser.add_argument("--merge-gap",  type=float, default=1.0,
                        help="Merge consecutive same-speaker segments with gap ≤ N s "
                             "for more natural TTS (default: 1.0, set 0 to disable)")
    parser.add_argument("--merge-max-dur", type=float, default=10.0,
                        help="Max duration (seconds) for a merged segment "
                             "(default: 10 — prevents giant TTS blocks)")
    parser.add_argument("--tts-workers", default="auto",
                        help="Number of parallel TTS workers, or 'auto' to detect from VRAM "
                             "(default: auto). Each worker needs ~4 GB VRAM.")
    parser.add_argument("--tts-devices", default="0",
                        help="Comma-separated GPU IDs for workers, e.g. '0,1,2' "
                             "(default: '0'). Workers round-robin across devices.")
    parser.add_argument("--source-script", type=Path, help="Original ASR subtitles: enables measured word alignment and final combined subtitles")
    parser.add_argument("--source-language", default="en")
    parser.add_argument("--placement-mode", choices=["continuous_synced", "continuous", "source_anchored"], default="continuous_synced",
                        help="With --source-script: bounded continuous sync, raw continuous, or source-anchored placement")
    parser.add_argument("--source-timing", type=Path)
    parser.add_argument("--comparison-script", action="append", default=[], help="LANG=original-aligned subtitle path")
    parser.add_argument("--voice-reference", type=Path)
    args = parser.parse_args()
    if args.min_speed != 1.0 or not 1.0 <= args.max_speed <= 1.15:
        parser.error("Natural fitting requires --min-speed 1.0 and --max-speed between 1.0 and 1.15")

    if args.strict_timing and args.max_speed != 1.:
        parser.error("--strict-timing requires --max-speed 1.0")

    if args.tts_engine == "supertonic":
        from supertonic_tts_worker import language_code
        try:
            language_code(args.language)
        except ValueError as exc:
            parser.error(str(exc))
        args.qwen_mode = "custom"  # Supertonic uses preset voices, no clone extraction.

    search_dir = Path(args.search_dir).resolve()
    VIDEO_EXT  = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}

    # ── Auto-discover video + SRT ─────────────────────────────────────────────
    if args.video is None or args.srt is None:
        if not search_dir.exists():
            log.error(f"Search dir not found: {search_dir}"); return 1

        # Find translated SRTs: *.diarize_??.srt
        srt_candidates = sorted(search_dir.glob("*.diarize_??.srt"))
        if not srt_candidates and args.srt is None:
            log.error(
                f"No translated SRTs found in {search_dir} — "
                "expected pattern: *.nemo.LANG.diarize_TARGETLANG.srt"
            )
            return 1

        chosen_srt = Path(args.srt).resolve() if args.srt else srt_candidates[0]
        if len(srt_candidates) > 1 and args.srt is None:
            log.info(f"Multiple SRTs found, using: {chosen_srt.name}")

        # Derive video stem: impost_trimmed_2min.nemo.de.diarize_fr.srt → impost_trimmed_2min
        stem_match = re.match(r"^(.+?)\.nemo\.", chosen_srt.name)
        video_stem = stem_match.group(1) if stem_match else None

        if args.video:
            chosen_video = Path(args.video).resolve()
        elif video_stem:
            matches = [f for f in search_dir.iterdir()
                       if f.stem == video_stem and f.suffix.lower() in VIDEO_EXT]
            if not matches:
                log.error(f"No video found for stem '{video_stem}' in {search_dir}"); return 1
            chosen_video = matches[0]
        else:
            videos = [f for f in search_dir.iterdir() if f.suffix.lower() in VIDEO_EXT]
            if not videos:
                log.error(f"No video files found in {search_dir}"); return 1
            chosen_video = videos[0]

        video_path = chosen_video.resolve()
        srt_path   = chosen_srt.resolve()
    else:
        video_path = Path(args.video).resolve()
        srt_path   = Path(args.srt).resolve()

    if not video_path.exists():
        log.error(f"Video not found: {video_path}"); return 1
    if not srt_path.exists():
        log.error(f"SRT not found: {srt_path}"); return 1

    qwen_dir   = Path(args.qwen_dir).resolve()
    work_dir   = Path(args.workdir).resolve()
    import hashlib
    cache_key = hashlib.sha256(srt_path.read_bytes() + repr((
        "natural-fit-v8-selected-placement", args.placement_mode, args.strict_timing, args.tts_engine, args.supertonic_voice, args.qwen_mode, args.language,
        args.merge_gap, args.merge_max_dur, args.min_speed, args.max_speed,
        str(video_path), video_path.stat().st_mtime_ns,
    )).encode()).hexdigest()[:16]
    cache_dir = work_dir / "cache" / args.tts_engine / cache_key
    temp_dir   = cache_dir / "temp"
    output_dir = work_dir / "output"
    cast_dir   = work_dir / "cast_samples"

    for d in (work_dir, temp_dir, output_dir):
        d.mkdir(parents=True, exist_ok=True)

    script_dir  = Path(__file__).resolve().parent
    qwen_python = _qwen_python(qwen_dir)
    qwen_worker = (str(script_dir / "supertonic_tts_worker.py")
                   if args.tts_engine == "supertonic" else _qwen_worker(script_dir))

    t_start = time.perf_counter()

    log.info("=" * 60)
    log.info(f"Video        : {video_path.name}")
    log.info(f"SRT          : {srt_path.name}")
    log.info(f"Language     : {args.language}")
    log.info(f"TTS engine   : {args.tts_engine}")
    log.info(f"Voice mode   : {args.supertonic_voice if args.tts_engine == 'supertonic' else args.qwen_mode}")
    log.info(f"Demucs       : {'disabled' if args.no_demucs else 'enabled'}")
    log.info(f"Qwen python  : {qwen_python}")
    log.info("=" * 60)

    # ── 1. Parse SRT ─────────────────────────────────────────────────────────
    segments = parse_srt(srt_path, strict=True)
    if not segments:
        log.error("No segments parsed — make sure this is a diarized+translated SRT")
        return 1
    alignment_plan = None
    synthesis_bundles = None
    if args.source_script:
        if args.merge_gap != 0 or args.max_speed != 1.:
            parser.error("Source-aligned dubbing requires native speed and already grouped passages (--merge-gap 0)")
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nemo"))
        from batch_dubbing.aligned_finish import source_plan
        from batch_dubbing.continuous_context import bundle_segments
        import importlib.util
        import os
        nemo_dir = Path(__file__).resolve().parent.parent / "nemo"
        alignment_python = os.environ.get("ALIGNMENT_PYTHON") or (
            sys.executable if importlib.util.find_spec("qwen_asr") else str(nemo_dir / ".venv/bin/python"))
        source_timing = work_dir / "source_forced_alignment.json"
        source_request = dict(operation="align_source", video=str(video_path),
                              source_script=str(args.source_script), source_language=args.source_language,
                              output=str(source_timing))
        source_request_path = work_dir / "source_alignment_request.json"
        source_request_path.write_text(json.dumps(source_request, indent=2))
        log.info("Aligning original speech before synthesis")
        from batch_dubbing.align_service import run_stage
        run_stage(source_request_path, python=alignment_python, timeout=max(180, 90*len(segments)), check=True)
        comparisons = dict(value.split("=", 1) for value in args.comparison_script)
        alignment_plan = source_plan(args.source_script, srt_path, comparisons, args.source_language, args.language, source_timing=source_timing)
        segments = bundle_segments(segments)
        synthesis_bundles = [seg["passage_indices"] for seg in segments]
        os.environ["QWEN_TTS_CONTINUOUS"] = "1"
        log.info("Continuous synthesis: %s contexts for %s complete source passages", len(segments), len(alignment_plan["passages"]))
        (work_dir / "source_alignment_plan.json").write_text(json.dumps(alignment_plan, ensure_ascii=False, indent=2))
    if args.merge_gap > 0:
        segments = merge_segments(segments, gap_sec=args.merge_gap,
                                  max_dur=args.merge_max_dur)

    report_path = work_dir / "fit_report.json"
    report_base = {"schema_version": 2, "script_sha256": file_digest(srt_path),
                   "engine": args.tts_engine, "language": args.language,
                   "cache_dir": str(cache_dir), "max_speed": args.max_speed,
                   "expected_segments": segments}
    report_path.write_text(json.dumps({**report_base, "status": "running",
                                       "segments": []}, ensure_ascii=False, indent=1))

    # Compute reference extraction extent; never use it to trim the picture
    srt_end = max(s["end"] for s in segments)

    # ── 2. Audio separation or raw extract ───────────────────────────────────
    # Pass srt_end as trim so demucs/ffmpeg only processes the audio we actually need.
    background: Optional[Path] = None
    if args.no_demucs:
        # Just extract raw audio for clone refs (if needed)
        audio_for_refs = temp_dir / "input_raw.wav"
        if args.qwen_mode == "clone":
            extract_audio(video_path, audio_for_refs, trim_sec=srt_end)
    else:
        vocals, background = separate_audio(video_path, temp_dir, trim_sec=0)
        audio_for_refs = vocals

    # ── 3. Clone refs ─────────────────────────────────────────────────────────
    clone_refs: Dict[str, Path] = {}
    if args.qwen_mode == "clone":
        if args.voice_reference:
            if not args.voice_reference.is_file() or len({seg["speaker"] for seg in segments}) != 1:
                raise ValueError("A supplied voice reference requires an existing WAV and one source speaker")
            clone_refs = {seg["speaker"]: args.voice_reference.resolve() for seg in segments}
        else:
            clone_refs = extract_clone_refs(segments, audio_for_refs, cast_dir)
        if alignment_plan is not None and not clone_refs:
            raise ValueError("No clone reference; refusing an unrequested voice substitution")
        if not clone_refs:
            log.warning("⚠️  No clone refs extracted — falling back to custom mode")
            args.qwen_mode = "custom"

    # ── 4. Voice map ─────────────────────────────────────────────────────────
    # Always build as fallback, but only log if we're actually in custom mode
    gender_hints = detect_speaker_genders(clone_refs) if clone_refs else {}
    voice_map = ({seg["speaker"]: args.supertonic_voice for seg in segments}
                 if args.tts_engine == "supertonic"
                 else build_voice_map(segments, gender_hints=gender_hints))
    if args.qwen_mode == "custom":
        log.info("🎤 Voice assignments (custom mode):")
        for spk, voice in voice_map.items():
            log.info(f"   {spk} → {voice}")
    else:
        log.info("🎤 Mode: clone — custom voices are fallback only")
        for spk, voice in voice_map.items():
            log.info(f"   {spk} → clone ref (fallback: {voice})")

    # ── 5. TTS loop ──────────────────────────────────────────────────────────
    # N workers (--tts-workers), each on its assigned GPU (--tts-devices).
    # Workers run in parallel threads, each owning one TTS subprocess.
    # Model loading is serialized via PersistentTTSWorker._startup_lock so
    # VRAM isn't double-allocated during init.
    # Speed-fit (CPU/ffmpeg) runs in a shared thread pool alongside TTS.
    #
    # Sizing guide (with SharedTTSManager):
    #   Shared model architecture: VRAM is constant (~6GB) regardless of worker count
    #   Always use 2 workers for optimal CPU/GPU overlap
    #   Works on any GPU with ~6GB+ VRAM (8GB, 12GB, 16GB+, etc.)
    #   Multi-GPU: same logic, but each GPU gets its own model (not yet implemented)
    qwen_language = (args.language if args.tts_engine == "supertonic"
                     else _qwen_lang(args.language))
    log.info(f"Qwen language: '{args.language}' → '{qwen_language}'")
    checkpoint_path = cache_dir / "checkpoint.json"
    final_files: List[Tuple[Path, float, float]] = _load_checkpoint(checkpoint_path)
    done_indices = {int(Path(c).stem.split("_")[1]) for c, _, _ in final_files}

    todo = [seg for seg in segments if seg["index"] not in done_indices]
    log.info(f"🗣️  Synthesising {len(segments)} segments "
             f"({len(done_indices)} cached, {len(todo)} remaining)")

    # Preserve pauses: only a small end allowance may be borrowed, and the
    # next utterance onset is always a hard deadline.
    _available: Dict[int, float] = {}
    for idx, seg in enumerate(segments):
        next_start = segments[idx + 1]["start"] if idx + 1 < len(segments) else None
        _available[seg["index"]] = _fit_window(seg["start"], seg["end"], next_start)

    # A whole-script hash is useful for fitted checkpoints, but it used to
    # force every unchanged sentence to be synthesised again on each retry.
    # Share raw speech by exact synthesis identity; refit it for the new slot.
    raw_cache = work_dir / "raw_speech"
    raw_keys = {}
    worker_digest = file_digest(Path(qwen_worker))
    for seg in todo:
        ref = clone_refs.get(seg["speaker"])
        settings = {
            "engine": args.tts_engine, "mode": args.qwen_mode,
            "language": qwen_language, "speaker": seg["speaker"],
            "voice": voice_map.get(seg["speaker"]), "worker": worker_digest,
            "reference": file_digest(ref) if ref and ref.exists() else None,
        }
        key = speech_key(seg["text"], settings)
        raw_keys[seg["index"]] = key
        raw = temp_dir / f"seg_{seg['index']:04d}.wav"
        if not raw.exists():
            restore_speech(raw_cache, key, raw)
    needs_synthesis = any(not (temp_dir / f"seg_{seg['index']:04d}.wav").exists()
                          or (temp_dir / f"seg_{seg['index']:04d}.wav").stat().st_size <= 500
                          for seg in todo)

    device_ids = ([None] if args.tts_engine == "supertonic"
                  else [int(x.strip()) for x in args.tts_devices.split(",")])
    n_workers  = (_auto_workers(device_ids)
                  if args.tts_workers == "auto"
                  else int(args.tts_workers))
    log.info(f"TTS workers: {n_workers}  |  devices: {device_ids}")
    t_tts_start = time.perf_counter()

    pbar             = tqdm(total=len(segments), initial=len(done_indices), desc="TTS",
                            unit="seg", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                                    "[{elapsed}<{remaining}, {rate_fmt}]")
    final_files_lock = threading.Lock()
    fit_lock         = threading.Lock()
    worker_exc_lock  = threading.Lock()
    worker_exceptions: List[Exception] = []

    seg_queue: _queue.Queue = _queue.Queue()
    for seg in todo:
        seg_queue.put(seg)

    fit_pool    = ThreadPoolExecutor(max_workers=1)
    fit_futures: List = []
    fit_report: List[dict] = []
    fit_report_lock = threading.Lock()

    def _fit_record(seg, fitted=None, error=None):
        record = {"index": seg["index"], "start": seg["start"],
                  "end": seg["end"], "text": seg["text"],
                  "span": _available[seg["index"]]}
        if error is not None:
            record.update(needed=_needed_seconds(str(error)),
                          tempo=_required_tempo(str(error)), error=str(error),
                          status="failed")
        else:
            sidecar = json.loads(fitted.with_suffix(".json").read_text())
            needed = sidecar["edges"]["retained_seconds"]
            record.update(needed=needed, tempo=max(1.0, needed / record["span"]),
                          applied_tempo=sidecar["tempo"], status="native_measured" if alignment_plan is not None else "passed")
        return record

    by_index = {seg["index"]: seg for seg in segments}
    for fitted, start, end in final_files:
        idx = int(fitted.stem.split("_")[1])
        fit_report.append(_fit_record(by_index[idx], fitted=fitted))

    # ── Shared TTS Managers (one model instance per mode) ─────────────────────
    # Instead of each worker loading its own model, we use shared managers.
    # This keeps VRAM constant (~6GB) regardless of n_workers.
    # Workers submit requests to the shared manager and wait for results.
    clone_manager: Optional[SharedTTSManager] = None
    custom_manager: Optional[SharedTTSManager] = None
    clone_broken_global = (args.qwen_mode != "clone")
    
    # Only load clone model if we have clone refs and clone mode is enabled
    if needs_synthesis and not clone_broken_global and clone_refs:
        log.info("Loading shared clone TTS model...")
        clone_manager = SharedTTSManager(
            "clone", qwen_python, qwen_worker, device_id=device_ids[0]
        )
        clone_manager.start()
    elif needs_synthesis and clone_broken_global:
        # Not in clone mode - load custom model upfront
        log.info("Loading shared custom TTS model...")
        custom_manager = SharedTTSManager(
            "custom", qwen_python, qwen_worker, device_id=device_ids[0]
        )
        custom_manager.start()
    else:
        # No clone refs but in clone mode - will fallback to custom lazily
        clone_broken_global = True

    def _do_fit(raw_out: Path, available_dur: float, start: float, end: float,
                seg_index: int = -1) -> None:
        # The safe fitter measures speech after edge cleanup. Native short
        # speech stays native even when a small end allowance is available.
        try:
            fitted = speed_fit(raw_out, (_audio_duration(raw_out) + .1) if alignment_plan is not None else available_dur, max_speed=args.max_speed,
                               min_speed=args.min_speed)
            if args.strict_timing:
                from batch_dubbing.anchored_timeline import validate_end_timing
                validate_end_timing(json.loads(fitted.with_suffix(".json").read_text()), start, end)
        except Exception as exc:
            with fit_report_lock:
                fit_report.append(_fit_record(by_index[seg_index], error=exc))
            raise
        with fit_report_lock:
            fit_report.append(_fit_record(by_index[seg_index], fitted=fitted))
        with final_files_lock:
            final_files.append((fitted, start, end))
            _save_checkpoint(checkpoint_path, final_files)

    def _do_fit_split(raw_out: Path, subsegments: list) -> None:
        """Split TTS proportionally across original sub timings, fit each slice."""
        slices = split_tts_proportional(raw_out, subsegments, temp_dir, raw_out.stem)
        for slice_wav, sub_start, sub_end in slices:
            slot = max(0.1, sub_end - sub_start)
            fitted = speed_fit(slice_wav, slot, max_speed=args.max_speed)
            with final_files_lock:
                final_files.append((fitted, sub_start, sub_end))
        _save_checkpoint(checkpoint_path, final_files)

    def _run_worker(device_id: Optional[int]) -> None:
        """Worker thread: pulls segments from queue, uses shared TTS managers."""
        nonlocal custom_manager, clone_broken_global
        clone_broken_local = clone_broken_global
        try:
            while True:
                try:
                    seg = seg_queue.get_nowait()
                except _queue.Empty:
                    break
                i          = seg["index"]
                spk        = seg["speaker"]
                text       = seg["text"]
                start, end = seg["start"], seg["end"]
                target_dur = _available[i]
                log.info(f"   [{i:04d}] text={repr(text)}")
                raw_out    = temp_dir / f"seg_{i:04d}.wav"
                ok         = False
                generated = False
                used_fallback = False

                if raw_out.exists() and raw_out.stat().st_size > 500:
                    ok = True
                else:
                    generated = True
                    # Try clone mode first (if available and not broken)
                    if not clone_broken_local and clone_manager:
                        ref = clone_refs.get(spk)
                        if ref and ref.exists():
                            log.info(f"   [{i:04d}] 🎙️  clone ({spk})")
                            ok = clone_manager.generate_clone(text, ref, qwen_language, raw_out)
                            if not ok:
                                log.warning(f"   [{i:04d}] Clone failed — falling back to custom")
                                clone_broken_local = True
                        else:
                            log.warning(f"   [{i:04d}] No clone ref for '{spk}'")

                    if not ok and alignment_plan is not None:
                        raise RuntimeError("Requested clone synthesis failed; no automatic voice fallback")

                    # Fallback to custom mode - load model lazily if needed
                    if not ok:
                        used_fallback = args.qwen_mode == "clone"
                        # Load custom model on first fallback (thread-safe via lock)
                        if custom_manager is None:
                            with fit_lock:  # reuse fit_lock as general lock
                                if custom_manager is None:
                                    log.info("Loading shared custom TTS model (fallback)...")
                                    custom_manager = SharedTTSManager(
                                        "custom", qwen_python, qwen_worker, device_id=device_ids[0]
                                    )
                                    custom_manager.start()
                        
                        if custom_manager:
                            voice = voice_map.get(spk, QWEN_FEMALE_VOICES[0])
                            log.info(f"   [{i:04d}] 🔊 custom voice: {voice}")
                            ok = custom_manager.generate_custom(text, voice, qwen_language, raw_out)
                            if not ok:
                                log.error(f"   [{i:04d}] Custom TTS also failed — skipping")

                if ok and raw_out.exists():
                    # Never label a fallback voice as the requested clone.
                    if generated and not used_fallback:
                        save_speech(raw_cache, raw_keys[i], raw_out)
                    fut = fit_pool.submit(_do_fit, raw_out, target_dur, start, end, i)
                    with fit_lock:
                        fit_futures.append(fut)

                pbar.update(1)

        except Exception as exc:
            log.error(f"TTS worker thread failed: {exc}")
            with worker_exc_lock:
                worker_exceptions.append(exc)
        # No finally block needed - shared managers are closed outside

    worker_threads = [
        threading.Thread(
            target=_run_worker,
            args=(device_ids[i % len(device_ids)],),
            daemon=True,
            name=f"tts-{i}",
        )
        for i in range(n_workers)
    ]
    for t in worker_threads:
        t.start()
    for t in worker_threads:
        t.join()

    # Close shared TTS managers (free VRAM)
    if clone_manager:
        clone_manager.close()
    if custom_manager:
        custom_manager.close()

    pbar.close()

    # Resolve every fit before publishing; a missing phrase is a failed dub.
    fit_errors = []
    for fut in fit_futures:
        try:
            fut.result()
        except Exception as exc:
            fit_errors.append(exc)
            log.error(f"Speech could not fit safely: {exc}")
    fit_pool.shutdown(wait=True)

    reported = {row["index"] for row in fit_report}
    for seg in segments:
        if seg["index"] not in reported:
            fit_report.append(_fit_record(seg, error="No audio was generated"))
    complete = not (fit_errors or worker_exceptions or len(final_files) != len(segments))
    report_path.write_text(json.dumps({**report_base,
                                      "status": ("native_measured" if alignment_plan is not None else "passed") if complete else "failed",
                                      "segments": sorted(fit_report, key=lambda r: r["start"])},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
    log.info(f"Fit report → {report_path}")

    if fit_errors or worker_exceptions or len(final_files) != len(segments):
        log.error("Dubbing incomplete; refusing to publish a track with missing speech. "
                  "Successful checkpoints are preserved for review and resume.")
        return 1

    # Sort by start time — parallel workers may have appended out of order
    final_files.sort(key=lambda x: x[1])

    if not final_files:
        log.error("No audio was generated. Check TTS errors above.")
        return 1

    if alignment_plan is not None:
        import importlib.util
        import os
        nemo_dir = Path(__file__).resolve().parent.parent / "nemo"
        alignment_python = os.environ.get("ALIGNMENT_PYTHON")
        if not alignment_python:
            alignment_python = sys.executable if importlib.util.find_spec("qwen_asr") else str(nemo_dir / ".venv/bin/python")
        takes = []
        for fitted, start, end in final_files:
            index = int(fitted.stem.split("_")[1])
            sidecar = json.loads(fitted.with_suffix(".json").read_text())
            raw = temp_dir / f"seg_{index:04d}.wav"
            if not raw.is_file():
                raise ValueError("Continuous placement requires its complete original recording")
            takes.append(dict(index=index, file=str(raw.resolve()), text=by_index[index]["text"],
                              sha256=file_digest(raw), tempo=sidecar["tempo"]))
        request = dict(video=str(video_path), output=str(output_dir), plan=alignment_plan, placement_mode=args.placement_mode,
                       takes=takes, bundles=synthesis_bundles, background=str(background) if background else None)
        request_path = work_dir / "alignment_request.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2))
        from batch_dubbing.align_service import run_stage
        returncode = run_stage(request_path, python=alignment_python, timeout=max(180, 90*len(takes)))
        if returncode:
            log.error("Alignment or native placement failed; see alignment_report.json. No partial final dub is accepted.")
        return returncode

    # ── 6. Stitch + mix ──────────────────────────────────────────────────────
    # Preserve the full source video, including scenes after the final subtitle.
    t_stitch_start = time.perf_counter()
    log.info("🎬 Stitching and mixing…")
    final, actual_positions = stitch_and_mix(
        final_files, video_path, output_dir, temp_dir,
        background=background,   # None when --no-demucs
        trim_to=None,
    )

    # ── 7. Write dubbed SRT with actual audio timestamps ─────────────────
    dub_srt_path = output_dir / srt_path.name.replace(".srt", "_dub.srt")
    write_dub_srt(dub_srt_path, actual_positions, segments)

    t_end = time.perf_counter()
    t_total   = t_end - t_start
    t_tts     = t_stitch_start - t_tts_start
    t_stitch  = t_end - t_stitch_start
    t_pre     = t_tts_start - t_start

    def _fmt(s: float) -> str:
        m, sec = divmod(int(s), 60)
        return f"{m}m {sec:02d}s" if m else f"{sec}s"

    log.info("=" * 60)
    log.info(f"✅ Done!  →  {final}")
    log.info(f"📝 Dub SRT →  {dub_srt_path}")
    log.info(f"⏱  Pre-TTS  (dub): {_fmt(t_pre)}")
    log.info(f"⏱  TTS+fit  (dub): {_fmt(t_tts)}")
    log.info(f"⏱  Stitch   (dub): {_fmt(t_stitch)}")
    log.info(f"⏱  Total    (dub): {_fmt(t_total)}")
    log.info("=" * 60)

    import json as _json, datetime as _dt
    timing_path = output_dir / f"{video_path.stem}_timing_dub.json"
    timing_path.write_text(_json.dumps({
        "tts_engine": args.tts_engine,
        "tts_device": "cpu" if args.tts_engine == "supertonic" else args.tts_devices,
        "voice": args.supertonic_voice if args.tts_engine == "supertonic" else args.qwen_mode,
        "video":    video_path.name,
        "srt":      srt_path.name,
        "date":     _dt.datetime.now().isoformat(timespec="seconds"),
        "segments": len(segments),
        "pre_tts_sec":  round(t_pre,    1),
        "tts_sec":      round(t_tts,    1),
        "stitch_sec":   round(t_stitch, 1),
        "total_sec":    round(t_total,  1),
        "pre_tts":  _fmt(t_pre),
        "tts":      _fmt(t_tts),
        "stitch":   _fmt(t_stitch),
        "total":    _fmt(t_total),
    }, indent=2))
    log.info(f"⏱  Timing saved → {timing_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
