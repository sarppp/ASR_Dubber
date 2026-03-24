#!/usr/bin/env python3
"""
dub.py — Video dubbing pipeline (NeMo SRT + Qwen TTS)
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
  uv run python dub.py video.mp4 video.nemo.de.diarize_fr.srt --qwen-mode clone

  # Clone but skip demucs (use original video audio for voice refs):
  uv run python dub.py video.mp4 video.nemo.de.diarize_fr.srt --qwen-mode clone --no-demucs
"""

import argparse
import logging
import queue as _queue
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from dub_srt import (
    QWEN_FEMALE_VOICES, _qwen_lang, parse_srt, merge_segments,
    build_voice_map, write_dub_srt,
)
from dub_audio import (
    extract_audio,
    separate_audio,
    extract_clone_refs,
    _qwen_python,
    _qwen_worker,
    PersistentTTSWorker,
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

def _auto_workers(device_ids: List[int]) -> int:
    """Return the number of TTS workers that fit across the given devices.

    On a SINGLE GPU multiple workers don't speed up inference — they all share
    the same GPU compute and the autoregressive decoder runs sequentially.
    Extra workers only help overlap CPU speed_fit with GPU synthesis; 2-3
    saturates that benefit. Beyond that, startup cost dominates.

    On MULTIPLE GPUs each device gets its own worker for true parallelism.

    Single-GPU tiers (free VRAM at query time):
      < 8 GB  → 1 worker
      8-11 GB → 2 workers
      ≥ 12 GB → 3 workers  (sweet spot regardless of total VRAM)

    Multi-GPU: same tiers applied per device, summed across all devices.
    e.g. 4× 16 GB → 2 workers per GPU → 8 workers total.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        # nvidia-smi returns MiB, one line per physical GPU
        free_mib = [int(x.strip()) for x in result.stdout.strip().splitlines()]
    except Exception:
        return 1

    n_gpus = len(free_mib)
    if n_gpus == 0:
        return 1

    def _workers_for_gpu(free_gb: float) -> int:
        # Each worker uses ~5.5 GB at runtime (weights + CUDA context + activations).
        # Tiers leave headroom for synthesis allocations:
        #   ≥ 17 GB → 3 workers (16.5 GB + 0.5 GB buffer)
        #   ≥ 12 GB → 2 workers (11.0 GB + 1.0 GB buffer)
        #   else    → 1 worker
        if   free_gb >= 17: return 3
        elif free_gb >= 12: return 2
        else:               return 1

    if n_gpus > 1:
        # Sum workers across all requested devices
        total = sum(
            _workers_for_gpu(free_mib[d] / 1024)
            for d in device_ids if d < len(free_mib)
        )
        return max(1, total)

    # Single GPU
    free_gb = free_mib[0] / 1024
    workers = _workers_for_gpu(free_gb)
    log.info(f"GPU 0: {free_gb:.1f} GB free → {workers} TTS worker(s) (auto)")
    return workers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dub a video from a pre-translated diarized SRT using Qwen TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Full pipeline:
  1. uv run --project ../nemo python nemo.py video.mp4 --language de --diarize
  2. python translate.py   (Gemma via Ollama → translated SRT)
  3. uv run python dub.py video.mp4 translated.srt --language fr

Examples:
  uv run python dub.py video.mp4 video.nemo.de.diarize_fr.srt --language fr
  uv run python dub.py video.mp4 video.nemo.de.diarize_fr.srt --language fr --qwen-mode clone
  uv run python dub.py video.mp4 video.nemo.de.diarize_fr.srt --language fr --no-demucs
        """,
    )
    parser.add_argument("video", nargs="?", default=None,
                        help="Input video file (auto-discovered in --search-dir if omitted)")
    parser.add_argument("srt",   nargs="?", default=None,
                        help="Pre-translated diarized SRT (auto-discovered if omitted)")
    parser.add_argument("--language",   default="fr",
                        help="Target language to pass to Qwen TTS (default: fr)")
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
    parser.add_argument("--max-speed",  type=float, default=1.35,
                        help="Max TTS speed-up before capping (default: 1.35)")
    parser.add_argument("--min-speed",  type=float, default=0.65,
                        help="Min TTS slow-down for short clips (default: 0.65). "
                             "Clips shorter than this ratio are slowed to 0.65x speed "
                             "to fill the slot, matching academic lecture pace.")
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
    args = parser.parse_args()

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
    temp_dir   = work_dir / "temp"
    output_dir = work_dir / "output"
    cast_dir   = work_dir / "cast_samples"

    for d in (work_dir, temp_dir, output_dir):
        d.mkdir(parents=True, exist_ok=True)

    script_dir  = Path(__file__).resolve().parent
    qwen_python = _qwen_python(qwen_dir)
    qwen_worker = _qwen_worker(script_dir)

    log.info("=" * 60)
    log.info(f"Video        : {video_path.name}")
    log.info(f"SRT          : {srt_path.name}")
    log.info(f"Language     : {args.language}")
    log.info(f"Qwen mode    : {args.qwen_mode}")
    log.info(f"Demucs       : {'disabled' if args.no_demucs else 'enabled'}")
    log.info(f"Qwen python  : {qwen_python}")
    log.info("=" * 60)

    # ── 1. Parse SRT ─────────────────────────────────────────────────────────
    segments = parse_srt(srt_path)
    if not segments:
        log.error("No segments parsed — make sure this is a diarized+translated SRT")
        return 1
    if args.merge_gap > 0:
        segments = merge_segments(segments, gap_sec=args.merge_gap,
                                  max_dur=args.merge_max_dur)

    # Compute SRT duration early — used for trimming audio AND final video
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
        vocals, background = separate_audio(video_path, temp_dir, trim_sec=srt_end)
        audio_for_refs = vocals

    # ── 3. Clone refs ─────────────────────────────────────────────────────────
    clone_refs: Dict[str, Path] = {}
    if args.qwen_mode == "clone":
        clone_refs = extract_clone_refs(segments, audio_for_refs, cast_dir)
        if not clone_refs:
            log.warning("⚠️  No clone refs extracted — falling back to custom mode")
            args.qwen_mode = "custom"

    # ── 4. Voice map ─────────────────────────────────────────────────────────
    # Always build as fallback, but only log if we're actually in custom mode
    voice_map = build_voice_map(segments)
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
    # Sizing guide:
    #   Each worker holds one 1.7B bfloat16 model (~3.4 GB).
    #   6 GB GPU  → 1 worker
    #   16 GB GPU → 4 workers
    #   48 GB GPU → 14 workers
    #   Multi-GPU → set --tts-devices 0,1,2 --tts-workers 6 (2 per GPU)
    qwen_language = _qwen_lang(args.language)
    log.info(f"Qwen language: '{args.language}' → '{qwen_language}'")
    checkpoint_path = work_dir / "checkpoint.json"
    final_files: List[Tuple[Path, float, float]] = _load_checkpoint(checkpoint_path)
    done_indices = {int(Path(c).stem.split("_")[1]) for c, _, _ in final_files}

    todo = [seg for seg in segments if seg["index"] not in done_indices]
    log.info(f"🗣️  Synthesising {len(segments)} segments "
             f"({len(done_indices)} cached, {len(todo)} remaining)")

    # Pre-compute available window per segment: slot + silence gap to next
    # speaker.  TTS is allowed to overflow into that gap (no one is talking
    # there) rather than being hard-compressed into just the slot.
    # Cap: slot + 5 s max so we never stretch speech unnaturally over a long
    # pause.
    _available: Dict[int, float] = {}
    for idx, seg in enumerate(segments):
        slot = seg["end"] - seg["start"]
        if idx + 1 < len(segments):
            gap_end = segments[idx + 1]["start"]
        else:
            gap_end = seg["end"]
        _available[seg["index"]] = min(gap_end - seg["start"], slot + 5.0)

    device_ids = [int(x.strip()) for x in args.tts_devices.split(",")]
    n_workers  = (_auto_workers(device_ids)
                  if args.tts_workers == "auto"
                  else int(args.tts_workers))
    log.info(f"TTS workers: {n_workers}  |  devices: {device_ids}")

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

    fit_pool    = ThreadPoolExecutor(max_workers=max(4, n_workers * 2))
    fit_futures: List = []

    def _do_fit(raw_out: Path, available_dur: float, start: float, end: float) -> None:
        slot = max(0.1, end - start)
        raw_dur = _audio_duration(raw_out)
        # Use overflow window only when TTS is too long for the original slot.
        # Short/fitting clips use the slot so they aren't slowed unnecessarily.
        target_dur = available_dur if raw_dur > slot * args.max_speed else slot
        fitted = speed_fit(raw_out, target_dur, max_speed=args.max_speed, min_speed=args.min_speed)
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
        clone_w: Optional[PersistentTTSWorker] = None
        custom_w: Optional[PersistentTTSWorker] = None
        clone_broken_local = (args.qwen_mode != "clone")
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
                target_dur = max(0.1, _available.get(i, end - start))
                raw_out    = temp_dir / f"seg_{i:04d}.wav"
                ok         = False

                if raw_out.exists() and raw_out.stat().st_size > 500:
                    ok = True
                else:
                    if not clone_broken_local:
                        ref = clone_refs.get(spk)
                        if ref and ref.exists():
                            log.info(f"   [{i:04d}] 🎙️  clone ({spk})")
                            if clone_w is None:
                                clone_w = PersistentTTSWorker(
                                    "clone", qwen_python, qwen_worker, device_id=device_id)
                            ok = clone_w.generate_clone(text, ref, qwen_language, raw_out)
                            if not ok:
                                log.warning(f"   [{i:04d}] Clone failed — falling back to custom")
                                clone_broken_local = True
                                clone_w.close()   # free VRAM before loading custom
                                clone_w = None
                        else:
                            log.warning(f"   [{i:04d}] No clone ref for '{spk}'")

                    if not ok:
                        voice = voice_map.get(spk, QWEN_FEMALE_VOICES[0])
                        log.info(f"   [{i:04d}] 🔊 custom voice: {voice}")
                        if custom_w is None:
                            custom_w = PersistentTTSWorker(
                                "custom", qwen_python, qwen_worker, device_id=device_id)
                        ok = custom_w.generate_custom(text, voice, qwen_language, raw_out)
                        if not ok:
                            log.error(f"   [{i:04d}] Custom TTS also failed — skipping")

                if ok and raw_out.exists():
                    fut = fit_pool.submit(_do_fit, raw_out, target_dur, start, end)
                    with fit_lock:
                        fit_futures.append(fut)

                pbar.update(1)

        except Exception as exc:
            log.error(f"TTS worker thread failed: {exc}")
            with worker_exc_lock:
                worker_exceptions.append(exc)
        finally:
            if clone_w:  clone_w.close()
            if custom_w: custom_w.close()

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

    pbar.close()

    # Wait for all speed_fit jobs
    for fut in fit_futures:
        try:
            fut.result()
        except Exception as exc:
            log.error(f"speed_fit error (segment skipped on resume): {exc}")
    fit_pool.shutdown(wait=True)

    if worker_exceptions and not final_files:
        raise worker_exceptions[0]

    # Sort by start time — parallel workers may have appended out of order
    final_files.sort(key=lambda x: x[1])

    if not final_files:
        log.error("No audio was generated. Check Qwen TTS errors above.")
        return 1

    # ── 6. Stitch + mix ──────────────────────────────────────────────────────
    # srt_end already computed above (reused for audio trim + video trim)
    log.info("🎬 Stitching and mixing…")
    final, actual_positions = stitch_and_mix(
        final_files, video_path, output_dir, temp_dir,
        background=background,   # None when --no-demucs
        trim_to=srt_end,
    )

    # ── 7. Write dubbed SRT with actual audio timestamps ─────────────────
    dub_srt_path = output_dir / srt_path.name.replace(".srt", "_dub.srt")
    write_dub_srt(dub_srt_path, actual_positions, segments)

    log.info("=" * 60)
    log.info(f"✅ Done!  →  {final}")
    log.info(f"📝 Dub SRT →  {dub_srt_path}")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
