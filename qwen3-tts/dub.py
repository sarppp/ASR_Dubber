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
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from dub_srt import QWEN_FEMALE_VOICES, _qwen_lang, parse_srt, build_voice_map
from dub_audio import (
    extract_audio,
    separate_audio,
    extract_clone_refs,
    _qwen_python,
    _qwen_worker,
    PersistentTTSWorker,
    speed_fit,
    stitch_and_mix,
    _save_checkpoint,
    _load_checkpoint,
)


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
    qwen_language = _qwen_lang(args.language)
    log.info(f"Qwen language: '{args.language}' → '{qwen_language}'")
    checkpoint_path = work_dir / "checkpoint.json"
    final_files: List[Tuple[Path, float, float]] = _load_checkpoint(checkpoint_path)
    done_indices = {int(Path(c).stem.split("_")[1]) for c, _, _ in final_files}
    clone_broken  = False
    custom_broken = False

    # Persistent workers — model loaded once, kept alive for all segments.
    # Workers are started lazily (on first use) to avoid loading a model that
    # may never be needed (e.g. clone-only run with no fallback needed).
    clone_worker:  Optional[PersistentTTSWorker] = None
    custom_worker: Optional[PersistentTTSWorker] = None

    log.info(f"🗣️  Synthesising {len(segments)} segments…")
    pbar = tqdm(segments, desc="TTS", unit="seg",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    try:
        for seg in pbar:
            i          = seg["index"]
            spk        = seg["speaker"]
            text       = seg["text"]
            start, end = seg["start"], seg["end"]
            target_dur = max(0.1, end - start)

            raw_out = temp_dir / f"seg_{i:04d}.wav"

            pbar.set_postfix_str(f"{spk} #{i}")

            if i in done_indices:
                continue
            if raw_out.exists() and raw_out.stat().st_size > 500:
                pass  # cached — tqdm shows progress
            else:
                ok = False

                if args.qwen_mode == "clone" and not clone_broken:
                    ref = clone_refs.get(spk)
                    if ref and ref.exists():
                        log.info(f"   [{i:04d}] 🎙️  clone ({spk})")
                        if clone_worker is None:
                            clone_worker = PersistentTTSWorker("clone", qwen_python, qwen_worker)
                        ok = clone_worker.generate_clone(text, ref, qwen_language, raw_out)
                        if not ok:
                            log.warning(f"   [{i:04d}] Clone failed — falling back to custom")
                            clone_broken = True
                            # Free VRAM before starting the custom worker
                            clone_worker.close()
                            clone_worker = None
                    else:
                        log.warning(f"   [{i:04d}] No clone ref for '{spk}' — falling back to custom")

                if not ok and not custom_broken:
                    voice = voice_map.get(spk, QWEN_FEMALE_VOICES[0])
                    log.info(f"   [{i:04d}] 🔊 custom voice: {voice}")
                    if custom_worker is None:
                        custom_worker = PersistentTTSWorker("custom", qwen_python, qwen_worker)
                    ok = custom_worker.generate_custom(text, voice, qwen_language, raw_out)
                    if not ok:
                        log.error(f"   [{i:04d}] Custom TTS also failed — skipping segment")
                        custom_broken = True
                        continue

                if not ok:
                    continue

            if not raw_out.exists():
                continue

            fitted = speed_fit(raw_out, target_dur, max_speed=args.max_speed)
            final_files.append((fitted, start, end))
            _save_checkpoint(checkpoint_path, final_files)

    finally:
        # Always shut down workers cleanly to release VRAM
        if clone_worker:
            clone_worker.close()
        if custom_worker:
            custom_worker.close()

    if not final_files:
        log.error("No audio was generated. Check Qwen TTS errors above.")
        return 1

    # ── 6. Stitch + mix ──────────────────────────────────────────────────────
    # srt_end already computed above (reused for audio trim + video trim)
    log.info("🎬 Stitching and mixing…")
    final = stitch_and_mix(
        final_files, video_path, output_dir, temp_dir,
        background=background,   # None when --no-demucs
        trim_to=srt_end,
    )

    log.info("=" * 60)
    log.info(f"✅ Done!  →  {final}")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
