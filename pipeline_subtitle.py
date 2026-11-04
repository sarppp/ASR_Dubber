#!/usr/bin/env python3
"""Subtitle-first dubbing pipeline (default).

Subtitles come from the local subtitle pipeline — Parakeet Ultra ASR, cue
packing, whole-sentence Tower translation, deterministic split back onto the cue
boundaries, plus the combined multi-language SRT/ASS — and the dubbing step then
reads the target-language SRT exactly as the older pipeline did.

    video ──► Ultra ──► cues ──► sentences ──► Tower ──► <lang>.srt
                                                     └─► dub.py (target lang)

The old NeMo + Canary/Gemma flow still exists as ``--pipeline legacy-dubbing``.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from pipeline_utils import (
    END_PRODUCT_DIR,
    NEMO_CODE_DIR,
    NEMO_DIR,
    QWEN_DIR,
    QWEN_PY,
    WHISPER_PY,
    _banner,
    _python,
    _run,
)
from pipeline_paths import (
    _derive_run_label,
    _detect_source_language,
    _finalize_outputs,
    _find_video,
    _validate_translated_srt,
)

SUBTITLE_PY = NEMO_CODE_DIR / ".venv-local-asr" / "bin" / "python"
SUBTITLE_ENTRY = NEMO_CODE_DIR / "batch_subtitles.py"
KNOWN = ("en", "de", "fr")


def _interpreter() -> Path:
    if SUBTITLE_PY.exists():
        return SUBTITLE_PY
    fallback = NEMO_CODE_DIR / ".venv" / "bin" / "python"
    if fallback.exists():
        print(f"⚠️  {SUBTITLE_PY} missing — falling back to {fallback}")
        return fallback
    return Path(sys.executable)


def _video_base(video: Path, trim: int) -> str:
    base = re.split(r"[._]nemo|__", video.stem)[0]
    return f"{base}_t{trim}" if trim else base


def _subtitle_languages(args, source: str, target: str) -> list[str]:
    if args.subtitle_languages:
        ordered = [lang for lang in args.subtitle_languages if lang in KNOWN]
        if not ordered:
            sys.exit(f"❌  --subtitle-languages must use {', '.join(KNOWN)}")
        if target in KNOWN and target not in ordered:
            ordered.append(target)  # dubbing needs the target subtitle
        return ordered
    ordered = []
    for lang in (source, target):
        if lang in KNOWN and lang not in ordered:
            ordered.append(lang)
    if not ordered:
        sys.exit(f"❌  Neither source ('{source}') nor target ('{target}') is one of "
                 f"{', '.join(KNOWN)}; pass --subtitle-languages explicitly")
    return ordered


def _stage_video(video: Path, workdir: Path) -> Path:
    """batch_subtitles works on directories, so give it a folder with this one video."""
    workdir.mkdir(parents=True, exist_ok=True)
    link = workdir / video.name
    if not link.exists():
        try:
            link.symlink_to(video.resolve())
        except OSError:
            shutil.copy2(video, link)
    return link


def _produce_subtitles(args, video: Path, workdir: Path, source: str,
                       target: str, base: str, nemo_dir: Path) -> dict[str, Path]:
    languages = _subtitle_languages(args, source, target)
    _stage_video(video, workdir)

    produced: dict[str, Path] = {}
    missing = [lang for lang in languages
               if not (workdir / f"{video.stem}.nemo.{lang}.srt").exists()]
    multi_missing = not (workdir / f"{video.stem}.multi.srt").exists()

    if missing or multi_missing:
        cmd = _python(_interpreter(), NEMO_CODE_DIR) + [
            str(SUBTITLE_ENTRY),
            "--backend", "local",
            "--input-dir", str(workdir),
            "--source-language", source,
            "--languages", *languages,
        ]
        if args.asr_model:
            cmd += ["--asr-model", args.asr_model]
        if args.translator:
            cmd += ["--translator", args.translator]
        _banner(f"Step 1/3 — Subtitles ({' + '.join(languages)}) via Ultra + Tower")
        _run(cmd, cwd=str(NEMO_CODE_DIR), label="Subtitle generation")
    else:
        print(f"⏭️  Subtitles already generated for {video.name} ({', '.join(languages)})")

    # Move out of the staging dir so clean_subs/dub find them next to the video.
    for lang in languages:
        src = workdir / f"{video.stem}.nemo.{lang}.srt"
        if not src.exists():
            sys.exit(f"❌  Subtitle step produced no {lang} SRT for {video.name}")
        dest = nemo_dir / f"{base}.nemo.{lang}.srt"
        shutil.move(str(src), str(dest))
        produced[lang] = dest

    for suffix in (".multi.srt", ".multi.ass"):
        src = workdir / f"{video.stem}{suffix}"
        if src.exists():
            shutil.move(str(src), str(nemo_dir / f"{base}{suffix}"))
    return produced


def _dub(args, video: Path, srt: Path, base: str, target: str) -> Path:
    dub_workdir = QWEN_DIR / "output" / "dub" / base
    if args.skip_dub:
        print("⏭️  Skipping dub (--skip-dub)")
        return dub_workdir

    print(f"📄 Using SRT : {srt}")
    _validate_translated_srt(srt, target)
    dub_workdir.mkdir(parents=True, exist_ok=True)

    cmd = _python(QWEN_PY, QWEN_DIR) + [
        str(QWEN_DIR / "dub.py"),
        str(video),
        str(srt),
        "--language", target,
        "--tts-engine", args.tts_engine,
        "--supertonic-voice", args.supertonic_voice,
        "--qwen-mode", args.qwen_mode,
        "--workdir", str(dub_workdir),
    ]
    if args.no_demucs:
        cmd.append("--no-demucs")
    if args.max_speed is not None:
        cmd += ["--max-speed", str(args.max_speed)]
    if args.min_speed is not None:
        cmd += ["--min-speed", str(args.min_speed)]
    if args.merge_gap is not None:
        cmd += ["--merge-gap", str(args.merge_gap)]
    if args.merge_max_dur is not None:
        cmd += ["--merge-max-dur", str(args.merge_max_dur)]
    cmd += ["--tts-workers", str(args.tts_workers), "--tts-devices", args.tts_devices]

    _banner(f"Step 3/3 — Dubbing into {target} with {args.tts_engine}")
    _run(cmd, cwd=str(QWEN_DIR), label=f"Dubbing ({args.tts_engine})")
    return dub_workdir


def run(args) -> int:
    nemo_dir = Path(args.input_dir).resolve() if args.input_dir else NEMO_DIR
    end_product_dir = Path(args.output_dir).resolve() if args.output_dir else END_PRODUCT_DIR
    if args.input_file:
        input_file = Path(args.input_file).resolve()
        if not input_file.exists():
            sys.exit(f"❌  Input file not found: {input_file}")
        nemo_dir = input_file.parent
    end_product_dir.mkdir(parents=True, exist_ok=True)

    video = Path(args.input_file).resolve() if args.input_file else _find_video(
        target_lang=args.target_lang, nemo_dir=nemo_dir, end_product_dir=end_product_dir)
    if video is None:
        sys.exit(f"❌  No unprocessed video found in {nemo_dir} for target '{args.target_lang}'")

    base = _video_base(video, args.trim)
    source = args.language or _detect_source_language(video, whisper_model=args.whisper_model)
    if not source:
        sys.exit("❌  Language detection failed. Pass --language explicitly.")
    target = source if args.target_lang == "same" else args.target_lang

    print(f"\n🎬 {video.name}  (base '{base}')")
    print(f"🌐 Source: {source}  →  Target: {target}")
    print("🧩 Pipeline: subtitle-dubbing (Ultra + Tower)")

    workdir = nemo_dir / ".subtitle_dub" / base
    subtitles = _produce_subtitles(args, video, workdir, source, target, base, nemo_dir)
    target_srt = subtitles.get(target)
    if target_srt is None:
        sys.exit(f"❌  No {target} subtitle was produced for {video.name}")

    dub_workdir = _dub(args, video, target_srt, base, target)

    run_label = _derive_run_label(source, target, video=video,
                                  nemo_dir=nemo_dir, end_product_dir=end_product_dir)
    _finalize_outputs(run_label, dub_workdir=None if args.skip_dub else dub_workdir,
                      nemo_dir=nemo_dir)
    print(f"\n✅ Subtitle dubbing complete → {end_product_dir / run_label}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point; `run_pipeline.py` calls run() with its own args."""
    from run_pipeline import build_parser

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.target_lang is None and args.run_mode != "transcribe":
        parser.error("--target-lang is required")
    if args.run_mode == "transcribe":
        args.target_lang = args.target_lang or "same"
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
