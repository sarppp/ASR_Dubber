#!/usr/bin/env python3
"""
run_pipeline.py — Master runner for the ASR dubbing pipeline
=============================================================
Place in ASR/ root. Runs all three steps end-to-end.

Usage:
# Basic — auto-detects source lang, processes next unprocessed video
uv run python run_pipeline.py --target-lang fr

# With trim (first 30s only)
uv run python run_pipeline.py --target-lang fr --trim 30

# Custom input/output folders
uv run python run_pipeline.py --target-lang fr --input-dir /path/to/videos --output-dir /path/to/results

# Force source language (skip Whisper detection)
uv run python run_pipeline.py --target-lang fr --language de

# Partial runs
uv run python run_pipeline.py --target-lang fr --run-mode transcribe   # Step 1 only
uv run python run_pipeline.py --target-lang fr --run-mode translate    # Steps 1-2 only
uv run python run_pipeline.py --target-lang fr --run-mode full         # All steps (default)

# Skip individual steps (when files already exist)
uv run python run_pipeline.py --target-lang fr --skip-nemo
uv run python run_pipeline.py --target-lang fr --skip-translate
uv run python run_pipeline.py --target-lang fr --skip-dub

# Voice mode (default is clone)
uv run python run_pipeline.py --target-lang fr --qwen-mode clone       # clones original speakers
uv run python run_pipeline.py --target-lang fr --qwen-mode custom      # uses fixed Qwen voices

# Faster — skip background music preservation
uv run python run_pipeline.py --target-lang fr --no-demucs

# Whisper model for language detection
uv run python run_pipeline.py --target-lang fr --whisper-model large-v3

### NEMO
# Precision
uv run python run_pipeline.py --target-lang fr --precision fp16   # older GPUs
uv run python run_pipeline.py --target-lang fr --precision fp32   # max accuracy
uv run python run_pipeline.py --target-lang fr --precision bf16   # default

# Force chunk size — auto-detected from VRAM (capped at 600s max)
# Lower if getting OOM, e.g. 120 = 2 min chunks
uv run python run_pipeline.py --target-lang fr --chunk-override 120

# VRAM tuning
uv run python run_pipeline.py --target-lang fr --reserve-gb 3.0
uv run python run_pipeline.py --target-lang fr --safety-factor 0.7

# Override model
uv run python run_pipeline.py --target-lang fr --nemo-model nvidia/parakeet-tdt-1.1b

# Combine freely
uv run python run_pipeline.py --target-lang fr --trim 30 --precision fp16 --reserve-gb 2.0

ASR/
├── run_pipeline.py          ← goes here
├── whisper/
│   ├── detect_language.py   ← goes here (new)
│   └── whisper_local.py     ← already here
├── nemo/
├── gemma-translate/
└── qwen3-tts/


"""

import argparse
import os
import re
import sys
from pathlib import Path

from pipeline_utils import (
    NEMO_DIR as _DEFAULT_NEMO_DIR,
    END_PRODUCT_DIR as _DEFAULT_END_PRODUCT_DIR,
    TRANSLATE_DIR,
    QWEN_DIR,
    WHISPER_DIR,
    NEMO_PY,
    QWEN_PY,
    WHISPER_PY,
    TRANSLATE_PY,
    _banner,
    _run,
    _python,
    _ollama_start,
    _ollama_stop,
)
from pipeline_paths import (
    _find_video,
    _find_srt_for_video,
    _detect_source_language,
    _find_translate_script,
    _derive_run_label,
    _finalize_outputs,
    _validate_translated_srt,
)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="ASR dubbing pipeline: NeMo → Gemma translate → Qwen TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--target-lang",    required=True,
                   help="Target dubbing language code, e.g. fr, en, es")
    p.add_argument("--language",       default=None,
                   help="Source language code (auto-detected from SRT if omitted)")
    p.add_argument("--trim",           type=int, default=0, metavar="SEC",
                   help="Process only first N seconds of video (default: full)")
    p.add_argument("--qwen-mode",      default="clone", choices=["clone", "custom"],
                   help="Voice mode: clone (default) or custom")
    p.add_argument("--no-demucs",      action="store_true",
                   help="Skip demucs — faster, no background music preserved")
    p.add_argument("--whisper-model",  default="medium",
                   choices=["tiny", "base", "small", "medium", "large-v3", "turbo"],
                   help="Whisper model for language detection (default: medium)")
    p.add_argument("--skip-nemo",      action="store_true",
                   help="Skip NeMo step (diarized SRT already exists)")
    p.add_argument("--skip-translate", action="store_true",
                   help="Skip translation (translated SRT already exists)")
    p.add_argument("--skip-dub",       action="store_true",
                   help="Skip dubbing (only transcribe + translate)")
    p.add_argument("--run-mode",       default="full",
                   choices=["transcribe", "translate", "full"],
                   help=("Convenience preset: transcribe (Step 1 only), "
                         "translate (Steps 1-2), or full (default)"))
    p.add_argument("--input-dir",      default=None, metavar="DIR",
                   help="Folder containing input video(s) (default: nemo/)")
    p.add_argument("--output-dir",     default=None, metavar="DIR",
                   help="Folder for final outputs / end_product (default: nemo/end_product/)")
    # ── NeMo tuning flags (passed through to nemo.py) ────────────────────────
    p.add_argument("--precision",      default="bf16", choices=["fp32", "fp16", "bf16"],
                   help="ASR precision (default: bf16 — use fp16 on older GPUs, fp32 for max accuracy)")
    p.add_argument("--nemo-model",     default=None, metavar="MODEL",
                   help="Override NeMo model name (default: auto-selected by language)")
    p.add_argument("--chunk-override", default=None, type=int, metavar="SEC",
                   help="Force NeMo audio chunk size in seconds (default: auto from VRAM)")
    p.add_argument("--reserve-gb",     default=None, type=float, metavar="GB",
                   help="VRAM reserve for NeMo chunk estimation (default: 1.5)")
    p.add_argument("--safety-factor",  default=None, type=float, metavar="F",
                   help="VRAM safety multiplier for NeMo chunking (default: 0.85)")
    args = p.parse_args()

    # Convenience presets for common partial runs
    if args.run_mode == "transcribe":
        args.skip_translate = True
        args.skip_dub = True
    elif args.run_mode == "translate":
        args.skip_dub = True

    # ── Apply input/output dir overrides (local vars, no global mutation) ─────
    nemo_dir = _DEFAULT_NEMO_DIR
    end_product_dir = _DEFAULT_END_PRODUCT_DIR
    if args.input_dir:
        nemo_dir = Path(args.input_dir).resolve()
        print(f"📂 Input dir  : {nemo_dir}")
    if args.output_dir:
        end_product_dir = Path(args.output_dir).resolve()
        print(f"📂 Output dir : {end_product_dir}")
    end_product_dir.mkdir(parents=True, exist_ok=True)

    # ── Validate dirs ─────────────────────────────────────────────────────────
    for name, d in [("nemo", nemo_dir), ("qwen3-tts", QWEN_DIR)]:
        if not d.exists():
            print(f"❌  {name}/ not found at {d}"); sys.exit(1)

    # ── Step 0: Pick video + detect source language ──────────────────────────
    source_lang = args.language

    # Pin video FIRST — every skip-check anchors off the same file.
    # _find_video skips videos that already have a run-dir for this target lang.
    video = _find_video(target_lang=args.target_lang,
                        nemo_dir=nemo_dir, end_product_dir=end_product_dir)
    if not video:
        print(f"❌  No unprocessed video found in {nemo_dir} for target '{args.target_lang}'")
        sys.exit(1)

    # Stable base = stem up to the first '.nemo' or '__' marker
    video_base = re.split(r"[._]nemo|__", video.stem)[0]

    # Append trim suffix so all generated files are unique per trim length
    if args.trim:
        video_base = f"{video_base}_t{args.trim}"

    print(f"🎬 Selected video : {video.name}  (base: '{video_base}')", flush=True)

    # Try to infer source lang from an existing diarized SRT for THIS video only
    if not source_lang:
        srt = _find_srt_for_video(video_base, "*.nemo.*.diarize.srt",
                                   nemo_dir=nemo_dir, end_product_dir=end_product_dir)
        if srt:
            m = re.search(r"\.nemo\.([a-z]{2,3})\.diarize\.srt$", srt.name)
            if m:
                source_lang = m.group(1)
                print(f"⏭️  Source language from existing SRT: '{source_lang}' ({srt.name})")

    # Fall back to Whisper detection on the chosen video
    if not source_lang:
        source_lang = _detect_source_language(video, whisper_model=args.whisper_model)
        if not source_lang:
            print("❌  Language detection failed. Pass --language explicitly.")
            sys.exit(1)

    # ── Step 1: NeMo ──────────────────────────────────────────────────────────
    existing_diarize_srt = _find_srt_for_video(
        video_base, f"*.nemo.{source_lang}.diarize.srt",
        nemo_dir=nemo_dir, end_product_dir=end_product_dir,
    )

    if args.skip_nemo or existing_diarize_srt:
        if existing_diarize_srt:
            print(f"⏭️  Skipping NeMo — SRT already exists: {existing_diarize_srt.name}")
        else:
            print("⏭️  Skipping NeMo (--skip-nemo)")
    else:
        nemo_cmd = _python(NEMO_PY, nemo_dir) + [
            "nemo.py", str(video),
            "--language", source_lang, "--diarize",
            "--precision", args.precision,
        ]
        if args.trim:
            nemo_cmd += ["--trim", str(args.trim)]
        if args.nemo_model:
            nemo_cmd += ["--nemo-model", args.nemo_model]
        if args.chunk_override:
            nemo_cmd += ["--chunk-override", str(args.chunk_override)]
        if args.reserve_gb:
            nemo_cmd += ["--reserve-gb", str(args.reserve_gb)]
        if args.safety_factor:
            nemo_cmd += ["--safety-factor", str(args.safety_factor)]
        _run(nemo_cmd, cwd=nemo_dir, label="Step 1/3 — NeMo transcription + diarization")

        # If trimmed, rename NeMo's output SRT to include _t{N} suffix
        if args.trim:
            original_srt = nemo_dir / f"{video.stem}.nemo.{source_lang}.diarize.srt"
            trimmed_srt  = nemo_dir / f"{video_base}.nemo.{source_lang}.diarize.srt"
            if original_srt.exists() and original_srt != trimmed_srt:
                original_srt.rename(trimmed_srt)
                print(f"📝 Renamed SRT: {original_srt.name} → {trimmed_srt.name}", flush=True)

    print(f"\n🌐 Source: {source_lang}  →  Target: {args.target_lang}")

    # ── Step 2: Translate (Gemma via Ollama) ──────────────────────────────────
    existing_translated_srt = _find_srt_for_video(
        video_base, f"*.diarize_{args.target_lang}.srt",
        nemo_dir=nemo_dir, end_product_dir=end_product_dir,
    )

    ollama_proc = None
    if args.skip_translate or existing_translated_srt:
        if existing_translated_srt:
            print(f"⏭️  Skipping translation — SRT already exists: {existing_translated_srt.name}")
        else:
            print("⏭️  Skipping translation (--skip-translate)")
    elif not args.skip_translate:
        translate_script = _find_translate_script()
        # Use translate-gemma venv if it exists, else fall back to system python
        translate_py = str(TRANSLATE_PY) if TRANSLATE_PY.exists() else sys.executable
        if not TRANSLATE_PY.exists():
            print(f"⚠️  No venv at {TRANSLATE_PY}, using system python (pysrt may be missing)")

        ollama_proc = _ollama_start()
        try:
            env = os.environ.copy()
            env["TARGET_LANG_CODE"] = args.target_lang
            env["SOURCE_LANG_CODE"] = source_lang

            _banner(f"Step 2/3 — Translation ({source_lang} → {args.target_lang}) via Gemma")
            print(f"   cwd : {TRANSLATE_DIR}")
            print(f"   cmd : {translate_py} {translate_script.name}\n", flush=True)
            import subprocess
            result = subprocess.run(
                [translate_py, str(translate_script)],
                cwd=str(TRANSLATE_DIR),
                env=env,
            )
            if result.returncode != 0:
                print(f"\n❌  Translation failed (exit {result.returncode})")
                sys.exit(result.returncode)
            print("\n✅  Translation done", flush=True)
        finally:
            _ollama_stop(ollama_proc)

    # ── Step 3: Dub ───────────────────────────────────────────────────────────
    dub_workdir = QWEN_DIR / "output" / "dub" / video_base  # always defined for finalize
    if not args.skip_dub:
        # Find the translated SRT — checks nemo/ AND end_product/ (post-clean location)
        dub_srt = _find_srt_for_video(
            video_base, f"*.diarize_{args.target_lang}.srt",
            nemo_dir=nemo_dir, end_product_dir=end_product_dir,
        )
        if dub_srt is None:
            print(f"❌  No translated SRT found for '{video_base}' in {nemo_dir} or {end_product_dir}")
            sys.exit(1)
        print(f"📄 Using SRT : {dub_srt}")

        # Validate SRT has actual translated content — catch silent translation failures
        _validate_translated_srt(dub_srt, args.target_lang)

        # Per-video workdir (defined above, create it now)
        dub_workdir.mkdir(parents=True, exist_ok=True)

        dub_cmd = _python(QWEN_PY, QWEN_DIR) + [
            "dub.py",
            str(video),          # explicit video — no auto-discovery
            str(dub_srt),        # explicit SRT   — no auto-discovery
            "--language",  args.target_lang,
            "--qwen-mode", args.qwen_mode,
            "--workdir",   str(dub_workdir),
        ]
        if args.no_demucs:
            dub_cmd.append("--no-demucs")

        _run(dub_cmd, cwd=QWEN_DIR, label="Step 3/3 — Dubbing with Qwen TTS")
    else:
        print("⏭️  Skipping dub (--skip-dub)")

    run_label = _derive_run_label(source_lang, args.target_lang, video=video,
                                  nemo_dir=nemo_dir, end_product_dir=end_product_dir)
    _finalize_outputs(run_label, dub_workdir=dub_workdir if not args.skip_dub else None)

    summary_lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║                  ✅  Pipeline complete!                   ║",
    ]
    if not args.skip_dub:
        summary_lines.append("║  Dub : qwen3-tts/output/dub/output/final_dub.mp4       ║")
    summary_lines.append(f"║  End : {end_product_dir / run_label}                      ║")
    summary_lines.append("╚══════════════════════════════════════════════════════════╝")
    print("\n" + "\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
