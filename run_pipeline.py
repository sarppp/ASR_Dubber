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

# Fill gaps in SRT (detects missing speech, transcribes with Whisper)
uv run python run_pipeline.py --target-lang fr --fill-gaps 2           # fill gaps >= 2s

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
import atexit
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from pipeline_utils import (
    NEMO_CODE_DIR,
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
    _stream_proc,
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


# ── GPU auto-detection ────────────────────────────────────────────────────────

def _auto_tts_workers() -> tuple:
    """Detect available GPUs and decide how many TTS workers to run.

    Returns (n_workers, devices_str).

    On a **single GPU** more workers ≠ more throughput — they share the same
    memory bus, so the autoregressive decoder cannot run faster.  Extra workers
    only help overlap CPU work (ffmpeg speed_fit) with GPU inference.
    2-3 workers saturate that benefit; beyond that, startup cost dominates.

    Rules (single GPU):
      - < 16 GB free → 1 worker  (model needs ~6GB + inference buffers)
      - 16-24 GB     → 2 workers
      - ≥ 24 GB      → 3 workers

    Multi-GPU: 1 worker per GPU (each gets dedicated bandwidth).
    """
    import subprocess as _sp
    try:
        out = _sp.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stderr=_sp.DEVNULL, text=True,
        ).strip()
        free_mbs = [int(x.strip()) for x in out.splitlines() if x.strip()]
    except Exception:
        return 1, "0"

    n_gpus = len(free_mbs)
    if n_gpus == 0:
        return 1, "0"

    def _workers_for_gpu(free_gb: float) -> int:
        # Each worker needs ~6GB for model weights + headroom for inference buffers
        if   free_gb >= 24: return 3  # 3 * 6GB = 18GB + 6GB headroom
        elif free_gb >= 16: return 2  # 2 * 6GB = 12GB + 4GB headroom
        else:               return 1

    if n_gpus > 1:
        # Apply per-GPU tiers and sum — e.g. 4× 16 GB → 2 per GPU → 8 total
        devices = list(range(n_gpus))
        workers = sum(_workers_for_gpu(free_mbs[d] / 1024) for d in devices)
        print(f"🖥️  {n_gpus} GPUs → {workers} TTS worker(s) total")
        return max(1, workers), ",".join(str(i) for i in devices)

    # Single GPU
    free_gb = free_mbs[0] / 1024
    workers = _workers_for_gpu(free_gb)
    print(f"🖥️  GPU 0: {free_gb:.1f} GB free → {workers} TTS worker(s)")
    return workers, "0"


# ── Timing summary ────────────────────────────────────────────────────────────

def _append_timing_summary(run_dir: Path, pipeline_log: Path) -> None:
    import json
    lines = []
    total_sec = 0.0

    for f in run_dir.glob("*_timing_transcribe.json"):
        d = json.loads(f.read_text())
        t = d.get("total_sec", 0); total_sec += t
        audio = d.get("audio_dur_sec", 0)
        lines.append(f"  Transcribe : {t/60:.1f} min  "
                     f"(audio: {audio/60:.1f} min, RTF: {d.get('rtf','?')}, segments: {d.get('segments','?')})")
        break

    for f in run_dir.glob("*_timing_translate.json"):
        d = json.loads(f.read_text())
        t = d.get("total_sec", 0); total_sec += t
        lines.append(f"  Translate  : {t/60:.1f} min  "
                     f"({d.get('lines','?')} lines, {d.get('lines_per_sec',0):.2f} lines/s)")
        break

    for f in run_dir.glob("*_timing_dub.json"):
        d = json.loads(f.read_text())
        t = d.get("total_sec", 0); total_sec += t
        lines.append(f"  Dub        : {d.get('total', f'{t/60:.1f} min')}  "
                     f"(TTS: {d.get('tts','?')}, stitch: {d.get('stitch','?')}, segments: {d.get('segments','?')})")
        break

    if not lines:
        return

    block = (
        f"\n{'='*60}\n"
        f"PIPELINE TIMING SUMMARY\n"
        f"{'='*60}\n"
        + "\n".join(lines)
        + f"\n  {'─'*40}\n"
        + f"  Total      : {total_sec/60:.1f} min\n"
    )
    with open(pipeline_log, "a", encoding="utf-8") as pf:
        pf.write(block)
    print(block, flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="ASR dubbing pipeline: NeMo → Gemma translate → Qwen TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--target-lang",    default=None,
                   help="Target dubbing language code, e.g. fr, en, es (not required for --run-mode transcribe)")
    p.add_argument("--language",       default=None,
                   help="Source language code (auto-detected from SRT if omitted)")
    p.add_argument("--trim",           type=int, default=0, metavar="SEC",
                   help="Process only first N seconds of video (default: full)")
    p.add_argument("--qwen-mode",      default="clone", choices=["clone", "custom"],
                   help="Voice mode: clone (default) or custom")
    p.add_argument("--tts-engine",     default="qwen", choices=["qwen", "cosyvoice"],
                   help="TTS backend for dubbing: qwen (Qwen3-TTS, default) or "
                        "cosyvoice (Fun-CosyVoice3-0.5B). cosyvoice needs "
                        "`uv sync --project cosyvoice-tts` + submodule init.")
    p.add_argument("--no-demucs",      action="store_true",
                   help="Skip demucs — faster, no background music preserved")
    p.add_argument("--max-speed",      type=float, default=1.35, metavar="SPEED",
                   help="Max TTS speed-up before capping (default: 1.35)")
    p.add_argument("--min-speed",      type=float, default=0.65, metavar="SPEED",
                   help="Min TTS slow-down for short clips (default: 0.65)")
    p.add_argument("--merge-gap",      type=float, default=1.0, metavar="SEC",
                   help="Merge consecutive same-speaker segments with gap ≤ N s for more natural TTS (default: 1.0, set 0 to disable)")
    p.add_argument("--merge-max-dur",  type=float, default=10.0, metavar="SEC",
                   help="Max duration for a merged segment in seconds (default: 10)")
    p.add_argument("--tts-workers",    type=int,   default=1,
                   help="Parallel TTS workers (default: 1, auto-detected if not set)")
    p.add_argument("--tts-devices",    default="0",
                   help="GPU IDs for TTS workers, e.g. '0,1,2' (default: auto)")
    p.add_argument("--whisper-model",  default="medium",
                   choices=["tiny", "base", "small", "medium", "large-v3", "turbo"],
                   help="Whisper model for language detection (default: medium)")
    p.add_argument("--skip-nemo",      action="store_true",
                   help="Skip NeMo step (diarized SRT already exists)")
    p.add_argument("--no-diarize",    action="store_true",
                   help="Skip speaker diarization (transcribe only, no speaker labels)")
    p.add_argument("--fill-gaps",      type=float, default=2.0, metavar="SEC",
                   help="Fill gaps >= N seconds in SRT using Whisper (default: 2.0). "
                        "Detects missing speech and transcribes with Whisper, attributing speakers from context. "
                        "Set to 0 to disable.")
    p.add_argument("--fill-gaps-model", default="base",
                   choices=["tiny", "base", "small", "medium", "large-v3", "turbo"],
                   help="Whisper model for gap filling (default: base, faster but less accurate)")
    p.add_argument("--translate-model", default="translategemma:4b",
                   help="Ollama model for translation (default: translategemma:4b). "
                        "Shorthand: '4b' or '12b' expands to translategemma:4b/12b.")
    p.add_argument("--skip-translate", action="store_true",
                   help="Skip translation (translated SRT already exists)")
    p.add_argument("--skip-dub",       action="store_true",
                   help="Skip dubbing (only transcribe + translate)")
    p.add_argument("--run-mode",       default="full",
                   choices=["transcribe", "translate", "full"],
                   help=("Convenience preset: transcribe (Step 1 only), "
                         "translate (Steps 1-2), or full (default)"))
    p.add_argument("--input-file",     default=None, metavar="FILE",
                   help="Specific input file (overrides --input-dir)")
    p.add_argument("--input-dir",      default=None, metavar="DIR",
                   help="Folder containing input video(s) (default: nemo/)")
    p.add_argument("--output-dir",     default=None, metavar="DIR",
                   help="Folder for final outputs / end_product (default: nemo/end_product/)")
    # ── NeMo tuning flags (passed through to nemo.py) ────────────────────────
    p.add_argument("--precision",      default="bf16", choices=["fp32", "fp16", "bf16"],
                   help="ASR precision (default: bf16 — use fp16 on older GPUs, fp32 for max accuracy)")
    p.add_argument("--nemo-model",     default=None, metavar="MODEL",
                   help=("NeMo model shortname or full ID. Shortnames: parakeet-v2/v3, "
                         "canary, qwen3-asr, qwen3-asr-s. "
                         "Or a full HF ID like nvidia/parakeet-tdt-0.6b-v3."))
    p.add_argument("--chunk-override", default=None, type=int, metavar="SEC",
                   help="Force NeMo audio chunk size in seconds (default: auto from VRAM)")
    p.add_argument("--reserve-gb",     default=None, type=float, metavar="GB",
                   help="VRAM reserve for NeMo chunk estimation (default: 1.5)")
    p.add_argument("--safety-factor",  default=None, type=float, metavar="F",
                   help="VRAM safety multiplier for NeMo chunking (default: 0.85)")
    args = p.parse_args()

    # Expand translate model shorthands: '4b' → 'translategemma:4b', etc.
    if ":" not in args.translate_model:
        args.translate_model = f"translategemma:{args.translate_model}"

    # Convenience presets for common partial runs
    if args.run_mode == "transcribe":
        args.skip_translate = True
        args.skip_dub = True
    elif args.run_mode == "translate":
        args.skip_dub = True

    if args.target_lang is None:
        if args.run_mode != "transcribe":
            p.error("--target-lang is required unless --run-mode transcribe")
        # For transcribe-only runs, target == source (resolved later once source is known)
        args.target_lang = args.language or "same"

    # ── Apply input/output dir overrides (local vars, no global mutation) ─────
    nemo_dir = _DEFAULT_NEMO_DIR
    end_product_dir = _DEFAULT_END_PRODUCT_DIR
    if args.input_file:
        # Use specific file - derive directory from file location
        input_file = Path(args.input_file).resolve()
        if not input_file.exists():
            print(f"❌  Input file not found: {input_file}")
            sys.exit(1)
        nemo_dir = input_file.parent
        print(f"📂 Input file : {input_file}")
    elif args.input_dir:
        nemo_dir = Path(args.input_dir).resolve()
        print(f"📂 Input dir  : {nemo_dir}")
    if args.output_dir:
        end_product_dir = Path(args.output_dir).resolve()
        print(f"📂 Output dir : {end_product_dir}")
    end_product_dir.mkdir(parents=True, exist_ok=True)

    # Always clean up chunk WAVs — even on cancel or failure
    def _cleanup_chunks_on_exit():
        for c in nemo_dir.glob("_chunk_*.wav"):
            try:
                c.unlink()
            except OSError:
                pass
    atexit.register(_cleanup_chunks_on_exit)

    # ── Validate dirs ─────────────────────────────────────────────────────────
    for name, d in [("nemo", nemo_dir), ("qwen3-tts", QWEN_DIR)]:
        if not d.exists():
            print(f"❌  {name}/ not found at {d}"); sys.exit(1)

    # ── Step 0: Pick video + detect source language ──────────────────────────
    source_lang = args.language

    # Pin video FIRST — every skip-check anchors off the same file.
    # _find_video skips videos that already have a run-dir for this target lang.
    if args.input_file:
        video = input_file
    else:
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

    if video.suffix.lower() == ".wav":
        print(f"🎵 WAV input detected — dubbing step will be skipped automatically", flush=True)
        args.skip_dub = True

    print(f"🎬 Selected video : {video.name}  (base: '{video_base}')", flush=True)

    # ── Per-run log files (written to nemo_dir, moved to run_dir at the end) ──
    log_transcribe = nemo_dir / f"{video_base}_1_transcribe.log"
    log_translate  = nemo_dir / f"{video_base}_2_translate.log"
    log_dub        = nemo_dir / f"{video_base}_3_dub.log"
    log_finalize   = nemo_dir / f"{video_base}_4_finalize.log"

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
            str(NEMO_CODE_DIR / "nemo.py"), str(video),
            "--language", source_lang,
            "--precision", args.precision,
        ]
        if not args.no_diarize:
            nemo_cmd.append("--diarize")
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
        _run(nemo_cmd, cwd=nemo_dir, label="Step 1/3 — NeMo transcription + diarization",
             log_file=log_transcribe)

        # If trimmed, rename NeMo's output SRT to include _t{N} suffix
        if args.trim:
            original_srt = nemo_dir / f"{video.stem}.nemo.{source_lang}.diarize.srt"
            trimmed_srt  = nemo_dir / f"{video_base}.nemo.{source_lang}.diarize.srt"
            if original_srt.exists() and original_srt != trimmed_srt:
                original_srt.rename(trimmed_srt)
                print(f"📝 Renamed SRT: {original_srt.name} → {trimmed_srt.name}", flush=True)

    # ── Step 1b: Fill gaps in SRT (optional) ─────────────────────────────────────
    if args.fill_gaps > 0:
        # Find the SRT to fill gaps in
        srt_to_fill = _find_srt_for_video(
            video_base, f"*.nemo.{source_lang}.diarize.srt",
            nemo_dir=nemo_dir, end_product_dir=end_product_dir,
        )
        if srt_to_fill:
            filled_srt = srt_to_fill.parent / srt_to_fill.name.replace(".diarize.srt", ".diarize_filled.srt")
            if filled_srt.exists():
                print(f"⏭️  Skipping gap fill — already exists: {filled_srt.name}")
            else:
                fill_cmd = [
                    str(NEMO_PY),
                    str(NEMO_CODE_DIR / "srt_fill_gaps.py"),
                    str(video), str(srt_to_fill), str(filled_srt),
                    "--min-gap", str(args.fill_gaps),
                    "--whisper-model", args.fill_gaps_model,
                ]
                _fill_env = {**os.environ, "WHISPER_PY": str(WHISPER_PY)}
                _run(fill_cmd, cwd=nemo_dir, label="Step 1b — Fill SRT gaps with Whisper", env=_fill_env)
                if filled_srt.exists():
                    shutil.copy2(str(filled_srt), str(srt_to_fill))
                    print(f"� Replaced SRT with gap-filled version: {srt_to_fill.name}", flush=True)
        else:
            print(f"⚠️  No SRT found for gap filling")

    if args.target_lang == "same":
        args.target_lang = source_lang

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
            env["INPUT_DIR"]        = str(nemo_dir)
            env["OUTPUT_DIR"]       = str(end_product_dir)

            env["TRANSLATE_MODEL"]  = args.translate_model

            _banner(f"Step 2/3 — Translation ({source_lang} → {args.target_lang}) via Gemma")
            print(f"   cwd : {nemo_dir}")
            print(f"   cmd : {translate_py} {translate_script}\n", flush=True)
            proc = subprocess.Popen(
                [translate_py, str(translate_script)],
                cwd=str(nemo_dir), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            rc = _stream_proc(proc, log_translate)
            if rc != 0:
                print(f"\n❌  Translation failed (exit {rc})")
                sys.exit(rc)
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
            str(QWEN_DIR / "dub.py"),
            str(video),          # explicit video — no auto-discovery
            str(dub_srt),        # explicit SRT   — no auto-discovery
            "--language",   args.target_lang,
            "--qwen-mode",  args.qwen_mode,
            "--tts-engine", args.tts_engine,
            "--workdir",    str(dub_workdir),
        ]
        if args.no_demucs:
            dub_cmd.append("--no-demucs")
        if args.max_speed is not None:
            dub_cmd.extend(["--max-speed", str(args.max_speed)])
        if args.min_speed is not None:
            dub_cmd.extend(["--min-speed", str(args.min_speed)])
        if args.merge_gap is not None:
            dub_cmd.extend(["--merge-gap", str(args.merge_gap)])
        if args.merge_max_dur is not None:
            dub_cmd.extend(["--merge-max-dur", str(args.merge_max_dur)])
        # Auto-detect GPU count and set workers/devices if user didn't override
        tts_workers = args.tts_workers
        tts_devices = args.tts_devices
        if tts_workers == 1 and tts_devices == "0":
            tts_workers, tts_devices = _auto_tts_workers()
        dub_cmd.extend(["--tts-workers", str(tts_workers),
                        "--tts-devices", tts_devices])

        _engine_label = {"qwen": "Qwen TTS", "cosyvoice": "CosyVoice3"}[args.tts_engine]
        _run(dub_cmd, cwd=QWEN_DIR, label=f"Step 3/3 — Dubbing with {_engine_label}",
             log_file=log_dub)
    else:
        print("⏭️  Skipping dub (--skip-dub)")

    run_label = _derive_run_label(source_lang, args.target_lang, video=video,
                                  nemo_dir=nemo_dir, end_product_dir=end_product_dir)
    _finalize_outputs(run_label, dub_workdir=dub_workdir if not args.skip_dub else None,
                      nemo_dir=nemo_dir, log_file=log_finalize)

    # ── Assemble logs into run_dir ────────────────────────────────────────────
    run_dir = end_product_dir / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    pipeline_log = run_dir / "pipeline.log"
    step_logs = [log_transcribe, log_translate, log_dub, log_finalize]
    with open(pipeline_log, "w", encoding="utf-8") as pf:
        for lf in step_logs:
            if not lf.exists():
                continue
            dest = run_dir / lf.name
            shutil.move(str(lf), str(dest))
            pf.write(f"\n{'='*60}\n{lf.name}\n{'='*60}\n")
            pf.write(dest.read_text(encoding="utf-8", errors="replace"))
    # ── Append timing summary to pipeline.log ────────────────────────────────
    _append_timing_summary(run_dir, pipeline_log)
    print(f"📋 Logs → {pipeline_log}", flush=True)

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
