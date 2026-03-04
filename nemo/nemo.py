"""
nemo.py — NeMo local ASR/translation runner

Models (--asr-model shortname):
  parakeet-v2   nvidia/parakeet-tdt-0.6b-v2   English only,  ~2 GB, word timestamps
  parakeet-v3   nvidia/parakeet-tdt-0.6b-v3   25 EU langs,   ~2 GB, word timestamps  ← default multi
  canary        nvidia/canary-1b-v2            EN/DE/FR/ES,   ~5 GB, + AST translate
  canary-qwen   nvidia/canary-qwen-2.5b        English + LLM, ~10 GB

Auto-selection:
  --language en  → parakeet-v2
  any other lang → parakeet-v3

Usage:
  python nemo.py video.mp4 --language de
  python nemo.py video.mp4 --language fr --asr-model parakeet-v3
  python nemo.py video.mp4 --language de --asr-model canary --translate   # de → en
  python nemo.py video.mp4 --language de --diarize
  python nemo.py --language en --precision fp16 --all
"""

import argparse
import logging
import os
from pathlib import Path

import torch

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-8s │ %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("nemo_local")

from nemo_audio import ASR_MODELS, MODEL_EN, MODEL_MULTI, MULTI_LANGS, VIDEO_EXT, _fmt_dur
from nemo_diarize import _run_with_model
from nemo_model import _load_model


# ── CLI ───────────────────────────────────────────────────────────────────────

def _select_model(language: str, asr_model_key: str | None, nemo_model: str | None) -> str:
    """Resolve the final model ID.

    Priority: --nemo-model > --asr-model > NEMO_MODEL_EN/NEMO_MODEL_MULTI env > auto.
    """
    if nemo_model:
        return ASR_MODELS.get(nemo_model, nemo_model)
    if asr_model_key:
        return ASR_MODELS[asr_model_key]
    # Per-language env var defaults (set in docker-compose for remote GPU)
    env_key = "NEMO_MODEL_EN" if language == "en" else "NEMO_MODEL_MULTI"
    env_model = os.environ.get(env_key)
    if env_model:
        return ASR_MODELS.get(env_model, env_model)
    return MODEL_MULTI if language in MULTI_LANGS else MODEL_EN

def main():
    p = argparse.ArgumentParser(description="NeMo ASR local GPU transcription.")
    p.add_argument("video", nargs="?", help="Video file (auto-detect if omitted)")
    p.add_argument("--all", action="store_true", help="Process all pending videos")
    p.add_argument("--language", default="en", help="Source language code, e.g. en/de/fr/es [default: en]")
    p.add_argument("--asr-model", default=None, choices=list(ASR_MODELS),
                   help=(f"ASR model shortname (default: auto by language). "
                         f"Options: {', '.join(f'{k} ({v.split('/')[1]})' for k, v in ASR_MODELS.items())}"))
    p.add_argument("--nemo-model", default=None, metavar="MODEL",
                   help="Full NeMo model ID override (e.g. nvidia/parakeet-tdt-0.6b-v3). "
                        "Takes precedence over --asr-model.")
    p.add_argument("--precision", default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--translate", action="store_true", help="Translate to English (canary models only)")
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

    model_name = MODEL_MULTI if args.translate else _select_model(args.language, args.asr_model, args.nemo_model)
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
