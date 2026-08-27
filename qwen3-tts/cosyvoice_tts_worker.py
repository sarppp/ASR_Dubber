#!/usr/bin/env python3
"""
Persistent Fun-CosyVoice3 TTS worker — drop-in replacement for qwen_tts_worker.py.

Speaks the exact same stdin/stdout JSON-line protocol so dub.py / dub_audio.py
(PersistentTTSWorker + SharedTTSManager) can drive it unchanged:

  Startup:  prints "READY" on stdout once the model is loaded
  Request:  one JSON object per line on stdin
  Response: one JSON object per line on stdout

Request shapes:
  Clone:   {"text": "...", "ref_audio": "/ref.wav", "ref_text": "", "language": "french", "output": "/p.wav"}
  Custom:  {"text": "...", "voice": "ryan", "language": "french", "output": "/p.wav"}
  Quit:    {"quit": true}

Response shapes:
  {"ok": true}  |  {"ok": false, "error": "..."}

CosyVoice3 has no named speaker inventory, so "custom" mode falls back to a
default prompt wav (env COSY_FALLBACK_PROMPT, or asset/zero_shot_prompt.wav in
the repo).  In this pipeline dubbing runs in clone mode, so custom is only a
last-resort fallback.

Environment:
  COSYVOICE_REPO_DIR   path to a checkout of github.com/FunAudioLLM/CosyVoice
  COSYVOICE_MODEL_DIR  path to the Fun-CosyVoice3-0.5B weights directory
  COSY_FALLBACK_PROMPT optional wav used for "custom" mode (default: repo asset)
"""

import argparse
import json
import os
import sys
from pathlib import Path

_SYS_PROMPT = "You are a helpful assistant.<|endofprompt|>"


def _setup_paths() -> Path:
    repo = os.environ.get("COSYVOICE_REPO_DIR", "/opt/CosyVoice")
    repo_path = Path(repo)
    if not repo_path.exists():
        raise RuntimeError(f"COSYVOICE_REPO_DIR not found: {repo_path}")
    # CosyVoice expects to be run from its repo root with Matcha-TTS on the path
    os.chdir(repo_path)
    sys.path.insert(0, str(repo_path))
    matcha = repo_path / "third_party" / "Matcha-TTS"
    if matcha.exists():
        sys.path.insert(0, str(matcha))
    return repo_path


def _load_model(device: str):
    repo_path = _setup_paths()
    print("[cosy-worker] importing cosyvoice…", file=sys.stderr, flush=True)
    import torch  # noqa: F401

    model_dir = os.environ.get("COSYVOICE_MODEL_DIR")
    if not model_dir or not Path(model_dir).exists():
        raise RuntimeError(f"COSYVOICE_MODEL_DIR missing or not found: {model_dir!r}")

    try:
        from cosyvoice.cli.cosyvoice import AutoModel
        print(f"[cosy-worker] AutoModel.from {model_dir}", file=sys.stderr, flush=True)
        model = AutoModel(model_dir=model_dir)
    except Exception:
        # Older CosyVoice checkouts expose CosyVoice2 instead of AutoModel
        from cosyvoice.cli.cosyvoice import CosyVoice2
        print(f"[cosy-worker] CosyVoice2 from {model_dir}", file=sys.stderr, flush=True)
        model = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)

    from cosyvoice.utils.file_utils import load_wav
    model._load_wav = load_wav

    fallback = os.environ.get("COSY_FALLBACK_PROMPT") or str(
        repo_path / "asset" / "zero_shot_prompt.wav"
    )
    model._fallback_prompt = fallback if Path(fallback).exists() else None
    print(f"[cosy-worker] model loaded, sample_rate={model.sample_rate}", file=sys.stderr, flush=True)
    return model


def _synthesise(model, req: dict, mode: str) -> str | None:
    """Run one synthesis request. Returns an error string, or None on success."""
    import torch
    import torchaudio

    text = (req.get("text") or "").strip()
    output = req.get("output") or ""
    if not text or not output:
        return "missing 'text' or 'output' in request"

    ref_audio = req.get("ref_audio") or ""
    ref_text = (req.get("ref_text") or "").strip()

    if mode == "clone":
        if not ref_audio or not Path(ref_audio).exists():
            return f"ref_audio missing or not found: {ref_audio!r}"
        prompt_wav_path = ref_audio
    else:
        prompt_wav_path = ref_audio if (ref_audio and Path(ref_audio).exists()) else model._fallback_prompt
        if not prompt_wav_path:
            return "custom mode needs COSY_FALLBACK_PROMPT (no reference wav available)"

    try:
        prompt_16k = model._load_wav(prompt_wav_path, 16000)

        chunks = []
        if ref_text:
            # We know what the reference says → zero-shot keeps timbre + prosody best
            gen = model.inference_zero_shot(
                text, _SYS_PROMPT + ref_text, prompt_16k, stream=False
            )
        else:
            # No transcript of the reference → cross-lingual clone (EN ref → FR out)
            gen = model.inference_cross_lingual(
                _SYS_PROMPT + text, prompt_16k, stream=False
            )
        for out in gen:
            chunks.append(out["tts_speech"])

        if not chunks:
            return "model returned no audio"

        audio = torch.cat(chunks, dim=1)
        torchaudio.save(output, audio, model.sample_rate)
        if not Path(output).exists() or Path(output).stat().st_size < 100:
            return f"output WAV not written or empty: {output}"
        return None
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc(file=sys.stderr)
        return str(exc)


def _daemon(mode: str, device: str) -> None:
    try:
        model = _load_model(device)
    except Exception as exc:  # noqa: BLE001
        print(f"LOAD_ERROR: {exc}", flush=True)
        print(f"LOAD_ERROR: {exc}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    print("READY", flush=True)
    print(f"READY (mode={mode}, device={device})", file=sys.stderr, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"ok": False, "error": f"bad JSON: {e}"}), flush=True)
            continue

        if req.get("quit"):
            print(json.dumps({"ok": True}), flush=True)
            break

        err = _synthesise(model, req, mode)
        print(json.dumps({"ok": err is None, **({"error": err} if err else {})}), flush=True)


def _one_shot(mode: str, device: str, args: argparse.Namespace) -> int:
    model = _load_model(device)
    err = _synthesise(
        model,
        {
            "text": args.text,
            "output": args.output,
            "language": args.language,
            "voice": args.voice,
            "ref_audio": args.ref_audio,
            "ref_text": args.ref_text,
        },
        mode,
    )
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(f"OK: {args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fun-CosyVoice3 TTS worker")
    parser.add_argument("--mode", choices=["clone", "custom"], default="clone")
    parser.add_argument("--device", default=None)
    parser.add_argument("--text", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--voice", default="")
    parser.add_argument("--language", default="french")
    parser.add_argument("--ref-audio", default="")
    parser.add_argument("--ref-text", default="")
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[cosy-worker] device={device}, mode={args.mode}", file=sys.stderr, flush=True)

    if not sys.stdin.isatty():
        _daemon(args.mode, device)
        return 0

    if not args.text or not args.output:
        parser.error("--text and --output are required in one-shot mode")
    return _one_shot(args.mode, device, args)


if __name__ == "__main__":
    sys.exit(main())
