#!/usr/bin/env python3
"""
Persistent Qwen TTS worker — loads model once, serves all synthesis requests.

Daemon protocol (stdin/stdout JSON lines):
  Startup:  worker prints "READY" to stdout after model is loaded
  Request:  one JSON object per line written to stdin
  Response: one JSON object per line written to stdout

Request shapes:
  Custom:  {"text": "...", "voice": "Chelsie", "language": "French", "output": "/path.wav"}
  Clone:   {"text": "...", "ref_audio": "/ref.wav", "ref_text": "...", "language": "French", "output": "/path.wav"}
  Quit:    {"quit": true}

Response shapes:
  Success: {"ok": true}
  Failure: {"ok": false, "error": "..."}

All info/debug messages go to stderr so stdout stays clean for the protocol.
"""

import argparse
import json
import sys
from pathlib import Path


def _load_model(mode: str, device: str):
    import torch
    from qwen_tts import Qwen3TTSModel

    model_id = (
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        if mode == "clone"
        else "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    )
    print(f"Loading {model_id} on {device}…", file=sys.stderr, flush=True)
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=torch.bfloat16,
    )
    return model


def _synthesise(model, req: dict, mode: str) -> str | None:
    """Run one synthesis request. Returns error string or None on success."""
    import soundfile as sf
    import torch

    text     = req.get("text", "")
    output   = req.get("output", "")
    language = req.get("language", "English")

    if not text or not output:
        return "missing 'text' or 'output' in request"

    try:
        if mode == "clone":
            ref_audio = req.get("ref_audio", "")
            if not ref_audio or not Path(ref_audio).exists():
                return f"ref_audio missing or not found: {ref_audio!r}"
            ref_text = req.get("ref_text", "")
            if ref_text:
                wavs, sr = model.generate_voice_clone(
                    text=text, language=language,
                    ref_audio=ref_audio, ref_text=ref_text,
                )
            else:
                wavs, sr = model.generate_voice_clone(
                    text=text, language=language,
                    ref_audio=ref_audio, x_vector_only_mode=True,
                )
        else:
            voice = req.get("voice", "Chelsie")
            try:
                wavs, sr = model.generate_custom_voice(
                    text=text, language=language, speaker=voice, instruct="",
                )
            except AttributeError:
                wavs, sr = model.generate(text=text, voice=voice, language=language)

        if wavs is None or len(wavs) == 0:
            return "model returned no audio"

        audio = wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0]
        sf.write(output, audio, sr)
        if not Path(output).exists() or Path(output).stat().st_size < 100:
            return f"output WAV not written or empty: {output}"
        return None  # success

    except Exception as exc:
        return str(exc)


def _daemon(mode: str, device: str) -> None:
    """Load model once then serve all requests from stdin until quit or EOF."""
    model = _load_model(mode, device)
    # Signal readiness — PersistentTTSWorker blocks waiting for this line on stdout
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
        if err:
            print(json.dumps({"ok": False, "error": err}), flush=True)
        else:
            print(json.dumps({"ok": True}), flush=True)


def _one_shot(mode: str, device: str, args: argparse.Namespace) -> int:
    """Single synthesis — original CLI behaviour, kept for testing/debugging."""
    model = _load_model(mode, device)
    req = {
        "text":      args.text,
        "output":    args.output,
        "language":  args.language,
        "voice":     args.voice,
        "ref_audio": args.ref_audio,
        "ref_text":  args.ref_text,
    }
    err = _synthesise(model, req, mode)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(f"OK: {args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen TTS Worker")
    parser.add_argument("--mode",      choices=["clone", "custom"], default="custom")
    parser.add_argument("--device",    default=None,
                        help="Force device (cuda:0, cpu). Auto-detected if omitted.")
    # One-shot arguments (only used when stdin is a terminal)
    parser.add_argument("--text",      default="")
    parser.add_argument("--output",    default="")
    parser.add_argument("--voice",     default="Chelsie")
    parser.add_argument("--language",  default="English")
    parser.add_argument("--ref-audio", default="")
    parser.add_argument("--ref-text",  default="")
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Daemon mode when stdin is a pipe (called from dub_audio.py)
    if not sys.stdin.isatty():
        _daemon(args.mode, device)
        return 0

    # One-shot mode (direct CLI call / debugging)
    if not args.text or not args.output:
        parser.error("--text and --output are required in one-shot mode")
    return _one_shot(args.mode, device, args)


if __name__ == "__main__":
    sys.exit(main())
