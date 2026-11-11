#!/usr/bin/env python3
"""Persistent Supertonic CPU worker using the dubbing JSON-lines protocol."""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys

LANGUAGES = {"english": "en", "german": "de", "french": "fr", "spanish": "es", "portuguese": "pt", "korean": "ko"}

def language_code(value):
    code = LANGUAGES.get(value.strip().lower(), value.strip().lower())
    if code not in "en ko ja ar bg cs da de el es et fi fr hi hr hu id it lt lv nl pl pt ro ru sk sl sv tr uk vi".split():
        raise ValueError(f"Unsupported Supertonic language: {value!r}; use a supported Supertonic 3 ISO language code, or --tts-engine qwen")
    return code

def synthesise(model, request):
    text = request.get("text", "").translate(str.maketrans({'„': '"', '“': '"', '”': '"', '–': '-', '—': '-', '…': '...'}))
    if not text.strip() or not request.get("output"):
        raise ValueError("missing text or output")
    language = language_code(request.get("language", "en"))
    style = model.get_voice_style(voice_name=request.get("voice", "M1"))
    wav, _ = model.synthesize(text, voice_style=style, lang=language)
    output = Path(request["output"])
    model.save_audio(wav, str(output))
    if not output.exists() or output.stat().st_size < 100:
        raise RuntimeError(f"Output WAV missing or empty: {output}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["custom"], default="custom")
    parser.parse_args()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "2"
    try:
        with redirect_stdout(sys.stderr):
            from supertonic import TTS
            model = TTS(model="supertonic-3", auto_download=True,
                        intra_op_num_threads=2, inter_op_num_threads=1)
    except Exception as exc:
        print(f"LOAD_ERROR: {exc}", flush=True)
        return 1
    print("READY", flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("quit"):
                print(json.dumps({"ok": True}), flush=True)
                break
            with redirect_stdout(sys.stderr):
                synthesise(model, request)
            response = {"ok": True}
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        print(json.dumps(response), flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
