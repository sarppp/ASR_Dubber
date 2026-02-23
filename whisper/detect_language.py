#!/usr/bin/env python3
"""
detect_language.py — Detect spoken language of a video using Whisper
=====================================================================
Runs Whisper's built-in language detection on the first 30s of audio.
Prints ONLY the 2-letter language code to stdout (e.g. "de") so the
master runner can capture it cleanly.

Usage:
  python detect_language.py video.mp4
  python detect_language.py video.mp4 --model turbo
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import whisper


def main():
    p = argparse.ArgumentParser(description="Detect spoken language via Whisper")
    p.add_argument("video", help="Video or audio file")
    p.add_argument("--model", default="turbo",
                   choices=["tiny", "base", "small", "medium", "turbo", "large-v3"],
                   help="Whisper model to use (default: turbo)")
    args = p.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"ERROR: file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # Extract first 30s of audio to a temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-threads", "0",
             "-i", str(video_path),
             "-t", "30",
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
             tmp_wav],
            check=True,
            capture_output=True,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = whisper.load_model(args.model, device=device)

        # detect_language() runs a single encoder forward pass — very fast
        audio  = whisper.load_audio(tmp_wav)
        audio  = whisper.pad_or_trim(audio)
        mel    = whisper.log_mel_spectrogram(audio).to(model.device)

        _, probs = model.detect_language(mel)
        lang = max(probs, key=probs.get)

        # Print ONLY the language code — nothing else — so caller can capture it
        print(lang)

    finally:
        Path(tmp_wav).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
