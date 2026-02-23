#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Qwen TTS Worker")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", required=True, help="Output WAV path")
    parser.add_argument("--mode", choices=["clone", "custom"], default="custom", help="TTS mode")
    parser.add_argument("--ref-audio", default="", help="Reference audio for clone mode")
    parser.add_argument("--ref-text", default="", help="Reference text for clone mode (optional)")
    parser.add_argument("--voice", default="Chelsie", help="Voice name for custom mode")
    parser.add_argument("--language", default="French", help="Target language")
    args = parser.parse_args()

    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base" if args.mode == "clone" else "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    print(f"Loading {model_id}...")
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=torch.bfloat16,
    )

    if args.mode == "clone":
        if not args.ref_audio or not Path(args.ref_audio).exists():
            print(f"ERROR: Reference audio missing: {args.ref_audio}", file=sys.stderr)
            return 1
        
        print("Using generate_voice_clone...")
        if args.ref_text:
            wavs, sr = model.generate_voice_clone(
                text=args.text,
                language=args.language,
                ref_audio=args.ref_audio,
                ref_text=args.ref_text,
            )
        else:
            wavs, sr = model.generate_voice_clone(
                text=args.text,
                language=args.language,
                ref_audio=args.ref_audio,
                x_vector_only_mode=True,
            )
    else:
        try:
            print("Using generate_custom_voice...")
            wavs, sr = model.generate_custom_voice(
                text=args.text,
                language=args.language,
                speaker=args.voice,
                instruct="",
            )
        except AttributeError as e:
            print(f"Method not found! Error: {e}", file=sys.stderr)
            print("Attempting fallback: .generate()...")
            wavs, sr = model.generate(
                text=args.text,
                voice=args.voice,
                language=args.language,
            )

    if wavs is None or len(wavs) == 0:
        print("ERROR: No audio generated", file=sys.stderr)
        return 1

    audio_data = wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else wavs[0]
    sf.write(args.output, audio_data, sr)
    print(f"OK: {args.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())