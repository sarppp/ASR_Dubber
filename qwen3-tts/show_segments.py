#!/usr/bin/env python3
"""
show_segments.py — Dump every merged segment's text alongside its WAV status.

Usage (run from inside qwen3-tts/):
  python show_segments.py path/to/video.nemo.de.diarize_fr.srt [--workdir output/dub/MyVideo]

Reads the same SRT + applies the same merge_segments() logic as dub.py, then
cross-references with the checkpoint to show exactly what text was sent to TTS
for each segment.  Flags any segment whose text contains a literal newline.

Examples:
  # See all segments (no checkpoint needed)
  python show_segments.py ../nemo/video.nemo.de.diarize_fr.srt

  # Cross-reference with a completed run's checkpoint
  python show_segments.py ../nemo/video.nemo.de.diarize_fr.srt --workdir output/dub/Quantum
"""

import argparse
import json
import sys
from pathlib import Path

from dub_srt import parse_srt, merge_segments


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump segment texts from a diarized SRT")
    parser.add_argument("srt", help="Path to the translated+diarized SRT")
    parser.add_argument("--workdir", default=None,
                        help="dub.py workdir (contains checkpoint.json + temp/) for WAV status")
    parser.add_argument("--merge-gap",    type=float, default=1.0)
    parser.add_argument("--merge-max-dur", type=float, default=10.0)
    parser.add_argument("--no-merge",     action="store_true",
                        help="Show raw (unmerged) segments as parsed from SRT")
    parser.add_argument("--show-all",     action="store_true",
                        help="Print all segments, not just flagged ones")
    args = parser.parse_args()

    srt_path = Path(args.srt).resolve()
    if not srt_path.exists():
        print(f"ERROR: SRT not found: {srt_path}", file=sys.stderr)
        return 1

    segments = parse_srt(srt_path)
    if not segments:
        print("ERROR: No segments parsed", file=sys.stderr)
        return 1

    if not args.no_merge and args.merge_gap > 0:
        segments = merge_segments(segments, gap_sec=args.merge_gap, max_dur=args.merge_max_dur)

    # Load checkpoint if workdir given
    checkpoint: dict = {}   # index → clip path
    wav_status: dict = {}   # index → "fit" | "raw" | "missing"
    if args.workdir:
        work_dir = Path(args.workdir).resolve()
        ckpt_path = work_dir / "checkpoint.json"
        temp_dir  = work_dir / "temp"
        if ckpt_path.exists():
            data = json.loads(ckpt_path.read_text())
            for entry in data:
                clip = Path(entry["clip"])
                # Derive segment index from filename (seg_XXXX_fit.wav or seg_XXXX.wav)
                stem = clip.stem  # e.g. seg_0042_fit or seg_0042
                parts = stem.split("_")
                try:
                    idx = int(parts[1])
                    checkpoint[idx] = clip
                except (IndexError, ValueError):
                    pass
        if temp_dir.exists():
            for wav in temp_dir.glob("seg_*_fit.wav"):
                parts = wav.stem.split("_")
                try:
                    idx = int(parts[1])
                    wav_status[idx] = "fit"
                except (ValueError, IndexError):
                    pass
            for wav in temp_dir.glob("seg_*.wav"):
                if "_fit" not in wav.stem and "_cmp" not in wav.stem:
                    parts = wav.stem.split("_")
                    try:
                        idx = int(parts[1])
                        if idx not in wav_status:
                            wav_status[idx] = "raw"
                    except (ValueError, IndexError):
                        pass

    flagged = 0
    total   = len(segments)

    print(f"\n{'='*72}")
    print(f"SRT   : {srt_path.name}")
    print(f"Segs  : {total}  (after merge)" if not args.no_merge else f"Segs  : {total}  (raw)")
    if args.workdir:
        print(f"Work  : {args.workdir}")
    print(f"{'='*72}\n")

    for seg in segments:
        i    = seg["index"]
        text = seg["text"]
        spk  = seg["speaker"]
        has_newline = "\n" in text or "\r" in text

        status_str = ""
        if args.workdir:
            st = wav_status.get(i, "missing")
            marker = {"fit": "✓fit", "raw": "~raw", "missing": "✗miss"}[st]
            status_str = f"  [{marker}]"

        flag = "  ⚠️  EMBEDDED NEWLINE" if has_newline else ""

        show = args.show_all or has_newline
        if show:
            print(f"[{i:04d}] {spk}  {seg['start']:.2f}s–{seg['end']:.2f}s{status_str}{flag}")
            # Show repr so \n, \r etc. are visible
            print(f"       text={repr(text)}")
            if "subsegments" in seg and len(seg["subsegments"]) > 1:
                print(f"       merged from {len(seg['subsegments'])} sub-segments")
            print()

        if has_newline:
            flagged += 1

    print(f"{'='*72}")
    if flagged:
        print(f"⚠️   {flagged}/{total} segments contain embedded newlines  ← likely bug source")
    else:
        print(f"✅  No embedded newlines found in {total} segments")

    if not args.show_all and not flagged:
        print("    (run with --show-all to print every segment's text)")

    print()
    if args.workdir:
        missing = [seg["index"] for seg in segments
                   if wav_status.get(seg["index"], "missing") == "missing"]
        if missing:
            print(f"Missing WAVs ({len(missing)}): indices {missing[:20]}"
                  f"{'…' if len(missing) > 20 else ''}")
        else:
            print(f"All {total} segment WAVs present in {args.workdir}/temp/")
    else:
        print("Tip: pass --workdir output/dub/MyVideo to cross-reference with WAV files")

    return 0


if __name__ == "__main__":
    sys.exit(main())
