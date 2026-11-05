#!/usr/bin/env python3
"""Check harness for the subtitle-dubbing pipeline.

    --offline        L0 + L1: guards, SRT integrity, merge boundaries,
                     fake-TTS convergence, measured speech rate. Seconds, no
                     GPU, no models.
    --timing DIR     L4: does the speech sit inside its slot in a finished dub?
                     Reads the artefacts of a real run; no GPU.
    --meaning        L3: transcribe the finished dub again and check that the
                     meaning survived. Needs the local ASR (GPU).
    --batch DIR      (planned) run the failure-mode clip matrix, write metrics.json.

--offline and --timing need nothing but Python.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITES = [
    ROOT / "tests" / "test_dubbing_gate.py",
    ROOT / "tests" / "test_speech_rate.py",
    ROOT / "tests" / "test_dub_quality.py",
    ROOT / "tests" / "test_subtitle_pipeline.py",
]

# The dub of the 120 s CRISPR clip that the L3/L4 levels were checked on.
REVIEW = ROOT / "nemo" / "new_guys" / "crispr" / "auto_trial" / "review_10sent"
REVIEW_DUB = REVIEW / "CRISPR-120s__en_to_de"
REVIEW_DUB_WORKDIR = ROOT / "qwen3-tts" / "output" / "dub" / "CRISPR-120s"


def run_offline() -> int:
    failed = 0
    for suite in SUITES:
        print(f"\n▶ {suite.relative_to(ROOT)}", flush=True)
        result = subprocess.run([sys.executable, str(suite)], cwd=str(ROOT))
        failed += result.returncode != 0
    print(f"\n{'✅ offline checks passed' if not failed else '❌ offline checks failed'}")
    return 0 if not failed else 1


def run_timing(dub_dir: Path, script: Path | None) -> int:
    """L4: check the timing of a finished dub."""
    sys.path.insert(0, str(ROOT / "nemo"))
    from dub_timing import check_dub

    print(f"\n▶ L4 timing check — {dub_dir}")
    report = check_dub(dub_dir, script=script)
    summary = report.summary()
    for segment in report.segments:
        print(f"   seg {segment.index:3d} {segment.start:7.2f}-{segment.end:7.2f} "
              f"span {segment.span:6.2f} speech {segment.speech:6.2f} "
              f"({segment.overshoot:.3f}x) onset {segment.onset:.3f} "
              f"{segment.cps:5.1f} chars/s")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("✅ timing ok" if summary["ok"] else "❌ timing problems found")
    return 0 if summary["ok"] else 1


def _default_script(dub_dir: Path) -> Path | None:
    """The SRT that was spoken, when the caller did not name one.

    A successful dub writes it into ``<workdir>/output/``; a run that was
    reviewed by hand keeps it next to the video. Both are looked for.
    """
    candidates = sorted(dub_dir.rglob("*_dub.srt")) + \
        sorted(dub_dir.rglob("*.dub.srt"))
    if candidates:
        return candidates[0]
    review = REVIEW_DUB / "CRISPR-120s.nemo.de.adapted_dub.srt"
    return review if review.exists() else None


def run_meaning() -> int:
    """L3: transcribe the finished dub and check the meaning."""
    script = REVIEW_DUB / "CRISPR-120s.nemo.de.adapted_dub.srt"
    video = REVIEW_DUB / "final_dub.mp4"
    source = REVIEW / "CRISPR-120s.nemo.en.srt"
    if not (script.exists() and video.exists() and source.exists()):
        print(f"❌ the review artefacts are missing under {REVIEW}", file=sys.stderr)
        return 2
    print(f"\n▶ L3 meaning check — {video.name}")
    interpreter = ROOT / "nemo" / ".venv-local-asr" / "bin" / "python"
    if not interpreter.exists():
        print(f"❌ {interpreter} is missing (the local ASR environment)",
              file=sys.stderr)
        return 2
    return subprocess.run([str(interpreter), str(ROOT / "tools" / "check_meaning.py"),
                           "--dub", str(video), "--spoken", str(script),
                           "--source", str(source), "--language", "de"],
                          cwd=str(ROOT)).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="run L0 + L1 checks")
    parser.add_argument("--timing", type=Path, default=None,
                        metavar="DIR", help="L4 timing check on a finished dub")
    parser.add_argument("--meaning", action="store_true",
                        help="L3 meaning check on the review dub (needs the local ASR)")
    parser.add_argument("--batch", default=None, help="(not implemented yet) clip matrix")
    args = parser.parse_args()

    if args.batch:
        print("--batch is not implemented yet", file=sys.stderr)
        return 2
    if args.offline:
        return run_offline()
    if args.timing:
        return run_timing(args.timing, _default_script(args.timing))
    if args.meaning:
        return run_meaning()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
