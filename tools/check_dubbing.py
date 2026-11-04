#!/usr/bin/env python3
"""Check harness for the subtitle-dubbing pipeline.

    --offline        L0 + L1: guards, SRT integrity, merge boundaries,
                     fake-TTS convergence. Seconds, no GPU, no models.
    --clip X         (planned) L2-L4 on a real clip: measured fit, round-trip
                     ASR, timing/onset assertions.
    --batch DIR      (planned) run the failure-mode clip matrix, write metrics.json.

Only --offline is implemented so far.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITES = [
    ROOT / "tests" / "test_dubbing_gate.py",
    ROOT / "tests" / "test_subtitle_pipeline.py",
]


def run_offline() -> int:
    failed = 0
    for suite in SUITES:
        print(f"\n▶ {suite.relative_to(ROOT)}", flush=True)
        result = subprocess.run([sys.executable, str(suite)], cwd=str(ROOT))
        failed += result.returncode != 0
    print(f"\n{'✅ offline checks passed' if not failed else '❌ offline checks failed'}")
    return 0 if not failed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="run L0 + L1 checks")
    parser.add_argument("--clip", default=None, help="(not implemented yet) real-clip smoke")
    parser.add_argument("--batch", default=None, help="(not implemented yet) clip matrix")
    args = parser.parse_args()

    if args.clip or args.batch:
        print("--clip/--batch are not implemented yet; use --offline", file=sys.stderr)
        return 2
    if not args.offline:
        parser.print_help()
        return 2
    return run_offline()


if __name__ == "__main__":
    raise SystemExit(main())
