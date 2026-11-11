#!/usr/bin/env python3
"""Measure the true Qwen speech rate from the natural-fit sidecars.

Every synthesised clip has a `*_natural_fit_*.json` sidecar that records the
TTS `raw_seconds` (how long the voice actually spoke) next to the
`target_seconds` (the subtitle slot it had to fit into). Pairing those two
gives the exact overshoot per segment, and the character count of the text
that was spoken gives the real chars/second.

This is the ground truth for the character budget: if the real rate is not
close to the assumed 11.6 chars/s, no translation-length budget can put the
dub at natural pace.

    python3 tools/measure_real_cps.py
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DUB_DIR = ROOT / "qwen3-tts" / "output" / "dub" / "CRISPR-120s"
SCRIPT = (ROOT / "nemo/new_guys/crispr/auto_trial/review_10sent"
          "/CRISPR-120s__en_to_de/CRISPR-120s.nemo.de.adapted_dub.srt")


def parse_srt(path: Path) -> list[dict]:
    cues, cur = [], None
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if "-->" in line:
            left, right = line.split("-->")
            cur = [left.strip(), right.strip(), []]
        elif cur is not None and line:
            cur[2].append(line)
        elif cur is not None:
            cues.append(cur)
            cur = None
    if cur:
        cues.append(cur)
    return cues


def to_seconds(stamp: str) -> float:
    hours, minutes, seconds = stamp.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def main() -> int:
    cues = [{"start": to_seconds(a), "end": to_seconds(b), "text": " ".join(lines)}
            for a, b, lines in parse_srt(SCRIPT)]

    # seg_NNNN in the filename is dub.py's own segment index (1-based).
    rows = []
    for sidecar in sorted(DUB_DIR.rglob("*_natural_fit_*.json")):
        data = json.loads(sidecar.read_text())
        config = data.get("configuration", {})
        edges = data.get("edges", {})
        raw = edges.get("raw_seconds")
        target = config.get("target_seconds")
        match = re.search(r"seg_(\d+)", sidecar.name)
        if not match or not raw or not target:
            continue
        index = int(match.group(1))
        # The clip covers the cues that start at the segment's start time.
        # dub.py's segment index counts merged blocks; approximate the text by
        # taking the cues that fall inside [start, start+target].
        start = None
        for cue in cues:
            pass
        rows.append({"segment": index, "raw": raw, "target": target,
                     "over": raw / target if target else None,
                     "sidecar": sidecar.name,
                     "cache": sidecar.parent.parent.name})

    # Map segment index -> spoken text via the checkpoint entries.
    texts: dict[int, str] = {}
    for checkpoint in sorted(DUB_DIR.rglob("cache/qwen/*/checkpoint.json")):
        for entry in json.loads(checkpoint.read_text()):
            name = Path(entry.get("clip", "")).name
            match = re.search(r"seg_(\d+)", name)
            if match:
                texts.setdefault(int(match.group(1)), entry["start"])

    print(f"{'seg':>4} {'start':>7} {'target':>7} {'spoke':>7} {'over':>6}  text")
    rates = []
    for row in sorted(rows, key=lambda r: (r["cache"], r["segment"])):
        start = texts.get(row["segment"])
        block = ""
        if start is not None:
            span = row["target"]
            pieces = [c["text"] for c in cues
                      if start - 0.2 <= c["start"] < start + span]
            block = " ".join(pieces)
            if block:
                rates.append(len(block) / row["raw"])
        print(f"{row['segment']:4d} {start if start is not None else -1:7.2f} "
              f"{row['target']:7.2f} {row['raw']:7.2f} {row['over']:6.3f}  "
              f"{block[:52]}")

    if rates:
        print(f"\nMEASURED chars/s (raw TTS, no fitting):")
        print(f"  median {statistics.median(rates):.2f}  mean {statistics.mean(rates):.2f}  "
              f"min {min(rates):.2f}  max {max(rates):.2f}  spread {max(rates) / min(rates):.1f}x")
        print(f"  pipeline assumption (DE): 11.60 chars/s")
    overs = [r["over"] for r in rows if r["over"]]
    if overs:
        print(f"\nOVERSHOOT (speech/target): median {statistics.median(overs):.3f}  "
              f"max {max(overs):.3f}  over 1.0: {sum(1 for o in overs if o > 1.0)}/{len(overs)}")
    Path(ROOT / "tools" / "real_cps.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
