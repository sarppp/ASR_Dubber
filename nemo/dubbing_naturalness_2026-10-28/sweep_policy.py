"""Counterfactual timing only; infer duration only for previously compressed clips."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from batch_dubbing.natural_timing import plan_timing


def main():
    review = json.loads((HERE / "review_data.json").read_text())
    result = {"scope": "First600s Mind and Bill; current grouping held fixed; durations inferred only when reported rate > 1. Original phrase ends retained with bounded gap borrowing. Not resynthesis, not a quality prediction. Reconstructed historical durations need verification by fresh synthesis.", "videos": []}
    for video in review["videos"]:
        rows = video["merged_segments"]
        output = {"id": video["id"], "segments": len(rows),
                  "unknown_raw_duration": sum(row["rate"] == 1 for row in rows), "caps": []}
        for cap in (1.0, 1.1, 1.15, 1.25, 1.35, 1.5):
            counts = {}
            for index, row in enumerate(rows):
                raw = row["raw_seconds_inferred_from_rate"]
                if raw is None:
                    continue
                decision = plan_timing(row, raw, next_phrase=rows[index + 1] if index + 1 < len(rows) else None, max_rate=cap)
                counts[decision["status"]] = counts.get(decision["status"], 0) + 1
            output["caps"].append({"max_rate": cap, "statuses": counts})
        result["videos"].append(output)
    (HERE / "policy_sweep.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
