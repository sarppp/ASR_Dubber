"""Stream-validate the completed FLOAT WAV trial; stdlib only, no inference."""
import array
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import struct
import sys

HERE = Path(__file__).parent


def inspect_wav(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    with path.open("rb") as handle:
        header = handle.read(12)
        if header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise ValueError("Expected RIFF WAVE")
        fmt, frames, peak = None, None, 0.0
        while chunk := handle.read(8):
            name, size = struct.unpack("<4sI", chunk)
            if name == b"fmt ":
                fmt = struct.unpack("<HHIIHH", handle.read(size)[:16])
                if fmt[0] != 3 or fmt[1] != 1 or fmt[5] != 32:
                    raise ValueError("Expected mono IEEE FLOAT32 WAV")
            elif name == b"data":
                if fmt is None or size % 4:
                    raise ValueError("Invalid WAV data layout")
                frames = size // 4
                remaining = size
                while remaining:
                    data = handle.read(min(65536, remaining))
                    if not data:
                        raise ValueError("Truncated WAV")
                    remaining -= len(data)
                    samples = array.array("f", data)
                    if sys.byteorder != "little":
                        samples.byteswap()
                    for value in samples:
                        if not math.isfinite(value):
                            raise ValueError("Nonfinite audio sample")
                        peak = max(peak, abs(value))
            else:
                handle.seek(size, 1)
            if size % 2:
                handle.seek(1, 1)
        if frames is None or peak <= 1e-5:
            raise ValueError("Missing or silent waveform")
    return {"file": path.name, "frames": frames, "sample_rate": fmt[2],
            "seconds": frames / fmt[2], "peak": peak, "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size}


def main():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.nice(15)
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    sys.path.insert(0, str(HERE.parent))
    from batch_dubbing.naturalness_experiment import build_jobs
    inputs = json.loads((HERE / "experiment_inputs.json").read_text())
    expected_jobs = build_jobs(inputs)
    results = {"status": "complete", "checks": ["exact job/text/voice identity", "native frame durations",
               "all samples finite", "non-silent audio", "recorded peak matches", "SHA256 recorded"], "stages": {}}
    full_rows = []
    for stage in ("smoke", "full"):
        folder = HERE / "native_grouping_trial" / stage
        report = json.loads((folder / "report.json").read_text())
        rows = report["segments"]
        expected = expected_jobs[:1] if stage == "smoke" else expected_jobs
        assert report["status"] == "complete" and len(rows) == len(expected)
        verified = []
        for row, job in zip(rows, expected):
            assert all(row[key] == value for key, value in job.items()), row["id"]
            assert Path(row["file"]).name == row["file"]
            wav = inspect_wav(folder / row["file"])
            assert abs(wav["seconds"] - row["raw_duration"]) < 1e-9
            assert wav["sample_rate"] == row["sample_rate"]
            assert abs(wav["peak"] - row["peak"]) < 1e-7
            verified.append({"id": row["id"], **wav})
            if stage == "full":
                full_rows.append({**row, "sha256": wav["sha256"]})
        results["stages"][stage] = {"calls": len(rows), "worker_wall_seconds": report["wall_seconds"], "files": verified}
    grouped = defaultdict(list)
    for row in full_rows:
        grouped[row["excerpt_id"], row["arm"]].append(row)
    results["groups"] = [{"excerpt": key[0], "arm": key[1], "clips": len(rows),
        "raw_seconds": sum(row["raw_duration"] for row in rows),
        "timing_statuses": dict(Counter(row["timing_decision_only"]["status"] for row in rows))}
        for key, rows in grouped.items()]
    same_text = []
    for row in full_rows:
        if row["arm"] != "baseline":
            continue
        for other in full_rows:
            if (other["arm"] == "candidate" and row["excerpt_id"] == other["excerpt_id"]
                and row["voice"] == other["voice"] and row["synthesis_text"] == other["synthesis_text"]):
                same_text.append({"baseline": row["id"], "candidate": other["id"],
                                  "byte_identical": row["sha256"] == other["sha256"]})
    results["identical_text_voice_pairs"] = same_text
    results["limits"] = "No listening assessment, ASR, word-completeness proof or semantic quality score. All timing decisions are offline diagnostics, not applied fits."
    results["verification_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (HERE / "native_trial_verification.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: results[k] for k in ("status", "groups", "identical_text_voice_pairs", "verification_peak_rss_kib")}, indent=2))


if __name__ == "__main__":
    main()
