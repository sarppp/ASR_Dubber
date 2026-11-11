"""Read-only saved-audio analysis; writes decisions JSON, never audio.

Run: python3 dubbing_naturalness_2026-10-28/audit_edges.py
No inference, cloud call, FFmpeg process, waveform modification, or resampling.
"""
import array
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from batch_dubbing.audio_edges import analyze_edges


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            result.update(block)
    return result.hexdigest()


def read_float_wav(path):
    """Read only the bounded mono FLOAT32 artifacts produced by this trial."""
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("Audio artifact exceeds 64 MiB audit bound")
    samples, fmt = None, None
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise ValueError("Expected RIFF WAVE")
        file_end = path.stat().st_size
        while chunk := handle.read(8):
            if len(chunk) != 8:
                raise ValueError("Truncated RIFF chunk")
            name, size = struct.unpack("<4sI", chunk)
            if handle.tell() + size > file_end:
                raise ValueError("WAV chunk exceeds file bounds")
            if name == b"fmt ":
                data = handle.read(size)
                if len(data) < 16:
                    raise ValueError("Invalid WAV format")
                fmt = struct.unpack("<HHIIHH", data[:16])
                if fmt[0] != 3 or fmt[1] != 1 or fmt[5] != 32 or fmt[4] != 4:
                    raise ValueError("Expected mono IEEE FLOAT32 WAV")
            elif name == b"data":
                if fmt is None or size % 4 or samples is not None:
                    raise ValueError("Invalid FLOAT WAV data layout")
                samples = array.array("f", handle.read(size))
                if sys.byteorder != "little":
                    samples.byteswap()
            else:
                handle.seek(size, 1)
            if size % 2:
                handle.seek(1, 1)
    if samples is None:
        raise ValueError("WAV has no data")
    return samples, fmt[2]


def main():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.nice(15)
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    started = time.monotonic()
    folder = HERE / "native_grouping_trial/full"
    report_path = folder / "report.json"
    report = json.loads(report_path.read_text())
    if report["status"] != "complete" or report["calls"] != 22 or len(report["segments"]) != 22:
        raise ValueError("Expected the complete 22-call native trial")
    rows = []
    for source in report["segments"]:
        filename = source["file"]
        if Path(filename).name != filename or not filename.endswith(".wav"):
            raise ValueError("Invalid native artifact filename")
        path = folder / filename
        samples, sample_rate = read_float_wav(path)
        decision = analyze_edges(samples, sample_rate)
        if abs(decision["raw_seconds"] - source["raw_duration"]) > 1 / sample_rate:
            raise ValueError("Saved waveform duration differs from native report")
        rows.append({"id": source["id"], "excerpt_id": source["excerpt_id"],
                     "arm": source["arm"], "voice": source["voice"],
                     "text": source["synthesis_text"], "phrase": source["phrase"],
                     "input_file": str(path.relative_to(ROOT)), "input_sha256": digest(path),
                     "input_bytes": path.stat().st_size, "decision": decision})
        del samples
    result = {
        "status": "complete", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Conservative peak-based edge decisions on 22 saved native trial clips; no audio written",
        "policy_sha256": digest(ROOT / "batch_dubbing/audio_edges.py"),
        "input_report_sha256": digest(report_path), "segments": rows,
        "summary": {
            "clips": len(rows), "raw_seconds": sum(r["decision"]["raw_seconds"] for r in rows),
            "retained_seconds": sum(r["decision"]["retained_seconds"] for r in rows),
            "edge_seconds_removable": sum(r["decision"]["removed_seconds"] for r in rows),
            "clips_with_capped_edge": sum(bool(r["decision"]["flags"]) for r in rows),
        },
        "limits": {"cpu_affinity_cores": 1, "address_space_mib": 512, "cpu_seconds": 60,
                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "elapsed_seconds": time.monotonic() - started},
        "interpretation": "Activity threshold is not speech recognition. The slice preserves all above-threshold frames and guard samples, not proven phonetic boundaries. Interior audio is unchanged. Source windows may remain infeasible after edge removal.",
    }
    (HERE / "edge_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": result["summary"], "limits": result["limits"]}, indent=2))


if __name__ == "__main__":
    main()
