"""Bounded, model-free experiments on existing audio. Never edits inputs.

Run serially: python3 dubbing_naturalness_2026-10-28/run_audio_checks.py
User selected these comparisons and English/German ten-minute analysis in chat.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent
JOBS = []
START = time.monotonic()


def limits():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.nice(15)
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def ffmpeg(arguments, label):
    if time.monotonic() - START > 300:
        raise TimeoutError("Five-minute experiment wall-time budget exhausted")
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-n",
               "-threads", "1", "-filter_threads", "1", "-filter_complex_threads", "1",
               *map(str, arguments)]
    env = {**os.environ, **{key: "1" for key in
                           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, env=env,
                            preexec_fn=limits, timeout=min(90, 300 - (start - START)))
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    JOBS.append({"label": label, "command": command,
                 "wall_seconds": time.monotonic() - start,
                 "cpu_seconds": usage.ru_utime + usage.ru_stime - before.ru_utime - before.ru_stime,
                 "children_peak_rss_kib_so_far": usage.ru_maxrss,
                 "returncode": result.returncode})
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    return result.stderr


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def duration(path):
    with wave.open(str(path)) as wav:
        return wav.getnframes() / wav.getframerate()


def silence_intervals(log, total=600):
    intervals, start = [], None
    for line in log.splitlines():
        match = re.search(r"silence_start: ([\d.eE+\-]+)", line)
        if match:
            start = max(0, float(match[1]))
        match = re.search(r"silence_end: ([\d.eE+\-]+)", line)
        if match and start is not None:
            end = min(total, float(match[1]))
            if end > start:
                intervals.append([start, end])
            start = None
    if start is not None and start < total:
        intervals.append([start, total])
    return intervals


def main():
    comparisons = OUT / "speed_comparisons"
    comparisons.mkdir(exist_ok=False)
    results = {"limits": {"cpu_affinity_cores": 1, "serial_jobs": True,
                           "address_space_mib_per_child": 512, "cpu_seconds_per_child": 60,
                           "wall_seconds_per_child": 90, "total_wall_budget_seconds": 300,
                           "gpu_used": False, "model_inference": False},
               "comparisons": [], "acoustic": [], "jobs": JOBS}
    try:
        for model in ("supertonic_3", "qwen_tts_06b"):
            folder = ROOT / "speech_model_benchmark_2026-10-27/04_dubbing" / model
            row = json.loads((folder / "segments.json").read_text())[0]
            source = folder / row["path"]
            raw_seconds = duration(source)
            for rate in (1.0, 0.65, 1.15, 1.5, 2.0):
                target = comparisons / f"{model}_{rate:.2f}x.wav"
                ffmpeg(["-i", source, "-map", "0:a:0", "-af", f"atempo={rate}",
                        "-threads", "1", "-c:a", "pcm_s16le", target], target.stem)
                measured = duration(target)
                expected = raw_seconds / rate
                if abs(measured - expected) > .15:
                    raise ValueError(f"Unexpected tempo duration: {target}")
                results["comparisons"].append({"model": model, "rate": rate,
                    "source": str(source.relative_to(ROOT)), "source_sha256": digest(source),
                    "text": row["text"], "raw_seconds": raw_seconds,
                    "measured_seconds": measured, "expected_seconds": expected,
                    "path": str(target.relative_to(OUT)), "sha256": digest(target)})
                print("Completed", target.name, flush=True)
        for index in (0, 4):
            meta = json.loads((ROOT / f"modal_control/batch_inputs/video_{index:02d}.json").read_text())
            original = ROOT / meta["source"]
            # Some metadata predates per-video organization; resolve existing basename only.
            if not original.exists():
                matches = list((ROOT / "new_guys").glob("*/" + original.name))
                if len(matches) != 1:
                    raise ValueError("Original source cannot be resolved unambiguously")
                original = matches[0]
            dub = original.with_name(original.stem + ".german-dub.mp4")
            for language, source in (("en", original), ("de", dub)):
                for threshold in (-35, -45):
                    label = f"video_{index:02d}_{language}_{threshold}db"
                    log = ffmpeg(["-i", source, "-t", "600", "-map", "0:a:0", "-vn",
                        "-af", f"silencedetect=noise={threshold}dB:duration=0.20",
                        "-threads", "1", "-f", "null", "-"], label)
                    (OUT / (label + ".log")).write_text(log)
                    intervals = silence_intervals(log)
                    results["acoustic"].append({"video_index": index, "language": language,
                        "path": str(source.relative_to(ROOT)), "seconds": 600,
                        "threshold_db": threshold, "minimum_silence_seconds": .2,
                        "silence_intervals": intervals, "silence_seconds": sum(b-a for a,b in intervals),
                        "silences_over_1s": sum(b-a > 1 for a,b in intervals),
                        "longest_silence_seconds": max((b-a for a,b in intervals), default=0),
                        "limitation": "Amplitude threshold, not speech VAD or word alignment. Music, noise and level changes affect results. No quality score."})
                    print("Completed", label, flush=True)
        results["status"] = "complete"
    except Exception as exc:
        results["status"] = "failed"
        results["error"] = str(exc)
        raise
    finally:
        results["elapsed_seconds"] = time.monotonic() - START
        (OUT / "audio_checks.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
