"""Refresh review artifacts after completed jobs; evaluate newly generated TTS."""

import json, subprocess, sys, time
from pathlib import Path
from modal_control.core import atomic, clean, utc

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "speech_model_benchmark_2026-10-27"
CAT = ROOT / "modal_control/catalog/jobs.json"
last = None
attempted = set()
while True:
    try:
        metrics = sorted(BENCH.glob("runs/*/*/metrics.json"))
        identity = [(str(p), p.stat().st_mtime_ns) for p in metrics]
        if identity != last:
            subprocess.run(
                [sys.executable, "-m", "modal_control.organize_report"],
                cwd=ROOT,
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "modal_control.compare_benchmark"],
                cwd=ROOT,
                check=True,
            )
            last = identity
        for p in metrics:
            if p.parent.name != "tts":
                continue
            key = p.parts[-3]
            fingerprint = (key, p.stat().st_mtime_ns)
            quality = p.parent / "quality_proxy.json"
            if fingerprint in attempted or (
                quality.exists() and quality.stat().st_mtime >= p.stat().st_mtime
            ):
                continue
            if json.loads(p.read_text()).get("status") != "complete":
                continue
            attempted.add(fingerprint)
            cat = json.loads(CAT.read_text())
            name = "evaluate_" + key
            command = [
                sys.executable,
                "-m",
                "modal",
                "run",
                str(ROOT / "modal_control/evaluate_audio.py"),
                "--key",
                key,
            ]
            cat["jobs"][name] = {
                "cwd": str(ROOT),
                "command": command,
                "smoke_command": command + ["--smoke"],
                "success_files": [str(quality.relative_to(ROOT))],
                "smoke_success_files": [
                    str((p.parent / "quality_smoke.json").relative_to(ROOT))
                ],
                "source_files": ["modal_control/evaluate_audio.py"],
                "startup_timeout_seconds": 240,
                "max_seconds": 1800,
            }
            atomic(CAT, cat)
            subprocess.run(
                [sys.executable, "-m", "modal_control", "run", name], cwd=ROOT
            )
            subprocess.run(
                [sys.executable, "-m", "modal_control.organize_report"], cwd=ROOT
            )
            subprocess.run(
                [sys.executable, "-m", "modal_control.compare_benchmark"], cwd=ROOT
            )
    except Exception as exc:
        print(utc(), clean(str(exc)), flush=True)
    time.sleep(30)
