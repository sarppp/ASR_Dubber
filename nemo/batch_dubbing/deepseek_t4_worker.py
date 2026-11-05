"""Bounded Modal transport for the repaired dub.py; ten original SRT cues."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

import modal

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "new_guys/crispr/deepseek_t4_10cues_2026-11-05"
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg", "libsndfile1", "sox", "libsox-dev")
         .pip_install("torch==2.6.0", "torchaudio==2.6.0", "torchvision==0.21.0")
         .pip_install("qwen-tts==0.1.1", "faster-qwen3-tts==0.2.5", "transformers==4.57.3",
                      "accelerate", "huggingface_hub<1", "soundfile", "numpy<2", "scipy", "psutil")
         .env({"HF_HOME": "/cache/huggingface", "OMP_NUM_THREADS": "4",
               "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
               "TOKENIZERS_PARALLELISM": "false"})
         .add_local_dir(TRIAL / "code", "/work/qwen3-tts")
         .add_local_file(ROOT / "batch_dubbing/trial_telemetry.py", "/work/telemetry.py"))
for name in ("fit_audio", "audio_edges", "natural_render"):
    image = image.add_local_file(ROOT / f"batch_dubbing/{name}.py", f"/work/nemo/batch_dubbing/{name}.py")
for name in ("source_10cues.mp4", "voice_reference.wav", "input.de.srt", "smoke.de.srt", "scope.json"):
    image = image.add_local_file(TRIAL / name, f"/input/{name}")
volume = modal.Volume.from_name("speech-model-benchmark-20261027")
app = modal.App("deepseek-repair-ten-cues-t4")


@app.function(image=image, gpu="T4", cpu=(4, 4), memory=(24576, 24576),
              timeout=240, startup_timeout=120, max_containers=1,
              min_containers=0, retries=0, scaledown_window=2, volumes={"/cache": volume})
def validate(smoke: bool):
    import shutil
    import subprocess
    import sys
    import time
    import wave
    sys.path.insert(0, "/work")
    from telemetry import emit, start
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                  text=True, timeout=10).strip()
    if "T4" not in gpu:
        raise RuntimeError("T4 is required; no hardware fallback")
    emit("worker_ready", gpu=gpu, smoke=smoke)
    monitor = start()
    began = time.monotonic()
    scope_bytes = Path("/input/scope.json").read_bytes()
    scope = json.loads(scope_bytes)
    if scope["cues"] != 10 or scope["duration_seconds"] > 30 or scope["max_speed"] != 1.:
        raise ValueError("Input exceeds the reviewed ten-cue scope")
    for name, digest in scope["input_sha256"].items():
        if hashlib.sha256((Path("/input") / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Input digest mismatch: {name}")
    work = Path("/cache/deepseek-ten-cues") / hashlib.sha256(scope_bytes).hexdigest()[:20]
    work.mkdir(parents=True, exist_ok=True)
    cast = work / "cast_samples"
    cast.mkdir(exist_ok=True)
    ref = cast / "Speaker_1.wav"
    if not ref.exists():
        shutil.copy2("/input/voice_reference.wav", ref)
    stage = "smoke" if smoke else "workload"
    expected = 2 if smoke else 10
    script = "/input/smoke.de.srt" if smoke else "/input/input.de.srt"
    output = work / "output"
    if not smoke and output.exists():
        output.rename(work / "smoke_output")
    command = [sys.executable, "-u", "/work/qwen3-tts/dub.py", "/input/source_10cues.mp4", script,
               "--qwen-dir", "/work/qwen3-tts", "--workdir", str(work),
               "--tts-engine", "qwen", "--qwen-mode", "clone", "--language", "de",
               "--no-demucs", "--max-speed", "1.0", "--min-speed", "1.0",
               "--merge-gap", "0", "--tts-workers", "1", "--tts-devices", "0"]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=210)
        (work / f"{stage}.stdout.log").write_text(process.stdout)
        (work / f"{stage}.stderr.log").write_text(process.stderr)
        print(process.stdout, flush=True)
        print(process.stderr, flush=True)
        fit_path = work / "fit_report.json"
        fit = json.loads(fit_path.read_text()) if fit_path.exists() else {}
        rows = fit.get("segments", [])
        cache = Path(fit.get("cache_dir", str(work / "missing")))
        raw = [cache / "temp" / f"seg_{i:04d}.wav" for i in range(1, expected + 1)]
        valid = True
        durations = []
        for path in raw:
            if not path.exists():
                valid = False
                continue
            with wave.open(str(path)) as audio:
                duration = audio.getnframes() / audio.getframerate()
                valid &= 0 < duration < 20 and any(audio.readframes(audio.getnframes()))
                durations.append(duration)
        complete = (valid and len(rows) == expected
                    and all(row.get("needed") is not None for row in rows))
        report = {"status": "complete" if complete else "failed",
                  "measurement_status": "complete" if complete else "failed",
                  "dub_status": "passed" if process.returncode == 0 else "needs_review",
                  "returncode": process.returncode, "gpu": gpu, "expected_cues": expected,
                  "measured_cues": len(rows), "raw_seconds": durations,
                  "worker_seconds": time.monotonic() - began, "fit_report": fit,
                  "model": scope["model"], "revision": scope["revision"],
                  "scope_sha256": hashlib.sha256(scope_bytes).hexdigest(),
                  "note": "Timing validation only. Existing wording unchanged; meaning/listening not validated."}
        (work / f"{stage}.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        result = io.BytesIO()
        with tarfile.open(fileobj=result, mode="w:gz") as archive:
            for path in [work / f"{stage}.report.json", work / f"{stage}.stdout.log",
                         work / f"{stage}.stderr.log", fit_path]:
                if path.exists():
                    archive.add(path, arcname=path.name)
            if cache.exists():
                archive.add(cache, arcname="cache")
            if process.returncode == 0 and output.exists():
                archive.add(output, arcname="output")
        if result.tell() > 128 * 1024**2:
            raise ValueError("Result exceeds the 128 MiB transfer bound")
        emit("validation_complete", status=report["measurement_status"], dub_status=report["dub_status"])
        return result.getvalue()
    finally:
        volume.commit()
        monitor.set()


@app.local_entrypoint()
def main(smoke: bool = False):
    stage = "smoke" if smoke else "workload"
    destination = TRIAL / ("smoke_report_repair" if smoke else stage)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite previous validation: {destination}")
    data = validate.remote(smoke)
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(destination, filter="data")
    report = json.loads((destination / f"{stage}.report.json").read_text())
    if report["measurement_status"] != "complete":
        raise RuntimeError("Not every requested cue produced measured speech; evidence preserved")
    print("MEASUREMENT_RESULT", destination, report["dub_status"], flush=True)
