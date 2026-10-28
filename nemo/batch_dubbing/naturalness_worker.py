"""Unreleased bounded Modal experiment: native German speech, two groupings."""
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "dubbing_naturalness_2026-10-28/experiment_inputs.json"
OUTPUT = ROOT / "dubbing_naturalness_2026-10-28/native_grouping_trial"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install("supertonic==1.3.1", "huggingface_hub", "soundfile", "numpy<2", "psutil")
    .env({"HF_HOME": "/cache/huggingface", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
          "MKL_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2"})
    .add_local_file(ROOT / "batch_dubbing/naturalness_experiment.py", "/work/batch_dubbing/naturalness_experiment.py")
    .add_local_file(ROOT / "batch_dubbing/natural_timing.py", "/work/batch_dubbing/natural_timing.py")
    .add_local_file(ROOT / "modal_control/dub_segments.py", "/work/modal_control/dub_segments.py")
    .add_local_file(ROOT / "speech_model_benchmark_2026-10-27/code/telemetry.py", "/work/telemetry.py")
)
volume = modal.Volume.from_name("speech-model-benchmark-20261027")
app = modal.App("dubbing-naturalness-native-trial-20261028")


@app.function(image=image, cpu=2, memory=4096, timeout=600, max_containers=1,
              retries=0, volumes={"/cache": volume})
def synthesize(data, smoke=False):
    import io
    import sys
    import time

    import numpy as np
    import soundfile as sf
    from huggingface_hub import snapshot_download
    from supertonic import TTS

    sys.path.insert(0, "/work")
    from batch_dubbing.naturalness_experiment import build_jobs, MODEL, REVISION, PACKAGE
    from batch_dubbing.natural_timing import plan_timing
    from telemetry import emit, start

    jobs = build_jobs(data)
    jobs = jobs[:1] if smoke else jobs
    emit("worker_ready", hardware="CPU", cpu=2, memory_mib=4096, calls=len(jobs), smoke=smoke)
    monitor = start()
    started = time.monotonic()
    try:
        model_dir = snapshot_download(MODEL, revision=REVISION)
        model = TTS(model="supertonic-3", model_dir=model_dir, auto_download=False,
                    intra_op_num_threads=2, inter_op_num_threads=1)
        rows, audio_files, total_seconds = [], {}, 0.0
        for index, job in enumerate(jobs):
            phrase = job["phrase"]
            text = phrase["text"].translate(str.maketrans({'„': '"', '“': '"', '”': '"', '–': '-', '—': '-', '…': '...'}))
            before = time.monotonic()
            wav, _ = model.synthesize(text, voice_style=model.get_voice_style(job["voice"]), lang="de")
            wav = np.asarray(wav, dtype=np.float32).squeeze()
            if wav.ndim != 1 or not wav.size or not np.isfinite(wav).all():
                raise ValueError("Invalid raw synthesis output")
            duration = wav.size / model.sample_rate
            total_seconds += duration
            if duration > 120 or total_seconds > 300:
                raise ValueError("Raw generated audio exceeds reviewed artifact bounds")
            buffer = io.BytesIO()
            sf.write(buffer, wav, model.sample_rate, format="WAV", subtype="FLOAT")
            filename = job["id"] + ".wav"
            audio_files[filename] = buffer.getvalue()
            decision = plan_timing(phrase, duration, next_phrase=job["next_phrase"], video_duration=job["excerpt_end"])
            rows.append({**job, "synthesis_text": text, "file": filename,
                         "raw_duration": duration, "sample_rate": model.sample_rate,
                         "synthesis_seconds": time.monotonic() - before,
                         "peak": float(np.abs(wav).max()), "timing_decision_only": decision})
            emit("segment_complete", index=index + 1, total=len(jobs), raw_seconds=duration)
        return {"audio": audio_files, "report": {"status": "complete", "model": MODEL,
                "revision": REVISION, "package": PACKAGE, "hardware": "CPU", "cpu": 2,
                "memory_mib": 4096, "max_containers": 1, "smoke": smoke,
                "input_snapshot": data,
                "calls": len(rows), "wall_seconds": time.monotonic() - started,
                "processing": "Native synthesized speech only; no fit, trim, slowdown, subtitle rewrite or packaging",
                "grouping_candidate": {"max_seconds": 20, "max_characters": 500, "meaningful_gap": .25},
                "segments": rows}}
    finally:
        monitor.set()


@app.local_entrypoint()
def main(smoke: bool = False):
    import sys
    sys.path.insert(0, str(ROOT))
    from batch_dubbing.naturalness_experiment import build_jobs

    data = json.loads(INPUT.read_text())
    build_jobs(data)
    destination = OUTPUT / ("smoke" if smoke else "full")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {destination}")
    result = synthesize.remote(data, smoke)
    destination.mkdir(parents=True, exist_ok=False)
    for filename, content in result["audio"].items():
        if Path(filename).name != filename or not filename.endswith(".wav"):
            raise ValueError("Invalid returned artifact name")
        with (destination / filename).open("xb") as output:
            output.write(content)
    with (destination / "report.json").open("x") as output:
        json.dump(result["report"], output, ensure_ascii=False, indent=2)
        output.write("\n")
    print("RESULT", destination, flush=True)
