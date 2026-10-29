"""Prepared CPU-only probe using an existing image and cached model exclusively."""
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "dubbing_naturalness_2026-10-28/cheap_native_inputs.json"
OUTPUT = ROOT / "dubbing_naturalness_2026-10-28/cheap_native_trial"
# Existing successful grouping-trial image. No apt/pip/image build or fallback.
image = (modal.Image.from_id("im-40qUTZlx5ttgX1nwrQdvyw")
         .add_local_file(ROOT / "batch_dubbing/cheap_native_probe.py", "/work/batch_dubbing/cheap_native_probe.py"))
volume = modal.Volume.from_name("speech-model-benchmark-20261027")
app = modal.App("cheap-native-speed-probe")


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), timeout=90,
              startup_timeout=30, scaledown_window=2, min_containers=0,
              max_containers=1, retries=0, volumes={"/cache": volume})
def probe(data, smoke=False):
    import io
    import os
    import sys
    import time
    from importlib.metadata import version

    # No downloads; a missing image/model is a cheap failure, not a rebuild.
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    sys.path.insert(0, "/work")
    from batch_dubbing.cheap_native_probe import MODEL, REVISION, run_probe, validate_input
    validate_input(data)
    if version("supertonic") != "1.3.1":
        raise ValueError("Cached image has unexpected Supertonic version")
    print('MC_EVENT ' + json.dumps({"kind": "worker_ready", "hardware": "CPU", "max_containers": 1}), flush=True)
    started = time.monotonic()
    import numpy as np
    import soundfile as sf
    from huggingface_hub import snapshot_download
    from supertonic import TTS
    from supertonic.utils import chunk_text

    model_dir = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    model = TTS(model="supertonic-3", model_dir=model_dir, auto_download=False,
                intra_op_num_threads=2, inter_op_num_threads=1)

    def synthesize(row):
        if len(chunk_text(row["text"], 300)) != 1:
            raise ValueError("Probe text would introduce automatic chunk pauses")
        np.random.seed(20261029)
        audio, _ = model.synthesize(row["text"], voice_style=model.get_voice_style(row["voice"]),
                                    lang=row["language"], speed=1.0, total_steps=8,
                                    max_chunk_length=300, silence_duration=0.0)
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        if (audio.ndim != 1 or not 0 < audio.size <= model.sample_rate * 12
                or not np.isfinite(audio).all()):
            raise ValueError("Invalid or overlong generated native audio")
        stream = io.BytesIO()
        sf.write(stream, audio, model.sample_rate, format="WAV", subtype="FLOAT")
        print('MC_EVENT ' + json.dumps({"kind": "segment_complete", "id": row["id"]}), flush=True)
        return {"audio": stream.getvalue(), "sample_count": int(audio.size),
                "sample_rate": model.sample_rate, "peak": float(np.abs(audio).max())}

    result = run_probe(data, smoke, synthesize=synthesize)
    result["worker_seconds"] = time.monotonic() - started
    return result


@app.local_entrypoint()
def main(smoke: bool = False):
    import hashlib
    import sys
    sys.path.insert(0, str(ROOT))
    from batch_dubbing.cheap_native_probe import validate_input
    data = validate_input(json.loads(INPUT.read_text()))
    destination = OUTPUT / ("smoke" if smoke else "workload")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if not smoke:
        prior = json.loads((OUTPUT / "smoke/report.json").read_text())
        if prior.get("status") != "complete" or prior.get("input") != data or prior.get("calls") != 1:
            raise ValueError("Workload requires the successful matching smoke report")
    result = probe.remote(data, smoke)
    audio = result.pop("audio")
    expected = {row["file"]: row["sha256"] for row in result["samples"]}
    if set(audio) != set(expected) or any(hashlib.sha256(content).hexdigest() != expected[name]
                                          for name, content in audio.items()):
        raise ValueError("Returned audio integrity failure")
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in audio.items():
        if Path(name).name != name or not name.endswith(".wav"):
            raise ValueError("Invalid output filename")
        with (destination / name).open("xb") as handle:
            handle.write(content)
    with (destination / "report.json").open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("RESULT", destination, flush=True)
    if result.get("status") != "complete":
        raise RuntimeError("Probe failed; partial audio and report retained; no retry")
