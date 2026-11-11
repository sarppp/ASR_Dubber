"""Prepared cached CPU excerpt probe. Import declares Modal; inference is remote-only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
MAX_AUDIO_BYTES = 63 * 1024**2
MAX_REPORT_BYTES = 1024**2
image = (modal.Image.from_id("im-40qUTZlx5ttgX1nwrQdvyw")
         .add_local_file(ROOT / "batch_dubbing/cheap_native_probe.py", "/work/batch_dubbing/cheap_native_probe.py")
         .add_local_file(ROOT / "batch_dubbing/native_excerpt_probe.py", "/work/batch_dubbing/native_excerpt_probe.py")
         .add_local_file(ROOT / "speech_model_benchmark_2026-10-27/code/telemetry.py", "/work/telemetry.py"))
volume = modal.Volume.from_name("speech-model-benchmark-20261027")
app = modal.App("native-excerpt-comparison")


def save_result(result, destination):
    """Verify all transferred bytes, then preserve complete or partial results."""
    metadata = {key: value for key, value in result.items() if key != "audio"}
    audio = result.get("audio")
    records = metadata.get("samples", []) + metadata.get("sequences", [])
    expected = {}
    for row in records:
        filename = row.get("file")
        if (not isinstance(filename, str) or Path(filename).name != filename
                or not filename.endswith(".wav") or filename in expected):
            raise ValueError("Invalid or duplicated output filename")
        expected[filename] = row.get("sha256")
    encoded = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if (not isinstance(audio, dict) or set(audio) != set(expected)
            or any(not isinstance(content, bytes) for content in audio.values())
            or sum(map(len, audio.values())) > MAX_AUDIO_BYTES or len(encoded.encode()) > MAX_REPORT_BYTES
            or any(hashlib.sha256(content).hexdigest() != expected[name] for name, content in audio.items())):
        raise ValueError("Returned excerpt integrity or size failure")
    # Destination was reserved before the paid call; files are always exclusive.
    destination = Path(destination)
    if not destination.is_dir() or any(destination.iterdir()):
        raise FileExistsError("Result directory must be reserved and empty")
    for name, content in audio.items():
        with (destination / name).open("xb") as handle:
            handle.write(content)
    with (destination / "report.json").open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    if metadata.get("status") != "complete":
        raise RuntimeError("Excerpt probe failed; partial report/audio retained; no retry")


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), timeout=150,
              startup_timeout=30, scaledown_window=2, min_containers=0,
              max_containers=1, retries=0, volumes={"/cache": volume})
def probe(data, smoke=False, smoke_artifact=None):
    import os
    import sys
    import time
    from importlib.metadata import version
    os.environ.update(HF_HOME="/cache/huggingface", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    sys.path.insert(0, "/work")
    from telemetry import emit, start
    emit("worker_ready", hardware="CPU", max_containers=1, smoke=smoke)
    monitor = start()
    started = time.monotonic()
    try:
        from batch_dubbing.native_excerpt_probe import MODEL, REVISION, plan_jobs, validate_smoke, run_excerpt
        if version("supertonic") != "1.3.1":
            raise ValueError("Cached image has unexpected Supertonic version")
        from supertonic.utils import chunk_text
        # Entire scope and actual chunk counts are checked before loading any model.
        plan_jobs(data, smoke, chunker=chunk_text)
        if smoke and smoke_artifact is not None:
            raise ValueError("Smoke cannot consume another result")
        if not smoke:
            validate_smoke(smoke_artifact, data)
        import numpy as np
        from huggingface_hub import snapshot_download
        from supertonic import TTS
        model_dir = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
        model = TTS(model="supertonic-3", model_dir=model_dir, auto_download=False,
                    intra_op_num_threads=2, inter_op_num_threads=1)

        def synthesize(row):
            np.random.seed(row["seed"])
            emit("segment_start", id=row["id"], language=row["language"])
            audio, _ = model.synthesize(row["text"], voice_style=model.get_voice_style(row["voice"]),
                                       lang=row["language"], speed=1.0, total_steps=8,
                                       max_chunk_length=300, silence_duration=.3)
            audio = np.asarray(audio, dtype=np.float32).squeeze()
            if audio.ndim != 1 or not 0 < audio.size <= 120 * 44100:
                raise ValueError("Invalid or overlong native audio")
            emit("segment_complete", id=row["id"], frames=int(audio.size))
            return {"samples": audio.astype("<f4", copy=False).tobytes(), "sample_rate": model.sample_rate}

        result = run_excerpt(data, smoke, chunker=chunk_text, synthesize=synthesize, smoke_artifact=smoke_artifact)
        result["worker_seconds"] = time.monotonic() - started
        emit("worker_complete", status=result["status"], calls=result.get("calls"))
        return result
    finally:
        monitor.set()


@app.local_entrypoint()
def main(input_file: str, output_root: str, smoke: bool = False, smoke_result: str = ""):
    import sys
    sys.path.insert(0, str(ROOT))
    from batch_dubbing.native_excerpt_probe import validate_scope, load_smoke
    path = Path(input_file)
    if path.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError("Excerpt input exceeds one MiB")
    data = validate_scope(json.loads(path.read_text()))
    if smoke and smoke_result:
        raise ValueError("Smoke cannot consume another result")
    if not smoke and not smoke_result:
        raise ValueError("Workload requires --smoke-result report.json")
    artifact = None if smoke else load_smoke(smoke_result, data)
    destination = Path(output_root) / ("smoke" if smoke else "workload")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        result = probe.remote(data, smoke, artifact)
        save_result(result, destination)
    except Exception as error:
        # Remote startup/model exceptions still leave a local diagnostic receipt.
        if not (destination / "report.json").exists():
            with (destination / "report.json").open("x", encoding="utf-8") as handle:
                json.dump(dict(status="failed", stage="smoke" if smoke else "workload", input=data,
                               error_type=type(error).__name__, error=str(error)[:2000],
                               inference_call_count_known=False, naturalness_verified=False), handle, ensure_ascii=False, indent=2)
        raise
