import modal, json, subprocess
from pathlib import Path

app = modal.App("modal-control-gpu-health")


@app.function(
    serialized=True,
    image=modal.Image.debian_slim(python_version="3.11"),
    gpu="T4",
    cpu=4,
    memory=32768,
    timeout=60,
    max_containers=1,
)
def probe():
    print('MC_EVENT {"kind":"worker_ready"}', flush=True)
    return {
        "status": "complete",
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], text=True
        ),
    }


@app.local_entrypoint()
def main(smoke: bool = False):
    result = probe.remote()
    p = (
        Path(__file__).resolve().parent
        / ".state"
        / ("gpu_probe_smoke.json" if smoke else "gpu_probe_full.json")
    )
    p.write_text(json.dumps(result))
    print(result)
