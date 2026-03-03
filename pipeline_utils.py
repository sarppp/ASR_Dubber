"""
pipeline_utils.py — Path constants, basic helpers, and Ollama lifecycle
for the ASR dubbing pipeline.
"""

import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ── Folder layout ─────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
NEMO_CODE_DIR = ROOT / "nemo"                                         # code — always /app/nemo
NEMO_DIR      = Path(os.getenv("INPUT_DIR",  str(NEMO_CODE_DIR)))    # data  — /data/input in Docker
TRANSLATE_DIR = ROOT / "translate-gemma"
QWEN_DIR      = ROOT / "qwen3-tts"
WHISPER_DIR   = ROOT / "whisper"

NEMO_PY       = NEMO_CODE_DIR / ".venv" / "bin" / "python"
QWEN_PY       = QWEN_DIR           / ".venv" / "bin" / "python"
WHISPER_PY    = WHISPER_DIR        / ".venv" / "bin" / "python"
TRANSLATE_PY  = TRANSLATE_DIR      / ".venv" / "bin" / "python"
CLEAN_SUBS_SCRIPT = TRANSLATE_DIR / "clean_subs.py"
END_PRODUCT_DIR   = Path(os.getenv("OUTPUT_DIR", str(NEMO_DIR / "end_product")))

OLLAMA_HOST        = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_BIN         = os.getenv("OLLAMA_BIN", "ollama")
OLLAMA_DOCKER_IMAGE = os.getenv("OLLAMA_DOCKER_IMAGE", "ollama/ollama")
OLLAMA_MODELS_DIR  = os.getenv("OLLAMA_MODELS_DIR", str(Path.home() / "python-tools" / ".ollama_models"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n▶  {text}\n{bar}\n", flush=True)


def _run(cmd: list, cwd: Path, label: str) -> None:
    _banner(label)
    print(f"   cwd : {cwd}")
    print(f"   cmd : {' '.join(str(c) for c in cmd)}\n", flush=True)
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        print(f"\n❌  {label} failed (exit {result.returncode})", flush=True)
        sys.exit(result.returncode)
    print(f"\n✅  {label} done", flush=True)


def _python(venv_py: Path, fallback_dir: Path) -> list:
    """Return [python_executable] — venv directly, or uv run as fallback."""
    if venv_py.exists():
        return [str(venv_py)]
    print(f"⚠️  venv not found at {venv_py}, falling back to: uv run python")
    return ["uv", "run", "python"]


# ── Ollama lifecycle ──────────────────────────────────────────────────────────

def _ollama_is_running() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def _docker_available() -> bool:
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _ollama_start() -> subprocess.Popen | None:
    """
    Start Ollama — three strategies tried in order:
      1. Already running → do nothing
      2. Docker available → start ollama/ollama container (local machine)
      3. Native binary → ollama serve (RunPod / server)
    Returns Popen handle of what we started, or None if already running.
    """
    if _ollama_is_running():
        print("✓ Ollama already running", flush=True)
        return None

    # ── Strategy 1: Docker ────────────────────────────────────────────────────
    if _docker_available():
        print("🐳 Starting Ollama via Docker...", flush=True)
        # Check if container already exists but is stopped
        subprocess.run(["docker", "rm", "-f", "ollama"],
                       capture_output=True)
        cmd = [
            "docker", "run", "-d", "--rm",
            "--name", "ollama",
            "-e", "OLLAMA_HOST=0.0.0.0",
            "-e", "OLLAMA_FLASH_ATTENTION=1",
            "-p", "11434:11434",
        ]
        # Add GPU flags if nvidia-smi is available
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True)
            cmd += ["--runtime=nvidia", "--gpus", "all"]
            print("   GPU detected — enabling NVIDIA runtime", flush=True)
        except Exception:
            print("   No GPU detected — running Ollama on CPU", flush=True)
        cmd.append(OLLAMA_DOCKER_IMAGE)

        print(f"   Models dir : {OLLAMA_MODELS_DIR}", flush=True)
        if Path(OLLAMA_MODELS_DIR).exists():
            # Mount existing local models dir so Ollama finds them immediately
            cmd += ["-v", f"{OLLAMA_MODELS_DIR}:/root/.ollama"]
        else:
            # Remote/fresh machine — model is baked into the image or will be pulled
            print(f"   ℹ️  Models dir not found locally — Ollama will use image-baked or pull models", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Docker run -d exits immediately; the container runs in background
        # proc here is just the `docker run` CLI process, not the container
        started_via = "docker"

    # ── Strategy 2: Native binary (RunPod, servers) ───────────────────────────
    else:
        # Find ollama binary
        ollama_bin = OLLAMA_BIN
        if not Path(ollama_bin).exists():
            # Common install locations
            for candidate in ["/usr/local/bin/ollama", "/usr/bin/ollama",
                               str(Path.home() / ".local/bin/ollama")]:
                if Path(candidate).exists():
                    ollama_bin = candidate
                    break
            else:
                print("❌ Ollama not found. Install from https://ollama.com or set OLLAMA_BIN",
                      flush=True)
                sys.exit(1)

        print(f"🚀 Starting Ollama natively ({ollama_bin})...", flush=True)
        proc = subprocess.Popen(
            [ollama_bin, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        started_via = "native"

    # ── Wait for ready ────────────────────────────────────────────────────────
    for i in range(40):
        time.sleep(1)
        if _ollama_is_running():
            print(f"✓ Ollama ready via {started_via} (took {i+1}s)", flush=True)
            return proc
        if i % 5 == 0:
            print(f"   waiting for Ollama... ({i+1}/40)", flush=True)

    print("❌ Ollama did not start in time")
    sys.exit(1)


def _ollama_stop(proc: subprocess.Popen | None) -> None:
    """Stop what we started — Docker container or native process."""
    if proc is None:
        return
    print("🛑 Stopping Ollama...", flush=True)
    # If docker is available, stop the container by name (cleaner than killing proc)
    if _docker_available():
        subprocess.run(["docker", "stop", "ollama"],
                       capture_output=True, timeout=15)
        print("✓ Ollama container stopped")
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
            print("✓ Ollama stopped")
        except Exception as e:
            print(f"⚠️  Could not stop Ollama cleanly: {e}")
