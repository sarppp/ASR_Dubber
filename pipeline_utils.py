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
      2. Docker available → start ollama/ollama container; fall back to native on failure
      3. Native binary → ollama serve (RunPod / server)
    Returns Popen handle of what we started, or None if already running / Docker detached.
    """
    if _ollama_is_running():
        print("✓ Ollama already running", flush=True)
        return None

    proc: subprocess.Popen | None = None
    started_via = "unknown"

    # ── Strategy 1: Docker ────────────────────────────────────────────────────
    docker_ok = False
    if _docker_available():
        print("🐳 Starting Ollama via Docker...", flush=True)
        subprocess.run(["docker", "rm", "-f", "ollama"], capture_output=True)
        # No --rm: if the container crashes we need it to stay so we can read its logs.
        # We clean it up ourselves in _ollama_stop or on the next run (docker rm -f above).
        cmd = [
            "docker", "run", "-d",
            "--name", "ollama",
            "-e", "OLLAMA_HOST=0.0.0.0",
            "-e", "OLLAMA_FLASH_ATTENTION=1",
            "-p", "11434:11434",
        ]
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True)
            cmd += ["--gpus", "all"]
            print("   GPU detected — enabling GPU passthrough", flush=True)
        except Exception:
            print("   No GPU detected — running Ollama on CPU", flush=True)

        print(f"   Models dir : {OLLAMA_MODELS_DIR}", flush=True)
        if Path(OLLAMA_MODELS_DIR).exists():
            cmd += ["-v", f"{OLLAMA_MODELS_DIR}:/root/.ollama"]
        else:
            print("   ℹ️  Models dir not found locally — Ollama will pull models as needed", flush=True)

        cmd.append(OLLAMA_DOCKER_IMAGE)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            err = (result.stderr or result.stdout).strip()
            print(f"⚠️  Docker failed (rc={result.returncode}): {err}", flush=True)
            print("   Falling back to native ollama binary...", flush=True)
        else:
            # Wait briefly then verify container is still alive
            time.sleep(8)  # Give Ollama more time to fully start
            check = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", "ollama"],
                capture_output=True, text=True,
            )
            if check.stdout.strip() == "true":
                docker_ok = True
                started_via = "docker"
            else:
                logs = subprocess.run(
                    ["docker", "logs", "--tail", "30", "ollama"],
                    capture_output=True, text=True,
                )
                err = (logs.stderr or logs.stdout).strip()[-400:]
                subprocess.run(["docker", "rm", "-f", "ollama"], capture_output=True)
                print(f"⚠️  Ollama container exited immediately:\n{err}", flush=True)
                print("   Falling back to native ollama binary...", flush=True)

    # ── Strategy 2: Native binary (RunPod, servers, Docker fallback) ──────────
    if not docker_ok:
        ollama_bin = OLLAMA_BIN
        if not Path(ollama_bin).exists():
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
