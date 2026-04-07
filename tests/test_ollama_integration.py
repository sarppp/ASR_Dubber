"""
test_ollama_integration.py
==========================
REAL integration tests — actually start and stop Docker/Ollama.
No mocks. Verifies the automatic lifecycle works end-to-end.

These replace waiting for a 5-minute translation run.
LangSmith traces each test so you can inspect timing/results in the UI.

Run:
    uv run --with pytest,langsmith pytest test_ollama_integration.py -v -s

With LangSmith:
    LANGCHAIN_TRACING_V2=true \\
    LANGCHAIN_API_KEY=<key> \\
    LANGCHAIN_PROJECT=asr-pipeline \\
    uv run --with pytest,langsmith pytest test_ollama_integration.py -v -s
"""

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

# ── LangSmith ────────────────────────────────────────────────────────────────
try:
    from langsmith import traceable
except ImportError:
    def traceable(name=None, **kwargs):
        def _d(fn): return fn
        return _d(name) if callable(name) else _d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import _DOCKER_PROC, _ollama_is_running, _ollama_start, _ollama_stop


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_container():
    """Kill any existing ollama container before and after every test."""
    subprocess.run(["docker", "rm", "-f", "ollama"], capture_output=True)
    # wait for port to be free
    for _ in range(5):
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1)
            time.sleep(1)
        except Exception:
            break
    yield
    subprocess.run(["docker", "rm", "-f", "ollama"], capture_output=True)


def _container_running() -> bool:
    r = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", "ollama"],
        capture_output=True, text=True,
    )
    return r.stdout.strip() == "true"


# ═════════════════════════════════════════════════════════════════════════════
# Integration tests
# ═════════════════════════════════════════════════════════════════════════════

class TestDockerLifecycle:

    @traceable(name="integration_ollama_auto_starts")
    def test_ollama_starts_automatically(self):
        """
        _ollama_start() must bring up the Docker container without any manual step.
        Proves the pipeline's translate step can self-manage Ollama.
        """
        assert not _ollama_is_running(), "Precondition: Ollama must be offline before test"

        proc = _ollama_start()

        assert _ollama_is_running(), "Ollama API must be reachable after _ollama_start()"
        assert _container_running(),  "Docker container must be running"
        assert proc == _DOCKER_PROC,  f"Expected _DOCKER_PROC sentinel, got {proc!r}"

        # cleanup
        _ollama_stop(proc)

    @traceable(name="integration_ollama_auto_stops")
    def test_ollama_stops_automatically(self):
        """
        _ollama_stop(_DOCKER_PROC) must stop the container — no manual 'docker stop'.
        """
        proc = _ollama_start()
        assert _ollama_is_running(), "Precondition: Ollama must be online"
        assert proc == _DOCKER_PROC

        _ollama_stop(proc)

        # Give Docker a moment to stop
        time.sleep(3)
        assert not _container_running(), "Container must be stopped after _ollama_stop()"
        assert not _ollama_is_running(), "Ollama API must be unreachable after stop"

    @traceable(name="integration_full_roundtrip")
    def test_full_start_stop_roundtrip(self):
        """
        Complete automatic lifecycle: offline → start → API up → stop → offline.
        This is exactly what the translate step does.
        """
        # Before
        assert not _ollama_is_running()

        # Start
        proc = _ollama_start()
        assert proc == _DOCKER_PROC
        assert _ollama_is_running()
        assert _container_running()
        print(f"\n✅ Ollama started automatically (proc={proc!r})")

        # Stop
        _ollama_stop(proc)
        time.sleep(3)
        assert not _container_running()
        assert not _ollama_is_running()
        print("✅ Ollama stopped automatically — no manual intervention needed")

    @traceable(name="integration_already_running_not_stopped")
    def test_does_not_stop_pre_existing_ollama(self):
        """
        If Ollama was already running before the pipeline started,
        _ollama_stop(None) must leave it alone.
        """
        # Start manually to simulate pre-existing Ollama
        subprocess.run([
            "docker", "run", "-d", "--name", "ollama",
            "-e", "OLLAMA_HOST=0.0.0.0",
            "-p", "11434:11434",
            "-v", "/home/sarpk/python-tools/.ollama_models:/root/.ollama",
            "ollama/ollama",
        ], capture_output=True)

        # Wait for it to be ready
        for _ in range(30):
            if _ollama_is_running():
                break
            time.sleep(1)
        assert _ollama_is_running(), "Pre-existing Ollama must be running"

        # Pipeline detects it's already running → returns None
        proc = _ollama_start()
        assert proc is None, "Must return None when Ollama was already running"

        # Stop must be a no-op
        _ollama_stop(proc)
        time.sleep(2)

        assert _container_running(), "Pre-existing container must still be running after stop"
        assert _ollama_is_running(), "Pre-existing Ollama must still be reachable after stop"
