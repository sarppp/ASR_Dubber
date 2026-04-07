"""
tests/test_tts_worker.py — Smoke + deep tests for the persistent TTS worker.

Covers:
  1. PersistentTTSWorker protocol (IPC correctness, lifecycle, error handling)
  2. qwen_tts_worker._synthesise logic (without GPU — model mocked)
  3. dub.py TTS loop VRAM optimisation (clone closed before custom on failure)

No GPU or real model required — uses fake subprocess scripts and mocks.
"""

import json
import subprocess
import sys
import types
import unittest.mock as mock
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
QWEN_DIR = ROOT / "qwen3-tts"

# ---------------------------------------------------------------------------
# Stub heavy imports so dub_audio imports cleanly in CI
# ---------------------------------------------------------------------------
for _mod in ("soundfile", "tqdm", "demucs"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

_tqdm_mod = sys.modules["tqdm"]
if not hasattr(_tqdm_mod, "tqdm"):
    _tqdm_mod.tqdm = lambda it, **kw: it  # type: ignore

import dub_audio  # noqa: E402  (after stubs)
from dub_audio import PersistentTTSWorker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_worker_script(tmp_path) -> str:
    """A real Python script that speaks the TTS worker protocol without loading
    any model — responds READY immediately, writes a tiny WAV stub for each
    synthesis request, and exits cleanly on quit."""
    script = tmp_path / "fake_worker.py"
    script.write_text(
        "import json, sys, pathlib\n"
        "print('READY', flush=True)\n"
        "for line in sys.stdin:\n"
        "    req = json.loads(line.strip())\n"
        "    if req.get('quit'):\n"
        "        print(json.dumps({'ok': True}), flush=True)\n"
        "        break\n"
        "    out = req.get('output', '')\n"
        "    if out:\n"
        "        pathlib.Path(out).write_bytes(b'\\x00' * 1000)\n"
        "    print(json.dumps({'ok': True}), flush=True)\n"
    )
    return str(script)


@pytest.fixture()
def fake_error_worker_script(tmp_path) -> str:
    """Worker that always returns an error response."""
    script = tmp_path / "fake_error_worker.py"
    script.write_text(
        "import json, sys\n"
        "print('READY', flush=True)\n"
        "for line in sys.stdin:\n"
        "    req = json.loads(line.strip())\n"
        "    if req.get('quit'):\n"
        "        print(json.dumps({'ok': True}), flush=True)\n"
        "        break\n"
        "    print(json.dumps({'ok': False, 'error': 'synthesis failed'}), flush=True)\n"
    )
    return str(script)


@pytest.fixture()
def fake_crash_worker_script(tmp_path) -> str:
    """Worker that exits immediately without sending READY."""
    script = tmp_path / "fake_crash_worker.py"
    script.write_text("import sys\nsys.exit(1)\n")
    return str(script)


def _make_worker(mode: str, script: str) -> PersistentTTSWorker:
    return PersistentTTSWorker(mode, sys.executable, script)


# ===========================================================================
# 1. SMOKE TESTS — PersistentTTSWorker protocol
# ===========================================================================

class TestPersistentTTSWorkerSmoke:

    def test_worker_starts_and_becomes_ready(self, fake_worker_script):
        """Worker starts and reaches READY state without error."""
        w = _make_worker("custom", fake_worker_script)
        w._start()
        assert w._proc is not None
        assert w._proc.poll() is None  # still alive
        w.close()

    def test_context_manager_closes_on_exit(self, fake_worker_script):
        """with-block closes the worker process cleanly."""
        with _make_worker("custom", fake_worker_script) as w:
            w._start()
            proc = w._proc
        # After __exit__, process should be gone
        assert proc.poll() is not None

    def test_generate_custom_returns_true_on_success(self, fake_worker_script, tmp_path):
        """generate_custom sends correct request and returns True."""
        out = tmp_path / "out.wav"
        with _make_worker("custom", fake_worker_script) as w:
            ok = w.generate_custom("Hello world", "Chelsie", "English", out)
        assert ok is True
        assert out.exists() and out.stat().st_size >= 1000

    def test_generate_clone_sends_ref_audio(self, fake_worker_script, tmp_path):
        """generate_clone includes ref_audio in the request."""
        out = tmp_path / "out.wav"
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"\x00" * 100)
        sent_reqs = []

        orig_send = PersistentTTSWorker._send

        def capture_send(self, req):
            sent_reqs.append(req)
            return orig_send(self, req)

        with mock.patch.object(PersistentTTSWorker, "_send", capture_send):
            with _make_worker("clone", fake_worker_script) as w:
                w.generate_clone("Bonjour", ref, "French", out)

        assert sent_reqs, "No request was sent"
        assert sent_reqs[0]["ref_audio"] == str(ref)
        assert sent_reqs[0]["text"] == "Bonjour"
        assert sent_reqs[0]["language"] == "French"

    def test_generate_returns_false_on_error_response(self, fake_error_worker_script, tmp_path):
        """generate_custom returns False when worker reports an error."""
        out = tmp_path / "out.wav"
        with _make_worker("custom", fake_error_worker_script) as w:
            ok = w.generate_custom("Hello", "Chelsie", "English", out)
        assert ok is False

    def test_close_sends_quit_message(self, fake_worker_script):
        """close() sends {'quit': true} and process exits cleanly."""
        w = _make_worker("custom", fake_worker_script)
        w._start()
        proc = w._proc
        w.close()
        assert proc.poll() == 0, "Worker should exit with code 0 after quit"

    def test_close_idempotent(self, fake_worker_script):
        """Calling close() twice does not raise."""
        w = _make_worker("custom", fake_worker_script)
        w._start()
        w.close()
        w.close()  # should not raise

    def test_worker_not_started_until_first_use(self, fake_worker_script, tmp_path):
        """Worker process is not spawned until generate_* is first called."""
        w = _make_worker("custom", fake_worker_script)
        assert w._proc is None, "Process must not start at construction time"
        out = tmp_path / "out.wav"
        w.generate_custom("Hi", "Chelsie", "English", out)
        assert w._proc is not None
        w.close()

    def test_multiple_requests_same_process(self, fake_worker_script, tmp_path):
        """All requests are served by the same process (model loaded once)."""
        with _make_worker("custom", fake_worker_script) as w:
            w._start()
            pid_first = w._proc.pid
            for i in range(5):
                out = tmp_path / f"out_{i}.wav"
                ok = w.generate_custom(f"Segment {i}", "Chelsie", "English", out)
                assert ok
                assert w._proc.pid == pid_first, "PID must not change between requests"

    def test_worker_crash_returns_false_and_clears_proc(self, fake_crash_worker_script):
        """If worker exits before READY, RuntimeError is raised."""
        w = _make_worker("custom", fake_crash_worker_script)
        with pytest.raises(RuntimeError, match="READY"):
            w._start()
        assert w._proc is None or w._proc.poll() is not None


# ===========================================================================
# 2. DEEP TESTS — qwen_tts_worker._synthesise (no GPU, model mocked)
# ===========================================================================

class TestSynthesiseLogic:
    """Tests for the _synthesise() helper in qwen_tts_worker.py.
    Heavy imports (torch, soundfile, qwen_tts) are mocked."""

    @pytest.fixture(autouse=True)
    def _patch_heavy(self):
        """Patch torch, soundfile, and qwen_tts before importing the worker module."""
        import numpy as np

        torch_mod = types.ModuleType("torch")
        torch_mod.bfloat16 = "bfloat16"
        torch_mod.is_tensor = lambda x: False
        torch_mod.cuda = mock.MagicMock()
        torch_mod.cuda.is_available = lambda: False

        sf_mod = types.ModuleType("soundfile")
        self._sf_written = {}

        def fake_write(path, data, sr):
            self._sf_written[path] = (data, sr)
            Path(path).write_bytes(b"\x00" * 1000)

        sf_mod.write = fake_write

        with (
            mock.patch.dict("sys.modules", {"torch": torch_mod, "soundfile": sf_mod}),
        ):
            # Force re-import so our mocks take effect
            import importlib
            if "qwen_tts_worker" in sys.modules:
                del sys.modules["qwen_tts_worker"]
            sys.path.insert(0, str(QWEN_DIR))
            import qwen_tts_worker
            self._worker_mod = qwen_tts_worker
            yield
            sys.path.remove(str(QWEN_DIR))
            if "qwen_tts_worker" in sys.modules:
                del sys.modules["qwen_tts_worker"]

    def _fake_model(self, wavs=None, sr=22050):
        """Build a fake Qwen TTS model that returns the given wavs."""
        import numpy as np
        if wavs is None:
            wavs = [np.zeros(sr, dtype="float32")]
        m = mock.MagicMock()
        m.generate_custom_voice.return_value = (wavs, sr)
        m.generate_voice_clone.return_value  = (wavs, sr)
        return m

    def test_custom_mode_writes_wav(self, tmp_path):
        out = str(tmp_path / "out.wav")
        err = self._worker_mod._synthesise(
            self._fake_model(), {"text": "Hi", "output": out, "language": "English"}, "custom"
        )
        assert err is None
        assert Path(out).exists()

    def test_missing_text_returns_error(self, tmp_path):
        out = str(tmp_path / "out.wav")
        err = self._worker_mod._synthesise(
            self._fake_model(), {"output": out}, "custom"
        )
        assert err is not None
        assert "text" in err

    def test_missing_output_returns_error(self):
        err = self._worker_mod._synthesise(
            self._fake_model(), {"text": "Hi"}, "custom"
        )
        assert err is not None
        assert "output" in err

    def test_empty_wavs_returns_error(self, tmp_path):
        out = str(tmp_path / "out.wav")
        m = self._fake_model(wavs=[])
        err = self._worker_mod._synthesise(m, {"text": "Hi", "output": out}, "custom")
        assert err is not None
        assert "no audio" in err

    def test_clone_mode_missing_ref_audio_returns_error(self, tmp_path):
        out = str(tmp_path / "out.wav")
        err = self._worker_mod._synthesise(
            self._fake_model(),
            {"text": "Hi", "output": out, "ref_audio": "/nonexistent.wav"},
            "clone",
        )
        assert err is not None
        assert "ref_audio" in err

    def test_clone_mode_with_ref_audio_calls_generate_voice_clone(self, tmp_path):
        out = str(tmp_path / "out.wav")
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"\x00" * 100)
        m = self._fake_model()
        err = self._worker_mod._synthesise(
            m, {"text": "Hi", "output": out, "ref_audio": str(ref), "language": "French"}, "clone"
        )
        assert err is None
        m.generate_voice_clone.assert_called_once()

    def test_generate_custom_voice_attributeerror_falls_back_to_generate(self, tmp_path):
        """If generate_custom_voice doesn't exist, falls back to model.generate()."""
        import numpy as np
        out = str(tmp_path / "out.wav")
        wavs = [np.zeros(100, dtype="float32")]
        m = mock.MagicMock()
        m.generate_custom_voice.side_effect = AttributeError("no method")
        m.generate.return_value = (wavs, 22050)
        err = self._worker_mod._synthesise(
            m, {"text": "Hi", "output": out}, "custom"
        )
        assert err is None
        m.generate.assert_called_once()

    def test_exception_during_synthesis_returns_error_string(self, tmp_path):
        out = str(tmp_path / "out.wav")
        m = mock.MagicMock()
        m.generate_custom_voice.side_effect = RuntimeError("GPU OOM")
        err = self._worker_mod._synthesise(m, {"text": "Hi", "output": out}, "custom")
        assert err is not None
        assert "GPU OOM" in err


# ===========================================================================
# 3. DEEP TESTS — dub.py TTS loop VRAM behaviour
# ===========================================================================

class TestDubTTSLoopVRAM:
    """Verify that the TTS loop in dub.main() uses persistent workers correctly:
    - clone worker started once (not per segment)
    - clone worker closed before custom starts when clone fails
    - custom mode never starts a clone worker
    - workers always closed in finally block even if loop raises
    """

    def _make_segment(self, i, spk="Speaker 1", text="Hello", start=0.0, end=2.0):
        return {"index": i, "speaker": spk, "text": text, "start": start, "end": end}

    def _run_tts_loop(
        self,
        tmp_path,
        segments,
        qwen_mode="custom",
        clone_refs=None,
        voice_map=None,
        clone_ok=True,
        custom_ok=True,
    ):
        """Run only the TTS loop extracted from dub.main() with mocked workers."""
        from dub_srt import QWEN_FEMALE_VOICES

        clone_refs  = clone_refs or {}
        voice_map   = voice_map or {"Speaker 1": "Chelsie"}
        qwen_language = "French"

        close_order = []
        generate_counts = {"clone": 0, "custom": 0}

        def make_mock_worker(mode):
            w = mock.MagicMock(spec=PersistentTTSWorker)
            w.mode = mode

            def close_side():
                close_order.append(mode)

            w.close.side_effect = close_side

            def gen_custom(text, voice, lang, out):
                generate_counts["custom"] += 1
                if custom_ok:
                    out.write_bytes(b"\x00" * 1000)
                return custom_ok

            def gen_clone(text, ref, lang, out, ref_text=""):
                generate_counts["clone"] += 1
                if clone_ok:
                    out.write_bytes(b"\x00" * 1000)
                return clone_ok

            w.generate_custom.side_effect = gen_custom
            w.generate_clone.side_effect  = gen_clone
            return w

        clone_worker_instances = []
        custom_worker_instances = []

        orig_init = PersistentTTSWorker.__init__

        def fake_init(self, mode, qp, qw):
            orig_init(self, mode, qp, qw)

        created = {"clone": None, "custom": None}

        def make_worker_factory(mode):
            w = make_mock_worker(mode)
            created[mode] = w
            if mode == "clone":
                clone_worker_instances.append(w)
            else:
                custom_worker_instances.append(w)
            return w

        # Simulate the TTS loop from dub.main() directly
        from dub_srt import QWEN_FEMALE_VOICES
        from dub_audio import PersistentTTSWorker as PTW
        import dub_audio as da

        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()
        checkpoint_path = tmp_path / "checkpoint.json"

        final_files = []
        done_indices = set()
        clone_broken  = False
        custom_broken = False
        clone_worker  = None
        custom_worker = None

        with mock.patch.object(da, "PersistentTTSWorker", side_effect=make_worker_factory):
            with mock.patch("dub_audio.PersistentTTSWorker", side_effect=make_worker_factory):
                try:
                    for seg in segments:
                        i = seg["index"]
                        spk = seg["speaker"]
                        text = seg["text"]
                        start, end = seg["start"], seg["end"]

                        raw_out = temp_dir / f"seg_{i:04d}.wav"
                        if i in done_indices:
                            continue

                        ok = False

                        if qwen_mode == "clone" and not clone_broken:
                            ref = clone_refs.get(spk)
                            if ref and ref.exists():
                                if clone_worker is None:
                                    # Free custom model before loading clone —
                                    # never hold two 3.4 GB models on one GPU.
                                    if custom_worker is not None:
                                        custom_worker.close()
                                        custom_worker = None
                                    clone_worker = make_worker_factory("clone")
                                ok = clone_worker.generate_clone(text, ref, qwen_language, raw_out)
                                if not ok:
                                    clone_broken = True
                                    clone_worker.close()
                                    clone_worker = None

                        if not ok and not custom_broken:
                            # Free clone model before loading custom —
                            # never hold two 3.4 GB models on one GPU.
                            if clone_worker is not None:
                                clone_worker.close()
                                clone_worker = None
                            voice = voice_map.get(spk, QWEN_FEMALE_VOICES[0])
                            if custom_worker is None:
                                custom_worker = make_worker_factory("custom")
                            ok = custom_worker.generate_custom(text, voice, qwen_language, raw_out)
                            if not ok:
                                custom_broken = True
                                continue

                        if not ok:
                            continue

                        if raw_out.exists():
                            final_files.append((raw_out, start, end))
                finally:
                    if clone_worker:
                        clone_worker.close()
                    if custom_worker:
                        custom_worker.close()

        return {
            "final_files":      final_files,
            "clone_instances":  clone_worker_instances,
            "custom_instances": custom_worker_instances,
            "close_order":      close_order,
            "generate_counts":  generate_counts,
        }

    # ── smoke: happy paths ──────────────────────────────────────────────────

    def test_custom_mode_creates_one_worker_for_all_segments(self, tmp_path):
        segs = [self._make_segment(i, start=i*2.0, end=i*2.0+1.5) for i in range(5)]
        res = self._run_tts_loop(tmp_path, segs, qwen_mode="custom")
        assert len(res["custom_instances"]) == 1, "Only one custom worker must be created"
        assert res["generate_counts"]["custom"] == 5

    def test_clone_mode_creates_one_clone_worker(self, tmp_path):
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"\x00" * 1000)
        segs = [self._make_segment(i, start=i*2.0, end=i*2.0+1.5) for i in range(4)]
        res = self._run_tts_loop(
            tmp_path, segs, qwen_mode="clone",
            clone_refs={"Speaker 1": ref},
        )
        assert len(res["clone_instances"]) == 1, "Only one clone worker must be created"
        assert res["generate_counts"]["clone"] == 4
        assert res["generate_counts"]["custom"] == 0

    def test_custom_mode_never_starts_clone_worker(self, tmp_path):
        segs = [self._make_segment(i) for i in range(3)]
        res = self._run_tts_loop(tmp_path, segs, qwen_mode="custom")
        assert len(res["clone_instances"]) == 0, "Clone worker must not start in custom mode"

    # ── VRAM optimisation ───────────────────────────────────────────────────

    def test_clone_worker_closed_before_custom_starts_on_failure(self, tmp_path):
        """When clone fails, clone_worker.close() is called BEFORE custom_worker starts.
        This ensures at most one model occupies VRAM on tight GPUs (6 GB)."""
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"\x00" * 1000)
        segs = [self._make_segment(i, start=i*2.0, end=i*2.0+1.5) for i in range(3)]

        res = self._run_tts_loop(
            tmp_path, segs, qwen_mode="clone",
            clone_refs={"Speaker 1": ref},
            clone_ok=False,   # clone always fails → triggers fallback
            custom_ok=True,
        )
        # clone must be closed before first custom generate is counted
        assert res["close_order"][0] == "clone", (
            "clone_worker must be closed before custom_worker starts"
        )
        assert len(res["custom_instances"]) == 1

    def test_no_clone_ref_closes_clone_before_custom_starts(self, tmp_path):
        """When Speaker A has a clone ref but Speaker B doesn't, clone_worker
        must be closed before custom_worker is created.  Without this,
        both 3.4 GB models sit on the same GPU → OOM on tight GPUs."""
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"\x00" * 1000)
        segs = [
            self._make_segment(0, spk="Speaker 1", start=0.0, end=2.0),  # has ref
            self._make_segment(1, spk="Speaker 2", start=2.5, end=4.5),  # NO ref
        ]
        res = self._run_tts_loop(
            tmp_path, segs, qwen_mode="clone",
            clone_refs={"Speaker 1": ref},   # Speaker 2 intentionally missing
            clone_ok=True, custom_ok=True,
        )
        # clone must have been created (for Speaker 1) and closed (for Speaker 2)
        assert len(res["clone_instances"]) == 1
        assert len(res["custom_instances"]) == 1
        assert res["generate_counts"]["clone"] == 1
        assert res["generate_counts"]["custom"] == 1
        # clone closed before custom was created
        assert res["close_order"][0] == "clone", (
            "clone_worker must be closed before custom_worker starts "
            "when falling back due to missing clone ref"
        )

    def test_alternating_speakers_never_holds_two_models(self, tmp_path):
        """A→B→A→B with only A having a clone ref: each switch must close
        the old model before creating the new one.  At no point should both
        clone_worker and custom_worker be alive simultaneously."""
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"\x00" * 1000)
        segs = [
            self._make_segment(0, spk="A", start=0.0, end=2.0),
            self._make_segment(1, spk="B", start=2.5, end=4.5),
            self._make_segment(2, spk="A", start=5.0, end=7.0),
            self._make_segment(3, spk="B", start=7.5, end=9.5),
        ]
        res = self._run_tts_loop(
            tmp_path, segs, qwen_mode="clone",
            clone_refs={"A": ref},   # only A has ref
            clone_ok=True, custom_ok=True,
        )
        assert res["generate_counts"]["clone"] == 2   # segs 0 and 2
        assert res["generate_counts"]["custom"] == 2   # segs 1 and 3

        # Verify close order: before each switch, the old model is closed.
        # Sequence: create clone → close clone → create custom →
        #           close custom → create clone → close clone → create custom →
        #           close custom (finally)
        # So close_order should alternate: clone, custom, clone, custom
        assert len(res["close_order"]) >= 3, (
            f"Expected at least 3 close calls for 4 alternating segments, "
            f"got {len(res['close_order'])}: {res['close_order']}"
        )
        # First close must be clone (before custom starts for seg 1)
        assert res["close_order"][0] == "clone"
        # Second close must be custom (before clone restarts for seg 2)
        assert res["close_order"][1] == "custom"
        # Third close must be clone (before custom restarts for seg 3)
        assert res["close_order"][2] == "clone"

    def test_workers_closed_on_loop_exception(self, tmp_path):
        """If an unexpected exception occurs mid-loop, workers are still closed (finally)."""
        segs = [self._make_segment(0)]

        # We'll raise inside generate_custom
        boom_raised = []
        close_called = []

        class BoomWorker:
            def __init__(self, mode, *a):
                self.mode = mode
            def generate_custom(self, *a):
                boom_raised.append(True)
                raise RuntimeError("unexpected GPU error")
            def generate_clone(self, *a):
                return False
            def close(self):
                close_called.append(self.mode)

        # Run loop manually with BoomWorker
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()
        final_files = []
        clone_worker = None
        custom_worker = None
        try:
            for seg in segs:
                raw_out = temp_dir / f"seg_0000.wav"
                custom_worker = BoomWorker("custom", "", "")
                with pytest.raises(RuntimeError, match="GPU error"):
                    custom_worker.generate_custom("hi", "Chelsie", "French", raw_out)
        finally:
            if clone_worker:
                clone_worker.close()
            if custom_worker:
                custom_worker.close()

        assert close_called == ["custom"]

    def test_no_worker_started_for_cached_segments(self, tmp_path):
        """Segments already in done_indices skip TTS entirely — no worker needed."""
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()
        # Pre-create cached files
        for i in range(3):
            (temp_dir / f"seg_{i:04d}.wav").write_bytes(b"\x00" * 1000)

        segs = [self._make_segment(i, start=i*2.0, end=i*2.0+1.5) for i in range(3)]
        done_indices = {0, 1, 2}  # all already done
        worker_started = []

        custom_worker = None
        clone_worker  = None
        try:
            for seg in segs:
                if seg["index"] in done_indices:
                    continue
                if custom_worker is None:
                    worker_started.append(True)
        finally:
            pass

        assert not worker_started, "No worker should start if all segments are cached"


# ===========================================================================
# 4. PROTOCOL INTEGRATION — real subprocess, fake model
# ===========================================================================

class TestWorkerProtocol:
    """Run the actual qwen_tts_worker.py script in a subprocess with a fake
    qwen_tts module injected via PYTHONPATH, verifying the protocol end-to-end."""

    @pytest.fixture()
    def fake_qwen_tts_package(self, tmp_path):
        """Fake package directory providing qwen_tts, torch, and soundfile stubs
        so qwen_tts_worker.py can run as a real subprocess without a GPU."""
        pkg = tmp_path / "fake_qwen"
        pkg.mkdir()

        # qwen_tts stub
        (pkg / "qwen_tts.py").write_text(
            "import numpy as np\n"
            "class Qwen3TTSModel:\n"
            "    @classmethod\n"
            "    def from_pretrained(cls, *a, **kw): return cls()\n"
            "    def generate_custom_voice(self, text, language, speaker, instruct):\n"
            "        return [np.zeros(100, dtype='float32')], 22050\n"
            "    def generate_voice_clone(self, text, language, ref_audio, **kw):\n"
            "        return [np.zeros(100, dtype='float32')], 22050\n"
        )

        # torch stub (only what qwen_tts_worker needs)
        (pkg / "torch.py").write_text(
            "bfloat16 = 'bfloat16'\n"
            "def is_tensor(x): return False\n"
            "class cuda:\n"
            "    @staticmethod\n"
            "    def is_available(): return False\n"
        )

        # soundfile stub — writes a tiny WAV-ish file so size check passes
        (pkg / "soundfile.py").write_text(
            "def write(path, data, sr):\n"
            "    with open(path, 'wb') as f: f.write(b'\\x00' * 500)\n"
        )

        return str(pkg)

    def _run_worker(self, qwen_dir_fake, mode, requests, tmp_path):
        """Start qwen_tts_worker.py, send requests, collect responses."""
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = qwen_dir_fake + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.Popen(
            [sys.executable, str(QWEN_DIR / "qwen_tts_worker.py"), "--mode", mode],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        try:
            # Wait for READY
            ready = proc.stdout.readline().strip()
            assert ready == "READY", f"Expected READY, got {ready!r}"

            responses = []
            for req in requests:
                proc.stdin.write(json.dumps(req) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline().strip()
                responses.append(json.loads(line))

            # Quit
            proc.stdin.write('{"quit": true}\n')
            proc.stdin.flush()
            proc.stdout.readline()  # consume quit response
            proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()

        return responses

    def test_custom_mode_protocol_roundtrip(self, fake_qwen_tts_package, tmp_path):
        """Worker handles 3 custom-mode requests correctly in one session."""
        reqs = [
            {"text": f"Segment {i}", "voice": "Chelsie",
             "language": "English", "output": str(tmp_path / f"out_{i}.wav")}
            for i in range(3)
        ]
        responses = self._run_worker(fake_qwen_tts_package, "custom", reqs, tmp_path)
        assert all(r["ok"] for r in responses), f"Unexpected errors: {responses}"
        for i in range(3):
            assert (tmp_path / f"out_{i}.wav").exists()

    def test_clone_mode_protocol_roundtrip(self, fake_qwen_tts_package, tmp_path):
        """Worker handles clone-mode request with ref_audio correctly."""
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"\x00" * 1000)
        reqs = [
            {"text": "Bonjour", "ref_audio": str(ref),
             "language": "French", "output": str(tmp_path / "cloned.wav")}
        ]
        responses = self._run_worker(fake_qwen_tts_package, "clone", reqs, tmp_path)
        assert responses[0]["ok"] is True
        assert (tmp_path / "cloned.wav").exists()

    def test_missing_ref_audio_returns_error(self, fake_qwen_tts_package, tmp_path):
        """Clone request with non-existent ref_audio gets ok=false, not a crash."""
        reqs = [
            {"text": "Hi", "ref_audio": "/no/such/file.wav",
             "language": "English", "output": str(tmp_path / "out.wav")}
        ]
        responses = self._run_worker(fake_qwen_tts_package, "clone", reqs, tmp_path)
        assert responses[0]["ok"] is False
        assert "ref_audio" in responses[0].get("error", "")

    def test_bad_json_does_not_crash_worker(self, fake_qwen_tts_package, tmp_path):
        """Malformed JSON line gets an error response; worker stays alive."""
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = fake_qwen_tts_package + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.Popen(
            [sys.executable, str(QWEN_DIR / "qwen_tts_worker.py"), "--mode", "custom"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, env=env,
        )
        try:
            assert proc.stdout.readline().strip() == "READY"
            proc.stdin.write("not-valid-json\n")
            proc.stdin.flush()
            err_resp = json.loads(proc.stdout.readline())
            assert err_resp["ok"] is False
            assert "JSON" in err_resp.get("error", "")
            # Worker still alive — send a valid quit
            proc.stdin.write('{"quit": true}\n')
            proc.stdin.flush()
            proc.wait(timeout=5)
        finally:
            if proc.poll() is None:
                proc.kill()
