"""
test_chunk_sizing.py — Unit tests for _compute_max_chunk_sec and _calibrate_chunk_size.

Key invariants tested:
  - Canary: always returns 60s (quality cap, not VRAM)
  - Parakeet / Qwen3-ASR: no arbitrary model cap — VRAM drives the limit
  - Calibration can project UPWARD on powerful GPUs (no initial_guess ceiling)
  - Calibration is bounded by audio_dur + 60, not by initial_guess
  - Floor is always 60s, absolute ceiling 7200s for initial estimate

VRAM scenarios: 6 GB, 16 GB, 24 GB, 48 GB
Audio durations: 10 min (600s), 30 min (1800s), 60 min (3600s)
Models: parakeet-v3, canary (translation), qwen3-asr
"""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Bootstrap: stub torch and nemo_asr so nemo_modal_app imports without GPU
# ---------------------------------------------------------------------------

torch_stub = types.ModuleType("torch")
torch_stub.cuda = types.SimpleNamespace(
    is_available=lambda: True,
    mem_get_info=lambda: (0, 0),
    empty_cache=lambda: None,
    memory_allocated=lambda: 0,
    max_memory_allocated=lambda: 0,
    reset_peak_memory_stats=lambda: None,
    OutOfMemoryError=MemoryError,
)
torch_stub.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=False))
torch_stub.bfloat16  = "bfloat16"
torch_stub.float16   = "float16"
torch_stub.float32   = "float32"
torch_stub.inference_mode = lambda: MagicMock(
    __enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)
)
torch_stub.compile = lambda m, **kw: m
sys.modules.setdefault("torch", torch_stub)

for mod in ("nemo", "nemo.collections", "nemo.collections.asr",
            "nemo.collections.asr.models",
            "nemo.collections.asr.parts",
            "nemo.collections.asr.parts.utils",
            "nemo.collections.asr.parts.utils.diarization_utils",
            "omegaconf", "soundfile"):
    sys.modules.setdefault(mod, types.ModuleType(mod))

modal_stub = types.ModuleType("modal")
modal_stub.App = MagicMock()
modal_stub.Image = MagicMock()
modal_stub.Volume = MagicMock()
modal_stub.gpu = MagicMock()
modal_stub.method = lambda *a, **kw: (lambda f: f)
modal_stub.enter   = lambda *a, **kw: (lambda f: f)
modal_stub.build   = lambda *a, **kw: (lambda f: f)

class _FakeFunc:
    def __init__(self, f): self._f = f
    def remote(self, *a, **kw): return self._f(*a, **kw)
    def __call__(self, *a, **kw): return self._f(*a, **kw)

modal_stub.function = lambda *a, **kw: (lambda f: _FakeFunc(f))
sys.modules.setdefault("modal", modal_stub)

qwen_asr_stub = types.ModuleType("qwen_asr")
qwen_asr_stub.Qwen3ASRModel = MagicMock()
sys.modules.setdefault("qwen_asr", qwen_asr_stub)

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from nemo_modal_app import _compute_max_chunk_sec, _calibrate_chunk_size, _fmt_dur  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PARAKEET   = "nvidia/parakeet-tdt-0.6b-v3"
CANARY     = "nvidia/canary-1b-v2"
QWEN3      = "Qwen/Qwen3-ASR-1.7B"
QWEN3_S    = "Qwen/Qwen3-ASR-0.6B"

T_10MIN  =  10 * 60
T_30MIN  =  30 * 60
T_60MIN  =  60 * 60

SAFETY  = 0.85
RESERVE = 1.5
OVERLAP = 2  # CHUNK_OVERLAP_SEC


def free_vram_mock(free_gb: float):
    def _inner():
        return free_gb, free_gb + 2.0
    return _inner


def _usable(free_gb):
    return max(0.0, free_gb - RESERVE) * SAFETY


# ===========================================================================
# 1. _compute_max_chunk_sec
# ===========================================================================

class TestComputeMaxChunkSec(unittest.TestCase):

    def _run(self, free_gb: float, model: str) -> int:
        with patch("nemo_modal_app._vram_gb", free_vram_mock(free_gb)):
            return _compute_max_chunk_sec(model, SAFETY, RESERVE)

    # ── Canary: always 60s (quality cap, not VRAM) ────────────────────────

    def test_canary_always_60s_regardless_of_vram(self):
        for free_gb in (6.0, 16.0, 24.0, 48.0):
            with self.subTest(free_gb=free_gb):
                self.assertEqual(self._run(free_gb, CANARY), 60,
                                 f"Canary must always return 60s (quality cap), got != 60 at {free_gb} GB")

    # ── Parakeet: no model cap, VRAM drives result ────────────────────────

    def test_parakeet_6gb(self):
        sec = self._run(6.0, PARAKEET)
        expected = int(_usable(6.0) / 0.28 * 60)  # ~819s
        self.assertEqual(sec, expected)
        self.assertGreater(sec, 600, "6 GB should exceed old 600s cap")

    def test_parakeet_16gb(self):
        sec = self._run(16.0, PARAKEET)
        expected = int(_usable(16.0) / 0.28 * 60)  # ~2641s
        self.assertEqual(sec, expected)
        self.assertGreater(sec, 600)

    def test_parakeet_24gb(self):
        sec = self._run(24.0, PARAKEET)
        expected = int(_usable(24.0) / 0.28 * 60)  # ~4098s
        self.assertEqual(sec, expected)
        self.assertGreater(sec, 600)

    def test_parakeet_48gb(self):
        sec = self._run(48.0, PARAKEET)
        # int(_usable(48)/0.28*60) = 8469s → capped at 7200 absolute ceiling
        self.assertEqual(sec, 7200)

    # ── Qwen3-ASR: no model cap, VRAM drives result ───────────────────────

    def test_qwen3_6gb(self):
        sec = self._run(6.0, QWEN3)
        expected = int(_usable(6.0) / 0.35 * 60)  # ~655s
        self.assertEqual(sec, expected)
        self.assertGreater(sec, 600, "6 GB Qwen3 should exceed old 600s cap")

    def test_qwen3_16gb(self):
        sec = self._run(16.0, QWEN3)
        expected = int(_usable(16.0) / 0.35 * 60)  # ~2112s
        self.assertEqual(sec, expected)
        self.assertGreater(sec, 1800, "16 GB Qwen3 should exceed old 1800s cap")

    def test_qwen3_24gb(self):
        sec = self._run(24.0, QWEN3)
        expected = int(_usable(24.0) / 0.35 * 60)  # ~3278s
        self.assertEqual(sec, expected)
        self.assertGreater(sec, 1800)

    def test_qwen3_48gb(self):
        sec = self._run(48.0, QWEN3)
        # int(39.525/0.35*60) = 6775s < 7200 → not capped
        expected = int(_usable(48.0) / 0.35 * 60)
        self.assertEqual(sec, expected)
        self.assertGreater(sec, 1800)

    def test_qwen3_small_same_formula(self):
        """qwen3-asr-s uses the same gb/min formula as the 1.7B variant."""
        sec6  = self._run(6.0, QWEN3_S)
        sec16 = self._run(16.0, QWEN3_S)
        self.assertEqual(sec6,  int(_usable(6.0)  / 0.35 * 60))
        self.assertEqual(sec16, int(_usable(16.0) / 0.35 * 60))

    # ── Edge cases ────────────────────────────────────────────────────────

    def test_zero_free_vram_returns_300(self):
        with patch("nemo_modal_app._vram_gb", lambda: (0.0, 16.0)):
            self.assertEqual(_compute_max_chunk_sec(PARAKEET, SAFETY, RESERVE), 300)

    def test_reserve_exceeds_free_returns_60(self):
        with patch("nemo_modal_app._vram_gb", lambda: (1.0, 16.0)):
            self.assertEqual(_compute_max_chunk_sec(PARAKEET, SAFETY, RESERVE), 60)

    def test_barely_usable_returns_at_least_30(self):
        with patch("nemo_modal_app._vram_gb", lambda: (1.6, 16.0)):
            self.assertGreaterEqual(_compute_max_chunk_sec(PARAKEET, SAFETY, RESERVE), 30)

    def test_7200s_absolute_ceiling(self):
        """Very large VRAM should be capped at 7200s, not unlimited."""
        with patch("nemo_modal_app._vram_gb", lambda: (100.0, 100.0)):
            self.assertEqual(_compute_max_chunk_sec(PARAKEET, SAFETY, RESERVE), 7200)

    def test_canary_ignores_vram(self):
        """Even with 100 GB free, Canary returns 60s."""
        with patch("nemo_modal_app._vram_gb", lambda: (100.0, 100.0)):
            self.assertEqual(_compute_max_chunk_sec(CANARY, SAFETY, RESERVE), 60)


# ===========================================================================
# 2. _calibrate_chunk_size
# ===========================================================================

class TestCalibrateChunkSize(unittest.TestCase):

    def _run(
        self,
        audio_dur_sec: int,
        initial_guess_sec: int,
        vram_delta_gb: float,
        free_after_gb: float,
    ) -> int:
        peak_bytes = int(vram_delta_gb * 1024 ** 3)
        cuda_stub = types.SimpleNamespace(
            is_available=lambda: True,
            reset_peak_memory_stats=lambda: None,
            memory_allocated=lambda: 0,
            max_memory_allocated=lambda: peak_bytes,
            empty_cache=lambda: None,
        )
        with patch("nemo_modal_app._audio_duration", return_value=float(audio_dur_sec)), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)), \
             patch("nemo_modal_app._transcribe_manifest", return_value=([], [], None)), \
             patch("nemo_modal_app._vram_gb", return_value=(free_after_gb, free_after_gb + 2.0)), \
             patch("nemo_modal_app.torch") as mock_torch:
            mock_torch.cuda = cuda_stub
            return _calibrate_chunk_size(
                model=MagicMock(),
                audio_path="/tmp/fake_audio.wav",
                model_name=PARAKEET,
                language="en",
                target_lang=None,
                initial_guess_sec=initial_guess_sec,
                reserve_gb=RESERVE,
                safety_factor=SAFETY,
            )

    # ── Core invariant: result bounded by audio_dur, not initial_guess ───

    def test_calibration_can_exceed_initial_guess_on_powerful_gpu(self):
        """
        On a powerful GPU: 30-min audio, initial_guess=819s (6GB estimate),
        but we're actually running on 16GB. Calibration should project above 819s.
        delta=0.3GB in 180s test → 0.00167 GB/s; free_after=13GB → usable=9.775
        → projected = 9.775/0.00167 = 5854s → capped at audio_dur+60=1860s
        """
        result = self._run(T_30MIN, initial_guess_sec=819, vram_delta_gb=0.3, free_after_gb=13.0)
        self.assertGreater(result, 819, "Calibration must exceed initial_guess when VRAM allows it")
        self.assertLessEqual(result, T_30MIN + 60)

    def test_calibration_bounded_by_audio_dur_not_initial_guess(self):
        """
        1hr audio with a low initial_guess (conservative GPU estimate).
        Calibration should be allowed to project upward, bounded by audio_dur+60.
        delta=0.5 GB / 360s test = 0.00139 GB/s (passes 0.001 threshold);
        projected >> initial_guess=600 → capped at audio_dur+60=3660s.
        """
        result = self._run(T_60MIN, initial_guess_sec=600, vram_delta_gb=0.5, free_after_gb=20.0)
        self.assertGreater(result, 600)
        self.assertLessEqual(result, T_60MIN + 60)

    def test_calibration_reduces_chunk_on_tight_vram(self):
        """If test clip ate a lot of VRAM, chunk must be smaller than initial."""
        # delta=1.5GB in 60s test → 0.025 GB/s; free_after=4.0 → usable=2.125
        # projected = 2.125/0.025 = 85s < initial 600s
        result = self._run(T_30MIN, initial_guess_sec=600, vram_delta_gb=1.5, free_after_gb=4.0)
        self.assertLess(result, 600)
        self.assertGreaterEqual(result, 60)

    def test_floor_is_60s(self):
        """Even with almost no free VRAM, result is never < 60."""
        result = self._run(T_30MIN, initial_guess_sec=600, vram_delta_gb=2.0, free_after_gb=1.6)
        self.assertEqual(result, 60)

    def test_tiny_delta_returns_initial_guess(self):
        """Negligible delta (< 0.001 GB/s) → skip calibration, return initial."""
        result = self._run(T_10MIN, initial_guess_sec=480, vram_delta_gb=0.0, free_after_gb=14.0)
        self.assertEqual(result, 480)

    def test_audio_dur_cap_prevents_oversize_chunk(self):
        """Projected could be huge; must be capped at audio_dur+60."""
        # delta=0.1 GB in 60s → gb_per_sec=0.00167 (passes 0.001 threshold)
        # free_after=20: usable=15.725 GB → projected=9415s >> T_10MIN+60=660s
        result = self._run(T_10MIN, initial_guess_sec=2000, vram_delta_gb=0.1, free_after_gb=20.0)
        self.assertLessEqual(result, T_10MIN + 60)

    def test_no_cuda_returns_initial_guess(self):
        with patch("nemo_modal_app.torch") as mock_torch:
            mock_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
            result = _calibrate_chunk_size(
                MagicMock(), "/tmp/fake.wav", PARAKEET, "en", None, 600, RESERVE, SAFETY,
            )
        self.assertEqual(result, 600)

    def test_ffmpeg_failure_returns_initial_guess(self):
        import subprocess
        with patch("nemo_modal_app._audio_duration", return_value=float(T_30MIN)), \
             patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")), \
             patch("nemo_modal_app.torch") as mock_torch:
            mock_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
            result = _calibrate_chunk_size(
                MagicMock(), "/tmp/fake.wav", PARAKEET, "en", None, 600, RESERVE, SAFETY,
            )
        self.assertEqual(result, 600)


# ===========================================================================
# 3. End-to-end scenarios: initial estimate → chunk count
# ===========================================================================

class TestEndToEndScenarios(unittest.TestCase):
    """
    For each (VRAM, model, audio_duration) scenario:
      1. initial_chunk from _compute_max_chunk_sec
      2. calibration skipped when audio ≤ initial*1.5 (fits comfortably)
      3. assert: chunk count is reasonable, no VRAM left idle
    """

    def _initial_chunk(self, free_gb, model):
        with patch("nemo_modal_app._vram_gb", free_vram_mock(free_gb)):
            return _compute_max_chunk_sec(model, SAFETY, RESERVE)

    def _n_chunks(self, audio_sec, chunk_sec):
        """Mirrors _chunk_audio logic."""
        if audio_sec <= chunk_sec + 5:
            return 1
        step = chunk_sec - OVERLAP
        return max(1, -(-audio_sec // step))  # ceiling division

    def _assert(self, label, free_gb, model, audio_sec, max_chunks, min_chunk_sec=None):
        chunk = self._initial_chunk(free_gb, model)
        n = self._n_chunks(audio_sec, chunk)
        self.assertLessEqual(n, max_chunks,
                             f"{label}: {n} chunks > {max_chunks} (chunk={chunk}s)")
        self.assertGreaterEqual(chunk, 30, f"{label}: initial chunk < 30s")
        if min_chunk_sec:
            self.assertGreaterEqual(chunk, min_chunk_sec,
                                    f"{label}: chunk {chunk}s below expected {min_chunk_sec}s")
        return chunk, n

    # ── Canary (all VRAM → always 60s, fixed) ─────────────────────────────

    def test_canary_always_60s(self):
        for free_gb in (6.0, 16.0, 24.0, 48.0):
            with self.subTest(free_gb=free_gb):
                chunk = self._initial_chunk(free_gb, CANARY)
                self.assertEqual(chunk, 60, f"Canary must be 60s at {free_gb}GB, got {chunk}")

    def test_canary_6gb_30min_chunk_count(self):
        # 60s chunks, 1800s audio → ceil(1800/58) = 32 chunks
        chunk, n = self._assert("canary 6GB 30min", 6.0, CANARY, T_30MIN, max_chunks=35)
        self.assertEqual(chunk, 60)

    def test_canary_16gb_30min_chunk_count(self):
        chunk, n = self._assert("canary 16GB 30min", 16.0, CANARY, T_30MIN, max_chunks=35)
        self.assertEqual(chunk, 60)

    # ── Parakeet: VRAM-driven, no model cap ──────────────────────────────

    def test_parakeet_6gb_10min_single_pass(self):
        """10min audio on 6GB: chunk ~819s > 600s → fits in 1 chunk."""
        chunk, n = self._assert("parakeet 6GB 10min", 6.0, PARAKEET, T_10MIN,
                                max_chunks=1, min_chunk_sec=600)

    def test_parakeet_6gb_30min(self):
        """30min audio on 6GB: chunk ~819s → 2-3 chunks (was 4 with old 600s cap)."""
        chunk, n = self._assert("parakeet 6GB 30min", 6.0, PARAKEET, T_30MIN,
                                max_chunks=3, min_chunk_sec=600)

    def test_parakeet_6gb_1hr(self):
        chunk, n = self._assert("parakeet 6GB 1hr", 6.0, PARAKEET, T_60MIN,
                                max_chunks=5, min_chunk_sec=600)

    def test_parakeet_16gb_10min_single_pass(self):
        """16GB: chunk ~2641s >> 600s → 10min fits in 1 chunk."""
        chunk, n = self._assert("parakeet 16GB 10min", 16.0, PARAKEET, T_10MIN,
                                max_chunks=1, min_chunk_sec=2000)

    def test_parakeet_16gb_30min_single_pass(self):
        """30min audio on 16GB: chunk ~2641s > 1800s → 1 chunk."""
        chunk, n = self._assert("parakeet 16GB 30min", 16.0, PARAKEET, T_30MIN,
                                max_chunks=1, min_chunk_sec=2000)
        self.assertEqual(n, 1, "30min should fit in a single chunk on 16GB")

    def test_parakeet_16gb_1hr(self):
        """1hr on 16GB: chunk ~2641s → 2 chunks (was 7 with old 600s cap)."""
        chunk, n = self._assert("parakeet 16GB 1hr", 16.0, PARAKEET, T_60MIN,
                                max_chunks=2, min_chunk_sec=2000)

    def test_parakeet_24gb_1hr_single_pass(self):
        """24GB: chunk ~4098s >> 3600s → 1hr fits in 1 chunk."""
        chunk, n = self._assert("parakeet 24GB 1hr", 24.0, PARAKEET, T_60MIN,
                                max_chunks=1, min_chunk_sec=3600)
        self.assertEqual(n, 1, "1hr should fit in a single chunk on 24GB")

    def test_parakeet_48gb_1hr_single_pass(self):
        """48GB: chunk = 7200s (ceiling) >> 3600s → 1hr in 1 chunk."""
        chunk, n = self._assert("parakeet 48GB 1hr", 48.0, PARAKEET, T_60MIN,
                                max_chunks=1, min_chunk_sec=7200)
        self.assertEqual(chunk, 7200)
        self.assertEqual(n, 1)

    # ── Qwen3-ASR: VRAM-driven, no model cap ─────────────────────────────

    def test_qwen3_6gb_10min_single_pass(self):
        """6GB: chunk ~655s > 600s → 10min in 1 chunk."""
        chunk, n = self._assert("qwen3 6GB 10min", 6.0, QWEN3, T_10MIN,
                                max_chunks=1, min_chunk_sec=600)

    def test_qwen3_6gb_30min(self):
        """6GB: chunk ~655s → 3 chunks for 30min (was 4 with old 600s)."""
        chunk, n = self._assert("qwen3 6GB 30min", 6.0, QWEN3, T_30MIN,
                                max_chunks=3, min_chunk_sec=600)

    def test_qwen3_16gb_30min_single_pass(self):
        """16GB: chunk ~2112s > 1800s → 30min fits in 1 chunk."""
        chunk, n = self._assert("qwen3 16GB 30min", 16.0, QWEN3, T_30MIN,
                                max_chunks=1, min_chunk_sec=2000)
        self.assertEqual(n, 1)

    def test_qwen3_16gb_1hr(self):
        """16GB: chunk ~2112s → 2 chunks for 1hr (was 3 with old 1800s cap)."""
        chunk, n = self._assert("qwen3 16GB 1hr", 16.0, QWEN3, T_60MIN,
                                max_chunks=2, min_chunk_sec=2000)

    def test_qwen3_24gb_1hr(self):
        """24GB: chunk ~3278s < 3600s → 2 chunks for 1hr (was 3 before, now 2)."""
        chunk, n = self._assert("qwen3 24GB 1hr", 24.0, QWEN3, T_60MIN,
                                max_chunks=2, min_chunk_sec=3200)
        # Note: single pass needs chunk > 3600s; 24GB gives ~3278s (use 48GB for 1-pass)

    def test_qwen3_48gb_1hr_single_pass(self):
        """48GB: chunk ~6775s >> 3600s → 1hr in 1 chunk."""
        chunk, n = self._assert("qwen3 48GB 1hr", 48.0, QWEN3, T_60MIN,
                                max_chunks=1, min_chunk_sec=6000)
        self.assertEqual(n, 1)

    # ── A10G realistic scenario (user's GPU, ~17GB free) ──────────────────

    def test_qwen3_a10g_30min_single_pass(self):
        """A10G (~22GB total, ~17GB free after model): 30min in 1 chunk."""
        chunk, n = self._assert("qwen3 A10G 30min", 17.0, QWEN3, T_30MIN,
                                max_chunks=1, min_chunk_sec=2000)
        self.assertEqual(n, 1)

    def test_parakeet_a10g_1hr_max_2_chunks(self):
        """A10G parakeet: chunk ~2900s → 1hr in 2 chunks (was 7)."""
        chunk, n = self._assert("parakeet A10G 1hr", 17.0, PARAKEET, T_60MIN,
                                max_chunks=2, min_chunk_sec=2500)


# ===========================================================================
# 4. _fmt_dur smoke test
# ===========================================================================

class TestFmtDur(unittest.TestCase):
    def test_seconds(self):
        self.assertEqual(_fmt_dur(45), "45.0s")

    def test_minutes(self):
        self.assertEqual(_fmt_dur(90), "1m30s")

    def test_exact_hour(self):
        self.assertEqual(_fmt_dur(3600), "60m00s")


if __name__ == "__main__":
    unittest.main(verbosity=2)
