"""
tests/test_nemo_modal_cuda_graphs.py — Unit tests for CUDA graph disabling in
nemo_modal_app._load_model.

Root cause (full chain):
  1. open_dict only un-freezes the top-level OmegaConf struct; setting
     model.cfg.decoding.greedy.use_cuda_graphs = False on the still-frozen nested
     struct raises ReadonlyConfigError → caught silently → change_decoding_strategy
     never called → old decoder still had use_cuda_graphs=True.

  2. Even after fixing (1) with to_container, NeMo's transcribe(timestamps=True)
     calls change_decoding_strategy(self.cfg.decoding) INTERNALLY, re-reading
     the ORIGINAL model.cfg.decoding which still had use_cuda_graphs=True.
     Evidence: "Using RNNT Loss : tdt" log appearing during inference (not loading).
     This re-captured a ~17 GB CUDA graph → OOM.

Fix: OmegaConf.update() modifies model.cfg.decoding IN PLACE (works on existing
keys in frozen struct configs without needing open_dict). NeMo's own internal
change_decoding_strategy(self.cfg.decoding) call now reads the patched config →
new decoder has use_cuda_graphs=False → no graph capture → no OOM.

Run:
    uv run --with pytest pytest tests/test_nemo_modal_cuda_graphs.py -v
"""
from __future__ import annotations

import sys
import types
import unittest.mock as mock
from unittest.mock import MagicMock, patch, call
import pytest

# ── Modal stub (must be before nemo_modal_app import) ─────────────────────────
if "modal" not in sys.modules:
    _modal = types.ModuleType("modal")
    _app_inst = MagicMock()
    _app_inst.function = lambda **kw: (lambda f: f)
    _app_inst.local_entrypoint = lambda: (lambda f: f)
    _modal.App = MagicMock(return_value=_app_inst)
    _modal.Image = MagicMock()
    sys.modules["modal"] = _modal

# ── omegaconf stub needs OmegaConf ────────────────────────────────────────────
_oc_mod = sys.modules.get("omegaconf")
if _oc_mod and not hasattr(_oc_mod, "OmegaConf"):
    _oc_mod.OmegaConf = MagicMock()

import nemo_modal_app  # noqa: E402


# ── Spec classes for hasattr control ──────────────────────────────────────────

class _ModelBase:
    cfg = None
    def eval(self): ...
    def to(self, *a, **kw): return self
    def named_children(self): return []


class _ModelWithStrategy(_ModelBase):
    def change_decoding_strategy(self, cfg): ...


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_torch(cuda: bool = True, free_gb: float = 20.0, total_gb: float = 22.0):
    t = MagicMock()
    t.cuda.is_available.return_value = cuda
    t.cuda.mem_get_info.return_value = (free_gb * 1024 ** 3, total_gb * 1024 ** 3)
    t.cuda.is_bf16_supported.return_value = True
    t.bfloat16 = "bfloat16"
    t.float16 = "float16"
    t.float32 = "float32"
    return t


def _make_model(with_strategy: bool = True):
    spec = _ModelWithStrategy if with_strategy else _ModelBase
    model = MagicMock(spec=spec)
    model.to.return_value = model
    model.eval.return_value = model
    model.named_children.return_value = []
    return model


def _run_load(model, device: str = "cuda"):
    """Call _load_model with all GPU/NeMo/OmegaConf deps mocked."""
    mt = _mock_torch(cuda=(device == "cuda"))
    mock_nemo = MagicMock()
    mock_nemo.models.ASRModel.from_pretrained.return_value = model

    # Reset OmegaConf.update mock before each run
    oc = sys.modules["omegaconf"].OmegaConf
    oc.update.reset_mock()
    oc.update.side_effect = None  # no exception by default

    nemo_modal_app.torch = mt
    nemo_modal_app.nemo_asr = mock_nemo

    with patch("nemo_modal_app.gc"), \
         patch("nemo_modal_app.time") as mock_time:
        mock_time.perf_counter.return_value = 0.0
        return nemo_modal_app._load_model("nvidia/parakeet-tdt-0.6b-v3", "bf16", device)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. OmegaConf.update called IN PLACE on model.cfg.decoding
#    (this is what ensures NeMo's own transcribe() internal call also gets
#    use_cuda_graphs=False — to_container approach missed this)
# ═══════════════════════════════════════════════════════════════════════════════

EXPECTED_CUDA_GRAPH_KEYS = {
    # Confirmed from parakeet-v3 config dump printed during real run:
    "greedy.use_cuda_graph_decoder",  # main TDT decoder loop graph
    "greedy.loop_labels",             # label-loop CUDA graph variant
    "beam.allow_cuda_graphs",         # beam decoder graphs
    # Generic fallbacks for other model variants:
    "use_cuda_graphs",
    "greedy.use_cuda_graphs",
}


class TestCudaGraphsDisabledViaUpdate:

    def _updated_keys(self) -> set:
        oc = sys.modules["omegaconf"].OmegaConf
        return {c.args[1] for c in oc.update.call_args_list}

    def test_all_parakeet_v3_keys_targeted(self):
        """Every key confirmed from the real parakeet-v3 config dump must be updated."""
        model = _make_model()
        _run_load(model, device="cuda")
        updated = self._updated_keys()
        parakeet_keys = {
            "greedy.use_cuda_graph_decoder",
            "greedy.loop_labels",
            "beam.allow_cuda_graphs",
        }
        assert parakeet_keys.issubset(updated), (
            f"Missing keys: {parakeet_keys - updated}\n"
            f"Got: {updated}\n"
            "These are the ACTUAL keys in parakeet-v3 config — wrong key names "
            "silently pass through OmegaConf.update's per-key try/except, leaving "
            "CUDA graphs enabled and causing 17 GB graph capture + OOM."
        )

    def test_no_key_sets_true(self):
        """Every OmegaConf.update call must pass False — never True."""
        model = _make_model()
        _run_load(model, device="cuda")
        oc = sys.modules["omegaconf"].OmegaConf
        for c in oc.update.call_args_list:
            assert c.args[2] is False, f"Key {c.args[1]} was set to {c.args[2]}, not False"

    def test_update_targets_live_cfg_not_a_copy(self):
        """OmegaConf.update must modify model.cfg.decoding in place, not a detached copy.

        NeMo's transcribe(timestamps=True) calls change_decoding_strategy(self.cfg.decoding)
        internally. If we modify a copy, NeMo's re-read of self.cfg.decoding still has
        use_cuda_graph_decoder=True and re-captures the 17 GB CUDA graph.
        """
        model = _make_model()
        _run_load(model, device="cuda")
        oc = sys.modules["omegaconf"].OmegaConf
        for c in oc.update.call_args_list:
            assert c.args[0] is model.cfg.decoding, (
                "OmegaConf.update received a copy, not model.cfg.decoding. "
                "NeMo's internal change_decoding_strategy(self.cfg.decoding) "
                "will re-read the original unpatched config."
            )

    def test_change_decoding_strategy_called_with_live_cfg(self):
        """Our explicit change_decoding_strategy call also uses the live cfg object."""
        model = _make_model()
        _run_load(model, device="cuda")
        model.change_decoding_strategy.assert_called_once_with(model.cfg.decoding)

    def test_model_returned(self):
        model = _make_model()
        result = _run_load(model, device="cuda")
        assert result is model


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Skipped cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestCudaGraphsSkipped:

    def test_skipped_on_cpu(self):
        """CPU path: OmegaConf.update and change_decoding_strategy not called."""
        model = _make_model()
        _run_load(model, device="cpu")
        oc = sys.modules["omegaconf"].OmegaConf
        oc.update.assert_not_called()
        model.change_decoding_strategy.assert_not_called()

    def test_no_change_decoding_strategy_when_model_lacks_it(self):
        """Model without change_decoding_strategy: OmegaConf.update runs, no crash."""
        model = _make_model(with_strategy=False)
        _run_load(model, device="cuda")   # must not raise
        # OmegaConf.update should still run (patching cfg matters regardless)
        oc = sys.modules["omegaconf"].OmegaConf
        assert oc.update.called


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Fallbacks — individual key failures are silently swallowed
# ═══════════════════════════════════════════════════════════════════════════════

class TestCudaGraphsFallbacks:

    def test_one_key_absent_other_still_applied(self):
        """If one key path doesn't exist in the config, the other is still updated.

        OmegaConf.update raises on non-existent keys in struct configs.
        The inner try/except per key must let the second update proceed.
        """
        model = _make_model()
        oc = sys.modules["omegaconf"].OmegaConf

        # First two calls raise (keys absent); remaining succeed
        n_keys = len(nemo_modal_app._CUDA_GRAPH_KEYS)
        oc.update.side_effect = [Exception("key not found"), Exception("key not found")] + [None] * (n_keys - 2)
        _run_load(model, device="cuda")   # must not raise
        assert oc.update.call_count == n_keys

    def test_warns_when_change_decoding_strategy_raises(self, capsys):
        """If change_decoding_strategy raises, warning printed, model still returned."""
        model = _make_model()
        model.change_decoding_strategy.side_effect = RuntimeError("decoder rebuild failed")
        result = _run_load(model, device="cuda")
        assert result is model
        out = capsys.readouterr().out
        assert "Warning" in out or "could not" in out.lower()

    def test_warns_when_omegaconf_import_fails(self, capsys):
        """If OmegaConf import itself fails, warning printed, model still returned."""
        model = _make_model()
        mt = _mock_torch()
        mock_nemo = MagicMock()
        mock_nemo.models.ASRModel.from_pretrained.return_value = model
        nemo_modal_app.torch = mt
        nemo_modal_app.nemo_asr = mock_nemo
        # Can't use _run_load here — it accesses sys.modules["omegaconf"] directly.
        # Call _load_model directly with omegaconf removed from sys.modules.
        with patch.dict(sys.modules, {"omegaconf": None}), \
             patch("nemo_modal_app.gc"), \
             patch("nemo_modal_app.time") as mock_time:
            mock_time.perf_counter.return_value = 0.0
            result = nemo_modal_app._load_model("nvidia/parakeet-tdt-0.6b-v3", "bf16", "cuda")
        assert result is model
        out = capsys.readouterr().out
        assert "Warning" in out or "could not" in out.lower()
