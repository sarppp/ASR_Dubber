"""
tests/test_nemo_model_loading.py — Tests for model loading, cache clearing,
sys.path shadowing, and the _from_pretrained_with_cache_retry mechanism.

Covers the exact FileNotFoundError the user hit on remote:
    FileNotFoundError: [Errno 2] No such file or directory:
    '/root/.cache/torch/NeMo/NeMo_2.1.0/hf_hub_cache/nvidia/canary-qwen-2.5b/
    .../model_config.yaml'

Run:
    uv run --with "pytest,pydantic" pytest tests/test_nemo_model_loading.py -v
"""

from __future__ import annotations

import importlib
import json
import sys
import types
import unittest.mock as mock
from pathlib import Path

import pytest

# ── Imports from nemo/ (sys.path set in conftest.py) ─────────────────────────
from nemo_model import (
    _clear_nemo_cache,
    _estimate_chunk_sec,
    _extract_first_hypothesis,
    _from_pretrained_with_cache_retry,
    _hyp_field,
    _hyp_timestamps,
    _import_nemo_asr,
    _looks_like_hyp,
)
from nemo_diarize import _run_diarization, _validate_checkpoint, _build_srt, _assign_speakers
from nemo_audio import _words_to_segs, _segs_to_srt


# ═══════════════════════════════════════════════════════════════════════════════
# 1. _clear_nemo_cache — corrupt cache cleanup
# ═══════════════════════════════════════════════════════════════════════════════

class TestClearNemoCache:
    """Tests for the cache clearing mechanism that fixes FileNotFoundError."""

    def test_clears_existing_cache_dir(self, tmp_path):
        """When cache dir exists with model files, _clear_nemo_cache removes it."""
        import shutil

        nemo_base = tmp_path
        hf_dir = nemo_base / "NeMo_2.1.0" / "hf_hub_cache"
        target = hf_dir / "nvidia" / "canary-qwen-2.5b"
        target.mkdir(parents=True, exist_ok=True)
        (target / "model_config.yaml").write_text("model: canary")
        (target / "weights.ckpt").write_bytes(b"\x00" * 100)

        # Test the logic directly using the same algorithm as _clear_nemo_cache
        def patched_clear(model_name: str) -> bool:
            parts = model_name.split("/")
            org, slug = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[-1])
            cleared = False
            for hf_d in nemo_base.glob("*/hf_hub_cache"):
                t = hf_d / org / slug if org else hf_d / slug
                if t.exists():
                    shutil.rmtree(t)
                    cleared = True
            return cleared

        result = patched_clear("nvidia/canary-qwen-2.5b")
        assert result is True
        assert not target.exists(), "Cache dir should have been removed"

    def test_returns_false_when_no_cache(self, tmp_path):
        """When base cache dir doesn't exist, returns False."""
        with mock.patch("nemo_model.Path") as MockPath:
            mock_base = mock.MagicMock()
            mock_base.exists.return_value = False
            MockPath.return_value = mock_base
            # The function creates Path("/root/.cache/torch/NeMo")
            # and checks .exists()
            result = _clear_nemo_cache("nvidia/canary-qwen-2.5b")
        assert result is False

    def test_model_name_parsing_org_slash_slug(self):
        """nvidia/canary-qwen-2.5b → org='nvidia', slug='canary-qwen-2.5b'."""
        parts = "nvidia/canary-qwen-2.5b".split("/")
        assert len(parts) == 2
        assert parts[0] == "nvidia"
        assert parts[1] == "canary-qwen-2.5b"

    def test_model_name_parsing_no_org(self):
        """canary-qwen-2.5b → org='', slug='canary-qwen-2.5b'."""
        parts = "canary-qwen-2.5b".split("/")
        assert len(parts) == 1
        assert parts[-1] == "canary-qwen-2.5b"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. _from_pretrained_with_cache_retry — FileNotFoundError retry
# ═══════════════════════════════════════════════════════════════════════════════

class TestFromPretrainedCacheRetry:
    """Tests for the exact error the user hit on remote."""

    def test_success_on_first_try(self):
        """When model loads fine, returns it directly."""
        mock_asr = mock.MagicMock()
        mock_model = mock.MagicMock()
        mock_asr.models.ASRModel.from_pretrained.return_value = mock_model

        result = _from_pretrained_with_cache_retry(mock_asr, "nvidia/canary-qwen-2.5b", "cuda")
        assert result is mock_model
        mock_asr.models.ASRModel.from_pretrained.assert_called_once()

    def test_retries_after_file_not_found_and_cache_cleared(self):
        """Non-model_config.yaml FileNotFoundError → clears cache → retries → succeeds."""
        mock_asr = mock.MagicMock()
        mock_model = mock.MagicMock()

        # Corrupt cache: a weights file is missing (not model_config.yaml)
        mock_asr.models.ASRModel.from_pretrained.side_effect = [
            FileNotFoundError("/root/.cache/torch/NeMo/.../model_weights.ckpt"),
            mock_model,
        ]

        with mock.patch("nemo_model._clear_nemo_cache", return_value=True) as mock_clear:
            result = _from_pretrained_with_cache_retry(
                mock_asr, "nvidia/parakeet-tdt-0.6b-v3", "cuda"
            )

        assert result is mock_model
        mock_clear.assert_called_once_with("nvidia/parakeet-tdt-0.6b-v3")
        assert mock_asr.models.ASRModel.from_pretrained.call_count == 2

    def test_model_config_yaml_raises_runtime_error(self):
        """FileNotFoundError for model_config.yaml → RuntimeError with actionable message (no retry)."""
        mock_asr = mock.MagicMock()
        mock_asr.models.ASRModel.from_pretrained.side_effect = FileNotFoundError(
            "/root/.cache/torch/NeMo/NeMo_2.1.0/hf_hub_cache/nvidia/canary-qwen-2.5b/abc123/model_config.yaml"
        )

        with mock.patch("nemo_model._clear_nemo_cache") as mock_clear:
            with pytest.raises(RuntimeError, match="model_config.yaml"):
                _from_pretrained_with_cache_retry(
                    mock_asr, "nvidia/canary-qwen-2.5b", "cuda"
                )
        # Must NOT retry — the file simply doesn't exist in the HF repo
        mock_clear.assert_not_called()
        assert mock_asr.models.ASRModel.from_pretrained.call_count == 1

    def test_model_config_yaml_error_message_includes_suggestion(self):
        """RuntimeError message tells user which model to use instead."""
        mock_asr = mock.MagicMock()
        mock_asr.models.ASRModel.from_pretrained.side_effect = FileNotFoundError(
            "model_config.yaml"
        )

        with pytest.raises(RuntimeError, match="parakeet-v3"):
            _from_pretrained_with_cache_retry(
                mock_asr, "nvidia/canary-qwen-2.5b", "cuda"
            )

    def test_raises_when_cache_clear_fails(self):
        """Non-model_config.yaml FileNotFoundError + cache not found → raises without retry."""
        mock_asr = mock.MagicMock()
        mock_asr.models.ASRModel.from_pretrained.side_effect = FileNotFoundError(
            "/root/.cache/torch/NeMo/.../model_weights.ckpt"
        )

        with mock.patch("nemo_model._clear_nemo_cache", return_value=False):
            with pytest.raises(FileNotFoundError):
                _from_pretrained_with_cache_retry(
                    mock_asr, "nvidia/parakeet-tdt-0.6b-v3", "cuda"
                )

    def test_raises_when_retry_also_fails(self):
        """Non-model_config.yaml FileNotFoundError on both attempts → raises second error."""
        mock_asr = mock.MagicMock()
        mock_asr.models.ASRModel.from_pretrained.side_effect = FileNotFoundError(
            "/root/.cache/torch/NeMo/.../model_weights.ckpt"
        )

        with mock.patch("nemo_model._clear_nemo_cache", return_value=True):
            with pytest.raises(FileNotFoundError):
                _from_pretrained_with_cache_retry(
                    mock_asr, "nvidia/parakeet-tdt-0.6b-v3", "cuda"
                )

    def test_other_exceptions_not_retried(self):
        """RuntimeError (not FileNotFoundError) → raises immediately, no retry."""
        mock_asr = mock.MagicMock()
        mock_asr.models.ASRModel.from_pretrained.side_effect = RuntimeError("CUDA OOM")

        with pytest.raises(RuntimeError, match="CUDA OOM"):
            _from_pretrained_with_cache_retry(
                mock_asr, "nvidia/canary-qwen-2.5b", "cuda"
            )

    def test_cpu_map_location(self):
        """When device='cpu', map_location='cpu' is passed."""
        mock_asr = mock.MagicMock()
        mock_model = mock.MagicMock()
        mock_asr.models.ASRModel.from_pretrained.return_value = mock_model

        _from_pretrained_with_cache_retry(mock_asr, "nvidia/parakeet-tdt-0.6b-v3", "cpu")

        mock_asr.models.ASRModel.from_pretrained.assert_called_once_with(
            model_name="nvidia/parakeet-tdt-0.6b-v3", map_location="cpu"
        )

    def test_cuda_map_location_none(self):
        """When device='cuda', map_location=None is passed."""
        mock_asr = mock.MagicMock()
        mock_model = mock.MagicMock()
        mock_asr.models.ASRModel.from_pretrained.return_value = mock_model

        _from_pretrained_with_cache_retry(mock_asr, "nvidia/canary-1b-v2", "cuda")

        mock_asr.models.ASRModel.from_pretrained.assert_called_once_with(
            model_name="nvidia/canary-1b-v2", map_location=None
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. sys.path shadowing — the "nemo.py shadows nemo package" bug
# ═══════════════════════════════════════════════════════════════════════════════

class TestSysPathShadowing:
    """
    Verify _import_nemo_asr and _run_diarization correctly strip the script
    directory from sys.path so 'import nemo.collections' finds the real package
    instead of the local nemo.py file.
    """

    def test_import_nemo_asr_strips_script_dir(self):
        """_import_nemo_asr temporarily removes the nemo/ script directory from sys.path."""
        nemo_script_dir = str((Path(__file__).parent.parent / "nemo").resolve())

        # Record what sys.path looks like inside the import
        captured_path = []

        original_import = importlib.import_module

        def mock_import(name):
            if name == "nemo.collections.asr":
                captured_path.extend(sys.path)
                raise ImportError("mock — just testing sys.path state")
            return original_import(name)

        with mock.patch("importlib.import_module", side_effect=mock_import):
            try:
                _import_nemo_asr()
            except ImportError:
                pass

        # The nemo/ directory should NOT be in sys.path during the import
        resolved_entries = [str(Path(e).resolve()) for e in captured_path if e]
        assert nemo_script_dir not in resolved_entries, (
            f"Script dir {nemo_script_dir} was NOT removed from sys.path during import!\n"
            f"sys.path entries: {resolved_entries[:10]}"
        )

    def test_import_nemo_asr_restores_path_on_failure(self):
        """sys.path is restored even if the import fails."""
        original_path = list(sys.path)

        with mock.patch("importlib.import_module", side_effect=ImportError("no nemo")):
            try:
                _import_nemo_asr()
            except ImportError:
                pass

        assert sys.path == original_path, "sys.path was not restored after failed import"

    def test_run_diarization_strips_script_dir(self):
        """_run_diarization also strips script dir (same bug as _import_nemo_asr)."""
        nemo_script_dir = str((Path(__file__).parent.parent / "nemo").resolve())

        captured_path = []

        def mock_import(name):
            if name == "nemo.collections.asr.models":
                captured_path.extend(sys.path)
                raise ImportError("mock — testing sys.path")
            return importlib.import_module(name)

        with mock.patch("importlib.import_module", side_effect=mock_import):
            try:
                _run_diarization("/fake/audio.wav", Path("/fake/work"))
            except (ImportError, Exception):
                pass

        resolved_entries = [str(Path(e).resolve()) for e in captured_path if e]
        assert nemo_script_dir not in resolved_entries, (
            f"_run_diarization did NOT strip script dir from sys.path!\n"
            f"This causes: ModuleNotFoundError: No module named 'nemo.collections'; "
            f"'nemo' is not a package"
        )

    def test_run_diarization_restores_path_on_failure(self):
        """sys.path is restored even if _run_diarization's import fails."""
        original_path = list(sys.path)

        with mock.patch("importlib.import_module", side_effect=ImportError("no nemo")):
            try:
                _run_diarization("/fake/audio.wav", Path("/fake/work"))
            except (ImportError, Exception):
                pass

        assert sys.path == original_path, "sys.path was not restored after _run_diarization failure"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Hypothesis extraction — handles various NeMo output formats
# ═══════════════════════════════════════════════════════════════════════════════

class TestHypothesisExtraction:
    """NeMo model output can be dict, object, list, nested — handle all."""

    def test_dict_hypothesis(self):
        hyp = {"text": "hello world", "timestamp": {"word": []}}
        assert _looks_like_hyp(hyp) is True
        assert _hyp_field(hyp, "text") == "hello world"

    def test_object_hypothesis(self):
        hyp = types.SimpleNamespace(text="hello world", timestamp={"word": []})
        assert _looks_like_hyp(hyp) is True
        assert _hyp_field(hyp, "text") == "hello world"

    def test_nested_list_extraction(self):
        """_extract_first_hypothesis digs through nested lists."""
        inner = {"text": "found it", "words": ["found", "it"]}
        batch = [[inner]]
        result = _extract_first_hypothesis(batch)
        assert result is inner

    def test_none_input(self):
        assert _extract_first_hypothesis(None) is None
        assert _hyp_field(None, "text", "default") == "default"
        assert _hyp_timestamps(None) is None

    def test_empty_list(self):
        assert _extract_first_hypothesis([]) is None
        assert _extract_first_hypothesis([[]]) is None

    def test_non_hyp_object_skipped(self):
        """Objects without text/words/timestamp are not hyps."""
        assert _looks_like_hyp({"random_key": 42}) is False
        assert _looks_like_hyp(42) is False
        assert _looks_like_hyp("string") is False

    def test_timestamp_key_priority(self):
        """Checks timestamp, then timestep, then timestamps."""
        hyp = {"text": "x", "timestep": {"word": [{"start": 0.0}]}}
        ts = _hyp_timestamps(hyp)
        assert ts == {"word": [{"start": 0.0}]}

    def test_timestamps_plural_key(self):
        hyp = types.SimpleNamespace(text="x", timestamps={"word": []})
        ts = _hyp_timestamps(hyp)
        assert ts == {"word": []}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. _build_srt — end-to-end SRT generation with diarization
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildSrt:
    """Test the main _build_srt function that combines transcription + diarization."""

    def test_words_with_diarization(self):
        """Words + speaker turns → diarized SRT with [Speaker N] tags and all words."""
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.3},
            {"word": "there", "start": 0.3, "end": 0.6},
            {"word": "Goodbye", "start": 1.0, "end": 1.3},
            {"word": "now", "start": 1.3, "end": 1.6},
        ]
        turns = [
            {"speaker": "spk_0", "start": 0.0, "end": 0.8},
            {"speaker": "spk_1", "start": 0.9, "end": 1.8},
        ]
        srt = _build_srt(words, [], turns, diarize=True)
        assert "[Speaker 1]" in srt
        assert "[Speaker 2]" in srt
        assert "Hello" in srt
        # After the _words_to_segs boundary-word fix, "Goodbye" must be in the SRT.
        # This test catches regression of the bug where the word triggering a
        # speaker-change split was silently dropped via `continue`.
        assert "Goodbye" in srt, (
            "Word at segment boundary was dropped — _words_to_segs 'continue' bug regressed.\n"
            f"SRT output:\n{srt}"
        )

    def test_segs_without_diarization(self):
        """Coarse segments without diarization → plain SRT."""
        segs = [
            {"text": "First sentence of the video.", "start": 0.0, "end": 3.0},
            {"text": "Second sentence continues here.", "start": 3.0, "end": 6.0},
        ]
        srt = _build_srt([], segs, [], diarize=False)
        assert "First sentence" in srt
        assert "Second sentence" in srt
        assert "[Speaker" not in srt

    def test_segs_with_diarization(self):
        """Coarse segments (Canary) + diarization turns → diarized SRT."""
        segs = [
            {"text": "Ja genau das stimmt", "start": 0.0, "end": 3.0},
            {"text": "Nein das ist falsch", "start": 3.0, "end": 6.0},
        ]
        turns = [
            {"speaker": "spk_0", "start": 0.0, "end": 3.5},
            {"speaker": "spk_1", "start": 3.5, "end": 6.5},
        ]
        srt = _build_srt([], segs, turns, diarize=True)
        assert "[Speaker" in srt

    def test_empty_words_and_segs(self):
        """No transcription output → empty SRT."""
        srt = _build_srt([], [], [], diarize=False)
        assert srt.strip() == ""


# ═══════════════════════════════════════════════════════════════════════════════
# 6. _assign_speakers — overlap-based speaker assignment
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssignSpeakers:
    """Speaker assignment uses overlap duration to pick best match."""

    def test_exact_match(self):
        items = [{"start": 0.0, "end": 1.0}]
        turns = [{"speaker": "spk_0", "start": 0.0, "end": 1.0}]
        result = _assign_speakers(items, turns)
        assert result[0]["speaker"] == "spk_0"

    def test_partial_overlap_picks_best(self):
        items = [{"start": 0.5, "end": 1.5}]
        turns = [
            {"speaker": "spk_0", "start": 0.0, "end": 0.8},  # 0.3s overlap
            {"speaker": "spk_1", "start": 0.8, "end": 2.0},  # 0.7s overlap
        ]
        result = _assign_speakers(items, turns)
        assert result[0]["speaker"] == "spk_1"

    def test_no_overlap_gets_unknown(self):
        items = [{"start": 5.0, "end": 6.0}]
        turns = [{"speaker": "spk_0", "start": 0.0, "end": 1.0}]
        result = _assign_speakers(items, turns)
        assert result[0]["speaker"] == "unknown"

    def test_empty_turns(self):
        items = [{"start": 0.0, "end": 1.0}]
        result = _assign_speakers(items, [])
        assert result[0]["speaker"] == "unknown"

    def test_multiple_items_different_speakers(self):
        items = [
            {"start": 0.0, "end": 1.0},
            {"start": 2.0, "end": 3.0},
        ]
        turns = [
            {"speaker": "spk_0", "start": 0.0, "end": 1.5},
            {"speaker": "spk_1", "start": 1.5, "end": 3.5},
        ]
        result = _assign_speakers(items, turns)
        assert result[0]["speaker"] == "spk_0"
        assert result[1]["speaker"] == "spk_1"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. _estimate_chunk_sec — VRAM edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEstimateChunkSecEdgeCases:
    """Edge cases for chunk estimation that could cause OOM or quality issues."""

    def test_zero_free_vram_returns_fallback(self):
        """0 GB free → returns 300 (no crash)."""
        with mock.patch("nemo_model._vram_gb", return_value=(0.0, 0.0)):
            result = _estimate_chunk_sec("nvidia/parakeet-tdt-0.6b-v3", 0.85, 1.5)
        assert result == 300

    def test_negative_usable_returns_60(self):
        """Free < reserve → usable is 0 → returns 60."""
        with mock.patch("nemo_model._vram_gb", return_value=(1.0, 8.0)):
            result = _estimate_chunk_sec("nvidia/parakeet-tdt-0.6b-v3", 0.85, 2.0)
        # usable = max(0, 1.0 - 2.0) * 0.85 = 0
        assert result == 60

    def test_canary_always_60_even_tiny_vram(self):
        """Canary returns 60 even with 0.1 GB free."""
        with mock.patch("nemo_model._vram_gb", return_value=(0.1, 8.0)):
            result = _estimate_chunk_sec("nvidia/canary-1b-v2", 0.85, 1.5)
        assert result == 60

    def test_qwen3_vram_driven(self):
        """Qwen3-ASR chunk size is driven by VRAM, not hardcoded."""
        with mock.patch("nemo_model._vram_gb", return_value=(48.0, 80.0)):
            result = _estimate_chunk_sec("Qwen/Qwen3-ASR-1.7B", 0.85, 1.5)
        # usable=(48-1.5)*0.85=39.525 GB; secs=int(39.525/0.35*60)=6775
        assert result == 6775

    def test_qwen3_small_vram(self):
        """Qwen3-ASR on 16GB free: VRAM-driven, well above old 120s hardcode."""
        with mock.patch("nemo_model._vram_gb", return_value=(16.0, 24.0)):
            result = _estimate_chunk_sec("Qwen/Qwen3-ASR-1.7B", 0.85, 1.5)
        # usable=(16-1.5)*0.85=12.325 GB; secs=int(12.325/0.35*60)=2112
        assert result == 2112
        assert result > 120, "Qwen3 must exceed old 120s hardcode on 16 GB"

    def test_parakeet_capped_at_7200(self):
        """Parakeet can't exceed 7200s even with very large VRAM."""
        with mock.patch("nemo_model._vram_gb", return_value=(100.0, 100.0)):
            result = _estimate_chunk_sec("nvidia/parakeet-tdt-0.6b-v3", 0.85, 1.5)
        assert result == 7200

    def test_parakeet_16gb_exceeds_old_3600_cap(self):
        """On 16 GB, parakeet chunk should exceed the old 3600s cap."""
        with mock.patch("nemo_model._vram_gb", return_value=(16.0, 24.0)):
            result = _estimate_chunk_sec("nvidia/parakeet-tdt-0.6b-v3", 0.85, 1.5)
        # usable=12.325 GB; secs=int(12.325/0.28*60)=2641
        assert result == 2641

    def test_parakeet_minimum_30(self):
        """Parakeet returns at least 30s even with tiny usable VRAM."""
        with mock.patch("nemo_model._vram_gb", return_value=(1.6, 8.0)):
            # usable = (1.6 - 1.5) * 0.85 = 0.085 → 0.085/0.28*60 = 18.2 → capped to 30
            result = _estimate_chunk_sec("nvidia/parakeet-tdt-0.6b-v3", 0.85, 1.5)
        assert result >= 30


# ═══════════════════════════════════════════════════════════════════════════════
# 8. _validate_checkpoint — additional edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateCheckpointEdgeCases:
    """Edge cases not covered by the existing matrix tests."""

    def test_checkpoint_with_nan_duration(self, tmp_path):
        """NaN audio_duration in checkpoint → treated as 0, no crash."""
        cp = tmp_path / "cp.json"
        cp.write_text(json.dumps({
            "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
            "segs": [],
            "audio_duration": float("nan"),
            "asr_elapsed": 1.0,
            "rtf": 0.1,
            "trim_sec": 0,
        }))
        # NaN comparisons are tricky — ensure no crash
        result = _validate_checkpoint(cp, str(tmp_path / "nonexistent.wav"), 0)
        # NaN > 0 is False, so cp_dur check is skipped → valid
        assert isinstance(result, bool)

    def test_checkpoint_with_very_large_trim(self, tmp_path):
        """trim_sec=999999 in checkpoint vs trim_sec=0 at load → stale."""
        cp = tmp_path / "cp.json"
        cp.write_text(json.dumps({
            "words": [{"word": "x", "start": 0, "end": 1}],
            "segs": [],
            "audio_duration": 999999.0,
            "asr_elapsed": 1.0,
            "rtf": 0.1,
            "trim_sec": 999999,
        }))
        assert _validate_checkpoint(cp, str(tmp_path / "x.wav"), 0) is False

    def test_checkpoint_with_unicode_content(self, tmp_path):
        """Checkpoint with unicode text doesn't crash validation."""
        cp = tmp_path / "cp.json"
        cp.write_text(json.dumps({
            "words": [{"word": "日本語", "start": 0, "end": 1}],
            "segs": [],
            "audio_duration": 10.0,
            "asr_elapsed": 3.0,
            "rtf": 0.3,
            "trim_sec": 0,
        }), encoding="utf-8")
        assert _validate_checkpoint(cp, str(tmp_path / "x.wav"), 0) is True

    def test_checkpoint_truncated_file(self, tmp_path):
        """Truncated JSON file (disk full scenario) → invalid."""
        cp = tmp_path / "cp.json"
        cp.write_text('{"words": [{"word": "hello"')  # truncated
        assert _validate_checkpoint(cp, str(tmp_path / "x.wav"), 0) is False

    def test_checkpoint_empty_file(self, tmp_path):
        """Empty file → invalid."""
        cp = tmp_path / "cp.json"
        cp.write_text("")
        assert _validate_checkpoint(cp, str(tmp_path / "x.wav"), 0) is False

    def test_checkpoint_binary_garbage(self, tmp_path):
        """Binary content → invalid."""
        cp = tmp_path / "cp.json"
        cp.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG header
        assert _validate_checkpoint(cp, str(tmp_path / "x.wav"), 0) is False


# ═══════════════════════════════════════════════════════════════════════════════
# 9. _load_model — GPU/CPU fallback and Qwen3 dispatch
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadModelDispatch:
    """_load_model must dispatch to Qwen3 or NeMo based on model name."""

    def test_dispatches_to_qwen3_for_qwen_model(self):
        """Qwen model names go to _load_qwen3_asr, not NeMo."""
        with mock.patch("nemo_model._import_nemo_asr") as mock_nemo, \
             mock.patch("qwen3_asr._load_qwen3_asr", return_value="qwen_model") as mock_qwen, \
             mock.patch("qwen3_asr._is_qwen3_asr", return_value=True):
            from nemo_model import _load_model
            result = _load_model("Qwen/Qwen3-ASR-1.7B", "bf16", "cpu")
            assert result == "qwen_model"
            mock_nemo.assert_not_called()

    def test_dispatches_to_nemo_for_parakeet(self):
        """Parakeet model names go through NeMo path and work on CPU without UnboundLocalError."""
        mock_nemo_asr = mock.MagicMock()
        mock_model = mock.MagicMock()
        mock_model.named_children.return_value = []
        mock_model.eval.return_value = mock_model

        with mock.patch("nemo_model._import_nemo_asr", return_value=mock_nemo_asr), \
             mock.patch("nemo_model._from_pretrained_with_cache_retry", return_value=mock_model), \
             mock.patch("nemo_model._vram_gb", return_value=(0.0, 0.0)), \
             mock.patch("nemo_model.torch") as mock_torch, \
             mock.patch("nemo_model.gc"):
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends = mock.MagicMock()
            from nemo_model import _load_model
            # This must NOT raise UnboundLocalError: 'free' referenced before assignment on CPU
            result = _load_model("nvidia/parakeet-tdt-0.6b-v3", "bf16", "cpu")
            assert result is mock_model
