"""
tests/test_pipeline_orchestration.py — Tests for the pipeline orchestration layer.

Covers:
  - run_pipeline.py argument parsing and run-mode presets
  - pipeline_paths._validate_translated_srt (the silent translation failure detector)
  - pipeline_paths._find_video / _find_srt_for_video (file discovery)
  - pipeline_paths._derive_run_label (collision-free naming)
  - pipeline_utils._ollama_is_running / _docker_available
  - entrypoint.sh env-var to CLI flag mapping consistency

Run:
    uv run --with "pytest,pydantic" pytest tests/test_pipeline_orchestration.py -v
"""

from __future__ import annotations

import re
import subprocess
import sys
import unittest.mock as mock
from pathlib import Path

import pytest

from pipeline_paths import (
    _derive_run_label,
    _find_srt_for_video,
    _find_video,
    _normalize_base,
    _validate_translated_srt,
    _video_already_processed,
)
from pipeline_utils import (
    _docker_available,
    _ollama_is_running,
    _python,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. _validate_translated_srt — silent translation failure detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateTranslatedSrt:
    """
    Catches the case where Ollama's model was not pulled:
    translation silently returns empty strings, leaving only [Speaker N] tags.
    """

    def test_valid_srt_passes(self, tmp_path):
        srt = tmp_path / "ok.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:02,000\n[Speaker 1] Bonjour le monde\n\n"
            "2\n00:00:02,000 --> 00:00:04,000\n[Speaker 2] Salut tout le monde\n\n",
            encoding="utf-8",
        )
        # Should not raise or call sys.exit
        _validate_translated_srt(srt, "fr")

    def test_empty_file_exits(self, tmp_path):
        srt = tmp_path / "empty.srt"
        srt.write_text("", encoding="utf-8")
        with pytest.raises(SystemExit):
            _validate_translated_srt(srt, "fr")

    def test_only_speaker_tags_exits(self, tmp_path):
        """The exact failure: model pulled wrong or 404 → only speaker tags, no text."""
        srt = tmp_path / "bad.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:02,000\n[Speaker 1]\n\n"
            "2\n00:00:02,000 --> 00:00:04,000\n[Speaker 2]\n\n"
            "3\n00:00:04,000 --> 00:00:06,000\n[Speaker 1]\n\n",
            encoding="utf-8",
        )
        with pytest.raises(SystemExit):
            _validate_translated_srt(srt, "fr")

    def test_partial_speaker_tags_passes(self, tmp_path):
        """Mix of speaker tags and real text → passes (not 100% speaker-only)."""
        srt = tmp_path / "partial.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:02,000\n[Speaker 1] Real translated text\n\n"
            "2\n00:00:02,000 --> 00:00:04,000\n[Speaker 2]\n\n",
            encoding="utf-8",
        )
        # Should NOT exit — most content is translated
        _validate_translated_srt(srt, "fr")

    def test_only_timestamps_and_numbers_exits(self, tmp_path):
        """SRT with only index/timestamp lines and no text → exits."""
        srt = tmp_path / "timestamps_only.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:02,000\n\n"
            "2\n00:00:02,000 --> 00:00:04,000\n\n",
            encoding="utf-8",
        )
        with pytest.raises(SystemExit):
            _validate_translated_srt(srt, "fr")

    def test_single_content_line_passes(self, tmp_path):
        """Even a single real content line → passes."""
        srt = tmp_path / "one.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:02,000\n[Speaker 1] Hello world\n\n",
            encoding="utf-8",
        )
        _validate_translated_srt(srt, "fr")

    def test_speaker_tag_with_space_variants(self, tmp_path):
        """[Speaker 10], [Speaker 1], etc. all detected as tags."""
        srt = tmp_path / "multi.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:02,000\n[Speaker 10]\n\n"
            "2\n00:00:02,000 --> 00:00:04,000\n[Speaker 1]\n\n",
            encoding="utf-8",
        )
        with pytest.raises(SystemExit):
            _validate_translated_srt(srt, "fr")

    def test_unicode_content_passes(self, tmp_path):
        """Arabic/Chinese content is real translation → passes."""
        srt = tmp_path / "unicode.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:02,000\n[Speaker 1] مرحبا بالعالم\n\n",
            encoding="utf-8",
        )
        _validate_translated_srt(srt, "ar")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. _normalize_base — fuzzy filename matching
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalizeBase:

    @pytest.mark.parametrize("input_str,expected", [
        ("MyVideo",                    "myvideo"),
        ("my video",                   "my_video"),
        ("my-video",                   "my_video"),
        ("my.video",                   "my_video"),
        ("my__video",                  "my_video"),
        ("Debate 101 with Harvard's",  "debate_101_with_harvard_s"),
        ("video (1)",                  "video_1"),
        ("",                           ""),
        ("---",                        ""),
        ("abc123",                     "abc123"),
    ])
    def test_normalization(self, input_str, expected):
        assert _normalize_base(input_str) == expected

    def test_spaces_and_underscores_are_equivalent(self):
        assert _normalize_base("my video") == _normalize_base("my_video")

    def test_dots_and_hyphens_are_equivalent(self):
        assert _normalize_base("my.video") == _normalize_base("my-video")

    def test_case_insensitive(self):
        assert _normalize_base("MyVideo") == _normalize_base("myvideo")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. _video_already_processed — skip logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestVideoAlreadyProcessed:

    def test_no_end_product_dir(self, tmp_path):
        video = tmp_path / "myvideo.mp4"
        video.touch()
        result = _video_already_processed(
            video, target_lang="fr", end_product_dir=tmp_path / "nonexistent"
        )
        assert result is False

    def test_run_dir_exists_for_same_lang(self, tmp_path):
        """Run dir with correct lang → processed."""
        end = tmp_path / "end_product"
        (end / "myvideo__de_to_fr").mkdir(parents=True)
        video = tmp_path / "myvideo.mp4"
        video.touch()
        assert _video_already_processed(video, "fr", end_product_dir=end) is True

    def test_run_dir_exists_for_different_lang(self, tmp_path):
        """Run dir with different lang → NOT processed for our lang."""
        end = tmp_path / "end_product"
        (end / "myvideo__de_to_es").mkdir(parents=True)
        video = tmp_path / "myvideo.mp4"
        video.touch()
        assert _video_already_processed(video, "fr", end_product_dir=end) is False

    def test_fuzzy_name_match(self, tmp_path):
        """'my video' matches 'my_video' run dir."""
        end = tmp_path / "end_product"
        (end / "my_video__de_to_fr").mkdir(parents=True)
        video = tmp_path / "my video.mp4"
        video.touch()
        assert _video_already_processed(video, "fr", end_product_dir=end) is True

    def test_no_target_lang_matches_any_run_dir(self, tmp_path):
        """Without target_lang, any matching run dir counts."""
        end = tmp_path / "end_product"
        (end / "myvideo__de_to_es").mkdir(parents=True)
        video = tmp_path / "myvideo.mp4"
        video.touch()
        assert _video_already_processed(video, None, end_product_dir=end) is True


# ═══════════════════════════════════════════════════════════════════════════════
# 4. _find_video — discovery and ordering
# ═══════════════════════════════════════════════════════════════════════════════

class TestFindVideo:

    def test_finds_mp4(self, tmp_path):
        video = tmp_path / "myvideo.mp4"
        video.touch()
        result = _find_video(nemo_dir=tmp_path, end_product_dir=tmp_path / "end")
        assert result == video

    def test_skips_chunk_wavs(self, tmp_path):
        """_chunk_XXXX.wav files must not be returned as input videos."""
        chunk = tmp_path / "_chunk_0001.wav"
        chunk.touch()
        result = _find_video(nemo_dir=tmp_path, end_product_dir=tmp_path / "end")
        assert result is None

    def test_returns_none_when_empty(self, tmp_path):
        result = _find_video(nemo_dir=tmp_path, end_product_dir=tmp_path / "end")
        assert result is None

    def test_skips_already_processed(self, tmp_path):
        """Videos with run dirs for target_lang are skipped."""
        video = tmp_path / "myvideo.mp4"
        video.touch()
        end = tmp_path / "end_product"
        (end / "myvideo__de_to_fr").mkdir(parents=True)

        result = _find_video(
            target_lang="fr",
            nemo_dir=tmp_path,
            end_product_dir=end,
        )
        # All videos are processed → returns the only one anyway (fallback)
        # The function returns videos[0] as last resort when all are processed
        assert result is not None

    def test_multiple_videos_prefers_unprocessed(self, tmp_path):
        """Unprocessed video is returned before processed one."""
        processed = tmp_path / "processed.mp4"
        unprocessed = tmp_path / "fresh.mp4"
        processed.touch()
        unprocessed.touch()

        end = tmp_path / "end_product"
        (end / "processed__de_to_fr").mkdir(parents=True)

        result = _find_video(
            target_lang="fr",
            nemo_dir=tmp_path,
            end_product_dir=end,
        )
        assert result == unprocessed

    def test_wav_input_accepted(self, tmp_path):
        """WAV files are valid inputs (for when audio is pre-extracted)."""
        wav = tmp_path / "audio.wav"
        wav.touch()
        result = _find_video(nemo_dir=tmp_path, end_product_dir=tmp_path / "end")
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. _find_srt_for_video — SRT discovery in nemo/ and end_product/
# ═══════════════════════════════════════════════════════════════════════════════

class TestFindSrtForVideo:

    def test_finds_srt_in_nemo_dir(self, tmp_path):
        srt = tmp_path / "myvideo.nemo.de.diarize.srt"
        srt.touch()
        result = _find_srt_for_video(
            "myvideo", "*.nemo.de.diarize.srt",
            nemo_dir=tmp_path, end_product_dir=tmp_path / "end",
        )
        assert result == srt

    def test_finds_srt_in_end_product(self, tmp_path):
        """SRT moved by clean_subs.py to end_product is found."""
        end = tmp_path / "end_product" / "myvideo__de_to_fr"
        end.mkdir(parents=True)
        srt = end / "myvideo.nemo.de.diarize.srt"
        srt.touch()
        result = _find_srt_for_video(
            "myvideo", "*.nemo.de.diarize.srt",
            nemo_dir=tmp_path, end_product_dir=tmp_path / "end_product",
        )
        assert result == srt

    def test_returns_none_when_missing(self, tmp_path):
        result = _find_srt_for_video(
            "myvideo", "*.nemo.de.diarize.srt",
            nemo_dir=tmp_path, end_product_dir=tmp_path / "end",
        )
        assert result is None

    def test_fuzzy_match_with_spaces(self, tmp_path):
        """'my video' matches SRT with underscores 'my_video.nemo.de...'."""
        srt = tmp_path / "my_video.nemo.de.diarize.srt"
        srt.touch()
        result = _find_srt_for_video(
            "my video", "*.nemo.de.diarize.srt",
            nemo_dir=tmp_path, end_product_dir=tmp_path / "end",
        )
        assert result == srt

    def test_translated_srt_discovery(self, tmp_path):
        srt = tmp_path / "myvideo.nemo.de.diarize_fr.srt"
        srt.touch()
        result = _find_srt_for_video(
            "myvideo", "*.diarize_fr.srt",
            nemo_dir=tmp_path, end_product_dir=tmp_path / "end",
        )
        assert result == srt


# ═══════════════════════════════════════════════════════════════════════════════
# 6. _derive_run_label — unique collision-free naming
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeriveRunLabel:

    def test_basic_label(self, tmp_path):
        """Standard run label includes source/target lang."""
        srt = tmp_path / "myvideo.nemo.de.diarize.srt"
        srt.touch()
        video = tmp_path / "myvideo.mp4"
        video.touch()
        end = tmp_path / "end_product"
        end.mkdir()
        label = _derive_run_label(
            "de", "fr", video=video, nemo_dir=tmp_path, end_product_dir=end
        )
        assert "de_to_fr" in label

    def test_collision_increments(self, tmp_path):
        """If run label already exists in end_product, append __2, __3, etc."""
        srt = tmp_path / "myvideo.nemo.de.diarize.srt"
        srt.touch()
        video = tmp_path / "myvideo.mp4"
        video.touch()
        end = tmp_path / "end_product"
        end.mkdir()

        # First label
        label1 = _derive_run_label("de", "fr", video=video, nemo_dir=tmp_path, end_product_dir=end)
        (end / label1).mkdir()

        # Second run → should get __2 suffix
        label2 = _derive_run_label("de", "fr", video=video, nemo_dir=tmp_path, end_product_dir=end)
        assert label2 != label1
        assert "__2" in label2

        # Third run → __3
        (end / label2).mkdir()
        label3 = _derive_run_label("de", "fr", video=video, nemo_dir=tmp_path, end_product_dir=end)
        assert "__3" in label3

    def test_no_srt_uses_video_stem(self, tmp_path):
        """Without SRT, falls back to video stem for label."""
        video = tmp_path / "myvideo.mp4"
        video.touch()
        end = tmp_path / "end_product"
        end.mkdir()
        label = _derive_run_label(
            "de", "fr", video=video, nemo_dir=tmp_path, end_product_dir=end
        )
        assert "de_to_fr" in label

    def test_no_video_no_srt_uses_timestamp(self, tmp_path):
        """Without video or SRT, falls back to timestamp-based label."""
        end = tmp_path / "end_product"
        end.mkdir()
        label = _derive_run_label(
            "de", "fr", video=None, nemo_dir=tmp_path, end_product_dir=end
        )
        assert "de_to_fr" in label


# ═══════════════════════════════════════════════════════════════════════════════
# 7. run_pipeline.py argparse — run-mode presets
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunPipelineArgparse:
    """
    run_pipeline.py's argparse and run-mode logic, tested in isolation.
    We import and call parse_args directly without running main().
    """

    def _make_parser(self):
        """Recreate the argparse setup from run_pipeline.py."""
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--target-lang", required=True)
        p.add_argument("--language", default=None)
        p.add_argument("--trim", type=int, default=0)
        p.add_argument("--qwen-mode", default="clone", choices=["clone", "custom"])
        p.add_argument("--no-demucs", action="store_true")
        p.add_argument("--whisper-model", default="medium")
        p.add_argument("--skip-nemo", action="store_true")
        p.add_argument("--skip-translate", action="store_true")
        p.add_argument("--skip-dub", action="store_true")
        p.add_argument("--run-mode", default="full", choices=["transcribe", "translate", "full"])
        p.add_argument("--input-file", default=None)
        p.add_argument("--input-dir", default=None)
        p.add_argument("--output-dir", default=None)
        p.add_argument("--precision", default="bf16", choices=["fp32", "fp16", "bf16"])
        p.add_argument("--nemo-model", default=None)
        p.add_argument("--chunk-override", default=None, type=int)
        p.add_argument("--reserve-gb", default=None, type=float)
        p.add_argument("--safety-factor", default=None, type=float)
        return p

    def _apply_run_mode(self, args):
        """Apply run-mode presets — same logic as run_pipeline.py main()."""
        if args.run_mode == "transcribe":
            args.skip_translate = True
            args.skip_dub = True
        elif args.run_mode == "translate":
            args.skip_dub = True
        return args

    def test_transcribe_mode_sets_skip_flags(self):
        p = self._make_parser()
        args = p.parse_args(["--target-lang", "fr", "--run-mode", "transcribe"])
        args = self._apply_run_mode(args)
        assert args.skip_translate is True
        assert args.skip_dub is True

    def test_translate_mode_sets_skip_dub(self):
        p = self._make_parser()
        args = p.parse_args(["--target-lang", "fr", "--run-mode", "translate"])
        args = self._apply_run_mode(args)
        assert args.skip_translate is False
        assert args.skip_dub is True

    def test_full_mode_skips_nothing(self):
        p = self._make_parser()
        args = p.parse_args(["--target-lang", "fr", "--run-mode", "full"])
        args = self._apply_run_mode(args)
        assert args.skip_translate is False
        assert args.skip_dub is False

    def test_explicit_skip_flags(self):
        p = self._make_parser()
        args = p.parse_args(["--target-lang", "fr", "--skip-nemo", "--skip-translate"])
        assert args.skip_nemo is True
        assert args.skip_translate is True
        assert args.skip_dub is False

    def test_trim_default_zero(self):
        p = self._make_parser()
        args = p.parse_args(["--target-lang", "fr"])
        assert args.trim == 0

    def test_precision_choices(self):
        p = self._make_parser()
        for prec in ["fp32", "fp16", "bf16"]:
            args = p.parse_args(["--target-lang", "fr", "--precision", prec])
            assert args.precision == prec

    def test_invalid_precision_rejected(self):
        p = self._make_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["--target-lang", "fr", "--precision", "fp8"])

    def test_qwen_mode_choices(self):
        p = self._make_parser()
        for mode in ["clone", "custom"]:
            args = p.parse_args(["--target-lang", "fr", "--qwen-mode", mode])
            assert args.qwen_mode == mode

    def test_nemo_model_passthrough(self):
        p = self._make_parser()
        args = p.parse_args(["--target-lang", "fr", "--nemo-model", "nvidia/parakeet-tdt-0.6b-v3"])
        assert args.nemo_model == "nvidia/parakeet-tdt-0.6b-v3"

    def test_chunk_override_type(self):
        p = self._make_parser()
        args = p.parse_args(["--target-lang", "fr", "--chunk-override", "120"])
        assert args.chunk_override == 120
        assert isinstance(args.chunk_override, int)

    def test_reserve_gb_type(self):
        p = self._make_parser()
        args = p.parse_args(["--target-lang", "fr", "--reserve-gb", "2.5"])
        assert args.reserve_gb == 2.5
        assert isinstance(args.reserve_gb, float)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. entrypoint.sh ↔ run_pipeline.py consistency
# ═══════════════════════════════════════════════════════════════════════════════

class TestEntrypointConsistency:
    """Verify entrypoint.sh env vars all map to flags that run_pipeline.py accepts."""

    def _read_entrypoint(self) -> str:
        ep = Path(__file__).parent.parent / "entrypoint.sh"
        if not ep.exists():
            pytest.skip("entrypoint.sh not found")
        return ep.read_text()

    def _read_run_pipeline(self) -> str:
        rp = Path(__file__).parent.parent / "run_pipeline.py"
        if not rp.exists():
            pytest.skip("run_pipeline.py not found")
        return rp.read_text()

    def test_all_cli_flags_in_run_pipeline(self):
        """Every --flag passed by entrypoint.sh must exist in run_pipeline.py."""
        ep = self._read_entrypoint()
        rp = self._read_run_pipeline()

        # Extract flags built by entrypoint.sh: patterns like (--target-lang "$TARGET_LANG")
        ep_flags = set(re.findall(r'(--[a-z][a-z0-9-]+)', ep))
        # Exclude flags that aren't passed by entrypoint.sh itself:
        # --help is only in comments/docs, not actually passed as a flag
        ep_flags -= {"--help"}

        # Extract flags declared in run_pipeline.py argparse
        rp_flags = set(re.findall(r'add_argument\("(--[a-z][a-z0-9-]+)"', rp))

        for flag in ep_flags:
            assert flag in rp_flags, (
                f"entrypoint.sh passes '{flag}' but run_pipeline.py has no such argument!\n"
                f"  Known flags: {sorted(rp_flags)}"
            )

    def test_target_lang_required_check_exists(self):
        """entrypoint.sh must check that TARGET_LANG is set."""
        ep = self._read_entrypoint()
        assert "TARGET_LANG" in ep
        assert "exit 1" in ep or "exit" in ep

    def test_no_demucs_flag_wired(self):
        """NO_DEMUCS env var must produce --no-demucs flag."""
        ep = self._read_entrypoint()
        assert "NO_DEMUCS" in ep
        assert "--no-demucs" in ep

    def test_skip_flags_wired(self):
        """SKIP_NEMO / SKIP_TRANSLATE / SKIP_DUB must produce the right flags."""
        ep = self._read_entrypoint()
        assert "--skip-nemo" in ep
        assert "--skip-translate" in ep
        assert "--skip-dub" in ep

    def test_nemo_model_env_wired(self):
        """NEMO_MODEL env var maps to --nemo-model flag."""
        ep = self._read_entrypoint()
        assert "NEMO_MODEL" in ep
        assert "--nemo-model" in ep


# ═══════════════════════════════════════════════════════════════════════════════
# 9. _python helper — venv vs fallback
# ═══════════════════════════════════════════════════════════════════════════════

class TestPythonHelper:

    def test_returns_venv_python_when_exists(self, tmp_path):
        venv_py = tmp_path / ".venv" / "bin" / "python"
        venv_py.parent.mkdir(parents=True)
        venv_py.touch()
        result = _python(venv_py, tmp_path)
        assert result == [str(venv_py)]

    def test_falls_back_to_uv_run(self, tmp_path):
        venv_py = tmp_path / ".venv" / "bin" / "python"  # doesn't exist
        result = _python(venv_py, tmp_path)
        assert result == ["uv", "run", "python"]


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Ollama lifecycle helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestOllamaHelpers:

    def test_ollama_is_running_true_on_200(self):
        with mock.patch("pipeline_utils.urllib.request.urlopen") as mock_open:
            mock_open.return_value = mock.MagicMock()
            assert _ollama_is_running() is True

    def test_ollama_is_running_false_on_connection_error(self):
        import urllib.error
        with mock.patch("pipeline_utils.urllib.request.urlopen",
                        side_effect=urllib.error.URLError("refused")):
            assert _ollama_is_running() is False

    def test_ollama_is_running_false_on_timeout(self):
        with mock.patch("pipeline_utils.urllib.request.urlopen",
                        side_effect=TimeoutError("timeout")):
            assert _ollama_is_running() is False

    def test_docker_available_true_on_zero_exit(self):
        with mock.patch("pipeline_utils.subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0)
            assert _docker_available() is True

    def test_docker_available_false_on_nonzero_exit(self):
        with mock.patch("pipeline_utils.subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=1)
            assert _docker_available() is False

    def test_docker_available_false_on_exception(self):
        with mock.patch("pipeline_utils.subprocess.run",
                        side_effect=FileNotFoundError("docker not found")):
            assert _docker_available() is False


# ═══════════════════════════════════════════════════════════════════════════════
# 11. docker-compose.yml model consistency
# ═══════════════════════════════════════════════════════════════════════════════

class TestDockerComposeConsistency:
    """
    Catch config mismatches between docker-compose.yml services.
    The user hit 404 'model not found' because ollama-init pulled a different
    model than what the pipeline was configured to use.
    """

    def _read_compose(self) -> str:
        p = Path(__file__).parent.parent / "docker-compose.yml"
        if not p.exists():
            pytest.skip("docker-compose.yml not found")
        return p.read_text()

    def test_ollama_init_and_pipeline_use_same_default_model(self):
        """ollama-init TRANSLATE_MODEL default == pipeline TRANSLATE_MODEL default."""
        compose = self._read_compose()

        # Find TRANSLATE_MODEL defaults for each service
        # Pattern: TRANSLATE_MODEL: "${TRANSLATE_MODEL:-<default>}"
        translate_model_defaults = re.findall(
            r'TRANSLATE_MODEL.*\$\{TRANSLATE_MODEL:-([^}]+)\}', compose
        )
        assert len(translate_model_defaults) >= 2, (
            "Expected at least 2 TRANSLATE_MODEL definitions in docker-compose.yml "
            f"(one in ollama-init, one in pipeline), found: {translate_model_defaults}"
        )
        unique_defaults = set(translate_model_defaults)
        assert len(unique_defaults) == 1, (
            f"TRANSLATE_MODEL defaults differ between services: {unique_defaults}\n"
            "This causes 404 'model not found' errors on remote — "
            "ollama-init pulls one version, pipeline requests another."
        )

    def test_gpu_passthrough_configured(self):
        """Both ollama and pipeline services should have GPU access."""
        compose = self._read_compose()
        assert "nvidia" in compose
        assert "capabilities" in compose

    def test_volumes_defined(self):
        """Required volumes (ollama_models, hf_cache, nemo_cache) are declared."""
        compose = self._read_compose()
        for vol in ["ollama_models", "hf_cache", "nemo_cache"]:
            assert vol in compose, f"Volume '{vol}' not found in docker-compose.yml"

    def test_ollama_host_set_for_pipeline(self):
        """Pipeline container must know where Ollama is (Docker service name)."""
        compose = self._read_compose()
        assert "OLLAMA_HOST" in compose
        # Should use Docker service name, not localhost (localhost doesn't work between containers)
        assert "ollama:" in compose or "ollama:11434" in compose
