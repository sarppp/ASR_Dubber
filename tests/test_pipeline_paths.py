"""
tests/test_pipeline_paths.py — Tests for pipeline_paths.py
===========================================================

Tests the critical glue functions that find/validate files and manage outputs:
  1. _normalize_base()          — fuzzy filename matching
  2. _video_already_processed() — skip already-processed videos
  3. _find_video()              — pick next unprocessed video
  4. _find_srt_for_video()      — locate SRTs in nemo/ or end_product/
  5. _derive_run_label()        — stable run folder names
  6. _validate_translated_srt() — catch silent translation failures

Run:
    uv run --with "pytest,pysrt" pytest tests/test_pipeline_paths.py -v
"""

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline_paths import (
    _normalize_base,
    _video_already_processed,
    _find_video,
    _find_srt_for_video,
    _derive_run_label,
    _validate_translated_srt,
)


# ═════════════════════════════════════════════════════════════════════════════
# 1. _normalize_base() — fuzzy matching
# ═════════════════════════════════════════════════════════════════════════════

class TestNormalizeBase:

    @pytest.mark.parametrize("input_str,expected", [
        ("My Video File", "my_video_file"),
        ("my_video_file", "my_video_file"),
        ("My-Video.File", "my_video_file"),
        ("  spaces  around  ", "spaces_around"),
        ("UPPER_case_MiXeD", "upper_case_mixed"),
        ("file (2) copy", "file_2_copy"),
        ("Debate 101 with Harvard's", "debate_101_with_harvard_s"),
        ("video.nemo.de.diarize", "video_nemo_de_diarize"),
    ], ids=["spaces", "underscores", "mixed-separators", "extra-spaces",
            "mixed-case", "parens", "apostrophe", "dotted"])
    def test_normalization(self, input_str, expected):
        assert _normalize_base(input_str) == expected

    def test_symmetric_matching(self):
        """Two different representations of the same name must normalize equally."""
        assert _normalize_base("My Video") == _normalize_base("my_video")
        assert _normalize_base("Debate-101") == _normalize_base("Debate 101")


# ═════════════════════════════════════════════════════════════════════════════
# 2. _video_already_processed()
# ═════════════════════════════════════════════════════════════════════════════

class TestVideoAlreadyProcessed:

    def test_not_processed_when_no_end_product_dir(self, tmp_path):
        video = tmp_path / "nemo" / "video.mp4"
        video.parent.mkdir()
        video.touch()
        nonexistent = tmp_path / "end_product"
        assert not _video_already_processed(video, "fr", end_product_dir=nonexistent)

    def test_processed_when_matching_run_dir_exists(self, tmp_path):
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        video = nemo_dir / "video.mp4"
        video.touch()
        end_product = tmp_path / "end_product"
        (end_product / "video_nemo_de__de_to_fr").mkdir(parents=True)

        assert _video_already_processed(video, "fr", end_product_dir=end_product)

    def test_not_processed_for_different_target_lang(self, tmp_path):
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        video = nemo_dir / "video.mp4"
        video.touch()
        end_product = tmp_path / "end_product"
        # Processed for French, but we're asking about Spanish
        (end_product / "video_nemo_de__de_to_fr").mkdir(parents=True)

        assert not _video_already_processed(video, "es", end_product_dir=end_product)

    def test_processed_ignores_non_directories(self, tmp_path):
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        video = nemo_dir / "video.mp4"
        video.touch()
        end_product = tmp_path / "end_product"
        end_product.mkdir()
        # Create a file (not a dir) with a matching name
        (end_product / "video_nemo_de__de_to_fr").write_text("not a dir")

        assert not _video_already_processed(video, "fr", end_product_dir=end_product)


# ═════════════════════════════════════════════════════════════════════════════
# 3. _find_video()
# ═════════════════════════════════════════════════════════════════════════════

class TestFindVideo:

    def test_finds_mp4_in_nemo_dir(self, tmp_path):
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        video = nemo_dir / "test_video.mp4"
        video.write_bytes(b"\x00" * 10)

        result = _find_video(target_lang="fr", nemo_dir=nemo_dir,
                             end_product_dir=tmp_path / "end_product")
        assert result == video

    def test_finds_mkv_and_other_formats(self, tmp_path):
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        for ext in [".mkv", ".avi", ".mov", ".webm"]:
            video = nemo_dir / f"test{ext}"
            video.write_bytes(b"\x00" * 10)

        result = _find_video(target_lang="fr", nemo_dir=nemo_dir,
                             end_product_dir=tmp_path / "end_product")
        assert result is not None
        assert result.suffix.lower() in {".mkv", ".avi", ".mov", ".webm"}

    def test_skips_processed_videos(self, tmp_path):
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        processed = nemo_dir / "done.mp4"
        processed.write_bytes(b"\x00" * 10)
        unprocessed = nemo_dir / "fresh.mp4"
        unprocessed.write_bytes(b"\x00" * 10)

        end_product = tmp_path / "end_product"
        (end_product / "done_nemo_de__de_to_fr").mkdir(parents=True)

        result = _find_video(target_lang="fr", nemo_dir=nemo_dir,
                             end_product_dir=end_product)
        assert result == unprocessed

    def test_returns_none_when_empty_dir(self, tmp_path):
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        result = _find_video(target_lang="fr", nemo_dir=nemo_dir,
                             end_product_dir=tmp_path / "end_product")
        assert result is None

    def test_ignores_chunk_wav_files(self, tmp_path):
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        # Chunk WAVs should be ignored
        (nemo_dir / "_chunk_0000.wav").write_bytes(b"\x00" * 10)
        (nemo_dir / "_chunk_0001.wav").write_bytes(b"\x00" * 10)
        # Real video
        video = nemo_dir / "real_video.mp4"
        video.write_bytes(b"\x00" * 10)

        result = _find_video(target_lang="fr", nemo_dir=nemo_dir,
                             end_product_dir=tmp_path / "end_product")
        assert result == video

    def test_wav_input_detected(self, tmp_path):
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        wav = nemo_dir / "audio_input.wav"
        wav.write_bytes(b"\x00" * 10)

        result = _find_video(target_lang="fr", nemo_dir=nemo_dir,
                             end_product_dir=tmp_path / "end_product")
        assert result == wav

    def test_prefers_untouched_videos_over_started(self, tmp_path):
        """Videos without existing WAV files should be processed first."""
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        started = nemo_dir / "old_video.mp4"
        started.write_bytes(b"\x00" * 10)
        # Create a WAV that indicates NeMo already started on this
        (nemo_dir / "old_video_nemo_16k_full.wav").write_bytes(b"\x00" * 10)
        fresh = nemo_dir / "new_video.mp4"
        fresh.write_bytes(b"\x00" * 10)

        result = _find_video(target_lang="fr", nemo_dir=nemo_dir,
                             end_product_dir=tmp_path / "end_product")
        assert result == fresh

    def test_all_processed_returns_first_video(self, tmp_path):
        """When all videos are processed, returns the first (for re-processing)."""
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        video = nemo_dir / "video.mp4"
        video.write_bytes(b"\x00" * 10)
        end_product = tmp_path / "end_product"
        (end_product / "video_nemo_de__de_to_fr").mkdir(parents=True)

        result = _find_video(target_lang="fr", nemo_dir=nemo_dir,
                             end_product_dir=end_product)
        # Should still return the video (fallback to first)
        assert result == video


# ═════════════════════════════════════════════════════════════════════════════
# 4. _find_srt_for_video()
# ═════════════════════════════════════════════════════════════════════════════

class TestFindSrtForVideo:

    def test_finds_srt_in_nemo_dir(self, tmp_path):
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        srt = nemo_dir / "video.nemo.de.diarize.srt"
        srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n\n")

        result = _find_srt_for_video(
            "video", "*.nemo.de.diarize.srt",
            nemo_dir=nemo_dir, end_product_dir=tmp_path / "end_product")
        assert result == srt

    def test_finds_srt_in_end_product_after_cleanup(self, tmp_path):
        """After clean_subs moves files, SRTs live in end_product/<run>/."""
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        end_product = tmp_path / "end_product"
        run_dir = end_product / "video_nemo_de__de_to_fr"
        run_dir.mkdir(parents=True)
        srt = run_dir / "video.nemo.de.diarize.srt"
        srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n\n")

        result = _find_srt_for_video(
            "video", "*.nemo.de.diarize.srt",
            nemo_dir=nemo_dir, end_product_dir=end_product)
        assert result == srt

    def test_prefers_nemo_dir_over_end_product(self, tmp_path):
        """If SRT exists in both locations, nemo_dir takes priority."""
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        srt_live = nemo_dir / "video.nemo.de.diarize.srt"
        srt_live.write_text("live version\n")

        end_product = tmp_path / "end_product"
        run_dir = end_product / "video_nemo_de__de_to_fr"
        run_dir.mkdir(parents=True)
        srt_archived = run_dir / "video.nemo.de.diarize.srt"
        srt_archived.write_text("archived version\n")

        result = _find_srt_for_video(
            "video", "*.nemo.de.diarize.srt",
            nemo_dir=nemo_dir, end_product_dir=end_product)
        assert result == srt_live

    def test_fuzzy_matching_spaces_vs_underscores(self, tmp_path):
        """'My Video' base should match 'My_Video.nemo.de.diarize.srt'."""
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        srt = nemo_dir / "My_Video.nemo.de.diarize.srt"
        srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nTest\n\n")

        result = _find_srt_for_video(
            "My Video", "*.nemo.de.diarize.srt",
            nemo_dir=nemo_dir, end_product_dir=tmp_path / "end_product")
        assert result == srt

    def test_no_matching_srt_returns_none(self, tmp_path):
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        # SRT exists but for a different video
        (nemo_dir / "other.nemo.de.diarize.srt").write_text("test\n")

        result = _find_srt_for_video(
            "video", "*.nemo.de.diarize.srt",
            nemo_dir=nemo_dir, end_product_dir=tmp_path / "end_product")
        assert result is None

    def test_finds_translated_srt(self, tmp_path):
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        srt = nemo_dir / "video.nemo.de.diarize_fr.srt"
        srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nBonjour\n\n")

        result = _find_srt_for_video(
            "video", "*.diarize_fr.srt",
            nemo_dir=nemo_dir, end_product_dir=tmp_path / "end_product")
        assert result == srt

    def test_trim_suffix_matching(self, tmp_path):
        """video_t40 base should match video_t40.nemo.de.diarize.srt."""
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        srt = nemo_dir / "video_t40.nemo.de.diarize.srt"
        srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nTest\n\n")

        result = _find_srt_for_video(
            "video_t40", "*.nemo.de.diarize.srt",
            nemo_dir=nemo_dir, end_product_dir=tmp_path / "end_product")
        assert result == srt


# ═════════════════════════════════════════════════════════════════════════════
# 5. _derive_run_label()
# ═════════════════════════════════════════════════════════════════════════════

class TestDeriveRunLabel:

    def test_basic_label(self, tmp_path):
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        video = nemo_dir / "video.mp4"
        video.touch()
        end_product = tmp_path / "end_product"
        end_product.mkdir()

        label = _derive_run_label("de", "fr", video=video,
                                   nemo_dir=nemo_dir, end_product_dir=end_product)
        assert "__de_to_fr" in label
        assert "video" in label.lower()

    def test_uses_srt_stem_when_available(self, tmp_path):
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        video = nemo_dir / "video.mp4"
        video.touch()
        srt = nemo_dir / "video.nemo.de.diarize.srt"
        srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nTest\n\n")
        end_product = tmp_path / "end_product"
        end_product.mkdir()

        label = _derive_run_label("de", "fr", video=video,
                                   nemo_dir=nemo_dir, end_product_dir=end_product)
        assert "video.nemo.de.diarize" in label

    def test_increments_on_collision(self, tmp_path):
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        video = nemo_dir / "video.mp4"
        video.touch()
        end_product = tmp_path / "end_product"
        end_product.mkdir()

        # Create first run dir
        first_label = _derive_run_label("de", "fr", video=video,
                                         nemo_dir=nemo_dir, end_product_dir=end_product)
        (end_product / first_label).mkdir()

        # Second call should increment
        second_label = _derive_run_label("de", "fr", video=video,
                                          nemo_dir=nemo_dir, end_product_dir=end_product)
        assert second_label != first_label
        assert "__2" in second_label

    def test_fallback_when_no_video(self, tmp_path):
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        end_product = tmp_path / "end_product"
        end_product.mkdir()

        label = _derive_run_label("de", "fr", video=None,
                                   nemo_dir=nemo_dir, end_product_dir=end_product)
        assert "__de_to_fr" in label
        assert "run_" in label  # timestamp fallback

    def test_sanitizes_special_chars(self, tmp_path):
        nemo_dir = tmp_path / "nemo"
        nemo_dir.mkdir()
        video = nemo_dir / "My Video (2024) [HD].mp4"
        video.touch()
        end_product = tmp_path / "end_product"
        end_product.mkdir()

        label = _derive_run_label("de", "fr", video=video,
                                   nemo_dir=nemo_dir, end_product_dir=end_product)
        # Should not contain problematic filesystem chars
        assert "[" not in label
        assert "]" not in label
        assert "(" not in label
        assert ")" not in label


# ═════════════════════════════════════════════════════════════════════════════
# 6. _validate_translated_srt() — catch silent translation failures
# ═════════════════════════════════════════════════════════════════════════════

class TestValidateTranslatedSrt:

    def test_valid_srt_passes(self, tmp_path):
        srt = tmp_path / "video.diarize_fr.srt"
        srt.write_text(textwrap.dedent("""
            1
            00:00:01,000 --> 00:00:03,000
            [Speaker 1] C'est bien pour la santé.

            2
            00:00:03,500 --> 00:00:05,000
            [Speaker 2] Oui, exactement.
        """).strip())

        _validate_translated_srt(srt, "fr")  # Should not raise

    def test_empty_srt_fails(self, tmp_path):
        srt = tmp_path / "empty.srt"
        srt.write_text("")

        with pytest.raises(SystemExit):
            _validate_translated_srt(srt, "fr")

    def test_speaker_tags_only_fails(self, tmp_path):
        """If every line is just [Speaker N] with no translated content, fail."""
        srt = tmp_path / "broken.srt"
        srt.write_text(textwrap.dedent("""
            1
            00:00:01,000 --> 00:00:03,000
            [Speaker 1]

            2
            00:00:03,500 --> 00:00:05,000
            [Speaker 2]
        """).strip())

        with pytest.raises(SystemExit):
            _validate_translated_srt(srt, "fr")

    def test_mixed_content_passes(self, tmp_path):
        """Some lines with speaker tags + content, some without — should pass."""
        srt = tmp_path / "mixed.srt"
        srt.write_text(textwrap.dedent("""
            1
            00:00:01,000 --> 00:00:03,000
            [Speaker 1] Bonjour le monde.

            2
            00:00:03,500 --> 00:00:05,000
            Texte sans tag.
        """).strip())

        _validate_translated_srt(srt, "fr")  # Should not raise

    def test_only_timestamps_no_content_fails(self, tmp_path):
        """SRT with timestamps but no actual text content."""
        srt = tmp_path / "no_content.srt"
        srt.write_text(textwrap.dedent("""
            1
            00:00:01,000 --> 00:00:03,000

            2
            00:00:03,500 --> 00:00:05,000
        """).strip())

        with pytest.raises(SystemExit):
            _validate_translated_srt(srt, "fr")

    def test_whitespace_only_content_fails(self, tmp_path):
        """SRT where content lines are only whitespace."""
        srt = tmp_path / "whitespace.srt"
        srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n   \n\n")

        with pytest.raises(SystemExit):
            _validate_translated_srt(srt, "fr")

    def test_partial_speaker_tag_failure(self, tmp_path):
        """Some lines have content, but most are empty speaker tags.
        This should still pass since not ALL are speaker-only."""
        srt = tmp_path / "partial.srt"
        srt.write_text(textwrap.dedent("""
            1
            00:00:01,000 --> 00:00:02,000
            [Speaker 1] Real content here.

            2
            00:00:02,000 --> 00:00:03,000
            [Speaker 2]

            3
            00:00:03,000 --> 00:00:04,000
            [Speaker 1] More real content.
        """).strip())

        _validate_translated_srt(srt, "fr")  # Should pass — not ALL are empty
