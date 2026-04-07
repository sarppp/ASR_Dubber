"""
tests/test_clean_subs.py — Comprehensive tests for translate-gemma/clean_subs.py
================================================================================

Tests the four main functions:
  1. clean_srt_files()     — strip [Speaker N] tags, create *_clean.srt
  2. move_final_products() — gather SRTs/MP4s/intermediates into end_product/<run>/
  3. copy_source_video()   — move source video into run folder
  4. cleanup_wav_chunks()  — delete leftover _chunk_XXXX.wav files

These functions run as Step 4 in the pipeline (via _finalize_outputs).
The user's primary complaint: speaker tags not being removed properly.

Run:
    uv run --with "pytest,pysrt" pytest tests/test_clean_subs.py -v
"""

import os
import shutil
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import pysrt


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_srt(path: Path, content: str) -> Path:
    """Write SRT content to a file and return the path."""
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return path


def _make_diarized_srt(path: Path, entries: list[tuple[int, str, str, str]]) -> Path:
    """Create a diarized SRT file.

    entries: list of (index, start_ts, end_ts, text_with_speaker_tag)
    """
    lines = []
    for idx, start, end, text in entries:
        lines.extend([str(idx), f"{start} --> {end}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def workspace(tmp_path):
    """Set up a mock nemo_dir and end_product_dir."""
    nemo_dir = tmp_path / "nemo"
    nemo_dir.mkdir()
    end_product = tmp_path / "end_product"
    end_product.mkdir()
    return nemo_dir, end_product


# ═════════════════════════════════════════════════════════════════════════════
# 1. clean_srt_files() — speaker tag stripping
# ═════════════════════════════════════════════════════════════════════════════


class TestCleanSrtFiles:
    """Tests for clean_srt_files() — the [Speaker N] tag removal logic."""

    def test_strips_speaker_tags_basic(self, workspace):
        """Basic [Speaker 1] tags are removed and _clean.srt is created."""
        nemo_dir, end_product = workspace
        srt = _make_diarized_srt(nemo_dir / "video.nemo.de.diarize.srt", [
            (1, "00:00:01,000", "00:00:03,000", "[Speaker 1] Das ist gut."),
            (2, "00:00:03,500", "00:00:05,000", "[Speaker 2] Ja, genau."),
        ])

        with patch("clean_subs.NEMO_DIR", nemo_dir), \
             patch("clean_subs.END_PRODUCT_DIR", end_product):
            from clean_subs import clean_srt_files
            clean_srt_files()

        clean_path = nemo_dir / "video.nemo.de.diarize_clean.srt"
        assert clean_path.exists(), "No _clean.srt was created"
        clean_subs_content = pysrt.open(str(clean_path))
        assert clean_subs_content[0].text == "Das ist gut."
        assert clean_subs_content[1].text == "Ja, genau."

    def test_strips_multi_digit_speaker_tags(self, workspace):
        """[Speaker 10], [Speaker 99] etc. are properly stripped."""
        nemo_dir, end_product = workspace
        _make_diarized_srt(nemo_dir / "video.nemo.de.diarize.srt", [
            (1, "00:00:01,000", "00:00:02,000", "[Speaker 10] Hallo"),
            (2, "00:00:02,000", "00:00:03,000", "[Speaker 99] Tschüss"),
        ])

        with patch("clean_subs.NEMO_DIR", nemo_dir), \
             patch("clean_subs.END_PRODUCT_DIR", end_product):
            from clean_subs import clean_srt_files
            clean_srt_files()

        clean_path = nemo_dir / "video.nemo.de.diarize_clean.srt"
        assert clean_path.exists()
        content = pysrt.open(str(clean_path))
        assert content[0].text == "Hallo"
        assert content[1].text == "Tschüss"

    def test_no_speaker_tags_no_clean_file(self, workspace):
        """If SRT has no speaker tags, no _clean.srt is created (nothing to clean)."""
        nemo_dir, end_product = workspace
        _write_srt(nemo_dir / "video.nemo.de.diarize.srt", """
            1
            00:00:01,000 --> 00:00:02,000
            Normaler Text ohne Tags

        """)

        with patch("clean_subs.NEMO_DIR", nemo_dir), \
             patch("clean_subs.END_PRODUCT_DIR", end_product):
            from clean_subs import clean_srt_files
            clean_srt_files()

        clean_path = nemo_dir / "video.nemo.de.diarize_clean.srt"
        assert not clean_path.exists(), "Clean file created when no tags to strip"

    def test_skips_existing_clean_files(self, workspace):
        """Files already ending in _clean.srt are not re-processed."""
        nemo_dir, end_product = workspace
        _make_diarized_srt(nemo_dir / "video_clean.srt", [
            (1, "00:00:01,000", "00:00:02,000", "[Speaker 1] Should not be processed"),
        ])

        with patch("clean_subs.NEMO_DIR", nemo_dir), \
             patch("clean_subs.END_PRODUCT_DIR", end_product):
            from clean_subs import clean_srt_files
            clean_srt_files()

        # Should not create video_clean_clean.srt
        assert not (nemo_dir / "video_clean_clean.srt").exists()

    def test_preserves_translated_srt_speaker_tags(self, workspace):
        """Translated SRTs (video_fr.srt) with speaker tags are also cleaned."""
        nemo_dir, end_product = workspace
        _make_diarized_srt(nemo_dir / "video.nemo.de.diarize_fr.srt", [
            (1, "00:00:01,000", "00:00:03,000", "[Speaker 1] C'est bien pour la santé."),
            (2, "00:00:03,500", "00:00:05,000", "[Speaker 2] Oui, exactement."),
        ])

        with patch("clean_subs.NEMO_DIR", nemo_dir), \
             patch("clean_subs.END_PRODUCT_DIR", end_product):
            from clean_subs import clean_srt_files
            clean_srt_files()

        clean_path = nemo_dir / "video.nemo.de.diarize_fr_clean.srt"
        assert clean_path.exists(), "Translated SRT was not cleaned"
        content = pysrt.open(str(clean_path))
        assert "[Speaker" not in content[0].text

    def test_speaker_tag_with_no_space_after(self, workspace):
        """Edge case: [Speaker 1]text (no space after bracket)."""
        nemo_dir, end_product = workspace
        _make_diarized_srt(nemo_dir / "video.nemo.de.diarize.srt", [
            (1, "00:00:01,000", "00:00:02,000", "[Speaker 1]NoSpace"),
        ])

        with patch("clean_subs.NEMO_DIR", nemo_dir), \
             patch("clean_subs.END_PRODUCT_DIR", end_product):
            from clean_subs import clean_srt_files
            clean_srt_files()

        clean_path = nemo_dir / "video.nemo.de.diarize_clean.srt"
        assert clean_path.exists()
        content = pysrt.open(str(clean_path))
        # The regex in clean_subs.py uses \s* so this should still work
        assert "[Speaker" not in content[0].text
        assert "NoSpace" in content[0].text

    def test_empty_srt_directory(self, workspace):
        """No SRT files in directory — should not crash."""
        nemo_dir, end_product = workspace

        with patch("clean_subs.NEMO_DIR", nemo_dir), \
             patch("clean_subs.END_PRODUCT_DIR", end_product):
            from clean_subs import clean_srt_files
            clean_srt_files()  # Should not raise

    def test_multiple_srt_files_all_cleaned(self, workspace):
        """Multiple SRT files in the directory — all get cleaned."""
        nemo_dir, end_product = workspace
        for name in ["video1.nemo.de.diarize.srt", "video2.nemo.en.diarize.srt"]:
            _make_diarized_srt(nemo_dir / name, [
                (1, "00:00:01,000", "00:00:02,000", "[Speaker 1] Text here"),
            ])

        with patch("clean_subs.NEMO_DIR", nemo_dir), \
             patch("clean_subs.END_PRODUCT_DIR", end_product):
            from clean_subs import clean_srt_files
            clean_srt_files()

        assert (nemo_dir / "video1.nemo.de.diarize_clean.srt").exists()
        assert (nemo_dir / "video2.nemo.en.diarize_clean.srt").exists()


# ═════════════════════════════════════════════════════════════════════════════
# 2. move_final_products() — file gathering
# ═════════════════════════════════════════════════════════════════════════════

class TestMoveFinalProducts:
    """Tests for move_final_products() — moving SRTs/MP4s to end_product/."""

    def test_moves_all_srts_to_run_dir(self, workspace):
        nemo_dir, end_product = workspace
        # Create some SRT files
        for name in ["video.nemo.de.diarize.srt", "video.nemo.de.diarize_fr.srt",
                      "video.nemo.de.diarize_clean.srt"]:
            (nemo_dir / name).write_text("1\n00:00:01,000 --> 00:00:02,000\nTest\n\n")

        run_label = "video_nemo_de__de_to_fr"

        with patch("clean_subs.NEMO_DIR", nemo_dir), \
             patch("clean_subs.END_PRODUCT_DIR", end_product):
            from clean_subs import move_final_products
            dest = move_final_products(run_label)

        assert dest == end_product / run_label
        assert (dest / "video.nemo.de.diarize.srt").exists()
        assert (dest / "video.nemo.de.diarize_fr.srt").exists()
        # Original files should be gone from nemo_dir
        assert not (nemo_dir / "video.nemo.de.diarize.srt").exists()

    def test_moves_intermediate_json_files(self, workspace):
        """JSON transcript files matching the run label base are moved."""
        nemo_dir, end_product = workspace
        (nemo_dir / "video_nemo_de_transcript.json").write_text('{"words": []}')
        (nemo_dir / "unrelated_file.json").write_text('{}')

        run_label = "video_nemo_de__de_to_fr"

        with patch("clean_subs.NEMO_DIR", nemo_dir), \
             patch("clean_subs.END_PRODUCT_DIR", end_product):
            from clean_subs import move_final_products
            move_final_products(run_label)

        dest = end_product / run_label
        assert (dest / "video_nemo_de_transcript.json").exists()
        # Unrelated file should stay
        assert (nemo_dir / "unrelated_file.json").exists()

    def test_moves_dubbed_mp4_from_workdir(self, workspace):
        """Final dubbed MP4 is moved from dub_workdir/output/."""
        nemo_dir, end_product = workspace
        dub_workdir = workspace[0].parent / "qwen3-tts" / "output" / "dub" / "video"
        dub_output = dub_workdir / "output"
        dub_output.mkdir(parents=True)
        (dub_output / "final_dub.mp4").write_bytes(b"\x00" * 100)

        run_label = "video_nemo_de__de_to_fr"

        with patch("clean_subs.NEMO_DIR", nemo_dir), \
             patch("clean_subs.END_PRODUCT_DIR", end_product):
            from clean_subs import move_final_products
            move_final_products(run_label, dub_workdir=str(dub_workdir))

        dest = end_product / run_label
        assert (dest / "final_dub.mp4").exists()

    def test_no_files_to_move(self, workspace):
        """Empty directory — no crash, returns destination."""
        nemo_dir, end_product = workspace
        run_label = "empty_run"

        with patch("clean_subs.NEMO_DIR", nemo_dir), \
             patch("clean_subs.END_PRODUCT_DIR", end_product):
            from clean_subs import move_final_products
            dest = move_final_products(run_label)

        assert dest == end_product / run_label

    def test_no_run_label_uses_flat_end_product(self, workspace):
        """When run_label is None, files go to end_product/ directly."""
        nemo_dir, end_product = workspace
        (nemo_dir / "test.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nHi\n\n")

        with patch("clean_subs.NEMO_DIR", nemo_dir), \
             patch("clean_subs.END_PRODUCT_DIR", end_product):
            from clean_subs import move_final_products
            dest = move_final_products(None)

        assert dest == end_product
        assert (end_product / "test.srt").exists()


# ═════════════════════════════════════════════════════════════════════════════
# 3. copy_source_video() — source video movement
# ═════════════════════════════════════════════════════════════════════════════

class TestCopySourceVideo:
    """Tests for copy_source_video() — moves original video to run folder."""

    def test_moves_matching_video(self, workspace):
        nemo_dir, end_product = workspace
        (nemo_dir / "My Video.mp4").write_bytes(b"\x00" * 50)
        run_label = "My_Video_nemo_de__de_to_fr"
        (end_product / run_label).mkdir(parents=True)

        with patch("clean_subs.NEMO_DIR", nemo_dir), \
             patch("clean_subs.END_PRODUCT_DIR", end_product):
            from clean_subs import copy_source_video
            copy_source_video(run_label)

        assert (end_product / run_label / "My Video.mp4").exists()
        assert not (nemo_dir / "My Video.mp4").exists()

    def test_skips_when_already_in_destination(self, workspace):
        nemo_dir, end_product = workspace
        run_label = "video_nemo_de__de_to_fr"
        dest_dir = end_product / run_label
        dest_dir.mkdir(parents=True)
        (nemo_dir / "video.mp4").write_bytes(b"\x00" * 50)
        (dest_dir / "video.mp4").write_bytes(b"\x00" * 50)

        with patch("clean_subs.NEMO_DIR", nemo_dir), \
             patch("clean_subs.END_PRODUCT_DIR", end_product):
            from clean_subs import copy_source_video
            copy_source_video(run_label)

        # Source should still exist (not moved because dest already has it)
        assert (nemo_dir / "video.mp4").exists()

    def test_no_run_label_is_noop(self, workspace):
        """copy_source_video(None) should be a no-op."""
        nemo_dir, end_product = workspace
        (nemo_dir / "video.mp4").write_bytes(b"\x00" * 50)

        with patch("clean_subs.NEMO_DIR", nemo_dir), \
             patch("clean_subs.END_PRODUCT_DIR", end_product):
            from clean_subs import copy_source_video
            copy_source_video(None)

        # Video should still be in nemo_dir
        assert (nemo_dir / "video.mp4").exists()

    def test_handles_wav_input(self, workspace):
        """WAV inputs have _nemo_16k_full baked in — strip before matching."""
        nemo_dir, end_product = workspace
        (nemo_dir / "audio_nemo_16k_full.wav").write_bytes(b"\x00" * 50)
        run_label = "audio_nemo_de__de_to_fr"
        (end_product / run_label).mkdir(parents=True)

        with patch("clean_subs.NEMO_DIR", nemo_dir), \
             patch("clean_subs.END_PRODUCT_DIR", end_product):
            from clean_subs import copy_source_video
            copy_source_video(run_label)

        assert (end_product / run_label / "audio_nemo_16k_full.wav").exists()

    def test_strips_trim_suffix_for_matching(self, workspace):
        """Run label with _t40 trim suffix should match video without it."""
        nemo_dir, end_product = workspace
        (nemo_dir / "video.mp4").write_bytes(b"\x00" * 50)
        run_label = "video_t40_nemo_de__de_to_fr"
        # The label_base after stripping _t40 should be "video"
        (end_product / run_label).mkdir(parents=True)

        with patch("clean_subs.NEMO_DIR", nemo_dir), \
             patch("clean_subs.END_PRODUCT_DIR", end_product):
            from clean_subs import copy_source_video
            copy_source_video(run_label)

        # Should find and move video.mp4
        assert (end_product / run_label / "video.mp4").exists()


# ═════════════════════════════════════════════════════════════════════════════
# 4. cleanup_wav_chunks() — leftover chunk cleanup
# ═════════════════════════════════════════════════════════════════════════════

class TestCleanupWavChunks:
    """Tests for cleanup_wav_chunks() — deletes _chunk_XXXX.wav files."""

    def test_deletes_chunk_files(self, workspace):
        nemo_dir, _ = workspace
        for i in range(5):
            (nemo_dir / f"_chunk_{i:04d}.wav").write_bytes(b"\x00" * 10)

        with patch("clean_subs.NEMO_DIR", nemo_dir):
            from clean_subs import cleanup_wav_chunks
            cleanup_wav_chunks()

        remaining = list(nemo_dir.glob("_chunk_*.wav"))
        assert len(remaining) == 0, f"Leftover chunks: {remaining}"

    def test_ignores_non_chunk_wav_files(self, workspace):
        nemo_dir, _ = workspace
        (nemo_dir / "video_nemo_16k_full.wav").write_bytes(b"\x00" * 10)
        (nemo_dir / "_chunk_0000.wav").write_bytes(b"\x00" * 10)

        with patch("clean_subs.NEMO_DIR", nemo_dir):
            from clean_subs import cleanup_wav_chunks
            cleanup_wav_chunks()

        assert (nemo_dir / "video_nemo_16k_full.wav").exists()
        assert not (nemo_dir / "_chunk_0000.wav").exists()

    def test_no_chunks_is_noop(self, workspace):
        nemo_dir, _ = workspace

        with patch("clean_subs.NEMO_DIR", nemo_dir):
            from clean_subs import cleanup_wav_chunks
            cleanup_wav_chunks()  # Should not raise
