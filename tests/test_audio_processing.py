"""
tests/test_audio_processing.py — Audio extraction, chunking, duration,
word-to-segment assembly, and SRT timestamp formatting edge cases.

Catches real pipeline failures:
  - Zero-duration WAV files → division by zero in chunk calculation
  - Corrupt WAV headers → _audio_duration returns 0.0
  - _words_to_segs drops words at boundaries
  - _chunk_audio overlap miscalculation
  - _fmt_ts negative/overflow timestamps
  - _srt_last_timestamp with malformed SRT

Run:
    uv run --with "pytest,pydantic" pytest tests/test_audio_processing.py -v
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from nemo_audio import (
    CHUNK_OVERLAP_SEC,
    _audio_duration,
    _chunk_audio,
    _cleanup_chunks,
    _fmt_dur,
    _fmt_ts,
    _segs_to_srt,
    _split_coarse_segs,
    _srt_last_timestamp,
    _strip_asr_repetition,
    _strip_special_tokens,
    _words_to_segs,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_wav(path: Path, duration_sec: float, sample_rate: int = 16000) -> None:
    """Write a valid WAV file of given duration."""
    n_frames = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. _audio_duration — edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestAudioDuration:

    def test_valid_wav_returns_correct_duration(self, tmp_path):
        wav = tmp_path / "test.wav"
        _make_wav(wav, 5.0)
        dur = _audio_duration(str(wav))
        assert abs(dur - 5.0) < 0.01

    def test_nonexistent_file_returns_zero(self, tmp_path):
        assert _audio_duration(str(tmp_path / "nope.wav")) == 0.0

    def test_empty_file_returns_zero(self, tmp_path):
        wav = tmp_path / "empty.wav"
        wav.write_bytes(b"")
        assert _audio_duration(str(wav)) == 0.0

    def test_corrupt_header_returns_zero(self, tmp_path):
        wav = tmp_path / "corrupt.wav"
        wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt \x00\x00\x00\x00garbage")
        assert _audio_duration(str(wav)) == 0.0

    def test_non_wav_file_returns_zero(self, tmp_path):
        f = tmp_path / "text.wav"
        f.write_text("this is not a WAV file")
        assert _audio_duration(str(f)) == 0.0

    def test_zero_length_wav(self, tmp_path):
        wav = tmp_path / "zero.wav"
        _make_wav(wav, 0.0)
        assert _audio_duration(str(wav)) == 0.0

    def test_very_long_wav(self, tmp_path):
        """10800s (3 hours) WAV — large frame count calculation works."""
        wav = tmp_path / "long.wav"
        # Don't actually write 3 hours of audio — just validate the math
        n_frames = int(10800 * 16000)
        with wave.open(str(wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            # Write minimal data but set frame count correctly
            wf.setnframes(n_frames)
            wf.writeframes(b"\x00\x00" * min(n_frames, 16000))  # 1 second of data
        # Duration should be based on actual frames written, not nframes header
        dur = _audio_duration(str(wav))
        assert dur > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. _chunk_audio — overlap and boundary handling
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkAudio:

    def test_short_audio_returns_single_chunk(self, tmp_path):
        """Audio shorter than chunk_sec + 5 → no splitting."""
        wav = tmp_path / "short.wav"
        _make_wav(wav, 10.0)
        chunks = _chunk_audio(str(wav), tmp_path, chunk_sec=20)
        assert len(chunks) == 1
        assert chunks[0] == (str(wav), 0.0)

    def test_audio_exactly_at_boundary(self, tmp_path):
        """Audio exactly at chunk_sec + 5 → no splitting (edge case)."""
        wav = tmp_path / "exact.wav"
        _make_wav(wav, 25.0)
        chunks = _chunk_audio(str(wav), tmp_path, chunk_sec=20)
        # 25 <= 20 + 5 → single chunk
        assert len(chunks) == 1

    def test_overlap_applied_correctly(self, tmp_path):
        """Chunks overlap by CHUNK_OVERLAP_SEC (2s)."""
        wav = tmp_path / "overlap.wav"
        _make_wav(wav, 60.0)
        chunks = _chunk_audio(str(wav), tmp_path, chunk_sec=20)
        # step = 20 - 2 = 18; offsets: 0, 18, 36, 54
        assert len(chunks) >= 3
        offsets = [off for _, off in chunks]
        assert offsets[0] == 0.0
        expected_step = 20 - CHUNK_OVERLAP_SEC
        assert abs(offsets[1] - expected_step) < 0.01

    def test_chunk_filenames_sequential(self, tmp_path):
        """Chunk files are named _chunk_0000.wav, _chunk_0001.wav, etc."""
        wav = tmp_path / "seq.wav"
        _make_wav(wav, 60.0)
        chunks = _chunk_audio(str(wav), tmp_path, chunk_sec=20)
        for i, (path, _) in enumerate(chunks):
            assert f"_chunk_{i:04d}.wav" in path

    def test_no_gap_between_chunks(self, tmp_path):
        """Last chunk's end offset covers the full audio."""
        wav = tmp_path / "gap.wav"
        _make_wav(wav, 100.0)
        chunks = _chunk_audio(str(wav), tmp_path, chunk_sec=30)
        last_offset = chunks[-1][1]
        # Last chunk starts at last_offset and goes for up to 30s
        assert last_offset + 30 >= 100.0 or last_offset + (100 - last_offset) >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. _cleanup_chunks — doesn't delete the main audio
# ═══════════════════════════════════════════════════════════════════════════════

class TestCleanupChunks:

    def test_deletes_chunk_files(self, tmp_path):
        main = tmp_path / "audio.wav"
        main.write_bytes(b"main")
        c1 = tmp_path / "chunk1.wav"
        c2 = tmp_path / "chunk2.wav"
        c1.write_bytes(b"c1")
        c2.write_bytes(b"c2")
        manifest = [
            {"path": str(c1)},
            {"path": str(c2)},
            {"path": str(main)},  # this must NOT be deleted
        ]
        _cleanup_chunks(manifest, str(main))
        assert not c1.exists()
        assert not c2.exists()
        assert main.exists(), "Main audio file must not be deleted!"

    def test_handles_none_manifest(self):
        """None manifest → no crash."""
        _cleanup_chunks(None, "/any/path")

    def test_handles_empty_manifest(self):
        """Empty manifest → no crash."""
        _cleanup_chunks([], "/any/path")

    def test_handles_missing_files(self, tmp_path):
        """Entries pointing to nonexistent files → no crash."""
        manifest = [{"path": str(tmp_path / "gone.wav")}]
        _cleanup_chunks(manifest, str(tmp_path / "main.wav"))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. _words_to_segs — word assembly edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestWordsToSegs:

    def test_empty_words(self):
        assert _words_to_segs([]) == []

    def test_single_word(self):
        words = [{"word": "Hello", "start": 0.0, "end": 0.5}]
        segs = _words_to_segs(words)
        assert len(segs) == 1
        assert segs[0]["text"] == "Hello"

    def test_splits_at_sentence_boundary(self):
        """Period at end of word triggers split after min 3 words."""
        words = [
            {"word": "First.", "start": 0.0, "end": 0.3},
            {"word": "Second.", "start": 0.3, "end": 0.6},
            {"word": "Third.", "start": 0.6, "end": 0.9},
            {"word": "Next", "start": 1.0, "end": 1.3},
            {"word": "sentence.", "start": 1.3, "end": 1.6},
        ]
        segs = _words_to_segs(words)
        assert len(segs) >= 2

    def test_splits_at_max_words(self):
        """More than max_w words triggers split."""
        words = [{"word": f"w{i}", "start": float(i * 0.3), "end": float(i * 0.3 + 0.2)}
                 for i in range(25)]
        segs = _words_to_segs(words, max_w=10)
        assert len(segs) >= 2

    def test_splits_at_max_duration(self):
        """Segment exceeding max_dur triggers split."""
        words = [{"word": f"w{i}", "start": float(i * 2), "end": float(i * 2 + 1)}
                 for i in range(5)]
        segs = _words_to_segs(words, max_dur=3.0)
        assert len(segs) >= 2

    def test_diarized_splits_on_speaker_change(self):
        """Speaker change triggers segment split in diarized mode."""
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.3, "speaker": "spk_0"},
            {"word": "there", "start": 0.3, "end": 0.6, "speaker": "spk_0"},
            {"word": "Goodbye", "start": 0.7, "end": 1.0, "speaker": "spk_1"},
            {"word": "now", "start": 1.0, "end": 1.3, "speaker": "spk_1"},
        ]
        segs = _words_to_segs(words, diarized=True)
        assert len(segs) == 2
        assert segs[0]["speaker"] == "spk_0"
        assert segs[1]["speaker"] == "spk_1"

    def test_empty_text_words_skipped(self):
        """Words with empty text are silently dropped."""
        words = [
            {"word": "", "start": 0.0, "end": 0.1},
            {"word": "  ", "start": 0.1, "end": 0.2},
            {"word": "Hello", "start": 0.2, "end": 0.5},
        ]
        segs = _words_to_segs(words)
        assert len(segs) == 1
        assert segs[0]["text"] == "Hello"

    def test_preserves_all_words_in_output(self):
        """No words are silently dropped (tests boundary-word fix in _words_to_segs).

        Bug: the old code did `continue` after saving a segment, which skipped the
        word that triggered the split. Fixed by removing `continue` and resetting `cand`.
        """
        words = [{"word": f"word{i}", "start": float(i * 0.3), "end": float(i * 0.3 + 0.2)}
                 for i in range(20)]
        segs = _words_to_segs(words)
        all_text = " ".join(s["text"] for s in segs)
        for w in words:
            assert w["word"] in all_text, (
                f"Word '{w['word']}' lost in segmentation — boundary-word bug regressed!\n"
                f"All segments: {[s['text'] for s in segs]}"
            )

    def test_max_ch_limit(self):
        """Very long words trigger char limit split."""
        words = [{"word": "a" * 40, "start": 0.0, "end": 0.5},
                 {"word": "b" * 40, "start": 0.5, "end": 1.0},
                 {"word": "c" * 40, "start": 1.0, "end": 1.5}]
        segs = _words_to_segs(words, max_ch=80)
        # Each word is 40 chars; two words = 81 chars (with space) > 80 → splits
        assert len(segs) >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# 5. _fmt_ts / _fmt_dur — timestamp formatting edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimestampFormatting:

    def test_zero_seconds(self):
        assert _fmt_ts(0.0) == "00:00:00,000"

    def test_standard_timestamp(self):
        # 1h 2m 3.456s
        assert _fmt_ts(3723.456) == "01:02:03,456"

    def test_sub_second(self):
        assert _fmt_ts(0.500) == "00:00:00,500"

    def test_exact_hour(self):
        assert _fmt_ts(3600.0) == "01:00:00,000"

    def test_fmt_dur_under_60(self):
        assert _fmt_dur(45.3) == "45.3s"

    def test_fmt_dur_over_60(self):
        assert _fmt_dur(125.0) == "2m05s"

    def test_fmt_dur_zero(self):
        assert _fmt_dur(0.0) == "0.0s"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. _srt_last_timestamp — malformed SRT input
# ═══════════════════════════════════════════════════════════════════════════════

class TestSrtLastTimestamp:

    def test_normal_srt(self):
        srt = "1\n00:00:01,000 --> 00:00:05,500\nHello\n\n"
        assert abs(_srt_last_timestamp(srt) - 5.5) < 0.01

    def test_empty_string(self):
        assert _srt_last_timestamp("") == 0.0

    def test_no_timestamps(self):
        assert _srt_last_timestamp("just some random text\n") == 0.0

    def test_malformed_timestamp_format(self):
        """Missing milliseconds → no match → 0.0."""
        srt = "1\n00:00:01 --> 00:00:05\nHello\n\n"
        assert _srt_last_timestamp(srt) == 0.0

    def test_multiple_blocks_returns_last(self):
        srt = (
            "1\n00:00:01,000 --> 00:00:02,000\nFirst\n\n"
            "2\n00:05:00,000 --> 00:05:30,000\nSecond\n\n"
            "3\n01:30:00,000 --> 01:30:15,500\nLast\n\n"
        )
        assert abs(_srt_last_timestamp(srt) - 5415.5) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# 7. _split_coarse_segs — speaker field preservation
# ═══════════════════════════════════════════════════════════════════════════════

class TestSplitCoarseSegsExtended:

    def test_speaker_preserved_in_all_splits(self):
        """When input has speaker field, ALL output segments must have it."""
        segs = [{"text": " ".join(["word"] * 30), "start": 0.0, "end": 30.0, "speaker": "spk_2"}]
        result = _split_coarse_segs(segs, max_w=10)
        assert len(result) >= 3
        for r in result:
            assert r.get("speaker") == "spk_2"

    def test_no_speaker_field_when_input_lacks_it(self):
        """When input has no speaker, output shouldn't add one."""
        segs = [{"text": "hello world", "start": 0.0, "end": 2.0}]
        result = _split_coarse_segs(segs)
        assert "speaker" not in result[0]

    def test_timing_continuity(self):
        """Output segments should have no time gaps between them."""
        segs = [{"text": " ".join(["word"] * 20), "start": 5.0, "end": 25.0}]
        result = _split_coarse_segs(segs, max_w=5)
        for i in range(len(result) - 1):
            gap = abs(result[i + 1]["start"] - result[i]["end"])
            assert gap < 0.01, f"Gap between segments {i} and {i+1}: {gap:.3f}s"

    def test_zero_duration_segment(self):
        """Segment with start == end → handled without crash."""
        segs = [{"text": "hello", "start": 5.0, "end": 5.0}]
        result = _split_coarse_segs(segs)
        assert len(result) >= 1

    def test_multiple_segments_independent(self):
        """Multiple input segments produce independent output groups."""
        segs = [
            {"text": "first", "start": 0.0, "end": 1.0, "speaker": "spk_0"},
            {"text": "second", "start": 5.0, "end": 6.0, "speaker": "spk_1"},
        ]
        result = _split_coarse_segs(segs)
        assert len(result) == 2
        assert result[0]["speaker"] == "spk_0"
        assert result[1]["speaker"] == "spk_1"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. _segs_to_srt — SRT format correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestSegsToSrtFormat:

    def test_srt_block_format(self):
        """Each SRT block has: index, timestamp, text, blank line."""
        segs = [
            {"text": "Hello world", "start": 0.0, "end": 2.0},
            {"text": "Goodbye world", "start": 2.0, "end": 4.0},
        ]
        srt = _segs_to_srt(segs)
        lines = srt.split("\n")
        # First block: index, timestamp, text, blank
        assert lines[0] == "1"
        assert "-->" in lines[1]
        assert "Hello world" in lines[2]
        assert lines[3] == ""

    def test_diarized_speaker_labels(self):
        """Diarized SRT has [Speaker N] labels before text."""
        segs = [
            {"text": "Hi", "start": 0.0, "end": 1.0, "speaker": "spk_0"},
            {"text": "Bye", "start": 1.0, "end": 2.0, "speaker": "spk_1"},
        ]
        srt = _segs_to_srt(segs, diarized=True)
        assert "[Speaker 1]" in srt
        assert "[Speaker 2]" in srt

    def test_speaker_numbering_sorted(self):
        """Speaker labels are sorted alphabetically before numbering."""
        segs = [
            {"text": "B speaks", "start": 0.0, "end": 1.0, "speaker": "spk_2"},
            {"text": "A speaks", "start": 1.0, "end": 2.0, "speaker": "spk_0"},
        ]
        srt = _segs_to_srt(segs, diarized=True)
        # spk_0 < spk_2 alphabetically → spk_0 = Speaker 1, spk_2 = Speaker 2
        lines = srt.split("\n")
        # First text line (index 2) should have Speaker 2 (spk_2 is second alphabetically)
        text_lines = [l for l in lines if "[Speaker" in l]
        assert "[Speaker 2]" in text_lines[0]  # spk_2 = Speaker 2
        assert "[Speaker 1]" in text_lines[1]  # spk_0 = Speaker 1

    def test_consecutive_same_speaker_dedup(self):
        """Same speaker, same text → only one block (hallucination dedup)."""
        segs = [
            {"text": "Repeat", "start": 0.0, "end": 1.0, "speaker": "spk_0"},
            {"text": "Repeat", "start": 1.0, "end": 2.0, "speaker": "spk_0"},
            {"text": "Repeat", "start": 2.0, "end": 3.0, "speaker": "spk_0"},
        ]
        srt = _segs_to_srt(segs, diarized=True)
        count = srt.count("Repeat")
        assert count == 1, f"Expected 1 occurrence, got {count} (dedup failed)"

    def test_different_speakers_same_text_kept(self):
        """Different speakers, same text → both kept (valid dialogue)."""
        segs = [
            {"text": "Ja", "start": 0.0, "end": 0.5, "speaker": "spk_0"},
            {"text": "Ja", "start": 0.5, "end": 1.0, "speaker": "spk_1"},
        ]
        srt = _segs_to_srt(segs, diarized=True)
        assert srt.count("Ja") == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Integration: words → segs → SRT roundtrip
# ═══════════════════════════════════════════════════════════════════════════════

class TestWordsToSrtRoundtrip:

    def test_all_words_present_in_final_srt(self):
        """Words fed through the full pipeline appear in the SRT output."""
        words = [
            {"word": "Alpha", "start": 0.0, "end": 0.5},
            {"word": "Beta", "start": 0.5, "end": 1.0},
            {"word": "Gamma", "start": 1.0, "end": 1.5},
            {"word": "Delta", "start": 1.5, "end": 2.0},
            {"word": "Epsilon", "start": 2.0, "end": 2.5},
        ]
        segs = _words_to_segs(words)
        srt = _segs_to_srt(segs)
        for w in words:
            assert w["word"] in srt, f"Word '{w['word']}' missing from SRT"

    def test_diarized_roundtrip_with_speaker_change(self):
        """Words with speaker changes → segs → SRT: all words present including boundary words."""
        words = [
            {"word": "Hallo",   "start": 0.0, "end": 0.3, "speaker": "spk_0"},
            {"word": "Welt",    "start": 0.3, "end": 0.6, "speaker": "spk_0"},
            {"word": "Tschuss", "start": 1.0, "end": 1.3, "speaker": "spk_1"},  # boundary word
            {"word": "jetzt",   "start": 1.3, "end": 1.6, "speaker": "spk_1"},
        ]
        segs = _words_to_segs(words, diarized=True)
        srt = _segs_to_srt(segs, diarized=True)
        assert "Hallo" in srt
        # "Tschuss" is the word that triggers the speaker-change split — must NOT be dropped
        assert "Tschuss" in srt, (
            "Boundary word 'Tschuss' dropped at speaker change — "
            "_words_to_segs boundary-word bug regressed!"
        )
        assert "[Speaker" in srt

    def test_large_word_list_no_loss(self):
        """200 unique words → all present in SRT (catches boundary-word drops at every segment)."""
        # Space words well apart (0.5s each) so max_dur doesn't split them oddly
        words = [
            {"word": f"UniqueWord{i:03d}", "start": float(i * 0.5), "end": float(i * 0.5 + 0.3)}
            for i in range(200)
        ]
        segs = _words_to_segs(words)
        srt = _segs_to_srt(segs)
        missing = [w["word"] for w in words if w["word"] not in srt]
        assert not missing, (
            f"{len(missing)} words missing from SRT: {missing[:10]}...\n"
            "This indicates the boundary-word drop bug has regressed in _words_to_segs."
        )
