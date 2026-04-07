"""
tests/test_dub_srt.py — Tests for qwen3-tts/dub_srt.py
========================================================

Tests the SRT parsing, voice assignment, and language mapping used in Step 3.

  1. parse_srt()       — parse diarized+translated SRT for dubbing
  2. build_voice_map() — assign Qwen voices to speakers
  3. _qwen_lang()      — ISO code to Qwen language name
  4. _srt_ts()         — timestamp parsing

Run:
    uv run --with pytest pytest tests/test_dub_srt.py -v
"""

import sys
import textwrap
from pathlib import Path

import pytest

# Add qwen3-tts to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "qwen3-tts"))

from dub_srt import (
    parse_srt,
    build_voice_map,
    merge_segments,
    write_dub_srt,
    _split_subtitle_blocks,
    _qwen_lang,
    _srt_ts,
    _fmt_ts,
    QWEN_FEMALE_VOICES,
    QWEN_MALE_VOICES,
    LANG_CODE_TO_QWEN,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_srt(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# 1. _srt_ts() — timestamp parsing
# ═════════════════════════════════════════════════════════════════════════════

class TestSrtTimestamp:

    @pytest.mark.parametrize("ts,expected", [
        ("00:00:00,000", 0.0),
        ("00:00:01,000", 1.0),
        ("00:00:01,500", 1.5),
        ("00:01:30,000", 90.0),
        ("01:00:00,000", 3600.0),
        ("01:02:03,456", 3723.456),
        ("00:00:00.500", 0.5),   # dot format (ffmpeg style)
    ], ids=["zero", "1s", "1.5s", "1m30s", "1h", "complex", "dot-format"])
    def test_timestamp_parsing(self, ts, expected):
        result = _srt_ts(ts)
        assert abs(result - expected) < 0.001, f"Expected {expected}, got {result}"


# ═════════════════════════════════════════════════════════════════════════════
# 2. _qwen_lang() — language code mapping
# ═════════════════════════════════════════════════════════════════════════════

class TestQwenLang:

    @pytest.mark.parametrize("code,expected", [
        ("fr", "french"),
        ("en", "english"),
        ("de", "german"),
        ("es", "spanish"),
        ("it", "italian"),
        ("ja", "japanese"),
        ("ko", "korean"),
        ("zh", "chinese"),
        ("auto", "auto"),
    ])
    def test_known_codes(self, code, expected):
        assert _qwen_lang(code) == expected

    def test_unknown_code_passed_through(self):
        """Unknown codes are returned as-is (with a warning logged)."""
        result = _qwen_lang("xx")
        assert result == "xx"

    def test_case_insensitive(self):
        assert _qwen_lang("FR") == "french"
        assert _qwen_lang("En") == "english"

    def test_whitespace_stripped(self):
        assert _qwen_lang("  fr  ") == "french"


# ═════════════════════════════════════════════════════════════════════════════
# 3. parse_srt() — diarized + translated SRT parsing
# ═════════════════════════════════════════════════════════════════════════════

class TestParseSrt:

    def test_basic_diarized_srt(self, tmp_path):
        """Standard pipeline output: [Speaker N] translated_text."""
        srt = _write_srt(tmp_path / "test.srt", """
            1
            00:00:01,000 --> 00:00:03,500
            [Speaker 1] C'est bien pour la santé.

            2
            00:00:03,600 --> 00:00:05,100
            [Speaker 2] Oui, exactement.
        """)

        result = parse_srt(srt)
        assert len(result) == 2
        assert result[0]["speaker"] == "Speaker 1"
        assert result[0]["text"] == "C'est bien pour la santé."
        assert result[0]["start"] == pytest.approx(1.0, abs=0.01)
        assert result[0]["end"] == pytest.approx(3.5, abs=0.01)
        assert result[1]["speaker"] == "Speaker 2"
        assert result[1]["text"] == "Oui, exactement."

    def test_no_speaker_tags_defaults_to_speaker_1(self, tmp_path):
        """SRT without [Speaker N] tags — defaults to Speaker 1."""
        srt = _write_srt(tmp_path / "test.srt", """
            1
            00:00:01,000 --> 00:00:03,000
            Bonjour le monde.

            2
            00:00:03,500 --> 00:00:05,000
            Au revoir.
        """)

        result = parse_srt(srt)
        assert len(result) == 2
        assert result[0]["speaker"] == "Speaker 1"
        assert result[0]["text"] == "Bonjour le monde."
        assert result[1]["speaker"] == "Speaker 1"

    def test_multi_digit_speaker_numbers(self, tmp_path):
        """Handles [Speaker 10], [Speaker 99] etc."""
        srt = _write_srt(tmp_path / "test.srt", """
            1
            00:00:01,000 --> 00:00:02,000
            [Speaker 10] Text ten.

            2
            00:00:02,000 --> 00:00:03,000
            [Speaker 99] Text ninety-nine.
        """)

        result = parse_srt(srt)
        assert result[0]["speaker"] == "Speaker 10"
        assert result[1]["speaker"] == "Speaker 99"

    def test_continuation_lines_joined(self, tmp_path):
        """Long translations that wrap to multiple lines are joined."""
        srt = _write_srt(tmp_path / "test.srt", """
            1
            00:00:01,000 --> 00:00:05,000
            [Speaker 1] Ceci est une très longue phrase
            qui continue sur la ligne suivante.
        """)

        result = parse_srt(srt)
        assert len(result) == 1
        assert "très longue phrase" in result[0]["text"]
        assert "qui continue" in result[0]["text"]

    def test_pipe_separators_converted_to_spaces(self, tmp_path):
        """Pipe-encoded newlines from translate.py are converted to spaces."""
        srt = _write_srt(tmp_path / "test.srt", """
            1
            00:00:01,000 --> 00:00:03,000
            [Speaker 1] Première partie | deuxième partie
        """)

        result = parse_srt(srt)
        assert "|" not in result[0]["text"]

    def test_empty_text_after_tag_is_skipped(self, tmp_path):
        """Segments where text is empty after removing speaker tag are skipped."""
        srt = _write_srt(tmp_path / "test.srt", """
            1
            00:00:01,000 --> 00:00:02,000
            [Speaker 1]

            2
            00:00:02,000 --> 00:00:03,000
            [Speaker 2] Real content.
        """)

        result = parse_srt(srt)
        assert len(result) == 1
        assert result[0]["text"] == "Real content."

    def test_invalid_blocks_skipped_gracefully(self, tmp_path):
        """Malformed SRT blocks (missing timestamp, bad index) are skipped."""
        srt = _write_srt(tmp_path / "test.srt", """
            not_a_number
            00:00:01,000 --> 00:00:02,000
            Invalid block

            2
            bad timestamp line
            Also invalid

            3
            00:00:03,000 --> 00:00:04,000
            [Speaker 1] Valid block.
        """)

        result = parse_srt(srt)
        assert len(result) == 1
        assert result[0]["text"] == "Valid block."

    def test_empty_file_returns_empty_list(self, tmp_path):
        srt = tmp_path / "empty.srt"
        srt.write_text("", encoding="utf-8")
        result = parse_srt(srt)
        assert result == []

    def test_three_speakers_distinct(self, tmp_path):
        """Three distinct speakers are parsed correctly."""
        srt = _write_srt(tmp_path / "test.srt", """
            1
            00:00:01,000 --> 00:00:02,000
            [Speaker 1] Hello.

            2
            00:00:02,000 --> 00:00:03,000
            [Speaker 2] Hi there.

            3
            00:00:03,000 --> 00:00:04,000
            [Speaker 3] Good morning.
        """)

        result = parse_srt(srt)
        speakers = {s["speaker"] for s in result}
        assert speakers == {"Speaker 1", "Speaker 2", "Speaker 3"}

    def test_dot_format_timestamps(self, tmp_path):
        """Some SRTs use dots instead of commas in timestamps."""
        srt = _write_srt(tmp_path / "test.srt", """
            1
            00:00:01.000 --> 00:00:03.500
            [Speaker 1] Works with dots too.
        """)

        result = parse_srt(srt)
        assert len(result) == 1
        assert result[0]["start"] == pytest.approx(1.0, abs=0.01)

    def test_real_pipeline_output_format(self, tmp_path):
        """Tests with the exact format produced by the pipeline."""
        # NeMo diarize → translate → this is what Step 3 receives
        srt = _write_srt(tmp_path / "video.nemo.de.diarize_fr.srt", """
            1
            00:00:00,720 --> 00:00:02,460
            [Speaker 1] Bonjour et bienvenue à cette vidéo.

            2
            00:00:02,640 --> 00:00:05,880
            [Speaker 1] Aujourd'hui, nous allons parler de réseaux neuronaux.

            3
            00:00:06,000 --> 00:00:07,500
            [Speaker 2] Oui, c'est un sujet passionnant.

            4
            00:00:08,000 --> 00:00:10,200
            [Speaker 1] Commençons par les bases.
        """)

        result = parse_srt(srt)
        assert len(result) == 4
        # Verify speaker assignment
        assert result[0]["speaker"] == "Speaker 1"
        assert result[2]["speaker"] == "Speaker 2"
        # Verify text is clean (no tags)
        for seg in result:
            assert not seg["text"].startswith("[")
        # Verify timing is sequential
        for i in range(1, len(result)):
            assert result[i]["start"] >= result[i - 1]["start"]


# ═════════════════════════════════════════════════════════════════════════════
# 4. build_voice_map() — voice assignment
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildVoiceMap:

    def test_single_speaker(self):
        segments = [
            {"speaker": "Speaker 1", "text": "Hello", "start": 0.0, "end": 1.0},
        ]
        voice_map = build_voice_map(segments)
        assert "Speaker 1" in voice_map
        assert voice_map["Speaker 1"] in QWEN_FEMALE_VOICES  # first speaker = female

    def test_two_speakers_alternate_gender(self):
        segments = [
            {"speaker": "Speaker 1", "text": "Hello", "start": 0.0, "end": 1.0},
            {"speaker": "Speaker 2", "text": "Hi", "start": 1.0, "end": 2.0},
        ]
        voice_map = build_voice_map(segments)
        assert voice_map["Speaker 1"] in QWEN_FEMALE_VOICES
        assert voice_map["Speaker 2"] in QWEN_MALE_VOICES

    def test_three_speakers(self):
        segments = [
            {"speaker": "Speaker 1", "text": "A", "start": 0.0, "end": 1.0},
            {"speaker": "Speaker 2", "text": "B", "start": 1.0, "end": 2.0},
            {"speaker": "Speaker 3", "text": "C", "start": 2.0, "end": 3.0},
        ]
        voice_map = build_voice_map(segments)
        # Pattern: female, male, female
        assert voice_map["Speaker 1"] in QWEN_FEMALE_VOICES
        assert voice_map["Speaker 2"] in QWEN_MALE_VOICES
        assert voice_map["Speaker 3"] in QWEN_FEMALE_VOICES

    def test_repeated_speaker_gets_same_voice(self):
        """A speaker appearing multiple times always gets the same voice."""
        segments = [
            {"speaker": "Speaker 1", "text": "A", "start": 0.0, "end": 1.0},
            {"speaker": "Speaker 2", "text": "B", "start": 1.0, "end": 2.0},
            {"speaker": "Speaker 1", "text": "C", "start": 2.0, "end": 3.0},
            {"speaker": "Speaker 2", "text": "D", "start": 3.0, "end": 4.0},
        ]
        voice_map = build_voice_map(segments)
        assert len(voice_map) == 2  # Only 2 unique speakers

    def test_voice_pool_wraps_around(self):
        """With many speakers, voices cycle through the pools."""
        speakers = [f"Speaker {i}" for i in range(1, 11)]
        segments = [
            {"speaker": spk, "text": f"Line {i}", "start": float(i), "end": float(i + 1)}
            for i, spk in enumerate(speakers)
        ]
        voice_map = build_voice_map(segments)
        assert len(voice_map) == 10
        # Every speaker must have a voice assigned
        for spk in speakers:
            assert spk in voice_map
            assert voice_map[spk] in QWEN_FEMALE_VOICES + QWEN_MALE_VOICES

    def test_preserves_speaker_order(self):
        """Voice assignment follows order of first appearance, not alphabetical."""
        segments = [
            {"speaker": "Speaker 3", "text": "First", "start": 0.0, "end": 1.0},
            {"speaker": "Speaker 1", "text": "Second", "start": 1.0, "end": 2.0},
        ]
        voice_map = build_voice_map(segments)
        # Speaker 3 appears first → female voice
        assert voice_map["Speaker 3"] in QWEN_FEMALE_VOICES
        # Speaker 1 appears second → male voice
        assert voice_map["Speaker 1"] in QWEN_MALE_VOICES


# ═════════════════════════════════════════════════════════════════════════════
# 5. LANG_CODE_TO_QWEN completeness
# ═════════════════════════════════════════════════════════════════════════════

class TestLangCodeToQwen:

    @pytest.mark.parametrize("code", ["fr", "en", "de", "es", "it", "ja", "ko", "pt", "ru", "zh"])
    def test_all_pipeline_languages_mapped(self, code):
        """All languages supported by the pipeline must be in LANG_CODE_TO_QWEN."""
        assert code in LANG_CODE_TO_QWEN, f"Missing language code '{code}' in LANG_CODE_TO_QWEN"


# ═════════════════════════════════════════════════════════════════════════════
# 6. merge_segments()
# ═════════════════════════════════════════════════════════════════════════════

def _seg(i, spk, text, start, end):
    return {"index": i, "speaker": spk, "text": text, "start": start, "end": end}


class TestMergeSegments:

    def test_empty_returns_empty(self):
        assert merge_segments([]) == []

    def test_single_segment_unchanged(self):
        segs = [_seg(1, "Speaker 1", "Hello", 0.0, 1.5)]
        result = merge_segments(segs)
        assert result[0]["text"] == "Hello"
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 1.5
        assert result[0]["subsegments"] == [{"start": 0.0, "end": 1.5}]

    def test_subsegments_stored_on_merge(self):
        segs = [
            _seg(1, "Speaker 1", "Hello", 0.0, 1.0),
            _seg(2, "Speaker 1", "world", 1.5, 2.5),
        ]
        result = merge_segments(segs, gap_sec=1.0)
        assert len(result) == 1
        assert result[0]["subsegments"] == [
            {"start": 0.0, "end": 1.0},
            {"start": 1.5, "end": 2.5},
        ]

    def test_subsegments_single_when_not_merged(self):
        segs = [
            _seg(1, "Speaker 1", "A", 0.0, 1.0),
            _seg(2, "Speaker 2", "B", 1.2, 2.0),
        ]
        result = merge_segments(segs, gap_sec=1.0)
        assert result[0]["subsegments"] == [{"start": 0.0, "end": 1.0}]
        assert result[1]["subsegments"] == [{"start": 1.2, "end": 2.0}]

    def test_subsegments_three_in_chain(self):
        segs = [
            _seg(1, "Speaker 1", "One",   0.0, 1.0),
            _seg(2, "Speaker 1", "two",   1.3, 2.0),
            _seg(3, "Speaker 1", "three", 2.4, 3.5),
        ]
        result = merge_segments(segs, gap_sec=1.0)
        assert len(result) == 1
        assert result[0]["subsegments"] == [
            {"start": 0.0, "end": 1.0},
            {"start": 1.3, "end": 2.0},
            {"start": 2.4, "end": 3.5},
        ]

    def test_same_speaker_small_gap_merged(self):
        segs = [
            _seg(1, "Speaker 1", "Hello", 0.0, 1.0),
            _seg(2, "Speaker 1", "world", 1.5, 2.5),  # gap = 0.5 s
        ]
        result = merge_segments(segs, gap_sec=1.0)
        assert len(result) == 1
        assert result[0]["text"] == "Hello world"
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 2.5

    def test_same_speaker_large_gap_not_merged(self):
        segs = [
            _seg(1, "Speaker 1", "Hello", 0.0, 1.0),
            _seg(2, "Speaker 1", "world", 3.0, 4.0),  # gap = 2.0 s > threshold
        ]
        result = merge_segments(segs, gap_sec=1.0)
        assert len(result) == 2

    def test_different_speakers_not_merged(self):
        segs = [
            _seg(1, "Speaker 1", "Hello", 0.0, 1.0),
            _seg(2, "Speaker 2", "world", 1.2, 2.5),  # gap = 0.2 s but different speaker
        ]
        result = merge_segments(segs, gap_sec=1.0)
        assert len(result) == 2

    def test_chain_of_three_merged_into_one(self):
        segs = [
            _seg(1, "Speaker 1", "One",   0.0, 1.0),
            _seg(2, "Speaker 1", "two",   1.3, 2.0),
            _seg(3, "Speaker 1", "three", 2.4, 3.5),
        ]
        result = merge_segments(segs, gap_sec=1.0)
        assert len(result) == 1
        assert result[0]["text"] == "One two three"
        assert result[0]["end"] == 3.5

    def test_interleaved_speakers_not_merged_across(self):
        segs = [
            _seg(1, "Speaker 1", "A", 0.0, 1.0),
            _seg(2, "Speaker 2", "B", 1.2, 2.0),
            _seg(3, "Speaker 1", "C", 2.2, 3.0),
        ]
        result = merge_segments(segs, gap_sec=1.0)
        assert len(result) == 3  # each separated by a different speaker

    def test_gap_exactly_at_threshold_is_merged(self):
        segs = [
            _seg(1, "Speaker 1", "A", 0.0, 1.0),
            _seg(2, "Speaker 1", "B", 2.0, 3.0),  # gap = exactly 1.0 s
        ]
        result = merge_segments(segs, gap_sec=1.0)
        assert len(result) == 1

    def test_gap_just_over_threshold_not_merged(self):
        segs = [
            _seg(1, "Speaker 1", "A", 0.0, 1.0),
            _seg(2, "Speaker 1", "B", 2.01, 3.0),  # gap = 1.01 s
        ]
        result = merge_segments(segs, gap_sec=1.0)
        assert len(result) == 2

    def test_gap_zero_disables_merging(self):
        segs = [
            _seg(1, "Speaker 1", "A", 0.0, 1.0),
            _seg(2, "Speaker 1", "B", 1.2, 2.0),
        ]
        result = merge_segments(segs, gap_sec=0)
        assert len(result) == 2

    def test_merged_text_no_double_spaces(self):
        segs = [
            _seg(1, "Speaker 1", "Hello ", 0.0, 1.0),
            _seg(2, "Speaker 1", " world", 1.3, 2.0),
        ]
        result = merge_segments(segs, gap_sec=1.0)
        assert "  " not in result[0]["text"]
        assert result[0]["text"] == "Hello world"

    def test_original_segments_not_mutated(self):
        segs = [
            _seg(1, "Speaker 1", "A", 0.0, 1.0),
            _seg(2, "Speaker 1", "B", 1.3, 2.0),
        ]
        import copy
        original = copy.deepcopy(segs)
        merge_segments(segs, gap_sec=1.0)
        assert segs == original, "merge_segments must not mutate the input list"

    def test_first_index_preserved_in_merged_group(self):
        segs = [
            _seg(5, "Speaker 1", "A", 0.0, 1.0),
            _seg(6, "Speaker 1", "B", 1.3, 2.0),
        ]
        result = merge_segments(segs, gap_sec=1.0)
        assert result[0]["index"] == 5


# ── _fmt_ts tests ────────────────────────────────────────────────────────────

class TestFmtTs:
    def test_zero(self):
        assert _fmt_ts(0.0) == "00:00:00,000"

    def test_simple_seconds(self):
        assert _fmt_ts(1.5) == "00:00:01,500"

    def test_minutes_and_seconds(self):
        assert _fmt_ts(65.123) == "00:01:05,123"

    def test_hours(self):
        assert _fmt_ts(3661.0) == "01:01:01,000"

    def test_sub_millisecond_rounded(self):
        # 3 decimal places in SRT format
        result = _fmt_ts(1.9999)
        assert result == "00:00:02,000"


# ── write_dub_srt tests ─────────────────────────────────────────────────────

class TestSplitSubtitleBlocks:
    def test_short_text_single_block(self):
        blocks = _split_subtitle_blocks("Hello world", 0.0, 2.0)
        assert len(blocks) == 1
        assert blocks[0][0] == "Hello world"
        assert blocks[0][1] == 0.0
        assert blocks[0][2] == 2.0

    def test_long_text_splits_into_multiple_blocks(self):
        text = ("Je m'appelle Jumal Khalili et je suis professeur "
                "émérite de physique à l'Université de Surrey")
        blocks = _split_subtitle_blocks(text, 0.0, 10.0)
        assert len(blocks) >= 2
        for block_text, _, _ in blocks:
            assert len(block_text) <= 42

    def test_timestamps_are_proportional(self):
        text = "Short block and a much longer block with more words here"
        blocks = _split_subtitle_blocks(text, 0.0, 10.0)
        assert len(blocks) >= 2
        # First block starts at 0
        assert blocks[0][1] == 0.0
        # Last block ends at 10
        assert abs(blocks[-1][2] - 10.0) < 0.01
        # Blocks are contiguous
        for i in range(len(blocks) - 1):
            assert abs(blocks[i][2] - blocks[i + 1][1]) < 0.001

    def test_no_block_exceeds_max_ch(self):
        text = ("This is a very long subtitle that contains many words "
                "and should be split into several smaller blocks for display")
        blocks = _split_subtitle_blocks(text, 0.0, 5.0, max_ch=30)
        for block_text, _, _ in blocks:
            assert len(block_text) <= 30

    def test_single_word_stays_single(self):
        blocks = _split_subtitle_blocks("Bonjour", 1.0, 2.0)
        assert len(blocks) == 1
        assert blocks[0] == ("Bonjour", 1.0, 2.0)

    def test_empty_text(self):
        blocks = _split_subtitle_blocks("", 0.0, 1.0)
        assert len(blocks) == 1


class TestWriteDubSrt:
    def test_basic_output(self, tmp_path):
        segments = [
            _seg(1, "Speaker 1", "Hello world", 0.0, 2.0),
            _seg(2, "Speaker 2", "Bonjour", 2.5, 4.0),
        ]
        actual_positions = [
            (0.0, 2.3, 0.0, 2.0),
            (2.8, 4.5, 2.5, 4.0),
        ]
        out = tmp_path / "test_dub.srt"
        write_dub_srt(out, actual_positions, segments)

        content = out.read_text(encoding="utf-8")
        assert "00:00:00,000 --> 00:00:02,300" in content
        assert "00:00:02,800 --> 00:00:04,500" in content
        assert "Hello world" in content
        assert "Bonjour" in content

    def test_speaker_tags_stripped(self, tmp_path):
        """Dub SRT is for viewing — no [Speaker N] tags."""
        segments = [
            _seg(1, "Speaker 1", "Hello world", 0.0, 2.0),
            _seg(2, "Speaker 2", "Bonjour", 2.5, 4.0),
        ]
        actual_positions = [
            (0.0, 2.3, 0.0, 2.0),
            (2.8, 4.5, 2.5, 4.0),
        ]
        out = tmp_path / "test_dub.srt"
        write_dub_srt(out, actual_positions, segments)

        content = out.read_text(encoding="utf-8")
        assert "[Speaker 1]" not in content
        assert "[Speaker 2]" not in content

    def test_long_text_split_into_short_blocks(self, tmp_path):
        """Long merged text should produce multiple short SRT blocks."""
        long_text = ("This is a fairly long subtitle line that should "
                     "definitely be split into multiple short blocks")
        segments = [_seg(1, "Speaker 1", long_text, 0.0, 5.0)]
        actual_positions = [(0.0, 5.0, 0.0, 5.0)]
        out = tmp_path / "split.srt"
        write_dub_srt(out, actual_positions, segments)

        content = out.read_text(encoding="utf-8")
        # Should produce multiple SRT entries
        srt_blocks = [b for b in content.strip().split("\n\n") if b.strip()]
        assert len(srt_blocks) >= 2
        # Each text line should be ≤ 42 chars
        for block in srt_blocks:
            text_lines = [l for l in block.split("\n")
                          if l and not l[0].isdigit() and "-->" not in l]
            for line in text_lines:
                assert len(line) <= 42

    def test_sequential_indices(self, tmp_path):
        segments = [
            _seg(5, "Speaker 1", "A", 0.0, 1.0),
            _seg(10, "Speaker 2", "B", 1.5, 3.0),
        ]
        actual_positions = [
            (0.0, 1.1, 0.0, 1.0),
            (1.6, 3.2, 1.5, 3.0),
        ]
        out = tmp_path / "test.srt"
        write_dub_srt(out, actual_positions, segments)

        content = out.read_text(encoding="utf-8")
        lines = [l for l in content.strip().split("\n") if l.strip()]
        assert lines[0] == "1"
        assert lines[3] == "2"

    def test_empty_segments(self, tmp_path):
        out = tmp_path / "empty.srt"
        write_dub_srt(out, [], [])
        content = out.read_text(encoding="utf-8")
        assert content.strip() == ""

    def test_skips_unmatched_positions(self, tmp_path):
        segments = [_seg(1, "Speaker 1", "Hello", 0.0, 2.0)]
        actual_positions = [(0.0, 2.0, 99.0, 100.0)]
        out = tmp_path / "unmatched.srt"
        write_dub_srt(out, actual_positions, segments)
        content = out.read_text(encoding="utf-8")
        assert "Hello" not in content

    def test_merged_segments_match(self, tmp_path):
        """After merge_segments, write_dub_srt should still match correctly."""
        segments = [
            _seg(1, "Speaker 1", "Hello", 0.0, 1.0),
            _seg(2, "Speaker 1", "world", 1.2, 2.0),
            _seg(3, "Speaker 2", "Bonjour", 3.0, 4.0),
        ]
        merged = merge_segments(segments, gap_sec=1.0)
        actual_positions = [
            (0.0, 2.1, 0.0, 2.0),
            (3.0, 4.2, 3.0, 4.0),
        ]
        out = tmp_path / "merged.srt"
        write_dub_srt(out, actual_positions, merged)
        content = out.read_text(encoding="utf-8")
        assert "Hello world" in content
        assert "Bonjour" in content
        assert "00:00:02,100" in content

    def test_roundtrip_timestamps_differ_from_original(self, tmp_path):
        """The whole point: dub SRT timestamps != original SRT timestamps."""
        segments = [_seg(1, "Speaker 1", "Test", 10.0, 12.0)]
        actual_positions = [(9.5, 11.8, 10.0, 12.0)]
        out = tmp_path / "drift.srt"
        write_dub_srt(out, actual_positions, segments)
        content = out.read_text(encoding="utf-8")
        assert "00:00:09,500 --> 00:00:11,800" in content
        assert "00:00:10,000" not in content
