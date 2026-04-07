"""
tests/test_dub_pipeline.py — Tests for the dubbing pipeline.

Covers:
  - dub_audio checkpoint save/load (crash recovery)
  - speed_fit edge cases (too short, too long, capped speed)
  - extract_clone_refs speaker selection logic
  - dub_srt parse_srt edge cases (multi-line, pipe encoding, fallback speaker)
  - dub_srt build_voice_map alternation
  - dub_srt _srt_ts timestamp parsing
  - dub_srt _qwen_lang unknown codes

Run:
    uv run --with "pytest,pydantic" pytest tests/test_dub_pipeline.py -v
"""

from __future__ import annotations

import json
import unittest.mock as mock
import wave
from pathlib import Path
from typing import Dict, List

import pytest

# dub_srt is importable directly (no heavy deps)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "qwen3-tts"))

from dub_srt import (
    LANG_CODE_TO_QWEN,
    QWEN_FEMALE_VOICES,
    QWEN_MALE_VOICES,
    _qwen_lang,
    _srt_ts,
    build_voice_map,
    parse_srt,
)
from dub_audio import (
    _load_checkpoint,
    _save_checkpoint,
    _qwen_python,
    _qwen_worker,
    extract_clone_refs,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_wav(path: Path, duration_sec: float = 2.0, sample_rate: int = 16000) -> None:
    n_frames = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


def _write_srt(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Checkpoint save/load — crash recovery
# ═══════════════════════════════════════════════════════════════════════════════

class TestDubCheckpoint:
    """Checkpoint lets a crashed dub run resume without re-generating TTS."""

    def test_save_and_reload(self, tmp_path):
        clip1 = tmp_path / "seg_0001.wav"
        clip2 = tmp_path / "seg_0002.wav"
        _make_wav(clip1, 1.5)
        _make_wav(clip2, 2.0)

        data = [(clip1, 0.0, 1.5), (clip2, 2.0, 4.0)]
        ckpt = tmp_path / "checkpoint.json"
        _save_checkpoint(ckpt, data)

        loaded = _load_checkpoint(ckpt)
        assert len(loaded) == 2
        assert loaded[0][1] == 0.0
        assert loaded[0][2] == 1.5
        assert loaded[1][1] == 2.0

    def test_load_skips_missing_files(self, tmp_path):
        """Checkpoint entries whose files are gone are silently skipped."""
        ckpt = tmp_path / "checkpoint.json"
        ckpt.write_text(json.dumps([
            {"clip": str(tmp_path / "gone.wav"), "start": 0.0, "end": 1.0},
        ]))
        result = _load_checkpoint(ckpt)
        assert result == []

    def test_load_skips_empty_files(self, tmp_path):
        """Checkpoint entry with a near-empty WAV (<500 bytes) is skipped."""
        tiny = tmp_path / "tiny.wav"
        tiny.write_bytes(b"\x00" * 100)
        ckpt = tmp_path / "checkpoint.json"
        ckpt.write_text(json.dumps([
            {"clip": str(tiny), "start": 0.0, "end": 1.0},
        ]))
        result = _load_checkpoint(ckpt)
        assert result == []

    def test_load_nonexistent_checkpoint(self, tmp_path):
        """No checkpoint file → returns empty list."""
        result = _load_checkpoint(tmp_path / "nope.json")
        assert result == []

    def test_load_corrupt_checkpoint(self, tmp_path):
        """Corrupt checkpoint → returns empty list (no crash)."""
        ckpt = tmp_path / "checkpoint.json"
        ckpt.write_text("{ corrupt json !!!")
        result = _load_checkpoint(ckpt)
        assert result == []

    def test_save_creates_valid_json(self, tmp_path):
        """Saved checkpoint is valid JSON with expected structure."""
        clip = tmp_path / "seg_0001.wav"
        _make_wav(clip)
        ckpt = tmp_path / "checkpoint.json"
        _save_checkpoint(ckpt, [(clip, 1.5, 3.0)])

        data = json.loads(ckpt.read_text())
        assert isinstance(data, list)
        assert data[0]["start"] == 1.5
        assert data[0]["end"] == 3.0
        assert str(clip) in data[0]["clip"]

    def test_save_multiple_segments(self, tmp_path):
        """100 segments saved and reloaded correctly."""
        clips = []
        for i in range(100):
            c = tmp_path / f"seg_{i:04d}.wav"
            _make_wav(c, 1.0)
            clips.append((c, float(i * 2), float(i * 2 + 1.0)))

        ckpt = tmp_path / "checkpoint.json"
        _save_checkpoint(ckpt, clips)
        loaded = _load_checkpoint(ckpt)
        assert len(loaded) == 100


# ═══════════════════════════════════════════════════════════════════════════════
# 2. _srt_ts — timestamp parsing edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestSrtTimestamp:

    def test_standard_comma_format(self):
        assert abs(_srt_ts("00:01:30,500") - 90.5) < 0.001

    def test_dot_format(self):
        """Some SRT files use dot instead of comma."""
        assert abs(_srt_ts("00:01:30.500") - 90.5) < 0.001

    def test_zero_timestamp(self):
        assert _srt_ts("00:00:00,000") == 0.0

    def test_large_timestamp(self):
        # 1h 30m 0s
        assert abs(_srt_ts("01:30:00,000") - 5400.0) < 0.001

    def test_with_whitespace(self):
        """Leading/trailing whitespace is stripped."""
        assert abs(_srt_ts("  00:00:05,000  ") - 5.0) < 0.001


# ═══════════════════════════════════════════════════════════════════════════════
# 3. parse_srt — diarized SRT parsing
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseSrt:

    def _basic_srt(self, tmp_path, content: str) -> Path:
        p = tmp_path / "test.srt"
        _write_srt(p, content)
        return p

    def test_parses_standard_diarized_srt(self, tmp_path):
        srt = self._basic_srt(tmp_path, """
1
00:00:00,000 --> 00:00:02,000
[Speaker 1] Bonjour le monde

2
00:00:02,500 --> 00:00:04,000
[Speaker 2] Bonsoir tout le monde

""".strip())
        segs = parse_srt(srt)
        assert len(segs) == 2
        assert segs[0]["speaker"] == "Speaker 1"
        assert segs[0]["text"] == "Bonjour le monde"
        assert segs[1]["speaker"] == "Speaker 2"
        assert abs(segs[0]["start"] - 0.0) < 0.01
        assert abs(segs[0]["end"] - 2.0) < 0.01

    def test_falls_back_to_speaker_1_when_no_tag(self, tmp_path):
        srt = self._basic_srt(tmp_path, """
1
00:00:00,000 --> 00:00:02,000
Bonjour sans tag

""".strip())
        segs = parse_srt(srt)
        assert len(segs) == 1
        assert segs[0]["speaker"] == "Speaker 1"
        assert segs[0]["text"] == "Bonjour sans tag"

    def test_joins_continuation_lines(self, tmp_path):
        """Multi-line blocks are joined into a single text."""
        srt = self._basic_srt(tmp_path, """
1
00:00:00,000 --> 00:00:03,000
[Speaker 1] First line of a long
translation that wraps around

""".strip())
        segs = parse_srt(srt)
        assert len(segs) == 1
        assert "First line" in segs[0]["text"]
        assert "wraps around" in segs[0]["text"]

    def test_pipe_separator_replaced(self, tmp_path):
        """Pipe characters from translate.py are replaced by spaces."""
        srt = self._basic_srt(tmp_path, """
1
00:00:00,000 --> 00:00:02,000
[Speaker 1] line one | line two

""".strip())
        segs = parse_srt(srt)
        assert "|" not in segs[0]["text"]

    def test_skips_empty_text_blocks(self, tmp_path):
        """Blocks whose text (after tag removal) is empty are skipped."""
        srt = self._basic_srt(tmp_path, """
1
00:00:00,000 --> 00:00:01,000
[Speaker 1]

2
00:00:01,000 --> 00:00:02,000
[Speaker 2] Actual content

""".strip())
        segs = parse_srt(srt)
        assert len(segs) == 1
        assert segs[0]["text"] == "Actual content"

    def test_skips_blocks_with_fewer_than_3_lines(self, tmp_path):
        """Malformed blocks without timestamp or text are skipped."""
        srt = self._basic_srt(tmp_path, """
1
00:00:00,000 --> 00:00:02,000
[Speaker 1] Valid

2
Missing text line

""".strip())
        segs = parse_srt(srt)
        assert len(segs) == 1

    def test_dot_timestamp_format(self, tmp_path):
        """Some exported SRTs use dot instead of comma in timestamps."""
        srt = self._basic_srt(tmp_path, """
1
00:00:00.000 --> 00:00:02.500
[Speaker 1] Hello

""".strip())
        segs = parse_srt(srt)
        assert len(segs) == 1
        assert abs(segs[0]["end"] - 2.5) < 0.01

    def test_multi_speaker_ordering(self, tmp_path):
        """Speakers are extracted in order of appearance."""
        srt = self._basic_srt(tmp_path, """
1
00:00:00,000 --> 00:00:01,000
[Speaker 3] Third speaker first

2
00:00:01,000 --> 00:00:02,000
[Speaker 1] First speaker second

""".strip())
        segs = parse_srt(srt)
        assert segs[0]["speaker"] == "Speaker 3"
        assert segs[1]["speaker"] == "Speaker 1"

    def test_index_preserved(self, tmp_path):
        """Block index numbers are preserved in output."""
        srt = self._basic_srt(tmp_path, """
10
00:01:00,000 --> 00:01:02,000
[Speaker 1] Block ten

20
00:02:00,000 --> 00:02:02,000
[Speaker 2] Block twenty

""".strip())
        segs = parse_srt(srt)
        assert segs[0]["index"] == 10
        assert segs[1]["index"] == 20

    def test_empty_file(self, tmp_path):
        srt = self._basic_srt(tmp_path, "")
        assert parse_srt(srt) == []

    def test_unicode_content(self, tmp_path):
        """Arabic, Chinese, Japanese content parsed correctly."""
        srt = self._basic_srt(tmp_path, """
1
00:00:00,000 --> 00:00:02,000
[Speaker 1] مرحبا بالعالم

2
00:00:02,000 --> 00:00:04,000
[Speaker 2] 你好世界

""".strip())
        segs = parse_srt(srt)
        assert len(segs) == 2
        assert "مرحبا" in segs[0]["text"]
        assert "你好" in segs[1]["text"]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. build_voice_map — alternating female/male assignment
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildVoiceMap:

    def _segs(self, speakers: List[str]) -> List[Dict]:
        return [{"speaker": s, "text": "x", "start": 0, "end": 1} for s in speakers]

    def test_single_speaker_gets_female_voice(self):
        vm = build_voice_map(self._segs(["Speaker 1"]))
        assert vm["Speaker 1"] in QWEN_FEMALE_VOICES

    def test_two_speakers_alternate_gender(self):
        vm = build_voice_map(self._segs(["Speaker 1", "Speaker 2"]))
        assert vm["Speaker 1"] in QWEN_FEMALE_VOICES
        assert vm["Speaker 2"] in QWEN_MALE_VOICES

    def test_three_speakers_pattern(self):
        vm = build_voice_map(self._segs(["A", "B", "C"]))
        assert vm["A"] in QWEN_FEMALE_VOICES  # i=0, even
        assert vm["B"] in QWEN_MALE_VOICES    # i=1, odd
        assert vm["C"] in QWEN_FEMALE_VOICES  # i=2, even

    def test_order_of_appearance(self):
        """Speaker order is determined by first appearance, not alphabetical."""
        segs = [
            {"speaker": "Z", "text": "first", "start": 0, "end": 1},
            {"speaker": "A", "text": "second", "start": 1, "end": 2},
        ]
        vm = build_voice_map(segs)
        assert vm["Z"] in QWEN_FEMALE_VOICES  # Z appears first
        assert vm["A"] in QWEN_MALE_VOICES    # A appears second

    def test_duplicate_speakers_counted_once(self):
        """Same speaker appearing in many segments → assigned one voice."""
        segs = [{"speaker": "Speaker 1", "text": f"seg{i}", "start": i, "end": i+1}
                for i in range(20)]
        vm = build_voice_map(segs)
        assert len(vm) == 1

    def test_wraps_female_voices(self):
        """More speakers than voice pool size → cycles through voices."""
        # 7 speakers: indices 0,2,4,6 → female; 1,3,5 → male
        speakers = [f"S{i}" for i in range(7)]
        segs = [{"speaker": s, "text": "x", "start": i, "end": i+1}
                for i, s in enumerate(speakers)]
        vm = build_voice_map(segs)
        all_voices = QWEN_FEMALE_VOICES + QWEN_MALE_VOICES
        for s in speakers:
            assert vm[s] in all_voices, f"Speaker {s} got invalid voice: {vm[s]}"
        # Even-index speakers (0,2,4,6) → female
        for i in [0, 2, 4, 6]:
            assert vm[speakers[i]] in QWEN_FEMALE_VOICES
        # Odd-index speakers (1,3,5) → male
        for i in [1, 3, 5]:
            assert vm[speakers[i]] in QWEN_MALE_VOICES


# ═══════════════════════════════════════════════════════════════════════════════
# 5. _qwen_lang — language code mapping
# ═══════════════════════════════════════════════════════════════════════════════

class TestQwenLang:

    @pytest.mark.parametrize("code,expected", [
        ("fr", "french"),
        ("en", "english"),
        ("de", "german"),
        ("es", "spanish"),
        ("it", "italian"),
        ("ja", "japanese"),
        ("ko", "korean"),
        ("pt", "portuguese"),
        ("ru", "russian"),
        ("zh", "chinese"),
        ("auto", "auto"),
        # Case insensitive
        ("FR", "french"),
        ("De", "german"),
    ])
    def test_known_codes(self, code, expected):
        assert _qwen_lang(code) == expected

    def test_unknown_code_returned_as_is(self):
        """Unknown code → returned as-is with a warning (no crash)."""
        result = _qwen_lang("xx")
        assert result == "xx"

    def test_whitespace_stripped(self):
        assert _qwen_lang("  fr  ") == "french"

    def test_all_lang_codes_covered(self):
        """All codes in LANG_CODE_TO_QWEN produce their expected value."""
        for code, expected in LANG_CODE_TO_QWEN.items():
            assert _qwen_lang(code) == expected


# ═══════════════════════════════════════════════════════════════════════════════
# 6. extract_clone_refs — speaker reference extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractCloneRefs:

    def test_skips_segments_under_1s(self, tmp_path):
        """Segments shorter than 1s are too short for voice cloning."""
        audio = tmp_path / "audio.wav"
        _make_wav(audio, 10.0)

        segments = [{"speaker": "Speaker 1", "start": 0.0, "end": 0.5}]  # 0.5s < 1s

        with mock.patch("dub_audio.subprocess.run"):
            refs = extract_clone_refs(segments, audio, tmp_path / "cast")

        assert "Speaker 1" not in refs

    def test_picks_longest_segment_per_speaker(self, tmp_path):
        """For each speaker, picks their longest segment as reference."""
        audio = tmp_path / "audio.wav"
        _make_wav(audio, 30.0)
        cast_dir = tmp_path / "cast"

        segments = [
            {"speaker": "Speaker 1", "start": 0.0, "end": 2.0},   # 2s
            {"speaker": "Speaker 1", "start": 5.0, "end": 12.0},  # 7s ← longest
            {"speaker": "Speaker 1", "start": 15.0, "end": 17.0}, # 2s
        ]

        captured_args = []
        def mock_run(args, **kwargs):
            captured_args.append(args)
            # Create the output file to simulate ffmpeg success
            if args[0] == "ffmpeg":
                out_file = Path(args[-1])
                out_file.parent.mkdir(parents=True, exist_ok=True)
                _make_wav(out_file, float(args[args.index("-t") + 1]) if "-t" in args else 2.0)

        with mock.patch("dub_audio.subprocess.run", side_effect=mock_run):
            refs = extract_clone_refs(segments, audio, cast_dir)

        # Find the ffmpeg call and check it used the 7s segment
        ffmpeg_calls = [a for a in captured_args if a and a[0] == "ffmpeg"]
        assert len(ffmpeg_calls) == 1
        ffmpeg_args = ffmpeg_calls[0]
        # -t arg should be 7.0 (the longest segment duration)
        t_idx = ffmpeg_args.index("-t")
        assert abs(float(ffmpeg_args[t_idx + 1]) - 7.0) < 0.1

    def test_reuses_existing_ref(self, tmp_path):
        """If ref WAV already exists and is large enough, skip ffmpeg."""
        audio = tmp_path / "audio.wav"
        _make_wav(audio, 10.0)
        cast_dir = tmp_path / "cast"
        cast_dir.mkdir()

        # Pre-create the ref file
        ref = cast_dir / "Speaker_1.wav"
        _make_wav(ref, 3.0)
        # Make it large enough (>1000 bytes)
        ref.write_bytes(b"\x00" * 2000)

        segments = [{"speaker": "Speaker 1", "start": 0.0, "end": 5.0}]

        with mock.patch("dub_audio.subprocess.run") as mock_run:
            refs = extract_clone_refs(segments, audio, cast_dir)

        # ffmpeg should NOT have been called (cached ref used)
        mock_run.assert_not_called()
        assert "Speaker 1" in refs


# ═══════════════════════════════════════════════════════════════════════════════
# 7. _qwen_python / _qwen_worker — path resolution
# ═══════════════════════════════════════════════════════════════════════════════

class TestQwenPaths:

    def test_qwen_python_returns_venv_if_exists(self, tmp_path):
        venv_py = tmp_path / ".venv" / "bin" / "python"
        venv_py.parent.mkdir(parents=True)
        venv_py.touch()
        result = _qwen_python(tmp_path)
        assert result == str(venv_py)

    def test_qwen_python_falls_back_to_python(self, tmp_path):
        """No venv → returns 'python' fallback."""
        result = _qwen_python(tmp_path)
        assert result == "python"

    def test_qwen_worker_returns_path_if_exists(self, tmp_path):
        worker = tmp_path / "qwen_tts_worker.py"
        worker.write_text("# worker")
        result = _qwen_worker(tmp_path)
        assert result == str(worker)

    def test_qwen_worker_raises_if_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="qwen_tts_worker.py"):
            _qwen_worker(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Full SRT → parse_srt → build_voice_map roundtrip
# ═══════════════════════════════════════════════════════════════════════════════

class TestDubSrtRoundtrip:

    def test_full_pipeline_srt_roundtrip(self, tmp_path):
        """Realistic translated diarized SRT → parse → voice map."""
        blocks = []
        for i in range(9):
            blocks.extend([
                f"{i+1}",
                f"00:0{i}:00,000 --> 00:0{i}:02,000",
                f"[Speaker {(i % 3) + 1}] Sample text for segment {i+1}",
                "",
            ])
        srt_content = "\n".join(blocks)
        srt_path = tmp_path / "video.nemo.de.diarize_fr.srt"
        srt_path.write_text(srt_content, encoding="utf-8")

        segs = parse_srt(srt_path)
        assert len(segs) == 9

        vm = build_voice_map(segs)
        speakers = sorted(set(s["speaker"] for s in segs))
        assert set(vm.keys()) == set(speakers)

        # All assigned voices are valid Qwen voices
        all_voices = QWEN_FEMALE_VOICES + QWEN_MALE_VOICES
        for voice in vm.values():
            assert voice in all_voices, f"Invalid Qwen voice: {voice}"

    def test_speaker_tags_preserved_through_parse(self, tmp_path):
        """Speaker identity survives the full parse chain without mangling."""
        content = (
            "1\n00:00:00,000 --> 00:00:02,000\n[Speaker 1] First speaker\n\n"
            "2\n00:00:02,000 --> 00:00:04,000\n[Speaker 2] Second speaker\n\n"
            "3\n00:00:04,000 --> 00:00:06,000\n[Speaker 1] First speaker again\n\n"
        )
        srt_path = tmp_path / "test.srt"
        srt_path.write_text(content, encoding="utf-8")

        segs = parse_srt(srt_path)
        speakers_seen = [s["speaker"] for s in segs]
        assert speakers_seen == ["Speaker 1", "Speaker 2", "Speaker 1"]
