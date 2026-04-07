"""
tests/test_nemo_pipeline.py — Matrix tests for the NeMo ASR pipeline.

Tests cover five risk areas that can silently corrupt a 2-hour pipeline run:

  1. _validate_checkpoint  — stale checkpoint detection
                             (the trim-mismatch / partial-audio bug)
  2. _strip_asr_repetition — Canary hallucination removal
  3. _segs_to_srt          — subtitle deduplication (diarized + non-diarized)
  4. _split_coarse_segs    — timing proportionality and speaker passthrough
  5. SRT coverage ratio    — verify the SRT spans the expected audio duration

Test data is defined as Pydantic BaseModels for type-safe, self-documenting
case definitions.  Each group is parametrised so failures identify the
exact case that broke.

Run:
    uv run --with "pytest,pydantic" pytest tests/test_nemo_pipeline.py -v
"""

from __future__ import annotations

import json
import re
import tempfile
import wave
from pathlib import Path
from typing import Any, Optional

import pytest
from pydantic import BaseModel, model_validator

# ── Imports from nemo/ (sys.path set in conftest.py) ─────────────────────────
from nemo_audio import (
    _segs_to_srt, _split_coarse_segs, _strip_asr_repetition, _srt_last_timestamp,
    _strip_special_tokens,
)
from nemo_diarize import _validate_checkpoint
from nemo_model import _estimate_chunk_sec
from qwen3_asr import _is_qwen3_asr, _transcribe_qwen3_asr, QWEN3_LANG_MAP


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic models — structured, self-documenting test cases
# ═══════════════════════════════════════════════════════════════════════════════

class RepetitionCase(BaseModel):
    id: str
    description: str
    input: str
    should_strip: bool
    expected_output: Optional[str] = None   # required when should_strip=True

    @model_validator(mode="after")
    def _check_expected(self) -> "RepetitionCase":
        if self.should_strip and self.expected_output is None:
            raise ValueError("expected_output required when should_strip=True")
        return self


class SegToSrtCase(BaseModel):
    id: str
    description: str
    segments: list[dict[str, Any]]
    diarized: bool
    expected_count: int               # expected number of SRT blocks
    no_consecutive_dups: bool = True  # assert no consecutive identical lines


class CheckpointCase(BaseModel):
    id: str
    description: str
    checkpoint_data: dict[str, Any]   # written to JSON file
    trim_sec_at_load: int             # trim_sec passed to _validate_checkpoint
    audio_exists: bool                # whether to create a real WAV file
    actual_duration_sec: float        # duration of the WAV (if audio_exists)
    should_be_valid: bool


class SplitCoarseCase(BaseModel):
    id: str
    description: str
    segments: list[dict[str, Any]]
    max_w: int = 10
    max_ch: int = 80
    expected_line_count_min: int      # must produce at least this many lines
    total_duration_preserved: bool = True   # sum of output durations ≈ input


class SrtCoverageCase(BaseModel):
    id: str
    description: str
    srt_text: str
    audio_duration_sec: float
    min_coverage_ratio: float         # last SRT timestamp / audio_duration must exceed this


# ═══════════════════════════════════════════════════════════════════════════════
# 1. _validate_checkpoint — matrix
# ═══════════════════════════════════════════════════════════════════════════════

_CHECKPOINT_CASES: list[CheckpointCase] = [
    CheckpointCase(
        id="trim-mismatch-40-to-full",
        description="--trim 40 checkpoint loaded for full run: trim_sec=40 ≠ 0 → STALE",
        checkpoint_data={
            "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
            "segs": [],
            "audio_duration": 40.0,
            "asr_elapsed": 12.0,
            "rtf": 0.3,
            "trim_sec": 40,
        },
        trim_sec_at_load=0,
        audio_exists=False,
        actual_duration_sec=0.0,
        should_be_valid=False,
    ),
    CheckpointCase(
        id="trim-mismatch-full-to-trim40",
        description="Full checkpoint loaded for --trim 40 run: trim_sec=0 ≠ 40 → STALE",
        checkpoint_data={
            "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
            "segs": [],
            "audio_duration": 1800.0,
            "asr_elapsed": 600.0,
            "rtf": 0.33,
            "trim_sec": 0,
        },
        trim_sec_at_load=40,
        audio_exists=False,
        actual_duration_sec=0.0,
        should_be_valid=False,
    ),
    CheckpointCase(
        id="trim-match-40",
        description="--trim 40 checkpoint, --trim 40 run, audio matches → VALID",
        checkpoint_data={
            "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
            "segs": [],
            "audio_duration": 40.0,
            "asr_elapsed": 12.0,
            "rtf": 0.3,
            "trim_sec": 40,
        },
        trim_sec_at_load=40,
        audio_exists=True,
        actual_duration_sec=40.0,
        should_be_valid=True,
    ),
    CheckpointCase(
        id="full-run-match",
        description="Full run checkpoint, full run, duration matches → VALID",
        checkpoint_data={
            "words": [{"word": "guten", "start": 0.0, "end": 0.4}],
            "segs": [],
            "audio_duration": 1800.0,
            "asr_elapsed": 600.0,
            "rtf": 0.33,
            "trim_sec": 0,
        },
        trim_sec_at_load=0,
        audio_exists=True,
        actual_duration_sec=1800.0,
        should_be_valid=True,
    ),
    CheckpointCase(
        id="corrupt-json",
        description="Checkpoint file contains invalid JSON → STALE",
        checkpoint_data={"__corrupt__": True},   # marker: written as raw bytes
        trim_sec_at_load=0,
        audio_exists=False,
        actual_duration_sec=0.0,
        should_be_valid=False,
    ),
    CheckpointCase(
        id="empty-words-and-segs",
        description="Checkpoint has empty words [] and segs [] → STALE",
        checkpoint_data={
            "words": [],
            "segs": [],
            "audio_duration": 1800.0,
            "asr_elapsed": 600.0,
            "rtf": 0.33,
            "trim_sec": 0,
        },
        trim_sec_at_load=0,
        audio_exists=False,
        actual_duration_sec=0.0,
        should_be_valid=False,
    ),
    CheckpointCase(
        id="missing-words-key",
        description="Checkpoint JSON has no 'words' key → STALE",
        checkpoint_data={
            "segs": [{"text": "hi", "start": 0.0, "end": 1.0}],
            "audio_duration": 10.0,
            "asr_elapsed": 3.0,
            "rtf": 0.3,
        },
        trim_sec_at_load=0,
        audio_exists=False,
        actual_duration_sec=0.0,
        should_be_valid=False,
    ),
    CheckpointCase(
        id="duration-mismatch-50pct",
        description="Stored 40s, actual audio is 1800s (50x off) → STALE",
        checkpoint_data={
            "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
            "segs": [],
            "audio_duration": 40.0,
            "asr_elapsed": 12.0,
            "rtf": 0.3,
            # no trim_sec — old format checkpoint
        },
        trim_sec_at_load=0,
        audio_exists=True,
        actual_duration_sec=1800.0,
        should_be_valid=False,
    ),
    CheckpointCase(
        id="duration-within-tolerance",
        description="Stored 1798s, actual 1800s (0.1% off) → VALID",
        checkpoint_data={
            "words": [{"word": "hallo", "start": 0.0, "end": 0.4}],
            "segs": [],
            "audio_duration": 1798.0,
            "asr_elapsed": 600.0,
            "rtf": 0.33,
            "trim_sec": 0,
        },
        trim_sec_at_load=0,
        audio_exists=True,
        actual_duration_sec=1800.0,
        should_be_valid=True,
    ),
    CheckpointCase(
        id="old-format-no-trim-field-no-audio",
        description="Old checkpoint without trim_sec, audio file absent — cannot detect stale → VALID (best-effort)",
        checkpoint_data={
            "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
            "segs": [],
            "audio_duration": 40.0,
            "asr_elapsed": 12.0,
            "rtf": 0.3,
            # no trim_sec field
        },
        trim_sec_at_load=0,
        audio_exists=False,
        actual_duration_sec=0.0,
        should_be_valid=True,   # can't detect without either trim_sec or actual WAV
    ),
    CheckpointCase(
        id="segs-only-no-words-valid",
        description="Canary checkpoint: words=[], segs=[...] → VALID",
        checkpoint_data={
            "words": [],
            "segs": [{"text": "Das ist gut.", "start": 0.0, "end": 2.0}],
            "audio_duration": 120.0,
            "asr_elapsed": 40.0,
            "rtf": 0.33,
            "trim_sec": 0,
        },
        trim_sec_at_load=0,
        audio_exists=True,
        actual_duration_sec=120.0,
        should_be_valid=True,
    ),
]


def _make_wav(path: Path, duration_sec: float, sample_rate: int = 16000) -> None:
    """Write a minimal valid WAV file of the given duration."""
    n_frames = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


@pytest.mark.parametrize("case", _CHECKPOINT_CASES, ids=[c.id for c in _CHECKPOINT_CASES])
def test_validate_checkpoint(case: CheckpointCase, tmp_path: Path):
    checkpoint_file = tmp_path / "stem_nemo_de_transcript.json"

    if case.checkpoint_data.get("__corrupt__"):
        checkpoint_file.write_bytes(b"{ this is not valid json !!!")
    else:
        checkpoint_file.write_text(json.dumps(case.checkpoint_data), encoding="utf-8")

    if case.audio_exists:
        audio_path = tmp_path / "audio.wav"
        _make_wav(audio_path, case.actual_duration_sec)
    else:
        audio_path = tmp_path / "audio_nonexistent.wav"

    result = _validate_checkpoint(checkpoint_file, str(audio_path), case.trim_sec_at_load)

    assert result == case.should_be_valid, (
        f"[{case.id}] expected valid={case.should_be_valid}, got {result}\n"
        f"  description: {case.description}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. _strip_asr_repetition — matrix
# ═══════════════════════════════════════════════════════════════════════════════

_phrase5 = "Das ist gut für die"           # 5 words
_phrase6 = "Das ist gut für die Gesundheit"  # 6 words

_REPETITION_CASES: list[RepetitionCase] = [
    RepetitionCase(
        id="german-canary-loop-3x",
        description="Classic Canary hallucination: 6-word phrase repeated 3× exactly",
        input=f"{_phrase6}. {_phrase6}. {_phrase6}.",
        should_strip=True,
        expected_output=f"{_phrase6}.",
    ),
    RepetitionCase(
        id="german-canary-loop-10x",
        description="Severe hallucination: 6-word phrase repeated 10× → keep only first",
        input=(f"{_phrase6}. " * 10).strip(),
        should_strip=True,
        expected_output=f"{_phrase6}.",
    ),
    RepetitionCase(
        id="normal-german-text",
        description="Legitimate German text with no repetition",
        input="Guten Morgen. Wie geht es Ihnen heute? Ich hoffe, alles ist in Ordnung.",
        should_strip=False,
    ),
    RepetitionCase(
        id="short-text-below-threshold",
        description="Text shorter than min_unit_words * min_reps — never stripped",
        input="Hello world this is great",
        should_strip=False,
    ),
    RepetitionCase(
        id="two-reps-not-enough",
        description="Phrase repeated only 2× (min_reps=3) → should NOT be stripped",
        input=f"{_phrase5} is fine. {_phrase5} is fine.",
        should_strip=False,
    ),
    RepetitionCase(
        id="exactly-3-reps",
        description="Phrase repeated exactly 3× → stripped to first occurrence",
        input=f"{_phrase5} {_phrase5} {_phrase5}",
        should_strip=True,
        expected_output=_phrase5,
    ),
    RepetitionCase(
        id="single-word-repetition-stripped-as-5word-unit",
        description=(
            "'the'×20 IS stripped: algorithm detects 5-word unit 'the the the the the' "
            "repeating 4×. Correct — this is a real ASR hallucination pattern."
        ),
        input="the " * 20,
        should_strip=True,
        expected_output="the the the the the",
    ),
    RepetitionCase(
        id="repetition-starts-mid-sentence",
        description="Loop starts after a normal prefix — prefix preserved",
        input=f"Once upon a time {_phrase5} {_phrase5} {_phrase5}",
        should_strip=True,
        expected_output=f"Once upon a time {_phrase5}",
    ),
    RepetitionCase(
        id="english-parakeet-normal",
        description="Normal English Parakeet output, no repetition",
        input=(
            "In this video we will learn about neural networks and "
            "how they are used in modern machine learning applications."
        ),
        should_strip=False,
    ),
    RepetitionCase(
        id="empty-string",
        description="Empty input returns empty output",
        input="",
        should_strip=False,
    ),
]


@pytest.mark.parametrize("case", _REPETITION_CASES, ids=[c.id for c in _REPETITION_CASES])
def test_strip_asr_repetition(case: RepetitionCase):
    result = _strip_asr_repetition(case.input)

    if case.should_strip:
        assert len(result) < len(case.input), (
            f"[{case.id}] text should have been stripped but was not\n"
            f"  input length: {len(case.input)}, result length: {len(result)}"
        )
        assert result == case.expected_output, (
            f"[{case.id}] wrong stripped output\n"
            f"  expected: {case.expected_output!r}\n"
            f"  got     : {result!r}"
        )
    else:
        assert result == case.input, (
            f"[{case.id}] text should NOT have been modified\n"
            f"  input : {case.input!r}\n"
            f"  result: {result!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. _segs_to_srt — deduplication matrix
# ═══════════════════════════════════════════════════════════════════════════════

def _count_srt_blocks(srt: str) -> int:
    """Count subtitle blocks (non-empty numeric index lines)."""
    return len([l for l in srt.splitlines() if l.strip().isdigit()])


def _extract_srt_lines(srt: str) -> list[str]:
    """Return just the subtitle text lines (3rd line of each block)."""
    lines = srt.splitlines()
    texts = []
    i = 0
    while i < len(lines):
        if lines[i].strip().isdigit():
            # index, timestamp, text(s)
            i += 2  # skip index and timestamp
            while i < len(lines) and lines[i].strip():
                texts.append(lines[i].strip())
                i += 1
        else:
            i += 1
    return texts


_SEGS_TO_SRT_CASES: list[SegToSrtCase] = [
    SegToSrtCase(
        id="non-diarized-consecutive-dups-removed",
        description="Same text repeated 3× → only 1 SRT block",
        segments=[
            {"text": "Hello world", "start": 0.0, "end": 1.0},
            {"text": "Hello world", "start": 1.0, "end": 2.0},
            {"text": "Hello world", "start": 2.0, "end": 3.0},
        ],
        diarized=False,
        expected_count=1,
    ),
    SegToSrtCase(
        id="non-diarized-distinct-texts",
        description="5 distinct segments → 5 SRT blocks",
        segments=[
            {"text": f"Line {i}", "start": float(i), "end": float(i + 1)}
            for i in range(5)
        ],
        diarized=False,
        expected_count=5,
    ),
    SegToSrtCase(
        id="diarized-same-speaker-same-text-deduped",
        description="Diarized: same speaker, same text → 1 block (hallucination)",
        segments=[
            {"text": "Ja, genau.", "start": 0.0, "end": 1.0, "speaker": "spk_0"},
            {"text": "Ja, genau.", "start": 1.0, "end": 2.0, "speaker": "spk_0"},
        ],
        diarized=True,
        expected_count=1,
    ),
    SegToSrtCase(
        id="diarized-different-speakers-same-text-kept",
        description="Diarized: two different speakers, same text → 2 blocks (valid dialogue)",
        segments=[
            {"text": "Ja.", "start": 0.0, "end": 0.5, "speaker": "spk_0"},
            {"text": "Ja.", "start": 0.5, "end": 1.0, "speaker": "spk_1"},
        ],
        diarized=True,
        expected_count=2,
    ),
    SegToSrtCase(
        id="empty-text-filtered",
        description="Segments with empty text are silently skipped",
        segments=[
            {"text": "", "start": 0.0, "end": 1.0},
            {"text": "  ", "start": 1.0, "end": 2.0},
            {"text": "Hello", "start": 2.0, "end": 3.0},
        ],
        diarized=False,
        expected_count=1,
    ),
    SegToSrtCase(
        id="diarized-multi-speaker-distinct",
        description="3 speakers, distinct text → 3 blocks with speaker labels",
        segments=[
            {"text": "Guten Morgen.", "start": 0.0, "end": 1.0, "speaker": "spk_0"},
            {"text": "Hallo.", "start": 1.0, "end": 2.0, "speaker": "spk_1"},
            {"text": "Wie geht es?", "start": 2.0, "end": 3.5, "speaker": "spk_2"},
        ],
        diarized=True,
        expected_count=3,
    ),
    SegToSrtCase(
        id="diarized-same-text-alternating-speakers",
        description="ABAB pattern with same text — each is kept (different speakers)",
        segments=[
            {"text": "Okay.", "start": 0.0, "end": 0.5, "speaker": "spk_0"},
            {"text": "Okay.", "start": 0.5, "end": 1.0, "speaker": "spk_1"},
            {"text": "Okay.", "start": 1.0, "end": 1.5, "speaker": "spk_0"},
            {"text": "Okay.", "start": 1.5, "end": 2.0, "speaker": "spk_1"},
        ],
        diarized=True,
        expected_count=4,
    ),
    SegToSrtCase(
        id="non-diarized-alternating-same-different",
        description="dup, unique, dup, unique pattern → 3 blocks (dup collapses once)",
        segments=[
            {"text": "Same.", "start": 0.0, "end": 1.0},
            {"text": "Same.", "start": 1.0, "end": 2.0},
            {"text": "Different.", "start": 2.0, "end": 3.0},
            {"text": "Same.", "start": 3.0, "end": 4.0},   # NOT consecutive dup → kept
        ],
        diarized=False,
        expected_count=3,
    ),
]


@pytest.mark.parametrize("case", _SEGS_TO_SRT_CASES, ids=[c.id for c in _SEGS_TO_SRT_CASES])
def test_segs_to_srt(case: SegToSrtCase):
    srt = _segs_to_srt(case.segments, diarized=case.diarized)
    count = _count_srt_blocks(srt)

    assert count == case.expected_count, (
        f"[{case.id}] expected {case.expected_count} SRT blocks, got {count}\n"
        f"  description: {case.description}\n"
        f"  SRT output:\n{srt}"
    )

    if case.no_consecutive_dups and count > 1:
        text_lines = _extract_srt_lines(srt)
        for a, b in zip(text_lines, text_lines[1:]):
            # Strip speaker label before comparing (diarized mode adds "[Speaker N] ")
            a_clean = re.sub(r"^\[Speaker \d+\]\s*", "", a)
            b_clean = re.sub(r"^\[Speaker \d+\]\s*", "", b)
            if case.diarized:
                # In diarized mode, identical text from the same speaker is a dup;
                # different speakers with same text is valid (tested separately).
                # Here we only assert the test case's own expected_count is met.
                pass
            else:
                assert a_clean != b_clean, (
                    f"[{case.id}] consecutive duplicate SRT lines found: {a_clean!r}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. _split_coarse_segs — timing and speaker passthrough
# ═══════════════════════════════════════════════════════════════════════════════

_SPLIT_COARSE_CASES: list[SplitCoarseCase] = [
    SplitCoarseCase(
        id="single-short-seg-no-split",
        description="3-word segment fits in one line",
        segments=[{"text": "Hello world now", "start": 0.0, "end": 3.0}],
        expected_line_count_min=1,
        total_duration_preserved=True,
    ),
    SplitCoarseCase(
        id="long-seg-splits-into-multiple",
        description="30-word segment → at least 3 lines (max_w=10)",
        segments=[{
            "text": " ".join(["word"] * 30),
            "start": 0.0,
            "end": 30.0,
        }],
        max_w=10,
        expected_line_count_min=3,
        total_duration_preserved=True,
    ),
    SplitCoarseCase(
        id="speaker-tag-preserved",
        description="Speaker field passes through to all output lines",
        segments=[{
            "text": " ".join(["wort"] * 15),
            "start": 0.0,
            "end": 15.0,
            "speaker": "spk_0",
        }],
        max_w=5,
        expected_line_count_min=3,
        total_duration_preserved=True,
    ),
    SplitCoarseCase(
        id="empty-text-skipped",
        description="Segments with empty text produce no output lines",
        segments=[
            {"text": "", "start": 0.0, "end": 1.0},
            {"text": "  ", "start": 1.0, "end": 2.0},
        ],
        expected_line_count_min=0,
        total_duration_preserved=False,  # no output to compare
    ),
    SplitCoarseCase(
        id="multi-seg-timing-proportional",
        description="Two segments split correctly — durations proportional to word count",
        segments=[
            {"text": "short text", "start": 0.0, "end": 2.0},
            {"text": "this is a much longer piece of text with many more words", "start": 2.0, "end": 12.0},
        ],
        expected_line_count_min=2,
        total_duration_preserved=True,
    ),
]


@pytest.mark.parametrize("case", _SPLIT_COARSE_CASES, ids=[c.id for c in _SPLIT_COARSE_CASES])
def test_split_coarse_segs(case: SplitCoarseCase):
    result = _split_coarse_segs(case.segments, max_w=case.max_w, max_ch=case.max_ch)

    assert len(result) >= case.expected_line_count_min, (
        f"[{case.id}] expected ≥{case.expected_line_count_min} lines, got {len(result)}\n"
        f"  description: {case.description}"
    )

    if case.total_duration_preserved and result:
        input_dur = sum(
            max(0.0, s.get("end", 0.0) - s.get("start", 0.0))
            for s in case.segments
        )
        output_dur = sum(
            max(0.0, r.get("end", 0.0) - r.get("start", 0.0))
            for r in result
        )
        assert abs(output_dur - input_dur) < 0.01, (
            f"[{case.id}] total duration changed: input {input_dur:.3f}s, output {output_dur:.3f}s"
        )

    # Speaker must propagate when input has speaker field
    for seg in case.segments:
        if "speaker" in seg and result:
            for r in result:
                assert "speaker" in r, (
                    f"[{case.id}] speaker field lost in split output: {r}"
                )
            break

    # Every output segment must have start < end (or start == end for zero-dur edge case)
    for r in result:
        assert r.get("start", 0.0) <= r.get("end", 0.0), (
            f"[{case.id}] output segment has start > end: {r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SRT coverage ratio — the "only 40s of a 30min video" canary
# ═══════════════════════════════════════════════════════════════════════════════

def _last_srt_timestamp_sec(srt: str) -> float:
    """Extract the end timestamp (seconds) from the last SRT block."""
    # Match "HH:MM:SS,mmm --> HH:MM:SS,mmm" and take the end timestamp
    pattern = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})")
    matches = list(pattern.finditer(srt))
    if not matches:
        return 0.0
    h, m, s, ms = [int(x) for x in matches[-1].groups()[4:]]
    return h * 3600 + m * 60 + s + ms / 1000.0


def _build_srt_spanning(duration_sec: float) -> str:
    """Build a minimal valid SRT that spans approximately `duration_sec`."""
    lines = []
    t = 0.0
    idx = 1
    while t < duration_sec:
        end = min(t + 2.0, duration_sec)
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        ms = int((s % 1) * 1000)
        ts_start = f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"
        h2, rem2 = divmod(end, 3600)
        m2, s2 = divmod(rem2, 60)
        ms2 = int((s2 % 1) * 1000)
        ts_end = f"{int(h2):02d}:{int(m2):02d}:{int(s2):02d},{ms2:03d}"
        lines += [str(idx), f"{ts_start} --> {ts_end}", "Sample text.", ""]
        t += 2.0
        idx += 1
    return "\n".join(lines)


_SRT_COVERAGE_CASES: list[SrtCoverageCase] = [
    SrtCoverageCase(
        id="trim40-srt-against-full-1800s-audio",
        description="Stale --trim 40 SRT presented against 1800s audio → coverage ≈ 2% (fail threshold)",
        srt_text=_build_srt_spanning(40.0),
        audio_duration_sec=1800.0,
        min_coverage_ratio=0.5,   # 2% << 50% → should fail test
    ),
    SrtCoverageCase(
        id="full-srt-full-audio",
        description="Complete SRT for full 1800s audio → coverage ~100%",
        srt_text=_build_srt_spanning(1800.0),
        audio_duration_sec=1800.0,
        min_coverage_ratio=0.9,
    ),
    SrtCoverageCase(
        id="slight-underrun-acceptable",
        description="SRT ends at 1750s for 1800s audio (last 50s of silence) → ≥90% coverage",
        srt_text=_build_srt_spanning(1750.0),
        audio_duration_sec=1800.0,
        min_coverage_ratio=0.9,
    ),
    SrtCoverageCase(
        id="trim200-srt-against-full-3600s-audio",
        description="200s SRT against 3600s audio → coverage ≈ 5.5% (fail threshold)",
        srt_text=_build_srt_spanning(200.0),
        audio_duration_sec=3600.0,
        min_coverage_ratio=0.5,   # 5.5% << 50% → should fail
    ),
    SrtCoverageCase(
        id="empty-srt-zero-coverage",
        description="Empty SRT → 0% coverage",
        srt_text="",
        audio_duration_sec=1800.0,
        min_coverage_ratio=0.5,
    ),
]


@pytest.mark.parametrize("case", _SRT_COVERAGE_CASES, ids=[c.id for c in _SRT_COVERAGE_CASES])
def test_srt_coverage_ratio(case: SrtCoverageCase):
    last_ts = _last_srt_timestamp_sec(case.srt_text)
    coverage = last_ts / case.audio_duration_sec if case.audio_duration_sec > 0 else 0.0

    # Cases where coverage is expected to be BELOW the threshold are "negative" tests
    # (they validate that a stale SRT would indeed fail a real coverage check).
    below_threshold = coverage < case.min_coverage_ratio

    if case.id in {
        "trim40-srt-against-full-1800s-audio",
        "trim200-srt-against-full-3600s-audio",
        "empty-srt-zero-coverage",
    }:
        # These are negative cases: we EXPECT the coverage to be insufficient.
        # The test proves our helper correctly identifies the problem.
        assert below_threshold, (
            f"[{case.id}] Expected coverage < {case.min_coverage_ratio:.0%} "
            f"but got {coverage:.1%} — the stale-SRT detector would miss this"
        )
    else:
        # Positive cases: SRT must meet the coverage threshold.
        assert not below_threshold, (
            f"[{case.id}] SRT coverage {coverage:.1%} < required {case.min_coverage_ratio:.0%}\n"
            f"  last timestamp: {last_ts:.1f}s, audio: {case.audio_duration_sec:.1f}s"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. _srt_last_timestamp — the pipeline's own coverage helper
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("srt,expected_sec", [
    # Standard SRT with last block ending at 01:02:03,456
    (
        "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
        "2\n01:02:03,000 --> 01:02:03,456\nWorld\n\n",
        3723.456,
    ),
    # Empty SRT → 0.0
    ("", 0.0),
    # Single block
    ("1\n00:00:05,500 --> 00:00:07,800\nTest\n\n", 7.8),
    # Hours > 1
    ("1\n02:30:00,000 --> 02:30:10,250\nLong video\n\n", 9010.25),
], ids=["multi-block", "empty", "single-block", "long-video"])
def test_srt_last_timestamp(srt, expected_sec):
    result = _srt_last_timestamp(srt)
    assert abs(result - expected_sec) < 0.01, (
        f"Expected {expected_sec}s, got {result}s"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. _strip_asr_repetition — false positive resistance (legitimate German text)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Risk: The function truncates at the FIRST detected repetition.  In a real
# German academic lecture, the same 5-word phrase can appear scattered through
# the text — but NOT in 3+ consecutive occurrences.  These cases MUST pass
# through unmodified.

_GERMAN_LECTURE = (
    "In diesem Abschnitt betrachten wir die grundlegenden Konzepte der "
    "maschinellen Lernens. Zunächst ist es wichtig zu verstehen, dass "
    "neuronale Netze aus mehreren Schichten bestehen. Jede Schicht verarbeitet "
    "die Eingabedaten und gibt das Ergebnis an die nächste Schicht weiter. "
    "Dieser Prozess wird als Vorwärtsdurchlauf bezeichnet. In diesem Abschnitt "
    "haben wir also die wichtigsten Grundlagen kennengelernt. Es ist wichtig zu "
    "verstehen, dass das Training eines Modells viel Zeit in Anspruch nehmen kann. "
    "Die Optimierung der Hyperparameter ist dabei entscheidend für den Erfolg. "
    "Zusammenfassend lässt sich sagen, dass tiefe neuronale Netze sehr leistungsfähig "
    "sind und in vielen Bereichen eingesetzt werden können. In diesem Abschnitt "
    "wurden die theoretischen Grundlagen erläutert."
)
# Note: "In diesem Abschnitt" appears 3 times but NOT consecutively — valid text.

_GERMAN_REPEATED_CLAUSE = (
    "Es ist wichtig zu verstehen. Es ist wichtig zu verstehen. "
    "Es ist wichtig zu verstehen. Das Modell konvergiert."
)
# "Es ist wichtig zu verstehen" (5 words) × 3 consecutive — IS hallucination.


@pytest.mark.parametrize("text,should_strip,description", [
    (
        _GERMAN_LECTURE,
        False,
        "Realistic lecture: 'In diesem Abschnitt' appears 3× but not consecutively",
    ),
    (
        _GERMAN_REPEATED_CLAUSE,
        True,
        "5-word clause repeated 3× consecutively → hallucination, must strip",
    ),
    (
        # Phrase appears at positions 0 and ~middle but not ×3 consecutive
        "Das Modell wird trainiert und die Ergebnisse werden ausgewertet. "
        "Anschließend folgt die Evaluierung der Leistung. "
        "Das Modell wird trainiert mit neuen Daten zur Verbesserung.",
        False,
        "Same 4-word start at pos 0 and ~end but only 2× not consecutive → keep",
    ),
    (
        # Legitimate repetition for emphasis (2× only) → NOT stripped (min_reps=3)
        "Das ist sehr wichtig für das Verständnis. "
        "Das ist sehr wichtig für das Verständnis. "
        "Bitte merken Sie sich diesen Punkt.",
        False,
        "Same 7-word clause repeated exactly 2× (< min_reps=3) → keep",
    ),
])
def test_strip_asr_repetition_false_positives(text, should_strip, description):
    result = _strip_asr_repetition(text)
    if should_strip:
        assert len(result) < len(text), (
            f"Expected strip but text unchanged.\n  desc: {description}\n  text: {text[:80]!r}"
        )
    else:
        assert result == text, (
            f"False positive — legitimate text was modified!\n"
            f"  desc: {description}\n"
            f"  original ({len(text)} chars): {text[:100]!r}\n"
            f"  result   ({len(result)} chars): {result[:100]!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. _segs_to_srt word preservation — no silent content loss
# ═══════════════════════════════════════════════════════════════════════════════
#
# If every segment has unique text, ALL words must appear in the SRT output.
# Deduplication must only fire on consecutive identical text — never drop
# distinct content.

def _count_words_in_srt(srt: str) -> int:
    """Count content words in SRT (skipping index lines and timestamps)."""
    words = 0
    for line in srt.splitlines():
        line = line.strip()
        if not line or line.isdigit() or "-->" in line:
            continue
        words += len(line.split())
    return words


@pytest.mark.parametrize("seg_count,diarized", [
    (10, False),
    (10, True),
    (50, False),
    (50, True),
])
def test_segs_to_srt_preserves_all_words(seg_count, diarized):
    """All distinct segments → no words lost in SRT output."""
    speakers = ["spk_0", "spk_1", "spk_2"]
    segments = []
    total_input_words = 0
    for i in range(seg_count):
        # Unique text per segment: guaranteed no consecutive dups
        text = f"segment {i} unique content word alpha beta"
        total_input_words += len(text.split())
        seg = {"text": text, "start": float(i * 3), "end": float(i * 3 + 2)}
        if diarized:
            seg["speaker"] = speakers[i % len(speakers)]
        segments.append(seg)

    srt = _segs_to_srt(segments, diarized=diarized)
    output_words = _count_words_in_srt(srt)

    # In diarized mode, speaker labels add extra words — allow for them.
    # Core assertion: output must have at LEAST the input word count.
    label_overhead = seg_count * 2 if diarized else 0  # "[Speaker N]" = 2 words each
    assert output_words >= total_input_words, (
        f"Word count shrank: input {total_input_words} words, "
        f"output {output_words} words (seg_count={seg_count}, diarized={diarized})\n"
        f"  This means content was silently dropped by _segs_to_srt."
    )
    assert output_words <= total_input_words + label_overhead + 5, (
        f"Word count grew unexpectedly: input {total_input_words}, output {output_words}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. _estimate_chunk_sec — Canary chunk size cap
# ═══════════════════════════════════════════════════════════════════════════════
#
# THE BUG THIS CATCHES:
#   With a large GPU (≥8 GB free VRAM), _estimate_chunk_sec returned up to 600s.
#   Canary fed 600s (10-min) chunks produces compressed output (~320 words for
#   44 minutes of speech instead of the expected ~6,600 words).
#   Fix: cap Canary at 60s regardless of VRAM.

import unittest.mock as _mock

@pytest.mark.parametrize("free_gb,model_name,expected_sec", [
    # ── Canary (encoder-decoder): always exactly 60s regardless of VRAM ──────
    (8,   "nvidia/canary-1b-v2",       60),
    (16,  "nvidia/canary-1b-v2",       60),
    (48,  "nvidia/canary-1b-v2",       60),
    (0.5, "nvidia/canary-1b-v2",       60),
    # ── Parakeet v2 (CTC/TDT, EN): proportional — more VRAM → bigger chunks ──
    # 6GB free: usable=(6-2)*0.8=3.2GB → 3.2/0.28*60=685s → capped at 3600 → 685
    (6,   "nvidia/parakeet-tdt-0.6b-v2",  685),
    # 16GB free: usable=11.2GB → 2400s
    (16,  "nvidia/parakeet-tdt-0.6b-v2",  2400),
    # 48GB free: usable=36.8GB → 7886s → capped at 7200
    (48,  "nvidia/parakeet-tdt-0.6b-v2",  7200),
    # ── Parakeet v3 (CTC/TDT, 25 langs): same formula ─────────────────────
    (6,   "nvidia/parakeet-tdt-0.6b-v3",  685),
    (16,  "nvidia/parakeet-tdt-0.6b-v3",  2400),
    (48,  "nvidia/parakeet-tdt-0.6b-v3",  7200),
], ids=[
    "canary-8gb", "canary-16gb", "canary-48gb", "canary-tiny-vram",
    "parakeet-v2-6gb", "parakeet-v2-16gb", "parakeet-v2-48gb",
    "parakeet-v3-6gb", "parakeet-v3-16gb", "parakeet-v3-48gb",
])
def test_estimate_chunk_sec_canary_cap(free_gb, model_name, expected_sec):
    """Chunk size must match expected: Canary=60s always, Parakeet scales with VRAM."""
    total_gb = max(free_gb, 24.0)
    with _mock.patch("nemo_model._vram_gb", return_value=(free_gb, total_gb)):
        result = _estimate_chunk_sec(model_name, safety=0.8, reserve_gb=2.0)

    assert result == expected_sec, (
        f"[{model_name} @ {free_gb}GB free] chunk_sec={result}s, expected {expected_sec}s\n"
        f"  Canary must be 60s (encoder-decoder quality limit).\n"
        f"  Parakeet must scale with VRAM so 48GB GPU processes faster than 16GB."
    )
    assert result >= 30, f"chunk_sec={result}s is below 30s minimum"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. _strip_special_tokens — Canary/encoder-decoder EOS token cleanup
#
# Canary (and Whisper-style) models emit <|endoftext|> tokens when they run
# past the actual speech.  These must be removed before the text reaches the
# SRT builder, otherwise subtitle lines contain raw model tokens.
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("raw,expected", [
    # The exact pattern from the user's real transcript
    (
        "Laut Soko Tierschutz gab es die so nicht."
        + "<|endoftext|>" * 130
        + ".<|endoftext|>.<|endoftext|>" + "." * 200,
        "Laut Soko Tierschutz gab es die so nicht.",
    ),
    # No special tokens — text unchanged
    (
        "Das ist ein normaler Satz.",
        "Das ist ein normaler Satz.",
    ),
    # Mixed tokens in the middle
    (
        "Wir sind <|startoftranscript|> hier <|endoftext|> fertig.",
        "Wir sind hier fertig.",
    ),
    # Only tokens + dots → empty after strip
    (
        "<|endoftext|><|endoftext|>.<|endoftext|>....",
        "",
    ),
    # Long dot run collapsed to ellipsis
    (
        "Und dann" + "." * 20,
        "Und dann...",
    ),
    # Empty string stays empty
    ("", ""),
], ids=[
    "real-canary-endoftext-storm",
    "clean-text-unchanged",
    "token-in-middle",
    "only-tokens-becomes-empty",
    "dot-run-collapsed",
    "empty-string",
])
def test_strip_special_tokens(raw, expected):
    result = _strip_special_tokens(raw)
    assert result == expected, (
        f"Expected: {expected!r}\n"
        f"Got:      {result!r}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Qwen3-ASR — model detection, language mapping, transcription output parsing
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("model_name,expected", [
    ("Qwen/Qwen3-ASR-1.7B",  True),
    ("Qwen/Qwen3-ASR-0.6B",  True),
    ("qwen3-asr",            True),
    ("qwen3-asr-s",          True),
    ("nvidia/canary-1b-v2",  False),
    ("nvidia/parakeet-tdt-0.6b-v3", False),
], ids=["1.7b", "0.6b", "shortname-full", "shortname-small",
        "canary", "parakeet-v3"])
def test_is_qwen3_asr(model_name, expected):
    assert _is_qwen3_asr(model_name) == expected


@pytest.mark.parametrize("iso,expected_name", [
    ("en", "English"), ("de", "German"), ("fr", "French"),
    ("es", "Spanish"), ("it", "Italian"), ("nl", "Dutch"),
    ("zh", "Chinese"), ("ja", "Japanese"), ("ar", "Arabic"),
])
def test_qwen3_lang_map(iso, expected_name):
    assert QWEN3_LANG_MAP[iso] == expected_name


def test_qwen3_lang_map_unknown_returns_none():
    """Unknown ISO code → None (auto-detect mode)."""
    assert QWEN3_LANG_MAP.get("xx") is None


@pytest.mark.parametrize("ts_list,text,expected_words,expected_segs", [
    # Normal case: ForcedAligner returns word timestamps
    (
        [
            type("TS", (), {"text": "Hallo", "start_time": 0.5, "end_time": 1.0})(),
            type("TS", (), {"text": "Welt",  "start_time": 1.1, "end_time": 1.6})(),
        ],
        "Hallo Welt",
        [{"word": "Hallo", "start": 1.5, "end": 2.0},   # +1.0 offset
         {"word": "Welt",  "start": 2.1, "end": 2.6}],
        [],
    ),
    # Fallback: no timestamps → segment-level output
    (
        [],
        "Bonjour le monde",
        [],
        1,  # one segment expected (checked as count)
    ),
    # Special tokens in word text → stripped, offset=1.0 applied
    (
        [
            type("TS", (), {"text": "<|im_end|>", "start_time": 0.0, "end_time": 0.1})(),
            type("TS", (), {"text": "Hello",      "start_time": 0.2, "end_time": 0.5})(),
        ],
        "Hello",
        [{"word": "Hello", "start": 1.2, "end": 1.5}],   # 0.2+1.0, 0.5+1.0
        [],
    ),
], ids=["word-timestamps", "segment-fallback", "special-token-stripped"])
def test_qwen3_asr_output_parsing(ts_list, text, expected_words, expected_segs, tmp_path):
    """_transcribe_qwen3_asr parses model output into our word/seg format."""
    import wave, struct

    # Write a minimal valid WAV so _audio_duration works
    wav = tmp_path / "test.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))

    # Build a mock model result
    mock_result = type("R", (), {"text": text, "time_stamps": ts_list})()
    mock_model  = type("M", (), {"transcribe": lambda self, **kw: [mock_result]})()

    words, segs = _transcribe_qwen3_asr(mock_model, str(wav), offset=1.0, src_lang="de")

    assert words == expected_words, f"words mismatch: {words}"
    if isinstance(expected_segs, int):
        assert len(segs) == expected_segs, f"expected {expected_segs} seg(s), got {segs}"
        assert segs[0]["text"] == text
    else:
        assert segs == expected_segs


@pytest.mark.parametrize("free_gb,model_name", [
    (8,  "Qwen/Qwen3-ASR-1.7B"),
    (48, "Qwen/Qwen3-ASR-1.7B"),
    (4,  "Qwen/Qwen3-ASR-0.6B"),
], ids=["qwen3-8gb", "qwen3-48gb", "qwen3-small"])
def test_estimate_chunk_sec_qwen3(free_gb, model_name):
    """Qwen3-ASR chunk size is VRAM-driven (no hardcoded 120s cap)."""
    total_gb = max(free_gb, 24.0)
    safety, reserve = 0.8, 2.0
    with _mock.patch("nemo_model._vram_gb", return_value=(free_gb, total_gb)):
        result = _estimate_chunk_sec(model_name, safety=safety, reserve_gb=reserve)
    usable = max(0.0, free_gb - reserve) * safety
    expected = max(30, min(int(usable / 0.35 * 60), 7200))
    assert result == expected, f"Expected {expected}s (VRAM-driven), got {result}s"
    assert result > 120 or free_gb <= 2.0, "On any reasonable GPU, chunk must exceed old 120s hardcode"

# ═══════════════════════════════════════════════════════════════════════════════
# WAV + trim regression test
# Bug: when input is a .wav file AND trim_sec > 0, FFmpeg trim was silently
# skipped — the full audio (e.g. 44 minutes) was transcribed instead of 2 min.
# ═══════════════════════════════════════════════════════════════════════════════

from nemo_diarize import _run_with_model
from nemo_audio import _extract_audio


def _make_tiny_wav(path: Path, duration_sec: float = 5.0, sr: int = 16000) -> None:
    import wave, struct, math
    n = int(sr * duration_sec)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n}h", *[int(32767 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(n)]))


def test_wav_input_with_trim_calls_extract_audio(tmp_path):
    """Regression: WAV + trim_sec > 0 must call _extract_audio (not skip it)."""
    wav = tmp_path / "test_audio.wav"
    _make_tiny_wav(wav, duration_sec=5.0)

    model = _mock.MagicMock()
    calls = []

    def fake_extract(src, dst, trim):
        calls.append((src, dst, trim))
        # Create a minimal output file so the rest of _run_with_model can continue
        _make_tiny_wav(Path(dst), duration_sec=min(trim, 5.0) if trim else 5.0)

    with (
        _mock.patch("nemo_diarize._extract_audio", side_effect=fake_extract),
        _mock.patch("nemo_diarize._transcribe_chunked",
                    return_value=([{"word": "Hallo", "start": 0.0, "end": 0.5}], [], [])),
        _mock.patch("nemo_diarize._run_diarization", return_value=[]),
        _mock.patch("nemo_audio._vram_gb", return_value=(16.0, 24.0)),
        _mock.patch("nemo_model._vram_gb", return_value=(16.0, 24.0)),
        _mock.patch("torch.cuda.is_available", return_value=False),
    ):
        _run_with_model(
            model=model,
            video_path=str(wav),
            language="de",
            model_name="Qwen/Qwen3-ASR-1.7B",
            translate=False,
            diarize=True,
            trim_sec=120,
            safety_factor=0.85,
            reserve_gb=1.5,
            chunk_override_sec=None,
        )

    assert len(calls) == 1, "Expected _extract_audio to be called exactly once for WAV + trim"
    src, dst, trim = calls[0]
    assert src == str(wav), "Source must be the original WAV"
    assert dst != str(wav), "Destination must be a DIFFERENT (trimmed) path, not the original WAV"
    assert trim == 120, f"Trim must be 120, got {trim}"


def test_wav_input_without_trim_skips_extract_audio(tmp_path):
    """WAV + trim_sec=0 should use the WAV directly (no FFmpeg needed)."""
    wav = tmp_path / "test_audio.wav"
    _make_tiny_wav(wav, duration_sec=5.0)

    model = _mock.MagicMock()
    calls = []

    with (
        _mock.patch("nemo_diarize._extract_audio", side_effect=lambda s, d, t: calls.append((s, d, t))),
        _mock.patch("nemo_diarize._transcribe_chunked",
                    return_value=([{"word": "Hallo", "start": 0.0, "end": 0.5}], [], [])),
        _mock.patch("nemo_diarize._run_diarization", return_value=[]),
        _mock.patch("nemo_audio._vram_gb", return_value=(16.0, 24.0)),
        _mock.patch("nemo_model._vram_gb", return_value=(16.0, 24.0)),
        _mock.patch("torch.cuda.is_available", return_value=False),
    ):
        _run_with_model(
            model=model,
            video_path=str(wav),
            language="de",
            model_name="Qwen/Qwen3-ASR-1.7B",
            translate=False,
            diarize=False,
            trim_sec=0,
            safety_factor=0.85,
            reserve_gb=1.5,
            chunk_override_sec=None,
        )

    assert len(calls) == 0, "WAV without trim must NOT call _extract_audio"

# ═══════════════════════════════════════════════════════════════════════════════
# _run_with_model branch coverage — targeted tests for each critical path.
#
# CONTEXT: We had 500 tests but still hit the WAV+trim bug in production.
# Root cause: _run_with_model was never called in tests — only its sub-functions
# were tested individually. The bug lived in the COMBINATION of conditions.
# These tests fix that by exercising every branch of _run_with_model directly.
# ═══════════════════════════════════════════════════════════════════════════════

import gc


def _write_transcript_checkpoint(path, words=None, segs=None, audio_dur=5.0,
                                  asr_elapsed=1.0, rtf=0.2, trim_sec=0):
    import json
    data = {
        "words": words or [{"word": "Hallo", "start": 0.0, "end": 0.5}],
        "segs": segs or [],
        "audio_duration": audio_dur,
        "asr_elapsed": asr_elapsed,
        "rtf": rtf,
        "trim_sec": trim_sec,
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_diarization_checkpoint(path, turns=None):
    import json
    path.write_text(json.dumps({"turns": turns or []}), encoding="utf-8")


def _base_patches(transcribe_return=None):
    """Common patches needed for all _run_with_model tests.

    Returns a contextlib.ExitStack so callers can use: ``with _base_patches(): ...``
    Extra patches can be added inside the block via the stack returned by the ``as`` clause.
    """
    import contextlib
    if transcribe_return is None:
        transcribe_return = ([{"word": "Hallo", "start": 0.0, "end": 0.5}], [], [])
    stack = contextlib.ExitStack()
    stack.enter_context(_mock.patch("nemo_diarize._transcribe_chunked", return_value=transcribe_return))
    stack.enter_context(_mock.patch("nemo_diarize._run_diarization", return_value=[]))
    stack.enter_context(_mock.patch("nemo_audio._vram_gb", return_value=(16.0, 24.0)))
    stack.enter_context(_mock.patch("nemo_model._vram_gb", return_value=(16.0, 24.0)))
    stack.enter_context(_mock.patch("torch.cuda.is_available", return_value=False))
    return stack


# ── Branch 3: Fast resume (both checkpoints valid) ────────────────────────────

def test_fast_resume_skips_transcription(tmp_path):
    """Both checkpoints valid → _transcribe_chunked must NOT be called."""
    wav = tmp_path / "video.wav"
    _make_tiny_wav(wav)

    stem = "video"
    transcript = tmp_path / f"{stem}_nemo_de_transcript.json"
    diarization = tmp_path / f"{stem}_nemo_de_diarization.json"
    _write_transcript_checkpoint(transcript, audio_dur=5.0, trim_sec=0)
    _write_diarization_checkpoint(diarization)

    model = _mock.MagicMock()
    transcribe_calls = []

    with (
        _mock.patch("nemo_diarize._transcribe_chunked",
                    side_effect=lambda *a, **kw: transcribe_calls.append(1) or ([], [], [])),
        _mock.patch("nemo_audio._vram_gb", return_value=(16.0, 24.0)),
        _mock.patch("torch.cuda.is_available", return_value=False),
    ):
        result = _run_with_model(
            model=model, video_path=str(wav), language="de",
            model_name="nvidia/parakeet-tdt-0.6b-v3",
            translate=False, diarize=True, trim_sec=0,
            safety_factor=0.85, reserve_gb=1.5, chunk_override_sec=None,
        )

    assert len(transcribe_calls) == 0, "Fast resume must skip ASR entirely"
    assert result  # SRT was produced from checkpoints


# ── Branch 4: Fast resume — stale checkpoint discarded ────────────────────────

def test_fast_resume_stale_checkpoint_retranscribes(tmp_path):
    """Stale trim_sec in checkpoint → discards and re-transcribes."""
    wav = tmp_path / "video.wav"
    _make_tiny_wav(wav)

    stem = "video"
    transcript = tmp_path / f"{stem}_nemo_de_transcript.json"
    diarization = tmp_path / f"{stem}_nemo_de_diarization.json"
    # Checkpoint says trim=40 but we're running trim=0 → stale
    _write_transcript_checkpoint(transcript, audio_dur=5.0, trim_sec=40)
    _write_diarization_checkpoint(diarization)

    model = _mock.MagicMock()
    transcribe_calls = []

    with (
        _mock.patch("nemo_diarize._transcribe_chunked",
                    side_effect=lambda *a, **kw: (
                        transcribe_calls.append(1),
                        ([{"word": "Hallo", "start": 0.0, "end": 0.5}], [], [])
                    )[1]),
        _mock.patch("nemo_diarize._run_diarization", return_value=[]),
        _mock.patch("nemo_audio._vram_gb", return_value=(16.0, 24.0)),
        _mock.patch("nemo_model._vram_gb", return_value=(16.0, 24.0)),
        _mock.patch("torch.cuda.is_available", return_value=False),
    ):
        _run_with_model(
            model=model, video_path=str(wav), language="de",
            model_name="nvidia/parakeet-tdt-0.6b-v3",
            translate=False, diarize=False, trim_sec=0,
            safety_factor=0.85, reserve_gb=1.5, chunk_override_sec=None,
        )

    assert len(transcribe_calls) == 1, "Stale checkpoint must trigger re-transcription"
    assert not transcript.exists() or json.loads(transcript.read_text())["trim_sec"] == 0


# ── Branch 5a: Cached audio reuse ─────────────────────────────────────────────

def test_mp4_input_reuses_existing_audio(tmp_path):
    """If extracted WAV already exists, _extract_audio must NOT be called again."""
    mp4 = tmp_path / "video.mp4"
    mp4.write_bytes(b"fake-mp4")  # doesn't need to be real, extraction is mocked

    # Pre-create the extracted WAV
    extracted = tmp_path / "video_nemo_16k_full.wav"
    _make_tiny_wav(extracted)

    extract_calls = []

    with _base_patches() as stack:
        stack.enter_context(
            _mock.patch("nemo_diarize._extract_audio",
                        side_effect=lambda *a: extract_calls.append(1))
        )
        _run_with_model(
            model=_mock.MagicMock(), video_path=str(mp4), language="de",
            model_name="nvidia/parakeet-tdt-0.6b-v3",
            translate=False, diarize=False, trim_sec=0,
            safety_factor=0.85, reserve_gb=1.5, chunk_override_sec=None,
        )

    assert len(extract_calls) == 0, "Existing audio must be reused without re-extracting"


# ── Branch 6a: Transcript checkpoint resume ───────────────────────────────────

def test_transcript_checkpoint_skips_asr_reruns_diarization(tmp_path):
    """Valid transcript checkpoint → skip ASR, still run diarization if missing."""
    wav = tmp_path / "video.wav"
    _make_tiny_wav(wav)

    transcript = tmp_path / "video_nemo_de_transcript.json"
    _write_transcript_checkpoint(transcript, audio_dur=5.0, trim_sec=0)
    # No diarization checkpoint → must run diarization

    transcribe_calls = []
    diarize_calls = []

    with (
        _mock.patch("nemo_diarize._transcribe_chunked",
                    side_effect=lambda *a, **kw: transcribe_calls.append(1) or ([], [], [])),
        _mock.patch("nemo_diarize._run_diarization",
                    side_effect=lambda *a: diarize_calls.append(1) or []),
        _mock.patch("nemo_audio._vram_gb", return_value=(16.0, 24.0)),
        _mock.patch("nemo_model._vram_gb", return_value=(16.0, 24.0)),
        _mock.patch("torch.cuda.is_available", return_value=False),
    ):
        _run_with_model(
            model=_mock.MagicMock(), video_path=str(wav), language="de",
            model_name="nvidia/parakeet-tdt-0.6b-v3",
            translate=False, diarize=True, trim_sec=0,
            safety_factor=0.85, reserve_gb=1.5, chunk_override_sec=None,
        )

    assert len(transcribe_calls) == 0, "Transcript checkpoint must skip ASR"
    assert len(diarize_calls) == 1, "Diarization must still run if checkpoint missing"


# ── Branch 7a: Canary chunk override cap at 60s ───────────────────────────────

def test_canary_chunk_override_capped_at_60s(tmp_path):
    """Canary + chunk_override=300 → chunk must be capped at 60s."""
    wav = tmp_path / "video.wav"
    _make_tiny_wav(wav)

    chunk_used = []

    def fake_transcribe(model, audio_path, model_name, src_lang, tgt_lang, chunk_sec):
        chunk_used.append(chunk_sec)
        # Canary clears words after transcription, so return non-empty segs to avoid crash
        return (
            [{"word": "Hallo", "start": 0.0, "end": 0.5}],
            [{"text": "Hallo", "start": 0.0, "end": 0.5}],
            [],
        )

    with (
        _mock.patch("nemo_diarize._transcribe_chunked", side_effect=fake_transcribe),
        _mock.patch("nemo_diarize._run_diarization", return_value=[]),
        _mock.patch("nemo_audio._vram_gb", return_value=(16.0, 24.0)),
        _mock.patch("nemo_model._vram_gb", return_value=(16.0, 24.0)),
        _mock.patch("torch.cuda.is_available", return_value=False),
    ):
        _run_with_model(
            model=_mock.MagicMock(), video_path=str(wav), language="en",
            model_name="nvidia/canary-1b-v2",
            translate=False, diarize=False, trim_sec=0,
            safety_factor=0.85, reserve_gb=1.5, chunk_override_sec=300,
        )

    assert chunk_used, "Transcription must have been called"
    assert chunk_used[0] <= 60, (
        f"Canary chunk must be capped at 60s even if override is 300s, got {chunk_used[0]}"
    )


# ── Branch 8b/9: Canary model — words cleared, segs used ─────────────────────

def test_canary_words_cleared_segs_used_for_srt(tmp_path):
    """Canary returns segs (not words). words must be cleared; SRT built from segs."""
    wav = tmp_path / "video.wav"
    _make_tiny_wav(wav)

    canary_segs = [{"text": "Hallo Welt", "start": 0.0, "end": 1.5}]
    canary_words = [{"word": "Hallo", "start": 0.0, "end": 0.5},
                    {"word": "Welt",  "start": 0.6, "end": 1.5}]

    with (
        _mock.patch("nemo_diarize._transcribe_chunked",
                    return_value=(canary_words, canary_segs, [])),
        _mock.patch("nemo_diarize._run_diarization", return_value=[]),
        _mock.patch("nemo_audio._vram_gb", return_value=(16.0, 24.0)),
        _mock.patch("nemo_model._vram_gb", return_value=(16.0, 24.0)),
        _mock.patch("torch.cuda.is_available", return_value=False),
    ):
        srt = _run_with_model(
            model=_mock.MagicMock(), video_path=str(wav), language="en",
            model_name="nvidia/canary-1b-v2",
            translate=False, diarize=False, trim_sec=0,
            safety_factor=0.85, reserve_gb=1.5, chunk_override_sec=None,
        )

    assert "Hallo Welt" in srt, "SRT must contain text from canary segs"


# ── Branch 8c: Parakeet (default) transcription path ─────────────────────────

def test_parakeet_model_dispatches_via_transcribe_chunked(tmp_path):
    """Parakeet model (default) must call _transcribe_chunked and produce SRT."""
    wav = tmp_path / "video.wav"
    _make_tiny_wav(wav)

    with _base_patches():
        srt = _run_with_model(
            model=_mock.MagicMock(), video_path=str(wav), language="en",
            model_name="nvidia/parakeet-tdt-0.6b-v3",
            translate=False, diarize=False, trim_sec=0,
            safety_factor=0.85, reserve_gb=1.5, chunk_override_sec=None,
        )

    assert srt  # produced a non-empty SRT


# ── Branch 10a: Diarization checkpoint resume ─────────────────────────────────

def test_diarization_checkpoint_not_rerun(tmp_path):
    """Valid diarization checkpoint → _run_diarization must NOT be called again."""
    wav = tmp_path / "video.wav"
    _make_tiny_wav(wav)

    transcript = tmp_path / "video_nemo_de_transcript.json"
    diarization = tmp_path / "video_nemo_de_diarization.json"
    _write_transcript_checkpoint(transcript, audio_dur=5.0, trim_sec=0)
    _write_diarization_checkpoint(diarization, turns=[
        {"speaker": "SPEAKER_0", "start": 0.0, "end": 2.0}
    ])

    diarize_calls = []

    with (
        _mock.patch("nemo_diarize._run_diarization",
                    side_effect=lambda *a: diarize_calls.append(1) or []),
        _mock.patch("nemo_audio._vram_gb", return_value=(16.0, 24.0)),
        _mock.patch("torch.cuda.is_available", return_value=False),
    ):
        srt = _run_with_model(
            model=_mock.MagicMock(), video_path=str(wav), language="de",
            model_name="nvidia/parakeet-tdt-0.6b-v3",
            translate=False, diarize=True, trim_sec=0,
            safety_factor=0.85, reserve_gb=1.5, chunk_override_sec=None,
        )

    assert len(diarize_calls) == 0, "Cached diarization must not trigger re-run"
    assert "Speaker" in srt, "SRT must include speaker labels from cached diarization"


# ── Branch 11: Coverage warning for incomplete SRT ───────────────────────────

def test_coverage_warning_logged_for_short_srt(tmp_path, caplog):
    """Audio > 60s but SRT ends at <50% → error logged, no exception raised."""
    import logging

    # Write a WAV that looks long (we fake the duration, not actual length)
    wav = tmp_path / "video.wav"
    _make_tiny_wav(wav, duration_sec=1.0)

    # Return a segment that only covers 5 seconds of "120 second" audio
    short_seg = [{"text": "Hallo", "start": 0.0, "end": 5.0}]

    with (
        _mock.patch("nemo_diarize._transcribe_chunked",
                    return_value=([], short_seg, [])),
        _mock.patch("nemo_diarize._audio_duration", return_value=120.0),
        _mock.patch("nemo_diarize._run_diarization", return_value=[]),
        _mock.patch("nemo_audio._vram_gb", return_value=(16.0, 24.0)),
        _mock.patch("nemo_model._vram_gb", return_value=(16.0, 24.0)),
        _mock.patch("torch.cuda.is_available", return_value=False),
        caplog.at_level(logging.WARNING, logger="nemo_local"),
    ):
        srt = _run_with_model(
            model=_mock.MagicMock(), video_path=str(wav), language="de",
            model_name="nvidia/parakeet-tdt-0.6b-v3",
            translate=False, diarize=False, trim_sec=0,
            safety_factor=0.85, reserve_gb=1.5, chunk_override_sec=None,
        )

    assert srt  # still returns SRT even with low coverage (no crash)
    assert any("COVERAGE" in r.message.upper() or "coverage" in r.message.lower()
               for r in caplog.records), "Low coverage must log an error/warning"
