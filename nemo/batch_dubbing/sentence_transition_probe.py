"""Synthetic bilingual transition fixture and lossless placement checks.

No model imports, execution, audio processing or file IO. Fixture pauses are
invented test constraints, not measured source-video speech. Count/placement
checks cannot establish whether the synthesizer pronounced sentence endings.
"""
from copy import deepcopy
import math

from batch_dubbing.cheap_native_probe import MODEL, REVISION, IMAGE_ID, PACKAGE

MAX_INTERNAL_CALLS = 8
MAX_CLIP_SECONDS = 90
MAX_STAGE_SECONDS = 300
MAX_BYTES = 64 * 1024**2

_EN_LONG = (
    "Before we begin the demonstration, I want to explain why a long sentence needs enough time "
    "for every word, why the small pauses between its ideas should remain easy to hear, and why "
    "the final part must stay complete even when another speaker is waiting to continue, because "
    "a listener should understand the whole explanation without hearing a sudden cut or an "
    "unexpected change in the speaking pace. That is the complete explanation."
)
_DE_LONG = (
    "Bevor wir mit der Vorführung beginnen, möchte ich erklären, warum ein langer Satz genug "
    "Zeit für jedes einzelne Wort braucht, warum die kleinen Pausen zwischen seinen Gedanken "
    "gut hörbar bleiben sollten und warum sein letzter Teil vollständig erhalten bleiben muss, "
    "auch wenn jemand anderes schon darauf wartet weiterzusprechen, denn alle sollen die ganze "
    "Erklärung verstehen können, ohne einen plötzlichen Schnitt oder einen unerwarteten Wechsel "
    "des Sprechtempos zu hören. Damit ist die Erklärung vollständig."
)


def build_fixture():
    """Return a fresh copy of the fixed test material; this does not release it."""
    texts = (("en", "We can begin this short test.", 1), ("en", _EN_LONG, 2),
             ("en", "The next short sentence starts here.", 1),
             ("de", "Wir können diesen kurzen Test beginnen.", 1), ("de", _DE_LONG, 2),
             ("de", "Der nächste kurze Satz beginnt jetzt.", 1))
    return {"schema": 1, "model": MODEL, "revision": REVISION, "package": PACKAGE,
            "image_id": IMAGE_ID, "speed": 1.0, "steps": 8,
            "max_chunk_length": 300, "silence_duration": .3, "gap_seconds": .4,
            "language_order": ["en", "de"], "max_api_calls": 6, "max_internal_calls": MAX_INTERNAL_CALLS,
            "max_clip_seconds": MAX_CLIP_SECONDS, "max_stage_seconds": MAX_STAGE_SECONDS,
            "max_stage_bytes": MAX_BYTES, "evidence_kind": "synthetic_fixture",
            "samples": [{"id": f"sample_{i}", "language": language, "text": text,
                         "voice": "M1", "expected_chunks": chunks,
                         "role": "long" if chunks == 2 else "short",
                         "seed": 20261029 + i}
                        for i, (language, text, chunks) in enumerate(texts)],
            "quality_limit": "Synthetic phrase-transition test only. No source-video timing, translation, pronunciation or naturalness is verified."}


def validate_fixture(data, *, chunker):
    """Check frozen scope and actual pinned-package chunking before synthesis."""
    if not isinstance(data, dict) or data != build_fixture():
        raise ValueError("Sentence-transition fixture differs from the frozen reviewed scope")
    # Equality alone accepts bool == 1; retain explicit numeric schema types.
    if type(data["schema"]) is not int or type(data["steps"]) is not int or type(data["speed"]) not in (int, float):
        raise ValueError("Invalid fixture numeric types")
    if not callable(chunker):
        raise ValueError("An explicit pinned-package text chunker is required")
    internal_calls = 0
    for row in data["samples"]:
        if type(row["expected_chunks"]) is not int or not 0 < len(row["text"]) <= 1000:
            raise ValueError("Invalid sample bounds")
        chunks = chunker(row["text"], data["max_chunk_length"])
        if (not isinstance(chunks, (list, tuple)) or len(chunks) != row["expected_chunks"]
                or any(not isinstance(chunk, str) or not chunk.strip() for chunk in chunks)
                or " ".join(chunks) != row["text"]):
            raise ValueError(f"Unexpected chunking or lost text in {row['id']}")
        internal_calls += len(chunks)
    if internal_calls > MAX_INTERNAL_CALLS:
        raise ValueError("Internal synthesis call budget exceeded")
    return deepcopy(data)


def plan_jobs(data, smoke, *, chunker):
    if type(smoke) is not bool:
        raise ValueError("smoke must be Boolean")
    data = validate_fixture(data, chunker=chunker)
    return data["samples"][:1] if smoke else data["samples"][1:]


def _integer(value, label, *, positive=False):
    if type(value) is not int or value < 0 or (positive and value == 0):
        raise ValueError(f"{label} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def _settings(sample_rate, gap_seconds):
    if type(sample_rate) is not int or sample_rate != 44100:
        raise ValueError("This probe requires a 44100 Hz sample rate")
    if isinstance(gap_seconds, bool) or not isinstance(gap_seconds, (int, float)):
        raise ValueError("Invalid synthetic gap")
    try:
        gap_seconds = float(gap_seconds)
    except OverflowError as exc:
        raise ValueError("Invalid synthetic gap") from exc
    if not math.isfinite(gap_seconds) or not 0 <= gap_seconds <= 10:
        raise ValueError("Synthetic gap must be finite and in [0, 10]")
    frames = round(gap_seconds * sample_rate)
    if abs(frames / sample_rate - gap_seconds) > 1e-9:
        raise ValueError("Synthetic gap must be representable as whole sample frames")
    return frames


def _clips(clips, sample_rate):
    if not isinstance(clips, (list, tuple)) or not 1 <= len(clips) <= 6:
        raise ValueError("Expected one to six measured clips")
    ids = set()
    for row in clips:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip() or row["id"] in ids:
            raise ValueError("Clips require unique nonempty IDs")
        ids.add(row["id"])
        count = _integer(row.get("native_frames"), "native_frames", positive=True)
        if count > MAX_CLIP_SECONDS * sample_rate:
            raise ValueError("Native clip exceeds 90 seconds")
        if "playback_rate" in row and (type(row["playback_rate"]) not in (int, float) or row["playback_rate"] != 1.0):
            raise ValueError("Playback must remain exactly 1.0")
    if sum(row["native_frames"] for row in clips) > MAX_STAGE_SECONDS * sample_rate:
        raise ValueError("Native clips exceed the 300-second stage bound")


def check_schedule(clips, *, sample_rate=44100, gap_seconds=.4, total_frames=None):
    """Reject truncated placement, overlap or missing synthetic silence.

    ``native_frames`` comes from the complete raw waveform, not a predicted
    model duration. Returned count evidence is not a waveform-content hash.
    """
    gap = _settings(sample_rate, gap_seconds)
    _clips(clips, sample_rate)
    if total_frames is not None:
        _integer(total_frames, "total_frames", positive=True)
    failures = []
    previous_end = None
    for row in clips:
        start = _integer(row.get("start_frame"), "start_frame")
        end = _integer(row.get("end_frame"), "end_frame")
        if end - start != row["native_frames"]:
            failures.append({"id": row["id"], "reason": "sample_count_changed"})
        if previous_end is not None and start < previous_end + gap:
            failures.append({"id": row["id"], "reason": "collision_or_missing_fixture_gap",
                             "required_start_frame": previous_end + gap, "actual_start_frame": start})
        if total_frames is not None and end > total_frames:
            failures.append({"id": row["id"], "reason": "outside_timeline"})
        previous_end = end
    return {"status": "blocked" if failures else "ready", "failures": failures,
            "schedule": [] if failures else deepcopy(list(clips)), "sample_rate": sample_rate,
            "gap_frames": gap, "playback_rate": 1.0, "evidence_kind": "synthetic_fixture",
            "sample_count_retention_verified": not failures,
            "sample_content_retention_verified": False, "pronunciation_verified": False,
            "naturalness_verified": False}


def plan_sequence(clips, *, sample_rate=44100, gap_seconds=.4, starts=None, slot_ends=None):
    """Place full native waveforms or block; no fitting, fades or dropped samples.

    Without explicit starts, append each raw clip after the full preceding clip
    and fixture gap. Explicit slot ends are deadlines, never trimming commands.
    """
    gap = _settings(sample_rate, gap_seconds)
    _clips(clips, sample_rate)
    for values, label in ((starts, "starts"), (slot_ends, "slot_ends")):
        if values is not None and (not isinstance(values, (list, tuple)) or len(values) != len(clips)):
            raise ValueError(f"{label} must match clip count")
    schedule, failures, cursor = [], [], 0
    for index, original in enumerate(clips):
        start = cursor if starts is None else _integer(starts[index], "start_frame")
        end = start + original["native_frames"]
        if slot_ends is not None:
            deadline = _integer(slot_ends[index], "slot_end_frame")
            if end > deadline:
                failures.append({"id": original["id"], "reason": "slot_too_short",
                                 "required_end_frame": end, "slot_end_frame": deadline,
                                 "deficit_frames": end - deadline})
        schedule.append({**deepcopy(original), "start_frame": start, "end_frame": end})
        cursor = end + gap
    result = check_schedule(schedule, sample_rate=sample_rate, gap_seconds=gap_seconds)
    if failures:
        result.update(status="blocked", schedule=[], sample_count_retention_verified=False,
                      failures=failures + result["failures"])
    result["total_frames"] = max(row["end_frame"] for row in schedule)
    return result
