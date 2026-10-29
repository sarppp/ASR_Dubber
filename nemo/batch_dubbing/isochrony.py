"""Compare speech activity at original coordinates; no models or audio editing.

Activity labels are supplied evidence, not inferred truth. This module does not
turn energy thresholds, ASR word gaps, or subtitle windows into speech evidence.
One timeline describes one speaker/channel; compare overlapping speakers
separately. Matching measurements never certifies meaning or natural delivery.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real

MAX_INTERVALS = 100_000
MAX_FRAMES = 1_000_000
NUMERICAL_SLACK_SECONDS = 1e-9
ACOUSTIC_PROVENANCE = frozenset({"acoustic_vad", "manual_review"})
PROVENANCE = ACOUSTIC_PROVENANCE | {"synthetic", "asr_words", "subtitle", "energy_threshold"}


def _number(value, name, minimum=0.0, maximum=7200.0):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True)
class SpeechInterval:
    start_seconds: float
    end_seconds: float

    def __post_init__(self):
        start = _number(self.start_seconds, "start_seconds")
        end = _number(self.end_seconds, "end_seconds")
        if end <= start:
            raise ValueError("Speech intervals must have positive duration")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)


@dataclass(frozen=True)
class ActivityTimeline:
    duration_seconds: float
    speech: tuple[SpeechInterval, ...]
    provenance: str
    resolution_seconds: float
    evidence_id: str

    def __post_init__(self):
        duration = _number(self.duration_seconds, "duration_seconds")
        resolution = _number(self.resolution_seconds, "resolution_seconds")
        if duration <= 0 or not 0 < resolution <= duration:
            raise ValueError("Positive duration and measurement resolution required")
        if not isinstance(self.provenance, str) or self.provenance not in PROVENANCE:
            raise ValueError("Explicit supported measurement provenance required")
        if (not isinstance(self.evidence_id, str) or not self.evidence_id.strip()
                or len(self.evidence_id) > 1024):
            raise ValueError("Nonempty bounded evidence_id required")
        if not isinstance(self.speech, (tuple, list)) or len(self.speech) > MAX_INTERVALS:
            raise ValueError("Speech must be a bounded sequence of intervals")
        canonical = []
        for span in self.speech:
            if not isinstance(span, SpeechInterval) or span.end_seconds > duration:
                raise ValueError("Speech interval outside timeline or invalid type")
            if canonical and span.start_seconds < canonical[-1].end_seconds:
                raise ValueError("Speech intervals must be ordered and nonoverlapping")
            if canonical and span.start_seconds == canonical[-1].end_seconds:
                canonical[-1] = SpeechInterval(canonical[-1].start_seconds, span.end_seconds)
            else:
                canonical.append(span)
        object.__setattr__(self, "duration_seconds", duration)
        object.__setattr__(self, "resolution_seconds", resolution)
        object.__setattr__(self, "speech", tuple(canonical))

    def to_dict(self):
        return asdict(self)


def timeline_from_mask(mask, *, duration_seconds, frame_seconds, provenance, evidence_id):
    """Convert supplied Boolean activity labels, retaining a partial last frame.

    No smoothing, merging across silence, or automatic activity detection occurs.
    The caller must supply every frame; truncated or surplus labels are rejected.
    """
    duration = _number(duration_seconds, "duration_seconds")
    frame = _number(frame_seconds, "frame_seconds")
    if duration <= 0 or not 0 < frame <= duration:
        raise ValueError("Positive duration and frame width required")
    if not isinstance(mask, (tuple, list)) or not 1 <= len(mask) <= MAX_FRAMES:
        raise ValueError("Activity mask must be a bounded sequence")
    # Avoid an extra frame from division rounding at an exact frame boundary.
    expected = math.ceil(math.nextafter(duration / frame, -math.inf))
    if len(mask) != expected or any(type(active) is not bool for active in mask):
        raise ValueError("Mask must contain one Boolean per frame")
    intervals, start = [], None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index * frame
        if start is not None and (not active or index == len(mask) - 1):
            end = index * frame if not active else duration
            intervals.append(SpeechInterval(start, end))
            start = None
    return ActivityTimeline(duration, tuple(intervals), provenance, frame, evidence_id)


def _length(intervals):
    return math.fsum(end - start for start, end in intervals)


def _intersection(left, right):
    i = j = 0
    pieces = []
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            pieces.append(end - start)
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return math.fsum(pieces)


def _pauses(timeline):
    # Keep zero-length edge pauses so paired diagnostics cannot shift identity.
    pauses, previous = [], 0.0
    for span in timeline.speech:
        pauses.append((previous, span.start_seconds))
        previous = span.end_seconds
    pauses.append((previous, timeline.duration_seconds))
    return pauses


def _paired_errors(source, candidate):
    if len(source) != len(candidate):
        return None  # Never invent correspondence across missing/extra blocks.
    return [{"index": index, "source_start": a, "source_end": b,
             "candidate_start": c, "candidate_end": d,
             "onset_error_seconds": c - a, "offset_error_seconds": d - b,
             "duration_error_seconds": (d - c) - (b - a)}
            for index, ((a, b), (c, d)) in enumerate(zip(source, candidate))]


def compare_timelines(source, candidate, *, playback_rate=1.0,
                      waveform_time_stretched=False, speech_trimmed=False,
                      tolerance_seconds=0.0):
    """Return timing diagnostics without repairing, shifting, or publishing audio.

    Default tolerance is zero (apart from 1 ns floating-point slack). Declared
    measurement resolution is reported, NEVER silently added to tolerance.
    Nonzero tolerance is a caller-selected evaluation policy, not exact matching.
    Playback/processing declarations must describe the completed candidate, not
    merely the final player setting. Their truth requires adapter-side auditing.
    """
    if not isinstance(source, ActivityTimeline) or not isinstance(candidate, ActivityTimeline):
        raise ValueError("Validated source and candidate ActivityTimeline required")
    rate = _number(playback_rate, "playback_rate", 0, 100)
    tolerance = _number(tolerance_seconds, "tolerance_seconds", 0, 1)
    if type(waveform_time_stretched) is not bool or type(speech_trimmed) is not bool:
        raise ValueError("Processing declarations must be Boolean")
    left = [(s.start_seconds, s.end_seconds) for s in source.speech]
    right = [(s.start_seconds, s.end_seconds) for s in candidate.speech]
    left_pauses, right_pauses = _pauses(source), _pauses(candidate)
    source_speech, candidate_speech = _length(left), _length(right)
    intersection = _intersection(left, right)
    union = source_speech + candidate_speech - intersection
    pause_intersection = _intersection(left_pauses, right_pauses)
    pause_union = _length(left_pauses) + _length(right_pauses) - pause_intersection
    speech_errors = _paired_errors(left, right)
    pause_errors = _paired_errors(left_pauses, right_pauses)
    duration_error = candidate.duration_seconds - source.duration_seconds
    errors = [duration_error]
    for rows in (speech_errors, pause_errors):
        if rows is not None:
            errors.extend(row[key] for row in rows for key in
                          ("onset_error_seconds", "offset_error_seconds", "duration_error_seconds"))
    reasons = []
    if rate != 1.0:
        reasons.append("playback_rate_not_one")
    if waveform_time_stretched:
        reasons.append("waveform_time_stretched")
    if speech_trimmed:
        reasons.append("speech_trimmed")
    if len(left) != len(right):
        reasons.append("speech_block_count_mismatch")
    if abs(duration_error) > tolerance + NUMERICAL_SLACK_SECONDS:
        reasons.append("timeline_duration_mismatch")
    if any(abs(error) > tolerance + NUMERICAL_SLACK_SECONDS for error in errors[1:]):
        reasons.append("speech_or_pause_boundary_mismatch")
    exact = (len(left) == len(right)
             and all(abs(error) <= NUMERICAL_SLACK_SECONDS for error in errors))
    acoustic = source.provenance in ACOUSTIC_PROVENANCE and candidate.provenance in ACOUSTIC_PROVENANCE
    synthetic = source.provenance == candidate.provenance == "synthetic"
    status = ("mismatch" if reasons else "matches_measurements" if acoustic
              else "synthetic_fixture" if synthetic else "insufficient_evidence")
    return {
        "policy": "isochrony-v1", "status": status,
        "timing_matches_policy": not reasons, "exact_measured_match": exact,
        "reasons": reasons,
        "evidence_sufficient_for_speech_comparison": acoustic,
        "contains_source_speech": bool(left), "contains_candidate_speech": bool(right),
        "source": source.to_dict(), "candidate": candidate.to_dict(),
        "processing": {"playback_rate": rate, "waveform_time_stretched": waveform_time_stretched,
                       "speech_trimmed": speech_trimmed, "declarations_independently_verified": False},
        "tolerance_seconds": tolerance,
        "numerical_slack_seconds": NUMERICAL_SLACK_SECONDS,
        "measurement_resolution_seconds": {"source": source.resolution_seconds,
                                           "candidate": candidate.resolution_seconds},
        "metrics": {"timeline_duration_error_seconds": duration_error,
                    "source_speech_seconds": source_speech, "candidate_speech_seconds": candidate_speech,
                    "speech_duration_error_seconds": candidate_speech - source_speech,
                    "missing_speech_seconds": max(0.0, source_speech - intersection),
                    "extra_speech_seconds": max(0.0, candidate_speech - intersection),
                    "speech_iou": intersection / union if union > 0 else 1.0,
                    "pause_iou": pause_intersection / pause_union if pause_union > 0 else 1.0},
        "speech_errors": speech_errors, "pause_errors": pause_errors,
        "correspondence": "chronological_only" if speech_errors is not None else "unmatched_block_counts",
        "quality_limit": "Timing measurements do not verify translation, intelligibility, identity, or naturalness.",
    }
