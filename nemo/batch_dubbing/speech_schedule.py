"""Bounded scheduling for one explicitly aligned speaker turn; no audio IO.

The source intervals must describe corresponding speech, not proportional
subtitle display slots. Callers must attest that mapping for every clip. These
experimental limits establish timing feasibility, not perceived naturalness.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

EPSILON = 1e-9
MAX_CLIPS = 10_000


def _finite(value, name, minimum=0.0):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return number


def schedule_turn(
    clips: Sequence[Mapping], *, corridor_start, corridor_end,
    max_rate=1.15, max_shift=0.25,
):
    """Schedule all clips or return a blocked turn with no partial schedule.

    Required clip fields: ``id``, ``speaker``, ``source_start``, ``source_end``,
    ``retained_duration``, ``alignment_verified=True``. Optional ``head_guard``
    and ``tail_guard`` are retained low-amplitude handles, in native seconds.
    Durations/guards must come from the audio that will actually be rendered.

    Waveforms remain wholly inside the explicit speaker corridor and never
    overlap. Audible starts may move at most ``max_shift`` relative to verified
    source starts. Audible ends cannot exceed source ends by ``max_shift``;
    early endings are allowed, since short speech is never slowed to fill space.
    Adjacent audible intervals retain min(source gap, 100 ms) of silence. Retained
    handles count toward that silence rather than adding it a second time.

    Native speed is tried first. Otherwise binary search finds the smallest
    common tempo for the whole turn, at most 1.15. Nothing is trimmed or padded.
    ``ready`` means only that the timing constraints are feasible. A blocked
    turn contains bounded conflict diagnostics, not partially usable placements.
    """
    corridor_start = _finite(corridor_start, "corridor_start")
    corridor_end = _finite(corridor_end, "corridor_end")
    max_rate = _finite(max_rate, "max_rate", 1.0)
    max_shift = _finite(max_shift, "max_shift")
    if corridor_end <= corridor_start or max_rate > 1.15 or max_shift > 0.25:
        raise ValueError("Invalid corridor or policy exceeds experimental bounds")
    if not isinstance(clips, Sequence) or isinstance(clips, (str, bytes)):
        raise ValueError("clips must be a bounded sequence")
    if len(clips) > MAX_CLIPS:
        raise ValueError("Too many clips in one speaker turn")
    rows, identifiers, speaker = [], set(), None
    for clip in clips:
        if not isinstance(clip, Mapping) or clip.get("alignment_verified") is not True:
            raise ValueError("Each clip needs an explicitly verified source alignment")
        ident, voice = clip.get("id"), clip.get("speaker")
        if not isinstance(ident, str) or not ident.strip() or ident in identifiers:
            raise ValueError("Clip IDs must be unique nonempty strings")
        if not isinstance(voice, str) or not voice.strip():
            raise ValueError("Explicit speaker required")
        if speaker is not None and voice != speaker:
            raise ValueError("A turn cannot contain multiple speakers")
        speaker = voice
        identifiers.add(ident)
        row = {"id": ident, "speaker": voice}
        for key in ("source_start", "source_end", "retained_duration"):
            row[key] = _finite(clip.get(key), key)
        for key in ("head_guard", "tail_guard"):
            row[key] = _finite(clip.get(key, 0), key)
        if row["source_end"] <= row["source_start"]:
            raise ValueError("Source intervals must have positive duration")
        if row["retained_duration"] <= row["head_guard"] + row["tail_guard"]:
            raise ValueError("Retained duration must exceed the guard sum")
        if rows and row["source_start"] < rows[-1]["source_end"]:
            raise ValueError("Source intervals must be chronological and nonoverlapping")
        rows.append(row)

    summary = {"corridor_start": corridor_start, "corridor_end": corridor_end,
               "max_rate": max_rate, "max_shift": max_shift, "speaker": speaker}

    def constraints(rate):
        lower, upper, separation = [], [], []
        for i, row in enumerate(rows):
            duration = row["retained_duration"] / rate
            head = row["head_guard"] / rate
            tail = row["tail_guard"] / rate
            # Variables are audible onsets, which makes feasibility monotonic
            # as the common rate increases, including clips with retained guards.
            lower.append(max(row["source_start"] - max_shift, corridor_start + head))
            upper.append(min(row["source_start"] + max_shift,
                             row["source_end"] + max_shift - (duration - head - tail),
                             corridor_end - (duration - head)))
            if i:
                previous = rows[i - 1]
                gap = min(row["source_start"] - previous["source_end"], 0.10)
                previous_after_onset = (previous["retained_duration"] - previous["head_guard"]) / rate
                separation.append(max(previous_after_onset + head,
                                      previous_after_onset - previous["tail_guard"] / rate + gap))
        return lower, upper, separation

    def feasibility(rate, diagnostics=False):
        lower, upper, separation = constraints(rate)
        earliest, chain_start, first, worst = None, 0, None, None
        for i, row in enumerate(rows):
            previous_bound = earliest + separation[i - 1] if i else -math.inf
            if lower[i] >= previous_bound:
                earliest, chain_start = lower[i], i
            else:
                earliest = previous_bound
            deficit = earliest - upper[i]
            if deficit > EPSILON:
                if not diagnostics:
                    return False
                witness = {"first_index": chain_start, "last_index": i,
                           "earliest_audible_start": earliest,
                           "latest_allowed_audible_start": upper[i],
                           "deficit_seconds": deficit}
                if first is None:
                    first = witness
                if worst is None or deficit > worst["deficit_seconds"]:
                    worst = witness
        if not diagnostics:
            return True
        conflicts = []
        for witness in (first, worst):
            if witness is not None and witness not in conflicts:
                conflicts.append(witness)
        for witness in conflicts:
            witness["clip_ids"] = [rows[i]["id"] for i in range(witness["first_index"], witness["last_index"] + 1)]
        return conflicts

    if feasibility(1.0):
        rate = 1.0
    elif not feasibility(max_rate):
        return {**summary, "status": "blocked", "rate": None, "schedule": [],
                "diagnostics": {"tested_rate": max_rate, "conflicts": feasibility(max_rate, True)}}
    else:
        low, high = 1.0, max_rate
        for _ in range(48):
            mid = (low + high) / 2
            if feasibility(mid):
                high = mid
            else:
                low = mid
        rate = high

    lower, upper, separation = constraints(rate)
    latest = upper.copy()
    for i in range(len(rows) - 2, -1, -1):
        latest[i] = min(latest[i], latest[i + 1] - separation[i])
    schedule, previous_onset = [], None
    for i, row in enumerate(rows):
        minimum = max(lower[i], previous_onset + separation[i - 1] if i else lower[i])
        onset = min(max(row["source_start"], minimum), latest[i])
        duration = row["retained_duration"] / rate
        head, tail = row["head_guard"] / rate, row["tail_guard"] / rate
        end = onset + duration - head
        audible_end = end - tail
        schedule.append({**row, "rate": rate, "start": onset - head, "end": end,
                         "audible_start": onset, "audible_end": audible_end,
                         "start_shift": onset - row["source_start"],
                         "end_shift": audible_end - row["source_end"]})
        previous_onset = onset
    return {**summary, "status": "ready", "rate": rate, "schedule": schedule,
            "diagnostics": {"native_speed": rate == 1.0, "conflicts": []}}
