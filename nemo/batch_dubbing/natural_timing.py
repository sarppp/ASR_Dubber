"""Experimental speech grouping and timing decisions; no synthesis or audio IO.

This module is deliberately disconnected from production workers. It describes
when already synthesized speech can fit conservatively, and when text needs
review. Thresholds are hypotheses for listening tests, not quality guarantees.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence


def _number(value, name, *, minimum=0.0):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def _cue(value):
    if not isinstance(value, Mapping):
        raise ValueError("cue must be a mapping")
    start = _number(value.get("start"), "start")
    end = _number(value.get("end"), "end")
    text = value.get("text")
    speaker = value.get("speaker")
    if end <= start or not isinstance(text, str) or not text.strip():
        raise ValueError("cue needs positive duration and nonempty text")
    if not isinstance(speaker, str) or not speaker.strip():
        raise ValueError("cue needs an explicit speaker")
    return {**value, "start": start, "end": end, "text": text.strip()}


def group_speech_cues(
    cues: Sequence[Mapping], *, max_seconds=12.0, max_characters=240,
    meaningful_gap=0.25,
):
    """Join adjacent fragments until punctuation, pause, speaker, or size bound.

    Input must be chronological, finite, nonoverlapping, and speaker-attributed.
    No individual cue is split. Oversized input cues receive ``oversized_cue``;
    fragments stopped by a size limit receive ``incomplete_phrase``. Every result
    contains ``source_indices`` and shallow copies of ``source_cues`` so original
    caption timing and text remain inspectable. Input objects are never changed.
    Terminal punctuation uses a simple heuristic; abbreviations need review.
    """
    max_seconds = _number(max_seconds, "max_seconds")
    meaningful_gap = _number(meaningful_gap, "meaningful_gap")
    if max_seconds == 0 or isinstance(max_characters, bool) or not isinstance(max_characters, int) or max_characters <= 0:
        raise ValueError("size limits must be positive")
    validated = [_cue(cue) for cue in cues]
    for previous, current in zip(validated, validated[1:]):
        if current["start"] < previous["end"]:
            raise ValueError("cues must be chronological and nonoverlapping")

    def complete(text):
        return bool(re.search(r'[.!?…][\"\'»›”’\)\]]*$', text.rstrip()))

    result = []
    for index, cue in enumerate(validated):
        previous = result[-1] if result else None
        merge = (
            previous is not None
            and previous["speaker"] == cue["speaker"]
            and not complete(previous["text"])
            and cue["start"] - previous["end"] < meaningful_gap
            and cue["end"] - previous["start"] <= max_seconds
            and len(previous["text"]) + len(cue["text"]) + 1 <= max_characters
        )
        if merge:
            previous["text"] += " " + cue["text"]
            previous["end"] = cue["end"]
            previous["source_indices"].append(index)
            previous["source_cues"].append(dict(cues[index]))
        else:
            result.append({"start": cue["start"], "end": cue["end"],
                           "speaker": cue["speaker"], "text": cue["text"],
                           "source_indices": [index], "source_cues": [dict(cues[index])],
                           "flags": []})
        if cue["end"] - cue["start"] > max_seconds or len(cue["text"]) > max_characters:
            result[-1]["flags"].append("oversized_cue")
    for phrase in result:
        if not complete(phrase["text"]):
            phrase["flags"].append("incomplete_phrase")
    return result


def plan_timing(
    phrase: Mapping, raw_duration: float, *, next_phrase: Mapping | None = None,
    video_duration: float | None = None, max_rate=1.15, max_borrow=0.15,
    minimum_pause=0.10, long_pause=0.50, underfill_ratio=0.65,
):
    """Return a decision for one measured clip, without transforming its audio.

    Never slow speech to fill a window or drop samples to meet a deadline.
    Borrow only from a short gap before the same speaker, leave a minimum pause,
    and borrow only as much as needed. First try native speed; then compress up
    to ``max_rate``. An infeasible clip returns ``needs_rewrite`` with no applied
    rate or fabricated output end. An underfilled slot returns ``review_underfilled``.
    Other review flags survive in ``flags`` even if timing status is ``ready``.
    ``max_rate`` is configurable, but borrowing can never exceed 0.15 seconds,
    the retained pause can never be below 0.10 seconds, or the eligible gap above
    0.50 seconds. These are experimental guardrails, not evidence of naturalness.
    """
    cue = _cue(phrase)
    raw_duration = _number(raw_duration, "raw_duration")
    max_rate = _number(max_rate, "max_rate", minimum=1.0)
    max_borrow = _number(max_borrow, "max_borrow")
    minimum_pause = _number(minimum_pause, "minimum_pause", minimum=0.10)
    long_pause = _number(long_pause, "long_pause")
    underfill_ratio = _number(underfill_ratio, "underfill_ratio")
    if raw_duration == 0 or max_borrow > 0.15 or long_pause > 0.50 or not 0 < underfill_ratio <= 1:
        raise ValueError("duration and policy thresholds exceed permitted bounds")
    following = _cue(next_phrase) if next_phrase is not None else None
    if following is not None and following["start"] < cue["end"]:
        raise ValueError("next phrase overlaps this phrase")
    if video_duration is not None:
        video_duration = _number(video_duration, "video_duration")
        if cue["end"] > video_duration or (following and following["end"] > video_duration):
            raise ValueError("phrase exceeds video bounds")

    slot = cue["end"] - cue["start"]
    allowed_borrow = 0.0
    if following is not None and following["speaker"] == cue["speaker"]:
        gap = following["start"] - cue["end"]
        if gap <= long_pause:
            allowed_borrow = min(max_borrow, max(0.0, gap - minimum_pause))
    available = slot + allowed_borrow
    required_rate = raw_duration / available
    flags = list(phrase.get("flags", []))
    decision = {"start": cue["start"], "original_end": cue["end"],
                "deadline": cue["end"] + allowed_borrow,
                "raw_duration": raw_duration, "max_rate": max_rate,
                "required_rate": max(1.0, required_rate), "flags": flags}
    if required_rate > max_rate + 1e-12:
        return {**decision, "status": "needs_rewrite", "rate": None,
                "output_end": None, "borrowed_seconds": 0.0,
                "overflow_at_max_rate": raw_duration / max_rate - available}
    rate = max(1.0, min(max_rate, required_rate))
    duration = raw_duration / rate
    borrowed = max(0.0, duration - slot)
    underfilled = raw_duration / slot < underfill_ratio
    if underfilled:
        flags.append("underfilled_slot")
    return {**decision, "status": "review_underfilled" if underfilled else "ready",
            "rate": rate, "output_end": cue["start"] + duration,
            "borrowed_seconds": borrowed, "overflow_at_max_rate": 0.0}
