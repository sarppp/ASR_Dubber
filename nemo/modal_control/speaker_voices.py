"""Select stable Supertonic presets from measured speaker voice pitch.

This module deliberately has no model or Modal side effects. The caller must
collect reliable voiced samples for each diarized speaker and review uncertain
assignments before synthesis.
"""
from __future__ import annotations

from statistics import median


def assign_voices(speaker_pitches: dict[str, list[float]], *,
                  low_hz: float = 150.0, high_hz: float = 180.0,
                  min_samples: int = 10) -> dict[str, dict]:
    """Map speakers to stable presets; do not guess in an ambiguous range.

    Pitch is a crude vocal presentation heuristic, not a person's gender.
    The input contains voiced F0 estimates in hertz, already filtered for
    quality by the caller. Uncertain speakers have no assigned preset.
    """
    import math

    if not 0 < low_hz < high_hz or min_samples < 1:
        raise ValueError("invalid pitch thresholds or sample count")
    result = {}
    counts = {"F": 0, "M": 0}
    for speaker, samples in speaker_pitches.items():
        values = [float(x) for x in samples if math.isfinite(float(x)) and 50 <= float(x) <= 400]
        pitch = median(values) if values else None
        if len(values) < min_samples or pitch is None or low_hz <= pitch <= high_hz:
            result[speaker] = {"voice": None, "median_f0_hz": pitch, "voiced_samples": len(values), "status": "needs_review"}
            continue
        prefix = "M" if pitch < low_hz else "F"
        counts[prefix] += 1
        result[speaker] = {"voice": f"{prefix}{(counts[prefix] - 1) % 5 + 1}",
                           "median_f0_hz": pitch, "voiced_samples": len(values), "status": "estimated"}
    return result
