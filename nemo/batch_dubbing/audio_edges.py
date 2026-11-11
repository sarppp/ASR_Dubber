"""Conservative, model-free decisions for removing generated audio edge padding.

This module never edits audio. Activity is the peak absolute sample in each
10 ms frame, not RMS, to retain quiet consonants. Interior silence is untouched.
An activity threshold is not a speech recognizer; retain raw clips for review.
"""
from __future__ import annotations

import math
from numbers import Real


def _finite(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite in [{minimum}, {maximum}]")
    return value


def analyze_edges(
    samples, sample_rate, *, threshold_db=-60.0, head_guard_seconds=0.08,
    tail_guard_seconds=0.12, max_edge_trim_seconds=1.0,
    min_removable_seconds=0.10,
):
    """Return a safe slice ``samples[start_sample:end_sample]`` without editing.

    ``samples`` must be a nonempty mono sequence of finite real sample values.
    Booleans and nested channels are rejected. ``sample_rate`` is a positive
    integer up to 384 kHz. Input amplitudes may exceed 1; no clipping is applied.

    First/last activity bounds refer to whole frames: the first active frame's
    start and last active frame's exclusive end. Frame width and guard lengths
    round up to samples, while the maximum removal rounds down. Each edge is
    capped separately at no more than one second. An edge is removed only if
    the resulting removal is at least ``min_removable_seconds``. Empty, silent,
    and nonfinite inputs fail rather than yielding fabricated speech bounds.
    """
    if (isinstance(sample_rate, bool) or not isinstance(sample_rate, int)
            or not 1 <= sample_rate <= 384000):
        raise ValueError("sample_rate must be an integer in [1, 384000]")
    threshold_db = _finite(threshold_db, "threshold_db", -120.0, -20.0)
    head_guard_seconds = _finite(head_guard_seconds, "head_guard_seconds", 0.0, 1.0)
    tail_guard_seconds = _finite(tail_guard_seconds, "tail_guard_seconds", 0.0, 1.0)
    max_edge_trim_seconds = _finite(max_edge_trim_seconds, "max_edge_trim_seconds", 0.0, 1.0)
    min_removable_seconds = _finite(min_removable_seconds, "min_removable_seconds", 0.0, 1.0)
    if isinstance(samples, (str, bytes, bytearray, dict)):
        raise ValueError("samples must be a mono numeric sequence")
    try:
        count = len(samples)
        iterator = iter(samples)
    except TypeError as exc:
        raise ValueError("samples must be a mono numeric sequence") from exc
    if count == 0:
        raise ValueError("Audio is empty")

    frame_samples = max(1, math.ceil(sample_rate * 0.01))
    threshold = 10.0 ** (threshold_db / 20.0)
    first_active = last_active_end = None
    peak, frame_peak, seen = 0.0, 0.0, 0
    for index, sample in enumerate(iterator):
        if isinstance(sample, bool) or not isinstance(sample, Real):
            raise ValueError("samples must contain finite mono real numbers")
        value = float(sample)
        if not math.isfinite(value):
            raise ValueError("Audio contains nonfinite samples")
        absolute = abs(value)
        peak = max(peak, absolute)
        frame_peak = max(frame_peak, absolute)
        seen += 1
        if seen % frame_samples == 0 or seen == count:
            if frame_peak >= threshold:
                start = (index // frame_samples) * frame_samples
                if first_active is None:
                    first_active = start
                last_active_end = seen
            frame_peak = 0.0
    if seen != count:
        raise ValueError("Audio sequence length changed during analysis")
    if first_active is None:
        raise ValueError("No audio activity above the configured threshold")

    head_guard = math.ceil(head_guard_seconds * sample_rate)
    tail_guard = math.ceil(tail_guard_seconds * sample_rate)
    maximum = math.floor(max_edge_trim_seconds * sample_rate)
    minimum = math.ceil(min_removable_seconds * sample_rate)
    wanted_head = max(0, first_active - head_guard)
    wanted_tail = max(0, count - last_active_end - tail_guard)
    head, tail = min(wanted_head, maximum), min(wanted_tail, maximum)
    flags = []
    if wanted_head > maximum:
        flags.append("leading_trim_capped")
    if wanted_tail > maximum:
        flags.append("trailing_trim_capped")
    if head < minimum:
        head = 0
    if tail < minimum:
        tail = 0
    end = count - tail
    assert 0 <= head <= first_active < last_active_end <= end <= count
    return {
        "start_sample": head, "end_sample": end,
        "first_active_sample": first_active, "last_active_end_sample": last_active_end,
        "input_samples": count, "retained_samples": end - head,
        "sample_rate": sample_rate, "frame_samples": frame_samples,
        "raw_seconds": count / sample_rate, "retained_seconds": (end - head) / sample_rate,
        "leading_removed_seconds": head / sample_rate,
        "trailing_removed_seconds": tail / sample_rate,
        "removed_seconds": (head + tail) / sample_rate,
        "threshold_db": threshold_db, "threshold_amplitude": threshold,
        "head_guard_seconds": head_guard_seconds, "tail_guard_seconds": tail_guard_seconds,
        "max_edge_trim_seconds": max_edge_trim_seconds,
        "min_removable_seconds": min_removable_seconds,
        "peak": peak, "flags": flags,
    }
