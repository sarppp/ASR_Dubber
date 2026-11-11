"""Assign translated cues to diarized speakers without inventing labels."""
from __future__ import annotations


def assign_cue_speakers(cues: list[dict], turns: list[dict], *, min_share: float = 0.6) -> list[dict]:
    """Return new cues with a speaker when one voice dominates their timing.

    Diarization overlaps and unknown speakers remain explicitly unresolved.
    A cue's text is never altered; the caller must resolve uncertain cues
    before generating a speaker-dependent dub.
    """
    if not 0.5 < min_share <= 1:
        raise ValueError("min_share must be greater than 0.5 and at most 1")
    result = []
    for cue in cues:
        start, end = float(cue['start']), float(cue['end'])
        if end <= start:
            raise ValueError("cue must have a positive duration")
        # Partition at turn boundaries so overlapping turns cannot be counted
        # twice. Time with multiple active speakers is deliberately unresolved.
        relevant = []
        boundaries = {start, end}
        for turn in turns:
            a, b = max(start, float(turn['start'])), min(end, float(turn['end']))
            if b > a:
                relevant.append((a, b, turn['speaker']))
                boundaries.update((a, b))
        weights: dict[str, float] = {}
        points = sorted(boundaries)
        for a, b in zip(points, points[1:]):
            active = {s for t0, t1, s in relevant if t0 < b and t1 > a and s}
            if len(active) == 1:
                speaker = next(iter(active))
                weights[speaker] = weights.get(speaker, 0.0) + b - a
        best = max(weights, key=weights.get) if weights else None
        share = weights[best] / (end - start) if best is not None else 0.0
        result.append({**cue, 'speaker': best if share >= min_share else None,
                       'speaker_coverage': round(share, 4),
                       'speaker_status': 'assigned' if share >= min_share else 'needs_review'})
    return result
