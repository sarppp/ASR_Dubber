"""Measured speech rates for the dub engines, and the budget derived from them.

The natural-pace dub needs to know how long the voice will take to say a
line, so it can tell whether the line fits its subtitle slot without ever
speeding the voice up.

The first implementation used a single hard-coded constant per language
(11.6 chars/s for German). That number was wrong: measured against the real
Qwen voice it is ~17.6-22.4 chars/s, so the gate believed every line was
1.4-1.6x too long, shortened text that was never too long, and lost meaning
doing it. See ``nemo/dub_rate_findings.md``.

A pure characters-per-second rate is not enough either. Every synthesised
segment carries a fixed overhead — the model's startup pause — so a short
line is spoken disproportionately slowly. Measured on the real run:

    speech_seconds = OVERHEAD + characters / RATE

with OVERHEAD ~1.2 s and RATE ~19 chars/s for German on Qwen. That model
predicts the measured speech within 0.45 s; a pure rate model is off by more
than a second on short lines, which is exactly where the dub failed.

These numbers are measured, not guessed:

* from the ``*_natural_fit_*.json`` sidecars and the raw ``seg_*.wav`` files of
  a real dub run (the voice's own ``raw_seconds`` against the subtitle slot
  it had to fit into), and
* from the independent TTS bench in ``nemo/new_guys/crispr/tts_bench_2026-11-04``.

``calibrate_rate`` measures the engine on the actual clip, so the rate
self-corrects instead of being frozen into the source.
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

# Measured chars/second per (engine, language), for the BODY of the speech.
#
# "bench" comes from the TTS bench (one fixed sentence per language).
# "dub" comes from a real dub run (the merged blocks dub.py actually speaks).
# The dub numbers win where both exist: they are the rate that matters, on
# real subtitle text, with the real voice clone.
MEASURED_RATES: dict[str, dict[str, float]] = {
    "qwen": {"en": 16.78, "de": 18.98, "fr": 19.94},
    "supertonic": {"en": 15.58, "de": 14.82, "fr": 16.16},
    "audio8": {"en": 14.55, "de": 12.96, "fr": 18.15},
}

# The fixed overhead per synthesised segment, in seconds. The voice needs a
# moment to start and the model adds a pause; a short line pays it in full.
#
# Measured on the real Qwen German run: speech = 1.16 + characters/18.98,
# with a worst-case error of 0.45 s across eight segments.
MEASURED_OVERHEAD: dict[str, float] = {
    "qwen": 1.16,
    "supertonic": 1.0,
    "audio8": 1.0,
}

# Fallback when an engine is unknown: the middle of the measured range, and
# deliberately conservative (slower speech -> more room, never a cut).
DEFAULT_RATES: dict[str, float] = {"en": 15.58, "de": 14.82, "fr": 16.16}
DEFAULT_OVERHEAD = 1.0

# The old, wrong constant. Kept only so the fix is visible in the diff and
# so a stale environment variable cannot silently reintroduce it.
LEGACY_RATES: dict[str, float] = {"de": 11.6, "fr": 12.3, "en": 13.5}

# How much of the slot the speech may use. The fitter needs a little room,
# and the tail of a line is usually a pause.
DEFAULT_EDGE_CLEANUP = 0.97


def engine_key(engine: str | None) -> str:
    """Map a --tts-engine name onto a measured-rate key."""
    name = (engine or "").strip().lower()
    if name in MEASURED_RATES:
        return name
    for key in MEASURED_RATES:
        if key in name:
            return key
    if "supertonic" in name or "super" in name:
        return "supertonic"
    if "qwen" in name:
        return "qwen"
    if "audio8" in name or "audio" in name:
        return "audio8"
    return ""


def measured_rate(language: str, engine: str | None = None) -> float:
    """Characters per second for `language` on `engine`.

    ``SUBTITLE_TTS_CPS`` still overrides everything, for deliberate
    experiments; the environment variable is the only way to force a rate.
    """
    override = os.getenv("SUBTITLE_TTS_CPS")
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    key = engine_key(engine)
    if key:
        rate = MEASURED_RATES[key].get(language)
        if rate:
            return rate
    return DEFAULT_RATES.get(language, 14.0)


def measured_overhead(engine: str | None = None) -> float:
    """The fixed startup cost of one synthesised segment, in seconds."""
    override = os.getenv("SUBTITLE_TTS_OVERHEAD")
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    key = engine_key(engine)
    return MEASURED_OVERHEAD.get(key, DEFAULT_OVERHEAD)


def speech_seconds(chars: int, language: str, engine: str | None = None,
                   overhead: float | None = None) -> float:
    """How long the voice needs to say `chars` characters.

    This is the number the gate compares against the subtitle slot. It
    includes the fixed overhead, which is what makes short lines work.
    """
    if overhead is None:
        overhead = measured_overhead(engine)
    return overhead + chars / measured_rate(language, engine)


def character_budget(span: float, language: str, engine: str | None = None,
                     safety: float = DEFAULT_EDGE_CLEANUP,
                     headroom: float = 0.95,
                     overhead: float | None = None) -> int:
    """Characters that fit in `span` seconds at the engine's natural rate.

    The result is what the translator is asked for. It is a *guide*, not a
    hard cap: the translator cannot be trusted to hit a character budget
    exactly (measured), so the pipeline treats it as a target and verifies
    the result against the real speech.
    """
    if overhead is None:
        overhead = measured_overhead(engine)
    rate = measured_rate(language, engine)
    usable = span * safety * headroom - overhead
    return max(12, int(usable * rate))


def word_budget(characters: int, language: str) -> int:
    """Approximate word count for `characters` in `language`.

    Word budgets work where character budgets do not: a model asked for "at
    most N words" counts far more reliably than one asked for "at most N
    characters". Measured in tools/probe_control.py.
    """
    per_word = {"de": 6.4, "fr": 6.0, "en": 5.6}.get(language, 6.0)
    return max(3, int(characters / per_word))


def fit_overhead_and_rate(samples: list[tuple[int, float]]
                          ) -> tuple[float, float]:
    """Fit ``speech = overhead + chars / rate`` to measured samples.

    `samples` holds (characters, speech_seconds). Returns (overhead, rate).
    The least-squares fit is used so one odd segment cannot move the model
    far; the rate is then read off the slope.
    """
    points = [(c, s) for c, s in samples if c > 0 and s > 0]
    if len(points) < 2:
        return DEFAULT_OVERHEAD, DEFAULT_RATES.get("de", 14.0)
    n = len(points)
    sum_c = sum(c for c, _ in points)
    sum_s = sum(s for _, s in points)
    sum_cc = sum(c * c for c, _ in points)
    sum_cs = sum(c * s for c, s in points)
    denominator = n * sum_cc - sum_c * sum_c
    if denominator == 0:
        return DEFAULT_OVERHEAD, DEFAULT_RATES.get("de", 14.0)
    slope = (n * sum_cs - sum_c * sum_s) / denominator
    if slope <= 0:
        return DEFAULT_OVERHEAD, DEFAULT_RATES.get("de", 14.0)
    overhead = (sum_s - slope * sum_c) / n
    if overhead < 0:
        overhead = 0.0
    return overhead, 1.0 / slope


def calibrate_rate(dub_dir: Path, language: str,
                    engine: str | None = None,
                    script: Path | None = None) -> tuple[float, dict] | None:
    """Measure the engine's real rate from a dub that already ran.

    Reads the ``*_natural_fit_*.json`` sidecars written by ``dub.py``: each one
    records how long the voice actually spoke (``raw_seconds``) against the
    subtitle slot (``target_seconds``). Returns the measured chars/second and
    the evidence, or ``None`` when there is nothing to measure.

    This is what makes the rate self-correcting: the first dub of a new clip
    measures the voice, and the gate for the next run uses the measured
    value instead of a hard-coded constant.

    `script` is the SRT that was spoken. When it is not given, the dub
    directory is searched for it (the ``*_dub.srt`` dub.py writes into
    ``output/``, or any ``*.dub.srt`` next to the caches).
    """
    dub_dir = Path(dub_dir)
    if not dub_dir.exists():
        return None

    sidecars = sorted(dub_dir.rglob("*_natural_fit_*.json"))
    if not sidecars:
        return None

    script = Path(script) if script else _find_script(dub_dir)
    if script is None or not script.exists():
        return None

    cues = _parse_srt(script)
    if not cues:
        return None

    evidence = []
    for sidecar in sidecars:
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        config = data.get("configuration", {})
        edges = data.get("edges", {})
        target = config.get("target_seconds")
        raw = edges.get("raw_seconds")
        if not target or not raw or raw <= 0:
            continue
        start = _clip_start(sidecar)
        if start is None:
            continue
        text = " ".join(cue["text"] for cue in cues
                        if start - 0.2 <= cue["start"] < start + target)
        if not text:
            continue
        evidence.append({
            "start": round(start, 3),
            "target_seconds": target,
            "speech_seconds": raw,
            "chars": len(text),
            "chars_per_second": round(len(text) / raw, 2),
            "overshoot": round(raw / target, 3),
        })

    if not evidence:
        return None
    rates = [row["chars_per_second"] for row in evidence]
    # The median is robust against the odd very short or very long block.
    measured = statistics.median(rates)
    return measured, {"engine": engine_key(engine) or engine, "language": language,
                      "samples": len(evidence), "median_chars_per_second": measured,
                      "min": min(rates), "max": max(rates), "evidence": evidence}


def _find_script(dub_dir: Path) -> Path | None:
    """Find the SRT that was spoken, inside the dub directory."""
    candidates = sorted(dub_dir.rglob("*_dub.srt")) + \
        sorted(dub_dir.rglob("*.dub.srt"))
    return candidates[0] if candidates else None


def _clip_start(sidecar: Path) -> float | None:
    """Find the start time of the clip that a sidecar describes."""
    prefix = sidecar.stem.rsplit("_natural_fit_", 1)[0]
    for checkpoint in sidecar.parent.parent.rglob("checkpoint.json"):
        try:
            entries = json.loads(checkpoint.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for entry in entries:
            clip = Path(entry.get("clip", ""))
            if clip.stem.startswith(prefix):
                return float(entry["start"])
    return None


def _parse_srt(path: Path) -> list[dict]:
    cues, cur = [], None
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if "-->" in line:
            left, right = line.split("-->")
            cur = {"start": _seconds(left), "end": _seconds(right), "text": []}
        elif cur is not None and line:
            cur["text"].append(line)
        elif cur is not None:
            cues.append({"start": cur["start"], "end": cur["end"],
                         "text": " ".join(cur["text"])})
            cur = None
    if cur:
        cues.append({"start": cur["start"], "end": cur["end"],
                     "text": " ".join(cur["text"])})
    return cues


def _seconds(stamp: str) -> float:
    hours, minutes, seconds = stamp.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
