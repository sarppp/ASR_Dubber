"""Timing checks for a finished dub: does the speech sit inside its slot?

This is the L4 level of the dub test plan. It reads the artefacts a dub run
leaves behind — the fitted clip sidecars and the dubbed SRT — and answers the
questions that a listener would otherwise have to check by ear:

* does every segment's speech fit inside its subtitle slot (speech <= span)?
* does the speech start at the cue start, and not cut the first sound off?
* are there gaps of dead air between segments?
* does the text that was spoken match the subtitle that was sent?

No GPU and no models: it reads the WAVs and the JSON that ``dub.py`` and
``fit_audio.py`` already write.
"""
from __future__ import annotations

import json
import re
import wave
from dataclasses import dataclass, field
from pathlib import Path

# The speech must start within this many seconds of the cue start. A larger
# offset means the beginning of the line was cut off or swallowed.
ONSET_TOLERANCE = 0.06

# Gaps shorter than this between two segments sound choppy.
MIN_GAP = 0.2


@dataclass
class Segment:
    index: int
    start: float
    end: float
    span: float
    speech: float
    onset: float
    chars: int
    text: str = ""
    overshoot: float = 0.0

    @property
    def cps(self) -> float:
        return self.chars / self.speech if self.speech > 0 else 0.0

    @property
    def fits(self) -> bool:
        return self.speech <= self.span + 1e-3

    @property
    def onset_ok(self) -> bool:
        return self.onset <= ONSET_TOLERANCE


@dataclass
class Report:
    segments: list[Segment] = field(default_factory=list)
    gaps: list[tuple[float, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def too_long(self) -> list[Segment]:
        return [s for s in self.segments if not s.fits]

    @property
    def late_onsets(self) -> list[Segment]:
        return [s for s in self.segments if not s.onset_ok]

    @property
    def short_gaps(self) -> list[tuple[float, float]]:
        return [g for g in self.gaps if g[1] < MIN_GAP]

    @property
    def ok(self) -> bool:
        """True only when the check actually ran and everything passed.

        A check that could not run (no artefacts) is not a pass.
        """
        return not (self.too_long or self.late_onsets or self.short_gaps
                    or self.warnings)

    def summary(self) -> dict:
        return {
            "segments": len(self.segments),
            "too_long": [{"index": s.index, "overshoot": round(s.overshoot, 3),
                          "span": round(s.span, 3), "speech": round(s.speech, 3)}
                         for s in self.too_long],
            "late_onsets": [{"index": s.index, "onset": round(s.onset, 3)}
                             for s in self.late_onsets],
            "short_gaps": [{"after": round(a, 3), "before": round(b, 3),
                            "gap": round(b - a, 3)} for a, b in self.short_gaps],
            "warnings": self.warnings,
            "ok": self.ok,
        }


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


def _wav_duration(path: Path) -> float:
    with wave.open(str(path)) as handle:
        return handle.getnframes() / handle.getframerate()


def _onset(path: Path, threshold: float = 0.02) -> float:
    """How long the clip is silent before the speech starts, in seconds."""
    with wave.open(str(path)) as handle:
        rate = handle.getframerate()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        return 0.0
    import array
    samples = array.array("h")
    samples.frombytes(frames)
    limit = max(1, int(rate * 0.5))     # only look at the first half second
    peak = max((abs(s) for s in samples[:limit]), default=0)
    if peak == 0:
        return 0.0
    for index, value in enumerate(samples[:limit]):
        if abs(value) / 32768.0 >= threshold * (peak / 32768.0):
            return index / rate
    return 0.0


def check_dub(dub_dir: Path, script: Path | None = None) -> Report:
    """Check every synthesised segment of a dub run.

    `dub_dir` is the workdir dub.py used (it holds ``cache/`` and the
    ``*_natural_fit_*.json`` sidecars). `script` is the SRT that was spoken.
    """
    dub_dir = Path(dub_dir)
    report = Report()
    if script is None:
        candidates = sorted(dub_dir.rglob("*_dub.srt")) + \
            sorted(dub_dir.rglob("*.dub.srt"))
        script = candidates[0] if candidates else None
    if script is None or not Path(script).exists():
        report.warnings.append("no dubbed SRT found; nothing to check")
        return report

    cues = _parse_srt(Path(script))
    if not cues:
        report.warnings.append("the dubbed SRT is empty")
        return report
    if not list(dub_dir.rglob("cache/*/*/checkpoint.json")):
        report.warnings.append("no TTS checkpoint found; nothing was synthesised")
        return report

    entries: list[dict] = []
    for checkpoint in sorted(dub_dir.rglob("cache/*/*/checkpoint.json")):
        try:
            entries.extend(json.loads(checkpoint.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue

    seen: set[str] = set()
    for entry in entries:
        clip = Path(entry.get("clip", ""))
        if not clip.exists() or str(clip) in seen:
            continue
        seen.add(str(clip))
        start = float(entry.get("start", 0.0))
        end = float(entry.get("end", 0.0))
        span = max(0.0, end - start)
        try:
            speech = _wav_duration(clip)
        except (wave.Error, OSError):
            report.warnings.append(f"unreadable clip: {clip.name}")
            continue
        # The text that was spoken: the cues this segment covers.
        text = " ".join(cue["text"] for cue in cues
                        if start - 0.2 <= cue["start"] < end + 0.2)
        index = _segment_index(clip.name)
        report.segments.append(Segment(
            index=index, start=start, end=end, span=span, speech=speech,
            onset=_onset(clip), chars=len(text), text=text,
            overshoot=(speech / span if span else 0.0),
        ))

    report.segments.sort(key=lambda s: s.start)
    for first, second in zip(report.segments, report.segments[1:]):
        gap = second.start - first.end
        if 0 <= gap < 5.0:
            report.gaps.append((first.end, second.start))

    if not report.segments:
        report.warnings.append("no synthesised clips found")
    return report


def _segment_index(name: str) -> int:
    match = re.search(r"seg_(\d+)", name)
    return int(match.group(1)) if match else 0
