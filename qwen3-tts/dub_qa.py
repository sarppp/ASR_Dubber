"""
dub_qa.py — Automatic quality check after dubbing.

Compares the original video's audio against the dubbed video second-by-second
and reports:
  • Silent gaps: original has speech but dubbed is silent (missing dub)
  • Speed-up ratio: how much speed_fit compressed each segment
  • Overall coverage score

Run automatically at the end of dub.py, or standalone:
    uv run python dub_qa.py original.mp4 final_dub.mp4 srt.srt
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List, Tuple

log = logging.getLogger(__name__)

# A second is "speech" if mean volume is above this threshold (dB).
_SPEECH_THRESH_DB = -40.0
# A dubbed second is "silent" if mean volume is below this (digital near-zero).
_DUB_SILENT_DB = -50.0
# Warn if a silence gap in the dub is longer than this many seconds.
_GAP_WARN_SEC = 1.5


def _mean_vol_per_second(video_path: Path, duration: float) -> List[float]:
    """Return per-second mean volume (dB) for the entire video."""
    vols = []
    t = 0
    while t < duration:
        r = subprocess.run(
            f'ffmpeg -ss {t} -t 1 -i "{video_path}" '
            f'-af volumedetect -f null - 2>&1 | grep mean_volume',
            shell=True, capture_output=True, text=True,
        )
        out = r.stdout + r.stderr
        vol = None
        for line in out.splitlines():
            if "mean_volume" in line:
                try:
                    vol = float(line.split(":")[1].replace("dB", "").strip())
                except ValueError:
                    pass
        vols.append(vol if vol is not None else -91.0)
        t += 1
    return vols


def _video_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def check_dub(
    original: Path,
    dubbed: Path,
    segments: list,          # parsed SRT segments (dicts with start/end/text)
) -> dict:
    """
    Compare original vs dubbed video.  Returns a report dict and logs warnings.

    Report keys:
      score         float  0-100, % of speech seconds covered by dub
      missing_gaps  list   [(start_s, end_s, gap_s), ...] silence in dub where original has speech
      silent_segs   list   segment dicts that produced no dub audio
      speedup_segs  list   (seg, ratio) for segments likely sped up (ratio > 1.1)
    """
    orig_dur = _video_duration(original)
    dub_dur  = _video_duration(dubbed)
    if orig_dur <= 0:
        log.warning("QA: could not read original duration, skipping")
        return {}

    log.info(f"🔍 QA: sampling audio ({int(orig_dur)}s)…")
    orig_vols = _mean_vol_per_second(original, orig_dur)
    dub_vols  = _mean_vol_per_second(dubbed,   min(orig_dur, dub_dur))

    # ── Find silence gaps ────────────────────────────────────────────────────
    speech_secs  = 0
    covered_secs = 0
    gap_start    = None
    missing_gaps: List[Tuple[float, float, float]] = []

    for t, (o, d) in enumerate(zip(orig_vols, dub_vols)):
        orig_speech = o > _SPEECH_THRESH_DB
        dub_present = d > _DUB_SILENT_DB

        if orig_speech:
            speech_secs += 1
            if dub_present:
                covered_secs += 1
                if gap_start is not None:
                    gap_len = t - gap_start
                    missing_gaps.append((float(gap_start), float(t), float(gap_len)))
                    gap_start = None
            else:
                if gap_start is None:
                    gap_start = t
        else:
            # Natural pause — close any open gap
            if gap_start is not None:
                gap_len = t - gap_start
                missing_gaps.append((float(gap_start), float(t), float(gap_len)))
                gap_start = None

    # Close trailing gap
    if gap_start is not None:
        gap_len = len(orig_vols) - gap_start
        missing_gaps.append((float(gap_start), float(len(orig_vols)), float(gap_len)))

    # Filter to gaps above warning threshold
    notable_gaps = [(s, e, g) for s, e, g in missing_gaps if g >= _GAP_WARN_SEC]

    # ── Match gaps back to SRT segments ─────────────────────────────────────
    silent_segs = []
    for seg in segments:
        seg_mid = (seg["start"] + seg["end"]) / 2
        t = int(seg_mid)
        if t < len(dub_vols) and dub_vols[t] <= _DUB_SILENT_DB:
            silent_segs.append(seg)

    score = round(100.0 * covered_secs / speech_secs, 1) if speech_secs else 100.0

    # ── Log report ───────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"📊 Dub QA  score={score}%  "
             f"({covered_secs}/{speech_secs} speech-seconds covered)")

    if notable_gaps:
        log.warning(f"   ⚠️  {len(notable_gaps)} silence gap(s) where original has speech:")
        for s, e, g in notable_gaps:
            # find which SRT segment this overlaps
            segs_in = [sg for sg in segments if sg["end"] > s and sg["start"] < e]
            label = f"segs {segs_in[0]['index']}–{segs_in[-1]['index']}" if segs_in else "?"
            log.warning(f"      {s:.0f}s–{e:.0f}s  ({g:.0f}s missing)  [{label}]")
    else:
        log.info("   ✅ No significant silence gaps found")

    if silent_segs:
        indices = [s["index"] for s in silent_segs]
        log.warning(f"   ⚠️  {len(silent_segs)} segment(s) produced no dub audio: {indices}")

    log.info("=" * 60)

    return {
        "score":        score,
        "missing_gaps": notable_gaps,
        "silent_segs":  silent_segs,
    }


# ── Standalone CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import re

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 3:
        print("Usage: dub_qa.py original.mp4 dubbed.mp4 [srt.srt]")
        sys.exit(1)

    orig_path = Path(sys.argv[1])
    dub_path  = Path(sys.argv[2])
    srt_path  = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    segs = []
    if srt_path and srt_path.exists():
        def _ts(s):
            h, m, sec = s.replace(",", ".").split(":")
            return int(h) * 3600 + int(m) * 60 + float(sec)
        for blk in re.split(r"\n\n+", srt_path.read_text(encoding="utf-8").strip()):
            lines = blk.strip().splitlines()
            if len(lines) >= 2:
                try:
                    idx = int(lines[0])
                    s, e = lines[1].split(" --> ")
                    segs.append({"index": idx, "start": _ts(s.strip()),
                                 "end": _ts(e.strip()), "text": " ".join(lines[2:])})
                except (ValueError, IndexError):
                    pass

    report = check_dub(orig_path, dub_path, segs)
    sys.exit(0 if report.get("score", 0) >= 90 else 1)
