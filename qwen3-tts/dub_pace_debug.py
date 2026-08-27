#!/usr/bin/env python3
"""
dub_pace_debug.py — decisive diagnostic for dub speaking-pace problems
=====================================================================

Answers, for any dubbed video:

  1. Is the dub's pace inconsistent — "sometimes fast, sometimes slow"?
  2. If so, is that an artifact of the fitting pipeline, or is the dub
     faithfully tracking how the ORIGINAL speaker talked?
  3. Did a pipeline change actually fix it?  (--json writes a scorecard
     you can diff across videos / commits.)

It runs whatever checks the available inputs allow — more inputs, more checks.

INPUTS
------
  DIR                     a dub work-dir; auto-discovers the files below inside it
  --translated-srt PATH   the translated SRT fed to dub.py (e.g. *_fr.srt / *_de.srt)
  --temp-dir PATH         dub temp/ holding seg_XXXX.wav + seg_XXXX_fit.wav
  --original-srt PATH     the original-language diarized SRT
  --original-media PATH   the original video / audio file
  --dub-media PATH        final_dub.mp4  (delivered dub audio)
  --dub-srt PATH          final_dub.srt / *_dub.srt

  Legacy form still works:   dub_pace_debug.py TRANSLATED.srt TEMPDIR

CHECKS
------
  A  Translation-length pressure     needs: translated-srt
  B  speed_fit reconstruction        needs: translated-srt + temp-dir
  C  Delivered dub pace              needs: dub-media + dub-srt
  D  Original speaker pace           needs: original-media + original-srt
  E  DUB vs ORIGINAL (decisive)      needs: D  +  (B or C)
  V  Verdict + scorecard             always

Language-agnostic: any source/target pair. The headline metric is
syllables/second (estimated from vowel groups for Latin scripts, per-character
for CJK) — the ~8-9 syll/s ceiling on human articulation holds across languages,
whereas chars/second does not (German runs denser, CJK romanisations differ), so
c/s is shown only as a secondary readout. Check E normalises the dub and the
original to their own medians before comparing, so absolute per-language
calibration of the estimator cancels out.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── verdict thresholds (CLI-overridable) ─────────────────────────────────────
DEF = dict(
    natural_lo=0.90, natural_hi=1.15,   # applied-tempo band that "sounds untouched"
    stretched_frac_fail=0.35,           # >this share outside the band -> FAIL (B)
    jump=0.25, jump_frac_fail=0.12,     # adjacent |Δtempo| >= jump, >this share -> FAIL (B)
    sps_warn=7.0, sps_fail=8.5,         # syllables/sec: human comfort / hard ceiling
    sps_fail_count=3,                   # >this many segments over sps_warn -> FAIL
    cv_fail=0.18, cv_ratio_fail=1.4,    # dub pace CV, and dub_CV / orig_CV -> FAIL (E)
    corr_fail=0.30,                     # dub-vs-original pace correlation -> FAIL (E)
    pause_ratio_fail=1.20,              # dub speech-fraction / orig speech-fraction -> FAIL (E)
    pressure_warn=6.5, pressure_frac_warn=0.40,  # translation too long for slots (A)
    pressure_low=2.6,                   # slot demands < this syll/s -> translation far
                                        #   too short, segment WILL be stretched/padded (A)
    art_slow=3.2,                       # articulation rate (syll / voiced sec) below this
                                        #   sounds unnaturally slowed / drawled (C)
    fill_warn=0.62, fill_frac_fail=0.18,  # voiced/span below fill_warn = padded with
                                        #   silence; >this share of segments -> FAIL (C)
    tail_pad=1.2, tail_pad_frac_fail=0.15,  # trailing dead air per segment; share -> FAIL (C)
    stretch_slow=1.35, stretch_fast=0.78,  # dub voiced time / original voiced time for the
                                        #   same words; outside this band = pace changed (E)
    stretch_run=2,                      # this many breaching segments in a row -> FAIL (E),
                                        #   an audible slow/fast *patch*
    stretch_frac_fail=0.20,             # or this share of all segments breaching -> FAIL (E)
    noise_db=-35.0, min_sil=0.20,       # silencedetect params
)

# vowel letters across Latin-script dub languages (en/fr/de/es/it/pt/nl/…)
VOWELS = ("aeiouyàâäáãåæéèêëēïîíìóòôöõøœúùûüýÿ"
          "AEIOUYÀÂÄÁÃÅÆÉÈÊËĒÏÎÍÌÓÒÔÖÕØŒÚÙÛÜÝŸ")
CJK = r"぀-ヿ㐀-䶿一-鿿가-힯"


# ── SRT parsing / merging ───────────────────────────────────────────────────

def _ts(t: str) -> float:
    t = t.strip().replace(",", ".")
    h, m, s = t.split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)


def parse_srt(path: Path) -> List[Dict]:
    out: List[Dict] = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
        lines = [l for l in block.splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        # tolerate blocks with or without a leading numeric id
        if re.match(r"^\d+$", lines[0]):
            idx = int(lines[0]); rest = lines[1:]
        else:
            idx = len(out) + 1; rest = lines
        if not rest:
            continue
        m = re.match(r"([\d:,\.]+)\s*-->\s*([\d:,\.]+)", rest[0])
        if not m:
            continue
        raw = " ".join(rest[1:]).strip()
        sm = re.match(r"\[([^\]]+)\]\s*(.*)", raw, re.DOTALL)
        spk, text = (sm.group(1).strip(), sm.group(2).strip()) if sm else ("Speaker 1", raw)
        text = text.replace(" | ", " ").replace("|", " ").strip()
        if not text:
            continue
        out.append(dict(index=idx, start=_ts(m.group(1)), end=_ts(m.group(2)),
                        speaker=spk, text=text))
    return out


def merge_segments(segs: List[Dict], gap_sec: float, max_dur: float) -> List[Dict]:
    """Same merge rule dub.py / dub_srt.py uses."""
    if not segs:
        return segs
    hard_cap = max_dur * 2

    def done(t: str) -> bool:
        t = re.sub(r"\[Speaker\s+\d+\]\s*", "", t).strip().rstrip("\"'»›")
        return bool(t) and t[-1] in ".?!"

    out = [dict(segs[0], subsegments=[{"start": segs[0]["start"], "end": segs[0]["end"]}])]
    for seg in segs[1:]:
        cur = out[-1]
        gap = seg["start"] - cur["end"]
        md = seg["end"] - cur["start"]
        if (seg["speaker"] == cur["speaker"] and 0 <= gap <= gap_sec
                and (md <= max_dur or (not done(cur["text"]) and md <= hard_cap))):
            cur["end"] = seg["end"]
            cur["text"] = cur["text"].rstrip() + " " + seg["text"].lstrip()
            cur["subsegments"].append({"start": seg["start"], "end": seg["end"]})
        else:
            out.append(dict(seg, subsegments=[{"start": seg["start"], "end": seg["end"]}]))
    return out


# ── text metrics ────────────────────────────────────────────────────────────

def n_chars(t: str) -> int:
    return len(re.sub(r"\s+", "", t))


def n_syll(t: str) -> int:
    """Rough syllable / mora estimate, language-agnostic:
      - Latin scripts: count vowel groups (over-counts slightly; consistent)
      - CJK: one syllable per Han/Kana/Hangul character
    Used for relative comparison (dub vs original, each normalised to its own
    median) and as a conservative upper bound on articulation effort."""
    cjk = len(re.findall(f"[{CJK}]", t))
    latin = len(re.findall(f"[{VOWELS}]+", re.sub(f"[{CJK}]", " ", t)))
    return max(1, cjk + latin)


# ── audio ───────────────────────────────────────────────────────────────────

def ffprobe_dur(path: Path) -> float:
    try:
        return float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            stderr=subprocess.DEVNULL).strip())
    except Exception:
        return 0.0


def silence_intervals(media: Path, noise_db: float, min_sil: float) -> List[Tuple[float, float]]:
    """One ffmpeg pass -> list of (start,end) silence spans for the whole file."""
    r = subprocess.run(
        ["ffmpeg", "-i", str(media), "-af",
         f"silencedetect=noise={noise_db}dB:d={min_sil}", "-f", "null", "-"],
        capture_output=True, text=True)
    spans: List[Tuple[float, float]] = []
    start: Optional[float] = None
    for line in r.stderr.splitlines():
        a = re.search(r"silence_start:\s*(-?[\d.]+)", line)
        b = re.search(r"silence_end:\s*(-?[\d.]+)", line)
        if a:
            start = float(a.group(1))
        elif b and start is not None:
            spans.append((start, float(b.group(1))))
            start = None
    return spans


def voiced_seconds(spans: List[Tuple[float, float]], a: float, b: float) -> float:
    total = max(0.0, b - a)
    sil = 0.0
    for s0, s1 in spans:
        lo, hi = max(a, s0), min(b, s1)
        if hi > lo:
            sil += hi - lo
    return max(0.0, total - sil)


# ── small stats helpers ─────────────────────────────────────────────────────

def cv(xs: List[float]) -> float:
    xs = [x for x in xs if x > 0]
    if len(xs) < 2:
        return 0.0
    m = st.mean(xs)
    return st.pstdev(xs) / m if m else 0.0


def pearson(xs: List[float], ys: List[float]) -> float:
    pts = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pts) < 3:
        return float("nan")
    xs, ys = zip(*pts)
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in pts)
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def histo(xs: List[float], edges: List[float], label: str) -> None:
    n = len(xs)
    print(f"  {label}")
    for lo, hi in zip(edges, edges[1:]):
        c = sum(1 for x in xs if lo <= x < hi)
        bar = "#" * round(30 * c / n) if n else ""
        print(f"    {lo:6.2f}–{hi:<6.2f} {c:5d}  {bar:<30} {100*c/n if n else 0:4.1f}%")


# ── input discovery ─────────────────────────────────────────────────────────

def discover(d: Path, args) -> None:
    def pick(patterns, exclude=()):
        for pat in patterns:
            for p in sorted(d.glob(pat)):
                if any(x in p.name for x in exclude):
                    continue
                return p
        return None

    if not args.translated_srt:
        args.translated_srt = pick(["*.diarize_??.srt", "*_??.srt"],
                                   exclude=("_dub", "_clean"))
    if not args.original_srt:
        args.original_srt = pick(["*.nemo.??.diarize.srt", "*.diarize.srt"],
                                 exclude=("_dub", "_clean", "diarize_"))
    if not args.dub_srt:
        args.dub_srt = pick(["*_dub.srt", "final_dub.srt", "output/*_dub.srt",
                             "output/final_dub.srt"])
    if not args.dub_media:
        args.dub_media = pick(["final_dub.mp4", "output/final_dub.mp4", "*_dub.mp4"])
    if not args.temp_dir:
        for c in (d / "temp", *sorted(d.glob("*/temp")), *sorted(d.glob("output/*/temp"))):
            if c.is_dir() and list(c.glob("seg_*.wav")):
                args.temp_dir = c
                break


# ═══════════════════════════════════════════════════════════════════════════
#  CHECK A — translation-length pressure
# ═══════════════════════════════════════════════════════════════════════════

def check_A(merged: List[Dict], R: dict) -> dict:
    print("\n" + "═" * 74)
    print("A  TRANSLATION-LENGTH PRESSURE  (how fast each slot forces the dub to talk)")
    print("═" * 74)
    rows = []
    for s in merged:
        slot = max(0.1, s["end"] - s["start"])
        rows.append(dict(i=s["index"], slot=slot, ch=n_chars(s["text"]),
                         sy=n_syll(s["text"]), cps=n_chars(s["text"]) / slot,
                         sps=n_syll(s["text"]) / slot, text=s["text"]))
    cps = [r["cps"] for r in rows]
    sps = [r["sps"] for r in rows]
    over = [r for r in rows if r["sps"] > R["pressure_warn"]]
    under = [r for r in rows if r["sps"] < R["pressure_low"]]
    print(f"  segments: {len(rows)}")
    print(f"  chars/sec demanded : median {st.median(cps):4.1f}   p90 {_p(cps,90):4.1f}   max {max(cps):5.1f}")
    print(f"  syll/sec  demanded : median {st.median(sps):4.1f}   p10 {_p(sps,10):4.1f}   "
          f"p90 {_p(sps,90):4.1f}   max {max(sps):5.1f}")
    print(f"  slots too TIGHT  (> {R['pressure_warn']:.1f} syll/s, translation too long): "
          f"{len(over)}/{len(rows)}  ({100*len(over)/len(rows):.0f}%)")
    print(f"  slots too LOOSE  (< {R['pressure_low']:.1f} syll/s, translation far too short → "
          f"will be stretched/padded): {len(under)}/{len(rows)}  ({100*len(under)/len(rows):.0f}%)")
    histo(sps, [0, 2, 3, 4, 5, 6, 7, 8, 12], "syll/sec the slot demands:")
    if over:
        print("  tightest (dub forced to rush):")
        for r in sorted(rows, key=lambda r: -r["sps"])[:5]:
            print(f"    [{r['i']:04d}] slot {r['slot']:5.2f}s  need {r['sps']:4.1f} syll/s  {r['text'][:64]}")
    if under:
        print("  loosest (dub forced to drag/pad):")
        for r in sorted(rows, key=lambda r: r["sps"])[:5]:
            print(f"    [{r['i']:04d}] slot {r['slot']:5.2f}s  need {r['sps']:4.1f} syll/s  {r['text'][:64]}")
    return dict(n=len(rows), sps_median=st.median(sps), sps_p10=_p(sps, 10),
                sps_p90=_p(sps, 90), sps_max=max(sps),
                over_frac=len(over) / len(rows), under_frac=len(under) / len(rows))


def _p(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * q / 100
    lo = int(k)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (k - lo)


# ═══════════════════════════════════════════════════════════════════════════
#  CHECK B — speed_fit reconstruction (exact atempo the pipeline applied)
# ═══════════════════════════════════════════════════════════════════════════

def check_B(merged: List[Dict], temp: Path, R: dict, max_speed: float,
            min_speed: float) -> Optional[dict]:
    raws = {int(p.stem.split("_")[1]): p for p in temp.glob("seg_*.wav")
            if "_fit" not in p.stem and "_ns" not in p.stem and "_cmp" not in p.stem
            and "_sub" not in p.stem}
    if not raws:
        print("\nB  speed_fit reconstruction — skipped (no seg_*.wav in temp dir)")
        return None
    print("\n" + "═" * 74)
    print("B  SPEED_FIT RECONSTRUCTION  (the exact time-stretch applied per segment)")
    print("═" * 74)

    avail: Dict[int, float] = {}
    for i, seg in enumerate(merged):
        slot = seg["end"] - seg["start"]
        gap_end = merged[i + 1]["start"] if i + 1 < len(merged) else seg["end"]
        avail[seg["index"]] = min(gap_end - seg["start"], slot + 5.0)

    rows = []
    for seg in merged:
        i = seg["index"]
        if i not in raws:
            continue
        raw = ffprobe_dur(raws[i])
        if raw <= 0:
            continue
        slot = max(0.1, seg["end"] - seg["start"])
        av = max(0.1, avail.get(i, slot))
        fitp = raws[i].with_name(raws[i].stem + "_fit.wav")
        fit = ffprobe_dur(fitp) if fitp.exists() else 0.0
        if fit <= 0:
            continue
        # MEASURED raw→fit ratio — robust to whatever _do_fit/speed_fit does.
        # `raw` still carries the TTS edge silence that speed_fit trims, so a
        # ratio a little over 1.0 is just trimming, not speed.  Classify
        # conservatively; check C (delivered audio) is the authority on pace.
        applied = raw / fit
        if applied > max_speed + 0.15:
            branch = "CAP+trim (words cut)"          # >1.5×: words were cut
        elif applied > 1.15 and fit < slot:
            branch = "compressed"
        elif applied < 0.90:
            branch = "PAD/slow (stretched or padded to fill)"
        else:
            branch = "natural"
        tempo = 1.0 if branch == "natural" else applied
        rows.append(dict(i=i, start=seg["start"], slot=slot, avail=av, raw=raw,
                         target=fit, ratio=applied, applied=tempo, raw_fit=applied,
                         branch=branch, fit=fit,
                         sps_out=n_syll(seg["text"]) / fit, text=seg["text"]))
    if not rows:
        print("  no raw/fit pairs matched the SRT — skipped")
        return None

    ap = [r["applied"] for r in rows]
    outside = [x for x in ap if not (R["natural_lo"] <= x <= R["natural_hi"])]
    rs = sorted(rows, key=lambda r: r["start"])
    jumps = [abs(b["applied"] - a["applied"]) for a, b in zip(rs, rs[1:])]
    big = [j for j in jumps if j >= R["jump"]]
    branches = {b: sum(1 for r in rows if r["branch"] == b) for b in
                ("natural", "compressed", "CAP+trim (words cut)",
                 "PAD/slow (stretched or padded to fill)")}

    print(f"  segments with audio: {len(rows)}")
    print(f"  applied tempo   : min {min(ap):.2f}  median {st.median(ap):.2f}  "
          f"max {max(ap):.2f}  stdev {st.pstdev(ap):.2f}")
    print(f"  outside natural band [{R['natural_lo']}–{R['natural_hi']}]: "
          f"{len(outside)}/{len(rows)}  ({100*len(outside)/len(rows):.0f}%)")
    histo(ap, [0.60, 0.80, 0.90, 0.97, 1.03, 1.10, 1.20, 1.36, 9.9],
          "applied tempo (1.00 = untouched):")
    print(f"  branch hits: " + "  ".join(f"{k}={v}" for k, v in branches.items()))
    print(f"  adjacent |Δtempo| : median {st.median(jumps):.2f}  max {max(jumps):.2f}  "
          f"|  ≥{R['jump']}: {len(big)}/{len(jumps)}  ({100*len(big)/len(jumps):.0f}% of transitions)")
    print("\n  worst RUSHED (highest tempo):")
    for r in sorted(rows, key=lambda r: -r["applied"])[:6]:
        print(f"    [{r['i']:04d}] {r['start']:7.1f}s  tempo {r['applied']:.2f}  "
              f"raw {r['raw']:5.1f}→{r['target']:5.1f}s  {r['sps_out']:4.1f} syll/s  {r['branch']}")
        print(f"           {r['text'][:74]}")
    print("  worst DRAGGING (lowest tempo):")
    for r in sorted(rows, key=lambda r: r["applied"])[:6]:
        print(f"    [{r['i']:04d}] {r['start']:7.1f}s  tempo {r['applied']:.2f}  "
              f"raw {r['raw']:5.1f}→{r['target']:5.1f}s  {r['branch']}")
        print(f"           {r['text'][:74]}")
    print("  worst JUMPS (fast clip next to slow clip):")
    order = sorted(range(len(jumps)), key=lambda k: -jumps[k])[:6]
    for k in order:
        a, b = rs[k], rs[k + 1]
        print(f"    {a['start']:7.1f}s [{a['i']:04d}] {a['applied']:.2f}  →  "
              f"[{b['i']:04d}] {b['applied']:.2f}   (Δ {jumps[k]:.2f})")

    return dict(n=len(rows), applied_median=st.median(ap), applied_cv=cv(ap),
                outside_frac=len(outside) / len(rows),
                jump_frac=len(big) / len(jumps), jump_max=max(jumps),
                branch_cap=branches["CAP+trim (words cut)"],
                branch_pad=branches["PAD/slow (stretched or padded to fill)"],
                sps_out_max=max(r["sps_out"] for r in rows),
                sps_out_over=sum(1 for r in rows if r["sps_out"] > R["sps_warn"]),
                per_seg={r["i"]: dict(start=r["start"], target=r["target"],
                                      applied=r["applied"], sps_out=r["sps_out"])
                         for r in rows})


# ═══════════════════════════════════════════════════════════════════════════
#  CHECK C — delivered dub pace (measured from final_dub.mp4)
# ═══════════════════════════════════════════════════════════════════════════

def _regroup(blocks: List[Dict], merged: List[Dict]) -> List[Tuple[Dict, float, float]]:
    """Map the proportionally-split dub SRT blocks back onto the merged source
    segments (block boundaries inside a segment are synthetic; only the segment
    spans are real).  Returns (seg, dub_start, dub_end) per merged segment."""
    if not blocks:
        return []
    tot_b = sum(n_chars(b["text"]) for b in blocks) or 1
    tot_s = sum(n_chars(s["text"]) for s in merged) or 1
    scale = tot_b / tot_s
    out: List[Tuple[Dict, float, float]] = []
    bi = 0
    run = 0.0           # cumulative block-chars consumed
    cum = 0.0           # cumulative target (block-chars) up to and incl. this seg
    for si, seg in enumerate(merged):
        cum += n_chars(seg["text"]) * scale
        last = si == len(merged) - 1
        grp: List[Dict] = []
        while bi < len(blocks) and (last or not grp
                                    or run + n_chars(blocks[bi]["text"]) / 2 <= cum):
            grp.append(blocks[bi])
            run += n_chars(blocks[bi]["text"])
            bi += 1
        if grp:
            out.append((seg, grp[0]["start"], grp[-1]["end"]))
    return out


def check_C(dub_media: Path, dub_srt: Path, merged: List[Dict], R: dict,
           fa=None) -> Optional[dict]:
    print("\n" + "═" * 74)
    print("C  DELIVERED DUB PACE  (measured segment-by-segment from the dub audio)")
    print("═" * 74)
    dur = ffprobe_dur(dub_media)
    rows = []

    if fa and fa.ok:
        print("  source: forced alignment (word-level)")
        far = {r["index"]: r for r in fa.segments}
        for seg in merged:
            r = far.get(seg["index"])
            if not r or r["voiced"] < 0.3 or r["w_start"] is None:
                continue
            sy = n_syll(seg["text"])
            span = max(0.05, r["span"])
            v = r["voiced"]
            rows.append(dict(i=seg["index"], start=r["w_start"], span=span, voiced=v,
                             lead=0.0, tail=0.0, fill=min(1.0, v / span),
                             art=sy / v, eff=sy / span,
                             cps=n_chars(seg["text"]) / v, text=seg["text"]))
        spans = None
        speech_frac = sum(r["voiced"] for r in rows) / dur if dur else 0.0
    else:
        blocks = parse_srt(dub_srt)
        if not blocks:
            return None
        spans = silence_intervals(dub_media, R["noise_db"], R["min_sil"])
        for seg, s0, s1 in _regroup(blocks, merged):
            span = max(0.05, s1 - s0)
            v = voiced_seconds(spans, s0, s1)
            lead = next((min(e, s1) - s0 for st_, e in spans
                         if st_ <= s0 + 0.02 and e > s0), 0.0)
            tail = next((s1 - max(st_, s0) for st_, e in spans
                         if e >= s1 - 0.02 and st_ < s1), 0.0)
            lead, tail = max(0.0, lead), max(0.0, tail)
            if v < 0.3:
                continue
            sy = n_syll(seg["text"])
            rows.append(dict(i=seg["index"], start=s0, span=span, voiced=v,
                             lead=lead, tail=tail, fill=v / span,
                             art=sy / v, eff=sy / span,
                             cps=n_chars(seg["text"]) / v, text=seg["text"]))
        speech_frac = voiced_seconds(spans, 0, dur) / dur if dur else 0.0

    if not rows:
        print("  could not align dub SRT to segments — skipped")
        return None

    art = [r["art"] for r in rows]
    eff = [r["eff"] for r in rows]
    fill = [r["fill"] for r in rows]
    jumps = [abs(b["art"] - a["art"]) for a, b in zip(rows, rows[1:])]
    fast = [r for r in rows if r["art"] > R["sps_fail"]]
    slow = [r for r in rows if r["art"] < R["art_slow"]]
    padded = [r for r in rows if r["fill"] < R["fill_warn"] and r["span"] > 2.0]
    tailpad = [r for r in rows if r["tail"] > R["tail_pad"]]

    print(f"  segments measured: {len(rows)}   dub speech fills {100*speech_frac:.0f}% of the runtime")
    print(f"  articulation rate (syll / voiced sec) : median {st.median(art):4.1f}   "
          f"range {min(art):.1f}–{max(art):.1f}   CV {cv(art):.2f}")
    print(f"  as-experienced rate (syll / full span): median {st.median(eff):4.1f}   "
          f"range {min(eff):.1f}–{max(eff):.1f}   CV {cv(eff):.2f}")
    print(f"  segment fill (voiced / span)          : median {st.median(fill):.2f}   "
          f"range {min(fill):.2f}–{max(fill):.2f}")
    print(f"  segment-to-segment |Δ articulation|   : median {st.median(jumps):.1f}  max {max(jumps):.1f}")
    print(f"  too FAST  (> {R['sps_fail']} syll/s articulation) : {len(fast)}")
    print(f"  too SLOW  (< {R['art_slow']} syll/s articulation, drawled) : {len(slow)}")
    print(f"  PADDED    (fill < {R['fill_warn']:.2f}, silence stuffed into the slot) : {len(padded)}")
    print(f"  TRAILING DEAD AIR > {R['tail_pad']}s : {len(tailpad)} segments")
    histo(art, [0, 2, 3, 4, 5, 6, 7, 8, 12], "articulation syll/sec:")
    if slow or padded:
        seen = set()
        draggy = [r for r in sorted(slow + padded, key=lambda r: r["start"])
                  if not (r["i"] in seen or seen.add(r["i"]))]
        print("  draggy / padded segments:")
        for r in draggy[:8]:
            print(f"    [{r['i']:04d}] {_hms(r['start'])}  span {r['span']:4.1f}s  "
                  f"voiced {r['voiced']:4.1f}s  fill {r['fill']:.2f}  art {r['art']:4.1f}  "
                  f"tail-sil {r['tail']:4.1f}s   {r['text'][:44]}")
    if fast:
        print("  rushed segments:")
        for r in sorted(rows, key=lambda r: -r["art"])[:5]:
            print(f"    [{r['i']:04d}] {_hms(r['start'])}  art {r['art']:4.1f} syll/s  "
                  f"voiced {r['voiced']:4.1f}s   {r['text'][:44]}")
    # per-segment "sounds off" flag (drawled OR padded OR long tail dead-air),
    # then cluster into audible patches (≥3 flagged segments inside any 45 s window)
    for r in rows:
        r["bad"] = (r["art"] < R["art_slow"] or (r["fill"] < R["fill_warn"] and r["span"] > 2.0)
                    or r["tail"] > R["tail_pad"])
    patches = []
    for a in rows:
        win = [r for r in rows if a["start"] <= r["start"] < a["start"] + 45 and r["bad"]]
        if len(win) >= 3:
            patches.append((win[0]["start"], win[-1]["start"] + win[-1]["span"], len(win)))
    # merge overlapping patch windows
    merged_p = []
    for s0, s1, k in sorted(patches):
        if merged_p and s0 <= merged_p[-1][1]:
            merged_p[-1] = (merged_p[-1][0], max(merged_p[-1][1], s1),
                            max(merged_p[-1][2], k))
        else:
            merged_p.append((s0, s1, k))
    pad_total = sum(r["lead"] + r["tail"] for r in rows)
    if merged_p:
        print("  sustained draggy patches (≥3 off segments in 45s):")
        for s0, s1, k in merged_p:
            print(f"    {_hms(s0)}–{_hms(s1)}   {k} segments drawled/padded")
    print(f"  total leading+trailing dead air stuffed into segments: {pad_total:.0f}s "
          f"({100*pad_total/max(1,dur):.0f}% of runtime)")

    return dict(n=len(rows), art_median=st.median(art), art_cv=cv(art),
                fa_backed=bool(fa and fa.ok),
                art_max=max(art), art_min=min(art), eff_cv=cv(eff),
                fast=len(fast), slow=len(slow), padded=len(padded),
                tailpad=len(tailpad), padded_frac=len(padded) / len(rows),
                tailpad_frac=len(tailpad) / len(rows),
                bad_frac=sum(r["bad"] for r in rows) / len(rows),
                patches=[(round(s0, 1), round(s1, 1), k) for s0, s1, k in merged_p],
                pad_total=pad_total, pad_frac=pad_total / max(1, dur),
                impossible=len(fast), sps_max=max(art), sps_cv=cv(art),
                speech_frac=speech_frac, jump_median=st.median(jumps),
                per_seg={r["i"]: dict(start=r["start"], sps_out=r["art"],
                                      span=r["span"], voiced=r["voiced"],
                                      fill=r["fill"], tail=r["tail"], lead=r["lead"])
                         for r in rows})


def _hms(s: float) -> str:
    return f"{int(s // 60):d}:{s % 60:05.2f}"


# ═══════════════════════════════════════════════════════════════════════════
#  CHECK D — original speaker pace (the natural-pace baseline)
# ═══════════════════════════════════════════════════════════════════════════

def check_D(orig_media: Path, orig_srt: Path, R: dict, fa=None) -> Optional[dict]:
    segs = parse_srt(orig_srt)
    if not segs:
        return None
    print("\n" + "═" * 74)
    print("D  ORIGINAL SPEAKER PACE  (how fast the real person actually talks)")
    print("═" * 74)
    dur = ffprobe_dur(orig_media) or (segs[-1]["end"])

    if fa and fa.ok:
        # word-level forced alignment: exact speech timing, immune to music bed
        ref = "forced alignment (word-level, music-immune)"
        far = {r["index"]: r for r in fa.segments}
        rows = []
        for s in segs:
            r = far.get(s["index"])
            if not r or r["voiced"] < 0.2:
                continue
            v = r["voiced"]
            rows.append(dict(start=r["w_start"], end=r["w_end"], voiced=v,
                             sps=n_syll(s["text"]) / v, cps=n_chars(s["text"]) / v,
                             text=s["text"]))
        gaps = list(fa.pauses)
        srt_speech = sum(r["voiced"] for r in fa.segments)
        srt_gap = sum(b - a for a, b in gaps)
        bedded = False
        speech_frac = srt_speech / dur if dur else 0.0
    else:
        spans = silence_intervals(orig_media, R["noise_db"], R["min_sil"])
        ac_speech = voiced_seconds(spans, 0, dur) / dur if dur else 0.0
        # NeMo's diarized SRT is a VAD: spans = speech, gaps = silence.  Original
        # videos usually have a music/ambience bed, so acoustic silence on the
        # original is unreliable — detect that and fall back to the SRT.
        srt_speech = sum(s["end"] - s["start"] for s in segs)
        srt_gap = max(0.0, (segs[-1]["end"] - segs[0]["start"]) - srt_speech)
        bedded = ac_speech > 0.90 and srt_gap / max(1.0, dur) > 0.03
        ref = ("SRT (music/ambience bed on original — pass --align for word-level "
               "timing)" if bedded else "acoustic (voiced audio)")
        gaps = [(a["end"], b["start"]) for a, b in zip(segs, segs[1:])
                if b["start"] - a["end"] > 0.5]
        rows = []
        for s in segs:
            span = max(0.1, s["end"] - s["start"])
            v = span if bedded else voiced_seconds(spans, s["start"], s["end"])
            if v < 0.25:
                continue
            rows.append(dict(start=s["start"], end=s["end"], voiced=v,
                             sps=n_syll(s["text"]) / v, cps=n_chars(s["text"]) / v,
                             text=s["text"]))
        speech_frac = srt_speech / dur if (bedded and dur) else ac_speech

    sps = [r["sps"] for r in rows]
    cps = [r["cps"] for r in rows]
    print(f"  segments: {len(rows)}   speech reference: {ref}")
    print(f"  speaker talks {100*speech_frac:.0f}% of the runtime  "
          f"({len(gaps)} pauses, {srt_gap:.0f}s total)")
    print(f"  {'chars' } /sec : median {st.median(cps):4.1f}   range {min(cps):.1f}–{max(cps):.1f}")
    print(f"  syll/sec  : median {st.median(sps):4.1f}   range {min(sps):.1f}–{max(sps):.1f}"
          f"   CV {cv(sps):.2f}  <-- the natural variation to compare against")
    histo(sps, [0, 3, 4, 5, 6, 7, 8, 12], "syll/sec:")
    return dict(n=len(rows), sps_median=st.median(sps), sps_cv=cv(sps),
                sps_min=min(sps), speech_frac=speech_frac, bedded=bedded,
                segs=[(r["start"], r["end"], r["sps"]) for r in rows],
                gaps=gaps,
                vseg=[(r["start"], r["end"], r["voiced"], n_syll(r["text"])) for r in rows])


# ═══════════════════════════════════════════════════════════════════════════
#  CHECK E — DUB vs ORIGINAL  (the decisive comparison)
# ═══════════════════════════════════════════════════════════════════════════

def check_E(merged: List[Dict], D: dict, B: Optional[dict], C: Optional[dict],
            R: dict) -> Optional[dict]:
    if not D:
        return None
    print("\n" + "═" * 74)
    print("E  DUB vs ORIGINAL  (is the dub's fast/slow following the original, or invented?)")
    print("═" * 74)

    # original per-segment rate, sampled onto each merged translated segment window
    osegs = D["segs"]  # (start,end,sps)

    def orig_rate(a: float, b: float) -> float:
        num = den = 0.0
        for s0, s1, sp in osegs:
            lo, hi = max(a, s0), min(b, s1)
            if hi > lo:
                num += sp * (hi - lo)
                den += hi - lo
        return num / den if den else 0.0

    if C and C.get("fa_backed"):
        dub_seg = C["per_seg"]; dub_src = "forced-aligned dub audio (C)"
    elif B and B.get("per_seg"):
        dub_seg = B["per_seg"]; dub_src = "speed_fit target (B)"
    else:
        dub_seg = (C or {}).get("per_seg") or {}; dub_src = "measured dub audio (C)"
    if dub_seg:
        print(f"  dub per-segment pace source: {dub_src}")
    pairs = []  # (orig_sps, dub_sps, seg)
    for s in merged:
        o = orig_rate(s["start"], s["end"])
        ps = dub_seg.get(s["index"])
        d = ps["sps_out"] if ps else None
        if o > 0 and d and d > 0:
            pairs.append((o, d, s))

    if len(pairs) < 5:
        print("  not enough aligned segments for the per-segment comparison")
        method = "windowed"
    else:
        method = "per-segment"

    result: dict = dict(method=method)

    if method == "per-segment":
        orig = [p[0] for p in pairs]
        dub = [p[1] for p in pairs]
        om, dm = st.median(orig), st.median(dub)
        on = [x / om for x in orig]
        dn = [x / dm for x in dub]
        r = pearson(on, dn)
        print(f"  aligned segments: {len(pairs)}")
        print(f"  original pace : median {om:4.1f} syll/s   CV {cv(orig):.2f}")
        print(f"  dub pace      : median {dm:4.1f} syll/s   CV {cv(dub):.2f}")
        print(f"  dub CV / original CV      : {cv(dub)/cv(orig) if cv(orig) else float('nan'):.2f}  "
              f"(1.0 = dub is as steady as the speaker; >1 = dub adds wobble)")
        print(f"  correlation dub↔original  : r = {r:+.2f}  "
              f"(→1 dub tracks the speaker · →0 dub pace is unrelated · <0 inverted)")
        explained = (r * r) if r == r else 0.0
        print(f"  share of dub pace variation explained by the original performance: "
              f"{100*explained:.0f}%")
        print(f"  → {100*(1-explained):.0f}% of the wobble is introduced by the pipeline")
        # biggest mismatches
        mm = sorted(pairs, key=lambda p: -abs((p[1] / dm) - (p[0] / om)))[:6]
        print("\n  segments where dub pace disagrees most with the original:")
        for o, d, s in mm:
            tag = "dub RUSHES" if d / dm > o / om else "dub DRAGS"
            print(f"    [{s['index']:04d}] {s['start']:7.1f}s  original {o:4.1f}  dub {d:4.1f} syll/s"
                  f"   {tag}   {s['text'][:52]}")
        result.update(orig_cv=cv(orig), dub_cv=cv(dub), corr=r,
                      cv_ratio=cv(dub) / cv(orig) if cv(orig) else float("nan"),
                      explained=explained)
    else:
        # fall back to sliding-window comparison of C vs D
        if not C:
            print("  need either speed_fit reconstruction (B) or delivered pace (C)")
            return result
        dub_cv = C["sps_cv"]
        print(f"  original pace CV : {D['sps_cv']:.2f}")
        print(f"  dub pace CV      : {dub_cv:.2f}")
        print(f"  dub CV / original CV : "
              f"{dub_cv/D['sps_cv'] if D['sps_cv'] else float('nan'):.2f}")
        result.update(orig_cv=D["sps_cv"], dub_cv=dub_cv, corr=float("nan"),
                      cv_ratio=dub_cv / D["sps_cv"] if D["sps_cv"] else float("nan"),
                      explained=float("nan"))

    # ── content-stretch: same words, dub time vs original time ──────────────
    # This is what the ear catches: a run of segments where the dub takes much
    # longer (drawled + padded) or much less time than the original did.
    vseg = D.get("vseg") or []

    def orig_voiced(a: float, b: float) -> float:
        """original voiced seconds attributable to source window [a,b]."""
        tot = 0.0
        for s0, s1, v, _sy in vseg:
            lo, hi = max(a, s0), min(b, s1)
            if hi > lo and (s1 - s0) > 0:
                tot += v * (hi - lo) / (s1 - s0)
        return tot

    cps = (C or {}).get("per_seg") or {}
    bps = (B or {}).get("per_seg") or {}
    stretch_rows = []
    for s in merged:
        ov = orig_voiced(s["start"], s["end"])
        if ov < 0.4:
            continue
        cs = cps.get(s["index"])
        if cs:                       # measured dub audio: use voiced + span
            dv, span = cs["voiced"], cs["span"]
        elif s["index"] in bps:      # only speed_fit target available
            dv, span = None, bps[s["index"]]["span"]
        else:
            continue
        speech_ratio = (dv / ov) if dv else None
        slot_ratio = span / ov
        stretch_rows.append(dict(seg=s, ov=ov, dv=dv, span=span,
                                 speech_ratio=speech_ratio, slot_ratio=slot_ratio))

    if stretch_rows:
        key = "speech_ratio" if stretch_rows[0]["speech_ratio"] is not None else "slot_ratio"
        lab = ("dub speech time / original speech time"
               if key == "speech_ratio" else "dub slot / original speech time")
        vals = [r[key] for r in stretch_rows]
        breach = []
        for r in stretch_rows:
            x = r[key]
            if x > R["stretch_slow"]:
                r["tag"] = "DRAG"; breach.append(r)
            elif x < R["stretch_fast"]:
                r["tag"] = "RUSH"; breach.append(r)
            else:
                r["tag"] = ""
        # longest consecutive run of same-direction breaches
        run = best = 0
        cur_dir = None
        run_at = best_at = None
        for r in stretch_rows:
            if r["tag"] and r["tag"] == cur_dir:
                run += 1
            elif r["tag"]:
                run = 1; cur_dir = r["tag"]; run_at = r["seg"]["start"]
            else:
                run = 0; cur_dir = None
            if run > best:
                best = run; best_at = (run_at, r["seg"]["end"], cur_dir)
        print(f"\n  content-stretch  ({lab}):")
        print(f"    median {st.median(vals):.2f}   range {min(vals):.2f}–{max(vals):.2f}   "
              f"(1.00 = dub uses exactly the original's time)")
        print(f"    segments outside [{R['stretch_fast']}, {R['stretch_slow']}]: "
              f"{len(breach)}/{len(stretch_rows)}  ({100*len(breach)/len(stretch_rows):.0f}%)")
        print(f"    longest run in one direction: {best} segments"
              + (f"  →  {_hms(best_at[0])}–{_hms(best_at[1])}  ({best_at[2]})" if best_at else ""))
        for r in sorted(breach, key=lambda r: -abs(math.log(r[key])))[:8]:
            extra = ""
            if r["dv"]:
                extra = f"  (dub voiced {r['dv']:.1f}s vs original {r['ov']:.1f}s)"
            print(f"    [{r['seg']['index']:04d}] {_hms(r['seg']['start'])}  {r['tag']}  "
                  f"×{r[key]:.2f}{extra}  {r['seg']['text'][:44]}")
        result.update(stretch_median=st.median(vals),
                      stretch_breach_frac=len(breach) / len(stretch_rows),
                      stretch_run=best,
                      stretch_metric=key)

    # ── invented vs faithful silence ──────────────────────────────────────────
    # A dub segment's dead air is only a bug if the ORIGINAL didn't pause there.
    # Compare each dub segment's leading+trailing silence to the original's own
    # pause around the same content (from the original SRT / VAD).
    ogaps = D.get("gaps") or []
    cps2 = (C or {}).get("per_seg") or {}

    def orig_pause_near(a: float, b: float) -> float:
        tot = 0.0
        for g0, g1 in ogaps:
            lo, hi = max(a - 0.5, g0), min(b + 3.0, g1)
            if hi > lo:
                tot += hi - lo
        return tot

    invented = []
    faithful = 0
    for s in merged:
        cs = cps2.get(s["index"])
        if not cs:
            continue
        dub_sil = cs["lead"] + cs["tail"] if "lead" in cs else cs.get("tail", 0.0)
        op = orig_pause_near(s["start"], s["end"])
        extra = dub_sil - op - 1.0          # 1 s grace
        if dub_sil > R["tail_pad"]:
            if extra > R["tail_pad"]:
                invented.append(dict(seg=s, dub_sil=dub_sil, orig=op, extra=extra,
                                     start=cs["start"], span=cs["span"]))
            else:
                faithful += 1

    if cps2 and ogaps is not None:
        inv_total = sum(x["extra"] for x in invented)
        vid = merged[-1]["end"] if merged else 1.0
        print(f"\n  silence check (dub dead-air vs the original's own pauses):")
        print(f"    segments with real dead air: {len(invented) + faithful}   "
              f"faithful to an original pause: {faithful}   invented by the pipeline: {len(invented)}")
        print(f"    invented silence total: {inv_total:.0f}s "
              f"({100*inv_total/max(1.0, vid):.0f}% of the video)")
        for x in sorted(invented, key=lambda x: -x["extra"])[:6]:
            print(f"    [{x['seg']['index']:04d}] {_hms(x['start'])}  dub silence {x['dub_sil']:.1f}s "
                  f"vs original pause {x['orig']:.1f}s  → +{x['extra']:.1f}s invented   "
                  f"{x['seg']['text'][:40]}")
        # audible patch = ≥2 nearby segments that are BOTH drawled (stretch DRAG)
        # and/or carrying invented silence
        drag_idx = {r["seg"]["index"] for r in stretch_rows if r.get("tag") == "DRAG"} \
            if stretch_rows else set()
        inv_idx = {x["seg"]["index"] for x in invented}
        bad = sorted(({*drag_idx, *inv_idx}))
        starts = {s["index"]: cps2[s["index"]]["start"] for s in merged
                  if s["index"] in cps2}
        patch = []
        seq = sorted(i for i in bad if i in starts)
        cluster = []
        for i in seq:
            if cluster and starts[i] - starts[cluster[-1]] > 30:
                if len(cluster) >= 2:
                    patch.append((starts[cluster[0]], starts[cluster[-1]]))
                cluster = []
            cluster.append(i)
        if len(cluster) >= 2:
            patch.append((starts[cluster[0]], starts[cluster[-1]]))
        if patch:
            print("    audible slow/empty patches (drawled + invented silence together):")
            for p0, p1 in patch:
                print(f"      {_hms(p0)}–{_hms(p1)}")
        result.update(invented_pad_total=inv_total, invented_pad_n=len(invented),
                      faithful_pad_n=faithful,
                      patches=[(round(p0, 1), round(p1, 1)) for p0, p1 in patch])

    # pause / breathing budget
    df = (C or {}).get("speech_frac")
    if df is not None:
        ratio = df / D["speech_frac"] if D["speech_frac"] else float("nan")
        print(f"\n  breathing room: original speech = {100*D['speech_frac']:.0f}% of runtime, "
              f"dub speech = {100*df:.0f}%   (ratio {ratio:.2f})")
        if ratio > 1.05:
            print(f"  → the dub talks {100*(ratio-1):.0f}% more of the time than the original: "
                  f"pauses were squeezed out")
        result["pause_ratio"] = ratio
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  VERDICT
# ═══════════════════════════════════════════════════════════════════════════

def verdict(A, B, C, D, E, R) -> dict:
    print("\n" + "═" * 74)
    print("V  VERDICT")
    print("═" * 74)
    checks: List[Tuple[str, Optional[bool], str]] = []

    def add(name, ok, detail):
        checks.append((name, ok, detail))

    if B:
        add("segments time-stretched out of the natural band",
            B["outside_frac"] <= R["stretched_frac_fail"],
            f"{100*B['outside_frac']:.0f}% outside {R['natural_lo']}–{R['natural_hi']} "
            f"(fail > {100*R['stretched_frac_fail']:.0f}%)")
        add("abrupt segment-to-segment pace changes",
            B["jump_frac"] <= R["jump_frac_fail"],
            f"{100*B['jump_frac']:.0f}% of transitions jump ≥{R['jump']} tempo "
            f"(fail > {100*R['jump_frac_fail']:.0f}%)")
        add("segments compressed past the human ceiling",
            B["sps_out_over"] <= R["sps_fail_count"] and B["sps_out_max"] <= R["sps_fail"],
            f"{B['sps_out_over']} segs over {R['sps_warn']} syll/s, peak {B['sps_out_max']:.1f} "
            f"(fail > {R['sps_fail_count']} or peak > {R['sps_fail']})")
        add("segments slowed to the floor (draggy)",
            B["branch_pad"] <= max(2, 0.08 * B["n"]),
            f"{B['branch_pad']} segs hit the PAD/min-speed floor "
            f"(fail > {max(2, round(0.08 * B['n']))})")
    if C:
        add("segments articulated past the human ceiling",
            C["fast"] == 0,
            f"{C['fast']} segments over {R['sps_fail']} syll/s, peak {C['art_max']:.1f}")
        add("segments drawled below a natural rate",
            C["slow"] <= max(1, round(0.08 * C["n"])),
            f"{C['slow']}/{C['n']} segments under {R['art_slow']} syll/s articulation "
            f"(fail > {max(1, round(0.08 * C['n']))})")
        add("segments padded with silence",
            C["padded_frac"] <= R["fill_frac_fail"],
            f"{C['padded']}/{C['n']} segments fill < {R['fill_warn']:.2f} of their slot "
            f"({100*C['padded_frac']:.0f}%, fail > {100*R['fill_frac_fail']:.0f}%)")
        if not (E and "invented_pad_total" in E):
            add("no sustained draggy patch",
                not C["patches"],
                (f"draggy patches at " +
                 "; ".join(f"{_hms(s0)}–{_hms(s1)}" for s0, s1, _ in C["patches"]))
                if C["patches"] else "none found (≥3 drawled/padded segments in any 45s window)")
            add("dead-air padding within segments is small",
                C["pad_frac"] <= 0.10,
                f"{C['pad_total']:.0f}s of leading/trailing silence stuffed into segments "
                f"({100*C['pad_frac']:.0f}% of runtime, fail > 10%)")
    if E and "invented_pad_total" in E:
        add("dub does not invent pauses the original never had",
            E["invented_pad_n"] <= 2 and E["invented_pad_total"] <= 6.0,
            f"{E['invented_pad_n']} segments carry pauses absent from the original, "
            f"{E['invented_pad_total']:.0f}s total ({E['faithful_pad_n']} pauses are faithful)")
        add("no sustained slow/empty patch vs the original",
            not E["patches"],
            ("patches at " + "; ".join(f"{_hms(p0)}–{_hms(p1)}" for p0, p1 in E["patches"]))
            if E["patches"] else "none (drawled + invented-silence segments don't cluster)")
    if E and "stretch_run" in E:
        add("no sustained slow/fast patch vs the original",
            E["stretch_run"] < R["stretch_run"]
            and E["stretch_breach_frac"] <= R["stretch_frac_fail"],
            f"longest one-direction run {E['stretch_run']} segs "
            f"(fail ≥ {R['stretch_run']}); {100*E['stretch_breach_frac']:.0f}% of segments "
            f"outside [{R['stretch_fast']},{R['stretch_slow']}]× the original's time "
            f"(fail > {100*R['stretch_frac_fail']:.0f}%)")
    if E and E.get("dub_cv") is not None and E.get("orig_cv"):
        cvr = E["cv_ratio"]
        add("dub pace steadier than / as steady as the speaker",
            not (E["dub_cv"] > R["cv_fail"] and cvr > R["cv_ratio_fail"]),
            f"dub CV {E['dub_cv']:.2f} vs original {E['orig_cv']:.2f} (ratio {cvr:.2f}); "
            f"fail if dub CV > {R['cv_fail']} and ratio > {R['cv_ratio_fail']}")
    if E and E.get("corr") == E.get("corr") and E.get("method") == "per-segment":
        add("dub speed changes follow the original speaker",
            E["corr"] >= R["corr_fail"],
            f"correlation r = {E['corr']:+.2f} (fail < {R['corr_fail']:+.2f})")
    if E and "pause_ratio" in E:
        add("dub keeps the original's pauses / breathing room",
            E["pause_ratio"] <= R["pause_ratio_fail"],
            f"dub speaks {E['pause_ratio']:.2f}× as much of the runtime "
            f"(fail > {R['pause_ratio_fail']:.2f})")
    if A:
        add("translation short enough for the slots (upstream)",
            A["over_frac"] <= R["pressure_frac_warn"],
            f"{100*A['over_frac']:.0f}% of slots need > {R['pressure_warn']} syll/s "
            f"(warn > {100*R['pressure_frac_warn']:.0f}%)")
        add("slots not far larger than the translation (upstream)",
            A["under_frac"] <= R["pressure_frac_warn"],
            f"{100*A['under_frac']:.0f}% of slots demand < {R['pressure_low']} syll/s — "
            f"translation far too short, segment gets stretched/padded "
            f"(warn > {100*R['pressure_frac_warn']:.0f}%)")

    fails = [c for c in checks if c[1] is False]
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}")
        print(f"         {detail}")

    names = {c[0] for c in fails}
    severe = bool(names & {"segments articulated past the human ceiling",
                           "no sustained draggy patch",
                           "no sustained slow/empty patch vs the original",
                           "no sustained slow/fast patch vs the original"})
    if not checks:
        overall = "INSUFFICIENT DATA"
    elif len(fails) == 0:
        overall = "PASS — dub pace looks natural and consistent"
    elif len(fails) == 1 and not severe:
        overall = "WARN — 1 problem, pace is slightly off in places"
    else:
        overall = (f"FAIL — {len(fails)} problem{'s' if len(fails) != 1 else ''}; the dub "
                   f"does not track the original's rhythm (fast/slow or draggy patches audible)")
    print("\n  " + "─" * 70)
    print(f"  {overall}")
    print("  " + "─" * 70)
    return dict(overall=overall, n_fail=len(fails),
                checks=[dict(name=n, pass_=o, detail=d) for n, o, d in checks])


# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pos", nargs="*", help="DIR  |  legacy: TRANSLATED.srt TEMPDIR")
    ap.add_argument("--translated-srt")
    ap.add_argument("--temp-dir")
    ap.add_argument("--original-srt")
    ap.add_argument("--original-media")
    ap.add_argument("--dub-media")
    ap.add_argument("--dub-srt")
    ap.add_argument("--max-speed", type=float, default=1.35)
    ap.add_argument("--min-speed", type=float, default=0.65)
    ap.add_argument("--merge-gap", type=float, default=1.0)
    ap.add_argument("--merge-max-dur", type=float, default=10.0)
    ap.add_argument("--noise-db", type=float, default=DEF["noise_db"])
    ap.add_argument("--align", dest="align", action="store_true", default=None,
                    help="force word-level forced alignment (torchaudio MMS_FA) as "
                         "the speech-timing reference — music-immune. Default: auto "
                         "(on when torchaudio + an SRT + media are available).")
    ap.add_argument("--no-align", dest="align", action="store_false")
    ap.add_argument("--align-device", default="auto", help="cuda | cpu | auto")
    ap.add_argument("--json", help="write a scorecard JSON here (for tracking across videos)")
    for k, v in DEF.items():
        if k in ("noise_db",):
            continue
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = ap.parse_args()

    R = {k: getattr(args, k) for k in DEF}
    R["noise_db"] = args.noise_db

    # positionals
    if len(args.pos) == 2 and args.pos[0].endswith(".srt"):
        args.translated_srt = args.translated_srt or args.pos[0]
        args.temp_dir = args.temp_dir or args.pos[1]
    elif len(args.pos) == 1:
        p = Path(args.pos[0])
        if p.is_dir():
            discover(p, args)
        elif p.suffix == ".srt":
            args.translated_srt = args.translated_srt or str(p)

    for a in ("translated_srt", "temp_dir", "original_srt", "original_media",
              "dub_media", "dub_srt"):
        v = getattr(args, a)
        if v:
            setattr(args, a, Path(v))

    print("INPUTS")
    for a in ("translated_srt", "temp_dir", "original_srt", "original_media",
              "dub_media", "dub_srt"):
        v = getattr(args, a)
        ok = v and Path(v).exists()
        print(f"  {a:16s}: {v if v else '—'}   {'' if ok else '(missing)' if v else ''}")

    if not args.translated_srt or not Path(args.translated_srt).exists():
        print("\nNeed at least --translated-srt. Nothing to do.")
        return 2

    src = parse_srt(Path(args.translated_srt))
    merged = merge_segments(src, args.merge_gap, args.merge_max_dur)
    print(f"\n  parsed {len(src)} SRT segments → {len(merged)} after merge "
          f"(gap≤{args.merge_gap}s, max {args.merge_max_dur}s)")

    # ── optional: word-level forced alignment as the speech-timing reference ──
    fa_orig = fa_dub = None
    want_align = args.align
    if want_align is None:  # auto
        want_align = bool(args.original_media and Path(args.original_media).exists())
    if want_align:
        try:
            import forced_align as _fa
        except Exception as exc:
            print(f"\n(forced alignment unavailable: {exc})")
            _fa = None
        if _fa:
            if args.original_media and args.original_srt and \
               Path(args.original_media).exists() and Path(args.original_srt).exists():
                print("\n⏳ forced-aligning ORIGINAL audio to its transcript "
                      f"(device={args.align_device})…")
                fa_orig = _fa.align_segments(Path(args.original_media),
                                             parse_srt(Path(args.original_srt)),
                                             device=args.align_device)
                print(f"   {fa_orig.note}")
            if args.dub_media and Path(args.dub_media).exists():
                print("⏳ forced-aligning DUB audio to the translated transcript…")
                fa_dub = _fa.align_segments(Path(args.dub_media), merged,
                                            device=args.align_device)
                print(f"   {fa_dub.note}")

    A = check_A(merged, R)
    B = (check_B(merged, Path(args.temp_dir), R, args.max_speed, args.min_speed)
         if args.temp_dir and Path(args.temp_dir).is_dir() else None)
    C = (check_C(Path(args.dub_media), Path(args.dub_srt), merged, R,
                 fa=fa_dub if (fa_dub and fa_dub.ok) else None)
         if args.dub_media and args.dub_srt
         and Path(args.dub_media).exists() and Path(args.dub_srt).exists() else None)
    D = (check_D(Path(args.original_media), Path(args.original_srt), R,
                 fa=fa_orig if (fa_orig and fa_orig.ok) else None)
         if args.original_media and args.original_srt
         and Path(args.original_media).exists() and Path(args.original_srt).exists() else None)
    E = check_E(merged, D, B, C, R) if D else None
    if not D:
        print("\nD/E  original-speaker comparison — skipped "
              "(need --original-media + --original-srt)")

    V = verdict(A, B, C, D, E, R)

    if args.json:
        Path(args.json).write_text(json.dumps(dict(
            translated_srt=str(args.translated_srt),
            merged_segments=len(merged),
            A=A, B=_strip(B), C=_strip(C), D=_strip(D), E=E, verdict=V,
        ), indent=2, default=str))
        print(f"\n  scorecard → {args.json}")
    return 0


def _strip(d: Optional[dict]) -> Optional[dict]:
    if not d:
        return d
    return {k: v for k, v in d.items() if k not in ("per_seg", "rows", "segs", "vseg")}


if __name__ == "__main__":
    sys.exit(main())
