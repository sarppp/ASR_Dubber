"""
forced_align.py — word-level timing of an audio file against a known transcript.

Uses torchaudio's MMS_FA multilingual forced-alignment model (wav2vec2 CTC,
~1.2 GB, downloaded once and cached).  Because it aligns to *known text* it is
immune to background music / ambience — unlike a blind VAD or a silence
threshold, which fail on produced video that has a music bed.

Public API
----------
    words = align_words(media_path, transcript_words, device="auto")
        -> list[Word]  with .text .start .end  (seconds, absolute)

    result = align_segments(media_path, segments, device="auto")
        segments: list of dicts with "start","end","text" (approx SRT timing)
        -> AlignResult with:
             .segments  list of dicts: index, voiced, span, w_start, w_end, n_words
             .pauses    list of (start, end) gaps > 0.35 s between consecutive words
             .ok        bool

Everything degrades gracefully: on any failure `align_segments(...).ok` is False
and the caller should fall back to the SRT.
"""
from __future__ import annotations

import re
import subprocess
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SR = 16000
_MIN_PAUSE = 0.35


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class AlignResult:
    ok: bool = False
    segments: List[Dict] = field(default_factory=list)
    pauses: List[Tuple[float, float]] = field(default_factory=list)
    words: List[Word] = field(default_factory=list)
    note: str = ""


def _romanize(word: str) -> str:
    """MMS_FA's dictionary is romanized lowercase a-z + apostrophe.  Strip
    accents (é->e, ç->c, ü->u …) and anything else.  Digits/symbols romanize to
    '' and are dropped (they can't be CTC-aligned to letters anyway)."""
    w = unicodedata.normalize("NFKD", word)
    w = "".join(c for c in w if not unicodedata.combining(c))
    w = w.lower().replace("’", "'")
    w = re.sub(r"[^a-z']", "", w)
    return w.strip("'")


def _load_audio(media_path: Path):
    import torch
    wav_bytes = subprocess.run(
        ["ffmpeg", "-i", str(media_path), "-f", "f32le", "-ac", "1",
         "-ar", str(_SR), "-loglevel", "error", "-"],
        check=True, capture_output=True,
    ).stdout
    import numpy as np
    arr = np.frombuffer(wav_bytes, dtype="<f4").copy()
    return torch.from_numpy(arr).unsqueeze(0)


def _pick_device(device: str):
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def align_words(media_path: Path, transcript_words: List[str],
                device: str = "auto") -> List[Word]:
    """Align `transcript_words` (in order) to the audio.  Words that romanize to
    empty (numbers, symbols) are kept as zero-length markers interpolated between
    their neighbours so segment word-counts stay aligned."""
    import torch
    from torchaudio.pipelines import MMS_FA as B

    dev = _pick_device(device)
    wav = _load_audio(media_path).to(dev)
    model = B.get_model().to(dev)
    tokenizer = B.get_tokenizer()
    aligner = B.get_aligner()

    roman = [_romanize(w) for w in transcript_words]
    keep_idx = [i for i, r in enumerate(roman) if r]
    kept = [roman[i] for i in keep_idx]
    if not kept:
        return []

    with torch.inference_mode():
        emission, _ = model(wav)
        spans = aligner(emission[0], tokenizer(kept))

    sec_per_frame = wav.shape[1] / emission.shape[1] / _SR
    kept_times = [(s[0].start * sec_per_frame, s[-1].end * sec_per_frame) for s in spans]

    out: List[Word] = []
    ptr = 0
    for i, w in enumerate(transcript_words):
        if ptr < len(keep_idx) and keep_idx[ptr] == i:
            s, e = kept_times[ptr]
            out.append(Word(w, float(s), float(e)))
            ptr += 1
        else:
            anchor = out[-1].end if out else 0.0
            out.append(Word(w, anchor, anchor))
    return out


def align_segments(media_path: Path, segments: List[Dict],
                   device: str = "auto") -> AlignResult:
    media_path = Path(media_path)
    try:
        import torch  # noqa
        import torchaudio  # noqa
    except Exception as exc:
        return AlignResult(note=f"torchaudio unavailable: {exc}")

    # flatten transcript, remember which segment each word belongs to
    words_flat: List[str] = []
    seg_of_word: List[int] = []
    for si, seg in enumerate(segments):
        toks = re.findall(r"\S+", seg.get("text", ""))
        for t in toks:
            words_flat.append(t)
            seg_of_word.append(si)
    if not words_flat:
        return AlignResult(note="empty transcript")

    try:
        aligned = align_words(media_path, words_flat, device=device)
    except Exception as exc:
        return AlignResult(note=f"alignment failed: {exc}")
    if len(aligned) != len(words_flat):
        return AlignResult(note="word count mismatch after alignment")

    # per-segment aggregation
    seg_rows: List[Dict] = []
    real = [w for w in aligned if w.end > w.start]
    for si, seg in enumerate(segments):
        ws = [w for w, s in zip(aligned, seg_of_word) if s == si and w.end > w.start]
        if not ws:
            seg_rows.append(dict(index=seg.get("index", si), voiced=0.0, span=0.0,
                                 w_start=None, w_end=None, n_words=0))
            continue
        voiced = 0.0
        prev_end = ws[0].start
        for w in ws:
            # count speech, but also count short intra-segment gaps (< MIN_PAUSE)
            gap = w.start - prev_end
            if 0 < gap < _MIN_PAUSE:
                voiced += gap
            voiced += max(0.0, w.end - w.start)
            prev_end = w.end
        seg_rows.append(dict(index=seg.get("index", si),
                             voiced=round(voiced, 3),
                             span=round(ws[-1].end - ws[0].start, 3),
                             w_start=round(ws[0].start, 3),
                             w_end=round(ws[-1].end, 3),
                             n_words=len(ws)))

    # global pauses between consecutive real words
    pauses: List[Tuple[float, float]] = []
    for a, b in zip(real, real[1:]):
        g = b.start - a.end
        if g >= _MIN_PAUSE:
            pauses.append((round(a.end, 3), round(b.start, 3)))

    return AlignResult(ok=True, segments=seg_rows, pauses=pauses, words=aligned,
                       note=f"aligned {len(real)}/{len(words_flat)} words")


if __name__ == "__main__":
    import sys
    import json

    def _ts(t):
        t = t.strip().replace(",", "."); h, m, s = t.split(":")
        return float(h) * 3600 + float(m) * 60 + float(s)

    def _parse(p):
        segs = []
        for blk in re.split(r"\n\s*\n", Path(p).read_text().strip()):
            L = [x for x in blk.splitlines() if x.strip()]
            if len(L) < 3:
                continue
            m = re.match(r"([\d:,.]+)\s*-->\s*([\d:,.]+)", L[1])
            if not m:
                continue
            txt = re.sub(r"\[[^\]]+\]", "", " ".join(L[2:])).strip()
            segs.append(dict(index=len(segs) + 1, start=_ts(m.group(1)),
                             end=_ts(m.group(2)), text=txt))
        return segs

    media, srt = sys.argv[1], sys.argv[2]
    r = align_segments(Path(media), _parse(srt))
    print("ok:", r.ok, "|", r.note)
    print(f"{len(r.pauses)} pauses > {_MIN_PAUSE}s")
    for a, b in r.pauses:
        print(f"  {a:8.2f} - {b:8.2f}   ({b-a:.2f}s)")
    print("\nseg  srt_span  FA_voiced  FA_span   window")
    for seg, row in zip(_parse(srt), r.segments):
        print(f"{row['index']:3d}  {seg['end']-seg['start']:7.2f}  {row['voiced']:8.2f}  "
              f"{row['span']:7.2f}   {row['w_start']}–{row['w_end']}   {seg['text'][:44]}")
