"""
dub_srt.py — Voice tables, SRT parsing, voice assignment, and dubbed SRT
             generation for the dub pipeline.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Voice tables
# ---------------------------------------------------------------------------
QWEN_FEMALE_VOICES = ["vivian", "ono_anna", "Chelsie"]
QWEN_MALE_VOICES   = ["ryan", "aiden"]

# Qwen TTS requires full lowercase language names, not ISO codes
LANG_CODE_TO_QWEN = {
    "fr": "french",  "en": "english", "de": "german",  "es": "spanish",
    "it": "italian", "ja": "japanese","ko": "korean",  "pt": "portuguese",
    "ru": "russian", "zh": "chinese", "auto": "auto",
}

def _qwen_lang(code: str) -> str:
    """Convert ISO code to Qwen language name, e.g. 'fr' → 'french'."""
    code = code.strip().lower()
    name = LANG_CODE_TO_QWEN.get(code, code)
    if name not in LANG_CODE_TO_QWEN.values():
        log.warning(f"Unknown language code '{code}' — passing as-is. "
                    f"Supported: {sorted(LANG_CODE_TO_QWEN.values())}")
    return name


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

def _srt_ts(t: str) -> float:
    """HH:MM:SS,mmm → seconds."""
    t = t.strip().replace(",", ".")
    h, m, s = t.split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)


def parse_srt(path: Path) -> List[Dict]:
    """
    Parse a diarized (and already-translated) SRT.

    Handles both formats produced by the pipeline:
      [Speaker 2] Bonjour le monde          ← nemo.py --diarize style
      [Speaker 2] Bonjour le monde          ← translate.py preserves the tag

    Returns list of:
      {"index": int, "start": float, "end": float, "speaker": str, "text": str}
    """
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", text.strip())
    segments = []

    for block in blocks:
        lines = [l.rstrip() for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue

        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue

        ts_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
            lines[1],
        )
        if not ts_match:
            continue

        start = _srt_ts(ts_match.group(1))
        end   = _srt_ts(ts_match.group(2))

        # Join continuation lines (translate.py sometimes wraps long lines)
        raw_text = " ".join(lines[2:]).strip()

        # Extract [Speaker N] label
        spk_match = re.match(r"\[([^\]]+)\]\s*(.*)", raw_text, re.DOTALL)
        if spk_match:
            speaker = spk_match.group(1).strip()
            text    = spk_match.group(2).strip()
        else:
            speaker = "Speaker 1"
            text    = raw_text

        # Restore any pipe-encoded newlines that translate.py may have left
        text = text.replace(" | ", " ").replace("|", " ").strip()

        if not text:
            continue

        segments.append({
            "index":   idx,
            "start":   start,
            "end":     end,
            "speaker": speaker,
            "text":    text,
        })

    log.info(f"Parsed {len(segments)} segments from SRT")
    speakers = sorted({s["speaker"] for s in segments})
    log.info(f"Speakers found: {speakers}")
    return segments


# ---------------------------------------------------------------------------
# Segment merging (naturalness improvement)
# ---------------------------------------------------------------------------

def merge_segments(
    segments: List[Dict],
    gap_sec: float = 1.0,
    max_dur: float = 10.0,
) -> List[Dict]:
    """Merge consecutive same-speaker segments separated by a gap ≤ gap_sec.

    Synthesising short subtitle lines individually causes choppy speech — the
    TTS model has no context across lines and produces micro-pauses at every
    segment boundary.  Merging gives it longer, coherent text so it generates
    natural prosody and intonation.

    The merged segment keeps the first segment's index, spans start→end of the
    whole group, and concatenates text with a single space.  Pass gap_sec=0
    to disable merging entirely.

    max_dur is a *soft* cap: merging stops at max_dur only when the current
    text ends on a sentence boundary (. ? !).  If the sentence is incomplete
    (e.g. ends mid-clause), merging continues up to max_dur * 2 so that TTS
    never receives a fragment like "Comment réconcilier...avec" without its
    completion.  This prevents audible mid-sentence cuts regardless of the
    per-video timing.
    """
    if not segments:
        return segments

    def _sentence_complete(text: str) -> bool:
        """True when text ends with terminal punctuation."""
        t = re.sub(r'\[Speaker\s+\d+\]\s*', '', text).strip().rstrip('"\'»›')
        return bool(t) and t[-1] in '.?!'

    hard_cap = max_dur * 2   # absolute ceiling — prevents runaway monologues

    merged: List[Dict] = []
    current = dict(segments[0])
    current["subsegments"] = [{"start": segments[0]["start"], "end": segments[0]["end"]}]

    for seg in segments[1:]:
        gap = seg["start"] - current["end"]
        merged_dur = seg["end"] - current["start"]
        within_hard_cap  = merged_dur <= hard_cap
        within_soft_cap  = merged_dur <= max_dur
        sentence_done    = _sentence_complete(current["text"])
        # Merge when:
        #   - same speaker, gap within threshold, AND
        #   - either still within soft cap,
        #     or sentence is incomplete and hard cap not yet reached
        if (seg["speaker"] == current["speaker"]
                and 0 <= gap <= gap_sec
                and (within_soft_cap or (not sentence_done and within_hard_cap))):
            current["end"]  = seg["end"]
            current["text"] = current["text"].rstrip() + " " + seg["text"].lstrip()
            current["subsegments"].append({"start": seg["start"], "end": seg["end"]})
        else:
            merged.append(current)
            current = dict(seg)
            current["subsegments"] = [{"start": seg["start"], "end": seg["end"]}]

    merged.append(current)

    n_before, n_after = len(segments), len(merged)
    if n_before != n_after:
        log.info(
            f"Merged {n_before} segments → {n_after} "
            f"(gap_sec={gap_sec:.1f}s, saved {n_before - n_after} TTS calls)"
        )
    return merged


# ---------------------------------------------------------------------------
# Voice assignment (custom mode)
# ---------------------------------------------------------------------------

def build_voice_map(segments: List[Dict]) -> Dict[str, str]:
    """Assign a Qwen voice to each speaker, alternating female/male pools."""
    seen: List[str] = []
    for seg in segments:
        if seg["speaker"] not in seen:
            seen.append(seg["speaker"])

    voice_map: Dict[str, str] = {}
    fi = mi = 0
    for i, spk in enumerate(seen):
        if i % 2 == 0:
            voice_map[spk] = QWEN_FEMALE_VOICES[fi % len(QWEN_FEMALE_VOICES)]
            fi += 1
        else:
            voice_map[spk] = QWEN_MALE_VOICES[mi % len(QWEN_MALE_VOICES)]
            mi += 1

    return voice_map


# ---------------------------------------------------------------------------
# Write dubbed SRT with actual audio timestamps
# ---------------------------------------------------------------------------

def _fmt_ts(seconds: float) -> str:
    """Seconds → SRT timestamp  HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def _split_subtitle_blocks(
    text: str, start: float, end: float, max_ch: int = 42,
) -> List[Tuple[str, float, float]]:
    """Split long text into movie-style subtitle blocks with proportional timestamps.

    Each block has at most two lines of ≤ max_ch characters.  Duration is
    distributed proportionally by character count so each block stays on
    screen the right amount of time.

    Returns list of (display_text, block_start, block_end).
    """
    words = text.split()
    if not words:
        return [(text, start, end)]

    # Greedily pack words into blocks of ≤ max_ch (single line)
    # or ≤ 2*max_ch (two lines).  Prefer sentence boundaries.
    blocks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    for w in words:
        new_len = buf_len + len(w) + (1 if buf else 0)
        if new_len <= max_ch:
            buf.append(w)
            buf_len = new_len
        else:
            if buf:
                blocks.append(" ".join(buf))
            buf = [w]
            buf_len = len(w)
    if buf:
        blocks.append(" ".join(buf))

    if not blocks:
        return [(text, start, end)]

    # Distribute timestamps proportionally by character count
    total_chars = sum(len(b) for b in blocks)
    dur = end - start
    result: List[Tuple[str, float, float]] = []
    cur = start
    for b in blocks:
        frac = len(b) / total_chars if total_chars > 0 else 1.0 / len(blocks)
        block_end = cur + dur * frac
        result.append((b, cur, block_end))
        cur = block_end

    return result


def write_dub_srt(
    out_path: Path,
    actual_positions: List[Tuple[float, float, float, float]],
    segments: List[Dict],
) -> Path:
    """Write a clean, movie-style SRT whose timestamps match the dubbed audio.

    - Speaker tags (``[Speaker N]``) are stripped — this is for viewing.
    - Long segments are split into short subtitle blocks (≤42 chars/line)
      with proportional timestamps.
    - Timestamps come from the actual stitched audio, not the original SRT.

    Parameters
    ----------
    out_path : Path
        Where to write the SRT (e.g. ``output/video_dub.srt``).
    actual_positions : list of (actual_start, actual_end, orig_start, orig_end)
        Returned by ``stitch_and_mix`` — the real position of each clip in the
        dubbed audio track.
    segments : list of dicts
        The (merged) segments with ``start``, ``end``, ``speaker``, ``text``.

    Returns the path written.
    """
    # Build lookup: (orig_start, orig_end) → segment dict
    seg_lookup: Dict[Tuple[float, float], Dict] = {}
    for seg in segments:
        seg_lookup[(seg["start"], seg["end"])] = seg

    lines: List[str] = []
    idx = 1
    for actual_start, actual_end, orig_start, orig_end in actual_positions:
        seg = seg_lookup.get((orig_start, orig_end))
        if seg is None:
            log.warning(f"No segment match for orig ({orig_start:.3f}–{orig_end:.3f})")
            continue
        for block_text, blk_start, blk_end in _split_subtitle_blocks(
            seg["text"], actual_start, actual_end
        ):
            lines.append(
                f"{idx}\n"
                f"{_fmt_ts(blk_start)} --> {_fmt_ts(blk_end)}\n"
                f"{block_text}\n"
            )
            idx += 1

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"📝 Dubbed SRT written: {out_path}  ({idx - 1} subtitle blocks)")
    return out_path
