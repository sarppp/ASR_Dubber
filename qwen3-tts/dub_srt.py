"""
dub_srt.py — Voice tables, SRT parsing, and voice assignment for the dub pipeline.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Voice tables
# ---------------------------------------------------------------------------
QWEN_FEMALE_VOICES = ["vivian", "ono_anna", "Chelsie"]
QWEN_MALE_VOICES   = ["ryan", "ethan"]

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
