"""Pure validation for translating fixed English cue IDs into DE/FR.

No models or network calls. Translation workers return text only, never timing.
"""
from __future__ import annotations

import json
import re
from typing import Sequence

from bilingual_subtitles import Cue, plain_srt_text, validate_srt_layout


MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
BATCH_SIZE = 8


def messages_for(rows: Sequence[dict], before: str = "", after: str = "") -> list[dict]:
    return [
        {"role": "system", "content": (
            "You translate English subtitle cues into German and French. "
            "Translate ONLY the meaning within each cue's English text. Context "
            "helps grammar, but NEVER move words, facts or meaning between cue IDs. "
            "Keep names, numbers, negation and scientific terminology accurate. "
            "Use concise natural subtitle language. Each translation must be one "
            "line, preferably at most 60 characters. No HTML, styling or commentary. "
            "Return only a JSON array in input order, one object per cue, with "
            "exactly these keys: id (unchanged integer), de (text), fr (text). "
            "Do not translate the context or add/remove/reorder IDs."
        )},
        {"role": "user", "content": json.dumps({
            "context_before": before, "cues": [{"id": r["id"], "en": r["en"]} for r in rows],
            "context_after": after,
        }, ensure_ascii=False)},
    ]


def validate_response(raw: str, source: Sequence[dict]) -> list[dict]:
    """Reject missing/reordered IDs, style injection, and oversized SRT text.

    Structural validation is not a semantic quality score. A smoke review is
    required before accepting a model's full output.
    """
    raw = raw.strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)[:-3].strip()
    try:
        translated = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Translation did not return a complete JSON array") from exc
    if not isinstance(translated, list) or len(translated) != len(source):
        raise ValueError("Translation changed the cue count")
    result = []
    for row, original in zip(translated, source):
        if (not isinstance(row, dict) or set(row) != {"id", "de", "fr"}
                or type(row["id"]) is not int or row["id"] != original["id"]):
            raise ValueError("Translation changed, duplicated or reordered cue IDs")
        for lang in ("de", "fr"):
            value = row[lang]
            if (not isinstance(value, str) or not value.strip() or "\n" in value
                    or "\r" in value or plain_srt_text(value) != value):
                raise ValueError(f"Cue {row['id']} has invalid plain {lang} text")
        result.append({**original, "de": row["de"].strip(), "fr": row["fr"].strip()})
    # Measurement stays fixed at 38; overflow must be repaired in the text.
    validate_srt_layout([Cue(r["start_ms"], r["end_ms"],
                             "\n".join(r[lang] for lang in ("en", "de", "fr")))
                         for r in result])
    return result


def validate_complete(result: Sequence[dict], source: Sequence[dict]) -> list[Cue]:
    """Require full coverage and byte-exact English/timing before SRT export."""
    if len(result) != len(source):
        raise ValueError("Incomplete translated result; no SRT will be published")
    for row, expected in zip(result, source):
        if any(row.get(key) != expected[key] for key in ("id", "en", "start_ms", "end_ms")):
            raise ValueError("Source cue identity, text or timing changed")
    for offset in range(0, len(source), BATCH_SIZE):
        rows = result[offset:offset + BATCH_SIZE]
        validate_response(json.dumps([{k: row[k] for k in ("id", "de", "fr")} for row in rows]),
                          source[offset:offset + BATCH_SIZE])
    return [Cue(r["start_ms"], r["end_ms"], "\n".join(r[l] for l in ("en", "de", "fr")))
            for r in result]


def validate_smoke(rows: Sequence[dict], source: Sequence[dict]) -> None:
    """Check the reported meaning shift before spending on the full video.

    These semantic anchors cover the known regression only; they are not a
    blanket machine translation quality guarantee.
    """
    validate_complete(rows, source)
    by_id = {row["id"]: row for row in rows}
    for lang, transform, hope in (
        ("de", r"transform|revolution|veränder", r"hoff|erhofft|wünsch|erwart"),
        ("fr", r"transformat|révolution|boulevers", r"espér|espoir|souhait"),
    ):
        if (13 not in by_id or 14 not in by_id
                or not re.search(transform, by_id[13][lang], re.I)
                or not re.search(hope, by_id[14][lang], re.I)
                or re.search(transform, by_id[14][lang], re.I)):
            raise ValueError(f"{lang} smoke did not keep research and hoping in their correct cues")
