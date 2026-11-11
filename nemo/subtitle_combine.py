"""Combine per-language subtitle rows into one multi-language subtitle file.

The pipeline produces one cue list per language with identical timings. This
module merges them so a single file carries EN/DE/FR, one row per language.

  combine(...)            -> [{start, end, lines: [(lang, text), ...]}, ...]
  render_srt(cues, order) -> one SRT block per cue, one line per language
  render_ass(cues, order) -> same, one dialogue per cue, languages joined by \\N

House style, matching synchronized_trial: one ``Default`` style, DejaVu Sans 28,
1280x720, white text, no per-language colours. Files are written as UTF-8 with a
BOM; without it players read the file as ANSI and ``Gründe`` shows as ``GrÃ¼nde``.
"""
from __future__ import annotations

from typing import Iterable

DEFAULT_ORDER = ("en", "de", "fr")
LANG_LABELS = {"en": "EN", "de": "DE", "fr": "FR", "es": "ES", "it": "IT"}


def _fmt_srt(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_ass(seconds: float) -> str:
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def combine(rows_by_lang: dict[str, list[dict]],
            order: Iterable[str] = DEFAULT_ORDER) -> list[dict]:
    """Merge per-language cue lists that share the same timings.

    A language whose cue count differs is skipped rather than guessed at, so a
    broken translation can never silently shift another language's timings.
    """
    order = [lang for lang in order if lang in rows_by_lang]
    if not order:
        raise ValueError("no languages to combine")

    reference = rows_by_lang[order[0]]
    for lang in order[1:]:
        if len(rows_by_lang[lang]) != len(reference):
            raise ValueError(
                f"{lang} has {len(rows_by_lang[lang])} cues, "
                f"{order[0]} has {len(reference)} — timings would not line up")

    cues = []
    for index, base in enumerate(reference):
        lines = []
        for lang in order:
            text = (rows_by_lang[lang][index].get("text") or "").strip()
            if text:
                lines.append((lang, text))
        cues.append({"start": base["start"], "end": base["end"], "lines": lines})
    return cues


def render_srt(cues: list[dict], labels: bool = False) -> str:
    blocks = []
    for index, cue in enumerate(cues, 1):
        body = "\n".join(f"{LANG_LABELS.get(lang, lang)}: {text}" if labels else text
                         for lang, text in cue["lines"])
        blocks.append(f"{index}\n{_fmt_srt(cue['start'])} --> {_fmt_srt(cue['end'])}\n{body}\n")
    return "\n".join(blocks)


ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 0
ScaledBorderAndShadow: yes
; Maximum rendered subtitle height: 20% (including outline/shadow)

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,28,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,1.5,0.5,2,40,40,22,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def render_ass(cues: list[dict], title: str = "multi") -> str:
    """One dialogue per cue, languages stacked top-to-bottom (EN/DE/FR).

    Matches the house style used by synchronized_trial: one Default style,
    DejaVu Sans 28, 1280x720, no per-language colours.
    """
    events = []
    for cue in cues:
        body = "\\N".join(text.replace("\n", " ") for _, text in cue["lines"])
        events.append(
            f"Dialogue: 0,{_fmt_ass(cue['start'])},{_fmt_ass(cue['end'])},Default,,0,0,0,,{body}")
    return ASS_HEADER + "\n".join(events) + "\n"


def write_files(rows_by_lang: dict[str, list[dict]], stem,
                order: Iterable[str] = DEFAULT_ORDER) -> list:
    """Write <stem>.multi.srt and <stem>.multi.ass. Returns the paths written."""
    from pathlib import Path

    stem = Path(stem)
    cues = combine(rows_by_lang, order)
    srt_path = stem.with_suffix(".multi.srt")
    ass_path = stem.with_suffix(".multi.ass")
    srt_path.write_text(render_srt(cues), encoding="utf-8-sig")
    ass_path.write_text(render_ass(cues), encoding="utf-8-sig")
    return [srt_path, ass_path]
