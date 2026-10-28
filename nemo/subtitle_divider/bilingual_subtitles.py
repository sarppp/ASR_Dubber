#!/usr/bin/env python3
"""Shorten subtitles or stack language tracks, with optional compact ASS styling.

Input [EN], [DE], and [FR] prefixes are removed automatically. Language labels
are only added when explicitly requested. Use --tracks for two or more tracks
in display order; languages advance together in shared phrase groups. Use
--exact-boundaries to preserve independent source timings instead. Every CLI
output includes a styled ASS file with one consistent font size, chosen up to
38px to fit every cue within 20% of the 720p frame height. libass is required
to measure the actual rendered text. SRT outputs
also get an ASS companion, because SRT cannot enforce layout or font size.
All subtitle outputs use UTF-8 with BOM for reliable player encoding detection.

    python3 bilingual_subtitles.py --tracks movie.en.srt movie.de.srt movie.fr.srt \
        --output movie.en-de-fr.ass


The ``--top`` track is written above the ``--bottom`` track. Cue numbers do not
need to match. By default, line fragments are rebuilt into short sentence
groups and paired by timestamp overlap, avoiding flashes caused by slightly
different boundaries in independently generated translations.

Example:
    python3 bilingual_subtitles.py \
        --input movie.nemo.de.srt \
        --output movie.nemo.de.chunked.srt

    python3 bilingual_subtitles.py \
        --top movie.nemo.de.srt \
        --bottom movie.nemo.fr.srt \
        --output movie.nemo.de-fr.bilingual.srt
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


_TIMESTAMP_RE = re.compile(
    r"^\s*(\d+):([0-5]\d):([0-5]\d)[,.](\d{3})\s*-->\s*"
    r"(\d+):([0-5]\d):([0-5]\d)[,.](\d{3})(?:\s+.*)?$"
)
_SENTENCE_END_RE = re.compile(r"[.!?…][\"'»”’\)\]]*$")
_NUMBER_RE = re.compile(r"^\d+(?:[.,]\d+)?$")
_CONNECTORS = {
    "and", "but", "or", "so", "because",
    "und", "aber", "oder", "denn", "weil",
    "et", "mais", "ou", "donc", "car", "parce",
}
_UNSAFE_LINE_ENDS = {
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "from", "with",
    "ein", "eine", "einen", "einem", "einer", "der", "die", "das", "den", "dem",
    "des", "zu", "von", "im", "im", "am", "mit", "für", "als",
    "un", "une", "le", "la", "les", "des", "du", "de", "à", "au", "aux", "en",
    "pour", "avec", "dans", "sur",
}
_UNIT_WORDS = {
    "g", "kg", "gram", "grams", "gramm", "gramme", "grammes",
    "inch", "inches", "zoll", "pouce", "pouces", "cm", "mm", "%",
}
_RANGE_CONNECTORS = {"bis", "to", "à"}
_QUANTITY_LEADS = {"nur", "only", "seulement"}


@dataclass(frozen=True)
class Cue:
    start_ms: int
    end_ms: int
    text: str


def _parts_to_ms(parts: Sequence[str]) -> int:
    hours, minutes, seconds, milliseconds = (int(value) for value in parts)
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + milliseconds


def _format_timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _flatten_text(lines: Iterable[str]) -> str:
    """Keep each language on one display line while preserving inline markup."""
    return " ".join(
        part for line in lines
        if (part := re.sub(r"^\s*(?:\[(?:EN|DE|FR)\]\s*)+", "", line, flags=re.I).strip())
    )


def parse_srt(path: Path) -> list[Cue]:
    """Parse a standard SRT file and fail loudly on malformed non-empty blocks."""
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc

    normalized = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError(f"SRT is empty: {path}")

    cues: list[Cue] = []
    for block_number, block in enumerate(re.split(r"\n\s*\n", normalized), start=1):
        lines = block.splitlines()
        timestamp_index = next(
            (index for index, line in enumerate(lines) if _TIMESTAMP_RE.match(line)),
            None,
        )
        if timestamp_index is None:
            raise ValueError(
                f"Malformed SRT block {block_number} in {path}: missing timestamp"
            )

        match = _TIMESTAMP_RE.match(lines[timestamp_index])
        assert match is not None
        start_ms = _parts_to_ms(match.groups()[:4])
        end_ms = _parts_to_ms(match.groups()[4:])
        if end_ms <= start_ms:
            raise ValueError(
                f"Malformed SRT block {block_number} in {path}: "
                f"end time must be after start time"
            )

        text = _flatten_text(lines[timestamp_index + 1 :])
        if not text:
            raise ValueError(
                f"Malformed SRT block {block_number} in {path}: missing subtitle text"
            )
        cues.append(Cue(start_ms=start_ms, end_ms=end_ms, text=text))

    return sorted(cues, key=lambda cue: (cue.start_ms, cue.end_ms))


def _active_text(cues: Sequence[Cue], start_ms: int, end_ms: int) -> str:
    active = [
        cue.text
        for cue in cues
        if cue.start_ms < end_ms and cue.end_ms > start_ms
    ]
    # Overlaps inside a source track are unusual but legal. Preserve their
    # order without printing the same text twice.
    return " ".join(dict.fromkeys(active))


def _plain_word(word: str) -> str:
    return re.sub(r"(^[^\wÀ-ÿ%]+|[^\wÀ-ÿ%]+$)", "", word).lower()


def _boundary_penalty(words: Sequence[str], index: int) -> float:
    """Score a possible break before ``words[index]``; lower is better."""
    if index <= 0 or index >= len(words):
        return 0.0
    previous = words[index - 1]
    previous_plain = _plain_word(previous)
    next_plain = _plain_word(words[index])

    penalty = 0.0
    if re.search(r"[.!?…][\"'»”’\)\]]*$", previous):
        penalty -= 650.0
    elif re.search(r"[,;:][\"'»”’\)\]]*$", previous):
        penalty -= 420.0
    if next_plain in _CONNECTORS:
        penalty -= 160.0

    # Avoid dangling articles/prepositions and number-unit pairs.
    if previous_plain in _UNSAFE_LINE_ENDS:
        penalty += 900.0
    if next_plain in _UNIT_WORDS and (
        _NUMBER_RE.match(previous_plain) or any(char.isdigit() for char in previous_plain)
    ):
        penalty += 1_200.0
    if (
        (_NUMBER_RE.match(previous_plain) and next_plain in _RANGE_CONNECTORS)
        or (previous_plain in _RANGE_CONNECTORS and _NUMBER_RE.match(next_plain))
        or previous_plain in _QUANTITY_LEADS
        or (previous_plain in _UNIT_WORDS and next_plain in {"lang", "long"})
    ):
        penalty += 1_200.0
    if next_plain in _QUANTITY_LEADS:
        penalty -= 140.0
    if previous.endswith(("-", "–", "—")):
        penalty += 1_200.0
    return penalty


def split_text_naturally(text: str, parts: int, *, max_chars: int = 42) -> list[str]:
    """Split text into exactly ``parts`` balanced phrases using natural breaks."""
    words = text.split()
    if not words:
        return []
    parts = max(1, min(parts, len(words)))
    if parts == 1:
        return [text]

    target = max(1.0, (len(text) - (parts - 1)) / parts)
    word_count = len(words)
    lengths = [0]
    for word in words:
        lengths.append(lengths[-1] + len(word) + 1)
    penalties = [_boundary_penalty(words, end) if end < word_count else 0.0
                 for end in range(word_count + 1)]
    # dp[part_count][word_count_used] = (cost, previous_word_count)
    dp: list[dict[int, tuple[float, int]]] = [dict() for _ in range(parts + 1)]
    dp[0][0] = (0.0, -1)

    for part_count in range(1, parts + 1):
        min_end = part_count
        max_end = word_count - (parts - part_count)
        for end in range(min_end, max_end + 1):
            best: tuple[float, int] | None = None
            for start, (previous_cost, _) in dp[part_count - 1].items():
                if start >= end:
                    continue
                phrase_length = lengths[end] - lengths[start] - 1
                balance_cost = (phrase_length - target) ** 2
                overflow = max(0, phrase_length - max_chars)
                # ``max_chars`` is deliberately soft: a few extra characters
                # are preferable to cutting an article away from its noun or
                # ignoring a strong punctuation boundary.
                soft_overflow = min(overflow, 6)
                hard_overflow = max(0, overflow - 6)
                overflow_cost = (
                    40.0 * soft_overflow
                    + 5.0 * soft_overflow**2
                    + 500.0 * hard_overflow
                    + 40.0 * hard_overflow**2
                )
                break_cost = penalties[end]
                cost = previous_cost + balance_cost + overflow_cost + break_cost
                if best is None or cost < best[0]:
                    best = (cost, start)
            if best is not None:
                dp[part_count][end] = best

    if word_count not in dp[parts]:
        return [text]

    ranges: list[tuple[int, int]] = []
    end = word_count
    for part_count in range(parts, 0, -1):
        start = dp[part_count][end][1]
        ranges.append((start, end))
        end = start
    return [" ".join(words[start:end]) for start, end in reversed(ranges)]


def _allocate_chunk_times(
    start_ms: int,
    end_ms: int,
    weights: Sequence[int],
    *,
    min_duration_ms: int,
) -> list[tuple[int, int]]:
    """Allocate time proportionally while guaranteeing a readable minimum."""
    if len(weights) == 1:
        return [(start_ms, end_ms)]
    duration_ms = end_ms - start_ms
    guaranteed_ms = min(min_duration_ms, duration_ms // len(weights))
    remaining_ms = duration_ms - guaranteed_ms * len(weights)
    total_weight = max(1, sum(weights))

    boundaries = [start_ms]
    cumulative_weight = 0
    for index, weight in enumerate(weights[:-1], start=1):
        cumulative_weight += weight
        offset = index * guaranteed_ms + round(
            remaining_ms * cumulative_weight / total_weight
        )
        boundaries.append(start_ms + offset)
    boundaries.append(end_ms)
    return list(zip(boundaries, boundaries[1:]))


def split_bilingual_cue(
    cue: Cue,
    *,
    top_label: str = "",
    bottom_label: str = "",
    max_chars: int = 42,
    min_duration_ms: int = 1_300,
) -> list[Cue]:
    """Turn a long paired cue into short, equally counted bilingual chunks."""
    lines = cue.text.split("\n", 1)
    top_text = lines[0]
    bottom_text = lines[1] if len(lines) == 2 else ""
    raw_lengths = [len(text) for text in (top_text, bottom_text) if text]
    desired_parts = max(1, math.ceil(max(raw_lengths, default=1) / max_chars))
    duration_ms = cue.end_ms - cue.start_ms
    duration_limit = max(1, duration_ms // min_duration_ms)
    word_limit = min(len(text.split()) for text in (top_text, bottom_text) if text)
    max_parts = max(1, min(duration_limit, word_limit))
    parts = max(1, min(desired_parts, max_parts))

    while True:
        top_parts = split_text_naturally(top_text, parts, max_chars=max_chars)
        bottom_parts = (
            split_text_naturally(bottom_text, parts, max_chars=max_chars)
            if bottom_text
            else [""] * parts
        )
        longest_part = max(
            len(part) for part in (*top_parts, *bottom_parts) if part
        )
        if longest_part <= max_chars + 2 or parts >= max_parts:
            break
        parts += 1
    weights = [
        max(len(top_part), len(bottom_part), 1)
        for top_part, bottom_part in zip(top_parts, bottom_parts)
    ]
    timings = _allocate_chunk_times(
        cue.start_ms,
        cue.end_ms,
        weights,
        min_duration_ms=min_duration_ms,
    )

    result: list[Cue] = []
    for (start_ms, end_ms), top_part, bottom_part in zip(
        timings, top_parts, bottom_parts
    ):
        output_lines = []
        if top_part:
            output_lines.append(f"[{top_label}] {top_part}" if top_label else top_part)
        if bottom_part:
            output_lines.append(
                f"[{bottom_label}] {bottom_part}" if bottom_label else bottom_part
            )
        result.append(Cue(start_ms, end_ms, "\n".join(output_lines)))
    return result


def split_single_track(
    cues: Sequence[Cue],
    *,
    max_chars: int = 42,
    min_duration_ms: int = 1_300,
    max_group_gap_ms: int = 750,
    max_group_duration_ms: int = 12_000,
) -> list[Cue]:
    """Apply the same natural phrase and timing rules to one subtitle track."""
    if not cues:
        raise ValueError("Subtitle track has no cues")
    grouped = group_sentence_cues(
        cues,
        max_gap_ms=max_group_gap_ms,
        max_duration_ms=max_group_duration_ms,
    )
    result: list[Cue] = []
    for cue in grouped:
        result.extend(
            split_bilingual_cue(
                cue,
                max_chars=max_chars,
                min_duration_ms=min_duration_ms,
            )
        )
    return result


def combine_tracks_exact(
    top_cues: Sequence[Cue],
    bottom_cues: Sequence[Cue],
    *,
    top_label: str = "",
    bottom_label: str = "",
    keep_unpaired: bool = True,
) -> list[Cue]:
    """Split at every source boundary, preserving both timelines exactly."""
    if not top_cues:
        raise ValueError("Top subtitle track has no cues")
    if not bottom_cues:
        raise ValueError("Bottom subtitle track has no cues")

    boundaries = sorted(
        {value for cue in (*top_cues, *bottom_cues) for value in (cue.start_ms, cue.end_ms)}
    )
    combined: list[Cue] = []

    for start_ms, end_ms in zip(boundaries, boundaries[1:]):
        if end_ms <= start_ms:
            continue

        top_text = _active_text(top_cues, start_ms, end_ms)
        bottom_text = _active_text(bottom_cues, start_ms, end_ms)
        if not top_text and not bottom_text:
            continue
        if not keep_unpaired and (not top_text or not bottom_text):
            continue

        lines: list[str] = []
        if top_text:
            lines.append(f"[{top_label}] {top_text}" if top_label else top_text)
        if bottom_text:
            lines.append(
                f"[{bottom_label}] {bottom_text}" if bottom_label else bottom_text
            )
        text = "\n".join(lines)

        # A boundary from an overlapping cue can leave the visible state
        # unchanged. Coalesce those adjacent intervals to avoid needless flicker.
        if combined and combined[-1].end_ms == start_ms and combined[-1].text == text:
            previous = combined[-1]
            combined[-1] = Cue(previous.start_ms, end_ms, text)
        else:
            combined.append(Cue(start_ms, end_ms, text))

    return combined


def group_sentence_cues(
    cues: Sequence[Cue],
    *,
    max_gap_ms: int = 750,
    max_duration_ms: int = 12_000,
    max_chars: int = 220,
) -> list[Cue]:
    """Join adjacent line fragments without creating oversized subtitle cues."""
    if max_gap_ms < 0 or max_duration_ms <= 0 or max_chars <= 0:
        raise ValueError("Grouping limits must be positive")
    if not cues:
        return []

    grouped: list[Cue] = []
    current = cues[0]
    for cue in cues[1:]:
        gap_ms = cue.start_ms - current.end_ms
        joined_text = f"{current.text} {cue.text}"
        can_join = (
            gap_ms <= max_gap_ms
            and not _SENTENCE_END_RE.search(current.text)
            and cue.end_ms - current.start_ms <= max_duration_ms
            and len(joined_text) <= max_chars
        )
        if can_join:
            current = Cue(current.start_ms, max(current.end_ms, cue.end_ms), joined_text)
        else:
            grouped.append(current)
            current = cue
    grouped.append(current)
    return grouped


def _overlap_components(
    top_cues: Sequence[Cue], bottom_cues: Sequence[Cue]
) -> list[tuple[list[Cue], list[Cue]]]:
    """Return connected components whose cross-language intervals overlap."""
    top_count = len(top_cues)
    total = top_count + len(bottom_cues)
    parents = list(range(total))

    def find(node: int) -> int:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    top_index = 0
    bottom_index = 0
    while top_index < len(top_cues) and bottom_index < len(bottom_cues):
        top = top_cues[top_index]
        bottom = bottom_cues[bottom_index]
        if max(top.start_ms, bottom.start_ms) < min(top.end_ms, bottom.end_ms):
            union(top_index, top_count + bottom_index)

        if top.end_ms <= bottom.end_ms:
            top_index += 1
        else:
            bottom_index += 1

    members: dict[int, tuple[list[Cue], list[Cue]]] = {}
    for index, cue in enumerate(top_cues):
        members.setdefault(find(index), ([], []))[0].append(cue)
    for index, cue in enumerate(bottom_cues):
        members.setdefault(find(top_count + index), ([], []))[1].append(cue)
    return list(members.values())


def combine_tracks(
    top_cues: Sequence[Cue],
    bottom_cues: Sequence[Cue],
    *,
    top_label: str = "",
    bottom_label: str = "",
    keep_unpaired: bool = True,
    max_group_gap_ms: int = 750,
    max_group_duration_ms: int = 12_000,
    max_chars: int = 42,
    min_duration_ms: int = 1_300,
) -> list[Cue]:
    """Build cinema-style bilingual cues from overlapping sentence groups."""
    if not top_cues:
        raise ValueError("Top subtitle track has no cues")
    if not bottom_cues:
        raise ValueError("Bottom subtitle track has no cues")

    top_groups = group_sentence_cues(
        top_cues,
        max_gap_ms=max_group_gap_ms,
        max_duration_ms=max_group_duration_ms,
    )
    bottom_groups = group_sentence_cues(
        bottom_cues,
        max_gap_ms=max_group_gap_ms,
        max_duration_ms=max_group_duration_ms,
    )
    combined: list[Cue] = []

    for top_component, bottom_component in _overlap_components(top_groups, bottom_groups):
        if not keep_unpaired and (not top_component or not bottom_component):
            continue
        component = [*top_component, *bottom_component]
        lines: list[str] = []
        if top_component:
            top_text = " ".join(dict.fromkeys(cue.text for cue in top_component))
            lines.append(top_text)
        if bottom_component:
            bottom_text = " ".join(dict.fromkeys(cue.text for cue in bottom_component))
            lines.append(bottom_text)
        grouped_cue = Cue(
            start_ms=min(cue.start_ms for cue in component),
            end_ms=max(cue.end_ms for cue in component),
            text="\n".join(lines),
        )
        combined.extend(
            split_bilingual_cue(
                grouped_cue,
                top_label=top_label,
                bottom_label=bottom_label,
                max_chars=max_chars,
                min_duration_ms=min_duration_ms,
            )
        )

    return sorted(combined, key=lambda cue: (cue.start_ms, cue.end_ms))


def write_srt(path: Path, cues: Sequence[Cue]) -> None:
    if not cues:
        raise ValueError("Combined subtitle track has no cues")
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for index, cue in enumerate(cues, start=1):
        blocks.append(
            f"{index}\n{_format_timestamp(cue.start_ms)} --> "
            f"{_format_timestamp(cue.end_ms)}\n{cue.text}"
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8-sig")


def combine_multiple_tracks_exact(tracks: Sequence[Sequence[Cue]]) -> list[Cue]:
    """Stack all active languages in input order without changing their timings."""
    if len(tracks) < 2 or any(not track for track in tracks):
        raise ValueError("Provide at least two non-empty subtitle tracks")
    boundaries = sorted({t for track in tracks for cue in track
                         for t in (cue.start_ms, cue.end_ms)})
    result: list[Cue] = []
    for start, end in zip(boundaries, boundaries[1:]):
        lines = [" ".join(c.text for c in track
                          if c.start_ms < end and c.end_ms > start)
                 for track in tracks]
        text = "\n".join(line for line in lines if line)
        if not text:
            continue
        if result and result[-1].end_ms == start and result[-1].text == text:
            result[-1] = Cue(result[-1].start_ms, end, text)
        else:
            result.append(Cue(start, end, text))
    return result


def _split_multilingual_cue(cue: Cue, *, max_chars: int,
                            min_duration_ms: int) -> list[Cue]:
    """Split every language into the same number of phrases on one timeline."""
    lines = cue.text.split("\n")
    present = [line for line in lines if line]
    limit = max(1, min((cue.end_ms - cue.start_ms) // min_duration_ms,
                       min(len(line.split()) for line in present)))
    parts = min(limit, max(1, math.ceil(max(map(len, present)) / max_chars)))
    while True:
        chunks = [split_text_naturally(line, parts, max_chars=max_chars)
                  if line else [""] * parts for line in lines]
        if max(len(part) for track in chunks for part in track) <= max_chars + 2 or parts == limit:
            break
        parts += 1
    rows = list(zip(*chunks))
    times = _allocate_chunk_times(
        cue.start_ms, cue.end_ms, [max(map(len, row)) for row in rows],
        min_duration_ms=min_duration_ms,
    )
    return [Cue(start, end, "\n".join(part for part in row if part))
            for (start, end), row in zip(times, rows)]


def combine_multiple_tracks(
    tracks: Sequence[Sequence[Cue]], *, max_chars: int = 42,
    min_duration_ms: int = 1300,
) -> list[Cue]:
    """Align overlapping sentence groups and advance all languages together.

    Timings provide a heuristic, not semantic translation alignment. Small tail
    overlaps (under 20% of the shorter group) must not drag the next sentence
    into the current one. Exact source timing remains available separately.
    """
    if len(tracks) < 2 or any(not track for track in tracks):
        raise ValueError("Provide at least two non-empty subtitle tracks")
    if max_chars <= 0 or min_duration_ms <= 0:
        raise ValueError("Character and duration limits must be positive")
    nodes = sorted(
        ((language, cue) for language, track in enumerate(tracks)
         for cue in group_sentence_cues(sorted(track, key=lambda c: (c.start_ms, c.end_ms)))),
        key=lambda item: (item[1].start_ms, item[1].end_ms),
    )
    parents = list(range(len(nodes)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    active: list[int] = []
    for index, (language, cue) in enumerate(nodes):
        active = [other for other in active if nodes[other][1].end_ms > cue.start_ms]
        for other in active:
            other_language, other_cue = nodes[other]
            overlap = min(cue.end_ms, other_cue.end_ms) - cue.start_ms
            shorter = min(cue.end_ms - cue.start_ms, other_cue.end_ms - other_cue.start_ms)
            if language != other_language and overlap * 5 >= shorter:
                parents[root(index)] = root(other)
        active.append(index)

    components: dict[int, list[tuple[int, Cue]]] = {}
    for index, node in enumerate(nodes):
        components.setdefault(root(index), []).append(node)
    ordered = sorted(components.values(), key=lambda group: min(c.start_ms for _, c in group))
    # A component can surround another when inputs overlap within a track.
    # Merge those components so no text is reordered or given negative duration.
    groups: list[list[tuple[int, Cue]]] = []
    for component in ordered:
        while groups and max(c.end_ms for _, c in groups[-1]) >= max(c.end_ms for _, c in component):
            component = groups.pop() + component
        groups.append(component)
    starts = [min(c.start_ms for _, c in group) for group in groups]
    ends = [max(c.end_ms for _, c in group) for group in groups]
    for index in range(len(groups) - 1):
        if ends[index] > starts[index + 1]:
            boundary = (ends[index] + starts[index + 1]) // 2
            ends[index] = starts[index + 1] = boundary

    result: list[Cue] = []
    for start, end, group in zip(starts, ends, groups):
        lines = [" ".join(c.text for lang, c in sorted(group, key=lambda item: (item[1].start_ms, item[1].end_ms))
                          if lang == language) for language in range(len(tracks))]
        result.extend(_split_multilingual_cue(
            Cue(start, end, "\n".join(lines)), max_chars=max_chars,
            min_duration_ms=min_duration_ms,
        ))
    return result


class _AssImage(ctypes.Structure):
    pass


_AssImage._fields_ = [
    ("w", ctypes.c_int), ("h", ctypes.c_int), ("stride", ctypes.c_int),
    ("bitmap", ctypes.POINTER(ctypes.c_ubyte)), ("color", ctypes.c_uint32),
    ("dst_x", ctypes.c_int), ("dst_y", ctypes.c_int),
    ("next", ctypes.POINTER(_AssImage)), ("type", ctypes.c_int),
]


def _ass_header(font_size: int, max_height_percent: float) -> str:
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 0
ScaledBorderAndShadow: yes
; Maximum rendered subtitle height: {max_height_percent:g}% (including outline/shadow)

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,1.5,0.5,2,40,40,22,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_text(text: str) -> str:
    # Source text cannot inject ASS style overrides.
    return (text.replace("\\", "\\\\").replace("{", r"\{")
            .replace("}", r"\}").replace("\n", r"\N"))


class AssMeasurer:
    """Measure with libass itself, including wrapping, outline, and shadow.

    Font substitutions or player style overrides can change layout; use the
    supplied ASS styles and DejaVu Sans to reproduce these measurements.
    """
    def __init__(self, header: str):
        name = ctypes.util.find_library("ass")
        if not name:
            raise ValueError("Styled subtitles require libass (install libass9 on Debian/Ubuntu)")
        self.api = ctypes.CDLL(name)
        self.library = self.renderer = None
        self.header = header
        signatures = {
            "ass_library_init": (ctypes.c_void_p, []),
            "ass_library_done": (None, [ctypes.c_void_p]),
            "ass_renderer_init": (ctypes.c_void_p, [ctypes.c_void_p]),
            "ass_renderer_done": (None, [ctypes.c_void_p]),
            "ass_set_frame_size": (None, [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
            "ass_set_fonts": (None, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
                                      ctypes.c_int, ctypes.c_char_p, ctypes.c_int]),
            "ass_read_memory": (ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_char_p,
                                                  ctypes.c_size_t, ctypes.c_char_p]),
            "ass_free_track": (None, [ctypes.c_void_p]),
            "ass_render_frame": (ctypes.POINTER(_AssImage), [ctypes.c_void_p,
                                 ctypes.c_void_p, ctypes.c_longlong, ctypes.POINTER(ctypes.c_int)]),
        }
        for function, (restype, argtypes) in signatures.items():
            getattr(self.api, function).restype = restype
            getattr(self.api, function).argtypes = argtypes
        self.library = self.api.ass_library_init()
        if not self.library:
            raise ValueError("Could not initialize libass")
        # Suppress repeated library messages; initialization/render failures raise below.
        callback_type = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p,
                                         ctypes.c_void_p, ctypes.c_void_p)
        self._log_callback = callback_type(lambda level, fmt, args, data: None)
        self.api.ass_set_message_cb.restype = None
        self.api.ass_set_message_cb.argtypes = [ctypes.c_void_p, callback_type, ctypes.c_void_p]
        self.api.ass_set_message_cb(self.library, self._log_callback, None)
        self.renderer = self.api.ass_renderer_init(self.library)
        if not self.renderer:
            self.close()
            raise ValueError("Could not initialize ASS renderer")
        self.api.ass_set_frame_size(self.renderer, 1280, 720)
        self.api.ass_set_fonts(self.renderer, None, b"DejaVu Sans", 1, None, 1)

    def close(self) -> None:
        if self.renderer:
            self.api.ass_renderer_done(self.renderer)
            self.renderer = None
        if self.library:
            self.api.ass_library_done(self.library)
            self.library = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def bounds(self, text: str, font_size: int) -> tuple[int, int, int, int]:
        data = (self.header + "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,"
                + rf"{{\fs{font_size}}}" + _ass_text(text) + "\n").encode("utf-8")
        track = self.api.ass_read_memory(self.library, data, len(data), b"UTF-8")
        if not track:
            raise ValueError("Could not parse generated ASS for layout measurement")
        boxes = []
        try:
            changed = ctypes.c_int()
            node = self.api.ass_render_frame(self.renderer, track, 500, ctypes.byref(changed))
            while node:
                item = node.contents
                if item.w and item.h and item.color & 255 != 255:
                    boxes.append((item.dst_x, item.dst_y, item.dst_x + item.w, item.dst_y + item.h))
                node = item.next
        finally:
            self.api.ass_free_track(track)
        if not boxes:
            raise ValueError("Subtitle has no visible rendered glyphs")
        return (min(x[0] for x in boxes), min(x[1] for x in boxes),
                max(x[2] for x in boxes), max(x[3] for x in boxes))

    def fit(self, text: str, preferred_size: int, max_height: int) -> int:
        # Descending checks avoid assuming that font hinting/wrapping is monotonic.
        for size in range(preferred_size, 0, -1):
            left, top, right, bottom = self.bounds(text, size)
            if bottom - top <= max_height and left >= 20 and right <= 1260:
                return size
        raise ValueError("Subtitle cannot fit the requested frame-height limit")


def write_ass(path: Path, cues: Sequence[Cue], *, font_size: int = 38,
              max_height_percent: float = 20) -> None:
    """Write BOM-marked ASS with one font size that fits every caption."""
    if not cues or not 1 <= font_size <= 100 or not 1 <= max_height_percent <= 100:
        raise ValueError("Need non-empty subtitles and font size/height percent from 1 to 100")

    def timestamp(cs: int) -> str:
        hours, cs = divmod(cs, 360000)
        minutes, cs = divmod(cs, 6000)
        seconds, cs = divmod(cs, 100)
        return f"{hours}:{minutes:02}:{seconds:02}.{cs:02}"

    header = _ass_header(font_size, max_height_percent)
    events = []
    previous_end_cs = 0
    with AssMeasurer(header) as renderer:
        # Check every cue at the same size; per-cue fitting caused visible jumps.
        unique_texts = sorted(set(cue.text for cue in cues), key=len, reverse=True)
        for size in range(font_size, 0, -1):
            for content in unique_texts:
                left, top, right, bottom = renderer.bounds(content, size)
                if bottom - top > int(720 * max_height_percent / 100) or left < 20 or right > 1260:
                    break
            else:
                break
        else:
            raise ValueError("Subtitle cannot fit the requested frame-height limit")
        header = _ass_header(size, max_height_percent)
        for cue in cues:
            text = _ass_text(cue.text)
            # ASS has 10ms precision. Keep tiny transitions without creating
            # simultaneous events, which would stack two independently capped blocks.
            start_cs = max(round(cue.start_ms / 10), previous_end_cs)
            end_cs = max(round(cue.end_ms / 10), start_cs + 1)
            previous_end_cs = end_cs
            start, end = timestamp(start_cs), timestamp(end_cs)
            events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8-sig")


def write_subtitles(path: Path, cues: Sequence[Cue], *, font_size: int = 38,
                    max_height_percent: float = 20) -> None:
    ass_path = path if path.suffix.lower() == ".ass" else path.with_suffix(".ass")
    # Validate and render the styled output before writing an SRT companion.
    write_ass(ass_path, cues, font_size=font_size, max_height_percent=max_height_percent)
    if path.suffix.lower() != ".ass":
        write_srt(path, cues)
        print(f"Created styled companion {ass_path} (maximum height {max_height_percent:g}%)")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Naturally shorten one SRT or align multiple language tracks."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        help="Single SRT to shorten naturally (cannot be combined with --top/--bottom)",
    )
    parser.add_argument("--tracks", nargs="+", type=Path,
                        help="Two or more tracks in display order; align shared phrase boundaries")
    parser.add_argument("--font-size", type=int, default=38,
                        help="Maximum ASS font size at 720p (default 38); one size fits the whole file")
    parser.add_argument("--max-height-percent", type=float, default=20,
                        help="Maximum rendered ASS block height as percent of frame (default 20)")
    parser.add_argument("--top", type=Path, help="Bilingual mode: SRT shown on top")
    parser.add_argument(
        "--bottom", type=Path, help="Bilingual mode: SRT shown underneath"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output .ass or .srt; SRT always gets a styled .ass companion",
    )
    parser.add_argument(
        "--top-label", default="", help="Optional label such as DE (default: none)"
    )
    parser.add_argument(
        "--bottom-label", default="", help="Optional label such as FR (default: none)"
    )
    parser.add_argument(
        "--drop-unpaired",
        action="store_true",
        help="Hide intervals where only one language has an active cue",
    )
    parser.add_argument(
        "--exact-boundaries",
        action="store_true",
        help="Preserve every input boundary instead of grouping sentence fragments",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=42,
        help="Soft maximum characters per language and cue (default: 42)",
    )
    parser.add_argument(
        "--min-duration-ms",
        type=int,
        default=1300,
        help="Minimum duration for generated chunks in milliseconds (default: 1300)",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.max_height_percent <= 100:
        parser.error("--max-height-percent must be between 1 and 100")
    if not 1 <= args.font_size <= 100:
        parser.error("--font-size must be between 1 and 100")
    if args.max_chars <= 0 or args.min_duration_ms <= 0:
        parser.error("--max-chars and --min-duration-ms must be positive")
    if args.tracks:
        if len(args.tracks) < 2 or args.input or args.top or args.bottom:
            parser.error("--tracks needs at least two files and cannot use --input/--top/--bottom")
        if args.top_label or args.bottom_label or args.drop_unpaired:
            parser.error("labels and --drop-unpaired require --top/--bottom")
    elif args.input:
        if args.top or args.bottom:
            parser.error("--input cannot be combined with --top or --bottom")
        if args.drop_unpaired or args.exact_boundaries:
            parser.error("--drop-unpaired/--exact-boundaries require --top and --bottom")
    elif not (args.top and args.bottom):
        parser.error("provide --input, or provide both --top and --bottom")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.tracks:
        output = args.output or args.tracks[0].with_suffix(".multilingual.srt")
    elif args.input:
        output = args.output or args.input.with_name(f"{args.input.stem}.chunked.srt")
    else:
        output = args.output or args.top.with_name(f"{args.top.stem}.bilingual.srt")
    try:
        if args.tracks:
            tracks = [parse_srt(path) for path in args.tracks]
            combined = (combine_multiple_tracks_exact(tracks) if args.exact_boundaries
                        else combine_multiple_tracks(tracks, max_chars=args.max_chars,
                                                     min_duration_ms=args.min_duration_ms))
            write_subtitles(output, combined, font_size=args.font_size,
                            max_height_percent=args.max_height_percent)
            print(f"Created {output} ({len(combined)} cues)")
            return 0
        if args.input:
            source_cues = parse_srt(args.input)
            chunked = split_single_track(
                source_cues,
                max_chars=args.max_chars,
                min_duration_ms=args.min_duration_ms,
            )
            write_subtitles(output, chunked, font_size=args.font_size,
                            max_height_percent=args.max_height_percent)
            print(f"Created {output}")
            print(f"Input cues: {len(source_cues)} | Output cues: {len(chunked)}")
            return 0

        assert args.top is not None and args.bottom is not None
        top_cues = parse_srt(args.top)
        bottom_cues = parse_srt(args.bottom)
        if args.exact_boundaries:
            combined = combine_tracks_exact(
                top_cues,
                bottom_cues,
                top_label=args.top_label,
                bottom_label=args.bottom_label,
                keep_unpaired=not args.drop_unpaired,
            )
        else:
            combined = combine_tracks(
                top_cues,
                bottom_cues,
                top_label=args.top_label,
                bottom_label=args.bottom_label,
                keep_unpaired=not args.drop_unpaired,
                max_chars=args.max_chars,
                min_duration_ms=args.min_duration_ms,
            )
        write_subtitles(output, combined, font_size=args.font_size,
                            max_height_percent=args.max_height_percent)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    bilingual = sum(1 for cue in combined if "\n" in cue.text)
    print(f"Created {output}")
    print(
        f"Top cues: {len(top_cues)} | Bottom cues: {len(bottom_cues)} | "
        f"Output cues: {len(combined)} | Bilingual intervals: {bilingual}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
