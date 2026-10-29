"""Trial alignment of existing translations into shared subtitle events.

Uses timing and sentence punctuation, not a semantic model. Never translates,
deletes, or independently splits text to equal character counts. Review the
alignment report before treating a trial as a semantically verified alignment.

Example: python synchronized_subtitles.py --tracks movie.en.srt movie.de.srt
movie.fr.srt --output movie.synchronized.srt
Writes a separate SRT, styled ASS, and source-traceable alignment report.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
import hashlib
from itertools import product
import json
from pathlib import Path
import re
import tempfile

from bilingual_subtitles import (
    AssMeasurer, Cue, EMPTY_SUBTITLE_ROW, _ass_header, parse_srt,
    plain_srt_text, validate_srt_layout, write_subtitles,
)


def sentence_fragments(track: list[Cue]) -> tuple[list[Cue], list[int]]:
    """Expose sentence ends inside source cues, retaining all words in order.

    Intra-cue times are estimates from word position, not forced alignment.
    Source indices are retained so every displayed word can be traced back.
    """
    result, sources = [], []
    for index, cue in enumerate(track):
        text = plain_srt_text(cue.text)
        cuts = [0]
        for match in re.finditer(r'(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Ý])', text):
            previous = text[:match.start()].split()[-1]
            if previous.lower() not in {'dr.', 'mr.', 'mrs.', 'ms.', 'prof.', 'st.'}:
                cuts.append(match.end())
        parts = [text[a:z].strip() for a, z in zip(cuts, cuts[1:] + [len(text)])]
        total, offset = sum(len(p.split()) for p in parts), 0
        for part in parts:
            start = cue.start_ms + round((cue.end_ms - cue.start_ms) * offset / total)
            offset += len(part.split())
            end = cue.start_ms + round((cue.end_ms - cue.start_ms) * offset / total)
            if end <= start:
                raise ValueError('Source cue is too short to expose its sentence boundaries')
            result.append(Cue(start, end, part))
            sources.append(index)
    return result, sources


def synchronize(tracks: list[list[Cue]], *, font_size: int = 18,
                width: int = 1100, max_height_percent: float = 20,
                max_join: int = 12, max_duration_ms: int = 22000,
                preserve_silence_ms: int = 750) -> list[dict]:
    """Find monotone shared blocks fitting two reserved rows per language.

    Translated end candidates are bounded to five near each source end. The
    path prefers sentence endings, modest timing shifts, and readable durations.
    Common pauses >= preserve_silence_ms cannot be crossed. Failure to find a
    path raises instead of dropping text or relaxing the layout cap.
    """
    if len(tracks) not in (2, 3) or any(not t for t in tracks):
        raise ValueError('Provide two or three nonempty tracks')
    if (not 1 <= font_size <= 100 or not 1 <= width <= 1200
            or not 1 <= max_height_percent <= 100 or max_join < 1
            or max_duration_ms <= 0 or preserve_silence_ms <= 0):
        raise ValueError('Invalid layout or grouping limits')
    originals = tracks
    for track in tracks:
        if any(c.start_ms < 0 or c.end_ms <= c.start_ms or not plain_srt_text(c.text).strip()
               for c in track) or any(a.end_ms > z.start_ms for a, z in zip(track, track[1:])):
            raise ValueError('Source tracks must have ordered, nonoverlapping, nonempty cues')
    split = [sentence_fragments(t) for t in tracks]
    tracks, source_ids = [s[0] for s in split], [s[1] for s in split]
    sizes = list(map(len, tracks))
    ends = [[c.end_ms for c in track] for track in tracks]
    coverage = []
    for cue in sorted((c for t in tracks for c in t), key=lambda c: c.start_ms):
        if coverage and cue.start_ms <= coverage[-1][1]:
            coverage[-1][1] = max(coverage[-1][1], cue.end_ms)
        else:
            coverage.append([cue.start_ms, cue.end_ms])
    silences = [(a[1], z[0]) for a, z in zip(coverage, coverage[1:])
                if z[0] - a[1] >= preserve_silence_ms]

    spans = []
    header = _ass_header(font_size, max_height_percent).replace('WrapStyle: 0', 'WrapStyle: 2')
    with AssMeasurer(header) as renderer:
        measured = {}

        def wrap(text):
            lines, line = [], ''
            for word in text.split():
                candidate = (line + ' ' + word).strip()
                if candidate not in measured:
                    left, _, right, _ = renderer.bounds(candidate, font_size)
                    measured[candidate] = right - left
                if measured[candidate] <= width:
                    line = candidate
                else:
                    if not line:
                        return None
                    lines.append(line)
                    line = word
                    if len(lines) >= 2:
                        return None
                    left, _, right, _ = renderer.bounds(word, font_size)
                    if right - left > width:
                        return None
            return lines + ([line] if line else [])

        for track in tracks:
            spans.append({})
            for start in range(len(track)):
                for end in range(start + 1, min(start + max_join, len(track)) + 1):
                    text = ' '.join(c.text for c in track[start:end])
                    lines = wrap(text)
                    if lines is None:
                        break
                    spans[-1][start, end] = (text, lines, track[start].start_ms, track[end - 1].end_ms)

    levels = [{tuple(0 for _ in tracks): (0.0, None)}]
    for i in range(1, sizes[0] + 1):
        target = ends[0][i - 1]
        choices = []
        for t in range(1, len(tracks)):
            nearest = bisect_left(ends[t], target) + 1
            choices.append(range(max(0, nearest - 2), min(sizes[t], nearest + 2) + 1))
        nodes = product([i], *choices) if i < sizes[0] else [tuple(sizes)]
        level = {}
        for node in nodes:
            texts = [tracks[t][n - 1].text if n else '' for t, n in enumerate(node)]
            punctuation = sum(0 if re.search(r'[.!?]["”’»]*$', text)
                              else 2 if text.endswith((',', ';', ':')) else 4 for text in texts)
            timing = sum(abs((ends[t][node[t] - 1] if node[t] else tracks[t][0].start_ms)
                             - target) / 1000 for t in range(1, len(tracks)))
            best = (float('inf'), None)
            for previous_i in range(max(0, i - max_join), i):
                for previous, (cost, _) in levels[previous_i].items():
                    if any(n < a or n - a > max_join for n, a in zip(node, previous)):
                        continue
                    cells = [spans[t].get((a, z)) if a != z else ('', [], target, target)
                             for t, (a, z) in enumerate(zip(previous, node))]
                    if any(c is None for c in cells):
                        continue
                    active = [c for c in cells if c[0]]
                    start, end = min(c[2] for c in active), max(c[3] for c in active)
                    duration = (end - start) / 1000
                    if end - start > max_duration_ms or any(start < a and end > z for a, z in silences):
                        continue
                    empty = sum(not c[0] for c in cells)
                    score = (cost + 4 + punctuation + timing * 2 + empty * 12
                             + max(0, 2 - duration) * 8 + max(0, duration - 8) * 1.5)
                    if score < best[0]:
                        best = (score, previous)
            if best[1] is not None:
                level[node] = best
        levels.append(level)
    node = tuple(sizes)
    if node not in levels[-1]:
        raise ValueError('No complete synchronized path fits the layout and timing limits')
    blocks = []
    while any(node):
        previous = levels[node[0]][node][1]
        cells = [spans[t].get((a, z), ('', [], 0, 0)) for t, (a, z) in enumerate(zip(previous, node))]
        active = [c for c in cells if c[0]]
        blocks.append(dict(
            start_ms=min(c[2] for c in active), end_ms=max(c[3] for c in active),
            text=[c[0] for c in cells], lines=[c[1] for c in cells],
            source_cue_indices=[sorted(set(source_ids[t][a:z])) for t, (a, z) in enumerate(zip(previous, node))],
        ))
        node = previous
    blocks.reverse()
    for previous, current in zip(blocks, blocks[1:]):
        if previous['end_ms'] > current['start_ms']:
            boundary = (previous['end_ms'] + current['start_ms']) // 2
            previous['end_ms'] = current['start_ms'] = boundary
    for language, track in enumerate(originals):
        expected = ' '.join(plain_srt_text(c.text) for c in track).split()
        actual = ' '.join(c['text'][language] for c in blocks).split()
        if expected != actual:
            raise ValueError('Synchronized output changed the source word sequence')
    cues = block_cues(blocks)
    validate_srt_layout(cues, font_size=font_size, max_height_percent=max_height_percent)
    return blocks


def block_cues(blocks: list[dict], languages: list[int] | None = None) -> list[Cue]:
    if languages is None:
        languages = list(range(len(blocks[0]['lines'])))
    return [Cue(b['start_ms'], b['end_ms'], '\n'.join(
        line for language in languages
        for line in b['lines'][language] + [EMPTY_SUBTITLE_ROW] * (2 - len(b['lines'][language]))))
        for b in blocks if any(b['lines'][language] for language in languages)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New trial .srt; also writes .ass and .alignment.json')
    parser.add_argument('--font-size', type=int, default=18)
    parser.add_argument('--max-height-percent', type=float, default=20)
    args = parser.parse_args(argv)
    output = args.output
    if output.suffix.lower() != '.srt':
        parser.error('--output must end with .srt')
    outputs = [output, output.with_suffix('.ass'), output.with_suffix('.alignment.json')]
    if any(p.exists() for p in outputs):
        parser.error('Trial outputs already exist; use a new output name')
    inputs = [{'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
              for p in args.tracks]
    blocks = synchronize([parse_srt(p) for p in args.tracks], font_size=args.font_size,
                         max_height_percent=args.max_height_percent)
    report = dict(alignment_basis='Timing and punctuation; not a semantic model or full semantic verification',
                  inputs=inputs, font_size=args.font_size, rows_per_language=2,
                  max_height_percent=args.max_height_percent, blocks=blocks)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.synchronized-', dir=output.parent) as directory:
        staged = Path(directory) / output.name
        write_subtitles(staged, block_cues(blocks), font_size=args.font_size,
                        max_height_percent=args.max_height_percent, ass_preview=True)
        staged.with_suffix('.alignment.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        for inp in inputs:
            if hashlib.sha256(Path(inp['path']).read_bytes()).hexdigest() != inp['sha256']:
                raise ValueError('Source changed during generation')
        created = []
        try:
            for src, dest in zip([staged, staged.with_suffix('.ass'), staged.with_suffix('.alignment.json')], outputs):
                # Exclusive creation protects a comparison file created during the run.
                with dest.open('xb') as stream:
                    created.append(dest)
                    stream.write(src.read_bytes())
        except BaseException:
            for dest in created:
                dest.unlink()
            raise
    print(f'Created {len(blocks)} shared blocks: {output} and {output.with_suffix(".ass")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
