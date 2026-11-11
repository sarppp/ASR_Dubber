"""Source-linked phrase translation for a separate, fully Modal experiment.

Pure helpers: no inference, cloud calls, or source-media writes on import.
Subtitle rows share times; speech synthesis uses complete source utterances.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

MODEL = 'Qwen/Qwen3-14B-AWQ'
REVISION = '31c69efc29464b6bb0aee1398b5a7b50a99340c3'
BATCH_SIZE = 1
MAX_PHRASES = 600


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def phrase_rows(speech):
    """Split at word boundaries, favoring punctuation, without crossing speakers.

    Retains every ASR word and its original acoustic times. Short trailing cues
    may hold for up to 350 ms, bounded by the next cue and the recording end.
    Short utterances are reported, never lengthened across another speaker.
    """
    if speech.get('status') != 'complete' or not speech.get('source_units'):
        raise ValueError('Complete fresh ASR and source units are required')
    rows, expected = [], []
    for unit in speech['source_units']:
        words = unit['words']
        expected.extend(unit['source_word_indices'])
        n = len(words)
        # Dynamic programming avoids a long caption followed by a one-word tail.
        best = {n: (0.0, None)}
        for start in range(n - 1, -1, -1):
            choices = []
            for stop in range(start + 1, min(n, start + 24) + 1):
                text = ' '.join(w['word'].strip() for w in words[start:stop])
                duration = words[stop - 1]['end'] - words[start]['start']
                if len(text) > 52 or duration > 6:
                    break
                if duration <= 0 or stop not in best:
                    continue
                boundary = 0 if re.search(r'[.!?,;:]$', text) or stop == n else 1.5
                score = (best[stop][0] + 2 + boundary + max(0, 2 - duration) * 5
                         + max(0, duration - 4.5))
                choices.append((score, stop))
            if choices:
                best[start] = min(choices)
        if 0 not in best:
            raise ValueError(f'Cannot fit source unit {unit["id"]} without dropping words')
        start = 0
        while start < n:
            stop = best[start][1]
            part = words[start:stop]
            rows.append({'id': len(rows) + 1, 'unit_id': unit['id'], 'speaker': unit['speaker'],
                         'start_ms': round(part[0]['start'] * 1000),
                         'end_ms': round(part[-1]['end'] * 1000),
                         'en': ' '.join(w['word'].strip() for w in part),
                         'source_word_indices': [w['index'] for w in part]})
            start = stop
    cuts = speech.get('phrase_split_before_word_indices', [])
    if not isinstance(cuts, list) or any(type(index) is not int or index not in expected for index in cuts):
        raise ValueError('Invalid adaptive phrase boundaries')
    if cuts:
        word_map = {word['index']: word for unit in speech['source_units'] for word in unit['words']}
        revised = []
        for row in rows:
            parts = [[]]
            for index in row['source_word_indices']:
                if index in cuts and parts[-1]:
                    parts.append([])
                parts[-1].append(word_map[index])
            for part in parts:
                revised.append({**row, 'id': len(revised)+1,
                    'start_ms': round(part[0]['start']*1000), 'end_ms': round(part[-1]['end']*1000),
                    'en': ' '.join(w['word'].strip() for w in part),
                    'source_word_indices': [w['index'] for w in part]})
        rows = revised
    if not rows or len(rows) > MAX_PHRASES:
        raise ValueError('Phrase count outside reviewed bound')
    actual = [i for row in rows for i in row['source_word_indices']]
    if actual != expected or len(set(actual)) != len(actual):
        raise ValueError('Source words were lost, duplicated, or reordered')
    for index, row in enumerate(rows):
        limit = rows[index + 1]['start_ms'] if index + 1 < len(rows) else round(speech['duration_seconds'] * 1000)
        if row['end_ms'] > limit or row['end_ms'] <= row['start_ms']:
            raise ValueError('Overlapping or invalid source word timeline')
        row['end_ms'] = min(limit, row['end_ms'] + 350, row['start_ms'] + 6000)
    return rows


def split_phrase_for_layout(speech, row):
    """Split an untranslated display phrase, preserving every source word/time."""
    indices = row['source_word_indices']
    if len(indices) < 2:
        raise ValueError('Single source word cannot be split further for layout')
    word_map = {w['index']: w for u in speech['source_units'] for w in u['words']}
    words = [word_map[i] for i in indices]
    middle = len(row['en'])/2
    split = min(range(1, len(words)), key=lambda pos:
        abs(len(' '.join(w['word'] for w in words[:pos]))-middle)
        - (5 if re.search(r'[.!?,;:]$', words[pos-1]['word']) else 0))
    cut = indices[split]
    existing = list(speech.get('phrase_split_before_word_indices', []))
    if cut in existing:
        raise ValueError('Adaptive split made no progress')
    speech['phrase_split_before_word_indices'] = sorted(existing+[cut])
    return phrase_rows(speech)


def translation_messages(batch, before=(), after=(), repair=None):
    system = (
        'Translate ONLY the single target English phrase into natural concise German and French subtitles. '
        'All phrases are parts of a continuous conversation. Use surrounding context for grammar '
        'and consistent scientific terminology, but keep the meaning of each English phrase in '
        'that SAME phrase ID. Context is NOT part of the translation: do not borrow its facts or complete '
        'a fragment with words from neighboring phrases. Preserve names, numbers, negations and facts. Never invent or omit '
        'facts to meet a size limit. Aim for 65 characters or fewer; the HARD MAXIMUM is 84 characters per language per phrase. '
        'Prefer short everyday verbs and direct syntax, rather than verbose nominal phrases. '
        'Use idiomatic spoken phrasing that can be read aloud in the source duration. Avoid needless '
        'repetition and verbose literal wording. Do not translate context rows. Return JSON only: '
        '{"phrases":[{"id":1,"de":"...","fr":"..."}]}. Keep every input ID exactly once, in order.'
    )
    data = {'context_before': [r['en'] for r in before],
            'phrases': [{'id': r['id'], 'en': r['en'],
                         'seconds': round((r['end_ms'] - r['start_ms']) / 1000, 3)} for r in batch],
            'context_after': [r['en'] for r in after]}
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': json.dumps(data, ensure_ascii=False)}]
    if repair:
        messages.extend([{'role': 'assistant', 'content': repair['raw']},
                         {'role': 'user', 'content': repair['error'] +
                          '. Rewrite the rejected lines substantially more concisely, preserving meaning. '
                          'Return the complete corrected JSON batch. Maximum 84 characters per line.'}])
    return messages


def validate_translations(raw, source):
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Translation must be valid JSON') from exc
    if not isinstance(value, dict) or set(value) != {'phrases'}:
        raise ValueError('Expected the phrases object')
    translated = value['phrases']
    if not isinstance(translated, list) or len(translated) != len(source):
        raise ValueError(f'Changed phrase count: return exactly {len(source)} rows, with IDs {[r["id"] for r in source]}; never combine adjacent phrases')
    result = []
    for row, original in zip(translated, source):
        if (not isinstance(row, dict) or set(row) != {'id', 'de', 'fr'}
                or type(row['id']) is not int or row['id'] != original['id']):
            raise ValueError('Changed, duplicated, or reordered phrase IDs')
        for lang in ('de', 'fr'):
            text = row[lang]
            if (not isinstance(text, str) or not text.strip() or len(text) > 84
                    or any(c in text for c in '\r\n<>\\{}')
                    or any(ord(c) < 32 for c in text)):
                raise ValueError(f'Phrase {row["id"]}: invalid or overlong {lang} line ({len(text) if isinstance(text, str) else "non-text"}; maximum 84)')
        result.append({**original, 'de': row['de'].strip(), 'fr': row['fr'].strip()})
    return result


def translation_schema(batch):
    """Constrain batch size and source IDs during decoding, then validate order."""
    return {'type': 'object', 'properties': {'phrases': {
        'type': 'array', 'minItems': len(batch), 'maxItems': len(batch),
        'items': {'type': 'object', 'properties': {
            'id': {'type': 'integer', 'enum': [row['id'] for row in batch]},
            'de': {'type': 'string'}, 'fr': {'type': 'string'}},
            'required': ['id', 'de', 'fr'], 'additionalProperties': False}}},
        'required': ['phrases'], 'additionalProperties': False}


def bind_translations(speech, rows):
    expected = phrase_rows(speech)
    if len(expected) != len(rows):
        raise ValueError('Incomplete full-video translations')
    for a, b in zip(expected, rows):
        if any(a[k] != b.get(k) for k in a):
            raise ValueError('Translated output changed source wording, timing, or IDs')
    result = deepcopy(speech)
    units = []
    for unit in speech['source_units']:
        phrases = [row for row in rows if row['unit_id'] == unit['id']]
        if not phrases:
            raise ValueError('Missing translated speech unit')
        units.append({**deepcopy(unit), 'translated_text': ' '.join(p['de'] for p in phrases),
                      'source_language': 'en', 'target_language': 'de',
                      'phrase_ids': [p['id'] for p in phrases]})
    result.update(phrases=rows, speech_units={'de': units},
                  translation_model=MODEL, translation_revision=REVISION,
                  translation_method='English phrase IDs with shared DE/FR boundaries; no external inference API',
                  semantic_quality_verified=False)
    return result


def check_audio_input(root):
    root = Path(root)
    manifest = json.loads((root / 'input/manifest.json').read_text())
    audio = root / 'input/source.wav'
    if sha256(audio) != manifest['audio_sha256']:
        raise ValueError('Source audio changed after preparation')
    return audio, manifest


def validate_source_duration(seconds):
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 600:
        raise ValueError('Only the bounded ten-minute CRISPR clip is permitted')
    return seconds
