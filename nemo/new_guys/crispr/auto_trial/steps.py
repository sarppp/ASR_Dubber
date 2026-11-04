"""Automatic subtitle pipeline steps 0-1. No models, no network.

step 0: take N English cues from the original SRT.
step 1: re-segment those cues into readable phrases (1.5-6.0 s, <=84 chars)
        without dropping, duplicating or reordering a single source word.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SRT = ROOT / ('CRISPR is capable of more than we initially expected _ '
              'Jennifer Doudna_ Full Interview_720p_en-US_129kbps.en.srt')

MAX_CHARS, HARD_CHARS = 60, 84
MIN_MS, MAX_MS = 1500, 6000
HOLD_MS = 350


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def parse_srt(path):
    raw = Path(path).read_text(encoding='utf-8-sig')
    cues = []
    for block in re.split(r'\r?\n\r?\n', raw.strip()):
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        clock = re.match(r'(\d\d):(\d\d):(\d\d)[,.](\d\d\d)\s*-->\s*'
                         r'(\d\d):(\d\d):(\d\d)[,.](\d\d\d)', lines[1])
        if not clock:
            continue
        g = [int(v) for v in clock.groups()]
        cues.append({'id': len(cues) + 1,
                     'start_ms': ((g[0] * 60 + g[1]) * 60 + g[2]) * 1000 + g[3],
                     'end_ms': ((g[4] * 60 + g[5]) * 60 + g[6]) * 1000 + g[7],
                     'en': ' '.join(lines[2:]).strip()})
    return cues


def step0(count):
    cues = parse_srt(SRT)[:count]
    if len(cues) != count:
        raise ValueError(f'Source has fewer than {count} cues')
    document = {'step': 0, 'source': str(SRT), 'source_sha256': sha256(SRT),
                'cue_count': len(cues), 'cues': cues}
    write_json(HERE / 'step_0_english.json', document)
    return document


def segment(cues):
    n = len(cues)
    best = {n: (0.0, None)}
    for start in range(n - 1, -1, -1):
        choices = []
        for stop in range(start + 1, n + 1):
            part = cues[start:stop]
            text = ' '.join(c['en'] for c in part)
            duration = part[-1]['end_ms'] - part[0]['start_ms']
            if len(text) > HARD_CHARS or duration > MAX_MS:
                break
            if stop not in best or duration <= 0:
                continue
            boundary = 0 if re.search(r'[.!?,;:]$', text) or stop == n else 1.5
            score = (best[stop][0] + 2 + boundary + max(0, 2 - duration / 1000) * 5
                     + max(0, duration / 1000 - 4.5) + max(0, len(text) - MAX_CHARS) * 0.2)
            choices.append((score, stop))
        if choices:
            best[start] = min(choices)
    if 0 not in best:
        raise ValueError('Cannot segment without dropping words')
    runs, start = [], 0
    while start < n:
        stop = best[start][1]
        runs.append(cues[start:stop])
        start = stop
    return runs


def step1():
    source = json.loads((HERE / 'step_0_english.json').read_text(encoding='utf-8'))
    cues = source['cues']
    runs = segment(cues)
    phrases = []
    for run in runs:
        first, last = run[0], run[-1]
        end = last['end_ms']
        index = cues.index(last)
        if end - first['start_ms'] < MIN_MS:
            limit = cues[index + 1]['start_ms'] if index + 1 < len(cues) else end + HOLD_MS
            end = min(limit, end + HOLD_MS, first['start_ms'] + MAX_MS)
        phrases.append({'id': len(phrases) + 1,
                        'source_cue_ids': [c['id'] for c in run],
                        'start_ms': first['start_ms'], 'end_ms': end,
                        'en': ' '.join(c['en'] for c in run).strip()})
    expected = [w for c in cues for w in c['en'].split()]
    actual = [w for p in phrases for w in p['en'].split()]
    if expected != actual:
        raise ValueError('Segmentation changed the English word sequence')
    ids = [i for p in phrases for i in p['source_cue_ids']]
    if ids != [c['id'] for c in cues]:
        raise ValueError('Segmentation duplicated, dropped or reordered cues')
    for phrase in phrases:
        duration = phrase['end_ms'] - phrase['start_ms']
        if not MIN_MS <= duration <= MAX_MS:
            raise ValueError(f'Phrase {phrase["id"]} duration {duration} ms out of bounds')
        if len(phrase['en']) > HARD_CHARS:
            raise ValueError(f'Phrase {phrase["id"]} exceeds {HARD_CHARS} characters')
    document = {'step': 1, 'input': 'step_0_english.json',
                'input_sha256': sha256(HERE / 'step_0_english.json'),
                'limits': {'min_ms': MIN_MS, 'max_ms': MAX_MS,
                           'soft_chars': MAX_CHARS, 'hard_chars': HARD_CHARS},
                'cue_count': len(cues), 'phrase_count': len(phrases), 'phrases': phrases}
    write_json(HERE / 'step_1_phrases.json', document)
    return document


MODEL = 'inclusionai/ling-3.0-flash'
ENV = ROOT.parent.parent / '.env'
SYSTEM = (
    'Translate each English subtitle cue into natural concise German and French subtitles. '
    'Cues are parts of one continuous interview. Context helps grammar and terminology, but '
    'NEVER move facts or meaning between cue IDs. Preserve names, numbers, negations and facts. '
    'One plain line per language, no HTML, no newline. Prefer 65 characters or fewer; '
    'hard maximum 84. Return JSON only: '
    '{"phrases":[{"id":1,"de":"...","fr":"..."}]} with every input ID exactly once, in order.'
)


def api_key():
    key = None
    for line in ENV.read_text(encoding='utf-8').splitlines():
        if '=' not in line or line.lstrip().startswith('#'):
            continue
        name, value = line.split('=', 1)
        name = name.strip().upper().replace(' ', '')
        if name.replace('_', '') in {'OPENROUTERAPIKEY', 'OPENROUTER'} or name == 'OPENROUTER_KEY':
            key = value.strip().strip('"\'')
    if not key:
        raise SystemExit('OpenRouter key missing from nemo/.env')
    return key


def step2():
    import urllib.error
    import urllib.request
    source = json.loads((HERE / 'step_1_phrases.json').read_text(encoding='utf-8'))
    rows = [{'id': p['id'], 'en': p['en'],
             'seconds': round((p['end_ms'] - p['start_ms']) / 1000, 3)} for p in source['phrases']]
    body = json.dumps({
        'model': MODEL, 'temperature': 0, 'max_tokens': 4000,
        'response_format': {'type': 'json_object'},
        'provider': {'order': ['Novita', 'DeepInfra'], 'allow_fallbacks': True},
        'messages': [{'role': 'system', 'content': SYSTEM},
                     {'role': 'user', 'content': json.dumps({'phrases': rows}, ensure_ascii=False)}],
    }).encode()
    request = urllib.request.Request(
        'https://openrouter.ai/api/v1/chat/completions', data=body,
        headers={'Authorization': 'Bearer ' + api_key(), 'Content-Type': 'application/json',
                 'HTTP-Referer': 'https://localhost/asr-pipeline', 'X-Title': 'crispr-auto-trial'})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', 'replace')[:500]
        raise SystemExit(f'OpenRouter HTTP {exc.code}: {detail}') from exc
    raw = payload['choices'][0]['message']['content']
    value = json.loads(raw)
    translated = value['phrases']
    if not isinstance(translated, list) or len(translated) != len(rows):
        raise SystemExit('Model changed the cue count')
    result = []
    for row, original in zip(translated, rows):
        if set(row) != {'id', 'de', 'fr'} or row['id'] != original['id']:
            raise SystemExit(f'Bad IDs in model output: {row.get("id")}')
        result.append({**original, 'de': row['de'].strip(), 'fr': row['fr'].strip()})
    document = {'step': 2, 'input': 'step_1_phrases.json',
                'input_sha256': sha256(HERE / 'step_1_phrases.json'),
                'model': MODEL, 'provider_requested': ['Novita', 'DeepInfra'],
                'usage': payload.get('usage'), 'phrases': result,
                'raw_model_output': raw}
    write_json(HERE / 'step_2_translation.json', document)
    return document


STRIPPED = {
    'de': {'fur': 'für', 'uber': 'über', 'konnen': 'können', 'mussen': 'müssen',
           'moglich': 'möglich', 'moglichkeit': 'Möglichkeit', 'machtig': 'mächtig',
           'gross': 'groß', 'grosse': 'große', 'grosser': 'großer', 'spat': 'spät',
           'spater': 'später', 'naturlich': 'natürlich', 'wahrend': 'während',
           'horen': 'hören', 'gefuhl': 'Gefühl', 'zuruck': 'zurück', 'fahig': 'fähig',
           'erklaren': 'erklären', 'verandern': 'verändern', 'personlich': 'persönlich'},
    'fr': {'etre': 'être', 'cout': 'coût', 'couts': 'coûts', 'francais': 'français',
           'deja': 'déjà', 'apres': 'après', 'tres': 'très', 'meme': 'même',
           'societe': 'société', 'decouverte': 'découverte', 'gene': 'gène',
           'genes': 'gènes', 'therapie': 'thérapie', 'cree': 'créé', 'creee': 'créée',
           'etait': 'était', 'etaient': 'étaient', 'annees': 'années',
           'difference': 'différence', 'special': 'spécial', 'premiere': 'première',
           'interet': 'intérêt', 'etude': 'étude', 'etudes': 'études',
           'developpement': 'développement', 'reussi': 'réussi'},
}
WORD = re.compile(r"[A-Za-zÀ-ÿ']+")
ACCENT = re.compile(r'[À-ÿ]')


def phrase_findings(row):
    findings = []
    for lang in ('de', 'fr'):
        text = row[lang]
        if not text.strip() or len(text) > HARD_CHARS:
            findings.append({'lang': lang, 'kind': 'length', 'chars': len(text)})
        if any(c in text for c in '\r\n<>'):
            findings.append({'lang': lang, 'kind': 'format'})
        for word in WORD.findall(text):
            expected = STRIPPED[lang].get(word.lower().strip("'"))
            if expected:
                findings.append({'lang': lang, 'kind': 'stripped_accent',
                                 'word': word, 'expected': expected})
        if len(text) >= 30 and not ACCENT.search(text) and not any(
                f['lang'] == lang for f in findings):
            findings.append({'lang': lang, 'kind': 'no_accents_warning'})
        ratio = len(text) / max(1, len(row['en']))
        if ratio < 0.45:
            findings.append({'lang': lang, 'kind': 'possible_omission', 'ratio': round(ratio, 3)})
        elif ratio > 1.6:
            findings.append({'lang': lang, 'kind': 'possible_padding', 'ratio': round(ratio, 3)})
    return findings


def blocking_findings(findings):
    """Length padding and the no-accent hint never justify deleting meaning."""
    return [f for f in findings if f['kind'] not in ('no_accents_warning', 'possible_padding')]


def step3(input_name='step_2_translation.json'):
    source = json.loads((HERE / input_name).read_text(encoding='utf-8'))
    if source['input_sha256'] != sha256(HERE / source['input']):
        raise SystemExit('Input artifact changed')
    rows = []
    for phrase in source['phrases']:
        findings = phrase_findings(phrase)
        blocking = blocking_findings(findings)
        rows.append({**phrase, 'findings': findings, 'blocking': bool(blocking),
                     'pass': not blocking})
    document = {'step': 3, 'input': input_name,
                'input_sha256': sha256(HERE / input_name),
                'gate_rules': {'max_chars': HARD_CHARS, 'min_ratio': 0.45,
                               'max_ratio': 1.6, 'accent_lexicon': 'de/fr stripped forms'},
                'passed': sum(r['pass'] for r in rows),
                'blocked': sum(not r['pass'] for r in rows), 'phrases': rows}
    write_json(HERE / 'step_3_gates.json', document)
    return document




REPAIR = (
    'Fix ONLY this one subtitle. Preserve EVERY element of meaning, including actions and verbs. '
    'Never delete information to make it shorter. If it is too long, compress the wording '
    '(shorter synonyms, tighter syntax), never drop facts. Fix grammar. '
    'One plain line per language, at most 84 characters. Return JSON only: '
    '{"id":N,"de":"...","fr":"..."}'
)


def step4():
    import urllib.request
    source = json.loads((HERE / 'step_3_gates.json').read_text(encoding='utf-8'))
    repaired = []
    for row in source['phrases']:
        needs = (not row['pass']) or any(f['kind'] == 'possible_padding' for f in row['findings'])
        if not needs:
            repaired.append(row)
            continue
        body = json.dumps({
            'model': MODEL, 'temperature': 0, 'max_tokens': 800,
            'response_format': {'type': 'json_object'},
            'provider': {'order': ['Novita', 'DeepInfra'], 'allow_fallbacks': True},
            'messages': [
                {'role': 'system', 'content': SYSTEM + ' ' + REPAIR},
                {'role': 'user', 'content': json.dumps({
                    'id': row['id'], 'en': row['en'], 'de': row['de'], 'fr': row['fr'],
                    'problems': blocking_findings(row['findings']),
                    'rule': 'keep all meaning; do not remove verbs or facts to shorten',
                }, ensure_ascii=False)},
            ],
        }).encode()
        request = urllib.request.Request(
            'https://openrouter.ai/api/v1/chat/completions', data=body,
            headers={'Authorization': 'Bearer ' + api_key(), 'Content-Type': 'application/json'})
        for attempt in range(2):
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.load(response)
            content = (payload['choices'][0].get('message') or {}).get('content')
            if content:
                break
            body = json.loads(body)
            body.pop('response_format', None)
            body['messages'][0]['content'] += ' Reply with a single JSON object and nothing else.'
            body = json.dumps(body).encode()
            request = urllib.request.Request(
                'https://openrouter.ai/api/v1/chat/completions', data=body,
                headers={'Authorization': 'Bearer ' + api_key(), 'Content-Type': 'application/json'})
        if not content:
            raise SystemExit(f'Repair returned no text for cue {row["id"]}')
        content = re.sub(r'^```(?:json)?\s*', '', content.strip()).rstrip('`').strip()
        value = json.loads(content)
        if value.get('id') != row['id']:
            raise SystemExit('Repair changed the ID')
        fixed = {**row, 'de': value['de'].strip(), 'fr': value['fr'].strip()}
        findings = phrase_findings(fixed)
        blocking = blocking_findings(findings)
        repaired.append({**fixed, 'findings': findings, 'blocking': bool(blocking),
                         'pass': not blocking, 'repaired': True,
                         'usage': payload.get('usage')})
    document = {'step': 4, 'input': input_name,
                'input_sha256': sha256(HERE / input_name),
                'repair_attempts': 1,
                'passed': sum(r['pass'] for r in repaired),
                'blocked': sum(not r['pass'] for r in repaired),
                'phrases': repaired}
    write_json(HERE / 'step_4_repair.json', document)
    return document




SEGMENT_SYSTEM = (
    'You segment and translate subtitles. Group consecutive English cues into complete spoken '
    'sentences or clauses. NEVER split a verb from its object, a preposition from its complement, '
    'an article from its noun, or a negation from its verb. Prefer ending at sentence punctuation. '
    'Each group must last 1.5-6.0 seconds and its English text must be at most 84 characters. '
    'Translate each group into natural concise German and French: one plain line per language, '
    'at most 84 characters, no HTML, no newline. Every input cue ID must appear exactly once, '
    'in order. Do not change English words. Return JSON only: '
    '{"phrases":[{"source_cue_ids":[1,2],"en":"...","de":"...","fr":"..."}]}'
)


def step1b(window, offset=0, srt=SRT):
    import urllib.request
    cues = parse_srt(srt)[offset:offset + window]
    if len(cues) < window:
        raise SystemExit('Source is shorter than the requested window')
    source = {'step': '1b_source', 'source': str(srt), 'source_sha256': sha256(srt),
              'offset': offset, 'cues': cues}
    write_json(HERE / 'step_1b_source.json', source)
    body = json.dumps({
        'model': MODEL, 'temperature': 0, 'max_tokens': 8000,
        'response_format': {'type': 'json_object'},
        'provider': {'order': ['Novita', 'DeepInfra'], 'allow_fallbacks': True},
        'messages': [{'role': 'system', 'content': SEGMENT_SYSTEM},
                     {'role': 'user', 'content': json.dumps({'cues': cues}, ensure_ascii=False)}],
    }).encode()
    request = urllib.request.Request(
        'https://openrouter.ai/api/v1/chat/completions', data=body,
        headers={'Authorization': 'Bearer ' + api_key(), 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.load(response)
    content = (payload['choices'][0].get('message') or {}).get('content')
    if not content:
        raise SystemExit('Segmenter returned no text')
    content = re.sub(r'^```(?:json)?\s*', '', content.strip()).rstrip('`').strip()
    groups = json.loads(content)['phrases']

    by_id = {c['id']: c for c in cues}
    seen = [i for g in groups for i in g['source_cue_ids']]
    if seen != [c['id'] for c in cues]:
        raise SystemExit('Segmenter dropped, duplicated or reordered cues')
    phrases = []
    for index, group in enumerate(groups):
        run = [by_id[i] for i in group['source_cue_ids']]
        start, end = run[0]['start_ms'], run[-1]['end_ms']
        nxt = by_id[groups[index + 1]['source_cue_ids'][0]]['start_ms'] if index + 1 < len(groups) else end + HOLD_MS
        if end - start < MIN_MS:
            end = min(nxt, end + HOLD_MS, start + MAX_MS)
        en = ' '.join(c['en'] for c in run).strip()
        phrase = {'id': index + 1, 'source_cue_ids': group['source_cue_ids'],
                  'start_ms': start, 'end_ms': end, 'en': en,
                  'de': str(group['de']).strip(), 'fr': str(group['fr']).strip()}
        phrases.append(phrase)
    expected = [w for c in cues for w in c['en'].split()]
    actual = [w for p in phrases for w in p['en'].split()]
    if expected != actual:
        raise SystemExit('Segmenter changed English words')
    problems = []
    for phrase in phrases:
        duration = phrase['end_ms'] - phrase['start_ms']
        if duration < MIN_MS or duration > MAX_MS:
            problems.append({'id': phrase['id'], 'kind': 'duration', 'ms': duration})
        if len(phrase['en']) > HARD_CHARS:
            problems.append({'id': phrase['id'], 'kind': 'en_length', 'chars': len(phrase['en'])})
        for lang in ('de', 'fr'):
            if (not phrase[lang] or len(phrase[lang]) > HARD_CHARS
                    or any(ch in phrase[lang] for ch in '\r\n<>')):
                problems.append({'id': phrase['id'], 'kind': lang, 'chars': len(phrase[lang])})
    document = {'step': '1b', 'input': 'step_1b_source.json',
                'input_sha256': sha256(HERE / 'step_1b_source.json'),
                'model': MODEL, 'usage': payload.get('usage'),
                'window': window, 'offset': offset,
                'phrase_count': len(phrases), 'problems': problems,
                'phrases': phrases}
    write_json(HERE / 'step_1b_phrases.json', document)
    return document




PASSAGE_SYSTEM = (
    'You translate subtitle sentences from English into German and French. Each item is one complete '
    'sentence already assembled from consecutive cues. Translate the whole sentence into natural '
    'spoken German and French, preserving names, numbers, negation and facts. Do not add commentary. '
    'One plain line per language, no newline. Return JSON only: '
    '{"sentences":[{"cue_ids":[1,2],"de":"...","fr":"..."}]} with every input cue_ids exactly once, '
    'in order.'
)


def sentences_from(phrases, max_cues=6, max_ms=30000):
    groups, run = [], []
    for phrase in phrases:
        run.append(phrase)
        text = phrase['en'].strip()
        span = run[-1]['end_ms'] - run[0]['start_ms']
        if (re.search(r'[.!?]$', text) or len(run) >= max_cues or span >= max_ms):
            groups.append(run)
            run = []
    if run:
        groups.append(run)
    return groups


def split_translation(text, weights):
    words = text.split()
    if len(words) < len(weights):
        return None
    total = sum(weights)
    out, cursor = [], 0.0
    for index, weight in enumerate(weights):
        cursor += weight
        target = round(len(words) * cursor / total)
        low = index + 1
        high = len(words) - (len(weights) - index - 1)
        boundary = min(max(target, low), high)
        out.append(' '.join(words[low - 1:boundary]))
        previous = boundary
    # rebuild exactly, ignoring the running cursor artefacts
    result, cursor2, prev = [], 0, 0
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if index == len(weights) - 1:
            boundary = len(words)
        else:
            boundary = round(len(words) * cumulative / total)
            boundary = min(max(boundary, prev + 1), len(words) - (len(weights) - index - 1))
        result.append(' '.join(words[prev:boundary]))
        prev = boundary
    return result


def step2b(input_name='step_1_phrases.json'):
    import urllib.request
    source = json.loads((HERE / input_name).read_text(encoding='utf-8'))
    phrases = source['phrases']
    groups = sentences_from(phrases)
    payload_rows = [{'cue_ids': [p['id'] for p in group],
                     'en': ' '.join(p['en'] for p in group)} for group in groups]
    body = json.dumps({
        'model': MODEL, 'temperature': 0, 'max_tokens': 8000,
        'reasoning': {'enabled': False},
        'response_format': {'type': 'json_object'},
        'provider': {'order': ['Novita', 'DeepInfra'], 'allow_fallbacks': True},
        'messages': [{'role': 'system', 'content': PASSAGE_SYSTEM},
                     {'role': 'user', 'content': json.dumps({'sentences': payload_rows}, ensure_ascii=False)}],
    }).encode()
    request = urllib.request.Request(
        'https://openrouter.ai/api/v1/chat/completions', data=body,
        headers={'Authorization': 'Bearer ' + api_key(), 'Content-Type': 'application/json',
                 'HTTP-Referer': 'https://localhost/asr-pipeline', 'X-Title': 'crispr-auto-trial'})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.load(response)
    content = (payload['choices'][0].get('message') or {}).get('content')
    if not content:
        raise SystemExit('Model returned no text')
    content = re.sub(r'^```(?:json)?\s*', '', content.strip()).rstrip('`').strip()
    translated = json.loads(content)['sentences']
    if [set(t['cue_ids']) for t in translated] != [set(r['cue_ids']) for r in payload_rows]:
        raise SystemExit('Model changed the sentence cue IDs')
    by_id = {p['id']: p for p in phrases}
    out = []
    for group, sentence in zip(groups, translated):
        weights = [max(1, len(p['en'])) for p in group]
        de = split_translation(str(sentence['de']).strip(), weights)
        fr = split_translation(str(sentence['fr']).strip(), weights)
        if de is None or fr is None:
            raise SystemExit('Sentence translation shorter than its cue count')
        for phrase, d, f in zip(group, de, fr):
            out.append({**by_id[phrase['id']], 'de': d, 'fr': f})
    out.sort(key=lambda r: r['id'])
    document = {'step': '2b', 'mode': 'sentence-then-split', 'input': input_name,
                'input_sha256': sha256(HERE / input_name),
                'model': MODEL, 'usage': payload.get('usage'),
                'sentence_count': len(groups), 'phrases': out}
    write_json(HERE / 'step_2b_translation.json', document)
    return document





def step5(input_name='step_2b_translation.json'):
    import shutil
    import subprocess
    import sys
    sys.path.insert(0, str(ROOT.parents[1] / 'subtitle_divider'))
    from bilingual_subtitles import Cue, validate_srt_layout, write_subtitles
    source = json.loads((HERE / input_name).read_text(encoding='utf-8'))
    if source['input_sha256'] != sha256(HERE / source['input']):
        raise SystemExit('Translation input changed')
    rows = source['phrases']
    cues = [Cue(r['start_ms'], r['end_ms'],
                '\n'.join(r[lang] for lang in ('en', 'de', 'fr'))) for r in rows]
    validate_srt_layout(cues, font_size=28, max_height_percent=20)
    srt = HERE / 'CRISPR-auto.srt'
    ass = HERE / 'CRISPR-auto.ass'
    write_subtitles(srt, cues, font_size=28, max_height_percent=20)
    write_subtitles(ass, cues, font_size=28, max_height_percent=20)
    video = HERE / 'CRISPR-auto.mp4'
    if not video.exists():
        shutil.copy2(source_video(), video)
    preview = HERE / 'CRISPR-auto.preview.mp4'
    start = max(0, rows[0]['start_ms'] / 1000 - 1)
    end = rows[-1]['end_ms'] / 1000 + 2
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-ss', f'{start:.2f}',
                    '-to', f'{end:.2f}', '-i', str(source_video()),
                    '-vf', f'ass={ass}', '-c:v', 'libx264', '-crf', '22',
                    '-preset', 'veryfast', '-c:a', 'aac', '-b:a', '128k', str(preview)],
                   check=True, timeout=300)
    report = {'status': 'complete', 'input': input_name,
              'input_sha256': sha256(HERE / input_name),
              'cues': len(cues), 'font_size': 28, 'rows_per_language': 3,
              'srt': str(srt), 'srt_sha256': sha256(srt),
              'ass': str(ass), 'ass_sha256': sha256(ass),
              'video': str(video), 'video_sha256': sha256(video),
              'preview': str(preview), 'preview_span_seconds': [round(start, 2), round(end, 2)],
              'semantic_quality_verified': False, 'listening_review_complete': False}
    write_json(HERE / 'step_5_publish.json', report)
    return report


def source_video():
    return SRT.parent / (SRT.name[:-len('.en.srt')] + '.mp4')


if __name__ == '__main__':



    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=['0', '1', '2', '2b', '3', '4', '5', '0+1'])
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--window', type=int, default=24)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--input', default=None)
    args = parser.parse_args()
    if args.step in ('0', '0+1'):
        step0(args.count)
    if args.step in ('1', '0+1'):
        step1()
    if args.step == '2':
        step2()
    if args.step == '3':
        step3(args.input or 'step_2_translation.json')
    if args.step == '4':
        step4(args.input or 'step_3_gates.json')
    if args.step == '2b':
        step2b(args.input or 'step_1_phrases.json')
    if args.step == '5':
        step5(args.input or 'step_2b_translation.json')
