"""Source-linked sentence groups and bounded spoken translation validation."""
from copy import deepcopy
import hashlib
import json
import re


def sentence_groups(data):
    groups, pending = [], []

    def finish():
        if not pending:
            return
        first, last = pending[0], pending[-1]
        words = [w for u in pending for w in u['words']]
        indices = [w['index'] for w in words]
        source = ' '.join(u['source_text'] for u in pending)
        groups.append({**deepcopy(first), 'id': 'spoken_' + hashlib.sha256(
            json.dumps([first['source_id'], indices]).encode()).hexdigest(),
            'source_end': last['source_end'], 'source_text': source,
            'words': words, 'source_word_indices': indices,
            'translated_text': '', 'parent_unit_ids': [u['id'] for u in pending]})
        pending.clear()

    for unit in data['source_units']:
        if pending and (any(unit[k] != pending[0][k] for k in
                            ('speaker', 'corridor_start', 'corridor_end', 'turn_id'))
                        or unit['source_end'] - pending[0]['source_start'] > 22):
            finish()
        pending.append(unit)
        if unit['source_end'] - pending[0]['source_start'] >= 8 and re.search(r'[.!?]$', unit['source_text']):
            finish()
    finish()
    # A clipped final phrase stays attached to its preceding sentence when safe.
    if len(groups) > 1 and groups[-1]['source_end']-groups[-1]['source_start'] < 3:
        a, b = groups[-2:]
        if (all(a[k] == b[k] for k in ('speaker', 'corridor_start', 'corridor_end', 'turn_id'))
                and b['source_end']-a['source_start'] <= 25):
            a.update(source_end=b['source_end'], source_text=a['source_text']+' '+b['source_text'],
                     words=a['words']+b['words'], source_word_indices=a['source_word_indices']+b['source_word_indices'],
                     parent_unit_ids=a['parent_unit_ids']+b['parent_unit_ids'])
            a['id'] = 'spoken_' + hashlib.sha256(json.dumps([a['source_id'], a['source_word_indices']]).encode()).hexdigest()
            groups.pop()
    assert [i for u in groups for i in u['source_word_indices']] == [i for u in data['source_units'] for i in u['source_word_indices']]
    return groups


def character_budget(unit):
    return max(16, int((unit['source_end']-unit['source_start'])*13))


def validate_spoken(raw, unit):
    value = json.loads(raw)
    text = value.get('de') if isinstance(value, dict) else None
    if (not isinstance(text, str) or not text.strip() or len(text) > character_budget(unit)
            or any(c in text for c in '\r\n<>\\{}')
            or any(ord(c) < 32 for c in text)):
        raise ValueError(f'Return one natural German line, at most {character_budget(unit)} characters, preserving the source meaning')
    return {**deepcopy(unit), 'translated_text': text.strip(), 'source_language': 'en', 'target_language': 'de'}
