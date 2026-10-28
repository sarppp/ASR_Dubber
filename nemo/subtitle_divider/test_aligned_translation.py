import copy
import json
import pytest

from aligned_translation import validate_response, validate_complete, messages_for


@pytest.fixture
def source():
    return [dict(id=14, start_ms=31172, end_ms=32969, en='and we hoped, of course,'),
            dict(id=15, start_ms=32969, end_ms=35160, en="for impacting people's lives in the future.")]


@pytest.fixture
def translation():
    return [dict(id=14, de='und wir hofften natürlich,', fr='et nous espérions bien sûr,'),
            dict(id=15, de='das Leben der Menschen künftig zu beeinflussen.',
                 fr="avoir un impact sur la vie des gens à l'avenir.")]


def test_translation_text_cannot_change_english_or_timing(source, translation):
    result = validate_response(json.dumps(translation), source)
    cues = validate_complete(result, source)
    assert cues[0].start_ms == 31172 and cues[0].end_ms == 32969
    assert cues[0].text.splitlines() == [source[0]['en'], translation[0]['de'], translation[0]['fr']]
    assert 'espérions' in cues[0].text and 'hofften' in cues[0].text
    assert 'espérions' not in cues[1].text and 'hofften' not in cues[1].text


@pytest.mark.parametrize('kind', ['missing', 'reversed', 'duplicate', 'new_id', 'style', 'newline', 'timing', 'oversized'])
def test_invalid_model_outputs_are_rejected(source, translation, kind):
    rows = copy.deepcopy(translation)
    if kind == 'missing': rows.pop()
    if kind == 'reversed': rows.reverse()
    if kind == 'duplicate': rows[1]['id'] = rows[0]['id']
    if kind == 'new_id': rows[0]['id'] = 100
    if kind == 'style': rows[0]['de'] = '<font size="90">Hallo</font>'
    if kind == 'newline': rows[0]['fr'] = 'Bonjour\nAu revoir'
    if kind == 'timing': rows[0]['start_ms'] = 0
    if kind == 'oversized': rows[0]['de'] = 'W' * 160
    with pytest.raises(ValueError):
        validate_response(json.dumps(rows), source)


def test_export_refuses_partial_results_or_changed_source(source, translation):
    result = validate_response(json.dumps(translation), source)
    with pytest.raises(ValueError, match='Incomplete'):
        validate_complete(result[:1], source)
    result[0]['en'] = 'some unrelated sentence'
    with pytest.raises(ValueError, match='Source cue identity'):
        validate_complete(result, source)


def test_context_is_separate_from_cues_to_translate(source):
    messages = messages_for(source, 'Before.', 'After.')
    prompt = json.loads(messages[1]['content'])
    assert prompt['context_before'] == 'Before.'
    assert prompt['context_after'] == 'After.'
    assert prompt['cues'] == [{'id': r['id'], 'en': r['en']} for r in source]


def test_publisher_updates_canonical_srt_and_backs_up_old_versions(tmp_path, source, translation):
    import hashlib
    from aligned_translation import MODEL, REVISION
    from publish_aligned_srt import publish
    original = tmp_path / 'movie.en.srt'
    original.write_text('source identity')
    sha = hashlib.sha256(original.read_bytes()).hexdigest()
    source_file = tmp_path / 'source.json'
    source_file.write_text(json.dumps({'source': str(original), 'source_sha256': sha, 'cues': source}))
    result_file = tmp_path / 'result.json'
    result_file.write_text(json.dumps({'status': 'complete', 'model': MODEL, 'revision': REVISION,
        'source_sha256': sha, 'rows': validate_response(json.dumps(translation), source)}))
    old = tmp_path / 'movie.en-de-fr.srt'
    old.write_text('old captions')
    preview = publish(source_file, result_file)
    assert preview['status'] == 'preview'
    assert old.read_text() == 'old captions'
    receipt = publish(source_file, result_file, apply=True)
    from pathlib import Path
    assert len({Path(p).read_bytes() for p in receipt['outputs']}) == 1
    assert (Path(receipt['backup']) / '1.srt').read_text() == 'old captions'
    assert original.read_text() == 'source identity'
    assert not list(tmp_path.glob('*.ass'))
    original.write_text('concurrent change')
    with pytest.raises(ValueError, match='English source changed'):
        publish(source_file, result_file, apply=True)


def test_smoke_catches_the_reported_french_context_shift():
    from aligned_translation import validate_smoke
    source = [dict(id=13, start_ms=29022, end_ms=31172, en='a transformative technology for research,'),
              dict(id=14, start_ms=31172, end_ms=32969, en='and we hoped, of course,')]
    rows = validate_response(json.dumps([
        dict(id=13, de='eine transformative Technologie für die Forschung,',
             fr='une technologie transformatrice pour la recherche,'),
        dict(id=14, de='und wir hofften natürlich,', fr='et nous espérions bien sûr,')]), source)
    validate_smoke(rows, source)
    rows[1]['fr'] = 'transformatrice pour la recherche'
    with pytest.raises(ValueError, match='fr smoke'):
        validate_smoke(rows, source)
