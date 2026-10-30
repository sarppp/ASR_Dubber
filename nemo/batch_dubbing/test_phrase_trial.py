"""Local tests for phrase provenance, model output rejection, and bounded release."""
from copy import deepcopy
import json
import pytest

from batch_dubbing.phrase_trial import (phrase_rows, validate_translations, bind_translations,
                                        check_audio_input, sha256)
from batch_dubbing.speech_units import build_speech_units


def test_validation_preserves_german_french_and_unicode_escapes():
    source = [{'id': 206, 'en': 'frightening, especially for'}]
    for ascii_only in (False, True):
        raw = json.dumps({'phrases': [{'id': 206, 'de': 'für große Möglichkeiten',
                                      'fr': 'coût élevé, cœur, déjà'}]}, ensure_ascii=ascii_only)
        result = validate_translations(raw, source)
        assert result[0]['de'] == 'für große Möglichkeiten'
        assert result[0]['fr'] == 'coût élevé, cœur, déjà'


def speech_fixture():
    words = []
    groups = [('a', 0, 'Our research found a promising method. We hoped it would help people.'),
              ('b', 9, 'Did the treatment work?'), ('a', 12, 'Yes, but it did not cure every patient.')]
    turns = []
    for speaker, start, text in groups:
        for position, word in enumerate(text.split()):
            words.append({'word': word, 'start': start + position * .45,
                          'end': start + position * .45 + .4, 'speaker': speaker})
        turns.append({'start': start, 'end': words[-1]['end'], 'speaker': speaker})
    units = build_speech_units(words, turns, source_id='sha256:test')
    return {'status': 'complete', 'source_words': words, 'source_units': [u.to_dict() for u in units],
            'duration_seconds': 20, 'source_language': 'en', 'voices': {}}


def translated_rows(speech):
    rows = phrase_rows(speech)
    raw = json.dumps({'phrases': [{'id': r['id'], 'de': 'Kurzer deutscher Satz.',
                                  'fr': 'Une courte phrase.'} for r in rows]})
    return validate_translations(raw, rows)


def test_phrase_timing_uses_every_source_word_once_and_respects_speakers():
    speech = speech_fixture()
    rows = phrase_rows(speech)
    assert [i for r in rows for i in r['source_word_indices']] == list(range(len(speech['source_words'])))
    assert ' '.join(r['en'] for r in rows).split() == [w['word'] for w in speech['source_words']]
    for row in rows:
        words = [speech['source_words'][i] for i in row['source_word_indices']]
        assert {w['speaker'] for w in words} == {row['speaker']}
        assert row['start_ms'] == round(words[0]['start'] * 1000)
        assert 0 < row['end_ms'] - row['start_ms'] <= 6000
        assert len(row['en']) <= 52
    assert all(a['end_ms'] <= b['start_ms'] for a, b in zip(rows, rows[1:]))


@pytest.mark.parametrize('mutation', ['drop', 'reorder', 'long', 'markup', 'newline', 'bool'])
def test_rejects_unusable_translations_without_silently_repairing(mutation):
    rows = phrase_rows(speech_fixture())
    output = [{'id': r['id'], 'de': 'Deutsch', 'fr': 'Français'} for r in rows]
    if mutation == 'drop': output.pop()
    elif mutation == 'reorder': output.reverse()
    elif mutation == 'long': output[0]['de'] = 'x' * 85
    elif mutation == 'markup': output[0]['de'] = '<b>Text</b>'
    elif mutation == 'newline': output[0]['fr'] = 'un\ndeux'
    else: output[0]['id'] = True
    with pytest.raises(ValueError):
        validate_translations(json.dumps({'phrases': output}), rows)


def test_dub_units_preserve_source_provenance_and_join_phrase_text():
    speech = speech_fixture()
    rows = translated_rows(speech)
    result = bind_translations(speech, rows)
    for original, unit in zip(speech['source_units'], result['speech_units']['de']):
        for key in ('words', 'source_word_indices', 'source_start', 'source_end', 'speaker', 'turn_id'):
            assert original[key] == unit[key]
        assert unit['translated_text'] == ' '.join(r['de'] for r in rows if r['unit_id'] == unit['id'])
    assert result['semantic_quality_verified'] is False


def test_modified_phrase_boundaries_cannot_be_bound_to_speech():
    speech = speech_fixture()
    rows = translated_rows(speech)
    rows[0]['start_ms'] += 100
    with pytest.raises(ValueError, match='changed source'):
        bind_translations(speech, rows)


def test_audio_hash_is_checked_before_upload(tmp_path):
    root = tmp_path / 'input'
    root.mkdir()
    audio = root / 'source.wav'
    audio.write_bytes(b'original')
    (root / 'manifest.json').write_text(json.dumps({'audio_sha256': sha256(audio)}))
    assert check_audio_input(tmp_path)[0] == audio
    audio.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        check_audio_input(tmp_path)


def test_invalid_transcription_switch_fails_before_inference():
    from batch_dubbing.speech_stage import process_audio
    for invalid in (None, 1, 'true'):
        with pytest.raises(ValueError, match='Boolean'):
            process_audio('does-not-exist', 5, transcribe_only=invalid)


def test_ten_minute_trial_plan_is_serial_with_qwen_gpu_dubbing(tmp_path, monkeypatch):
    import wave
    from batch_dubbing import prepare_phrase_trial as preparation
    trial = tmp_path / 'experiments/trial'
    (trial / 'input').mkdir(parents=True)
    audio = trial / 'input/source.wav'
    with wave.open(str(audio), 'wb') as wav:
        wav.setparams((1, 2, 1, 0, 'NONE', 'not compressed'))
        wav.writeframes(b'\0\0' * 600)
    (trial / 'input/manifest.json').write_text(json.dumps({
        'audio_sha256': sha256(audio), 'source': 'fixture.mp4',
        'clip_start_seconds': 0, 'clip_duration_seconds': 600}))
    monkeypatch.setattr(preparation, 'ROOT', tmp_path)
    monkeypatch.setattr(preparation, 'TRIAL', trial)
    plan = preparation.prepare()
    assert plan['max_parallel'] == plan['max_gpu_parallel'] == 1
    steps = plan['steps']
    jobs = json.loads((trial / 'catalog.json').read_text())['jobs']
    assert [jobs[s['job']]['review']['hardware'] for s in steps] == ['T4', 'T4', 'T4']
    assert all(s['max_attempts'] == 1 for s in steps)
    assert plan['estimated_cost_ceiling_usd'] <= 4
    assert all(jobs[s['job']]['review']['input_seconds'] == 600 for s in steps)
    assert jobs['dub']['review']['model'] == 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'


def test_cost_report_excludes_other_apps_and_uses_decimal_money():
    from decimal import Decimal
    from batch_dubbing.phrase_trial_report import application_total
    rows = [{'Description': 'unrelated-user-job', 'Cost': '100'},
            {'Description': 'crispr-10min-qwen-20261030-asr', 'Cost': '.1'},
            {'description': 'crispr-10min-qwen-20261030-dub', 'cost': '.2'}]
    amount, selected = application_total(rows)
    assert amount == Decimal('.3')
    assert len(selected) == 2


def test_phase_costs_keep_failed_smoke_separate_from_verified_workload(tmp_path):
    from batch_dubbing.phrase_trial_costs import breakdown
    from modal_control.core import Store
    trial = tmp_path / 'trial'
    store = Store(tmp_path / 'state')
    parent = store.create('batch:test', {'release_plan': {'payload': {
        'spec_path': str((trial / 'batch.json').resolve())}}})
    child = store.create('batch-step:translation', {'batch_run': parent, 'step': 'translation'})
    store.event(child, 'process_started', {'stage': 'smoke'})
    store.event(child, 'cloud_apps', {'owned_ids': ['ap-failed']})
    store.event(child, 'process_exited', {'seconds': 4, 'returncode': 1})
    store.event(child, 'process_started', {'stage': 'workload'})
    store.event(child, 'cloud_apps', {'owned_ids': ['ap-ok']})
    store.event(child, 'artifacts_verified', {})
    result = breakdown(trial, store.root, [
        {'object_id': 'ap-failed', 'cost': '.1'},
        {'object_id': 'ap-ok', 'cost': '.2'},
        {'object_id': 'ap-unattributed', 'cost': '.05'}])
    assert result['by_phase_usd'] == {
        'translation/smoke/failed_or_cancelled': '0.1',
        'translation/workload/verified': '0.2'}
    assert result['unassigned_rows'] == [{'object_id': 'ap-unattributed', 'cost': '.05'}]
    assert breakdown(trial, tmp_path / 'no-state', []) is None


def test_billing_snapshot_alone_cannot_claim_a_completed_video(tmp_path):
    from batch_dubbing.phrase_trial_report import report
    (tmp_path / 'input').mkdir()
    (tmp_path / 'input/manifest.json').write_text(json.dumps({'extraction_seconds': 3}))
    (tmp_path / 'billing_before_snapshot.json').write_text(json.dumps({
        'observed_at': '2026-10-30T00:00:00Z', 'summary': {}, 'report': []}))
    result = report(tmp_path)
    assert result['status'] == 'incomplete'
    assert result['billing_final'] is False
    assert result['application_cost_usd_so_far'] == '0'
    assert len(result['issues']) == 7


def test_batch_timing_uses_matching_controller_run(tmp_path):
    from batch_dubbing.phrase_trial_report import batch_timing
    from modal_control.core import Store
    store = Store(tmp_path / 'state')
    trial = tmp_path / 'trial'
    assert batch_timing(trial, store.root) is None
    rid = store.create('batch:test', {'release_plan': {'payload': {
        'spec_path': str((trial / 'batch.json').resolve())}}})
    with store.db() as connection:
        connection.execute('UPDATE runs SET created=?,updated=?,status=? WHERE id=?',
                           ('2026-10-30T10:00:00+00:00', '2026-10-30T10:02:30+00:00', 'succeeded', rid))
    result = batch_timing(trial, store.root)
    assert result['elapsed_seconds'] == 150
    assert result['status'] == 'succeeded'
    assert batch_timing(tmp_path / 'different', store.root) is None


def test_clip_duration_rejects_full_video():
    from batch_dubbing.phrase_trial import validate_source_duration
    assert validate_source_duration(600) == 600
    for value in (600.01, 2209.96, float('nan'), True):
        with pytest.raises(ValueError):
            validate_source_duration(value)


def test_qwen_voice_mapping_is_stock_and_stable():
    from batch_dubbing.phrase_trial_qwen import voice_assignments
    units = [{'speaker': speaker} for speaker in ('host', 'guest', 'unknown')]
    data = {'voices': {'host': {'voice': 'M1'}, 'guest': {'voice': 'F1'}}}
    voices = voice_assignments(data, units)
    assert voices['host']['voice'] == 'Ryan'
    assert voices['guest']['voice'] == 'Serena'
    assert voices == voice_assignments(data, units[::-1])
    assert all(v['identity_cloning'] is False for v in voices.values())


def test_qwen_refuses_longer_input_before_loading_model():
    from batch_dubbing.phrase_trial_qwen import synthesize_qwen
    speech = speech_fixture()
    data = bind_translations(speech, translated_rows(speech))
    data['duration_seconds'] = 601
    with pytest.raises(ValueError, match='ten-minute'):
        synthesize_qwen(data, 601, emit=lambda *a, **k: None, commit=lambda: None)


def test_adaptive_layout_split_preserves_every_source_word_and_time():
    from batch_dubbing.phrase_trial import split_phrase_for_layout
    speech = speech_fixture()
    original = phrase_rows(speech)
    target = next(row for row in original if len(row['source_word_indices']) > 2)
    revised = split_phrase_for_layout(speech, target)
    assert len(revised) == len(original)+1
    assert [i for row in revised for i in row['source_word_indices']] == list(range(len(speech['source_words'])))
    assert revised[:target['id']-1] == original[:target['id']-1]
    assert all(a['end_ms'] <= b['start_ms'] for a, b in zip(revised, revised[1:]))
    translated = validate_translations(json.dumps({'phrases': [
        {'id': row['id'], 'de': 'Deutsch', 'fr': 'Français'} for row in revised]}), revised)
    result = bind_translations(speech, translated)
    assert len(result['speech_units']['de']) == len(speech['source_units'])
    assert result['phrase_split_before_word_indices'] == speech['phrase_split_before_word_indices']


def test_adaptive_split_cannot_drop_or_divide_a_single_source_word():
    from batch_dubbing.phrase_trial import split_phrase_for_layout
    speech = speech_fixture()
    with pytest.raises(ValueError, match='Single source word'):
        split_phrase_for_layout(speech, {'source_word_indices': [0]})
