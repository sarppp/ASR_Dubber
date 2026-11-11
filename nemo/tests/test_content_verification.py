import hashlib
from unittest.mock import Mock
import pytest
from batch_dubbing.content_verification import ContentVerifier


def fixture(tmp_path):
    audio=tmp_path/'raw.wav';audio.write_bytes(b'fixture audio')
    unit=dict(id='passage',source_id='video',source_text='There are 42 items.',source_language='en',target_language='de')
    take=dict(file=str(audio),sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),text='Es sind 42 Stück.',edges=dict(raw_seconds=2))
    return unit,take


def verifier(tmp_path,asr='Es sind 42 Stück.',meaning=None):
    recognize=Mock(return_value=asr)
    review=Mock(return_value=meaning or dict(faithful=True,omissions=[],additions=[],contradictions=[],reason='Synthetic test only'))
    align=Mock(return_value=[dict(text=t,start=i*.3,end=(i+1)*.3) for i,t in enumerate(['Es','sind','42','Stück'])])
    checker=ContentVerifier(transcribe=recognize,review_meaning=review,align=align,provider_identity='test-models',max_calls=3,cache_dir=tmp_path/'cache')
    return checker,recognize,review,align


def test_missing_spoken_number_blocks_before_alignment(tmp_path):
    unit,take=fixture(tmp_path);checker,asr,meaning,align=verifier(tmp_path,asr='Es sind Stück.')
    receipt=checker(unit,take)
    assert receipt['status']=='failed' and receipt['speech_coverage_verified'] is False
    meaning.assert_not_called();align.assert_not_called()


def test_semantic_omission_blocks_even_with_exact_spoken_text(tmp_path):
    unit,take=fixture(tmp_path)
    checker,asr,meaning,align=verifier(tmp_path,meaning=dict(faithful=False,omissions=['fact'],additions=[],contradictions=[],reason='Omission'))
    receipt=checker(unit,take)
    assert receipt['status']=='failed' and receipt['meaning_verified'] is False
    align.assert_not_called()


def test_three_independent_checks_and_hash_bound_resume(tmp_path):
    unit,take=fixture(tmp_path);checker,asr,meaning,align=verifier(tmp_path)
    first=checker(unit,take);second=checker(unit,take)
    assert first==second and first['status']=='passed'
    assert checker.calls==dict(asr=1,meaning=1,alignment=1) and checker.cache_hits==1
    assert first['audio_sha256']==take['sha256'] and first['aligned_words']


def test_source_change_does_not_reuse_previous_receipt(tmp_path):
    unit,take=fixture(tmp_path);checker,*_=verifier(tmp_path)
    checker(unit,take);unit['source_text']='There are not 42 items.'
    with pytest.raises(ValueError,match='budget'):checker(unit,take)


def test_changed_audio_is_rejected_before_any_model(tmp_path):
    unit,take=fixture(tmp_path);checker,asr,*_=verifier(tmp_path)
    from pathlib import Path
    Path(take['file']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):checker(unit,take)
    asr.assert_not_called()
