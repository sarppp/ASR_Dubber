import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from unittest.mock import Mock,patch

import pytest

from batch_dubbing.dialogue_trial import (MODEL,REVISION,PACKAGE,IMAGE_ID,
    validate_scope,verify_sources,jobs,run_trial,validate_smoke)
from batch_dubbing.native_excerpt_probe import decode_wave


def fixture():
    return dict(schema=1,model=MODEL,revision=REVISION,package=PACKAGE,image_id=IMAGE_ID,speed=1.,steps=8,
        pairs=[dict(id='phrase',language='de',voice='M1',seed=7,start=1.,end=3.,
            source_text='I do not calculate it.',old_text='Ich habe es nicht berechnet.',
            adapted_text='Ich rechne das nicht aus.',adaptation_note='Restore present tense.',
            source_indices=[0],old_translation_indices=[0],source_json='source.json',
            source_sha256='a'*64,original_video='source.mp4',older_dub='old.mp4')])


def chunker(text,size):return [text]


def generate(row):return dict(samples=struct.pack('<4f',.1,.2,-.2,.1),sample_rate=44100)


def smoke(data):
    result=run_trial(data,True,chunker=chunker,synthesize=generate)
    return dict(report={k:v for k,v in result.items() if k!='audio'},wav=next(iter(result['audio'].values())))


def test_paired_jobs_keep_voice_seed_and_source_window_constant():
    old,new=jobs(fixture(),chunker)
    assert old['arm']=='old' and new['arm']=='adapted'
    assert old['text']!=new['text']
    for key in ('voice','seed','language','pair_id','target_seconds'):assert old[key]==new[key]


def test_workload_reuses_smoke_and_preserves_every_native_sample():
    data=fixture();prior=smoke(data);synth=Mock(side_effect=generate)
    result=run_trial(data,False,chunker=chunker,synthesize=synth,prior=prior)
    assert result['status']=='complete' and result['calls']==1
    assert len(result['samples'])==2 and result['samples'][0]['reused_from_smoke']
    assert result['audio']['phrase_old.wav']==prior['wav']
    assert decode_wave(result['audio']['phrase_adapted.wav'])==generate(None)['samples']
    assert result['timing_verified'] is False and result['naturalness_verified'] is False
    assert result['samples'][1]['caption_duration_difference_seconds']<0


@pytest.mark.parametrize('damage',['text','wave','scope','processing'])
def test_bad_smoke_blocks_all_new_calls(damage):
    data=fixture();prior=smoke(data)
    if damage=='text':prior['report']['samples'][0]['text']='Different text'
    if damage=='wave':prior['wav']=prior['wav'][:-4]
    if damage=='scope':prior['report']['input']['pairs'][0]['seed']=9
    if damage=='processing':prior['report']['waveform_time_stretched']=True
    synth=Mock()
    with pytest.raises(ValueError):run_trial(data,False,chunker=chunker,synthesize=synth,prior=prior)
    synth.assert_not_called()


def test_hidden_chunk_split_is_rejected_before_synthesis():
    synth=Mock()
    with pytest.raises(ValueError,match='one internal chunk'):
        run_trial(fixture(),True,chunker=lambda t,s:['first','second'],synthesize=synth)
    synth.assert_not_called()


def test_continuous_passage_allows_only_declared_text_preserving_chunks():
    data=fixture();data.update(max_text_chars=600,max_internal_chunks_per_take=2)
    p=data['pairs'][0];p['old_text']=p['adapted_text']='A complete thought. Another complete thought.'
    split=lambda text,size:['A complete thought.','Another complete thought.']
    assert len(jobs(data,split))==2
    result=run_trial(data,True,chunker=split,synthesize=generate)
    assert result['status']=='complete' and result['calls']==1 and result['internal_calls']==2
    with pytest.raises(ValueError):jobs(data,lambda text,size:['A changed thought.','Another complete thought.'])
    with pytest.raises(ValueError):jobs(data,lambda text,size:['A','complete thought.','Another complete thought.'])


def test_failure_retains_prior_audio_and_never_retries():
    data=fixture();synth=Mock(side_effect=RuntimeError('provider failed'))
    result=run_trial(data,False,chunker=chunker,synthesize=synth,prior=smoke(data))
    assert result['status']=='failed' and result['calls']==1
    assert 'phrase_old.wav' in result['audio']
    synth.assert_called_once()


@pytest.mark.parametrize('mutation',[lambda d:d.update(speed=1.1),
    lambda d:d['pairs'][0].update(source_indices=[0,2]),
    lambda d:d['pairs'][0].update(adapted_text='x'*301),
    lambda d:d['pairs'][0].update(end=float('nan')),
    lambda d:d['pairs'][0].update(source_json='../other.json')])
def test_unsafe_scope_rejected(mutation):
    data=fixture();mutation(data)
    with pytest.raises(ValueError):validate_scope(data)


def test_source_binding_prevents_rewritten_english_or_wrong_bounds(tmp_path):
    data=fixture();p=data['pairs'][0]
    source=dict(english=[dict(start=1.,end=3.,text=p['source_text'],speaker='speaker_0')],
                translations={'de':[dict(text=p['old_text'])]})
    raw=json.dumps(source).encode();(tmp_path/'source.json').write_bytes(raw)
    for name in ('source.mp4','old.mp4'):(tmp_path/name).write_bytes(b'fixture')
    p['source_sha256']=hashlib.sha256(raw).hexdigest()
    assert verify_sources(data,tmp_path)==data
    p['source_text']='Changed source'
    with pytest.raises(ValueError,match='Source-linked'):verify_sources(data,tmp_path)


def test_worker_saves_failed_partial_and_refuses_overwrite(tmp_path):
    path=Path(__file__).with_name('dialogue_trial_worker.py')
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='save_result')
    scope=dict(Path=Path,json=json,hashlib=hashlib)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    result=run_trial(fixture(),True,chunker=chunker,synthesize=generate);result['status']='failed'
    with pytest.raises(RuntimeError,match='partial'):scope['save_result'](result,tmp_path)
    assert (tmp_path/'report.json').exists() and (tmp_path/'phrase_old.wav').exists()
    with pytest.raises(FileExistsError):scope['save_result'](result,tmp_path)


def test_worker_uses_cached_model_and_preserves_native_samples():
    path=Path(__file__).with_name('dialogue_trial_worker.py')
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='probe')
    node.decorator_list=[];scope={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    class Audio:
        ndim=1;size=4
        def squeeze(self):return self
        def astype(self,*a,**k):return self
        def tobytes(self):return generate(None)['samples']
    audio=Audio();model=Mock(sample_rate=44100);model.synthesize.return_value=(audio,1.)
    tts=Mock(return_value=model);snapshot=Mock(return_value='/cache/pinned');monitor=Mock()
    captured=[]
    def run(data,smoke,*,chunker,synthesize,prior):
        captured.append(synthesize(jobs(data,chunker)[0]));return dict(status='complete',calls=1)
    helper=SimpleNamespace(MODEL=MODEL,REVISION=REVISION,jobs=jobs,validate_smoke=validate_smoke,run_trial=run)
    modules={'batch_dubbing.dialogue_trial':helper,'supertonic':SimpleNamespace(TTS=tts),
        'supertonic.utils':SimpleNamespace(chunk_text=chunker),'huggingface_hub':SimpleNamespace(snapshot_download=snapshot),
        'numpy':SimpleNamespace(asarray=Mock(return_value=audio),float32='f32',random=SimpleNamespace(seed=Mock())),
        'telemetry':SimpleNamespace(start=Mock(return_value=monitor),emit=Mock())}
    with patch.dict(sys.modules,modules),patch.dict(os.environ),patch.object(sys,'path',list(sys.path)),patch('os.sched_setaffinity'),patch('importlib.metadata.version',return_value='1.3.1'):
        assert scope['probe'](fixture(),True)['status']=='complete'
    snapshot.assert_called_once_with(MODEL,revision=REVISION,local_files_only=True)
    assert tts.call_args.kwargs['auto_download'] is False
    assert model.synthesize.call_args.kwargs['speed']==1.
    assert captured==[generate(None)]
    monitor.set.assert_called_once()
