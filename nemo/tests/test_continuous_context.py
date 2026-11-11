from pathlib import Path
import wave
import pytest
import numpy as np
from batch_dubbing.continuous_context import bundle_segments,expand_continuous
from batch_dubbing.aligned_finish import sha


def test_context_keeps_complete_passages_and_bounds_long_videos():
    segments=[dict(index=i+1,start=i*10,end=i*10+9,text=f'Complete sentence {i}.',speaker='one') for i in range(6)]
    bundles=bundle_segments(segments)
    assert [b['passage_indices'] for b in bundles]==[[1,2,3],[4,5,6]]
    assert bundles[0]['text']=='Complete sentence 0. Complete sentence 1. Complete sentence 2.'
    segments[1]['speaker']='two'
    assert bundle_segments(segments)[0]['passage_indices']==[1]


def fixture(tmp_path):
    rate=24000
    signal=(5000*np.sin(2*np.pi*220*np.arange(rate)/rate)).astype('<i2')
    samples=np.r_[np.zeros(2400,dtype='<i2'),signal,np.zeros(4800,dtype='<i2'),signal,np.zeros(2400,dtype='<i2')]
    audio=tmp_path/'raw.wav'
    with wave.open(str(audio),'wb') as w:w.setparams((1,2,rate,0,'NONE','not compressed'));w.writeframes(samples.tobytes())
    plan=dict(target_language='de',passages=[dict(index=1,text='Erster ganzer Satz.'),dict(index=2,text='Zweiter ganzer Satz.')])
    text=' '.join(p['text'] for p in plan['passages'])
    words=[dict(text=t,start=base+i*.3,end=base+(i+1)*.3) for base,p in zip([.1,1.3],plan['passages']) for i,t in enumerate(p['text'].split())]
    request=dict(plan=plan,bundles=[[1,2]],takes=[dict(index=1,file=str(audio),text=text,tempo=1.,sha256=sha(audio))])
    return request,words,samples


def test_align_once_then_partition_retains_every_pcm_sample(tmp_path):
    request,words,samples=fixture(tmp_path);calls=[]
    def align(path,text,lang):calls.append(text);return words
    expanded,mapped,evidence=expand_continuous(request,align,tmp_path)
    assert calls==[request['takes'][0]['text']]
    decoded=[]
    for take in expanded['takes']:
        with wave.open(take['file']) as w:decoded.append(np.frombuffer(w.readframes(w.getnframes()),dtype='<i2'))
        assert len(mapped[take['file']])==3
    assert np.array_equal(np.concatenate(decoded),samples)
    assert evidence[0]['partition_preserves_every_sample']


def test_no_boundary_inside_audible_speech(tmp_path):
    request,words,samples=fixture(tmp_path)
    words[2]['end']=.8;words[3]['start']=1.0
    with pytest.raises(ValueError,match='contains speech'):
        expand_continuous(request,lambda *a:words,tmp_path)


def test_context_cannot_drop_a_source_passage(tmp_path):
    request,words,_=fixture(tmp_path);request['bundles']=[[1]]
    with pytest.raises(ValueError,match='cover every source passage'):
        expand_continuous(request,lambda *a:words,tmp_path)


def test_punctuation_is_preserved_without_inventing_an_acoustic_word():
    from batch_dubbing.aligned_finish import checked_words
    words=[dict(text='Hallo',start=.1,end=.4),dict(text='Welt',start=.5,end=.9)]
    result=checked_words(words,'Hallo – Welt.',1.)
    assert [w['text'] for w in result]==['Hallo –','Welt.']
    with pytest.raises(ValueError,match='entire'):
        checked_words(words[:1],'Hallo – Welt.',1.)
