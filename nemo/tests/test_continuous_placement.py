import json
from pathlib import Path
import wave

import pytest

from batch_dubbing.aligned_finish import finish, sha, read_srt
from test_aligned_finish import fixtures, align


def continuous_fixture(tmp_path):
    request=fixtures(tmp_path)
    audio=tmp_path/'whole.wav';frames=[]
    for take in request['takes']:
        with wave.open(take['file']) as wav:frames.append(wav.readframes(wav.getnframes()))
    with wave.open(str(audio),'wb') as wav:
        wav.setparams((1,2,24000,0,'NONE','not compressed'));wav.writeframes(b''.join(frames))
    passages=request['plan']['passages']
    words=[dict(text=t,start=base+i*.3,end=base+(i+1)*.3) for base,p in zip([.1,1.3],passages) for i,t in enumerate(p['text'].split())]
    request.update(placement_mode='continuous',bundles=[[1,2]],takes=[dict(index=1,file=str(audio),sha256=sha(audio),tempo=1.,text=' '.join(p['text'] for p in passages))])
    return request,words,b''.join(frames)


def test_continuous_delivery_preserves_every_sample_and_subtitle_onset(tmp_path):
    request,words,raw=continuous_fixture(tmp_path)
    report=finish(request,lambda *args:words)
    out=Path(request['output'])
    with wave.open(str(out/'dub_track.wav')) as wav:actual=wav.readframes(wav.getnframes())
    # Starts at the first original source cue (0.3); no sentence reanchoring,
    # no recording-edge trimming and no forced trailing padding.
    assert actual==bytes(round(.3*24000)*2)+raw
    cues=read_srt(out/'final_dub.de.srt')
    assert cues[1]['start']==pytest.approx(.3+1.3-.08)
    assert report['passages'][0]['all_raw_samples_retained']
    assert any(w['reason']=='continuous_delivery_differs_from_source_passage' for w in report['warnings'])
    assert report['video_seconds']==5.


def test_context_overrun_fails_instead_of_delaying_following_context(tmp_path):
    request=fixtures(tmp_path);request.update(placement_mode='continuous',bundles=[[1],[2]])
    raw=Path(request['takes'][0]['file'])
    with wave.open(str(raw)) as wav:frames=wav.readframes(wav.getnframes())
    with wave.open(str(raw),'wb') as wav:
        wav.setparams((1,2,24000,0,'NONE','not compressed'));wav.writeframes(frames*3)
    request['takes'][0]['sha256']=sha(raw)
    with pytest.raises(ValueError,match='cannot fit'):
        finish(request,align)
    report=json.loads((Path(request['output'])/'alignment_report.json').read_text())
    assert report['failed_placement']['required_onset_shift']==0
    assert not (Path(request['output'])/'final_dub.mp4').exists()


def test_distinct_contexts_keep_absolute_starts(tmp_path):
    request=fixtures(tmp_path);request.update(placement_mode='continuous',bundles=[[1],[2]])
    report=finish(request,align)
    assert [p['start'] for p in report['passages']]==[.3,3.]


def test_synced_contexts_realign_after_adjustment_without_filling_source_pause(tmp_path):
    from batch_dubbing.sync_continuous_dub import read_pcm,last_activity
    request=fixtures(tmp_path);request.update(placement_mode='continuous_synced',bundles=[[1],[2]])
    # First source window: 0.3..1.5. Second: 3.0..4.2. They must stay independent.
    request['plan']['passages'][1]['context_end']=4.2
    calls=[]
    def measured(path,text,lang):
        calls.append(str(path))
        data,rate=read_pcm(path);end=last_activity(data,rate,-40)
        return [dict(text=t,start=.1+i*(end-.1)/3,end=.1+(i+1)*(end-.1)/3) for i,t in enumerate(text.split())]
    result=finish(request,measured)
    assert len(calls)==4  # raw then adjusted for each context
    assert [p['start'] for p in result['passages']]==[.3,3.]
    assert result['passages'][0]['end']<2.
    assert result['passages'][1]['end']<4.6  # preserve the source's final silent scene
    assert all(.9<=p['effective_tempo']<=1. for p in result['passages'])
    assert all(p['duration_adjustment']['acoustic_end_gap_seconds']<=.4 for p in result['passages'])
    assert result['tempo'] is None  # never label adjusted speech as native
