"""Long-timeline and resumability tests: model callbacks are synthetic."""
import array
import hashlib
import json
import math
from pathlib import Path
import subprocess
from unittest.mock import Mock
import wave
import pytest
from batch_dubbing.anchored_timeline import plan_clips,render,RATE,packet_hash
from batch_dubbing.natural_pipeline import run_pipeline
from batch_dubbing.passage_batch import prepare_asset,run_batch
from batch_dubbing.speech_units import build_speech_units


def asset(name='one',duration=3600.,text='A complete sentence.'):
    words=[dict(word=text,start=.3,end=1.3,speaker='A')]
    turns=[dict(start=0.,end=2.,speaker='A')]
    source=build_speech_units(words,turns,source_id=name)
    return dict(id=name,source_id=name,duration_seconds=duration,source_words=words,turns=turns,
                translations={u.id:'Ein vollständiger Satz.' for u in source})


def speech(unit,text,seconds=.95):
    rate=44100;values=array.array('f',[0.])*4410
    values.extend(.2*math.sin(2*math.pi*220*i/rate) for i in range(round(seconds*rate)))
    values.extend(array.array('f',[0.])*4410)
    return dict(samples=values,sample_rate=rate,voice='fixture')


def content(unit,take):
    # Test double only; never used as real semantic validation.
    return dict(status='passed',unit_id=unit['id'],source_id=unit['source_id'],
        source_text_sha256=hashlib.sha256(unit['source_text'].encode()).hexdigest(),
        text_sha256=hashlib.sha256(take['text'].encode()).hexdigest(),audio_sha256=take['sha256'],
        speech_coverage_verified=True,meaning_verified=True,method='synthetic-test-double')


def strict(tmp_path,data=None,synth=speech,check=content):
    data=data or asset()
    return run_pipeline(prepare_asset(data),duration_seconds=data['duration_seconds'],output_dir=tmp_path/'attempt',
        synthesize=synth,max_synthesis_calls=1,max_rate=1,max_end_gap=.4,end_gap_policy='block',expected_source_words=data['source_words'],
        require_complete_passages=True,content_check=check,require_content_check=True)


def test_59_hour_timelines_have_no_cumulative_drift_or_shortening():
    for video in range(59):
        clips=[dict(file=f'{video}-{i}',start=i*6+.3,end=i*6+1.3,frames=round(.99*RATE)) for i in range(600)]
        plan=plan_clips(clips,3600.)
        assert plan['total_frames']==3600*RATE
        assert len(plan['clips'])==600
        assert all(p['start_frame']==round(c['start']*RATE) for p,c in zip(plan['clips'],clips))
        assert plan['clips'][-1]['start_frame']==round(3594.3*RATE)
        assert plan['clips'][-1]['end_frame']+plan['trailing_frames']==3600*RATE


@pytest.mark.parametrize('clips',[
    [dict(start=0,end=1,frames=2*RATE),dict(start=1,end=2,frames=RATE)],
    [dict(start=2,end=3,frames=RATE),dict(start=1,end=2,frames=RATE)],
    [dict(start=0,end=100,frames=RATE)],
])
def test_bad_timeline_cannot_shift_or_cut(clips):
    with pytest.raises(ValueError):plan_clips(clips,10)


def test_full_picture_preserved_when_last_subtitle_ends_early(tmp_path):
    video=tmp_path/'source.mp4';audio=tmp_path/'speech.wav';out=tmp_path/'out';temp=tmp_path/'temp';temp.mkdir()
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=blue:s=160x90:r=10:d=5','-c:v','libx264',str(video)],check=True)
    with wave.open(str(audio),'wb') as wav:
        wav.setparams((1,2,RATE,0,'NONE','not compressed'));wav.writeframes(b'\x10\x00'*RATE)
    final,positions=render([(audio,1.,2.)],video,out,temp)
    assert packet_hash(final)==packet_hash(video)
    with wave.open(str(out/'dub_track.wav')) as wav:assert wav.getnframes()==5*RATE
    assert positions==[(1.,2.,1.,2.)]
    with pytest.raises(ValueError,match='trimming'):render([(audio,1.,2.)],video,out,temp,trim_to=2)
    with pytest.raises(FileNotFoundError):render([(tmp_path/'missing.wav',1.,2.)],video,out,temp)


def test_incomplete_final_passage_is_reported_before_synthesis(tmp_path):
    synth=Mock(side_effect=speech)
    report=strict(tmp_path,asset(text='An unfinished phrase'),synth=synth)
    assert report['status']=='needs_source_review'
    assert not report['render']['rows']
    synth.assert_not_called()


@pytest.mark.parametrize('seconds',[.25,1.7])
def test_short_or_long_speech_blocks_instead_of_stretching(tmp_path,seconds):
    report=strict(tmp_path,synth=lambda u,t:speech(u,t,seconds))
    assert report['status']=='needs_revision'
    assert report['render']['rows']==[]
    assert report['synthesis_calls']==1
    assert Path(report['attempts'][0]['file']).exists()


def test_missing_semantic_or_speech_evidence_cannot_publish(tmp_path):
    report=strict(tmp_path,check=lambda u,t:{})
    assert report['status']=='needs_content_review' and report['render']['rows']==[]


def test_beginning_and_ending_word_omissions_are_rejected(tmp_path):
    data=asset();units=prepare_asset(data)
    for expected in ([dict(word='Lost.',start=.01,end=.1,speaker='A')]+data['source_words'],
                     data['source_words']+[dict(word='Lost.',start=2.1,end=2.2,speaker='A')]):
        synth=Mock()
        with pytest.raises(ValueError,match='coverage'):
            run_pipeline(units,duration_seconds=3600,output_dir=tmp_path/'unused',synthesize=synth,
                         max_synthesis_calls=1,expected_source_words=expected)
        synth.assert_not_called()


def test_resume_reuses_speech_and_failure_does_not_publish(tmp_path):
    synth=Mock(side_effect=speech);publish=Mock(return_value={'verified':True})
    items=[asset('good'),asset('bad')]
    def quality(unit,take):return content(unit,take) if unit['source_id']=='good' else {}
    for iteration in range(2):
        report=run_batch(items,output_dir=tmp_path,synthesize=synth,content_check=quality,
                         provider_identity={'model':'mock','revision':'test','voice':'mock'},render_ready=publish)
        assert [r['status'] for r in report['assets']]==['complete','needs_content_review']
        assert report['actual_synthesis_calls']==(2 if iteration==0 else 0)
        assert report['cache_hits']==(0 if iteration==0 else 2)
    assert synth.call_count==2 and publish.call_count==2


def test_source_turns_and_word_coverage_survive_many_passages():
    data=asset();data['source_words']=[];data['turns']=[]
    for i in range(600):
        voice='A' if i%2 else 'B'
        data['source_words'].extend([dict(word='A',start=i*6+.2,end=i*6+.5,speaker=voice),
                                    dict(word='sentence.',start=i*6+.5,end=i*6+1.2,speaker=voice)])
        data['turns'].append(dict(start=i*6,end=i*6+2,speaker=voice))
    source=build_speech_units(data['source_words'],data['turns'],source_id=data['source_id'])
    data['translations']={u.id:'Ein Satz.' for u in source}
    units=prepare_asset(data)
    assert len(units)==600
    assert [i for u in units for i in u['source_word_indices']]==list(range(1200))
    assert all(u['source_text']=='A sentence.' for u in units)


def test_default_batch_publisher_keeps_picture_and_aligns_target_subtitles(tmp_path):
    from batch_dubbing.passage_batch import digest_file
    source=tmp_path/'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=green:s=160x90:r=10:d=5','-c:v','libx264',str(source)],check=True)
    data=asset(duration=5);data.update(video=str(source),video_sha256=digest_file(source))
    def quality(unit,take):
        return dict(**content(unit,take),aligned_words=[dict(text='Ein',start=.1,end=.3),
            dict(text='vollständiger',start=.3,end=.8),dict(text='Satz',start=.8,end=1.05)])
    report=run_batch([data],output_dir=tmp_path/'run',synthesize=speech,content_check=quality,
                     provider_identity={'model':'fixture','voice':'fixture','revision':'v1'})
    assert report['status']=='complete',report
    result=report['assets'][0]['rendered']
    assert packet_hash(result['video'])==packet_hash(source)
    text=Path(result['target_subtitle']).read_text(encoding='utf-8-sig')
    assert 'Ein vollständiger Satz.' in text
    assert '00:00:00,300 --> 00:00:01,250' in text


def test_editing_one_passage_only_regenerates_that_passage(tmp_path):
    from copy import deepcopy
    first=asset('first');second=asset('second');synth=Mock(side_effect=speech)
    args=dict(output_dir=tmp_path,synthesize=synth,content_check=content,provider_identity='test-pinned-model',render_ready=lambda *a:{})
    run_batch([first,second],**args)
    second=deepcopy(second);key=next(iter(second['translations']));second['translations'][key]='Ein anderer Satz.'
    result=run_batch([first,second],**args)
    assert result['actual_synthesis_calls']==1 and result['cache_hits']==1 and synth.call_count==3


def test_corrupt_raw_cache_is_reported_without_another_model_call(tmp_path):
    synth=Mock(side_effect=speech)
    args=dict(output_dir=tmp_path,synthesize=synth,content_check=content,provider_identity='pinned',render_ready=lambda *a:{})
    run_batch([asset()],**args)
    next(tmp_path.glob('*/raw_cache/*/samples.f32')).write_bytes(b'broken')
    result=run_batch([asset()],**args)
    assert result['assets'][0]['status']=='failed'
    assert 'Corrupt' in result['assets'][0]['error']
    assert synth.call_count==1


def test_stale_content_receipt_cannot_publish(tmp_path):
    def stale(unit,take):return dict(content(unit,take),audio_sha256='old')
    report=strict(tmp_path,check=stale)
    assert report['status']=='needs_content_review' and report['render']['rows']==[]


def test_production_short_take_guard():
    from batch_dubbing.anchored_timeline import validate_end_timing
    sidecar=dict(tempo=1.,edges=dict(sample_rate=24000,start_sample=0,last_active_end_sample=24000))
    with pytest.raises(ValueError,match='ends'):validate_end_timing(sidecar,10,12)
    assert validate_end_timing(sidecar,10,11.2)==pytest.approx(.2)


def test_malformed_srt_is_never_silently_skipped_in_strict_mode(tmp_path):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'qwen3-tts'))
    from dub_srt import parse_srt
    path=tmp_path/'bad.srt';path.write_text('1\n00:00:00,000 --> 00:00:01,000\nGood.\n\n2\nbroken time\nImportant words.\n')
    with pytest.raises(ValueError,match='timestamps'):parse_srt(path,strict=True)


def test_offline_manifest_preflight_59_assets_and_incomplete_input():
    from batch_dubbing.passage_batch import audit_batch
    assets=[asset(str(i)) for i in range(59)]
    report=audit_batch(assets)
    assert report['duration_seconds']==59*3600
    assert report['status']=='ready_for_synthesis' and report['model_calls']==0
    assets[-1]=asset('58',text='Unfinished input')
    report=audit_batch(assets)
    assert report['status']=='needs_source_review' and report['assets'][-1]['source_issues']


def test_batch_manifest_loaded_one_asset_at_a_time(tmp_path):
    data=asset('isolated');path=tmp_path/'source.json';path.write_text(json.dumps(data))
    result=run_batch([dict(id='isolated',manifest=str(path))],output_dir=tmp_path/'output',synthesize=speech,
                     content_check=content,provider_identity='mock',render_ready=lambda *a:{'test_only':True})
    assert result['status']=='complete' and result['actual_synthesis_calls']==1


def test_missing_video_rejected_before_synthesis(tmp_path):
    synth=Mock(side_effect=speech)
    data=asset();data.update(video=str(tmp_path/'missing.mp4'),video_sha256='0'*64)
    result=run_batch([data],output_dir=tmp_path/'out',synthesize=synth,
                     content_check=content,provider_identity='mock')
    assert result['assets'][0]['status']=='failed'
    synth.assert_not_called()


def test_receipt_for_different_source_text_cannot_publish(tmp_path):
    def stale(unit,take):return dict(content(unit,take),source_text_sha256='old-source')
    report=strict(tmp_path,check=stale)
    assert report['status']=='needs_content_review' and report['render']['rows']==[]


def test_short_sentence_warns_without_filling_pause_or_moving_next_passage(tmp_path):
    data=asset(duration=13.)
    data['source_words'].append(dict(word='Another complete sentence.',start=10.3,end=11.3,speaker='A'))
    data['turns'].append(dict(start=10.,end=12.,speaker='A'))
    units=build_speech_units(data['source_words'],data['turns'],source_id=data['source_id'])
    data['translations']={u.id:'Ein vollständiger Satz.' for u in units}
    revision=Mock()
    report=run_pipeline(prepare_asset(data),duration_seconds=13.,output_dir=tmp_path/'attempt',
        synthesize=lambda u,t:speech(u,t,.25),revise=revision,max_synthesis_calls=4,max_revisions_per_unit=1,
        max_rate=1.,max_end_gap=.4,expected_source_words=data['source_words'],
        content_check=content,require_content_check=True)
    assert report['status']=='ready' and report['end_gap_policy']=='warn'
    assert report['synthesis_calls']==2
    revision.assert_not_called()
    plans=report['turn_plans'];rows=report['render']['rows']
    assert all(p['diagnostics']['warnings'][0]['reason']=='speech_ends_early' for p in plans)
    assert all(r['tempo']==1. for r in rows)
    assert abs(plans[1]['schedule'][0]['start_shift'])<=.25
    assert rows[1]['start']-rows[0]['end']>8.


def test_warning_policy_still_rejects_overlap_and_missing_content(tmp_path):
    args=dict(output_dir=tmp_path/'out',provider_identity='mock',render_ready=lambda *a:{})
    failed=run_batch([asset()],synthesize=lambda u,t:speech(u,t,1.7),content_check=content,**args)
    assert failed['assets'][0]['status']=='needs_revision'
    args['output_dir']=tmp_path/'content'
    failed=run_batch([asset()],synthesize=lambda u,t:speech(u,t,.25),content_check=lambda *a:{},**args)
    assert failed['assets'][0]['status']=='needs_content_review'
