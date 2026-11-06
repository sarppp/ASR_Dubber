from pathlib import Path
import json
import subprocess
import wave
import numpy as np
import pytest
from batch_dubbing.aligned_finish import source_plan,finish,sha,read_srt,write_srt
from batch_dubbing.anchored_timeline import packet_hash


def fixtures(tmp_path):
    video=tmp_path/'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=blue:s=160x90:r=10:d=5','-c:v','libx264',str(video)],check=True)
    source=tmp_path/'en.srt';target=tmp_path/'de.srt'
    write_srt(source,[dict(start=.3,end=1.5,text='One complete sentence.'),dict(start=3.,end=4.3,text='Another complete sentence.')])
    write_srt(target,[dict(start=.3,end=1.5,text='Ein ganzer Satz.'),dict(start=3.,end=4.3,text='Noch ein Satz.')])
    plan=source_plan(source,target,{'en':source,'de':target},'en','de')
    takes=[]
    for passage in plan['passages']:
        audio=tmp_path/f'{passage["index"]}.wav';rate=24000
        samples=np.r_[np.zeros(2400),.2*np.sin(2*np.pi*220*np.arange(24000)/rate),np.zeros(2400)]
        with wave.open(str(audio),'wb') as w:
            w.setparams((1,2,rate,0,'NONE','not compressed'));w.writeframes((samples*32767).astype('<i2').tobytes())
        takes.append(dict(index=passage['index'],file=str(audio),text=passage['text'],tempo=1.,sha256=sha(audio)))
    return dict(video=str(video),output=str(tmp_path/'out'),plan=plan,takes=takes)


def align(path,text,lang):
    return [dict(text=t,start=.1+i*.3,end=.1+(i+1)*.3) for i,t in enumerate(text.split())]


def test_integrated_finish_preserves_picture_pauses_and_combines_final_timings(tmp_path):
    request=fixtures(tmp_path);report=finish(request,align)
    out=Path(request['output'])
    assert report['status']=='complete' and report['video_seconds']==5
    assert packet_hash(out/'final_dub.mp4')==packet_hash(request['video'])
    assert all(p['all_active_samples_retained'] for p in report['passages'])
    assert report['passages'][1]['first_word']==pytest.approx(3.,abs=.001)
    assert report['passages'][1]['start']-report['passages'][0]['end']>1
    german=read_srt(out/'final_dub.de.srt');english=read_srt(out/'final_dub.en.srt');combined=read_srt(out/'final_dub.multi.srt')
    assert [(r['start'],r['end']) for r in german]==[(r['start'],r['end']) for r in english]
    assert 'Ein ganzer Satz.' in combined[0]['text'] and 'One complete sentence.' in combined[0]['text']
    assert (out/'final_dub.multi.ass').is_file()
    assert (out/'final_dub.multi.srt').read_bytes().startswith(b'\xef\xbb\xbf')
    streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(out/'final_dub.mp4')]))['streams']
    subtitles=[s for s in streams if s['codec_type']=='subtitle']
    assert [(s['tags']['language'],s['disposition']['default']) for s in subtitles]==[('eng',1),('deu',0)]
    assert next(s for s in streams if s['codec_type']=='audio')['tags']['language']=='deu'


def test_no_partial_final_video_when_a_word_is_missing(tmp_path):
    request=fixtures(tmp_path)
    with pytest.raises(ValueError,match='entire'):
        finish(request,lambda p,t,l:align(p,t,l)[:-1])
    assert not (Path(request['output'])/'final_dub.mp4').exists()
    assert json.loads((Path(request['output'])/'alignment_report.json').read_text())['status']=='needs_review'


def test_changed_raw_take_rejected_before_alignment(tmp_path):
    request=fixtures(tmp_path);Path(request['takes'][0]['file']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):finish(request,lambda *a:pytest.fail('must not align'))


def test_source_plan_cannot_drop_sentences(tmp_path):
    request=fixtures(tmp_path)
    short=tmp_path/'short.srt';write_srt(short,[dict(start=.3,end=1.5,text='Ein Satz.')])
    with pytest.raises(ValueError,match='Every source cue'):source_plan(tmp_path/'en.srt',short,{},'en','de')


def test_native_overrun_blocks_publication(tmp_path):
    request=fixtures(tmp_path)
    request['plan']['passages'][-1]['source_start']=4.6
    request['plan']['passages'][-1]['source_end']=4.9
    with pytest.raises(ValueError,match='cannot fit'):finish(request,align)
    assert not (Path(request['output'])/'final_dub.mp4').exists()


def test_complete_passages_not_arbitrary_subtitle_chunks():
    from batch_dubbing.aligned_finish import passage_spans
    cues=[dict(start=i,end=i+.9,text='words continue,') for i in range(8)]
    cues[-1]['text']='the sentence ends.'
    assert passage_spans(cues)==[(0,7)]
    with pytest.raises(ValueError,match='review'):
        passage_spans([dict(start=0,end=31,text='Still speaking.')])


def test_introductory_clause_keeps_its_main_clause():
    from batch_dubbing.aligned_finish import passage_spans
    assert passage_spans([dict(start=0,end=2,text='When we published our research.'),
                          dict(start=2,end=4,text='It was clear this mattered.')])==[(0,1)]


def test_t4_worker_calls_actual_default_runner_with_tower():
    import ast
    source=Path(__file__).resolve().parents[1]/'batch_dubbing/tower_pipeline_t4_worker.py'
    tree=ast.parse(source.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pipeline_command')
    scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'test','exec'),scope)
    cmd=scope['pipeline_command']('/python')
    assert cmd[2]=='/app/run_pipeline.py'
    for flag,value in [('--pipeline','subtitle-dubbing'),('--asr-model','parakeet-ultra'),('--translator','tower'),('--dub-retries','0')]:
        assert cmd[cmd.index(flag)+1]==value
    decorator=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='function')
    settings={k.arg:ast.literal_eval(k.value) for k in decorator.keywords if k.arg in {'gpu','max_containers','min_containers','retries','timeout'}}
    assert settings==dict(gpu='T4',max_containers=1,min_containers=0,retries=0,timeout=600)


def test_source_word_timeline_is_used_before_synthesis(tmp_path):
    fixtures(tmp_path)
    timing=tmp_path/'source-timing.json'
    words=[dict(word=t,start=base+i*.25,end=base+(i+1)*.25) for base,text in [(.4,'One complete sentence.'),(3.1,'Another complete sentence.')] for i,t in enumerate(text.split())]
    timing.write_text(json.dumps(dict(words=words)))
    plan=source_plan(tmp_path/'en.srt',tmp_path/'de.srt',{},'en','de',source_timing=timing)
    assert plan['boundary_evidence']=='ASR word timeline'
    assert plan['passages'][0]['source_start']==.4
    assert plan['passages'][1]['source_end']==3.85
    words.pop();timing.write_text(json.dumps(dict(words=words)))
    with pytest.raises(ValueError,match='Source words'):
        source_plan(tmp_path/'en.srt',tmp_path/'de.srt',{},'en','de',source_timing=timing)


def test_unbudgeted_translation_keeps_complete_source_passage():
    from local_mt import translate_cues
    seen=[]
    class Translator:
        def translate_sentence(self,text,source,target,max_chars=None):
            seen.append((text,max_chars))
            return 'Als wir es veröffentlichten, hofften wir auf eine große Wirkung.'
    cues=[dict(start=0,end=4,text='When we first published it, it was a transformation.'),
          dict(start=4,end=8,text='Transformative research and we hoped for an impact.')]
    result=translate_cues(cues,'en','de',Translator(),fit_budget=False)
    assert len(seen)==1 and seen[0][1] is None
    assert seen[0][0].endswith('we hoped for an impact.')
    assert len(result)==2


def test_source_complete_mode_never_calls_unverified_shortener(tmp_path):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    import pipeline_subtitle as ps
    source=tmp_path/'en.srt';target=tmp_path/'de.srt'
    write_srt(source,[dict(start=0,end=2,text='It improves research and we hoped for an impact.')])
    text='Es verbessert die Forschung und wir hofften auf eine große Wirkung.'
    write_srt(target,[dict(start=0,end=2,text=text)])
    class Forbidden:
        def __getattr__(self,name):raise AssertionError('No model rewrite may silently drop a clause')
    _,rows,spoken=ps._adapt(target,source,'en','de',Forbidden(),target=1.,cps=1.,compressor=Forbidden(),allow_rewrite=False)
    assert rows[0]['text_after']==text
    assert read_srt(spoken)[0]['text']==text


def test_final_alignment_preserves_paired_cuts_and_line_breaks(tmp_path):
    request=fixtures(tmp_path)
    source=tmp_path/'en.srt';target=tmp_path/'de.srt';spoken=tmp_path/'spoken.srt'
    source_rows=[dict(start=.3,end=.8,text='Original first'),dict(start=.8,end=1.5,text='source cue.')]
    target_rows=[dict(start=.3,end=.8,text='Ein\nSatz'),dict(start=.8,end=1.5,text='endet hier.')]
    write_srt(source,source_rows);write_srt(target,target_rows)
    write_srt(spoken,[dict(start=.3,end=1.5,text='Ein Satz endet hier.')])
    request['plan']=source_plan(source,spoken,{'en':source,'de':target},'en','de')
    request['takes']=request['takes'][:1];request['takes'][0]['text']='Ein Satz endet hier.'
    def words(path,text,language):
        return [dict(text=t,start=.1+i*.2,end=.1+(i+1)*.2) for i,t in enumerate(text.split())]
    report=finish(request,words);out=Path(request['output'])
    assert report['caption_contract_verified'] and report['cue_count']==2
    assert [r['text'] for r in read_srt(out/'final_dub.en.srt')]==[r['text'] for r in source_rows]
    assert [r['text'] for r in read_srt(out/'final_dub.de.srt')]==[r['text'] for r in target_rows]
    assert b'\r\n' in (out/'final_dub.de.srt').read_bytes()


def test_changed_frozen_caption_text_blocks_publication(tmp_path):
    request=fixtures(tmp_path)
    request['plan']['passages'][0]['caption_cues'][0]['texts']['de']='Words were silently replaced.'
    with pytest.raises(ValueError,match='tokens do not match'):
        finish(request,align)
    assert not (Path(request['output'])/'final_dub.mp4').exists()


def test_pipeline_gather_uses_selected_directories(tmp_path,monkeypatch):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    import pipeline_paths
    seen=[]
    monkeypatch.setattr(pipeline_paths,'_run',lambda *args,**kwargs:seen.append(kwargs))
    pipeline_paths._finalize_outputs('review',nemo_dir=tmp_path/'input',end_product_dir=tmp_path/'result')
    assert seen[0]['env']['INPUT_DIR']==str(tmp_path/'input')
    assert seen[0]['env']['OUTPUT_DIR']==str(tmp_path/'result')
