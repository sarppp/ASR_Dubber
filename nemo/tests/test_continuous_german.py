import hashlib
import io
import json
from pathlib import Path
from unittest.mock import Mock, patch
import wave
import pytest
from batch_dubbing.continuous_german import passage, synthesize, save_result
from modal_control.core import Store, stage
from modal_control.offline import verify_receipt

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'tests/fixtures/continuous_german/input.de.srt'


def fixture_inputs(destination):
    inputs={'input.de.srt':SCRIPT.read_bytes(), 'voice_reference.wav':b'mocked voice reference',
            'source_10cues.mp4':b'mocked video input'}
    for name,data in inputs.items():
        (destination/name).write_bytes(data)
    (destination/'scope.json').write_text(json.dumps({'input_sha256':{
        name:hashlib.sha256(data).hexdigest() for name,data in inputs.items()}}))


def test_single_call_preserves_every_word_without_cue_pauses():
    model=Mock()
    synthesize(model,SCRIPT,'reference.wav')
    model.generate_voice_clone.assert_called_once()
    args=model.generate_voice_clone.call_args.kwargs
    assert args['text']== ' '.join(' '.join(b.splitlines()[2:]) for b in SCRIPT.read_text(encoding='utf-8-sig').strip().split('\n\n'))
    assert '\n' not in args['text']
    assert args['text'].endswith('dass sie das Leben')
    assert args['max_new_tokens']==600
    assert args['language']=='German' and args['xvec_only'] is True


def test_bom_and_multiline_preserved(tmp_path):
    script=tmp_path/'bom.srt'
    script.write_text(SCRIPT.read_text(encoding='utf-8-sig'),encoding='utf-8-sig')
    assert passage(script)==passage(SCRIPT)
    script.write_text('1\n00:00:00,000 --> 00:00:01,000\nHello')
    with pytest.raises(ValueError,match='ten'):passage(script)


def result(seconds=12):
    out=io.BytesIO()
    with wave.open(out,'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000)
        wav.writeframes(b'\x01\x01'*24000*seconds)
    data=out.getvalue()
    return dict(audio=data,report=dict(status='complete',synthesis_calls=1,audio_seconds=seconds,audio_sha256=hashlib.sha256(data).hexdigest()))


def test_result_transport_and_no_overwrite(tmp_path):
    dest=tmp_path/'output'
    data=result()
    save_result(data,dest)
    assert (dest/'CONTINUOUS_GERMAN.wav').read_bytes()==data['audio']
    with pytest.raises(FileExistsError):save_result(data,dest)


@pytest.mark.parametrize('mutation',['hash','calls','cap','silent'])
def test_bad_results_never_publish(tmp_path,mutation):
    data=result(50 if mutation=='cap' else 12)
    if mutation=='hash':data['report']['audio_sha256']='wrong'
    if mutation=='calls':data['report']['synthesis_calls']=10
    if mutation=='silent':
        data['audio']=data['audio'][:44]+bytes(len(data['audio'])-44)
        data['report']['audio_sha256']=hashlib.sha256(data['audio']).hexdigest()
    with pytest.raises(ValueError):save_result(data,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def offline_job(tmp_path):
    code=tmp_path/'worker.py';code.write_text('# tested code')
    receipt=tmp_path/'smoke.json'
    receipt.write_text(json.dumps(dict(status='complete',kind='offline_smoke',model_calls=0,cloud_calls=0,
        checks=['single request'],source_sha256={str(code):hashlib.sha256(code.read_bytes()).hexdigest()})))
    return dict(cwd=str(tmp_path),smoke_mode='offline_receipt',source_files=['worker.py','smoke.json'],smoke_success_files=['smoke.json'])


def test_offline_controller_smoke_spawns_nothing(tmp_path):
    job=offline_job(tmp_path);store=Store(tmp_path/'state');rid=store.create('test',{})
    with patch('modal_control.core.subprocess.Popen') as process:
        stage(store,rid,job,True,{},['modal'])
        process.assert_not_called()


def test_offline_smoke_bound_to_current_sources(tmp_path):
    job=offline_job(tmp_path)
    verify_receipt(job)
    (tmp_path/'worker.py').write_text('# changed after tests')
    with pytest.raises(ValueError,match='exact current'):verify_receipt(job)


def test_offline_receipt_must_be_pinned(tmp_path):
    job=offline_job(tmp_path);job['source_files'].remove('smoke.json')
    with pytest.raises(ValueError,match='Pin exactly'):verify_receipt(job)


def test_actual_worker_body_offline(tmp_path,monkeypatch):
    import ast
    import sys
    import types
    import numpy as np
    import batch_dubbing.continuous_german as helper
    fixture_inputs(tmp_path)
    def mapped_path(*parts):
        p=Path(*parts)
        return tmp_path/p.relative_to('/input') if p.is_relative_to('/input') else p
    monkeypatch.setattr(helper, 'Path', mapped_path)
    # Execute the actual worker body without importing Modal or allocating hardware.
    tree=ast.parse((ROOT/'batch_dubbing/continuous_german_worker.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate')
    fn.decorator_list=[]
    model=Mock()
    model.generate_voice_clone.return_value=([np.ones(12*24000,dtype=np.float32)*.01],24000)
    loader=Mock();loader.from_pretrained.return_value=model
    monitor=Mock()
    modules={'continuous_german':helper,'telemetry':types.SimpleNamespace(emit=Mock(),start=lambda:monitor),
        'huggingface_hub':types.SimpleNamespace(snapshot_download=Mock(return_value='/cached/model')),
        'torch':types.SimpleNamespace(manual_seed=Mock(),bfloat16='bf16',is_tensor=lambda x:False),
        'faster_qwen3_tts':types.SimpleNamespace(FasterQwen3TTS=loader)}
    for name,value in modules.items():monkeypatch.setitem(sys.modules,name,value)
    ns=dict(Path=mapped_path,hashlib=hashlib,io=io,json=json)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_worker','exec'),ns)
    with patch('subprocess.check_output',return_value='Tesla T4'):
        output=ns['generate']()
    model.generate_voice_clone.assert_called_once()
    assert output['report']['text']==passage(SCRIPT)
    assert output['report']['synthesis_calls']==1
    save_result(output,tmp_path/'downloaded')
    monitor.set.assert_called_once()
    with patch('subprocess.check_output',return_value='NVIDIA A10G'):
        with pytest.raises(RuntimeError,match='T4'):ns['generate']()
    model.generate_voice_clone.assert_called_once()


def test_real_ffmpeg_packaging(tmp_path):
    import subprocess
    from batch_dubbing.continuous_german import package
    source=tmp_path/'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=black:s=160x90:r=10:d=1',
        '-c:v','libx264',str(source)],check=True)
    take=tmp_path/'take.wav';take.write_bytes(result()['audio'])
    subtitles=tmp_path/'display.srt'
    subtitles.write_text('1\n00:00:00,000 --> 00:00:01,000\nTest\n')
    target=tmp_path/'review.mp4'
    report=package(source,take,subtitles,target)
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(target)]))
    assert [s['codec_type'] for s in info['streams']]==['video','audio','subtitle']
    assert abs(float(info['format']['duration'])-12)<.2
    assert report['picture_hold_seconds']==11
    assert info['streams'][1]['tags']['language']=='deu'
    with pytest.raises(FileExistsError):package(source,take,SCRIPT,target)


def test_local_runner_offline(tmp_path,monkeypatch):
    import sys
    import types
    import numpy as np
    import batch_dubbing.continuous_german_local as runner
    fixture_inputs(tmp_path)
    monkeypatch.setattr(runner,'TRIAL',tmp_path)
    model=Mock()
    model.generate_voice_clone.return_value=([np.ones(12*24000,dtype=np.float32)*.01],24000)
    loader=Mock();loader.from_pretrained.return_value=model
    cuda=Mock()
    cuda.is_available.return_value=True
    cuda.mem_get_info.return_value=(5*1024**3,6*1024**3)
    cuda.get_device_name.return_value='RTX 2060'
    cuda.max_memory_allocated.return_value=4*1024**3
    modules={'torch':types.SimpleNamespace(cuda=cuda,manual_seed=Mock(),set_num_threads=Mock(),float16='fp16',is_tensor=lambda x:False),
        'faster_qwen3_tts':types.SimpleNamespace(FasterQwen3TTS=loader),
        'huggingface_hub':types.SimpleNamespace(snapshot_download=Mock(return_value='/cached/model'))}
    for name,value in modules.items():monkeypatch.setitem(sys.modules,name,value)
    runner.main()
    model.generate_voice_clone.assert_called_once()
    assert loader.from_pretrained.call_args.kwargs==dict(device='cuda:0',dtype='fp16',max_seq_len=1024)
    report=json.loads((tmp_path/'continuous_local/report.json').read_text())
    assert report['execution']=='local CUDA' and report['synthesis_calls']==1
    assert report['text']==passage(SCRIPT)
    with pytest.raises(FileExistsError):runner.main()
    model.generate_voice_clone.assert_called_once()
