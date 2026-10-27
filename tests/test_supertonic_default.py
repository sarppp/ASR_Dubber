import sys
from pathlib import Path
from unittest.mock import Mock
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'qwen3-tts'))
import dub
from supertonic_tts_worker import synthesise, language_code

@pytest.mark.parametrize('engine', ['supertonic', 'qwen'])
def test_dub_engine_routing_and_cache(engine, tmp_path, monkeypatch):
    video=tmp_path/'video.mp4'; video.touch()
    srt=tmp_path/'translation.srt'; srt.write_text('translation')
    args=['dub.py',str(video),str(srt),'--language','de','--no-demucs','--workdir',str(tmp_path/'work')]
    if engine=='qwen': args+=['--tts-engine','qwen','--qwen-mode','custom']
    monkeypatch.setattr(sys,'argv',args)
    segments=[dict(index=1,speaker='SPEAKER_00',text='Hallo',start=0,end=2)]
    monkeypatch.setattr(dub,'parse_srt',lambda p: segments)
    monkeypatch.setattr(dub,'merge_segments',lambda s,**kw:s)
    extract=Mock(); monkeypatch.setattr(dub,'extract_clone_refs',extract)
    monkeypatch.setattr(dub,'write_dub_srt',Mock())
    monkeypatch.setattr(dub,'_audio_duration',lambda p:2)
    monkeypatch.setattr(dub,'speed_fit',lambda p,*a,**kw:p)
    monkeypatch.setattr(dub,'stitch_and_mix',lambda *a,**kw:(tmp_path/'final.mp4',[]))
    manager=Mock()
    def generate(text,voice,language,output):
        output.write_bytes(b'0'*600); return True
    manager.generate_custom.side_effect=generate
    constructor=Mock(return_value=manager); monkeypatch.setattr(dub,'SharedTTSManager',constructor)
    assert dub.main()==0
    assert constructor.call_args.args[2].endswith('supertonic_tts_worker.py' if engine=='supertonic' else 'qwen_tts_worker.py')
    assert constructor.call_args.kwargs['device_id']==(None if engine=='supertonic' else 0)
    call=manager.generate_custom.call_args
    if engine=='supertonic': assert call.args[1:3]==('M1','de')
    assert f'/cache/{engine}/' in str(call.args[3])
    extract.assert_not_called()
    manager.close.assert_called_once()

def test_german_punctuation_and_output(tmp_path):
    model=Mock(); model.synthesize.return_value=([1],1)
    model.save_audio.side_effect=lambda wav,p:Path(p).write_bytes(b'0'*101)
    synthesise(model,dict(text='„Hallo“ — schön…',output=str(tmp_path/'a.wav'),language='German'))
    assert model.synthesize.call_args.args[0]=='"Hallo" - schön...'
    assert model.synthesize.call_args.kwargs['lang']=='de'
    model.get_voice_style.assert_called_once_with(voice_name='M1')

def test_missing_audio_is_failure(tmp_path):
    model=Mock(); model.synthesize.return_value=([1],1)
    with pytest.raises(RuntimeError):
        synthesise(model,dict(text='Hallo',output=str(tmp_path/'missing.wav'),language='de'))

def test_languages():
    assert language_code(' German ')=='de'
    assert language_code('tr')=='tr'
    with pytest.raises(ValueError): language_code('bad')
