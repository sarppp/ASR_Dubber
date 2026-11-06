import hashlib
import json
from pathlib import Path
import subprocess
import wave
import numpy as np
import pytest
from batch_dubbing.sync_continuous_dub import (plan_fit,fit_quiet_tail,synchronize,last_activity,read_pcm)


def test_plan_is_measured_and_bounded():
    samples=np.zeros(2800,dtype=np.int16);samples[:2750]=1000
    plan=plan_fit([dict(end=27.4)],samples,100,29.5,29.48)
    assert plan['tempo']==pytest.approx(27.5/29.38)
    with pytest.raises(ValueError,match='outside allowed'):
        plan_fit([dict(end=20)],samples[:2000],100,29.5,29.48)
    with pytest.raises(ValueError,match='outside allowed'):
        plan_fit([dict(end=32)],samples,100,29.5,29.48)


def test_never_cut_active_audio():
    samples=np.array([1000]*100+[0]*20,dtype=np.int16)
    result,report=fit_quiet_tail(samples,100,1.1)
    assert len(result)==110 and report['removed_quiet_seconds']==.1
    with pytest.raises(ValueError,match='Active audio'):
        fit_quiet_tail(samples,100,.9)


def test_already_fitting_speech_stays_native_instead_of_speeding_to_margin():
    samples=np.full(2904,1000,dtype=np.int16)
    plan=plan_fit([dict(end=28.88)],samples,100,29.247792,29.04)
    assert plan['tempo']==1. and plan['native_within_tolerance']
    with pytest.raises(ValueError,match='outside allowed'):
        plan_fit([dict(end=29.6)],samples,100,29.247792,29.04)


def test_cpu_pipeline_with_real_stretch_mux_and_fake_aligner(tmp_path):
    source=tmp_path/'original.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=blue:s=160x90:r=25:d=3',
        '-c:v','libx264',str(source)],check=True)
    rate=24000;samples=np.zeros(round(2.85*rate),dtype=np.int16)
    t=np.arange(round(2.5*rate))/rate
    samples[round(.2*rate):round(2.7*rate)]=(3000*np.sin(2*np.pi*440*t)).astype(np.int16)
    audio=tmp_path/'raw.wav'
    with wave.open(str(audio),'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(rate);wav.writeframes(samples.tobytes())
    de=tmp_path/'de.srt';en=tmp_path/'en.srt'
    de.write_text('1\n00:00:00,000 --> 00:00:01,500\nGuten\n\n2\n00:00:01,500 --> 00:00:03,000\nTag\n')
    en.write_text(de.read_text().replace('Guten','Good').replace('Tag','day'))
    alignment=tmp_path/'alignment.json'
    alignment.write_text(json.dumps(dict(words=[dict(text='Guten',start=.2,end=1.),dict(text='Tag',start=1.1,end=2.7)],
        media_sha256={str(audio):hashlib.sha256(audio.read_bytes()).hexdigest()})))
    calls=[]
    def align(path,text):
        calls.append(text);fitted,sr=read_pcm(path)
        return [dict(text='Guten',start=.2,end=1.1),dict(text='Tag',start=1.2,end=last_activity(fitted,sr,-40))]
    out=tmp_path/'result'
    report=synchronize(source,audio,de,en,alignment,out,align=align)
    assert calls==['Guten Tag']
    assert report['status']=='complete' and report['video_packets_unchanged']
    assert report['acoustic_end_gap_seconds']<.4
    assert report['tts_calls']==0 and report['gpu_used'] is False
    for lang in ('de','en'):
        assert (out/f'DUB_SYNCED.{lang}.srt').read_bytes().startswith(b'\xef\xbb\xbf')
    # Pitch-preserving stretch should retain the 440 Hz source frequency.
    fitted,sr=read_pcm(out/'DUB_SYNCED.wav');segment=fitted[sr:2*sr]
    frequency=np.argmax(np.abs(np.fft.rfft(segment)))*sr/len(segment)
    assert abs(frequency-440)<2
    with pytest.raises(FileExistsError):synchronize(source,audio,de,en,alignment,out,align=align)
    assert calls==['Guten Tag']
