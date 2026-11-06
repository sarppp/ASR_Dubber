"""Automatically fit a short continuous dub, then align and package both SRTs.

CPU only. No TTS regeneration, no per-cue cuts, no hand-entered timestamps.
A bounded slowdown preserves pitch. Excess duration changes fail for review.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import wave
import numpy as np
from .align_continuous_subtitles import (aligned_spans,align_words_cpu,cues,render,MODEL,REVISION)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seconds(stamp):
    h,m,s,ms=map(int,re.split('[:,]',stamp))
    return h*3600+m*60+s+ms/1000


def read_pcm(path):
    with wave.open(str(path)) as wav:
        if wav.getnchannels()!=1 or wav.getsampwidth()!=2:
            raise ValueError('Expected mono PCM16 input')
        return np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').copy(),wav.getframerate()


def last_activity(samples,rate,threshold_db=-55.):
    indices=np.flatnonzero(np.abs(samples.astype(np.float64))/32768>=10**(threshold_db/20))
    if not len(indices):raise ValueError('Silent recording')
    return (int(indices[-1])+1)/rate


def plan_fit(words,samples,rate,video_seconds,source_end,max_slowdown=.10):
    if not words or not 0<=max_slowdown<=.10:
        raise ValueError('Missing words or unsupported slowdown limit')
    if not all(math.isfinite(x) and x>0 for x in (video_seconds,source_end)):
        raise ValueError('Invalid source duration')
    target_end=min(video_seconds,source_end)-.10
    speech_end=max(words[-1]['end'],last_activity(samples,rate))
    tempo=speech_end/target_end
    native_within_tolerance=tempo>1 and 0<=video_seconds-speech_end<=.40
    if native_within_tolerance:
        # The 100 ms margin is a target, not a reason to speed up already
        # fitting speech. The post-fit word/activity checks remain mandatory.
        tempo=1.
    if not 1-max_slowdown-1e-12<=tempo<=1+1e-12:
        raise ValueError(f'Required tempo {tempo:.4f} outside allowed [{1-max_slowdown}, 1]; revise speech rather than force fitting')
    tempo=min(1.,max(1-max_slowdown,tempo))
    return dict(tempo=tempo,pitch=1.,speech_end=speech_end,target_end=target_end,
                video_seconds=video_seconds,source_end=source_end,slowdown_limit=max_slowdown,
                native_within_tolerance=native_within_tolerance)


def fit_quiet_tail(samples,rate,duration,threshold_db=-55.):
    """Permit only measured near-silent overflow; never truncate active samples."""
    count=round(duration*rate)
    if count<=0:raise ValueError('Invalid output duration')
    if len(samples)>count:
        peak=float(np.max(np.abs(samples[count:].astype(float)))/32768)
        if peak>=10**(threshold_db/20):
            raise ValueError('Active audio would be cut at the picture end')
        return samples[:count],dict(removed_quiet_seconds=(len(samples)-count)/rate,removed_peak=peak,padded_seconds=0.)
    return np.pad(samples,(0,count-len(samples))),dict(removed_quiet_seconds=0.,removed_peak=0.,padded_seconds=(count-len(samples))/rate)


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(path)],text=True,timeout=15))


def packet_hash(path,stream):
    return subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(path),'-map',stream,'-c','copy','-f','hash','-hash','sha256','-'],text=True,timeout=30).strip()


def fit_recording(audio,words,video_seconds,source_end,output):
    """Shared CPU duration adjustment; no ASR/TTS or manual word positions."""
    samples,rate=read_pcm(audio)
    plan=plan_fit(words,samples,rate,video_seconds,source_end)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    stretched=output/'full_stretched_take.wav';fitted_audio=output/'DUB_SYNCED.wav'
    if stretched.exists() or fitted_audio.exists():raise FileExistsError('Never overwrite fit evidence')
    subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-i',str(audio),'-af',f'rubberband=tempo={plan["tempo"]:.12f}:pitch=1',
        '-ar',str(rate),'-ac','1','-c:a','pcm_s16le',str(stretched)],check=True,timeout=60)
    new_samples,new_rate=read_pcm(stretched)
    fitted,tail=fit_quiet_tail(new_samples,new_rate,video_seconds)
    with wave.open(str(fitted_audio),'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(new_rate);wav.writeframes(fitted.astype('<i2').tobytes())
    (output/'fit.json').write_text(json.dumps(dict(plan=plan,tail=tail,raw_sha256=sha(audio),fitted_sha256=sha(fitted_audio)),indent=2)+'\n')
    return fitted_audio,plan,tail


def synchronize(video,audio,spoken_srt,english_srt,alignment,output,*,align=align_words_cpu):
    paths=[Path(p).resolve() for p in (video,audio,spoken_srt,english_srt,alignment)]
    video,audio,spoken_srt,english_srt,alignment=paths
    original={str(p):sha(p) for p in paths}
    german,english=cues(spoken_srt),cues(english_srt)
    if len(german)!=len(english):raise ValueError('Cue correspondence required')
    data=json.loads(alignment.read_text());words=data['words']
    identities=data.get('media_sha256',{})
    if sha(audio) not in identities.values():raise ValueError('Alignment is not bound to this audio')
    samples,rate=read_pcm(audio)
    aligned_spans(german,words,len(samples)/rate)
    video_seconds=float(next(s['duration'] for s in probe(video)['streams'] if s['codec_type']=='video'))
    # This helper handles one short continuous passage, never silently a long video.
    if video_seconds>60:raise ValueError('Split a long video at semantic passage boundaries first')
    plan=plan_fit(words,samples,rate,video_seconds,seconds(english[-1]['timing'].split(' --> ')[1]))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'plan.json').write_text(json.dumps(dict(**plan,inputs=original),indent=2)+'\n')
    print('Automatic duration plan:',json.dumps(plan),flush=True)
    fitted_audio,plan,tail=fit_recording(audio,words,video_seconds,plan['source_end'],output)
    fitted,new_rate=read_pcm(fitted_audio)
    new_words=align(fitted_audio,' '.join(c['text'].replace('\n',' ') for c in german))
    raw=dict(model=MODEL,revision=REVISION,device='cpu',words=new_words,media_sha256={str(fitted_audio.resolve()):sha(fitted_audio)})
    (output/'aligned_words.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2)+'\n')
    spans=aligned_spans(german,new_words,len(fitted)/new_rate)
    end_gap=video_seconds-new_words[-1]['end']
    acoustic_gap=video_seconds-last_activity(fitted,new_rate,-40)
    if not 0<=end_gap<=.40 or not 0<=acoustic_gap<=.40:
        raise ValueError(f'Ending still unsynchronized: aligned gap {end_gap:.3f}s, acoustic gap {acoustic_gap:.3f}s')
    for lang,rows in [('de',german),('en',english)]:
        (output/f'DUB_SYNCED.{lang}.srt').write_bytes(render(rows,spans))
    movie=output/'DUB_SYNCED.mp4'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-i',str(video),'-i',str(fitted_audio),
        '-i',str(output/'DUB_SYNCED.en.srt'),'-i',str(output/'DUB_SYNCED.de.srt'),
        '-map','0:v:0','-map','1:a:0','-map','2:0','-map','3:0','-c:v','copy','-c:a','aac','-b:a','192k','-c:s','mov_text',
        '-metadata:s:a:0','language=deu','-metadata:s:s:0','language=eng','-metadata:s:s:1','language=deu',
        '-disposition:s:0','default','-disposition:s:1','0','-movflags','+faststart',str(movie)],check=True,timeout=60)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(movie),'-f','null','-'],check=True,timeout=60)
    assert packet_hash(video,'0:v:0')==packet_hash(movie,'0:v:0'),'Source picture changed'
    assert all(sha(p)==expected for p,expected in original.items()),'Original input changed'
    info=probe(movie)
    if abs(float(info['format']['duration'])-video_seconds)>.06:raise ValueError('Output duration drift')
    report=dict(status='complete',plan=plan,tail=tail,aligned_end_gap_seconds=end_gap,acoustic_end_gap_seconds=acoustic_gap,
        words=len(new_words),cues=len(spans),alignment_calls=1,tts_calls=0,gpu_used=False,video_packets_unchanged=True,
        original_inputs_unchanged=True,subtitles='English and German share newly measured German word spans',
        limitations='End duration matching, not lip sync or semantic validation; original text remains incomplete if input was cut mid-sentence.',
        outputs={str(p):sha(p) for p in (movie,fitted_audio,output/'DUB_SYNCED.en.srt',output/'DUB_SYNCED.de.srt')})
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('video','audio','spoken-srt','english-srt','alignment','output'):parser.add_argument('--'+arg,required=True,type=Path)
    args=parser.parse_args()
    synchronize(args.video,args.audio,args.spoken_srt,args.english_srt,args.alignment,args.output)


if __name__=='__main__':main()
