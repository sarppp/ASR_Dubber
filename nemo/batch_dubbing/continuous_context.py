"""Bounded continuous synthesis, followed by measured quiet-boundary placement."""
from pathlib import Path
import json
import wave
import numpy as np


def bundle_segments(segments, max_seconds=30.):
    """Keep voice context across whole passages; caption boundaries are irrelevant."""
    bundles=[]
    for segment in segments:
        if segment['end']-segment['start']>max_seconds:
            raise ValueError('A complete source passage exceeds the synthesis context bound')
        if (not bundles or bundles[-1]['speaker']!=segment['speaker']
                or segment['end']-bundles[-1]['start']>max_seconds
                or segment['start']-bundles[-1]['end']>2.):
            bundles.append(dict(index=len(bundles)+1,start=segment['start'],end=segment['end'],
                                speaker=segment['speaker'],text=segment['text'],passage_indices=[segment['index']]))
        else:
            bundles[-1]['end']=segment['end']
            bundles[-1]['text']+=' '+segment['text']
            bundles[-1]['passage_indices'].append(segment['index'])
    return bundles


def expand_continuous(request,align,work):
    """Align each complete take once. Partition only verified quiet boundaries."""
    from batch_dubbing.aligned_finish import checked_words,sha
    from batch_dubbing.align_continuous_subtitles import tokens
    passages=request['plan']['passages'];groups=request['bundles'];takes=request['takes']
    if len(groups)!=len(takes) or [i for g in groups for i in g]!=[p['index'] for p in passages]:
        raise ValueError('Continuous contexts do not cover every source passage exactly once')
    by_index={p['index']:p for p in passages};expanded=[];aligned={};evidence=[]
    for group,take in zip(groups,takes):
        parts=[by_index[i] for i in group];script=' '.join(p['text'] for p in parts)
        audio=Path(take['file'])
        if take['text']!=script or take['tempo']!=1. or sha(audio)!=take['sha256']:
            raise ValueError('Continuous take identity mismatch')
        with wave.open(str(audio)) as wav:
            rate=wav.getframerate()
            if (wav.getnchannels(),wav.getsampwidth(),rate)!=(1,2,24000):raise ValueError('Expected native PCM16 mono 24 kHz')
            data=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').copy()
        words=checked_words(align(audio,script,request['plan']['target_language']),script,len(data)/rate)
        offsets=[0]
        for p in parts:offsets.append(offsets[-1]+len(tokens(p['text'])))
        boundaries=[0];peaks=[];width=round(.02*rate)
        for offset in offsets[1:-1]:
            left=round(words[offset-1]['end']*rate);right=round(words[offset]['start']*rate)
            candidates=[(float(np.max(np.abs(data[p:p+width].astype(float)))),p+width//2)
                        for p in range(left,right-width+1,max(1,width//4))]
            if not candidates:raise ValueError('No aligned quiet interval between complete passages')
            peak,position=min(candidates)
            if peak/32768>10**(-45/20):raise ValueError('Passage boundary contains speech; refusing to cut it')
            boundaries.append(position);peaks.append(peak/32768)
        boundaries.append(len(data))
        for j,part in enumerate(parts):
            a,b=boundaries[j:j+2];path=Path(work)/f"passage_{part['index']:04d}.wav"
            with wave.open(str(path),'wb') as wav:
                wav.setparams((1,2,rate,0,'NONE','not compressed'));wav.writeframes(data[a:b].tobytes())
            expanded.append(dict(index=part['index'],file=str(path),text=part['text'],tempo=1.,sha256=sha(path)))
            aligned[str(path)]=[dict(w,start=w['start']-a/rate,end=w['end']-a/rate) for w in words[offsets[j]:offsets[j+1]]]
        evidence.append(dict(raw_sha256=take['sha256'],passage_indices=group,boundary_frames=boundaries,
                             boundary_peaks=peaks,partition_preserves_every_sample=True))
    result={k:v for k,v in request.items() if k!='bundles'};result['takes']=expanded
    return result,aligned,evidence


def preserve_continuous(request):
    """Keep each bounded recording whole at its absolute source cue start.

    Source sentence timings remain diagnostics. They must not rearrange words
    or pauses inside the user's selected continuous-delivery mode.
    """
    from batch_dubbing.align_continuous_subtitles import tokens
    plan=request['plan'];groups=request['bundles'];takes=request['takes']
    if len(groups)!=len(takes) or [i for g in groups for i in g]!=[p['index'] for p in plan['passages']]:
        raise ValueError('Continuous contexts do not cover every source passage exactly once')
    by_index={p['index']:p for p in plan['passages']};contexts=[]
    for group,take in zip(groups,takes):
        parts=[by_index[i] for i in group]
        start=parts[0]['context_start']
        if parts[-1]['source_end']-start>30.02:
            raise ValueError('Continuous context exceeds 30 seconds; never concatenate an entire long video')
        offset=0;source_passages=[]
        for p in parts:
            count=len(tokens(p['text']))
            source_passages.append(dict(index=p['index'],source_start=p['source_start'],source_end=p['source_end'],
                                        word_start=offset,word_end=offset+count))
            offset+=count
        contexts.append(dict(index=take['index'],context_start=start,context_end=parts[-1].get('context_end',parts[-1]['source_end']),source_start=parts[0]['source_start'],
                             source_end=parts[-1]['source_end'],text=' '.join(p['text'] for p in parts),
                             caption_cues=[cue for p in parts for cue in p['caption_cues']],
                             comparisons={lang:' '.join(p['comparisons'][lang] for p in parts) for lang in parts[0]['comparisons']},
                             source_passages=source_passages))
    return {**{k:v for k,v in request.items() if k!='bundles'},'plan':dict(plan,passages=contexts)}


def align_source(request,align):
    """Measure source words before synthesis, in bounded contexts from source cues."""
    import subprocess,tempfile
    from batch_dubbing.aligned_finish import read_srt,passage_spans,checked_words,sha,video_duration
    source=read_srt(request['source_script']);video=Path(request['video']);duration=video_duration(video)
    spans=passage_spans(source)
    segments=[dict(index=i+1,start=source[a]['start'],end=source[b]['end'],speaker='source',
                   text=' '.join(c['text'] for c in source[a:b+1])) for i,(a,b) in enumerate(spans)]
    bundles=bundle_segments(segments);all_words=[]
    with tempfile.TemporaryDirectory() as temp:
        for i,bundle in enumerate(bundles):
            start=max(0,bundle['start']-.5);end=min(duration,bundle['end']+.5)
            wav=Path(temp)/f'source_{i}.wav'
            subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(video),'-ss',str(start),'-t',str(end-start),
                            '-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',str(wav)],check=True,timeout=60)
            words=checked_words(align(wav,bundle['text'],request['source_language']),bundle['text'],end-start)
            all_words.extend(dict(word=w['text'],start=w['start']+start,end=w['end']+start) for w in words)
    result=dict(words=all_words,word_evidence='CPU forced alignment',source_sha256=sha(video),
                script_sha256=sha(request['source_script']),source_language=request['source_language'])
    Path(request['output']).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result
