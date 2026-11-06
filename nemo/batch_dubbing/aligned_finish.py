"""Final word-aligned dubbing stage shared by the default Tower pipeline.

Source passage boundaries come from ASR before synthesis. This stage measures
finished native speech, places it, and writes dub-specific combined subtitles.
No ASR, translation or synthesis is performed here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import wave

from batch_dubbing.align_continuous_subtitles import tokens, timestamp, aligned_spans
from batch_dubbing.anchored_timeline import render as render_audio, packet_hash, video_duration
from subtitle_combine import write_files
from local_mt import split_translation


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(1024*1024):result.update(block)
    return result.hexdigest()


def passage_spans(cues):
    """Keep complete sentences together, including an introductory clause.

    An unpunctuated input ending stays visible as the selected input boundary;
    a passage exceeding the bounded synthesis window requires source review.
    """
    spans=[];begin=0;intro=False
    for i,cue in enumerate(cues):
        if i==begin:
            intro=bool(re.match(r'(?i)^(when|if|although|because|while|unless)\b',cue['text'].strip()))
        if cue['end']-cues[begin]['start']>30:
            raise ValueError('Source passage exceeds 30 seconds; review a complete speech boundary, never truncate it')
        if re.search(r'[.!?][\s"\u201d]*$',cue['text']):
            if intro:intro=False
            else:spans.append((begin,i));begin=i+1
    if begin<len(cues):spans.append((begin,len(cues)-1))
    return spans


def read_srt(path):
    rows=[]
    for block in re.split(r'\n\s*\n',Path(path).read_text(encoding='utf-8-sig').strip()):
        lines=block.splitlines()
        if len(lines)<3 or not lines[0].isdigit():raise ValueError('Malformed subtitle block')
        match=re.fullmatch(r'(\d+):(\d\d):(\d\d),(\d{3}) --> (\d+):(\d\d):(\d\d),(\d{3})',lines[1])
        if not match:raise ValueError('Invalid subtitle timestamps')
        v=list(map(int,match.groups()))
        start=v[0]*3600+v[1]*60+v[2]+v[3]/1000;end=v[4]*3600+v[5]*60+v[6]+v[7]/1000
        if end<=start or (rows and start<rows[-1]['end']-.001):raise ValueError('Overlapping or empty subtitles')
        text='\n'.join(lines[2:]).strip()
        if not text:raise ValueError('Empty subtitle text')
        rows.append(dict(index=int(lines[0]),start=start,end=end,text=text))
    if not rows:raise ValueError('No subtitles')
    return rows


def source_plan(source_srt,spoken_srt,comparison_srts,source_language,target_language,*,source_timing=None):
    """Bind complete source cue ranges to the actual adapted TTS text before TTS."""
    source=read_srt(source_srt);spoken=read_srt(spoken_srt)
    source_words=None;word_offsets=[0]
    if source_timing:
        evidence=json.loads(Path(source_timing).read_text())
        source_words=evidence['words']
        if not source_words or tokens(' '.join(c['text'] for c in source)) != [t for w in source_words for t in tokens(w['word'])]:
            raise ValueError('Source words do not match the original subtitle text')
        if any(len(tokens(w['word']))!=1 for w in source_words):raise ValueError('Expected one timed source word per entry')
        for cue in source:word_offsets.append(word_offsets[-1]+len(tokens(cue['text'])))
    comparisons={lang:read_srt(path) for lang,path in comparison_srts.items() if lang!=target_language}
    comparisons[source_language]=source
    target_cues=read_srt(comparison_srts[target_language]) if target_language in comparison_srts else None
    if target_cues is not None and len(target_cues)!=len(source):raise ValueError('Target cue coverage differs')
    if any(len(rows)!=len(source) for rows in comparisons.values()):raise ValueError('Comparison cue coverage differs')
    result=[];covered=[]
    for row in spoken:
        indices=[i for i,c in enumerate(source) if c['start']>=row['start']-.002 and c['end']<=row['end']+.002]
        if not indices or indices!=list(range(indices[0],indices[-1]+1)):raise ValueError('Unlinked source passage')
        covered.extend(indices)
        original_target=[target_cues[i]['text'] for i in indices] if target_cues is not None else None
        preserved=original_target is not None and ' '.join(' '.join(t.split()) for t in original_target)==' '.join(row['text'].split())
        pieces=original_target if preserved else split_translation(row['text'],[len(source[i]['text']) for i in indices])
        if pieces is None:raise ValueError('Cannot freeze paired captions without dropping a cue')
        caption_cues=[dict(index=source[i]['index'],texts={**{lang:rows[i]['text'] for lang,rows in comparisons.items()},target_language:piece})
                      for i,piece in zip(indices,pieces)]
        if tokens(' '.join(c['texts'][target_language] for c in caption_cues))!=tokens(row['text']):
            raise ValueError('Frozen captions differ from the complete spoken text')
        result.append(dict(index=row['index'],source_start=source[indices[0]]['start'],source_end=source[indices[-1]]['end'],
            context_start=source[indices[0]]['start'],context_end=source[indices[-1]]['end'],
            text=row['text'],source_cue_indices=indices,caption_cues=caption_cues,target_caption_texts_preserved=preserved,
            comparisons={lang:' '.join(rows[i]['text'] for i in indices) for lang,rows in comparisons.items()}))
        if source_words:
            words=source_words[word_offsets[indices[0]]:word_offsets[indices[-1]+1]]
            result[-1].update(source_start=words[0]['start'],source_end=words[-1]['end'],source_words=words)
    if covered!=list(range(len(source))):raise ValueError('Every source cue must belong to exactly one spoken passage; provide an explicit input excerpt instead of dropping sentences')
    return dict(version=2,source_language=source_language,target_language=target_language,passages=result,
                caption_contract='Freeze paired cue IDs and texts before synthesis; final alignment only changes timestamps',
                source_script_sha256=sha(source_srt),spoken_script_sha256=sha(spoken_srt),
                comparison_sha256={lang:sha(path) for lang,path in comparison_srts.items()},
                boundary_evidence=evidence.get('word_evidence','ASR word timeline') if source_words else 'Original ASR cue first/last word timestamps; never derived from target subtitle durations')


def checked_words(words,text,duration):
    # Punctuation has no independent acoustic word. Preserve it in the caption
    # attached to its neighbour instead of reporting a dropped spoken word.
    parts=[];prefix=[]
    for part in text.split():
        if tokens(part):
            parts.append(' '.join(prefix+[part]));prefix=[]
        elif parts:parts[-1]+=' '+part
        else:prefix.append(part)
    words=[w for w in words if tokens(w['text'])]
    if len(words)!=len(parts) or [t for w in words for t in tokens(w['text'])]!=tokens(text):
        raise ValueError('Generated-word alignment does not cover the entire spoken script')
    previous=0.
    for word in words:
        start,end=word['start'],word['end']
        if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (start,end)) or not 0<=start<=end<=duration+.02 or start<previous-.02:
            raise ValueError('Invalid or non-monotonic aligned words')
        previous=end
    return [dict(w,text=t) for w,t in zip(words,parts)]


def pack_words(words):
    groups=[];group=[]
    for word in words:
        if group and (len(group)>=10 or word['end']-group[0]['start']>5 or len(' '.join(w['text'] for w in group)+' '+word['text'])>80):
            groups.append(group);group=[]
        group.append(word)
    if group:groups.append(group)
    return groups


def write_srt(path,rows):
    text='\n\n'.join(f'{r.get("index",i)}\n{timestamp(r["start"])} --> {timestamp(r["end"])}\n{r["text"]}' for i,r in enumerate(rows,1))+'\n'
    Path(path).write_bytes(text.replace('\r\n','\n').replace('\n','\r\n').encode('utf-8-sig'))


def finish(request,align):
    """align(path,text,language) is a real forced aligner or an offline test double."""
    import numpy as np
    video=Path(request['video']);out=Path(request['output']);out.mkdir(parents=True,exist_ok=True)
    mode=request.get('placement_mode','source_anchored')
    if mode not in {'continuous','continuous_synced','source_anchored'}:raise ValueError('Unknown placement mode')
    whole_context=mode in {'continuous','continuous_synced'}
    if request.get('bundles') and whole_context:
        from batch_dubbing.continuous_context import preserve_continuous
        return finish(preserve_continuous(request),align)
    if request.get('bundles'):
        from batch_dubbing.continuous_context import expand_continuous
        try:
            with tempfile.TemporaryDirectory(dir=out,prefix='continuous-') as temporary:
                expanded,aligned,evidence=expand_continuous(request,align,Path(temporary))
                result=finish(expanded,lambda path,text,lang:aligned[str(path)])
            result['synthesis_context']='Continuous bounded take; split after word alignment at verified quiet passage boundaries'
            result['continuous_takes']=evidence
            (out/'alignment_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            return result
        except Exception as exc:
            if not (out/'alignment_report.json').exists():
                (out/'alignment_report.json').write_text(json.dumps(dict(status='needs_review',error=str(exc)),indent=2))
            raise
    plan=request['plan'];takes=request['takes'];duration=video_duration(video)
    if len(takes)!=len(plan['passages']):raise ValueError('Missing generated passage')
    source_hash=sha(video);report=dict(status='checking',video_seconds=duration,tempo=None if mode=='continuous_synced' else 1.,placement_mode=mode,passages=[],warnings=[],
        source_plan=plan,source_sha256=source_hash,quality_limit='Forced alignment is timing evidence, not independent semantic or spoken-content certification.')
    report_path=out/'alignment_report.json'
    def save():report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    rows_by_lang={lang:[] for lang in dict.fromkeys([plan['source_language'],plan['target_language']]+list(plan['passages'][0]['comparisons']))}
    try:
        with tempfile.TemporaryDirectory(dir=out,prefix='aligned-') as temporary:
            work=Path(temporary);placements=[];previous_end=0.
            for i,(passage,take) in enumerate(zip(plan['passages'],takes)):
                if take['index']!=passage['index'] or take['text']!=passage['text'] or take['tempo']!=1.:
                    raise ValueError('Wrong passage identity or non-native speech')
                audio=Path(take['file'])
                if sha(audio)!=take['sha256']:raise ValueError('Generated audio changed')
                with wave.open(str(audio)) as wav:
                    if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate())!=(1,2,24000):raise ValueError('Expected mono PCM16 24 kHz')
                    data=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').copy();rate=wav.getframerate()
                words=checked_words(align(audio,passage['text'],plan['target_language']),passage['text'],len(data)/rate)
                sync=None
                if mode=='continuous_synced':
                    from batch_dubbing.sync_continuous_dub import fit_recording,read_pcm,last_activity
                    start=passage['context_start']
                    context_deadline=min(duration,passage['context_end']+.25,
                                         plan['passages'][i+1]['context_start'] if i+1<len(takes) else duration)
                    evidence=Path(tempfile.mkdtemp(dir=out,prefix=f'sync-context-{passage["index"]}-'))
                    adjusted,fit,tail=fit_recording(audio,words,context_deadline-start,passage['context_end']-start,evidence)
                    data,rate=read_pcm(adjusted)
                    words=checked_words(align(adjusted,passage['text'],plan['target_language']),passage['text'],len(data)/rate)
                    aligned_gap=fit['video_seconds']-words[-1]['end']
                    acoustic_gap=fit['video_seconds']-last_activity(data,rate,-40)
                    sync=dict(plan=fit,tail=tail,evidence=str(evidence),fitted_sha256=sha(adjusted),
                              aligned_end_gap_seconds=aligned_gap,acoustic_end_gap_seconds=acoustic_gap)
                    report.setdefault('duration_adjustments',[]).append(sync);save()
                    if not 0<=aligned_gap<=.40 or not 0<=acoustic_gap<=.40:
                        raise ValueError(f'Ending still unsynchronized: aligned gap {aligned_gap:.3f}s, acoustic gap {acoustic_gap:.3f}s')
                active=np.flatnonzero(np.abs(data.astype(float))>=32768*10**(-55/20))
                if not len(active):raise ValueError('Silent passage')
                # Quiet margins only. Internal pauses and every active sample survive.
                first=max(0,int(active[0])-round(.04*rate));last=min(len(data),int(active[-1])+1+round(.04*rate))
                onset=words[0]['start']-first/rate
                proposed=passage['source_start']-onset
                length=(last-first)/rate
                deadline=min(duration,passage['source_end']+.25)
                if i+1<len(takes):deadline=min(deadline,plan['passages'][i+1]['source_start']-.04)
                start=max(0.,proposed)
                if start+length>deadline:start=deadline-length
                if whole_context:
                    # Preserve every original sample, pause and relative onset.
                    # Anchor contexts independently so overruns cannot accumulate.
                    first=0;last=len(data);length=len(data)/rate
                    proposed=start=passage['context_start']
                    if i+1<len(takes):deadline=min(deadline,plan['passages'][i+1]['context_start'])
                    if sync:deadline=context_deadline
                frame=round(start*rate);start=frame/rate;offset=start-first/rate
                if start<previous_end-1/rate or start<0 or abs(start-proposed)>.25+1/rate or frame+last-first>round(duration*rate) or (whole_context and start+length>deadline+1/rate):
                    report['failed_placement']=dict(index=passage['index'],native_seconds=length,
                        proposed_start=proposed,required_start=start,deadline=deadline,
                        required_onset_shift=start-proposed,source_start=passage['source_start'],source_end=passage['source_end'])
                    raise ValueError(f'Passage {passage["index"]} cannot fit source timing at native speed; no speech cut or stretch')
                previous_end=start+length
                native=work/f'passage-{i}.wav'
                with wave.open(str(native),'wb') as wav:
                    wav.setparams((1,2,rate,0,'NONE','not compressed'));wav.writeframes(data[first:last].astype('<i2').tobytes())
                absolute=[dict(w,start=max(start,w['start']+offset),end=min(previous_end,w['end']+offset)) for w in words]
                frozen=passage.get('caption_cues')
                if not frozen:raise ValueError('Missing frozen paired captions; refusing to invent subtitle cuts after synthesis')
                target_cues=[dict(index=c['index'],text=c['texts'][plan['target_language']]) for c in frozen]
                spans=aligned_spans(target_cues,absolute,duration)
                for cue,(a,b) in zip(frozen,spans):
                    for lang,text in cue['texts'].items():
                        rows_by_lang[lang].append(dict(index=cue['index'],start=a,end=b,text=text))
                acoustic_end=(int(active[-1])+1)/rate+offset
                row=dict(index=passage['index'],source_start=passage['source_start'],source_end=passage['source_end'],
                         start=start,end=previous_end,first_word=absolute[0]['start'],last_word=absolute[-1]['end'],
                         acoustic_end=acoustic_end,end_difference=passage['source_end']-acoustic_end,words=absolute,
                         raw_sha256=take['sha256'],all_active_samples_retained=sync is None,
                         active_audio_truncated=False)
                report['passages'].append(row)
                if whole_context:
                    row['all_raw_samples_retained']=sync is None
                    row['effective_tempo']=sync['plan']['tempo'] if sync else 1.
                    if sync:row['duration_adjustment']=sync
                    row['source_passage_timing']=[]
                    for p in passage.get('source_passages',[]):
                        timed=absolute[p['word_start']:p['word_end']]
                        drift=dict(index=p['index'],source_start=p['source_start'],source_end=p['source_end'],
                                   target_start=timed[0]['start'],target_end=timed[-1]['end'],
                                   onset_difference=timed[0]['start']-p['source_start'],end_difference=timed[-1]['end']-p['source_end'])
                        row['source_passage_timing'].append(drift)
                        if abs(drift['onset_difference'])>.25 or abs(drift['end_difference'])>.4:
                            report['warnings'].append(dict(reason='continuous_delivery_differs_from_source_passage',**drift))
                if row['end_difference']>.4:report['warnings'].append(dict(index=row['index'],reason='sentence_ends_early',seconds=row['end_difference']))
                placements.append((native,start,context_deadline if sync else passage['source_end']))
                save()
            # Shared 80 ms lead / 120 ms tail, with cross-passage overlap caps.
            expected=[cue for passage in plan['passages'] for cue in passage['caption_cues']]
            for lang,rows in rows_by_lang.items():
                if [(r['index'],r['text']) for r in rows]!=[(c['index'],c['texts'][lang]) for c in expected]:
                    raise ValueError('Final caption IDs or texts changed')
                for left,right in zip(rows,rows[1:]):left['end']=min(left['end'],right['start'])
                if any(r['end']-r['start']<.1 for r in rows):raise ValueError('Caption has too little reliable aligned time')
            report.update(caption_contract_verified=True,cue_count=len(expected),subtitle_display_lead=.08,subtitle_display_tail=.12)
            published=work/'output';published.mkdir()
            movie,_=render_audio(placements,video,published,work,background=request.get('background'),pad_audio=not whole_context)
            if mode=='continuous_synced':
                timeline=json.loads((published/'timeline_report.json').read_text())
                timeline.update(tempo=None,context_tempos=[p['effective_tempo'] for p in report['passages']],
                                note='Pitch-preserving bounded context slowdown; full original video timeline; see duration_adjustments')
                (published/'timeline_report.json').write_text(json.dumps(timeline,indent=2)+'\n')
            for lang,rows in rows_by_lang.items():write_srt(published/f'final_dub.{lang}.srt',rows)
            combined=write_files({lang:[dict(r,text=' '.join(r['text'].split())) for r in rows] for lang,rows in rows_by_lang.items()},published/'final_dub',order=list(rows_by_lang))
            # Match the approved player workflow: source comparison selected by
            # default, target selectable. Combined SRT/ASS remain sidecars.
            subtitled=work/'subtitled.mp4'
            languages=[plan['source_language']]+[lang for lang in rows_by_lang if lang!=plan['source_language']]
            iso={'en':'eng','de':'deu','fr':'fra','es':'spa','it':'ita','pt':'por','nl':'nld'}
            command=['ffmpeg','-nostdin','-v','error','-n','-i',str(movie)]
            for lang in languages:command+=['-i',str(published/f'final_dub.{lang}.srt')]
            command+=['-map','0:v','-map','0:a']
            for i in range(len(languages)):command+=['-map',f'{i+1}:0']
            command+=['-c:v','copy','-c:a','copy','-c:s','mov_text','-metadata:s:a:0',f'language={iso.get(plan["target_language"],plan["target_language"])}']
            for i,lang in enumerate(languages):
                command += [f'-metadata:s:s:{i}',f'language={iso.get(lang,lang)}',f'-disposition:s:{i}','default' if i==0 else '0']
            subprocess.run(command+['-movflags','+faststart',str(subtitled)],check=True,timeout=600)
            if packet_hash(subtitled)!=packet_hash(video) or abs(video_duration(subtitled)-duration)>.001 or sha(video)!=source_hash:
                raise ValueError('Source picture or identity changed')
            subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(subtitled),'-f','null','-'],check=True,timeout=600)
            subtitled.replace(movie)
            report.update(status='complete',video_packets_unchanged=True,model_timing='Original ASR first; native TTS second; target word alignment third; placement and combined captions last',
                          subtitle_languages=list(rows_by_lang),ending_gap_seconds=duration-report['passages'][-1]['acoustic_end'])
            for path in published.iterdir():path.replace(out/path.name)
            report['outputs']={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='alignment_report.json'}
            save()
            return report
    except Exception as exc:
        report.update(status='needs_review',error=str(exc));save();raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('request',type=Path);parser.add_argument('--device',default='cpu',choices=['cpu','cuda:0'])
    args=parser.parse_args()
    import os
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
    if args.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from huggingface_hub import snapshot_download
    from qwen_asr import Qwen3ForcedAligner
    from batch_dubbing.align_continuous_subtitles import MODEL,REVISION
    torch.set_num_threads(4)
    model=Qwen3ForcedAligner.from_pretrained(snapshot_download(MODEL,revision=REVISION),device_map=args.device,dtype=torch.float32 if args.device=='cpu' else torch.float16,attn_implementation='sdpa')
    names={'en':'English','de':'German','fr':'French','es':'Spanish','it':'Italian'}
    def align(path,text,language):
        identity=dict(audio_sha256=sha(path),text=text,language=language,model=MODEL,revision=REVISION,device=args.device)
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        cache=args.request.parent/'word_alignment_cache';cache.mkdir(exist_ok=True)
        evidence=cache/(key+'.json')
        if evidence.exists():
            saved=json.loads(evidence.read_text())
            if saved.get('identity')==identity:return saved['words']
        result=model.align(audio=str(path),text=text,language=names.get(language,language))
        words=[dict(text=w.text,start=float(w.start_time),end=float(w.end_time)) for w in result[0].items]
        evidence.write_text(json.dumps(dict(identity=identity,words=words),ensure_ascii=False,indent=2))
        return words
    request=json.loads(args.request.read_text())
    if request.get('operation')=='align_source':
        from batch_dubbing.continuous_context import align_source
        align_source(request,align)
    else:
        finish(request,align)


if __name__=='__main__':main()
