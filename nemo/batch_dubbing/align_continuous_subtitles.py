"""Retiming only: align known German words and share cue spans with English."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import unicodedata
import wave

MODEL='Qwen/Qwen3-ForcedAligner-0.6B'
REVISION='c7cbfc2048c462b0d63a45797104fc9db3ad62b7'


def cues(path):
    blocks=re.split(r'\n\s*\n',Path(path).read_text(encoding='utf-8-sig').strip())
    result=[]
    for block in blocks:
        lines=block.splitlines()
        if len(lines)<3 or not lines[0].isdigit() or ' --> ' not in lines[1]:
            raise ValueError('Invalid SRT cue')
        result.append(dict(index=int(lines[0]),timing=lines[1],text='\n'.join(lines[2:])))
    return result


def tokens(text):
    return [word for token in text.split()
            if (word:=''.join(c for c in token if c=="'" or unicodedata.category(c)[0] in 'LN').casefold())]


def aligned_spans(german,words,duration):
    expected=[word for cue in german for word in tokens(cue['text'])]
    actual=[token for row in words for token in tokens(row['text'])]
    if expected!=actual or len(actual)!=len(words):
        raise ValueError('Alignment tokens do not match the original German text')
    previous_start=previous_end=0.
    for row in words:
        start,end=row['start'],row['end']
        if (not all(math.isfinite(x) for x in (start,end)) or start<previous_start
                or end<previous_end or not 0<=start<=end<=duration+.05):
            raise ValueError('Invalid or non-monotonic word timing')
        previous_start,previous_end=start,end
    spans=[];offset=0
    for cue in german:
        count=len(tokens(cue['text']))
        if not count:raise ValueError('Empty cue')
        segment=words[offset:offset+count];offset+=count
        spans.append([max(0.,segment[0]['start']-.08),min(duration,segment[-1]['end']+.12)])
    for i in range(len(spans)-1):
        spans[i][1]=min(spans[i][1],spans[i+1][0])
    if any(end-start<.1 for start,end in spans):
        raise ValueError('A cue has too little reliable aligned time')
    return spans


def timestamp(seconds):
    millis=round(seconds*1000)
    hours,millis=divmod(millis,3600000);minutes,millis=divmod(millis,60000)
    seconds,millis=divmod(millis,1000)
    return f'{hours:02}:{minutes:02}:{seconds:02},{millis:03}'


def render(rows,spans):
    if len(rows)!=len(spans):raise ValueError('English/German cue count mismatch')
    return ('\r\n\r\n'.join(f'{cue["index"]}\r\n{timestamp(start)} --> {timestamp(end)}\r\n'+cue['text'].replace('\n','\r\n')
        for cue,(start,end) in zip(rows,spans))+'\r\n').encode('utf-8-sig')


def align_words_cpu(audio,text):
    """One offline alignment pass; shared by retiming and duration matching."""
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false')
    import torch
    from huggingface_hub import snapshot_download
    from qwen_asr import Qwen3ForcedAligner
    torch.set_num_threads(4)
    model_path=snapshot_download(MODEL,revision=REVISION,local_files_only=True)
    print('Loading cached forced aligner on CPU; CUDA disabled',flush=True)
    model=Qwen3ForcedAligner.from_pretrained(model_path,device_map='cpu',dtype=torch.float32,attn_implementation='sdpa')
    print('Aligning the supplied recording in one pass',flush=True)
    result=model.align(audio=str(audio),text=text,language='German')
    return [dict(text=w.text,start=float(w.start_time),end=float(w.end_time)) for w in result[0].items]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('trial',type=Path)
    args=parser.parse_args();trial=args.trial;output=trial/'continuous_local'
    audio=output/'CONTINUOUS_GERMAN.wav';video=output/'CONTINUOUS_GERMAN.mp4'
    report_path=output/'subtitle_alignment.json'
    if report_path.exists():raise FileExistsError('Alignment already exists; inspect it before rerunning')
    german=cues(trial/'input.de.srt');english=cues(trial/'input.en.srt')
    if len(german)!=10 or len(english)!=10:raise ValueError('Expected the ten selected cues')
    original_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (audio,video)}
    with wave.open(str(audio)) as wav:duration=wav.getnframes()/wav.getframerate()
    began=time.monotonic()
    words=align_words_cpu(audio,' '.join(c['text'].replace('\n',' ') for c in german))
    # Save measured word times before validation; a failed check does not require inference again.
    raw=dict(model=MODEL,revision=REVISION,device='cpu',calls=1,seconds=time.monotonic()-began,words=words,media_sha256=original_hashes)
    (output/'subtitle_alignment_words.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2)+'\n')
    spans=aligned_spans(german,words,duration)
    for lang,rows in (('de',german),('en',english)):
        data=render(rows,spans)
        (output/f'CONTINUOUS_GERMAN.aligned.{lang}.srt').write_bytes(data)
        canonical=output/('CONTINUOUS_GERMAN.srt' if lang=='de' else 'CONTINUOUS_GERMAN.en.srt')
        backup=canonical.with_name(canonical.stem+'.original-timing.srt')
        if canonical.exists() and not backup.exists():backup.write_bytes(canonical.read_bytes())
        temporary=canonical.with_suffix('.srt.tmp');temporary.write_bytes(data);temporary.replace(canonical)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in original_hashes.items())
    raw.update(status='complete',cues=[dict(index=c['index'],old=c['timing'],start=start,end=end,text=c['text'])
        for c,(start,end) in zip(german,spans)],note='English text uses the corresponding German cue spans; existing translations are unchanged. Embedded MP4 subtitles retain old timestamps. Load the external aligned SRT.')
    report_path.write_text(json.dumps(raw,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='complete',words=len(words),cues=len(spans),seconds=raw['seconds'],spans=spans)),flush=True)


if __name__=='__main__':main()
