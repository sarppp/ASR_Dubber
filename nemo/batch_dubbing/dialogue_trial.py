"""Source-linked old/new dialogue comparison at native speed. No model imports.

This measures the effect of wording, not source-performance transfer or exact
acoustic timing. Every complete native take is kept, including failed-to-fit ones.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import re

from batch_dubbing.native_excerpt_probe import (
    MODEL, REVISION, PACKAGE, IMAGE_ID, SAMPLE_RATE, digest, input_digest,
    validate_samples, encode_wave, decode_wave,
)

MAX_CALLS = 16
MAX_AUDIO_BYTES = 32 * 1024**2
MAX_STAGE_SECONDS = 180
MAX_CLIP_SECONDS = 30


def validate_scope(data):
    if not isinstance(data, dict) or type(data.get('schema')) is not int or data['schema'] != 1:
        raise ValueError('Expected dialogue trial schema 1')
    for key,value in dict(model=MODEL,revision=REVISION,package=PACKAGE,image_id=IMAGE_ID).items():
        if data.get(key)!=value:
            raise ValueError('Changed pinned '+key)
    if type(data.get('speed')) not in (int,float) or data['speed']!=1. or type(data.get('steps')) is not int or data['steps']!=8:
        raise ValueError('Native speed 1 and eight steps required')
    max_chars=data.get('max_text_chars',300)
    max_chunks=data.get('max_internal_chunks_per_take',1)
    if type(max_chars) is not int or max_chars not in (300,600) or type(max_chunks) is not int or max_chunks not in (1,2):
        raise ValueError('Explicit bounded text/chunk limits required')
    pairs=data.get('pairs')
    if not isinstance(pairs,list) or not 1<=len(pairs)<=MAX_CALLS//2:
        raise ValueError('Expected one to eight source phrase pairs')
    ids=set()
    for pair in pairs:
        if not isinstance(pair,dict): raise ValueError('Invalid pair')
        ident=pair.get('id')
        if not isinstance(ident,str) or not re.fullmatch(r'[a-z0-9_]{1,64}',ident) or ident in ids:
            raise ValueError('Invalid or duplicate source ID')
        ids.add(ident)
        if pair.get('voice') not in ('M1','M2','F1') or pair.get('language')!='de':
            raise ValueError('Explicit reviewed German voice required')
        if type(pair.get('seed')) is not int or not 0<=pair['seed']<2**32:
            raise ValueError('Explicit unsigned seed required')
        a,b=pair.get('start'),pair.get('end')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (a,b)) or not 0<=a<b<=7200 or b-a>30:
            raise ValueError('Invalid source bounds')
        for key in ('source_text','old_text','adapted_text','adaptation_note'):
            text=pair.get(key)
            limit=max_chars if key in ('old_text','adapted_text') else 2000
            if not isinstance(text,str) or not text.strip() or len(text)>limit or text!=' '.join(text.split()):
                raise ValueError('Invalid or overlong '+key)
        for key in ('source_indices','old_translation_indices'):
            indices=pair.get(key)
            if not isinstance(indices,list) or not indices or any(type(i) is not int or i<0 for i in indices) or indices!=list(range(indices[0],indices[-1]+1)):
                raise ValueError('Exact ordered '+key+' required')
        for key in ('source_json','original_video','older_dub'):
            path=pair.get(key)
            if not isinstance(path,str) or not path or Path(path).is_absolute() or '..' in Path(path).parts:
                raise ValueError('Relative source path required')
        if not re.fullmatch(r'[a-f0-9]{64}',pair.get('source_sha256','')):
            raise ValueError('Source metadata digest required')
    encoded=json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
    if len(encoded)>128*1024: raise ValueError('Input exceeds 128 KiB')
    return deepcopy(data)


def verify_sources(data,root):
    """Validate full English and old German text, source indices and endpoints."""
    data=validate_scope(data);root=Path(root).resolve()
    for pair in data['pairs']:
        paths={key:(root/pair[key]).resolve() for key in ('source_json','original_video','older_dub')}
        if any(not p.is_relative_to(root) or not p.is_file() for p in paths.values()):
            raise ValueError('Source file is missing or outside repository')
        raw=paths['source_json'].read_bytes()
        if digest(raw)!=pair['source_sha256']: raise ValueError('Source metadata changed')
        source=json.loads(raw)
        en=[source['english'][i] for i in pair['source_indices']]
        de=[source['translations']['de'][i] for i in pair['old_translation_indices']]
        if (pair['source_text']!=' '.join(c['text'] for c in en)
                or pair['old_text']!=' '.join(c['text'] for c in de)
                or pair['start']!=en[0]['start'] or pair['end']!=en[-1]['end']):
            raise ValueError('Source-linked text or bounds changed')
        if len({c.get('speaker') for c in en})!=1 or not en[0].get('speaker'):
            raise ValueError('Phrase crosses source speakers')
    return data


def jobs(data,chunker):
    data=validate_scope(data);rows=[]
    for pair in data['pairs']:
        for arm,key in [('old','old_text'),('adapted','adapted_text')]:
            text=pair[key]
            chunks=chunker(text,300)
            if (not isinstance(chunks,list) or not 1<=len(chunks)<=data.get('max_internal_chunks_per_take',1)
                    or any(not isinstance(c,str) or not c.strip() for c in chunks) or ' '.join(chunks)!=text):
                raise ValueError('Each reviewed phrase must preserve its text within the declared bound (default one internal chunk)')
            rows.append(dict(id=pair['id']+'_'+arm,pair_id=pair['id'],arm=arm,text=text,
                             voice=pair['voice'],language=pair['language'],seed=pair['seed'],
                             target_seconds=pair['end']-pair['start']))
    return rows


def smoke_artifact(path):
    path=Path(path)
    if path.stat().st_size>1024**2: raise ValueError('Oversized report')
    report=json.loads(path.read_text())
    row=report['samples'][0]
    if Path(row['file']).name!=row['file']: raise ValueError('Invalid smoke filename')
    wavpath=path.parent/row['file']
    if wavpath.stat().st_size>MAX_CLIP_SECONDS*SAMPLE_RATE*4+56: raise ValueError('Oversized smoke')
    return dict(report=report,wav=wavpath.read_bytes())


def validate_smoke(artifact,data,first):
    if not isinstance(artifact,dict) or set(artifact)!= {'report','wav'}: raise ValueError('Smoke required')
    report=artifact['report']
    if (report.get('status')!='complete' or report.get('stage')!='smoke' or report.get('input')!=data
            or report.get('input_sha256')!=input_digest(data) or report.get('calls')!=1
            or len(report.get('samples',[]))!=1 or report.get('playback_rate')!=1.
            or report.get('speech_trimmed') is not False or report.get('waveform_time_stretched') is not False):
        raise ValueError('Successful matching smoke required')
    row=report['samples'][0];payload=decode_wave(artifact['wav'])
    if (any(row.get(k)!=v for k,v in first.items()) or row.get('file')!=first['id']+'.wav'
            or row.get('sha256')!=digest(artifact['wav']) or row.get('samples_sha256')!=digest(payload)
            or row.get('native_frames')!=len(payload)//4 or len(payload)>MAX_CLIP_SECONDS*SAMPLE_RATE*4):
        raise ValueError('Smoke sample identity changed')
    return row


def run_trial(data,smoke,*,chunker,synthesize,prior=None):
    rows=jobs(data,chunker)
    if type(smoke) is not bool: raise ValueError('Boolean stage required')
    if smoke and prior is not None: raise ValueError('Smoke cannot reuse another result')
    first=None if smoke else validate_smoke(prior,data,rows[0])
    result=dict(status='running',stage='smoke' if smoke else 'workload',input=deepcopy(data),
                input_sha256=input_digest(data),calls=0,internal_calls=0,samples=[],audio={},
                playback_rate=1.,synthesis_speed=1.,speech_trimmed=False,waveform_time_stretched=False,
                timing_verified=False,naturalness_verified=False,duration_conditioning=False,
                pause_conditioning=False,pronunciation_verified=False,
                evidence='Complete native take duration versus source caption span; not acoustic speech alignment')
    if first:
        result['samples'].append({**deepcopy(first),'reused_from_smoke':True})
        result['audio'][first['file']]=prior['wav']
    frames_total=0
    try:
        for row in (rows[:1] if smoke else rows[1:]):
            result['calls']+=1
            result['internal_calls']+=len(chunker(row['text'],300))
            value=synthesize(deepcopy(row))
            payload=value['samples'];frames,peak=validate_samples(payload,value['sample_rate'])
            if frames>MAX_CLIP_SECONDS*SAMPLE_RATE or frames_total+frames>MAX_STAGE_SECONDS*SAMPLE_RATE:
                raise ValueError('Native duration budget exceeded')
            wav=encode_wave(payload)
            if sum(map(len,result['audio'].values()))+len(wav)>MAX_AUDIO_BYTES:
                raise ValueError('Audio transfer bound exceeded')
            frames_total+=frames
            record={**row,'file':row['id']+'.wav','sha256':digest(wav),'samples_sha256':digest(payload),
                    'native_frames':frames,'raw_duration_seconds':frames/SAMPLE_RATE,'peak':peak,
                    'caption_duration_difference_seconds':frames/SAMPLE_RATE-row['target_seconds'],
                    'reused_from_smoke':False}
            result['samples'].append(record);result['audio'][record['file']]=wav
        result['status']='complete'
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    return result
