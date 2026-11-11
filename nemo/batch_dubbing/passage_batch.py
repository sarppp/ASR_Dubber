"""Source-anchored batch orchestration using the existing natural pipeline.

Providers are supplied by the authorized local/Modal controller; this module
loads no model and launches no cloud work. Processing is serial across assets.
"""
import array
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

from .natural_pipeline import run_pipeline,validate_units
from .speech_units import build_speech_units,bind_translations


def digest_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024*1024):digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def prepare_asset(data):
    """Build complete source passages and require every source word exactly once."""
    if not isinstance(data.get('id'),str) or not data['id'].strip():raise ValueError('Asset ID required')
    if not isinstance(data.get('source_id'),str) or not data['source_id']:raise ValueError('Source identity required')
    source=build_speech_units(data['source_words'],data['turns'],source_id=data['source_id'])
    # Translation keyed to stable passage IDs; no correspondence inferred from cue count.
    units=[u.to_dict() for u in bind_translations(source,data['translations'])]
    validated=validate_units(units,data['duration_seconds'])
    indices=[i for u in validated for i in u['source_word_indices']]
    if indices!=list(range(len(data['source_words']))):raise ValueError('Incomplete source coverage')
    for unit in validated:
        unit["source_language"] = data.get("source_language", "en")
        unit["target_language"] = data.get("target_language", "de")
    return validated


class CachedSynthesis:
    """Keep exact native takes across retries, bound to model/voice/source/text."""
    def __init__(self,root,provider,provider_identity,max_calls):
        if not provider_identity:raise ValueError('Pin model, revision, voice/reference and settings')
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.provider=provider;self.provider_identity=provider_identity
        self.max_calls=max_calls;self.calls=0;self.hits=0

    def __call__(self,unit,text):
        key=identity(dict(unit=unit,text=text,provider=self.provider_identity))
        entry=self.root/key
        if entry.exists():
            metadata=json.loads((entry/'metadata.json').read_text())
            if metadata['key']!=key or digest_file(entry/'samples.f32')!=metadata['sha256']:
                raise ValueError('Corrupt speech cache; refusing silent reuse or retry')
            raw=(entry/'samples.f32').read_bytes();samples=array.array('f');samples.frombytes(raw)
            if sys.byteorder!='little':samples.byteswap()
            self.hits+=1
            return dict(samples=samples,sample_rate=metadata['sample_rate'],voice=metadata.get('voice'),predicted_duration=metadata.get('predicted_duration'))
        if self.calls>=self.max_calls:raise ValueError('Actual synthesis budget exhausted')
        self.calls+=1
        result=self.provider(deepcopy(unit),text)
        sr=result['sample_rate'];samples=array.array('f',result['samples'])
        if sr!=44100 or not 0<len(samples)<=120*sr:raise ValueError('Unexpected synthesis format or length')
        if sys.byteorder!='little':samples.byteswap()
        raw=samples.tobytes()
        with tempfile.TemporaryDirectory(dir=self.root,prefix='pending-') as temporary:
            path=Path(temporary);(path/'samples.f32').write_bytes(raw)
            metadata=dict(key=key,sha256=hashlib.sha256(raw).hexdigest(),sample_rate=sr,
                          voice=result.get('voice'),predicted_duration=result.get('predicted_duration'))
            (path/'metadata.json').write_text(json.dumps(metadata,allow_nan=False))
            # A cache entry becomes visible only when both files are complete.
            ready=path/'ready';ready.mkdir()
            (path/'samples.f32').replace(ready/'samples.f32');(path/'metadata.json').replace(ready/'metadata.json')
            ready.rename(entry)
        return result


def _run_batch(assets,*,output_dir,synthesize,content_check,provider_identity,render_ready=None):
    """Run a bounded serial batch, isolate failures, resume exact raw takes.

    content_check(unit,take) must return hash-bound speech-coverage AND semantic
    receipts accepted by natural_pipeline. Missing/failed evidence blocks output.
    render_ready(asset,report,attempt_dir) is the existing CPU rendering/mux stage;
    it is never called on partial or unverified speech. No automatic model retry.
    """
    if not isinstance(assets,list) or not 1<=len(assets)<=59:raise ValueError('Batch must contain 1..59 explicit assets')
    ids=[a.get('id') for a in assets]
    if any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):raise ValueError('Unique asset IDs required')
    if render_ready is None:
        render_ready = publish_ready
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    summary=dict(status='running',assets=[],max_parallel=1,automatic_retries=0,actual_synthesis_calls=0,cache_hits=0)
    def persist():
        temporary=output/'batch.json.tmp';temporary.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');temporary.replace(output/'batch.json')
    persist()
    for asset in assets:
        row=dict(id=asset['id'],status='running');summary['assets'].append(row);persist()
        cache=None
        try:
            if 'manifest' in asset:
                ident=asset['id']
                asset=json.loads(Path(asset['manifest']).read_text())
                if asset.get('id')!=ident:raise ValueError('Asset manifest identity changed')
            units=prepare_asset(asset)
            if render_ready is publish_ready:
                verify_asset_video(asset)
            asset_root=output/identity(dict(id=asset['id'],source=asset['source_id']))
            asset_root.mkdir(exist_ok=True)
            attempt=Path(tempfile.mkdtemp(prefix='attempt-',dir=asset_root));work=attempt/'synthesis'
            row['report']=str(work/'report.json')
            cache=CachedSynthesis(asset_root/'raw_cache',synthesize,provider_identity,len(units))
            report=run_pipeline(units,duration_seconds=asset['duration_seconds'],output_dir=work,
                synthesize=cache,max_synthesis_calls=len(units),max_revisions_per_unit=0,
                max_rate=1.,max_shift=.25,max_end_gap=.4,end_gap_policy='warn',expected_source_words=asset['source_words'],
                require_complete_passages=True,content_check=content_check,require_content_check=True)
            row.update(status=report['status'],passages=len(units),source_words=len(asset['source_words']))
            if report['status']=='ready' and render_ready:
                row['rendered']=render_ready(asset,report,attempt)
                row['status']='complete'
        except Exception as exc:
            row.update(status='failed',error=str(exc))
        if cache:
            row.update(actual_synthesis_calls=cache.calls,cache_hits=cache.hits)
            summary['actual_synthesis_calls']+=cache.calls;summary['cache_hits']+=cache.hits
        persist()
    summary['status']='complete' if all(r['status']=='complete' for r in summary['assets']) else 'needs_review'
    persist()
    return summary


def verify_asset_video(asset):
    """Reject changed or missing video before any model call and before muxing."""
    from .anchored_timeline import video_duration
    video=Path(asset['video'])
    if digest_file(video)!=asset['video_sha256']:raise ValueError('Source video identity changed')
    if abs(video_duration(video)-asset['duration_seconds'])>.05:raise ValueError('Source video duration differs from its word timeline')
    return video


def publish_ready(asset,report,attempt):
    """Publish verified native speech and aligned target captions on the full video."""
    import re
    import subprocess
    from .natural_render import render_timeline
    from .anchored_timeline import video_duration,packet_hash
    from .align_continuous_subtitles import render,tokens
    if report.get('status')!='ready' or any(not x['passed'] for x in report['content_checks']):
        raise ValueError('Only fully verified speech can be published')
    if any(row['tempo']!=1. for row in report['render']['rows']):
        raise ValueError('Only native speech may be published by the passage batch')
    video=verify_asset_video(asset)
    language=asset.get('target_language','de');source_language=asset.get('source_language','en')
    if any(not re.fullmatch('[a-z]{2,3}',v) for v in (language,source_language)):raise ValueError('Invalid subtitle language code')
    selected={u['id']:u for u in report['units']}
    receipts={r['id']:r['receipt'] for r in report['content_checks']}
    takes={(t['id'],t['attempt']):t for t in report['attempts']}
    captions=[];spans=[]
    def pack(words):
        group=[]
        def finish():
            if group:
                captions.append(dict(index=len(captions)+1,text=' '.join(w['text'] for w in group)))
                spans.append((group[0]['start'],group[-1]['end']))
                group.clear()
        for word in words:
            if group and (len(group)>=10 or len(' '.join(w['text'] for w in group)+' '+word['text'])>80
                          or word['end']-group[0]['start']>5):finish()
            group.append(word)
        finish()
    for row in report['render']['rows']:
        unit=selected[row['id']];receipt=receipts[row['id']]
        take=takes[row['id'],report['selected_attempts'][row['id']]]
        words=receipt.get('aligned_words',[]);parts=take['text'].split()
        if len(parts)!=len(words) or tokens(take['text'])!=[t for w in words for t in tokens(w['text'])]:
            raise ValueError(f'Exact target word alignment required for {row["id"]}')
        shift=row['start']-take['edges']['start_sample']/take['edges']['sample_rate']
        absolute=[];previous=row['start']
        for word,text in zip(words,parts):
            start,end=word['start']+shift,word['end']+shift
            if not row['start']<=start<end<=row['end'] or start<previous:
                raise ValueError('Word alignment falls outside the accepted speech')
            absolute.append(dict(text=text,start=start,end=end));previous=end
        pack(absolute)
    attempt=Path(attempt)
    target_srt=attempt/f'dub.{language}.srt';target_srt.write_bytes(render(captions,spans))
    captions=[];spans=[]
    for unit in report['units']:
        pack([dict(text=w['word'],start=w['start'],end=w['end']) for w in unit['words']])
    source_srt=attempt/f'dub.source.{source_language}.srt';source_srt.write_bytes(render(captions,spans))
    audio=attempt/'timeline.wav';rendered=render_timeline(report['render'],audio,max_wall_seconds=600)
    movie=attempt/'dub.mp4'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-i',str(video),'-i',str(audio),'-i',str(target_srt),
        '-map','0:v','-map','1:a:0','-map','2:0','-c:v','copy','-c:a','aac','-b:a','192k','-c:s','mov_text',
        '-movflags','+faststart',str(movie)],check=True,timeout=600)
    if packet_hash(video)!=packet_hash(movie) or abs(video_duration(movie)-asset['duration_seconds'])>.05:
        raise ValueError('Picture preservation check failed')
    if digest_file(video)!=asset['video_sha256']:raise ValueError('Source changed during rendering')
    return dict(video=str(movie),target_subtitle=str(target_srt),source_subtitle=str(source_srt),
                video_packets_unchanged=True,timeline=rendered,
                sha256={str(p):digest_file(p) for p in (movie,target_srt,source_srt,audio)})


def run_batch(assets,*,output_dir,synthesize,content_check,provider_identity,render_ready=None):
    """Exclusive per-output supervisor; manifests can be loaded one asset at a time."""
    import fcntl
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    with (output/'batch.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another supervisor owns this batch output')
        return _run_batch(assets,output_dir=output,synthesize=synthesize,content_check=content_check,
                          provider_identity=provider_identity,render_ready=render_ready)


def audit_batch(assets):
    """Read-only source preflight, one manifest at a time; zero model calls."""
    if not isinstance(assets,list) or not 1<=len(assets)<=59:raise ValueError('Expected 1..59 assets')
    identifiers=[a.get('id') for a in assets]
    if any(not isinstance(i,str) or not i for i in identifiers) or len(set(identifiers))!=len(identifiers):
        raise ValueError('Unique explicit asset IDs required')
    result=dict(status='ready_for_synthesis',assets=[],duration_seconds=0.,passages=0,source_words=0,model_calls=0)
    for descriptor in assets:
        row=dict(id=descriptor['id'])
        try:
            asset=json.loads(Path(descriptor['manifest']).read_text()) if 'manifest' in descriptor else descriptor
            if asset['id']!=descriptor['id']:raise ValueError('Manifest identity mismatch')
            units=prepare_asset(asset)
            flags=[dict(id=u['id'],flags=u['flags']) for u in units if set(u['flags']) &
                   {'unfinished_at_size_limit','unfinished_at_input_end','oversized_source_word'}]
            row.update(status='needs_source_review' if flags else 'ready_for_synthesis',passages=len(units),
                       source_words=len(asset['source_words']),duration_seconds=asset['duration_seconds'],source_issues=flags)
            result['duration_seconds']+=asset['duration_seconds'];result['passages']+=len(units);result['source_words']+=len(asset['source_words'])
        except Exception as exc:row.update(status='failed',error=str(exc))
        result['assets'].append(row)
    if any(r['status']!='ready_for_synthesis' for r in result['assets']):result['status']='needs_source_review'
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description='Offline source-passage preflight; never loads models or launches jobs.')
    parser.add_argument('--manifest',type=Path,required=True,help='JSON list of assets or {assets: [...]}')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    data=json.loads(args.manifest.read_text());assets=data['assets'] if isinstance(data,dict) else data
    for asset in assets:
        if 'manifest' in asset and not Path(asset['manifest']).is_absolute():
            asset['manifest']=str((args.manifest.parent/asset['manifest']).resolve())
    report=audit_batch(assets)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='assets'}))
    return 0 if report['status']=='ready_for_synthesis' else 1


if __name__=='__main__':raise SystemExit(main())
