"""Cached CPU old/new dialogue trial. Remote inference requires controller release."""
import hashlib
import json
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[1]
image=modal.Image.from_id('im-40qUTZlx5ttgX1nwrQdvyw')
for name in ('cheap_native_probe.py','native_excerpt_probe.py','dialogue_trial.py'):
    image=image.add_local_file(ROOT/'batch_dubbing'/name,'/work/batch_dubbing/'+name)
image=image.add_local_file(ROOT/'speech_model_benchmark_2026-10-27/code/telemetry.py','/work/telemetry.py')
volume=modal.Volume.from_name('speech-model-benchmark-20261027')
app=modal.App('meaning-first-dialogue-trial')


def save_result(result,destination):
    metadata={k:v for k,v in result.items() if k!='audio'}
    expected={}
    for row in metadata.get('samples',[]):
        name=row.get('file')
        if not isinstance(name,str) or Path(name).name!=name or not name.endswith('.wav') or name in expected:
            raise ValueError('Invalid result filename')
        expected[name]=row.get('sha256')
    encoded=json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    audio=result.get('audio')
    if (not isinstance(audio,dict) or set(audio)!=set(expected)
            or any(not isinstance(v,bytes) for v in audio.values())
            or sum(map(len,audio.values()))>32*1024**2 or len(encoded.encode())>1024**2
            or any(hashlib.sha256(v).hexdigest()!=expected[k] for k,v in audio.items())):
        raise ValueError('Result integrity/size check failed')
    destination=Path(destination)
    if not destination.is_dir() or any(destination.iterdir()): raise FileExistsError('Output must be reserved and empty')
    for name,content in audio.items():
        with (destination/name).open('xb') as f:f.write(content)
    with (destination/'report.json').open('x') as f:f.write(encoded)
    if metadata.get('status')!='complete': raise RuntimeError('Trial failed; partial audio/report retained; no retry')


@app.function(image=image,cpu=(2,2),memory=(4096,4096),timeout=150,
              startup_timeout=30,scaledown_window=2,min_containers=0,
              max_containers=1,retries=0,volumes={'/cache':volume})
def probe(data,smoke=False,prior=None):
    import os
    import sys
    import time
    from importlib.metadata import version
    os.environ.update(HF_HOME='/cache/huggingface',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
                      OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    sys.path.insert(0,'/work')
    from telemetry import emit,start
    emit('worker_ready',hardware='CPU',max_containers=1,smoke=smoke)
    monitor=start();began=time.monotonic()
    try:
        from batch_dubbing.dialogue_trial import MODEL,REVISION,jobs,validate_smoke,run_trial
        if version('supertonic')!='1.3.1':raise ValueError('Unexpected cached package')
        from supertonic.utils import chunk_text
        rows=jobs(data,chunk_text)
        if type(smoke) is not bool or (smoke and prior is not None):raise ValueError('Invalid smoke stage')
        if not smoke:validate_smoke(prior,data,rows[0])
        import numpy as np
        from huggingface_hub import snapshot_download
        from supertonic import TTS
        model_dir=snapshot_download(MODEL,revision=REVISION,local_files_only=True)
        model=TTS(model='supertonic-3',model_dir=model_dir,auto_download=False,
                  intra_op_num_threads=2,inter_op_num_threads=1)
        def synthesize(row):
            np.random.seed(row['seed'])
            emit('segment_start',id=row['id'],language=row['language'])
            audio,_=model.synthesize(row['text'],voice_style=model.get_voice_style(row['voice']),
                lang=row['language'],speed=1.,total_steps=8,max_chunk_length=300,silence_duration=.3)
            audio=np.asarray(audio,dtype=np.float32).squeeze()
            if audio.ndim!=1 or not 0<audio.size<=30*44100:raise ValueError('Invalid or overlong native audio')
            emit('segment_complete',id=row['id'],frames=int(audio.size))
            return dict(samples=audio.astype('<f4',copy=False).tobytes(),sample_rate=model.sample_rate)
        result=run_trial(data,smoke,chunker=chunk_text,synthesize=synthesize,prior=prior)
        result['worker_seconds']=time.monotonic()-began
        emit('worker_complete',status=result['status'],calls=result['calls'])
        return result
    finally:monitor.set()


@app.local_entrypoint()
def main(input_file:str,output_root:str,smoke:bool=False,smoke_result:str=''):
    import sys
    sys.path.insert(0,str(ROOT))
    from batch_dubbing.dialogue_trial import verify_sources,smoke_artifact
    path=Path(input_file)
    if path.stat().st_size>128*1024:raise ValueError('Oversized input')
    data=verify_sources(json.loads(path.read_text()),ROOT)
    if (smoke and smoke_result) or (not smoke and not smoke_result):raise ValueError('Specify smoke or matching smoke result')
    prior=None if smoke else smoke_artifact(smoke_result)
    destination=Path(output_root)/('smoke' if smoke else 'workload')
    destination.mkdir(parents=True,exist_ok=False)
    try:save_result(probe.remote(data,smoke,prior),destination)
    except Exception as exc:
        if not (destination/'report.json').exists():
            with (destination/'report.json').open('x') as f:
                json.dump(dict(status='failed',input=data,error=str(exc)[:2000],inference_call_count_known=False),f)
        raise
