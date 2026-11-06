"""Historical source-anchored T4 validation scaffold, superseded by local sync.

Not the accepted continuous-synced replay. No cloud run is authorized by this file.
"""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import modal

ROOT=Path(__file__).resolve().parents[2]
NEMO=ROOT/'nemo'
TRIAL=NEMO/'new_guys/crispr/deepseek_t4_10cues_2026-11-05'
DEST=TRIAL/'tower_pipeline_t4'
TOWER=NEMO/'new_guys/crispr/translation_trial/models/tower2b.gguf'
TOWER_SHA='5f044a360ffed31a6461ed9bc6f5c269b1d2e03922f1dcbb31e0d44b9877549a'
ULTRA_REV='73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b'
SOURCES=sorted(set(list(ROOT.glob('*.py'))+list(NEMO.glob('*.py'))+
                   list((ROOT/'qwen3-tts').glob('*.py'))+list((ROOT/'translate-gemma').glob('*.py'))+
                   list((NEMO/'batch_dubbing').glob('*.py'))))
image=(modal.Image.from_registry('ollama/ollama:0.17.5',add_python='3.11')
    .entrypoint([])
    .apt_install('ffmpeg','libsndfile1','sox','libsox-dev','build-essential')
    .pip_install('torch==2.8.0','torchaudio==2.8.0','torchvision==0.23.0')
    .pip_install('moondream==2.6.1','qwen-tts==0.1.1','faster-qwen3-tts==0.2.5',
                 'transformers==4.57.3','accelerate==1.12.0','huggingface_hub<1','numpy<2',
                 'soundfile','scipy','psutil','pysrt','tqdm','supertonic==1.3.1')
    .run_commands('python -m venv --system-site-packages /opt/align',
                  '/opt/align/bin/pip install qwen-asr==0.0.6',
                  'mkdir -p /app/nemo/.venv-local-asr/bin /app/nemo/.venv/bin /app/qwen3-tts/.venv/bin /app/translate-gemma/.venv/bin',
                  'ln -s $(which python) /app/nemo/.venv-local-asr/bin/python',
                  'ln -s /opt/align/bin/python /app/nemo/.venv/bin/python',
                  'ln -s $(which python) /app/qwen3-tts/.venv/bin/python',
                  'ln -s $(which python) /app/translate-gemma/.venv/bin/python')
    .env({'HF_HOME':'/cache/huggingface','HF_HUB_DISABLE_TELEMETRY':'1','TOKENIZERS_PARALLELISM':'false',
          'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
          'OLLAMA_HOST':'http://127.0.0.1:11434','OLLAMA_MODELS':'/cache/tower-pipeline-ollama',
          'OLLAMA_NUM_PARALLEL':'1','OLLAMA_MAX_LOADED_MODELS':'1',
          'ALIGNMENT_PYTHON':'/opt/align/bin/python','INPUT_DIR':'/input','OUTPUT_DIR':'/result',
          'ASR_MODEL':'parakeet-ultra','TRANSLATE_MODEL':'tower-plus-2b:latest','PYTHONUNBUFFERED':'1'})
    .add_local_file(TOWER,'/models/tower.gguf',copy=True))
for path in SOURCES:image=image.add_local_file(path,'/app/'+str(path.relative_to(ROOT)))
for name in ('source_10cues.mp4','voice_reference.wav','scope.json'):
    image=image.add_local_file(TRIAL/name,'/input/'+name)
volume=modal.Volume.from_name('speech-model-benchmark-20261027')
app=modal.App('tower-default-pipeline-29s-t4')


def pipeline_command(python):
    return [python,'-u','/app/run_pipeline.py','--pipeline','subtitle-dubbing',
        '--input-file','/input/source_10cues.mp4','--language','en','--target-lang','de',
        '--subtitle-languages','en','de','--asr-model','parakeet-ultra','--translator','tower',
        '--tts-engine','qwen','--qwen-mode','clone','--voice-reference','/input/voice_reference.wav',
        '--no-demucs','--tts-workers','1','--tts-devices','0','--dub-retries','0',
        '--dub-timing','source-anchored']


@app.function(image=image,gpu='T4',cpu=(4,4),memory=(24576,24576),
              timeout=600,startup_timeout=180,max_containers=1,min_containers=0,
              retries=0,scaledown_window=2,volumes={'/cache':volume})
def validate():
    import os
    import subprocess
    import sys
    import time
    import urllib.request
    sys.path.insert(0,'/app/nemo')
    from batch_dubbing.trial_telemetry import emit,start
    from batch_dubbing.aligned_finish import sha
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True,timeout=10).strip()
    if 'T4' not in gpu:raise RuntimeError('T4 required; no fallback')
    emit('worker_ready',gpu=gpu);monitor=start();began=time.monotonic();server=None
    report=dict(status='running',gpu=gpu,pipeline='subtitle-dubbing',asr='parakeet-ultra',translator='tower-plus-2b:latest',tts='Qwen3-TTS Base',source_seconds=29.487792,automatic_retries=0)
    artifacts=Path('/artifacts');artifacts.mkdir()
    try:
        scope=json.loads(Path('/input/scope.json').read_text())
        for name in ('source_10cues.mp4','voice_reference.wav'):
            if sha(Path('/input')/name)!=scope['input_sha256'][name]:raise ValueError('Input identity mismatch')
        if sha('/models/tower.gguf')!=TOWER_SHA:raise ValueError('Tower model identity mismatch')
        from huggingface_hub import snapshot_download
        from batch_dubbing.align_continuous_subtitles import MODEL,REVISION
        ultra=snapshot_download('moondream/parakeet-ultra',revision=ULTRA_REV)
        os.environ['PARAKEET_ULTRA_MODEL_PATH']=ultra
        os.environ['QWEN_TTS_MODEL_PATH']=snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base',revision='fd4b254389122332181a7c3db7f27e918eec64e3')
        snapshot_download(MODEL,revision=REVISION)
        os.environ['HF_HUB_OFFLINE']='1'
        server=subprocess.Popen(['ollama','serve'],stdout=(artifacts/'ollama.log').open('w'),stderr=subprocess.STDOUT)
        ready=False
        for _ in range(30):
            try:
                with urllib.request.urlopen('http://127.0.0.1:11434/api/version',timeout=1) as response:
                    report['ollama']=json.load(response);ready=True;break
            except Exception:time.sleep(.5)
        if not ready:raise RuntimeError('Ollama did not become ready')
        modelfile=Path('/tmp/Tower.Modelfile')
        modelfile.write_text('FROM /models/tower.gguf\nPARAMETER stop "<end_of_turn>"\nPARAMETER stop "<eos>"\nPARAMETER temperature 0.1\nPARAMETER top_p 0.9\nPARAMETER num_ctx 4096\n')
        subprocess.run(['ollama','create','tower-plus-2b:latest','-f',str(modelfile)],check=True,timeout=45)
        emit('pipeline_started',command=pipeline_command(sys.executable))
        # Execute the real entry point. No alternate ASR, translator or hand-built review path.
        with (artifacts/'pipeline.log').open('w') as log:
            result=subprocess.run(pipeline_command(sys.executable),cwd='/app',stdout=log,stderr=subprocess.STDOUT,timeout=max(30,560-(time.monotonic()-began)))
        report.update(returncode=result.returncode,status='complete' if result.returncode==0 else 'needs_review')
        if result.returncode==0:
            required=['final_dub.mp4','final_dub.en.srt','final_dub.de.srt','final_dub.multi.srt','final_dub.multi.ass','alignment_report.json']
            report['outputs']={}
            for name in required:
                matches=list(Path('/result').rglob(name))
                if len(matches)!=1:raise ValueError(f'Missing or ambiguous final output: {name}')
                report['outputs'][str(matches[0])]=sha(matches[0])
            alignment=json.loads(next(Path('/result').rglob('alignment_report.json')).read_text())
            if alignment['status']!='complete' or not alignment['video_packets_unchanged'] or alignment['tempo']!=1.:
                raise ValueError('Final alignment/picture gate did not pass')
            report['alignment_summary']={k:alignment[k] for k in ('status','video_seconds','warnings','ending_gap_seconds','subtitle_languages')}
        report['pipeline_log_tail']=(artifacts/'pipeline.log').read_text()[-6000:]
        print(report['pipeline_log_tail'],flush=True)
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    finally:
        if server:
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill();server.wait()
        monitor.set();report['seconds']=time.monotonic()-began
        (artifacts/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        # Preserve failed raw takes and native-fit reports as well as successful outputs.
        paths=[artifacts,Path('/result'),Path('/app/qwen3-tts/output'),Path('/input/.subtitle_dub')]
        for path in Path('/input').glob('*.srt'):paths.append(path)
        for path in Path('/input').glob('*.json'):paths.append(path)
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
            for path in paths:
                if path.exists():archive.add(path,arcname=str(path).lstrip('/'),recursive=True)
        volume.commit();emit('worker_complete',status=report['status'],seconds=report['seconds'])
        if len(buffer.getvalue())>100_000_000:raise ValueError('Artifact archive exceeds bound')
    return buffer.getvalue()


@app.local_entrypoint()
def main():
    if DEST.exists():raise FileExistsError('One run only; preserve previous output')
    payload=validate.remote();DEST.mkdir()
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
        archive.extractall(DEST,filter='data')
    report=json.loads((DEST/'artifacts/report.json').read_text())
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if report['status']!='complete':raise RuntimeError('Default pipeline requires review; no retry authorized')
