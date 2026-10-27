"""Render only the requested current Qwen 1.7B dub on T4, with durable segment checkpoints."""
import io,json,tarfile,time,hashlib
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]/'speech_model_benchmark_2026-10-27'
OUT=ROOT/'07_german_dub_comparison'
SPEC=json.loads((OUT/'spec.json').read_text()) if modal.is_local() else {}
image=(modal.Image.debian_slim(python_version='3.11')
 .apt_install('ffmpeg','git','build-essential','cmake','pkg-config','libgomp1','libsndfile1')
 .pip_install('huggingface_hub','soundfile','numpy<2','psutil')
 .pip_install('torch==2.6.0','torchaudio==2.6.0','torchvision==0.21.0')
 .pip_install('qwen-tts==0.1.1','faster-qwen3-tts==0.2.5','transformers==4.57.3','accelerate')
 .env({'HF_HOME':'/cache/huggingface','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'})
 .add_local_dir(ROOT/'00_inputs','/input',ignore=['*.mp4'])
 .add_local_file(ROOT/'code/telemetry.py','/root/telemetry.py'))
volume=modal.Volume.from_name('speech-model-benchmark-20261027')
app=modal.App('german-dub-qwen17-t4')
@app.function(image=image,serialized=True,gpu='T4',cpu=4,memory=24576,timeout=1800,max_containers=1,retries=0,volumes={'/cache':volume})
def render(spec,smoke):
 import sys,subprocess,traceback,os
 import numpy as np,soundfile as sf,torch
 from huggingface_hub import snapshot_download
 from telemetry import emit,start
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv'],text=True)
 if 'T4' not in gpu:raise RuntimeError('T4 required; refusing any other GPU')
 emit('worker_ready',gpu=gpu,model=spec['repo'],smoke=smoke)
 stop=start();torch.set_num_threads(4);torch.manual_seed(42)
 key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()+Path('/input/voice_reference.wav').read_bytes()).hexdigest()[:20]
 out=Path('/cache/german-dub-renders')/key;out.mkdir(parents=True,exist_ok=True)
 def save(name,data):(out/name).write_text(json.dumps(data,indent=2,ensure_ascii=False))
 save('spec.json',spec);save('hardware.json',{'gpu':gpu});script=json.loads(Path('/input/german_script.json').read_text());script=script[:1] if smoke else script
 started=time.perf_counter();rows=[]
 try:
  from faster_qwen3_tts import FasterQwen3TTS
  local=snapshot_download(spec['repo'],revision=spec['revision'])
  model=FasterQwen3TTS.from_pretrained(local,device='cuda',dtype=torch.bfloat16,attn_implementation='sdpa')
  load=time.perf_counter()-started
  for i,seg in enumerate(script):
   audio=out/f'{i:03d}.wav';meta=out/f'{i:03d}.json'
   if audio.exists() and meta.exists():
    row=json.loads(meta.read_text());emit('segment_reused',index=i+1,total=len(script))
   else:
    begin=time.perf_counter()
    waves,sr=model.generate_voice_clone(text=seg['text'],language='German',ref_audio='/input/voice_reference.wav',ref_text=Path('/input/voice_reference.txt').read_text().strip())
    torch.cuda.synchronize();wav=waves[0];wav=wav.detach().cpu().numpy() if torch.is_tensor(wav) else np.asarray(wav);wav=wav.squeeze()
    if not np.isfinite(wav).all() or len(wav)/sr<1 or float(np.sqrt(np.mean(wav.astype(float)**2)))<1e-5:raise RuntimeError(f'Invalid or silent output at segment {i}')
    sf.write(audio,wav,sr);row={**seg,'index':i,'path':audio.name,'audio_seconds':len(wav)/sr,'synthesis_seconds':time.perf_counter()-begin,'sample_rate':sr};save(meta.name,row);volume.commit()
    emit('segment_complete',index=i+1,total=len(script),seconds=row['synthesis_seconds'],audio_seconds=row['audio_seconds'])
   rows.append(row);save('segments.json',rows)
  duration=sum(x['audio_seconds'] for x in rows);seconds=sum(x['synthesis_seconds'] for x in rows)
  save('metrics.json',{'status':'complete','source_seconds':25 if smoke else 600,'segments':len(rows),'generated_audio_seconds':duration,'synthesis_seconds':seconds,'realtime_multiple':duration/seconds,'load_seconds_this_call':load,'gpu':gpu,'runtime':spec['runtime'],'dtype':spec['dtype'],'note':'First segment reused from preflight; per-segment times include its original synthesis.'})
  (out/'failure.json').unlink(missing_ok=True)
 except Exception as exc:
  save('failure.json',{'status':'failed','error':str(exc),'traceback':traceback.format_exc()});traceback.print_exc()
 finally:
  (out/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True));volume.commit();stop.set()
 buf=io.BytesIO()
 with tarfile.open(fileobj=buf,mode='w:gz') as tar:
  for p in out.iterdir():
   if p.is_file():tar.add(p,arcname=p.name)
 return buf.getvalue()
@app.local_entrypoint()
def main(smoke:bool=False):
 dest=OUT/('qwen_17b_preflight' if smoke else 'qwen_17b_t4');dest.mkdir(parents=True,exist_ok=True)
 data=render.remote(SPEC,smoke)
 with tarfile.open(fileobj=io.BytesIO(data)) as tar:tar.extractall(dest,filter='data')
 print('RESULT_DIRECTORY',dest,flush=True)
 if (dest/'failure.json').exists():raise RuntimeError((dest/'failure.json').read_text())
