"""Modal T4 Qwen dubbing of the new source-linked phrase translations."""
import json
from pathlib import Path
import time
import modal

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / 'experiments/crispr_phrase_modal_2026-10-30_10min_qwen'
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('ffmpeg', 'libsndfile1', 'sox', 'libsox-dev')
         .pip_install('torch==2.6.0', 'torchaudio==2.6.0', 'torchvision==0.21.0')
         .pip_install('qwen-tts==0.1.1', 'transformers==4.57.3', 'accelerate',
                      'huggingface_hub<1', 'soundfile', 'numpy<2', 'scipy', 'psutil')
         .env({'HF_HOME': '/cache/huggingface', 'OMP_NUM_THREADS': '4',
               'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
         .add_local_file(ROOT / 'batch_dubbing/trial_telemetry.py', '/work/telemetry.py'))
for name in ('natural_pipeline', 'audio_edges', 'speech_schedule', 'natural_render', 'cpu_natural', 'phrase_trial_qwen'):
    image = image.add_local_file(ROOT / f'batch_dubbing/{name}.py', f'/work/batch_dubbing/{name}.py')
volume = modal.Volume.from_name('speech-model-benchmark-20261027')
app = modal.App('crispr-10min-qwen-20261030-dub')


@app.function(image=image, gpu='T4', cpu=(4, 4), memory=(32768, 32768), timeout=3600,
              startup_timeout=180, max_containers=1, min_containers=0, retries=0,
              scaledown_window=2, volumes={'/cache': volume})
def synthesize(data, seconds):
    import sys
    sys.path.insert(0, '/work')
    from telemetry import emit, start
    import subprocess
    from batch_dubbing.phrase_trial_qwen import synthesize_qwen
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                  text=True, timeout=15).strip()
    if 'T4' not in gpu:
        raise RuntimeError('This release permits T4 only')
    emit('worker_ready', gpu=gpu, seconds=seconds)
    monitor = start()
    try:
        return synthesize_qwen(data, seconds, emit=emit, commit=volume.commit)
    finally:
        volume.commit()
        monitor.set()


@app.local_entrypoint()
def main(smoke: bool = False):
    from batch_dubbing.phrase_trial import write_json, sha256
    from batch_dubbing.cpu_natural import select_units
    source = TRIAL / 'translated.json'
    data = json.loads(source.read_text())
    destination = TRIAL / ('dub_smoke.json' if smoke else 'dub.json')
    audio = TRIAL / ('dub_smoke.m4a' if smoke else 'german-dub.m4a')
    if destination.exists() or audio.exists():
        raise FileExistsError('Dubbing test artifacts already exist')
    seconds = data['duration_seconds']
    if smoke:
        first = data['speech_units']['de'][:2]
        seconds = max(first[-1]['source_end'], first[-1]['corridor_end'])
        data['speech_units']['de'] = first
    select_units(data, seconds, 'de')
    started = time.monotonic()
    result = synthesize.remote(data, seconds)
    report = result['report']
    report.update(client_call_seconds=time.monotonic()-started, input_sha256=sha256(source))
    write_json(destination, report)
    if report.get('status') != 'complete' or not result.get('audio'):
        raise RuntimeError('Speech could not fit safely; raw takes and timing diagnostics retained on Modal')
    with audio.open('xb') as stream:
        stream.write(result['audio'])
    print('RESULT', destination, flush=True)
