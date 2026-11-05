"""Single bounded T4 call, after a source-pinned offline smoke receipt."""
import hashlib
import io
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / 'new_guys/crispr/deepseek_t4_10cues_2026-11-05'
image = (modal.Image.debian_slim(python_version='3.11')
    .apt_install('ffmpeg', 'libsndfile1', 'sox', 'libsox-dev')
    .pip_install('torch==2.6.0', 'torchaudio==2.6.0', 'torchvision==0.21.0')
    .pip_install('qwen-tts==0.1.1', 'faster-qwen3-tts==0.2.5', 'transformers==4.57.3',
                 'accelerate', 'huggingface_hub<1', 'soundfile', 'numpy<2', 'scipy', 'psutil')
    .env({'HF_HOME':'/cache/huggingface', 'HF_HUB_OFFLINE':'1',
          'OMP_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4',
          'TOKENIZERS_PARALLELISM':'false'})
    .add_local_file(ROOT/'batch_dubbing/continuous_german.py', '/work/continuous_german.py')
    .add_local_file(ROOT/'batch_dubbing/trial_telemetry.py', '/work/telemetry.py'))
for name in ('input.de.srt', 'voice_reference.wav', 'scope.json'):
    image = image.add_local_file(TRIAL/name, '/input/'+name)
volume = modal.Volume.from_name('speech-model-benchmark-20261027')
app = modal.App('continuous-german-ten-cues-t4')


@app.function(image=image, gpu='T4', cpu=(4,4), memory=(24576,24576),
              timeout=240, startup_timeout=120, max_containers=1,
              min_containers=0, retries=0, scaledown_window=2, volumes={'/cache':volume})
def generate():
    import subprocess
    import sys
    import time
    sys.path.insert(0, '/work')
    from telemetry import emit, start
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],
                                  text=True,timeout=10).strip()
    if 'T4' not in gpu:
        raise RuntimeError('Only T4 is authorized')
    emit('worker_ready', gpu=gpu)
    monitor = start()
    began = time.monotonic()
    try:
        from continuous_german import MODEL, REVISION, synthesize, passage
        scope = json.loads(Path('/input/scope.json').read_text())
        for name in ('input.de.srt','voice_reference.wav'):
            if hashlib.sha256(Path('/input',name).read_bytes()).hexdigest() != scope['input_sha256'][name]:
                raise ValueError('Input identity changed')
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
        import torch
        import numpy as np
        import soundfile as sf
        from faster_qwen3_tts import FasterQwen3TTS
        torch.manual_seed(20261105)
        model = FasterQwen3TTS.from_pretrained(model_path, device='cuda:0', dtype=torch.bfloat16)
        emit('synthesis_started', calls=1, characters=len(passage('/input/input.de.srt')))
        wavs, sr = synthesize(model, '/input/input.de.srt', '/input/voice_reference.wav')
        if len(wavs) != 1:
            raise ValueError('Expected one complete take')
        audio = wavs[0].cpu().numpy() if torch.is_tensor(wavs[0]) else np.asarray(wavs[0])
        if audio.ndim != 1 or sr != 24000 or not np.isfinite(audio).all() or not 10 < audio.size/sr < 49:
            raise ValueError('Invalid or possibly token-capped take; no retry')
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format='WAV', subtype='PCM_16')
        data = buffer.getvalue()
        report = dict(status='complete', synthesis_calls=1, gpu=gpu, model=MODEL,
            revision=REVISION, text=passage('/input/input.de.srt'), audio_seconds=audio.size/sr,
            audio_sha256=hashlib.sha256(data).hexdigest(), worker_seconds=time.monotonic()-began,
            note='One continuous model request; pronunciation and completeness need listening review.')
        emit('worker_complete', audio_seconds=report['audio_seconds'])
        return dict(audio=data, report=report)
    finally:
        monitor.set()


@app.local_entrypoint()
def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from batch_dubbing.continuous_german import save_result
    destination = TRIAL/'continuous'
    if destination.exists():
        raise FileExistsError('Refusing another run over existing result')
    save_result(generate.remote(), destination)
    print('CONTINUOUS_RESULT', destination, flush=True)
