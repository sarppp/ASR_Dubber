"""Controller-supervised GPU smoke: source ASR, diarization and translations.

This is a preparation-stage worker. It does not synthesize or launch other
workers. The controller must pin this file, every helper and the WAV input.
"""
from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('ffmpeg', 'git', 'build-essential', 'libgomp1', 'libsndfile1')
         .pip_install('torch==2.6.0', 'torchaudio==2.6.0', 'torchvision==0.21.0')
         .pip_install('nemo_toolkit[asr]==2.5.3', 'numpy<2', 'pyarrow<22', 'transformers<5', 'librosa', 'soundfile', 'psutil', 'huggingface_hub')
         .env({'HF_HOME': '/cache/huggingface', 'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false', 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': '1'})
         .add_local_file(ROOT / 'modal_control/batch_inputs/smoke.wav', '/input/smoke.wav')
         .add_local_file(ROOT / 'modal_control/smoke_nemo_helpers.py', '/work/smoke_nemo_helpers.py')
         .add_local_file(ROOT / 'nemo_audio.py', '/work/nemo_audio.py')
         .add_local_file(ROOT / 'local_subtitles.py', '/work/local_subtitles.py')
         .add_local_file(ROOT / 'batch_dubbing/speech_units.py', '/work/batch_dubbing/speech_units.py')
         .add_local_file(ROOT / 'batch_dubbing/speech_stage.py', '/work/batch_dubbing/speech_stage.py')
         .add_local_file(ROOT / 'modal_control/speaker_timeline.py', '/work/speaker_timeline.py')
         .add_local_file(ROOT / 'modal_control/speaker_voices.py', '/work/speaker_voices.py')
         .add_local_file(ROOT / 'speech_model_benchmark_2026-10-27/code/telemetry.py', '/work/telemetry.py'))
volume = modal.Volume.from_name('speech-model-benchmark-20261027')
app = modal.App('new-guys-speech-smoke')


@app.function(image=image, gpu='T4', cpu=4, memory=32768, timeout=1800,
              max_containers=1, retries=0, volumes={'/cache': volume})
def process(seconds):
    import sys
    sys.path.insert(0, '/work')
    from batch_dubbing.speech_stage import process_audio
    try:
        return process_audio('/input/smoke.wav', seconds, commit_artifacts=volume.commit)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(smoke: bool = False):
    destination = ROOT / 'modal_control/batch_inputs' / ('gpu_preflight.json' if smoke else 'gpu_smoke.json')
    result = process.remote(15 if smoke else 60)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print('RESULT', destination, flush=True)
    if result.get('status') != 'complete':
        raise RuntimeError(f"Speech stage blocked; review preserved diagnostics: {destination}")
