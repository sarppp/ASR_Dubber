"""Fresh CRISPR ASR in a single T4 container; no Canary translation."""
import json
from pathlib import Path
import time
import wave
import modal

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / 'experiments/crispr_phrase_modal_2026-10-30_10min_qwen'
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('ffmpeg', 'git', 'build-essential', 'libgomp1', 'libsndfile1')
         .pip_install('torch==2.6.0', 'torchaudio==2.6.0', 'torchvision==0.21.0')
         .pip_install('nemo_toolkit[asr]==2.5.3', 'numpy<2', 'pyarrow<22', 'transformers==4.53.3',
                      'librosa', 'soundfile', 'psutil', 'huggingface_hub<1')
         .env({'HF_HOME': '/cache/huggingface', 'OMP_NUM_THREADS': '4',
               'TOKENIZERS_PARALLELISM': 'false', 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': '1'})
         .add_local_file(TRIAL / 'input/source.wav', '/input/source.wav')
         .add_local_file(ROOT / 'modal_control/smoke_nemo_helpers.py', '/work/smoke_nemo_helpers.py')
         .add_local_file(ROOT / 'modal_control/speaker_voices.py', '/work/speaker_voices.py')
         .add_local_file(ROOT / 'nemo_audio.py', '/work/nemo_audio.py')
         .add_local_file(ROOT / 'local_subtitles.py', '/work/local_subtitles.py')
         .add_local_file(ROOT / 'batch_dubbing/trial_telemetry.py', '/work/telemetry.py')
         .add_local_file(ROOT / 'batch_dubbing/speech_units.py', '/work/batch_dubbing/speech_units.py')
         .add_local_file(ROOT / 'batch_dubbing/speech_stage.py', '/work/batch_dubbing/speech_stage.py'))
volume = modal.Volume.from_name('speech-model-benchmark-20261027')
app = modal.App('crispr-10min-qwen-20261030-asr')


@app.function(image=image, gpu='T4', cpu=(4, 4), memory=(32768, 32768), timeout=900,
              startup_timeout=180, max_containers=1, min_containers=0, retries=0,
              scaledown_window=2, volumes={'/cache': volume})
def process(seconds):
    import sys
    sys.path.insert(0, '/work')
    from batch_dubbing.speech_stage import process_audio
    try:
        return process_audio('/input/source.wav', seconds, transcribe_only=True,
                             max_source_units=400, commit_artifacts=volume.commit)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(smoke: bool = False):
    from batch_dubbing.phrase_trial import check_audio_input, write_json, validate_source_duration
    audio, manifest = check_audio_input(TRIAL)
    destination = TRIAL / ('asr_smoke.json' if smoke else 'asr.json')
    if destination.exists():
        raise FileExistsError(destination)
    with wave.open(str(audio)) as source:
        seconds = validate_source_duration(source.getnframes() / source.getframerate())
    if smoke:
        seconds = min(74.24, seconds)
    started = time.monotonic()
    result = process.remote(seconds)
    result['client_call_seconds'] = time.monotonic() - started
    result['input_manifest'] = manifest
    write_json(destination, result)
    if result.get('status') != 'complete' or result.get('translation_performed') is not False:
        raise RuntimeError('Fresh ASR needs review; preserved diagnostics')
    print('RESULT', destination, flush=True)
