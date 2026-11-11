"""Controller-supervised video ASR, diarization and speech translation."""
from pathlib import Path
import json
import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('ffmpeg', 'git', 'build-essential', 'libgomp1', 'libsndfile1')
         .pip_install('torch==2.6.0', 'torchaudio==2.6.0', 'torchvision==0.21.0')
         .pip_install('nemo_toolkit[asr]==2.5.3', 'numpy<2', 'pyarrow<22', 'transformers<5', 'librosa', 'soundfile', 'psutil', 'huggingface_hub')
         .env({'HF_HOME': '/cache/huggingface', 'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false', 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': '1'})
         .add_local_file(ROOT / 'modal_control/batch_inputs/video_00.wav', '/input/video_00.wav')
         .add_local_file(ROOT / 'modal_control/batch_inputs/video_01.wav', '/input/video_01.wav')
         .add_local_file(ROOT / 'modal_control/batch_inputs/video_02.wav', '/input/video_02.wav')
         .add_local_file(ROOT / 'modal_control/batch_inputs/video_03.wav', '/input/video_03.wav')
         .add_local_file(ROOT / 'modal_control/batch_inputs/video_04.wav', '/input/video_04.wav')
         .add_local_file(ROOT / 'modal_control/smoke_nemo_helpers.py', '/work/smoke_nemo_helpers.py')
         .add_local_file(ROOT / 'nemo_audio.py', '/work/nemo_audio.py')
         .add_local_file(ROOT / 'local_subtitles.py', '/work/local_subtitles.py')
         .add_local_file(ROOT / 'batch_dubbing/speech_units.py', '/work/batch_dubbing/speech_units.py')
         .add_local_file(ROOT / 'batch_dubbing/speech_stage.py', '/work/batch_dubbing/speech_stage.py')
         .add_local_file(ROOT / 'modal_control/speaker_timeline.py', '/work/speaker_timeline.py')
         .add_local_file(ROOT / 'modal_control/speaker_voices.py', '/work/speaker_voices.py')
         .add_local_file(ROOT / 'speech_model_benchmark_2026-10-27/code/telemetry.py', '/work/telemetry.py'))
volume = modal.Volume.from_name('speech-model-benchmark-20261027')
app = modal.App('new-guys-speech-full')


@app.function(image=image, gpu='T4', cpu=4, memory=32768, timeout=7200,
              max_containers=1, retries=0, volumes={'/cache': volume})
def process(seconds, video_index):
    import sys
    sys.path.insert(0, '/work')
    from batch_dubbing.speech_stage import process_audio
    try:
        return process_audio(f'/input/video_{video_index:02d}.wav', seconds, commit_artifacts=volume.commit)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(video_index: int, smoke: bool = False):
    import wave
    if video_index not in range(5):
        raise ValueError('video index out of range')
    source = ROOT / f'modal_control/batch_inputs/video_{video_index:02d}.wav'
    with wave.open(str(source), 'rb') as audio:
        duration = audio.getnframes() / audio.getframerate()
    destination = ROOT / 'modal_control/batch_inputs' / f'video_{video_index:02d}.gpu_{"preflight" if smoke else "full"}.json'
    if destination.exists():
        raise FileExistsError(f'Output already exists: {destination}')
    result = process.remote(min(15, duration) if smoke else duration, video_index)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print('RESULT', destination, flush=True)
    if result.get('status') != 'complete':
        raise RuntimeError(f"Speech stage blocked; review preserved diagnostics: {destination}")
