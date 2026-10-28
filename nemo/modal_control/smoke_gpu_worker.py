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
         .add_local_file(ROOT / 'modal_control/speaker_timeline.py', '/work/speaker_timeline.py')
         .add_local_file(ROOT / 'modal_control/speaker_voices.py', '/work/speaker_voices.py')
         .add_local_file(ROOT / 'speech_model_benchmark_2026-10-27/code/telemetry.py', '/work/telemetry.py'))
volume = modal.Volume.from_name('speech-model-benchmark-20261027')
app = modal.App('new-guys-speech-smoke')


@app.function(image=image, gpu='T4', cpu=4, memory=32768, timeout=1800,
              max_containers=1, retries=0, volumes={'/cache': volume})
def process(seconds):
    import gc
    import subprocess
    import sys
    import tempfile
    import time
    import torch
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf
    sys.path.insert(0, '/work')
    from telemetry import emit, start
    import smoke_nemo_helpers as nemo
    from local_subtitles import chunk_boundaries, infer_chunk
    from speaker_timeline import assign_cue_speakers
    from speaker_voices import assign_voices
    import nemo.collections.asr as nasr

    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    if 'T4' not in gpu:
        raise RuntimeError(f'T4 required, got {gpu}')
    emit('worker_ready', gpu=gpu, seconds=seconds)
    monitor = start()
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            wav = work / 'input.wav'
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', '/input/smoke.wav', '-t', str(seconds), str(wav)], check=True)
            nemo._ensure_remote_imports()
            stage = time.monotonic()
            turns = nemo._run_diarization(str(wav), work)
            if not turns:
                raise RuntimeError('Diarization returned no speakers')
            # Estimate perceived voice presentation from voiced F0. Ambiguous
            # pitch remains unassigned; this is not a gender identification.
            import librosa
            import numpy as np
            import soundfile as sf
            audio, sample_rate = sf.read(str(wav), dtype='float32')
            pitch_by_speaker = {}
            for speaker in sorted({t['speaker'] for t in turns}):
                estimates = []
                remaining = 12.0
                for turn in (t for t in turns if t['speaker'] == speaker):
                    duration = min(remaining, turn['end'] - turn['start'])
                    if duration < 0.5:
                        continue
                    samples = audio[int(turn['start'] * sample_rate):int((turn['start'] + duration) * sample_rate)]
                    f0, voiced, probability = librosa.pyin(samples, fmin=50, fmax=400, sr=sample_rate)
                    estimates.extend(f0[voiced & (probability >= 0.7) & np.isfinite(f0)].tolist())
                    remaining -= duration
                    if remaining <= 0:
                        break
                pitch_by_speaker[speaker] = estimates
            voices = assign_voices(pitch_by_speaker)
            timings = {'diarization_seconds': time.monotonic() - stage}
            emit('stage_complete', stage='diarization', speakers=len({t['speaker'] for t in turns}))
            repo = 'nvidia/parakeet-tdt-0.6b-v3'
            local = snapshot_download(repo, revision='541d1f99c6b0c3cd0b11a95167540bb8edefd82b', allow_patterns=['*.nemo'])
            stage = time.monotonic()
            model = nasr.models.ASRModel.restore_from(str(next(Path(local).glob('*.nemo'))), map_location='cpu').half().cuda().eval()
            for key in ('greedy.use_cuda_graph_decoder', 'greedy.loop_labels', 'use_cuda_graphs'):
                OmegaConf.update(model.cfg.decoding, key, False, force_add=True)
            model.change_decoding_strategy(model.cfg.decoding)
            bounds = chunk_boundaries(wav)
            chunks = []
            for i, (a, b) in enumerate(bounds):
                p = work / f'chunk_{i}.wav'
                subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-ss', str(a), '-i', str(wav), '-t', str(b-a), str(p)], check=True)
                chunks.append((p, a, b))
            words, _, _ = nemo._transcribe_manifest(model, [{'path': str(p), 'offset': a} for p, a, b in chunks], repo, 'en', 'en')
            if not words:
                raise RuntimeError('Parakeet returned no word timestamps')
            english = nemo._words_to_segments_diarized(nemo._assign_speakers(words, turns))
            timings['transcription_seconds'] = time.monotonic() - stage
            emit('stage_complete', stage='transcription', cues=len(english))
            del model
            gc.collect()
            torch.cuda.empty_cache()
            repo = 'nvidia/canary-1b-v2'
            local = snapshot_download(repo, revision='d455706339a6b32e1aa40f82c713a482a0c938e2', allow_patterns=['*.nemo'])
            stage = time.monotonic()
            model = nasr.models.ASRModel.restore_from(str(next(Path(local).glob('*.nemo'))), map_location='cpu').half().cuda().eval()
            if model.timestamps_asr_model is None:
                raise RuntimeError('Canary timestamp aligner missing')
            model.timestamps_asr_model.half().cuda().eval()
            translations = {}
            for lang in ('de', 'fr'):
                rows = []
                for p, a, b in chunks:
                    rows.extend(infer_chunk(model, p, 'en', lang, a, b))
                if not rows:
                    raise RuntimeError(f'No {lang} translation returned')
                translations[lang] = assign_cue_speakers(rows, turns)
                emit('stage_complete', stage='translation', language=lang, cues=len(rows))
            timings['translation_seconds'] = time.monotonic() - stage
            timings['worker_seconds'] = time.monotonic() - started
            return {'status': 'complete', 'duration_seconds': seconds, 'turns': turns, 'voices': voices, 'english': english,
                    'translations': translations, 'timings': timings, 'gpu': gpu,
                    'notice': 'Speaker voice assignment and dubbing are separate CPU work.'}
    finally:
        monitor.set()


@app.local_entrypoint()
def main(smoke: bool = False):
    destination = ROOT / 'modal_control/batch_inputs' / ('gpu_preflight.json' if smoke else 'gpu_smoke.json')
    result = process.remote(15 if smoke else 60)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print('RESULT', destination, flush=True)
