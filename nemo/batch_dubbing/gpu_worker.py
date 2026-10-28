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
         .add_local_file(ROOT / 'modal_control/speaker_timeline.py', '/work/speaker_timeline.py')
         .add_local_file(ROOT / 'modal_control/speaker_voices.py', '/work/speaker_voices.py')
         .add_local_file(ROOT / 'speech_model_benchmark_2026-10-27/code/telemetry.py', '/work/telemetry.py'))
volume = modal.Volume.from_name('speech-model-benchmark-20261027')
app = modal.App('new-guys-speech-full')


@app.function(image=image, gpu='T4', cpu=4, memory=32768, timeout=7200,
              max_containers=1, retries=0, volumes={'/cache': volume})
def process(seconds, video_index):
    import gc
    import hashlib
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
    from batch_dubbing.speech_units import build_speech_units, bind_caption_translations, reconcile_speaker_corridors
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
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', f'/input/video_{video_index:02d}.wav', '-t', str(seconds), str(wav)], check=True)
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
            source_words = nemo._assign_speakers([dict(word) for word in words], turns)
            source_digest = hashlib.sha256()
            with wav.open('rb') as source_audio:
                for block in iter(lambda: source_audio.read(1024 * 1024), b''):
                    source_digest.update(block)
            source_id = 'sha256:' + source_digest.hexdigest()
            alignment = reconcile_speaker_corridors(source_words, turns)
            # Retain costly ASR evidence even if alignment or later translation fails.
            import uuid
            alignment_path = Path('/cache/source-alignment') / (uuid.uuid4().hex + '.json')
            alignment_path.parent.mkdir(parents=True, exist_ok=True)
            alignment_bytes = (json.dumps({'source_id': source_id, 'source_language': 'en',
                'source_words': source_words, 'raw_turns': turns, 'alignment': alignment},
                ensure_ascii=False, indent=2) + '\n').encode('utf-8')
            if len(alignment_bytes) > 64 * 1024**2:
                raise ValueError('Alignment diagnostic exceeds the 64 MiB artifact bound')
            alignment_sha256 = hashlib.sha256(alignment_bytes).hexdigest()
            with alignment_path.open('xb') as alignment_file:
                alignment_file.write(alignment_bytes)
            del alignment_bytes
            volume.commit()
            if alignment['status'] != 'ready':
                emit('alignment_blocked', conflicts=alignment['conflicts'], artifact=str(alignment_path))
                return {'status': 'needs_alignment', 'duration_seconds': seconds, 'turns': turns,
                        'raw_turns': turns, 'reconciled_turns': [], 'alignment': alignment,
                        'alignment_artifact': str(alignment_path), 'source_words': source_words,
                        'source_id': source_id, 'source_language': 'en', 'voices': voices,
                        'alignment_sha256': alignment_sha256, 'speech_units': {}, 'translations': {},
                        'timings': timings, 'gpu': gpu}
            reconciled_turns = alignment['turns']
            source_units = build_speech_units(source_words, reconciled_turns, source_id=source_id)
            if not source_units:
                raise RuntimeError('No source speech units could be constructed')
            english = nemo._words_to_segments_diarized(source_words)
            timings['transcription_seconds'] = time.monotonic() - stage
            emit('stage_complete', stage='transcription', cues=len(english))
            del model
            gc.collect()
            torch.cuda.empty_cache()
            utterance_audio = {}
            for index, unit in enumerate(source_units):
                if unit.source_end - unit.source_start > 30:
                    raise RuntimeError(f'Source unit {unit.id} exceeds the bounded translation duration')
                path = work / f'utterance_{index}.wav'
                subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1',
                                '-ss', str(unit.source_start), '-i', str(wav),
                                '-t', str(unit.source_end - unit.source_start),
                                '-threads', '1', str(path)], check=True)
                utterance_audio[unit.id] = path
            repo = 'nvidia/canary-1b-v2'
            local = snapshot_download(repo, revision='d455706339a6b32e1aa40f82c713a482a0c938e2', allow_patterns=['*.nemo'])
            stage = time.monotonic()
            model = nasr.models.ASRModel.restore_from(str(next(Path(local).glob('*.nemo'))), map_location='cpu').half().cuda().eval()
            if model.timestamps_asr_model is None:
                raise RuntimeError('Canary timestamp aligner missing')
            model.timestamps_asr_model.half().cuda().eval()
            translations, speech_units = {}, {}
            for lang in ('de', 'fr'):
                captions_by_id = {}
                for index, unit in enumerate(source_units):
                    captions_by_id[unit.id] = infer_chunk(model, utterance_audio[unit.id], 'en', lang,
                                                          unit.source_start, unit.source_end)
                    emit('unit_translated', source_id=unit.id, language=lang,
                         index=index + 1, total=len(source_units))
                bound_units, rows = bind_caption_translations(source_units, captions_by_id)
                translations[lang] = rows
                speech_units[lang] = [unit.to_dict() for unit in bound_units]
                emit('stage_complete', stage='translation', language=lang, cues=len(rows))
            timings['translation_seconds'] = time.monotonic() - stage
            timings['worker_seconds'] = time.monotonic() - started
            return {'status': 'complete', 'duration_seconds': seconds, 'turns': turns, 'voices': voices, 'english': english,
                    'translations': translations, 'speech_units': speech_units,
                    'source_words': source_words, 'source_id': source_id, 'timings': timings, 'gpu': gpu,
                    'raw_turns': turns, 'reconciled_turns': reconciled_turns, 'alignment': alignment,
                    'alignment_artifact': str(alignment_path), 'alignment_sha256': alignment_sha256,
                    'source_language': 'en',
                    'notice': 'Speaker voice assignment and dubbing are separate CPU work.'}
    finally:
        monitor.set()


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
