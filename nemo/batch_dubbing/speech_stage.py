"""Shared released T4 speech stage; importing this module performs no inference."""
import json
import math
from pathlib import Path
import subprocess
import wave

MAX_SECONDS = 7200
CROP_TIMEOUT_SECONDS = 120
SUPPORTED_TARGET_LANGUAGES = frozenset({'de', 'fr'})


def _finite_number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if value < 0 or (positive and value == 0):
        raise ValueError(f'Invalid {name}')
    return float(value)


def validate_request(source_path, seconds, *, offset_seconds=0, target_languages=('de', 'fr')):
    """Validate WAV headers and scope before model imports or child processes."""
    seconds = _finite_number(seconds, 'seconds', positive=True)
    offset_seconds = _finite_number(offset_seconds, 'offset_seconds')
    if seconds > MAX_SECONDS:
        raise ValueError('Speech stage duration exceeds 7200-second bound')
    if (not isinstance(target_languages, (tuple, list)) or not target_languages
            or any(not isinstance(lang, str) or lang not in SUPPORTED_TARGET_LANGUAGES
                   for lang in target_languages)
            or len(set(target_languages)) != len(target_languages)):
        raise ValueError('Target languages must be unique supported codes: de, fr')
    source_path = Path(source_path)
    if not source_path.is_file():
        raise ValueError('Source WAV does not exist')
    with wave.open(str(source_path), 'rb') as source:
        duration = source.getnframes() / source.getframerate()
        tolerance = 1 / source.getframerate()
    if offset_seconds + seconds > duration + tolerance:
        raise ValueError('Requested range exceeds source WAV duration')
    return source_path, seconds, offset_seconds, tuple(target_languages)


def _crop_audio(source, destination, offset, seconds):
    """One bounded, single-threaded crop; no overwrites or parallel subprocesses."""
    offset = _finite_number(offset, 'crop offset')
    seconds = _finite_number(seconds, 'crop seconds', positive=True)
    if seconds > MAX_SECONDS:
        raise ValueError('Crop exceeds duration bound')
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-threads', '1',
                    '-ss', str(offset), '-i', str(source), '-t', str(seconds),
                    '-threads', '1', str(destination)],
                   check=True, timeout=CROP_TIMEOUT_SECONDS)


def _check_source_unit_count(units, limit):
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10000:
        raise ValueError('max_source_units must be an integer in [1, 10000]')
    if len(units) > limit:
        raise ValueError(f'Source utterance count {len(units)} exceeds translation call bound {limit}')


def process_audio(source_path, seconds, *, offset_seconds=0, target_languages=('de', 'fr'),
                  commit_artifacts=None, max_source_units=10000, transcribe_only=False):
    """Process one source WAV range serially; returned timestamps are crop-relative.

    Model imports and inference occur only when called in a released T4 worker.
    ``commit_artifacts`` optionally persists a mounted cache volume at the ASR
    checkpoint; workers also commit on exit. No Modal app is constructed here.
    """
    if type(transcribe_only) is not bool:
        raise ValueError('transcribe_only must be Boolean')
    _check_source_unit_count([], max_source_units)
    source_path, seconds, offset_seconds, target_languages = validate_request(
        source_path, seconds, offset_seconds=offset_seconds, target_languages=target_languages)
    if commit_artifacts is not None and not callable(commit_artifacts):
        raise ValueError('commit_artifacts must be callable')
    import gc
    import hashlib
    import subprocess
    import sys
    import tempfile
    import time
    import torch
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf
    from telemetry import emit, start
    import smoke_nemo_helpers as nemo
    from local_subtitles import chunk_boundaries, infer_chunk
    from batch_dubbing.speech_units import build_speech_units, bind_caption_translations, reconcile_speaker_corridors
    from speaker_voices import assign_voices
    import nemo.collections.asr as nasr

    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True, timeout=15).strip()
    if 'T4' not in gpu:
        raise RuntimeError(f'T4 required, got {gpu}')
    emit('worker_ready', gpu=gpu, seconds=seconds)
    monitor = start()
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            wav = work / 'input.wav'
            _crop_audio(source_path, wav, offset_seconds, seconds)
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
                _crop_audio(wav, p, a, b - a)
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
            alignment_bytes = (json.dumps({'source_id': source_id, 'source_language': 'en', 'source_offset_seconds': offset_seconds,
                'source_words': source_words, 'raw_turns': turns, 'alignment': alignment},
                ensure_ascii=False, indent=2) + '\n').encode('utf-8')
            if len(alignment_bytes) > 64 * 1024**2:
                raise ValueError('Alignment diagnostic exceeds the 64 MiB artifact bound')
            alignment_sha256 = hashlib.sha256(alignment_bytes).hexdigest()
            with alignment_path.open('xb') as alignment_file:
                alignment_file.write(alignment_bytes)
            del alignment_bytes
            if commit_artifacts is not None:
                commit_artifacts()
            if alignment['status'] != 'ready':
                emit('alignment_blocked', conflicts=alignment['conflicts'], artifact=str(alignment_path))
                return {'status': 'needs_alignment', 'duration_seconds': seconds, 'turns': turns,
                        'raw_turns': turns, 'reconciled_turns': [], 'alignment': alignment,
                        'alignment_artifact': str(alignment_path), 'source_words': source_words,
                        'source_id': source_id, 'source_language': 'en', 'source_offset_seconds': offset_seconds, 'voices': voices,
                        'alignment_sha256': alignment_sha256, 'speech_units': {}, 'translations': {},
                        'timings': timings, 'gpu': gpu}
            reconciled_turns = alignment['turns']
            source_units = build_speech_units(source_words, reconciled_turns, source_id=source_id)
            if not source_units:
                raise RuntimeError('No source speech units could be constructed')
            _check_source_unit_count(source_units, max_source_units)
            english = nemo._words_to_segments_diarized(source_words)
            timings['transcription_seconds'] = time.monotonic() - stage
            emit('stage_complete', stage='transcription', cues=len(english))
            del model
            gc.collect()
            torch.cuda.empty_cache()
            if transcribe_only:
                timings['worker_seconds'] = time.monotonic() - started
                return {'status': 'complete', 'duration_seconds': seconds,
                        'source_words': source_words, 'source_units': [u.to_dict() for u in source_units],
                        'source_id': source_id, 'source_language': 'en', 'voices': voices,
                        'english': english, 'raw_turns': turns, 'turns': reconciled_turns,
                        'alignment': alignment, 'timings': timings, 'gpu': gpu,
                        'translation_performed': False}
            utterance_audio = {}
            for index, unit in enumerate(source_units):
                if unit.source_end - unit.source_start > 30:
                    raise RuntimeError(f'Source unit {unit.id} exceeds the bounded translation duration')
                path = work / f'utterance_{index}.wav'
                _crop_audio(wav, path, unit.source_start, unit.source_end - unit.source_start)
                utterance_audio[unit.id] = path
            repo = 'nvidia/canary-1b-v2'
            local = snapshot_download(repo, revision='d455706339a6b32e1aa40f82c713a482a0c938e2', allow_patterns=['*.nemo'])
            stage = time.monotonic()
            model = nasr.models.ASRModel.restore_from(str(next(Path(local).glob('*.nemo'))), map_location='cpu').half().cuda().eval()
            if model.timestamps_asr_model is None:
                raise RuntimeError('Canary timestamp aligner missing')
            model.timestamps_asr_model.half().cuda().eval()
            translations, speech_units = {}, {}
            for lang in target_languages:
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
                    'source_language': 'en', 'source_offset_seconds': offset_seconds,
                    'notice': 'Speaker voice assignment and dubbing are separate CPU work.'}
    finally:
        monitor.set()
