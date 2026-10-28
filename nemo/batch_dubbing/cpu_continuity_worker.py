"""Supertonic CPU stage for a controller-released speaker-aware smoke test."""
import io
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('ffmpeg', 'libsndfile1')
         .pip_install('supertonic==1.3.1', 'huggingface_hub', 'soundfile', 'numpy<2', 'psutil')
         .env({'HF_HOME': '/cache/huggingface', 'OMP_NUM_THREADS': '4'})
         .add_local_file(ROOT / 'speech_model_benchmark_2026-10-27/code/telemetry.py', '/work/telemetry.py')
         .add_local_file(ROOT / 'modal_control/dub_segments.py', '/work/dub_segments.py')
         .add_local_file(ROOT / 'batch_dubbing/voice_fallback.py', '/work/voice_fallback.py'))
volume = modal.Volume.from_name('speech-model-benchmark-20261027')
app = modal.App('new-guys-dub-full-cpu')


@app.function(image=image, cpu=4, memory=16384, timeout=7200, max_containers=1,
              retries=0, volumes={'/cache': volume})
def synthesize(data, seconds):
    import sys
    import tempfile
    import subprocess
    import time
    import numpy as np
    import soundfile as sf
    from huggingface_hub import snapshot_download
    from supertonic import TTS
    sys.path.insert(0, '/work')
    from telemetry import emit, start
    from dub_segments import merge_speech_cues
    from voice_fallback import resolve_borderline_voices
    assigned = {cue.get('speaker') for cue in data['translations']['de'] if cue.get('speaker')}
    # A diarization fragment with no assigned German cues has no usable voice
    # evidence. If only one speaker owns all attributed cues, use that one as
    # the continuity estimate for unattributed cues and record the assumption.
    if len(assigned) == 1:
        sole = next(iter(assigned))
        data['voices'] = {sole: data['voices'][sole]}
    data['voices'] = resolve_borderline_voices(data['voices'])
    emit('worker_ready', hardware='CPU', seconds=seconds)
    monitor = start()
    begin = time.monotonic()
    try:
        cues = [dict(c) for c in data['translations']['de'] if float(c['end']) <= seconds]
        inferred = 0
        # If diarization found exactly one speaker, an isolated coverage gap
        # can use that speaker's preset. Keep the inference explicit in results.
        if len(data['voices']) == 1:
            sole_speaker = next(iter(data['voices']))
            for cue in cues:
                if cue.get('speaker') is None:
                    cue['speaker'] = sole_speaker
                    inferred += 1
        # For multiple speakers, use overlaps with the diarized English cues
        # when a translated cue has no direct diarization assignment. This is
        # an estimate, not proof of speaker identity; record every inference.
        if len(data['voices']) > 1:
            for cue in cues:
                if cue.get('speaker') is not None:
                    continue
                scores = {}
                for word_cue in data['english']:
                    speaker = word_cue.get('speaker')
                    overlap = max(0.0, min(cue['end'], word_cue['end']) - max(cue['start'], word_cue['start']))
                    if speaker in data['voices'] and overlap:
                        scores[speaker] = scores.get(speaker, 0.0) + overlap
                if not scores:
                    raise ValueError('No source speaker evidence for translated cue')
                cue['speaker'] = max(scores, key=scores.get)
                inferred += 1
        segments = merge_speech_cues(cues)
        if not segments:
            raise ValueError('No German cues in the smoke range')
        voices = data['voices']
        for cue in segments:
            speaker = cue.get('speaker')
            if speaker is None or not voices.get(speaker, {}).get('voice'):
                raise ValueError('Speaker or voice uncertain; review required before synthesis')
        model_dir = snapshot_download('Supertone/supertonic-3', revision='3cadd1ee6394adea1bd021217a0e650ede09a323')
        model = TTS(model='supertonic-3', model_dir=model_dir, auto_download=False,
                    intra_op_num_threads=4, inter_op_num_threads=1)
        sr = model.sample_rate
        timeline = np.zeros(int(seconds * sr), dtype=np.float32)
        rows = []
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            for i, cue in enumerate(segments):
                start, end = float(cue['start']), float(cue['end'])
                if i + 1 < len(segments):
                    end = float(segments[i + 1]['start'])
                else:
                    end = float(seconds)
                if start < 0 or end <= start:
                    raise ValueError('Overlapping or invalid cue timeline')
                voice = voices[cue['speaker']]['voice']
                text = cue['text'].translate(str.maketrans({'„': '"', '“': '"', '”': '"', '–': '-', '—': '-', '…': '...'}))
                audio, _ = model.synthesize(text, voice_style=model.get_voice_style(voice), lang='de')
                audio = np.asarray(audio, dtype=np.float32).squeeze()
                if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
                    raise ValueError('Invalid generated audio')
                source = temp / 'source.wav'
                fit = temp / 'fit.wav'
                sf.write(source, audio, sr)
                rate = max(1.0, len(audio) / sr / (end - start))
                # Each atempo filter supports at most 2x. Chain filters rather
                # than dropping words when a translated span is unusually long.
                remaining = rate
                factors = []
                while remaining > 2:
                    factors.append('atempo=2')
                    remaining /= 2
                factors.append(f'atempo={remaining:.8f}')
                subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(source), '-af', ','.join(factors), str(fit)], check=True)
                fitted, actual_sr = sf.read(fit, dtype='float32')
                if actual_sr != sr or len(fitted) > int((end - start) * sr) + sr // 20:
                    raise ValueError('Fitted audio exceeds its cue window')
                position = int(start * sr)
                n = min(len(fitted), int((end - start) * sr), len(timeline) - position)
                timeline[position:position+n] += fitted[:n]
                rows.append({'index': i, 'speaker': cue['speaker'], 'voice': voice, 'rate': rate})
                emit('segment_complete', index=i+1, total=len(segments))
        peak = float(np.max(np.abs(timeline)))
        if peak < 1e-5:
            raise ValueError('Dubbed audio is silent')
        if peak > 0.99:
            timeline *= 0.99 / peak
        with tempfile.TemporaryDirectory() as tmp2:
            raw = Path(tmp2) / 'timeline.wav'
            encoded = Path(tmp2) / 'timeline.m4a'
            sf.write(raw, timeline, sr, subtype='PCM_16')
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(raw), '-c:a', 'aac', '-b:a', '128k', str(encoded)], check=True)
            audio_bytes = encoded.read_bytes()
        return {'audio': audio_bytes, 'report': {'status': 'complete', 'seconds': seconds,
                'synthesis_wall_seconds': time.monotonic() - begin, 'segments': rows, 'sample_rate': sr,
                'source_or_single_speaker_gap_inferences': inferred, 'voice_assignments': data['voices']}}
    finally:
        monitor.set()


@app.local_entrypoint()
def main(video_index: int, smoke: bool = False):
    if video_index not in range(5):
        raise ValueError('video index out of range')
    source = ROOT / f'modal_control/batch_inputs/video_{video_index:02d}.gpu_full.json'
    data = json.loads(source.read_text())
    pitch_file = ROOT / f'modal_control/batch_inputs/video_{video_index:02d}.pitch_review_v2.json'
    data['voices'] = json.loads(pitch_file.read_text())['voices']
    duration = min(15, data['duration_seconds']) if smoke else data['duration_seconds']
    name = f'video_{video_index:02d}.cpu_v2_{"preflight" if smoke else "full"}'
    audio = ROOT / f'modal_control/batch_inputs/{name}.m4a'
    report = ROOT / f'modal_control/batch_inputs/{name}.json'
    if audio.exists() or report.exists():
        raise FileExistsError(f'Output already exists for {name}')
    result = synthesize.remote(data, duration)
    audio.write_bytes(result['audio'])
    report.write_text(json.dumps(result['report'], indent=2) + '\n')
