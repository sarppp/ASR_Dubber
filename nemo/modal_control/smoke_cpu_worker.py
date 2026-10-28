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
         .add_local_file(ROOT / 'modal_control/dub_segments.py', '/work/dub_segments.py'))
volume = modal.Volume.from_name('speech-model-benchmark-20261027')
app = modal.App('new-guys-dub-smoke-cpu')


@app.function(image=image, cpu=4, memory=16384, timeout=1800, max_containers=1,
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
    emit('worker_ready', hardware='CPU', seconds=seconds)
    monitor = start()
    begin = time.monotonic()
    try:
        segments = merge_speech_cues([c for c in data['translations']['de'] if float(c['end']) <= seconds])
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
                    end = min(end, float(segments[i + 1]['start']))
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
                if rate > 2.0:
                    raise ValueError(f'Speech exceeds available time by more than 2x in cue {i}')
                subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(source), '-af', f'atempo={rate:.8f}', str(fit)], check=True)
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
        buf = io.BytesIO()
        sf.write(buf, timeline, sr, format='WAV', subtype='PCM_16')
        return {'wav': buf.getvalue(), 'report': {'status': 'complete', 'seconds': seconds,
                'synthesis_wall_seconds': time.monotonic() - begin, 'segments': rows, 'sample_rate': sr}}
    finally:
        monitor.set()


@app.local_entrypoint()
def main(smoke: bool = False):
    source = ROOT / 'modal_control/batch_inputs/gpu_smoke.json'
    data = json.loads(source.read_text())
    result = synthesize.remote(data, 15 if smoke else 60)
    name = 'cpu_v2_preflight' if smoke else 'cpu_v2_smoke'
    if (ROOT / f'modal_control/batch_inputs/{name}.wav').exists() or (ROOT / f'modal_control/batch_inputs/{name}.json').exists():
        raise FileExistsError(f'Output already exists for {name}')
    (ROOT / f'modal_control/batch_inputs/{name}.wav').write_bytes(result['wav'])
    (ROOT / f'modal_control/batch_inputs/{name}.json').write_text(json.dumps(result['report'], indent=2) + '\n')
