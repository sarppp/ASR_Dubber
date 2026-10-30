"""Qwen TTS adapter for the ten-minute phrase experiment; inference is Modal-only."""
import json
from pathlib import Path
import time
import uuid

from batch_dubbing.cpu_natural import select_units
from batch_dubbing.natural_pipeline import run_pipeline
from batch_dubbing.natural_render import render_timeline, _child_limits

MODEL = 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'
REVISION = '0c0e3051f131929182e2c023b9537f8b1c68adfe'
MAX_UNITS = 400


def voice_assignments(data, units):
    result = {}
    for index, speaker in enumerate(sorted({u['speaker'] for u in units})):
        source_voice = data.get('voices', {}).get(speaker, {}).get('voice', '')
        preset = 'Serena' if source_voice.startswith('F') else 'Ryan' if source_voice.startswith('M') else ('Serena', 'Ryan')[index % 2]
        result[speaker] = {'voice': preset, 'mode': 'stock_preset',
                           'basis': 'pitch heuristic' if source_voice.startswith(('F', 'M')) else 'stable fallback',
                           'identity_cloning': False}
    return result


def synthesize_qwen(data, seconds, *, emit, commit):
    units, scope = select_units(data, seconds, 'de')
    if seconds > 600 or len(units) > MAX_UNITS:
        raise ValueError('Only the reviewed ten-minute Qwen test is allowed')
    import numpy as np
    import soundfile as sf
    import subprocess
    import torch
    from scipy.signal import resample_poly
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel

    started = time.monotonic()
    destination = Path('/cache/crispr-10min-qwen') / uuid.uuid4().hex
    destination.mkdir(parents=True)
    voices = voice_assignments(data, units)
    report, stage, model = {}, 'load', None
    native, load_seconds = [], None

    def persist():
        report.update(model=MODEL, revision=REVISION, runtime='qwen-tts==0.1.1',
                      dtype='float32', attention='sdpa', seconds=seconds, scope=scope,
                      native_artifact_directory=str(destination), native_takes=native,
                      voice_assignments=voices, load_seconds=load_seconds,
                      worker_wall_seconds=time.monotonic()-started,
                      speech_text_basis='Adjacent German phrase translations joined by source utterance',
                      timing_conditioning='Brisk delivery with source-duration instruction; measured fit afterward, maximum 1.15x tempo',
                      semantic_quality_verified=False, listening_review_complete=False)
        (destination / 'worker-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        commit()

    try:
        torch.set_num_threads(4)
        torch.manual_seed(42)
        local = snapshot_download(MODEL, revision=REVISION)
        model = Qwen3TTSModel.from_pretrained(local, device_map='cuda:0', dtype=torch.float32,
                                             attn_implementation='sdpa')
        load_seconds = time.monotonic()-started
        emit('model_loaded', seconds=load_seconds, model=MODEL)

        def generate(unit, text):
            before = time.monotonic()
            voice = voices[unit['speaker']]['voice']
            target_seconds = unit['source_end'] - unit['source_start']
            waves, sr = model.generate_custom_voice(
                text=text, language='German', speaker=voice,
                instruct=('Speak clearly in German at a brisk, fluent interview pace. '
                          f'Aim to finish this text in about {target_seconds:.1f} seconds. '
                          'Preserve every word. Avoid dramatic pauses and drawn-out syllables.'),
                max_new_tokens=1440)
            torch.cuda.synchronize()
            audio = waves[0].detach().cpu().numpy() if torch.is_tensor(waves[0]) else np.asarray(waves[0])
            audio = np.asarray(audio, dtype=np.float32).squeeze()
            if (sr != 24000 or audio.ndim != 1 or not audio.size or
                    not np.isfinite(audio).all() or len(audio)/sr >= 119 or
                    float(np.sqrt(np.mean(audio.astype(float)**2))) < 1e-5):
                raise ValueError('Invalid, silent or generation-limit Qwen output')
            path = destination / f'native_{len(native):05d}.wav'
            sf.write(path, audio, sr, subtype='FLOAT')
            native.append({'id': unit['id'], 'text': text, 'voice': voice,
                           'file': str(path), 'sample_rate': sr, 'seconds': len(audio)/sr,
                           'synthesis_seconds': time.monotonic()-before})
            persist()
            emit('segment_complete', id=unit['id'], completed=len(native), total=len(units),
                 synthesis_seconds=native[-1]['synthesis_seconds'])
            # The existing timing/render utilities expect 44.1 kHz. Preserve native takes too.
            return {'samples': resample_poly(audio, 147, 80), 'sample_rate': 44100, 'voice': voice}

        stage = 'synthesis'
        report = run_pipeline(units, duration_seconds=seconds, output_dir=destination / 'fit',
                              synthesize=generate, max_synthesis_calls=len(units))
        report['synthesis_wall_seconds'] = time.monotonic()-started-load_seconds
        del model
        model = None
        torch.cuda.empty_cache()
        if report['status'] != 'ready':
            persist()
            return {'audio': None, 'report': report}
        stage = 'render'
        timeline = destination / 'timeline.wav'
        report['rendered'] = render_timeline(report['render'], timeline, max_wall_seconds=300)
        stage = 'encode'
        media = destination / 'german-dub.m4a'
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-threads', '1', '-i', str(timeline),
                        '-threads', '1', '-c:a', 'aac', '-b:a', '128k', str(media)],
                       check=True, timeout=90, preexec_fn=_child_limits)
        report['status'] = 'complete'
        persist()
        return {'audio': media.read_bytes(), 'report': report}
    except Exception as exc:
        fitted = destination / 'fit/report.json'
        if fitted.exists():
            report = json.loads(fitted.read_text())
        report.update(status='failed', failure_stage=stage, error=str(exc))
        persist()
        return {'audio': None, 'report': report}
