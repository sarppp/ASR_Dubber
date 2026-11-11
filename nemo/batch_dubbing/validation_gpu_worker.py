"""Prepared three-case source-timing evaluation. Execution requires release."""
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'dubbing_naturalness_2026-10-28/source_validation_inputs.json'
OUTPUT = ROOT / 'dubbing_naturalness_2026-10-28/source_validation_2026-10-29'
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('ffmpeg', 'git', 'build-essential', 'libgomp1', 'libsndfile1')
         .pip_install('torch==2.6.0', 'torchaudio==2.6.0', 'torchvision==0.21.0')
         .pip_install('nemo_toolkit[asr]==2.5.3', 'numpy<2', 'pyarrow<22', 'transformers<5', 'librosa', 'soundfile', 'psutil', 'huggingface_hub')
         .env({'HF_HOME': '/cache/huggingface', 'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false', 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': '1'})
         .add_local_file(CONFIG, '/work/source_validation_inputs.json')
         .add_local_file(ROOT / 'modal_control/batch_inputs/video_00.wav', '/input/video_00.wav')
         .add_local_file(ROOT / 'modal_control/batch_inputs/video_01.wav', '/input/video_01.wav')
         .add_local_file(ROOT / 'modal_control/batch_inputs/video_04.wav', '/input/video_04.wav')
         .add_local_file(ROOT / 'modal_control/smoke_nemo_helpers.py', '/work/smoke_nemo_helpers.py')
         .add_local_file(ROOT / 'modal_control/speaker_voices.py', '/work/speaker_voices.py')
         .add_local_file(ROOT / 'modal_control/speaker_timeline.py', '/work/speaker_timeline.py')
         .add_local_file(ROOT / 'nemo_audio.py', '/work/nemo_audio.py')
         .add_local_file(ROOT / 'local_subtitles.py', '/work/local_subtitles.py')
         .add_local_file(ROOT / 'batch_dubbing/speech_units.py', '/work/batch_dubbing/speech_units.py')
         .add_local_file(ROOT / 'batch_dubbing/speech_stage.py', '/work/batch_dubbing/speech_stage.py')
         .add_local_file(ROOT / 'batch_dubbing/validation_config.py', '/work/batch_dubbing/validation_config.py')
         .add_local_file(ROOT / 'speech_model_benchmark_2026-10-27/code/telemetry.py', '/work/telemetry.py'))
volume = modal.Volume.from_name('speech-model-benchmark-20261027')
app = modal.App('source-validation-gpu')


def run_cases(config, smoke, *, processor, verifier, commit, cleanup=lambda: None, emit=None, input_root=Path('/input')):
    """Validate every input before the first callback, then process serially."""
    from batch_dubbing.validation_config import validate_config, selected_cases
    config = validate_config(config)
    for case in config['cases']:
        verifier(case, input_root / Path(case['input_file']).name)
    results = []
    for case in selected_cases(config, smoke=smoke):
        if emit:
            emit('case_started', id=case['id'], stage='speech')
        try:
            data = processor(input_root / Path(case['input_file']).name,
                             case['duration_seconds'], offset_seconds=case['offset_seconds'],
                             target_languages=(config['target_language'],),
                             max_source_units=config['max_units_per_case'], commit_artifacts=commit)
        except Exception as exc:
            data = {'status': 'failed', 'error': str(exc), 'failure_stage': 'speech',
                    'source_offset_seconds': case['offset_seconds'],
                    'duration_seconds': case['duration_seconds']}
        finally:
            cleanup()
        results.append({'id': case['id'], 'data': data})
        if emit:
            emit('case_complete', id=case['id'], status=data.get('status'))
        commit()
    return {'status': 'complete' if all(row['data'].get('status') == 'complete' for row in results) else 'blocked',
            'stage': 'gpu_smoke' if smoke else 'gpu_full', 'config': config, 'cases': results}


@app.function(image=image, gpu='T4', cpu=4, memory=32768, timeout=600,
              max_containers=1, retries=0, volumes={'/cache': volume})
def process(config, smoke=False):
    import sys
    sys.path.insert(0, '/work')
    from telemetry import emit
    emit('worker_ready', hardware='T4', evaluation='source_validation', smoke=smoke)
    from batch_dubbing.speech_stage import process_audio
    from batch_dubbing.validation_config import verify_input
    def cleanup():
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    try:
        return run_cases(config, smoke, processor=process_audio, verifier=verify_input,
                         commit=volume.commit, cleanup=cleanup, emit=emit)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(smoke: bool = False):
    from batch_dubbing.validation_config import load_config
    config = load_config(CONFIG, root=ROOT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / ('gpu_smoke.json' if smoke else 'gpu_full.json')
    # Reserve exclusively before any paid job, retaining failure diagnostics.
    with destination.open('x') as handle:
        handle.write(json.dumps({'status': 'running', 'config': config}) + '\n')
    try:
        result = process.remote(config, smoke)
    except Exception as exc:
        destination.write_text(json.dumps({'status': 'failed', 'error': str(exc), 'config': config}, indent=2) + '\n')
        raise
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('RESULT', destination, flush=True)
    if result['status'] != 'complete':
        raise RuntimeError(f'Speech validation blocked; diagnostics saved to {destination}')
