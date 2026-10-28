"""Supertonic CPU stage for a controller-released speaker-aware smoke test."""
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('ffmpeg', 'libsndfile1')
         .pip_install('supertonic==1.3.1', 'huggingface_hub', 'soundfile', 'numpy<2', 'psutil')
         .env({'HF_HOME': '/cache/huggingface', 'OMP_NUM_THREADS': '4'})
         .add_local_file(ROOT / 'speech_model_benchmark_2026-10-27/code/telemetry.py', '/work/telemetry.py')
         .add_local_file(ROOT / 'batch_dubbing/natural_pipeline.py', '/work/batch_dubbing/natural_pipeline.py')
         .add_local_file(ROOT / 'batch_dubbing/audio_edges.py', '/work/batch_dubbing/audio_edges.py')
         .add_local_file(ROOT / 'batch_dubbing/speech_schedule.py', '/work/batch_dubbing/speech_schedule.py')
         .add_local_file(ROOT / 'batch_dubbing/natural_render.py', '/work/batch_dubbing/natural_render.py')
         .add_local_file(ROOT / 'batch_dubbing/cpu_natural.py', '/work/batch_dubbing/cpu_natural.py'))
volume = modal.Volume.from_name('speech-model-benchmark-20261027')
app = modal.App('new-guys-dub-smoke-cpu')


@app.function(image=image, cpu=4, memory=16384, timeout=1800, max_containers=1,
              retries=0, volumes={'/cache': volume})
def synthesize(data, seconds):
    import sys
    sys.path.insert(0, '/work')
    from telemetry import emit, start
    from batch_dubbing.cpu_natural import synthesize_natural
    emit('worker_ready', hardware='CPU', seconds=seconds)
    monitor = start()
    try:
        voices = dict(data['voices'])
        result = synthesize_natural(data, seconds, voices, output_format='wav', emit=emit)
        result['report']['voice_assignments'] = voices
        result['report']['speaker_assignment_basis'] = 'verified_source_speech_units'
        return result
    finally:
        try:
            # Retain native attempts and diagnostics even when synthesis or fitting fails.
            volume.commit()
        finally:
            monitor.set()


@app.local_entrypoint()
def main(smoke: bool = False):
    source = ROOT / 'modal_control/batch_inputs/gpu_smoke.json'
    data = json.loads(source.read_text())
    name = 'cpu_v2_preflight' if smoke else 'cpu_v2_smoke'
    audio = ROOT / f'modal_control/batch_inputs/{name}.wav'
    report = ROOT / f'modal_control/batch_inputs/{name}.json'
    if audio.exists() or report.exists():
        raise FileExistsError(f'Output already exists for {name}')
    from batch_dubbing.cpu_natural import select_units
    duration = 15 if smoke else 60
    select_units(data, duration, 'de')
    result = synthesize.remote(data, duration)
    report.write_text(json.dumps(result['report'], indent=2) + '\n')
    if result['audio'] is None or result['report']['status'] != 'complete':
        raise RuntimeError(f"Natural dubbing needs review; diagnostics saved to {report}")
    audio.write_bytes(result['audio'])
