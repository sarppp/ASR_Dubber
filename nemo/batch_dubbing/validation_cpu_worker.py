"""Prepared bounded native-speech evaluation; no models run on import."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'dubbing_naturalness_2026-10-28/source_validation_inputs.json'
OUTPUT = ROOT / 'dubbing_naturalness_2026-10-28/source_validation_2026-10-29'
NATIVE_BYTES_LIMIT = 64 * 1024**2
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('ffmpeg', 'libsndfile1')
         .pip_install('supertonic==1.3.1', 'huggingface_hub', 'soundfile', 'numpy<2', 'psutil')
         .env({'HF_HOME': '/cache/huggingface', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
         .add_local_file(CONFIG, '/work/source_validation_inputs.json')
         .add_local_file(ROOT / 'batch_dubbing/validation_config.py', '/work/batch_dubbing/validation_config.py')
         .add_local_file(ROOT / 'batch_dubbing/natural_pipeline.py', '/work/batch_dubbing/natural_pipeline.py')
         .add_local_file(ROOT / 'batch_dubbing/audio_edges.py', '/work/batch_dubbing/audio_edges.py')
         .add_local_file(ROOT / 'batch_dubbing/speech_schedule.py', '/work/batch_dubbing/speech_schedule.py')
         .add_local_file(ROOT / 'batch_dubbing/natural_render.py', '/work/batch_dubbing/natural_render.py')
         .add_local_file(ROOT / 'batch_dubbing/cpu_natural.py', '/work/batch_dubbing/cpu_natural.py')
         .add_local_file(ROOT / 'speech_model_benchmark_2026-10-27/code/telemetry.py', '/work/telemetry.py'))
volume = modal.Volume.from_name('speech-model-benchmark-20261027')
app = modal.App('source-validation-cpu')


def prepare_cases(config, gpu_result, smoke=False):
    """Check the complete GPU result and voices before any remote/model call."""
    from batch_dubbing.validation_config import validate_config, selected_cases
    from batch_dubbing.cpu_natural import select_units
    config = validate_config(config)
    if gpu_result.get('status') != 'complete' or gpu_result.get('stage') != 'gpu_full':
        raise ValueError('CPU validation requires a complete gpu_full result')
    if validate_config(gpu_result.get('config')) != config:
        raise ValueError('GPU result configuration differs from the released input configuration')
    rows = gpu_result.get('cases', [])
    if len(rows) != len(config['cases']) or {r['id'] for r in rows} != {c['id'] for c in config['cases']}:
        raise ValueError('GPU result must contain every requested case exactly once')
    by_id = {row['id']: row['data'] for row in rows}
    prepared = {}
    for case in config['cases']:
        data = by_id[case['id']]
        if (data.get('status') != 'complete' or data.get('duration_seconds') != case['duration_seconds']
                or data.get('source_offset_seconds') != case['offset_seconds'] or data.get('source_language') != 'en'):
            raise ValueError('GPU case is incomplete or its duration changed')
        units, scope = select_units(data, case['duration_seconds'], config['target_language'])
        if len(units) > config['max_units_per_case'] or scope['excluded_boundary_units']:
            raise ValueError('Case exceeds the unit cap or has excluded boundary speech')
        voices = data.get('voices', {})
        if any(not voices.get(unit['speaker'], {}).get('voice') for unit in units):
            raise ValueError('Uncertain speaker voice requires review before synthesis')
        prepared[case['id']] = {'id': case['id'], 'data': deepcopy(data),
                               'seconds': case['duration_seconds'], 'unit_ids': [u['id'] for u in units]}
    selected = [prepared[case['id']] for case in selected_cases(config, smoke=smoke)]
    if smoke:
        item = selected[0]
        first = item['data']['speech_units'][config['target_language']][0]
        item['seconds'] = first['source_end']
        item['unit_ids'] = [first['id']]
        # Keep the full GPU data unchanged; select_units chooses this whole unit.
        units, _ = select_units(item['data'], item['seconds'], config['target_language'])
        if [u['id'] for u in units] != item['unit_ids']:
            raise ValueError('Smoke scope must contain exactly the first whole utterance')
    return selected


def collect_native(report, remaining):
    """Read only hashed raw attempts below the reported artifact directory."""
    if not report.get('attempts'):
        return [], remaining
    root = Path(report['native_artifact_directory']).resolve()
    artifacts = []
    for index, attempt in enumerate(report.get('attempts', [])):
        source = Path(attempt['file']).resolve()
        if source.parent != root or source.suffix != '.wav':
            raise ValueError('Native artifact escapes its report directory')
        size = source.stat().st_size
        if size > remaining:
            raise ValueError('Native audio transfer exceeds the 64 MiB total cap')
        content = source.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != attempt['sha256']:
            raise ValueError('Native audio hash differs from its measured attempt')
        remaining -= len(content)
        artifacts.append({'name': f'raw_{index:03d}.wav', 'sha256': digest, 'bytes': content,
                          'unit_id': attempt['id'], 'attempt': attempt['attempt']})
    return artifacts, remaining


def run_cases(config, gpu_result, smoke, *, synthesizer, commit, emit=None):
    cases = prepare_cases(config, gpu_result, smoke)
    results, remaining = [], NATIVE_BYTES_LIMIT
    for case in cases:
        value = {}
        if emit:
            emit('case_started', id=case['id'], stage='synthesis', unit_count=len(case['unit_ids']))
        try:
            value = synthesizer(case['data'], case['seconds'], case['data']['voices'],
                                language=config['target_language'], output_format='wav', emit=emit)
            native, remaining = collect_native(value['report'], remaining)
            results.append({'id': case['id'], 'report': value['report'],
                            'audio': value['audio'], 'native': native, 'unit_ids': case['unit_ids']})
        except Exception as exc:
            failed = deepcopy(value.get('report', {}))
            failed.update(status='failed', evaluation_error=str(exc))
            results.append({'id': case['id'], 'report': failed,
                            'audio': None, 'native': [], 'unit_ids': case['unit_ids']})
        if emit:
            emit('case_complete', id=case['id'], status=results[-1]['report'].get('status'))
        commit()
    complete = all(row['report'].get('status') == 'complete' and row['audio'] is not None for row in results)
    return {'status': 'complete' if complete else 'blocked', 'stage': 'cpu_smoke' if smoke else 'cpu_full',
            'config': config, 'cases': results, 'native_bytes': NATIVE_BYTES_LIMIT - remaining,
            'automatic_revisions': 0, 'retries': 0}


def publish_result(destination, result):
    """Verify all hashes first; publish timelines only for an entirely ready batch."""
    destination = Path(destination)
    metadata = {key: value for key, value in result.items() if key != 'cases'}
    metadata['cases'] = []
    seen, total = set(), 0
    try:
        for row in result['cases']:
            ident = row['id']
            if not isinstance(ident, str) or not ident.isascii() or not ident.replace('_', '').replace('-', '').isalnum() or ident in seen:
                raise ValueError('Invalid or duplicate output case ID')
            seen.add(ident)
            names = set()
            for artifact in row['native']:
                name = artifact['name']
                if Path(name).name != name or not name.endswith('.wav') or name in names:
                    raise ValueError('Invalid native filename')
                names.add(name)
                total += len(artifact['bytes'])
                if total > NATIVE_BYTES_LIMIT or hashlib.sha256(artifact['bytes']).hexdigest() != artifact['sha256']:
                    raise ValueError('Native artifact transfer hash or size mismatch')
            if result['status'] == 'complete':
                if row['report'].get('status') != 'complete' or row['audio'] is None:
                    raise ValueError('Incomplete batch cannot publish timelines')
                if hashlib.sha256(row['audio']).hexdigest() != row['report']['rendered']['sha256']:
                    raise ValueError('Timeline transfer hash mismatch')
    except Exception as exc:
        metadata.update(status='failed', publication_error=str(exc))
        with (destination / 'report.json').open('x') as handle:
            json.dump(metadata, handle, indent=2)
        raise
    for row in result['cases']:
        case_dir = destination / row['id']
        case_dir.mkdir()
        native_dir = case_dir / 'native'
        native_dir.mkdir()
        manifest = []
        for artifact in row['native']:
            with (native_dir / artifact['name']).open('xb') as handle:
                handle.write(artifact['bytes'])
            manifest.append({key: value for key, value in artifact.items() if key != 'bytes'})
        with (case_dir / 'report.json').open('x') as handle:
            json.dump(row['report'], handle, ensure_ascii=False, indent=2)
        if result['status'] == 'complete':
            with (case_dir / 'timeline.wav').open('xb') as handle:
                handle.write(row['audio'])
        metadata['cases'].append({'id': row['id'], 'status': row['report']['status'],
                                  'unit_ids': row['unit_ids'], 'native': manifest})
    with (destination / 'report.json').open('x') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


@app.function(image=image, cpu=2, memory=4096, timeout=600, max_containers=1,
              retries=0, volumes={'/cache': volume})
def process(config, gpu_result, smoke=False):
    import sys
    sys.path.insert(0, '/work')
    from telemetry import emit, start
    emit('worker_ready', hardware='CPU', evaluation='source_validation', smoke=smoke)
    monitor = start()
    try:
        from batch_dubbing.cpu_natural import synthesize_natural
        return run_cases(config, gpu_result, smoke, synthesizer=synthesize_natural, commit=volume.commit, emit=emit)
    finally:
        try:
            volume.commit()
        finally:
            monitor.set()


@app.local_entrypoint()
def main(smoke: bool = False):
    from batch_dubbing.validation_config import validate_config
    config = validate_config(json.loads(CONFIG.read_text()))
    gpu_result = json.loads((OUTPUT / 'gpu_full.json').read_text())
    prepare_cases(config, gpu_result, smoke)
    destination = OUTPUT / ('cpu_smoke' if smoke else 'cpu_full')
    destination.mkdir(parents=True, exist_ok=False)
    try:
        result = process.remote(config, gpu_result, smoke)
    except Exception as exc:
        with (destination / 'report.json').open('x') as handle:
            json.dump({'status': 'failed', 'error': str(exc)}, handle, indent=2)
        raise
    metadata = publish_result(destination, result)
    print('RESULT', destination / 'report.json', flush=True)
    if metadata['status'] != 'complete':
        raise RuntimeError(f'CPU validation blocked; diagnostics saved to {destination}')
