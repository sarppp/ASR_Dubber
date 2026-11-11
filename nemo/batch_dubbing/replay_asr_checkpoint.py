"""Reproduce a confirmed dub through the actual runner from validated ASR/MT checkpoints.

Preparation is model-free. --run generates fresh Qwen speech and CPU alignments.
It never substitutes the reference audio or final subtitle outputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess

from batch_dubbing.aligned_finish import read_srt, sha
from batch_dubbing.reference_regression import verified_assets, compare_artifacts
from subtitle_combine import write_files


def prepare(manifest_path, output):
    manifest, assets = verified_assets(manifest_path)
    if manifest.get('placement_mode') != 'continuous_synced':
        raise ValueError('This replay requires the explicitly selected synced reference')
    voice_entry = manifest['voice_reference']
    voice = (Path(manifest_path).resolve().parent / voice_entry['path']).resolve()
    if sha(voice) != voice_entry['sha256']:
        raise ValueError('Voice reference changed')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    base = 'synced_replay_' + hashlib.sha256(str(output).encode()).hexdigest()[:10]
    inputs = output / 'input'; inputs.mkdir()
    video = inputs / f'{base}.mp4'
    shutil.copyfile(assets['video'], video)
    source, target = manifest['source_language'], manifest['target_language']
    stage = inputs / '.subtitle_dub' / base; stage.mkdir(parents=True)
    for lang, role in ((source, 'source_srt'), (target, 'target_srt')):
        shutil.copyfile(assets[role], stage / f'{base}.nemo.{lang}.srt')
    write_files({source: read_srt(assets['source_srt']), target: read_srt(assets['target_srt'])},
                stage / base, order=[source, target])
    command = [str(root / 'qwen3-tts/.venv/bin/python'), '-u', str(root / 'run_pipeline.py'),
               '--pipeline', 'subtitle-dubbing', '--input-file', str(video),
               '--output-dir', str(output / 'result'), '--language', source, '--target-lang', target,
               '--subtitle-languages', source, target, '--asr-model', 'parakeet-ultra', '--translator', 'tower',
               '--tts-engine', 'qwen', '--qwen-mode', 'clone', '--voice-reference', str(voice),
               '--no-demucs', '--tts-workers', '1', '--tts-devices', '0', '--dub-retries', '0',
               '--dub-timing', 'continuous-synced']
    request = dict(command=command, base=base, timeout_seconds=720, manifest_sha256=sha(manifest_path),
                   cached_stages='Original verified Ultra ASR and Tower subtitle checkpoints',
                   fresh_stages='Qwen clone synthesis, source/raw/adjusted CPU alignment, measured sync and mux',
                   input_sha256={role: sha(assets[role]) for role in ('video', 'source_srt', 'target_srt')})
    (output / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
    return request, manifest, assets


def run(manifest_path, output):
    # Check availability before creating a run; don't interfere with another job.
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('GPU occupied; wait for other jobs, never terminate them')
    request, manifest, assets = prepare(manifest_path, output)
    output = Path(output).resolve(); root = Path(__file__).resolve().parents[2]
    from huggingface_hub import snapshot_download
    model = snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base',
                              revision='fd4b254389122332181a7c3db7f27e918eec64e3', local_files_only=True)
    env = dict(os.environ, HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
               TOKENIZERS_PARALLELISM='false', QWEN_TTS_MODEL_PATH=model,
               ALIGNMENT_PYTHON=str(root / 'nemo/.venv/bin/python'))
    with (output / 'pipeline.log').open('w') as log:
        proc = subprocess.Popen(request['command'], cwd=root, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=request['timeout_seconds'])
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            code = 124
    (output / 'execution.json').write_text(json.dumps(dict(returncode=code), indent=2) + '\n')
    if code:
        raise RuntimeError(f'Pipeline failed ({code}); inspect pipeline.log')
    movies = list((output / 'result').rglob('final_dub.mp4'))
    if len(movies) != 1:
        raise ValueError('Missing or ambiguous generated result')
    result = movies[0].parent
    work = root / 'qwen3-tts/output/dub' / request['base']
    # Gather this run's freshly rendered WAV for exact comparison, not a reference.
    shutil.copyfile(work / 'output/dub_track.wav', result / 'dub_track.wav')
    comparison = compare_artifacts(assets, result, manifest['source_language'],
                                   manifest['target_language'], placement_mode='continuous_synced')
    raw = json.loads((work / 'alignment_request.json').read_text())['takes'][0]['file']
    comparison['newly_generated_raw_voice_matches_reference'] = sha(raw) == sha(assets['raw_take'])
    comparison['cached_stages'] = request['cached_stages']
    comparison['fresh_stages'] = request['fresh_stages']
    (output / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    if comparison['status'] != 'match':
        raise RuntimeError('Generated result differs; inspect comparison.json')
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', action='store_true', help='Run fresh local synthesis; otherwise only prepare')
    args = parser.parse_args()
    result = run(args.manifest, args.output) if args.run else prepare(args.manifest, args.output)[0]
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
