"""Prepare the complete bounded CRISPR trial; never launch model inference."""
import json
from pathlib import Path
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / 'experiments/crispr_phrase_modal_2026-10-30_10min_qwen'


def prepare():
    from batch_dubbing.phrase_trial import check_audio_input, MODEL, REVISION
    from batch_dubbing.phrase_trial_qwen import MODEL as TTS_MODEL, REVISION as TTS_REVISION
    audio, manifest = check_audio_input(TRIAL)
    with wave.open(str(audio)) as wav:
        duration = wav.getnframes() / wav.getframerate()
    if duration != 600 or manifest.get('clip_duration_seconds') != 600 or manifest.get('clip_start_seconds') != 0:
        raise ValueError('This plan only accepts the first ten minutes')
    python = str(ROOT / 'experiments/crispr_phrase_modal_2026-10-30/.runner/bin/python')
    prefix = str(TRIAL.relative_to(ROOT))
    shared = ['batch_dubbing/phrase_trial.py', 'batch_dubbing/trial_telemetry.py',
              'batch_dubbing/prepare_phrase_trial.py', 'batch_dubbing/phrase_trial_report.py',
              'batch_dubbing/phrase_trial_publish.py', 'batch_dubbing/phrase_trial_qwen.py',
              'batch_dubbing/phrase_trial_costs.py',
              prefix + '/input/manifest.json']
    asr_sources = shared + ['batch_dubbing/speech_stage.py', 'batch_dubbing/speech_units.py',
        'modal_control/smoke_nemo_helpers.py', 'modal_control/speaker_voices.py',
        'local_subtitles.py', 'nemo_audio.py', prefix + '/input/source.wav']
    text_sources = shared + ['subtitle_divider/bilingual_subtitles.py', prefix + '/asr.json']
    if (TRIAL / 'translation_unicode_resume.json').exists():
        text_sources.append(prefix + '/translation_unicode_resume.json')
    dub_sources = shared + [f'batch_dubbing/{name}.py' for name in
        ('cpu_natural', 'natural_pipeline', 'natural_render', 'speech_schedule', 'audio_edges')]
    dub_sources += [prefix + '/translated.json']
    jobs = {}
    specs = [
        ('asr', 'T4', 900, 32768, asr_sources, 'asr_smoke.json', ['asr.json'],
         'Parakeet TDT 0.6B v3; NeMo MarbleNet VAD and TitaNet speaker embeddings',
         'Parakeet 541d1f99c6b0c3cd0b11a95167540bb8edefd82b; NeMo 2.5.3 registry VAD/TitaNet',
         'FP16 ASR; cached diarization weights',
         'Fresh English words, acoustic timestamps, source-linked utterances and speaker assignments; NO Canary translation.',
         '74.24-second opening smoke, then the complete 00:00-10:00 clip.'),
        ('translation', 'T4', 1800, 32768, text_sources, 'translation_smoke_unicode.json', ['translated.json'],
         MODEL, REVISION, 'AWQ int4 weights, FP16 activations; vLLM 0.9.2 offline Python calls',
         'Short English phrases translated directly to DE/FR with shared immutable IDs and surrounding context, as in the agent-edited SRT.',
         'Eight phrases from a valid checkpoint, or phrases 206-213 for the Unicode regression smoke, then all phrases. At most 600 final phrases and 1200 generation calls including retained audit; one target per request, one repair per phrase, 384 output tokens per call. XGrammar JSON decoding; overlong lines split automatically at source word boundaries. Prior corrupted target text is discarded.'),
        ('dub', 'T4', 3600, 32768, dub_sources, 'dub_smoke.json', ['dub.json', 'german-dub.m4a'],
         TTS_MODEL, TTS_REVISION + '; qwen-tts 0.1.1',
         'FP32, SDPA, stock Serena/Ryan voices; native 24000 Hz, resampled to 44100 for timing utilities',
         'German phrases joined into source utterances for natural delivery, then Qwen stock-voice synthesis and complete-utterance timing fit.',
         'First two complete utterances as smoke, then the full ten-minute clip; at most 400 utterances; one synthesis each, max 1440 tokens; no cloud retry.')]
    for name, hardware, timeout, memory, sources, smoke_file, success, model, revision, precision, purpose, smoke_scope in specs:
        worker = f'batch_dubbing/phrase_trial_{name}_worker.py'
        command = [python, '-m', 'modal', 'run', worker]
        jobs[name] = {'cwd': str(ROOT), 'command': command, 'smoke_command': command + ['--smoke'],
            'source_files': [worker] + sources,
            'smoke_success_files': [prefix + '/' + smoke_file],
            'success_files': [prefix + '/' + path for path in success],
            'max_seconds': timeout, 'build_timeout_seconds': 900, 'startup_timeout_seconds': 240,
            'idle_timeout_seconds': 300,
            'review': {'purpose': purpose, 'model': model, 'revision': revision,
                'precision': precision, 'reason': 'User-requested all-Modal comparison with stronger phrase translation and complete dubbing.',
                'input': f'Only CRISPR Jennifer Doudna, {duration:.6f} seconds, original audio. {smoke_scope}',
                'input_seconds': duration, 'hardware': hardware, 'max_containers': 1,
                'max_total_seconds': 2 * timeout,
                'outputs': f'Isolated {TRIAL}; original media and previous subtitles/dub remain unchanged.',
                'quality_tradeoffs': 'Structural validation is not semantic/listening approval. Word alignment conflicts or unsatisfied layout limits fail with evidence. Dub never clips words or exceeds 1.15x compression; if faithful speech cannot fit, preserve native takes and report needs_revision. Pitch is only a preset-selection heuristic; uncertain voices use disclosed stable stock presets.',
                'cost_estimate': 'Actual cost unknown until billing settles. Bounded compute uses T4 $0.000164/s, CPU $0.0000131/core/s, RAM $0.00000222/GiB/s. Smoke/full counts, load times and failures included; cloud image build/storage/egress can add cost. This is not a provider billing cap.'}}
    catalog_path = TRIAL / 'catalog.json'
    catalog = {'version': 1, 'env_file': str(ROOT / '.env'),
               'modal_command': [python, '-m', 'modal'],
               'hardware_policy': {'allowed_gpus': ['T4'], 'fallback_allowed': False}, 'jobs': jobs}
    catalog_path.write_text(json.dumps(catalog, indent=2) + '\n')
    steps = []
    for name, hardware, timeout, memory, *_ in specs:
        step = {'id': name, 'catalog': str(catalog_path), 'job': name,
                'cpu_cores': 4, 'memory_mib': memory, 'max_attempts': 1}
        if name == 'translation':
            step.update(depends_on=['asr'], derived_inputs={prefix + '/asr.json': 'asr'})
        if name == 'dub':
            step.update(depends_on=['translation'], derived_inputs={prefix + '/translated.json': 'translation'})
        steps.append(step)
    batch = {'version': 1, 'max_parallel': 1, 'max_gpu_parallel': 1,
             'max_wall_seconds': 12600, 'max_resource_seconds': 12600,
             'estimated_cost_ceiling_usd': 4, 'steps': steps}
    (TRIAL / 'batch.json').write_text(json.dumps(batch, indent=2) + '\n')
    scope = {'status': 'prepared_not_released', 'source': manifest['source'],
             'input_seconds': duration, 'clip_start_seconds': 0, 'clip_end_seconds': 600, 'all_inference_on_modal': True,
             'external_inference_services': [], 'gpu': 'T4', 'translation_model': MODEL,
             'dubbing_model': TTS_MODEL, 'dubbing_hardware': 'T4', 'global_container_cap': 1,
             'smoke_and_full_jobs': 6, 'cloud_retries': 0,
             'automatic_text_repairs_per_batch': 1,
             'subtitle_languages': ['en', 'de', 'fr'], 'dub_languages': ['de'],
             'outputs': ['fresh ASR words and speaker turns', 'source-linked phrase JSON',
                         'EN/DE/FR and combined SRT, fixed-size ASS preview',
                         'German dubbed MP4 and M4A, timing checks', 'app-specific billing and timings'],
             'local_processing': 'WAV extraction before model work; SRT/ASS layout checks and stream-copy MP4 assembly after successful jobs.',
             'comparison_baseline': 'Existing CRISPR-synchronized phrases plus original German dub within 00:00-10:00; not model inputs.',
             'cost_note': 'Use app-specific billing rows, not rounded dashboard deltas. Report billed spend separately from credit draw and remaining credit. Cost may not be final immediately after shutdown.'}
    (TRIAL / 'scope.json').write_text(json.dumps(scope, indent=2) + '\n')
    return batch


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
