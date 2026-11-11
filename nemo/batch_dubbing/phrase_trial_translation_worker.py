"""In-process Qwen 14B inference on Modal T4. No external inference service."""
import json
from pathlib import Path
import time
import modal

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / 'experiments/crispr_phrase_modal_2026-10-30_10min_qwen'
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('libass9', 'fonts-dejavu-core', 'fontconfig')
         .pip_install('vllm==0.9.2', 'transformers==4.51.3', 'huggingface_hub==0.34.4',
                      'xformers==0.0.30', 'psutil')
         .run_commands('python -c "from vllm import LLM; from vllm.sampling_params import GuidedDecodingParams; import xgrammar"')
         .env({'HF_HOME': '/cache/huggingface', 'TOKENIZERS_PARALLELISM': 'false',
               'VLLM_USE_V1': '0', 'VLLM_ATTENTION_BACKEND': 'XFORMERS',
               'OMP_NUM_THREADS': '4', 'VLLM_NO_USAGE_STATS': '1', 'DO_NOT_TRACK': '1'})
         .add_local_file(ROOT / 'batch_dubbing/phrase_trial.py', '/work/batch_dubbing/phrase_trial.py')
         .add_local_file(ROOT / 'batch_dubbing/trial_telemetry.py', '/work/telemetry.py')
         .add_local_file(ROOT / 'subtitle_divider/bilingual_subtitles.py', '/work/bilingual_subtitles.py'))
volume = modal.Volume.from_name('speech-model-benchmark-20261027')
app = modal.App('crispr-10min-qwen-20261030-translation')


@app.function(image=image, gpu='T4', cpu=(4, 4), memory=(32768, 32768), timeout=1800,
              startup_timeout=180, max_containers=1, min_containers=0, retries=0,
              scaledown_window=2, volumes={'/cache': volume})
def translate(speech, smoke=False, resume=None):
    import sys
    import subprocess
    import uuid
    sys.path.insert(0, '/work')
    from batch_dubbing.phrase_trial import (MODEL, REVISION, BATCH_SIZE, phrase_rows,
                                           translation_messages, validate_translations, bind_translations, translation_schema,
                                           split_phrase_for_layout)
    from bilingual_subtitles import AssMeasurer, _ass_header
    from telemetry import emit, start
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                  text=True, timeout=15).strip()
    if 'T4' not in gpu:
        raise RuntimeError('This release permits T4 only')
    emit('worker_ready', gpu=gpu, model=MODEL, smoke=smoke)
    monitor = start()
    started = time.monotonic()
    artifact = Path('/cache/crispr-phrase-test') / uuid.uuid4().hex
    artifact.mkdir(parents=True)
    emit('artifact_directory', path=str(artifact))
    audit, translated = [], []
    if resume:
        speech['phrase_split_before_word_indices'] = resume.get('phrase_split_before_word_indices', [])
    all_rows = phrase_rows(speech)
    rows = all_rows
    start_at = 0
    if smoke:
        start_at = len(resume['rows']) if resume and resume['rows'] else 205
        rows = rows[start_at:start_at+8]
    elif resume:
        completed = resume['rows']
        translated = validate_translations(json.dumps({'phrases': [
            {key: row[key] for key in ('id', 'de', 'fr')} for row in completed]}), rows[:len(completed)])
        if translated != completed or len(completed) % BATCH_SIZE:
            raise ValueError('Checkpoint does not match current source phrases')
        audit = list(resume['audit'])
    try:
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        model = LLM(model=MODEL, revision=REVISION, tokenizer_revision=REVISION,
                    quantization='awq', dtype='half', max_model_len=8192,
                    gpu_memory_utilization=0.88, max_num_seqs=1, enforce_eager=True,
                    trust_remote_code=False, disable_log_stats=True, num_scheduler_steps=1)
        tokenizer = model.get_tokenizer()
        loaded = time.monotonic()
        with AssMeasurer(_ass_header(28, 20).replace('WrapStyle: 0', 'WrapStyle: 2')) as measurer:
            offset = len(translated)
            while offset < len(rows):
                batch = rows[offset:offset + BATCH_SIZE]
                repair = None
                split_required = False
                for attempt in range(2):
                    if len(audit) >= 1200:
                        raise ValueError('Translation generation-call budget exhausted')
                    if time.monotonic() - started > (600 if smoke else 1650):
                        raise TimeoutError('Translation reached its reviewed deadline')
                    target_index = batch[0]['id'] - 1
                    messages = translation_messages(batch, all_rows[max(0, target_index-2):target_index],
                                                    all_rows[target_index+1:target_index+3], repair)
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                           add_generation_prompt=True, enable_thinking=False)
                    if len(tokenizer.encode(prompt)) + 384 > 8192:
                        raise ValueError('Prompt exceeds reviewed context limit')
                    emit('translating', first_id=batch[0]['id'], attempt=attempt + 1)
                    # Native byte-level grammar supports Qwen multilingual tokens.
                    # This XGrammar version cannot enforce array length; the
                    # validator below checks exact count and IDs before saving.
                    schema = translation_schema(batch)
                    schema['properties']['phrases'].pop('minItems')
                    schema['properties']['phrases'].pop('maxItems')
                    parameters = SamplingParams(temperature=0, max_tokens=384, seed=17,
                        guided_decoding=GuidedDecodingParams(json=schema, backend='xgrammar',
                            disable_fallback=True, disable_any_whitespace=True))
                    before = time.monotonic()
                    output = model.generate([prompt], parameters, use_tqdm=False)[0]
                    raw = output.outputs[0].text
                    record = {'first_id': batch[0]['id'], 'attempt': attempt + 1, 'raw': raw,
                              'input_tokens': len(output.prompt_token_ids),
                              'output_tokens': len(output.outputs[0].token_ids),
                              'finish_reason': output.outputs[0].finish_reason,
                              'seconds': time.monotonic() - before}
                    audit.append(record)
                    try:
                        result = validate_translations(raw, batch)
                        for row in result:
                            for lang in ('en', 'de', 'fr'):
                                left, _, right, _ = measurer.bounds(row[lang], 28)
                                if right - left > 1100:
                                    raise ValueError(f'Phrase {row["id"]} {lang} is too wide at fixed 28px')
                        translated.extend(result)
                        break
                    except ValueError as exc:
                        record['validation_error'] = str(exc)
                        repair = {'error': str(exc), 'raw': raw}
                        if attempt:
                            import re
                            length = re.search(r'line \((\d+);', str(exc))
                            if (length and int(length[1]) > 84) or 'too wide' in str(exc):
                                all_rows = split_phrase_for_layout(speech, batch[0])
                                rows = all_rows[start_at:start_at+8] if smoke else all_rows
                                record['adaptive_split'] = speech['phrase_split_before_word_indices'][-1:]
                                emit('phrase_split', id=batch[0]['id'], source_words=batch[0]['source_word_indices'])
                                split_required = True
                                break
                            raise
                (artifact / 'checkpoint.json').write_text(json.dumps({'rows': translated, 'audit': audit,
                    'phrase_split_before_word_indices': speech.get('phrase_split_before_word_indices', [])}, ensure_ascii=False))
                volume.commit()
                if split_required:
                    continue
                offset = len(translated)
                emit('translated', completed=len(translated), total=len(rows))
        result = {'status': 'complete', 'phrases': translated} if smoke else bind_translations(speech, translated)
        result.update(model=MODEL, revision=REVISION, smoke=smoke, gpu=gpu,
                      translation_strategy='One target phrase per request; neighboring English only as context',
                      json_enforcement='vLLM 0.9.2 V0 XGrammar; exact count and IDs validated afterward',
                      phrase_split_before_word_indices=speech.get('phrase_split_before_word_indices', []),
                      audit=audit, artifact_directory=str(artifact),
                      timings={'load_seconds': loaded-started, 'generation_seconds': time.monotonic()-loaded,
                               'worker_seconds': time.monotonic()-started},
                      inference_location='Modal T4; local vLLM Python inference, no external inference API')
        return result
    finally:
        (artifact / 'partial.json').write_text(json.dumps({'rows': translated, 'audit': audit,
            'phrase_split_before_word_indices': speech.get('phrase_split_before_word_indices', [])}, ensure_ascii=False))
        volume.commit()
        monitor.set()


@app.local_entrypoint()
def main(smoke: bool = False):
    from batch_dubbing.phrase_trial import write_json, sha256
    source = TRIAL / 'asr.json'
    destination = TRIAL / ('translation_smoke_unicode.json' if smoke else 'translated.json')
    if destination.exists():
        raise FileExistsError(destination)
    started = time.monotonic()
    resume_file = TRIAL / 'translation_unicode_resume.json'
    resume = json.loads(resume_file.read_text()) if resume_file.exists() else None
    if not smoke and resume and (TRIAL / 'translation_smoke_unicode.json').exists():
        tested = json.loads((TRIAL / 'translation_smoke_unicode.json').read_text())
        resume['phrase_split_before_word_indices'] = tested['phrase_split_before_word_indices']
    result = translate.remote(json.loads(source.read_text()), smoke, resume)
    result['input_sha256'] = sha256(source)
    result['client_call_seconds'] = time.monotonic() - started
    write_json(destination, result)
    if result['status'] != 'complete':
        raise RuntimeError('Translation incomplete')
    print('RESULT', destination, flush=True)
