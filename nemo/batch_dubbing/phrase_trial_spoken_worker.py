"""Duration-aware German speech text on the same reviewed Modal T4 model."""
import json
from pathlib import Path
import time
import modal
from batch_dubbing.phrase_trial_translation_worker import image, volume

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / 'experiments/crispr_phrase_modal_2026-10-30_10min_qwen'
image = image.add_local_file(ROOT / 'batch_dubbing/phrase_trial_spoken.py', '/work/batch_dubbing/phrase_trial_spoken.py')
app = modal.App('crispr-10min-qwen-20261030-spoken')


@app.function(image=image, gpu='T4', cpu=(4, 4), memory=(32768, 32768), timeout=900,
              startup_timeout=180, max_containers=1, min_containers=0, retries=0,
              scaledown_window=2, volumes={'/cache': volume})
def adapt(data, smoke=False, completed=None):
    import sys, subprocess, uuid
    sys.path.insert(0, '/work')
    from batch_dubbing.phrase_trial import MODEL, REVISION
    from batch_dubbing.phrase_trial_spoken import sentence_groups, character_budget, validate_spoken
    from telemetry import emit, start
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    if 'T4' not in gpu:
        raise ValueError('Only T4 is released')
    emit('worker_ready', gpu=gpu, model=MODEL, smoke=smoke)
    monitor = start(); started = time.monotonic()
    folder = Path('/cache/crispr-spoken') / uuid.uuid4().hex; folder.mkdir(parents=True)
    rows = sentence_groups(data)
    if smoke: rows = rows[:2]
    units, audit = [], []
    if completed and not smoke:
        for old, unit in zip(completed['units'], rows):
            verified = validate_spoken(json.dumps({'de': old['translated_text']}), unit)
            if verified != old: raise ValueError('Spoken smoke provenance mismatch')
            units.append(verified)
    def persist():
        (folder/'checkpoint.json').write_text(json.dumps({'units':units,'audit':audit},ensure_ascii=False))
        volume.commit()
    try:
        model = LLM(model=MODEL, revision=REVISION, tokenizer_revision=REVISION, quantization='awq',
                    dtype='half', max_model_len=8192, gpu_memory_utilization=.88, max_num_seqs=1,
                    enforce_eager=True, trust_remote_code=False, disable_log_stats=True, num_scheduler_steps=1)
        tokenizer = model.get_tokenizer(); loaded = time.monotonic()
        for unit in rows[len(units):]:
            messages = [{'role':'system','content':
                'Translate the English passage into concise, fluent spoken German for a documentary dub. '
                'Translate the complete meaning as coherent German sentences, not isolated subtitle fragments. '
                'Preserve every factual claim, scientific term, uncertainty, negation and speaker intent. '
                'Omit only empty conversational fillers and redundant verbal repetitions. Do not add ideas. '
                'Use correct German spelling and accents. Return JSON with only the key de.'},
                {'role':'user','content':json.dumps({'source':unit['source_text'],
                    'seconds':unit['source_end']-unit['source_start'],
                    'maximum_characters':character_budget(unit)},ensure_ascii=False)}]
            for attempt in range(2):
                if time.monotonic()-started > 800 or len(audit) >= 160:
                    raise TimeoutError('Spoken translation budget exhausted')
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                if len(tokenizer.encode(prompt))+384 > 8192: raise ValueError('Context bound exceeded')
                params = SamplingParams(temperature=0,max_tokens=384,seed=17,guided_decoding=GuidedDecodingParams(
                    json={'type':'object','properties':{'de':{'type':'string'}},'required':['de'],'additionalProperties':False},
                    backend='xgrammar',disable_fallback=True,disable_any_whitespace=True))
                before=time.monotonic();raw=model.generate([prompt],params,use_tqdm=False)[0].outputs[0].text
                record={'id':unit['id'],'attempt':attempt+1,'raw':raw,'seconds':time.monotonic()-before};audit.append(record)
                try:
                    units.append(validate_spoken(raw,unit));break
                except (ValueError,TypeError) as exc:
                    record['error']=str(exc)
                    if attempt: raise
                    messages += [{'role':'assistant','content':raw},{'role':'user','content':str(exc)}]
            persist();emit('spoken_translated',completed=len(units),total=len(rows))
        return {'status':'complete','units':units,'audit':audit,'model':MODEL,'revision':REVISION,
                'timings':{'worker_seconds':time.monotonic()-started,'load_seconds':loaded-started},
                'artifact_directory':str(folder),'semantic_quality_verified':False}
    finally:
        persist();monitor.set()


@app.local_entrypoint()
def main(smoke: bool=False):
    from batch_dubbing.phrase_trial import sha256,write_json
    source=TRIAL/'translated.json';data=json.loads(source.read_text())
    destination=TRIAL/('spoken_smoke.json' if smoke else 'dub_ready.json')
    if destination.exists():raise FileExistsError(destination)
    completed=None if smoke else json.loads((TRIAL/'spoken_smoke.json').read_text())
    if completed and completed['input_sha256']!=sha256(source):raise ValueError('Smoke input changed')
    started=time.monotonic();result=adapt.remote(data,smoke,completed)
    result.update(input_sha256=sha256(source),client_call_seconds=time.monotonic()-started)
    if not smoke:
        data.update(speech_units={'de':result['units']},spoken_translation=result)
        data['input_sha256']=sha256(source);result=data
    write_json(destination,result)
