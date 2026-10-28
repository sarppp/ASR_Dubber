"""Bounded, controller-released CRISPR subtitle translation; no local inference."""
from pathlib import Path
import json

import modal

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "subtitle_divider"
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("libass9", "fonts-dejavu-core", "fontconfig")
         .pip_install("torch==2.6.0", "transformers==4.56.2", "accelerate==1.10.1")
         .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_file(HERE / "bilingual_subtitles.py", "/work/bilingual_subtitles.py")
         .add_local_file(HERE / "aligned_translation.py", "/work/aligned_translation.py")
         .add_local_file(HERE / "repair/crispr-source.json", "/work/source.json"))
volume = modal.Volume.from_name("speech-model-benchmark-20261027")
app = modal.App("crispr-source-aligned-srt")


@app.function(image=image, gpu="T4", cpu=2, memory=16384, timeout=3600,
              max_containers=1, retries=0, scaledown_window=2,
              volumes={"/cache": volume})
def translate(smoke: bool):
    import sys
    import time
    import subprocess
    sys.path.insert(0, "/work")
    from aligned_translation import MODEL, REVISION, BATCH_SIZE, messages_for, validate_response, validate_smoke
    started = time.monotonic()
    deadline = started + (600 if smoke else 3500)
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
    if "T4" not in gpu:
        raise RuntimeError(f"T4 required; received {gpu}")
    def emit(kind, **fields):
        print("MC_EVENT " + json.dumps({"kind": kind, **fields}), flush=True)
    emit("worker_ready", model=MODEL, revision=REVISION, gpu=gpu, smoke=smoke)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa",
    ).eval()
    document = json.loads(Path("/work/source.json").read_text())
    source = document["cues"]
    selected = [row for row in source if 23000 <= row["start_ms"] < 37000] if smoke else source
    translated, attempts = [], []
    for offset in range(0, len(selected), BATCH_SIZE):
        batch = selected[offset:offset + BATCH_SIZE]
        first = batch[0]["id"] - 1
        last = batch[-1]["id"]
        messages = messages_for(batch, " ".join(r["en"] for r in source[max(0, first-2):first]),
                                " ".join(r["en"] for r in source[last:last+2]))
        for attempt in range(2):  # One disclosed text repair; never retry cloud jobs.
            remaining = deadline - time.monotonic()
            if remaining < 10:
                raise TimeoutError("Subtitle translation reached its reviewed deadline")
            emit("translating", first_id=batch[0]["id"], attempt=attempt+1)
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                   tokenize=True, return_dict=True,
                                                   return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=2048,
                                           max_time=min(180, remaining), do_sample=False)
            raw = tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            attempts.append({"ids": [r["id"] for r in batch], "attempt": attempt+1, "output": raw})
            try:
                rows = validate_response(raw, batch)
                break
            except ValueError as exc:
                if attempt:
                    raise ValueError(f"Cue {batch[0]['id']} failed both bounded attempts: {exc}") from exc
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": f"Repair this error, preserving every cue's meaning and ID: {exc}. Return the complete corrected JSON array."}]
        translated.extend(rows)
        emit("translated", completed=len(translated), total=len(selected), seconds=time.monotonic()-started)
    if smoke:
        validate_smoke(translated, selected)
    return {"status": "complete", "model": MODEL, "revision": REVISION,
            "source_sha256": document["source_sha256"], "rows": translated,
            "attempts": attempts, "seconds": time.monotonic()-started, "gpu": gpu}


@app.local_entrypoint()
def main(smoke: bool = False):
    output = HERE / "repair" / ("crispr-smoke.json" if smoke else "crispr-translated.json")
    if output.exists():
        raise FileExistsError(output)
    result = translate.remote(smoke)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print("RESULT", output, flush=True)
