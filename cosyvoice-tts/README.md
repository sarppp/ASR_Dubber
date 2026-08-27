# cosyvoice-tts

Fun-CosyVoice3-0.5B TTS worker for the dubbing pipeline. Selected with
`--tts-engine cosyvoice` on `run_pipeline.py`, `dub.py`, or `dub_modal.py`.

`cosyvoice_tts_worker.py` speaks the same stdin/stdout JSON protocol as
`qwen3-tts/qwen_tts_worker.py`, so it is a drop-in TTS backend.

## Setup

```bash
git submodule update --init --recursive     # pulls CosyVoice/ + Matcha-TTS
uv sync --project cosyvoice-tts
```

Model weights (~10 GB) download automatically on first run to the
Hugging Face cache, or set `COSYVOICE_MODEL_DIR` to a pre-downloaded
`Fun-CosyVoice3-0.5B` directory.

## Env vars

| var | default | meaning |
| --- | --- | --- |
| `COSYVOICE_REPO_DIR`  | `./CosyVoice` (bundled submodule) | CosyVoice checkout |
| `COSYVOICE_MODEL_DIR` | HF-downloaded | Fun-CosyVoice3-0.5B weights dir |
| `COSY_FALLBACK_PROMPT`| `CosyVoice/asset/zero_shot_prompt.wav` | prompt wav for `custom` mode |
