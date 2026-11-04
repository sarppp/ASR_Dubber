# English, German, and French subtitles

Run from the `nemo` directory. The runner processes media files inside
`--input-dir` (default `new_ones`) and writes, next to each video:

| File | Content |
|---|---|
| `<video>.nemo.en.srt` | English only |
| `<video>.nemo.de.srt` | German only |
| `<video>.nemo.fr.srt` | French only |
| **`<video>.multi.srt`** | **all languages, one cue with one line per language** |
| **`<video>.multi.ass`** | same, ASS house style |

The combined `.multi.*` files are **always** produced; they are the deliverable.
Every file shares identical cue timings, so a player can switch between them
mid-video and the text stays in sync.

## Current default pipeline

1. **ASR once per video, in the source language** — `parakeet-ultra`
   (`moondream/parakeet-ultra`), word-level timestamps.
2. **Word packing into cues** — max 10 words, 5 s, 80 chars
   (`nemo_audio._words_to_segs`).
3. **Sentence grouping** — cues are joined into complete sentences.
4. **Whole-sentence translation** — one call per sentence to
   `tower-plus-2b:latest` through the local Ollama container
   (`nemo/local_mt.py`). Sentences are never translated as fragments: fragment
   input is what produced invented subjects and broken compounds.
5. **Deterministic split** — the sentence translation is cut back onto the
   original cue boundaries by character weight, so DE/FR keep the EN timing.
6. **Combined output** — `nemo/subtitle_combine.py` writes `.multi.srt` and
   `.multi.ass` (UTF-8 with BOM, `DejaVu Sans 28`, 1280x720, `\N` between
   languages).

The model only ever sees the source-language text. No reference translation,
existing target subtitle, or other model output is placed in the prompt.

## Local run

```bash
# Default: parakeet-ultra + tower-plus-2b, EN/DE/FR + combined files
.venv-local-asr/bin/python -u batch_subtitles.py --backend local --input-dir new_ones

# English only
.venv-local-asr/bin/python -u batch_subtitles.py --backend local \
    --input-dir new_ones --languages en

# Pick models explicitly
.venv-local-asr/bin/python -u batch_subtitles.py --backend local --input-dir new_ones \
    --asr-model parakeet-ultra --translator tower

# Preview planned outputs without loading a model
.venv-local-asr/bin/python -u batch_subtitles.py --backend local --input-dir new_ones --dry-run
```

| Option | Values | Default |
|---|---|---|
| `--asr-model` | `parakeet-ultra`, `parakeet-v3`, `parakeet-v2`, `canary`, `qwen3-asr` | `parakeet-ultra` |
| `--translator` | `tower` (`tower-plus-2b:latest`), `gemma` (`translategemma:4b`), `tower-7b` (`hymt2-7b`), `canary` (legacy) | `tower` |
| `--source-language` | `en`, `fr`, `de`, `es` | `en` |
| `--languages` | any of `en`, `fr`, `de` | `en fr de` |

`ASR_MODEL` and `TRANSLATOR` environment variables work as well.

## Requirements

- **Ultra** runs in-process in `.venv-local-asr` (torch 2.8.0+cu128 + `moondream`
  2.6.1). Its weights are taken from the local Hugging Face cache
  (`~/.cache/huggingface/hub/models--moondream--parakeet-ultra`); no download and
  no network call happens at run time. The `ultra-asr:local` Docker image (built
  from `new_guys/crispr/ultra_trial/Dockerfile`, weights in the `ultra_hf`
  volume) is only a fallback for environments where `moondream` is not
  importable. Photon's usage telemetry is stubbed off by
  `parakeet_ultra._offline_moondream()`.
- **Tower** needs the Ollama container with `tower-plus-2b:latest` registered
  (`translate-gemma/setup_tower.sh`, `translate-gemma/Modelfile.tower-plus-2b`).
  One model loaded at a time (`OLLAMA_MAX_LOADED_MODELS=1`), `OLLAMA_KEEP_ALIVE=0`.
- The NeMo cache is resolved from `$NEMO_CACHE_DIR`, `$TORCH_HOME` or `$HOME`
  (it used to be hardcoded to `/root/.cache/torch/NeMo`, which broke non-root
  runs).
- GPU access is required. The local backend does not upload media and does not
  invoke Modal.

The original `.venv` is kept. To recreate the local ASR runtime with Python 3.12:

```bash
uv venv --python 3.12 .venv-local-asr
uv pip install --python .venv-local-asr/bin/python \
  --index-strategy unsafe-best-match -r local-asr-requirements.txt
```

## Legacy Canary path

`--translator canary` restores the old behaviour: Canary translates the audio
directly into each target language in one pass, one SRT per language, and **no
combined file**. It is kept for comparison only; its decoder loops and
cross-sentence bleeding are what the new pipeline avoids.

## Progress, cache, and resume

Progress is recorded in `<input-dir>/subtitle_local_report.json` (includes the
selected `asr_model` and `mt_model`). Intermediate audio and ASR checkpoints are
in `<input-dir>/.local_asr/`:

- `<stem>.wav` — full extracted audio, gaps preserved so subtitles stay on the
  video timeline;
- `<stem>.asr.json` — per-chunk ASR output, keyed by file size/mtime, model,
  source language, and chunk boundaries.

The same command resumes completed chunks and skips valid SRT files. Keep the
cache until the batch is finished, and run only one local batch at a time.

To organize outputs while a batch is running:

```bash
python3 organize_subtitles.py --input-dir new_ones --watch
```

Each finished video and its subtitles move into `new_ones/<video-stem>/`; future
runs find them there and add missing languages.

## Checks and known limits

SRT syntax, ordering, cue count, and time bounds are checked before files are
saved, and cue timings are identical across languages. These checks do not prove
transcription or translation accuracy.

- A clip that ends mid-sentence leaves the trailing word in a very short cue
  (for example a 0.16 s "know"). Full videos do not have this.
- Tower occasionally drops a syllable in German (`viskales` for `viszerales`).
- `split_translation` can strand a pronoun at a cue seam; the words are correct
  for the sentence, only the cue break is imperfect.

## Modal

Modal is not used by this subtitle flow. `--backend modal` still exists for the
older Parakeet/Canary path and requires an explicit release as described in
`AGENTS.md`; it does not produce the combined `.multi.*` files.
