# ASR Dubber — End-to-End Local Dubbing Stack

## What This Project Solves

- **Full-stack dubbing**: diarize + translate + TTS + audio mixing, all locally.
- **Hardware-resilient & Cloud**: runs on a single GPU with aggressive cache reuse, chunked ASR, adaptive chunking, OOM retries, and staged demucs. Also fully compatible with **Modal** for offloading heavy ASR and TTS tasks to serverless A100/H100 infrastructure.
- **Reproducible**: every sub-project keeps its own `pyproject.toml` + `uv.lock`, so a `uv sync` brings it back on any machine.

> **Still on the roadmap:**
> - Add automatic lip-synchronization (phoneme alignment + face-reenactment) once the audio stack is fully synchronized as I plan to do.

| Pipeline Stage | Video Preview |
| :--- | :--- |
| **Original Source** <br> Raw video input with original audio | <video src="https://github.com/user-attachments/assets/5f3ec7c2-6f39-441e-aef7-bf8931eaae98" controls></video> |
| **Final Output** <br> Synthesized AI dub generated via NeMo | <video src="https://github.com/user-attachments/assets/c9aa4f0a-a5cb-46ef-b482-51236d73e43a" controls></video> |

<sub>Original footage credit: [YouTube – "Quantum entanglement and the illusion of time, in 79 minutes | Jim Al-Khalili: Full Interview"](https://youtu.be/8xp3Bs6nZ-Y?si=-3ooWsXV42J-gjUf).</sub>

## Table of Contents

1. [What This Project Solves](#what-this-project-solves)
2. [Pipeline at a Glance](#pipeline-at-a-glance)
3. [Technical Highlights](#technical-highlights)
4. [Subsystems & Key Scripts](#subsystems--key-scripts)
5. [Refactoring & Modular Architecture](#refactoring--modular-architecture)
6. [NeMo](#nemo)
7. [translate-gemma](#translate-gemma)
8. [qwen3-tts](#qwen3-tts)
9. [Usage](#usage)

## Pipeline at a Glance

```text
video.mp4 / audio.wav
   |
   |- (whisper/detect_language.py)--- detects spoken language when unknown
   |
   |- (nemo/nemo_diarize.py)--------------|
   |                                      |  -> (nemo/srt_fill_gaps.py) -> filled SRT
   | generates diarized SRT               |  (runs by default with 2s gap threshold)
   |                                      |
   v                                      |
video.nemo.{src}.diarize.srt               |
   |                                      |
   |- (translate-gemma/translate_diarize.py)-----|
   |                                      |
   | produces translated SRT              |
   v                                      |
video.nemo.{src}.diarize_{tgt}.srt         |
   |                                      |
   - (qwen3-tts/dub.py)-------------------|
          |
   Final dubbed video with preserved timing
```

## Technical Highlights

This project showcases advanced engineering solutions for robust, production-grade dubbing:

### **Memory-Efficient Shared TTS Architecture**
- **SharedTTSManager**: New design maintaining constant ~6GB VRAM regardless of worker count
- **Optimal CPU/GPU Overlap**: 2 workers overlapping GPU synthesis with CPU speed fitting — while one worker waits for the shared synthesizer, the other processes speed fitting
- **OOM Prevention**: Works reliably on 8GB GPUs with multiple workers via intelligent resource sharing
- **Lazy Loading**: Additional ~6GB VRAM savings when voice cloning succeeds

### **Automatic Gap Filling with Speaker Attribution**
- **Whisper Integration**: Detects and transcribes missed speech segments using configurable Whisper models
- **TitaNet Speaker Embeddings**: Advanced speaker identification using cosine similarity matching
- **Checkpoint System**: Resume capability after interruptions with progress persistence
- **Fallback Strategies**: Robust error handling with proximity-based speaker attribution backup

### **Enhanced Windowed Diarization**
- **90-Second Windows**: Prevents minority speaker collapse in long audio files
- **3-Pass Speaker Merging**: Sophisticated algorithm for speaker consolidation across windows
- **Overlap Alignment**: 30-second overlap ensures speaker consistency across window boundaries
- **Dynamic Batch Sizing**: Adaptive batch allocation (64/128/256) based on audio duration

### **Voice Cloning Enhancements**
- **Segment Concatenation**: Intelligently combines short reference segments for better cloning
- **Quality Thresholds**: Minimum duration requirements with automatic fallback strategies
- **Gender Detection**: Pitch-based speaker gender estimation for voice assignment

### **Production-Grade Robustness**
- **Subprocess Isolation**: Whisper execution in separate virtual environment prevents Python conflicts
- **File Metadata Preservation**: Maintains SRT file attributes through processing pipeline
- **Comprehensive Testing**: IPC testing utilities, quality assurance tools, and debugging frameworks
- **Error Recovery**: Graceful degradation strategies for model failures and resource constraints

Everything can be fired via `run_pipeline.py`, which orchestrates the three stages, boots Ollama when needed, automatically reuses cached artifacts, and now finishes by cleaning SRTs plus gathering every run's artifacts into `nemo/end_product/<video>__<src>_to_<tgt>`.

**Audio-Only pipelines:** The pipeline natively supports `.wav` audio files via `--input-file your_audio.wav`. Pass `--run-mode transcribe` or `--run-mode translate` (or `--skip-dub`) — dubbing requires a video source to produce a final `.mp4`, so it is automatically skipped for `.wav` inputs. The transcription and translation stages work identically whether the source is a video or a `.wav`.







## Subsystems & Key Scripts



| Folder | Purpose | Highlights |
| --- | --- | --- |
| `run_pipeline.py` | Single entry point | Auto-detects source language, spins up Ollama, supports `--skip-*` flags, enforces logging banners so you can show progress shots. Per-video workdir isolation so checkpoints never bleed between videos; SRT lookup checks both `nemo/` and `nemo/end_product/` for seamless resume after archiving; `--input-dir` / `--output-dir` for custom folder layouts. |
| `nemo/` | Diarization + transcription + gap filling | **Refactored into modular components**: `nemo_diarize.py` (main orchestration), `nemo_audio.py` (audio processing), `nemo_model.py` (model loading & transcription), `srt_fill_gaps.py` (automatic gap filling), plus helper modules. Canary/Parakeet auto-selection, VRAM-adaptive chunking, windowed diarization (90s windows), 3-pass speaker merging, automatic gap detection with Whisper transcription, TitaNet speaker embedding matching, and checkpoint-based resume capability. |
| `translate-gemma/` | Translation (Gemma via Ollama) | **Refactored into modular components**: `translate_diarize.py` (main translation logic), `translate.py` (standalone SRT translation), `clean_subs.py` (subtitle cleaning). Chunked SRT translation with strict `[idx]` preservation, Docker-friendly Ollama client, low-temperature prompts for subtitle-safe formatting. |
| `qwen3-tts/` | Dubbing | **Refactored into modular components**: `dub.py` (main orchestration), `dub_audio.py` (audio processing), `dub_srt.py` (SRT parsing & voice assignment), `qwen_tts_worker.py` (TTS worker). Demucs-based vocal separation, clone-vs-custom fallback ladder, per-segment checkpoints, silence synthesis to keep alignment tight, final mix either with preserved background music or direct replacement. Per-video workdir isolation keeps checkpoints segregated. |
| `whisper/` | Language detection | `detect_language.py` (30s Whisper probe when no diarized SRT exists yet; called automatically from `run_pipeline.py`). |


## Refactoring & Modular Architecture

The codebase has been refactored from large monolithic files into smaller, focused modules to improve maintainability and readability:

### nemo/ Module Structure
- **`nemo_diarize.py`** - Main orchestration and diarization pipeline
- **`nemo_audio.py`** - Audio processing utilities (extraction, chunking, SRT generation)
- **`nemo_model.py`** - Model loading and transcription logic

### translate-gemma/ Module Structure
- **`translate_diarize.py`** - Main translation engine with Ollama integration
- **`translate.py`** - Standalone SRT translation for non-diarized files
- **`clean_subs.py`** - Subtitle cleaning, formatting, and auto-shortening (prevents Windows MAX_PATH errors)

### qwen3-tts/ Module Structure
- **`dub.py`** - Main dubbing orchestration and pipeline coordination
- **`dub_audio.py`** - Audio processing, Demucs separation, and mixing
- **`dub_srt.py`** - SRT parsing, voice assignment, and language mapping
- **`qwen_tts_worker.py`** - Dedicated TTS synthesis worker

### run_pipeline.py Refactoring
The main orchestrator has been streamlined while maintaining full compatibility with all existing CLI flags and functionality.

**Benefits of the refactoring:**
- **Improved maintainability** - Smaller files are easier to understand and modify
- **Better testability** - Individual components can be tested in isolation
- **Enhanced reusability** - Utilities can be imported and used across different contexts
- **Clearer separation of concerns** - Each module has a focused responsibility
- **Preserved compatibility** - All existing CLI interfaces and workflows remain unchanged





## NeMo



### Supported ASR Models & Auto-Selection

The pipeline automatically selects the best ASR model based on the source language:
* **Parakeet v3** (`nvidia/parakeet-tdt-0.6b-v3`): The default model for all languages in local execution. Extremely fast with word-level timestamps.
* **Parakeet v2** (`nvidia/parakeet-tdt-0.6b-v2`): English-only alternative to v3. Available if preferred via `--nemo-model parakeet-v2`.
* **Qwen3-ASR 1.7B** (`Qwen/Qwen3-ASR-1.7B`): The default model for non-EN languages in Docker/remote execution (overridden via `NEMO_MODEL_MULTI=qwen3-asr`). Supports 30+ languages with higher accuracy than Parakeet for non-English content.
* **Canary 1B** (`nvidia/canary-1b-v2`): Optional fallback that supports direct AST translation (EN/DE/FR/ES). One of the best quality models for non EN languages. According to https://huggingface.co/spaces/hf-audio/open_asr_leaderboard

You can override these defaults at runtime using the `--nemo-model` flag in the pipeline.

```bash
# Force Qwen3-ASR 1.7B regardless of language
uv run python run_pipeline.py --target-lang fr --nemo-model qwen3-asr

# Force Canary 1B
uv run python run_pipeline.py --target-lang fr --nemo-model canary
```

### Advanced VRAM & Memory Management

Designed for local execution on consumer-grade hardware, the engine implements several strategies to prevent crashes and optimize GPU utilization:

* **Adaptive Chunking with OOM Retries:** The `_estimate_chunk_sec()` function measures available VRAM, subtracts a safety reserve, and converts the remaining capacity into a time-based chunk target tailored to the specific model's footprint.

* **Intelligent Recovery:** If a `torch.cuda.OutOfMemoryError` occurs, `_transcribe_chunked()` automatically halves the chunk size and cleans up temporary WAV files before retrying, ensuring long-form videos finish even on limited hardware. For Parakeet v3 specifically, CUDA graph capture is guarded to prevent XID 31 GPU crashes on consumer drivers.

* **Layer-by-Layer GPU Loading:** To avoid initialization spikes, `_load_model()` streams weights to the CUDA device module-by-module rather than all at once.

* **Memory-Efficient Precision:** Supports `fp16` and `bf16` to maximize throughput, while using `PYTORCH_ALLOC_CONF` to manage expandable segments and prevent memory fragmentation.



### Sophisticated Processing Pipeline

The script handles the edge cases of AI transcription that standard wrappers often ignore:

* **Shadow-Safe Imports:** A custom `_import_nemo_asr()` handler resolves path conflicts, allowing the script to be named `nemo.py` without shadowing the official NVIDIA library.

* **Fused Kernel Optimization:** When sufficient VRAM is detected after loading, the script enables `torch.compile(mode="reduce-overhead")` to accelerate inference via fused kernels.

* **Overlap Deduplication:** The engine automatically detects and removes duplicate words generated at the boundaries of overlapping audio chunks.

* **Integrated Diarization:** Uses `titanet_large` to identify distinct speakers, mapping "[Speaker N]" labels directly onto the transcription segments.



### Production-Ready Subtitle Logic

Output is formatted specifically for media players, avoiding unreadable "walls of text":

* **Smart Segmentation:** Text is split into subtitle-compliant blocks based on word count (max 10), duration (max 5s), and line length (max 80 chars).

* **Linguistic Awareness:** The segmenter prioritizes splits at natural punctuation boundaries (periods, question marks) to maintain readability.

* **Canary Integration:** Fully supports NVIDIA Canary for high-accuracy multilingual ASR and direct translation into English.





























## translate-gemma


### LLM-Powered Diarization Preservation

Unlike standard translation tools that lose speaker context, this script utilizes a custom-prompted LLM (Gemma via Ollama) to translate dialogue while maintaining metadata:

* **Speaker Tag Detachment:** The engine uses regex to strip `[Speaker N]` tags before translation, storing them in a temporary map to prevent the model from translating or hallucinating speaker names.

* **Tag Re-attachment:** Once the core text is translated, the script precisely re-injects the original speaker tags back into the subtitle block, ensuring the downstream dubbing pipeline knows exactly which voice to use.

* **One-Shot Prompting Strategy:** Implements a rigorous one-shot instruction set that provides the model with concrete examples of index preservation and line-count requirements to maximize reliability.



### Structural Integrity & Parsing

To handle the limitations of smaller local LLMs (like `translategemma:4b`), the pipeline employs several "guardrail" techniques:

* **Global Sentence Grouping:** Before chunking, the pipeline reassembles sentences that were split across chunk boundaries in the NeMo SRT. This prevents broken translations where the first half of a sentence is in one chunk and the second half in the next — the most common source of awkward phrasing in long videos.

* **Chunked Processing:** Subtitles are processed in configurable batches (default 15 lines, set via `CHUNK_SIZE` env var) to reduce model confusion and prevent the "forgetting" of indices or merging of lines common in larger batches.

* **Index-Based Verification:** Every line is tagged with a unique `[index]`. Post-translation, the script uses robust regex to parse these indices, mapping translated text back to the specific `pysrt` object to ensure no subtitles are skipped or misaligned.

* **Newline Pipe-Encoding:** To prevent the LLM from breaking subtitle formatting with unwanted line breaks, the script flattens multi-line subtitles using a `|` (pipe) character during translation and restores them during reassembly.

### Automation & Reliability

* **Auto-Language Detection:** The script automatically parses the source language code (e.g., `.de.`) directly from the filename of the NeMo-generated SRT to configure the translation prompt.

* **Deterministic Configuration:** Sets `temperature: 0.1` and `num_ctx: 2048` to balance translation creativity with strict adherence to the input format.

* **Garbage Filtering:** Includes automated post-processing to strip common LLM artifacts like `<|endoftext|>` and unexpected whitespace before saving the final file.

* **Filename Auto-Shortening:** When cleaning and moving files to the final `end_product` directory, `clean_subs.py` automatically detects filenames exceeding 60 characters and shortens them to `output.*` (preserving original extensions and suffixes). This strictly prevents Windows `MAX_PATH` limitations that would otherwise cause files to appear "corrupted" or empty when accessed from a host OS.

#### Standalone `translate.py`

When you just need to turn an existing SRT into another language or you have srt file already and it has no Speaker tags (no NeMo input required), run `uv run python translate-gemma/translate.py -i input.srt --src fr --tgt de`. It reuses the same micro-chunk prompt logic as the diarized translator but lets you point at any cleaned subtitle file. However, the script auto-starts Ollama if needed, so make sure the `ollama` binary/Docker image is installed and the translation model is already pulled (default `translategemma:4b`, override with `TRANSLATE_MODEL` env var), otherwise the first run will fail before translating.






















































## qwen3-tts



### Intelligent Voice Cloning & Assignment

The dubbing engine utilizes a dual-mode strategy to ensure every speaker in the video has a distinct, high-quality voice:

* **Zero-Shot Voice Cloning:** In `clone` mode, the script automatically extracts the longest clean audio segment for each identified speaker to use as a reference for the Qwen3-TTS model.

* **Dynamic Speaker Mapping:** For non-cloning tasks, `build_voice_map()` automatically rotates through gender-specific voice pools to assign unique identities to every "Speaker N" tag found in the SRT.

* **Fallback Resilience:** The pipeline features a tiered failure system; if a voice clone fails to synthesize, it automatically falls back to a high-quality custom voice for that segment. The fallback is gender-aware: pitch detection is used to classify the speaker's gender and select a matching male or female preset voice, so the character still sounds natural.



### 🎶 Background Music Preservation (Demucs Integration)

To maintain the cinematic quality of the original video, the script handles complex audio separation:

* **Neural Stem Splitting:** Integrates Facebook’s `demucs` (htdemucs) to isolate vocals from background music and environmental noise.

* **Intelligent Mixing:** The final stage uses FFmpeg’s `filter_complex` to re-layer the new dubbed track (at 150% volume) over the original isolated background track (at 40% volume), preserving the "feel" of the original media.

* **No-Demucs Fast Mode:** Includes a `--no-demucs` toggle for rapid prototyping, which bypasses separation and replaces the audio track entirely.



### ⏱️ Temporal Alignment & Speed Fitting

Ensuring the dubbed audio matches the on-screen action is handled through automated rhythmic adjustment:

* **Dynamic Speed-Fitting:** The `speed_fit()` function compares TTS duration to the original SRT timestamps. If the synthesized speech is too long, it applies an `atempo` filter (capped at 1.35x by default, customizable via `--max-speed`) to prevent "chipmunk" effects while staying in sync.

* **Sentence Merging (Merge Gap):** Consecutive segments from the same speaker with short pauses (default ≤ 1.0s) can be seamlessly merged together before TTS synthesis using the `--merge-gap` flag. This significantly improves the natural flow and rhythm of the synthetic voice over multiple lines of dialogue.

* **Precision Padding:** For shorter segments, the engine pads the tail with silence to maintain the natural cadence of the dialogue without stretching the audio.

* **Frame-Accurate Stitching:** Automatically generates `anullsrc` silence gaps between clips to ensure every line starts at the exact millisecond defined in the source SRT.

### TTS Architecture & Worker Management
* **Checkpointed Progress:** The pipeline saves a `checkpoint.json` after every successfully synthesized segment. If the process is interrupted or the GPU crashes, it resumes exactly where it left off.

* **Isolated Worker Execution:** TTS synthesis is decoupled into a dedicated `qwen_tts_worker.py`. This allows the main script to manage the heavy FFmpeg/Demucs logic while the worker handles specialized `torch` environments and `bfloat16` model loading.

* **Dubbed SRT Output:** In addition to the final `final_dub.mp4`, the dubbing stage now also saves a clean `final_dub.srt` alongside it in the `end_product` directory - useful for adding subtitles to the dubbed video in a media player.

### Workspace Management
* **Checkpoint Resilience:** The checkpoint system automatically handles reprocessing scenarios. If you rerun with different trim/target language, segment indices will change and old checkpoints won't interfere - the system only counts segments whose audio files actually exist and match current indices.



### ⚙️ Scalable Workflow & Robustness
* **Automatic Video Trimming:** To ensure a clean finish, the script automatically trims the final video output to match the end of the last subtitle segment.































## Usage





### Environment Setup



Each subfolder is a standalone `uv` project.



```bash
uv sync --project nemo
uv sync --project translate-gemma
uv sync --project qwen3-tts
uv sync --project whisper
```





### Docker & Remote PC Usage

For running the full pipeline on a remote workstation or headless Linux machine, a `docker-compose.yml` is provided. This spins up the LLM inference server (Ollama) and the worker pipeline with full GPU passthrough.

```bash
# 1. Place the video file in the data folder
cp your_video.mp4 data/input/

# 2. Run the pipeline container using Docker Compose
docker compose run pipeline --target-lang fr

# 2. Or run the pipeline without docker
uv run python run_pipeline.py --target-lang fr --precision fp16 --translate-model 12b --fill-gaps-model turbo


# 3. Running with Runtime Overrides (Environment Variables)
NEMO_MODEL_MULTI=qwen3-asr TRANSLATE_MODEL=translategemma:12b docker compose run pipeline --target-lang fr
```

The Docker container accepts all the same CLI arguments as the local script. You can also configure the environment directly via modifying `docker-compose.yml` properties or passing them dynamically at runtime (useful when you don't want to modify the YAML file):

| Environment Variable | Description |
| --- | --- |
| `TARGET_LANG` | Required target language code (e.g., `fr`, `es`). Alternatively, pass via CLI `--target-lang`. |
| `NEMO_MODEL_EN` | NeMo ASR model to use for English (default: `parakeet-v3`). |
| `NEMO_MODEL_MULTI` | NeMo ASR model to use for all other languages (default: `qwen3-asr` for Docker). |
| `TRANSLATE_MODEL` | Translation model to run via Ollama (default: `translategemma:12b` for Docker). |
| `PRECISION` | GPU precision for NeMo (default: `fp16`). |
| `CHUNK_SIZE` | Subtitle lines to pass per Ollama translation call (default: `40`). |
| `TTS_WORKERS` | Number of parallel TTS synthesis workers (default: 2 for optimal CPU/GPU overlap with SharedTTSManager). |
| `TTS_DEVICES` | Comma-separated CUDA device indices for TTS workers, e.g. `0,1` (default: all visible GPUs). |

> **Note:** For configuring the target language, always use the `--target-lang` CLI flag as it is the most reliable method. Environment variable overrides via YAML or inline runtime are best used for hardware or model tuning.

### Quickstart (full pipeline)



```bash
python run_pipeline.py --target-lang fr
```



Key flags:
- `--language de/fr/en` — force source language (skips Whisper detection).
- `--trim 30` — only process the first minute for rapid iteration.
- `--run-mode transcribe|translate|full` — convenience presets that toggle the skip flags for you (e.g., `transcribe` = NeMo only, `translate` = NeMo + Gemma).
- `--skip-nemo / --skip-translate / --skip-dub` — resume partially completed runs when you prefer explicit control over stages.
- `--input-dir` / `--output-dir` — override default `nemo/` input and `nemo/end_product/` output folders.
- `--whisper-model tiny|base|small|medium|large-v3|turbo` — choose Whisper model for language detection (default: `medium`).
- `--qwen-mode custom` — bypass voice cloning.
- `--no-demucs` — speed up dubbing when background music doesn’t matter.

Common CLI snippets:

```bash
# Full run with defaults (auto language detection, cleans + archives outputs)
python run_pipeline.py --target-lang fr

# Audio-only run (skips dubbing automatically)
python run_pipeline.py --target-lang fr --input-file your_audio.wav

# Quick transcription-only dry run (first 60s)
python run_pipeline.py --target-lang fr --trim 60 --run-mode transcribe

# Translate-only pass when NeMo output already exists and source language is known
python run_pipeline.py --target-lang fr --language de --run-mode translate
```



### Argument Reference & Flag Mapping



#### run_pipeline.py (master orchestrator)

| Flag | Required | Default | Purpose |
| --- | --- | --- | --- |
| `--target-lang` | ✅ | — | Language of the final dub (e.g., `fr`, `es`, `en`). |
| `--input-file` |  | auto-detect from `--input-dir` | Explicit path to a `.mp4` or `.wav` file. `.wav` inputs automatically skip dubbing (requires a video for the dub stage). |
| `--language` |  | auto (Whisper / existing SRT) | Force source language and skip Whisper detection. |
| `--trim SEC` |  | `0` (full) | Only process the first N seconds — great for quick experiments. |
| `--run-mode {transcribe,translate,full}` |  | `full` | Convenience presets that flip the `--skip-*` flags for you. |
| `--skip-nemo / --skip-translate / --skip-dub` |  | `False` | Manually resume from any stage when artifacts already exist. |
| `--input-dir / --output-dir` |  | `nemo/` / `nemo/end_product/` | Override where videos are pulled from and where finished runs land. |
| `--whisper-model` |  | `medium` | Whisper size for auto language ID (`tiny`, `base`, `small`, `medium`, `large-v3`, `turbo`). |
| `--qwen-mode {clone,custom}` |  | `clone` | Switch between zero-shot cloning or fixed speaker presets. |
| `--max-speed SPEED` |  | `1.35` | Maximum TTS speed-up multiplier before capping. |
| `--min-speed SPEED` |  | `0.65` | Minimum TTS slow-down multiplier (applied when speech is too short relative to the subtitle window). |
| `--merge-gap SEC` |  | `1.0` | Merge consecutive same-speaker segments with gap ≤ N seconds for more natural TTS flow (set `0` to disable). |
| `--merge-max-dur SEC` |  | `10.0` | Hard cap on the duration of a merged segment. Prevents runaway merges from creating overly long TTS chunks that drift out of sync. |
| `--no-demucs` |  | `False` | Skip background music separation for faster dubbing. |
| `--fill-gaps` |  | `False` | Enable automatic gap filling in SRT files using Whisper transcription (default 2s threshold). |
| `--fill-gaps-model MODEL` |  | `base` | Whisper model for gap filling (`base`, `large-v3`, `turbo`). |
| `--no-embeddings` |  | `False` | Disable speaker embedding matching, use proximity-based speaker attribution instead. |
| `--embedding-threshold N` |  | `0.7` | Similarity threshold for speaker embedding matching (0.0-1.0). |
| `--precision {fp32,fp16,bf16}` |  | `bf16` | ASR precision passed straight through to `nemo.py`. Use `fp16` on older GPUs or `fp32` for max accuracy. |
| `--nemo-model MODEL` |  | auto | Force a specific NeMo checkpoint (e.g., `nvidia/parakeet-tdt-1.1b`). |
| `--chunk-override SEC` |  | auto | Lock NeMo chunk size if VRAM auto-detect over/underestimates (auto sizing caps at 600 s — e.g., pass `120` for fixed 2‑minute chunks). |
| `--reserve-gb GB` |  | `1.5` | VRAM the chunk estimator should keep free. Increase when hitting OOM. |
| `--safety-factor F` |  | `0.85` | Multiplier applied to the detected free VRAM before computing chunk length. |

> **NeMo tuning flags** (`--precision`, `--nemo-model`, `--chunk-override`, `--reserve-gb`, `--safety-factor`) are forwarded verbatim to `nemo/nemo.py`. If you don’t specify them, `run_pipeline.py` leaves them unset so NeMo falls back to its own defaults.

Example combos:

```bash
# Old GPU that struggles with bf16 auto-chunking
uv run python run_pipeline.py --target-lang fr --precision fp16 --reserve-gb 3.0

# Force 2-minute chunks, custom model, 45-second trim preview
uv run python run_pipeline.py \
  --target-lang fr --trim 45 --chunk-override 120 --nemo-model nvidia/parakeet-tdt-1.1b
```



#### nemo/nemo.py (ASR + diarization)

Running `uv run --project nemo python nemo.py --help` lists every low-level flag. For quick reference:

| Argument | Default | Description |
| --- | --- | --- |
| `video` (positional) | auto-detect newest pending | Optional explicit video path. If omitted, NeMo scans the current folder for files without matching `.nemo.<lang>.srt`. |
| `--all` | `False` | Process every pending video instead of just the newest one. |
| `--language` | `en` | Source language (drives model auto-selection and SRT suffixes). |
| `--nemo-model` | `nvidia/parakeet-tdt-0.6b-v3` (all languages locally) | Override the exact NeMo checkpoint to download/run. |
| `--precision {fp32,fp16,bf16}` | `bf16` | Controls CUDA dtype; `fp32` is safest, `fp16/bf16` are faster and lighter. |
| `--translate` | `False` | Canary-only: translate to English directly inside NeMo (pipeline normally handles translation via Gemma). |
| `--diarize` | `False` | Attach `[Speaker N]` labels to the generated SRT. `run_pipeline.py` always enables this so downstream TTS can map voices. |
| `--trim SEC` | `0` | Only decode the first N seconds (mirrors the pipeline’s `--trim`). |
| `--safety-factor` | `0.85` | VRAM safety multiplier used by `_estimate_chunk_sec()`. |
| `--reserve-gb` | `1.5` | VRAM (in GB) to keep unused before computing chunk length. |
| `--chunk-override SEC` | auto | Hard-code the chunk size (seconds). Useful when auto-estimates are still too large/small. |

Because `run_pipeline.py` invokes `nemo_diarize.py` under the hood, you can tweak NeMo without leaving the one-command workflow. For example, `--precision fp32 --chunk-override 90` on the pipeline CLI becomes `... nemo_diarize.py ... --precision fp32 --chunk-override 90` during Step 1.



### Manual Stage Control



1. **NeMo diarization**

   ```bash
   uv run --project nemo python nemo.py video.mp4 --language de --diarize
   ```

   Output → `video.nemo.de.diarize.srt`



2. **Gemma translation**

   ```bash
   uv run --project translate-gemma python translate_diarize.py
   ```

   Looks for `TARGET_LANG_CODE` env var (`fr`, `es`, ...). Produces `*.diarize_fr.srt`.



3. **Qwen dubbing**

   ```bash
   uv run --project qwen3-tts python dub.py video.mp4 video.nemo.de.diarize_fr.srt --language fr --qwen-mode clone
   ```

