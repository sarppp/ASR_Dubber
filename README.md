# ASR Dubber — End-to-End Local Dubbing Stack

## What This Project Solves

- **Full-stack dubbing**: diarize + translate + TTS + audio mixing, all locally.
- **Hardware-resilient & Cloud**: runs on a single GPU with aggressive cache reuse, chunked ASR, adaptive chunking, OOM retries, and staged demucs. Also fully compatible with **Modal** for offloading heavy ASR and TTS tasks to serverless A100/H100 infrastructure.
- **Reproducible**: every sub-project keeps its own `pyproject.toml` + `uv.lock`, so a `uv sync` brings it back on any machine.

> **Still on the roadmap:** add automatic lip-synchronization (phoneme alignment + face-reenactment) once the audio stack is fully synchronized as I plan to do.

| Pipeline Stage | Video Preview |
| :--- | :--- |
| **Original Source** <br> Raw video input with original audio | <video src="https://github.com/user-attachments/assets/59d5d076-db7f-4179-be0b-0f6656de69c4" controls></video> |
| **Final Output** <br> Synthesized AI dub generated via NeMo | <video src="https://github.com/user-attachments/assets/473d36e3-b7f6-4993-ab5d-311ab7112bdc" controls></video> |

<sub>Original footage credit: [YouTube – "Impostor-Syndrom: Warum Anna glaubt, nichts zu können I 37 Grad"](https://youtu.be/ElWPwt-ecJc?si=E39gR8giOeXs5MHA).</sub>

## Table of Contents

1. [What This Project Solves](#what-this-project-solves)
2. [Pipeline at a Glance](#pipeline-at-a-glance)
3. [Subsystems & Key Scripts](#subsystems--key-scripts)
4. [NeMo](#nemo)
5. [translate-gemma](#translate-gemma)
6. [qwen3-tts](#qwen3-tts)
7. [Usage](#usage)

## Pipeline at a Glance

```text
video.mp4
   │
   ├─(whisper/detect_language.py)─── detects spoken language when unknown
   │
   ├─(nemo/nemo.py --diarize)──────────────┐
   │                                       │
   │ generates diarized SRT                │
   ▼                                       │
video.nemo.{src}.diarize.srt               │
   │                                       │
   ├─(translate-gemma/translate*.py)───────┤
   │                                       │
   │ produces translated SRT               │
   ▼                                       │
video.nemo.{src}.diarize_{tgt}.srt         │
   │                                       │
   └─(qwen3-tts/dub.py)────────────────────┘
          │
          └─ demucs (optional) + Qwen clone/custom voices + ffmpeg stitching
         
Result → `qwen3-tts/output/dub/output/final_dub.mp4`
```



Everything can be fired via `run_pipeline.py`, which orchestrates the three stages, boots Ollama when needed, and automatically reuses cached artifacts.









## Subsystems & Key Scripts



| Folder | Purpose | Highlights |
| --- | --- | --- |
| `run_pipeline.py` | Single entry point | Auto-detects source language, spins up Ollama, supports `--skip--` flags, enforces logging banners so you can show progress shots. |
| `nemo/nemo.py` | Diarization + transcription | Canary/Parakeet auto-selection, VRAM-adaptive chunking, diarization via `ClusteringDiarizer`, custom patches in `canary_patch.py` to bypass canary EOS assertions and force correct manifest langs. |
| `translate-gemma/translate*.py` | Translation (Gemma via Ollama) | Chunked SRT translation with strict `[idx]` preservation, Docker-friendly Ollama client, low-temperature prompts for subtitle-safe formatting. |
| `qwen3-tts/dub.py` | Dubbing | Demucs-based vocal separation, clone-vs-custom fallback ladder, per-segment checkpoints, silence synthesis to keep alignment tight, final mix either with preserved background music or direct replacement. |
| `whisper/detect_language.py` | Language detection | 30s Whisper probe when no diarized SRT exists yet; called automatically from `run_pipeline.py`. |





## NeMo



### Advanced VRAM & Memory Management

Designed for local execution on consumer-grade hardware, the engine implements several strategies to prevent crashes and optimize GPU utilization:

* **Adaptive Chunking with OOM Retries:** The `_estimate_chunk_sec()` function measures available VRAM, subtracts a safety reserve, and converts the remaining capacity into a time-based chunk target tailored to the specific model's footprint.

* **Intelligent Recovery:** If a `torch.cuda.OutOfMemoryError` occurs, `_transcribe_chunked()` automatically halves the chunk size and cleans up temporary WAV files before retrying, ensuring long-form videos finish even on limited hardware.

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

* **Micro-Chunking Logic:** Subtitles are processed in small batches (5 lines per chunk) to reduce model confusion and prevent the "forgetting" of indices or merging of lines common in larger batches.

* **Index-Based Verification:** Every line is tagged with a unique `[index]`. Post-translation, the script uses robust regex to parse these indices, mapping translated text back to the specific `pysrt` object to ensure no subtitles are skipped or misaligned.

* **Newline Pipe-Encoding:** To prevent the LLM from breaking subtitle formatting with unwanted line breaks, the script flattens multi-line subtitles using a `|` (pipe) character during translation and restores them during reassembly.



### Automation & Reliability

* **Auto-Language Detection:** The script automatically parses the source language code (e.g., `.de.`) directly from the filename of the NeMo-generated SRT to configure the translation prompt.

* **Deterministic Configuration:** Sets `temperature: 0.1` and increases `num_ctx` to 4096 to balance translation creativity with strict adherence to the input format.

* **Garbage Filtering:** Includes automated post-processing to strip common LLM artifacts like `<|endoftext|>` and unexpected whitespace before saving the final file.

























































## qwen3-tts



### Intelligent Voice Cloning & Assignment

The dubbing engine utilizes a dual-mode strategy to ensure every speaker in the video has a distinct, high-quality voice:

* **Zero-Shot Voice Cloning:** In `clone` mode, the script automatically extracts the longest clean audio segment for each identified speaker to use as a reference for the Qwen3-TTS model.

* **Dynamic Speaker Mapping:** For non-cloning tasks, `build_voice_map()` automatically rotates through gender-specific voice pools to assign unique identities to every "Speaker N" tag found in the SRT.

* **Fallback Resilience:** The pipeline features a tiered failure system; if a voice clone fails to synthesize, it automatically falls back to a high-quality custom voice for that segment to ensure the render finishes.



### 🎶 Background Music Preservation (Demucs Integration)

To maintain the cinematic quality of the original video, the script handles complex audio separation:

* **Neural Stem Splitting:** Integrates Facebook’s `demucs` (htdemucs) to isolate vocals from background music and environmental noise.

* **Intelligent Mixing:** The final stage uses FFmpeg’s `filter_complex` to re-layer the new dubbed track (at 150% volume) over the original isolated background track (at 40% volume), preserving the "feel" of the original media.

* **No-Demucs Fast Mode:** Includes a `--no-demucs` toggle for rapid prototyping, which bypasses separation and replaces the audio track entirely.



### ⏱️ Temporal Alignment & Speed Fitting

Ensuring the dubbed audio matches the on-screen action is handled through automated rhythmic adjustment:

* **Dynamic Speed-Fitting:** The `speed_fit()` function compares TTS duration to the original SRT timestamps. If the synthesized speech is too long, it applies an `atempo` filter (capped at 1.35x) to prevent "chipmunk" effects while staying in sync.

* **Precision Padding:** For shorter segments, the engine pads the tail with silence to maintain the natural cadence of the dialogue without stretching the audio.

* **Frame-Accurate Stitching:** Automatically generates `anullsrc` silence gaps between clips to ensure every line starts at the exact millisecond defined in the source SRT.



### ⚙️ Scalable Workflow & Robustness

* **Checkpointed Progress:** The pipeline saves a `checkpoint.json` after every successfully synthesized segment. If the process is interrupted or the GPU crashes, it resumes exactly where it left off.

* **Isolated Worker Execution:** TTS synthesis is decoupled into a dedicated `qwen_tts_worker.py`. This allows the main script to manage the heavy FFmpeg/Demucs logic while the worker handles specialized `torch` environments and `bfloat16` model loading.

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





### Quickstart (full pipeline)



```bash
python run_pipeline.py --target-lang fr
```



Key flags:
- `--language de/fr/en` — force source language (skips Whisper detection).
- `--trim 30` — only process the first minute for rapid iteration.
- `--skip-nemo / --skip-translate / --skip-dub` — resume partially completed runs.
- `--qwen-mode custom` — bypass voice cloning.
- `--no-demucs` — speed up dubbing when background music doesn’t matter.



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

