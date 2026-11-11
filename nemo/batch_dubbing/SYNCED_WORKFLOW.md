# Confirmed continuous dubbing workflow — 2026-11-06

The user-confirmed reference is **`continuous_local/synced_result/DUB_SYNCED.mp4`**
under `new_guys/crispr/deepseek_t4_10cues_2026-11-05/`, together with
`DUB_SYNCED.en.srt` and `DUB_SYNCED.de.srt` in that same directory.
It is neither the unadjusted `continuous_local/CONTINUOUS_GERMAN.mp4` nor
`cpu_passage_review/GERMAN_CPU.mp4`. Do not substitute either in a comparison.

The reference's ASR/translation context matters: `scope.json` records its origin
as `auto_trial/review_10sent/CRISPR-120s.mp4`, with the first ten paired Ultra/Tower
cues from source time **4.320–33.799 seconds**. Both EN and DE cue texts match
the first ten `CRISPR-120s.nemo.en/de.srt` entries after whitespace normalization.
The 29-second excerpt and rebased input SRTs were selected *after* those stages.
Re-running ASR/translation on the excerpt alone is a different source context;
exact replay requires the same validated source/translation checkpoints.

## Production order

1. Run the existing source-only Parakeet Ultra ASR. Pack its word timestamps
   into source subtitle cues. Preserve the complete input video.
2. Translate complete source passages with the existing Tower Plus 2B prompt.
   Split target text back onto the source cue correspondence. Freeze cue IDs,
   both texts and line breaks **before synthesis**. Do not translate subtitle
   fragments or independently repack English/German after synthesis.
3. Measure source anchors on CPU. Retain the original source cue context start
   and end separately from forced word timings: the approved duration target
   comes from the **source cue end**, not the last forced-aligned source word.
4. Generate continuous Qwen Base clone speech in bounded contexts, at most
   30 seconds of source timeline. Never synthesize separately per subtitle
   cue. Use the selected voice reference and preserve the raw recording.
   Unload each GPU model before loading the next; forced alignment stays on CPU.
5. Align the raw target speech. Validate exact token coverage and audio identity.
   Measure the later of its last aligned word end and its last active PCM sample
   at -55 dB. Derive `tempo = measured_end / (min(window_end, source_cue_end) - 0.10)`
   using times relative to that context's start. Accept only 0.90–1.00; otherwise
   report the failure. If reaching the preferred 100 ms margin would require
   speed-up but the native take already fits the window's 400 ms ending
   tolerance, keep 1.00 and enforce both final word/activity checks. No
   hard-coded 0.938 value and no automatic speed-up.
6. Apply FFmpeg `rubberband=tempo=VALUE:pitch=1` to the **whole context**; VALUE
   uses 12 decimal places. Preserve the full stretched take as evidence. Only
   overflow below -55 dB may be removed; active overflow blocks publication.
   Do not move sentences individually. Do not cut or shorten the picture.
7. **Align the adjusted audio again on CPU.** Do not merely scale old word
   timestamps. Require both last-word and -40 dB activity ending gaps within
   400 ms of the bounded context window. Real source silence outside that window
   remains; never stretch an entire long movie to eliminate all silence.
8. Rebuild the frozen target cue timestamps from their aligned word counts:
   80 ms lead, 120 ms tail, capped before the next cue. Give corresponding
   source-language cues those same timestamps. Preserve all cue texts/cuts.
9. Publish BOM/CRLF per-language SRTs, combined SRT/ASS, and the full source
   video with German audio and selectable EN/DE tracks (English default).
   Validate video packet identity, full duration, decoding, subtitle coverage,
   and the reported effective tempo. Reports must not label adjusted audio 1×.

Each context retains an independent absolute source start. An overrun cannot
shift later contexts and accumulate drift. Internal source/target passage
differences remain reported; forced alignment is not semantic certification.

## Selected CLI mode

The existing entry point remains `run_pipeline.py --pipeline subtitle-dubbing`
with Ultra/Tower. Its defaults are Qwen Base cloning,
`--dub-timing continuous-synced`, and `--no-demucs` (replace source audio).
`continuous` preserves the raw take without the ending adjustment;
`source-anchored` repositions whole passages. Those are distinct alternatives.

For the identical benchmark voice select the existing `voice_reference.wav`
with `--voice-reference`; ordinary new videos extract their own speaker reference.
`--tts-engine supertonic`, Qwen custom voices, and `--preserve-background` (Demucs)
remain explicit choices. Legacy-dubbing and standalone dub.py retain Supertonic
and their previous background behavior. All continuous-mode failures preserve
paired text and evidence for review; automatic rewriting is disabled.

Ordinary default run, from the ASR repository root:

```bash
qwen3-tts/.venv/bin/python run_pipeline.py \
  --input-file /path/to/video.mp4 --output-dir /path/to/results \
  --language en --target-lang de
```

### Warm forced aligner (2026-11-11)

The CPU forced aligner used to load twice per run (source alignment, then the
final finish), each time in a fresh process. `pipeline_subtitle.run` now starts
one CPU-only aligner service (`batch_dubbing/align_service.py`) before ASR, so
its ~12 s load overlaps the GPU stages, and both `dub.py` alignment stages
reuse it. Requests run the unchanged `aligned_finish` code path; on the
accepted 29 s clip, two separate processes and the service produced
byte-identical source/German word timings, SRT/ASS files and `final_dub.mp4`.
The aligner portion went from 51.8 s to 35.7 s per run, or 21.4 s when a
service is already warm.

Keep one running across runs to skip the load entirely (CPU RAM only, no GPU):

```bash
cd nemo && .venv/bin/python -m batch_dubbing.align_service serve   # leave running
.venv/bin/python -m batch_dubbing.align_service status              # or: stop
```

A service whose code differs from the files on disk is never used, and any
unavailable service falls back to the previous fresh-process stage.
`ALIGN_SERVICE=off` restores the old behaviour exactly. GPU models (Parakeet,
Tower, Qwen) still load per run: the 6 GB GPU holds one model at a time.

**GPU aligner is the default** (`--alignment-device cuda:0`; `cpu` stays
selectable and is the fallback without a GPU). FP16 weights wait in CPU RAM and
move to the GPU only for each request. RTX 2060, accepted 29 s clip, warm:
source 2.65 s + final 4.87 s on GPU vs 6.04 s + 14.17 s on CPU. Same fixed
recordings: 84/84 German words identical, final mp4/SRT/ASS byte-identical; one
English source boundary ("You"/"know") moved by one 80 ms aligner step. Fresh
full local run with Qwen: 71.7 s (CPU aligner 95.3 s, previous main 126.4 s),
GPU peak 5.9 of 6.1 GB (Qwen 5.5 GB plus the aligner's 224 MB CUDA context).

### Opt-in MAI-Voice engine

```bash
qwen3-tts/.venv/bin/python run_pipeline.py --input-file video.mp4 --output-dir out \
  --language en --target-lang de --tts-engine mai [--mai-voice Harper] [--mai-model microsoft/mai-voice-2-flash]
```

Paid OpenRouter API (~$15 per million characters, ~$1 per hour of German
speech), preset voices only (default female per language: Mia for German).
`qwen3-tts/mai_tts_worker.py` receives each context's finisher target and
deadline, asks MAI for its own pace within 0.85-1.15 when needed (at most three
requests), silences isolated trailing blips, and writes `<take>.mai.json`.
Offline check with the earlier paid take replayed through `MAI_TTS_API_URL`:
one request, `complete`, ending gap 0.093 s.

The user accepted the actual runner result on 2026-11-06: “yes now it mathces!”
The exact replay below proves the accepted output with original ASR/MT checkpoints
and newly generated speech/alignments; new media uses the same workflow, subject
to the timing/content checks rather than an unconditional quality guarantee.
The pinned Qwen Base revision is `fd4b254389122332181a7c3db7f27e918eec64e3`,
FP16, seed 20261105, continuous max sequence 1024 / token cap 600. The forced
aligner is `Qwen/Qwen3-ForcedAligner-0.6B`, revision
`c7cbfc2048c462b0d63a45797104fc9db3ad62b7`, CPU.

## Exact fixed-input regression

From `nemo`:

```bash
.venv/bin/python -m batch_dubbing.reference_regression \
  batch_dubbing/crispr_29_synced_reference.json \
  --output /tmp/crispr-synced-reference-check
```

The output directory must be new. Every input and reference is hash-bound.
The production `source_plan` / `finish` code calls the same `fit_recording`
function as the original `sync_continuous_dub.py`. Cached alignments are accepted
only for the exact regenerated recording hash. The command does not copy the
old final media or subtitle files into the result directory.

Verified reference reconstruction: `synced_workflow_regression/result/`.
All nine checks passed: PCM, encoded AAC, video packets/full duration, both SRT
files byte-for-byte, embedded track layout and subtitle packets. Measured tempo
is 0.9378623620046065; final acoustic gap is about 0.263 s. All 83 target words
and ten paired cues remain. No TTS/model call is made by this regression.

## Fresh ASR/translation/synthesis is a separate check

Fresh local validation lives in `local_synced_asr_2026-11-06/`, with its exact
command/model settings in `request.json`, live stages in `pipeline.log`, and
exit status in `execution.json`. It completed: fresh Ultra/Tower/Qwen on the
short excerpt produced nine cues and a 29.12-second raw recording. After the
native-fit edge-case correction, the same cached recording passed at 1.00 with
a 0.343-second acoustic ending gap. The first attempt and its rejection are
preserved in `initial_pipeline.log` / `initial_execution.json`; the successful
resume made no new TTS call. This is not an exact ten-cue reference match.
No Modal work is authorized for this validation.

`asr_checkpoint_synced_2026-11-06/` is a separate actual-runner reproduction
using the hash-bound original ASR/Tower cue checkpoints and fresh Qwen synthesis.
It preserves the original source context instead of silently replacing reference
text with the different short-excerpt ASR/translation. **Completed in 92.99 s:**
one new Qwen synthesis call and three new CPU alignments. The new raw WAV is
byte-identical to the original; all nine final audio/video/subtitle comparisons
against DUB_SYNCED pass. The final video is under
`result/tower_synced_reference_29__en_to_de/final_dub.mp4` in that run directory.

The fixed-input result is byte-exact. Fresh ASR punctuation/cue cuts, Tower
wording, or TTS delivery can differ. Report those differences; do not relabel a
cached reconstruction as fresh inference, replace generated text with the
reference silently, or use timing success as proof of correct translation.

## Repeat through the actual ASR pipeline

From `nemo`, with the existing local environments, model cache and reference
media available, this command prepares a new run and prints the concrete command:

```bash
.venv/bin/python -m batch_dubbing.replay_asr_checkpoint \
  batch_dubbing/crispr_29_synced_reference.json \
  --output /tmp/crispr-asr-checkpoint-replay
```

Add `--run` on the first invocation to execute fresh Qwen synthesis and three
CPU alignments through `run_pipeline.py`. It requires an idle local GPU, uses
one TTS worker, zero automatic retries, cached-only model downloads and a
720-second execution limit. Use a new output directory for every run.
It stages the original hash-verified ASR/Tower SRTs and generates their combined
subtitle inputs. It does **not** stage the old raw/final voice or final captions.
The newly rendered WAV and video are compared with the selected reference;
a mismatch returns failure. Prepared-only and GPU-busy paths are covered by
tests. The equivalent actual-runner command completed successfully above.

Final verification: 367 regression tests plus two replay tests passed. Local
GPU compute inventory and loaded Ollama model inventory were empty afterward.
