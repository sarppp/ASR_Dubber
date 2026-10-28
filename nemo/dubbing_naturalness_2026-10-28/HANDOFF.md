# Dubbing naturalness investigation — handoff

Date: 2026-10-28. Working directory: `/home/sarpk/uv_tests/ASR/nemo`.
Repository root: `/home/sarpk/uv_tests/ASR`.

## Read this first

The user has struggled for months with dubbing that alternates between rushed,
slow, and disconnected delivery. They want extensive tests and a defensible fix,
with special attention to not destabilizing WSL2. Their GPU has 6 GB VRAM.
They explicitly authorized parallel agents, short audio comparisons, continuous
ten-minute comparisons of original English versus German dubbing with both
subtitle tracks, and subsequently the exact bounded synthesis experiment below.

The investigation establishes several implementation and text defects. It does
**not** establish that the candidate now sounds natural. Separate code correctness,
timing feasibility, intelligibility, translation fidelity, and human preference.

The original dubbing workers remain unchanged. The candidate timing code is
isolated and is not the default production path. Existing videos, subtitles, and
generated audio have not been overwritten.

## Current delivery and execution state

- Source branch: `modal-control`, created from local `main` at `9a3624e`.
- Source commit: `afcaaddcef8a7d89659eb159b6af1ec36424a08d` —
  `Add bounded Modal dubbing workflows and naturalness investigation`.
- Push to `origin/modal-control` succeeded; a separate `git ls-remote` confirmed
  that exact commit. Remote: `git@github.com:sarppp/ASR_Dubber.git`.
- That commit contains all 47 pending source/configuration/documentation files,
  including changes that predated this investigation, plus the small frozen test
  fixture and telemetry helper that were initially hidden by ignore rules.
- This handoff is a subsequent documentation change. Do not confuse the source
  commit above with the eventual documentation commit.
- Generated evidence/media and controller state remain local and ignored. A new
  clone will have the scripts and frozen experiment input, not the original media,
  cached models, local virtual environments, or generated review results.
- Released plan: `mp-3b5d18844ffb730aaf868ecc83517222`.
- Controller run: `2026-10-28-a973f4b68d86`.
- Exact user release recorded: `Say “release it” to run that comparison release`.
- Final run outcome: **succeeded**. One smoke call plus all 22 workload calls
  completed. Both Modal apps are stopped and the after-run inventory contains no
  running containers. The full receipt and new findings are below. Do not rerun
  this consumed plan or request the already-given release again.

## Authorization and resource boundaries

Read `../AGENTS.md` before continuing. Model inference, including Supertonic CPU,
uses Modal through the controller. Local inference is not authorized by the
general request to investigate; it needs an explicit override of that rule.
The user's synthesis release covers only the exact experiment below. It does not
authorize a model sweep, retranslation, a new ten-minute synthesis run, local
GPU inference, or retries after failure.

GPU work is T4-only unless separately authorized. This experiment uses **no GPU**.
Do not run `.map()`, `.starmap()`, or concurrent submissions without the approved
global cap. A prior incident in this repository autoscaled to 100 containers.
The current trial uses one CPU container at a time and sequential calls.

Optional subtitle/media operations need the user's choice. Their earlier approval
explicitly covered the ten speed variants and the English/German ten-minute review.
Do not infer approval for background separation, new translations, burn-in, or
wholesale repackaging. Read-only inspection and lightweight code tests are allowed.

## Evidence inventory

Paths below are relative to this directory unless prefixed with `../`.

| Artifact | Purpose |
|---|---|
| `review.html` | Local interactive review; selectable Mind Control/Bill, English/German switch at same time, captions, tempo overview, speed samples |
| `review_data.json` | First 600 seconds of both videos; actual EN/DE SRT cues, reconstructed synthesis spans, distributions and 30-second bins |
| `metrics.json` | Five recent dub reports, 906 tempo rates, 96 old benchmark WAV headers, source/JSON provenance |
| `audio_checks.json` | All 18 serial FFmpeg jobs, commands, bounds, durations, hashes, resource observations and acoustic intervals |
| `speed_comparisons/*.wav` | Two existing German voices at 1.00, 0.65, 1.15, 1.50 and 2.00 times original tempo |
| `text_findings.json` | Five manually inspected translation problems with surrounding English transcript context |
| `policy_sweep.json` | Counterfactual timing feasibility at six speed ceilings; current grouping held fixed |
| `tests.log`, `validation.json` | Thirty batch-dubbing tests and bounded local runner information |
| `experiment_inputs.json` | Version-controlled, frozen German cues and English context for new synthesis |
| `release_plan.json` | Full prepared controller plan, source hashes and exact scope |
| `native_grouping_trial/{smoke,full}/` | Outputs of the released experiment, if completed |
| `native_trial_verification.json` | All 23 raw WAVs validated; per-arm measurements and SHA256 hashes |
| `cloud_during_trial.json`, `cloud_after_trial.json` | One owned container during the run; no containers after completion |
| `billing_trial_apps.json` | Exact billing rows for the two owned apps, separated from unrelated account spending |
| `../modal_control/.state/runs/2026-10-28-a973f4b68d86/` | Authoritative run manifest, events, stage stdout/stderr, source snapshots and artifact receipts |

The HTML's JavaScript syntax and local links were checked. Interactive playback
was **not browser-verified** because no browser automation runtime was available.
The player references the existing full video files but limits the review to 600
seconds; this is not a newly encoded standalone ten-minute movie.

## Findings: recent five-video pipeline

Input/output metadata are under `../modal_control/batch_inputs/video_XX.*`.
The `.gpu_full.json` files hold English cues, German/French translations and
speaker information. `.cpu_v2_full.json` files hold fitted speech reports.

Across all **906** synthesized segments:

- Median post-synthesis tempo multiplier: **1.19155**.
- 95th percentile: **1.69519**; maximum: **2.24943**.
- **367** segments exceed 1.25×, **249** exceed 1.35×,
  **128** exceed 1.5×, and **7** exceed 2×.
- **183 within-video adjacent transitions** change tempo by more than 0.35.
  The top-level aggregate in `metrics.json` concatenates videos; use the per-video
  sum for a rigorously within-video transition count.

| Video | Segments | Median tempo | Above 1.5× | Above 2× | Maximum |
|---|---:|---:|---:|---:|---:|
| Bill Ackman, index 00 | 458 | 1.194× | 92 | 5 | 2.249× |
| CRISPR, index 01 | 215 | 1.111× | 8 | 1 | 2.068× |
| Restart Humanity, index 02 | 54 | 1.118× | 1 | 0 | 1.596× |
| Political Systems, index 03 | 61 | 1.192× | 4 | 0 | 1.538× |
| Mind Control, index 04 | 118 | 1.340× | 23 | 1 | 2.133× |

These are **audio tempo multipliers**, not model inference speed or real-time
factors. They do not measure how many correctly pronounced words remain.

Concrete reconstructed examples:

- Bill at **335.480 s**, segment 54: `Ich habe es nicht berechnet.` A **0.960 s**
  fitting window and recorded **2.249×** multiplier imply **2.159 s** raw speech.
- CRISPR at **1336.130 s**, segment 131: `Warum?` A **0.640 s** window and
  **2.068×** imply **1.324 s** raw speech. A one-word question illustrates why
  shortening translations alone cannot resolve every bad timing window.
- Mind Control at **278.320 s**, segment 27: a two-sentence advertisement has a
  **7.902 s** fitting window and **2.133×**, implying **16.858 s** raw speech.

Historical raw speech WAVs were overwritten in temporary directories and are no
longer available. Raw duration is inferred as `rate * fit_window` only when
`rate > 1`. When the recorded rate is 1, the true duration is unknown. Do not
pretend those clips exactly filled their windows.

Reconstructed segment counts and speaker order match all five reports. That does
not prove exact historical text/interval identity: reconstruction uses today's
merger and the worker's speaker inference rules. This limitation is in the JSON.

## Actual code path and failure mechanisms

### 1. Estimated subtitle windows are treated as performance timing

`../batch_dubbing/gpu_worker.py` obtains Parakeet word timestamps, then calls the
grouping helpers in `../modal_control/smoke_nemo_helpers.py`. English caption
starts and ends preserve first/last word bounds, but the final JSON discards the
individual word list. Pauses inside an English caption cannot be reconstructed
from its endpoints alone.

German translation uses `../local_subtitles.py` and
`../nemo_audio.py::_split_coarse_segs`. Coarse translated spans are subdivided
proportionally by **word count**. Those are estimated caption display times,
not alignment of a real German utterance. Do not use cross-language words/sec
as a phonetic or perceptual equivalence measure.

### 2. Grouping breaks sentence context and loses pause provenance

`../modal_control/dub_segments.py::merge_speech_cues` merges same-speaker cues
using a 20-second limit, 280 characters, and a 0.4-second maximum gap. It does
not honor sentence endings and retains no constituent-cue provenance.

Mind Control's first grouped utterance ends `ins Gehirn`; the next begins
`zu gelangen`. This splits an unfinished grammatical construction into separate
TTS calls. Of its 118 full-video groups, 48 lack sentence-terminal punctuation.
Punctuation detection is a heuristic; abbreviations and recognition errors exist.

### 3. Recent workers permit unbounded speed-up

The relevant files are `../batch_dubbing/cpu_worker.py`, `cpu_duo_worker.py`,
`cpu_override_worker.py`, and `cpu_continuity_worker.py`.

They replace the current segment's end with the next segment's start, or video
end for the last segment; synthesize at native speed; compute
`rate = max(1, raw_seconds / window)`; then chain FFmpeg `atempo` filters with
no overall perceptual ceiling. This can consume source pauses and accelerate
successive phrases by very different amounts. The subsequent array copy can
discard a residual tail of up to the accepted 50 ms fitting tolerance.

Chaining filters avoids FFmpeg's sample-skipping behavior for a single factor
above 2, but **does not make 2× delivery sound natural**.

### 4. The older pipeline also slows speech and can lose words

`../../qwen3-tts/dub_audio.py::speed_fit` (around lines 604–665 at investigation
time) slows short clips as far as 0.65×. Overlong speech is sped up to 1.35×,
then trimmed at a detected silence boundary and padded to the target window.
Trimming at a silence can avoid cutting inside a phoneme while still dropping
whole trailing words. `dub.py::_do_fit` activates that behavior.

Do not accidentally describe the older slowdown as a behavior of the recent
CPU workers; those use `max(1, ...)` and therefore do not slow audio.

The older local fitter also starts at least four FFmpeg workers and does not
explicitly cap their thread counts. Its Supertonic worker constructs TTS without
the explicit ONNX thread limits used by the newer Modal workers. Do not run that
older path locally without implementing and reviewing resource controls.

### 5. Acoustic context is removed

`../batch_dubbing/package_video.py` maps the original video stream and generated
audio only. Source music/ambience is not mixed back. This is a distinct plausible
contributor to a detached voice; it has not yet been isolated by a listening test.
Do not simply mix the original speech back in and call that ambience restoration.

### 6. Translation errors exist before TTS

Examples from text inspection, with full context in `text_findings.json`:

- Mind, 12.718–24.960 s: `caught my eye` becomes the ungrammatical
  `hat mich das bereits aufgefallen`.
- Mind, 25.720–41.800 s: investigating questions becomes having answered them.
- Mind, 89.310–104.710 s: source `500 microseconds` / `10 milliseconds` becomes
  ambiguous `MS` in both places, with a spurious `Oxygen` fragment.
- Bill, 45.320–56.400 s: financial `compounding` becomes `Zusammenstellung`.
- Bill, 65.840–74.320 s: markets being shut `for twenty years` becomes
  `in zwanzig Jahren`, changing the temporal meaning.

These compare existing transcripts; no claim is made that the original ASR or
real-world statements in the video were independently verified.

## Continuous ten-minute review

Both review videos cover `[0, 600)` seconds. Actual `.en.srt` and `.de.srt` content
and times match their corresponding stored metadata. Cues crossing 600 seconds
retain their full text and real bounds, with separate visible bounds/flags.

| First ten minutes | Mind Control | Bill Ackman |
|---|---:|---:|
| English / German cues | 214 / 198 | 242 / 219 |
| German cues shorter than 2 seconds | 48 | 87 |
| Synthesized phrases | 58 | 95 |
| Median tempo | 1.337× | 1.293× |
| Phrases above 1.5× | 10 | 25 |
| Phrases above 2× | 1 | 3 |
| English inter-caption gaps | 23.866 s | 65.587 s |

English inter-caption gaps are only a source-pause proxy. They omit pauses inside
captions and cannot replace word-level alignment or speech activity detection.

An amplitude-only FFmpeg detector was run at both −35 and −45 dB with minimum
silence 0.20 seconds. Results for the same ten-minute ranges:

| Video/language | Low-amplitude seconds at −35 dB | At −45 dB |
|---|---:|---:|
| Bill original English | 0.92 | 0.00 |
| Bill German dub | 153.62 | 138.79 |
| Mind original English | 8.79 | 0.69 |
| Mind German dub | 98.50 | 83.92 |

**Do not conclude that the English speakers never pause.** Original music/noise
masks silence, while synthesized tracks have a much quieter floor. This reveals
a confound in naive silence metrics and supports checking acoustic context; it
does not quantify speech-pause alignment or assign a naturalness score.

## Older fixed-window benchmark

Reusable raw WAVs exist in
`../speech_model_benchmark_2026-10-27/04_dubbing/{supertonic_3,qwen_tts_06b,ming_05b,voxcpm2}/`.
There are 24 clips per model, each assigned to a 25-second source window.
All **96 WAV-header durations exactly matched their metadata**.

The packager speeds only overlong clips and pads shorter ones. Derived padding
totals for a 600-second timeline are:

| Model | Added padding |
|---|---:|
| Supertonic 3 | 50.317 s |
| Qwen 0.6B | 149.480 s |
| Ming 0.5B | 359.300 s |
| VoxCPM2 | 164.960 s |

Nineteen of 24 text chunks lack terminal punctuation. Many adjacent chunks plainly
continue the same sentence. Long padding at those artificial boundaries can
break the flow even when the voice itself is understandable.

Do not rank models by short duration alone. Ming's existing ASR round-trip proxy
reports WER 0.839 and 771 deletions; its short outputs may omit words. That proxy
is not human ground truth or a naturalness measurement.

## Local tests and resource evidence

`run_audio_checks.py` created ten copies from raw benchmark clip `000.wav` for
Supertonic 3 and Qwen 0.6B at the five tempos above. It also performed eight
ten-minute amplitude analyses: two videos × two languages × two thresholds.

All 18 FFmpeg jobs ran serially with:

- CPU affinity limited to one core, nice level 15.
- Explicit FFmpeg/BLAS thread bounds.
- 512 MiB process address-space ceiling, 60 CPU seconds and 90 wall seconds per job.
- Five-minute overall budget and 64 MiB file-size ceiling.
- No model loading, GPU use, or cloud execution for this local phase.

Measured: **13.108 seconds elapsed**, **12.771 CPU seconds**, peak child RSS
**64,764 KiB / 63.25 MiB**. All ten output durations were within 150 ms of
`raw_duration / tempo`. This validates transformation behavior, not audio quality.

At inspection, WSL reported approximately 32 GiB RAM and 28 GiB available;
`nvidia-smi` was blocked inside the sandbox. The user's 6 GB GPU statement was
not reverified through NVML. No local model was started.

Validation:

- 30 batch-dubbing unit tests passed, including 14 candidate-policy tests and
  six frozen-experiment tests. The randomized policy case covers 1,000 timing
  inputs; the grouping case preserves text/provenance across 250 cues.
- Three speaker-timeline tests passed before the branch push.
- Pending Python files parsed successfully; JSON parsed; no staged file exceeded
  1 MiB; staged whitespace check and credential-pattern scan passed.
- Earlier broad project tests were not rerun or represented as current evidence.

## Candidate implementation and its limits

`../batch_dubbing/natural_timing.py` contains two pure functions:

1. `group_speech_cues`: validates ordered finite nonoverlapping attributed cues;
   respects punctuation, meaningful gaps and speaker changes; preserves source
   indices/cues and text; flags oversized/incomplete phrases rather than dropping
   content or splitting an individual cue.
2. `plan_timing`: works from measured raw duration; never slows to fill a slot;
   permits at most 150 ms borrowing from a same-speaker gap no longer than 500 ms,
   leaving at least 100 ms; applies an experimental default ceiling of 1.15×;
   returns `needs_rewrite` for infeasible speech and `review_underfilled` for
   substantially short speech. It does not transform audio.

`needs_rewrite` means intervention is needed; rewriting is not necessarily the
right intervention for a one-word question. Options include a better aligned
window, a different performance, or a shorter faithful translation.

The library grouping defaults are 12 seconds / 240 characters / 0.25-second gap.
The released experiment deliberately tests **20 seconds / 500 characters** to
keep the long opening Mind Control sentence intact. Do not conflate those settings.
Sentence punctuation remains heuristic, and source-caption timing is still only
an estimate. A `ready` timing status is not proof of semantic or perceptual quality.

`sweep_policy.py` holds historical grouping fixed and evaluates six tempo caps.
At 1.15×, retaining original phrase ends with limited gap borrowing leaves
54/56 compressed Mind phrases and 75/79 compressed Bill phrases infeasible in
their first ten minutes. Uncompressed clips are excluded because raw duration is
unknown. This counterfactual demonstrates that capping speed alone is not enough;
it is not a resynthesis or a prediction of a new performance.

## Exact released synthesis experiment

- Catalog: `../modal_control/catalog/naturalness_native_grouping.json`.
- Worker: `../batch_dubbing/naturalness_worker.py`.
- Pure input/job builder: `../batch_dubbing/naturalness_experiment.py`.
- Model: `Supertone/supertonic-3`.
- Revision: `3cadd1ee6394adea1bd021217a0e650ede09a323`.
- Package: `supertonic==1.3.1`; ONNX CPU; fixed existing M1/M2 voice assignments.
- Bill: **325.000–349.960 seconds**, 11 complete German cues.
- Mind: **0.160–24.960 seconds**, 7 complete German cues.
- Unique source span: **49.76 seconds**.
- Baseline: original merger, 20 seconds / 280 chars / 0.4-second gap.
- Candidate: sentence-aware merger, 20 seconds / 500 chars / 0.25-second gap.
- Bill calls: 8 baseline + 10 candidate. Mind: 2 + 2. Total workload **22**.
- Smoke: first baseline Bill phrase, **one** additional call. Total **23**;
  hard scope guard permits no more than 32 including smoke.
- One CPU container at a time, **2 CPU / 4096 MiB**, ONNX intra-op 2/inter-op 1.
- **600-second timeout per stage**, smoke then workload, no retries; total execution
  ceiling 1,200 seconds. Build/startup time and charges are additional possibilities.
- Maximum raw output: 120 seconds per clip and 300 seconds per stage.
- Output: individual native FLOAT WAVs, full input snapshot, native durations,
  voice/text/bounds, generation timings, peaks, timing decisions and telemetry.
- No fitting, clipping, slowdown, retranslation, media packaging, or local inference.
- Cost was disclosed as **unknown**, not a monetary cap.

This changes grouping only and preserves the input words and voices. Generation
is stochastic; a single realization per condition does not isolate randomness.
Do not interpret a small duration difference as conclusive grouping causality.

The exact released plan is single-use. Do not rerun its execute command after it
has been consumed. A failed stage stops progression. Investigate first; paid
retries or altered scope need a new release. The user's existing release must
not be requested again for later steps already inside the unchanged plan.

## Final execution receipt

The released run completed successfully on 2026-10-28:

- Controller terminal status: `succeeded`, recorded at **16:23:08 UTC**.
- Smoke app: `ap-9wSiPcneyRkaN7xJszQD6Z`; one call, then stopped.
- Workload app: `ap-6x5iWz9FhEx7LaITjWM5nx`; 22 calls, then stopped.
- Workload worker wall time: **36.238 seconds**, excluding image build and startup.
- Highest sampled workload process RSS: **759,713,792 bytes / 724.52 MiB**;
  this is a five-second sample maximum, not a guaranteed instantaneous peak.
- During-run snapshot at 16:22:25 UTC recorded **one** owned container.
- After-run snapshot at 16:26:40 UTC recorded **zero containers**, both apps
  `stopped`, and zero tasks. Their attached entrypoints stopped automatically;
  there was no still-running owned app requiring a separate stop call.
- `modal billing report --for today --json` recorded **$0.00040378** for smoke
  and **$0.00221439** for workload: **$0.00261817 total**, approximately $0.0026.
  These are the two app rows observed after completion, not total account spending;
  billing may update later and shared storage/build costs are not separately
  attributed by this sum.

`verify_native_trial.py` streamed all 23 saved FLOAT WAVs under one-core/512 MiB
local bounds. It checked job/text/voice identity, frame counts and sample rates,
metadata duration, finite samples, non-silent output, recorded peaks and SHA256
hashes. It used about **16.25 MiB** peak RSS. It did not run ASR, infer words from
audio, or establish intelligibility or naturalness.

### What the new synthesis actually showed

| Excerpt / grouping | Clips | Native speech seconds | Fits candidate timing | Needs intervention |
|---|---:|---:|---:|---:|
| Bill / baseline | 8 | 25.913 | 2 | 6 |
| Bill / candidate | 10 | 26.610 | 3 | 7 |
| Mind / baseline | 2 | 31.835 | 0 | 2 |
| Mind / candidate | 2 | 31.417 | 0 | 2 |

The timing columns evaluate both arms with the same conservative candidate policy;
they are not measured ratings of the audio, and phrase counts differ between arms.
Do not compare raw failure counts as if there were identical test units.

The candidate repairs the opening Mind sentence boundary: its first utterance now
ends `aktivieren.` instead of `ins Gehirn`. However, its **21.873 seconds** of
native audio must fit a source span from 0.160 to 17.520 seconds (**17.360 s**).
The second utterance takes **9.543 seconds**. Total native speech remains
**31.417 seconds for a 24.800-second source excerpt**. Grouping alone therefore
does not resolve the duration problem even when it restores sentence continuity.

Bill's native speech total increases slightly with smaller sentence groups.
The problem phrase `Ich habe es nicht berechnet.` is **2.159 seconds** in both
arms, confirming the earlier duration inferred from the 2.249× fitted report.
Its original caption lasts 0.880 seconds; the recent worker's expanded fitting
window was 0.960 seconds. This is direct evidence of a tight window, not just
an assumption about the model being slow.

Six unchanged text/voice pairs across arms have the same recorded duration but
different WAV hashes. Do not equate matching durations with identical waveforms;
do not claim determinism from this trial. Preserve this stochastic variability
limitation when discussing subtle prosody differences.

**Result:** native audio and useful measurements are now available, but the
candidate has not fixed timing feasibility. Human listening, translation/window
improvements and a separately released follow-up remain outstanding. No ten-minute
video has been regenerated and no production policy has been changed.

## Commands useful to the next agent

From `/home/sarpk/uv_tests/ASR/nemo`:

```bash
git status --short
git branch --show-current
./modal-control show 2026-10-28-a973f4b68d86
./modal-control logs 2026-10-28-a973f4b68d86
# If the user requests stop while this exact run is active:
./modal-control stop 2026-10-28-a973f4b68d86

# Read-only model-free audit, writes local evidence JSON:
python3 dubbing_naturalness_2026-10-28/audit_existing.py
python3 dubbing_naturalness_2026-10-28/sweep_policy.py
python3 dubbing_naturalness_2026-10-28/verify_native_trial.py
python3 dubbing_naturalness_2026-10-28/build_review.py
```

Do not rerun `run_audio_checks.py` blindly: it deliberately refuses an existing
comparison directory. Preserve the current samples. Likewise, fresh synthesis
outputs are exclusive and must not be overwritten for convenience.

## Recommended continuation after this released experiment

1. Verify controller terminal status, stage receipts and all expected raw WAVs;
   inspect actual durations and retained words. Do not substitute metadata-only
   tests for listening. Verify owned Modal apps/containers have stopped and record
   actual billing information if available, stating attribution limits.
2. Compare native baseline and candidate phrases while holding words/voice fixed.
   Listen to the original English context too. Flag broken clauses, missing words,
   pronunciation, stress, pace and voice consistency independently.
3. Determine whether benefits come from grouping or merely random generation.
   Prepare a bounded repeated trial only if needed and obtain release for it.
4. Improve the speech timing source: retain word timestamps and real source pauses;
   separate subtitle display segmentation from speech phrasing. Do not invent
   precise word alignment from proportional German caption windows.
5. Address translation fidelity and duration separately: preserve named entities,
   quantities, temporal meaning and discourse structure; test shorter faithful
   German formulations only when content truly cannot fit naturally.
6. Test background preservation separately, without leaving the original dialogue
   audible beneath the dub. Any separation model/media packaging needs approval.
7. Only then expand a successful policy to a full ten-minute synthesis and seek a
   human preference comparison. Do not silently make the candidate the default.

## External references consulted

- FFmpeg `atempo`: <https://ffmpeg.org/ffmpeg-filters.html#atempo>.
  Chaining tempo filters addresses the implementation's >2× sample-skipping mode;
  it is not a guarantee of perceptual naturalness.
- Tam et al., *Isochrony-Aware Neural Machine Translation for Automatic Dubbing*:
  <https://www.isca-archive.org/interspeech_2022/tam22_interspeech.html>.
- Federico et al., *Evaluating and Optimizing Prosodic Alignment for Automatic
  Dubbing*: <https://www.isca-archive.org/interspeech_2020/federico20_interspeech.html>.

These support investigating speech/pause structure and prosodic alignment. The
specific numeric findings above come from this repository's artifacts, not from
the papers. No claim of a completed naturalness fix should be made yet.
