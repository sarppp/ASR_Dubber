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

**Latest requirement: 2026-10-29 — 1.0x playback, original talking and silence durations first; emotional performance later. No new inference released or run.**
The user explicitly requires a general automatic solution across videos and
languages, not manually corrected timestamps or individually rewritten phrases.
They also request periodic updates to this handoff. The new source-timing and
safe-fitting code is integrated into shared workers; see the current
checkpoint below before relying on older experiment-only descriptions. Existing
videos, subtitles, and generated audio have not been overwritten. No new model
job has run since the released grouping trial.

## Latest budget and speed-default checkpoint: 2026-10-29

The user rejected the $1–$2 initial allowance and requested the cheapest $0.10
test. A much smaller CPU-only baseline probe is now prepared, not released or
run. This does not replace the researched architecture with an unsupported claim
that Supertonic provides duration/pause conditioning.

New verified finding: the cached, RECORD-verified `supertonic==1.3.1` wheel sets
`DEFAULT_SPEED=1.05` in `config.py:202`; `pipeline.py:158` inherits it, and
`core.py:507` divides predicted duration by that speed before generating audio.
Both `cpu_natural.py` and `naturalness_worker.py` previously omitted the argument.
Historical samples therefore used internal speed1.05 (about4.76% shorter predicted
duration than1.0), even though later playback was labelled native. This does not
explain all large timing failures. Both calls now explicitly pass `speed=1.0`;
new reports disclose this and the unchanged automatic chunking/300ms pause
defaults. Historical evidence files remain untouched. The API has no target
duration or source-pause-timeline parameter.

Prepared cheapest probe:

- `batch_dubbing/cheap_native_probe.py` and `cheap_native_worker.py`;
  frozen `cheap_native_inputs.json`, `cheap_native_batch.json`, and
  `modal_control/catalog/cheap_native_probe.json`.
- Reuses successful trial image `im-40qUTZlx5ttgX1nwrQdvyw` and the existing
  pinned model cache at revision `3cadd1ee6394adea1bd021217a0e650ede09a323`.
  No apt/pip build, new model download, GPU, VAD/ASR model or translation API.
  A missing cache/image fails; there is no rebuild or fallback.
- Four unchanged existing texts: English Bill cue129 and German116 (question),
  English130 and German117 (answer). M2 question/M1 answer. First English text
  is the smoke call; only its successful matching report allows the other three.
  This is a native-speed baseline, not a comparison to acoustically measured
  original speech. Existing translation defects remain visible.
- Explicit speed1.0, eight steps, fixed random seed. Enforce one text chunk per
  sample and zero wrapper-inserted silence. No waveform fitting, trimming or
  video packaging. Four WAVs maximum,12seconds each,8MiB transfer perstage.
- One CPU container at a time, CPU request/limit2, RAM request/hardlimit4GiB,
  two ONNX threads, no retries;90s function,30s startup,2s scale-down. Controller
  stage deadline120s includes client startup; two stages total240s and batchwall300s.
  Controller now accepts literal resource pairs and estimates cost from upper
  limits. Source/model cache presence has not been checked through a cloud job.
- Public base-rate calculation:240s×(2×$0.0000131+4×$0.00000222)=$0.0084192.
  With24s shutdown/cancellation allowance and16MiB worst paid transfer, about
  $0.00989. Approximately one cent estimated, with substantial room below the
  requested ten cents. This is not a verified provider-enforced currency cap;
  cancellation/billing overrun is not impossible. Do not call it a guaranteed
  invoice ceiling or silently increase spending after failure.
- Local generated review plan: `cheap_native_release_plan.json`, ID
  `mb-51cfdc453423af285674f0fe046f7d28`, expires2026-10-29T08:41:51UTC.
  The earlier same-task plan `mb-a953479028bcb004bcce3f555f055fd5` is superseded
  after a controller documentation edit changed its pinned hash. Reprepare an
  expired plan from the unchanged reviewed specification; no inference released.

Final local checks: **331 passed,1 skipped,465 subtests passed**;17.58s pytest,
17.907s outer wall,83,772KiB peak child RSS (about81.8MiB),one core,512MiB address
space. The13 probe tests include failure/partial-artifact preservation, explicit
settings, cache-only loading, resource caps and smoke ordering. No live model,
network inference, video or existing audio processing ran. Full long-form timing
and naturalness remain unverified. The old1.15x catalog stays disabled.

## Strict timing implementation checkpoint: 2026-10-29

The user approved the researched direction: adapt wording and condition synthesis
to the original speech/pause pattern, with playback fixed at 1.0x. The following
shared modules now implement the model-independent comparison stage:

- `batch_dubbing/isochrony.py`: immutable, provenance-labelled speech intervals;
  Boolean activity-mask conversion; speech/pause overlap and missing/extra speech;
  onset, offset and duration errors including edge silence. Default timing
  tolerance is zero apart from 1ns numeric slack. Measurement resolution is
  reported separately and never used to silently relax the policy. Unequal
  block counts do not receive fabricated correspondence. Subtitle/ASR-word gaps
  and energy thresholds cannot become verified acoustic speech evidence.
- `batch_dubbing/isochrony_backend.py`: explicit language/duration/pause capability
  declarations and immutable candidate specifications. IDs bind wording, source
  timing/evidence, model revision and seed. A duration-only provider cannot
  silently claim pause control. Up to eight serial candidates; no automatic
  retries, rate adjustment or trimming. Capability declarations are not proof
  of achieved timing or naturalness.
- `batch_dubbing/isochrony_revision.py`: prepares full-context feedback for
  shorter or longer faithful wording, mixed phrase budgets, or pause repair.
  Preserves names, numbers, negation and intent; forbids added facts/filler and
  dropped information. Allows `cannot_fit_faithfully` or a synthesis repair.
  Chronological blocks are not claimed to be semantic alignments. This prepares
  a provider request; it does not call a translation model.
- `batch_dubbing/isochrony_pipeline.py`: connects prepared candidates, a supplied
  synthesis/measurement callback, strict comparison, and wording feedback. All
  candidates are validated before the first call. Failures stop without retry;
  earlier attempts and raw-audio references/digests remain in a new report folder.
  Timelines are limited to 256 speech intervals per comparison to bound report
  growth. Unsafe speed/trim observations request synthesis repair instead of
  deriving misleading wording feedback from altered audio. Synthetic matches
  are separate from acoustic matches. Meaning, listening and artifact/processing
  audit stay pending; this harness never publishes audio.

Important boundary: these are general components with synthetic provider tests,
not an installed English/German synthesis solution. Existing production workers
still use the older scheduler and are not wired to this strict comparison path.
No new model adapter, acoustic measurement worker, translation provider, model
download, cloud inference or real-media processing ran in this implementation.
The external Modal controller must enforce hardware/time limits around callbacks.
Audio references and processing declarations are recorded, not independently
downloaded/verified by the pure comparison harness.

Independent review caught and corrected JSON numeric normalization, contradictory
processing metadata in nested revision feedback, and excessive possible report
growth. Tests include a maximum-sized comparison (eight candidates, 256 intervals)
and 1,000 randomized mask comparisons against a separate discrete overlap oracle.
The final combined run reports **313 passed, 1 skipped, 370 subtests passed**
(56 new strict-timing tests, including 1,000 randomized cases). The existing skip
is the optional MCP dependency. The test command below used one core, 512MiB
address space, serial execution and synthetic media only: 18.24s pytest,
18.655s outer wall time, 78,396KiB peak child RSS (about76.6MiB). No model or GPU
ran. `test_package_video.py` remains outside this bounded suite as in the earlier
checkpoint; no claim of that separate packaging workflow being revalidated.

Next concrete work: choose and implement a reproducible English/German backend
adapter with source acoustic measurement, connect these comparisons inside the
controller, and prepare a bounded model experiment. F5's duration-only candidate
must remain labelled as such; internal pause matching is not established. Do not
re-enable the old 1.15x evaluation or present code tests as listening evidence.

## Research checkpoint: 2026-10-29

The user asked to research cinema practice and papers before proceeding. Read
[the primary-source research report](RESEARCH_ISOCHRONY_2026-10-29.md) before
selecting a synthesis backend or preparing another release.

The investigation supports adapting translation and conditioning synthesis on
speech/pause budgets. It does not establish a ready English/German system that
guarantees exact timing and natural pace. Professional practice is not evidence
that every source pause must already be identical in cinema dubs. Keep the
user's stricter target explicit. Equal duration, playback 1.0x, and matching
speech activity are three different checks. The report records language,
release-availability, hidden rate-control, and evaluation-metric limitations.

No inference or media processing ran for this research. The superseded catalog
remains disabled. Strict timeline evaluation and backend capability contracts are
now implemented as described above; a model comparison still needs a concrete
bounded plan and a real backend/measurement integration.

## Latest priority correction: 1.0x and matching speech/pause timing

This section supersedes the emotional-performance-first plan below. The user
clarified: "It doesn't matter if they are doubtful etc at first. At first speed
and silences. It should be always 1.0 so same silence time same talking time.
Just language is different."

Operational target:

- Original speech blocks and intervening pauses define the target timeline.
  The translated language must occupy the corresponding talking intervals and
  preserve the pauses. Total video length alone is not the acceptance test.
- Playback/assembly tempo stays exactly **1.0x**. No atempo acceleration or
  slowdown, no consuming the source pause, no clipping words to meet a deadline.
- A short generated utterance followed by added silence does **not** satisfy
  equal talking time: it changes the speech/pause structure. An overlong utterance
  does not pass merely because it fits inside a wider speaker turn either.
- Translation adaptation and synthesis must jointly produce an appropriate native
  utterance duration. Too-short and too-long output both require a new wording or
  synthesis candidate under an explicitly bounded plan. Meaning must be preserved.
- "1.0x" alone does not specify how quickly a model naturally speaks. Verify
  generated duration and source pauses directly rather than relabeling rushed
  generation as natural because its playback multiplier happens to be one.
- Source acoustic speech/pause evidence is needed. Subtitle display timings and
  word-boundary gaps are not exact measurements of original silence. Distinguish
  within-word unvoiced sounds from actual pauses; retain measurement uncertainty
  and timing error rather than claiming sample-perfect speech detection.
- Natural multilingual delivery remains the final goal, but emotion transfer,
  elaborate expressive conditioning and ambience restoration are not the first
  evaluation priorities now.

Current implementation gap: the source scheduler permits 1.15x, up to250ms
boundary shifts, and early speech endings. It tests bounded fit, not equal
speech/pause duration. Existing tests validate that older policy, not this new
requirement. Changing only the cap to1.0 would leave short speech, incorrect
internal pauses and infeasible longer wording unresolved. Do not claim that
change alone implements the user's request.

The previous evaluation catalog `source_naturalness_validation.json` is now
explicitly `submissions_disabled` with this reason. Its frozen plan
`mb-c20afdeffc3ad48fd06e197b60ec1d7b` is superseded; changing the catalog invalidates
its pinned digest as well. Do not ask the user to release that old experiment.
No existing audio, models or general-purpose historical worker policy were
silently changed or executed in this clarification.

Next implementation/evaluation must target duration-conditioned native synthesis
plus faithful dialogue adaptation, with separate assertions for playback rate,
each speech onset/offset, each pause and preserved content. Do not promise every
language/phrase can meet exact duration naturally before testing that capability.

## User correction: cinema-style delivery — 2026-10-29

The user explicitly clarified: "I want it to be like cinema. Not less rushed."
They want natural tone in different languages even when translated dialogue has
a different length. Treat this as the acceptance target for the ongoing task.
Passing timing tests or imposing a 1.15x cap does not meet that target.

Current concrete limitation: `cpu_natural.py` passes translated text, a preset
voice style and a language code to Supertonic. It does not pass the original
utterance audio, emotion, emphasis, pitch/energy movement or delivery intention
to synthesis. Source-word timing and speaker IDs are retained, but the source
performance is not used to condition the generated performance. The committed
work prevents known damage; it cannot restore information never supplied to TTS.

The earlier engineering choice was to repair measurable timing defects before
changing models. Retain those protections and diagnostics as infrastructure, but
revise the next evaluation around performance quality. The prepared three-case
Supertonic-only plan remains an unreleased timing baseline; do not present it as
a sufficient test or solution for cinema-style dubbing, and do not launch it by
default in response to this correction.

Required architecture and acceptance dimensions:

1. Carry original dialogue audio and contextual delivery information through the
   pipeline, alongside words and speaker identity. Emotion labels alone are not
   equivalent to preserving a performance.
2. Adapt dialogue into idiomatic target-language speech with context and a duration
   budget. Preserve meaning, facts, negation and intention; do not enforce literal
   word counts or shorten away information. Different sentence lengths require
   different wording/rhythm, not uniformly accelerated waveforms.
3. Evaluate expressive multilingual synthesis or speech-to-speech translation
   conditioned on source delivery. Distinguish voice identity from acting/emotion:
   matching a voice alone does not establish matching performance. Target-language
   stress and intonation should remain natural rather than copying every source
   pitch movement mechanically. No new model is selected or benchmarked yet.
4. Let synthesis and adapted wording solve duration jointly. Permit phrase-level
   timing flexibility within meaningful speaker/visual anchors, preserve pauses
   that carry intention, and avoid global tempo as the main fitting method. Fixed
   video still creates real constraints at speaker changes and visible action.
5. Preserve/reconstruct the music/effects bed and match loudness/room acoustics;
   this remains unimplemented and any separation/media execution needs its own
   disclosed approval within the new reviewed scope.
6. Evaluate meaning, idiomatic speech, emotional intent, emphasis, voice consistency,
   audible edits, audiovisual timing and mix separately using listening comparisons.
   English/German are the initial evaluation pair; do not generalize that result
   into an all-language quality guarantee. Automated checks are supporting evidence.

Before requesting another cloud release, compare candidate architectures/models
against these requirements, check supported languages and reference/duration
conditioning, and verify bounded hardware feasibility. The user's 6 GB local GPU
does not constrain a separately approved cloud model to the CPU preset already
chosen. Local WSL protection and controller release rules remain unchanged.

Primary references consulted for this correction:

- [Amazon Science: Evaluating and optimizing prosodic alignment for automatic dubbing](https://www.amazon.science/publications/evaluating-and-optimizing-prosodic-alignment-for-automatic-dubbing)
  describes duration, audiovisual timing, emotion/prosody and background acoustics
  as separate dubbing requirements, and studies relaxed synchronization.
- [Meta: Seamless Communication](https://ai.meta.com/research/seamless-communication/)
  describes expressive cross-language translation preserving speech rate, pauses,
  vocal style and emotional tone. This establishes a relevant architecture to
  investigate, not a model selection or proof of cinema-level results here.

## Continuation — 2026-10-29

The current user asked to continue and explain the latest point. The general
timing/fitting implementation is committed and its 229-test result is recorded
below. It is not yet a validated natural-sounding dub. Automatic faithful
shortening is still unconfigured; timing failures are explicit diagnostics.

Preparation now uses three frozen excerpts without manual timing/text repairs:
Bill 325–350 s, Mind 0–30 s, and CRISPR 0–30 s. CRISPR was not part of the previous
native grouping trial. SHA256 and PCM16/mono/16 kHz headers of the existing source
WAVs were verified locally; only streamed file reads occurred. These WAVs remain
available even though historical catalog MP4 paths have moved.

- `source_validation_inputs.json` freezes inputs/hashes/ranges and the German-only
  target. `validation_config.py` checks the scope and file provenance without
  models or media transforms. At most three cases, 30 s each, 12 units per case.
- Shared `batch_dubbing/speech_stage.py` now holds the GPU processing path used by
  both existing GPU workers and the evaluation. It validates ranges before model
  imports, keeps excerpt offsets, and caps units before loading Canary. No copied
  per-video synthesis algorithm is introduced.
- New GPU/CPU evaluation workers and catalogs passed local checks. Prepared
  order: GPU smoke on CRISPR, GPU full on all three; CPU smoke on one whole CRISPR
  utterance, then CPU full on all three. Global concurrency one; no retries.
- Proposed hardware: T4 + 4 CPU / 32 GiB for speech; CPU-only 2 CPU / 4 GiB for
  Supertonic. Four stage deadlines of 600 s each. All inference remains remote.
- Process 85 s of workload source audio, plus repeated 30 s GPU smoke; at most
  48 utterance translations and 37 synthesis calls including smoke. Full original
  WAVs (212,616,256 bytes total) are mounted, but only reviewed ranges are processed.
- Preserve original English words, German text, alignment evidence, raw native
  WAVs and timing reports. Publish final WAVs only if every requested case fits;
  no optional video packaging/background separation. Infeasible results remain
  failures, with native listening evidence downloaded up to 64 MiB per stage.
- Actual cost is unknown. The controller's embedded historical-rate calculation
  is a planning estimate, not current verified pricing or a provider billing cap.
  Build/startup, storage and egress can add cost. No cloud operation has launched.

The final combined regression run on this checkpoint reports **257 passed,
1 skipped, 229 subtests passed in 15.48 s** (16.052 s outer wall, **56,472 KiB**
peak child RSS). Same one-core/512 MiB limits and test selection as the older run
below; the sole skip remains the unavailable MCP dependency. New tests exercise
scope limits, source mutation, shared-stage delegation, GPU/CPU ordering, offsets,
voice rejection before allocation, raw transfer hashes, failure publication and
telemetry. No model or user-media transformation ran.

Prepared controller batch: **`mb-c20afdeffc3ad48fd06e197b60ec1d7b`**.

- Batch specification: `modal_control/catalog/source_naturalness_validation_batch.json`.
- Job catalog: `modal_control/catalog/source_naturalness_validation.json`.
- Local full plan copy: `source_validation_release_plan.json` in this investigation
  directory (ignored generated state, also retained in the controller plan store).
- `batch-inspect` succeeded. All pinned source/controller hashes matched a second
  streamed verification; every declared output path was absent.
- Two dependent jobs, each smoke then full; at most **2,400 total stage seconds**,
  **5,400 s batch wall limit**, one global container, one T4 at most, zero retries.
- Controller's historical-rate compute ceiling is **$0.387024** against a $1.00
  planning allowance. Neither number is a provider-enforced spend cap or a current
  billing quote; build/startup, storage and egress may be additional.
- Plan expires **2026-10-29 06:56:19 UTC / 07:56:19 Zurich**. If the user responds
  later, regenerate and inspect an identical-scope plan; do not interpret expiry
  as permission to expand inputs/resources/retries or ask for a repeat decision
  when the user's explicit release already covers the unchanged reviewed scope.

**No release has been received for this batch.** Request the new release required
by `../AGENTS.md` for this materially different ASR/translation/synthesis experiment.
The previous consumed grouping plan cannot authorize it. On release, execute only
the inspected batch through the controller, monitor owned jobs, stop/reconcile owned
apps afterward and record real billed usage plus actual results. If any stage is
blocked, preserve diagnostics and seek a separately reviewed next step; do not retry
or silently relax speaker/timing/voice checks. Continue updating this section.

## Current implementation checkpoint — 2026-10-28 17:22 UTC

This section supersedes older statements that workers are unchanged. These are
committed source changes on `modal-control`, not a claim of deployment, a newly
synthesized video, or verified perceptual naturalness.

### Newly measured causes

- Native TTS files include removable edge padding. The conservative **peak**
  detector in `batch_dubbing/audio_edges.py` uses -60 dB activity, 10 ms frames,
  80 ms leading / 120 ms trailing guards, and at most one second removed per edge.
  It preserves interior pauses and quiet activity rather than trimming by RMS.
- `edge_audit.json` measures 22 saved native clips: **115.774694 seconds raw,
  101.800000 seconds retained, 13.974694 seconds removable edge padding**.
  This read-only waveform audit used one core, about 25 MiB RSS, and no model.
  Earlier larger RMS estimates are superseded by these conservative figures.
- The problematic German caption for `Ich habe es nicht berechnet.` lasts
  0.880 seconds, while original English word timing spans 334.840–336.280
  (1.440 seconds) and the diarized corridor spans 334.845–336.345 (1.500 seconds).
  Thus translated caption display windows are not reliable speech deadlines.
- Guarded edge removal still leaves the first Mind candidate at about 21.270
  seconds against 17.360 seconds of source speech. Removing padding and restoring
  source timing cannot by themselves make every translation fit.
- Cached Supertonic code returns a predicted duration, not measured speech
  activity; retain both predicted duration and waveform measurements in reports.

### General code now present

- `batch_dubbing/speech_units.py`: immutable source-word/utterance IDs, real ASR
  timestamps and speaker corridors; sentence/pause grouping; exact translation
  binding by ID. Rejects missing/ambiguous alignment. Subtitle display estimates
  remain distinct from synthesis timing. Both GPU workers now persist
  `source_words` and `speech_units[language]` and translate each unit's own audio.
  GPU catalog source pins were updated; hardware limits were not changed.
- `batch_dubbing/audio_edges.py`: conservative outer-padding measurements, with
  quiet consonant protection and explicit retained sample boundaries.
- `batch_dubbing/speech_schedule.py`: schedules a speaker's utterances jointly
  inside its corridor, tries native speed first, then at most 1.15×, with bounded
  onset/end shifts and pause protection. Infeasible speech returns diagnostics.
- `batch_dubbing/natural_pipeline.py`: provider-independent serial synthesis,
  persisted raw attempts/hashes/measurements, scheduling, and an optional bounded
  faithful-shortening callback (at most one revision per utterance, explicit call
  budget). No model is loaded by this module. Protected-term checks alone cannot
  establish semantic fidelity. Twelve orchestration tests now pass, including
  bounded revisions, rejection before inference, failure receipts, language-varied
  fixtures and a tiny synthetic source-to-timeline FFmpeg test.
- `batch_dubbing/natural_render.py`: validates source hashes/samples and the whole
  schedule before publishing a new timeline. Serial bounded FFmpeg, no slowing,
  no post-tempo clipping, and no partial timeline on overflow. Shared gain avoids
  per-phrase loudness normalization. Synthetic/mocked checks are not listening.
- `batch_dubbing/cpu_natural.py`: adapter for the pinned Supertonic model within
  a released Modal CPU job. Requires source-linked speech units; old translated
  subtitle artifacts must be regenerated. Preserves native attempts in the cache
  volume. **All five CPU workers now call this helper**, save reports before
  rejecting unsuccessful output, and commit native attempts even on failure.
  Local entrypoints also validate source units before remote allocation, including
  rejecting old subtitle-only artifacts and empty complete-utterance ranges.
  Eight CPU catalogs pin the shared helpers. The old smoke catalog's output-name
  mismatch was corrected to the worker's `_v2` filenames. The
  adapter currently has no revision provider: impossible timing yields an explicit
  `needs_revision` report rather than a distorted finished dub.
- Legacy `qwen3-tts/dub_audio.py` now delegates to `batch_dubbing/fit_audio.py`:
  no artificial slowdown, no cropped trailing words, at most 1.15×, one fit job
  at a time, and fit failure prevents final stitching/publication. Old unsafe
  fitted checkpoints are invalidated. Matching CLI defaults and Supertonic thread
  limits were changed. The legacy local-inference CLI still exists; executing it
  is not authorized under `AGENTS.md`. Following-gap borrowing is now capped at
  250 ms and bounded by the next utterance; the last phrase gets no extra allowance.

### Integration review findings and fixes

- A strict assumption that every ASR word fits inside a diarization turn within
  20 ms would reject real input. Read-only endpoint checks on saved English cues
  found discrepancies in 65/2442, 15/1406, 15/250, 201/416 and 4/866 cues across
  the five existing videos. Cue endpoints come from ASR words, but this check is
  not a substitute for a fresh word-level alignment audit. **An explicit generic
  reconciliation step is implemented:** keep word timestamps unchanged,
  use positive speaker-overlap evidence, retain raw turns and adjustment records,
  reject ambiguous attribution, and prevent adjacent speakers' corridors overlapping.
  Both GPU workers save raw words/turns, reconciliation diagnostics and a SHA256
  receipt under `/cache/source-alignment` before translation. An unresolved
  alignment returns `needs_alignment` and the entrypoint saves diagnostics then
  fails. The five historical raw-turn sets contain no actual overlapping
  different-speaker turns, but individual-word acceptance still needs fresh ASR.
- Adapter preflight now rejects invalid or >7200-second ranges before loading a
  model. A partial-range smoke clips only corridor bounds, never word bounds, and
  records any utterance excluded because it crosses the requested end. A range
  with no complete utterance fails with an explanation instead of empty synthesis.
- Renderer/encoder failures now persist `failed`, the failing stage and error,
  retaining native attempts. An incomplete AAC file is never published as the final
  timeline. Previously the scheduling report could misleadingly remain `ready`.
- Word provenance validation now rejects duplicate/mismatched indices, altered
  first/last bounds and wrong speakers. Provider callbacks receive copied data so
  they cannot mutate source evidence in place.

### Verification and limits at this checkpoint

- Agent-reported source-unit tests: **23 passed + 23 subtests**, including worker source-contract
  checks, language-varied text, and word preservation. No cloud/model execution.
- Agent-reported legacy fitter plus existing routing/checkpoint tests: **70 passed
  + 4 subtests**. Includes tiny synthetic native/mild-tempo FFmpeg fixtures and a
  regression proving one failed phrase prevents final stitching. No existing
  user media was transformed in these tests.
- Earlier module checks: 11 edge tests, 13 scheduling tests, 14 mocked renderer
  tests. The combined run below supersedes these partial runs; do not add
  these counts to historical totals as if they were independent runs.
- Main-agent verification: **25 passed + 28 subtests** for orchestration/rendering;
  **25 passed + 64 subtests** for adapter, worker integration and orchestration.
  These runs overlap and must not be summed. One CPU core, 512 MiB address-space
  cap, BLAS/OMP thread limits 1; each finished in under two seconds. Only tiny
  synthetic FFmpeg fixtures ran; no model or user-media transformation occurred.
- Final combined run: **229 passed, 1 skipped, 185 subtests passed in 14.84 s**
  (15.517 s outer wall time), **56,300 KiB peak child RSS**, one CPU core and a
  512 MiB address-space ceiling. The skip is the MCP test's missing `mcp`
  dependency. This includes all `batch_dubbing/test_*.py` except the unrelated
  video packaging test, all `modal_control/tests`, and the existing
  `tests/test_supertonic_default.py` / `tests/test_dub_pipeline.py` suites.
  No model was loaded and no cloud job or existing-media transformation ran.
- Static review validates **18 changed catalog jobs** against controller resource
  rules. All changed helper source pins exist. Six historical GPU job entries
  still reference five original MP4 paths that no longer exist at those locations;
  do not call those catalogs runnable. A fresh evaluation needs verified current
  input locations and fresh exclusive output paths, plus the new release below.
  `git -c core.whitespace=cr-at-eol diff --check` passed; legacy CRLF is preserved.
- No fresh synthesis validates the new source-unit translation path yet. Translation
  fidelity, voice/prosody, ambience restoration, and English/German listening
  validation remain separate open questions. A universal naturalness guarantee is
  neither established nor realistic from timing tests alone.
- `plan_repair.py`, `repair_alignments.json`, and `repair_plan.json` are earlier
  **diagnostic, manually aligned experiments only**. Do not wire them into production,
  use their per-video corrections as the general solution, or publish their feasible
  schedule as if repaired audio had been rendered. No repaired media was produced.

### Remaining work and next handoff update

1. Verify delivery state below, and preserve any concurrent bilingual-subtitle work.
2. Prepare a fresh bounded controller evaluation with verified input paths and new
   output names. The previous trial did not retain individual ASR words, so old
   JSON cannot establish acceptance of the new source-unit path. Do not turn cue
   endpoints into invented word alignments. No new evaluation plan is released.
3. Validate translation quality with utterance-based Canary input; shorter context
   can affect meaning or prosody. Record actual call count/runtime/resource usage.
4. Connect and separately evaluate a bounded faithful revision provider if needed.
   Current default CPU workers deliberately have none: an infeasible translation
   is diagnosed rather than automatically repaired. The pure callback tests do
   not prove that any translation model preserves meaning. A semantic check must
   cover facts, numbers, negation, names, temporal meaning and speaker intent.
5. Listen to source English and target German at corresponding times; expand to a
   ten-minute comparison only after a successful bounded test. Music/ambience,
   pronunciation and expressiveness remain unvalidated. The legacy subtitle-based
   CLI received safe fitting but does not yet consume the source-unit grouping path.
6. Update this handoff after each material implementation, test, release or listening
   result. Keep code correctness, timing feasibility and perceptual quality distinct.

Reproduce the combined local suite with Python/BLAS restricted to one core/thread,
512 MiB address space, and no model imports (the small FFmpeg fixtures are synthetic):

```python
tests = sorted(str(p) for p in Path('batch_dubbing').glob('test_*.py')
               if p.name != 'test_package_video.py')
tests += ['modal_control/tests', '../tests/test_supertonic_default.py',
          '../tests/test_dub_pipeline.py']
# Inside the bounded subprocess used for the recorded run:
subprocess.run(['.venv/bin/python', '-m', 'pytest', '-q', '-p',
                'no:cacheprovider', *tests], check=True, timeout=120)
```

## Current delivery and execution state

- Source branch: `modal-control`, created from local `main` at `9a3624e`.
- **General timing/fitting implementation:**
  `ba136640ad14b33db5c6a69d1be5af0a5ab7f360` —
  `Use source-linked speech timing and bounded natural audio fitting`.
  Push to `origin/modal-control` succeeded after the 229-test combined run.
  This code commit contains 38 source/config/test/documentation files; this
  periodic handoff update follows it as a separate documentation commit.
- The earlier viewer/handoff commit is
  `219feb5b130f1b6481cd6d2c6bb559fe936c80a3`. Concurrent subtitle work in `27d9a0a`
  and `c0708b9` was preserved; it is not part of the dubbing fix's test evidence.
- Initial source commit: `afcaaddcef8a7d89659eb159b6af1ec36424a08d` —
  `Add bounded Modal dubbing workflows and naturalness investigation`.
- Push to `origin/modal-control` succeeded; a separate `git ls-remote` confirmed
  that exact commit. Remote: `git@github.com:sarppp/ASR_Dubber.git`.
- That commit contains all 47 pending source/configuration/documentation files,
  including changes that predated this investigation, plus the small frozen test
  fixture and telemetry helper that were initially hidden by ignore rules.
- The diagnostic-only `plan_repair.py` remains an untracked local artifact;
  it was deliberately excluded from the general solution commit.
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
