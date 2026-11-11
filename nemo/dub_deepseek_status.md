# Subtitle dubbing — status (updated 2026-11-11)

## Fresh-run review accepted — 2026-11-11

The user watched the fresh-from-excerpt run
`new_guys/crispr/deepseek_t4_10cues_2026-11-05/local_synced_asr_2026-11-06/result/tower_synced_29__en_to_de/final_dub.mp4`
(fresh Ultra → Tower → Qwen clone, `continuous-synced`, nine cues, German audio,
embedded EN/default + DE tracks) and accepted it: "ok its good actually."
Nine cues instead of the reference's ten is expected from the shorter ASR
context and is not a subtitle-timing defect.

Merged to `main` the same day (accepted commit `b0bcc35` only; the opt-in GPU
alignment experiment and the single-clip H100 candidate are not included).
On the merged tree, 434 offline tests passed and
`batch_dubbing.reference_regression` against `crispr_29_synced_reference.json`
passed all nine exact checks versus `DUB_SYNCED.mp4` with zero model calls.

## Accepted reproduction and production defaults — 2026-11-06

**User review accepted:** “yes now it mathces!” The accepted new result is
`new_guys/crispr/deepseek_t4_10cues_2026-11-05/asr_checkpoint_synced_2026-11-06/result/tower_synced_reference_29__en_to_de/final_dub.mp4`
with its adjacent EN/DE subtitles. Its `comparison.json` verifies all nine exact
checks against `continuous_local/synced_result/DUB_SYNCED.mp4`: audio, full video,
and subtitle text, cue cuts, timestamps and embedded tracks. This is both an
artifact comparison and the user's explicit listening/viewing acceptance.

**Default now selects the accepted workflow:** Ultra → whole-passage Tower →
frozen paired captions → CPU source alignment → continuous Qwen Base cloning →
measured bounded synchronization → fresh CPU target alignment → aligned paired
and combined subtitles with the full video. The ordinary subtitle-dubbing CLI
selects Qwen clone, `continuous-synced`, and source audio replacement automatically.
Qwen's confirmed model revision is pinned even without a replay-wrapper override.
`--preserve-background` explicitly enables Demucs; `--tts-engine supertonic`
remains available. Legacy and standalone dubbing keep their earlier defaults.
All continuous-mode failures stop for review, including synthesis failures before
an alignment report exists; they cannot fall through to automatic text rewriting.

The default is this algorithm, not a fixed 29-second tempo or a copied subtitle
file. Exact benchmark replay retains the original 120-second ASR/Tower context;
fresh recognition of only the excerpt can yield different wording/cue cuts.
No whole-video trimming, global stretching, or cumulative context shifts.

Final validation for this default promotion: **447 tests passed in 27.87 s**
(the local `tests/test_*.py` suite plus parent dubbing gate, speech rate, quality,
subtitle pipeline, SRT, dub pipeline, cleaner, engine routing, paths, and runner
orchestration tests). This includes all three exact reference regressions and
new checks of ordinary CLI dispatch, explicit overrides, pinned Qwen loading,
and no rewriting after continuous synthesis failure. `git diff --check` passed.
GPU compute inventory is empty. No new model synthesis or Modal job was launched
for this default promotion. The historical T4 scaffold is explicitly pinned to
its older source-anchored mode; it is not the accepted replay or an active run.

Delivery branch: `feature/natural-pace-dubbing`. This commit includes the full
integration, regression fixtures/manifests, replay helper and current status;
local media/model caches remain excluded from Git.

## Authoritative target: DUB_SYNCED, integrated into ASR/Tower — 2026-11-06

User confirmed the intended version is
`continuous_local/synced_result/DUB_SYNCED.mp4` and requested exact reproduction
through the ASR pipeline and durable documentation. This supersedes both raw
`CONTINUOUS_GERMAN.mp4` and `cpu_passage_review/GERMAN_CPU.mp4` as the target.

The default `run_pipeline.py --pipeline subtitle-dubbing` route now selects
`--dub-timing continuous-synced`, still using source-only Ultra ASR and Tower.
Qwen cloning is the subtitle-dubbing engine default; `--voice-reference` can
select an existing reference, otherwise it is extracted from the source speaker. Continuous-synced adds the
missing measured Rubber Band slowdown (0.90–1.00, pitch unchanged), followed
by a second CPU alignment and reconstruction of the frozen paired subtitles.
It operates on bounded contexts at independent absolute starts, preserving
source gaps and the full video. It never slows down an entire long video.

The standalone original helper and production finishing route share the same
`fit_recording` function. Fixed-input regression through production finishing
passed **all nine exact checks** against the named DUB_SYNCED reference: raw PCM,
encoded AAC, full original picture packets/duration, both SRT bytes/cuts/timings,
embedded track layout and subtitle packets. Output:
`new_guys/crispr/deepseek_t4_10cues_2026-11-05/synced_workflow_regression/result/`.
Tempo is computed as 0.9378623620046065 from measurements, not hard-coded.

Exact steps, limits, pins and commands are in `batch_dubbing/SYNCED_WORKFLOW.md`.
Reference manifest: `batch_dubbing/crispr_29_synced_reference.json`.
**325 tests passed in 34.57 s**, including three explicitly named reference
variants, independent multi-context starts, preservation of source pauses,
realignment after stretching, pitch retention and strict active-overflow rejection.
Whitespace validation passed.

Additional real CPU validation in `synced_cpu_alignment_validation/` performed
two fresh forced-aligner calls (raw and adjusted audio), with no cached alignment
results or new TTS. All nine exact comparisons still passed against DUB_SYNCED.
The reference source context was traced and verified: its paired EN/DE cues
are the first ten Ultra/Tower entries from `auto_trial/review_10sent/CRISPR-120s`,
source 4.320–33.799 s, normalized for whitespace and rebased to the excerpt.
Standalone ASR on 29 seconds is a different recognition/translation context.

**Actual ASR pipeline reproduction completed in 92.99 seconds:**
`asr_checkpoint_synced_2026-11-06/result/tower_synced_reference_29__en_to_de/final_dub.mp4`.
It used the original hash-bound Ultra/Tower subtitle checkpoints, then made
**one fresh Qwen synthesis call and three fresh CPU alignments**. The newly
synthesized raw WAV matches `CONTINUOUS_GERMAN.wav` byte-for-byte; every one of
the nine final comparisons against `DUB_SYNCED` passes, including audio PCM/AAC,
video packets/duration and both external/embedded subtitle tracks. This is the
exact result the user reviewed and accepted. Its `comparison.json` identifies cached versus fresh stages.

A separate fresh-from-excerpt Ultra/Tower/Qwen run in
`local_synced_asr_2026-11-06/` also completed, but produced nine cues instead of
ten. Its native voice already fitted with a 0.343-second final acoustic gap.
The first attempt exposed an edge case: the preferred 100 ms margin demanded
1.0036× speed even though native speech fitted the 400 ms acceptance window.
Fixed it to keep 1.00× and retain final checks. The successful resume reused the
same generated speech; original failed logs remain. It is not an exact reference
match because its ASR/translation context differs.

The actual runner's output gathering now passes its explicit input/output
folders to the cleaner, preventing fallback to repository-wide defaults.
Final verification: **367 regression tests passed in 32.78 s**, plus **two
checkpoint-replay preparation/isolation tests**. `git diff --check` passed.
Local GPU inventory was empty and Ollama had no loaded models after completion.
No Modal work ran. The user-requested memory note records the confirmed recipe.

Repeatable actual-runner reproduction is now provided by
`batch_dubbing/replay_asr_checkpoint.py`: validate hashes, stage only original
ASR/MT checkpoints, then invoke `run_pipeline.py` with fresh local voice and
alignment. No reference audio/final subtitles are copied into its output.
Use the command in `batch_dubbing/SYNCED_WORKFLOW.md`; without `--run` it only
prepares and prints the concrete command. Exact comparison failures stay visible.

## Earlier raw-continuous selection — superseded by DUB_SYNCED

User: "continuous_local its what matters try to make it like this one".
This explicitly supersedes `cpu_passage_review` as the target. The new delivery
is `new_guys/crispr/deepseek_t4_10cues_2026-11-05/continuous_workflow_regression/result/`.

Rebuilt through the reusable production finish code from the selected raw voice
and cached word alignments, with no model/Modal run. All nine checks pass against
`continuous_local/CONTINUOUS_GERMAN.mp4`, `.wav`, and `.aligned.en/de.srt`:
PCM samples and duration identical (27.76 s), AAC packets identical, original
selected video packets and 29.487792-second duration preserved, both SRTs
byte-identical including ten cue cuts/texts/timestamps. The only intentional
packaging change is replacing the stale embedded German captions with the aligned
EN/DE tracks, English default. Extracted embedded tracks match their reference SRTs.

Default production routing now selects `--dub-timing continuous`. Bounded takes
stay whole at absolute source cue context starts; no passage splitting, edge
trimming, sentence relocation, time stretching or forced audio-end padding.
Complete video duration is preserved. Context overlap/overrun fails instead of
shifting later contexts or triggering a wording rewrite. Differences from internal
source speech timing remain explicit diagnostics. The selected take's roughly
2.08-second quiet ending remains, matching the user's chosen reference.
`--dub-timing source-anchored` retains the earlier selectable behavior.

The exact reference manifest is `batch_dubbing/crispr_29_continuous_reference.json`.
`continuous_workflow_regression/comparison.json` names every reference path and
records the PCM/packet/subtitle comparisons. The original files and previous
results are unchanged. This verifies finishing from fixed inputs; it does not
claim a new ASR/Tower/Qwen generation will be identical. No GPU was used.

Validation after selecting continuous delivery: **323 tests passed in 27.85 s**,
including both named real-media references, byte-exact raw-take preservation,
absolute starts for separate contexts, rejection of context overruns, and actual
production CLI routing to continuous mode. `git diff --check` passed.


## Reference clarification after direct folder comparison — 2026-11-06

The user compared `continuous_local/` with
`approved_workflow_regression/result/` and correctly noticed differences.
The earlier "exact match" claim referred specifically to
`cpu_passage_review/GERMAN_CPU.mp4` and its SRTs, NOT to
`continuous_local/CONTINUOUS_GERMAN.mp4`. That reference distinction must be
stated in handoffs; the two user-selected folders are not interchangeable.

Direct inspection confirms all ten EN and DE cue texts/cuts are identical in
the aligned external SRTs, but their timings differ. Relative to
`continuous_local`, the rebuilt result moves cues 1–4 approximately 74 ms
earlier, cues 5–6 exactly 1.040 s later, and cues 7–10 exactly 1.857 s later.
It reuses the same native voice recording but repositions three whole speech
passages at source anchors. It does not preserve the continuous take's original
audio timeline. Both picture streams have 707 frames and last 29.487792 s;
their compressed video packet hashes differ, so they are not byte-identical
picture streams. The rebuilt result copies the selected original source video.

There is also a playback trap: `CONTINUOUS_GERMAN.mp4` still embeds the OLD
German subtitle timings (matching `.original-timing.srt`), while its usual
external EN/DE SRTs were subsequently aligned to the continuous voice. The
rebuilt MP4 instead embeds its current EN/default and DE/selectable timings.

Evidence, including every cue and extracted embedded track:
`new_guys/crispr/deepseek_t4_10cues_2026-11-05/approved_workflow_regression/continuous_vs_rebuilt_audit_2026-11-06.json`.
Rechecking against the explicitly named CPU-passage reference still passes all
nine checks. No model ran and neither compared media/subtitle set was changed.

## Current result: approved video and subtitle workflow reproduced exactly

**This section supersedes earlier pending/integration claims below.** The
production finish code now freezes paired cue IDs/text/line breaks before TTS
and changes only their timestamps after German word alignment. It no longer
re-packs German words and independently redistributes English text after TTS.
This was the reason the final subtitle cuts differed from the approved preview.

A fresh CPU-only reconstruction was written to:
`new_guys/crispr/deepseek_t4_10cues_2026-11-05/approved_workflow_regression/result/`.
It used the original approved raw continuous recording, original ten paired
input cues, and cached CPU source/target alignments. No model inference or
Modal job ran, and no old final video/subtitle was copied into this directory.

`comparison.json` reports **all nine checks passing**:
- decoded PCM and encoded AAC packets identical to the approved recording;
- original video packets and full 29.487792-second timeline preserved;
- EN and DE SRTs byte-identical to `cpu_passage_review/GERMAN_CPU.en/de.srt`,
  including all ten cue cuts, line breaks, timings, BOM and CRLF;
- embedded EN/default and DE/selectable track layout and subtitle packets
  identical; German audio is correctly tagged `deu`.

The reusable check is `batch_dubbing/reference_regression.py`, with hash-bound
inputs and expected results in `batch_dubbing/crispr_29_reference.json`.
Run it with `--output` pointing to a new directory; it refuses stale evidence
and exits unsuccessfully on any mismatch. The local-media regression test
also invokes it automatically, skipping explicitly if media is unavailable.
The exact workflow and command are in `batch_dubbing/PASSAGE_PIPELINE.md`.

**Scope:** this proves the timing/caption/mux workflow from fixed approved
inputs. It does not prove a fresh ASR/Tower/Qwen run will produce identical
words or delivery. The latest fresh run had the same English lexical transcript
but nine cues rather than ten (different punctuation/packing), different final
German wording, and a 1.394-second ending gap. Its completed output is not an
exact reproduction; its post-TTS subtitle repacking is now superseded.
`approved_workflow_regression/upstream_comparison.json` records those differences.

Current production order: Ultra source words/cues → whole-passage Tower
translation with its original soft budget → CPU source alignment and frozen
paired captions → bounded continuous TTS → CPU target alignment → native
source-anchored placement → retimed unchanged paired captions and full-video
mux. One measured correction remains available by default; it is source-only,
guarded and bounded, but its lexical back-translation check does not guarantee
semantic equivalence. No automatic cutting or tempo changes hide a failure.

Verification after the cue-freeze and embedded-track fix: **319 tests passed**
in 14.19 seconds, including the real approved-media regression, source-locked
timelines, content gates, retry/cache behavior, output gathering and both legacy
TTS engine routes. The legacy engine test now uses a valid SRT/PCM fixture and
the current fit-report contract instead of mocking the obsolete parser signature.
`git diff --check` passed. The local GPU compute-process inventory was empty.
The latest comparison used CPU media processing only and made zero model calls.

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_*.py \
  ../tests/test_dubbing_gate.py ../tests/test_speech_rate.py \
  ../tests/test_dub_quality.py ../tests/test_subtitle_pipeline.py \
  ../tests/test_dub_srt.py ../tests/test_dub_pipeline.py \
  ../tests/test_clean_subs.py ../tests/test_supertonic_default.py
```


## Correcting the integration to reproduce the approved method — 2026-11-05

The earlier integration did not faithfully preserve the successful experiment:
it generated three separate takes, used raw ASR cue timing instead of forced
source alignment, and changed the translation prompt. The prior failure did
not prove the accepted result could not be automated.

`continuous_context.py` now groups whole same-speaker passages into bounded
30-second synthesis contexts. A 29-second input uses one continuous recording.
After generation, the complete take is word-aligned once and partitioned only
at measured quiet boundaries between complete passages. No active samples are
cut; passage placement still preserves the complete original video and native
tempo. Source forced alignment runs before synthesis. Contexts do not span
speaker changes or long gaps and never grow with the full video length.

**Measured regression:** the reusable `aligned_finish.finish` route recreated
the approved `cpu_passage_review/GERMAN_CPU.wav` with **bit-identical PCM** from
the same saved raw take, scripts and cached source/target alignments. No TTS or
model call was used for that check. Original video packet identity also passed.
Evidence: `continuous_method_regression/comparison.json` and `result/`.
This verifies placement reuse, not a claim that fresh speech is identical.

The new ASR execution had also split the short clip at 25 seconds, repeating
part of "transformative" across that boundary. Ultra now uses bounded 30-second
windows so this input remains a single ASR context. Forced source alignment
then recovered exactly the approved source anchors: 0.40–12.40, 12.64–17.76,
and 19.28–29.20 seconds.

The first fresh continuous run (`local_continuous_pipeline`) used the changed
unbudgeted Tower prompt and produced a 30.88-second take. It correctly failed
native placement. Restoring the original soft-budget Tower prompt on complete
source passages returned 83 words, matching the accepted recording's word
count, while keeping the final clause. The unsafe post-translation shortener
stays disabled. The local validation in `local_matched_pipeline` has finished; the current
section above records its differences and the later exact-reference check.

## Local validation outcome and omission fix — 2026-11-05

User cancelled Modal and instructed all remaining work to run locally. Modal
app `ap-fXDoYYnyQuXpJxrn1AQHCu` stopped during image preparation; verified cloud
container inventory was empty. No T4 model worker started. No more cloud work
was launched. Posted credit usage was unchanged; later build charges are unknown.

The real default production runner completed on the local RTX 2060 in
**162.44 seconds**, including fresh Ultra ASR, Tower translation, Qwen speech,
CPU word alignment and full-video mux. The report preserved all video packets
and duration (29.487792 s) at tempo 1.0, but the voice ended **3.3542 s early**.
Review found the existing automatic compressor had removed the final source
clause about hoped-for impact. That first result is explicitly rejected in
`local_default_pipeline/result/tower_local_29__en_to_de/quality_review.json`.
Timing success alone did not establish content preservation.

Fixed the default route to translate complete source passages without a speech
length budget and to retain Tower's returned wording without automatic
compression. This applies generically, not just to the example: the same
passage boundaries are used before translation and before synthesis. The
shortener can no longer silently remove a clause on this default route.
Regression checks cover complete dependent clauses and prove the default
content-preserving mode never calls the rewrite/compression model.

A fresh local validation is in `local_default_pipeline_v2/`. It retained the
final clause but was **rejected by the native timing gate**: passage 1 required
13.04 s for a 12.32 s window; passage 3 required 11.84 s for a 10.20 s window.
These are measured native take lengths; CPU alignment stopped on passage 1.
No accepted new video was published. No video/content cutting, speed change or
concealed timing drift was used. The earlier user-approved CPU preview remains
a distinct result, not evidence that fresh Tower/Qwen synthesis now matches it.

Final regression run: **639 passed, 1 skipped, 574 subtests passed** in 28.28 s.
Whitespace checks passed.

Both local calls used zero automatic retries. Raw takes, scripts, logs and
reports are preserved for diagnosis. Final GPU inspection found no inference
processes and no loaded Ollama models. The GPU is free of this task.

## Default Tower pipeline integration — 2026-11-05

Built on the current `feature/natural-pace-dubbing` checkout at `fdd0e12`
with the subsequent approved CPU timing work preserved. Live remote branch
heads were checked; no newer version of this working route was found.
`fdd0e12` fixed dubbing retries and delivered continuous German narration;
the Ultra/Tower subtitle pipeline already existed before that commit.

The production entry point now saves Ultra source word timings before TTS,
generates complete native passages with the selected existing engine, aligns
those recordings on CPU, and produces final dub-timed EN/DE and combined
SRT/ASS. Full video packet identity and duration are verified. Natural pauses
are retained; early endings are diagnostic; overlap/overrun blocks publication.
Source and generated-speech alignment are distinct stages. The approved CPU
preview reused prior speech; this validation generates everything afresh.

Offline verification: **637 passed, 1 skipped, 574 subtests passed** in 27.34 s;
48-file clean package import smoke; source compilation and whitespace checks.
No models or cloud calls were made by those smoke checks.

One explicitly authorized T4 validation was submitted through the controller:
`2026-11-05-d28b2148d3c2`, plan `mp-8a68f450f1f27a99e3f8fac2bdfa104a`.
Input: existing 29.487792-second source and same 8-second voice reference.
Actual production runner: Ultra -> Tower Plus 2B -> Qwen Base FP16 clone ->
CPU forced alignment -> full-picture mux and combined captions. One T4,
4 CPU, 24 GiB, worker 600 s, build 900 s, total wall 1500 s, retries zero.
The catalog pins source, inputs, Tower weights, model revisions and offline
smoke receipt. User cancelled Modal during image preparation and requested
everything locally. Controller cancellation completed; cloud inventory verified
zero containers and app stopped at 14:49:17 UTC. No T4 model worker started.
Posted account credit use was unchanged at $0.75; build charges may post later.

Local validation now uses the same actual production entry point on the RTX 2060
with cached models, Tower-only adaptation, the same reference, and CPU alignment.
Evidence: `local_default_pipeline/pipeline.log` and `request.json`.
Final outcome is recorded in the local validation section above.

This integration does not claim independent semantic/spoken-content proof,
speaker diarization, or validated 59-hour operation. The source-word batch API's
stricter content callbacks remain separate from the selected default route.

## Speaker timing clarification and playback — 2026-11-05

User judged the 29-second preview to seem fixed and clarified the criterion:
follow the speaker's speech, slow delivery, pauses and resumption; sentences
need not be contiguous or have identical source/target end timestamps.

Rechecked and reopened the **same** `cpu_passage_review/GERMAN_CPU.mp4`.
No voice regeneration or media change was necessary. Full duration and original
video packet identity still match; decoding succeeds. New evidence is in
`cpu_passage_review/speaker_timing_review.json`. The long source pause before
CRISPR is approximately 1.52 seconds; the corresponding dub pause is about
1.64 seconds. The 1.252-second first-sentence difference remains visible as a
diagnostic, not proof that the clip sounds wrong or a reason to stretch it.
Listening approval is distinct from automated timing feasibility.

`natural_pipeline.run_pipeline` now defaults to `end_gap_policy="warn"`;
`passage_batch` selects it explicitly. Early sentence endings produce warnings
without requesting revisions, changing rate, or moving subsequent passages.
Explicit `end_gap_policy="block"` retains strict testing when requested.
Overlaps, passage overruns and missing content evidence remain blocking.
This change concerns the source-word pipeline; the legacy CLI's explicitly
strict timing option is separate. Focused tests: **36 passed, 12 subtests**,
including preservation of a long pause and rejection of real conflicts.
No model, GPU or cloud jobs ran for this playback/check.

## 29-second CPU review delivered — 2026-11-05

User requested the existing 29-second example first, with CPU processing.
Freshly rebuilt review:
`new_guys/crispr/deepseek_t4_10cues_2026-11-05/cpu_passage_review/GERMAN_CPU.mp4`.
Matching `.en.srt` and `.de.srt` are next to it; both are embedded, English default.
The Windows player launch returned success. Visual playback was not independently
confirmed. This is a review preview, not an all-checks-passed batch artifact.

CPU alignment located the original CRISPR passage at 19.28 seconds, whereas the
previously liked native German take started it at 17.20 seconds. Rebuilding
whole passages at source anchors reduced the final quiet gap to **0.2195 seconds**;
German acoustic activity now ends at **29.2683 seconds**. Full video duration
remains **29.487792 seconds**, video packet hashes match, decoding passes,
all 83 German words remain in the aligned captions, and every raw sample above
-55 dB is retained. Tempo is **1.0**. No whole-clip slowdown was used. Only quiet
recording edges and inter-passage placement changed; source passage shifts
stay within 250 ms. The previous voice recording was reused, not regenerated
for this delivered preview.

**Unresolved, explicitly reported:** the first passage ends **1.252 seconds
early**. The preview is labeled `review_with_timing_warnings`, not fully fixed.
Both supplied texts end mid-sentence because the selected source excerpt does.
No continuation was invented. `report.json` and `README.txt` disclose these
limits; the earlier globally slowed preview is a different artifact.

One additional native CPU Qwen experiment completed: cached
`Qwen/Qwen3-TTS-12Hz-1.7B-Base`, pinned revision
`fd4b254389122332181a7c3db7f27e918eec64e3`, float32, four threads, seed 20261106,
one 1.2-second smoke plus one 28.72-second full take, 667.82 seconds total.
An initial environment import failed before model loading; setting a writable
Numba cache resolved it. No synthesis retry occurred. Independent cached
Qwen3-ASR-0.6B on CPU flagged `veröffentlichen` versus supplied `veröffentlichten`.
CPU word alignment and source-anchor placement rejected this new take because
one passage could not fit with the allowed shift. It was **not substituted**,
clipped, or stretched. Evidence and executable reproduction scripts are under
`new_guys/crispr/deepseek_t4_10cues_2026-11-05/cpu_native_review/`.
All inference processes finished. No GPU or cloud jobs ran.

## Implemented source-anchored timeline and batch checks — 2026-11-05

The existing production stitcher now preserves the **full original video**
instead of trimming it to the final subtitle. Clips use absolute source
positions; missing clips and overruns fail instead of silently shifting later
speech. Strict parsing rejects malformed/dropped subtitle blocks. The subtitle
runner requests native speech speed and rejects early-ending passages.

The source-word batch implementation reuses the existing passage builder,
natural pipeline, scheduler and renderer. It requires every supplied source
word exactly once, rejects incomplete passages before synthesis, keeps one
continuous native take per passage, blocks timing/content failures, and aligns
target SRT to accepted speech. Assets run serially with durable reports and
hash-checked reuse of unchanged recordings. No global slowdown is used.
Independent ASR and source-to-target meaning receipts are required before
publication; forced alignment alone is not accepted as content verification.

Implementation and input/provider contracts:
[batch_dubbing/PASSAGE_PIPELINE.md](batch_dubbing/PASSAGE_PIPELINE.md).

Executed model-free scale preflight:
`new_guys/source_locked_scale_smoke_2026-11-05/preflight.json` — **59 synthetic
one-hour assets, 35,400 passages, 70,800 source words, zero model calls**.
Tests also verify every absolute placement across those 59 synthetic timelines
and use actual FFmpeg muxing to compare unchanged video packets/duration.
The full regression suite passed **624 tests** (one skipped, 574 subtests).
After final video-identity and source-receipt checks, the focused suite passed
**43 tests and 19 subtests**, including two additional regression cases.
Missing/changed video is rejected before synthesis; content receipts bind the
exact source text as well as target text and raw speech.

**Delivery boundary:** the production trim/drift fixes are wired in. The strict
source-word batch API and offline CLI are implemented and tested, but live
synthesis/recognition/meaning/alignment adapters still require connection
through the selected model controller. Neither 59 real videos nor a real
hour-long dub has been validated. Supplied-word coverage does not establish
that upstream ASR missed no speech. No GPU or cloud work ran for this change;
the previous short-clip experimental artifact was not replaced.

## Scale requirement — 59 separate videos of approximately one hour each

User clarified that the intended workload is **59 separate hour-long videos**,
not shortening a one-hour video to 50 minutes. The previous response
misunderstood this. A successful 29-second preview is not evidence that the
workflow is suitable for this batch.

The global slowdown in `sync_continuous_dub.py` is a bounded short-passage
experiment, **not the production duration-matching strategy**. It currently
rejects inputs longer than 60 seconds. That guard does not make blindly
splitting long videos into 60-second pieces a valid solution. Do not enable
this experiment as a batch default or run the 59-video workload without a
separately scoped implementation and validation.

Requirements for the automatic workflow:

- Preserve the complete source picture, scene order, total timeline, and
  meaning. Never remove scenes or words to satisfy timing.
- Detect complete speech passages and speaker turns from source audio;
  subtitle line breaks and arbitrary fixed-size chunks are not speech units.
- Generate connected speech within those source-anchored passages, retaining
  real pauses and turn boundaries. Maintain absolute source timestamps so
  timing differences cannot accumulate across an hour.
- Check duration fit, content coverage, overlaps, gaps, and speaker turns for
  each passage. Use bounded source-based wording/delivery adjustment when
  needed; report unresolved passages instead of silently clipping, globally
  stretching an hour, or forcing every passage to fill its entire window.
- Align subtitles to the final accepted speech. Validate representative long
  videos and cumulative drift before making any claim of 59-video readiness.

The earlier short-clip artifacts remain a local comparison. The implementation
and its validation boundary are recorded above. No real long-video or batch
model run was started in response to this clarification.

## Short-clip experiment — automatic continuous-dub duration matching

Both the original three-step method (continuous TTS, CPU word alignment,
subtitle reconstruction) and the reported silent ending are recorded below.
User requested an automatic fix with no manual timing edits. Implemented
`batch_dubbing/sync_continuous_dub.py` and ran it successfully on the short
review. The previous raw continuous take and all original inputs remain
unchanged.

**Latest playable output:**
`new_guys/crispr/deepseek_t4_10cues_2026-11-05/continuous_local/synced_result/DUB_SYNCED.mp4`.
Companions: `DUB_SYNCED.en.srt`, `DUB_SYNCED.de.srt`, and `DUB_SYNCED.wav`.
The MP4 embeds both newly aligned subtitle tracks, with English marked as
default; this version does not contain the stale embedded timings.

Automatic steps, with no clip-specific word times or manually chosen speed:

1. Validate the existing alignment against the recording hash and exact
   German words. Measure the passage's target end from source subtitles and
   the actual video duration. Detect conservative audio activity at -55 dB.
2. Derive a single whole-passage tempo from the later of the last aligned
   word and last detected audio activity, leaving 100 ms of ending margin.
   Permit only **0.90–1.00×** tempo; fail for review outside that range.
3. Apply FFmpeg Rubber Band with unchanged pitch to the entire continuous
   recording. Preserve its internal pauses; never cut it at subtitle lines.
   Fit to video length only after checking overflow samples are below
   -55 dB. Preserve the complete stretched WAV as evidence.
4. Realign the adjusted speech with the cached Qwen forced aligner on CPU,
   rebuild both SRTs from the measured German word spans, and reject a
   remaining ending gap above 400 ms. English uses corresponding cue spans.
5. Mux German audio and both subtitle tracks while copying the original
   video packets unchanged. Verify full decoding, video packet hashes,
   original input hashes, and output duration. Save plan and result JSON.

Measured result: **0.937862362×** tempo (about 6.2% slower, unchanged pitch),
video **29.487792 s**, audio **29.488 s**. Last aligned word ends at **29.28 s**;
-40 dB acoustic activity ends around **29.225 s**. The previous approximately
**2.076 s** quiet ending is reduced to **0.263 s** within the video. Only
**111 ms** of near-silent stretched overflow was removed (peak 0.000061,
about -84 dBFS); no trailing silence was added. All **83 words / 10 cues**
were realigned. Zero TTS calls, one CPU alignment call, no GPU or Modal use.
The decoded AAC also shows final quiet beginning at approximately 29.225 s.

**Validation:** 22 focused offline tests passed before execution, including
an actual FFmpeg stretch/mux/decode smoke with mocked alignment. A 440 Hz
fixture retained its pitch; excessive tempo changes, active audio truncation,
input mismatches, invalid word timing and output overwrites are rejected.
The real run's report and artifacts passed the same structural checks.

Reusable command (run from `nemo` with the existing Qwen alignment environment):

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 .venv/bin/python -m batch_dubbing.sync_continuous_dub \
  --video SOURCE.mp4 --audio CONTINUOUS_TAKE.wav \
  --spoken-srt GERMAN.srt --english-srt ENGLISH.srt \
  --alignment WORD_ALIGNMENT.json --output NEW_REVIEW_DIRECTORY
```

This is a reusable command for one continuous passage up to 60 seconds, not
a change to the production runner's default pipeline. Longer videos need
semantic passage selection upstream. It fixes the measured ending-duration
mismatch; exact lip sync, meaning verification and incomplete source phrases
remain separate. The supplied final sentence is still incomplete. Listening
review of the mildly slowed version remains with the user.

---

## Earlier handoff — continuous local dub and English comparison subtitles

The latest review is the **single continuous German take**, not the earlier
stitched `GERMAN_DUB.mp4`. User authorized local execution with “run it in our
gpu”; no Modal job was launched for this correction.

- **Input:** the same ten `input.de.srt` texts, joined with spaces into one
  request. No translation changes or new punctuation. Six of the old
  per-cue boundaries were inside sentences, explaining the voice resets.
- **Execution:** cached `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, revision
  `fd4b254389122332181a7c3db7f27e918eec64e3`; FasterQwen 0.2.5 on the local
  RTX 2060, FP16, sequence limit 1024, token limit 600. One synthesis call,
  five-minute deadline, completed in **69.09 seconds**.
- **Delivered locally:** under
  `new_guys/crispr/deepseek_t4_10cues_2026-11-05/continuous_local/`:
  `CONTINUOUS_GERMAN.mp4` (29.487792 s), `CONTINUOUS_GERMAN.wav`
  (27.76 s, mono 24 kHz PCM16), and `CONTINUOUS_GERMAN.srt`.
  Full MP4 decode passed; there are no inserted per-cue pauses, speech cuts,
  speed changes, or final-frame extension.
- **English comparison:** user explicitly requested **only an SRT** to load
  on the new German video. Delivered `CONTINUOUS_GERMAN.en.srt`, ten original
  English cues, UTF-8 BOM and CRLF. After the timing repair below, both SRT
  files follow measured German word spans; English retains the corresponding
  cue text. The embedded MP4 subtitle track still has its original timestamps. The extra English captioned video made before that clarification
  is not the requested handoff.
- **GPU released:** synthesis process exited; the recorded check found no
  compute processes and 5514 MiB free. See local `gpu_released.json`.
- **Limits:** user liked the aligned result; the ending gap and semantic
  review remain open. The supplied
  text ends mid-sentence at “dass sie das Leben”; that was preserved. Existing
  translation problems such as “viskales Gefühl” were not silently rewritten.

### Subtitle timing repair after user review

The user found captions lagging behind the continuous voice. A single local
CPU pass of cached `Qwen/Qwen3-ForcedAligner-0.6B`, revision
`c7cbfc2048c462b0d63a45797104fc9db3ad62b7`, aligned all **83 German words**
in **20.44 seconds**. CUDA was disabled; no GPU or Modal call was used.
Six offline retiming regression tests passed first.

Delivered `CONTINUOUS_GERMAN.aligned.en.srt` and
`CONTINUOUS_GERMAN.aligned.de.srt` beside the video. The usual
`CONTINUOUS_GERMAN.en.srt` and `CONTINUOUS_GERMAN.srt` were updated too;
old sidecars are preserved as `*.original-timing.srt`.
All ten cue texts are unchanged, cue timings are monotonic/non-overlapping,
and both languages use the measured German cue spans (80 ms display lead,
up to 120 ms tail, clipped before the next cue). For example, cue 8 moves
from 20.840 to 18.160 s; the last cue ends at 27.560 s. English text retains
its original cue correspondence rather than independently aligning English
words to German audio.

WAV and MP4 hashes are unchanged. **Load the external aligned English SRT**;
the embedded MP4 track has old timestamps because the user requested only
subtitles. Evidence: local `subtitle_alignment_words.json` and
`subtitle_alignment.json`. Forced alignment locates supplied words; it is
not independent ASR or verification of the translation's meaning.

### User review and remaining silent ending

User feedback after loading the aligned English SRT: “its very good”; the
remaining complaint is that the woman still speaks on screen during the
last roughly three seconds while the German voice has finished. This is
positive listening/timing feedback for this short example, not a general
semantic or lip-synchronization validation.

How the improvement was produced:

1. Synthesize the ten selected German cue texts as **one continuous passage**
   with Qwen TTS; SRT boundaries do not become TTS calls or forced pauses.
2. Feed the finished WAV and the exact supplied German text into cached
   Qwen3-ForcedAligner on **CPU**. This locates word start/end times in the
   existing speech without generating another voice or translating again.
3. Group those word times using each German cue's word count. Show each cue
   80 ms before its first aligned word, keep up to 120 ms after its last
   word, and cap at the next cue's onset to avoid overlaps.
4. Apply the same spans to the corresponding unchanged English cue texts.
   Write BOM/CRLF SRTs, retain the old timing backups, and verify all ten
   cues, exact text preservation, monotonic times, and unchanged audio/video
   hashes. English-German cue correspondence is inherited, not a separately
   verified word-for-word translation alignment.

Implementation: `batch_dubbing/align_continuous_subtitles.py`;
regression coverage: `tests/test_align_continuous_subtitles.py`.
The runner refuses to replace an existing alignment report; preserve and
inspect its evidence before considering another inference pass.

The ending gap is real and separate from subtitle timing. FFprobe confirms
video **29.487792 s**, audio **27.760000 s**: the audio stream ends **1.727792 s**
before the picture. FFmpeg silence detection at -40 dB (minimum 150 ms)
estimates trailing quiet from **27.4117 s**, leaving approximately
**2.0761 s** of picture after audible speech. That threshold is an acoustic
estimate, not a manually verified phoneme boundary. The original final cue
also ends mid-sentence (“for impacting” / “dass sie das Leben”).

At that checkpoint no media was changed. Retiming captions
makes them follow the generated speech; it cannot extend the speech to match
the source speaker's visible ending. Remaining work is source-scene duration
matching with a complete phrase, while preserving continuous delivery.
Do not mark the short trial as fully synchronized dubbing yet.

### Source changes included in this push

Source-linked retry fixes, measured timing calibration, BOM handling,
identity-checked raw TTS reuse, and complete fit reports are accompanied by
regression tests. The fitter handles filesystems without hard links using
exclusive creation. The continuous helper and local/Modal runners read one
passage instead of generating individual subtitle entries. The controller
can verify a pinned **offline smoke receipt** without a separate cloud
allocation; the old prepared Modal plan is unused and its receipt is stale
following later source/test edits, so it must not be executed as-is.

The pending phrase/dialogue trial changes and test-archive moves are included
with this source handoff. Archive allowlists keep generated media local;
compatibility links preserve existing worker and fixture paths. Continuous
smoke tests now use a tracked SRT fixture and temporary mocked media, so they
do not depend on the ignored local review files. Trial catalogs reference
local input artifacts and are not standalone downloadable datasets.

Validation before push: **509 passed, 1 skipped, 574 subtests passed** across
continuous-generation, DeepSeek retry/transport, controller, batch-dubbing,
subtitle-pipeline, speech-rate, gate and quality suites. These were offline
checks with mocked inference; no additional GPU or cloud job was launched.
The staged diff also passed whitespace checks and a credential-pattern scan.
A clean export of the staged source, without local review media, passed the
focused generation/retry/transport/controller suite: **96 passed, 1 skipped,
56 subtests passed**.

All earlier sections below are historical checkpoints; their “open” or
“not generated” statements do not supersede this current handoff.

---

## Playable German dub delivered from the saved T4 takes

`new_guys/crispr/deepseek_t4_10cues_2026-11-05/GERMAN_DUB.mp4` now contains
the actual German voice track and embedded German subtitles. The exact ten
texts from `input.de.srt` are unchanged. `GERMAN_DUB.srt` follows the actual
speech onsets; `GERMAN_DUB.manifest.json` records the timing and verification.

All ten original PCM takes were preserved byte-for-byte in the assembled
audio at tempo 1.0. Cues were delayed where needed to avoid speech overlap.
The source video's last frame is held for approximately 2.4 seconds to retain
the complete final utterance: audio 31.880 s, MP4 31.907 s. This is a playable
native-speed review, **not a fix for fitting the original 29.479-second slots**.
Full video/audio decoding passed. No additional model inference or T4 run was
used for this delivery.

---

## 2026-11-05 — short T4 validation result

User selected Modal/T4 and a short test of ten SRT entries. The exact first ten
German entries cover source time 4.320-33.799 s: **29.479 seconds**, not the older
approximately 98-second ten-block test. Some entries are sentence fragments.
Existing wording and cue boundaries were preserved, with merging disabled.

- All **10 raw utterances** were generated on Tesla T4 with the pinned Qwen
  1.7B Base clone model. The first two smoke takes were reused byte-for-byte;
  only eight new utterances were generated for the full test.
- Independent local checking of those downloaded WAVs gives **3 fitting cues
  (4, 5, 6), 7 overlong cues (1, 2, 3, 7, 8, 9, 10)** at maximum tempo 1.0.
  Raw speech totals 31.84 s. No final dubbed video was published.
- The first controller attempt stopped because the transport report lacked
  the required top-level `status`. The correction preserved a separate
  `dub_status: needs_review`; it did not relabel fitting failures as success.
  Remaining controller deadlines were reduced to keep combined stage bounds
  within the original 600 seconds.
- The T4 workload exposed a separate storage issue: Modal volume hard links
  raised EPERM for fitted receipts. `fit_audio.py` now falls back to exclusive
  file creation without overwriting existing artifacts. The fix is verified
  offline with simulated unsupported hard links, collisions and partial writes;
  it has **not** been rerun on Modal. The three affected takes pass the actual
  fitter locally. Original remote reports/logs remain intact.
- Offline wrapper tests now exercise the actual transport function and
  controller completion contract with fake synthesis, including ten-cue output,
  missing-audio rejection and wrong-GPU rejection. Together with fitting and
  retry regressions: **38 tests + 4 subtests pass**, no model loaded.
- All owned Modal containers were confirmed stopped. Posted owned-app costs
  were **$0.04349934**; billing can still lag. Unrelated apps were untouched.

Evidence directory: `nemo/new_guys/crispr/deepseek_t4_10cues_2026-11-05/`.
`verified_fit_report.json` is the final local audit of real T4 audio;
`workload/` preserves the remote result; `cloud_closeout.json` records shutdown
and billing; `smoke_reuse_verified.json` records byte identity.

**Latest user boundary:** do small local/offline smoke tests for every component
first, then only one bounded T4 run for the result. No further T4 execution was
performed after that instruction. Any next attempt must follow that sequence,
rather than using paid smoke/retry runs to discover wrapper errors.

---

## 2026-11-05 — verified repair, fresh model run still pending

This section supersedes the historical claims below that all offline checks
passed and that all real speech fitted. The checkout initially failed its own
offline harness. The current repair passes **213 focused pytest tests**,
including 20 new regressions in `nemo/tests/test_deepseek_retry.py`. The standard
`python3 tools/check_dubbing.py --offline` harness includes those regressions
and passes without loading any model.

Confirmed bugs and fixes:

- **Missing opening speech:** the pipeline writes UTF-8 with BOM, but
  `qwen3-tts/dub_srt.py` read plain UTF-8 and silently skipped cue 1. A two-cue
  reproduction returned only cue 2. Reading `utf-8-sig` preserves both.
- **Wrong sentence rewritten:** `_row_at()` expanded sentence boundaries by
  300 ms and selected the preceding row at touching/nearby boundaries. Exact
  onsets now take priority; an unmatched timestamp does not select a distant
  sentence. A merged failure spanning several rows is left unresolved rather
  than charged to only one row.
- **Compressor bypassed:** `_adapt_from_report()` accepted unchanged or longer
  translations and recorded them as changes. It now rejects those and tries
  the compressor when a candidate is absent or still above the measured budget.
  Retry estimates are labelled `estimate_pending_tts`, not measured passes.
- **Lost source:** custom subtitle-language selections could omit the source.
  Source SRT generation is now mandatory for adaptation, and retry rejects a
  missing source explicitly. The documented `None` traceback was not available;
  it is not claimed as reproduced. A full mocked pipeline retry verifies that
  the actual source path reaches the retry.
- **Unstable retries:** the whole-script cache hash forced unchanged lines to
  be synthesised again after a text edit. A separate raw-speech cache now checks
  exact text, speaker, voice/reference, language, engine and worker identity,
  plus audio integrity. Only changed speech is synthesised; reused speech is
  fitted again for its current time window. Historical checkpoints are retained.
- **Misleading evidence:** fit reports now include passed, failed and cached
  segments, expected input segments, script hash and cache directory. The timing
  checker rejects missing cues, pending runs and ambiguous cache histories.
  Successful sidecars alone cannot establish that every segment fitted.
- **Calibration and limits:** the calibrated rate is now actually used by the
  gate. Calibration requires a matching report and includes failed fits; it no
  longer pools historical success-only sidecars. The dub command respects
  `--fit-target 1.0` even when `--max-speed` retains its 1.15 default. Translation
  models are unloaded before TTS; compressor rounds check against the original
  text so repeated small content losses cannot compound past the guard.

Verification:

```bash
python3 tools/check_dubbing.py --offline
qwen3-tts/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_dubbing_gate.py tests/test_speech_rate.py tests/test_dub_quality.py \
  tests/test_subtitle_pipeline.py tests/test_dub_srt.py tests/test_dub_pipeline.py \
  nemo/tests/test_deepseek_retry.py
```

The integration regression runs the actual dub orchestration with a fake voice
and fitter: first attempt fails one sentence and refuses publication; the retry
synthesises only the changed sentence, reports both sentences, and passes.
A fully cached resume starts no voice worker.

Rechecking the existing CRISPR artifacts now returns `ok: false`, with
`failed speech segments: 4, 9` and
`ambiguous cache history; no single current run identified`. The saved report
still records required tempos **1.231** and **1.139**. No existing audio was
regenerated, and no vendor/cloud inference was performed during this repair.

Remaining verification: a fresh bounded Qwen run at 1.0x, followed by review of
the actual speech and meaning. The live worker selects **Qwen3-TTS 1.7B**
(Base for clone; CustomVoice for fallback), so the historical 0.6B benchmark is
not the same configuration. Character/number/language heuristics do not prove
semantic preservation, and these code fixes do not guarantee every sentence
can be shortened enough. ASR/AGENTS.md requires a reviewed Modal release or an
explicit user override before local TTS inference.

---

Pipeline: **subtitle-dubbing** = Parakeet Ultra ASR → cue packing → whole-sentence
Tower translation → deterministic split → gate + adaptation → Qwen dub.
Legacy flow (NeMo + Canary/Gemma) is `--pipeline legacy-dubbing`.

## What is done and verified

| Piece | State |
|---|---|
| Ultra + Tower subtitles (EN/DE/FR, combined `.multi.srt/.ass`) | working, measured |
| `.venv-local-asr` on torch 2.8.0+cu128 + moondream (Ultra in-process, no Docker) | working |
| Vendor telemetry (moondream/kestrel) stubbed | working |
| Sentence-level dub script (one cue = one complete sentence) | working |
| Gate + adaptation before the dub | working, but timing not yet reliable |
| Qwen clone dub on 120 s / 10 sentence blocks | publishes only when every block fits |

## Fix 1 — "rewrite this German" broke the adapter

**Symptom.** Asking Tower to rewrite a German line produced an **English** line,
then degenerated into repetition (`und manchmal, und manchmal, …`), ratio 13.6×.

**Cause.** Tower is a translation model, not a rewriter. Given a German input
plus an instruction, it re-translates, and its language choice follows the
instruction/prompt context, not the input language.

**Fix.** Never adapt the target text. Re-translate the **source English**
sentence with a character budget:

```
translate_sentence(source_english, "en", "de", max_chars=budget)
```

**Fix (guards).** Every candidate is rejected, and the original line kept, when:

- a number from the original is missing,
- a negation marker (`nicht/kein/nie/ohne/…`) is missing,
- it is shorter than 55 % of the original (possible dropped content),
- it is longer than 2× the original,
- it repeats itself (trigram variety < 1/3).

Rejections are reported, never silent: the line keeps full meaning and shows up
in `adaptation.md`.

## Fix 2 — the dub started mid-sentence

**Symptom.** A segment began on `"ist, dass es hin und wieder …"`, i.e. the
voice started in the middle of a sentence.

**Cause.** `dub.py` merges cues upward from whatever cue it starts on; our cue
boundaries are word-timestamp slices, so a merged block could begin mid-sentence.

**Fix.** The dub now receives a **sentence-level** script: one cue per complete
sentence, span = first cue start → last cue end, text = full sentence. In
addition `qwen3-tts/dub_srt.py::merge_segments` no longer merges past a completed
sentence.

## How I fixed the "rewrite the German → returns English → loops" problem

1. Stopped rewriting the target text. The adapter no longer says "rewrite this German line". It goes back to the source English sentence and translates it again with a character budget:
translate_sentence(source_english, "en", "de", max_chars=budget)
A translation model asked to "rewrite" was re-translating, and its language followed the instruction rather than the input — that's how English came out.
2. Budget instead of a vague goal. budget = chars_now × (target/ratio) × 0.95, recomputed after each pass. Two bounded passes; it stops as soon as the block is under the aim.
3. Guards that reject bad candidates and keep the original, so meaning can't silently disappear:
- missing number,
- missing negation (nicht/kein/nie/ohne, pas/rien, not/never),
- shorter than 55% of the original,
- longer than 2× the original,
- repetition loop (trigram variety < ⅓) — this is what catches und manchmal, und manchmal….
A rejected line keeps its full meaning and is marked in adaptation.md instead of being silently replaced.
4. Sentence-level dub script (fixes the other thing you noticed): the dub now gets one cue per complete sentence, and merge_segments no longer merges past a finished sentence — so the voice starts at a sentence start, not on "ist, dass …".

## Measurements (120 s clip, target DE, Qwen clone, `--max-sentences 10`)

| Run | Total wall | Blocks failing the 1.0× guard |
|---|---|---|
| supertonic | 43–63 s | 3 → 2 (after first gate) |
| qwen (10 sentences) | 137–177 s | 2–3, worst 1.17× |
| qwen, sentence-level script | 149 s | 3, worst 1.19× |

Subtitle step ≈ 25 s, gate+adaptation ≈ 25 s, Qwen TTS + mux ≈ 100–125 s.

German needs ~1.2× the English slot at natural pace; measured totals:
`CRISPR-auto.srt` 38.2 s speech / 31.2 s span = 1.22×; our 80 s track 1.09–1.22×.

## Open problem (the real blocker)

The gate estimates speech duration from characters (`11.6–13.5 chars/s`), but the
real TTS differs per engine, per sentence, and per run: failures land at
1.02–1.19× even when the estimate said ≤ 0.92×. Adjusting the constant or the
safety margin does not converge.

**Next step (recommended):** stop estimating and measure. Add a JSON fit report
to `dub.py` (segment index, span, actual speech seconds, required tempo), then:

1. run the dub once,
2. read the report,
3. re-adapt only the failing sentences with the **measured** ratio,
4. re-run the dub (TTS checkpoints for unchanged segments are reused).

Two dub passes should converge; the guard stays at 1.0×.

## Testing plan

Here's how I'd test it, cheapest and most decisive first.
L0 — offline, seconds, no models (extend tests/test_subtitle_pipeline.py)
- guards: number dropped, negation dropped, 40% shrink, repetition loop → all must reject and keep the original.
- sentence grouping: every group ends on .?! or hits the 6-cue/30 s cap.
- SRT integrity: monotonic times, no overlaps, cue text joined == sentence text, no cue > 84 chars.
- merge_segments never joins across a completed sentence.
- report: every block has before/after ratio and text.
L1 — fake TTS, no GPU (the one that would have caught our bug)
- stub the voice as duration = f(text) with a deliberately pessimistic rate; run gate → adapt → gate.
- assert: converges within 2 passes, final ratio ≤ 1.0, and the loop stops if it can't.
- add a stub whose rate varies per sentence by ±20% — that mimics Qwen and would expose the estimate problem immediately.
L2 — measure-then-adapt loop (what I recommend building next, and testing)
- dub.py writes fit_report.json with measured speech seconds per segment.
- test: fake report where 3 of 10 segments need 1.05–1.19×; pipeline re-adapts exactly those, second pass passes, unrelated segments untouched (TTS cache reused).
L3 — round-trip ASR on the real dub (automatic meaning check, no human)
- take final_dub.mp4, strip audio, transcribe the German with Ultra again.
- compare to the adapted SRT: WER, and numbers/names must survive exactly.
- back-translate that German to English with Tower and compare to the source English (embedding similarity or COMET) — this measures whether adaptation kept the meaning, which is your hard requirement.
- this is the test that catches "shortened into nonsense", not just "fits the time".
L4 — timing/onset assertions (cheap, on the wav)
- for each segment: speech_seconds / span ≤ 1.0, and onset within ±60 ms of the cue start — that's the "cuts the sound at the beginning" bug, measurable with ffmpeg/wave alone.
- no gap < 0.2 s between segments (choppiness).
L5 — listening, small and blind
- 6 clips × 3 versions (Tower plain, adapted, Ling reference) shuffled, rubric 1–5 on meaning / fluency / sync / naturalness.
- clips chosen for failure modes: a number sentence ("14 years"), a negation, a name ("Douda, CRISPR"), a long compound, fast speech, and one with music (demucs on).
Practical shape — one harness tools/check_dubbing.py with flags:
--offline          L0+L1, ~5 s, no GPU
--clip X --smoke   L2+L3+L4 on a 15 s clip, ~1 min
--batch dir/       runs the 6-clip matrix, writes metrics.json
Metrics to keep per sentence: span, measured speech, tempo, chars before/after, WER after round-trip, back-translation similarity. Then regressions become numbers, not impressions.
My suggestion: implement L0+L1 now (fast, no models), then L2 with the fit_report.json change, since that also fixes the open timing problem. L3 is the one that actually proves "meaning preserved" and is worth doing right after.

## How to run

```bash
uv run python run_pipeline.py --target-lang de --language en \
    --input-file <clip.mp4> --no-demucs --tts-engine qwen --max-sentences 10
```

Artifacts of a run: `<clip>.adaptation.json/.md` (per sentence: span, ratio
before/after, text before/after), `<clip>.nemo.<lang>.dub.srt` (what was spoken),
`<clip>.multi.srt/.ass` (viewing), `end_product/<label>/final_dub.mp4`.


## Test harness status (2026-11-04, later)

```bash
python3 tools/check_dubbing.py --offline     # L0 + L1, seconds, no GPU
```

| Level | State |
|---|---|
| L0 guards / SRT integrity / merge boundaries / report | implemented, passing |
| L1 fake TTS convergence + honest failure | implemented, passing |
| L2 fit report + re-adapt loop | implemented, tested offline; real run does not converge yet |
| L3 round-trip ASR (meaning) | not implemented |
| L4 timing/onset assertions | not implemented |
| L5 blind listening | manual, not started |

Files: `tests/test_dubbing_gate.py` (13 tests), `tests/test_subtitle_pipeline.py`
(3 tests), `tools/check_dubbing.py`.

### L2 result on the real 120 s clip (Qwen)

1. First dub: 3 sentences over 1.0x (1.04, 1.07, 1.21).
2. `dub.py` wrote `fit_report.json` with measured seconds per segment.
3. Pipeline re-adapted exactly those three from the source with a budget.
4. Second dub: **worse** — 1.13, 1.27, 1.26.

Two things surfaced:

- **Cache bug (fixed).** `dub.py`'s checkpoint only checked that the clip file
  existed, not that the text was unchanged, so an edited line re-used its old
  take. The pipeline now drops the checkpoint entries (and clips) for edited
  sentences before the retry.
- **Tower ignores the budget.** Asked for 143 characters it returned 183
  (28% over); asked for 41 it returned 48. Prompt-only length control is not
  reliable on a translation model, so the measure loop cannot converge.

To reach true 1.0x, the compressor must be a different component:

- a length-controllable MT (vLLM/transformers build, not this GGUF prompt), or
- a dedicated rewriting model small enough to fit beside Tower, or
- accept a small speed-up (1.05-1.10) for the few lines that overshoot.

Implemented and passing
- tests/test_dubbing_gate.py — 13 tests: guards (number, negation, 40% shrink, repetition loop), sentence grouping (.?! and 6-cue cap), SRT monotonicity/overlap/text, merge_segments never crossing a finished sentence, fake-TTS convergence, honest failure when the voice ignores the budget, original kept when a candidate drops a number, and L2 report-driven selection (only the failing sentences re-adapted).
- tests/test_subtitle_pipeline.py — 3 tests, updated.
- tools/check_dubbing.py --offline → runs all 16 in seconds, no GPU.
L2 (measure-then-adapt) — built, with a real-run verdict
- dub.py now writes fit_report.json: segment index, span, measured speech seconds, required tempo.
- Pipeline re-adapts exactly the failing sentences and retries once.
- Found and fixed a real cache bug: the checkpoint only checked that the clip file existed, not that the text was unchanged — an edited sentence silently reused its old take. Edited entries are now dropped before the retry.

## 2026-11-05 — the diagnosis above was wrong

Everything above concluded that the loop could not converge "because Tower
ignores the character budget". That was the wrong cause. The translator was
blamed for a bug in the **measurement**. Two constants were wrong.

### 1. The gate's speech rate was ~1.9× too low

The gate estimated speech duration from characters using a hard-coded
**11.6 chars/s** for German. Qwen actually speaks German at **~19 chars/s**.
So the gate believed every line was 1.4–1.6× too long, shortened text that
already fitted its slot, and lost meaning while doing it.

Measured on the real 120 s run (`qwen3-tts/output/dub/CRISPR-120s/`), from
the `*_natural_fit_*.json` sidecars — `raw_seconds` is how long the voice
spoke, `target_seconds` is the slot it had:

| dub segment | gate estimate | MEASURED speech/slot |
|---|---|---|
| 2 | 1.41× | 0.885 |
| 3 | 1.59× | 0.925 |
| 5 | 1.15× | 0.882 |
| 6 | 1.34× | 0.903 |
| 7 | 1.16× | 0.731 |
| 8 | 1.10× | 0.710 |
| 10 | 0.94× | 0.915 |

Overshoot across all 12 synthesised segments: **median 0.882, max 0.948,
0 of 12 over 1.0**. The dub was already inside its slots. Every segment the
gate rejected actually fitted.

The independent TTS bench (`nemo/new_guys/crispr/tts_bench_2026-11-04/`)
agrees:

| engine | EN | DE | FR |
|---|---|---|---|
| Qwen3-TTS 0.6B | 16.78 | 22.36 | 19.94 |
| Supertonic 3 | 15.58 | 14.82 | 16.16 |
| Audio8 0.6B | 14.55 | 12.96 | 18.15 |

### 2. A pure characters-per-second rate is also wrong

Every synthesised segment carries a **fixed startup overhead** — the voice
needs a moment to start and the model adds a pause. A short line pays it in
full, so short lines are spoken disproportionately slowly. The gate
under-estimated them by more than a second, which is exactly where the dub
failed.

Measured model, fitted by least squares on the real run:

```
speech_seconds = 1.16 + characters / 18.98
```

It predicts the measured speech within **0.45 s** across eight segments and
identifies **all four** segments that failed, with 0 false positives and
0 false negatives. A pure rate model is off by more than a second on short
lines.

With the corrected rate the gate flags **4/10** blocks instead of 10/10,
and those four sit at 0.95–1.06×.

### The translator probes agreed with the wrong diagnosis

Tower, Ling, Gemini and Haiku all return the **same output** for budgets
143 / 100 / 80 chars. That looks like "no model can shorten". It is real,
but it was never the blocker — the budget was computed from a constant that
was 1.9× too low. Two things follow:

1. With the correct rate the gate rejects almost nothing, so the unreliable
   shortening path is rarely taken.
2. Where shortening *is* needed, `{"text": "..."}` JSON output plus a
   **word** budget (not a character budget) is the technique that measurably
   works: 15 words returned for a 14-word budget, versus 190–247 chars for a
   143-char budget (1.4–1.7× over).

## What works now (branch `feature/natural-pace-dubbing`)

- **Per-engine measured rates + self-calibration** (`nemo/speech_rate.py`).
  `calibrate_rate()` measures the voice from a previous dub's sidecars, so
  the rate self-corrects instead of being frozen into the source.
- **Gate uses rate + overhead**; flags 4/10 instead of 10/10.
- **L3 meaning check** (`nemo/dub_meaning.py`): on the shipped dub, mean WER
  **0.143**, similarity **0.933**, no lost facts. No word lists — it scales
  to any language and any number of videos.
- **L4 timing check** (`nemo/dub_timing.py`): **12/12** segments fit, no late
  onsets, no short gaps.
- **Automatic compressor** (`nemo/dub_compress.py`) that shortens via the
  models, with a language check and the meaning guard. No hand-written rules.
- **57 offline tests pass**; `tools/check_dubbing.py --offline|--timing|--meaning`.

## Open (where this was stopped)

1. **Unfinished bug in `_adapt_from_report`** — `source_srt` arrives as
   `None` at `pipeline_subtitle.py:470`. The call path was narrowed but the
   fix was not landed. This is why the L2 retry never ran.
2. **Two segments still do not fit** (indices 4 and 9) — segment 4 needs
   1.23×, segment 9 needs 1.14×. The compressor shortens but not enough; it
   needs to iterate harder or a stronger engine.
3. **Tower cannot be made to shorten** (returns identical text for every
   budget). hymt2 and eurollm can, but erratically and sometimes in the
   wrong language. Using Ling (OpenRouter) would likely fix it — it was not
   used, and no OpenRouter calls were spent on the dub.
4. **Uncommitted**: the `dub_compress.py` work and the `_adapt_from_report`
   fix are in the working tree, not committed.

## Known translation issues (Tower, not the pipeline)

- `viskales Gefühl` → should be `viszerales`.
- `Die Welt und das Wrestling mit einem mächtigen Werkzeug` — literal
  "wrestling with".
- `CRISPR ist ein akronym` — clipped / lower-case.
- `Kapitel 1. Die Anatomie eines Durchbruchs.` fine, but the following sentence
  can lose its subject on long blocks.


## 2026-11-05: voice pauses between subtitle lines

User confirmed the problem is the voice stopping between lines. Audit: ten separate TTS takes, six boundaries inside sentences, only 40 ms inserted by assembly. The raw takes themselves contain the pauses/resets. Prepared `batch_dubbing/continuous_german_worker.py`: one unchanged full-passage call to the same pinned Qwen Base clone on one T4. Offline smoke receipt support in the controller avoids a second cloud smoke allocation and pins all tested inputs/helpers. 109 tests passed, one skipped, 60 subtests passed, including mocked actual worker execution and real FFmpeg packaging. No new inference run started. Review plan: `new_guys/crispr/deepseek_t4_10cues_2026-11-05/CONTINUOUS_PLAN.md`. Current GERMAN_DUB.mp4 is still the rejected stitched version; do not present it as fixed.


## 2026-11-05: continuous take generated on local RTX 2060

User override: "run it in our gpu". Ran cached pinned Qwen Base once in FP16, max sequence 1024, token cap 600, five-minute shell deadline. Completed in 69.09 s; 27.76 s mono 24 kHz PCM16 output. Thirteen focused offline tests passed before inference. Local deliverable: `new_guys/crispr/deepseek_t4_10cues_2026-11-05/continuous_local/CONTINUOUS_GERMAN.mp4`, with matching WAV and original display-time SRT. Full video decoding passed. No speech speeding/trimming/inserted cue gaps and no final-frame hold. Subtitle timing is not force-aligned; human listening review remains. No Modal job started. GPU release verified: no compute processes, 5514 MiB free. The prepared Modal plan was not used.
