# Subtitle dubbing — status (updated 2026-11-05)

## Current handoff — continuous local dub and English comparison subtitles

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
  English cues shifted for this clip, UTF-8 BOM and CRLF. Its timing follows
  the English source; neither language is force-aligned to the new German
  voice. The extra English captioned video made before that clarification
  is not the requested handoff.
- **GPU released:** synthesis process exited; the recorded check found no
  compute processes and 5514 MiB free. See local `gpu_released.json`.
- **Limits:** human listening and semantic review are still open. The supplied
  text ends mid-sentence at “dass sie das Leben”; that was preserved. Existing
  translation problems such as “viskales Gefühl” were not silently rewritten.

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
