# Subtitle dubbing — status (DeepSeek work, 2026-11-04)

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

## Known translation issues (Tower, not the pipeline)

- `viskales Gefühl` → should be `viszerales`.
- `Die Welt und das Wrestling mit einem mächtigen Werkzeug` — literal
  "wrestling with".
- `CRISPR ist ein akronym` — clipped / lower-case.
- `Kapitel 1. Die Anatomie eines Durchbruchs.` fine, but the following sentence
  can lose its subject on long blocks.
