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
