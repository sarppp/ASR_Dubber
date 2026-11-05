# Root cause: the natural-pace dub does not need a better translator

Status: **measured, reproduced**. Date: 2026-11-05. Branch: `feature/natural-pace-dubbing`.

## Summary

The 2026-11-04 status doc blamed the translator: "Tower ignores the budget",
so the measure-then-adapt loop cannot converge. That diagnosis is wrong.

The gate estimates speech duration from **characters**, using a hard-coded
constant of **11.6 chars/s** for German. The real Qwen voice speaks German at
**~17.6–22.4 chars/s**. The estimate therefore says every line is 1.4–1.6×
too long, the gate adapts it (cutting meaning), and the dub then fits easily —
because the audio was never too long in the first place.

**Fixing the translator cannot fix this.** The budget is computed from the
wrong constant. Correcting the constant removes the whole problem.

## Evidence

### 1. Every rejected segment actually fits

From the `*_natural_fit_*.json` sidecars of the real 120 s run
(`qwen3-tts/output/dub/CRISPR-120s/`). `raw_seconds` is how long the voice
spoke; `target_seconds` is the subtitle slot it had to fit into.

| dub segment | gate estimate | MEASURED speech/slot | verdict |
|---|---|---|---|
| 2 | 1.41× | 0.885 | fits |
| 3 | 1.59× | 0.925 | fits |
| 5 | 1.15× | 0.882 | fits |
| 6 | 1.34× | 0.903 | fits |
| 7 | 1.16× | 0.731 | fits |
| 8 | 1.10× | 0.710 | fits |
| 10 | 0.94× | 0.915 | fits |

Overshoot across all 12 synthesised segments: **median 0.882, max 0.948,
0 of 12 over 1.0**. The dub was already inside its slots.

### 2. The real speech rate is ~1.5–2× the assumed rate

Measured on the merged blocks dub.py actually speaks:
**median 17.64 chars/s, mean 17.82, min 10.35, max 24.42**.

The independent TTS bench (`nemo/new_guys/crispr/tts_bench_2026-11-04/`)
agrees:

| engine | EN | DE | FR |
|---|---|---|---|
| Qwen3-TTS 0.6B | 16.78 | **22.36** | 19.94 |
| Supertonic 3 | 15.58 | 14.82 | 16.16 |
| Audio8 0.6B | 14.55 | 12.96 | 18.15 |

Pipeline constant: `TTS_CHARS_PER_SECOND = {"de": 11.6, "fr": 12.3, "en": 13.5}`
in `nemo/local_mt.py`. For German under Qwen it is **1.9× too low**.

### 3. The adaptation pass was destroying meaning for no reason

Because the gate believed the lines were 1.4–1.6× too long, it re-translated
every one of the 10 blocks with a character budget, and the guards then
rejected most of the results. From `adaptation.json`:

| block | chars before → after | gate before → after |
|---|---|---|
| 1 | 152 → 152 | 1.49× → 1.49× (kept original) |
| 2 | 161 → 161 | 1.41× → 1.41× (kept original) |
| 3 | 122 → 120 | 1.61× → 1.59× |
| 4 | 175 → 175 | 1.60× → 1.60× (kept original) |
| 5 | 132 → 118 | 1.28× → 1.15× |
| 9 | 111 → 97 | 1.16× → 1.01× |

Blocks 1, 2 and 4 kept the original (the guards rejected the candidate), so
the "adaptation" mostly did nothing except shorten a few lines that were
never too long. The known quality problems in the shipped German —
`viskales Gefühl`, "Wrestling mit einem mächtigen Werkzeug",
`CRISPR ist ein akronym` — are all artifacts of this pointless compression.

## Why the translator probes agreed with the wrong diagnosis

Probing showed Tower, Ling, Gemini and Haiku all return **the same output**
for budgets 143 / 100 / 80 chars. That looked like "no model can shorten".

It is real, but it is not the blocker. Prompt-only length control on a
translation model is unreliable — true — but the pipeline never needed it,
because the budget itself was computed from a constant that was 1.9× too low.

Two things follow:

1. With the correct constant, the gate rejects almost nothing, so the
   unreliable shortening path is rarely taken.
2. Where shortening *is* needed, `{"text": "..."}` JSON output plus a **word**
   budget (not a character budget) is the technique that measurably works:
   technique C in `tools/probe_control.py` returned 15 words for a 14-word
   budget, versus 190–247 chars for a 143-char budget (1.4–1.7× over).

## What to change

- Replace the hard-coded `TTS_CHARS_PER_SECOND` with per-engine measured
  rates, and derive the budget from the real rate.
- Keep the guard at 1.0× (natural pace, never speed the voice).
- Make the translator's length control word-based + JSON, and only apply it
  when the measured gate actually says the line is too long.
- Add a calibration step that measures the engine's chars/s on the actual
  dub (cheap: a handful of segments) so the constant self-corrects instead
  of being hard-coded.

## Not needed

- A length-controllable MT.
- Ling / OpenRouter for the dub (cost + no benefit once the rate is right).
- Accepting 1.05–1.10 speed-ups.
