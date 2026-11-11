# CRISPR auto subtitle pipeline (10-cue trial)

Goal: reproduce the hand-made `../synchronized_trial/CRISPR-synchronized` result
automatically, for any video, without hand-editing translations.

Status: **10-cue trial only**. Not run on the full 907 cues. Semantic and
listening review are not done (`semantic_quality_verified=false`).

## Folder

| Path | What it is |
|---|---|
| `steps.py` | All pipeline steps. No model loads except the OpenRouter calls. |
| `step_0_english.json` | First 10 English cues from the original `.en.srt` |
| `step_1_phrases.json` | Same cues re-segmented into readable phrases |
| `step_2b_translation.json` | DE/FR translation (canonical output of step 2) |
| `step_3_gates.json` | Structural/accent/length findings per phrase |
| `step_4_repair.json` | One bounded repair pass for flagged phrases |
| `step_5_publish.json` | Published files, hashes, status |
| `CRISPR-auto.preview.mp4` | 0:03–0:37 burn-in preview |
| `CRISPR-auto.srt` / `.ass` | 3 rows: EN / DE / FR, 28 pt, 20% height cap |

## Pipeline

| Step | Command | Does | Checks |
|---|---|---|---|
| 0 | `python3 steps.py 0 --count 10` | take N English cues from the SRT | source SHA pinned |
| 1 | `python3 steps.py 1` | re-segment cues into phrases (1.5–6.0 s, ≤84 chars, punctuation) | every English word kept, in order; no cue lost/duplicated |
| 2 | `python3 steps.py 2b` | **translate complete sentences, then split the translation across their cues** | IDs once/in order; DE/FR one line; ≤84 chars |
| 3 | `python3 steps.py 3 --input step_2b_translation.json` | quality gates | length, format, stripped accents, omission/padding |
| 4 | `python3 steps.py 4` | one repair pass for flagged lines | keeps all meaning; never deletes verbs/facts |
| 5 | `python3 steps.py 5` | write SRT/ASS, copy video, burn preview | 3 rows, 28 pt, ≤20% height |

Model: `inclusionai/ling-3.0-flash` via OpenRouter.
Total cost for this trial: **≈ $0.0005**.

## Why step 2 works (the actual fix)

Cue-level translation breaks sentences: cue 9 ends "…for impacting" and cue 10
is "people's lives in the future.", so German lost `zu beeinflussen`.

Final design:

1. group consecutive cues into **complete sentences** (punctuation, max 6 cues / 30 s)
2. translate each whole sentence
3. split the translation back across that sentence's cues by character length
4. **never change cue times** → no timing drift, safe for 907+ cues

The concatenation of a sentence's cue texts is a complete, grammatical sentence
in DE and FR. Individual cues are fragments, exactly like the manual file.

## Problems found and fixed

| Problem | Cause | Fix |
|---|---|---|
| Model returned `content: null` | Ling is a reasoning model; reasoning used all tokens | send `"reasoning": {"enabled": false}` |
| DE/FR broke across cue boundaries | translated each cue as a standalone sentence | translate complete sentences, then split |
| Model ignored "move the boundary" permission | unreliable, made FR worse (duplicated `technologie`) | dropped; mechanical split instead |
| Repair deleted meaning (`zu beeinflussen`) | prompt said "shorten" | prompt now says preserve every verb/fact, compress wording only |
| Padding flag forced deletions | ratio >1.6 treated as failure | padding is now a warning; length is already capped at 84 chars |
| Accent damage (`mchtiges`) not caught | no check | lexicon gate for stripped DE/FR forms |

Gates are optional: the trial passes 10/10 with only soft "no accented
character" hints, which are false positives for lines that legitimately have no
umlauts/accents.

## Environment

- Key: `OPENROUTER_API_KEY` (accepts `OPEN_ROUTER` / `OPENROUTER_API_KEY`) in `nemo/.env`
- Key is never printed or logged
- Repo rule "all inference on Modal" is overridden for this text-only trial by
  explicit user instruction; ASR would still be Modal T4 if added

## Limits / next

- Only 10 cues of 907. Full run is one command (`2b` → `3` → `4` → `5`).
- No ASR yet: step 0 needs an existing English SRT. From raw audio, add a
  Parakeet T4 step before step 0.
- Listening review not done. Do not publish as final without it.
