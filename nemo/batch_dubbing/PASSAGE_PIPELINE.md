# Continuous delivery and source-anchored dubbing

**Current selection (2026-11-06):** `continuous_local/synced_result/DUB_SYNCED.mp4`.
The production default is now `--dub-timing continuous-synced`: continuous TTS,
measured bounded pitch-preserving slowdown, then fresh target word alignment.
The authoritative steps and reference distinction are in [SYNCED_WORKFLOW.md](SYNCED_WORKFLOW.md).
The raw-continuous and source-anchored descriptions below are retained as
selectable alternatives and historical validation, not the selected target.

Implemented in `passage_batch.py`, reusing `speech_units.py`,
`natural_pipeline.py`, `speech_schedule.py`, and `natural_render.py`.
`sync_continuous_dub.py` is a separate short-clip experiment; this route does
not call it or change speech tempo.

## Separate source-anchored batch API contracts

1. Supply the complete source ASR word timeline and speaker turns. Build units
   with `build_speech_units`, then translate those units using their stable IDs.
   Caption breaks do not define speech boundaries. Every supplied source word
   must occur exactly once. Unfinished final/size-limited passages block synthesis.
2. Generate one continuous native-speed take per unit. Keep its raw recording.
   Anchor each unit to its own absolute source position; never append an
   overrun to the previous unit. The permitted local shift is at most 250 ms;
   an ending gap above 400 ms is reported as a diagnostic. It does not trigger
   a retry or force a sentence to fill its window. Real source pauses remain.
   `end_gap_policy="block"` is available only for explicitly stricter checks.
3. Verify spoken words with independent ASR, review source-to-target meaning,
   and align target words to the accepted recording. Hash-bound receipts are
   required. A forced aligner alone is not evidence that words were spoken.
4. Publish only an entirely accepted asset. The native renderer removes only
   its validated recording-edge silence; it preserves speech samples and
   internal pauses. Mux the complete source video stream without `-shortest`,
   `-t`, scene deletion, or tempo correction. Verify duration and video packet
   hashes. Target SRT follows the accepted target word times. The separately
   labeled source SRT retains source word times.
5. Process up to 59 assets serially. Persist per-asset reports, isolate failed
   assets, and reuse hash-checked raw takes. Pin model revision, voice/reference,
   and settings in provider identity. No automatic synthesis retry or wording
   revision occurs. Failures remain visible for a subsequent bounded correction.

## Offline preflight

From `nemo`:

```bash
../qwen3-tts/.venv/bin/python -m batch_dubbing.passage_batch \
  --manifest jobs.json --report preflight.json
```

`jobs.json` contains `{"assets": [{"id": "video-01", "manifest": "video-01.json"}]}`.
Child paths are relative to that file. Each asset manifest contains:

- `id`, `source_id`, `duration_seconds`, `source_language`, `target_language`;
- `source_words`: objects with `word`, `start`, `end`, `speaker`;
- `turns`: objects with `start`, `end`, `speaker`;
- `translations`: complete target passages keyed by `build_speech_units` IDs;
- for publication, `video` (absolute path) and `video_sha256`.

The command loads one asset manifest at a time and never loads a model. Its
`ready_for_synthesis` status approves source structure only, not dubbing quality.
It writes a new report and refuses to overwrite an existing one.

## Model integration

`run_batch(assets, output_dir=..., synthesize=..., content_check=...,
provider_identity=...)` is the serial execution API. `synthesize(unit, text)`
returns mono float samples at 44100 Hz, voice identity, and optionally predicted
duration. Keep provider calls under the approved local/Modal controller and
explicit resource scope; this module does not allocate hardware itself.

Use `ContentVerifier` for `content_check`. Its adapters are:

- `transcribe(wav_path, target_language) -> independently recognized str`;
- `review_meaning(request) -> {faithful, omissions, additions, contradictions,
  reason}`;
- `align(wav_path, text, target_language) -> [{text, start, end}, ...]`.

Give it pinned provider identities, an explicit total call cap, and a cache
directory. ASR mismatch, incomplete meaning evidence, or invalid alignment
blocks output. Checks are cached against audio, source, target, and provider
identities. Cached failures do not trigger hidden retries.

## Existing production entry point

The default `run_pipeline.py --pipeline subtitle-dubbing` path remains
`pipeline_subtitle.py` -> `batch_subtitles.py --backend local` ->
`local_sentence_pipeline.py`: Parakeet Ultra source ASR and Tower Plus 2B.
The subtitle-only and legacy routes remain available.

The integrated order is:

1. Ultra measures source words and packs source cues. Tower translates complete
   source passages using the existing soft length-budget prompt and splits the
   translations back across those source cues. The initial post-translation
   shortener is disabled: it previously removed a clause.
2. CPU forced alignment measures the original speech anchors before TTS.
   Freeze the paired source/target caption IDs, text and line breaks in
   `source_alignment_plan.json`. Caption cuts do not become speech breaks.
3. `qwen3-tts/dub.py` groups complete same-speaker passages into continuous
   contexts of at most 30 seconds. The 29-second reference uses one recording.
4. After synthesis workers close, align each complete generated recording on
   CPU. The alternative `--dub-timing continuous` keeps the whole recording
   at its original source cue context start. Every raw sample, recording edge
   and internal pause survives; no sentence is repositioned. Each bounded
   context has an independent absolute start, so an overrun cannot push later
   contexts forward. Overrun/overlap blocks publication. Source sentence onset
   and ending differences remain visible in `source_passage_timing` and warnings.
   The previous behavior remains selectable with `--dub-timing source-anchored`.
5. Retiming uses the frozen target cue word counts, with an 80 ms display lead
   and 120 ms tail capped at the next cue. Copy those timestamps to the paired
   English comparison cues. Do not repack words or redistribute English text
   after synthesis. English words are not acoustically aligned to German.
6. Publish per-language SRTs and combined SRT/ASS. Embed separate source and
   target subtitle tracks in the MP4, with source comparison selected by default
   and the audio tagged with the target language. Verify video packet identity.

Continuous delivery preserves the take's ending, including the original short
silent ending in the selected `continuous_local` example. It does not pad the
WAV, stretch the voice or relocate sentences to fill that gap; the MP4 still
contains the full video timeline. A failed continuous context remains for
review without an automatic wording rewrite. The source-anchored option still
allows at most 250 ms of passage adjustment and one guarded measured wording
correction (`--dub-retries 0` disables that correction). Its lexical checks do
not certify semantic fidelity.

A source passage longer than 30 seconds requires review, rather than being
silently split or truncated. A selected excerpt's incomplete final sentence
is retained, not completed with invented words. The original video always
retains its full timeline. Existing source-timed captions are kept separate
from final dub-timed captions.

The default route now supplies source-word timing and target-word alignment,
but does **not** provide the independent spoken-content/meaning verification
contract of `run_batch`. Forced alignment is timing evidence, not proof of
faithful speech or translation. Reports state this limitation. Multi-speaker
turn detection and long-video quality remain separate validation work.

## Evidence and remaining validation

Offline tests cover missing words, malformed SRTs, incomplete passages,
overruns, early endings, stale receipts, corrupted caches, selective reuse,
and actual FFmpeg packet preservation. The generated scale preflight contains
59 synthetic one-hour timelines, 35,400 passages, and 70,800 source words.
Another test checks all 35,400 absolute PCM placements and complete durations.

Model callbacks in these tests are synthetic. No real 59-video synthesis or
quality evaluation has run. Live synthesis/ASR/meaning/alignment adapters still
need connecting through the selected controller before that workload can run.
Complete coverage here means coverage of the **supplied** ASR words: it does
not prove the source ASR itself missed no audible speech. Independent source
speech coverage and representative long-video listening remain necessary
before claiming production readiness. Exact ASR text matching is conservative
and may flag recognition errors for review.

## Selected continuous reference (2026-11-06)

The user explicitly selected `continuous_local/CONTINUOUS_GERMAN.mp4` and its
aligned external SRTs. This supersedes the CPU-passage version as the target.
Rebuild from the raw take through the same production finishing functions:

```bash
.venv/bin/python -m batch_dubbing.reference_regression \
  batch_dubbing/crispr_29_continuous_reference.json \
  --output /tmp/crispr-continuous-check
```

Use a new output directory. The manifest hashes the inputs and the exact named
reference files. The verified delivery is
`new_guys/crispr/deepseek_t4_10cues_2026-11-05/continuous_workflow_regression/result/`.
All nine checks pass: raw PCM, encoded audio packets, video packets/full duration,
and both external SRTs including every cue/text/timestamp match `continuous_local`.
The intentional packaging change is to replace the reference MP4's stale embedded
subtitle track with its aligned external EN/DE subtitles, EN selected by default.
Both embedded tracks are extracted and checked against the selected aligned SRTs.
The report lists the exact reference file paths, not just a generic "match".

No models run in this regression: it tests automatic finishing from the existing
raw recording and cached word alignments. Fresh ASR/MT/TTS may still change the
words/delivery and has not been rerun in this step. All tests involving real
local media skip explicitly when that ignored media is absent in a clean checkout.

## Earlier CPU-passage regression (2026-11-05; retained, no longer selected)

From `nemo`, with the original local trial media present:

```bash
../qwen3-tts/.venv/bin/python -m batch_dubbing.reference_regression \
  batch_dubbing/crispr_29_reference.json \
  --output /tmp/crispr-reference-check
```

The output directory must be new. The manifest hashes every input and expected
artifact. The command uses the same `source_plan` / `finish` functions as the
production route; it regenerates media from the approved raw recording and
cached source/target word alignments, without loading a model or calling Modal.
`comparison.json` reports exact PCM, encoded audio, video packet preservation,
full duration, embedded track layout and subtitle packets, and both SRT byte
comparisons including cue cuts, text, line breaks and timestamps. Any mismatch
returns a failing exit status. The local-media pytest also runs this check;
it skips explicitly in clean checkouts where the ignored media is absent.

Verified output: `new_guys/crispr/deepseek_t4_10cues_2026-11-05/approved_workflow_regression/`.
All nine checks pass: the 29.487792-second video, audio and ten EN/DE cues match
the user-approved CPU preview. This establishes deterministic finishing from
fixed evidence, not identical fresh ASR, translation or synthesis.

The earlier fresh `local_matched_pipeline` run differs upstream: its English
lexical transcript matches but its punctuation/packing produces nine cues
instead of ten, and its final German passage differs. Its older post-TTS
caption repacking is superseded by the frozen-cue fix. That video's ending gap
is 1.394 seconds; it must not be described as the approved result. No new model
run was made for this comparison. See `dub_deepseek_status.md` for earlier runs.
