# CRISPR opening: native speech comparison

Prepared 2026-10-29. The user liked the preceding long-sentence test and selected
the opening74seconds of the CRISPR interview in `new_guys/crispr`.
The exact interval is0.00–74.24seconds so the last German caption is complete.
The user released this scope and execution completed successfully. Frozen
`scope.json`/`release_plan.json` preserve the reviewed preparation;
`authorization.json` and `result.json` record the actual release and result.


## Completed result

Open [the listening comparison](review.html). Original/older-dub references cover
0–74.24seconds. The new English sequence lasts71.920272seconds and German
82.631973seconds. All raw samples were retained at1.0, so the timing difference
remains visible. German would overlap by2.231156seconds if both raw blocks were
placed at their original German caption starts; the strict timing check does not
claim an aligned result.

Four calls/nine internal chunks completed, with smoke reused, no retries and
no GPU/ASR/translation calls. Actual Modal cost: **$0.00419829**. Both owned apps
stopped and final container inventory was empty. Independent byte/hash checks
passed; local verification/page generation used one CPU and peaked near91.6MiB.
Listening preference, pronunciation and original acoustic pause matching still
need evaluation. Full records and app IDs are in the parent HANDOFF.md.

## Inputs and outputs

`inputs.json` pins existing caption and video sources. Original English and
existing German wording are retained. The fixed F1 preset comes from the old
voice map; it is a listening-test choice, not a new identity inference or clone.
The old final German caption has unresolved speaker attribution because its
timing extends beyond the source diarization interval; the test does not repair
or conceal that evidence.

The reusable preparation function groups adjacent captions into longer blocks.
It splits only at a gap of at least1second after terminal punctuation and rejects
partial edge captions. This yields:

| Block | Caption interval | Requested words | Internal chunks |
|---|---|---:|---:|
| English0 |4.32–22.32s|55|1|
| English1 |23.36–73.76s|125|3|
| German0 |4.64–23.28s|58|2|
| German1 |25.00–74.24s|118|3|

The actual cached Supertonic1.3.1 chunker was inspected as pure code without
loading a model. It yields four API calls/nine internal chunks. The first English
block is the smoke; workload reuses it and synthesizes only the other three.

Planned artifacts: four complete raw FLOAT32 mono44100Hz WAVs, `native_en.wav`,
`native_de.wav`, and reports in `trial/smoke` and `trial/workload`. Native sequences
copy all samples and insert caption-derived leading, between-block and trailing
silence. Exact caption-slot matches can reference those same files as timelines;
no duplicate aligned WAVs consume the transfer budget.

`review.html` references the existing original and older German-dub videos,
restricted to the selected opening, with the new English/German audio after
generation. It displays original caption windows versus raw durations. The page
does not transcode, extract, overwrite or publish media. New tracks contain
speech only; background sound is not separated or remixed.

## What this comparison establishes

Playback and synthesis remain1.0; no sample trimming, acceleration, artificial
slowing or normalization. Reports check full sample retention and show overfill,
underfill and potential collisions at caption starts. The listening sequence can
have a different duration from the video; it is not claimed as a synchronized dub.

Only gaps between grouped blocks and at the excerpt edges are explicitly inserted.
Six positive gaps inside English blocks total1.44seconds; five inside German
blocks total1.64seconds. Those caption gaps are not independently inserted.
Within blocks, the synthesizer controls pauses, including its0.3second chunk gap.
Caption times themselves are not verified acoustic speech boundaries.

Existing text defects remain visible: German `Tiefkes`, awkward phrasing, and an
English period after a subordinate clause. Punctuation-based grouping cannot
prove grammatical completeness. Keeping wording fixed isolates the new synthesis
behavior; translation repair and exact source-silence matching remain separate
work. Nemotron is omitted from this run. The user reports allowing DeepInfra in
OpenRouter, but that setting has not been independently checked or used.

## Execution bounds

Use `scope.json`, `batch.json`, the controller catalog, and generated
`release_plan.json`. All inference stays on cached Modal CPU: one container,
2CPUs/4GiB, noGPU, download or image build. Pinned Supertonic3 revision and image
are identical to the successful preceding test. Each function has150s runtime,
30s startup and a180s controller stage deadline; two stages, no retries.
Maximum120s per clip,300s generated perstage,63MiB audio plus1MiB metadata.

The reviewed base-rate estimate is about$0.01848 including conservative cleanup
and transfer, within the$0.10 target. It is not a provider-enforced invoice cap.
Report actual owned-app billing after completion and verify all owned containers
stopped. Do not stop unrelated jobs or automatically expand the scope.

Local controller, verification and page generation use one CPU/512MiB. Set
`MALLOC_ARENA_MAX=2` and a1MiB thread-stack limit before starting the controller,
as verified in the preceding release; do not raise the memory bound to work
around virtual-memory thread reservations. Never load a model locally.

After successful synthesis, run the prepared page writer with `--input-file
dubbing_naturalness_2026-10-28/crispr_native/inputs.json`, `--report-file
dubbing_naturalness_2026-10-28/crispr_native/trial/workload/report.json` and
`--output-file dubbing_naturalness_2026-10-28/crispr_native/review.html`.
This postprocessing is included in the reviewed comparison scope.
