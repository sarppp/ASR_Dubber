# ASR project agent instructions

These instructions apply to this repository and all subdirectories, including
`nemo`, `qwen3-tts`, and the translation pipeline.

## Modal controller and batch release

- Use the Modal controller for all Modal work. Before a batch starts, present
  the complete bounded scope, models, input set, output set, T4/CPU hardware,
  global container limits, smoke checks, time limits, retry limits and cost
  estimate (or say when cost is unknown).
- A single explicit user release authorizes the entire reviewed batch, including
  its specified later steps and bounded retries. Do not ask for another release
  merely because a dependent input becomes available or a step needs retry, as
  long as it stays within the released scope and limits. Preserve the actual
  user message in the audit record and do not claim it proves human identity.
- Ask for a new release only for a material change outside the approved
  model, input, hardware, concurrency, retry, time or cost scope.
- Keep GPU work T4-only unless explicitly authorized otherwise; bound each
  worker and the global concurrent GPU count. Preserve monitoring, logs,
  source provenance and smoke-before-full checks. Do not hide allocations.
- Finish independent implementation and local checks before requesting the
  initial release. Do not repeatedly ask for decisions already made.

## Optional processing tools also require user selection

- Before using `bilingual_subtitles.py` (including the copy under
  `nemo/subtitle_divider/`) or other optional subtitle/translation/media processing
  tools, explain why the step is needed and ask whether the user wants it. Wait
  for their answer before execution, even if the tool is local, free, or uses no
  model. Do not silently add it to an otherwise authorized pipeline.
- This includes optional bilingual subtitle creation, subtitle splitting,
  cleanup, gap filling, retranslation, organization, and subtitle burn-in or
  other optional media packaging. Present the relevant options and describe
  inputs, outputs, whether existing files change, and any model or Modal cost.
- Check wrapper scripts and automatic post-processing too: selecting a main
  pipeline does not authorize optional steps that were not disclosed and chosen.
  Disable or skip those steps until the user chooses them.
- An explicit request for the named tool or processing step in the current task
  counts as the user's choice; do not ask again for that same approved scope.
- Reading source, inspecting existing artifacts, and editing code or instructions
  do not require this processing-tool approval. The local-work allowance below
  does not override approval for executing optional processing steps.

## Modal execution and spending

### Modal: never fan out without a container cap (incident 2026-10-28)
**What happened:** an agent generated a dataset with `shard.starmap(...)` over 192 jobs and no
`max_containers`. Modal autoscaled straight to 100 containers at once (the default limit). The owner
found 99 running containers in the inventory without any warning. Cost $1.50 (the estimate was
$1.15), and the owner had never agreed to 100 parallel machines.
**Why:** `.map()`, `.starmap()`, `.spawn()` in a loop and batch submissions autoscale by default up to
the workspace limit. The agent only checked total cost (CPU-seconds are the same in parallel or in
series) and never stated or limited how many containers would run. A mistake inside such a burst
(wrong code, too much memory, a GPU type) is multiplied by 100 before anyone can react.
**Rules from now on:**
1. Every Modal function or class gets an explicit `max_containers=` in its decorator: at most 5 for
   CPU and 1 for GPU, unless the owner approves a higher number for that specific run.
2. Before the batch, disclose its job count, parallel container cap, hardware, duration, cost and
   bounded retries. One explicit release covers launches within that reviewed batch.
3. Smoke test with 1-2 jobs first; only then the full run, still within the cap.
4. While it runs, check `modal container list`; if the owner says stop, run `modal app stop <app-id>`
   immediately (finished outputs in volumes are kept).
5. Afterwards report the real cost from `modal billing report --for today`.


- Always use Modal for model inference, dubbing synthesis, and model benchmarks,
  including CPU models. Supertonic CPU means a Modal CPU container, not local
  inference. Do not fall back to local inference without an explicit user override.
- GPU runs must use T4 only. Never substitute A10, A10G, L4, or another GPU without
  explicit user authorization. Use Modal CPU for the selected CPU model.
- Before launching, establish the selected model(s), input/time range, hardware,
  and bounded scope. Reuse existing results when appropriate. Do not start broad
  model sweeps, extra evaluation jobs, automatic retry loops, or additional tests
  beyond what the user authorized.
- Use the existing `nemo/modal_control` observability/controller workflow. Save
  run configuration, model/version, app/function IDs, hardware, stdout/stderr,
  events, timings, results, errors, and available cost/usage data. Never log secrets.
- Check startup and progress proactively. Stop failing or runaway jobs promptly;
  report the failure and ask before a paid retry outside the authorized batch scope.
- After the authorized run, stop owned jobs and verify that their containers are
  no longer running. Preserve artifacts and logs for review.
- Local source edits, lightweight tests with mocked models, artifact inspection,
  and media packaging are allowed without a model run; they must not load models
  or launch cloud jobs implicitly.
