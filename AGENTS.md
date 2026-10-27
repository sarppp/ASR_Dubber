# ASR project agent instructions

These instructions apply to this repository and all subdirectories, including
`nemo`, `qwen3-tts`, and the translation pipeline.

## Model selection belongs to the user

- Before each new model-based task or run, present the relevant model options and
  wait for the user to choose. Include the current model and realistic alternatives,
  their purpose, available speed/quality evidence, and expected hardware and cost
  when known. Clearly distinguish measured results from estimates.
- The user makes the selection. Do not silently select, substitute, upgrade, or
  benchmark models. A configured default is a recommendation, not permission to run.
- Supertonic 3 with preset M1 on CPU is the current preferred dubbing option.
  Qwen 1.7B remains an explicit alternative. Still ask for the user's choice for
  each new task unless the user has already explicitly selected it for that task.
- An explicit selection in the current task satisfies this requirement; do not
  ask repeatedly for the same choice. A change of model or added comparison needs
  a new user choice. Do not interpret a status question as permission for more runs.

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
  report the failure and ask before a paid retry outside the authorized scope.
- After the authorized run, stop owned jobs and verify that their containers are
  no longer running. Preserve artifacts and logs for review.
- Local source edits, lightweight tests with mocked models, artifact inspection,
  and media packaging are allowed without a model run; they must not load models
  or launch cloud jobs implicitly.
