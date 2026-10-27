# ASR project agent instructions

These instructions apply to this repository and all subdirectories, including
`nemo`, `qwen3-tts`, and the translation pipeline.

## Model selection and Modal controller delegation

- For an authorized task executed through `nemo/modal_control`, the agent may
  choose the model, version, precision/quantization, and runtime settings. This
  is delegated selection: a separate model-choice question is not required.
- Before launching, tell the user what was chosen and why: model/version,
  settings, Modal CPU or T4, input/time range, expected speed/quality tradeoffs,
  and cost/runtime estimate when supported by evidence. State unknowns rather
  than inventing numbers. Record the choice with the run configuration.
- Respect any model, hardware, budget, or scope the user explicitly specifies.
  Supertonic 3 with preset M1 on Modal CPU remains the preferred dubbing option;
  Qwen 1.7B is an alternative when the task warrants it. Explain departures from
  the preferred option. GPU work remains T4-only.
- Delegated selection does not authorize new tasks, broad model sweeps, extra
  comparisons, unbounded retries, or spending outside the approved scope. A
  status question or permission to choose a model is not permission to launch
  a new job. Report failures; do not silently switch models and rerun.
- After the run, tell the user what actually ran, measured timing, available
  quality results and their limits, outputs, and observed usage/cost if available.
- Outside the Modal controller workflow, present relevant model options and
  wait for the user's choice, unless they already selected one for the current
  task. This does not override the requirement to run inference on Modal.
- Optional processing tools still require the user's choice under the next
  section; delegated model selection does not authorize those extra steps.

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
