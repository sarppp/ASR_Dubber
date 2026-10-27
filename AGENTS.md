# ASR project agent instructions

These instructions apply to this repository and all subdirectories, including
`nemo`, `qwen3-tts`, and the translation pipeline.

## Mandatory Modal controller and human release gate

- ALL agent work with Modal must go through `nemo/modal_control` and its
  `nemo/modal-control` CLI or plan/inspect MCP. Start with controller status.
  Never bypass it with direct `modal run`, Python SDK calls, shell wrappers,
  ad hoc scripts, or a separate deployment path. Do not weaken this gate.
- Agents may research and propose the model/version, precision, CPU/T4 hardware,
  and runtime settings. Model selection is preparation, NOT permission to launch.
  Supertonic 3 M1 on Modal CPU remains the preferred dubbing option.
- Prepare an exact expiring plan with the controller, then explain its plan ID,
  purpose, model/revision, settings, input/time range, outputs, CPU or T4,
  smoke and full stages, container/time limits, expected speed/quality, and
  evidence-backed cost estimate (or explicitly unknown cost). Disclose all
  optional processing steps. Wait for the user's explicit go for this plan.
- The user can simply say **"release it"** in chat. The agent then executes
  that exact reviewed plan through the controller CLI, with its exact ID and
  the user's actual approval message. The user does NOT have to run a terminal
  command. Never fabricate approval or treat silence, a default, general task
  intent, or a status question as approval of a newly prepared plan.
- MCP is plan/inspect-only. Calling the controller execute CLI after the user's
  explicit go is permitted; calling it before that go is forbidden.
- Any change to model, hardware, code, input, catalog, or limits invalidates the
  approval: prepare a new plan and obtain a new go. Expired or consumed plans
  also need a new plan and approval; there are no automatic fallback retries.
- GPU allocation is T4-only, with no environment-selected hardware or GPU
  fallback. CPU allocations are allowed. Worker declarations must explicitly
  bound timeout, max_containers=1, and retries=0. Do not work around validation
  by hiding allocations in helpers or dynamic code. Include all execution code,
  imported local helpers, configuration and inputs in source_files.
- Report the actual result, timing, quality evidence and limitations, artifacts,
  errors and observed usage/cost afterward. Preserve the approval and run logs.
- This is an operational gate, not an OS sandbox: an agent with shell access
  and Modal credentials must still obey this contract. Never claim the MCP or
  plan receipt cryptographically proves a chat message came from the human.

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
