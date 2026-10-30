# Modal Control

A reusable local supervisor and read-only observability dashboard for Modal jobs. The speech benchmark is its first catalog, with 21 model configurations. **This benchmark is T4-only for GPU work, with no fallback. CPU candidates remain CPU. The worker checks the actual GPU before model download/inference; earlier L4/A10G attempts are excluded from the requested comparison and retained as history.** All state lives in `modal_control/.state/`, separate from the existing prod-control repository.

From `/home/sarpk/uv_tests/ASR/nemo`:

```bash
./modal-control doctor
./modal-control serve                 # http://localhost:8091
./modal-control watch                 # cloud inventory every 30 seconds
./modal-control --catalog /path/to/reviewed-jobs.json plan JOB
./modal-control inspect-plan PLAN_ID
# Single-job compatibility; use batch-plan for a multi-stage batch:
./modal-control execute PLAN_ID --confirm-plan-id PLAN_ID --approval-message "release it"
./modal-control batch-plan /path/to/reviewed-batch.json
./modal-control batch-inspect BATCH_PLAN_ID
./modal-control batch-execute BATCH_PLAN_ID --confirm-plan-id BATCH_PLAN_ID --approval-message "release all"
./modal-control mcp                  # plan/inspect-only stdio server
./modal-control status
./modal-control logs RUN_ID --follow
./modal-control stop RUN_ID
./modal-control recover
./modal-control export RUN_ID /tmp/run-evidence.tar.gz
./modal-control backup /tmp/modal-control-backup.tar.gz
```

## Container cap (read first)

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
   bounded retries. One explicit release covers all launches inside that reviewed batch.
3. Smoke test with 1-2 jobs first; only then the full run, still within the cap.
4. While it runs, check `modal container list`; if the owner says stop, run `modal app stop <app-id>`
   immediately (finished outputs in volumes are kept).
5. Afterwards report the real cost from `modal billing report --for today`.

Plans must state `max_containers`; a plan without a cap must not be released.

## Batch release workflow

A user can release one bounded batch once, including its predeclared dependent
steps and bounded retries. Create a version-1 JSON specification with `steps`,
`max_parallel` (1 or 2), `max_gpu_parallel` (1 or 2), `max_wall_seconds`,
`max_resource_seconds`, and `estimated_cost_ceiling_usd`. Each step declares
`id`, `catalog`, `job`, literal `cpu_cores` and `memory_mib`, optional
`depends_on`, `max_attempts` (1 to 3), and `derived_inputs` mapping input paths
to the producing dependency. The producer must list each such path as a success
file; the controller hashes it after production and checks it before later use.

Example specification (use actual catalog paths, job names and resource values):

```json
{
  "version": 1,
  "max_parallel": 2,
  "max_gpu_parallel": 2,
  "max_wall_seconds": 7200,
  "max_resource_seconds": 28800,
  "estimated_cost_ceiling_usd": 10,
  "steps": [
    {"id": "gpu", "catalog": "/path/gpu.json", "job": "gpu_job",
     "cpu_cores": 4, "memory_mib": 32768, "max_attempts": 2},
    {"id": "cpu", "catalog": "/path/cpu.json", "job": "cpu_job",
     "cpu_cores": 4, "memory_mib": 16384, "max_attempts": 2,
     "depends_on": ["gpu"],
     "derived_inputs": {"/path/gpu-output.json": "gpu"}}
  ]
}
```

`batch-plan` hashes the specification, catalogs, static inputs, worker and
controller sources, checks each worker's literal CPU/T4 configuration, rejects
unbounded concurrency, and checks that worst-case resource seconds and the
pinned-rate compute estimate fit the stated ceilings. The cost estimate is not
a provider-enforced billing limit: image builds, storage, network, price changes
and billing behavior can add costs. Use a separate Modal workspace spend limit
when a hard monetary cap is needed. Plans expire before release after at most
one hour; the approved batch may continue within its wall and stage limits.

`batch-execute` records the user's actual release message once and executes
ready steps within the global cap. Each step has a smoke and workload stage.
Reviewed retries repeat the same code and configuration; partial output blocks
automatic retry and is preserved. Failure stops admission of new steps.
Existing single-job `plan` and `execute` remain supported. A process crash is
recorded and recovered by the existing `recover` workflow; automatic batch
resume after a crash is not yet implemented.

## MCP agent interface

Install `modal_control/requirements.txt` into the controller Python environment.
Launch the stdio server with the absolute path to `nemo/modal-control mcp`.
MCP tools include `get_status`, `prepare_run_plan`, `get_run_plan`,
`prepare_batch_plan`, `get_batch_plan`, `get_run`, and `get_run_events`. There is no execute/approve tool. The CLI and MCP share the
same planning/inspection implementation. Register the server in your agent
client to expose these tools; creating this server does not automatically
connect an already-running client session.

The agent executes an explicitly authorized batch through the CLI; the MCP
interface remains plan/inspect only. The same user release message is recorded
once for the full batch, without asking again for dependent stages or approved
retries. See [AGENTS.md](../../AGENTS.md).

**Enforcement boundary:** the controller blocks unsupported paths through its
public API/CLI, but is not an OS sandbox. An agent with arbitrary shell access,
mutable source and Modal credentials can technically bypass it; the agent
contract forbids doing so. Approval text is an audit record, not independent
human authentication. Static worker checks cannot prove arbitrary imported
Python code will behave honestly. Credential isolation or an external approval
service would be needed for a security boundary against a noncompliant agent.

## What it records

- SQLite run/event/alert history, plus timestamped stdout and stderr and JSONL events per attempt.
- Model/configuration identity, source SHA256 hashes, commands, app IDs, function-call IDs, worker lifecycle, preemptions, exceptions, exit status, phase durations.
- Instrumented worker CPU usage, process RSS, GPU utilization, memory and power samples every five seconds, in the event stream.
- Verified output files and full output directories, with a SHA256 inventory. Each attempt has its own evidence directory.
- Timestamped cloud app, container and volume inventories. Historical benchmark logs/results can be imported with `adopt PATH`; they remain explicitly unsupervised observations.

## Startup and failure policy

Every job must pass its own smoke command, worker readiness event, fresh completion files, and unchanged source identity before the full command starts. A generic GPU test is insufficient. Startup/import failures, timeouts and explicit cancellation stop only app IDs observed in the supervised command's output. Failed smoke tests prevent full execution. Failed cloud-stop requests create alerts requiring reconciliation. A watcher marks dead supervisors interrupted and attempts to stop their recorded apps.

The catalog requires argv arrays, working directories, smoke/full success files, and supports deadlines, artifact directories, source files and non-secret environment settings. Credentials remain in the inherited environment or the configured env file; they are not copied into manifests. Known token formats and credential assignments are redacted. Arbitrary secrets printed in unrecognizable formats cannot be guaranteed redacted, so workers must never print credentials.

## Limits and durability

The dashboard is loopback-only and read-only. Alerts are local dashboard/database records; this does not send email or Slack messages. Monitoring runs while the local supervisor/watch processes and WSL are running. It is not an always-on cloud monitoring service. After a host outage, run `recover` or restart `watch` to reconcile. Cloud API failures and stale inventory timestamps remain visible. PID reuse can delay dead-supervisor detection; recovery never kills arbitrary local processes.

SQLite is the authoritative durable record (WAL, FULL synchronous); JSONL is an additional export. There is no automatic deletion/retention policy. Export evidence and back up `.state` with `backup`, which uses SQLite's backup API; copying a live SQLite file alone can miss WAL data. Disk exhaustion or host loss still requires operational backups. Artifacts are captured after completion; interrupted workers may leave only logs, not complete audio. Existing cloud volume stores model caches, not an off-machine backup of this controller.

Resource samples are observations, not billed GPU seconds. Account quotas, container-internal logs not emitted by the attached command, and unrelated apps' stdout are not collected. No fabricated cost estimates are shown. To supervise another project, add its job to the catalog and emit `MC_EVENT {"kind":"worker_ready"}` only after its container entrypoint starts.

## Checks

### Source-linked natural dubbing

The speaker-aware GPU workers retain original ASR words, reconcile speaker
corridors using overlap evidence, and translate each source utterance by stable
ID. Their `speech_units[language]` records drive the shared CPU implementation in
`batch_dubbing/cpu_natural.py`; subtitle display windows do not set speech timing.
Old subtitle-only JSON inputs fail local validation before remote synthesis.

The CPU path measures conservative outer padding, schedules complete turns at
native speed or at most 1.15×, and refuses to cut speech or publish a partial dub.
Native attempts and measurements survive in the mounted cache volume. Impossible
timing returns `needs_revision`; failed rendering/encoding returns `failed` with
the stage and error. Local entrypoints save the report and exit unsuccessfully,
so the controller cannot count either as completed audio.

The provider-independent planner supports an explicitly budgeted revision callback;
the current Supertonic adapter does **not** configure a translation/revision model.
These timing checks do not prove translation fidelity or perceptual naturalness.
Legacy fitting also uses the safe limits, but its subtitle-based input and local
inference routing remain separate. Do not run local inference without the explicit
override required by `AGENTS.md`.

Changed code and catalogs require a newly reviewed release plan. Existing output
files are exclusive, and historical inputs may lack source words or have moved;
prepare fresh input/output paths and verify their hashes before proposing a new
experiment. The earlier grouping comparison release does not authorize this changed
pipeline. See [the investigation handoff](../dubbing_naturalness_2026-10-28/HANDOFF.md)
for measurements, validation results and remaining work.

### Controller tests

```bash
python3 -m unittest discover -s modal_control/tests -v
```

Tests cover smoke rejection, success verification, missing readiness, stale artifacts, source changes, secret redaction, owned-app cancellation, deadlines and persisted events. `/metrics` exposes Prometheus run-state gauges and an alert counter; `/api/overview` and `/api/runs/ID/events` expose recorded evidence.


## Posted billing at launch and during runs

Every released single job or batch attempts a billing snapshot **before the first
worker launches**, then every two minutes and after shutdown. The CLI prints
credits used, cash charged, and the change from the starting snapshot. The
dashboard and `status` expose the latest observation; each run keeps timestamped
`billing_snapshot` events and JSON artifacts, including failure/unavailable states.

These are posted Modal figures, not a live meter or a spending cap. Workspace
changes can include unrelated work. Owned-app cost is selected by recorded app
IDs, including a batch's child jobs, rather than by matching descriptions. A
missing credit field, unavailable CLI, or failed request is shown as unknown;
remaining credits are never calculated from an assumed plan allowance. Month
rollovers do not produce a misleading cross-month delta.

Install the controller dependencies into its own environment:

```sh
uv venv modal_control/.state/runtime
uv pip install --python modal_control/.state/runtime/bin/python -r modal_control/requirements.txt
```

The launcher uses that environment, or `MODAL_CONTROL_PYTHON` when explicitly set.
Billing uses the controller's Modal CLI by default, independently of older worker
CLIs. A catalog can specify `billing_command` as an argument list when it needs a
separate compatible CLI. Credentials are loaded from the catalog's `env_file` in
memory and are not saved in billing artifacts.
