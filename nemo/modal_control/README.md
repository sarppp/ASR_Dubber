# Modal Control

A reusable local supervisor and read-only observability dashboard for Modal jobs. The speech benchmark is its first catalog, with 21 model configurations. **This benchmark is T4-only for GPU work, with no fallback. CPU candidates remain CPU. The worker checks the actual GPU before model download/inference; earlier L4/A10G attempts are excluded from the requested comparison and retained as history.** All state lives in `modal_control/.state/`, separate from the existing prod-control repository.

From `/home/sarpk/uv_tests/ASR/nemo`:

```bash
./modal-control doctor
./modal-control serve                 # http://localhost:8091
./modal-control watch                 # cloud inventory every 30 seconds
./modal-control --catalog /path/to/reviewed-jobs.json plan JOB
./modal-control inspect-plan PLAN_ID
# Agent runs this ONLY after the user explicitly says "release it":
./modal-control execute PLAN_ID --confirm-plan-id PLAN_ID --approval-message "release it"
./modal-control mcp                  # plan/inspect-only stdio server
./modal-control status
./modal-control logs RUN_ID --follow
./modal-control stop RUN_ID
./modal-control recover
./modal-control export RUN_ID /tmp/run-evidence.tar.gz
./modal-control backup /tmp/modal-control-backup.tar.gz
```

## Mandatory plan and release workflow

All agent Modal operations use this controller. Direct `run` and `batch` are
blocked, including the old background path. Agents may propose models/settings,
but must prepare and explain the exact plan and wait for the user's go. The user
says **"release it"** in chat; the agent executes that plan and records the actual
approval message. The user need not run a terminal command. General permission
to do a task does not approve a newly prepared plan.

Plans expire after 30 minutes (maximum one hour), hash their complete payload,
and bind the catalog, worker/helper/input hashes, controller implementation,
model/settings, hardware, output expectations and smoke/full limits. Execution
rejects altered, stale, expired or consumed plans. An atomic single-use receipt
prevents concurrent release/retries. A failed attempt needs a new plan and go.
Catalogs marked submissions-disabled remain blocked; do not restore archived
benchmark sweeps. Prepare a separate small catalog for a newly requested task.

Each job requires `review` fields: `purpose`, `model`, `revision`, `precision`,
`reason`, `input`, `input_seconds`, `outputs`, `quality_tradeoffs`, `cost_estimate`,
`hardware` (`CPU` or `T4`), `max_containers` (1), and `max_total_seconds`.
Cost may be explicitly unknown; it is not a billing cap. Job `max_seconds` is a
per-stage deadline, so the reviewed total must cover smoke plus full execution.
Commands must directly invoke `python -m modal run worker.py`, without detach or
shell wrappers. Pin worker code, imported local helpers and inputs/configuration
in `source_files`. Function decorators must use literal CPU/T4 allocation,
`max_containers=1`, `retries=0`, and a timeout within the stage deadline. Dynamic
GPU selectors and fallback lists are rejected. T4 workers must include observed
`gpu` in the `worker_ready` event. These checks fail closed on legacy jobs: adapt
them explicitly before proposing a new plan, never bypass the gate.

## MCP agent interface

Install `modal_control/requirements.txt` into the controller Python environment.
Launch the stdio server with the absolute path to `nemo/modal-control mcp`.
MCP tools: `get_status`, `prepare_run_plan`, `get_run_plan`, `get_run`, and
`get_run_events`. There is no execute/approve tool. The CLI and MCP share the
same planning/inspection implementation. Register the server in your agent
client to expose these tools; creating this server does not automatically
connect an already-running client session.

This follows prod-control's immutable plan/inspect pattern, with the user's
explicitly requested chat-release workflow: the agent may execute through the
CLI only AFTER the user approves. See [AGENTS.md](../../AGENTS.md).

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

Resource samples are observations, not billed GPU seconds. Actual billing, account quotas, container-internal logs not emitted by the attached command, and unrelated apps' stdout are not collected by this version. No fabricated cost estimates are shown. To supervise another project, add its job to the catalog and emit `MC_EVENT {"kind":"worker_ready"}` only after its container entrypoint starts.

## Checks

```bash
python3 -m unittest discover -s modal_control/tests -v
```

Tests cover smoke rejection, success verification, missing readiness, stale artifacts, source changes, secret redaction, owned-app cancellation, deadlines and persisted events. `/metrics` exposes Prometheus run-state gauges and an alert counter; `/api/overview` and `/api/runs/ID/events` expose recorded evidence.
