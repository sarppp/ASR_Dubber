# Modal Control

A reusable local supervisor and read-only observability dashboard for Modal jobs. The speech benchmark is its first catalog, with 21 model configurations. **This benchmark is T4-only for GPU work, with no fallback. CPU candidates remain CPU. The worker checks the actual GPU before model download/inference; earlier L4/A10G attempts are excluded from the requested comparison and retained as history.** All state lives in `modal_control/.state/`, separate from the existing prod-control repository.

From `/home/sarpk/uv_tests/ASR/nemo`:

```bash
./modal-control doctor
./modal-control serve                 # http://localhost:8091
./modal-control watch                 # cloud inventory every 30 seconds
./modal-control run parakeet_int8
./modal-control run qwen_asr_06b --background
./modal-control status
./modal-control logs RUN_ID --follow
./modal-control stop RUN_ID
./modal-control recover
./modal-control export RUN_ID /tmp/run-evidence.tar.gz
./modal-control backup /tmp/modal-control-backup.tar.gz
```

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
