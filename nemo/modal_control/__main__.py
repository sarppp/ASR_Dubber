import argparse, json, os, sys, subprocess, time, urllib.parse
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from .core import (
    Store,
    run_job,
    load_catalog,
    atomic,
    utc,
    clean,
    TERMINAL,
    recover,
)

DEFAULT = Path(__file__).resolve().parent


def snapshot(store, catalog_path):
    catalog = load_catalog(catalog_path)
    env = os.environ.copy()
    if catalog.get("env_file"):
        from dotenv import dotenv_values

        env.update(
            {
                k: v
                for k, v in dotenv_values(catalog["env_file"]).items()
                if v is not None
            }
        )
    cmd = catalog.get("modal_command", [sys.executable, "-m", "modal"])
    result = {"observed_at": utc()}
    for resource in ("app", "container", "volume"):
        try:
            p = subprocess.run(
                [*cmd, resource, "list", "--json"],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            result[resource] = (
                json.loads(p.stdout)
                if p.returncode == 0
                else {"error": clean(p.stderr[-3000:]), "returncode": p.returncode}
            )
        except Exception as e:
            result[resource] = {"error": str(e)}
    d = store.root / "snapshots"
    d.mkdir(exist_ok=True)
    atomic(d / (utc().replace(":", "-") + ".json"), result)
    atomic(store.root / "latest_cloud.json", result)
    return result


def adopt(store, path):
    root = Path(path).resolve()
    runs = []
    for p in sorted((root / "06_logs").glob("*.log")):
        key = "import:" + p.stem
        existing = [r for r in store.overview()["runs"] if r["job"] == key]
        if existing:
            continue
        rid = store.create(
            key,
            {
                "origin": "pre-controller benchmark",
                "source_log": str(p),
                "imported_at": utc(),
                "supervised": False,
            },
        )
        text = clean(p.read_text(errors="replace"))
        dest = store.root / "runs" / rid / "imported.log"
        dest.write_text(text)
        for line in text.splitlines():
            if (
                "Error" in line
                or "Traceback" in line
                or "preemption" in line
                or "crash-loop" in line
            ):
                store.event(rid, "historical_log", {"text": line})
        store.state(rid, "observed", note="Historical import; not proof of success")
        runs.append(rid)
    for p in (root / "runs").glob("*/**/metrics.json"):
        key = "result:" + str(p.parent.relative_to(root / "runs"))
        if any(r["job"] == key for r in store.overview()["runs"]):
            continue
        rid = store.create(
            key, {"origin": "benchmark result", "metrics": str(p), "supervised": False}
        )
        d = store.root / "runs" / rid / "artifacts"
        d.mkdir()
        (d / "metrics.json").write_text(p.read_text())
        store.event(rid, "imported_metrics", json.loads(p.read_text()))
        store.state(
            rid,
            "observed",
            note="Completed result imported; no startup gate was applied",
        )
    return runs


def serve(store, host, port):
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Dashboard binds to loopback only")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            parsed = urllib.parse.urlsplit(self.path)
            path = parsed.path
            q = urllib.parse.parse_qs(parsed.query)
            try:
                if path == "/":
                    body = (DEFAULT / "web/index.html").read_bytes()
                    kind = "text/html; charset=utf-8"
                elif path == "/api/overview":
                    data = store.overview()
                    cloud = store.root / "latest_cloud.json"
                    data["cloud"] = (
                        json.loads(cloud.read_text()) if cloud.exists() else None
                    )
                    body = json.dumps(data).encode()
                    kind = "application/json"
                elif path.startswith("/api/runs/"):
                    rid = path.split("/")[3]
                    store.get(rid)
                    if path.endswith("/events"):
                        data = store.events(rid, int(q.get("after", ["0"])[0]))
                    else:
                        data = store.get(rid)
                        data["manifest"] = json.loads(data["manifest"])
                        d = store.root / "runs" / rid
                        data["artifacts"] = (
                            json.loads((d / "artifact_manifest.json").read_text())
                            if (d / "artifact_manifest.json").exists()
                            else []
                        )
                    body = json.dumps(data).encode()
                    kind = "application/json"
                elif path == "/metrics":
                    overview = store.overview()
                    lines = [
                        "# HELP modal_control_runs Number of recorded runs by state.",
                        "# TYPE modal_control_runs gauge",
                    ]
                    for state in sorted({r["status"] for r in overview["runs"]}):
                        lines.append(
                            f'modal_control_runs{{state="{state}"}} {sum(r["status"] == state for r in overview["runs"])}'
                        )
                    with store.db() as c:
                        alert_count = c.execute(
                            "SELECT COUNT(*) FROM alerts"
                        ).fetchone()[0]
                    lines += [
                        "# TYPE modal_control_alerts_total counter",
                        f"modal_control_alerts_total {alert_count}",
                    ]
                    body = ("\n".join(lines) + "\n").encode()
                    kind = "text/plain; version=0.0.4"
                elif path.startswith("/benchmark/"):
                    base = (
                        DEFAULT.parent / "speech_model_benchmark_2026-10-27"
                    ).resolve()
                    p = (
                        base / urllib.parse.unquote(path[len("/benchmark/") :])
                    ).resolve()
                    if (
                        not p.is_relative_to(base)
                        or not p.is_file()
                        or any(
                            part.startswith(".") for part in p.relative_to(base).parts
                        )
                    ):
                        raise FileNotFoundError()
                    import mimetypes

                    body = p.read_bytes()
                    kind = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
                elif path.startswith("/files/"):
                    relative = urllib.parse.unquote(path[7:])
                    p = (store.root / "runs" / relative).resolve()
                    base = (store.root / "runs").resolve()
                    if not p.is_relative_to(base) or not p.is_file():
                        raise FileNotFoundError()
                    body = p.read_bytes()
                    kind = "application/octet-stream"
                else:
                    raise FileNotFoundError()
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
            except (KeyError, ValueError, FileNotFoundError):
                self.send_error(404)

    print(f"Modal Control: http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main():
    p = argparse.ArgumentParser(prog="modal-control")
    p.add_argument("--state", default=str(DEFAULT / ".state"))
    p.add_argument("--catalog", default=str(DEFAULT / "catalog/jobs.json"))
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("status", "doctor", "snapshot", "watch", "recover"):
        sub.add_parser(name)
    r = sub.add_parser("plan")
    r.add_argument("job")
    r.add_argument("--ttl-seconds", type=int, default=1800,
                   help="Plan review lifetime in seconds (maximum 3600)")
    r = sub.add_parser("batch-plan")
    r.add_argument("spec")
    r.add_argument("--ttl-seconds", type=int, default=3600)
    r = sub.add_parser("batch-inspect")
    r.add_argument("plan_id")
    r = sub.add_parser("batch-execute")
    r.add_argument("plan_id")
    r.add_argument("--confirm-plan-id", required=True)
    r.add_argument("--approval-message", required=True, help="Actual user's batch release message")
    r = sub.add_parser("inspect-plan")
    r.add_argument("plan_id")
    r = sub.add_parser("execute")
    r.add_argument("plan_id")
    r.add_argument("--confirm-plan-id", required=True)
    r.add_argument("--approval-message", required=True, help="Exact user go for this plan; never fabricate")
    sub.add_parser("mcp")
    r = sub.add_parser("run")
    r.add_argument("job")
    r.add_argument("--background", action="store_true")
    r = sub.add_parser("batch")
    r.add_argument("jobs", nargs="+")
    r.add_argument("--continue-on-failure", action="store_true")
    r = sub.add_parser("show")
    r.add_argument("run_id")
    r = sub.add_parser("logs")
    r.add_argument("run_id")
    r.add_argument("--follow", action="store_true")
    r = sub.add_parser("stop")
    r.add_argument("run_id")
    r = sub.add_parser("adopt")
    r.add_argument("benchmark")
    r = sub.add_parser("serve")
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int, default=8091)
    r = sub.add_parser("backup")
    r.add_argument("destination")
    r = sub.add_parser("export")
    r.add_argument("run_id")
    r.add_argument("destination")
    a = p.parse_args()
    s = Store(a.state)
    if a.cmd == "status":
        print(json.dumps(s.overview(), indent=2))
    elif a.cmd == "doctor":
        cat = load_catalog(a.catalog)
        print(
            json.dumps(
                {
                    "catalog_valid": True,
                    "jobs": len(cat["jobs"]),
                    "python": sys.version,
                    "state_writable": os.access(s.root, os.W_OK),
                    "startup_gate": "required for every run",
                    "secret_values_persisted": False,
                },
                indent=2,
            )
        )
    elif a.cmd == "plan":
        from .plans import prepare_plan
        print(json.dumps(prepare_plan(s, a.catalog, a.job, ttl_seconds=a.ttl_seconds), indent=2))
    elif a.cmd == "batch-plan":
        from .plans import prepare_batch_plan
        print(json.dumps(prepare_batch_plan(s, a.spec, ttl_seconds=a.ttl_seconds), indent=2))
    elif a.cmd == "batch-inspect":
        from .plans import show_batch_plan
        print(json.dumps(show_batch_plan(s, a.plan_id), indent=2))
    elif a.cmd == "batch-execute":
        from .batch_plans import run_batch
        run_id, rc = run_batch(s, a.plan_id, a.confirm_plan_id, a.approval_message)
        print(json.dumps({'run_id': run_id, 'exit_code': rc}))
        return rc
    elif a.cmd == "inspect-plan":
        from .plans import show_plan
        print(json.dumps(show_plan(s, a.plan_id), indent=2))
    elif a.cmd == "execute":
        from .plans import show_plan
        plan = show_plan(s, a.plan_id)
        payload = plan["payload"]
        _, rc = run_job(s, payload["catalog_path"], payload["job"],
                        plan_id=a.plan_id, confirmation=a.confirm_plan_id, approval_message=a.approval_message)
        return rc
    elif a.cmd == "mcp":
        from .mcp_server import create_server
        create_server(s, a.catalog).run()
    elif a.cmd in ("run", "batch"):
        raise ValueError("Direct run/batch is disabled. Use plan, inspect-plan, then execute only after the user explicitly says go.")
    elif a.cmd == "show":
        print(json.dumps(s.get(a.run_id), indent=2))
    elif a.cmd == "logs":
        cursor = 0
        while True:
            events = s.events(a.run_id, cursor)
            for e in events:
                print(json.dumps(e, ensure_ascii=False), flush=True)
                cursor = e["seq"]
            if not a.follow or (s.get(a.run_id)["status"] in TERMINAL and not events):
                break
            time.sleep(1)
    elif a.cmd == "stop":
        row = s.get(a.run_id)
        if row["status"] in TERMINAL:
            raise ValueError("Run is already terminal; no active job stopped")
        (s.root / "runs" / a.run_id / "cancel.request").write_text(utc())
        s.event(a.run_id, "cancel_requested", {})
        print("Cancellation requested; supervisor will stop its owned Modal apps.")
    elif a.cmd == "recover":
        print(json.dumps(recover(s, a.catalog)))
    elif a.cmd == "snapshot":
        print(json.dumps(snapshot(s, a.catalog), indent=2))
    elif a.cmd == "watch":
        while True:
            try:
                recover(s, a.catalog)
                snapshot(s, a.catalog)
            except Exception as e:
                print(clean(str(e)), file=sys.stderr, flush=True)
            time.sleep(30)
    elif a.cmd == "adopt":
        print(json.dumps(adopt(s, a.benchmark)))
    elif a.cmd == "serve":
        serve(s, a.host, a.port)
    elif a.cmd == "backup":
        import sqlite3, tarfile, tempfile

        with tempfile.TemporaryDirectory(prefix="modal-control-backup-") as td:
            backup = Path(td) / "state.sqlite"
            with s.db() as source, sqlite3.connect(backup) as destination:
                source.backup(destination)
            with tarfile.open(a.destination, "x:gz") as archive:
                archive.add(backup, arcname="state.sqlite")
                for item in s.root.iterdir():
                    if (
                        item.name.startswith("state.sqlite")
                        or item.resolve() == Path(a.destination).resolve()
                    ):
                        continue
                    archive.add(item, arcname=item.name)
        print(a.destination)
    elif a.cmd == "export":
        import tarfile

        s.get(a.run_id)
        with tarfile.open(a.destination, "x:gz") as t:
            t.add(s.root / "runs" / a.run_id, arcname=a.run_id)
        print(a.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
