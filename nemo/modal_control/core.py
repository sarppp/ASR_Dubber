from __future__ import annotations
import contextlib, datetime, hashlib, json, os, re, selectors, signal, sqlite3, subprocess, sys, time, uuid, shutil
from pathlib import Path

TERMINAL = {
    "succeeded",
    "failed",
    "cancelled",
    "startup_timeout",
    "idle_timeout",
    "deadline_exceeded",
    "preflight_failed",
    "interrupted",
    "observed",
}
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
APP = re.compile(r"\bap-[A-Za-z0-9]{10,}\b")
FATAL = re.compile(
    r"crash.looping|repeatedly failing to start|Runner failed with exception|ModuleNotFoundError:|ImportError:|FileNotFoundError:",
    re.I,
)
SECRET = re.compile(
    r"(?i)((?:api[_-]?key|access[_-]?token|hf_token|modal_token_secret|authorization|password|secret)\s*[=:]\s*)([^\s,;\"{}]+)"
)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def clean(text):
    text = ANSI.sub("", text)
    text = re.sub(r"\bhf_[A-Za-z0-9]{15,}\b", "[REDACTED_HF_TOKEN]", text)
    text = re.sub(r"\b(?:ak|as)-[A-Za-z0-9_-]{15,}\b", "[REDACTED_MODAL_TOKEN]", text)
    text = re.sub(r"(?i)Bearer\s+[^\s\"{}]+", "Bearer [REDACTED]", text)
    text = re.sub(
        r'(?i)("(?:api[_-]?key|access[_-]?token|hf_token|modal_token_secret|authorization|password|secret)"\s*:\s*)"[^"]*"',
        r'\1"[REDACTED]"',
        text,
    )
    return SECRET.sub(r"\1[REDACTED]", text)


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.db() as c:
            c.executescript("""CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,job TEXT,status TEXT,created TEXT,updated TEXT,pid INTEGER,manifest TEXT);
CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT,at TEXT,kind TEXT,payload TEXT);
CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT,at TEXT,severity TEXT,code TEXT,message TEXT);
CREATE INDEX IF NOT EXISTS events_run ON events(run_id,seq);""")

    @contextlib.contextmanager
    def db(self):
        c = sqlite3.connect(self.root / "state.sqlite", timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=FULL")
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def create(self, job, manifest):
        rid = utc()[:10] + "-" + uuid.uuid4().hex[:12]
        now = utc()
        d = self.root / "runs" / rid
        d.mkdir(parents=True)
        safe = json.loads(clean(json.dumps(manifest)))
        atomic(d / "manifest.json", safe)
        with self.db() as c:
            c.execute(
                "INSERT INTO runs VALUES(?,?,?,?,?,?,?)",
                (rid, job, "created", now, now, os.getpid(), json.dumps(safe)),
            )
        self.event(rid, "created", {"job": job})
        return rid

    def event(self, rid, kind, payload):
        data = json.loads(clean(json.dumps(payload, default=str)))
        at = utc()
        with self.db() as c:
            cursor = c.execute(
                "INSERT INTO events(run_id,at,kind,payload) VALUES(?,?,?,?)",
                (rid, at, kind, json.dumps(data)),
            )
            record = {
                "seq": cursor.lastrowid,
                "run_id": rid,
                "at": at,
                "kind": kind,
                "payload": data,
            }
        # SQLite is authoritative; JSONL is an additional append-only export.
        p = self.root / "runs" / rid / "events.jsonl"
        with p.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
        return record

    def state(self, rid, status, **details):
        with self.db() as c:
            c.execute(
                "UPDATE runs SET status=?,updated=? WHERE id=?", (status, utc(), rid)
            )
        self.event(rid, "state", {"status": status, **details})

    def alert(self, rid, code, message, severity="error"):
        with self.db() as c:
            c.execute(
                "INSERT INTO alerts(run_id,at,severity,code,message) VALUES(?,?,?,?,?)",
                (rid, utc(), severity, code, clean(message)),
            )
        self.event(
            rid, "alert", {"code": code, "message": message, "severity": severity}
        )

    def overview(self):
        with self.db() as c:
            runs = [
                dict(x)
                for x in c.execute(
                    "SELECT id,job,status,created,updated,pid FROM runs ORDER BY created DESC"
                )
            ]
            alerts = [
                dict(x)
                for x in c.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 100")
            ]
        return {
            "observed_at": utc(),
            "runs": runs,
            "alerts": alerts,
            "state_root": str(self.root),
            "billing": json.loads((self.root / "latest_billing.json").read_text()) if (self.root / "latest_billing.json").exists() else None,
        }

    def get(self, rid):
        with self.db() as c:
            r = c.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
        if not r:
            raise KeyError(rid)
        return dict(r)

    def events(self, rid, after=0, limit=500):
        with self.db() as c:
            rows = c.execute(
                "SELECT * FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?",
                (rid, after, min(limit, 2000)),
            ).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]


def load_catalog(path):
    d = json.loads(Path(path).read_text())
    if d.get("version") != 1 or not isinstance(d.get("jobs"), dict):
        raise ValueError("Unsupported catalog")
    for name, job in d["jobs"].items():
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", name):
            raise ValueError("Invalid job name")
        for key in ("command", "smoke_command"):
            if (
                not isinstance(job.get(key), list)
                or not job[key]
                or any(not isinstance(v, str) for v in job[key])
            ):
                raise ValueError(f"{name}: {key} must be an argv array")
        if not Path(job["cwd"]).is_dir():
            raise ValueError(f"{name}: missing cwd")
        if any(
            re.search(r"(?i)token|secret|password|api.?key", k)
            for k in job.get("env", {})
        ):
            raise ValueError(
                "Put credentials in the environment or env_file, never the catalog"
            )
        if not job.get("success_files") or not job.get("smoke_success_files"):
            raise ValueError(f"{name}: output verification required")
    return d


def verify_files(job, smoke=False, started=None):
    outputs = []
    for filename in job["smoke_success_files" if smoke else "success_files"]:
        p = Path(job["cwd"]) / filename
        if not p.is_file():
            raise ValueError(f"Missing required artifact: {p}")
        if started is not None and p.stat().st_mtime < started - 1:
            raise ValueError(f"Stale required artifact: {p}")
        if p.suffix == ".json":
            data = json.loads(p.read_text())
            if data.get("status") != "complete":
                raise ValueError(f"Artifact does not report completion: {p}")
        outputs.append({"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)})
    return outputs


def cancel_apps(store, rid, app_ids, modal_command, env):
    for app in sorted(app_ids):
        if not APP.fullmatch(app):
            raise ValueError("Invalid owned app id")
        try:
            r = subprocess.run(
                [*modal_command, "app", "stop", app],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            store.event(
                rid,
                "cloud_stop",
                {
                    "app_id": app,
                    "returncode": r.returncode,
                    "output": clean(r.stdout + r.stderr)[-6000:],
                },
            )
            if r.returncode:
                store.alert(
                    rid,
                    "cloud_stop_failed",
                    f"{app}: stop failed; manual reconciliation required",
                )
        except Exception as e:
            store.alert(rid, "cloud_stop_failed", f"{app}: {e}")


def stage(store, rid, job, smoke, env, modal_command):
    name = "smoke" if smoke else "workload"
    cmd = job["smoke_command" if smoke else "command"]
    d = store.root / "runs" / rid
    wall_start = time.time()
    start = time.monotonic()
    last_worker = start
    submitted = None
    ready = False
    apps = set()
    stopping = None
    store.state(rid, "preflight" if smoke else "building", stage=name)
    p = subprocess.Popen(
        cmd,
        cwd=job["cwd"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        start_new_session=True,
    )
    store.event(rid, "process_started", {"stage": name, "pid": p.pid, "argv": cmd})
    atomic(
        d / "active_process.json",
        {"pid": p.pid, "stage": name, "apps": [], "supervisor_pid": os.getpid()},
    )
    sel = selectors.DefaultSelector()
    buffers = {}
    files = {}
    for label, stream in [("stdout", p.stdout), ("stderr", p.stderr)]:
        os.set_blocking(stream.fileno(), False)
        sel.register(stream, selectors.EVENT_READ, label)
        buffers[label] = b""
        files[label] = (d / f"{name}.{label}.log").open("a")
    try:
        while sel.get_map() or p.poll() is None:
            now = time.monotonic()
            if (d / "cancel.request").exists():
                stopping = "cancelled"
            if now - start > job.get("max_seconds", 3600):
                stopping = "deadline_exceeded"
            if (
                submitted
                and not ready
                and now - submitted > job.get("startup_timeout_seconds", 180)
            ):
                stopping = "startup_timeout"
            if (
                not ready
                and not submitted
                and now - start > job.get("build_timeout_seconds", 1200)
            ):
                stopping = "startup_timeout"
            if ready and now - last_worker > job.get("idle_timeout_seconds", 180):
                stopping = "idle_timeout"
            if stopping:
                store.alert(rid, stopping, f"{name} stopped: {stopping}")
                cancel_apps(store, rid, apps, modal_command, env)
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGINT)
                    try:
                        p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL)
                break
            for key, _ in sel.select(timeout=1):
                label = key.data
                try:
                    chunk = os.read(key.fileobj.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    sel.unregister(key.fileobj)
                    chunk = b"\n" if buffers[label] else b""
                buffers[label] += chunk
                # CR progress bars are saved as individual timestamped lines.
                parts = re.split(b"[\r\n]", buffers[label])
                buffers[label] = parts.pop()
                for raw in parts:
                    if not raw:
                        continue
                    line = clean(raw.decode("utf-8", "replace"))
                    at = utc()
                    files[label].write(at + " " + line + "\n")
                    files[label].flush()
                    store.event(
                        rid, "log", {"stage": name, "stream": label, "text": line}
                    )
                    new = set(APP.findall(line)) - apps
                    if new:
                        apps.update(new)
                        store.event(rid, "cloud_apps", {"owned_ids": sorted(apps)})
                        atomic(
                            d / "active_process.json",
                            {
                                "pid": p.pid,
                                "stage": name,
                                "apps": sorted(apps),
                                "supervisor_pid": os.getpid(),
                            },
                        )
                    if "Created function" in line and not submitted:
                        submitted = time.monotonic()
                        store.state(rid, "waiting", stage=name)
                    marker = line.find("MC_EVENT ")
                    if marker >= 0:
                        try:
                            e = json.loads(line[marker + 9 :])
                            store.event(rid, "worker_event", e)
                            if e.get("kind") not in ("cloud_stats", "submitted"):
                                last_worker = time.monotonic()
                            if e.get("kind") == "worker_ready":
                                if job.get("_approved_hardware") == "T4" and not re.search(r"\bT4\b", str(e.get("gpu", ""))):
                                    stopping = "hardware_mismatch"
                                    store.alert(rid, stopping, "Worker did not attest approved T4 hardware")
                                    continue
                                ready = True
                                store.state(rid, "running", stage=name)
                        except json.JSONDecodeError:
                            store.alert(
                                rid,
                                "invalid_telemetry",
                                "Malformed worker event",
                                severity="warning",
                            )
                    if "preemption" in line.lower():
                        store.alert(rid, "preemption", line, severity="warning")
                        ready = False
                        submitted = time.monotonic()
                    if FATAL.search(line):
                        stopping = "failed"
                        store.alert(rid, "startup_or_import_failure", line)
            if p.poll() is not None and not sel.get_map():
                break
        rc = p.wait(timeout=10)
    except BaseException:
        cancel_apps(store, rid, apps, modal_command, env)
        raise
    finally:
        sel.close()
        for f in files.values():
            f.close()
        p.stdout.close()
        p.stderr.close()
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout=10)
    store.event(
        rid,
        "process_exited",
        {"stage": name, "returncode": rc, "seconds": time.monotonic() - start},
    )
    if stopping:
        cancel_apps(store, rid, apps, modal_command, env)
        raise RuntimeError(stopping)
    if rc:
        raise RuntimeError(f"{name} exited {rc}")
    if not ready:
        raise RuntimeError("Worker never reported readiness")
    verified = verify_files(job, smoke, wall_start)
    store.event(rid, "artifacts_verified", {"stage": name, "files": verified})
    return verified


def run_job(store, catalog_path, name, *, plan_id=None, confirmation=None, approval_message=None):
    if not plan_id:
        raise ValueError("Direct runs are disabled. Prepare a plan and have the human release it.")
    from .plans import release_plan
    plan = release_plan(store, plan_id, confirmation, approval_message)
    payload = plan["payload"]
    if str(Path(catalog_path).resolve()) != payload["catalog_path"] or name != payload["job"]:
        raise ValueError("Run does not match released plan")
    plan["approval_message"] = approval_message
    return _run_approved_job(store, catalog_path, name, plan=plan)


def _run_approved_job(store, catalog_path, name, plan=None):
    catalog = load_catalog(catalog_path)
    job = catalog["jobs"][name]
    if plan:
        if digest(catalog_path) != plan["payload"]["catalog_sha256"] or job != plan["payload"]["configuration"]:
            raise ValueError("Catalog changed after release")
        for item in plan["payload"]["sources"]:
            if digest(item["path"]) != item["sha256"]:
                raise ValueError("Source changed after release")
        job = dict(job, _approved_hardware=job["review"]["hardware"])
    source = []
    for filename in job.get("source_files", []):
        p = Path(job["cwd"]) / filename
        source.append({"path": str(p), "sha256": digest(p)})
    manifest = {
        "release_plan": plan,
        "job": name,
        "catalog_path": str(Path(catalog_path).resolve()),
        "catalog_sha256": digest(catalog_path),
        "configuration": job,
        "sources": source,
        "python": sys.version,
        "created_at": utc(),
    }
    rid = store.create(name, manifest)
    print(rid, flush=True)
    provenance = store.root / "runs" / rid / "sources"
    provenance.mkdir()
    shutil.copy2(catalog_path, provenance / "catalog.json")
    for i, item in enumerate(source):
        shutil.copy2(item["path"], provenance / f"{i:03d}-{Path(item['path']).name}")
    env = os.environ.copy()
    env.update(job.get("env", {}))
    env["MODAL_CONTROL_RUN_ID"] = rid
    env["PYTHONUNBUFFERED"] = "1"
    if catalog.get("env_file"):
        # Only load keys into process memory; never persist values.
        from dotenv import dotenv_values

        env.update(
            {
                k: v
                for k, v in dotenv_values(catalog["env_file"]).items()
                if v is not None
            }
        )
    modal_command = catalog.get("modal_command", [sys.executable, "-m", "modal"])
    lock = store.root / (name + ".lock")
    import fcntl

    with lock.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            store.state(rid, "failed", reason="Another supervisor owns this job")
            return rid, 1
        from .billing import BillingMonitor
        billing = BillingMonitor(store, rid, catalog).start()
        try:
            try:
                stage(store, rid, job, True, env, modal_command)
            except Exception as exc:
                store.event(rid, "preflight_rejected", {"reason": str(exc)})
                if str(exc) == "cancelled":
                    raise
                raise RuntimeError("preflight_failed") from exc
            if plan:
                if digest(catalog_path) != plan["payload"]["catalog_sha256"]:
                    raise RuntimeError("Catalog changed after approval")
                for item in plan["payload"]["controller_sources"]:
                    if digest(item["path"]) != item["sha256"]:
                        raise RuntimeError("Controller changed after approval")
            # Recheck exact source before promotion from smoke to the full workload.
            for item in source:
                if digest(item["path"]) != item["sha256"]:
                    raise RuntimeError(
                        "Source changed after preflight; refusing full run"
                    )
            store.event(rid, "preflight_passed", {"source_identity": source})
            artifacts = stage(store, rid, job, False, env, modal_command)
            archive = store.root / "runs" / rid / "artifacts"
            archive.mkdir(exist_ok=True)
            for i, artifact in enumerate(artifacts):
                shutil.copy2(
                    artifact["path"], archive / f"{i:03d}-{Path(artifact['path']).name}"
                )
            for root in job.get("artifact_roots", []):
                p = Path(job["cwd"]) / root
                if p.is_dir():
                    shutil.copytree(p, archive / p.name, dirs_exist_ok=True)
            inventory = []
            for p in archive.rglob("*"):
                if p.is_file():
                    inventory.append(
                        {
                            "path": str(p.relative_to(archive)),
                            "bytes": p.stat().st_size,
                            "sha256": digest(p),
                        }
                    )
            atomic(store.root / "runs" / rid / "artifact_manifest.json", inventory)
            store.state(rid, "succeeded", verified_artifacts=len(inventory))
            return rid, 0
        except BaseException as e:
            status = (
                "cancelled"
                if isinstance(e, KeyboardInterrupt)
                else (str(e) if str(e) in TERMINAL else "failed")
            )
            store.alert(rid, "run_failed", str(e))
            store.state(rid, status, error=str(e))
            return rid, 1
        finally:
            billing.finish()


def recover(store, catalog_path):
    """Reconcile dead supervisors; never signal an unverified local PID."""
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
    recovered = []
    for row in store.overview()["runs"]:
        if row["status"] in TERMINAL:
            continue
        try:
            os.kill(row["pid"], 0)
            continue
        except PermissionError:
            continue
        except ProcessLookupError:
            pass
        d = store.root / "runs" / row["id"]
        p = d / "active_process.json"
        active = json.loads(p.read_text()) if p.exists() else {}
        store.alert(
            row["id"],
            "supervisor_lost",
            "Supervisor exited without a terminal state; reconciling recorded cloud apps",
        )
        cancel_apps(
            store,
            row["id"],
            set(active.get("apps", [])),
            catalog.get("modal_command", [sys.executable, "-m", "modal"]),
            env,
        )
        store.state(
            row["id"], "interrupted", reason="Supervisor process no longer exists"
        )
        recovered.append(row["id"])
    return recovered
