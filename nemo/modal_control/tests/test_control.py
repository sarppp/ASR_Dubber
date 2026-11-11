import json, sys, tempfile, time, unittest
from pathlib import Path
from unittest.mock import patch
from modal_control.core import (
    Store,
    clean,
    load_catalog,
    verify_files,
    _run_approved_job as run_job,
    recover,
)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        billing_patch = patch('modal_control.billing.BillingMonitor.read', side_effect=RuntimeError('offline test'))
        billing_patch.start(); self.addCleanup(billing_patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "state")

    def job(self, smoke=None, full=None):
        def command(filename):
            return [
                sys.executable,
                "-c",
                f'import json; print(\'MC_EVENT {{"kind":"worker_ready"}}\',flush=True); open({filename!r},"w").write(json.dumps({{"status":"complete"}}))',
            ]

        return {
            "cwd": str(self.root),
            "smoke_command": smoke or command("smoke.json"),
            "command": full or command("full.json"),
            "smoke_success_files": ["smoke.json"],
            "success_files": ["full.json"],
            "source_files": [],
        }

    def catalog(self, job):
        p = self.root / "catalog.json"
        p.write_text(json.dumps({"version": 1, "jobs": {"test": job}}))
        return p

    def test_success_requires_both_stages_and_archives(self):
        rid, rc = run_job(self.store, self.catalog(self.job()), "test")
        self.assertEqual(rc, 0)
        self.assertEqual(self.store.get(rid)["status"], "succeeded")
        self.assertTrue(
            (self.store.root / "runs" / rid / "artifact_manifest.json").exists()
        )

    def test_failed_smoke_prevents_full(self):
        rid, rc = run_job(
            self.store,
            self.catalog(self.job(smoke=[sys.executable, "-c", "raise SystemExit(3)"])),
            "test",
        )
        self.assertEqual(rc, 1)
        self.assertFalse((self.root / "full.json").exists())
        self.assertEqual(self.store.get(rid)["status"], "preflight_failed")

    def test_stale_artifact_rejected(self):
        (self.root / "full.json").write_text('{"status":"complete"}')
        with self.assertRaisesRegex(ValueError, "Stale"):
            verify_files(self.job(), started=time.time() + 5)

    def test_no_readiness_is_not_success(self):
        cmd = [
            sys.executable,
            "-c",
            'open("smoke.json","w").write(\'{"status":"complete"}\')',
        ]
        rid, rc = run_job(self.store, self.catalog(self.job(smoke=cmd)), "test")
        self.assertEqual(rc, 1)

    def test_secret_redaction_remains_valid_json(self):
        data = {
            "HF_TOKEN": "hf_" + "x" * 25,
            "password": "do not show this",
            "text": "Authorization=Bearer abc123",
        }
        safe = clean(json.dumps(data))
        json.loads(safe)
        self.assertNotIn("do not show", safe)
        self.assertNotIn("x" * 25, safe)
        self.assertNotIn("abc123", safe)

    def test_catalog_rejects_inline_credentials(self):
        job = self.job()
        job["env"] = {"HF_TOKEN": "private"}
        with self.assertRaisesRegex(ValueError, "credentials"):
            load_catalog(self.catalog(job))

    def test_crash_stops_only_observed_app(self):
        job = self.job(
            smoke=[
                sys.executable,
                "-c",
                'print("https://modal.com/apps/account/main/ap-1234567890ABC");print("ModuleNotFoundError: missing")',
            ]
        )
        with patch("modal_control.core.cancel_apps") as stop:
            rid, rc = run_job(self.store, self.catalog(job), "test")
            self.assertEqual(rc, 1)
            self.assertTrue(stop.called)
            self.assertEqual(stop.call_args.args[2], {"ap-1234567890ABC"})

    def test_deadline_stops_child(self):
        job = self.job(smoke=[sys.executable, "-c", "import time;time.sleep(20)"])
        job["max_seconds"] = 0.1
        rid, rc = run_job(self.store, self.catalog(job), "test")
        self.assertEqual(rc, 1)

    def test_source_change_prevents_promotion(self):
        source = self.root / "source.py"
        source.write_text("before")
        job = self.job()
        job["source_files"] = ["source.py"]
        job["smoke_command"][-1] += ';open("source.py","w").write("after")'
        rid, rc = run_job(self.store, self.catalog(job), "test")
        self.assertEqual(rc, 1)
        self.assertFalse((self.root / "full.json").exists())

    def test_catalog_gpu_workers_are_t4_only(self):
        import ast

        root = Path(__file__).resolve().parents[2]
        catalog = json.loads((root / "modal_control/catalog/jobs.json").read_text())
        self.assertEqual(catalog["hardware_policy"]["allowed_gpus"], ["T4"])
        for job in catalog["jobs"].values():
            if job.get("hardware") != "gpu":
                continue
            module = next(
                Path(job["cwd"]) / arg
                for arg in job["command"]
                if Path(arg).name.startswith("modal_bench_t4") and arg.endswith(".py")
            )
            tree = ast.parse(module.read_text())
            gpu_values = [
                kw.value
                for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                for kw in n.keywords
                if kw.arg == "gpu"
            ]
            self.assertEqual(len(gpu_values), 1)
            self.assertIsInstance(gpu_values[0], ast.IfExp)
            self.assertEqual(ast.literal_eval(gpu_values[0].body), "T4")

    def test_backup_has_consistent_database(self):
        import subprocess, tarfile, sqlite3

        rid = self.store.create("saved", {})
        target = self.root / "backup.tar.gz"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "modal_control",
                "--state",
                str(self.store.root),
                "backup",
                str(target),
            ],
            check=True,
            capture_output=True,
        )
        with tarfile.open(target) as archive:
            data = archive.extractfile("state.sqlite").read()
        restored = self.root / "restored.sqlite"
        restored.write_bytes(data)
        with sqlite3.connect(restored) as db:
            self.assertEqual(db.execute("SELECT id FROM runs").fetchone()[0], rid)

    def test_recover_dead_supervisor(self):
        rid = self.store.create("test", {})
        catalog = self.catalog(self.job())
        with self.store.db() as c:
            c.execute("UPDATE runs SET pid=? WHERE id=?", (99999999, rid))
        with (
            patch("modal_control.core.os.kill", side_effect=ProcessLookupError),
            patch("modal_control.core.cancel_apps") as stop,
        ):
            self.assertEqual(recover(self.store, catalog), [rid])
            self.assertTrue(stop.called)
        self.assertEqual(self.store.get(rid)["status"], "interrupted")

    def test_events_persist_after_reopen(self):
        rid = self.store.create("job", {})
        self.store.event(rid, "test", {"value": 42})
        self.assertEqual(Store(self.store.root).events(rid)[-1]["payload"]["value"], 42)


if __name__ == "__main__":
    unittest.main()
