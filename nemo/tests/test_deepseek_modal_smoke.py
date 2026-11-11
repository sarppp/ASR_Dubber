"""Exercise the actual Modal wrapper and controller contract entirely offline."""
import ast
import hashlib
import io
import json
import shutil
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modal_control.core import verify_files


class ModalWrapperSmoke(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        source = ast.parse((ROOT / "batch_dubbing/deepseek_t4_worker.py").read_text())
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "validate")
        function.decorator_list = []
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))

        def mapped_path(value):
            text = str(value)
            return self.root / text.lstrip("/") if text.startswith(("/input", "/cache")) else Path(value)
        self.mapped_path = mapped_path

        self.volume = SimpleNamespace(commit=Mock())
        namespace = dict(Path=mapped_path, hashlib=hashlib, json=json, io=io,
                         tarfile=tarfile, volume=self.volume)
        exec(compile(module, "offline-modal-wrapper", "exec"), namespace)
        self.validate = namespace["validate"]
        inputs = self.root / "input"
        inputs.mkdir()
        (inputs / "voice_reference.wav").write_bytes(b"reference fixture")
        (inputs / "scope.json").write_text(json.dumps({"cues": 10, "duration_seconds": 29.479,
            "max_speed": 1., "input_sha256": {}, "model": "fixture", "revision": "fixture"}))
        self.omit_audio = False
        self.count = 2

    def fake_dub(self, command, **kwargs):
        work = Path(command[command.index("--workdir") + 1])
        cache = work / "cache/qwen/fixture"
        (cache / "temp").mkdir(parents=True)
        segments = []
        for i in range(1, self.count + 1):
            if not (self.omit_audio and i == self.count):
                with wave.open(str(cache / "temp" / f"seg_{i:04d}.wav"), "wb") as audio:
                    audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                    audio.writeframes(b"\x20\x00" * 24000)
            segments.append({"index": i, "needed": 1., "tempo": 1.2, "status": "failed"})
        (work / "fit_report.json").write_text(json.dumps({"cache_dir": str(cache), "segments": segments}))
        return SimpleNamespace(returncode=1, stdout="fixture", stderr="fit rejected")

    def invoke(self, smoke=True, gpu="Tesla T4"):
        telemetry = SimpleNamespace(emit=Mock(), start=Mock(return_value=SimpleNamespace(set=Mock())))
        copy = shutil.copy2
        with patch.dict(sys.modules, {"telemetry": telemetry}), \
             patch.object(sys, "path", list(sys.path)), \
             patch("shutil.copy2", side_effect=lambda src, dst: copy(self.mapped_path(src), dst)), \
             patch("subprocess.check_output", return_value=gpu), \
             patch("subprocess.run", side_effect=self.fake_dub) as run:
            result = self.validate(smoke)
            run.assert_called_once()
        destination = self.root / "download"
        destination.mkdir()
        with tarfile.open(fileobj=io.BytesIO(result)) as archive:
            archive.extractall(destination, filter="data")
        name = "smoke.report.json" if smoke else "workload.report.json"
        report = json.loads((destination / name).read_text())
        job = {"cwd": str(destination), "smoke_success_files": [name]}
        return report, job

    def test_fit_failure_is_a_completed_measurement_not_a_successful_dub(self):
        report, job = self.invoke()
        self.assertEqual(report["measurement_status"], "complete")
        self.assertEqual(report["dub_status"], "needs_review")
        self.assertEqual(report["measured_cues"], 2)
        self.assertEqual(len(verify_files(job, smoke=True)), 1)
        self.volume.commit.assert_called_once()

    def test_ten_cue_workload_contract(self):
        self.count = 10
        report, job = self.invoke(smoke=False)
        self.assertEqual(report["expected_cues"], 10)
        self.assertEqual(len(report["raw_seconds"]), 10)
        verify_files(job, smoke=True)

    def test_missing_audio_cannot_pass_controller_gate(self):
        self.omit_audio = True
        report, job = self.invoke()
        self.assertEqual(report["measurement_status"], "failed")
        with self.assertRaisesRegex(ValueError, "completion"):
            verify_files(job, smoke=True)

    def test_non_t4_is_rejected_before_dub_launch(self):
        with self.assertRaisesRegex(RuntimeError, "T4 is required"):
            self.invoke(gpu="NVIDIA L4")


if __name__ == "__main__":
    unittest.main()
