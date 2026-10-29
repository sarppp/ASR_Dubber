"""AST-loaded worker wiring and mocked callbacks; no Modal/model/media execution."""
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

WORKER = Path(__file__).with_name("native_excerpt_worker.py")
TREE = ast.parse(WORKER.read_text())


def helpers():
    nodes = deepcopy([node for node in TREE.body if isinstance(node, ast.FunctionDef)])
    for node in nodes:
        node.decorator_list = []
    scope = dict(deepcopy=deepcopy, hashlib=hashlib, json=json, Path=Path,
                 ROOT=WORKER.parents[1], MAX_AUDIO_BYTES=63*1024**2, MAX_REPORT_BYTES=1024**2)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(WORKER), "exec"), scope)
    return scope


class FakeAudio:
    ndim, size = 1, 4
    def squeeze(self): return self
    def astype(self, *args, **kwargs): return self
    def tobytes(self): return struct.pack("<4f", .1, -.2, 0., .3)


class ExcerptWorkerTests(unittest.TestCase):
    def setUp(self):
        self.scope = helpers()
        self.row = dict(id="en_000", text="A complete sentence.", language="en", voice="F1", seed=123)
        self.model = Mock(sample_rate=44100)
        self.model.synthesize.return_value = (FakeAudio(), 1)
        self.tts = Mock(return_value=self.model)
        self.snapshot = Mock(return_value="/cache/pinned-snapshot")
        self.monitor = Mock()
        self.emit = Mock()
        self.chunker = Mock(return_value=[self.row["text"]])
        self.helper = SimpleNamespace(MODEL="pinned-model", REVISION="pinned-revision",
            plan_jobs=Mock(return_value=[self.row]), validate_smoke=Mock(),
            validate_scope=Mock(side_effect=lambda data: data), load_smoke=Mock(return_value={"cached": True}),
            run_excerpt=Mock(return_value={"status": "complete", "calls": 1}))
        self.modules = {
            "batch_dubbing.native_excerpt_probe": self.helper,
            "supertonic": SimpleNamespace(TTS=self.tts),
            "supertonic.utils": SimpleNamespace(chunk_text=self.chunker),
            "huggingface_hub": SimpleNamespace(snapshot_download=self.snapshot),
            "numpy": SimpleNamespace(asarray=Mock(return_value=FakeAudio()), float32="float32", random=SimpleNamespace(seed=Mock())),
            "telemetry": SimpleNamespace(emit=self.emit, start=Mock(return_value=self.monitor)),
        }

    def probe(self, *, smoke=True, artifact=None):
        with patch.dict(sys.modules, self.modules), patch.dict(os.environ), patch.object(sys, "path", list(sys.path)), \
                patch("importlib.metadata.version", return_value="1.3.1"), patch("os.sched_setaffinity"):
            return self.scope["probe"]({"frozen": True}, smoke, artifact)

    def test_complete_actual_chunk_preflight_before_model_initialization(self):
        self.helper.plan_jobs.side_effect = ValueError("chunk budget exceeded")
        with self.assertRaisesRegex(ValueError, "chunk budget"):
            self.probe()
        self.snapshot.assert_not_called(); self.tts.assert_not_called()
        self.helper.run_excerpt.assert_not_called(); self.monitor.set.assert_called_once()
        self.assertEqual(self.emit.call_args_list[0].args, ("worker_ready",))

    def test_bad_smoke_or_cache_miss_has_no_fallback_or_synthesis(self):
        self.helper.validate_smoke.side_effect = ValueError("smoke hash changed")
        with self.assertRaisesRegex(ValueError, "smoke hash"):
            self.probe(smoke=False, artifact={"bad": True})
        self.snapshot.assert_not_called(); self.tts.assert_not_called()
        self.helper.validate_smoke.side_effect = None
        self.snapshot.side_effect = FileNotFoundError("cache miss")
        with self.assertRaises(FileNotFoundError): self.probe()
        self.snapshot.assert_called_once_with("pinned-model", revision="pinned-revision", local_files_only=True)
        self.tts.assert_not_called()

    def test_native_synthesis_settings_and_exact_float_samples(self):
        captured = []
        def run(data, smoke, *, chunker, synthesize, smoke_artifact):
            captured.append(synthesize(self.row))
            return {"status": "complete", "calls": 1}
        self.helper.run_excerpt.side_effect = run
        result = self.probe()
        self.assertEqual(result["status"], "complete")
        self.tts.assert_called_once_with(model="supertonic-3", model_dir="/cache/pinned-snapshot",
                                        auto_download=False, intra_op_num_threads=2, inter_op_num_threads=1)
        self.model.get_voice_style.assert_called_once_with("F1")
        kwargs = self.model.synthesize.call_args.kwargs
        self.assertEqual({key: kwargs[key] for key in ("lang", "speed", "total_steps", "max_chunk_length", "silence_duration")},
                         dict(lang="en", speed=1., total_steps=8, max_chunk_length=300, silence_duration=.3))
        self.assertEqual(captured, [{"samples": FakeAudio().tobytes(), "sample_rate": 44100}])
        self.monitor.set.assert_called_once()

    def test_workload_passes_verified_smoke_to_shared_runner(self):
        artifact = {"saved": True}
        self.probe(smoke=False, artifact=artifact)
        self.helper.validate_smoke.assert_called_once_with(artifact, {"frozen": True})
        self.assertIs(self.helper.run_excerpt.call_args.kwargs["smoke_artifact"], artifact)

    @staticmethod
    def result(status="complete"):
        raw = b"synthetic-fixture-bytes"
        return dict(status=status, samples=[dict(file="en_000.wav", sha256=hashlib.sha256(raw).hexdigest())],
                    sequences=[], audio={"en_000.wav": raw}, naturalness_verified=False)

    def test_publish_preserves_input_and_partial_failures_exclusively(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = self.result(); before = deepcopy(result)
            self.scope["save_result"](result, root)
            self.assertEqual(result, before)
            self.assertEqual((root / "en_000.wav").read_bytes(), result["audio"]["en_000.wav"])
            with self.assertRaises(FileExistsError): self.scope["save_result"](result, root)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "partial"):
                self.scope["save_result"](self.result("failed"), Path(folder))
            self.assertEqual(json.loads((Path(folder) / "report.json").read_text())["status"], "failed")

    def test_publish_rejects_corrupt_hashes_paths_duplicates_and_transfer_overflow(self):
        variants = []
        result = self.result(); result["samples"][0]["sha256"] = "bad"; variants.append(result)
        result = self.result(); result["samples"][0]["file"] = "../escape.wav"; variants.append(result)
        result = self.result(); result["samples"] *= 2; variants.append(result)
        for result in variants:
            with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
                self.scope["save_result"](result, Path(folder))
        self.scope["MAX_AUDIO_BYTES"] = 1
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, "size"):
            self.scope["save_result"](self.result(), Path(folder))

    def test_main_reserves_output_before_remote_and_saves_remote_failure(self):
        remote = Mock(side_effect=RuntimeError("remote startup failed"))
        self.scope["probe"] = SimpleNamespace(remote=remote)
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, self.modules):
            root = Path(folder); source = root / "input.json"; source.write_text('{"frozen": true}')
            with self.assertRaisesRegex(RuntimeError, "startup"):
                self.scope["main"](str(source), str(root / "output"), smoke=True)
            report = json.loads((root / "output/smoke/report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["inference_call_count_known"])
            with self.assertRaises(FileExistsError):
                self.scope["main"](str(source), str(root / "output"), smoke=True)
        remote.assert_called_once()

    def test_workload_requires_smoke_before_remote(self):
        remote = Mock(); self.scope["probe"] = SimpleNamespace(remote=remote)
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, self.modules):
            source = Path(folder) / "input.json"; source.write_text('{}')
            with self.assertRaisesRegex(ValueError, "smoke-result"):
                self.scope["main"](str(source), str(Path(folder) / "output"))
        remote.assert_not_called()

    def test_shared_helper_result_saves_all_raw_and_native_sequences(self):
        from batch_dubbing.native_excerpt_probe import MODEL, REVISION, PACKAGE, IMAGE_ID, run_excerpt
        data = dict(schema=1, model=MODEL, revision=REVISION, package=PACKAGE, image_id=IMAGE_ID,
                    speed=1., steps=8, max_chunk_length=300, silence_duration=.3,
                    excerpt_start=0., excerpt_end=4., source_metadata={"test": "synthetic"},
                    samples=[dict(self.row, id=f"{lang}_{index:03d}", language=lang,
                                  srt_start=index*2., srt_end=index*2.+1., source_indices=[index], speaker="speaker_0")
                             for lang in ("en", "de") for index in range(2)])
        chunker = lambda text, limit: [text]
        synthesize = Mock(return_value={"samples": FakeAudio().tobytes(), "sample_rate": 44100})
        smoke = run_excerpt(data, True, chunker=chunker, synthesize=synthesize)
        encoded = json.dumps({k: v for k, v in smoke.items() if k != "audio"}).encode()
        artifact = dict(report=encoded, report_sha256=hashlib.sha256(encoded).hexdigest(), wav=smoke["audio"]["en_000.wav"])
        work = run_excerpt(data, False, chunker=chunker, synthesize=synthesize, smoke_artifact=artifact)
        self.assertEqual(work["status"], "complete")
        self.assertEqual(synthesize.call_count, 4)
        with tempfile.TemporaryDirectory() as folder:
            self.scope["save_result"](work, Path(folder))
            self.assertEqual({p.name for p in Path(folder).iterdir()},
                             {"report.json", "en_000.wav", "en_001.wav", "de_000.wav", "de_001.wav", "native_en.wav", "native_de.wav"})

    def test_explicit_cached_resources_and_no_build_or_download_configuration(self):
        node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "probe")
        decorator = node.decorator_list[0]
        args = {k.arg: k.value for k in decorator.keywords}
        for key, expected in dict(cpu=(2, 2), memory=(4096, 4096), timeout=150, startup_timeout=30,
                                  scaledown_window=2, min_containers=0, max_containers=1, retries=0).items():
            self.assertEqual(ast.literal_eval(args[key]), expected)
        calls = [n for n in ast.walk(TREE) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertFalse(any(n.func.attr in ("pip_install", "apt_install", "debian_slim", "map", "starmap", "spawn") for n in calls))
        image = next(n for n in calls if n.func.attr == "from_id")
        self.assertEqual(ast.literal_eval(image.args[0]), "im-40qUTZlx5ttgX1nwrQdvyw")


if __name__ == "__main__":
    unittest.main()
