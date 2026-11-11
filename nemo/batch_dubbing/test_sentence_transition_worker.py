"""Synthetic sample bytes and AST-only worker loading; no Modal/model imports."""
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from batch_dubbing.sentence_transition_probe import build_fixture

WORKER = Path(__file__).with_name("sentence_transition_worker.py")
TREE = ast.parse(WORKER.read_text())


def helpers():
    nodes = deepcopy([node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name != "probe"])
    for node in nodes:
        node.decorator_list = []
    scope = {"hashlib": hashlib, "json": json, "math": math, "Path": Path, "struct": struct}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(WORKER), "exec"), scope)
    return scope


def chunker(text, maximum):
    # Deliberate deterministic fixture stub; never imports a model package.
    if len(text) > maximum:
        first, last = text.rsplit(". ", 1)
        return [first + ".", last]
    return [text]


def measurement(row):
    count = (int(row["id"].split("_")[-1]) + 1) * 10
    # Include negative and signed-zero samples to expose bit-changing assembly.
    samples = (struct.pack("<5f", .25, -.75, -0., .125, .5) * (count // 5))
    return {"samples": samples, "sample_rate": 44100}


def artifact(scope):
    report = scope["run_stage"](build_fixture(), True, chunker=chunker, synthesize=measurement)
    wav = report.pop("audio")["sample_0.wav"]
    encoded = json.dumps(report).encode()
    return {"report": encoded, "report_sha256": scope["digest"](encoded), "wav": wav}


class TransitionWorkerTests(unittest.TestCase):
    def setUp(self):
        self.scope = helpers()
        self.fixture = build_fixture()

    def run_work(self, callback=measurement, **kwargs):
        return self.scope["run_stage"](self.fixture, False, chunker=chunker, synthesize=callback,
                                        smoke_artifact=artifact(self.scope), **kwargs)

    def test_six_api_calls_eight_internal_calls_smoke_is_reused(self):
        call = Mock(side_effect=measurement)
        smoke = self.scope["run_stage"](self.fixture, True, chunker=chunker, synthesize=call)
        encoded = json.dumps({k: v for k, v in smoke.items() if k != "audio"}).encode()
        prior = {"report": encoded, "report_sha256": self.scope["digest"](encoded), "wav": smoke["audio"]["sample_0.wav"]}
        work = self.scope["run_stage"](self.fixture, False, chunker=chunker, synthesize=call, smoke_artifact=prior)
        self.assertEqual(work["status"], "complete")
        self.assertEqual([args.args[0]["id"] for args in call.call_args_list], [f"sample_{i}" for i in range(6)])
        self.assertEqual((smoke["calls"], work["calls"], smoke["internal_calls"], work["internal_calls"]), (1, 5, 1, 7))
        self.assertEqual(work["audio"]["sample_0.wav"], smoke["audio"]["sample_0.wav"])
        self.assertTrue(work["samples"][0]["reused_from_smoke"])
        self.assertEqual(len(work["audio"]), 8)

    def test_full_endings_exact_slices_and_exact_zero_gaps_for_both_languages(self):
        result = self.run_work()
        for seq in result["sequences"]:
            raw = result["audio"][seq["file"]][56:]
            previous_end = None
            for row in seq["schedule"]:
                expected = measurement(next(item for item in self.fixture["samples"] if item["id"] == row["id"]))["samples"]
                actual = raw[row["start_frame"] * 4:row["end_frame"] * 4]
                self.assertEqual(actual, expected)
                self.assertEqual(row["source_samples_sha256"], row["sequence_slice_sha256"])
                if previous_end is not None:
                    self.assertEqual(row["start_frame"] - previous_end, 17640)
                    self.assertEqual(raw[previous_end * 4:row["start_frame"] * 4], b'\0' * (17640 * 4))
                previous_end = row["end_frame"]
            self.assertEqual(len(raw), previous_end * 4)
            self.assertTrue(seq["sample_content_retention_verified"])
        for key in ("timing_verified", "naturalness_verified", "pronunciation_verified", "duration_conditioning", "pause_conditioning"):
            self.assertFalse(result[key])
        self.assertEqual(result["synthesis_speed"], 1.)
        self.assertEqual(result["internal_chunk_silence_seconds"], .3)
        self.assertEqual(result["synthetic_join_silence_seconds"], .4)

    def test_corrupt_smoke_report_or_wave_rejected_before_callback(self):
        valid = artifact(self.scope)
        variants = [None, {**valid, "report_sha256": "0" * 64}, {**valid, "wav": valid["wav"][:-4]}]
        for field, value in (("status", "failed"), ("calls", 2), ("calls", True), ("internal_calls", 2),
                             ("input_sha256", "changed"), ("synthesis_speed", 1.05), ("playback_rate", 1.5),
                             ("speech_trimmed", True), ("internal_chunk_silence_seconds", 0)):
            report = json.loads(valid["report"])
            report[field] = value
            encoded = json.dumps(report).encode()
            variants.append({**valid, "report": encoded, "report_sha256": self.scope["digest"](encoded)})
        for prior in variants:
            call = Mock()
            with self.subTest(prior=str(prior)[:70]), self.assertRaises((ValueError, TypeError)):
                self.scope["run_stage"](self.fixture, False, chunker=chunker, synthesize=call, smoke_artifact=prior)
            call.assert_not_called()

    def test_failed_callback_preserves_prior_raw_and_stops_without_retry(self):
        call = Mock(side_effect=[measurement(self.fixture["samples"][1]), RuntimeError("stop"), measurement(self.fixture["samples"][3])])
        result = self.run_work(call)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["calls"], 2)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(set(result["audio"]), {"sample_0.wav", "sample_1.wav"})
        self.assertEqual(result["sequences"], [])
        self.assertEqual(result["active_sample_id"], "sample_2")

    def test_invalid_samples_fail_without_later_synthesis(self):
        outputs = [None, {}, {"samples": b'', "sample_rate": 44100},
                   {"samples": b'123', "sample_rate": 44100},
                   {"samples": struct.pack('<f', float('nan')), "sample_rate": 44100},
                   {"samples": struct.pack('<f', float('inf')), "sample_rate": 44100},
                   {"samples": struct.pack('<f', 0), "sample_rate": 44100},
                   {"samples": struct.pack('<f', .5), "sample_rate": 24000}]
        for output in outputs:
            callback = Mock(return_value=output)
            result = self.run_work(callback)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(callback.call_count, 1)
            self.assertEqual(set(result["audio"]), {"sample_0.wav"})

    def test_ninety_second_limit_rejects_before_inspecting_sample_values(self):
        payload = b'\0' * (90 * 44100 * 4 + 4)
        with self.assertRaisesRegex(ValueError, "90 seconds"):
            self.scope["validate_samples"](payload)

    def test_stage_duration_cap_stops_callback_sequence(self):
        original = self.scope["validate_samples"]
        count = 0
        def oversized_total(payload, sample_rate=44100):
            nonlocal count
            count += 1
            # First validation reads reused smoke; generated outputs each90s.
            return original(payload, sample_rate) if count == 1 else (90 * 44100, .5)
        self.scope["validate_samples"] = oversized_total
        callback = Mock(side_effect=measurement)
        result = self.run_work(callback)
        # artifact() also validates its generated smoke before reused validation.
        self.assertEqual(result["status"], "failed")
        self.assertIn("300 seconds", result["error"])
        self.assertEqual(callback.call_count, 4)

    def test_transfer_bound_preserves_completed_raw_outputs(self):
        import batch_dubbing.sentence_transition_probe as fixture_module
        with patch.object(fixture_module, "MAX_BYTES", 1024 * 1024 + 500):
            self.fixture = build_fixture()
            result = self.run_work()
        self.assertEqual(result["status"], "failed")
        self.assertIn("64 MiB", result["error"])
        self.assertLessEqual(result["audio_transfer_bytes"], 500)
        self.assertEqual(result["metadata_reserve_bytes"], 1024 * 1024)
        self.assertGreaterEqual(len(result["audio"]), 1)

    def test_lossy_or_colliding_placement_is_rejected(self):
        from batch_dubbing.sentence_transition_probe import plan_sequence
        rows = [{"id": "a", "native_frames": 10}, {"id": "b", "native_frames": 5}]
        self.assertEqual(plan_sequence(rows, starts=[0, 10])["status"], "blocked")
        self.assertEqual(plan_sequence(rows, slot_ends=[9, 99999])["status"], "blocked")
        with self.assertRaisesRegex(ValueError, "count changed"):
            self.scope["assemble_sequence"](rows, {"a": b'1234', "b": b'1234' * 5})

    def test_failed_stage_saves_report_and_partial_audio_then_raises(self):
        result = self.run_work(Mock(side_effect=[measurement(self.fixture["samples"][1]), RuntimeError("stop")]))
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "out"
            with self.assertRaises(RuntimeError):
                self.scope["save_result"](result, destination)
            self.assertEqual(json.loads((destination / "report.json").read_text())["status"], "failed")
            self.assertEqual(len(list(destination.glob("*.wav"))), 2)
            with self.assertRaises(FileExistsError):
                self.scope["save_result"](self.run_work(), destination)

    def test_workload_local_preflight_blocks_bad_smoke_before_remote(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.scope.update(ROOT=root, OUTPUT=root / 'out', probe=SimpleNamespace(remote=Mock()))
            with patch.object(sys, "path", list(sys.path)), self.assertRaises(ValueError):
                self.scope["main"](False, "")
            report = root / "report.json"
            report.write_text('{}')
            (root / 'sample_0.wav').write_bytes(b'invalid')
            with patch.object(sys, "path", list(sys.path)), self.assertRaises(ValueError):
                self.scope["main"](False, str(report))
            self.scope["probe"].remote.assert_not_called()

    def test_worker_resource_and_cache_contract(self):
        calls = [node for node in ast.walk(TREE) if isinstance(node, ast.Call)]
        forbidden = {'pip_install', 'apt_install', 'debian_slim', 'run_commands', 'from_registry', 'map', 'starmap', 'spawn'}
        self.assertFalse(any(isinstance(call.func, ast.Attribute) and call.func.attr in forbidden for call in calls))
        fn = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == 'probe')
        args = {kw.arg: ast.literal_eval(kw.value) for kw in fn.decorator_list[0].keywords if kw.arg not in ('image', 'volumes')}
        self.assertEqual(args, {'cpu': (2, 2), 'memory': (4096, 4096), 'timeout': 150, 'startup_timeout': 30,
                                'scaledown_window': 2, 'min_containers': 0, 'max_containers': 1, 'retries': 0})
        synth = next(call for call in calls if isinstance(call.func, ast.Attribute) and call.func.attr == 'synthesize')
        settings = {kw.arg: ast.literal_eval(kw.value) for kw in synth.keywords if kw.arg not in ('voice_style', 'lang')}
        self.assertEqual(settings, {'speed': 1., 'total_steps': 8, 'max_chunk_length': 300, 'silence_duration': .3})
        image = next(call for call in calls if isinstance(call.func, ast.Attribute) and call.func.attr == 'from_id')
        self.assertEqual(ast.literal_eval(image.args[0]), 'im-40qUTZlx5ttgX1nwrQdvyw')
        cached = next(call for call in calls if isinstance(call.func, ast.Name) and call.func.id == 'snapshot_download')
        self.assertTrue(ast.literal_eval(next(kw.value for kw in cached.keywords if kw.arg == 'local_files_only')))
        model = next(call for call in calls if isinstance(call.func, ast.Name) and call.func.id == 'TTS')
        self.assertFalse(ast.literal_eval(next(kw.value for kw in model.keywords if kw.arg == 'auto_download')))
        self.assertIn('HF_HUB_OFFLINE="1"', WORKER.read_text())


if __name__ == '__main__':
    unittest.main()
