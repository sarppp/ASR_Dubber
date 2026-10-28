"""Adapter validation and failure receipts without importing any model package."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from batch_dubbing import cpu_natural
from batch_dubbing.test_natural_pipeline import audio, units


class FakeArray(list):
    ndim = 1

    def squeeze(self):
        return self

    def reshape(self, shape):
        return self


class CpuNaturalTests(unittest.TestCase):
    def data(self):
        return {"duration_seconds": 3, "source_language": "en", "speech_units": {"de": units()}}

    def test_range_clips_corridor_without_changing_word_times(self):
        data = self.data()
        before = copy.deepcopy(data)
        selected, scope = cpu_natural.select_units(data, 1.5, "de")
        self.assertEqual(data, before)
        self.assertEqual(selected[0]["corridor_end"], 1.5)
        self.assertEqual(selected[0]["source_end"], 1.3)
        self.assertEqual(selected[0]["words"], data["speech_units"]["de"][0]["words"])
        self.assertEqual(selected[0]["source_language"], "en")
        self.assertEqual(scope["excluded_boundary_units"], [])

    def test_crossing_utterance_is_explicitly_excluded(self):
        data = self.data()
        later = copy.deepcopy(data["speech_units"]["de"][0])
        later.update(id="later", source_start=1.6, source_end=2.5)
        data["speech_units"]["de"].append(later)
        selected, scope = cpu_natural.select_units(data, 2, "de")
        self.assertEqual(len(selected), 1)
        self.assertEqual(scope["excluded_boundary_units"][0]["id"], "later")
        self.assertEqual(scope["included_source_end"], 1.3)

    def test_bad_or_empty_range_fails_before_dependency_imports(self):
        for seconds in (0, -1, 7201, float("nan"), True, .5, 4):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                cpu_natural.synthesize_natural(self.data(), seconds, {})
        with self.assertRaisesRegex(ValueError, "source-linked"):
            cpu_natural.select_units({"translations": {"de": []}}, 3, "de")

    def test_renderer_failure_is_reported_and_native_audio_survives(self):
        fake_model = SimpleNamespace(sample_rate=44100, get_voice_style=lambda voice: voice,
            synthesize=lambda *args, **kwargs: (audio()["samples"], [1.2]))
        tts = Mock(return_value=fake_model)
        packages = {
            "numpy": SimpleNamespace(float32="float32", asarray=lambda x, **kwargs: FakeArray(x)),
            "huggingface_hub": SimpleNamespace(snapshot_download=Mock(return_value="mock-model")),
            "supertonic": SimpleNamespace(TTS=tts),
        }
        with tempfile.TemporaryDirectory() as temp, patch.dict("sys.modules", packages), \
                patch.object(cpu_natural, "ARTIFACT_ROOT", Path(temp)), \
                patch.object(cpu_natural, "render_timeline", side_effect=RuntimeError("sample overflow")):
            result = cpu_natural.synthesize_natural(self.data(), 3, {"voice_a": {"voice": "M1"}})
            self.assertIsNone(result["audio"])
            self.assertEqual(result["report"]["status"], "failed")
            self.assertEqual(result["report"]["failure_stage"], "render")
            directory = Path(result["report"]["native_artifact_directory"])
            saved = json.loads((directory / "report.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["error"], "sample overflow")
            self.assertTrue(list(directory.glob("raw*.wav")))
            self.assertEqual(tts.call_args.kwargs["intra_op_num_threads"], 2)


if __name__ == "__main__":
    unittest.main()
