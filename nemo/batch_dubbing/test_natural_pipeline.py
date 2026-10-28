"""General source-to-timeline regressions with synthetic speech, never models."""
import array
import copy
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock
import wave

from batch_dubbing.natural_pipeline import run_pipeline, validate_units
from batch_dubbing.natural_render import render_timeline
from batch_dubbing.speech_units import build_speech_units, bind_translations


def units(text="A sentence.", translation="Ein Satz.", end=1.3):
    source = build_speech_units(
        [{"word": text, "start": .3, "end": end, "speaker": "voice_a"}],
        [{"start": 0, "end": 3, "speaker": "voice_a"}], source_id="synthetic-recording")
    return [u.to_dict() for u in bind_translations(source, {source[0].id: translation})]


def audio(seconds=.6, padding=.3):
    rate = 44100
    samples = array.array("f", [0.]) * round(padding * rate)
    samples.extend(.2 * math.sin(2 * math.pi * 220 * i / rate) for i in range(round(seconds * rate)))
    samples.extend(array.array("f", [0.]) * round(padding * rate))
    return {"samples": samples, "sample_rate": rate, "voice": "test", "predicted_duration": len(samples) / rate}


class NaturalPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "attempts"

    def run_job(self, source=None, **kwargs):
        return run_pipeline(source or units(), duration_seconds=3, output_dir=self.path,
                            **{"synthesize": lambda unit, text: audio(), "max_synthesis_calls": 1, **kwargs})

    def saved(self):
        return json.loads((self.path / "report.json").read_text())

    def test_padding_is_measured_and_native_speech_does_not_fill_caption_slot(self):
        source = units()
        original = copy.deepcopy(source)
        report = self.run_job(source)
        self.assertEqual(source, original)
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["synthesis_calls"], 1)
        self.assertEqual(report["render"]["rows"][0]["tempo"], 1)
        take = report["attempts"][0]
        self.assertAlmostEqual(take["edges"]["retained_seconds"], .8, places=2)
        self.assertGreater(take["edges"]["removed_seconds"], .35)
        self.assertTrue(Path(take["file"]).exists())
        self.assertEqual(self.saved()["status"], "ready")

    def test_long_translation_blocks_whole_output_without_hidden_retries(self):
        synth = Mock(return_value=audio(2))
        revise = Mock(return_value="Short.")
        report = self.run_job(synthesize=synth, revise=revise)
        self.assertEqual(report["status"], "needs_revision")
        self.assertEqual(report["render"]["rows"], [])
        self.assertEqual(len(report["revision_requests"]), 1)
        synth.assert_called_once()
        revise.assert_not_called()
        self.assertTrue(Path(report["attempts"][0]["file"]).exists())

    def test_one_budgeted_revision_keeps_identity_and_measures_new_audio(self):
        synth = Mock(side_effect=[audio(2), audio(.7)])
        revise = Mock(return_value="Kurz.")
        report = self.run_job(synthesize=synth, revise=revise, max_synthesis_calls=2,
                              max_revisions_per_unit=1)
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["synthesis_calls"], 2)
        self.assertEqual(report["revision_calls"], 1)
        self.assertEqual(report["selected_attempts"], {units()[0]["id"]: 1})
        self.assertEqual([take["text"] for take in report["attempts"]], ["Ein Satz.", "Kurz."])
        self.assertEqual(synth.call_args_list[0].args[0], synth.call_args_list[1].args[0])

    def test_lost_protected_fact_does_not_trigger_resynthesis(self):
        source = units(translation="Es waren 42 Stück.")
        source[0]["protected_terms"] = ["42"]
        synth = Mock(return_value=audio(2))
        report = self.run_job(source, synthesize=synth, revise=lambda request: "Es waren einige.",
                              max_synthesis_calls=2, max_revisions_per_unit=1)
        self.assertEqual(report["status"], "needs_revision")
        self.assertEqual(report["revision_failures"][0]["reason"], "Protected term lost")
        synth.assert_called_once()

    def test_call_budget_prevents_even_revision_provider_call(self):
        revise = Mock()
        report = self.run_job(synthesize=lambda unit, text: audio(2), revise=revise,
                              max_synthesis_calls=1, max_revisions_per_unit=1)
        self.assertEqual(report["status"], "needs_revision")
        revise.assert_not_called()

    def test_identical_or_longer_revision_never_loops(self):
        for candidate in ("Ein Satz.", "Eine andere Formulierung."):
            with self.subTest(candidate=candidate):
                self.path = Path(self.temp.name) / str(len(list(Path(self.temp.name).iterdir())))
                synth = Mock(side_effect=[audio(2), audio(2.2)])
                report = self.run_job(synthesize=synth, revise=lambda request: candidate,
                                      max_synthesis_calls=2, max_revisions_per_unit=1)
                self.assertEqual(report["status"], "needs_revision")
                self.assertEqual(report["selected_attempts"], {units()[0]["id"]: 0})
                self.assertLessEqual(synth.call_count, 2)

    def test_failure_persists_raw_attempt_and_never_provides_partial_schedule(self):
        def fail(request):
            raise RuntimeError("provider failed")
        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            self.run_job(synthesize=lambda unit, text: audio(2), revise=fail,
                         max_synthesis_calls=2, max_revisions_per_unit=1)
        report = self.saved()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["render"]["rows"], [])
        self.assertTrue(Path(report["attempts"][0]["file"]).exists())

    def test_model_callback_cannot_mutate_source_evidence(self):
        def mutate(unit, text):
            unit["words"][0]["start"] = 99
            return audio()
        report = self.run_job(synthesize=mutate)
        self.assertEqual(report["units"][0]["words"][0]["start"], .3)

    def test_individually_feasible_overlapping_turns_never_publish_partial_audio(self):
        source = units()
        second = copy.deepcopy(source[0])
        second.update(id="second", speaker="voice_b", source_start=1.3, source_end=2.3,
                      source_word_indices=[1])
        second["words"][0].update(index=1, speaker="voice_b", start=1.3, end=2.3)
        source.append(second)
        report = self.run_job(source, synthesize=lambda unit, text: audio(1.1), max_synthesis_calls=2)
        self.assertTrue(report["cross_turn_conflict"])
        self.assertEqual(report["status"], "needs_revision")
        self.assertEqual(report["render"]["rows"], [])

    def test_policy_and_word_provenance_rejected_before_model_call(self):
        cases = []
        for key, value in (("timing_basis", "display_estimate"), ("source_start", .2),
                           ("source_word_indices", [9]), ("words", []), ("protected_terms", [42])):
            source = units()
            source[0][key] = value
            cases.append(source)
        for source in cases:
            synth = Mock()
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.run_job(source, synthesize=synth)
            synth.assert_not_called()
        for options in ({"max_synthesis_calls": None}, {"max_synthesis_calls": 3},
                        {"max_revisions_per_unit": 1}, {"max_rate": 1.5}, {"max_shift": 1}):
            synth = Mock()
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_job(synthesize=synth, **options)
            synth.assert_not_called()

    def test_rules_do_not_depend_on_language_or_video_specific_text(self):
        for index, (source, translation) in enumerate((("Hello.", "Hallo."), ("Hallo.", "Hello."),
                                                       ("Bonjour.", "今日は。"))):
            self.path = Path(self.temp.name) / f"language-{index}"
            report = self.run_job(units(source, translation))
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["attempts"][0]["text"], translation)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg unavailable")
    def test_actual_synthetic_source_to_timeline_preserves_native_crop(self):
        report = self.run_job()
        output = Path(self.temp.name) / "timeline.wav"
        rendered = render_timeline(report["render"], output, max_wall_seconds=20)
        self.assertEqual(rendered["rows"][0]["output_samples"], report["attempts"][0]["edges"]["retained_samples"])
        with wave.open(str(output)) as wav:
            self.assertEqual(wav.getnframes(), 3 * 44100)
        self.assertEqual(len(list(self.path.glob("raw*.wav"))), 1)


if __name__ == "__main__":
    unittest.main()
