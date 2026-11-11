"""End-to-end contract/measurement checks with fake providers only."""
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from batch_dubbing.isochrony import ActivityTimeline, SpeechInterval
from batch_dubbing.isochrony_backend import BackendCapabilities, SynthesisRequest, plan_candidates
from batch_dubbing.isochrony_pipeline import CandidateObservation, run_comparison

DIGEST = "a" * 64


def timeline(spans=((1, 2), (3, 4)), provenance="synthetic", evidence_id="source-fixture"):
    return ActivityTimeline(5, tuple(SpeechInterval(*span) for span in spans), provenance, .01, evidence_id)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "comparison"
        self.source = timeline()
        self.backend = BackendCapabilities("fixture", "fixture-v1", ("en", "de"), True, True, "unit-test")

    def candidates(self, source=None, texts=("Ein Satz. Noch einer.",), seeds=(0,)):
        source = source or self.source
        return tuple(spec for text in texts for spec in plan_candidates(self.backend,
            SynthesisRequest("source-1", text, "de", 5, source), seeds=seeds, max_candidates=len(seeds)))

    def observation(self, spec, spans=((1, 2), (3, 4)), provenance="synthetic", **kwargs):
        return CandidateObservation(spec.candidate_id, timeline(spans, provenance, "sha256:" + DIGEST),
                                    "fixture://retained-raw-audio", DIGEST, **kwargs)

    def run_job(self, **kwargs):
        return run_comparison(**{"source": self.source, "capabilities": self.backend,
            "source_text": "A sentence. Another one.", "source_language": "en",
            "candidates": self.candidates(), "synthesize": self.observation,
            "output_dir": self.path, "max_calls": 1, **kwargs})

    def saved(self):
        return json.loads((self.path / "report.json").read_text())

    def test_short_long_and_exact_candidates_are_retained_without_modification(self):
        specs = self.candidates(texts=("Kurz.", "Eine längere Formulierung.", "Passende Formulierung."))
        measurements = [((1, 1.5), (3, 3.5)), ((1, 2.5), (3, 4.5)), ((1, 2), (3, 4))]
        calls = []
        def synth(spec):
            calls.append(spec)
            return self.observation(spec, measurements[len(calls) - 1])
        result = self.run_job(candidates=specs, synthesize=synth, max_calls=3)
        self.assertEqual(tuple(calls), specs)
        self.assertEqual(result["calls"], 3)
        self.assertEqual(len(result["attempts"]), 3)
        self.assertEqual(result["timing_candidate_ids"], [])
        self.assertEqual(result["synthetic_match_ids"], [specs[2].candidate_id])
        self.assertEqual([row["wording_feedback"]["action"] for row in result["attempts"]],
                         ["longer_equivalent_wording", "shorter_equivalent_wording", "none"])
        self.assertEqual(result["publication"]["status"], "blocked")
        self.assertEqual(result, self.saved())

    def test_acoustic_timing_match_keeps_content_and_listening_review_pending(self):
        source = timeline(provenance="manual_review")
        result = self.run_job(source=source, candidates=self.candidates(source),
                              synthesize=lambda spec: self.observation(spec, provenance="acoustic_vad"))
        self.assertEqual(result["status"], "timing_candidates_available")
        self.assertEqual(len(result["timing_candidate_ids"]), 1)
        self.assertEqual(result["publication"]["content_review"], "pending")
        self.assertEqual(result["publication"]["listening_review"], "pending")
        self.assertEqual(result["publication"]["status"], "blocked")

    def test_rate_violation_cannot_be_selected_from_exact_boundary_match(self):
        source = timeline(provenance="acoustic_vad")
        result = self.run_job(source=source, candidates=self.candidates(source),
            synthesize=lambda spec: self.observation(spec, provenance="acoustic_vad", playback_rate=1.15))
        self.assertEqual(result["status"], "needs_revision")
        self.assertEqual(result["timing_candidate_ids"], [])
        self.assertTrue(result["attempts"][0]["comparison"]["exact_measured_match"])
        feedback = result["attempts"][0]["wording_feedback"]
        self.assertEqual(feedback["status"], "synthesis_processing_repair_required")
        self.assertFalse(feedback["revision_requested"])
        self.assertEqual(feedback["comparison"]["processing"]["playback_rate"], 1.15)

    def test_failure_counts_attempt_stops_and_retains_previous_observation(self):
        specs = self.candidates(seeds=(0, 1, 2))
        synth = Mock(side_effect=[self.observation(specs[0]), RuntimeError("provider unavailable")])
        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            self.run_job(candidates=specs, max_calls=3, synthesize=synth)
        self.assertEqual(synth.call_count, 2)
        result = self.saved()
        self.assertEqual(result["calls"], 2)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["attempts"][0]["observation"]["audio_sha256"], DIGEST)
        self.assertEqual(result["attempts"][1]["status"], "failed")
        self.assertEqual(result["publication"]["status"], "blocked")

    def test_over_budget_or_invalid_late_candidate_fails_before_first_callback(self):
        specs = self.candidates(seeds=(0, 1))
        invalid = replace(specs[1], candidate_id="forged")
        for candidates, budget in ((specs, 1), ((specs[0], invalid), 2),
                                   ((specs[0], specs[0]), 2)):
            synth = Mock()
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                self.run_job(candidates=candidates, max_calls=budget, synthesize=synth)
            synth.assert_not_called()
            self.assertFalse(self.path.exists())

    def test_changed_target_or_unit_identity_rejected_before_call(self):
        other = self.candidates(timeline(((.5, 2), (3, 4))))
        original = self.candidates()[0]
        changed_identity = plan_candidates(self.backend, replace(original.request, source_id="different"),
                                          seeds=(1,), max_candidates=1)[0]
        for specs in (other, (original, changed_identity)):
            synth = Mock()
            with self.subTest(specs=specs), self.assertRaises(ValueError):
                self.run_job(candidates=specs, max_calls=2, synthesize=synth)
            synth.assert_not_called()

    def test_no_overwrite(self):
        result = self.run_job()
        synth = Mock()
        with self.assertRaises(FileExistsError):
            self.run_job(synthesize=synth)
        self.assertEqual(result, self.saved())
        synth.assert_not_called()

    def test_missing_speech_or_nonacoustic_source_blocks_provider(self):
        for source in (timeline(()), timeline(provenance="subtitle"), timeline(provenance="asr_words")):
            synth = Mock()
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.run_job(source=source, synthesize=synth)
            synth.assert_not_called()

    def test_candidate_measurement_with_weak_evidence_is_not_selected(self):
        result = self.run_job(synthesize=lambda spec: self.observation(spec, provenance="energy_threshold"))
        self.assertEqual(result["status"], "needs_revision")
        self.assertEqual(result["timing_candidate_ids"], [])
        self.assertEqual(result["attempts"][0]["comparison"]["status"], "insufficient_evidence")

    def test_foreign_candidate_output_and_wrong_hash_fail(self):
        spec = self.candidates()[0]
        observation = replace(self.observation(spec), candidate_id="another-candidate")
        with self.assertRaisesRegex(ValueError, "another candidate"):
            self.run_job(synthesize=lambda spec: observation)
        self.assertEqual(self.saved()["status"], "failed")
        self.assertEqual(self.saved()["calls"], 1)
        with self.assertRaisesRegex(ValueError, "digest"):
            replace(self.observation(spec), audio_sha256="b" * 64)

    def test_duration_only_experiment_is_explicit_and_still_evaluates_pauses(self):
        backend = replace(self.backend, internal_pause_conditioning=False)
        request = replace(self.candidates()[0].request, target_activity=None)
        specs = plan_candidates(backend, request, seeds=(0,), max_candidates=1)
        result = self.run_job(capabilities=backend, candidates=specs,
            synthesize=lambda spec: self.observation(spec, ((1, 2.5), (3.5, 4))))
        self.assertEqual(result["status"], "needs_revision")
        self.assertEqual(result["attempts"][0]["comparison"]["metrics"]["speech_duration_error_seconds"], 0)
        self.assertEqual(result["attempts"][0]["comparison"]["metrics"]["timeline_duration_error_seconds"], 0)

    def test_numeric_rate_is_normalized_before_evidence_is_persisted(self):
        result = self.run_job(synthesize=lambda spec: self.observation(spec, playback_rate=Fraction(1, 1)))
        self.assertEqual(result["status"], "synthetic_comparison_complete")
        self.assertEqual(result["attempts"][0]["observation"]["audio_sha256"], DIGEST)
        self.assertIs(type(result["attempts"][0]["observation"]["playback_rate"]), float)
        self.assertEqual(result, self.saved())

    def test_report_expansion_is_bounded_for_source_and_observations(self):
        spans = tuple(SpeechInterval(i / 100, i / 100 + .005) for i in range(257))
        source = ActivityTimeline(5, spans, "synthetic", .001, "large-fixture")
        synth = Mock()
        with self.assertRaisesRegex(ValueError, "interval budget"):
            self.run_job(source=source, synthesize=synth)
        synth.assert_not_called()
        spec = self.candidates()[0]
        with self.assertRaisesRegex(ValueError, "interval budget"):
            CandidateObservation(spec.candidate_id, replace(source, evidence_id="sha256:" + DIGEST),
                                 "fixture://large", DIGEST)

    def test_maximum_comparison_finishes_under_the_bounded_test_process(self):
        spans = tuple(SpeechInterval(i / 100, i / 100 + .005) for i in range(256))
        source = ActivityTimeline(5, spans, "synthetic", .001, "maximum-fixture")
        specs = self.candidates(source, seeds=tuple(range(8)))
        def synth(spec):
            return CandidateObservation(spec.candidate_id, replace(source, evidence_id="sha256:" + DIGEST),
                                        "fixture://retained", DIGEST)
        result = self.run_job(source=source, candidates=specs, max_calls=8, synthesize=synth)
        self.assertEqual(result["calls"], 8)
        self.assertEqual(len(result["synthetic_match_ids"]), 8)
        self.assertLess((self.path / "report.json").stat().st_size, 16 * 1024**2)


if __name__ == "__main__":
    unittest.main()
