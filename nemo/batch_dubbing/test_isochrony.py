"""Strict timing regressions: synthetic intervals, no media or model loading."""
from dataclasses import FrozenInstanceError
import json
import math
import random
import unittest

from batch_dubbing.isochrony import (
    ActivityTimeline, SpeechInterval, compare_timelines, timeline_from_mask,
)


def timeline(spans=(), duration=4, provenance="synthetic", resolution=.01):
    return ActivityTimeline(duration, tuple(SpeechInterval(*span) for span in spans),
                            provenance, resolution, "test-measurement")


class IsochronyTests(unittest.TestCase):
    def test_same_duration_and_speech_total_with_wrong_pause_positions_fails(self):
        source = timeline(((.5, 1.5), (2.5, 3.5)))
        candidate = timeline(((.5, 2), (3, 3.5)))
        result = compare_timelines(source, candidate)
        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(result["metrics"]["speech_duration_error_seconds"], 0)
        self.assertEqual(result["metrics"]["timeline_duration_error_seconds"], 0)
        self.assertAlmostEqual(result["metrics"]["missing_speech_seconds"], .5)
        self.assertAlmostEqual(result["metrics"]["extra_speech_seconds"], .5)
        self.assertLess(result["metrics"]["pause_iou"], 1)

    def test_short_speech_plus_padding_does_not_pass(self):
        result = compare_timelines(timeline(((1, 3),)), timeline(((1, 2),)))
        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(result["metrics"]["missing_speech_seconds"], 1)
        self.assertEqual(result["pause_errors"][-1]["duration_error_seconds"], 1)

    def test_long_speech_consuming_pause_does_not_pass(self):
        result = compare_timelines(timeline(((1, 2), (3, 3.5))), timeline(((1, 2.5), (3, 3.5))))
        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(result["pause_errors"][1]["duration_error_seconds"], -.5)

    def test_missing_or_extra_block_does_not_invent_correspondence(self):
        one, two = timeline(((1, 3),)), timeline(((1, 2), (2.5, 3)))
        for source, candidate in ((one, two), (two, one)):
            result = compare_timelines(source, candidate)
            self.assertIn("speech_block_count_mismatch", result["reasons"])
            self.assertIsNone(result["speech_errors"])
            self.assertIsNone(result["pause_errors"])

    def test_matching_synthetic_fixture_never_claims_acoustic_evidence(self):
        source = timeline(((1, 3),))
        report = compare_timelines(source, source)
        self.assertEqual(report["status"], "synthetic_fixture")
        self.assertFalse(report["evidence_sufficient_for_speech_comparison"])
        self.assertTrue(report["exact_measured_match"])
        json.dumps(report, allow_nan=False)

    def test_matching_acoustic_evidence_is_only_a_measurement_verdict(self):
        source = timeline(((1, 3),), provenance="manual_review")
        candidate = timeline(((1, 3),), provenance="acoustic_vad")
        report = compare_timelines(source, candidate)
        self.assertEqual(report["status"], "matches_measurements")
        self.assertIn("do not verify translation", report["quality_limit"])
        self.assertFalse(report["processing"]["declarations_independently_verified"])

    def test_nonacoustic_sources_never_become_verified_by_matching(self):
        for provenance in ("asr_words", "subtitle", "energy_threshold"):
            with self.subTest(provenance=provenance):
                source = timeline(((1, 3),), provenance=provenance)
                result = compare_timelines(source, timeline(((1, 3),), provenance="acoustic_vad"))
                self.assertEqual(result["status"], "insufficient_evidence")

    def test_speed_and_clipping_are_rejected_even_if_timing_matches(self):
        source = timeline(((1, 3),))
        for kwargs in ({"playback_rate": .65}, {"playback_rate": 1.15},
                       {"playback_rate": math.nextafter(1, 2)},
                       {"waveform_time_stretched": True}, {"speech_trimmed": True}):
            with self.subTest(kwargs=kwargs):
                result = compare_timelines(source, source, **kwargs)
                self.assertEqual(result["status"], "mismatch")
                self.assertTrue(result["exact_measured_match"])
                self.assertFalse(result["timing_matches_policy"])

    def test_resolution_is_not_an_implicit_tolerance(self):
        source = timeline(((1, 3),), resolution=.16)
        candidate = timeline(((1.01, 3),), resolution=.16)
        report = compare_timelines(source, candidate)
        self.assertEqual(report["status"], "mismatch")
        self.assertEqual(report["tolerance_seconds"], 0)
        self.assertEqual(report["measurement_resolution_seconds"]["source"], .16)

    def test_explicit_tolerance_never_claims_exact_matching(self):
        result = compare_timelines(timeline(((1, 3),)), timeline(((1.01, 3),)), tolerance_seconds=.02)
        self.assertTrue(result["timing_matches_policy"])
        self.assertFalse(result["exact_measured_match"])

    def test_leading_trailing_silence_and_file_duration_are_measured(self):
        result = compare_timelines(timeline(((1, 3),)), timeline(((1, 3),), duration=5))
        self.assertIn("timeline_duration_mismatch", result["reasons"])
        self.assertEqual(result["pause_errors"][-1]["duration_error_seconds"], 1)
        self.assertEqual(result["metrics"]["speech_iou"], 1)
        self.assertLess(result["metrics"]["pause_iou"], 1)

    def test_silence_only_and_speech_only_are_well_defined(self):
        for source in (timeline(), timeline(((0, 4),))):
            result = compare_timelines(source, source)
            self.assertTrue(result["timing_matches_policy"])
            self.assertEqual(result["metrics"]["speech_iou"], 1)
            self.assertEqual(result["metrics"]["pause_iou"], 1)
        self.assertEqual(compare_timelines(timeline(), timeline(((0, 4),)))["status"], "mismatch")

    def test_intervals_are_immutable_and_adjacent_activity_is_canonical(self):
        intervals = [SpeechInterval(0, 1), SpeechInterval(1, 2)]
        source = ActivityTimeline(4, intervals, "synthetic", .01, "fixture")
        intervals.clear()
        self.assertEqual(source.speech, (SpeechInterval(0, 2),))
        with self.assertRaises(FrozenInstanceError):
            source.speech = ()

    def test_invalid_timestamps_and_types_rejected(self):
        for value in (True, "1", None, float("nan"), float("inf"), -1, 10**400):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SpeechInterval(value, 3)
        for spans in (((2, 3), (1, 2)), ((1, 3), (2, 4)), ((1, 5),), ((1, 1),)):
            with self.subTest(spans=spans), self.assertRaises(ValueError):
                timeline(spans)
        for kwargs in ({"duration": 0}, {"resolution": 0}, {"resolution": 5}, {"provenance": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                timeline(**kwargs)
        with self.assertRaises(ValueError):
            ActivityTimeline(4, (), "synthetic", .01, " " * 2000 + "x")

    def test_malformed_policy_fails(self):
        source = timeline(((1, 3),))
        for kwargs in ({"playback_rate": True}, {"tolerance_seconds": float("nan")},
                       {"tolerance_seconds": -1}, {"speech_trimmed": 1},
                       {"waveform_time_stretched": "false"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                compare_timelines(source, source, **kwargs)

    def test_partial_final_frame_and_silence_are_preserved(self):
        source = timeline_from_mask([False, True, True, False, True], duration_seconds=.45,
                                    frame_seconds=.1, provenance="synthetic", evidence_id="mask")
        self.assertEqual(len(source.speech), 2)
        self.assertAlmostEqual(source.speech[0].start_seconds, .1)
        self.assertAlmostEqual(source.speech[0].end_seconds, .3)
        self.assertEqual(source.speech[1], SpeechInterval(.4, .45))
        for mask in ([False, False, False], [True, True, True], [True, False, False]):
            result = timeline_from_mask(mask, duration_seconds=.3, frame_seconds=.1,
                                        provenance="synthetic", evidence_id="exact-grid")
            self.assertEqual(result.duration_seconds, .3)
        self.assertEqual(result.speech, (SpeechInterval(0, .1),))

    def test_missing_extra_or_nonboolean_frames_rejected(self):
        for mask in ([], [True], [True] * 4, [1, 0, 1], "111"):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                timeline_from_mask(mask, duration_seconds=.3, frame_seconds=.1,
                                   provenance="synthetic", evidence_id="invalid")

    def test_random_masks_agree_with_independent_discrete_oracle(self):
        rng = random.Random(20261029)
        # 1000 independent timelines include split/merged speech and both empty cases.
        for case in range(1000):
            size = rng.randrange(1, 100)
            source_mask = [bool(rng.getrandbits(1)) for _ in range(size)]
            target_mask = source_mask[:] if case % 10 == 0 else [bool(rng.getrandbits(1)) for _ in range(size)]
            def make(mask):
                return timeline_from_mask(mask, duration_seconds=size / 100,
                                          frame_seconds=.01, provenance="synthetic", evidence_id=str(case))
            report = compare_timelines(make(source_mask), make(target_mask))
            missing = sum(a and not b for a, b in zip(source_mask, target_mask)) / 100
            extra = sum(b and not a for a, b in zip(source_mask, target_mask)) / 100
            self.assertAlmostEqual(report["metrics"]["missing_speech_seconds"], missing)
            self.assertAlmostEqual(report["metrics"]["extra_speech_seconds"], extra)
            self.assertEqual(report["exact_measured_match"], source_mask == target_mask)
            reverse = compare_timelines(make(target_mask), make(source_mask))
            self.assertAlmostEqual(report["metrics"]["speech_iou"], reverse["metrics"]["speech_iou"])
            self.assertAlmostEqual(report["metrics"]["pause_iou"], reverse["metrics"]["pause_iou"])


if __name__ == "__main__":
    unittest.main()
