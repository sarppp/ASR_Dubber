import array
import math
import random
import unittest

from batch_dubbing.audio_edges import analyze_edges


class AudioEdgesTests(unittest.TestCase):
    def test_trims_only_edges_with_guards_and_preserves_input(self):
        samples = array.array("f", [0] * 500 + [.1] * 300 + [0] * 500)
        before = samples.tobytes()
        result = analyze_edges(samples, 1000)
        self.assertEqual((result["start_sample"], result["end_sample"]), (420, 920))
        self.assertEqual(result["retained_seconds"], .5)
        self.assertEqual(samples.tobytes(), before)

    def test_peak_activity_preserves_quiet_initial_consonant(self):
        samples = [0.0] * 200 + [10 ** (-55 / 20)] + [0.0] * 99 + [.2] * 200 + [0.0] * 400
        result = analyze_edges(samples, 1000)
        self.assertEqual(result["first_active_sample"], 200)
        self.assertEqual(result["start_sample"], 120)
        self.assertLessEqual(result["start_sample"], 200)

    def test_interior_silence_is_never_removed(self):
        samples = [0.0] * 500 + [.1] * 100 + [0.0] * 2000 + [.1] * 100 + [0.0] * 500
        result = analyze_edges(samples, 1000)
        self.assertEqual(result["retained_samples"], 2400)
        self.assertEqual(samples[result["start_sample"]:result["end_sample"]].count(0), 2200)

    def test_empty_silence_and_nonfinite_are_rejected(self):
        for samples in ([], [0.0] * 100, [1e-8] * 100,
                        [.1, math.nan], [.1, math.inf], [.1, -math.inf],
                        [[.1]], [True], "audio", b"audio", {0: .1}):
            with self.subTest(samples=repr(samples)[:30]), self.assertRaises(ValueError):
                analyze_edges(samples, 1000)

    def test_sample_rate_and_policy_bounds(self):
        for sample_rate in (0, -1, 1000.0, True, 384001):
            with self.subTest(sample_rate=sample_rate), self.assertRaises(ValueError):
                analyze_edges([.1], sample_rate)
        for key, value in (("max_edge_trim_seconds", 1.01), ("max_edge_trim_seconds", -1),
                           ("head_guard_seconds", math.inf), ("tail_guard_seconds", -1),
                           ("threshold_db", math.nan), ("threshold_db", -10),
                           ("min_removable_seconds", True), ("threshold_db", "-60")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                analyze_edges([.1], 1000, **{key: value})

    def test_partial_last_frame_is_included(self):
        samples = [0.0] * 200 + [.1] * 3
        result = analyze_edges(samples, 1000)
        self.assertEqual(result["last_active_end_sample"], 203)
        self.assertEqual(result["end_sample"], 203)
        self.assertEqual(result["start_sample"], 120)

    def test_guard_rounding_is_outward(self):
        result = analyze_edges([0] * 330 + [.1] * 220 + [0] * 400, 1001)
        self.assertEqual(result["frame_samples"], 11)
        self.assertEqual(result["start_sample"], 330 - 81)
        self.assertEqual(result["end_sample"], 550 + 121)

    def test_minimum_removal_is_inclusive_and_per_edge(self):
        exact = analyze_edges([0] * 180 + [.1] * 100 + [0] * 220, 1000)
        self.assertEqual(exact["start_sample"], 100)
        self.assertEqual(exact["trailing_removed_seconds"], .1)
        short = analyze_edges([0] * 170 + [.1] * 100 + [0] * 210, 1000)
        self.assertEqual(short["start_sample"], 0)
        self.assertEqual(short["end_sample"], 480)

    def test_maximum_trim_is_per_edge_and_never_exceeds_one_second(self):
        samples = [0] * 3000 + [.1] * 100 + [0] * 3000
        result = analyze_edges(samples, 1000)
        self.assertEqual((result["start_sample"], result["end_sample"]), (1000, 5100))
        self.assertEqual(set(result["flags"]), {"leading_trim_capped", "trailing_trim_capped"})
        lower = analyze_edges(samples, 1001, max_edge_trim_seconds=.333)
        self.assertEqual(lower["start_sample"], 333)
        disabled = analyze_edges(samples, 1000, max_edge_trim_seconds=0)
        self.assertEqual(disabled["removed_seconds"], 0)

    def test_wholly_active_clip_remains_intact_even_above_unity(self):
        result = analyze_edges([1.2] * 300, 1000)
        self.assertEqual((result["start_sample"], result["end_sample"]), (0, 300))
        self.assertEqual(result["peak"], 1.2)

    def test_randomized_invariants(self):
        rng = random.Random(20261028)
        for _ in range(400):
            sample_rate = rng.choice((1000, 1001, 8000))
            lead, length, tail = (rng.randrange(0, 2000), rng.randrange(1, 100), rng.randrange(0, 2000))
            samples = [0.0] * lead + [.01] * length + [0.0] * tail
            result = analyze_edges(samples, sample_rate)
            start, end = result["start_sample"], result["end_sample"]
            self.assertTrue(0 <= start <= lead < lead + length <= end <= len(samples))
            self.assertLessEqual(start / sample_rate, 1)
            self.assertLessEqual((len(samples) - end) / sample_rate, 1)
            self.assertEqual(samples[start:end].count(.01), length)
            self.assertAlmostEqual(result["raw_seconds"] - result["retained_seconds"], result["removed_seconds"])


if __name__ == "__main__":
    unittest.main()
