import copy
import math
import random
import unittest

from batch_dubbing.natural_timing import group_speech_cues, plan_timing


def cue(start=0, end=2, text="A sentence.", speaker="a"):
    return dict(start=start, end=end, text=text, speaker=speaker)


class GroupingTests(unittest.TestCase):
    def test_fragment_joins_but_sentence_closes_phrase(self):
        source = [cue(0, 1, "This is"), cue(1, 2, "complete.”"), cue(2, 3, "Next.")]
        before = copy.deepcopy(source)
        phrases = group_speech_cues(source)
        self.assertEqual([p["text"] for p in phrases], ["This is complete.”", "Next."])
        self.assertEqual(phrases[0]["source_indices"], [0, 1])
        self.assertEqual(phrases[0]["source_cues"], source[:2])
        self.assertEqual(source, before)

    def test_pause_and_speaker_are_boundaries(self):
        source = [cue(0, 1, "fragment"), cue(1.3, 2, "continuation"), cue(2, 3, "reply", "b")]
        self.assertEqual(len(group_speech_cues(source)), 3)

    def test_caps_never_drop_or_split_text(self):
        source = [cue(0, 2, "first"), cue(2, 4, "second"), cue(4, 20, "x" * 300)]
        phrases = group_speech_cues(source, max_seconds=3, max_characters=10)
        self.assertEqual([p["text"] for p in phrases], [s["text"] for s in source])
        self.assertIn("oversized_cue", phrases[-1]["flags"])
        self.assertIn("incomplete_phrase", phrases[0]["flags"])

    def test_empty(self):
        self.assertEqual(group_speech_cues([]), [])

    def test_invalid_inputs(self):
        for bad in [cue(end=0), cue(start=-1), cue(end=math.inf), cue(start=math.nan),
                    cue(text=" "), cue(speaker=None), cue(start=True)]:
            with self.subTest(cue=bad), self.assertRaises(ValueError):
                group_speech_cues([bad])
        with self.assertRaises(ValueError):
            group_speech_cues([cue(0, 2), cue(1.9999, 3)])
        with self.assertRaises(ValueError):
            group_speech_cues([cue()], max_characters=0)

    def test_random_inputs_preserve_every_cue_in_order(self):
        rng = random.Random(709)
        source, time = [], 0.0
        for i in range(250):
            duration = rng.uniform(.1, 8)
            source.append(cue(time, time + duration, f"word{i}" + ("." if i % 4 == 0 else ""), str(i // 8)))
            time += duration + rng.uniform(0, .4)
        phrases = group_speech_cues(source)
        self.assertEqual([i for p in phrases for i in p["source_indices"]], list(range(len(source))))
        self.assertEqual(" ".join(p["text"] for p in phrases), " ".join(s["text"] for s in source))
        self.assertTrue(all(a["end"] <= b["start"] for a, b in zip(phrases, phrases[1:])))


class TimingTests(unittest.TestCase):
    def test_short_speech_is_not_slowed(self):
        decision = plan_timing(cue(end=10), 3)
        self.assertEqual(decision["rate"], 1)
        self.assertEqual(decision["output_end"], 3)
        self.assertEqual(decision["status"], "review_underfilled")

    def test_native_speed_borrows_only_needed_gap(self):
        decision = plan_timing(cue(), 2.05, next_phrase=cue(2.3, 4))
        self.assertEqual(decision["rate"], 1)
        self.assertAlmostEqual(decision["borrowed_seconds"], .05)

    def test_same_speaker_borrow_retains_pause(self):
        decision = plan_timing(cue(), 2.35, next_phrase=cue(2.2, 4))
        self.assertLessEqual(decision["rate"], 1.15)
        self.assertAlmostEqual(decision["output_end"], 2.1)
        self.assertAlmostEqual(decision["borrowed_seconds"], .1)

    def test_no_borrow_across_speaker_or_long_pause_or_no_successor(self):
        for following in [cue(2.3, 4, speaker="b"), cue(2.6, 4), None]:
            with self.subTest(next=following):
                decision = plan_timing(cue(), 2.4, next_phrase=following)
                self.assertEqual(decision["status"], "needs_rewrite")
                self.assertIsNone(decision["rate"])
                self.assertIsNone(decision["output_end"])

    def test_exact_rate_cap_fits_without_truncation(self):
        decision = plan_timing(cue(), 2.3)
        self.assertEqual(decision["status"], "ready")
        self.assertAlmostEqual(decision["rate"], 1.15)
        self.assertAlmostEqual(decision["output_end"], 2)

    def test_overflow_is_explicit(self):
        decision = plan_timing(cue(), 4)
        self.assertEqual(decision["status"], "needs_rewrite")
        self.assertGreater(decision["overflow_at_max_rate"], 1)

    def test_input_and_policy_validation(self):
        for raw in [0, -1, math.nan, math.inf, True]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                plan_timing(cue(), raw)
        for kwargs in [dict(max_rate=.9), dict(max_borrow=.2), dict(minimum_pause=.05),
                       dict(long_pause=1), dict(underfill_ratio=0), dict(video_duration=1),
                       dict(next_phrase=cue(1.9, 4)), dict(video_duration=3, next_phrase=cue(2.1, 4))]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                plan_timing(cue(), 2, **kwargs)

    def test_random_timing_decisions_preserve_invariants(self):
        rng = random.Random(34)
        for _ in range(1000):
            start = rng.uniform(0, 100)
            duration = rng.uniform(.1, 20)
            end = start + duration
            following = cue(end + rng.uniform(0, 1), end + 2)
            decision = plan_timing(cue(start, end), rng.uniform(.01, 30), next_phrase=following)
            if decision["status"] == "needs_rewrite":
                self.assertIsNone(decision["output_end"])
                self.assertGreater(decision["overflow_at_max_rate"], 0)
            else:
                self.assertGreaterEqual(decision["rate"], 1)
                self.assertLessEqual(decision["rate"], 1.15)
                self.assertLessEqual(decision["output_end"], decision["deadline"] + 1e-9)
                self.assertLessEqual(decision["borrowed_seconds"], .15 + 1e-9)
                self.assertLessEqual(decision["output_end"], following["start"] + 1e-9)
                self.assertAlmostEqual((decision["output_end"] - start) * decision["rate"], decision["raw_duration"])


if __name__ == "__main__":
    unittest.main()
