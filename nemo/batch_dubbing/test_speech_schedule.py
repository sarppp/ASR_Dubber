import copy
import math
import random
import unittest

from batch_dubbing.speech_schedule import schedule_turn


def clip(ident="a", start=1, end=2, duration=1, **extra):
    return {"id": ident, "speaker": "speaker_0", "source_start": start,
            "source_end": end, "retained_duration": duration,
            "alignment_verified": True, **extra}


class ScheduleTests(unittest.TestCase):
    def test_native_speed_does_not_fill_short_speech(self):
        result = schedule_turn([clip(end=4, duration=1)], corridor_start=0, corridor_end=5)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["rate"], 1)
        self.assertEqual(result["schedule"][0]["audible_end"], 2)

    def test_guard_aligns_audible_onset_without_double_counting_pause(self):
        rows = [clip("a", 1, 2, 1.2, head_guard=.1, tail_guard=.1),
                clip("b", 2.1, 3.1, 1.2, head_guard=.1, tail_guard=.1)]
        result = schedule_turn(rows, corridor_start=0, corridor_end=4)
        self.assertEqual(result["rate"], 1)
        a, b = result["schedule"]
        self.assertAlmostEqual(a["audible_start"], 1)
        self.assertAlmostEqual(a["start"], .9)
        self.assertGreaterEqual(b["start"] + 1e-8, a["end"])
        self.assertAlmostEqual(b["audible_start"] - a["audible_end"], .2)

    def test_global_collision_blocks_independently_fitting_clips(self):
        rows = [clip("a", 0, 1, 1.2), clip("b", 1, 2, 1.2)]
        for row in rows:
            self.assertEqual(schedule_turn([row], corridor_start=0, corridor_end=2.25,
                                           max_rate=1)["status"], "ready")
        result = schedule_turn(rows, corridor_start=0, corridor_end=2.25, max_rate=1)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["schedule"], [])
        self.assertIsNone(result["rate"])
        self.assertEqual(result["diagnostics"]["conflicts"][0]["clip_ids"], ["a", "b"])
        self.assertAlmostEqual(result["diagnostics"]["conflicts"][0]["deficit_seconds"], .15)

    def test_common_tempo_fits_entire_turn(self):
        rows = [clip("a", 0, 1, 1.1), clip("b", 1, 2, 1.1)]
        result = schedule_turn(rows, corridor_start=0, corridor_end=2)
        self.assertEqual(result["status"], "ready")
        self.assertAlmostEqual(result["rate"], 1.1, places=7)
        self.assertEqual({r["rate"] for r in result["schedule"]}, {result["rate"]})
        self.assertLessEqual(result["schedule"][-1]["end"], 2 + 1e-8)

    def test_actual_bill_question_source_alignment_changes_feasibility(self):
        raw = 3.0650340136054424
        caption = clip(start=328.76, end=329.96, duration=raw)
        source = clip(start=327.64, end=330.12, duration=raw)
        options = dict(corridor_start=324.545, corridor_end=330.22)
        self.assertEqual(schedule_turn([caption], **options)["status"], "blocked")
        result = schedule_turn([source], **options)
        self.assertEqual(result["status"], "ready")
        self.assertLessEqual(result["rate"], 1.15)
        self.assertGreaterEqual(result["schedule"][0]["audible_start"], 327.39 - 1e-8)

    def test_point_eight_eight_caption_is_not_the_source_window(self):
        # Source mapping gains 560 ms but still cannot fit the saved raw clip
        # inside the actual speaker turn. Do not claim alignment solves this.
        raw = 2.159455782312925
        options = dict(corridor_start=334.845, corridor_end=336.345)
        narrow = schedule_turn([clip(start=335.48, end=336.36, duration=raw)], **options)
        source = schedule_turn([clip(start=334.84, end=336.28, duration=raw)], **options)
        self.assertEqual(narrow["status"], "blocked")
        self.assertEqual(source["status"], "blocked")
        # A retained waveform of 1.4s (hypothetical fixture, not measured trim)
        # fits the true source turn, while the caption anchor still blocks it.
        self.assertEqual(schedule_turn([clip(start=334.84, end=336.28, duration=1.4)], **options)["rate"], 1)
        self.assertEqual(schedule_turn([clip(start=335.48, end=336.36, duration=1.4)], **options)["status"], "blocked")

    def test_long_passage_returns_no_clipped_or_partial_output(self):
        rows = [clip("a", .16, 17.52, 21.873197278911565),
                clip("b", 17.6, 24.96, 9.543401360544218)]
        result = schedule_turn(rows, corridor_start=0, corridor_end=25.21)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["schedule"], [])
        self.assertGreater(result["diagnostics"]["conflicts"][0]["deficit_seconds"], 1)

    def test_backward_pass_advances_prior_clip_to_protect_next_anchor(self):
        rows = [clip("a", 1, 2, 1.15), clip("b", 2, 3, 1.1)]
        result = schedule_turn(rows, corridor_start=0, corridor_end=3, max_rate=1)
        self.assertEqual(result["status"], "ready")
        a, b = result["schedule"]
        self.assertAlmostEqual(a["audible_start"], .75)
        self.assertAlmostEqual(b["audible_start"], 1.9)

    def test_short_source_pause_is_retained(self):
        result = schedule_turn([clip("a", 1, 2, 1), clip("b", 2.08, 3.08, 1)],
                               corridor_start=0, corridor_end=4)
        a, b = result["schedule"]
        self.assertAlmostEqual(b["audible_start"] - a["audible_end"], .08)

    def test_empty_turn(self):
        result = schedule_turn([], corridor_start=0, corridor_end=1)
        self.assertEqual(result["schedule"], [])
        self.assertEqual(result["rate"], 1)

    def test_precise_cap_and_corridor(self):
        result = schedule_turn([clip(start=0, end=1, duration=1.15)],
                               corridor_start=0, corridor_end=1, max_shift=0)
        self.assertEqual(result["status"], "ready")
        self.assertAlmostEqual(result["rate"], 1.15, places=7)
        self.assertLessEqual(result["schedule"][0]["end"], 1 + 1e-8)

    def test_invalid_input_and_experimental_bounds(self):
        invalid = [clip(alignment_verified=False), clip(alignment_verified=1),
                   clip(speaker=None), clip(start=-1), clip(start=math.nan),
                   clip(duration=math.inf), clip(duration=0), clip(duration=True),
                   clip(duration=.2, head_guard=.1, tail_guard=.1), clip(head_guard=-.1)]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                schedule_turn([row], corridor_start=0, corridor_end=4)
        for rows in ([clip(), clip()], [clip("a"), clip("b", 1.5, 3)],
                     [clip("a"), clip("b", 3, 4, speaker="other")], [clip()] * 10001):
            with self.assertRaises(ValueError):
                schedule_turn(rows, corridor_start=0, corridor_end=5)
        for options in ({"max_rate": 1.16}, {"max_rate": .9}, {"max_shift": .251},
                        {"corridor_end": 0}, {"corridor_start": math.nan}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                schedule_turn([clip()], **{**dict(corridor_start=0, corridor_end=5), **options})

    def test_randomized_ready_results_preserve_all_constraints_and_inputs(self):
        rng = random.Random(813)
        for case in range(400):
            rows, time = [], .5
            for i in range(rng.randint(1, 12)):
                window = rng.uniform(.1, 3)
                head, tail = rng.uniform(0, .08), rng.uniform(0, .08)
                duration = window * rng.uniform(.5, 1.6) + head + tail
                rows.append(clip(str(i), time, time + window, duration, head_guard=head, tail_guard=tail))
                time += window + rng.uniform(0, .7)
            before = copy.deepcopy(rows)
            result = schedule_turn(rows, corridor_start=0, corridor_end=time + .2)
            self.assertEqual(rows, before)
            if result["status"] == "blocked":
                self.assertEqual(result["schedule"], [])
                self.assertTrue(result["diagnostics"]["conflicts"])
                continue
            self.assertTrue(1 <= result["rate"] <= 1.15)
            self.assertEqual(len(rows), len(result["schedule"]))
            for i, row in enumerate(result["schedule"]):
                self.assertGreaterEqual(row["start"], -1e-8)
                self.assertLessEqual(row["end"], time + .2 + 1e-8)
                self.assertLessEqual(abs(row["start_shift"]), .25 + 1e-8)
                self.assertLessEqual(row["end_shift"], .25 + 1e-8)
                self.assertAlmostEqual(row["end"] - row["start"], row["retained_duration"] / result["rate"])
                if i:
                    previous = result["schedule"][i - 1]
                    self.assertGreaterEqual(row["start"] + 1e-8, previous["end"])
                    gap = min(rows[i]["source_start"] - rows[i-1]["source_end"], .1)
                    self.assertGreaterEqual(row["audible_start"] - previous["audible_end"] + 1e-8, gap)


if __name__ == "__main__":
    unittest.main()
