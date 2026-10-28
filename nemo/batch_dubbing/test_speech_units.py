import copy
import ast
from dataclasses import FrozenInstanceError
import json
import math
from pathlib import Path
import random
import unittest

from batch_dubbing.speech_units import (
    bind_caption_translations, bind_translations, build_shorter_revision_request, build_speech_units,
    reconcile_speaker_corridors,
)


def word(text, start, end, speaker="speaker_a"):
    return {"word": text, "start": start, "end": end, "speaker": speaker}


def turn(start=0, end=100, speaker="speaker_a"):
    return {"start": start, "end": end, "speaker": speaker}


class SpeechUnitTests(unittest.TestCase):
    def test_reconciliation_handles_realistic_boundary_errors_without_retiming_words(self):
        for start, turn_start in ((4.32, 4.48), (.96, 1.11), (.64, .92)):
            rows = [word("First.", start, turn_start + .4)]
            turns = [turn(turn_start, turn_start + .2)]
            original = copy.deepcopy((rows, turns))
            result = reconcile_speaker_corridors(rows, turns)
            self.assertEqual(result["status"], "ready")
            units = build_speech_units(rows, result["turns"], rounding_tolerance=0)
            self.assertEqual(units[0].source_start, start)
            self.assertEqual(units[0].source_end, turn_start + .4)
            self.assertEqual((rows, turns), original)
            self.assertGreater(result["adjustments"][0]["leading_expansion_seconds"], .02)
            self.assertGreater(result["adjustments"][0]["trailing_expansion_seconds"], .02)

    def test_reconciliation_combines_same_speaker_turns_and_preserves_pause(self):
        rows = [word("One.", .2, .9), word("Two.", 3, 3.8)]
        result = reconcile_speaker_corridors(rows, [turn(.3, .8), turn(3.1, 3.6)])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(len(result["turns"]), 1)
        self.assertEqual(result["adjustments"][0]["raw_turn_indices"], [0, 1])
        units = build_speech_units(rows, result["turns"])
        self.assertEqual([u.source_start for u in units], [.2, 3])

    def test_boundary_word_uses_unique_overlap_winner_and_corridors_never_cross(self):
        rows = [word("A", .8, 1.2), word("B", 1.2, 1.8, "speaker_b")]
        result = reconcile_speaker_corridors(rows, [turn(0, 1.1), turn(1.1, 2, "speaker_b")])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["turns"][0]["end"], 1.2)
        self.assertEqual(result["turns"][1]["start"], 1.2)
        self.assertEqual(result["word_evidence"][0]["speaker"], "speaker_a")
        units = build_speech_units(rows, result["turns"], rounding_tolerance=0)
        self.assertEqual([i for u in units for i in u.source_word_indices], [0, 1])

    def test_reconciliation_refuses_missing_ambiguous_conflicting_and_overlap_evidence(self):
        cases = [
            ([word("A", 0, 1)], [turn(2, 3)]),
            ([word("A", .8, 1.2)], [turn(0, 1), turn(1, 2, "speaker_b")]),
            ([word("A", .8, 1.2, "speaker_b")], [turn(0, 1.1), turn(1.1, 2, "speaker_b")]),
            ([word("A", 0, 1, "unknown")], [turn(0, 1)]),
            ([word("A", 0, 1)], [turn(0, 1), turn(.5, 2, "speaker_b")]),
            ([word("A", 0, 1), word("B", .9, 2)], [turn(0, 2)]),
        ]
        for rows, turns in cases:
            with self.subTest(rows=rows, turns=turns):
                result = reconcile_speaker_corridors(rows, turns)
                self.assertEqual(result["status"], "blocked")
                self.assertEqual(result["turns"], [])
                self.assertTrue(result["conflicts"])

    def test_same_speaker_overlap_counts_each_sample_once(self):
        result = reconcile_speaker_corridors([word("A", 0, 2)], [turn(0, 1.5), turn(.5, 2)])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["word_evidence"][0]["overlap_seconds"], {"speaker_a": 2})

    def test_random_reconciled_runs_preserve_all_source_word_bounds(self):
        rng = random.Random(991)
        rows, turns, now = [], [], 0.
        for index in range(500):
            duration = rng.uniform(.2, .8)
            speaker = "speaker_a" if index % 7 < 4 else "speaker_b"
            rows.append(word(str(index), now, now + duration, speaker))
            turns.append(turn(now + duration*.2, now + duration*.8, speaker))
            now += duration + rng.uniform(.01, .2)
        result = reconcile_speaker_corridors(rows, turns)
        self.assertEqual(result["status"], "ready")
        self.assertTrue(all(a["end"] <= b["start"] for a, b in zip(result["turns"], result["turns"][1:])))
        units = build_speech_units(rows, result["turns"], rounding_tolerance=0)
        self.assertEqual([(w.start, w.end) for u in units for w in u.words],
                         [(r["start"], r["end"]) for r in rows])

    def test_sentence_boundaries_keep_exact_words_and_times(self):
        rows = [word(" Hello", .123456, .4), word("world.", .4, .93),
                word("Next!", .94, 1.123456)]
        original = copy.deepcopy(rows)
        units = build_speech_units(rows, [turn()])
        self.assertEqual([u.source_text for u in units], ["Hello world.", "Next!"])
        self.assertEqual(units[0].source_start, .123456)
        self.assertEqual(units[-1].source_end, 1.123456)
        self.assertEqual([w.word for u in units for w in u.words], [r["word"] for r in rows])
        self.assertEqual(rows, original)

    def test_same_rules_work_for_varied_language_text(self):
        for text in ("A short sentence.", "Ein kurzer Satz.", "Une phrase courte.", "句子。"):
            with self.subTest(text=text):
                units = build_speech_units([word(text, .1, 1), word("Following", 1, 2)], [turn()])
                self.assertEqual(len(units), 2)
                self.assertEqual(units[0].flags, ())

    def test_speaker_turn_and_meaningful_pause_boundaries(self):
        rows = [word("one", .1, .3), word("two", .9, 1.1), word("three", 2, 2.5, "speaker_b")]
        units = build_speech_units(rows, [turn(0, 1.5), turn(1.5, 3, "speaker_b")])
        self.assertEqual(len(units), 3)
        self.assertEqual(units[-1].speaker, "speaker_b")
        self.assertEqual(units[-1].corridor_start, 1.5)

    def test_soft_clause_boundary_and_hard_limit_do_not_drop_words(self):
        rows = [word("word", i, i + 1) for i in range(35)]
        rows[20]["word"] = "clause,"
        units = build_speech_units(rows, [turn()])
        self.assertEqual(units[0].source_end, 21)
        self.assertIn("soft_duration_boundary", units[0].flags)
        rows[20]["word"] = "clause"
        units = build_speech_units(rows, [turn()])
        self.assertEqual(units[0].source_end, 30)
        self.assertIn("unfinished_at_size_limit", units[0].flags)
        self.assertEqual([i for u in units for i in u.source_word_indices], list(range(35)))

    def test_character_limit_and_oversized_word_are_explicit(self):
        units = build_speech_units([word("abc", 0, 1), word("def", 1, 2),
                                    word("x" * 20, 2, 40)], [turn()], max_characters=5)
        self.assertEqual([u.source_text for u in units], ["abc", "def", "x" * 20])
        self.assertIn("character_limit", units[0].flags)
        self.assertIn("oversized_source_word", units[-1].flags)

    def test_ids_are_stable_namespaced_and_independent_of_translation(self):
        rows = [word("source.", 1, 2)]
        a = build_speech_units(rows, [turn()], source_id="asset1")
        b = build_speech_units(copy.deepcopy(rows), [turn()], source_id="asset1")
        c = build_speech_units(rows, [turn()], source_id="asset2")
        self.assertEqual(a, b)
        self.assertNotEqual(a[0].id, c[0].id)
        self.assertEqual(bind_translations(a, {a[0].id: "Cible."})[0].id, a[0].id)

    def test_frozen_records_and_export_cannot_mutate_source(self):
        unit = build_speech_units([word("source.", 1, 2)], [turn()])[0]
        with self.assertRaises(FrozenInstanceError):
            unit.source_start = 9
        with self.assertRaises(FrozenInstanceError):
            unit.words[0].start = 9
        exported = unit.to_dict()
        exported["words"][0]["start"] = 9
        self.assertEqual(unit.words[0].start, 1)
        self.assertEqual(exported["timing_basis"], "asr_words")
        json.dumps(unit.to_dict(), allow_nan=False)

    def test_translation_binding_preserves_all_provenance(self):
        units = build_speech_units([word("First.", 0, 1), word("Second.", 1, 2)], [turn()])
        bound = bind_translations(units, {u.id: "Une traduction de longueur différente." for u in units})
        for before, after in zip(units, bound):
            a, b = before.to_dict(), after.to_dict()
            b["translated_text"] = None
            self.assertEqual(a, b)
            self.assertIsNone(before.translated_text)

    def test_caption_translation_joins_text_without_retiming_source(self):
        units = build_speech_units([word("First", 0, 1), word("sentence.", 1, 2),
                                   word("Second.", 3, 4)], [turn()])
        results = {
            units[0].id: [{"start": .1, "end": .7, "text": "Ein", "speaker": "wrong"},
                          {"start": .7, "end": 1.8, "text": "Satz."}],
            units[1].id: [{"start": 3, "end": 4, "text": "Noch einer."}],
        }
        before = copy.deepcopy(results)
        bound, captions = bind_caption_translations(units, results)
        self.assertEqual(bound[0].translated_text, "Ein Satz.")
        self.assertEqual(bound[0].source_start, 0)
        self.assertEqual(bound[0].source_end, 2)
        self.assertEqual([row["source_id"] for row in captions], [units[0].id, units[0].id, units[1].id])
        self.assertTrue(all(row["speaker"] == "speaker_a" and row["timing_basis"] == "display_estimate" for row in captions))
        self.assertEqual(results, before)

    def test_empty_missing_or_outside_unit_caption_translation_fails(self):
        units = build_speech_units([word("source.", 1, 2)], [turn()])
        cases = ({}, {units[0].id: []},
                 {units[0].id: [{"start": 1, "end": 2, "text": " "}]},
                 {units[0].id: [{"start": .9, "end": 2, "text": "target"}]},
                 {units[0].id: [{"start": 1, "end": math.inf, "text": "target"}]})
        for result in cases:
            with self.subTest(result=result), self.assertRaises(ValueError):
                bind_caption_translations(units, result)

    def test_gpu_workers_persist_source_contract_without_importing_modal(self):
        root = Path(__file__).resolve().parents[1]
        for filename in ("batch_dubbing/gpu_worker.py", "modal_control/smoke_gpu_worker.py"):
            tree = ast.parse((root / filename).read_text())
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            self.assertTrue(any(isinstance(call.func, ast.Name) and call.func.id == "build_speech_units" for call in calls))
            self.assertTrue(any(isinstance(call.func, ast.Name) and call.func.id == "reconcile_speaker_corridors" for call in calls))
            self.assertTrue(any(isinstance(call.func, ast.Name) and call.func.id == "bind_caption_translations" for call in calls))
            inference = [call for call in calls if isinstance(call.func, ast.Name) and call.func.id == "infer_chunk"]
            self.assertEqual(len(inference), 1)
            self.assertEqual([arg.attr for arg in inference[0].args[-2:]], ["source_start", "source_end"])
            result = next(node for node in ast.walk(tree) if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict))
            keys = {key.value for key in result.value.keys if isinstance(key, ast.Constant)}
            self.assertTrue({"source_words", "speech_units", "translations", "source_id",
                             "raw_turns", "reconciled_turns", "alignment", "alignment_artifact",
                             "source_language", "alignment_sha256"} <= keys)
        for filename in ("new_guys_gpu_smoke.json", "new_guys_full_gpu.json"):
            catalog = json.loads((root / "modal_control/catalog" / filename).read_text())
            for job in catalog["jobs"].values():
                self.assertIn("batch_dubbing/speech_units.py", job["source_files"])
                self.assertEqual(job["review"]["hardware"], "T4")
                self.assertEqual(job["review"]["max_containers"], 1)

    def test_missing_extra_blank_and_duplicate_translation_identity_rejected(self):
        units = build_speech_units([word("Text.", 0, 1)], [turn()])
        for mapping in ({}, {units[0].id: "ok", "extra": "bad"}, {units[0].id: " "},
                        {units[0].id: math.nan}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                bind_translations(units, mapping)
        with self.assertRaises(ValueError):
            bind_translations((units[0], units[0]), {units[0].id: "ok"})

    def test_unknown_uncovered_and_ambiguous_assignments_rejected(self):
        for rows, turns in (([word("x", 0, 1, None)], [turn()]),
                            ([word("x", 0, 1, "unknown")], [turn(speaker="unknown")]),
                            ([word("x", 0, 1)], [turn(2, 3)]),
                            ([word("x", .99, 1.01)], [turn(0, 1), turn(1, 2)])):
            with self.subTest(rows=rows, turns=turns), self.assertRaises(ValueError):
                build_speech_units(rows, turns)

    def test_rounding_tolerance_is_flagged_without_clamping_times(self):
        unit = build_speech_units([word("x.", .99, 1.99)], [turn(1, 2)])[0]
        self.assertEqual(unit.source_start, .99)
        self.assertEqual(unit.corridor_start, 1)
        self.assertIn("turn_boundary_rounding_tolerance_used", unit.flags)
        with self.assertRaises(ValueError):
            build_speech_units([word("x.", .97, 1.99)], [turn(1, 2)])

    def test_overlap_nonfinite_empty_words_and_invalid_policy_rejected(self):
        bad_rows = ([word("x", 0, 1), word("y", .9, 2)],
                    [word("x", math.nan, 1)], [word("x", 0, math.inf)],
                    [word("", 0, 1)], [word("x", 0, 0)])
        for rows in bad_rows:
            with self.assertRaises(ValueError):
                build_speech_units(rows, [turn()])
        with self.assertRaises(ValueError):
            build_speech_units([], [turn(0, 2), turn(1, 3)])
        for options in ({"soft_max_seconds": 31}, {"hard_max_seconds": 0},
                        {"max_characters": 0}, {"rounding_tolerance": .021},
                        {"pause_seconds": 0}):
            with self.assertRaises(ValueError):
                build_speech_units([], [], **options)

    def test_revision_request_preserves_meaning_and_actual_timing(self):
        unit = build_speech_units([word("A named value: 25 volts.", 1, 3)], [turn()])[0]
        unit = bind_translations([unit], {unit.id: "Une valeur nommée : 25 volts."})[0]
        request = build_shorter_revision_request(unit, retained_duration=4,
                                                 target_available_seconds=2, max_rate=1.15)
        self.assertEqual(request["source_unit_id"], unit.id)
        self.assertEqual(request["source_text"], unit.source_text)
        self.assertAlmostEqual(request["maximum_native_duration_seconds"], 2.3)
        self.assertIn("Do not delete information", " ".join(request["instructions"]))
        self.assertIn("cannot_shorten_faithfully", " ".join(request["instructions"]))
        for args in ({"retained_duration": math.nan}, {"target_available_seconds": 0}, {"max_rate": .9}):
            with self.assertRaises(ValueError):
                build_shorter_revision_request(unit, **{**dict(retained_duration=4, target_available_seconds=2), **args})

    def test_random_source_partition_has_no_missing_or_duplicated_words(self):
        rng = random.Random(671)
        rows, time = [], 0.0
        for index in range(1200):
            duration = rng.uniform(.01, .5)
            token = f"w{index}" + ("." if index % 11 == 0 else "")
            rows.append(word(token, time, time + duration))
            time += duration + rng.uniform(0, .6)
        units = build_speech_units(rows, [turn(0, time + 1)], source_id="randomized")
        flattened = [w for u in units for w in u.words]
        self.assertEqual([w.index for w in flattened], list(range(len(rows))))
        self.assertEqual([(w.word, w.start, w.end, w.speaker) for w in flattened],
                         [(r["word"], r["start"], r["end"], r["speaker"]) for r in rows])


if __name__ == "__main__":
    unittest.main()
