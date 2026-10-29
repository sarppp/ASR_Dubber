"""Validate the frozen experimental scope without importing Modal or a model."""
import ast
import copy
import json
from pathlib import Path
import unittest

from batch_dubbing.naturalness_experiment import build_jobs
from modal_control.plans import validate_job

ROOT = Path(__file__).resolve().parents[1]


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "dubbing_naturalness_2026-10-28/experiment_inputs.json").read_text())

    def test_reviewed_scope_and_call_count(self):
        jobs = build_jobs(self.data)
        self.assertEqual(len(jobs), 22)
        self.assertLessEqual(len(jobs) + 1, 32)
        self.assertAlmostEqual(sum(e["end"] - e["start"] for e in self.data["excerpts"]), 49.76)
        self.assertEqual(jobs[0]["excerpt_id"], "bill_ackman")
        self.assertEqual(jobs[0]["arm"], "baseline")

    def test_both_arms_retain_identical_words(self):
        jobs = build_jobs(self.data)
        for excerpt in self.data["excerpts"]:
            expected = " ".join(c["text"] for c in excerpt["german_cues"])
            for arm in ("baseline", "candidate"):
                actual = " ".join(j["phrase"]["text"] for j in jobs if j["excerpt_id"] == excerpt["id"] and j["arm"] == arm)
                self.assertEqual(actual, expected)

    def test_mind_control_sentence_is_not_cut_at_brain(self):
        jobs = build_jobs(self.data)
        baseline = next(j for j in jobs if j["excerpt_id"] == "mind_control" and j["arm"] == "baseline")
        candidate = next(j for j in jobs if j["excerpt_id"] == "mind_control" and j["arm"] == "candidate")
        self.assertTrue(baseline["phrase"]["text"].endswith("ins Gehirn"))
        self.assertTrue(candidate["phrase"]["text"].endswith("aktivieren."))
        self.assertEqual(candidate["phrase"]["end"], 17.52)

    def test_unknown_speaker_or_modified_model_rejected(self):
        for mutation in ("voice", "model", "scope", "count"):
            data = copy.deepcopy(self.data)
            if mutation == "voice":
                data["excerpts"][0]["voices"] = {}
            elif mutation == "model":
                data["model"] = "other"
            elif mutation == "scope":
                data["excerpts"].append(copy.deepcopy(data["excerpts"][0]))
            else:
                data["workload_calls"] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                build_jobs(data)

    def test_overlap_and_partial_cue_bounds_rejected(self):
        for mutation in ("overlap", "bounds"):
            data = copy.deepcopy(self.data)
            if mutation == "overlap":
                data["excerpts"][0]["german_cues"][1]["start"] = 325
            else:
                data["excerpts"][0]["start"] += .1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                build_jobs(data)

    def test_worker_uses_explicit_speed_and_discloses_unchanged_chunk_defaults(self):
        tree = ast.parse((ROOT / "batch_dubbing/naturalness_worker.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "synthesize"]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: item.value for item in calls[0].keywords}
        self.assertEqual(ast.literal_eval(keywords["speed"]), 1.0)
        self.assertNotIn("max_chunk_length", keywords)
        self.assertNotIn("silence_duration", keywords)
        report_fields = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                report_fields.update({key.value: value for key, value in zip(node.keys, node.values)
                                      if isinstance(key, ast.Constant) and isinstance(key.value, str)})
        self.assertEqual(ast.literal_eval(report_fields["synthesis_speed"]), 1.0)
        chunking = ast.literal_eval(report_fields["chunking"])
        self.assertEqual(chunking["max_chunk_length"], 300)
        self.assertEqual(chunking["silence_duration_seconds"], .3)
        self.assertEqual(ast.literal_eval(report_fields["timing_conditioning"]),
                         {"target_duration": False, "internal_pause_timeline": False})

    def test_catalog_passes_controller_static_validation(self):
        catalog = json.loads((ROOT / "modal_control/catalog/naturalness_native_grouping.json").read_text())
        job = catalog["jobs"]["naturalness_native_grouping"]
        self.assertEqual(validate_job(job)["hardware"], "CPU")
        self.assertEqual(job["max_seconds"], 600)
        self.assertEqual(job["review"]["max_total_seconds"], 1200)
        for source in job["source_files"]:
            self.assertTrue((ROOT / source).is_file())


if __name__ == "__main__":
    unittest.main()
