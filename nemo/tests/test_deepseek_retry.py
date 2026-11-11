"""Offline regressions for the DeepSeek handoff: no models or vendor calls.

Run: python3 nemo/tests/test_deepseek_retry.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import wave

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "nemo"), str(ROOT / "qwen3-tts")]
if os.getenv("DEEPSEEK_STAGED_CODE"):
    sys.path.insert(0, os.environ["DEEPSEEK_STAGED_CODE"])

import pipeline_subtitle as ps
from dub_srt import parse_srt
from dub_compress import Compressor, Candidate
from dub_timing import check_dub, Report
from dub_cache import speech_key, save, restore


def wav(path, seconds=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as h:
        h.setnchannels(1)
        h.setsampwidth(2)
        h.setframerate(24000)
        h.writeframes(b"\x20\x00" * int(24000 * seconds))


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.original = "Diese wichtige neue Methode verändert die Forschung in vielen Ländern."
        self.short = "Diese neue Methode verändert die Forschung in vielen Ländern."
        self.rows = [{"index": i + 1, "start": a, "end": b, "span": b - a,
                      "text_before": self.original, "text_after": self.original,
                      "ratio_before": 1., "ratio_after": 1.}
                     for i, (a, b) in enumerate([(0., 4.), (4., 8.), (8.1, 12.)])]
        self.dub = self.root / "clip.de.dub.srt"
        self.dub.write_text(ps._render_srt([
            (r["start"], r["end"], r["text_after"]) for r in self.rows]), encoding="utf-8-sig")
        self.source = self.root / "clip.en.srt"
        self.source.write_text(ps._render_srt([
            (r["start"], r["end"], f"Source sentence {i}.") for i, r in enumerate(self.rows)]))
        self.report = self.root / "fit_report.json"
        self.failure = {"index": 99, "start": 4., "end": 8., "span": 4.,
                        "tempo": 1.2, "needed": 4.8}
        self.report.write_text(json.dumps({"segments": [self.failure]}))

    def retry(self, answer, compressor=None, source=True):
        translator = Mock()
        translator.translate_sentence.return_value = answer
        changed = ps._adapt_from_report(
            self.dub, self.source if source else None, "en", "de", self.rows,
            self.report, translator, 1., engine="qwen", compressor=compressor)
        return changed, translator

    def test_bom_preserves_first_sentence(self):
        self.assertEqual([s["index"] for s in parse_srt(self.dub)], [1, 2, 3])

    def test_touching_and_close_sentences_match_exact_onset(self):
        self.assertIs(ps._row_at(self.rows, 4.), self.rows[1])
        self.assertIs(ps._row_at(self.rows, 8.1), self.rows[2])
        self.assertIsNone(ps._row_at(self.rows, 8.05))
        self.assertIsNone(ps._row_at(self.rows, 100.))

    def test_unchanged_translation_reaches_compressor(self):
        compressor = Mock()
        compressor.shorten.return_value = self.short, {"engine": "fake"}
        changed, translator = self.retry(self.original, compressor)
        self.assertEqual(changed, [4.])
        self.assertEqual(translator.translate_sentence.call_args.args[0], "Source sentence 1.")
        compressor.shorten.assert_called_once()
        self.assertEqual(self.rows[0]["text_after"], self.original)
        self.assertEqual(self.rows[2]["text_after"], self.original)
        self.assertEqual(self.rows[1]["ratio_after_kind"], "estimate_pending_tts")

    def test_unchanged_without_compressor_does_not_retry(self):
        before = self.dub.read_bytes()
        self.assertEqual(self.retry(self.original)[0], [])
        self.assertEqual(self.dub.read_bytes(), before)

    def test_longer_translation_does_not_retry(self):
        self.assertEqual(self.retry(self.original + " Und vieles mehr.")[0], [])

    def test_shorter_but_over_budget_reaches_compressor(self):
        compressor = Mock()
        compressor.shorten.return_value = self.short, {"engine": "fake"}
        self.retry(self.original.replace("wichtige ", ""), compressor)
        compressor.shorten.assert_called_once()
        fits = compressor.shorten.call_args.kwargs["fits"]
        self.assertFalse(fits(self.original))
        self.assertTrue(fits("x" * self.rows[1]["retry_budget"]))

    def test_missing_source_fails_before_translation(self):
        with self.assertRaisesRegex(ValueError, "original source SRT"):
            self.retry(self.short, source=False)

    def test_stale_report_text_is_rejected(self):
        self.failure["text"] = "A different script revision"
        self.report.write_text(json.dumps({"segments": [self.failure]}))
        changed, translator = self.retry(self.short)
        self.assertEqual(changed, [])
        translator.translate_sentence.assert_not_called()

    def test_merged_failure_does_not_rewrite_one_sentence(self):
        self.failure["end"] = 12.
        self.report.write_text(json.dumps({"segments": [self.failure]}))
        changed, translator = self.retry(self.short)
        self.assertEqual(changed, [])
        translator.translate_sentence.assert_not_called()

    def test_source_range_does_not_fall_back_to_whole_document(self):
        with self.assertRaises(ValueError):
            ps._span_for_time([(0, 0)], [{"start": 0., "end": 2.}],
                              {"start": 10., "end": 12.})

    def test_number_substring_cannot_pass_guard(self):
        self.assertIsNotNone(ps._preservation_problem(
            "Es dauert 14 Jahre bis zum Ergebnis.",
            "Es dauert 140 Jahre bis zum Ergebnis.", "de"))

    def test_calibrated_cps_is_actually_used(self):
        translator = Mock()
        with patch.object(ps, "measured_overhead", return_value=0.):
            _, rows, _ = ps._adapt(self.dub, self.source, "en", "de", translator,
                                    cps=100., engine="qwen")
        translator.translate_sentence.assert_not_called()
        self.assertLess(rows[0]["ratio_before"], 0.3)

    def test_compression_guard_keeps_original_baseline(self):
        compressor = Compressor([])
        original = "a" * 100
        compressor.candidates = Mock(side_effect=[
            [Candidate("a" * 60, "fake", "plain", 1, 60)],
            [Candidate("a" * 40, "fake", "plain", 1, 40)]])
        text, report = compressor.shorten(original, 20,
            guard=lambda before, after: ps._preservation_problem(before, after, "de"),
            fits=lambda t: len(t) <= 40)
        self.assertEqual(len(text), 60)
        self.assertTrue(any("too short" in r["reason"] for r in report["rejected"]))

    def test_cache_checks_text_voice_and_audio_integrity(self):
        raw = self.root / "raw.wav"
        wav(raw)
        cache = self.root / "raw_cache"
        key = speech_key("Text", {"voice": "A"})
        save(cache, key, raw)
        self.assertTrue(restore(cache, key, self.root / "copy.wav"))
        self.assertFalse(restore(cache, speech_key("Changed", {"voice": "A"}), self.root / "bad.wav"))
        self.assertFalse(restore(cache, speech_key("Text", {"voice": "B"}), self.root / "bad.wav"))
        (cache / f"{key}.wav").write_bytes(b"corrupt" * 200)
        self.assertFalse(restore(cache, key, self.root / "bad.wav"))

    def test_timing_rejects_missing_speech_and_mixed_runs(self):
        clip = self.root / "cache/qwen/one/temp/seg_0001.wav"
        wav(clip)
        checkpoint = clip.parent.parent / "checkpoint.json"
        checkpoint.write_text(json.dumps([{"clip": str(clip), "start": 0., "end": 4.}]))
        result = check_dub(self.root, script=self.dub)
        self.assertFalse(result.ok)
        self.assertTrue(any("missing speech" in w for w in result.warnings))
        other = self.root / "cache/qwen/two/checkpoint.json"
        other.parent.mkdir()
        other.write_text(checkpoint.read_text())
        result = check_dub(self.root, script=self.dub)
        self.assertTrue(any("ambiguous" in w for w in result.warnings))

    def test_short_gap_uses_duration_not_absolute_timestamp(self):
        result = Report(gaps=[(40., 40.1), (60., 60.4)])
        self.assertEqual(result.short_gaps, [(40., 40.1)])

    def test_source_anchored_run_passes_source_through_to_measured_retry(self):
        video = self.root / "clip.mp4"
        video.write_bytes(b"fixture")
        args = SimpleNamespace(input_dir=None, input_file=str(video),
            output_dir=str(self.root / "output"), trim=0, language="en", target_lang="de",
            whisper_model="base", tts_engine="qwen", fit_target=1., max_sentences=3,
            skip_dub=False, dub_timing="source-anchored")
        translator = Mock()
        translator.translate_sentence.return_value = self.short
        with patch.object(ps, "_produce_subtitles", return_value={"en": self.source, "de": self.dub}), \
             patch.object(ps, "_adapt", return_value=(self.dub, self.rows, self.dub)), \
             patch.object(ps, "_calibrated_rate", return_value=20.), \
             patch.object(ps, "_make_compressor", return_value=None), \
             patch.object(ps, "OllamaTranslator", return_value=translator), \
             patch.object(ps, "_dub", side_effect=[(self.root, 1), (self.root, 0)]) as dub_call, \
             patch.object(ps, "_derive_run_label", return_value="review"), \
             patch.object(ps, "_finalize_outputs") as finalize:
            self.assertEqual(ps.run(args), 0)
            self.assertEqual(dub_call.call_count, 2)
            finalize.assert_called_once()
            self.assertEqual(translator.translate_sentence.call_args.args[0], "Source sentence 1.")

    def test_calibration_uses_matching_report_and_includes_failures(self):
        import hashlib
        overhead = 1.
        report = {"engine": "qwen", "language": "de",
                  "script_sha256": hashlib.sha256(self.dub.read_bytes()).hexdigest(),
                  "segments": [{"text": "a" * n, "needed": overhead + n / 20.,
                                "status": "failed" if n == 60 else "passed"}
                               for n in (20, 40, 60)]}
        self.report.write_text(json.dumps(report))
        with patch.object(ps, "measured_overhead", return_value=overhead):
            self.assertEqual(ps._calibrated_rate(self.root, "de", "qwen", self.dub), 20.)
            self.assertEqual(ps._calibrated_rate(self.root, "fr", "qwen", self.dub), ps._rate("fr", "qwen"))

    def test_dub_enforces_fit_target_and_does_not_reuse_stale_report(self):
        work = self.root / "output/dub/clip"
        work.mkdir(parents=True)
        report = work / "fit_report.json"
        report.write_text('{"segments": [{"tempo": 2.0}]}')
        args = SimpleNamespace(skip_dub=False, tts_engine="qwen", supertonic_voice="M1",
            qwen_mode="clone", no_demucs=True, max_speed=1.15, fit_target=1., min_speed=1.,
            merge_gap=1., merge_max_dur=10., tts_workers=1, tts_devices="0")
        with patch.object(ps, "QWEN_DIR", self.root), \
             patch.object(ps, "_validate_translated_srt"), \
             patch.object(ps, "_run", return_value=1) as run:
            ps._dub(args, self.root / "clip.mp4", self.dub, "clip", "de")
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--max-speed") + 1], "1.0")
            self.assertFalse(report.exists())
            self.assertTrue((work / "fit_report.previous.json").exists())


class DubIntegrationTests(unittest.TestCase):
    def test_failed_fit_retry_reuses_unchanged_speech_and_reports_all_cues(self):
        # The progress bar is irrelevant to offline orchestration and is not
        # installed in the system interpreter used by check_dubbing.py.
        with patch.dict(sys.modules, {"tqdm": SimpleNamespace(tqdm=Mock())}):
            import dub
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, script = root / "clip.mp4", root / "clip.srt"
            video.write_bytes(b"fixture")
            work = root / "work"
            spoken = []
            starts = []

            class Voice:
                def __init__(self, *a, **kw): pass
                def start(self): starts.append(True)
                def close(self): pass
                def generate_custom(self, text, voice, language, output):
                    spoken.append(text)
                    wav(output, 5. if text == "Long sentence." else 1.)
                    return True

            def fit(raw, available, **kwargs):
                with wave.open(str(raw)) as h:
                    duration = h.getnframes() / h.getframerate()
                if duration > available:
                    raise ValueError(f"Speech needs {duration:.3f}s after edge cleanup; "
                                     f"required tempo {duration / available:.3f} exceeds 1.000")
                fitted = raw.with_name(raw.stem + "_natural_fit_test.wav")
                wav(fitted, duration)
                fitted.with_suffix(".json").write_text(json.dumps({
                    "edges": {"retained_seconds": duration}, "tempo": 1.}))
                return fitted

            argv = ["dub.py", str(video), str(script), "--workdir", str(work),
                    "--tts-engine", "qwen", "--qwen-mode", "custom", "--language", "de",
                    "--no-demucs", "--max-speed", "1.0", "--tts-workers", "1"]
            with patch.object(sys, "argv", argv), \
                 patch.object(dub, "_qwen_python", return_value=sys.executable), \
                 patch.object(dub, "_qwen_worker", return_value=str(Path(dub.__file__))), \
                 patch.object(dub, "SharedTTSManager", Voice), \
                 patch.object(dub, "speed_fit", side_effect=fit), \
                 patch.object(dub, "write_dub_srt"), \
                 patch.object(dub, "stitch_and_mix", return_value=(root / "final.mp4", [])) as publish:
                script.write_text(ps._render_srt([(0., 3., "First sentence."),
                                                  (4., 7., "Long sentence.")]), encoding="utf-8-sig")
                self.assertEqual(dub.main(), 1)
                publish.assert_not_called()
                first = json.loads((work / "fit_report.json").read_text())
                self.assertEqual([s["status"] for s in first["segments"]], ["passed", "failed"])
                self.assertEqual(spoken, ["First sentence.", "Long sentence."])
                script.write_text(ps._render_srt([(0., 3., "First sentence."),
                                                  (4., 7., "Short sentence.")]), encoding="utf-8-sig")
                self.assertEqual(dub.main(), 0)
                self.assertEqual(spoken, ["First sentence.", "Long sentence.", "Short sentence."])
                second = json.loads((work / "fit_report.json").read_text())
                self.assertEqual(second["status"], "passed")
                self.assertEqual(len(second["segments"]), 2)
                self.assertNotEqual(first["cache_dir"], second["cache_dir"])
                self.assertTrue((Path(first["cache_dir"]) / "checkpoint.json").exists())
                self.assertEqual(dub.main(), 0)  # all fitted clips cached
                self.assertEqual(len(starts), 2)  # no model on cached resume
                self.assertEqual(len(json.loads((work / "fit_report.json").read_text())["segments"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
