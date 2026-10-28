"""Renderer contracts tested with tiny generated fixtures and mocked FFmpeg."""
import array
import copy
import hashlib
import math
from pathlib import Path
import re
import resource
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

from batch_dubbing.natural_render import render_timeline, SAMPLE_RATE, _child_limits


class NaturalRenderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.destination = self.root / "new.wav"
        self.commands = []

    def row(self, identifier="a", *, start=0, end=.15, frames=4410, peak=.25, tempo=1):
        source = self.root / (identifier + ".wav")
        fmt = struct.pack("<HHIIHH", 3, 1, SAMPLE_RATE, SAMPLE_RATE * 4, 4, 32)
        payload = struct.pack("<f", peak) * frames
        content = b"WAVEfmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
        source.write_bytes(b"RIFF" + struct.pack("<I", len(content)) + content)
        return {"id": identifier, "file": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "start": start, "end": end, "tempo": tempo,
                "edges": {"start_sample": 0, "end_sample": frames, "input_samples": frames,
                          "sample_rate": SAMPLE_RATE, "peak": peak}}

    @staticmethod
    def schedule(*rows):
        return {"status": "ready", "sample_rate": SAMPLE_RATE, "duration_seconds": .5, "rows": list(rows)}

    def fake_ffmpeg(self, command, timeout):
        self.commands.append(command)
        self.assertLessEqual(timeout, 90)
        filters = command[command.index("-af") + 1]
        start, end = map(int, re.search(r"start_sample=(\d+):end_sample=(\d+)", filters).groups())
        rate = re.search(r"atempo=([\d.]+)", filters)
        count = round((end - start) / float(rate[1])) if rate else end - start
        with wave.open(command[-1], "wb") as handle:
            handle.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
            handle.writeframes(struct.pack("<h", 1234) * count)

    def render(self, schedule):
        with patch("batch_dubbing.natural_render._run_ffmpeg", side_effect=self.fake_ffmpeg):
            return render_timeline(schedule, self.destination)

    def test_native_render_keeps_every_cropped_sample_and_silent_gaps(self):
        row = self.row(start=.05, end=.2)
        original = Path(row["file"]).read_bytes()
        report = self.render(self.schedule(row))
        self.assertEqual(report["rows"][0]["output_samples"], 4410)
        self.assertEqual(report["global_gain"], 1)
        self.assertFalse(any("atempo" in item for item in self.commands[0]))
        with wave.open(str(self.destination), "rb") as handle:
            self.assertEqual(handle.getnframes(), 22050)
            data = array.array("h", handle.readframes(handle.getnframes()))
        start = math.ceil(.05 * SAMPLE_RATE)
        self.assertTrue(all(value == 0 for value in data[:start]))
        self.assertEqual(list(data[start:start + 4410]), [1234] * 4410)
        self.assertTrue(all(value == 0 for value in data[start + 4410:]))
        self.assertEqual(Path(row["file"]).read_bytes(), original)

    def test_exact_edge_crop_and_tempo_has_no_post_fit_trimming(self):
        row = self.row(tempo=1.15)
        row["edges"].update(start_sample=100, end_sample=4000)
        report = self.render(self.schedule(row))
        filters = report["rows"][0]["filters"]
        self.assertEqual(filters[0], "atrim=start_sample=100:end_sample=4000")
        self.assertTrue(filters[-1].startswith("atempo=1.150"))
        self.assertEqual(sum("atrim" in item for item in filters), 1)

    def test_one_shared_attenuation_for_all_inputs(self):
        report = self.render(self.schedule(self.row("a", peak=2), self.row("b", start=.2, end=.4)))
        self.assertAlmostEqual(report["global_gain"], .495)
        self.assertTrue(report["global_gain_applied"])
        gains = [next(item for item in row["filters"] if item.startswith("volume=")) for row in report["rows"]]
        self.assertEqual(gains, ["volume=0.495000000000"] * 2)

    def test_overflow_fails_whole_render_without_published_partial_timeline(self):
        first = self.row("a", end=.15)
        second = self.row("b", start=.2, end=.23, tempo=1.1)
        with self.assertRaisesRegex(ValueError, "nothing was truncated"):
            self.render(self.schedule(first, second))
        self.assertEqual(len(self.commands), 2)
        self.assertFalse(self.destination.exists())
        self.assertFalse(list(self.root.glob(".natural-render-*")))

    def test_fractional_deadline_does_not_drop_a_sample_to_fit(self):
        row = self.row(start=.000001, end=.1)
        with self.assertRaisesRegex(ValueError, "nothing was truncated"):
            self.render(self.schedule(row))
        self.assertFalse(self.destination.exists())

    def test_blocked_schedules_and_duplicate_ids_fail_before_any_child(self):
        row = self.row()
        cases = [self.schedule(row), self.schedule(row, copy.deepcopy(row)), self.schedule(row)]
        cases[0]["status"] = "needs_rewrite"
        cases[2]["rows"][0] = {**row, "status": "blocked"}
        for schedule in cases:
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                self.render(schedule)
        self.assertEqual(self.commands, [])

    def test_overlapping_windows_rejected(self):
        with self.assertRaises(ValueError):
            self.render(self.schedule(self.row("a", end=.2), self.row("b", start=.1, end=.4)))
        self.assertEqual(self.commands, [])

    def test_invalid_times_rates_and_crops_rejected(self):
        original = self.row()
        cases = []
        for key, value in (("tempo", .99), ("tempo", 1.151), ("tempo", math.nan),
                           ("start", -.1), ("start", True), ("end", math.inf), ("end", 0)):
            row = copy.deepcopy(original)
            row[key] = value
            cases.append(row)
        for key, value in (("start_sample", -1), ("start_sample", True), ("end_sample", 4411),
                           ("input_samples", 99), ("sample_rate", 24000), ("peak", .1)):
            row = copy.deepcopy(original)
            row["edges"][key] = value
            cases.append(row)
        for row in cases:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.render(self.schedule(row))
        self.assertEqual(self.commands, [])

    def test_changed_source_hash_fails_before_child(self):
        row = self.row()
        row["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash changed"):
            self.render(self.schedule(row))
        self.assertEqual(self.commands, [])

    def test_existing_destination_is_never_overwritten(self):
        self.destination.write_bytes(b"existing user artifact")
        with self.assertRaises(FileExistsError):
            self.render(self.schedule(self.row()))
        self.assertEqual(self.destination.read_bytes(), b"existing user artifact")
        self.assertEqual(self.commands, [])

    def test_late_destination_collision_is_exclusive(self):
        def collision(command, timeout):
            self.fake_ffmpeg(command, timeout)
            self.destination.write_bytes(b"concurrent artifact")
        with patch("batch_dubbing.natural_render._run_ffmpeg", side_effect=collision):
            with self.assertRaises(FileExistsError):
                render_timeline(self.schedule(self.row()), self.destination)
        self.assertEqual(self.destination.read_bytes(), b"concurrent artifact")

    def test_missing_native_samples_are_not_silently_accepted(self):
        def shortened(command, timeout):
            self.fake_ffmpeg(command, timeout)
            with wave.open(command[-1], "wb") as handle:
                handle.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
                handle.writeframes(b"\0\0" * 200)
        with patch("batch_dubbing.natural_render._run_ffmpeg", side_effect=shortened):
            with self.assertRaisesRegex(ValueError, "sample count"):
                render_timeline(self.schedule(self.row()), self.destination)
        self.assertFalse(self.destination.exists())

    def test_child_resource_guards_are_installed_without_touching_parent(self):
        with patch("batch_dubbing.natural_render.os.sched_getaffinity", return_value={3, 5}), \
                patch("batch_dubbing.natural_render.os.sched_setaffinity") as affinity, \
                patch("batch_dubbing.natural_render.os.nice") as nice, \
                patch("batch_dubbing.natural_render.resource.setrlimit") as limit:
            _child_limits()
        affinity.assert_called_once_with(0, {3})
        nice.assert_called_once_with(15)
        limit.assert_any_call(resource.RLIMIT_AS, (512 * 1024**2,) * 2)
        limit.assert_any_call(resource.RLIMIT_CPU, (60, 60))

    def test_total_deadline_stops_before_child_or_publication(self):
        row = self.row()
        with patch("batch_dubbing.natural_render.time.monotonic", side_effect=[0, 301]):
            with self.assertRaises(TimeoutError):
                self.render(self.schedule(row))
        self.assertFalse(self.destination.exists())
        self.assertEqual(self.commands, [])


if __name__ == "__main__":
    unittest.main()
