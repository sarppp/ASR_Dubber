"""Tiny synthetic WAV regressions; no model or source-media processing."""
import array
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

from batch_dubbing import fit_audio


def write_pcm(path, values, rate=24000):
    with wave.open(str(path), "wb") as out:
        out.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        samples = array.array("h", (round(v * 32767) for v in values))
        if sys.byteorder != "little":
            samples.byteswap()
        out.writeframes(samples.tobytes())


def tone(seconds=.3):
    return [.2 * math.sin(2 * math.pi * 440 * i / 24000) for i in range(round(seconds * 24000))]


class FitAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "seg_1.wav"
        write_pcm(self.path, tone())

    def metadata(self, output):
        return json.loads(output.with_suffix(".json").read_text())

    def test_invalid_old_policy_rejected_before_processing(self):
        for minimum, maximum in ((.65, 1.15), (1, 1.5), (1, float("nan")), (True, 1.15)):
            with self.subTest(minimum=minimum, maximum=maximum), patch.object(fit_audio, "_run_ffmpeg") as run:
                with self.assertRaises(ValueError):
                    fit_audio.fit_file(self.path, 1, maximum, minimum)
                run.assert_not_called()

    def test_overflow_never_cuts_or_publishes(self):
        before = self.path.read_bytes()
        with patch.object(fit_audio, "_run_ffmpeg") as run:
            with self.assertRaisesRegex(fit_audio.AudioFitError, "no words were cut"):
                fit_audio.fit_file(self.path, .1)
            run.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_actual_overflow_does_not_publish(self):
        def fake(command, timeout):
            write_pcm(Path(command[-1]), tone(.4))
        with patch.object(fit_audio, "_run_ffmpeg", side_effect=fake):
            with self.assertRaisesRegex(fit_audio.AudioFitError, "refusing to cut"):
                fit_audio.fit_file(self.path, .31)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_silence_and_nonfinite_rejected(self):
        write_pcm(self.path, [0.] * 100)
        with self.assertRaises(ValueError):
            fit_audio.fit_file(self.path, 1)
        raw = struct.pack("<f", float("nan"))
        fmt = struct.pack("<HHIIHH", 3, 1, 24000, 96000, 4, 32)
        self.path.write_bytes(b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVEfmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(raw)) + raw)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            fit_audio.fit_file(self.path, 1)

    def test_pcm24_decoder(self):
        raw = b"".join(v.to_bytes(3, "little", signed=True) for v in (-8388608, 0, 4194304))
        fmt = struct.pack("<HHIIHH", 1, 1, 24000, 72000, 3, 24)
        self.path.write_bytes(b"RIFF" + struct.pack("<I", 36 + len(raw)) + b"WAVEfmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(raw)) + raw + b"\0")
        samples, rate = fit_audio.read_mono_wav(self.path)
        self.assertEqual(list(samples), [-1., 0., .5])
        self.assertEqual(rate, 24000)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg unavailable")
    def test_short_clip_stays_native_and_cache_is_verified(self):
        source = self.path.read_bytes()
        result = fit_audio.fit_file(self.path, 1)
        meta = self.metadata(result)
        self.assertEqual(meta["tempo"], 1)
        self.assertEqual(meta["output_samples"], 7200)
        self.assertFalse(any("atempo" in f or "apad" in f for f in meta["filters"]))
        self.assertEqual(self.path.read_bytes(), source)
        with patch.object(fit_audio, "_run_ffmpeg") as run:
            self.assertEqual(fit_audio.fit_file(self.path, 1), result)
            run.assert_not_called()
        result.write_bytes(b"changed")
        with self.assertRaises(FileExistsError):
            fit_audio.fit_file(self.path, 1)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg unavailable")
    def test_quiet_initial_consonant_and_interior_pause_survive(self):
        values = [0.] * 7200 + [10 ** (-55 / 20)] + [0.] * 2399 + tone(.1) + [0.] * 2400 + tone(.1) + [0.] * 7200
        write_pcm(self.path, values)
        output = fit_audio.fit_file(self.path, 2)
        meta = self.metadata(output)
        self.assertLess(meta["edges"]["start_sample"], 7200)
        self.assertGreater(meta["edges"]["end_sample"], 16800)
        samples, _ = fit_audio.read_mono_wav(output)
        self.assertGreater(samples[7200 - meta["edges"]["start_sample"]], .001)
        self.assertEqual(meta["output_samples"], meta["edges"]["retained_samples"])

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg unavailable")
    def test_mild_acceleration_has_no_tail_trimming(self):
        write_pcm(self.path, tone(1.03))
        output = fit_audio.fit_file(self.path, 1)
        meta = self.metadata(output)
        self.assertGreater(meta["tempo"], 1)
        self.assertLessEqual(meta["tempo"], 1.15)
        self.assertLessEqual(meta["output_seconds"], 1)
        self.assertEqual(sum("atrim" in value for value in meta["filters"]), 1)
        self.assertTrue(meta["filters"][-1].startswith("atempo="))

    def test_legacy_wrapper_calls_only_safe_fitter(self):
        legacy = Path(__file__).resolve().parents[2] / "qwen3-tts"
        sys.path.insert(0, str(legacy))
        self.addCleanup(sys.path.remove, str(legacy))
        import dub_audio
        with patch.object(fit_audio, "fit_file", return_value=self.path) as safe:
            self.assertEqual(dub_audio.speed_fit(self.path, 1), self.path)
            safe.assert_called_once_with(self.path, 1, max_speed=1.15, min_speed=1.0)

    def test_legacy_gap_allowance_and_next_onset_deadline(self):
        legacy = Path(__file__).resolve().parents[2] / "qwen3-tts"
        sys.path.insert(0, str(legacy))
        self.addCleanup(sys.path.remove, str(legacy))
        import dub
        self.assertEqual(dub._fit_window(0, 1, 10), 1.25)
        self.assertAlmostEqual(dub._fit_window(0, 1, 1.1), 1.1)
        self.assertEqual(dub._fit_window(0, 1, .8), .8)
        self.assertEqual(dub._fit_window(0, 1), 1)
        self.assertEqual(dub._fit_window(0, .05, .08), .08)
        for values in ((0, 1, 0), (0, 0, 1), (0, 1, float("nan"))):
            with self.assertRaises(ValueError):
                dub._fit_window(*values)

    def test_legacy_fit_failure_prevents_partial_publication(self):
        legacy = Path(__file__).resolve().parents[2] / "qwen3-tts"
        sys.path.insert(0, str(legacy))
        self.addCleanup(sys.path.remove, str(legacy))
        import dub
        video = self.path.parent / "source.mp4"
        video.touch()
        srt = self.path.parent / "source.srt"
        srt.write_text("synthetic test")
        segments = [dict(index=i, speaker="SPEAKER_00", text="Hello", start=i*2, end=i*2+1)
                    for i in (1, 2)]
        manager = Mock()
        def generate(text, voice, language, output):
            output.write_bytes(b"0" * 600)
            return True
        manager.generate_custom.side_effect = generate
        fitted_count = 0
        def fit(path, *args, **kwargs):
            nonlocal fitted_count
            fitted_count += 1
            if fitted_count == 2:
                raise fit_audio.AudioFitError("Speech exceeds deadline")
            return path
        args = ["dub.py", str(video), str(srt), "--no-demucs", "--workdir", str(self.path.parent / "work")]
        with patch.object(sys, "argv", args), patch.multiple(dub,
                parse_srt=Mock(return_value=segments), merge_segments=Mock(side_effect=lambda s, **kw: s),
                _audio_duration=Mock(return_value=1), speed_fit=Mock(side_effect=fit),
                SharedTTSManager=Mock(return_value=manager),
                stitch_and_mix=Mock(), write_dub_srt=Mock()):
            self.assertEqual(dub.main(), 1)
            dub.stitch_and_mix.assert_not_called()
            dub.write_dub_srt.assert_not_called()
            deadlines = {call.args[0].stem: call.args[1] for call in dub.speed_fit.call_args_list}
            self.assertEqual(deadlines, {"seg_0001": 1.25, "seg_0002": 1})
        manager.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
