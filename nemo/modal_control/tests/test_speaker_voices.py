import unittest

from modal_control.speaker_voices import assign_voices


class SpeakerVoiceTests(unittest.TestCase):
    def test_assigns_distinct_stable_presets_and_flags_uncertain(self):
        result = assign_voices({"speaker_a": [110.0] * 12,
                                "speaker_b": [220.0] * 12,
                                "speaker_c": [205.0] * 12,
                                "speaker_d": [165.0] * 12,
                                "speaker_e": [90.0] * 2})
        self.assertEqual([result[s]["voice"] for s in ("speaker_a", "speaker_b", "speaker_c")], ["M1", "F1", "F2"])
        self.assertEqual(result["speaker_d"]["status"], "needs_review")
        self.assertIsNone(result["speaker_e"]["voice"])

    def test_ignores_nonfinite_and_out_of_range_values(self):
        result = assign_voices({"speaker": [float('nan'), float('inf'), 20, 500] + [210] * 10})
        self.assertEqual(result["speaker"]["voice"], "F1")
        self.assertEqual(result["speaker"]["voiced_samples"], 10)

    def test_invalid_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            assign_voices({}, low_hz=180, high_hz=150)


if __name__ == "__main__":
    unittest.main()
