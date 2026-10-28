import unittest
from batch_dubbing.voice_fallback import resolve_borderline_voices


class VoiceFallbackTests(unittest.TestCase):
    def test_borderline_estimate_is_explicit_and_does_not_overwrite(self):
        source = {'s': {'voice': None, 'median_f0_hz': 160, 'voiced_samples': 36}}
        result = resolve_borderline_voices(source)
        self.assertEqual(result['s']['voice'], 'M1')
        self.assertEqual(result['s']['status'], 'borderline_pitch_estimate')
        self.assertIsNone(source['s']['voice'])

    def test_insufficient_pitch_does_not_invent_a_voice(self):
        with self.assertRaises(ValueError):
            resolve_borderline_voices({'s': {'voice': None, 'median_f0_hz': 140, 'voiced_samples': 3}})


if __name__ == '__main__':
    unittest.main()
