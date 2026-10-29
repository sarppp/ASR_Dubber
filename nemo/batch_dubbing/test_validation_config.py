import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import wave

from batch_dubbing.validation_config import load_config, selected_cases, validate_config

ROOT = Path(__file__).resolve().parents[1]


class ValidationConfigTests(unittest.TestCase):
    def scope(self):
        return json.loads((ROOT / 'dubbing_naturalness_2026-10-28/source_validation_inputs.json').read_text())

    def test_frozen_scope_has_three_general_cases_and_bounded_calls(self):
        config = validate_config(self.scope())
        self.assertEqual(sum(c['duration_seconds'] for c in config['cases']), 85)
        self.assertEqual(len(selected_cases(config, smoke=True)), 1)
        self.assertEqual(selected_cases(config, smoke=True)[0]['id'], 'crispr')
        self.assertEqual(1 + len(config['cases']) * config['max_units_per_case'], 37)
        self.assertEqual(config['target_language'], 'de')

    def test_scope_rejects_expansion_invalid_inputs_and_ambiguous_ids(self):
        base = self.scope()
        changes = [lambda c: c.update(target_language='fr'),
                   lambda c: c.update(max_units_per_case=13),
                   lambda c: c.update(max_units_per_case=True),
                   lambda c: c.update(smoke_case_id='absent'),
                   lambda c: c['cases'].append(copy.deepcopy(c['cases'][0])),
                   lambda c: c['cases'][0].update(input_file='../secret.wav'),
                   lambda c: c['cases'][0].update(input_file='/tmp/secret.wav'),
                   lambda c: c['cases'][0].update(source_sha256='bad'),
                   lambda c: c['cases'][0].update(duration_seconds=31),
                   lambda c: c['cases'][0].update(offset_seconds=float('nan'))]
        for change in changes:
            config = copy.deepcopy(base)
            change(config)
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_config(config)

    def test_local_validation_hashes_existing_wav_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wav = root / 'source.wav'
            with wave.open(str(wav), 'wb') as output:
                output.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                output.writeframes(b'\0\0' * 16000)
            config = {'version': 1, 'target_language': 'de', 'smoke_case_id': 'fixture',
                      'max_units_per_case': 12, 'cases': [{'id': 'fixture', 'input_file': 'source.wav',
                      'source_sha256': hashlib.sha256(wav.read_bytes()).hexdigest(),
                      'source_duration_seconds': 1, 'offset_seconds': 0, 'duration_seconds': 1}]}
            path = root / 'scope.json'
            path.write_text(json.dumps(config))
            self.assertEqual(load_config(path, root=root), config)
            content = bytearray(wav.read_bytes())
            content[-1] = 1
            wav.write_bytes(content)
            with self.assertRaisesRegex(ValueError, 'hash changed'):
                load_config(path, root=root)

    def test_returned_scope_cannot_mutate_original(self):
        original = self.scope()
        result = selected_cases(original)
        result[0]['duration_seconds'] = 99
        self.assertEqual(original['cases'][0]['duration_seconds'], 25)


if __name__ == '__main__':
    unittest.main()
