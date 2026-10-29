"""Synthetic float sample fixtures only; never import a model or cloud worker."""
from copy import deepcopy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

from batch_dubbing import native_excerpt_probe as probe


def fixture():
    rows = []
    for i, (language, start, end) in enumerate((("en", .25, 1.), ("en", 2., 3.),
                                               ("de", .3, 1.5), ("de", 2.2, 3.7))):
        rows.append({"id": f"{language}_{i}", "language": language, "voice": "F1",
                     "speaker": "comparison_voice_F1", "seed": 41 + i,
                     "srt_start": start, "srt_end": end,
                     "text": "A complete sentence." if language == "en" else "Ein vollständiger Satz.",
                     "source_indices": [i]})
    return {"schema": 1, "model": probe.MODEL, "revision": probe.REVISION,
            "package": probe.PACKAGE, "image_id": probe.IMAGE_ID,
            "speed": 1., "steps": 8, "max_chunk_length": 300, "silence_duration": .3,
            "excerpt_start": 0., "excerpt_end": 4.,
            "source_metadata": {"kind": "synthetic_caption_fixture", "speaker_identity_verified": False},
            "samples": rows}


def chunker(text, maximum):
    return [text]


def measured(row, exact=False):
    count = (round(row["srt_end"] * 44100) - round(row["srt_start"] * 44100)) if exact else 10
    # Signed zeros and a nonzero final sample expose destructive copying.
    payload = struct.pack('<5f', .25, -.5, -0., 0., .75) * (count // 5)
    payload += struct.pack('<f', .75) * (count % 5)
    return {"samples": payload, "sample_rate": 44100}


def smoke_artifact(data, callback=measured, use_chunker=chunker):
    report = probe.run_excerpt(data, True, chunker=use_chunker, synthesize=callback)
    wav = report.pop('audio')[data['samples'][0]['id'] + '.wav']
    encoded = json.dumps(report).encode()
    return {'report': encoded, 'report_sha256': probe.digest(encoded), 'wav': wav}


class NativeExcerptTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()

    def run_work(self, callback=measured, use_chunker=chunker, prior=None):
        return probe.run_excerpt(self.data, False, chunker=use_chunker, synthesize=callback,
                                 smoke_artifact=prior or smoke_artifact(self.data, use_chunker=use_chunker))

    def test_scope_is_generic_and_preserves_provenance_and_input(self):
        original = deepcopy(self.data)
        selected = probe.validate_input(self.data, chunker=chunker)
        selected['samples'][0]['text'] = 'Changed'
        self.assertEqual(self.data, original)
        self.data['source_metadata']['arbitrary_source'] = 'different-video.json'
        self.assertEqual(probe.validate_scope(self.data), self.data)

    def test_scope_rejects_changes_nonfinite_invalid_seed_and_bad_slots(self):
        for field, value in [('model', 'other'), ('speed', 1.05), ('steps', 9), ('silence_duration', 0),
                             ('schema', True), ('excerpt_end', float('nan')), ('excerpt_end', 121),
                             ('excerpt_start', -1), ('source_metadata', {})]:
            data = {**self.data, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                probe.validate_scope(data)
        for field, value in [('id', '../escape'), ('id', 'native_en'), ('voice', 'M6'), ('seed', True),
                             ('seed', 2**32), ('speaker', ''), ('text', ''), ('language', 'fr'),
                             ('srt_end', 5), ('srt_start', -.1), ('source_indices', [True]),
                             ('source_indices', [0, 0])]:
            data = deepcopy(self.data)
            data['samples'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                probe.validate_scope(data)

    def test_order_duplicates_overlap_and_source_word_reuse_rejected(self):
        mutations = [lambda d: d['samples'][1].update(srt_start=.9),
                     lambda d: d['samples'][1].update(id=d['samples'][0]['id']),
                     lambda d: d['samples'][1].update(source_indices=[0]),
                     lambda d: d['samples'].reverse(),
                     lambda d: d.update(samples=d['samples'] * 5)]
        for mutation in mutations:
            data = deepcopy(self.data)
            mutation(data)
            with self.assertRaises(ValueError):
                probe.validate_scope(data)

    def test_actual_chunk_count_and_exact_text_retention_checked_before_callbacks(self):
        call = Mock()
        for bad_chunker in (None, lambda text, maximum: [], lambda text, maximum: ['Dropped text']):
            with self.assertRaises(ValueError):
                probe.run_excerpt(self.data, True, chunker=bad_chunker, synthesize=call)
        data = deepcopy(self.data)
        for row in data['samples']:
            row['text'] = ' '.join(['word'] * 7)
        with self.assertRaisesRegex(ValueError, '24'):
            probe.run_excerpt(data, True, chunker=lambda text, maximum: text.split(' '), synthesize=call)
        call.assert_not_called()

    def test_first_smoke_consumed_once_and_generic_ids_used(self):
        call = Mock(side_effect=measured)
        prior = smoke_artifact(self.data, call)
        result = self.run_work(call, prior=prior)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(call.call_count, 4)
        self.assertEqual([args.args[0]['id'] for args in call.call_args_list], [r['id'] for r in self.data['samples']])
        self.assertEqual(result['calls'], 3)
        self.assertEqual(result['internal_calls'], 3)
        self.assertEqual(result['internal_calls_evidence'],
                         'expected_chunks_for_dispatched_api_calls_not_observed_completions')
        self.assertTrue(result['samples'][0]['reused_from_smoke'])
        self.assertEqual(result['audio']['en_0.wav'], prior['wav'])

    def test_reference_lead_gaps_tail_preserved_without_forcing_duration(self):
        result = self.run_work()
        self.assertEqual(result['strict_caption_placement']['status'], 'caption_slot_mismatch')
        self.assertFalse(any(name.startswith('aligned_') for name in result['audio']))
        self.assertEqual(result['aligned_timeline_files'], {})
        for sequence in result['sequences']:
            payload = result['audio'][sequence['file']][56:]
            source = [row for row in self.data['samples'] if row['language'] == sequence['language']]
            gaps = [round(source[0]['srt_start'] * 44100),
                    round(source[1]['srt_start'] * 44100) - round(source[0]['srt_end'] * 44100),
                    round(4 * 44100) - round(source[1]['srt_end'] * 44100)]
            self.assertEqual([r['end_frame'] - r['start_frame'] for r in sequence['silence']], gaps)
            self.assertEqual(sequence['total_frames'], sum(gaps) + 20)
            self.assertNotEqual(sequence['duration_seconds'], 4.)
            for row in sequence['placements']:
                expected = measured(next(item for item in source if item['id'] == row['id']))['samples']
                self.assertEqual(payload[row['start_frame'] * 4:row['end_frame'] * 4], expected)
                self.assertEqual(row['source_samples_sha256'], row['sequence_slice_sha256'])
            for row in sequence['silence']:
                self.assertEqual(payload[row['start_frame'] * 4:row['end_frame'] * 4], b'\0' * ((row['end_frame']-row['start_frame']) * 4))
        for flag in ('timing_verified', 'naturalness_verified', 'pronunciation_verified', 'duration_conditioning', 'pause_conditioning'):
            self.assertFalse(result[flag])

    def test_exact_slots_allow_caption_timeline_but_no_acoustic_claim(self):
        exact = lambda row: measured(row, exact=True)
        result = self.run_work(exact, prior=smoke_artifact(self.data, exact))
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['strict_caption_placement']['all_slots_match_exactly'])
        self.assertEqual(result['aligned_timeline_files'], {'en': 'native_en.wav', 'de': 'native_de.wav'})
        self.assertEqual(len(result['audio']), 6)
        self.assertFalse(any(name.startswith('aligned_') for name in result['audio']))
        self.assertEqual(len(result['sequences']), 2)
        self.assertFalse(result['timing_verified'])
        self.assertTrue(all(seq['total_frames'] == 4 * 44100 for seq in result['sequences']))

    def test_exact_fit_aliases_do_not_duplicate_transfer_bytes(self):
        exact = lambda row: measured(row, exact=True)
        prior = smoke_artifact(self.data, exact)
        # Enough for raw clips plus two sequences, but not duplicate timelines.
        limit = sum(len(probe.encode_wave(exact(row)['samples'])) for row in self.data['samples'])
        limit += 2 * (4 * 44100 * 4 + 56)
        with patch.object(probe, 'MAX_AUDIO_BYTES', limit):
            result = self.run_work(exact, prior=prior)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['audio_transfer_bytes'], limit)
        self.assertEqual(set(result['aligned_timeline_files']), {'en', 'de'})

    def test_equal_global_duration_does_not_hide_underfill_overfill(self):
        rows = [{**row, 'native_frames': round((row['srt_end'] - row['srt_start']) * 44100)} for row in self.data['samples']]
        rows[0]['native_frames'] += 1
        rows[1]['native_frames'] -= 1
        report = probe.strict_caption_placement(self.data, rows)
        self.assertFalse(report['all_slots_match_exactly'])
        self.assertEqual(report['rows'][0]['overfill_frames'], 1)
        self.assertEqual(report['rows'][1]['underfill_frames'], 1)
        rows[0]['native_frames'] = 3 * 44100
        self.assertGreater(probe.strict_caption_placement(self.data, rows)['collisions'][0]['overlap_frames'], 0)

    def test_tampered_smoke_or_changed_scope_fails_before_synthesis(self):
        prior = smoke_artifact(self.data)
        call = Mock()
        for invalid in ({**prior, 'wav': prior['wav'][:-4]}, {**prior, 'report_sha256': 'changed'}, None):
            with self.assertRaises(ValueError):
                probe.run_excerpt(self.data, False, chunker=chunker, synthesize=call, smoke_artifact=invalid)
        self.data['samples'][0]['text'] = 'Changed complete text.'
        with self.assertRaises(ValueError):
            probe.run_excerpt(self.data, False, chunker=chunker, synthesize=call, smoke_artifact=prior)
        call.assert_not_called()

    def test_chunked_smoke_count_verified_against_actual_chunker(self):
        data = self.data
        for row in data['samples']:
            row['text'] = 'One. Two.'
        chunks = lambda text, maximum: ['One.', 'Two.']
        prior = smoke_artifact(data, use_chunker=chunks)
        result = self.run_work(use_chunker=chunks, prior=prior)
        self.assertEqual(result['internal_calls'], 6)
        with self.assertRaisesRegex(ValueError, 'internal call count'):
            self.run_work(prior=prior)

    def test_callback_failure_retains_previous_outputs_and_stops(self):
        call = Mock(side_effect=[measured(self.data['samples'][1]), RuntimeError('stop'), measured(self.data['samples'][3])])
        result = self.run_work(call)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(call.call_count, 2)
        self.assertEqual(set(result['audio']), {'en_0.wav', 'en_1.wav'})
        self.assertEqual(result['sequences'], [])
        self.assertIn('not_observed_completions', result['internal_calls_evidence'])

    def test_invalid_audio_outputs_are_rejected(self):
        for output in (None, {}, {'samples': b'123', 'sample_rate': 44100},
                       {'samples': struct.pack('<f', float('nan')), 'sample_rate': 44100},
                       {'samples': struct.pack('<f', 0), 'sample_rate': 44100},
                       {'samples': struct.pack('<f', .5), 'sample_rate': True}):
            call = Mock(return_value=output)
            result = self.run_work(call)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(call.call_count, 1)
        with self.assertRaisesRegex(ValueError, '120 seconds'):
            probe.validate_samples(b'\0' * (120 * 44100 * 4 + 4), 44100)

    def test_stage300_second_bound_and_combined_transfer_bound(self):
        prior = smoke_artifact(self.data)
        original = probe.validate_samples
        calls = 0
        def large_counts(payload, rate):
            nonlocal calls
            calls += 1
            return original(payload, rate) if calls == 1 else (120 * 44100, .5)
        with patch.object(probe, 'validate_samples', side_effect=large_counts):
            result = self.run_work(prior=prior)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('300 seconds', result['error'])
        self.assertEqual(result['calls'], 3)
        with patch.object(probe, 'MAX_AUDIO_BYTES', 500):
            result = self.run_work(prior=prior)
        self.assertEqual(result['status'], 'failed')
        self.assertLessEqual(result['audio_transfer_bytes'], 500)
        self.assertEqual(len(result['samples']), 4)
        self.assertEqual(result['sequences'], [])

    def test_load_smoke_uses_first_generic_id_and_preserves_raw_bytes(self):
        prior = smoke_artifact(self.data)
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / 'report.json'
            report.write_bytes(prior['report'])
            (report.parent / 'en_0.wav').write_bytes(prior['wav'])
            self.assertEqual(probe.load_smoke(report, self.data), prior)
            (report.parent / 'en_0.wav').write_bytes(prior['wav'] + b'bad')
            with self.assertRaises(ValueError):
                probe.load_smoke(report, self.data)


if __name__ == '__main__':
    unittest.main()
