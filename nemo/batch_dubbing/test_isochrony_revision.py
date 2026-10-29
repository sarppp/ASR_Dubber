"""Pure synthetic tests for wording feedback; no provider or media execution."""
import copy
import json
import unittest

from batch_dubbing.isochrony import ActivityTimeline, SpeechInterval
from batch_dubbing.isochrony_revision import build_revision_request
from batch_dubbing.isochrony_backend import SynthesisRequest


def timeline(spans=((1, 3), (4, 6)), *, duration=8, provenance='synthetic', evidence='fixture'):
    return ActivityTimeline(duration, tuple(SpeechInterval(a, b) for a, b in spans), provenance, .01, evidence)


def build(observed=None, target=None, **changes):
    args = dict(source_text='In 2026, Alex did not buy 42 items.',
                translated_text='Im Jahr 2026 kaufte Alex nicht 42 Artikel.',
                source_language='en', target_language='de',
                target=target if target is not None else timeline(),
                observed=observed if observed is not None else timeline(),
                protected_terms=('2026', 'Alex', '42'))
    return build_revision_request(**{**args, **changes})


class IsochronyRevisionTests(unittest.TestCase):
    def test_long_speech_requests_shorter_meaning_preserving_wording(self):
        result = build(timeline(((1, 3.5), (4, 6.5))))
        self.assertEqual(result['status'], 'revision_required')
        self.assertEqual(result['action'], 'shorter_equivalent_wording')
        self.assertEqual(result['speech_budgets'][0]['target_speech_seconds'], 2)
        self.assertEqual(result['speech_budgets'][0]['observed_speech_seconds'], 2.5)
        self.assertEqual(result['protected_terms'], ['2026', 'Alex', '42'])
        self.assertIn('cannot_fit_faithfully', result['response_contract']['allowed_statuses'])

    def test_short_speech_permits_fuller_equivalent_wording_but_not_filler(self):
        result = build(timeline(((1, 2.5), (4, 5.5))))
        self.assertEqual(result['action'], 'longer_equivalent_wording')
        instructions = ' '.join(result['instructions'])
        self.assertIn('do not drop information', instructions)
        self.assertIn('Do not add facts, filler', instructions)
        self.assertIn('negation', instructions)
        self.assertIn('communicative intent', instructions)

    def test_equal_total_duration_with_wrong_pauses_is_not_accepted(self):
        result = build(timeline(((1, 3), (4.5, 6.5))))
        self.assertEqual(result['target']['duration_seconds'], result['observed']['duration_seconds'])
        self.assertEqual(result['target_speech_seconds'], result['observed_speech_seconds'])
        self.assertEqual(result['status'], 'revision_required')
        self.assertEqual(result['action'], 'keep_wording_repair_pause_placement')
        self.assertEqual(result['speech_budgets'][0]['suggestion'], 'no_change')
        self.assertEqual(result['speech_budgets'][1]['suggestion'], 'keep_wording_repair_pause_placement')
        self.assertLess(result['comparison']['metrics']['pause_iou'], 1)

    def test_mixed_phrase_lengths_are_rebalanced_even_when_totals_match(self):
        result = build(timeline(((1, 3.5), (4, 5.5))))
        self.assertEqual(result['action'], 'rebalance_phrase_lengths')
        self.assertEqual([row['suggestion'] for row in result['speech_budgets']],
                         ['shorter_equivalent_wording', 'longer_equivalent_wording'])
        self.assertEqual(result['target_speech_seconds'], result['observed_speech_seconds'])

    def test_missing_blocks_never_receive_invented_correspondence(self):
        result = build(timeline(((1, 6),)))
        self.assertEqual(result['action'], 'repair_speech_and_pause_structure')
        self.assertIsNone(result['speech_budgets'])
        self.assertEqual(result['comparison']['correspondence'], 'unmatched_block_counts')
        self.assertEqual(result['target']['speech'][1]['start_seconds'], 4)

    def test_no_observed_speech_is_structure_failure_not_a_license_for_filler(self):
        result = build(timeline(()))
        self.assertEqual(result['status'], 'revision_required')
        self.assertEqual(result['action'], 'repair_speech_and_pause_structure')
        self.assertIsNone(result['speech_budgets'])
        self.assertFalse(result['comparison']['contains_candidate_speech'])

    def test_matching_measurements_request_no_revision_and_no_quality_claim(self):
        for provenance in ('synthetic', 'acoustic_vad', 'manual_review'):
            with self.subTest(provenance=provenance):
                activity = timeline(provenance=provenance)
                result = build(activity, activity)
                self.assertEqual(result['status'], 'no_revision_required')
                self.assertEqual(result['action'], 'none')
                self.assertFalse(result['revision_requested'])
                self.assertEqual(result['instructions'], [])
                self.assertIn('no candidate was accepted', result['quality_limit'])

    def test_nonacoustic_evidence_cannot_drive_wording_revision(self):
        for provenance in ('asr_words', 'subtitle', 'energy_threshold'):
            with self.subTest(provenance=provenance):
                result = build(timeline(((1, 2),), provenance=provenance))
                self.assertEqual(result['status'], 'needs_activity_evidence')
                self.assertFalse(result['revision_requested'])
                self.assertEqual(result['instructions'], [])

    def test_full_context_evidence_and_timestamps_are_preserved(self):
        target = timeline(evidence='original-evidence')
        observed = timeline(((1, 3.1), (4, 6)), evidence='candidate-evidence')
        before = copy.deepcopy((target.to_dict(), observed.to_dict()))
        result = build(observed, target)
        self.assertEqual((target.to_dict(), observed.to_dict()), before)
        self.assertEqual(result['target'], before[0])
        self.assertEqual(result['observed'], before[1])
        self.assertEqual(result['source_text'], 'In 2026, Alex did not buy 42 items.')
        self.assertEqual(result['translated_text'], 'Im Jahr 2026 kaufte Alex nicht 42 Artikel.')
        self.assertTrue(result['constraints']['target_timestamps_immutable'])
        self.assertEqual(result['constraints']['playback_rate'], 1)
        self.assertEqual(result['constraints']['speaking_rate_scale'], 1)
        self.assertFalse(result['constraints']['waveform_time_stretching_allowed'])
        json.dumps(result, allow_nan=False)

    def test_input_text_language_terms_and_types_are_bounded(self):
        cases = ({'source_text': ''}, {'translated_text': None}, {'source_text': 'x' * 10001},
                 {'source_language': 'English'}, {'target_language': 'de-' + 'x' * 100},
                 {'protected_terms': ['Alex']}, {'protected_terms': ('Alex', 'Alex')},
                 {'protected_terms': (None,)}, {'protected_terms': ('',)},
                 {'protected_terms': ('x' * 257,)},
                 {'protected_terms': tuple(str(i) for i in range(129))},
                 {'target': {}}, {'observed': {}}, {'target': timeline(())})
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build(**changes)

    def test_no_language_specific_rewrite_rules(self):
        result = build(timeline(((1, 2), (4, 5))), source_language='fr', target_language='en',
                       source_text='Une phrase complète.', translated_text='A complete sentence.', protected_terms=())
        self.assertEqual(result['source_language'], 'fr')
        self.assertEqual(result['translated_text'], 'A complete sentence.')
        self.assertEqual(result['action'], 'longer_equivalent_wording')

    def test_backend_large_integer_is_validation_error_not_overflow(self):
        for field in ('target_duration_seconds', 'playback_rate', 'speaking_rate_scale'):
            args = dict(source_id='source', text='Text.', language='en', target_duration_seconds=1)
            args[field] = 10**1000
            with self.subTest(field=field), self.assertRaises(ValueError):
                SynthesisRequest(**args)


if __name__ == '__main__':
    unittest.main()
