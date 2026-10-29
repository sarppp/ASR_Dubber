"""Synthetic fixture and sample-coordinate tests; no model or media execution."""
from copy import deepcopy
import random
import unittest

from batch_dubbing.sentence_transition_probe import (
    build_fixture, validate_fixture, plan_jobs, plan_sequence, check_schedule,
    MAX_INTERNAL_CALLS, MAX_BYTES,
)


def fixture_chunker(text, maximum):
    """Deterministic stub for these fixture texts, not a provider implementation."""
    assert maximum == 300
    if len(text) <= maximum:
        return [text]
    prefix, suffix = text.rsplit('. ', 1)
    return [prefix + '.', suffix]


def clips(frames=(100, 1000, 100)):
    return [{'id': f'clip_{i}', 'native_frames': count} for i, count in enumerate(frames)]


class SentenceTransitionProbeTests(unittest.TestCase):
    def test_fixture_contains_short_long_short_in_both_languages(self):
        data = build_fixture()
        self.assertEqual(data['language_order'], ['en', 'de'])
        self.assertEqual([row['language'] for row in data['samples']], ['en'] * 3 + ['de'] * 3)
        for language in data['language_order']:
            rows = [row for row in data['samples'] if row['language'] == language]
            self.assertEqual([row['role'] for row in rows], ['short', 'long', 'short'])
            self.assertEqual([len(row['text'].split()) for row in rows], [6, 73, 6])
            self.assertGreater(len(rows[1]['text']), 300)
            self.assertLessEqual(len(rows[1]['text']), 1000)
            self.assertEqual([row['expected_chunks'] for row in rows], [1, 2, 1])
        self.assertEqual(data['speed'], 1.)
        self.assertEqual(data['silence_duration'], .3)
        self.assertEqual(data['gap_seconds'], .4)
        self.assertEqual(data['evidence_kind'], 'synthetic_fixture')
        self.assertEqual(data['max_stage_bytes'], 64 * 1024**2)

    def test_six_api_eight_internal_calls_partitioned_without_duplicate_smoke(self):
        data = build_fixture()
        smoke = plan_jobs(data, True, chunker=fixture_chunker)
        work = plan_jobs(data, False, chunker=fixture_chunker)
        self.assertEqual((len(smoke), len(work)), (1, 5))
        combined = smoke + work
        self.assertEqual([row['id'] for row in combined], [row['id'] for row in data['samples']])
        self.assertEqual(sum(row['expected_chunks'] for row in combined), MAX_INTERNAL_CALLS)
        self.assertEqual(sum(row['expected_chunks'] for row in work), 7)
        self.assertEqual(MAX_BYTES, 64 * 1024**2)

    def test_preparation_preserves_input_and_returns_new_data(self):
        data = build_fixture()
        before = deepcopy(data)
        output = validate_fixture(data, chunker=fixture_chunker)
        output['samples'][0]['text'] = 'Changed'
        self.assertEqual(data, before)
        self.assertEqual(build_fixture(), before)

    def test_changed_chunker_or_lost_text_fails_before_any_synthesis(self):
        for chunker in (lambda text, limit: [text],
                        lambda text, limit: [text[:5], text[5:]],
                        lambda text, limit: [],
                        lambda text, limit: ['Changed.'],
                        None):
            with self.subTest(chunker=chunker), self.assertRaises(ValueError):
                validate_fixture(build_fixture(), chunker=chunker)

    def test_frozen_configuration_and_stage_type_cannot_change(self):
        for name, value in (('speed', 1.05), ('steps', 12), ('gap_seconds', .3),
                            ('schema', True), ('image_id', 'other'), ('max_internal_calls', 9)):
            data = build_fixture()
            data[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_fixture(data, chunker=fixture_chunker)
        data = build_fixture()
        data['samples'][1]['text'] += ' Extra words.'
        with self.assertRaises(ValueError):
            validate_fixture(data, chunker=fixture_chunker)
        for smoke in (0, 1, None):
            with self.assertRaises(ValueError):
                plan_jobs(build_fixture(), smoke, chunker=fixture_chunker)

    def test_short_long_short_keeps_every_native_frame_and_gap(self):
        source = clips()
        before = deepcopy(source)
        result = plan_sequence(source)
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(source, before)
        self.assertEqual(result['gap_frames'], 17640)
        self.assertEqual([row['start_frame'] for row in result['schedule']], [0, 17740, 36380])
        self.assertEqual([row['end_frame'] - row['start_frame'] for row in result['schedule']], [100, 1000, 100])
        self.assertEqual(result['total_frames'], 36480)
        self.assertTrue(result['sample_count_retention_verified'])
        self.assertFalse(result['sample_content_retention_verified'])
        self.assertFalse(result['pronunciation_verified'])
        self.assertFalse(result['naturalness_verified'])

    def test_short_slot_blocks_instead_of_cutting_ending(self):
        result = plan_sequence(clips(), slot_ends=(100, 18000, 37000))
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['schedule'], [])
        self.assertEqual(result['failures'][0], {'id': 'clip_1', 'reason': 'slot_too_short',
                         'required_end_frame': 18740, 'slot_end_frame': 18000, 'deficit_frames': 740})

    def test_next_short_clip_cannot_overwrite_long_clip_or_synthetic_pause(self):
        for starts in ((0, 17740, 18000), (0, 17740, 18740), (0, 17740, 36379)):
            with self.subTest(starts=starts):
                result = plan_sequence(clips(), starts=starts)
                self.assertEqual(result['status'], 'blocked')
                self.assertEqual(result['schedule'], [])
                self.assertIn('collision_or_missing_fixture_gap', [row['reason'] for row in result['failures']])

    def test_one_frame_ending_loss_is_detected(self):
        schedule = plan_sequence(clips())['schedule']
        schedule[1]['end_frame'] -= 1
        result = check_schedule(schedule)
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['schedule'], [])
        self.assertIn({'id': 'clip_1', 'reason': 'sample_count_changed'}, result['failures'])

    def test_timeline_end_cannot_clip_final_short_sentence(self):
        plan = plan_sequence(clips())
        result = check_schedule(plan['schedule'], total_frames=plan['total_frames'] - 1)
        self.assertEqual(result['status'], 'blocked')
        self.assertIn({'id': 'clip_2', 'reason': 'outside_timeline'}, result['failures'])
        self.assertEqual(check_schedule(plan['schedule'], total_frames=plan['total_frames'])['status'], 'ready')

    def test_native_duration_and_integer_coordinate_bounds(self):
        for rows in ([], clips((0,)), clips((True,)), clips((90 * 44100 + 1,)),
                     clips((90 * 44100,) * 4), [{'id': 'x', 'native_frames': 1}] * 2):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                plan_sequence(rows)
        for changes in ({'sample_rate': 24000}, {'gap_seconds': float('inf')},
                        {'gap_seconds': True}, {'gap_seconds': .0000001},
                        {'starts': (0, 1)}, {'slot_ends': (1,)}, {'starts': (0, -1, 2)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                plan_sequence(clips(), **changes)
        for rate in (1.0001, .99, True):
            rows = clips()
            rows[0]['playback_rate'] = rate
            with self.assertRaises(ValueError):
                plan_sequence(rows)

    def test_random_native_lengths_never_lose_samples_or_compress_fixture_gaps(self):
        randomizer = random.Random(20261029)
        for _ in range(200):
            lengths = tuple(randomizer.randrange(1, 40 * 44100) for _ in range(3))
            result = plan_sequence(clips(lengths))
            self.assertEqual(result['status'], 'ready')
            self.assertEqual(result['total_frames'], sum(lengths) + 2 * 17640)
            for index, row in enumerate(result['schedule']):
                self.assertEqual(row['end_frame'] - row['start_frame'], lengths[index])
                if index:
                    self.assertEqual(row['start_frame'] - result['schedule'][index - 1]['end_frame'], 17640)


if __name__ == '__main__':
    unittest.main()
