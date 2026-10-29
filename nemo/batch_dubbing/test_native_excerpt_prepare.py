import unittest
from copy import deepcopy
from batch_dubbing.native_excerpt_prepare import group_caption_blocks, build_input


class CaptionBlockPreparationTests(unittest.TestCase):
    def test_long_gap_requires_complete_boundary_and_retains_all_words(self):
        cues = [dict(start=1, end=2, text="When we began"),
                dict(start=4, end=5, text="it worked."),
                dict(start=5.2, end=6, text="We continued."),
                dict(start=8, end=9, text="The next block.")]
        before = deepcopy(cues)
        result = group_caption_blocks(cues, start=0, end=10)
        self.assertEqual([r['text'] for r in result],
                         ['When we began it worked. We continued.', 'The next block.'])
        self.assertEqual([r['source_indices'] for r in result], [[0, 1, 2], [3]])
        self.assertEqual(cues, before)

    def test_partial_edges_and_unfinished_final_caption_block(self):
        for cues, start, end in [([dict(start=1, end=3, text='A sentence.')], 2, 4),
                                 ([dict(start=1, end=3, text='A sentence.')], 0, 2),
                                 ([dict(start=1, end=3, text='An unfinished')], 0, 4)]:
            with self.subTest(cues=cues, start=start, end=end), self.assertRaises(ValueError):
                group_caption_blocks(cues, start=start, end=end)

    def test_overlapping_caption_records_rejected(self):
        with self.assertRaises(ValueError):
            group_caption_blocks([dict(start=1, end=3, text='One.'),
                                  dict(start=2, end=4, text='Two.')], start=0, end=5)

    def test_preset_does_not_upgrade_unknown_source_speaker(self):
        cue = dict(start=1, end=2, text='A whole sentence.', speaker=None)
        payload = dict(english=[cue], translations=dict(de=[{**cue, 'text': 'Ein ganzer Satz.'}]))
        before = deepcopy(payload)
        result = build_input(payload, start=0, end=3, voice='F1', source_metadata={'kind': 'captions'})
        self.assertEqual([r['id'] for r in result['samples']], ['en_000', 'de_000'])
        self.assertTrue(all(r['speaker'] == 'comparison_voice_F1' for r in result['samples']))
        self.assertEqual(payload, before)


if __name__ == '__main__':
    unittest.main()
