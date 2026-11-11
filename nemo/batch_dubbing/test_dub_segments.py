import unittest
from modal_control.dub_segments import merge_speech_cues


class CueMergeTests(unittest.TestCase):
    def test_same_speaker_is_merged_but_speaker_change_is_preserved(self):
        cues = [{'start': 0, 'end': 1, 'text': 'Erster', 'speaker': 'a'},
                {'start': 1, 'end': 1.5, 'text': 'Satz.', 'speaker': 'a'},
                {'start': 1.6, 'end': 3, 'text': 'Antwort.', 'speaker': 'b'}]
        result = merge_speech_cues(cues)
        self.assertEqual([x['text'] for x in result], ['Erster Satz.', 'Antwort.'])
        self.assertEqual([x['speaker'] for x in result], ['a', 'b'])

    def test_does_not_guess_on_overlap_or_missing_speaker(self):
        with self.assertRaises(ValueError):
            merge_speech_cues([{'start': 0, 'end': 1, 'text': 'x', 'speaker': None}])
        with self.assertRaises(ValueError):
            merge_speech_cues([{'start': 0, 'end': 2, 'text': 'x', 'speaker': 'a'},
                               {'start': 1, 'end': 3, 'text': 'y', 'speaker': 'b'}])


if __name__ == '__main__':
    unittest.main()
