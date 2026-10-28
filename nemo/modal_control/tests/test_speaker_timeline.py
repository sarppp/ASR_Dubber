import unittest

from modal_control.speaker_timeline import assign_cue_speakers


class SpeakerTimelineTests(unittest.TestCase):
    def test_follows_speaker_turns(self):
        turns = [{'start': 0, 'end': 2, 'speaker': 'a'},
                 {'start': 2, 'end': 4, 'speaker': 'b'}]
        cues = [{'start': 0, 'end': 1, 'text': 'eins'},
                {'start': 2, 'end': 3, 'text': 'zwei'}]
        assigned = assign_cue_speakers(cues, turns)
        self.assertEqual([x['speaker'] for x in assigned], ['a', 'b'])
        self.assertNotIn('speaker', cues[0])

    def test_overlapping_voices_are_not_guessed(self):
        cues = [{'start': 0, 'end': 2, 'text': 'Hallo'}]
        turns = [{'start': 0, 'end': 2, 'speaker': 'a'},
                 {'start': 0, 'end': 2, 'speaker': 'b'}]
        cue = assign_cue_speakers(cues, turns)[0]
        self.assertIsNone(cue['speaker'])
        self.assertEqual(cue['speaker_status'], 'needs_review')

    def test_insufficient_coverage_and_invalid_cues(self):
        self.assertIsNone(assign_cue_speakers([{'start': 0, 'end': 3}],
                          [{'start': 0, 'end': 1, 'speaker': 'a'}])[0]['speaker'])
        with self.assertRaises(ValueError):
            assign_cue_speakers([{'start': 1, 'end': 1}], [])


if __name__ == '__main__':
    unittest.main()
