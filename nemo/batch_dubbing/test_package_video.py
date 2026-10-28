import unittest
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from batch_dubbing.package_video import srt, package
from batch_subtitles import srt_stats


class SubtitlePackagingTests(unittest.TestCase):
    def test_valid_cues_are_sorted_and_validated(self):
        data = srt([{'start': 2, 'end': 3, 'text': 'Zwei'},
                    {'start': 0, 'end': 1, 'text': 'Eins'}], 4)
        self.assertEqual(srt_stats(data, 4)['cues'], 2)
        self.assertLess(data.index(b'Eins'), data.index(b'Zwei'))

    def test_out_of_bounds_or_empty_cues_are_rejected(self):
        for cue in ({'start': 0, 'end': 5, 'text': 'x'},
                    {'start': 1, 'end': 1, 'text': 'x'},
                    {'start': 1, 'end': 2, 'text': ''}):
            with self.assertRaises(ValueError):
                srt([cue], 4)

    def test_packages_video_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / 'modal_control/batch_inputs'
            work.mkdir(parents=True)
            source = root / 'source.mp4'
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i', 'color=size=64x64:rate=10',
                            '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=44100', '-t', '1',
                            '-c:v', 'mpeg4', '-c:a', 'aac', str(source)], check=True)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                            'sine=frequency=550:sample_rate=44100', '-t', '1', '-c:a', 'aac',
                            str(work / 'video_00.cpu_full.m4a')], check=True)
            cue = {'start': 0.1, 'end': 0.8, 'text': 'Hallo'}
            (work / 'video_00.json').write_text(json.dumps({'source': str(source), 'source_sha256': before, 'duration_seconds': 1}))
            (work / 'video_00.gpu_full.json').write_text(json.dumps({'status': 'complete', 'duration_seconds': 1,
                    'english': [cue], 'translations': {'de': [cue], 'fr': [cue]}}))
            (work / 'video_00.cpu_full.json').write_text(json.dumps({'status': 'complete'}))
            with patch('batch_dubbing.package_video.ROOT', root):
                result = package(0)
                self.assertTrue(Path(result['video']).is_file())
                self.assertEqual(len(result['subtitles']), 3)
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
                with self.assertRaises(FileExistsError):
                    package(0)


if __name__ == '__main__':
    unittest.main()
