"""Pure and mocked speech-stage tests: no model, container, or media processing."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, MagicMock

from batch_dubbing import speech_stage as stage

ROOT = Path(__file__).resolve().parents[1]
WORKERS = (ROOT / 'batch_dubbing/gpu_worker.py', ROOT / 'modal_control/smoke_gpu_worker.py')


class SpeechStageTests(unittest.TestCase):
    def request(self, seconds=10, **kwargs):
        reader = MagicMock()
        reader.__enter__.return_value.getnframes.return_value = 60 * 16000
        reader.__enter__.return_value.getframerate.return_value = 16000
        with patch.object(Path, 'is_file', return_value=True), patch.object(stage.wave, 'open', return_value=reader):
            return stage.validate_request('source.wav', seconds, **kwargs)

    def test_valid_offset_and_language_scope_preserved(self):
        path, seconds, offset, langs = self.request(25, offset_seconds=30, target_languages=['de'])
        self.assertEqual((path, seconds, offset, langs), (Path('source.wav'), 25., 30., ('de',)))

    def test_invalid_numbers_rejected(self):
        for value in (True, False, None, '10', float('nan'), float('inf'), -1, 0, 7201):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.request(value)
        for offset in (True, '1', float('nan'), float('inf'), -1):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                self.request(offset_seconds=offset)

    def test_range_cannot_exceed_actual_source(self):
        with self.assertRaisesRegex(ValueError, 'exceeds source'):
            self.request(25, offset_seconds=40)
        self.assertEqual(self.request(25, offset_seconds=35)[2], 35)

    def test_missing_source_rejected_without_open(self):
        with patch.object(Path, 'is_file', return_value=False), patch.object(stage.wave, 'open') as opened:
            with self.assertRaisesRegex(ValueError, 'does not exist'):
                stage.validate_request('missing.wav', 5)
            opened.assert_not_called()

    def test_unknown_duplicate_empty_languages_rejected(self):
        for langs in ('de', [], ['de', 'de'], ['en'], ['xx'], [None], [42]):
            with self.subTest(langs=langs), self.assertRaises(ValueError):
                self.request(target_languages=langs)

    def test_invalid_requests_fail_before_gpu_or_model_imports(self):
        with patch.object(stage.subprocess, 'check_output') as gpu, patch.object(stage.subprocess, 'run') as crop:
            for kwargs in ({'seconds': 0}, {'seconds': 1, 'target_languages': ['xx']},
                           {'seconds': 1, 'max_source_units': 0}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    stage.process_audio('source.wav', **kwargs)
            gpu.assert_not_called()
            crop.assert_not_called()

    def test_source_unit_cap(self):
        stage._check_source_unit_count([None] * 12, 12)
        with self.assertRaisesRegex(ValueError, 'translation call bound'):
            stage._check_source_unit_count([None] * 13, 12)
        for cap in (True, 0, -1, 12.0, 10001, None):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                stage._check_source_unit_count([], cap)

    def test_all_crops_bounded_single_thread_no_overwrite(self):
        with patch.object(stage.subprocess, 'run') as run:
            stage._crop_audio('source.wav', 'output.wav', 325, 25)
            run.assert_called_once()
            args = run.call_args.args[0]
            self.assertIn('-n', args)
            self.assertNotIn('-y', args)
            self.assertEqual([args[i + 1] for i, value in enumerate(args) if value == '-threads'], ['1', '1'])
            self.assertEqual(args[args.index('-ss') + 1], '325.0')
            self.assertEqual(args[args.index('-t') + 1], '25.0')
            self.assertEqual(run.call_args.kwargs, {'check': True, 'timeout': 120})

    def test_invalid_crop_never_starts_child(self):
        with patch.object(stage.subprocess, 'run') as run:
            for offset, seconds in ((-1, 1), (0, 0), (float('nan'), 1), (0, 7201)):
                with self.assertRaises(ValueError):
                    stage._crop_audio('source', 'target', offset, seconds)
            run.assert_not_called()

    def test_cap_precedes_canary_loading_and_translation(self):
        source = (ROOT / 'batch_dubbing/speech_stage.py').read_text()
        count_check = source.index('_check_source_unit_count(source_units, max_source_units)')
        self.assertLess(count_check, source.index("repo = 'nvidia/canary-1b-v2'"))
        self.assertLess(count_check, source.index('for lang in target_languages:'))
        tree = ast.parse(source)
        ffmpeg_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute) and node.func.attr == 'run']
        self.assertEqual(len(ffmpeg_calls), 1)

    def test_workers_delegate_and_commit_on_success_or_error(self):
        for path in WORKERS:
            for fails in (False, True):
                with self.subTest(path=path.name, fails=fails):
                    tree = ast.parse(path.read_text())
                    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'process')
                    node.decorator_list = []
                    volume = Mock()
                    process_audio = Mock(return_value={'status': 'complete'},
                                         side_effect=RuntimeError('sentinel') if fails else None)
                    scope = {'volume': volume}
                    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
                    with patch.dict(sys.modules, {'batch_dubbing.speech_stage': SimpleNamespace(process_audio=process_audio)}), patch.object(sys, 'path', list(sys.path)):
                        args = (25, 2) if path.name == 'gpu_worker.py' else (25,)
                        if fails:
                            with self.assertRaisesRegex(RuntimeError, 'sentinel'):
                                scope['process'](*args)
                        else:
                            self.assertEqual(scope['process'](*args), {'status': 'complete'})
                    process_audio.assert_called_once_with('/input/video_02.wav' if path.name == 'gpu_worker.py' else '/input/smoke.wav', 25, commit_artifacts=volume.commit)
                    volume.commit.assert_called_once_with()

    def test_mount_and_catalog_pin_shared_stage(self):
        for path in WORKERS:
            self.assertIn("'/work/batch_dubbing/speech_stage.py'", path.read_text())
        for name in ('new_guys_gpu_smoke.json', 'new_guys_full_gpu.json'):
            for job in json.loads((ROOT / 'modal_control/catalog' / name).read_text())['jobs'].values():
                self.assertIn('batch_dubbing/speech_stage.py', job['source_files'])


if __name__ == '__main__':
    unittest.main()
