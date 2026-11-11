"""Exercise worker glue with fake synthesis; never import Modal or audio models."""
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from batch_dubbing.speech_units import build_speech_units, bind_translations

ROOT = Path(__file__).resolve().parents[1]
WORKERS = [ROOT / 'batch_dubbing' / f'{name}.py' for name in
           ('cpu_worker', 'cpu_duo_worker', 'cpu_override_worker', 'cpu_continuity_worker')]
WORKERS.append(ROOT / 'modal_control/smoke_cpu_worker.py')
HELPERS = ('natural_pipeline', 'audio_edges', 'speech_schedule', 'natural_render', 'cpu_natural')


def extracted(path, name, **namespace):
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    scope = {'json': json, 'Path': Path, **namespace}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


class CpuWorkerIntegrationTests(unittest.TestCase):
    def remote_call(self, path, *, error=None, status='complete', volume_error=None):
        result = {'audio': b'fixture' if status == 'complete' else None, 'report': {'status': status}}
        generate = Mock(return_value=result, side_effect=error)
        monitor = Mock()
        volume = Mock()
        volume.commit.side_effect = volume_error
        fallback = Mock(side_effect=lambda voices: voices)
        imports = {'telemetry': SimpleNamespace(emit=Mock(), start=Mock(return_value=monitor)),
                   'batch_dubbing.cpu_natural': SimpleNamespace(synthesize_natural=generate),
                   'voice_fallback': SimpleNamespace(resolve_borderline_voices=fallback)}
        data = {'voices': {'s': {'voice': 'M2'}, 'unused': {'voice': 'F1'}},
                'speech_units': {'de': [{'speaker': 's'}]}}
        with patch.dict(sys.modules, imports), patch.object(sys, 'path', list(sys.path)):
            fn = extracted(path, 'synthesize', volume=volume)
            if error or volume_error:
                with self.assertRaisesRegex(RuntimeError, 'sentinel'):
                    fn(data, 15)
            else:
                self.assertIs(fn(data, 15), result)
        volume.commit.assert_called_once_with()
        monitor.set.assert_called_once_with()
        return generate, fallback, result, data

    def test_every_worker_delegates_with_correct_format_and_explicit_speaker_basis(self):
        for path in WORKERS:
            with self.subTest(worker=path.name):
                generate, _, result, _ = self.remote_call(path)
                generate.assert_called_once()
                self.assertEqual(generate.call_args.kwargs['output_format'],
                                 'wav' if path.stem == 'smoke_cpu_worker' else 'm4a')
                self.assertEqual(result['report']['speaker_assignment_basis'], 'verified_source_speech_units')

    def test_native_attempts_commit_and_monitor_stops_after_synthesis_exception(self):
        for path in WORKERS:
            with self.subTest(worker=path.name):
                self.remote_call(path, error=RuntimeError('sentinel'))

    def test_monitor_stops_if_volume_commit_fails(self):
        for path in WORKERS:
            with self.subTest(worker=path.name):
                self.remote_call(path, volume_error=RuntimeError('sentinel'))

    def test_blocked_result_is_returned_with_native_diagnostics(self):
        for path in WORKERS:
            with self.subTest(worker=path.name):
                _, _, result, _ = self.remote_call(path, status='needs_revision')
                self.assertIsNone(result['audio'])
                self.assertEqual(result['report']['status'], 'needs_revision')

    def test_voice_policies_are_preserved_without_mutating_input(self):
        for path in WORKERS:
            with self.subTest(worker=path.name):
                generate, fallback, result, data = self.remote_call(path)
                voices = generate.call_args.args[2]
                self.assertEqual(voices['s']['voice'], 'M2')
                self.assertEqual(len(voices), 1 if path.stem == 'cpu_continuity_worker' else 2)
                self.assertEqual(fallback.call_count, int(path.stem in ('cpu_override_worker', 'cpu_continuity_worker')))
                self.assertEqual(len(data['voices']), 2)
                self.assertEqual(result['report']['voice_assignments'], voices)

    def local_call(self, path, status, audio, *, existing=False, invalid_source=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'modal_control/batch_inputs'
            folder.mkdir(parents=True)
            word_start = 20 if invalid_source == 'empty_range' else 1
            units = build_speech_units(
                [{'word': 'Example.', 'start': word_start, 'end': word_start + 1, 'speaker': 's'}],
                [{'start': 0, 'end': 60, 'speaker': 's'}], source_id='test')
            units = bind_translations(units, {unit.id: 'Beispiel.' for unit in units})
            source = {'duration_seconds': 60, 'voices': {'s': {'voice': 'M1'}},
                      'speech_units': {'de': [unit.to_dict() for unit in units]}}
            if invalid_source == 'legacy':
                del source['speech_units']
                source['translations'] = {'de': [{'start': 1, 'end': 2, 'text': 'Beispiel.'}]}
            for name in ('video_00.gpu_full.json', 'gpu_smoke.json', 'video_00.pitch_review_v2.json'):
                (folder / name).write_text(json.dumps(source))
            name = 'cpu_v2_preflight' if path.stem == 'smoke_cpu_worker' else 'video_00.cpu_v2_preflight'
            media = folder / (name + ('.wav' if path.stem == 'smoke_cpu_worker' else '.m4a'))
            report = folder / (name + '.json')
            if existing:
                report.write_text('existing')
            remote = Mock(return_value={'audio': audio, 'report': {'status': status}})
            fn = extracted(path, 'main', ROOT=root, synthesize=SimpleNamespace(remote=remote))
            args = {'smoke': True}
            if path.stem != 'smoke_cpu_worker':
                args['video_index'] = 0
            if existing:
                with self.assertRaises(FileExistsError):
                    fn(**args)
                remote.assert_not_called()
                self.assertEqual(report.read_text(), 'existing')
            elif invalid_source:
                with self.assertRaises(ValueError):
                    fn(**args)
                remote.assert_not_called()
                self.assertFalse(media.exists())
                self.assertFalse(report.exists())
            elif status != 'complete' or audio is None:
                with self.assertRaisesRegex(RuntimeError, 'diagnostics saved'):
                    fn(**args)
                self.assertEqual(json.loads(report.read_text())['status'], status)
                self.assertFalse(media.exists())
            else:
                fn(**args)
                self.assertEqual(media.read_bytes(), audio)
                self.assertEqual(json.loads(report.read_text())['status'], 'complete')

    def test_legacy_subtitle_artifacts_fail_before_remote_allocation(self):
        for path in WORKERS:
            with self.subTest(worker=path.name):
                self.local_call(path, 'complete', b'fixture', invalid_source='legacy')

    def test_range_without_complete_utterance_fails_before_remote_allocation(self):
        for path in WORKERS:
            with self.subTest(worker=path.name):
                self.local_call(path, 'complete', b'fixture', invalid_source='empty_range')

    def test_entrypoints_persist_blocked_report_without_packaging(self):
        for path in WORKERS:
            with self.subTest(worker=path.name):
                self.local_call(path, 'needs_revision', None)

    def test_entrypoints_reject_inconsistent_result_even_if_audio_present(self):
        for path in WORKERS:
            with self.subTest(worker=path.name):
                self.local_call(path, 'needs_revision', b'invalid')
                self.local_call(path, 'complete', None)

    def test_entrypoints_write_successful_results(self):
        for path in WORKERS:
            with self.subTest(worker=path.name):
                self.local_call(path, 'complete', b'fixture')

    def test_preexisting_outputs_prevent_remote_calls(self):
        for path in WORKERS:
            with self.subTest(worker=path.name):
                self.local_call(path, 'complete', b'fixture', existing=True)

    def test_worker_mounts_and_catalogs_pin_actual_dependencies(self):
        paths = {str(path.relative_to(ROOT)) for path in WORKERS}
        for path in WORKERS:
            tree = ast.parse(path.read_text())
            strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            for helper in HELPERS:
                self.assertIn(f'/work/batch_dubbing/{helper}.py', strings)
                self.assertIn(f'batch_dubbing/{helper}.py', strings)
            decorator = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'synthesize').decorator_list[0]
            caps = {kw.arg: ast.literal_eval(kw.value) for kw in decorator.keywords
                    if kw.arg in {'cpu', 'memory', 'timeout', 'max_containers', 'retries'}}
            self.assertEqual(caps, {'cpu': 4, 'memory': 16384, 'timeout': 1800 if path.stem == 'smoke_cpu_worker' else 7200, 'max_containers': 1, 'retries': 0})
        matched = 0
        for catalog in (ROOT / 'modal_control/catalog').glob('*.json'):
            for job in json.loads(catalog.read_text()).get('jobs', {}).values():
                if not paths.intersection(job.get('command', [])):
                    continue
                matched += 1
                for helper in HELPERS:
                    self.assertIn(f'batch_dubbing/{helper}.py', job['source_files'])
        self.assertGreaterEqual(matched, 5)


if __name__ == '__main__':
    unittest.main()
