"""AST-isolated worker tests: no Modal import, model, or media processing."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from batch_dubbing.speech_units import build_speech_units, bind_translations

ROOT = Path(__file__).resolve().parents[1]


def functions_only(name):
    source = ROOT / 'batch_dubbing' / name
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for function in functions:
        function.decorator_list = []
    namespace = dict(Path=Path, json=json, deepcopy=deepcopy, hashlib=hashlib,
                     NATIVE_BYTES_LIMIT=64*1024**2)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def fixture():
    config = json.loads((ROOT / 'dubbing_naturalness_2026-10-28/source_validation_inputs.json').read_text())
    results = []
    for case in config['cases']:
        words = [dict(word='First.', start=.1, end=.5, speaker='speaker_a'),
                 dict(word='Second.', start=1, end=1.5, speaker='speaker_a')]
        units = build_speech_units(words, [dict(start=0, end=case['duration_seconds'], speaker='speaker_a')], source_id=case['id'])
        units = bind_translations(units, {unit.id: 'Ein Satz.' for unit in units})
        data = dict(status='complete', duration_seconds=case['duration_seconds'], source_language='en',
                    source_offset_seconds=case['offset_seconds'], voices={'speaker_a': {'voice': 'M1'}},
                    speech_units={'de': [unit.to_dict() for unit in units]})
        results.append({'id': case['id'], 'data': data})
    return config, dict(status='complete', stage='gpu_full', config=deepcopy(config), cases=results)


class ValidationWorkerTests(unittest.TestCase):
    def setUp(self):
        self.gpu = functions_only('validation_gpu_worker.py')
        self.cpu = functions_only('validation_cpu_worker.py')
        self.config, self.result = fixture()

    def test_resources_single_container_serial_and_no_direct_models(self):
        for name, cpu, memory, gpu in (('validation_gpu_worker.py', 4, 32768, 'T4'),
                                       ('validation_cpu_worker.py', 2, 4096, None)):
            tree = ast.parse((ROOT / 'batch_dubbing' / name).read_text())
            process = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'process')
            kwargs = {kw.arg: kw.value.value for kw in process.decorator_list[0].keywords if isinstance(kw.value, ast.Constant)}
            self.assertEqual((kwargs['cpu'], kwargs['memory'], kwargs['timeout'], kwargs['max_containers'], kwargs['retries']),
                             (cpu, memory, 600, 1, 0))
            self.assertEqual(kwargs.get('gpu'), gpu)
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            self.assertFalse(any(isinstance(c.func, ast.Attribute) and c.func.attr in ('map', 'starmap', 'spawn') for c in calls))
            self.assertFalse(any(isinstance(c.func, ast.Name) and c.func.id in ('TTS', 'snapshot_download') for c in calls))

    def test_worker_readiness_before_dispatch_and_cpu_monitor_shutdown(self):
        for worker in (self.cpu, self.gpu):
            events, monitor = [], Mock()
            emit = Mock(side_effect=lambda name, **kwargs: events.append(name))
            telemetry = SimpleNamespace(emit=emit, start=Mock(return_value=monitor))
            worker['volume'] = Mock()
            worker['run_cases'] = Mock(side_effect=lambda *args, **kwargs: events.append('dispatch') or {'status': 'complete'})
            old_path = list(sys.path)
            try:
                with patch.dict(sys.modules, {'telemetry': telemetry}):
                    if worker is self.cpu:
                        worker['process'](self.config, self.result, True)
                    else:
                        worker['process'](self.config, True)
            finally:
                sys.path[:] = old_path
            self.assertEqual(events, ['worker_ready', 'dispatch'])
            self.assertIs(worker['run_cases'].call_args.kwargs['emit'], emit)
            worker['volume'].commit.assert_called_once()
            if worker is self.cpu:
                telemetry.start.assert_called_once()
                monitor.set.assert_called_once()

    def test_cpu_case_and_segment_telemetry_forwarded(self):
        emit = Mock()
        synthesize = Mock(return_value={'audio': None, 'report': {'status': 'needs_revision'}})
        self.cpu['run_cases'](self.config, self.result, True, synthesizer=synthesize, commit=Mock(), emit=emit)
        self.assertIs(synthesize.call_args.kwargs['emit'], emit)
        self.assertEqual([call.args[0] for call in emit.call_args_list], ['case_started', 'case_complete'])

    def test_gpu_validates_all_inputs_before_serial_cases_and_cleans_after_failure(self):
        verified = []
        def process(path, seconds, **kwargs):
            self.assertEqual(len(verified), 3)
            if path.name == 'video_04.wav':
                raise RuntimeError('Synthetic failure')
            return {'status': 'complete'}
        cleanup, commit = Mock(), Mock()
        result = self.gpu['run_cases'](self.config, False, processor=Mock(side_effect=process),
            verifier=lambda case, path: verified.append(case['id']), commit=commit, cleanup=cleanup)
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual([row['id'] for row in result['cases']], ['bill', 'mind', 'crispr'])
        self.assertEqual(cleanup.call_count, 3)
        self.assertEqual(commit.call_count, 3)
        self.assertIn('Synthetic failure', result['cases'][1]['data']['error'])

    def test_gpu_smoke_only_heldout_case_and_invalid_input_never_calls_processor(self):
        process = Mock(return_value={'status': 'complete'})
        result = self.gpu['run_cases'](self.config, True, processor=process, verifier=Mock(), commit=Mock())
        self.assertEqual([row['id'] for row in result['cases']], ['crispr'])
        self.assertEqual(process.call_count, 1)
        self.assertEqual(process.call_args.kwargs['target_languages'], ('de',))
        self.assertEqual(process.call_args.kwargs['max_source_units'], 12)
        process.reset_mock()
        with self.assertRaises(ValueError):
            self.gpu['run_cases'](self.config, False, processor=process,
                verifier=Mock(side_effect=ValueError('Hash mismatch')), commit=Mock())
        process.assert_not_called()

    def test_cpu_smoke_selects_whole_unit_and_preserves_full_source_metadata(self):
        original = deepcopy(self.result)
        cases = self.cpu['prepare_cases'](self.config, self.result, True)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]['id'], 'crispr')
        self.assertEqual(cases[0]['seconds'], .5)
        self.assertEqual(len(cases[0]['unit_ids']), 1)
        self.assertEqual(len(cases[0]['data']['speech_units']['de']), 2)
        self.assertEqual(self.result, original)

    def test_cpu_uncertain_voice_and_changed_offset_fail_before_synthesis(self):
        for mutate in (lambda data: data['voices']['speaker_a'].update(voice=None),
                       lambda data: data.update(source_offset_seconds=99)):
            modified = deepcopy(self.result)
            mutate(modified['cases'][0]['data'])
            synthesize = Mock()
            with self.assertRaises(ValueError):
                self.cpu['run_cases'](self.config, modified, False, synthesizer=synthesize, commit=Mock())
            synthesize.assert_not_called()

    def test_cpu_unit_cap_checked_before_any_synthesis(self):
        modified = deepcopy(self.result)
        units = modified['cases'][0]['data']['speech_units']['de']
        first = units[0]
        modified['cases'][0]['data']['speech_units']['de'] = []
        for i in range(13):
            unit = deepcopy(first)
            unit.update(id=f'u{i}', source_start=i, source_end=i+.4, source_word_indices=[i])
            unit['words'][0].update(index=i, start=i, end=i+.4)
            modified['cases'][0]['data']['speech_units']['de'].append(unit)
        synthesize = Mock()
        with self.assertRaisesRegex(ValueError, 'unit cap'):
            self.cpu['run_cases'](self.config, modified, False, synthesizer=synthesize, commit=Mock())
        synthesize.assert_not_called()

    def test_cpu_failure_preserves_diagnostics_without_retry(self):
        synthesize = Mock(return_value={'audio': None, 'report': {'status': 'needs_revision', 'why': 'too long'}})
        result = self.cpu['run_cases'](self.config, self.result, True, synthesizer=synthesize, commit=Mock())
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(synthesize.call_count, 1)
        self.assertEqual(result['cases'][0]['report']['why'], 'too long')
        self.assertEqual(synthesize.call_args.args[1], .5)

    def test_native_transfer_hash_cap_and_path_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'native.wav'
            path.write_bytes(b'synthetic fixture')
            attempt = dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), id='unit', attempt=0)
            report = dict(native_artifact_directory=temp, attempts=[attempt])
            artifacts, remaining = self.cpu['collect_native'](report, 100)
            self.assertEqual(remaining, 100-path.stat().st_size)
            self.assertEqual(artifacts[0]['bytes'], path.read_bytes())
            with self.assertRaises(ValueError):
                self.cpu['collect_native'](report, 1)
            attempt['sha256'] = '0'*64
            with self.assertRaises(ValueError):
                self.cpu['collect_native'](report, 100)

    def test_blocked_batch_saves_native_and_report_but_no_final_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            row = dict(id='crispr', report={'status': 'needs_revision'}, audio=b'unpublished', unit_ids=['u'],
                       native=[dict(name='raw_000.wav', bytes=b'raw', sha256=hashlib.sha256(b'raw').hexdigest())])
            result = dict(status='blocked', cases=[row])
            self.cpu['publish_result'](temp, result)
            self.assertTrue((Path(temp)/'report.json').exists())
            self.assertTrue((Path(temp)/'crispr/native/raw_000.wav').exists())
            self.assertFalse((Path(temp)/'crispr/timeline.wav').exists())
            with self.assertRaises(FileExistsError):
                self.cpu['publish_result'](temp, result)

    def test_bad_timeline_hash_never_publishes_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            result = dict(status='complete', cases=[dict(id='crispr', audio=b'bad', native=[], unit_ids=['u'],
                          report={'status': 'complete', 'rendered': {'sha256': '0'*64}})])
            with self.assertRaisesRegex(ValueError, 'Timeline transfer hash'):
                self.cpu['publish_result'](temp, result)
            self.assertEqual(list(Path(temp).iterdir()), [Path(temp)/'report.json'])
            self.assertEqual(json.loads((Path(temp)/'report.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
