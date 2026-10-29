"""Pure fixtures and AST worker checks; never import Modal or model packages."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from batch_dubbing.cheap_native_probe import MODEL, REVISION, IMAGE_ID, PACKAGE, MAX_BYTES, run_probe, validate_input

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / 'batch_dubbing/cheap_native_worker.py'


def inputs():
    return {'schema': 1, 'model': MODEL, 'revision': REVISION, 'image_id': IMAGE_ID,
            'package': PACKAGE, 'speed': 1., 'steps': 8,
            'samples': [{'id': f'sample_{i}', 'language': language,
                         'text': 'An example.' if language == 'en' else 'Ein Beispiel.',
                         'voice': 'M1' if i < 2 else 'M2'}
                        for i, language in enumerate(('en', 'de', 'en', 'de'))]}


def measured(**changes):
    # Encoded in-memory synthetic fixture only; no decoder or media tool runs.
    payload = struct.pack('<100h', *([1000] * 100))
    fmt = struct.pack('<HHIIHH', 1, 1, 44100, 88200, 2, 16)
    body = b'WAVEfmt ' + struct.pack('<I', len(fmt)) + fmt + b'data' + struct.pack('<I', len(payload)) + payload
    audio = b'RIFF' + struct.pack('<I', len(body)) + body
    return {'audio': audio, 'sample_count': 100, 'sample_rate': 44100, 'peak': .1, **changes}


class CheapNativeProbeTests(unittest.TestCase):
    def test_four_calls_are_partitioned_once_smoke_then_workload(self):
        call = Mock(side_effect=lambda row: measured())
        smoke = run_probe(inputs(), True, synthesize=call)
        work = run_probe(inputs(), False, synthesize=call)
        self.assertEqual([args.args[0]['id'] for args in call.call_args_list],
                         ['sample_0', 'sample_1', 'sample_2', 'sample_3'])
        self.assertEqual((smoke['calls'], work['calls']), (1, 3))
        self.assertEqual((smoke['stage'], work['stage']), ('smoke', 'workload'))
        self.assertEqual(smoke['status'], 'complete')
        self.assertEqual(work['status'], 'complete')
        self.assertEqual(set(smoke['audio']) & set(work['audio']), set())

    def test_probe_labels_speed_explicitly_without_claiming_timing_or_naturalness(self):
        result = run_probe(inputs(), True, synthesize=lambda row: measured())
        self.assertEqual(result['synthesis_speed'], 1)
        self.assertEqual(result['playback_rate'], 1)
        self.assertEqual(result['steps'], 8)
        for field in ('duration_conditioning', 'pause_conditioning', 'timing_verified', 'naturalness_verified'):
            self.assertFalse(result[field])
        record = result['samples'][0]
        self.assertEqual(record['sha256'], hashlib.sha256(result['audio'][record['file']]).hexdigest())
        self.assertEqual(record['raw_duration_seconds'], 100 / 44100)

    def test_input_and_callback_rows_are_isolated_copies(self):
        data = inputs()
        before = deepcopy(data)
        selected = validate_input(data)
        selected['samples'][0]['text'] = 'Changed'
        self.assertEqual(data, before)
        def callback(row):
            row['text'] = 'Changed'
            return measured()
        result = run_probe(data, True, synthesize=callback)
        self.assertEqual(data, before)
        self.assertEqual(result['input'], before)
        self.assertEqual(result['samples'][0]['text'], before['samples'][0]['text'])

    def test_all_pins_and_small_input_scope_are_enforced_before_callback(self):
        cases = []
        for field in ('model', 'revision', 'image_id', 'package'):
            cases.append({**inputs(), field: 'changed'})
        for field, value in (('schema', True), ('speed', True), ('speed', 1.05), ('speed', float('nan')),
                             ('steps', True), ('steps', 9)):
            cases.append({**inputs(), field: value})
        for mutation in ('missing', 'extra', 'language', 'id', 'voice', 'text', 'blank', 'row_type'):
            data = inputs()
            if mutation == 'missing':
                data['samples'].pop()
            elif mutation == 'extra':
                data['samples'].append(data['samples'][0])
            elif mutation == 'row_type':
                data['samples'][0] = None
            else:
                field, value = {'language': ('language', 'fr'), 'id': ('id', '../escape'),
                                'voice': ('voice', 'M5'), 'text': ('text', 'x' * 81),
                                'blank': ('text', ' ')}[mutation]
                data['samples'][0][field] = value
            cases.append(data)
        call = Mock()
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                run_probe(data, True, synthesize=call)
        call.assert_not_called()
        for smoke in (0, 1, None, 'true'):
            with self.subTest(smoke=smoke), self.assertRaises(ValueError):
                run_probe(inputs(), smoke, synthesize=call)
        call.assert_not_called()

    def test_callback_failure_stops_without_retry_and_retains_prior_audio(self):
        call = Mock(side_effect=[measured(), RuntimeError('sentinel'), measured()])
        result = run_probe(inputs(), False, synthesize=call)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['calls'], 2)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(set(result['audio']), {'sample_1.wav'})
        self.assertEqual([row['id'] for row in result['samples']], ['sample_1'])
        self.assertIn('sentinel', json.dumps({k: v for k, v in result.items() if k != 'audio'}))

    def test_invalid_output_bounds_fail_and_stop_later_calls(self):
        cases = [None, measured(audio=b'not-wave'), measured(audio=b'RIFF0000WAVE'), measured(audio=b'RIFF' + b'0' * 12),
                 measured(sample_count=0), measured(sample_count=12 * 44100 + 1),
                 measured(sample_count=True), measured(sample_rate=24000),
                 measured(sample_rate=True), measured(peak=0), measured(peak=-.1),
                 measured(peak=float('inf')), measured(peak=float('nan')), measured(peak=True)]
        for value in cases:
            with self.subTest(value=str(value)[:80]):
                call = Mock(side_effect=[measured(), value, measured()])
                result = run_probe(inputs(), False, synthesize=call)
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['calls'], 2)
                self.assertEqual(call.call_count, 2)
                self.assertEqual(set(result['audio']), {'sample_1.wav'})

    def test_transfer_size_guard_stops_without_discarding_earlier_artifacts(self):
        oversized = measured(audio=b'RIFF0000WAVE' + b'x' * MAX_BYTES)
        call = Mock(side_effect=[measured(), oversized, measured()])
        result = run_probe(inputs(), False, synthesize=call)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(call.call_count, 2)
        self.assertEqual(set(result['audio']), {'sample_1.wav'})
        self.assertLessEqual(sum(map(len, result['audio'].values())), MAX_BYTES)

    def test_exact_twelve_second_count_boundary_is_allowed(self):
        result = run_probe(inputs(), True, synthesize=lambda row: measured(sample_count=12 * 44100))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['samples'][0]['raw_duration_seconds'], 12)

    def test_frozen_input_scope_is_valid_and_source_digest_matches(self):
        folder = ROOT / 'dubbing_naturalness_2026-10-28'
        data = json.loads((folder / 'cheap_native_inputs.json').read_text())
        self.assertEqual(validate_input(data), data)
        self.assertEqual(hashlib.sha256((folder / data['source']).read_bytes()).hexdigest(), data['source_sha256'])
        self.assertEqual([row['language'] for row in data['samples']], ['en', 'de', 'en', 'de'])
        self.assertTrue(all(len(row['text']) <= 80 for row in data['samples']))

    def test_worker_literal_resource_caps_and_cached_image_only(self):
        tree = ast.parse(WORKER.read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        forbidden = {'pip_install', 'apt_install', 'debian_slim', 'run_commands', 'from_registry',
                     'map', 'starmap', 'spawn'}
        self.assertFalse(any(isinstance(call.func, ast.Attribute) and call.func.attr in forbidden for call in calls))
        images = [call for call in calls if isinstance(call.func, ast.Attribute) and call.func.attr == 'from_id']
        self.assertEqual(len(images), 1)
        self.assertEqual(ast.literal_eval(images[0].args[0]), IMAGE_ID)
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'probe')
        resource_args = {kw.arg: ast.literal_eval(kw.value) for kw in fn.decorator_list[0].keywords
                         if kw.arg not in ('image', 'volumes')}
        self.assertEqual(resource_args, {'cpu': (2, 2), 'memory': (4096, 4096), 'timeout': 90,
                                        'startup_timeout': 30, 'scaledown_window': 2, 'min_containers': 0,
                                        'max_containers': 1, 'retries': 0})

    def test_worker_synthesis_and_cache_flags_are_explicit(self):
        tree = ast.parse(WORKER.read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        by_name = lambda name: [call for call in calls if isinstance(call.func, ast.Name) and call.func.id == name]
        downloads = by_name('snapshot_download')
        self.assertEqual(len(downloads), 1)
        self.assertIs(ast.literal_eval(next(kw.value for kw in downloads[0].keywords if kw.arg == 'local_files_only')), True)
        tts = by_name('TTS')
        self.assertEqual(len(tts), 1)
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in tts[0].keywords if kw.arg != 'model_dir'}
        self.assertFalse(kwargs['auto_download'])
        self.assertEqual(kwargs['intra_op_num_threads'], 2)
        self.assertEqual(kwargs['inter_op_num_threads'], 1)
        synth = [call for call in calls if isinstance(call.func, ast.Attribute) and call.func.attr == 'synthesize']
        self.assertEqual(len(synth), 1)
        settings = {kw.arg: ast.literal_eval(kw.value) for kw in synth[0].keywords
                    if kw.arg not in ('voice_style', 'lang')}
        self.assertEqual(settings, {'speed': 1., 'total_steps': 8, 'max_chunk_length': 300, 'silence_duration': 0.})
        chunk = by_name('chunk_text')
        self.assertEqual(len(chunk), 1)
        self.assertEqual(ast.literal_eval(chunk[0].args[1]), 300)
        self.assertIn('HF_HUB_OFFLINE="1"', WORKER.read_text())
        event_prefixes = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                          and isinstance(node.value, str) and node.value.endswith('EVENT ')]
        self.assertEqual(event_prefixes, ['MC_EVENT ', 'MC_EVENT '])

    def test_workload_requires_successful_matching_smoke_before_remote(self):
        tree = ast.parse(WORKER.read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        fn.decorator_list = []
        for prior in ({'status': 'failed', 'input': inputs(), 'calls': 1},
                      {'status': 'complete', 'input': {}, 'calls': 1},
                      {'status': 'complete', 'input': inputs(), 'calls': 0}):
            with self.subTest(prior=prior), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                input_path = root / 'inputs.json'
                input_path.write_text(json.dumps(inputs()))
                output = root / 'trial'
                smoke = output / 'smoke'
                smoke.mkdir(parents=True)
                (smoke / 'report.json').write_text(json.dumps(prior))
                remote = Mock()
                scope = {'ROOT': ROOT, 'INPUT': input_path, 'OUTPUT': output,
                         'probe': SimpleNamespace(remote=remote), 'Path': Path, 'json': json}
                exec(compile(ast.Module(body=[fn], type_ignores=[]), str(WORKER), 'exec'), scope)
                with patch.object(sys, 'path', list(sys.path)), self.assertRaises(ValueError):
                    scope['main'](False)
                remote.assert_not_called()
                self.assertFalse((output / 'workload').exists())

    def test_local_entrypoint_persists_failed_attempts_then_fails(self):
        result = run_probe(inputs(), False, synthesize=Mock(side_effect=[measured(), RuntimeError('sentinel')]))
        tree = ast.parse(WORKER.read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        fn.decorator_list = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / 'inputs.json'
            input_path.write_text(json.dumps(inputs()))
            output = root / 'trial'
            smoke = output / 'smoke'
            smoke.mkdir(parents=True)
            (smoke / 'report.json').write_text(json.dumps({'status': 'complete', 'input': inputs(), 'calls': 1}))
            remote = Mock(return_value=deepcopy(result))
            scope = {'ROOT': ROOT, 'INPUT': input_path, 'OUTPUT': output,
                     'probe': SimpleNamespace(remote=remote), 'Path': Path, 'json': json}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), str(WORKER), 'exec'), scope)
            with patch.object(sys, 'path', list(sys.path)), self.assertRaises(RuntimeError):
                scope['main'](False)
            destination = output / 'workload'
            saved = json.loads((destination / 'report.json').read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertEqual((destination / 'sample_1.wav').read_bytes(), result['audio']['sample_1.wav'])
            self.assertFalse((destination / 'sample_2.wav').exists())
            remote.assert_called_once()


if __name__ == '__main__':
    unittest.main()
