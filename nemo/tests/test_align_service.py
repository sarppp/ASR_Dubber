"""Warm aligner service: one load, same request code path, safe fallback. No models."""
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest

from batch_dubbing import align_service as svc


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')   # serve() pins CPU; restore afterwards
    monkeypatch.setenv('ALIGN_SERVICE', 'auto')
    path = Path(tempfile.mkdtemp(dir='/tmp')) / 's.sock'
    monkeypatch.setenv('ALIGN_SERVICE_SOCKET', str(path))
    loads, seen = [], []

    def loader():
        loads.append(1); time.sleep(.2); return 'MODEL'

    def runner(model, request):
        assert model == 'MODEL'
        seen.append(Path(request).name)
        print('aligned', Path(request).name)
        if 'bad' in Path(request).name:
            raise ValueError('Ending still unsynchronized')

    thread = threading.Thread(target=svc.serve, args=(path,), kwargs=dict(loader=loader, runner=runner), daemon=True)
    thread.start()
    for _ in range(100):
        if path.exists():
            break
        time.sleep(.02)
    yield path, loads, seen
    svc._call(path, dict(op='shutdown'), 5)
    thread.join(5)
    assert not path.exists()


def test_one_load_serves_every_request_with_captured_output(service, tmp_path):
    path, loads, seen = service
    first = svc.run(path, tmp_path / 'source_alignment_request.json', 30)
    second = svc.run(path, tmp_path / 'alignment_request.json', 30)
    assert loads == [1]
    assert seen == ['source_alignment_request.json', 'alignment_request.json']
    assert (first['returncode'], first['request_number']) == (0, 1)
    assert (second['returncode'], second['request_number']) == (0, 2)
    assert 'aligned alignment_request.json' in second['output']


def test_failed_request_reports_error_and_service_survives(service, tmp_path):
    path, _, _ = service
    reply = svc.run(path, tmp_path / 'bad_request.json', 30)
    assert reply['returncode'] == 1 and 'Ending still unsynchronized' in reply['error']
    assert svc.run(path, tmp_path / 'alignment_request.json', 30)['returncode'] == 0


def test_run_stage_uses_service_without_a_new_process(service, tmp_path, monkeypatch):
    def no_process(*a, **k):
        raise AssertionError('must not start a fresh aligner process')
    monkeypatch.setattr(svc.subprocess, 'run', no_process)
    assert svc.run_stage(tmp_path / 'alignment_request.json', python='py', timeout=30) == 0
    with pytest.raises(subprocess.CalledProcessError):
        svc.run_stage(tmp_path / 'bad_request.json', python='py', timeout=30, check=True)


def _record(monkeypatch):
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs)); return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(svc.subprocess, 'run', fake_run)
    return calls


def test_falls_back_to_the_original_stage_when_no_service(tmp_path, monkeypatch):
    monkeypatch.setenv('ALIGN_SERVICE', 'auto')
    monkeypatch.setenv('ALIGN_SERVICE_SOCKET', str(tmp_path / 'missing.sock'))
    calls = _record(monkeypatch)
    assert svc.run_stage(tmp_path / 'r.json', python='py', timeout=30, check=True) == 0
    cmd, kwargs = calls[0]
    assert cmd == ['py', '-m', 'batch_dubbing.aligned_finish', str(tmp_path / 'r.json')]
    assert kwargs['check'] is True and kwargs['cwd'] == svc.NEMO_DIR


def test_off_switch_and_changed_code_never_use_the_service(service, tmp_path, monkeypatch):
    calls = _record(monkeypatch)
    monkeypatch.setenv('ALIGN_SERVICE', 'off')
    svc.run_stage(tmp_path / 'a.json', python='py', timeout=30)
    monkeypatch.setenv('ALIGN_SERVICE', 'auto')
    monkeypatch.setattr(svc, 'code_digest', lambda: 'edited-since-start')
    svc.run_stage(tmp_path / 'b.json', python='py', timeout=30)
    assert [c[0][-1] for c in calls] == [str(tmp_path / 'a.json'), str(tmp_path / 'b.json')]
    _, _, seen = service
    assert seen == []


def test_ping_reports_identity(service):
    path, _, _ = service
    status = svc.ping(path)
    assert status['protocol'] == svc.PROTOCOL and status['code'] == svc.code_digest()
    assert svc.usable(path)


def test_device_must_match_and_gpu_fallback_passes_device(service, tmp_path, monkeypatch):
    path, _, seen = service
    assert svc.usable(path, 'cpu') and not svc.usable(path, 'cuda:0')
    calls = _record(monkeypatch)
    svc.run_stage(tmp_path / 'g.json', python='py', timeout=30, device='cuda:0')
    assert calls[0][0] == ['py', '-m', 'batch_dubbing.aligned_finish', str(tmp_path / 'g.json'), '--device', 'cuda:0']
    assert seen == []   # the CPU service never serves a GPU request


def test_gpu_service_reports_device(monkeypatch):
    monkeypatch.setenv('ALIGN_SERVICE', 'auto')
    path = Path(tempfile.mkdtemp(dir='/tmp')) / 'g.sock'
    thread = threading.Thread(target=svc.serve, args=(path,), daemon=True,
                              kwargs=dict(device='cuda:0', loader=lambda: 'M', runner=lambda m, r: None))
    thread.start()
    for _ in range(100):
        if path.exists():
            break
        time.sleep(.02)
    try:
        assert svc.ping(path)['device'] == 'cuda:0' and svc.usable(path, 'cuda:0')
        assert svc.run(path, '/tmp/x.json', 30, 'cuda:0')['returncode'] == 0
    finally:
        svc._call(path, dict(op='shutdown'), 5); thread.join(5)
