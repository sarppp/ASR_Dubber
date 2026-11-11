"""Warm forced-aligner service: load the pinned CPU aligner once, reuse it.

Every request runs the same code as ``python -m batch_dubbing.aligned_finish``
(``load_aligner`` -> ``make_align`` -> ``run_request``); only the model load is
shared. CPU only, one request at a time, local Unix socket (mode 0600).

Per run: ``pipeline_subtitle`` starts an ephemeral service before ASR so the
load overlaps GPU work, and both alignment stages of ``dub.py`` reuse it.
Across runs: keep one running yourself, and every run skips the load::

    cd nemo && .venv/bin/python -m batch_dubbing.align_service serve   # foreground
    .venv/bin/python -m batch_dubbing.align_service status | stop

``ALIGN_SERVICE=off`` restores the one-process-per-stage behaviour exactly.
A service whose code differs from the files on disk is never used.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback

PROTOCOL = 1
NEMO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SOCKET = Path.home() / '.cache' / 'asr-aligner' / 'aligner.sock'


class Unavailable(RuntimeError):
    """No usable service; the caller falls back to a fresh process."""


def code_digest() -> str:
    """Identity of every source file a request can execute."""
    files = sorted((NEMO_DIR / 'batch_dubbing').glob('*.py'))
    files += [NEMO_DIR / name for name in ('subtitle_combine.py', 'local_mt.py') if (NEMO_DIR / name).exists()]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode() + b'\0' + path.read_bytes() + b'\0')
    return digest.hexdigest()


def socket_path() -> Path:
    return Path(os.environ.get('ALIGN_SERVICE_SOCKET') or DEFAULT_SOCKET)


def _send(conn, data):
    conn.sendall(json.dumps(data, ensure_ascii=False).encode() + b'\n')


def _recv(conn):
    chunks = []
    while True:
        block = conn.recv(65536)
        if not block:
            break
        chunks.append(block)
        if block.endswith(b'\n'):
            break
    if not chunks:
        raise ConnectionError('Empty message')
    return json.loads(b''.join(chunks))


def _call(path, message, timeout):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(timeout)
            conn.connect(str(path))
            _send(conn, message)
            return _recv(conn)
    except (OSError, ConnectionError, ValueError) as exc:
        raise Unavailable(f'Alignment service at {path} unavailable: {exc}') from exc


def ping(path=None, timeout=2.0):
    """Status dict, or None when nothing answers."""
    try:
        return _call(Path(path or socket_path()), dict(op='ping'), timeout)
    except Unavailable:
        return None


def usable(path=None):
    """True only for a live service running exactly the code on disk."""
    status = ping(path)
    return bool(status and status.get('protocol') == PROTOCOL and status.get('code') == code_digest())


def run(path, request_path, timeout):
    """Run one request file on the warm aligner; returns the reply dict."""
    path = Path(path)
    if not usable(path):
        raise Unavailable(f'No matching alignment service at {path}')
    reply = _call(path, dict(op='run', request=str(Path(request_path).resolve()), code=code_digest()), timeout)
    if not reply.get('ok'):
        raise Unavailable(reply.get('error') or 'Alignment service refused the request')
    return reply


def run_stage(request_path, *, python, timeout, check=False):
    """Use a matching warm service when one is reachable, else a fresh process.

    Output is echoed like the subprocess stage; returns the process-style code.
    """
    if os.environ.get('ALIGN_SERVICE', 'auto') != 'off':
        path = socket_path()
        try:
            reply = run(path, request_path, timeout)
        except Unavailable as exc:
            if path.exists():
                print(f'Warm aligner not used ({exc}); loading a fresh aligner process', flush=True)
        else:
            sys.stdout.write(reply.get('output') or '')
            if reply.get('error'):
                sys.stderr.write(reply['error'])
            print(f"Warm aligner: request {reply['request_number']} on one model load "
                  f"(load {reply['load_seconds']:.1f}s, this request {reply['seconds']:.1f}s)", flush=True)
            if check and reply['returncode']:
                raise subprocess.CalledProcessError(reply['returncode'], ['align_service', str(request_path)])
            return reply['returncode']
    result = subprocess.run([str(python), '-m', 'batch_dubbing.aligned_finish', str(request_path)],
                            cwd=NEMO_DIR, check=check, timeout=timeout)
    return result.returncode


def serve(path=None, *, parent_pid=None, loader=None, runner=None, ready_timeout=900):
    """Bind first, load in the background, then answer requests one at a time."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''   # CPU-only stage; never touch the GPU
    from batch_dubbing.aligned_finish import load_aligner, make_align, run_request
    loader = loader or (lambda: load_aligner('cpu'))
    runner = runner or (lambda model, request: run_request(request, make_align(model, 'cpu', request)))
    path = Path(path or socket_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if ping(path):
            raise SystemExit(f'An alignment service already answers at {path}')
        path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path)); os.chmod(path, 0o600); server.listen(4); server.settimeout(1.0)
    digest = code_digest()
    state = dict(model=None, error=None, load_seconds=None, requests=0)
    ready = threading.Event()

    def load():
        began = time.perf_counter()
        try:
            state['model'] = loader()
        except BaseException:
            state['error'] = traceback.format_exc()
        state['load_seconds'] = time.perf_counter() - began
        ready.set()

    threading.Thread(target=load, daemon=True).start()
    print(f'Alignment service listening on {path} (pid {os.getpid()}); loading CPU aligner', flush=True)
    try:
        while True:
            if parent_pid and not _alive(parent_pid):
                print('Parent process ended; stopping alignment service', flush=True)
                break
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            with conn:
                conn.settimeout(None)
                try:
                    message = _recv(conn)
                except (OSError, ConnectionError, ValueError):
                    continue
                op = message.get('op')
                if op == 'ping':
                    _send(conn, dict(ok=True, pid=os.getpid(), protocol=PROTOCOL, code=digest, loaded=ready.is_set(),
                                     load_error=state['error'], load_seconds=state['load_seconds'], requests=state['requests']))
                elif op == 'shutdown':
                    _send(conn, dict(ok=True)); break
                elif op == 'run':
                    if message.get('code') != digest or code_digest() != digest:
                        _send(conn, dict(ok=False, error='Service code differs from the files on disk; restart it')); continue
                    if not ready.wait(ready_timeout) or state['error']:
                        _send(conn, dict(ok=False, error=state['error'] or 'Aligner did not load in time')); continue
                    output, began, error = io.StringIO(), time.perf_counter(), None
                    try:
                        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                            runner(state['model'], Path(message['request']))
                        returncode = 0
                    except BaseException:
                        returncode, error = 1, traceback.format_exc()
                    state['requests'] += 1
                    _send(conn, dict(ok=True, returncode=returncode, output=output.getvalue(), error=error,
                                     seconds=time.perf_counter() - began, load_seconds=state['load_seconds'],
                                     request_number=state['requests']))
                else:
                    _send(conn, dict(ok=False, error=f'Unknown operation {op!r}'))
    finally:
        server.close()
        path.unlink(missing_ok=True)


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextlib.contextmanager
def ephemeral(python, *, log=None):
    """Per-run service: started early, shared via ALIGN_SERVICE_SOCKET, stopped after.

    Reuses a matching persistent service instead when one already answers.
    Yields the socket path in use, or None when disabled or unavailable.
    """
    if os.environ.get('ALIGN_SERVICE', 'auto') == 'off':
        yield None; return
    if usable(socket_path()):
        os.environ['ALIGN_SERVICE_SOCKET'] = str(socket_path())
        yield socket_path(); return
    path = Path(tempfile.gettempdir()) / f'asr-aligner-{os.getpid()}.sock'
    previous = os.environ.get('ALIGN_SERVICE_SOCKET')
    stream = open(log, 'a') if log else subprocess.DEVNULL
    process = subprocess.Popen([str(python), '-m', 'batch_dubbing.align_service', 'serve', '--socket', str(path),
                                '--parent-pid', str(os.getpid())], cwd=NEMO_DIR, stdout=stream, stderr=subprocess.STDOUT)
    try:
        for _ in range(100):   # binding is immediate; the model keeps loading in the background
            if path.exists() or process.poll() is not None:
                break
            time.sleep(.05)
        os.environ['ALIGN_SERVICE_SOCKET'] = str(path)
        yield path if process.poll() is None else None
    finally:
        if previous is None:
            os.environ.pop('ALIGN_SERVICE_SOCKET', None)
        else:
            os.environ['ALIGN_SERVICE_SOCKET'] = previous
        with contextlib.suppress(Unavailable):
            _call(path, dict(op='shutdown'), 5)
        try:
            process.wait(10)
        except subprocess.TimeoutExpired:
            process.kill()
        if stream is not subprocess.DEVNULL:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['serve', 'status', 'stop'])
    parser.add_argument('--socket', type=Path, default=None)
    parser.add_argument('--parent-pid', type=int, default=None)
    args = parser.parse_args()
    path = args.socket or socket_path()
    if args.command == 'serve':
        serve(path, parent_pid=args.parent_pid)
    elif args.command == 'status':
        status = ping(path)
        if status:
            status['matches_code_on_disk'] = status.get('code') == code_digest()
        print(json.dumps(status or dict(running=False, socket=str(path)), indent=2))
    else:
        try:
            _call(path, dict(op='shutdown'), 5); print('stopped')
        except Unavailable:
            print('not running')


if __name__ == '__main__':
    main()
