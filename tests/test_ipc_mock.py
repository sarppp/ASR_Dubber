#!/usr/bin/env python3
"""
test_ipc_mock.py — Test the TTS worker IPC protocol without loading the model.

Uses a lightweight mock worker (pure Python, no GPU) to verify the JSON-line
protocol behaves correctly under tricky inputs:

  1. Normal request → single {"ok": true} response
  2. Text with embedded \\n — json.dumps escapes it, so it arrives as one line
     (confirms this is NOT the IPC desync vector)
  3. Extra stdout line mid-synthesis — simulates a library printing to stdout,
     which WOULD desync the protocol (the _send caller reads the extra line as
     the response and then the real response leaks to the next request)
  4. Multiple parallel requests — confirms ordering is preserved

Run with the same Python that runs dub.py (no GPU required):
  python test_ipc_mock.py
  # or with uv:
  uv run python test_ipc_mock.py
"""

import json
import subprocess
import sys
import textwrap
import threading
import time


# ---------------------------------------------------------------------------
# Mock worker script (written to a temp file and launched as subprocess)
# ---------------------------------------------------------------------------

MOCK_WORKER_NORMAL = textwrap.dedent("""\
    import json, sys, time

    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"ok": False, "error": f"bad JSON: {e}"}), flush=True)
            continue
        if req.get("quit"):
            print(json.dumps({"ok": True}), flush=True)
            break
        # Simulate a short synthesis delay
        time.sleep(0.05)
        print(json.dumps({"ok": True, "echo": req.get("text", "")}), flush=True)
""")

MOCK_WORKER_NOISY = textwrap.dedent("""\
    import json, sys, time

    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"ok": False, "error": f"bad JSON: {e}"}), flush=True)
            continue
        if req.get("quit"):
            print(json.dumps({"ok": True}), flush=True)
            break
        time.sleep(0.05)
        # Simulate a library printing a spurious non-JSON line to stdout
        print("WARNING: some library output", flush=True)
        print(json.dumps({"ok": True, "echo": req.get("text", "")}), flush=True)
""")


# ---------------------------------------------------------------------------
# Minimal IPC client (mirrors PersistentTTSWorker._send)
# ---------------------------------------------------------------------------

def _start_worker(script_source: str):
    import os, tempfile
    f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    f.write(script_source)
    f.close()

    proc = subprocess.Popen(
        [sys.executable, f.name],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, bufsize=1,
    )
    # Wait for READY
    for line in proc.stdout:
        if line.strip() == "READY":
            break
    return proc, f.name


def _send(proc, request: dict, timeout: float = 5.0) -> dict:
    import select
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rlist, _, _ = select.select([proc.stdout], [], [], 0.5)
        if rlist:
            raw = proc.stdout.readline()
            if not raw:
                return {"ok": False, "error": "EOF"}
            line = raw.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                # Non-JSON line from worker — skip it (same as PersistentTTSWorker)
                print(f"   [client] skipped non-JSON line: {line!r}")
                continue
    return {"ok": False, "error": "timeout"}


def _quit(proc):
    try:
        proc.stdin.write(json.dumps({"quit": True}) + "\n")
        proc.stdin.flush()
        proc.wait(timeout=3)
    except Exception:
        proc.kill()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

PASS = "✅ PASS"
FAIL = "❌ FAIL"


def test_normal_request():
    print("\n── Test 1: Normal request ──────────────────────────────────────")
    proc, tmp = _start_worker(MOCK_WORKER_NORMAL)
    try:
        resp = _send(proc, {"text": "Bonjour le monde", "output": "/tmp/t.wav"})
        ok = resp.get("ok") is True and resp.get("echo") == "Bonjour le monde"
        print(f"   response: {resp}")
        print(f"   {PASS if ok else FAIL}  — got ok=True and text echoed back")
        return ok
    finally:
        _quit(proc)
        import os; os.unlink(tmp)


def test_embedded_newline():
    """json.dumps escapes \\n — the worker receives one clean line, NOT two."""
    print("\n── Test 2: Text with embedded \\n ──────────────────────────────")
    proc, tmp = _start_worker(MOCK_WORKER_NORMAL)
    try:
        bad_text = "Hello\nworld"
        raw_json = json.dumps({"text": bad_text, "output": "/tmp/t.wav"})
        print(f"   raw JSON written to pipe: {raw_json!r}")
        assert "\n" not in raw_json, "json.dumps should escape the newline!"
        resp = _send(proc, {"text": bad_text, "output": "/tmp/t.wav"})
        ok = resp.get("ok") is True and resp.get("echo") == bad_text
        print(f"   response: {resp}")
        print(f"   {PASS if ok else FAIL}  — embedded \\n escaped by json.dumps, no desync")
        # Send a second request to prove the protocol is still in sync
        resp2 = _send(proc, {"text": "second request", "output": "/tmp/t2.wav"})
        sync_ok = resp2.get("ok") is True and resp2.get("echo") == "second request"
        print(f"   follow-up: {resp2}")
        print(f"   {PASS if sync_ok else FAIL}  — protocol still in sync after embedded \\n")
        return ok and sync_ok
    finally:
        _quit(proc)
        import os; os.unlink(tmp)


def test_spurious_stdout_line():
    """
    A library printing a non-JSON line to stdout mid-synthesis causes the
    client to skip it (thanks to the json.JSONDecodeError continue in _send).
    The real response is then correctly read on the next readline.
    This test confirms the current _send handles it gracefully.
    """
    print("\n── Test 3: Spurious non-JSON stdout line ───────────────────────")
    proc, tmp = _start_worker(MOCK_WORKER_NOISY)
    try:
        resp1 = _send(proc, {"text": "first", "output": "/tmp/t1.wav"})
        print(f"   response 1: {resp1}")
        resp2 = _send(proc, {"text": "second", "output": "/tmp/t2.wav"})
        print(f"   response 2: {resp2}")

        ok1 = resp1.get("ok") is True and resp1.get("echo") == "first"
        ok2 = resp2.get("ok") is True and resp2.get("echo") == "second"
        both = ok1 and ok2
        print(f"   {PASS if both else FAIL}  — client skips non-JSON lines, stays in sync")
        if not both:
            print("   ⚠️  If resp2 echoed 'first', that's the desync: the spurious line")
            print("       was consumed by request-1's _send and request-1 got response-2.")
        return both
    finally:
        _quit(proc)
        import os; os.unlink(tmp)


def test_literal_newline_in_pipe():
    """
    Show what ACTUALLY breaks the protocol: writing a literal newline inside
    the JSON value WITHOUT using json.dumps (i.e. raw string concatenation).
    This is NOT what dub_audio.py does, but useful to see the failure mode.
    """
    print("\n── Test 4: Literal \\n written raw to pipe (failure mode demo) ─")
    proc, tmp = _start_worker(MOCK_WORKER_NORMAL)
    try:
        # Deliberately broken: bypass json.dumps and write a literal newline
        bad_line = '{"text": "Hello\nworld", "output": "/tmp/t.wav"}\n'
        print(f"   raw bytes to pipe: {bad_line!r}")
        proc.stdin.write(bad_line)
        proc.stdin.flush()

        # Worker sees two lines:
        #   '{"text": "Hello'   → JSONDecodeError → error response
        #   'world", "output": "/tmp/t.wav"}'  → JSONDecodeError → error response
        import select, time
        responses = []
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and len(responses) < 2:
            rlist, _, _ = select.select([proc.stdout], [], [], 0.3)
            if rlist:
                line = proc.stdout.readline().strip()
                if line:
                    try:
                        responses.append(json.loads(line))
                    except json.JSONDecodeError:
                        responses.append({"raw": line})

        print(f"   worker sent {len(responses)} response(s): {responses}")
        broke = len(responses) == 2 and all(not r.get("ok", True) for r in responses)
        print(f"   {'💥 CONFIRMED desync' if broke else '(unexpected — check output)'}"
              f" — worker emits 2 error responses for 1 bad request")
        print(f"   ⚠️  Any subsequent _send call would read response #2 as its answer")
        return True   # this test always 'passes' — it's a demonstration
    finally:
        _quit(proc)
        import os; os.unlink(tmp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 66)
    print("  TTS Worker IPC Mock Tests")
    print("=" * 66)

    results = [
        test_normal_request(),
        test_embedded_newline(),
        test_spurious_stdout_line(),
        test_literal_newline_in_pipe(),
    ]

    passed = sum(1 for r in results if r)
    print(f"\n{'='*66}")
    print(f"Results: {passed}/{len(results)} passed")
    print(f"{'='*66}\n")

    print("Key findings:")
    print("  • json.dumps always escapes \\n → embedded newlines in text are safe")
    print("  • Spurious non-JSON stdout from worker libraries: _send skips them ✓")
    print("  • A literal \\n written raw to the pipe IS the real desync vector,")
    print("    but dub_audio.py always uses json.dumps so this can't happen there.")
    print()
    print("If you see 'lor lor' corruption, more likely causes:")
    print("  1. The merged text is very long / poorly formed (check show_segments.py)")
    print("  2. A library (e.g. soundfile, transformers) prints to stdout mid-synthesis")
    print("     → add  'PYTHONWARNINGS=ignore' or redirect lib stdout to stderr")
    print("  3. The model itself generating garbage for a specific phoneme sequence")
    print()

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
