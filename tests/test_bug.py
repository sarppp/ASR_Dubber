import subprocess
import json
import sys
import time
import select
from pathlib import Path

worker_path = Path('qwen_tts_worker.py')

proc = subprocess.Popen(
    [sys.executable, str(worker_path), '--mode', 'custom'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
)

print('Waiting for READY...', file=sys.stderr)
for line in proc.stdout:
    if line.strip() == 'READY':
        print('Worker is READY')
        break

# Test 1: Send a request with embedded newline (BUG)
print('\n=== TEST 1: Text with embedded newline (BUG) ===', file=sys.stderr)
bad_request = {
    'text': 'Hello\nworld',
    'voice': 'aiden',
    'language': 'English',
    'output': '/tmp/test_bad.wav'
}

print(f'Sending: {repr(bad_request["text"])}', file=sys.stderr)
proc.stdin.write(json.dumps(bad_request) + '\n')
proc.stdin.flush()

# Read responses with timeout
deadline = time.time() + 10
responses = []
while time.time() < deadline:
    rlist, _, _ = select.select([proc.stdout], [], [], 0.5)
    if rlist:
        line = proc.stdout.readline()
        if not line:
            break
        responses.append(line)
        print(f'Got: {line.rstrip()}', file=sys.stderr)
        if line.strip().startswith('{'):
            break

print(f'Received {len(responses)} responses (expected 1)', file=sys.stderr)

# Test 2: Send a request with escaped newline (FIX)
print('\n=== TEST 2: Text with escaped newline (FIX) ===', file=sys.stderr)
good_request = {
    'text': 'Hello world',  # No embedded newline
    'voice': 'aiden',
    'language': 'English',
    'output': '/tmp/test_good.wav'
}

print(f'Sending: {repr(good_request["text"])}', file=sys.stderr)
proc.stdin.write(json.dumps(good_request) + '\n')
proc.stdin.flush()

# Read responses with timeout
deadline = time.time() + 30
responses = []
while time.time() < deadline:
    rlist, _, _ = select.select([proc.stdout], [], [], 0.5)
    if rlist:
        line = proc.stdout.readline()
        if not line:
            break
        line_stripped = line.strip()
        if line_stripped.startswith('{'):
            responses.append(line)
            print(f'Got: {line_stripped}', file=sys.stderr)
            break
        else:
            print(f'Warming up: {line_stripped[:50]}...', file=sys.stderr)

print(f'Received {len(responses)} responses', file=sys.stderr)

# Clean up
proc.stdin.write('{"quit": true}\n')
proc.stdin.flush()
proc.wait(timeout=5)
