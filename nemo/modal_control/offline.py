"""Verify local smoke evidence without launching a second Modal workload."""
import json
from pathlib import Path
from .core import digest, verify_files


def verify_receipt(job):
    sources = {(Path(job['cwd']) / p).resolve() for p in job['source_files']}
    receipts = [(Path(job['cwd']) / p).resolve() for p in job['smoke_success_files']]
    if len(receipts) != 1 or receipts[0] not in sources:
        raise ValueError('Pin exactly one offline smoke receipt in source_files')
    data = json.loads(receipts[0].read_text())
    if (data.get('kind') != 'offline_smoke' or data.get('model_calls') != 0
            or data.get('cloud_calls') != 0 or not data.get('checks')):
        raise ValueError('Invalid offline smoke evidence')
    expected = sources - set(receipts)
    checked = {Path(p).resolve(): sha for p, sha in data.get('source_sha256', {}).items()}
    if set(checked) != expected or any(digest(p) != sha for p, sha in checked.items()):
        raise ValueError('Offline smoke does not cover the exact current sources')
    return verify_files(job, smoke=True)
