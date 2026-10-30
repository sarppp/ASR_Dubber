"""Read-only Modal billing snapshots and honest per-video cost/time reporting."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'crispr-10min-qwen-20261030-'


def capture(trial, label, billing_python):
    from dotenv import dotenv_values
    from modal_control.core import clean
    env = os.environ.copy()
    env.update({k: v for k, v in dotenv_values(ROOT / '.env').items() if v is not None})
    outputs = {}
    for kind in ('report', 'summary'):
        command = [str(billing_python), '-m', 'modal', 'billing', kind, '--for', 'this month', '--json']
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError(f'Billing {kind} unavailable: {clean(result.stderr[-1200:])}')
        outputs[kind] = json.loads(result.stdout)
    record = {'observed_at': datetime.now(timezone.utc).isoformat(), **outputs}
    destination = Path(trial) / f'billing_{label}_snapshot.json'
    with destination.open('x') as stream:
        json.dump(record, stream, indent=2)
    return {'path': str(destination), 'summary': outputs['summary']}


def application_total(rows):
    selected = [r for r in rows if r.get('description', r.get('Description', '')).startswith(PREFIX)]
    return sum((Decimal(r.get('cost', r.get('Cost'))) for r in selected), Decimal(0)), selected


def batch_timing(trial, state=ROOT / 'modal_control/.state'):
    database = Path(state) / 'state.sqlite'
    if not database.exists():
        return None
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        runs = connection.execute("SELECT * FROM runs WHERE job LIKE 'batch:%' ORDER BY created DESC").fetchall()
    attempts = []
    for run in reversed(runs):
        manifest = json.loads(run['manifest'])
        payload = manifest.get('release_plan', {}).get('payload', {})
        if Path(payload.get('spec_path', '')).parent != Path(trial).resolve():
            continue
        terminal = run['status'] in ('succeeded', 'failed', 'cancelled')
        end = datetime.fromisoformat(run['updated']) if terminal else datetime.now(timezone.utc)
        elapsed = (end - datetime.fromisoformat(run['created'])).total_seconds()
        attempts.append({'run_id': run['id'], 'status': run['status'], 'started_at': run['created'],
                'finished_at': run['updated'] if terminal else None,
                'elapsed_seconds': elapsed})
    if not attempts:
        return None
    last = attempts[-1]
    end = datetime.fromisoformat(last['finished_at']) if last['finished_at'] else datetime.now(timezone.utc)
    return {**last, 'started_at': attempts[0]['started_at'], 'attempts': attempts,
            'elapsed_seconds': (end-datetime.fromisoformat(attempts[0]['started_at'])).total_seconds(),
            'controller_active_seconds': sum(row['elapsed_seconds'] for row in attempts),
            'includes': 'All controller attempts; wall elapsed also includes repair gaps. Extraction and final packaging are separate.'}


def report(trial):
    trial = Path(trial)
    snapshots = sorted(trial.glob('billing_*_snapshot.json'), key=lambda p: p.stat().st_mtime)
    if not snapshots:
        raise ValueError('Capture billing before running the experiment')
    before, after = (json.loads(p.read_text()) for p in (snapshots[0], snapshots[-1]))
    old, _ = application_total(before['report'])
    new, items = application_total(after['report'])
    timings, issues = {}, []
    for stage, name in [('asr', 'asr'), ('translation', 'translated'), ('dub', 'dub')]:
        timings[stage] = {}
        for part, filename in [('smoke', stage + '_smoke.json'), ('full', name + '.json')]:
            path = trial / filename
            if part == 'smoke':
                candidates = sorted(trial.glob(stage + '_smoke*.json'), key=lambda p: p.stat().st_mtime)
                if candidates:
                    path = candidates[-1]
            if not path.exists():
                issues.append(f'{stage}/{part}: no completed local artifact')
                continue
            data = json.loads(path.read_text())
            timings[stage][part] = {'status': data.get('status'),
                'client_call_seconds': data.get('client_call_seconds'),
                'worker_timings': data.get('timings', {}),
                'worker_wall_seconds': data.get('worker_wall_seconds'),
                'synthesis_seconds': data.get('synthesis_wall_seconds')}
            if data.get('status') != 'complete':
                issues.append(f'{stage}/{part}: {data.get("status")}')
    manifest = json.loads((trial / 'input/manifest.json').read_text())
    metadata = trial / 'output/validation.json'
    packaged = json.loads(metadata.read_text()) if metadata.exists() else None
    if packaged is None:
        issues.append('No validated packaged video/subtitles')
    from batch_dubbing.phrase_trial_costs import breakdown
    phases = breakdown(trial, ROOT / 'modal_control/.state', items)
    summaries = {'before': before['summary'], 'after': after['summary']}
    return {'experiment': str(trial), 'status': 'complete' if not issues else 'incomplete',
            'application_cost_usd_so_far': str(new-old), 'application_billing_rows': items,
            'phase_costs': phases, 'clean_repeat_run_cost_measured': False,
            'billing_observed_at': after['observed_at'], 'billing_final': False,
            'billing_note': 'Modal may post charges later. App totals include smoke/full failures; workspace credit changes may also include unrelated jobs. Refresh after jobs stop before calling cost final.',
            'workspace_billing': summaries, 'stage_timings': timings,
            'batch_timing': batch_timing(trial),
            'local_extraction_seconds': manifest['extraction_seconds'],
            'local_packaging_seconds': packaged.get('packaging_seconds') if packaged else None,
            'issues': issues}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('--capture', help='Unique label such as before, after, settled')
    parser.add_argument('--billing-python', type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    value = capture(args.trial, args.capture, args.billing_python) if args.capture else report(args.trial)
    print(json.dumps(value, indent=2))
