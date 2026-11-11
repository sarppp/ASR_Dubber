"""Attribute posted experiment app costs to controller phases and outcomes."""
import json
from decimal import Decimal
from pathlib import Path
import sqlite3


def breakdown(trial, state, billing_rows):
    trial = Path(trial).resolve()
    database = Path(state) / 'state.sqlite'
    if not database.exists():
        return None
    connection = sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        runs = connection.execute('SELECT * FROM runs ORDER BY created').fetchall()
        parents = set()
        for run in runs:
            payload = json.loads(run['manifest']).get('release_plan', {}).get('payload', {})
            if Path(payload.get('spec_path', '')).parent == trial:
                parents.add(run['id'])
        phases = []
        for run in runs:
            manifest = json.loads(run['manifest'])
            if manifest.get('batch_run') not in parents:
                continue
            parts, active = {}, None
            for event in connection.execute('SELECT * FROM events WHERE run_id=? ORDER BY seq', (run['id'],)):
                payload = json.loads(event['payload'])
                if event['kind'] == 'process_started':
                    active = payload['stage']
                    parts.setdefault(active, {'run_id': run['id'], 'pipeline_stage': manifest['step'],
                        'phase': active, 'run_status': run['status'], 'app_ids': [], 'worker_started': False,
                        'artifacts_verified': False, 'started_at': event['at']})
                if active is None:
                    continue
                part = parts[active]
                if event['kind'] == 'cloud_apps':
                    part['app_ids'].extend(payload['owned_ids'])
                if event['kind'] == 'worker_event' and payload.get('kind') == 'worker_ready':
                    part['worker_started'] = True
                if event['kind'] == 'process_exited':
                    part.update(process_seconds=payload['seconds'], returncode=payload['returncode'])
                if event['kind'] == 'artifacts_verified':
                    part['artifacts_verified'] = True
            phases.extend(parts.values())
    finally:
        connection.close()
    assigned = set()
    totals = {}
    for part in phases:
        ids = set(part['app_ids'])
        rows = [r for r in billing_rows if r.get('object_id', r.get('Object ID', r.get('App ID'))) in ids]
        part['posted_cost_usd'] = str(sum((Decimal(r.get('cost', r.get('Cost'))) for r in rows), Decimal(0)))
        part['billing_rows_present'] = bool(rows)
        assigned.update(ids)
        outcome = ('verified' if part['artifacts_verified'] else
                   'failed_or_cancelled' if part.get('returncode') not in (None, 0)
                   or part['run_status'] in ('failed', 'cancelled') else 'in_progress')
        category = f"{part['pipeline_stage']}/{part['phase']}/{outcome}"
        totals[category] = totals.get(category, Decimal(0)) + Decimal(part['posted_cost_usd'])
    unassigned = [r for r in billing_rows if r.get('object_id', r.get('Object ID', r.get('App ID'))) not in assigned]
    return {'by_phase_usd': {key: str(value) for key, value in totals.items()},
            'phases': phases, 'unassigned_rows': unassigned,
            'note': 'Actual posted app charges include build/startup overhead. Verified means controller artifact checks, not semantic or listening approval. Failed workloads may contain checkpointed translations reused later. Verified workload cost alone is not a measured clean run. Missing billing rows may still arrive.'}
