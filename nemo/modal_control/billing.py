"""Read-only posted billing snapshots; never estimate credits or conceal stale data."""
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from modal_control.core import atomic, clean, utc


def money(value):
    if isinstance(value, bool):
        raise ValueError('Invalid billing amount')
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('Invalid billing amount') from exc
    if not result.is_finite():
        raise ValueError('Non-finite billing amount')
    return result


def normalize_summary(data):
    adjustments = data.get('adjustments', {})
    credit = adjustments.get('credits')
    return {'metered_usd': str(money(data['metered_cost'])),
            'charged_usd': str(money(data['billed_cost'])),
            'credits_used_usd': str(-money(credit)) if credit is not None else None,
            'credits_remaining_usd': None}


def owned_app_ids(store, run_id):
    with store.db() as db:
        children = [row['id'] for row in db.execute('SELECT id,manifest FROM runs')
                    if json.loads(row['manifest']).get('batch_run') == run_id]
        ids = [run_id, *children]
        marks = ','.join('?' for _ in ids)
        events = db.execute(f'SELECT payload FROM events WHERE kind=? AND run_id IN ({marks})',
                            ['cloud_apps', *ids]).fetchall()
    return {app for event in events for app in json.loads(event['payload']).get('owned_ids', [])}


def app_total(rows, app_ids):
    if not isinstance(rows, list):
        raise ValueError('Billing report must contain rows')
    total = Decimal(0)
    selected = []
    for row in rows:
        if row.get('object_id', row.get('Object ID', row.get('App ID'))) in app_ids:
            total += money(row.get('cost', row.get('Cost')))
            selected.append(row)
    return str(total), selected


class BillingMonitor:
    """Snapshot before execution, every two minutes, and after shutdown.

    Unavailable billing never changes a job's status or becomes a fabricated zero.
    Workspace deltas may include other apps; owned-app rows are separate.
    """
    def __init__(self, store, run_id, catalog, interval=120):
        self.store, self.run_id = store, run_id
        self.command = catalog.get('billing_command', [sys.executable, '-m', 'modal'])
        self.env = os.environ.copy()
        if catalog.get('env_file'):
            from dotenv import dotenv_values
            self.env.update({k: v for k, v in dotenv_values(catalog['env_file']).items() if v is not None})
        self.interval = max(30, interval)
        self.baseline = None
        self.stop = threading.Event()
        self.thread = None

    def read(self, kind):
        result = subprocess.run([*self.command, 'billing', kind, '--for', 'this month', '--json'],
                                env=self.env, capture_output=True, text=True, timeout=12)
        if result.returncode:
            raise RuntimeError(clean(result.stderr[-1200:]) or f'Billing {kind} exited {result.returncode}')
        return json.loads(result.stdout)

    def capture(self, phase):
        record = {'run_id': self.run_id, 'phase': phase, 'observed_at': utc(),
                  'period': datetime.now(timezone.utc).strftime('%Y-%m'),
                  'status': 'unavailable', 'billing_final': False,
                  'note': 'Posted charges can lag. Workspace deltas may include unrelated jobs. Remaining credit is unknown unless supplied by the provider.'}
        try:
            record['summary'] = self.read('summary')
            record['amounts'] = normalize_summary(record['summary'])
            record['status'] = 'available'
        except Exception as exc:
            record['error'] = clean(str(exc))
        try:
            total, rows = app_total(self.read('report'), owned_app_ids(self.store, self.run_id))
            record.update(owned_app_cost_usd=total, owned_app_rows=rows)
        except Exception as exc:
            record.update(owned_app_cost_usd=None, report_error=clean(str(exc)))
        if phase == 'start':
            self.baseline = record
        record['baseline'] = None if self.baseline is None else {
            k: self.baseline.get(k) for k in ('observed_at', 'period', 'amounts', 'status')}
        record['workspace_delta'] = None
        if (record['status'] == 'available' and self.baseline and
                self.baseline['status'] == 'available' and record['period'] == self.baseline['period']):
            record['workspace_delta'] = {
                key: str(money(value)-money(self.baseline['amounts'][key]))
                if value is not None and self.baseline['amounts'][key] is not None else None
                for key, value in record['amounts'].items()}
        event = self.store.event(self.run_id, 'billing_snapshot', record)
        atomic(self.store.root / 'runs' / self.run_id / f'billing-{event["seq"]}.json', record)
        atomic(self.store.root / 'runs' / self.run_id / 'billing_latest.json', record)
        atomic(self.store.root / 'latest_billing.json', record)
        if record['status'] == 'available':
            amounts = record['amounts']
            credit = 'unknown' if amounts['credits_used_usd'] is None else '$' + amounts['credits_used_usd']
            delta = record['workspace_delta']
            change = '' if delta is None or delta['credits_used_usd'] is None else f'; credit change ${delta["credits_used_usd"]}'
            print(f'[billing/{phase}] credits used {credit}; cash charged ${amounts["charged_usd"]}{change}; posted figures may lag', flush=True)
        else:
            print(f'[billing/{phase}] unavailable: {record["error"]}; no amount assumed', flush=True)
        return record

    def start(self):
        self.capture('start')
        def refresh():
            while not self.stop.wait(self.interval):
                self.capture('progress')
        self.thread = threading.Thread(target=refresh, daemon=True)
        self.thread.start()
        return self

    def finish(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=30)
        if self.thread and self.thread.is_alive():
            self.store.event(self.run_id, 'billing_unavailable', {'phase': 'end', 'reason': 'Previous billing request still pending'})
            return
        self.capture('end')
