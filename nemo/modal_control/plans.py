"""Immutable, expiring, single-use human release plans. No cloud calls here."""
import ast
import hashlib
import json
import math
import re
import time
from pathlib import Path
from .core import atomic, digest, load_catalog

POLICY = 'modal-human-release-v1'

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def identity(payload):
    return 'mp-' + hashlib.sha256(canonical(payload).encode()).hexdigest()[:32]

def positive(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{field} must be a positive finite number')

def validate_job(job):
    review = job.get('review', {})
    for field in ('purpose', 'model', 'revision', 'precision', 'reason', 'input', 'outputs', 'quality_tradeoffs', 'cost_estimate'):
        if not isinstance(review.get(field), str) or not review[field].strip():
            raise ValueError(f'review.{field} is required; disclose unknown estimates explicitly')
    hardware = review.get('hardware')
    if hardware not in ('CPU', 'T4'):
        raise ValueError('Only CPU or T4 is permitted; no GPU fallback')
    for field in ('input_seconds', 'max_containers', 'max_total_seconds'):
        positive(review.get(field), 'review.' + field)
    if review['max_containers'] != 1:
        raise ValueError('Only one container per function is allowed')
    positive(job.get('max_seconds'), 'max_seconds')
    if review['max_total_seconds'] < 2 * job['max_seconds']:
        raise ValueError('max_total_seconds must cover both smoke and workload deadlines')
    sources = {(Path(job['cwd']) / f).resolve() for f in job.get('source_files', [])}
    if not sources:
        raise ValueError('source_files must pin all worker code, imported helpers and input/configuration files')
    # Direct Modal entrypoints only; shell wrappers and arbitrary local commands
    # cannot hide unreviewed hardware allocation behind a CPU/T4 declaration.
    workers = set()
    for key in ('smoke_command', 'command'):
        cmd = job[key]
        if len(cmd) < 5 or cmd[1:4] != ['-m', 'modal', 'run'] or not cmd[4].endswith('.py'):
            raise ValueError(f'{key}: use python -m modal run worker.py; wrappers/detached runs are forbidden')
        if '--detach' in cmd:
            raise ValueError('Detached runs are forbidden')
        worker = (Path(job['cwd']) / cmd[4]).resolve()
        if worker not in sources:
            raise ValueError('Worker must be included in source_files')
        workers.add(worker)
    for worker in workers:
        tree = ast.parse(worker.read_text())
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ('function', 'cls')]
        if not functions:
            raise ValueError('No explicit Modal function configuration found')
        for node in functions:
            kwargs = {k.arg: k.value for k in node.keywords}
            if None in kwargs:
                raise ValueError('Dynamic Modal function settings are forbidden')
            def literal(key, default=None):
                try:
                    return ast.literal_eval(kwargs[key]) if key in kwargs else default
                except (ValueError, TypeError):
                    raise ValueError(f'Modal {key} must be literal, not computed or environment-selected')
            expected = None if hardware == 'CPU' else 'T4'
            if literal('gpu') != expected:
                raise ValueError(f'Worker GPU must exactly match {hardware}')
            if literal('max_containers') != 1 or literal('retries') != 0:
                raise ValueError('Worker must declare max_containers=1 and retries=0')
            timeout = literal('timeout')
            positive(timeout, 'worker timeout')
            if timeout > job['max_seconds']:
                raise ValueError('Worker timeout exceeds approved stage deadline')
    if job.get('env'):
        # Environment-based selection makes the plan harder to review. Put
        # non-secret selection into the pinned worker/config instead.
        raise ValueError('Job env overrides are not permitted in release plans')
    return review

def prepare_plan(store, catalog_path, name, ttl_seconds=1800):
    positive(ttl_seconds, 'ttl_seconds')
    if ttl_seconds > 3600:
        raise ValueError('Plan TTL cannot exceed one hour')
    path = Path(catalog_path).resolve()
    catalog = load_catalog(path)
    if catalog.get('submissions_disabled'):
        raise ValueError('Catalog submissions are disabled; prepare a separate reviewed job catalog')
    job = catalog['jobs'][name]
    validate_job(job)
    sources = [{'path': str((Path(job['cwd']) / f).resolve()), 'sha256': digest(Path(job['cwd']) / f)} for f in job['source_files']]
    now = time.time()
    payload = dict(policy=POLICY, created=now, expires=now + ttl_seconds,
                   catalog_path=str(path), catalog_sha256=digest(path), job=name,
                   configuration=job, sources=sources,
                   controller_sources=[{'path':str(p.resolve()), 'sha256':digest(p)} for p in sorted(Path(__file__).parent.glob('*.py'))],
                   stages=['smoke', 'workload'], automatic_retries=0)
    plan = dict(plan_id=identity(payload), payload=payload)
    atomic(store.root / 'plans' / (plan['plan_id'] + '.json'), plan)
    return plan

def show_plan(store, plan_id):
    if not re.fullmatch(r'mp-[0-9a-f]{32}', plan_id):
        raise ValueError('Invalid plan id')
    plan = json.loads((store.root / 'plans' / (plan_id + '.json')).read_text())
    payload = plan['payload']
    if plan.get('plan_id') != plan_id or identity(payload) != plan_id:
        raise ValueError('Plan integrity check failed')
    if payload.get('policy') != POLICY or time.time() >= payload['expires']:
        raise ValueError('Plan expired or policy changed; prepare a new plan')
    if (store.root / 'plans' / (plan_id + '.consumed')).exists():
        raise ValueError('Plan already consumed; no automatic retry')
    return plan

def release_plan(store, plan_id, confirmation, approval_message):
    if not isinstance(approval_message, str) or not approval_message.strip():
        raise ValueError("The explicit user approval message must be recorded")
    if confirmation != plan_id:
        raise ValueError('Exact plan-id confirmation required from the human')
    plan = show_plan(store, plan_id)
    payload = plan['payload']
    if digest(payload['catalog_path']) != payload['catalog_sha256']:
        raise ValueError('Catalog changed; prepare a new plan')
    for source in payload['sources'] + payload['controller_sources']:
        if digest(source['path']) != source['sha256']:
            raise ValueError('Source/input/controller changed; prepare a new plan')
    validate_job(payload['configuration'])
    # O_EXCL gives exactly one winner even across simultaneous processes.
    receipt = store.root / 'plans' / (plan_id + '.consumed')
    with receipt.open('x') as f:
        json.dump({'plan_id': plan_id, 'released_at': time.time(), 'approval_message': approval_message}, f)
    return plan


def prepare_batch_plan(store, spec_path, ttl_seconds=3600):
    """Prepare one bounded multi-step plan; no Modal calls occur here."""
    from .batch_plans import prepare_batch
    return prepare_batch(store, spec_path, ttl_seconds=ttl_seconds)


def show_batch_plan(store, plan_id):
    from .batch_plans import show_batch
    return show_batch(store, plan_id)


def release_batch_plan(store, plan_id, confirmation, approval_message):
    from .batch_plans import release_batch
    return release_batch(store, plan_id, confirmation, approval_message)
