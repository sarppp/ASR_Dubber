"""Explicit, bounded batch plans. Planning never contacts Modal."""
from __future__ import annotations
import ast, hashlib, json, math, re, time
from pathlib import Path
from .core import atomic, digest, load_catalog
from .plans import canonical, positive, validate_job

BATCH_POLICY = 'modal-batch-release-v1'

def batch_identity(payload):
    return 'mb-' + hashlib.sha256(canonical(payload).encode()).hexdigest()[:32]

def _absolute(job, path):
    return str((Path(job['cwd']) / path).resolve())

def _resource_upper_limit(value, field):
    """Use the configured upper limit of a literal (request, limit) pair."""
    if not isinstance(value, tuple):
        return value
    if len(value) != 2:
        raise ValueError(f'Worker {field} must be a (request, limit) pair')
    request, limit = value
    try:
        positive(request, f'worker {field} request')
        positive(limit, f'worker {field} limit')
    except OverflowError:
        raise ValueError(f'Worker {field} pair must contain finite numbers') from None
    if request > limit:
        raise ValueError(f'Worker {field} request must not exceed its limit')
    return limit

def prepare_batch(store, spec_path, ttl_seconds=3600):
    positive(ttl_seconds, 'ttl_seconds')
    if ttl_seconds > 3600:
        raise ValueError('Plan TTL cannot exceed one hour')
    spec_file = Path(spec_path).resolve()
    spec = json.loads(spec_file.read_text())
    if spec.get('version') != 1 or not isinstance(spec.get('steps'), list) or not spec['steps']:
        raise ValueError('A nonempty version 1 step list is required')
    for name in ('max_parallel', 'max_gpu_parallel', 'max_resource_seconds', 'max_wall_seconds', 'estimated_cost_ceiling_usd'):
        positive(spec.get(name), name)
    if (isinstance(spec['max_parallel'], bool) or not isinstance(spec['max_parallel'], int)
            or isinstance(spec['max_gpu_parallel'], bool) or not isinstance(spec['max_gpu_parallel'], int)
            or spec['max_parallel'] not in (1, 2) or spec['max_gpu_parallel'] not in (1, 2)
            or spec['max_gpu_parallel'] > spec['max_parallel']):
        raise ValueError('Global concurrency cannot exceed two slots or two GPUs')
    seen = {}
    input_hashes = {}
    catalogs = {}
    prepared = []
    all_output_paths = set()
    worst_seconds = 0
    estimated_ceiling = 0.0
    for step in spec['steps']:
        name = step.get('id')
        if not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9_.-]+', name) or name in seen:
            raise ValueError('Invalid or duplicate step ID')
        deps = step.get('depends_on', [])
        if not isinstance(deps, list) or any(d not in seen for d in deps):
            raise ValueError('Dependencies must be earlier step IDs')
        attempts = step.get('max_attempts', 1)
        if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 3:
            raise ValueError('max_attempts must be between one and three')
        catalog_path = str(Path(step['catalog']).resolve())
        catalog = load_catalog(catalog_path)
        if catalog.get('submissions_disabled'):
            raise ValueError('Disabled catalogs cannot be used')
        jobname = step['job']
        job = catalog['jobs'][jobname]
        validate_job(job)
        cpu = step.get('cpu_cores')
        memory = step.get('memory_mib')
        positive(cpu, 'cpu_cores')
        positive(memory, 'memory_mib')
        for worker_name in {job['command'][4], job['smoke_command'][4]}:
            worker = Path(job['cwd']) / worker_name
            tree = ast.parse(worker.read_text())
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ('function', 'cls')]
            for call in calls:
                args = {k.arg: k.value for k in call.keywords}
                try:
                    actual_cpu = ast.literal_eval(args['cpu'])
                    actual_memory = ast.literal_eval(args['memory'])
                except (KeyError, ValueError, TypeError):
                    raise ValueError('Batch workers must declare literal CPU and memory limits')
                actual_cpu = _resource_upper_limit(actual_cpu, 'cpu')
                actual_memory = _resource_upper_limit(actual_memory, 'memory')
                if actual_cpu != cpu or actual_memory != memory:
                    raise ValueError('Declared CPU or memory does not match worker')
        catalogs[catalog_path] = digest(catalog_path)
        declared = {_absolute(job, p) for p in job['source_files']}
        derived = step.get('derived_inputs', {})
        if not isinstance(derived, dict):
            raise ValueError('derived_inputs must map source path to preceding producer ID')
        derived_abs = {}
        for path, producer in derived.items():
            target = _absolute(job, path)
            if target not in declared or producer not in deps or target not in seen[producer]:
                raise ValueError('Derived inputs must be declared producer outputs and dependencies')
            if Path(target).suffix == '.py':
                raise ValueError('Executable source code cannot be a derived input')
            derived_abs[target] = producer
        for path in declared - set(derived_abs):
            file = Path(path)
            if not file.is_file():
                raise ValueError(f'Missing input or source: {path}')
            value = digest(file)
            if path in input_hashes and input_hashes[path] != value:
                raise ValueError('Input changed while preparing plan')
            input_hashes[path] = value
        outputs = [_absolute(job, p) for p in job['success_files']]
        all_step_outputs = outputs + [_absolute(job, p) for p in job['smoke_success_files']]
        if len(all_step_outputs) != len(set(all_step_outputs)) or all_output_paths.intersection(all_step_outputs):
            raise ValueError('Output paths must be unique across steps')
        all_output_paths.update(all_step_outputs)
        seen[name] = outputs
        worst_seconds += 2 * job['max_seconds'] * attempts
        # Pinned public rates used for a transparent planning estimate, not a
        # provider-enforced spending cap. Resource pairs use their upper limits,
        # with CPU cores conservatively charged as physical cores here.
        seconds = 2 * job['max_seconds'] * attempts
        gpu_rate = 0.000164 if job['review']['hardware'] == 'T4' else 0
        estimated_ceiling += seconds * (gpu_rate + cpu * 0.0000131 + memory / 1024 * 0.00000222)
        prepared.append({'id': name, 'depends_on': deps, 'max_attempts': attempts,
                         'cpu_cores': cpu, 'memory_mib': memory,
                         'catalog_path': catalog_path, 'job_name': jobname,
                         'configuration': job, 'derived_inputs': derived_abs, 'outputs': outputs})
    if worst_seconds > spec['max_resource_seconds']:
        raise ValueError('Resource-second ceiling is smaller than bounded stage deadlines')
    if estimated_ceiling > spec['estimated_cost_ceiling_usd']:
        raise ValueError('Estimated compute ceiling exceeds the reviewed cost ceiling')
    now = time.time()
    payload = {'policy': BATCH_POLICY, 'created': now, 'expires': now + ttl_seconds,
               'scope': spec, 'spec_path': str(spec_file), 'spec_sha256': digest(spec_file),
               'catalogs': catalogs, 'sources': [{'path': p, 'sha256': h} for p, h in sorted(input_hashes.items())],
               'controller_sources': [{'path': str(p.resolve()), 'sha256': digest(p)} for p in sorted(Path(__file__).parent.glob('*.py'))],
               'steps': prepared, 'worst_case_stage_seconds': worst_seconds,
               'estimated_compute_ceiling_usd': round(estimated_ceiling, 6),
               'budget_note': 'Estimated compute ceiling is not a verified provider billing cap; startup, storage and network may be additional.'}
    plan = {'plan_id': batch_identity(payload), 'payload': payload}
    atomic(store.root / 'plans' / (plan['plan_id'] + '.json'), plan)
    return plan

def show_batch(store, plan_id):
    if not re.fullmatch(r'mb-[0-9a-f]{32}', plan_id):
        raise ValueError('Invalid batch plan ID')
    plan = json.loads((store.root / 'plans' / (plan_id + '.json')).read_text())
    if plan.get('plan_id') != plan_id or batch_identity(plan['payload']) != plan_id:
        raise ValueError('Batch integrity check failed')
    if plan['payload']['policy'] != BATCH_POLICY or time.time() >= plan['payload']['expires']:
        raise ValueError('Batch plan expired or policy changed')
    if (store.root / 'plans' / (plan_id + '.consumed')).exists():
        raise ValueError('Batch plan already released')
    return plan

def release_batch(store, plan_id, confirmation, approval_message):
    if confirmation != plan_id or not isinstance(approval_message, str) or not approval_message.strip():
        raise ValueError('Exact batch confirmation and actual release message required')
    plan = show_batch(store, plan_id)
    p = plan['payload']
    if digest(p['spec_path']) != p['spec_sha256']:
        raise ValueError('Batch specification changed')
    for path, sha in p['catalogs'].items():
        if digest(path) != sha:
            raise ValueError('Batch catalog changed')
    for item in p['sources'] + p['controller_sources']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Batch source or input changed')
    for step in p['steps']:
        validate_job(step['configuration'])
    with (store.root / 'plans' / (plan_id + '.consumed')).open('x') as f:
        json.dump({'plan_id': plan_id, 'released_at': time.time(), 'approval_message': approval_message}, f)
    return plan


def run_batch(store, plan_id, confirmation, approval_message):
    """Release once and execute the exact reviewed DAG, with bounded retries.

    The caller must obtain the user's release of this exact batch. Current
    session tool policies may be stricter than this repository implementation.
    """
    import concurrent.futures
    import os
    import shutil
    import threading
    from .core import stage, utc
    plan = release_batch(store, plan_id, confirmation, approval_message)
    payload = plan['payload']
    scope = payload['scope']
    batch_run = store.create('batch:' + plan_id, {'release_plan': plan,
                             'approval_receipt': str(store.root / 'plans' / (plan_id + '.consumed'))})
    store.event(batch_run, 'batch_released', {'plan_id': plan_id, 'steps': [s['id'] for s in payload['steps']]})
    store.state(batch_run, 'running')
    guard = threading.Lock()
    active = {}
    results = {}
    artifact_hashes = {}
    started = time.monotonic()
    stopping = False
    cancelled_by_user = False

    def verify_static():
        if digest(payload['spec_path']) != payload['spec_sha256']:
            raise RuntimeError('Batch specification changed')
        for path, sha in payload['catalogs'].items():
            if digest(path) != sha:
                raise RuntimeError('Batch catalog changed')
        for item in payload['sources'] + payload['controller_sources']:
            if digest(item['path']) != item['sha256']:
                raise RuntimeError('Batch source or input changed')

    def execute_step(step):
        verify_static()
        job = dict(step['configuration'], _approved_hardware=step['configuration']['review']['hardware'])
        catalog = load_catalog(step['catalog_path'])
        if catalog['jobs'][step['job_name']] != step['configuration']:
            raise RuntimeError('Batch job configuration changed')
        def verify_derived():
            with guard:
                for path, producer in step['derived_inputs'].items():
                    expected = artifact_hashes.get((producer, path))
                    if expected is None or not Path(path).is_file() or digest(path) != expected:
                        raise RuntimeError('Produced input missing or changed')
        verify_derived()
        for key in ('smoke_success_files', 'success_files'):
            for filename in job[key]:
                if (Path(job['cwd']) / filename).exists():
                    raise FileExistsError('Output already exists; refusing to overwrite')
        child = store.create('batch-step:' + step['id'], {'batch_run': batch_run, 'plan_id': plan_id,
                             'step': step['id'], 'configuration': job})
        with guard:
            active[step['id']] = child
        env = os.environ.copy()
        env['MODAL_CONTROL_RUN_ID'] = child
        env['PYTHONUNBUFFERED'] = '1'
        if catalog.get('env_file'):
            from dotenv import dotenv_values
            env.update({k: v for k, v in dotenv_values(catalog['env_file']).items() if v is not None})
        modal_command = catalog.get('modal_command', [os.sys.executable, '-m', 'modal'])
        try:
            outputs = None
            for smoke in (True, False):
                errors = []
                for attempt in range(1, step['max_attempts'] + 1):
                    verify_static()
                    verify_derived()
                    if stopping:
                        raise RuntimeError('batch_stopped')
                    store.event(child, 'batch_attempt', {'stage': 'smoke' if smoke else 'workload', 'attempt': attempt})
                    try:
                        outputs = stage(store, child, job, smoke, env, modal_command)
                        verify_static()
                        verify_derived()
                        break
                    except Exception as exc:
                        errors.append(str(exc))
                        if attempt >= step['max_attempts'] or str(exc) in ('cancelled', 'batch_stopped'):
                            raise
                        # Never delete or overwrite partial results to make a retry work.
                        files = job['smoke_success_files' if smoke else 'success_files']
                        if any((Path(job['cwd']) / f).exists() for f in files):
                            raise RuntimeError('Partial output exists; automatic retry stopped') from exc
                        store.event(child, 'bounded_retry', {'attempt': attempt, 'error': str(exc)})
            archive = store.root / 'runs' / child / 'artifacts'
            archive.mkdir(exist_ok=True)
            inventory = []
            for index, item in enumerate(outputs or []):
                destination = archive / f"{index:03d}-{Path(item['path']).name}"
                shutil.copy2(item['path'], destination)
                if digest(destination) != item['sha256']:
                    raise RuntimeError('Output changed while archiving')
                inventory.append({'path': str(destination.relative_to(archive)), 'bytes': destination.stat().st_size,
                                  'sha256': item['sha256']})
            for root in job.get('artifact_roots', []):
                source = Path(job['cwd']) / root
                if source.is_dir():
                    shutil.copytree(source, archive / source.name, dirs_exist_ok=True)
            for path in archive.rglob('*'):
                if path.is_file() and not any(x['path'] == str(path.relative_to(archive)) for x in inventory):
                    inventory.append({'path': str(path.relative_to(archive)), 'bytes': path.stat().st_size,
                                      'sha256': digest(path)})
            atomic(store.root / 'runs' / child / 'artifact_manifest.json', inventory)
            with guard:
                for item in outputs or []:
                    artifact_hashes[(step['id'], str(Path(item['path']).resolve()))] = item['sha256']
            store.state(child, 'succeeded')
            return {'status': 'succeeded', 'run_id': child, 'outputs': outputs}
        except Exception as exc:
            store.alert(child, 'batch_step_failed', str(exc))
            store.state(child, 'failed', error=str(exc))
            raise
        finally:
            with guard:
                active.pop(step['id'], None)

    steps = {s['id']: s for s in payload['steps']}
    pending = list(steps)
    running = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=scope['max_parallel']) as pool:
            while pending or running:
                cancelled = (store.root / 'runs' / batch_run / 'cancel.request').exists()
                if cancelled or time.monotonic() - started > scope['max_wall_seconds']:
                    stopping = True
                    cancelled_by_user = cancelled_by_user or cancelled
                    store.alert(batch_run, 'batch_cancelled' if cancelled else 'batch_deadline',
                                'Batch cancelled' if cancelled else 'Batch wall deadline exceeded')
                    with guard:
                        for child in active.values():
                            (store.root / 'runs' / child / 'cancel.request').write_text(utc())
                if not stopping:
                    for sid in list(pending):
                        s = steps[sid]
                        if not all(d in results and results[d]['status'] == 'succeeded' for d in s['depends_on']):
                            continue
                        if len(running) >= scope['max_parallel']:
                            break
                        gpu_active = sum(steps[v]['configuration']['review']['hardware'] == 'T4' for v in running.values())
                        if s['configuration']['review']['hardware'] == 'T4' and gpu_active >= scope['max_gpu_parallel']:
                            continue
                        pending.remove(sid)
                        running[pool.submit(execute_step, s)] = sid
                if not running:
                    if pending and not stopping:
                        raise RuntimeError('No runnable steps; dependency failure')
                    break
                done, _ = concurrent.futures.wait(running, timeout=1, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    sid = running.pop(future)
                    try:
                        results[sid] = future.result()
                        store.event(batch_run, 'batch_step_succeeded', {'step': sid, 'run_id': results[sid]['run_id']})
                    except Exception as exc:
                        results[sid] = {'status': 'failed', 'error': str(exc)}
                        stopping = True
                        store.alert(batch_run, 'batch_step_failed', f'{sid}: {exc}')
        status = ('cancelled' if cancelled_by_user else 'failed') if stopping or pending or any(v['status'] != 'succeeded' for v in results.values()) else 'succeeded'
        store.state(batch_run, status, results=results, pending=pending)
        return batch_run, 0 if status == 'succeeded' else 1
    except Exception as exc:
        store.alert(batch_run, 'batch_failed', str(exc))
        store.state(batch_run, 'failed', error=str(exc), results=results, pending=pending)
        return batch_run, 1
