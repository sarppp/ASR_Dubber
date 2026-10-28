import json, sys, tempfile, threading, time, unittest
from pathlib import Path
from unittest.mock import patch
from modal_control.core import Store, digest
from modal_control.batch_plans import prepare_batch, show_batch, run_batch

class BatchTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.store=Store(self.root/'state')
  self.worker=self.root/'worker.py';self.worker.write_text("@app.function(gpu='T4',cpu=1,memory=1024,max_containers=1,retries=0,timeout=5)\ndef work(): pass\n")
  self.input=self.root/'source.txt';self.input.write_text('original')
  self.catalog=self.root/'catalog.json';self.spec=self.root/'spec.json';self.jobs={}
  self.add_job('a');self.add_job('b');self.save()
 def add_job(self,name,derived=None):
  files=['worker.py','source.txt']+([derived] if derived else [])
  self.jobs[name]=dict(cwd=str(self.root),command=[sys.executable,'-m','modal','run','worker.py'],smoke_command=[sys.executable,'-m','modal','run','worker.py','--smoke'],source_files=files,success_files=[name+'.json'],smoke_success_files=[name+'_smoke.json'],max_seconds=5,review=dict(purpose='test',model='model',revision='r',precision='fp16',reason='test',input='source',input_seconds=1,outputs='json',quality_tradeoffs='unknown',cost_estimate='unknown',hardware='T4',max_containers=1,max_total_seconds=10))
 def save(self,steps=None,**kw):
  self.catalog.write_text(json.dumps({'version':1,'jobs':self.jobs}))
  if steps is None:steps=[dict(id=x,job=x,catalog=str(self.catalog),cpu_cores=1,memory_mib=1024,max_attempts=1) for x in self.jobs]
  spec=dict(version=1,steps=steps,max_parallel=2,max_gpu_parallel=2,max_resource_seconds=100,max_wall_seconds=30,estimated_cost_ceiling_usd=10);spec.update(kw);self.spec.write_text(json.dumps(spec))
 def prepare(self):return prepare_batch(self.store,self.spec)
 def fake_stage(self,store,rid,job,smoke,env,cmd):
  files=job['smoke_success_files' if smoke else 'success_files'];out=[]
  for path in files:
   f=Path(job['cwd'])/path;f.write_text('{"status":"complete"}');out.append({'path':str(f),'sha256':digest(f),'bytes':f.stat().st_size})
  return out
 def test_one_release_runs_all_and_records_once(self):
  p=self.prepare();msg='release all'
  with patch('modal_control.core.stage',side_effect=self.fake_stage):rid,rc=run_batch(self.store,p['plan_id'],p['plan_id'],msg)
  self.assertEqual(rc,0);self.assertEqual(self.store.get(rid)['status'],'succeeded');receipt=json.loads((self.store.root/'plans'/f"{p['plan_id']}.consumed").read_text());self.assertEqual(receipt['approval_message'],msg)
  with self.assertRaises(ValueError):show_batch(self.store,p['plan_id'])
 def test_dependencies_bind_produced_input(self):
  self.add_job('b','a.json');self.save([dict(id='a',job='a',catalog=str(self.catalog),cpu_cores=1,memory_mib=1024),dict(id='b',job='b',catalog=str(self.catalog),cpu_cores=1,memory_mib=1024,depends_on=['a'],derived_inputs={'a.json':'a'})])
  p=self.prepare()
  with patch('modal_control.core.stage',side_effect=self.fake_stage):rid,rc=run_batch(self.store,p['plan_id'],p['plan_id'],'release')
  self.assertEqual(rc,0);self.assertTrue((self.root/'b.json').exists())
 def test_retry_within_approved_budget(self):
  self.save([dict(id='a',job='a',catalog=str(self.catalog),cpu_cores=1,memory_mib=1024,max_attempts=2)])
  p=self.prepare();calls=[]
  def stage(store,rid,job,smoke,env,cmd):
   calls.append(smoke)
   if not smoke and calls.count(False)==1:raise RuntimeError('transient')
   return self.fake_stage(store,rid,job,smoke,env,cmd)
  with patch('modal_control.core.stage',side_effect=stage):rid,rc=run_batch(self.store,p['plan_id'],p['plan_id'],'release')
  self.assertEqual(rc,0);self.assertEqual(calls,[True,False,False])
 def test_rejects_scope_changes_and_invalid_limits(self):
  p=self.prepare();self.input.write_text('changed')
  with self.assertRaisesRegex(ValueError,'changed'):run_batch(self.store,p['plan_id'],p['plan_id'],'release')
  self.input.write_text('original');self.save(max_parallel=3)
  with self.assertRaisesRegex(ValueError,'concurrency'):self.prepare()
  self.save(estimated_cost_ceiling_usd=0.00001)
  with self.assertRaisesRegex(ValueError,'cost ceiling'):self.prepare()
 def test_no_release_without_exact_id_and_message(self):
  p=self.prepare()
  for confirmation,message in [('wrong','release'),(p['plan_id'],'')]:
   with self.assertRaises(ValueError):run_batch(self.store,p['plan_id'],confirmation,message)
  self.assertEqual(show_batch(self.store,p['plan_id'])['plan_id'],p['plan_id'])
 def test_rejects_unbounded_retries_and_unowned_derived_inputs(self):
  self.save([dict(id='a',job='a',catalog=str(self.catalog),cpu_cores=1,memory_mib=1024,max_attempts=4)])
  with self.assertRaisesRegex(ValueError,'max_attempts'):self.prepare()
  self.add_job('b','a.json');self.save([dict(id='a',job='a',catalog=str(self.catalog),cpu_cores=1,memory_mib=1024),dict(id='b',job='b',catalog=str(self.catalog),cpu_cores=1,memory_mib=1024,depends_on=[],derived_inputs={'a.json':'a'})])
  with self.assertRaisesRegex(ValueError,'producer outputs'):self.prepare()
 def test_partial_outputs_prevent_retry(self):
  self.save([dict(id='a',job='a',catalog=str(self.catalog),cpu_cores=1,memory_mib=1024,max_attempts=2)]);p=self.prepare();calls=[]
  def stage(store,rid,job,smoke,env,cmd):
   calls.append(smoke)
   if not smoke:
    (self.root/'a.json').write_text('partial');raise RuntimeError('failed')
   return self.fake_stage(store,rid,job,smoke,env,cmd)
  with patch('modal_control.core.stage',side_effect=stage):rid,rc=run_batch(self.store,p['plan_id'],p['plan_id'],'release')
  self.assertEqual(rc,1);self.assertEqual(calls,[True,False]);self.assertEqual((self.root/'a.json').read_text(),'partial')
 def test_parallel_never_exceeds_two_and_gpu_limit_one(self):
  self.add_job('c');self.add_job('d');self.save(max_gpu_parallel=1)
  plan=self.prepare();lock=threading.Lock();active=0;peak=0
  def measured(store,rid,job,smoke,env,cmd):
   nonlocal active,peak
   with lock:active+=1;peak=max(peak,active)
   time.sleep(.02)
   result=self.fake_stage(store,rid,job,smoke,env,cmd)
   with lock:active-=1
   return result
  with patch('modal_control.core.stage',side_effect=measured):rid,rc=run_batch(self.store,plan['plan_id'],plan['plan_id'],'release')
  self.assertEqual(rc,0);self.assertEqual(peak,1)
 def test_two_gpu_slots_are_used_but_never_exceeded(self):
  self.add_job('c');self.add_job('d');self.save(max_gpu_parallel=2)
  plan=self.prepare();lock=threading.Lock();active=0;peak=0
  def measured(store,rid,job,smoke,env,cmd):
   nonlocal active,peak
   with lock:active+=1;peak=max(peak,active)
   time.sleep(.03)
   result=self.fake_stage(store,rid,job,smoke,env,cmd)
   with lock:active-=1
   return result
  with patch('modal_control.core.stage',side_effect=measured):rid,rc=run_batch(self.store,plan['plan_id'],plan['plan_id'],'release')
  self.assertEqual(rc,0);self.assertEqual(peak,2)
 def test_changed_derived_input_stops_before_workload(self):
  self.add_job('b','a.json');self.save([dict(id='a',job='a',catalog=str(self.catalog),cpu_cores=1,memory_mib=1024),dict(id='b',job='b',catalog=str(self.catalog),cpu_cores=1,memory_mib=1024,depends_on=['a'],derived_inputs={'a.json':'a'})]);p=self.prepare()
  def modified(store,rid,job,smoke,env,cmd):
   result=self.fake_stage(store,rid,job,smoke,env,cmd)
   if job['success_files']==['b.json'] and smoke:(self.root/'a.json').write_text('changed')
   return result
  with patch('modal_control.core.stage',side_effect=modified):rid,rc=run_batch(self.store,p['plan_id'],p['plan_id'],'release')
  self.assertEqual(rc,1);self.assertFalse((self.root/'b.json').exists())
 def test_parent_cancellation_propagates_to_active_child(self):
  self.save([dict(id='a',job='a',catalog=str(self.catalog),cpu_cores=1,memory_mib=1024)])
  plan=self.prepare();started=threading.Event();result=[]
  def stage(store,rid,job,smoke,env,cmd):
   started.set();cancel=store.root/'runs'/rid/'cancel.request'
   for _ in range(500):
    if cancel.exists():raise RuntimeError('cancelled')
    time.sleep(.01)
   raise RuntimeError('test timed out')
  def execute():
   with patch('modal_control.core.stage',side_effect=stage):result.append(run_batch(self.store,plan['plan_id'],plan['plan_id'],'release'))
  thread=threading.Thread(target=execute);thread.start();self.assertTrue(started.wait(5))
  parent=next(r for r in self.store.overview()['runs'] if r['job'].startswith('batch:'))
  (self.store.root/'runs'/parent['id']/'cancel.request').write_text('stop')
  thread.join(5);self.assertFalse(thread.is_alive());self.assertEqual(result[0][1],1)
  self.assertEqual(self.store.get(parent['id'])['status'],'cancelled')
