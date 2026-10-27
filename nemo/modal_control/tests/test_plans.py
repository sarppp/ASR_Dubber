import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from modal_control.core import Store, run_job
from modal_control.plans import prepare_plan, show_plan, release_plan


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = Store(self.root / 'state')
        self.worker = self.root / 'worker.py'
        self.worker.write_text("@app.function(gpu='T4', max_containers=1, retries=0, timeout=60)\ndef work(): pass\n")
        self.catalog = self.root / 'catalog.json'
        self.job = dict(cwd=str(self.root), command=[sys.executable,'-m','modal','run','worker.py'],
                        smoke_command=[sys.executable,'-m','modal','run','worker.py','--smoke'],
                        source_files=['worker.py'], success_files=['done.json'], smoke_success_files=['smoke.json'], max_seconds=60,
                        review=dict(purpose='German dub', model='Qwen', revision='pinned', precision='bf16', reason='Voice cloning',
                                    input='video first 10min', outputs='dub.mp4', quality_tradeoffs='Unverified', cost_estimate='Unknown',
                                    hardware='T4', input_seconds=600, max_containers=1, max_total_seconds=120))
        self.save()

    def save(self):
        self.catalog.write_text(json.dumps(dict(version=1,jobs={'dub':self.job})))

    def plan(self):
        return prepare_plan(self.store,self.catalog,'dub')

    def test_direct_run_cannot_spawn(self):
        with patch('modal_control.core.subprocess.Popen') as popen:
            with self.assertRaisesRegex(ValueError,'Direct runs'):
                run_job(self.store,self.catalog,'dub')
            popen.assert_not_called()

    def test_prepare_never_spawns(self):
        with patch('modal_control.core.subprocess.Popen') as popen:
            plan=self.plan()
            self.assertEqual(show_plan(self.store,plan['plan_id']),plan)
            popen.assert_not_called()

    def test_explicit_go_is_saved_and_single_use(self):
        plan=self.plan(); pid=plan['plan_id']
        with patch('modal_control.core._run_approved_job',return_value=('run',0)) as execute:
            self.assertEqual(run_job(self.store,self.catalog,'dub',plan_id=pid,confirmation=pid,approval_message='release it'),('run',0))
            self.assertEqual(execute.call_args.kwargs['plan']['approval_message'],'release it')
        with self.assertRaisesRegex(ValueError,'consumed'):
            release_plan(self.store,pid,pid,'release it')

    def test_confirmation_and_go_required(self):
        pid=self.plan()['plan_id']
        for confirm,message in [('wrong','release it'),(pid,'')]:
            with self.assertRaises(ValueError): release_plan(self.store,pid,confirm,message)
        show_plan(self.store,pid)

    def test_stale_source_rejected(self):
        pid=self.plan()['plan_id'];self.worker.write_text('# changed')
        with self.assertRaisesRegex(ValueError,'changed'):release_plan(self.store,pid,pid,'release it')

    def test_stale_catalog_rejected(self):
        pid=self.plan()['plan_id'];self.job['review']['model']='different';self.save()
        with self.assertRaisesRegex(ValueError,'changed'):release_plan(self.store,pid,pid,'release it')

    def test_expired_plan_rejected(self):
        plan=self.plan()
        with patch('modal_control.plans.time.time',return_value=plan['payload']['expires']+1):
            with self.assertRaisesRegex(ValueError,'expired'):show_plan(self.store,plan['plan_id'])

    def test_tampering_rejected(self):
        plan=self.plan();pid=plan['plan_id'];plan['payload']['configuration']['review']['model']='changed'
        (self.store.root/'plans'/f'{pid}.json').write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError,'integrity'):show_plan(self.store,pid)

    def test_gpu_mismatch_and_dynamic_gpu_rejected(self):
        for gpu in ["'A10G'", "'L4'", "['T4','A10G']", "os.getenv('GPU','T4')"]:
            self.worker.write_text(f'@app.function(gpu={gpu}, max_containers=1, retries=0, timeout=60)\ndef work():pass\n')
            with self.assertRaises(ValueError):self.plan()

    def test_missing_limits_and_review_rejected(self):
        for key in ['max_total_seconds','model','hardware']:
            value=self.job['review'].pop(key);self.save()
            with self.assertRaises(ValueError):self.plan()
            self.job['review'][key]=value
        self.save()

    def test_shell_wrapper_rejected(self):
        self.job['command']=['bash','worker.sh'];self.save()
        with self.assertRaisesRegex(ValueError,'wrappers'):self.plan()

    def test_disabled_catalog_rejected(self):
        data=json.loads(self.catalog.read_text());data['submissions_disabled']={'reason':'stopped'}
        self.catalog.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'disabled'):self.plan()

    def test_cpu_plan(self):
        self.job['review']['hardware']='CPU';self.save()
        self.worker.write_text('@app.function(max_containers=1,retries=0,timeout=60)\ndef work():pass\n')
        self.assertEqual(self.plan()['payload']['configuration']['review']['hardware'],'CPU')
