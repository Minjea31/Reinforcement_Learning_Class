import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from training.schedule import StageHandoff


class HandoffTests(unittest.TestCase):
    def test_fresh_one_kg_stage_one_uses_new_model_and_matching_physics(self):
        root=Path(__file__).resolve().parents[1]
        process_env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)/'fresh'
            result=subprocess.run([sys.executable,'-B','-m','training.train','--stage','1','--task-version','2',
                '--payload-mass','1','--layout-mode','random_separated','--total-steps','8','--num-envs','1',
                '--rollout-steps','4','--batch-size','4','--epochs','1','--max-steps','4',
                '--eval-every','100','--eval-episodes','1','--final-eval-episodes','0',
                '--record-every','0','--no-report','--run-dir',str(run),'--device','cpu'],
                cwd=root,env=process_env,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            checkpoint=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
            self.assertEqual(checkpoint['steps'],8)
            self.assertIsNone(checkpoint['config']['resume'])
            self.assertEqual(checkpoint['env_config']['payload_mass'],1.)
            self.assertEqual(checkpoint['curriculum']['stage'],1)
            from simulation.curriculum import make_env
            env=make_env(checkpoint['env_config'])
            try:
                self.assertEqual(env.mass,1.)
                _,info=env.reset(seed=1);self.assertEqual(info['weight_newtons'],9.81)
                env.set_state(checkpoint['worker_states'][0]);self.assertEqual(env.mass,1.)
            finally:env.close()

    def test_fixed_budget_and_completed_schedule(self):
        plan=StageHandoff(1,20_000_000,2,20_000_000)
        self.assertFalse(plan.due(1,19_999_992))
        self.assertTrue(plan.due(1,20_000_000))
        self.assertFalse(plan.due(2,20_000_000))
        self.assertEqual(plan.final_steps,40_000_000)
        plan.done=True
        self.assertFalse(StageHandoff(**plan.state_dict()).due(1,20_000_000))
        for values in [(1,16,3,16),(4,16,4,16),(1,0,2,16),(1,16,2,-1)]:
            with self.assertRaises(ValueError):StageHandoff(*values)

    def assert_state_equal(self,left,right):
        if isinstance(left,torch.Tensor):
            self.assertTrue(torch.equal(left,right))
        elif isinstance(left,np.ndarray):
            np.testing.assert_array_equal(left,right)
        elif isinstance(left,dict):
            self.assertEqual(left.keys(),right.keys())
            for key in left:self.assert_state_equal(left[key],right[key])
        elif isinstance(left,(list,tuple)):
            self.assertEqual(len(left),len(right))
            for x,y in zip(left,right):self.assert_state_equal(x,y)
        else:self.assertEqual(left,right)

    def test_training_and_boundary_resume_preserve_learning_state(self):
        root=Path(__file__).resolve().parents[1]
        env=os.environ.copy()
        env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        def train(*args):
            result=subprocess.run([sys.executable,'-B','-m','training.train','--no-report',*args],cwd=root,env=env,
                capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        def load(path):return torch.load(path,map_location='cpu',weights_only=False)
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)/'first';resumed=Path(directory)/'resumed'
            train('--stage','1','--total-steps','16','--next-stage','2','--next-stage-steps','16',
                  '--num-envs','2','--rollout-steps','7','--batch-size','14','--epochs','1',
                  '--max-steps','4','--eval-every','16','--eval-episodes','1',
                  '--final-eval-episodes','0','--device','cpu','--run-dir',str(run))
            before=load(run/'stage1_final.pt');after=load(run/'stage2_initial.pt')
            self.assertEqual((before['steps'],after['steps']),(16,16))
            for key in ('policy','optimizer','normalizer','rng'):
                self.assert_state_equal(before[key],after[key])
            final=load(run/'latest.pt')
            self.assertEqual((final['steps'],final['curriculum']['stage']),(32,2))
            self.assertEqual(final['config']['total_steps'],32)
            self.assertTrue(final['config']['handoff']['done'])
            # A restart exactly at the old boundary must still perform the handoff.
            # Omitted CLI budgets must preserve the saved 16+16 schedule.
            train('--resume',str(run/'stage1_final.pt'),'--device','cpu','--run-dir',str(resumed))
            restart=load(resumed/'stage2_initial.pt')
            for key in ('policy','optimizer','normalizer','rng'):
                self.assert_state_equal(before[key],restart[key])
            final=load(resumed/'latest.pt')
            self.assertEqual((final['steps'],final['curriculum']['stage']),(32,2))
            self.assertEqual(final['config']['stage_start_steps'],16)


if __name__=='__main__':unittest.main()
