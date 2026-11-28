import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from simulation.curriculum import make_env
from test_placement_curriculum import pickup_fixture

GPU_AVAILABLE=importlib.util.find_spec('mujoco_warp') is not None and torch.cuda.is_available()


@unittest.skipUnless(GPU_AVAILABLE,'Requires dual-reacher-gpu Conda environment and CUDA')
class GPUPhysicsTests(unittest.TestCase):
    def test_random_reset_uploads_layout_and_preserves_effort_costs(self):
        env=make_env(dict(stage=4,difficulty=1.,effort_scale=1.,task_version=2,physics_backend='warp'))
        self.addCleanup(env.close)
        obstacles=[]
        for seed in (3,9):
            env.reset(seed=seed)
            _,reward,_,_,info=env.step(np.zeros(14))
            self.assertTrue(np.isfinite(reward))
            self.assertLess(info['reward_terms']['torque'],0.)
            np.testing.assert_allclose(env.physics.model.geom_size.numpy().reshape(-1,3),env.model.geom_size,atol=1e-7)
            np.testing.assert_allclose(env.physics.data.mocap_pos.numpy()[0,env.obstacle_mocap],env.obstacle_position,atol=1e-7)
            np.testing.assert_allclose(env.physics.model.site_pos.numpy().reshape(-1,3)[env.goal_site],env.model.site_pos[env.goal_site],atol=1e-7)
            obstacles.append(env.obstacle_position.copy())
        self.assertFalse(np.array_equal(*obstacles))

    def test_one_cuda_world_without_cpu_integration_and_resume(self):
        config=dict(stage=2,difficulty=0.,effort_scale=0.)
        cpu=make_env(config);gpu=make_env(dict(config,physics_backend='warp'))
        self.addCleanup(cpu.close);self.addCleanup(gpu.close)
        cpu.reset(seed=31);gpu.reset(seed=31)
        action=np.linspace(-.1,.1,14)
        expected=cpu.step(action)
        with patch('mujoco.mj_step',side_effect=AssertionError('CPU integration must not run')):
            actual=gpu.step(action)
        self.assertEqual(gpu.physics.data.nworld,1)
        self.assertTrue(gpu.physics.data.qpos.device.is_cuda)
        self.assertEqual(actual[4]['physics_substeps'],10)
        np.testing.assert_allclose(gpu.data.qpos,cpu.data.qpos,atol=1e-5,rtol=0.)
        self.assertAlmostEqual(actual[4]['torque_integral'],expected[4]['torque_integral'],places=6)
        gpu.step(action);state=gpu.get_state();expected=gpu.step(action)
        gpu.set_state(state);actual=gpu.step(action)
        np.testing.assert_allclose(actual[0],expected[0],atol=2e-6,rtol=0.)
        self.assertAlmostEqual(actual[1],expected[1],places=6)

    def test_five_kg_can_be_released_and_settle_on_gpu_with_either_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'pickup.pt';pickup_fixture(source)
            env=make_env(dict(stage=2,difficulty=0.,effort_scale=0.,task_version=2,
                              pickup_checkpoint=str(source),physics_backend='warp'))
            try:
                for seed in (0,1):
                    env.reset(seed=seed)
                    for _ in range(250):
                        action=np.zeros(14);action[6+7*env.initial_side]=.2
                        _,reward,term,trunc,info=env.step(action)
                        self.assertTrue(np.isfinite(reward))
                        if term or trunc:break
                    self.assertTrue(info['is_success'])
                    self.assertTrue(info['placed'])
                    self.assertEqual(info['payload_mass'],1.)
                    self.assertEqual(info['payload_robot_contacts'],0)
                    self.assertLessEqual(np.abs(info['motor_torque']).max(),65.001)
                    self.assertLessEqual(np.abs(info['gripper_force']).max(),200.001)
            finally:env.close()

    def test_cuda_ppo_transfer_evaluation_and_resume(self):
        process_env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        root=Path(__file__).resolve().parents[1]
        def train(*args):
            result=subprocess.run([sys.executable,'-u','-B','-m','training.train','--no-report',*args],cwd=root,
                                  env=process_env,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('Physics: warp; simulations: 1',result.stdout)
        def load(path):return torch.load(path,map_location='cpu',weights_only=False)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'pickup.pt';original=pickup_fixture(source)
            run=Path(directory)/'gpu'
            train('--resume',str(source),'--stage','2','--task-version','2','--physics-backend','warp',
                  '--num-envs','1','--device','cuda','--run-dir',str(run),'--total-steps','24')
            initial=load(run/'transfer_initial.pt');latest=load(run/'latest.pt')
            self.assertEqual(initial['version'],4)
            self.assertEqual(latest['env_config']['physics_backend'],'warp')
            self.assertEqual(latest['config']['num_envs'],1)
            self.assertEqual(len(latest['worker_states']),1)
            self.assertEqual(latest['steps'],24)
            self.assertEqual(latest['updates'],original['updates']+2)
            self.assertIsNotNone(latest['rng']['cuda'])
            for key,value in original['policy'].items():self.assertTrue(torch.equal(value,initial['policy'][key]))
            self.assertTrue(any(not torch.equal(original['policy'][k],latest['policy'][k]) for k in original['policy']))
            validation=run/'validation_24.json';self.assertTrue(validation.exists())
            train('--resume',str(run/'latest.pt'),'--device','cuda','--total-steps','28')
            self.assertEqual(load(run/'latest.pt')['steps'],28)


if __name__=='__main__':unittest.main()
