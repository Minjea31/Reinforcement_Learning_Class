import copy
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import mujoco
import numpy as np
import torch

from simulation.curriculum import Curriculum, CurriculumReacherEnv, make_env
from evaluation.evaluate import curriculum_ranking
from simulation.placement_curriculum import PlacementCurriculumEnv
from training.ppo import ActorCritic, RunningNorm
from training.train import parser


def pickup_fixture(path):
    """Portable neural controller for a physical lift, not a trained PPO result."""
    torch.set_num_threads(1)
    env=CurriculumReacherEnv(stage=1,difficulty=0.)
    try:
        env.reset(seed=31)
        desired=env.solve_ik(np.array([[-.3,-.25,.9],[.3,-.25,.9]]),1.,iterations=180)
        desired[[6,13]]=-1.
    finally:env.close()
    policy=ActorCritic(105)
    with torch.no_grad():
        for parameter in policy.actor.parameters():parameter.zero_()
        # Observations include current targets at 68:82. This fixture moves
        # toward fixed raised targets through the normal incremental action API.
        for k in range(14):
            policy.actor[0].weight[k,68+k]=.1
            policy.actor[2].weight[k,k]=1.
            policy.actor[4].weight[k,k]=-30.
            policy.actor[4].bias[k]=3.*desired[k]
    optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
    policy.critic(torch.zeros(1,105)).sum().backward()
    optimizer.step()  # Exercise transfer of nonempty Adam moments too.
    config=vars(parser().parse_args([]))
    config.update(curriculum=True,stage=1,total_steps=16,num_envs=2,rollout_steps=4,
                  batch_size=8,epochs=1,max_steps=4,eval_every=8,eval_episodes=1,
                  final_eval_episodes=0,device='cpu',record_every=0,
                  next_stage=2,next_stage_steps=16,
                  handoff=dict(from_stage=1,at_steps=16,to_stage=2,additional_steps=16,done=False))
    checkpoint=dict(version=2,steps=16,updates=1,obs_dim=105,
        control_mode='incremental_targets',env_config=Curriculum().env_config(),
        curriculum=Curriculum().state_dict(),policy=policy.state_dict(),
        optimizer=optimizer.state_dict(),normalizer=RunningNorm((105,),passthrough=6).state_dict(),
        config=config,worker_states=[],best_rank=(-1.,0.,0.,-float('inf')),next_eval=24,
        rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state()))
    torch.save(checkpoint,path)
    return checkpoint


class PlacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory()
        cls.source=Path(cls.directory.name)/'pickup.pt'
        cls.checkpoint=pickup_fixture(cls.source)

    @classmethod
    def tearDownClass(cls):cls.directory.cleanup()

    def env(self,**kwargs):
        env=PlacementCurriculumEnv(stage=2,pickup_checkpoint=self.source,**kwargs)
        self.addCleanup(env.close)
        return env

    def test_physical_pickup_handoff_and_release_work_for_either_arm(self):
        env=self.env()
        for seed in (0,1):
            obs,info=env.reset(seed=seed)
            teacher=env._pickup.env
            np.testing.assert_array_equal(env.data.qpos,teacher.data.qpos)
            np.testing.assert_array_equal(env.data.qvel,teacher.data.qvel)
            np.testing.assert_array_equal(env.target_action,teacher.target_action)
            self.assertGreater(np.linalg.norm(env.data.qvel),0.)
            self.assertEqual(obs.shape,(105,))
            self.assertEqual(env.action_space.shape,(14,))
            self.assertEqual((env.steps,env.data.time,env.episode_return,env.episode_effort),(0,0.,0.,0.))
            self.assertTrue(info['carried'] and any(info['grasped']))
            self.assertFalse(info['table_support'])
            self.assertGreater(env.pickup_steps,0)
            for _ in range(250):
                action=np.zeros(14);action[6+7*env.initial_side]=.2
                _,_,terminated,truncated,info=env.step(action)
                if terminated or truncated:break
            self.assertTrue(info['is_success'])
            self.assertTrue(info['placed'])
            self.assertEqual(info['payload_robot_contacts'],0)
            self.assertGreaterEqual(info['success_hold_seconds'],1.-1e-9)

    def test_distance_curriculum_and_exact_resume(self):
        env=self.env()
        for difficulty,separation in zip((0.,.25,.5,.75,1.),(0.,.06,.12,.18,.25)):
            env.configure(dict(stage=2,difficulty=difficulty,effort_scale=0.))
            env.reset(seed=1)
            np.testing.assert_allclose(env.goal[:2]-env.data.xpos[env.payload_id,:2],[0.,separation],atol=1e-12)
        action=np.linspace(-.2,.2,14)
        env.step(action);state=env.get_state()
        expected=env.step(action);expected_reset=env.reset()[0]
        env.set_state(state)
        actual=env.step(action)
        np.testing.assert_array_equal(actual[0],expected[0])
        self.assertEqual(actual[1],expected[1])
        np.testing.assert_array_equal(env.reset()[0],expected_reset)
        with self.assertRaisesRegex(ValueError,'version-one'):
            invalid=copy.deepcopy(state);invalid.pop('placement_v2');env.set_state(invalid)
        with self.assertRaisesRegex(ValueError,'changed'):
            invalid=copy.deepcopy(state);invalid['placement_v2']['pickup_source_hash']='different';env.set_state(invalid)

    def test_random_starts_and_goals_are_separate_and_reproducible(self):
        env=self.env(layout_mode='random_separated');starts=[];goals=[]
        for seed in range(6):
            _,info=env.reset(seed=seed)
            self.assertTrue(info['carried'] and any(info['grasped']))
            self.assertTrue(env._markers_separated(env.start,env.goal))
            self.assertGreaterEqual(np.linalg.norm(env.start[:2]-env.goal[:2]),.22)
            starts.append(env.start[:2].copy());goals.append(env.goal[:2].copy())
        self.assertGreater(np.ptp(starts,axis=0).max(),.15)
        self.assertGreater(np.ptp(goals,axis=0).max(),.15)
        state=env.get_state();expected=env.reset()[0];env.set_state(state)
        np.testing.assert_array_equal(env.reset()[0],expected)

    def test_sideways_placement_requires_carry_and_full_release(self):
        # State injection checks the success predicate, not transport feasibility.
        env=self.env()
        for carried in (False,True):
            env.reset(seed=1)
            env.goal[:2]=[0.,.25]
            env.model.site_pos[env.goal_site,:2]=env.goal[:2]
            env.data.qpos[env.payload_qpos:env.payload_qpos+3]=[0.,.25,.635]
            env.data.qpos[env.payload_qpos+3:env.payload_qpos+7]=[2**-.5,2**-.5,0.,0.]
            env.data.qvel[:]=0.
            env.carried=carried
            mujoco.mj_forward(env.model,env.data)
            for _ in range(65):
                _,_,terminated,_,info=env.step(np.zeros(14))
                if terminated:break
            self.assertTrue(info['placed'])
            self.assertEqual(info['is_success'],carried)
            if carried:self.assertEqual(info['reward_terms']['success'],100.)

    def test_free_gripper_control_and_premature_release_cost(self):
        env=self.env(difficulty=1.)
        env.reset(seed=1)
        before=env.target_action[6+7*env.initial_side]
        losses=0
        for _ in range(80):
            action=np.zeros(14);action[6+7*env.initial_side]=.2
            _,_,terminated,_,info=env.step(action)
            losses+=int(info['premature_release'])
            if info['premature_release']:self.assertEqual(info['reward_terms']['premature_release'],-2.)
            if terminated:break
        self.assertGreater(env.target_action[6+7*env.initial_side],before)
        self.assertEqual(losses,1)
        self.assertFalse(info['is_success'])

    def test_task_dispatch_and_progress_ranking(self):
        legacy=make_env(Curriculum().env_config());self.addCleanup(legacy.close)
        self.assertIs(type(legacy),CurriculumReacherEnv)
        v2=make_env(Curriculum(stage=2,task_version=2,pickup_checkpoint=str(self.source)).env_config())
        self.addCleanup(v2.close)
        self.assertIs(type(v2),PlacementCurriculumEnv)
        idle=dict(success_rate=0.,placement_rate=0.,release_rate=0.,goal_reach_rate=0.,mean_goal_progress=0.,mean_return=20.)
        moving=dict(idle,mean_goal_progress=.5,mean_return=-10.)
        self.assertGreater(curriculum_ranking(moving,2,2),curriculum_ranking(idle,2,2))
        self.assertEqual(curriculum_ranking(idle,2,1),(0.,0.,0.,20.))

    def assert_same(self,left,right):
        if isinstance(left,torch.Tensor):self.assertTrue(torch.equal(left,right))
        elif isinstance(left,np.ndarray):np.testing.assert_array_equal(left,right)
        elif isinstance(left,dict):
            self.assertEqual(left.keys(),right.keys())
            for key in left:self.assert_same(left[key],right[key])
        elif isinstance(left,(tuple,list)):
            self.assertEqual(len(left),len(right))
            for a,b in zip(left,right):self.assert_same(a,b)
        else:self.assertEqual(left,right)

    def test_cli_transfer_preserves_model_optimizer_norm_and_resumes_v2(self):
        root=Path(__file__).resolve().parents[1]
        process_env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        def train(*args):
            result=subprocess.run([sys.executable,'-B','-m','training.train','--no-report',*args],cwd=root,env=process_env,
                                  capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        def load(path):return torch.load(path,map_location='cpu',weights_only=False)
        run=Path(self.directory.name)/'transfer'
        train('--resume',str(self.source),'--stage','2','--task-version','2',
              '--total-steps','32','--run-dir',str(run),'--device','cpu')
        initial=load(run/'transfer_initial.pt')
        for key in ('policy','optimizer','normalizer'):self.assert_same(self.checkpoint[key],initial[key])
        self.assertEqual((initial['steps'],initial['version'],initial['curriculum']['stage']),(16,3,2))
        self.assertIsNone(initial['config']['handoff'])
        self.assertEqual((run/'pickup_source.pt').read_bytes(),self.source.read_bytes())
        for state in initial['worker_states']:
            self.assertTrue(state['task']['carried'])
            self.assertGreater(state['placement_v2']['pickup_steps'],0)
        final=load(run/'latest.pt')
        self.assertEqual((final['steps'],final['curriculum']['stage']),(32,2))
        train('--resume',str(run/'latest.pt'),'--total-steps','40','--device','cpu')
        self.assertEqual(load(run/'latest.pt')['steps'],40)
        random_run=Path(self.directory.name)/'random_transfer'
        before=load(run/'latest.pt')
        train('--resume',str(run/'latest.pt'),'--layout-mode','random_separated',
              '--total-steps','48','--run-dir',str(random_run),'--device','cpu')
        initial=load(random_run/'transfer_initial.pt')
        for key in ('policy','optimizer','normalizer'):self.assert_same(before[key],initial[key])
        self.assertEqual(initial['steps'],40)
        self.assertEqual(initial['env_config']['layout_mode'],'random_separated')
        self.assertEqual(load(random_run/'latest.pt')['steps'],48)


if __name__=='__main__':unittest.main()
