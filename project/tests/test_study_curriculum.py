import unittest
import numpy as np
from unittest.mock import patch

from simulation.curriculum import Curriculum,make_env


class StudyCurriculumTests(unittest.TestCase):
    def test_masked_and_both_enabled_evaluation_share_seeded_physics(self):
        config = dict(stage=1, difficulty=0., effort_scale=0., task_version=2,
                      layout_mode='progressive', reward_scheme='reviewed')
        masked = make_env(dict(config, arm_mask='single'))
        both = make_env(dict(config, arm_mask='both'))
        try:
            for seed in (1001000000, 1001000001):
                single_obs, _ = masked.reset(seed=seed)
                both_obs, _ = both.reset(seed=seed)
                self.assertEqual(masked.initial_side, seed % 2)
                np.testing.assert_array_equal(single_obs[:105], both_obs[:105])
                np.testing.assert_array_equal(single_obs[-2:], np.eye(2)[seed % 2])
                np.testing.assert_array_equal(both_obs[-2:], [1., 1.])
                other = 1 - seed % 2
                idle = slice(7 * other, 7 * (other + 1))
                before = masked.target_action[idle].copy()
                masked.step(-np.ones(14))
                both.step(-np.ones(14))
                np.testing.assert_allclose(masked.target_action[idle], before, atol=1e-12, rtol=0.)
                self.assertFalse(np.array_equal(both.target_action[idle], before))
        finally:
            masked.close()
            both.close()

    def test_observation_uses_only_active_reward_clocks(self):
        config=dict(stage=1,difficulty=0.,effort_scale=0.,task_version=2,layout_mode='progressive',reward_scheme='reviewed')
        env=make_env(config)
        try:
            env.reset(seed=31)
            env.carried=True;env.lift_seconds=.2;env.success_steps=100
            env.lift_hold=0.;env.placement_hold=0.;env.qualified=False
            obs=env._observation()
            self.assertEqual(obs.shape,(105,))
            self.assertAlmostEqual(float(obs[98]),env.initial_goal_distance,places=6)
            np.testing.assert_array_equal(obs[[82,87,88]],[0.,0.,0.])
            env.lift_hold=.28;env.qualified=True
            np.testing.assert_allclose(env._observation()[[82,87,88]],[.56,1.,.28])
            env.stage=3;env.placement_hold=.74
            self.assertAlmostEqual(float(env._observation()[88]),.74,places=6)
        finally:env.close()

    def test_one_physics_advance_and_exact_reward_accounting_per_action(self):
        config=dict(stage=1,difficulty=0.,effort_scale=0.,task_version=2,layout_mode='progressive',reward_scheme='reviewed')
        env=make_env(config)
        try:
            env.reset(seed=31);total=0.
            with patch.object(env,'_advance_physics',wraps=env._advance_physics) as advance:
                for count in range(1,11):
                    obs,reward,_,_,info=env.step(np.zeros(14));total+=reward
                    self.assertEqual(advance.call_count,count)
                    self.assertEqual(env.steps,count)
                    self.assertAlmostEqual(total,info['episode_return'])
                    self.assertAlmostEqual(reward,sum(info['reward_terms'].values()))
                    self.assertEqual(info['carried'],env.qualified)
                    self.assertEqual(info['lift_seconds'],env.lift_hold)
                    self.assertAlmostEqual(float(obs[88]),info['success_hold_seconds'],places=6)
        finally:env.close()

    def test_models_promote_independently(self):
        controllers={name:Curriculum(stage=1) for name in ('ppo','a2c','sac')}
        controllers['ppo'].observe(.9,100000,locked_stage=1)
        self.assertTrue(controllers['ppo'].observe(.9,200000,locked_stage=1))
        self.assertEqual([controllers[n].level for n in ('ppo','a2c','sac')],[1,0,0])
        self.assertEqual(controllers['a2c'].passes,0)

    def test_initial_open_pose_is_near_reproducible_and_reward_resume_is_exact(self):
        config=dict(stage=1,difficulty=0.,effort_scale=0.,task_version=2,layout_mode='progressive',reward_scheme='reviewed')
        env=make_env(config,max_steps=32)
        try:
            first,info=env.reset(seed=1)
            self.assertFalse(any(info['grasped']));self.assertFalse(info['carried'])
            distance=np.linalg.norm(env.data.site_xpos[env.sites[env.initial_side]]-env.data.xpos[env.payload_id])
            self.assertGreaterEqual(distance,.05-.007);self.assertLessEqual(distance,.10+.007)
            env.reset(seed=8);np.testing.assert_array_equal(env.reset(seed=1)[0],first)
            action=np.linspace(-.2,.2,14)
            env.step(action);saved=env.get_state();expected=env.step(action)
            env.set_state(saved);actual=env.step(action)
            np.testing.assert_array_equal(actual[0],expected[0]);self.assertAlmostEqual(actual[1],expected[1],places=9)
            self.assertEqual(actual[4]['reward_terms'],expected[4]['reward_terms'])
        finally:env.close()

    def test_level_change_resamples_only_that_environment(self):
        config=dict(stage=1,difficulty=0.,effort_scale=0.,task_version=2,layout_mode='progressive',reward_scheme='reviewed')
        first=make_env(config);second=make_env(config)
        try:
            second.reset(seed=31);pose=second.data.qpos.copy()
            first.configure(dict(config,difficulty=.25))
            self.assertEqual(first.difficulty,.25);self.assertEqual(second.difficulty,0.)
            np.testing.assert_array_equal(second.data.qpos,pose)
        finally:first.close();second.close()

    def test_both_arms_can_physically_succeed_under_the_new_reward(self):
        config=dict(stage=1,difficulty=0.,effort_scale=0.,task_version=2,layout_mode='progressive',reward_scheme='reviewed')
        env=make_env(config)
        try:
            for seed in (0,1):
                env.reset(seed=seed);arm=env.initial_side;targets=env.data.site_xpos[env.sites].copy()
                origin=targets[arm].copy();pickup=env.start+[0.,0.,.005]
                for step in range(500):
                    if step<100:targets[arm]=origin+(pickup-origin)*min(1.,(step+1)/100)
                    elif step>=130:targets[arm]=pickup+[0.,0.,.16*min(1.,(step-130)/100)]
                    desired=env.solve_ik(targets,1.,iterations=8,seed_action=env.target_action)
                    desired[6+7*arm]=1. if step<100 else -1.
                    action=np.clip((desired-env.target_action)/np.array([.012]*6+[.15]+[.012]*6+[.15]),-1,1)
                    obs,_,terminated,_,info=env.step(action)
                    self.assertEqual(obs[87],float(info['carried']))
                    self.assertAlmostEqual(float(obs[82]),min(.5,env.lift_hold)/.5,places=6)
                    self.assertAlmostEqual(float(obs[88]),info['success_hold_seconds'],places=6)
                    if terminated:break
                self.assertTrue(info['is_success'],f'arm {arm} failed')
                self.assertTrue(info['episode_lifted'])
        finally:env.close()


if __name__=='__main__':unittest.main()
