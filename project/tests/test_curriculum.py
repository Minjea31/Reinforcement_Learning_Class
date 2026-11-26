import unittest
import mujoco
import numpy as np
import torch
from curriculum import Curriculum, CurriculumReacherEnv
from ppo import ActorCritic, RunningNorm


class CurriculumTests(unittest.TestCase):
    def test_promotions_require_repeated_success_and_minimum_practice(self):
        c=Curriculum()
        self.assertFalse(c.observe(.9,10,min_steps=100))
        self.assertFalse(c.observe(.7,100,min_steps=100))
        self.assertEqual(c.passes,0)
        self.assertFalse(c.observe(.9,110,min_steps=100))
        self.assertTrue(c.observe(.9,120,min_steps=100))
        self.assertEqual((c.stage,c.level),(1,1))
        c=Curriculum(**c.state_dict())
        for _ in range(19):
            c.observe(1.,1000,required=1,min_steps=0)
        self.assertTrue(c.completed)
        self.assertEqual(c.env_config(),dict(stage=4,difficulty=1.,effort_scale=1.))

    def test_locked_stage_keeps_training_without_promoting_to_stage_two(self):
        c=Curriculum(stage=1)
        for step in range(10):
            c.observe(1.,step+1,required=1,min_steps=0,locked_stage=1)
        self.assertEqual(c.stage,1)
        self.assertEqual(c.level,4)
        self.assertTrue(c.stage_complete)
        self.assertFalse(c.completed)
        restored=Curriculum(**c.state_dict())
        self.assertFalse(restored.observe(1.,100,required=1,min_steps=0,locked_stage=1))
        self.assertEqual(restored.stage,1)

    def test_one_policy_schema_across_all_four_stages(self):
        torch.set_num_threads(1)
        policy=ActorCritic(105)
        norm=RunningNorm((105,),passthrough=6)
        env=CurriculumReacherEnv()
        try:
            for stage in range(1,5):
                obs=env.configure(dict(stage=stage,difficulty=1.,effort_scale=1. if stage==4 else 0.))
                self.assertEqual(obs.shape,(105,))
                self.assertEqual(env.action_space.shape,(14,))
                with torch.no_grad():
                    action=policy.sample(torch.from_numpy(norm(obs)[None]))[0][0].numpy()
                next_obs,reward,_,_,info=env.step(action)
                self.assertTrue(np.isfinite(next_obs).all())
                self.assertTrue(np.isfinite(reward))
                self.assertEqual(info['stage'],stage)
                np.testing.assert_array_equal(norm(obs)[-6:],obs[-6:])
            restored=RunningNorm((105,))
            restored.load_state_dict(norm.state_dict())
            self.assertEqual(restored.passthrough,6)
        finally:env.close()

    def test_curriculum_resume_restores_stage_control_and_physics(self):
        env=CurriculumReacherEnv()
        try:
            env.reset(seed=31)
            action=np.linspace(-.4,.4,14)
            env.step(action)
            state=env.get_state()
            expected=env.step(action)
            next_reset=env.reset()[0]
            env.configure(dict(stage=4,difficulty=1.,effort_scale=1.))
            env.set_state(state)
            actual=env.step(action)
            self.assertEqual(env.stage,1)
            np.testing.assert_allclose(actual[0],expected[0],atol=1e-7)
            self.assertAlmostEqual(actual[1],expected[1],places=7)
            np.testing.assert_array_equal(env.reset()[0],next_reset)
        finally:env.close()

    def test_force_cost_is_delayed_and_ramped_without_removing_physics(self):
        envs=[CurriculumReacherEnv(stage=4,difficulty=1.,effort_scale=scale) for scale in (0.,.5,1.)]
        try:
            info=[]
            for env in envs:
                env.reset(seed=31)
                info.append(env.step(np.zeros(14))[4])
            self.assertGreater(info[0]['torque_integral'],0.)
            self.assertEqual(info[0]['reward_terms']['torque'],0.)
            self.assertAlmostEqual(info[1]['reward_terms']['torque'],.5*info[2]['reward_terms']['torque'])
            self.assertLess(info[2]['reward_terms']['torque'],0.)
        finally:
            for env in envs:env.close()

    def test_stage_one_can_physically_grasp_lift_and_succeed_with_either_arm(self):
        # This is a mechanics check; training never calls this IK controller.
        for seed in (0,1):
            env=CurriculumReacherEnv(stage=1,difficulty=0.)
            try:
                env.reset(seed=seed)
                self.assertFalse(any(env.info()['grasped']))
                side=env.initial_side
                targets=env.data.site_xpos[env.sites].copy()
                for step in range(350):
                    if step<30:
                        action=np.zeros(14);action[6+side*7]=-1.
                    else:
                        targets[side]=env.start+[0.,0.,.005+.27*min(1.,(step-30)/150)]
                        desired=env.solve_ik(targets,1.,iterations=12,seed_action=env.target_action)
                        desired[6+side*7]=-1.
                        action=np.clip((desired-env.target_action)/np.array([.012]*6+[.15]+[.012]*6+[.15]),-1,1)
                    _,_,term,_,info=env.step(action)
                    if term:break
                self.assertTrue(info['is_success'])
                self.assertTrue(info['episode_lifted'])
                self.assertEqual(info['payload_mass'],5.)
                self.assertFalse(info['placed'])
            finally:env.close()

    def test_stage_two_still_requires_carry_before_placement(self):
        env=CurriculumReacherEnv(stage=2)
        try:
            env.reset(seed=31)
            env.data.qpos[env.payload_qpos:env.payload_qpos+3]=env.goal
            env.data.qvel[:]=0.
            mujoco.mj_forward(env.model,env.data)
            for _ in range(60):
                _,_,_,_,info=env.step(np.zeros(14))
            self.assertTrue(info['placed'])
            self.assertFalse(info['is_success'])
        finally:env.close()

    def test_unused_closed_gripper_does_not_block_completed_placement(self):
        env=CurriculumReacherEnv(stage=2)
        try:
            env.reset(seed=31)
            unused=1-env.initial_side
            for q in env.finger_joints[unused*2:unused*2+2]:
                env.data.qpos[q]=0.
            env.target_action[unused*7+6]=-1.
            env.data.qpos[env.payload_qpos:env.payload_qpos+3]=env.goal
            env.data.qvel[:]=0.
            env.carried=True
            mujoco.mj_forward(env.model,env.data)
            for _ in range(60):
                _,_,term,_,info=env.step(np.zeros(14))
                if term:break
            self.assertTrue(info['is_success'])
            self.assertFalse(any(info['grasped']))
        finally:env.close()

    def test_last_randomization_level_covers_full_random_task(self):
        env=CurriculumReacherEnv(stage=3,difficulty=1.)
        try:
            starts=[]
            for seed in range(12):
                env.reset(seed=seed)
                starts.append(env.start.copy())
                self.assertTrue(env.random_episode)
                self.assertLess(abs(env.obstacle_position[0]),1.)
                self.assertEqual(env.mass,5.)
            self.assertGreater(np.ptp(np.array(starts)[:,0]),.6)
        finally:env.close()


if __name__=='__main__':unittest.main()
