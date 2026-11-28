import unittest
import numpy as np
import torch
from torch.distributions import Normal, TransformedDistribution, TanhTransform
from simulation.dual_reacher import DualReacherEnv
from training.ppo import ActorCritic, compute_gae
from evaluation.evaluate import ranking


class PPOTests(unittest.TestCase):
    def test_squashed_density_and_gradients(self):
        torch.set_num_threads(1)
        policy = ActorCritic(89)
        obs = torch.zeros(4,89)
        action, latent, logp, _ = policy.sample(obs)
        dist = policy.distribution(obs)
        expected = TransformedDistribution(Normal(dist.mean,dist.stddev), [TanhTransform()]).log_prob(action).sum(-1)
        torch.testing.assert_close(logp,expected)
        actual, entropy, values = policy.evaluate(obs,latent)
        torch.testing.assert_close(actual,logp)
        (-actual.mean()-.005*entropy.mean()+values.square().mean()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None))
        extreme = torch.tensor([-100.,100.])
        self.assertTrue(torch.isfinite(policy.log_jacobian(extreme)).all())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA device unavailable')
    def test_cuda_policy_update_and_evaluation(self):
        from evaluation.evaluate import evaluate_policy
        from training.ppo import RunningNorm
        torch.set_num_threads(1)
        policy = ActorCritic(89).to('cuda')
        observations = torch.zeros(8,89,device='cuda')
        _, latent, _, _ = policy.sample(observations)
        logp, entropy, values = policy.evaluate(observations,latent)
        optimizer = torch.optim.Adam(policy.parameters())
        optimizer.zero_grad()
        loss = -logp.mean()-.005*entropy.mean()+values.square().mean()
        loss.backward()
        optimizer.step()
        self.assertTrue(all(p.is_cuda for p in policy.parameters()))
        self.assertTrue(torch.isfinite(loss))
        result = evaluate_policy(policy,RunningNorm((89,)),1,2_000_000_000,2,progress_label=None)
        self.assertEqual(len(result['records']),1)

    def test_timeout_bootstraps_without_leaking_reset_episode(self):
        rewards = np.array([[1.],[100.]],dtype=np.float32)
        values = np.array([[2.],[3.]],dtype=np.float32)
        next_values = np.array([[7.],[0.]],dtype=np.float32)
        term = np.array([[False],[True]])
        trunc = np.array([[True],[False]])
        advantage, returns = compute_gae(rewards,values,next_values,term,trunc,gamma=.9,lam=1.)
        self.assertAlmostEqual(float(returns[0,0]),7.3,places=5)
        self.assertAlmostEqual(float(returns[1,0]),100.,places=5)
        term[0] = True
        _, returns = compute_gae(rewards,values,next_values,term,trunc,gamma=.9)
        self.assertAlmostEqual(float(returns[0,0]),1.)

    def test_success_is_first_selection_criterion(self):
        def result(s,a,e):
            return dict(success_rate=s,successful_used_arms=a,successful_effort=e,mean_return=0)
        self.assertGreater(ranking(result(.9,2,100)),ranking(result(.8,1,1)))
        self.assertGreater(ranking(result(.9,1,100)),ranking(result(.9,2,1)))
        self.assertGreater(ranking(result(.9,1,1)),ranking(result(.9,1,100)))


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = DualReacherEnv(max_steps=200)
        self.env.reset(seed=31)

    def tearDown(self):
        self.env.close()

    def test_checkpoint_restores_physics_task_and_reset_rng(self):
        e = self.env
        e.step(e.target_action)
        state = e.get_state()
        action = e.target_action.copy()
        expected = e.step(action)
        next_reset = e.reset()[0]
        e.set_state(state)
        actual = e.step(action)
        np.testing.assert_allclose(actual[0],expected[0],atol=1e-7)
        self.assertAlmostEqual(actual[1],expected[1],places=7)
        np.testing.assert_array_equal(e.reset()[0],next_reset)

    def test_default_mass_and_static_effort(self):
        e = self.env
        self.assertEqual(e.mass,1.)
        obs, reward, _, _, info = e.step(e.target_action)
        self.assertEqual(obs.shape,(89,))
        self.assertTrue(np.isfinite(obs).all())
        self.assertGreater(info['torque_integral'],0.)
        self.assertLess(info['reward_terms']['torque'],0.)
        self.assertAlmostEqual(reward,sum(info['reward_terms'].values()))

    def test_pushing_to_goal_without_lift_is_not_success(self):
        e = self.env
        import mujoco
        # A settled, released object on the target still needs a qualified carry.
        e.data.qpos[e.payload_qpos:e.payload_qpos+3] = e.goal
        e.data.qvel[:] = 0
        mujoco.mj_forward(e.model,e.data)
        for _ in range(55):
            _, _, _, _, info = e.step(e.target_action)
        self.assertTrue(info['placed'])
        self.assertFalse(info['is_success'])
        self.assertFalse(e.carried)

    def test_stable_placement_after_carry_can_succeed(self):
        e = self.env
        import mujoco
        e.data.qpos[e.payload_qpos:e.payload_qpos+3] = e.goal
        e.data.qvel[:] = 0
        e.carried = True
        mujoco.mj_forward(e.model,e.data)
        for _ in range(55):
            _, _, term, _, info = e.step(e.target_action)
            if term:
                break
        self.assertTrue(info['is_success'])
        self.assertGreaterEqual(info['success_hold_seconds'],1.)

    def test_five_kg_single_arm_can_lift_and_hold_within_limits(self):
        # Mechanical feasibility only. This IK controller is never used by PPO.
        e = self.env
        e.max_steps = 2000
        e.reset(options={'randomize': False, 'start': [-.30,-.25,.625],
                         'goal': [-.30,.25,.625]})
        targets = e.home_targets.copy()
        for point, opening, duration in [([-.30,-.25,.76],1.,150),
                                         ([-.30,-.25,.625],1.,150),
                                         ([-.30,-.25,.625],-1.,100),
                                         ([-.30,-.25,.90],-1.,200),
                                         ([-.30,-.25,.90],-1.,100)]:
            initial = targets.copy()
            for t in range(duration):
                targets[0] = initial[0]+(np.array(point)-initial[0])*min(1.,(t+1)/100)
                action = e.solve_ik(targets,1.,iterations=12,seed_action=e.target_action)
                action[6] = opening
                _, _, _, _, info = e.step(action)
        self.assertTrue(info['grasped'][0])
        self.assertFalse(info['grasped'][1])
        self.assertTrue(info['carried'])
        self.assertGreater(info['payload_position'][2]-info['vertical_half_extent'],.63)
        self.assertGreater(abs(info['gripper_force'][0]),10.)
        self.assertLessEqual(np.abs(info['gripper_force']).max(),200.+1e-6)
        self.assertLessEqual(np.abs(info['motor_torque']).max(),65.+1e-6)
        self.assertLess(info['reward_terms']['torque'],0.)

    def test_random_layouts_are_diverse_and_finite(self):
        starts, goals = [], []
        for seed in range(20):
            obs, info = self.env.reset(seed=seed)
            starts.append(info['start'][0])
            goals.append(info['goal'][0])
            self.assertTrue(np.isfinite(obs).all())
            self.assertGreaterEqual(np.linalg.norm(info['start']-info['goal']),.24)
        self.assertLess(min(starts),-.3)
        self.assertGreater(max(starts),.3)
        self.assertLess(min(goals),-.3)
        self.assertGreater(max(goals),.3)


if __name__ == '__main__':
    unittest.main()
