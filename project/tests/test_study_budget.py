import unittest

import gymnasium as gym
from stable_baselines3 import A2C, PPO, SAC
import torch

from training.study_budget import learn_exact, stage_deadline, stage_progress


class StageBudgetTests(unittest.TestCase):
    def test_next_stage_gets_a_fresh_budget(self):
        config = dict(total_steps=40_000_000, stage_max_steps=10_000_000, stage_start_steps=0)
        self.assertEqual(stage_deadline(config), 10_000_000)
        config['stage_start_steps'] = 2_000_000
        self.assertEqual(stage_deadline(config), 12_000_000)
        self.assertEqual(stage_progress(config, 2_000_000)['stage_steps'], 0)
        self.assertEqual(stage_progress(config, 12_000_000)['stage_steps'], 10_000_000)

    def test_all_algorithms_stop_at_exact_environment_step_boundary(self):
        torch.set_num_threads(1)
        profiles = (
            (PPO, dict(n_steps=4, batch_size=4, n_epochs=1)),
            (A2C, dict(n_steps=4)),
            (SAC, dict(learning_starts=0, buffer_size=32, batch_size=4)),
        )
        for algorithm, options in profiles:
            with self.subTest(algorithm=algorithm.__name__):
                env = gym.make('MountainCarContinuous-v0')
                try:
                    model = algorithm('MlpPolicy', env, device='cpu', seed=31,
                                      policy_kwargs=dict(net_arch=[8]), **options)
                    learn_exact(model, 7, None, lambda:False)
                    self.assertEqual(model.num_timesteps, 7)
                    self.assertGreater(model._n_updates, 0)
                    if algorithm in (PPO, A2C):
                        self.assertEqual(model.n_steps, 4)
                        self.assertEqual(model.rollout_buffer.buffer_size, 4)
                    learn_exact(model, 1, None, lambda:False)
                    self.assertEqual(model.num_timesteps, 8)
                finally:
                    env.close()


if __name__ == '__main__':
    unittest.main()
