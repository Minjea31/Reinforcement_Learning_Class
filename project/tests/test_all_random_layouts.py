import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from simulation.curriculum import Curriculum, make_env
from simulation.layout_sampling import reachable, single_arm_route
from tests.test_placement_curriculum import pickup_fixture


class AllRandomLayoutTests(unittest.TestCase):
    def make(self, stage):
        env = make_env(dict(stage=stage, task_version=2, layout_mode='random_all',
                            difficulty=1. if stage == 4 else 0., effort_scale=1. if stage == 4 else 0.))
        self.addCleanup(env.close)
        return env

    def assert_separate(self, env):
        delta = np.abs(env.start[:2] - env.goal[:2])
        self.assertGreaterEqual(np.linalg.norm(delta), .22)
        self.assertTrue(np.any(delta - [.26, .16] > .025))
        self.assertTrue(env._clear_table_area(env.start, [.13, .08], env.obstacle_position, env.model.geom_size[env.obstacle_geom]))
        self.assertTrue(env._clear_table_area(env.goal, [.13, .08], env.obstacle_position, env.model.geom_size[env.obstacle_geom]))

    def test_stage_one_varies_both_arms_and_keeps_a_single_arm_path(self):
        env = self.make(1)
        starts, goals, arms = [], [], set()
        for seed in range(8):
            obs, info = env.reset(seed=seed)
            self.assertEqual(obs.shape, (105,))
            self.assertEqual(env.action_space.shape, (14,))
            self.assertFalse(any(info['grasped']))
            self.assertEqual(env.mass, 1.)
            self.assert_separate(env)
            self.assertTrue(single_arm_route(env, env.start, env.goal, env.initial_side))
            starts.append(env.start.copy()); goals.append(env.goal.copy()); arms.add(env.initial_side)
        self.assertEqual(arms, {0, 1})
        self.assertGreater(np.ptp(starts, axis=0)[0], .7)
        self.assertGreater(np.ptp(goals, axis=0)[1], .3)
        self.assertEqual(len({tuple(point) for point in starts}), 8)

    def test_stages_three_and_four_always_randomize_obstacles_and_endpoints(self):
        for stage in (3, 4):
            env = self.make(stage)
            obstacles = []
            for seed in range(6):
                env.reset(seed=seed)
                self.assertTrue(env.random_episode)
                self.assert_separate(env)
                self.assertLess(abs(env.obstacle_position[0]), 1.)
                self.assertTrue(any(reachable(env, env.start + [0, 0, .005], arm) for arm in (0, 1)))
                self.assertTrue(any(reachable(env, env.goal + [0, 0, .005], arm) for arm in (0, 1)))
                obstacles.append(env.obstacle_position.copy())
            self.assertGreater(np.ptp(obstacles, axis=0)[0], .2)

    def test_random_layout_resume_reproduces_physics_and_next_reset(self):
        env = self.make(1)
        env.reset(seed=31)
        action = np.linspace(-.2, .2, 14)
        env.step(action); state = env.get_state()
        expected = env.step(action); next_reset = env.reset()[0]
        env.set_state(state)
        actual = env.step(action)
        np.testing.assert_array_equal(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        np.testing.assert_array_equal(env.reset()[0], next_reset)

    def test_stage_two_preparation_preserves_actual_grip_and_randomizes_the_goal(self):
        # The test controller only checks physical reset transfer, not PPO quality.
        def prepare(pickup, seed):
            env = pickup.env; env.reset(seed=seed); side = env.initial_side
            for step in range(260):
                if step < 30:
                    action = np.zeros(14); action[6+7*side] = -1.
                elif step < 170:
                    targets = env.data.site_xpos[env.sites].copy()
                    targets[side] = env.start + [0., 0., .005+.20*min(1., (step-30)/100)]
                    desired = env.solve_ik(targets, 1., iterations=12, seed_action=env.target_action)
                    desired[6+7*side] = -1.
                    action = np.clip((desired-env.target_action)/np.array([.012]*6+[.15]+[.012]*6+[.15]), -1, 1)
                else:action = np.zeros(14)
                _, _, _, _, info = env.step(action)
                speed = np.linalg.norm(env.data.qvel[env.payload_dof:env.payload_dof+6])
                if step >= 170 and info['carried'] and any(info['grasped']) and not info['table_support'] and speed < .1:
                    return env.get_state(), env.steps
            raise RuntimeError('Test pickup did not settle')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'fixture.pt'; pickup_fixture(source)
            with patch('simulation.placement_curriculum.PickupPolicy.rollout', prepare):
                env = make_env(dict(stage=2, task_version=2, layout_mode='random_all', difficulty=0.,
                                    effort_scale=0., pickup_checkpoint=str(source)))
                try:
                    starts, goals = [], []
                    for seed in (0, 1):
                        _, info = env.reset(seed=seed)
                        self.assertTrue(info['carried'] and any(info['grasped']))
                        self.assertFalse(info['table_support'])
                        self.assert_separate(env)
                        self.assertTrue(single_arm_route(env, env.start, env.goal, env.initial_side))
                        np.testing.assert_array_equal(env.data.qpos, env._pickup.env.data.qpos)
                        np.testing.assert_array_equal(env.data.qvel, env._pickup.env.data.qvel)
                        starts.append(env.start.copy()); goals.append(env.goal.copy())
                    self.assertFalse(np.array_equal(starts[0], starts[1]))
                    self.assertFalse(np.array_equal(goals[0], goals[1]))
                finally:env.close()

    def test_switching_stage_one_distribution_preserves_model_optimizer_and_budget(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory); source = directory / 'source.pt'
            checkpoint = pickup_fixture(source)
            checkpoint['curriculum'] = Curriculum(task_version=2, layout_mode='random_separated').state_dict()
            checkpoint['env_config'] = Curriculum(task_version=2, layout_mode='random_separated').env_config()
            plan = dict(from_stage=1, at_steps=1000, to_stage=2, additional_steps=1000, done=False)
            checkpoint['config'].update(task_version=2, layout_mode='random_separated', payload_mass=1., handoff=plan)
            torch.save(checkpoint, source)
            run = directory / 'new_distribution'
            result = subprocess.run([sys.executable, '-B', '-m', 'training.train', '--resume', str(source),
                '--layout-mode', 'random_all', '--run-dir', str(run), '--total-steps', '24', '--device', 'cpu', '--no-report'],
                cwd=root, env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            initial = torch.load(run / 'transfer_initial.pt', map_location='cpu', weights_only=False)
            def same(left, right):
                if isinstance(left, torch.Tensor):self.assertTrue(torch.equal(left, right))
                elif isinstance(left, np.ndarray):np.testing.assert_array_equal(left, right)
                elif isinstance(left, dict):
                    self.assertEqual(left.keys(), right.keys())
                    for key in left:same(left[key], right[key])
                elif isinstance(left, (list, tuple)):
                    for a, b in zip(left, right):same(a, b)
                else:self.assertEqual(left, right)
            for key in ('policy', 'optimizer', 'normalizer'):same(checkpoint[key], initial[key])
            self.assertEqual(initial['env_config']['layout_mode'], 'random_all')
            self.assertEqual(initial['config']['handoff'], plan)
            self.assertEqual(initial['steps'], 16)
            final = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False)
            self.assertEqual(final['steps'], 24)
            self.assertEqual(final['curriculum']['stage'], 1)


if __name__ == '__main__':
    unittest.main()
