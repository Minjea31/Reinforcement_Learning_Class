import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from simulation.arm_initialization import randomize_arms
from simulation.curriculum import Curriculum, make_env
from simulation.dual_reacher import MAX_ROT, quat_rotvec
from tests.test_placement_curriculum import pickup_fixture


class RandomApproachTests(unittest.TestCase):
    def env(self, stage=1, difficulty=0.):
        env = make_env(dict(stage=stage, difficulty=difficulty, effort_scale=difficulty if stage == 4 else 0.,
                            task_version=2, payload_mass=1., layout_mode='random_approach'))
        self.addCleanup(env.close)
        return env

    def test_random_poses_start_open_and_far_from_object_without_collisions(self):
        for stage, difficulty in ((1, 0.), (3, 0.), (4, 1.)):
            env = self.env(stage, difficulty)
            poses = []
            for seed in range(6):
                obs, info = env.reset(seed=seed)
                distances = np.linalg.norm(env.data.site_xpos[env.sites]-info['payload_position'], axis=1)
                self.assertTrue(np.all(distances >= .15+.05*difficulty))
                selected = distances[env.initial_side] if stage == 1 else distances.min()
                self.assertLessEqual(selected, .55+.20*difficulty)
                self.assertEqual(info['grasped'], (False, False))
                self.assertFalse(info['carried'])
                self.assertEqual(info['payload_robot_contacts'], 0)
                self.assertEqual(info['arm_contacts'], 0)
                self.assertEqual(info['obstacle_contacts'], 0)
                self.assertEqual((env.steps, env.data.time, env.episode_return), (0, 0., 0.))
                self.assertEqual(obs.shape, (105,))
                self.assertEqual(env.action_space.shape, (14,))
                np.testing.assert_array_equal(env.data.qvel, 0.)
                for joint in env.finger_joints:self.assertEqual(env.data.qpos[joint], .045)
                for joint, _ in env.joints:
                    self.assertLessEqual(np.linalg.norm(quat_rotvec(env.data.qpos[joint:joint+4])), MAX_ROT)
                self.assertTrue(env._markers_separated(env.start, env.goal))
                poses.append(env.target_action.copy())
            self.assertGreater(np.ptp(poses, axis=0)[:6].max(), .2)

    def test_pose_can_change_without_moving_object_start_or_goal(self):
        env = self.env()
        env.reset(seed=31)
        payload = env.data.qpos[env.payload_qpos:env.payload_qpos+7].copy()
        start, goal = env.start.copy(), env.goal.copy()
        first = env.target_action.copy()
        randomize_arms(env)
        np.testing.assert_array_equal(env.data.qpos[env.payload_qpos:env.payload_qpos+7], payload)
        np.testing.assert_array_equal(env.start, start)
        np.testing.assert_array_equal(env.goal, goal)
        self.assertFalse(np.array_equal(env.target_action, first))

    def test_stage_one_distance_curriculum_is_applied_at_every_level(self):
        env = self.env()
        for difficulty in (0., .25, .5, .75, 1.):
            env.configure(dict(stage=1, difficulty=difficulty, effort_scale=0.,
                               task_version=2, layout_mode='random_approach', payload_mass=1.))
            _, info = env.reset(seed=1)
            distances = np.linalg.norm(env.data.site_xpos[env.sites]-info['payload_position'], axis=1)
            self.assertGreaterEqual(distances.min(), .15+.05*difficulty)
            self.assertLessEqual(distances[env.initial_side], .55+.20*difficulty)
            self.assertFalse(any(info['grasped']))
            np.testing.assert_allclose(info['gripper_force'], 0., atol=1e-10)

    def test_same_seed_and_exact_resume_reproduce_arm_pose_and_next_reset(self):
        env = self.env()
        expected = env.reset(seed=31)[0]
        env.reset(seed=4)
        np.testing.assert_array_equal(env.reset(seed=31)[0], expected)
        action = np.linspace(-.2, .2, 14)
        env.step(action); state = env.get_state()
        expected_step = env.step(action); expected_reset = env.reset()[0]
        env.set_state(state); actual_step = env.step(action)
        np.testing.assert_array_equal(actual_step[0], expected_step[0])
        self.assertEqual(actual_step[1], expected_step[1])
        np.testing.assert_array_equal(env.reset()[0], expected_reset)

    def test_stage_two_teacher_itself_starts_from_random_ungrasped_pose(self):
        from simulation.placement_curriculum import PickupPolicy
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.pt'; pickup_fixture(source)
            pickup = PickupPolicy(source, layout_mode='random_approach')
            try:
                _, info = pickup.env.reset(seed=1)
                self.assertEqual(pickup.env.layout_mode, 'random_approach')
                self.assertFalse(any(info['grasped']))
                self.assertFalse(info['carried'])
                distances = np.linalg.norm(pickup.env.data.site_xpos[pickup.env.sites]-info['payload_position'], axis=1)
                self.assertGreaterEqual(distances.min(), .15)
            finally:pickup.close()

    def test_cli_switch_preserves_policy_adam_normalizer_and_schedule(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory); source = directory/'source.pt'
            checkpoint = pickup_fixture(source)
            curriculum = Curriculum(task_version=2, layout_mode='random_all')
            plan = dict(from_stage=1, at_steps=1000, to_stage=2, additional_steps=1000, done=False)
            checkpoint.update(env_config=curriculum.env_config(), curriculum=curriculum.state_dict())
            checkpoint['config'].update(task_version=2, layout_mode='random_all', payload_mass=1., handoff=plan)
            torch.save(checkpoint, source)
            run = directory/'approach'
            result = subprocess.run([sys.executable, '-B', '-m', 'training.train', '--resume', str(source),
                '--layout-mode', 'random_approach', '--run-dir', str(run), '--total-steps', '24', '--device', 'cpu', '--no-report'],
                cwd=root, env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            initial = torch.load(run/'transfer_initial.pt', map_location='cpu', weights_only=False)
            def same(a, b):
                if isinstance(a, torch.Tensor):self.assertTrue(torch.equal(a, b))
                elif isinstance(a, np.ndarray):np.testing.assert_array_equal(a, b)
                elif isinstance(a, dict):
                    self.assertEqual(a.keys(), b.keys())
                    for key in a:same(a[key], b[key])
                elif isinstance(a, (tuple, list)):
                    for x, y in zip(a, b):same(x, y)
                else:self.assertEqual(a, b)
            for key in ('policy', 'optimizer', 'normalizer'):same(checkpoint[key], initial[key])
            self.assertEqual(initial['steps'], 16)
            self.assertEqual(initial['config']['handoff'], plan)
            self.assertEqual(initial['env_config']['layout_mode'], 'random_approach')
            final = torch.load(run/'latest.pt', map_location='cpu', weights_only=False)
            self.assertEqual(final['steps'], 24)


if __name__ == '__main__':unittest.main()
