"""One immutable observation transform shared by the three masked policies."""
from dataclasses import dataclass
import json
import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np

from simulation.curriculum import make_env
from training.history import atomic_json


VERSION = 'masked_fixed_v1'
CALIBRATION_SEED = 900_000_000
PASSTHROUGH = [*range(82, 89), *range(98, 107)]


@dataclass(frozen=True)
class FixedStudyNormalize:
    mean: np.ndarray
    scale: np.ndarray

    def normalize_obs(self, obs):
        raw = np.asarray(obs)
        transformed = np.clip((raw - self.mean) / self.scale, -10., 10.).astype(np.float32)
        transformed[..., PASSTHROUGH] = raw[..., PASSTHROUGH]
        return transformed

    def save(self, path):
        with Path(path).open('wb') as stream:
            pickle.dump(self, stream)


class FixedObservation(gym.ObservationWrapper):
    """Transform observations before VecEnv handles resets and final states."""

    def __init__(self, env, normalizer):
        super().__init__(env)
        self.normalizer = normalizer

    def observation(self, observation):
        return self.normalizer.normalize_obs(observation)


def load_fixed_stats(path):
    data = json.loads(Path(path).read_text())
    mean = np.asarray(data['mean'], dtype=np.float64)
    scale = np.asarray(data['scale'], dtype=np.float64)
    if (data['version'] != VERSION or mean.shape != (107,) or scale.shape != (107,)
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError('Invalid fixed observation statistics for the masked study')
    mean.setflags(write=False)
    scale.setflags(write=False)
    return FixedStudyNormalize(mean, scale)


def prepare_fixed_stats(path):
    """Fit from independent physical traces, before any policy is trained."""
    path = Path(path)
    if path.exists():
        return load_fixed_stats(path)

    env = make_env(dict(stage=1, difficulty=0., effort_scale=0., payload_mass=1.,
                        task_version=2, layout_mode='progressive', reward_scheme='reviewed',
                        arm_mask='single'))
    rng = np.random.default_rng(8681)
    delta = np.tile([.012]*6 + [.15], 2)
    observations = []
    seeds = list(range(CALIBRATION_SEED, CALIBRATION_SEED + 12))
    try:
        for index, seed in enumerate(seeds):
            obs, _ = env.reset(seed=seed)
            observations.append(obs)
            side = env.initial_side
            start = env.data.site_xpos[env.sites[side]].copy()
            pickup = env.start + [0., 0., .005]
            for step in range(300):
                if index % 4 < 2:
                    targets = env.data.site_xpos[env.sites].copy()
                    if step < 100:
                        targets[side] = start + (pickup - start) * ((step + 1) / 100)
                    else:
                        targets[side] = pickup + [0., 0., .16 * np.clip((step - 130) / 100, 0., 1.)]
                    desired = env.solve_ik(targets, iterations=8, seed_action=env.target_action)
                    desired[6 + 7*side] = 1. if step < 100 else -1.
                    action = (desired - env.target_action) / delta
                else:
                    action = rng.normal(0., .45, 14)
                obs, _, terminated, truncated, _ = env.step(np.clip(action, -1., 1.))
                observations.append(obs)
                if terminated or truncated:
                    break
    finally:
        env.close()

    values = np.asarray(observations, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite calibration observations')
    mean, scale = values.mean(axis=0), np.maximum(values.std(axis=0), .03)
    mean[PASSTHROUGH] = 0.
    scale[PASSTHROUGH] = 1.
    atomic_json(path, dict(version=VERSION, mean=mean.tolist(), scale=scale.tolist(),
                           calibration_seeds=seeds, observations=len(observations)))
    return load_fixed_stats(path)
