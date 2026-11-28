"""Freeze a completed SB3 grasp policy to prepare real stage-two starts."""
import hashlib
import json
from pathlib import Path
import pickle

from stable_baselines3 import A2C, PPO, SAC

from simulation.curriculum import make_env
from simulation.reward_design import STUDY_REWARD_VERSIONS


class FrozenStudyPickup:
    def __init__(self, bundle):
        bundle = Path(bundle)
        metadata = json.loads((bundle/'metadata.json').read_text())
        config = metadata['env_config']
        if config['stage'] != 1 or config.get('reward_scheme') not in STUDY_REWARD_VERSIONS:
            raise ValueError('Stage two requires a reviewed stage-one pickup checkpoint')
        self.source_steps = metadata['steps']
        self.source_hash = metadata['files']['model.zip']
        normalizer = (bundle/'normalization.pkl').read_bytes()
        if hashlib.sha256(normalizer).hexdigest() != metadata['files']['normalization.pkl']:
            raise ValueError('The frozen pickup normalizer does not match the checkpoint')
        self.normalizer = pickle.loads(normalizer)
        algorithm = {'ppo': PPO, 'a2c': A2C, 'sac': SAC}[metadata['config']['algorithm']]
        self.policy = algorithm.load(bundle/'model.zip', device='cpu')
        self.env = make_env(config, max_steps=metadata['config']['max_steps'])

    def rollout(self, seed):
        obs, _ = self.env.reset(seed=seed)
        for _ in range(self.env.max_steps):
            action, _ = self.policy.predict(self.normalizer.normalize_obs(obs), deterministic=True)
            obs, _, terminated, truncated, info = self.env.step(action)
            if info['is_success']:
                return self.env.get_state(), self.env.steps
            if terminated or truncated:
                break
        raise RuntimeError('Frozen stage-one policy did not complete a stable lift on this reset')

    def close(self):
        self.env.close()
