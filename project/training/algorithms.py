"""Independent continuous-control PPO, A2C and SAC models."""
import gymnasium as gym
import numpy as np
import torch
from torch import nn


ALGORITHMS = ('ppo', 'a2c', 'sac')


class EpisodeSeeds(gym.Wrapper):
    """Same nth reset seed across algorithms, independent of episode length."""
    def __init__(self, env, first_seed):
        super().__init__(env)
        self.next_seed = first_seed

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.next_seed = seed
        chosen = self.next_seed
        self.next_seed += 1
        obs, info = self.env.reset(seed=chosen, options=options)
        return obs, dict(info, scenario_seed=chosen)


def build_model(algorithm, env, args):
    from stable_baselines3 import A2C, PPO, SAC
    classes = dict(ppo=PPO, a2c=A2C, sac=SAC)
    common = dict(policy='MlpPolicy', env=env, seed=args.seed, device=args.device,
                  learning_rate=3e-4, gamma=.999, verbose=0,
                  policy_kwargs=dict(net_arch=[256, 256], activation_fn=nn.Tanh))
    if algorithm in ('ppo', 'a2c'):
        from training.bounded_policy import BoundedActorCriticPolicy
        common.update(policy=BoundedActorCriticPolicy,learning_rate=1e-4,
                      n_steps=args.rollout_steps,gae_lambda=.95,ent_coef=0.)
        common['policy_kwargs']['log_std_init']=-1.
        if algorithm == 'ppo':
            common.update(learning_rate=3e-5,batch_size=min(256,args.rollout_steps),
                          n_epochs=5,clip_range=.1,target_kl=.005)
    else:
        common.update(buffer_size=args.buffer_size, learning_starts=args.learning_starts,
                      batch_size=min(256, args.buffer_size), train_freq=1, gradient_steps=1)
    return classes[algorithm](**common)


class EvaluationPolicy(nn.Module):
    """Adapt SB3 prediction to the existing frozen-policy evaluation routine."""
    def __init__(self, model, algorithm):
        super().__init__()
        self.model, self.algorithm = model, algorithm
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)

    def sample(self, observations, deterministic=False):
        actions, _ = self.model.predict(observations.cpu().numpy(), deterministic=deterministic)
        return torch.as_tensor(actions), None, None, None
