"""PyTorch PPO primitives shared by training and evaluation."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


class RunningNorm:
    def __init__(self, shape, passthrough=0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        self.passthrough = passthrough

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        n = len(values)
        delta = values.mean(0) - self.mean
        total = self.count + n
        self.var = (self.var*self.count + values.var(0)*n + delta**2*self.count*n/total)/total
        self.mean += delta*n/total
        self.count = total

    def __call__(self, values):
        result = np.clip((values-self.mean)/np.sqrt(self.var+1e-8), -10, 10).astype(np.float32)
        if self.passthrough:
            result[..., -self.passthrough:] = np.asarray(values)[..., -self.passthrough:]
        return result

    def state_dict(self):
        return dict(mean=self.mean.copy(), var=self.var.copy(), count=self.count,passthrough=self.passthrough)

    def load_state_dict(self, state):
        self.mean, self.var, self.count = state['mean'].copy(), state['var'].copy(), state['count']
        self.passthrough = state.get('passthrough',0)


def network(inputs, outputs, final_gain):
    result = nn.Sequential(nn.Linear(inputs,256), nn.Tanh(), nn.Linear(256,256), nn.Tanh(), nn.Linear(256,outputs))
    for layer in result:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, np.sqrt(2))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(result[-1].weight, final_gain)
    return result


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim=14):
        super().__init__()
        self.actor = network(obs_dim, action_dim, .01)
        self.critic = network(obs_dim, 1, 1.)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.5))

    @staticmethod
    def log_jacobian(latent):
        # Stable log(1-tanh(x)^2), including near the boundaries.
        return 2*(np.log(2.)-latent-F.softplus(-2*latent))

    def distribution(self, obs):
        return Normal(self.actor(obs), self.log_std.clamp(-5, 1).exp())

    def sample(self, obs, deterministic=False):
        dist = self.distribution(obs)
        latent = dist.mean if deterministic else dist.sample()
        log_prob = (dist.log_prob(latent)-self.log_jacobian(latent)).sum(-1)
        return latent.tanh(), latent, log_prob, self.critic(obs).squeeze(-1)

    def evaluate(self, obs, latent):
        dist = self.distribution(obs)
        log_prob = (dist.log_prob(latent)-self.log_jacobian(latent)).sum(-1)
        entropy = (dist.entropy()+self.log_jacobian(dist.rsample())).sum(-1)
        return log_prob, entropy, self.critic(obs).squeeze(-1)


def compute_gae(rewards, values, next_values, terminated, truncated, gamma=.999, lam=.95):
    """Timeouts bootstrap their FINAL state but never leak across episode resets."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        delta = rewards[t]+gamma*next_values[t]*(~terminated[t])-values[t]
        carry = delta+gamma*lam*(~(terminated[t]|truncated[t]))*carry
        advantages[t] = carry
    return advantages, advantages+values


def load_checkpoint(path):
    # Local checkpoints contain numpy and RNG states as well as tensor weights.
    return torch.load(path, map_location='cpu', weights_only=False)


def resolve_device(requested='auto'):
    if requested not in ('auto', 'cpu', 'cuda'):
        raise ValueError(f'Unknown device: {requested}')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install CUDA-enabled PyTorch and check nvidia-smi, or use --device cpu.')
    return torch.device('cuda' if requested != 'cpu' and torch.cuda.is_available() else 'cpu')
