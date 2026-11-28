"""On-policy Gaussian actions transformed by tanh, with corrected log density.

SB3's default PPO/A2C Gaussian is unbounded and clips only the action passed to
the simulator. Here the rollout action, its likelihood and the applied action
all describe the same bounded random variable. SAC already uses this transform.
"""
from gymnasium import spaces
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class BoundedActorCriticPolicy(ActorCriticPolicy):
    def _build(self, lr_schedule):
        if not isinstance(self.action_space, spaces.Box) or self.use_sde:
            raise ValueError('Bounded policy requires continuous Box actions without gSDE')
        # The squashed distribution is a DiagGaussian subclass and uses exactly
        # the same action_net/log_std parameters and optimizer as the base class.
        self.action_dist=SquashedDiagGaussianDistribution(self.action_space.shape[0])
        super()._build(lr_schedule)
        self._squash_output=True
