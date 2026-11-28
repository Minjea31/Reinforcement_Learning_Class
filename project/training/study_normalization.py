"""Normalize sensors while preserving bounded task-state and context values."""
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize


class ContextNormalize(VecNormalize):
    def __init__(self,venv,*,preserve_task_state=True,**kwargs):
        self.preserve_task_state=preserve_task_state
        super().__init__(venv,**kwargs)

    def normalize_obs(self,obs):
        result=super().normalize_obs(obs)
        result[...,-6:]=np.asarray(obs)[...,-6:]
        # Rare hold/grip events otherwise acquire tiny running variances and
        # clip to the same value long before the required hold time is reached.
        # Old saved normalizers have no flag and retain their original mapping.
        if getattr(self,'preserve_task_state',False):
            result[...,82:89]=np.asarray(obs)[...,82:89]
            result[...,98]=np.asarray(obs)[...,98]
        return result
