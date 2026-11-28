import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from training.study_normalization import ContextNormalize
from evaluation.stability_experiment import prepare,digest,norm_digest,classify


class Toy(gym.Env):
    observation_space=gym.spaces.Box(-100,100,(105,),np.float32)
    action_space=gym.spaces.Box(-1,1,(14,),np.float32)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.steps=0
        return np.zeros(105,np.float32),{}
    def step(self,action):
        self.steps+=1
        return np.full(105,self.steps,np.float32),float(action.sum()),False,self.steps==8,{}


class StabilityExperimentTests(unittest.TestCase):
    def test_clones_restore_optimizer_and_build_the_requested_buffer(self):
        torch.set_num_threads(1)
        for name,cls in [('ppo',PPO),('a2c',A2C)]:
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory);env=ContextNormalize(DummyVecEnv([Toy]),norm_reward=False)
                kwargs=dict(n_steps=8,policy_kwargs=dict(net_arch=[16,16]),device='cpu')
                if name=='ppo':kwargs.update(batch_size=8,n_epochs=1)
                model=cls('MlpPolicy',env,**kwargs);model.learn(16)
                expected=dict(policy=digest(model.policy.state_dict()),optimizer=digest(model.policy.optimizer.state_dict()),normalization=norm_digest(env))
                model.save(path/'model.zip');env.save(path/'normalization.pkl');env.close()
                for rollout,freeze in ((256,False),(256,True),(2048,False)):
                    with patch('evaluation.stability_experiment.make_env',side_effect=lambda *a,**k:Toy()):
                        clone,normalized,fingerprint=prepare(name,path,{},rollout,freeze,31)
                    try:
                        self.assertEqual(fingerprint,expected)
                        self.assertEqual(clone.rollout_buffer.buffer_size,rollout)
                        before=norm_digest(normalized);normalized.reset();normalized.step(np.zeros((1,14)))
                        self.assertEqual(before==norm_digest(normalized),freeze)
                    finally:normalized.close()

    def test_candidate_needs_gain_and_no_weak_side_regression(self):
        def value(rate,left,right):
            return dict(fresh=dict(success_rate=rate,left_success=left,right_success=right),validation=[dict(success_rate=rate)]*2)
        conditions=dict(baseline=value(.5,.6,.4),frozen_norm=value(.6,.7,.5),long_rollout=value(.7,1.,.3))
        result=classify(dict(ppo=conditions,a2c=conditions))
        self.assertTrue(result['ppo']['frozen_norm']['candidate'])
        self.assertFalse(result['ppo']['long_rollout']['candidate'])


if __name__=='__main__':unittest.main()
