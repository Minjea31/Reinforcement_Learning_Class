import unittest
import numpy as np
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from training.algorithms import ALGORITHMS, EpisodeSeeds
from simulation.curriculum import make_env


class AlgorithmStudyTests(unittest.TestCase):
    def test_stage_one_best_model_prioritizes_actual_progress_over_idle_return(self):
        from evaluation.evaluate import curriculum_ranking
        idle=dict(success_rate=0.,grasp_rate=0.,lift_rate=0.,mean_return=-10.)
        gripping=dict(idle,grasp_rate=.5,mean_return=-18.)
        lifting=dict(gripping,lift_rate=.25,mean_return=-20.)
        success=dict(idle,success_rate=.0625,mean_return=-25.)
        ranks=[curriculum_ranking(v,1,2) for v in (idle,gripping,lifting,success)]
        self.assertEqual(sorted(ranks),ranks)

    def test_task_clocks_stay_distinguishable_with_rare_event_statistics(self):
        from stable_baselines3.common.vec_env import DummyVecEnv
        from training.study_normalization import ContextNormalize
        config=dict(stage=1,difficulty=0.,effort_scale=0.,task_version=2,layout_mode='progressive',reward_scheme='reviewed')
        env=ContextNormalize(DummyVecEnv([lambda:make_env(config)]))
        try:
            env.reset();env.obs_rms.mean[:]=0.;env.obs_rms.var[:]=1e-8
            data=np.zeros((2,105),dtype=np.float32)
            data[:,82:89]=[[.4,1.,0.,1.,0.,1.,.2],[.8,1.,0.,1.,0.,1.,.4]]
            data[:,98]=.35;data[:,-6:]=[1.,0.,0.,0.,.25,0.]
            normalized=env.normalize_obs(data)
            np.testing.assert_array_equal(normalized[:,82:89],data[:,82:89])
            np.testing.assert_array_equal(normalized[:,98:],data[:,98:])
            self.assertNotEqual(normalized[0,88],normalized[1,88])
            # Loading an older pickle must preserve its original semantics.
            del env.preserve_task_state
            self.assertEqual(env.normalize_obs(data)[0,88],env.normalize_obs(data)[1,88])
        finally:env.close()

    def test_on_policy_actions_and_log_probabilities_agree_and_survive_reload(self):
        import torch
        from argparse import Namespace
        from stable_baselines3 import PPO,A2C
        from stable_baselines3.common.vec_env import DummyVecEnv
        from training.algorithms import build_model
        config=dict(stage=1,difficulty=0.,effort_scale=0.,task_version=2,layout_mode='progressive',reward_scheme='reviewed')
        torch.set_num_threads(1)
        for algorithm,cls in [('ppo',PPO),('a2c',A2C)]:
            env=DummyVecEnv([lambda:make_env(config,max_steps=8)])
            try:
                args=Namespace(seed=31,device='cpu',rollout_steps=16)
                model=build_model(algorithm,env,args)
                obs=torch.as_tensor(np.repeat(env.reset(),128,axis=0))
                with torch.no_grad():
                    actions,_,sample_log_prob=model.policy(obs)
                    _,evaluated_log_prob,_=model.policy.evaluate_actions(obs,actions)
                self.assertTrue(model.policy.squash_output)
                self.assertLess(float(actions.abs().max()),1.)
                torch.testing.assert_close(sample_log_prob,evaluated_log_prob)
                self.assertEqual(model.ent_coef,0.)
                model.learn(32)
                self.assertGreater(model._n_updates,0)
                self.assertTrue(all(torch.isfinite(p).all() for p in model.policy.parameters()))
                with tempfile.TemporaryDirectory() as folder:
                    path=Path(folder)/'model.zip';model.save(path)
                    loaded=cls.load(path,env=env,device='cpu')
                    self.assertTrue(loaded.policy.squash_output)
                    np.testing.assert_array_equal(model.predict(obs.numpy(),deterministic=True)[0],loaded.predict(obs.numpy(),deterministic=True)[0])
                    loaded.learn(16,reset_num_timesteps=False)
                    self.assertEqual(loaded.num_timesteps,48)
            finally:env.close()

    def test_one_failed_algorithm_does_not_signal_the_other_models(self):
        from training.compare import main
        created=[]
        class Process:
            def __init__(self,command,**kwargs):
                self.name=command[command.index('--algorithm')+1];self.pid=900000+len(created);self.calls=0;self.signals=[];created.append(self)
            def poll(self):
                self.calls+=1
                return 1 if self.name=='ppo' else 0 if self.calls>2 else None
            @property
            def returncode(self):return 1 if self.name=='ppo' else 0
            def send_signal(self,value):self.signals.append(value)
            def wait(self):return self.returncode
        with tempfile.TemporaryDirectory() as directory,patch('sys.argv',['compare','--run-dir',directory,'--no-web','--total-steps','16','--rollout-steps','16']),patch('training.compare.subprocess.Popen',Process),patch('training.compare.time.sleep'):
            self.assertEqual(main(),1)
        self.assertEqual({p.name for p in created},{'ppo','a2c','sac'})
        self.assertTrue(all(not p.signals for p in created))

    def test_resume_skips_models_that_already_finished_their_budget(self):
        from training.compare import main
        with tempfile.TemporaryDirectory() as directory:
            for name in ALGORITHMS:
                folder=Path(directory)/name/'latest';folder.mkdir(parents=True);(folder/'metadata.json').write_text(json.dumps(dict(steps=16)))
            with patch('sys.argv',['compare','--run-dir',directory,'--no-web','--total-steps','16','--rollout-steps','16','--resume']),patch('training.compare.subprocess.Popen') as launch:
                self.assertEqual(main(),0);launch.assert_not_called()

    def test_only_three_continuous_control_algorithms_are_available(self):
        self.assertEqual(ALGORITHMS,('ppo','a2c','sac'))

    def test_nth_episode_layout_matches_across_independent_algorithms(self):
        config=dict(stage=1,difficulty=0.,effort_scale=0.,task_version=2,layout_mode='random_approach')
        left=EpisodeSeeds(make_env(config),31);right=EpisodeSeeds(make_env(config),31)
        try:
            for _ in range(3):
                a,info_a=left.reset();b,info_b=right.reset()
                np.testing.assert_array_equal(a,b)
                self.assertEqual(info_a['scenario_seed'],info_b['scenario_seed'])
                left.step(np.ones(14));right.step(-np.ones(14))
        finally:left.close();right.close()


if __name__=='__main__':unittest.main()
