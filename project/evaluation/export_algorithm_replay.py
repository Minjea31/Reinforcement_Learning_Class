"""Render frozen SB3 policies using their own saved observation normalization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle

os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('LP_NUM_THREADS','2')
import torch

from evaluation.evaluate import evaluate_policy
from evaluation.export_replay import export_scenario,select_scenarios
from training.algorithms import EvaluationPolicy
from training.history import atomic_json


def bundle_digest(folder):
    folder=Path(folder);metadata=json.loads((folder/'metadata.json').read_text())
    raw=[]
    for name in ('model.zip','normalization.pkl'):
        value=(folder/name).read_bytes()
        expected=metadata.get('files',{}).get(name)
        if expected and hashlib.sha256(value).hexdigest()!=expected:raise RuntimeError('Checkpoint is being replaced')
        raw.append(value)
    return hashlib.sha256(b''.join(raw)+(folder/'metadata.json').read_bytes()).hexdigest()


def load_bundle(folder):
    from stable_baselines3 import PPO,A2C,SAC
    folder=Path(folder);metadata=json.loads((folder/'metadata.json').read_text());config=metadata['config']
    if config.get('reward_version')=='reviewed_v3':
        raise ValueError('First-trial bundles must be evaluated with their archived source, not the v4 environment')
    algorithm=config['algorithm'];model={'ppo':PPO,'a2c':A2C,'sac':SAC}[algorithm].load(folder/'model.zip',device='cpu')
    with (folder/'normalization.pkl').open('rb') as stream:normalizer=pickle.load(stream)
    config=dict(config,gamma=.999,max_steps=config.get('max_steps',2000))
    env_config=metadata['env_config']
    checkpoint=dict(config=config,steps=metadata['steps'],obs_dim=model.observation_space.shape[0],env_config=env_config,
                    control_mode='incremental_targets',algorithm=algorithm)
    return checkpoint,EvaluationPolicy(model,algorithm),normalizer.normalize_obs


def export(bundle,output,top_k=10,episodes=50,record_only=False):
    torch.set_num_threads(1);bundle=Path(bundle);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    digest=bundle_digest(bundle);checkpoint,policy,normalizer=load_bundle(bundle);config=checkpoint['config'];stage=config['stage'];level=config['level']
    evaluation=None
    if record_only:
        selected=[dict(seed=2_100_000_000+i,episode=i+1,episode_return=None) for i in range(top_k)]
    else:
        if checkpoint['env_config'].get('arm_mask'):
            episodes = config.get('eval_episodes', episodes)
            top_k = min(top_k, episodes)
        evaluation=evaluate_policy(policy,normalizer,episodes,1_000_000_000+stage*1_000_000+level*10_000,
            config.get('max_steps',2000),.999,env_config=checkpoint['env_config'],progress_label=None)
        atomic_json(output/'evaluation.json',evaluation);selected=select_scenarios(evaluation,None,top_k)
    args=argparse.Namespace(fps=50,run_name=config['web_run_name'],save_scenes=True)
    entries=[]
    for rank,scenario in enumerate(selected,1):
        folder=f'rank-{rank:02d}'
        metadata=export_scenario(args,checkpoint,digest,evaluation,scenario,rank,output/folder,policy,normalizer,torch.device('cpu'))
        entry={key:metadata[key] for key in ('rank','seed','episode','episode_return','success','duration_seconds','motion')}
        entry.update(video=folder+'/replay.mp4',poster=folder+'/poster.jpg',metadata=folder+'/replay.json');entries.append(entry)
    manifest=dict(run_name=config['web_run_name'],algorithm=config['algorithm'],study_id=config['study_id'],
        checkpoint_sha256=digest,checkpoint_steps=checkpoint['steps'],env_config=checkpoint['env_config'],stage=stage,level=level,
        policy_selection='best_validation_checkpoint',source_checkpoint='best/model.zip',task_version=2,reward_scheme=checkpoint['env_config'].get('reward_scheme'),
        run_profile=dict(task_version=2,physics_backend='cpu',num_envs=1,layout_mode=config['layout_mode'],algorithm=config['algorithm'],study_id=config['study_id']),
        asset_base=f'/api/mission-assets/stage-{stage}/{digest}/',fps=50,selection='ranked_validation_episode_return',
        validation_episodes=episodes if evaluation else None,validation_success_rate=evaluation['success_rate'] if evaluation else None,
        entries=entries,generated_at=metadata['generated_at'])
    atomic_json(output/'replays.json',manifest)
    return output,manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True);p.add_argument('--output',required=True)
    p.add_argument('--record-only',action='store_true');p.add_argument('--episodes',type=int,default=5);args=p.parse_args()
    export(args.bundle,args.output,top_k=args.episodes,episodes=50,record_only=args.record_only)


if __name__=='__main__':main()
