"""Isolated, sequential PPO/A2C normalization and rollout ablations.

Never publishes to the training website or writes to a source training run.
Only result CSV/JSON/figures survive; copied checkpoints live in TemporaryDirectory.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import tempfile
import time

for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''

import numpy as np
import torch
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from simulation.curriculum import make_env
from training.algorithms import EpisodeSeeds
from training.history import atomic_json

CONDITIONS={'baseline':(256,False),'frozen_norm':(256,True),'long_rollout':(2048,False)}


def digest(value):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):visit(v.detach().cpu().numpy())
        elif isinstance(v,np.ndarray):h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=repr):visit(k);visit(v[k])
        elif isinstance(v,(list,tuple)):
            for item in v:visit(item)
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()


def norm_digest(env):
    return digest((env.obs_rms.mean,env.obs_rms.var,env.obs_rms.count))


def freeze_bundle(source,destination):
    for _ in range(20):
        try:
            raw=(source/'metadata.json').read_bytes();metadata=json.loads(raw)
            files={name:(source/name).read_bytes() for name in ('model.zip','normalization.pkl')}
            if raw!=(source/'metadata.json').read_bytes():continue
            if any(hashlib.sha256(value).hexdigest()!=metadata['files'][name] for name,value in files.items()):continue
            destination.mkdir(parents=True)
            for name,value in files.items():(destination/name).write_bytes(value)
            (destination/'metadata.json').write_bytes(raw)
            return metadata
        except FileNotFoundError:pass
        time.sleep(.1)
    raise RuntimeError('Could not copy a coherent best checkpoint')


def balanced_seeds(config,start,per_side):
    env=make_env(config,max_steps=2000);groups={0:[],1:[]}
    try:
        for seed in range(start,start+1024):
            env.reset(seed=seed);side=env.initial_side
            if len(groups[side])<per_side:groups[side].append(seed)
            if all(len(v)==per_side for v in groups.values()):return groups[0]+groups[1]
    finally:env.close()
    raise RuntimeError('Could not obtain balanced evaluation scenarios')


def prepare(algorithm,bundle,config,rollout,freeze,seed):
    raw=EpisodeSeeds(make_env(config,max_steps=2000),100000)
    vec=DummyVecEnv([lambda:raw])
    try:
        env=VecNormalize.load(bundle/'normalization.pkl',vec)
        model={'ppo':PPO,'a2c':A2C}[algorithm].load(bundle/'model.zip',env=env,device='cpu',n_steps=rollout)
        fingerprint=dict(policy=digest(model.policy.state_dict()),optimizer=digest(model.policy.optimizer.state_dict()),normalization=norm_digest(env))
        env.training=not freeze;env.norm_reward=False
        model.set_random_seed(seed);env.seed(100000)
        assert model.rollout_buffer.buffer_size==rollout
        return model,env,fingerprint
    except BaseException:vec.close();raise


class Guard:
    def __init__(self,source,out):self.source=source;self.out=out;self.stopped=False;self.last=0.
    def stop(self,*_):self.stopped=True
    def check(self,force=False):
        if self.stopped:raise InterruptedError('Experiment stopped')
        if not force and time.monotonic()-self.last<2:return
        while True:
            healthy=True
            for name in ('ppo','a2c','sac'):
                try:
                    state=json.loads((self.source/name/'run_status.json').read_text())
                    cmd=Path(f'/proc/{int(state["pid"])}/cmdline').read_bytes()
                    healthy &= state['phase'] in ('training','evaluating') and b'training.algorithm_study' in cmd and str(self.source/name).encode() in cmd
                except (OSError,ValueError,KeyError):healthy=False
            memory=next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
            if healthy and memory>=2*1024*1024:break
            atomic_json(self.out/'status.json',dict(phase='paused',reason='Training unavailable or less than 2 GiB free memory'))
            if self.stopped:raise InterruptedError('Experiment stopped')
            time.sleep(5)
        self.last=time.monotonic()


def evaluate(model,normalizer,config,seeds,guard):
    before=norm_digest(normalizer);env=make_env(config,max_steps=2000);records=[]
    try:
        for seed in seeds:
            guard.check();obs,_=env.reset(seed=seed);side=env.initial_side
            for _ in range(env.max_steps):
                guard.check();action,_=model.predict(normalizer.normalize_obs(obs),deterministic=True)
                obs,_,term,trunc,info=env.step(action)
                if term or trunc:break
            records.append(dict(seed=seed,side=side,success=bool(info['is_success']),grasp=bool(info['used_arms'].any()),lift=bool(info['episode_lifted']),episode_return=info['episode_return']))
    finally:env.close()
    assert before==norm_digest(normalizer),'Evaluation changed normalization statistics'
    return dict(success_rate=float(np.mean([r['success'] for r in records])),left_success=float(np.mean([r['success'] for r in records if r['side']==0])),right_success=float(np.mean([r['success'] for r in records if r['side']==1])),grasp_rate=float(np.mean([r['grasp'] for r in records])),lift_rate=float(np.mean([r['lift'] for r in records])),records=records)


class Monitor(BaseCallback):
    def __init__(self,guard,out,name,condition,episodes):
        super().__init__();self.guard=guard;self.out=out;self.name=name;self.condition=condition;self.episodes=episodes
    def _on_step(self):
        self.guard.check()
        if not np.isfinite(self.locals['rewards']).all():raise RuntimeError('Nonfinite reward')
        if self.locals['dones'][0]:
            i=self.locals['infos'][0];self.episodes.append(dict(steps=self.num_timesteps,success=bool(i['is_success']),reward=i['episode_return']))
        if self.num_timesteps%2048==0:atomic_json(self.out/'status.json',dict(phase='training',algorithm=self.name,condition=self.condition,steps=self.num_timesteps))
        return True


def classify(results):
    verdict={}
    for name in ('ppo','a2c'):
        baseline=results[name]['baseline'];reference=baseline['fresh']
        verdict[name]={}
        for condition in ('frozen_norm','long_rollout'):
            value=results[name][condition];fresh=value['fresh']
            improved=fresh['success_rate']>=reference['success_rate']+.10-1e-9
            balanced=min(fresh['left_success'],fresh['right_success'])>=min(reference['left_success'],reference['right_success'])-1e-9
            late=np.mean([r['success_rate'] for r in value['validation'][-2:]])>=np.mean([r['success_rate'] for r in baseline['validation'][-2:]])-1e-9
            verdict[name][condition]=dict(candidate=bool(improved and balanced and late),fresh_gain=fresh['success_rate']-reference['success_rate'])
    return verdict


def figures(out,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for axis,name in zip(axes,('ppo','a2c')):
        for condition,value in results[name].items():
            axis.plot([r['steps'] for r in value['validation']],[100*r['success_rate'] for r in value['validation']],marker='o',label=condition)
        axis.set(title=name.upper(),xlabel='Additional experiment steps',ylabel='Balanced validation success (%)',ylim=(0,100));axis.legend();axis.grid(alpha=.2)
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(out/('comparison.'+ext),dpi=300)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='runs/studies/three_models_1kg_v3')
    parser.add_argument('--output',required=True)
    parser.add_argument('--steps',type=int,default=131072)
    parser.add_argument('--interval',type=int,default=32768)
    parser.add_argument('--eval-per-side',type=int,default=10)
    parser.add_argument('--fresh-per-side',type=int,default=25)
    args=parser.parse_args()
    if args.steps<2048 or args.steps%2048 or args.interval%2048 or args.interval<2048 or min(args.eval_per_side,args.fresh_per_side)<1:parser.error('Positive counts and multiples of 2048 are required')
    source=Path(args.source).resolve();out=Path(args.output).resolve()
    if source in (out,*out.parents) or out in source.parents:parser.error('Output must be outside the source run')
    if out.exists() and any(out.iterdir()):parser.error('Use an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES']='';os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    import subprocess
    subprocess.run(['ionice','-c','3','-p',str(os.getpid())],check=True)
    cpu_times=lambda:{int(l.split()[0][3:]):list(map(int,l.split()[1:])) for l in Path('/proc/stat').read_text().splitlines() if l.startswith('cpu') and l[3:4].isdigit()}
    a=cpu_times();time.sleep(.3);b=cpu_times();cpu=min(os.sched_getaffinity(0),key=lambda c:1-(b[c][3]-a[c][3])/max(1,sum(b[c])-sum(a[c])))
    os.sched_setaffinity(0,{cpu});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    guard=Guard(source,out)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,guard.stop)
    results={};manifest=dict(source=str(source),steps=args.steps,interval=args.interval,seed=31,reset_seed=100000,cpu=cpu,nice=19,snapshots={})
    try:
        with tempfile.TemporaryDirectory(prefix='dual_reacher_ablation_') as temp:
            bundles={};configs={};seeds={}
            for name in ('ppo','a2c'):
                guard.check(force=True);bundle=Path(temp)/name
                metadata=freeze_bundle(source/name/'best',bundle)
                assert metadata['config']['algorithm']==name
                config=metadata['env_config'];assert config['stage']==1 and config['difficulty']==0 and config['payload_mass']==1
                manifest['snapshots'][name]=dict(steps=metadata['steps'],files=metadata['files'],env_config=config)
                bundles[name]=bundle;configs[name]=config
                seeds[name]=dict(validation=balanced_seeds(config,1001000000,args.eval_per_side),fresh=balanced_seeds(config,1500200000,args.fresh_per_side))
            sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('simulation','training') for p in Path(folder).glob('*.py')}
            atomic_json(out/'manifest.json',dict(manifest,seeds=seeds,source_hashes=sources))
            for name in ('ppo','a2c'):
                results[name]={};reference=None
                for condition,(rollout,freeze) in CONDITIONS.items():
                    guard.check(force=True);model,env,fingerprint=prepare(name,bundles[name],configs[name],rollout,freeze,31)
                    try:
                        if reference is None:reference=fingerprint
                        assert fingerprint==reference,'Conditions did not start from identical state'
                        initial_norm=norm_digest(env);initial_policy=digest(model.policy.state_dict());episodes=[];initial_updates=model._n_updates
                        value=dict(rollout=rollout,freeze=freeze,initial=fingerprint,validation=[]);callback=Monitor(guard,out,name,condition,episodes)
                        value['validation'].append(dict(steps=0,**evaluate(model,env,configs[name],seeds[name]['validation'],guard)))
                        for target in range(args.interval,args.steps+args.interval,args.interval):
                            target=min(target,args.steps)
                            previous=0 if len(value['validation'])==1 else model.num_timesteps
                            model.learn(target-previous,reset_num_timesteps=len(value['validation'])==1,callback=callback)
                            assert model.num_timesteps==target
                            assert all(torch.isfinite(p).all() for p in model.policy.parameters())
                            if freeze:assert norm_digest(env)==initial_norm
                            atomic_json(out/'status.json',dict(phase='evaluating',algorithm=name,condition=condition,steps=target))
                            value['validation'].append(dict(steps=target,**evaluate(model,env,configs[name],seeds[name]['validation'],guard)))
                            atomic_json(out/(name+'_'+condition+'_partial.json'),value)
                            if target==args.steps:break
                        assert digest(model.policy.state_dict())!=initial_policy,'Policy did not update'
                        value['normalization_changed']=norm_digest(env)!=initial_norm
                        value['optimizer_updates_added']=model._n_updates-initial_updates
                        if not freeze:assert value['normalization_changed']
                        value['fresh']=evaluate(model,env,configs[name],seeds[name]['fresh'],guard)
                        value['training_episodes']=episodes;results[name][condition]=value
                        atomic_json(out/'results.json',results)
                        (out/(name+'_'+condition+'_partial.json')).unlink(missing_ok=True)
                    finally:env.close()
                    del model,env;gc.collect()
            verdict=classify(results);atomic_json(out/'verdict.json',dict(exploratory=True,automatic_apply=False,results=verdict))
            with (out/'comparison.csv').open('w',newline='') as stream:
                fields=['algorithm','condition','steps','success_rate','left_success','right_success','grasp_rate','lift_rate'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                for name,conditions in results.items():
                    for condition,value in conditions.items():
                        for row in value['validation']:writer.writerow(dict(algorithm=name,condition=condition,**{k:row[k] for k in fields[2:]}))
            figures(out,results)
        atomic_json(out/'status.json',dict(phase='finished',temporary_checkpoints_removed=True,automatic_apply=False))
    except BaseException as exc:
        atomic_json(out/'status.json',dict(phase='stopped' if isinstance(exc,InterruptedError) else 'error',error=str(exc),temporary_checkpoints_removed=True));raise


if __name__=='__main__':main()
