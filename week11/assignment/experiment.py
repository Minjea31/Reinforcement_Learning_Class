"""Lecture Practice #3: native OpenAI Gym + per-transition 1-step Actor-Critic.
No reward modification, n-step target, replay, entropy term or greedy evaluation.
"""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,time,hashlib
from backend import ROOT,make_env
import numpy as np
from dezero import Model,no_grad
import dezero.layers as L
import dezero.functions as F
from dezero.optimizers import Adam
RESULTS=ROOT/'results';RESULTS.mkdir(exist_ok=True)
BASE=dict(gamma=.98,actor_lr=.0002,critic_lr=.0005,width=128,depth=1,beta1=.9,beta2=.999,velocity_scale=100.,episodes=1000)
SWEEPS=dict(gamma=[.9,.98,.995],actor_lr=[.0002,.001,.003],critic_lr=[.0005,.001,.003],width=[32,64,128],depth=[1,2,3],beta1=[.8,.9,.95],velocity_scale=[1.,100.,1000.],episodes=[500,1000,2000])
PROTOCOL='gym0262_dezero_online_td1_raw_reward_categorical_v4'
REFINEMENT_BASE=dict(BASE,width=256,velocity_scale=10000.)
REFINEMENT_SWEEPS=dict(gamma=[.9,.98,.995],actor_lr=[.00005,.0002,.001],critic_lr=[.0001,.0005,.003],width=[64,128,256],beta1=[.8,.9,.95])
class Network(Model):
 def __init__(self,width,depth,out,actor=False,velocity_scale=1.):
  super().__init__();self.depth=depth;self.actor=actor;self.velocity_scale=velocity_scale
  for i in range(depth):setattr(self,f'l{i}',L.Linear(width))
  setattr(self,f'l{depth}',L.Linear(out))
 def forward(self,x):
  x=x*np.array([1.,self.velocity_scale],dtype=np.float32)
  for i in range(self.depth):x=F.relu(getattr(self,f'l{i}')(x))
  x=getattr(self,f'l{self.depth}')(x)
  return F.softmax(x) if self.actor else x
class Agent:
 def __init__(self,c):
  self.c=c;self.pi=Network(c['width'],c['depth'],3,True,c['velocity_scale']);self.v=Network(c['width'],c['depth'],1,False,c['velocity_scale'])
  self.opt_pi=Adam(c['actor_lr'],c['beta1'],c['beta2']).setup(self.pi)
  self.opt_v=Adam(c['critic_lr'],c['beta1'],c['beta2']).setup(self.v);self.updates=0
 def get_action(self,state):
  probs=self.pi(state[np.newaxis,:])[0]
  p=probs.data.astype(np.float64);p/=p.sum()
  action=int(np.random.choice(3,p=p))
  return action,probs[action]
 def update(self,state,action_prob,reward,next_state,done):
  # The lecture's done mask includes goal and the environment's time limit.
  state=state[np.newaxis,:];next_state=next_state[np.newaxis,:]
  target=reward+self.c['gamma']*self.v(next_state)*(1-done)
  target.unchain()
  value=self.v(state)
  loss_v=F.mean_squared_error(value,target)
  delta=target-value;delta.unchain()
  loss_pi=-F.log(action_prob)*delta
  self.v.cleargrads();self.pi.cleargrads()
  loss_v.backward();loss_pi.backward()
  self.opt_v.update();self.opt_pi.update();self.updates+=1
 def save(self,d):
  self.pi.save_weights(d/'actor.npz');self.v.save_weights(d/'critic.npz')
  for file in ['actor.npz','critic.npz']:
   arrays=dict(np.load(d/file));arrays['_velocity_scale']=np.array(self.c['velocity_scale']);np.savez_compressed(d/file,**arrays)

def key(c):return hashlib.sha256(json.dumps(dict(protocol=PROTOCOL,**c),sort_keys=True).encode()).hexdigest()[:12]
def evaluate(agent,n=30,split='validation'):
 from evaluate import evaluate_policy
 def probabilities(state):
  with no_grad():return agent.pi(np.asarray(state)[None])[0].data
 return evaluate_policy(probabilities,n,split)
def train(c,tag='sweep',output=None):
 d=Path(output or RESULTS)/key(c);d.mkdir(parents=True,exist_ok=True)
 if (d/'result.json').exists():
  cached=json.loads((d/'result.json').read_text());assert cached['protocol']==PROTOCOL;return cached
 np.random.seed(0);agent=Agent(c);env=make_env();history=[];start=time.time();returns=[];steps=0
 for episode in range(c['episodes']):
  state,_=env.reset(seed=0 if episode==0 else None);total=0
  while True:
   action,prob=agent.get_action(state)
   next_state,reward,terminated,truncated,_=env.step(action)
   done=terminated or truncated
   agent.update(state,prob,reward,next_state,done)
   steps+=1;total+=reward;state=next_state
   if done:break
  returns.append(total)
  if (episode+1)%100==0:history.append(dict(episode=episode+1,steps=steps,train_mean=float(np.mean(returns[-100:]))))
 env.close();assert agent.updates==steps
 validation=evaluate(agent)
 agent.save(d)
 result=dict(protocol=PROTOCOL,config=c,key=key(c),tag=tag,train_seed=0,validation=validation,history=history,train_returns=returns,total_steps=steps,optimizer_updates=agent.updates,seconds=time.time()-start)
 (d/'result.json').write_text(json.dumps(result,indent=2));print(tag,key(c),validation['mean'],validation['success'],round(result['seconds'],1),flush=True);return result

def settings(base,sweeps):
 configs={key(base):base}
 for name,vals in sweeps.items():
  for value in vals:
   c=dict(base,**{name:value});configs[key(c)]=c
 return configs

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 from concurrent.futures import ProcessPoolExecutor
 basic=settings(BASE,SWEEPS);extra=settings(REFINEMENT_BASE,REFINEMENT_SWEEPS)
 configs={**basic,**extra}
 with ProcessPoolExecutor(a.workers) as pool:results=list(pool.map(train,configs.values()))
 def rank(c):
  r=next(r for r in results if r['key']==key(c));return r['validation']['mean'],-r['validation']['std']
 def combine(base,sweeps):
  c=dict(base)
  for name,vals in sweeps.items():c[name]=max([dict(base,**{name:v}) for v in vals],key=rank)[name]
  return c
 combination=combine(BASE,SWEEPS);refined=combine(REFINEMENT_BASE,REFINEMENT_SWEEPS)
 combo=train(combination,'combination',ROOT/'combination_retrain'/'basic')
 rcombo=train(refined,'refinement_combination',ROOT/'combination_retrain'/'additional')
 for c,r in [(combination,combo),(refined,rcombo)]:
  if key(c) not in configs:
   configs[key(c)]=c;results.append(r)
   import shutil
   source=ROOT/'combination_retrain'/('basic' if c==combination else 'additional')/key(c)
   shutil.copytree(source,RESULTS/key(c),dirs_exist_ok=True)
 winner=max(configs.values(),key=rank)
 from evaluate import SavedPolicy,evaluate_policy
 pol=SavedPolicy(RESULTS/key(winner)/'actor.npz');test=evaluate_policy(pol.probabilities,100,'test')
 summary=dict(protocol=PROTOCOL,base=BASE,sweeps=SWEEPS,refinement_base=REFINEMENT_BASE,refinement_sweeps=REFINEMENT_SWEEPS,configs=list(configs.values()),results=results,combination=combination,combination_result=combo,refinement_combination=refined,refinement_combination_result=rcombo,winner=winner,winner_key=key(winner),test=test,
 metadata=dict(environment='OpenAI Gym 0.26.2 MountainCar-v0',method='DeZero per-step 1-step TD Actor-Critic',policy='categorical sampling in training, validation, test and video',reward='original -1 per step',done='terminated or truncated as lecture done',train_seed=0,validation_episodes=30,test_episodes=100,basic_settings=len(basic),additional_settings=len(extra),training_runs=len(basic)+len(extra)+2))
 (RESULTS/'summary.json').write_text(json.dumps(summary,indent=2));print('WINNER',winner,'TEST',test['mean'],test['success'],flush=True)
if __name__=='__main__':main()
