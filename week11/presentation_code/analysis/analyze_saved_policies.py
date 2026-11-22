"""Read-only diagnostics on the same 30 validation episodes, without training."""
from pathlib import Path
import sys,json
import numpy as np
C=Path('/home/a/Reinforcement_Learning_class/week11/assignment');D=Path(__file__).resolve().parent;sys.path.insert(0,str(C))
from backend import make_env
from evaluate import SavedPolicy
S=json.loads((C/'results/summary.json').read_text());records={}
for r in S['results']:
 pol=SavedPolicy(C/f"results/{r['key']}/actor.npz");env=make_env();xs=[];vs=[];entropies=[];counts=np.zeros(3,int);maxps=[];totals=[];goals=[]
 for i in range(30):
  state,_=env.reset(seed=500000+i);rng=np.random.default_rng(600000+i);total=0
  while True:
   p=pol.probabilities(state);p=p/p.sum();entropies.append(float(-(p*np.log(np.maximum(p,1e-300))).sum()));maxps.append(float(p.max()));xs.append(float(state[0]));vs.append(float(state[1]));action=int(rng.choice(3,p=p));counts[action]+=1
   state,reward,term,trunc,_=env.step(action);total+=reward
   if term or trunc:
    xs.append(float(state[0]));vs.append(float(state[1]));goals.append(bool(term));break
  totals.append(total)
 env.close();assert totals==r['validation']['returns']
 training=r['train_returns'];success_ids=[i+1 for i,x in enumerate(training) if x>-200]
 records[r['key']]=dict(config=r['config'],validation_returns_reproduced=True,validation_mean=r['validation']['mean'],validation_std=r['validation']['std'],validation_success=r['validation']['success'],min_position=min(xs),max_position=max(xs),mean_abs_velocity=float(np.abs(vs).mean()),max_abs_velocity=float(np.abs(vs).max()),action_fractions=(counts/counts.sum()).tolist(),mean_policy_entropy=float(np.mean(entropies)),confident_state_fraction=float(np.mean(np.array(maxps)>=.99)),train_goal_episodes=len(success_ids),first_train_goal_episode=success_ids[0] if success_ids else None,last100_train_mean=float(np.mean(training[-100:])),best_training_reward=max(training))
(D/'policy_diagnostics.json').write_text(json.dumps(records,indent=2));print('Verified diagnostics for',len(records),'saved policies')
for phase,base,sweeps in [('basic',S['base'],S['sweeps']),('additional',S['refinement_base'],S['refinement_sweeps'])]:
 print(phase)
 for name,values in sweeps.items():
  print(name)
  for v in values:
   r=next(r for r in S['results'] if r['config']==dict(base,**{name:v}));a=records[r['key']]
   print(v,'max_x',round(a['max_position'],3),'|v|',round(a['mean_abs_velocity'],3),'actions',np.round(a['action_fractions'],3).tolist(),'H',round(a['mean_policy_entropy'],3),'confident',round(a['confident_state_fraction'],3),'train_goals',a['train_goal_episodes'],'first',a['first_train_goal_episode'])
