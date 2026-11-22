"""Verify lecture equivalence, native Gym use, all policies and saved results."""
from pathlib import Path
import json,inspect
from backend import ROOT,make_env,gym
import numpy as np
from experiment import Agent,Network,BASE,PROTOCOL,key
from evaluate import SavedPolicy,evaluate_policy
from dezero import no_grad
from dezero.optimizers import SGD

def check_one_step():
 records=[]
 for done in [False,True]:
  c=dict(BASE,width=4,depth=0,velocity_scale=1.)
  a=Agent(c)
  a.pi.l0.W.data=np.zeros((2,3),np.float32);a.pi.l0.b.data=np.zeros(3,np.float32)
  a.v.l0.W.data=np.array([[.3],[-.2]],np.float32);a.v.l0.b.data=np.array([.1],np.float32)
  a.opt_pi=SGD(.01).setup(a.pi);a.opt_v=SGD(.01).setup(a.v)
  s=np.array([.1,.02],np.float32);ns=np.array([.2,.01],np.float32)
  oldW=a.v.l0.W.data.copy();oldb=a.v.l0.b.data.copy()
  value=float((s@oldW+oldb).item());nvalue=float((ns@oldW+oldb).item())
  target=-1+c['gamma']*nvalue*(1-done);gradient=2*(value-target)
  probability=a.pi(s[None])[0][2]
  a.update(s,probability,-1.,ns,done)
  assert np.allclose(a.v.l0.W.data,oldW-.01*gradient*s[:,None],rtol=0,atol=1e-6)
  assert np.allclose(a.v.l0.b.data,oldb-.01*gradient,rtol=0,atol=1e-6)
  with no_grad():selected_probability=float(a.pi(s[None])[0].data[2])
  assert selected_probability<1/3 and a.updates==1
  records.append(dict(done=done,correct_value_gradient=True,negative_td_reduces_selected_probability=True))
 e=make_env();assert e.unwrapped.__class__.__module__=='gym.envs.classic_control.mountain_car';assert gym.__version__=='0.26.2';assert e.spec.max_episode_steps==200 and e.action_space.n==3;e.close()
 return records

def main():
 s=json.loads((ROOT/'results/summary.json').read_text());assert s['protocol']==PROTOCOL
 td=check_one_step();checked=[]
 for name,vs in s['sweeps'].items():
  for v in vs:
   c=dict(s['base'],**{name:v});assert c in s['configs']
   assert sum(c[k]!=s['base'][k] for k in c)<=1
 for r in s['results']:
  assert r['protocol']==PROTOCOL and r['key']==key(r['config'])
  assert r['optimizer_updates']==r['total_steps']==-sum(r['train_returns'])
  assert len(r['train_returns'])==r['config']['episodes']
  weights=ROOT/f"results/{r['key']}/actor.npz";p=SavedPolicy(weights)
  actual=evaluate_policy(p.probabilities,30,'validation')
  assert actual['returns']==r['validation']['returns'],r['key']
  for k in ['mean','std','success']:assert np.isclose(actual[k],r['validation'][k],rtol=0,atol=1e-12)
  network=Network(r['config']['width'],r['config']['depth'],3,True,r['config']['velocity_scale']);network.load_weights(weights)
  rng=np.random.default_rng(245)
  for _ in range(15):
   state=np.array([rng.uniform(-1.2,.6),rng.uniform(-.07,.07)],np.float32)
   with no_grad():expected=network(state[None])[0].data.astype(np.float64);expected/=expected.sum()
   assert np.allclose(p.probabilities(state),expected,rtol=0,atol=3e-6)
  checked.append(r['key'])
 best=max(s['results'],key=lambda r:(r['validation']['mean'],-r['validation']['std']));assert best['key']==s['winner_key']
 p=SavedPolicy(ROOT/f"results/{s['winner_key']}/actor.npz");test=evaluate_policy(p.probabilities,100,'test',True)
 assert test['returns']==s['test']['returns']
 for ep in test['episodes']:assert ep['reward']==-len(ep['actions']) and len(ep['trajectory'])==len(ep['actions'])+1
 short=next(r for r in s['results'] if r['config']==dict(s['base'],episodes=500));long=next(r for r in s['results'] if r['config']==dict(s['base'],episodes=2000));assert short['train_returns']==long['train_returns'][:500]
 assert s['combination_result']['tag']=='combination'
 result=dict(passed=True,protocol=PROTOCOL,environment_class='gym.envs.classic_control.mountain_car.MountainCarEnv',one_update_per_environment_transition=True,raw_environment_reward=True,categorical_training_validation_test=True,exact_lecture_td_gradient=td,all_saved_model_probabilities_match_DeZero=True,all_validation_episode_rewards_reproduced=True,settings=len(checked),independent_episodes=100,training_budget_prefix_matches=True)
 (ROOT/'results/validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
