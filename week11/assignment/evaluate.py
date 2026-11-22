"""Saved DeZero policy inference, native OpenAI Gym categorical evaluation and video."""
from pathlib import Path
import json
import numpy as np
from backend import ROOT,make_env
class SavedPolicy:
 def __init__(self,path):
  self.weights=dict(np.load(path));self.depth=len([k for k in self.weights if k.endswith('/W')])-1
 def logits(self,state):
  original=np.asarray(state);single=original.ndim==1
  x=original[None] if single else original
  x=x*np.array([1.,float(self.weights.get('_velocity_scale',1.))],dtype=np.float32)
  for i in range(self.depth):x=np.maximum(x.dot(self.weights[f'l{i}/W'])+self.weights[f'l{i}/b'],0)
  z=x.dot(self.weights[f'l{self.depth}/W'])+self.weights[f'l{self.depth}/b']
  return z[0] if single else z
 def probabilities(self,state):
  # Match DeZero's softmax dtype and operation order before float64 sampling.
  z=self.logits(state);ex=np.exp(z-np.max(z,axis=-1,keepdims=True));ex/=ex.sum(axis=-1,keepdims=True)
  ex=ex.astype(np.float64);return ex/ex.sum(axis=-1,keepdims=True)
def evaluate_policy(probabilities,n=30,split='validation',trajectories=False):
 env=make_env();base=500000 if split=='validation' else 900000;episodes=[]
 for i in range(n):
  state,_=env.reset(seed=base+i);rng=np.random.default_rng(base+100000+i);total=0;states=[state.tolist()];actions=[]
  while True:
   p=np.asarray(probabilities(state),dtype=np.float64);p=p/p.sum();action=int(rng.choice(3,p=p))
   ns,r,terminated,truncated,_=env.step(action);total+=r;actions.append(action);states.append(ns.tolist());state=ns
   if terminated or truncated:break
  entry=dict(index=i,reward=total,success=bool(terminated),steps=len(actions),environment_seed=base+i,action_rng_seed=base+100000+i)
  if trajectories:entry.update(trajectory=states,actions=actions)
  episodes.append(entry)
 env.close();rewards=[e['reward'] for e in episodes]
 return dict(mean=float(np.mean(rewards)),std=float(np.std(rewards)),success=float(np.mean([e['success'] for e in episodes])),returns=rewards,episodes=episodes,split=split,policy='categorical')

def record_video(policy,episode,path):
 import subprocess
 from PIL import Image,ImageDraw,ImageFont
 env=make_env('rgb_array');state,_=env.reset(seed=episode['environment_seed']);rng=np.random.default_rng(episode['action_rng_seed'])
 fontpath=str(ROOT/'assets/fonts/Pretendard-Bold.ttf');font=ImageFont.truetype(fontpath,24);small=ImageFont.truetype(fontpath,20)
 total=0;done=False;frames=0
 def frame(step):
  canvas=Image.new('RGB',(960,720),'white');draw=ImageDraw.Draw(canvas)
  draw.rectangle((0,0,960,49),fill='#1C50B3');draw.text((20,10),'OpenAI Gym MountainCar | Online 1-step Actor-Critic',font=font,fill='white')
  native=Image.fromarray(env.render());canvas.paste(native.resize((900,600)),(30,50))
  draw.text((25,655),f'Step {step}/200 | Return {total:.3f} | Goal {done}',font=font,fill='black')
  draw.text((25,689),f'Position {state[0]:.3f} | Velocity {state[1]:.3f} | Categorical policy sampling',font=small,fill='black')
  return canvas
 proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','960x720','-r','30','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-crf','20',str(path)],stdin=subprocess.PIPE)
 first=frame(0);first.save(path.parent/'first_frame.png')
 for _ in range(15):proc.stdin.write(first.tobytes());frames+=1
 for t,expected_action in enumerate(episode['actions'],1):
  p=policy.probabilities(state);action=int(rng.choice(3,p=p/p.sum()));assert action==expected_action
  state,r,terminated,truncated,_=env.step(action);total+=r;done=bool(terminated)
  assert np.allclose(state,episode['trajectory'][t],rtol=0,atol=1e-7)
  im=frame(t);proc.stdin.write(im.tobytes());frames+=1
 assert total==episode['reward'] and done==episode['success']
 for _ in range(30):proc.stdin.write(im.tobytes());frames+=1
 proc.stdin.close();assert proc.wait()==0;im.save(path.parent/'result_poster.png');env.close()
 return dict(renderer='unmodified OpenAI Gym MountainCarEnv.render',environment='MountainCar-v0',policy='categorical',episode_index=0,frames=frames,fps=30,duration=frames/30,return_=total,success=done)
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--video',action='store_true');a=p.parse_args()
 s=json.loads((ROOT/'results/summary.json').read_text());policy=SavedPolicy(ROOT/f"results/{s['winner_key']}/actor.npz")
 result=evaluate_policy(policy.probabilities,100,'test',True)
 assert result['returns']==s['test']['returns']
 (ROOT/'results/independent_evaluation.json').write_text(json.dumps(result,indent=2))
 if a.video:
  videos=ROOT/'videos';videos.mkdir(exist_ok=True);metadata=record_video(policy,result['episodes'][0],videos/'result.mp4');(videos/'metadata.json').write_text(json.dumps(metadata,indent=2));print(metadata)
 print({k:result[k] for k in ['mean','std','success']})
if __name__=='__main__':main()
