"""Common episode/update telemetry for all three independent algorithms."""
import csv
import math
import time
import os

from stable_baselines3.common.callbacks import BaseCallback
from training.study_budget import stage_progress


FIELDS=['steps','updates','fps','episodes','training_success_rate','mean_return','mean_effort','mean_used_arms',
        'stage','level','difficulty','effort_scale','grasp_rate','lift_rate',
        'policy_loss','value_loss','entropy','kl','actor_loss','critic_loss','temperature',
        'action_saturation','action_std']


class StudyMonitor(BaseCallback):
    def __init__(self,journal,raw,live,config,stopping):
        super().__init__();self.journal=journal;self.raw=raw;self.live=live;self.config=config;self.stopping=stopping
        self.pending=[];self.started=time.monotonic();self.initial_steps=None;self.last_log=None
        self.action_near_limit=0;self.action_values=0
        path=journal.run/'metrics.csv'
        if path.exists() and path.stat().st_size:
            with path.open(newline='') as old:
                reader=csv.DictReader(old)
                if reader.fieldnames!=FIELDS:
                    temporary=path.with_suffix('.partial.csv')
                    with temporary.open('w',newline='') as updated:
                        writer=csv.DictWriter(updated,fieldnames=FIELDS,extrasaction='ignore');writer.writeheader();writer.writerows(reader)
                    os.replace(temporary,path)
        self.stream=path.open('a',newline='');self.writer=csv.DictWriter(self.stream,fieldnames=FIELDS)
        if self.stream.tell()==0:self.writer.writeheader();self.stream.flush()

    def _on_training_start(self):
        if self.initial_steps is None:self.initial_steps=self.model.num_timesteps

    def metrics(self,info):
        env=self.raw.unwrapped;c=self.config
        breakdown=info.get('reward_breakdown',info.get('reward_terms',{}))
        return dict(phase='training',algorithm=c['algorithm'],study_id=c['study_id'],run_name=c['web_run_name'],
            task_version=2,physics_backend='cpu',num_envs=1,layout_mode=c['layout_mode'],payload_mass=c['payload_mass'],
            steps=self.num_timesteps,total_steps=c['total_steps'],
            updates=getattr(self.model,'_n_updates',0),stage=c['stage'],level=c['level'],
            fps=int((self.num_timesteps-(self.initial_steps or 0))/max(.001,time.monotonic()-self.started)),
            episode_step=info.get('episode_step',0),episode_return=float(info.get('episode_return',0.)),
            reward_delta=float(sum(breakdown.values())) if breakdown else None,
            reward_breakdown={key:float(value) for key,value in breakdown.items() if value},
            grasped=list(info.get('grasped',(False,False))),lifted=bool(info.get('episode_lifted',False)),success=bool(info.get('is_success',False)),
            max_torque=float(abs(info['motor_torque']).max()),max_gripper_force=float(abs(info['gripper_force']).max()),
            **stage_progress(c,self.num_timesteps))

    def log(self,force=False):
        if self.num_timesteps==self.last_log and not force:return
        self.last_log=self.num_timesteps;episodes=self.pending;self.pending=[];c=self.config
        def average(key,fn=lambda x:x):
            return sum(float(fn(e[key])) for e in episodes)/len(episodes) if episodes else ''
        values=self.model.logger.name_to_value
        row=dict(steps=self.num_timesteps,updates=getattr(self.model,'_n_updates',0),
            fps=int((self.num_timesteps-(self.initial_steps or 0))/max(.001,time.monotonic()-self.started)),episodes=len(episodes),
            training_success_rate=average('is_success'),mean_return=average('episode_return'),mean_effort=average('episode_effort'),
            mean_used_arms=average('used_arms',lambda a:a.sum()),stage=c['stage'],level=c['level'],difficulty=c['level']/4.,
            effort_scale=c['level']/4. if c['stage']==4 else 0.,grasp_rate=average('used_arms',lambda a:a.any()),lift_rate=average('episode_lifted'))
        keys=dict(policy_loss='train/policy_gradient_loss',value_loss='train/value_loss',entropy='train/entropy_loss',
                  kl='train/approx_kl',actor_loss='train/actor_loss',critic_loss='train/critic_loss',temperature='train/ent_coef')
        for target,source in keys.items():
            value=values.get(source)
            row[target]=float(value) if value is not None and math.isfinite(float(value)) else ''
            if target=='entropy' and row[target]!='':row[target]=-row[target]
        row['action_saturation']=self.action_near_limit/self.action_values if self.action_values else ''
        log_std=getattr(self.model.policy,'log_std',None)
        row['action_std']=float(log_std.detach().exp().mean().cpu()) if log_std is not None else ''
        self.action_near_limit=0;self.action_values=0
        self.writer.writerow(row);self.stream.flush();self.journal.update(row)

    def _on_step(self):
        info=self.locals['infos'][0]
        actions=self.locals['actions']
        self.action_near_limit+=int((abs(actions)>=.98).sum());self.action_values+=actions.size
        if self.locals['dones'][0]:
            self.journal.episode(info,self.num_timesteps,0,self.config['stage'],self.config['level'],not info.get('TimeLimit.truncated',False))
            self.pending.append(info)
        if self.num_timesteps % 256==0:self.log()
        if self.live.ready():
            # VecEnv resets before invoking callbacks. On terminal transitions
            # send the new physical state with its own counters, not stale info.
            current=self.raw.unwrapped.info() if self.locals['dones'][0] else info
            if self.locals['dones'][0]:current=dict(current,episode_return=0.,episode_step=0)
            self.live.send(self.raw.unwrapped.get_state(),self.metrics(current))
        return not self.stopping()

    def close(self):
        if getattr(self,'model',None) is not None:self.log()
        self.stream.close()
