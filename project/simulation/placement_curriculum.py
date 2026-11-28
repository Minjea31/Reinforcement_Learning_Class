"""Version 2: continue placement from a real, learned stage-one pickup."""
import copy
import hashlib
import io
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from simulation.curriculum import CurriculumReacherEnv
from simulation.dual_reacher import DualReacherEnv
from simulation.layout_sampling import full_layout, solo_layout, solo_goal
from simulation.arm_initialization import randomize_arms


class _PickupPreparationEnv(CurriculumReacherEnv):
    def _goal_reached(self,info):
        # Keep the physical reset simulator running while the learned target settles.
        return False


class _RandomPickupPreparationEnv(_PickupPreparationEnv):
    def _sample_random_layout(self,randomize,options):
        start,goal,obstacle,size=super()._sample_random_layout(randomize,options)
        offset=np.r_[self.np_random.uniform([-.12,-.15],[.12,.15]),0.]
        return start+offset,goal+offset,obstacle,size


class PickupPolicy:
    def __init__(self,path,random_starts=False,payload_mass=1.,layout_mode='fixed'):
        import torch
        from training.ppo import ActorCritic,RunningNorm
        torch.set_num_threads(1)
        raw=Path(path).read_bytes()
        checkpoint=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
        if (checkpoint.get('env_config') or {}).get('stage')!=1 or checkpoint.get('obs_dim')!=105 or checkpoint.get('control_mode')!='incremental_targets':
            raise ValueError('Pickup continuation requires a stage-one incremental-control checkpoint')
        self.source_hash=hashlib.sha256(raw).hexdigest();self.source_steps=checkpoint['steps']
        self.policy=ActorCritic(105);self.policy.load_state_dict(checkpoint['policy']);self.policy.eval()
        self.norm=RunningNorm((105,));self.norm.load_state_dict(checkpoint['normalizer'])
        env_class=(_RandomApproachPickupPreparationEnv if layout_mode=='random_approach' else
                   _AllRandomPickupPreparationEnv if layout_mode=='random_all' else
                   _RandomPickupPreparationEnv if random_starts else _PickupPreparationEnv)
        self.env=env_class(stage=1,difficulty=checkpoint['env_config']['difficulty'],
            max_steps=600,reward_gamma=checkpoint['config']['gamma'],payload_mass=payload_mass)

    def rollout(self,seed):
        import torch
        obs,_=self.env.reset(seed=seed)
        for _ in range(self.env.max_steps):
            with torch.no_grad():action=self.policy.sample(torch.from_numpy(self.norm(obs)[None]),deterministic=True)[0][0].numpy()
            obs,_,term,trunc,info=self.env.step(action)
            if info['carried'] and any(info['grasped']) and not info['table_support']:
                # Hold the already learned motor targets; do not overwrite positions,
                # velocities, contact forces, or the gripper opening.
                quiet=0
                for _ in range(100):
                    _,_,_,_,info=self.env.step(np.zeros(14))
                    if not info['carried'] or not any(info['grasped']) or info['table_support']:break
                    speed=np.linalg.norm(self.env.data.qvel[self.env.payload_dof:self.env.payload_dof+6])
                    quiet=quiet+1 if speed<.1 else 0
                    if quiet>=3:return self.env.get_state(),self.env.steps
                raise RuntimeError('The learned pickup could not settle while preserving its grip')
            if term or trunc:
                break
        raise RuntimeError('The stage-one policy did not produce a stable carry; no synthetic grip was substituted')

    def close(self):self.env.close()


class PlacementCurriculumEnv(CurriculumReacherEnv):
    """Same 105 observations, 14 actions and physical motor limits as version 1.

    A frozen stage-one policy only prepares the reset state. Every stage-two
    action, including gripping and releasing, is chosen by the trainable policy.
    """
    def __init__(self,*,task_version=2,pickup_checkpoint=None,layout_mode='fixed',**kwargs):
        if task_version!=2:raise ValueError('Unsupported placement task version')
        if layout_mode not in ('fixed','random_separated','random_all','random_approach'):raise ValueError('Unsupported layout mode')
        self.layout_mode=layout_mode
        self.task_version=2
        self.pickup_checkpoint=str(Path(pickup_checkpoint).resolve()) if pickup_checkpoint else None
        self._pickup=None;self._initializing=True
        self._clear_progress()
        super().__init__(**kwargs)
        self._initializing=False
        if self.stage==2 or self.layout_mode=='random_approach':self.reset()

    def _clear_progress(self):
        self.pickup_steps=0;self.pickup_seed=None;self.pickup_source_steps=None;self.pickup_source_hash=None
        self.transport_scale=.1;self.episode_goal_reached=False;self.episode_released_in_goal=False
        self.episode_placed=False;self.episode_premature_releases=0;self.episode_goal_progress=0.

    def _load_pickup(self):
        if self._pickup is None:
            if not self.pickup_checkpoint:raise ValueError('Stage two requires a frozen --pickup-checkpoint')
            self._pickup=PickupPolicy(self.pickup_checkpoint,random_starts=self.layout_mode=='random_separated',payload_mass=self.mass,layout_mode=self.layout_mode)
        return self._pickup

    def configure(self,config):
        if config.get('task_version',2)!=2:raise ValueError('An environment cannot change task versions in place')
        source=config.get('pickup_checkpoint',self.pickup_checkpoint)
        source=str(Path(source).resolve()) if source else None
        layout=config.get('layout_mode',self.layout_mode)
        if layout not in ('fixed','random_separated','random_all','random_approach'):raise ValueError('Unsupported layout mode')
        if source!=self.pickup_checkpoint or layout!=self.layout_mode or config.get('payload_mass',self.mass)!=self.mass:
            if self._pickup:self._pickup.close()
            self._pickup=None;self.pickup_checkpoint=source;self.layout_mode=layout
        return super().configure(config)

    def _sample_random_layout(self,randomize,options):
        if self.layout_mode in ('random_all','random_approach'):
            if options:raise ValueError('All-stage random layouts are sampled by the reset constraints')
            if self.stage in (1,2):return solo_layout(self)
            return full_layout(self,options)
        if self.layout_mode=='random_separated' and self.stage>=2:
            for _ in range(64):
                start,goal,obstacle,size=super()._sample_random_layout(randomize,options)
                if self.stage==2 or not self.random_episode:goal=self._separated_goal(start,obstacle,size)
                if goal is not None and self._markers_separated(start,goal):return start,goal,obstacle,size
            raise RuntimeError('Could not sample separate start and goal regions')
        result=super()._sample_random_layout(randomize,options)
        if self.stage==2:
            start,goal,obstacle,size=result
            goal=start+[0.,np.interp(self.difficulty,[0.,.25,.5,.75,1.],[0.,.06,.12,.18,.25]),0.]
            return start,goal,obstacle,size
        return result

    @staticmethod
    def _markers_separated(start,goal):
        # The yellow and cyan markers have half sizes (0.13m, 0.08m).
        delta=np.abs(np.asarray(goal)[:2]-np.asarray(start)[:2])
        return bool(np.linalg.norm(delta)>=.22 and np.any(delta>np.array([.26,.16])+.025))

    def _separated_goal(self,start,obstacle,size,arm=None):
        if self.layout_mode in ('random_all','random_approach'):return solo_goal(self,start,arm)
        for _ in range(512):
            goal=np.r_[self.np_random.uniform([-.55,-.36],[.55,.36]),.625]
            distance=np.linalg.norm(goal[:2]-start[:2])
            if distance>.38+.12*self.difficulty or not self._markers_separated(start,goal):continue
            if self._clear_table_area(goal,[.13,.08],obstacle,size):return goal
        return None

    def _random_pickup_start(self):
        pickup=self._load_pickup()
        for _ in range(32):
            pickup_seed=int(self.np_random.integers(0,2**31-1))
            try:state,warmup=pickup.rollout(pickup_seed)
            except RuntimeError:continue
            position=pickup.env.data.xpos[pickup.env.payload_id].copy()
            start=np.r_[position[:2],.625]
            obstacle=state['task']['obstacle_position'];size=state['model']['geom_size'][self.obstacle_geom]
            if not self._clear_table_area(start,[.13,.08],obstacle,size):continue
            goal=self._separated_goal(start,obstacle,size,arm=pickup.env.initial_side)
            if goal is not None:return state,warmup,pickup_seed,goal
        raise RuntimeError('Frozen pickup policy could not prepare a valid varied start; no synthetic grip was used')

    def reset(self,*,seed=None,options=None):
        if self._initializing:return super().reset(seed=seed,options=options)
        self._clear_progress()
        if self.stage!=2:
            super().reset(seed=seed,options=options)
            if self.layout_mode=='random_approach':randomize_arms(self)
        else:
            if options:raise ValueError('Stage-two pickup starts are defined by the frozen stage-one policy')
            gym.Env.reset(self,seed=seed)
            if self.layout_mode in ('random_separated','random_all','random_approach'):
                state,warmup,pickup_seed,random_goal=self._random_pickup_start()
                reset_rng=copy.deepcopy(self.np_random.bit_generator.state)
                pickup=self._load_pickup()
            else:
                pickup_seed=seed if seed is not None else int(self.np_random.integers(0,2**31-1))
                reset_rng=copy.deepcopy(self.np_random.bit_generator.state)
                pickup=self._load_pickup();state,warmup=pickup.rollout(pickup_seed)
            DualReacherEnv.set_state(self,state)
            self.np_random.bit_generator.state=reset_rng
            self.initial_side=state['curriculum']['initial_side'];self.random_episode=self.layout_mode in ('random_separated','random_all','random_approach')
            self.pickup_steps=warmup;self.pickup_seed=pickup_seed
            self.pickup_source_steps=pickup.source_steps;self.pickup_source_hash=pickup.source_hash
            separation=float(np.interp(self.difficulty,[0.,.25,.5,.75,1.],[0.,.06,.12,.18,.25]))
            payload=self.data.xpos[self.payload_id].copy()
            self.goal=random_goal if self.random_episode else np.r_[payload[:2]+[0.,separation],.625]
            if self.random_episode:
                self.start=np.r_[payload[:2],.625]
                self.model.site_pos[self.start_site]=[self.start[0],self.start[1],.601]
            if not self._clear_table_area(self.goal,self.model.site_size[self.goal_site,:2],self.obstacle_position,self.model.geom_size[self.obstacle_geom]):
                raise RuntimeError('The learned pickup produced a goal outside the placement workspace')
            self.model.site_pos[self.goal_site]=[self.goal[0],self.goal[1],.601]
            self.data.time=0.;self.steps=0;self.success_steps=0
            self.episode_return=0.;self.episode_effort=0.;self.collision_total=0
            mujoco.mj_forward(self.model,self.data)
        self.transport_scale=max(.04,float(np.linalg.norm(self.data.xpos[self.payload_id,:2]-self.goal[:2])))
        self.previous_potential=self._potential(self.info())
        return self._observation(),self.info()

    def info(self):
        info=super().info()
        distance=float(np.linalg.norm(info['payload_position'][:2]-self.goal[:2]))
        contacts=0
        for index in range(self.data.ncon):
            contact=self.data.contact[index]
            if self.payload_geom not in (contact.geom1,contact.geom2):continue
            other=contact.geom2 if contact.geom1==self.payload_geom else contact.geom1
            name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,other) or ''
            if name.startswith(('left_','right_')):
                force=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,index,force)
                contacts+=int(force[0]>.1)
        speed=float(np.linalg.norm(self.data.qvel[self.payload_dof:self.payload_dof+6]))
        info.update(task_version=2,layout_mode=self.layout_mode,goal_distance_3d=info['goal_distance'],goal_distance=distance,
            carried=self.carried,episode_lifted=self.episode_lifted,
            payload_robot_contacts=contacts,payload_speed=speed,
            placed=bool(distance<.04 and speed<.08 and contacts==0 and info['table_support'] and info['inside_goal']),
            pickup_steps=self.pickup_steps,pickup_source_steps=self.pickup_source_steps,
            pickup_source_hash=self.pickup_source_hash,episode_goal_reached=self.episode_goal_reached,
            episode_released_in_goal=self.episode_released_in_goal,episode_placed=self.episode_placed,
            episode_premature_releases=self.episode_premature_releases,
            episode_goal_progress=self.episode_goal_progress)
        return info

    def _potential(self,info):
        if self._initializing or self.stage==1:return super()._potential(info)
        if not self.carried:
            distance=np.min(np.linalg.norm(self.data.site_xpos[self.sites]-info['payload_position'],axis=1))
            grasp=min(1.,float(self.grasp_seconds.max())/.1)
            clearance=info['payload_position'][2]-info['vertical_half_extent']-.6
            return float(10.*(1.-np.tanh(5.*distance))+5.*grasp+15.*np.clip(clearance/.12,0.,1.)*grasp)
        progress=float(np.clip(1.-info['goal_distance']/self.transport_scale,0.,1.))
        clearance=max(0.,info['payload_position'][2]-info['vertical_half_extent']-.6)
        lowering=float(np.clip(1.-clearance/.12,0.,1.))
        settle=min(1.,self.success_steps*self.dt/self._success_duration())
        # Qualified pickup credit persists through legitimate release in the goal.
        return float(30.+25.*progress+15.*progress*lowering+10.*settle)

    def step(self,action):
        stable_before=bool(np.any(self.grasp_seconds>=.1-1e-9));carried_before=self.carried
        obs,reward,term,trunc,info=super().step(action)
        early_loss=self.stage>=2 and carried_before and stable_before and not any(info['grasped']) and not info['inside_goal']
        penalty=-2. if early_loss else 0.
        reward+=penalty;self.episode_return+=penalty
        self.episode_premature_releases+=int(early_loss)
        progress=float(np.clip(1.-info['goal_distance']/self.transport_scale,0.,1.)) if self.carried else 0.
        self.episode_goal_progress=max(self.episode_goal_progress,progress)
        self.episode_goal_reached|=bool(self.carried and info['inside_goal'])
        self.episode_released_in_goal|=bool(self.carried and info['inside_goal'] and info['payload_robot_contacts']==0)
        self.episode_placed|=bool(self.carried and info['placed'])
        info['reward_terms']['premature_release']=penalty
        info.update(episode_return=self.episode_return,premature_release=early_loss,
            episode_goal_reached=self.episode_goal_reached,episode_released_in_goal=self.episode_released_in_goal,
            episode_placed=self.episode_placed,episode_premature_releases=self.episode_premature_releases,
            episode_goal_progress=self.episode_goal_progress)
        return obs,float(reward),term,trunc,info

    def get_state(self):
        state=super().get_state()
        state['placement_v2']={name:copy.deepcopy(getattr(self,name)) for name in (
            'layout_mode','pickup_checkpoint','pickup_steps','pickup_seed','pickup_source_steps','pickup_source_hash',
            'transport_scale','episode_goal_reached','episode_released_in_goal','episode_placed',
            'episode_premature_releases','episode_goal_progress')}
        return state

    def set_state(self,state):
        if 'placement_v2' not in state:raise ValueError('Cannot restore version-one task states into version two')
        layout=state['placement_v2'].get('layout_mode','fixed')
        if layout!=self.layout_mode and self._pickup:
            self._pickup.close();self._pickup=None
        self.layout_mode=layout
        expected=state['placement_v2'].get('pickup_source_hash')
        if self._pickup and expected and self._pickup.source_hash!=expected:raise ValueError('Pickup policy changed since checkpoint')
        for name,value in state['placement_v2'].items():setattr(self,name,copy.deepcopy(value))
        return super().set_state(state)

    def close(self):
        if self._pickup:self._pickup.close()
        super().close()


class _AllRandomPickupPreparationEnv(PlacementCurriculumEnv):
    def __init__(self, **kwargs):
        super().__init__(layout_mode='random_all', **kwargs)

    def _goal_reached(self, info):
        return False


class _RandomApproachPickupPreparationEnv(PlacementCurriculumEnv):
    def __init__(self, **kwargs):
        super().__init__(layout_mode='random_approach', **kwargs)

    def _goal_reached(self, info):
        return False
