"""Reviewed reward and near-to-far stage-one initialization for new studies."""
from dataclasses import replace
from pathlib import Path
import mujoco
import numpy as np

from simulation.arm_initialization import _home_action, randomize_arms
from simulation.dual_reacher import MAX_ROT, quat_rotvec
from simulation.layout_sampling import solo_layout
from simulation.placement_curriculum import PlacementCurriculumEnv
from simulation.reward_design import (PREGRASP_REWARD_SCHEMES, STUDY_REWARD_VERSIONS,
                                      RewardState, compute_reward, lift_condition,
                                      placement_condition)


class StudyCurriculumEnv(PlacementCurriculumEnv):
    def __init__(self,layout_mode='progressive',reward_scheme='reviewed',arm_mask=None,**kwargs):
        if layout_mode!='progressive' or reward_scheme not in STUDY_REWARD_VERSIONS:
            raise ValueError('Unsupported study curriculum')
        if arm_mask not in (None, 'single', 'mixed', 'both'):
            raise ValueError('Invalid arm-mask mode')
        if reward_scheme in PREGRASP_REWARD_SCHEMES and arm_mask is not None:
            raise ValueError('Grasp progress uses both arms throughout the curriculum')
        self.reward_scheme = reward_scheme
        self.reward_version = STUDY_REWARD_VERSIONS[reward_scheme]
        self.arm_mask = arm_mask
        self._mask_reset_side = None
        self._mask_single = arm_mask == 'single'
        self._study_initializing=True
        self.lift_hold=0.;self.placement_hold=0.;self.qualified=False;self.ever_lifted=False
        super().__init__(layout_mode='random_all',**kwargs)
        if reward_scheme in PREGRASP_REWARD_SCHEMES:
            pads = np.array([[self.id(mujoco.mjtObj.mjOBJ_GEOM, f'{arm}_pad_{pad}')
                              for pad in ('a', 'b')] for arm in ('left', 'right')])
            finger_joints = [self.id(mujoco.mjtObj.mjOBJ_JOINT, f'{arm}_finger_{pad}_joint')
                             for arm in ('left', 'right') for pad in ('a', 'b')]
            pad_centers = self.model.body_pos[self.model.geom_bodyid[pads], 1]
            self._closed_gap = np.abs(pad_centers[:, 0] - pad_centers[:, 1]) - self.model.geom_size[pads, 1].sum(axis=1)
            self._open_gap = self._closed_gap + self.model.jnt_range[finger_joints, 1].reshape(2, 2).sum(axis=1)
        self._study_initializing=False
        self.reset()

    def _sample_random_layout(self, randomize, options):
        if self.arm_mask and self.stage == 1:
            if options:
                raise ValueError('Masked comparison uses seeded layouts')
            return solo_layout(self, arm=self._mask_reset_side)
        return super()._sample_random_layout(randomize, options)

    def _load_pickup(self):
        if self._pickup is None and self.pickup_checkpoint and Path(self.pickup_checkpoint).is_dir():
            from training.study_pickup import FrozenStudyPickup
            self._pickup = FrozenStudyPickup(self.pickup_checkpoint)
        return super()._load_pickup()

    def _initial_pose(self):
        arm=self.initial_side;level=round(self.difficulty*4)
        ranges=[(.05,.10),(.10,.20),(.20,.35),(.35,.50),(.50,.70)]
        lower,upper=ranges[level]
        if level==4:
            for _ in range(24):
                randomize_arms(self)
                distance=np.linalg.norm(self.data.site_xpos[self.sites[arm]]-self.data.xpos[self.payload_id])
                if lower<=distance<=upper:return True
            return False
        baseline=_home_action(self)
        self.apply_pose(baseline)
        targets=self.data.site_xpos[self.sites].copy()
        canonical=np.ones(14)
        for i in range(4):
            off=(i//2)*7+(i%2)*3
            canonical[off:off+3]=np.array([0.,.3 if i%2==0 else -.55,-.7 if i%2==0 else 1.3])/MAX_ROT
        for _ in range(32):
            direction=np.r_[self.np_random.uniform(-.18*level,.18*level,2),1.]
            direction/=np.linalg.norm(direction)
            targets[arm]=self.data.xpos[self.payload_id]+self.np_random.uniform(lower,upper)*direction
            action=self.solve_ik(targets,1.,iterations=110,seed_action=canonical)
            if any(np.linalg.norm(quat_rotvec(self.ik_data.qpos[q:q+4]))>MAX_ROT+1e-8 for q,_ in self.joints[arm*2:arm*2+2]):continue
            idle=1-arm;action[idle*7:idle*7+6]=baseline[idle*7:idle*7+6]
            action[[6,13]]=1.;self.apply_pose(action)
            for q in self.finger_joints:self.data.qpos[q]=.045
            self.target_action=action.copy();self._control(action);mujoco.mj_forward(self.model,self.data)
            if np.linalg.norm(self.data.site_xpos[self.sites[arm]]-targets[arm])>.007:continue
            if any(c.dist<-.0001 and any((mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) or '').startswith(('left_','right_')) for g in (c.geom1,c.geom2)) for c in self.data.contact[:self.data.ncon]):continue
            return True
        return False

    def reset(self,*,seed=None,options=None):
        if self._study_initializing:return super().reset(seed=seed,options=options)
        if self.arm_mask:
            # Consecutive seeds alternate sides. Both-enabled evaluation reuses
            # exactly the same physical starting pose as single-arm evaluation.
            self._mask_reset_side = int(seed) % 2 if seed is not None else None
            self._mask_single = self.arm_mask == 'single' or (
                self.arm_mask == 'mixed' and seed is not None and int(seed) % 4 < 2)
        for attempt in range(32):
            super().reset(seed=seed if attempt==0 else None,options=options)
            if self.stage in (3,4):randomize_arms(self)
            if self.stage!=1 or self._initial_pose():break
        else:raise RuntimeError('Could not sample a valid progressive starting pose')
        self.lift_hold=0.;self.placement_hold=0.;self.qualified=bool(self.stage==2 and self.carried);self.ever_lifted=self.qualified
        self.initial_goal_distance=float(np.linalg.norm(self.data.xpos[self.payload_id,:2]-self.goal[:2]))
        return self._observation(),self.info()

    def _observation(self):
        obs=super()._observation()
        # Keep the 105-value schema, but expose the SAME clocks and carry flag
        # that determine this task's reward. Legacy clocks use different tests.
        obs[82]=min(self.lift_hold,.5)/.5
        obs[87]=float(self.qualified)
        obs[88]=self.lift_hold if self.stage==1 else self.placement_hold
        # Transport shaping depends on the episode's initial goal distance.
        # Replace redundant payload height (already present in qpos) so that
        # this reward-relevant episode variable is observed without adding dims.
        obs[98]=getattr(self,'initial_goal_distance',float(np.linalg.norm(self.data.xpos[self.payload_id,:2]-self.goal[:2])))
        if self.arm_mask:
            available = np.ones(2, dtype=np.float32)
            if self._mask_single:
                available[1 - self.initial_side] = 0.
            obs = np.concatenate((obs, available))
        return obs

    def configure(self,config):
        values=dict(config)
        if values.pop('reward_scheme',self.reward_scheme) != self.reward_scheme:
            raise ValueError('Cannot change the reward scheme during a study')
        arm_mask = values.pop('arm_mask', self.arm_mask)
        if self.reward_scheme in PREGRASP_REWARD_SCHEMES and arm_mask is not None:
            raise ValueError('Grasp progress uses both arms throughout the curriculum')
        self.arm_mask = arm_mask
        if values.get('layout_mode')=='progressive':values['layout_mode']='random_all'
        return super().configure(values)

    def _pregrasp_progress(self, distances, alignments, pad_forces):
        width = 2 * self.model.geom_size[self.payload_geom, 1]
        gaps = self._closed_gap + self.data.qpos[self.finger_joints].reshape(2, 2).sum(axis=1)
        closing = np.clip((self._open_gap-gaps)/np.maximum(self._open_gap-width, .001), 0., 1.)
        alignment_floor = np.cos(np.pi/4)
        aligned = np.clip((alignments-alignment_floor)/(1.-alignment_floor), 0., 1.)
        close_to_target = np.clip(1.-distances/.04, 0., 1.)
        approach = 4.*np.clip(1.-distances/.12, 0., 1.)**2
        paired_contact = np.clip(np.min(pad_forces, axis=1)/.5, 0., 1.)
        return tuple(approach + 3.*close_to_target*aligned*closing + 2.*paired_contact)

    def reward_state(self,info):
        rotation=self.data.xmat[self.payload_id].reshape(3,3)
        points=np.array([[-.055,0,0],[0,0,0],[.055,0,0]])@rotation.T+info['payload_position']
        hands=self.data.site_xpos[self.sites]
        distances=np.linalg.norm(hands[:,None,:]-points[None,:,:],axis=2).min(axis=1)
        axes=[self.data.site_xmat[s].reshape(3,3)[:,1] for s in self.sites]
        alignments=np.clip([abs(float(np.dot(axis,rotation[:,1]))) for axis in axes],0.,1.)
        pregrasp = (self._pregrasp_progress(distances, alignments, info['pad_normal_forces'])
                    if self.reward_scheme in PREGRASP_REWARD_SCHEMES and self.stage == 1 else (0., 0.))
        return RewardState(tuple(distances),closing_alignments=tuple(alignments),pregrasp_progress=pregrasp,
            grasped=tuple(info['grasped']),grasp_seconds=tuple(self.grasp_seconds),used_arms=tuple(bool(v) for v in self.used_arms),
            clearance=float(info['payload_position'][2]-info['vertical_half_extent']-.6),lift_hold_seconds=self.lift_hold,
            carry_qualified=self.qualified,goal_distance=info['goal_distance'],initial_goal_distance=getattr(self,'initial_goal_distance',1.),
            inside_goal=bool(info['inside_goal']),table_support=bool(info['table_support']),robot_contacts=info['payload_robot_contacts'],
            linear_speed=float(np.linalg.norm(self.data.qvel[self.payload_dof:self.payload_dof+3])),
            angular_speed=float(np.linalg.norm(self.data.qvel[self.payload_dof+3:self.payload_dof+6])),placement_hold_seconds=self.placement_hold,
            obstacle_contacts=info['obstacle_contacts'],arm_contacts=info['arm_contacts'],torque_integral=info.get('torque_integral',0.),
            gripper_integral=info.get('grip_integral',info.get('gripper_integral',0.)),fallen=bool(info['payload_position'][2]<.48))

    def step(self,action):
        # Advance physics once, then calculate only this version's progress and
        # reward. Calling the legacy step also updated incompatible carry/goal
        # histories before our own task had assessed this transition.
        before=self.reward_state(super().info())
        if self._mask_single:
            action = np.asarray(action, dtype=float).copy()
            if action.shape != (14,) or not np.isfinite(action).all():
                raise ValueError('Action must contain 14 finite increments')
            idle = 1 - self.initial_side
            action[7*idle:7*(idle+1)] = 0.
        self.target_action=self._incremental_targets(action)
        torque_integral,gripper_integral=self._advance_physics()
        self.steps+=1
        info=super().info()
        self.collision_total+=info['obstacle_contacts']+info['arm_contacts']
        self.grasp_seconds=np.where(info['grasped'],np.minimum(.1,self.grasp_seconds+self.dt),0.)
        self.used_arms|=self.grasp_seconds>=.1-1e-9
        info.update(torque_integral=torque_integral,gripper_integral=gripper_integral)
        after=self.reward_state(info)
        self.lift_hold=self.lift_hold+self.dt if lift_condition(after) else 0.
        if self.lift_hold>=.2-1e-9:self.qualified=True;self.ever_lifted=True
        if info['table_support'] and not info['inside_goal']:
            self.qualified=False
            # Reacquiring carry eligibility requires a new continuous lift.
            self.lift_hold=0.
        after=replace(after,lift_hold_seconds=self.lift_hold,carry_qualified=self.qualified)
        self.placement_hold=self.placement_hold+self.dt if placement_condition(after) else 0.
        after=replace(after,placement_hold_seconds=self.placement_hold)
        result=compute_reward(before,after,stage=self.stage,gamma=self.reward_gamma,dt=self.dt,
                              effort_scale=self.effort_scale,reward_scheme=self.reward_scheme)
        self.episode_return+=result.total
        self.episode_effort+=torque_integral+.25*gripper_integral
        # Compatibility fields for state snapshots and older viewers agree
        # with the active task; none are used to run a second reward system.
        self.carried=self.qualified;self.episode_lifted=self.ever_lifted
        self.lift_seconds=self.lift_hold
        hold=self.lift_hold if self.stage==1 else self.placement_hold
        self.success_steps=round(hold/self.dt)
        if self.stage>=2:
            progress=float(np.clip(1.-after.goal_distance/max(.25,self.initial_goal_distance),0.,1.)) if self.qualified else 0.
            self.episode_goal_progress=max(self.episode_goal_progress,progress)
            self.episode_goal_reached|=bool(self.qualified and after.inside_goal)
            self.episode_released_in_goal|=bool(self.qualified and after.inside_goal and after.table_support and after.robot_contacts==0 and not any(after.grasped))
            self.episode_placed|=bool(placement_condition(after))
            self.episode_premature_releases+=int(result.terms['premature_release']<0.)
        info.update(is_success=result.success,fallen=after.fallen,episode_return=self.episode_return,reward_terms=result.terms,
                    reward_breakdown=result.breakdown,
                    episode_effort=self.episode_effort,episode_step=self.steps,used_arms=self.used_arms.copy(),
                    premature_release=result.terms['premature_release']<0.,
                    carried=self.qualified,episode_lifted=self.ever_lifted,success_hold_seconds=hold,
                    collision_total=self.collision_total,placed=placement_condition(after),
                    episode_goal_reached=self.episode_goal_reached,episode_released_in_goal=self.episode_released_in_goal,
                    episode_placed=self.episode_placed,episode_premature_releases=self.episode_premature_releases,
                    episode_goal_progress=self.episode_goal_progress,
                    approach_distance=float(min(after.hand_distances)),reward_scheme=self.reward_scheme,
                    reward_version=self.reward_version,lift_seconds=self.lift_hold,layout_mode='progressive')
        # Build observations AFTER updating clocks, including terminal states
        # that VecEnv uses to bootstrap time-limit truncations.
        return self._observation(),result.total,result.terminated,self.steps>=self.max_steps and not result.terminated,info

    def info(self):
        info=super().info()
        if not self._study_initializing:
            info.update(carried=self.qualified,episode_lifted=self.ever_lifted,layout_mode='progressive',reward_scheme=self.reward_scheme,
                        reward_version=self.reward_version,lift_seconds=self.lift_hold,
                        success_hold_seconds=self.lift_hold if self.stage==1 else self.placement_hold,
                        placed=placement_condition(self.reward_state(info)))
        return info

    def get_state(self):
        state=super().get_state()
        state['study_reward']={name:getattr(self,name) for name in ('lift_hold','placement_hold','qualified','ever_lifted','initial_goal_distance')}
        if self.arm_mask:
            state['study_reward']['_mask_single'] = self._mask_single
        return state

    def set_state(self,state):
        for name,value in state.get('study_reward',{}).items():setattr(self,name,value)
        return super().set_state(state)
