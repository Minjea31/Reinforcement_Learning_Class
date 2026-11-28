"""Four training stages with one observation schema and one incremental policy."""
from dataclasses import asdict, dataclass
import numpy as np
import mujoco
from simulation.dual_reacher import DualReacherEnv
from simulation.reward_design import STUDY_REWARD_VERSIONS

STAGE_NAMES = {1:'grasp_and_lift',2:'pick_and_place',3:'random_scenarios',4:'minimum_effort'}
LEVELS = (0., .25, .5, .75, 1.)


@dataclass
class Curriculum:
    stage: int = 1
    level: int = 0
    passes: int = 0
    level_start_steps: int = 0
    completed: bool = False
    stage_complete: bool = False
    task_version: int = 1
    pickup_checkpoint: str | None = None
    physics_backend: str = 'cpu'
    layout_mode: str = 'fixed'
    payload_mass: float = 1.

    def env_config(self):
        progress = LEVELS[self.level]
        config=dict(stage=self.stage,difficulty=progress if self.stage < 4 else 1.,
                    effort_scale=progress if self.stage == 4 else 0.,payload_mass=self.payload_mass)
        if self.task_version==2:
            config.update(task_version=2,pickup_checkpoint=self.pickup_checkpoint)
        if self.physics_backend!='cpu':config['physics_backend']=self.physics_backend
        if self.layout_mode!='fixed':config['layout_mode']=self.layout_mode
        return config

    def observe(self, success_rate, steps, threshold=.8, required=2, min_steps=100_000, locked_stage=None):
        if self.completed or (self.stage_complete and locked_stage == self.stage):
            return False
        self.passes = self.passes+1 if success_rate >= threshold else 0
        if self.passes < required or steps-self.level_start_steps < min_steps:
            return False
        if locked_stage == self.stage and self.level == len(LEVELS)-1:
            self.stage_complete = True
            if self.stage == 4:
                self.completed = True
            return False
        if self.stage == 4 and self.level == len(LEVELS)-1:
            self.completed = True
            return False
        if self.level < len(LEVELS)-1:
            self.level += 1
        else:
            self.stage += 1
            self.level = 0
        self.passes = 0
        self.level_start_steps = steps
        return True

    def state_dict(self):
        return asdict(self)


class CurriculumReacherEnv(DualReacherEnv):
    """Use the configured payload mass throughout all learning stages.

    All 14 outputs are target increments. There is no IK or scripted controller
    during step(). IK prepares an ungrasped initial pose only during reset().
    """
    def __init__(self, *, stage=1, difficulty=0., effort_scale=0., max_steps=2000, **kwargs):
        self.stage = int(stage)
        self.difficulty = float(difficulty)
        self.effort_scale = float(effort_scale)
        self.requested_max_steps = max_steps
        self.random_episode = False
        self.initial_side = 0
        self._validate_config()
        super().__init__(max_steps=min(max_steps,self.stage_limit),**kwargs)

    @property
    def stage_limit(self):
        return {1:600,2:1200,3:2000,4:2000}[self.stage]

    def _validate_config(self):
        if self.stage not in STAGE_NAMES or not 0 <= self.difficulty <= 1 or not 0 <= self.effort_scale <= 1:
            raise ValueError('Invalid curriculum stage, difficulty or effort scale')
        if self.stage < 4 and self.effort_scale != 0:
            raise ValueError('Force and second-arm costs are enabled only in stage 4')

    def configure(self, config):
        if 'payload_mass' in config:self.set_mass(config['payload_mass'])
        self.stage = int(config['stage'])
        self.difficulty = float(config['difficulty'])
        self.effort_scale = float(config['effort_scale'])
        self._validate_config()
        self.max_steps = min(self.requested_max_steps,self.stage_limit)
        return self.reset()[0]

    def _sample_random_layout(self, randomize, options):
        self.random_episode = self.stage == 4 or (self.stage == 3 and self.np_random.random() < self.difficulty)
        if self.random_episode:
            start,goal,obstacle,size = super()._sample_random_layout(True,options)
            # Random episodes start from the normal waiting poses, without assigned roles.
            if self.stage == 3 and self.np_random.random() >= self.difficulty:
                obstacle = np.array([3.,0.,.7])
            return start,goal,obstacle,size
        # Two mirrored, otherwise fixed teaching layouts expose both arms to pickup.
        self.initial_side = int(self.np_random.integers(2))
        sign = -1. if self.initial_side == 0 else 1.
        start = np.array([sign*.30,-.25,.625])
        separation = .22 if self.stage == 1 else .12+.13*self.difficulty
        goal = start + [0.,separation,0.]
        return start,goal,np.array([3.,0.,.7]),np.array([.06,.035,.05])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed,options=options)
        if not self.random_episode:
            # Open fingers start near the object; there is no object attachment.
            height = .005+.075*self.difficulty if self.stage == 1 else .04+.08*self.difficulty
            targets = self.home_targets.copy()
            targets[self.initial_side] = self.start+[0.,0.,height]
            action = self.solve_ik(targets,1.,iterations=180)
            self.apply_pose(action)
            opening = .031+.014*self.difficulty if self.stage == 1 else .035+.010*self.difficulty
            for q in self.finger_joints:
                self.data.qpos[q] = opening
            action[[6,13]] = opening/.0225-1.
            self.target_action = action.copy()
            mujoco.mj_forward(self.model,self.data)
        self.previous_potential = self._potential(self.info())
        return self._observation(),self.info()

    def _incremental_targets(self, action):
        action = np.asarray(action,dtype=float)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError('Action must contain 14 finite increments')
        increments = np.clip(action,-1,1)*np.array([.012]*6+[.15]+[.012]*6+[.15])
        targets = np.clip(self.target_action+increments,-1,1)
        for start in (0,3,7,10):
            length = np.linalg.norm(targets[start:start+3])
            if length > 1:
                targets[start:start+3] /= length
        return targets

    def step(self, action):
        return super().step(self._incremental_targets(action))

    def _goal_reached(self, info):
        if self.stage == 1:
            return self.carried and not info['table_support'] and any(info['grasped'])
        return super()._goal_reached(info)

    def _success_duration(self):
        return .5 if self.stage == 1 else 1.

    def _cost_scales(self):
        return dict(effort=self.effort_scale,second_arm=self.effort_scale,time=10.)

    def _potential(self, info):
        distance = np.min(np.linalg.norm(self.data.site_xpos[self.sites]-info['payload_position'],axis=1))
        approach = 10.*(1.-np.tanh(5.*distance))
        grasp = 5.*min(1.,float(self.grasp_seconds.max())/.1)
        clearance = info['payload_position'][2]-info['vertical_half_extent']-.6
        lift = 15.*np.clip(clearance/.12,0.,1.)*min(1.,float(self.grasp_seconds.max())/.1)
        transport = 0. if self.stage == 1 or not self.carried else 20.*(1.-min(1.,info['goal_distance']/1.4))
        settle = 4.*min(1.,self.success_steps*self.dt/self._success_duration())
        return float(approach+grasp+lift+transport+settle)

    def _observation(self):
        base = super()._observation()
        payload = self.data.xpos[self.payload_id]
        relative = np.r_[(payload-self.data.site_xpos[self.sites]).ravel(),self.goal-payload,
                         payload[2]-.625]
        context = np.r_[np.eye(4)[self.stage-1],self.difficulty,self.effort_scale]
        return np.r_[base,relative,context].astype(np.float32)

    def info(self):
        info = super().info()
        info.update(stage=self.stage,stage_name=STAGE_NAMES[self.stage],difficulty=self.difficulty,
                    effort_scale=self.effort_scale,control_mode='incremental_targets')
        return info

    def get_state(self):
        state = super().get_state()
        state['curriculum'] = dict(stage=self.stage,difficulty=self.difficulty,effort_scale=self.effort_scale,
                                   random_episode=self.random_episode,initial_side=self.initial_side,
                                   requested_max_steps=self.requested_max_steps)
        return state

    def set_state(self, state):
        for name,value in state['curriculum'].items():
            setattr(self,name,value)
        self.max_steps = min(self.requested_max_steps,self.stage_limit)
        return super().set_state(state)


def make_env(env_config=None, **kwargs):
    config=dict(env_config) if env_config is not None else None
    backend=config.pop('physics_backend','cpu') if config is not None else kwargs.pop('physics_backend','cpu')
    if backend not in ('cpu','warp'):raise ValueError('Unsupported physics backend')
    if config is not None and config.get('reward_scheme') in STUDY_REWARD_VERSIONS:
        from simulation.study_curriculum import StudyCurriculumEnv
        env=StudyCurriculumEnv(**config,**kwargs)
    elif config is not None and config.get('task_version',1)==2:
        from simulation.placement_curriculum import PlacementCurriculumEnv
        env=PlacementCurriculumEnv(**config,**kwargs)
    else:env=CurriculumReacherEnv(**config,**kwargs) if config is not None else DualReacherEnv(**kwargs)
    if backend=='warp':
        from simulation.warp_physics import WarpPhysics
        try:env.physics=WarpPhysics(env)
        except Exception:
            env.close()
            raise
    return env
