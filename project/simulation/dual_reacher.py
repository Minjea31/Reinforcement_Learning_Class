"""Two 2-ball-joint arms with physical grippers, exposed as a Gymnasium env.

Action order per arm: shoulder rotation vector XYZ, elbow rotation vector XYZ,
gripper opening. Rotation vectors are normalized by MAX_ROT; -1 closes the
gripper and +1 opens it. Low-level joint PD control produces actual motor torque.
No weld, teleportation or artificial object attachment is used during steps.
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "dual_reacher.xml"
MAX_ROT = 2.75


def rotvec_quat(vector):
    angle = np.linalg.norm(vector)
    quat = np.array([1., 0., 0., 0.])
    if angle > 1e-9:
        mujoco.mju_axisAngle2Quat(quat, vector / angle, angle)
    return quat


def quat_rotvec(quat):
    quat = np.array(quat, dtype=float)
    if quat[0] < 0:
        quat *= -1
    result = np.zeros(3)
    mujoco.mju_quat2Vel(result, quat, 1.)
    return result


class DualReacherEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, payload_mass=1.0, max_steps=2000,
                 task='random', randomize=True, reward_gamma=.999):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.renderer = None
        self.physics = None
        self.max_steps = max_steps
        self.reward_gamma = reward_gamma
        if task != 'random':
            raise ValueError(f'Unknown task: {task}')
        self.task_selection = task
        self.randomize = randomize
        self.frame_skip = 10
        self.dt = self.model.opt.timestep * self.frame_skip
        self.joints = []
        self.finger_joints = []
        self.sites = []
        for side in ("left", "right"):
            self.sites.append(self.id(mujoco.mjtObj.mjOBJ_SITE, f"{side}_grasp"))
            for name in ("shoulder", "elbow"):
                jid = self.id(mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{name}")
                self.joints.append((int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid])))
            for name in ("a", "b"):
                jid = self.id(mujoco.mjtObj.mjOBJ_JOINT, f"{side}_finger_{name}_joint")
                self.finger_joints.append(int(self.model.jnt_qposadr[jid]))
        self.ball_dofs = np.array([d+k for _, d in self.joints for k in range(3)])
        self.payload_id = self.id(mujoco.mjtObj.mjOBJ_BODY, "payload")
        self.payload_geom = self.id(mujoco.mjtObj.mjOBJ_GEOM, "payload_geom")
        self.obstacle_geom = self.id(mujoco.mjtObj.mjOBJ_GEOM, "obstacle")
        self.obstacle_body = self.id(mujoco.mjtObj.mjOBJ_BODY, 'obstacle_body')
        self.obstacle_mocap = int(self.model.body_mocapid[self.obstacle_body])
        self.payload_joint = self.id(mujoco.mjtObj.mjOBJ_JOINT, 'payload_free')
        self.payload_qpos = int(self.model.jnt_qposadr[self.payload_joint])
        self.payload_dof = int(self.model.jnt_dofadr[self.payload_joint])
        self.payload_sites = [self.id(mujoco.mjtObj.mjOBJ_SITE, f'payload_{side}') for side in ('left','right')]
        self.goal_site = self.id(mujoco.mjtObj.mjOBJ_SITE, 'goal')
        self.start_site = self.id(mujoco.mjtObj.mjOBJ_SITE, 'start')
        self.goal = np.array([0., .30, .625])
        self.gravity_compensation = True
        self.motor_enabled = True
        self.target_action = np.zeros(14)
        self.action_space = spaces.Box(-1., 1., (14,), dtype=np.float32)
        self.set_mass(payload_mass)
        self.reset()
        self.observation_space = spaces.Box(-np.inf, np.inf, self._observation().shape, dtype=np.float32)

    def id(self, kind, name):
        result = mujoco.mj_name2id(self.model, kind, name)
        if result < 0:
            raise KeyError(name)
        return result

    def set_mass(self, mass):
        if not .05 <= mass <= 5.:
            raise ValueError("Payload mass must be between 0.05 and 5 kg")
        half = self.model.geom_size[self.payload_geom]
        self.model.body_mass[self.payload_id] = mass
        self.model.body_inertia[self.payload_id] = mass/3 * np.array([
            half[1]**2+half[2]**2, half[0]**2+half[2]**2, half[0]**2+half[1]**2])
        # Recompute constant inertial terms without changing current object pose.
        qpos, qvel, when = self.data.qpos.copy(), self.data.qvel.copy(), self.data.time
        mocap_pos, mocap_quat = self.data.mocap_pos.copy(), self.data.mocap_quat.copy()
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos[:], self.data.qvel[:], self.data.time = qpos, qvel, when
        self.data.mocap_pos[:], self.data.mocap_quat[:] = mocap_pos, mocap_quat
        mujoco.mj_forward(self.model, self.data)

    @property
    def mass(self):
        return float(self.model.body_mass[self.payload_id])

    @staticmethod
    def _clear_table_area(point, half_xy, obstacle, obstacle_size):
        """Placement clearance only; does not select a robot or a strategy."""
        point, half_xy = np.asarray(point), np.asarray(half_xy)
        if np.any(np.abs(point[:2])+half_xy > [.95,.62]):
            return False
        if np.all(np.abs(point[:2]-obstacle[:2]) < half_xy+obstacle_size[:2]+.025):
            return False
        for base_x in (-.68,.68):
            separation = np.maximum(np.abs(point[:2]-[base_x,0.])-half_xy,0.)
            if np.linalg.norm(separation) < .105:
                return False
        return True

    def _sample_random_layout(self, randomize, options):
        """Independent table positions including same-side and cross-table pairs.

        Rejection removes intersections, not single-arm or cooperative layouts.
        All accepted positions receive exactly the same role-free objective.
        """
        if randomize:
            size = self.np_random.uniform([.06,.035,.05],[.13,.07,.09])
            obstacle = np.r_[self.np_random.uniform([-.32,-.22],[.32,.22]),.60+size[2]]
        else:
            obstacle, size = np.array([0.,.02,.69]), np.array([.12,.045,.09])
        for _ in range(5000):
            start = np.r_[self.np_random.uniform([-.55,-.36],[.55,.36]),.625] if randomize else np.array([0.,-.28,.625])
            goal = np.r_[self.np_random.uniform([-.55,-.36],[.55,.36]),.625] if randomize else np.array([0.,.28,.625])
            start = np.asarray(options.get('start',start),dtype=float)
            goal = np.asarray(options.get('goal',goal),dtype=float)
            if start.shape != (3,) or goal.shape != (3,) or not np.isfinite(start).all() or not np.isfinite(goal).all():
                raise ValueError('start and goal must be finite XYZ vectors')
            if abs(start[2]-.625) > .001 or abs(goal[2]-.625) > .001:
                raise ValueError('start and goal must be on the tabletop (z=0.625)')
            if not self._clear_table_area(start,[.09,.035],obstacle,size):
                continue
            if not self._clear_table_area(goal,[.13,.08],obstacle,size):
                continue
            if not ('start' in options and 'goal' in options) and np.linalg.norm(start[:2]-goal[:2]) < .24:
                continue
            return start.copy(),goal.copy(),obstacle,size
        raise ValueError('Could not generate non-overlapping tabletop start and goal')

    def reset(self, *, seed=None, options=None):
        if self.physics is not None:self.physics.invalidate()
        super().reset(seed=seed)
        options = options or {}
        selection = options.get('task', self.task_selection)
        if selection != 'random':
            raise ValueError(f'Unknown task: {selection}')
        self.selected_mode = selection
        randomize = options.get('randomize', self.randomize)
        self.grasp_offset, half_length = .055, .09
        self.task = 'random'
        self.start,self.goal,self.obstacle_position,obstacle_size = self._sample_random_layout(randomize,options)
        self.model.geom_size[self.obstacle_geom] = obstacle_size
        self.model.geom_size[self.payload_geom] = [half_length, .035, .025]
        for sid, sign in zip(self.payload_sites, (-1, 1)):
            self.model.site_pos[sid] = [sign*self.grasp_offset, 0., 0.]
        # Explicit scenarios make repeatable experiments possible; reject invalid placements.
        for name in ('start', 'goal'):
            if name in options:
                point = np.asarray(options[name], dtype=float)
                if point.shape != (3,) or not np.isfinite(point).all():
                    raise ValueError(f'{name} must be a finite XYZ vector')
                if abs(point[0])+half_length > .95 or abs(point[1]) > .48 or abs(point[2]-.625) > .001:
                    raise ValueError(f'{name} is outside the supported tabletop workspace')
                obstacle = self.obstacle_position
                size = self.model.geom_size[self.obstacle_geom]
                if abs(point[0]-obstacle[0]) < half_length+size[0]+.02 and abs(point[1]-obstacle[1]) < .035+size[1]+.02:
                    raise ValueError(f'{name} overlaps the obstacle')
                setattr(self, name, point.copy())
        for sid, point in ((self.start_site,self.start),(self.goal_site,self.goal)):
            self.model.site_pos[sid] = [point[0],point[1],.601]
            self.model.site_size[sid] = [half_length+.04,.08,.002]
        self.set_mass(self.mass)
        mujoco.mj_resetData(self.model, self.data)
        self.data.mocap_pos[self.obstacle_mocap] = self.obstacle_position
        self.data.qpos[self.payload_qpos:self.payload_qpos+3] = self.start + [0.,0.,.002]
        for q in self.finger_joints:
            self.data.qpos[q] = .045
        # Bent configurations avoid starting at the fully straight singularity.
        for i, (q, _) in enumerate(self.joints):
            self.data.qpos[q:q+4] = rotvec_quat(np.array([0., .3 if i%2 == 0 else -.55, -.7 if i%2 == 0 else 1.3]))
        mujoco.mj_forward(self.model, self.data)
        self.home_targets = np.array([[-.30,-.34,.94],[.30,-.34,.94]])
        action = self.solve_ik(self.home_targets, 1., iterations=160)
        self.apply_pose(action)
        self.target_action = action.copy()
        self.steps = 0
        self.success_steps = 0
        self.collision_total = 0
        self.lift_seconds = 0.
        self.grasp_seconds = np.zeros(2)
        self.used_arms = np.zeros(2, dtype=bool)
        self.carried = False
        self.episode_lifted = False
        self.episode_effort = 0.
        self.episode_return = 0.
        mujoco.mj_forward(self.model, self.data)
        self.previous_potential = self._potential(self.info())
        return self._observation(), self.info()

    def action_from_data(self, data):
        action = np.ones(14, dtype=float)
        for i, (q, _) in enumerate(self.joints):
            start = (i//2)*7 + (i%2)*3
            action[start:start+3] = quat_rotvec(data.qpos[q:q+4]) / MAX_ROT
        return np.clip(action, -1, 1)

    def apply_pose(self, action):
        """Set the initial arm pose during reset."""
        for i, (q, _) in enumerate(self.joints):
            start = (i//2)*7 + (i%2)*3
            self.data.qpos[q:q+4] = rotvec_quat(action[start:start+3]*MAX_ROT)
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def solve_ik(self, targets, opening=1., iterations=12, seed_action=None):
        """Damped Jacobian IK: grasp position + gripper closing-axis alignment.

        Uses scratch data; it does not teleport the simulation or the payload.
        Used by reset to calculate the initial waiting pose.
        """
        scratch = self.ik_data
        scratch.qpos[:] = self.data.qpos
        if seed_action is not None:
            for i, (q, _) in enumerate(self.joints):
                start = (i//2)*7 + (i%2)*3
                scratch.qpos[q:q+4] = rotvec_quat(seed_action[start:start+3]*MAX_ROT)
        scratch.qvel[:] = 0
        for _ in range(iterations):
            mujoco.mj_forward(self.model, scratch)
            delta = np.zeros(self.model.nv)
            for arm, sid in enumerate(self.sites):
                jacp = np.zeros((3, self.model.nv))
                jacr = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, scratch, jacp, jacr, sid)
                axis = scratch.site_xmat[sid].reshape(3, 3)[:, 1]
                desired_axis = np.array([0., 1. if arm == 0 else -1., 0.])
                projection = np.eye(3) - np.outer(axis, axis)
                columns = self.ball_dofs[arm*6:arm*6+6]
                jac = np.vstack([jacp[:, columns], .18*projection@jacr[:, columns]])
                err = np.r_[targets[arm]-scratch.site_xpos[sid], .18*np.cross(axis, desired_axis)]
                update = jac.T @ np.linalg.solve(jac@jac.T + .0002*np.eye(6), err)
                delta[columns] = np.clip(update, -.12, .12)
            mujoco.mj_integratePos(self.model, scratch.qpos, delta, 1.)
        result = self.action_from_data(scratch)
        result[[6, 13]] = opening
        return result

    def _control(self, action):
        for i, (q, d) in enumerate(self.joints):
            start = (i//2)*7 + (i%2)*3
            vector = action[start:start+3]*MAX_ROT
            length = np.linalg.norm(vector)
            if length > MAX_ROT:
                vector = vector * MAX_ROT/length
            error = np.zeros(3)
            mujoco.mju_subQuat(error, rotvec_quat(vector), self.data.qpos[q:q+4])
            torque = 180*error - 16*self.data.qvel[d:d+3]
            if self.gravity_compensation:
                torque += self.data.qfrc_bias[d:d+3]
            self.data.ctrl[start:start+3] = np.clip(torque, -65, 65) if self.motor_enabled else 0
        for idx in (6, 13):
            self.data.ctrl[idx] = .0225*(1+action[idx])

    def _potential(self, info):
        # Differences, rather than repeated stage bonuses, prevent reward farming.
        distance = np.min(np.linalg.norm(
            self.data.site_xpos[self.sites] - info['payload_position'], axis=1))
        approach = 4. * np.exp(-4. * distance)
        grasp = 4. * min(1., float(self.grasp_seconds.max()) / .1)
        lift = 8. * min(1., self.lift_seconds / .2)
        transport = 20. * (1. - min(1., info['goal_distance'] / 1.4)) if self.carried else 0.
        settle = 4. * min(1., self.success_steps * self.dt) if self.carried else 0.
        return float(approach + grasp + lift + transport + settle)

    def _goal_reached(self, info):
        return info['placed'] and self.carried

    def _success_duration(self):
        return 1.

    def _cost_scales(self):
        return dict(effort=1.,second_arm=1.,time=1.)

    def _advance_physics(self):
        if self.physics is not None:return self.physics.advance(self.target_action)
        torque_integral = grip_integral = 0.
        for _ in range(self.frame_skip):
            self._control(self.target_action)
            mujoco.mj_step(self.model, self.data)
            # Integrate actual motor forces, including gravity compensation and holding.
            torque_integral += float(np.abs(self.data.qfrc_actuator[self.ball_dofs]).mean()/65.) * self.model.opt.timestep
            grip_integral += float(np.abs(self.data.actuator_force[[6, 13]]).mean()/200.) * self.model.opt.timestep
        mujoco.mj_forward(self.model, self.data)
        return torque_integral,grip_integral

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("Action must contain 14 finite numbers")
        self.target_action = np.clip(action, -1, 1)
        torque_integral,grip_integral = self._advance_physics()
        self.steps += 1
        info = self.info()
        self.collision_total += info['obstacle_contacts'] + info['arm_contacts']
        self.grasp_seconds = np.where(info['grasped'], np.minimum(.1, self.grasp_seconds+self.dt), 0.)
        previously_both = bool(self.used_arms.all())
        self.used_arms |= self.grasp_seconds >= .1 - 1e-9
        second_arm = bool(self.used_arms.all() and not previously_both)
        bottom = info['payload_position'][2] - info['vertical_half_extent']
        lifting = bottom >= .63 and bool(np.any(self.grasp_seconds >= .1-1e-9))
        self.lift_seconds = min(.2, self.lift_seconds+self.dt) if lifting else 0.
        if self.lift_seconds >= .2-1e-9:
            self.carried = True
            self.episode_lifted = True
        if info['table_support'] and not info['inside_goal']:
            self.carried = False
        self.success_steps = self.success_steps+1 if self._goal_reached(info) else 0
        success = self.success_steps*self.dt >= self._success_duration()-1e-9
        fallen = bool(info['payload_position'][2] < .48)
        terminated = fallen or success
        truncated = self.steps >= self.max_steps and not terminated
        potential = 0. if terminated else self._potential(info)
        scales = self._cost_scales()
        costs = {
            'shaping': self.reward_gamma*potential-self.previous_potential,
            'success': 100. if success else 0.,
            'fall': -25. if fallen else 0.,
            'second_arm': -10.*scales['second_arm'] if second_arm else 0.,
            'torque': -.1*scales['effort']*torque_integral,
            'gripper': -.025*scales['effort']*grip_integral,
            'collision': -.5*self.dt*(info['obstacle_contacts']+info['arm_contacts']),
            'time': -.02*scales['time']*self.dt,
        }
        self.previous_potential = potential
        reward = float(sum(costs.values()))
        self.episode_effort += torque_integral + .25*grip_integral
        self.episode_return += reward
        info.update(is_success=success, fallen=fallen, episode_step=self.steps,reward_terms=costs,
                    torque_integral=torque_integral, gripper_integral=grip_integral,
                    episode_effort=self.episode_effort, episode_return=self.episode_return,
                    episode_lifted=self.episode_lifted,
                    used_arms=self.used_arms.copy(), carried=self.carried,
                    lift_seconds=self.lift_seconds, collision_total=self.collision_total,
                    success_hold_seconds=self.success_steps*self.dt)
        if self.physics is not None:
            info.update(physics_backend='warp',physics_device=str(self.physics.device),
                        physics_substeps=self.physics.substeps)
        return self._observation(), reward, terminated, truncated, info

    def _observation(self):
        return np.concatenate([self.data.qpos, self.data.qvel, self.goal,
            self.data.site_xpos[self.sites].ravel(), self.data.geom_xpos[self.obstacle_geom],
            self.model.geom_size[self.obstacle_geom], [self.mass],
            self.model.geom_size[self.payload_geom], self.target_action,
            [self.lift_seconds/.2], self.grasp_seconds/.1, self.used_arms.astype(float),
            [float(self.carried), self.success_steps*self.dt]]).astype(np.float32)

    def get_state(self):
        """Complete integration + task state for exact rollout-boundary resume."""
        import copy
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        physics = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, self.data, physics, spec)
        names = ('start', 'goal', 'obstacle_position', 'target_action', 'steps',
                 'success_steps', 'collision_total', 'lift_seconds', 'grasp_seconds',
                 'used_arms', 'carried', 'episode_lifted', 'episode_effort', 'episode_return', 'previous_potential')
        return {'physics': physics, 'rng': copy.deepcopy(self.np_random.bit_generator.state),
                'task': {name: copy.deepcopy(getattr(self, name)) for name in names},
                'model': {name: getattr(self.model, name).copy() for name in
                          ('geom_size', 'site_pos', 'site_size', 'body_mass', 'body_inertia')}}

    def set_state(self, state):
        import copy
        if self.physics is not None:self.physics.invalidate()
        for name, value in state['model'].items():
            getattr(self.model, name)[:] = value
        mujoco.mj_setConst(self.model, self.data)
        for name, value in state['task'].items():
            setattr(self, name, copy.deepcopy(value))
        self.np_random.bit_generator.state = copy.deepcopy(state['rng'])
        mujoco.mj_setState(self.model, self.data, state['physics'], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.model, self.data)
        return self._observation()

    def info(self):
        position = self.data.xpos[self.payload_id].copy()
        obstacle_contacts = 0
        finger_contacts = set()
        pad_contacts = {'left': set(), 'right': set()}
        arm_contacts = 0
        table_support = False
        for contact_index, contact in enumerate(self.data.contact[:self.data.ncon]):
            pair = {int(contact.geom1), int(contact.geom2)}
            names = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in (int(contact.geom1),int(contact.geom2))]
            if any(n.startswith('left_') for n in names) and any(n.startswith('right_') for n in names):
                arm_contacts += 1
            if self.obstacle_geom in pair:
                obstacle_contacts += 1
            if self.payload_geom in pair:
                other = next((g for g in pair if g != self.payload_geom), self.payload_geom)
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other) or ''
                contact_force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, contact_index, contact_force)
                if '_pad_' in name and contact_force[0] > .5:
                    finger_contacts.add(name.split('_')[0])
                    pad_contacts[name.split('_')[0]].add(name.rsplit('_',1)[-1])
                table_support |= name == 'table' and contact_force[0] > .1
        distance = float(np.linalg.norm(position-self.goal))
        jid = self.id(mujoco.mjtObj.mjOBJ_JOINT, 'payload_free')
        d = self.model.jnt_dofadr[jid]
        speed = np.linalg.norm(self.data.qvel[d:d+6])
        rotation = self.data.xmat[self.payload_id].reshape(3,3)
        grasped = tuple(len(pad_contacts[side]) == 2 for side in ('left','right'))
        tilt = float(np.degrees(np.arccos(np.clip(rotation[2,2],-1,1))))
        footprint = np.abs(rotation) @ self.model.geom_size[self.payload_geom]
        inside_goal = bool(np.all(np.abs(position[:2]-self.goal[:2])+footprint[:2] <=
                                  self.model.site_size[self.goal_site,:2]+.002))
        # Physical release matters; an unused, empty gripper may remain closed.
        placed = bool(distance < .04 and speed < .08 and not finger_contacts and
                      table_support and tilt < 12 and inside_goal and not any(grasped))
        return {'payload_position': position, 'goal_distance': distance,
                'task': self.task, 'start': self.start.copy(), 'goal': self.goal.copy(),
                'selected_mode':self.selected_mode,
                'obstacle_position':self.data.geom_xpos[self.obstacle_geom].copy(),
                'obstacle_size':self.model.geom_size[self.obstacle_geom].copy(),
                'inside_goal': inside_goal,
                'grasped': grasped, 'arm_contacts': arm_contacts, 'tilt_degrees': tilt,
                'success_hold_seconds': getattr(self,'success_steps',0)*self.dt,
                'collision_total': getattr(self,'collision_total',0),
                'payload_mass': self.mass, 'weight_newtons': self.mass*abs(self.model.opt.gravity[2]),
                'obstacle_contacts': obstacle_contacts, 'gripping_arms': len(finger_contacts),
                'placed': placed, 'table_support': bool(table_support),
                'vertical_half_extent': float(footprint[2]),
                'gripper_force': self.data.actuator_force[[6,13]].copy(),
                'motor_torque': self.data.qfrc_actuator[self.ball_dofs].copy(),
                'mechanical_power': float(np.dot(self.data.qfrc_actuator[self.ball_dofs], self.data.qvel[self.ball_dofs]))}

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=900, width=1300)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = [0, 0, .80]
        camera.distance, camera.azimuth, camera.elevation = 2.55, 110, -35
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def close(self):
        if self.physics is not None:self.physics.close()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
