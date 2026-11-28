"""Independently sample safe joint poses during reset, before any policy action."""
import mujoco
import numpy as np

from simulation.dual_reacher import MAX_ROT, rotvec_quat
from simulation.layout_sampling import HOME


def _home_action(env):
    cached = env.__dict__.get('_independent_home_action')
    if cached is None:
        seed = np.ones(14)
        for index in range(4):
            offset = (index // 2)*7+(index % 2)*3
            seed[offset:offset+3] = np.array([0., .3 if index % 2 == 0 else -.55,
                                             -.7 if index % 2 == 0 else 1.3])/MAX_ROT
        cached = env.solve_ik(HOME, 1., iterations=160, seed_action=seed)
        for offset in (0, 3, 7, 10):
            length = np.linalg.norm(cached[offset:offset+3])
            if length > .95:
                cached[offset:offset+3] *= .95/length
        env._independent_home_action = cached.copy()
    return cached.copy()


def randomize_arms(env, attempts=1024):
    """Sample joints independently of object coordinates, then reject unsafe poses.

    The object layout stays fixed. Collision/distance filters condition the
    samples, but no IK target is placed at the object or its goal.
    """
    baseline = _home_action(env)
    scratch = env.ik_data
    minimum = .15+.05*env.difficulty
    maximum = .55+.20*env.difficulty
    amplitude = .25+.15*env.difficulty
    robot_geoms = env.__dict__.get('_robot_geom_mask')
    if robot_geoms is None:
        robot_geoms = np.array([(mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or '').startswith(('left_', 'right_'))
                                for geom in range(env.model.ngeom)])
        env._robot_geom_mask = robot_geoms
    for _ in range(attempts):
        action = baseline.copy()
        scratch.qpos[:] = env.data.qpos
        scratch.qvel[:] = 0.
        scratch.mocap_pos[:] = env.data.mocap_pos
        scratch.mocap_quat[:] = env.data.mocap_quat
        for index, (qpos, _) in enumerate(env.joints):
            offset = (index // 2)*7+(index % 2)*3
            vector = baseline[offset:offset+3]+env.np_random.uniform(-amplitude, amplitude, 3)
            length = np.linalg.norm(vector)
            if length > .98:
                vector *= .98/length
            action[offset:offset+3] = vector
            scratch.qpos[qpos:qpos+4] = rotvec_quat(vector*MAX_ROT)
        for qpos in env.finger_joints:
            scratch.qpos[qpos] = .045
        action[[6, 13]] = 1.
        mujoco.mj_forward(env.model, scratch)
        hands = scratch.site_xpos[env.sites]
        if (np.any(hands[:, 2] < .78) or np.any(hands[:, 2] > 1.35) or
                np.any(np.abs(hands[:, :2]) > [.9, .62])):
            continue
        distances = np.linalg.norm(hands-scratch.xpos[env.payload_id], axis=1)
        if np.any(distances < minimum):
            continue
        distance = distances[env.initial_side] if env.stage == 1 else distances.min()
        if distance > maximum:
            continue
        if any(contact.dist < -.0001 and (robot_geoms[contact.geom1] or robot_geoms[contact.geom2])
               for contact in scratch.contact[:scratch.ncon]):
            continue
        env.apply_pose(action)
        for qpos in env.finger_joints:
            env.data.qpos[qpos] = .045
        env.target_action = action.copy()
        mujoco.mj_forward(env.model, env.data)
        env._control(action)
        mujoco.mj_forward(env.model, env.data)
        return
    raise RuntimeError('Could not sample collision-free random arm poses away from the object')
