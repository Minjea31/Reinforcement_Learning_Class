"""Reset-only layout constraints; actions and robot roles remain policy outputs."""
import mujoco
import numpy as np

from simulation.dual_reacher import DualReacherEnv, MAX_ROT, quat_rotvec

HIDDEN_OBSTACLE = np.array([3., 0., .7])
OBSTACLE_SIZE = np.array([.06, .035, .05])
HOME = np.array([[-.30, -.34, .94], [.30, -.34, .94]])


def markers_separated(start, goal):
    delta = np.abs(np.asarray(goal)[:2] - np.asarray(start)[:2])
    return bool(np.linalg.norm(delta) >= .22 and np.any(delta > np.array([.26, .16]) + .025))


def reachable(env, point, arm):
    key = (arm, *np.asarray(point, dtype=float))
    cache = env.__dict__.setdefault('_layout_reach_cache', {})
    if key not in cache:
        if len(cache) > 4096:
            cache.clear()
        cache[key] = _reachable(env, point, arm)
    return cache[key]


def _reachable(env, point, arm):
    """Check bounded joint IK in scratch data, without modifying the simulation."""
    base_x = -.68 if arm == 0 else .68
    if np.linalg.norm(np.asarray(point)[:2] - [base_x, 0.]) > .65:
        return False
    targets = HOME.copy()
    targets[arm] = point
    seed = np.ones(14)
    for index in range(4):
        offset = (index // 2) * 7 + (index % 2) * 3
        seed[offset:offset+3] = np.array([0., .3 if index % 2 == 0 else -.55,
                                         -.7 if index % 2 == 0 else 1.3]) / MAX_ROT
    action = env.solve_ik(targets, 1., iterations=90, seed_action=seed)
    for joint in (arm*2, arm*2+1):
        qpos, _ = env.joints[joint]
        if np.linalg.norm(quat_rotvec(env.ik_data.qpos[qpos:qpos+4])) > MAX_ROT + 1e-8:
            return False
    if any(np.linalg.norm(action[offset:offset+3]) > 1. + 1e-8 for offset in (arm*7, arm*7+3)):
        return False
    scratch = env.ik_data
    for joint in env.finger_joints:
        scratch.qpos[joint] = .045
    scratch.mocap_pos[env.obstacle_mocap] = HIDDEN_OBSTACLE
    mujoco.mj_forward(env.model, scratch)
    site = env.sites[arm]
    axis = scratch.site_xmat[site].reshape(3, 3)[:, 1]
    if np.linalg.norm(scratch.site_xpos[site] - point) > .007 or axis[1] * (1 if arm == 0 else -1) < .98:
        return False
    for contact in scratch.contact[:scratch.ncon]:
        if contact.dist >= -.001 or env.payload_geom in (contact.geom1, contact.geom2):
            continue
        names = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ''
                 for geom in (contact.geom1, contact.geom2)]
        arms = [name.startswith(('left_', 'right_')) for name in names]
        if any(arms) and (all(arms) or any(name in ('table', 'floor', 'obstacle') for name in names)):
            return False
    return True


def single_arm_route(env, start, goal, arm):
    """One arm can reach pickup, raised straight-line waypoints and placement."""
    for fraction in np.linspace(0., 1., 5):
        point = (1-fraction) * np.asarray(start) + fraction * np.asarray(goal)
        if not env._clear_table_area(point, [.13, .08], HIDDEN_OBSTACLE, OBSTACLE_SIZE):
            return False
        if not reachable(env, point + [0., 0., .14], arm):
            return False
    return all(reachable(env, np.asarray(point) + [0., 0., .005], arm) for point in (start, goal))


def solo_point(env, arm):
    point = np.r_[env.np_random.uniform([.16, -.32], [.48, .32]), .625]
    point[0] *= -1 if arm == 0 else 1
    return point


def solo_goal(env, start, arm):
    if not env._clear_table_area(start, [.13,.08], HIDDEN_OBSTACLE, OBSTACLE_SIZE):
        return None
    if not all(reachable(env, np.asarray(start)+[0.,0.,height], arm) for height in (.005,.14)):
        return None
    for _ in range(96):
        goal = solo_point(env, arm)
        if markers_separated(start, goal) and single_arm_route(env, start, goal, arm):
            return goal
    return None


def solo_layout(env, arm=None):
    sampled_arm = int(env.np_random.integers(2))
    arm = sampled_arm if arm is None else arm
    for _ in range(64):
        start = solo_point(env, arm)
        if not env._clear_table_area(start, [.13, .08], HIDDEN_OBSTACLE, OBSTACLE_SIZE):
            continue
        goal = solo_goal(env, start, arm)
        if goal is not None:
            env.initial_side = arm
            # Keep the curriculum's open gripper waiting near the random start.
            env.random_episode = False
            return start, goal, HIDDEN_OBSTACLE.copy(), OBSTACLE_SIZE.copy()
    raise RuntimeError('Could not sample a separated single-arm reachable layout')


def full_layout(env, options):
    for _ in range(128):
        start, goal, obstacle, size = DualReacherEnv._sample_random_layout(env, True, options)
        if not markers_separated(start, goal):
            continue
        if not env._clear_table_area(start, [.13,.08], obstacle, size):
            continue
        if not any(reachable(env, start + [0., 0., .005], arm) for arm in (0, 1)):
            continue
        if not any(reachable(env, goal + [0., 0., .005], arm) for arm in (0, 1)):
            continue
        env.random_episode = True
        return start, goal, obstacle, size
    raise RuntimeError('Could not sample reachable random start/goal and obstacle positions')
