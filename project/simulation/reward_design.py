"""Active v4 rewards for the progressive PPO, A2C and SAC curriculum.

Inputs describe physical states, not action commands. Distances are measured
to valid grasp points on the object, including off-centre grips for handoffs.
The caller must maintain continuous hold timers and physical carry eligibility.
This module neither assigns arm roles nor generates actions.
"""
from dataclasses import dataclass
import math


def _unit(value):
    return min(1., max(0., value))


@dataclass(frozen=True)
class RewardState:
    hand_distances: tuple[float, float]
    closing_alignments: tuple[float, float] = (0., 0.)
    grasped: tuple[bool, bool] = (False, False)
    grasp_seconds: tuple[float, float] = (0., 0.)
    used_arms: tuple[bool, bool] = (False, False)
    clearance: float = 0.
    lift_hold_seconds: float = 0.
    carry_qualified: bool = False
    goal_distance: float = 1.
    initial_goal_distance: float = 1.
    inside_goal: bool = False
    table_support: bool = True
    robot_contacts: int = 0
    linear_speed: float = 0.
    angular_speed: float = 0.
    placement_hold_seconds: float = 0.
    obstacle_contacts: int = 0
    arm_contacts: int = 0
    torque_integral: float = 0.
    gripper_integral: float = 0.
    fallen: bool = False

    def __post_init__(self):
        for name in ('hand_distances', 'closing_alignments', 'grasped', 'grasp_seconds', 'used_arms'):
            if len(getattr(self, name)) != 2:
                raise ValueError(f'{name} must describe exactly two arms')
        nonnegative = (*self.hand_distances, *self.grasp_seconds,
            self.lift_hold_seconds, self.goal_distance, self.initial_goal_distance,
            self.robot_contacts, self.linear_speed, self.angular_speed,
            self.placement_hold_seconds, self.obstacle_contacts, self.arm_contacts,
            self.torque_integral, self.gripper_integral)
        if any(not math.isfinite(v) or v < 0 for v in nonnegative):
            raise ValueError('Distances, timers, speeds, contacts and force integrals must be finite and nonnegative')
        if not math.isfinite(self.clearance):
            raise ValueError('clearance must be finite')
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in self.closing_alignments):
            raise ValueError('closing_alignments must be in [0, 1]')


@dataclass(frozen=True)
class RewardResult:
    terms: dict[str, float]
    success: bool
    terminated: bool
    potential_before: float
    potential_after: float
    breakdown: dict[str, float]

    @property
    def total(self):
        return float(sum(self.terms.values()))


def _check_stage(stage):
    if stage not in (1, 2, 3, 4):
        raise ValueError('stage must be 1, 2, 3 or 4')


def lift_condition(state):
    """The caller increments lift_hold_seconds only while this is true."""
    stable_grip = any(g and seconds >= .1-1e-9
                      for g, seconds in zip(state.grasped, state.grasp_seconds))
    return bool(stable_grip and state.clearance >= .03-1e-9
                and not state.table_support and state.linear_speed <= .1
                and state.angular_speed <= .5 and not state.fallen)


def placement_condition(state):
    """Use separate SI-unit velocity limits instead of mixing m/s and rad/s."""
    return bool(state.carry_qualified and state.goal_distance < .04
                and state.inside_goal and state.table_support
                and state.robot_contacts == 0 and not any(state.grasped)
                and state.linear_speed < .03 and state.angular_speed < .15
                and not state.fallen)


def success_condition(state, stage):
    _check_stage(stage)
    if stage == 1:
        return lift_condition(state) and state.lift_hold_seconds >= .5-1e-9
    return placement_condition(state) and state.placement_hold_seconds >= 1.-1e-9


def _pickup_progress(state, include_hold):
    # A single arm must supply both the distance and alignment score. This also
    # treats the two arms symmetrically and does not reward using both of them.
    approach = max(10.*(1.-_unit(distance/.8))
                   + 2.*(1.-_unit(distance/.15))*alignment
                   for distance, alignment in zip(state.hand_distances, state.closing_alignments))
    grasp = max(seconds if present else 0.
                for present, seconds in zip(state.grasped, state.grasp_seconds))
    g = _unit(grasp/.1)
    # Give a smooth signal before the strict velocity thresholds are met.
    # This is a bounded state potential, not a recurring holding bonus, and
    # cannot pay for a quiet object left on the table or lifted without a grip.
    quiet = 1./(1.+(state.linear_speed/.1)**2+(state.angular_speed/.5)**2)
    stability = 4.*_unit(state.clearance/.03)*g*quiet if not state.table_support else 0.
    hold = 4.*_unit(state.lift_hold_seconds/.5) if include_hold else 0.
    return dict(approach=approach, grasp=5.*g, lift=15.*_unit(state.clearance/.06)*g,
                stability=stability, lift_hold=hold)


def _progress_components(state, stage):
    _check_stage(stage)
    if stage == 1 or not state.carry_qualified:
        return _pickup_progress(state, include_hold=stage == 1)
    scale = max(.25, state.initial_goal_distance)
    progress = _unit(1.-state.goal_distance/scale)
    near_goal = 1.-_unit(state.goal_distance/.10)
    lowering = 1.-_unit(max(0., state.clearance)/.10)
    released = float(state.inside_goal and state.table_support
                     and state.robot_contacts == 0 and not any(state.grasped))
    settled = _unit(state.placement_hold_seconds/1.)
    # Preserve physical pickup credit through legitimate release. Qualified
    # carry must be cleared if the payload is put on the table outside the goal.
    return dict(carry_ready=36., transport=25.*progress, lowering=12.*near_goal*lowering,
                release=5.*released, placement_hold=8.*settled)


def potential(state, stage):
    return float(sum(_progress_components(state, stage).values()))


def compute_reward(before, after, *, stage, gamma=.999, dt=.02, effort_scale=0.):
    """Calculate one transition in a single episode, without mutating inputs.

    At success/fall the next potential is zero. A time limit alone is not a
    terminal state here: keep its physical next potential and bootstrap in PPO.
    Force integrals on ``after`` cover this action interval, not the episode.
    The first ``before`` must be sampled after reset/pickup preparation.
    """
    _check_stage(stage)
    if not math.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError('gamma must be in (0, 1]')
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be positive')
    if not math.isfinite(effort_scale) or not 0 <= effort_scale <= 1:
        raise ValueError('effort_scale must be in [0, 1]')
    if stage != 4 and effort_scale != 0:
        raise ValueError('Force and arm-use costs are enabled only in stage four')
    if any(a and not b for a, b in zip(before.used_arms, after.used_arms)):
        raise ValueError('used_arms must be monotonic within one episode')
    success = success_condition(after, stage)
    terminated = bool(success or after.fallen)
    before_progress = _progress_components(before, stage)
    after_progress = {} if terminated else _progress_components(after, stage)
    previous = float(sum(before_progress.values()))
    following = float(sum(after_progress.values()))
    stable_before = any(g and seconds >= .1-1e-9
                        for g, seconds in zip(before.grasped, before.grasp_seconds))
    released_safely = (after.inside_goal and after.table_support
                       and after.linear_speed <= .1 and after.angular_speed <= .5)
    lost_grip = (stage >= 2 and before.carry_qualified and stable_before
                 and not any(after.grasped) and not released_safely)
    second_arm = all(after.used_arms) and not all(before.used_arms)
    contacts = min(5, after.obstacle_contacts + after.arm_contacts)
    terms = dict(shaping=gamma*following-previous,
                 success=100. if success else 0.,
                 fall=-25. if after.fallen else 0.,
                 collision=-.5*dt*contacts,
                 time=-.2*dt,
                 premature_release=-2. if lost_grip else 0.,
                 torque=-.5*effort_scale*after.torque_integral,
                 gripper=-.05*effort_scale*after.gripper_integral,
                 second_arm=-2.*effort_scale if second_arm else 0.)
    breakdown = {key:value for key,value in terms.items() if key != 'shaping'}
    if terminated:
        breakdown['terminal_progress'] = -previous
    else:
        for key in before_progress.keys() | after_progress.keys():
            breakdown[key] = float(after_progress.get(key, 0.) - before_progress.get(key, 0.))
        breakdown['progress_discount'] = (gamma-1.)*following
    return RewardResult(terms, bool(success), terminated, previous, following, breakdown)
