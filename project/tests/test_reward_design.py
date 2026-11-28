from dataclasses import replace
import unittest

from simulation.reward_design import (RewardState, compute_reward, lift_condition,
                                      placement_condition, potential)


class RewardDesignTests(unittest.TestCase):
    def test_stability_signal_is_smooth_physical_and_cannot_be_farmed(self):
        held=RewardState((.02,.5),grasped=(True,False),grasp_seconds=(.1,0.),
                         clearance=.04,table_support=False,linear_speed=.12,angular_speed=.8)
        quieter=replace(held,linear_speed=.11,angular_speed=.7)
        self.assertFalse(lift_condition(held));self.assertFalse(lift_condition(quieter))
        self.assertGreater(potential(quieter,1),potential(held,1))
        loop=compute_reward(held,quieter,stage=1).terms['shaping']+.999*compute_reward(quieter,held,stage=1).terms['shaping']
        self.assertAlmostEqual(loop,(.999**2-1)*potential(held,1))
        unheld=replace(held,grasped=(False,False),grasp_seconds=(0.,0.))
        self.assertEqual(potential(unheld,1),potential(replace(unheld,linear_speed=0.,angular_speed=0.),1))
        supported=replace(held,table_support=True)
        self.assertEqual(potential(supported,1),potential(replace(supported,linear_speed=0.,angular_speed=0.),1))

    def test_far_approach_is_not_saturated_and_cannot_be_farmed_in_a_loop(self):
        far = RewardState((.70, 1.))
        near = replace(far, hand_distances=(.60, 1.))
        self.assertAlmostEqual(potential(near, 1)-potential(far, 1), 1.25)
        forward = compute_reward(far, near, stage=1)
        reverse = compute_reward(near, far, stage=1)
        self.assertAlmostEqual(sum(forward.breakdown.values()), forward.total)
        self.assertAlmostEqual(forward.breakdown['approach'], 1.25)
        self.assertLess(reverse.breakdown['approach'], 0.)
        # The discounted shaping sum of a closed loop is negative, not profit.
        shaped = forward.terms['shaping']+.999*reverse.terms['shaping']
        self.assertAlmostEqual(shaped, (.999**2-1)*potential(far, 1))
        self.assertLess(forward.total + reverse.total, 0.)

    def test_arms_are_symmetric_and_alignment_is_scored_for_the_same_arm(self):
        a = RewardState((.05, .5), closing_alignments=(.8, .2))
        b = replace(a, hand_distances=a.hand_distances[::-1],
                    closing_alignments=a.closing_alignments[::-1])
        self.assertEqual(potential(a, 1), potential(b, 1))
        self.assertGreater(potential(a, 1), potential(replace(a, closing_alignments=(0., .2)), 1))

    def test_pregrasp_progress_is_symmetric_bounded_to_one_arm_and_cannot_be_farmed(self):
        a = RewardState((.05, .5), closing_alignments=(.8, .2))
        b = replace(a, pregrasp_progress=(2., 3.))
        swapped = replace(b, hand_distances=b.hand_distances[::-1],
                          closing_alignments=b.closing_alignments[::-1],
                          pregrasp_progress=b.pregrasp_progress[::-1])
        self.assertEqual(potential(b, 1), potential(swapped, 1))
        self.assertAlmostEqual(potential(b, 1)-potential(a, 1), 2.)
        self.assertEqual(potential(b, 2), potential(a, 2))
        forward = compute_reward(a, b, stage=1)
        back = compute_reward(b, a, stage=1)
        self.assertAlmostEqual(forward.breakdown['pregrasp'], 2.)
        self.assertLess(forward.terms['shaping']+.999*back.terms['shaping'], 0.)

    def test_high_object_without_physical_grip_gets_no_lifting_credit(self):
        s = RewardState((.3, .5))
        self.assertEqual(potential(s, 1), potential(replace(s, clearance=.2), 1))
        self.assertFalse(lift_condition(replace(s, clearance=.2, table_support=False)))

    def test_initial_lift_shaping_weights_low_clearance_and_keeps_success_rules(self):
        grip = RewardState((.02, .5), grasped=(True, False), grasp_seconds=(.1, 0.))
        for clearance, expected in ((.01, 5.), (.015, 7.5), (.03, 12.), (.06, 15.)):
            with self.subTest(clearance=clearance):
                lifted = replace(grip, clearance=clearance, table_support=False)
                v5 = compute_reward(grip, lifted, stage=1, reward_scheme='grasp_progress')
                v6 = compute_reward(grip, lifted, stage=1, reward_scheme='lift_progress')
                self.assertAlmostEqual(v6.breakdown['lift'], expected)
                self.assertAlmostEqual(v5.breakdown['lift'], 15.*clearance/.06)
                self.assertFalse(v6.success)
                self.assertEqual(potential(lifted, 2), potential(lifted, 2, reward_scheme='lift_progress'))
        lifted = replace(grip, clearance=.06, table_support=False)
        unheld = replace(lifted, grasped=(False, False), grasp_seconds=(0., 0.))
        self.assertEqual(compute_reward(grip, unheld, stage=1, reward_scheme='lift_progress').breakdown['lift'], 0.)
        success = replace(lifted, clearance=.03, lift_hold_seconds=.5)
        self.assertTrue(compute_reward(lifted, success, stage=1, reward_scheme='lift_progress').success)
        up = compute_reward(grip, lifted, stage=1, reward_scheme='lift_progress')
        down = compute_reward(lifted, grip, stage=1, reward_scheme='lift_progress')
        self.assertLess(up.terms['shaping']+.999*down.terms['shaping'], 0.)

    def test_success_requires_actual_stable_lift_and_terminal_potential_is_zero(self):
        s = RewardState((.005, .6), grasped=(True, False), grasp_seconds=(.1, 0.),
                        clearance=.04, table_support=False, lift_hold_seconds=.5)
        before = replace(s, lift_hold_seconds=.48)
        result = compute_reward(before, s, stage=1)
        self.assertTrue(result.success and result.terminated)
        self.assertEqual(result.potential_after, 0.)
        self.assertAlmostEqual(result.total, 100.-potential(before, 1)-.004)
        self.assertAlmostEqual(sum(result.breakdown.values()), result.total)
        self.assertEqual(result.breakdown['success'], 100.)
        self.assertAlmostEqual(result.breakdown['terminal_progress'], -potential(before, 1))
        self.assertFalse(compute_reward(before, replace(s, clearance=.02), stage=1).success)

    def test_goal_approach_does_not_reward_lowering_far_from_goal(self):
        s = RewardState((.005, .6), carry_qualified=True, initial_goal_distance=.6,
                        goal_distance=.4, clearance=.15)
        self.assertEqual(potential(s, 2), potential(replace(s, clearance=0.), 2))
        near = replace(s, goal_distance=.02)
        self.assertGreater(potential(replace(near, clearance=0.), 2), potential(near, 2))

    def test_safe_release_keeps_pickup_credit_and_unsafe_release_is_penalized(self):
        held = RewardState((.005, .5), grasped=(True, False), grasp_seconds=(.1, 0.),
                           carry_qualified=True, goal_distance=.01, initial_goal_distance=.3,
                           clearance=0., table_support=True, inside_goal=True, robot_contacts=2)
        released = replace(held, grasped=(False, False), grasp_seconds=(0., 0.), robot_contacts=0)
        safe = compute_reward(held, released, stage=2)
        self.assertEqual(safe.terms['premature_release'], 0.)
        self.assertGreater(potential(released, 2), potential(held, 2))
        unsafe = replace(released, table_support=False, clearance=.2)
        self.assertEqual(compute_reward(held, unsafe, stage=2).terms['premature_release'], -2.)

    def test_transfer_does_not_require_fixed_arm_roles_or_pay_a_transfer_bonus(self):
        left = RewardState((.05, .05), grasped=(True, False), grasp_seconds=(.1, 0.),
                           used_arms=(True, False), carry_qualified=True)
        right = replace(left, grasped=(False, True), grasp_seconds=(0., .1), used_arms=(True, True))
        result = compute_reward(left, right, stage=3)
        self.assertEqual(result.terms['premature_release'], 0.)
        self.assertEqual(result.terms['second_arm'], 0.)
        self.assertNotIn('handoff_bonus', result.terms)

    def test_placement_needs_full_release_support_and_separate_velocity_limits(self):
        s = RewardState((.4, .5), carry_qualified=True, goal_distance=.01,
                        inside_goal=True, table_support=True, placement_hold_seconds=1.)
        self.assertTrue(placement_condition(s))
        self.assertTrue(compute_reward(replace(s, placement_hold_seconds=.98), s, stage=2).success)
        for bad in [replace(s, robot_contacts=1), replace(s, linear_speed=.04),
                    replace(s, angular_speed=.2), replace(s, table_support=False),
                    replace(s, carry_qualified=False)]:
            self.assertFalse(placement_condition(bad))

    def test_effort_costs_ramp_and_count_holding_force_and_second_arm_once(self):
        before = RewardState((.1, .1), used_arms=(True, False))
        after = replace(before, used_arms=(True, True), torque_integral=.02, gripper_integral=.01)
        full = compute_reward(before, after, stage=4, effort_scale=1.)
        half = compute_reward(before, after, stage=4, effort_scale=.5)
        for name in ('torque', 'gripper', 'second_arm'):
            self.assertAlmostEqual(half.terms[name], full.terms[name]/2)
            self.assertLess(full.terms[name], 0.)
        self.assertEqual(compute_reward(after, after, stage=4, effort_scale=1.).terms['second_arm'], 0.)
        self.assertEqual(compute_reward(before, after, stage=3).terms['torque'], 0.)

    def test_fall_cannot_coincide_with_success_and_timeout_keeps_physical_potential(self):
        before = RewardState((.2, .5))
        fallen = replace(before, fallen=True, lift_hold_seconds=10., placement_hold_seconds=10.)
        result = compute_reward(before, fallen, stage=1)
        self.assertFalse(result.success)
        self.assertTrue(result.terminated)
        self.assertEqual(result.terms['fall'], -25.)
        waiting = compute_reward(before, before, stage=1)
        self.assertFalse(waiting.terminated)
        self.assertEqual(waiting.potential_after, potential(before, 1))


if __name__ == '__main__':
    unittest.main()
