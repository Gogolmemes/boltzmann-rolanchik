import unittest

import numpy as np

from boltzmann_lab.core.simulation import BoltzmannSimulation


class DemonTests(unittest.TestCase):
    def setUp(self):
        self.sim = BoltzmannSimulation()
        self.sim.n = 1
        self.sim.barrier_mode = 5
        self.sim.positions = np.array([[self.sim.barrier_x - 0.1, 0.5]])
        self.sim.velocities = np.array([[0.6, 0.8]])
        self.sim.colors = np.array([1])
        self.sim.demon_assign_sides()

    def cross(self, side=0):
        self.sim.demon_side[:] = side
        self.sim.positions[0, 0] = self.sim.barrier_x + (0.01 if side == 0 else -0.01)
        self.sim.velocities[0, 0] = abs(self.sim.velocities[0, 0]) * (1 if side == 0 else -1)
        self.sim._demon_collision()

    def test_speed_rules_use_full_speed_and_include_equality_in_slow_group(self):
        self.sim.demon_rule_mode = "speed"
        self.sim.demon_speed_pass = [3, 0]
        self.sim.demon_speed_else = [1, 1]
        self.sim.demon_speed_threshold = 1.0
        self.cross()
        self.assertEqual(self.sim.demon_side[0], 1)
        self.sim.demon_speed_threshold = 0.9
        self.cross()
        self.assertEqual(self.sim.demon_side[0], 0)
        self.assertLess(self.sim.velocities[0, 0], 0)
        self.assertEqual(self.sim.demon_queue, [])

    def test_speed_directions_for_both_groups(self):
        self.sim.demon_rule_mode = "speed"
        self.sim.demon_speed_else = [1, 1]
        for speed in (0.5, 2.0):
            for rule in range(4):
                for side in (0, 1):
                    with self.subTest(speed=speed, rule=rule, side=side):
                        self.sim.velocities[:] = [speed, 0]
                        self.sim.demon_speed_pass = [rule, rule]
                        self.cross(side)
                        allowed = rule == 3 or (rule == 1 and side == 0) or (rule == 2 and side == 1)
                        self.assertEqual(self.sim.demon_side[0], 1 - side if allowed else side)
                        self.assertEqual(self.sim.demon_queue, [])

    def test_speed_rule_can_ask_and_manual_decision_still_works(self):
        self.sim.demon_rule_mode = "speed"
        self.sim.demon_speed_pass = [0, 0]
        self.sim.demon_speed_else = [0, 0]
        self.cross()
        self.assertEqual(self.sim.demon_queue, [0])
        self.sim.demon_resolve(True)
        self.assertEqual(self.sim.demon_side[0], 1)
        self.assertGreater(self.sim.positions[0, 0], self.sim.barrier_x)
        self.assertEqual(self.sim.demon_queue, [])

    def test_dynamic_colors_preserve_original_colors(self):
        self.sim.demon_color_mode = "speed"
        self.sim.demon_color_threshold = 1.0
        self.assertEqual(self.sim.particle_color_indices().tolist(), [2])
        self.sim.velocities[:] = [1, 1]
        self.assertEqual(self.sim.particle_color_indices().tolist(), [0])
        self.sim.demon_color_mode = "random"
        self.assertEqual(self.sim.particle_color_indices().tolist(), [1])

    def test_color_rules_follow_the_visible_color(self):
        self.sim.demon_color_mode = "speed"
        self.sim.demon_color_threshold = 0.9
        self.sim.demon_pass = [3, 0, 0]
        self.sim.demon_else = [1, 1, 1]
        self.cross()
        self.assertEqual(self.sim.demon_side[0], 1)

    def test_original_color_rules_are_preserved(self):
        self.sim.demon_pass = [0, 3, 0]
        self.cross()
        self.assertEqual(self.sim.demon_side[0], 1)

    def test_speed_rules_and_coloring_have_independent_thresholds(self):
        self.sim.demon_rule_mode = "speed"
        self.sim.demon_color_mode = "speed"
        self.sim.demon_color_threshold = 0.5
        self.sim.demon_speed_threshold = 2.0
        self.sim.demon_speed_pass = [3, 0]
        self.assertEqual(self.sim.particle_color_indices().tolist(), [0])
        self.cross()
        self.assertEqual(self.sim.demon_side[0], 1)


if __name__ == "__main__":
    unittest.main()
