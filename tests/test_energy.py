import unittest

import numpy as np

from boltzmann_lab.core.config import DT, POTENTIAL_POINTS
from boltzmann_lab.core.simulation import BoltzmannSimulation


class EnergyTests(unittest.TestCase):
    def make_gas(self, potential=None):
        sim = BoltzmannSimulation()
        sim.rng = np.random.default_rng(2026)
        sim.temperature = 0.3
        if potential is None:
            x = np.linspace(0, 1, POTENTIAL_POINTS)
            potential = 4 * (x - 0.5) ** 2
        sim.set_potential(potential)
        sim.reset_particles()
        return sim

    def test_long_run_has_no_secular_energy_drift_but_temperature_fluctuates(self):
        sim = self.make_gas()
        initial = sim.mechanical_energy()
        energies, temperatures = [], []
        for i in range(30000):
            sim.step(DT)
            if i % 300 == 0:
                energies.append(sim.mechanical_energy())
                temperatures.append(sim.effective_temperature())
        self.assertLess(max(abs(e - initial) for e in energies) / initial, 1e-8)
        self.assertGreater(np.std(temperatures), 0.002)

    def test_nonflat_potentials_and_demon_remain_stable(self):
        x = np.linspace(0, 1, POTENTIAL_POINTS)
        for potential in (0.5 + 0.5 * np.cos(4 * np.pi * x), np.exp(-((x - 0.5) / 0.13) ** 2)):
            sim = self.make_gas(potential)
            sim.barrier_mode = 5
            sim.demon_rule_mode = "speed"
            sim.demon_assign_sides()
            initial = sim.mechanical_energy()
            for _ in range(5000):
                sim.step(DT)
            self.assertAlmostEqual(sim.mechanical_energy() / initial, 1.0, places=8)
            self.assertEqual(sim.demon_queue, [])

    def moving_wall_gas(self):
        sim = self.make_gas(np.zeros(POTENTIAL_POINTS))
        sim.n = 1
        sim.barrier_mode = 1
        sim.positions = np.array([[sim.barrier_x - sim.radius - 0.001, 0.5]])
        sim.velocities = np.array([[1.0, 0.0]])
        sim.barrier_goal = 0.4999
        return sim

    def test_physical_work_of_moving_wall_is_preserved(self):
        sim = self.moving_wall_gas()
        sim.step(DT)
        expected_velocity = 2 * sim.barrier_velocity - 1.0
        self.assertLess(expected_velocity, -1)
        self.assertAlmostEqual(sim.velocities[0, 0], expected_velocity)
        self.assertAlmostEqual(sim.mechanical_energy(), 0.5 * expected_velocity ** 2)
        self.assertGreater(sim.mechanical_energy(), 0.5)

    def test_wall_without_work_does_not_heat_the_gas(self):
        sim = self.moving_wall_gas()
        sim.barrier_does_work = False
        sim.step(DT)
        self.assertAlmostEqual(sim.mechanical_energy(), 0.5)

    def test_parameter_changes_define_a_new_energy_without_resetting_temperature(self):
        sim = self.make_gas()
        sim.energy_scale = 2.0
        target = sim.mechanical_energy()
        for _ in range(1000):
            sim.step(DT)
        self.assertAlmostEqual(sim.mechanical_energy(), target, places=8)

    def test_zero_velocities_remain_finite(self):
        sim = self.make_gas(np.zeros(POTENTIAL_POINTS))
        sim.velocities[:] = 0
        for _ in range(10):
            sim.step(DT)
        self.assertTrue(np.isfinite(sim.velocities).all())
        self.assertEqual(sim.kinetic_energy(), 0)


if __name__ == "__main__":
    unittest.main()
