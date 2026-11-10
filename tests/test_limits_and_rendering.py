import copy
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
from PyQt6.QtWidgets import QApplication, QPushButton
from boltzmann_lab.core.config import DT, FIELD_WIDTH, FIELD_HEIGHT, POTENTIAL_POINTS, WORLD_W
from boltzmann_lab.core.limits import (
    MAX_PARTICLES, MAX_FILL_FRACTION, MAX_CHAMBER_FILL_FRACTION,
    PARAMETER_RANGES, barrier_motion_limits, fill_fraction,
    particle_capacity, validate_parameters,
)
from boltzmann_lab.core.physics import NUMBA_OK, sample_uniform, warmup_numba
from boltzmann_lab.core.simulation import BoltzmannSimulation
from boltzmann_lab.ui.raster import warmup_raster
from boltzmann_lab.ui.renderer import ParticleRendererMixin
from boltzmann_lab.ui.windows import MainWindow


class ModelLimitTests(unittest.TestCase):
    def test_density_leaves_room_for_many_small_particles(self):
        self.assertEqual(particle_capacity(1), MAX_PARTICLES)
        for radius in range(1, 6):
            capacity = particle_capacity(radius)
            self.assertLessEqual(fill_fraction(capacity, radius), MAX_FILL_FRACTION)
            if capacity < MAX_PARTICLES:
                self.assertGreater(fill_fraction(capacity + 1, radius), MAX_FILL_FRACTION)

    def test_invalid_parameters_fail_before_a_calculation(self):
        sim = BoltzmannSimulation()
        for name, (_, maximum) in PARAMETER_RANGES.items():
            original = getattr(sim, name)
            for invalid in (float("nan"), float("inf"), maximum + 1, True):
                with self.subTest(name=name, invalid=invalid):
                    setattr(sim, name, invalid)
                    with self.assertRaises(ValueError):
                        validate_parameters(sim)
                    setattr(sim, name, original)

    def test_impossible_density_and_radius_are_rejected_without_changing_radius(self):
        sim = BoltzmannSimulation()
        sim.set_radius_px(1)
        sim.n = MAX_PARTICLES
        sim.reset_particles(equilibrium=False)
        with self.assertRaises(ValueError):
            sim.set_radius_px(5)
        self.assertEqual(sim.radius_px, 1)
        sim.n = MAX_PARTICLES + 1
        with self.assertRaises(ValueError):
            sim.reset_particles()

    def test_bad_curves_and_time_steps_are_rejected_without_mutation(self):
        sim = BoltzmannSimulation()
        original = sim.potential.copy()
        for curve in ([], [1], [[0, 1]], [0, np.nan], [0, np.inf], [-0.1, 1], [0, 2]):
            with self.assertRaises(ValueError):
                sim.set_potential(curve)
            np.testing.assert_array_equal(sim.potential, original)
        sim.set_potential([0, 0])
        original = sim.positions.copy()
        for dt in (0, -1, np.nan, np.inf, 2 * DT):
            with self.assertRaises(ValueError):
                sim.step(dt)
        np.testing.assert_array_equal(sim.positions, original)

    def test_compression_limits_only_block_closed_directions(self):
        sim = BoltzmannSimulation()
        sim.n = 1000
        sim.positions = np.column_stack((np.full(sim.n, WORLD_W / 4), np.full(sim.n, 0.5)))
        sim.barrier_mode = 1
        lower, upper = barrier_motion_limits(sim)
        self.assertGreater(lower, 0.1)
        self.assertAlmostEqual(upper, 0.9)
        self.assertLessEqual(fill_fraction(sim.n, sim.radius_px, lower), MAX_CHAMBER_FILL_FRACTION + 1e-12)
        sim.barrier_mode = 4
        sim.p_lr = 1.0
        self.assertEqual(barrier_motion_limits(sim), (0.1, 0.9))
        sim.p_lr = 0.0
        self.assertGreater(barrier_motion_limits(sim)[0], 0.1)

    def test_new_interpolation_matches_numpy_including_endpoints(self):
        rng = np.random.default_rng(8)
        values = rng.random(POTENTIAL_POINTS)
        xs = np.r_[[-1, 0, WORLD_W, WORLD_W + 1], rng.uniform(0, WORLD_W, 500)]
        actual = np.array([sample_uniform(values, x, WORLD_W) for x in xs])
        expected = np.interp(xs, np.linspace(0, WORLD_W, POTENTIAL_POINTS), values)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=2e-13)

    @unittest.skipUnless(NUMBA_OK, "Numba is not installed")
    def test_optimized_step_agrees_with_numpy_reference_in_all_barrier_modes(self):
        for mode in range(6):
            with self.subTest(mode=mode):
                sim = BoltzmannSimulation()
                sim.n = 2
                sim.set_radius_px(1)
                sim.set_potential(np.linspace(0, 1, POTENTIAL_POINTS))
                sim.positions = np.array([[0.7, 0.3], [2.4, 0.7]])
                sim.velocities = np.array([[0.3, 0.1], [-0.4, -0.2]])
                sim.colors = np.zeros(2, dtype=int)
                sim.barrier_mode = mode
                sim.demon_assign_sides()
                reference = copy.deepcopy(sim)
                sim.step(DT)
                with patch("boltzmann_lab.core.simulation.NUMBA_OK", False):
                    reference.step(DT)
                np.testing.assert_allclose(sim.positions, reference.positions, atol=1e-12)
                np.testing.assert_allclose(sim.velocities, reference.velocities, atol=1e-12)

    @unittest.skipUnless(NUMBA_OK, "Numba is not installed")
    def test_scatterer_reflections_match_reference_for_all_shapes_and_motion(self):
        rng = np.random.default_rng(2026)
        for shape in range(6):
            for angle in (0, 37, 359):
                with self.subTest(shape=shape, angle=angle):
                    sim = BoltzmannSimulation()
                    sim.barrier_mode = 3
                    sim.scatter_shape = shape
                    sim.scatter_angle = angle
                    sim.scatter_columns = 3
                    sim.positions = rng.uniform([sim.barrier_x - 0.5, 0], [sim.barrier_x + 0.5, 1], (sim.n, 2))
                    sim.velocities = rng.normal(size=(sim.n, 2))
                    sim.barrier_velocity = 0.5 if angle == 37 else -0.2
                    reference = copy.deepcopy(sim)
                    sim._collide_scatterers()
                    with patch("boltzmann_lab.core.simulation.NUMBA_OK", False):
                        reference._collide_scatterers()
                    np.testing.assert_allclose(sim.positions, reference.positions, atol=1e-10)
                    np.testing.assert_allclose(sim.velocities, reference.velocities, atol=1e-10)

    def test_statistics_preserve_the_original_free_energy(self):
        sim = BoltzmannSimulation()
        sim.set_potential([0, 1, 0])
        for bins in (20, 40, 60):
            snapshot = sim.statistics(bins)
            self.assertAlmostEqual(snapshot[1], sim.entropy(bins))
            self.assertAlmostEqual(snapshot[4], sim.free_energy())
            self.assertAlmostEqual(snapshot[2], sim.effective_temperature())
            self.assertAlmostEqual(snapshot[3], sim.mean_potential_energy())


class RenderingAndLimitsInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        warmup_numba()
        warmup_raster()

    def setUp(self):
        self.window = MainWindow()
        self.window.timer.stop()

    def tearDown(self):
        self.window.shutdown()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_compiled_disks_match_pygame_pixel_for_pixel_at_edges_and_overlaps(self):
        painter = ParticleRendererMixin()
        rng = np.random.default_rng(9)
        xs = np.array([-2, 0, 1, 22, 22, 39, 42], dtype=np.int64)
        ys = np.array([10, 0, 1, 15, 15, 29, 29], dtype=np.int64)
        colors = rng.integers(0, 256, size=(len(xs), 3), dtype=np.uint32)
        for radius in range(1, 6):
            reference = pygame.Surface((40, 30), depth=32)
            optimized = pygame.Surface((40, 30), depth=32)
            reference.fill((5, 10, 22))
            optimized.fill((5, 10, 22))
            for x, y, color in zip(xs.tolist(), ys.tolist(), colors.tolist()):
                pygame.draw.circle(reference, color, (x, y), radius)
                pygame.draw.circle(reference, (235, 245, 255), (x, y), radius, 1)
            painter._paint_disks(optimized, xs, ys, colors, radius)
            np.testing.assert_array_equal(pygame.surfarray.array2d(optimized),
                                          pygame.surfarray.array2d(reference))
            self.assertFalse(optimized.get_locked())

    def test_count_and_radius_controls_never_create_an_impossible_gas(self):
        w = self.window
        w.radius_spin.setValue(1)
        w.particles_spin.setValue(10000)
        self.assertEqual(w.sim.n, MAX_PARTICLES)
        self.assertEqual(w.radius_spin.maximum(), 3)
        w.radius_spin.setValue(5)
        self.assertEqual(w.sim.radius_px, 3)
        w.particles_spin.setValue(120)
        w.radius_spin.setValue(5)
        self.assertEqual(w.particles_spin.maximum(), particle_capacity(5))
        w.particles_spin.setValue(10000)
        self.assertEqual(w.sim.n, particle_capacity(5))
        validate_parameters(w.sim)

    def test_limits_are_identical_with_and_without_acceleration(self):
        w = self.window
        w.radius_spin.setValue(1)
        count_limit = w.particles_spin.maximum()
        speed_limit = w.playback_speed_spin.maximum()
        with patch("boltzmann_lab.core.simulation.NUMBA_OK", False), patch("boltzmann_lab.ui.renderer.NUMBA_OK", False):
            for _ in range(3):
                w.animation_frame()
        self.assertEqual(w.particles_spin.maximum(), count_limit)
        self.assertEqual(w.playback_speed_spin.maximum(), speed_limit)
        self.assertTrue(w.running)
        self.assertFalse(hasattr(w, "performance"))

    def test_capacity_is_enforced_in_controls_without_extra_explanations(self):
        w = self.window
        w.particles_spin.setValue(1)
        for radius in range(1, 6):
            w.radius_spin.setValue(radius)
            self.assertEqual(w.particles_spin.maximum(), particle_capacity(radius))
        for language in ("ru", "en"):
            w.set_language(language)
            self.assertFalse(w.particles_spin.toolTip())
            self.assertFalse(any("performance" in button.text().lower()
                                 or "\u043f\u0440\u043e\u0438\u0437\u0432\u043e\u0434\u0438\u0442\u0435\u043b\u044c\u043d\u043e\u0441\u0442\u044c" in button.text().lower()
                                 for button in w.findChildren(QPushButton)))

    def test_static_paused_field_is_not_repainted_on_every_timer_tick(self):
        w = self.window
        w.running = False
        w.animation_frame()
        with patch.object(w, "render_particle_field") as render:
            for _ in range(5):
                w.animation_frame()
        render.assert_not_called()

    def test_paused_edits_repaint_the_field_once(self):
        w = self.window
        w.running = False
        for action in (w.demon_recolor, lambda: w.demon_colors_changed(3), w.clear_potential, w.new_potential):
            w.animation_frame()
            action()
            with patch.object(w, "render_particle_field") as render:
                w.animation_frame()
                w.animation_frame()
            self.assertEqual(render.call_count, 1)

    def test_compression_updates_the_position_control_without_resetting_particles(self):
        w = self.window
        w.particles_spin.setValue(1000)
        w.barrier_combo.setCurrentIndex(1)
        w.sim.positions[:, 0] = 0.3
        before = w.sim.positions.copy()
        w._update_parameter_limits()
        self.assertGreater(w.barrier_position_spin.minimum(), 0.1)
        w.barrier_position_spin.setValue(0.1)
        w.running = False
        w.animation_frame()
        lower, _ = barrier_motion_limits(w.sim)
        self.assertGreaterEqual(w.sim.barrier_goal + 0.001, lower)
        np.testing.assert_array_equal(w.sim.positions, before)

    def test_scatterer_controls_expose_the_actual_geometry(self):
        w = self.window
        w.scatter_size_spin.setValue(4)
        w.scatter_count_spin.setValue(8)
        self.assertEqual(w.scatter_size_spin.maximum(), 8)
        w.barrier_combo.setCurrentIndex(3)
        validate_parameters(w.sim)


if __name__ == "__main__":
    unittest.main()
