import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication, QWidget, QStyle, QStyleOptionComboBox

from boltzmann_lab.core.config import DT, FPS
from boltzmann_lab.ui.windows import AppWindow, MainWindow
from boltzmann_lab.ui.theme import apply_presentation_theme, relative_luminance


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_presentation_theme(cls.app)

    def setUp(self):
        self.window = MainWindow()
        self.window.timer.stop()

    def tearDown(self):
        self.window.shutdown()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_playback_speed_changes_simulated_time_without_large_steps(self):
        for speed in (0.05, 0.25, 1.0, 1.3, 4.0, 8.0):
            with self.subTest(speed=speed):
                self.window.playback_speed_spin.setValue(speed)
                with patch.object(self.window.sim, "step") as step:
                    self.window.animation_frame()
                steps = [call.args[0] for call in step.call_args_list]
                self.assertAlmostEqual(sum(steps), speed / FPS)
                self.assertLessEqual(max(steps), DT + 1e-12)
        self.window.running = False
        with patch.object(self.window.sim, "step") as step:
            self.window.animation_frame()
        step.assert_not_called()

    def test_waiting_demon_stays_frozen_at_high_playback_speed(self):
        self.window.mode_combo.setCurrentIndex(1)
        self.window.sim.demon_queue = [0]
        self.window.playback_speed_spin.setValue(8)
        with patch.object(self.window.sim, "step") as step:
            self.window.animation_frame()
        step.assert_not_called()
        self.window.demon_coloring_combo.setCurrentIndex(1)
        self.window.demon_color_threshold_spin.setValue(2)
        self.assertEqual(self.window.sim.demon_queue, [0])

    def test_controls_and_translations(self):
        self.window.mode_combo.setCurrentIndex(1)
        self.window.demon_rules_combo.setCurrentIndex(1)
        self.window.demon_coloring_combo.setCurrentIndex(1)
        self.assertEqual(self.window.sim.demon_rule_mode, "speed")
        self.assertEqual(self.window.sim.demon_color_mode, "speed")
        self.assertTrue(self.window.rule_combos[0].isHidden())
        self.assertFalse(self.window.speed_rule_combos[0].isHidden())
        self.window.set_language("en")
        self.assertEqual(self.window.demon_rules_combo.currentText(), "By speed")
        self.window.demon_rules_combo.setCurrentIndex(0)
        self.assertFalse(self.window.rule_combos[0].isHidden())
        self.assertTrue(self.window.rule_combos[1].isHidden())
        self.assertFalse(self.window.rule_combos[2].isHidden())
        self.window.set_language("ru")
        self.assertEqual(self.window.demon_rules_combo.currentText(), "По цвету")

    def test_graph_text_contrast_tracks_background_and_survives_language_switch(self):
        self.assertGreaterEqual(self.app.font().pointSize(), 18)
        for language in ("ru", "en"):
            self.window.set_language(language)
            for background, expected in (("#ffffff", "#000000"), ("#0b1220", "#ffffff")):
                plot = self.window.distribution_plot
                plot.setBackground(background)
                for name in ("bottom", "left"):
                    axis = plot.getAxis(name)
                    self.assertEqual(axis.textPen().color().name(), expected)
                    self.assertEqual(axis.labelStyle["color"], expected)
                    self.assertGreaterEqual(axis.style["tickFont"].pointSize(), 16)
                self.assertIsNone(plot.getPlotItem().legend)
                a, b = relative_luminance(expected), relative_luminance(background)
                self.assertGreater((max(a, b) + 0.05) / (min(a, b) + 0.05), 7)

    def test_free_energy_history_collects_while_hidden_and_keeps_model_time(self):
        self.assertTrue(self.window.free_energy_plot.isHidden())
        self.assertEqual(self.window.free_energy_times, [0.0])
        expected_time = 0.0
        for speed in (0.25, 1.0, 4.0):
            self.window.playback_speed_spin.setValue(speed)
            for _ in range(20):
                self.window.animation_frame()
            expected_time += 20 * speed / FPS
            self.assertAlmostEqual(self.window.simulation_time, expected_time)
        self.assertGreater(len(self.window.free_energy_times), 2)
        times = list(self.window.free_energy_times)
        self.window.free_energy_title.setChecked(True)
        self.assertFalse(self.window.free_energy_plot.isHidden())
        self.assertNotIn("F =", self.window.stats_label.text())
        self.assertEqual(list(self.window.free_energy_curve.xData), times)
        self.assertEqual(list(self.window.free_energy_curve.yData), self.window.free_energy_values)
        for language in ("en", "ru"):
            self.window.set_language(language)
            self.assertNotIn("F =", self.window.stats_label.text())
        self.window.free_energy_title.setChecked(False)
        self.assertIn("F =", self.window.stats_label.text())
        self.assertEqual(self.window.free_energy_times, times)

    def test_free_energy_clock_stops_on_pause_and_demon_decision(self):
        self.window.animation_frame()
        before = self.window.simulation_time
        history = list(self.window.free_energy_times)
        self.window.running = False
        for _ in range(10):
            self.window.animation_frame()
        self.assertEqual(self.window.simulation_time, before)
        self.assertEqual(self.window.free_energy_times, history)
        self.window.mode_combo.setCurrentIndex(1)
        self.window.running = True
        self.window.sim.demon_queue = [0]
        before = self.window.simulation_time
        history = list(self.window.free_energy_times)
        for _ in range(10):
            self.window.animation_frame()
        self.assertEqual(self.window.simulation_time, before)
        self.assertEqual(self.window.free_energy_times, history)

    def test_free_energy_history_restarts_for_new_experiments(self):
        for restart in (self.window.reset_particles, self.window.apply_preset,
                        lambda: self.window.particles_spin.setValue(80),
                        lambda: self.window.energy_spin.setValue(2.0)):
            for _ in range(10):
                self.window.animation_frame()
            restart()
            self.assertEqual(self.window.simulation_time, 0.0)
            self.assertEqual(self.window.free_energy_times, [0.0])
            self.assertAlmostEqual(self.window.free_energy_values[0], self.window.sim.free_energy())
        self.window.clear_potential()
        self.assertEqual(self.window.free_energy_times, [])
        self.assertEqual(self.window.simulation_time, 0.0)

    def test_free_energy_history_stays_bounded_and_keeps_start(self):
        with patch.object(self.window, "FREE_ENERGY_MAX_POINTS", 8):
            for _ in range(100):
                self.window.simulation_time = self.window._next_free_energy_sample
                self.window._record_free_energy()
        times = self.window.free_energy_times
        self.assertLessEqual(len(times), 8)
        self.assertEqual(times[0], 0.0)
        self.assertEqual(times[-1], self.window.simulation_time)
        self.assertTrue(all(a < b for a, b in zip(times, times[1:])))

    def test_selected_direction_and_coloring_are_fully_visible(self):
        self.window.resize(1600, 1000)
        self.window.show()
        self.window.mode_combo.setCurrentIndex(1)
        self.window.demon_rules_combo.setCurrentIndex(1)
        self.window.demon_coloring_combo.setCurrentIndex(1)
        self.app.processEvents()
        for combo in (*self.window.speed_rule_combos, self.window.demon_rules_combo,
                      self.window.demon_coloring_combo):
            option = QStyleOptionComboBox()
            combo.initStyleOption(option)
            field = combo.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, option,
                                                QStyle.SubControl.SC_ComboBoxEditField, combo)
            self.assertLessEqual(combo.fontMetrics().horizontalAdvance(combo.currentText()), field.width())

    def test_collapsing_graph_preserves_potential_and_draft(self):
        potential = self.window.sim.potential.copy()
        self.window.potential_plot.raw_points = [(0.1, 0.2), (0.5, 0.7)]
        self.window.potential_title.setChecked(False)
        self.assertTrue(self.window.potential_plot.isHidden())
        self.window.potential_title.setChecked(True)
        self.assertFalse(self.window.potential_plot.isHidden())
        self.assertEqual(self.window.potential_plot.raw_points, [(0.1, 0.2), (0.5, 0.7)])
        self.assertTrue((self.window.sim.potential == potential).all())
        self.window.potential_title.setChecked(False)
        self.window.new_potential()
        self.assertTrue(self.window.potential_title.isChecked())

    def test_simulation_inside_application_window(self):
        # WebEngine needs a desktop; keep this test focused on the app layout.
        class DocumentPlaceholder(QWidget):
            menuRequested = pyqtSignal()

            def __init__(self, html):
                super().__init__()

            def set_language(self, language):
                pass

        with patch("boltzmann_lab.ui.windows.InfoPage", DocumentPlaceholder):
            shell = AppWindow()
        try:
            shell.resize(800, 600)
            shell.show()
            shell.open_simulation()
            shell.sim_window.timer.stop()
            self.app.processEvents()
            shell.sim_window.animation_frame()
            self.assertEqual(shell.width(), 800)
            self.assertEqual(shell.height(), 600)
            shell.sim_window.mode_combo.setCurrentIndex(1)
            shell.sim_window.demon_coloring_combo.setCurrentIndex(1)
            self.app.processEvents()
            self.assertEqual(shell.width(), 800)
            self.assertEqual(shell.height(), 600)
            shell.back_to_menu()
            self.assertIsNone(shell.sim_window)
        finally:
            shell.close()
            shell.deleteLater()
            self.app.processEvents()

    def test_small_window_and_barrier_drag_mapping_after_resize(self):
        self.window.show()
        for size in ((800, 600), (1280, 800), (1600, 1000), (800, 600)):
            self.window.resize(*size)
            self.app.processEvents()
            self.window.animation_frame()
            self.app.processEvents()
            self.assertLessEqual(self.window.width(), size[0])
            self.assertLessEqual(self.window.height(), size[1])
            plots = self.window.plots_scroll
            # Large presentation fonts can require scrolling on a small display.
            minimum = plots.widget().layout().heightForWidth(plots.widget().width())
            self.assertLessEqual(plots.widget().height(), max(minimum, plots.viewport().height()) + 2)
            view = self.window.particle_view
            pm = view.pixmap()
            self.assertLessEqual(pm.width(), view.width())
            self.assertLessEqual(pm.height(), view.height())
            offset = (view.width() - pm.width()) / 2
            event = QMouseEvent(
                QMouseEvent.Type.MouseMove, QPointF(offset + pm.width() * 0.3, view.height() / 2),
                Qt.MouseButton.NoButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            )
            fraction, _ = view._fraction(event)
            self.assertAlmostEqual(fraction, 0.3)
        self.assertTrue(self.window.stats_label.wordWrap())


if __name__ == "__main__":
    unittest.main()
