import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QWidget
from boltzmann_lab.ui.memory_game import MemoryGameWindow
from boltzmann_lab.ui.windows import AppWindow, MainWindow
from boltzmann_lab.ui.theme import apply_presentation_theme


class MemoryGameInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_presentation_theme(cls.app)

    def setUp(self):
        self.window = MemoryGameWindow()
        self.window.timer.stop()
        self.window.count_spin.setValue(20)
        self.window.delay_spin.setValue(0.1)

    def tearDown(self):
        self.window.shutdown()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def finish(self):
        for _ in range(100):
            if self.window.game.phase != "evolving":
                break
            self.window.animation_frame()
        self.assertEqual(self.window.game.phase, "guess")

    def test_no_initial_state_leaks_and_answer_is_revealed_after_guess(self):
        w = self.window
        self.assertTrue(w.particle_view.pixmap() is None or w.particle_view.pixmap().isNull())
        self.assertTrue(w.distribution_plot.isHidden())
        w.start_round()
        self.assertEqual(w.game.phase, "evolving")
        self.assertTrue(w.config_group.isHidden())
        self.assertTrue(w.distribution_plot.isHidden())
        self.assertTrue(w.answer_panel.isHidden())
        w.set_language("en")
        self.assertTrue(w.config_group.isHidden())
        self.assertTrue(w.distribution_plot.isHidden())
        self.assertTrue(w.result_label.isHidden())
        self.assertTrue(all(spot.brush().color().name() == "#fbbf24" for spot in w.well_markers.points()))
        self.assertTrue(w.reveal_initial.isHidden())
        self.assertTrue(w.particle_view.pixmap() is None or w.particle_view.pixmap().isNull())
        with patch.object(w, "render_particle_field") as render:
            w.animation_frame()
        render.assert_not_called()
        self.finish()
        self.assertTrue(w.config_group.isHidden())
        self.assertTrue(w.reveal_initial.isHidden())
        self.assertTrue(w.initial_curve.xData is None or len(w.initial_curve.xData) == 0)
        self.assertFalse(w.particle_view.pixmap().isNull())
        actual = w.game.origin
        w.submit_answer(actual)
        self.assertEqual(w.game.correct, 1)
        colors = [spot.brush().color().name() for spot in w.well_markers.points()]
        self.assertEqual(colors[actual], "#34d399")
        self.assertEqual(colors.count("#34d399"), 1)
        self.assertFalse(w.reveal_initial.isHidden())
        self.assertGreater(len(w.initial_curve.xData), 0)
        self.assertEqual(w.history_table.rowCount(), 1)
        w.submit_answer(actual)
        self.assertEqual(w.game.played, 1)

    def test_duration_uses_model_time_at_different_playback_speeds(self):
        for speed in (0.25, 1, 8):
            self.window.game.clear_round()
            self.window.update_view()
            self.window.speed_spin.setValue(speed)
            self.window.start_round()
            self.finish()
            self.assertEqual(self.window.game.elapsed, self.window.delay_spin.value())

    def test_host_can_select_by_plot_and_custom_drawing_can_be_applied(self):
        w = self.window
        w.select_well(0.8)
        self.assertEqual(w.origin_combo.currentIndex(), 2)
        w.start_round()
        self.assertEqual(w.game.origin, 1)
        self.finish()
        w.submit_answer(1)
        w.begin_drawing()
        self.assertTrue(w.potential_plot.drawing_enabled)
        self.assertFalse(w.start_button.isEnabled())
        x = np.linspace(0, 1, 300)
        w.potential_plot.raw_points = list(zip(x, np.cos(3 * np.pi * x) ** 2))
        w.apply_drawing()
        self.assertEqual(len(w.game.wells), 3)
        self.assertEqual(len(w.answer_buttons), 3)
        self.assertTrue(w.start_button.isEnabled())
        self.assertFalse(w.potential_plot.drawing_enabled)

    def test_invalid_drawing_is_rejected_without_starting_a_round(self):
        w = self.window
        w.begin_drawing()
        w.potential_plot.raw_points = [(0.0, 0.8), (0.5, 0.0), (1.0, 0.8)]
        with patch.object(w, "show_error") as error:
            w.apply_drawing()
        error.assert_called_once()
        self.assertEqual(w.game.phase, "setup")
        self.assertFalse(w.start_button.isEnabled())
        self.assertTrue(w.potential_plot.drawing_enabled)

    def test_language_and_collapsible_distribution_keep_round(self):
        w = self.window
        w.start_round()
        self.finish()
        snapshot = w.sim.positions.copy()
        w.distribution_title.setChecked(False)
        self.assertTrue(w.distribution_plot.isHidden())
        w.set_language("en")
        self.assertEqual(w.game.phase, "guess")
        self.assertEqual(w.answer_buttons[0].text(), "1. Left well")
        w.distribution_title.setChecked(True)
        self.assertFalse(w.distribution_plot.isHidden())
        self.assertTrue(np.array_equal(snapshot, w.sim.positions))
        w.submit_answer(w.game.origin)
        w.reset_score()
        self.assertEqual(w.game.played, 0)
        self.assertEqual(w.history_table.rowCount(), 0)

    def test_normal_distribution_collapses_without_stopping_simulation(self):
        normal = MainWindow()
        normal.timer.stop()
        try:
            normal.distribution_title.setChecked(False)
            self.assertTrue(normal.distribution_plot.isHidden())
            before = normal.simulation_time
            normal.animation_frame()
            self.assertGreater(normal.simulation_time, before)
            normal.set_language("en")
            self.assertTrue(normal.distribution_plot.isHidden())
            normal.distribution_title.setChecked(True)
            self.assertFalse(normal.distribution_plot.isHidden())
            self.assertIsNone(normal.distribution_plot.getPlotItem().legend)
        finally:
            normal.shutdown()
            normal.close()
            normal.deleteLater()

    def test_game_is_available_from_menu_and_stops_on_exit(self):
        class DocumentPlaceholder(QWidget):
            menuRequested = pyqtSignal()
            def __init__(self, html):
                super().__init__()
            def set_language(self, language):
                pass
        with patch("boltzmann_lab.ui.windows.InfoPage", DocumentPlaceholder):
            shell = AppWindow()
        try:
            self.assertNotIn("memory_game", shell.menu_buttons)
            shell.menu_buttons["simulation"].click()
            self.assertEqual(shell.stack.currentIndex(), shell.PAGE_MODES)
            self.assertIsNone(shell.game_window)
            self.assertIsNone(shell.sim_window)
            self.assertEqual(shell.mode_buttons["laboratory"].text(), "Исследование газа")
            shell.mode_buttons["memory_game"].click()
            game_window = shell.game_window
            game_window.set_language("en")
            self.assertTrue(game_window.timer.isActive())
            self.assertEqual(game_window.menu_button.text(), "← Mode selection")
            game_window.menu_button.click()
            self.assertIsNone(shell.game_window)
            self.assertFalse(game_window.timer.isActive())
            self.assertEqual(shell.stack.currentIndex(), shell.PAGE_MODES)
            self.assertEqual(shell.language, "en")
            self.assertEqual(shell.mode_buttons["laboratory"].text(), "Gas laboratory")
            shell.mode_buttons["laboratory"].click()
            normal_window = shell.sim_window
            self.assertEqual(normal_window.menu_button.text(), "← Mode selection")
            normal_window.menu_button.click()
            self.assertIsNone(shell.sim_window)
            self.assertFalse(normal_window.timer.isActive())
            self.assertEqual(shell.stack.currentIndex(), shell.PAGE_MODES)
            shell.modes_back_button.click()
            self.assertEqual(shell.stack.currentIndex(), shell.PAGE_MENU)
        finally:
            shell.close()
            shell.deleteLater()
