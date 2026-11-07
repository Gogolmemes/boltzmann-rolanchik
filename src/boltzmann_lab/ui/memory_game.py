"""Presentation-friendly UI for the concealed initial-condition experiment."""
import math
import numpy as np
import pygame
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget,
)
from boltzmann_lab.core.config import DT, FPS, SUBSTEPS, FIELD_WIDTH, FIELD_HEIGHT, WORLD_W
from boltzmann_lab.core.memory_game import MemoryGame, preset_potential
from boltzmann_lab.core.simulation import TemperatureLimitExceeded
from boltzmann_lab.ui.renderer import ParticleRendererMixin
from boltzmann_lab.ui.widgets import ParticleView, PotentialPlot, PresentationPlot, PlotLayout, fit_window_to_screen
from boltzmann_lab.utils.resources import load_translation


class MemoryGameWindow(ParticleRendererMixin, QMainWindow):
    menuRequested = pyqtSignal()

    def __init__(self, language="ru", back_menu_key="main_menu"):
        super().__init__()
        self.language = language
        self.back_menu_key = back_menu_key
        self.game = MemoryGame()
        self.sim = self.game.sim
        self.potential_defined = True
        self._bg_surface = None
        self.particle_draw_radius_px = 4
        self.surface = pygame.Surface((FIELD_WIDTH, FIELD_HEIGHT), depth=32)
        self._well_labels = []
        self._build_ui()
        self.set_language(language)
        self.preset_changed(0)
        fit_window_to_screen(self)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animation_frame)
        self.timer.start(round(1000 / FPS))

    def tr(self, key):
        return load_translation(self.language)["memory_game"][key]

    def _header(self):
        header = QToolButton()
        header.setCheckable(True)
        header.setChecked(True)
        header.setArrowType(Qt.ArrowType.DownArrow)
        header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        header.setStyleSheet("font-size: 22pt; font-weight: 800;")
        return header

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        root.addWidget(self.splitter)
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setMinimumWidth(300)
        controls = QWidget()
        left = QVBoxLayout(controls)
        self.controls_scroll.setWidget(controls)
        self.splitter.addWidget(self.controls_scroll)
        self.menu_button = QPushButton()
        self.menu_button.clicked.connect(self.menuRequested.emit)
        left.addWidget(self.menu_button)
        language_row = QHBoxLayout()
        self.language_label = QLabel()
        self.language_combo = QComboBox()
        self.language_combo.addItems(["Русский", "English"])
        self.language_combo.currentIndexChanged.connect(lambda i: self.set_language("ru" if i == 0 else "en"))
        language_row.addWidget(self.language_label)
        language_row.addWidget(self.language_combo)
        left.addLayout(language_row)
        self.instructions = QLabel()
        self.instructions.setWordWrap(True)
        left.addWidget(self.instructions)
        self.conditions_label = QLabel()
        self.conditions_label.setWordWrap(True)
        self.conditions_label.setStyleSheet("background: #dbeafe; padding: 12px; border-radius: 8px;")
        left.addWidget(self.conditions_label)
        self.config_group = QGroupBox()
        form = QFormLayout(self.config_group)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.preset_label = QLabel()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["", "", "", ""])
        form.addRow(self.preset_label, self.preset_combo)
        self.draw_button, self.apply_button = QPushButton(), QPushButton()
        form.addRow(self.draw_button)
        form.addRow(self.apply_button)
        self.origin_label = QLabel()
        self.origin_combo = QComboBox()
        form.addRow(self.origin_label, self.origin_combo)
        self.origin_hint = QLabel()
        self.origin_hint.setWordWrap(True)
        form.addRow(self.origin_hint)
        self.temperature_label = QLabel()
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.01, 10)
        self.temperature_spin.setDecimals(2)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(0.3)
        form.addRow(self.temperature_label, self.temperature_spin)
        self.height_label = QLabel()
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.1, 12)
        self.height_spin.setSingleStep(0.25)
        self.height_spin.setValue(1)
        form.addRow(self.height_label, self.height_spin)
        self.delay_label = QLabel()
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.1, 120)
        self.delay_spin.setSingleStep(1)
        self.delay_spin.setValue(5)
        form.addRow(self.delay_label, self.delay_spin)
        self.count_label = QLabel()
        self.count_spin = QSpinBox()
        self.count_spin.setRange(10, 100)
        self.count_spin.setValue(60)
        form.addRow(self.count_label, self.count_spin)
        left.addWidget(self.config_group)
        self.start_button = QPushButton()
        self.start_button.setStyleSheet("background: #1e40af; color: white; font-weight: 800;")
        self.start_button.clicked.connect(self.start_round)
        speed_row = QFormLayout()
        self.speed_label = QLabel()
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.25, 8)
        self.speed_spin.setValue(1)
        self.speed_spin.setSingleStep(0.25)
        self.speed_spin.setSuffix(" ×")
        speed_row.addRow(self.speed_label, self.speed_spin)
        left.addLayout(speed_row)
        self.score_label = QLabel()
        self.score_label.setWordWrap(True)
        self.score_label.setStyleSheet("background: #e2e8f0; padding: 12px; border: 2px solid #64748b; border-radius: 8px;")
        left.insertWidget(2, self.score_label)
        self.reset_score_button = QPushButton()
        self.reset_score_button.clicked.connect(self.reset_score)
        left.addWidget(self.reset_score_button)
        left.addStretch(1)
        right = QWidget()
        right_layout = PlotLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        self.stage_label = QLabel()
        self.stage_label.setWordWrap(True)
        self.stage_label.setStyleSheet("font-size: 22pt; font-weight: 800; background: #dbeafe; padding: 12px; border-radius: 8px;")
        right_layout.addWidget(self.stage_label)
        right_layout.addWidget(self.start_button)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        right_layout.addWidget(self.progress)
        self.potential_title = self._header()
        right_layout.addWidget(self.potential_title)
        self.potential_plot = PotentialPlot()
        self.potential_plot.setMinimumSize(160, 180)
        right_layout.addWidget(self.potential_plot, 2)
        self.well_markers = pg.ScatterPlotItem(size=13, brush=pg.mkBrush("#fbbf24"), pen=pg.mkPen("#ffffff", width=2))
        self.potential_plot.addItem(self.well_markers)
        self.particle_view = ParticleView()
        self.particle_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.particle_view.setWordWrap(True)
        self.particle_view.setMinimumSize(160, 180)
        self.particle_view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.particle_view.setStyleSheet("background: #0b1220; color: #ffffff; padding: 16px; font-size: 22pt; font-weight: 800;")
        right_layout.addWidget(self.particle_view, 3)
        self.answer_panel = QWidget()
        self.answer_layout = QGridLayout(self.answer_panel)
        self.answer_buttons = []
        right_layout.addWidget(self.answer_panel)
        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        right_layout.addWidget(self.result_label)
        self.reveal_initial = QCheckBox()
        self.reveal_initial.setChecked(True)
        self.reveal_initial.toggled.connect(self.update_distribution)
        right_layout.addWidget(self.reveal_initial)
        self.distribution_title = self._header()
        right_layout.addWidget(self.distribution_title)
        self.distribution_plot = PresentationPlot()
        self.distribution_plot.setMinimumSize(160, 180)
        self.distribution_plot.setMouseEnabled(x=False, y=False)
        self.distribution_plot.setMenuEnabled(False)
        self.distribution_plot.showGrid(x=True, y=True, alpha=0.2)
        self.distribution_plot.setXRange(0, 1, padding=0)
        self.distribution_plot.getAxis("left").setStyle(showValues=False)
        self.histogram = pg.BarGraphItem(x=[], height=[], width=0.04, brush=pg.mkBrush(56, 189, 248, 180), pen=pg.mkPen("#38bdf8"))
        self.distribution_plot.addItem(self.histogram)
        self.initial_curve = self.distribution_plot.plot([], [], pen=pg.mkPen("#fbbf24", width=4, style=Qt.PenStyle.DashLine))
        right_layout.addWidget(self.distribution_plot, 2)
        self.history_title = self._header()
        self.history_title.setChecked(False)
        self.history_title.setArrowType(Qt.ArrowType.RightArrow)
        right_layout.addWidget(self.history_title)
        self.history_table = QTableWidget(0, 5)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setMinimumHeight(180)
        self.history_table.setMaximumHeight(320)
        self.history_table.hide()
        right_layout.addWidget(self.history_table)
        self.plots_scroll = QScrollArea()
        self.plots_scroll.setWidgetResizable(True)
        self.plots_scroll.setMinimumWidth(180)
        self.plots_scroll.setWidget(right)
        self.splitter.addWidget(self.plots_scroll)
        self.splitter.setStretchFactor(1, 1)
        self._splitter_initialized = False
        self.preset_combo.currentIndexChanged.connect(self.preset_changed)
        self.draw_button.clicked.connect(self.begin_drawing)
        self.apply_button.clicked.connect(self.apply_drawing)
        self.origin_combo.currentIndexChanged.connect(self.update_well_markers)
        self.potential_plot.positionSelected.connect(self.select_well)
        self.potential_title.toggled.connect(lambda value: self._toggle(self.potential_title, self.potential_plot, value))
        self.distribution_title.toggled.connect(lambda _: self.update_view())
        self.history_title.toggled.connect(lambda value: self._toggle(self.history_title, self.history_table, value))
        for label in controls.findChildren(QLabel):
            label.setWordWrap(True)


    def showEvent(self, event):
        super().showEvent(event)
        if not self._splitter_initialized:
            width = self.splitter.width()
            controls_width = min(580, max(360, int(width * 0.32)))
            self.splitter.setSizes([controls_width, max(180, width - controls_width)])
            self._splitter_initialized = True

    def _toggle(self, header, widget, visible):
        widget.setVisible(visible)
        header.setArrowType(Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow)

    def well_name(self, index):
        count = len(self.game.wells)
        if count == 2:
            key = "left_well" if index == 0 else "right_well"
        elif count == 3:
            key = ("left_well", "middle_well", "right_well")[index]
        else:
            return self.tr("well_number").format(number=index + 1)
        return f"{index + 1}. {self.tr(key)}"

    def set_language(self, language):
        self.language = language
        self.setWindowTitle(self.tr("title"))
        self.language_combo.blockSignals(True)
        self.language_combo.setCurrentIndex(0 if language == "ru" else 1)
        self.language_combo.blockSignals(False)
        self.menu_button.setText(load_translation(language)["menu"][self.back_menu_key])
        self.language_label.setText(load_translation(language)["menu"]["language"])
        for widget, key in ((self.instructions, "instructions"), (self.config_group, "setup"),
                            (self.preset_label, "potential"), (self.draw_button, "draw"),
                            (self.apply_button, "apply"), (self.origin_label, "origin"),
                            (self.origin_hint, "origin_hint"), (self.temperature_label, "temperature"),
                            (self.height_label, "height"), (self.delay_label, "delay"),
                            (self.count_label, "count"), (self.speed_label, "speed"),
                            (self.reset_score_button, "reset_score"), (self.potential_title, "potential"),
                            (self.distribution_title, "distribution"), (self.history_title, "history"),
                            (self.reveal_initial, "initial_overlay")):
            if isinstance(widget, QGroupBox):
                widget.setTitle(self.tr(key))
            else:
                widget.setText(self.tr(key))
        self.temperature_spin.setToolTip(self.tr("temperature_hint"))
        self.preset_combo.blockSignals(True)
        for i, key in enumerate(("two_wells", "three_wells", "four_wells", "custom")):
            self.preset_combo.setItemText(i, self.tr(key))
        self.preset_combo.blockSignals(False)
        self._populate_origins()
        for combo in (self.preset_combo, self.origin_combo):
            combo.ensurePolished()
            width = max((combo.fontMetrics().horizontalAdvance(combo.itemText(i)) for i in range(combo.count())), default=0)
            combo.setMinimumWidth(width + 64)
        self.update_history()
        self.update_view()

    def _populate_origins(self):
        previous = max(0, self.origin_combo.currentIndex())
        self.origin_combo.blockSignals(True)
        self.origin_combo.clear()
        self.origin_combo.addItem(self.tr("random_well"))
        for i in range(len(self.game.wells)):
            self.origin_combo.addItem(self.well_name(i))
        self.origin_combo.setCurrentIndex(min(previous, self.origin_combo.count() - 1))
        self.origin_combo.blockSignals(False)
        for button in self.answer_buttons:
            self.answer_layout.removeWidget(button)
            button.deleteLater()
        self.answer_buttons = []
        for i in range(len(self.game.wells)):
            button = QPushButton(self.well_name(i))
            button.clicked.connect(lambda _=False, index=i: self.submit_answer(index))
            self.answer_layout.addWidget(button, i // 2, i % 2)
            self.answer_buttons.append(button)

    def preset_changed(self, index):
        if self.game.phase in ("evolving", "guess"):
            return
        if index == 3:
            self.begin_drawing()
            return
        self.game.set_potential(preset_potential(index + 2))
        self._bg_surface = None
        self.potential_defined = True
        self.potential_plot.show_applied(self.sim.potential)
        self._populate_origins()
        self.update_view()

    def begin_drawing(self):
        if self.game.phase in ("evolving", "guess"):
            return
        self.game.clear_round()
        self.game.wells = []
        self.potential_defined = False
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(3)
        self.preset_combo.blockSignals(False)
        self.potential_title.setChecked(True)
        self.potential_plot.begin_new()
        self._populate_origins()
        self.update_view()

    def apply_drawing(self):
        potential = self.potential_plot.compile_potential()
        if potential is None:
            self.show_error("need_wells")
            return
        try:
            self.game.set_potential(potential)
        except ValueError as error:
            self.show_error(str(error))
            return
        self.potential_defined = True
        self._bg_surface = None
        self.potential_plot.show_applied(self.sim.potential)
        self._populate_origins()
        self.update_view()

    def select_well(self, fraction):
        if self.game.phase in ("setup", "result") and self.game.wells:
            self.origin_combo.setCurrentIndex(self.game.well_at(fraction) + 1)

    def update_well_markers(self, *_):
        for label in self._well_labels:
            self.potential_plot.removeItem(label)
        self._well_labels = []
        self.well_markers.clear()
        selected = self.origin_combo.currentIndex() - 1 if self.game.phase == "setup" else -1
        if self.game.phase == "result":
            selected = self.game.origin
        for i, well in enumerate(self.game.wells):
            y = float(np.interp(well.minimum, np.linspace(0, 1, len(self.sim.potential)), self.sim.potential))
            color = "#34d399" if i == selected else "#fbbf24"
            self.well_markers.addPoints([{ "pos": (well.minimum, y), "brush": color }])
            label = pg.TextItem(str(i + 1), color=color, anchor=(0.5, 1))
            label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
            label.setPos(well.minimum, min(0.98, y + 0.05))
            self.potential_plot.addItem(label)
            self._well_labels.append(label)

    def start_round(self):
        if self.game.phase not in ("setup", "result"):
            return
        origin = self.origin_combo.currentIndex() - 1
        try:
            self.game.start(self.temperature_spin.value(), self.delay_spin.value(),
                            self.count_spin.value(), self.height_spin.value(),
                            None if origin < 0 else origin)
        except ValueError as error:
            self.game.clear_round()
            self.update_view()
            self.show_error(str(error))
            return
        self._bg_surface = None
        self.update_view()

    def animation_frame(self):
        if self.game.phase == "evolving":
            steps = max(1, math.ceil(SUBSTEPS * self.speed_spin.value()))
            dt = DT * SUBSTEPS * self.speed_spin.value() / steps
            try:
                for _ in range(steps):
                    self.game.advance(dt)
                    if self.game.phase != "evolving":
                        self.update_view()
                        break
            except TemperatureLimitExceeded:
                self.game.clear_round()
                self.show_error("overheat")
                self.update_view()
        self.progress.setValue(round(1000 * self.game.elapsed / self.game.duration))
        self.progress.setFormat(self.tr("elapsed").format(elapsed=self.game.elapsed, duration=self.game.duration))
        if self.game.phase in ("guess", "result"):
            self.render_particle_field()

    def submit_answer(self, index):
        if self.game.phase != "guess":
            return
        self.game.answer(index)
        self.update_history()
        self.update_view()

    def reset_score(self):
        if self.game.phase in ("evolving", "guess"):
            return
        self.game.results.clear()
        self.game.played = self.game.correct = self.game.streak = 0
        self.game.chance_sum = 0.0
        self.game.clear_round()
        self.update_history()
        self.update_view()

    def update_distribution(self, *_):
        if self.game.phase not in ("guess", "result"):
            self.histogram.setOpts(x=[], height=[])
            self.initial_curve.setData([], [])
            return
        bins = 24
        centers = (np.arange(bins) + 0.5) / bins
        counts, _ = np.histogram(self.sim.positions[:, 0] / WORLD_W, bins=bins, range=(0, 1))
        density = counts / self.sim.n * bins
        self.histogram.setOpts(x=centers, height=density, width=0.85 / bins)
        maximum = max(1.0, float(density.max()))
        if self.game.phase == "result" and self.reveal_initial.isChecked():
            initial, _ = np.histogram(self.game.initial_positions[:, 0] / WORLD_W, bins=bins, range=(0, 1))
            initial_density = initial / self.sim.n * bins
            self.initial_curve.setData(centers, initial_density)
            maximum = max(maximum, float(initial_density.max()))
        else:
            self.initial_curve.setData([], [])
        self.distribution_plot.setYRange(0, maximum * 1.1, padding=0)

    def update_history(self):
        self.history_table.setHorizontalHeaderLabels([self.tr(k) for k in ("table_temperature", "table_delay", "table_wells", "table_answer", "table_result")])
        self.history_table.setRowCount(len(self.game.results))
        for row, result in enumerate(self.game.results):
            values = (f"{result.temperature:.2f}", f"{result.duration:.1f}", str(result.well_count),
                      str(result.guess + 1), self.tr("correct" if result.correct else "wrong"))
            for column, value in enumerate(values):
                self.history_table.setItem(row, column, QTableWidgetItem(value))
        self.history_table.resizeRowsToContents()

    def update_view(self):
        phase = self.game.phase
        locked = phase in ("evolving", "guess")
        self.config_group.setVisible(not locked)
        self.apply_button.setEnabled(not locked and self.potential_plot.drawing_enabled)
        self.instructions.setVisible(phase == "setup")
        self.conditions_label.setVisible(phase != "setup")
        self.conditions_label.setText(self.tr("conditions").format(
            temperature=self.sim.temperature, duration=self.game.duration, count=self.sim.n))
        self.reset_score_button.setEnabled(not locked)
        self.start_button.setVisible(not locked)
        self.start_button.setEnabled(not locked and bool(self.game.wells))
        self.start_button.setText(self.tr("next_round" if phase == "result" else "start"))
        self.stage_label.setText(self.tr({"setup": "setup_message", "evolving": "evolving_message",
                                         "guess": "guess_message", "result": "result_message"}[phase]))
        self.progress.setVisible(phase == "evolving")
        self.progress.setValue(round(1000 * self.game.elapsed / self.game.duration))
        self.progress.setFormat(self.tr("elapsed").format(elapsed=self.game.elapsed, duration=self.game.duration))
        self.answer_panel.setVisible(phase == "guess")
        for button in self.answer_buttons:
            button.setEnabled(phase == "guess")
        self.result_label.setVisible(phase == "result")
        self.reveal_initial.setVisible(phase == "result")
        self._toggle(self.distribution_title, self.distribution_plot,
                     self.distribution_title.isChecked() and phase in ("guess", "result"))
        self.distribution_title.setVisible(phase in ("guess", "result"))
        chance = 100 * self.game.chance_sum / self.game.played if self.game.played else 100 / max(1, len(self.game.wells))
        self.score_label.setText(self.tr("score").format(correct=self.game.correct, played=self.game.played,
                                                        streak=self.game.streak, chance=chance))
        if phase == "result":
            result = self.game.result
            self.result_label.setText(self.tr("result_detail").format(
                verdict=self.tr("correct" if result.correct else "wrong"),
                origin=self.well_name(result.origin), retained=result.retained, count=result.particle_count))
            color = "#166534" if result.correct else "#991b1b"
            self.result_label.setStyleSheet(f"font-weight: 800; color: {color}; background: #ffffff; padding: 12px; border: 2px solid {color}; border-radius: 8px;")
        if phase in ("setup", "evolving"):
            self.particle_view.clear()
            self.particle_view.setText(self.tr("concealed" if phase == "evolving" else "preview"))
        else:
            self.render_particle_field()
        self.update_well_markers()
        self.update_distribution()

    def show_error(self, code):
        translations = load_translation(self.language)["memory_game"]
        QMessageBox.information(self, self.tr("title"), translations.get("error_" + code, self.tr("error_need_wells")))

    def shutdown(self):
        if hasattr(self, "timer"):
            self.timer.stop()

    def closeEvent(self, event):
        self.shutdown()
        event.accept()

