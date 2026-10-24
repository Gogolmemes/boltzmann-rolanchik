import numpy as np
import pygame
import pyqtgraph as pg

from PyQt6.QtCore import (
    Qt,
    QTimer,
    pyqtSignal,
    QUrl,
)

from PyQt6.QtGui import (
    QKeySequence,
    QShortcut,
    QPainter, 
    QPixmap, 
    QColor,
    QDesktopServices,
)

from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtWebEngineWidgets import QWebEngineView

from boltzmann_lab.core.simulation import (
    TemperatureLimitExceeded,
)

from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QStackedWidget,
)

from boltzmann_lab.core.config import (
    POTENTIAL_POINTS,
    HIST_BINS,
    FIELD_WIDTH,
    FIELD_HEIGHT,
    WORLD_H,
    WORLD_W,
    FPS,
    SUBSTEPS,
    DT,
    BARRIER_MIN,
    BARRIER_MAX,
    DEMON_COLORS,
    DEMON_HEX,
)

from boltzmann_lab.core.simulation import BoltzmannSimulation

from boltzmann_lab.ui.widgets import (
    ParticleView,
    PotentialPlot,
)

from boltzmann_lab.ui.renderer import ParticleRendererMixin

from boltzmann_lab.utils.resources import (
    load_translation,
    load_document,
    resource_path,
)

TRANSLATIONS = {
    language: load_translation(language)
    for language in ("ru", "en")
}

MENU_TEXT = {
    language: translations["menu"]
    for language, translations in TRANSLATIONS.items()
}

TEXT = {
    language: translations["simulation"]
    for language, translations in TRANSLATIONS.items()
}


class MainWindow(ParticleRendererMixin, QMainWindow):
    menuRequested = pyqtSignal()

    def __init__(self, language="ru"):
        super().__init__()

        self.language = language

        self.sim = BoltzmannSimulation()

        self.running = False
        self.potential_defined = False

        self._last_queue_len = 0
        self._bg_surface = None
        self.frame_counter = 0

        # Pygame используется как off-screen renderer.
        # Отдельного окна pygame нет.
        self.surface = pygame.Surface(
            (FIELD_WIDTH, FIELD_HEIGHT),
            depth=32,
        )

        # 1) сначала создаём виджеты и сигналы
        self._build_ui()
        self._connect_signals()

        # 2) только потом работаем с language_combo
        self.language_combo.blockSignals(True)
        self.language_combo.setCurrentIndex(0 if language == "ru" else 1)
        self.language_combo.blockSignals(False)

        self.set_language(language)

        self.new_potential()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animation_frame)
        self.timer.start(round(1000 / FPS))

    def barrier_dragged(self, fraction):
        fraction = float(np.clip(fraction, BARRIER_MIN, BARRIER_MAX))

        self.sim.barrier_goal = fraction

        # Синхронизируем спинбокс без повторного вызова parameters_changed.
        self.barrier_position_spin.blockSignals(True)
        self.barrier_position_spin.setValue(fraction)
        self.barrier_position_spin.blockSignals(False)

    # ========================================================
    # UI
    # ========================================================

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)

        # ----------------------------------------------------
        # ЛЕВАЯ ПАНЕЛЬ
        # ----------------------------------------------------

        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setMinimumWidth(300)
        self.controls_scroll.setMaximumWidth(370)

        controls = QWidget()
        self.controls_scroll.setWidget(controls)

        self.controls_layout = QVBoxLayout(controls)

        self.menu_button = QPushButton()
        self.menu_button.clicked.connect(self.menuRequested.emit)

        self.controls_layout.addWidget(self.menu_button)

        # Язык
        language_row = QHBoxLayout()

        self.language_label = QLabel()
        self.language_combo = QComboBox()
        self.language_combo.addItems(
            ["Русский", "English"]
        )

        language_row.addWidget(self.language_label)
        language_row.addWidget(self.language_combo)

        self.controls_layout.addLayout(language_row)

        mode_row = QHBoxLayout()

        self.mode_label = QLabel()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["", ""])

        mode_row.addWidget(self.mode_label)
        mode_row.addWidget(self.mode_combo)

        self.controls_layout.addLayout(mode_row)

        # ----------------------------------------------------
        # Потенциал
        # ----------------------------------------------------

        self.potential_group = QGroupBox()
        potential_layout = QVBoxLayout(
            self.potential_group
        )

        self.new_button = QPushButton()
        self.clear_button = QPushButton()
        self.apply_button = QPushButton()

        potential_layout.addWidget(self.new_button)
        potential_layout.addWidget(self.clear_button)
        potential_layout.addWidget(self.apply_button)

        self.preset_label = QLabel()
        self.preset_combo = QComboBox()

        for _ in range(5):
            self.preset_combo.addItem("")

        self.apply_preset_button = QPushButton()

        potential_layout.addWidget(self.preset_label)
        potential_layout.addWidget(self.preset_combo)
        potential_layout.addWidget(
            self.apply_preset_button
        )

        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(
            "color: #64748b; font-size: 11px;"
        )

        potential_layout.addWidget(self.hint_label)

        self.controls_layout.addWidget(
            self.potential_group
        )

        # ----------------------------------------------------
        # Симуляция
        # ----------------------------------------------------

        self.simulation_group = QGroupBox()

        simulation_layout = QVBoxLayout(
            self.simulation_group
        )

        button_row = QHBoxLayout()

        self.start_button = QPushButton()
        self.reset_button = QPushButton()

        button_row.addWidget(self.start_button)
        button_row.addWidget(self.reset_button)

        simulation_layout.addLayout(button_row)

        simulation_form = QFormLayout()

        self.particles_spin = QSpinBox()
        self.particles_spin.setRange(1, 10000)
        self.particles_spin.setValue(120)

        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(1, 5)
        self.radius_spin.setValue(4)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 10.00)
        self.temperature_spin.setDecimals(2)
        self.temperature_spin.setSingleStep(0.05)
        self.temperature_spin.setValue(0.80)

        self.energy_spin = QDoubleSpinBox()
        self.energy_spin.setRange(0.0, 12.0)
        self.energy_spin.setDecimals(2)
        self.energy_spin.setSingleStep(0.25)
        self.energy_spin.setValue(5.0)

        self.particles_label = QLabel()
        self.radius_label = QLabel()
        self.temperature_label = QLabel()
        self.energy_label = QLabel()

        simulation_form.addRow(
            self.particles_label,
            self.particles_spin,
        )

        simulation_form.addRow(
            self.radius_label,
            self.radius_spin,
        )

        simulation_form.addRow(
            self.temperature_label,
            self.temperature_spin,
        )

        simulation_form.addRow(
            self.energy_label,
            self.energy_spin,
        )

        simulation_layout.addLayout(simulation_form)

        self.controls_layout.addWidget(
            self.simulation_group
        )

        # ----------------------------------------------------
        # Перегородка
        # ----------------------------------------------------

        self.barrier_group = QGroupBox()

        barrier_form = QFormLayout(
            self.barrier_group
        )

        self.barrier_combo = QComboBox()
        self.barrier_combo.addItems(
            ["", "", "", "", ""]
        )

        self.barrier_position_spin = (
            QDoubleSpinBox()
        )

        self.barrier_position_spin.setRange(
            0.10,
            0.90,
        )
        self.barrier_position_spin.setDecimals(3)
        self.barrier_position_spin.setSingleStep(
            0.001
        )
        self.barrier_position_spin.setValue(0.50)

        self.pore_spin = QSpinBox()
        self.pore_spin.setRange(4, 100)
        self.pore_spin.setValue(26)

        self.barrier_type_label = QLabel()
        self.barrier_x_label = QLabel()
        self.pore_label = QLabel()

        self.scatter_shape_combo = QComboBox()
        self.scatter_shape_combo.addItems(["", "", "", "", "", ""])

        self.scatter_count_spin = QSpinBox()
        self.scatter_count_spin.setRange(1, 8)
        self.scatter_count_spin.setValue(3)

        self.scatter_size_spin = QSpinBox()
        self.scatter_size_spin.setRange(4, 40)
        self.scatter_size_spin.setValue(16)

        self.scatter_columns_spin = QSpinBox()
        self.scatter_columns_spin.setRange(1, 3)
        self.scatter_columns_spin.setValue(1)

        self.scatter_angle_spin = QSpinBox()
        self.scatter_angle_spin.setRange(0, 359)
        self.scatter_angle_spin.setSingleStep(15)
        self.scatter_angle_spin.setWrapping(True)

        self.p_lr_spin = QDoubleSpinBox()
        self.p_rl_spin = QDoubleSpinBox()

        for spin in (self.p_lr_spin, self.p_rl_spin):
            spin.setRange(0.0, 1.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.05)
            spin.setValue(0.50)

        self.scatter_shape_label = QLabel()
        self.scatter_count_label = QLabel()
        self.scatter_size_label = QLabel()
        self.scatter_columns_label = QLabel()
        self.scatter_angle_label = QLabel()
        self.p_lr_label = QLabel()
        self.p_rl_label = QLabel()

        self.scatter_widgets = [
            self.scatter_shape_label, self.scatter_shape_combo,
            self.scatter_count_label, self.scatter_count_spin,
            self.scatter_size_label, self.scatter_size_spin,
            self.scatter_columns_label, self.scatter_columns_spin,
            self.scatter_angle_label, self.scatter_angle_spin,
        ]

        self.semi_widgets = [
            self.p_lr_label, self.p_lr_spin,
            self.p_rl_label, self.p_rl_spin,
        ]

        barrier_form.addRow(
            self.barrier_type_label,
            self.barrier_combo,
        )

        barrier_form.addRow(
            self.barrier_x_label,
            self.barrier_position_spin,
        )

        barrier_form.addRow(
            self.pore_label,
            self.pore_spin,
        )

        barrier_form.addRow(self.scatter_shape_label, self.scatter_shape_combo)
        barrier_form.addRow(self.scatter_count_label, self.scatter_count_spin)
        barrier_form.addRow(self.scatter_size_label, self.scatter_size_spin)
        barrier_form.addRow(self.scatter_columns_label, self.scatter_columns_spin)
        barrier_form.addRow(self.scatter_angle_label, self.scatter_angle_spin)
        barrier_form.addRow(self.p_lr_label, self.p_lr_spin)
        barrier_form.addRow(self.p_rl_label, self.p_rl_spin)

        self.barrier_work_check = QCheckBox()
        self.barrier_work_check.setChecked(True)

        barrier_form.addRow(self.barrier_work_check)

        self.controls_layout.addWidget(
            self.barrier_group
        )

        self.demon_group = QGroupBox()
        demon_form = QFormLayout(self.demon_group)

        self.demon_colors_label = QLabel()
        self.demon_colors_spin = QSpinBox()
        self.demon_colors_spin.setRange(1, 3)
        self.demon_colors_spin.setValue(2)

        demon_form.addRow(self.demon_colors_label, self.demon_colors_spin)

        self.rule_labels = []
        self.rule_combos = []
        self.else_labels = []
        self.else_combos = []

        for k in range(3):
            label = QLabel()
            label.setStyleSheet(
                f"color: {DEMON_HEX[k]}; font-weight: bold;"
            )

            combo = QComboBox()
            combo.addItems(["", "", "", ""])

            else_label = QLabel()
            else_label.setStyleSheet(f"color: {DEMON_HEX[k]};")

            else_combo = QComboBox()
            else_combo.addItems(["", ""])

            demon_form.addRow(label, combo)
            demon_form.addRow(else_label, else_combo)

            self.rule_labels.append(label)
            self.rule_combos.append(combo)
            self.else_labels.append(else_label)
            self.else_combos.append(else_combo)

        self.recolor_button = QPushButton()
        demon_form.addRow(self.recolor_button)

        self.controls_layout.addWidget(self.demon_group)

        # Статус
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            """
            QLabel {
                background: #e2e8f0;
                color: #0f172a;
                border-radius: 6px;
                padding: 8px;
            }
            """
        )

        self.controls_layout.addWidget(
            self.status_label
        )

        self.controls_layout.addStretch(1)

        main_layout.addWidget(self.controls_scroll)

        # ----------------------------------------------------
        # ПРАВАЯ ЧАСТЬ
        # ----------------------------------------------------

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)

        # Потенциал
        self.potential_title = QLabel()
        self.potential_title.setStyleSheet(
            "font-size: 16px; font-weight: bold;"
        )

        right_layout.addWidget(
            self.potential_title
        )

        self.potential_plot = PotentialPlot()
        self.potential_plot.setMinimumHeight(180)

        right_layout.addWidget(
            self.potential_plot,
            2,
        )

        # Частицы
        self.particle_title = QLabel()
        self.particle_title.setStyleSheet(
            "font-size: 16px; font-weight: bold;"
        )

        right_layout.addWidget(
            self.particle_title
        )

        self.particle_view = ParticleView()
        self.particle_view.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.particle_view.setMinimumHeight(220)

        self.particle_view.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self.particle_view.setStyleSheet(
            """
            QLabel {
                background: #020617;
                border: 1px solid #334155;
            }
            """
        )

        right_layout.addWidget(
            self.particle_view,
            3,
        )

        self.demon_panel = QFrame()
        self.demon_panel.setObjectName("demonPanel")
        self.demon_panel.setStyleSheet(
            """
            #demonPanel {
                background: #fef3c7;
                border: 1px solid #f59e0b;
                border-radius: 6px;
            }
            """
        )

        demon_panel_layout = QHBoxLayout(self.demon_panel)

        self.demon_label = QLabel()
        self.demon_label.setStyleSheet("color: #0f172a; font-weight: bold;")
        self.demon_label.setWordWrap(True)

        self.demon_pass_button = QPushButton()
        self.demon_reflect_button = QPushButton()

        for button in (self.demon_pass_button, self.demon_reflect_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setMinimumWidth(130)

        demon_panel_layout.addWidget(self.demon_label, 1)
        demon_panel_layout.addWidget(self.demon_pass_button)
        demon_panel_layout.addWidget(self.demon_reflect_button)

        self.demon_panel.setVisible(False)

        right_layout.addWidget(self.demon_panel)

        # Горячие клавиши: → / Пробел пропускают, ← / Esc отражают.
        for key, value in (
            ("Right", True), ("Space", True),
            ("Left", False), ("Escape", False),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(
                lambda v=value: self.demon_decide(v)
            )

        # Распределение
        self.distribution_title = QLabel()
        self.distribution_title.setStyleSheet(
            "font-size: 16px; font-weight: bold;"
        )

        right_layout.addWidget(
            self.distribution_title
        )

        self.distribution_plot = pg.PlotWidget()
        self.distribution_plot.setBackground(
            "#0b1220"
        )

        self.distribution_plot.showGrid(
            x=True,
            y=True,
            alpha=0.20,
        )

        self.distribution_plot.setXRange(
            0.0,
            1.0,
            padding=0,
        )

        self.distribution_plot.setYRange(
            0.0,
            0.15,
            padding=0,
        )

        self.distribution_plot.setMouseEnabled(
            x=False,
            y=False,
        )

        self.distribution_plot.setMenuEnabled(
            False
        )

        centers = (
            np.arange(HIST_BINS) + 0.5
        ) / HIST_BINS

        self.histogram_bars = pg.BarGraphItem(
            x=centers,
            height=np.zeros(HIST_BINS),
            width=0.85 / HIST_BINS,
            pen=pg.mkPen("#38bdf8"),
            brush=pg.mkBrush(
                56,
                189,
                248,
                150,
            ),
        )

        self.distribution_plot.addItem(
            self.histogram_bars
        )

        self.theory_curve = (
            self.distribution_plot.plot(
                centers,
                np.zeros(HIST_BINS),
                pen=pg.mkPen(
                    "#fb7185",
                    width=3,
                ),
            )
        )

        self.legend = (
            self.distribution_plot.addLegend(
                offset=(10, 10)
            )
        )

        self.legend.addItem(
            self.histogram_bars,
            "",
        )

        self.legend.addItem(
            self.theory_curve,
            "",
        )

        right_layout.addWidget(
            self.distribution_plot,
            2,
        )

        # Статистика
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet(
            """
            QLabel {
                font-size: 13px;
                font-weight: bold;
                color: #0f172a;
                padding: 5px;
            }
            """
        )

        right_layout.addWidget(
            self.stats_label
        )

        main_layout.addWidget(
            right_widget,
            1,
        )

    # ========================================================
    # СИГНАЛЫ
    # ========================================================

    def _connect_signals(self):
        self.particle_view.barrierMoved.connect(
            self.barrier_dragged
        )
    
        self.language_combo.currentIndexChanged.connect(
            self.language_changed
        )

        self.new_button.clicked.connect(
            self.new_potential
        )

        self.clear_button.clicked.connect(
            self.clear_potential
        )

        self.apply_button.clicked.connect(
            self.apply_drawn_potential
        )

        self.apply_preset_button.clicked.connect(
            self.apply_preset
        )

        self.start_button.clicked.connect(
            self.toggle_running
        )

        self.reset_button.clicked.connect(
            self.reset_particles
        )

        self.particles_spin.valueChanged.connect(
            self.particle_count_changed
        )

        self.radius_spin.valueChanged.connect(
            self.radius_changed
        )

        self.temperature_spin.valueChanged.connect(
            self.parameters_changed
        )

        self.energy_spin.valueChanged.connect(
            self.parameters_changed
        )

        self.barrier_combo.currentIndexChanged.connect(
            self.parameters_changed
        )

        self.barrier_position_spin.valueChanged.connect(
            self.parameters_changed
        )

        self.pore_spin.valueChanged.connect(
            self.parameters_changed
        )

        self.scatter_shape_combo.currentIndexChanged.connect(self.parameters_changed)
        self.scatter_count_spin.valueChanged.connect(self.parameters_changed)
        self.scatter_size_spin.valueChanged.connect(self.parameters_changed)
        self.scatter_columns_spin.valueChanged.connect(self.parameters_changed)
        self.scatter_angle_spin.valueChanged.connect(self.parameters_changed)
        self.p_lr_spin.valueChanged.connect(self.parameters_changed)
        self.p_rl_spin.valueChanged.connect(self.parameters_changed)
        self.barrier_work_check.toggled.connect(self.parameters_changed)

        self.mode_combo.currentIndexChanged.connect(self.mode_changed)
        self.demon_colors_spin.valueChanged.connect(self.demon_colors_changed)
        self.recolor_button.clicked.connect(self.demon_recolor)

        for combo in self.rule_combos + self.else_combos:
            combo.currentIndexChanged.connect(self.parameters_changed)

        self.demon_pass_button.clicked.connect(
            lambda _=False: self.demon_decide(True)
        )
        self.demon_reflect_button.clicked.connect(
            lambda _=False: self.demon_decide(False)
        )

    # ========================================================
    # ПЕРЕВОД
    # ========================================================

    def tr(self, key):
        return TEXT[self.language][key]

    def language_changed(self, index):
        self.set_language(
            "ru" if index == 0 else "en"
        )

    def set_language(self, language):
        self.language = language

        self.setWindowTitle(
            self.tr("window")
        )

        self.language_label.setText(
            self.tr("language")
        )

        self.menu_button.setText(MENU_TEXT[language]["main_menu"])

        self.potential_group.setTitle(
            self.tr("potential_group")
        )

        self.new_button.setText(
            self.tr("new_potential")
        )

        self.clear_button.setText(
            self.tr("clear")
        )

        self.apply_button.setText(
            self.tr("apply")
        )

        self.preset_label.setText(
            self.tr("preset")
        )

        preset_names = [
            self.tr("preset_flat"),
            self.tr("preset_well"),
            self.tr("preset_double"),
            self.tr("preset_hill"),
            self.tr("preset_wave"),
        ]

        for i, name in enumerate(preset_names):
            self.preset_combo.setItemText(
                i,
                name,
            )

        self.apply_preset_button.setText(
            self.tr("apply_preset")
        )

        self.hint_label.setText(
            self.tr("hint")
        )

        self.simulation_group.setTitle(
            self.tr("simulation_group")
        )

        self.particles_label.setText(
            self.tr("particles")
        )

        self.radius_label.setText(
            self.tr("radius")
        )

        self.temperature_label.setText(
            self.tr("temperature")
        )

        self.energy_label.setText(
            self.tr("energy_scale")
        )

        self.reset_button.setText(
            self.tr("reset")
        )

        self.barrier_group.setTitle(
            self.tr("barrier_group")
        )

        self.barrier_type_label.setText(
            self.tr("barrier_type")
        )

        self.barrier_x_label.setText(
            self.tr("barrier_x")
        )

        self.pore_label.setText(
            self.tr("pore_size")
        )

        barrier_names = [
            self.tr("barrier_none"),
            self.tr("barrier_solid"),
            self.tr("barrier_porous"),
            self.tr("barrier_scatter"),
            self.tr("barrier_semi"),
        ]

        for i, name in enumerate(barrier_names):
            self.barrier_combo.setItemText(
                i,
                name,
            )

        self.potential_title.setText(
            self.tr("potential_title")
        )

        self.particle_title.setText(
            self.tr("particle_title")
        )

        self.distribution_title.setText(
            self.tr("distribution_title")
        )

        self.potential_plot.setLabel(
            "bottom",
            self.tr("x_axis"),
        )

        self.potential_plot.setLabel(
            "left",
            self.tr("u_axis"),
        )

        self.distribution_plot.setLabel(
            "bottom",
            self.tr("x_axis"),
        )

        self.distribution_plot.setLabel(
            "left",
            self.tr("probability"),
        )

        # Обновление легенды.
        self.legend.clear()

        self.legend.addItem(
            self.histogram_bars,
            self.tr("experimental"),
        )

        self.legend.addItem(
            self.theory_curve,
            self.tr("theoretical"),
        )

        shape_names = [
            self.tr("shape_circle"),
            self.tr("shape_ellipse"),
            self.tr("shape_triangle"),
            self.tr("shape_mixed"),
            self.tr("shape_grad_lr"),
            self.tr("shape_grad_rl"),
        ]

        for i, name in enumerate(shape_names):
            self.scatter_shape_combo.setItemText(i, name)

        self.scatter_shape_label.setText(self.tr("scatter_shape"))
        self.scatter_count_label.setText(self.tr("scatter_count"))
        self.scatter_size_label.setText(self.tr("scatter_size"))
        self.scatter_columns_label.setText(self.tr("scatter_columns"))
        self.scatter_angle_label.setText(self.tr("scatter_angle"))
        self.p_lr_label.setText(self.tr("p_lr"))
        self.p_rl_label.setText(self.tr("p_rl"))
        self.barrier_work_check.setText(self.tr("barrier_work"))

        self.mode_label.setText(self.tr("mode_label"))
        self.mode_combo.setItemText(0, self.tr("mode_normal"))
        self.mode_combo.setItemText(1, self.tr("mode_demon"))

        self.demon_group.setTitle(self.tr("demon_group"))
        self.demon_colors_label.setText(self.tr("demon_colors"))
        self.recolor_button.setText(self.tr("demon_recolor"))

        for k in range(3):
            self.rule_labels[k].setText(self.tr(f"demon_color_{k + 1}"))

        for combo in self.rule_combos:
            for i, key in enumerate(
                ("pass_never", "pass_lr", "pass_rl", "pass_all")
            ):
                combo.setItemText(i, self.tr(key))

        for combo in self.else_combos:
            for i, key in enumerate(("else_ask", "else_reflect")):
                combo.setItemText(i, self.tr(key))

        for label in self.else_labels:
            label.setText(self.tr("else_label"))

        self.demon_pass_button.setText(self.tr("demon_pass"))
        self.demon_reflect_button.setText(self.tr("demon_reflect"))

        self.update_barrier_controls()

        self.update_demon_panel()

        self.update_start_button()
        self.update_status()
        self.update_statistics()



    # ========================================================
    # ПОТЕНЦИАЛ
    # ========================================================

    def new_potential(self):
        self._bg_surface = None
        self.running = False
        self.potential_defined = False

        self.potential_plot.begin_new()

        self.histogram_bars.setOpts(
            height=np.zeros(HIST_BINS)
        )

        self.theory_curve.setData(
            [],
            [],
        )

        self.update_start_button()
        self.update_status()

    def clear_potential(self):
        self._bg_surface = None
        self.running = False
        self.potential_defined = False

        self.potential_plot.clear_drawing()
        self.potential_plot.drawing_enabled = True

        self.histogram_bars.setOpts(
            height=np.zeros(HIST_BINS)
        )

        self.theory_curve.setData(
            [],
            [],
        )

        self.update_start_button()
        self.update_status()

    def apply_drawn_potential(self):
        potential = (
            self.potential_plot.compile_potential()
        )

        if potential is None:
            QMessageBox.warning(
                self,
                self.tr("window"),
                self.tr("bad_drawing"),
            )
            return

        self.set_new_potential(potential)

    def apply_preset(self):
        x = np.linspace(
            0.0,
            1.0,
            POTENTIAL_POINTS,
        )

        index = self.preset_combo.currentIndex()

        if index == 0:
            potential = np.zeros_like(x)

        elif index == 1:
            # Одна потенциальная яма.
            potential = 4.0 * (x - 0.5) ** 2

        elif index == 2:
            # Две ямы.
            potential = (
                (x - 0.28)
                * (x - 0.72)
            ) ** 2

        elif index == 3:
            # Энергетический барьер.
            potential = np.exp(
                -((x - 0.5) / 0.13) ** 2
            )

        else:
            potential = (
                0.5
                + 0.5
                * np.cos(4.0 * np.pi * x)
            )

        minimum = np.min(potential)
        maximum = np.max(potential)

        if maximum - minimum > 1e-12:
            potential = (
                potential - minimum
            ) / (maximum - minimum)
        else:
            potential = np.zeros_like(
                potential
            )

        self.set_new_potential(potential)

    def set_new_potential(self, potential):
        """
        Потенциал меняется без мгновенного перемещения частиц.

        Благодаря этому видно, как старое распределение постепенно
        перестраивается в новое.
        """
        self._bg_surface = None
        self.sim.set_potential(potential)

        self.potential_defined = True

        self.potential_plot.show_applied(
            potential
        )

        self.running = True

        self.update_start_button()
        self.update_status()
        self.update_distribution()

    # ========================================================
    # ПАРАМЕТРЫ
    # ========================================================

    def parameters_changed(self, *_):
        self.sim.temperature = (
            self.temperature_spin.value()
        )

        self.sim.energy_scale = (
            self.energy_spin.value()
        )

        demon = self.mode_combo.currentIndex() == 1

        self.sim.barrier_mode = (
            5 if demon else self.barrier_combo.currentIndex()
        )

        self.sim.barrier_goal = (
            self.barrier_position_spin.value()
        )

        self.sim.pore_px = (
            self.pore_spin.value()
        )

        self.sim.scatter_shape = self.scatter_shape_combo.currentIndex()
        self.sim.scatter_count = self.scatter_count_spin.value()
        self.sim.scatter_size_px = self.scatter_size_spin.value()
        self.sim.scatter_columns = self.scatter_columns_spin.value()
        self.sim.scatter_angle = float(self.scatter_angle_spin.value())
        self.sim.p_lr = self.p_lr_spin.value()
        self.sim.p_rl = self.p_rl_spin.value()

        self.sim.barrier_does_work = self.barrier_work_check.isChecked()

        self.sim.demon_pass = [
            combo.currentIndex() for combo in self.rule_combos
        ]
        self.sim.demon_else = [
            combo.currentIndex() for combo in self.else_combos
        ]

        if demon:
            self.sim.barrier_fraction = float(
                np.clip(self.sim.barrier_goal, BARRIER_MIN, BARRIER_MAX)
            )
            self.sim.demon_assign_sides()
        else:
            self.sim.demon_queue.clear()

        self.update_barrier_controls()
        self.update_demon_panel()
        self.update_distribution()

    def update_barrier_controls(self):
        demon = self.mode_combo.currentIndex() == 1
        mode = self.barrier_combo.currentIndex()

        self.barrier_type_label.setVisible(not demon)
        self.barrier_combo.setVisible(not demon)

        self.barrier_x_label.setVisible(demon or mode != 0)
        self.barrier_position_spin.setVisible(demon or mode != 0)

        self.barrier_work_check.setVisible(not demon and mode != 0)

        self.pore_label.setVisible(not demon and mode == 2)
        self.pore_spin.setVisible(not demon and mode == 2)

        for widget in self.scatter_widgets:
            widget.setVisible(not demon and mode == 3)

        for widget in self.semi_widgets:
            widget.setVisible(not demon and mode == 4)

        self.demon_group.setVisible(demon)

        for k in range(3):
            visible = demon and k < self.demon_colors_spin.value()
            self.rule_labels[k].setVisible(visible)
            self.rule_combos[k].setVisible(visible)
            self.else_labels[k].setVisible(visible)
            self.else_combos[k].setVisible(visible)

    def particle_count_changed(self, value):
        self.sim.n = int(value)
        self.sim.reset_particles(
            equilibrium=self.potential_defined
        )

        self.update_distribution()

    def radius_changed(self, value):
        self.sim.set_radius_px(value)

        # При сильном увеличении размера безопаснее
        # немного перераспределить частицы.
        for _ in range(5):
            self.sim._resolve_collisions()

    def reset_particles(self):
        self.parameters_changed()

        self.sim.n = (
            self.particles_spin.value()
        )

        self.sim.reset_particles(
            equilibrium=self.potential_defined
        )

        self.update_distribution()

    # ========================================================
    # СТАРТ / ПАУЗА
    # ========================================================

    def toggle_running(self):
        if not self.potential_defined:
            QMessageBox.information(
                self,
                self.tr("window"),
                self.tr("no_potential"),
            )
            return

        self.running = not self.running

        self.update_start_button()
        self.update_status()

    def update_start_button(self):
        if self.running:
            self.start_button.setText(
                self.tr("pause")
            )
        else:
            self.start_button.setText(
                self.tr("start")
            )

    def update_status(self):
        if self.sim.demon_mode and self.sim.demon_queue:
            self.status_label.setText(self.tr("demon_wait"))
            return

        if not self.potential_defined:
            self.status_label.setText(
                self.tr("draw_status")
            )

        elif self.running:
            self.status_label.setText(
                self.tr("running_status")
            )

        else:
            self.status_label.setText(
                self.tr("paused_status")
            )

    # ========================================================
    # АНИМАЦИЯ
    # ========================================================

    def handle_temperature_limit(
        self,
        error: TemperatureLimitExceeded,
    ):
        self.running = False
        QMessageBox.warning(
            self,
            self.tr("temperature_limit_title"),
            self.tr("temperature_limit_message").format(
                temperature=error.temperature,
                limit=error.limit,
            ),
        )

        # Сбрасываем физическую модель
        self.sim.reset_particles()

        # Обновляем изображение
        self.render_particle_field()

    def animation_frame(self):
        if self.running and self.potential_defined:
            for _ in range(SUBSTEPS):
                # Симуляция стоит, пока есть нерешённая частица.
                if self.sim.demon_queue:
                    break
                try:
                    self.sim.step(DT)

                except TemperatureLimitExceeded as error:
                    self.handle_temperature_limit(error)
                    return
        else:
            self.sim.barrier_fraction = float(
                np.clip(self.sim.barrier_goal, BARRIER_MIN, BARRIER_MAX)
            )
            self.sim.barrier_velocity = 0.0

        queue_len = len(self.sim.demon_queue)

        if queue_len != self._last_queue_len:
            self._last_queue_len = queue_len
            self.update_demon_panel()

        self.particle_view.barrier_fraction = (
            self.sim.barrier_fraction
            if self.sim.barrier_mode not in (0, 5)
            else None
        )

        self.render_particle_field()

        self.frame_counter += 1

        if self.frame_counter % 5 == 0:
            self.update_distribution()
            self.update_statistics()

    # ========================================================
    # PYGAME-ОТРИСОВКА
    # ========================================================

    # ========================================================
    # РАСПРЕДЕЛЕНИЕ
    # ========================================================

    def update_distribution(self):
        if not self.potential_defined:
            return

        experimental, edges = (
            self.sim.histogram()
        )

        centers, theoretical = (
            self.sim.theoretical_distribution()
        )

        self.histogram_bars.setOpts(
            x=centers,
            height=experimental,
            width=0.85 / HIST_BINS,
        )

        self.theory_curve.setData(
            centers,
            theoretical,
        )

        maximum = max(
            float(
                np.max(experimental)
            ),
            float(
                np.max(theoretical)
            ),
            0.05,
        )

        self.distribution_plot.setYRange(
            0.0,
            maximum * 1.25,
            padding=0,
        )

    # ========================================================
    # СТАТИСТИКА
    # ========================================================

    def update_statistics(self):
        com = self.sim.center_of_mass()
        entropy = self.sim.entropy()
        temp = self.sim.effective_temperature()

        if self.sim.n > 0:
            left = int(
                np.sum(
                    self.sim.positions[:, 0]
                    < self.sim.barrier_x
                )
            )
        else:
            left = 0

        right = self.sim.n - left

        self.stats_label.setText(
            f"{self.tr('com')}: "
            f"x/L = {com:.3f}     |     "
            f"{self.tr('entropy')}: "
            f"S = {entropy:.4f}  (kB = 1)     |     "
            f"T = {temp:.4f}     |     "
            f"{self.tr('left_particles')}: "
            f"{left}     |     "
            f"{self.tr('right_particles')}: "
            f"{right}"
        )

    # ========================================================
    # ЗАКРЫТИЕ
    # ========================================================

    def shutdown(self):
        if hasattr(self, "timer"):
            self.timer.stop()

    def closeEvent(self, event):
        if hasattr(self, "timer"):
            self.timer.stop()

        pygame.quit()

        event.accept()

    def mode_changed(self, index):
        if index == 1 and self.particles_spin.value() > 60:
            # Иначе удары о линию будут слишком частыми.
            self.particles_spin.setValue(40)

        self.parameters_changed()

    def demon_colors_changed(self, value):
        self.sim.color_count = int(value)
        self.sim.recolor()

        self.update_barrier_controls()
        self.update_demon_panel()

    def demon_recolor(self):
        self.sim.recolor()
        self.update_demon_panel()

    def demon_decide(self, let_pass):
        if not (self.sim.demon_mode and self.sim.demon_queue):
            return

        self.sim.demon_resolve(let_pass)
        self.update_demon_panel()

    def update_demon_panel(self):
        queue = self.sim.demon_queue
        demon = self.sim.demon_mode
        waiting = demon and len(queue) > 0

        self.demon_panel.setVisible(demon)
        self.demon_pass_button.setEnabled(waiting)
        self.demon_reflect_button.setEnabled(waiting)

        if waiting:
            i = queue[0]
            c = int(self.sim.colors[i])
            speed = float(np.linalg.norm(self.sim.velocities[i]))

            direction = self.tr(
                "dir_lr" if self.sim.demon_side[i] == 0 else "dir_rl"
            )

            self.demon_label.setText(
                f"{self.tr('demon_particle')}: "
                f"<span style='color:{DEMON_HEX[c]}'>"
                f"{self.tr(f'demon_color_{c + 1}')}</span>, "
                f"|v| = {speed:.2f}, {direction} "
                f"({self.tr('demon_queue')}: {len(queue)})"
            )
        else:
            self.demon_label.setText(self.tr("demon_idle"))

        self.update_status()


class InfoPage(QWidget):
    """Страница с текстом и кнопкой «Главное меню»."""

    menuRequested = pyqtSignal()

    class ExternalLinksPage(QWebEnginePage):
        def acceptNavigationRequest(
            self,
            url,
            navigation_type,
            is_main_frame,
        ):
            if (
                navigation_type
                == QWebEnginePage.NavigationType.NavigationTypeLinkClicked
                and url.scheme() in ("http", "https")
            ):
                QDesktopServices.openUrl(url)
                return False

            return super().acceptNavigationRequest(
                url,
                navigation_type,
                is_main_frame,
            )

    def __init__(self, html_by_language):
        super().__init__()

        self.html_by_language = html_by_language

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        top = QHBoxLayout()

        self.menu_button = QPushButton()
        self.menu_button.setMinimumWidth(160)
        self.menu_button.clicked.connect(self.menuRequested.emit)

        top.addWidget(self.menu_button)
        top.addStretch(1)

        layout.addLayout(top)

        self.browser = QWebEngineView()
        self.browser.setPage(self.ExternalLinksPage(self.browser))
        images_path = resource_path("images").resolve()
        self.base_url = (
            QUrl.fromLocalFile(str(images_path) + "/")
        )
        self.browser.setStyleSheet(
            """
            QWebEngineView {
                border: 1px solid #cbd5e1;
                border-radius: 6px;
            }
            """
        )

        layout.addWidget(self.browser, 1)

    def set_language(self, language):
        self.menu_button.setText(MENU_TEXT[language]["main_menu"])
        self.browser.setHtml(
            self.html_by_language[language],
            self.base_url,
        )

class BackgroundPage(QWidget):
    """Страница, которая рисует картинку на фоне."""

    def __init__(self, image_path, overlay_alpha=90):
        super().__init__()

        self.pixmap = QPixmap(str(image_path))
        self.overlay_alpha = overlay_alpha  # 0..255, затемнение для читаемости

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if not self.pixmap.isNull():
            # Масштабируем с заполнением (cover), обрезая лишнее по краям.
            scaled = self.pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (scaled.width() - self.width()) // 2
            y = (scaled.height() - self.height()) // 2
            painter.drawPixmap(0, 0, scaled, x, y, self.width(), self.height())

            # Полупрозрачное затемнение, чтобы белый/тёмный текст читался.
            painter.fillRect(self.rect(), QColor(0, 0, 0, self.overlay_alpha))
        else:
            painter.fillRect(self.rect(), QColor("#f8fafc"))

        painter.end()

class AppWindow(QMainWindow):
    """Оболочка: главное меню, разделы и симуляция."""

    PAGE_MENU = 0
    PAGE_INSTRUCTION = 1
    PAGE_AUTHORSHIP = 2
    PAGE_THEORY = 3

    def __init__(self):
        super().__init__()

        self.language = "ru"
        self.sim_window = None

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self._build_menu_page()

        self.pages = [
            InfoPage({
                language: load_document(
                    language,
                    "instructions",
                )
                for language in ("ru", "en")
            }),

            InfoPage({
                language: load_document(
                    language,
                    "authorship",
                )
                for language in ("ru", "en")
            }),

            InfoPage({
                language: load_document(
                    language,
                    "theory",
                )
                for language in ("ru", "en")
            }),
        ]

        for page in self.pages:
            page.menuRequested.connect(self.back_to_menu)
            self.stack.addWidget(page)

        self.set_language("ru")
        self.stack.setCurrentIndex(self.PAGE_MENU)

        self.resize(1280, 900)
        self.setMinimumSize(950, 650)

    # --------------------------------------------------------

    def _build_menu_page(self):
        page = BackgroundPage(resource_path("images", "background.png"))

        outer = QVBoxLayout(page)
        outer.setContentsMargins(40, 40, 40, 40)

        outer.addStretch(2)

        self.menu_title = QLabel()
        self.menu_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.menu_title.setStyleSheet(
            "font-size: 30px; font-weight: bold; color: #0f172a;"
        )

        self.menu_subtitle = QLabel()
        self.menu_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.menu_subtitle.setStyleSheet("font-size: 15px; color: #64748b;")

        outer.addWidget(self.menu_title)
        outer.addWidget(self.menu_subtitle)
        outer.addSpacing(30)

        button_style = """
            QPushButton {
                font-size: 17px;
                min-height: 48px;
                min-width: 280px;
                max-width: 280px;
                border-radius: 8px;
                background: #1e293b;
                color: white;
            }
            QPushButton:hover {
                background: #334155;
            }
        """

        self.menu_buttons = {}

        for key, handler in (
            ("instruction", lambda: self.open_page(self.PAGE_INSTRUCTION)),
            ("authorship", lambda: self.open_page(self.PAGE_AUTHORSHIP)),
            ("theory", lambda: self.open_page(self.PAGE_THEORY)),
            ("simulation", self.open_simulation),
        ):
            button = QPushButton()
            button.setStyleSheet(button_style)
            button.clicked.connect(lambda _=False, h=handler: h())

            row = QHBoxLayout()
            row.addStretch(1)
            row.addWidget(button)
            row.addStretch(1)

            outer.addLayout(row)

            self.menu_buttons[key] = button

        outer.addSpacing(20)

        language_row = QHBoxLayout()
        language_row.addStretch(1)

        self.menu_language_label = QLabel()
        self.menu_language_combo = QComboBox()
        self.menu_language_combo.addItems(["Русский", "English"])
        self.menu_language_combo.currentIndexChanged.connect(
            lambda index: self.set_language("ru" if index == 0 else "en")
        )

        language_row.addWidget(self.menu_language_label)
        language_row.addWidget(self.menu_language_combo)
        language_row.addStretch(1)

        outer.addLayout(language_row)
        outer.addStretch(3)

        self.stack.addWidget(page)

    def set_language(self, language):
        self.language = language

        text = MENU_TEXT[language]

        self.setWindowTitle(text["title"])
        self.menu_title.setText(text["title"])
        self.menu_subtitle.setText(text["subtitle"])
        self.menu_language_label.setText(text["language"])

        for key, button in self.menu_buttons.items():
            button.setText(text[key])

        for page in getattr(self, "pages", []):
            page.set_language(language)

        index = 0 if language == "ru" else 1

        if self.menu_language_combo.currentIndex() != index:
            self.menu_language_combo.blockSignals(True)
            self.menu_language_combo.setCurrentIndex(index)
            self.menu_language_combo.blockSignals(False)

    # --------------------------------------------------------

    def open_page(self, index):
        self.stack.setCurrentIndex(index)

    def open_simulation(self):
        if self.sim_window is not None:
            return

        # Каждый раз создаётся новая симуляция с настройками по умолчанию.
        self.sim_window = MainWindow(self.language)
        self.sim_window.menuRequested.connect(self.back_to_menu)

        self.stack.addWidget(self.sim_window)
        self.stack.setCurrentWidget(self.sim_window)

    def _destroy_simulation(self):
        if self.sim_window is None:
            return

        window = self.sim_window
        self.sim_window = None

        # Язык, выбранный внутри симуляции, переносим в меню.
        language = window.language

        window.shutdown()

        self.stack.removeWidget(window)
        window.deleteLater()

        self.set_language(language)

    def back_to_menu(self):
        self._destroy_simulation()
        self.stack.setCurrentIndex(self.PAGE_MENU)

    def closeEvent(self, event):
        self._destroy_simulation()
        pygame.quit()
        event.accept()