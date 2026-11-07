import math  # Проверка чисел и корень для числа бинов
import numpy as np  # Импорт математики
import pygame  # Импорт pygame для рисования в памяти
import pyqtgraph as pg  # Импорт графиков PyQtGraph

from PyQt6.QtCore import (  # Импорт базовых классов PyQt
    Qt, QTimer, pyqtSignal, QUrl,
)

from PyQt6.QtGui import (  # Импорт графических элементов
    QKeySequence, QShortcut, QPainter, QPixmap, QColor, QDesktopServices,
)

from PyQt6.QtWebEngineCore import QWebEnginePage  # Для открытия ссылок в браузере
from PyQt6.QtWebEngineWidgets import QWebEngineView  # Для встроенного браузера (теория)

from boltzmann_lab.core.simulation import (  # Импорт ошибки перегрева
    TemperatureLimitExceeded,
)

from PyQt6.QtWidgets import (  # Импорт виджетов интерфейса
    QApplication, QComboBox, QDoubleSpinBox, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QVBoxLayout, QWidget, QCheckBox,
    QStackedWidget, QSplitter, QToolButton,
)

from boltzmann_lab.core.config import (  # Импорт констант
    POTENTIAL_POINTS, FIELD_WIDTH, FIELD_HEIGHT,
    WORLD_H, WORLD_W, FPS, SUBSTEPS, DT, BARRIER_MIN, BARRIER_MAX,
    BINS_MIN, BINS_MAX, MIN_PARTICLES_FOR_HIST, THEORY_POINTS,
    START_PARTICLES, START_TEMPERATURE, START_PRESET,
)

from boltzmann_lab.core.simulation import BoltzmannSimulation  # Импорт симуляции

from boltzmann_lab.ui.widgets import (  # Импорт виджетов (график и поле)
    ParticleView, PotentialPlot, PlotLayout, PresentationPlot, fit_window_to_screen,
)

from boltzmann_lab.ui.theme import DEMON_TEXT_COLORS, presentation_document
from boltzmann_lab.ui.renderer import ParticleRendererMixin  # Импорт "художника"

from boltzmann_lab.utils.resources import (  # Импорт загрузки ресурсов
    load_translation, load_document, resource_path,
)

TRANSLATIONS = {  # Словари переводов
    language: load_translation(language)
    for language in ("ru", "en")
}

MENU_TEXT = {  # Тексты для меню
    language: translations["menu"]
    for language, translations in TRANSLATIONS.items()
}

TEXT = {  # Тексты для симуляции
    language: translations["simulation"]
    for language, translations in TRANSLATIONS.items()
}


def bin_count(n_particles):
    """
    Число бинов гистограммы: round(sqrt(N)) в пределах [BINS_MIN, BINS_MAX].

    Крайние случаи:
    - N не число, NaN, N < 1 -> BINS_MIN; +бесконечность -> BINS_MAX;
    - мало частиц (sqrt(N) < BINS_MIN) -> BINS_MIN, бинов меньше не делаем;
    - много частиц -> BINS_MAX, более тонкие бины теряются в шуме.
    """
    try:
        n = float(n_particles)
    except (TypeError, ValueError):
        return BINS_MIN
    if math.isnan(n) or n < 1.0:
        return BINS_MIN
    if math.isinf(n):
        return BINS_MAX
    return int(min(BINS_MAX, max(BINS_MIN, round(math.sqrt(n)))))


DIST_TEXT = {  # Тексты для нижнего графика и статистики
    "ru": {
        "bins": "бинов",
        "few": "Мало частиц: гистограмма грубая, увеличьте число частиц",
        "density": "Плотность<br>вероятности",
        "mean_u": "Uср",      # <-- Добавили
        "free_f": "F",        # <-- Добавили
    },
    "en": {
        "bins": "bins",
        "few": "Few particles: the histogram is coarse, increase the particle count",
        "density": "Probability<br>density",
        "mean_u": "Mean U",      # <-- Добавили
        "free_f": "F",        # <-- Добавили
    },
}

if BINS_MIN < 1 or BINS_MIN > BINS_MAX:
    raise ValueError("config: нужно 1 <= BINS_MIN <= BINS_MAX")


class MainWindow(ParticleRendererMixin, QMainWindow):
    FREE_ENERGY_SAMPLE_INTERVAL = 0.1
    FREE_ENERGY_MAX_POINTS = 8192

    menuRequested = pyqtSignal()  # Сигнал: "пользователь нажал Главное меню"

    def __init__(self, language="ru", back_menu_key="main_menu"):
        super().__init__()

        self.language = language  # Текущий язык
        self.back_menu_key = back_menu_key
        self.sim = BoltzmannSimulation()  # Создание физической симуляции
        self.bins = bin_count(self.sim.n)  # Число бинов гистограммы по числу частиц
        self.running = False  # Флаг: запущена ли анимация
        self.potential_defined = False  # Флаг: нарисован ли график энергии
        self._last_queue_len = 0  # Для отслеживания очереди Демона
        self._bg_surface = None  # Кэш для фона (потенциальной энергии)
        self.simulation_time = 0.0
        self.free_energy_times = []
        self.free_energy_values = []
        self._free_energy_sample_interval = self.FREE_ENERGY_SAMPLE_INTERVAL
        self._next_free_energy_sample = 0.0
        self._free_energy_dirty = False
        self.frame_counter = 0  # Счетчик кадров (для обновления графиков)

        # Pygame Surface для рисования в памяти
        self.surface = pygame.Surface((FIELD_WIDTH, FIELD_HEIGHT), depth=32)

        self._build_ui()  # Создание интерфейса (кнопки, графики)
        self._connect_signals()  # Подключение кнопок к действиям

        self.language_combo.blockSignals(True)  # Временно отключаем сигналы
        self.language_combo.setCurrentIndex(0 if language == "ru" else 1)  # Выбор языка
        self.language_combo.blockSignals(False)  # Включаем обратно

        self.set_language(language)  # Применение языка ко всем надписям
        fit_window_to_screen(self)
        self._start_default_state()  # Стартовое состояние: одна яма, движение

        self.timer = QTimer(self)  # Таймер для анимации
        self.timer.timeout.connect(self.animation_frame)  # Что делать по таймеру
        self.timer.start(round(1000 / FPS))  # Запуск с частотой FPS

    def barrier_dragged(self, fraction):  # Вызывается, когда мышку тащат по перегородке
        fraction = float(np.clip(fraction, BARRIER_MIN, BARRIER_MAX))  # Ограничение позиции
        self.sim.barrier_goal = fraction  # Запоминаем новую цель для перегородки

        self.barrier_position_spin.blockSignals(True)  # Обновляем спинбокс без вызова сигналов
        self.barrier_position_spin.setValue(fraction)
        self.barrier_position_spin.blockSignals(False)

    def _build_ui(self):  # Создание всего интерфейса (слева панель, справа графики)
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)  # Горизонтальное расположение
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)

        # ЛЕВАЯ ПАНЕЛЬ (прокручиваемая)
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setMinimumWidth(280)
        self.controls_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        controls = QWidget()
        self.controls_scroll.setWidget(controls)
        self.controls_layout = QVBoxLayout(controls)

        # Кнопка "Главное меню"
        self.menu_button = QPushButton()
        self.menu_button.clicked.connect(self.menuRequested.emit)
        self.controls_layout.addWidget(self.menu_button)

        # Выбор языка
        language_row = QHBoxLayout()
        self.language_label = QLabel()
        self.language_combo = QComboBox()
        self.language_combo.addItems(["Русский", "English"])
        language_row.addWidget(self.language_label)
        language_row.addWidget(self.language_combo)
        self.controls_layout.addLayout(language_row)

        # Выбор режима (Обычный / Демон)
        mode_row = QHBoxLayout()
        self.mode_label = QLabel()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["", ""])
        mode_row.addWidget(self.mode_label)
        mode_row.addWidget(self.mode_combo)
        self.controls_layout.addLayout(mode_row)

        # Группа "Потенциальная энергия"
        self.potential_group = QGroupBox()
        potential_layout = QVBoxLayout(self.potential_group)
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
        potential_layout.addWidget(self.apply_preset_button)

        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("color: #334155; font-weight: 600;")
        potential_layout.addWidget(self.hint_label)
        self.controls_layout.addWidget(self.potential_group)

        # Группа "Параметры моделирования"
        self.simulation_group = QGroupBox()
        simulation_layout = QVBoxLayout(self.simulation_group)
        button_row = QHBoxLayout()
        self.start_button = QPushButton()
        self.reset_button = QPushButton()
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.reset_button)
        simulation_layout.addLayout(button_row)

        simulation_form = QFormLayout()
        self.particles_spin = QSpinBox()
        self.particles_spin.setRange(1, 10000)
        self.particles_spin.setValue(START_PARTICLES)

        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(1, 5)
        self.radius_spin.setValue(4)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 10.00)
        self.temperature_spin.setDecimals(2)
        self.temperature_spin.setSingleStep(0.05)
        self.temperature_spin.setValue(START_TEMPERATURE)

        self.energy_spin = QDoubleSpinBox()
        self.energy_spin.setRange(0.0, 12.0)
        self.energy_spin.setDecimals(2)
        self.energy_spin.setSingleStep(0.25)
        self.energy_spin.setValue(5.0)

        self.particles_label = QLabel()
        self.radius_label = QLabel()
        self.temperature_label = QLabel()
        self.energy_label = QLabel()

        simulation_form.addRow(self.particles_label, self.particles_spin)
        simulation_form.addRow(self.radius_label, self.radius_spin)
        simulation_form.addRow(self.temperature_label, self.temperature_spin)
        simulation_form.addRow(self.energy_label, self.energy_spin)
        self.playback_speed_label = QLabel()
        self.playback_speed_spin = QDoubleSpinBox()
        self.playback_speed_spin.setRange(0.05, 8.0)
        self.playback_speed_spin.setDecimals(2)
        self.playback_speed_spin.setSingleStep(0.25)
        self.playback_speed_spin.setValue(1.0)
        self.playback_speed_spin.setSuffix(" ×")
        simulation_form.addRow(self.playback_speed_label, self.playback_speed_spin)
        simulation_layout.addLayout(simulation_form)
        self.controls_layout.addWidget(self.simulation_group)

        # Группа "Перегородка"
        self.barrier_group = QGroupBox()
        barrier_form = QFormLayout(self.barrier_group)
        self.barrier_combo = QComboBox()
        self.barrier_combo.addItems(["", "", "", "", ""])
        self.barrier_position_spin = QDoubleSpinBox()
        self.barrier_position_spin.setRange(0.10, 0.90)
        self.barrier_position_spin.setDecimals(3)
        self.barrier_position_spin.setSingleStep(0.001)
        self.barrier_position_spin.setValue(0.50)

        self.pore_spin = QSpinBox()
        self.pore_spin.setRange(4, 100)
        self.pore_spin.setValue(26)

        self.barrier_type_label = QLabel()
        self.barrier_x_label = QLabel()
        self.pore_label = QLabel()

        # Рассеиватели (круги/эллипсы/треугольники)
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

        # Полупроницаемая стенка
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

        self.scatter_widgets = [  # Список виджетов рассеивателей (для скрытия)
            self.scatter_shape_label, self.scatter_shape_combo,
            self.scatter_count_label, self.scatter_count_spin,
            self.scatter_size_label, self.scatter_size_spin,
            self.scatter_columns_label, self.scatter_columns_spin,
            self.scatter_angle_label, self.scatter_angle_spin,
        ]
        self.semi_widgets = [  # Список виджетов полупроницаемой стенки
            self.p_lr_label, self.p_lr_spin,
            self.p_rl_label, self.p_rl_spin,
        ]

        barrier_form.addRow(self.barrier_type_label, self.barrier_combo)
        barrier_form.addRow(self.barrier_x_label, self.barrier_position_spin)
        barrier_form.addRow(self.pore_label, self.pore_spin)
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
        self.controls_layout.addWidget(self.barrier_group)

        # Группа "Демон Максвелла"
        self.demon_group = QGroupBox()
        demon_form = QFormLayout(self.demon_group)
        self.demon_coloring_label = QLabel()
        self.demon_coloring_combo = QComboBox()
        self.demon_coloring_combo.addItems(["", ""])
        demon_form.addRow(self.demon_coloring_label, self.demon_coloring_combo)
        self.demon_color_threshold_label = QLabel()
        self.demon_color_threshold_spin = QDoubleSpinBox()
        self.demon_color_threshold_spin.setRange(0.0, 1000.0)
        self.demon_color_threshold_spin.setDecimals(3)
        self.demon_color_threshold_spin.setSingleStep(0.1)
        self.demon_color_threshold_spin.setValue(1.0)
        demon_form.addRow(self.demon_color_threshold_label, self.demon_color_threshold_spin)
        self.demon_color_legend = QLabel()
        self.demon_color_legend.setWordWrap(True)
        demon_form.addRow(self.demon_color_legend)
        self.demon_rules_label = QLabel()
        self.demon_rules_combo = QComboBox()
        self.demon_rules_combo.addItems(["", ""])
        demon_form.addRow(self.demon_rules_label, self.demon_rules_combo)
        self.demon_speed_threshold_label = QLabel()
        self.demon_speed_threshold_spin = QDoubleSpinBox()
        self.demon_speed_threshold_spin.setRange(0.0, 1000.0)
        self.demon_speed_threshold_spin.setDecimals(3)
        self.demon_speed_threshold_spin.setSingleStep(0.1)
        self.demon_speed_threshold_spin.setValue(1.0)
        demon_form.addRow(self.demon_speed_threshold_label, self.demon_speed_threshold_spin)
        self.speed_rule_labels = []
        self.speed_rule_combos = []
        self.speed_else_labels = []
        self.speed_else_combos = []
        for k in range(2):
            label, else_label = QLabel(), QLabel()
            combo, else_combo = QComboBox(), QComboBox()
            combo.addItems(["", "", "", ""])
            combo.setCurrentIndex(self.sim.demon_speed_pass[k])
            else_combo.addItems(["", ""])
            else_combo.setCurrentIndex(self.sim.demon_speed_else[k])
            demon_form.addRow(label, combo)
            demon_form.addRow(else_label, else_combo)
            self.speed_rule_labels.append(label)
            self.speed_rule_combos.append(combo)
            self.speed_else_labels.append(else_label)
            self.speed_else_combos.append(else_combo)
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
            label.setStyleSheet(f"color: {DEMON_TEXT_COLORS[k]}; font-weight: 800;")
            combo = QComboBox()
            combo.addItems(["", "", "", ""])
            else_label = QLabel()
            else_label.setStyleSheet(f"color: {DEMON_TEXT_COLORS[k]};")
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

        # Статус (внизу слева)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("""
            QLabel {
                background: #e2e8f0;
                color: #0f172a;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        self.controls_layout.addWidget(self.status_label)
        self.controls_layout.addStretch(1)  # Прижимает всё вверх
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(self.controls_scroll)
        main_layout.addWidget(self.main_splitter)

        # ПРАВАЯ ЧАСТЬ (графики и поле)
        right_widget = QWidget()
        right_layout = PlotLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)

        # Заголовок "Потенциальная энергия"
        self.potential_title = QToolButton()
        self.potential_title.setCheckable(True)
        self.potential_title.setChecked(True)
        self.potential_title.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.potential_title.setArrowType(Qt.ArrowType.DownArrow)
        self.potential_title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.potential_title.setStyleSheet("font-size: 22pt; font-weight: 800; color: #0f172a;")
        right_layout.addWidget(self.potential_title)

        # Сам график потенциала
        self.potential_plot = PotentialPlot()
        self.potential_plot.setMinimumSize(160, 120)
        right_layout.addWidget(self.potential_plot, 2)

        # Заголовок "Движение частиц"
        self.particle_title = QLabel()
        self.particle_title.setStyleSheet("font-size: 22pt; font-weight: 800; color: #0f172a;")
        right_layout.addWidget(self.particle_title)

        # Поле с частицами
        self.particle_view = ParticleView()
        self.particle_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.particle_view.setMinimumSize(160, 140)
        # Размер QLabel не должен зависеть от последнего отрисованного pixmap.
        self.particle_view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.particle_view.setStyleSheet("""
            QLabel {
                background: #020617;
                border: 1px solid #334155;
            }
        """)
        right_layout.addWidget(self.particle_view, 3)

        # Панель Демона (появляется в режиме Демона)
        self.demon_panel = QFrame()
        self.demon_panel.setObjectName("demonPanel")
        self.demon_panel.setStyleSheet("""
            #demonPanel {
                background: #fef3c7;
                border: 1px solid #f59e0b;
                border-radius: 6px;
            }
        """)
        demon_panel_layout = QVBoxLayout(self.demon_panel)
        self.demon_label = QLabel()
        self.demon_label.setStyleSheet("color: #0f172a; font-weight: bold;")
        self.demon_label.setWordWrap(True)
        self.demon_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.demon_pass_button = QPushButton()
        self.demon_reflect_button = QPushButton()
        for button in (self.demon_pass_button, self.demon_reflect_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setMinimumWidth(0)
        demon_panel_layout.addWidget(self.demon_label)
        demon_buttons = QHBoxLayout()
        demon_buttons.addWidget(self.demon_pass_button)
        demon_buttons.addWidget(self.demon_reflect_button)
        demon_panel_layout.addLayout(demon_buttons)
        self.demon_panel.setVisible(False)
        right_layout.addWidget(self.demon_panel)

        # Горячие клавиши для Демона (Пропустить / Отразить)
        for key, value in (
            ("Right", True), ("Space", True),
            ("Left", False), ("Escape", False),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda v=value: self.demon_decide(v))

        # Заголовок "Распределение вероятностей"
        self.distribution_title = QToolButton()
        self.distribution_title.setCheckable(True)
        self.distribution_title.setChecked(True)
        self.distribution_title.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.distribution_title.setArrowType(Qt.ArrowType.DownArrow)
        self.distribution_title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.distribution_title.setStyleSheet("font-size: 22pt; font-weight: 800; color: #0f172a;")
        right_layout.addWidget(self.distribution_title)

        # Нижний график (гистограмма + теория)
        self.distribution_plot = PresentationPlot()
        self.distribution_plot.setBackground("#0b1220")
        self.distribution_plot.setMinimumSize(160, 240)
        self.distribution_plot.getAxis("left").setStyle(showValues=False)
        self.distribution_plot.showGrid(x=True, y=True, alpha=0.20)
        self.distribution_plot.setXRange(0.0, 1.0, padding=0)
        self.distribution_plot.setYRange(0.0, 2.0, padding=0)
        self.distribution_plot.setMouseEnabled(x=False, y=False)
        self.distribution_plot.setMenuEnabled(False)

        centers = (np.arange(self.bins) + 0.5) / self.bins

        # Синие столбики (эксперимент)
        self.histogram_bars = pg.BarGraphItem(
            x=centers,
            height=np.zeros(self.bins),
            width=0.85 / self.bins,
            pen=pg.mkPen("#38bdf8"),
            brush=pg.mkBrush(56, 189, 248, 150),
        )
        self.distribution_plot.addItem(self.histogram_bars)

        # Красная линия (теория)
        self.theory_curve = self.distribution_plot.plot(
            centers,
            np.zeros(self.bins),
            pen=pg.mkPen("#fb7185", width=4),
        )

        right_layout.addWidget(self.distribution_plot, 2)

        # Статистика внизу (центр масс, энтропия, температура)
        self.stats_label = QLabel()
        self.stats_label.setWordWrap(True)
        self.stats_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.stats_label.setStyleSheet("""
            QLabel {
                font-size: 18pt;
                font-weight: 750;
                color: #0f172a;
                background: #e2e8f0;
                border: 2px solid #64748b;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        right_layout.addWidget(self.stats_label)

        self.free_energy_title = QToolButton()
        self.free_energy_title.setCheckable(True)
        self.free_energy_title.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.free_energy_title.setArrowType(Qt.ArrowType.RightArrow)
        self.free_energy_title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.free_energy_title.setStyleSheet("font-size: 22pt; font-weight: 800; color: #0f172a;")
        right_layout.addWidget(self.free_energy_title)

        self.free_energy_plot = PresentationPlot()
        self.free_energy_plot.setMinimumSize(160, 180)
        self.free_energy_plot.showGrid(x=True, y=True, alpha=0.20)
        self.free_energy_plot.setMouseEnabled(x=False, y=False)
        self.free_energy_plot.setMenuEnabled(False)
        self.free_energy_plot.setXRange(0.0, 1.0, padding=0)
        self.free_energy_curve = self.free_energy_plot.plot(
            [], [], pen=pg.mkPen("#fbbf24", width=4),
        )
        self.free_energy_current = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen("#0b1220", width=2), brush=pg.mkBrush("#fbbf24"),
        )
        self.free_energy_plot.addItem(self.free_energy_current)
        self.free_energy_plot.hide()
        right_layout.addWidget(self.free_energy_plot, 2)

        self.plots_scroll = QScrollArea()
        self.plots_scroll.setWidgetResizable(True)
        self.plots_scroll.setMinimumWidth(180)
        self.plots_scroll.setWidget(right_widget)
        self.main_splitter.addWidget(self.plots_scroll)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self._splitter_initialized = False
        for label in controls.findChildren(QLabel):
            label.setWordWrap(True)
        for combo in controls.findChildren(QComboBox):
            # Направление находится в конце строки и не должно обрезаться.
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        for form in self.findChildren(QFormLayout):
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._splitter_initialized:
            width = self.main_splitter.width()
            controls_width = min(580, max(360, int(width * 0.32)))
            self.main_splitter.setSizes([controls_width, max(180, width - controls_width)])
            self._splitter_initialized = True

    def toggle_potential_plot(self, expanded):
        self.potential_plot.setVisible(expanded)
        self.potential_title.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)

    def toggle_distribution_plot(self, expanded):
        self.distribution_plot.setVisible(expanded)
        self.distribution_title.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        if expanded:
            self.update_distribution()

    def toggle_free_energy_plot(self, expanded):
        self.free_energy_plot.setVisible(expanded)
        self.free_energy_title.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.update_statistics()
        if expanded:
            self._update_free_energy_plot()
            QTimer.singleShot(0, self._reveal_free_energy_plot)

    def _reveal_free_energy_plot(self):
        if self.free_energy_title.isChecked():
            self.plots_scroll.ensureWidgetVisible(self.free_energy_plot, 0, 12)

    def _reset_free_energy_history(self):
        self.simulation_time = 0.0
        self.free_energy_times.clear()
        self.free_energy_values.clear()
        self._free_energy_sample_interval = self.FREE_ENERGY_SAMPLE_INTERVAL
        self._next_free_energy_sample = 0.0
        self._free_energy_dirty = True
        self.free_energy_curve.setData([], [])
        self.free_energy_current.setData([], [])
        self.free_energy_plot.setXRange(0.0, 1.0, padding=0)
        if self.potential_defined:
            self._record_free_energy()
        self._update_free_energy_plot()
        self.update_statistics()

    def _record_free_energy(self):
        if self.simulation_time + 1e-12 < self._next_free_energy_sample:
            return
        value = self.sim.free_energy()
        if not np.isfinite(value):
            return
        self.free_energy_times.append(self.simulation_time)
        self.free_energy_values.append(value)
        # Сохраняем начало и всю длительность опыта без бесконечного роста памяти.
        if len(self.free_energy_times) >= self.FREE_ENERGY_MAX_POINTS:
            last_time, last_value = self.free_energy_times[-1], self.free_energy_values[-1]
            self.free_energy_times = self.free_energy_times[::2]
            self.free_energy_values = self.free_energy_values[::2]
            if self.free_energy_times[-1] != last_time:
                self.free_energy_times.append(last_time)
                self.free_energy_values.append(last_value)
            self._free_energy_sample_interval *= 2.0
        self._next_free_energy_sample = self.simulation_time + self._free_energy_sample_interval
        self._free_energy_dirty = True

    def _update_free_energy_plot(self):
        if not self.free_energy_title.isChecked() or not self._free_energy_dirty:
            return
        self.free_energy_curve.setData(self.free_energy_times, self.free_energy_values)
        if self.free_energy_times:
            self.free_energy_current.setData([self.free_energy_times[-1]], [self.free_energy_values[-1]])
            self.free_energy_plot.setXRange(0.0, max(1.0, self.free_energy_times[-1]), padding=0.02)
        self._free_energy_dirty = False

    def _connect_signals(self):  # Подключение всех кнопок и ползунков к функциям
        self.potential_title.toggled.connect(self.toggle_potential_plot)
        self.distribution_title.toggled.connect(self.toggle_distribution_plot)
        self.free_energy_title.toggled.connect(self.toggle_free_energy_plot)
        self.particle_view.barrierMoved.connect(self.barrier_dragged)
        self.language_combo.currentIndexChanged.connect(self.language_changed)
        self.new_button.clicked.connect(self.new_potential)
        self.clear_button.clicked.connect(self.clear_potential)
        self.apply_button.clicked.connect(self.apply_drawn_potential)
        self.apply_preset_button.clicked.connect(self.apply_preset)
        self.start_button.clicked.connect(self.toggle_running)
        self.reset_button.clicked.connect(self.reset_particles)
        self.particles_spin.valueChanged.connect(self.particle_count_changed)
        self.radius_spin.valueChanged.connect(self.radius_changed)
        self.temperature_spin.valueChanged.connect(self.parameters_changed)
        self.energy_spin.valueChanged.connect(self.parameters_changed)
        self.barrier_combo.currentIndexChanged.connect(self.parameters_changed)
        self.barrier_position_spin.valueChanged.connect(self.parameters_changed)
        self.pore_spin.valueChanged.connect(self.parameters_changed)
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
        self.demon_coloring_combo.currentIndexChanged.connect(self.parameters_changed)
        self.demon_rules_combo.currentIndexChanged.connect(self.parameters_changed)
        self.demon_color_threshold_spin.valueChanged.connect(self.parameters_changed)
        self.demon_speed_threshold_spin.valueChanged.connect(self.parameters_changed)
        for combo in self.rule_combos + self.else_combos + self.speed_rule_combos + self.speed_else_combos:
            combo.currentIndexChanged.connect(self.parameters_changed)
        self.demon_pass_button.clicked.connect(lambda _=False: self.demon_decide(True))
        self.demon_reflect_button.clicked.connect(lambda _=False: self.demon_decide(False))

    def tr(self, key):  # Перевод текста по ключу
        return TEXT[self.language][key]

    def language_changed(self, index):  # Смена языка через выпадающий список
        self.set_language("ru" if index == 0 else "en")

    def set_language(self, language):  # Установка всех надписей на выбранном языке
        self.language = language
        self.setWindowTitle(self.tr("window"))
        self.language_label.setText(self.tr("language"))
        self.menu_button.setText(MENU_TEXT[language][self.back_menu_key])
        self.potential_group.setTitle(self.tr("potential_group"))
        self.new_button.setText(self.tr("new_potential"))
        self.clear_button.setText(self.tr("clear"))
        self.apply_button.setText(self.tr("apply"))
        self.preset_label.setText(self.tr("preset"))
        preset_names = [
            self.tr("preset_flat"), self.tr("preset_well"),
            self.tr("preset_double"), self.tr("preset_hill"),
            self.tr("preset_wave"),
        ]
        for i, name in enumerate(preset_names):
            self.preset_combo.setItemText(i, name)
        self.apply_preset_button.setText(self.tr("apply_preset"))
        self.hint_label.setText(self.tr("hint"))
        self.simulation_group.setTitle(self.tr("simulation_group"))
        self.particles_label.setText(self.tr("particles"))
        self.radius_label.setText(self.tr("radius"))
        self.temperature_label.setText(self.tr("temperature"))
        self.energy_label.setText(self.tr("energy_scale"))
        self.playback_speed_label.setText(self.tr("playback_speed"))
        self.playback_speed_spin.setToolTip(self.tr("playback_speed_hint"))
        self.reset_button.setText(self.tr("reset"))
        self.barrier_group.setTitle(self.tr("barrier_group"))
        self.barrier_type_label.setText(self.tr("barrier_type"))
        self.barrier_x_label.setText(self.tr("barrier_x"))
        self.pore_label.setText(self.tr("pore_size"))
        barrier_names = [
            self.tr("barrier_none"), self.tr("barrier_solid"),
            self.tr("barrier_porous"), self.tr("barrier_scatter"),
            self.tr("barrier_semi"),
        ]
        for i, name in enumerate(barrier_names):
            self.barrier_combo.setItemText(i, name)
        self.potential_title.setText(self.tr("potential_title"))
        self.particle_title.setText(self.tr("particle_title"))
        self.distribution_title.setText(self.tr("distribution_title"))
        self.free_energy_title.setText(self.tr("free_energy_title"))
        self.free_energy_plot.setLabel("bottom", self.tr("simulation_time"))
        self.potential_plot.setLabel("bottom", "")
        self.potential_plot.setLabel("left", "")
        self.distribution_plot.setLabel("bottom", "")
        self.distribution_plot.setLabel("left", "")
        shape_names = [
            self.tr("shape_circle"), self.tr("shape_ellipse"),
            self.tr("shape_triangle"), self.tr("shape_mixed"),
            self.tr("shape_grad_lr"), self.tr("shape_grad_rl"),
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
        self.demon_coloring_label.setText(self.tr("demon_coloring"))
        self.demon_coloring_combo.setItemText(0, self.tr("color_random"))
        self.demon_coloring_combo.setItemText(1, self.tr("color_speed"))
        self.demon_rules_label.setText(self.tr("demon_rules"))
        self.demon_rules_combo.setItemText(0, self.tr("rules_color"))
        self.demon_rules_combo.setItemText(1, self.tr("rules_speed"))
        self.demon_color_threshold_label.setText(self.tr("color_threshold"))
        self.demon_speed_threshold_label.setText(self.tr("speed_threshold"))
        self.demon_color_legend.setText(self.tr("speed_color_legend"))
        self.demon_color_threshold_spin.setToolTip(self.tr("speed_threshold_hint"))
        self.demon_speed_threshold_spin.setToolTip(self.tr("speed_threshold_hint"))
        self.speed_rule_labels[0].setText(self.tr("speed_slow"))
        self.speed_rule_labels[1].setText(self.tr("speed_fast"))
        self.recolor_button.setText(self.tr("demon_recolor"))
        for k in range(3):
            self.rule_labels[k].setText(self.tr(f"demon_color_{k + 1}"))
        for combo in self.rule_combos + self.speed_rule_combos:
            for i, key in enumerate(("pass_never", "pass_lr", "pass_rl", "pass_all")):
                combo.setItemText(i, self.tr(key))
        for combo in self.else_combos + self.speed_else_combos:
            for i, key in enumerate(("else_ask", "else_reflect")):
                combo.setItemText(i, self.tr(key))
        for label in self.else_labels + self.speed_else_labels:
            label.setText(self.tr("else_label"))
        self.demon_pass_button.setText(self.tr("demon_pass"))
        self.demon_reflect_button.setText(self.tr("demon_reflect"))
        # QComboBox кеширует sizeHint до перевода пустых пунктов. Явный минимум
        # сохраняет конец надписи со стрелкой и заставляет QFormLayout переносить ряд.
        for combo in self.controls_scroll.widget().findChildren(QComboBox):
            combo.ensurePolished()
            metrics = combo.fontMetrics()
            text_width = max((metrics.horizontalAdvance(combo.itemText(i))
                              for i in range(combo.count())), default=0)
            combo.setMinimumWidth(text_width + 64)
        self.update_barrier_controls()
        self.update_demon_panel()
        self.update_start_button()
        self.update_status()
        self.update_statistics()

    def _start_default_state(self):  # Сразу после запуска: готовый график, частицы в движении
        self.parameters_changed()  # переносим температуру, энергию и т.д. в симуляцию
        self.sim.n = self.particles_spin.value()
        self.bins = bin_count(self.sim.n)
        self.preset_combo.setCurrentIndex(START_PRESET)
        self.apply_preset()  # ставит потенциал и запускает движение (running = True)
        self.sim.reset_particles(equilibrium=True)  # частицы сразу по распределению Больцмана
        self._reset_free_energy_history()
        self.update_distribution()
        self.update_statistics()

    def new_potential(self):  # Начать рисовать новый график с нуля
        self._bg_surface = None
        self.running = False
        self.potential_defined = False
        self.potential_title.setChecked(True)
        self.potential_plot.begin_new()
        self._reset_distribution_plot()
        self._reset_free_energy_history()
        self.update_start_button()
        self.update_status()

    def clear_potential(self):  # Полностью очистить график
        self._bg_surface = None
        self.running = False
        self.potential_defined = False
        self.potential_title.setChecked(True)
        self.potential_plot.clear_drawing()
        self.potential_plot.drawing_enabled = True
        self._reset_distribution_plot()
        self._reset_free_energy_history()
        self.update_start_button()
        self.update_status()

    def apply_drawn_potential(self):  # Применить нарисованный мышкой график
        potential = self.potential_plot.compile_potential()
        if potential is None:
            QMessageBox.warning(self, self.tr("window"), self.tr("bad_drawing"))
            return
        self.set_new_potential(potential)

    def apply_preset(self):  # Применить готовый график из списка
        x = np.linspace(0.0, 1.0, POTENTIAL_POINTS)
        index = self.preset_combo.currentIndex()

        if index == 0:
            potential = np.zeros_like(x)  # Плоский
        elif index == 1:
            potential = 4.0 * (x - 0.5) ** 2  # Одна яма
        elif index == 2:
            potential = ((x - 0.28) * (x - 0.72)) ** 2  # Две ямы
        elif index == 3:
            potential = np.exp(-((x - 0.5) / 0.13) ** 2)  # Барьер
        else:
            potential = 0.5 + 0.5 * np.cos(4.0 * np.pi * x)  # Волны

        minimum = np.min(potential)
        maximum = np.max(potential)
        if maximum - minimum > 1e-12:
            potential = (potential - minimum) / (maximum - minimum)
        else:
            potential = np.zeros_like(potential)
        self.set_new_potential(potential)

    def set_new_potential(self, potential):  # Установить новый график в симуляцию
        self._bg_surface = None  # Сбрасываем кэш фона
        self.sim.set_potential(potential)
        self.potential_defined = True
        self.potential_plot.show_applied(potential)
        self._reset_free_energy_history()
        self.running = True
        self.update_start_button()
        self.update_status()
        self.update_distribution()

    def parameters_changed(self, *_):  # Вызывается при изменении любого ползунка
        old_temperature = self.sim.temperature
        old_energy_scale = self.sim.energy_scale
        self.sim.temperature = self.temperature_spin.value()
        self.sim.energy_scale = self.energy_spin.value()
        demon = self.mode_combo.currentIndex() == 1
        old_mode = self.sim.barrier_mode
        old_barrier = self.sim.barrier_fraction
        self.sim.barrier_mode = 5 if demon else self.barrier_combo.currentIndex()
        self.sim.barrier_goal = self.barrier_position_spin.value()
        self.sim.pore_px = self.pore_spin.value()
        self.sim.scatter_shape = self.scatter_shape_combo.currentIndex()
        self.sim.scatter_count = self.scatter_count_spin.value()
        self.sim.scatter_size_px = self.scatter_size_spin.value()
        self.sim.scatter_columns = self.scatter_columns_spin.value()
        self.sim.scatter_angle = float(self.scatter_angle_spin.value())
        self.sim.p_lr = self.p_lr_spin.value()
        self.sim.p_rl = self.p_rl_spin.value()
        self.sim.barrier_does_work = self.barrier_work_check.isChecked()
        self.sim.demon_pass = [combo.currentIndex() for combo in self.rule_combos]
        self.sim.demon_else = [combo.currentIndex() for combo in self.else_combos]
        self.sim.demon_speed_pass = [combo.currentIndex() for combo in self.speed_rule_combos]
        self.sim.demon_speed_else = [combo.currentIndex() for combo in self.speed_else_combos]
        self.sim.demon_rule_mode = "speed" if self.demon_rules_combo.currentIndex() == 1 else "color"
        self.sim.demon_color_mode = "speed" if self.demon_coloring_combo.currentIndex() == 1 else "random"
        self.sim.demon_speed_threshold = self.demon_speed_threshold_spin.value()
        self.sim.demon_color_threshold = self.demon_color_threshold_spin.value()
        if demon:
            self.sim.barrier_fraction = float(np.clip(self.sim.barrier_goal, BARRIER_MIN, BARRIER_MAX))
            if old_mode != 5 or old_barrier != self.sim.barrier_fraction:
                self.sim.demon_assign_sides()
        else:
            self.sim.demon_queue.clear()
        self.update_barrier_controls()
        self.update_demon_panel()
        self.update_distribution()
        if old_temperature != self.sim.temperature or old_energy_scale != self.sim.energy_scale:
            self._reset_free_energy_history()

    def update_barrier_controls(self):  # Скрывает/показывает настройки в зависимости от режима
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
        speed_colors = self.demon_coloring_combo.currentIndex() == 1
        speed_rules = self.demon_rules_combo.currentIndex() == 1
        self.demon_colors_label.setVisible(demon and not speed_colors)
        self.demon_colors_spin.setVisible(demon and not speed_colors)
        self.recolor_button.setVisible(demon and not speed_colors)
        for widget in (self.demon_color_threshold_label, self.demon_color_threshold_spin, self.demon_color_legend):
            widget.setVisible(demon and speed_colors)
        for widget in (self.demon_speed_threshold_label, self.demon_speed_threshold_spin,
                       *self.speed_rule_labels, *self.speed_rule_combos,
                       *self.speed_else_labels, *self.speed_else_combos):
            widget.setVisible(demon and speed_rules)
        for k in range(3):
            active_color = k in (0, 2) if speed_colors else k < self.demon_colors_spin.value()
            visible = demon and not speed_rules and active_color
            self.rule_labels[k].setVisible(visible)
            self.rule_combos[k].setVisible(visible)
            self.else_labels[k].setVisible(visible)
            self.else_combos[k].setVisible(visible)

    def particle_count_changed(self, value):  # Изменение количества частиц
        self.sim.n = int(value)
        self.bins = bin_count(self.sim.n)
        self.sim.reset_particles(equilibrium=self.potential_defined)
        self._reset_free_energy_history()
        self.update_distribution()

    def radius_changed(self, value):  # Изменение радиуса частиц
        self.sim.set_radius_px(value)
        for _ in range(5):
            self.sim._resolve_collisions()

    def reset_particles(self):  # Кнопка "Распределить заново"
        self.parameters_changed()
        self.sim.n = self.particles_spin.value()
        self.sim.reset_particles(equilibrium=self.potential_defined)
        self._reset_free_energy_history()
        self.update_distribution()

    def toggle_running(self):  # Кнопка "Старт/Пауза"
        if not self.potential_defined:
            QMessageBox.information(self, self.tr("window"), self.tr("no_potential"))
            return
        self.running = not self.running
        self.update_start_button()
        self.update_status()

    def update_start_button(self):  # Меняет текст на кнопке Старт/Пауза
        if self.running:
            self.start_button.setText(self.tr("pause"))
        else:
            self.start_button.setText(self.tr("start"))

    def update_status(self):  # Обновляет текст статуса в левом нижнем углу
        if self.sim.demon_mode and self.sim.demon_queue:
            self.status_label.setText(self.tr("demon_wait"))
            return
        if not self.potential_defined:
            self.status_label.setText(self.tr("draw_status"))
        elif self.running:
            self.status_label.setText(self.tr("running_status"))
        else:
            self.status_label.setText(self.tr("paused_status"))

    def handle_temperature_limit(self, error: TemperatureLimitExceeded):  # Обработка перегрева
        self.running = False
        QMessageBox.warning(
            self,
            self.tr("temperature_limit_title"),
            self.tr("temperature_limit_message").format(
                temperature=error.temperature,
                limit=error.limit,
            ),
        )
        self.sim.reset_particles()
        self._reset_free_energy_history()
        self.render_particle_field()

    def animation_frame(self):  # Главный цикл анимации (вызывается 60 раз в секунду)
        if self.running and self.potential_defined:
            # Темп меняет модельное время за кадр; шаг интегратора не превышает DT.
            speed = self.playback_speed_spin.value()
            steps = max(1, math.ceil(SUBSTEPS * speed))
            step_dt = DT * SUBSTEPS * speed / steps
            for _ in range(steps):
                if self.sim.demon_queue:
                    break
                try:
                    self.sim.step(step_dt)
                    self.simulation_time += step_dt
                    self._record_free_energy()
                except TemperatureLimitExceeded as error:
                    self.handle_temperature_limit(error)
                    return
        else:
            self.sim.barrier_fraction = float(np.clip(self.sim.barrier_goal, BARRIER_MIN, BARRIER_MAX))
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
        self.render_particle_field()  # Рисуем поле с частицами
        self.frame_counter += 1
        if self.frame_counter % 5 == 0:  # Обновляем графики не каждый кадр, а реже (для скорости)
            self.update_distribution()
            self.update_statistics()
            self._update_free_energy_plot()

    def update_distribution(self):  # Обновление нижнего графика (гистограмма и теория)
        if not self.potential_defined:
            return
        bins = self.bins
        experimental, _ = self.sim.histogram(bins=bins)
        centers = (np.arange(bins) + 0.5) / bins

        # Теория считается на плотной сетке THEORY_POINTS, а не на центрах бинов,
        # поэтому кривая гладкая при любом числе бинов гистограммы.
        theory_x, theory_p = self.sim.theoretical_distribution(bins=THEORY_POINTS)

        # Вероятность в бине делим на ширину бина (1/bins): график — плотность,
        # и его масштаб не меняется при смене числа бинов.
        density_exp = experimental * bins
        density_th = theory_p * THEORY_POINTS

        self.histogram_bars.setOpts(
            x=centers,
            height=density_exp,
            width=0.85 / bins,
        )
        self.theory_curve.setData(theory_x, density_th)
        maximum = max(
            float(np.max(density_exp)),
            float(np.max(density_th)),
            0.5,
        )
        self.distribution_plot.setYRange(0.0, maximum * 1.25, padding=0)

    def _reset_distribution_plot(self):  # Пустая гистограмма нужного размера
        centers = (np.arange(self.bins) + 0.5) / self.bins
        self.histogram_bars.setOpts(
            x=centers,
            height=np.zeros(self.bins),
            width=0.85 / self.bins,
        )
        self.theory_curve.setData([], [])

    def update_statistics(self):  # Обновление текста внизу (центр масс, энтропия, температура)
        com = self.sim.center_of_mass()
        p, _ = self.sim.histogram(bins=self.bins)
        p = p[p > 0.0]
        entropy = float(-np.sum(p * np.log(p))) if len(p) else 0.0
        temp = self.sim.effective_temperature()

        mean_u = self.sim.mean_potential_energy()
        free_f = self.sim.free_energy()

        text_pack = DIST_TEXT[self.language]
        text = (
            f"{self.tr('com')}: {com:.3f} | "
            f"{self.tr('entropy')}: {entropy:.4f} | "
            f"T = {temp:.4f} | "
            f"{text_pack['mean_u']} = {mean_u:.3f}"
        )
        if not self.free_energy_title.isChecked():
            text += f" | {text_pack['free_f']} = {free_f:.3f}"
        if self.sim.n < MIN_PARTICLES_FOR_HIST:
            text += f"\n{text_pack['few']}"
        self.stats_label.setText(text)

    def shutdown(self):  # Остановка таймера
        if hasattr(self, "timer"):
            self.timer.stop()

    def closeEvent(self, event):  # При закрытии окна
        if hasattr(self, "timer"):
            self.timer.stop()
        pygame.quit()
        event.accept()

    def mode_changed(self, index):  # Смена режима (Обычный / Демон)
        if index == 1 and self.particles_spin.value() > 60:
            self.particles_spin.setValue(40)  # Уменьшаем количество частиц для Демона
        self.parameters_changed()

    def demon_colors_changed(self, value):  # Изменение количества цветов у Демона
        self.sim.color_count = int(value)
        self.sim.recolor()
        self.update_barrier_controls()
        self.update_demon_panel()

    def demon_recolor(self):  # Перекрасить частицы заново
        self.sim.recolor()
        self.update_demon_panel()

    def demon_decide(self, let_pass):  # Решение Демона (Пропустить/Отразить)
        if not (self.sim.demon_mode and self.sim.demon_queue):
            return
        self.sim.demon_resolve(let_pass)
        self.update_demon_panel()

    def update_demon_panel(self):  # Обновление панели Демона (какая частица ждет)
        queue = self.sim.demon_queue
        demon = self.sim.demon_mode
        waiting = demon and len(queue) > 0
        self.demon_panel.setVisible(demon)
        self.demon_pass_button.setEnabled(waiting)
        self.demon_reflect_button.setEnabled(waiting)
        if waiting:
            i = queue[0]
            c = int(self.sim.particle_color_indices()[i])
            speed = float(np.linalg.norm(self.sim.velocities[i]))
            direction = self.tr("dir_lr" if self.sim.demon_side[i] == 0 else "dir_rl")
            self.demon_label.setText(
                f"{self.tr('demon_particle')}: "
                f"<span style='color:{DEMON_TEXT_COLORS[c]}'>{self.tr(f'demon_color_{c + 1}')}</span>, "
                f"|v| = {speed:.2f}, {direction} "
                f"({self.tr('demon_queue')}: {len(queue)})"
            )
        else:
            self.demon_label.setText(self.tr("demon_idle"))
        self.update_status()


class InfoPage(QWidget):  # Класс для страниц с текстом (Теория, Инструкция, Авторы)
    menuRequested = pyqtSignal()

    class ExternalLinksPage(QWebEnginePage):  # Внутренний класс для открытия ссылок в браузере
        def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
            if (navigation_type == QWebEnginePage.NavigationType.NavigationTypeLinkClicked
                    and url.scheme() in ("http", "https")):
                QDesktopServices.openUrl(url)
                return False
            return super().acceptNavigationRequest(url, navigation_type, is_main_frame)

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
        self.browser = QWebEngineView()  # Встроенный браузер для HTML
        self.browser.setPage(self.ExternalLinksPage(self.browser))
        images_path = resource_path("images").resolve()
        self.base_url = QUrl.fromLocalFile(str(images_path) + "/")
        self.browser.setStyleSheet("""
            QWebEngineView {
                border: 1px solid #cbd5e1;
                border-radius: 6px;
            }
        """)
        layout.addWidget(self.browser, 1)

    def set_language(self, language):  # Установка языка для страницы
        self.menu_button.setText(MENU_TEXT[language]["main_menu"])
        self.browser.setHtml(presentation_document(self.html_by_language[language]), self.base_url)


class BackgroundPage(QWidget):  # Класс для рисования фона в главном меню
    def __init__(self, image_path, overlay_alpha=90):
        super().__init__()
        self.pixmap = QPixmap(str(image_path))
        self.overlay_alpha = overlay_alpha

    def paintEvent(self, event):  # Рисует картинку на фоне
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if not self.pixmap.isNull():
            scaled = self.pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (scaled.width() - self.width()) // 2
            y = (scaled.height() - self.height()) // 2
            painter.drawPixmap(0, 0, scaled, x, y, self.width(), self.height())
            painter.fillRect(self.rect(), QColor(0, 0, 0, self.overlay_alpha))
        else:
            painter.fillRect(self.rect(), QColor("#f8fafc"))
        painter.end()


class AppWindow(QMainWindow):  # Главное окно приложения (оболочка)
    PAGE_MENU = 0
    PAGE_INSTRUCTION = 1
    PAGE_AUTHORSHIP = 2
    PAGE_THEORY = 3
    PAGE_MODES = 4

    def __init__(self):
        super().__init__()
        self.language = "ru"
        self.sim_window = None
        self.game_window = None
        self.stack = QStackedWidget()  # Стопка страниц (меню, справка, симуляция)
        self.setCentralWidget(self.stack)
        self._build_menu_page()
        self.pages = [
            InfoPage({language: load_document(language, "instructions") for language in ("ru", "en")}),
            InfoPage({language: load_document(language, "authorship") for language in ("ru", "en")}),
            InfoPage({language: load_document(language, "theory") for language in ("ru", "en")}),
        ]
        for page in self.pages:
            page.menuRequested.connect(self.back_to_menu)
            self.stack.addWidget(page)
        self._build_modes_page()
        self.set_language("ru")
        self.stack.setCurrentIndex(self.PAGE_MENU)
        fit_window_to_screen(self)

    def _build_menu_page(self):  # Создание страницы главного меню
        page = BackgroundPage(resource_path("images", "background.png"))
        outer = QVBoxLayout(page)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.addStretch(2)

        self.menu_title = QLabel()
        self.menu_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.menu_title.setWordWrap(True)
        self.menu_title.setStyleSheet("font-size: 32pt; font-weight: 800; color: #0f172a; background: #ffffff; padding: 16px; border-radius: 10px;")

        self.menu_subtitle = QLabel()
        self.menu_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.menu_subtitle.setStyleSheet("font-size: 20pt; font-weight: 700; color: #0f172a; background: #ffffff; padding: 10px; border-radius: 8px;")

        outer.addWidget(self.menu_title)
        outer.addWidget(self.menu_subtitle)
        outer.addSpacing(30)

        button_style = """
            QPushButton {
                font-size: 22pt;
                font-weight: 800;
                min-height: 56px;
                min-width: 300px;
                max-width: 480px;
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
            ("simulation", self.open_modes_menu),
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
        menu_scroll = QScrollArea()
        menu_scroll.setWidgetResizable(True)
        menu_scroll.setWidget(page)
        self.stack.addWidget(menu_scroll)

    def _build_modes_page(self):
        page = BackgroundPage(resource_path("images", "background.png"))
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        back_row = QHBoxLayout()
        self.modes_back_button = QPushButton()
        self.modes_back_button.clicked.connect(self.back_to_menu)
        back_row.addWidget(self.modes_back_button)
        back_row.addStretch(1)
        layout.addLayout(back_row)
        layout.addStretch(2)
        self.modes_title = QLabel()
        self.modes_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.modes_title.setWordWrap(True)
        self.modes_title.setStyleSheet("font-size: 32pt; font-weight: 800; color: #0f172a; background: #ffffff; padding: 20px; border-radius: 10px;")
        layout.addWidget(self.modes_title)
        layout.addSpacing(24)
        self.mode_buttons = {}
        for key, handler in (("laboratory", self.open_simulation),
                             ("memory_game", self.open_memory_game)):
            button = QPushButton()
            button.setStyleSheet("""
                QPushButton { font-size: 24pt; font-weight: 800; min-height: 80px;
                    min-width: 300px; max-width: 650px; border-radius: 10px;
                    background: #1e293b; color: #ffffff; padding: 12px 24px; }
                QPushButton:hover { background: #334155; border-color: #93c5fd; }
            """)
            button.clicked.connect(lambda _=False, action=handler: action())
            row = QHBoxLayout()
            row.addStretch(1)
            row.addWidget(button)
            row.addStretch(1)
            layout.addLayout(row)
            layout.addSpacing(12)
            self.mode_buttons[key] = button
        layout.addStretch(3)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.stack.addWidget(scroll)

    def set_language(self, language):  # Установка языка для всего приложения
        self.language = language
        text = MENU_TEXT[language]
        self.setWindowTitle(text["title"])
        self.menu_title.setText(text["title"])
        self.menu_subtitle.setText(text["subtitle"])
        self.menu_language_label.setText(text["language"])
        self.modes_title.setText(text["mode_menu_title"])
        self.modes_back_button.setText(text["main_menu"])
        self.mode_buttons["laboratory"].setText(text["mode_lab"])
        self.mode_buttons["memory_game"].setText(text["memory_game"])
        for key, button in self.menu_buttons.items():
            button.setText(text[key])
        for page in getattr(self, "pages", []):
            page.set_language(language)
        index = 0 if language == "ru" else 1
        if self.menu_language_combo.currentIndex() != index:
            self.menu_language_combo.blockSignals(True)
            self.menu_language_combo.setCurrentIndex(index)
            self.menu_language_combo.blockSignals(False)

    def open_page(self, index):  # Открыть страницу со справкой
        self.stack.setCurrentIndex(index)

    def open_modes_menu(self):
        self.stack.setCurrentIndex(self.PAGE_MODES)

    def back_to_modes(self):
        self._destroy_simulation()
        self._destroy_game()
        self.open_modes_menu()

    def open_simulation(self):  # Открыть окно симуляции
        if self.sim_window is not None:
            return
        self.sim_window = MainWindow(self.language, back_menu_key="mode_menu_back")
        self.sim_window.menuRequested.connect(self.back_to_modes)
        self.stack.addWidget(self.sim_window)
        self.stack.setCurrentWidget(self.sim_window)

    def open_memory_game(self):
        from boltzmann_lab.ui.memory_game import MemoryGameWindow
        if self.game_window is not None:
            self.stack.setCurrentWidget(self.game_window)
            return
        self.game_window = MemoryGameWindow(self.language, back_menu_key="mode_menu_back")
        self.game_window.menuRequested.connect(self.back_to_modes)
        self.stack.addWidget(self.game_window)
        self.stack.setCurrentWidget(self.game_window)

    def _destroy_game(self):
        if self.game_window is None:
            return
        window = self.game_window
        self.game_window = None
        window.shutdown()
        self.stack.removeWidget(window)
        self.set_language(window.language)
        window.deleteLater()

    def _destroy_simulation(self):  # Удалить окно симуляции (для экономии памяти)
        if self.sim_window is None:
            return
        window = self.sim_window
        self.sim_window = None
        language = window.language
        window.shutdown()
        self.stack.removeWidget(window)
        window.deleteLater()
        self.set_language(language)

    def back_to_menu(self):  # Вернуться в главное меню
        self._destroy_simulation()
        self._destroy_game()
        self.stack.setCurrentIndex(self.PAGE_MENU)

    def closeEvent(self, event):  # Закрытие приложения
        self._destroy_simulation()
        self._destroy_game()
        pygame.quit()
        event.accept()
