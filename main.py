import sys
import numpy as np
import pygame
import pyqtgraph as pg

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QImage, 
    QPixmap,
    QImage, 
    QKeySequence, 
    QPixmap, 
    QShortcut
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
    QCheckBox
)


# ============================================================
# НАСТРОЙКИ
# ============================================================

POTENTIAL_POINTS = 512
HIST_BINS = 40

FIELD_WIDTH = 900
FIELD_HEIGHT = 280

# Физическое поле имеет те же пропорции, что и картинка.
WORLD_H = 1.0
WORLD_W = FIELD_WIDTH / FIELD_HEIGHT

FPS = 60
SUBSTEPS = 4
DT = 1.0 / (FPS * SUBSTEPS)

BARRIER_MIN = 0.10          # допустимый диапазон положения перегородки
BARRIER_MAX = 0.90
BARRIER_MAX_SPEED = 4.0     # макс. скорость стенки, в единицах поля в секунду
BARRIER_RESPONSE = 40.0     # 1/с: насколько быстро стенка догоняет курсор

DEMON_COLORS = [(239, 68, 68), (34, 197, 94), (59, 130, 246)]
DEMON_HEX = ["#ef4444", "#22c55e", "#3b82f6"]


# ============================================================
# ПЕРЕВОД
# ============================================================

TEXT = {
    "ru": {
        "barrier_scatter": "Рассеиватели",
        "barrier_semi": "Полупроницаемая",
        "scatter_shape": "Форма",
        "shape_circle": "Круги",
        "shape_ellipse": "Эллипсы",
        "shape_triangle": "Треугольники",
        "shape_mixed": "Смешанные",
        "scatter_count": "Число в ряду",
        "scatter_size": "Размер, px",
        "scatter_columns": "Число рядов",
        "scatter_angle": "Поворот, °",
        "p_lr": "P слева → направо",
        "p_rl": "P справа → налево",
        "window": "Лаборатория распределения Больцмана",
        "language": "Язык",
        "potential_group": "Потенциальная энергия",
        "new_potential": "Нарисовать новый график",
        "clear": "Очистить",
        "apply": "Готово / применить",
        "preset": "Готовые графики",
        "apply_preset": "Применить готовый",
        "preset_flat": "Плоский",
        "preset_well": "Одна яма",
        "preset_double": "Две ямы",
        "preset_hill": "Барьер",
        "preset_wave": "Периодический",
        "simulation_group": "Параметры моделирования",
        "particles": "Число частиц",
        "radius": "Радиус частицы, px",
        "temperature": "Температура T",
        "energy_scale": "Масштаб энергии",
        # "friction": "Трение γ",
        "start": "Старт",
        "pause": "Пауза",
        "reset": "Распределить заново",
        "barrier_group": "Перегородка",
        "barrier_type": "Тип",
        "barrier_none": "Нет",
        "barrier_solid": "Сплошная",
        "barrier_porous": "Пористая",
        "barrier_x": "Положение",
        "pore_size": "Размер пор, px",
        "potential_title": "Потенциальная энергия U(x)",
        "particle_title": "Движение частиц",
        "distribution_title": "Распределение вероятностей",
        "x_axis": "Положение x / L",
        "u_axis": "U / Umax",
        "probability": "Вероятность",
        "experimental": "Эксперимент",
        "theoretical": "Теория Больцмана",
        "draw_status": "Нарисуйте U(x) мышкой слева направо и нажмите «Готово / применить».",
        "running_status": "Моделирование выполняется.",
        "paused_status": "Моделирование приостановлено.",
        "no_potential": "Сначала нарисуйте потенциальную энергию.",
        "bad_drawing": "Нужно нарисовать линию хотя бы по некоторой части графика.",
        "com": "Центр масс",
        "entropy": "Энтропия",
        "left_particles": "Слева",
        "right_particles": "Справа",
        "hint": (
            "ЛКМ по верхнему графику — рисование потенциальной энергии.\n"
            "Новый график начинается с чистого поля."
        ),
        "barrier_work": "Перегородка совершает работу",
        "mode_label": "Режим",
        "mode_normal": "Обычный",
        "mode_demon": "Демон Максвелла",
        "demon_group": "Демон Максвелла",
        "demon_colors": "Число цветов",
        "demon_recolor": "Перекрасить частицы",
        "demon_color_1": "Красные",
        "demon_color_2": "Зелёные",
        "demon_color_3": "Синие",
        "pass_never": "Не пропускать",
        "pass_lr": "Пропускать →",
        "pass_rl": "Пропускать ←",
        "pass_all": "Пропускать всегда",
        "else_label": "   иначе",
        "else_ask": "Спрашивать",
        "else_reflect": "Отражать",
        "demon_pass": "Пропустить (→)",
        "demon_reflect": "Отразить (←)",
        "demon_particle": "Частица у перегородки",
        "demon_queue": "в очереди",
        "dir_lr": "летит слева направо",
        "dir_rl": "летит справа налево",
        "demon_wait": "Демон ждёт решения: пропустить или отразить частицу.",
        "demon_idle": "Демон наблюдает. Решение понадобится, когда частица подлетит к линии.",
        "shape_grad_lr": "Градиент → (малые к большим)",
        "shape_grad_rl": "Градиент ← (большие к малым)",
    },
    "en": {
        "barrier_scatter": "Scatterers",
        "barrier_semi": "Semi-permeable",
        "scatter_shape": "Shape",
        "shape_circle": "Circles",
        "shape_ellipse": "Ellipses",
        "shape_triangle": "Triangles",
        "shape_mixed": "Mixed",
        "scatter_count": "Count per row",
        "scatter_size": "Size, px",
        "scatter_columns": "Rows",
        "scatter_angle": "Rotation, °",
        "p_lr": "P left → right",
        "p_rl": "P right → left",
        "window": "Boltzmann Distribution Laboratory",
        "language": "Language",
        "potential_group": "Potential energy",
        "new_potential": "Draw new potential",
        "clear": "Clear",
        "apply": "Apply",
        "preset": "Presets",
        "apply_preset": "Apply preset",
        "preset_flat": "Flat",
        "preset_well": "Single well",
        "preset_double": "Double well",
        "preset_hill": "Barrier",
        "preset_wave": "Periodic",
        "simulation_group": "Simulation parameters",
        "particles": "Number of particles",
        "radius": "Particle radius, px",
        "temperature": "Temperature T",
        "energy_scale": "Energy scale",
        # "friction": "Friction γ",
        "start": "Start",
        "pause": "Pause",
        "reset": "Redistribute",
        "barrier_group": "Partition",
        "barrier_type": "Type",
        "barrier_none": "None",
        "barrier_solid": "Solid",
        "barrier_porous": "Porous",
        "barrier_x": "Position",
        "pore_size": "Pore size, px",
        "potential_title": "Potential energy U(x)",
        "particle_title": "Particle motion",
        "distribution_title": "Probability distribution",
        "x_axis": "Position x / L",
        "u_axis": "U / Umax",
        "probability": "Probability",
        "experimental": "Experiment",
        "theoretical": "Boltzmann theory",
        "draw_status": "Draw U(x) with the mouse from left to right and press Apply.",
        "running_status": "Simulation is running.",
        "paused_status": "Simulation is paused.",
        "no_potential": "Draw the potential energy first.",
        "bad_drawing": "Draw a line over at least part of the graph.",
        "com": "Center of mass",
        "entropy": "Entropy",
        "left_particles": "Left",
        "right_particles": "Right",
        "hint": (
            "Left mouse button on the upper graph draws potential energy.\n"
            "A new drawing starts from an empty graph."
        ),
        "barrier_work": "Перегородка совершает работу",
        "mode_label": "Mode",
        "mode_normal": "Normal",
        "mode_demon": "Maxwell's demon",
        "demon_group": "Maxwell's demon",
        "demon_colors": "Number of colors",
        "demon_recolor": "Recolor particles",
        "demon_color_1": "Red",
        "demon_color_2": "Green",
        "demon_color_3": "Blue",
        "pass_never": "Never let through",
        "pass_lr": "Let through →",
        "pass_rl": "Let through ←",
        "pass_all": "Always let through",
        "else_label": "   otherwise",
        "else_ask": "Ask",
        "else_reflect": "Reflect",
        "demon_pass": "Let through (→)",
        "demon_reflect": "Reflect (←)",
        "demon_particle": "Particle at the partition",
        "demon_queue": "queued",
        "dir_lr": "moving left to right",
        "dir_rl": "moving right to left",
        "demon_wait": "The demon is waiting: let the particle through or reflect it.",
        "demon_idle": "The demon is watching. A decision is needed when a particle reaches the line.",
        "shape_grad_lr": "Gradient → (small to large)",
        "shape_grad_rl": "Gradient ← (large to small)",
    },
}

class Obstacle:
    """
    Выпуклое препятствие, жёстко привязанное к перегородке.
    kind = "ellipse" (круг, если a == b) или "triangle".
    Координаты dx, cy: смещение по x от перегородки и y в мировых единицах.
    """

    def __init__(self, kind, dx, cy, angle, a=0.0, b=0.0, radius=0.0):
        self.kind = kind
        self.dx = dx
        self.cy = cy
        self.cs = float(np.cos(angle))
        self.sn = float(np.sin(angle))

        if kind == "ellipse":
            self.a = a
            self.b = b
            self.bound = max(a, b)
        else:
            t = angle + np.arange(3) * 2.0 * np.pi / 3.0
            self.local = radius * np.column_stack((np.cos(t), np.sin(t)))
            self.bound = radius

    def center(self, bx):
        return np.array([bx + self.dx, self.cy])

    def distance(self, p, bx):
        """Знаковое расстояние от центров частиц до поверхности и внешняя нормаль."""
        c = self.center(bx)

        if self.kind == "ellipse":
            return self._ellipse_distance(p, c)

        return self._polygon_distance(p, c + self.local)

    def _ellipse_distance(self, p, c):
        d = p - c
        u = self.cs * d[:, 0] + self.sn * d[:, 1]
        v = -self.sn * d[:, 0] + self.cs * d[:, 1]

        f = np.maximum(np.sqrt((u / self.a) ** 2 + (v / self.b) ** 2), 1e-9)

        gu = u / (self.a ** 2 * f)
        gv = v / (self.b ** 2 * f)
        g = np.sqrt(gu ** 2 + gv ** 2) + 1e-12

        # Расстояние в первом порядке: (F - 1) / |grad F|.
        dist = (f - 1.0) / g

        nu = gu / g
        nv = gv / g

        normal = np.column_stack((
            self.cs * nu - self.sn * nv,
            self.sn * nu + self.cs * nv,
        ))

        return dist, normal

    @staticmethod
    def _polygon_distance(p, verts):
        count = len(verts)

        best = np.full(len(p), np.inf)
        best_normal = np.zeros((len(p), 2))
        best_edge_normal = np.zeros((len(p), 2))
        inside = np.ones(len(p), dtype=bool)

        for k in range(count):
            a = verts[k]
            b = verts[(k + 1) % count]

            e = b - a
            length2 = float(e @ e)
            edge_normal = np.array([e[1], -e[0]]) / np.sqrt(length2)

            t = np.clip(((p - a) @ e) / length2, 0.0, 1.0)
            q = a + t[:, None] * e

            diff = p - q
            dist = np.linalg.norm(diff, axis=1)

            inside &= ((p - a) @ edge_normal) <= 0.0

            better = dist < best
            best[better] = dist[better]
            best_normal[better] = (
                diff[better] / np.maximum(dist[better], 1e-12)[:, None]
            )
            best_edge_normal[better] = edge_normal

        d = np.where(inside, -best, best)
        n = np.where(inside[:, None], best_edge_normal, best_normal)

        return d, n

    def outline(self, bx, points=36):
        """Контур для рисования (в мировых координатах)."""
        c = self.center(bx)

        if self.kind == "ellipse":
            t = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
            lx = self.a * np.cos(t)
            ly = self.b * np.sin(t)

            return np.column_stack((
                c[0] + self.cs * lx - self.sn * ly,
                c[1] + self.sn * lx + self.cs * ly,
            ))

        return c + self.local

class ParticleView(QLabel):
    """QLabel, который позволяет мышью схватить и двигать перегородку."""

    barrierMoved = pyqtSignal(float)   # новое положение в долях ширины поля

    GRAB_PX = 8

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.barrier_fraction = None   # None — перегородки нет
        self.dragging = False
        self.grab_offset = 0.0

    def _fraction(self, event):
        pm = self.pixmap()

        if pm is None or pm.isNull() or pm.width() == 0:
            return None, 1

        offset_x = (self.width() - pm.width()) / 2.0
        fx = (event.position().x() - offset_x) / pm.width()

        return fx, pm.width()

    def _hit(self, fx, width_px):
        if self.barrier_fraction is None or fx is None:
            return False

        return abs(fx - self.barrier_fraction) * width_px <= self.GRAB_PX

    def mousePressEvent(self, event):
        fx, w = self._fraction(event)

        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._hit(fx, w)
        ):
            self.dragging = True
            self.grab_offset = fx - self.barrier_fraction
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        fx, w = self._fraction(event)

        if self.dragging:
            if fx is not None:
                self.barrierMoved.emit(fx - self.grab_offset)
            event.accept()
            return

        if self._hit(fx, w):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.unsetCursor()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.dragging and event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.unsetCursor()
            event.accept()
            return

        super().mouseReleaseEvent(event)


# ============================================================
# ГРАФИК ДЛЯ РИСОВАНИЯ ПОТЕНЦИАЛА
# ============================================================

class PotentialPlot(pg.PlotWidget):
    """
    График PyQtGraph, на котором пользователь рисует U(x) мышкой.
    """

    drawingChanged = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.setBackground("#0b1220")
        self.showGrid(x=True, y=True, alpha=0.22)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)

        self.setXRange(0.0, 1.0, padding=0)
        self.setYRange(0.0, 1.0, padding=0)

        self.setLimits(
            xMin=0.0,
            xMax=1.0,
            yMin=0.0,
            yMax=1.0,
        )

        self.raw_points = []
        self.drawing = False
        self.drawing_enabled = True

        self.applied_curve = self.plot(
            [],
            [],
            pen=pg.mkPen("#34d399", width=3),
        )

        self.draft_curve = self.plot(
            [],
            [],
            pen=pg.mkPen("#fbbf24", width=3),
        )

    def begin_new(self):
        """Начать рисование полностью с нуля."""
        self.raw_points = []
        self.drawing = False
        self.drawing_enabled = True

        self.draft_curve.setData([], [])
        self.applied_curve.setData([], [])

    def clear_drawing(self):
        self.raw_points = []
        self.drawing = False
        self.draft_curve.setData([], [])
        self.applied_curve.setData([], [])
        self.drawingChanged.emit()

    def show_applied(self, potential):
        x = np.linspace(0.0, 1.0, len(potential))

        self.draft_curve.setData([], [])
        self.applied_curve.setData(x, potential)

        self.drawing_enabled = False

    def _event_to_data(self, event):
        point = event.position().toPoint()
        scene_pos = self.mapToScene(point)

        view_box = self.getPlotItem().getViewBox()

        if not view_box.sceneBoundingRect().contains(scene_pos):
            return None

        p = view_box.mapSceneToView(scene_pos)

        x = float(np.clip(p.x(), 0.0, 1.0))
        y = float(np.clip(p.y(), 0.0, 1.0))

        return x, y

    def _append_event_point(self, event):
        point = self._event_to_data(event)

        if point is None:
            return

        self.raw_points.append(point)

        data = np.asarray(self.raw_points)

        self.draft_curve.setData(
            data[:, 0],
            data[:, 1],
        )

    def mousePressEvent(self, event):
        if (
            self.drawing_enabled
            and event.button() == Qt.MouseButton.LeftButton
        ):
            # Каждый новый штрих заменяет старый.
            self.raw_points = []
            self.draft_curve.setData([], [])

            self.drawing = True
            self._append_event_point(event)

            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            self.drawing_enabled
            and self.drawing
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self._append_event_point(event)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if (
            self.drawing_enabled
            and self.drawing
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._append_event_point(event)
            self.drawing = False

            self.drawingChanged.emit()

            event.accept()
            return

        super().mouseReleaseEvent(event)

    def compile_potential(self, samples=POTENTIAL_POINTS):
        """
        Преобразование произвольного движения мыши в функцию U(x).

        1. Для каждого x берётся последнее нарисованное значение.
        2. Пропуски интерполируются.
        3. Края продолжаются горизонтально.
        4. График немного сглаживается.
        5. Значения нормируются от 0 до 1.
        """

        if len(self.raw_points) < 2:
            return None

        pts = np.asarray(self.raw_points, dtype=float)

        x = np.clip(pts[:, 0], 0.0, 1.0)
        y = np.clip(pts[:, 1], 0.0, 1.0)

        indices = np.rint(x * (samples - 1)).astype(int)

        result = np.full(samples, np.nan)

        # Если пользователь возвращался мышкой назад,
        # используется последнее значение y для данного x.
        for index, value in zip(indices, y):
            result[index] = value

        valid = np.flatnonzero(~np.isnan(result))

        if len(valid) < 2:
            return None

        all_indices = np.arange(samples)

        result = np.interp(
            all_indices,
            valid,
            result[valid],
        )

        # Лёгкое сглаживание, чтобы резкие движения мыши
        # не создавали огромные численные силы.
        kernel = np.array([0.25, 0.50, 0.25])

        for _ in range(3):
            padded = np.pad(result, 1, mode="edge")
            result = np.convolve(padded, kernel, mode="valid")

        minimum = np.min(result)
        maximum = np.max(result)

        if maximum - minimum < 1e-12:
            return np.zeros(samples)

        result = (result - minimum) / (maximum - minimum)

        return result


# ============================================================
# ФИЗИЧЕСКАЯ МОДЕЛЬ
# ============================================================

class BoltzmannSimulation:
    """
    Квазидвумерная визуализация частиц.

    Потенциальная энергия зависит только от x:
        U = U(x)

    Поэтому распределение по x является одномерным.

    """

    def __init__(self):
        self.rng = np.random.default_rng()

        self.n = 120

        self.radius_px = 4
        self.radius = self.radius_px / FIELD_HEIGHT

        self.temperature = 0.8
        self.energy_scale = 5.0
        self.gamma = 2.0

        self.barrier_mode = 0
        self.barrier_fraction = 0.5

        self.barrier_goal = 0.5        # куда стенку тянет мышь
        self.barrier_velocity = 0.0    # скорость стенки по x

        self.barrier_does_work = True   # False: стенка «как неподвижная»

        self.color_count = 2
        self.colors = np.zeros(0, dtype=int)
        self.demon_pass = [0, 0, 0]   # 0 не пропускать, 1 →, 2 ←, 3 всегда
        self.demon_else = [0, 0, 0]   # 0 спрашивать, 1 отражать
        self.demon_side = np.zeros(0, dtype=int)   # 0 слева, 1 справа
        self.demon_queue = []           # частицы, ждущие решения

        self.scatter_shape = 0         # 0 круги, 1 эллипсы, 2 треугольники, 3 смешанные
        self.scatter_count = 3
        self.scatter_size_px = 16
        self.scatter_columns = 1
        self.scatter_angle = 0.0       # градусы

        self.p_lr = 0.5                # вероятность пройти слева направо
        self.p_rl = 0.5                # вероятность пройти справа налево

        self._obstacle_key = None
        self._obstacles = []
        
        self.pore_px = 26

        self.potential_defined = False

        self.potential = np.zeros(POTENTIAL_POINTS)
        self.potential_x = np.linspace(
            0.0,
            WORLD_W,
            POTENTIAL_POINTS,
        )

        self.gradient = np.zeros(POTENTIAL_POINTS)

        self.positions = np.zeros((0, 2), dtype=float)
        self.velocities = np.zeros((0, 2), dtype=float)

        self.reset_particles(equilibrium=False)

    # --------------------------------------------------------
    # Потенциал
    # --------------------------------------------------------

    def set_potential(self, potential):
        potential = np.asarray(potential, dtype=float)

        if len(potential) != POTENTIAL_POINTS:
            old_x = np.linspace(0.0, 1.0, len(potential))
            new_x = np.linspace(0.0, 1.0, POTENTIAL_POINTS)

            potential = np.interp(
                new_x,
                old_x,
                potential,
            )

        self.potential = potential.copy()

        self.gradient = np.gradient(
            self.potential,
            self.potential_x,
            edge_order=2,
        )

        # Ограничиваем очень резкие силы после рисования мышью.
        self.gradient = np.clip(
            self.gradient,
            -20.0,
            20.0,
        )

        self.potential_defined = True

    def force_x(self, x):
        grad = np.interp(
            x,
            self.potential_x,
            self.gradient,
        )

        return -self.energy_scale * grad

    # --------------------------------------------------------
    # Частицы
    # --------------------------------------------------------

    def set_radius_px(self, value):
        self.radius_px = int(value)
        self.radius = self.radius_px / FIELD_HEIGHT

        self._apply_outer_walls()

        for _ in range(4):
            self._resolve_collisions()

    def _sample_boltzmann_x(self, count):
        """
        Выбор x с вероятностью exp(-U/T).
        """

        if not self.potential_defined:
            return self.rng.uniform(
                self.radius,
                WORLD_W - self.radius,
                count,
            )

        T = max(self.temperature, 1e-6)

        energy = self.energy_scale * self.potential

        energy = energy - np.min(energy)

        weights = np.exp(
            -np.clip(energy / T, 0.0, 700.0)
        )

        total = np.sum(weights)

        if total <= 0.0 or not np.isfinite(total):
            weights = np.ones_like(weights)

        weights = weights / np.sum(weights)

        indices = self.rng.choice(
            len(self.potential),
            size=count,
            p=weights,
        )

        dx = WORLD_W / (POTENTIAL_POINTS - 1)

        x = self.potential_x[indices]

        x += self.rng.uniform(
            -0.5 * dx,
            0.5 * dx,
            count,
        )

        return np.clip(
            x,
            self.radius,
            WORLD_W - self.radius,
        )

    def reset_particles(self, equilibrium=True):
        """
        Перераспределяет частицы.

        Если потенциал уже существует, x выбирается сразу
        согласно распределению Больцмана.
        """

        if equilibrium and self.potential_defined:
            x = self._sample_boltzmann_x(self.n)
        else:
            x = self.rng.uniform(
                self.radius,
                WORLD_W - self.radius,
                self.n,
            )

        y = self.rng.uniform(
            self.radius,
            WORLD_H - self.radius,
            self.n,
        )

        self.positions = np.column_stack((x, y))

        self.recolor()
        self.demon_assign_sides()

        thermal_speed = np.sqrt(
            max(self.temperature, 1e-6)
        )

        self.velocities = self.rng.normal(
            0.0,
            thermal_speed,
            size=(self.n, 2),
        )

        self.velocities -= np.mean(self.velocities, axis=0)

        self._apply_outer_walls()

        for _ in range(8):
            self._resolve_collisions()

    # --------------------------------------------------------
    # Перегородка
    # --------------------------------------------------------

    @property
    def demon_mode(self):
        return self.barrier_mode == 5

    @property
    def barrier_x(self):
        return self.barrier_fraction * WORLD_W

    @property
    def pore_height(self):
        return self.pore_px / FIELD_HEIGHT

    def pore_intervals(self):
        """
        Поры располагаются по высоте перегородки.
        """

        if self.barrier_mode != 2:
            return []

        pore_h = self.pore_height

        centers = [
            0.20 * WORLD_H,
            0.50 * WORLD_H,
            0.80 * WORLD_H,
        ]

        intervals = []

        for center in centers:
            a = max(0.0, center - pore_h / 2)
            b = min(WORLD_H, center + pore_h / 2)

            intervals.append((a, b))

        # Объединение пересекающихся пор.
        intervals.sort()

        merged = []

        for a, b in intervals:
            if not merged or a > merged[-1][1]:
                merged.append([a, b])
            else:
                merged[-1][1] = max(
                    merged[-1][1],
                    b,
                )

        return [
            (a, b)
            for a, b in merged
        ]

    def obstacles(self):
        """Список препятствий. Пересобирается только при смене параметров."""
        key = (
            self.scatter_shape,
            self.scatter_count,
            self.scatter_size_px,
            self.scatter_columns,
            self.scatter_angle,
        )

        if key == self._obstacle_key:
            return self._obstacles

        n = self.scatter_count

        # Чтобы соседние препятствия не слипались в сплошную стенку.
        size = min(self.scatter_size_px / FIELD_HEIGHT, 0.25 / n)

        angle = np.radians(self.scatter_angle)

        gradient = self.scatter_shape >= 4
        columns = 5 if gradient else self.scatter_columns
        pitch = (2.4 if gradient else 3.4) * size

        result = []
        k = 0

        for c in range(columns):
            dx = (c - (columns - 1) / 2.0) * pitch

            if c % 2 == 0:
                ys = [(i + 0.5) / n * WORLD_H for i in range(n)]
            else:
                # Сдвинутый ряд, как в доске Гальтона.
                ys = [i / n * WORLD_H for i in range(n + 1)]

            if gradient:
                t = c / (columns - 1)

                if self.scatter_shape == 5:
                    t = 1.0 - t

                # Радиус от 35% до 100% от заданного размера.
                radius = size * (0.35 + 0.65 * t)

            for cy in ys:
                if gradient:
                    ob = Obstacle(
                        "ellipse", dx, cy, angle, a=radius, b=radius
                    )
                else:
                    shape = (
                        self.scatter_shape
                        if self.scatter_shape < 3
                        else k % 3
                    )

                    if shape == 0:
                        ob = Obstacle("ellipse", dx, cy, angle, a=size, b=size)
                    elif shape == 1:
                        ob = Obstacle(
                            "ellipse", dx, cy, angle,
                            a=1.6 * size, b=0.7 * size,
                        )
                    else:
                        ob = Obstacle(
                            "triangle", dx, cy, angle, radius=1.4 * size
                        )

                k += 1
                result.append(ob)

        self._obstacle_key = key
        self._obstacles = result

        return result

    def _collide_scatterers(self):
        """
        Упругое отражение от бесконечно тяжёлых препятствий,
        движущихся вместе с перегородкой со скоростью w вдоль x.
        В системе отсчёта препятствия нормальная компонента скорости
        меняет знак: v' = v - 2 ((v - W) . n) n, где W = (w, 0).
        """
        bx = self.barrier_x
        w = self.collision_wall_velocity()
        r = self.radius

        pos = self.positions
        vel = self.velocities

        for ob in self.obstacles():
            c = ob.center(bx)

            near = np.flatnonzero(
                np.sum((pos - c) ** 2, axis=1) < (ob.bound + r) ** 2
            )

            if len(near) == 0:
                continue

            d, normal = ob.distance(pos[near], bx)

            hit = d < r

            if not np.any(hit):
                continue

            idx = near[hit]
            nh = normal[hit]

            pos[idx] += nh * (r - d[hit])[:, None]

            vn = (vel[idx, 0] - w) * nh[:, 0] + vel[idx, 1] * nh[:, 1]

            approaching = vn < 0.0
            j = idx[approaching]

            vel[j] -= 2.0 * vn[approaching][:, None] * nh[approaching]

    def recolor(self):
        self.colors = self.rng.integers(0, self.color_count, self.n)

    def demon_assign_sides(self):
        """Запоминает, по какую сторону линии находится каждая частица."""
        self.demon_side = (self.positions[:, 0] >= self.barrier_x).astype(int)
        self.demon_queue.clear()

    def _demon_place(self, i, side):
        eps = 1e-6
        self.positions[i, 0] = self.barrier_x + (eps if side == 1 else -eps)

    def _demon_reflect(self, i):
        side = int(self.demon_side[i])
        self._demon_place(i, side)

        speed = abs(self.velocities[i, 0])
        self.velocities[i, 0] = speed if side == 1 else -speed

    def _demon_collision(self):
        """
        Удар = центр частицы пересёк линию. Правило по цвету:
        пропустить, отразить или поставить частицу в очередь на решение.
        Ждущая частица возвращается на свою сторону, скорость не меняется.
        """
        bx = self.barrier_x
        x = self.positions[:, 0]
        side = self.demon_side

        crossed = (
            ((side == 0) & (x >= bx))
            | ((side == 1) & (x < bx))
        )

        for i in np.flatnonzero(crossed):
            c = int(self.colors[i])
            mode = self.demon_pass[c]

            # side == 0: частица летит слева направо (→), side == 1: справа налево (←)
            allowed = (
                mode == 3
                or (mode == 1 and side[i] == 0)
                or (mode == 2 and side[i] == 1)
            )

            if allowed:
                side[i] = 1 - side[i]
            elif self.demon_else[c] == 1:
                self._demon_reflect(i)
            else:
                self._demon_place(i, side[i])
                self.demon_queue.append(int(i))

    def _demon_enforce(self):
        """После парных столкновений никто не должен проскочить без решения."""
        if len(self.demon_side) != len(self.positions):
            return

        bx = self.barrier_x
        x = self.positions[:, 0]

        left = (self.demon_side == 0) & (x >= bx)
        right = (self.demon_side == 1) & (x < bx)

        self.positions[left, 0] = bx - 1e-6
        self.positions[right, 0] = bx + 1e-6

    def demon_resolve(self, let_pass):
        """Решение пользователя для первой частицы в очереди."""
        if not self.demon_queue:
            return

        i = self.demon_queue.pop(0)

        if let_pass:
            side = 1 - int(self.demon_side[i])
            self.demon_side[i] = side
            self._demon_place(i, side)
        else:
            self._demon_reflect(i)

    def _semipermeable_crossing(self, old_x, old_bx):
        """
        Каждое пересечение плоскости перегородки — случайное решение.
        Прошла: скорость и положение не меняются.
        Не прошла: отражение от стенки, v' = 2w - v.
        """
        bx = self.barrier_x
        w = self.collision_wall_velocity()
        
        x = self.positions[:, 0]

        left_to_right = (old_x < old_bx) & (x >= bx)
        right_to_left = (old_x >= old_bx) & (x < bx)

        hit = left_to_right | right_to_left

        if not np.any(hit):
            return

        probability = np.where(left_to_right, self.p_lr, self.p_rl)
        rejected = hit & (self.rng.random(self.n) >= probability)

        for mask, side in (
            (rejected & left_to_right, -1.0),
            (rejected & right_to_left, +1.0),
        ):
            self.positions[mask, 0] = bx + side * 1e-6

            v = self.velocities[:, 0]

            # Отражаем только если частица и стенка сближаются.
            if side < 0.0:
                bounce = mask & (v > w)
            else:
                bounce = mask & (v < w)

            self.velocities[bounce, 0] = 2.0 * w - v[bounce]

    def can_pass_array(self, y):
        """Векторная версия particle_can_pass."""
        y = np.asarray(y)

        if self.barrier_mode == 0:
            return np.ones(len(y), dtype=bool)

        if self.barrier_mode == 1:
            return np.zeros(len(y), dtype=bool)

        result = np.zeros(len(y), dtype=bool)
        r = self.radius

        if self.pore_height <= 2.0 * r:
            return result

        for a, b in self.pore_intervals():
            result |= (y - r >= a) & (y + r <= b)

        return result

    def particle_can_pass(self, y):
        if self.barrier_mode == 0:
            return True

        if self.barrier_mode == 1:
            return False

        # Диаметр должен помещаться в пору.
        if self.pore_height <= 2.0 * self.radius:
            return False

        for a, b in self.pore_intervals():
            if (
                y - self.radius >= a
                and y + self.radius <= b
            ):
                return True

        return False

    def collision_wall_velocity(self):
        """Скорость стенки, которая входит в формулы столкновения."""
        return self.barrier_velocity if self.barrier_does_work else 0.0

    def _move_barrier(self, dt):
        """Плавно двигает стенку к цели и запоминает её скорость."""
        goal = float(np.clip(self.barrier_goal, BARRIER_MIN, BARRIER_MAX))

        move = (goal - self.barrier_fraction) * WORLD_W
        move *= min(1.0, dt * BARRIER_RESPONSE)

        limit = BARRIER_MAX_SPEED * dt
        move = float(np.clip(move, -limit, limit))

        self.barrier_fraction += move / WORLD_W
        self.barrier_velocity = move / dt

    def _barrier_collision(self, old_x, old_bx):
        """
        Столкновение с бесконечно тяжёлой подвижной стенкой.

        Сторону определяем по положению ДО шага, чтобы быстрая стенка
        не могла «проскочить» сквозь частицу.

        В системе отсчёта стенки нормальная компонента скорости
        меняет знак: v' - w = -(v - w)  =>  v' = 2w - v.
        """
        if self.barrier_mode == 0:
            return

        if self.barrier_mode == 3:
            self._collide_scatterers()
            return

        if self.barrier_mode == 4:
            self._semipermeable_crossing(old_x, old_bx)
            return

        if self.barrier_mode == 5:
            self._demon_collision()
            return

        bx = self.barrier_x
        w = self.collision_wall_velocity()
        r = self.radius

        blocked = ~self.can_pass_array(self.positions[:, 1])

        x = self.positions[:, 0]
        v = self.velocities[:, 0]

        left = blocked & (old_x < old_bx)
        right = blocked & (old_x >= old_bx)

        hit = left & (x > bx - r)
        self.positions[hit, 0] = bx - r
        bounce = hit & (v > w)
        self.velocities[bounce, 0] = 2.0 * w - self.velocities[bounce, 0]

        hit = right & (x < bx + r)
        self.positions[hit, 0] = bx + r
        bounce = hit & (v < w)
        self.velocities[bounce, 0] = 2.0 * w - self.velocities[bounce, 0]

    def _enforce_barrier(self):
        """Страховка после столкновений частиц друг с другом."""

        if self.barrier_mode == 5:
            self._demon_enforce()
            return

        if self.barrier_mode in (0, 3, 4):
            return

        bx = self.barrier_x
        w = self.collision_wall_velocity()
        r = self.radius

        blocked = ~self.can_pass_array(self.positions[:, 1])

        x = self.positions[:, 0]
        v = self.velocities[:, 0]

        near = blocked & (np.abs(x - bx) < r)

        left = near & (x < bx)
        self.positions[left, 0] = bx - r
        bounce = left & (v > w)
        self.velocities[bounce, 0] = 2.0 * w - self.velocities[bounce, 0]

        right = near & (x >= bx)
        self.positions[right, 0] = bx + r
        bounce = right & (v < w)
        self.velocities[bounce, 0] = 2.0 * w - self.velocities[bounce, 0]

    # --------------------------------------------------------
    # Стенки
    # --------------------------------------------------------

    def _apply_outer_walls(self):
        if len(self.positions) == 0:
            return

        r = self.radius

        # Левая стенка
        mask = self.positions[:, 0] < r
        self.positions[mask, 0] = r
        self.velocities[mask, 0] = np.abs(
            self.velocities[mask, 0]
        )

        # Правая стенка
        mask = self.positions[:, 0] > WORLD_W - r
        self.positions[mask, 0] = WORLD_W - r
        self.velocities[mask, 0] = -np.abs(
            self.velocities[mask, 0]
        )

        # Верх
        mask = self.positions[:, 1] < r
        self.positions[mask, 1] = r
        self.velocities[mask, 1] = np.abs(
            self.velocities[mask, 1]
        )

        # Низ
        mask = self.positions[:, 1] > WORLD_H - r
        self.positions[mask, 1] = WORLD_H - r
        self.velocities[mask, 1] = -np.abs(
            self.velocities[mask, 1]
        )

    # --------------------------------------------------------
    # Столкновения частиц
    # --------------------------------------------------------

    def _resolve_collisions(self):
        """
        Упругие столкновения одинаковых круглых частиц.

        Для ускорения используется пространственная сетка,
        а не полный перебор всех N^2 пар.
        """

        if self.n <= 1:
            return

        diameter = 2.0 * self.radius

        if diameter <= 0:
            return

        cell_size = max(
            diameter * 1.05,
            1e-6,
        )

        grid = {}

        for i, position in enumerate(self.positions):
            cell = (
                int(position[0] / cell_size),
                int(position[1] / cell_size),
            )

            grid.setdefault(cell, []).append(i)

        checked = set()

        neighbour_offsets = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0),  (0, 0),  (1, 0),
            (-1, 1),  (0, 1),  (1, 1),
        ]

        min_distance_sq = diameter * diameter

        for cell, particles in grid.items():
            cx, cy = cell

            candidates = []

            for ox, oy in neighbour_offsets:
                candidates.extend(
                    grid.get(
                        (cx + ox, cy + oy),
                        [],
                    )
                )

            for i in particles:
                for j in candidates:
                    if j <= i:
                        continue

                    pair = (i, j)

                    if pair in checked:
                        continue

                    checked.add(pair)

                    delta = (
                        self.positions[j]
                        - self.positions[i]
                    )

                    distance_sq = float(
                        np.dot(delta, delta)
                    )

                    if (
                        distance_sq >= min_distance_sq
                        or distance_sq <= 1e-16
                    ):
                        continue

                    distance = np.sqrt(distance_sq)
                    normal = delta / distance

                    overlap = diameter - distance

                    # Раздвигаем частицы.
                    self.positions[i] -= (
                        normal * overlap * 0.5
                    )

                    self.positions[j] += (
                        normal * overlap * 0.5
                    )

                    # Относительная скорость вдоль нормали.
                    relative = float(
                        np.dot(
                            self.velocities[j]
                            - self.velocities[i],
                            normal,
                        )
                    )

                    # Они движутся навстречу друг другу.
                    if relative < 0.0:
                        self.velocities[i] += (
                            relative * normal
                        )

                        self.velocities[j] -= (
                            relative * normal
                        )

        self._apply_outer_walls()
        self._enforce_barrier()

    # --------------------------------------------------------
    # Ланжевеновская динамика
    # --------------------------------------------------------

    def step(self, dt):
        if not self.potential_defined:
            return

        old_bx = self.barrier_x
        if self.demon_mode:
            # Линия неподвижна и ставится строго по координате.
            self.barrier_fraction = float(
                np.clip(self.barrier_goal, BARRIER_MIN, BARRIER_MAX)
            )
            self.barrier_velocity = 0.0
        else:
            self._move_barrier(dt)

        force = self.force_x(self.positions[:, 0])
        self.velocities[:, 0] += 0.5 * dt * force

        old_x = self.positions[:, 0].copy()
        self.positions += self.velocities * dt

        self._apply_outer_walls()
        self._barrier_collision(old_x, old_bx)

        force_new = self.force_x(self.positions[:, 0])
        self.velocities[:, 0] += 0.5 * dt * force_new

        self._resolve_collisions()

    # --------------------------------------------------------
    # Статистика
    # --------------------------------------------------------

    def histogram(self, bins=HIST_BINS):
        if self.n == 0:
            return (
                np.zeros(bins),
                np.linspace(0, 1, bins + 1),
            )

        normalized_x = (
            self.positions[:, 0] / WORLD_W
        )

        counts, edges = np.histogram(
            normalized_x,
            bins=bins,
            range=(0.0, 1.0),
        )

        probabilities = (
            counts.astype(float) / self.n
        )

        return probabilities, edges

    def theoretical_distribution(self, bins=HIST_BINS):
        centers = (
            np.arange(bins) + 0.5
        ) / bins

        potential = np.interp(
            centers,
            np.linspace(
                0.0,
                1.0,
                POTENTIAL_POINTS,
            ),
            self.potential,
        )

        T = max(self.temperature, 1e-8)

        energy = self.energy_scale * potential
        energy -= np.min(energy)

        weights = np.exp(
            -np.clip(
                energy / T,
                0.0,
                700.0,
            )
        )

        # Если перегородка сплошная, число частиц
        # с каждой стороны сохраняется.
        # Поэтому нормируем теорию отдельно по двум областям.
        if self.barrier_mode in (1, 4):
            barrier = self.barrier_fraction

            left_mask = centers < barrier
            right_mask = ~left_mask

            current_left = float(
                np.mean(self.positions[:, 0] < self.barrier_x)
            )

            if self.barrier_mode == 4:
                # Стационарное состояние: потоки через стенку равны,
                # N_L p_LR / Z_L = N_R p_RL / Z_R.
                zl = float(np.sum(weights[left_mask]))
                zr = float(np.sum(weights[right_mask]))

                a = self.p_rl * zl
                b = self.p_lr * zr

                if a + b > 0.0:
                    current_left = a / (a + b)

            current_right = 1.0 - current_left

            result = np.zeros_like(weights)

            if np.any(left_mask):
                left_weights = weights[left_mask]

                if np.sum(left_weights) > 0:
                    result[left_mask] = (
                        current_left
                        * left_weights
                        / np.sum(left_weights)
                    )

            if np.any(right_mask):
                right_weights = weights[right_mask]

                if np.sum(right_weights) > 0:
                    result[right_mask] = (
                        current_right
                        * right_weights
                        / np.sum(right_weights)
                    )

            return centers, result

        total = np.sum(weights)

        if total <= 0:
            weights[:] = 1.0
            total = np.sum(weights)

        return centers, weights / total

    def center_of_mass(self):
        if self.n == 0:
            return 0.0

        return float(
            np.mean(self.positions[:, 0])
            / WORLD_W
        )

    def entropy(self):
        """
        Дискретная крупнозернистая энтропия:
            S = -sum p_i ln(p_i)

        k_B = 1.
        """

        p, _ = self.histogram()

        p = p[p > 0.0]

        if len(p) == 0:
            return 0.0

        return float(
            -np.sum(p * np.log(p))
        )

    def effective_temperature(self):
        if self.n == 0:
            return 0.0
    
        v2 = np.sum(self.velocities**2, axis=0)

        return float(np.mean(v2))


# ============================================================
# ГЛАВНОЕ ОКНО
# ============================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.language = "ru"

        self.sim = BoltzmannSimulation()

        self.running = False
        self.potential_defined = False

        self._last_queue_len = 0
        self.frame_counter = 0

        # Pygame используется как off-screen renderer.
        # Отдельного окна pygame нет.
        self.surface = pygame.Surface(
            (FIELD_WIDTH, FIELD_HEIGHT),
            depth=32,
        )

        self._build_ui()
        self._connect_signals()
        self.set_language("ru")

        self.new_potential()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animation_frame)
        self.timer.start(round(1000 / FPS))

        self.resize(1280, 900)
        self.setMinimumSize(950, 650)

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
        self.particles_spin.setRange(5, 1000)
        self.particles_spin.setValue(120)

        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(2, 10)
        self.radius_spin.setValue(4)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.10, 4.00)
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

    def animation_frame(self):
        if self.running and self.potential_defined:
            for _ in range(SUBSTEPS):
                # Симуляция стоит, пока есть нерешённая частица.
                if self.sim.demon_queue:
                    break

                self.sim.step(DT)
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

    def render_particle_field(self):
        surface = self.surface

        surface.fill((5, 10, 22))

        # ----------------------------------------------------
        # Фон показывает потенциальную энергию
        # ----------------------------------------------------

        if self.potential_defined:
            strips = 150

            for i in range(strips):
                x0 = int(
                    i * FIELD_WIDTH / strips
                )

                x1 = int(
                    (i + 1)
                    * FIELD_WIDTH
                    / strips
                )

                s = (
                    i + 0.5
                ) / strips

                u = np.interp(
                    s,
                    np.linspace(
                        0.0,
                        1.0,
                        POTENTIAL_POINTS,
                    ),
                    self.sim.potential,
                )

                u = float(
                    np.clip(u, 0.0, 1.0)
                )

                color = (
                    int(8 + 50 * u),
                    int(18 + 12 * u),
                    int(45 + 35 * (1.0 - u)),
                )

                pygame.draw.rect(
                    surface,
                    color,
                    (
                        x0,
                        0,
                        max(1, x1 - x0),
                        FIELD_HEIGHT,
                    ),
                )

        # ----------------------------------------------------
        # Сетка
        # ----------------------------------------------------

        grid_color = (35, 48, 70)

        for i in range(1, 10):
            x = int(
                FIELD_WIDTH * i / 10
            )

            pygame.draw.line(
                surface,
                grid_color,
                (x, 0),
                (x, FIELD_HEIGHT),
                1,
            )

        for i in range(1, 4):
            y = int(
                FIELD_HEIGHT * i / 4
            )

            pygame.draw.line(
                surface,
                grid_color,
                (0, y),
                (FIELD_WIDTH, y),
                1,
            )

        # ----------------------------------------------------
        # Перегородка
        # ----------------------------------------------------

        if self.sim.barrier_mode != 0:
            mode = self.sim.barrier_mode

            bx = int(
                self.sim.barrier_fraction * FIELD_WIDTH
            )

            barrier_color = (248, 113, 113)

            if mode == 1:
                pygame.draw.line(
                    surface,
                    barrier_color,
                    (bx, 0),
                    (bx, FIELD_HEIGHT),
                    5,
                )

            elif mode == 2:
                intervals = self.sim.pore_intervals()

                pixel_intervals = [
                    (int(a * FIELD_HEIGHT), int(b * FIELD_HEIGHT))
                    for a, b in intervals
                ]

                current = 0

                for a, b in pixel_intervals:
                    if a > current:
                        pygame.draw.line(
                            surface, barrier_color,
                            (bx, current), (bx, a), 5,
                        )

                    current = max(current, b)

                if current < FIELD_HEIGHT:
                    pygame.draw.line(
                        surface, barrier_color,
                        (bx, current), (bx, FIELD_HEIGHT), 5,
                    )

            elif mode == 3:
                self._draw_obstacles(surface, bx)
            elif mode == 5:
                self._draw_demon(surface, bx)
            else:
                self._draw_semipermeable(surface, bx)

        # ----------------------------------------------------
        # Центр масс
        # ----------------------------------------------------

        if self.sim.n > 0:
            com = self.sim.center_of_mass()

            com_x = int(
                com * FIELD_WIDTH
            )

            pygame.draw.line(
                surface,
                (250, 204, 21),
                (com_x, 0),
                (com_x, FIELD_HEIGHT),
                2,
            )

        # ----------------------------------------------------
        # Частицы
        # ----------------------------------------------------

        T = max(
            self.sim.temperature,
            1e-6,
        )

        reference_speed = (
            3.0 * np.sqrt(T)
        )

        speeds = np.linalg.norm(
            self.sim.velocities,
            axis=1,
        )

        for i in range(self.sim.n):
            px = int(
                self.sim.positions[i, 0]
                / WORLD_W
                * FIELD_WIDTH
            )

            py = int(
                self.sim.positions[i, 1]
                / WORLD_H
                * FIELD_HEIGHT
            )

            ratio = float(
                np.clip(
                    speeds[i]
                    / reference_speed,
                    0.0,
                    1.0,
                )
            )

            # Медленные — голубые,
            # быстрые — жёлто-красные.
            red = int(
                60 + 195 * ratio
            )

            green = int(
                180 + 60 * ratio
            )

            blue = int(
                255 - 190 * ratio
            )

            color = (
                red,
                min(255, green),
                max(0, blue),
            )

            if self.sim.demon_mode:
                color = DEMON_COLORS[int(self.sim.colors[i])]

            pygame.draw.circle(
                surface,
                color,
                (px, py),
                self.sim.radius_px,
            )

            pygame.draw.circle(
                surface,
                (235, 245, 255),
                (px, py),
                self.sim.radius_px,
                1,
            )

        if self.sim.demon_mode and self.sim.demon_queue:
            i = self.sim.demon_queue[0]

            px = int(self.sim.positions[i, 0] / WORLD_W * FIELD_WIDTH)
            py = int(self.sim.positions[i, 1] / WORLD_H * FIELD_HEIGHT)

            pygame.draw.circle(
                surface, (250, 204, 21),
                (px, py), self.sim.radius_px + 7, 3,
            )

        # Рамка
        pygame.draw.rect(
            surface,
            (100, 116, 139),
            (
                0,
                0,
                FIELD_WIDTH - 1,
                FIELD_HEIGHT - 1,
            ),
            2,
        )

        # ----------------------------------------------------
        # Pygame Surface -> NumPy -> QImage -> QLabel
        # ----------------------------------------------------

        rgb = pygame.surfarray.array3d(
            surface
        )

        # pygame: [x, y, rgb]
        # Qt:     [y, x, rgb]
        rgb = np.transpose(
            rgb,
            (1, 0, 2),
        )

        rgb = np.ascontiguousarray(rgb)

        image = QImage(
            rgb.data,
            FIELD_WIDTH,
            FIELD_HEIGHT,
            FIELD_WIDTH * 3,
            QImage.Format.Format_RGB888,
        ).copy()

        pixmap = QPixmap.fromImage(image)

        target_size = (
            self.particle_view.size()
        )

        pixmap = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.particle_view.setPixmap(
            pixmap
        )

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

    def closeEvent(self, event):
        if hasattr(self, "timer"):
            self.timer.stop()

        pygame.quit()

        event.accept()

    def _draw_obstacles(self, surface, bx):
        # Тонкая направляющая: за неё удобно хватать перегородку.
        pygame.draw.line(
            surface, (90, 60, 70), (bx, 0), (bx, FIELD_HEIGHT), 1
        )

        scale = FIELD_HEIGHT / WORLD_H

        for ob in self.sim.obstacles():
            points = [
                (int(x * scale), int(y * scale))
                for x, y in ob.outline(self.sim.barrier_x)
            ]

            pygame.draw.polygon(surface, (120, 50, 60), points)
            pygame.draw.polygon(surface, (248, 113, 113), points, 2)

    def _draw_semipermeable(self, surface, bx):
        base = (167, 139, 250)

        for y in range(0, FIELD_HEIGHT, 14):
            pygame.draw.line(
                surface, base,
                (bx, y), (bx, min(y + 8, FIELD_HEIGHT)), 4,
            )

        def shade(p):
            return tuple(int(40 + (c - 40) * p) for c in base)

        # Стрелки: яркость пропорциональна вероятности прохода.
        for fy in (0.2, 0.5, 0.8):
            y = int(fy * FIELD_HEIGHT)

            pygame.draw.polygon(
                surface, shade(self.sim.p_lr),
                [(bx - 20, y - 7), (bx - 20, y + 7), (bx - 8, y)],
            )

            pygame.draw.polygon(
                surface, shade(self.sim.p_rl),
                [(bx + 20, y - 7), (bx + 20, y + 7), (bx + 8, y)],
            )

    def _draw_demon(self, surface, bx):
        for y in range(0, FIELD_HEIGHT, 10):
            pygame.draw.line(
                surface, (226, 232, 240),
                (bx, y), (bx, min(y + 5, FIELD_HEIGHT)), 3,
            )

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


# ============================================================
# ЗАПУСК
# ============================================================

def main():
    app = QApplication(sys.argv)

    # Небольшой общий стиль.
    app.setStyleSheet(
        """
        QMainWindow {
            background: #f8fafc;
        }

        QGroupBox {
            font-weight: bold;
            border: 1px solid #cbd5e1;
            border-radius: 7px;
            margin-top: 10px;
            padding-top: 8px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }

        QPushButton {
            min-height: 28px;
            padding: 3px 8px;
        }

        QSpinBox, QDoubleSpinBox, QComboBox {
            min-height: 25px;
        }
        """
    )

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
