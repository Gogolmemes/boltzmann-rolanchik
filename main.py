import sys
import numpy as np
import pygame
import pyqtgraph as pg

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
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


# ============================================================
# ПЕРЕВОД
# ============================================================

TEXT = {
    "ru": {
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
        "friction": "Трение γ",
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
    },
    "en": {
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
        "friction": "Friction γ",
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
    },
}


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

    Динамика:
        dx/dt = v
        dv/dt = F - gamma*v + thermal noise

    где
        F = -dU/dx.

    Используется схема типа BAOAB для Ланжевеновской динамики.
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

        thermal_speed = np.sqrt(
            max(self.temperature, 1e-6)
        )

        self.velocities = self.rng.normal(
            0.0,
            thermal_speed,
            size=(self.n, 2),
        )

        self._apply_outer_walls()

        for _ in range(8):
            self._resolve_collisions()

    # --------------------------------------------------------
    # Перегородка
    # --------------------------------------------------------

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

    def _handle_barrier_crossing(self, old_x):
        if self.barrier_mode == 0:
            return

        bx = self.barrier_x
        r = self.radius

        for i in range(self.n):
            previous = old_x[i]
            current = self.positions[i, 0]

            crossed_left_to_right = (
                previous < bx
                and current >= bx
            )

            crossed_right_to_left = (
                previous > bx
                and current <= bx
            )

            if not (
                crossed_left_to_right
                or crossed_right_to_left
            ):
                continue

            if self.particle_can_pass(
                self.positions[i, 1]
            ):
                continue

            if crossed_left_to_right:
                self.positions[i, 0] = bx - r
                self.velocities[i, 0] = -abs(
                    self.velocities[i, 0]
                )

            else:
                self.positions[i, 0] = bx + r
                self.velocities[i, 0] = abs(
                    self.velocities[i, 0]
                )

    def _enforce_barrier(self):
        if self.barrier_mode == 0:
            return

        bx = self.barrier_x
        r = self.radius

        for i in range(self.n):
            x, y = self.positions[i]

            if self.particle_can_pass(y):
                continue

            if abs(x - bx) < r:
                if x < bx:
                    self.positions[i, 0] = bx - r
                    self.velocities[i, 0] = -abs(
                        self.velocities[i, 0]
                    )
                else:
                    self.positions[i, 0] = bx + r
                    self.velocities[i, 0] = abs(
                        self.velocities[i, 0]
                    )

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

    def _drift(self, half_dt):
        old_x = self.positions[:, 0].copy()

        self.positions += (
            self.velocities * half_dt
        )

        self._apply_outer_walls()
        self._handle_barrier_crossing(old_x)

    def step(self, dt):
        if not self.potential_defined:
            return

        # B: половина действия силы
        force = self.force_x(
            self.positions[:, 0]
        )

        self.velocities[:, 0] += (
            0.5 * dt * force
        )

        # A: половина перемещения
        self._drift(0.5 * dt)

        # O: точный шаг Орнштейна-Уленбека
        gamma = max(self.gamma, 0.0)

        c = np.exp(-gamma * dt)

        T = max(self.temperature, 1e-8)

        sigma = np.sqrt(
            T * max(0.0, 1.0 - c * c)
        )

        noise = self.rng.normal(
            0.0,
            1.0,
            size=self.velocities.shape,
        )

        self.velocities = (
            c * self.velocities
            + sigma * noise
        )

        # A
        self._drift(0.5 * dt)

        # Столкновения
        self._resolve_collisions()

        # B
        force = self.force_x(
            self.positions[:, 0]
        )

        self.velocities[:, 0] += (
            0.5 * dt * force
        )

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
        if self.barrier_mode == 1:
            barrier = self.barrier_fraction

            left_mask = centers < barrier
            right_mask = ~left_mask

            current_left = np.mean(
                self.positions[:, 0]
                < self.barrier_x
            )

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
        self.particles_spin.setRange(20, 300)
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

        self.friction_spin = QDoubleSpinBox()
        self.friction_spin.setRange(0.10, 8.0)
        self.friction_spin.setDecimals(2)
        self.friction_spin.setSingleStep(0.1)
        self.friction_spin.setValue(2.0)

        self.particles_label = QLabel()
        self.radius_label = QLabel()
        self.temperature_label = QLabel()
        self.energy_label = QLabel()
        self.friction_label = QLabel()

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

        simulation_form.addRow(
            self.friction_label,
            self.friction_spin,
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
            ["", "", ""]
        )

        self.barrier_position_spin = (
            QDoubleSpinBox()
        )

        self.barrier_position_spin.setRange(
            0.10,
            0.90,
        )
        self.barrier_position_spin.setDecimals(2)
        self.barrier_position_spin.setSingleStep(
            0.05
        )
        self.barrier_position_spin.setValue(0.50)

        self.pore_spin = QSpinBox()
        self.pore_spin.setRange(4, 100)
        self.pore_spin.setValue(26)

        self.barrier_type_label = QLabel()
        self.barrier_x_label = QLabel()
        self.pore_label = QLabel()

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

        self.controls_layout.addWidget(
            self.barrier_group
        )

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

        self.particle_view = QLabel()
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

        self.friction_spin.valueChanged.connect(
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

        self.friction_label.setText(
            self.tr("friction")
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

        self.sim.gamma = (
            self.friction_spin.value()
        )

        self.sim.barrier_mode = (
            self.barrier_combo.currentIndex()
        )

        self.sim.barrier_fraction = (
            self.barrier_position_spin.value()
        )

        self.sim.pore_px = (
            self.pore_spin.value()
        )

        self.update_distribution()

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
        if (
            self.running
            and self.potential_defined
        ):
            for _ in range(SUBSTEPS):
                self.sim.step(DT)

        self.render_particle_field()

        self.frame_counter += 1

        # Графики необязательно обновлять 60 раз в секунду.
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
            bx = int(
                self.sim.barrier_fraction
                * FIELD_WIDTH
            )

            barrier_color = (
                248,
                113,
                113,
            )

            if self.sim.barrier_mode == 1:
                pygame.draw.line(
                    surface,
                    barrier_color,
                    (bx, 0),
                    (bx, FIELD_HEIGHT),
                    5,
                )

            else:
                intervals = (
                    self.sim.pore_intervals()
                )

                pixel_intervals = [
                    (
                        int(a * FIELD_HEIGHT),
                        int(b * FIELD_HEIGHT),
                    )
                    for a, b in intervals
                ]

                current = 0

                for a, b in pixel_intervals:
                    if a > current:
                        pygame.draw.line(
                            surface,
                            barrier_color,
                            (bx, current),
                            (bx, a),
                            5,
                        )

                    current = max(
                        current,
                        b,
                    )

                if current < FIELD_HEIGHT:
                    pygame.draw.line(
                        surface,
                        barrier_color,
                        (bx, current),
                        (bx, FIELD_HEIGHT),
                        5,
                    )

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
