import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel

from boltzmann_lab.core.config import POTENTIAL_POINTS


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