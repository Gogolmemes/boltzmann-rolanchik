import numpy as np
import pyqtgraph as pg

from PyQt6.QtCore import QEvent, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QLabel, QSizePolicy, QToolButton, QVBoxLayout

from boltzmann_lab.core.config import FIELD_HEIGHT, FIELD_WIDTH, POTENTIAL_POINTS
from boltzmann_lab.ui.theme import PLOT_BACKGROUND, contrast_text_color


def fit_window_to_screen(window):
    """Размеры в логических пикселях Qt с учётом рабочей области экрана."""
    available = window.screen().availableGeometry()
    window.setMinimumSize(min(640, int(available.width() * 0.8)),
                          min(420, int(available.height() * 0.8)))
    window.resize(min(1600, int(available.width() * 0.9)),
                  min(1000, int(available.height() * 0.9)))
    window.move(available.center() - window.rect().center())


class PlotLayout(QVBoxLayout):
    """Учитывает перенос текста, позволяя графикам сжиматься до минимума."""

    def heightForWidth(self, width):
        # Обычный QBoxLayout складывает желаемые высоты графиков.
        # Для QScrollArea нужна минимальная высота при данной ширине.
        margins = self.contentsMargins()
        inner_width = max(0, width - margins.left() - margins.right())
        items = [self.itemAt(i) for i in range(self.count())]
        items = [item for item in items if not item.isEmpty()]
        height = margins.top() + margins.bottom()
        for item in items:
            minimum = item.minimumSize().height()
            widget = item.widget()
            if widget is not None and widget.hasHeightForWidth():
                minimum = max(minimum, widget.heightForWidth(inner_width))
            height += minimum
        return height + max(0, len(items) - 1) * max(0, self.spacing())


class CollapsiblePlotHeader(QToolButton):
    """Full heading when collapsed; a caption and small control inside an open plot."""

    def __init__(self, plot, expanded=True):
        super().__init__(plot)
        self.plot = plot
        self._title = ""
        self.setCheckable(True)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setStyleSheet("""
            QToolButton {
                font-size: 22pt; font-weight: 800; color: #0f172a;
                min-height: 0; padding: 3px 8px;
            }
        """)

        self.caption = QLabel(plot.viewport())
        self.caption.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.caption.setStyleSheet("""
            QLabel {
                color: #ffffff; background: rgba(11, 18, 32, 210);
                font-size: 14pt; font-weight: 700;
                border: none; border-radius: 4px; padding: 2px 6px;
            }
        """)
        self.collapse_button = QToolButton(plot.viewport())
        self.collapse_button.setArrowType(Qt.ArrowType.UpArrow)
        self.collapse_button.setFixedSize(28, 28)
        self.collapse_button.setStyleSheet("""
            QToolButton {
                min-height: 0; min-width: 0; padding: 0;
                color: #ffffff; background: #1e293b;
                border: 1px solid #94a3b8; border-radius: 4px;
            }
            QToolButton:hover, QToolButton:focus {
                background: #334155; border-color: #38bdf8;
            }
        """)
        self.collapse_button.clicked.connect(lambda: self.setChecked(False))
        self.toggled.connect(self._set_expanded)
        plot.getViewBox().sigResized.connect(self._position_overlay)
        plot.viewport().installEventFilter(self)
        self.setChecked(expanded)
        if not expanded:
            self._set_expanded(False)

    def setText(self, text):
        self._title = text
        self.setToolTip(text)
        self.setAccessibleName(text)
        self.caption.setToolTip(text)
        self._update_header_text()
        self._position_overlay()

    def text(self):
        return self._title

    def _update_header_text(self):
        super().setText(self.fontMetrics().elidedText(
            self._title, Qt.TextElideMode.ElideRight, max(0, self.width() - 44)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_header_text()

    def setCollapseText(self, text):
        description = f"{text}: {self.text()}"
        self.collapse_button.setToolTip(description)
        self.collapse_button.setAccessibleName(description)

    def _set_expanded(self, expanded):
        self.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.setVisible(not expanded)
        self.plot.setVisible(expanded)
        self.caption.setVisible(expanded)
        self.collapse_button.setVisible(expanded)
        if expanded:
            self._position_overlay()

    def _position_overlay(self):
        # Anchor in the data area so captions never cover axis labels.
        bounds = self.plot.mapFromScene(self.plot.getViewBox().sceneBoundingRect()).boundingRect()
        button = self.collapse_button
        button.move(max(bounds.left(), bounds.right() - button.width() - 6), bounds.top() + 6)
        available = max(0, button.x() - bounds.left() - 14)
        text = self.caption.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight,
                                                   max(0, available - 12))
        self.caption.setText(text)
        self.caption.adjustSize()
        self.caption.resize(min(available, self.caption.width()), self.caption.height())
        self.caption.move(bounds.left() + 6, bounds.top() + 6)
        self.caption.raise_()
        button.raise_()

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self._position_overlay()
        return super().eventFilter(watched, event)


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

class ParticleFieldView(ParticleView):
    """Reserve the vessel's full aspect ratio before distributing plot heights."""

    def __init__(self):
        super().__init__()
        policy = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(160)
        self._source_pixmap = None

    def setPixmap(self, pixmap):
        self._source_pixmap = pixmap
        self._fit_pixmap()

    def _fit_pixmap(self):
        if self._source_pixmap is not None and not self._source_pixmap.isNull():
            super().setPixmap(self._source_pixmap.scaledToWidth(
                max(1, self.contentsRect().width()),
                Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_pixmap()

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        border = 2 * self.frameWidth()
        return round(max(0, width - border) * FIELD_HEIGHT / FIELD_WIDTH) + border

    def minimumSizeHint(self):
        return QSize(160, self.heightForWidth(160))

    def sizeHint(self):
        return QSize(FIELD_WIDTH, self.heightForWidth(FIELD_WIDTH))


class PresentationPlot(pg.PlotWidget):
    """Large graph labels whose colors follow the actual graph background."""

    def __init__(self):
        super().__init__(background=PLOT_BACKGROUND)
        self.setBackground(PLOT_BACKGROUND)

    def setBackground(self, background):
        super().setBackground(background)
        if self.plotItem is None:
            return
        color = contrast_text_color(self.backgroundBrush().color())
        tick_font = QFont("Segoe UI", 16, QFont.Weight.Bold)
        for name in ("left", "bottom", "right", "top"):
            axis = self.plotItem.getAxis(name)
            axis.setPen(pg.mkPen(color, width=2))
            axis.setTextPen(color)
            axis.setTickFont(tick_font)
            axis.setStyle(tickTextOffset=10)
            axis.setLabel(axis.labelText, units=axis.labelUnits,
                          **{"color": color.name(), "font-size": "18pt", "font-weight": "700"})
        self._style_legend()

    def addLegend(self, **kwargs):
        legend = self.plotItem.addLegend(**kwargs)
        self._style_legend()
        return legend

    def _style_legend(self):
        legend = self.plotItem.legend
        if legend is None:
            return
        background = self.backgroundBrush().color()
        color = contrast_text_color(background)
        legend.setBrush(pg.mkBrush(background))
        legend.setPen(pg.mkPen(color, width=2))
        legend.setLabelTextColor(color)
        legend.setLabelTextSize("16pt")
        for _, label in legend.items:
            label.setAttr("bold", True)


class PotentialPlot(PresentationPlot):
    """
    График PyQtGraph, на котором пользователь рисует U(x) мышкой.
    """

    drawingChanged = pyqtSignal()
    positionSelected = pyqtSignal(float)

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
            pen=pg.mkPen("#34d399", width=4),
        )

        self.draft_curve = self.plot(
            [],
            [],
            pen=pg.mkPen("#fbbf24", width=4),
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
        if not self.drawing_enabled and event.button() == Qt.MouseButton.LeftButton:
            point = self._event_to_data(event)
            if point is not None:
                self.positionSelected.emit(point[0])
                event.accept()
                return
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