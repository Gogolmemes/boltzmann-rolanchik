import sys

from PyQt6.QtWidgets import QApplication

from boltzmann_lab.core.physics import warmup_numba
from boltzmann_lab.ui.windows import AppWindow
from boltzmann_lab.utils.resources import load_stylesheet


def run_app():
    app = QApplication(sys.argv)

    app.setStyle("Fusion")
    app.setStyleSheet(load_stylesheet())

    # Предварительная компиляция вычислительного ядра
    warmup_numba()

    window = AppWindow()
    window.show()

    return app.exec()