import sys

from PyQt6.QtWidgets import QApplication

from boltzmann_lab.core.physics import warmup_numba
from boltzmann_lab.ui.windows import AppWindow
from boltzmann_lab.ui.theme import apply_presentation_theme


def run_app():
    app = QApplication(sys.argv)

    apply_presentation_theme(app)

    # Предварительная компиляция вычислительного ядра
    warmup_numba()

    window = AppWindow()
    window.show()

    return app.exec()