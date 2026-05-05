import sys

import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication

from verso.gui.main_window import MainWindow


def _center_on_screen(window: MainWindow) -> None:
    screen = window.screen() or QApplication.primaryScreen()
    if screen is None:
        return
    frame = window.frameGeometry()
    frame.moveCenter(screen.availableGeometry().center())
    window.move(frame.topLeft())


def run() -> None:
    # Must be called before any pg widget is created.
    pg.setConfigOption("imageAxisOrder", "row-major")
    pg.setConfigOption("antialias", True)

    app = QApplication(sys.argv)
    app.setApplicationName("VERSO")
    app.setOrganizationName("VERSO")

    window = MainWindow()
    window.show()
    _center_on_screen(window)
    sys.exit(app.exec())
