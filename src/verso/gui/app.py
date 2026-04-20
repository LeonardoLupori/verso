import sys

import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication

from verso.gui.main_window import MainWindow


def run() -> None:
    # Must be called before any pg widget is created.
    pg.setConfigOption("imageAxisOrder", "row-major")
    pg.setConfigOption("antialias", True)

    app = QApplication(sys.argv)
    app.setApplicationName("VERSO")
    app.setOrganizationName("VERSO")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
