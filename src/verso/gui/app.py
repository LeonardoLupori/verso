import sys
from pathlib import Path

import pyqtgraph as pg
from PyQt6.QtGui import QColor, QIcon, QPalette
from PyQt6.QtWidgets import QApplication

from verso.gui.main_window import MainWindow


def _center_on_screen(window: MainWindow) -> None:
    screen = window.screen() or QApplication.primaryScreen()
    if screen is None:
        return
    frame = window.frameGeometry()
    frame.moveCenter(screen.availableGeometry().center())
    window.move(frame.topLeft())


def _build_dark_palette() -> QPalette:
    palette = QPalette()

    window = QColor(45, 45, 45)
    base = QColor(30, 30, 30)
    alt_base = QColor(38, 38, 38)
    text = QColor(220, 220, 220)
    disabled_text = QColor(127, 127, 127)
    button = QColor(53, 53, 53)
    highlight = QColor(38, 110, 183)
    tooltip_bg = QColor(50, 50, 50)

    palette.setColor(QPalette.ColorRole.Window, window)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, base)
    palette.setColor(QPalette.ColorRole.AlternateBase, alt_base)
    palette.setColor(QPalette.ColorRole.ToolTipBase, tooltip_bg)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, button)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(86, 156, 214))
    palette.setColor(QPalette.ColorRole.Highlight, highlight)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.PlaceholderText, disabled_text)

    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.HighlightedText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, disabled_text)

    return palette


def run() -> None:
    # Must be called before any pg widget is created.
    pg.setConfigOption("imageAxisOrder", "row-major")
    pg.setConfigOption("antialias", True)
    pg.setConfigOption("background", (30, 30, 30))
    pg.setConfigOption("foreground", (220, 220, 220))

    app = QApplication(sys.argv)
    app.setApplicationName("VERSO")
    app.setOrganizationName("VERSO")

    _ICO = Path(__file__).parent.parent / "resources" / "verso.ico"
    app_icon = QIcon(str(_ICO))
    app.setWindowIcon(app_icon)

    app.setStyle("Fusion")
    app.setPalette(_build_dark_palette())

    window = MainWindow()
    window.setWindowIcon(app_icon)
    window.show()
    _center_on_screen(window)
    sys.exit(app.exec())
