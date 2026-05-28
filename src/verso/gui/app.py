import sys
from pathlib import Path

import pyqtgraph as pg
from PyQt6.QtGui import QColor, QIcon, QPalette
from PyQt6.QtWidgets import QApplication

from verso.gui.main_window import MainWindow

_APP_ID = "verso.app"
_RESOURCES = Path(__file__).parent.parent / "resources"


def _set_taskbar_identity() -> None:
    """Tell the OS which app this process is, so the taskbar/dock groups it
    under VERSO's icon rather than the host Python interpreter's."""
    if sys.platform == "win32":
        # Without an explicit AppUserModelID, Windows groups the taskbar entry
        # under python.exe / pythonw.exe and uses Python's icon. The call must
        # happen before any window is shown.
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(_APP_ID)
        except Exception:
            pass


def _load_app_icon() -> QIcon:
    """Build a QIcon with every size we can find on disk.

    Prefers a set of PNGs named ``verso-<N>.png`` (one per size) because PNG
    works identically on every OS and lets the shell pick the size it wants.
    Falls back to ``verso.ico`` for backwards compatibility.
    """
    icon = QIcon()
    pngs = sorted(_RESOURCES.glob("verso-*.png"))
    if pngs:
        for png in pngs:
            icon.addFile(str(png))
        return icon
    ico = _RESOURCES / "verso.ico"
    if ico.exists():
        icon.addFile(str(ico))
    return icon


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


def run(project_path: Path | None = None) -> None:
    # Must be called before any pg widget is created.
    pg.setConfigOption("imageAxisOrder", "row-major")
    pg.setConfigOption("antialias", True)
    pg.setConfigOption("background", (30, 30, 30))
    pg.setConfigOption("foreground", (220, 220, 220))

    _set_taskbar_identity()

    app = QApplication(sys.argv)
    app.setApplicationName("VERSO")
    app.setOrganizationName("VERSO")
    if sys.platform.startswith("linux"):
        # GNOME/KDE/Wayland use this to match the window to its .desktop file
        # (and therefore its icon) instead of falling back to WM_CLASS guesses.
        app.setDesktopFileName(_APP_ID)

    app_icon = _load_app_icon()
    app.setWindowIcon(app_icon)

    app.setStyle("Fusion")
    app.setPalette(_build_dark_palette())

    window = MainWindow()
    window.setWindowIcon(app_icon)
    window.show()
    _center_on_screen(window)
    if project_path is not None:
        window.open_project_path(project_path)
    sys.exit(app.exec())
