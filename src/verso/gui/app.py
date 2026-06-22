import sys
from pathlib import Path

import pyqtgraph as pg
from PyQt6.QtCore import QSize, QTimer
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
    """Build a QIcon from ``verso.ico``, registering every embedded size.

    A single ``addFile`` on a multi-size ``.ico`` does not reliably expose all
    frames to Qt's native HICON conversion, so at non-100% DPI Windows can ask
    for a size that comes back empty and the taskbar icon goes blank. Adding
    each size explicitly guarantees the shell always finds a matching frame.
    """
    icon = QIcon()
    ico = _RESOURCES / "verso.ico"
    if ico.exists():
        for size in (16, 24, 32, 48, 64, 128, 256):
            icon.addFile(str(ico), QSize(size, size))
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
    # Once a widget carries its own stylesheet, Qt renders that widget's
    # tooltips with the stylesheet engine (a darker default) instead of the
    # palette's ToolTipBase.  Pin every tooltip to the palette colours so styled
    # widgets (e.g. the navigator buttons) match the rest of the app.
    app.setStyleSheet(
        "QToolTip { background-color: #323232; color: #dcdcdc; border: 1px solid #555; }"
    )

    window = MainWindow()
    window.setWindowIcon(app_icon)
    window.show()
    _center_on_screen(window)

    if sys.platform == "win32":
        # On a cold first launch Windows creates the taskbar button slightly
        # after show() and reads its icon before Qt has finished converting the
        # QIcon to a native HICON, so the button comes up blank. (Subsequent
        # launches hit the shell's per-AppUserModelID icon cache and look fine,
        # which is why this only ever bites the first run.) Re-applying the icon
        # once the event loop is running issues a WM_SETICON after the taskbar
        # button exists, forcing it to refresh.
        QTimer.singleShot(10, lambda: window.setWindowIcon(app_icon))

    if project_path is not None:
        window.open_project_path(project_path)
    sys.exit(app.exec())
