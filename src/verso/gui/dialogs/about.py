"""Help / About VERSO popup: version, repo link, and third-party credits."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from verso import __version__

_REPO_URL = "https://github.com/LeonardoLupori/verso"
_ISSUES_URL = f"{_REPO_URL}/issues"
_LINK_COLOR = "#5aa0e0"

_DEPENDENCIES = [
    ("QuickNII", "https://github.com/Neural-Systems-at-UIO/QuickNII"),
    ("VisuAlign", "https://github.com/Tevemadar/VisuAlign"),
    ("DeepSlice", "https://github.com/PolarBean/DeepSlice"),
    ("Elastix", "https://github.com/InsightSoftwareConsortium/ITKElastix"),
    ("BrainGlobe", "https://brainglobe.info/index.html"),
    ("Allen Brain Atlas API", "https://mouse.brain-map.org/static/api"),
]


def _link(url: str, text: str | None = None) -> str:
    return f"<a href='{url}' style='color:{_LINK_COLOR};'>{text or url}</a>"


def show_about_dialog(parent: QWidget) -> None:
    """Show the Help / About VERSO popup.

    Args:
        parent: Widget to parent the dialog to.
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle("About VERSO")
    dlg.setMinimumWidth(440)

    outer = QVBoxLayout(dlg)
    outer.setContentsMargins(20, 18, 20, 14)
    outer.setSpacing(14)

    heading = QLabel(f"VERSO  <span style='color:#888; font-weight:normal;'>v{__version__}</span>")
    heading.setTextFormat(Qt.TextFormat.RichText)
    heading.setStyleSheet("font-size: 15px; font-weight: bold; color: #e0e0e0;")
    outer.addWidget(heading)

    repo_link = QLabel(_link(_REPO_URL))
    repo_link.setTextFormat(Qt.TextFormat.RichText)
    repo_link.setOpenExternalLinks(True)
    repo_link.setStyleSheet("font-size: 12px;")
    outer.addWidget(repo_link)

    separator = QFrame()
    separator.setFrameShape(QFrame.Shape.HLine)
    separator.setStyleSheet("color: #444;")
    outer.addWidget(separator)

    issues_label = QLabel(
        f"Found a bug or have a feature request? {_link(_ISSUES_URL, 'Open an issue on GitHub')}."
    )
    issues_label.setTextFormat(Qt.TextFormat.RichText)
    issues_label.setOpenExternalLinks(True)
    issues_label.setWordWrap(True)
    issues_label.setStyleSheet("color: #ddd; font-size: 12px;")
    outer.addWidget(issues_label)

    credits_intro = QLabel(
        "VERSO is free and open source. It builds on and integrates these excellent tools:"
    )
    credits_intro.setWordWrap(True)
    credits_intro.setStyleSheet("color: #ddd; font-size: 12px;")
    outer.addWidget(credits_intro)

    links_html = "<br>".join(f"&bull;&nbsp;{_link(url, name)}" for name, url in _DEPENDENCIES)
    links_label = QLabel(links_html)
    links_label.setTextFormat(Qt.TextFormat.RichText)
    links_label.setOpenExternalLinks(True)
    links_label.setStyleSheet("color: #ddd; font-size: 12px;")
    outer.addWidget(links_label)

    outer.addStretch()

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dlg.accept)
    outer.addWidget(buttons)

    dlg.exec()
