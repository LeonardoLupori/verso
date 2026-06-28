"""Simple read-only key/value information dialog."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QLabel,
    QVBoxLayout,
    QWidget,
)


def show_info_dialog(parent: QWidget, title: str, rows: list[tuple[str, str]]) -> None:
    """Show a modal dialog listing label/value rows.

    Args:
        parent: Widget to parent the dialog to.
        title: Window title, also shown as the heading.
        rows: ``(label, value)`` pairs rendered as a read-only form.
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setMinimumWidth(440)

    outer = QVBoxLayout(dlg)
    outer.setContentsMargins(20, 18, 20, 14)
    outer.setSpacing(14)

    heading = QLabel(title)
    heading.setStyleSheet("font-size: 15px; font-weight: bold; color: #e0e0e0;")
    outer.addWidget(heading)

    separator = QFrame()
    separator.setFrameShape(QFrame.Shape.HLine)
    separator.setStyleSheet("color: #444;")
    outer.addWidget(separator)

    form = QFormLayout()
    form.setSpacing(8)
    form.setHorizontalSpacing(16)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

    for label, value in rows:
        lbl = QLabel(label + ":")
        lbl.setStyleSheet("color: #888; font-size: 12px;")
        val = QLabel(value)
        val.setWordWrap(True)
        val.setStyleSheet("color: #ddd; font-size: 12px;")
        val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(lbl, val)

    outer.addLayout(form)
    outer.addStretch()

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dlg.accept)
    outer.addWidget(buttons)

    dlg.exec()
