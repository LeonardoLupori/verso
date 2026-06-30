"""Warning dialog shown before a flip that would delete an existing alignment."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from verso.gui.utils import require


def confirm_flip_deletes_alignment(parent: QWidget) -> tuple[bool, bool]:
    """Show a modal warning that flipping will reset the alignment.

    Returns:
        (confirmed, do_not_show_again) — ``confirmed`` is True when the user
        accepted the flip; ``do_not_show_again`` is True when the checkbox was
        ticked (only meaningful when ``confirmed`` is True).
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle("Flip image")
    dlg.setMinimumWidth(400)

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(4)

    heading = QLabel("Reset alignment?")
    heading.setStyleSheet("font-size: 14px; font-weight: bold; color: #e0e0e0;")
    layout.addWidget(heading)

    separator = QFrame()
    separator.setFrameShape(QFrame.Shape.HLine)
    separator.setStyleSheet("color: #444;")
    layout.addWidget(separator)

    msg = QLabel(
        "Flipping will reset this section's alignment to the default suggestion."
        "\nAny manual alignment or warp control points will be lost."
    )
    msg.setWordWrap(True)
    msg.setStyleSheet("color: #ccc; font-size: 12px;")
    layout.addWidget(msg)

    layout.addSpacing(4)

    no_show_cb = QCheckBox("Do not show this again")
    no_show_cb.setStyleSheet("color: #888; font-size: 11px;")
    layout.addWidget(no_show_cb)

    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    require(buttons.button(QDialogButtonBox.StandardButton.Ok)).setText("Flip anyway")
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)

    confirmed = dlg.exec() == QDialog.DialogCode.Accepted
    return confirmed, no_show_cb.isChecked()
