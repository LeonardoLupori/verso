"""Shared chrome (status bar / header strip) used by Prep / Align / Warp views.

Keeps the bar's height, background, and the filename label's font / colour
identical across all three views.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

_STATUS_BAR_HEIGHT = 36
_STATUS_BAR_STYLE = "background: #252525; border-bottom: 1px solid #333;"
_STATUS_LABEL_STYLE = "color: #aaa; font-size: 11px;"


def style_status_label(label: QLabel) -> QLabel:
    """Apply the canonical filename-label style (font, colour, alignment)."""
    label.setStyleSheet(_STATUS_LABEL_STYLE)
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    return label


def make_view_status_bar(label: QLabel) -> QWidget:
    """Wrap ``label`` in the standard fixed-height view status bar."""
    style_status_label(label)
    container = QWidget()
    container.setFixedHeight(_STATUS_BAR_HEIGHT)
    container.setStyleSheet(_STATUS_BAR_STYLE)
    layout = QHBoxLayout(container)
    layout.setContentsMargins(8, 0, 8, 0)
    layout.setSpacing(0)
    layout.addWidget(label)
    layout.addStretch(1)
    return container
