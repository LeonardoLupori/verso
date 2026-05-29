"""Shared helpers for the properties panel sections."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QWidget

_ICONS_DIR = Path(__file__).parent.parent.parent / "icons"


def white_icon(name: str) -> QIcon:
    svg = (_ICONS_DIR / name).read_text(encoding="utf-8").replace("currentColor", "#ffffff")
    pixmap = QPixmap()
    pixmap.loadFromData(svg.encode())
    return QIcon(pixmap)


def eye_icon(visible: bool) -> QIcon:
    name = "eye.svg" if visible else "eye-off.svg"
    return white_icon(name)


def make_eye_btn() -> QPushButton:
    btn = QPushButton()
    btn.setCheckable(True)
    btn.setChecked(True)
    btn.setFixedSize(24, 24)
    btn.setFlat(True)
    btn.setIcon(eye_icon(True))
    btn.setIconSize(QSize(16, 16))
    btn.toggled.connect(lambda checked, b=btn: b.setIcon(eye_icon(checked)))
    return btn


def color_swatch_style(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return (
        f"QPushButton {{ background-color: rgb({r}, {g}, {b}); border: 1px solid #555;"
        " border-radius: 2px; }"
    )


def make_segmented_buttons(
    parent: QWidget,
    specs: list[tuple[str, str]],
    tooltips: list[str] | None = None,
    initial_key: str | None = None,
) -> tuple[QHBoxLayout, dict[str, QPushButton], QButtonGroup]:
    """Build a row of joined toggle buttons (radio-style)."""
    btns: dict[str, QPushButton] = {}
    group = QButtonGroup(parent)
    group.setExclusive(True)
    row = QHBoxLayout()
    row.setSpacing(0)
    n = len(specs)
    for i, (key, label) in enumerate(specs):
        btn = QPushButton(label)
        btn.setCheckable(True)
        btn.setChecked(key == initial_key if initial_key is not None else i == 0)
        btn.setFixedHeight(24)
        if tooltips is not None:
            btn.setToolTip(tooltips[i])
        if i == 0:
            radius = (
                "border-top-left-radius: 4px; border-bottom-left-radius: 4px;"
                " border-top-right-radius: 0px; border-bottom-right-radius: 0px;"
            )
            margin = ""
        elif i == n - 1:
            radius = (
                "border-top-right-radius: 4px; border-bottom-right-radius: 4px;"
                " border-top-left-radius: 0px; border-bottom-left-radius: 0px;"
            )
            margin = "margin-left: -1px;"
        else:
            radius = "border-radius: 0px;"
            margin = "margin-left: -1px;"
        btn.setStyleSheet(
            f"QPushButton {{ {radius} {margin} padding: 2px 6px; color: #ccc;"
            f" background: #3a3a3a; border: 1px solid #555; }}"
            "QPushButton:checked { background: #1e5a8a; color: #fff;"
            " border-color: #1e5a8a; }"
            "QPushButton:hover:!checked { background: #4a4a4a; }"
        )
        btns[key] = btn
        group.addButton(btn)
        row.addWidget(btn)
    return row, btns, group
