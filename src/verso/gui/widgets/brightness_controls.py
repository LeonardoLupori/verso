"""Per-channel brightness/visibility/color controls.

Hosts one row per :class:`~verso.engine.model.project.ChannelSpec`.
Used by :class:`verso.gui.dialogs.brightness.BrightnessDialog` to drive
live updates of the canvas overlay.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QColorDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.project import ChannelSpec

_ICONS_DIR = Path(__file__).parent.parent / "icons"


def _eye_icon(visible: bool) -> QIcon:
    name = "eye.svg" if visible else "eye-off.svg"
    svg = (_ICONS_DIR / name).read_text(encoding="utf-8").replace("currentColor", "#ffffff")
    pixmap = QPixmap()
    pixmap.loadFromData(svg.encode())
    return QIcon(pixmap)


class _ChannelRow(QWidget):
    """One row inside :class:`BrightnessControls` — visibility, name, color, slider.

    Emits two signals:
      * :attr:`changed` — fires continuously while the slider is dragged
        (used for live canvas updates).
      * :attr:`committed` — fires once the user releases the slider, picks a
        color, or toggles visibility (used for expensive refreshes such as
        the filmstrip).
    """

    changed = pyqtSignal(int, object)    # index, ChannelSpec — live
    committed = pyqtSignal(int, object)  # index, ChannelSpec — on release

    def __init__(self, index: int, spec: ChannelSpec) -> None:
        super().__init__()
        self._index = index
        self._spec = ChannelSpec(
            name=spec.name,
            color=tuple(spec.color),
            scale=spec.scale,
            visible=spec.visible,
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._visible_btn = QToolButton()
        self._visible_btn.setCheckable(True)
        self._visible_btn.setChecked(self._spec.visible)
        self._visible_btn.setFixedSize(22, 22)
        self._visible_btn.setIconSize(QSize(16, 16))
        self._visible_btn.setToolTip("Toggle channel visibility")
        self._visible_btn.toggled.connect(self._on_visible)
        self._refresh_visible_btn()
        layout.addWidget(self._visible_btn)

        self._name_label = QLabel(self._spec.name)
        self._name_label.setMinimumWidth(36)
        self._name_label.setMaximumWidth(60)
        self._name_label.setToolTip(self._spec.name)
        layout.addWidget(self._name_label)

        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(20, 20)
        self._color_btn.setToolTip("Pick channel color")
        self._color_btn.clicked.connect(self._on_color)
        self._refresh_color_btn()
        layout.addWidget(self._color_btn)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(1, 100)
        self._slider.setValue(int(round(max(1.0, min(100.0, self._spec.scale * 100.0)))))
        self._slider.valueChanged.connect(self._on_slider)
        self._slider.sliderReleased.connect(self._on_slider_released)
        layout.addWidget(self._slider, stretch=1)

    def spec(self) -> ChannelSpec:
        return ChannelSpec(
            name=self._spec.name,
            color=tuple(self._spec.color),
            scale=self._spec.scale,
            visible=self._spec.visible,
        )

    def update_values(self, spec: ChannelSpec) -> None:
        """Update widget state silently from an external ``ChannelSpec``.

        Used when the panel is re-synced from the project — keeps the row's
        widgets stable so an in-progress slider drag isn't interrupted.
        """
        self._spec = ChannelSpec(
            name=spec.name,
            color=tuple(spec.color),
            scale=spec.scale,
            visible=spec.visible,
        )
        target = int(round(max(1.0, min(100.0, self._spec.scale * 100.0))))
        if self._slider.value() != target:
            self._slider.blockSignals(True)
            self._slider.setValue(target)
            self._slider.blockSignals(False)
        if self._visible_btn.isChecked() != self._spec.visible:
            self._visible_btn.blockSignals(True)
            self._visible_btn.setChecked(self._spec.visible)
            self._visible_btn.blockSignals(False)
        self._refresh_visible_btn()
        self._refresh_color_btn()

    def _refresh_visible_btn(self) -> None:
        self._visible_btn.setIcon(_eye_icon(self._spec.visible))

    def _refresh_color_btn(self) -> None:
        r, g, b = self._spec.color
        self._color_btn.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); border: 1px solid #555;"
            " border-radius: 2px;"
        )

    def _on_visible(self, checked: bool) -> None:
        self._spec.visible = bool(checked)
        self._refresh_visible_btn()
        spec = self.spec()
        self.changed.emit(self._index, spec)
        self.committed.emit(self._index, spec)

    def _on_color(self) -> None:
        current = QColor(*self._spec.color)
        color = QColorDialog.getColor(current, self, f"Color for {self._spec.name}")
        if color.isValid():
            self._spec.color = (color.red(), color.green(), color.blue())
            self._refresh_color_btn()
            spec = self.spec()
            self.changed.emit(self._index, spec)
            self.committed.emit(self._index, spec)

    def _on_slider(self, value: int) -> None:
        self._spec.scale = value / 100.0
        spec = self.spec()
        self.changed.emit(self._index, spec)
        # Keyboard / programmatic changes don't go through sliderReleased,
        # so commit immediately when the slider isn't being dragged.
        if not self._slider.isSliderDown():
            self.committed.emit(self._index, spec)

    def _on_slider_released(self) -> None:
        self.committed.emit(self._index, self.spec())


class BrightnessControls(QWidget):
    """Dynamic per-channel brightness/color/visibility controls.

    Hosts one :class:`_ChannelRow` per project-level
    :class:`~verso.engine.model.project.ChannelSpec`. Emits
    :attr:`channels_changed` whenever the user touches any control.
    """

    channels_changed = pyqtSignal(list)    # live, on every slider tick
    channels_committed = pyqtSignal(list)  # on slider release / discrete edits

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._channels: list[ChannelSpec] = []
        self._rows: list[_ChannelRow] = []

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

        self._empty_label = QLabel("No channels")
        self._empty_label.setStyleSheet("color: #888; font-style: italic;")
        self._layout.addWidget(self._empty_label)

    def set_channels(self, channels: list[ChannelSpec]) -> None:
        new_specs = [
            ChannelSpec(
                name=c.name,
                color=tuple(c.color),
                scale=c.scale,
                visible=c.visible,
            )
            for c in channels
        ]
        # Fast path: same channel layout (count + names) → update values in
        # place. Avoids destroying the row that owns the slider currently
        # being dragged when the parent re-syncs us.
        same_structure = len(new_specs) == len(self._channels) and all(
            new_specs[i].name == self._channels[i].name
            for i in range(len(new_specs))
        )
        if same_structure and len(self._rows) == len(new_specs):
            self._channels = new_specs
            for i, row in enumerate(self._rows):
                row.update_values(new_specs[i])
            return

        self._channels = new_specs
        self._rebuild()

    def _rebuild(self) -> None:
        for row in self._rows:
            self._layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()

        self._empty_label.setVisible(not self._channels)
        for i, spec in enumerate(self._channels):
            row = _ChannelRow(i, spec)
            row.changed.connect(self._on_row_changed)
            row.committed.connect(self._on_row_committed)
            self._rows.append(row)
            self._layout.addWidget(row)

    def _snapshot(self) -> list[ChannelSpec]:
        return [
            ChannelSpec(
                name=c.name,
                color=tuple(c.color),
                scale=c.scale,
                visible=c.visible,
            )
            for c in self._channels
        ]

    def _on_row_changed(self, idx: int, spec: ChannelSpec) -> None:
        if 0 <= idx < len(self._channels):
            self._channels[idx] = spec
        self.channels_changed.emit(self._snapshot())

    def _on_row_committed(self, idx: int, spec: ChannelSpec) -> None:
        if 0 <= idx < len(self._channels):
            self._channels[idx] = spec
        self.channels_committed.emit(self._snapshot())
