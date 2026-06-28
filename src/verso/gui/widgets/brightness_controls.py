"""Per-channel brightness/visibility/color controls.

Hosts one row per :class:`~verso.engine.model.project.ChannelSpec`.
Used by :class:`verso.gui.dialogs.brightness.BrightnessDialog` to drive
live updates of the canvas overlay.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QKeyEvent, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QColorDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.project import ChannelSpec

_ICONS_DIR = Path(__file__).parent.parent / "icons"

# Inline name editor styled after the overview table's slice-index chip
# (see ``_SliceIndexDelegate`` in views/overview_view.py): a faint outline at
# rest that brightens and fills on hover to hint the value is editable, and a
# distinct focused look while the user is actually typing.
_NAME_EDIT_QSS = """
QLineEdit {
    border: 1px solid #3f3f3f;
    border-radius: 5px;
    background: transparent;
    color: #d6d6d6;
    padding: 1px 4px;
}
QLineEdit:hover {
    border-color: #3a6d99;
    background: #27414f;
    color: #9fd0f2;
}
QLineEdit:focus {
    border-color: #3a6d99;
    background: #1e2e38;
    color: #ffffff;
}
"""


def _eye_icon(visible: bool) -> QIcon:
    name = "eye.svg" if visible else "eye-off.svg"
    svg = (_ICONS_DIR / name).read_text(encoding="utf-8").replace("currentColor", "#ffffff")
    pixmap = QPixmap()
    pixmap.loadFromData(svg.encode())
    return QIcon(pixmap)


class _EditableName(QLineEdit):
    """Inline channel-name field: a read-only chip until double-clicked.

    At rest it reads as an editable input (faint outline, hover highlight),
    matching the overview table's slice-index cell. Double-clicking unlocks
    editing; Return or focus-out commits, Escape reverts. Empty or unchanged
    text is silently reverted. The trimmed proposed name is emitted via
    :attr:`name_committed`; the parent validates/deduplicates it and may push
    a corrected name back via :meth:`set_name`.
    """

    name_committed = pyqtSignal(str)  # trimmed, non-empty, changed proposed name

    def __init__(self, name: str, parent: QWidget | None = None) -> None:
        super().__init__(name, parent)
        self._committed_text = name
        self.setReadOnly(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"{name}\nDouble-click to rename")
        self.setStyleSheet(_NAME_EDIT_QSS)
        self.setMinimumWidth(60)
        self.setMaximumWidth(120)
        self.editingFinished.connect(self._finish_edit)

    def set_name(self, name: str) -> None:
        """Set the displayed name without triggering a commit."""
        self._committed_text = name
        if self.text() != name:
            self.setText(name)
        self.setToolTip(f"{name}\nDouble-click to rename")

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self.isReadOnly():
            self._begin_edit()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if not self.isReadOnly():
            key = event.key()
            if key == Qt.Key.Key_Escape:
                self.setText(self._committed_text)
                self._lock()
                self.clearFocus()
                return
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                # Commit and consume the event so it doesn't bubble up to the
                # dialog and trigger the auto-default button (the color picker).
                self._finish_edit()
                self.clearFocus()
                event.accept()
                return
        super().keyPressEvent(event)

    def _begin_edit(self) -> None:
        self._committed_text = self.text()
        self.setReadOnly(False)
        self.setCursor(Qt.CursorShape.IBeamCursor)
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self.selectAll()

    def _lock(self) -> None:
        self.setReadOnly(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.deselect()

    def _finish_edit(self) -> None:
        # editingFinished fires on Return and on focus-out; guard against the
        # double fire and against firing while already locked.
        if self.isReadOnly():
            return
        proposed = self.text().strip()
        self._lock()
        if not proposed or proposed == self._committed_text:
            self.setText(self._committed_text)
            return
        self.name_committed.emit(proposed)


class _ChannelRow(QWidget):
    """One row inside :class:`BrightnessControls` — visibility, name, color, slider.

    Emits two signals:
      * :attr:`changed` — fires continuously while the slider is dragged
        (used for live canvas updates).
      * :attr:`committed` — fires once the user releases the slider, picks a
        color, or toggles visibility (used for expensive refreshes such as
        the filmstrip).
    """

    changed = pyqtSignal(int, object)  # index, ChannelSpec — live
    committed = pyqtSignal(int, object)  # index, ChannelSpec — on release
    rename_requested = pyqtSignal(int, str)  # index, proposed name — needs dedup

    def __init__(self, index: int, spec: ChannelSpec) -> None:
        super().__init__()
        self._index = index
        self._spec = ChannelSpec(
            name=spec.name,
            color=spec.color,
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

        self._name_edit = _EditableName(self._spec.name)
        self._name_edit.name_committed.connect(self._on_name_committed)
        layout.addWidget(self._name_edit)

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
            color=self._spec.color,
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
            color=spec.color,
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
        # Don't clobber an in-progress rename; sync only while the field is idle.
        if self._name_edit.isReadOnly():
            self._name_edit.set_name(self._spec.name)
        self._refresh_visible_btn()
        self._refresh_color_btn()

    def set_name(self, name: str) -> None:
        """Apply a (possibly deduplicated) name pushed back by the parent."""
        self._spec.name = name
        self._name_edit.set_name(name)

    def _refresh_visible_btn(self) -> None:
        self._visible_btn.setIcon(_eye_icon(self._spec.visible))

    def _refresh_color_btn(self) -> None:
        r, g, b = self._spec.color
        self._color_btn.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); border: 1px solid #555; border-radius: 2px;"
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

    def _on_name_committed(self, proposed: str) -> None:
        # Defer to the parent, which sees all channels and can deduplicate.
        self.rename_requested.emit(self._index, proposed)


class BrightnessControls(QWidget):
    """Dynamic per-channel brightness/color/visibility controls.

    Hosts one :class:`_ChannelRow` per project-level
    :class:`~verso.engine.model.project.ChannelSpec`. Emits
    :attr:`channels_changed` whenever the user touches any control.
    """

    channels_changed = pyqtSignal(list)  # live, on every slider tick
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
                color=c.color,
                scale=c.scale,
                visible=c.visible,
            )
            for c in channels
        ]
        # Fast path: same channel layout (count + names) → update values in
        # place. Avoids destroying the row that owns the slider currently
        # being dragged when the parent re-syncs us.
        same_structure = len(new_specs) == len(self._channels) and all(
            new_specs[i].name == self._channels[i].name for i in range(len(new_specs))
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
            row.rename_requested.connect(self._on_rename_requested)
            self._rows.append(row)
            self._layout.addWidget(row)

    def _snapshot(self) -> list[ChannelSpec]:
        return [
            ChannelSpec(
                name=c.name,
                color=c.color,
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
            # Preserve the existing name — color/brightness/visibility commits
            # must not resurrect a stale name from the row's spec.
            spec.name = self._channels[idx].name
            self._channels[idx] = spec
        self.channels_committed.emit(self._snapshot())

    def _on_rename_requested(self, idx: int, proposed: str) -> None:
        if not (0 <= idx < len(self._channels)):
            return
        final = self._dedupe_name(proposed, exclude=idx)
        self._channels[idx].name = final
        # Reflect the (possibly disambiguated) name back into the row.
        self._rows[idx].set_name(final)
        self.channels_committed.emit(self._snapshot())

    def _dedupe_name(self, name: str, exclude: int) -> str:
        """Return *name*, suffixed with ``(2)``, ``(3)``… if another channel
        already uses it (case-insensitive)."""
        taken = {c.name.casefold() for i, c in enumerate(self._channels) if i != exclude}
        if name.casefold() not in taken:
            return name
        n = 2
        while f"{name} ({n})".casefold() in taken:
            n += 1
        return f"{name} ({n})"
