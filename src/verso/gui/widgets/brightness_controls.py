"""Per-channel brightness/visibility/color controls.

Hosts one row per :class:`~verso.engine.model.project.ChannelSpec`.
Used by :class:`verso.gui.dialogs.brightness.BrightnessDialog` to drive
live updates of the canvas overlay.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import (
    QColorDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.project import ChannelSpec
from verso.gui.widgets.properties._common import (
    color_swatch_style,
    eye_icon,
    make_eye_btn,
)

# Fixed column widths so the header labels line up over the row controls even
# though every row is its own independent HBox (not a shared grid).
_EYE_W = 24
_NAME_W = 104
_COLOR_W = 20
_VALUE_W = 38
_GAMMA_W = 68
_ROW_SPACING = 6

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

# Muted column titles above the channel rows, matching the panel hint labels.
_HEADER_QSS = "color: #888; font-size: 11px; font-weight: 600;"

# Numeric slider readout, as in the mask/annotation sections' slider rows.
_VALUE_QSS = "color: #aaa; font-size: 11px;"


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
        self.setFixedWidth(_NAME_W)
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
            gamma=spec.gamma,
            visible=spec.visible,
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(_ROW_SPACING)

        # Flat eye toggle, identical to the mask/annotation section toggles.
        self._visible_btn = make_eye_btn()
        self._visible_btn.setFixedSize(_EYE_W, _EYE_W)
        self._visible_btn.setChecked(self._spec.visible)
        self._visible_btn.setToolTip("Toggle channel visibility")
        self._visible_btn.toggled.connect(self._on_visible)
        self._refresh_visible_btn()
        layout.addWidget(self._visible_btn)

        self._name_edit = _EditableName(self._spec.name)
        self._name_edit.name_committed.connect(self._on_name_committed)
        layout.addWidget(self._name_edit)

        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(_COLOR_W, _COLOR_W)
        # Not a dialog default button: otherwise Return from the name field or
        # the gamma spin-box would pop the colour picker.
        self._color_btn.setAutoDefault(False)
        self._color_btn.setDefault(False)
        self._color_btn.setToolTip("Pick channel color")
        self._color_btn.clicked.connect(self._on_color)
        self._refresh_color_btn()
        layout.addWidget(self._color_btn)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(1, 100)
        self._slider.setValue(round(max(1.0, min(100.0, self._spec.scale * 100.0))))
        self._slider.valueChanged.connect(self._on_slider)
        self._slider.sliderReleased.connect(self._on_slider_released)
        layout.addWidget(self._slider, stretch=1)

        self._value_label = QLabel()
        self._value_label.setFixedWidth(_VALUE_W)
        self._value_label.setStyleSheet(_VALUE_QSS)
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._refresh_value_label()
        layout.addWidget(self._value_label)

        self._gamma_spin = QDoubleSpinBox()
        self._gamma_spin.setRange(0.1, 5.0)
        self._gamma_spin.setSingleStep(0.1)
        self._gamma_spin.setDecimals(2)
        self._gamma_spin.setFixedWidth(_GAMMA_W)
        self._gamma_spin.setKeyboardTracking(False)
        self._gamma_spin.setValue(self._spec.gamma)
        self._gamma_spin.setToolTip("Gamma (1 = linear; <1 brightens shadows, >1 darkens)")
        self._gamma_spin.valueChanged.connect(self._on_gamma)
        layout.addWidget(self._gamma_spin)

    def spec(self) -> ChannelSpec:
        return ChannelSpec(
            name=self._spec.name,
            color=self._spec.color,
            scale=self._spec.scale,
            gamma=self._spec.gamma,
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
            gamma=spec.gamma,
            visible=spec.visible,
        )
        target = round(max(1.0, min(100.0, self._spec.scale * 100.0)))
        if self._slider.value() != target:
            self._slider.blockSignals(True)
            self._slider.setValue(target)
            self._slider.blockSignals(False)
        if self._gamma_spin.value() != self._spec.gamma:
            self._gamma_spin.blockSignals(True)
            self._gamma_spin.setValue(self._spec.gamma)
            self._gamma_spin.blockSignals(False)
        if self._visible_btn.isChecked() != self._spec.visible:
            self._visible_btn.blockSignals(True)
            self._visible_btn.setChecked(self._spec.visible)
            self._visible_btn.blockSignals(False)
        # Don't clobber an in-progress rename; sync only while the field is idle.
        if self._name_edit.isReadOnly():
            self._name_edit.set_name(self._spec.name)
        self._refresh_visible_btn()
        self._refresh_color_btn()
        self._refresh_value_label()

    def set_name(self, name: str) -> None:
        """Apply a (possibly deduplicated) name pushed back by the parent."""
        self._spec.name = name
        self._name_edit.set_name(name)

    def _refresh_visible_btn(self) -> None:
        self._visible_btn.setIcon(eye_icon(self._spec.visible))

    def _refresh_color_btn(self) -> None:
        self._color_btn.setStyleSheet(color_swatch_style(self._spec.color))

    def _refresh_value_label(self) -> None:
        self._value_label.setText(f"{round(self._spec.scale * 100.0)}%")

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
        self._refresh_value_label()
        spec = self.spec()
        self.changed.emit(self._index, spec)
        # Keyboard / programmatic changes don't go through sliderReleased,
        # so commit immediately when the slider isn't being dragged.
        if not self._slider.isSliderDown():
            self.committed.emit(self._index, spec)

    def _on_slider_released(self) -> None:
        self.committed.emit(self._index, self.spec())

    def _on_gamma(self, value: float) -> None:
        self._spec.gamma = float(value)
        spec = self.spec()
        self.changed.emit(self._index, spec)
        self.committed.emit(self._index, spec)

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
        self._layout.setSpacing(6)

        self._header = self._build_header()
        self._header.setVisible(False)
        self._layout.addWidget(self._header)

        self._empty_label = QLabel("No channels")
        self._empty_label.setStyleSheet("color: #888; font-style: italic;")
        self._layout.addWidget(self._empty_label)

    def _build_header(self) -> QWidget:
        """Column-title strip whose fixed widths line up over the row controls.

        Titles + rule read as a table header, so the rows below scan as columns
        rather than as three unrelated control clusters.
        """
        header = QWidget()
        vl = QVBoxLayout(header)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(3)

        titles = QHBoxLayout()
        titles.setContentsMargins(0, 0, 0, 0)
        titles.setSpacing(_ROW_SPACING)

        # Skip the eye column — it needs no title — so "Name" starts over the
        # name field rather than over the visibility toggle.
        eye_spacer = QWidget()
        eye_spacer.setFixedWidth(_EYE_W)
        titles.addWidget(eye_spacer)

        name = QLabel("Name")
        name.setStyleSheet(_HEADER_QSS)
        name.setFixedWidth(_NAME_W)
        # Matches the name field's 1px border + 4px padding, so the title sits
        # directly over the text inside it rather than over the field's edge.
        name.setIndent(5)
        titles.addWidget(name)

        # The colour swatch is self-evident and needs no title.
        color_spacer = QWidget()
        color_spacer.setFixedWidth(_COLOR_W)
        titles.addWidget(color_spacer)

        brightness = QLabel("Brightness")
        brightness.setStyleSheet(_HEADER_QSS)
        brightness.setAlignment(Qt.AlignmentFlag.AlignCenter)
        titles.addWidget(brightness, stretch=1)

        # Sits over the per-row percentage readout, which has no title of its own.
        value_spacer = QWidget()
        value_spacer.setFixedWidth(_VALUE_W)
        titles.addWidget(value_spacer)

        gamma = QLabel("Gamma")
        gamma.setStyleSheet(_HEADER_QSS)
        gamma.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gamma.setFixedWidth(_GAMMA_W)
        gamma.setToolTip("Gamma (1 = linear; <1 brightens shadows, >1 darkens)")
        titles.addWidget(gamma)

        vl.addLayout(titles)

        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        # Plain shadow draws a flat 1px line in the stylesheet's ``color``;
        # the default Sunken shadow would give it a 3D bevel.
        rule.setFrameShadow(QFrame.Shadow.Plain)
        rule.setFixedHeight(1)
        rule.setStyleSheet("color: #4d4d4d;")
        vl.addWidget(rule)

        return header

    def set_channels(self, channels: list[ChannelSpec]) -> None:
        new_specs = [
            ChannelSpec(
                name=c.name,
                color=c.color,
                scale=c.scale,
                gamma=c.gamma,
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
        self._header.setVisible(bool(self._channels))
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
                gamma=c.gamma,
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
