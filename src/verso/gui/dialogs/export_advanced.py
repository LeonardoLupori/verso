"""Advanced tonal controls for the images-with-overlay export.

Opened from the export dialog's "Advanced…" button. Shows one section rendered
at preview size and lets the user set a black point, white point and gamma
against it, with the chosen colormap applied so the preview matches the export.

The levels can also be picked straight off the image — click an empty area and
its local level moves whichever end of the window the background belongs to:
the minimum for a dark-background fluorescence section, the maximum for a
bright-background brightfield one.

Everything here works on the parent dialog's cached preview RGB, so every
adjustment is a LUT rebuild rather than a re-read of the section.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from verso.engine.colormaps import NONE as COLORMAP_NONE
from verso.engine.colormaps import apply_colormap
from verso.engine.levels import (
    MAX_GAMMA,
    MIN_GAMMA,
    Levels,
    apply_levels,
    is_bright_background,
    sample_background_level,
)
from verso.gui.utils import ndarray_to_pixmap

#: Longest side the preview is displayed at. The source RGB stays full preview
#: size; only the on-screen pixmap is scaled, and clicks are mapped back.
_DISPLAY_MAX_SIDE = 520

#: Gamma slider steps per unit, so the integer slider can carry 0.01 precision.
_GAMMA_STEPS = 100


class ExportAdvancedDialog(QDialog):
    """Collect a :class:`~verso.engine.levels.Levels` against a live preview."""

    def __init__(
        self,
        preview_rgb: np.ndarray,
        levels: Levels,
        *,
        colormap: str = COLORMAP_NONE,
        colormap_invert: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        """
        Args:
            preview_rgb: uint8 ``(H, W, 3)`` un-adjusted section render.
            levels: The adjustment to open with (the parent dialog's current one).
            colormap: Colormap name applied after the levels, for the preview.
            colormap_invert: Whether that colormap is reversed.
            parent: Owning widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Advanced image adjustments")
        self.setModal(True)

        self._source = preview_rgb
        self._colormap = colormap
        self._colormap_invert = colormap_invert
        self._picking = False

        start = levels.normalized()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        # --- Preview ---------------------------------------------------
        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # The label receives the clicks that drive the background picker.
        self._preview_label.installEventFilter(self)
        layout.addWidget(self._preview_label)

        # --- Levels ----------------------------------------------------
        levels_box = QGroupBox("Levels")
        grid = QGridLayout(levels_box)

        self._black_slider, self._black_value = self._add_slider_row(
            grid, 0, "Minimum:", 0, 255, start.black
        )
        self._white_slider, self._white_value = self._add_slider_row(
            grid, 1, "Maximum:", 0, 255, start.white
        )
        self._gamma_slider, self._gamma_value = self._add_slider_row(
            grid,
            2,
            "Gamma:",
            round(MIN_GAMMA * _GAMMA_STEPS),
            round(MAX_GAMMA * _GAMMA_STEPS),
            round(start.gamma * _GAMMA_STEPS),
        )
        layout.addWidget(levels_box)

        # --- Background picker / reset ---------------------------------
        tools = QHBoxLayout()
        self._pick_button = QPushButton("Pick background…")
        self._pick_button.setCheckable(True)
        self._pick_button.setToolTip(
            "Click an empty area of the preview to set the levels from the background."
        )
        self._pick_button.toggled.connect(self._on_pick_toggled)
        reset_button = QPushButton("Reset")
        reset_button.setToolTip("Back to minimum 0, maximum 255, gamma 1.00.")
        reset_button.clicked.connect(self._on_reset)
        tools.addWidget(self._pick_button)
        tools.addWidget(reset_button)
        tools.addStretch()
        layout.addLayout(tools)

        self._hint = QLabel("")
        self._hint.setStyleSheet("color: #999;")
        layout.addWidget(self._hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        for slider in (self._black_slider, self._white_slider, self._gamma_slider):
            slider.valueChanged.connect(self._on_levels_changed)
        self._on_levels_changed()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _add_slider_row(
        grid: QGridLayout, row: int, label: str, lo: int, hi: int, value: int
    ) -> tuple[QSlider, QLabel]:
        """Add a ``label / slider / readout`` row and return the slider + readout."""
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(value)
        readout = QLabel()
        readout.setMinimumWidth(44)
        readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(slider, row, 1)
        grid.addWidget(readout, row, 2)
        return slider, readout

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def levels(self) -> Levels:
        """Return the adjustment the sliders currently describe."""
        return Levels(
            black=self._black_slider.value(),
            white=self._white_slider.value(),
            gamma=self._gamma_slider.value() / _GAMMA_STEPS,
        ).normalized()

    def _rendered(self) -> np.ndarray:
        """The preview RGB with the current levels and colormap applied."""
        rgb = apply_levels(self._source, self.levels())
        return apply_colormap(rgb, self._colormap, invert=self._colormap_invert)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_levels_changed(self) -> None:
        """Keep the two ends ordered, refresh the readouts, redraw the preview."""
        # The window must stay open, so each end pushes the other rather than
        # crossing it. Signals are blocked to avoid re-entering this slot.
        if self._white_slider.value() <= self._black_slider.value():
            other = (
                self._white_slider if self.sender() is self._black_slider else self._black_slider
            )
            if other is self._white_slider:
                self._set_silently(other, min(255, self._black_slider.value() + 1))
            else:
                self._set_silently(other, max(0, self._white_slider.value() - 1))

        current = self.levels()
        self._black_value.setText(str(current.black))
        self._white_value.setText(str(current.white))
        self._gamma_value.setText(f"{current.gamma:.2f}")
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        pixmap = ndarray_to_pixmap(self._rendered())
        if max(pixmap.width(), pixmap.height()) > _DISPLAY_MAX_SIDE:
            pixmap = pixmap.scaled(
                _DISPLAY_MAX_SIDE,
                _DISPLAY_MAX_SIDE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._preview_label.setPixmap(pixmap)

    def _on_pick_toggled(self, checked: bool) -> None:
        self._picking = checked
        self._preview_label.setCursor(
            Qt.CursorShape.CrossCursor if checked else Qt.CursorShape.ArrowCursor
        )
        self._hint.setText(
            "Click an empty area of the preview to set the levels from the background."
            if checked
            else ""
        )

    def _on_reset(self) -> None:
        for slider, value in (
            (self._black_slider, 0),
            (self._white_slider, 255),
            (self._gamma_slider, _GAMMA_STEPS),
        ):
            self._set_silently(slider, value)
        self._pick_button.setChecked(False)
        self._on_levels_changed()

    # ------------------------------------------------------------------
    # Background picking
    # ------------------------------------------------------------------
    def eventFilter(self, obj, event) -> bool:
        """Turn a click on the preview into a black-point sample while picking."""
        if (
            self._picking
            and obj is self._preview_label
            and event.type() == QEvent.Type.MouseButtonPress
        ):
            position = event.position()
            self._pick_at(position.x(), position.y())
            return True
        return super().eventFilter(obj, event)

    def _pick_at(self, label_x: float, label_y: float) -> None:
        """Map a click in label coordinates to a source pixel and sample it."""
        pixmap = self._preview_label.pixmap()
        if pixmap is None or pixmap.isNull():
            return
        src_h, src_w = self._source.shape[:2]

        # The pixmap is centred in the label and may be scaled down, so undo
        # both before indexing the source array.
        offset_x = (self._preview_label.width() - pixmap.width()) / 2.0
        offset_y = (self._preview_label.height() - pixmap.height()) / 2.0
        x = (label_x - offset_x) * src_w / pixmap.width()
        y = (label_y - offset_y) * src_h / pixmap.height()
        if not (0 <= x < src_w and 0 <= y < src_h):
            self._hint.setText("That click was outside the image — try again.")
            return

        value = sample_background_level(self._source, int(x), int(y))
        # Disarm first: un-checking clears the hint, which would wipe the message
        # _apply_picked_level is about to leave there.
        self._pick_button.setChecked(False)
        self._apply_picked_level(value)

    def _apply_picked_level(self, value: int) -> None:
        """Move the end of the window the picked background actually belongs to.

        On a fluorescence section the background is the dark end, so it sets the
        minimum. On a brightfield one it is the *bright* end: forcing it into the
        minimum there would put the black point above the stain and crush the
        whole section, so it sets the maximum instead. The hint names whichever
        end moved, since the button cannot say in advance.
        """
        if is_bright_background(self._source, value):
            target, other = self._white_slider, self._black_slider
            if value <= other.value():
                self._set_silently(other, max(0, value - 1))
            label = "Maximum"
        else:
            target, other = self._black_slider, self._white_slider
            if value >= other.value():
                self._set_silently(other, min(255, value + 1))
            label = "Minimum"
        target.setValue(value)
        self._hint.setText(f"{label} set to {value} from the background.")

    @staticmethod
    def _set_silently(slider: QSlider, value: int) -> None:
        """Move *slider* without re-entering :meth:`_on_levels_changed`."""
        blocked = slider.blockSignals(True)
        slider.setValue(value)
        slider.blockSignals(blocked)
