"""Dialog for selecting bad (excluded) sections before a DeepSlice run."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from verso.engine.model.project import ChannelSpec, Section
from verso.gui.utils import require

_CHANNEL_HINT_TOOLTIP = (
    "DeepSlice converts whatever it receives to greyscale, so VERSO sends the "
    "visible channels combined by a per-pixel maximum.\n\n"
    "Channel colour and brightness are display-only and do not affect the "
    "prediction — only visibility does.\n\n"
    "Hiding sparse signal channels, so DeepSlice sees the anatomical channel "
    "alone, usually gives the best alignment: most of DeepSlice's training data "
    "was single-channel anatomy."
)


def _channel_hint_text(channels: list[ChannelSpec]) -> str:
    """One-line summary of which channels this run will actually use."""
    if not channels:
        return "All channels will be combined to greyscale."
    visible = [c.name for c in channels if c.visible]
    if not visible:
        return "No channels are visible — all channels will be used."
    if len(visible) == 1:
        return f"Using visible channel: {visible[0]} — converted to greyscale."
    return f"Using visible channels: {', '.join(visible)} — combined to greyscale."


class BadSectionsDialog(QDialog):
    """Configure and launch a DeepSlice run.

    Lets the user mark bad sections (passed to ``DSModel.set_bad_sections()``)
    and declare whether section numbering runs posterior→anterior.

    Also surfaces which channels the run will use: ``visible`` is a real input to
    the alignment, not a display preference, so it should not be something the
    user has to know from reading the engine.
    """

    def __init__(
        self,
        sections: list[Section],
        channels: list[ChannelSpec] | None = None,
        reverse_order: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Run DeepSlice")
        self.setMinimumWidth(460)
        self.setMinimumHeight(460)

        self._checkboxes: list[tuple[str, QCheckBox]] = []  # (section_id, checkbox)

        layout = QVBoxLayout(self)

        # --- channel hint ---
        # Mildly highlighted, not a warning: this is context, not a problem.
        # Palette roles rather than literal colours so it tracks light/dark themes.
        self._channel_hint = QLabel(_channel_hint_text(list(channels or [])))
        self._channel_hint.setWordWrap(True)
        self._channel_hint.setToolTip(_CHANNEL_HINT_TOOLTIP)
        self._channel_hint.setStyleSheet(
            "QLabel { background: palette(alternate-base);"
            " border: 1px solid palette(mid);"
            " border-radius: 4px; padding: 6px; }"
        )
        layout.addWidget(self._channel_hint)

        # --- section order option ---
        # Unticked is VERSO's default direction (posterior → anterior as
        # slice_index increases); tick only when the series runs the other way.
        self._reverse_cb = QCheckBox("Sections are ordered anterior → posterior")
        self._reverse_cb.setToolTip(
            "Leave unticked when slice_index increases posterior → anterior "
            "(VERSO's default). Tick only if your series runs anterior → posterior."
        )
        self._reverse_cb.setChecked(reverse_order)
        layout.addWidget(self._reverse_cb)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # --- bad sections list ---
        info = QLabel(
            "Optionally check sections DeepSlice should treat as bad "
            "(damaged or artefacted slices excluded from angle propagation)."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self._list = QListWidget()
        self._list.setSpacing(2)
        layout.addWidget(self._list)

        for section in sections:
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(6, 3, 6, 3)
            row_layout.setSpacing(8)

            checkbox = QCheckBox()
            row_layout.addWidget(checkbox)

            thumb_label = QLabel()
            thumb_label.setFixedSize(64, 44)
            thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb_path = Path(section.thumbnail_path or "")
            if thumb_path.exists():
                pix = QPixmap(str(thumb_path)).scaled(
                    64,
                    44,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                thumb_label.setPixmap(pix)
            else:
                thumb_label.setText("—")
            row_layout.addWidget(thumb_label)

            name = Path(section.original_path).name
            text_label = QLabel(f"s{section.slice_index:03d}  {name}")
            text_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            row_layout.addWidget(text_label)

            item = QListWidgetItem()
            row_widget.adjustSize()
            item.setSizeHint(row_widget.sizeHint())
            self._list.addItem(item)
            self._list.setItemWidget(item, row_widget)

            self._checkboxes.append((section.id, checkbox))

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        require(buttons.button(QDialogButtonBox.StandardButton.Ok)).setText("Run DeepSlice")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def bad_section_ids(self) -> list[str]:
        """Return section IDs for checked (bad) items."""
        return [sid for sid, cb in self._checkboxes if cb.isChecked()]

    def reverse_section_order(self) -> bool:
        """Return True if sections are ordered posterior → anterior."""
        return self._reverse_cb.isChecked()
