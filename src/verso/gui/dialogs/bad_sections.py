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

from verso.engine.model.project import Section
from verso.gui.utils import require


class BadSectionsDialog(QDialog):
    """Configure and launch a DeepSlice run.

    Lets the user mark bad sections (passed to ``DSModel.set_bad_sections()``)
    and declare whether section numbering runs posterior→anterior.
    """

    def __init__(
        self,
        sections: list[Section],
        reverse_order: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Run DeepSlice")
        self.setMinimumWidth(460)
        self.setMinimumHeight(460)

        self._checkboxes: list[tuple[str, QCheckBox]] = []  # (section_id, checkbox)

        layout = QVBoxLayout(self)

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
