"""Column-mapping dialog for importing annotation points from a CSV.

Shown only when :func:`~verso.engine.io.annotation_io.guess_point_columns` cannot
confidently resolve the required ``x``/``y`` columns. The user maps the CSV
headers to x, y and an optional image column; picking "(current section)" for the
image assigns every imported point to the section currently open.
"""

from __future__ import annotations

from collections.abc import Sequence

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QVBoxLayout,
    QWidget,
)

_CURRENT_SECTION_LABEL = "(current section)"


class AnnotationCsvDialog(QDialog):
    """Let the user map CSV columns to x, y and an optional image column."""

    def __init__(
        self,
        headers: Sequence[str],
        guess: dict[str, str | None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import points from CSV")
        self._headers = list(headers)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self._x_combo = QComboBox()
        self._y_combo = QComboBox()
        self._x_combo.addItems(self._headers)
        self._y_combo.addItems(self._headers)
        self._image_combo = QComboBox()
        self._image_combo.addItem(_CURRENT_SECTION_LABEL)
        self._image_combo.addItems(self._headers)

        self._preselect(self._x_combo, guess.get("x"))
        self._preselect(self._y_combo, guess.get("y"))
        if guess.get("image") is not None:
            self._preselect(self._image_combo, guess["image"])

        form.addRow("X column:", self._x_combo)
        form.addRow("Y column:", self._y_combo)
        form.addRow("Image column:", self._image_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _preselect(self, combo: QComboBox, value: str | None) -> None:
        if value is None:
            return
        idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def result_columns(self) -> tuple[str, str, str | None]:
        """Return (x_col, y_col, image_col) — image_col is None for current section."""
        image = self._image_combo.currentText()
        image_col = None if image == _CURRENT_SECTION_LABEL else image
        return self._x_combo.currentText(), self._y_combo.currentText(), image_col
