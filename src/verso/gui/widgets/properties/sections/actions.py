"""Align actions section — store / revert / clear / clear-all buttons."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QGridLayout, QGroupBox, QPushButton


class AlignActionsBox(QGroupBox):
    store_requested = pyqtSignal()
    revert_requested = pyqtSignal()
    clear_requested = pyqtSignal()
    clear_all_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__("Actions")

        self._store_btn = QPushButton("Store")
        self._store_btn.setToolTip("Lock current atlas plane to this section")
        self._store_btn.setEnabled(False)
        self._store_btn.clicked.connect(self.store_requested)

        self._revert_btn = QPushButton("Revert")
        self._revert_btn.setToolTip(
            "Restore the last stored plane, discarding unsaved edits"
        )
        self._revert_btn.setEnabled(False)
        self._revert_btn.clicked.connect(self.revert_requested)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setToolTip("Remove stored plane and revert to interpolated")
        self._clear_btn.setEnabled(False)
        self._clear_btn.clicked.connect(self.clear_requested)

        self._clear_all_btn = QPushButton("Clear all")
        self._clear_all_btn.setToolTip(
            "Clear every stored alignment and restore the default AP proposal"
        )
        self._clear_all_btn.setEnabled(False)
        self._clear_all_btn.clicked.connect(self.clear_all_requested)

        layout = QGridLayout(self)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(6)
        layout.addWidget(self._store_btn, 0, 0)
        layout.addWidget(self._revert_btn, 0, 1)
        layout.addWidget(self._clear_btn, 1, 0)
        layout.addWidget(self._clear_all_btn, 1, 1)

    def set_store_enabled(self, enabled: bool) -> None:
        self._store_btn.setEnabled(enabled)

    def set_revert_enabled(self, enabled: bool) -> None:
        self._revert_btn.setEnabled(enabled)

    def set_clear_enabled(self, enabled: bool) -> None:
        self._clear_btn.setEnabled(enabled)

    def set_clear_all_enabled(self, enabled: bool) -> None:
        self._clear_all_btn.setEnabled(enabled)
