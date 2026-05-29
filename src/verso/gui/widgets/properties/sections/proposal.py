"""Proposal info section (Align/Warp views)."""

from __future__ import annotations

from PyQt6.QtWidgets import QFormLayout, QGroupBox, QLabel

from verso.engine.model.project import Section


class ProposalBox(QGroupBox):
    def __init__(self) -> None:
        super().__init__("Proposal")
        layout = QFormLayout(self)
        self._source = QLabel("-")
        self._source.setWordWrap(True)
        self._confidence = QLabel("-")
        layout.addRow("Source:", self._source)
        layout.addRow("Confidence:", self._confidence)

    def update_section(self, section: Section | None) -> None:
        if section is None:
            self._source.setText("-")
            self._confidence.setText("-")
            return
        source = section.alignment.source
        labels = {
            "deepslice": "DeepSlice suggestion",
            "quicknii_default": "Default proposal",
            "manual": "Manual edit",
        }
        self._source.setText(labels.get(source, "-"))
        confidence = section.alignment.proposal_confidence
        self._confidence.setText("-" if confidence is None else f"{confidence:.3f}")
