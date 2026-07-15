"""warn_if_missing_dimensions: guards atlas-registration ops on corrupt projects."""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget

from verso.engine.model.project import Section
from verso.gui.utils import warn_if_missing_dimensions


@pytest.fixture(scope="module")
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _section(sid: str, idx: int, wh: tuple[int, int]) -> Section:
    return Section(sid, idx, f"{sid}.png", f"{sid}.png", resolution_thumbnail_wh=wh)


def test_returns_true_and_shows_no_dialog_when_all_dimensions_present(_qapp, monkeypatch):
    calls: list = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: calls.append(a))

    sections = [_section("s001", 1, (100, 80)), _section("s002", 2, (100, 80))]
    assert warn_if_missing_dimensions(QWidget(), sections) is True
    assert calls == []


def test_returns_false_and_names_offending_slice_and_attribute(_qapp, monkeypatch):
    shown: dict[str, str] = {}

    def fake_critical(_parent, title, text):
        shown["title"] = title
        shown["text"] = text

    monkeypatch.setattr(QMessageBox, "critical", fake_critical)

    sections = [_section("s001", 1, (100, 80)), _section("s017", 17, (0, 0))]
    assert warn_if_missing_dimensions(QWidget(), sections) is False
    assert shown["title"] == "Project may be corrupt"
    # Points at the specific slice and the missing attribute.
    assert "slice 17" in shown["text"]
    assert "id s017" in shown["text"]
    assert "resolution_thumbnail_wh" in shown["text"]
    # The healthy section is not listed.
    assert "id s001" not in shown["text"]
