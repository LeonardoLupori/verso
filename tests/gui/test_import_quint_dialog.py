"""ImportQuintDialog: JSON parsing, image matching, and OK-gating logic.

The end-to-end project build is covered by ``tests/engine/test_quint_import.py``;
these tests exercise the dialog's own wiring without touching the atlas download
or thumbnail generation (``_on_accept``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from PyQt6.QtWidgets import QApplication, QDialogButtonBox, QFileDialog

from verso.gui.dialogs.import_quint import ImportQuintDialog


@pytest.fixture(scope="module")
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _png(path: Path, size: tuple[int, int]) -> None:
    Image.new("RGB", size).save(path)


def _write_json(tmp_path: Path, target: str = "allen_mouse_25um") -> Path:
    data = {
        "name": "imported",
        "target": target,
        "slices": [
            {
                "filename": "IMG_0001.png",
                "nr": 1,
                "width": 1000,
                "height": 800,
                "anchoring": [0.0, 160.0, 228.0, 456.0, 0.0, 0.0, 0.0, 320.0, 0.0],
            },
            {
                "filename": "IMG_0002.png",
                "nr": 2,
                "width": 1000,
                "height": 800,
                "anchoring": [0.0, 160.0, 250.0, 456.0, 0.0, 0.0, 0.0, 320.0, 0.0],
            },
        ],
    }
    path = tmp_path / "va.json"
    path.write_text(json.dumps(data))
    return path


def _ok_enabled(dlg: ImportQuintDialog) -> bool:
    btn = dlg._buttons.button(QDialogButtonBox.StandardButton.Ok)
    return btn is not None and btn.isEnabled()


def test_load_json_populates_table_and_fixes_known_atlas(_qapp, tmp_path: Path):
    dlg = ImportQuintDialog()
    dlg._load_json(_write_json(tmp_path))

    assert dlg._table.rowCount() == 2
    assert dlg._atlas_combo.currentText() == "allen_mouse_25um"
    assert not dlg._atlas_combo.isEnabled()  # known target fixes the atlas
    assert not dlg._atlas_warning.isVisibleTo(dlg)
    assert not _ok_enabled(dlg)  # images not matched yet


def test_unknown_target_enables_atlas_picker_and_warns(_qapp, tmp_path: Path):
    dlg = ImportQuintDialog()
    dlg._load_json(_write_json(tmp_path, target="ABA_Custom.cutlas"))

    assert dlg._atlas_combo.isEnabled()
    assert dlg._atlas_warning.isVisibleTo(dlg)


def test_ok_enables_once_all_matched(_qapp, tmp_path: Path, monkeypatch):
    json_path = _write_json(tmp_path)
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    _png(imgs / "IMG_0001.png", (1000, 800))
    _png(imgs / "IMG_0002.png", (1000, 800))

    dlg = ImportQuintDialog()
    dlg._load_json(json_path)
    dlg._name_edit.setText("Proj")
    dlg._location_edit.setText(str(tmp_path))
    assert not _ok_enabled(dlg)  # folder not chosen yet

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(imgs))
    dlg._browse_registration()

    assert _ok_enabled(dlg)  # reuse default → originals are the registration images


def test_missing_image_gates_ok_until_manual_assign(_qapp, tmp_path: Path, monkeypatch):
    json_path = _write_json(tmp_path)
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    _png(imgs / "IMG_0001.png", (1000, 800))  # IMG_0002 deliberately absent

    dlg = ImportQuintDialog()
    dlg._load_json(json_path)
    dlg._name_edit.setText("Proj")
    dlg._location_edit.setText(str(tmp_path))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(imgs))
    dlg._browse_registration()
    assert not _ok_enabled(dlg)  # one slice unmatched

    other = tmp_path / "renamed_scan.png"
    _png(other, (1000, 800))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(other), ""))
    dlg._on_cell_double_clicked(1, 2)  # row 1 (nr=2), Registration column

    assert dlg._reg[1] == other
    assert _ok_enabled(dlg)


def test_reuse_toggle_reveals_originals_row(_qapp, tmp_path: Path):
    dlg = ImportQuintDialog()
    dlg._load_json(_write_json(tmp_path))

    assert not dlg._orig_row.isVisibleTo(dlg)  # hidden while reusing registration images
    dlg._reuse_check.setChecked(False)
    assert dlg._orig_row.isVisibleTo(dlg)
