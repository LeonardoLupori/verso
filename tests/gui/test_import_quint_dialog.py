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
from PyQt6.QtGui import QColor
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
    # Isolate the JSON in its own folder so the browse-flow tests are not
    # perturbed by the on-load auto-detection of co-located registration images.
    align = tmp_path / "align"
    align.mkdir(exist_ok=True)
    path = align / "va.json"
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


def _thumbnail_json(tmp_path: Path) -> Path:
    """Write a 2-section thumbnail-style VisuAlign JSON with its sibling thumbnails."""
    align = tmp_path / "align"
    align.mkdir()
    data = {
        "name": "imported",
        "target": "allen_mouse_25um",
        "slices": [
            {
                "filename": "thumbnails/IMG_0001-thumb.png",
                "nr": 1,
                "width": 1000,
                "height": 800,
                "anchoring": [0.0] * 9,
            },
            {
                "filename": "thumbnails/IMG_0002-thumb.png",
                "nr": 2,
                "width": 1000,
                "height": 800,
                "anchoring": [0.0] * 9,
            },
        ],
    }
    json_path = align / "va.json"
    json_path.write_text(json.dumps(data))
    thumbs = align / "thumbnails"
    thumbs.mkdir()
    _png(thumbs / "IMG_0001-thumb.png", (1000, 800))
    _png(thumbs / "IMG_0002-thumb.png", (1000, 800))
    return json_path


def test_thumbnail_json_autodetects_reg_and_matches_originals_by_similarity(
    _qapp, tmp_path, monkeypatch
):
    """Thumbnails auto-resolve as registration; originals are added as files and fuzzy-matched."""
    json_path = _thumbnail_json(tmp_path)
    dlg = ImportQuintDialog()
    dlg._load_json(json_path)

    # Thumbnails auto-resolved as registration images…
    assert set(dlg._reg) == {0, 1}
    assert dlg._reg_edit.text() == str(json_path.parent)
    # …and, being thumbnails, reuse is off and the originals row is revealed.
    assert not dlg._reuse_check.isChecked()
    assert dlg._orig_row.isVisibleTo(dlg)
    dlg._name_edit.setText("Proj")
    dlg._location_edit.setText(str(tmp_path))
    assert not _ok_enabled(dlg)  # blocked until originals are added

    # Add originals as individual files (New-Project style), shuffled; names differ
    # from the thumbnails only by the -thumb suffix and extension.
    originals = tmp_path / "originals"
    originals.mkdir()
    o1 = originals / "IMG_0001.tif"
    o2 = originals / "IMG_0002.tif"
    _png(o1, (4000, 3200))
    _png(o2, (4000, 3200))
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *a, **k: ([str(o2), str(o1)], ""))
    dlg._add_originals()

    assert dlg._orig[0].name == "IMG_0001.tif"  # fuzzy-matched to the right section
    assert dlg._orig[1].name == "IMG_0002.tif"
    assert _ok_enabled(dlg)


def test_unset_originals_are_calm_not_alarming(_qapp, tmp_path):
    """Before any originals are chosen the Full-res column is neutral, not a warning."""
    json_path = _thumbnail_json(tmp_path)
    dlg = ImportQuintDialog()
    dlg._load_json(json_path)
    dlg._name_edit.setText("Proj")
    dlg._location_edit.setText(str(tmp_path))

    cell = dlg._table.item(0, 3)  # column 3 = Full-res
    assert cell.text() == "—"  # calm placeholder, not "⚠ double-click to locate…"
    assert cell.foreground().color() != QColor("red")
    status = dlg._status_label.text().lower()
    assert "unmatched" not in status
    assert "add the full-resolution images" in status


def test_manual_original_override_moves_file_and_stays_unique(_qapp, tmp_path, monkeypatch):
    """Double-clicking a Full-res cell pins a file to that section, keeping 1:1."""
    json_path = _thumbnail_json(tmp_path)
    dlg = ImportQuintDialog()
    dlg._load_json(json_path)

    a = tmp_path / "IMG_0001.tif"
    b = tmp_path / "IMG_0002.tif"
    _png(a, (10, 10))
    _png(b, (10, 10))
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *args, **k: ([str(a), str(b)], ""))
    dlg._add_originals()
    assert dlg._orig[0] == a and dlg._orig[1] == b

    # Pin A to section 1 → A leaves section 0, and B re-fills the freed section 0.
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **k: (str(a), ""))
    dlg._on_cell_double_clicked(1, 3)  # column 3 = Full-res

    assert dlg._orig[1] == a
    assert dlg._orig.get(0) == b
    assert len({str(p) for p in dlg._orig.values()}) == len(dlg._orig)  # unique files


def test_load_quicknii_xml_alignment(_qapp, tmp_path: Path):
    """The dialog reads a QuickNII .xml alignment, not only JSON."""
    xml = tmp_path / "AL1A-quicknii.xml"
    xml.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<series name="AL1A" target="ABA_Mouse_CCFv3_2017_25um.cutlas">\n'
        '  <slice filename="IMG_0001.png" nr="1" width="1000" height="800" '
        'anchoring="ox=0&amp;oy=160&amp;oz=228&amp;ux=456&amp;uy=0&amp;uz=0&amp;'
        'vx=0&amp;vy=0&amp;vz=320"/>\n'
        '  <slice filename="IMG_0002.png" nr="2" width="1000" height="800" '
        'anchoring="ox=0&amp;oy=160&amp;oz=250&amp;ux=456&amp;uy=0&amp;uz=0&amp;'
        'vx=0&amp;vy=0&amp;vz=320"/>\n'
        "</series>\n"
    )

    dlg = ImportQuintDialog()
    dlg._load_json(xml)

    assert dlg._table.rowCount() == 2  # both slices parsed from XML
    assert dlg._name_edit.text() == "AL1A"
    assert dlg._orientation_combo.currentData() == "coronal"  # inferred from anchoring


def test_load_json_presets_slicing_orientation(_qapp, tmp_path: Path):
    """The orientation combo is preset from the anchoring geometry (here: sagittal)."""
    data = {
        "name": "s",
        "target": "allen_mouse_25um",
        "slices": [
            {
                "filename": "IMG_0001.png",
                "nr": 1,
                "width": 1000,
                "height": 800,
                "anchoring": [0.0, 100.0, 100.0, 0.0, 300.0, 0.0, 0.0, 0.0, 300.0],
            },
        ],
    }
    json_path = tmp_path / "sag.json"
    json_path.write_text(json.dumps(data))

    dlg = ImportQuintDialog()
    dlg._load_json(json_path)

    assert dlg._orientation_combo.currentData() == "sagittal"


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
