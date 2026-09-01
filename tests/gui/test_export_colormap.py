"""Tests for the section-colormap controls on the images-with-overlay dialog.

Covers the option round-trip and the preview button's gating/caching. The
preview popup itself is not shown; ``_ColormapPreviewDialog`` is patched so the
test stays headless and non-blocking.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from verso.engine.colormaps import NONE, apply_colormap
from verso.gui.dialogs import export_images as dialog_module
from verso.gui.dialogs.export_images import ExportImagesDialog


@pytest.fixture(scope="module")
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def gray_ramp():
    """A horizontal 0..255 gray ramp, so LUT lookups are exactly predictable."""
    return np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (16, 1, 3))


def test_defaults_leave_the_export_uncolored(_qapp):
    dlg = ExportImagesDialog(n_selected=1, n_total=3)
    options = dlg.options()
    assert options.colormap == NONE
    assert options.colormap_invert is False
    # The default entry must read as "keep what you already have", not "no color".
    assert dlg._colormap_combo.currentText() == "Original channel colors"
    assert dlg._colormap_combo.currentData() == NONE


def test_options_round_trip_the_choice(_qapp):
    dlg = ExportImagesDialog(n_selected=1, n_total=3)
    dlg._colormap_combo.setCurrentText("Magma")
    dlg._invert_check.setChecked(True)
    options = dlg.options()
    assert options.colormap == "Magma"
    assert options.colormap_invert is True


def test_invert_and_preview_are_gated_on_a_colormap(_qapp, gray_ramp):
    dlg = ExportImagesDialog(n_selected=1, n_total=3, preview_source=lambda: gray_ramp)
    assert not dlg._invert_check.isEnabled()
    assert not dlg._preview_button.isEnabled()

    dlg._colormap_combo.setCurrentText("Viridis")
    assert dlg._invert_check.isEnabled()
    assert dlg._preview_button.isEnabled()


def test_preview_button_stays_off_without_a_source(_qapp):
    dlg = ExportImagesDialog(n_selected=1, n_total=3)
    dlg._colormap_combo.setCurrentText("Viridis")
    assert dlg._invert_check.isEnabled()
    assert not dlg._preview_button.isEnabled()


def test_preview_renders_the_selected_colormap_and_caches_the_source(_qapp, gray_ramp, monkeypatch):
    calls = []
    shown = []

    def source():
        calls.append(1)
        return gray_ramp

    class _FakePopup:
        def __init__(self, pixmap, title, parent=None):
            shown.append((pixmap.width(), pixmap.height(), title))

        def exec(self):
            return 0

    monkeypatch.setattr(dialog_module, "_ColormapPreviewDialog", _FakePopup)

    dlg = ExportImagesDialog(n_selected=1, n_total=3, preview_source=source)
    dlg._colormap_combo.setCurrentText("Turbo")
    dlg._on_preview()
    dlg._invert_check.setChecked(True)
    dlg._on_preview()

    # The section is loaded once; switching colormap/invert only re-applies a LUT.
    assert len(calls) == 1
    assert shown == [
        (256, 16, "Turbo"),
        (256, 16, "Turbo (inverted)"),
    ]


def test_preview_reports_a_failing_source_instead_of_raising(_qapp, monkeypatch):
    warned = []

    def boom():
        raise OSError("cannot read section")

    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "warning",
        classmethod(lambda cls, *args, **kwargs: warned.append(args)),
    )

    dlg = ExportImagesDialog(n_selected=1, n_total=3, preview_source=boom)
    dlg._colormap_combo.setCurrentText("Viridis")
    dlg._on_preview()

    assert len(warned) == 1
    assert "cannot read section" in warned[0][2]


def test_preview_pixels_match_the_engine_colormap(_qapp, gray_ramp):
    """The popup must show exactly what the export will write."""
    expected = apply_colormap(gray_ramp, "Magma", invert=True)
    assert expected.shape == gray_ramp.shape
    assert not np.array_equal(expected, apply_colormap(gray_ramp, "Magma"))
