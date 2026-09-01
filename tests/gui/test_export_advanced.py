"""Tests for the advanced (levels) export dialog and its hand-back to the parent.

The dialogs are never shown; the sliders and the picker are driven directly, so
these stay headless. ``_pick_at`` is called with label coordinates to exercise
the scaled-pixmap → source-pixel mapping without synthesising mouse events.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QDialog

from verso.engine.levels import Levels, apply_levels
from verso.gui.dialogs import export_images as parent_module
from verso.gui.dialogs.export_advanced import ExportAdvancedDialog
from verso.gui.dialogs.export_images import ExportImagesDialog


@pytest.fixture(scope="module")
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _label_point(dlg, u: float, v: float) -> tuple[float, float]:
    """Convert normalized source coordinates to a click point in label space.

    The pixmap is centred in the (larger) label and may be scaled down, so a
    test that wants to click a particular *image* pixel has to go through the
    same offset + scale the dialog undoes.
    """
    pixmap = dlg._preview_label.pixmap()
    offset_x = (dlg._preview_label.width() - pixmap.width()) / 2.0
    offset_y = (dlg._preview_label.height() - pixmap.height()) / 2.0
    return offset_x + u * pixmap.width(), offset_y + v * pixmap.height()


@pytest.fixture
def preview():
    """A 200x120 image: uniform value-20 background with a bright blob."""
    rgb = np.full((120, 200, 3), 20, dtype=np.uint8)
    rgb[40:80, 60:140] = 200
    return rgb


def test_opens_with_the_levels_it_was_given(_qapp, preview):
    dlg = ExportAdvancedDialog(preview, Levels(black=30, white=220, gamma=1.5))
    assert dlg._black_slider.value() == 30
    assert dlg._white_slider.value() == 220
    assert dlg.levels() == Levels(black=30, white=220, gamma=1.5)


def test_sliders_round_trip_through_levels(_qapp, preview):
    dlg = ExportAdvancedDialog(preview, Levels())
    dlg._black_slider.setValue(64)
    dlg._white_slider.setValue(192)
    dlg._gamma_slider.setValue(250)  # 2.50 at 100 steps per unit
    got = dlg.levels()
    assert (got.black, got.white) == (64, 192)
    assert got.gamma == pytest.approx(2.5)


def test_readouts_follow_the_sliders(_qapp, preview):
    dlg = ExportAdvancedDialog(preview, Levels())
    dlg._black_slider.setValue(77)
    dlg._gamma_slider.setValue(180)
    assert dlg._black_value.text() == "77"
    assert dlg._gamma_value.text() == "1.80"


def test_minimum_pushes_the_maximum_instead_of_crossing_it(_qapp, preview):
    dlg = ExportAdvancedDialog(preview, Levels())
    dlg._white_slider.setValue(100)
    dlg._black_slider.setValue(150)
    assert dlg._white_slider.value() > dlg._black_slider.value()
    assert dlg.levels().white > dlg.levels().black


def test_maximum_pushes_the_minimum_instead_of_crossing_it(_qapp, preview):
    dlg = ExportAdvancedDialog(preview, Levels())
    dlg._black_slider.setValue(150)
    dlg._white_slider.setValue(100)
    assert dlg._white_slider.value() > dlg._black_slider.value()


def test_reset_returns_to_the_identity(_qapp, preview):
    dlg = ExportAdvancedDialog(preview, Levels(black=40, white=180, gamma=2.0))
    dlg._on_reset()
    assert dlg.levels().is_identity
    assert dlg._black_value.text() == "0"
    assert dlg._gamma_value.text() == "1.00"


def test_preview_pixmap_tracks_the_sliders(_qapp, preview):
    dlg = ExportAdvancedDialog(preview, Levels())
    before = dlg._preview_label.pixmap().toImage()
    dlg._black_slider.setValue(120)
    after = dlg._preview_label.pixmap().toImage()
    assert before != after


def test_rendered_matches_the_engine_pipeline(_qapp, preview):
    """What the dialog draws must be what the exporter will write."""
    dlg = ExportAdvancedDialog(preview, Levels(black=25, white=210, gamma=1.2))
    np.testing.assert_array_equal(dlg._rendered(), apply_levels(preview, dlg.levels()))


def test_picking_the_background_sets_the_minimum(_qapp, preview):
    dlg = ExportAdvancedDialog(preview, Levels())
    dlg._pick_button.setChecked(True)
    assert dlg._picking

    # Near the top-left of the image: background, value 20.
    dlg._pick_at(*_label_point(dlg, 0.02, 0.02))
    assert dlg.levels().black == 20
    # The picker is a one-shot tool; it disarms after a successful sample.
    assert not dlg._pick_button.isChecked()
    assert "20" in dlg._hint.text()


def test_picking_maps_through_a_scaled_pixmap(_qapp):
    """A preview wider than the display cap is scaled; clicks must be unmapped."""
    rgb = np.full((400, 1040, 3), 30, dtype=np.uint8)
    rgb[:, 520:] = 90  # right half brighter
    dlg = ExportAdvancedDialog(rgb, Levels())
    pixmap = dlg._preview_label.pixmap()
    assert pixmap.width() < rgb.shape[1]  # actually scaled down

    # Three-quarters across the image lands in the bright right half. Here 90 is
    # that image's bright end, so it lands on the maximum — what this test pins
    # is that the *sampled value* came from the right half (90) and not the left
    # (30), i.e. that the click was unmapped through the scale correctly.
    dlg._pick_button.setChecked(True)
    dlg._pick_at(*_label_point(dlg, 0.75, 0.5))
    assert dlg.levels().white == 90

    # The mirror case: a quarter across samples the left half.
    dlg._on_reset()
    dlg._pick_button.setChecked(True)
    dlg._pick_at(*_label_point(dlg, 0.25, 0.5))
    assert dlg.levels().black == 30


def test_picking_a_bright_background_sets_the_maximum(_qapp):
    """Brightfield regression: a bright background is the *maximum*, not the minimum.

    Forcing it into the minimum put the black point above the stain and crushed
    the whole section to black.
    """
    rgb = np.full((120, 200, 3), 245, dtype=np.uint8)
    rgb[40:80, 60:140] = 40  # dark stain on a bright field
    dlg = ExportAdvancedDialog(rgb, Levels())
    dlg._pick_button.setChecked(True)
    dlg._pick_at(*_label_point(dlg, 0.02, 0.02))

    got = dlg.levels()
    assert got.white == 245
    assert got.black < got.white
    assert "Maximum" in dlg._hint.text()

    # The stain must survive: it is what the user wants to see.
    rendered = dlg._rendered()
    assert rendered[60, 100].max() < 60  # stain stays dark
    assert (rendered.max(axis=2) == 0).mean() < 0.5  # not crushed to black


def test_picking_a_dark_background_still_sets_the_minimum(_qapp, preview):
    """Fluorescence keeps the original behaviour."""
    dlg = ExportAdvancedDialog(preview, Levels())
    dlg._pick_button.setChecked(True)
    dlg._pick_at(*_label_point(dlg, 0.02, 0.02))
    got = dlg.levels()
    assert got.black == 20
    assert got.white == 255
    assert "Minimum" in dlg._hint.text()


def test_picking_uses_the_tinted_channel_level(_qapp):
    """A green-tinted background must be read from green, not an RGB average."""
    rgb = np.zeros((120, 200, 3), dtype=np.uint8)
    rgb[:, :, 1] = 40
    rgb[40:80, 60:140, 1] = 220
    dlg = ExportAdvancedDialog(rgb, Levels())
    dlg._pick_button.setChecked(True)
    dlg._pick_at(*_label_point(dlg, 0.02, 0.02))
    assert dlg.levels().black == 40  # not 13, the RGB mean
    # And the background is genuinely gone.
    assert dlg._rendered()[2, 2].max() == 0


def test_picking_keeps_the_window_ordered(_qapp):
    """A dark pick brighter than the current maximum must still leave min < max."""
    rgb = np.full((60, 60, 3), 90, dtype=np.uint8)
    rgb[10:50, 10:50] = 255  # bright signal, so 90 reads as the dark end
    dlg = ExportAdvancedDialog(rgb, Levels(black=0, white=50))
    dlg._pick_button.setChecked(True)
    dlg._pick_at(*_label_point(dlg, 0.02, 0.02))
    got = dlg.levels()
    assert got.black == 90
    assert got.white > got.black


def test_click_outside_the_image_is_reported_not_applied(_qapp, preview):
    dlg = ExportAdvancedDialog(preview, Levels())
    dlg._pick_button.setChecked(True)
    dlg._pick_at(*_label_point(dlg, -1.5, -1.5))
    assert dlg.levels().is_identity
    assert dlg._pick_button.isChecked()  # still armed, nothing was sampled
    assert "outside" in dlg._hint.text()


# ----------------------------------------------------------------------
# Hand-back to the main export dialog
# ----------------------------------------------------------------------


def test_parent_defaults_to_identity_levels(_qapp):
    dlg = ExportImagesDialog(n_selected=1, n_total=3)
    assert dlg.options().levels.is_identity
    assert dlg._levels_label.text() == "default"


def test_advanced_button_needs_a_preview_source(_qapp, preview):
    assert not ExportImagesDialog(1, 3)._advanced_button.isEnabled()
    assert ExportImagesDialog(1, 3, preview_source=lambda: preview)._advanced_button.isEnabled()


def test_accepted_levels_reach_the_export_options(_qapp, preview, monkeypatch):
    chosen = Levels(black=42, white=201, gamma=1.8)

    class _FakeAdvanced:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def levels(self):
            return chosen

    monkeypatch.setattr(parent_module, "ExportAdvancedDialog", _FakeAdvanced)

    dlg = ExportImagesDialog(1, 3, preview_source=lambda: preview)
    dlg._on_advanced()

    assert dlg.options().levels == chosen
    assert dlg._levels_label.text() == "min 42 · max 201 · γ 1.80"


def test_cancelling_advanced_leaves_the_levels_alone(_qapp, preview, monkeypatch):
    class _FakeAdvanced:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.DialogCode.Rejected

        def levels(self):
            return Levels(black=99)

    monkeypatch.setattr(parent_module, "ExportAdvancedDialog", _FakeAdvanced)

    dlg = ExportImagesDialog(1, 3, preview_source=lambda: preview)
    dlg._on_advanced()

    assert dlg.options().levels.is_identity
    assert dlg._levels_label.text() == "default"


def test_the_colormap_preview_shows_the_chosen_levels(_qapp, preview, monkeypatch):
    """Preview… must reflect Advanced, or the two windows would disagree."""
    shown = []

    class _FakePopup:
        def __init__(self, pixmap, title, parent=None):
            shown.append(pixmap)

        def exec(self):
            return 0

    monkeypatch.setattr(parent_module, "_ColormapPreviewDialog", _FakePopup)

    dlg = ExportImagesDialog(1, 3, preview_source=lambda: preview)
    dlg._colormap_combo.setCurrentText("Viridis")
    dlg._on_preview()
    dlg._levels = Levels(black=100, white=210)
    dlg._on_preview()

    assert len(shown) == 2
    assert shown[0].toImage() != shown[1].toImage()
