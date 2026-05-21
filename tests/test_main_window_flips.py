from types import SimpleNamespace

import numpy as np

from verso.engine.model.alignment import Alignment, AlignmentStatus
from verso.engine.model.project import Section
from verso.engine.registration import anchoring_to_vectors, flip_anchoring_horizontal
from verso.gui.main_window import MainWindow


class _NoopOverview:
    def refresh_row(self, _index):
        pass


class _NoopPrep:
    def cancel_lr_draw_if_active(self):
        return False


class _NoopMainWindow(SimpleNamespace):
    def _update_ap_plot(self):
        pass


def test_horizontal_flip_updates_in_progress_default_anchoring_before_store():
    anchoring = [
        10.0, 20.0, 30.0,
        100.0, 12.0, 0.0,
        0.0, 0.0, 80.0,
    ]
    section = Section(
        id="s001",
        serial_number=1,
        original_path="s001.png",
        thumbnail_path="s001.png",
        alignment=Alignment(
            anchoring=list(anchoring),
            status=AlignmentStatus.IN_PROGRESS,
            source="quicknii_default",
            proposal_anchoring=list(anchoring),
        ),
    )
    window = _NoopMainWindow(
        _state=SimpleNamespace(current_section=section, atlas=None, section_index=0),
        _current_mode="overview",
        _overview=_NoopOverview(),
        _prep=_NoopPrep(),
    )

    MainWindow._on_flip_h_changed(window, True)

    expected = flip_anchoring_horizontal(anchoring)
    np.testing.assert_allclose(section.alignment.anchoring, expected)
    np.testing.assert_allclose(section.alignment.proposal_anchoring, expected)
    o, u, v = anchoring_to_vectors(section.alignment.anchoring)
    old_o, old_u, old_v = anchoring_to_vectors(anchoring)
    np.testing.assert_allclose(o, old_o + old_u)
    np.testing.assert_allclose(u, -old_u)
    np.testing.assert_allclose(v, old_v)


def test_horizontal_flip_does_not_update_stored_anchoring():
    anchoring = [
        10.0, 20.0, 30.0,
        100.0, 12.0, 0.0,
        0.0, 0.0, 80.0,
    ]
    section = Section(
        id="s001",
        serial_number=1,
        original_path="s001.png",
        thumbnail_path="s001.png",
        alignment=Alignment(
            anchoring=list(anchoring),
            stored_anchoring=list(anchoring),
            status=AlignmentStatus.COMPLETE,
        ),
    )
    original_stored = list(anchoring)
    window = _NoopMainWindow(
        _state=SimpleNamespace(current_section=section, atlas=None, section_index=0),
        _current_mode="overview",
        _overview=_NoopOverview(),
        _prep=_NoopPrep(),
    )

    MainWindow._on_flip_h_changed(window, True)

    np.testing.assert_allclose(section.alignment.anchoring, flip_anchoring_horizontal(anchoring))
    np.testing.assert_allclose(section.alignment.stored_anchoring, original_stored)
