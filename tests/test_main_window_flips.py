from types import SimpleNamespace

from verso.engine.model.alignment import Alignment, AlignmentStatus
from verso.engine.model.project import Section
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

    def _clear_alignment_for_flip(self, section):
        MainWindow._clear_alignment_for_flip(self, section)

    def _after_flip_refresh(self):
        pass


def test_horizontal_flip_clears_alignment():
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

    assert section.alignment.anchoring == [0.0] * 9
    assert section.alignment.status == AlignmentStatus.NOT_STARTED
    assert section.alignment.source is None
    assert section.alignment.proposal_anchoring is None


def test_horizontal_flip_clears_complete_alignment():
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
    window = _NoopMainWindow(
        _state=SimpleNamespace(current_section=section, atlas=None, section_index=0),
        _current_mode="overview",
        _overview=_NoopOverview(),
        _prep=_NoopPrep(),
    )

    MainWindow._on_flip_h_changed(window, True)

    assert section.alignment.anchoring == [0.0] * 9
    assert section.alignment.status == AlignmentStatus.NOT_STARTED
    assert section.alignment.stored_anchoring is None
