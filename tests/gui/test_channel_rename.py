"""Inline channel renaming in the brightness controls.

The channel name field is a read-only chip until double-clicked. Committing a
changed, non-empty name emits ``channels_committed`` with the new name; empty,
unchanged, or Escaped edits emit nothing. Duplicate names are disambiguated by
the controls (the only place that sees every channel).
"""

from __future__ import annotations

from PyQt6.QtCore import Qt

from verso.engine.model.project import ChannelSpec
from verso.gui.widgets.brightness_controls import BrightnessControls


def _make_controls(qtbot, names: list[str]) -> BrightnessControls:
    controls = BrightnessControls()
    qtbot.addWidget(controls)
    controls.set_channels([ChannelSpec(name=n, color=(255, 255, 255)) for n in names])
    return controls


def _edit(field, text: str) -> None:
    """Drive the field as a user double-click + type + commit would."""
    field._begin_edit()
    field.setText(text)
    field._finish_edit()  # what editingFinished (Return / focus-out) triggers


def test_rename_emits_committed_with_new_name(qtbot):
    controls = _make_controls(qtbot, ["DAPI", "GFP"])
    field = controls._rows[0]._name_edit

    with qtbot.waitSignal(controls.channels_committed) as blocker:
        _edit(field, "NeuN")

    assert [c.name for c in blocker.args[0]] == ["NeuN", "GFP"]
    assert controls._channels[0].name == "NeuN"
    assert field.text() == "NeuN"
    assert field.isReadOnly()


def test_empty_name_reverts_and_emits_nothing(qtbot):
    controls = _make_controls(qtbot, ["DAPI", "GFP"])
    field = controls._rows[0]._name_edit
    emitted: list = []
    controls.channels_committed.connect(emitted.append)

    _edit(field, "   ")

    assert not emitted
    assert controls._channels[0].name == "DAPI"
    assert field.text() == "DAPI"


def test_unchanged_name_emits_nothing(qtbot):
    controls = _make_controls(qtbot, ["DAPI", "GFP"])
    field = controls._rows[0]._name_edit
    emitted: list = []
    controls.channels_committed.connect(emitted.append)

    _edit(field, "DAPI")

    assert not emitted
    assert controls._channels[0].name == "DAPI"


def test_escape_cancels(qtbot):
    controls = _make_controls(qtbot, ["DAPI", "GFP"])
    field = controls._rows[0]._name_edit
    emitted: list = []
    controls.channels_committed.connect(emitted.append)

    field._begin_edit()
    field.setText("scratch")
    qtbot.keyClick(field, Qt.Key.Key_Escape)

    assert not emitted
    assert controls._channels[0].name == "DAPI"
    assert field.text() == "DAPI"
    assert field.isReadOnly()


def test_enter_key_commits_and_locks(qtbot):
    controls = _make_controls(qtbot, ["DAPI", "GFP"])
    field = controls._rows[0]._name_edit

    field._begin_edit()
    field.setText("NeuN")
    with qtbot.waitSignal(controls.channels_committed):
        qtbot.keyClick(field, Qt.Key.Key_Return)

    assert controls._channels[0].name == "NeuN"
    assert field.isReadOnly()


def test_duplicate_name_is_disambiguated(qtbot):
    controls = _make_controls(qtbot, ["DAPI", "GFP"])
    field = controls._rows[1]._name_edit

    with qtbot.waitSignal(controls.channels_committed) as blocker:
        _edit(field, "DAPI")  # collides with row 0

    assert controls._channels[1].name == "DAPI (2)"
    assert [c.name for c in blocker.args[0]] == ["DAPI", "DAPI (2)"]
    assert field.text() == "DAPI (2)"


def test_duplicate_is_case_insensitive(qtbot):
    controls = _make_controls(qtbot, ["DAPI", "GFP"])
    field = controls._rows[1]._name_edit

    _edit(field, "dapi")

    assert controls._channels[1].name == "dapi (2)"


def test_rename_then_color_commit_keeps_new_name(qtbot):
    controls = _make_controls(qtbot, ["DAPI", "GFP"])
    row = controls._rows[0]
    _edit(row._name_edit, "NeuN")

    # A subsequent visibility/color/brightness commit must not resurrect "DAPI".
    with qtbot.waitSignal(controls.channels_committed) as blocker:
        row._on_visible(False)

    assert controls._channels[0].name == "NeuN"
    assert blocker.args[0][0].name == "NeuN"
