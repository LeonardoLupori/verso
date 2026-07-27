"""The DeepSlice dialog states which channels the run will actually use.

``ChannelSpec.visible`` is a real input to the predicted alignment (VERSO stages
the max of the visible planes as greyscale), so the dialog has to say so rather
than leaving it as engine trivia. Colour and brightness are display-only and are
deliberately absent from the summary.
"""

from __future__ import annotations

from pathlib import Path

from verso.engine.model.project import ChannelSpec, Section
from verso.gui.dialogs.bad_sections import BadSectionsDialog


def _sections(tmp_path: Path) -> list[Section]:
    img = tmp_path / "s001.png"
    img.touch()
    return [Section("s001", 1, str(img), str(img))]


def _hint(qtbot, tmp_path: Path, channels: list[ChannelSpec] | None) -> str:
    dlg = BadSectionsDialog(_sections(tmp_path), channels=channels)
    qtbot.addWidget(dlg)
    return dlg._channel_hint.text()


def test_hint_lists_only_visible_channels(qtbot, tmp_path: Path):
    channels = [
        ChannelSpec(name="DAPI"),
        ChannelSpec(name="GFP", visible=False),
        ChannelSpec(name="Cy5"),
    ]

    text = _hint(qtbot, tmp_path, channels)

    assert "DAPI" in text
    assert "Cy5" in text
    assert "GFP" not in text


def test_hint_uses_singular_wording_for_one_visible_channel(qtbot, tmp_path: Path):
    channels = [ChannelSpec(name="DAPI"), ChannelSpec(name="GFP", visible=False)]

    text = _hint(qtbot, tmp_path, channels)

    assert "visible channel:" in text
    assert "DAPI" in text


def test_hint_flags_the_all_hidden_fallback(qtbot, tmp_path: Path):
    """All-hidden falls back to every plane, so the dialog must not imply otherwise."""
    channels = [ChannelSpec(name=n, visible=False) for n in ("DAPI", "GFP")]

    text = _hint(qtbot, tmp_path, channels)

    assert "No channels are visible" in text
    assert "all channels will be used" in text


def test_hint_handles_a_project_without_channel_specs(qtbot, tmp_path: Path):
    text = _hint(qtbot, tmp_path, [])

    assert "All channels" in text


def test_hint_does_not_mention_display_only_settings(qtbot, tmp_path: Path):
    """Colour/brightness/gamma no longer reach DeepSlice; the hint shouldn't imply they do."""
    channels = [ChannelSpec(name="DAPI", color=(255, 0, 0), scale=0.2, gamma=2.0)]

    text = _hint(qtbot, tmp_path, channels).lower()

    assert "colour" not in text
    assert "color" not in text
    assert "brightness" not in text
