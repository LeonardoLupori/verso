"""push_channel_display — the shared background-channel pipeline.

Extracted from PrepView / SectionCanvasPanel so it can be exercised headless
against a fake canvas that records the calls each host used to make.
"""

from __future__ import annotations

import numpy as np

from verso.engine.model.project import ChannelSpec, Preprocessing, Section
from verso.gui.widgets.channel_display import push_channel_display


class _FakeCanvas:
    """Records the canvas calls the pipeline makes, no Qt/GPU needed."""

    def __init__(self) -> None:
        self.cleared = 0
        self.planes_pushes: list[int] = []  # channel count per push
        self.visible: list[tuple[int, bool]] = []
        self.luts: list[int] = []  # channel indices given a LUT

    def clear(self) -> None:
        self.cleared += 1

    def set_channel_planes(self, planes) -> None:
        self.planes_pushes.append(len(planes))

    def set_channel_visible(self, i, visible) -> None:
        self.visible.append((i, visible))

    def set_channel_lut(self, i, lut) -> None:
        self.luts.append(i)


def _section(flip_h=False, flip_v=False) -> Section:
    return Section(
        id="s0",
        slice_index=0,
        original_path="s0.png",
        thumbnail_path="thumbnails/s0.tif",
        preprocessing=Preprocessing(flip_horizontal=flip_h, flip_vertical=flip_v),
    )


def _channels(n: int):
    return [ChannelSpec(name=f"c{i}") for i in range(n)]


def test_none_image_clears_and_returns_none():
    canvas = _FakeCanvas()
    key = push_channel_display(canvas, None, _section(), _channels(2), 0, ("stale",))
    assert key is None
    assert canvas.cleared == 1
    assert canvas.planes_pushes == []


def test_first_render_pushes_planes_and_luts():
    canvas = _FakeCanvas()
    img = np.zeros((4, 5, 3), dtype=np.uint8)
    key = push_channel_display(canvas, img, _section(), _channels(3), 1, None)
    assert key == (1, False, False, 3)
    assert canvas.planes_pushes == [3]
    assert canvas.luts == [0, 1, 2]


def test_planes_cached_when_key_unchanged():
    canvas = _FakeCanvas()
    img = np.zeros((4, 5, 2), dtype=np.uint8)
    key = push_channel_display(canvas, img, _section(), _channels(2), 1, None)
    # Same version/flip/count → no re-upload, but LUTs still refresh (cheap path).
    push_channel_display(canvas, img, _section(), _channels(2), 1, key)
    assert canvas.planes_pushes == [2]  # only the first call uploaded
    assert canvas.luts == [0, 1, 0, 1]


def test_flip_change_invalidates_plane_cache():
    canvas = _FakeCanvas()
    img = np.zeros((4, 5, 2), dtype=np.uint8)
    key = push_channel_display(canvas, img, _section(), _channels(2), 1, None)
    key2 = push_channel_display(canvas, img, _section(flip_h=True), _channels(2), 1, key)
    assert key2 == (1, True, False, 2)
    assert canvas.planes_pushes == [2, 2]  # re-uploaded after the flip


def test_hidden_channel_marked_invisible():
    canvas = _FakeCanvas()
    img = np.zeros((4, 5, 2), dtype=np.uint8)
    channels = [ChannelSpec(name="a", visible=False), ChannelSpec(name="b", scale=0.0)]
    push_channel_display(canvas, img, _section(), channels, 1, None)
    # Channel 0 hidden by flag, channel 1 hidden by non-positive scale → no LUTs.
    assert canvas.visible == [(0, False), (1, False)]
    assert canvas.luts == []
