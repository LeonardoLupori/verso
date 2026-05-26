"""Tests for engine/io/export_images.py."""

import numpy as np

from verso.engine.io.export_images import render_overlay_rgba
from verso.engine.model.alignment import Alignment, ControlPoint, WarpState
from verso.engine.model.project import Preprocessing, Section


class _StubAtlas:
    """Minimal atlas stand-in exposing only ``sample_labels``.

    Returns a deterministic, asymmetric label map at the requested resolution
    (ignoring the anchoring) so that any spurious horizontal/vertical mirror in
    the overlay pipeline would change the rendered output.
    """

    def sample_labels(self, anchoring, out_w, out_h):  # noqa: ARG002 - anchoring unused
        labels = np.zeros((out_h, out_w), dtype=np.int32)
        labels[: out_h // 2, : out_w // 3] = 1  # tall block, left
        labels[: out_h // 4, out_w // 2 :] = 2  # short block, top-right
        labels[out_h // 2 :, : out_w // 4] = 3  # block, bottom-left
        in_bounds = np.ones((out_h, out_w), dtype=bool)
        return labels, in_bounds


def _make_section(*, flip_h: bool = False, flip_v: bool = False) -> Section:
    """Section anchored to the atlas with a single off-centre control point."""
    anchoring = [0.0, 264.0, 0.0, 456.0, 0.0, 0.0, 0.0, 0.0, 320.0]
    return Section(
        id="s1",
        serial_number=1,
        original_path="img.tif",
        thumbnail_path="thumb.png",
        preprocessing=Preprocessing(flip_horizontal=flip_h, flip_vertical=flip_v),
        alignment=Alignment(anchoring=anchoring),
        warp=WarpState(control_points=[ControlPoint(0.3, 0.3, 0.55, 0.45)]),
    )


def _render(section: Section) -> np.ndarray:
    return render_overlay_rgba(section, _StubAtlas(), out_w=200, out_h=150, thickness=1)


def test_overlay_export_invariant_to_horizontal_flip():
    """Horizontal flip must not change the exported overlay.

    The GUI flips only the section background, never the atlas overlay (whose
    orientation is encoded by the anchoring). Re-introducing a flip in the
    overlay pipeline mirrors the contours and reverses warp displacements.
    """
    base = _render(_make_section())
    flipped = _render(_make_section(flip_h=True))
    assert base[..., 3].any()  # overlay actually drew something
    np.testing.assert_array_equal(base, flipped)


def test_overlay_export_invariant_to_vertical_flip():
    """Vertical flip must not change the exported overlay."""
    base = _render(_make_section())
    flipped = _render(_make_section(flip_v=True))
    assert base[..., 3].any()
    np.testing.assert_array_equal(base, flipped)


def test_overlay_export_invariant_to_both_flips():
    """Combined horizontal + vertical flip must not change the exported overlay."""
    base = _render(_make_section())
    flipped = _render(_make_section(flip_h=True, flip_v=True))
    assert base[..., 3].any()
    np.testing.assert_array_equal(base, flipped)
