"""Tests for engine/io/export_images.py."""

import numpy as np
from scipy import ndimage

from verso.engine.io.export_images import (
    _smooth_label_map,
    render_overlay_rgba,
)
from verso.engine.model.alignment import Alignment, ControlPoint, WarpState
from verso.engine.model.project import Preprocessing, Section


class _StubAtlas:
    """Minimal atlas stand-in exposing ``sample_labels`` + ``colorize_labels``.

    Returns a deterministic, asymmetric label map at the requested resolution
    (ignoring the anchoring) so that any spurious horizontal/vertical mirror in
    the overlay pipeline would change the rendered output.
    """

    def sample_labels(self, anchoring, out_w, out_h):
        labels = np.zeros((out_h, out_w), dtype=np.int32)
        labels[: out_h // 2, : out_w // 3] = 1  # tall block, left
        labels[: out_h // 4, out_w // 2 :] = 2  # short block, top-right
        labels[out_h // 2 :, : out_w // 4] = 3  # block, bottom-left
        in_bounds = np.ones((out_h, out_w), dtype=bool)
        return labels, in_bounds

    def colorize_labels(self, labels):
        palette = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}
        unique, inverse = np.unique(labels, return_inverse=True)
        colors = np.array([palette.get(int(u), (128, 128, 128)) for u in unique], dtype=np.uint8)
        return colors[inverse].reshape(*labels.shape, 3)


def _make_section(*, flip_h: bool = False, flip_v: bool = False) -> Section:
    """Section anchored to the atlas with a single off-centre control point."""
    anchoring = [0.0, 264.0, 0.0, 456.0, 0.0, 0.0, 0.0, 0.0, 320.0]
    return Section(
        id="s1",
        slice_index=1,
        original_path="img.tif",
        thumbnail_path="thumb.png",
        preprocessing=Preprocessing(flip_horizontal=flip_h, flip_vertical=flip_v),
        alignment=Alignment(anchoring=anchoring),
        warp=WarpState(control_points=[ControlPoint(60.0, 45.0, 110.0, 67.5)]),
    )


def _render(section: Section, **kw) -> np.ndarray:
    return render_overlay_rgba(section, _StubAtlas(), out_w=200, out_h=150, thickness=1, **kw)


# --- Flip invariance ---------------------------------------------------------


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


# --- Filled-regions mode -----------------------------------------------------


def test_filled_overlay_colors_regions_with_scaled_alpha():
    """Filled mode paints atlas colors with opacity-scaled alpha inside the brain."""
    rgba = _render(_make_section(), overlay_style="filled", opacity=0.5)
    alpha = rgba[..., 3]
    # Opaque inside annotated regions (alpha == round(0.5*255)), transparent in
    # the background.
    assert set(np.unique(alpha)).issubset({0, 128})
    assert (alpha == 128).any()
    assert (alpha == 0).any()
    # Painted pixels carry a non-black color from the palette.
    painted = rgba[alpha == 128]
    assert painted[:, :3].any()


# --- SDF label smoothing -----------------------------------------------------


def _staircase_labels() -> np.ndarray:
    """A diagonal two-region split — heavily aliased boundary to smooth."""
    h = w = 40
    rr, cc = np.mgrid[0:h, 0:w]
    return (cc > rr).astype(np.int32)  # 0 below diagonal, 1 above


def test_smooth_label_map_preserves_ids_and_shape():
    labels = _staircase_labels()
    out_w, out_h = 120, 120  # 3x upscale
    smoothed, best = _smooth_label_map(labels, out_w, out_h, sigma=1.0)
    assert smoothed.shape == (out_h, out_w)
    assert best.shape == (out_h, out_w)
    assert set(np.unique(smoothed).tolist()).issubset(set(np.unique(labels).tolist()))


def _boundary_count(label_map: np.ndarray) -> int:
    edges = np.zeros(label_map.shape, dtype=bool)
    edges[:, :-1] |= label_map[:, :-1] != label_map[:, 1:]
    edges[:-1, :] |= label_map[:-1, :] != label_map[1:, :]
    return int(edges.sum())


def test_smooth_label_map_reduces_boundary_roughness():
    """Stronger smoothing yields a shorter (straighter) boundary."""
    labels = _staircase_labels()
    out_w = out_h = 120
    low = _smooth_label_map(labels, out_w, out_h, sigma=0.0)[0]
    high = _smooth_label_map(labels, out_w, out_h, sigma=3.0)[0]
    assert _boundary_count(high) <= _boundary_count(low)


def _naive_smooth(labels: np.ndarray, out_w: int, out_h: int, sigma: float) -> np.ndarray:
    """Reference full-frame implementation (the user's original algorithm)."""
    import cv2

    ids, compact = np.unique(labels, return_inverse=True)
    compact = compact.reshape(labels.shape)
    best_val = np.full((out_h, out_w), -1e30, np.float32)
    best_lab = np.zeros((out_h, out_w), np.int32)
    for li in range(len(ids)):
        m = compact == li
        sdf = (ndimage.distance_transform_edt(m) - ndimage.distance_transform_edt(~m)).astype(
            np.float32
        )
        if sigma > 0:
            sdf = cv2.GaussianBlur(sdf, (0, 0), sigma)
        up = cv2.resize(sdf, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        better = up > best_val
        best_val[better], best_lab[better] = up[better], li
    return ids[best_lab]


def test_smooth_label_map_matches_naive_reference():
    """The bbox-crop optimization matches the full-frame reference everywhere
    except a sub-pixel band along boundaries (independent per-crop cubic resizes
    sample slightly differently than one full-frame resize)."""
    # A few compact blobs on a background so regions have small bounding boxes.
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[3:10, 4:12] = 1
    labels[18:27, 5:13] = 2
    labels[6:14, 18:26] = 3
    out_w = out_h = 90
    fast = _smooth_label_map(labels, out_w, out_h, sigma=1.0)[0]
    ref = _naive_smooth(labels, out_w, out_h, sigma=1.0)

    disagree = fast != ref
    assert disagree.mean() < 0.02  # <2% of pixels, all near boundaries
    # Every disagreement must sit next to a reference boundary pixel.
    ref_edges = np.zeros(ref.shape, dtype=bool)
    ref_edges[:, :-1] |= ref[:, :-1] != ref[:, 1:]
    ref_edges[:-1, :] |= ref[:-1, :] != ref[1:, :]
    ref_edge_dilated = ndimage.binary_dilation(ref_edges, iterations=2)
    assert not (disagree & ~ref_edge_dilated).any()
