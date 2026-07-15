"""Shared background-channel display pipeline.

Both :class:`~verso.gui.views.prep_view.PrepView` and
:class:`~verso.gui.widgets.section_canvas_panel.SectionCanvasPanel` render the
same multi-channel section image onto an :class:`~verso.gui.widgets.canvas.ImageCanvas`
before layering their own (mask vs. atlas) overlay on top.  That raw-image path —
2D→3D promotion, flip handling, the GPU-upload cache guard, and the per-channel
LUT/visibility loop — is identical, so it lives here once instead of being copied
into each host.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from verso.engine.preprocessing import channel_lut

if TYPE_CHECKING:
    from verso.engine.model.project import ChannelSpec, Section
    from verso.gui.widgets.canvas import ImageCanvas


def push_channel_display(
    canvas: ImageCanvas,
    raw_image: np.ndarray | None,
    section: Section | None,
    channels: list[ChannelSpec],
    planes_version: int,
    prev_planes_key: tuple | None,
) -> tuple | None:
    """Render ``raw_image``'s channels onto ``canvas`` and return the plane key.

    The raw uint8 planes are only re-uploaded to the GPU when the section, flip
    state, or channel count changes — tracked by the returned cache key
    ``(planes_version, flip_h, flip_v, n)``.  Brightness/colour/visibility edits
    take the cheap path (a per-channel LUT swap) and never re-upload textures.

    Args:
        canvas: the target image canvas.
        raw_image: (H, W) or (H, W, C) uint8 working-resolution image, or None.
        section: the section whose ``preprocessing`` flips are applied, or None.
        channels: per-channel display specs (colour / scale / visibility).
        planes_version: bumped by the caller on every raw-image (re)load; part of
            the cache key so a reused ``raw_image`` object can't skip an upload.
        prev_planes_key: the key returned by the previous call, or None.

    Returns:
        The new plane cache key, or None when there is no image (canvas cleared).
        The caller should store this and pass it back as ``prev_planes_key``.
    """
    if raw_image is None:
        canvas.clear()
        return None

    img = raw_image
    if img.ndim == 2:
        img = img[..., np.newaxis]
    flip_h = bool(section and section.preprocessing.flip_horizontal)
    flip_v = bool(section and section.preprocessing.flip_vertical)
    if flip_h:
        img = np.fliplr(img)
    if flip_v:
        img = np.flipud(img)
    n = min(img.shape[2], len(channels))

    # Re-push raw planes only when section / flip / channel count changes; this is
    # the only path that touches the GPU texture.
    planes_key = (planes_version, flip_h, flip_v, n)
    if planes_key != prev_planes_key:
        canvas.set_channel_planes([np.ascontiguousarray(img[:, :, i]) for i in range(n)])

    # Apply per-channel LUT + visibility — what the brightness slider drives, and
    # the cheap path (a ~1 KB table swap per tick).
    for i in range(n):
        spec = channels[i]
        if not getattr(spec, "visible", True) or float(spec.scale) <= 0:
            canvas.set_channel_visible(i, False)
        else:
            canvas.set_channel_lut(i, channel_lut(spec))

    return planes_key
