"""Scene-aware readers for multi-scene microscopy container formats.

VERSO's data model treats one :class:`~verso.engine.model.project.Section` as a
single 2-D multichannel image. Vendor container formats break that assumption:

* **CZI** (Zeiss) and **LIF** (Leica) hold *multiple scenes/images* in one file,
  and each scene may carry a tiled **mosaic**, a **Z-stack**, and **timepoints**.

This module reduces each scene to the single ``(H, W, C)`` plane VERSO expects,
using one uniform flattening rule (decided with the user):

* **Z**  → per-channel maximum-intensity projection (on raw pixel values).
* **T**  → first timepoint only (time is not a supported analysis axis; a
  multi-timepoint file never errors, it just uses ``T=0``).
* **mosaic** → tiles stitched into the full scene.
* **channels** → preserved.

The public surface is small and library-agnostic so callers in
:mod:`verso.engine.io.image_io` never import a vendor library directly:

* :func:`enumerate_scenes` — list the scenes in a container.
* :func:`read_scene` — read one scene as ``(H, W, C)`` native-dtype.
* :func:`channel_names` — channel names for a container.
* :func:`scene_dimensions` — ``(width, height)`` of a scene without decoding it.
* :data:`CONTAINER_EXTENSIONS` — extensions handled here.

Reader libraries (``pylibCZIrw``, ``liffile``) are core dependencies but are
imported lazily inside each reader so importing this module stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Extensions dispatched to a scene reader here instead of the plain
# tifffile/PIL path in image_io.
CONTAINER_EXTENSIONS = (".czi", ".lif")


@dataclass(frozen=True)
class SceneInfo:
    """One scene/image inside a container file.

    Attributes:
        scene_index: Index of the scene within the file (0-based).
        name: Human-readable scene name from metadata (falls back to ``Scene N``).
        width: Scene width in pixels at full resolution.
        height: Scene height in pixels at full resolution.
    """

    scene_index: int
    name: str
    width: int
    height: int


def is_container(path: str | Path) -> bool:
    """Return ``True`` if *path* is a multi-scene container format (CZI/LIF)."""
    return Path(path).suffix.lower() in CONTAINER_EXTENSIONS


# ---------------------------------------------------------------------------
# Shared: reduce a dimension-labelled array to (H, W, C)
# ---------------------------------------------------------------------------


def reduce_to_hwc(arr: np.ndarray, axes: str) -> np.ndarray:
    """Reduce a dimension-labelled array to channels-last ``(H, W, C)``.

    Applies the uniform flattening rule to any array whose axis order is known:

    * ``Z`` axes are collapsed by **maximum** (MIP).
    * Any axis other than ``Y``/``X``/``C``/``S`` (e.g. ``T``, ``M`` if not
      already stitched, ``I``, ``Q``) is reduced to its **first** element.
    * The remaining ``C`` and/or ``S`` (sample) axes become the channel axis,
      moved last; a purely 2-D ``(Y, X)`` result gains a trailing length-1 axis.

    The array dtype is preserved. Axis letters are matched case-insensitively.

    Args:
        arr: Raw array whose shape matches *axes* element-for-element.
        axes: One letter per axis of *arr* (e.g. ``"TZCYX"``, ``"CYX"``,
            ``"YXS"``). Must contain ``Y`` and ``X``.

    Returns:
        ``(H, W, C)`` array in the input dtype.
    """
    axes = axes.upper()
    if len(axes) != arr.ndim:
        raise ValueError(f"axes {axes!r} does not match array ndim {arr.ndim}")
    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"axes {axes!r} must contain Y and X")

    # 1. Max-project every Z axis.
    while "Z" in axes:
        i = axes.index("Z")
        arr = arr.max(axis=i)
        axes = axes[:i] + axes[i + 1 :]

    # 2. Take the first element of every non-spatial, non-channel axis.
    keep = set("YXCS")
    letter = next((c for c in axes if c not in keep), None)
    while letter is not None:
        i = axes.index(letter)
        arr = np.take(arr, 0, axis=i)
        axes = axes[:i] + axes[i + 1 :]
        letter = next((c for c in axes if c not in keep), None)

    # 3. Move Y, X to the front, remaining (C/S) axes to the back as channels.
    yx = [axes.index("Y"), axes.index("X")]
    chan = [i for i in range(len(axes)) if i not in yx]
    arr = np.transpose(arr, yx + chan)  # (H, W, *channel_axes)

    h, w = arr.shape[0], arr.shape[1]
    if arr.ndim == 2:
        return arr[:, :, np.newaxis]
    return np.ascontiguousarray(arr.reshape(h, w, -1))


# ---------------------------------------------------------------------------
# CZI (Zeiss) — pylibCZIrw
# ---------------------------------------------------------------------------


def _czi_axis_range(box: dict, letter: str) -> tuple[int, int]:
    """Return ``(start, size)`` for a CZI bounding-box dimension, default ``(0, 1)``."""
    start, size = box.get(letter, (0, 1))
    return int(start), int(size)


def _czi_scene_rects(doc) -> dict[int, object]:
    """Return ``{scene_index: Rectangle}``; a single-scene file maps ``{0: total_rect}``."""
    rects = dict(doc.scenes_bounding_rectangle)
    if not rects:
        return {0: doc.total_bounding_rectangle}
    return rects


def czi_list_scenes(path: str | Path) -> list[SceneInfo]:
    from pylibCZIrw import czi as pyczi

    stem = Path(path).stem
    with pyczi.open_czi(str(path)) as doc:
        rects = _czi_scene_rects(doc)
        multi = len(rects) > 1
        out: list[SceneInfo] = []
        for idx in sorted(rects):
            rect = rects[idx]
            name = f"{stem} — Scene {idx}" if multi else stem
            out.append(SceneInfo(idx, name, int(rect.w), int(rect.h)))
    return out


def czi_scene_dimensions(path: str | Path, scene_index: int = 0) -> tuple[int, int]:
    from pylibCZIrw import czi as pyczi

    with pyczi.open_czi(str(path)) as doc:
        rects = _czi_scene_rects(doc)
        rect = rects.get(scene_index) or doc.total_bounding_rectangle
        return int(rect.w), int(rect.h)


def czi_channel_names(path: str | Path, scene_index: int = 0) -> list[str]:
    """Channel names from CZI metadata, expanded to match :func:`czi_read_scene`.

    Gray channels keep their metadata name; a BGR (brightfield) channel expands
    to ``<name> R/G/B`` because :func:`czi_read_scene` splits it into three
    planes.
    """
    import re

    from pylibCZIrw import czi as pyczi

    with pyczi.open_czi(str(path)) as doc:
        box = doc.total_bounding_box
        _, n_ch = _czi_axis_range(box, "C")
        c_start, _ = _czi_axis_range(box, "C")
        try:
            raw = doc.raw_metadata
        except Exception:
            raw = ""
        # Prefer the acquisition channel list (Information/Image/Dimensions);
        # fall back to any <Channel ... Name="..."> then to generic names.
        names = re.findall(r'<Channel[^>]*\bName="([^"]+)"', raw)
        # Deduplicate preserving order (DisplaySetting repeats the same names).
        seen: dict[str, None] = {}
        for nm in names:
            seen.setdefault(nm, None)
        names = list(seen)

        pixel_bgr = []
        for ci in range(c_start, c_start + n_ch):
            try:
                pixel_bgr.append("bgr" in doc.get_channel_pixel_type(ci).lower())
            except Exception:
                pixel_bgr.append(False)

    out: list[str] = []
    for i in range(n_ch):
        base = names[i] if i < len(names) else f"Ch {i}"
        if pixel_bgr[i]:
            out.extend([f"{base} R", f"{base} G", f"{base} B"])
        else:
            out.append(base)
    return out


def czi_read_scene(path: str | Path, scene_index: int = 0, zoom: float = 1.0) -> np.ndarray:
    """Read one CZI scene as ``(H, W, C)`` native-dtype — MIP over Z, T=0.

    Per channel, every Z plane is read at the scene ROI (``zoom`` reduces the
    read for memory-cheap working copies; mosaic tiles are auto-stitched by the
    ROI read) and combined by ``np.maximum``. Timepoint 0 only. A BGR channel is
    split into three planes (BGR → RGB) so it renders in colour.
    """
    from pylibCZIrw import czi as pyczi

    with pyczi.open_czi(str(path)) as doc:
        scene_arg = scene_index if len(doc.scenes_bounding_rectangle) else None
        box = doc.total_bounding_box
        c_start, n_ch = _czi_axis_range(box, "C")
        z_start, n_z = _czi_axis_range(box, "Z")
        t_start, _ = _czi_axis_range(box, "T")

        planes: list[np.ndarray] = []
        for ci in range(c_start, c_start + n_ch):
            acc: np.ndarray | None = None
            for zi in range(z_start, z_start + n_z):
                frame = doc.read(
                    plane={"C": ci, "Z": zi, "T": t_start},
                    scene=scene_arg,
                    zoom=float(zoom),
                )
                acc = frame if acc is None else np.maximum(acc, frame)
            assert acc is not None  # n_z >= 1
            # acc is (H, W, S): S==1 for gray, S==3 for BGR.
            if acc.shape[-1] == 1:
                planes.append(acc[:, :, 0])
            else:
                # BGR -> RGB, each sample a separate channel.
                for s in range(acc.shape[-1] - 1, -1, -1):
                    planes.append(acc[:, :, s])

    return np.ascontiguousarray(np.stack(planes, axis=-1))


# ---------------------------------------------------------------------------
# LIF (Leica) — liffile
# ---------------------------------------------------------------------------


def _lif_leaf_images(lif) -> list:
    """Return the container's addressable images in a stable order.

    ``LifFile.images`` is a flat series over the file's image nodes; this keeps
    only genuine image data (non-empty shape) so folders/metadata nodes don't
    become phantom scenes.
    """
    images = []
    for img in lif.images:
        try:
            if img.size and "Y" in img.dims and "X" in img.dims:
                images.append(img)
        except Exception:
            continue
    return images


def lif_list_scenes(path: str | Path) -> list[SceneInfo]:
    import liffile

    stem = Path(path).stem
    out: list[SceneInfo] = []
    with liffile.LifFile(str(path)) as lif:
        images = _lif_leaf_images(lif)
        multi = len(images) > 1
        for idx, img in enumerate(images):
            sizes = dict(zip(img.dims, img.shape, strict=True))
            w, h = int(sizes.get("X", 0)), int(sizes.get("Y", 0))
            w, h = _lif_stitched_wh(img, w, h)
            base = img.name or f"Scene {idx}"
            name = f"{stem} — {base}" if multi else (base if img.name else stem)
            out.append(SceneInfo(idx, name, w, h))
    return out


def _lif_pixel_sizes(img) -> tuple[float, float] | None:
    """Physical pixel size ``(px_x, px_y)`` in metres/pixel from the X/Y coords.

    Returns ``None`` when neither axis carries a usable step. A missing axis
    borrows the other's step (pixels are square on Leica confocal/camera scans).
    """
    coords = getattr(img, "coords", {}) or {}
    steps: dict[str, float] = {}
    for key in ("X", "Y"):
        c = coords.get(key)
        if c is None:
            continue
        a = np.asarray(c, dtype=float)
        if a.size >= 2:
            step = abs(float(a[1] - a[0]))
            if step > 0:
                steps[key] = step
    if not steps:
        return None
    px_x = steps.get("X", steps.get("Y"))
    px_y = steps.get("Y", steps.get("X"))
    return px_x, px_y


def _lif_tile_offsets(
    img, tile_w: int, tile_h: int
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """Per-tile pixel offsets and stitched canvas size for a tilescan.

    Leica ``field_x``/``field_y`` indices are unreliable (they can be a flat
    counter that ignores the 2-D grid), so tiles are placed by their **physical
    stage position** (``pos_x``/``pos_y``) converted to pixels via the per-axis
    physical pixel size — which correctly accounts for inter-tile **overlap**.
    Falls back to the ``field_x``/``field_y`` grid only when the physical pixel
    size is unavailable.

    Returns ``(ox, oy, width, height)`` (offsets in pixels, top-left origin) or
    ``None`` when there is no usable tile geometry.
    """
    ts = img.tilescan
    if ts is None:
        return None
    tiles = ts.tiles
    px = _lif_pixel_sizes(img)

    if px is not None:
        px_x, px_y = px
        pos_x = np.asarray(tiles["pos_x"], dtype=float)
        pos_y = np.asarray(tiles["pos_y"], dtype=float)
        if ts.swap_xy:
            pos_x, pos_y = pos_y, pos_x
        ox = np.rint((pos_x - pos_x.min()) / px_x).astype(int)
        oy = np.rint((pos_y - pos_y.min()) / px_y).astype(int)
    else:
        ox = np.asarray(tiles["field_x"], dtype=int) * tile_w
        oy = np.asarray(tiles["field_y"], dtype=int) * tile_h

    if ts.flip_x:
        ox = int(ox.max()) - ox
    if ts.flip_y:
        oy = int(oy.max()) - oy

    width = int(ox.max()) + tile_w
    height = int(oy.max()) + tile_h
    return ox, oy, width, height


def _lif_stitched_wh(img, tile_w: int, tile_h: int) -> tuple[int, int]:
    """Full scene ``(w, h)`` after mosaic stitching (== tile size when not a tilescan)."""
    sizes = dict(zip(img.dims, img.shape, strict=True))
    if int(sizes.get("M", 1)) <= 1:
        return tile_w, tile_h
    placement = _lif_tile_offsets(img, tile_w, tile_h)
    if placement is None:
        return tile_w, tile_h
    _, _, width, height = placement
    return width, height


def lif_scene_dimensions(path: str | Path, scene_index: int = 0) -> tuple[int, int]:
    import liffile

    with liffile.LifFile(str(path)) as lif:
        img = _lif_leaf_images(lif)[scene_index]
        sizes = dict(zip(img.dims, img.shape, strict=True))
        return _lif_stitched_wh(img, int(sizes.get("X", 0)), int(sizes.get("Y", 0)))


def lif_channel_names(path: str | Path, scene_index: int = 0) -> list[str]:
    import liffile

    from verso.engine.io.image_io import _default_channel_names

    with liffile.LifFile(str(path)) as lif:
        img = _lif_leaf_images(lif)[scene_index]
        sizes = dict(zip(img.dims, img.shape, strict=True))
        n_ch = int(sizes.get("C", 1))
        # liffile exposes channel coordinates when named; else fall back.
        coords = getattr(img, "coords", {}) or {}
        chan = coords.get("C")
        if chan is not None:
            names = [str(c) for c in list(chan)]
            if len(names) == n_ch:
                return names
    return _default_channel_names(n_ch)


def _lif_stitch_mosaic(arr: np.ndarray, axes: str, img) -> tuple[np.ndarray, str]:
    """Stitch the ``M`` (mosaic) axis of *arr* into ``Y``/``X``.

    Tiles are placed by their physical stage position (see :func:`_lif_tile_offsets`,
    which accounts for inter-tile overlap and honours ``SwapXY``/``FlipX``/``FlipY``).
    Overlap follows the **painter's algorithm**: tiles are drawn in acquisition
    (``M``) order and a later tile overwrites an earlier one in the overlap — the
    same raw-composite convention Zeiss's libCZI uses for CZI mosaics (and
    Bio-Formats/ImageJ for tiled reads), so a LIF and a CZI of the same sample
    assemble consistently. No feather/linear blending or shading correction is
    applied (that is a separate stitching step, e.g. BigStitcher). Returns
    ``(stitched_arr, new_axes)`` with ``M`` removed. If there is no mosaic axis
    the input is returned unchanged; if tile geometry is missing the first tile
    is used.
    """
    axes = axes.upper()
    if "M" not in axes:
        return arr, axes
    m_ax = axes.index("M")
    n_m = arr.shape[m_ax]
    tile_h = arr.shape[axes.index("Y")]
    tile_w = arr.shape[axes.index("X")]
    ts = img.tilescan
    placement = _lif_tile_offsets(img, tile_w, tile_h)
    if ts is None or len(ts) != n_m or placement is None:
        # No usable tile geometry: fall back to the first tile.
        return np.take(arr, 0, axis=m_ax), axes[:m_ax] + axes[m_ax + 1 :]

    ox, oy, width, height = placement

    moved = np.moveaxis(arr, m_ax, 0)  # (M, ...) with Y/X shifted by -1 after removal
    rest_axes = axes[:m_ax] + axes[m_ax + 1 :]
    ry, rx = rest_axes.index("Y"), rest_axes.index("X")
    out_shape = list(moved.shape[1:])
    out_shape[ry] = height
    out_shape[rx] = width
    out = np.zeros(out_shape, dtype=arr.dtype)
    for m in range(n_m):  # painter's algorithm: later tiles overwrite earlier
        y0, x0 = int(oy[m]), int(ox[m])
        dst: list[object] = [slice(None)] * (moved.ndim - 1)
        dst[ry] = slice(y0, y0 + tile_h)
        dst[rx] = slice(x0, x0 + tile_w)
        out[tuple(dst)] = moved[m]
    return out, rest_axes


def lif_read_scene(path: str | Path, scene_index: int = 0, zoom: float = 1.0) -> np.ndarray:
    """Read one LIF scene as ``(H, W, C)`` native-dtype.

    liffile has no reduced-resolution read, so the full scene is decoded and
    ``zoom`` is ignored here (the caller downscales the uint8 working copy).
    Mosaic tiles are stitched, Z is max-projected, T=0.
    """
    import liffile

    del zoom  # liffile reads full resolution; caller handles downscaling.
    with liffile.LifFile(str(path)) as lif:
        img = _lif_leaf_images(lif)[scene_index]
        axes = "".join(img.dims).upper()
        arr = np.asarray(img.asarray())
        arr, axes = _lif_stitch_mosaic(arr, axes, img)
    return reduce_to_hwc(arr, axes)


# ---------------------------------------------------------------------------
# Registry / dispatch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Reader:
    list_scenes: object
    read_scene: object
    channel_names: object
    scene_dimensions: object


_READERS: dict[str, _Reader] = {
    ".czi": _Reader(czi_list_scenes, czi_read_scene, czi_channel_names, czi_scene_dimensions),
    ".lif": _Reader(lif_list_scenes, lif_read_scene, lif_channel_names, lif_scene_dimensions),
}


def _reader_for(path: str | Path) -> _Reader:
    ext = Path(path).suffix.lower()
    reader = _READERS.get(ext)
    if reader is None:
        raise ValueError(f"{ext!r} is not a supported container format")
    return reader


def enumerate_scenes(path: str | Path) -> list[SceneInfo]:
    """List the scenes in a container file (CZI/LIF)."""
    try:
        return list(_reader_for(path).list_scenes(path))
    except Exception as exc:
        raise RuntimeError(f"Cannot read scenes from '{Path(path).name}': {exc}") from exc


def read_scene(path: str | Path, scene_index: int = 0, zoom: float = 1.0) -> np.ndarray:
    """Read one scene as ``(H, W, C)`` native-dtype (MIP over Z, T=0, mosaic stitched)."""
    try:
        return _reader_for(path).read_scene(path, scene_index, zoom)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read scene {scene_index} of '{Path(path).name}': {exc}"
        ) from exc


def channel_names(path: str | Path, scene_index: int = 0) -> list[str]:
    """Return channel names for a container scene, aligned to :func:`read_scene`."""
    return list(_reader_for(path).channel_names(path, scene_index))


def scene_dimensions(path: str | Path, scene_index: int = 0) -> tuple[int, int]:
    """Return ``(width, height)`` of a container scene without decoding pixels."""
    return _reader_for(path).scene_dimensions(path, scene_index)
