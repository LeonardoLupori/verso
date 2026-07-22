"""Scene-aware readers for multi-scene microscopy container formats.

VERSO's data model treats one :class:`~verso.engine.model.project.Section` as a
single 2-D multichannel image. Vendor container formats break that assumption:

* **CZI** (Zeiss) holds *multiple scenes/images* in one file, and each scene may
  carry a tiled **mosaic**, a **Z-stack**, and **timepoints**.

This module reduces each scene to the single ``(H, W, C)`` plane VERSO expects,
using one uniform flattening rule (decided with the user):

* **Z**  → per-channel maximum-intensity projection (on raw pixel values).
* **T**  → first timepoint only (time is not a supported analysis axis; a
  multi-timepoint file never errors, it just uses ``T=0``).
* **mosaic** → tiles stitched into the full scene (by libCZI).
* **channels** → preserved.

The public surface is small and library-agnostic so callers in
:mod:`verso.engine.io.image_io` never import a vendor library directly:

* :func:`enumerate_scenes` — list the scenes in a container.
* :func:`read_scene` — read one scene as ``(H, W, C)`` native-dtype.
* :func:`channel_names` — channel names for a container.
* :func:`scene_dimensions` — ``(width, height)`` of a scene without decoding it.
* :data:`CONTAINER_EXTENSIONS` — extensions handled here.

The reader library (``pylibCZIrw``) is a core dependency but is imported lazily
inside each reader so importing this module stays cheap. The registry is kept
format-agnostic so additional container formats can be added later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

# Extensions dispatched to a scene reader here instead of the plain
# tifffile/PIL path in image_io.
CONTAINER_EXTENSIONS = (".czi",)


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
    """Return ``True`` if *path* is a multi-scene container format (CZI)."""
    return Path(path).suffix.lower() in CONTAINER_EXTENSIONS


# ---------------------------------------------------------------------------
# Shared: reduce a dimension-labelled array to (H, W, C)
# ---------------------------------------------------------------------------


def reduce_to_hwc(arr: np.ndarray, axes: str) -> np.ndarray:
    """Reduce a dimension-labelled array to channels-last ``(H, W, C)``.

    Applies the uniform flattening rule to any array whose axis order is known:

    * ``Z`` axes are collapsed by **maximum** (MIP).
    * Any axis other than ``Y``/``X``/``C``/``S`` (e.g. ``T``, ``I``, ``Q``) is
      reduced to its **first** element.
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
            _log.debug("CZI raw_metadata unavailable for %s; using generic names", path)
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
                _log.debug("CZI pixel-type probe failed for channel %d of %s", ci, path)
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
}


def _reader_for(path: str | Path) -> _Reader:
    ext = Path(path).suffix.lower()
    reader = _READERS.get(ext)
    if reader is None:
        raise ValueError(f"{ext!r} is not a supported container format")
    return reader


def enumerate_scenes(path: str | Path) -> list[SceneInfo]:
    """List the scenes in a container file (CZI)."""
    try:
        return list(_reader_for(path).list_scenes(path))
    except Exception as exc:
        _log.debug("Scene enumeration failed for %s", path, exc_info=True)
        raise RuntimeError(f"Cannot read scenes from '{Path(path).name}': {exc}") from exc


def read_scene(path: str | Path, scene_index: int = 0, zoom: float = 1.0) -> np.ndarray:
    """Read one scene as ``(H, W, C)`` native-dtype (MIP over Z, T=0, mosaic stitched)."""
    try:
        return _reader_for(path).read_scene(path, scene_index, zoom)
    except Exception as exc:
        _log.debug("Scene %d read failed for %s (zoom=%s)", scene_index, path, zoom, exc_info=True)
        raise RuntimeError(
            f"Cannot read scene {scene_index} of '{Path(path).name}': {exc}"
        ) from exc


def channel_names(path: str | Path, scene_index: int = 0) -> list[str]:
    """Return channel names for a container scene, aligned to :func:`read_scene`."""
    return list(_reader_for(path).channel_names(path, scene_index))


def scene_dimensions(path: str | Path, scene_index: int = 0) -> tuple[int, int]:
    """Return ``(width, height)`` of a container scene without decoding pixels."""
    return _reader_for(path).scene_dimensions(path, scene_index)
