"""Image loading and resolution management.

Resolution tiers
----------------
Full resolution   Original file on disk (e.g. 20000×15000). Never loaded
                  for interactive use; only for final export.
Working           Downscaled by WORKING_SCALE (default 0.2) from the original.
                  Cached on disk as a multichannel OME-TIFF in
                  ``thumbnails/{stem}-thumb.ome.tif``. Channel count and dtype
                  match the source — channels are preserved, not collapsed.
Filmstrip         Composited to RGB and resized to FILMSTRIP_MAX_SIDE (150 px)
                  on demand from the working copy.

Display path
------------
The working copy is ``uint8 (H, W, C)``. Composition to RGB for the canvas
happens via :func:`verso.engine.preprocessing.composite_channels`, which is
parameterised by the project-level :class:`~verso.engine.model.project.ChannelSpec`
list (per-channel color + brightness scale + visibility).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# Default scale factor for working copies (fraction of original resolution).
# Used as a fallback when no per-import scale has been computed (e.g. lazily
# generating a thumbnail for a QuickNII-imported section).
WORKING_SCALE: float = 0.2

# Target longest-side, in pixels, for the working copy of the *largest* image in
# an import batch. The per-batch scale is derived from this so the biggest
# section fits within THUMBNAIL_MAX_SIDE and every section shares one scale.
THUMBNAIL_MAX_SIDE = 2000
FILMSTRIP_MAX_SIDE = 150

# Heuristic upper bound on plausible channel count. Used to disambiguate
# (C, H, W) vs (H, W, C) layouts: channel axes are always small.
_MAX_PLAUSIBLE_CHANNELS = 8


def _natural_key(stem: str) -> list[object]:
    """Sort key that orders embedded numbers numerically (``s2`` before ``s10``)."""
    return [
        int(chunk) if chunk.isdigit() else chunk
        for chunk in re.split(r"(\d+)", stem)
    ]


def guess_slice_indices(paths: list[str | Path]) -> list[int]:
    """Guess a slice index per file from numbers embedded in the filenames.

    A batch of histology filenames usually shares a layout where one numeric
    field is the section number and the rest are constants (mouse id, channel,
    magnification). This tokenises every filename stem into its numeric runs and
    picks the *token position* that best discriminates the series — preferring
    more distinct values, then a wider spread, then values that increase with
    the natural filename order. That position's integer becomes each file's
    slice index.

    Only token positions present in *every* file are eligible. When no such
    position carries usable numbers, indices fall back to ``1..N`` assigned by
    natural-sorted filename order.

    Args:
        paths: Source image paths (any order).

    Returns:
        One slice index per input path, in the same order as ``paths``. Values
        may be non-contiguous and may repeat; the caller can override them.
    """
    n = len(paths)
    if n == 0:
        return []

    stems = [Path(p).stem for p in paths]
    per_file = [[int(tok) for tok in re.findall(r"\d+", stem)] for stem in stems]
    max_tokens = max((len(nums) for nums in per_file), default=0)

    name_order = sorted(range(n), key=lambda i: _natural_key(stems[i]))

    best: tuple[int, int, float, int] | None = None
    for j in range(max_tokens):
        if any(len(nums) <= j for nums in per_file):
            continue  # not present in every file — ineligible
        values = [nums[j] for nums in per_file]
        distinct = len(set(values))
        spread = max(values) - min(values)
        ordered = [per_file[i][j] for i in name_order]
        rising = sum(a < b for a, b in zip(ordered, ordered[1:]))
        monotonic = rising / (n - 1) if n > 1 else 0.0
        score = (distinct, spread, monotonic, -j)
        if best is None or score > best:
            best = score

    if best is None:
        # No fully-covered numeric field: assign 1..N by natural name order.
        indices = [0] * n
        for rank, i in enumerate(name_order, start=1):
            indices[i] = rank
        return indices

    chosen = -best[3]
    return [nums[chosen] for nums in per_file]


def thumbnail_filename(path: str | Path) -> str:
    """Return the working-thumbnail filename for a source image path."""
    return f"{Path(path).stem}-thumb.ome.tif"


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> np.ndarray:
    """Read any supported image file as a numpy array (original dtype/shape).

    TIFF files (including multi-channel OME-TIFF) are read with tifffile.
    All other formats are read with PIL.
    """
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(str(path))
    from PIL import Image
    return np.asarray(Image.open(str(path)))


def image_dimensions(path: str | Path) -> tuple[int, int]:
    """Return image dimensions as ``(width, height)`` without full decoding."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile
            with tifffile.TiffFile(str(path)) as tif:
                series = tif.series[0]
                shape = series.shape
                axes = series.axes
            if "X" in axes and "Y" in axes:
                return int(shape[axes.index("X")]), int(shape[axes.index("Y")])
            if len(shape) >= 3 and shape[-1] <= 4:
                return int(shape[-2]), int(shape[-3])
            if len(shape) >= 2:
                return int(shape[-1]), int(shape[-2])
        except Exception:
            pass

    from PIL import Image
    with Image.open(str(path)) as im:
        return im.size


def compute_working_scale(
    paths: list[str | Path], max_side: int = THUMBNAIL_MAX_SIDE
) -> float:
    """Compute a single working-copy scale factor for an import batch.

    Reads the dimensions of every image (without fully decoding them), finds the
    largest longest-side across the batch, and returns the scale that brings
    that largest image's longest side down to ``max_side``. The same factor is
    meant to be applied to every section so they share a consistent resolution.

    The factor is rounded to two decimal places and never upscales (capped at
    ``1.0``). Images whose dimensions cannot be read are skipped.

    Args:
        paths: Source image paths in the import batch.
        max_side: Target longest-side, in pixels, for the largest image.

    Returns:
        Scale factor in ``(0, 1]``, rounded to two decimals. Falls back to
        :data:`WORKING_SCALE` when no dimensions could be read.
    """
    longest = 0
    for path in paths:
        try:
            w, h = image_dimensions(path)
        except Exception:
            continue
        longest = max(longest, w, h)

    if longest == 0:
        return WORKING_SCALE
    if longest <= max_side:
        return 1.0

    scale = round(max_side / longest, 2)
    # Guard against very large images rounding the factor down to 0.0.
    return max(scale, 0.01)


# ---------------------------------------------------------------------------
# Layout + normalization
# ---------------------------------------------------------------------------

def _stretch_uint8(image: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.8) -> np.ndarray:
    """Percentile contrast stretch to uint8."""
    lo = float(np.percentile(image, low_pct))
    hi = float(np.percentile(image, high_pct))
    if hi <= lo:
        if image.dtype == np.uint8:
            return image.copy()
        return np.zeros_like(image, dtype=np.uint8)
    out = (image.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return out.clip(0, 255).astype(np.uint8)


def _stretch_per_channel(image: np.ndarray) -> np.ndarray:
    """Apply percentile stretch independently per channel.

    Input is expected ``(H, W, C)``; output is the same shape, uint8.
    """
    if image.ndim == 2:
        return _stretch_uint8(image)
    out = np.empty(image.shape, dtype=np.uint8)
    for c in range(image.shape[2]):
        out[:, :, c] = _stretch_uint8(image[:, :, c])
    return out


def to_multichannel(image: np.ndarray) -> np.ndarray:
    """Normalise any raw array into uint8 ``(H, W, C)`` channels-last layout.

    Handles:
      * 2-D grayscale → ``(H, W, 1)``
      * Channels-last RGB / RGBA ``(H, W, 3 | 4)`` → drops alpha if present
      * Channels-first ``(C, H, W)`` (typical OME-TIFF) → transposed
      * 4-D and higher → leading axes collapsed (``image[0]``) until 3-D
    """
    img = image
    while img.ndim > 3:
        img = img[0]

    if img.ndim == 2:
        img = img[..., np.newaxis]
    elif img.ndim == 3:
        s0, s1, s2 = img.shape
        # Channels-first heuristic: first axis is small, last two are large.
        # When this matches we treat all C planes as real channels (not RGBA).
        max_c = _MAX_PLAUSIBLE_CHANNELS
        was_channels_first = s0 <= max_c and s1 > max_c and s2 > max_c
        if was_channels_first:
            img = np.transpose(img, (1, 2, 0))
        elif img.shape[2] == 4:
            # Channels-last RGBA → drop alpha.
            img = img[:, :, :3]

    return _stretch_per_channel(img)


# ---------------------------------------------------------------------------
# Resize (multichannel-aware)
# ---------------------------------------------------------------------------

def _resize_multichannel(
    image: np.ndarray, new_size: tuple[int, int]
) -> np.ndarray:
    """Lanczos-resize an ``(H, W)`` or ``(H, W, C)`` uint8 array.

    new_size is ``(new_w, new_h)`` to match PIL convention.
    """
    from PIL import Image as PILImage

    if image.ndim == 2:
        pil = PILImage.fromarray(image, mode="L").resize(new_size, PILImage.LANCZOS)
        return np.array(pil)

    new_w, new_h = new_size
    out = np.empty((new_h, new_w, image.shape[2]), dtype=image.dtype)
    for c in range(image.shape[2]):
        pil = PILImage.fromarray(image[:, :, c], mode="L").resize(
            new_size, PILImage.LANCZOS
        )
        out[:, :, c] = np.array(pil)
    return out


def resize_to_max_side(
    image: np.ndarray, max_side: int
) -> tuple[np.ndarray, float]:
    """Resize image so its longest side ≤ max_side. Multichannel-aware."""
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image, 1.0

    scale = max_side / max(h, w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return _resize_multichannel(image, (new_w, new_h)), scale


def resize_by_scale(
    image: np.ndarray, scale: float
) -> tuple[np.ndarray, float]:
    """Resize image by a fixed scale factor. Multichannel-aware."""
    if scale == 1.0:
        return image, 1.0

    h, w = image.shape[:2]
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return _resize_multichannel(image, (new_w, new_h)), scale


# ---------------------------------------------------------------------------
# Working copy
# ---------------------------------------------------------------------------

def _canonical_thumbnail(section) -> Path:
    """Return the canonical OME-TIFF thumbnail path for *section*.

    Uses the parent directory of the existing ``thumbnail_path`` when it has a
    real parent (i.e. is not a bare filename resolving to CWD).  Falls back to
    a ``thumbnails/`` subfolder next to the original image — this covers
    QuickNII-imported sections where ``thumbnail_path`` was never set.
    """
    existing = Path(section.thumbnail_path) if section.thumbnail_path else None
    if existing and existing.parent != Path("."):
        thumb_dir = existing.parent
    else:
        thumb_dir = Path(section.original_path).parent / "thumbnails"
    return thumb_dir / thumbnail_filename(section.original_path)


def ensure_working_copy(
    section,
    scale: float | None = None,
) -> np.ndarray | None:
    """Return the working-resolution multichannel array for a section.

    Thumbnail resolution strategy (tried in order):

    1. Canonical OME-TIFF path — the expected ``{thumbnails_dir}/{stem}-thumb.ome.tif``.
       Used directly if it exists.
    2. Legacy thumbnail path — for old projects whose ``thumbnail_path`` still
       points to a ``.png``.  Loaded, then immediately re-saved as OME-TIFF at
       the canonical path so future calls hit path 1.
    3. Original image — loaded, downscaled, saved as OME-TIFF at canonical path.

    ``section.thumbnail_path`` is updated to the canonical path after any
    successful load so the next ``Project.save()`` persists the migrated value.

    Args:
        section: A :class:`~verso.engine.model.project.Section` instance.
        scale: Override :data:`WORKING_SCALE`. ``None`` uses the default.

    Returns:
        uint8 ``(H, W, C)`` array, or ``None`` if no source image is available.
    """
    working_scale = scale if scale is not None else WORKING_SCALE

    canonical = _canonical_thumbnail(section)
    existing = Path(section.thumbnail_path) if section.thumbnail_path else None

    # 1. Canonical OME-TIFF already on disk.
    if canonical.exists():
        try:
            raw = load_image(canonical)
            img = to_multichannel(raw)
            section.thumbnail_path = str(canonical)
            return img
        except Exception:
            pass

    # 2. Legacy thumbnail (e.g. PNG from old project) — load and migrate.
    if existing and existing != canonical and existing.exists():
        try:
            raw = load_image(existing)
            img = to_multichannel(raw)
            try:
                canonical.parent.mkdir(parents=True, exist_ok=True)
                _save_ome_tiff(img, canonical, channel_names=_default_channel_names(img.shape[2]))
                section.thumbnail_path = str(canonical)
            except OSError:
                pass
            return img
        except Exception:
            pass

    # 3. No cached thumbnail — generate from original image.
    orig = Path(section.original_path)
    if not orig.exists():
        return None

    try:
        raw = load_image(orig)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load image '{orig.name}': {exc}\n\n"
            "If this is a compressed TIFF, make sure 'imagecodecs' is installed."
        ) from exc

    img = to_multichannel(raw)
    img, actual_scale = resize_by_scale(img, working_scale)
    section.scale = actual_scale

    try:
        canonical.parent.mkdir(parents=True, exist_ok=True)
        _save_ome_tiff(img, canonical, channel_names=_default_channel_names(img.shape[2]))
        section.thumbnail_path = str(canonical)
    except OSError:
        pass

    return img


def _default_channel_names(n: int) -> list[str]:
    return [f"Ch {i}" for i in range(n)]


def _save_ome_tiff(
    image: np.ndarray, path: Path, channel_names: list[str]
) -> None:
    """Write an ``(H, W, C)`` uint8 array as a multichannel OME-TIFF.

    Channel names are stored in the OME XML metadata.
    """
    import tifffile

    if image.ndim == 2:
        data = image[np.newaxis, ...]  # (1, H, W)
    else:
        data = np.transpose(image, (2, 0, 1))  # (H, W, C) → (C, H, W)

    tifffile.imwrite(
        str(path),
        data,
        photometric="minisblack",
        metadata={"axes": "CYX", "Channel": {"Name": list(channel_names)}},
    )


def probe_channels(path: str | Path) -> list[str]:
    """Return channel names for a source image without fully decoding it.

    For OME-TIFFs with named channels in the metadata, returns those names.
    Otherwise returns generic ``["Ch 0", "Ch 1", …]`` based on the channel
    count detected from the array shape.
    """
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile
            with tifffile.TiffFile(str(path)) as tif:
                ome_meta = getattr(tif, "ome_metadata", None)
                if ome_meta:
                    names = _extract_ome_channel_names(ome_meta)
                    if names:
                        return names
                series = tif.series[0]
                shape = series.shape
                axes = series.axes
                if "C" in axes:
                    n = int(shape[axes.index("C")])
                    return _default_channel_names(n)
                if "S" in axes:  # samples (RGB-style)
                    n = int(shape[axes.index("S")])
                    return _default_channel_names(n)
                # Fallback: infer from shape
                if len(shape) == 2:
                    return _default_channel_names(1)
                if shape[-1] <= _MAX_PLAUSIBLE_CHANNELS:
                    return _default_channel_names(int(shape[-1]))
                if shape[0] <= _MAX_PLAUSIBLE_CHANNELS:
                    return _default_channel_names(int(shape[0]))
                return _default_channel_names(1)
        except Exception:
            pass

    # PNG / JPG path
    try:
        from PIL import Image
        with Image.open(str(path)) as im:
            mode = im.mode
        if mode in ("L", "1", "I", "F"):
            return _default_channel_names(1)
        if mode == "RGB":
            return _default_channel_names(3)
        if mode == "RGBA":
            return _default_channel_names(3)  # alpha is dropped on load
    except Exception:
        pass
    return _default_channel_names(1)


def _extract_ome_channel_names(ome_xml: str) -> list[str]:
    """Pull <Channel Name="..."> values out of an OME XML string."""
    return re.findall(r'<Channel[^>]*Name="([^"]+)"', ome_xml)


def registration_dimensions(section) -> tuple[int, int]:
    """Return dimensions of the image used for interactive registration.

    Checks the canonical OME-TIFF path first, then the stored thumbnail_path
    (legacy PNG), and falls back to the original image.
    """
    thumbnail = Path(section.thumbnail_path) if section.thumbnail_path else None
    for candidate in (_canonical_thumbnail(section), thumbnail):
        if candidate and candidate.exists():
            return image_dimensions(candidate)
    return image_dimensions(section.original_path)


def load_filmstrip_thumbnail(
    section,
    channels=None,
) -> np.ndarray | None:
    """Return a tiny (≤ FILMSTRIP_MAX_SIDE) grayscale RGB tile for the filmstrip.

    Uses a max-projection across channels so any signal channel contributes.
    The ``channels`` parameter is accepted for API compatibility but ignored —
    filmstrip thumbnails are always grayscale to avoid recompositing on every
    channel change.
    """
    arr = ensure_working_copy(section)
    if arr is None:
        return None

    gray = arr if arr.ndim == 2 else arr.max(axis=2)
    rgb = np.stack([gray, gray, gray], axis=2)
    rgb, _ = resize_to_max_side(rgb, FILMSTRIP_MAX_SIDE)
    return rgb
