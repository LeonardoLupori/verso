"""Local script (not committed): visualise the LR-mirror relationship between the
verso/VisuAlign atlas and the MATLAB-pipeline annotation volume.

All three sources are the same Allen CCFv3 2017 25um annotation (same region IDs):
  - verso/VisuAlign .cutlas labels.nii.gz   -- (LR, AP, DV), LR index 0 = RIGHT
  - MATLAB annotationVolume.mat             -- (LR, AP, DV), LR index 0 = LEFT
verso's cutlas is bit-identical to VisuAlign's; the MATLAB volume is the *exact*
LR mirror (flip axis 0 -> perfect label match; AP/DV unchanged). The atlas itself
is only ~98.4% LR-symmetric, so the mirror is real, not a symmetry artefact.

Each row is one coronal level; the columns are:
  1. verso / VisuAlign            (reference)
  2. MATLAB (raw)                 -- looks mirrored: its R/L is swapped vs col 1
  3. MATLAB . LR-flip             -- flipping axis 0 reproduces col 1
  4. col1 XOR col3 (red)          -- mismatches; essentially empty == exact mirror

Regions are coloured with the Allen RGB palette from the .cutlas labels.txt, so
the slices look like real atlas sections. "R"/"L" markers on column 1 vs column 2
make the hemisphere swap obvious.

Two modes:
  - default      : render a PNG montage with PIL (works with installed deps) and
                   open it.
  - --interactive: matplotlib window with an AP slider. matplotlib is not a project
                   dependency, so run that mode via:
                       uv run --with matplotlib python visualize_atlas_lr_mirror.py --interactive

Usage:
    uv run python visualize_atlas_lr_mirror.py [--plane coronal|axial]
        [--levels N] [--out atlas_lr_mirror.png] [--no-open]
        [--ref CUTLAS_NII] [--matlab MAT] [--labels LABELS_TXT]
    uv run --with matplotlib python visualize_atlas_lr_mirror.py --interactive
"""

from __future__ import annotations

import argparse
import gzip
import struct
import sys
from pathlib import Path

import numpy as np

# --- default source locations -------------------------------------------------
DEF_REF = Path(r"c:\Users\Valentino\Documents\Python\verso\allen_mouse_25um.cutlas\labels.nii.gz")
DEF_MAT = Path(
    r"C:\Users\Valentino\Documents\MATLAB\PipelineOfficial\brainAlignment\annotationVolume.mat"
)

_NIfTI_DTYPE = {2: "<u1", 4: "<i2", 8: "<i4", 16: "<f4", 768: "<u4", 512: "<u2"}


# --- loaders ------------------------------------------------------------------
def read_cutlas_nii(path: Path) -> np.ndarray:
    """Read a .cutlas labels.nii.gz into an (LR, AP, DV) uint array."""
    with gzip.open(path, "rb") as f:
        blob = f.read()
    hdr = blob[:348]
    x, y, z = (struct.unpack_from("<h", hdr, 42 + 2 * i)[0] for i in range(3))
    dt = struct.unpack_from("<h", hdr, 70)[0]
    vox = int(struct.unpack_from("<f", hdr, 108)[0])
    data = np.frombuffer(blob[vox:], dtype=_NIfTI_DTYPE[dt])[: x * y * z]
    return data.reshape((x, y, z), order="F")  # NIfTI x-fastest


def read_matlab_mat(path: Path) -> np.ndarray:
    """Read MATLAB v7.3 annotationVolume.mat into MATLAB-native (LR, AP, DV)."""
    import h5py

    with h5py.File(path, "r") as f:
        # h5py reads HDF5 in C order; MATLAB is column-major -> reverse axes.
        return f["annotationVolume"][:].transpose(2, 1, 0)


def parse_labels_txt(path: Path) -> dict[int, tuple[int, int, int]]:
    """Parse an ITK-SNAP labels.txt into {label_id: (r, g, b)}."""
    lut: dict[int, tuple[int, int, int]] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            idx, r, g, b = (int(parts[i]) for i in range(4))
        except ValueError:
            continue
        lut[idx] = (r, g, b)
    return lut


# --- slicing & colouring ------------------------------------------------------
def get_slice(vol: np.ndarray, plane: str, idx: int) -> np.ndarray:
    """Return an anatomically-oriented 2D label slice (dorsal/anterior up, LR cols).

    Frame is (LR, AP, DV) with LR col 0 on the left of the image.
    """
    if plane == "coronal":          # fix AP -> (DV, LR), dorsal up
        sl = vol[:, idx, :].T[::-1, :]
    elif plane == "axial":          # fix DV -> (AP, LR), anterior up
        sl = vol[:, :, idx].T[::-1, :]
    else:
        raise ValueError(f"unknown plane {plane!r}")
    return sl


def colorize(label_slice: np.ndarray, lut: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """Map a 2D label slice to an (H, W, 3) uint8 RGB image via the Allen palette."""
    uniq, inv = np.unique(label_slice, return_inverse=True)
    pal = np.array([lut.get(int(u), (0, 0, 0)) for u in uniq], dtype=np.uint8)
    return pal[inv].reshape(label_slice.shape + (3,))


def diff_image(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Red where labels differ, dim grey foreground elsewhere (mismatch map)."""
    rgb = np.zeros(a.shape + (3,), dtype=np.uint8)
    fg = (a > 0) | (b > 0)
    rgb[fg] = (45, 45, 45)
    rgb[a != b] = (255, 30, 30)
    return rgb


def pick_levels(vol: np.ndarray, plane: str, n: int) -> list[int]:
    """Pick *n* evenly spaced slice indices within the foreground extent."""
    axis = {"coronal": 1, "axial": 2}[plane]
    occ = np.any(vol > 0, axis=tuple(i for i in range(3) if i != axis))
    idx = np.flatnonzero(occ)
    lo, hi = (int(idx[0]), int(idx[-1])) if idx.size else (0, vol.shape[axis] - 1)
    return [int(round(lo + (hi - lo) * (k + 1) / (n + 1))) for k in range(n)]


# --- PNG montage (PIL) --------------------------------------------------------
def render_montage(ref, mat, lut, plane, levels, out_path: Path, asym_pct: float):
    from PIL import Image, ImageDraw

    mat_flip = mat[::-1, :, :]  # LR mirror of the MATLAB volume
    col_titles = ["verso / VisuAlign", "MATLAB (raw)", "MATLAB · LR-flip", "ref vs flip (red=diff)"]
    pad, head = 6, 18
    rows = []
    for a in levels:
        ref_l = get_slice(ref, plane, a)
        mat_l = get_slice(mat, plane, a)
        flip_l = get_slice(mat_flip, plane, a)
        panels = [
            colorize(ref_l, lut),
            colorize(mat_l, lut),
            colorize(flip_l, lut),
            diff_image(ref_l, flip_l),
        ]
        h, w = ref_l.shape
        strip = np.full((h, pad, 3), 255, np.uint8)
        row = strip.copy()
        for p in panels:
            row = np.concatenate([row, p, strip], axis=1)
        rows.append((a, row))

    total_w = rows[0][1].shape[1]
    sep = np.full((pad, total_w, 3), 255, np.uint8)
    body = sep.copy()
    for _, row in rows:
        body = np.concatenate([body, row, sep], axis=0)

    img = Image.fromarray(body)
    canvas = Image.new("RGB", (total_w, head + img.height), (255, 255, 255))
    canvas.paste(img, (0, head))
    draw = ImageDraw.Draw(canvas)

    panel_w = ref.shape[2] if plane == "coronal" else ref.shape[1]  # LR width
    panel_h = rows[0][1].shape[0]
    # column headers
    for c, title in enumerate(col_titles):
        x = pad + c * (panel_w + pad)
        draw.text((x + 2, 4), title, fill=(0, 0, 0))
    # R/L markers + slice index on each row
    for r, (a, _) in enumerate(rows):
        y0 = head + pad + r * (panel_h + pad)
        # col1 = verso: R on left, L on right
        draw.text((pad + 2, y0 + 2), "R", fill=(255, 255, 0))
        draw.text((pad + panel_w - 9, y0 + 2), "L", fill=(255, 255, 0))
        # col2 = MATLAB raw: swapped
        x2 = pad + (panel_w + pad)
        draw.text((x2 + 2, y0 + 2), "L", fill=(255, 255, 0))
        draw.text((x2 + panel_w - 9, y0 + 2), "R", fill=(255, 255, 0))
        draw.text((pad + 2, y0 + panel_h - 12), f"{plane[:3]} {a}", fill=(0, 255, 255))

    canvas.save(out_path)
    print(f"wrote {out_path}  ({canvas.width}x{canvas.height})")
    print(f"verso == VisuAlign; MATLAB = exact LR mirror; atlas self-asymmetry {asym_pct:.2f}%")


# --- interactive (matplotlib) -------------------------------------------------
def run_interactive(ref, mat, lut, plane, asym_pct: float):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    mat_flip = mat[::-1, :, :]
    axis = {"coronal": 1, "axial": 2}[plane]
    n = ref.shape[axis]
    a0 = pick_levels(ref, plane, 1)[0]

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    fig.subplots_adjust(bottom=0.18, wspace=0.05)
    titles = ["verso / VisuAlign", "MATLAB (raw)", "MATLAB · LR-flip", "ref vs flip (red=diff)"]

    def panels(a):
        ref_l, mat_l, flip_l = (get_slice(v, plane, a) for v in (ref, mat, mat_flip))
        return [colorize(ref_l, lut), colorize(mat_l, lut),
                colorize(flip_l, lut), diff_image(ref_l, flip_l)]

    ims = []
    for ax, im, t in zip(axes, panels(a0), titles):
        ims.append(ax.imshow(im))
        ax.set_title(t, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].text(0.02, 0.97, "R", color="y", transform=axes[0].transAxes, va="top")
    axes[0].text(0.95, 0.97, "L", color="y", transform=axes[0].transAxes, va="top")
    axes[1].text(0.02, 0.97, "L", color="y", transform=axes[1].transAxes, va="top")
    axes[1].text(0.95, 0.97, "R", color="y", transform=axes[1].transAxes, va="top")
    fig.suptitle(
        f"{plane} — verso==VisuAlign, MATLAB=exact LR mirror, "
        f"self-asymmetry {asym_pct:.2f}%"
    )

    sax = fig.add_axes([0.15, 0.06, 0.7, 0.03])
    slider = Slider(sax, plane[:3] + " idx", 0, n - 1, valinit=a0, valstep=1)

    def update(val):
        for im_obj, arr in zip(ims, panels(int(slider.val))):
            im_obj.set_data(arr)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", type=Path, default=DEF_REF, help="verso/VisuAlign labels.nii.gz")
    ap.add_argument("--matlab", type=Path, default=DEF_MAT, help="MATLAB annotationVolume.mat")
    ap.add_argument("--labels", type=Path, default=None, help="labels.txt (default: next to --ref)")
    ap.add_argument("--plane", choices=("coronal", "axial"), default="coronal")
    ap.add_argument("--levels", type=int, default=4, help="number of slice rows (montage)")
    ap.add_argument("--out", type=Path, default=Path("atlas_lr_mirror.png"))
    ap.add_argument("--interactive", action="store_true", help="matplotlib slider instead of a PNG")
    ap.add_argument("--no-open", action="store_true", help="don't auto-open the PNG")
    args = ap.parse_args(argv)

    for p in (args.ref, args.matlab):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1
    labels_txt = args.labels or (args.ref.parent / "labels.txt")
    if not labels_txt.exists():
        print(f"ERROR: missing labels.txt ({labels_txt})", file=sys.stderr)
        return 1

    print("loading volumes…")
    ref = read_cutlas_nii(args.ref)
    mat = read_matlab_mat(args.matlab)
    lut = parse_labels_txt(labels_txt)
    if ref.shape != mat.shape:
        print(f"WARNING: shape mismatch ref{ref.shape} vs matlab{mat.shape}", file=sys.stderr)
    asym_pct = float((ref != ref[::-1, :, :]).mean()) * 100.0

    if args.interactive:
        try:
            run_interactive(ref, mat, lut, args.plane, asym_pct)
        except ImportError:
            print("matplotlib not available — run:\n"
                  "  uv run --with matplotlib python visualize_atlas_lr_mirror.py --interactive",
                  file=sys.stderr)
            return 1
        return 0

    levels = pick_levels(ref, args.plane, args.levels)
    render_montage(ref, mat, lut, args.plane, levels, args.out, asym_pct)
    if not args.no_open and sys.platform == "win32":
        import os
        os.startfile(args.out.resolve())  # noqa: S606 - local convenience
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
