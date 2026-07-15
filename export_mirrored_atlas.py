"""Local script (not committed): make an LR-mirrored copy of a VisuAlign .cutlas atlas.

Produces an "alternative" atlas mirrored in the coronal plane — the left/right
hemispheres are swapped (the LR axis is reversed; AP and DV are unchanged). Useful
when sections are mounted/imaged with the opposite L/R handedness from the standard
atlas.

It reads ``{source}.cutlas/labels.nii.gz``, reverses the LR (x) axis of the volume,
and writes ``{name}.cutlas/`` with the **same NIfTI header** (so it stays
VisuAlign-valid) and an unchanged ``labels.txt`` (region IDs are unaffected by a
spatial mirror).

NIfTI stores the first axis (x = LR) fastest, so the on-disk voxels are Fortran
order for the (LR, AP, DV) volume; the mirror reverses that first axis.

Usage:
    python export_mirrored_atlas.py SOURCE.cutlas [--output DIR] [--name NAME]

    # e.g. allen_mouse_25um.cutlas -> allen_mouse_25um_mirror.cutlas (same parent)
    python export_mirrored_atlas.py "D:/.../VisuAlign-v0_9/allen_mouse_25um.cutlas"
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

_DTYPE = {2: "<u1", 4: "<i2", 8: "<i4", 16: "<f4", 768: "<u4", 512: "<u2"}


def mirror_cutlas(source: Path, out_dir: Path, name: str) -> Path:
    """Write an LR-mirrored copy of *source* .cutlas into ``out_dir/name.cutlas``."""
    src_nii = source / "labels.nii.gz"
    src_txt = source / "labels.txt"
    if not src_nii.exists():
        raise FileNotFoundError(f"no labels.nii.gz in {source}")

    with gzip.open(src_nii, "rb") as f:
        blob = f.read()

    header = blob[:348]
    # dims: dim[1]=x(LR) @42, dim[2]=y(AP) @44, dim[3]=z(DV) @46
    x = struct.unpack_from("<h", header, 42)[0]
    y = struct.unpack_from("<h", header, 44)[0]
    z = struct.unpack_from("<h", header, 46)[0]
    datatype = struct.unpack_from("<h", header, 70)[0]
    vox_offset = int(struct.unpack_from("<f", header, 108)[0])
    if datatype not in _DTYPE:
        raise ValueError(f"unsupported NIfTI datatype {datatype}")

    data = np.frombuffer(blob[vox_offset:], dtype=_DTYPE[datatype])
    if data.size != x * y * z:
        raise ValueError(
            f"voxel count {data.size} != {x}*{y}*{z}={x * y * z}; header/data mismatch"
        )

    # NIfTI x-fastest -> Fortran order for (LR, AP, DV). Reverse the LR axis.
    vol = data.reshape((x, y, z), order="F")
    mirrored = vol[::-1, :, :]

    cutlas_dir = out_dir / f"{name}.cutlas"
    cutlas_dir.mkdir(parents=True, exist_ok=True)

    # Keep the byte region before the voxel data (header + any padding) verbatim,
    # so the output is identical to the source except for the mirrored voxels.
    prefix = blob[:vox_offset]
    with gzip.open(cutlas_dir / "labels.nii.gz", "wb", compresslevel=6) as f:
        f.write(prefix)
        f.write(np.ascontiguousarray(mirrored).tobytes(order="F"))

    if src_txt.exists():
        shutil.copy2(src_txt, cutlas_dir / "labels.txt")

    return cutlas_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="Source .cutlas directory")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (default: same parent as source).",
    )
    parser.add_argument(
        "--name", default=None,
        help="Output atlas name without .cutlas (default: '{source}_mirror').",
    )
    args = parser.parse_args(argv)

    source: Path = args.source
    if source.suffix == "":
        # allow passing without trailing slash issues; ensure it's a dir
        pass
    if not source.is_dir():
        print(f"ERROR: not a directory: {source}", file=sys.stderr)
        return 1

    out_dir = args.output or source.parent
    name = args.name or f"{source.name.removesuffix('.cutlas')}_mirror"

    try:
        cut = mirror_cutlas(source, out_dir, name)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote LR-mirrored atlas: {cut}")
    print("Point your VisuAlign JSON 'target' at this folder name to use it:")
    print(f"  --target {cut.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
