"""Local script (not committed): write an LR-flipped annotationVolume.mat.

The MATLAB pipeline's ``annotationVolume.mat`` stores the Allen CCFv3 2017 25um
annotation with **opposite left/right handedness** from VisuAlign/VERSO: its LR
index 0 is the *Left* hemisphere, whereas VisuAlign/VERSO (and the .cutlas
atlases) use LR index 0 = *Right*. The two volumes are otherwise identical — a
pure left-right mirror (AP/DV unchanged), verified by exact label match.

This produces a copy of ``annotationVolume.mat`` with the LR axis reversed, so it
carries the **same handedness as VisuAlign and VERSO**. The output is byte-for-byte
the same MATLAB v7.3/HDF5 container (same variable name ``annotationVolume``, same
uint32 dtype, same ``MATLAB_class`` attribute) — only the voxel values are mirrored
— so the MATLAB pipeline can ``load`` it exactly like the original.

The source HDF5 dataset is stored as (DV, AP, LR) in h5py/C order (MATLAB loads it
transposed to native (LR, AP, DV) = (456, 528, 320)); LR is the last axis here, so
the mirror reverses axis 2.

Usage:
    python export_annotation_volume_mat.py [SOURCE.mat] [--output OUT.mat]

    # default: read the pipeline's annotationVolume.mat, write
    # annotationVolume_visualign.mat next to it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

DEFAULT_SOURCE = Path(
    r"C:\Users\Valentino\Documents\MATLAB\PipelineOfficial"
    r"\brainAlignment\annotationVolume.mat"
)
VAR_NAME = "annotationVolume"
LR_AXIS = 2  # h5py/C-order storage is (DV, AP, LR); LR is the last axis.


def lr_flip_mat(source: Path, output: Path) -> None:
    """Copy *source* .mat to *output* with the LR axis of ``annotationVolume`` reversed."""
    shutil.copy2(source, output)
    with h5py.File(output, "r+") as f:
        if VAR_NAME not in f:
            raise KeyError(f"{output} has no '{VAR_NAME}' variable (found {list(f.keys())})")
        dset = f[VAR_NAME]
        vol = dset[...]  # (DV, AP, LR)
        flipped = np.flip(vol, axis=LR_AXIS)
        changed = np.count_nonzero(vol != flipped)
        dset[...] = flipped
    pct = 100.0 * changed / vol.size
    print(f"  shape={vol.shape} dtype={vol.dtype}, voxels changed by LR flip: "
          f"{changed:,} ({pct:.3f}%)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, nargs="?", default=DEFAULT_SOURCE,
                        help=f"Source annotationVolume.mat (default: {DEFAULT_SOURCE})")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output .mat path (default: '{source}_visualign.mat').")
    args = parser.parse_args(argv)

    source: Path = args.source
    if not source.is_file():
        print(f"ERROR: no such file: {source}", file=sys.stderr)
        return 1
    output: Path = args.output or source.with_name(f"{source.stem}_visualign.mat")

    print(f"{source}\n  -> {output}")
    try:
        lr_flip_mat(source, output)
    except (OSError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote LR-flipped annotation volume (VisuAlign/VERSO handedness): {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
