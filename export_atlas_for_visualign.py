"""Export the BrainGlobe atlas as a VisuAlign-compatible .cutlas directory.

Usage:
    uv run python export_atlas_for_visualign.py
    uv run python export_atlas_for_visualign.py --atlas allen_mouse_25um --output D:/my_dir
"""

import argparse
import time
from pathlib import Path

from verso.engine import export_brainglobe_atlas_for_visualign

DEFAULT_ATLAS = "allen_mouse_25um"
DEFAULT_OUTPUT = Path(__file__).parent  # same folder as this script

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--atlas", default=DEFAULT_ATLAS, help=f"BrainGlobe atlas name (default: {DEFAULT_ATLAS})")
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory to create the .cutlas folder in")
args = parser.parse_args()

print(f"Loading BrainGlobe atlas '{args.atlas}' ...")
t0 = time.time()

cutlas_dir = export_brainglobe_atlas_for_visualign(args.atlas, args.output)

elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")
print(f"Created: {cutlas_dir}")
print()
print("Next step: copy the folder above into your VisuAlign atlas directory")
print(f"  e.g.  D:/proj/VisuAlign-v0_9/{cutlas_dir.name}")
