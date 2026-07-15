# VERSO for MATLAB

A native MATLAB port of `engine/registration.py`'s public API
(`VersoRegistration`) — no Python installation or `py.*` bridge required.
Given a `project-verso.json` written by VERSO, it lets you:

- map image pixels on a section to Allen CCFv3 atlas coordinates and back
  (`coord_image_to_atlas`, `coord_atlas_to_image`)
- resample a whole atlas volume (region labels / grayscale template /
  region-boundary mask) onto a section's own pixel grid
  (`image_to_atlas`), pixel-matched 1:1 to the original image — the same
  kind of output QuickNII → VisuAlign → PyNutil would produce, without
  leaving MATLAB.

## Requirements

- MATLAB with the **Image Processing Toolbox** (for `tiffreadVolume`).
- Internet access **the first time** `image_to_atlas` is used for a given
  atlas, unless you already have it cached locally (from having run VERSO's
  Python side at least once) or pass `AtlasDir` explicitly. `coord_image_to_atlas`
  / `coord_atlas_to_image` never need the atlas volume and work fully offline.

## Setup

```matlab
addpath("path/to/verso/matlab");
```

## Usage

```matlab
r = verso.VersoRegistration("my_experiment/project-verso.json");

% Pixel -> atlas voxel, and back
xyz = r.coord_image_to_atlas("s001", [1200 3400; 1500 3600]);
res = r.coord_atlas_to_image(xyz);   % res.section_id, res.xy, res.distance, res.valid

% Whole-image atlas resampling, pixel-matched to the section
labels   = r.image_to_atlas("s001");                              % annotation (default)
template = r.image_to_atlas("s001", "Kind", "template");
boundary = r.image_to_atlas("s001", "Kind", "boundary");
```

A section is addressed by its id, or by the original image's file stem or
basename (same resolution rules as the Python side).

### Atlas volume location

`image_to_atlas` needs the actual BrainGlobe atlas volumes (annotation +
reference). By default it:

1. Looks for an already-downloaded copy under the standard BrainGlobe cache
   (`~/.brainglobe/{atlas_name}_v*/`).
2. If not found, downloads it from the BrainGlobe/GIN remote automatically.

To point at a specific folder instead (e.g. a copy on a shared drive, or a
machine that has never run Python at all):

```matlab
r = verso.VersoRegistration("project-verso.json", "AtlasDir", "D:/atlases/allen_mouse_25um_v1.2");
```

That folder must contain `annotation.tiff`, `reference.tiff`,
`structures.json`, and `metadata.json` (the standard BrainGlobe atlas folder
layout — see `.claude/matlab-port.md` in the repo root).

## Tests

```matlab
runtests("matlab/tests")
```

Tests run fully offline against small synthetic projects/atlases — no
network access or real BrainGlobe atlas required.

## Maintenance

This port must track `engine/registration.py` — see the "MATLAB parity for
`registration.py`" note in the repo's `CLAUDE.md` and the full reference in
`.claude/matlab-port.md`.
