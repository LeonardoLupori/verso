# MATLAB Port Reference

## Why this exists

`engine/registration.py` (`VersoRegistration`) is VERSO's public entry point
for scripting/quantification users. Many of those users work in MATLAB, so
`matlab/+verso/VersoRegistration.m` is a native MATLAB port of its three
public operations:

- `coord_image_to_atlas` — pixel(s) on a section -> atlas voxel coordinate(s)
- `coord_atlas_to_image` — atlas voxel(s) -> nearest section pixel(s)
- `image_to_atlas` — resample a whole atlas volume (`annotation` /
  `template` / `boundary`) onto a section's own pixel grid

The MATLAB class is a from-scratch reimplementation (no Python interop, no
`py.*` bridge) so it works on a machine with only MATLAB + Image Processing
Toolbox installed. This doc records the facts that reimplementation depends
on, so future changes to `registration.py` can be ported correctly.

**Critical requirement:** whenever `registration.py`'s public API changes,
`matlab/+verso/VersoRegistration.m` must change with it in the same commit —
same method names, same calling semantics, same numeric results.

## Atlas data: reading BrainGlobe's cache directly

VERSO's Python side uses `brainglobe_atlasapi.BrainGlobeAtlas(atlas_name)` to
get the annotation/reference volumes. That library caches every atlas as a
**plain folder** — no Python needed to read it once it's cached — so the
MATLAB port reads these files directly:

```
{brainglobe_dir}/{atlas_name}_v{X.Y}/
├── annotation.tiff     region-ID volume, multi-page TIFF, uint32
├── reference.tiff      grayscale template volume, same shape
├── structures.json     [{id, name, rgb_triplet:[r,g,b], ...}, ...]
├── metadata.json       {resolution:[um,...], shape:[AP,DV,LR], symmetric, ...}
├── hemispheres.tiff     (not used by VERSO)
└── meshes/              (not used by VERSO)
```

- `brainglobe_dir` defaults to `~/.brainglobe`; overridable via a
  `bg_config.conf` INI file (`[default_dirs] brainglobe_dir = ...`) under
  `$BRAINGLOBE_CONFIG_DIR` or `~/.config/brainglobe/`.
- `annotation.tiff`/`reference.tiff` are read in Python via
  `tifffile.imread(path)`, which returns shape `(pages, rows, cols)` = `(AP,
  DV, LR)` — page axis first. **MATLAB's `tiffreadVolume` (Image Processing
  Toolbox) returns `(rows, cols, pages)` instead** — the page axis is last,
  not first. Every volume load in MATLAB must do:
  ```matlab
  vol = permute(tiffreadVolume(path), [3 1 2]);  % -> (AP, DV, LR), matches Python
  ```
  Getting this permute wrong silently transposes every plane and produces
  garbage that still "looks like" a brain — verify against a known landmark
  (e.g. compare a region ID at a known voxel to the Python side) after any
  change here.
- `structures.json` entries key region IDs to `name`/`rgb_triplet` — same
  role as `AtlasVolume._color_dict` in `engine/atlas.py`.
- `metadata.json.resolution[0]` is `resolution_um`; `metadata.json.shape` is
  `(AP, DV, LR)`, matching `AtlasRef.shape` in `project-verso.json`.

### Auto-download when no local cache exists

If the constructor's `AtlasDir` argument is omitted and no matching
`{atlas_name}_v*` folder exists locally, the MATLAB port downloads the atlas
itself, mirroring `brainglobe_atlasapi/bg_atlas.py`:

1. `webread` the version index:
   `https://gin.g-node.org/brainglobe/atlases/raw/master/last_versions.conf`
   — an INI file, `[atlases]` section, lines `{atlas_name} = X.Y`.
2. Build the download URL:
   `https://gin.g-node.org/brainglobe/atlases/raw/master/{atlas_name}_v{X.Y}.tar.gz`
3. `websave` it to a temp file, then `untar` it into `brainglobe_dir` (MATLAB's
   `untar` handles `.tar.gz` natively) — this creates
   `{brainglobe_dir}/{atlas_name}_v{X.Y}/` with the files listed above.
4. Delete the downloaded archive.

If more than one `{atlas_name}_v*` folder already exists locally, error out
(ambiguous — mirrors `brainglobe_atlasapi`'s own `FileExistsError` for the
same situation) rather than guessing which to use.

## Coordinate conventions (shared with the Python side)

These are unchanged from Python — see [quint-compat.md](quint-compat.md) and
[warping.md](warping.md) for the full derivations. Summary for the MATLAB
port:

- **Anchoring plane**: `atlas_voxel = o + s*u + t*v` for normalized section
  coordinates `(s, t) ∈ [0, 1]²`, where `anchoring = [ox,oy,oz,ux,uy,uz,vx,vy,vz]`
  (9-element vector, exactly as stored in `project-verso.json`).
- **Voxel indexing convention** (`_quicknii_floor_indices` in
  `engine/atlas.py`): given a continuous atlas coordinate `(lr, ap, dv)`,
  the sampled voxel index is `lr_idx = floor(lr)`, `ap_idx = ceil(ap)`,
  `dv_idx = ceil(dv)` — AP/DV use `ceil` (not `floor`) because BrainGlobe's
  raw array has AP/DV reversed relative to QuickNII's convention; this
  asymmetric floor/ceil reproduces VisuAlign/PyNutil's sampling exactly.
  **MATLAB reads the same raw BrainGlobe arrays** (not a QuickNII-reordered
  export), so this exact floor/ceil split must be reproduced as-is. Add `1`
  only when turning the result into a MATLAB array subscript (MATLAB is
  1-indexed; the floor/ceil formulas themselves stay 0-based).
- **Nonlinear warp** (`engine/warping.py`): piecewise-affine Delaunay warp,
  VisuAlign-compatible.
  - Control points are stored in **working-resolution pixels**
    (`ControlPoint.src_x/y`, `dst_x/y` in `project-verso.json`) and are
    normalized to `[0,1]²` internally (divided by `work_w`/`work_h` in
    `prepareWarp`); they then get **four invisible corner anchors** prepended
    before triangulating: `[-0.1,-0.1], [1.1,-0.1], [-0.1,1.1], [1.1,1.1]`
    (identity, src==dst).
  - Before triangulating, points are scaled by `[aspect, 1]` where
    `aspect = work_w / work_h` (the section's working-resolution aspect
    ratio) — Delaunay triangulation is invariant to uniform scaling but not
    to this anisotropic normalization, so this scale must be applied to
    reproduce VisuAlign's triangle topology exactly (see
    `warping.py::_tri_scale`).
  - `warp_points_section_to_atlas` triangulates on the **dst** (section)
    points and interpolates the **src** (atlas) coordinate; the reverse
    direction triangulates on **src**. Both fall back to identity
    (clipped to `[0,1]`) when there are no control points, or when a query
    point falls outside the triangulation's convex hull.
  - **MATLAB equivalent**: `DT = delaunayTriangulation(scaledPoints);
    [triId, bary] = pointLocation(DT, scaledQueryPoints);` — `pointLocation`
    on a 2-D `delaunayTriangulation` returns barycentric coordinates
    directly as the second output (`NaN` row when `triId` is `NaN`, i.e.
    outside the hull), which replaces `scipy.spatial.Delaunay` +
    `tri.transform` in one call. No toolbox required (base MATLAB).
- **Nearest-section search** (`coord_atlas_to_image`'s reverse direction):
  for each section, project the query voxel onto the section's plane via
  `pinv([u v])` (MATLAB's `pinv` is equivalent to `numpy.linalg.pinv`),
  check the projected `(s,t)` falls in `[0,1]²`, and take the plane with
  smallest perpendicular distance (`|dot(rel, normalize(cross(u,v)))|`)
  among sections whose footprint covers the point.

## Cross-language numerical parity

The MATLAB port is kept in numerical lockstep with the Python engine by a
**golden-fixture contract** rather than by eye:

- `tests/engine/test_matlab_parity.py` (Python) runs the live engine over a fixed
  set of inputs and writes the outputs to `matlab/tests/fixtures/parity.json`
  (plus one project file per coord case under `fixtures/projects/`). In normal
  mode it re-derives those outputs and asserts they still equal the committed
  fixture, so any Python numeric change fails CI until the fixture is regenerated
  on purpose. This runs in the existing GitHub Actions job.
- `matlab/tests/tParity.m` (MATLAB) reads the *same* JSON, runs the MATLAB
  implementation over the identical inputs, and asserts equality within
  `AbsTol = 1e-9`. It is run locally with `runtests('matlab/tests')` (MATLAB is
  not in CI).

JSON is the language-neutral boundary — MATLAB never runs Python. Transitively
`python == fixture` and `matlab == fixture` give `python == matlab`.

Coverage: both warp directions, `anchoringToVectors`, `quickniiVoxelIndices`
(the floor/ceil convention), and the public `coord_image_to_atlas` /
`coord_atlas_to_image` end-to-end. The private numeric primitives are reached
from the test via `Hidden, Static` `*ForTesting` forwarders on
`VersoRegistration` (same pattern as `setAtlasVolumeForTesting`) — they are not
public API. `image_to_atlas` (array output) is not yet covered.

**Workflow after an intentional numeric change** (either language):

```bash
UPDATE_PARITY_FIXTURES=1 uv run pytest tests/engine/test_matlab_parity.py  # regenerate
runtests('matlab/tests')                                                   # confirm MATLAB
# commit matlab/tests/fixtures/ together with the code change
```

## File layout

```
matlab/
├── +verso/
│   ├── VersoRegistration.m      public API (classdef, handle)
│   │                            + Hidden Static *ForTesting warp/anchoring/voxel forwarders
│   └── private/                  helpers, package-internal only
│       ├── loadProjectJson.m
│       ├── anchoringToVectors.m
│       ├── warpPointsSectionToAtlas.m   thin wrapper over barycentricMap
│       ├── warpPointsAtlasToSection.m   thin wrapper (roles swapped)
│       ├── prepareWarp.m                shared normalise + [aspect,1] scale
│       ├── withCorners.m                prepend the four corner anchors
│       ├── barycentricMap.m             shared triangulate + interpolate core
│       ├── quickniiVoxelIndices.m
│       ├── boundaryMask.m
│       ├── resolveAtlasDir.m
│       ├── downloadBrainglobeAtlas.m
│       └── loadAtlasVolume.m
├── tests/
│   ├── tVersoRegistration.m      matlab.unittest, offline (fake in-memory atlas)
│   ├── tParity.m                 cross-language parity vs the Python golden fixture
│   └── fixtures/                 parity.json + projects/*.json (generated by Python)
└── README.md
```

`+verso` is a MATLAB package folder, so usage is:

```matlab
r = verso.VersoRegistration("my_experiment/project-verso.json");
xyz = r.coord_image_to_atlas("s001", [[1200, 3400]; [1500, 3600]]);
res = r.coord_atlas_to_image(xyz);
labels = r.image_to_atlas("s001", "Kind", "annotation");
```

Requires MATLAB with the **Image Processing Toolbox** (for `tiffreadVolume`)
and, for the one-time atlas download path, internet access. Everything else
(JSON parsing, anchoring/warp math, atlas indexing) is base MATLAB.
