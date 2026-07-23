# QuickNII / VisuAlign Compatibility Reference

## Why this matters

VERSO must be a drop-in replacement for the QuickNII → VisuAlign pipeline. Users must be able to:

1. Open a QuickNII JSON file and see their registered sections
2. Open a VisuAlign JSON file with control points and continue refining
3. Save results that QuickNII, VisuAlign, PyNutil, and Nutil can read
4. Import DeepSlice output as initial registration

This is the single most important I/O compatibility requirement.

## QUINT ecosystem tools

| Tool | Function | Format |
|---|---|---|
| QuickNII | Affine 2D-to-3D registration | JSON with anchoring matrix per section |
| VisuAlign | Nonlinear refinement | Extends QuickNII JSON with control point data |
| WebAlign / WebWarp | Web-based successors | Same JSON format |
| DeepSlice | Automatic initial registration | Outputs QuickNII-compatible JSON |
| PyNutil | Python quantification library | Reads QuickNII/VisuAlign JSON |
| Nutil | C++ quantification | Reads QuickNII/VisuAlign JSON |
| MeshView | 3D point cloud viewer | Reads point cloud CSV/JSON |

## QuickNII JSON format

### Top-level structure

```json
{
  "name": "My Experiment",
  "target": "allen_mouse_25um",
  "slices": [ ... ]
}
```

### Per-section fields

```json
{
  "filename": "IMG_0234.png",
  "nr": 234,
  "anchoring": [ox, oy, oz, ux, uy, uz, vx, vy, vz],
  "width": 1000,
  "height": 750
}
```

| Field | Type | Meaning |
|---|---|---|
| `filename` | string | Image filename (base name only) |
| `nr` | int | Section serial number |
| `anchoring` | float[9] | 9-element array encoding the cut plane in atlas voxel space |
| `width` | int | Image width in pixels (working resolution) |
| `height` | int | Image height in pixels (working resolution) |

### Anchoring semantics

The `anchoring` array encodes the position and orientation of the 2D section within the 3D atlas volume. For a normalised section coordinate `(s, t)` where both are in `[0, 1]`:

```
atlas_voxel = [ox, oy, oz] + s * [ux, uy, uz] + t * [vx, vy, vz]
```

- `(ox, oy, oz)` — origin: atlas voxel at the section's top-left corner (s=0, t=0)
- `(ux, uy, uz)` — u-vector: atlas displacement across the full section width (s: 0→1)
- `(vx, vy, vz)` — v-vector: atlas displacement across the full section height (t: 0→1)

Atlas coordinate convention (Allen Mouse 25µm) — QuickNII voxel space:
- x (component 0): left → right (LR axis); 0 = leftmost, 455 = rightmost
- y (component 1): posterior → anterior (AP axis); 0 = most posterior, 527 = most anterior
- z (component 2): inferior → superior (DV axis); 0 = most inferior

VERSO internal convention (BrainGlobe) uses the same component ordering [LR, AP, DV]
but with the AP and DV axes inverted:
- component 1: 0 = most anterior (opposite to QuickNII y)
- component 2: 0 = most dorsal (opposite to QuickNII z)

`_to_quicknii_convention()` converts between the two by mirroring the AP/DV origin
and negating their vectors: `[LR, AP_bg, DV_bg] → [LR, (AP_max−1)−AP_bg, (DV_max−1)−DV_bg]`.
The offset is `N−1`, not `N`, because the QuickNII/VisuAlign atlas volume is the
BrainGlobe annotation with AP/DV *array-reversed* (`annotation[::-1, ::-1, :]`, index
`i → N−1−i`); flipping the continuous origin about `N` shifts the sampled plane by one
voxel (verified 100% vs the stock `.cutlas` with `N−1`, ~93.5% with `N`). The function
is self-inverse so it also converts QuickNII → BrainGlobe.

### Slicing plane is not stored — it is inferred

QuickNII/VisuAlign JSON has **no field for the cutting plane** (coronal / sagittal /
horizontal). The plane is implicit in the anchoring: the `u`/`v` vectors span the two
in-plane atlas axes, so the slicing axis is the third — the direction of the plane
normal `u × v`. On load, `load_quicknii()` recovers it via
`infer_interpolation_axis()` (dominant axis of the summed unit normals across the
series) and sets `Project.interpolation_axis`, so an imported project is complete
(right orientation + interpolation axis) instead of always defaulting to coronal/AP.
The inference is convention-independent: the BrainGlobe↔QuickNII AP/DV sign flip
negates normal components without changing which axis dominates. The Import dialog
presets its editable "Slicing orientation" combo from this and can override it.

### QuickNII interpolation (internal)

QuickNII converts anchorings to an **unpacked 11-component** format for interpolation:

```
[mid_x, mid_y, mid_z, unit_ux, unit_uy, unit_uz, unit_vx, unit_vy, unit_vz, u_stretch, v_stretch]
```

Where:
- `mid = o + 0.5*u + 0.5*v` — section midpoint in atlas space
- `unit_u = u / |u|`, `u_stretch = |u|`
- `unit_v = v / |v|`, `v_stretch = |v|`

Interpolation (linear regression / piecewise linear) is performed in unpacked space, then repacked to the 9-component anchoring format. VERSO's `propagate_series_anchorings()` in `engine/anchoring.py` mirrors this algorithm and parameterizes it on the project's `interpolation_axis` (the original QuickNII algorithm is coronal-only).

## VisuAlign JSON format

VisuAlign extends QuickNII JSON by adding a `markers` array to each section:

```json
{
  "filename": "IMG_0234.png",
  "nr": 234,
  "anchoring": [ox, oy, oz, ux, uy, uz, vx, vy, vz],
  "width": 1000,
  "height": 750,
  "markers": [
    [520.0, 232.5, 540.0, 220.0],
    [180.0, 525.0, 170.0, 540.0]
  ]
}
```

### Marker semantics

Each marker is a control point (warp pin) stored as a **4-element array in
section pixel coordinates** at the working resolution (`width`/`height` above).
This is VisuAlign's native format — confirmed against its source
(`data/Marker.java`: `marker(ox, oy, nx, ny)`):

```
[ox, oy, nx, ny]  =  [atlas_x_px, atlas_y_px, section_x_px, section_y_px]
```

| Component | Meaning |
|---|---|
| `ox, oy` (0,1) | "original" — atlas-overlay position (the **src** pin location) |
| `nx, ny` (2,3) | "new" — where it was dragged to on the section (the **dst**) |

VisuAlign builds its Delaunay triangulation on `(nx, ny)` (section pixels) and
`transform` interpolates `(ox, oy)` (atlas pixels) — i.e. a backward section→atlas
map (`nonlin/Triangle.java`, `visualign/QNLController.java#sample`). VERSO's warp
matches this direction (triangulate on dst, interpolate src). **For the interior
warp to match VisuAlign, the triangulation must be done in section *pixel* space,
not normalised `[0,1]²`** — see [warping.md](warping.md#triangulation-space--aspect-ratio-visualign-parity).

VERSO stores control points internally as `ControlPoint(src_x, src_y, dst_x, dst_y)`
in **working-resolution pixel coordinates** (src = atlas overlay, dst = section
image) — the same units as the VisuAlign marker itself, so no conversion happens
at the I/O boundary:

- **Load**: `src = (ox, oy)`, `dst = (nx, ny)` — pixel values pass straight through.
- **Save**: `[src_x, src_y, dst_x, dst_y]` — pixel values pass straight through
  (see `_control_points_to_markers` in `engine/io/quint_io.py`).

A legacy normalised-dict form `{"x", "y", "dx", "dy"}` (in `[0, 1]`, where
`dst = (x+dx, y+dy)`) is still accepted on load for backward compatibility with
old VERSO exports — that legacy form *is* normalised and gets multiplied by
`(w, h)` on load — but is never written.

> **Corner anchors are not exported.** Both tools synthesise four identity
> anchors 10% outside the frame on load (`_CORNERS` / `Slice.triangulate()`), so
> only real markers belong in the JSON. Do **not** inject image-corner markers
> into exports — that double-anchors the border and breaks parity.

## DeepSlice output

DeepSlice outputs QuickNII-compatible JSON (same `anchoring` field, same top-level structure). No `markers` field. VERSO can load it directly via the QuickNII reader.

Known differences from QuickNII output:
- May include a `confidence` field per section (ignored on load)
- Section filenames may use full paths rather than base names

## Implementation in VERSO

The compatibility layer lives in `engine/io/quint_io.py`.

### Public functions

| Function | Purpose |
|---|---|
| `load_quicknii(path, atlas_name=...)` | Load a QuickNII JSON (no markers) → `Project`. |
| `load_visualign(path, ...)` | Load a VisuAlign JSON (with markers) → `Project`. |
| `save_quicknii(project, path, atlas_shape=...)` | Write QuickNII JSON (no markers). |
| `save_quicknii_xml(project, path, atlas_shape=...)` | Write the QuickNII XML variant. |
| `save_visualign(project, path, atlas_shape=...)` | Write VisuAlign JSON (sections with control points emit a `markers` array). |

These are exposed in the File menu (Import QuickNII, Open VisuAlign,
Export QuickNII XML / JSON, Export VisuAlign JSON).

Internal helpers:
- `_markers_to_control_points(markers, width, height)` — `[[ox, oy, nx, ny]]`
  pixel arrays (or legacy `{x, y, dx, dy}` dicts) → `[ControlPoint]`
  (working-resolution pixels; only the legacy normalised dicts are multiplied by `width`/`height`).
- `_control_points_to_markers(cps)` — `[ControlPoint]` →
  `[[src_x, src_y, dst_x, dst_y]]` pixel arrays (rounded to 6 dp; pixel values pass
  straight through, no scaling).

### Round-trip guarantee

Loading a QuickNII/VisuAlign JSON and immediately saving it must produce a file that is semantically identical (same anchoring values, same marker values) to within floating-point precision. The `tests/engine/test_quint_io.py` suite enforces this.
