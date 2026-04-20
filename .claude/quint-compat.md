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
  "sections": [ ... ]
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

Atlas coordinate convention (Allen Mouse 25µm):
- x axis: posterior → anterior (AP axis); 0 = most posterior
- y axis: inferior → superior (DV axis)
- z axis: left → right (ML axis)

### QuickNII interpolation (internal)

QuickNII converts anchorings to an **unpacked 11-component** format for interpolation:

```
[mid_x, mid_y, mid_z, unit_ux, unit_uy, unit_uz, unit_vx, unit_vy, unit_vz, u_stretch, v_stretch]
```

Where:
- `mid = o + 0.5*u + 0.5*v` — section midpoint in atlas space
- `unit_u = u / |u|`, `u_stretch = |u|`
- `unit_v = v / |v|`, `v_stretch = |v|`

Interpolation (linear regression / piecewise linear) is performed in unpacked space, then repacked to the 9-component anchoring format. VERSO's `quicknii_coronal_series_anchorings()` in `engine/registration.py` mirrors this algorithm.

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
    {"x": 0.52, "y": 0.31, "dx": 0.02, "dy": -0.01},
    {"x": 0.18, "y": 0.70, "dx": -0.01, "dy": 0.02}
  ]
}
```

### Marker semantics

Each marker is a control point (warp pin) stored in **normalised section coordinates** `[0, 1]`:

| Field | Meaning |
|---|---|
| `x`, `y` | Source position — where the pin is anchored in atlas space (normalised) |
| `dx`, `dy` | Displacement — `dst - src` in normalised section coordinates |

So the destination position is `(x + dx, y + dy)`.

VERSO stores control points internally as `ControlPoint(src_x, src_y, dst_x, dst_y)` in normalised `[0, 1]`. Conversion at the I/O boundary:

- **Load**: `dst_x = x + dx`, `dst_y = y + dy`
- **Save**: `dx = dst_x - src_x`, `dy = dst_y - src_y`

This matches VisuAlign's internal representation exactly and is resolution-independent.

## DeepSlice output

DeepSlice outputs QuickNII-compatible JSON (same `anchoring` field, same top-level structure). No `markers` field. VERSO can load it directly via the QuickNII reader.

Known differences from QuickNII output:
- May include a `confidence` field per section (ignored on load)
- Section filenames may use full paths rather than base names

## Implementation in VERSO

The compatibility layer lives in `engine/io/quint_io.py`.

### Key functions

```python
def load_quint_json(path: Path) -> list[Section]:
    """Load QuickNII or VisuAlign JSON → list of Section objects."""

def save_quint_json(sections: list[Section], path: Path) -> None:
    """Save sections as VisuAlign-compatible JSON (includes markers if any CPs present)."""
```

Internal helpers:
- `_markers_to_control_points(markers)` — `[{x, y, dx, dy}]` → `[ControlPoint]`
- `_control_points_to_markers(cps)` — `[ControlPoint]` → `[{x, y, dx, dy}]`

### Round-trip guarantee

Loading a QuickNII/VisuAlign JSON and immediately saving it must produce a file that is semantically identical (same anchoring values, same marker values) to within floating-point precision. Write unit tests to verify this.
