# Data Model Reference

## Project folder structure

Each user project is a folder (not a single file):

```
my_experiment/
    project.json      # all project state, settings, metadata
    thumbnails/       # auto-generated 1200px working copies
    masks/            # slice masks, L/R masks as PNGs
    alignments/       # JSON files compatible with QuickNII/VisuAlign
    exports/          # final warped images, CSVs, point clouds
```

### Why a folder, not a single file

- Researchers can inspect and manually edit JSON
- Other tools (PyNutil, custom scripts) read directly from the folder
- Git-friendly (JSON + images)
- Crash-resilient (individual file saves, no database corruption)

### Image handling

Original high-resolution images are **not copied** into the project folder. `project.json` stores the path to originals. Only thumbnails (1200px) are generated and stored. If originals move, the tool works with thumbnails and prompts the user to relocate originals only when exporting full-resolution results.

## project.json schema

```json
{
  "version": "1.0",
  "name": "My Experiment",
  "atlas": {
    "name": "allen_mouse_25um",
    "source": "brainglobe"
  },
  "interpolation_axis": "AP",
  "sections": [
    {
      "id": "s001",
      "slice_index": 17,
      "original_path": "/data/raw/IMG_0234.tif",
      "thumbnail_path": "thumbnails/s001.png",
      "channels": ["DAPI", "GFP", "autofluorescence"],
      "registration_channel": "autofluorescence",
      "preprocessing": {
        "flip_horizontal": true,
        "slice_mask_path": "masks/s001_slice.png",
        "lr_mask_path": "masks/s001_lr.png"
      },
      "alignment": {
        "ap_position_mm": -1.2,
        "anchoring": [ox, oy, oz, ux, uy, uz, vx, vy, vz],
        "status": "complete"
      },
      "warp": {
        "control_points": [
          {"src_x": 0.52, "src_y": 0.31, "dst_x": 0.54, "dst_y": 0.30},
          {"src_x": 0.18, "src_y": 0.70, "dst_x": 0.17, "dst_y": 0.72}
        ],
        "status": "complete"
      }
    }
  ]
}
```

### Anchoring format

`anchoring` is a 9-element flat array matching the QuickNII JSON format:

```
[ox, oy, oz,  ux, uy, uz,  vx, vy, vz]
```

All values are in **atlas voxel coordinates**. For a normalised section point `(s, t)` where both are in `[0, 1]` (s = horizontal, t = vertical), the corresponding atlas voxel is:

```
atlas_voxel = [ox, oy, oz] + s * [ux, uy, uz] + t * [vx, vy, vz]
```

- `(ox, oy, oz)` — origin: atlas voxel at the section's top-left corner
- `(ux, uy, uz)` — u-vector: atlas displacement across the full section width
- `(vx, vy, vz)` — v-vector: atlas displacement across the full section height

This is identical to the QuickNII `anchoring` field and can be written directly into QuickNII/VisuAlign JSON without conversion.

### Control point format

Control points are stored in **normalised section coordinates** `[0, 1]` in both source (atlas) and destination (section) spaces. This matches VisuAlign's internal representation and is resolution-independent.

| Field | Meaning |
|---|---|
| `src_x`, `src_y` | Atlas-space origin of the pin, normalised to `[0, 1]` |
| `dst_x`, `dst_y` | Section-space destination of the pin, normalised to `[0, 1]` |

The displacement in VisuAlign's export JSON is `dx = dst_x - src_x`, `dy = dst_y - src_y` (see [quint-compat.md](quint-compat.md)).

## Python data model

Use `@dataclass` for all model types. These live in `engine/model/`.

### Core types

- `Project` — top-level container: name, atlas reference, list of sections
- `Section` — one histological section: paths, channel info, preprocessing state, alignment, warp
- `Alignment` — affine registration: AP position, 9-float anchoring array, status
- `ControlPoint` — one warp pin: `src_x, src_y, dst_x, dst_y` in normalised `[0, 1]`
- `WarpState` — nonlinear refinement: list of `ControlPoint`, status
- `Mask` — mask metadata: path, type (slice or L/R)

### Serialization

All model types must round-trip cleanly through JSON:
- `project.to_json()` → write to `project.json`
- `Project.from_json(path)` → reconstruct full project state
- Write unit tests that create a project, save, reload, and verify equality

### Status enum

Each pipeline step has three states:
- `not_started` — empty (no work done)
- `in_progress` — partially done (e.g., some control points placed)
- `complete` — finished

## Coordinate spaces

Three coordinate spaces exist. The transform chain must be explicitly defined.

### 1. Full resolution space

The original image pixel coordinates. Example: 20000×15000 pixels.

### 2. Working resolution space

Thumbnail at 1200px on the long side. All interactive operations (control points, masks, registration) happen here.

**Scaling factor**: `scale = working_width / original_width`, stored per section.

### 3. Atlas space

3D voxel coordinates of the reference atlas volume (e.g., Allen Mouse 25µm → 528×320×456 voxels). The anchoring matrix maps between normalised section coordinates and atlas voxel space.

### Transform chain

```
full resolution ←→ working resolution ←→ normalised [0,1] ←→ atlas voxel space
         (scale factor)          (pixel / dimension)       (anchoring matrix)
```

When exporting full-resolution results, working-resolution transforms are scaled up by `1/scale`. Control points in normalised space need no rescaling.

## Slice index semantics

`slice_index` is the physical position of a section in the cutting series. It is the **only** ordering signal — the overview table and filmstrip are always sorted by `slice_index`. There is no separate "display order" field and no manual drag-to-reorder gesture; reordering happens by editing `slice_index`.

### Properties

- **Integer**, mutable, editable from the overview table.
- **Not unique**: multiple sections may share the same `slice_index` when one physical slice was photographed as several images (split slices, e.g. left + right hemispheres or three fragments of a torn slice).
- **Not contiguous**: gaps are meaningful and must be preserved. A series like `[1, 15, 16, 17, 17, 17, 80]` is valid — it encodes that slices 2–14 and 18–79 were discarded or never imaged, and that slice 17 was split into three fragments. Interpolation relies on these gaps to estimate physical AP spacing.
- **Never auto-renumbered.** Drag-reorder is disallowed precisely because it cannot be reconciled with these properties without destroying information. The only mutation is direct edit.

### Interpolation axis

The project-level `interpolation_axis` field (`"AP"` | `"ML"` | `"DV"`, default `"AP"`) declares which brain axis `slice_index` runs along. Almost always AP (coronal series); rarely ML for sagittal series.

### Filename-based seeding

On import, VERSO attempts to infer `slice_index` from filenames so the user rarely has to type numbers:

1. Extract all integer runs from each filename (`brain01_slice_017_DAPI_ch2.tif` → `[1, 17, 2]`).
2. Pick the integer column whose value **varies most across the imported batch** — this is robust to arbitrary naming schemes and beats regex heuristics.
3. Tiebreakers: widest range → consistent zero-padding → rightmost.
4. Strip split-slice suffixes before parsing: trailing `[a-z]`, `_L` / `_R`, `-\d` (`017a.tif`, `017_L.tif`, `017-1.tif` all map to `slice_index = 17`).
5. Files that fail to parse get `slice_index = null` and float to the top of the table with a warning chip — never silently assigned a guess.

The import dialog must show the inferred mapping as a preview before committing, with a one-click override to pick a different integer column when the heuristic guesses wrong.

## Alignment interpolation

When some sections in the series are aligned and others are not, VERSO fills the unaligned ones with linearly interpolated proposals so the user starts each section from a near-correct pose. This matches QuickNII's `MgmtPanel.dointerpolate` semantics and is implemented in `engine/registration.py` (`quicknii_coronal_series_anchorings`, `interpolate_anchorings`).

### What is interpolated

The 9-float anchoring is unpacked into 11 components (origin xyz, u-unit-vector xyz, v-unit-vector xyz, u-scale, v-scale) for interpolation. Component-wise linear interpolation between the nearest stored neighbors (by `slice_index`) determines:

| Component | Interpolated? |
|---|---|
| Position along `interpolation_axis` (typically AP) | ✅ yes |
| Physical tilt of the cutting plane (u_y, v_y) | ✅ yes |
| Stretch / scale (u-scale, v-scale) | ✅ yes |
| In-plane rotation (rotation around the interpolation axis) | ❌ stripped after interpolation |
| Position on the two non-interpolation axes (e.g. LR/DV for AP series) | ❌ reset to atlas midpoint |

Rationale: the cutting plane (where + how tilted) is what changes smoothly between adjacent slices and is worth interpolating. In-plane rotation and lateral framing vary erratically and are quicker to fix by hand than to clean up from a bad interpolation.

### Split slices

Sections sharing a `slice_index` represent **one physical cutting plane** photographed as multiple images. They are treated as a single interpolation control point:

- For interpolation purposes, the plane parameters (position-along-axis, tilt, scale) of fragments with the same `slice_index` are combined by **median** to form one virtual control point. Median is preferred over mean to be robust to a fragment the user aligned carelessly.
- Disagreement between fragments is **not surfaced as a warning** — the median absorbs it silently.
- When a fragment inherits a proposed anchoring from an aligned sibling, it copies the **plane** (AP + tilt + scale) but recenters lateral framing to the atlas midpoint, rather than overlapping the sibling's framing. The user then drags the framing into place for the correct fragment of tissue.

### Effect on neighbors

For an unaligned section with `slice_index = N`, the proposal is built from the nearest aligned `slice_index < N` and `slice_index > N` (or extrapolated from a regression on all stored anchors if only one side exists, matching QuickNII's fallback). Gaps in `slice_index` are respected: the lerp parameter `t` uses raw `slice_index` differences so spacing is proportional to physical distance, not row count.

## Settled decisions

1. **Anchoring format**: 9-float array `[ox, oy, oz, ux, uy, uz, vx, vy, vz]` in atlas voxel space — identical to QuickNII JSON.
2. **Control point format**: `(src_x, src_y, dst_x, dst_y)` in normalised `[0, 1]` — matches VisuAlign's internal representation; resolution-independent.
3. **Undo/redo**: deferred; not in scope for initial implementation.
4. **Mask format**: binary PNG at working resolution (1200px); path stored in `project.json`.
5. **Ordering**: `slice_index` is the sole ordering signal. No manual drag-to-reorder; reorder by editing the index. Duplicates and gaps are legal and meaningful.
6. **Split-slice interpolation**: fragments sharing a `slice_index` are combined by median of their plane parameters when acting as an interpolation control point. No UI warning on disagreement.
