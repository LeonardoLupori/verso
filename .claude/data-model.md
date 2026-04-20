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
  "sections": [
    {
      "id": "s001",
      "serial_number": 1,
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

## Settled decisions

1. **Anchoring format**: 9-float array `[ox, oy, oz, ux, uy, uz, vx, vy, vz]` in atlas voxel space — identical to QuickNII JSON.
2. **Control point format**: `(src_x, src_y, dst_x, dst_y)` in normalised `[0, 1]` — matches VisuAlign's internal representation; resolution-independent.
3. **Undo/redo**: deferred; not in scope for initial implementation.
4. **Mask format**: binary PNG at working resolution (1200px); path stored in `project.json`.
