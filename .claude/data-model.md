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
        "anchoring": ["QuickNII-compatible anchoring matrix"],
        "status": "complete"
      },
      "warp": {
        "control_points": [
          {"x": 120, "y": 85, "dx": 3.2, "dy": -1.5},
          {"x": 340, "y": 200, "dx": -2.1, "dy": 4.0}
        ],
        "status": "complete"
      }
    }
  ]
}
```

## Python data model

Use `@dataclass` for all model types. These live in `engine/model/`.

### Core types (to be finalized)

- `Project` — top-level container: name, atlas reference, list of sections
- `Section` — one histological section: paths, channel info, preprocessing state, alignment, warp
- `Alignment` — affine registration: AP position, anchoring matrix, status
- `WarpState` — nonlinear refinement: list of control points, status
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

3D voxel coordinates of the reference atlas volume (e.g., Allen Mouse 25µm → 528×320×456 voxels). The alignment anchoring matrix maps between working resolution space and atlas space.

### Transform chain

```
full resolution ←→ working resolution ←→ atlas space
         (scale factor)         (anchoring matrix)
```

When exporting full-resolution results, working-resolution transforms are scaled up by `1/scale`. The scaling is purely proportional (uniform in x and y).

## Open decisions

These must be finalized before implementing the model layer:

1. **Exact Section fields**: required vs optional, defaults for new sections
2. **Alignment anchoring format**: must match QuickNII JSON — study their spec first
3. **Control point format**: `(x, y, dx, dy)` displacements or `(src_x, src_y, dst_x, dst_y)` pairs?
4. **Undo/redo**: command pattern with per-section undo stacks. What granularity? One undo per control point move? Per mask stroke?
5. **Mask format**: binary PNG (resolution-dependent) vs polygon vertices in JSON (resolution-independent)