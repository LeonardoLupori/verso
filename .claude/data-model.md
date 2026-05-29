# Data Model Reference

Source of truth: `src/verso/engine/model/project.py` and
`src/verso/engine/model/alignment.py`.

## Project folder layout

A user project is a folder containing one JSON file plus generated assets.
The JSON filename defaults to `project-verso.json`
(`DEFAULT_PROJECT_FILENAME` in `engine/model/project.py`).

```
my_experiment/
    project-verso.json     # all project state, settings, metadata
    thumbnails/            # working-resolution OME-TIFFs + filmstrip PNGs
    masks/                 # slice masks as 1-bit PNGs
    lr_masks/              # L/R hemisphere masks as uint8 PNGs (0/1/2)
    exports/               # export outputs (images with overlays, etc.)
```

Original full-resolution images are **not copied**. `project.json` stores
their absolute paths in `section.original_path`. Only working copies
(see "Resolution tiers" below) and filmstrip thumbnails live in the
project folder.

## project.json schema

Top level:

```json
{
  "version": "1.0",
  "name": "My Experiment",
  "atlas": { "name": "allen_mouse_25um", "source": "brainglobe" },
  "channels": [ ... ],
  "cp_size": 10,
  "cp_shape": "Cross",
  "cp_color": "#fff500",
  "sections": [ ... ]
}
```

| Field | Type | Notes |
|---|---|---|
| `version` | str | Schema version. |
| `name` | str | Project display name. |
| `atlas` | `AtlasRef` | `{name, source}`. Source defaults to `"brainglobe"`. |
| `channels` | `list[ChannelSpec]` | Project-wide channel display settings (shared across all sections). |
| `cp_size` / `cp_shape` / `cp_color` | int / str / hex | Warp control-point drawing style, project-wide. |
| `sections` | `list[Section]` | Sections in the cutting series. |

### `ChannelSpec`

```json
{ "name": "DAPI", "color": [0, 0, 255], "scale": 1.0, "visible": true }
```

- `color`: RGB triple in `[0..255]`.
- `scale`: brightness multiplier (driven by the Image →
  Adjust channels/brightness dialog).
- `visible`: per-channel visibility.

### `Section`

```json
{
  "id": "s001",
  "serial_number": 17,
  "original_path": "/data/raw/IMG_0234.tif",
  "thumbnail_path": "thumbnails/IMG_0234-thumb.ome.tif",
  "preprocessing": { ... },
  "alignment": { ... },
  "warp": { ... },
  "scale": 0.2
}
```

| Field | Type | Notes |
|---|---|---|
| `id` | str | Stable identifier (used internally; not user-edited). |
| `serial_number` | int | Section's physical order in the cutting series. Drives ordering in the overview / filmstrip. Parsed from the original filename on import via `parse_section_serial_number`. |
| `original_path` | str | Absolute path to the full-resolution source image. |
| `thumbnail_path` | str | Path to the working-resolution OME-TIFF (relative or absolute). |
| `scale` | float | Ratio `working_long_side / original_long_side`. Default `WORKING_SCALE = 0.2` (`engine/io/image_io.py`). |

### `Preprocessing`

```json
{
  "flip_horizontal": false,
  "flip_vertical": false,
  "slice_mask_path": "masks/IMG_0234-slice-mask.png",
  "lr_mask_path": "lr_masks/IMG_0234_lr.png",
  "lr_line": [[510.0, 80.0], [510.0, 720.0]]
}
```

- Mask PNGs are at working resolution and stored in the **unflipped**
  frame.
- `lr_line`: optional storage-frame endpoints `[[x0, y0], [x1, y1]]`
  used to re-seed the L/R line editor next time.

### `Alignment`

```json
{
  "anchoring": [ox, oy, oz, ux, uy, uz, vx, vy, vz],
  "ap_position_mm": -1.2,
  "status": "complete",
  "source": "manual",
  "stored_anchoring": [...],
  "proposal_anchoring": [...],
  "proposal_confidence": 0.92,
  "proposal_run_id": "20260214-deepslice"
}
```

| Field | Notes |
|---|---|
| `anchoring` | Current 9-float plane (see "Anchoring format" below). Mutated live by the navigator; only `stored_anchoring` is the canonical "saved" plane. |
| `ap_position_mm` | Derived from `anchoring` via the atlas. Refreshed on every navigator move. |
| `status` | `not_started`, `in_progress`, or `complete`. `complete` ⇔ user clicked Save in Align. |
| `source` | Origin of the current plane: `quicknii_default`, `deepslice`, `manual`, or `null`. |
| `stored_anchoring` | The plane the user explicitly saved. Set by `AlignView.save()`. |
| `proposal_anchoring` / `proposal_confidence` / `proposal_run_id` | Last automated proposal (e.g. from DeepSlice) shown alongside the user's edits. |

### `WarpState`

```json
{
  "control_points": [
    {"src_x": 0.52, "src_y": 0.31, "dst_x": 0.54, "dst_y": 0.30}
  ],
  "status": "in_progress"
}
```

`status` follows the same enum as `Alignment.status`.

### Anchoring format

`anchoring` is a 9-element array matching the QuickNII JSON layout:

```
[ox, oy, oz, ux, uy, uz, vx, vy, vz]
```

All values are atlas voxel coordinates. For a normalised section point
`(s, t)` with both in `[0, 1]` (s = horizontal, t = vertical):

```
atlas_voxel = [ox, oy, oz] + s * [ux, uy, uz] + t * [vx, vy, vz]
```

- `(ox, oy, oz)` — atlas voxel at the section's top-left corner.
- `(ux, uy, uz)` — atlas displacement across the full section width.
- `(vx, vy, vz)` — atlas displacement across the full section height.

This is identical to the QuickNII `anchoring` field and is written
straight into QuickNII / VisuAlign JSON without conversion. See
[quint-compat.md](quint-compat.md) for the convention shift between the
QuickNII voxel space and BrainGlobe's coordinate frame.

### Control point format

Control points are stored in **normalised section coordinates** `[0, 1]`
in both source (atlas) and destination (section) spaces. Matches
VisuAlign's internal representation and is resolution-independent.

| Field | Meaning |
|---|---|
| `src_x`, `src_y` | Atlas-space origin of the pin, normalised to `[0, 1]`. |
| `dst_x`, `dst_y` | Section-space destination of the pin, normalised to `[0, 1]`. |

In VisuAlign's export JSON the displacement is `dx = dst_x - src_x`,
`dy = dst_y - src_y` (see [quint-compat.md](quint-compat.md)).

### Status enum (`AlignmentStatus`)

Used for both `Alignment.status` and `WarpState.status`.

| Value | Meaning |
|---|---|
| `not_started` | No work done. |
| `in_progress` | Partially done (e.g. proposal applied, manual edits in flight, control points placed). |
| `complete` | User explicitly saved this step for the slice. |

## Python data model

All model types are `@dataclass`es in `engine/model/`:

| Class | File | Notes |
|---|---|---|
| `Project` | `model/project.py` | Top-level container; `save(path)` / `load(path)` round-trip JSON. |
| `AtlasRef` | `model/project.py` | `{name, source}`. |
| `ChannelSpec` | `model/project.py` | Per-project channel display config. |
| `Section` | `model/project.py` | One histological section. |
| `Preprocessing` | `model/project.py` | Flips + mask paths + `lr_line`. |
| `Alignment` | `model/alignment.py` | 9-float anchoring + status + proposal/stored variants. |
| `WarpState` | `model/alignment.py` | List of `ControlPoint`s + status. |
| `ControlPoint` | `model/alignment.py` | `(src_x, src_y, dst_x, dst_y)` normalised. |

Each class implements `to_dict()` / `from_dict()` for JSON round-trip.
`Project.save(path)` writes formatted JSON; `Project.load(path)`
reconstructs. The `tests/engine/test_model.py` suite covers the
round-trip.

## Resolution tiers

Three tiers exist; transforms must be explicit at the boundaries.

| Tier | Where | Notes |
|---|---|---|
| Full resolution | Original on disk (e.g. 20000 × 15000). | Loaded only for export. |
| Working resolution | `WORKING_SCALE × full` (default 0.2). | All interactive operations — masks, anchoring, control points — happen here. Cached as a multichannel OME-TIFF in `thumbnails/<stem>-thumb.ome.tif`. |
| Filmstrip thumbnail | Long side ≤ `FILMSTRIP_MAX_SIDE` (150 px). | RGB composite generated on demand. |

Scaling factor `Section.scale = working_long_side / original_long_side`
is stored per section so full-resolution export can scale back up.
Control points in normalised `[0, 1]` need no rescaling.

### Transform chain

```
full resolution ──→ working resolution ──→ normalised [0, 1] ──→ atlas voxel space
        (scale)                (pixel / dimension)              (anchoring)
```

## Section ordering

`Section.serial_number` is the **only** ordering signal. The overview
table and filmstrip both sort by `(serial_number, list-position)`.
Drag-to-reorder is not implemented; reorder by editing the serial
number column in the overview.

`serial_number` is:
- An **int**, mutable.
- **Not guaranteed unique** in principle (two images of the same
  physical slice could share a number), but VERSO does not currently
  treat duplicates specially in any pipeline.
- Parsed from filenames on import by `parse_section_serial_number`
  (`engine/io/image_io.py`) — first integer run after the leading
  underscore, falling back to the first integer anywhere, falling back
  to the import index.

## Alignment interpolation

When some sections are aligned and others are not, VERSO fills the
unaligned ones with linearly-interpolated proposals so the user starts
each section near the right pose. Implementation:
`engine/registration.py` (`quicknii_coronal_series_anchorings`,
`interpolate_anchorings`); matches QuickNII's `MgmtPanel.dointerpolate`.

The 9-float anchoring is unpacked into 11 components (midpoint xyz,
unit u-vector xyz, unit v-vector xyz, u-stretch, v-stretch) for
component-wise interpolation between the nearest stored neighbours
sorted by `serial_number`. After interpolation, in-plane rotation is
stripped and the two non-AP axes are reset to the atlas midpoint —
those vary erratically and are quicker to fix by hand than to clean up
from a bad interpolation.

The Batch → Align → *Reverse proposal* action flips the AP direction
of the proposal series (used when the user imported their sections
back-to-front); it is only available before any alignment has been
saved.

## Persistence rules

- The project file is written only when the user explicitly saves a
  view's draft, runs Save / Save As, or triggers a batch / export
  action.
- Mask PNGs are written only on `PrepView.save()` (or deleted on
  `PrepView.clear()` / batch wipe).
- Control points, anchorings, statuses live entirely in `Section` and
  are persisted only when `Project.save(path)` runs.

See [ui-design.md](ui-design.md) for the SaveBar / Clear / discard flow
and the confirmation prompts on destructive transitions.
