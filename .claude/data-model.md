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
    annotations/           # point series (points.csv) + area masks, one folder each
    exports/               # export outputs (images with overlays, aligned stacks, etc.)
        quantification_<YYYYMMDD-HHMMSS>/   # one folder per Export ▸ Quantify run
            intensity.csv / area.csv / dots.csv / dots_regions.csv   # pooled tables
            <image-slug>/  # only when "Separate output per slice" is on: same files per section
```

Quantification CSVs are written by `engine/io/quant_export.py`; each run gets its
own timestamped folder. Pooled runs write the tables at the top level; per-slice
runs write one `<slugified-image-stem>/` subfolder per section instead. Aggregated
levels add `*_mid.csv` / `*_coarse.csv` companions. See
[quantification.md](quantification.md).

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
  "atlas": {
    "name": "allen_mouse_25um",
    "source": "brainglobe",
    "resolution_um": 25.0,
    "shape": [528, 320, 456]
  },
  "interpolation_axis": "AP",
  "channels": [ ... ],
  "cp_size": 10,
  "cp_shape": "Cross",
  "cp_color": "#fff500",
  "working_scale": 0.2,
  "sections": [ ... ]
}
```

| Field | Type | Notes |
|---|---|---|
| `version` | str | Schema version the file was written under; currently `"1.0"`. Informational only — see "Schema versioning" below. **Early development: no migration / backward-compatibility support.** |
| `name` | str | Project display name. |
| `atlas` | `AtlasRef` | `{name, source, resolution_um, shape}`. `source` defaults to `"brainglobe"`. `resolution_um` is the isotropic atlas voxel size (microns); `shape` is the atlas voxel grid `[AP, DV, LR]` in BrainGlobe order. Both are cached so the project file is self-contained for pixel ↔ atlas voxel mapping without re-fetching the atlas; `0.0` / `[0, 0, 0]` until populated. |
| `interpolation_axis` | str | Brain axis the cutting series runs along: `"AP"` (coronal, default), `"ML"` (sagittal), or `"DV"` (horizontal). Set at project creation; drives the anchoring voxel axis used by `propagate_series_anchorings`. See "Interpolation axis" below. |
| `channels` | `list[ChannelSpec]` | Project-wide channel display settings (shared across all sections). |
| `cp_size` / `cp_shape` / `cp_color` | int / str / hex | Warp control-point drawing style, project-wide. |
| `working_scale` | float | Ratio `working_long_side / original_long_side`, **uniform across all sections**. Derived once at import from the largest image so its longest side fits within `THUMBNAIL_MAX_SIDE` (2000 px); see `compute_working_scale` in `engine/io/image_io.py`. Full-resolution export scales back up by this factor. Default `0.2`. |
| `sections` | `list[Section]` | Sections in the cutting series. |

### Schema versioning

VERSO is in **early development: there is no schema migration or backward-compatibility
support.** The `version` field is stamped on every saved file (currently always `"1.0"`)
but nothing compares or upgrades it — `Project.load` simply calls `Project.from_dict`, and
older or foreign project files are not guaranteed to load. When the schema changes, expect
to recreate projects rather than migrate them. The field is kept so that, once versioning
does matter, a missing `version` unambiguously means "pre-1.0".

Derived metadata (per-section pixel dimensions, atlas `resolution_um`/`shape`) is populated
once at project creation by `populate_metadata` (`engine/io/project_metadata.py`), which
reads the image files and the `AtlasVolume`, so the saved file is self-contained. Atlas
download requires internet the first time an atlas is used; creation fails loudly otherwise.

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
  "slice_index": 17,
  "original_path": "/data/raw/IMG_0234.tif",
  "thumbnail_path": "thumbnails/IMG_0234-thumb.ome.tif",
  "resolution_original_wh": [20000, 15000],
  "resolution_thumbnail_wh": [1200, 900],
  "preprocessing": { ... },
  "alignment": { ... },
  "warp": { ... }
}
```

| Field | Type | Notes |
|---|---|---|
| `id` | str | Stable identifier (used internally; not user-edited). Breaks ties when two sections share a `slice_index`. |
| `slice_index` | int | Section's physical position along the project's interpolation axis (e.g. AP). Ground truth for ordering everywhere (overview / filmstrip / interpolation). **Need not be contiguous** (1, 2, 18, 19 encodes a gap) and **may repeat** (a slice that broke into several images shares one index). Guessed from filenames on import via `guess_slice_indices` and editable afterwards in the overview `#` column. |
| `original_path` | str | Absolute path to the full-resolution source image. |
| `thumbnail_path` | str | Path to the working-resolution OME-TIFF (relative or absolute). |
| `resolution_original_wh` | `[int, int]` | Pixel dimensions `[width, height]` of the original (full-resolution) image. Cached so the file alone maps pixels ↔ atlas voxels; `[0, 0]` until populated (added in v1.2). |
| `resolution_thumbnail_wh` | `[int, int]` | Pixel dimensions `[width, height]` of the working-resolution thumbnail. `[0, 0]` until populated (added in v1.2). |

### `Preprocessing`

```json
{
  "flip_horizontal": false,
  "flip_vertical": false,
  "slice_mask_path": "masks/IMG_0234-slice-mask.png"
}
```

- Mask PNGs are at working resolution and stored in the **unflipped**
  frame.

### `Alignment`

```json
{
  "anchoring": [ox, oy, oz, ux, uy, uz, vx, vy, vz],
  "status": "complete",
  "source": "manual"
}
```

The **saved plane is the only plane persisted**, under the key `"anchoring"`.
It is emitted only when a saved plane exists (i.e. `stored_anchoring` is set —
`complete` sections). In memory the `Alignment` dataclass holds two planes:

| Field (in memory) | Persisted? | Notes |
|---|---|---|
| `stored_anchoring` | **yes**, as JSON `"anchoring"` | The plane the user explicitly saved (set on commit by `AlignView.save()`). The single source of truth; the only value written to disk, and what export/interpolation read. |
| `current_anchoring` | no | The live working copy, mutated by the navigator during editing. Seeded from `stored_anchoring` on load; transient. |
| `position_mm` | no | Section position in mm along the project's `interpolation_axis`. A derived display cache, recomputed from the plane via the atlas on load; never persisted. |
| `status` | yes | `not_started`, `in_progress`, or `complete`. `complete` ⇔ user clicked Save in Align. |
| `source` | yes (when set) | Origin of the current plane: `series_interpolation`, `deepslice`, `manual`, or `null`. |

`in_progress` proposal planes are **not** persisted — they are regenerated on
load by `interpolate_anchorings`/`_initialize_default_anchorings` from the saved
keyframes (GUI), so a section without a saved plane carries no `"anchoring"` on
disk.

### `WarpState`

```json
{
  "control_points": [
    {"src_x": 120.0, "src_y": 84.0, "dst_x": 128.0, "dst_y": 80.0}
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

Control points (`ControlPoint` in `engine/model/alignment.py`) are stored in
**working-resolution pixel coordinates**, in both source (atlas overlay) and
destination (section image) spaces — *not* normalised. `warping.py`'s warp
functions divide by the section's working width/height to normalise
internally; `to_dict`/`from_dict` pass the pixel values straight through.

| Field | Meaning |
|---|---|
| `src_x`, `src_y` | Atlas-overlay pixel position of the pin (working resolution) — fixed when the point is created. |
| `dst_x`, `dst_y` | Section-image pixel position of the pin (working resolution) — updated as the user drags it. |

This differs from VisuAlign's own JSON export, which stores control points as
normalised `[0, 1]` `markers` (see [quint-compat.md](quint-compat.md)) — the
conversion between the two happens at the QuickNII/VisuAlign I/O boundary in
`engine/io/quint_io.py`, not in the native `project-verso.json` format.

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
| `AtlasRef` | `model/project.py` | `{name, source, resolution_um, shape}`. |
| `ChannelSpec` | `model/project.py` | Per-project channel display config. |
| `Section` | `model/project.py` | One histological section. |
| `Preprocessing` | `model/project.py` | Flips + slice-mask path. |
| `Alignment` | `model/alignment.py` | Saved + live 9-float planes (`stored_anchoring` / `current_anchoring`) + status. Only the saved plane persists. |
| `WarpState` | `model/alignment.py` | List of `ControlPoint`s + status. |
| `ControlPoint` | `model/alignment.py` | `(src_x, src_y, dst_x, dst_y)` in working-resolution pixels. |

Each class implements `to_dict()` / `from_dict()` for JSON round-trip.
`Project.save(path)` writes formatted JSON; `Project.load(path)`
reconstructs. The `tests/engine/test_model.py` suite covers the
round-trip.

## Resolution tiers

Three tiers exist; transforms must be explicit at the boundaries.

| Tier | Where | Notes |
|---|---|---|
| Full resolution | Original on disk (e.g. 20000 × 15000). | Loaded only for export. |
| Working resolution | `Project.working_scale × full`. | All interactive operations — masks, anchoring, control points — happen here. Cached as a multichannel OME-TIFF in `thumbnails/<stem>-thumb.ome.tif`. |
| Filmstrip thumbnail | Long side ≤ `FILMSTRIP_MAX_SIDE` (150 px). | RGB composite generated on demand. |

Scaling factor `Project.working_scale = working_long_side / original_long_side`
is uniform across all sections (derived at import from the largest image, so
its longest side fits within `THUMBNAIL_MAX_SIDE` = 2000 px) so full-resolution
export can scale back up. Control points in normalised `[0, 1]` need no rescaling.

### Transform chain

```
full resolution ──→ working resolution ──→ normalised [0, 1] ──→ atlas voxel space
        (scale)                (pixel / dimension)              (anchoring)
```

## Section ordering

`Section.slice_index` is the **only** ordering signal. `Project.sections`
is kept **canonically sorted by `(slice_index, id)`** — `Project.sort_sections()`
re-sorts in place and is called on load (`from_dict`), after import, and after
a manual index edit. The overview table, filmstrip, and navigation all just
iterate the list, so they follow increasing `slice_index` with `id` (import
order) breaking ties for duplicates.

`slice_index` is:
- An **int**, mutable.
- **Need not be contiguous** — `1, 2, 18, 19` encodes two adjacent slices, a
  gap, then two more.
- **Allowed to repeat** — a physical slice that broke into several images
  shares one index. Interpolation collapses equal indices to the same position
  (the `denom == 0 → t = 0` guard in `propagate_series_anchorings`); no separate
  `replicate` field exists.
- Guessed from filenames on import by `guess_slice_indices`
  (`engine/io/image_io.py`): every filename stem is tokenised into its numeric
  runs and the **token position with the widest range / most distinct values /
  most monotonic order** is chosen, falling back to `1..N` by natural-sorted
  name when no fully-covered numeric field exists. The New Project dialog shows
  the guesses in an editable `File | Slice index` preview table.
- Editable after import via the overview `#` column (double-click; rows re-sort
  on commit). Drag-to-reorder is intentionally not implemented.

## Interpolation axis

The project-level `interpolation_axis` field declares which atlas axis
the cutting series runs along. It is one of:

| Value | Slicing orientation (UI label) | Anchoring voxel axis |
|---|---|---|
| `"AP"` (default) | Coronal | 1 |
| `"ML"` | Sagittal | 0 |
| `"DV"` | Horizontal | 2 |

The user picks the orientation at project creation (the New Project
dialog shows the friendly slicing-orientation labels; the JSON stores
the axis name). Currently read-only after creation. Defaults to
`"AP"` for back-compat with v1.0 project files.

`Project.interpolation_axis_index` returns the matching integer for
engine math. Mappings live in `engine/model/project.py` as
`AXIS_NAME_TO_INDEX` and `SLICING_ORIENTATION_TO_AXIS`.

DeepSlice is coronal-only and is disabled in the UI when
`interpolation_axis != "AP"`.

## Alignment interpolation

When some sections are aligned and others are not, VERSO fills the
unaligned ones with linearly-interpolated proposals so the user starts
each section near the right pose. Implementation:
`engine/anchoring.py` (`propagate_series_anchorings`,
`interpolate_anchorings`); ported from QuickNII's `MgmtPanel.dointerpolate`
algorithm (see [quint-compat.md](quint-compat.md)).

The 9-float anchoring is unpacked into 11 components (midpoint xyz,
unit u-vector xyz, unit v-vector xyz, u-stretch, v-stretch) for
component-wise interpolation between the nearest stored neighbours
sorted by `slice_index`. After interpolation:

| Component | Interpolated? |
|---|---|
| Position along the slicing axis (`interpolation_axis`) | ✅ yes |
| Plane tilt (slicing-axis component of each unit vector) | ✅ yes |
| Stretch / scale | ✅ yes |
| In-plane rotation (rotation around the slicing axis) | ❌ stripped after interpolation |
| Position on the two in-plane axes | ❌ reset to atlas midpoint |

The two non-slicing axes vary erratically and are quicker to fix by
hand than to clean up from a bad interpolation.

The Batch → Align → *Reverse proposal* action flips the direction of
the proposal series along the slicing axis (used when the user
imported their sections back-to-front); it is only available before
any alignment has been saved.

### Future spec (not yet implemented)

Filename-based variance-driven index seeding (`guess_slice_indices`) and
physical-distance-proportional lerp using raw `slice_index` differences are now
implemented. Split-slice handling that *combines* sections sharing a
`slice_index` (e.g. by median anchoring during interpolation) is still on the
roadmap — today duplicate indices simply collapse to the same interpolated
position rather than being merged.

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
