# Analysis & Cells Views — Proposal

> **Status: proposal / not implemented.** This document specifies two new
> toolbar views that add the quantification step VERSO is missing — the role
> PyNutil plays at the end of the QuickNII → VisuAlign → PyNutil pipeline.
> No code from this document has been written yet.

---

## 1. Goal

After registration (Prep / Align / Warp), the user needs to *quantify* the
sections against the atlas. Four capabilities:

1. **Per-region average pixel intensity** (mean by pixel, per channel).
2. **Load external cell counts** (optional).
3. **Built-in CV bright-spot detection** for cfos-like puncta.
4. **Export results** as modular files (area / intensity / cell numbers /
   density). Density requires a user-specified pixel size.

The engine already maps a section into atlas space and samples region labels
with VisuAlign/PyNutil-parity voxel selection
([`atlas.py`](../src/verso/engine/atlas.py): `AtlasVolume.sample_labels`) and
applies the Delaunay warp ([`warping.py`](../src/verso/engine/warping.py)). The
quantification entry points exist only as stubs in
[`quantification.py`](../src/verso/engine/quantification.py)
(`quantify_points`, `quantify_area` raise `NotImplementedError`). This proposal
fills that gap.

---

## 2. Design decisions

- **Two toolbar views**: a per-section **Cells** view (optional) and a
  project-wide **Analysis** view. Intensity analysis must work with no cells.
- **Cell import formats**: generic CSV (`X,Y[,channel]`) and VisuAlign/PyNutil
  JSON points.
- **Detection parameters are global** (stored on the project, constant across
  sections). A **Detect** button runs on the *current* section only; results
  follow the same **draft → Save / Clear / Revert** semantics as alignment.
  If the user **imported** counts for a section, detection is disabled for it
  (imported source wins). A **batch "detect all slices"** action lives in the
  Batch menu. Counts live in a dedicated `counts/` project folder.
- **Density / area**: region area measured in **section pixels**, converted to
  mm² via a user-specified original pixel size (µm/px) ÷ `working_scale`.
  Density = object_count / region_mm².

---

## 3. Detection algorithm — Laplacian-of-Gaussian + Gaussian correlation

A registration + puncta-quantification approach in VERSO's exact lineage.
Classical (non-deep-learning), 2D per-section, scikit-image (already a
dependency):

1. **Candidates** — Laplacian-of-Gaussian blob detection tuned to the expected
   puncta size: local maxima of the LoG scale space via
   `skimage.feature.blob_log(img, min_sigma, max_sigma, num_sigma, threshold)`,
   returning `(y, x, sigma)` per candidate.
2. **Selection** — gate each candidate by **correlation to an ideal 2D Gaussian**
   of the detected sigma: extract the patch around `(y, x)`, normalized-cross-
   correlate against a Gaussian template, keep candidates ≥ `correlation_min`.
   This rejects edges / streaks / noise that survive a plain threshold, which a
   threshold + connected-components counter would miscount.

`DetectionParams` (all on `Project`, constant across sections): `channel`,
`min_sigma`, `max_sigma`, `num_sigma`, `log_threshold`, `correlation_min`,
optional `intensity_min`. Optional white top-hat background subtraction and a
min-distance NMS are available as secondary pre/post steps.

---

## 4. Engine — new `engine/analysis/` package

Pure, headless, GUI-free. Modular files, one concern each, re-exported from
[`engine/__init__.py`](../src/verso/engine/__init__.py).

| File | Public function(s) | Responsibility |
|---|---|---|
| `analysis/region_map.py` | `section_region_labels(section, atlas, project) -> (labels HxW int32, in_bounds bool)` | **Shared backbone.** Per-pixel atlas region-ID map in the section working-resolution frame: `atlas.sample_labels(anchoring, w, h)`, then (if `section.warp.control_points`) apply the backward remap (`warping.build_backward_remap` + `cv2.remap(..., INTER_NEAREST)`); apply the section's flips so labels share the working-image frame. |
| `analysis/intensity.py` | `region_intensity(pixels, labels, in_bounds, *, channels) -> dict[int, RegionIntensity]` | Per-region **mean intensity per channel** + contributing pixel count (`np.bincount` / `np.add.at` over labels). |
| `analysis/area.py` | `region_area(labels, in_bounds, *, restrict_mask=None) -> dict[int, int]` | Per-region pixel count (optionally restricted to slice / L-R mask). Replaces the `quantify_area` stub. |
| `analysis/log_detect.py` | `detect_log_puncta(channel_img, params) -> (points_xy, props)` | Core LoG + Gaussian-correlation detector (section 3). |
| `analysis/objects.py` | `DetectionParams`; `detect_blobs(channel_img, params) -> ObjectSet`; `assign_objects_to_regions(points_xy, labels, in_bounds) -> (counts: dict[int,int], per_object_label)` | Detection wrapper + region assignment. Replaces the `quantify_points` stub. |
| `analysis/density.py` | `region_density(counts, areas_px, pixel_size_um, working_scale) -> dict[int, float]` | Combine counts + pixel area + pixel size → objects/mm². |
| `analysis/report.py` | `build_region_table(section_or_project, atlas, …) -> list[RegionRow]` | Merge area / intensity / counts / density per region; attach acronym + name from `atlas._bg.structures`; per-section and pooled rows. Stdlib only (no pandas). |

**Gotcha — intensity must use raw pixels.** `ensure_working_copy` returns a
per-image percentile-stretched uint8 array (display-oriented), which would
corrupt intensity means. Add `load_working_pixels(section, working_scale, *,
raw=True)` to [`image_io.py`](../src/verso/engine/io/image_io.py) returning the
linearly-downsampled original (no stretch, dtype preserved). `region_intensity`
and `detect_blobs` consume the **raw** array; the GUI display path keeps the
stretched one.

**Gotcha — coordinate frames.** `section_region_labels` is the single place
flips + warp are reconciled, so labels, area, and detected points all live in
the same working-res frame. Imported cells are converted into that frame before
assignment.

### New `engine/io/` modules

- `io/cell_io.py` — `load_cells_csv(path, *, units, working_scale, section)`
  (generic `X,Y[,channel]`; `units` selects original-px vs working-px, scaled +
  flipped into the working frame), `load_cells_quint_json(path, …)`, and
  `save_cells_csv(object_set, path)` for the `counts/` sidecar.
- `io/analysis_export.py` — **modular** writers, one CSV each:
  `write_area_report`, `write_intensity_report`, `write_objects_report`,
  `write_density_report`, plus `write_object_coordinates` (atlas-space points,
  PyNutil-style). Top-level `export_analysis(project, atlas, out_dir, options)`
  runs the pipeline over all sections and writes per-section + pooled files,
  mirroring `export_section` in
  [`export_images.py`](../src/verso/engine/io/export_images.py). Stdlib `csv`.

The old [`quantification.py`](../src/verso/engine/quantification.py) stubs are
superseded — re-export the new implementations there (or delete + fix imports).

---

## 5. Data model changes

In [`model/project.py`](../src/verso/engine/model/project.py) /
[`model/alignment.py`](../src/verso/engine/model/alignment.py):

- **`Section.counts: CountsState`** — `{ points_path: str | None, source:
  "detected" | "imported" | None, status: AlignmentStatus }`. Points stored as a
  CSV sidecar under `<project>/counts/<stem>.csv` (path-by-reference, like
  masks); the dataclass records source + status. Draft semantics identical to
  warp.
- **`Project.detection_params: DetectionParams`** — global, constant across
  sections.
- **`Project.pixel_size_um: float | None`** — original-resolution pixel size for
  density (working-res pixel size = `pixel_size_um / working_scale`).
- Extend `to_dict` / `from_dict`; add `counts/` to the project folder layout in
  [`data-model.md`](data-model.md).
- Extend the status model
  ([`model/status.py`](../src/verso/engine/model/status.py)
  `section_step_color`) with a `"cells"` step for filmstrip / overview dots.

Project folder layout gains:

```
my_experiment/
    counts/              # per-section object points (CSV sidecars)
```

---

## 6. GUI

### 6.1 Cells view — per-section, optional (mirrors Warp)

- `gui/views/cells_view.py` — `CellsView`. Reuses the shared
  `SectionCanvasPanel` + `Filmstrip` (like Align/Warp). Detected/imported points
  drawn as a `pyqtgraph.ScatterPlotItem` overlay above the atlas overlay.
  Standard lifecycle copied from
  [`warp_view.py`](../src/verso/gui/views/warp_view.py) (`save` / `revert` /
  `clear` ~lines 441-486, baseline deep-copy + `_state.pop_baseline(section.id,
  "cells")`, `dirty_changed` signal, `activate` / `deactivate`).
- `gui/widgets/properties/cells_page.py` + new `sections/` boxes:
  - **DetectionBox** — channel + global LoG params, **Detect** button (current
    section). Bound to `Project.detection_params`. Disabled when
    `section.counts.source == "imported"`.
  - **ImportCountsBox** — import generic CSV / QUINT JSON for the current section
    (sets source = "imported").
  - **PointsDisplayBox** — point size / color / visibility.
  - **SaveBarBox** (reused) — Save / Clear edits / Reset.

### 6.2 Analysis view — project-wide table + export (mirrors Overview)

- `gui/views/analysis_view.py` — `AnalysisView`. A `QTableWidget` of region rows
  × metric columns (acronym, name, area px, area mm², mean intensity per channel,
  object count, density). Computed on demand via
  `analysis/report.build_region_table`; toggle pooled vs per-section. Filmstrip
  hidden (like Overview).
- `gui/widgets/properties/analysis_page.py` — **pixel size** input (writes
  `Project.pixel_size_um`), metric/channel toggles, hemisphere/region filter,
  **Compute** and **Export…** buttons (folder picker → `export_analysis`).

### 6.3 MainWindow wiring ([`main_window.py`](../src/verso/gui/main_window.py))

- Add `_VIEW_CELLS = 4`, `_VIEW_ANALYSIS = 5`; instantiate in `_build_central`;
  add toolbar buttons in `_build_toolbar`'s `view_specs`; extend `_switch_view`
  `modes` tuple to `(…, "cells", "analysis")` with activate/deactivate + draft
  flush.
- Extend `PropertiesPanel._MODES` / `set_mode` with `"cells"` / `"analysis"`;
  add `self.cells` / `self.analysis` pages.
- Filmstrip visible in Cells, hidden in Analysis.
- **Batch menu** (~line 248): add `&Cells` submenu → "Detect cells in **all**
  slices" (uses `Project.detection_params`, background thread like
  `_BatchMaskWorker` at line 92) + "Clear all counts".
- **Export menu** (~line 292): add "Export **analysis** (CSV)…" →
  `export_analysis`.
- Wire `CellsView.dirty_changed` → `props.cells.save_bar.set_dirty`, and
  Save/Clear/Reset signals exactly as Warp is wired.

---

## 7. Files (when implemented)

**Create**
- `src/verso/engine/analysis/{__init__,region_map,intensity,area,objects,log_detect,density,report}.py`
- `src/verso/engine/io/cell_io.py`, `analysis_export.py`
- `src/verso/gui/views/cells_view.py`, `analysis_view.py`
- `src/verso/gui/widgets/properties/cells_page.py`, `analysis_page.py` (+ new
  `sections/` boxes)
- Engine tests under `tests/engine/`: `test_region_map`, `test_intensity`,
  `test_area`, `test_objects`, `test_density`, `test_cell_io`,
  `test_analysis_export`
- Split this proposal into `.claude/cells-view.md` + `.claude/analysis-view.md`
  specs and add them to the Reference docs list in
  [`CLAUDE.md`](../CLAUDE.md)

**Modify**
- `engine/__init__.py`, `engine/quantification.py`, `engine/io/image_io.py`
- `engine/model/project.py`, `engine/model/status.py`
- `gui/main_window.py`, `gui/widgets/properties/panel.py`
- `.claude/data-model.md`, `CLAUDE.md`

---

## 8. Verification (when implemented)

1. **Engine unit tests** (headless, `uv run pytest tests/engine/`):
   - `section_region_labels` matches `sample_labels` on a known anchoring; warp
     case shifts labels where control points move.
   - `region_intensity` returns the correct mean on a synthetic 2-region image
     and uses raw (non-stretched) pixels.
   - `region_area` pixel counts sum to the in-bounds total.
   - `detect_log_puncta` finds N synthetic Gaussian spots at the right sigma and
     the correlation gate **rejects** a non-Gaussian streak; assignment bins
     points into the right regions.
   - `region_density` = count / mm² with a known pixel size + `working_scale`.
   - `cell_io` round-trips generic CSV and QUINT JSON in the right frame.
   - `analysis_export` writes the modular CSVs with expected headers; pooled vs
     per-section split is consistent.
2. **Lint/format**: `uv run ruff check src/ tests/` + `ruff format`.
3. **Manual GUI** (`uv run python -m verso`): registered project → **Cells** →
   Detect on current section, see spots, Save (writes `counts/<stem>.csv`);
   import a CSV on another section (Detect disabled) → **Batch ▸ Cells ▸ Detect
   all** → **Analysis**: set pixel size, Compute, inspect the region table,
   Export → confirm modular CSVs. Confirm intensity-only analysis works on a
   project with **no** cells.
