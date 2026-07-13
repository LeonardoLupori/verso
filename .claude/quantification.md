# Quantification — Implementation Notes

> **Status: implemented.** This feature quantifies data that already lives in a
> VERSO project (superseding an earlier cell-*detector* proposal):
> raw image pixels and user-drawn **annotations** (point series = "dots", area
> annotations). Engine: `engine/quantification/` (public API
> `quantify_intensity` / `quantify_area` / `quantify_dots` / `QuantifyOptions`).
> GUI: Export ▸ Quantify. Tests: `tests/engine/test_quantification.py`. This
> document is both the design record and the implementation reference.

---

## 1. Goal

Add the quantification step VERSO is missing (the role PyNutil plays at the end
of the QuickNII → VisuAlign → PyNutil pipeline). Given an aligned VERSO project,
produce per-atlas-region tables for three analyses:

1. **Image intensity** — pixel-value statistics per region, per channel.
2. **Area annotations** — pixel-value statistics per region, restricted to an
   area annotation's footprint.
3. **Dots annotations** — counts, density, and a per-dot coordinate table.

Every analysis is available both from the **GUI** (Export ▸ Quantify) and as a
**public engine function** so a scripting user can align in the GUI and then call
quantification from their own pipeline using only `project-verso.json`.

---

## 2. Confirmed design decisions

| Topic | Decision |
|---|---|
| **Resolution** | Pixel quantification runs at **full-resolution original** images, all channels loaded at once. Peak RAM (~4 GB/section) is acceptable on an analysis machine. |
| **The RULE mask** | "Quantify only within the mask" ⇒ the **Prep slice mask** (`Preprocessing.slice_mask_path`), *not* area-annotation masks. |
| **No slice mask** | If a section has no slice mask and the "use sections without a mask" box is ticked, its scope is the **whole image frame** (every pixel; pixels whose atlas voxel is out-of-brain or out-of-atlas simply fall under `region_id 0`). If the box is **unticked** and any included section lacks a mask, the run **aborts** (parity with the warp gate below). |
| **The slice mask is the *only* silent filter** | Scope masks contain **no `label > 0` constraint**. The slice mask is the single thing that removes pixels/dots from quantification; everything else is quantified. Pixels/dots with no atlas region are pooled under `region_id 0` (a real output row, acronym/name = `background`). |
| **Aggregation source** | brainglobe lacks the Allen curated sets, so **bundle a static membership table** (`src/verso/resources/region_sets.json` — `resources/` is VERSO's existing ship-with-the-package folder) plus `scripts/generate_region_sets.py` (uses AllenSDK) to regenerate it. |
| **Public API return type** | **`list[dict]` records** (stdlib only). Trivially `pd.DataFrame(rows)` for pandas users; CSV written with stdlib `csv`. No new dependency. |
| **Annotation selection** | GUI: dropdown of existing annotations of that type. API: pass the annotation **name**. One annotation per run. |
| **Dots density** | `dots_density = n_dots / n_pixels` (per pixel). No pixel-size value is added to the model. |
| **Output granularity** | **Pooled project-wide** by default — one row per region, aggregated across all included slices. A dialog checkbox **"Separate output per slice"** (default off) instead emits one independent output per section: every file (and every returned records list) is produced once per slice rather than pooled. |
| **Hemisphere split** | Dialog checkbox **"Separate left/right hemispheres"** (default off; `QuantifyOptions.split_hemispheres`). When on, every per-region row (intensity, area, dots-per-region, and their mid/coarse variants) and each per-dot row gain a `hemisphere` column valued `l` / `r` (or `none` for out-of-atlas background). See §4.4. |
| **Dot CCF coords** | **Allen CCFv3 microns**, axis order `x=AP, y=DV, z=LR` (PyNutil/QUINT convention). |
| **Precondition gates** | Alignment is mandatory (no checkbox): abort if any included section is unaligned. **"Use sections without warping control points"** (default off): ticked ⇒ affine-only mapping for CP-less sections; unticked ⇒ abort if any lacks CPs. **"Use sections without a slice mask"** (default off): as above. |
| **Channel columns** | `mean_ch_<Name>` and `tot_ch_<Name>` where `<Name>` is the slugified `ChannelSpec.name`. All channels quantified. **No median** — mean is sufficient. |
| **Dot intensity** | Optional `mean_intensity_ch_<Name>` for **one or more user-chosen channels**; circle **diameter in original px** (default 1 px = the single pixel under the dot). |
| **Missing originals** | Verify every section's `original_path` is reachable before running; if any is missing, **report the list and abort**. |
| **Outputs** | Written to `<project>/exports/quantification_<YYYYMMDD-HHMMSS>/*.csv`. With "Separate output per slice" on, the timestamped folder instead holds one subfolder per section — named by the **slugified image stem** (`slugify(Path(original_path).stem)`, de-duplicated with `_unique_slug` on collision, reusing the helpers from `io/annotation_io.py`) — each with the same filenames inside, no pooled files at the top level. CSV rows carry **no** `section_id`/`image` column: the folder name (and, from the API, the dict key) identifies the slice. |

---

## 3. The shared backbone: per-pixel region map + reconciled mask

Every analysis reduces to: *for each included section, get a full-resolution,
on-disk-oriented map of atlas region IDs, and a boolean mask marking which pixels
are in scope.* This is the single place all coordinate-frame subtleties are
resolved.

### 3.1 Region labels (already implemented)

`VersoRegistration.image_to_atlas(section, kind="annotation", space="full")`
returns an `(H, W)` int32 region-ID array **pixel-matched to the on-disk
original image**, having already applied the section's affine anchoring, the
Delaunay warp, and the preprocessing flips internally. `0` = background /
out-of-atlas. It streams in row-tiles, so memory is bounded regardless of image
size. This is exactly what we need — no new coordinate code.

### 3.2 Frame reconciliation (numerical-correctness critical)

**Verified against the GUI**: raw pixels, region labels, masks, and points all
already live in the **on-disk (un-flipped)** frame, so the only reconciliation
needed is a resolution rescale — **no flips**.

- **Raw pixels** — `load_image(original_path)`, on-disk frame, native dtype. Must
  **not** be contrast-stretched (the cached working thumbnail *is* stretched —
  see `to_multichannel` → `_stretch_per_channel` — and is therefore unusable for
  intensity). We add a raw, un-stretched full-res loader.
- **Region labels** — `image_to_atlas(space="full")`, on-disk frame. ✔ (it applies
  the section's flips *internally* so the returned array indexes on-disk pixels.)
- **Slice / area masks** — stored in the **on-disk working-res frame** (the Prep
  and Annotate views allocate the mask over the un-flipped working image and flip
  it only *for display* — `prep_view._masked_overlay` lines 494-498,
  `annotate_view._build_area_layers` line 339). So: load the working-res mask
  (`load_mask` / `load_area_masks`) and **nearest-neighbour upsample to
  `(full_h, full_w)`** — no flip.
- **Points** — `AnnotationPoint.(x,y)` are on-disk **original-resolution** pixels
  (the Annotate view scales + flips them only at render time, `annotate_view.py`
  line 316), matching `coord_image_to_atlas(space="full")` directly.

The **scope mask** per section is then (no `label > 0` term — see §2):
- **Intensity**: `slice_mask` (or the whole frame when unmasked + box ticked).
- **Area annotation**: `slice_mask ∧ area_mask`.
- **Dots per-region footprint**: same as intensity (defines `n_pixels`).

Pixels inside the scope whose `region_id` is `0` (out-of-brain or out-of-atlas)
are **kept** and pooled into a `region_id 0` row — the slice mask is the only
silent filter.

`region_map(reg, section, atlas, *, slice_mask_policy) -> (labels, scope)` is the
one function that assembles this. **All mask/label resizing uses nearest-neighbour**
(these are integer maps): the working→full-res mask upsample, the area-mask
upsample, and the atlas label resampling (`image_to_atlas` already samples labels
with `cv2.INTER_NEAREST` internally).

---

## 4. Per-analysis algorithm

All three pool across slices region-by-region (the default). With `per_slice=True`
the same region accumulator is simply kept per section instead of merged, so every
description below applies unchanged to a single slice's pixels/dots. Region
metadata (`acronym`, `name`) comes from brainglobe (`AtlasVolume` gains a small
public accessor over `self._bg.structures`).

### 4.1 Image intensity

For each included section: compute `labels`, `scope`, and load raw pixels
`(H, W, C)`. For each channel `c`, over pixels where `scope` is true, accumulate
**per region** into a project-wide accumulator:

- `n_pixels` — running count, `np.bincount(labels[scope])`,
- `tot_ch_c` — running `sum`, `np.bincount(labels[scope], weights=pixels_c)`.

Mean is derived at the end as `tot / n_pixels`. That's the whole computation —
two `bincount`s per channel, no per-region value storage. (No median: means and
sums are exact from running accumulators, so nothing needs to hold pixel lists or
histograms.)

Output rows: `region_id, acronym, name, n_pixels, {mean,tot}_ch_<Name>…`.

### 4.2 Area annotations

Identical to intensity, but `scope = slice_mask ∧ area_mask` for the user-selected
`AreaAnnotation`. Same columns (`n_pixels`, `{mean,tot}_ch_<Name>`); rows are the
regions the annotation touches.

### 4.3 Dots annotations

For the user-selected `PointSeries`:

- **Per-dot table** — for each point (original-res px, grouped by `image`
  basename → section):
  - map to atlas: `reg.coord_image_to_atlas(section, xy, space="full", units="um")`
    → reorder VERSO `(LR,AP,DV)` → Allen `x=AP, y=DV, z=LR` microns,
  - region: `atlas.sample_labels_at(voxel)` (or reuse the returned voxel),
  - **mask gate (RULE)**: keep the dot only if it falls inside the section's slice
    mask (scale the on-disk original px by `working_scale` and index the on-disk
    working mask — no flip),
  - optional `mean_intensity_ch_<Name>` per chosen channel: mean of raw pixels in
    a disk of the given diameter (original px) centred on the dot,
  - columns: `x, y, image, x_ccf, y_ccf, z_ccf, region_id, acronym[, mean_intensity_ch_…]`.
- **Per-region table** — `region_id, acronym, name, n_pixels, n_dots,
  dots_density` where `n_pixels` is the region footprint from `region_map`
  (pooled), `n_dots` counts kept dots per region, `dots_density = n_dots /
  n_pixels`.

Only dots inside the slice mask are counted (RULE); dots on out-of-brain /
out-of-atlas pixels are **kept** with `region_id = 0` in both the per-dot table and
a `region_id 0` per-region row (the slice mask is the only silent filter).

---

### 4.4 Hemisphere split (L / R)

Optional per-hemisphere breakdown, driven by `QuantifyOptions.split_hemispheres`
(default off). The L/R source is **BrainGlobe's authoritative `hemispheres`
volume** (same shape as the annotation; `left_hemisphere_value` /
`right_hemisphere_value`, typically `1` / `2`), **not** a midline threshold — so
it is correct for asymmetric atlases too.

- **Sampling.** A new `AtlasVolume.sample_hemispheres_at` mirrors
  `sample_labels_at` (identical VisuAlign/QUINT voxel selection), and
  `VersoRegistration.image_to_atlas` gained a `kind="hemisphere"` that carries it
  through the **same** affine + Delaunay + flip warp as the region labels. The
  per-pixel hemisphere map is therefore pixel-matched 1:1 with the label map — no
  separate binary subvolumes, no separate warp geometry. `region_map(...,
  split_hemispheres=True)` returns `(labels, scope, hemi)`; otherwise `hemi` is
  `None` and every output is byte-identical to the unsplit run (no new column).
- **Keys.** The pixel/dot accumulators key on `(region_id, hemi)` where `hemi` is
  `None` (unsplit) or the raw atlas value (`1`/`2`/`0`). The `l`/`r`/`none` label
  conversion (`AtlasVolume.hemisphere_label`) is deferred to the row builders,
  which hold the atlas, so the accumulators stay atlas-free.
- **Aggregation is orthogonal.** Mid/coarse regrouping rewrites only the
  `region_id` component of the key to its representative; the hemisphere is
  preserved, so a split survives aggregation.
- **`none` bucket.** Out-of-atlas pixels/dots (`region_id 0`, voxel outside the
  atlas bounds → hemisphere undefined) get `hemisphere = "none"`. In-brain regions
  (`region_id > 0`) always have an in-bounds voxel, so they split cleanly into
  `l`/`r`; only background can carry `none`. Nothing is dropped — the slice mask
  remains the only silent filter.
- **One-sided regions & non-AP projects — no special-casing.** Rows are emitted
  only for `(region, hemi)` keys that actually accumulate, so a region seen on
  only one side yields a single row (no zero-filled partner, no symmetry
  assumption). Hemisphere is read per-pixel from the warped voxel, never from the
  project's interpolation axis, so an ML/sagittal project whose sections are each
  wholly one side flows through identically. For dots, the per-region footprint
  (density denominator) and each dot's hemisphere are read from the **same** `hemi`
  map under the **same** scope, so any bucket with a dot necessarily has a nonzero
  footprint (no divide-by-zero).

## 5. Aggregation (mid / coarse)

The granularity is defined by two Allen structure sets (pinned, matching the
reference implementation in
[wholeBrain_PNN_analysis/wholebrain_tools/aba.py](https://github.com/LeonardoLupori/wholeBrain_PNN_analysis/blob/main/wholebrain_tools/aba.py)):

| Set ID | Members | VERSO level |
|---|---|---|
| `687527670` | 12 major divisions (+ Fiber tracts `1009` appended, → 13) | **coarse** |
| `167587189` | 316 mid-ontology structures (+ Fiber tracts `1009` appended) | **mid** |

Fiber tracts (`1009`) are appended to **both** levels so fibre-tract regions
aggregate to a real bucket rather than falling into `unassigned`.

brainglobe exposes per-structure `structure_id_path` (ancestry) + acronym/name,
but not set membership, so:

- **`scripts/generate_region_sets.py`** (offline, AllenSDK): mirrors `aba.py` —
  `OntologiesApi().get_structures_with_sets(structure_graph_ids=1)` then filters
  nodes whose `structure_set_ids` contains each target set — and writes
  `src/verso/resources/region_sets.json`:
  `{"mid": {"set_id": 167587189, "members": [..., 1009]}, "coarse": {"set_id": 687527670, "members": [..., 1009]}}`
  (Fiber tracts `1009` appended to both levels).
  Committed so the provenance is visible and the table is refreshable if Allen
  changes the sets.
- **Runtime mapping** (`aggregate.py`, no AllenSDK). The reference `aba.py`
  `match_structure_id_lists2` was analysed and **deliberately not ported verbatim**
  — it is order-dependent (takes the "first ancestor in the list", correct only
  because these two sets happen to be non-overlapping partitions), computes the
  ancestor chain twice, and uses O(M) list membership. VERSO uses a simpler,
  order-independent, faster equivalent:
  1. `members = set(level_members)` per level (O(1) membership).
  2. Representative of a region = walk its brainglobe `structure_id_path`
     **in reverse (self → root)** and return the **first** member hit. This is the
     *nearest* member ancestor **by construction**, so it is correct regardless of
     any list ordering and regardless of whether the set is a strict partition;
     `self` is included, so a region that is itself a member maps to itself.
  3. Memoize `{region_id → rep_id}` once for every atlas structure (~1300 × depth
     ≈ a few thousand set lookups, negligible vs the pixel pass); aggregation is
     then a pure dict lookup per row.
  4. Defensive: if a path contains ≥2 members of one level (a malformed/overlapping
     set), warn and keep the nearest.

  Regions with **no** member ancestor are grouped under an `unassigned` bucket
  (`region_id` blank) — documented, not dropped. Because `1009` (Fiber tracts) is
  appended to **both** levels, fibre-tract regions resolve at mid and coarse alike.
  This runtime path touches **only brainglobe**; AllenSDK is confined to the
  offline generator, used solely to *enumerate* members.

Aggregated outputs, produced on request (none / mid / coarse / both):

- **Intensity, area, dots-per-region**: one additional CSV per requested level,
  rows = grouped regions (`n_pixels`/`tot`/`n_dots` summed; `mean` recomputed as
  `Σtot/Σn_pixels`; density recomputed as `Σn_dots/Σn_pixels`).
- **Per-dot table**: add columns `mid_region_id, mid_acronym, coarse_region_id,
  coarse_acronym` (no separate file).

---

## 6. Coordinate & unit reference

- VERSO anchoring voxel order: `(LR=0, AP=1, DV=2)`; brainglobe annotation shape
  `(AP, DV, LR)`.
- `coord_image_to_atlas(..., units="um")` returns `(N,3)` in VERSO order scaled by
  `resolution_um`. Allen CCF output: `x_ccf = col[1] (AP)`, `y_ccf = col[2] (DV)`,
  `z_ccf = col[0] (LR)`, all microns.

---

## 7. Engine module layout

Convert the `engine/quantification.py` stub into a package:

```
engine/quantification/
    __init__.py        # public: quantify_intensity, quantify_area, quantify_dots,
                       #         QuantifyOptions, check_originals_reachable
    region_map.py      # region_map(): full-res labels + reconciled scope mask
    intensity.py       # per-region, per-channel mean + tot via running bincount sums
    area.py            # intensity restricted to (slice ∧ area) scope
    dots.py            # dot→region assignment, per-dot + per-region tables, circle intensity
    aggregate.py       # load region_sets.json; fine→set representative; regroup rows
    tables.py          # RegionRow assembly, list[dict] helpers, channel-column naming
src/verso/resources/
    region_sets.json   # bundled membership table (generated; ships with VERSO)
engine/io/
    quant_export.py    # write records → exports/quantification_<ts>/*.csv (stdlib csv)
engine/io/image_io.py  # + load_full_res_raw(section): un-stretched original, native dtype
engine/atlas.py        # + region_meta(id) -> (acronym, name); structure_id_path accessor
scripts/
    generate_region_sets.py   # AllenSDK regeneration of region_sets.json
```

New public exports in `engine/__init__.py`: `quantify_intensity`, `quantify_area`,
`quantify_dots`, `QuantifyOptions`.

**MATLAB parity**: quantification is added as **standalone functions that *use*
`VersoRegistration` internally**, not as new public methods on `VersoRegistration`.
This deliberately avoids triggering the `matlab/+verso/VersoRegistration.m` parity
requirement (see CLAUDE.md) for image-reading/annotation logic that has no natural
MATLAB mirror. `VersoRegistration`'s public surface is unchanged.

### Public API sketch

```python
from verso.engine import quantify_intensity, QuantifyOptions

rows = quantify_intensity(
    "experiment/project-verso.json",
    options=QuantifyOptions(
        include_unwarped_affine=True,   # else abort if a section lacks CPs
        include_unmasked_wholeframe=False,
        channels=None,                  # None = all
        aggregate=("mid", "coarse"),    # or ()
        per_slice=False,                # True = one independent output per section
        split_hemispheres=False,        # True = add an l/r hemisphere column
        out_dir=None,                   # None = don't write; return records only
    ),
)   # per_slice=False -> {"regions": [ {region_id, acronym, name, n_pixels, mean_ch_…}, … ],
    #                     "regions_mid": [...], "regions_coarse": [...] }
    # per_slice=True  -> {image_name: {"regions": [...], "regions_mid": [...], …}, …}
    #                     (image_name = slugified unique image stem)

quantify_dots("…/project-verso.json", annotation="cells_ch1",
              intensity_channels=["cfos"], dot_diameter_px=3, options=…)
```

`out_dir=None` returns records only (pipeline use); a path writes CSVs **and**
returns them. With `per_slice=True` the return value is keyed by the **slugified
unique image name** (each value the same shape as the pooled result), and CSVs are
written into a like-named subfolder directly under the timestamped export folder —
one independent set of files per section, no pooling. Because the API caller gets
the image name as the dict key, the CSVs deliberately omit any `section_id`/`image`
column; keeping separate files straight is the caller's responsibility.

---

## 8. GUI

`Export` menu (`gui/menus.py`, after the QUINT submenu) → **`&Quantify`** submenu:
`&Intensity…`, `&Dots annotations…`, `&Area annotations…`.

Each opens a modal dialog (`gui/dialogs/quantify_dialog.py`):

- **Common**: precondition checkboxes ("Use sections without a slice mask", "Use
  sections without warping control points"), **"Separate output per slice"**
  (default off), **"Separate left/right hemispheres"** (default off; §4.4),
  aggregation choice (none / mid / coarse / both), output-folder note (defaults
  under `exports/`).
- **Intensity**: channel multi-select (all by default).
- **Area**: area-annotation dropdown + channel multi-select.
- **Dots**: point-series dropdown; "add mean intensity" toggle → channel
  multi-select + diameter (px, default 1).

On **Run**: precondition scan → if unaligned sections, or (unticked box ∧ missing
step), or any unreachable original → show a blocking message listing the offending
sections and abort. Otherwise run in a background worker (pattern of the existing
`_Batch*Worker` in `main_window.py`), then show a completion dialog linking the
output folder. Reuses `require`, the existing job/progress plumbing.

---

## 9. Data-model / doc changes

- **No** new persisted fields (density is per-pixel; annotation chosen per run).
- `.claude/data-model.md`: document `exports/quantification_<ts>/` outputs.
- `CLAUDE.md`: add this file to Reference docs; note the `region_sets.json` +
  `scripts/generate_region_sets.py` provenance.

---

## 10. Files

**Create**: `engine/quantification/{__init__,region_map,intensity,area,dots,aggregate,tables}.py`,
`src/verso/resources/region_sets.json`, `engine/io/quant_export.py`,
`scripts/generate_region_sets.py`, `gui/dialogs/quantify_dialog.py`, engine tests
under `tests/engine/` (`test_region_map`, `test_intensity`, `test_area`,
`test_dots`, `test_aggregate`, `test_quant_export`).

**Modify**: `engine/__init__.py`, `engine/io/image_io.py` (raw loader),
`engine/atlas.py` (region metadata accessor), `gui/menus.py`,
`gui/jobs.py` (QuantifyWorker), `gui/controllers/export_controller.py` (dialog
wiring + worker), `.claude/data-model.md`, `CLAUDE.md`.

---

## 11. Verification

Engine unit tests (headless, `uv run pytest tests/engine/`):

- `region_map`: on a synthetic anchoring, labels match `sample_labels`; the
  reconciled slice mask restricts scope; flips + warp reconcile so mask, labels,
  and raw pixels index the same pixels.
- `intensity`: exact mean/tot on a 2-region synthetic image; verify **raw**
  (non-stretched) pixels are used.
- `area`: scope is the intersection of slice and area masks.
- `dots`: known dots bin into the right regions; CCF axis order/units correct;
  out-of-mask dots dropped from counts; circle mean over a known patch is exact.
- `aggregate`: on a known structure (e.g. a VISp layer), the mid representative is
  the area (VISp) and the coarse is Isocortex — the **nearest** member ancestor,
  asserted to be independent of member-list ordering (shuffle the members and get
  the same result); a region that is itself a member maps to itself; a fibre-tract
  region resolves via `1009` at **both** mid and coarse; a region with no member
  ancestor → `unassigned`; regrouped `mean = Σtot/Σn_pixels` is consistent with the
  pooled fine-level means.
- `per_slice`: with `per_slice=True`, results are keyed by the slugified unique
  image name, CSVs land in a like-named subfolder of the export folder, two
  sections with the same basename get de-duplicated folder names, and summing the
  per-slice `n_pixels`/`tot`/`n_dots` across sections reproduces the pooled result.
- `quant_export`: CSV headers + timestamped folder; `list[dict]` round-trips.

Lint/format: `uv run ruff check src/ tests/` + `ruff format`.

Manual GUI (`uv run python -m verso`): aligned project → Export ▸ Quantify ▸ each
mode → confirm gates fire (unaligned/missing-CP/missing-image), then a good run
writes the expected CSVs; confirm intensity works with no annotations at all.

---

## 12. Critique & open risks

1. **Full-res memory — accepted.** Peak per section ≈ raw `(H,W,C)` + int32 label
   map (~4 GB for a 20000×15000×4 uint16 image), fine on an analysis machine. All
   channels are loaded at once; no RAM-minimizing / per-channel split — the script
   stays simple.
2–3. **No median.** Removed everywhere — only mean and `tot` are emitted, computed
   from running `bincount` sums with no per-region value storage or histograms.
   This drops the plan's most expensive/fiddly piece entirely.
4. **Slice-mask gate — confirmed.** "Unticked box ∧ some section unmasked" **aborts**
   (parity with the warp gate), same as missing CPs.
5. **Frame assumptions — resolved.** Points and masks are stored in the on-disk
   frame (verified in the GUI render paths, §3.2), so no flip reconciliation is
   needed. Unit tests still assert it on a flipped synthetic section.
6. **`unassigned` aggregation bucket — accepted.** Regions with no member ancestor
   are pooled into one `unassigned` row, surfaced in the CSV (transparency over
   silent dropping).
7. **`region_id 0` rows — accepted.** Because the slice mask is the only silent
   filter, an unmasked / whole-frame run pools all out-of-brain/out-of-atlas pixels
   into one `region_id 0` row (acronym `background`). Intentional and surfaced
   clearly so it's easy to ignore downstream — same transparency principle as #6.
```
