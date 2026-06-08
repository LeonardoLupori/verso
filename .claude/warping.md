# Warping Algorithm Reference

Implementation: `src/verso/engine/warping.py`. Public API:

```python
from verso.engine.warping import warp_overlay, find_atlas_position
```

## Algorithm: Delaunay piecewise affine, backward warp

This matches VisuAlign's `sample(x, y)` approach. Not TPS or RBF.

### Warping scenario

- Background: histological section at working resolution (~1000–4000 px) — **static**, never recomputed during interaction.
- Overlay: atlas image with transparency, sampled at **display resolution** (so its outline stays ~1 screen pixel wide — see [Overlay sampling resolution](#overlay-sampling-resolution--line-thickness)) and **warped in real time** on every control-point drag. Capped at `_OUTLINE_MAX_SIDE = 1280` px settled, `_OUTLINE_DRAG_MAX_SIDE = 820` px while actively dragging.
- Control points: 20–30 user-placed markers.

### How it works

Each control point holds two normalised positions:
- `src_x, src_y` ∈ [0, 1] — position in the **atlas overlay** (where the feature lives in the atlas plane).
- `dst_x, dst_y` ∈ [0, 1] — position in the **section image** (where the user dragged the feature to).

The warp uses a **backward remap** (needed by `cv2.remap`):

1. Add four invisible corner anchors at `(-0.1,-0.1), (1.1,-0.1), (-0.1,1.1), (1.1,1.1)` (10% outside the image on every side) with `src = dst`, matching VisuAlign's `Slice.triangulate()`, so the convex hull covers the full overlay with margin and border pixels are interpolated rather than clamped.
2. Build a Delaunay triangulation on the **destination** (section-space) points.
3. Collapse each triangle to a single affine map `atlas = coef · p + bias`
   (atlas-space output from a section-space pixel). The warp is piecewise
   affine, so one affine per triangle is exact. `build_backward_remap`
   precomputes these affines by reusing qhull's barycentric `tri.transform`
   (no per-triangle solve), then for every output pixel:
   1. Normalise its pixel coords to `[0, 1]` (section-space fractions).
   2. Find the enclosing Delaunay triangle in **dst** space
      (`find_simplex`).
   3. Apply that triangle's affine — a handful of flat multiplies — to get
      the **src** (atlas-space) coords.
   4. Convert back to pixel coords and record in the remap arrays.
4. Apply the remap with `cv2.remap`. RGBA atlas overlays use
   `cv2.INTER_NEAREST` so outline / fill opacity stays constant; opaque
   overlays use `cv2.INTER_LINEAR`.

> **Implementation note.** The per-triangle affine is mathematically
> identical to a barycentric blend of the triangle's `src` vertices, but
> avoids the large per-pixel `(M,2,2)` / `(M,3)` temporaries a literal
> barycentric pass builds — those dominated the cost during live drags.

### Why backward (and Delaunay) and not TPS/RBF

- `cv2.remap` requires "where did this output pixel come from?" maps, so
  the natural direction is dst → src.
- Each triangle is a local affine map — trivially cheap to evaluate.
- Dragging one point only affects neighbouring triangles, so the
  triangulation can be rebuilt every frame.
- TPS requires re-solving an `(N+3)×(N+3)` linear system on every drag
  *and* evaluating `N` kernel terms per pixel.
- Delaunay: one triangle lookup + a 3-coefficient barycentric blend per
  pixel.

### Convex hull constraint

`scipy.spatial.Delaunay.find_simplex` returns `-1` for pixels outside
the convex hull of the input points. The four `(-0.1,-0.1) / (1.1,-0.1)
/ (-0.1,1.1) / (1.1,1.1)` corner anchors (`_CORNERS` in `warping.py`,
prepended via `_with_corners`) sit 10% outside the image on every side
— matching VisuAlign's `Slice.triangulate()` — so the hull covers the
whole `[0,1]²` overlay square with margin and every in-image pixel is
interpolated rather than clamped. These anchors are invisible and not
draggable.

### Triangulation space — aspect ratio (VisuAlign parity)

**Critical for VisuAlign parity.** VisuAlign builds its Delaunay
triangulation in raw section **pixel** space (`width`×`height`, see
`data/Slice.java` + `nonlin/Triangle.java`), but VERSO stores control points
**normalised** to `[0,1]²` (x ÷ width, y ÷ height). A Delaunay triangulation is
invariant under *similarity* transforms but **not** under the anisotropic
`(x/W, y/H)` scaling that normalisation applies when `W ≠ H` — so triangulating
the normalised points directly produces a *different triangle topology*, and
thus a visibly different interior warp, from VisuAlign. (Measured: at aspect
1.425 the topology differs in ~75% of random control-point sets, displacing
interior points by up to ~11% of the image width.)

The warp functions therefore take an `aspect = width / height` argument and
triangulate in `[aspect, 1]`-scaled space (`_tri_scale`), restoring the
section's true pixel aspect up to a uniform factor Delaunay *is* invariant to.
Barycentric interpolation is affine-invariant, so only *which* triangle a point
falls in changes — the interpolated coordinates are unchanged in value. Callers
pass the section's working `width/height` (the Warp view from `panel.raw_image`;
export from `out_w/out_h`); the default `aspect=1.0` assumes a square section.
This is what makes exported VisuAlign markers reproduce VERSO's warp (and vice
versa) exactly inside the frame.

### Public functions

| Function | Purpose |
|---|---|
| `warp_overlay(overlay, src_norm, dst_norm, aspect=1.0)` | Compute the backward remap and apply `cv2.remap`; returns the warped overlay (same shape/dtype). Do **not** pass corner anchors — they are added internally. Pass `aspect=width/height` for VisuAlign parity. |
| `find_atlas_position(s, t, src_norm, dst_norm, aspect=1.0)` | Given a section-space cursor `(s, t) ∈ [0, 1]`, return the matching atlas-space `(u, v)`. Used by the Warp view to display atlas region labels under the cursor. |

Internally, `build_backward_remap(h, w, src_norm, dst_norm, aspect=1.0)`
produces the `(map_x, map_y)` arrays consumed by `warp_overlay`. All four warp
functions share the same `aspect` convention (see [Triangulation space](#triangulation-space--aspect-ratio-visualign-parity)).

### Performance budget

Per-drag-event cost is dominated by `build_backward_remap` and scales with
the warped overlay's pixel count, so the drag cap (`_OUTLINE_DRAG_MAX_SIDE
= 820` px) keeps it interactive. Measured (30 control points):

| Step | 820 px (drag) | 1280 px (settled) |
|---|---|---|
| Rebuild Delaunay (30 pts) | < 1 ms | < 1 ms |
| `find_simplex` (per-pixel) | ~ 10 ms | ~ 25 ms |
| Build dense remap (affine apply) | ~ 35 ms | ~ 85 ms |
| `cv2.remap` | ~ 2 ms | ~ 22 ms |
| **`warp_overlay` total** | **~ 47 ms (~21 fps)** | **~ 113 ms** |

Two things keep a CP drag responsive on top of this:

- **The atlas slice is cached.** `slice_outline` (the affine atlas slice,
  ~44 ms at 820 px / ~124 ms at 1280 px) does *not* change while only the
  control points move, so `SectionCanvasPanel` caches it (keyed on mode /
  plane / colour / sample size) and re-warps the cached slice each frame.
  Only `warp_overlay` runs per tick — the table above.
- **The drag cap.** While a CP is held the outline samples at
  `_OUTLINE_DRAG_MAX_SIDE` instead of the settled `_OUTLINE_MAX_SIDE`, then
  snaps back to full resolution on release. The Warp view toggles this via
  `SectionCanvasPanel.set_overlay_fast()` in its drag start/end handlers
  (Align does the same around its space+drag pan run).

`WarpView._warp_timer` (33 ms interval, ~30 fps) throttles the
`update_overlay` calls during a CP drag so we don't saturate the sampler.

### cv2.remap notes

- Works natively on RGB `(H, W, 3)` and RGBA `(H, W, 4)` — no
  per-channel loop.
- `map_x` and `map_y` must be `float32`.
- Channel-order agnostic (BGR vs RGB doesn't matter for `cv2.remap`).
- Pass `np.ascontiguousarray(img)` to avoid an internal copy.

### Full-resolution export

For exporting the warped overlay at full resolution (non-interactive,
`engine/io/export_images.py`):

- Sample the atlas at a capped working size (~1000 px on the long side)
  via `_ATLAS_SAMPLING_LONG_SIDE`, then upscale.
- `cv2.remap` on a full ~1000 × 1000 image: ~ 10 ms.

### GUI integration

- Background section: one (or more, per channel) pyqtgraph `ImageItem`s,
  set once per section load.
- Atlas overlay: a separate `ImageItem` updated via
  `SectionCanvasPanel.update_overlay()` — which calls
  `warp_overlay` if the active view set a `overlay_post_processor` (Warp
  does; Align does not).
- `pg.setConfigOption('imageAxisOrder', 'row-major')` for NumPy
  conventions.

### Overlay sampling resolution & line thickness

The overlay is sampled small and stretched to fill the section via
`ImageItem.setRect`, so the *sample* resolution sets line thickness, not
placement. The two overlay families size that sample differently
(`SectionCanvasPanel`):

- **Filled** (annotation / reference): sampled at a fixed
  `_FILLED_MAX_SIDE = 512` cap and GPU-stretched. Fill quality is
  resolution-independent, so a small map is plenty.
- **Outline**: sampled at the **on-screen size** of the section
  (`ImageCanvas.image_to_screen_scale()` → image px per screen px), so the
  region-boundary lines are ~1 *screen* pixel wide regardless of the
  section's pixel dimensions or how much of the frame the brain fills. This
  matches VisuAlign, which traces edges in screen space. Sampling at a fixed
  image-proportional cap and stretching (as the filled modes do) instead
  makes line width scale with image dimensions — the bug this replaced.

Because the outline is tied to *display* resolution, it is re-rendered when
the view zooms/pans: `ImageCanvas` emits `view_range_changed`, and the panel
re-samples on a short debounce (`_OUTLINE_REFRESH_MS`). Past the
`_OUTLINE_MAX_SIDE` cap the line simply thickens on deep zoom-in (as
VisuAlign's does past native scale) rather than sampling the atlas ever
finer.
