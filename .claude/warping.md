# Warping Algorithm Reference

Implementation: `src/verso/engine/warping.py`. Public API:

```python
from verso.engine.warping import warp_overlay, find_atlas_position
```

## Algorithm: Delaunay piecewise affine, backward warp

This matches VisuAlign's `sample(x, y)` approach. Not TPS or RBF.

### Warping scenario

- Background: histological section at working resolution (~1000–4000 px) — **static**, never recomputed during interaction.
- Overlay: smaller atlas image (typically a few hundred px on the long side) with transparency — **warped in real time** on every control-point drag.
- Control points: 20–30 user-placed markers.

### How it works

Each control point holds two normalised positions:
- `src_x, src_y` ∈ [0, 1] — position in the **atlas overlay** (where the feature lives in the atlas plane).
- `dst_x, dst_y` ∈ [0, 1] — position in the **section image** (where the user dragged the feature to).

The warp uses a **backward remap** (needed by `cv2.remap`):

1. Add four invisible corner anchors at `(0,0), (1,0), (0,1), (1,1)` with `src = dst` so the convex hull covers the full overlay.
2. Build a Delaunay triangulation on the **destination** (section-space) points.
3. For every pixel in the output (overlay-sized image):
   1. Normalise its pixel coords to `[0, 1]` (section-space fractions).
   2. Find the enclosing Delaunay triangle in **dst** space.
   3. Compute barycentric coordinates inside that triangle.
   4. Interpolate the matching **src** (atlas-space) coords from the triangle's vertices.
   5. Convert back to pixel coords and record in the remap arrays.
4. Apply the remap with `cv2.remap`. RGBA atlas overlays use
   `cv2.INTER_NEAREST` so outline / fill opacity stays constant; opaque
   overlays use `cv2.INTER_LINEAR`.

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
the convex hull of the input points. The four `(0,0) / (1,0) / (0,1) /
(1,1)` corner anchors (`_CORNERS` in `warping.py`, prepended via
`_with_corners`) guarantee the hull covers the whole `[0,1]²` overlay
square. These anchors are invisible and not draggable.

### Public functions

| Function | Purpose |
|---|---|
| `warp_overlay(overlay, src_norm, dst_norm)` | Compute the backward remap and apply `cv2.remap`; returns the warped overlay (same shape/dtype). Do **not** pass corner anchors — they are added internally. |
| `find_atlas_position(s, t, src_norm, dst_norm)` | Given a section-space cursor `(s, t) ∈ [0, 1]`, return the matching atlas-space `(u, v)`. Used by the Warp view to display atlas region labels under the cursor. |

Internally, `build_backward_remap(h, w, src_norm, dst_norm)` produces
the `(map_x, map_y)` arrays consumed by `warp_overlay`.

### Performance budget

| Step | Time (300 × 200 overlay) |
|---|---|
| Rebuild Delaunay (30 pts) | < 1 ms |
| Build dense remap (vectorised) | ~ 5 ms |
| `cv2.remap` | ~ 1 ms |
| **Total per drag event** | **~ 6 ms** |

`WarpView._warp_timer` (33 ms interval, ~30 fps) throttles the
`update_overlay` calls during a CP drag so we don't saturate the
sampler.

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
