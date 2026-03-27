# Warping Algorithm Reference

## Algorithm: Delaunay triangulation (piecewise affine)

This matches VisuAlign's approach. Not TPS or RBF.

### Warping scenario

- Background: large histological section (~1000×1000 working resolution) — **static**, never recomputed during interaction
- Overlay: smaller atlas image (~300×200) with transparency — **warped in real time** on every control point drag
- Control points: 20–30 user-placed markers

### How it works

1. Each control point has two coordinate pairs: `(src_x, src_y)` in atlas space and `(dst_x, dst_y)` in warped space
2. Build a Delaunay triangulation on the `src` positions using `scipy.spatial.Delaunay`
3. For each pixel in the overlay, find its containing triangle and compute barycentric coordinates
4. Interpolate the `dst` position using those barycentric coords → produces the dense warp map
5. Apply the warp map with `cv2.remap`

### Why Delaunay over TPS/RBF

- Each triangle is a local affine transform — trivially cheap to evaluate
- Dragging one point only affects neighboring triangles
- 20–30 points → ~40–60 triangles, trivial to rebuild on every drag
- TPS requires re-solving an (N+3)×(N+3) linear system on every drag, then evaluating N kernel terms per pixel (~1.8M kernel calls at 60K pixels with 30 points)
- Delaunay: one triangle lookup + one 2×2 matrix multiply per pixel

### Convex hull constraint

Delaunay triangulation only interpolates — `find_simplex` returns -1 for pixels outside the convex hull of control points. To ensure the entire overlay is warped:

**Add four invisible corner anchor points** at the overlay image corners with `src = dst` (identity mapping). These are managed internally, not visible to or draggable by the user. This matches VisuAlign's behavior.

### Reference implementation

```python
from scipy.spatial import Delaunay
import numpy as np
import cv2

def compute_warp(overlay, src_pts, dst_pts):
    """
    Warp an overlay image using piecewise affine Delaunay triangulation.

    Args:
        overlay: (H, W, C) or (H, W) numpy array, uint8 or float32
        src_pts: (N, 2) array — control point positions in overlay/atlas space
        dst_pts: (N, 2) array — target positions in warped space
            Both arrays must include corner anchors.

    Returns:
        warped: same shape/dtype as overlay
    """
    tri = Delaunay(src_pts)

    h, w = overlay.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Find which triangle each pixel belongs to
    simplices = tri.find_simplex(pixels)

    # Initialize identity warp maps
    map_x = grid_x.astype(np.float32).copy()
    map_y = grid_y.astype(np.float32).copy()

    # Mask: only process pixels inside the triangulation
    valid = simplices >= 0
    valid_pixels = pixels[valid]
    valid_simplices = simplices[valid]

    # Vectorized barycentric coordinate computation
    # transform[s, :2] is a 2x2 matrix, transform[s, 2] is the origin
    T = tri.transform[valid_simplices, :2]       # (M, 2, 2)
    r = valid_pixels - tri.transform[valid_simplices, 2]  # (M, 2)
    b = np.einsum('ijk,ij->ik', T, r)            # (M, 2)
    bary = np.column_stack([b, 1 - b.sum(axis=1)])  # (M, 3)

    # Look up destination coordinates via barycentric interpolation
    idx = tri.simplices[valid_simplices]          # (M, 3) vertex indices
    dx = (bary * dst_pts[idx, 0]).sum(axis=1)
    dy = (bary * dst_pts[idx, 1]).sum(axis=1)

    map_x.ravel()[valid] = dx.astype(np.float32)
    map_y.ravel()[valid] = dy.astype(np.float32)

    warped = cv2.remap(overlay, map_x, map_y, cv2.INTER_LINEAR)
    return warped
```

**Important**: This is the vectorized version. Never use a Python for-loop over pixels.

### Corner anchor helper

```python
def add_corner_anchors(src_pts, dst_pts, overlay_shape):
    """
    Add four corner anchors to ensure full overlay coverage.
    Corners map to themselves (identity).
    """
    h, w = overlay_shape[:2]
    corners = np.array([
        [0, 0], [w - 1, 0],
        [0, h - 1], [w - 1, h - 1]
    ], dtype=np.float64)
    src_all = np.vstack([corners, src_pts])
    dst_all = np.vstack([corners, dst_pts])
    return src_all, dst_all
```

### Performance budget

| Step | Time (300×200 overlay) |
|---|---|
| Rebuild Delaunay (30 pts) | <1ms |
| Build dense warp map (vectorized) | ~5ms |
| `cv2.remap` | ~1ms |
| **Total per drag event** | **~6ms** |

### cv2.remap notes

- Works natively on RGB `(H, W, 3)` — no per-channel loop
- Accepts `float32` or `uint8`. Cast from `float64` if needed: `img.astype(np.float32)`
- Channel-order agnostic (BGR vs RGB doesn't matter for remap)
- Ensure contiguous arrays: `np.ascontiguousarray(img)` if needed

### Full-resolution export

For warping at full resolution during export (not interactive):

- Evaluate warp map on a coarse grid (e.g., 50×50), upsample with `cv2.resize`
- Or use `scipy.ndimage.map_coordinates` for higher-order interpolation
- `cv2.remap` on a full 1000×1000 image: ~10ms

### GUI integration

- Background section: one pyqtgraph `ImageItem`, set once
- Atlas overlay: second `ImageItem`, updated via `setImage(warped)` on every drag event
- `pg.setConfigOption('imageAxisOrder', 'row-major')` to match NumPy
- Drag signal → call `compute_warp()` → update overlay ImageItem