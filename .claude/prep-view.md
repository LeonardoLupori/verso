# Prep View — Specification

Canvas view for section preprocessing: flipping, slice-mask editing, and
left/right hemisphere labelling. Implementation source of truth is
`src/verso/gui/views/prep_view.py` and the `_PrepProperties` page in
`src/verso/gui/widgets/properties.py`.

Originally derived from `maskEditor.m` (MATLAB predecessor) and adapted to
VERSO's architecture (PyQt6 / pyqtgraph, engine/GUI separation,
non-destructive workflow).

---

## 1. Data model

Persisted on `section.preprocessing` (see [data-model.md](data-model.md)):

- `flip_horizontal`, `flip_vertical` — booleans
- `slice_mask_path` — PNG, bool H×W in working-resolution
- `lr_mask_path` — PNG, uint8 H×W with values {0, 1=left, 2=right}
- `lr_line` — optional `[[x0, y0], [x1, y1]]` storage-frame endpoints used to
  re-seed the line editor next time

Display-only state, held in `PrepView` (never persisted):

```python
_current_mask: np.ndarray | None      # H×W bool, working-res, storage frame
_lr_mask: np.ndarray | None           # H×W uint8 {0,1,2}, storage frame
_mask_dirty: bool                     # unsaved slice-mask changes
_lr_dirty: bool
_mask_visible: bool                   # M shortcut + eye button
_lr_visible: bool                     # eye button
_negative_mask: bool                  # N shortcut + checkbox
_mask_opacity: float = 0.4
_lr_mask_opacity: float = 0.5
_mask_color: (r,g,b) = (255,255,255)
_lr_left_color: (r,g,b)  = (220, 60, 60)
_lr_right_color: (r,g,b) = (60, 130, 220)
_draw_mode: "freehand" | "brush"      # default "freehand"
_brush_radius: int = 20               # mask pixels
_undo_stack: list[np.ndarray]         # max depth 20
_lr_draw_mode: bool                   # line editor active?
```

Masks are stored in the **unflipped** (storage) frame; flips are applied at
display time via `np.fliplr` / `np.flipud`. Mouse coordinates from the canvas
are in the display frame and converted back to storage frame before being
written into the mask.

---

## 2. Engine functions (`engine/preprocessing.py`)

Pure / headless-safe. Public API used by `PrepView`:

| Function | Purpose |
|---|---|
| `channel_lut(spec)` | Build per-channel 256-entry uint8 LUT for display |
| `load_mask(path, shape)` | Load slice-mask PNG, resize to working-res bool |
| `save_mask(mask, path)` | Write bool mask as 1-channel PNG |
| `mask_to_rgba(mask, negative, opacity, color)` | Bool → RGBA overlay |
| `apply_freehand_stroke(mask, polygon_xy, add)` | Polygon fill (on drag-release) |
| `apply_brush_stroke(mask, points_xy, radius, add)` | Disk stamps along a segment |
| `detect_foreground(rgb)` | Auto-segment tissue (Otsu + largest CC + close) |
| `morph_mask(mask, pixels, operation)` | Erode / dilate |
| `load_lr_mask(path, shape)` / `save_lr_mask(mask, path)` | L/R mask I/O (uint8) |
| `lr_mask_to_rgba(mask, opacity, left_color, right_color)` | L/R → RGBA overlay |
| `rasterize_lr_line(p0, p1, shape)` | Rasterize a line into an L/R uint8 mask |
| `flip_lr_mask(mask, horizontal, vertical)` | Mirror L/R mask + swap left/right labels under H flips |

---

## 3. GUI layout

### 3.1 Canvas (central widget)

- A single `ImageCanvas` in `interaction_mode="prep"`
- Status strip above the canvas shows the current section filename
- Three overlay layers (managed by `ImageCanvas`):
  1. Background — per-channel uint8 planes with per-channel LUT + visibility
  2. Slice-mask overlay — RGBA, controlled by `_mask_color`/`_mask_opacity`/`_negative_mask`
  3. L/R overlay — RGBA, controlled by `_lr_left_color`/`_lr_right_color`/`_lr_mask_opacity`
- Drag emits `canvas_drag_started` / `canvas_dragged` / `canvas_drag_ended`
- Alt + wheel emits `alt_wheel_scrolled` (brush-size adjust in brush mode)
- The canvas swaps cursors (draw vs erase vs line-edit) based on Shift state
  and current mode

### 3.2 Right panel — `_PrepProperties`

Single scrollable column inside the shared `PropertiesPanel` dock (the dock is
a fixed-width `QWidget`, not a `QDockWidget`). The panel contains three
`QGroupBox`es, in this order:

**Flip image** — `flip_box`
```
[ ⇄ Horizontal ]   [ ⇅ Vertical ]
```
- Two icon-only checkable buttons
- Emit `flip_h_changed(bool)` / `flip_v_changed(bool)`

**Slice mask** — `mask_box`
```
[👁] [■]                          [ ] Show negative
Opacity   [━━━●━━━━━━] 0.40

[ Freehand ] [ Brush ]
Brush     [━━●━━━━━━━] 20

[ Erode ] [ Expand ]  [ 5 ⇕ ]
[ Auto-detect ]      [ Clear ]
```
- Eye button toggles `_mask_visible` → `mask_visibility_changed(bool)`
- Color swatch opens `QColorDialog` → `mask_color_changed(tuple)`
- "Show negative" checkbox → `mask_negative_changed(bool)`
- Opacity slider 0–100 → `mask_opacity_changed(float)`
- Freehand / Brush segmented buttons (exclusive) → `mask_draw_mode_changed(str)`
- Brush size slider 5–200 → `brush_size_changed(int)`
- Erode / Expand buttons read the spinbox (1–20 px) and emit
  `erode_mask_requested(int)` / `expand_mask_requested(int)`
- Auto-detect → `autodetect_requested`, Clear → `clear_mask_requested`

**Hemisphere** — `hemi_box`
```
[👁]  <status: 'Not set' | 'All left' | 'All right' | 'Line drawn'>  [■L] [■R]
Opacity   [━━━━━●━━━━] 0.50

[ All left ]  [ All right ]
[ Draw line ] [ Clear ]

(while drawing only:)
[ ✓ Apply ] [ ✕ Cancel ]
```
- Eye button → `lr_visibility_changed(bool)`
- Two color swatches → `lr_left_color_changed(tuple)` / `lr_right_color_changed(tuple)`
- Opacity slider 0–100 → `lr_opacity_changed(float)`
- All left / All right → `lr_set_all_left_requested` / `lr_set_all_right_requested`
- Draw line (checkable) → `lr_draw_mode_toggled(bool)`; label changes to
  "Drawing..." while active; Apply/Cancel toolbar appears below
- Clear → `lr_clear_requested`
- Apply/Cancel → `lr_apply_requested` / `lr_cancel_requested`
- While draw mode is active, "All left" / "All right" / "Clear" are disabled

All signals are re-exposed on the outer `PropertiesPanel` for `MainWindow` to
connect to. `PropertiesPanel.setMinimumWidth(130)` — the dock splitter allows
the user to widen it freely.

> **Not in the panel**: per-channel R/G brightness sliders. Brightness is
> handled globally for the whole project via the brightness dialog
> (`dialogs/brightness.py`), not per-view. The earlier plan for in-panel R/G
> sliders was dropped.

---

## 4. Canvas interaction

### 4.1 Slice-mask drawing

Two draw modes, toggled from the right panel:

**Freehand** (default):
- On drag, points accumulate and a `PlotCurveItem` previews the stroke in
  `_DRAW_COLOR` (blue) or `_ERASE_COLOR` (red) based on Shift state at
  drag-start
- On release, the polygon is baked into the mask via `apply_freehand_stroke`
- Strokes with < 3 points are dropped
- One undo snapshot per stroke

**Brush**:
- Single click or drag stamps disks along the cursor path via
  `apply_brush_stroke` (live, no preview-then-bake)
- Brush size: slider in panel, or **Alt + mouse wheel** while hovering the
  canvas (step 5, clamped 5–200)
- Cursor is replaced with a circle-outline pixmap sized to the brush radius
- One undo snapshot per stroke (snapshot taken on drag-start / click)

**Add vs erase**: hold **Shift** to erase. The mode is latched at the start
of a stroke — releasing Shift mid-stroke does not flip add/erase.

### 4.2 L/R line editor

Activated by toggling "Draw line" in the Hemisphere group.

- A `LRLineEditor` overlay (separate widget, `widgets/lr_line_editor.py`) is
  created on the canvas with two draggable endpoint handles
- Initial endpoints: re-seed from `section.preprocessing.lr_line` if present,
  else a vertical line down the centre at 10 %–90 % of image height
- Line color uses the configured L/R left/right colors
- **Apply** rasterises the line via `rasterize_lr_line`, saves the L/R mask
  to disk, and persists `lr_line` endpoints to the section
- **Cancel** restores the previous `lr_mask_path` / `lr_line` and reloads
- Switching sections, toggling a flip, or any other "would orphan the edit"
  event calls `cancel_lr_draw_if_active()` first
- Endpoints are converted between display and storage frames (involutive
  H/V flip) by `_line_endpoint_to_display`

### 4.3 "All left" / "All right"

Sets `_lr_mask` to a constant uint8 array, clears `lr_line`, and saves the
mask to disk immediately (the section is "edited").

### 4.4 Clear (L/R)

Deletes the on-disk L/R mask file, clears `lr_mask_path`, `lr_line`, and
`_lr_mask` — the section returns to "Not set".

---

## 5. Undo stack

- Single Python list `_undo_stack: list[np.ndarray]` for the slice mask only
- Snapshot pushed before every stroke (freehand or brush), auto-detect,
  clear, or erode/expand
- Max depth 20; oldest is popped on overflow
- `undo_mask_edit()` (bound to **U** and `Ctrl+Z`) pops and restores
- L/R changes are **not** undoable — Apply/Cancel in the line editor serves
  that purpose, and All-Left/All-Right/Clear write through immediately

---

## 6. Persistence

- On `load_section`: read PNG from `slice_mask_path` and `lr_mask_path` if
  set; otherwise start with an empty slice mask and no L/R overlay
- On section change: `save_current_mask_if_dirty()` is called before loading
  the next section — no confirmation dialog (non-destructive workflow)
- L/R mask is saved immediately by All-left / All-right / Apply, so there is
  no L/R dirty path on section change
- Slice masks live at `<project>/masks/<stem>-slice-mask.png`
- L/R masks live at `<project>/lr_masks/<stem>_lr.png`
- Every save emits `section_modified` so the main window can persist
  `project.json` and refresh the overview

---

## 7. Keyboard shortcuts

All registered as `WindowShortcut` on `PrepView` and gated so they no-op when
the view is not visible.

| Key | Action |
|---|---|
| **M** | Toggle slice-mask visibility |
| **N** | Toggle "show negative" |
| **U** or **Ctrl+Z** | Undo last slice-mask edit |
| **Shift** (held during drag/click) | Erase instead of draw |
| **Alt + wheel** (over canvas, brush mode) | Adjust brush size |
| **← / →** (from main window) | Previous / next section |

Note: the originally-planned A/D (draw/erase tool switch) and R/G (channel
toggle) shortcuts were not implemented. Draw vs erase is a Shift modifier on
a single tool; channel visibility is handled outside the prep panel.

---

## 8. Deferred / out of scope

- Polygon and bucket-fill drawing tools
- Mask import from external PNG (no menu action yet)
- L/R mask undo
- Per-channel brightness controls inside the prep panel (handled by the
  global brightness dialog)
