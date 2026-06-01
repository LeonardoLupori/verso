# Prep View — Specification

Canvas view for section preprocessing: flipping, slice-mask editing, and
left/right hemisphere labelling. Implementation source of truth:
`src/verso/gui/views/prep_view.py` and the page / sub-panels under
`src/verso/gui/widgets/properties/`.

---

## 1. Data model

Persisted on `section.preprocessing` (see [data-model.md](data-model.md)):

- `flip_horizontal`, `flip_vertical` — booleans
- `slice_mask_path` — PNG, bool H×W in working-resolution
- `lr_mask_path` — PNG, uint8 H×W with values {0, 1=left, 2=right}
- `lr_line` — optional `[[x0, y0], [x1, y1]]` storage-frame endpoints used to
  re-seed the line editor

Display / draft state, held in `PrepView` (never persisted directly):

```python
_current_mask: np.ndarray | None       # H×W bool, working-res, storage frame
_lr_mask: np.ndarray | None            # H×W uint8 {0,1,2}, storage frame
_mask_dirty: bool                      # in-memory mask differs from saved PNG
_lr_dirty: bool                        # in-memory L/R differs from saved PNG
_baseline_preprocessing: Preprocessing # deep-copy snapshot at load time
_dirty: bool                           # any draft change since baseline
_mask_visible / _lr_visible: bool      # eye toggles + M shortcut
_negative_mask: bool                   # N shortcut + checkbox
_mask_opacity / _lr_mask_opacity: float
_mask_color, _lr_left_color, _lr_right_color: (r, g, b)
_draw_mode: "freehand" | "brush"       # default "freehand"
_brush_radius: int                     # mask pixels (Alt+wheel to adjust)
_undo_stack: list[np.ndarray]          # slice mask only, depth 20
_lr_draw_mode: bool                    # line editor active?
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
| `flip_lr_mask(mask, horizontal, vertical)` | Mirror L/R mask + swap labels under H flips |

---

## 3. GUI layout

### 3.1 Canvas (central widget)

- A single `ImageCanvas` in `interaction_mode="prep"`.
- A 36 px status bar (built via `widgets/view_chrome.make_view_status_bar`)
  shows the current section filename, matching Align / Warp exactly.
- Three overlay layers managed by `ImageCanvas`:
  1. Background — per-channel uint8 planes with per-channel LUT + visibility.
  2. Slice-mask overlay — RGBA, controlled by `_mask_color` /
     `_mask_opacity` / `_negative_mask`.
  3. L/R overlay — RGBA, controlled by `_lr_left_color` /
     `_lr_right_color` / `_lr_mask_opacity`.
- Drag emits `canvas_drag_started` / `canvas_dragged` / `canvas_drag_ended`.
- Alt + wheel emits `alt_wheel_scrolled` (brush-size adjust in brush mode).
- The canvas swaps cursors (draw vs erase vs line-edit) based on Shift
  state and current mode.

### 3.2 Right panel — `PrepPage`

Lives in `widgets/properties/prep_page.py`. Sub-panels (each its own
`QGroupBox` file under `widgets/properties/sections/`):

1. **`FlipBox`** (`flip.py`) — horizontal + vertical flip toggles.
2. **`MaskBox`** (`mask.py`) — slice-mask visibility, opacity, color,
   negative, draw-mode selector, brush size, erode / expand, autodetect,
   clear.
3. **`HemisphereBox`** (`hemisphere.py`) — L/R visibility, opacity, left /
   right colors, status label, set-all-left / set-all-right, draw line,
   clear, plus apply / cancel while the line editor is active.
4. **`SaveBarBox`** (`save_bar.py`) — *Clear* / *Save* buttons, pinned to
   the bottom of the page (outside the scroll area).

Each sub-panel is a public attribute (`page.flip`, `page.mask`,
`page.hemisphere`, `page.save_bar`) so `MainWindow` wires signals to them
directly — there is no signal re-export on `PropertiesPanel`.

> **Not in the panel**: per-channel brightness sliders. Channel
> brightness is project-wide and edited through the floating
> `dialogs/brightness.py` dialog (opened from Image → Adjust
> channels/brightness).

---

## 4. Canvas interaction

### 4.1 Slice-mask drawing

Two draw modes, toggled from the right panel:

**Freehand** (default):
- On drag, points accumulate and a `PlotCurveItem` previews the stroke in
  blue (draw) or red (erase) based on Shift state at drag-start.
- On release, the polygon is baked into the mask via `apply_freehand_stroke`.
- Strokes with < 3 points are dropped.
- One undo snapshot per stroke.

**Brush**:
- Single click or drag stamps disks along the cursor path via
  `apply_brush_stroke` (live, no preview-then-bake).
- Brush size: slider in `MaskBox`, or **Alt + mouse wheel** while hovering
  the canvas (step 5, clamped 5–200).
- Cursor is replaced with a circle-outline pixmap sized to the brush
  radius.
- One undo snapshot per stroke (snapshot taken on drag-start / click).

**Add vs erase**: hold **Shift** to erase. The mode is latched at the
start of a stroke — releasing Shift mid-stroke does not flip add/erase.

### 4.2 L/R line editor

Activated by toggling "Draw line" in `HemisphereBox`.

- A `LRLineEditor` overlay (`widgets/lr_line_editor.py`) is created on
  the canvas with two draggable endpoint handles.
- Initial endpoints: re-seed from `section.preprocessing.lr_line` if
  present, else a vertical line down the centre at 10 %–90 % of image
  height.
- Line colour uses the configured L/R left/right colours.
- **Apply** rasterises the line via `rasterize_lr_line` into `_lr_mask`,
  records the endpoints on `section.preprocessing.lr_line`, and marks
  the view dirty. The PNG is **not** written until Save.
- **Cancel** drops the editor without altering state.
- Switching sections, toggling a flip, or any other "would orphan the
  edit" event calls `cancel_lr_draw_if_active()` first.
- Endpoints convert between display and storage frames (involutive H/V
  flip) via `_line_endpoint_to_display`.

### 4.3 "All left" / "All right"

Sets `_lr_mask` to a constant uint8 array, clears `lr_line`, marks the
view dirty. PNG is written on Save.

### 4.4 Clear (L/R) from `HemisphereBox`

Drops the in-memory `_lr_mask`, clears `lr_line`, and marks the view
dirty. The on-disk PNG is only deleted when the user hits Save (or the
view-level **Clear** in `SaveBarBox`, which wipes everything).

---

## 5. Draft / save / clear / discard

PrepView never writes to disk implicitly. The four shared draft methods
are:

| Method | Effect |
|---|---|
| `save()` | Write dirty slice-mask + L/R PNGs, sync `section.preprocessing.*_path`. If a flip changed during the draft, also wipe the slice's alignment + warp (the old alignment was anchored in the un-flipped frame). Re-snapshot `_baseline_preprocessing`. |
| `clear()` | Delete the slice-mask + L/R PNGs from disk, reset `section.preprocessing` to defaults, drop in-memory buffers, re-snapshot baseline. If a flip toggled to False as part of clearing, also wipe alignment + warp. |
| `discard()` | Deep-copy `_baseline_preprocessing` back onto the section, drop in-memory buffers, reload mask / L/R from disk. Used on slice change, view switch, and when the user picks Discard in a Save/Discard/Cancel dialog. |
| `is_dirty()` | True iff any draft change is pending. Drives `SaveBarBox.set_dirty`. |
| `has_persisted_state()` | True iff there's anything saved on disk to clear. Drives `SaveBarBox.set_clear_enabled`. |

`mark_flip_changed()` is called by `MainWindow._on_flip_*_changed` after
toggling a flip flag on the section, so the view marks itself dirty and
the alignment-wipe side-effect is deferred to `save()`.

A `dirty_changed(bool)` signal is emitted whenever `_dirty` transitions.
`MainWindow` wires it to `props.prep.save_bar.set_dirty`.

---

## 6. Undo stack

- Single Python list `_undo_stack: list[np.ndarray]` for the slice mask
  only.
- Snapshot pushed before every stroke (freehand or brush), autodetect,
  clear, or erode/expand.
- Max depth 20; oldest popped on overflow.
- `undo_mask_edit()` (bound to **U** and `Ctrl+Z`) pops and restores.
- L/R changes are **not** undoable — Apply/Cancel in the line editor
  serves that purpose, and the SaveBar Clear lets the user start over.
- The undo stack is cleared on `load_section`, `discard`, and `clear`.

---

## 7. Persistence

- On `load_section`: read PNGs from `slice_mask_path` and `lr_mask_path`
  if set; otherwise start with an empty slice mask and no L/R overlay.
  Deep-copy `section.preprocessing` into `_baseline_preprocessing`.
- On slice change or view switch: `discard()` reverts the section's
  preprocessing to baseline and reloads PNGs.
- On **Save** (via `SaveBarBox`): write any dirty PNGs, persist
  preprocessing field changes, then `MainWindow` writes `project.json`.
  The SaveBar button is scoped to the current slice/view. `Ctrl+S` is
  global instead — it saves the active view plus every other dirty
  section/step (see below).
- On **Clear**: delete PNGs, reset preprocessing, then `project.json` is
  written.
- Slice masks live at `<project>/masks/<stem>-slice-mask.png`.
- L/R masks live at `<project>/lr_masks/<stem>_lr.png`.

---

## 8. Keyboard shortcuts

All registered as `WindowShortcut` on `PrepView` and gated so they no-op
when the view is not visible.

| Key | Action |
|---|---|
| **M** | Toggle slice-mask visibility |
| **N** | Toggle "show negative" |
| **U** or **Ctrl+Z** | Undo last slice-mask edit |
| **Shift** (held during drag/click) | Erase instead of draw |
| **Alt + wheel** (over canvas, brush mode) | Adjust brush size |
| **Ctrl+S** | Save **all** unsaved edits across every slice/step + write `project.json` (global, not Prep-only) |
| **← / →** (from main window) | Previous / next section |

---

## 9. Deferred / out of scope

- Polygon and bucket-fill drawing tools.
- Mask import from external PNG (no menu action yet).
- L/R mask undo.
- Per-channel brightness controls inside the prep panel (handled by the
  global brightness dialog).
