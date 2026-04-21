# Prep View — Implementation Plan

Derived from `maskEditor.m` (MATLAB predecessor) and adapted to VERSO's
architecture (PyQt6 / pyqtgraph, engine/GUI separation, non-destructive workflow).

---

## 1. Feature inventory (from maskEditor.m)

| MATLAB feature | VERSO equivalent |
|---|---|
| Channel luminance sliders (R, G per-channel imadjust) | Display-only brightness sliders in right panel |
| Toggle red / green channel visibility (R / G keys) | R/G keyboard shortcuts; sliders in right panel go to 0 when toggled off |
| Mask transparency slider | Mask opacity slider in right panel |
| Freehand draw — add pixels (A key) | Freehand draw tool on canvas |
| Freehand erase — delete pixels (D key) | Eraser tool (or modifier key while drawing) |
| Toggle mask visibility (M key) | Mask visibility button + keyboard shortcut |
| Invert mask polarity (Spacebar) | "Negative" toggle button |
| Navigation (← / →) | Already handled by filmstrip; keep arrow key shortcuts |
| Save mask (Enter, with confirmation) | Auto-save on section change + explicit Save button |

---

## 2. Data model changes

`Preprocessing` already has `slice_mask_path` and `lr_mask_path`. No new fields
needed in the model for mask data.

Add display-only fields **only in PrepView state** (never persisted):

```python
_red_scale: float = 1.0       # [0.01, 1]  multiplier for red channel
_green_scale: float = 1.0     # [0.01, 1]  multiplier for green channel
_mask_opacity: float = 0.4    # matches maskEditor default
_negative_mask: bool = False
_mask_visible: bool = True    # mirrors _show_slice checkbox in _PrepProperties
_current_mask: np.ndarray | None = None   # H×W bool, working-res
_mask_dirty: bool = False                 # unsaved changes flag
```

`_red_scale` / `_green_scale` drop to 0.0 when the R or G key shortcut
toggles the channel off, then restore to the previous slider value when toggled
back on.  No separate bool needed — zero scale means hidden.

---

## 3. Engine additions (`engine/`)

### 3.1 `engine/preprocessing.py`

New functions (pure, headless-safe):

```python
def apply_channel_scale(rgb: np.ndarray, red: float, green: float) -> np.ndarray:
    """Scale R and G channels independently; clip to uint8.
    Equivalent to maskEditor's per-channel imadjust([0, scale], [0, 1]).
    """

def load_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """Load a PNG mask and resize to working-resolution shape (H, W) bool."""

def save_mask(mask: np.ndarray, path: str | Path) -> None:
    """Write a bool H×W mask as a single-channel PNG."""

def mask_to_rgba(mask: np.ndarray, negative: bool, opacity: float,
                 color: tuple[int,int,int] = (255, 255, 255)) -> np.ndarray:
    """Convert bool mask → RGBA overlay (H×W×4 uint8) for display.
    negative=True → unmasked region is highlighted instead.
    """

def apply_freehand_stroke(mask: np.ndarray, polygon_xy: np.ndarray,
                           add: bool) -> np.ndarray:
    """Fill a freehand polygon into mask, adding or erasing.
    polygon_xy: (N, 2) float array of image-pixel coordinates.
    Returns updated mask (copy).
    Uses skimage.draw.polygon or cv2.fillPoly.
    """

def detect_foreground(rgb: np.ndarray) -> np.ndarray:
    """Auto-segment tissue from background using Otsu threshold + largest
    connected component.

    Detects brightfield vs. fluorescence automatically from mean luminance
    (dark mean → brightfield, invert before thresholding; bright mean →
    fluorescence, threshold directly).  Applies morphological closing to fill
    small holes inside the section.

    Returns a bool H×W mask (True = tissue).  Always succeeds — falls back to
    an all-True mask if thresholding produces nothing usable.
    Uses only skimage (already a dependency).
    """
```

---

## 4. GUI changes

### 4.1 Left toolbar (existing, extend)

Replace current placeholder tool buttons with working ones:

| Button | Icon | Keyboard | Action |
|---|---|---|---|
| Draw | pencil | A | Freehand add-to-mask |
| Erase | eraser | D | Freehand erase-from-mask |
| _(separator)_ | | | |
| Undo | ↩ | Ctrl+Z | Undo last stroke |

Remove the Polygon and Fill placeholders for now (can be added later).

Mask visibility, negative toggle, and channel toggles are controlled via the
right panel and keyboard shortcuts only — no duplicate toolbar buttons for them.

Both paths — auto-detect and manual drawing — produce the same `_current_mask`
boolean array and push to the same undo stack.  The user can freely mix them:
run auto-detect, then refine with draw/erase, or skip auto-detect entirely and
draw from scratch.

### 4.2 Right panel — extend `_PrepProperties` (already exists)

`_PrepProperties` in `widgets/properties.py` already exists and is already
mounted inside `PropertiesPanel` (a fixed-width 220 px `QWidget`, **not** a
`QDockWidget`).  Do not create a new widget — extend the existing class.

**Already present — wiring needed**:
- `_show_slice` / `_show_lr` checkboxes in a "Mask visibility" `QGroupBox` —
  exist but are not yet connected to PrepView; add `mask_visibility_changed(bool)`
  and `lr_visibility_changed(bool)` signals and wire them in PrepView
- `_channel_combo` `QComboBox` — exists and emits `channel_changed(int)`;
  already connected in `PropertiesPanel`; keep as-is
- `_flip_h` checkbox; `flip_h_changed` wired through `PropertiesPanel` — keep as-is
- Section info labels (dims, channels) — keep as-is

**Add below "Channel" combo (or replace with a richer group)**:
```
Channel  [Default ▼]          ← existing combo, keep it
Red      [━━━━━●━━━━] 1.00
Green    [━━━━━━━━━●] 1.00
```
- Two `QSlider`s (range 1–100, mapped to 0.01–1.0); default 100
- Sliders are display-only; never persisted
- Emit `channel_scale_changed(red: float, green: float)` signal

**Add to "Mask visibility" `QGroupBox`** (existing group, extend):
```
[■ Show slice mask]     ← existing checkbox
[■ Show L/R boundary]   ← existing checkbox
[⊖ Negative mask]       ← new checkbox
Opacity  [━━━●━━━━━━] 0.40   ← new slider
```
- Emit `mask_opacity_changed(float)` and `mask_negative_changed(bool)` signals

**Add a new "Mask editing" `QGroupBox`**:
```
[  Auto-detect  ]
[  Save mask  ] [ Clear ]
```
- **Auto-detect**: emits `autodetect_requested` signal (PrepView calls engine)
- **Save mask**: emits `save_mask_requested`
- **Clear**: emits `clear_mask_requested`

**Signal propagation**: all new signals must be forwarded through
`PropertiesPanel` as top-level signals (same pattern as `flip_h_changed`,
`opacity_changed`, etc.).  `PropertiesPanel.setFixedWidth(220)` stays as-is.

### 4.3 Canvas interaction — freehand drawing

The `_OverlayViewBox` already captures `canvas_drag_started`, `canvas_dragged`,
`canvas_drag_ended`. Extend PrepView to intercept these when a draw/erase tool
is active:

```
canvas_drag_started → begin stroke, collect first point
canvas_dragged      → append point to stroke buffer
canvas_drag_ended   → finalise polygon, call apply_freehand_stroke, push undo stack
```

The stroke is rendered live as a `pg.PlotCurveItem` (thin red/blue line) on top
of the mask overlay while the user drags. On release it is baked into the mask.

> **Note**: space+drag is already taken by overlay panning in AlignView, but
> PrepView doesn't use the atlas overlay so space+drag can be re-purposed as
> pan (default pyqtgraph behaviour) without conflict.

### 4.4 Mask overlay display

Use `canvas.overlay_item` (already a `pg.ImageItem` at z=10) to show the mask
RGBA.  PrepView has its own `ImageCanvas` instance separate from AlignView's,
so reusing `overlay_item` for the mask does not conflict with the atlas overlay.
Update it whenever:
- the mask changes (after each stroke)
- opacity slider moves
- negative toggle changes
- mask visibility is toggled (set opacity to 0 rather than hide the item)

### 4.5 Undo stack

Simple Python list `_undo_stack: list[np.ndarray]` in PrepView. Push a copy of
`_current_mask` before every stroke. Max depth: 20. Ctrl+Z pops and restores.

### 4.6 Auto-save / explicit save

- Call `save_mask()` + update `section.preprocessing.slice_mask_path` and emit
  `section_modified` when the user explicitly clicks **Save mask** or navigates
  away from the section (with a dirty-flag check).
- No confirmation dialog (unlike maskEditor) — VERSO saves non-destructively and
  can undo.

---

## 5. Keyboard shortcuts (PrepView-local)

| Key | Action |
|---|---|
| A | Switch to Draw tool |
| D | Switch to Erase tool |
| M | Toggle mask visibility |
| N | Toggle negative mask |
| R | Toggle red channel |
| G | Toggle green channel |
| Ctrl+Z | Undo last stroke |
| ← / → | Previous / next section (delegate to filmstrip) |

---

## 6. Implementation order

1. **Engine functions** in `engine/preprocessing.py`:
   `apply_channel_scale`, `load_mask`, `save_mask`, `mask_to_rgba`,
   `apply_freehand_stroke`, `detect_foreground`. Write unit tests in
   `tests/engine/test_preprocessing.py`. `detect_foreground` should be tested
   with a synthetic bright-on-dark and dark-on-bright image to confirm the
   brightfield/fluorescence branch works correctly.

2. **Canvas stroke capture**: extend `_OverlayViewBox` with a mode flag
   (`"draw"` / `"erase"` / `None`) and emit stroke-specific signals, or handle
   in PrepView by connecting to the existing drag signals and ignoring them when
   tool is not active.

3. **PrepView wiring**: connect drag signals → stroke buffer → mask update →
   overlay refresh. Add undo stack.

4. **Extend `_PrepProperties`**: wire existing `_show_slice`/`_show_lr` checkboxes;
   add channel luminance sliders, negative/opacity mask controls, and the
   Auto-detect / Save / Clear buttons. Add all new signals to `PropertiesPanel`
   as forwarded top-level signals.

5. **Toolbar buttons**: replace placeholders with working Draw / Erase / Mask
   visibility / Negative buttons; wire keyboard shortcuts.

6. **Flip**: `flip_horizontal` is already wired through `PropertiesPanel` — no
   changes needed.

7. **Persistence**: on section change, auto-save dirty mask; on load, call
   `load_mask` if `slice_mask_path` is set.

---

## 7. Deferred / out of scope for now

- L/R mask (second mask type) — model field exists, UI can be added in step 4
  alongside the mask-type selector, but actual split-hemisphere logic is separate.
- Polygon and Fill drawing tools (placeholder buttons can remain greyed out).
- Mask import from external PNG (can be added via a menu action later).
