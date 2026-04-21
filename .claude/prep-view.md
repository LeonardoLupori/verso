# Prep View - Implementation Plan

Derived from `maskEditor.m` (MATLAB predecessor) and adapted to VERSO's
architecture: PyQt6 / pyqtgraph, engine/GUI separation, and a non-destructive
workflow.

---

## 1. Feature inventory (from maskEditor.m)

| MATLAB feature | VERSO equivalent |
|---|---|
| Channel luminance sliders (R, G per-channel `imadjust`) | Display-only channel luminance sliders in the right panel |
| Toggle red / green channel visibility (R / G keys) | R/G keyboard shortcuts; toggled-off channels render as zero while preserving the previous slider value |
| Mask transparency slider | Mask opacity slider in the right panel |
| Freehand draw - add pixels (A key) | Freehand Draw tool on canvas |
| Freehand erase - delete pixels (D key) | Erase tool on canvas |
| Toggle mask visibility (M key) | Right-panel checkbox + keyboard shortcut |
| Invert mask polarity (Spacebar) | Negative-mask checkbox + keyboard shortcut |
| Navigation (left / right arrows) | Delegate to MainWindow/filmstrip section navigation |
| Save mask (Enter, with confirmation) | Explicit Save mask button + Enter shortcut + automatic dirty-mask save on section/view/project transitions |
| New VERSO addition | Auto-detect foreground/background segmentation, then manually refine with Draw/Erase |

---

## 2. Data model changes

`Preprocessing` already has `slice_mask_path` and `lr_mask_path`. No new
persisted model fields are needed for mask data.

Add display-only fields only in `PrepView` state:

```python
_red_luminance: float = 1.0
_green_luminance: float = 1.0
_red_previous_luminance: float = 1.0
_green_previous_luminance: float = 1.0
_mask_opacity: float = 0.4
_negative_mask: bool = False
_mask_visible: bool = True
_current_mask: np.ndarray | None = None
_mask_dirty: bool = False
_undo_stack: list[np.ndarray] = []
```

The R/G shortcuts should not destroy the user's slider value. When a channel is
toggled off, store its last non-zero luminance value and render that channel as
hidden. When toggled back on, restore the stored value.

---

## 3. Engine additions (`engine/preprocessing.py`)

New functions should be pure and headless-safe.

```python
def apply_channel_luminance(rgb: np.ndarray, red: float, green: float) -> np.ndarray:
    """Apply maskEditor-style per-channel display luminance.

    This is not a simple multiplier. MATLAB's
    imadjust(channel, [0, scale], [0, 1]) maps input value `scale` to output
    white. Lower scale values brighten/saturate the channel. A value of 0 means
    hidden for VERSO's toggle behavior.
    """

def load_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """Load a PNG mask and resize to working-resolution shape (H, W) bool."""

def save_mask(mask: np.ndarray, path: str | Path) -> None:
    """Write a bool HxW mask as a single-channel PNG."""

def mask_to_rgba(
    mask: np.ndarray,
    negative: bool,
    opacity: float,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Convert bool mask to RGBA.

    negative=False highlights True mask pixels.
    negative=True highlights False mask pixels.
    """

def apply_freehand_stroke(
    mask: np.ndarray,
    polygon_xy: np.ndarray,
    add: bool,
) -> np.ndarray:
    """Fill a freehand polygon into a copy of the mask.

    polygon_xy is an (N, 2) array of image-pixel coordinates.
    add=True sets pixels to True; add=False sets pixels to False.
    Use skimage.draw.polygon or cv2.fillPoly.
    """

def detect_foreground(rgb: np.ndarray) -> np.ndarray:
    """Auto-segment tissue from background.

    Estimate background polarity from border luminance, not global mean:
    bright border -> brightfield-like background, tissue tends darker;
    dark border -> fluorescence-like background, tissue tends brighter.

    Use Otsu thresholding, morphological cleanup, hole filling, and largest
    connected-component selection. Return True=tissue. Always return a usable
    bool HxW mask; if segmentation fails, fall back to an all-True mask.
    """
```

Recommended tests in `tests/engine/test_preprocessing.py`:

- `apply_channel_luminance()` lowers scale and brightens/saturates the channel.
- zero scale hides a channel.
- mask load/save round-trips bool masks and resizes with nearest-neighbor.
- freehand add/erase changes only the polygon area.
- `detect_foreground()` works for dark tissue on bright background and bright
  tissue on dark background.

---

## 4. GUI changes

### 4.1 Left toolbar

Replace the current placeholder buttons in `PrepView` with canvas-tool buttons:

| Button | Icon/text | Keyboard | Action |
|---|---|---|---|
| Draw | pencil | A | Freehand add-to-mask |
| Erase | eraser | D | Freehand erase-from-mask |
| Undo | undo arrow | Ctrl+Z | Undo last mask edit |

Remove Polygon and Fill for now. Mask visibility, negative mask, opacity, and
channel controls belong in the right properties panel and keyboard shortcuts,
not duplicated in the toolbar.

Both auto-detect and manual drawing produce the same `_current_mask` bool array
and push to the same undo stack. The user can run auto-detect, then refine with
Draw/Erase, or skip auto-detect and draw from scratch.

### 4.2 Right panel - extend `_PrepProperties`

`_PrepProperties` already exists in `widgets/properties.py` and is mounted
inside `PropertiesPanel`. Extend that class; do not add a second properties
widget.

Wrap `_PrepProperties` content in a `QScrollArea`, mirroring `_AlignProperties`,
because the fixed-width 220 px panel will otherwise become cramped.

Already present, but needs wiring:

- `_show_slice` / `_show_lr` checkboxes in "Mask visibility": add
  `mask_visibility_changed(bool)` and `lr_visibility_changed(bool)`.
- `_channel_combo`: forward existing `channel_changed(int)` through
  `PropertiesPanel` and connect it in `MainWindow` if channel selection is meant
  to affect the loaded working image.
- `_flip_h`: keep the existing `flip_h_changed` forwarding.
- Section info labels: keep as-is, but fill dimensions when available.

Add channel controls below the existing channel combo:

```text
Channel  [Default v]
Red      [slider] 1.00
Green    [slider] 1.00
```

- Slider range: 1-100, mapped to 0.01-1.0.
- Values are display-only and are not persisted.
- Emit `channel_luminance_changed(red: float, green: float)`.

Extend "Mask visibility":

```text
[x] Show slice mask
[x] Show L/R boundary
[ ] Negative mask
Opacity [slider] 0.40
```

- Emit `mask_opacity_changed(float)` and `mask_negative_changed(bool)`.
- Negative mask changes display polarity only; the saved mask remains
  True=tissue/foreground.

Add "Mask editing":

```text
[ Auto-detect ]
[ Save mask ] [ Clear ]
```

- `autodetect_requested`: PrepView calls `detect_foreground()`.
- `save_mask_requested`: PrepView saves the current mask.
- `clear_mask_requested`: PrepView clears the current mask after pushing undo.

Forward all Prep signals through `PropertiesPanel`, following the existing
pattern used for `flip_h_changed`, `opacity_changed`, `ap_changed`, and
`cp_style_changed`.

### 4.3 Canvas interaction

`ImageCanvas` is shared by PrepView and AlignView, so it should not hard-code a
single drag interpretation for every view. Add a small interaction mode API,
for example:

```python
set_interaction_mode("align" | "prep" | "view")
```

Expected behavior:

- Align mode keeps the current behavior: space+left-drag emits `overlay_panned`;
  left-drag emits CP drag signals.
- Prep mode lets PrepView consume left-drag for Draw/Erase strokes when a mask
  tool is active.
- View/default mode should allow pyqtgraph's normal pan/zoom behavior.

PrepView should collect stroke points using the existing canvas drag signal
shape, or stroke-specific signals if the canvas API is expanded:

```text
canvas_drag_started -> begin stroke, collect first point
canvas_dragged      -> append point and update live preview
canvas_drag_ended   -> finalize polygon, push undo, apply_freehand_stroke()
```

Render the live stroke as a `pg.PlotCurveItem` above the mask overlay. Use a
blue-ish line for Draw and a red-ish line for Erase. Clear the preview item once
the stroke is baked into `_current_mask`.

### 4.4 Mask overlay display

Use `canvas.overlay_item` to show the mask RGBA in PrepView. PrepView has its
own `ImageCanvas` instance, so this does not conflict with AlignView's atlas
overlay.

Update the overlay whenever:

- the mask changes,
- opacity changes,
- negative mode changes,
- mask visibility changes,
- a section is loaded or cleared.

When hidden, set the overlay opacity to 0 or clear the overlay; keep
`_current_mask` unchanged.

### 4.5 Undo stack

Use a simple list in PrepView:

```python
_undo_stack: list[np.ndarray]
_UNDO_LIMIT = 20
```

Push a copy of `_current_mask` before every destructive mask edit:

- freehand Draw,
- freehand Erase,
- Auto-detect,
- Clear.

Ctrl+Z and the Undo toolbar button pop and restore the previous mask.

### 4.6 Persistence and dirty-mask lifecycle

Add a `PrepView.save_current_mask_if_dirty()` method. It should:

1. Return immediately if no section, no mask, or not dirty.
2. Derive a deterministic path under the project `masks/` folder.
3. Call `save_mask()`.
4. Update `section.preprocessing.slice_mask_path`.
5. Clear `_mask_dirty`.
6. Emit `section_modified`.

Suggested save path:

```python
masks_dir = Path(section.thumbnail_path).parent.parent / "masks"
mask_path = masks_dir / f"{Path(section.original_path).stem}-slice-mask.png"
```

This matches the existing project structure created by `NewProjectDialog`
(`thumbnails/` and `masks/` sibling folders) without writing beside arbitrary
source images.

Call `save_current_mask_if_dirty()` before:

- `PrepView.load_section()` replaces `self._section`,
- MainWindow switches away from Prep view,
- MainWindow opens/closes/replaces the project,
- application close, if close handling is added.

On explicit Save mask, call the same save method regardless of dirty state if a
mask exists, so the button is predictable.

On section load:

- If `section.preprocessing.slice_mask_path` exists, load it with `load_mask()`.
- Otherwise initialize `_current_mask` as all-False and show no highlighted
  mask until the user draws, clears, or auto-detects.

---

## 5. Keyboard shortcuts (PrepView-local)

| Key | Action |
|---|---|
| A | Switch to Draw tool |
| D | Switch to Erase tool |
| M | Toggle mask visibility |
| N or Spacebar | Toggle negative mask |
| R | Toggle red channel hidden/restored |
| G | Toggle green channel hidden/restored |
| Ctrl+Z | Undo last mask edit |
| Enter / Return | Save mask |
| Left / Right arrows | Previous / next section via MainWindow/filmstrip navigation |

Shortcuts should be scoped to PrepView with
`Qt.ShortcutContext.WidgetWithChildrenShortcut`, matching the approach used by
AlignView for CP deletion.

---

## 6. Implementation order

1. Engine functions in `engine/preprocessing.py`: `apply_channel_luminance`,
   `load_mask`, `save_mask`, `mask_to_rgba`, `apply_freehand_stroke`,
   `detect_foreground`; add focused unit tests.
2. `ImageCanvas` interaction mode and optional stroke-preview item support.
   Preserve AlignView behavior.
3. PrepView state and mask lifecycle: load/init mask, overlay refresh,
   save-current-mask hook, undo stack.
4. PrepView stroke wiring: Draw/Erase tools, drag collection, live preview,
   apply stroke on release.
5. Extend `_PrepProperties` with scroll area, channel luminance controls,
   mask controls, Auto-detect/Save/Clear buttons, and all signal forwarding
   through `PropertiesPanel`.
6. MainWindow wiring: connect Prep properties to PrepView; call dirty-mask save
   hooks before leaving/replacing the current Prep section/project.
7. Toolbar and keyboard shortcuts: Draw, Erase, Undo, R/G, M, N/Space, Enter,
   Ctrl+Z, and left/right navigation delegation.
8. Overview/filmstrip refresh: after saving or clearing a mask, refresh the
   relevant Overview row so "Slice mask" status updates.

---

## 7. Deferred / out of scope for now

- L/R mask editing and split-hemisphere logic.
- Polygon and Fill drawing tools.
- Mask import from external PNG.
- Brush-radius painting. The first pass matches `maskEditor.m` freehand polygon
  behavior, where the closed freehand region is filled on release.
