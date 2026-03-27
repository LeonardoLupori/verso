# UI Design Reference

## Framework

- **PyQt6** with **QDockWidget** panels
- Central widget: pyqtgraph canvas (always fixed in center)
- Surrounding panels: QDockWidget instances, locked in place (not undockable):
  ```python
  panel.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
  ```
- Panels are resizable via draggable splitters

## Application views

Three main views, switchable via tabs in a toolbar.

### 1. Overview (table view)

The hub for organizing sections and tracking progress.

**Table columns**: serial number, thumbnail, filename, AP position, status checkmarks per pipeline step (Flip, L/R mask, Slice mask, Align, Warp)

**Status states per step**:
- Empty — not done
- Amber/tilde — in progress
- Green checkmark — complete

**Right side panel**: preview pane showing selected section's thumbnail with overlay indicators, metadata, action buttons, batch operation controls

**Interactions**:
- Drag-to-reorder rows
- Right-click to rename
- Double-click a section to open in Prep or Align/Warp view

**Summary bar** (bottom): total section count, fully prepped count, in-progress count

**Batch actions**: "Auto-number AP" (interpolate AP positions from first/last), "Flip all selected"

### 2. Prep (canvas view)

Full canvas for preprocessing a single section.

**Canvas**: pyqtgraph image viewer showing the section at working resolution (1200px)

**Left toolbar** (vertical tool palette):
- Pen tool
- Eraser tool
- Polygon tool
- Flood fill
- Undo

**Right panel**:
- Brush size slider
- Mask visibility toggles (show/hide slice mask, L/R mask)
- Section metadata (filename, dimensions, channels)
- Flip horizontal toggle
- Channel selector (for multichannel data)

**Bottom**: filmstrip for navigating between sections without returning to Overview

**Drawing targets**:
- Slice mask: segment the tissue slice from the background
- L/R boundary: mark the left/right hemisphere division

### 3. Align / Warp (canvas view)

Full canvas for atlas registration and nonlinear refinement.

**Canvas**: pyqtgraph image viewer showing histological section with semi-transparent atlas overlay on top

**Two sub-modes** (switchable within this view):
- **Align mode**: adjust the atlas cut plane — AP position, rotation, scaling. This is the QuickNII equivalent.
- **Warp mode**: place and drag control points for nonlinear refinement. This is the VisuAlign equivalent.

**Right panel**:
- Atlas selector (dropdown)
- Overlay opacity slider
- Channel selector
- AP position (numeric input + slider)
- Control point count display
- Registration parameters

**Bottom**: filmstrip with border colors reflecting registration status:
- Green = complete
- Amber = in progress
- Gray = not started

## Filmstrip (shared widget)

Horizontal strip of section thumbnails at the bottom of canvas views (Prep and Align/Warp).

- Click a thumbnail to load that section in the current canvas
- Border color reflects status (green/amber/gray)
- Current section is highlighted
- Not shown in Overview (Overview has its own table)
- Thumbnail size: ~100–150px on long side

## Navigation flow

```
Overview (hub) ←→ Prep (per-section canvas)
Overview (hub) ←→ Align/Warp (per-section canvas)

Within Prep: filmstrip allows sequential section navigation
Within Align/Warp: filmstrip allows sequential section navigation

Align/Warp → Quantify (batch operation, triggered when all sections aligned)
```

The user can always return to Overview from any view. The filmstrip allows working through sections sequentially without returning to Overview each time.

## pyqtgraph canvas setup

```python
import pyqtgraph as pg
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

pg.setConfigOption('imageAxisOrder', 'row-major')  # match NumPy

# Create graphics view with OpenGL acceleration
view = pg.GraphicsLayoutWidget()
view.viewport().setParent(None)
view.setViewport(QOpenGLWidget())

# Add image items
plot = view.addPlot()
plot.setAspectLocked(True)

bg_item = pg.ImageItem()     # histological section (static)
overlay_item = pg.ImageItem()  # atlas overlay (updated on warp)

plot.addItem(bg_item)
plot.addItem(overlay_item)

# Set overlay transparency
overlay_item.setOpacity(0.5)
```

## Panel layout sketch

```
┌─────────────────────────────────────────────────────────┐
│  [Overview]  [Prep]  [Align/Warp]           toolbar     │
├──────┬──────────────────────────────────┬───────────────┤
│      │                                  │               │
│ tool │                                  │  properties   │
│ bar  │        pyqtgraph canvas          │  panel        │
│      │                                  │  (right)      │
│      │                                  │               │
├──────┴──────────────────────────────────┴───────────────┤
│  [ thumb ] [ thumb ] [ thumb ] [ thumb ]   filmstrip    │
└─────────────────────────────────────────────────────────┘
```

The left toolbar only appears in Prep view. The filmstrip appears in Prep and Align/Warp views. The properties panel content changes based on the active view.