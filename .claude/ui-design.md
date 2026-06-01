# UI Design Reference

## Framework

- **PyQt6** with **QDockWidget** panels (locked, no undocking):
  `panel.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)`
- Central widget: `QStackedWidget` with one entry per view.
- **pyqtgraph** for image display, with the OpenGL viewport for fast composition.
- Surrounding panels are resizable via splitters.

## Application views

Four views are wired into the central stack and switched via the top toolbar:

1. **Overview** (`OverviewView`) — table of all sections with progress badges.
2. **Prep** (`PrepView`) — canvas for flips, slice mask, and L/R hemisphere editing.
3. **Align** (`AlignView`) — canvas for affine atlas registration.
4. **Warp** (`WarpView`) — canvas for nonlinear control-point refinement.

`AlignView` and `WarpView` share one `SectionCanvasPanel` instance that is
re-parented into whichever view is active. The shared panel preserves zoom,
pan, and the channel GPU texture across mode switches.

The Prep view has its own `ImageCanvas` (independent of the shared panel).

### Status bar (top of each canvas view)

`Prep`, `Align`, and `Warp` all share the same chrome via
`widgets/view_chrome.make_view_status_bar(label)`: a 36 px-high dark
(`#252525`) bar with a left-aligned filename label (color `#aaa`, 11 px,
vertically centered) and a 1 px bottom border. No buttons live in this bar.

### 1. Overview

Hub for organizing sections and tracking progress. Implemented in
`src/verso/gui/views/overview_view.py`.

- Table columns: serial number, thumbnail, filename, AP position (mm),
  status badges for Prep / Align / Warp.
- Double-click a row → switch to Prep on that section.
- Single-click sets the current section in the shared `AppState`.

### 2. Prep

Implemented in `src/verso/gui/views/prep_view.py`. See [prep-view.md](prep-view.md)
for the full specification.

- Single canvas with the section, slice-mask overlay, and L/R-mask overlay.
- All Prep interactions (flip, mask paint, L/R draw mode) are draft mutations
  that only persist on **Save**.

### 3. Align

Implemented in `src/verso/gui/views/align_view.py`.

- Shared canvas with the atlas overlay (outline or fill, picked from the
  Overlay properties group).
- Left rail: `NavigatorPanel` with translate / rotate / tilt buttons next to
  each axis label, driven by the atlas dimensions.
- Space + mouse drag pans the overlay; navigator scale buttons stretch.
- Anchoring edits are drafts; **Save** commits `anchoring → stored_anchoring`
  and sets the slice's alignment status to `COMPLETE`. **Clear edits** reverts
  unsaved anchoring back to the last-saved plane. **Reset** wipes the
  alignment + the slice's warp control points (since they were anchored to
  the old plane).

### 4. Warp

Implemented in `src/verso/gui/views/warp_view.py`.

- Same shared canvas, but in `warp` interaction mode.
- Click empty canvas to place a control point; drag an existing point to
  move it; **Delete** / **Backspace** removes the hovered point.
- A `_warp_timer` throttles overlay re-warps to ~30 fps during drag.
- Control-point edits are drafts; **Save** persists, **Clear edits** reverts
  to the last-saved control points, **Reset** wipes the slice's control points.

## Properties panel (right dock)

`PropertiesPanel` (in `widgets/properties/panel.py`) is a `QStackedWidget`
with one page per mode:

| Page | File | Sections (`QGroupBox`) |
|---|---|---|
| OverviewPage | `properties/overview_page.py` | overview-specific summary |
| PrepPage     | `properties/prep_page.py`     | `FlipBox`, `MaskBox`, `HemisphereBox`, `SaveBarBox` |
| AlignPage    | `properties/align_page.py`    | `OverlayBox`, `APPlotBox`, `SaveBarBox` |
| WarpPage     | `properties/warp_page.py`     | `OverlayBox`, `ControlPointsBox`, `SaveBarBox` |

Each `*Box` lives in its own file under `widgets/properties/sections/`. The
sub-panels are exposed as plain public attributes (`panel.prep.mask`, etc.)
so MainWindow wires signals to them directly — `PropertiesPanel` itself
does not re-export them.

On Prep / Align / Warp pages the section list scrolls inside a
`QScrollArea`, but `SaveBarBox` is pinned **outside** the scroll area so
the Save / Clear edits / Reset buttons are always visible.

### SaveBarBox

`widgets/properties/sections/save_bar.py`. Titled **"Local changes"**,
with three buttons:

- **Save** — enabled when the view has unsaved draft edits; commits them
  to the in-memory `Section` and triggers a `project.json` write.
- **Clear edits** — enabled when the view has unsaved edits; reverts those
  edits back to the last-saved version (or to default if this slice/view
  was never saved). The on-disk project is **not** touched.
- **Reset** — enabled when the slice has persisted state **or** unsaved
  edits; wipes both saved and unsaved changes back to default and writes
  `project.json`.

Each view exposes `save()`, `revert()`, `clear()`, `is_dirty()`,
`has_persisted_state()`, and a `dirty_changed(bool)` signal. MainWindow
mirrors `dirty_changed` to `SaveBarBox.set_dirty` (which drives the Save +
Clear-edits buttons and one half of Reset) and calls `set_reset_enabled`
with `has_persisted_state()` (the other half of Reset).

**Draft semantics.** All in-canvas edits mutate `Section` in memory but
take a deep-copy *baseline* snapshot of the last-saved state. `save()`
commits the draft to disk; `revert()` rolls the in-memory `Section` back
to that baseline (the per-view **Clear edits**); `clear()` wipes the
slice's view-specific state to disk (the per-view **Reset**). Because
edits survive slice/view navigation, the baseline is preserved across
navigation in `AppState` (`set_baseline` / `get_baseline` / `pop_baseline`,
keyed by `(section_id, step)`) so **Clear edits** still reverts to the
genuine last-saved state after navigating away and back. Close,
open-other-project, import, batch, and export all route through
`MainWindow._confirm_discard_active_draft()` which offers
**Save / Discard / Cancel** across every dirty section.

`Ctrl+S` (File → **Save all**) is global, not view-scoped. It calls the
active view's `save()` first, then persists every remaining dirty
`(section, step)` across all slices (prep / align / warp), and finally
writes `project.json` once. The per-view SaveBar button is the narrower
option — it saves only the current slice/view.

## Filmstrip

`widgets/filmstrip.py`, bottom dock. Visible in Prep / Align / Warp;
hidden in Overview.

- Horizontal strip of section thumbnails (`FILMSTRIP_MAX_SIDE = 150 px`,
  see `engine/io/image_io.py`).
- Click a thumbnail to set the current section.
- Border colour reflects alignment status.

## Top toolbar

Four view-switch buttons (Overview, Preprocess, Align, Warp), styled as
checkable pill buttons. Right-aligned: the current project name.

## Menu bar

- **File** — New / Open / Import (QuickNII, VisuAlign) / Import settings /
  Save / Save As / Export (XML, QuickNII JSON, VisuAlign JSON) / Quit.
- **Image** — Adjust channels / brightness (opens the global brightness
  dialog; brightness is project-wide, not per-section).
- **Batch**
  - **Preprocess**: Autodetect slice masks; *Clear all slice masks*;
    *Clear all L/R masks*.
  - **Align**: Run DeepSlice; Default proposal; Reverse proposal;
    *Clear all alignments*.
  - **Warp**: *Clear all warps*.
- **Export** — Images with atlas overlay.
- **Help** — Atlas info; Project info.

All "Clear all …" entries prompt for confirmation and route through
`MainWindow._after_batch_clear` (resyncs the current view, refreshes the
overview, writes `project.json`).

## Navigation flow

```
Overview (hub) ──→ Prep / Align / Warp (canvas views)
        ▲                │
        └──── toolbar ───┘
```

The filmstrip (or **← / →** keys at the main window) walks through
sections without leaving the canvas. Switching slice or view discards the
active view's unsaved draft; destructive transitions (close, open another
project, import, batch, export) prompt **Save / Discard / Cancel**.

## pyqtgraph canvas setup

```python
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')  # match NumPy
```

The shared `SectionCanvasPanel.canvas` stacks one `ImageItem` per channel
plus overlay `ImageItem`s (mask, L/R mask, atlas outline / fill). The
OpenGL viewport is opt-in inside `widgets/canvas.py`.

## Panel layout sketch

```
┌──────────────────────────────────────────────────────────┐
│  [Overview] [Preprocess] [Align] [Warp]      menubar      │
├──────────────────────────────────────────┬────────────────┤
│                                          │                │
│  central QStackedWidget                  │  properties    │
│  (Overview / Prep / Align / Warp)        │  panel         │
│                                          │  (right dock)  │
│                                          │                │
│                                          │ [Save][Clr edt]│  ← pinned bottom
│                                          │ [   Reset    ] │
├──────────────────────────────────────────┴────────────────┤
│  filmstrip (bottom dock; hidden in Overview)              │
└──────────────────────────────────────────────────────────┘
```
