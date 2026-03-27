# CLAUDE.md

This file provides guidance to Claude Code when working on the VERSO codebase.

**VERSO** — Easy Registration of Sections, Obviously. A desktop application for registering serial histological brain sections to 3D reference atlases, replacing the QuickNII → VisuAlign → PyNutil pipeline with a single tool.

For full project context, see [PROJECT_SPEC.md](PROJECT_SPEC.md).

## Quick reference

```bash
uv sync                              # install all dependencies
uv run python -m verso               # launch GUI
uv run pytest                        # run all tests
uv run pytest tests/engine/          # engine tests only (no display needed)
uv run pytest tests/test_file.py::test_name  # single test
uv run ruff check src/ tests/        # lint
uv run ruff format src/ tests/       # format
```

## Architecture

### Engine / GUI separation (strict)

This is the single most important architectural rule. **Every computation lives in `engine/`. The GUI only calls engine functions and displays results.**

```
user scripts ──→ verso.engine ←── verso.gui
```

- `engine/` is pure Python. It must **never** import from PyQt6, pyqtgraph, or `verso.gui`.
- `gui/` consumes engine functions. All GUI code lives here.
- `engine/__init__.py` is the public API surface — it re-exports key functions so users can write `from verso.engine import slice_volume`.

If you are writing a function that does computation, image processing, file I/O, or data manipulation, it goes in `engine/`. If you are writing a function that creates widgets, handles mouse events, or updates the display, it goes in `gui/`.

### Package layout

```
src/verso/
├── __init__.py              # version
├── __main__.py              # python -m verso → launches gui
│
├── engine/
│   ├── __init__.py          # PUBLIC API — re-exports from submodules
│   ├── model/
│   │   ├── project.py       # Project, Section dataclasses
│   │   ├── alignment.py     # Alignment, anchoring matrix
│   │   ├── mask.py          # Mask metadata (slice, L/R)
│   │   └── coordinates.py   # coordinate space definitions, transform chain
│   ├── io/
│   │   ├── image_io.py      # load TIFF/PNG/JPEG, thumbnails, multi-channel
│   │   ├── project_io.py    # save/load project.json
│   │   ├── quint_io.py      # QuickNII/VisuAlign/DeepSlice JSON read/write
│   │   └── export.py        # warped images, CSVs, point clouds
│   ├── atlas.py             # atlas volume loading (brainglobe), slicing
│   ├── registration.py      # affine registration logic
│   ├── warping.py           # Delaunay triangulation warp, cv2.remap
│   ├── preprocessing.py     # flip, masking (non-destructive)
│   └── quantification.py    # region quantification
│
└── gui/
    ├── app.py               # QApplication setup
    ├── main_window.py       # QMainWindow, mode switching
    ├── views/
    │   ├── overview_view.py # table view, progress tracking
    │   ├── prep_view.py     # canvas for masks, flipping
    │   └── align_view.py    # canvas for affine + warp
    └── widgets/
        ├── canvas.py        # pyqtgraph image viewer + overlay
        ├── filmstrip.py     # horizontal thumbnail strip
        ├── properties.py    # context-sensitive right panel
        └── roi_tools.py     # drawing tools for masks and control points
```

### Non-destructive workflow

Original images are never modified. All preprocessing (flip, masks, contrast) is stored as parameters in `project.json` and applied on-the-fly. Copies are only created on export.

## Technology stack

- **Python 3.12**, managed by uv
- **PyQt6** — GUI framework
- **pyqtgraph** — image canvas (QGraphicsView + OpenGL viewport)
- **numpy** — array computation
- **scipy** — Delaunay triangulation (`scipy.spatial.Delaunay`)
- **opencv-python (cv2)** — image warping (`cv2.remap`)
- **tifffile** — TIFF/OME-TIFF I/O
- **scikit-image** — general image operations
- **brainglobe-atlasapi** — atlas volumes and metadata

## Key technical decisions

### Image resolution tiers

| Tier | Resolution | Purpose |
|---|---|---|
| Full resolution | Original (e.g., 20000×15000) | On disk, used only for final export |
| Working resolution | 1000px on long side | Interactive registration, masks, warping |
| Filmstrip thumbnail | ~100–150px on long side | Overview table, filmstrip nav |

Control points and masks are defined in working resolution space. Scaling factor stored per section.

### Warping algorithm

**Delaunay triangulation (piecewise affine)**, matching VisuAlign's approach. Not TPS or RBF. See [.claude/warping.md](.claude/warping.md) for algorithm details and reference implementation.

Key points:
- Only the atlas overlay (~300×200) is warped per drag event, not the background
- ~6ms total per drag event at 30 control points
- Four invisible corner anchors (src=dst) ensure full overlay coverage (convex hull constraint)

### QuickNII/VisuAlign compatibility

**Critical requirement.** VERSO must read and write the QuickNII/VisuAlign JSON alignment format natively. See [.claude/quint-compat.md](.claude/quint-compat.md) for format details.

### pyqtgraph display

- Use `ImageItem` for numpy arrays (GPU-accelerated via OpenGL viewport)
- Stack two `ImageItem`s: background section (static) + atlas overlay (updated on warp)
- `pg.setConfigOption('imageAxisOrder', 'row-major')` to match NumPy convention
- Built-in histogram widget for contrast adjustment

## Coding conventions

### Python style

- Use `ruff` for linting and formatting (configured in pyproject.toml)
- Type hints on all public functions
- Docstrings on all public functions (Google style)
- Dataclasses (or `@dataclass`) for data model types, not dicts

### Testing

- Engine tests go in `tests/engine/` — these must run headless (no display server)
- GUI tests go in `tests/gui/` — these need a display
- Use `conftest.py` for shared fixtures (sample projects, test images)
- Every engine function gets a unit test before moving on

### Git

- Commit messages: imperative mood, concise (`Add section serialization`, not `Added section serialization support`)
- One logical change per commit

## Project data model

User projects are folders, not single files:

```
my_experiment/
    project.json      # all state, settings, metadata
    thumbnails/       # 1200px working copies
    masks/            # slice masks, L/R masks (PNG)
    alignments/       # QuickNII/VisuAlign-compatible JSON
    exports/          # warped images, CSVs, point clouds
```

Original images are referenced by path in `project.json`, not copied. See [.claude/data-model.md](.claude/data-model.md) for the JSON schema.

## GUI structure

Three application views (modes), switchable via toolbar:

1. **Overview** — table of all sections with progress tracking, batch operations
2. **Prep** — canvas for preprocessing (masks, flipping) with drawing tools
3. **Align/Warp** — canvas for atlas registration (affine) and nonlinear control points

Filmstrip (horizontal thumbnail strip) appears at the bottom of Prep and Align/Warp views. Panels use locked `QDockWidget` (no undocking, resize only).

See [.claude/ui-design.md](.claude/ui-design.md) for detailed view specifications.

## Implementation status

See [PROJECT_SPEC.md](PROJECT_SPEC.md) § 9 for the full roadmap.

**Done**: Project scaffolding (repo, pyproject.toml, src layout, uv environment)

**Next**: Complete dependency setup, create full folder structure, then implement data model (Step 2).

## Reference docs

Detailed specifications that are too long for this file:

- [PROJECT_SPEC.md](PROJECT_SPEC.md) — full project specification
- [.claude/warping.md](.claude/warping.md) — Delaunay triangulation warping algorithm, reference code, performance budget
- [.claude/data-model.md](.claude/data-model.md) — project.json schema, Section fields, coordinate conventions
- [.claude/ui-design.md](.claude/ui-design.md) — detailed view layouts, navigation flow, widget specs
- [.claude/quint-compat.md](.claude/quint-compat.md) — QuickNII/VisuAlign JSON format, compatibility requirements
