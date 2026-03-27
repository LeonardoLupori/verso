# QuickNII / VisuAlign Compatibility Reference

## Why this matters

VERSO must be a drop-in replacement for the QuickNII → VisuAlign pipeline. Users must be able to:

1. Open a QuickNII JSON file and see their registered sections
2. Open a VisuAlign JSON file with control points and continue refining
3. Save results that QuickNII, VisuAlign, PyNutil, and Nutil can read
4. Import DeepSlice output as initial registration

This is the single most important I/O compatibility requirement.

## QUINT ecosystem tools

| Tool | Function | Format |
|---|---|---|
| QuickNII | Affine 2D-to-3D registration | JSON with anchoring matrix per section |
| VisuAlign | Nonlinear refinement | Extends QuickNII JSON with control point data |
| WebAlign / WebWarp | Web-based successors | Same JSON format |
| DeepSlice | Automatic initial registration | Outputs QuickNII-compatible JSON |
| PyNutil | Python quantification library | Reads QuickNII/VisuAlign JSON |
| Nutil | C++ quantification | Reads QuickNII/VisuAlign JSON |
| MeshView | 3D point cloud viewer | Reads point cloud CSV/JSON |

## JSON format (to be documented)

**This section must be filled in before implementing `quint_io.py`.** Download sample QuickNII and VisuAlign JSON files and document the exact schema here.

### What to document

- Top-level structure (metadata, section list)
- Per-section fields (filename, anchoring matrix format)
- Anchoring matrix semantics: what do the numbers mean? How do they map to atlas coordinates?
- VisuAlign extensions: how are control points stored? What coordinate space?
- DeepSlice output: any differences from QuickNII format?

### Known information

- QuickNII uses a JSON format with an anchoring matrix per section
- VisuAlign extends this with control point arrays
- The anchoring matrix encodes the position and orientation of the 2D cut plane within the 3D atlas volume
- The exact matrix format and coordinate conventions need to be reverse-engineered from sample files and documentation

### Action items

1. Download sample QuickNII JSON output files
2. Download sample VisuAlign JSON output files (with control points)
3. Download DeepSlice output JSON
4. Document the exact field names, types, and semantics
5. Write test cases: load QuickNII JSON → save → diff should be minimal
6. Write test cases: load VisuAlign JSON → save → verify control points survive round-trip

## Implementation in VERSO

The compatibility layer lives in `engine/io/quint_io.py`.

### Functions to implement

```python
def load_quicknii(path: Path) -> Project:
    """Load a QuickNII JSON file into a VERSO Project."""

def save_quicknii(project: Project, path: Path) -> None:
    """Save alignment data in QuickNII-compatible JSON format."""

def load_visualign(path: Path) -> Project:
    """Load a VisuAlign JSON file (with control points) into a VERSO Project."""

def save_visualign(project: Project, path: Path) -> None:
    """Save alignment + warp data in VisuAlign-compatible JSON format."""

def load_deepslice(path: Path) -> Project:
    """Load DeepSlice output as initial registration."""
```

### Internal representation

VERSO's internal data model may differ from the QuickNII/VisuAlign JSON format. The conversion happens at the I/O boundary:

- On load: parse QuickNII/VisuAlign JSON → convert to VERSO model types
- On save: convert VERSO model types → QuickNII/VisuAlign JSON format

The VERSO project.json stores data in VERSO's own format. QuickNII/VisuAlign JSON files are written to the `alignments/` subfolder for interoperability.