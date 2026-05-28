# VERSO

[![Tests](https://github.com/LeonardoLupori/verso/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/LeonardoLupori/verso/actions/workflows/tests.yml)


**V**erso, **E**asy **R**egistration of **S**ections, **O**bviously

VERSO is a desktop application for registering serial histological brain sections to 3D reference atlases. It replaces the QuickNII → VisuAlign → PyNutil pipeline with a single integrated tool.

## What it does

Histological brain sections imaged at high resolution need to be mapped onto a 3D reference atlas (such as the Allen Mouse Brain Atlas) before cell counts or signal measurements can be assigned to brain regions. This process — atlas registration — traditionally requires several separate tools and manual file handoffs between them.

VERSO handles the entire workflow in one place:

- **Import** microscopy images (TIFF, OME-TIFF, PNG, JPEG), including 16-bit and multi-channel files
- **Preprocess** sections non-destructively: flip orientation, draw slice masks and left/right hemisphere masks
- **Register** each section to the atlas by positioning an atlas overlay (anteroposterior position, rotation, scale) using an affine transformation
- **Warp** the atlas overlay onto curved or distorted sections using nonlinear control points (Delaunay triangulation, matching VisuAlign's algorithm)
- **Export** warped images, region-annotated data, and point clouds for downstream quantification

## GUI overview

The interface has four views, switchable from the toolbar:

| View | Purpose |
|---|---|
| **Overview** | Table of all sections with pipeline status at a glance |
| **Prep** | Canvas for preprocessing — masks and flipping |
| **Align** | Canvas for affine atlas registration (AP position, rotation, scale) |
| **Warp** | Canvas for nonlinear refinement using manually placed control points |

A filmstrip of section thumbnails runs along the bottom of the Prep, Align, and Warp views for quick navigation.

## Compatibility

VERSO reads and writes the QuickNII/VisuAlign JSON alignment format natively, so projects can be exchanged with existing tools in the QUINT pipeline.

## Quick start

```bash
uv sync                   # install dependencies
uv run python -m verso    # launch the GUI
```

Requires Python 3.12 and [uv](https://github.com/astral-sh/uv).
