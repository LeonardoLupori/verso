"""Write quantification records (lists of dicts) to CSV.

Pooled results land as ``*.csv`` directly in a timestamped
``exports/quantification_<ts>/`` folder; per-slice results land in one
slugified-image-name subfolder each (see plan §2 / §7). Stdlib ``csv`` only.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from verso.engine.io.annotation_io import _unique_slug, slugify


def make_output_dir(base_dir: str | Path) -> Path:
    """Create and return ``<base_dir>/quantification_<YYYYMMDD-HHMMSS>/``."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(base_dir) / f"quantification_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_csv(path: str | Path, records: Sequence[dict]) -> None:
    """Write ``records`` (list of dicts) to ``path`` as CSV.

    Columns are the union of all record keys, preserving first-seen order. An
    empty ``records`` writes a header-less empty file so the file always exists.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for rec in records:
        for k in rec:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def slug_for_section(section, used: set[str]) -> str:
    """Unique, filesystem-safe subfolder name from a section's image stem."""
    base = slugify(Path(section.original_path).stem)
    name = _unique_slug(base, used)
    used.add(name)
    return name


def write_result_tables(out_dir: Path, tables: dict[str, Sequence[dict]]) -> list[Path]:
    """Write a ``{filename_stem: records}`` mapping as CSVs in ``out_dir``.

    Returns the paths written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for stem, records in tables.items():
        path = out_dir / f"{stem}.csv"
        write_csv(path, records)
        written.append(path)
    return written
