"""
migrate_project_v1.py — Convert a VERSO project.json from the old anchoring
convention to the new one.

OLD convention (pre-fix):
  stored_anchoring was saved in *original-image space* — the flip transform
  was applied at Store time to undo the display-space flip.

NEW convention:
  stored_anchoring is saved in *display space* — exactly what the user sees.
  No conversion is applied at Store or export time.

Migration logic:
  For every section that has flip_horizontal or flip_vertical AND a non-zero
  stored_anchoring, the stored value needs to be re-flipped into display space:
      new_stored = flip(old_stored_in_original_space) = display_space

Usage:
    python migrate_project_v1.py path/to/project.json
    python migrate_project_v1.py path/to/project_folder/

The original file is backed up as project.json.bak before writing.
"""

import json
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Anchoring helpers (inlined so the script has no dependency on verso itself)
# ---------------------------------------------------------------------------

def _anchoring_to_vectors(a: list[float]):
    o = a[0:3]
    u = a[3:6]
    v = a[6:9]
    return o, u, v


def _vectors_to_anchoring(o, u, v) -> list[float]:
    return list(o) + list(u) + list(v)


def flip_horizontal(a: list[float]) -> list[float]:
    """Flip anchoring left-right (self-inverse)."""
    o, u, v = _anchoring_to_vectors(a)
    new_o = [o[i] + u[i] for i in range(3)]
    new_u = [-u[i] for i in range(3)]
    return _vectors_to_anchoring(new_o, new_u, v)


def flip_vertical(a: list[float]) -> list[float]:
    """Flip anchoring top-bottom (self-inverse)."""
    o, u, v = _anchoring_to_vectors(a)
    new_o = [o[i] + v[i] for i in range(3)]
    new_v = [-v[i] for i in range(3)]
    return _vectors_to_anchoring(o, u, new_v)


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate(project_path: Path) -> None:
    data = json.loads(project_path.read_text(encoding="utf-8"))
    changed = 0

    for section in data.get("sections", []):
        preprocessing = section.get("preprocessing", {})
        flip_h = preprocessing.get("flip_horizontal", False)
        flip_v = preprocessing.get("flip_vertical", False)

        if not flip_h and not flip_v:
            continue

        alignment = section.get("alignment", {})
        stored = alignment.get("stored_anchoring")
        if not stored or all(v == 0.0 for v in stored):
            continue

        # old stored_anchoring is in original space; flip it into display space
        if flip_h:
            stored = flip_horizontal(stored)
        if flip_v:
            stored = flip_vertical(stored)

        alignment["stored_anchoring"] = stored
        changed += 1
        print(f"  migrated section: {section.get('name', '?')}")

    if changed == 0:
        print("No sections needed migration.")
        return

    backup = project_path.with_suffix(".json.bak")
    shutil.copy2(project_path, backup)
    print(f"Backup written to {backup}")

    project_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Migrated {changed} section(s) in {project_path}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python migrate_project_v1.py <project.json | project_folder>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        target = target / "project.json"

    if not target.exists():
        print(f"File not found: {target}")
        sys.exit(1)

    print(f"Migrating {target} ...")
    migrate(target)


if __name__ == "__main__":
    main()
