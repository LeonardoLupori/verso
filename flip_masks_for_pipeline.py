"""Local migration script (not committed): bake VERSO flips into the slice mask PNGs.

VERSO stores masks **unflipped** (on-disk convention) and applies the flip on the
fly. The other pipeline consumes the mask PNGs directly alongside the (physically
flipped) thumbnails / hiRes / counts, so for any section flipped in VERSO we must
physically flip its mask too.

For each VERSO project found, this:

  1. Reads ``project-verso.json`` and, per section, its flip flags.
  2. Locates the section's mask in the project's ``masks/`` subfolder
     (``{original_stem}-mask.png`` — the 1-bit name produced by
     migrate_masks_to_1bit.py; override with --suffix for ``-slice-mask``).
  3. If a flip flag is set, flips the mask in place (flip_horizontal -> left/right,
     flip_vertical -> up/down) and overwrites it, preserving 1-bit depth.

Idempotency: each ``masks/`` folder gets a ledger (``.verso-flip-applied.json``)
recording what was applied, so re-runs skip already-flipped masks.

Usage:
    python flip_masks_for_pipeline.py PROJECT_OR_FOLDER [-r] [--dry-run]
                                      [--suffix -mask]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_PROJECT_FILENAME = "project-verso.json"
PROJECT_GLOB = "*-verso.json"
DEFAULT_SUFFIX = "-mask"
LEDGER_NAME = ".verso-flip-applied.json"


def looks_like_project(path: Path) -> bool:
    """True if *path* parses as a JSON object with a ``sections`` list."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and isinstance(data.get("sections"), list)


def find_project_files(target: Path, recursive: bool) -> list[Path]:
    """Resolve *target* to a list of project json files."""
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(f"No such file or directory: {target}")
    if recursive:
        return sorted(p for p in target.rglob(PROJECT_GLOB) if looks_like_project(p))
    matches = sorted(target.glob(PROJECT_GLOB))
    if matches:
        return matches
    raise FileNotFoundError(
        f"No {PROJECT_GLOB} file in {target} (use --recursive to scan subfolders)"
    )


def _section_flips(section: dict) -> tuple[bool, bool]:
    """Return ``(flip_horizontal, flip_vertical)`` for a section dict."""
    pp = section.get("preprocessing") or {}
    return bool(pp.get("flip_horizontal", False)), bool(pp.get("flip_vertical", False))


def _load_ledger(folder: Path) -> dict:
    try:
        return json.loads((folder / LEDGER_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_ledger(folder: Path, ledger: dict) -> None:
    (folder / LEDGER_NAME).write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def flip_mask(path: Path, *, flip_h: bool, flip_v: bool) -> None:
    """Flip a mask PNG in place (overwrite), preserving 1-bit depth."""
    with Image.open(path) as im:
        im.load()
        is_bilevel = im.mode == "1"
        arr = np.asarray(im.convert("L"))
    if flip_h:
        arr = np.flip(arr, axis=1)
    if flip_v:
        arr = np.flip(arr, axis=0)
    out = Image.fromarray(np.ascontiguousarray(arr), mode="L")
    if is_bilevel:
        out = out.convert("1")
    out.save(path, format="PNG")


def process_project(
    project_json: Path, suffix: str, dry_run: bool
) -> tuple[int, int]:
    """Flip masks for one project. Returns ``(flipped, skipped)`` counts."""
    print(f"\n{project_json}")
    try:
        data = json.loads(project_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  ERROR: could not read/parse: {exc}")
        return 0, 0

    masks_dir = project_json.parent / "masks"
    if not masks_dir.is_dir():
        print("  no masks/ folder — skipping")
        return 0, 0

    ledger = _load_ledger(masks_dir)
    flipped = skipped = 0

    for section in data.get("sections", []):
        if not isinstance(section, dict):
            continue
        flip_h, flip_v = _section_flips(section)
        if not (flip_h or flip_v):
            continue

        original_path = section.get("original_path")
        if not original_path:
            print("  WARNING: section without original_path — skipped")
            continue
        stem = Path(original_path).stem
        mask = masks_dir / f"{stem}{suffix}.png"
        if not mask.exists():
            print(f"  MISSING: {mask.name} (flip_h={flip_h}, flip_v={flip_v})")
            continue

        want = {"flip_h": flip_h, "flip_v": flip_v}
        already = ledger.get(mask.name)
        if already == want:
            print(f"  skip (already flipped): {mask.name}")
            skipped += 1
            continue
        if already is not None and already != want:
            print(
                f"  WARNING: {mask.name} previously flipped {already} but project "
                f"now says {want}; leaving file untouched to avoid double-flip. "
                f"Restore the original mask and delete its ledger entry to redo."
            )
            skipped += 1
            continue

        label = "+".join([d for d, on in (("H", flip_h), ("V", flip_v)) if on])
        if dry_run:
            print(f"  [dry-run] flip {label}: {mask.name}")
        else:
            flip_mask(mask, flip_h=flip_h, flip_v=flip_v)
            ledger[mask.name] = want
            print(f"  flipped {label}: {mask.name}")
        flipped += 1

    if flipped and not dry_run:
        _save_ledger(masks_dir, ledger)

    return flipped, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "target",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Project folder, path to project-verso.json, or (with --recursive) "
        "a directory to scan. Defaults to the current directory.",
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="Recursively find and process every project under the target.",
    )
    parser.add_argument(
        "-n", "--dry-run", action="store_true",
        help="Report what would be flipped without writing any files.",
    )
    parser.add_argument(
        "--suffix", default=DEFAULT_SUFFIX,
        help=f"Mask filename suffix after the original stem (default: {DEFAULT_SUFFIX!r}, "
        f"i.e. <stem>{DEFAULT_SUFFIX}.png). Use '-slice-mask' for un-migrated masks.",
    )
    args = parser.parse_args(argv)

    try:
        files = find_project_files(args.target, args.recursive)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not files:
        print("No project files found.")
        return 0

    total_flipped = total_skipped = 0
    for f in files:
        fl, sk = process_project(f, args.suffix, args.dry_run)
        total_flipped += fl
        total_skipped += sk

    print(
        f"\nDone. {total_flipped} mask(s) "
        f"{'would be ' if args.dry_run else ''}flipped, {total_skipped} skipped, "
        f"across {len(files)} project(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
