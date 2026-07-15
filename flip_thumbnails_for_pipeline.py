"""Local migration script (not committed): bake VERSO flips into PNG thumbnails.

VERSO stores flips non-destructively: ``preprocessing.flip_horizontal`` /
``flip_vertical`` flags in ``project-verso.json``, applied on-the-fly. The other
pipeline consumes the PNG thumbnails directly, so for any section that was
flipped in VERSO we must physically flip its PNG.

For each VERSO project found, this:

  1. Reads ``project-verso.json`` and, per section, its flip flags.
  2. Locates the section's PNG in the project's ``thumbnails/`` subfolder
     (``{original_stem}{suffix}.png``; default suffix ``-thumb``).
  3. If a flip flag is set, flips the PNG in place
     (flip_horizontal -> left/right, flip_vertical -> up/down) and overwrites it.

Only ``.png`` files are touched; the ``*-thumb.ome.tif`` working thumbnails are
never modified.

Idempotency: flipping in place is not idempotent, so each ``thumbnails/`` folder
gets a ledger (``.verso-flip-applied.json``) recording what was applied. A PNG
already flipped with the same flags is skipped, making re-runs safe.

Usage:
    python flip_thumbnails_for_pipeline.py PROJECT_OR_FOLDER [-r] [--dry-run]
                                          [--suffix -thumb]
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
DEFAULT_SUFFIX = "-thumb"
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


def _load_ledger(thumbs_dir: Path) -> dict:
    ledger = thumbs_dir / LEDGER_NAME
    try:
        return json.loads(ledger.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_ledger(thumbs_dir: Path, ledger: dict) -> None:
    (thumbs_dir / LEDGER_NAME).write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def flip_png(png: Path, *, flip_h: bool, flip_v: bool) -> None:
    """Flip *png* in place (overwrite) according to the flags."""
    with Image.open(png) as im:
        im.load()
        mode = im.mode
        arr = np.asarray(im)
    if flip_h:
        arr = np.flip(arr, axis=1)
    if flip_v:
        arr = np.flip(arr, axis=0)
    Image.fromarray(np.ascontiguousarray(arr), mode=mode).save(png, format="PNG")


def process_project(project_json: Path, suffix: str, dry_run: bool) -> tuple[int, int]:
    """Flip PNGs for one project. Returns ``(flipped, skipped)`` counts."""
    print(f"\n{project_json}")
    try:
        data = json.loads(project_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  ERROR: could not read/parse: {exc}")
        return 0, 0

    thumbs_dir = project_json.parent / "thumbnails"
    if not thumbs_dir.is_dir():
        print(f"  no thumbnails/ folder — skipping")
        return 0, 0

    ledger = _load_ledger(thumbs_dir)
    flipped = skipped = 0

    for section in data.get("sections", []):
        if not isinstance(section, dict):
            continue
        flip_h, flip_v = _section_flips(section)
        if not (flip_h or flip_v):
            continue

        original_path = section.get("original_path")
        if not original_path:
            print(f"  WARNING: section without original_path — skipped")
            continue
        stem = Path(original_path).stem
        png = thumbs_dir / f"{stem}{suffix}.png"
        if not png.exists():
            print(f"  MISSING: {png.name} (flip_h={flip_h}, flip_v={flip_v})")
            continue

        want = {"flip_h": flip_h, "flip_v": flip_v}
        already = ledger.get(png.name)
        if already == want:
            print(f"  skip (already flipped): {png.name}")
            skipped += 1
            continue
        if already is not None and already != want:
            print(
                f"  WARNING: {png.name} previously flipped {already} but project "
                f"now says {want}; leaving file untouched to avoid double-flip. "
                f"Restore the original PNG and delete its ledger entry to redo."
            )
            skipped += 1
            continue

        action = []
        if flip_h:
            action.append("H")
        if flip_v:
            action.append("V")
        label = "+".join(action)
        if dry_run:
            print(f"  [dry-run] flip {label}: {png.name}")
        else:
            flip_png(png, flip_h=flip_h, flip_v=flip_v)
            ledger[png.name] = want
            print(f"  flipped {label}: {png.name}")
        flipped += 1

    if flipped and not dry_run:
        _save_ledger(thumbs_dir, ledger)

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
        help=f"PNG filename suffix after the original stem (default: {DEFAULT_SUFFIX!r}, "
        f"i.e. <stem>{DEFAULT_SUFFIX}.png).",
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
        f"\nDone. {total_flipped} PNG(s) "
        f"{'would be ' if args.dry_run else ''}flipped, {total_skipped} skipped, "
        f"across {len(files)} project(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
