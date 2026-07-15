"""Local migration script (not committed): bake VERSO flips into hiRes channel TIFFs.

The ``hiRes/`` subfolder (not used by VERSO) keeps the full-resolution grayscale
channels of each section, named ``{original_stem}-C1.tif``, ``{original_stem}-C2.tif``,
etc. VERSO stores flips non-destructively, so for any section flipped in VERSO we
physically flip every one of its hiRes channel TIFFs to match.

For each VERSO project found, this:

  1. Reads ``project-verso.json`` and, per section, its flip flags.
  2. Finds that section's channel files in ``hiRes/`` (``{stem}-C*.tif``).
  3. If a flip flag is set, flips each channel in place
     (flip_horizontal -> left/right, flip_vertical -> up/down) and overwrites it.

TIFFs are read/written with tifffile to preserve dtype (e.g. uint16). Only files
matching ``{stem}-C*.tif`` are touched.

Idempotency: flipping in place is not idempotent, so each ``hiRes/`` folder gets a
ledger (``.verso-flip-applied.json``) recording what was applied. A file already
flipped with the same flags is skipped, making re-runs safe.

Usage:
    python flip_hires_for_pipeline.py PROJECT_OR_FOLDER [-r] [--dry-run]
                                      [--channel-glob "-C*.tif"]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

PROJECT_GLOB = "*-verso.json"
DEFAULT_CHANNEL_GLOB = "-C*.tif"
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


def flip_tiff(path: Path, *, flip_h: bool, flip_v: bool) -> None:
    """Flip a single-channel TIFF in place, preserving dtype/compression/photometric."""
    with tifffile.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        arr = page.asarray()
        compression = page.compression  # COMPRESSION enum, e.g. PACKBITS
        photometric = page.photometric  # PHOTOMETRIC enum, e.g. MINISBLACK
        resolution = page.resolution
        resolutionunit = page.resolutionunit
    # Spatial axes are the last two (handles 2-D grayscale and any leading axes).
    if flip_h:
        arr = np.flip(arr, axis=-1)
    if flip_v:
        arr = np.flip(arr, axis=-2)
    tifffile.imwrite(
        str(path),
        np.ascontiguousarray(arr),
        compression=compression,
        photometric=photometric,
        resolution=resolution,
        resolutionunit=resolutionunit,
    )


def process_project(
    project_json: Path, channel_glob: str, dry_run: bool
) -> tuple[int, int]:
    """Flip hiRes channels for one project. Returns ``(flipped, skipped)`` counts."""
    print(f"\n{project_json}")
    try:
        data = json.loads(project_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  ERROR: could not read/parse: {exc}")
        return 0, 0

    hires_dir = project_json.parent / "hiRes"
    if not hires_dir.is_dir():
        print("  no hiRes/ folder — skipping")
        return 0, 0

    ledger = _load_ledger(hires_dir)
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

        channels = sorted(hires_dir.glob(f"{stem}{channel_glob}"))
        if not channels:
            print(f"  MISSING: no {stem}{channel_glob} (flip_h={flip_h}, flip_v={flip_v})")
            continue

        want = {"flip_h": flip_h, "flip_v": flip_v}
        label = "+".join([d for d, on in (("H", flip_h), ("V", flip_v)) if on])

        for ch in channels:
            already = ledger.get(ch.name)
            if already == want:
                print(f"  skip (already flipped): {ch.name}")
                skipped += 1
                continue
            if already is not None and already != want:
                print(
                    f"  WARNING: {ch.name} previously flipped {already} but project "
                    f"now says {want}; leaving file untouched to avoid double-flip. "
                    f"Restore the original and delete its ledger entry to redo."
                )
                skipped += 1
                continue

            if dry_run:
                print(f"  [dry-run] flip {label}: {ch.name}")
            else:
                flip_tiff(ch, flip_h=flip_h, flip_v=flip_v)
                ledger[ch.name] = want
                print(f"  flipped {label}: {ch.name}")
            flipped += 1

    if flipped and not dry_run:
        _save_ledger(hires_dir, ledger)

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
        "--channel-glob", default=DEFAULT_CHANNEL_GLOB,
        help=f"Glob (after the stem) matching a section's channel files "
        f"(default: {DEFAULT_CHANNEL_GLOB!r}, i.e. <stem>-C1.tif, <stem>-C2.tif, ...).",
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
        fl, sk = process_project(f, args.channel_glob, args.dry_run)
        total_flipped += fl
        total_skipped += sk

    print(
        f"\nDone. {total_flipped} channel file(s) "
        f"{'would be ' if args.dry_run else ''}flipped, {total_skipped} skipped, "
        f"across {len(files)} project(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
