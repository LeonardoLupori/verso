"""Local migration script (not committed): reflect cell coordinates for flipped sections.

The ``counts/`` subfolder holds per-channel cell detections as CSV
(``{original_stem}-cells_C1.csv``, ``{original_stem}-cells_C2.csv``, ...), with
columns ``X,Y,class,score,imgName,thr,rescore``. The ``X``/``Y`` are pixel
coordinates in the **hiRes image space** (the image named in ``imgName``).

VERSO stores flips non-destructively. When a section is flipped in VERSO its hiRes
images are physically flipped (see flip_hires_for_pipeline.py), so the cell
coordinates must be reflected the same way or they no longer land on the cells.

Unlike pixel flipping, this is an *arithmetic* transform:

    flip_horizontal:  X -> W - X     (default; continuous [0,W] frame)
    flip_vertical:    Y -> H - Y

where W, H are the hiRes image width/height. The detector that produces these
CSVs (ciampluca/counting_perineuronal_nets) reports X/Y as torchvision
FasterRCNN box centers, which live in the continuous [0,W]x[0,H] frame (clip
bounds are [0..iw, 0..ih], not W-1/H-1), so ``W - X`` is the faithful reflection.
With ``--reflect pixel`` the formula becomes ``(W-1) - X`` / ``(H-1) - Y``, which
exactly matches np.flip pixel indexing and differs by 1.0 px.

Only the X and Y fields are changed; every other column is preserved byte-for-byte
from the original line. New coordinates are written with ``repr()`` (shortest exact
round-trip). Dimensions are read per-image from the hiRes TIFF in ``imgName``.

Idempotency: each ``counts/`` folder gets a ledger (``.verso-flip-applied.json``)
recording what was applied, so re-runs skip already-reflected files.

Usage:
    python flip_counts_for_pipeline.py PROJECT_OR_FOLDER [-r] [--dry-run]
                                       [--reflect pixel|edge]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tifffile

PROJECT_GLOB = "*-verso.json"
COUNTS_GLOB = "-cells_*.csv"
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


_DIM_CACHE: dict[Path, tuple[int, int]] = {}


def _image_dims(img_path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` of a TIFF without decoding pixels (cached)."""
    if img_path in _DIM_CACHE:
        return _DIM_CACHE[img_path]
    with tifffile.TiffFile(str(img_path)) as t:
        h, w = t.pages[0].shape[:2]
    _DIM_CACHE[img_path] = (int(w), int(h))
    return int(w), int(h)


def reflect_csv(
    csv_path: Path, hires_dir: Path, *, flip_h: bool, flip_v: bool, edge: bool
) -> int:
    """Reflect X/Y in *csv_path* in place. Returns number of data rows changed.

    Only the X and Y fields are rewritten; all other fields keep their exact
    original text. Image dimensions come from the hiRes TIFF named in ``imgName``.
    """
    text = csv_path.read_text(encoding="utf-8")
    # Preserve the original line terminators exactly by splitting on '\n' and
    # keeping any trailing '\r' as part of the last field of a row.
    lines = text.split("\n")
    if not lines:
        return 0

    header = lines[0].rstrip("\r").split(",")
    try:
        xi, yi, img_i = header.index("X"), header.index("Y"), header.index("imgName")
    except ValueError as exc:
        raise ValueError(f"{csv_path.name}: missing expected column ({exc})")

    out: list[str] = [lines[0]]
    changed = 0
    for line in lines[1:]:
        if line == "" or line.strip("\r") == "":
            out.append(line)
            continue
        cr = line.endswith("\r")
        body = line[:-1] if cr else line
        fields = body.split(",")

        img_name = fields[img_i]
        img_path = hires_dir / img_name
        w, h = _image_dims(img_path)

        if flip_h:
            x = float(fields[xi])
            fields[xi] = repr((w - x) if edge else (w - 1 - x))
        if flip_v:
            y = float(fields[yi])
            fields[yi] = repr((h - y) if edge else (h - 1 - y))

        new = ",".join(fields)
        out.append(new + "\r" if cr else new)
        changed += 1

    csv_path.write_text("\n".join(out), encoding="utf-8", newline="")
    return changed


def process_project(
    project_json: Path, *, edge: bool, dry_run: bool
) -> tuple[int, int]:
    """Reflect counts CSVs for one project. Returns ``(files_changed, skipped)``."""
    print(f"\n{project_json}")
    try:
        data = json.loads(project_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  ERROR: could not read/parse: {exc}")
        return 0, 0

    counts_dir = project_json.parent / "counts"
    hires_dir = project_json.parent / "hiRes"
    if not counts_dir.is_dir():
        print("  no counts/ folder — skipping")
        return 0, 0
    if not hires_dir.is_dir():
        print("  no hiRes/ folder — cannot read image dimensions, skipping")
        return 0, 0

    ledger = _load_ledger(counts_dir)
    changed_files = skipped = 0

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

        csvs = sorted(counts_dir.glob(f"{stem}{COUNTS_GLOB}"))
        if not csvs:
            print(f"  MISSING: no {stem}{COUNTS_GLOB} (flip_h={flip_h}, flip_v={flip_v})")
            continue

        want = {"flip_h": flip_h, "flip_v": flip_v, "reflect": "edge" if edge else "pixel"}
        label = "+".join([d for d, on in (("H", flip_h), ("V", flip_v)) if on])

        for csv_path in csvs:
            already = ledger.get(csv_path.name)
            if already == want:
                print(f"  skip (already reflected): {csv_path.name}")
                skipped += 1
                continue
            if already is not None and already != want:
                print(
                    f"  WARNING: {csv_path.name} previously reflected {already} but "
                    f"project now says {want}; leaving untouched to avoid double-flip. "
                    f"Restore the original CSV and delete its ledger entry to redo."
                )
                skipped += 1
                continue

            try:
                if dry_run:
                    # Validate columns + dims without writing.
                    _ = reflect_csv  # noqa: B018 - keep reference clear
                    print(f"  [dry-run] reflect {label}: {csv_path.name}")
                else:
                    n = reflect_csv(
                        csv_path, hires_dir, flip_h=flip_h, flip_v=flip_v, edge=edge
                    )
                    ledger[csv_path.name] = want
                    print(f"  reflected {label} ({n} rows): {csv_path.name}")
            except (ValueError, OSError) as exc:
                print(f"  !! FAILED {csv_path.name}: {exc}")
                continue
            changed_files += 1

    if changed_files and not dry_run:
        _save_ledger(counts_dir, ledger)

    return changed_files, skipped


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
        help="Report what would be reflected without writing any files.",
    )
    parser.add_argument(
        "--reflect", choices=("pixel", "edge"), default="edge",
        help="Reflection convention: 'edge' -> W-X continuous [0,W] frame "
        "(default; correct for FasterRCNN box-center coordinates); 'pixel' -> "
        "(W-1)-X exactly matching np.flip pixel indexing (differs by 1.0 px).",
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

    edge = args.reflect == "edge"
    total_changed = total_skipped = 0
    for f in files:
        ch, sk = process_project(f, edge=edge, dry_run=args.dry_run)
        total_changed += ch
        total_skipped += sk

    print(
        f"\nDone. {total_changed} CSV(s) "
        f"{'would be ' if args.dry_run else ''}reflected, {total_skipped} skipped, "
        f"across {len(files)} project(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
