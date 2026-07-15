"""Local migration script (not committed): generate a ``{mouseID}-info.xml`` from a VERSO project.

The downstream lab pipeline consumes a per-mouse ``*-info.xml`` (see
proj_PNN-highFatDiet/.../CC4B-info.xml for the format). VERSO projects don't have
one, so this builds it from the VERSO ``project-verso.json`` plus a small sidecar
CSV of per-mouse metadata that VERSO does not store.

Fields and where they come from:

  mouseID       <- VERSO project name
  treatment     ) per-mouse metadata, read from the sidecar CSV keyed by mouseID
  genotype      )   (columns: mouseID,treatment,genotype,sex,age,channelNames)
  sex           )
  age           )
  channelNames  <- sidecar CSV 'channelNames', ';'-separated in C1,C2,... order
                   (emitted as one <channelNames> tag each)
  slices:                                    one <slices> per VERSO section
    name        <- original_path stem
    number      <- section slice_index
    well        <- 3rd '_'-token of the name (e.g. CC11B_002_A1_1 -> A1)
    flipped     <- 1 if preprocessing.flip_horizontal else 0
    valid       <- 1 (VERSO has no valid/exclude concept; override not needed yet)

Sections are emitted in VERSO's canonical order (slice_index, then id).

Usage:
    # 1. Make a template CSV listing every mouse found, then fill it in:
    python generate_info_xml.py --make-template metadata.csv -r PATH

    # 2. Generate the info.xml files using that CSV:
    python generate_info_xml.py --metadata metadata.csv -r PATH [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

PROJECT_GLOB = "*-verso.json"
XML_DECL = '<?xml version="1.0" encoding="UTF-8"?>\n'
META_COLUMNS = ["mouseID", "treatment", "genotype", "sex", "age", "channelNames"]


def looks_like_project(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and isinstance(data.get("sections"), list)


def find_project_files(target: Path, recursive: bool) -> list[Path]:
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


def load_metadata(csv_path: Path) -> dict[str, dict[str, str]]:
    """Read the sidecar CSV into ``{mouseID: row}``."""
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in META_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"{csv_path.name}: missing column(s) {missing}; expected {META_COLUMNS}"
            )
        out: dict[str, dict[str, str]] = {}
        for row in reader:
            mid = (row.get("mouseID") or "").strip()
            if mid:
                out[mid] = row
    return out


def _section_sort_key(section: dict) -> tuple[int, str]:
    try:
        idx = int(section.get("slice_index"))
    except (TypeError, ValueError):
        idx = 0
    return idx, str(section.get("id", ""))


def build_info_xml(project: dict, meta: dict[str, str]) -> tuple[str, list[str]]:
    """Return ``(xml_text, warnings)`` for one project + its metadata row."""
    warnings: list[str] = []
    root = ET.Element("struct")

    def child(parent: ET.Element, tag: str, text: str) -> None:
        el = ET.SubElement(parent, tag)
        el.text = text

    mouse_id = str(project.get("name", "")).strip()
    child(root, "mouseID", mouse_id)
    child(root, "treatment", (meta.get("treatment") or "").strip())
    child(root, "genotype", (meta.get("genotype") or "").strip())
    child(root, "sex", (meta.get("sex") or "").strip())
    child(root, "age", (meta.get("age") or "").strip())

    channel_names = [
        c.strip() for c in (meta.get("channelNames") or "").split(";") if c.strip()
    ]
    if not channel_names:
        warnings.append("no channelNames in metadata CSV")
    for cname in channel_names:
        child(root, "channelNames", cname)

    sections = sorted(
        (s for s in project.get("sections", []) if isinstance(s, dict)),
        key=_section_sort_key,
    )
    for s in sections:
        stem = Path(s.get("original_path", "")).stem
        tokens = stem.split("_")
        well = tokens[2] if len(tokens) >= 3 else ""
        if not well:
            warnings.append(f"section '{stem}': could not parse well from name")

        pp = s.get("preprocessing") or {}
        flipped = "1" if pp.get("flip_horizontal") else "0"
        if pp.get("flip_vertical"):
            warnings.append(
                f"section '{stem}': has flip_vertical=True, which the info.xml "
                f"'flipped' field cannot represent (only horizontal)."
            )

        sl = ET.SubElement(root, "slices")
        child(sl, "name", stem)
        child(sl, "number", str(s.get("slice_index", "")))
        child(sl, "well", well)
        child(sl, "flipped", flipped)
        child(sl, "valid", "1")

    ET.indent(root, space="    ")
    body = ET.tostring(root, encoding="unicode")
    return XML_DECL + body + "\n", warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "target", type=Path, nargs="?", default=Path.cwd(),
        help="Project folder, path to project-verso.json, or (with --recursive) "
        "a directory to scan. Defaults to the current directory.",
    )
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Recursively find every project under the target.")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="Report what would be written without writing files.")
    parser.add_argument("--metadata", type=Path,
                        help="Sidecar CSV with per-mouse metadata (required to generate).")
    parser.add_argument("--make-template", type=Path, metavar="CSV",
                        help="Instead of generating, write a template CSV listing every "
                        "discovered mouseID with blank metadata columns, then exit.")
    args = parser.parse_args(argv)

    try:
        files = find_project_files(args.target, args.recursive)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not files:
        print("No project files found.")
        return 0

    # Map each project file -> mouseID (project name), keeping file order.
    projects: list[tuple[Path, dict]] = []
    for f in files:
        try:
            projects.append((f, json.loads(f.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"ERROR: {f}: {exc}", file=sys.stderr)

    # --- template mode ------------------------------------------------------
    if args.make_template:
        ids = sorted({str(p.get("name", "")).strip() for _, p in projects if p.get("name")})
        with args.make_template.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(META_COLUMNS)
            for mid in ids:
                writer.writerow([mid, "", "", "", "", ""])
        print(f"Wrote template with {len(ids)} mouse(s) to {args.make_template}")
        print("Fill in treatment,genotype,sex,age,channelNames (channelNames ';'-separated).")
        return 0

    # --- generate mode ------------------------------------------------------
    if not args.metadata:
        parser.error("--metadata CSV is required (or use --make-template first)")
    try:
        metadata = load_metadata(args.metadata)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    written = skipped = 0
    for f, project in projects:
        mouse_id = str(project.get("name", "")).strip()
        print(f"\n{f}  (mouseID={mouse_id})")
        if mouse_id not in metadata:
            print(f"  SKIP: no metadata row for '{mouse_id}' in {args.metadata.name}")
            skipped += 1
            continue

        xml_text, warnings = build_info_xml(project, metadata[mouse_id])
        for w in warnings:
            print(f"  WARNING: {w}")

        out_path = f.parent / f"{mouse_id}-info.xml"
        if args.dry_run:
            n_slices = xml_text.count("<slices>")
            print(f"  [dry-run] would write {out_path.name} ({n_slices} slices)")
        else:
            out_path.write_text(xml_text, encoding="utf-8", newline="")
            print(f"  wrote {out_path.name}")
        written += 1

    print(
        f"\nDone. {written} info.xml "
        f"{'would be ' if args.dry_run else ''}written, {skipped} skipped (no metadata)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
