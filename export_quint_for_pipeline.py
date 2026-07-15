"""Local migration script (not committed): export QuickNII XML + VisuAlign JSON from VERSO projects.

For every VERSO project (``*-verso.json``) found in a folder tree, this writes:

    {mouseID}-quicknii.xml    (QuickNII native XML — affine anchoring)
    {mouseID}-visualign.json  (VisuAlign JSON — anchoring + warp control points)

next to the project file, using VERSO's own engine exporters (``save_quicknii_xml``
/ ``save_visualign`` — the same functions the GUI calls). Both write coordinates in
display space (anchoring and warp markers as VERSO stores them, no flip), which
matches the physically-flipped thumbnail PNGs / hiRes / reflected counts produced
by the other migration scripts.

Image references point at the existing ``thumbnails/{stem}-thumb.png`` files;
**no PNGs are written or overwritten** — only the .xml/.json outputs. The atlas
voxel shape is inferred from the project's atlas name (allen_mouse_25um/10um/50um),
so no BrainGlobe download is required.

By default both files name VisuAlign's native atlas target (e.g.
ABA_Mouse_CCFv3_2017_25um.cutlas), since we've assessed VisuAlign's atlas is
identical to ours. Pass --target to point them at a different atlas folder
instead, e.g. ``--target allen_mouse_25um.cutlas`` to use the BrainGlobe export.

Usage:
    python export_quint_for_pipeline.py PROJECT_OR_FOLDER [-r] [--dry-run]
                                        [--target allen_mouse_25um.cutlas]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from verso.engine.io.quint_io import (
    _visualign_target,
    save_quicknii_xml,
    save_visualign,
)
from verso.engine.model.project import Project

PROJECT_GLOB = "*-verso.json"


def _retarget(xml_path: Path, json_path: Path, target: str) -> None:
    """Rewrite the atlas ``target`` in the exported QuickNII XML + VisuAlign JSON.

    VERSO's exporters hard-code the stock atlas name (ABA_Mouse_CCFv3_2017_25um.cutlas).
    Pass ``--target`` to point the files at your own atlas folder instead (e.g. the
    BrainGlobe export ``allen_mouse_25um.cutlas``). The target-resolution is left
    unchanged (the BrainGlobe export has identical voxel dimensions).
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    data["target"] = target
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    xml = xml_path.read_text(encoding="utf-8")
    xml = re.sub(r"target='[^']*'", f"target='{target}'", xml, count=1)
    xml_path.write_text(xml, encoding="utf-8")


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


def export_project(project_json: Path, dry_run: bool, target: str | None) -> bool:
    """Export QuickNII XML + VisuAlign JSON for one project. Returns success."""
    print(f"\n{project_json}")
    try:
        project = Project.load(project_json)
    except Exception as exc:  # noqa: BLE001 - local script, report and continue
        print(f"  ERROR: could not load project: {exc}")
        return False

    folder = project_json.parent
    xml_path = folder / f"{project.name}-quicknii.xml"
    json_path = folder / f"{project.name}-visualign.json"

    # Default target: VisuAlign's native .cutlas name for this project's atlas.
    # We've assessed VisuAlign's atlas is identical to ours, so both the QuickNII
    # XML and the VisuAlign JSON should reference the same VisuAlign bundle. An
    # explicit --target still wins. (_visualign_target returns the input unchanged
    # for unknown atlases, preserving the old behaviour.)
    effective_target = target or _visualign_target(
        project.atlas.name if project.atlas else ""
    )[0]

    aligned = sum(1 for s in project.sections if s.alignment.status.name == "COMPLETE")
    warped = sum(1 for s in project.sections if s.warp.control_points)
    print(
        f"  {len(project.sections)} sections "
        f"({aligned} aligned, {warped} with warp), atlas={project.atlas.name}"
    )

    if dry_run:
        print(
            f"  [dry-run] would write {xml_path.name} and {json_path.name}"
            f", target={effective_target}"
        )
        return True

    try:
        save_quicknii_xml(project, xml_path)
        save_visualign(project, json_path)
        _retarget(xml_path, json_path, effective_target)
        print(
            f"  wrote {xml_path.name} and {json_path.name}"
            f"  (target={effective_target})"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR: export failed: {exc}")
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "path", type=Path, nargs="?", default=Path.cwd(),
        help="Project folder, path to project-verso.json, or (with --recursive) "
        "a directory to scan. Defaults to the current directory.",
    )
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Recursively find and export every project under the path.")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="Report what would be exported without writing files.")
    parser.add_argument(
        "--target", default=None,
        help="Override the atlas 'target' in the exported files (e.g. "
        "'allen_mouse_25um.cutlas' to use your BrainGlobe atlas export). "
        "Default is VisuAlign's native target for the project's atlas "
        "(e.g. 'ABA_Mouse_CCFv3_2017_25um.cutlas').",
    )
    args = parser.parse_args(argv)

    try:
        files = find_project_files(args.path, args.recursive)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not files:
        print("No project files found.")
        return 0

    ok = sum(export_project(f, args.dry_run, args.target) for f in files)
    print(
        f"\nDone. {ok} of {len(files)} project(s) "
        f"{'would be ' if args.dry_run else ''}exported "
        f"(QuickNII XML + VisuAlign JSON; PNGs untouched)."
    )
    return 0 if ok == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
