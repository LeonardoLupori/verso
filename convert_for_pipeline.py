"""Local orchestrator (not committed): run the full VERSO -> legacy-pipeline
conversion over one project or a tree of projects, in the correct order.

The individual ``*_for_pipeline.py`` / ``migrate_*`` scripts each already handle
multiple files; what they don't do on their own is enforce the dependency order
between steps, let you pick *which* mice to (re)process, or confirm before
writing. This wrapper does all three: it discovers the VERSO projects under the
path, lets you allow-list / skip mice, prints the plan for confirmation, then
runs each step **per selected project** as a subprocess (so a failure is isolated
and reported).

Steps, in execution order
-------------------------
  1. thumbnails   flip_thumbnails_for_pipeline.py
  2. masks-flip   flip_masks_for_pipeline.py          (masks are already '-mask.png')
  3. hires        flip_hires_for_pipeline.py
  4. counts       flip_counts_for_pipeline.py
  5. quint        export_quint_for_pipeline.py       (QuickNII XML + VisuAlign JSON)
  6. info-xml     generate_info_xml.py --metadata CSV   (only if --metadata given)

(Two earlier steps were removed as obsolete for the current projects: 'scale'
[schema already migrated], 'masks-1bit' [masks are already 1-bit '-mask.png', so
there is no '-slice-mask.png' to convert/rename], and 'corners' [verso's own warp
anchors now match VisuAlign's, so injecting image-corner anchors would
double-correct and re-introduce the very misalignment it once fixed].)

Atlas-level one-offs (export_mirrored_atlas.py, export_annotation_volume_mat.py,
visualize_atlas_lr_mirror.py) are deliberately NOT part of this wrapper: they are
run once per machine against the atlas, not per project.

Picking which mice to run
-------------------------
A "mouse" is one VERSO project, identified by its project ``name`` (falling back
to its folder name). Use ``--mice`` to allow-list and ``--skip-mice`` to exclude;
both take comma-separated names. Without them, every discovered project runs.
Before anything is written the selected mice are printed and you are asked to
confirm (skip the prompt with ``-y``; a ``--dry-run`` never prompts).

Usage:
    # See which mice would be processed (no prompt, writes nothing)
    uv run python convert_for_pipeline.py PATH -r --dry-run

    # Run only two specific mice, with info.xml, confirming first
    uv run python convert_for_pipeline.py PATH -r --mice CC11B,CC4B --metadata metadata.csv

    # Run everything except already-done mice, no confirmation prompt
    uv run python convert_for_pipeline.py PATH -r --skip-mice CC7A,CC9B -y

    # Run a subset of steps; list steps and exit
    uv run python convert_for_pipeline.py PATH -r --only thumbnails,quint
    uv run python convert_for_pipeline.py --list-steps
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_GLOB = "*-verso.json"


@dataclass(frozen=True)
class Step:
    """One conversion step: a script plus the extra args it always wants."""

    key: str
    title: str
    script: str
    # Extra step-specific args appended after the path/--dry-run flags.
    extra: list[str] = field(default_factory=list)
    # If True, this step is only run when --metadata is supplied.
    needs_metadata: bool = False


# Execution order matters — see the module docstring.
STEPS: list[Step] = [
    Step("thumbnails", "Bake flips into thumbnail PNGs",
         "flip_thumbnails_for_pipeline.py"),
    Step("masks-flip", "Bake flips into masks",
         # Masks on disk are already named '<stem>-mask.png' (no rename needed),
         # so the flip step uses flip_masks_for_pipeline.py's default '-mask'.
         "flip_masks_for_pipeline.py"),
    Step("hires", "Bake flips into hiRes channel TIFFs",
         "flip_hires_for_pipeline.py"),
    Step("counts", "Reflect cell-count CSV coordinates",
         "flip_counts_for_pipeline.py"),
    Step("quint", "Export QuickNII XML + VisuAlign JSON",
         "export_quint_for_pipeline.py"),
    Step("info-xml", "Generate per-mouse info.xml",
         "generate_info_xml.py", needs_metadata=True),
]
STEP_BY_KEY = {s.key: s for s in STEPS}


@dataclass(frozen=True)
class MouseProject:
    """A discovered VERSO project: its mouse name, json file, and folder."""

    name: str
    json_path: Path

    @property
    def folder(self) -> Path:
        return self.json_path.parent


# --- step / mouse selection ---------------------------------------------------
def _parse_step_list(value: str) -> list[str]:
    keys = [k.strip() for k in value.split(",") if k.strip()]
    unknown = [k for k in keys if k not in STEP_BY_KEY]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown step(s) {unknown}; valid: {', '.join(STEP_BY_KEY)}"
        )
    return keys


def _parse_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def selected_steps(only: list[str] | None, skip: list[str] | None,
                   metadata: Path | None) -> list[Step]:
    """Resolve the ordered list of steps to run, applying --only/--skip/metadata."""
    keys = list(only) if only else [s.key for s in STEPS]
    if skip:
        keys = [k for k in keys if k not in skip]
    out: list[Step] = []
    for s in (STEP_BY_KEY[k] for k in keys):
        if s.needs_metadata and metadata is None:
            note = "explicitly selected but " if only and s.key in only else ""
            print(f"  (skipping step '{s.key}': {note}no --metadata CSV supplied)")
            continue
        out.append(s)
    return out


# --- project discovery --------------------------------------------------------
def _looks_like_project(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and isinstance(data.get("sections"), list)


def _project_name(json_path: Path) -> str:
    """Mouse name = project ``name`` field, falling back to the folder name."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        name = str(data.get("name") or "").strip()
    except (OSError, json.JSONDecodeError):
        name = ""
    return name or json_path.parent.name


def discover_projects(path: Path, recursive: bool) -> list[MouseProject]:
    """Find VERSO project json files under *path* and read each mouse name."""
    if path.is_file():
        jsons = [path]
    elif not path.is_dir():
        raise FileNotFoundError(f"No such file or directory: {path}")
    elif recursive:
        jsons = sorted(p for p in path.rglob(PROJECT_GLOB) if _looks_like_project(p))
    else:
        jsons = sorted(path.glob(PROJECT_GLOB))
        if not jsons:
            raise FileNotFoundError(
                f"No {PROJECT_GLOB} in {path} (use --recursive to scan subfolders)"
            )
    return [MouseProject(_project_name(j), j) for j in jsons]


def filter_mice(projects: list[MouseProject], only: list[str] | None,
                skip: list[str] | None) -> tuple[list[MouseProject], list[str]]:
    """Apply --mice / --skip-mice (case-insensitive). Returns (kept, not_found)."""
    def matches(p: MouseProject, names: list[str]) -> bool:
        lowered = {n.lower() for n in names}
        return p.name.lower() in lowered or p.folder.name.lower() in lowered

    kept = list(projects)
    not_found: list[str] = []
    if only:
        present = {p.name.lower() for p in projects} | {
            p.folder.name.lower() for p in projects
        }
        not_found = [n for n in only if n.lower() not in present]
        kept = [p for p in kept if matches(p, only)]
    if skip:
        kept = [p for p in kept if not matches(p, skip)]
    return kept, not_found


# --- execution ----------------------------------------------------------------
def build_argv(step: Step, project: MouseProject, *, dry_run: bool,
               metadata: Path | None, target: str | None) -> list[str]:
    """Build the subprocess argv for one step against one project."""
    argv = [sys.executable, str(HERE / step.script), str(project.json_path)]
    if dry_run:
        argv.append("--dry-run")
    argv += step.extra
    if step.key == "quint" and target:
        argv += ["--target", target]
    if step.needs_metadata and metadata is not None:
        argv += ["--metadata", str(metadata)]
    return argv


def run_step(step: Step, argv: list[str]) -> int:
    """Run one step as a subprocess, streaming its output. Returns its exit code."""
    print(f"\n  --- [{step.key}] {step.title}\n      $ {' '.join(argv)}")
    return subprocess.run(argv, cwd=HERE).returncode


def confirm(prompt: str) -> bool:
    """Interactive yes/no, defaulting to No. EOF/Ctrl-C counts as No."""
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path", type=Path, nargs="?", default=Path.cwd(),
        help="Project folder, path to a *-verso.json, or (with --recursive) a "
        "directory to scan. Defaults to the current directory.",
    )
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Discover every project under the path.")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="Pass --dry-run to every step (writes nothing, no prompt).")
    parser.add_argument("--mice", type=_parse_csv, default=None, metavar="NAMES",
                        help="Comma-separated mouse names to process (default: all "
                        "discovered). Matches the project name or its folder name.")
    parser.add_argument("--skip-mice", type=_parse_csv, default=None, metavar="NAMES",
                        help="Comma-separated mouse names to exclude.")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip the confirmation prompt.")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="Per-mouse metadata CSV; enables the 'info-xml' step.")
    parser.add_argument("--target", default=None,
                        help="Atlas 'target' override forwarded to the 'quint' step "
                        "(e.g. 'allen_mouse_25um.cutlas').")
    parser.add_argument("--only", type=_parse_step_list, default=None, metavar="STEPS",
                        help="Comma-separated step keys to run (default: all).")
    parser.add_argument("--skip", type=_parse_step_list, default=None, metavar="STEPS",
                        help="Comma-separated step keys to skip.")
    parser.add_argument("--keep-going", action="store_true",
                        help="Within a mouse, continue to later steps even if one "
                        "fails (default: stop that mouse at the first failure). "
                        "Remaining mice are always attempted.")
    parser.add_argument("--list-steps", action="store_true",
                        help="Print the ordered steps and exit.")
    args = parser.parse_args(argv)

    # Line-buffer our own stdout so the plan and the confirm prompt are flushed
    # before any subprocess (which writes to the inherited fd) prints over them.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    if args.list_steps:
        print("Conversion steps (in order):")
        for i, s in enumerate(STEPS, 1):
            tag = " [needs --metadata]" if s.needs_metadata else ""
            print(f"  {i}. {s.key:<11} {s.title}{tag}")
        return 0

    if args.metadata is not None and not args.metadata.is_file():
        print(f"ERROR: --metadata CSV not found: {args.metadata}", file=sys.stderr)
        return 2

    # --- discover + filter projects -----------------------------------------
    try:
        projects = discover_projects(args.path, args.recursive)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not projects:
        print("No VERSO projects found.")
        return 0

    mice, not_found = filter_mice(projects, args.mice, args.skip_mice)
    if not_found:
        print(f"WARNING: --mice names not found: {', '.join(not_found)}", file=sys.stderr)
    if not mice:
        print("No mice selected after --mice/--skip-mice filtering.")
        return 0

    steps = selected_steps(args.only, args.skip, args.metadata)
    if not steps:
        print("No steps selected.")
        return 0

    # --- plan + confirmation -------------------------------------------------
    print(f"\nPath: {args.path}{'  [DRY RUN]' if args.dry_run else ''}")
    print(f"Steps: {', '.join(s.key for s in steps)}")
    print(f"Mice ({len(mice)} of {len(projects)} discovered):")
    for m in mice:
        print(f"  - {m.name:<14} {m.folder}")

    if not args.dry_run and not args.yes:
        if not confirm(f"\nProceed with these {len(mice)} mouse(-s)?"):
            print("Aborted.")
            return 1

    # --- run -----------------------------------------------------------------
    # results[mouse_name] -> list of (step_key, returncode)
    results: dict[str, list[tuple[str, int]]] = {}
    for m in mice:
        print(f"\n{'=' * 72}\n# {m.name}  ({m.folder})\n{'=' * 72}")
        results[m.name] = []
        for step in steps:
            rc = run_step(step, build_argv(
                step, m, dry_run=args.dry_run,
                metadata=args.metadata, target=args.target,
            ))
            results[m.name].append((step.key, rc))
            if rc != 0 and not args.keep_going:
                print(f"  !! '{step.key}' failed (exit {rc}); skipping the rest of "
                      f"{m.name}. Use --keep-going to continue.", file=sys.stderr)
                break

    # --- summary -------------------------------------------------------------
    print(f"\n{'=' * 72}\nSummary")
    any_fail = False
    for name, step_results in results.items():
        failed = [k for k, rc in step_results if rc != 0]
        ran = {k for k, _ in step_results}
        not_run = [s.key for s in steps if s.key not in ran]
        any_fail = any_fail or bool(failed)
        status = "OK" if not failed else f"FAIL ({', '.join(failed)})"
        tail = f"; not run: {', '.join(not_run)}" if not_run else ""
        print(f"  {status:<22} {name}{tail}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
