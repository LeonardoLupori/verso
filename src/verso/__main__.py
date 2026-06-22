import argparse
import ctypes
import sys
from pathlib import Path

if sys.platform == "win32":
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("verso.app")


def main() -> None:
    parser = argparse.ArgumentParser(prog="verso")
    parser.add_argument(
        "--load-project",
        "--load_project",
        dest="load_project",
        type=Path,
        default=None,
        help="Path to a VERSO project.json to open on startup.",
    )
    args = parser.parse_args()

    from verso.gui.app import run

    run(project_path=args.load_project)


if __name__ == "__main__":
    main()
