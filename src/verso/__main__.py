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
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (overrides the VERSO_LOG_LEVEL env var; default INFO).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Shortcut for --log-level DEBUG.",
    )
    args = parser.parse_args()

    from verso.gui.app import run

    log_level = "DEBUG" if args.verbose else args.log_level
    run(project_path=args.load_project, log_level=log_level)


if __name__ == "__main__":
    main()
