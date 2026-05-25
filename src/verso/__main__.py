import ctypes
import sys

if sys.platform == "win32":
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("verso.app")


def main() -> None:
    from verso.gui.app import run
    run()


if __name__ == "__main__":
    main()
