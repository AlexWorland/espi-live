"""Command-line entry point: choose the Qt or Tk front end."""

import argparse
import importlib
import sys


def cli(argv=None):
    """Parse options and run the selected front end.

    Args: argv is an optional list of argument strings; None uses sys.argv.
    Returns: None; exits with the front end's status code.
    """
    parser = argparse.ArgumentParser(
        prog="espi_live",
        description="Live Blackfly Mono8 preview and ESPI measurement viewer.")
    parser.add_argument("--demo", action="store_true",
                        help="start synthetic video without detecting a camera")
    parser.add_argument("--ui", choices=("auto", "qt", "tk"), default="auto",
                        help="front end: qt (GPU display), tk (legacy), or auto "
                             "(qt when PySide6 is installed, else tk)")
    parser.add_argument("--check", action="store_true",
                        help="load the processing engine and the Qt front end "
                             "(no window, no camera), report versions, and exit")
    args = parser.parse_args(argv)
    if args.check:
        sys.exit(self_check())
    ui = args.ui
    if ui in ("auto", "qt"):
        try:
            from espi_live import qt_app
        except ImportError as exc:
            if ui == "qt":
                raise SystemExit(f"The Qt front end needs PySide6: {exc}")
            print(f"PySide6 unavailable ({exc}); using the legacy Tk front end.",
                  file=sys.stderr)
            ui = "tk"
        else:
            sys.exit(qt_app.run(force_demo=args.demo))
    from espi_live.app import main
    main(force_demo=args.demo)


def self_check():
    """Import every bundled module a run needs, without opening a window.

    Used by CI to catch a frozen binary that is missing Qt, OpenCV or NumPy
    pieces, which --help alone would not load.

    Args: None.
    Returns: 0 when everything imports, 1 otherwise (the error is printed).
    """
    try:
        import cv2
        import numpy
        import PySide6
        for module in ("espi_live.engine", "espi_live.qt_app", "espi_live.app"):
            importlib.import_module(module)
    except Exception as exc:
        print(f"Self-check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    try:
        from espi_live.camera import load_pyspin
        pyspin = f"PySpin {load_pyspin().__file__}"
    except ImportError:
        pyspin = "PySpin not bundled (demo only)"
    print(f"Self-check OK: Python {sys.version.split()[0]}, NumPy {numpy.__version__}, "
          f"OpenCV {cv2.__version__}, PySide6 {PySide6.__version__}; {pyspin}")
    return 0


if __name__ == "__main__":
    cli()
