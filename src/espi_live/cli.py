"""Command-line entry point for the Qt front end."""

import argparse
import importlib
import sys

from espi_live import updater
from espi_live.logs import LEVELS


def cli(argv=None):
    """Parse options and run the Qt front end.

    Args: argv is an optional list of argument strings; None uses sys.argv.
    Returns: None; exits with the front end's status code, or with a message
        when PySide6 is missing.
    """
    parser = argparse.ArgumentParser(
        prog="espi_live",
        description="Live Blackfly Mono8 preview and ESPI measurement viewer.")
    parser.add_argument("--demo", action="store_true",
                        help="start synthetic video without detecting a camera")
    parser.add_argument("--demo-cameras", type=int, default=1, metavar="N",
                        help="number of synthetic cameras in demo mode "
                             "(default 1)")
    parser.add_argument("--check", action="store_true",
                        help="load the processing engine and the Qt front end "
                             "(no window, no camera), report versions, and exit")
    parser.add_argument("--version", action="store_true",
                        help="print the build number and exit")
    parser.add_argument("--update", action="store_true",
                        help="install the newest prebuilt binary from GitHub and exit")
    parser.add_argument("--no-update-check", action="store_true",
                        help="don't check GitHub for a newer build at startup")
    parser.add_argument("--log-level", choices=sorted(LEVELS), default=None,
                        help="log detail (default: ESPI_LOG_LEVEL, else info); "
                             "debug adds camera node writes, settings changes "
                             "and frame timing")
    parser.add_argument("--no-error-reports", action="store_true",
                        help="don't send unexpected errors to GitHub as issues "
                             "(also ESPI_ERROR_REPORTS=0)")
    args = parser.parse_args(argv)
    if args.version:
        print(updater.describe_build())
        sys.exit(0)
    if args.check:
        sys.exit(self_check())
    if args.update:
        sys.exit(update_now())
    try:
        from espi_live import qt_app
    except ImportError as exc:
        raise SystemExit(f"ESPI live needs PySide6 (pip install -r requirements.txt): {exc}")
    sys.exit(qt_app.run(force_demo=args.demo,
                        check_updates=not args.no_update_check,
                        demo_cameras=args.demo_cameras,
                        log_level=args.log_level,
                        send_reports=not args.no_error_reports))


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
        for module in ("espi_live.engine", "espi_live.qt_app", "espi_live.updater",
                       "espi_live.error_reports"):
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
          f"OpenCV {cv2.__version__}, PySide6 {PySide6.__version__}; {pyspin}; "
          f"{updater.describe_build()}")
    return 0


def update_now():
    """Check for, download and install a newer build without opening a window.

    The new build takes effect the next time the program starts.

    Args: None.
    Returns: 0 when up to date or updated, 1 on failure (the error is printed).
    """
    print(updater.describe_build())
    try:
        info = updater.check_for_update()
        if info is None:
            print("Already up to date.")
            return 0
        print(f"Downloading {info.label()} ({info.size / 1e6:.0f} MB)…")
        path = updater.install_update(updater.download_update(info))
    except updater.UpdateError as exc:
        print(f"Update failed: {exc}", file=sys.stderr)
        return 1
    print(f"Installed {info.label()} at {path}; start the program again to use it.")
    return 0


if __name__ == "__main__":
    cli()
