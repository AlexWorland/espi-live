"""Project-relative locations shared by every front end."""

import os
import sys
from datetime import datetime


def resolve_project_root():
    """Find the folder that holds the run-time ``output/`` directory.

    From source this file lives at <project>/src/espi_live/paths.py. In a
    PyInstaller build ``__file__`` points into the build's library folder
    (``_internal-<build>``, deleted after the next update), so outputs go
    beside the executable instead (or beside the .app bundle on macOS).

    Args: None.
    Returns: The absolute path of the output parent folder.
    """
    if getattr(sys, "frozen", False):
        folder = os.path.dirname(os.path.abspath(sys.executable))
        if folder.endswith(os.path.join(".app", "Contents", "MacOS")):
            folder = os.path.dirname(os.path.dirname(os.path.dirname(folder)))
        return folder
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))


PROJECT_ROOT = resolve_project_root()
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def timestamp():
    """Create a unique filename timestamp.

    Args: None.
    Returns: A local-time string with microsecond precision.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def output_folder(name):
    """Return (and create) a named folder under the project output directory.

    Args: name is the subfolder, such as "logs", "frames" or "recordings".
    Returns: The absolute folder path. Raises OSError if it cannot be created.
    """
    folder = os.path.join(OUTPUT_DIR, name)
    os.makedirs(folder, exist_ok=True)
    return folder
