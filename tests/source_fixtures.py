"""Shared helpers for the source-copy tests: GitHub-style source zips, a
downloaded copy in a temporary folder and a pretend prebuilt program.

Nothing here touches the network or the real program folder.
"""

import contextlib
import io
import os
import types
import zipfile
from unittest import mock

from espi_live import paths, source_code, updater

COMMIT = "0123456789abcdef0123456789abcdef01234567"
OTHER_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"
TOP = "owner-repo-0123456"
FILES = {"README.md": b"# ESPI live\n\nA viewer.\n",
         "src/espi_live/engine.py": b'"""Engine."""\n\nRATE = 30\n',
         "scripts/windows/start.bat": b"@echo off\r\necho start\r\n",
         "docs/logo.png": b"\x89PNG\r\n\x1a\n\0\0"}


class FakeResponse(io.BytesIO):
    """A urlopen result usable as a context manager."""

    def __enter__(self):
        """Enter the context.

        Args: None.
        Returns: This response.
        """
        return self

    def __exit__(self, *exc):
        """Leave the context.

        Args: exc is the exception triple.
        Returns: False, so exceptions propagate.
        """
        return False


def zip_bytes(files=None, commit=COMMIT, top=TOP, extra=()):
    """Build a zip laid out like GitHub's /zipball/<commit>.

    Args: files maps project paths to bytes (default FILES); commit becomes
        the zip comment ("" for none); top is the top-level folder; extra is
        a list of (ZipInfo or name, bytes) added as they are.
    Returns: The zip file's bytes.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr(f"{top}/", b"")
        for name, data in (FILES if files is None else files).items():
            bundle.writestr(f"{top}/{name}", data)
        for info, data in extra:
            bundle.writestr(info, data)
        bundle.comment = commit.encode("ascii")
    return buffer.getvalue()


def make_copy(root, files=None, commit=COMMIT, build=7):
    """Create a downloaded source copy without the network.

    Args: root is the folder that stands for the program's folder; files
        maps project paths to bytes (default FILES); commit and build are
        recorded as the copy's source.
    Returns: The copy's folder.
    """
    folder = os.path.join(root, source_code.SOURCE_DIR)
    os.makedirs(os.path.join(folder, source_code.STATE_DIR))
    archive = source_code.bookkeeping(folder, source_code.BASELINE_NAME)
    with open(archive, "wb") as handle:
        handle.write(zip_bytes(files, commit))
    source_code.extract(archive, folder, commit)
    source_code.write_json(source_code.bookkeeping(folder, source_code.STATE_NAME),
                           {"commit": commit, "repo": "owner/repo", "build": build})
    return folder


def write(folder, relative, data):
    """Write one file inside a copy, creating its folders.

    Args: folder is the copy; relative is the path with "/"; data is bytes
        or text.
    Returns: The absolute path.
    """
    path = os.path.join(folder, *relative.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data if isinstance(data, bytes) else data.encode("utf-8"))
    return path


@contextlib.contextmanager
def prebuilt(root, commit=COMMIT, repo="owner/repo", build=7, feedback_token=""):
    """Make the program look like a prebuilt copy installed in root.

    Args: root is the program's folder; commit, repo and build are its
        build facts; feedback_token is the baked-in Issues token.
    Returns: A context manager.
    """
    info = types.SimpleNamespace(BUILD=build, COMMIT=commit, REPO=repo, ASSET="a.zip",
                                 TOKEN="", FEEDBACK_TOKEN=feedback_token)
    with mock.patch.object(updater, "_build_info", info), \
            mock.patch.object(updater, "install_target",
                              return_value=os.path.join(root, "espi-live")), \
            mock.patch.object(paths, "PROJECT_ROOT", root), \
            mock.patch.dict(os.environ, {"ESPI_UPDATE_TOKEN": "", "ESPI_FEEDBACK_TOKEN": "",
                                         "ESPI_FEEDBACK_REPO": ""}):
        yield
