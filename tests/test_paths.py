"""Camera-free checks for project path resolution.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import importlib
import os
import sys
import tempfile
import unittest
from unittest import mock

from espi_live import paths


class ResolveRootTests(unittest.TestCase):
    """The output parent folder in source and frozen runs."""

    def test_source_run_uses_project_root(self):
        """From source, the root is the folder above src/.

        Args: None.
        Returns: None.
        """
        self.assertTrue(os.path.isdir(os.path.join(paths.resolve_project_root(), "src")))

    def test_frozen_run_uses_executable_folder(self):
        """A frozen build keeps outputs beside the executable.

        Args: None.
        Returns: None.
        """
        exe = os.path.join(os.sep, "apps", "espi", "espi-live")
        with mock.patch.object(sys, "frozen", True, create=True), \
                mock.patch.object(sys, "executable", exe):
            self.assertEqual(paths.resolve_project_root(),
                             os.path.dirname(os.path.abspath(exe)))

    def test_frozen_app_bundle_uses_folder_beside_bundle(self):
        """A macOS .app build keeps outputs beside the bundle.

        Args: None.
        Returns: None.
        """
        exe = os.path.join(os.sep, "apps", "ESPI.app", "Contents", "MacOS", "espi")
        with mock.patch.object(sys, "frozen", True, create=True), \
                mock.patch.object(sys, "executable", exe):
            self.assertEqual(paths.resolve_project_root(),
                             os.path.abspath(os.path.join(os.sep, "apps")))

    def test_reload_with_frozen_flag_recomputes_constants(self):
        """Reloading the module while frozen moves OUTPUT_DIR, and restoring works.

        Args: None.
        Returns: None.
        """
        original = paths.OUTPUT_DIR
        exe = os.path.join(os.sep, "apps", "espi", "espi-live")
        try:
            with mock.patch.object(sys, "frozen", True, create=True), \
                    mock.patch.object(sys, "executable", exe):
                importlib.reload(paths)
                self.assertEqual(
                    paths.OUTPUT_DIR,
                    os.path.join(os.path.abspath(os.path.join(os.sep, "apps", "espi")),
                                 "output"))
        finally:
            importlib.reload(paths)
        self.assertEqual(paths.OUTPUT_DIR, original)


class FolderTests(unittest.TestCase):
    """Timestamps and output folder creation."""

    def test_timestamp_has_microsecond_format(self):
        """Timestamps look like YYYYmmdd_HHMMSS_ffffff.

        Args: None.
        Returns: None.
        """
        self.assertRegex(paths.timestamp(), r"^\d{8}_\d{6}_\d{6}$")

    def test_output_folder_is_created_under_output_dir(self):
        """output_folder creates and returns the named subfolder.

        Args: None.
        Returns: None.
        """
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(paths, "OUTPUT_DIR", tmp):
            folder = paths.output_folder("logs")
            self.assertEqual(folder, os.path.join(tmp, "logs"))
            self.assertTrue(os.path.isdir(folder))
            self.assertEqual(paths.output_folder("logs"), folder)


if __name__ == "__main__":
    unittest.main()
