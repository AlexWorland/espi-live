"""Checks for the Codex deep link and the guide written for Codex.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import os
import shutil
import sys
import tempfile
import unittest
import urllib.parse
from unittest import mock

from espi_live import codex, updater
from source_fixtures import make_copy, prebuilt


class LinkTests(unittest.TestCase):
    """deep_link and send_command."""

    def test_link_opens_a_new_chat_in_the_folder(self):
        """The absolute folder path is percent-encoded into the link.

        Args: None.
        Returns: None.
        """
        folder = os.path.join(tempfile.gettempdir(), "My Programs", "espi-live-source")
        link = codex.deep_link(folder)
        prefix = "codex://threads/new?path="
        self.assertTrue(link.startswith(prefix))
        self.assertNotIn(" ", link)
        self.assertNotIn("/", link[len(prefix):])
        self.assertEqual(urllib.parse.unquote(link[len(prefix):]), os.path.abspath(folder))

    def test_send_command_suits_the_shell(self):
        """PowerShell needs the call operator; POSIX shells need quoting.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(sys, "executable", "/Apps/ESPI Live.app/Contents/MacOS/ESPI Live"):
            self.assertEqual(codex.send_command("darwin"),
                             "'/Apps/ESPI Live.app/Contents/MacOS/ESPI Live' --send-changes")
            self.assertTrue(codex.send_command("win32").startswith('& "'))
            self.assertTrue(codex.send_command("win32").endswith('ESPI Live" --send-changes'))
            self.assertEqual(codex.send_command(), codex.send_command(sys.platform))


class GuideTests(unittest.TestCase):
    """guide_text and write_guide."""

    def setUp(self):
        """Create a temporary program folder.

        Args: None.
        Returns: None.
        """
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_guide_names_the_build_the_command_and_the_rules(self):
        """The guide carries the build, the send command and the approval step.

        Args: None.
        Returns: None.
        """
        text = codex.guide_text({"commit": "0123456789abcdef", "build": 42})
        self.assertIn("build 42, commit `0123456`", text)
        self.assertIn(codex.send_command(), text)
        self.assertIn("Are you happy with these changes?", text)
        self.assertIn("never use git or gh", text)
        self.assertIn("CHANGE_REQUEST.md", text)
        self.assertNotIn("{", text.replace("{build}", ""))
        self.assertIn("a prebuilt copy, commit", codex.guide_text({"commit": "abc"}))

    def test_guide_goes_only_into_a_downloaded_copy(self):
        """A downloaded copy gets AGENTS.override.md; a checkout keeps AGENTS.md.

        Args: None.
        Returns: None.
        """
        folder = make_copy(self.root, {"AGENTS.md": b"# Developer rules\n"})
        with mock.patch.object(updater, "install_target", return_value=None):
            self.assertIsNone(codex.write_guide(folder))
        with prebuilt(self.root):
            self.assertIsNone(codex.write_guide(self.root))
            path = codex.write_guide(folder)
        self.assertEqual(path, os.path.join(folder, "AGENTS.override.md"))
        with open(path, encoding="utf-8") as handle:
            self.assertIn("You are helping a user of ESPI live", handle.read())


if __name__ == "__main__":
    unittest.main()
