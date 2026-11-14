"""Checks for finding and starting Codex and for the guide written for it.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from espi_live import codex, updater
from source_fixtures import make_copy, prebuilt


class FindTests(unittest.TestCase):
    """find_cli."""

    def test_command_on_the_path_is_found(self):
        """The command found on PATH is returned as it is.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(codex.shutil, "which", return_value="/usr/bin/codex") as which:
            self.assertEqual(codex.find_cli("linux"), "/usr/bin/codex")
        which.assert_called_once_with("codex")

    def test_usual_install_folders_are_searched_off_windows(self):
        """A program started from the desktop still finds a Homebrew or npm install.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(codex.shutil, "which",
                               side_effect=[None, "/opt/homebrew/bin/codex"]) as which:
            self.assertEqual(codex.find_cli("darwin"), "/opt/homebrew/bin/codex")
        searched = which.call_args.kwargs["path"].split(os.pathsep)
        self.assertIn("/opt/homebrew/bin", searched)
        self.assertIn(os.path.expanduser("~/.npm-global/bin"), searched)

    def test_missing_command_gives_none(self):
        """No command anywhere means Codex is not installed; Windows has no extra search.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(codex.shutil, "which", return_value=None) as which:
            self.assertIsNone(codex.find_cli("linux"))
            self.assertEqual(which.call_count, 2)
            which.reset_mock()
            self.assertIsNone(codex.find_cli("win32"))
            self.assertEqual(which.call_count, 1)


class TerminalTests(unittest.TestCase):
    """terminal_command and open_cli."""

    FOLDER = os.path.join(tempfile.gettempdir(), "My Programs", "espi-live-source")

    def test_windows_opens_a_console_that_stays(self):
        """A new console runs cmd /k so an early Codex error can be read.

        Args: None.
        Returns: None.
        """
        command, options = codex.terminal_command(r"C:\npm\codex.cmd", self.FOLDER, "win32")
        self.assertEqual(command, ["cmd.exe", "/k", r"C:\npm\codex.cmd"])
        self.assertEqual(options["cwd"], self.FOLDER)
        self.assertTrue(options["creationflags"])

    def test_windows_console_flag_comes_from_subprocess_when_it_exists(self):
        """On Windows the real CREATE_NEW_CONSOLE value is used.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(codex.subprocess, "CREATE_NEW_CONSOLE", 0x10, create=True):
            _, options = codex.terminal_command("codex.cmd", self.FOLDER, "win32")
        self.assertEqual(options["creationflags"], 0x10)

    def test_macos_runs_it_in_terminal_app(self):
        """Terminal.app is told to cd to the folder and run Codex, quoting both.

        Args: None.
        Returns: None.
        """
        command, options = codex.terminal_command("/opt/homebrew/bin/codex",
                                                  "/Users/Al's \"Lab\"/source", "darwin")
        self.assertEqual(command[:2], ["osascript", "-e"])
        script = command[2]
        self.assertTrue(script.startswith('tell application "Terminal" to do script "'))
        self.assertIn("/opt/homebrew/bin/codex", script)
        self.assertNotIn('"Lab"', script)  # the quotes in the folder name are escaped
        self.assertIn("activate", command[-1])
        self.assertEqual(options, {})

    def test_apple_script_quote(self):
        """Backslashes and double quotes are escaped.

        Args: None.
        Returns: None.
        """
        self.assertEqual(codex.apple_script_quote('a"b\\c'), '"a\\"b\\\\c"')

    def test_linux_uses_the_first_terminal_it_finds(self):
        """Each terminal gets its own flag, and the folder is entered by a shell.

        Args: None.
        Returns: None.
        """
        found = {"gnome-terminal": "/usr/bin/gnome-terminal", "xterm": "/usr/bin/xterm"}
        command, options = codex.terminal_command("/usr/bin/codex", self.FOLDER, "linux",
                                                  which=found.get)
        self.assertEqual(command[:2], ["/usr/bin/gnome-terminal", "--"])
        self.assertEqual(command[-2:], [self.FOLDER, "/usr/bin/codex"])
        self.assertEqual(options, {})
        command, _ = codex.terminal_command("/usr/bin/codex", self.FOLDER, "linux",
                                            which={"xterm": "/usr/bin/xterm"}.get)
        self.assertEqual(command[:2], ["/usr/bin/xterm", "-e"])

    def test_linux_without_a_terminal_raises(self):
        """Having no terminal program is an error the window can report.

        Args: None.
        Returns: None.
        """
        with self.assertRaises(OSError):
            codex.terminal_command("/usr/bin/codex", self.FOLDER, "linux", which=lambda name: None)

    def test_open_cli_starts_the_terminal(self):
        """The terminal is started and the command's path returned.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(codex, "find_cli", return_value="/bin/codex"), \
                mock.patch.object(codex, "terminal_command",
                                  return_value=(["term"], {"cwd": "x"})) as build, \
                mock.patch.object(codex.subprocess, "Popen") as popen:
            self.assertEqual(codex.open_cli(self.FOLDER), "/bin/codex")
        build.assert_called_once_with("/bin/codex", self.FOLDER)
        popen.assert_called_once_with(["term"], cwd="x")

    def test_open_cli_without_codex_starts_nothing(self):
        """A missing Codex returns None and opens no window.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(codex, "find_cli", return_value=None), \
                mock.patch.object(codex.subprocess, "Popen") as popen:
            self.assertIsNone(codex.open_cli(self.FOLDER))
        popen.assert_not_called()


class LinkTests(unittest.TestCase):
    """send_command."""

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
