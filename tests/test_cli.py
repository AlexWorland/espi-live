"""Camera-free checks for the command-line entry points.

No window, camera or network is ever used: the front ends and the updater are
replaced with mocks.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import contextlib
import io
import runpy
import sys
import types
import unittest
import warnings
from unittest import mock

import espi_live
from espi_live import camera, cli, updater


@contextlib.contextmanager
def fake_submodule(name, module):
    """Make `from espi_live import <name>` yield a replacement module.

    Args: name is the submodule name; module is the replacement object, or
        None to make the import raise ImportError.
    Returns: A context manager; the original state is restored on exit.
    """
    full = f"espi_live.{name}"
    missing = object()
    saved_attr = espi_live.__dict__.pop(name, missing)
    saved_mod = sys.modules.get(full, missing)
    sys.modules[full] = module
    if module is not None:
        setattr(espi_live, name, module)
    try:
        yield
    finally:
        if saved_mod is missing:
            sys.modules.pop(full, None)
        else:
            sys.modules[full] = saved_mod
        espi_live.__dict__.pop(name, None)
        if saved_attr is not missing:
            setattr(espi_live, name, saved_attr)


def run_cli(argv):
    """Run cli.cli and capture its exit code and output.

    Args: argv is the list of argument strings.
    Returns: (exit code or None, stdout text, stderr text).
    """
    out, err = io.StringIO(), io.StringIO()
    code = None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            cli.cli(argv)
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue(), err.getvalue()


class CliDispatchTests(unittest.TestCase):
    """Argument parsing and front-end selection."""

    def setUp(self):
        """Replace the updater's startup cleanup with a mock.

        Args: None.
        Returns: None.
        """
        patcher = mock.patch.object(updater, "cleanup_previous_update")
        self.cleanup = patcher.start()
        self.addCleanup(patcher.stop)

    def test_version_prints_build_and_exits_zero(self):
        """--version prints the build description and exits 0.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(updater, "describe_build",
                               return_value="build 7"):
            code, out, _ = run_cli(["--version"])
        self.assertEqual((code, out), (0, "build 7\n"))
        self.cleanup.assert_not_called()

    def test_check_exits_with_self_check_status(self):
        """--check exits with the value self_check returns.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(cli, "self_check", return_value=3) as check:
            code, _, _ = run_cli(["--check"])
        check.assert_called_once_with()
        self.assertEqual(code, 3)

    def test_update_exits_with_update_status(self):
        """--update exits with the value update_now returns.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(cli, "update_now", return_value=1) as update:
            code, _, _ = run_cli(["--update"])
        update.assert_called_once_with()
        self.assertEqual(code, 1)

    def test_default_runs_qt_with_update_check(self):
        """Without options the Qt front end runs and update checks are on.

        Args: None.
        Returns: None.
        """
        qt = types.SimpleNamespace(run=mock.Mock(return_value=5))
        with fake_submodule("qt_app", qt):
            code, _, _ = run_cli([])
        qt.run.assert_called_once_with(force_demo=False, check_updates=True,
                                       demo_cameras=1)
        self.cleanup.assert_called_once_with()
        self.assertEqual(code, 5)

    def test_demo_and_no_update_check_reach_qt(self):
        """--demo, --demo-cameras and --no-update-check reach the Qt front end.

        Args: None.
        Returns: None.
        """
        qt = types.SimpleNamespace(run=mock.Mock(return_value=0))
        with fake_submodule("qt_app", qt):
            code, _, _ = run_cli(["--ui", "qt", "--demo", "--demo-cameras", "3",
                                   "--no-update-check"])
        qt.run.assert_called_once_with(force_demo=True, check_updates=False,
                                       demo_cameras=3)
        self.assertEqual(code, 0)

    def test_auto_falls_back_to_tk_without_pyside(self):
        """--ui auto uses Tk and warns on stderr when Qt cannot import.

        Args: None.
        Returns: None.
        """
        tk_app = types.SimpleNamespace(main=mock.Mock())
        with fake_submodule("qt_app", None), fake_submodule("app", tk_app):
            code, _, err = run_cli(["--demo"])
        tk_app.main.assert_called_once_with(force_demo=True)
        self.assertIsNone(code)
        self.assertIn("PySide6 unavailable", err)

    def test_explicit_qt_without_pyside_exits_with_message(self):
        """--ui qt exits with an explanatory message when Qt is missing.

        Args: None.
        Returns: None.
        """
        with fake_submodule("qt_app", None):
            code, _, _ = run_cli(["--ui", "qt"])
        self.assertIn("The Qt front end needs PySide6", code)

    def test_tk_runs_legacy_front_end_directly(self):
        """--ui tk skips Qt entirely and runs the Tk main.

        Args: None.
        Returns: None.
        """
        tk_app = types.SimpleNamespace(main=mock.Mock())
        with fake_submodule("app", tk_app):
            code, _, err = run_cli(["--ui", "tk"])
        tk_app.main.assert_called_once_with(force_demo=False)
        self.assertIsNone(code)
        self.assertEqual(err, "")

    def test_invalid_ui_choice_is_rejected(self):
        """An unknown --ui value makes argparse exit with status 2.

        Args: None.
        Returns: None.
        """
        code, _, err = run_cli(["--ui", "gtk"])
        self.assertEqual(code, 2)
        self.assertIn("invalid choice", err)


class SelfCheckTests(unittest.TestCase):
    """The --check import probe."""

    def test_reports_ok_with_bundled_pyspin(self):
        """A passing check prints versions and the PySpin location.

        Args: None.
        Returns: None.
        """
        fake = types.SimpleNamespace(__file__="/sdk/PySpin.py")
        with mock.patch.object(cli.importlib, "import_module"), \
                mock.patch.object(camera, "load_pyspin", return_value=fake), \
                mock.patch.object(updater, "describe_build",
                                  return_value="build 7"):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                status = cli.self_check()
        self.assertEqual(status, 0)
        self.assertIn("Self-check OK", out.getvalue())
        self.assertIn("PySpin /sdk/PySpin.py", out.getvalue())
        self.assertIn("build 7", out.getvalue())

    def test_reports_demo_only_when_pyspin_missing(self):
        """A missing PySpin is not a failure; the check says demo only.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(cli.importlib, "import_module"), \
                mock.patch.object(camera, "load_pyspin",
                                  side_effect=ImportError("no PySpin")):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                status = cli.self_check()
        self.assertEqual(status, 0)
        self.assertIn("PySpin not bundled (demo only)", out.getvalue())

    def test_fails_when_a_module_cannot_import(self):
        """A broken bundled module returns 1 and prints the error.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(cli.importlib, "import_module",
                               side_effect=ValueError("broken")):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                status = cli.self_check()
        self.assertEqual(status, 1)
        self.assertIn("Self-check failed: ValueError: broken", err.getvalue())


class UpdateNowTests(unittest.TestCase):
    """The --update flow with a mocked updater."""

    def run_update(self, **patches):
        """Run update_now with updater functions replaced.

        Args: patches maps updater attribute names to mocks.
        Returns: (status, stdout text, stderr text).
        """
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(updater, "describe_build",
                               return_value="build 7"), \
                contextlib.ExitStack() as stack:
            for name, value in patches.items():
                stack.enter_context(mock.patch.object(updater, name, value))
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                status = cli.update_now()
        return status, out.getvalue(), err.getvalue()

    def test_up_to_date_returns_zero(self):
        """No newer release means success without downloading.

        Args: None.
        Returns: None.
        """
        download = mock.Mock()
        status, out, _ = self.run_update(
            check_for_update=mock.Mock(return_value=None),
            download_update=download)
        self.assertEqual(status, 0)
        self.assertIn("Already up to date.", out)
        download.assert_not_called()

    def test_installs_newer_build(self):
        """A newer release is downloaded, installed and announced.

        Args: None.
        Returns: None.
        """
        info = mock.Mock(size=50_000_000)
        info.label.return_value = "build 9"
        install = mock.Mock(return_value="/apps/espi")
        status, out, _ = self.run_update(
            check_for_update=mock.Mock(return_value=info),
            download_update=mock.Mock(return_value="/tmp/staged"),
            install_update=install)
        install.assert_called_once_with("/tmp/staged")
        self.assertEqual(status, 0)
        self.assertIn("Downloading build 9 (50 MB)", out)
        self.assertIn("Installed build 9 at /apps/espi", out)

    def test_update_error_returns_one(self):
        """An UpdateError is printed to stderr and returns 1.

        Args: None.
        Returns: None.
        """
        status, _, err = self.run_update(
            check_for_update=mock.Mock(side_effect=updater.UpdateError("offline")))
        self.assertEqual(status, 1)
        self.assertIn("Update failed: offline", err)


class ModuleEntryTests(unittest.TestCase):
    """python -m espi_live and running cli.py as a script."""

    def test_package_main_calls_cli(self):
        """Running the package as __main__ invokes cli.cli once.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(cli, "cli") as entry, \
                warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            runpy.run_module("espi_live", run_name="__main__")
        entry.assert_called_once_with()

    def test_package_main_does_nothing_when_imported(self):
        """Importing the module under another name does not start the app.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(cli, "cli") as entry, \
                warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            runpy.run_module("espi_live.__main__", run_name="not_main")
        entry.assert_not_called()

    def test_cli_script_entry_runs_cli(self):
        """Running cli.py as __main__ parses sys.argv and exits via --version.

        Args: None.
        Returns: None.
        """
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["espi_live", "--version"]), \
                warnings.catch_warnings(), contextlib.redirect_stdout(out):
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaises(SystemExit) as caught:
                runpy.run_module("espi_live.cli", run_name="__main__")
        self.assertEqual(caught.exception.code, 0)
        self.assertTrue(out.getvalue().strip())


if __name__ == "__main__":
    unittest.main()
