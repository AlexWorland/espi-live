"""Real Qt/worker/HTTP/filesystem update flows; only loopback HTTP is allowed.

No camera, GitHub, real settings, output folder or blocking dialog is used.
Binary launch and native file locks are exercised separately in tests/e2e/.
"""

import io
import os
import tarfile
import tempfile
import time
import unittest
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from espi_live import cli, paths, qt_app, updater
from espi_live.engine import Engine
from test_qt_app import APP, FakeLog, FakeSettings, demo_window
from update_fixtures import ReleaseServer


class UpdateIntegrationTests(unittest.TestCase):
    """Join the real updater pieces that the unit tests isolate."""

    def setUp(self):
        """Prepare a disposable installation and a local newer release.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.target = self.root / "espi-live"
        self.target.write_bytes(b"old executable")
        self.old_library = self.root / "_internal-7"
        self.old_library.mkdir()
        (self.old_library / "library").write_bytes(b"old library")
        self.kept = {"output/recordings/measurement.avi": b"recording",
                     "update-token.txt": b"test-token", "feedback-token.txt": b"test-feedback"}
        for name, data in self.kept.items():
            file = self.root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(data)
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
            for name, data in (("espi-live", b"new executable"),
                               ("_internal-8/library", b"new library")):
                entry = tarfile.TarInfo("espi-live/" + name)
                entry.size = len(data)
                bundle.addfile(entry, io.BytesIO(data))
        self.server = ReleaseServer(archive.getvalue(), "espi-live-test.tar.gz")
        self.server.__enter__()
        self.addCleanup(self.server.__exit__)
        real_open = urllib.request.urlopen

        def loopback_only(request, *args, **kwargs):
            """Reject unexpected external traffic while keeping real HTTP.

            Args: request, args and kwargs are urlopen arguments.
            Returns: The real local HTTP response.
            """
            url = request.full_url if hasattr(request, "full_url") else request
            if not url.startswith(self.server.url + "/"):
                raise AssertionError(f"External request in update test: {url}")
            return real_open(request, *args, **kwargs)

        FakeSettings.store = {}
        self.questions = []
        self.warnings = []
        self.information = []
        self.answers = [QMessageBox.Yes, QMessageBox.No]
        for patch in (
                mock.patch.object(updater, "API_ROOT", self.server.url),
                mock.patch.object(updater, "_build_info", SimpleNamespace(
                    BUILD=7, COMMIT="oldcommit", REPO="test/espi-live",
                    ASSET=self.server.asset_name, TOKEN="")),
                mock.patch.object(updater, "install_target", return_value=str(self.target)),
                mock.patch.object(updater, "running_contents", return_value=str(self.old_library)),
                mock.patch.object(urllib.request, "urlopen", side_effect=loopback_only),
                mock.patch.dict(os.environ, {"ESPI_UPDATE_TOKEN": ""}),
                mock.patch.object(paths, "OUTPUT_DIR", str(self.root / "output")),
                mock.patch.object(qt_app, "QSettings", FakeSettings),
                mock.patch.object(qt_app.source_code.SourceWorker, "start"),
                mock.patch.object(QMessageBox, "question", side_effect=self.answer),
                mock.patch.object(QMessageBox, "warning", side_effect=self.warn),
                mock.patch.object(QMessageBox, "information", side_effect=self.inform)):
            patch.start()
            self.addCleanup(patch.stop)
        self.window = None
        self.log = FakeLog()
        self.addCleanup(self.close_window)

    def answer(self, parent, title, text, *args):
        """Answer update dialogs without blocking the event loop.

        Args: parent, title, text and args are QMessageBox arguments.
        Returns: The next scripted answer.
        """
        self.questions.append(title)
        return self.answers.pop(0)

    def warn(self, parent, title, text, *args):
        """Collect visible failure messages.

        Args: parent, title, text and args are QMessageBox arguments.
        Returns: None.
        """
        self.warnings.append(text)

    def inform(self, parent, title, text, *args):
        """Collect visible check results.

        Args: parent, title, text and args are QMessageBox arguments.
        Returns: None.
        """
        self.information.append(text)

    def open_window(self, automatic=False):
        """Start the real demo engine, widgets, timers and updater worker.

        Args: automatic enables startup and periodic checks.
        Returns: None.
        """
        self.engine = Engine(self.log)
        self.engine.use_demo()
        self.window = demo_window(self.engine, self.log, check_updates=automatic)
        self.window.show()
        self.engine.start()
        self.until(lambda: self.window.drawn_frames > 0)

    def until(self, predicate):
        """Run Qt events until an observable result arrives, with a deadline.

        Args: predicate returns True when the expected state has arrived.
        Returns: None; fails after five seconds.
        """
        deadline = time.monotonic() + 5
        while not predicate() and time.monotonic() < deadline:
            APP.processEvents()
            QTest.qWait(5)
        self.assertTrue(predicate(), f"Timed out; log={self.log.lines}; warnings={self.warnings}")

    def close_window(self):
        """Stop real workers before restoring patches or removing their files.

        Args: None.
        Returns: None.
        """
        self.server.allow_download.set()
        if self.window is not None:
            self.window.close()
            self.until(lambda: not self.window.updates.busy)
            for timer in (self.window.frame_timer, self.window.status_timer,
                          self.window.tip_timer, self.window.update_timer):
                timer.stop()
            self.window.deleteLater()
            APP.processEvents()

    def assert_installed(self):
        """Check replacement, backup and preservation using actual disk contents.

        Args: None.
        Returns: None.
        """
        self.assertEqual(self.target.read_bytes(), b"new executable")
        self.assertEqual(Path(str(self.target) + ".old").read_bytes(), b"old executable")
        self.assertEqual((self.root / "_internal-8/library").read_bytes(), b"new library")
        for name, data in self.kept.items():
            self.assertEqual((self.root / name).read_bytes(), data)

    def test_startup_discovery_download_and_install_on_exit(self):
        """Automatic discovery downloads only after consent and installs on exit.

        Args: None.
        Returns: None.
        """
        self.open_window(automatic=True)
        self.until(lambda: self.window.update_info is not None)
        self.assertEqual(self.questions, [])
        self.assertFalse(self.server.download_started.is_set())
        QTest.mouseClick(self.window.update_button, Qt.LeftButton)
        self.until(lambda: self.window.pending_update is not None)
        self.assertEqual(self.target.read_bytes(), b"old executable")
        self.assertFalse(self.window.closing)
        self.assertEqual(self.questions, ["Update available", "Update ready"])
        self.window.close()
        qt_app.finish_update(self.window, self.log)
        self.assert_installed()
        self.assertFalse(list(self.root.glob(".espi-update-*")))
        with mock.patch.object(updater, "running_contents",
                               return_value=str(self.root / "_internal-8")):
            self.assertTrue(updater.cleanup_previous_update())
        self.assertFalse(self.old_library.exists())
        self.assertFalse(Path(str(self.target) + ".old").exists())
        self.assertTrue((self.root / "_internal-8/library").exists())
        for name, data in self.kept.items():
            self.assertEqual((self.root / name).read_bytes(), data)
        self.assertIn(("/asset", "Bearer test-token"), self.server.requests)
        self.assertIn(("/download", None), self.server.requests)

    def test_manual_check_downloads_while_frames_continue(self):
        """The manual button uses the real worker without freezing acquisition.

        Args: None.
        Returns: None.
        """
        self.server.allow_download.clear()
        self.open_window()
        QTest.mouseClick(self.window.check_button, Qt.LeftButton)
        self.until(self.server.download_started.is_set)
        frames = self.window.drawn_frames
        self.until(lambda: self.window.drawn_frames > frames + 2)
        self.assertTrue(self.window.updates.busy)
        self.assertEqual(self.target.read_bytes(), b"old executable")
        self.server.allow_download.set()
        self.until(lambda: self.window.pending_update is not None)
        self.window.close()
        qt_app.finish_update(self.window, self.log)
        self.assert_installed()

    def test_periodic_timer_discovers_a_release_published_after_startup(self):
        """A real timer later discovers a new build without prompting to install.

        Args: None.
        Returns: None.
        """
        self.server.build = 7
        self.open_window(automatic=True)
        self.until(lambda: any("Up to date" in line for line in self.log.lines))
        self.server.build = 8
        self.window.update_timer.start(10)
        self.until(lambda: self.window.update_info is not None)
        self.assertTrue(self.window.update_button.isVisible())
        self.assertFalse(self.server.download_started.is_set())
        self.assertEqual(self.questions, [])

    def test_declining_download_leaves_installation_untouched(self):
        """Declining the manual offer never downloads or stages a replacement.

        Args: None.
        Returns: None.
        """
        self.answers = [QMessageBox.No]
        self.open_window()
        QTest.mouseClick(self.window.check_button, Qt.LeftButton)
        self.until(lambda: bool(self.questions))
        self.assertFalse(self.server.download_started.is_set())
        self.assertIsNone(self.window.pending_update)
        self.assertEqual(self.target.read_bytes(), b"old executable")

    def test_failed_download_can_be_retried_successfully(self):
        """Corruption or interruption never installs; a retry can still succeed.

        Args: None.
        Returns: None.
        """
        self.open_window()
        for failure in ("corrupt", "interrupted"):
            with self.subTest(failure=failure):
                self.server.failure = failure
                self.answers = [QMessageBox.Yes]
                QTest.mouseClick(self.window.check_button if self.window.update_info is None
                                 else self.window.update_button, Qt.LeftButton)
                self.until(lambda: len(self.warnings) > 0)
                self.assertIn("SHA-256", self.warnings.pop())
                self.assertIsNone(self.window.pending_update)
                self.assertTrue(self.window.update_button.isEnabled())
                self.assertEqual(self.target.read_bytes(), b"old executable")
                self.assertFalse((self.root / "_internal-8").exists())
                self.until(lambda: not self.window.updates.busy)
        self.server.failure = None
        self.answers = [QMessageBox.Yes, QMessageBox.No]
        QTest.mouseClick(self.window.update_button, Qt.LeftButton)
        self.until(lambda: self.window.pending_update is not None)
        self.window.close()
        qt_app.finish_update(self.window, self.log)
        self.assert_installed()

    def test_failed_automatic_check_is_silent_but_manual_check_reports_it(self):
        """HTTP failures remain nonfatal and manual failures are visible.

        Args: None.
        Returns: None.
        """
        self.server.failure = "http"
        self.open_window(automatic=True)
        self.until(lambda: any("Update check failed" in line for line in self.log.lines))
        self.assertEqual(self.warnings, [])
        self.until(lambda: self.window.check_button.isEnabled())
        QTest.mouseClick(self.window.check_button, Qt.LeftButton)
        self.until(lambda: bool(self.warnings))
        self.assertIn("503", self.warnings[0])
        self.assertFalse(self.window.closing)

    def test_no_newer_build_reports_up_to_date_without_downloading(self):
        """Equal and older builds cannot replace a newer local installation.

        Args: None.
        Returns: None.
        """
        self.open_window()
        for build in (7, 6):
            self.server.build = build
            self.information.clear()
            QTest.mouseClick(self.window.check_button, Qt.LeftButton)
            self.until(lambda: bool(self.information))
            self.assertIn("up to date", self.information[0])
            self.until(lambda: self.window.check_button.isEnabled())
        self.assertFalse(self.server.download_started.is_set())
        self.assertEqual(self.target.read_bytes(), b"old executable")

    def test_cli_update_installs_from_the_local_release(self):
        """The manual CLI path checks, downloads and installs real archive bytes.

        Args: None.
        Returns: None.
        """
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli.update_now(), 0)
        self.assert_installed()

    def test_shutdown_timeout_preserves_the_running_installation(self):
        """A worker that cannot stop must prevent installation as well as relaunch.

        Args: None.
        Returns: None.
        """
        self.open_window()
        QTest.mouseClick(self.window.check_button, Qt.LeftButton)
        self.until(lambda: self.window.pending_update is not None)
        self.window.close()
        for restart in (False, True):
            with self.subTest(restart=restart):
                self.window.relaunch_after_exit = restart
                with mock.patch.object(self.window, "wait_for_engines", return_value=False):
                    qt_app.finish_update(self.window, self.log)
                self.assertEqual(self.target.read_bytes(), b"old executable")
                self.assertTrue(Path(self.window.pending_update).exists())
                self.assertFalse(Path(str(self.target) + ".old").exists())

    def test_cli_failed_download_returns_failure_without_replacing_the_program(self):
        """A bad archive produces a nonzero CLI result and preserves the old build.

        Args: None.
        Returns: None.
        """
        self.server.failure = "corrupt"
        error = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(error):
            self.assertEqual(cli.update_now(), 1)
        self.assertIn("SHA-256", error.getvalue())
        self.assertEqual(self.target.read_bytes(), b"old executable")
