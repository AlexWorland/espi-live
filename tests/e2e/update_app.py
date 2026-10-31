"""Test-only launcher: real app and updater, scripted input and local releases.

This entry point is frozen only by test_update_binaries.py, never shipped.
Build metadata comes from a bundled fixture; production code is unchanged.
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from espi_live import cli, qt_app, updater


def write_event(name, **fields):
    """Save process evidence separately for each build.

    Args: name is the event; fields are its JSON-serializable observations.
    Returns: None.
    """
    path = Path(os.environ["ESPI_TEST_RUN"]) / f"events-{updater.current_build()}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": name, "pid": os.getpid(), **fields}) + "\n")


class DrivenWindow(qt_app.MainWindow):
    """Run the normal window with bounded, automatic test input."""

    instance = None

    def __init__(self, *args, **kwargs):
        """Create the real widgets and start the test input timer.

        Args: args and kwargs are MainWindow arguments.
        Returns: None.
        """
        super().__init__(*args, **kwargs)
        DrivenWindow.instance = self
        self.test_started = time.monotonic()
        self.test_phase = "waiting"
        self.test_timer = QTimer(self)
        self.test_timer.timeout.connect(self.drive)
        self.test_timer.start(20)
        write_event("started", args=sys.argv[1:], library=updater.running_contents())

    def drive(self):
        """Click actual update buttons and observe the replacement's live frames.

        Args: None.
        Returns: None.
        """
        try:
            if time.monotonic() - self.test_started > 25:
                raise AssertionError(f"Update timed out in phase {self.test_phase}")
            if self.closing or self.drawn_frames == 0:
                return
            for channel, checkbox in zip(self.channels, self.camera_checks):
                if not channel.active:
                    checkbox.click()
                    return
                if channel.engine.latest_packet() is None:
                    return
            if updater.current_build() == 8:
                target = Path(updater.install_target())
                old = Path(str(target) + ".old")
                old_library = target.parent / "_internal-7"
                if old.exists() or old_library.exists():
                    return  # Windows cleanup retries until the previous process exits.
                write_event("healthy", frames=self.drawn_frames)
                self.test_timer.stop()
                self.close()
                return
            mode = os.environ["ESPI_TEST_MODE"]
            if self.test_phase == "waiting":
                self.test_phase = "discovering"
                if mode == "manual":
                    QTest.mouseClick(self.check_button, Qt.LeftButton)
                elif mode == "periodic":
                    # Startup sees build 7; publish 8 only after that check finishes.
                    if self.updates.busy or not self.updates.events.empty():
                        self.test_phase = "waiting"
                        return
                    with urllib.request.urlopen(updater.API_ROOT + "/publish") as response:
                        response.read()
                    self.update_timer.start(50)
            if self.update_info is not None and self.test_phase == "discovering":
                self.test_phase = "downloading"
                if mode != "manual":
                    QTest.mouseClick(self.update_button, Qt.LeftButton)
            if self.pending_update and mode == "deferred":
                write_event("deferred", frames=self.drawn_frames)
                self.test_timer.stop()
                self.close()
        except Exception as exc:
            write_event("failed", error=str(exc))
            self.test_timer.stop()
            self.close()


def question(parent, title, text, *args):
    """Accept download and optionally defer restart without modal dialogs.

    Args: parent, title, text and args are QMessageBox arguments.
    Returns: The scripted QMessageBox answer.
    """
    write_event("question", title=title)
    if title == "Update ready" and os.environ["ESPI_TEST_MODE"] == "deferred":
        return QMessageBox.No
    if title not in ("Update available", "Update ready"):
        raise AssertionError(f"Unexpected dialog: {title}: {text}")
    return QMessageBox.Yes


def warning(parent, title, text, *args):
    """Record unexpected update warnings instead of blocking the binary test.

    Args: parent, title, text and args are QMessageBox arguments.
    Returns: None.
    """
    write_event("failed", error=f"{title}: {text}")


def main():
    """Configure isolated test boundaries, then enter the production CLI.

    Args: None; scenario settings are supplied by the parent test's environment.
    Returns: None; cli() exits with the real application's status.
    """
    metadata = json.loads((Path(sys._MEIPASS) / "test-build.json").read_text())
    updater._build_info = SimpleNamespace(**metadata)
    updater.API_ROOT = os.environ["ESPI_TEST_RELEASE_URL"]
    updater.CLEANUP_WAIT_S = 0.1
    updater.CLEANUP_ATTEMPTS = 200
    real_open = urllib.request.urlopen

    def local_open(request, *args, **kwargs):
        """Keep the frozen fixture from contacting external services.

        Args: request, args and kwargs are urllib urlopen arguments.
        Returns: The real loopback HTTP response.
        """
        url = request.full_url if hasattr(request, "full_url") else request
        if not url.startswith(updater.API_ROOT + "/"):
            raise AssertionError(f"External request from binary test: {url}")
        return real_open(request, *args, **kwargs)

    def settings(*args):
        """Keep real persistent Qt settings inside this test's temporary folder.

        Args: args are the normal organization/application names.
        Returns: A real QSettings using the isolated ini file.
        """
        return QSettings(str(Path(os.environ["ESPI_TEST_RUN"]) / "settings.ini"),
                         QSettings.IniFormat)

    def skip_source(self):
        """Skip the unrelated startup source-code download.

        Args: self is the SourceWorker.
        Returns: False.
        """
        return False

    def cpu_display():
        """Select the supported CPU display for headless CI.

        Args: None.
        Returns: The normal OpenGL probe result pair.
        """
        return False, "offscreen update test"

    real_install = updater.install_update
    real_popen = updater.subprocess.Popen

    def launch_process(command, *args, **kwargs):
        """Record the replacement PID while keeping native relaunch unchanged.

        Args: command, args and kwargs are subprocess.Popen arguments.
        Returns: The actual child process.
        """
        process = real_popen(command, *args, **kwargs)
        if updater.same_path(str(command[0]), sys.executable):
            write_event("spawned", child_pid=process.pid)
        return process

    def install_after_stop(*args, **kwargs):
        """Observe worker shutdown at the real native install boundary.

        Args: args and kwargs are install_update arguments.
        Returns: The real install result.
        """
        window = DrivenWindow.instance
        if window is not None:
            engines = window._all_engines()
            stopped = all(engine.wait(0) for engine in engines)
            write_event("installing", engines=len(engines), stopped=stopped)
            if not stopped:
                raise AssertionError("Installation began before all engines stopped")
        return real_install(*args, **kwargs)

    urllib.request.urlopen = local_open
    qt_app.QSettings = settings
    qt_app.source_code.SourceWorker.start = skip_source
    qt_app.opengl_available = cpu_display
    qt_app.MainWindow = DrivenWindow
    updater.install_update = install_after_stop
    updater.subprocess.Popen = launch_process
    QMessageBox.question = question
    QMessageBox.warning = warning
    try:
        cli.cli()
    finally:
        Path(os.environ["ESPI_TEST_RUN"], f"exited-{updater.current_build()}").touch()


if __name__ == "__main__":
    main()
