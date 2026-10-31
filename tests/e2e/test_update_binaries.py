"""Opt-in native PyInstaller A -> B tests, run on each OS in CI.

    PYTHONPATH=src python -m unittest discover -s tests/e2e -v

Builds and installations live in temporary directories, including output,
tokens and settings. No SDK, camera, GitHub credentials or release is needed.
"""

import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from update_fixtures import ReleaseServer

ROOT = Path(__file__).resolve().parents[2]
MAC = sys.platform == "darwin"
PROGRAM = "ESPI Live.app" if MAC else "espi-live"
ASSET = "espi-live-test.zip" if MAC or os.name == "nt" else "espi-live-test.tar.gz"


def executable(program):
    """Locate the native executable in a program directory or app bundle.

    Args: program is the program's top-level path.
    Returns: Its executable path.
    """
    return (program / "Contents/MacOS/ESPI Live" if MAC else
            program / ("espi-live.exe" if os.name == "nt" else "espi-live"))


def events(folder, build):
    """Read complete evidence records while a child may still be writing.

    Args: folder is the scenario directory; build identifies the child.
    Returns: The completed JSON event records.
    """
    path = folder / f"events-{build}.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith("\n")]


def wait_for_exit(pid, timeout):
    """Wait for OS process termination, including Windows DLL unloading.

    Args: pid is a test-launched process; timeout is the maximum seconds to wait.
    Returns: True when it has exited, False when it is still running.
    """
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            if ctypes.get_last_error() == 87:  # PID no longer exists.
                return True
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            result = kernel.WaitForSingleObject(handle, int(timeout * 1000))
            if result == 0xffffffff:
                raise ctypes.WinError(ctypes.get_last_error())
            return result == 0
        finally:
            kernel.CloseHandle(handle)
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


def replacement_pids(folder):
    """Find replacements even if they fail before constructing their window.

    Args: folder holds the scenario evidence.
    Returns: The set of replacement process IDs.
    """
    spawned = {item["child_pid"] for item in events(folder, 7) if item["event"] == "spawned"}
    return spawned | {item["pid"] for item in events(folder, 8)}


@contextmanager
def scenario_folder(mode):
    """Keep optional CI evidence before deleting a disposable installation.

    Args: mode names the scenario's artifact subdirectory.
    Returns: A context manager yielding a temporary Path.
    """
    with tempfile.TemporaryDirectory(prefix="espi-update-run-") as tmp:
        folder = Path(tmp)
        try:
            yield folder
        finally:
            artifacts = os.environ.get("ESPI_TEST_ARTIFACTS")
            if artifacts:
                destination = Path(artifacts).resolve() / mode
                for pattern in ("*.jsonl", "*.log", "*.ini", "**/output/logs/*"):
                    for path in folder.glob(pattern):
                        if path.is_file():
                            target = destination / path.relative_to(folder)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(path, target)


class BinaryUpdateTests(unittest.TestCase):
    """Actually replace and relaunch frozen apps with distinct library folders."""

    @classmethod
    def setUpClass(cls):
        """Freeze two credential-free copies of the real app's update path.

        Args: None.
        Returns: None.
        """
        cls.build_tmp = tempfile.TemporaryDirectory(prefix="espi-update-builds-")
        cls.addClassCleanup(cls.build_tmp.cleanup)
        cls.builds = Path(cls.build_tmp.name)
        for build in (7, 8):
            print(f"Freezing update test build {build}...", flush=True)
            folder = cls.builds / str(build)
            folder.mkdir()
            metadata = folder / "test-build.json"
            metadata.write_text(json.dumps({"BUILD": build, "COMMIT": f"test{build}",
                                            "REPO": "test/espi-live", "ASSET": ASSET,
                                            "TOKEN": "", "FEEDBACK_TOKEN": ""}))
            command = [sys.executable, "-m", "PyInstaller", "--noconfirm",
                       "--paths", str(ROOT / "src"), "--distpath", str(folder / "dist"),
                       "--workpath", str(folder / "work"), "--specpath", str(folder),
                       "--add-data", str(metadata) + os.pathsep + ".",
                       "--add-data", str(ROOT / "src/espi_live/user_guide.html")
                       + os.pathsep + "espi_live", "--onedir"]
            if MAC:
                command += ["--name", "ESPI Live", "--windowed",
                            "--osx-bundle-identifier", "io.github.espi-live.update-test"]
            else:
                command += ["--name", "espi-live", "--console", "--contents-directory",
                            f"_internal-{build}"]
            command.append(str(ROOT / "tests/e2e/update_app.py"))
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=300, cwd=ROOT)
            if result.returncode:
                raise AssertionError(f"PyInstaller build {build} failed:\n{result.stdout}")
        archive_base = cls.builds / "release"
        source = cls.builds / "8/dist"
        if MAC:
            archive = str(archive_base) + ".zip"
            subprocess.run(["ditto", "-c", "-k", "--keepParent", str(source / PROGRAM), archive],
                           check=True, timeout=60)
        else:
            archive = shutil.make_archive(str(archive_base),
                                          "zip" if os.name == "nt" else "gztar",
                                          root_dir=source, base_dir=PROGRAM)
        cls.archive = Path(archive).read_bytes()

    def scenario(self, mode):
        """Update a running A and require a healthy native B plus startup cleanup.

        Args: mode is startup, periodic, manual, deferred or cli.
        Returns: None; assertions include the per-process event evidence.
        """
        with scenario_folder(mode) as folder:
            program = folder / PROGRAM
            shutil.copytree(self.builds / "7/dist" / PROGRAM, program, symlinks=True)
            output_root = folder if MAC else program
            keep = {"output/recordings/measurement.avi": b"keep recording",
                    "output/frames/frame.png": b"keep frame",
                    "update-token.txt": b"test-token", "feedback-token.txt": b"test-feedback"}
            for name, data in keep.items():
                file = output_root / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(data)
            with ReleaseServer(self.archive, ASSET, build=7 if mode == "periodic" else 8) as server:
                env = dict(os.environ, QT_QPA_PLATFORM="offscreen", ESPI_ERROR_REPORTS="0",
                           ESPI_UPDATE_TOKEN="", ESPI_FEEDBACK_TOKEN="",
                           ESPI_TEST_RUN=str(folder), ESPI_TEST_MODE=mode,
                           ESPI_TEST_RELEASE_URL=server.url, PYINSTALLER_RESET_ENVIRONMENT="1")
                args = ["--demo", "--demo-cameras", "2", "--no-error-reports"]
                if mode == "manual":
                    args.append("--no-update-check")
                log_path = folder / "process.log"
                with log_path.open("wb") as log:
                    old = subprocess.Popen([str(executable(program)),
                                            *(["--update"] if mode == "cli" else args)],
                                           env=env, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        deadline = time.monotonic() + 45
                        while old.poll() is None and time.monotonic() < deadline:
                            time.sleep(0.02)
                        self.assertEqual(old.poll(), 0, log_path.read_text(errors="replace"))
                        if mode in ("cli", "deferred"):
                            # Neither path should relaunch without the user's request.
                            self.assertEqual(events(folder, 8), [])
                            self.assertFalse(any(item["event"] == "spawned"
                                                 for item in events(folder, 7)))
                            old = subprocess.Popen([str(executable(program)), *args], env=env,
                                                   stdout=log, stderr=subprocess.STDOUT)
                        deadline = time.monotonic() + 40
                        while time.monotonic() < deadline:
                            evidence = events(folder, 8)
                            if any(item["event"] in ("healthy", "failed") for item in evidence):
                                break
                            time.sleep(0.02)
                        self.assertTrue(any(item["event"] == "healthy" for item in evidence),
                                        f"A: {events(folder, 7)}; B: {evidence}; "
                                        + log_path.read_text(errors="replace"))
                        self.assertFalse(any(item["event"] == "failed"
                                             for item in events(folder, 7) + evidence))
                        # B closes after reporting health; let its log and settings flush.
                        deadline = time.monotonic() + 5
                        while not (folder / "exited-8").exists() and time.monotonic() < deadline:
                            time.sleep(0.02)
                        self.assertTrue((folder / "exited-8").exists())
                        self.assertEqual(old.wait(timeout=10), 0)
                        for pid in replacement_pids(folder):
                            self.assertTrue(wait_for_exit(pid, 10), f"Build B ({pid}) did not exit")
                    finally:
                        if old.poll() is None:
                            old.kill()
                        old.wait(timeout=10)
                        for pid in replacement_pids(folder):
                            if not wait_for_exit(pid, 0):
                                os.kill(pid, signal.SIGTERM)
                                self.assertTrue(wait_for_exit(pid, 10),
                                                f"Cannot stop test replacement {pid}")
                for name, data in keep.items():
                    self.assertEqual((output_root / name).read_bytes(), data)
                self.assertFalse(Path(str(program if MAC else executable(program)) + ".old").exists())
                self.assertFalse(list(output_root.glob(".espi-update-*")))
                if not MAC:
                    self.assertFalse((program / "_internal-7").exists())
                    self.assertTrue((program / "_internal-8").is_dir())
                self.assertTrue((folder / "settings.ini").is_file())
                if mode != "cli":
                    installed = next(item for item in events(folder, 7)
                                     if item["event"] == "installing")
                    self.assertEqual(installed["engines"], 2)
                    self.assertTrue(installed["stopped"])
                self.assertEqual(sum(item["event"] == "spawned" for item in events(folder, 7)),
                                 0 if mode in ("cli", "deferred") else 1)
                started = next(item for item in events(folder, 8) if item["event"] == "started")
                self.assertEqual(started["args"], args)

    def test_automatic_startup_update(self):
        """Startup discovery replaces A, relaunches B and cleans up A.

        Args: None.
        Returns: None.
        """
        self.scenario("startup")

    def test_automatic_periodic_update(self):
        """A timer discovers a release offered after startup and updates to it.

        Args: None.
        Returns: None.
        """
        self.scenario("periodic")

    def test_manual_button_update(self):
        """The manual check button updates even with automatic checking disabled.

        Args: None.
        Returns: None.
        """
        self.scenario("manual")

    def test_deferred_restart(self):
        """Declining restart installs on exit and B starts on the next launch.

        Args: None.
        Returns: None.
        """
        self.scenario("deferred")

    def test_cli_update(self):
        """The frozen --update command installs B without opening the UI.

        Args: None.
        Returns: None.
        """
        self.scenario("cli")
