"""Tests for the OpenCV capture source, against fake devices (no hardware)."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from espi_live import capture
from espi_live.camera import CameraInfo
from fake_capture import (DSHOW_BASE, FakeCaptureFactory, FakeEnumerator, ListLog,
                          bgr_frame, device, integrated_webcam, usb_live2, webcam)

YUY2 = float(cv2.VideoWriter_fourcc(*"YUY2"))


def capture_info(enumerated, position=1):
    """Turn an enumerated fake device into the CameraInfo the source opens.

    Args: enumerated is a fake_capture device; position is its list place.
    Returns: A capture CameraInfo.
    """
    return CameraInfo(f"video {position}", enumerated.name, kind="capture",
                      index=enumerated.index, backend=enumerated.backend,
                      vid=enumerated.vid, pid=enumerated.pid, path=enumerated.path)


class CaptureTestCase(unittest.TestCase):
    """Base class that replaces cv2.VideoCapture with a fake factory."""

    def use_devices(self, **options):
        """Install a fake cv2.VideoCapture for this test.

        Args: options are FakeCaptureFactory arguments.
        Returns: The factory, whose .made lists the devices opened.
        """
        factory = FakeCaptureFactory(**options)
        patcher = mock.patch.object(capture.cv2, "VideoCapture", factory)
        patcher.start()
        self.addCleanup(patcher.stop)
        return factory

    def open_session(self, enumerated=None, **options):
        """Open a capture session on a fake device.

        Args: enumerated is the device to open (a USB-Live2 by default);
            options configure the fake device.
        Returns: (session, fake device, log).
        """
        factory = self.use_devices(**options)
        log = ListLog()
        session = capture.CaptureSession(log, capture_info(enumerated or usb_live2()))
        session.open()
        return session, factory.made[-1], log


class ListCaptureDevicesTests(unittest.TestCase):
    """Enumerate video devices without opening them."""

    def list_with(self, enumerator):
        """List devices through a fake enumerator.

        Args: enumerator is a FakeEnumerator, or None for a missing package.
        Returns: (infos, log).
        """
        log = ListLog()
        with mock.patch.object(capture, "import_enumerator",
                               return_value=(enumerator, "" if enumerator else "is not installed")):
            return capture.list_capture_devices(log), log

    def test_list_capture_devices(self):
        """Every device becomes a capture CameraInfo with a short, unique id.

        Args: None.
        Returns: None.
        """
        enumerator = FakeEnumerator([usb_live2(0), usb_live2(1),
                                     device(DSHOW_BASE + 3, "OBS Virtual Camera")])
        infos, _ = self.list_with(enumerator)
        self.assertEqual(enumerator.calls, [capture.capture_backend()])
        self.assertEqual([info.serial for info in infos], ["video 1", "video 2", "video 3"])
        self.assertTrue(all(info.kind == "capture" for info in infos))
        first = infos[0]
        self.assertEqual((first.model, first.index, first.backend, first.vid, first.pid),
                         ("Hauppauge USB-Live2", DSHOW_BASE, DSHOW_BASE, 0x2040, 0xC200))
        self.assertIn("vid_2040", first.path)
        self.assertNotEqual(infos[0].path, infos[1].path)
        self.assertEqual((infos[2].path, infos[2].vid), ("", None))
        self.assertEqual(infos[2].label, "OBS Virtual Camera (video 3)")

    def test_built_in_cameras_are_left_out(self):
        """A laptop's own cameras are not listed; the log says which were skipped.

        Args: None.
        Returns: None.
        """
        enumerator = FakeEnumerator([
            integrated_webcam(0), usb_live2(1),
            integrated_webcam(2, "Integrated IR Webcam"),
            device(DSHOW_BASE + 3, "FaceTime HD Camera"), webcam(4)])
        infos, log = self.list_with(enumerator)
        self.assertEqual([info.label for info in infos],
                         ["Hauppauge USB-Live2 (video 1)", "Webcam C270 (video 2)"])
        self.assertEqual(log.lines, [
            "Ignoring the built-in camera Integrated Webcam (USB 0c45:674c).",
            "Ignoring the built-in camera Integrated IR Webcam (USB 0c45:674c).",
            "Ignoring the built-in camera FaceTime HD Camera."])
        _, log = self.list_with(FakeEnumerator([integrated_webcam(0, None)]))
        self.assertEqual(log.lines,
                         ["Ignoring the built-in camera video device (USB 0c45:674c)."])

    def test_list_without_enumerator(self):
        """A missing cv2_enumerate_cameras lists nothing and says why.

        Args: None.
        Returns: None.
        """
        infos, log = self.list_with(None)
        self.assertEqual(infos, [])
        self.assertEqual(len(log.lines), 1)
        self.assertEqual(log.lines, ["Video devices are not listed: cv2_enumerate_cameras "
                                     "is not installed."])

    def test_list_says_why_the_enumerator_failed(self):
        """A package that is present but cannot load is reported with its error.

        Args: None.
        Returns: None.
        """
        log = ListLog()
        problem = "could not be loaded: DLL load failed while importing _windows_backend"
        with mock.patch.object(capture, "import_enumerator", return_value=(None, problem)):
            self.assertEqual(capture.list_capture_devices(log), [])
        self.assertEqual(log.lines, [f"Video devices are not listed: cv2_enumerate_cameras {problem}."])

    def test_list_error(self):
        """An enumeration failure lists nothing and is logged.

        Args: None.
        Returns: None.
        """
        infos, log = self.list_with(FakeEnumerator(error=RuntimeError("boom")))
        self.assertEqual(infos, [])
        self.assertEqual(log.lines, ["Could not list video devices: boom"])

    def test_capture_backend(self):
        """Each platform uses its native capture backend.

        Args: None.
        Returns: None.
        """
        for platform, backend in (("win32", cv2.CAP_DSHOW), ("linux", cv2.CAP_V4L2),
                                  ("darwin", cv2.CAP_AVFOUNDATION)):
            with mock.patch.object(capture.sys, "platform", platform):
                self.assertEqual(capture.capture_backend(), backend)

    def test_import_enumerator(self):
        """The enumerator imports lazily and is optional.

        Args: None.
        Returns: None.
        """
        fake = types.ModuleType("cv2_enumerate_cameras")
        with mock.patch.dict(sys.modules, {"cv2_enumerate_cameras": fake}):
            self.assertEqual(capture.import_enumerator(), (fake, ""))
        with mock.patch.dict(sys.modules, {"cv2_enumerate_cameras": None}):
            self.assertEqual(capture.import_enumerator(), (None, "is not installed"))

    def test_import_enumerator_reports_a_failing_native_module(self):
        """A DLL error is reported as such, not as "not installed".

        Args: None.
        Returns: None.
        """
        error = ImportError("DLL load failed while importing _windows_backend")
        with mock.patch("builtins.__import__", side_effect=error), \
                mock.patch.object(capture, "bundled_dll_directories", return_value=[]):
            module, problem = capture.import_enumerator()
        self.assertIsNone(module)
        self.assertEqual(problem, "could not be loaded: DLL load failed while importing "
                                  "_windows_backend")

    def test_import_enumerator_retries_with_bundled_dll_folders(self):
        """A frozen build adds its `*.libs` folders to the DLL search path and retries.

        Args: None.
        Returns: None.
        """
        fake = types.ModuleType("cv2_enumerate_cameras")
        real_import = __import__
        calls = []

        def flaky_import(name, *args, **kwargs):
            if name != "cv2_enumerate_cameras":
                return real_import(name, *args, **kwargs)
            calls.append(name)
            if len(calls) == 1:
                raise ImportError("DLL load failed while importing _windows_backend")
            return fake

        added = []
        folders = [Path("a.libs"), Path("b.libs")]
        before = len(capture._dll_directory_handles)
        with mock.patch("builtins.__import__", side_effect=flaky_import), \
                mock.patch.object(capture, "bundled_dll_directories", return_value=folders), \
                mock.patch.object(capture.os, "add_dll_directory",
                                  side_effect=lambda path: added.append(path) or path,
                                  create=True):
            self.assertEqual(capture.import_enumerator(), (fake, ""))
        self.assertEqual(added, ["a.libs", "b.libs"])
        self.assertEqual(len(calls), 2)
        del capture._dll_directory_handles[before:]

    def test_import_enumerator_still_failing_after_the_retry(self):
        """When the retry fails too, the second error is the one reported.

        Args: None.
        Returns: None.
        """
        errors = [ImportError("first"), ImportError("DLL load failed again")]

        def always_fail(name, *args, **kwargs):
            raise errors.pop(0)

        before = len(capture._dll_directory_handles)
        with mock.patch("builtins.__import__", side_effect=always_fail), \
                mock.patch.object(capture, "bundled_dll_directories",
                                  return_value=[Path("a.libs")]), \
                mock.patch.object(capture.os, "add_dll_directory", create=True):
            module, problem = capture.import_enumerator()
        self.assertIsNone(module)
        self.assertEqual(problem, "could not be loaded: DLL load failed again")
        del capture._dll_directory_handles[before:]

    def test_bundled_dll_directories(self):
        """Only a frozen build on a platform with DLL search paths has folders to add.

        Args: None.
        Returns: None.
        """
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "pandas.libs").mkdir()
            (root / "cv2_enumerate_cameras.libs").mkdir()
            (root / "numpy").mkdir()
            (root / "stray.libs").write_text("a file, not a folder")
            with mock.patch.object(capture.os, "add_dll_directory", create=True):
                with mock.patch.object(sys, "_MEIPASS", folder, create=True):
                    self.assertEqual([path.name for path in capture.bundled_dll_directories()],
                                     ["cv2_enumerate_cameras.libs", "pandas.libs"])
                with mock.patch.object(sys, "_MEIPASS", "", create=True):
                    self.assertEqual(capture.bundled_dll_directories(), [])
            with mock.patch.object(sys, "_MEIPASS", folder, create=True), \
                    mock.patch.object(capture, "os", types.SimpleNamespace()):
                self.assertEqual(capture.bundled_dll_directories(), [])
        self.assertEqual(capture.bundled_dll_directories(), [])  # not frozen


class SessionOpenTests(CaptureTestCase):
    """Open devices, request the adapter's native format, fail clearly."""

    def test_open_usb_live2_requests_native_size(self):
        """A USB-Live2 is switched to composite at 720 × 480.

        Args: None.
        Returns: None.
        """
        session, cap, _ = self.open_session(
            props={cv2.CAP_PROP_FOURCC: YUY2, cv2.CAP_PROP_FPS: 29.97})
        self.assertEqual((cap.index, cap.backend), (DSHOW_BASE, DSHOW_BASE))
        self.assertEqual(cap.sets, [(cv2.CAP_PROP_CHANNEL, 0),
                                    (cv2.CAP_PROP_FRAME_WIDTH, 720),
                                    (cv2.CAP_PROP_FRAME_HEIGHT, 480)])
        self.assertTrue(session.interlaced)
        identity = session.identity
        self.assertTrue(identity.supported)
        self.assertEqual((identity.width, identity.height, identity.fourcc),
                         (720, 480, "YUY2"))
        self.assertAlmostEqual(identity.fps, 29.97)
        self.assertEqual(identity.backend, cv2.videoio_registry.getBackendName(DSHOW_BASE))

    def test_size_comes_from_the_frame(self):
        """The identity reports the size actually delivered (here PAL).

        Args: None.
        Returns: None.
        """
        session, _, _ = self.open_session(frames=[bgr_frame(720, 576)])
        self.assertEqual((session.identity.width, session.identity.height), (720, 576))
        self.assertTrue(any("PAL" in note for note in session.identity.warnings()))

    def test_open_unknown_device_sets_nothing(self):
        """An unrecognised device is used as delivered.

        Args: None.
        Returns: None.
        """
        session, cap, _ = self.open_session(webcam(), props={cv2.CAP_PROP_FPS: float("nan")})
        self.assertEqual(cap.sets, [])
        self.assertFalse(session.interlaced)
        self.assertFalse(session.identity.supported)
        self.assertEqual(session.identity.fps, 0.0)

    def test_open_failure_names_device_and_wintv(self):
        """A device that will not open, or sends nothing, fails with advice.

        Args: None.
        Returns: None.
        """
        for options in ({"opened": False}, {"frames": [None]}, {"frames": [np.zeros(3)]}):
            factory = self.use_devices(**options)
            session = capture.CaptureSession(ListLog(), capture_info(usb_live2()))
            with self.assertRaises(RuntimeError) as caught:
                session.open()
            self.assertIn("Hauppauge USB-Live2 (video 1)", str(caught.exception))
            self.assertIn("WinTV", str(caught.exception))
            self.assertTrue(factory.made[-1].released)
            self.assertIsNone(session.cap)

    def test_backend_name_falls_back_to_number(self):
        """An unknown backend number is shown as the number.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(capture.cv2.videoio_registry, "getBackendName",
                               side_effect=cv2.error("unknown")):
            self.assertEqual(capture.backend_name(12345), "12345")


class SessionStreamTests(CaptureTestCase):
    """Grab frames, measure the rate, describe and close."""

    def test_grab_converts_bgr_to_mono8(self):
        """BGR frames come back as 2-D Mono8; a failed read is an error.

        Args: None.
        Returns: None.
        """
        session, cap, _ = self.open_session(
            frames=[bgr_frame(value=90), bgr_frame(value=90), None])
        frame, state = session.grab(100)
        self.assertEqual(state, "ok")
        self.assertEqual((frame.shape, frame.dtype), ((480, 720), np.uint8))
        self.assertEqual(int(frame[0, 0]), 90)
        error, state = session.grab(100)
        self.assertEqual(state, "error")
        self.assertIsInstance(error, RuntimeError)

    def test_failed_read_waits_like_a_timeout(self):
        """A failed read waits out the timeout, so an unplugged device cannot spin the loop.

        Args: None.
        Returns: None.
        """
        session, _, _ = self.open_session(frames=[bgr_frame(), None])
        with mock.patch.object(capture.time, "sleep") as sleep:
            _, state = session.grab(100)
        self.assertEqual(state, "error")
        sleep.assert_called_once_with(0.1)

    def test_grab_reports_opencv_errors(self):
        """An exception from the driver is returned as an error status.

        Args: None.
        Returns: None.
        """
        calls = []

        def frames():
            """Deliver one frame, then fail like a driver error.

            Args: None.
            Returns: (True, frame) on the first call.
            """
            calls.append(1)
            if len(calls) > 1:
                raise cv2.error("device lost")
            return True, bgr_frame()

        session, _, _ = self.open_session(frames=frames)
        with mock.patch.object(capture.time, "sleep") as sleep:
            error, state = session.grab(100)
        self.assertEqual(state, "error")
        self.assertIn("device lost", str(error))
        sleep.assert_called_once_with(0.1)

    def test_grab_rejects_unreadable_layout(self):
        """A frame layout that cannot become Mono8 is an error status.

        Args: None.
        Returns: None.
        """
        session, _, _ = self.open_session(frames=[bgr_frame(), np.zeros((2, 2, 2, 2))])
        error, state = session.grab(100)
        self.assertEqual(state, "error")
        self.assertIsInstance(error, ValueError)

    def test_to_mono8_variants(self):
        """Mono8 passes through, 16-bit is shifted, odd layouts are refused.

        Args: None.
        Returns: None.
        """
        mono = np.full((4, 4), 7, dtype=np.uint8)
        self.assertIs(capture.to_mono8(mono), mono)
        deep = np.full((4, 4), 0x1234, dtype=np.uint16)
        self.assertTrue(np.all(capture.to_mono8(deep) == 0x12))
        self.assertEqual(capture.to_mono8(deep).dtype, np.uint8)
        with self.assertRaises(ValueError):
            capture.to_mono8(np.zeros((2, 2, 2, 2), dtype=np.uint8))
        with self.assertRaises(ValueError):
            capture.to_mono8(np.zeros((2, 2), dtype=np.float32))

    def test_read_values_measures_rate(self):
        """The resulting rate is frames per second since the last reading.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(capture.time, "monotonic", side_effect=[10.0, 10.5, 11.5]):
            session, _, _ = self.open_session()
            self.assertIsNone(session.read_values()["resulting"])
            for _ in range(30):
                session.grab(100)
            values = session.read_values()
        self.assertAlmostEqual(values["resulting"], 30.0)
        self.assertEqual((values["exposure"], values["gain"], values["requested"]),
                         (None, None, None))

    def test_read_stream_counters_is_empty(self):
        """OpenCV capture exposes no transport counters.

        Args: None.
        Returns: None.
        """
        session, _, _ = self.open_session()
        self.assertEqual(session.read_stream_counters(), {})

    def test_describe(self):
        """The status line names the capture, its size and measured rate.

        Args: None.
        Returns: None.
        """
        session, _, _ = self.open_session()
        self.assertEqual(session.describe({"resulting": 29.94}),
                         "Analog capture 720 × 480: 29.9 fps measured; exposure, "
                         "gain and frame rate are set on the camera")
        self.assertIn("measuring", session.describe({"resulting": None}))
        other, _, _ = self.open_session(webcam())
        self.assertTrue(other.describe({"resulting": 30.0}).startswith(
            "Video capture 720 × 480: 30.0 fps measured"))

    def test_settings_are_fixed(self):
        """Exposure, gain and rate cannot be set and have no limits.

        Args: None.
        Returns: None.
        """
        session, cap, _ = self.open_session()
        sets = list(cap.sets)
        self.assertIsNone(session.set_float("ExposureTime", 1000.0))
        self.assertEqual(cap.sets, sets)
        self.assertEqual(session.limits(), {})

    def test_read_details_lists_device_facts(self):
        """The report details cover the device, format, properties and frame.

        Args: None.
        Returns: None.
        """
        session, _, _ = self.open_session(
            webcam(), frames=[bgr_frame(value=50)],
            props={cv2.CAP_PROP_FOURCC: YUY2, cv2.CAP_PROP_BRIGHTNESS: 128.0})
        details = dict(session.read_details())
        self.assertEqual(details["Device name"], "Webcam C270")
        self.assertEqual(details["USB ID"], "046d:0825")
        self.assertEqual(details["FOURCC"], "YUY2")
        self.assertEqual(details["Backend"], session.identity.backend)
        self.assertEqual(details["OpenCV"], cv2.__version__)
        self.assertEqual(details["Brightness"], "128")
        self.assertEqual(details["First frame (raw)"], "480 × 720 × 3, uint8")
        self.assertEqual(details["First frame min / mean / max"], "50 / 50.0 / 50")
        bare = capture.CaptureSession(ListLog(), CameraInfo("video 1", "", kind="capture",
                                                            index=1, backend=1))
        self.use_devices()
        bare.open()
        bare_details = dict(bare.read_details())
        self.assertEqual((bare_details["USB ID"], bare_details["Device path"],
                          bare_details["FOURCC"]), ("not reported", "not reported", "unknown"))

    def test_close_is_idempotent_and_logs_errors(self):
        """Closing twice is safe; a release error is logged, not raised.

        Args: None.
        Returns: None.
        """
        session, cap, log = self.open_session()
        session.close()
        session.close()
        self.assertTrue(cap.released)
        session, cap, log = self.open_session()
        cap.release = mock.Mock(side_effect=RuntimeError("busy"))
        session.close()
        self.assertEqual(log.lines, ["Capture release error: busy"])
        self.assertIsNone(session.cap)


if __name__ == "__main__":
    unittest.main()
