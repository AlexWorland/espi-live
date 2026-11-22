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
from fake_capture import (DSHOW_BASE, MSMF, YUY2_CODE, YUYV_CODE, FakeCaptureFactory,
                          FakeEnumerator, ListLog, bgr_frame, device, integrated_webcam,
                          ramp_luma, usb_live2, usb_live2_rev_e, webcam, yuy2_frame)
from espi_live.processing import single_field

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
        self.assertEqual(log.lines[-1], "Capture release error: busy")
        self.assertIsNone(session.cap)


def reference_for(luma):
    """Build the BGR frame the BGR path delivers for a luma image.

    Args: luma is a 2-D uint8 array of video-range codes.
    Returns: A (H, W, 3) uint8 array, grey at the expanded level.
    """
    return np.repeat(capture.expand_video_range(luma)[:, :, None], 3, axis=2)


RAW_PROPS = {cv2.CAP_PROP_FRAME_WIDTH: 720.0, cv2.CAP_PROP_FRAME_HEIGHT: 480.0}


class LumaConversionTests(unittest.TestCase):
    """The video-range table and the YUY2 layouts backends deliver."""

    def test_table_matches_the_bgr_conversion_scale(self):
        """Expanded raw Y has the scale of YUY2 → BGR → grey (BT.601, video range).

        Args: None.
        Returns: None.
        """
        table = capture.EXPAND_TABLE
        self.assertEqual((table.dtype, table.shape), (np.uint8, (256,)))
        self.assertEqual([int(table[code]) for code in (0, 16, 17, 126, 235, 255)],
                         [0, 0, 1, 128, 255, 255])
        self.assertTrue(np.all(np.diff(table.astype(int)) >= 0))
        codes = np.arange(256, dtype=np.uint8)[None, :]
        grey = cv2.cvtColor(cv2.cvtColor(yuy2_frame(codes), cv2.COLOR_YUV2BGR_YUY2),
                            cv2.COLOR_BGR2GRAY)[0]
        self.assertLessEqual(int(np.abs(grey.astype(int) - table.astype(int)).max()), 1)

    def test_chroma_noise_moves_only_the_bgr_path(self):
        """Noisy chroma shifts the BGR path's grey; the raw luma ignores it.

        Args: None.
        Returns: None.
        """
        luma = ramp_luma(64, 8, low=8, high=250)
        chroma = np.random.default_rng(3).normal(128, 6, (8, 64)).clip(0, 255)
        packed = yuy2_frame(luma)
        packed[:, :, 1] = chroma.astype(np.uint8)
        old = cv2.cvtColor(cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_YUY2), cv2.COLOR_BGR2GRAY)
        new = capture.expand_video_range(capture.yuy2_luma(packed, 64, 8))
        self.assertTrue(np.array_equal(new, capture.EXPAND_TABLE[luma]))
        self.assertGreater(int(np.abs(old.astype(int) - new.astype(int)).max()), 1)

    def test_yuy2_luma_layouts(self):
        """Pairs, wide rows, byte rows (padded or not) and 1-D buffers give the luma.

        Args: None.
        Returns: None.
        """
        luma = ramp_luma(8, 4)
        for frame in (yuy2_frame(luma, "pairs"), yuy2_frame(luma, "wide"),
                      yuy2_frame(luma, "row"), yuy2_frame(luma, "row", stride=20),
                      yuy2_frame(luma, "row").reshape(-1)):
            with self.subTest(shape=frame.shape):
                self.assertTrue(np.array_equal(capture.yuy2_luma(frame, 8, 4), luma))

    def test_yuy2_luma_refuses_what_is_not_certainly_yuy2(self):
        """Other types, sizes and layouts give None, so the caller keeps BGR.

        Args: None.
        Returns: None.
        """
        luma = ramp_luma(8, 4)
        row = yuy2_frame(luma, "row")
        for frame, width, height in (
                (bgr_frame(8, 4), 8, 4), (yuy2_frame(luma).astype(np.uint16), 8, 4),
                (yuy2_frame(luma), 10, 4), (row, 8, 0), (row[:, :-1], 8, 4),
                (row, 2, 4), (yuy2_frame(luma, "row", stride=40), 8, 4),
                (np.zeros((2, 3, 4), dtype=np.uint8), 8, 4)):
            with self.subTest(shape=frame.shape, size=(width, height)):
                self.assertIsNone(capture.yuy2_luma(frame, width, height))

    def test_describe_layout(self):
        """A layout is its shape and type.

        Args: None.
        Returns: None.
        """
        self.assertEqual(capture.describe_layout(yuy2_frame(ramp_luma(), "row")),
                         "1 × 691200, uint8")

    def test_raw_route_and_setting(self):
        """V4L2 reads raw anywhere, DirectShow only on Windows; the setting forces BGR.

        Args: None.
        Returns: None.
        """
        v4l2 = CameraInfo("video 1", "x", kind="capture", index=0, backend=cv2.CAP_V4L2)
        dshow = capture_info(usb_live2())
        self.assertEqual(capture.raw_route(v4l2), "v4l2")
        with mock.patch.object(capture.sys, "platform", "darwin"):
            self.assertEqual(capture.raw_route(dshow), "")
        with mock.patch.object(capture.sys, "platform", "win32"):
            self.assertEqual(capture.raw_route(dshow), "msmf")
        for value, expected in (("", "auto"), ("raw", "auto"), (" BGR ", "bgr")):
            with mock.patch.dict(capture.os.environ, {capture.LUMA_SETTING: value}):
                self.assertEqual(capture.luma_setting(), expected)

    def test_device_instance(self):
        """DirectShow and Media Foundation paths of one device reduce alike.

        Args: None.
        Returns: None.
        """
        dshow = usb_live2_rev_e().path
        msmf = usb_live2_rev_e(MSMF, interface="e5323777").path
        self.assertNotEqual(dshow, msmf)
        self.assertEqual(capture.device_instance(dshow), capture.device_instance(msmf.upper()))
        self.assertEqual(capture.device_instance("video0"), "")
        self.assertEqual(capture.device_instance(None), "")


class V4L2RawLumaTests(CaptureTestCase):
    """Linux: V4L2 hands over YUYV once its conversion is off."""

    def open_v4l2(self, raw=None, props=None, reference=None, fixed=()):
        """Open a USB-Live2 listed by V4L2.

        Args: raw is the raw frames list (None: the flag is ignored); props
            overrides the device properties; reference is the BGR frame;
            fixed lists properties the device will not change.
        Returns: (session, fake device, log).
        """
        luma = ramp_luma()
        enumerated = usb_live2()
        enumerated.backend = cv2.CAP_V4L2
        options = {cv2.CAP_PROP_FOURCC: YUYV_CODE, **RAW_PROPS, **(props or {})}
        return self.open_session(
            enumerated, frames=[reference_for(luma) if reference is None else reference],
            raw_frames=raw, props=options, fixed=fixed)

    def test_raw_luma_is_used(self):
        """The Y bytes, expanded, become the Mono8 frames; single field still works.

        Args: None.
        Returns: None.
        """
        luma = ramp_luma()
        session, cap, log = self.open_v4l2(raw=[yuy2_frame(luma)])
        expected = capture.expand_video_range(luma)
        self.assertIn((cv2.CAP_PROP_CONVERT_RGB, 0), cap.sets)
        self.assertEqual(session.luma_path,
                         "raw YUY2 luma, video range 16–235 expanded to 0–255")
        self.assertEqual(session.raw_layout, "480 × 720 × 2, uint8")
        self.assertTrue(np.array_equal(session.first_frame, expected))
        frame, state = session.grab(100)
        self.assertEqual(state, "ok")
        self.assertEqual((frame.shape, frame.dtype), ((480, 720), np.uint8))
        self.assertTrue(np.array_equal(frame, expected))
        self.assertEqual(single_field(frame).shape, (480, 720))
        self.assertTrue(session.interlaced)
        self.assertEqual(session.identity.fourcc, "YUYV")
        line = log.lines[-1]
        self.assertTrue(line.startswith("Analog luma: raw YUY2 luma"), line)
        self.assertIn("frame 480 × 720 × 2, uint8; raw Y 40–200", line)
        self.assertEqual(dict(session.read_details())["Luma path"], session.luma_path)

    def test_a_backend_that_ignores_the_flag_keeps_bgr(self):
        """BGR frames after CONVERT_RGB 0 are refused, and conversion is turned back on.

        Args: None.
        Returns: None.
        """
        session, cap, log = self.open_v4l2(raw=None)
        self.assertEqual(cap.sets[-1], (cv2.CAP_PROP_CONVERT_RGB, 1))
        self.assertEqual(session.luma_path, "BGR → grey")
        self.assertIn("raw YUY2 through V4L2 was not used: a 480 × 720 × 3, uint8 "
                      "frame is not YUY2 of that size", log.lines[-1])
        frame, state = session.grab(100)
        self.assertEqual((state, frame.shape), ("ok", (480, 720)))
        self.assertEqual(session.raw_layout, "480 × 720 × 3, uint8")

    def test_wrong_format_size_or_brightness_keeps_bgr(self):
        """A non-YUY2 format, another size or a brightness mismatch keeps BGR.

        Args: None.
        Returns: None.
        """
        luma = ramp_luma()
        raw = [yuy2_frame(luma)]
        uyvy = float(cv2.VideoWriter_fourcc(*"UYVY"))
        cases = (
            ({cv2.CAP_PROP_FOURCC: uyvy}, None, "it delivers UYVY, not YUY2"),
            ({cv2.CAP_PROP_FOURCC: 0.0}, None, "it delivers an unknown format, not YUY2"),
            ({cv2.CAP_PROP_FRAME_HEIGHT: 576.0}, None, "it delivers 720 × 576, not 720 × 480"),
            ({}, bgr_frame(value=10), "levels from the BGR path's 10.0"),
        )
        for props, reference, message in cases:
            with self.subTest(message=message):
                session, _, log = self.open_v4l2(raw=list(raw), props=props,
                                                 reference=reference, fixed=tuple(props))
                self.assertEqual(session.luma_path, "BGR → grey")
                self.assertIn(message, log.lines[-1])

    def test_setting_forces_bgr(self):
        """ESPI_ANALOG_LUMA=bgr never turns conversion off.

        Args: None.
        Returns: None.
        """
        with mock.patch.dict(capture.os.environ, {capture.LUMA_SETTING: "bgr"}):
            session, cap, log = self.open_v4l2(raw=[yuy2_frame(ramp_luma())])
        self.assertNotIn((cv2.CAP_PROP_CONVERT_RGB, 0), cap.sets)
        self.assertEqual(session.luma_path, "BGR → grey")
        self.assertIn("(ESPI_ANALOG_LUMA=bgr)", log.lines[-1])

    def test_a_changed_frame_layout_is_a_grab_error(self):
        """A later frame that is no longer YUY2 of the size is an error status.

        Args: None.
        Returns: None.
        """
        luma = ramp_luma()
        session, _, _ = self.open_v4l2(raw=[yuy2_frame(luma), bgr_frame()])
        error, state = session.grab(100)
        self.assertEqual(state, "error")
        self.assertIn("not the raw YUY2", str(error))


class MediaFoundationRawLumaTests(CaptureTestCase):
    """Windows: DirectShow selects the input, Media Foundation hands over YUY2."""

    def setUp(self):
        """Pretend to run on Windows.

        Args: None.
        Returns: None.
        """
        patcher = mock.patch.object(capture.sys, "platform", "win32")
        patcher.start()
        self.addCleanup(patcher.stop)

    def open_windows(self, msmf=None, listed=None, enumerator_problem="",
                     dshow_frames=None):
        """Open a Rev E1 USB-Live2 through DirectShow, then Media Foundation.

        Args: msmf overrides the Media Foundation device's options; listed is
            what Media Foundation enumerates (default: the same stick at index
            3); enumerator_problem, if set, makes the enumerator unavailable;
            dshow_frames are the DirectShow frames.
        Returns: (session, factory, log, enumerator).
        """
        luma = ramp_luma()
        options = {"frames": [None], "raw_frames": [None, yuy2_frame(luma, "row")],
                   "props": {cv2.CAP_PROP_FOURCC: YUY2_CODE, cv2.CAP_PROP_FPS: 29.97,
                             **RAW_PROPS}}
        options.update(msmf or {})
        factory = self.use_devices(frames=dshow_frames or [reference_for(luma)],
                                   backends={MSMF: options})
        if listed is None:
            listed = [usb_live2_rev_e(MSMF, 0, "aaaaaaaa"),
                      usb_live2_rev_e(MSMF, 3, "e5323777")]
            listed[0].path = "\\\\?\\usb#vid_046d&pid_0825#x#{e5323777}"
            listed[0].vid, listed[0].pid = 0x046D, 0x0825
        enumerator = FakeEnumerator(listed)
        result = (None, enumerator_problem) if enumerator_problem else (enumerator, "")
        patcher = mock.patch.object(capture, "import_enumerator", return_value=result)
        patcher.start()
        self.addCleanup(patcher.stop)
        log = ListLog()
        session = capture.CaptureSession(log, capture_info(usb_live2_rev_e()))
        session.open()
        return session, factory, log, enumerator

    def test_raw_luma_through_media_foundation(self):
        """DirectShow routes composite and is released; Media Foundation reads Y.

        Args: None.
        Returns: None.
        """
        session, factory, log, enumerator = self.open_windows()
        dshow, msmf = factory.made
        self.assertEqual(enumerator.calls, [MSMF])
        self.assertTrue(dshow.released)
        self.assertEqual(dshow.sets[0], (cv2.CAP_PROP_CHANNEL, 0))
        self.assertEqual((msmf.index, msmf.backend, msmf.released), (3, MSMF, False))
        self.assertEqual(msmf.sets, [(cv2.CAP_PROP_CONVERT_RGB, 0),
                                     (cv2.CAP_PROP_FRAME_WIDTH, 720),
                                     (cv2.CAP_PROP_FRAME_HEIGHT, 480)])
        self.assertIs(session.cap, msmf)
        self.assertEqual(session.backend, MSMF)
        self.assertEqual(session.identity.backend, capture.backend_name(MSMF))
        self.assertEqual((session.identity.fourcc, session.identity.width,
                          session.identity.height), ("YUY2", 720, 480))
        self.assertAlmostEqual(session.identity.fps, 29.97)
        self.assertEqual(session.raw_layout, "1 × 691200, uint8")
        self.assertEqual(msmf.reads, 2)  # one empty read while the device starts
        frame, state = session.grab(100)
        self.assertEqual(state, "ok")
        self.assertTrue(np.array_equal(frame, capture.expand_video_range(ramp_luma())))
        self.assertIn("through " + capture.backend_name(MSMF), log.lines[-1])

    def test_matched_by_usb_id_when_paths_differ(self):
        """A unique USB ID matches when the paths do not.

        Args: None.
        Returns: None.
        """
        other = usb_live2_rev_e(MSMF, 5)
        other.path = "different"
        session, factory, _, _ = self.open_windows(listed=[other])
        self.assertEqual(factory.made[-1].index, 5)
        self.assertEqual(session.backend, MSMF)

    def test_unmatched_device_keeps_directshow_open(self):
        """An unlisted, ambiguous or unlistable device keeps the first DirectShow open.

        Args: None.
        Returns: None.
        """
        twins = [usb_live2_rev_e(MSMF, 1), usb_live2_rev_e(MSMF, 2)]
        for twin in twins:
            twin.path = ""
        cases = (({"listed": []}, "Media Foundation does not list this device"),
                 ({"listed": twins}, "several devices share its USB ID"),
                 ({"enumerator_problem": "is not installed"},
                  "cv2_enumerate_cameras is not installed"))
        for options, message in cases:
            with self.subTest(message=message):
                session, factory, log, _ = self.open_windows(**options)
                self.assertEqual(len(factory.made), 1)
                self.assertFalse(factory.made[0].released)
                self.assertEqual(session.backend, DSHOW_BASE)
                self.assertEqual(session.luma_path, "BGR → grey")
                self.assertIn("raw YUY2 through Media Foundation was not used: " + message,
                              log.lines[-1])

    def test_media_foundation_problems_reopen_directshow(self):
        """Not opening, no frame, a refused property or another layout re-opens DirectShow.

        Args: None.
        Returns: None.
        """
        cases = (({"opened": False}, "Media Foundation did not open it"),
                 ({"raw_frames": [None]}, "no raw frame arrived"),
                 ({"set_error": cv2.error("refused")}, "refused"),
                 ({"raw_frames": [bgr_frame()]}, "frame is not YUY2 of that size"))
        for msmf, message in cases:
            with self.subTest(message=message):
                session, factory, log, _ = self.open_windows(msmf=msmf)
                first, media_foundation, reopened = factory.made
                self.assertTrue(first.released and media_foundation.released)
                self.assertIs(session.cap, reopened)
                self.assertEqual(reopened.backend, DSHOW_BASE)
                self.assertEqual(session.backend, DSHOW_BASE)
                self.assertEqual(session.luma_path, "BGR → grey")
                self.assertIn(message, log.lines[-1])
                frame, state = session.grab(100)
                self.assertEqual((state, frame.shape), ("ok", (480, 720)))

    def test_failed_reopen_is_an_open_error(self):
        """When DirectShow cannot be re-opened, open() fails as before.

        Args: None.
        Returns: None.
        """
        reads = []

        def dshow():
            """Deliver the first frame, then nothing.

            Args: None.
            Returns: (ok, frame).
            """
            reads.append(1)
            return (True, reference_for(ramp_luma())) if len(reads) == 1 else (False, None)

        with self.assertRaises(RuntimeError) as caught:
            self.open_windows(msmf={"opened": False}, dshow_frames=dshow)
        self.assertIn("no video arrived", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
