"""Tests for choosing between FLIR cameras and capture devices (no hardware)."""

import unittest
from unittest import mock

from espi_live import camera, capture, iphone, sources
from espi_live.camera import CameraInfo, CameraSession
from espi_live.capture import CaptureSession
from espi_live.iphone import IPhoneSession
from fake_capture import FakeCaptureFactory, FakeEnumerator, ListLog, usb_live2, webcam
from fake_pyspin import FakeCamera, FakeSystem, make_pyspin


def capture_info(enumerated, position):
    """Turn an enumerated fake device into a capture CameraInfo.

    Args: enumerated is a fake_capture device; position is its list place.
    Returns: A CameraInfo of kind "capture".
    """
    return CameraInfo(f"video {position}", enumerated.name, kind="capture",
                      index=enumerated.index, backend=enumerated.backend,
                      vid=enumerated.vid, pid=enumerated.pid, path=enumerated.path)


PHONE = CameraInfo("iphone-4d5e801e", "iPhone (USB)", kind="iphone",
                   path="00008110-001A2B3C4D5E801E")


class SourcesTestCase(unittest.TestCase):
    """Base class: a fake PySpin, fake video devices and a log."""

    def setUp(self):
        """Install one fake FLIR camera, no video devices and a fake OpenCV.

        Args: None.
        Returns: None.
        """
        self.log = ListLog()
        self.system = FakeSystem([FakeCamera(serial="1", model="BFS")])
        self.enumerator = FakeEnumerator()
        self.factory = FakeCaptureFactory()
        self.phones = []  # iPhones over USB unless a test adds some
        for patcher in (mock.patch.object(camera, "PySpin", make_pyspin(self.system)),
                        mock.patch.object(capture, "load_enumerator",
                                          return_value=self.enumerator),
                        mock.patch.object(capture.cv2, "VideoCapture", self.factory),
                        mock.patch.object(iphone, "list_iphones",
                                          side_effect=lambda log: list(self.phones))):
            patcher.start()
            self.addCleanup(patcher.stop)


class DetectTests(SourcesTestCase):
    """List every camera, FLIR first."""

    def test_detect_merges_spinnaker_then_capture(self):
        """FLIR cameras come first, then video devices.

        Args: None.
        Returns: None.
        """
        self.enumerator.devices = [webcam(0), usb_live2(1)]
        infos = sources.detect_cameras(self.log)
        self.assertEqual([(info.kind, info.serial) for info in infos],
                         [("spinnaker", "1"), ("capture", "video 1"),
                          ("capture", "video 2")])
        self.assertEqual(self.factory.made, [])  # listing opens nothing

    def test_detect_survives_missing_pyspin(self):
        """Without PySpin only the video devices are listed, and that is logged.

        Args: None.
        Returns: None.
        """
        self.enumerator.devices = [usb_live2()]
        with mock.patch.object(camera, "list_cameras",
                               side_effect=ImportError("No module named 'PySpin'")):
            infos = sources.detect_cameras(self.log)
        self.assertEqual([info.kind for info in infos], ["capture"])
        self.assertTrue(any("PySpin" in line for line in self.log.lines))

    def test_detect_logs_flir_errors(self):
        """A FLIR listing failure is logged and leaves the video devices.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(camera, "list_cameras", side_effect=RuntimeError("bus")):
            self.assertEqual(sources.detect_cameras(self.log), [])
        self.assertIn("Could not list FLIR cameras: bus", self.log.lines)


class ChoiceTests(SourcesTestCase):
    """Support, open order and session type."""

    def test_is_supported(self):
        """FLIR cameras and recognised adapters are supported; webcams are not.

        Args: None.
        Returns: None.
        """
        self.assertTrue(sources.is_supported(CameraInfo("1", "BFS")))
        self.assertTrue(sources.is_supported(capture_info(usb_live2(), 1)))
        self.assertFalse(sources.is_supported(capture_info(webcam(), 1)))

    def test_auto_open_order_skips_unknown(self):
        """Startup tries FLIR cameras, then recognised adapters, never webcams.

        Args: None.
        Returns: None.
        """
        cam, live2, other = (CameraInfo("1", "BFS"), capture_info(usb_live2(1), 2),
                             capture_info(webcam(0), 1))
        self.assertEqual(sources.auto_open_order([other, live2, cam]), [cam, live2])

    def test_make_session(self):
        """Each kind gets its own session class.

        Args: None.
        Returns: None.
        """
        video = sources.make_session(self.log, capture_info(usb_live2(), 1))
        self.assertIsInstance(video, CaptureSession)
        flir = sources.make_session(self.log, CameraInfo("1", "BFS"))
        self.assertIsInstance(flir, CameraSession)
        self.assertEqual(flir.serial, "1")
        phone = sources.make_session(self.log, PHONE, 500.0, 6.0, 15.0)
        self.assertIsInstance(phone, IPhoneSession)
        self.assertEqual(phone.requested, {"exposure_us": 500.0, "gain_db": 6.0,
                                           "fps": 15.0})

    def test_iphones_are_listed_supported_and_tried_last(self):
        """iPhones come after FLIR cameras and adapters, and are supported.

        Args: None.
        Returns: None.
        """
        self.phones = [PHONE]
        self.enumerator.devices = [usb_live2(0)]
        infos = sources.detect_cameras(self.log)
        self.assertEqual([info.kind for info in infos], ["spinnaker", "capture", "iphone"])
        self.assertTrue(sources.is_supported(PHONE))
        self.assertEqual([info.kind for info in sources.auto_open_order(infos[::-1])],
                         ["spinnaker", "capture", "iphone"])


LENS_PHONES = [CameraInfo("iphone-4d5e801e", "iPhone Wide 1× (USB)", kind="iphone",
                          path=PHONE.path, lens="wide"),
               CameraInfo("iphone-4d5e801e-ultra-wide", "iPhone Ultra Wide 0.5× (USB)",
                          kind="iphone", path=PHONE.path, lens="ultra_wide"),
               CameraInfo("iphone-4d5e801e-telephoto", "iPhone Telephoto 5× (USB)",
                          kind="iphone", path=PHONE.path, lens="telephoto")]


class PhoneLensTests(SourcesTestCase):
    """A phone with several back cameras opens only its main one unasked."""

    def test_startup_order_keeps_one_camera_per_phone(self):
        """Every lens is listed, but only the main one may open without being asked.

        Args: None.
        Returns: None.
        """
        self.phones = LENS_PHONES
        infos = sources.detect_cameras(self.log)
        self.assertEqual([info.lens for info in infos if info.kind == "iphone"],
                         ["wide", "ultra_wide", "telephoto"])
        self.assertEqual([info.serial for info in sources.auto_open_order(infos)
                          if info.kind == "iphone"], ["iphone-4d5e801e"])

    def test_leaving_demo_mode_tries_only_the_main_camera(self):
        """A failed phone is not retried once per lens.

        Args: None.
        Returns: None.
        """
        self.system.camera_list.cameras.clear()
        self.phones = LENS_PHONES
        tried = []

        def refuse(session):
            """Record the camera tried, then fail as an app that is closed would.

            Args: session is the IPhoneSession.
            Returns: None. Raises RuntimeError.
            """
            tried.append(session.info.lens)
            raise RuntimeError("app is not open")

        with mock.patch.object(IPhoneSession, "open", refuse), \
                self.assertRaisesRegex(RuntimeError, "app is not open"):
            sources.open_first_live(self.log)
        self.assertEqual(tried, ["wide"])


class OpenFirstLiveTests(SourcesTestCase):
    """Switching from demo mode to the first usable camera."""

    def test_falls_back_to_iphone(self):
        """Without a FLIR camera or adapter, a plugged-in iPhone is tried.

        Args: None.
        Returns: None.
        """
        self.system.camera_list.cameras.clear()
        self.phones = [PHONE]
        opened = []
        with mock.patch.object(IPhoneSession, "open",
                               lambda session: opened.append(session.info)):
            session = sources.open_first_live(self.log)
        self.assertIsInstance(session, IPhoneSession)
        self.assertEqual(opened, [PHONE])

    def test_prefers_flir(self):
        """A FLIR camera wins over a capture adapter.

        Args: None.
        Returns: None.
        """
        self.enumerator.devices = [usb_live2()]
        session = sources.open_first_live(self.log)
        self.addCleanup(session.close)
        self.assertIsInstance(session, CameraSession)
        self.assertEqual(self.factory.made, [])

    def test_falls_back_to_usb_live2_and_skips_webcams(self):
        """Without a FLIR camera the recognised adapter opens; webcams never do.

        Args: None.
        Returns: None.
        """
        self.system.camera_list.cameras.clear()
        self.enumerator.devices = [webcam(0), usb_live2(1)]
        session = sources.open_first_live(self.log)
        self.addCleanup(session.close)
        self.assertIsInstance(session, CaptureSession)
        self.assertEqual([made.index for made in self.factory.made], [usb_live2(1).index])

    def test_nothing_raises_the_flir_error(self):
        """With nothing usable, the FLIR camera's error is raised.

        Args: None.
        Returns: None.
        """
        self.system.camera_list.cameras.clear()
        self.enumerator.devices = [webcam()]
        with self.assertRaisesRegex(RuntimeError, "No FLIR"):
            sources.open_first_live(self.log)
        self.assertEqual(self.factory.made, [])

    def test_failing_adapter_is_skipped(self):
        """An adapter that cannot open is passed over for the next one.

        Args: None.
        Returns: None.
        """
        self.system.camera_list.cameras.clear()
        self.enumerator.devices = [usb_live2(0), usb_live2(1)]
        self.factory.opened = False
        with self.assertRaisesRegex(RuntimeError, "WinTV"):
            sources.open_first_live(self.log)
        self.assertEqual(len(self.factory.made), 2)
        attempts = [line for line in self.log.lines if line.startswith("Could not open")]
        self.assertEqual(len(attempts), 3)  # the FLIR camera, then both adapters
        self.assertTrue(any("Could not open the FLIR camera: No FLIR" in line
                            for line in self.log.lines))
        self.assertTrue(any("Hauppauge USB-Live2 (video 1)" in line
                            for line in self.log.lines))
        self.assertTrue(any("Hauppauge USB-Live2 (video 2)" in line
                            for line in self.log.lines))


if __name__ == "__main__":
    unittest.main()
