"""Tests for choosing between FLIR cameras and capture devices (no hardware)."""

import unittest
from unittest import mock

from espi_live import camera, capture, sources
from espi_live.camera import CameraInfo, CameraSession
from espi_live.capture import CaptureSession
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
        for patcher in (mock.patch.object(camera, "PySpin", make_pyspin(self.system)),
                        mock.patch.object(capture, "load_enumerator",
                                          return_value=self.enumerator),
                        mock.patch.object(capture.cv2, "VideoCapture", self.factory)):
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


class OpenFirstLiveTests(SourcesTestCase):
    """Switching from demo mode to the first usable camera."""

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
