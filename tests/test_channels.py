"""Checks for espi_live.channels: per-camera engines and settings.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests

Real engines run on demo sources or on the in-memory fake PySpin; no camera,
display or real output folder is touched.
"""

import tempfile
import unittest
from unittest import mock

from espi_live import camera, capture, channels, paths
from espi_live.camera import CameraInfo
from espi_live.engine import Engine
from fake_capture import FakeCaptureFactory, FakeEnumerator, usb_live2, webcam
from fake_pyspin import FakeCamera, FakeSystem, make_pyspin
from test_engine_camera import ListLog, wait_until


class ChannelTestCase(unittest.TestCase):
    """Base class: temporary outputs, a fake two-camera rig and cleanup."""

    def setUp(self):
        """Redirect outputs and install a fake PySpin with two cameras.

        Args: None.
        Returns: None.
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for patcher in (mock.patch.object(paths, "OUTPUT_DIR", tmp.name),):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.log = ListLog()
        self.system = FakeSystem([FakeCamera(serial="1", model="BFS"),
                                  FakeCamera(serial="2", model="BFS")])
        self.enumerator = FakeEnumerator()  # no video devices unless a test adds some
        self.factory = FakeCaptureFactory()
        for patcher in (mock.patch.object(camera, "PySpin", make_pyspin(self.system)),
                        mock.patch.object(capture, "load_enumerator",
                                          return_value=self.enumerator),
                        mock.patch.object(capture.cv2, "VideoCapture", self.factory)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.channels = []
        self.addCleanup(self.stop_all)

    def stop_all(self):
        """Stop every channel made by a test and wait for its engines.

        Args: None.
        Returns: None.
        """
        for channel in self.channels:
            for engine in channel.engines():
                engine.stop()
                engine.wait(10)

    def channel(self, *args, **kwargs):
        """Create a channel that is cleaned up after the test.

        Args: args and kwargs are passed to Channel.
        Returns: The Channel.
        """
        made = channels.Channel(self.log, *args, **kwargs)
        self.channels.append(made)
        return made


class PrefixedLogTests(unittest.TestCase):
    """PrefixedLog."""

    def test_prefix_is_added_and_can_change(self):
        """Every line gets the current prefix.

        Args: None.
        Returns: None.
        """
        log = ListLog()
        prefixed = channels.PrefixedLog(log, "[1] ")
        prefixed.write("a")
        prefixed.prefix = ""
        prefixed.write("b")
        self.assertEqual(log.lines, ["[1] a", "b"])


class ChannelTests(ChannelTestCase):
    """Channel state, start/stop and settings."""

    def test_demo_infos_are_numbered_and_at_least_one(self):
        """demo_infos names DEMO1..N and never returns an empty list.

        Args: None.
        Returns: None.
        """
        self.assertEqual([info.serial for info in channels.demo_infos(2)],
                         ["DEMO1", "DEMO2"])
        self.assertEqual(len(channels.demo_infos(0)), 1)

    def test_idle_channel_reports_its_source(self):
        """An idle channel is inactive and reports demo from its source kind.

        Args: None.
        Returns: None.
        """
        idle = self.channel(CameraInfo("1", "BFS"))
        self.assertFalse(idle.active)
        self.assertFalse(idle.demo)
        self.assertEqual(idle.label, "BFS (1)")
        self.assertTrue(self.channel(CameraInfo("DEMO1"), demo=True).demo)
        self.assertEqual(idle.engines(), [])

    def test_adopted_engine_supplies_settings(self):
        """An engine passed in becomes the running engine with its settings.

        Args: None.
        Returns: None.
        """
        engine = Engine(self.log)
        engine.use_demo()
        engine.update_settings(gain_db=3.0)
        adopted = self.channel(CameraInfo("DEMO1"), demo=True, engine=engine)
        self.assertTrue(adopted.active and adopted.demo)
        self.assertEqual(adopted.settings.gain_db, 3.0)

    def test_open_opens_the_selected_camera_without_starting(self):
        """open() opens the channel's camera on the calling thread.

        Args: None.
        Returns: None.
        """
        second = self.channel(CameraInfo("2", "BFS"))
        second.open()
        self.assertEqual(second.engine.camera_info.serial, "2")
        self.assertEqual(second.engine.tag, "2")
        self.assertIsNone(second.engine._thread)

    def test_stop_keeps_settings_and_start_restores_them(self):
        """Settings survive a stop/start; recording does not restart.

        Args: None.
        Returns: None.
        """
        demo = self.channel(CameraInfo("DEMO1"), demo=True)
        demo.start()
        demo.start()  # already running: no second engine
        first = demo.engine
        first.update_settings(exposure_us=1234.0, mode="Previous", record=True)
        demo.stop()
        demo.stop()  # already stopped: nothing to do
        self.assertFalse(demo.active)
        self.assertEqual((demo.settings.exposure_us, demo.settings.record), (1234.0, False))
        demo.start()
        self.assertIsNot(demo.engine, first)
        s = demo.engine.settings()
        self.assertEqual((s.exposure_us, s.mode, s.record), (1234.0, "Previous", False))
        self.assertTrue(wait_until(lambda: demo.engine.latest_packet() is not None))
        self.assertEqual(demo.engines(), [first, demo.engine])
        self.assertTrue(demo.wait(10))
        self.assertEqual(demo.retired, [])

    def test_stop_drops_a_frozen_frame(self):
        """A camera turned off forgets the frame its display was frozen on.

        Args: None.
        Returns: None.
        """
        demo = self.channel(CameraInfo("DEMO1"), demo=True)
        self.assertIsNone(demo.frozen)
        demo.start()
        demo.frozen, demo.frozen_at = "packet", "when"
        demo.stop()
        self.assertIsNone(demo.frozen)
        self.assertIsNone(demo.frozen_at)
        self.assertTrue(demo.wait(10))

    def test_restarted_camera_waits_for_its_previous_engine(self):
        """A camera turned back on opens after the old engine released it.

        Args: None.
        Returns: None.
        """
        cam = self.channel(CameraInfo("1", "BFS"))
        cam.start()
        old = cam.engine
        self.assertTrue(wait_until(lambda: old.latest_packet() is not None))
        cam.stop()
        with mock.patch.object(Engine, "open_on_start", autospec=True) as deferred:
            deferred.side_effect = lambda engine, after=None: setattr(
                engine, "_open_after", (after,))
            cam.start()
        self.assertIs(deferred.call_args.args[1], old)
        self.assertTrue(wait_until(lambda: cam.engine.latest_packet() is not None))

    def test_wait_reports_engines_still_running(self):
        """wait() keeps engines that have not finished.

        Args: None.
        Returns: None.
        """
        stuck = mock.Mock()
        stuck.wait.return_value = False
        idle = self.channel(CameraInfo("1"))
        idle.retired = [stuck]
        self.assertFalse(idle.wait(0.01))
        self.assertEqual(idle.retired, [stuck])


class StartupHelperTests(ChannelTestCase):
    """open_camera_channels and demo_channels."""

    def test_every_camera_gets_a_channel_and_the_first_opens(self):
        """Two cameras: both listed, the first opened, the count logged.

        Args: None.
        Returns: None.
        """
        found = channels.open_camera_channels(self.log)
        self.channels.extend(found)
        self.assertEqual([c.info.serial for c in found], ["1", "2"])
        self.assertEqual([c.active for c in found], [True, False])
        self.assertTrue(self.log.has("2 cameras detected: BFS (1), BFS (2)"))

    def test_a_failing_camera_is_skipped(self):
        """If the first camera fails to open the next one is used.

        Args: None.
        Returns: None.
        """
        self.system.camera_list.cameras[0].fail.add("Init")
        found = channels.open_camera_channels(self.log)
        self.channels.extend(found)
        self.assertEqual([c.active for c in found], [False, True])
        self.assertTrue(self.log.has("Could not open BFS (1): Init failed"))

    def test_no_camera_or_none_opening_raises(self):
        """No camera, or no camera that opens, raises for the demo prompt.

        Args: None.
        Returns: None.
        """
        for cam in self.system.camera_list.cameras:
            cam.fail.add("Init")
        with self.assertRaisesRegex(RuntimeError, "Init failed"):
            channels.open_camera_channels(self.log)
        self.system.camera_list.cameras.clear()
        with self.assertRaisesRegex(RuntimeError, "No FLIR"):
            channels.open_camera_channels(self.log)

    def test_startup_opens_usb_live2_without_blackfly(self):
        """With no FLIR camera the USB-Live2 opens; a webcam is listed but left off.

        Args: None.
        Returns: None.
        """
        self.system.camera_list.cameras.clear()
        self.enumerator.devices = [webcam(0), usb_live2(1)]
        found = channels.open_camera_channels(self.log)
        self.channels.extend(found)
        self.assertEqual([c.info.serial for c in found], ["video 1", "video 2"])
        self.assertEqual([c.active for c in found], [False, True])
        self.assertEqual([made.index for made in self.factory.made], [usb_live2(1).index])
        self.assertEqual(found[1].engine.camera_info.kind, "capture")

    def test_startup_with_only_webcam_raises(self):
        """A webcam alone is never opened at startup; demo mode is offered.

        Args: None.
        Returns: None.
        """
        self.system.camera_list.cameras.clear()
        self.enumerator.devices = [webcam()]
        with self.assertRaisesRegex(
                RuntimeError, "No FLIR camera or recognised capture adapter detected"):
            channels.open_camera_channels(self.log)
        self.assertEqual(self.factory.made, [])

    def test_channel_supported(self):
        """FLIR cameras, recognised adapters and demo sources are supported.

        Args: None.
        Returns: None.
        """
        def info(enumerated):
            """Build a capture CameraInfo for a fake enumerated device.

            Args: enumerated is a fake_capture device.
            Returns: A CameraInfo of kind "capture".
            """
            return CameraInfo("video 1", enumerated.name, kind="capture",
                              index=enumerated.index, backend=enumerated.backend,
                              vid=enumerated.vid, pid=enumerated.pid)

        self.assertTrue(self.channel(CameraInfo("1", "BFS")).supported)
        self.assertTrue(self.channel(info(usb_live2())).supported)
        self.assertFalse(self.channel(info(webcam())).supported)
        self.assertTrue(self.channel(CameraInfo("dummy 1"), demo=True).supported)

    def test_demo_channels_open_only_the_first(self):
        """demo_channels makes N synthetic cameras with the first opened.

        Args: None.
        Returns: None.
        """
        found = channels.demo_channels(self.log, 3)
        self.channels.extend(found)
        self.assertEqual([c.active for c in found], [True, False, False])
        self.assertTrue(found[0].engine.demo)
