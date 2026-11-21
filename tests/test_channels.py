"""Checks for espi_live.channels: per-camera engines and settings.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests

Real engines run on demo sources or on the in-memory fake PySpin; no camera,
display or real output folder is touched.
"""

import tempfile
import unittest
from dataclasses import replace
from unittest import mock

from espi_live import camera, capture, channels, paths, profiles
from espi_live.camera import CameraInfo
from espi_live.engine import Engine
from fake_capture import FakeCaptureFactory, FakeEnumerator, usb_live2, webcam
from fake_pyspin import FakeCamera, FakeSystem, make_pyspin
from test_engine_camera import ListLog, wait_until
from test_profiles import DictStore


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

    def test_level_methods_prefix_and_keep_the_level(self):
        """debug, info, warning, error and exception all gain the prefix.

        Args: None.
        Returns: None.
        """
        log = ListLog()
        prefixed = channels.PrefixedLog(log, "[1] ")
        prefixed.debug("d")
        prefixed.info("i")
        prefixed.warning("w")
        prefixed.error("e", exc_info=True)
        prefixed.exception("x")
        self.assertEqual([level for level, _, _ in log.records],
                         ["DEBUG", "INFO", "WARNING", "ERROR", "ERROR"])
        self.assertEqual(log.lines, ["[1] d", "[1] i", "[1] w", "[1] e", "[1] x"])
        self.assertEqual(log.records[3][2], True)


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

    def test_analog_channel_starts_in_the_labview_flow(self):
        """An analog channel starts in the LabVIEW flow and keeps its own settings.

        Args: None.
        Returns: None.
        """
        analog = self.channel(CameraInfo("video 1", "Hauppauge USB-Live2",
                                         kind="capture", vid=0x2040, pid=0xC200))
        self.assertEqual(analog.settings.mode, "Reference")
        self.assertTrue(analog.settings.auto_reference)
        self.assertEqual(analog.labview_saved["mode"], "Raw")
        self.assertFalse(analog.labview_saved["auto_reference"])
        self.assertIsNone(self.channel(CameraInfo("1", "BFS")).labview_saved)
        demo = self.channel(CameraInfo("video 1", "Hauppauge USB-Live2", kind="capture",
                                       vid=0x2040, pid=0xC200), demo=True)
        self.assertIsNone(demo.labview_saved)

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


class ProfileChannelTests(ChannelTestCase):
    """Each channel's own profile: snapshot, apply and restore at startup."""

    def setUp(self):
        """Add an in-memory profile store to the fake rig.

        Args: None.
        Returns: None.
        """
        super().setUp()
        self.store = DictStore()
        self.profiles = profiles.ProfileStore(self.store, self.log)

    def saved(self, info, demo=False, **values):
        """Store a profile for a camera, as an earlier run would have.

        Args: info is the CameraInfo; demo selects the demo namespace;
            values are Settings fields to change from its defaults.
        Returns: The stored Profile.
        """
        defaults = profiles.default_profile(info, demo)
        profile = replace(defaults, settings=replace(defaults.settings, **values))
        self.profiles.save_profile(profiles.profile_key(info, demo), profile)
        return profile

    def test_profile_is_a_snapshot_without_recording(self):
        """The running engine's settings and the LabVIEW state, never recording.

        Args: None.
        Returns: None.
        """
        demo = self.channel(CameraInfo("DEMO1"), demo=True)
        self.assertEqual(demo.profile_key, "profiles/demo/DEMO1")
        self.assertEqual(demo.profile(), demo.default_profile())
        demo.start()
        demo.engine.update_settings(gain_db=5.0, record=True)
        demo.labview_saved = {"gamma": 2.0}
        snapshot = demo.profile()
        self.assertEqual((snapshot.settings.gain_db, snapshot.settings.record), (5.0, False))
        demo.labview_saved["gamma"] = 3.0  # the snapshot keeps its own copy
        self.assertEqual(snapshot.labview_saved, {"gamma": 2.0})

    def test_apply_profile_changes_only_this_channel_and_keeps_recording(self):
        """A running engine gets the values; a recording keeps going.

        Args: None.
        Returns: None.
        """
        first = self.channel(CameraInfo("DEMO1"), demo=True)
        second = self.channel(CameraInfo("DEMO2"), demo=True)
        first.start()
        first.engine.update_settings(record=True)
        profile = profiles.default_profile(CameraInfo("x"), demo=True)
        profile = replace(profile, settings=replace(profile.settings, exposure_us=1500.0,
                                                    mode="Previous", record=False),
                          labview_saved={"gamma": 1.0})
        first.apply_profile(profile)
        s = first.engine.settings()
        self.assertEqual((s.exposure_us, s.mode, s.record), (1500.0, "Previous", True))
        self.assertEqual(first.labview_saved, {"gamma": 1.0})
        self.assertEqual(second.profile(), second.default_profile())
        second.apply_profile(profile)  # idle: kept for the next start
        self.assertEqual(second.settings.exposure_us, 1500.0)
        second.start()
        self.assertEqual(second.engine.settings().exposure_us, 1500.0)

    def test_a_given_profile_replaces_the_defaults(self):
        """Channel(profile=...) starts from it, also on an adopted engine.

        Args: None.
        Returns: None.
        """
        profile = self.saved(CameraInfo("DEMO1"), demo=True, gain_db=7.0)
        idle = self.channel(CameraInfo("DEMO1"), demo=True, profile=profile)
        self.assertEqual(idle.settings.gain_db, 7.0)
        engine = Engine(self.log)
        engine.use_demo()
        adopted = self.channel(CameraInfo("DEMO1"), demo=True, engine=engine,
                               profile=profile)
        self.assertEqual(adopted.engine.settings().gain_db, 7.0)

    def test_demo_channels_restore_their_own_profiles(self):
        """DEMO2's saved values come back for DEMO2 only, and are logged.

        Args: None.
        Returns: None.
        """
        self.saved(CameraInfo("DEMO2"), demo=True, gain_db=9.0, mode="Previous")
        found = channels.demo_channels(self.log, 2, self.profiles)
        self.channels.extend(found)
        self.assertEqual(found[0].engine.settings().gain_db, 0.0)
        self.assertEqual((found[1].settings.gain_db, found[1].settings.mode),
                         (9.0, "Previous"))
        self.assertTrue(self.log.has("Synthetic camera (DEMO2): saved settings restored "
                                     "(exposure 8000 µs, gain 9 dB, 30 fps, Previous view)."))
        self.assertFalse(self.log.has("Synthetic camera (DEMO1): saved settings restored"))

    def test_a_restored_camera_opens_with_its_saved_values(self):
        """The saved exposure, gain and rate reach the camera when it opens.

        Args: None.
        Returns: None.
        """
        self.saved(CameraInfo("1", "BFS"), exposure_us=12000.0, gain_db=4.0, fps=20.0)
        found = channels.open_camera_channels(self.log, self.profiles)
        self.channels.extend(found)
        nodes = self.system.camera_list.cameras[0].nodemap.nodes
        self.assertEqual((nodes["ExposureTime"].value, nodes["Gain"].value,
                          nodes["AcquisitionFrameRate"].value), (12000.0, 4.0, 20.0))
        self.assertEqual(found[1].settings.exposure_us, 8000.0)  # camera 2: defaults

    def test_a_demo_profile_never_lands_on_a_real_camera(self):
        """A demo profile under a real camera's serial is not used for it.

        Args: None.
        Returns: None.
        """
        self.saved(CameraInfo("1", "BFS"), demo=True, gain_db=9.0)
        found = channels.open_camera_channels(self.log, self.profiles)
        self.channels.extend(found)
        self.assertEqual(found[0].engine.settings().gain_db, 0.0)

    def test_an_analog_profile_restores_the_labview_state(self):
        """Turned off last time, the analog camera starts with the flow off.

        Args: None.
        Returns: None.
        """
        analog = CameraInfo("video 1", "Hauppauge USB-Live2", kind="capture",
                            vid=0x2040, pid=0xC200)
        own = profiles.default_profile(analog).labview_saved
        self.profiles.save_profile(profiles.profile_key(analog), profiles.Profile(
            replace(profiles.default_profile(analog).settings, **own)))
        restored = channels.restore_channel(self.log, analog, profiles=self.profiles)
        self.assertIsNone(restored.labview_saved)
        self.assertEqual(restored.settings.mode, "Raw")
        self.assertTrue(self.log.has("saved settings restored"))
        fresh = channels.restore_channel(self.log, analog, profiles=None)
        self.assertIsNotNone(fresh.labview_saved)


class ReportingStartupTests(ChannelTestCase):
    """Camera identifiers reach reporting before startup can fail."""

    def test_startup_failure_scrubs_discovered_serials(self):
        """An exception while opening cameras cannot publish enumeration serials.

        Args: None.
        Returns: None.
        """
        from espi_live import error_reports
        infos = [CameraInfo('PRIVATE-SERIAL-111', 'Blackfly'),
                 CameraInfo('PRIVATE-SERIAL-222', 'Blackfly')]
        reporter = error_reports.ErrorReporter(
            enabled=True, folder=paths.output_folder('logs'), log_tail=lambda: self.log.lines)
        with mock.patch.object(error_reports, '_reporter', reporter), \
                mock.patch.object(channels.sources, 'detect_cameras', return_value=infos), \
                mock.patch.object(channels.Channel, 'open', side_effect=ValueError('startup bug')):
            with self.assertRaises(ValueError):
                channels.open_camera_channels(self.log)
        self.assertEqual(len(reporter.reports), 1)
        body = next(iter(reporter.reports.values())).body
        self.assertIn('<serial>', body)
        for info in infos:
            self.assertNotIn(info.serial, body)

    def test_opened_identity_is_registered_before_logging(self):
        """Successful fake-camera startup registers the attached identity too.

        Args: None.
        Returns: None.
        """
        from espi_live import error_reports
        reporter = error_reports.ErrorReporter()
        with mock.patch.object(error_reports, '_reporter', reporter):
            self.channels = channels.open_camera_channels(self.log)
        self.assertEqual(set(reporter.context['serials']), {'1', '2'})
