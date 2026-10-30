"""Engine checks that use the fake PySpin camera and the recorder paths.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import queue
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

from espi_live import camera, capture, engine, paths
from espi_live.camera import CameraInfo
from espi_live.engine import Engine, WindowsPreviewTimer
from fake_capture import (FakeCaptureFactory, FakeEnumerator, bgr_frame,
                          interlaced_frame, usb_live2, webcam)
from fake_pyspin import (FakeCamera, FakeImage, FakeSpinnakerException,
                         FakeSystem, make_pyspin)


class ListLog:
    """Collect log messages in memory."""

    def __init__(self):
        """Create an empty log.

        Args: None.
        Returns: None.
        """
        self.lines = []

    def write(self, message):
        """Record one message.

        Args: message is the text to keep.
        Returns: None.
        """
        self.lines.append(message)

    def has(self, text):
        """Check whether any logged line contains text.

        Args: text is the substring to look for.
        Returns: True if some line contains it.
        """
        return any(text in line for line in self.lines)


def wait_until(predicate, timeout=5.0):
    """Poll a condition until it is true or a timeout passes.

    Args: predicate is a zero-argument callable; timeout is in seconds.
    Returns: The final value of predicate().
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class FakeRecorder:
    """A recorder double with controllable completion."""

    def __init__(self, finished=True, error=None):
        """Create the double.

        Args: finished sets whether done is already set; error is the
            recorder error to expose.
        Returns: None.
        """
        self.done = threading.Event()
        if finished:
            self.done.set()
        self.error = error
        self.path = "/tmp/fake.avi"
        self.stopped = False
        self.frames = []

    def stop(self):
        """Record that a stop was requested.

        Args: None.
        Returns: None.
        """
        self.stopped = True

    def submit(self, frame):
        """Keep a submitted frame.

        Args: frame is the image.
        Returns: None.
        """
        self.frames.append(frame)

    def summary(self):
        """Describe the recording.

        Args: None.
        Returns: A fixed summary string.
        """
        return "fake summary"


class CameraEngineCase(unittest.TestCase):
    """Base class: temporary outputs, a fake camera rig and cleanup."""

    def setUp(self):
        """Redirect outputs, install the fake PySpin and create an engine.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.saved_output = paths.OUTPUT_DIR
        paths.OUTPUT_DIR = self.tmp.name
        self.log = ListLog()
        self.cam = FakeCamera()
        self.system = FakeSystem([self.cam])
        self.enumerator = FakeEnumerator()  # no video devices unless a test adds some
        self.factory = FakeCaptureFactory()
        for patcher in (mock.patch.object(camera, "PySpin", make_pyspin(self.system)),
                        mock.patch.object(capture, "load_enumerator",
                                          return_value=self.enumerator),
                        mock.patch.object(capture.cv2, "VideoCapture", self.factory)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.engine = Engine(self.log)
        self.addCleanup(self.shutdown)

    def shutdown(self):
        """Stop the engine and restore the output folder.

        Args: None.
        Returns: None.
        """
        self.engine.stop()
        self.engine.wait(10)
        paths.OUTPUT_DIR = self.saved_output
        self.tmp.cleanup()

    def events(self):
        """Drain the engine's event queue.

        Args: None.
        Returns: A list of (kind, payload) tuples.
        """
        drained = []
        while True:
            try:
                drained.append(self.engine.events.get_nowait())
            except queue.Empty:
                return drained

    def wait_event(self, kind, timeout=5.0):
        """Wait for an event kind, discarding the others seen on the way.

        Args: kind is the event name; timeout is in seconds.
        Returns: The payload, or raises AssertionError on timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                got, payload = self.engine.events.get(timeout=0.05)
            except queue.Empty:
                continue
            if got == kind:
                return payload
        raise AssertionError(f"no {kind} event")


class WindowsTimerTests(unittest.TestCase):
    """WindowsPreviewTimer with a mocked OS layer."""

    def test_not_windows_does_nothing(self):
        """Off Windows the timer is never requested.

        Args: None.
        Returns: None.
        """
        timer = WindowsPreviewTimer()
        with mock.patch.object(engine.os, "name", "posix"):
            self.assertFalse(timer.start())
        timer.stop()
        self.assertFalse(timer.active)

    def test_windows_requests_and_restores_resolution(self):
        """On Windows timeBeginPeriod(1) is requested and later undone.

        Args: None.
        Returns: None.
        """
        winmm = mock.Mock()
        winmm.timeBeginPeriod.return_value = 0
        fake_ctypes = types.SimpleNamespace(WinDLL=mock.Mock(return_value=winmm))
        timer = WindowsPreviewTimer()
        with mock.patch.object(engine.os, "name", "nt"), \
                mock.patch.object(engine, "ctypes", fake_ctypes):
            self.assertTrue(timer.start())
        timer.stop()
        winmm.timeBeginPeriod.assert_called_once_with(1)
        winmm.timeEndPeriod.assert_called_once_with(1)
        self.assertFalse(timer.active)

    def test_windows_failure_to_load_winmm_is_tolerated(self):
        """An OSError loading winmm leaves the timer inactive.

        Args: None.
        Returns: None.
        """
        fake_ctypes = types.SimpleNamespace(
            WinDLL=mock.Mock(side_effect=OSError("no winmm")))
        timer = WindowsPreviewTimer()
        with mock.patch.object(engine.os, "name", "nt"), \
                mock.patch.object(engine, "ctypes", fake_ctypes):
            self.assertFalse(timer.start())
        self.assertIsNone(timer.winmm)


class SetupTests(CameraEngineCase):
    """Source selection before start()."""

    def test_start_without_source_raises(self):
        """start() refuses to run with no camera or demo source.

        Args: None.
        Returns: None.
        """
        with self.assertRaises(RuntimeError):
            self.engine.start()

    def test_open_camera_publishes_limits_and_status(self):
        """A successful open attaches the camera and emits its model and limits.

        Args: None.
        Returns: None.
        """
        self.engine.open_camera()
        self.assertFalse(self.engine.demo)
        self.assertIn("Applied: 8000", self.engine.status()["camera"])
        (kind, identity), (kind2, limits) = self.events()[:2]
        self.assertEqual(kind, "camera_identity")
        self.assertEqual(identity.short_name(), "BFS-U3-63S4M")
        self.assertEqual(identity.serial, "21234567")
        self.assertEqual(kind2, "camera_limits")
        self.assertIn("Gain", limits)

    def test_open_camera_writes_saved_exposure_gain_and_frame_rate(self):
        """Opening applies the engine's settings, not the hard-coded defaults.

        Args: None.
        Returns: None.
        """
        self.engine.update_settings(exposure_us=5000.0, gain_db=1.5, fps=15.0)
        self.engine.open_camera()
        nodes = self.cam.nodemap.nodes
        self.assertEqual(nodes["ExposureTime"].value, 5000.0)
        self.assertEqual(nodes["Gain"].value, 1.5)
        self.assertEqual(nodes["AcquisitionFrameRate"].value, 15.0)

    def test_open_camera_logs_model_notes(self):
        """An untested colour model is named in the log with its notes.

        Args: None.
        Returns: None.
        """
        self.cam.tl_device_nodemap.nodes["DeviceModelName"].value = (
            "Blackfly S BFS-U3-70S7C")
        with mock.patch.object(self.engine.log, "write") as write:
            self.engine.open_camera()
        text = "\n".join(call.args[0] for call in write.call_args_list)
        self.assertIn("Camera: Blackfly S BFS-U3-70S7C (FLIR)", text)
        self.assertIn("Camera note: Colour camera", text)
        self.assertIn("BFS-U3-70S7C has not been tested", text)

    def test_open_camera_failure_leaves_engine_unchanged(self):
        """No camera raises and the engine may still choose demo.

        Args: None.
        Returns: None.
        """
        self.system.camera_list.cameras.clear()
        with self.assertRaises(RuntimeError):
            self.engine.open_camera()
        self.assertIsNone(self.engine.camera)
        self.engine.use_demo()
        self.assertTrue(self.engine.demo)

    def test_wait_closes_camera_that_was_never_started(self):
        """wait() releases a camera opened but never started.

        Args: None.
        Returns: None.
        """
        self.engine.open_camera()
        self.assertTrue(self.engine.wait(1))
        self.assertIsNone(self.engine.camera)
        self.assertTrue(self.system.released)

    def test_start_logs_when_windows_timer_accepted(self):
        """A granted timer request is logged on start.

        Args: None.
        Returns: None.
        """
        self.engine.open_camera()
        self.engine.preview_timer = mock.Mock(start=mock.Mock(return_value=True))
        self.engine.start()
        self.assertTrue(self.log.has("Windows timer: 1 ms resolution"))

    def test_update_settings_rejects_unknown_field(self):
        """Unknown settings names raise TypeError and change nothing.

        Args: None.
        Returns: None.
        """
        with self.assertRaises(TypeError):
            self.engine.update_settings(bogus=1)
        self.engine.update_settings(gain_db=2.0)
        self.assertEqual(self.engine.settings().gain_db, 2.0)


class CameraStreamingTests(CameraEngineCase):
    """The worker loop with a fake camera."""

    def start_camera(self):
        """Open the fake camera and start the worker.

        Args: None.
        Returns: None.
        """
        self.engine.open_camera()
        self.engine.start()

    def test_raw_frames_publish_and_status_updates(self):
        """Camera frames are published unwatermarked with live status.

        Args: None.
        Returns: None.
        """
        self.start_camera()
        self.assertTrue(wait_until(lambda: self.engine.latest_packet() is not None))
        packet = self.engine.latest_packet()
        self.assertFalse(packet.demo)
        self.assertEqual(packet.image.shape, (48, 64))
        self.assertFalse(self.engine.status()["demo"])

    def test_errors_incomplete_frames_readback_and_setting_changes(self):
        """Grab failures are logged and throttled, then streaming recovers.

        Args: None.
        Returns: None.
        """
        good = self.cam.frame_source
        begin = time.monotonic()
        counter = [0]

        def flaky():
            """Fail for just over a second, alternating failure kinds.

            Args: None.
            Returns: A good image once the failure window has passed.
            """
            if time.monotonic() - begin < 1.1:
                counter[0] += 1
                if counter[0] % 2:
                    return FakeImage(np.zeros((4, 4), np.uint8), incomplete=True)
                raise FakeSpinnakerException("timeout")
            return good()
        self.cam.frame_source = flaky
        self.start_camera()
        self.assertTrue(wait_until(
            lambda: self.engine.status()["processing"] ==
            "Waiting for camera frame", 3))
        self.assertTrue(wait_until(lambda: self.engine.latest_packet() is not None, 3))
        self.engine.update_settings(exposure_us=5000.0, gain_db=3.0, fps=20.0)
        nodes = self.cam.nodemap.nodes
        self.assertTrue(wait_until(lambda: (
            nodes["ExposureTime"].value == 5000.0 and
            nodes["Gain"].value == 3.0 and
            nodes["AcquisitionFrameRate"].value == 20.0)))
        self.assertTrue(self.log.has("Incomplete frames: 1"))
        self.assertTrue(self.log.has("Camera timeout or acquisition error: timeout"))
        self.assertTrue(wait_until(
            lambda: "Camera: 29.5 fps" == self.engine.status()["camera_fps"], 3))

    def test_readback_reports_na_when_resulting_rate_unreadable(self):
        """An unreadable resulting rate shows "Camera: n/a".

        Args: None.
        Returns: None.
        """
        self.cam.nodemap.nodes["AcquisitionResultingFrameRate"].readable = False
        with mock.patch.object(engine, "READBACK_INTERVAL_S", 0.05):
            self.start_camera()
            self.assertTrue(wait_until(
                lambda: self.engine.status()["camera_fps"] == "Camera: n/a"))

    def test_non_mono8_frame_is_fatal_and_releases_camera(self):
        """A non-Mono8 frame stops the worker with a fatal event.

        Args: None.
        Returns: None.
        """
        self.cam.frame_source = lambda: FakeImage(np.zeros((4, 4), np.uint16))
        with mock.patch.object(engine.traceback, "print_exc"):
            self.start_camera()
            payload = self.wait_event("fatal")
            self.wait_event("stopped")
        self.assertIn("Expected a Mono8 image", payload)
        self.assertTrue(self.log.has("Stopped: RuntimeError"))
        self.assertTrue(self.system.released)
        self.assertIsNone(self.engine.camera)

    def test_measurement_modes_work_on_camera_frames(self):
        """Previous mode with averaging publishes processed camera frames.

        Args: None.
        Returns: None.
        """
        self.start_camera()
        self.engine.update_settings(mode="Previous", average=2, palette="Gray",
                                    scale_range=False)
        self.assertTrue(wait_until(lambda: (
            self.engine.latest_packet() is not None and
            self.engine.latest_packet().mode == "Previous")))
        self.assertTrue(wait_until(
            lambda: self.engine.status()["range"].startswith("Input:")))

    def test_record_in_camera_mode_uses_resulting_frame_rate(self):
        """Recording starts at the camera's measured rate and then finishes.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(engine, "READBACK_INTERVAL_S", 0.05):
            self.start_camera()
            self.assertTrue(wait_until(
                lambda: self.engine.latest_packet() is not None))
            self.assertTrue(wait_until(
                lambda: self.engine.status()["camera_fps"] == "Camera: 29.5 fps"))
            self.engine.update_settings(record=True)
            self.assertTrue(wait_until(
                lambda: self.engine.status()["record"].startswith(
                    "Recording: 29.5")))
            self.engine.update_settings(record=False)
            self.assertTrue(wait_until(lambda: self.log.has("Saved AVI")))

    def test_failed_recorder_turns_recording_off(self):
        """A recorder that errors out is reported and record is cleared.

        Args: None.
        Returns: None.
        """
        recorder = FakeRecorder(finished=False)
        with mock.patch.object(engine, "AVIRecorder",
                               lambda *a, **k: recorder):
            self.start_camera()
            self.assertTrue(wait_until(
                lambda: self.engine.latest_packet() is not None))
            self.engine.update_settings(record=True)
            self.assertTrue(wait_until(lambda: recorder.frames))
            recorder.error = OSError("disk full")
            recorder.done.set()
            self.assertEqual(self.wait_event("record_off"), None)
            self.assertFalse(self.engine.settings().record)
            self.assertTrue(wait_until(lambda: self.log.has("Recording error: disk full")))

    def test_recording_open_failure_is_reported(self):
        """An OSError opening the AVI logs and clears the record flag.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(engine, "recording_path",
                               side_effect=OSError("no folder")):
            self.start_camera()
            self.engine.update_settings(record=True)
            self.assertEqual(self.wait_event("record_off"), None)
        self.assertTrue(self.log.has("no folder"))


class DemoLoopTests(CameraEngineCase):
    """Demo-loop branches that need particular timing or settings."""

    def test_block_average_reference_refresh_restarts_averaging(self):
        """Auto-refreshing the Reference with block averaging keeps publishing.

        Args: None.
        Returns: None.
        """
        self.engine.use_demo()
        self.engine.start()
        self.assertTrue(wait_until(lambda: self.engine.latest_packet() is not None))
        self.engine.update_settings(mode="Reference", average=2,
                                    ref_refresh_s=0.1, rolling=False)
        self.assertTrue(self.log_wait("Reference auto-refresh: every 0.1 s"))
        first = self.engine.latest_packet().sequence
        self.assertTrue(wait_until(
            lambda: self.engine.latest_packet().sequence > first + 5))

    def test_demo_reports_simulated_rate_after_one_second(self):
        """After a one-second window demo status shows the simulated rate.

        Args: None.
        Returns: None.
        """
        self.engine.use_demo()
        self.engine.start()
        self.assertTrue(wait_until(lambda: self.engine.status()[
            "camera_fps"] == "Camera: 30.0 fps (simulated)", 3))

    def test_demo_schedule_restarts_when_processing_falls_behind(self):
        """Slow synthetic frames reset the pacing schedule instead of bursting.

        Args: None.
        Returns: None.
        """
        real = engine.DemoSource.next_frame

        def slow(source, exposure, gain, now):
            """Produce a frame after exceeding the frame interval.

            Args: source is the DemoSource; exposure, gain and now are the
                usual arguments.
            Returns: The synthetic frame.
            """
            time.sleep(0.08)
            return real(source, exposure, gain, now)
        with mock.patch.object(engine.DemoSource, "next_frame", slow):
            self.engine.use_demo()
            self.engine.start()
            self.assertTrue(wait_until(
                lambda: (self.engine.latest_packet() is not None and
                         self.engine.latest_packet().sequence >= 3)))

    def log_wait(self, text):
        """Wait for a log line.

        Args: text is the substring to wait for.
        Returns: True if it appeared.
        """
        return wait_until(lambda: self.log.has(text))


class SwitchLiveTests(CameraEngineCase):
    """Demo to live switching on the worker."""

    def setUp(self):
        """Start a demo engine before any camera is available.

        Args: None.
        Returns: None.
        """
        super().setUp()
        self.engine.use_demo()
        self.engine.start()
        self.assertTrue(wait_until(lambda: self.engine.latest_packet() is not None))

    def test_switch_succeeds_and_resets_state(self):
        """A found camera replaces the demo, stops recording and resets mode.

        Args: None.
        Returns: None.
        """
        self.engine.update_settings(record=True, mode="Previous")
        self.assertTrue(wait_until(
            lambda: self.engine.status()["record"].startswith("Recording:")))
        self.engine.request("switch_live")
        self.wait_event("switched_live")
        self.assertFalse(self.engine.demo)
        settings = self.engine.settings()
        self.assertEqual((settings.record, settings.mode), (False, "Raw"))
        self.assertTrue(wait_until(lambda: (
            self.engine.latest_packet() is not None and
            not self.engine.latest_packet().demo)))
        self.assertTrue(self.log.has("LIVE CAPTURE ACTIVE"))
        self.assertTrue(wait_until(lambda: self.log.has("Saved AVI")))

    def test_switch_without_camera_keeps_demo_running(self):
        """No camera reports switch_failed and the synthetic stream continues.

        Args: None.
        Returns: None.
        """
        self.system.camera_list.cameras.clear()
        self.engine.request("switch_live")
        message = self.wait_event("switch_failed")
        self.assertIn("No FLIR", message)
        self.assertTrue(self.engine.demo)
        before = self.engine.latest_packet().sequence
        self.assertTrue(wait_until(
            lambda: self.engine.latest_packet().sequence > before))
        self.assertTrue(self.log.has("Still in dummy mode"))


class DirectWorkerTests(CameraEngineCase):
    """Worker helpers called directly, without a running thread."""

    def setUp(self):
        """Give the engine the recorder bookkeeping _run normally creates.

        Args: None.
        Returns: None.
        """
        super().setUp()
        self.engine._recorder = None
        self.engine._retired = []

    def packet(self):
        """Build a small frame packet.

        Args: None.
        Returns: An engine.FramePacket.
        """
        return engine.FramePacket(1, np.zeros((8, 8), np.uint8), "Gray", 1.0,
                                  "Raw", False)

    def test_reap_logs_error_and_marks_recording_off(self):
        """Finished recorders are summarized, errors logged and status reset.

        Args: None.
        Returns: None.
        """
        self.engine._retired = [FakeRecorder(error=OSError("bad"))]
        self.engine._reap_recorders()
        self.assertTrue(self.log.has("fake summary"))
        self.assertTrue(self.log.has("Recording error: bad"))
        self.assertEqual(self.engine._retired, [])
        self.assertEqual(self.engine.status()["record"], "Recording: off")

    def test_reap_keeps_unfinished_recorders(self):
        """Recorders still writing stay on the retired list.

        Args: None.
        Returns: None.
        """
        pending = FakeRecorder(finished=False)
        self.engine._retired = [pending]
        self.engine._reap_recorders()
        self.assertEqual(self.engine._retired, [pending])

    def test_feed_recorder_ignored_when_not_recording(self):
        """Without the record flag the signature is returned unchanged.

        Args: None.
        Returns: None.
        """
        result = self.engine._feed_recorder(
            engine.Settings(), ("Raw", 1, False), "old", self.packet(), None)
        self.assertEqual(result, "old")

    def test_feed_recorder_submits_to_active_recorder(self):
        """An active recorder receives the rendered frame.

        Args: None.
        Returns: None.
        """
        recorder = FakeRecorder()
        self.engine._recorder = recorder
        result = self.engine._feed_recorder(
            engine.Settings(record=True), ("Raw", 1, False), "sig",
            self.packet(), None)
        self.assertEqual(result, "sig")
        self.assertEqual(len(recorder.frames), 1)

    def test_feed_recorder_rolling_uses_every_frame_rate(self):
        """Rolling averages record at the source rate, capped at 30 fps.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(engine, "AVIRecorder") as recorder_class:
            self.engine._feed_recorder(
                engine.Settings(record=True, fps=60.0), ("Previous", 10, True),
                None, self.packet(), None)
        self.assertEqual(recorder_class.call_args.args[2], 30.0)

    def test_handle_actions_save_without_frame_logs(self):
        """Saving before the first frame explains why nothing was written.

        Args: None.
        Returns: None.
        """
        self.engine.request("save", title="x", notes="")
        self.engine._handle_actions("processor", None)
        self.assertTrue(self.log.has("No current display frame"))

    def test_handle_actions_save_writes_a_given_frozen_packet(self):
        """A save carrying a frozen packet writes that packet, not the newest one.

        Args: None.
        Returns: None.
        """
        frozen = engine.FramePacket(1, np.full((8, 8), 7, np.uint8), "Gray", 1.0,
                                    "Raw", True)
        self.engine._packet = self.packet()  # newer live frame, not demo
        when = object()
        self.engine.request("save", title="x", notes="", packet=frozen, frozen_at=when)
        with mock.patch.object(engine, "save_frame") as save:
            self.engine._handle_actions("processor", None)
        image, title, notes, demo, _, tag = save.call_args.args
        self.assertEqual(int(image[0, 0]), 7)
        self.assertEqual((title, notes, demo, tag), ("x", "", True, self.engine.tag))
        self.assertIs(save.call_args.kwargs["frozen_at"], when)

    def test_handle_actions_save_without_frozen_packet_saves_newest(self):
        """A plain save writes the newest packet with no freeze time.

        Args: None.
        Returns: None.
        """
        self.engine._packet = self.packet()
        self.engine.request("save", title="", notes="")
        with mock.patch.object(engine, "save_frame") as save:
            self.engine._handle_actions("processor", None)
        self.assertFalse(save.call_args.args[3])
        self.assertIsNone(save.call_args.kwargs["frozen_at"])

    def test_handle_actions_reference_without_frame_logs(self):
        """A reference request before any frame is reported.

        Args: None.
        Returns: None.
        """
        self.engine.request("reference")
        self.engine._handle_actions("processor", None)
        self.assertTrue(self.log.has("No frame available yet"))

    def test_handle_actions_ignores_switch_live_when_not_demo(self):
        """switch_live does nothing unless the engine is in demo mode.

        Args: None.
        Returns: None.
        """
        self.engine.request("switch_live")
        with mock.patch.object(self.engine, "_switch_live") as switch:
            result = self.engine._handle_actions("processor", "frame")
        switch.assert_not_called()
        self.assertEqual(result, ("processor", "frame"))

    def test_switch_live_stops_active_recorder(self):
        """A successful switch stops and retires the demo recording.

        Args: None.
        Returns: None.
        """
        recorder = FakeRecorder()
        self.engine._recorder = recorder
        self.engine.use_demo()
        processor, frame = self.engine._switch_live("old", "frame")
        self.assertTrue(recorder.stopped)
        self.assertEqual(self.engine._retired, [recorder])
        self.assertIsNone(frame)
        self.assertIsNone(self.engine.latest_packet())
        self.assertFalse(self.engine.demo)

    def test_switch_live_opens_at_the_engine_settings(self):
        """Leaving demo mode starts the camera at the current slider values.

        Args: None.
        Returns: None.
        """
        self.engine.use_demo()
        self.engine.update_settings(exposure_us=5000.0, gain_db=1.5, fps=15.0)
        self.engine._switch_live(None, None)
        nodes = self.cam.nodemap.nodes
        self.assertEqual(nodes["ExposureTime"].value, 5000.0)
        self.assertEqual(nodes["Gain"].value, 1.5)
        self.assertEqual(nodes["AcquisitionFrameRate"].value, 15.0)
        self.assertFalse(self.engine.demo)

    def test_run_waits_for_recorders_and_reports_unfinished(self):
        """Shutdown stops the active recorder and reports slow writers.

        Args: None.
        Returns: None.
        """
        active = FakeRecorder(finished=False)
        slow = FakeRecorder(finished=False)
        failed = FakeRecorder(error=OSError("late"))

        def loop():
            """Install recorders instead of running frames.

            Args: None.
            Returns: None.
            """
            self.engine._recorder = active
            self.engine._retired = [slow, failed]
        self.engine.open_camera()
        with mock.patch.object(engine, "RECORDER_FINISH_TIMEOUT_S", 0.05), \
                mock.patch.object(self.engine, "_loop", loop):
            self.engine._run()
        self.assertTrue(active.stopped)
        self.assertTrue(self.log.has("Recording did not finish within 5 seconds"))
        self.assertTrue(self.log.has("Recording error: late"))
        self.assertIsNone(self.engine.camera)
        self.assertEqual(self.wait_event("stopped"), None)


if __name__ == "__main__":
    unittest.main()


class OpenOnStartTests(CameraEngineCase):
    """open_on_start: opening a camera on the worker thread."""

    def test_camera_opens_on_the_worker_and_streams(self):
        """The camera is opened after start() and frames follow.

        Args: None.
        Returns: None.
        """
        self.engine = Engine(self.log, serial="21234567", tag="21234567")
        self.engine.open_on_start()
        self.engine.start()
        self.assertEqual(self.wait_event("camera_limits")["Gain"], (0.0, 30.0))
        self.assertTrue(wait_until(lambda: self.engine.latest_packet() is not None))
        self.assertEqual(self.engine.camera_info.serial, "21234567")

    def test_open_failure_is_reported_and_the_worker_stops(self):
        """A missing camera sends open_failed, then stopped.

        Args: None.
        Returns: None.
        """
        self.engine = Engine(self.log, serial="99999999")
        self.engine.open_on_start()
        self.engine.start()
        self.assertIn("no longer connected", self.wait_event("open_failed"))
        self.wait_event("stopped")
        self.assertTrue(self.log.has("Could not open the camera"))

    def test_waits_for_the_previous_engine_first(self):
        """A predecessor that never stops blocks the open with a clear reason.

        Args: None.
        Returns: None.
        """
        stuck = mock.Mock()
        stuck.wait.return_value = False
        self.engine.open_on_start(after=stuck)
        with mock.patch.object(engine, "PREDECESSOR_TIMEOUT_S", 0.01):
            self.engine.start()
            self.assertIn("still in use", self.wait_event("open_failed"))
        stuck.wait.assert_called_once_with(0.01)

    def test_start_without_a_source_names_every_option(self):
        """start() before choosing a source explains all three choices.

        Args: None.
        Returns: None.
        """
        with self.assertRaisesRegex(RuntimeError, "open_on_start"):
            self.engine.start()


class CameraTagTests(CameraEngineCase):
    """File names carry the camera tag."""

    def test_recordings_and_frames_are_tagged(self):
        """The engine's tag goes into recording and saved-frame names.

        Args: None.
        Returns: None.
        """
        self.engine.tag = "CAM7"
        self.engine.use_demo("CAM7")
        self.engine.update_settings(record=True)
        self.engine.start()
        self.assertTrue(wait_until(lambda: self.log.has("Recording AVI")))
        self.engine.request("save", title="", notes="")
        self.assertTrue(wait_until(lambda: self.log.has("Saved: ")))
        self.assertTrue(self.log.has("dummy_espi_raw_CAM7_"))
        self.assertTrue(self.log.has("dummy_frame_CAM7_"))


class SwitchLiveCameraListTests(CameraEngineCase):
    """The switched_live event lists the connected cameras."""

    def setUp(self):
        """Prepare a demo engine and a second fake camera.

        Args: None.
        Returns: None.
        """
        super().setUp()
        self.system.camera_list.cameras.append(FakeCamera(serial="2"))
        self.engine.use_demo()
        self.engine._recorder = None  # as _run() sets up before the loop
        self.engine._retired = []

    def test_switched_live_carries_every_camera(self):
        """Both detected cameras are reported with the switch.

        Args: None.
        Returns: None.
        """
        self.engine._switch_live(None, None)
        cameras = self.wait_event("switched_live")
        self.assertEqual([info.serial for info in cameras], ["21234567", "2"])
        self.assertEqual(self.engine.camera_info.serial, "21234567")

    def test_listing_failure_still_switches(self):
        """If the cameras cannot be listed the switch still succeeds.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(camera, "list_cameras", side_effect=RuntimeError("bus")):
            self.engine._switch_live(None, None)
        self.assertEqual(self.wait_event("switched_live"), [])
        self.assertTrue(self.log.has("Could not list FLIR cameras: bus"))

    def test_switch_live_falls_back_to_usb_live2(self):
        """Without a FLIR camera, demo mode switches to a recognised adapter.

        Args: None.
        Returns: None.
        """
        self.system.camera_list.cameras.clear()
        self.enumerator.devices = [usb_live2()]
        self.engine._switch_live(None, None)
        cameras = self.wait_event("switched_live")
        self.assertEqual([(info.kind, info.serial) for info in cameras],
                         [("capture", "video 1")])
        self.assertFalse(self.engine.demo)
        self.assertEqual(self.engine.camera_info.kind, "capture")
        self.assertTrue(self.engine.status()["camera"].startswith("Analog capture"))

    def test_switch_failure_names_the_device(self):
        """A capture device that cannot open reports why in switch_failed.

        The FLIR camera's generic "no cameras" error must not hide the
        adapter's actionable message (close WinTV / Hauppauge Capture).

        Args: None.
        Returns: None.
        """
        self.system.camera_list.cameras.clear()
        self.enumerator.devices = [usb_live2()]
        self.factory.opened = False
        self.engine._switch_live(None, None)
        message = self.wait_event("switch_failed")
        self.assertIn("WinTV", message)
        self.assertIn("Hauppauge USB-Live2", message)
        self.assertTrue(self.engine.demo)
        self.assertTrue(self.log.has("Could not open the FLIR camera"))
        self.assertTrue(self.log.has("Still in dummy mode"))


class CaptureEngineCase(CameraEngineCase):
    """Base class: engines whose source is a video-capture device."""

    def capture_engine(self, enumerated=None):
        """Create an engine for an enumerated video device.

        Args: enumerated is the fake device to open (a USB-Live2 by default).
        Returns: The Engine (opened, not started).
        """
        enumerated = enumerated or usb_live2()
        info = CameraInfo("video 1", enumerated.name, kind="capture",
                          index=enumerated.index, backend=enumerated.backend,
                          vid=enumerated.vid, pid=enumerated.pid)
        self.engine = Engine(self.log, tag=info.serial, info=info)
        self.engine.open_camera()
        return self.engine


class CaptureEngineTests(CaptureEngineCase):
    """An engine whose source is a video-capture device."""

    def test_engine_streams_from_capture(self):
        """Frames flow from the capture device; the status names the capture.

        Args: None.
        Returns: None.
        """
        engine_ = self.capture_engine()
        identity = self.wait_event("camera_identity")
        self.assertEqual(identity.short_name(), "Hauppauge USB-Live2")
        self.assertEqual(self.wait_event("camera_limits"), {})
        engine_.start()
        self.assertTrue(wait_until(lambda: engine_.latest_packet() is not None))
        self.assertEqual(engine_.latest_packet().image.shape, (480, 720))
        self.assertTrue(engine_.status()["camera"].startswith("Analog capture 720 × 480"))
        self.assertEqual(engine_.camera_info.kind, "capture")

    def test_unplugged_device_keeps_engine_running(self):
        """A device that stops sending frames leaves the engine waiting, not dead.

        Args: None.
        Returns: None.
        """
        self.factory.frames = [bgr_frame()] * 4 + [None]
        engine_ = self.capture_engine()
        engine_.start()
        self.assertTrue(wait_until(
            lambda: engine_.status()["processing"] == "Waiting for camera frame",
            timeout=5.0))
        self.assertTrue(engine_._thread.is_alive())
        errors = [line for line in self.log.lines if "acquisition error" in line]
        self.assertEqual(len(errors), 1)

    def test_engine_applies_single_field_to_interlaced_source(self):
        """Single field is on by default for the USB-Live2 and can be turned off.

        Args: None.
        Returns: None.
        """
        self.factory.frames = [interlaced_frame()]
        engine_ = self.capture_engine()
        self.assertTrue(engine_.settings().single_field)
        engine_.start()
        self.assertTrue(wait_until(lambda: engine_.latest_packet() is not None))
        self.assertEqual(int(engine_.latest_packet().image.max()), 0)
        engine_.update_settings(single_field=False)
        self.assertTrue(wait_until(
            lambda: int(engine_.latest_packet().image[1].max()) == 255))
        self.assertTrue(self.log.has("Fields: full interlaced frames"))

    def test_toggle_resets_reference(self):
        """Switching fields drops the reference, which must be captured again.

        Args: None.
        Returns: None.
        """
        engine_ = self.capture_engine()
        engine_.update_settings(mode="Reference")
        engine_.start()
        self.assertTrue(wait_until(lambda: "press F" in engine_.status()["hint"]))
        engine_.request("reference")
        self.assertTrue(wait_until(lambda: "press F" not in engine_.status()["hint"]))
        engine_.update_settings(single_field=False)
        self.assertTrue(wait_until(lambda: "press F" in engine_.status()["hint"]))
        engine_.update_settings(single_field=True)
        self.assertTrue(wait_until(
            lambda: self.log.has("Fields: single field (even rows, line-doubled)")))


class SingleFieldFlirTests(CameraEngineCase):
    """Single-field mode never touches progressive FLIR frames."""

    def test_single_field_ignored_for_blackfly(self):
        """Odd rows of a FLIR frame survive with single field on.

        Args: None.
        Returns: None.
        """
        frame = np.zeros((48, 64), dtype=np.uint8)
        frame[1::2] = 255
        self.cam.frame_source = lambda: FakeImage(frame.copy())
        self.engine.open_camera()
        self.assertTrue(self.engine.settings().single_field)
        self.engine.start()
        self.assertTrue(wait_until(lambda: self.engine.latest_packet() is not None))
        self.assertEqual(int(self.engine.latest_packet().image[1].max()), 255)


class CameraReportEventTests(CaptureEngineCase):
    """Unrecognised video devices produce a camera report when opened."""

    def reports(self):
        """Collect the camera_report payloads queued so far.

        Args: None.
        Returns: A list of Report.
        """
        return [payload for kind, payload in self.events() if kind == "camera_report"]

    def test_engine_reports_only_unrecognised_capture(self):
        """A webcam is reported; the USB-Live2 and a Blackfly are not.

        Args: None.
        Returns: None.
        """
        self.capture_engine(webcam())
        reports = self.reports()
        self.assertEqual(len(reports), 1)
        self.assertTrue(reports[0].key.startswith("Webcam C270"))
        self.assertIn("| Device name | Webcam C270 |", reports[0].body)
        self.capture_engine(usb_live2())
        self.assertEqual(self.reports(), [])
        self.engine = Engine(self.log)
        self.engine.open_camera()
        self.assertEqual(self.reports(), [])

    def test_engine_report_survives_detail_failure(self):
        """If the details cannot be read the report is still sent, without them.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(capture.CaptureSession, "read_details",
                               side_effect=RuntimeError("driver")):
            self.capture_engine(webcam())
        reports = self.reports()
        self.assertEqual(len(reports), 1)
        self.assertNotIn("| Device name |", reports[0].body)
        self.assertTrue(self.log.has("Could not read the device details: driver"))
