"""Tests for the capture scheduler (fake channels and engines; no camera, network or real output)."""

import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from espi_live import capture_uploads
from espi_live.camera import CameraInfo
from espi_live.capture_uploads import (DAILY_BYTES, MAX_QUEUE_BYTES, CaptureLedger,
                                       CaptureQueue, CaptureUploader, CaptureWorker)
from espi_live.engine import FramePacket, RawFrame, Settings

SERIAL = "21234567"


class FakeEngine:
    """The engine surface the worker reads."""

    def __init__(self, raw=None, rate=10.0, record=False, never_raw=False,
                 stall_after=None, record_after=None):
        """Create it.

        Args: raw is the raw frame (a gradient by default); rate is the
            displayed packet rate; record is the user's recording flag;
            never_raw makes the tap deliver nothing; stall_after stops the raw
            sequence advancing after that many frames (a camera that hung);
            record_after makes the record flag turn on after that many reads.
        Returns: None.
        """
        self.stall_after = stall_after
        self.record_after = record_after
        self.settings_reads = 0
        grid = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64)
        self.raw = raw if raw is not None else (grid % 251).astype(np.uint8)
        self.image = (grid % 241).astype(np.uint8)
        self.rate = rate
        self.record = record
        self.never_raw = never_raw
        self.tap = False
        self.tap_calls = []
        self.sequence = 0

    def set_raw_tap(self, on):
        """Record a tap change.

        Args: on is the new state.
        Returns: None.
        """
        self.tap = on
        self.tap_calls.append(on)

    def latest_raw(self):
        """Return a raw frame while the tap is on.

        Args: None.
        Returns: A RawFrame, or None.
        """
        if not self.tap or self.never_raw:
            return None
        if self.stall_after is None or self.sequence < self.stall_after:
            self.sequence += 1
        return RawFrame(self.sequence, self.raw.copy())

    def latest_packet(self):
        """Return a processed packet.

        Args: None.
        Returns: A FramePacket.
        """
        return FramePacket(1, self.image, 1.0, "Reference", False)

    def status(self):
        """Return the status dict.

        Args: None.
        Returns: A dict with processed_fps.
        """
        return {"processed_fps": self.rate}

    def settings(self):
        """Return the settings.

        Args: None.
        Returns: A Settings with the record flag.
        """
        self.settings_reads += 1
        if self.record_after is not None and self.settings_reads > self.record_after:
            return Settings(record=True)
        return Settings(record=self.record)


class FakeChannel:
    """The channel surface the worker reads."""

    def __init__(self, engine=None, info=None, demo=False, frozen=None, active=True,
                 supported=True, active_reads=None, stop_after_release=False):
        """Create it.

        Args: engine, info, demo, frozen, active and supported are the channel's
            fields; active_reads makes active turn False after that many
            reads; stop_after_release makes it False once the engine's raw
            tap has been released after a capture.
        Returns: None.
        """
        self.stop_after_release = stop_after_release
        self.engine = engine or FakeEngine()
        self.info = info or CameraInfo(SERIAL, "Blackfly")
        self.demo = demo
        self.frozen = frozen
        self.supported = supported
        self._active = active
        self._reads = 0
        self._active_reads = active_reads

    @property
    def active(self):
        """Report whether the channel is processing.

        Args: None.
        Returns: True, until active_reads reads have happened.
        """
        self._reads += 1
        if self._active_reads is not None and self._reads > self._active_reads:
            return False
        if self.stop_after_release and self.engine.tap_calls[-1:] == [False]:
            return False
        return self._active


class Recorder:
    """A fake GitHub call that records the files PUT."""

    def __init__(self):
        """Create it.

        Args: None.
        Returns: None.
        """
        self.paths = []

    def __call__(self, method, path, token, payload=None):
        """Answer a PUT.

        Args: method, path, token and payload as feedback.github_request takes them.
        Returns: A success reply.
        """
        self.paths.append(path.split("/contents/", 1)[1])
        return {}


class WorkerCase(unittest.TestCase):
    """A worker over a temporary captures folder."""

    def setUp(self):
        """Make the folder.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = self.tmp.name

    def tearDown(self):
        """Remove the folder.

        Args: None.
        Returns: None.
        """
        self.tmp.cleanup()

    def make(self, channels, token="", folder=None, **options):
        """Build a worker.

        Args: channels is a list of FakeChannels; token is the upload token
            ("" keeps files queued); folder overrides the captures folder;
            options are CaptureWorker keywords.
        Returns: The worker.
        """
        folder = folder or self.folder
        self.queue = CaptureQueue(folder, MAX_QUEUE_BYTES)
        self.ledger = CaptureLedger(folder)
        self.request = Recorder()
        self.uploader = CaptureUploader(self.queue, self.request, token)
        options.setdefault("clip_seconds", 0.3)
        return CaptureWorker(lambda: channels, folder, self.uploader, self.ledger,
                             self.queue, "ab12cd34", 7, **options)

    def mark(self, kind, camera="Blackfly-0"):
        """Pretend a camera was just captured so only the other kind is due.

        Args: kind is "still" or "clip"; camera is the label.
        Returns: None.
        """
        self.ledger.mark(camera, kind)

    def only_group(self):
        """Read the one queued group.

        Args: None.
        Returns: (stem, {suffix: bytes}).
        """
        groups = self.queue.groups()
        self.assertEqual(len(groups), 1)
        return groups[0], self.queue.read(groups[0])


class StillTests(WorkerCase):
    """Still pairs."""

    def test_still_pair_is_queued_with_sidecar(self):
        """A still is a raw PNG, a processed PNG and a sidecar, under a serial-free name.

        Args: None.
        Returns: None.
        """
        channel = FakeChannel()
        worker = self.make([channel])
        self.mark("clip")
        self.assertEqual(worker.run_once(), 1)
        stem, files = self.only_group()
        self.assertEqual(sorted(files), ["-processed.png", "-raw.png", ".json"])
        self.assertNotIn(SERIAL, stem)
        self.assertTrue(stem.startswith("captures/"))
        self.assertTrue(stem.endswith("-ab12cd34-Blackfly-0"))
        for name in ("-raw.png", "-processed.png"):
            image = cv2.imdecode(np.frombuffer(files[name], np.uint8), cv2.IMREAD_UNCHANGED)
            self.assertEqual(image.shape, (48, 64))
        meta = json.loads(files[".json"])
        self.assertEqual(meta["view"], "Reference")
        self.assertEqual(meta["build"], 7)
        self.assertFalse(meta["raw"]["rescaled"])
        self.assertNotIn(SERIAL, files[".json"].decode())
        self.assertFalse(self.ledger.due("Blackfly-0", "still", 1800))

    def test_float_raw_frame_is_saved_as_png_and_flagged(self):
        """A deep raw frame becomes an 8-bit PNG and the sidecar says it was rescaled.

        Args: None.
        Returns: None.
        """
        deep = (np.arange(48 * 64, dtype=np.float32).reshape(48, 64) / 7.0)
        worker = self.make([FakeChannel(FakeEngine(raw=deep))])
        self.mark("clip")
        self.assertEqual(worker.run_once(), 1)
        _stem, files = self.only_group()
        image = cv2.imdecode(np.frombuffer(files["-raw.png"], np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(image.dtype, np.uint8)
        self.assertTrue(json.loads(files[".json"])["raw"]["rescaled"])

    def test_two_cameras_get_distinct_names(self):
        """Each channel is labelled by its position.

        Args: None.
        Returns: None.
        """
        worker = self.make([FakeChannel(), FakeChannel()])
        self.mark("clip", "Blackfly-0")
        self.mark("clip", "Blackfly-1")
        self.assertEqual(worker.run_once(), 2)
        stems = self.queue.groups()
        self.assertEqual(len(set(stems)), 2)
        self.assertTrue(stems[0].endswith("Blackfly-0") or stems[1].endswith("Blackfly-0"))

    def test_ineligible_channels_are_skipped(self):
        """Demo, stopped, frozen and iPhone channels are never tapped or queued.

        Args: None.
        Returns: None.
        """
        phone = CameraInfo("udid", "iPhone", kind="iphone")
        channels = [FakeChannel(demo=True), FakeChannel(active=False),
                    FakeChannel(frozen=object()), FakeChannel(info=phone)]
        worker = self.make(channels)
        self.assertEqual(worker.run_once(), 0)
        self.assertEqual(self.queue.groups(), [])
        self.assertTrue(all(channel.engine.tap_calls == [] for channel in channels))

    def test_blank_raw_frame_is_skipped_without_marking(self):
        """A constant raw frame (camera not delivering) is not uploaded and is retried.

        Args: None.
        Returns: None.
        """
        engine = FakeEngine(raw=np.zeros((48, 64), np.uint8))
        worker = self.make([FakeChannel(engine)])
        self.mark("clip")
        self.assertEqual(worker.run_once(), 0)
        self.assertEqual(self.queue.groups(), [])
        self.assertTrue(self.ledger.due("Blackfly-0", "still", 1800))

    def test_no_raw_frame_in_time_is_skipped_and_tap_turned_off(self):
        """If the camera never delivers a raw frame the wait ends and the tap is released.

        Args: None.
        Returns: None.
        """
        clock = [0.0]

        def sleep(seconds):
            """Advance the fake clock.

            Args: seconds is the wait.
            Returns: None.
            """
            clock[0] += seconds

        engine = FakeEngine(never_raw=True)
        worker = self.make([FakeChannel(engine)], sleep=sleep, monotonic=lambda: clock[0])
        self.mark("clip")
        self.assertEqual(worker.run_once(), 0)
        self.assertEqual(engine.tap_calls, [True, False])
        self.assertGreaterEqual(clock[0], 5.0)

    def test_tap_is_released_after_every_capture(self):
        """The raw tap is on only while a capture needs it.

        Args: None.
        Returns: None.
        """
        engine = FakeEngine()
        worker = self.make([FakeChannel(engine)])
        worker.run_once()
        self.assertFalse(engine.tap)
        self.assertEqual(engine.tap_calls.count(True), engine.tap_calls.count(False))
        self.assertGreaterEqual(len(engine.tap_calls), 4)  # a still and a clip

    def test_a_pair_that_cannot_fit_is_dropped_whole(self):
        """When a PNG cannot be fitted, nothing is queued and the camera waits its turn.

        Args: None.
        Returns: None.
        """
        worker = self.make([FakeChannel()], max_still_bytes=10)
        self.mark("clip")
        self.assertEqual(worker.run_once(), 0)
        self.assertEqual(self.queue.groups(), [])
        self.assertFalse(self.ledger.due("Blackfly-0", "still", 1800))

    def test_over_budget_still_is_dropped_whole(self):
        """With no room in today's budget no half pair is queued.

        Args: None.
        Returns: None.
        """
        worker = self.make([FakeChannel()])
        self.mark("clip")
        self.ledger.spend("still", DAILY_BYTES)
        self.assertEqual(worker.run_once(), 0)
        self.assertEqual(self.queue.groups(), [])
        self.assertFalse(self.ledger.due("Blackfly-0", "still", 1800))

    def test_unwritable_folder_drops_the_capture_and_does_not_raise(self):
        """A captures folder that cannot be written costs only the capture.

        Args: None.
        Returns: None.
        """
        blocked = os.path.join(self.folder, "file")
        with open(blocked, "w"):
            pass
        engine = FakeEngine()
        worker = self.make([FakeChannel(engine)], folder=os.path.join(blocked, "inside"))
        self.assertEqual(worker.run_once(), 0)
        self.assertEqual(self.queue.groups(), [])
        self.assertFalse(engine.tap)

    def test_uploads_run_after_the_captures(self):
        """A pass hands the queue to the uploader.

        Args: None.
        Returns: None.
        """
        worker = self.make([FakeChannel()], token="tok")
        self.mark("clip")
        self.assertEqual(worker.run_once(), 1)
        self.assertEqual(self.queue.groups(), [])
        self.assertEqual(sorted(path.rsplit("-Blackfly-0", 1)[1] for path in self.request.paths),
                         ["-processed.png", "-raw.png", ".json"])

    def test_a_channel_that_stops_before_queueing_is_not_captured(self):
        """Eligibility is checked again just before a capture is queued.

        Args: None.
        Returns: None.
        """
        worker = self.make([FakeChannel(active_reads=1)])
        self.mark("clip")
        self.assertEqual(worker.run_once(), 0)
        self.assertEqual(self.queue.groups(), [])

    def test_no_packet_yet_is_skipped(self):
        """A channel that has published nothing is retried later.

        Args: None.
        Returns: None.
        """
        engine = FakeEngine()
        engine.latest_packet = lambda: None
        worker = self.make([FakeChannel(engine)])
        self.assertEqual(worker.run_once(), 0)
        self.assertEqual(engine.tap_calls, [])


class ClipTests(WorkerCase):
    """Clip pairs."""

    def work_is_clean(self):
        """Check that no clip work folders or files are left.

        Args: None.
        Returns: True when the work folder is empty or missing.
        """
        work = os.path.join(self.folder, "work")
        return not os.path.isdir(work) or os.listdir(work) == []

    def test_raw_and_processed_clip_pair_are_queued_together(self):
        """A clip is two AVIs and a sidecar sharing one stem.

        Args: None.
        Returns: None.
        """
        worker = self.make([FakeChannel()])
        self.mark("still")
        self.assertEqual(worker.run_once(), 1)
        _stem, files = self.only_group()
        self.assertEqual(sorted(files), ["-processed.avi", "-raw.avi", ".json"])
        path = os.path.join(self.folder, "probe.avi")
        for name in ("-raw.avi", "-processed.avi"):
            with open(path, "wb") as handle:
                handle.write(files[name])
            video = cv2.VideoCapture(path)
            ok, frame = video.read()
            video.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape[:2], (48, 64))
        self.assertEqual(json.loads(files[".json"])["view"], "Reference")
        self.assertFalse(self.ledger.due("Blackfly-0", "clip", 21600))
        self.assertTrue(self.work_is_clean())

    def test_clip_skipped_when_user_is_recording_but_still_taken(self):
        """A user's recording is left alone: the clip waits, the still goes ahead.

        Args: None.
        Returns: None.
        """
        worker = self.make([FakeChannel(FakeEngine(record=True))])
        self.assertEqual(worker.run_once(), 1)
        _stem, files = self.only_group()
        self.assertIn("-raw.png", files)
        self.assertTrue(self.ledger.due("Blackfly-0", "clip", 21600))

    def test_clip_pair_skipped_when_processed_rate_below_two(self):
        """Slow views (block averaging, phase-shifting) get stills but no clips.

        Args: None.
        Returns: None.
        """
        worker = self.make([FakeChannel(FakeEngine(rate=1.0))])
        self.assertEqual(worker.run_once(), 1)
        _stem, files = self.only_group()
        self.assertIn("-raw.png", files)
        self.assertTrue(self.ledger.due("Blackfly-0", "clip", 21600))

    def test_channel_stopping_mid_clip_deletes_both_partials_and_queues_nothing(self):
        """A channel that stops during a clip leaves no files and no half pair.

        Args: None.
        Returns: None.
        """
        engine = FakeEngine()
        worker = self.make([FakeChannel(engine, active_reads=2)])
        self.mark("still")
        self.assertEqual(worker.run_once(), 0)
        self.assertEqual(self.queue.groups(), [])
        self.assertTrue(self.work_is_clean())
        self.assertFalse(engine.tap)
        self.assertTrue(self.ledger.due("Blackfly-0", "clip", 21600))

    def test_spent_clip_budget_skips_the_clip_before_recording(self):
        """With the day's budget gone no clip is even recorded.

        Args: None.
        Returns: None.
        """
        engine = FakeEngine()
        worker = self.make([FakeChannel(engine)])
        self.mark("still")
        self.ledger.spend("clip", DAILY_BYTES)
        self.assertEqual(worker.run_once(), 0)
        self.assertEqual(engine.tap_calls, [])
        self.assertFalse(self.ledger.due("Blackfly-0", "clip", 21600))

    def test_budget_used_up_during_the_clip_drops_the_pair(self):
        """A clip that no longer fits the budget once encoded is dropped whole.

        Args: None.
        Returns: None.
        """
        worker = self.make([FakeChannel()])
        self.mark("still")
        self.ledger.spend("clip", DAILY_BYTES - 10)
        self.assertEqual(worker.run_once(), 0)
        self.assertEqual(self.queue.groups(), [])
        self.assertTrue(self.work_is_clean())
        self.assertFalse(self.ledger.due("Blackfly-0", "clip", 21600))

    def test_a_clip_that_cannot_be_shrunk_is_dropped_whole(self):
        """Clips that stay over the cap after re-encoding are dropped, files and all.

        Args: None.
        Returns: None.
        """
        worker = self.make([FakeChannel()], max_clip_bytes=100)
        self.mark("still")
        self.assertEqual(worker.run_once(), 0)
        self.assertEqual(self.queue.groups(), [])
        self.assertTrue(self.work_is_clean())
        self.assertFalse(self.ledger.due("Blackfly-0", "clip", 21600))

    def fake_clock(self):
        """Make a clock that only moves when the worker sleeps.

        Args: None.
        Returns: (sleep, monotonic) functions.
        """
        clock = [0.0]

        def sleep(seconds):
            """Advance the fake clock.

            Args: seconds is the wait.
            Returns: None.
            """
            clock[0] += seconds

        return sleep, lambda: clock[0]

    def test_a_channel_that_stops_as_the_clip_ends_is_not_queued(self):
        """Eligibility is checked again once both clips are encoded.

        Args: None.
        Returns: None.
        """
        worker = self.make([FakeChannel(stop_after_release=True)])
        self.mark("still")
        self.assertEqual(worker.run_once(), 0)
        self.assertEqual(self.queue.groups(), [])
        self.assertTrue(self.work_is_clean())
        self.assertTrue(self.ledger.due("Blackfly-0", "clip", 21600))

    def test_a_clip_is_dropped_when_the_raw_frames_stall(self):
        """A camera that stops delivering mid-clip leaves no frozen clip behind.

        Args: None.
        Returns: None.
        """
        sleep, monotonic = self.fake_clock()
        engine = FakeEngine(stall_after=3)
        worker = self.make([FakeChannel(engine)], sleep=sleep, monotonic=monotonic,
                           clip_seconds=5.0)
        self.mark("still")
        self.assertEqual(worker.run_once(), 0)
        self.assertEqual(self.queue.groups(), [])
        self.assertTrue(self.work_is_clean())
        self.assertFalse(engine.tap)
        self.assertTrue(self.ledger.due("Blackfly-0", "clip", 21600))

    def test_a_clip_is_dropped_when_the_user_starts_recording_mid_clip(self):
        """The user's recording wins: a clip in progress is abandoned.

        Args: None.
        Returns: None.
        """
        sleep, monotonic = self.fake_clock()
        engine = FakeEngine(record_after=3)
        worker = self.make([FakeChannel(engine)], sleep=sleep, monotonic=monotonic,
                           clip_seconds=5.0)
        self.mark("still")
        self.assertEqual(worker.run_once(), 0)
        self.assertEqual(self.queue.groups(), [])
        self.assertTrue(self.work_is_clean())
        self.assertTrue(self.ledger.due("Blackfly-0", "clip", 21600))

    def test_clip_frames_are_scaled_down_and_the_rate_is_capped(self):
        """Big frames are shrunk before encoding and the clip rate never exceeds 10 fps.

        Args: None.
        Returns: None.
        """
        seen = []

        class Spy(capture_uploads.AVIRecorder):
            """A real recorder that remembers how it was built."""

            def __init__(self, path, frame, fps):
                """Record the arguments, then build the real recorder.

                Args: path, frame and fps as AVIRecorder takes them.
                Returns: None.
                """
                seen.append((frame.shape[:2], fps))
                super().__init__(path, frame, fps)

        big = (np.arange(1536 * 2048, dtype=np.uint32).reshape(1536, 2048) % 251).astype(np.uint8)
        engine = FakeEngine(raw=big, rate=30.0)
        engine.image = big
        worker = self.make([FakeChannel(engine)])
        self.mark("still")
        with mock.patch.object(capture_uploads, "AVIRecorder", Spy):
            self.assertEqual(worker.run_once(), 1)
        self.assertEqual(len(seen), 2)
        for shape, fps in seen:
            self.assertLessEqual(shape[1], capture_uploads.CLIP_MAX_WIDTH)
            self.assertEqual(shape[1] * 3, shape[0] * 4)  # aspect kept
            self.assertLessEqual(fps, capture_uploads.CLIP_MAX_FPS)
        _stem, files = self.only_group()
        meta = json.loads(files[".json"])
        self.assertLess(meta["raw"]["scale"], 1.0)
        self.assertEqual(meta["raw"]["shape"], [1536, 2048])

    def test_the_processed_clip_is_not_encoded_when_the_raw_clip_cannot_fit(self):
        """One clip that cannot fit ends the attempt without shrinking the other.

        Args: None.
        Returns: None.
        """
        calls = []
        real = capture_uploads.snapshots.shrink_clip

        def counting(path, max_bytes):
            """Count the call, then shrink for real.

            Args: path and max_bytes as shrink_clip takes them.
            Returns: shrink_clip's result.
            """
            calls.append(path)
            return real(path, max_bytes)

        worker = self.make([FakeChannel()], max_clip_bytes=100)
        self.mark("still")
        with mock.patch.object(capture_uploads.snapshots, "shrink_clip", counting):
            self.assertEqual(worker.run_once(), 0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(self.work_is_clean())

    def fake_recorders(self, stop_sets_done=True, error=None, fail_second=False):
        """Build a fake AVIRecorder class that remembers its instances.

        Args: stop_sets_done makes stop() finish the recorder; error is the
            recorder's error after stopping; fail_second makes the second
            construction raise.
        Returns: The class; its .made list holds the instances.
        """
        class Fake:
            """A recorder that writes nothing."""

            made = []

            def __init__(self, path, frame, fps):
                """Create it (or fail like a missing thread).

                Args: path, frame and fps as AVIRecorder takes them.
                Returns: None.
                """
                if fail_second and Fake.made:
                    raise RuntimeError("can't start new thread")
                self.stopped = False
                self.done = threading.Event()
                self.error = None
                Fake.made.append(self)
                height, width = frame.shape[:2]  # a real, valid file, so only the checks decide
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
                                         (width, height), True)
                for _ in range(3):
                    writer.write(np.zeros((height, width, 3), np.uint8))
                writer.release()

            def submit(self, frame):
                """Ignore a frame.

                Args: frame is the image.
                Returns: None.
                """

            def stop(self):
                """Stop, finishing when asked to.

                Args: None.
                Returns: None.
                """
                self.stopped = True
                self.error = error
                if stop_sets_done:
                    self.done.set()

        return Fake

    def test_a_failed_second_recorder_stops_the_first(self):
        """If the second recorder cannot start, the first does not keep writing.

        Args: None.
        Returns: None.
        """
        fake = self.fake_recorders(fail_second=True)
        worker = self.make([FakeChannel()])
        self.mark("still")
        with mock.patch.object(capture_uploads, "AVIRecorder", fake):
            self.assertEqual(worker.run_once(), 0)
        self.assertEqual(len(fake.made), 1)
        self.assertTrue(fake.made[0].stopped)
        self.assertTrue(self.work_is_clean())

    def test_a_recorder_that_failed_drops_the_clip(self):
        """An encoder error means the files are not trusted and nothing is queued.

        Args: None.
        Returns: None.
        """
        fake = self.fake_recorders(error=OSError("disk full"))
        worker = self.make([FakeChannel()])
        self.mark("still")
        with mock.patch.object(capture_uploads, "AVIRecorder", fake):
            self.assertEqual(worker.run_once(), 0)
        self.assertEqual(self.queue.groups(), [])
        self.assertFalse(self.ledger.due("Blackfly-0", "clip", 21600))

    def test_a_recorder_that_never_finishes_drops_the_clip(self):
        """A writer still running after the wait is not read; nothing is queued.

        Args: None.
        Returns: None.
        """
        fake = self.fake_recorders(stop_sets_done=False)
        worker = self.make([FakeChannel()])
        self.mark("still")
        with mock.patch.object(capture_uploads, "AVIRecorder", fake), \
                mock.patch.object(capture_uploads, "RAW_WAIT_S", 0.01):
            self.assertEqual(worker.run_once(), 0)
        self.assertEqual(self.queue.groups(), [])
        self.assertFalse(self.ledger.due("Blackfly-0", "clip", 21600))

    def test_blank_raw_frame_skips_the_clip(self):
        """A camera that delivers nothing records no clip and is retried.

        Args: None.
        Returns: None.
        """
        engine = FakeEngine(raw=np.zeros((48, 64), np.uint8))
        worker = self.make([FakeChannel(engine)])
        self.mark("still")
        self.assertEqual(worker.run_once(), 0)
        self.assertEqual(engine.tap_calls, [True, False])
        self.assertTrue(self.work_is_clean())
        self.assertTrue(self.ledger.due("Blackfly-0", "clip", 21600))

    def test_unwritable_folder_skips_the_clip(self):
        """Without a writable work folder the clip is dropped quietly.

        Args: None.
        Returns: None.
        """
        blocked = os.path.join(self.folder, "file")
        with open(blocked, "w"):
            pass
        worker = self.make([FakeChannel()], folder=os.path.join(blocked, "inside"))
        self.assertEqual(worker.run_once(), 0)


class ThreadTests(WorkerCase):
    """The scheduler thread."""

    def test_thread_runs_at_start_on_kick_and_survives_errors(self):
        """The thread runs a pass at once and on kick, and an exception ends nothing.

        Args: None.
        Returns: None.
        """
        calls = []

        def channels():
            """List channels, failing the first time.

            Args: None.
            Returns: An empty list after the first call.
            """
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom")
            return []

        self.queue = CaptureQueue(self.folder, MAX_QUEUE_BYTES)
        worker = CaptureWorker(channels, self.folder,
                               CaptureUploader(self.queue, Recorder(), ""),
                               CaptureLedger(self.folder), self.queue, "ab12cd34",
                               interval_s=30.0)
        worker.start()
        self.wait_for(lambda: len(calls) >= 1)
        worker.kick()
        self.wait_for(lambda: len(calls) >= 2)
        self.assertTrue(worker.thread.is_alive())
        worker.stop()
        worker.thread.join(5)
        self.assertFalse(worker.thread.is_alive())

    def wait_for(self, condition, timeout=5.0):
        """Wait until a condition holds.

        Args: condition is a callable; timeout is how long to wait.
        Returns: None; fails the test on timeout.
        """
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if condition():
                return
            time.sleep(0.01)
        self.fail("timed out")


class ConstantsTests(unittest.TestCase):
    """The defaults the design fixes."""

    def test_defaults_match_the_design(self):
        """The documented defaults are the shipped ones.

        Args: None.
        Returns: None.
        """
        self.assertEqual(capture_uploads.STILL_INTERVAL_S, 1800)
        self.assertEqual(capture_uploads.CLIP_INTERVAL_S, 21600)
        self.assertEqual(capture_uploads.CLIP_SECONDS, 5)
        self.assertEqual(capture_uploads.MAX_STILL_BYTES, 3 * 1024 * 1024)
        self.assertEqual(capture_uploads.MAX_CLIP_BYTES, 7 * 1024 * 1024)
        self.assertEqual(capture_uploads.DAILY_BYTES, 60 * 1024 * 1024)
        self.assertEqual(capture_uploads.CLIP_RESERVE_BYTES, 30 * 1024 * 1024)
        self.assertEqual(capture_uploads.MIN_PACKET_RATE, 2.0)


if __name__ == "__main__":
    unittest.main()
