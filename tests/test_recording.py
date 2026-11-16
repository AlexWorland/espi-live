"""Camera-free checks for AVI recording and still-frame saving.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import os
import tempfile
import time
import unittest
from datetime import datetime
from unittest import mock

import cv2
import numpy as np

from espi_live import paths, recording
from list_log import ListLog


class FakeWriter:
    """A stand-in for cv2.VideoWriter that records frames."""

    def __init__(self, opened=True, delay=0.0, fail=False):
        """Configure the fake.

        Args: opened is what isOpened reports; delay is seconds each write
            takes; fail makes write raise.
        Returns: None.
        """
        self.opened = opened
        self.delay = delay
        self.fail = fail
        self.frames = []
        self.released = 0

    def isOpened(self):
        """Report whether the fake file opened.

        Args: None.
        Returns: The configured flag.
        """
        return self.opened

    def write(self, frame):
        """Store a frame, optionally slowly or failing.

        Args: frame is the image written.
        Returns: None. Raises RuntimeError when configured to fail.
        """
        if self.fail:
            raise RuntimeError("encoder died")
        if self.delay:
            time.sleep(self.delay)
        self.frames.append(frame)

    def release(self):
        """Count a release.

        Args: None.
        Returns: None.
        """
        self.released += 1


class AVIRecorderTests(unittest.TestCase):
    """The constant-rate writer thread."""

    def make(self, writer, frame=None, fps=50):
        """Create a recorder around a fake writer.

        Args: writer is the FakeWriter; frame is the first frame; fps is the rate.
        Returns: The AVIRecorder.
        """
        frame = np.zeros((4, 6), dtype=np.uint8) if frame is None else frame
        with mock.patch.object(recording.cv2, "VideoWriter", return_value=writer):
            return recording.AVIRecorder("x.avi", frame, fps)

    def finish(self, recorder):
        """Stop a recorder and wait for its thread.

        Args: recorder is the AVIRecorder.
        Returns: None.
        """
        recorder.stop()
        self.assertTrue(recorder.done.wait(5))

    def test_unopenable_file_raises_and_releases_writer(self):
        """A writer that cannot open raises OSError after releasing.

        Args: None.
        Returns: None.
        """
        writer = FakeWriter(opened=False)
        with self.assertRaises(OSError):
            self.make(writer)
        self.assertEqual(writer.released, 1)

    def test_thread_start_failure_releases_writer_and_reraises(self):
        """If the thread cannot start, the writer is released.

        Args: None.
        Returns: None.
        """
        writer = FakeWriter()
        with mock.patch.object(recording.threading.Thread, "start",
                               side_effect=RuntimeError("no threads")):
            with self.assertRaises(RuntimeError):
                self.make(writer)
        self.assertEqual(writer.released, 1)

    def test_stop_writes_at_least_one_gray_frame_as_bgr(self):
        """A gray frame is converted to 3 channels before encoding.

        Args: None.
        Returns: None.
        """
        writer = FakeWriter()
        recorder = self.make(writer)
        self.finish(recorder)
        self.assertGreaterEqual(recorder.frames_written, 1)
        self.assertEqual(writer.frames[0].shape, (4, 6, 3))
        self.assertEqual(writer.released, 1)
        self.assertIsNone(recorder.error)

    def test_color_frame_is_written_unchanged(self):
        """A BGR frame is passed to the writer as is.

        Args: None.
        Returns: None.
        """
        writer = FakeWriter()
        frame = np.full((4, 6, 3), 7, dtype=np.uint8)
        recorder = self.make(writer, frame)
        self.finish(recorder)
        self.assertIs(writer.frames[0], frame)

    def test_gray_codec_writes_single_channel(self):
        """With color off a gray frame is written as is, with the chosen codec.

        Args: None.
        Returns: None.
        """
        writer = FakeWriter()
        with mock.patch.object(recording.cv2, "VideoWriter",
                               return_value=writer) as factory:
            recorder = recording.AVIRecorder(
                "x.avi", np.zeros((4, 6), dtype=np.uint8), 50,
                codec="FFV1", color=False)
        self.finish(recorder)
        self.assertEqual(writer.frames[0].ndim, 2)
        self.assertEqual(factory.call_args.args[1],
                         cv2.VideoWriter_fourcc(*"FFV1"))
        self.assertFalse(factory.call_args.args[4])

    def exact(self, writer, fps=30, frame=None):
        """Create a frame-exact recorder around a fake writer.

        Args: writer is the FakeWriter; fps is the container rate; frame is
            the first frame.
        Returns: The AVIRecorder with timeline=False.
        """
        frame = np.zeros((4, 6), dtype=np.uint8) if frame is None else frame
        with mock.patch.object(recording.cv2, "VideoWriter", return_value=writer):
            return recording.AVIRecorder("x.avi", frame, fps, color=False,
                                         timeline=False)

    def wait_frames(self, recorder, count):
        """Wait until a recorder has written a number of frames.

        Args: recorder is the AVIRecorder; count is the frame total to wait for.
        Returns: None; fails the test after five seconds.
        """
        deadline = time.monotonic() + 5
        while recorder.frames_written < count and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertGreaterEqual(recorder.frames_written, count)

    def test_frame_exact_writes_each_submitted_frame_once(self):
        """Without a timeline nothing is repeated while the camera is quiet.

        Args: None.
        Returns: None.
        """
        writer = FakeWriter()
        recorder = self.exact(writer)
        for value in (1, 2, 3):
            self.wait_frames(recorder, value)
            recorder.submit(np.full((4, 6), value, dtype=np.uint8))
        self.wait_frames(recorder, 4)
        time.sleep(0.3)  # a timeline writer would repeat the last frame ~9 times
        self.finish(recorder)
        self.assertEqual([int(frame[0, 0]) for frame in writer.frames], [0, 1, 2, 3])
        self.assertEqual(recorder.skipped_preview_frames, 0)
        self.assertEqual(recorder.late_intervals, 0)

    def test_frame_exact_stop_writes_no_padding_and_returns_quickly(self):
        """Stopping after idle time adds no repeated frames and ends at once.

        Args: None.
        Returns: None.
        """
        writer = FakeWriter()
        recorder = self.exact(writer)
        self.wait_frames(recorder, 1)
        time.sleep(0.2)
        started = time.monotonic()
        self.finish(recorder)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(recorder.frames_written, 1)
        self.assertEqual(writer.released, 1)

    def test_frame_exact_flushes_pending_frame_on_stop(self):
        """A frame submitted just before stop is still written.

        Args: None.
        Returns: None.
        """
        writer = FakeWriter(delay=0.05)
        recorder = self.exact(writer)
        recorder.submit(np.full((4, 6), 9, dtype=np.uint8))
        recorder.stop()
        self.assertTrue(recorder.done.wait(5))
        self.assertEqual(int(writer.frames[-1][0, 0]), 9)

    def test_frame_exact_counts_dropped_frames_in_summary(self):
        """An encoder slower than the camera drops frames and says so.

        Args: None.
        Returns: None.
        """
        recorder = self.exact(FakeWriter(delay=0.1))
        for value in range(1, 8):
            recorder.submit(np.full((4, 6), value, dtype=np.uint8))
        self.finish(recorder)
        self.assertGreaterEqual(recorder.skipped_preview_frames, 1)
        self.assertIn("dropped", recorder.summary())
        self.assertIn(f"{recorder.skipped_preview_frames} camera frames dropped",
                      recorder.summary())

    def test_frame_exact_encoder_error_is_recorded(self):
        """A writer exception ends a frame-exact recorder and is kept.

        Args: None.
        Returns: None.
        """
        writer = FakeWriter(fail=True)
        recorder = self.exact(writer)
        self.assertTrue(recorder.done.wait(5))
        self.assertIsInstance(recorder.error, RuntimeError)
        self.assertEqual(writer.released, 1)

    def test_stop_is_idempotent(self):
        """A second stop keeps the first stop time.

        Args: None.
        Returns: None.
        """
        recorder = self.make(FakeWriter())
        recorder.stop()
        first = recorder.stop_time
        recorder.stop()
        self.assertEqual(recorder.stop_time, first)
        self.assertTrue(recorder.done.wait(5))

    def test_submit_replaces_pending_frame_and_counts_skips(self):
        """Frames submitted between samples are counted as skipped.

        Args: None.
        Returns: None.
        """
        recorder = self.make(FakeWriter(), fps=100)
        self.finish(recorder)
        before = recorder.skipped_preview_frames
        for value in range(1, 4):
            recorder.submit(np.full((4, 6), value, dtype=np.uint8))
        sampled = recorder._write_latest(recorder.sequence - 4)
        self.assertEqual(sampled, recorder.sequence)
        self.assertEqual(recorder.skipped_preview_frames, before + 3)

    def test_slow_encoder_counts_late_intervals(self):
        """An encoder slower than the frame interval is counted as late.

        Args: None.
        Returns: None.
        """
        recorder = self.make(FakeWriter(delay=0.03), fps=100)
        time.sleep(0.1)
        self.finish(recorder)
        self.assertGreaterEqual(recorder.late_intervals, 1)

    def test_encoder_error_is_recorded_and_thread_finishes(self):
        """A writer exception is stored and the writer still released.

        Args: None.
        Returns: None.
        """
        writer = FakeWriter(fail=True)
        recorder = self.make(writer)
        self.assertTrue(recorder.done.wait(5))
        self.assertIsInstance(recorder.error, RuntimeError)
        self.assertEqual(writer.released, 1)

    def test_summary_reports_counts(self):
        """The summary mentions path, frames and rate.

        Args: None.
        Returns: None.
        """
        recorder = self.make(FakeWriter())
        self.finish(recorder)
        text = recorder.summary()
        self.assertIn("x.avi", text)
        self.assertIn("50.0 fps", text)

    def test_real_avi_file_is_written(self):
        """A real OpenCV writer produces a non-empty AVI.

        Args: None.
        Returns: None.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.avi")
            recorder = recording.AVIRecorder(
                path, np.full((16, 16), 90, dtype=np.uint8), 30)
            time.sleep(0.1)
            self.finish(recorder)
            self.assertGreaterEqual(recorder.frames_written, 1)
            self.assertGreater(os.path.getsize(path), 0)


class RecordingSessionTests(unittest.TestCase):
    """One timestamped folder holding the raw, processed and window files."""

    def setUp(self):
        """Redirect output/ to a temp folder.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(paths, "OUTPUT_DIR", self.tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.log = ListLog()
        self.gray = np.zeros((4, 6), dtype=np.uint8)

    def test_session_creates_folder_and_info(self):
        """The folder sits under recordings, carries the tag and has session.txt.

        Args: None.
        Returns: None.
        """
        session = recording.RecordingSession(False, "SN1", "Exposure: 8000")
        name = os.path.basename(session.folder)
        self.assertRegex(name, r"^espi_\d{8}_\d{6}_\d{6}_SN1$")
        self.assertEqual(os.path.dirname(session.folder),
                         os.path.join(self.tmp.name, "recordings"))
        with open(os.path.join(session.folder, "session.txt"),
                  encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("Source: Camera", text)
        self.assertIn("Exposure: 8000", text)

    def test_session_demo_prefix_and_label(self):
        """Demo folders get the dummy_ prefix and a Demo source line.

        Args: None.
        Returns: None.
        """
        session = recording.RecordingSession(True)
        self.assertTrue(os.path.basename(session.folder).startswith("dummy_espi_"))
        with open(os.path.join(session.folder, "session.txt"),
                  encoding="utf-8") as handle:
            self.assertIn("Source: Demo", handle.read())

    def test_session_tag_is_sanitized(self):
        """Unsafe characters in a camera tag are replaced in the folder name.

        Args: None.
        Returns: None.
        """
        session = recording.RecordingSession(False, "cam 1/a")
        self.assertTrue(os.path.basename(session.folder).endswith("_cam_1_a"))

    def test_session_unusable_folder_raises(self):
        """An OSError creating the folder reaches the caller.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(recording, "output_folder", side_effect=OSError("ro")):
            with self.assertRaises(OSError):
                recording.RecordingSession(False)

    def test_session_unwritable_info_raises(self):
        """An OSError writing session.txt reaches the caller.

        Args: None.
        Returns: None.
        """
        with mock.patch("builtins.open", side_effect=OSError("denied")):
            with self.assertRaises(OSError):
                recording.RecordingSession(False)

    def test_processed_names_increment(self):
        """Processed files are numbered from 01 with a lowercase mode.

        Args: None.
        Returns: None.
        """
        session = recording.RecordingSession(False)
        with mock.patch.object(recording, "AVIRecorder") as recorder_class:
            session.open_processed("Reference", self.gray, 10.0)
            session.open_processed("Raw", self.gray, 10.0)
        names = [os.path.basename(call.args[0])
                 for call in recorder_class.call_args_list]
        self.assertEqual(names, ["processed_01_reference.avi", "processed_02_raw.avi"])
        self.assertEqual(recorder_class.call_args.args[2], 10.0)

    def test_open_raw_uses_lossless_gray_writer(self):
        """The raw file is single-channel FFV1.

        Args: None.
        Returns: None.
        """
        session = recording.RecordingSession(False)
        with mock.patch.object(recording, "AVIRecorder") as recorder_class:
            session.open_raw(self.gray, 25.0, self.log)
        self.assertEqual(os.path.basename(recorder_class.call_args.args[0]), "raw.avi")
        self.assertEqual(recorder_class.call_args.kwargs,
                         {"codec": "FFV1", "color": False, "timeline": False})
        self.assertEqual(self.log.lines, [])

    def test_open_raw_falls_back_to_mjpg(self):
        """If FFV1 will not open, raw.avi is written as MJPG and the log says so.

        Args: None.
        Returns: None.
        """
        ffv1 = cv2.VideoWriter_fourcc(*"FFV1")
        made = []

        def factory(path, fourcc, fps, size, color):
            """Open every writer except FFV1 ones.

            Args: the cv2.VideoWriter arguments.
            Returns: A FakeWriter that reports whether it opened.
            """
            writer = FakeWriter(opened=fourcc != ffv1)
            made.append((fourcc, color, writer))
            return writer

        session = recording.RecordingSession(False)
        with mock.patch.object(recording.cv2, "VideoWriter", side_effect=factory):
            recorder = session.open_raw(self.gray, 25.0, self.log)
            recorder.stop()
            self.assertTrue(recorder.done.wait(5))
        self.assertEqual(os.path.basename(recorder.path), "raw.avi")
        self.assertEqual([entry[1] for entry in made], [False, True])
        self.assertEqual(len(self.log.lines), 1)
        self.assertIn("FFV1", self.log.lines[0])
        self.assertEqual(made[1][2].frames[0].shape, (4, 6, 3))

    def read_info(self, session):
        """Read a session's session.txt.

        Args: session is the RecordingSession.
        Returns: The file's text.
        """
        with open(os.path.join(session.folder, "session.txt"),
                  encoding="utf-8") as handle:
            return handle.read()

    def test_session_info_names_the_build(self):
        """session.txt says which build wrote the recording.

        Args: None.
        Returns: None.
        """
        self.assertIn("Build: source checkout",
                      self.read_info(recording.RecordingSession(False)))
        with mock.patch.object(recording.updater, "current_build", return_value=42), \
                mock.patch.object(recording.updater, "build_value",
                                  return_value="abc1234"):
            info = self.read_info(recording.RecordingSession(False))
        self.assertIn("Build: 42 (abc1234)", info)

    def test_open_raw_notes_the_codec_in_session_info(self):
        """session.txt records the raw codec, rate and whether it is lossy.

        Args: None.
        Returns: None.
        """
        session = recording.RecordingSession(False)
        with mock.patch.object(recording, "AVIRecorder"):
            session.open_raw(self.gray, 25.0, self.log)
        info = self.read_info(session)
        self.assertIn("Raw codec: FFV1 (lossless)", info)
        self.assertIn("Raw rate: 25.0 fps", info)

    def test_open_raw_fallback_is_noted_as_lossy(self):
        """After an MJPG fallback session.txt warns that raw.avi is lossy.

        Args: None.
        Returns: None.
        """
        calls = []

        def recorder(path, frame, fps, **options):
            """Fail the FFV1 attempt and accept the MJPG one.

            Args: path, frame, fps and options match AVIRecorder.
            Returns: A stand-in object. Raises OSError for FFV1.
            """
            calls.append(options)
            if options.get("codec") == "FFV1":
                raise OSError("no ffv1")
            return mock.Mock()
        session = recording.RecordingSession(False)
        with mock.patch.object(recording, "AVIRecorder", recorder):
            session.open_raw(self.gray, 25.0, self.log)
        self.assertEqual(calls[1], {"timeline": False})
        self.assertIn("Raw codec: MJPG (lossy fallback)", self.read_info(session))

    def test_window_recorder_path_and_rate(self):
        """The window video is window.avi in the session folder at 10 fps.

        Args: None.
        Returns: None.
        """
        session = recording.RecordingSession(False)
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        recorder = recording.open_window_recorder(session.folder, frame)
        recorder.stop()
        self.assertTrue(recorder.done.wait(5))
        self.assertEqual(recorder.path, os.path.join(session.folder, "window.avi"))
        self.assertEqual(recorder.fps, recording.WINDOW_FPS)
        self.assertEqual(recording.WINDOW_FPS, 10.0)

    def test_real_ffv1_raw_file_is_written(self):
        """A real OpenCV writer produces a raw.avi (FFV1 or its MJPG fallback).

        Args: None.
        Returns: None.
        """
        session = recording.RecordingSession(False)
        recorder = session.open_raw(np.full((16, 16), 90, dtype=np.uint8), 30.0, self.log)
        time.sleep(0.1)
        recorder.stop()
        self.assertTrue(recorder.done.wait(5))
        self.assertGreater(os.path.getsize(recorder.path), 0)


class SaveFrameTests(unittest.TestCase):
    """PNG saving and the title/notes sidecar."""

    def setUp(self):
        """Redirect output/ to a temp folder.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(paths, "OUTPUT_DIR", self.tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.log = ListLog()
        self.image = np.full((8, 8), 120, dtype=np.uint8)

    def test_saves_png_without_sidecar_when_no_text(self):
        """Without title or notes only the PNG is written.

        Args: None.
        Returns: None.
        """
        path = recording.save_frame(self.image, " ", "", False, self.log)
        self.assertTrue(os.path.isfile(path))
        self.assertFalse(os.path.exists(os.path.splitext(path)[0] + ".txt"))
        self.assertEqual(self.log.lines, [f"Saved: {path}"])

    def test_title_and_notes_go_to_sidecar_with_safe_filename(self):
        """The title is sanitized in the filename and kept in the sidecar.

        Args: None.
        Returns: None.
        """
        path = recording.save_frame(self.image, "run 1/a?", "some notes", False, self.log)
        self.assertTrue(path.endswith("_run_1_a.png"))
        with open(os.path.splitext(path)[0] + ".txt", encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("Source: Camera", text)
        self.assertIn("Title: run 1/a?", text)
        self.assertIn("some notes", text)

    def test_demo_frames_are_prefixed_and_labelled(self):
        """Demo saves carry the dummy_ prefix and Demo source label.

        Args: None.
        Returns: None.
        """
        path = recording.save_frame(self.image, "", "note", True, self.log)
        self.assertTrue(os.path.basename(path).startswith("dummy_frame_"))
        with open(os.path.splitext(path)[0] + ".txt", encoding="utf-8") as handle:
            self.assertIn("Source: Demo", handle.read())

    def test_camera_tag_names_the_file_and_sidecar_source(self):
        """A camera tag goes into the PNG name and the sidecar's source line.

        Args: None.
        Returns: None.
        """
        path = recording.save_frame(self.image, "t", "", False, self.log, "12345678")
        self.assertTrue(os.path.basename(path).startswith("frame_12345678_"))
        with open(os.path.splitext(path)[0] + ".txt", encoding="utf-8") as handle:
            self.assertIn("Source: Camera 12345678", handle.read())

    def test_frozen_frame_records_when_it_was_frozen(self):
        """A frame saved from a frozen view notes the freeze time in the sidecar.

        Args: None.
        Returns: None.
        """
        path = recording.save_frame(self.image, "t", "", False, self.log,
                                    frozen_at=datetime(2026, 10, 29, 15, 42, 7))
        with open(os.path.splitext(path)[0] + ".txt", encoding="utf-8") as handle:
            self.assertIn("Frozen at: 2026-10-29 15:42:07\n", handle.read())

    def test_live_frame_sidecar_has_no_freeze_line(self):
        """A frame saved from the live view does not mention freezing.

        Args: None.
        Returns: None.
        """
        path = recording.save_frame(self.image, "t", "", False, self.log)
        with open(os.path.splitext(path)[0] + ".txt", encoding="utf-8") as handle:
            self.assertNotIn("Frozen", handle.read())

    def test_imwrite_returning_false_logs_and_returns_none(self):
        """A refused PNG write is logged and returns None.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(recording.cv2, "imwrite", return_value=False):
            self.assertIsNone(recording.save_frame(self.image, "", "", False, self.log))
        self.assertTrue(self.log.lines[0].startswith("Could not save: "))

    def test_unusable_folder_logs_and_returns_none(self):
        """An OSError creating the folder is logged and returns None.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(recording, "output_folder", side_effect=OSError("ro")):
            self.assertIsNone(recording.save_frame(self.image, "", "", False, self.log))
        self.assertIn("Could not save frame: ro", self.log.lines)

    def test_opencv_error_logs_and_returns_none(self):
        """A cv2.error while encoding is logged and returns None.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(recording.cv2, "imwrite", side_effect=cv2.error("bad")):
            self.assertIsNone(recording.save_frame(self.image, "", "", False, self.log))
        self.assertTrue(self.log.lines[0].startswith("Could not save frame:"))

    def test_sidecar_failure_still_returns_png_path(self):
        """If the sidecar cannot be written the PNG is still reported saved.

        Args: None.
        Returns: None.
        """
        with mock.patch("builtins.open", side_effect=OSError("denied")):
            path = recording.save_frame(self.image, "t", "", False, self.log)
        self.assertTrue(os.path.isfile(path))
        self.assertIn("Frame saved, but could not save info: denied", self.log.lines)


if __name__ == "__main__":
    unittest.main()
