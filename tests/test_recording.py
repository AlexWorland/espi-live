"""Camera-free checks for AVI recording and still-frame saving.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import os
import tempfile
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from espi_live import paths, recording


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


class RecordingPathTests(unittest.TestCase):
    """AVI path naming."""

    def test_paths_are_under_recordings_with_demo_prefix(self):
        """Demo recordings get the dummy_ prefix and a lowercase mode.

        Args: None.
        Returns: None.
        """
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(paths, "OUTPUT_DIR", tmp):
            live = recording.recording_path("Raw", False)
            demo = recording.recording_path("Previous", True)
        self.assertTrue(os.path.basename(live).startswith("espi_raw_"))
        self.assertTrue(os.path.basename(demo).startswith("dummy_espi_previous_"))
        self.assertTrue(live.endswith(".avi"))
        self.assertEqual(os.path.dirname(live), os.path.join(tmp, "recordings"))

    def test_camera_tag_is_sanitized_into_the_name(self):
        """A camera tag is added after the mode, with unsafe characters replaced.

        Args: None.
        Returns: None.
        """
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(paths, "OUTPUT_DIR", tmp):
            path = recording.recording_path("Raw", False, "cam 1/a")
        self.assertTrue(os.path.basename(path).startswith("espi_raw_cam_1_a_"))


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
