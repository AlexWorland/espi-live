"""Averaged references, the reference comparison and the noise-reduction notice.

Camera-free: the processor directly, the engine's helpers on the test thread,
and a demo engine for the frame loop.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import os
import tempfile
import unittest
from dataclasses import replace
from unittest import mock

import cv2
import numpy as np

from espi_live import paths, recording
from espi_live.engine import (ComparePacket, Engine, FramePacket, Settings,
                              condition_difference, finish_difference, present,
                              range_text, reference_text, view_hint)
from espi_live.processing import (ESPIProcessor, PeakHold, RangeTracker,
                                  ReferenceCapture, watermark_demo)
from list_log import ListLog
from test_engine import wait_until


def flat(value, shape=(80, 120)):
    """Make a constant Mono8 frame.

    Args: value is the pixel value; shape is (height, width).
    Returns: A uint8 array.
    """
    return np.full(shape, value, dtype=np.uint8)


class ReferenceCaptureTests(unittest.TestCase):
    """ESPIProcessor's averaged reference."""

    def test_mean_of_frames_is_exact_and_rounded(self):
        """The reference is the rounded mean of exactly the target frames.

        Args: None.
        Returns: None.
        """
        rng = np.random.default_rng(3)
        frames = [rng.integers(0, 256, (40, 60), dtype=np.uint8) for _ in range(7)]
        processor = ESPIProcessor()
        processor.set_reference(flat(9, (40, 60)), 1.0, 0.0)
        processor.start_reference(7, 12.0, 5.0)
        for frame in frames[:-1]:
            self.assertIsNone(processor.add_reference_frame(frame))
            np.testing.assert_array_equal(processor.ref, flat(9, (40, 60)))  # old one in use
        done = processor.add_reference_frame(frames[-1])
        self.assertEqual((done.count, done.target, done.manual), (7, 7, True))
        expected = np.rint(np.mean(np.stack(frames).astype(np.float64), axis=0))
        np.testing.assert_array_equal(processor.ref, expected.astype(np.uint8))
        self.assertEqual((processor.ref_time, processor.ref_noise, processor.ref_frames),
                         (5.0, 12.0, 7))
        self.assertIsNone(processor.ref_capture)
        self.assertIsNone(processor.add_reference_frame(frames[0]))  # no capture: no-op

    def test_one_frame_reference_is_unchanged(self):
        """set_reference still stores an exact copy of one frame.

        Args: None.
        Returns: None.
        """
        frame = np.arange(48, dtype=np.uint8).reshape(6, 8)
        processor = ESPIProcessor()
        processor.set_reference(frame, 2.0)
        frame[0, 0] = 200
        self.assertEqual(int(processor.ref[0, 0]), 0)
        self.assertEqual((processor.ref_frames, processor.ref_noise), (1, 0.0))
        np.testing.assert_array_equal(processor.difference(frame, "Reference"),
                                      cv2.absdiff(frame, processor.ref))

    def test_size_change_restarts_and_clear_cancels(self):
        """A new frame size restarts the sum; clear_reference drops everything.

        Args: None.
        Returns: None.
        """
        capture = ReferenceCapture(2, 0.0, 0.0, False)
        self.assertIsNone(capture.add(flat(10)))
        self.assertIsNone(capture.add(flat(50, (40, 120))))  # field change: restart
        np.testing.assert_array_equal(capture.add(flat(70, (40, 120))), flat(60, (40, 120)))
        processor = ESPIProcessor()
        processor.set_reference(flat(1), 0.0)
        processor.start_reference(3, 0.0, 0.0)
        processor.clear_reference()
        self.assertIsNone(processor.ref)
        self.assertIsNone(processor.ref_capture)
        self.assertEqual(processor.ref_frames, 0)


class ReferenceHelperTests(unittest.TestCase):
    """view_hint, reference_text, finish_difference and present."""

    def test_view_hints(self):
        """Each reference state has its own hint, none longer than the Qt measure.

        Args: None.
        Returns: None.
        """
        from espi_live.qt_app import LONGEST_HINT
        processor = ESPIProcessor()
        s = Settings(mode="Reference")
        hints = [view_hint(s, processor)]
        self.assertIn("press F", hints[-1])
        processor.start_reference(30, 0.0, 0.0)
        processor.add_reference_frame(flat(1))
        hints.append(view_hint(s, processor))
        self.assertEqual(hints[-1], "Capturing reference: 1/30 frames")
        processor.set_reference(flat(1), 0.0, 0.0)
        processor.start_reference(60, 0.0, 0.0)
        processor.ref_capture.count = 59
        hints.append(view_hint(replace(s, mode="Raw"), processor))
        self.assertIn("59/60 frames; the old one stays in use", hints[-1])
        processor.ref_capture = None
        hints.append(view_hint(replace(s, noise_reduction=95.0), processor))
        self.assertEqual(hints[-1],
                         "Reference was taken at noise reduction 0; press F to retake")
        hints.append(view_hint(replace(s, ref_refresh_s=2.0), processor))
        self.assertIn("refreshes every 2.0 s", hints[-1])
        hints.append(view_hint(s, processor))
        self.assertIn("ESPI difference view", hints[-1])
        hints.append(view_hint(replace(s, mode="Raw"), processor))
        self.assertIn("Raw view", hints[-1])
        hints.append(view_hint(replace(s, mode="Previous", noise_reduction=5.0), processor))
        self.assertIn("ESPI difference view", hints[-1])  # the notice is Reference-only
        self.assertTrue(all(len(hint) <= len(LONGEST_HINT) for hint in hints))

    def test_reference_text(self):
        """The description names the frame count, noise reduction and age.

        Args: None.
        Returns: None.
        """
        processor = ESPIProcessor()
        self.assertEqual(reference_text(processor, 0.0), "none yet")
        processor.set_reference(flat(1), 10.0, 20.0)
        self.assertEqual(reference_text(processor, 13.2),
                         "1 frame, noise reduction 20, 3 s old")
        processor.set_reference(flat(1), 10.0, 0.0, frames=8)
        self.assertEqual(reference_text(processor, 9.0),
                         "average of 8 frames, noise reduction 0, 0 s old")

    def test_finish_difference_steps(self):
        """Stretch, measured or unmeasured raw range, invert and CLAHE.

        Args: None.
        Returns: None.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        diff = np.tile(np.arange(0, 6, dtype=np.uint8), (64, 16))
        image, palette, gamma, levels = finish_difference(
            diff, Settings(clahe=False), clahe)
        self.assertEqual((palette, gamma, levels), ("Gray", 1.0, (0, 5, 0, 255)))
        self.assertEqual(int(image.max()), 255)
        image, _, _, levels = finish_difference(
            flat(3), Settings(clahe=False), clahe)
        self.assertEqual(levels, (3, 3, 0, 0))
        raw = Settings(scale_range=False, difference_gain=1.0, invert=True, clahe=False)
        image, _, _, levels = finish_difference(diff, raw, clahe)
        self.assertEqual(levels, (0, 5, 0, 5))
        np.testing.assert_array_equal(image, cv2.bitwise_not(diff))
        self.assertIsNone(finish_difference(diff, raw, clahe, measure=False)[3])
        for gamma_in in (1.0, 2.0):
            _, palette, gamma, _ = finish_difference(
                diff, Settings(palette="Turbo", gamma=gamma_in), clahe)
            self.assertEqual((palette, gamma), ("Turbo", 1.0))

    def test_finish_difference_main_steps(self):
        """Fixed gain, local normalization, peak hold, the sampler and the steady range.

        Args: None.
        Returns: None.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        diff = np.tile(np.arange(0, 6, dtype=np.uint8), (64, 16))
        gained = Settings(scale_range=False, difference_gain=10.0, clahe=False)
        image, _, _, levels = finish_difference(diff, gained, clahe)
        self.assertEqual(levels, (0, 5, 0, 50))
        self.assertEqual(int(image.max()), 50)
        image, _, _, levels = finish_difference(diff, gained, clahe, labview_order=True)
        np.testing.assert_array_equal(image, diff)  # gain was applied before averaging
        normalized = Settings(normalize_window=15, clahe=False)
        image, _, _, levels = finish_difference(diff, normalized, clahe)
        self.assertEqual(levels, (0, 5, None, None))
        self.assertIsNone(finish_difference(diff, normalized, clahe, measure=False)[3])
        self.assertEqual(image.shape, diff.shape)
        seen = []
        peaks = PeakHold()
        finish_difference(diff, Settings(peak_hold_s=1.0), clahe, peaks=peaks,
                          now=1.0, sample=lambda img, now: seen.append((img.shape, now)))
        self.assertEqual(seen, [(diff.shape, 1.0)])
        ranges = RangeTracker()
        loud = np.full((8, 8), 100, np.uint8)
        loud[0, 0] = 0
        finish_difference(loud, Settings(steady_range_s=10.0), clahe, ranges=ranges, now=0.0)
        _, _, _, levels = finish_difference(
            diff, Settings(steady_range_s=10.0), clahe, ranges=ranges, now=0.1)
        self.assertGreater(levels[1], 5)  # the range shrinks only slowly

    def test_range_text(self):
        """The status text names the scaled range or the normalization window.

        Args: None.
        Returns: None.
        """
        self.assertEqual(range_text((1, 9, 0, 255), Settings()),
                         "Input: 1–9    Scaled: 0–255")
        self.assertEqual(range_text((1, 9, None, None), Settings(normalize_window=15)),
                         "Input: 1–9    Local normalization 15 px")

    def test_condition_difference(self):
        """Threshold, even brightness, LabVIEW-order gain and outliers run before averaging.

        Args: None.
        Returns: None.
        """
        diff = np.array([[1, 5], [9, 3]], np.uint8)
        frame = flat(100, (2, 2))
        out, labview = condition_difference(diff, frame, Settings())
        self.assertFalse(labview)
        np.testing.assert_array_equal(out, diff)
        out, _ = condition_difference(diff, frame, Settings(threshold=4))
        np.testing.assert_array_equal(out, [[0, 5], [9, 0]])
        out, labview = condition_difference(
            diff, frame, Settings(scale_range=False, gain_before_average=True,
                                  difference_gain=2.0))
        self.assertTrue(labview)
        np.testing.assert_array_equal(out, diff * 2)
        out, _ = condition_difference(diff, frame, Settings(even_brightness=True))
        self.assertEqual(out.shape, diff.shape)

    def test_present_flips_then_watermarks(self):
        """present mirrors first, so the watermark text is never mirrored.

        Args: None.
        Returns: None.
        """
        ramp = np.tile(np.arange(0, 240, 2, dtype=np.uint8), (80, 1))
        self.assertIs(present(ramp, False, False), ramp)
        np.testing.assert_array_equal(present(ramp, True, False), ramp[:, ::-1])
        np.testing.assert_array_equal(present(ramp, True, True),
                                      watermark_demo(ramp[:, ::-1]))
        tone = np.arange(256, dtype=np.uint8)[::-1].copy()
        np.testing.assert_array_equal(present(ramp, False, False, tone), 255 - ramp)


class EngineReferenceTests(unittest.TestCase):
    """Engine reference handling, driven on the test thread."""

    def setUp(self):
        """Make an unstarted demo engine and a temporary output folder.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.saved_output = paths.OUTPUT_DIR
        paths.OUTPUT_DIR = self.tmp.name
        self.log = ListLog()
        self.engine = Engine(self.log)
        self.processor = ESPIProcessor()

    def tearDown(self):
        """Restore the output folder.

        Args: None.
        Returns: None.
        """
        paths.OUTPUT_DIR = self.saved_output
        self.tmp.cleanup()

    def logged(self, text):
        """Report whether any log line contains text.

        Args: text is the substring.
        Returns: True when found.
        """
        return any(text in line for line in self.log.lines)

    def test_f_with_one_frame_matches_the_old_behaviour(self):
        """Ref. frames 1 stores the current frame at once and resets averaging.

        Args: None.
        Returns: None.
        """
        self.engine.update_settings(noise_reduction=10)
        self.processor.start_reference(5, 0.0, 0.0)  # an old capture is dropped
        self.processor.avg_count = 3
        self.engine.request("reference")
        self.engine._handle_actions(self.processor, flat(42))
        np.testing.assert_array_equal(self.processor.ref, flat(42))
        self.assertIsNone(self.processor.ref_capture)
        self.assertEqual((self.processor.ref_noise, self.processor.avg_count), (10, 0))
        self.assertTrue(self.logged("Captured reference frame at noise reduction 10."))
        self.engine.request("reference")
        self.engine._handle_actions(self.processor, None)
        self.assertTrue(self.logged("No frame available yet"))

    def test_f_with_several_frames_averages_the_next_ones(self):
        """Ref. frames N starts a capture that ends in one log line.

        Args: None.
        Returns: None.
        """
        self.engine.update_settings(ref_frames=3, noise_reduction=4)
        self.engine.request("reference")
        self.engine._handle_actions(self.processor, None)  # needs no current frame
        self.assertTrue(self.logged("averaging the next 3 frames"))
        s = self.engine.settings()
        self.processor.avg_count = 2
        for value in (10, 20):
            self.engine._reference_frame(self.processor, flat(value), s)
        self.assertIsNone(self.processor.ref)
        self.assertEqual(self.processor.avg_count, 2)
        self.engine._reference_frame(self.processor, flat(60), s)
        np.testing.assert_array_equal(self.processor.ref, flat(30))
        self.assertEqual(self.processor.avg_count, 0)
        self.assertTrue(self.logged(
            "Captured reference frame (average of 3) at noise reduction 4."))

    def test_automatic_capture_is_quiet_and_keeps_a_rolling_average(self):
        """A refresh capture logs nothing; a rolling average keeps running.

        Args: None.
        Returns: None.
        """
        self.processor.start_reference(2, 0.0, 0.0, manual=False)
        self.processor.rolling = flat(5).astype(np.float32)
        s = Settings(rolling=True)
        self.engine._reference_frame(self.processor, flat(1), s)
        self.engine._reference_frame(self.processor, flat(3), s)
        np.testing.assert_array_equal(self.processor.ref, flat(2))
        self.assertIsNotNone(self.processor.rolling)
        self.assertFalse(self.logged("Captured"))

    def test_refresh_shorter_than_the_capture_is_noted(self):
        """Ref. frames that outlast the refresh time get one note.

        Args: None.
        Returns: None.
        """
        self.engine._note_refresh_overlap(Settings(ref_refresh_s=0.5, ref_frames=30), None)
        self.assertTrue(self.logged("averaging 30 frames takes about 1.0 s, longer "
                                    "than the 0.5 s refresh"))
        self.log.lines.clear()
        for s, fps in ((Settings(ref_refresh_s=2.0, ref_frames=30), 30.0),
                       (Settings(ref_refresh_s=0.0, ref_frames=30), None),
                       (Settings(ref_refresh_s=0.1, ref_frames=1), None)):
            self.engine._note_refresh_overlap(s, fps)
        self.assertEqual(self.log.lines, [])

    def test_noise_reduction_change_notice(self):
        """A held reference is flagged once; a running capture is cancelled.

        Args: None.
        Returns: None.
        """
        self.engine._noise_changed(self.processor, 0.0, 10.0)  # no reference: quiet
        self.processor.set_reference(flat(1), 0.0, 0.0)
        self.engine._noise_changed(self.processor, 0.0, 10.0)
        self.assertTrue(self.logged(
            "Noise reduction is now 10, but the reference was taken at 0. "
            "Press F to retake it."))
        self.log.lines.clear()
        self.engine._noise_changed(self.processor, 10.0, 20.0)  # already flagged
        self.assertEqual(self.log.lines, [])
        np.testing.assert_array_equal(self.processor.ref, flat(1))  # never retaken
        for manual, text in ((True, "Press F to start again."),
                             (False, "The next refresh starts a new one.")):
            self.processor.start_reference(4, 20.0, 0.0, manual=manual)
            self.engine._noise_changed(self.processor, 20.0, 30.0)
            self.assertIsNone(self.processor.ref_capture)
            self.assertTrue(self.logged(text))

    def test_compare_in_reference_view_reuses_the_published_packet(self):
        """In Reference view the difference is the main packet itself.

        Args: None.
        Returns: None.
        """
        self.assertIsNone(self.engine.latest_compare())
        self.engine.set_compare(True)
        clahe = cv2.createCLAHE()
        s = Settings(mode="Reference", gamma=1.5, average=4)
        self.processor.set_reference(flat(20), 0.0, 0.0, frames=8)
        self.processor.start_reference(8, 0.0, 0.0)
        packet = FramePacket(7, flat(99), "Gray", 1.5, "Reference", False)
        self.engine._publish_compare(self.processor, flat(30), s, packet, clahe, 4.0)
        compare = self.engine.latest_compare()
        self.assertIs(compare.difference, packet)
        self.assertEqual(compare.sequence, 7)
        np.testing.assert_array_equal(compare.reference.image, flat(20))
        np.testing.assert_array_equal(compare.current.image, flat(30))
        self.assertEqual((compare.reference.palette, compare.reference.gamma,
                          compare.current.gamma), ("Gray", 1.5, 1.5))
        self.assertEqual(compare.reference_info, "capturing 0/8; in use: average "
                         "of 8 frames, noise reduction 0, 4 s old")
        self.assertEqual(compare.difference_info, "the Reference view, average 4")
        self.engine.set_compare(False)
        self.assertIsNone(self.engine.latest_compare())

    def test_compare_in_other_views_computes_one_difference(self):
        """Outside Reference view the difference is processed without averaging.

        Args: None.
        Returns: None.
        """
        clahe = cv2.createCLAHE()
        ramp = np.tile(np.arange(0, 120, dtype=np.uint8), (80, 1))
        s = Settings(mode="Previous", flip_horizontal=True, clahe=False)
        self.processor.set_reference(flat(10), 0.0, 0.0)
        self.engine.demo_source = object()  # demo: watermark every image
        packet = FramePacket(3, flat(0), "Gray", 1.0, "Previous", True)
        self.engine._publish_compare(self.processor, ramp, s, packet, clahe, 1.0)
        compare = self.engine.latest_compare()
        expected, _, _, _ = finish_difference(cv2.absdiff(ramp, flat(10)), s, clahe)
        np.testing.assert_array_equal(compare.difference.image,
                                      watermark_demo(expected[:, ::-1]))
        np.testing.assert_array_equal(compare.current.image, watermark_demo(ramp[:, ::-1]))
        self.assertTrue(compare.demo and compare.difference.demo)
        self.assertIn("no averaging (the view is Previous)", compare.difference_info)
        s = Settings(mode="Reference", average=3, rolling=True)
        self.processor.set_reference(flat(10, (40, 120)), 0.0, 0.0)  # other size
        self.engine._publish_compare(self.processor, ramp, s, packet, clahe, 1.0)
        compare = self.engine.latest_compare()
        self.assertIsNone(compare.reference)
        self.assertIsNone(compare.difference)
        self.assertEqual(compare.difference_info, "the Reference view, average 3 rolling")
        self.processor.clear_reference()
        self.engine._publish_compare(self.processor, ramp, Settings(mode="Reference"),
                                     packet, clahe, 1.0)
        self.assertEqual(self.engine.latest_compare().reference_info, "none yet")
        self.assertIn("no averaging", self.engine.latest_compare().difference_info)

    def test_saving_a_comparison_writes_exactly_what_is_shown(self):
        """Save all three writes each rendered image and one notes file.

        Args: None.
        Returns: None.
        """
        diff = FramePacket(1, np.tile(np.arange(0, 256, 4, dtype=np.uint8), (8, 1)),
                           "Turbo", 1.0, "Reference", False)
        ref = FramePacket(1, flat(50, (8, 64)), "Gray", 2.0, "Reference frame", False)
        cur = FramePacket(1, flat(60, (8, 64)), "Gray", 2.0, "Current", False)
        compare = ComparePacket(1, cur, ref, diff, "1 frame", "the Reference view", False)
        self.engine.request("save_compare", packet=compare, title="run 2", notes="n")
        self.assertTrue(self.engine.save_pending())
        self.engine._handle_actions(self.processor, None)
        self.assertFalse(self.engine.save_pending())
        folder = os.path.join(self.tmp.name, "frames")
        names = sorted(os.listdir(folder))
        self.assertEqual(len(names), 4)
        for name, part in (("current", cur), ("difference", diff), ("reference", ref)):
            path = os.path.join(folder, next(n for n in names if f"_{name}_run_2" in n))
            saved = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(saved, part.render())
        with open(os.path.join(folder, next(n for n in names if n.endswith(".txt"))),
                  encoding="utf-8") as info:
            text = info.read()
        self.assertIn("Reference: 1 frame", text)
        self.assertIn("Difference: the Reference view", text)
        self.assertIn("Title: run 2", text)

        self.engine.demo_source = object()
        bare = ComparePacket(2, cur, None, None, "none yet", "x", True)
        self.engine.request("save_compare", packet=bare, title="", notes="")
        self.engine._handle_actions(self.processor, None)
        self.assertTrue(self.logged("No reference yet: saving the current frame only."))
        self.assertEqual(len([n for n in os.listdir(folder) if n.startswith("dummy_")]), 2)


class SaveFrameSetTests(unittest.TestCase):
    """recording.save_frame_set failure paths."""

    def setUp(self):
        """Point output at a temporary folder.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.saved_output = paths.OUTPUT_DIR
        paths.OUTPUT_DIR = self.tmp.name
        self.log = ListLog()

    def tearDown(self):
        """Restore the output folder.

        Args: None.
        Returns: None.
        """
        paths.OUTPUT_DIR = self.saved_output
        self.tmp.cleanup()

    def test_failures_are_logged(self):
        """A refused image, an unusable folder and an unwritable notes file.

        Args: None.
        Returns: None.
        """
        images = [("current", flat(1)), ("reference", flat(2))]
        with mock.patch.object(recording.cv2, "imwrite", return_value=False):
            self.assertEqual(recording.save_frame_set(images, "", "", False, self.log), [])
        self.assertTrue(any("Could not save" in line for line in self.log.lines))
        with mock.patch.object(recording, "output_folder", side_effect=OSError("ro")), \
                mock.patch.object(recording.error_reports, "caught") as caught:
            self.assertEqual(recording.save_frame_set(images, "", "", False, self.log), [])
        caught.assert_called_once()
        real_open = open

        def refuse_text(path, *args, **kwargs):
            """Refuse to open the notes file only.

            Args: path and the rest are open()'s arguments.
            Returns: The opened file for other paths.
            """
            if str(path).endswith(".txt"):
                raise OSError("disk full")
            return real_open(path, *args, **kwargs)

        with mock.patch("builtins.open", refuse_text):
            saved = recording.save_frame_set(images, "t", "", False, self.log, tag="cam 1")
        self.assertEqual(len(saved), 2)
        self.assertIn("_cam_1_", saved[0])
        self.assertTrue(any("could not save info" in line for line in self.log.lines))


class EngineLoopReferenceTests(unittest.TestCase):
    """The frame loop with a demo source of known frames."""

    def setUp(self):
        """Start a fast demo engine whose frames cycle through four flat values.

        Any four consecutive frames average to 25, wherever a capture starts.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.saved_output = paths.OUTPUT_DIR
        paths.OUTPUT_DIR = self.tmp.name
        self.log = ListLog()
        self.engine = Engine(self.log)
        self.engine.use_demo()
        frames = [flat(value) for value in (10, 20, 30, 40)]
        counter = iter(range(10 ** 9))
        self.engine.demo_source.next_frame = (
            lambda exposure_us, gain_db, now: frames[next(counter) % 4].copy())
        self.engine.update_settings(fps=200, mode="Reference", clahe=False)
        self.engine.start()

    def tearDown(self):
        """Stop the engine and restore the output folder.

        Args: None.
        Returns: None.
        """
        self.engine.stop()
        self.assertTrue(self.engine.wait(10))
        paths.OUTPUT_DIR = self.saved_output
        self.tmp.cleanup()

    def logged(self, text):
        """Report whether any log line contains text.

        Args: text is the substring.
        Returns: True when found.
        """
        return any(text in line for line in self.log.lines)

    def test_averaged_reference_reaches_the_comparison(self):
        """F with Ref. frames 4 stores the mean, shown in the comparison.

        Args: None.
        Returns: None.
        """
        self.assertTrue(wait_until(lambda: self.engine.latest_packet() is not None))
        self.engine.set_compare(True)
        self.assertTrue(wait_until(lambda: self.engine.latest_compare() is not None))
        self.assertIsNone(self.engine.latest_compare().reference)
        self.engine.update_settings(ref_frames=4)
        self.assertTrue(wait_until(lambda: self.logged("Reference frames: 4")))
        self.engine.request("reference")
        self.engine.update_settings(mode="Previous")  # a view change does not stop it
        self.assertTrue(wait_until(lambda: self.logged(
            "Captured reference frame (average of 4) at noise reduction 0.")))
        self.assertTrue(wait_until(lambda: (
            self.engine.latest_compare().reference is not None)))
        compare = self.engine.latest_compare()
        np.testing.assert_array_equal(compare.reference.image[60:], flat(25)[60:])
        self.assertEqual(compare.reference.mode, "Reference frame")
        self.engine.update_settings(noise_reduction=10, mode="Reference")
        self.assertTrue(wait_until(lambda: self.logged("but the reference was taken at 0")))
        self.assertTrue(wait_until(lambda: "taken at noise reduction 0" in
                                   self.engine.status()["hint"]))

    def test_auto_refresh_builds_averaged_references(self):
        """Ref. refresh with Ref. frames N captures N-frame references by itself.

        Args: None.
        Returns: None.
        """
        self.assertTrue(wait_until(lambda: self.engine.latest_packet() is not None))
        self.engine.set_compare(True)
        self.engine.update_settings(ref_frames=4, ref_refresh_s=0.01)
        self.assertTrue(wait_until(lambda: self.logged("references are captured back to back")))
        self.assertTrue(wait_until(lambda: (
            self.engine.latest_compare() is not None and
            self.engine.latest_compare().reference is not None)))
        compare = self.engine.latest_compare()
        np.testing.assert_array_equal(compare.reference.image[60:], flat(25)[60:])
        self.assertIn("average of 4 frames", compare.reference_info)
        self.assertFalse(self.logged("Captured reference"))  # refreshes are quiet


if __name__ == "__main__":
    unittest.main()
