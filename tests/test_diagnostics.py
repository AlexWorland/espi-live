"""Camera-free checks for espi_live.diagnostics."""

import unittest

import cv2
import numpy as np

from espi_live import diagnostics
from espi_live.engine import Settings
from list_log import ListLog


def context():
    """Stand in for the engine's context callable.

    Args: None.
    Returns: A fixed text.
    """
    return "ctx"


def noise_image(seed=1):
    """Build a sharp, detailed test frame.

    Args: seed selects the pseudo-random pattern.
    Returns: A 480 x 640 uint8 array of noise.
    """
    return np.random.default_rng(seed).integers(0, 256, (480, 640), dtype=np.uint8)


class ThrottleTests(unittest.TestCase):
    """One line per period, with a count of the rest."""

    def test_allows_one_line_per_period_and_counts_the_rest(self):
        """Lines inside the period are held back and counted once.

        Args: None.
        Returns: None.
        """
        throttle = diagnostics.Throttle(1.0)
        self.assertTrue(throttle.allow(10.0))
        self.assertEqual(throttle.note(), "")
        self.assertFalse(throttle.allow(10.4))
        self.assertFalse(throttle.allow(10.9))
        self.assertTrue(throttle.allow(11.0))
        self.assertEqual(throttle.note(), " (+2 more since the last line)")
        self.assertEqual(throttle.note(), "")


class SummaryTests(unittest.TestCase):
    """Settings text and the sharpness measure."""

    def test_settings_summary_names_what_filters_the_frame(self):
        """Averaging, smoothing and noise reduction appear in the summary.

        Args: None.
        Returns: None.
        """
        plain = diagnostics.settings_summary(Settings())
        self.assertIn("mode=Raw", plain)
        self.assertIn("noise_reduction=0", plain)
        self.assertIn("average=1 block", plain)
        self.assertIn("smoothing=off", plain)
        self.assertIn("exposure=8000us gain=0dB fps=30 clahe=on", plain)
        busy = diagnostics.settings_summary(Settings(
            mode="Previous", noise_reduction=40, average=8, rolling=True,
            smoothing=5, smoothing_method="Median", clahe=False))
        self.assertIn("mode=Previous noise_reduction=40", busy)
        self.assertIn("average=8 rolling", busy)
        self.assertIn("smoothing=Median 5", busy)
        self.assertIn("clahe=off", busy)

    def test_sharpness_falls_with_blur_for_large_and_small_frames(self):
        """Blur lowers the value; small frames are measured without resizing.

        Args: None.
        Returns: None.
        """
        sharp = noise_image()
        blurred = cv2.GaussianBlur(sharp, (0, 0), 3)
        self.assertGreater(diagnostics.sharpness(sharp),
                           5 * diagnostics.sharpness(blurred))
        small = sharp[:48, :64]
        self.assertGreater(diagnostics.sharpness(small),
                           5 * diagnostics.sharpness(cv2.GaussianBlur(small, (0, 0), 3)))
        self.assertEqual(diagnostics.sharpness(np.zeros((48, 64), np.uint8)), 0.0)


class FrameGapMonitorTests(unittest.TestCase):
    """Gaps between good frames and bursts of incomplete ones."""

    def setUp(self):
        """Create a monitor with an in-memory log.

        Args: None.
        Returns: None.
        """
        self.log = ListLog()
        self.monitor = diagnostics.FrameGapMonitor(self.log)

    def test_steady_frames_write_nothing(self):
        """Frames at the expected interval produce no lines.

        Args: None.
        Returns: None.
        """
        for index in range(20):
            self.monitor.frame_ok(index * 0.033, 0.033, context)
        self.assertEqual(self.log.lines, [])

    def test_a_long_gap_is_logged_with_context(self):
        """A pause well beyond the expected interval is a warning.

        Args: None.
        Returns: None.
        """
        self.monitor.frame_ok(0.0, 0.033, context)
        self.monitor.frame_ok(0.5, 0.033, context)
        self.assertEqual(len(self.log.records), 1)
        level, message, _ = self.log.records[0]
        self.assertEqual(level, "WARNING")
        self.assertIn("Frame gap: 0.500 s", message)
        self.assertIn("1 gap(s) since start, longest 0.500 s", message)
        self.assertTrue(message.endswith("; ctx"))

    def test_gap_lines_are_throttled_but_counted(self):
        """A run of gaps writes one line and reports the rest on the next.

        Args: None.
        Returns: None.
        """
        self.monitor.frame_ok(0.0, 0.033, context)
        self.monitor.frame_ok(0.4, 0.033, context)
        self.monitor.frame_ok(0.8, 0.033, context)
        self.assertEqual(len(self.log.lines), 1)
        self.monitor.frame_ok(1.6, 0.033, context)
        self.assertEqual(len(self.log.lines), 2)
        self.assertIn("3 gap(s) since start, longest 0.800 s", self.log.lines[1])
        self.assertIn("(+1 more since the last line)", self.log.lines[1])

    def test_a_burst_of_incomplete_frames_is_logged_when_it_ends(self):
        """The burst line gives the count and duration at the next good frame.

        Args: None.
        Returns: None.
        """
        self.monitor.frame_ok(0.0, 0.033, context)
        for moment in (0.03, 0.06, 0.09):
            self.monitor.incomplete(moment)
        self.assertEqual(self.log.lines, [])
        self.monitor.frame_ok(0.12, 0.033, context)
        self.assertEqual(len(self.log.lines), 1)
        self.assertIn("Incomplete-frame burst: 3 frame(s) over 0.06 s ended "
                      "(3 incomplete since start)", self.log.lines[0])

    def test_burst_lines_are_throttled(self):
        """Repeated single incomplete frames do not flood the log.

        Args: None.
        Returns: None.
        """
        self.monitor.frame_ok(0.0, 0.033, context)
        for moment in (0.1, 0.2, 0.3):
            self.monitor.incomplete(moment - 0.01)
            self.monitor.frame_ok(moment, 0.033, context)
        self.assertEqual(len(self.log.lines), 1)


class StreamCounterMonitorTests(unittest.TestCase):
    """Lost and dropped frame counters."""

    def setUp(self):
        """Create a monitor with an in-memory log.

        Args: None.
        Returns: None.
        """
        self.log = ListLog()
        self.monitor = diagnostics.StreamCounterMonitor(self.log)

    def test_no_counters_are_noted_once(self):
        """A camera without counters is mentioned a single time.

        Args: None.
        Returns: None.
        """
        self.monitor.update({}, context)
        self.monitor.update({}, context)
        self.assertEqual(self.log.records,
                         [("DEBUG", "Stream counters: none readable on this camera", None)])

    def test_first_reading_is_a_baseline_and_rises_are_warnings(self):
        """Only increases of loss counters are warnings.

        Args: None.
        Returns: None.
        """
        self.monitor.update({"StreamDeliveredFrameCount": 10,
                             "StreamLostFrameCount": 0}, context)
        self.assertEqual([record[0] for record in self.log.records], ["DEBUG"])
        self.monitor.update({"StreamDeliveredFrameCount": 40,
                             "StreamLostFrameCount": 0}, context)
        self.assertEqual(len(self.log.records), 1)  # delivering frames is normal
        self.monitor.update({"StreamDeliveredFrameCount": 50,
                             "StreamLostFrameCount": 3}, context)
        level, message, _ = self.log.records[-1]
        self.assertEqual(level, "WARNING")
        self.assertIn("rose: StreamLostFrameCount +3", message)
        self.assertIn("StreamDeliveredFrameCount=50", message)
        self.assertTrue(message.endswith("; ctx"))


class SharpnessMonitorTests(unittest.TestCase):
    """Drops below the typical sharpness and the recovery."""

    def setUp(self):
        """Create a monitor sampling every frame after a short warm-up.

        Args: None.
        Returns: None.
        """
        self.log = ListLog()
        self.monitor = diagnostics.SharpnessMonitor(self.log, "camera frame")
        self.sharp = noise_image()
        self.soft = cv2.GaussianBlur(self.sharp, (0, 0), 3)
        self.clock = 0.0

    def feed(self, image, seconds):
        """Observe a frame every sampling interval for a while.

        Args: image is the frame; seconds is how long to keep feeding.
        Returns: None.
        """
        end = self.clock + seconds
        while self.clock < end:
            self.monitor.observe(image, self.clock, context)
            self.clock += diagnostics.SHARPNESS_INTERVAL_S

    def test_samples_are_taken_at_a_limited_rate(self):
        """Frames arriving faster than the interval are not measured.

        Args: None.
        Returns: None.
        """
        self.monitor.observe(self.sharp, 0.0, context)
        self.monitor.observe(self.sharp, 0.01, context)
        self.assertEqual(len(self.monitor.history), 1)

    def test_a_steady_picture_writes_nothing(self):
        """No episode begins without a drop.

        Args: None.
        Returns: None.
        """
        self.feed(self.sharp, 10.0)
        self.assertEqual(self.log.lines, [])

    def test_no_verdict_before_enough_samples(self):
        """A blurred frame during warm-up only joins the history.

        Args: None.
        Returns: None.
        """
        self.feed(self.sharp, 1.0)
        self.feed(self.soft, 1.0)
        self.assertEqual(self.log.lines, [])

    def test_flat_frames_never_start_an_episode(self):
        """A typical value of zero gives no ratio, so nothing is reported.

        Args: None.
        Returns: None.
        """
        self.feed(np.zeros((48, 64), np.uint8), 10.0)
        self.assertEqual(self.log.lines, [])

    def test_a_drop_and_its_recovery_are_logged_with_the_duration(self):
        """The episode starts and ends with one line each; the baseline holds.

        Args: None.
        Returns: None.
        """
        self.feed(self.sharp, 8.0)
        self.feed(self.soft, 3.5)
        self.feed(self.sharp, 2.0)
        starts = [line for line in self.log.lines if "Sharpness drop" in line]
        ends = [line for line in self.log.lines if "Sharpness recovered" in line]
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(ends), 1)
        self.assertIn("(camera frame)", starts[0])
        self.assertTrue(starts[0].endswith("; ctx"))
        self.assertIn("after 3.5 s", ends[0])
        self.assertTrue(all(record[0] == "WARNING" for record in self.log.records))
        self.feed(self.soft, 1.0)  # a second episode is judged against the same typical
        self.assertEqual(len([line for line in self.log.lines
                              if "Sharpness drop" in line]), 2)


if __name__ == "__main__":
    unittest.main()
