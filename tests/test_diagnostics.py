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
        self.assertIn("mode=Raw frames_back=1", plain)
        self.assertIn("outlier_filter=off", plain)
        self.assertIn("scaling=stretch clip 0%", plain)
        self.assertIn("noise_reduction=0", plain)
        self.assertIn("average=1 block", plain)
        self.assertIn("smoothing=off", plain)
        self.assertIn("exposure=8000us gain=0dB fps=30 clahe=on", plain)
        busy = diagnostics.settings_summary(Settings(
            mode="Previous", noise_reduction=40, average=8, rolling=True,
            smoothing=5, smoothing_method="Median", clahe=False, frames_back=4,
            outlier_filter=3, scale_range=False, difference_gain=6))
        self.assertIn("mode=Previous frames_back=4 noise_reduction=40", busy)
        self.assertIn("outlier_filter=3px", busy)
        self.assertIn("scaling=gain x6 exposure", busy)
        ordered = diagnostics.settings_summary(Settings(
            scale_range=False, gain_before_average=True, average_multiplier=2.5))
        self.assertIn("scaling=gain x42 before average, multiplier x2.5", ordered)
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



def speckle_image(grain, seed=0, height=480, width=720):
    """Simulate fully developed laser speckle of a chosen size.

    Args: grain is the speckle size in pixels; seed selects the pattern;
        height and width are the frame size.
    Returns: A uint8 array with a mean near 80.
    """
    rng = np.random.default_rng(seed)
    field = rng.normal(size=(height, width)) + 1j * rng.normal(size=(height, width))
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    pupil = (fx ** 2 + fy ** 2) < (1 / (2 * grain)) ** 2
    intensity = np.abs(np.fft.ifft2(np.fft.fft2(field) * pupil)) ** 2
    return np.clip(intensity / intensity.mean() * 80, 0, 255).astype(np.uint8)


def notch_across(image):
    """Remove the colour-subcarrier band from every row, as a decoder would.

    Args: image is a 2-D uint8 array.
    Returns: The filtered uint8 array.
    """
    spectrum = np.fft.rfft(image.astype(np.float32), axis=1)
    frequencies = np.fft.rfftfreq(image.shape[1])
    spectrum[:, (frequencies > 0.21) & (frequencies < 0.32)] *= 0.1
    filtered = np.fft.irfft(spectrum, n=image.shape[1], axis=1)
    return np.clip(filtered, 0, 255).astype(np.uint8)


class FrameMeasureTests(unittest.TestCase):
    """The measurements behind the frame-check line."""

    def test_level_stats_reports_percentiles_and_clipping(self):
        """Percentiles and the shares at the video and 8-bit limits.

        Args: None.
        Returns: None.
        """
        image = np.zeros((10, 10), np.uint8)
        image[:, 1:] = 100
        image[:, 9] = 255
        levels = diagnostics.level_stats(image)
        self.assertEqual((levels["p1"], levels["p50"], levels["p99"]), (0, 100, 255))
        self.assertAlmostEqual(levels["zero"], 10.0)
        self.assertAlmostEqual(levels["black"], 10.0)
        self.assertAlmostEqual(levels["white"], 10.0)
        self.assertAlmostEqual(levels["full"], 10.0)

    def test_change_stats_measures_frame_to_frame_difference(self):
        """Median, 95th percentile and maximum of the absolute difference.

        Args: None.
        Returns: None.
        """
        before = np.full((10, 10), 50, np.uint8)
        after = before.copy()
        after[:, :6] = 51
        after[0, 0] = 80
        self.assertEqual(diagnostics.change_stats(before, after),
                         {"median": 1, "p95": 1, "max": 30})
        self.assertEqual(diagnostics.change_stats(before, before)["max"], 0)

    def test_brightest_crop_finds_the_lit_area(self):
        """The crop lands on the brightest tile; small frames give what fits.

        Args: None.
        Returns: None.
        """
        image = np.zeros((480, 720), np.uint8)
        image[240:480, 470:720] = 200  # the bottom-right corner, past any whole tile
        crop, x, y = diagnostics.brightest_crop(image)
        self.assertEqual(crop.shape, (256, 256))
        self.assertGreater(x, 400)
        self.assertGreater(y, 200)
        self.assertGreater(crop.mean(), 170)
        small, x, y = diagnostics.brightest_crop(np.ones((40, 60), np.uint8))
        self.assertEqual((small.shape, x, y), ((40, 40), 0, 0))

    def test_speckle_stats_measures_contrast_and_size(self):
        """Speckle of 1, 2 and 4 px is measured as such, with contrast near 1.

        Args: None.
        Returns: None.
        """
        for grain in (1, 2, 4):
            crop, _, _ = diagnostics.brightest_crop(speckle_image(grain))
            stats = diagnostics.speckle_stats(crop)
            self.assertGreater(stats["contrast"], 0.7)
            self.assertAlmostEqual(stats["size_x"], grain, delta=0.4)
            self.assertAlmostEqual(stats["size_y"], grain, delta=0.4)

    def test_speckle_stats_tells_noise_and_flat_frames_apart(self):
        """Pixel noise has low contrast and ~1 px size; a flat crop gives None.

        Args: None.
        Returns: None.
        """
        noise = np.random.default_rng(3).integers(90, 110, (256, 256), dtype=np.uint8)
        stats = diagnostics.speckle_stats(noise)
        self.assertLess(stats["contrast"], 0.1)
        self.assertLess(stats["size_x"], 1.3)
        self.assertIsNone(diagnostics.speckle_stats(np.full((64, 64), 50, np.uint8)))

    def test_half_width_of_a_profile_that_never_halves(self):
        """A profile that stays above half gives twice its length.

        Args: None.
        Returns: None.
        """
        self.assertEqual(diagnostics._half_width(np.array([1.0, 0.9, 0.8])), 6.0)
        self.assertAlmostEqual(diagnostics._half_width(np.array([1.0, 0.75, 0.25])), 3.0)

    def test_detail_ratios_spot_a_decoder_notch(self):
        """Even speckle gives ~1; a notch across lowers only the subcarrier band.

        Args: None.
        Returns: None.
        """
        crop, _, _ = diagnostics.brightest_crop(speckle_image(2))
        even = diagnostics.detail_ratios(crop)
        self.assertAlmostEqual(even["low"], 1.0, delta=0.2)
        self.assertAlmostEqual(even["subcarrier"], 1.0, delta=0.2)
        crop, _, _ = diagnostics.brightest_crop(notch_across(speckle_image(2)))
        notched = diagnostics.detail_ratios(crop)
        self.assertGreater(notched["low"], 0.8)
        self.assertLess(notched["subcarrier"], 0.2)

    def test_detail_ratios_need_detail_down_and_any_detail(self):
        """Vertical stripes (nothing down) and a flat crop give None.

        Args: None.
        Returns: None.
        """
        stripes = np.tile((np.arange(64) % 4 * 60).astype(np.uint8), (64, 1))
        self.assertIsNone(diagnostics.detail_ratios(stripes))
        self.assertIsNone(diagnostics.detail_ratios(np.full((64, 64), 9, np.uint8)))

    def test_colour_spread_is_zero_for_grey(self):
        """Grey frames have no spread; coloured ones do.

        Args: None.
        Returns: None.
        """
        grey = np.full((4, 4, 3), 90, np.uint8)
        self.assertEqual(diagnostics.colour_spread(grey), (0.0, 0.0))
        grey[..., 0] = 100  # blue
        grey[..., 2] = 85   # red
        self.assertEqual(diagnostics.colour_spread(grey), (10.0, 5.0))

    def test_field_mismatch_separates_still_and_moved_fields(self):
        """Fields of one smooth scene agree; a shifted field does not.

        Args: None.
        Returns: None.
        """
        smooth = cv2.GaussianBlur(speckle_image(2), (0, 0), 3)
        self.assertLess(diagnostics.field_mismatch(smooth), 1.0)
        moved = smooth.copy()
        moved[1::2] = np.roll(smooth, 9, axis=1)[1::2]
        self.assertGreater(diagnostics.field_mismatch(moved), 1.5)
        self.assertIsNone(diagnostics.field_mismatch(np.zeros((6, 4), np.uint8)))


class FrameCheckMonitorTests(unittest.TestCase):
    """The periodic frame-check line."""

    def setUp(self):
        """Create a monitor and two consecutive speckle frames.

        Args: None.
        Returns: None.
        """
        self.log = ListLog()
        self.monitor = diagnostics.FrameCheckMonitor(self.log, interval=10.0)
        self.first = speckle_image(2, seed=1)
        self.second = cv2.add(self.first, np.ones_like(self.first))

    def test_measures_a_pair_then_waits_for_the_interval(self):
        """The first two frames make one INFO line; the next comes 10 s later.

        Args: None.
        Returns: None.
        """
        self.monitor.observe(self.first, 0.0, context)
        self.assertEqual(self.log.lines, [])
        self.monitor.observe(self.second, 0.04, context)
        self.assertEqual(len(self.log.lines), 1)
        level, line, _ = self.log.records[0]
        self.assertEqual(level, "INFO")
        self.assertTrue(line.startswith("Frame check: levels p1/p50/p99 "))
        self.assertIn("change to next frame median 1 p95 1 max 1", line)
        self.assertIn("repeated frames 0 of 2", line)
        self.assertIn("contrast", line)
        self.assertIn("detail across/down", line)
        self.assertNotIn("colour", line)
        self.assertNotIn("field mismatch", line)
        self.assertNotIn("view", line)
        self.assertTrue(line.endswith("; ctx"))
        for step in range(1, 100):
            self.monitor.observe(self.first, 0.04 + step * 0.1, context)
        self.assertEqual(len(self.log.lines), 1)
        self.monitor.observe(self.first, 10.1, context)
        self.monitor.observe(self.first, 10.2, context)
        self.assertEqual(len(self.log.lines), 2)
        self.assertIn("repeated frames", self.log.lines[1])
        self.assertIn("change to next frame median 0 p95 0 max 0", self.log.lines[1])

    def test_counts_repeated_frames(self):
        """Identical consecutive frames are counted since the previous line.

        Args: None.
        Returns: None.
        """
        self.monitor.interval = 1.0
        self.monitor.observe(self.first, 0.0, context)
        self.monitor.observe(self.second, 0.1, context)
        for step, frame in enumerate((self.first, self.first, self.second, self.second)):
            self.monitor.observe(frame, 0.2 + step * 0.1, context)
        self.monitor.observe(self.first, 1.2, context)
        self.monitor.observe(self.first, 1.3, context)
        self.assertIn("repeated frames 3 of 6", self.log.lines[-1])

    def test_adds_colour_fields_and_view(self):
        """Colour, field mismatch and the view range appear when known.

        Args: None.
        Returns: None.
        """
        colour = cv2.merge([self.first, self.first, self.first])
        self.monitor.note_view("Previous", 0, 2)
        self.monitor.observe(self.first, 0.0, context, colour=colour, interlaced=True)
        self.monitor.observe(self.second, 0.04, context, interlaced=True)
        line = self.log.lines[-1]
        self.assertIn("colour B−G 0.00 R−G 0.00", line)
        self.assertIn("field mismatch ", line)
        self.assertIn("view Previous input 0–2", line)
        self.monitor.note_view("Raw")
        self.monitor.observe(np.zeros((6, 4), np.uint8), 20.0, context, interlaced=True)
        self.monitor.observe(np.zeros((6, 4), np.uint8), 20.1, context, interlaced=True)
        line = self.log.lines[-1]
        self.assertIn("): flat;", line)
        self.assertNotIn("detail across/down", line)
        self.assertIn("field mismatch n/a", line)
        self.assertIn("view Raw;", line)

    def test_a_size_change_between_the_pair_skips_the_line(self):
        """A frame of another size (a switched source) is not compared.

        Args: None.
        Returns: None.
        """
        self.monitor.observe(self.first, 0.0, context)
        self.monitor.observe(self.first[:100], 0.04, context)
        self.assertEqual(self.log.lines, [])
        self.monitor.observe(self.first, 10.1, context)
        self.monitor.observe(self.second, 10.2, context)
        self.assertEqual(len(self.log.lines), 1)


if __name__ == "__main__":
    unittest.main()
