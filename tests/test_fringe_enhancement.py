"""Fringe-enhancement processing (floor, threshold, peak hold, normalization,
display tone) and the engine paths that use it, without a camera.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import tempfile
import unittest
from unittest import mock

import numpy as np

from espi_live import engine as engine_module
from espi_live import paths
from espi_live.engine import Engine
from espi_live.processing import (DARK_LEVEL, MAX_BRIGHTNESS_GAIN, NOISE_MARGIN,
                                  PEAK_SLICES, NoiseCalibration, PeakHold,
                                  RangeTracker, apply_levels, display_lut,
                                  even_out_brightness, local_normalize,
                                  stretch_levels)
from list_log import ListLog
from test_engine import wait_until


class LevelsTests(unittest.TestCase):
    """stretch_levels, apply_levels and RangeTracker."""

    def test_stretch_levels_exact_and_clipped(self):
        """0% gives the exact extremes; a clip ignores the outliers at each end.

        Args: None.
        Returns: None.
        """
        image = np.full((10, 10), 50, np.uint8)
        image[0, 0], image[0, 1] = 0, 255
        self.assertEqual(stretch_levels(image, 0), (0, 255))
        self.assertEqual(stretch_levels(image, 2.0), (50, 50))
        self.assertEqual(stretch_levels(image, 80.0), stretch_levels(image, 50.0))

    def test_apply_levels_maps_and_saturates(self):
        """low goes to 0, high to 255, and an empty range gives black.

        Args: None.
        Returns: None.
        """
        image = np.array([[10, 20, 30]], np.uint8)
        self.assertEqual(apply_levels(image, 10, 30).tolist(), [[0, 128, 255]])
        # A floor can push high past 255: small differences stay dark.
        self.assertEqual(apply_levels(image, 0, 510).tolist(), [[5, 10, 15]])
        self.assertFalse(apply_levels(image, 30, 30).any())

    def test_range_tracker_floor_release_and_reset(self):
        """Growth is followed at once, shrinking is gradual, the floor holds.

        Args: None.
        Returns: None.
        """
        ranges = RangeTracker()
        self.assertEqual(ranges.update(0, 100, 0, 1.0, 0.0), (0, 100))
        self.assertEqual(ranges.update(0, 200, 0, 1.0, 0.1), (0, 200))  # grows at once
        low, high = ranges.update(10, 50, 0, 1.0, 1.1)  # one time constant later
        self.assertLess(high, 200)
        self.assertGreater(high, 50)
        self.assertGreater(low, 0)
        self.assertLess(low, 10)
        self.assertEqual(ranges.update(5, 9, 0, 0.0, 1.2), (5, 9))  # 0 s: per frame
        self.assertEqual(ranges.update(5, 9, 30, 0.0, 1.3), (5, 35))  # floor
        ranges.reset()
        self.assertIsNone(ranges.high)
        self.assertEqual(ranges.update(1, 2, 0, 1.0, 2.0), (1, 2))


class NoiseCalibrationTests(unittest.TestCase):
    """The still-scene measurement behind Calibrate noise."""

    def test_floor_is_twice_the_median_peak(self):
        """The result is the median near-maximum times NOISE_MARGIN, clamped.

        Args: None.
        Returns: None.
        """
        calibration = NoiseCalibration(10.0, duration_s=2.0)
        self.assertIsNone(calibration.result())
        self.assertFalse(calibration.done(11.9))
        self.assertTrue(calibration.done(12.0))
        for level in (4, 6, 5):
            calibration.add(np.full((20, 20), level, np.uint8))
        self.assertEqual(calibration.result(), (5.0, int(5 * NOISE_MARGIN)))
        quiet = NoiseCalibration(0.0)
        quiet.add(np.zeros((4, 4), np.uint8))
        self.assertEqual(quiet.result(), (0.0, 1))
        loud = NoiseCalibration(0.0)
        loud.add(np.full((4, 4), 200, np.uint8))
        self.assertEqual(loud.result()[1], 255)


class PeakHoldTests(unittest.TestCase):
    """The sliding-window maximum."""

    def test_off_passes_the_image_through(self):
        """hold 0 returns the same array and forgets what was held.

        Args: None.
        Returns: None.
        """
        hold = PeakHold()
        image = np.zeros((2, 2), np.uint8)
        hold.update(np.full((2, 2), 9, np.uint8), 1.0, 0.0)
        self.assertIs(hold.update(image, 0.0, 0.1), image)
        self.assertIsNone(hold.current)

    def test_holds_the_peak_then_lets_it_go(self):
        """A spike stays for the window and drops out once it is older.

        Args: None.
        Returns: None.
        """
        hold = PeakHold()
        hold_s = 1.2
        span = hold_s / PEAK_SLICES
        quiet = np.zeros((3, 3), np.uint8)
        spike = quiet.copy()
        spike[1, 1] = 200
        first = hold.update(spike, hold_s, 0.0)
        self.assertEqual(first[1, 1], 200)
        self.assertIsNot(first, spike)
        same_slice = hold.update(quiet, hold_s, span / 2)
        self.assertEqual(same_slice[1, 1], 200)  # still in the current slice
        now = 0.0
        held = []
        for _ in range(PEAK_SLICES + 3):
            now += span
            held.append(int(hold.update(quiet, hold_s, now)[1, 1]))
        self.assertEqual(held[0], 200)  # closed into the window
        self.assertEqual(held[-1], 0)   # older than the window: gone
        self.assertLessEqual(len(hold.slices), PEAK_SLICES + 1)

    def test_window_or_shape_change_restarts(self):
        """A new window length or image size starts over.

        Args: None.
        Returns: None.
        """
        hold = PeakHold()
        hold.update(np.full((2, 2), 50, np.uint8), 1.0, 0.0)
        out = hold.update(np.zeros((2, 2), np.uint8), 2.0, 0.01)
        self.assertFalse(out.any())
        out = hold.update(np.zeros((3, 3), np.uint8), 2.0, 0.02)
        self.assertEqual(out.shape, (3, 3))


class NormalizeAndBrightnessTests(unittest.TestCase):
    """local_normalize and even_out_brightness."""

    def test_local_normalize_centres_on_mid_grey(self):
        """A flat image maps to 128; an edge spreads above and below it.

        Args: None.
        Returns: None.
        """
        flat = np.full((16, 16), 40, np.uint8)
        self.assertTrue((local_normalize(flat, 8) == 128).all())
        edge = np.zeros((16, 16), np.uint8)
        edge[:, 8:] = 100
        out = local_normalize(edge, 6)
        self.assertLess(int(out[8, 7]), 128)
        self.assertGreater(int(out[8, 8]), 128)
        # A high floor caps the boost: the same edge reads quieter.
        quiet = local_normalize(edge, 6, floor=400)
        self.assertLess(int(quiet[8, 8]), int(out[8, 8]))

    def test_even_out_brightness_flattens_and_caps(self):
        """Dim lit areas are raised (up to the cap); dark areas become 0.

        Args: None.
        Returns: None.
        """
        frame = np.full((64, 64), 200, np.uint8)
        frame[:, :32] = 50
        difference = np.full((64, 64), 10, np.uint8)
        out = even_out_brightness(difference, frame, sigma=4)
        self.assertGreater(int(out[32, 2]), int(out[32, 61]))
        self.assertLessEqual(int(out.max()), int(10 * MAX_BRIGHTNESS_GAIN))
        dark = np.full((64, 64), int(DARK_LEVEL) - 1, np.uint8)
        self.assertFalse(even_out_brightness(difference, dark).any())


class DisplayLutTests(unittest.TestCase):
    """The gain, S-curve and brightness table."""

    def test_off_and_each_control(self):
        """All off gives None; each control shapes the table as described.

        Args: None.
        Returns: None.
        """
        self.assertIsNone(display_lut())
        gain = display_lut(gain=2.0)
        self.assertEqual((int(gain[0]), int(gain[64]), int(gain[200])), (0, 128, 255))
        curve = display_lut(contrast=30.0)
        self.assertEqual((int(curve[0]), int(curve[255])), (0, 255))
        self.assertLess(int(curve[64]), 64)
        self.assertGreater(int(curve[192]), 192)
        self.assertEqual(display_lut(contrast=500.0).tolist(),
                         display_lut(contrast=100.0).tolist())
        bright = display_lut(brightness=20)
        self.assertEqual((int(bright[0]), int(bright[250])), (20, 255))


class EngineFringeTests(unittest.TestCase):
    """The worker with the fringe tools on, on synthetic video."""

    def setUp(self):
        """Point outputs at a temporary folder and start a demo engine.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.saved_output = paths.OUTPUT_DIR
        paths.OUTPUT_DIR = self.tmp.name
        self.log = ListLog()
        self.engine = Engine(self.log)
        self.engine.use_demo()
        self.engine.start()
        self.assertTrue(wait_until(lambda: self.engine.latest_packet() is not None))

    def tearDown(self):
        """Stop the engine and restore the output folder.

        Args: None.
        Returns: None.
        """
        self.engine.stop()
        self.engine.wait(5.0)
        paths.OUTPUT_DIR = self.saved_output
        self.tmp.cleanup()

    def next_packets(self, count=3):
        """Wait for a few new frames so the current settings have run.

        Args: count is how many new packets to wait for.
        Returns: The newest FramePacket.
        """
        first = self.engine.latest_packet().sequence
        self.assertTrue(wait_until(
            lambda: self.engine.latest_packet().sequence >= first + count))
        return self.engine.latest_packet()

    def test_peak_hold_preset_values_and_local_normalization_run(self):
        """Threshold, even brightness, peak hold, tone and normalization all run.

        Args: None.
        Returns: None.
        """
        self.engine.update_settings(
            mode="Previous", threshold=6, even_brightness=True, smoothing=3,
            smoothing_method="Median", peak_hold_s=1.0, steady_range_s=1.0,
            min_range=30, clip_percent=0.5, contrast=30.0, brightness=5,
            display_gain=1.5)
        packet = self.next_packets()
        self.assertEqual(packet.mode, "Previous")
        self.assertTrue(wait_until(
            lambda: "Scaled" in self.engine.status().get("range", "")))
        self.engine.update_settings(normalize_window=31)
        self.next_packets()
        self.assertTrue(wait_until(
            lambda: "Local normalization 31 px" in self.engine.status().get("range", "")))

    def test_calibration_needs_a_difference_view_then_sets_the_floor(self):
        """In Raw it warns and cancels; in Previous it measures and reports.

        Args: None.
        Returns: None.
        """
        short = lambda now: NoiseCalibration(now, duration_s=0.2)  # noqa: E731
        with mock.patch.object(engine_module, "NoiseCalibration", short):
            self.engine.update_settings(mode="Raw")
            self.next_packets()
            self.engine.request("calibrate_noise")
            self.assertTrue(wait_until(lambda: any(
                "needs the Previous or Reference view" in line
                for line in self.log.lines)))
            self.engine.update_settings(mode="Previous", min_range=0)
            self.next_packets()
            self.engine.request("calibrate_noise")
            self.assertTrue(wait_until(lambda: any(
                "sensitivity floor set to" in line for line in self.log.lines)))
        self.assertIn("Noise calibration: keep the object still for 2 s.", self.log.lines)
        events = []
        while not self.engine.events.empty():
            events.append(self.engine.events.get())
        floors = [payload for kind, payload in events if kind == "noise_calibrated"]
        self.assertEqual(len(floors), 1)
        self.assertEqual(self.engine.settings().min_range, floors[0])
        self.assertGreaterEqual(floors[0], 1)


if __name__ == "__main__":
    unittest.main()
