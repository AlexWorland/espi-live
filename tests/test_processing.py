"""Camera-free checks for the processing helpers' edge cases.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import unittest

import numpy as np

from espi_live.processing import (FULL_SCALE, MAX_BRIGHTNESS_GAIN,
                                  MAX_DENOISE_PERCENT, MAX_DENOISE_STRENGTH,
                                  MAX_FRAMES_BACK, NOISE_MARGIN, PEAK_SLICES,
                                  DemoSource, ESPIProcessor, NoiseCalibration,
                                  PeakHold, RangeTracker, ReferenceCapture,
                                  apply_gain, apply_levels, denoise_strength,
                                  display_lut, even_out_brightness,
                                  local_normalize, normalize_min_max,
                                  reject_outliers, saturate, single_field,
                                  stretch_contrast, stretch_levels, to_uint8,
                                  watermark_demo)


class NormalizeTests(unittest.TestCase):
    """Min/max normalization of constant and varying images."""

    def test_constant_image_becomes_zeros(self):
        """A constant image maps to zeros and reports its single level.

        Args: None.
        Returns: None.
        """
        image = np.full((4, 4), 77, dtype=np.uint8)
        scaled, low, high = normalize_min_max(image)
        self.assertEqual((low, high), (77, 77))
        self.assertFalse(scaled.any())

    def test_varying_image_spans_full_range(self):
        """A two-level image is stretched to 0 and 255.

        Args: None.
        Returns: None.
        """
        image = np.array([[10, 20]], dtype=np.uint8)
        scaled, low, high = normalize_min_max(image)
        self.assertEqual((low, high), (10, 20))
        self.assertEqual(scaled.tolist(), [[0, 255]])

    def test_stretch_with_clipping_collapses_flat_range_to_zeros(self):
        """Clipping that leaves no range returns zeros.

        Args: None.
        Returns: None.
        """
        image = np.full((8, 8), 5, dtype=np.uint8)
        scaled, low, high = stretch_contrast(image, 10)
        self.assertFalse(scaled.any())
        self.assertLessEqual(high, low)

    def test_stretch_with_clipping_scales_a_gradient(self):
        """Clipped stretching scales a gradient into 0..255.

        Args: None.
        Returns: None.
        """
        image = np.tile(np.arange(0, 200, dtype=np.uint8), (4, 1))
        scaled, low, high = stretch_contrast(image, 5)
        self.assertLess(low, high)
        self.assertEqual(int(scaled.max()), 255)


class ProcessorTests(unittest.TestCase):
    """ESPIProcessor baseline and averaging behavior."""

    def test_reference_mode_without_baseline_returns_zeros(self):
        """Reference differencing yields zeros before a reference is stored.

        Args: None.
        Returns: None.
        """
        frame = np.full((3, 3), 9, dtype=np.uint8)
        self.assertFalse(ESPIProcessor().difference(frame, "Reference").any())

    def test_reference_mode_subtracts_stored_baseline(self):
        """Reference differencing returns the absolute difference.

        Args: None.
        Returns: None.
        """
        proc = ESPIProcessor()
        proc.set_reference(np.full((2, 2), 10, dtype=np.uint8), 0.0)
        result = proc.difference(np.full((2, 2), 4, dtype=np.uint8), "Reference")
        self.assertTrue((result == 6).all())

    def test_previous_mode_first_frame_is_zero_then_differences(self):
        """Previous differencing needs two frames to produce output.

        Args: None.
        Returns: None.
        """
        proc = ESPIProcessor()
        first = np.full((2, 2), 10, dtype=np.uint8)
        self.assertFalse(proc.difference(first, "Previous").any())
        result = proc.difference(np.full((2, 2), 13, dtype=np.uint8), "Previous")
        self.assertTrue((result == 3).all())

    def test_rolling_average_with_count_one_passes_image_through(self):
        """A rolling count of one disables averaging and clears state.

        Args: None.
        Returns: None.
        """
        proc = ESPIProcessor()
        image = np.full((2, 2), 8, dtype=np.uint8)
        proc.rolling_average(image, 4)
        self.assertIs(proc.rolling_average(image, 1), image)
        self.assertIsNone(proc.rolling)

    def test_rolling_average_blends_toward_new_frames(self):
        """A rolling average moves partway toward a new frame.

        Args: None.
        Returns: None.
        """
        proc = ESPIProcessor()
        proc.rolling_average(np.zeros((2, 2), dtype=np.uint8), 2)
        result = proc.rolling_average(np.full((2, 2), 100, dtype=np.uint8), 2)
        self.assertTrue((result == 50).all())

    def test_denoise_off_passes_frame_through_and_clears_history(self):
        """Zero strength returns the input itself and drops the running history.

        Args: None.
        Returns: None.
        """
        proc = ESPIProcessor()
        frame = np.full((2, 2), 40, dtype=np.uint8)
        proc.denoise(frame, 0.5)
        self.assertIsNotNone(proc.denoised)
        self.assertIs(proc.denoise(frame, 0), frame)
        self.assertIsNone(proc.denoised)

    def test_denoise_blends_frames_and_restarts_on_new_size(self):
        """Each frame keeps the requested share of history; a resize restarts it.

        Args: None.
        Returns: None.
        """
        proc = ESPIProcessor()
        first = np.zeros((2, 2), dtype=np.uint8)
        self.assertIs(proc.denoise(first, 0.75), first)
        result = proc.denoise(np.full((2, 2), 200, dtype=np.uint8), 0.75)
        self.assertTrue((result == 50).all())
        bigger = np.full((3, 3), 9, dtype=np.uint8)
        self.assertIs(proc.denoise(bigger, 0.75), bigger)
        self.assertEqual(proc.denoised.shape, (3, 3))

    def test_denoise_caps_strength_so_the_feed_still_moves(self):
        """Strength above MAX_DENOISE_STRENGTH still lets new frames through.

        Args: None.
        Returns: None.
        """
        proc = ESPIProcessor()
        proc.denoise(np.zeros((2, 2), dtype=np.uint8), 1.0)
        result = proc.denoise(np.full((2, 2), 200, dtype=np.uint8), 1.0)
        expected = round(200 * (1 - MAX_DENOISE_STRENGTH))
        self.assertTrue((result == expected).all())

    def test_denoise_reduces_random_noise_on_a_static_scene(self):
        """Blending frames of a fixed speckle pattern lowers the noise around it.

        Args: None.
        Returns: None.
        """
        rng = np.random.default_rng(0)
        speckle = rng.integers(40, 200, (64, 64)).astype(np.float32)
        proc = ESPIProcessor()
        for _ in range(40):
            noisy = np.clip(speckle + rng.normal(0, 12, speckle.shape), 0, 255)
            result = proc.denoise(noisy.astype(np.uint8), 0.8)
        raw_error = np.abs(noisy - speckle).mean()
        filtered_error = np.abs(result.astype(np.float32) - speckle).mean()
        self.assertLess(filtered_error, raw_error / 2)

    def test_denoise_strength_is_off_at_zero_and_capped_at_the_top(self):
        """The slider's ends stay off and at the old strongest blend.

        Args: None.
        Returns: None.
        """
        self.assertEqual(denoise_strength(0), 0.0)
        self.assertEqual(denoise_strength(-5), 0.0)
        self.assertEqual(denoise_strength(MAX_DENOISE_PERCENT), MAX_DENOISE_STRENGTH)
        self.assertEqual(denoise_strength(100), MAX_DENOISE_STRENGTH)

    def test_denoise_strength_grows_evenly_and_the_low_end_is_useful(self):
        """Low slider positions blend several frames, and higher means more.

        A linear history weight at 10 was 0.10, about one frame. The mapped
        weight must be well above that, and the sequence must not go backwards.

        Args: None.
        Returns: None.
        """
        strengths = [denoise_strength(percent) for percent in range(0, 96)]
        self.assertEqual(strengths, sorted(strengths))
        self.assertGreater(denoise_strength(10), 0.5)
        self.assertLess(denoise_strength(94), MAX_DENOISE_STRENGTH)

    def test_low_slider_changes_raw_previous_and_reference(self):
        """A low setting moves all three views more than the old linear blend.

        Frames alternate black and white, so an unfiltered difference is 255
        and a blend pulls Raw toward gray. The reference is taken once the
        blend has settled, while noise reduction is on.

        Args: None.
        Returns: None.
        """
        frames = [np.full((8, 8), 0 if i % 2 == 0 else 255, dtype=np.uint8)
                  for i in range(40)]
        off = self._views(frames, 0)
        low = self._views(frames, 10)
        high = self._views(frames, MAX_DENOISE_PERCENT)
        self.assertIn(int(off["Raw"][-1][0, 0]), (0, 255))
        self.assertGreater(int(low["Raw"][-1][0, 0]), 40)
        self.assertLess(int(low["Raw"][-1][0, 0]), 220)
        self.assertLess(abs(int(high["Raw"][-1][0, 0]) - 127),
                        abs(int(low["Raw"][-1][0, 0]) - 127))
        self.assertEqual(int(off["Previous"][-1][0, 0]), 255)
        self.assertLess(int(low["Previous"][-1][0, 0]), 80)
        self.assertLess(int(high["Previous"][-1][0, 0]),
                        int(low["Previous"][-1][0, 0]))
        self.assertIn(int(off["Reference"][-1][0, 0]), (0, 255))
        self.assertLess(int(low["Reference"][-1][0, 0]), 200)
        self.assertLess(int(high["Reference"][-1][0, 0]),
                        int(low["Reference"][-1][0, 0]))

    def _views(self, frames, percent):
        """Run the engine's order: denoise, then each view, on one processor.

        Args: frames is a list of Mono8 arrays; percent is the slider value.
        Returns: A dict of mode name to the list of uint8 images.
        """
        proc = ESPIProcessor()
        strength = denoise_strength(percent)
        images = {mode: [] for mode in ("Raw", "Previous", "Reference")}
        for index, frame in enumerate(frames):
            frame = proc.denoise(frame, strength)
            if index == 30:
                proc.set_reference(frame, 0.0)
            images["Raw"].append(frame.copy())
            images["Previous"].append(proc.difference(frame, "Previous").copy())
            images["Reference"].append(proc.difference(frame, "Reference").copy())
        return images

    def test_block_average_returns_none_until_full_then_mean(self):
        """Block averaging withholds output until the count is reached.

        Args: None.
        Returns: None.
        """
        proc = ESPIProcessor()
        self.assertIsNone(proc.average(np.full((2, 2), 2, dtype=np.uint8), 2))
        result = proc.average(np.full((2, 2), 4, dtype=np.uint8), 2)
        self.assertTrue((result == 3).all())

    def test_block_average_with_count_one_passes_image_through(self):
        """A block count of one returns the image itself.

        Args: None.
        Returns: None.
        """
        image = np.full((2, 2), 2, dtype=np.uint8)
        self.assertIs(ESPIProcessor().average(image, 1), image)


class SingleFieldTests(unittest.TestCase):
    """Show one field of an interlaced frame at full height."""

    def test_single_field_doubles_even_rows(self):
        """Even rows are kept and each one repeated; the input is untouched.

        Args: None.
        Returns: None.
        """
        frame = np.arange(4, dtype=np.uint8).repeat(3).reshape(4, 3)
        result = single_field(frame)
        self.assertEqual(result[:, 0].tolist(), [0, 0, 2, 2])
        self.assertEqual(frame[:, 0].tolist(), [0, 1, 2, 3])
        self.assertEqual(result.dtype, np.uint8)

    def test_single_field_keeps_every_height(self):
        """NTSC, PAL and odd heights keep their shape.

        Args: None.
        Returns: None.
        """
        for height in (480, 576, 481):
            frame = np.zeros((height, 8), dtype=np.uint8)
            frame[1::2] = 255
            result = single_field(frame)
            self.assertEqual(result.shape, (height, 8))
            self.assertEqual(int(result.max()), 0)


class DemoTests(unittest.TestCase):
    """Demo frames and the watermark."""

    def test_demo_frames_have_requested_shape(self):
        """DemoSource produces uint8 frames of the configured size.

        Args: None.
        Returns: None.
        """
        frame = DemoSource(64, 48).next_frame(8000, 0, 0.0)
        self.assertEqual((frame.shape, frame.dtype), ((48, 64), np.uint8))

    def test_watermark_leaves_source_unchanged_for_gray_and_bgr(self):
        """Watermarking copies gray and BGR frames and marks the copy.

        Args: None.
        Returns: None.
        """
        for shape in ((100, 200), (100, 200, 3)):
            image = np.zeros(shape, dtype=np.uint8)
            marked = watermark_demo(image)
            self.assertFalse(image.any())
            self.assertTrue(marked.any())


class LabviewFlowTests(unittest.TestCase):
    """Fixed gain, the outlier filter, frames back and scaled averages."""

    def test_apply_gain_saturates_and_leaves_unit_gain_alone(self):
        """Gain multiplies, rounds and stops at 255; 1.0 returns the input.

        Args: None.
        Returns: None.
        """
        image = np.array([[0, 3, 40, 200]], dtype=np.uint8)
        self.assertIs(apply_gain(image, 1.0), image)
        self.assertEqual(apply_gain(image, 8).tolist(), [[0, 24, 255, 255]])
        self.assertEqual(apply_gain(image, 1.5).tolist(), [[0, 4, 60, 255]])

    def test_reject_outliers_replaces_only_spikes(self):
        """A lone spike becomes the neighbourhood mean; flat and smooth areas stay.

        Args: None.
        Returns: None.
        """
        flat = np.full((9, 9), 20, dtype=np.uint8)
        self.assertIs(reject_outliers(flat, 1), flat)
        self.assertIs(reject_outliers(flat, 0), flat)
        self.assertTrue(np.array_equal(reject_outliers(flat, 3), flat))
        spiked = flat.copy()
        spiked[4, 4] = 200
        result = reject_outliers(spiked, 3)
        self.assertEqual(int(result[4, 4]), 40)  # mean of eight 20s and one 200
        self.assertEqual(int(result[0, 0]), 20)
        self.assertEqual(int(spiked[4, 4]), 200)  # the input is not changed
        ramp = np.tile(np.arange(0, 90, 10, dtype=np.uint8), (9, 1))
        self.assertTrue(np.array_equal(reject_outliers(ramp, 4)[:, 2:-2],
                                       ramp[:, 2:-2]))  # even width made odd

    def test_previous_compares_with_the_frame_n_back(self):
        """Frames back picks the earlier frame and refills after a change.

        Args: None.
        Returns: None.
        """
        processor = ESPIProcessor()
        frames = [np.full((2, 2), value, dtype=np.uint8) for value in (0, 10, 30, 60)]
        outputs = [int(processor.difference(frame, "Previous", 2)[0, 0])
                   for frame in frames]
        self.assertEqual(outputs, [0, 0, 30, 50])
        self.assertEqual(int(processor.difference(frames[0], "Previous", 1)[0, 0]), 60)
        self.assertEqual(processor.history.maxlen, 1)
        processor.difference(frames[0], "Previous", 99)
        self.assertEqual(processor.history.maxlen, MAX_FRAMES_BACK)
        processor.difference(np.zeros((3, 3), dtype=np.uint8), "Previous", 2)
        self.assertEqual(len(processor.history), 1)  # a new size starts over
        processor.reset_previous()
        self.assertEqual(len(processor.history), 0)

    def test_scaled_block_average_rounds_and_saturates(self):
        """With a scale, the mean is multiplied, rounded and clipped to 255.

        Args: None.
        Returns: None.
        """
        processor = ESPIProcessor()
        one = np.full((2, 2), 1, dtype=np.uint8)
        two = np.full((2, 2), 2, dtype=np.uint8)
        self.assertIs(processor.average(one, 1, 3.0), one)
        self.assertIsNone(processor.average(one, 2, 3.0))
        self.assertEqual(int(processor.average(two, 2, 3.0)[0, 0]), 4)  # 1.5 x 3 = 4.5
        self.assertIsNone(processor.average(two, 2, 200.0))
        self.assertEqual(int(processor.average(two, 2, 200.0)[0, 0]), 255)

    def test_scaled_rolling_average(self):
        """A rolling average can be scaled; count 1 returns the input.

        Args: None.
        Returns: None.
        """
        processor = ESPIProcessor()
        image = np.full((2, 2), 10, dtype=np.uint8)
        self.assertIs(processor.rolling_average(image, 1, 2.0), image)
        self.assertEqual(int(processor.rolling_average(image, 4, 2.0)[0, 0]), 20)
        self.assertEqual(int(processor.rolling_average(image, 4)[0, 0]), 10)


class FringeEnhancementTests(unittest.TestCase):
    """Sensitivity floor, steady range, noise calibration, peak hold and tone."""

    def test_stretch_levels_and_apply_levels(self):
        """Levels come from min/max or a clipped histogram; apply_levels maps them.

        Args: None.
        Returns: None.
        """
        image = np.arange(100, dtype=np.uint8).reshape(10, 10)
        self.assertEqual(stretch_levels(image, 0), (0, 99))
        low, high = stretch_levels(image, 5)
        self.assertEqual((low, high), (5, 94))
        mapped = apply_levels(image, 10, 20)
        self.assertEqual((int(mapped[0, 5]), int(mapped[1, 5]), int(mapped[2, 5])),
                         (0, 128, 255))
        self.assertFalse(apply_levels(image, 20, 20).any())
        self.assertEqual(int(apply_levels(image, 0, 510)[9, 9]), 50)  # floor above 255

    def test_range_tracker_floor_and_slow_release(self):
        """The floor widens a quiet range; a shrinking range is followed slowly.

        Args: None.
        Returns: None.
        """
        tracker = RangeTracker()
        self.assertEqual(tracker.update(2, 5, 30, 0, 0.0), (2, 32))
        self.assertEqual(tracker.update(0, 200, 0, 1.0, 1.0), (0, 200))  # grows at once
        low, high = tracker.update(0, 100, 0, 1.0, 2.0)
        self.assertTrue(100 < high < 200)  # shrinks part of the way
        low, high = tracker.update(10, 100, 0, 1.0, 3.0)
        self.assertTrue(0 < low < 10)
        tracker.reset()
        self.assertIsNone(tracker.high)
        self.assertEqual(tracker.update(4, 8, 0, 1.0, 4.0), (4, 8))

    def test_noise_calibration_suggests_a_floor(self):
        """The floor is the median near-peak level times the margin.

        Args: None.
        Returns: None.
        """
        calibration = NoiseCalibration(now=0.0, duration_s=2.0)
        self.assertIsNone(calibration.result())
        self.assertFalse(calibration.done(1.0))
        for level in (4, 6, 5):
            calibration.add(np.full((10, 10), level, dtype=np.uint8))
        self.assertTrue(calibration.done(2.0))
        noise, floor = calibration.result()
        self.assertEqual(noise, 5.0)
        self.assertEqual(floor, int(5 * NOISE_MARGIN))

    def test_peak_hold_keeps_the_maximum_within_the_window(self):
        """A peak stays for the window, then drops out; 0 turns it off.

        Args: None.
        Returns: None.
        """
        hold = PeakHold()
        quiet = np.zeros((4, 4), dtype=np.uint8)
        spike = np.full((4, 4), 200, dtype=np.uint8)
        self.assertIs(hold.update(quiet, 0, 0.0), quiet)
        span = 3.0 / PEAK_SLICES
        self.assertEqual(int(hold.update(spike, 3.0, 0.0)[0, 0]), 200)
        self.assertEqual(int(hold.update(quiet, 3.0, span / 2)[0, 0]), 200)  # same slice
        self.assertEqual(int(hold.update(quiet, 3.0, span * 2)[0, 0]), 200)  # closed slice
        self.assertEqual(int(hold.update(quiet, 3.0, 4.0)[0, 0]), 0)  # aged out
        self.assertEqual(int(hold.update(quiet, 3.0, 4.0 + span)[0, 0]), 0)
        bigger = np.zeros((5, 5), dtype=np.uint8)
        self.assertEqual(hold.update(bigger, 3.0, 5.0).shape, (5, 5))  # new size restarts

    def test_local_normalize_centres_on_mid_grey(self):
        """A flat image is mid-grey; a step is pushed apart; the floor limits gain.

        Args: None.
        Returns: None.
        """
        flat = np.full((20, 20), 40, dtype=np.uint8)
        self.assertEqual(int(local_normalize(flat, 5)[10, 10]), 128)
        step = np.zeros((20, 20), dtype=np.uint8)
        step[:, 10:] = 20
        result = local_normalize(step, 7)
        self.assertLess(int(result[10, 9]), 128)
        self.assertGreater(int(result[10, 10]), 128)
        gentle = local_normalize(step, 7, floor=400)
        self.assertLess(int(gentle[10, 10]) - 128, int(result[10, 10]) - 128)

    def test_even_out_brightness_flattens_the_beam(self):
        """Dim lit areas are boosted (capped); unlit areas go black.

        Args: None.
        Returns: None.
        """
        frame = np.full((64, 64), 200, dtype=np.uint8)
        frame[:, :32] = 50
        difference = np.full((64, 64), 10, dtype=np.uint8)
        result = even_out_brightness(difference, frame, sigma=4)
        self.assertGreater(int(result[32, 2]), int(result[32, 60]))
        self.assertLessEqual(int(result.max()), int(10 * MAX_BRIGHTNESS_GAIN))
        dark = np.zeros((16, 16), dtype=np.uint8)
        self.assertFalse(even_out_brightness(np.full((16, 16), 9, dtype=np.uint8),
                                             dark).any())

    def test_display_lut_gain_contrast_and_brightness(self):
        """Off returns None; gain, S-curve and brightness shape the table.

        Args: None.
        Returns: None.
        """
        self.assertIsNone(display_lut())
        self.assertEqual(int(display_lut(gain=2.0)[100]), 200)
        self.assertEqual(int(display_lut(gain=2.0)[200]), 255)
        curve = display_lut(contrast=50)
        self.assertEqual((int(curve[0]), int(curve[255])), (0, 255))
        self.assertLess(int(curve[64]), 64)
        self.assertGreater(int(curve[192]), 192)
        self.assertEqual(int(display_lut(brightness=10)[0]), 10)


def deep(value, shape=(2, 2)):
    """Make a constant deep (float32) frame.

    Args: value is the grey level, fractions allowed; shape is the array shape.
    Returns: A float32 array.
    """
    return np.full(shape, value, np.float32)


class DeepFrameTests(unittest.TestCase):
    """Deep (float32) frames, as the iPhone's RAW mode sends, keep their fractions."""

    def test_to_uint8_rounds_and_clips_negatives_to_black(self):
        """A deep image becomes the 8-bit picture: rounded, black below 0, white above 255.

        Args: None.
        Returns: None.
        """
        image = np.array([[-3.0, 0.4, 127.6, 300.0]], np.float32)
        result = to_uint8(image)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.tolist(), [[0, 0, 128, 255]])  # -3 is black, not 3
        picture = np.array([[1, 2]], np.uint8)
        self.assertIs(to_uint8(picture), picture)

    def test_saturate_follows_the_frames_type(self):
        """A product is rounded to uint8 for 8-bit frames, capped in place for deep ones.

        Args: None.
        Returns: None.
        """
        product = np.array([[1.4, 1.6, 300.0]], np.float32)
        rounded = saturate(product.copy(), np.zeros(1, np.uint8))
        self.assertEqual(rounded.dtype, np.uint8)
        self.assertEqual(rounded.tolist(), [[1, 2, 255]])
        capped = saturate(product, product)
        self.assertIs(capped, product)
        self.assertEqual(capped.dtype, np.float32)
        self.assertEqual(capped.tolist(), [[np.float32(1.4), np.float32(1.6), FULL_SCALE]])

    def test_stretch_levels_of_a_deep_image_keep_fractions(self):
        """A difference under one grey level gets fractional levels, not 0 and 1.

        Args: None.
        Returns: None.
        """
        image = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
        self.assertEqual(stretch_levels(image, 0), (0.0, 1.0))
        low, high = stretch_levels(image, 5)
        self.assertIsInstance(low, float)
        # The uint8 rule on 0..99 gives (5, 94); here within two of 1024 bins.
        self.assertAlmostEqual(low, 5 / 99, delta=0.002)
        self.assertAlmostEqual(high, 94 / 99, delta=0.002)
        flat = deep(0.3, (3, 3))
        self.assertEqual(stretch_levels(flat, 5), (float(np.float32(0.3)),) * 2)

    def test_apply_levels_maps_fractional_levels(self):
        """Levels half a grey level apart still span black to white.

        Args: None.
        Returns: None.
        """
        image = np.array([[0.5, 1.0, 1.5, 2.0]], np.float32)
        result = apply_levels(image, 0.5, 1.5)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.tolist(), [[0, 128, 255, 255]])
        self.assertFalse(apply_levels(image, 1.0, 1.0).any())

    def test_range_tracker_fine_keeps_fractions(self):
        """For deep images the tracked levels and the floor stay fractional.

        Args: None.
        Returns: None.
        """
        tracker = RangeTracker()
        self.assertEqual(tracker.update(0.2, 0.7, 0, 0, 0.0, fine=True), (0.2, 0.7))
        low, high = tracker.update(0.2, 0.7, 2, 0, 1.0, fine=True)
        self.assertEqual(low, 0.2)
        self.assertAlmostEqual(high, 2.2)  # the floor widens the range
        self.assertEqual(tracker.update(0.2, 0.7, 0, 0, 2.0), (0, 1))  # 8-bit rounding

    def test_noise_calibration_allows_no_floor_for_quiet_deep_frames(self):
        """A RAW still scene under half a grey level of noise needs no floor.

        Args: None.
        Returns: None.
        """
        quiet = NoiseCalibration(now=0.0)
        for level in (0.2, 0.3, 0.4):
            quiet.add(deep(level, (10, 10)))
        self.assertTrue(quiet.deep)
        noise, floor = quiet.result()
        self.assertAlmostEqual(noise, 0.3, places=6)
        self.assertEqual(floor, 0)
        louder = NoiseCalibration(now=0.0)
        louder.add(deep(2.0, (10, 10)))
        self.assertEqual(louder.result(), (2.0, int(2 * NOISE_MARGIN)))
        eight_bit = NoiseCalibration(now=0.0)  # 8-bit frames keep the 1-level minimum
        eight_bit.add(np.zeros((10, 10), np.uint8))
        self.assertEqual(eight_bit.result(), (0.0, 1))

    def test_apply_gain_keeps_fractions_and_caps_at_full_scale(self):
        """Gain on a deep frame keeps fractions and stops at 255 without rounding.

        Args: None.
        Returns: None.
        """
        image = np.array([[0.25, 1.5, 100.0]], np.float32)
        result = apply_gain(image, 42.0)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.tolist(), [[10.5, 63.0, 255.0]])
        self.assertEqual(image.tolist(), [[0.25, 1.5, 100.0]])  # the input is kept
        self.assertIs(apply_gain(image, 1.0), image)

    def test_reference_capture_mean_of_deep_frames(self):
        """A deep reference is the exact float mean of its frames.

        Args: None.
        Returns: None.
        """
        capture = ReferenceCapture(2, 0.0, 0.0, True)
        self.assertIsNone(capture.add(deep(1.0)))
        mean = capture.add(deep(2.5))
        self.assertEqual(mean.dtype, np.float32)
        self.assertTrue((mean == 1.75).all())  # uint8 would round to 2

    def test_average_frames_of_deep_frames(self):
        """Frames averaged before the difference keep the fractional mean.

        Args: None.
        Returns: None.
        """
        processor = ESPIProcessor()
        self.assertIsNone(processor.average_frames(deep(1.0), 2))
        mean = processor.average_frames(deep(2.5), 2)
        self.assertEqual(mean.dtype, np.float32)
        self.assertTrue((mean == 1.75).all())
        self.assertIsNone(processor.frame_accum)  # the next block starts fresh

    def test_denoise_of_deep_frames_hands_out_a_copy(self):
        """Noise reduction keeps fractions and never hands out its running sum.

        Args: None.
        Returns: None.
        """
        processor = ESPIProcessor()
        first = deep(1.0)
        self.assertIs(processor.denoise(first, 0.5), first)
        blended = processor.denoise(deep(2.0), 0.5)
        self.assertEqual(blended.dtype, np.float32)
        self.assertTrue((blended == 1.5).all())
        self.assertIsNot(blended, processor.denoised)
        processor.denoise(deep(4.0), 0.5)
        self.assertTrue((blended == 1.5).all())  # a later frame leaves it alone

    def test_block_average_of_deep_frames(self):
        """The block mean keeps fractions; a scale multiplies it and caps at 255.

        Args: None.
        Returns: None.
        """
        processor = ESPIProcessor()
        self.assertIsNone(processor.average(deep(0.5), 2, 3.0))
        scaled = processor.average(deep(1.0), 2, 3.0)
        self.assertEqual(scaled.dtype, np.float32)
        self.assertTrue((scaled == 2.25).all())  # (0.5 + 1.0) x 3 / 2
        self.assertIsNone(processor.average(deep(100.0), 2, 400.0))
        self.assertTrue((processor.average(deep(100.0), 2, 400.0) == FULL_SCALE).all())
        self.assertIsNone(processor.average(deep(0.5), 2))
        plain = processor.average(deep(1.0), 2)
        self.assertEqual(plain.dtype, np.float32)
        self.assertTrue((plain == 0.75).all())

    def test_rolling_average_of_deep_frames(self):
        """The running average keeps fractions; a scale multiplies it and caps at 255.

        Args: None.
        Returns: None.
        """
        processor = ESPIProcessor()
        first = processor.rolling_average(deep(0.25), 4)
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue((first == 0.25).all())
        self.assertIsNot(first, processor.rolling)
        scaled = processor.rolling_average(deep(1.25), 4, 2.0)
        self.assertTrue((scaled == 1.0).all())  # (0.25 + (1.25 - 0.25) / 4) x 2
        bright = ESPIProcessor().rolling_average(deep(200.0), 4, 2.0)
        self.assertTrue((bright == FULL_SCALE).all())


if __name__ == "__main__":
    unittest.main()
