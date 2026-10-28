"""Camera-free checks for the processing helpers' edge cases.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import unittest

import numpy as np

from espi_live.processing import (DemoSource, ESPIProcessor, normalize_min_max,
                                  stretch_contrast, watermark_demo)


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


if __name__ == "__main__":
    unittest.main()
