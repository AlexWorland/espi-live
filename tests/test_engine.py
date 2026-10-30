"""Camera-free checks for processing and the engine (demo source only).

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import os
import tempfile
import time
import unittest

import numpy as np

from espi_live import paths
from espi_live.engine import Engine
from espi_live.processing import (PALETTES, SMOOTHING_METHODS, ESPIProcessor,
                                  apply_palette, normalize_min_max,
                                  palette_rgb_table, smooth_speckle,
                                  stretch_contrast)


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


def wait_until(predicate, timeout=5.0):
    """Poll a condition until it is true or a timeout passes.

    Args: predicate is a zero-argument callable; timeout is in seconds.
    Returns: The final value of predicate().
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class ProcessingTests(unittest.TestCase):
    """Pure image-processing behavior."""

    def test_average_matches_previous_implementation(self):
        """The faster averaging must stay bit-identical to the old code.

        Args: None.
        Returns: None.
        """
        rng = np.random.default_rng(1)
        for count in (2, 3, 7, 60):
            frames = [rng.integers(0, 256, (64, 96), dtype=np.uint8)
                      for _ in range(count)]
            accum = frames[0].astype(np.float32)
            for frame in frames[1:]:
                accum += frame
            expected = np.clip(accum / count, 0, 255).astype(np.uint8)
            processor = ESPIProcessor()
            results = [processor.average(frame, count) for frame in frames]
            self.assertTrue(all(r is None for r in results[:-1]))
            np.testing.assert_array_equal(results[-1], expected)

    def test_gpu_table_matches_cpu_palette(self):
        """The GPU lookup table must reproduce the exported colors.

        Args: None.
        Returns: None.
        """
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)
        for palette in PALETTES:
            for gamma in (1.0, 0.6, 2.2):
                table = palette_rgb_table(palette, gamma)
                cpu = apply_palette(image, palette, gamma)
                if cpu.ndim == 2:
                    cpu_rgb = np.repeat(cpu[..., None], 3, axis=2)
                else:
                    cpu_rgb = cpu[..., ::-1]
                np.testing.assert_array_equal(table[image], cpu_rgb)

    def test_stretch_contrast(self):
        """0% clip matches min/max; a clip ignores a few hot pixels.

        Args: None.
        Returns: None.
        """
        rng = np.random.default_rng(2)
        image = rng.integers(40, 120, (100, 100), dtype=np.uint8)
        image[0, :5] = 255  # stray bright speckles
        exact = stretch_contrast(image, 0)
        np.testing.assert_array_equal(exact[0], normalize_min_max(image)[0])
        clipped, low, high = stretch_contrast(image, 1.0)
        self.assertLess(high, 255)
        self.assertGreaterEqual(low, 40)
        self.assertEqual(int(clipped.max()), 255)
        flat = stretch_contrast(np.full((8, 8), 7, np.uint8), 1.0)[0]
        self.assertEqual(int(flat.max()), 0)

    def test_rolling_average_and_smoothing(self):
        """The running average converges and every frame returns an image.

        Args: None.
        Returns: None.
        """
        processor = ESPIProcessor()
        results = [processor.rolling_average(np.full((4, 4), 100, np.uint8), 5)
                   for _ in range(3)]
        self.assertTrue(all(r is not None and int(r[0, 0]) == 100 for r in results))
        result = processor.rolling_average(np.full((4, 4), 200, np.uint8), 5)
        self.assertEqual(int(result[0, 0]), 120)  # 100 + (200 - 100) / 5
        noisy = np.random.default_rng(3).integers(0, 256, (64, 64), dtype=np.uint8)
        for method in SMOOTHING_METHODS:
            smoothed = smooth_speckle(noisy, method, 7)
            self.assertLess(float(smoothed.std()), float(noisy.std()))
        self.assertIs(smooth_speckle(noisy, "Gaussian", 0), noisy)


class EngineDemoTests(unittest.TestCase):
    """The worker thread with synthetic video and a temporary output folder."""

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

    def tearDown(self):
        """Stop the engine and restore the output folder.

        Args: None.
        Returns: None.
        """
        self.engine.stop()
        self.assertTrue(self.engine.wait(10))
        paths.OUTPUT_DIR = self.saved_output
        self.tmp.cleanup()

    def test_raw_frames_are_published_and_watermarked(self):
        """Demo frames arrive, stay Mono8, and carry the dummy banner.

        Args: None.
        Returns: None.
        """
        self.assertTrue(wait_until(lambda: self.engine.latest_packet() is not None))
        packet = self.engine.latest_packet()
        self.assertEqual(packet.image.dtype, np.uint8)
        self.assertEqual(packet.image.ndim, 2)
        self.assertTrue(packet.demo)
        self.assertEqual((packet.palette, packet.gamma), ("Gray", 1.0))
        self.assertEqual(int(packet.image[:20, :].min()), 0)  # banner background
        self.assertEqual(int(packet.image[:42, :].max()), 255)  # banner text

    def test_gamma_reaches_every_view_and_changes_midtones(self):
        """Raw, Previous and Reference publish the display gamma.

        Raw stays gray even when a color palette is selected. A gamma other
        than 1.0 changes rendered pixels that are not already black or white.

        Args: None.
        Returns: None.
        """
        self.assertTrue(wait_until(lambda: self.engine.latest_packet() is not None))
        for mode in ("Raw", "Previous", "Reference"):
            self.engine.update_settings(mode=mode, gamma=0.5, clahe=False,
                                        palette="Turbo", scale_range=True)
            if mode == "Reference":
                self.engine.request("reference")

            def ready(mode=mode):
                """True once this view carries gamma and a mid-tone pixel.

                Args: mode is the view this iteration is waiting for.
                Returns: True when the newest packet matches.
                """
                packet = self.engine.latest_packet()
                return (packet is not None and packet.mode == mode and
                        packet.gamma == 0.5 and
                        np.any((packet.image > 0) & (packet.image < 255)))

            self.assertTrue(wait_until(ready))
            packet = self.engine.latest_packet()
            self.assertEqual(packet.palette, "Gray" if mode == "Raw" else "Turbo")
            plain = apply_palette(packet.image, packet.palette, 1.0)
            self.assertFalse(np.array_equal(packet.render(), plain))

    def test_noise_reduction_filters_the_raw_feed(self):
        """Noise reduction runs on camera frames, so Raw view uses it too.

        Args: None.
        Returns: None.
        """
        self.engine.update_settings(noise_reduction=80)
        self.assertTrue(wait_until(lambda: (
            self.engine.latest_packet() is not None and
            self.engine.latest_packet().sequence > 3)))
        self.assertEqual(self.engine.latest_packet().mode, "Raw")

    def test_measurement_save_and_recording(self):
        """Averaged differences publish, a save writes PNG+notes, AVI records.

        Args: None.
        Returns: None.
        """
        self.engine.update_settings(mode="Previous", average=3,
                                    palette="Turbo", gamma=1.5)
        self.assertTrue(wait_until(lambda: (
            self.engine.latest_packet() is not None and
            self.engine.latest_packet().mode == "Previous")))
        packet = self.engine.latest_packet()
        self.assertEqual(packet.palette, "Turbo")
        self.assertEqual(packet.gamma, 1.0)  # CLAHE on: gamma applied on CPU
        self.assertEqual(packet.render().ndim, 3)

        self.engine.request("save", title="test run/1", notes="hello")
        frames = os.path.join(self.tmp.name, "frames")
        self.assertTrue(wait_until(lambda: os.path.isdir(frames) and
                                   len(os.listdir(frames)) == 2))
        names = sorted(os.listdir(frames))
        self.assertTrue(names[0].startswith("dummy_frame_"))
        self.assertTrue(names[0].endswith("_test_run_1.png"))
        with open(os.path.join(frames, names[1]), encoding="utf-8") as info:
            self.assertIn("Source: Demo", info.read())

        self.engine.update_settings(record=True)
        recordings = os.path.join(self.tmp.name, "recordings")
        self.assertTrue(wait_until(lambda: os.path.isdir(recordings) and
                                   os.listdir(recordings)))
        time.sleep(0.6)
        self.engine.update_settings(record=False)
        self.assertTrue(wait_until(lambda: any(
            "Saved AVI" in line for line in self.log.lines)))
        avi = os.path.join(recordings, os.listdir(recordings)[0])
        self.assertTrue(os.path.basename(avi).startswith("dummy_espi_previous_"))
        self.assertGreater(os.path.getsize(avi), 1000)

    def test_reference_mode_and_actions(self):
        """Reference capture and resets run on the worker without errors.

        Args: None.
        Returns: None.
        """
        self.engine.update_settings(mode="Reference", average=1, clahe=False,
                                    palette="Gray", gamma=2.0)
        self.engine.request("reference")
        self.engine.request("reset_average")
        self.engine.request("reset_previous")
        self.assertTrue(wait_until(lambda: any(
            "Captured reference frame." in line for line in self.log.lines)))
        self.assertTrue(wait_until(lambda: (
            self.engine.latest_packet() is not None and
            self.engine.latest_packet().mode == "Reference")))
        packet = self.engine.latest_packet()
        self.assertEqual((packet.palette, packet.gamma), ("Gray", 2.0))
        self.assertEqual(self.engine.status()["demo"], True)

    def test_fringe_tools_and_reference_refresh(self):
        """Rolling average, filters, clip, invert and auto-refresh run together.

        Args: None.
        Returns: None.
        """
        # The worker snapshots its starting settings when its loop begins; a
        # change made before that is the starting state and is never logged.
        self.assertTrue(wait_until(lambda: self.engine.latest_packet() is not None))
        self.engine.update_settings(
            mode="Reference", average=4, rolling=True, ref_refresh_s=0.2,
            smoothing=5, smoothing_method="Median", clip_percent=1.0,
            invert=True, clahe=False)
        self.assertTrue(wait_until(lambda: any(
            "Reference auto-refresh: every 0.2 s" in line for line in self.log.lines)))
        self.assertTrue(wait_until(lambda: (
            self.engine.latest_packet() is not None and
            self.engine.latest_packet().mode == "Reference")))
        first = self.engine.latest_packet().sequence
        self.assertTrue(wait_until(
            lambda: self.engine.latest_packet().sequence > first + 3))
        self.assertIn("refreshes every", self.engine.status()["hint"])


if __name__ == "__main__":
    unittest.main()
