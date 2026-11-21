"""Noise-floor subtraction and frame averaging before the difference, and
the engine paths that use them, without a camera.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import tempfile
import unittest
from dataclasses import replace
from unittest import mock

import cv2
import numpy as np

from espi_live import engine as engine_module
from espi_live import paths
from espi_live.engine import (Engine, Settings, finish_difference, floor_key,
                              view_hint)
from espi_live.processing import (FLOOR_MIN_IMAGES, FLOOR_SD_MARGIN,
                                  ESPIProcessor, NoiseFloorMap, subtract_floor)
from list_log import ListLog
from test_engine import wait_until


class NoiseFloorMapTests(unittest.TestCase):
    """NoiseFloorMap and subtract_floor."""

    def test_map_is_mean_plus_margin_spread_once_time_and_count_pass(self):
        """Each pixel's floor is its mean plus FLOOR_SD_MARGIN deviations.

        Args: None.
        Returns: None.
        """
        floor = NoiseFloorMap(10.0, duration_s=1.0, min_images=4)
        self.assertIsNone(floor.result())
        for value in (2, 4, 2, 4):
            floor.add(np.full((2, 3), value, np.uint8))
        self.assertFalse(floor.done(10.5))   # enough images, too soon
        self.assertTrue(floor.done(11.0))
        result = floor.result()
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, 3.0 + FLOOR_SD_MARGIN * 1.0)
        short = NoiseFloorMap(0.0, duration_s=0.0, min_images=4)
        short.add(np.zeros((2, 2), np.uint8))
        self.assertFalse(short.done(5.0))     # time is up, too few images
        self.assertEqual(NoiseFloorMap(0.0).min_images, FLOOR_MIN_IMAGES)

    def test_new_size_starts_the_sums_over(self):
        """Images of two sizes are never mixed.

        Args: None.
        Returns: None.
        """
        floor = NoiseFloorMap(0.0)
        floor.add(np.full((2, 2), 9, np.uint8))
        floor.add(np.full((3, 3), 1, np.uint8))
        self.assertEqual(floor.count, 1)
        np.testing.assert_allclose(floor.result(), np.ones((3, 3)))

    def test_subtract_floor_rounds_and_stops_at_zero(self):
        """image - floor, rounded, never below 0, as a new uint8 image.

        Args: None.
        Returns: None.
        """
        image = np.array([[1, 5, 200]], np.uint8)
        floor = np.array([[3.0, 2.4, 0.6]], np.float32)
        out = subtract_floor(image, floor)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.tolist(), [[0, 3, 199]])
        self.assertEqual(image.tolist(), [[1, 5, 200]])

    def test_still_noise_goes_dark_and_a_change_stays(self):
        """On simulated noise the floor removes most grain but keeps a change.

        Args: None.
        Returns: None.
        """
        rng = np.random.default_rng(1)

        def averaged(extra):
            """One average of 8 absolute differences of Gaussian noise.

            Args: extra is the added change's standard deviation per pixel.
            Returns: The uint8 average.
            """
            total = np.zeros((60, 80), np.float32)
            for _ in range(8):
                noise = rng.normal(0, np.sqrt(2) * 1.5 + extra, (60, 80))
                total += np.floor(np.abs(noise))
            return (total / 8).astype(np.uint8)

        floor = NoiseFloorMap(0.0)
        for _ in range(16):
            floor.add(averaged(0.0))
        still = subtract_floor(averaged(0.0), floor.result())
        moving = subtract_floor(averaged(3.0), floor.result())
        self.assertLess(still.mean(), 0.3)
        self.assertGreater(moving.mean(), 1.0)


class AverageFramesTests(unittest.TestCase):
    """ESPIProcessor.average_frames."""

    def test_count_one_passes_through_and_resets(self):
        """With one frame there is no averaging and no state.

        Args: None.
        Returns: None.
        """
        processor = ESPIProcessor()
        frame = np.full((2, 2), 7, np.uint8)
        self.assertIsNone(processor.average_frames(frame, 3))
        self.assertIs(processor.average_frames(frame, 1), frame)
        self.assertEqual((processor.frame_count, processor.frame_accum), (0, None))

    def test_block_mean_then_a_fresh_start(self):
        """None until count frames, then the rounded mean, then start again.

        Args: None.
        Returns: None.
        """
        processor = ESPIProcessor()
        frames = [np.full((2, 2), v, np.uint8) for v in (10, 11, 13)]
        self.assertIsNone(processor.average_frames(frames[0], 3))
        self.assertIsNone(processor.average_frames(frames[1], 3))
        mean = processor.average_frames(frames[2], 3)
        self.assertEqual(mean.dtype, np.uint8)
        self.assertTrue((mean == 11).all())   # 34 / 3 = 11.33
        self.assertIsNone(processor.frame_accum)
        self.assertIsNone(processor.average_frames(frames[0], 3))

    def test_count_or_size_change_restarts(self):
        """A new count or frame size never mixes with the frames before.

        Args: None.
        Returns: None.
        """
        processor = ESPIProcessor()
        processor.average_frames(np.full((2, 2), 200, np.uint8), 3)
        processor.average_frames(np.full((2, 2), 200, np.uint8), 3)
        self.assertIsNone(processor.average_frames(np.zeros((2, 2), np.uint8), 2))
        self.assertTrue((processor.average_frames(np.zeros((2, 2), np.uint8), 2) == 0).all())
        processor.average_frames(np.full((2, 2), 200, np.uint8), 2)
        self.assertIsNone(processor.average_frames(np.zeros((3, 3), np.uint8), 2))
        self.assertEqual(processor.frame_count, 1)
        processor.reset_frame_average()
        self.assertEqual((processor.frame_count, processor.frame_target), (0, None))

    def test_difference_of_averages_is_quieter(self):
        """Averaging 8 frames first cuts the difference noise by about sqrt(8).

        Args: None.
        Returns: None.
        """
        rng = np.random.default_rng(2)

        def frame():
            """A still scene at 128 with Gaussian noise of 4 levels.

            Args: None.
            Returns: A uint8 frame.
            """
            return np.clip(np.rint(rng.normal(128, 4, (100, 100))), 0, 255).astype(np.uint8)

        single = cv2.absdiff(frame(), frame()).mean()
        processor = ESPIProcessor()
        means = []
        for _ in range(16):
            out = processor.average_frames(frame(), 8)
            if out is not None:
                means.append(out)
        self.assertEqual(len(means), 2)
        averaged = cv2.absdiff(means[0], means[1]).mean()
        self.assertLess(averaged, single / 2.2)


class FloorHelpersTests(unittest.TestCase):
    """floor_key, finish_difference with a floor, and view_hint."""

    def test_floor_key_tracks_what_shapes_the_difference(self):
        """The key changes with the baseline and each step before the floor.

        Args: None.
        Returns: None.
        """
        processor = ESPIProcessor()
        s = Settings(mode="Previous", scale_range=True)
        key = floor_key(s, processor, (4, 4))
        self.assertEqual(key, floor_key(replace(s, difference_gain=3.0), processor, (4, 4)))
        self.assertEqual(key, floor_key(replace(s, palette="Jet"), processor, (4, 4)))
        for changed in (replace(s, frames_back=2), replace(s, average=4),
                        replace(s, average_frames_first=True), replace(s, smoothing=3),
                        replace(s, exposure_us=1.0), replace(s, mode="Reference")):
            self.assertNotEqual(key, floor_key(changed, processor, (4, 4)))
        self.assertNotEqual(key, floor_key(s, processor, (5, 4)))
        labview = replace(s, scale_range=False, gain_before_average=True)
        self.assertNotEqual(floor_key(labview, processor, (4, 4)),
                            floor_key(replace(labview, difference_gain=3.0),
                                      processor, (4, 4)))
        reference = replace(s, mode="Reference")
        before = floor_key(reference, processor, (4, 4))
        processor.set_reference(np.zeros((4, 4), np.uint8), 5.0)
        self.assertNotEqual(before, floor_key(reference, processor, (4, 4)))

    def test_finish_difference_samples_then_subtracts(self):
        """The measurement sees the image before the floor comes off.

        Args: None.
        Returns: None.
        """
        s = Settings(mode="Previous", scale_range=False, difference_gain=10.0,
                     clahe=False)
        seen = []
        image = np.full((2, 2), 5, np.uint8)
        out, _, _, levels = finish_difference(
            image, s, None, floor=np.full((2, 2), 3.0, np.float32),
            floor_sample=lambda img, now: seen.append((img.copy(), now)), now=4.0)
        self.assertEqual(seen[0][0].tolist(), image.tolist())
        self.assertEqual(seen[0][1], 4.0)
        self.assertTrue((out == 20).all())       # (5 - 3) x 10
        self.assertEqual(levels[:2], (2, 2))

    def test_view_hint_shows_the_floor_note_after_a_capture(self):
        """A reference capture outranks the floor note, which outranks the rest.

        Args: None.
        Returns: None.
        """
        processor = ESPIProcessor()
        s = Settings(mode="Previous")
        self.assertEqual(view_hint(s, processor, "floor note"), "floor note")
        processor.start_reference(4, 0.0, 0.0)
        self.assertIn("Capturing reference", view_hint(s, processor, "floor note"))


class EngineFloorStateTests(unittest.TestCase):
    """The engine's floor helpers, called directly on an engine not started."""

    def setUp(self):
        """Build a demo engine without starting its worker.

        Args: None.
        Returns: None.
        """
        self.log = ListLog()
        self.engine = Engine(self.log)
        self.processor = ESPIProcessor()
        self.s = Settings(mode="Previous", floor_subtract=True)

    def test_note_for_raw_measuring_missing_stale_and_ready(self):
        """Raw: none; measuring: a count; no or stale map: ask; matching: none.

        Args: None.
        Returns: None.
        """
        engine, s, shape = self.engine, self.s, (2, 2)
        self.assertIsNone(engine._floor_note(replace(s, mode="Raw"), self.processor, shape))
        self.assertIn("press Measure noise floor",
                      engine._floor_note(s, self.processor, shape))
        engine._floor = (floor_key(s, self.processor, shape), np.zeros(shape, np.float32))
        self.assertIsNone(engine._floor_note(s, self.processor, shape))
        self.assertIn("press Measure noise floor",
                      engine._floor_note(replace(s, average=3), self.processor, shape))
        self.assertIsNone(engine._floor_note(replace(s, floor_subtract=False),
                                             self.processor, (9, 9)))
        engine._floor_measure = (None, NoiseFloorMap(0.0))
        note = engine._floor_note(s, self.processor, shape)
        self.assertEqual(note, f"Measuring noise floor: 0/{FLOOR_MIN_IMAGES} images; "
                               "keep the object still")
        self.assertLessEqual(len(note), len(
            "ESPI difference view; min/max range is before gamma and palette"))

    def test_floor_use_matches_the_key_and_feeds_a_measurement(self):
        """The map is used only when on and matching; samples reach the measurement.

        Args: None.
        Returns: None.
        """
        engine, s, shape = self.engine, self.s, (2, 2)
        self.assertEqual(engine._floor_use(s, self.processor, shape), (None, None))
        stored = np.ones(shape, np.float32)
        engine._floor = (floor_key(s, self.processor, shape), stored)
        floor, sample = engine._floor_use(s, self.processor, shape)
        self.assertIs(floor, stored)
        self.assertIsNone(sample)
        self.assertEqual(engine._floor_use(replace(s, floor_subtract=False),
                                           self.processor, shape)[0], None)
        running = NoiseFloorMap(0.0, duration_s=0.0, min_images=2)
        engine._floor_measure = (None, running)
        _, sample = engine._floor_use(s, self.processor, shape)
        sample(np.full(shape, 4, np.uint8), 1.0)
        self.assertEqual(running.count, 1)
        self.assertEqual(engine._floor_measure[0], floor_key(s, self.processor, shape))
        sample(np.full(shape, 4, np.uint8), 1.0)
        self.assertIsNone(engine._floor_measure)
        np.testing.assert_allclose(engine._floor[1], np.full(shape, 4.0))
        self.assertTrue(any("Noise floor measured from 2 images: median 4.0, "
                            "highest 4.0 grey levels." in line for line in self.log.lines))

    def test_settings_change_cancels_a_measurement(self):
        """Images made two ways are never mixed into one map.

        Args: None.
        Returns: None.
        """
        engine, shape = self.engine, (2, 2)
        engine._floor_measure = (None, NoiseFloorMap(0.0))
        engine._sample_floor(np.ones(shape, np.uint8), ("a",), 0.0)
        engine._sample_floor(np.ones(shape, np.uint8), ("b",), 0.1)
        self.assertIsNone(engine._floor_measure)
        self.assertIsNone(engine._floor)
        self.assertTrue(any("Noise floor measurement cancelled" in line
                            for line in self.log.lines))


class EngineNoiseFloorTests(unittest.TestCase):
    """The worker with frame averaging and the noise floor, on synthetic video."""

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

    def test_frames_first_publishes_averages_and_logs_the_switch(self):
        """Average N with frames first publishes one image per N frames.

        Args: None.
        Returns: None.
        """
        self.engine.update_settings(mode="Previous", average=2, rolling=True,
                                    average_frames_first=True, record=False)
        self.engine.set_compare(True)
        self.assertTrue(wait_until(lambda: "Averaging: camera frames, then the "
                                           "difference" in self.log.lines))
        packet = self.next_packets(2)
        self.assertEqual(packet.mode, "Previous")
        self.assertTrue(wait_until(lambda: self.engine.latest_compare() is not None))
        self.engine.update_settings(average_frames_first=False)
        self.assertTrue(wait_until(lambda: "Averaging: differences" in self.log.lines))
        self.next_packets()

    def test_measure_in_raw_warns_then_measures_and_subtracts(self):
        """Raw cancels with a warning; Previous measures and the floor is used.

        Args: None.
        Returns: None.
        """
        short = lambda now: NoiseFloorMap(now, duration_s=0.1, min_images=3)  # noqa: E731
        used = []
        real = engine_module.subtract_floor

        def spy(image, floor):
            """Record that the floor was taken off, then do it.

            Args: image and floor are subtract_floor's arguments.
            Returns: subtract_floor's result.
            """
            used.append(floor.shape)
            return real(image, floor)

        with mock.patch.object(engine_module, "NoiseFloorMap", short), \
                mock.patch.object(engine_module, "subtract_floor", spy):
            self.engine.update_settings(mode="Raw")
            self.next_packets()
            self.engine.request("measure_floor")
            self.assertTrue(wait_until(lambda: any(
                "Measuring the noise floor needs the Previous" in line
                for line in self.log.lines)))
            self.engine.update_settings(mode="Previous", floor_subtract=True)
            self.next_packets()
            self.engine.request("measure_floor")
            self.assertTrue(wait_until(lambda: any(
                line.startswith("Noise floor measured from") for line in self.log.lines)))
            self.assertTrue(wait_until(lambda: bool(used)))
        self.assertIn("Measuring the noise floor: keep the object still for at "
                      "least 2 s.", self.log.lines)


if __name__ == "__main__":
    unittest.main()
