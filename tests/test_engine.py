"""Camera-free checks for processing and the engine (demo source only).

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import os
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

from espi_live import diagnostics
from espi_live import engine as engine_module
from espi_live import paths
from espi_live.camera import CameraInfo
from espi_live.engine import (LABVIEW_FLOW, Engine, default_settings, is_analog,
                              own_settings)
from espi_live.processing import (SMOOTHING_METHODS, ESPIProcessor,
                                  NoiseCalibration,
                                  apply_display, display_rgb_table, normalize_min_max,
                                  raw_difference,
                                  smooth_speckle,
                                  stretch_contrast)
from espi_live.daq import Drive
from classic_pipeline import CLASSIC_PIPELINE, classic_settings
from list_log import ListLog


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

    def test_gpu_table_matches_cpu_display(self):
        """The GPU lookup table must reproduce the exported gray values.

        Args: None.
        Returns: None.
        """
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)
        for gamma in (1.0, 0.6, 2.2):
            table = display_rgb_table(gamma)
            cpu = apply_display(image, gamma)
            self.assertEqual(cpu.ndim, 2)
            np.testing.assert_array_equal(table[image],
                                          np.repeat(cpu[..., None], 3, axis=2))

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
        self.engine.update_settings(**CLASSIC_PIPELINE)
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
        self.assertEqual(packet.gamma, 1.0)
        self.assertEqual(int(packet.image[:20, :].min()), 0)  # banner background
        self.assertEqual(int(packet.image[:42, :].max()), 255)  # banner text

    def test_raw_tap_off_stores_nothing(self):
        """Without the tap the worker keeps no raw frame.

        Args: None.
        Returns: None.
        """
        self.assertTrue(wait_until(lambda: self.engine.latest_packet() is not None))
        self.assertIsNone(self.engine.latest_raw())

    def test_raw_tap_on_publishes_copy_before_processing(self):
        """The raw frame is a private copy, newer on every read, taken before processing.

        Args: None.
        Returns: None.
        """
        self.engine.set_raw_tap(True)
        self.assertTrue(wait_until(lambda: self.engine.latest_raw() is not None))
        first = self.engine.latest_raw()
        self.assertEqual(first.image.dtype, np.uint8)
        self.assertEqual(first.image.ndim, 2)
        self.assertIsNot(first.image, self.engine.latest_packet().image)
        before = first.image.copy()
        first.image[:] = 7  # a reader scribbling on it must not reach the engine
        self.assertTrue(wait_until(
            lambda: self.engine.latest_raw().sequence > first.sequence))
        self.assertFalse(np.array_equal(self.engine.latest_raw().image, first.image))
        self.assertEqual(before.shape, first.image.shape)

    def test_raw_tap_off_clears_stored_frame(self):
        """Turning the tap off drops the stored raw frame.

        Args: None.
        Returns: None.
        """
        self.engine.set_raw_tap(True)
        self.assertTrue(wait_until(lambda: self.engine.latest_raw() is not None))
        self.engine.set_raw_tap(False)
        self.assertIsNone(self.engine.latest_raw())
        time.sleep(0.2)
        self.assertIsNone(self.engine.latest_raw())

    def test_gamma_reaches_every_view_and_changes_midtones(self):
        """Raw, Previous and Reference publish the display gamma.

        A gamma other
        than 1.0 changes rendered pixels that are not already black or white.

        Args: None.
        Returns: None.
        """
        self.assertTrue(wait_until(lambda: self.engine.latest_packet() is not None))
        for mode in ("Raw", "Previous", "Reference"):
            self.engine.update_settings(mode=mode, gamma=0.5, scale_range=True)
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
            self.assertFalse(np.array_equal(packet.render(), packet.image))

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

    def test_fixed_gain_and_labview_order(self):
        """Fixed gain scales the difference; LabVIEW order scales the average.

        The source alternates 0 and 3, so every Previous difference is 3.
        With gain 8 it shows 24. With the gain before averaging, Average N 2
        and a multiplier of 2 it shows 72: as in the LabVIEW VI, three frames
        are summed and divided by two. The outlier filter leaves a flat
        difference alone.

        Args: None.
        Returns: None.
        """
        state = {"n": 0}

        def next_frame(exposure_us, gain_db, fps):
            """Return the next frame, alternating 0 and 3.

            Args: exposure_us, gain_db and fps are the demo source arguments.
            Returns: A Mono8 frame.
            """
            state["n"] += 1
            return np.full((80, 60), 3 * (state["n"] % 2), dtype=np.uint8)

        self.engine.demo_source.next_frame = next_frame
        self.engine.update_settings(
            fps=120, mode="Previous", scale_range=False, difference_gain=8.0,
            outlier_filter=3, average=1)

        def pixel_is(value):
            """Wait for a published Previous frame with this pixel value.

            Args: value is the expected pixel below the dummy-mode banner.
            Returns: True when it arrived.
            """
            return wait_until(lambda: (
                (packet := self.engine.latest_packet()) is not None and
                packet.mode == "Previous" and int(packet.image[60, 30]) == value))

        self.assertTrue(pixel_is(24))
        self.assertTrue(wait_until(lambda: "Scaled: 24–24" in self.engine.status()["range"]))
        self.engine.update_settings(gain_before_average=True, average=2,
                                    average_multiplier=2.0)
        self.assertTrue(pixel_is(72))
        self.engine.update_settings(rolling=True)
        self.assertTrue(pixel_is(48))  # rolling: the multiplier only
        self.engine.update_settings(scale_range=True)
        self.assertTrue(pixel_is(0))  # a constant difference stretches to zeros

    def test_noise_calibration_sets_the_sensitivity_floor(self):
        """Calibration needs a difference view; there it sets min_range.

        The source alternates 0 and 4, so every Previous difference is 4 and
        the floor becomes twice that.

        Args: None.
        Returns: None.
        """
        state = {"n": 0}

        def next_frame(exposure_us, gain_db, fps):
            """Return the next frame, alternating 0 and 4.

            Args: exposure_us, gain_db and fps are the demo source arguments.
            Returns: A Mono8 frame.
            """
            state["n"] += 1
            return np.full((80, 60), 4 * (state["n"] % 2), dtype=np.uint8)

        self.engine.demo_source.next_frame = next_frame
        self.engine.update_settings(fps=120, mode="Raw")
        self.engine.request("calibrate_noise")
        self.assertTrue(wait_until(lambda: self.log.has("Noise calibration needs")))
        self.engine.update_settings(mode="Previous")
        with mock.patch.object(engine_module, "NoiseCalibration",
                               lambda now: NoiseCalibration(now, duration_s=0.2)):
            self.engine.request("calibrate_noise")
            self.assertTrue(wait_until(lambda: self.engine.settings().min_range == 8))
        events = []
        while not self.engine.events.empty():
            events.append(self.engine.events.get())
        self.assertIn(("noise_calibrated", 8), events)
        self.assertTrue(self.log.has("sensitivity floor set to 8"))

    def test_fringe_enhancement_stages_publish_frames(self):
        """Threshold, beam correction, peak hold, local normalization and tone run.

        Args: None.
        Returns: None.
        """
        self.engine.update_settings(
            fps=120, mode="Previous", threshold=3, even_brightness=True,
            peak_hold_s=1.0, normalize_window=9, display_gain=2.0, contrast=30,
            brightness=5)
        self.assertTrue(wait_until(lambda: "Local normalization 9 px"
                                   in self.engine.status()["range"]))
        self.assertTrue(wait_until(lambda: self.engine.latest_packet() is not None))
        packet = self.engine.latest_packet()  # the status text comes just before it
        self.assertEqual(packet.mode, "Previous")
        self.engine.update_settings(normalize_window=0, scale_range=True, min_range=30,
                                    steady_range_s=0.5)
        self.assertTrue(wait_until(lambda: "Scaled:" in self.engine.status()["range"]))
        self.engine.update_settings(mode="Raw")  # a view change resets the trackers
        self.assertTrue(wait_until(lambda: "Raw Mono8" in self.engine.status()["range"]))

    def test_raw_difference_bypasses_every_processing_step(self):
        """Raw difference publishes |live - previous| of raw frames, nothing else.

        The source alternates a black frame and one whose pixels equal their
        column, so every raw difference equals the column number. Noise
        reduction, Fixed gain, stretch, gamma, flip and display tone
        are all set and would each change that; Raw difference ignores them.
        The comparison window shows the two raw frames. Turning it off brings
        the processed view back.

        Args: None.
        Returns: None.
        """
        state = {"n": 0}
        columns = np.tile(np.arange(60, dtype=np.uint8), (80, 1))

        def next_frame(exposure_us, gain_db, fps):
            """Return the next frame, alternating black and the column ramp.

            Args: exposure_us, gain_db and fps are the demo source arguments.
            Returns: A Mono8 frame.
            """
            state["n"] += 1
            return columns.copy() if state["n"] % 2 else np.zeros_like(columns)

        self.engine.demo_source.next_frame = next_frame
        self.engine.update_settings(
            fps=120, mode="Reference", auto_reference=True, noise_reduction=80,
            scale_range=False, difference_gain=42.0, gamma=0.5,
            flip_horizontal=True, display_gain=2.0, contrast=30, brightness=5,
            smoothing=7, average=4, raw_difference=True)
        self.engine.set_compare(True)

        def raw_packet():
            """Wait for a Raw difference packet.

            Args: None.
            Returns: True when one arrived.
            """
            return wait_until(lambda: (
                (packet := self.engine.latest_packet()) is not None and
                packet.mode == engine_module.RAW_DIFFERENCE))

        self.assertTrue(raw_packet())
        packet = self.engine.latest_packet()
        np.testing.assert_array_equal(packet.image[50:], columns[50:])
        self.assertEqual(packet.gamma, 1.0)  # the display gamma (0.5) is bypassed too
        self.assertTrue(wait_until(lambda: "Raw difference: |live - previous|"
                                   in self.engine.status()["range"]))
        self.assertTrue(wait_until(lambda: self.engine.status()["hint"].startswith(
            "Raw difference")))
        self.assertTrue(self.log.has("Raw difference on"))
        self.assertTrue(wait_until(lambda: (
            (compare := self.engine.latest_compare()) is not None and
            compare.baseline_label == "Previous raw frame")))
        compare = self.engine.latest_compare()
        self.assertEqual({int(compare.current.image[60, 30]),
                          int(compare.baseline.image[60, 30])}, {0, 30})
        self.assertEqual(compare.difference.mode, engine_module.RAW_DIFFERENCE)
        self.engine.request("calibrate_noise")  # nothing to sample here
        self.assertTrue(wait_until(lambda: self.log.has("not Raw difference")))

        self.engine.update_settings(raw_difference=False)
        self.assertTrue(wait_until(lambda: (
            (packet := self.engine.latest_packet()) is not None and
            packet.mode == "Reference")))
        self.assertTrue(self.log.has("Raw difference off"))

    def test_raw_difference_records_under_its_own_name(self):
        """A Raw difference recording gets its own file, one frame per image.

        Args: None.
        Returns: None.
        """
        self.engine.update_settings(mode="Previous", average=4, raw_difference=True,
                                    record=True)
        self.assertTrue(wait_until(lambda: self.log.has("espi_rawdiff_")))
        self.engine.update_settings(record=False)

    def test_raw_difference_waits_for_a_second_frame_of_the_same_size(self):
        """No previous frame, or one of another size, gives no difference.

        Args: None.
        Returns: None.
        """
        frame = np.full((4, 6), 9, np.uint8)
        self.assertIsNone(raw_difference(frame, None))
        self.assertIsNone(raw_difference(frame, np.zeros((6, 4), np.uint8)))
        np.testing.assert_array_equal(
            raw_difference(frame, np.full((4, 6), 12, np.uint8)), np.full((4, 6), 3))

    def test_default_settings_choose_fixed_gain_for_analog_cameras(self):
        """Every camera starts with the noise pipeline; analog ones in Reference, Fixed gain.

        The LabVIEW flow is no longer the analog default: it is a toggle, and
        LABVIEW_FLOW still turns every step of the pipeline off.

        Args: None.
        Returns: None.
        """
        analog = CameraInfo("video 1", "Hauppauge USB-Live2", kind="capture",
                            vid=0x2040, pid=0xC200)
        webcam = CameraInfo("video 2", "Some webcam", kind="capture", vid=1, pid=2)
        self.assertTrue(is_analog(analog))
        self.assertFalse(is_analog(webcam))
        self.assertFalse(is_analog(CameraInfo("123")))
        self.assertFalse(is_analog(None))
        started = default_settings(analog)
        self.assertEqual(started, own_settings(analog))
        self.assertEqual(started.mode, "Reference")
        self.assertFalse(started.scale_range)
        self.assertEqual(started.difference_gain, 42.0)
        for source in (analog, webcam, None):
            s = default_settings(source)
            self.assertEqual((s.average, s.average_frames_first, s.ref_frames,
                              s.smoothing, s.smoothing_method, s.auto_reference),
                             (4, True, 16, 11, "Oriented", True), source)
        for key, value in {"average": 1, "average_frames_first": False, "ref_frames": 1,
                           "smoothing": 0}.items():
            self.assertEqual(LABVIEW_FLOW[key], value, key)
        self.assertTrue(default_settings(webcam).scale_range)
        self.assertTrue(default_settings(None).scale_range)
        self.assertFalse(Engine(self.log, info=analog).settings().scale_range)
        self.assertEqual(default_settings(webcam).mode, "Raw")

    def test_auto_reference_captures_one_when_reference_has_none(self):
        """Reference view with auto_reference captures its own baseline.

        Args: None.
        Returns: None.
        """
        self.engine.update_settings(mode="Reference", auto_reference=True)
        self.assertTrue(wait_until(
            lambda: self.log.has("Captured reference frame (automatic).")))
        self.assertTrue(wait_until(lambda: "press F" not in self.engine.status()["hint"]))

    def test_noise_reduction_changes_raw_previous_and_reference(self):
        """The mapped slider changes the published pixels in every view.

        The source alternates black and white. With the slider off, Raw is
        one of those and the differences are 255. Turned up, Raw moves toward
        gray and both differences shrink. Stretch is off so it cannot repaint
        a quiet difference at full brightness.

        Args: None.
        Returns: None.
        """
        shape = (80, 60)
        state = {"n": 0}

        def next_frame(exposure_us, gain_db, fps):
            """Return the next black or white frame.

            Args: exposure_us, gain_db and fps are the demo source arguments.
            Returns: A Mono8 frame.
            """
            state["n"] += 1
            value = 0 if state["n"] % 2 == 0 else 255
            return np.full(shape, value, dtype=np.uint8)

        self.engine.demo_source.next_frame = next_frame
        self.engine.update_settings(
            fps=120, scale_range=False, difference_gain=1.0,
            smoothing=0, average=1,
            rolling=False, invert=False, mode="Raw",
            noise_reduction=0)

        def body():
            """Read one pixel below the dummy-mode banner.

            Args: None.
            Returns: The pixel value, or None before a frame exists.
            """
            packet = self.engine.latest_packet()
            if packet is None:
                return None
            return packet, int(packet.image[60, 30])

        def wait_pixel(mode, percent, predicate, frames=40):
            """Wait until a newer frame in this view meets the predicate.

            Args: mode and percent are applied first; predicate takes the
                pixel; frames is how many new frames to require.
            Returns: True when the condition held.
            """
            self.engine.update_settings(mode=mode, noise_reduction=percent)
            current = body()
            start = 0 if current is None else current[0].sequence
            return wait_until(lambda: (
                (got := body()) is not None and got[0].mode == mode and
                got[0].sequence >= start + frames and predicate(got[1])))

        self.assertTrue(wait_pixel("Raw", 0, lambda v: v in (0, 255)))
        self.assertTrue(wait_pixel("Raw", 10, lambda v: 40 < v < 220))
        self.assertTrue(wait_pixel(
            "Raw", 95, lambda v: abs(v - 127) < abs(153 - 127), frames=80))
        self.assertTrue(wait_pixel("Previous", 0, lambda v: v == 255))
        self.assertTrue(wait_pixel("Previous", 10, lambda v: v < 80))
        self.assertTrue(wait_pixel("Previous", 95, lambda v: v < 20, frames=80))
        self.assertTrue(wait_pixel("Reference", 0, lambda v: v in (0, 255)))
        self.engine.request("reference")
        self.assertTrue(wait_until(lambda: any(
            "Captured reference" in line for line in self.log.lines)))
        self.assertTrue(wait_pixel("Reference", 0, lambda v: v in (0, 255), frames=8))
        self.assertTrue(wait_pixel("Reference", 95, lambda v: 80 < v < 200, frames=80))

    def test_measurement_save_and_recording(self):
        """Averaged differences publish, a save writes PNG+notes, AVI records.

        Args: None.
        Returns: None.
        """
        self.engine.update_settings(mode="Previous", average=3, gamma=1.5)
        self.assertTrue(wait_until(lambda: (
            self.engine.latest_packet() is not None and
            self.engine.latest_packet().mode == "Previous")))
        packet = self.engine.latest_packet()
        self.assertEqual(packet.gamma, 1.5)
        self.assertEqual(packet.render().ndim, 2)  # always grayscale

        self.engine.request("save", title="test run/1", notes="hello")
        frames = os.path.join(self.tmp.name, "frames")
        self.assertTrue(wait_until(lambda: os.path.isdir(frames) and
                                   len(os.listdir(frames)) == 2))
        names = sorted(os.listdir(frames))
        self.assertTrue(names[0].startswith("dummy_frame_"))
        self.assertTrue(names[0].endswith("_test_run_1.png"))
        notes_path = os.path.join(frames, names[1])
        # The notes file exists (empty) a moment before its text is written.
        self.assertTrue(wait_until(lambda: os.path.getsize(notes_path) > 0))
        with open(notes_path, encoding="utf-8") as info:
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

    def test_save_pending_until_the_worker_writes_the_frame(self):
        """A queued save counts as pending until the worker finishes it.

        Args: None.
        Returns: None.
        """
        engine = Engine(self.log)
        self.assertFalse(engine.save_pending())
        engine.request("save", title="", notes="")
        self.assertTrue(engine.save_pending())
        engine._handle_actions(ESPIProcessor(), None)
        self.assertFalse(engine.save_pending())
        self.assertTrue(any("No current display frame" in line for line in self.log.lines))

    def test_reference_mode_and_actions(self):
        """Reference capture and resets run on the worker without errors.

        Args: None.
        Returns: None.
        """
        self.engine.update_settings(mode="Reference", average=1, gamma=2.0)
        self.engine.request("reference")
        self.engine.request("reset_average")
        self.engine.request("reset_previous")
        self.assertTrue(wait_until(lambda: any(
            "Captured reference frame at noise reduction 0." in line for line in self.log.lines)))
        self.assertTrue(wait_until(lambda: (
            self.engine.latest_packet() is not None and
            self.engine.latest_packet().mode == "Reference")))
        packet = self.engine.latest_packet()
        self.assertEqual(packet.gamma, 2.0)
        self.assertEqual(self.engine.status()["demo"], True)

    def test_flip_horizontal_mirrors_the_published_frame(self):
        """Flip horizontally mirrors columns and leaves rows in place.

        The source is a left-to-right ramp, so the first column is the
        darkest unflipped and the brightest flipped.

        Args: None.
        Returns: None.
        """
        ramp = np.tile(np.arange(0, 240, 4, dtype=np.uint8), (80, 1))
        self.engine.demo_source.next_frame = lambda exposure_us, gain_db, fps: ramp.copy()
        self.engine.update_settings(
            fps=120, mode="Raw", gamma=1.0, noise_reduction=0,
            flip_horizontal=False)

        def row(flipped):
            """Return the ramp row (below the banner) once the setting is live.

            Args: flipped is the expected state of the first column.
            Returns: The row, or None until a matching frame is published.
            """
            packet = self.engine.latest_packet()
            if packet is None:
                return None
            got = packet.image[60]
            return got if (got[0] > got[-1]) == flipped else None

        self.assertTrue(wait_until(lambda: row(False) is not None))
        self.engine.update_settings(flip_horizontal=True)
        self.assertTrue(wait_until(lambda: row(True) is not None))
        self.assertTrue(np.array_equal(row(True)[::-1], ramp[60]))

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
            invert=True)
        self.assertTrue(wait_until(lambda: any(
            "Reference auto-refresh: every 0.2 s" in line for line in self.log.lines)))
        self.assertTrue(wait_until(lambda: (
            self.engine.latest_packet() is not None and
            self.engine.latest_packet().mode == "Reference")))
        first = self.engine.latest_packet().sequence
        self.assertTrue(wait_until(
            lambda: self.engine.latest_packet().sequence > first + 3))
        self.assertIn("refreshes every", self.engine.status()["hint"])


class SimulatedObjectEngineTests(EngineDemoTests):
    """The demo engine feeds its simulated source the object and the DAQ drive."""

    def test_demo_object_reaches_the_source(self):
        """Choosing an object changes what the source simulates, and is logged.

        Args: None.
        Returns: None.
        """
        self.engine.update_settings(demo_object="Still")
        self.assertTrue(wait_until(lambda: self.engine.demo_source.object == "Still"))
        self.assertTrue(wait_until(lambda: self.log.has("Simulated object: Still")))

    def test_reference_piezo_reaches_the_source(self):
        """Turning the reference piezo on reaches the source and is logged.

        Args: None.
        Returns: None.
        """
        self.assertFalse(self.engine.settings().demo_piezo)
        self.engine.update_settings(demo_piezo=True)
        self.assertTrue(wait_until(lambda: self.engine.demo_source.piezo_on))
        self.assertTrue(wait_until(lambda: self.log.has("Reference piezo: triangle on")))
        self.engine.update_settings(demo_piezo=False)
        self.assertTrue(wait_until(lambda: self.log.has("Reference piezo: off")))

    def test_drive_action_reaches_a_demo_source(self):
        """The DAQ drive is handed to the source on the worker.

        Args: None.
        Returns: None.
        """
        self.engine.update_settings(demo_object="Follow DAQ")
        self.engine.request("drive", drive=Drive("sine", 1000.0, 0.1))
        self.assertTrue(wait_until(
            lambda: self.engine.demo_source.harmonic_path_nm()[0] == 1000.0))

    def test_simulated_frames_raise_no_sharpness_warnings(self):
        """Fringes coming and going on the simulated drum are not camera blur.

        Previous view of a deforming drum, then Still: the displayed
        difference drops to exactly black, which the sharpness monitors would
        report as a blur episode on a real camera.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(diagnostics, "SHARPNESS_INTERVAL_S", 0.0), \
                mock.patch.object(diagnostics, "SHARPNESS_MIN_SAMPLES", 3):
            self.engine.update_settings(mode="Previous", demo_object="Slow deformation")
            self.engine.demo_source.clock_s = 2.0
            time.sleep(0.6)
            self.engine.update_settings(demo_object="Still")
            self.assertTrue(wait_until(lambda: self.log.has("Simulated object: Still")))
            time.sleep(0.8)
        self.assertFalse(self.log.has("Sharpness drop"))

    def test_status_names_the_simulated_signal(self):
        """The dummy status line says the video is a simulated ESPI signal.

        Args: None.
        Returns: None.
        """
        self.assertIn("simulated ESPI signal", self.engine.status()["camera"])
        self.assertTrue(self.log.has("simulated ESPI signal (ideal sensor)"))


if __name__ == "__main__":
    unittest.main()
