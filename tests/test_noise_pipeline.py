"""Camera-free checks for the default noise pipeline and the engine's Phase view.

Run from the project root:  PYTHONPATH=src:tests python -m unittest tests.test_noise_pipeline
"""

import math
import queue
import tempfile
import time
import unittest

import cv2
import numpy as np

from espi_live import daq, paths, phase
from espi_live import engine as engine_module
from espi_live.engine import (Engine, Settings, averaging_text, phase_frames_per_image,
                              phase_range_text, view_hint)
from espi_live.processing import ESPIProcessor, oriented_smooth, smooth_speckle
from list_log import ListLog
from test_phase import PIEZO, FakeController


def wait_until(predicate, timeout=10.0):
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


def wait_event(engine, kinds, timeout=20.0):
    """Wait for one of the given engine events.

    Args: engine is an Engine; kinds is a tuple of event names; timeout is
        in seconds.
    Returns: (kind, payload), or None when none arrived in time.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            kind, payload = engine.events.get(timeout=0.1)
        except queue.Empty:
            continue
        if kind in kinds:
            return kind, payload
    return None


def simulated_fringes(width, height, period):
    """Simulate subtraction ESPI: |I2 - I1| of speckle with diagonal fringes.

    Args: width and height are the size in pixels; period is the fringe
        spacing in pixels.
    Returns: (difference, true fringe envelope |sin(dphi / 2)|), both float32.
    """
    rng = np.random.default_rng(1)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    dphi = 2 * math.pi * (x * math.cos(0.6) + y * math.sin(0.6)) / period
    field = rng.normal(size=(height, width)) + 1j * rng.normal(size=(height, width))
    grain = cv2.getGaussianKernel(9, 1.5)
    real = cv2.sepFilter2D(field.real.astype(np.float32), -1, grain, grain)
    imaginary = cv2.sepFilter2D(field.imag.astype(np.float32), -1, grain, grain)
    intensity = real * real + imaginary * imaginary
    obj = intensity / intensity.mean() * 30
    speckle = np.arctan2(imaginary, real)
    first = obj + 30 + 2 * np.sqrt(obj * 30) * np.cos(speckle)
    second = obj + 30 + 2 * np.sqrt(obj * 30) * np.cos(speckle + dphi)
    difference = np.abs(second - first).astype(np.float32)
    return difference, np.abs(np.sin(dphi / 2)).astype(np.float32)


class OrientedFilterTests(unittest.TestCase):
    """The Oriented speckle filter."""

    def correlation(self, image, truth):
        """Correlate a filtered image with the true fringes.

        Args: image and truth are equally shaped arrays.
        Returns: The Pearson correlation.
        """
        return float(np.corrcoef(image.ravel(), truth.ravel())[0, 1])

    def test_dense_fringes_survive_better_than_with_a_gaussian(self):
        """At the same width, Oriented keeps 20 px fringes a Gaussian blurs.

        Args: None.
        Returns: None.
        """
        difference, truth = simulated_fringes(320, 240, 20)
        oriented = oriented_smooth(difference, 13)
        gaussian = cv2.GaussianBlur(difference, (13, 13), 0)
        self.assertEqual(oriented.dtype, np.float32)  # deep in, deep out
        self.assertGreater(self.correlation(oriented, truth),
                           self.correlation(gaussian, truth) + 0.03)

    def test_uint8_and_small_images(self):
        """uint8 stays uint8; an image too small to orient gets a Gaussian.

        Args: None.
        Returns: None.
        """
        difference, _ = simulated_fringes(160, 120, 20)
        image = np.clip(difference, 0, 255).astype(np.uint8)
        smoothed = smooth_speckle(image, "Oriented", 10)  # made odd
        self.assertEqual((smoothed.dtype, smoothed.shape), (np.uint8, image.shape))
        tiny = (np.arange(80, dtype=np.uint8).reshape(8, 10) * 3)
        np.testing.assert_array_equal(oriented_smooth(tiny, 5),
                                      cv2.convertScaleAbs(cv2.GaussianBlur(
                                          tiny.astype(np.float32), (5, 5), 0)))
        tiny_deep = tiny.astype(np.float32)
        self.assertEqual(oriented_smooth(tiny_deep, 5).dtype, np.float32)
        self.assertIs(smooth_speckle(image, "Oriented", 1), image)


class WordingTests(unittest.TestCase):
    """Status and comparison wording for the pipeline and the Phase view."""

    def test_averaging_text(self):
        """Frames first, rolling, block and off are each named.

        Args: None.
        Returns: None.
        """
        self.assertEqual(averaging_text(Settings()),
                         "4 camera frames averaged before the difference")
        self.assertEqual(averaging_text(Settings(average_frames_first=False, rolling=True)),
                         "average 4 rolling")
        self.assertEqual(averaging_text(Settings(average_frames_first=False)), "average 4")
        self.assertEqual(averaging_text(Settings(average=1)), "no averaging")

    def test_phase_counts_and_texts(self):
        """Frames per image, the range line and the hint.

        Args: None.
        Returns: None.
        """
        self.assertEqual(phase_frames_per_image(Settings()), 5 * (4 + 2) + 1)
        self.assertEqual(phase_frames_per_image(Settings(average_frames_first=False,
                                                         phase_steps=4)), 4 * 3 + 1)
        self.assertIn("filter 9 px", phase_range_text(Settings()))
        self.assertIn("no filter", phase_range_text(Settings(phase_filter=0)))
        processor = ESPIProcessor()
        s = Settings(mode="Phase")
        self.assertEqual(view_hint(s, processor, phase_note="note"), "note")
        self.assertIn("wrapped phase", view_hint(s, processor))


class DirectPhaseTests(unittest.TestCase):
    """Engine Phase-view helpers, called on the test's thread."""

    def setUp(self):
        """Make an idle demo engine in Phase view.

        Args: None.
        Returns: None.
        """
        self.log = ListLog()
        self.engine = Engine(self.log)
        self.engine.use_demo()
        self.s = Settings(mode="Phase", average_frames_first=False, phase_settle_frames=0)

    def frames(self, count, now=0.0):
        """Run frames through _phase_frame.

        Args: count is how many demo frames to use; now is the time.
        Returns: The list of results.
        """
        return [self.engine._phase_frame(self.engine._demo_frame(self.s, now), self.s, now)
                for _ in range(count)]

    def test_without_a_piezo_the_camera_frame_is_shown(self):
        """No stepper: the frame passes through and the hint says what to do.

        Args: None.
        Returns: None.
        """
        frame = self.engine._demo_frame(self.s, 0.0)
        self.assertIs(self.engine._phase_frame(frame, self.s, 0.0), frame)
        self.assertIn("needs the piezo", self.engine._phase_note(self.s))
        self.assertIsNone(self.engine._phase_note(Settings(mode="Reference")))
        self.assertIsNone(self.engine._phase_note(Settings(mode="Phase", raw_difference=True)))
        other = self.engine._demo_frame(Settings(mode="Raw"), 0.0)
        self.assertEqual(other.shape, (640, 960))  # the moving test pattern

    def test_steps_reference_and_hints(self):
        """A stepper moves through the steps; the first set is the reference.

        Args: None.
        Returns: None.
        """
        controller = FakeController()
        self.engine.set_phase_stepper(phase.PiezoStepper(controller, PIEZO, True))
        self.assertIsNotNone(self.engine.phase_stepper)
        # One frame waits for the move to step 0, one settles after it, then 5 steps.
        results = self.frames(7)
        images = [image for image in results if image is not None]
        self.assertEqual(len(images), 1)
        self.assertEqual([round(volts, 3) for _, volts, _ in controller.moves],
                         [0.1, 0.15, 0.2, 0.25, 0.3, 0.1])
        self.assertTrue(self.log.has("Phase reference captured (5-step)."))
        self.assertIn("step measured", self.engine._phase_note(self.s))
        self.assertIn("Phase change", self.engine._phase_note(self.s))
        four = Settings(mode="Phase", average_frames_first=False, phase_settle_frames=0,
                        phase_steps=4)
        self.engine._phase.angle = None
        self.assertIn("F takes a new reference", self.engine._phase_note(four))
        self.engine._phase.clear_reference()
        self.assertIn("measuring the reference", self.engine._phase_note(self.s))

    def test_problems_are_logged_once(self):
        """A stepper that cannot run shows the frame and warns once.

        Args: None.
        Returns: None.
        """
        bad = daq.replace(PIEZO, piezo_output=daq.OFF)
        self.engine.set_phase_stepper(phase.PiezoStepper(FakeController(), bad))
        frame = self.engine._demo_frame(self.s, 0.0)
        self.assertIs(self.engine._phase_frame(frame, self.s, 0.0), frame)
        self.assertIs(self.engine._phase_frame(frame, self.s, 0.0), frame)
        warnings = [m for level, m, _ in self.log.records if level == "WARNING"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("need attention", self.engine._phase_note(self.s))

    def test_a_failed_move_stops_the_view(self):
        """PiezoMoveError drops the stepper and sends phase_stopped.

        Args: None.
        Returns: None.
        """
        stepper = phase.PiezoStepper(FakeController(phase.PiezoMoveError("gone")), PIEZO)
        self.engine.set_phase_stepper(stepper)
        self.engine._phase.start_calibration()
        self.frames(1)
        self.assertIsNone(self.engine.phase_stepper)
        self.assertIsNone(self.engine._phase.calibration)
        self.assertEqual(self.engine.events.get_nowait(), ("phase_stopped", "gone"))
        # A stepper that is replaced meanwhile is not dropped.
        replacement = phase.PiezoStepper(FakeController(), PIEZO)
        self.engine.set_phase_stepper(replacement)
        self.engine._phase_failed(stepper, phase.PiezoMoveError("late"))
        self.assertIs(self.engine.phase_stepper, replacement)

    def test_removing_the_stepper_ends_a_calibration(self):
        """A calibration cannot outlive its piezo.

        Args: None.
        Returns: None.
        """
        self.engine.set_phase_stepper(phase.PiezoStepper(FakeController(), PIEZO))
        self.frames(1)
        self.engine._phase.start_calibration()
        self.engine.set_phase_stepper(None)
        self.frames(1)
        self.assertIsNone(self.engine._phase.calibration)

    def test_calibration_results(self):
        """A good estimate sends the new voltage; a wild one says so.

        Args: None.
        Returns: None.
        """
        stepper = phase.PiezoStepper(FakeController(), PIEZO)
        self.engine.set_phase_stepper(stepper)
        self.engine._phase.calibration = [math.radians(72)] * 5
        self.engine._finish_phase_calibration(stepper)
        kind, (degrees, volts, index) = self.engine.events.get_nowait()
        self.assertEqual(kind, "phase_calibrated")
        self.assertAlmostEqual(degrees, 72)
        self.assertAlmostEqual(volts, 0.0625)
        self.engine._phase.calibration = [math.radians(10)] * 5
        self.engine._finish_phase_calibration(stepper)
        kind, (degrees, index) = self.engine.events.get_nowait()
        self.assertEqual(kind, "phase_calibration_failed")
        self.assertAlmostEqual(degrees, 10)
        self.engine._phase.calibration = [math.radians(80)] * 2
        self.assertIn("set 3 of 5", self.engine._phase_note(self.s))
        self.engine._finish_phase_calibration(stepper)  # not enough sets: nothing yet
        self.assertTrue(self.engine.events.empty())

    def test_calibration_names_the_second_driver(self):
        """A Thor 2 calibration's events carry its driver index.

        Args: None.
        Returns: None.
        """
        settings = daq.DaqSettings(piezo2_output="ao0", piezo2_voltage_v=0.1,
                                   piezo2_step_v=0.05, piezo2_limit_v=1.0)
        stepper = phase.PiezoStepper(FakeController(), settings, index=2)
        self.engine.set_phase_stepper(stepper)
        self.engine._phase.calibration = [math.radians(72)] * 5
        self.engine._finish_phase_calibration(stepper)
        kind, (degrees, volts, index) = self.engine.events.get_nowait()
        self.assertEqual((kind, index), ("phase_calibrated", 2))
        self.assertAlmostEqual(volts, 0.0625)
        self.engine._phase.calibration = [math.radians(10)] * 5
        self.engine._finish_phase_calibration(stepper)
        kind, (degrees, index) = self.engine.events.get_nowait()
        self.assertEqual((kind, index), ("phase_calibration_failed", 2))

    def test_actions(self):
        """F takes a phase reference; Calibrate needs a piezo.

        Args: None.
        Returns: None.
        """
        self.engine.update_settings(mode="Phase")
        self.engine._phase.reference = np.ones((2, 2), np.complex64)
        self.engine.request("reference")
        self.engine.request("calibrate_phase")
        self.engine._handle_actions(ESPIProcessor(), None)
        self.assertIsNone(self.engine._phase.reference)
        self.assertTrue(self.log.has("next complete set of phase steps"))
        self.assertTrue(any("needs the piezo" in m for _, m, _ in self.log.records))
        self.engine.set_phase_stepper(phase.PiezoStepper(FakeController(), PIEZO))
        self.engine.request("calibrate_phase")
        self.engine._handle_actions(ESPIProcessor(), None)
        self.assertEqual(self.engine._phase.calibration, [])
        self.assertTrue(self.log.has("Calibrating the phase step over 5"))


class EngineLoopTests(unittest.TestCase):
    """The worker thread with the new defaults and the Phase view."""

    def setUp(self):
        """Point outputs at a temporary folder and make a demo engine.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.saved_output = paths.OUTPUT_DIR
        paths.OUTPUT_DIR = self.tmp.name
        self.log = ListLog()
        self.engine = Engine(self.log)
        self.engine.use_demo()
        self.controller = None

    def tearDown(self):
        """Stop the engine and the DAQ, restore the output folder.

        Args: None.
        Returns: None.
        """
        self.engine.stop()
        self.assertTrue(self.engine.wait(10))
        if self.controller is not None:
            self.controller.shutdown(5.0)
        paths.OUTPUT_DIR = self.saved_output
        self.tmp.cleanup()

    def test_reference_view_averages_its_own_reference(self):
        """With nothing stored, Reference view averages Ref. frames frames itself.

        Args: None.
        Returns: None.
        """
        self.engine.update_settings(mode="Reference", fps=60.0)
        self.engine.start()
        self.assertTrue(wait_until(lambda: self.log.has(
            "Capturing reference automatically: averaging the next 16 frames.")))
        self.assertTrue(wait_until(lambda: self.engine.latest_packet() is not None))
        self.assertTrue(wait_until(
            lambda: "Capturing" not in self.engine.status()["hint"]))

    def test_phase_view_with_the_simulated_daq(self):
        """Dummy mode's interferometer, a simulated piezo and a calibration.

        Args: None.
        Returns: None.
        """
        self.controller = daq.DaqController(self.log)
        self.controller.request("connect", device=daq.SIMULATED)
        settings = daq.DaqSettings(piezo_output="ao1", piezo_step_v=0.05,
                                   generator_output=daq.OFF)
        stepper = phase.PiezoStepper(self.controller, settings, simulated=True)
        self.engine.update_settings(mode="Phase", fps=120.0, average=1, record=True,
                                    flip_horizontal=True)
        self.engine.set_phase_stepper(stepper)
        self.engine.set_compare(True)
        self.engine.start()
        self.assertTrue(wait_until(lambda: self.log.has("Phase reference captured (5-step).")))
        self.assertTrue(wait_until(lambda: "step measured" in self.engine.status()["hint"]))
        self.assertIn("Phase: black to white", self.engine.status()["range"])
        packet = self.engine.latest_packet()
        self.assertEqual((packet.mode, packet.gamma), ("Phase", 1.0))
        self.assertIsNone(self.engine.latest_compare())  # no comparison in Phase view
        self.assertTrue(wait_until(lambda: self.log.has("Recording AVI")))
        self.engine.request("calibrate_phase")
        event = wait_event(self.engine, ("phase_calibrated", "phase_calibration_failed"))
        self.assertIsNotNone(event)
        kind, (degrees, volts, index) = event
        self.assertEqual(kind, "phase_calibrated")
        self.assertAlmostEqual(degrees, 72, delta=6)
        self.assertAlmostEqual(volts, 0.0625, delta=0.005)
        self.engine.update_settings(mode="Reference", record=False)  # leaving restarts the set
        self.assertTrue(wait_until(lambda: "Phase" not in self.engine.status()["range"]))


if __name__ == "__main__":
    unittest.main()
