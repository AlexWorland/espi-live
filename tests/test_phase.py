"""Camera-free checks for phase-shifting ESPI (espi_live.phase).

Run from the project root:  PYTHONPATH=src:tests python -m unittest tests.test_phase
"""

import math
import time
import unittest
from unittest import mock

import numpy as np

from espi_live import daq, phase

H, W = 60, 80


def fringes(steps, step, shift=0.0, seed=0):
    """Make phase-stepped interferograms of a random speckle field.

    Args: steps is the number of frames; step is the phase step in radians;
        shift is a phase added to every pixel (a uniform deformation), or an
        array of per-pixel phases; seed picks the speckle.
    Returns: (frames, speckle phase): float32 frames I_k = a + b cos(psi +
        shift + k step).
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(40, 80, (H, W))
    b = rng.uniform(10, 30, (H, W))
    psi = rng.uniform(-math.pi, math.pi, (H, W))
    frames = [(a + b * np.cos(psi + shift + k * step)).astype(np.float32)
              for k in range(steps)]
    return frames, psi


def wrapped(values):
    """Wrap phases to (-pi, pi].

    Args: values is an array of radians.
    Returns: The wrapped array.
    """
    return np.angle(np.exp(1j * values))


class PhaseMathTests(unittest.TestCase):
    """Wrapped phase, step estimate, filter and display."""

    def test_both_algorithms_recover_a_deformation(self):
        """The phase difference equals the applied phase change at a quarter-wave step.

        Args: None.
        Returns: None.
        """
        y, x = np.mgrid[0:H, 0:W]
        deformation = 2 * math.pi * x / 30.0
        for steps in phase.PHASE_STEPS:
            before, _ = fringes(steps, phase.NOMINAL_STEP)
            after, _ = fringes(steps, phase.NOMINAL_STEP, deformation)
            change = np.angle(phase.phase_difference(phase.wrapped_phase(after),
                                                     phase.wrapped_phase(before)))
            error = wrapped(change - deformation)
            self.assertLess(float(np.abs(error).max()), 1e-3, steps)

    def test_hariharan_tolerates_a_wrong_step(self):
        """At a 75 degree step 5-step stays near 1 degree while 4-step ripples.

        Args: None.
        Returns: None.
        """
        errors = {}
        for steps in phase.PHASE_STEPS:
            before, _ = fringes(steps, math.radians(75))
            after, _ = fringes(steps, math.radians(75), 1.0)
            change = np.angle(phase.phase_difference(phase.wrapped_phase(after),
                                                     phase.wrapped_phase(before)))
            errors[steps] = math.degrees(float(np.sqrt(np.mean(wrapped(change - 1.0) ** 2))))
        self.assertLess(errors[5], 2.0)
        self.assertGreater(errors[4], 4 * errors[5])

    def test_other_frame_counts_are_refused(self):
        """Only 4 or 5 frames make a phase map.

        Args: None.
        Returns: None.
        """
        with self.assertRaises(ValueError):
            phase.wrapped_phase(fringes(3, phase.NOMINAL_STEP)[0])

    def test_step_angle_measures_the_step(self):
        """Five frames give the step; four frames or a flat scene give None.

        Args: None.
        Returns: None.
        """
        for degrees in (60, 90, 110):
            frames, _ = fringes(5, math.radians(degrees))
            self.assertAlmostEqual(math.degrees(phase.step_angle(frames)), degrees, 3)
        self.assertIsNone(phase.step_angle(fringes(4, phase.NOMINAL_STEP)[0]))
        flat = [np.full((H, W), 50, np.uint8)] * 5
        self.assertIsNone(phase.step_angle(flat))

    def test_calibrated_step_scales_to_a_quarter_wave(self):
        """The voltage scales by 90 over the measured angle, within the trusted range.

        Args: None.
        Returns: None.
        """
        self.assertAlmostEqual(phase.calibrated_step(0.05, math.radians(72)), 0.0625)
        self.assertIsNone(phase.calibrated_step(0.05, None))
        self.assertIsNone(phase.calibrated_step(0.05, math.radians(10)))
        self.assertIsNone(phase.calibrated_step(0.05, math.radians(170)))

    def test_filter_smooths_without_blurring_across_the_wrap(self):
        """The sine/cosine filter keeps a ramp's jump from pi to -pi sharp.

        Args: None.
        Returns: None.
        """
        y, x = np.mgrid[0:H, 0:W]
        ramp = 2 * math.pi * x / 20.0
        rng = np.random.default_rng(3)
        noisy = np.exp(1j * (ramp + rng.normal(0, 0.6, ramp.shape))).astype(np.complex64)
        self.assertIs(phase.filter_phase(noisy, 0), noisy)
        smoothed = phase.filter_phase(noisy, 9)
        inner = (slice(8, -8), slice(8, -8))
        before = np.abs(wrapped(np.angle(noisy) - ramp))[inner].mean()
        after = np.abs(wrapped(np.angle(smoothed) - ramp))[inner].mean()
        self.assertLess(after, before / 3)

    def test_phase_image_maps_one_fringe_to_the_grey_range(self):
        """No change is mid-grey (from either side), -pi black, just under pi white.

        Args: None.
        Returns: None.
        """
        angles = np.array([[0.0, 1e-6, -1e-6, -math.pi + 0.005, math.pi - 0.005,
                            math.pi / 2]])
        image = phase.phase_image(np.exp(1j * angles).astype(np.complex64))
        self.assertEqual(image.dtype, np.uint8)
        for column in range(3):
            self.assertIn(int(image[0, column]), (127, 128))
        self.assertLessEqual(int(image[0, 3]), 1)
        self.assertGreaterEqual(int(image[0, 4]), 254)
        self.assertIn(int(image[0, 5]), (191, 192))


class PhaseSequenceTests(unittest.TestCase):
    """Collecting frames step by step, with settling."""

    def run_sequence(self, sequence, frames):
        """Feed frames, moving the piezo whenever asked.

        Args: sequence is a PhaseSequence; frames is an iterable of frames.
        Returns: (completed sets, moves made as step numbers).
        """
        done, moves = [], []
        for frame in frames:
            if sequence.move_pending:
                moves.append(sequence.step)
                sequence.moved()
            result = sequence.add(frame)
            if result is not None:
                done.append(result)
        return done, moves

    def test_settle_frames_are_dropped_and_steps_averaged(self):
        """Two settle frames per move, one more at step 0, then two frames per step.

        Args: None.
        Returns: None.
        """
        sequence = phase.PhaseSequence(4, frames_per_step=2, settle_frames=2)
        frames = [np.full((2, 2), value, np.uint8) for value in range(40)]
        done, moves = self.run_sequence(sequence, frames)
        self.assertEqual(moves[:5], [0, 1, 2, 3, 0])
        means = [float(step[0, 0]) for step in done[0]]
        self.assertEqual(means, [3.5, 7.5, 11.5, 15.5])  # 3 skipped, then 2 skipped
        self.assertEqual(done[0][0].dtype, np.float32)
        self.assertEqual(len(done), 2)

    def test_a_frame_is_ignored_until_the_piezo_has_moved(self):
        """Nothing is collected while a move is pending.

        Args: None.
        Returns: None.
        """
        sequence = phase.PhaseSequence(5, settle_frames=0)
        self.assertTrue(sequence.move_pending)
        self.assertIsNone(sequence.add(np.zeros((2, 2), np.uint8)))
        self.assertEqual(sequence.frames, [])

    def test_a_new_frame_size_restarts_the_set(self):
        """Frames of two sizes are never mixed in one set.

        Args: None.
        Returns: None.
        """
        sequence = phase.PhaseSequence(4, frames_per_step=2, settle_frames=0)
        sequence.moved()
        sequence.skip = 0
        sequence.add(np.zeros((2, 2), np.uint8))
        self.assertIsNone(sequence.add(np.zeros((3, 3), np.uint8)))  # partial step
        self.assertTrue(sequence.move_pending)
        self.assertEqual(sequence.step, 0)
        sequence.moved()
        sequence.skip = 0
        for _ in range(2):
            sequence.add(np.zeros((2, 2), np.uint8))
        self.assertEqual(len(sequence.frames), 1)
        sequence.moved()
        self.assertIsNone(sequence.add(np.zeros((3, 3), np.uint8)))  # finished step
        self.assertEqual(sequence.frames, [])


class PhaseStateTests(unittest.TestCase):
    """The engine's phase state: references, step counts and calibration."""

    def test_configure_keeps_or_replaces_the_sequence(self):
        """The same settings keep the sequence; others start a new one.

        Args: None.
        Returns: None.
        """
        state = phase.PhaseState()
        first = state.configure(5, 1, 2)
        self.assertIs(state.configure(5, 1, 2), first)
        self.assertIsNot(state.configure(4, 1, 2), first)
        state.restart()
        self.assertIsNone(state.sequence)

    def test_first_set_is_the_reference_and_steps_must_match(self):
        """The first set, F, and a change of algorithm each take a new reference.

        Args: None.
        Returns: None.
        """
        state = phase.PhaseState()
        image, new = state.complete(fringes(5, phase.NOMINAL_STEP)[0], 0)
        self.assertTrue(new)
        self.assertLessEqual(int(np.ptp(image)), 1)  # no change against itself: flat grey
        self.assertIn(int(image[0, 0]), (127, 128))
        self.assertAlmostEqual(math.degrees(state.angle), 90, 3)
        image, new = state.complete(fringes(5, phase.NOMINAL_STEP, 1.0)[0], 3)
        self.assertFalse(new)
        self.assertAlmostEqual(float(np.median(image)), 127.5 + 255 / (2 * math.pi),
                               delta=2)
        _, new = state.complete(fringes(4, phase.NOMINAL_STEP)[0], 0)
        self.assertTrue(new)  # 4-step after 5-step
        self.assertIsNone(state.angle)
        state.clear_reference()
        _, new = state.complete(fringes(4, phase.NOMINAL_STEP)[0], 0)
        self.assertTrue(new)

    def test_calibration_collects_five_estimates(self):
        """The median step comes back once CALIBRATION_SETS sets are measured.

        Args: None.
        Returns: None.
        """
        state = phase.PhaseState()
        self.assertIsNone(state.calibration_result())
        state.start_calibration()
        for degrees in (70, 72, 74, 71, 73):
            state.complete(fringes(5, math.radians(degrees))[0], 0)
            if len(state.calibration) < phase.CALIBRATION_SETS:
                self.assertIsNone(state.calibration_result())
        self.assertAlmostEqual(math.degrees(state.calibration_result()), 72, 2)
        self.assertIsNone(state.calibration)


class FakeController:
    """Stands in for DaqController.move_piezo, recording the levels asked for."""

    def __init__(self, error=None):
        """Start with no moves.

        Args: error is an exception to raise on every move, or None.
        Returns: None.
        """
        self.moves = []
        self.error = error

    def move_piezo(self, settings, volts, timeout):
        """Record a move, or fail.

        Args: settings, volts and timeout as DaqController.move_piezo takes them.
        Returns: None.
        """
        if self.error is not None:
            raise self.error
        self.moves.append((settings.piezo_output, volts, timeout))


PIEZO = daq.DaqSettings(piezo_output="ao1", piezo_voltage_v=0.1, piezo_step_v=0.05,
                        piezo_limit_v=1.0)


class PiezoStepperTests(unittest.TestCase):
    """Levels, safety checks and moves."""

    def test_levels_and_moves(self):
        """Step N is the base voltage plus N steps; a move records the level.

        Args: None.
        Returns: None.
        """
        controller = FakeController()
        stepper = phase.PiezoStepper(controller, PIEZO, simulated=True)
        self.assertEqual(stepper.step_v, 0.05)
        self.assertAlmostEqual(stepper.level(4), 0.3)
        self.assertEqual(stepper.phase(), 0.0)
        stepper.move(2, timeout=0.5)
        self.assertEqual(controller.moves[0][0], "ao1")
        self.assertAlmostEqual(controller.moves[0][1], 0.2)
        self.assertEqual(controller.moves[0][2], 0.5)
        self.assertAlmostEqual(stepper.phase(), 0.2 * phase.DEMO_RAD_PER_VOLT)
        changed = stepper.with_step(0.0625)
        self.assertEqual(changed.step_v, 0.0625)
        self.assertIs(changed.controller, controller)
        self.assertTrue(changed.simulated)
        self.assertEqual(stepper.step_v, 0.05)  # the original is unchanged

    def test_problems(self):
        """No output, no step or a level past the limit is refused in words.

        Args: None.
        Returns: None.
        """
        controller = FakeController()
        self.assertEqual(phase.PiezoStepper(controller, PIEZO).problems(5), [])
        cases = ((daq.replace(PIEZO, piezo_output=daq.OFF), "Assign the piezo"),
                 (daq.replace(PIEZO, piezo_step_v=0.0), "above 0 V"),
                 (daq.replace(PIEZO, piezo_step_v=0.3), "safety limit"),
                 (daq.replace(PIEZO, piezo_voltage_v=-0.1), "below 0 V"))
        for settings, words in cases:
            found = phase.PiezoStepper(controller, settings).problems(5)
            self.assertEqual(len(found), 1)
            self.assertIn(words, found[0])

    def test_a_failed_move_raises_and_keeps_the_old_level(self):
        """PiezoMoveError comes through; the level is not updated.

        Args: None.
        Returns: None.
        """
        stepper = phase.PiezoStepper(FakeController(phase.PiezoMoveError("gone")), PIEZO)
        with self.assertRaises(phase.PiezoMoveError):
            stepper.move(1)
        self.assertIsNone(stepper.volts)


class DemoInterferometerTests(unittest.TestCase):
    """Dummy mode's synthetic interferometer."""

    def test_frames_follow_the_piezo_and_the_deformation(self):
        """Phase steps from the piezo recover the dome's phase change.

        Args: None.
        Returns: None.
        """
        demo = phase.DemoInterferometer(80, 60, seed=1)
        start = demo.start
        self.assertEqual(demo.deformation(start), 0.0)
        self.assertGreater(demo.deformation(start + 5.0), 1.0)

        def phase_map(now):
            """Take a noise-free 5-step set at one moment.

            Args: now is the moment.
            Returns: The complex phase map.
            """
            with mock.patch.object(demo, "rng") as rng:
                rng.normal.return_value = np.zeros((60, 80))
                return phase.wrapped_phase([demo.frame(k * phase.NOMINAL_STEP, now)
                                            for k in range(5)])
        frame = demo.frame(0.0, start)
        self.assertEqual((frame.shape, frame.dtype), ((60, 80), np.uint8))
        change = np.angle(phase.phase_difference(phase_map(start + 2.0), phase_map(start)))
        expected = demo.dome * demo.deformation(start + 2.0)
        bright = (demo.object > 20)
        error = np.abs(wrapped(change - expected))[bright]
        self.assertLess(float(np.median(error)), 0.2)


if __name__ == "__main__":
    unittest.main()
