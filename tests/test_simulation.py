"""Camera-free checks for the simulated ESPI signal used in dummy mode.

Run from the project root:
    PYTHONPATH=src:tests python -m unittest test_simulation
"""

import math
import unittest

import numpy as np

from espi_live import simulation
from espi_live.simulation import (DEFAULT_OBJECT, K_PER_NM, SIMULATED_OBJECTS,
                                  DemoSource, FringeTable, deformation_nm)


def brute(u, beta, n, samples=20000):
    """Average exp(i u sin(theta)) over n cycles from theta = beta, by sampling.

    Args: u and beta are numbers; n is the number of cycles in the exposure;
        samples is how many evenly spaced instants to average.
    Returns: The complex time average.
    """
    s = (np.arange(samples) + 0.5) / samples
    return np.exp(1j * u * np.sin(2 * np.pi * n * s + beta)).mean()


def j0(u):
    """Bessel J0 by direct numerical integration (independent of the module).

    Args: u is a number.
    Returns: J0(u).
    """
    t = (np.arange(20000) + 0.5) * np.pi / 20000
    return float(np.cos(u * np.sin(t)).mean())


class FringeTableTests(unittest.TestCase):
    """The exposure-average table matches direct integration."""

    def test_whole_cycles_give_j0(self):
        """A whole number of cycles per exposure averages to J0, whatever the start phase.

        Args: None.
        Returns: None.
        """
        table = FringeTable(30, 8)
        u = np.array([0, 2.404826, 5.520078, 10, 16, 29.9])
        for beta in (0.0, 1.0, 4.0):
            got = table.values(u, np.full_like(u, beta))
            for value, expected in zip(got, (j0(x) for x in u)):
                self.assertLess(abs(value - expected), 3e-3)
        zeros = table.values(np.array([2.404826, 5.520078]), np.zeros(2))
        self.assertTrue(np.all(np.abs(zeros) < 3e-3))

    def test_fractional_cycles_match_brute_force(self):
        """Few or fractional cycles give the exact average, start phase included.

        Args: None.
        Returns: None.
        """
        betas = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        for n in (0.01, 0.25, 1.0, 1.3, 8.7):
            table = FringeTable(30, n)
            for u in np.linspace(0, 30, 31):
                got = table.values(np.full_like(betas, u), betas)
                expected = np.array([brute(u, b, n) for b in betas])
                self.assertLessEqual(float(np.abs(got - expected).max()), 3e-3,
                                     f"n={n}, u={u}")

    def test_beta_is_periodic(self):
        """The start phase is taken modulo 2 pi.

        Args: None.
        Returns: None.
        """
        table = FringeTable(20, 1.3)
        u = np.array([3.0, 7.5, 19.0])
        beta = np.array([0.4, 2.0, 5.9])
        base = table.values(u, beta)
        self.assertLess(np.abs(table.values(u, beta + 2 * np.pi) - base).max(), 1e-6)
        self.assertLess(np.abs(table.values(u, beta - 6 * np.pi) - base).max(), 1e-6)

    def test_zero_amplitude_is_one(self):
        """No motion leaves the interference term unchanged.

        Args: None.
        Returns: None.
        """
        table = FringeTable(10, 0.3)
        beta = np.linspace(0, 2 * np.pi, 7)
        self.assertLess(np.abs(table.values(np.zeros(7), beta) - 1).max(), 1e-6)

    def test_u_beyond_range_is_clamped(self):
        """An amplitude past the table's range reads its last entry.

        Args: None.
        Returns: None.
        """
        table = FringeTable(12, 2.5)
        far = table.values(np.array([24.0]), np.array([0.0]))
        edge = table.values(np.array([12.0]), np.array([0.0]))
        self.assertTrue(np.isfinite(far).all())
        self.assertEqual(far[0], edge[0])

    def test_values_keep_the_input_shape(self):
        """A 2-D amplitude map gives a 2-D complex64 result.

        Args: None.
        Returns: None.
        """
        table = FringeTable(5, 1.0)
        got = table.values(np.zeros((3, 4)), np.zeros((3, 4)))
        self.assertEqual((got.shape, got.dtype), ((3, 4), np.complex64))
        self.assertEqual((table.u_max, table.cycles), (5, 1.0))

    def test_sensitivity_matches_the_wavelength(self):
        """K is the out-of-plane sensitivity 4 pi / lambda for HeNe light.

        Args: None.
        Returns: None.
        """
        self.assertAlmostEqual(simulation.K_PER_NM, 4 * math.pi / 632.8)


def expected_frame(intensity, scale=1.0):
    """Render what the ideal sensor should report for an intensity map.

    Args: intensity is a float array in grey levels at 8000 us and 0 dB;
        scale is the exposure x gain factor relative to that.
    Returns: The uint8 frame: rounded, clipped to 0..255.
    """
    return np.clip(np.rint(intensity * scale), 0, 255).astype(np.uint8)


def max_error(frame, expected):
    """Largest absolute difference between two uint8 frames.

    Args: frame and expected are uint8 arrays of one shape.
    Returns: The difference in grey levels, as an int.
    """
    return int(np.abs(frame.astype(np.int16) - expected.astype(np.int16)).max())


class SceneTests(unittest.TestCase):
    """Speckle, plate, ideal sensor, Still and Slow deformation."""

    def source(self, obj=None, width=96, height=64, label="DEMO1"):
        """Make a small simulated source.

        Args: obj is a SIMULATED_OBJECTS name or None for the default;
            width, height and label are passed to DemoSource.
        Returns: The DemoSource.
        """
        source = DemoSource(width, height, label)
        if obj is not None:
            source.set_object(obj)
        return source

    def test_frames_have_requested_shape(self):
        """Frames are Mono8 at the configured size; the default object is Slow deformation.

        Args: None.
        Returns: None.
        """
        source = self.source()
        frame = source.next_frame(8000, 0, 30)
        self.assertEqual((frame.shape, frame.dtype), ((64, 96), np.uint8))
        self.assertEqual(source.object, DEFAULT_OBJECT)
        self.assertEqual(DEFAULT_OBJECT, "Slow deformation")
        self.assertEqual(SIMULATED_OBJECTS,
                         ("Still", "Slow deformation", "Vibration", "Follow DAQ"))
        self.assertEqual(DemoSource().next_frame(8000, 0, 30).shape, (640, 960))

    def test_same_label_repeats_and_labels_differ(self):
        """Each demo camera has its own speckle, identical from run to run.

        Args: None.
        Returns: None.
        """
        first = self.source().next_frame(8000, 0, 30)
        again = self.source().next_frame(8000, 0, 30)
        other = self.source(label="DEMO2").next_frame(8000, 0, 30)
        np.testing.assert_array_equal(first, again)
        self.assertFalse(np.array_equal(first, other))
        unlabeled = DemoSource(96, 64).next_frame(8000, 0, 30)
        np.testing.assert_array_equal(unlabeled, self.source(label="DEMO").next_frame(8000, 0, 30))

    def test_speckle_statistics(self):
        """The spot centre averages about 64 levels; speckle is fully developed; little clips.

        Args: None.
        Returns: None.
        """
        source = self.source("Still", 480, 320)
        middle = source.background[128:192, 192:288]
        self.assertLess(abs(float(middle.mean()) - 64), 10)
        speckle = source.speckle_intensity[source.plate_mask]
        self.assertTrue(0.8 < float(speckle.std() / speckle.mean()) < 1.2)
        frame = source.next_frame(8000, 0, 30)
        self.assertLess(float((frame[source.plate_mask] == 255).mean()), 0.01)

    def test_sensor_is_linear_and_clips(self):
        """Brightness is proportional to exposure and to linear gain, then clips at 255.

        Args: None.
        Returns: None.
        """
        source = self.source("Still")
        intensity = source.background + source.modulation * np.cos(source.speckle_phase)
        self.assertLessEqual(max_error(source.next_frame(4000, 0, 30),
                                       expected_frame(intensity, 0.5)), 1)
        bright = source.next_frame(8000, 6.0206, 30)
        self.assertLessEqual(max_error(bright, expected_frame(intensity, 2.0)), 1)
        self.assertTrue((source.next_frame(50000, 20, 30) == 255).any())
        self.assertEqual(int(source.next_frame(1, 0, 30).max()), 0)

    def test_still_frames_are_identical(self):
        """Nothing moves, so every frame is the same.

        Args: None.
        Returns: None.
        """
        source = self.source("Still")
        first = source.next_frame(8000, 0, 30)
        for _ in range(9):
            np.testing.assert_array_equal(source.next_frame(8000, 0, 30), first)

    def test_deformation_cycle(self):
        """Ramp up over 5 s, hold 1 s, ramp down over 5 s, rest 1 s, repeat.

        Args: None.
        Returns: None.
        """
        peak = 8 * 632.8 / 2
        self.assertEqual(deformation_nm(0), 0)
        self.assertAlmostEqual(deformation_nm(5.0), peak)
        self.assertAlmostEqual(deformation_nm(5.5), peak)
        self.assertAlmostEqual(deformation_nm(11.0), 0)
        self.assertEqual(deformation_nm(11.5), 0)
        self.assertAlmostEqual(deformation_nm(14.5), deformation_nm(2.5))
        self.assertAlmostEqual(deformation_nm(8.5), deformation_nm(2.5))
        ramp = [deformation_nm(t) for t in np.linspace(0.01, 4.99, 50)]
        self.assertTrue(all(b > a for a, b in zip(ramp, ramp[1:])))

    def test_deformation_never_moves_outside_the_plate(self):
        """The mount around the plate is rigid.

        Args: None.
        Returns: None.
        """
        source = self.source()
        rest = source.next_frame(8000, 0, 30)
        source.clock_s = 5.5
        bent = source.next_frame(8000, 0, 30)
        outside = ~source.plate_mask
        np.testing.assert_array_equal(rest[outside], bent[outside])
        self.assertFalse(np.array_equal(rest[source.plate_mask], bent[source.plate_mask]))

    def test_deformation_frame_matches_the_phase(self):
        """A deformed frame is the interferogram with phase K * d * load shape.

        Args: None.
        Returns: None.
        """
        source = self.source()
        source.clock_s = 5.5
        frame = source.next_frame(8000, 0, 30)
        phase = source.speckle_phase + K_PER_NM * deformation_nm(5.5) * source.load_shape
        intensity = source.background + source.modulation * np.cos(phase)
        self.assertLessEqual(max_error(frame, expected_frame(intensity)), 1)

    def test_load_peaks_near_the_drive_point(self):
        """The point-load deflection peaks next to the drive point, and is 0 off the plate.

        Args: None.
        Returns: None.
        """
        source = self.source(width=480, height=320)
        self.assertAlmostEqual(float(source.load_shape.max()), 1.0, places=5)
        self.assertEqual(float(np.abs(source.load_shape[~source.plate_mask]).max()), 0.0)
        y, x = np.unravel_index(np.argmax(source.load_shape), source.load_shape.shape)
        columns = np.flatnonzero(source.plate_mask.any(axis=0))
        rows = np.flatnonzero(source.plate_mask.any(axis=1))
        u = (x - columns[0]) / (columns[-1] - columns[0])
        v = (y - rows[0]) / (rows[-1] - rows[0])
        self.assertLess(abs(u - 0.37), 0.1)
        self.assertLess(abs(v - 0.41), 0.1)

    def test_clock_advances_by_frame_period(self):
        """The sensor clock moves one frame period per frame, independent of wall time.

        Args: None.
        Returns: None.
        """
        source = self.source("Still")
        for _ in range(3):
            source.next_frame(8000, 0, 30)
        self.assertAlmostEqual(source.clock_s, 0.1, delta=1e-9)
        source.next_frame(8000, 0, 0)
        self.assertAlmostEqual(source.clock_s, 1.1, delta=1e-9)

    def test_unknown_object_is_rejected(self):
        """Only the four simulated objects exist; re-choosing one is harmless.

        Args: None.
        Returns: None.
        """
        source = self.source()
        with self.assertRaises(ValueError):
            source.set_object("Wobble")
        source.set_object("Still")
        source.set_object("Still")
        self.assertEqual(source.object, "Still")


if __name__ == "__main__":
    unittest.main()
