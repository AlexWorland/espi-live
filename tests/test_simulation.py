"""Camera-free checks for the simulated ESPI signal used in dummy mode.

Run from the project root:
    PYTHONPATH=src:tests python -m unittest test_simulation
"""

import math
import unittest

import numpy as np

from espi_live import simulation
from espi_live.simulation import FringeTable


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


if __name__ == "__main__":
    unittest.main()
