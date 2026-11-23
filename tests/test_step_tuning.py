"""Camera-free checks for the experimental piezo auto-tune (step_tuning).

Run from the project root:
    PYTHONPATH=src python -m unittest tests.test_step_tuning

A synthetic speckle interferometer stands in for the camera: each pixel is
a + b cos(psi + phase) with a random speckle phase psi, the reference phase
moving by a set step per frame.
"""

import math
import unittest

import numpy as np

from espi_live import step_tuning as st


class Speckle:
    """Frames of a still speckle pattern with a moving reference phase."""

    def __init__(self, seed=0, size=128, noise=0.0):
        """Draw the speckle phases.

        Args: seed seeds the random numbers; size is the frame's width and
            height; noise is the camera noise's standard deviation.
        Returns: None.
        """
        self.rng = np.random.default_rng(seed)
        self.psi = self.rng.uniform(0, 2 * math.pi, (size, size))
        self.noise = noise
        self.phase = 0.0

    def frame(self, step):
        """Advance the reference phase by a step and return the frame.

        Args: step is the phase step in radians.
        Returns: A float image.
        """
        self.phase += step
        image = 100 + 40 * np.cos(self.psi + self.phase)
        if self.noise:
            image = image + self.rng.normal(0, self.noise, self.psi.shape)
        return image


def run(tuner, start_step, seconds, seed=0, noise=0.0, fps=30.0):
    """Drive a tuner against the synthetic interferometer.

    The step is start_step times the tuner's factor, nothing while it
    holds the piezo still.

    Args: tuner is a StepTuner; start_step the step at factor 1 (radians);
        seconds how long to run; seed and noise as for Speckle; fps the rate.
    Returns: (the StepMonitor, the time reached).
    """
    monitor = st.StepMonitor()
    monitor.switch(True)
    source = Speckle(seed, noise=noise)
    now = 0.0
    for _ in range(int(seconds * fps)):
        step = 0.0 if tuner.wants_hold else start_step * tuner.scale
        monitor.feed(source.frame(step), now)
        if tuner.update(now, monitor) is not None:
            tuner.changed(now)
        now += 1 / fps
    return monitor, now


class MonitorTests(unittest.TestCase):
    """D(1) and D(2) from successive frames, and what they say about the step."""

    def test_step_from_ratio(self):
        """q = 2 (1 + cos step): 0 at 180°, 2 at 90°, 4 at 0°; clipped outside.

        Args: None.
        Returns: None.
        """
        self.assertAlmostEqual(st.step_from_ratio(0.0), 180.0)
        self.assertAlmostEqual(st.step_from_ratio(2.0), 90.0)
        self.assertAlmostEqual(st.step_from_ratio(4.0), 0.0)
        self.assertAlmostEqual(st.step_from_ratio(9.0), 0.0)
        self.assertAlmostEqual(st.step_from_ratio(-1.0), 180.0)

    def test_measures_the_step(self):
        """The monitor reads 60°, 120° and 180° steps from the frames.

        Args: None.
        Returns: None.
        """
        for degrees in (60.0, 120.0, 180.0):
            monitor = st.StepMonitor()
            self.assertFalse(monitor.on)
            monitor.feed(np.zeros((8, 8)), 0.0)  # off: ignored
            self.assertIsNone(monitor.stats(0.0))
            monitor.switch(True)
            self.assertTrue(monitor.on)
            source = Speckle()
            for n in range(10):
                monitor.feed(source.frame(math.radians(degrees)), n / 30)
            stats = monitor.stats(0.0)
            self.assertEqual(stats.samples, 8)
            self.assertAlmostEqual(stats.step_deg, degrees, delta=2.0)
            self.assertLess(stats.flicker, 0.05)
            self.assertIsNone(monitor.stats(1.0))  # nothing that late
        monitor.switch(True)
        still = np.ones((8, 8))
        for n in range(3):
            monitor.feed(still, n)  # a capture stick repeating a frame
        self.assertEqual((monitor.stats(0.0).d1, monitor.stats(0.0).flicker), (0.0, 0.0))

    def test_noise_is_taken_off(self):
        """With the still-piezo noise taken off, a noisy 90° step reads right.

        Args: None.
        Returns: None.
        """
        monitor = st.StepMonitor()
        monitor.switch(True)
        source = Speckle(noise=15.0)
        for n in range(40):
            monitor.feed(source.frame(0.0), n / 30)
        noise = monitor.stats(0.0).d1
        for n in range(40, 80):
            monitor.feed(source.frame(math.pi / 2), n / 30)
        start = 41 / 30
        self.assertGreater(monitor.stats(start).step_deg, 95.0)  # biased by the noise
        self.assertAlmostEqual(monitor.stats(start, noise).step_deg, 90.0, delta=4.0)
        self.assertEqual(monitor.stats(0.0, 10 * noise).q, 2.0)  # all noise

    def test_new_size_starts_again(self):
        """A frame of another size drops the frames before it.

        Args: None.
        Returns: None.
        """
        monitor = st.StepMonitor()
        monitor.switch(True)
        source = Speckle()
        for n in range(3):
            monitor.feed(source.frame(1.0), n)
        monitor.feed(np.ones((32, 32)), 3)
        monitor.feed(np.ones((32, 32)), 4)
        self.assertEqual(monitor.stats(0.0).samples, 1)
        monitor.switch(False)
        self.assertIsNone(monitor.stats(0.0))


class TunerTests(unittest.TestCase):
    """The tuner finds a 180° step from either side and holds it."""

    def test_finds_pi_from_either_side(self):
        """Steps from 0.5 pi to 1.7 pi end within 6% of pi, holding.

        Args: None.
        Returns: None.
        """
        for start in (0.5, 0.9, 1.3, 1.7):
            tuner = st.StepTuner()
            run(tuner, start * math.pi, 120, noise=3.0)
            self.assertEqual(tuner.state, "holding", start)
            self.assertAlmostEqual(start * tuner.scale, 1.0, delta=0.06)
            self.assertIn("holding ×", tuner.describe())
            self.assertGreater(tuner.noise, 0)

    def test_starts_by_holding_still_for_the_noise(self):
        """It first asks for the piezo held, then measures, then moves.

        Args: None.
        Returns: None.
        """
        tuner = st.StepTuner()
        self.assertTrue(tuner.wants_hold)
        self.assertIn("held still", tuner.describe())
        monitor = st.StepMonitor()
        monitor.switch(True)
        self.assertIsNone(tuner.update(0.0, monitor))  # no frames yet
        tuner.since = None
        run(tuner, math.pi / 2, 4.5)  # no noise: repeated frames measure as 0
        self.assertFalse(tuner.wants_hold)
        self.assertEqual(tuner.state, "measuring")
        self.assertIn("measuring the step", tuner.describe())

    def test_probe_repeats_when_lost_in_noise(self):
        """A probe that barely moves the step is repeated larger.

        Args: None.
        Returns: None.
        """
        tuner = st.StepTuner()
        tuner.state, tuner.seen, tuner.kept, tuner.since = "probing", 90.0, 1.0, 0.0
        monitor = FixedMonitor(st.StepStats(2.0, 90.5, 0.0, 100, 10.0))
        self.assertAlmostEqual(tuner.update(10.0, monitor), 1.1)
        self.assertEqual(tuner.probe, 2 * st.PROBE)
        self.assertIn("probing ×1.100", tuner.describe())

    def test_a_wrong_jump_is_undone(self):
        """A jump that makes the step look worse goes to the other side instead.

        Args: None.
        Returns: None.
        """
        tuner = st.StepTuner()
        tuner.state, tuner.scale, tuner.since = "measuring", 2.0, 0.0
        tuner.jump = (1.0, 120.0, 240.0)
        monitor = FixedMonitor(st.StepStats(3.0, 60.0, 0.0, 100, 10.0))
        self.assertAlmostEqual(tuner.update(10.0, monitor), 0.75)  # 180 / 240
        self.assertIsNone(tuner.jump)

    def test_a_zero_step_jumps_a_limited_way(self):
        """A step read as 0° jumps by at most MAX_JUMP.

        Args: None.
        Returns: None.
        """
        tuner = st.StepTuner()
        tuner.state, tuner.seen, tuner.kept, tuner.since = "probing", 0.0, 1.0, 0.0
        monitor = FixedMonitor(st.StepStats(3.9, 10.0, 0.0, 100, 10.0))
        self.assertAlmostEqual(tuner.update(10.0, monitor), st.MAX_JUMP)

    def test_gives_up_instead_of_running_away(self):
        """A factor outside its range sends it back to 1 and stops it.

        Args: None.
        Returns: None.
        """
        tuner = st.StepTuner()
        tuner.state, tuner.seen, tuner.since = "probing", 10.0, 0.0
        tuner.kept = tuner.scale = 3.5
        monitor = FixedMonitor(st.StepStats(3.9, 20.0, 0.0, 100, 10.0))
        self.assertEqual(tuner.update(10.0, monitor), 1.0)
        self.assertEqual(tuner.state, "lost")
        self.assertIsNone(tuner.update(20.0, monitor))
        self.assertIn("could not find the step", tuner.describe())

    def test_holding_watches_and_restarts(self):
        """While holding, a drift well away from pi starts the search again.

        Args: None.
        Returns: None.
        """
        tuner = st.StepTuner()
        tuner.state, tuner.best, tuner.since = "holding", 0.1, 0.0
        steady = FixedMonitor(st.StepStats(0.05, 175.0, 0.0, 100, 10.0))
        self.assertIsNone(tuner.update(10.0, steady))
        self.assertEqual(tuner.best, 0.05)
        drifted = FixedMonitor(st.StepStats(0.5, 140.0, 0.0, 100, 10.0))
        self.assertIsNone(tuner.update(20.0, drifted))
        self.assertEqual(tuner.state, "measuring")

    def test_hill_climb_near_pi(self):
        """Near pi it keeps what improves, turns and halves, then holds.

        Args: None.
        Returns: None.
        """
        tuner = st.StepTuner()
        tuner.state, tuner.since = "measuring", 0.0
        monitor = FixedMonitor(st.StepStats(0.3, 160.0, 0.0, 100, 10.0))
        self.assertAlmostEqual(tuner.update(10.0, monitor), 1 + st.FINE_STEP)
        monitor.stats_value = st.StepStats(0.1, 170.0, 0.0, 100, 10.0)  # better
        self.assertAlmostEqual(tuner.update(20.0, monitor), (1 + st.FINE_STEP) ** 2)
        monitor.stats_value = st.StepStats(0.2, 165.0, 0.0, 100, 10.0)  # worse
        kept = 1 + st.FINE_STEP
        self.assertAlmostEqual(tuner.update(30.0, monitor), kept * (1 - st.FINE_STEP / 2))
        tuner.step = st.MIN_STEP  # the next miss holds
        self.assertAlmostEqual(tuner.update(40.0, monitor), kept)
        self.assertEqual(tuner.state, "holding")


class FixedMonitor:
    """A monitor that always reports the same statistics."""

    def __init__(self, stats):
        """Keep the statistics to report.

        Args: stats is a StepStats.
        Returns: None.
        """
        self.stats_value = stats

    def stats(self, since, noise=0.0):
        """Report the fixed statistics.

        Args: since and noise are ignored.
        Returns: The StepStats given.
        """
        return self.stats_value


if __name__ == "__main__":
    unittest.main()
