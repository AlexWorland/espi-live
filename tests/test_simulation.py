"""Camera-free checks for the simulated ESPI signal used in dummy mode.

Run from the project root:
    PYTHONPATH=src:tests python -m unittest test_simulation
"""

import math
import unittest
from unittest import mock

import numpy as np

from espi_live import simulation
from espi_live.daq import HOLD, SINE, Drive
from espi_live.simulation import (DEFAULT_OBJECT, K_PER_NM, SIMULATED_OBJECTS,
                                  PIEZO_TRIANGLE_HZ, PIEZO_TRIANGLE_NM, DemoSource,
                                  FringeTable, bessel_j, bessel_zero, deformation_nm,
                                  mode_frequency_hz, object_note,
                                  reference_ramp_average, triangle_phase)
from list_log import ListLog


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


def jm(m, x, samples=400):
    """Bessel J_m by Bessel's integral, independent of the module.

    Args: m is the order; x is an array.
    Returns: J_m(x), same shape.
    """
    t = (np.arange(samples) + 0.5) * np.pi / samples
    x = np.asarray(x, np.float64)
    return np.cos(m * t - x[..., None] * np.sin(t)).mean(axis=-1)


def drum_polar(width, height):
    """Each pixel centre's drum coordinates, as the design defines them.

    Args: width and height are the frame size.
    Returns: (rho, theta): radius over the drum radius (0.44 x the shorter
        side) and the angle counter-clockwise from +x, with y up on screen.
    """
    y = np.arange(height)[:, None] + 0.5 - height / 2
    x = np.arange(width)[None, :] + 0.5 - width / 2
    radius = 0.44 * min(width, height)
    return np.hypot(x, y) / radius, np.arctan2(-y, x)


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

    def test_whole_cycles_ignore_the_start_phase(self):
        """Only a fractional cycle count makes the average depend on the start phase.

        Args: None.
        Returns: None.
        """
        self.assertFalse(FringeTable(16, 8).depends_on_beta)
        self.assertFalse(FringeTable(16, 8000e-6 * 1000).depends_on_beta)
        self.assertTrue(FringeTable(16, 8.7).depends_on_beta)
        self.assertTrue(FringeTable(16, 0.3).depends_on_beta)

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
    """Speckle, drumhead, ideal sensor, Still and Slow deformation."""

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
        speckle = source.speckle_intensity[source.drum_mask]
        self.assertTrue(0.8 < float(speckle.std() / speckle.mean()) < 1.2)
        frame = source.next_frame(8000, 0, 30)
        self.assertLess(float((frame[source.drum_mask] == 255).mean()), 0.01)

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

    def test_deformation_never_moves_outside_the_drum(self):
        """The rim and the mount around the drumhead are rigid.

        Args: None.
        Returns: None.
        """
        source = self.source()
        rest = source.next_frame(8000, 0, 30)
        source.clock_s = 5.5
        bent = source.next_frame(8000, 0, 30)
        outside = ~source.drum_mask
        np.testing.assert_array_equal(rest[outside], bent[outside])
        self.assertFalse(np.array_equal(rest[source.drum_mask], bent[source.drum_mask]))

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

    def test_drum_is_a_centred_circle(self):
        """The drumhead fills 88 % of the frame height, centred; the corners are mount.

        Args: None.
        Returns: None.
        """
        source = self.source(width=480, height=320)
        rho, _ = drum_polar(480, 320)
        np.testing.assert_array_equal(source.drum_mask, rho < 1)
        self.assertTrue(source.drum_mask[160, 240])
        self.assertFalse(source.drum_mask[0, 0] or source.drum_mask[160, 20])

    def test_load_peaks_near_the_drive_point(self):
        """The point-load dent peaks at the drive point (1/3 radius, 30 degrees), 0 off the drum.

        Args: None.
        Returns: None.
        """
        source = self.source(width=480, height=320)
        self.assertAlmostEqual(float(source.load_shape.max()), 1.0, places=5)
        self.assertEqual(float(np.abs(source.load_shape[~source.drum_mask]).max()), 0.0)
        y, x = np.unravel_index(np.argmax(source.load_shape), source.load_shape.shape)
        radius = 0.44 * 320
        expected_x = 240 + 0.33 * radius * math.cos(math.radians(30))
        expected_y = 160 - 0.33 * radius * math.sin(math.radians(30))
        self.assertLess(math.hypot(x + 0.5 - expected_x, y + 0.5 - expected_y), 0.1 * radius)

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


def brute_frame(source, exposure_us, t0, samples=8000):
    """Integrate the ideal sensor's exposure by brute force.

    Args: source is a DemoSource whose object and drive are set; exposure_us
        is the exposure; t0 is the exposure start (s); samples is how many
        instants to average.
    Returns: The float intensity map, scaled for the exposure (0 dB).
    """
    frequency, path = source.harmonic_path_nm()
    static = source.speckle_phase.astype(np.float64) - K_PER_NM * source.static_piezo_nm()
    duration = exposure_us * 1e-6
    total = np.zeros(static.shape)
    for step in range(samples):
        t = t0 + (step + 0.5) * duration / samples
        motion = (K_PER_NM * np.imag(path * np.exp(2j * np.pi * frequency * t))
                  if path is not None else 0.0)
        ramp = triangle_phase(t) if source.piezo_on else 0.0
        total += np.cos(static + motion - ramp)
    intensity = source.background + source.modulation * total / samples
    return intensity * exposure_us / 8000


class VibrationTests(unittest.TestCase):
    """Time-averaged vibration, the DAQ drive and the piezo reference mirror."""

    def source(self, obj, drive=None, width=96, height=64, log=None):
        """Make a small simulated source with an object and a drive.

        Args: obj is a SIMULATED_OBJECTS name; drive is a Drive or None;
            width and height are the frame size; log is passed through.
        Returns: The DemoSource.
        """
        source = DemoSource(width, height, "DEMO1", log)
        source.set_object(obj)
        if drive is not None:
            source.set_drive(drive)
        return source

    def test_bessel_functions_and_zeros(self):
        """J_m and its zeros match the standard values.

        Args: None.
        Returns: None.
        """
        np.testing.assert_allclose(bessel_j(0, np.array([0.0, 2.404826])), [1.0, 0.0],
                                   atol=1e-6)
        self.assertAlmostEqual(float(bessel_j(1, np.array([1.0]))[0]), 0.4400506, places=6)
        x = np.linspace(0, 25, 101)
        for m in (0, 1, 4, 9):
            np.testing.assert_allclose(bessel_j(m, x), jm(m, x), atol=1e-6)
        for (m, n), value in {(0, 1): 2.404826, (1, 1): 3.831706, (2, 1): 5.135622,
                              (0, 2): 5.520078, (3, 2): 9.761023}.items():
            self.assertAlmostEqual(bessel_zero(m, n), value, delta=1e-5)

    def test_mode_frequencies(self):
        """Drum modes follow the Bessel zeros; (1, 1) is exactly 1000 Hz.

        Args: None.
        Returns: None.
        """
        self.assertAlmostEqual(mode_frequency_hz(1, 1), 1000.0, delta=1e-9)
        self.assertAlmostEqual(mode_frequency_hz(0, 1), 1000 * 2.404826 / 3.831706, delta=1e-2)
        self.assertAlmostEqual(mode_frequency_hz(2, 1), 1000 * 5.135622 / 3.831706, delta=1e-2)

    def test_vibration_is_the_drum_mode(self):
        """The built-in drive is mode (1, 1): J1 across a nodal diameter, 16 rad peak.

        Args: None.
        Returns: None.
        """
        source = self.source("Vibration")
        frequency, path = source.harmonic_path_nm()
        self.assertAlmostEqual(frequency, 1000.0, delta=1e-9)
        rho, theta = drum_polar(96, 64)
        shape = jm(1, 3.831706 * np.minimum(rho, 1)) * np.cos(theta - math.radians(30))
        shape = np.where(rho < 1, shape, 0)
        expected = shape * (16.0 / K_PER_NM / np.abs(shape).max())
        scale = float(np.abs(expected).max())
        self.assertLess(float(np.abs(path - expected).max()), 2e-3 * scale)
        self.assertAlmostEqual(float(K_PER_NM * np.abs(path).max()), 16.0, delta=0.05)
        self.assertEqual(float(np.abs(path[~source.drum_mask]).max()), 0.0)
        self.assertEqual(self.source("Still").harmonic_path_nm(), (0.0, None))
        self.assertEqual(self.source("Slow deformation").harmonic_path_nm(), (0.0, None))

    def test_vibration_matches_brute_force_whole_and_fractional(self):
        """Frames equal the exposure integral within a grey level, whole or fractional cycles.

        Args: None.
        Returns: None.
        """
        for exposure in (8000, 8700):
            for t0 in (0.0, 0.0123):
                source = self.source("Vibration")
                source.clock_s = t0
                frame = source.next_frame(exposure, 0, 30)
                expected = expected_frame(brute_frame(source, exposure, t0))
                self.assertLessEqual(max_error(frame, expected), 1, (exposure, t0))

    def test_follow_daq_matches_brute_force_with_reference_modulation(self):
        """A driven drumhead with a piezo sine and a held centre is integrated exactly.

        Args: None.
        Returns: None.
        """
        drive = Drive("sine", 1000.0, 0.4, piezo_mode=SINE, piezo_v=0.02,
                      piezo_amplitude_v=0.3, piezo_phase_deg=40.0)
        source = self.source("Follow DAQ", drive)
        source.clock_s = 0.0071
        frame = source.next_frame(3300, 0, 30)
        expected = expected_frame(brute_frame(source, 3300, 0.0071))
        self.assertLessEqual(max_error(frame, expected), 1)

    def test_whole_cycles_do_not_flicker(self):
        """With whole cycles per exposure every frame is the same; otherwise the phase walks.

        Args: None.
        Returns: None.
        """
        source = self.source("Vibration")
        first = source.next_frame(8000, 0, 30)
        for _ in range(4):
            np.testing.assert_array_equal(source.next_frame(8000, 0, 30), first)
        walking = self.source("Vibration")
        frames = [walking.next_frame(8700, 0, 30) for _ in range(2)]
        self.assertFalse(np.array_equal(frames[0], frames[1]))

    def test_follow_daq_resonance(self):
        """0.1 V at the fundamental (0, 1) gives 16 rad; off resonance far less; off is still.

        Args: None.
        Returns: None.
        """
        fundamental = mode_frequency_hz(0, 1)
        _, path = self.source("Follow DAQ", Drive("sine", fundamental, 0.1)).harmonic_path_nm()
        peak = float(K_PER_NM * np.abs(path).max())
        self.assertAlmostEqual(peak, 16.0, delta=0.32)
        _, off = self.source("Follow DAQ", Drive("sine", 800.0, 0.1)).harmonic_path_nm()
        self.assertLess(float(K_PER_NM * np.abs(off).max()), 0.3 * peak)
        idle = self.source("Follow DAQ", Drive())
        self.assertEqual(idle.harmonic_path_nm(), (0.0, None))
        still = self.source("Still")
        np.testing.assert_array_equal(idle.next_frame(8000, 0, 30), still.next_frame(8000, 0, 30))

    def test_follow_daq_square_uses_the_fundamental(self):
        """A TTL square wave drives like a sine of (4 / pi) x 2.5 V.

        Args: None.
        Returns: None.
        """
        square = self.source("Follow DAQ", Drive("square", 627.0)).harmonic_path_nm()[1]
        sine = self.source("Follow DAQ", Drive("sine", 627.0, 4 / np.pi * 2.5)).harmonic_path_nm()[1]
        np.testing.assert_allclose(square, sine, rtol=1e-6, atol=1e-6)

    def test_piezo_hold_shifts_the_phase_in_every_object(self):
        """A held piezo level moves the reference mirror 1500 nm per volt.

        Args: None.
        Returns: None.
        """
        hold = Drive(piezo_mode=HOLD, piezo_v=0.05)
        for obj in SIMULATED_OBJECTS:
            self.assertAlmostEqual(self.source(obj, hold).static_piezo_nm(), 75.0)
        source = self.source("Still", hold)
        phase = source.speckle_phase - K_PER_NM * 75.0
        intensity = source.background + source.modulation * np.cos(phase)
        self.assertLessEqual(max_error(source.next_frame(8000, 0, 30),
                                       expected_frame(intensity)), 1)
        self.assertEqual(self.source("Still", Drive()).static_piezo_nm(), 0.0)

    def test_piezo_sine_only_in_follow_daq(self):
        """The piezo's sine is locked to the generator; elsewhere only its centre acts.

        Args: None.
        Returns: None.
        """
        drive = Drive("sine", 1000.0, 0.1, piezo_mode=SINE, piezo_v=0.01,
                      piezo_amplitude_v=0.02, piezo_phase_deg=30.0)
        vibrating = self.source("Vibration", drive)
        builtin = self.source("Vibration")
        np.testing.assert_allclose(vibrating.harmonic_path_nm()[1], builtin.harmonic_path_nm()[1])
        self.assertAlmostEqual(vibrating.static_piezo_nm(), 15.0)
        following = self.source("Follow DAQ", drive)
        plain = self.source("Follow DAQ", Drive("sine", 1000.0, 0.1))
        reference = 30.0 * np.exp(1j * np.pi / 6)
        np.testing.assert_allclose(following.harmonic_path_nm()[1],
                                   plain.harmonic_path_nm()[1] - reference,
                                   rtol=1e-5, atol=1e-4)

    def test_reference_modulation_cancels_where_matched(self):
        """A piezo sine equal to the drum's motion at a pixel makes that pixel still.

        Args: None.
        Returns: None.
        """
        _, motion = self.source("Follow DAQ", Drive("sine", 627.0, 0.1)).harmonic_path_nm()
        target = complex(motion[30, 40])
        drive = Drive("sine", 627.0, 0.1, piezo_mode=SINE,
                      piezo_amplitude_v=abs(target) / 1500.0,
                      piezo_phase_deg=math.degrees(np.angle(target)))
        _, path = self.source("Follow DAQ", drive).harmonic_path_nm()
        self.assertLess(abs(complex(path[30, 40])), 1e-3)

    def test_drag_rebuilds_table_once_per_change(self):
        """The exposure table is rebuilt only when the cycle count changes.

        Args: None.
        Returns: None.
        """
        built = []
        real = simulation.FringeTable

        def counting(u_max, cycles):
            """Build a real table and count it.

            Args: u_max and cycles are FringeTable's.
            Returns: The FringeTable.
            """
            built.append(cycles)
            return real(u_max, cycles)
        source = self.source("Vibration")
        with mock.patch.object(simulation, "FringeTable", counting):
            for exposure in (8000, 8000, 8100, 8100):
                source.next_frame(exposure, 0, 30)
        self.assertEqual(len(built), 2)

    def test_strong_drive_stays_finite(self):
        """An overdriven drum clamps the table range, warns once and stays finite.

        Args: None.
        Returns: None.
        """
        log = ListLog()
        source = self.source("Follow DAQ", Drive("sine", 627.6, 10.0), log=log)
        for _ in range(2):
            frame = source.next_frame(8000, 0, 30)
        self.assertEqual(frame.dtype, np.uint8)
        warnings = [m for level, m, _ in log.records if level == "WARNING"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("finer than", warnings[0])
        self.source("Follow DAQ", Drive("sine", 627.6, 10.0)).next_frame(8000, 0, 30)

    def test_object_note_texts(self):
        """The note under the control says what the simulation is doing.

        Args: None.
        Returns: None.
        """
        self.assertIn("Nothing moves", object_note("Still", Drive()))
        self.assertIn("12 s", object_note("Slow deformation", Drive()))
        vibration = object_note("Vibration", Drive())
        self.assertIn("mode (1, 1)", vibration)
        self.assertIn("1000 Hz", vibration)
        idle = object_note("Follow DAQ", Drive())
        self.assertIn("no DAQ drive", idle)
        self.assertIn("Devices → NI DAQ…", idle)
        sine = object_note("Follow DAQ", Drive("sine", 1000.0, 0.1, device="Dev1: USB-6251",
                                               piezo_mode=HOLD, piezo_v=0.05))
        self.assertIn("Dev1: USB-6251", sine)
        self.assertIn("1000 Hz sine, 0.10 V", sine)
        self.assertIn("piezo holds 0.05 V", sine)
        square = object_note("Follow DAQ", Drive("square", 777.0, device="Sim"))
        self.assertIn("777 Hz square", square)
        piezo = object_note("Follow DAQ", Drive("sine", 1000.0, 0.1, piezo_mode=SINE,
                                                piezo_amplitude_v=0.2, piezo_phase_deg=90))
        self.assertIn("piezo sine 0.20 V at 90°", piezo)
        self.assertIn("piezo holds 0.05 V",
                      object_note("Still", Drive(piezo_mode=HOLD, piezo_v=0.05)))
        self.assertIn("piezo holds 0.01 V",
                      object_note("Vibration", Drive(piezo_mode=SINE, piezo_v=0.01,
                                                     piezo_amplitude_v=0.2)))
        self.assertIn("generator off",
                      object_note("Follow DAQ", Drive(piezo_mode=HOLD, piezo_v=0.05,
                                                      device="Sim")))


class ReferencePiezoTests(unittest.TestCase):
    """The reference piezo's triangle wave: decorrelation between frames."""

    def source(self, obj, piezo=True, width=96, height=64):
        """Make a small simulated source with the triangle on or off.

        Args: obj is a SIMULATED_OBJECTS name; piezo turns the triangle on;
            width and height are the frame size.
        Returns: The DemoSource.
        """
        source = DemoSource(width, height, "DEMO1")
        source.set_object(obj)
        source.set_piezo(piezo)
        return source

    def test_triangle_phase(self):
        """The mirror ramps up for half a period and back down, linearly.

        Args: None.
        Returns: None.
        """
        peak = K_PER_NM * PIEZO_TRIANGLE_NM
        period = 1 / PIEZO_TRIANGLE_HZ
        self.assertEqual((PIEZO_TRIANGLE_HZ, PIEZO_TRIANGLE_NM), (1.0, 1200.0))
        self.assertAlmostEqual(triangle_phase(0.0), 0.0)
        self.assertAlmostEqual(triangle_phase(period / 2), peak)
        self.assertAlmostEqual(triangle_phase(period / 4), peak / 2)
        self.assertAlmostEqual(triangle_phase(3 * period / 4), peak / 2)
        self.assertAlmostEqual(triangle_phase(period), 0.0)
        self.assertAlmostEqual(triangle_phase(5.3 * period), triangle_phase(0.3 * period))

    def test_ramp_average_matches_brute_force(self):
        """The exposure average of the reference ramp is exact, turning points included.

        Args: None.
        Returns: None.
        """
        for t0, duration in ((0.1, 0.008), (0.496, 0.008), (0.998, 0.004),
                             (0.2, 1.7), (0.37, 0.0001)):
            t = t0 + (np.arange(200000) + 0.5) * duration / 200000
            expected = np.exp(-1j * np.vectorize(triangle_phase)(t)).mean()
            self.assertLess(abs(reference_ramp_average(t0, duration) - expected), 1e-6,
                            (t0, duration))
        self.assertLess(abs(reference_ramp_average(0.3, 0.0)
                            - np.exp(-1j * triangle_phase(0.3))), 1e-12)

    def test_still_frames_decorrelate_and_match(self):
        """With the triangle on, a still drum's frames change and match the exposure integral.

        Args: None.
        Returns: None.
        """
        source = self.source("Still")
        self.assertTrue(source.piezo_on)
        source.clock_s = 0.0123
        frame = source.next_frame(8000, 0, 30)
        expected = expected_frame(brute_frame(source, 8000, 0.0123))
        self.assertLessEqual(max_error(frame, expected), 1)
        self.assertFalse(np.array_equal(frame, source.next_frame(8000, 0, 30)))

    def test_vibration_with_piezo_matches_brute_force(self):
        """Vibration and the ramp together, whole and fractional cycles, across a turning point.

        Args: None.
        Returns: None.
        """
        for exposure, t0 in ((8000, 0.0123), (8700, 0.2), (8000, 0.4985), (8700, 0.997)):
            source = self.source("Vibration")
            source.clock_s = t0
            frame = source.next_frame(exposure, 0, 30)
            expected = expected_frame(brute_frame(source, exposure, t0))
            self.assertLessEqual(max_error(frame, expected), 1, (exposure, t0))

    def test_nodal_line_turns_white(self):
        """Frame-to-frame, the (1, 1) mode's still nodal line is bright with the piezo, black without.

        Args: None.
        Returns: None.
        """
        source = self.source("Vibration", width=240, height=160)
        _, path = source.harmonic_path_nm()
        amplitude = K_PER_NM * np.abs(path)
        node = source.drum_mask & (amplitude < 1.0)
        dark = source.drum_mask & (np.abs(amplitude - 2.405) < 0.25)
        source.clock_s = 0.1
        first = source.next_frame(8000, 0, 30).astype(np.float32)
        second = source.next_frame(8000, 0, 30).astype(np.float32)
        change = np.abs(second - first)
        self.assertGreater(float(change[node].mean()), 4 * float(change[dark].mean()))
        self.assertGreater(float(change[node].mean()), 10)
        still = self.source("Vibration", piezo=False, width=240, height=160)
        frames = [still.next_frame(8000, 0, 30) for _ in range(2)]
        np.testing.assert_array_equal(frames[0], frames[1])

    def test_piezo_toggles_and_is_named_in_the_note(self):
        """The toggle switches the ramp; the note names it in every object.

        Args: None.
        Returns: None.
        """
        source = self.source("Still", piezo=False)
        self.assertFalse(source.piezo_on)
        first = source.next_frame(8000, 0, 30)
        np.testing.assert_array_equal(first, source.next_frame(8000, 0, 30))
        source.set_piezo(True)
        self.assertTrue(source.piezo_on)
        for obj in SIMULATED_OBJECTS:
            note = object_note(obj, Drive(), piezo=True)
            self.assertIn("reference piezo", note)
            self.assertIn("1 Hz triangle", note)
            self.assertNotIn("reference piezo", object_note(obj, Drive()))


if __name__ == "__main__":
    unittest.main()
