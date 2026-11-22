"""A simulated ESPI signal for dummy mode: what an ideal camera would record.

The scene is a speckle interferometer: a drumhead (an ideal circular
membrane, no material model) scatters laser light into a speckle field, a
smooth reference beam interferes with it, and the drumhead's out-of-plane
motion w shifts the optical phase by K * w (light
along the surface normal, K = 4 pi / lambda). The DAQ's piezo moves the
reference mirror the same way. The sensor is perfectly clean: each pixel
is the intensity averaged over the exposure, scaled by exposure and gain,
rounded to 8 bits and clipped at 255, with no shot, read, dark or
fixed-pattern noise.

A vibration's exposure average is exact for any number of cycles per
exposure (FringeTable), so few-cycle exposures flicker and walk in phase as
a real camera's do. See docs/notes/simulated-espi-signal-design.md.

Nothing here touches a GUI toolkit or PySpin.
"""

import collections
import functools
import math
import zlib

import cv2
import numpy as np

from espi_live.daq import SINE, TRIANGLE, Drive

# Optics.
WAVELENGTH_NM = 632.8  # HeNe
K_PER_NM = 4 * math.pi / WAVELENGTH_NM  # phase per nm of out-of-plane motion

# Specimen: a drumhead, an ideal circular membrane clamped at its rim, centred
# with a diameter of DRUM_FRACTION of the frame's shorter side; the mount
# around it is rigid. There is no material model: mode (m, n) has m nodal
# diameters and n nodal circles (the rim included), shape
# J_m(j_mn r / R) cos(m (theta - drive angle)) and a frequency proportional
# to the Bessel zero j_mn, scaled so the built-in mode (1, 1) is 1000 Hz.
# Modes with j_mn up to MODE_J_LIMIT are simulated.
DRUM_FRACTION = 0.88
MODE_J_LIMIT = 30.0
MODE_Q = 50.0
# The drive (a small pad: a shaker, a finger) presses at this radius (share of
# R) and angle (counter-clockwise from +x, y up). A point drive on that axis
# excites only the cos(m theta) member of each mode pair. The pad's size
# tapers mode j by exp(-(j / DRIVE_PAD_J)^2).
DRIVE_RADIUS = 0.33
DRIVE_ANGLE_DEG = 30.0
DRIVE_PAD_J = 14.0
RADIAL_SAMPLES = 2048
# Bessel functions: a table over 0..BESSEL_X_MAX, step BESSEL_STEP, orders
# 0..BESSEL_ORDERS - 1.
BESSEL_X_MAX = 34.0
BESSEL_STEP = 0.002
BESSEL_ORDERS = 41

# Slow deformation: a raised-cosine ramp to DEFORMATION_FRINGES fringes over
# RAMP_S, a HOLD_S hold, the ramp back down, then STILL_S at rest.
DEFORMATION_FRINGES = 8
RAMP_S = 5.0
HOLD_S = 1.0
STILL_S = 1.0
CYCLE_S = 2 * RAMP_S + HOLD_S + STILL_S

# Vibration: the built-in drive excites one mode at its resonance.
VIBRATION_MODE = (1, 1)
VIBRATION_HZ = 1000.0
VIBRATION_PEAK_RAD = 16.0  # peak K * amplitude: 5 J0 fringes
# Follow DAQ: a 0.1 V sine at the fundamental, f(0, 1), gives this peak.
GENERATOR_PEAK_RAD = 16.0
GENERATOR_REFERENCE_V = 0.1
TTL_HIGH_V = 5.0  # a counter's square wave; only its fundamental is simulated
PIEZO_NM_PER_V = 1500.0  # 100 nm/V stack behind a gain-15 amplifier
# Reference piezo (the demo toggle): a low triangle wave moves the reference
# mirror PIEZO_TRIANGLE_NM peak to peak, so the reference phase changes
# between frames and the speckle decorrelates (about 90 degrees per frame at
# 30 fps). A vibration's still nodes then light up in Previous view.
PIEZO_TRIANGLE_HZ = 1.0
PIEZO_TRIANGLE_NM = 1200.0

# Ideal sensor: the spot centre averages SPOT_LEVEL grey levels at the
# default exposure and gain; brightness is proportional to both.
REFERENCE_EXPOSURE_US = 8000.0
SPOT_LEVEL = 64.0
# The imaging pupil passes spatial frequencies up to this (cycles per pixel),
# so the speckle intensity (twice that) stays within the sensor's Nyquist
# limit and each pixel's point sample is exact. Grains look 2-3 px wide.
PUPIL_CUTOFF_PER_PX = 0.25
# The Gaussian laser spot falls to this share of its peak at the drum's rim.
SPOT_RIM_LEVEL = 0.5

SIMULATED_OBJECTS = ("Still", "Slow deformation", "Vibration", "Follow DAQ")
DEFAULT_OBJECT = "Slow deformation"

# FringeTable accuracy: the largest interpolation error allowed along each
# axis (the two add), and the amplitude step of the table.
TABLE_U_STEP = 0.08
TABLE_BETA_ERROR = 1e-3
MAX_TABLE_POINTS = 1 << 15
# Above this phase amplitude a vibration's fringes are finer than the
# speckle; pixels beyond it are simulated at this amplitude (with a warning).
TABLE_U_LIMIT = 64.0
# Tables kept per source: the two triangle directions, and a spare.
TABLE_CACHE_SIZE = 4


def _power_of_two(count):
    """Round a point count up to a power of two.

    Args: count is a positive number.
    Returns: The smallest power of two at or above it.
    """
    return 1 << max(1, math.ceil(math.log2(max(2.0, count))))


class FringeTable:
    """The exposure average of exp(i u sin(omega t + beta)), tabulated.

    With Delta phi(t) = u sin(omega t + beta) the harmonic part of the phase
    and beta its phase at the start of the exposure, the sensor records the
    average F(u, beta; n) = (1 / Te) * integral of exp(i Delta phi) over the
    exposure, n = f * Te cycles long. By Jacobi-Anger, F = sum over k of
    J_k(u) exp(ik (beta + pi n)) sinc(pi k n): exactly J0(u) for a whole
    number of cycles, and dependent on beta otherwise.

    The table is built per (u_max, n): an FFT of exp(i u sin theta) over
    theta gives every J_k(u) at once, the sinc weights are applied, and an
    inverse FFT onto a finer grid gives F on a beta grid. Values between grid
    points are bilinear; the beta step is chosen from the spectrum so the
    interpolation stays within TABLE_BETA_ERROR, and the amplitude step
    TABLE_U_STEP keeps it within TABLE_U_STEP ** 2 / 8 (|d2F/du2| <= 1).
    """

    def __init__(self, u_max, cycles, pieces=None):
        """Build the table.

        Args: u_max is the largest phase amplitude (radians) to cover;
            cycles is the number of drive cycles per exposure (n >= 0);
            pieces describes a reference phase phi(s) that changes linearly
            during the exposure (the reference piezo's triangle), averaged
            together with the vibration as exp(-i phi(s)): a tuple of (start,
            length, phase at its middle, rate) per linear piece, with start
            and length as fractions of the exposure and the rate in radians
            per exposure; None means no ramp.
        Returns: None. depends_on_beta is False for a whole number of cycles
            without a ramp, where F is exactly J0(u) whatever the start phase.
        """
        self.u_max = u_max
        self.cycles = cycles
        top = max(float(u_max), TABLE_U_STEP)
        rows = math.ceil(top / TABLE_U_STEP) + 1
        self.u_step = top / (rows - 1)
        u = np.arange(rows) * self.u_step
        coarse = _power_of_two(2 * (top + 32))
        theta = 2 * np.pi * np.arange(coarse) / coarse
        k = np.fft.fftfreq(coarse, 1.0 / coarse)
        spectrum = np.fft.fft(np.exp(1j * u[:, None] * np.sin(theta[None, :])), axis=1)
        # Each term's time average over the exposure: exp(ik(omega t + beta))
        # times the ramp's exp(-i phi) integrates in closed form per linear piece.
        weights = np.zeros(coarse, np.complex128)
        for start, length, middle, rate in pieces or ((0.0, 1.0, 0.0, 0.0),):
            spin = 2 * np.pi * k * cycles
            weights += length * np.exp(1j * (spin * (start + length / 2) - middle)) * np.sinc(
                (spin - rate) * length / (2 * np.pi))
        # A whole number of cycles without a ramp leaves exactly J0: drop the
        # k != 0 terms, whose weights are only rounding noise, so beta cannot
        # matter at all.
        self.depends_on_beta = not (pieces is None and cycles >= 0.5
                                    and abs(cycles - round(cycles)) < 1e-9)
        if not self.depends_on_beta:
            weights[1:] = 0
        spectrum *= weights / coarse
        # |d2F/dbeta2| is at most sum k^2 |a_k| and at most u + u^2.
        curvature = min(float(np.max(np.abs(spectrum) @ (k * k))), top + top * top)
        fine = coarse
        if curvature > 0:
            step = math.sqrt(8 * TABLE_BETA_ERROR / curvature)
            fine = min(MAX_TABLE_POINTS, max(coarse, _power_of_two(2 * np.pi / step)))
        padded = np.zeros((rows, fine), np.complex128)
        half = coarse // 2
        padded[:, :half] = spectrum[:, :half]
        padded[:, fine - half:] = spectrum[:, half:]
        self.columns = fine
        self.beta_step = 2 * np.pi / fine
        self.table = (np.fft.ifft(padded, axis=1) * fine).astype(np.complex64).ravel()

    def prepare(self, u):
        """Work out the amplitude half of the lookup, reusable across frames.

        Args: u is an array of phase amplitudes; values above u_max are
            clamped to it.
        Returns: (offsets, weights): the flat-table offset of each value's
            lower row and the float32 weight of the row above.
        """
        rows = self.table.size // self.columns
        position = np.clip(np.asarray(u, np.float64), 0.0, self.u_step * (rows - 1))
        position /= self.u_step
        lower = np.minimum(position.astype(np.int64), rows - 2)
        return lower * self.columns, (position - lower).astype(np.float32)

    def lookup(self, prepared, beta):
        """Read F for prepared amplitudes at the given start phases.

        Args: prepared is what prepare() returned; beta is an array of start
            phases (radians, any real value) of the same shape.
        Returns: A complex64 array of that shape.
        """
        offsets, weights = prepared
        position = np.mod(np.asarray(beta, np.float64), 2 * np.pi) / self.beta_step
        left = position.astype(np.int64) % self.columns
        right = (left + 1) % self.columns
        along = (position - np.floor(position)).astype(np.float32)
        table = self.table
        low = table[offsets + left] * (1 - along) + table[offsets + right] * along
        offsets = offsets + self.columns
        high = table[offsets + left] * (1 - along) + table[offsets + right] * along
        return (low * (1 - weights) + high * weights).astype(np.complex64)

    def values(self, u, beta):
        """Read F(u, beta) for this table's number of cycles.

        Args: u is an array of phase amplitudes (radians; above u_max they
            are clamped); beta is an array of start phases of the same shape.
        Returns: A complex64 array of that shape.
        """
        return self.lookup(self.prepare(u), beta)


def deformation_nm(t_s):
    """Slow deformation's displacement at the bulge's peak.

    Args: t_s is the simulation time in seconds.
    Returns: The peak displacement in nm: a raised-cosine ramp from 0 to
        DEFORMATION_FRINGES half-wavelengths over RAMP_S, held for HOLD_S,
        ramped back over RAMP_S, then 0 for STILL_S, repeating every CYCLE_S.
    """
    peak = DEFORMATION_FRINGES * WAVELENGTH_NM / 2
    t = t_s % CYCLE_S
    if t < RAMP_S:
        return peak * (1 - math.cos(math.pi * t / RAMP_S)) / 2
    if t < RAMP_S + HOLD_S:
        return peak
    if t < 2 * RAMP_S + HOLD_S:
        return peak * (1 + math.cos(math.pi * (t - RAMP_S - HOLD_S) / RAMP_S)) / 2
    return 0.0


@functools.lru_cache(maxsize=1)
def _bessel_table():
    """Tabulate J_0 .. J_(BESSEL_ORDERS - 1) on 0..BESSEL_X_MAX.

    By Jacobi-Anger, exp(i x sin t) = sum over k of J_k(x) exp(ikt), so an
    FFT over t gives every order at once (numpy only; no SciPy).

    Args: None.
    Returns: (x grid, table) with table[m] = J_m on the grid (float64).
    """
    x = np.arange(0.0, BESSEL_X_MAX + BESSEL_STEP / 2, BESSEL_STEP)
    points = 128  # J_k(x) is negligible for k > x + 30, so 128 cannot alias
    t = 2 * np.pi * np.arange(points) / points
    spectrum = np.fft.fft(np.exp(1j * x[:, None] * np.sin(t[None, :])), axis=1) / points
    return x, np.ascontiguousarray(spectrum[:, :BESSEL_ORDERS].real.T)


def bessel_j(m, x):
    """Bessel function of the first kind.

    Args: m is the order (0..BESSEL_ORDERS - 1); x is an array in
        0..BESSEL_X_MAX.
    Returns: J_m(x) as a float64 array of x's shape.
    """
    grid, table = _bessel_table()
    return np.interp(x, grid, table[m])


@functools.lru_cache(maxsize=None)
def bessel_zero(m, n):
    """The n-th positive zero of J_m.

    Args: m is the order; n counts from 1.
    Returns: j_mn as a float, or math.inf when it lies beyond the table.
    """
    grid, table = _bessel_table()
    values = table[m]
    # j_m1 > m, and below x = m a high order is ~1e-20, where rounding noise
    # flips its sign: search from there.
    start = max(1, int(m / BESSEL_STEP))
    crossings = np.flatnonzero(np.signbit(values[start:-1]) != np.signbit(values[start + 1:]))
    if len(crossings) < n:
        return math.inf
    i = start + crossings[n - 1]
    return float(grid[i] - values[i] * BESSEL_STEP / (values[i + 1] - values[i]))


@functools.lru_cache(maxsize=1)
def drum_modes():
    """List the simulated drum modes.

    Args: None.
    Returns: A tuple of (m, n, j_mn) with j_mn <= MODE_J_LIMIT, by m then n.
    """
    modes = []
    for m in range(BESSEL_ORDERS - 1):
        n = 1
        while bessel_zero(m, n) <= MODE_J_LIMIT:
            modes.append((m, n, bessel_zero(m, n)))
            n += 1
    return tuple(modes)


def mode_frequency_hz(m, n):
    """Resonance frequency of drum mode (m, n).

    Args: m is the number of nodal diameters; n the number of nodal circles,
        the rim included.
    Returns: The frequency in Hz; VIBRATION_MODE, (1, 1), is VIBRATION_HZ.
    """
    return VIBRATION_HZ * bessel_zero(m, n) / bessel_zero(*VIBRATION_MODE)


def triangle_phase(t_s):
    """The reference piezo's phase shift at a moment, for the triangle toggle.

    Args: t_s is the time on the sensor clock in seconds.
    Returns: K times the mirror's displacement: 0 at the start of each
        period, K * PIEZO_TRIANGLE_NM half a period later, linear between.
    """
    cycles = t_s * PIEZO_TRIANGLE_HZ
    return K_PER_NM * PIEZO_TRIANGLE_NM * 2 * abs(cycles - math.floor(cycles + 0.5))


def reference_ramp_average(t0, duration):
    """Average exp(-i triangle_phase(t)) over an exposure, exactly.

    The triangle is linear between its turning points, and each linear
    piece integrates in closed form (a sinc), so an exposure that crosses a
    turning point, or several, is exact too.

    Args: t0 is the exposure's start (s); duration is its length (s).
    Returns: The complex average; exp(-i phase(t0)) for a zero duration.
    """
    if duration <= 0:
        return complex(np.exp(-1j * triangle_phase(t0)))
    return complex(sum(length * np.exp(-1j * middle) * np.sinc(rate * length / (2 * math.pi))
                       for _, length, middle, rate in ramp_pieces(t0, duration)))


def ramp_pieces(t0, duration):
    """Split the triangle into its linear pieces over one exposure.

    Args: t0 is the exposure's start (s); duration is its length (s), > 0.
    Returns: A tuple of (start, length, phase at the middle, rate) per
        piece, start and length as fractions of the exposure and the rate in
        radians per exposure (FringeTable's pieces). A single piece is
        exactly (0.0, 1.0, ...), so its table can be reused frame to frame.
    """
    half = 0.5 / PIEZO_TRIANGLE_HZ
    slope = K_PER_NM * PIEZO_TRIANGLE_NM / half
    pieces, start, end = [], t0, t0 + duration
    while start < end:
        # The nudge keeps a start that rounding left a hair below a turning
        # point in the next piece (it misplaces at most 1e-9 of a half period).
        piece = math.floor(start / half + 1e-9)
        stop = min((piece + 1) * half, end)
        rate = slope if piece % 2 == 0 else -slope
        middle = triangle_phase(start) + rate * (stop - start) / 2
        pieces.append([(start - t0) / duration, (stop - start) / duration, middle,
                       rate * duration])
        start = stop
    if len(pieces) == 1:
        pieces[0][:2] = 0.0, 1.0
    return tuple(tuple(piece) for piece in pieces)


def object_note(name, drive, piezo=False):
    """Say in a line what the simulation is doing, for the note under the control.

    Args: name is a SIMULATED_OBJECTS name; drive is the DAQ's Drive; piezo
        is True while the reference piezo's triangle wave is on.
    Returns: The text. A held piezo level is named in every object; the
        piezo's sine only in Follow DAQ, which it is locked to.
    """
    text = _object_text(name, drive)
    if piezo and name == "Vibration":  # the white nodal line is a Previous-view result
        text = text.replace("Set the reference on Still first", "Use Previous view")
    if piezo:
        text = (f"{text[:-1]}; reference piezo: {PIEZO_TRIANGLE_HZ:g} Hz triangle, "
                f"{PIEZO_TRIANGLE_NM / 1000:g} µm peak to peak (decorrelation).")
    return text


def _object_text(name, drive):
    """Describe the object and the DAQ drive, for object_note.

    Args: name is a SIMULATED_OBJECTS name; drive is the DAQ's Drive.
    Returns: One sentence ending in a full stop.
    """
    if name == "Still":
        parts = ["Nothing moves"]
    elif name == "Slow deformation":
        parts = [f"The drumhead is pressed in to {DEFORMATION_FRINGES} fringes and "
                 f"released every {CYCLE_S:g} s, resting for the last {STILL_S:g} s"]
    elif name == "Vibration":
        m, n = VIBRATION_MODE
        parts = [f"Built-in drive: drum mode ({m}, {n}) at {mode_frequency_hz(m, n):.0f} Hz, "
                 "peak 16 rad (5 J0 fringes). Set the reference on Still first"]
    elif not drive.active:
        return ("Follow DAQ: no DAQ drive — open Devices → NI DAQ… and choose the "
                "Simulated device.")
    else:
        if drive.waveform == "sine":
            generator = f"{drive.frequency_hz:.6g} Hz sine, {drive.amplitude_v:.2f} V"
        elif drive.waveform == "square":
            generator = f"{drive.frequency_hz:.6g} Hz square (TTL)"
        else:
            generator = "generator off"
        parts = [f"{drive.device}: {generator}"]
    if drive.piezo_mode == SINE and name == "Follow DAQ":
        parts.append(f"piezo sine {drive.piezo_amplitude_v:.2f} V at "
                     f"{drive.piezo_phase_deg:.0f}°, centre {drive.piezo_v:.2f} V")
    elif drive.piezo_mode == TRIANGLE:
        low = drive.piezo_v - drive.piezo_amplitude_v
        high = drive.piezo_v + drive.piezo_amplitude_v
        parts.append(f"piezo triangle {low:.2f}–{high:.2f} V at "
                     f"{drive.piezo_frequency_hz:.3g} Hz (not simulated: the image keeps "
                     f"its centre, {drive.piezo_v:.2f} V)")
    elif drive.piezo_mode:
        shift = math.degrees(K_PER_NM * PIEZO_NM_PER_V * drive.piezo_v)
        parts.append(f"piezo holds {drive.piezo_v:.2f} V ({shift:.0f}° reference shift)")
    return "; ".join(parts) + "."


class DemoSource:
    """An ideal camera looking at a simulated speckle interferometer.

    The object beam is a fully developed speckle field under a Gaussian laser
    spot, the reference beam is smooth and uniform (as bright as the object's
    mean at the spot centre), and the drumhead centred in the frame moves
    according to the chosen object (SIMULATED_OBJECTS); the mount around it
    stays still.

    Attributes: width and height are the frame size; object is the chosen
        SIMULATED_OBJECTS name; background and modulation are B = I_o + I_r
        and M = 2 sqrt(I_o I_r) in grey levels at REFERENCE_EXPOSURE_US and
        0 dB; speckle_phase is the object speckle's phase psi; speckle_intensity
        is |E|^2 (mean 1); drum_mask marks the drumhead; load_shape is the
        dent the drive pad presses (peak 1, 0 off the drum); clock_s is the
        start time of the next frame on the sensor's own clock.
    """

    def __init__(self, width=960, height=640, label=None, log=None):
        """Build the speckle field, beams and drumhead.

        Args: width and height are the frame size in pixels; label (a camera
            serial such as "DEMO1") seeds the speckle, so each demo camera
            has its own, the same in every run; log is an AppLog-like object
            for warnings, or None.
        Returns: None.
        """
        self.width = width
        self.height = height
        self.log = log
        self.object = DEFAULT_OBJECT
        self.drive = Drive()
        self.piezo_on = False  # the reference piezo's triangle wave
        self.clock_s = 0.0
        self._cycle_start = 0.0  # when Slow deformation was chosen (its cycle's 0)
        self._harmonic = None   # (key, frequency, Z, u map, arg Z map)
        self._tables = collections.OrderedDict()  # key -> [FringeTable, prepared, fixed F]
        self._warned = False
        rng = np.random.default_rng(zlib.crc32((label or "DEMO").encode()))
        noise = rng.standard_normal((height, width)) + 1j * rng.standard_normal((height, width))
        fy = np.fft.fftfreq(height)[:, None]
        fx = np.fft.fftfreq(width)[None, :]
        pupil = fx * fx + fy * fy <= PUPIL_CUTOFF_PER_PX ** 2
        field = np.fft.ifft2(np.fft.fft2(noise) * pupil)
        intensity = np.abs(field) ** 2
        intensity /= intensity.mean()
        self.speckle_intensity = intensity.astype(np.float32)
        self.speckle_phase = np.angle(field).astype(np.float32)

        y = np.arange(height)[:, None] + 0.5 - height / 2
        x = np.arange(width)[None, :] + 0.5 - width / 2
        radius = DRUM_FRACTION / 2 * min(width, height)
        rho = np.hypot(x, y) / radius
        self.drum_mask = rho < 1
        self._inside = np.flatnonzero(self.drum_mask)
        position = rho.ravel()[self._inside] * (RADIAL_SAMPLES - 1)
        self._radial_index = position.astype(np.int64)
        self._radial_weight = position - self._radial_index
        angle = np.arctan2(-y, x) - math.radians(DRIVE_ANGLE_DEG)
        self._cos_angle = np.cos(np.broadcast_to(angle, rho.shape).ravel()[self._inside])
        self.modes = drum_modes()
        grid = np.linspace(0.0, 1.0, RADIAL_SAMPLES)
        self._profiles = np.array([bessel_j(m, j * grid) for m, _, j in self.modes])
        load = self.mode_sum(self._drive_weights().real)
        self.load_shape = (load / load.max()).astype(np.float32)

        spot = np.exp(math.log(SPOT_RIM_LEVEL) * rho * rho)
        level = SPOT_LEVEL / 2  # object mean and reference are 1 each at the centre
        objects = intensity * spot
        self.background = (level * (objects + 1.0)).astype(np.float32)
        self.modulation = (level * 2 * np.sqrt(objects)).astype(np.float32)

    def mode_sum(self, coefficients):
        """Evaluate a sum of drum modes over the frame.

        Each order m gathers its modes' radial profiles into one, which is
        read at every pixel's radius and multiplied by cos(m theta), built
        by the Chebyshev recurrence; only pixels on the drum are computed.

        Args: coefficients is an array aligned with self.modes, real or complex.
        Returns: A (height, width) array, 0 off the drum.
        """
        coefficients = np.asarray(coefficients)
        orders = np.array([m for m, _, _ in self.modes])
        total = np.zeros(self._inside.size, np.result_type(coefficients, np.float64))
        lower, upper = np.ones_like(self._cos_angle), self._cos_angle
        for m in range(orders.max() + 1):
            angular = lower if m == 0 else upper
            if m >= 2:
                lower, upper = upper, 2 * self._cos_angle * upper - lower
                angular = upper
            chosen = orders == m
            if not np.any(coefficients[chosen]):
                continue
            profile = coefficients[chosen] @ self._profiles[chosen]
            near = profile[self._radial_index]
            far = profile[self._radial_index + 1]
            total += (near + (far - near) * self._radial_weight) * angular
        result = np.zeros(self.width * self.height, total.dtype)
        result[self._inside] = total
        return result.reshape(self.height, self.width)

    def _drive_weights(self, frequency=0.0):
        """Each mode's response to the drive pad, as mode_sum coefficients.

        A mode's static share is its shape at the pad over its norm and
        stiffness (j_mn^2), tapered for the pad's size; at a frequency it is
        also a resonator of quality MODE_Q. At 0 Hz the sum is the pad's dent.

        Args: frequency is the drive frequency in Hz.
        Returns: A complex array aligned with self.modes.
        """
        weights = []
        for m, n, j in self.modes:
            norm = (2 * math.pi if m == 0 else math.pi) * bessel_j(m + 1, j) ** 2 / 2
            share = bessel_j(m, j * DRIVE_RADIUS) / norm / j ** 2
            ratio = frequency / mode_frequency_hz(m, n)
            weights.append(share * math.exp(-(j / DRIVE_PAD_J) ** 2)
                           / (1 - ratio ** 2 + 1j * ratio / MODE_Q))
        return np.array(weights)

    def set_object(self, name):
        """Choose what the simulated drumhead does.

        Args: name is one of SIMULATED_OBJECTS.
        Returns: None. Raises ValueError for any other name. Choosing Slow
            deformation (from another object) starts its cycle at rest.
        """
        if name not in SIMULATED_OBJECTS:
            raise ValueError(f"Unknown simulated object: {name!r}")
        if name == "Slow deformation" and self.object != name:
            self._cycle_start = self.clock_s  # start the dent's cycle at rest
        self.object = name

    def set_piezo(self, on):
        """Turn the reference piezo's triangle wave on or off.

        Args: on is True to ramp the reference mirror (decorrelation).
        Returns: None.
        """
        self.piezo_on = bool(on)

    def set_drive(self, drive):
        """Follow what the DAQ outputs produce.

        Args: drive is a daq.Drive (Drive() when nothing is driven).
        Returns: None.
        """
        self.drive = drive

    def static_piezo_nm(self):
        """The reference mirror's steady displacement.

        Args: None.
        Returns: PIEZO_NM_PER_V times the piezo's held level (or its sine's
            centre), in nm; 0 when the piezo is off.
        """
        return PIEZO_NM_PER_V * self.drive.piezo_v if self.drive.piezo_mode else 0.0

    def harmonic_path_nm(self):
        """The vibration's complex path-difference amplitude, object minus mirror.

        With Z this map, the harmonic phase is K * Im(Z exp(i omega t)).

        Args: None.
        Returns: (frequency_hz, Z) with Z a complex64 (height, width) map in
            nm, or (0.0, None) when nothing vibrates.
        """
        state = self._harmonic_state()
        return (state[1], state[2]) if state[2] is not None else (0.0, None)

    def _harmonic_state(self):
        """Work out (and cache) the vibration for the current object and drive.

        Args: None.
        Returns: (key, frequency, Z, K|Z| as float32, arg Z as float32); Z
            and the maps are None when nothing vibrates.
        """
        key = (self.object, self.drive if self.object == "Follow DAQ" else None)
        if self._harmonic is not None and self._harmonic[0] == key:
            return self._harmonic
        frequency, path = 0.0, None
        if self.object == "Vibration":
            m, n = VIBRATION_MODE
            shape = self.mode_sum([1.0 if (mode[0], mode[1]) == (m, n) else 0.0
                                   for mode in self.modes])
            frequency = mode_frequency_hz(m, n)
            path = shape * (VIBRATION_PEAK_RAD / K_PER_NM / np.abs(shape).max())
        elif self.object == "Follow DAQ" and self.drive.waveform:
            frequency = self.drive.frequency_hz
            volts = (self.drive.amplitude_v if self.drive.waveform == "sine"
                     else 4 / math.pi * TTL_HIGH_V / 2)
            path = volts * self._drum_response(frequency)
            if self.drive.piezo_mode == SINE:
                path = path - PIEZO_NM_PER_V * self.drive.piezo_amplitude_v * np.exp(
                    1j * math.radians(self.drive.piezo_phase_deg))
        if path is None:
            self._harmonic = (key, 0.0, None, None, None)
        else:
            path = path.astype(np.complex64)
            self._harmonic = (key, frequency, path,
                              (K_PER_NM * np.abs(path)).astype(np.float32),
                              np.angle(path).astype(np.float32))
        self._tables.clear()
        return self._harmonic

    def _drum_response(self, frequency):
        """The drumhead's displacement per volt of a sine drive at one frequency.

        The scale makes GENERATOR_REFERENCE_V at the fundamental, f(0, 1),
        peak at GENERATOR_PEAK_RAD.

        Args: frequency is the drive frequency in Hz.
        Returns: A complex (height, width) map in nm per volt.
        """
        fundamental = mode_frequency_hz(0, 1)
        reference = np.abs(self.mode_sum(self._drive_weights(fundamental))).max()
        scale = GENERATOR_PEAK_RAD / K_PER_NM / (reference * GENERATOR_REFERENCE_V)
        return self.mode_sum(self._drive_weights(frequency)) * scale

    def _fringe_map(self, state, exposure_us, t0):
        """The exposure average F at every pixel for this frame.

        With the reference piezo's triangle on, F also averages the ramp,
        exactly (the triangle's pieces go into the table's weights); its
        phase at the exposure's middle is a factor outside the table, so the
        two ramp directions each keep one table frame to frame.

        Args: state is _harmonic_state(); exposure_us is the exposure; t0 is
            the exposure's start on the sensor clock.
        Returns: A complex64 (height, width) map.
        """
        _, frequency, _, amplitude, angle = state
        duration = exposure_us * 1e-6
        cycles = frequency * duration
        needed = float(amplitude.max())
        if needed > TABLE_U_LIMIT and not self._warned and self.log is not None:
            self.log.warning(
                f"Simulated vibration reaches {needed:.0f} rad; above {TABLE_U_LIMIT:.0f} "
                "rad its fringes are finer than the speckle, so those pixels are "
                f"simulated at {TABLE_U_LIMIT:.0f} rad.")
            self._warned = True
        pieces, factor = None, None
        if self.piezo_on and duration > 0:
            pieces = ramp_pieces(t0, duration)
            reference = pieces[0][2]
            factor = np.complex64(np.exp(-1j * reference))
            pieces = tuple((start, length, middle - reference, rate)
                           for start, length, middle, rate in pieces)
        key = (cycles, min(needed, TABLE_U_LIMIT), pieces)
        entry = self._tables.get(key)
        if entry is None:
            table = FringeTable(key[1], cycles, pieces)
            entry = self._tables[key] = [table, table.prepare(amplitude), None]
            while len(self._tables) > TABLE_CACHE_SIZE:
                self._tables.popitem(last=False)
        self._tables.move_to_end(key)
        table, prepared, fixed = entry
        if not table.depends_on_beta:  # whole cycles, no ramp: the same every frame
            if fixed is None:
                fixed = entry[2] = table.lookup(prepared, angle)
            return fixed
        shift = np.float32(2 * math.pi * ((frequency * t0) % 1.0))
        fringe = table.lookup(prepared, angle + shift)
        return fringe if factor is None else fringe * factor

    def next_frame(self, exposure_us, gain_db, fps):
        """Expose one frame starting at clock_s, then advance the clock a frame period.

        Args: exposure_us and gain_db are the simulated camera's settings;
            fps is the requested frame rate (the sensor clock's rate).
        Returns: A new (height, width) uint8 frame.
        """
        static = -K_PER_NM * self.static_piezo_nm()
        phase = self.speckle_phase + np.float32(static) if static else self.speckle_phase
        if self.object == "Slow deformation":
            dent = deformation_nm(self.clock_s - self._cycle_start)
            phase = phase + np.float32(K_PER_NM * dent) * self.load_shape
        state = self._harmonic_state()
        if state[2] is None and not self.piezo_on:
            intensity = self.background + self.modulation * np.cos(phase)
        else:
            # A vibration's F includes the triangle exactly; without one the
            # triangle on the uniform reference beam is a single number.
            fringe = (self._fringe_map(state, exposure_us, self.clock_s)
                      if state[2] is not None else
                      np.complex64(reference_ramp_average(self.clock_s, exposure_us * 1e-6)))
            intensity = self.background + self.modulation * (
                np.real(fringe) * np.cos(phase) - np.imag(fringe) * np.sin(phase))
        scale = exposure_us / REFERENCE_EXPOSURE_US * 10.0 ** (gain_db / 20.0)
        self.clock_s += 1.0 / max(float(fps), 1.0)
        return cv2.convertScaleAbs(intensity, alpha=scale)
