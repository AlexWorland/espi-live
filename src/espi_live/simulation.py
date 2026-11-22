"""A simulated ESPI signal for dummy mode: what an ideal camera would record.

The scene is a speckle interferometer: a rough plate scatters laser light
into a speckle field, a smooth reference beam interferes with it, and the
plate's out-of-plane motion w shifts the optical phase by K * w (light
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

import math
import zlib

import cv2
import numpy as np

# Optics.
WAVELENGTH_NM = 632.8  # HeNe
K_PER_NM = 4 * math.pi / WAVELENGTH_NM  # phase per nm of out-of-plane motion

# Specimen: a simply supported rectangular plate on the centred part of the
# frame, its modes (m, n) up to MAX_MODE each way. Frequencies use the fixed
# physical aspect PLATE_ASPECT whatever the frame size, so f(2, 1) is
# exactly 1000 Hz.
PLATE_FRACTION = 0.88
PLATE_ASPECT = 1.5
MAX_MODE = 15
F11_HZ = 520.0
MODE_Q = 50.0
DRIVE_POINT = (0.37, 0.41)  # plate coordinates, off every low mode's nodal lines

# Slow deformation: a raised-cosine ramp to DEFORMATION_FRINGES fringes over
# RAMP_S, a HOLD_S hold, the ramp back down, then STILL_S at rest.
DEFORMATION_FRINGES = 8
RAMP_S = 5.0
HOLD_S = 1.0
STILL_S = 1.0
CYCLE_S = 2 * RAMP_S + HOLD_S + STILL_S

# Vibration: the built-in drive excites one mode at its resonance.
VIBRATION_MODE = (2, 1)
VIBRATION_PEAK_RAD = 16.0  # peak K * amplitude: about 5 dark J0 fringes
# Follow DAQ: a 0.1 V sine at f(1, 1) gives this peak phase amplitude.
GENERATOR_PEAK_RAD = 16.0
GENERATOR_REFERENCE_V = 0.1
TTL_HIGH_V = 5.0  # a counter's square wave; only its fundamental is simulated
PIEZO_NM_PER_V = 1500.0  # 100 nm/V stack behind a gain-15 amplifier

# Ideal sensor: the spot centre averages SPOT_LEVEL grey levels at the
# default exposure and gain; brightness is proportional to both.
REFERENCE_EXPOSURE_US = 8000.0
SPOT_LEVEL = 64.0
# The imaging pupil passes spatial frequencies up to this (cycles per pixel),
# so the speckle intensity (twice that) stays within the sensor's Nyquist
# limit and each pixel's point sample is exact. Grains look 2-3 px wide.
PUPIL_CUTOFF_PER_PX = 0.25
# The Gaussian laser spot falls to this share of its peak at the plate's corners.
SPOT_CORNER_LEVEL = 0.25

SIMULATED_OBJECTS = ("Still", "Slow deformation", "Vibration", "Follow DAQ")
DEFAULT_OBJECT = "Slow deformation"

# FringeTable accuracy: the largest interpolation error allowed along each
# axis (the two add), and the amplitude step of the table.
TABLE_U_STEP = 0.08
TABLE_BETA_ERROR = 1e-3
MAX_TABLE_POINTS = 1 << 15


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

    def __init__(self, u_max, cycles):
        """Build the table.

        Args: u_max is the largest phase amplitude (radians) to cover;
            cycles is the number of drive cycles per exposure (n >= 0).
        Returns: None.
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
        spectrum *= np.exp(1j * np.pi * k * cycles) * np.sinc(k * cycles) / coarse
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


def _plate_axis(size):
    """Place the plate along one frame axis.

    Args: size is the frame's width or height in pixels.
    Returns: (coordinate, inside): each pixel centre's plate coordinate
        (0..1 across the plate) and whether it lies on the plate.
    """
    start = size * (1 - PLATE_FRACTION) / 2
    coordinate = (np.arange(size) + 0.5 - start) / (size * PLATE_FRACTION)
    return coordinate, (coordinate > 0) & (coordinate < 1)


def _sine_basis(coordinate, inside):
    """Tabulate sin(m pi s) for m = 1..MAX_MODE along one axis, 0 off the plate.

    Args: coordinate and inside are what _plate_axis returned.
    Returns: A float64 array of shape (MAX_MODE, size).
    """
    modes = np.arange(1, MAX_MODE + 1)[:, None]
    return np.sin(np.pi * modes * coordinate[None, :]) * inside[None, :]


def _mode_stiffness():
    """Each mode's (m / aspect)^2 + n^2, the plate's eigenvalue up to a constant.

    Args: None.
    Returns: A float64 array of shape (MAX_MODE n, MAX_MODE m).
    """
    m = np.arange(1, MAX_MODE + 1)[None, :]
    n = np.arange(1, MAX_MODE + 1)[:, None]
    return (m / PLATE_ASPECT) ** 2 + n ** 2


def _drive_participation():
    """Each mode's shape at the drive point, S_mn(DRIVE_POINT).

    Args: None.
    Returns: A float64 array of shape (MAX_MODE n, MAX_MODE m).
    """
    modes = np.arange(1, MAX_MODE + 1)
    along_x = np.sin(np.pi * modes * DRIVE_POINT[0])
    along_y = np.sin(np.pi * modes * DRIVE_POINT[1])
    return along_y[:, None] * along_x[None, :]


class DemoSource:
    """An ideal camera looking at a simulated speckle interferometer.

    The object beam is a fully developed speckle field under a Gaussian laser
    spot, the reference beam is smooth and uniform (as bright as the object's
    mean at the spot centre), and the plate on the centred PLATE_FRACTION of
    the frame moves according to the chosen object (SIMULATED_OBJECTS).

    Attributes: width and height are the frame size; object is the chosen
        SIMULATED_OBJECTS name; background and modulation are B = I_o + I_r
        and M = 2 sqrt(I_o I_r) in grey levels at REFERENCE_EXPOSURE_US and
        0 dB; speckle_phase is the object speckle's phase psi; speckle_intensity
        is |E|^2 (mean 1); plate_mask marks the plate; load_shape is the
        point-load deflection (peak 1, 0 off the plate); clock_s is the start
        time of the next frame on the sensor's own clock.
    """

    def __init__(self, width=960, height=640, label=None, log=None):
        """Build the speckle field, beams and plate.

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
        self.clock_s = 0.0
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

        x_plate, x_inside = _plate_axis(width)
        y_plate, y_inside = _plate_axis(height)
        self.plate_mask = y_inside[:, None] & x_inside[None, :]
        self.basis_x = _sine_basis(x_plate, x_inside)
        self.basis_y = _sine_basis(y_plate, y_inside)
        load = self.mode_sum(_drive_participation() / _mode_stiffness() ** 2)
        self.load_shape = (load / load.max()).astype(np.float32)

        y = np.arange(height)[:, None] + 0.5 - height / 2
        x = np.arange(width)[None, :] + 0.5 - width / 2
        corner = (PLATE_FRACTION / 2) ** 2 * (width ** 2 + height ** 2)
        spot = np.exp(math.log(SPOT_CORNER_LEVEL) * (x * x + y * y) / corner)
        level = SPOT_LEVEL / 2  # object mean and reference are 1 each at the centre
        objects = intensity * spot
        self.background = (level * (objects + 1.0)).astype(np.float32)
        self.modulation = (level * 2 * np.sqrt(objects)).astype(np.float32)

    def mode_sum(self, coefficients):
        """Evaluate a sum of plate modes over the frame.

        Args: coefficients is a (MAX_MODE, MAX_MODE) array, real or complex,
            indexed [n - 1, m - 1] for mode (m, n).
        Returns: A (height, width) array, 0 off the plate.
        """
        return self.basis_y.T @ coefficients @ self.basis_x

    def set_object(self, name):
        """Choose what the simulated plate does.

        Args: name is one of SIMULATED_OBJECTS.
        Returns: None. Raises ValueError for any other name.
        """
        if name not in SIMULATED_OBJECTS:
            raise ValueError(f"Unknown simulated object: {name!r}")
        self.object = name

    def next_frame(self, exposure_us, gain_db, fps):
        """Expose one frame starting at clock_s, then advance the clock a frame period.

        Args: exposure_us and gain_db are the simulated camera's settings;
            fps is the requested frame rate (the sensor clock's rate).
        Returns: A new (height, width) uint8 frame.
        """
        phase = self.speckle_phase
        if self.object == "Slow deformation":
            phase = phase + np.float32(K_PER_NM * deformation_nm(self.clock_s)) * self.load_shape
        intensity = self.background + self.modulation * np.cos(phase)
        scale = exposure_us / REFERENCE_EXPOSURE_US * 10.0 ** (gain_db / 20.0)
        self.clock_s += 1.0 / max(float(fps), 1.0)
        return cv2.convertScaleAbs(intensity, alpha=scale)
