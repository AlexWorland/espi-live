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
SPECKLE_DIAMETER_PX = 3.0

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
