"""Phase-shifting ESPI: piezo phase steps, the wrapped phase and its filter.

Subtraction ESPI shows |live - reference|, whose fringes are broken into
speckle and lose the sign of the change. Phase-shifting ESPI instead moves
the reference mirror's piezo through known phase steps, grabs one frame at
each, and computes every pixel's speckle phase. The difference between a
live phase map and a reference one is the deformation's phase, wrapped to
(-pi, pi], which is shown as a sawtooth from black to white (no change is
mid-grey).

- 4-step: steps of pi/2. Simple, but a miscalibrated step shows as ripple.
- 5-step (Hariharan): steps of pi/2 and insensitive to moderate step error;
  the same five frames also measure the step actually made, which
  Calibrate phase step uses to set the piezo voltage per step.

The phase difference is filtered as sine and cosine (here the real and
imaginary parts of z * conj(z_ref)), so the filter never smears across the
jump from pi to -pi, and pixels with strong speckle modulation count most.

This module is GUI-free and camera-free. PiezoStepper moves the piezo through
a daq.DaqController (whose worker thread owns the device); the engine's
worker thread calls it, waits for each move, then discards the frames
exposed while the piezo moved.
"""

import math
import time
from dataclasses import replace

import cv2
import numpy as np

from espi_live import daq

PHASE_STEPS = (4, 5)          # the algorithms offered: frames per phase map
DEFAULT_PHASE_STEPS = 5       # Hariharan: tolerant of an uncalibrated piezo
NOMINAL_STEP = math.pi / 2    # the phase step every algorithm here expects
DEFAULT_PHASE_FILTER = 9      # sine/cosine filter width in pixels
MAX_PHASE_FILTER = 31
DEFAULT_SETTLE_FRAMES = 2     # frames discarded after each piezo move
MAX_SETTLE_FRAMES = 10
RETURN_EXTRA_SETTLE = 1       # the jump back to step 0 is the largest move
MOVE_TIMEOUT_S = 2.0          # longest wait for the DAQ to apply one level
CALIBRATION_SETS = 5          # 5-step sets whose step estimates are combined
MIN_STEP_RAD = math.radians(20)   # outside this range an estimate is not used
MAX_STEP_RAD = math.radians(160)
# Dummy mode's simulated piezo: phase per volt at the DAQ. The DAQ window's
# default 0.05 V step then gives 72 degrees, so Calibrate phase step has
# something to correct (to 0.0625 V for 90 degrees).
DEMO_RAD_PER_VOLT = 8 * math.pi
DEMO_DEFORMATION_RATE = 0.25  # rad/s of the dummy deformation's slow cycle


PiezoMoveError = daq.PiezoMoveError  # raised by PiezoStepper.move


def _planes(frames):
    """Convert phase-step frames to float32 for arithmetic.

    Args: frames is a sequence of equally shaped 2-D arrays.
    Returns: A list of float32 arrays.
    """
    return [np.asarray(frame, dtype=np.float32) for frame in frames]


def wrapped_phase(frames):
    """Compute each pixel's phase from 4 or 5 phase-stepped frames.

    With I_k = a + b cos(phi + k * pi/2), the 4-step result is
    (I0 - I2) + i (I3 - I1) = 2b e^(i phi); Hariharan's 5-step result is
    (2 I2 - I4 - I0) + i 2 (I1 - I3) = 4b e^(i phi'), phi' being the phase at
    the middle frame. Both keep the modulation b as the magnitude, so a later
    filter can weight pixels by how strongly their speckle interferes.

    Args: frames is a list of 4 or 5 equally shaped 2-D arrays (uint8 or
        float32), in step order.
    Returns: A complex64 array; its angle is the wrapped phase. Raises
        ValueError for another number of frames.
    """
    planes = _planes(frames)
    if len(planes) == 4:
        real = planes[0] - planes[2]
        imaginary = planes[3] - planes[1]
    elif len(planes) == 5:
        real = 2.0 * planes[2] - planes[4] - planes[0]
        imaginary = 2.0 * (planes[1] - planes[3])
    else:
        raise ValueError(f"phase stepping needs 4 or 5 frames, not {len(planes)}")
    return (real + 1j * imaginary).astype(np.complex64)


def step_angle(frames):
    """Measure the phase step actually made, from 5 phase-stepped frames.

    For equal steps alpha, I4 - I0 = 2 cos(alpha) (I3 - I1) at every pixel
    (Hariharan, Oreb and Eiju 1987). A least-squares fit over all pixels gives
    cos(alpha), weighting pixels by how strongly they are modulated.

    Args: frames is a list of 5 equally shaped 2-D arrays in step order.
    Returns: The step in radians (0..pi), or None for another number of
        frames or when nothing in the image is modulated.
    """
    if len(frames) != 5:
        return None
    planes = _planes(frames)
    inner = planes[3] - planes[1]
    outer = planes[4] - planes[0]
    power = float(np.dot(inner.ravel(), inner.ravel()))
    if power <= 1e-6 * inner.size:
        return None
    cosine = float(np.dot(outer.ravel(), inner.ravel())) / (2.0 * power)
    return math.acos(min(1.0, max(-1.0, cosine)))


def calibrated_step(step_v, angle):
    """Scale a piezo step voltage so that it gives a quarter-wave step.

    The piezo moves about linearly over one small step, so the voltage
    scales by pi/2 over the measured angle.

    Args: step_v is the voltage per step used to measure; angle is the
        measured step in radians.
    Returns: The new voltage per step, or None when the angle is outside
        MIN_STEP_RAD..MAX_STEP_RAD (too far off to trust a linear scale).
    """
    if angle is None or not MIN_STEP_RAD <= angle <= MAX_STEP_RAD:
        return None
    return step_v * NOMINAL_STEP / angle


def phase_difference(phase, reference):
    """Subtract a reference phase map, keeping both maps' modulation.

    Args: phase and reference are complex arrays from wrapped_phase().
    Returns: phase * conj(reference): its angle is the wrapped phase change
        and its magnitude the product of the two modulations.
    """
    return phase * np.conj(reference)


def filter_phase(difference, size):
    """Smooth a phase difference as its sine and cosine (real and imaginary).

    Filtering the two parts instead of the angle never blurs across the
    jump from pi to -pi, and because the parts still carry the modulation,
    strongly interfering speckles count more than dark or saturated ones.

    Args: difference is a complex array from phase_difference(); size is the
        filter width in pixels (made odd), 0 or 1 meaning off.
    Returns: The filtered complex array (the input itself when off).
    """
    size = int(size)
    if size <= 1:
        return difference
    size |= 1
    sigma = size / 3.0
    real = cv2.GaussianBlur(np.ascontiguousarray(difference.real, dtype=np.float32),
                            (size, size), sigma)
    imaginary = cv2.GaussianBlur(np.ascontiguousarray(difference.imag, dtype=np.float32),
                                 (size, size), sigma)
    return (real + 1j * imaginary).astype(np.complex64)


def phase_image(difference):
    """Show a wrapped phase as grey levels.

    No change is mid-grey, so a still area stays one calm grey instead of
    flickering between black and white around 0; -pi is black and +pi white,
    so each fringe is one ramp and the wrap sits at +-pi.

    Args: difference is a complex array whose angle is the phase.
    Returns: A uint8 image: (phase + pi) * 255 / (2 pi), rounded.
    """
    angle = cv2.phase(np.ascontiguousarray(difference.real, dtype=np.float32),
                      np.ascontiguousarray(difference.imag, dtype=np.float32))
    # cv2.phase gives 0..2 pi; adding pi modulo 2 pi puts phase 0 in the middle.
    shifted = np.where(angle < math.pi, angle + np.float32(math.pi),
                       angle - np.float32(math.pi))
    return cv2.convertScaleAbs(shifted, alpha=255.0 / (2.0 * math.pi))


class PhaseSequence:
    """Collect one frame (or the mean of several) at each phase step.

    The caller moves the piezo whenever move_pending is True, to step, then
    calls moved(); the next settle frames are discarded because they were
    exposed while the piezo moved.
    """

    def __init__(self, steps, frames_per_step=1, settle_frames=DEFAULT_SETTLE_FRAMES):
        """Start a sequence that first needs the piezo at step 0.

        Args: steps is 4 or 5; frames_per_step is how many camera frames are
            averaged at each step (at least 1); settle_frames is how many
            frames to discard after each move.
        Returns: None.
        """
        self.steps = int(steps)
        self.frames_per_step = max(1, int(frames_per_step))
        self.settle_frames = max(0, int(settle_frames))
        self.step = 0
        self.move_pending = True
        self.skip = 0
        self.frames = []
        self._sum = None
        self._count = 0

    def moved(self):
        """Note that the piezo has reached self.step.

        Args: None.
        Returns: None; the next frames up to the settle count are discarded,
            one more after the return to step 0.
        """
        self.move_pending = False
        self.skip = self.settle_frames + (RETURN_EXTRA_SETTLE if self.step == 0 else 0)

    def restart(self):
        """Drop a partial set and start again at step 0.

        Args: None.
        Returns: None.
        """
        self.step = 0
        self.move_pending = True
        self.skip = 0
        self.frames = []
        self._sum = None
        self._count = 0

    def add(self, frame):
        """Use one camera frame.

        Args: frame is the newest 2-D frame (uint8 or float32).
        Returns: The list of step frames (float32 means, in step order) when
            this frame completed a set, otherwise None. A frame of another
            size than the set so far starts the set over.
        """
        if self.move_pending:
            return None
        if self.skip > 0:
            self.skip -= 1
            return None
        if ((self._sum is not None and self._sum.shape != frame.shape) or
                (self.frames and self.frames[0].shape != frame.shape)):
            self.restart()
            return None
        if self._sum is None:
            self._sum = frame.astype(np.float32)
            self._count = 1
        else:
            cv2.accumulate(frame, self._sum)
            self._count += 1
        if self._count < self.frames_per_step:
            return None
        self.frames.append(self._sum / np.float32(self._count))
        self._sum = None
        self._count = 0
        self.move_pending = True
        if len(self.frames) < self.steps:
            self.step += 1
            return None
        done, self.frames, self.step = self.frames, [], 0
        return done


class PhaseState:
    """The engine's phase-shifting state: sequence, reference and calibration."""

    def __init__(self):
        """Start with no sequence and no reference.

        Args: None.
        Returns: None.
        """
        self.sequence = None
        self.key = None
        self.reference = None      # complex phase map of the reference set
        self.reference_steps = None  # how many steps the reference set had
        self.reference_wanted = True
        self.angle = None          # the last measured step, radians
        self.calibration = None    # list of step estimates while calibrating

    def configure(self, steps, frames_per_step, settle_frames):
        """Make sure the sequence matches the settings, restarting it if not.

        Args: steps, frames_per_step and settle_frames as for PhaseSequence.
        Returns: The PhaseSequence in use.
        """
        key = (int(steps), max(1, int(frames_per_step)), max(0, int(settle_frames)))
        if key != self.key:
            self.sequence = PhaseSequence(*key)
            self.key = key
        return self.sequence

    def restart(self):
        """Forget the sequence (the next frame starts a new set at step 0).

        Args: None.
        Returns: None.
        """
        self.sequence = None
        self.key = None

    def clear_reference(self):
        """Make the next complete set the new reference.

        Args: None.
        Returns: None.
        """
        self.reference = None
        self.reference_wanted = True

    def start_calibration(self):
        """Begin collecting step estimates for Calibrate phase step.

        Args: None.
        Returns: None.
        """
        self.calibration = []

    def complete(self, frames, filter_size):
        """Turn a finished set into the displayed phase difference.

        Args: frames is the list PhaseSequence.add() returned; filter_size is
            the sine/cosine filter width in pixels.
        Returns: (image, new_reference): the uint8 phase image, and True when
            this set became the reference (the image is then flat grey). A
            set of another size or step count than the reference replaces it.
        """
        phase = wrapped_phase(frames)
        self.angle = step_angle(frames)
        if self.calibration is not None and self.angle is not None:
            self.calibration.append(self.angle)
        # 4-step phases are those of the first frame, 5-step ones of the middle
        # frame, so a reference taken with the other algorithm does not fit.
        new_reference = (self.reference_wanted or self.reference is None or
                         self.reference.shape != phase.shape or
                         self.reference_steps != len(frames))
        if new_reference:
            self.reference = phase
            self.reference_steps = len(frames)
            self.reference_wanted = False
        difference = filter_phase(phase_difference(phase, self.reference), filter_size)
        return phase_image(difference), new_reference

    def calibration_result(self):
        """Combine the step estimates once enough sets are measured.

        Args: None.
        Returns: The median step in radians when CALIBRATION_SETS estimates
            are in (calibration then ends), otherwise None.
        """
        if self.calibration is None or len(self.calibration) < CALIBRATION_SETS:
            return None
        angle = float(np.median(self.calibration))
        self.calibration = None
        return angle


class PiezoStepper:
    """Move the reference piezo to phase-step levels through the DAQ.

    settings are the DAQ window's DaqSettings: piezo_voltage_v is the level
    of step 0 and piezo_step_v the voltage per step (a quarter wave once
    calibrated). The generator settings are passed through unchanged, so a
    step never switches the generator on or off. Instances are replaced, not
    changed, so the engine thread never sees a half-updated stepper.
    """

    def __init__(self, controller, settings, simulated=False):
        """Wrap a connected DAQ controller.

        Args: controller is a daq.DaqController (or anything with
            move_piezo(settings, volts, timeout)); settings is daq.DaqSettings;
            simulated is True when the device drives nothing.
        Returns: None.
        """
        self.controller = controller
        self.settings = settings
        self.simulated = simulated
        self.volts = None  # the level last reached, None before the first move

    @property
    def step_v(self):
        """The voltage per phase step.

        Args: None.
        Returns: Volts at the DAQ.
        """
        return self.settings.piezo_step_v

    def level(self, index):
        """The DAQ voltage of one phase step.

        Args: index is the step number, 0 first.
        Returns: Volts.
        """
        return self.settings.piezo_voltage_v + index * self.settings.piezo_step_v

    def problems(self, steps):
        """Check that every step of a sequence can be driven.

        Args: steps is the number of phase steps.
        Returns: A list of sentences for the user; empty when it can run.
        """
        s = self.settings
        if s.piezo_output == daq.OFF:
            return ["Assign the piezo driver to an analog output in the NI DAQ window."]
        if s.piezo_step_v <= 0:
            return ["Set a piezo step voltage above 0 V in the NI DAQ window."]
        if self.level(0) < daq.PIEZO_FLOOR_V:
            return [f"Phase steps would start at {self.level(0):.4g} V, below 0 V: the "
                    f"piezo amplifier's input takes 0–{daq.MAX_VOLTS:g} V. Raise the "
                    "piezo voltage in the NI DAQ window."]
        peak = max(abs(self.level(0)), abs(self.level(steps - 1)))
        if peak > s.piezo_limit_v:
            return [f"Phase steps would reach {peak:.4g} V, above the piezo's "
                    f"{s.piezo_limit_v:g} V safety limit: lower the voltage or the step."]
        return []

    def move(self, index, timeout=MOVE_TIMEOUT_S):
        """Drive the piezo to a step and wait until the DAQ has applied it.

        Args: index is the step number; timeout is the longest wait in seconds.
        Returns: None. Raises PiezoMoveError when the DAQ refuses, fails or
            does not answer in time.
        """
        volts = self.level(index)
        self.controller.move_piezo(self.settings, volts, timeout)
        self.volts = volts

    def phase(self):
        """The phase the simulated piezo adds, for dummy mode's interferometer.

        Args: None.
        Returns: Radians (0 before the first move).
        """
        return 0.0 if self.volts is None else self.volts * DEMO_RAD_PER_VOLT

    def with_step(self, step_v):
        """Make a stepper like this one with another voltage per step.

        Args: step_v is the new voltage per step.
        Returns: A new PiezoStepper.
        """
        return PiezoStepper(self.controller, replace(self.settings, piezo_step_v=step_v),
                            self.simulated)


class DemoInterferometer:
    """Synthetic speckle interferograms for the Phase view in dummy mode.

    A fixed random speckle field interferes with a smooth reference beam;
    a dome-shaped deformation grows and relaxes over about 25 seconds (slowly,
    so it barely moves during one set of phase steps), so fringes appear in
    the phase difference; the simulated piezo adds its phase to every pixel.
    """

    def __init__(self, width=640, height=480, seed=7):
        """Build the speckle field and the deformation shape.

        Args: width and height are the frame size in pixels; seed makes the
            speckle repeatable.
        Returns: None.
        """
        self.rng = np.random.default_rng(seed)
        field = (self.rng.normal(size=(height, width)) +
                 1j * self.rng.normal(size=(height, width)))
        grain = cv2.getGaussianKernel(7, 1.2)
        real = cv2.sepFilter2D(field.real.astype(np.float32), -1, grain, grain)
        imaginary = cv2.sepFilter2D(field.imag.astype(np.float32), -1, grain, grain)
        intensity = real * real + imaginary * imaginary
        self.object = (intensity / intensity.mean() * 45.0).astype(np.float32)
        self.speckle = np.arctan2(imaginary, real).astype(np.float32)
        self.reference = np.float32(45.0)
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        radius = 0.3 * min(width, height)
        self.dome = np.exp(-((x - width / 2) ** 2 + (y - height / 2) ** 2)
                           / (2 * radius * radius)).astype(np.float32)
        self.start = time.monotonic()

    def deformation(self, now):
        """The deformation's peak phase at a time.

        Args: now is the monotonic time.
        Returns: Radians, swinging between 0 and 6 pi.
        """
        return 3.0 * math.pi * (1.0 - math.cos(DEMO_DEFORMATION_RATE * (now - self.start)))

    def frame(self, piezo_phase, now):
        """Make one interferogram.

        Args: piezo_phase is the phase the reference piezo adds, in radians;
            now is the monotonic time.
        Returns: A 2-D uint8 frame with a little camera noise.
        """
        phase = self.speckle + self.dome * np.float32(self.deformation(now)) \
            + np.float32(piezo_phase)
        image = (self.object + self.reference
                 + 2.0 * np.sqrt(self.object * self.reference) * np.cos(phase)
                 + self.rng.normal(0.0, 1.5, phase.shape).astype(np.float32))
        return cv2.convertScaleAbs(image)
