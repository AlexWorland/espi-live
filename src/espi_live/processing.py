"""ESPI image processing shared by the engine and the Qt UI.

Nothing here touches a GUI toolkit or PySpin, so it is safe to call from any
thread and from camera-free tests.
"""

import collections
import functools
import math
import time
from collections import deque

import cv2
import numpy as np

VIEW_MODES = ("Raw", "Previous", "Reference")
SMOOTHING_METHODS = ("Gaussian", "Median")
# OpenCV's vectorized median handles widths up to 5; wider uint8 medians take
# ~30 ms per 6 MP frame, which would cut the camera's 30 fps.
MAX_MEDIAN_WIDTH = 5
# Camera-feed noise reduction keeps at most this share of the history per
# frame (about a 20-frame memory), so the live image never freezes.
MAX_DENOISE_STRENGTH = 0.95
# The slider runs from 0 (off) to this. The number is how far it is turned
# up, not the history weight: denoise_strength spreads the 20-frame memory
# across the whole travel, so the low end is not a dead zone.
MAX_DENOISE_PERCENT = 95
# Previous view can compare with a frame up to this many frames back; each
# remembered frame is a full copy (about 6 MB on the Blackfly).
MAX_FRAMES_BACK = 30
# Brightness scaling of the difference image: a fixed multiplier, as the
# LabVIEW ESPI 1.5 VI did (IMAQ Multiply by "Constant"), or a stretch of each
# frame's range to 0..255. Settings.scale_range is True for "Auto stretch".
SCALING_MODES = ("Fixed gain", "Auto stretch")
# Default fixed gain: the LabVIEW VI's "Constant" default (42.0, read from its
# front-panel default data). Differences of 7 grey levels or more reach white.
DEFAULT_DIFFERENCE_GAIN = 42.0
# Noise calibration: each frame's level at the top 0.5% of pixels is its noise
# peak, and the suggested sensitivity floor is the median peak times this.
NOISE_PERCENTILE_CLIP = 0.5
NOISE_MARGIN = 2.0
# Brightness correction: blur width (pixels) for the illumination estimate,
# the darkest local level still treated as lit, and the largest boost.
BEAM_SIGMA_PX = 24.0
DARK_LEVEL = 8.0
MAX_BRIGHTNESS_GAIN = 6.0
# Peak hold splits its window into this many slices; the window's edge is
# accurate to one slice (0.25 s for a 3 s window).
PEAK_SLICES = 12
# Noise floor: a still scene's difference never averages to zero, because an
# absolute difference of noise is always positive. A map of each pixel's
# still-scene level is measured for at least this long and over at least this
# many images, and each pixel's spread is added this many times, so the
# subtraction leaves a still scene nearly black while a change still shows.
FLOOR_MIN_S = 2.0
FLOOR_MIN_IMAGES = 16
FLOOR_SD_MARGIN = 1.0
# Deep frames: a source with more than 8 bits (the iPhone's RAW mode) hands the
# pipeline float32 frames in 8-bit-equivalent grey levels (0 is black, 255 the
# sensor's full scale, fractions kept). Every step up to the final scaling keeps
# that precision; only the image that is shown, saved or recorded becomes uint8.
DEEP_DTYPE = np.float32
FULL_SCALE = 255.0
DEEP_BINS = 1024  # histogram bins for a deep image's percentile levels


def is_deep(image):
    """Tell whether an image carries more than 8 bits (float32 grey levels).

    Args: image is a NumPy array.
    Returns: True for a deep frame, False for uint8.
    """
    return image.dtype != np.uint8


def to_uint8(image):
    """Round an image to the 8-bit picture that is shown, saved or recorded.

    Args: image is a 2-D uint8 or float32 array in 8-bit-equivalent grey
        levels; values outside 0..255 saturate.
    Returns: A uint8 array (the input itself when it already is one).
    """
    if image.dtype == np.uint8:
        return image
    return cv2.convertScaleAbs(np.maximum(image, 0.0, dtype=np.float32))


def saturate(values, like):
    """Limit a product to the top of the scale of the frames it came from.

    Args: values is a float32 array; like is an array whose dtype decides the
        result.
    Returns: For uint8 `like`, values rounded and saturated to uint8; for a
        deep one, values capped at FULL_SCALE and kept as float32.
    """
    if like.dtype == np.uint8:
        return cv2.convertScaleAbs(values)
    return np.minimum(values, FULL_SCALE, out=values)


def normalize_min_max(image):
    """Map a Mono8 image's minimum and maximum to 0 and 255.

    Args: image is a nonempty, single-channel uint8 NumPy array.
    Returns: A tuple of scaled image, original minimum, and original maximum.
        A constant image becomes zeros.
    """
    low, high, _, _ = cv2.minMaxLoc(image)
    low, high = int(low), int(high)
    if high == low:
        return np.zeros_like(image), low, high
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX,
                         dtype=cv2.CV_8U), low, high


def stretch_contrast(image, clip_percent):
    """Stretch a measurement image to 0..255, optionally ignoring outliers.

    Speckle leaves a few very bright pixels in a difference image; with an
    exact min/max stretch they set the top of the range and flatten the
    fringes. Clipping a small percentage at each end keeps the contrast.

    Args: image is a nonempty, single-channel uint8 array; clip_percent is
        the share of pixels (0..50) allowed to saturate at each end, where 0
        means the exact minimum and maximum.
    Returns: A tuple of scaled image, low and high input levels. A constant
        image (or clipped range) becomes zeros, as in normalize_min_max.
    """
    if clip_percent <= 0:
        return normalize_min_max(image)
    counts = np.cumsum(cv2.calcHist([image], [0], None, [256], [0, 256]).ravel())
    clipped = counts[-1] * min(clip_percent, 50.0) / 100.0
    low = int(np.searchsorted(counts, clipped, side="right"))
    high = int(np.searchsorted(counts, counts[-1] - clipped, side="left"))
    if high <= low:
        return np.zeros_like(image), low, high
    levels = (np.arange(256, dtype=np.float32) - low) * (255.0 / (high - low))
    return cv2.LUT(image, np.clip(np.rint(levels), 0, 255).astype(np.uint8)), low, high


def deep_levels(image, clip_percent):
    """Find a deep image's stretch levels, to a fraction of a grey level.

    A RAW difference is often well under one 8-bit grey level, so the levels
    are kept as floats and found with a fine histogram.

    Args: image is a nonempty float32 array; clip_percent is as in
        stretch_levels.
    Returns: A (low, high) tuple of float grey levels.
    """
    low, high, _, _ = cv2.minMaxLoc(image)
    if clip_percent <= 0 or high <= low:
        return float(low), float(high)
    width = (high - low) / DEEP_BINS
    counts = np.cumsum(cv2.calcHist([image], [0], None, [DEEP_BINS],
                                    [low, high + width]).ravel())
    clipped = counts[-1] * min(clip_percent, 50.0) / 100.0
    first = int(np.searchsorted(counts, clipped, side="right"))
    last = int(np.searchsorted(counts, counts[-1] - clipped, side="left"))
    return float(low + first * width), float(low + last * width)


def stretch_levels(image, clip_percent):
    """Find the input levels a contrast stretch maps to black and white.

    Args: image is a nonempty, single-channel uint8 or deep (float32) array;
        clip_percent is the share of pixels (0..50) allowed to saturate at
        each end, where 0 means the exact minimum and maximum.
    Returns: A (low, high) tuple: integer grey levels for uint8, floats for a
        deep image.
    """
    if is_deep(image):
        return deep_levels(image, clip_percent)
    if clip_percent <= 0:
        low, high, _, _ = cv2.minMaxLoc(image)
        return int(low), int(high)
    counts = np.cumsum(cv2.calcHist([image], [0], None, [256], [0, 256]).ravel())
    clipped = counts[-1] * min(clip_percent, 50.0) / 100.0
    low = int(np.searchsorted(counts, clipped, side="right"))
    high = int(np.searchsorted(counts, counts[-1] - clipped, side="left"))
    return low, high


def apply_levels(image, low, high):
    """Map grey level low to 0 and high to 255, saturating outside them.

    high may lie above 255: a sensitivity floor widens the range so that
    small differences stay dark instead of filling the scale.

    Args: image is a single-channel uint8 or deep (float32) array; low and
        high are levels.
    Returns: The stretched uint8 image, or zeros when high <= low. A deep
        image keeps its fractional levels until this rounding to 8 bits.
    """
    if high <= low:
        return np.zeros(image.shape, np.uint8)
    if is_deep(image):
        scale = 255.0 / (high - low)
        return to_uint8((image - np.float32(low)) * np.float32(scale))
    levels = (np.arange(256, dtype=np.float32) - low) * (255.0 / (high - low))
    return cv2.LUT(image, np.clip(np.rint(levels), 0, 255).astype(np.uint8))


class RangeTracker:
    """Keep the stretch steady: a sensitivity floor plus slow release.

    The floor (min_range) stops a quiet difference image, which holds only
    camera noise, from being stretched to full contrast. The release time
    (steady_s) lets the range shrink only gradually, so the contrast does not
    pump from frame to frame; a range that grows is followed at once, so new
    fringes show immediately.
    """

    def __init__(self):
        """Start with no remembered range.

        Args: None.
        Returns: None.
        """
        self.low = None
        self.high = None
        self.time = None

    def reset(self):
        """Forget the remembered range, e.g. after a view change.

        Args: None.
        Returns: None.
        """
        self.low = self.high = self.time = None

    def update(self, low, high, min_range, steady_s, now, fine=False):
        """Combine this frame's levels with the floor and the history.

        Args: low and high are this frame's stretch levels; min_range is the
            sensitivity floor in grey levels (0 = off); steady_s is the
            release time in seconds (0 = follow every frame); now is the
            monotonic time; fine keeps fractional levels (deep images).
        Returns: The (low, high) levels to stretch with: integers, or floats
            when fine.
        """
        low, high = float(low), float(high)
        if self.high is None or steady_s <= 0:
            self.low, self.high = low, high
        else:
            alpha = 1.0 - math.exp(-max(0.0, now - self.time) / steady_s)
            self.high = high if high > self.high else self.high + alpha * (high - self.high)
            self.low = low if low < self.low else self.low + alpha * (low - self.low)
        self.time = now
        if fine:
            low, high = self.low, self.high
            if min_range > 0:
                high = max(high, low + float(min_range))
            return low, high
        low, high = int(round(self.low)), int(round(self.high))
        if min_range > 0:
            high = max(high, low + int(min_range))
        return low, high


class NoiseCalibration:
    """Measure how large a still scene's difference images get.

    Run it with the object at rest: each processed difference frame adds its
    near-maximum level, and the median over the run is the noise peak. The
    suggested sensitivity floor is that peak times NOISE_MARGIN.
    """

    def __init__(self, now, duration_s=2.0):
        """Start a measurement.

        Args: now is the monotonic start time; duration_s is how long to sample.
        Returns: None.
        """
        self.until = now + duration_s
        self.peaks = []
        self.deep = False  # the frames are deep: a floor under one level is allowed

    def add(self, image):
        """Record one processed difference frame.

        Args: image is a uint8 or deep difference image (before the stretch).
        Returns: None.
        """
        self.deep = is_deep(image)
        self.peaks.append(stretch_levels(image, NOISE_PERCENTILE_CLIP)[1])

    def done(self, now):
        """Report whether the sampling time has passed.

        Args: now is the monotonic time.
        Returns: True once the measurement is over.
        """
        return now >= self.until

    def result(self):
        """Summarize the measurement.

        Args: None.
        Returns: (noise peak, suggested floor) in grey levels, or None when no
            frame was recorded.
        """
        if not self.peaks:
            return None
        noise = float(np.median(self.peaks))
        floor = math.ceil(noise * NOISE_MARGIN)
        if self.deep and floor <= 1 and noise * NOISE_MARGIN <= 1.0:
            return noise, 0  # a RAW still scene's noise is under one grey level
        return noise, int(min(255, max(1, floor)))


class PeakHold:
    """Show each pixel's largest difference within the last hold_s seconds.

    Previous view catches a vibration only in the frames where the motion
    happens to differ; holding the peak keeps those moments on screen. It is
    a sliding window: a value older than hold_s drops out, so the image is
    exactly "the peak in the last hold_s seconds", to within one slice.

    The window is split into PEAK_SLICES time slices, each holding the
    maximum of its own frames. A new frame updates only the current slice;
    the older slices are combined once, when a slice is closed, so each
    frame costs one pixel-wise maximum whatever the window length.
    """

    def __init__(self):
        """Start with nothing held.

        Args: None.
        Returns: None.
        """
        self.reset()

    def reset(self):
        """Drop everything held, e.g. after a view or baseline change.

        Args: None.
        Returns: None.
        """
        self.slices = collections.deque()  # (start time, uint8 max) closed slices
        self.current = None                # max of the slice being filled
        self.current_start = None
        self.closed = None                 # max over self.slices, or None
        self.window = None

    def update(self, image, hold_s, now):
        """Add a difference image and return the peak over the window.

        Args: image is a uint8 difference image; hold_s is the window length
            in seconds (0 = off); now is the monotonic time.
        Returns: A new uint8 image holding each pixel's maximum over the last
            hold_s seconds, or image itself when off.
        """
        if hold_s <= 0:
            self.reset()
            return image
        if (hold_s != self.window or self.current is None
                or self.current.shape != image.shape):
            self.reset()
            self.window = hold_s
            self.current, self.current_start = image.copy(), now
            return image.copy()
        span = hold_s / PEAK_SLICES
        if now - self.current_start >= span:
            self.slices.append((self.current_start, self.current))
            self.current, self.current_start = image.copy(), now
            while self.slices and self.slices[0][0] + span <= now - hold_s:
                self.slices.popleft()
            self.closed = (functools.reduce(np.maximum, (s for _, s in self.slices))
                           if self.slices else None)
        else:
            np.maximum(self.current, image, out=self.current)
        if self.closed is None:
            return self.current.copy()
        return np.maximum(self.closed, self.current)


def local_normalize(image, window, floor=0.0):
    """Even out fringe contrast by normalizing against the local statistics.

    Each pixel becomes (value - local mean) / local standard deviation, both
    measured over a window wider than the fringe spacing, so faint and strong
    fringes get the same contrast wherever they are (the fringe-normalization
    approach of the ESPI literature). The result is centred on mid-grey: 128
    is the local average, and +-2 standard deviations span black to white.
    The deviation is never taken below floor / 4, so a still scene, whose
    deviation is only camera noise, is not boosted into full contrast.

    Args: image is a uint8 difference image (already speckle-filtered);
        window is the window width in pixels; floor is the sensitivity floor
        in grey levels (0 = no limit beyond one grey level).
    Returns: The normalized uint8 image.
    """
    size = max(3, int(window) | 1)
    data = image.astype(np.float32)
    mean = cv2.blur(data, (size, size))
    spread = cv2.blur(data * data, (size, size)) - mean * mean
    std = np.sqrt(np.maximum(spread, 0.0))
    np.maximum(std, max(1e-3 if is_deep(image) else 1.0, floor / 4.0), out=std)
    out = 128.0 + (data - mean) * (63.75 / std)
    return np.clip(out, 0, 255).astype(np.uint8)


def even_out_brightness(difference, frame, sigma=BEAM_SIGMA_PX):
    """Divide out the illumination profile so fringes read evenly.

    A difference grows with the local brightness, so a Gaussian laser spot
    makes the centre look busiest. Scaling each pixel by the average lit
    brightness over its local brightness gives the same fringe contrast
    across the spot. Pixels darker than DARK_LEVEL are set to 0, and the
    boost is capped at MAX_BRIGHTNESS_GAIN so the beam's edge is not
    amplified into noise.

    Args: difference is a uint8 or deep difference image; frame is the camera
        frame it came from (same shape); sigma is the blur width in pixels
        used to estimate the illumination.
    Returns: An image of the same shape and type as difference.
    """
    height, width = frame.shape
    shrink = max(1, int(sigma // 4))
    small = cv2.resize(frame, (max(1, width // shrink), max(1, height // shrink)),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    small = cv2.GaussianBlur(small, (0, 0), max(0.5, sigma / shrink))
    local = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    lit = local >= DARK_LEVEL
    if not lit.any():
        return np.zeros_like(difference)
    target = float(local[lit].mean())
    gain = np.where(lit, target / np.maximum(local, DARK_LEVEL), 0.0)
    gain = np.minimum(gain, MAX_BRIGHTNESS_GAIN).astype(np.float32)
    boosted = np.clip(difference.astype(np.float32) * gain, 0, 255)
    return boosted.astype(np.uint8) if difference.dtype == np.uint8 else boosted


def smooth_speckle(image, method, size):
    """Blur away pixel-scale speckle so broad fringes stand out.

    Args: image is a 2-D uint8 array; method is a SMOOTHING_METHODS name;
        size is the kernel width in pixels (made odd), with 0 or 1 meaning off;
        Median is capped at MAX_MEDIAN_WIDTH to stay within a frame interval.
    Returns: The smoothed uint8 image, or the input itself when off.
    """
    size = int(size)
    if size <= 1:
        return image
    size |= 1  # OpenCV kernels must have odd widths
    if method == "Median":
        return cv2.medianBlur(image, min(size, MAX_MEDIAN_WIDTH))
    return cv2.GaussianBlur(image, (size, size), 0)


def apply_gain(image, gain):
    """Multiply a difference image by a fixed gain, saturating at 255.

    This is the LabVIEW ESPI 1.5 VI's IMAQ Multiply by "Constant": a still
    scene's few levels of noise stay dark while real changes reach white, and
    the brightness does not follow each frame's range as a stretch does.

    Args: image is a 2-D uint8 or deep array; gain is the multiplier (1.0 =
        none).
    Returns: The saturated product (rounded for uint8, fractions kept for a
        deep image), or the input itself at 1.0.
    """
    if gain == 1.0:
        return image
    if is_deep(image):
        return saturate(image * np.float32(gain), image)
    return cv2.convertScaleAbs(image, alpha=float(gain))


def reject_outliers(image, size):
    """Replace pixels that stand out from their neighbourhood by its mean.

    The rule of NI Vision's Lowpass filter (IMAQ LowPass, used by the LabVIEW
    ESPI 1.5 VI): with M the mean and S the standard deviation of a pixel and
    its size x size neighbourhood, a pixel further than S from M becomes M and
    every other pixel is kept. Unlike a blur it leaves smooth areas and
    fringes alone and only pulls isolated spikes back (IMAQ Vision Concepts
    Manual, "Lowpass Filter").

    Args: image is a 2-D uint8 or deep array; size is the neighbourhood width
        in pixels (made odd), with 0 or 1 meaning off.
    Returns: A new image of the same type, or the input itself when off.
    """
    size = int(size)
    if size <= 1:
        return image
    size |= 1
    values = image.astype(np.float32)
    mean = cv2.blur(values, (size, size), borderType=cv2.BORDER_REFLECT)
    mean_square = cv2.blur(values * values, (size, size), borderType=cv2.BORDER_REFLECT)
    spread = np.sqrt(np.maximum(mean_square - mean * mean, 0.0))
    outliers = np.abs(values - mean) > spread
    result = image.copy()
    replacement = mean[outliers]
    result[outliers] = (np.rint(replacement).astype(np.uint8)
                        if image.dtype == np.uint8 else replacement)
    return result


def single_field(frame):
    """Show one field of an interlaced frame at full height.

    The two fields of an analog frame are exposed 1/60 s apart; keeping only
    the even rows removes that offset, and repeating each row keeps the
    frame's size and aspect, so display, saving and recording stay alike.

    Args: frame is a 2-D image.
    Returns: A new array of the same shape built from the even rows.
    """
    return np.repeat(frame[0::2], 2, axis=0)[:frame.shape[0]]


def raw_difference(frame, previous):
    """Subtract the previous raw camera frame from the live one, and nothing else.

    No noise reduction, field selection, gain, averaging, stretch,
    gamma or display tone: each output pixel is |live - previous| in the
    camera's own grey levels.

    Args: frame is the live 2-D uint8 frame; previous is the frame before
        it, or None.
    Returns: A new uint8 array, or None when there is no previous frame of
        the same size yet.
    """
    if previous is None or previous.shape != frame.shape:
        return None
    return cv2.absdiff(frame, previous)


def denoise_strength(percent):
    """Map the Noise reduction slider onto a history weight.

    Reading the slider as a history weight (20 meaning "keep 20% of the
    previous frame") leaves almost the whole travel invisible: 20 is only
    about one frame of memory, so a still picture barely changes until the
    slider is near the top. The slider instead means how far the control is
    turned up. 0 is off, and each step adds the same amount of memory, from
    one frame up to the 20-frame cap of MAX_DENOISE_STRENGTH. 10 is about
    three frames. Speckle is unchanged either way; only frame-to-frame
    flicker and motion trails change.

    Args: percent is the slider value. 0 and below are off; values at or
        above MAX_DENOISE_PERCENT stay at the cap.
    Returns: The history weight for ESPIProcessor.denoise, from 0 to
        MAX_DENOISE_STRENGTH.
    """
    if percent <= 0:
        return 0.0
    if percent >= MAX_DENOISE_PERCENT:
        return MAX_DENOISE_STRENGTH
    fraction = float(percent) / MAX_DENOISE_PERCENT
    tau_max = 1.0 / (1.0 - MAX_DENOISE_STRENGTH)
    tau = 1.0 + fraction * (tau_max - 1.0)
    return 1.0 - 1.0 / tau


def gamma_lut(gamma):
    """Build the 256-entry gamma lookup table used for display.

    Args: gamma is the display gamma; 1.0 means no change.
    Returns: A uint8 lookup table, or None when gamma is 1.0.
    """
    if gamma == 1.0:
        return None
    return np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
                    dtype=np.uint8)


def display_lut(gain=1.0, contrast=0.0, brightness=0):
    """Build one lookup table for gain, S-curve contrast and brightness.

    Applied in that order: gain multiplies (black stays black, bright values
    saturate at 255); the S-curve darkens values below mid-grey and brightens
    those above it, keeping 0 and 255 fixed; brightness then adds a constant.

    Args: gain is the multiplier (1 = off); contrast is the S-curve strength
        in percent, 0..100 (0 = off); brightness is the grey levels added
        (0 = off).
    Returns: A 256-entry uint8 table, or None when every control is off.
    """
    if gain == 1.0 and contrast <= 0 and brightness <= 0:
        return None
    x = np.clip(np.arange(256, dtype=np.float64) * gain / 255.0, 0.0, 1.0)
    steep = 12.0 * min(float(contrast), 100.0) / 100.0
    if steep > 0:
        def sigmoid(v):
            """Logistic function.

            Args: v is a number or array.
            Returns: 1 / (1 + exp(-v)).
            """
            return 1.0 / (1.0 + np.exp(-v))
        low, high = sigmoid(-steep / 2), sigmoid(steep / 2)
        x = (sigmoid(steep * (x - 0.5)) - low) / (high - low)
    return np.clip(np.rint(x * 255.0 + brightness), 0, 255).astype(np.uint8)


def apply_display(image, gamma):
    """Apply the display gamma to a uint8 measurement image.

    This is the single definition of display tone: the Qt viewer builds its
    GPU lookup table by passing 0..255 through it, and saved frames and
    recordings call it directly, so the screen and the files always match.
    The display is always grayscale.

    Args: image is a 2-D uint8 array; gamma is the display gamma.
    Returns: The 2-D uint8 gray image (the input itself when gamma is 1).
    """
    lut = gamma_lut(gamma)
    if lut is not None:
        image = cv2.LUT(image, lut)
    return image


def display_rgb_table(gamma):
    """Describe the display gamma as a 256-entry gray RGB table for the GPU.

    Args: gamma is the display gamma.
    Returns: A (256, 3) uint8 RGB array; entry i is the gray of value i.
    """
    ramp = np.arange(256, dtype=np.uint8).reshape(1, 256)
    return np.repeat(apply_display(ramp, gamma).reshape(256, 1), 3, axis=1)


# The Reference frames control runs from 1 (one frame, as before) to this.
MAX_REFERENCE_FRAMES = 60


class NoiseFloorMap:
    """Measure each pixel's still-scene difference level, to subtract it later.

    Run it with the object at rest: every processed difference image adds
    to a per-pixel sum and sum of squares. The floor is each pixel's mean
    plus FLOOR_SD_MARGIN of its spread. It is per pixel because noise grows
    with brightness: the middle of a laser spot is noisier than its edge.
    """

    def __init__(self, now, duration_s=FLOOR_MIN_S, min_images=FLOOR_MIN_IMAGES):
        """Start a measurement.

        Args: now is the monotonic start time; duration_s is the shortest
            sampling time; min_images is the fewest images to sample.
        Returns: None.
        """
        self.until = now + duration_s
        self.min_images = int(min_images)
        self.count = 0
        self.total = None
        self.squares = None

    def add(self, image):
        """Record one processed difference image.

        An image of a different size than the ones before starts the sums
        over, so images of two sizes are never mixed.

        Args: image is a uint8 difference image (before the floor is subtracted).
        Returns: None.
        """
        if self.total is None or self.total.shape != image.shape:
            self.total = image.astype(np.float32)
            self.squares = self.total * self.total
            self.count = 1
            return
        cv2.accumulate(image, self.total)
        cv2.accumulateSquare(image, self.squares)
        self.count += 1

    def done(self, now):
        """Report whether enough time and images have been sampled.

        Args: now is the monotonic time.
        Returns: True once the measurement is over.
        """
        return self.count >= self.min_images and now >= self.until

    def result(self):
        """Build the floor map.

        Args: None.
        Returns: A float32 array the size of the images, each pixel's mean
            plus FLOOR_SD_MARGIN standard deviations, or None when no image
            was recorded.
        """
        if self.total is None:
            return None
        mean = self.total / self.count
        spread = np.sqrt(np.maximum(self.squares / self.count - mean * mean, 0.0))
        return mean + np.float32(FLOOR_SD_MARGIN) * spread


def subtract_floor(image, floor):
    """Take a measured noise floor off a difference image.

    Args: image is a uint8 or deep difference image; floor is a float32 map of
        the same shape (NoiseFloorMap.result).
    Returns: A new image, image minus floor, never below 0 (rounded to uint8,
        or float32 for a deep image).
    """
    out = image.astype(np.float32)
    out -= floor
    np.maximum(out, 0.0, out=out)
    return out if is_deep(image) else cv2.convertScaleAbs(out)


class ReferenceCapture:
    """Average several camera frames into one reference baseline.

    The sum is kept as float32, so the mean is exact before it is rounded
    to the uint8 reference the difference step uses.
    """

    def __init__(self, target, noise_reduction, started, manual):
        """Start an empty capture.

        Args: target is how many frames to average (2 or more);
            noise_reduction is the slider value the frames are taken at;
            started is the monotonic time the capture began (the reference's
            age is counted from it); manual is True for F / Set reference
            and False for an automatic refresh.
        Returns: None.
        """
        self.target = int(target)
        self.noise_reduction = noise_reduction
        self.started = started
        self.manual = manual
        self.count = 0
        self.accum = None

    def add(self, frame):
        """Add one frame to the sum.

        A frame of a different size than the ones before it starts the sum
        over, so frames of two sizes are never mixed.

        Args: frame is a Mono8 (or deep) array taken after single-field and
            noise reduction.
        Returns: The mean once target frames are summed (rounded to uint8, or
            float32 for deep frames), otherwise None.
        """
        if self.accum is None or self.accum.shape != frame.shape:
            self.accum = frame.astype(np.float32)
            self.count = 1
        else:
            cv2.accumulate(frame, self.accum)
            self.count += 1
        if self.count < self.target:
            return None
        if is_deep(frame):
            return self.accum / np.float32(self.count)
        return cv2.convertScaleAbs(self.accum, alpha=1.0 / self.count)


class ESPIProcessor:
    """Keep the previous/reference frame and temporal averaging state."""

    def __init__(self):
        """Initialize empty ESPI frame state.

        Args: None.
        Returns: None.
        """
        self.history = deque(maxlen=1)  # earlier frames for Previous view
        self.previous = None           # the earlier frame the last difference() used
        self.ref = None
        self.ref_time = None
        self.ref_frames = 0            # frames averaged into ref
        self.ref_noise = None          # noise reduction ref was taken at
        self.ref_capture = None        # ReferenceCapture in progress, or None
        self.avg_accum = None
        self.avg_count = 0
        self.avg_target = None
        self.rolling = None
        self.denoised = None
        self.frame_accum = None        # camera frames summed before the difference
        self.frame_count = 0
        self.frame_target = None

    def reset_avg(self):
        """Discard any partial temporal average.

        Args: None.
        Returns: None.
        """
        self.avg_accum = None
        self.avg_count = 0
        self.avg_target = None
        self.rolling = None

    def reset_frame_average(self):
        """Discard camera frames summed toward a pre-difference average.

        Args: None.
        Returns: None.
        """
        self.frame_accum = None
        self.frame_count = 0
        self.frame_target = None

    def average_frames(self, frame, count):
        """Block-average camera frames before they are subtracted.

        The mean of count frames has 1/sqrt(count) of a single frame's noise,
        and consecutive block means are independent, so a difference of two
        of them is quieter by the same factor while a steady change (time-
        averaged fringes, a static deformation) is kept. Averaging absolute
        differences instead can never fall below the noise floor.

        Args: frame is the Mono8 frame after single field and noise
            reduction; count is the number of frames per average, with one
            disabling it.
        Returns: The rounded uint8 mean when count frames are summed, the
            frame itself when count is 1, otherwise None. A frame of a new
            size or a changed count starts the sum over.
        """
        if count <= 1:
            self.reset_frame_average()
            return frame
        if (self.frame_accum is None or self.frame_target != count or
                self.frame_accum.shape != frame.shape):
            self.frame_accum = frame.astype(np.float32)
            self.frame_count = 1
            self.frame_target = count
        else:
            cv2.accumulate(frame, self.frame_accum)
            self.frame_count += 1
        if self.frame_count < count:
            return None
        if is_deep(frame):
            result = self.frame_accum / np.float32(self.frame_count)
        else:
            result = cv2.convertScaleAbs(self.frame_accum,
                                         alpha=1.0 / self.frame_count)
        self.reset_frame_average()
        return result

    def denoise(self, frame, strength):
        """Reduce sensor noise in the camera feed by blending frames over time.

        A running average keeps the static speckle that ESPI relies on and
        averages out random shot and read noise; a spatial blur would erase
        the speckle. Moving objects leave a trail that grows with strength.

        Args: frame is the current Mono8 (or deep) array; strength is the
            share (0..1) of the running history kept in each output frame,
            with 0 meaning off. Values are capped at MAX_DENOISE_STRENGTH so
            the feed still follows changes.
        Returns: The filtered frame (rounded for uint8, fractions kept for a
            deep frame), or the input itself when off or on the first frame
            of a new size.
        """
        if strength <= 0:
            self.denoised = None
            return frame
        if self.denoised is None or self.denoised.shape != frame.shape:
            self.denoised = frame.astype(np.float32)
            return frame
        weight = 1.0 - min(strength, MAX_DENOISE_STRENGTH)
        cv2.accumulateWeighted(frame, self.denoised, weight)
        if is_deep(frame):
            return self.denoised.copy()  # the sum keeps changing; hand out a copy
        return cv2.convertScaleAbs(self.denoised)

    def reset_previous(self):
        """Forget the earlier frames that Previous view subtracts.

        Args: None.
        Returns: None.
        """
        self.history.clear()
        self.previous = None

    def set_reference(self, frame, now, noise_reduction=0.0, frames=1):
        """Store a copy of a frame as the Reference baseline.

        Args: frame is the Mono8 array to keep; now is the monotonic time,
            used to schedule automatic refreshes; noise_reduction is the
            slider value it was taken at; frames is how many camera frames
            were averaged into it.
        Returns: None.
        """
        self.ref = frame.copy()
        self.ref_time = now
        self.ref_noise = noise_reduction
        self.ref_frames = frames

    def start_reference(self, count, noise_reduction, now, manual=True):
        """Begin averaging the next frames into a new reference.

        The stored reference stays in use until the new one is complete.

        Args: count is the number of frames to average (2 or more);
            noise_reduction is the current slider value; now is the
            monotonic time; manual is False for an automatic refresh.
        Returns: None; a capture already running is replaced.
        """
        self.ref_capture = ReferenceCapture(count, noise_reduction, now, manual)

    def add_reference_frame(self, frame):
        """Feed one frame to the capture in progress.

        Args: frame is the Mono8 array, after single-field and noise reduction.
        Returns: The finished ReferenceCapture when this frame completed it
            (the reference is then stored, its age counted from the start of
            the capture), otherwise None. Does nothing without a capture.
        """
        capture = self.ref_capture
        if capture is None:
            return None
        mean = capture.add(frame)
        if mean is None:
            return None
        self.ref_capture = None
        self.ref = mean
        self.ref_time = capture.started
        self.ref_noise = capture.noise_reduction
        self.ref_frames = capture.count
        return capture

    def clear_reference(self):
        """Forget the stored reference and any capture in progress.

        Args: None.
        Returns: None.
        """
        self.ref = self.ref_time = self.ref_noise = self.ref_capture = None
        self.ref_frames = 0

    def difference(self, frame, mode, frames_back=1):
        """Subtract an earlier frame or the captured reference.

        Args: frame is the current Mono8 array; mode is Previous or Reference;
            frames_back is how many frames back Previous view looks (1 is the
            frame just before; capped at MAX_FRAMES_BACK). Comparing further
            back shows slower change, as the LabVIEW VI did by waiting between
            grabs.
        Returns: A uint8 absolute-difference image, or zeros until a baseline
            frame exists. For Previous view, self.previous afterwards holds
            the earlier frame that was subtracted (None while there was
            none), which the comparison window shows as its Previous panel.
        """
        self.previous = None
        if mode == "Previous":
            back = min(max(1, int(frames_back)), MAX_FRAMES_BACK)
            if self.history and (self.history[-1].shape != frame.shape
                                 or self.history[-1].dtype != frame.dtype):
                self.history.clear()
            if self.history.maxlen != back:
                self.history = deque(self.history, maxlen=back)  # keeps the newest
            if len(self.history) < back:
                self.history.append(frame.copy())
                return np.zeros_like(frame)
            self.previous = self.history[0]  # the deque drops it on the append
            result = cv2.absdiff(frame, self.previous)
            self.history.append(frame.copy())
            return result
        if (self.ref is None or self.ref.shape != frame.shape
                or self.ref.dtype != frame.dtype):
            return np.zeros_like(frame)
        return cv2.absdiff(frame, self.ref)

    def average(self, image, count, scale=None):
        """Accumulate a requested number of difference frames.

        Args: image is a uint8 or deep difference frame; count is the number of
            frames per displayed average, with one disabling averaging; scale,
            when given, multiplies the mean, which is then rounded and
            saturated (the LabVIEW VI's sum x Multiplier / N to U8).
        Returns: An average of the same type when ready (a deep frame keeps
            its fractions), or None while accumulating.
            Without scale the mean is truncated toward zero, as in earlier
            versions. With count 1 the image is returned unscaled.
        """
        if count <= 1:
            self.reset_avg()
            return image
        if (self.avg_accum is None or self.avg_target != count or
                self.avg_accum.shape != image.shape):
            self.avg_accum = image.astype(np.float32)
            self.avg_count = 1
            self.avg_target = count
        else:
            # cv2.accumulate is an exact float32 add, ~5x faster than numpy +=.
            cv2.accumulate(image, self.avg_accum)
            self.avg_count += 1
        if self.avg_count < count:
            return None
        deep = is_deep(image)
        if scale is not None:
            if deep:
                result = saturate(self.avg_accum * np.float32(scale / self.avg_count),
                                  image)
            else:
                result = cv2.convertScaleAbs(self.avg_accum,
                                             alpha=scale / self.avg_count)
            self.reset_avg()
            return result
        # The mean of uint8 values already lies in 0..255, so no clip is needed;
        # dividing in place (not multiplying by 1/n) keeps results bit-identical.
        np.divide(self.avg_accum, self.avg_count, out=self.avg_accum)
        result = self.avg_accum.copy() if deep else self.avg_accum.astype(np.uint8)
        self.reset_avg()
        return result

    def rolling_average(self, image, count, scale=None):
        """Blend each difference frame into a running average.

        Unlike average(), this returns a frame every time, so the display
        keeps the camera's rate; older frames fade with weight 1/count.

        Args: image is a uint8 or deep difference frame; count sets the
            effective number of frames remembered, with one disabling
            averaging; scale, when given, multiplies the average before
            rounding and saturation.
        Returns: The running average (rounded uint8, or float32 for a deep
            frame). With count 1 the image is returned unscaled.
        """
        if count <= 1:
            self.reset_avg()
            return image
        if self.rolling is None or self.rolling.shape != image.shape:
            self.rolling = image.astype(np.float32)
        else:
            cv2.accumulateWeighted(image, self.rolling, 1.0 / count)
        if is_deep(image):
            return saturate(self.rolling * np.float32(1.0 if scale is None else scale),
                            image)
        return cv2.convertScaleAbs(self.rolling, alpha=1.0 if scale is None else scale)


class DemoSource:
    """Generate moving Mono8 test frames without a camera or SDK."""

    def __init__(self, width=960, height=640, label=None):
        """Build a repeatable test pattern with a broad brightness range.

        Args: width and height are the synthetic frame dimensions in pixels;
            label is optional text that replaces the pattern's caption.
        Returns: None.
        """
        self.width = width
        self.height = height
        x = np.linspace(16, 215, width, dtype=np.float32)
        gradient = np.broadcast_to(x, (height, width)).copy()
        checker = (((np.arange(height)[:, None] // 40 +
                     np.arange(width)[None, :] // 40) % 2) * 28)
        self.base = np.clip(gradient + checker, 0, 255).astype(np.uint8)
        cv2.putText(self.base, label or "ESPI DEMO PATTERN", (30, height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.15, 255, 2, cv2.LINE_AA)
        self.start_time = time.monotonic()

    def next_frame(self, exposure_us, gain_db, now):
        """Make one moving grayscale frame using the simulated camera sliders.

        Args: exposure_us and gain_db are control values; now is monotonic time.
        Returns: A new two-dimensional uint8 frame, independent of the base.
        """
        frame = self.base.copy()
        phase = now - self.start_time
        center_x = int(self.width * (0.5 + 0.32 * math.sin(phase * 1.3)))
        center_y = int(self.height * (0.48 + 0.23 * math.cos(phase * 0.9)))
        cv2.circle(frame, (center_x, center_y), 65, 255, -1, cv2.LINE_AA)
        cv2.circle(frame, (center_x, center_y), 30, 0, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (30, 90), (105, 165), 0, -1)
        brightness = max(0.1, min(2.5, exposure_us / 8000.0))
        brightness *= max(1.0, min(4.0, 10.0 ** (gain_db / 20.0)))
        return cv2.convertScaleAbs(frame, alpha=brightness)


def watermark_demo(image):
    """Permanently mark a synthetic preview or export as dummy data.

    Args: image is a grayscale or BGR display frame.
    Returns: A marked copy of the frame, leaving its source unchanged.
    """
    marked = image.copy()
    height, width = marked.shape[:2]
    banner_height = min(height, max(42, height // 10))
    background = 0 if marked.ndim == 2 else (0, 0, 80)
    foreground = 255 if marked.ndim == 2 else (0, 255, 255)
    cv2.rectangle(marked, (0, 0), (width - 1, banner_height - 1), background, -1)
    scale = min(1.0, max(0.45, width / 900.0))
    cv2.putText(marked, "DUMMY MODE ACTIVE", (15, max(30, banner_height - 13)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, foreground, 2, cv2.LINE_AA)
    return marked
