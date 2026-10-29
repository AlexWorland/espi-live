"""ESPI image processing shared by the engine, the Qt UI and the Tk UI.

Nothing here touches a GUI toolkit or PySpin, so it is safe to call from any
thread and from camera-free tests.
"""

import math
import time

import cv2
import numpy as np

COLOR_MAPS = {
    "Turbo": cv2.COLORMAP_TURBO,
    "Jet": cv2.COLORMAP_JET,
    "Hot": cv2.COLORMAP_HOT,
    "Viridis": cv2.COLORMAP_VIRIDIS,
    "Plasma": cv2.COLORMAP_PLASMA,
    "Inferno": cv2.COLORMAP_INFERNO,
}
PALETTES = ("Gray", *COLOR_MAPS)
VIEW_MODES = ("Raw", "Previous", "Reference")
SMOOTHING_METHODS = ("Gaussian", "Median")
# OpenCV's vectorized median handles widths up to 5; wider uint8 medians take
# ~30 ms per 6 MP frame, which would cut the camera's 30 fps.
MAX_MEDIAN_WIDTH = 5
# Camera-feed noise reduction keeps at most this share of the history per
# frame (about a 20-frame memory), so the live image never freezes.
MAX_DENOISE_STRENGTH = 0.95


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


def gamma_lut(gamma):
    """Build the 256-entry gamma lookup table used for display.

    Args: gamma is the display gamma; 1.0 means no change.
    Returns: A uint8 lookup table, or None when gamma is 1.0.
    """
    if gamma == 1.0:
        return None
    return np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
                    dtype=np.uint8)


def apply_palette(image, palette, gamma):
    """Apply gamma and a named palette to a uint8 measurement image.

    This is the single definition of display color: the Qt viewer builds its
    GPU lookup table by passing 0..255 through it, and saved frames and
    recordings call it directly, so the screen and the files always match.

    Args: image is a 2-D uint8 array; palette is "Gray" or a COLOR_MAPS key;
        gamma is the display gamma.
    Returns: The 2-D gray image for "Gray", otherwise a BGR uint8 image.
    """
    lut = gamma_lut(gamma)
    if lut is not None:
        image = cv2.LUT(image, lut)
    if palette == "Gray":
        return image
    return cv2.applyColorMap(image, COLOR_MAPS[palette])


def palette_rgb_table(palette, gamma):
    """Describe gamma plus palette as a 256-entry RGB table for the GPU.

    Args: palette is "Gray" or a COLOR_MAPS key; gamma is the display gamma.
    Returns: A (256, 3) uint8 RGB array; entry i is the color of value i.
    """
    ramp = np.arange(256, dtype=np.uint8).reshape(1, 256)
    colored = apply_palette(ramp, palette, gamma)
    if colored.ndim == 2:
        return np.repeat(colored.reshape(256, 1), 3, axis=1)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).reshape(256, 3)


class ESPIProcessor:
    """Keep the previous/reference frame and temporal averaging state."""

    def __init__(self):
        """Initialize empty ESPI frame state.

        Args: None.
        Returns: None.
        """
        self.prev = None
        self.ref = None
        self.ref_time = None
        self.avg_accum = None
        self.avg_count = 0
        self.avg_target = None
        self.rolling = None
        self.denoised = None

    def reset_avg(self):
        """Discard any partial temporal average.

        Args: None.
        Returns: None.
        """
        self.avg_accum = None
        self.avg_count = 0
        self.avg_target = None
        self.rolling = None

    def denoise(self, frame, strength):
        """Reduce sensor noise in the camera feed by blending frames over time.

        A running average keeps the static speckle that ESPI relies on and
        averages out random shot and read noise; a spatial blur would erase
        the speckle. Moving objects leave a trail that grows with strength.

        Args: frame is the current Mono8 array; strength is the share (0..1)
            of the running history kept in each output frame, with 0 meaning
            off. Values are capped at MAX_DENOISE_STRENGTH so the feed still
            follows changes.
        Returns: The rounded uint8 filtered frame, or the input itself when off
            or on the first frame of a new size.
        """
        if strength <= 0:
            self.denoised = None
            return frame
        if self.denoised is None or self.denoised.shape != frame.shape:
            self.denoised = frame.astype(np.float32)
            return frame
        weight = 1.0 - min(strength, MAX_DENOISE_STRENGTH)
        cv2.accumulateWeighted(frame, self.denoised, weight)
        return cv2.convertScaleAbs(self.denoised)

    def set_reference(self, frame, now):
        """Store a copy of a frame as the Reference baseline.

        Args: frame is the Mono8 array to keep; now is the monotonic time,
            used to schedule automatic refreshes.
        Returns: None.
        """
        self.ref = frame.copy()
        self.ref_time = now

    def difference(self, frame, mode):
        """Subtract the previous frame or the captured reference.

        Args: frame is the current Mono8 array; mode is Previous or Reference.
        Returns: A uint8 absolute-difference image, or zeros until a baseline
            frame exists.
        """
        if mode == "Previous":
            if self.prev is None or self.prev.shape != frame.shape:
                self.prev = frame.copy()
                return np.zeros_like(frame)
            result = cv2.absdiff(frame, self.prev)
            self.prev = frame.copy()
            return result
        if self.ref is None or self.ref.shape != frame.shape:
            return np.zeros_like(frame)
        return cv2.absdiff(frame, self.ref)

    def average(self, image, count):
        """Accumulate a requested number of difference frames.

        Args: image is a uint8 difference frame; count is the number of frames
            per displayed average, with one disabling averaging.
        Returns: A uint8 average when ready, or None while accumulating. The
            mean is truncated toward zero, as in earlier versions.
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
        # The mean of uint8 values already lies in 0..255, so no clip is needed;
        # dividing in place (not multiplying by 1/n) keeps results bit-identical.
        np.divide(self.avg_accum, self.avg_count, out=self.avg_accum)
        result = self.avg_accum.astype(np.uint8)
        self.reset_avg()
        return result

    def rolling_average(self, image, count):
        """Blend each difference frame into a running average.

        Unlike average(), this returns a frame every time, so the display
        keeps the camera's rate; older frames fade with weight 1/count.

        Args: image is a uint8 difference frame; count sets the effective
            number of frames remembered, with one disabling averaging.
        Returns: The rounded uint8 running average.
        """
        if count <= 1:
            self.reset_avg()
            return image
        if self.rolling is None or self.rolling.shape != image.shape:
            self.rolling = image.astype(np.float32)
        else:
            cv2.accumulateWeighted(image, self.rolling, 1.0 / count)
        return cv2.convertScaleAbs(self.rolling)


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
