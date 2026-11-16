"""Frame-quality diagnostics for the acquisition loop.

Intermittent blur or lost detail for a few seconds can come from the camera
link (late or incomplete frames), from the sensor or optics, from the temporal
filters, or from the display. These monitors write the evidence to the log so
a short episode can be told apart afterwards:

- FrameGapMonitor: gaps between good frames and bursts of incomplete frames.
- StreamCounterMonitor: the transport layer's own lost/dropped frame counters.
- SharpnessMonitor: a cheap sharpness value, measured on the camera frame and
  again on the frame that is displayed, with the start and end of every drop.
- settings_summary: the settings in force, attached to every episode line.
- FrameCheckMonitor: a periodic INFO line describing the raw camera frame
  (levels and clipping, frame-to-frame change, repeated frames, speckle
  contrast and size, detail across against down, colour in a monochrome
  signal, field mismatch) and the difference range on screen. It tells weak
  or missing fringes caused by the capture path apart from ones caused by the
  scene, which the processed image cannot.

Everything here is GUI-free and camera-free. Each monitor is told the time by
its caller, never reads a clock, and writes only WARNING lines for episodes
(rate-limited), DEBUG lines for background detail and the frame check's
periodic INFO line.
"""

import collections
import dataclasses
import statistics

import cv2
import numpy as np

GAP_FACTOR = 3.0           # a gap is this many expected frame intervals ...
GAP_MIN_S = 0.15           # ... and at least this long
LOG_INTERVAL_S = 1.0       # at most one line per kind this often
SHARPNESS_INTERVAL_S = 0.25   # sharpness is measured at most this often
SHARPNESS_SIZE = 256       # frames are shrunk to about this many pixels first
SHARPNESS_HISTORY = 60     # samples (about 15 s) that make the typical value
SHARPNESS_MIN_SAMPLES = 20  # no verdict before this many samples
SHARPNESS_DROP = 0.6       # an episode starts below this share of typical
SHARPNESS_RECOVER = 0.8    # and ends at or above this share
DELIVERED_COUNTER = "StreamDeliveredFrameCount"
FRAME_CHECK_INTERVAL_S = 10.0  # one frame-check line per camera this often
BURST_COLOURS = 2          # a diagnostic capture keeps this many BGR frames
CHECK_CROP = 256           # speckle and detail are measured on this square crop
LOCAL_MEAN_SIGMA = 8.0     # px; removes illumination so only speckle remains
REPEAT_STEP = 16           # repeated frames are spotted on every 16th pixel
VIDEO_BLACK = 16           # analog video's legal luma range (BT.601)
VIDEO_WHITE = 235
# Detail is compared across and down in two bands of cycles per pixel: a low
# band, which only overall softness lowers, and the band around NTSC's colour
# subcarrier (3.58 MHz sampled at 13.5 MHz: one cycle per ~3.8 px), which a
# colour decoder's notch or comb filter removes from the brightness signal.
LOW_BAND = (0.10, 0.18)
SUBCARRIER_BAND = (0.22, 0.31)


class Throttle:
    """Allow one log line per period and count the ones held back."""

    def __init__(self, period=LOG_INTERVAL_S):
        """Create a throttle that has not yet let anything through.

        Args: period is the minimum time between allowed lines, in seconds.
        Returns: None.
        """
        self.period = period
        self.last = None
        self.suppressed = 0

    def allow(self, now):
        """Decide whether a line may be written now.

        Args: now is a monotonic time in seconds.
        Returns: True when the line may be written; otherwise the held-back
            count rises and False is returned.
        """
        if self.last is not None and now - self.last < self.period:
            self.suppressed += 1
            return False
        self.last = now
        return True

    def note(self):
        """Take the count of lines held back since the last one written.

        Args: None.
        Returns: A text such as " (+3 more since the last line)", or "" when
            nothing was held back; the count restarts.
        """
        held, self.suppressed = self.suppressed, 0
        return f" (+{held} more since the last line)" if held else ""


def settings_summary(settings):
    """Describe the settings that decide how a frame is filtered and shown.

    Args: settings is an engine.Settings (any object with its fields).
    Returns: One line of text for the log.
    """
    averaging = f"{settings.average}{' rolling' if settings.rolling else ' block'}"
    smoothing = (f"{settings.smoothing_method} {settings.smoothing}"
                 if settings.smoothing > 1 else "off")
    if settings.scale_range:
        scaling = f"stretch clip {settings.clip_percent:g}%"
    elif settings.gain_before_average:
        scaling = (f"gain x{settings.difference_gain:g} before average, "
                   f"multiplier x{settings.average_multiplier:g}")
    else:
        scaling = f"gain x{settings.difference_gain:g}"
    outliers = (f"{settings.outlier_filter}px" if settings.outlier_filter > 1
                else "off")
    return (f"mode={settings.mode} frames_back={settings.frames_back} "
            f"noise_reduction={settings.noise_reduction:g} "
            f"outlier_filter={outliers} average={averaging} smoothing={smoothing} "
            f"scaling={scaling} "
            f"exposure={settings.exposure_us:g}us gain={settings.gain_db:g}dB "
            f"fps={settings.fps:g} clahe={'on' if settings.clahe else 'off'}")


def sharpness(image):
    """Measure how much fine detail a frame holds.

    The frame is shrunk to about SHARPNESS_SIZE pixels, then the variance of
    its Laplacian is taken. Blur lowers the value; only changes in it from
    frame to frame are meaningful, not its absolute size.

    Args: image is a 2-D uint8 array.
    Returns: The sharpness value as a float (0.0 for a flat image).
    """
    height, width = image.shape[:2]
    step = max(1, max(height, width) // SHARPNESS_SIZE)
    if step > 1:
        image = cv2.resize(image, (max(1, width // step), max(1, height // step)),
                           interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


class FrameGapMonitor:
    """Log long gaps between good frames and bursts of incomplete frames."""

    def __init__(self, log):
        """Create a monitor that has seen no frames yet.

        Args: log is an AppLog or PrefixedLog.
        Returns: None.
        """
        self.log = log
        self.last_frame = None
        self.gaps = 0
        self.longest = 0.0
        self.gap_throttle = Throttle()
        self.burst = 0
        self.burst_start = 0.0
        self.burst_last = 0.0
        self.incomplete_total = 0
        self.burst_throttle = Throttle()

    def incomplete(self, now):
        """Count one incomplete frame towards the current burst.

        Args: now is the monotonic time the frame arrived.
        Returns: None; the burst is written when the next good frame arrives.
        """
        self.incomplete_total += 1
        if not self.burst:
            self.burst_start = now
        self.burst += 1
        self.burst_last = now

    def frame_ok(self, now, expected_s, context):
        """Record a good frame, writing any burst or gap that just ended.

        Args: now is the monotonic time of the frame; expected_s is the
            expected interval between frames in seconds; context is a
            zero-argument callable returning the text to attach (settings and
            rates), called only when a line is written.
        Returns: None.
        """
        last, self.last_frame = self.last_frame, now
        if self.burst:
            count, self.burst = self.burst, 0
            if self.burst_throttle.allow(now):
                self.log.warning(
                    f"Incomplete-frame burst: {count} frame(s) over "
                    f"{self.burst_last - self.burst_start:.2f} s ended "
                    f"({self.incomplete_total} incomplete since start)"
                    f"{self.burst_throttle.note()}; {context()}")
        if last is None:
            return
        gap = now - last
        if gap <= max(GAP_MIN_S, GAP_FACTOR * expected_s):
            return
        self.gaps += 1
        self.longest = max(self.longest, gap)
        if self.gap_throttle.allow(now):
            self.log.warning(
                f"Frame gap: {gap:.3f} s without a good frame (expected "
                f"{expected_s:.3f} s); {self.gaps} gap(s) since start, longest "
                f"{self.longest:.3f} s{self.gap_throttle.note()}; {context()}")


class StreamCounterMonitor:
    """Log increases in the transport layer's lost and dropped frame counters."""

    def __init__(self, log):
        """Create a monitor with no earlier reading.

        Args: log is an AppLog or PrefixedLog.
        Returns: None.
        """
        self.log = log
        self.previous = None
        self.noted_empty = False

    def update(self, counters, context):
        """Compare a new reading with the last one.

        Args: counters maps counter names to totals (empty when the camera
            offers none); context is a zero-argument callable returning the
            text to attach to a warning.
        Returns: None.
        """
        if not counters:
            if not self.noted_empty:
                self.log.debug("Stream counters: none readable on this camera")
                self.noted_empty = True
            return
        previous, self.previous = self.previous, dict(counters)
        shown = ", ".join(f"{name}={value}" for name, value in sorted(counters.items()))
        if previous is None:
            self.log.debug(f"Stream counters at start: {shown}")
            return
        rises = {name: value - previous[name] for name, value in counters.items()
                 if name in previous and value > previous[name]}
        lost = {name: rise for name, rise in rises.items() if name != DELIVERED_COUNTER}
        if lost:
            changes = ", ".join(f"{name} +{rise}" for name, rise in sorted(lost.items()))
            self.log.warning(f"Camera stream counters rose: {changes}; totals {shown}; "
                             f"{context()}")


class SharpnessMonitor:
    """Watch one stage's sharpness and log each drop below its typical value."""

    def __init__(self, log, stage):
        """Create a monitor with no history.

        Args: log is an AppLog or PrefixedLog; stage names the measured frames
            in the log ("camera frame", "displayed frame").
        Returns: None.
        """
        self.log = log
        self.stage = stage
        self.history = collections.deque(maxlen=SHARPNESS_HISTORY)
        self.next_sample = 0.0
        self.episode_start = None
        self.lowest = 1.0

    def observe(self, image, now, context):
        """Measure a frame (at a limited rate) and log drops and recoveries.

        The typical value is the median of recent samples and is frozen while
        an episode lasts, so a long blur does not become the new normal.

        Args: image is a 2-D uint8 array; now is the monotonic time in
            seconds; context is a zero-argument callable returning the text
            to attach to a line.
        Returns: None.
        """
        if now < self.next_sample:
            return
        self.next_sample = now + SHARPNESS_INTERVAL_S
        value = sharpness(image)
        typical = (statistics.median(self.history)
                   if len(self.history) >= SHARPNESS_MIN_SAMPLES else 0.0)
        if typical <= 0.0:
            self.history.append(value)
            return
        share = value / typical
        if self.episode_start is None:
            if share < SHARPNESS_DROP:
                self.episode_start = now
                self.lowest = share
                self.log.warning(
                    f"Sharpness drop ({self.stage}): {value:.1f} against typical "
                    f"{typical:.1f} ({share:.0%}); {context()}")
            else:
                self.history.append(value)
            return
        self.lowest = min(self.lowest, share)
        if share >= SHARPNESS_RECOVER:
            self.log.warning(
                f"Sharpness recovered ({self.stage}) after "
                f"{now - self.episode_start:.1f} s; lowest {self.lowest:.0%} of "
                f"typical; {context()}")
            self.episode_start = None
            self.history.append(value)


def _percentile(counts, share):
    """Find the level below which a share of the pixels lie.

    Args: counts is the cumulative 256-bin histogram of a uint8 image (a
        float array); share is a fraction from 0 to 1.
    Returns: The level as an int.
    """
    return int(np.searchsorted(counts, counts[-1] * share, side="left"))


def level_stats(image):
    """Describe how a frame uses the 8-bit range.

    Analog video's legal luma runs from VIDEO_BLACK to VIDEO_WHITE; pixels at
    or beyond those levels may have been clipped by the capture decoder.

    Args: image is a nonempty 2-D uint8 array.
    Returns: A dict with p1, p50 and p99 (levels), and the percentages of
        pixels at 0, at or below VIDEO_BLACK, at or above VIDEO_WHITE and
        at 255.
    """
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    counts = np.cumsum(hist)
    total = counts[-1] / 100.0
    return {"p1": _percentile(counts, 0.01), "p50": _percentile(counts, 0.5),
            "p99": _percentile(counts, 0.99),
            "zero": hist[0] / total, "black": counts[VIDEO_BLACK] / total,
            "white": hist[VIDEO_WHITE:].sum() / total, "full": hist[255] / total}


def change_stats(previous, current):
    """Measure how much a frame differs from the one before it.

    Args: previous and current are 2-D uint8 arrays of the same shape.
    Returns: A dict with the median, 95th percentile and maximum absolute
        difference, in grey levels.
    """
    diff = cv2.absdiff(previous, current)
    counts = np.cumsum(cv2.calcHist([diff], [0], None, [256], [0, 256]).ravel())
    return {"median": _percentile(counts, 0.5), "p95": _percentile(counts, 0.95),
            "max": int(diff.max())}


def brightest_crop(image, size=CHECK_CROP):
    """Cut a square around the brightest part of a frame.

    The speckle of a laser-lit object is measured where the object is lit,
    not in a dark corner.

    The frame is shrunk 8 times and box-averaged over the crop's size, so the
    whole frame is searched at little cost; the crop is centred on the
    brightest average and kept inside the frame.

    Args: image is a 2-D uint8 array; size is the side of the square.
    Returns: A tuple of (crop, x, y), where x and y are the crop's top-left
        corner. A frame smaller than size gives a crop as large as it allows.
    """
    height, width = image.shape
    side = min(size, height, width)
    small = cv2.resize(image, (max(1, width // 8), max(1, height // 8)),
                       interpolation=cv2.INTER_AREA)
    window = max(1, side // 8)
    means = cv2.blur(small.astype(np.float32), (window, window))
    row, col = np.unravel_index(int(np.argmax(means)), means.shape)
    y = min(max(0, int(row) * 8 + 4 - side // 2), height - side)
    x = min(max(0, int(col) * 8 + 4 - side // 2), width - side)
    return image[y:y + side, x:x + side], x, y


def _half_width(profile):
    """Measure the full width at half maximum of an autocorrelation profile.

    Args: profile is the normalized autocorrelation from lag 0 outwards
        (profile[0] == 1).
    Returns: Twice the lag, interpolated, at which it first falls to 0.5, in
        pixels; the profile's whole length when it never does.
    """
    below = np.nonzero(profile < 0.5)[0]
    if not below.size:
        return 2.0 * len(profile)
    lag = int(below[0])
    high, low = profile[lag - 1], profile[lag]
    return 2.0 * (lag - 1 + (high - 0.5) / (high - low))


def speckle_stats(crop):
    """Measure speckle contrast and size in a lit part of the frame.

    The illumination is removed by dividing (contrast) or subtracting (size)
    a heavy blur. Fully developed, well-resolved speckle has a contrast near
    1 and is a few pixels across; speckle smaller than the pixels, light
    spread by the capture path, or an incoherent background pull contrast
    down; pixel noise alone gives a size near 1 px.

    Args: crop is a 2-D uint8 array, ideally CHECK_CROP square.
    Returns: A dict with contrast (standard deviation over local mean) and
        size_x and size_y (autocorrelation full width at half maximum,
        across and down, in pixels), or None for a flat crop.
    """
    values = crop.astype(np.float32)
    local = cv2.GaussianBlur(values, (0, 0), LOCAL_MEAN_SIGMA)
    detail = values - local
    if float(np.std(detail)) < 1e-3:
        return None
    power = np.abs(np.fft.fft2(detail)) ** 2
    autocorrelation = np.fft.ifft2(power).real
    autocorrelation /= autocorrelation[0, 0]
    half_y, half_x = crop.shape[0] // 2, crop.shape[1] // 2
    contrast = float(np.std(values / np.maximum(local, 1.0)))
    return {"contrast": contrast,
            "size_x": _half_width(autocorrelation[0, :half_x + 1]),
            "size_y": _half_width(autocorrelation[:half_y + 1, 0])}


def _band_power(detail, axis, band):
    """Average the power of a crop's detail in one frequency band.

    Args: detail is a 2-D float array with the illumination removed; axis
        is 1 to look across rows (horizontal detail), 0 down columns; band is
        (low, high) in cycles per pixel.
    Returns: The mean power in that band, as a float.
    """
    power = np.abs(np.fft.rfft(detail, axis=axis)) ** 2
    frequencies = np.fft.rfftfreq(detail.shape[axis])
    chosen = (frequencies >= band[0]) & (frequencies <= band[1])
    picked = power[:, chosen] if axis == 1 else power[chosen, :]
    return float(picked.mean())


def detail_ratios(crop):
    """Compare detail across the frame with detail down it, in two bands.

    Speckle is the same in every direction, so a ratio near 1 is normal. An
    analog line softens everything across a little (both ratios low); a
    colour decoder's notch removes the subcarrier band from the brightness
    signal, which leaves the low band near 1 and the subcarrier band low.

    Args: crop is a 2-D uint8 array.
    Returns: A dict with low and subcarrier (power across / power down in
        LOW_BAND and SUBCARRIER_BAND), or None for a flat crop.
    """
    values = crop.astype(np.float32)
    detail = values - cv2.GaussianBlur(values, (0, 0), LOCAL_MEAN_SIGMA)
    if float(np.std(detail)) < 1e-3:
        return None
    ratios = {}
    for name, band in (("low", LOW_BAND), ("subcarrier", SUBCARRIER_BAND)):
        down = _band_power(detail, 0, band)
        if down <= 1e-6:
            return None
        ratios[name] = _band_power(detail, 1, band) / down
    return ratios


def colour_spread(bgr):
    """Measure colour in a frame that should be grey.

    A monochrome camera's video should decode to R = G = B. Colour means the
    decoder read fine brightness detail as a colour signal and moved it out of
    the brightness channel, which is the only one the viewer keeps.

    Args: bgr is a 3-D uint8 BGR array.
    Returns: A tuple of the mean absolute B-G and R-G differences, in grey
        levels.
    """
    blue, green, red = cv2.split(bgr)
    return (float(cv2.absdiff(blue, green).mean()),
            float(cv2.absdiff(red, green).mean()))


def field_mismatch(image):
    """Compare the two fields of an interlaced frame.

    Neighbouring rows from different fields (one row apart) are compared with
    rows of the same field (two rows apart). In a still scene the nearer rows
    agree at least as well, giving 1 or less; a field that moved or changed
    between its 1/60 s exposures raises the ratio well above 1.

    Args: image is a 2-D uint8 array with at least three rows.
    Returns: The ratio as a float, or None when both differences are zero.
    """
    values = image.astype(np.float32)
    even, odd = values[0::2], values[1::2]
    rows = min(len(even), len(odd))
    between = float(np.abs(even[:rows] - odd[:rows]).mean())
    within = float(np.abs(even[:-1] - even[1:]).mean())
    if within <= 1e-6:
        return None
    return between / within


class FrameCheckMonitor:
    """Log what the raw camera frames look like every FRAME_CHECK_INTERVAL_S."""

    def __init__(self, log, interval=FRAME_CHECK_INTERVAL_S):
        """Create a monitor that checks the first frames it sees.

        Args: log is an AppLog or PrefixedLog; interval is the time between
            lines, in seconds.
        Returns: None.
        """
        self.log = log
        self.interval = interval
        self.next_check = None
        self.pending = None  # (frame copy, colour frame) awaiting its successor
        self.signature = None
        self.frames = 0
        self.repeats = 0
        self.view = None  # (mode, low, high) last shown

    def note_view(self, mode, low=None, high=None):
        """Remember the view and its difference range for the next line.

        Args: mode is the view name; low and high are the input range of a
            difference view, or None for Raw.
        Returns: None.
        """
        self.view = (mode, low, high)

    def observe(self, frame, now, context, colour=None, interlaced=False):
        """Count repeated frames and, when due, measure a pair of frames.

        A check copies one frame and measures it against the next, so the
        frame-to-frame change is between two consecutive frames.

        Args: frame is the raw 2-D uint8 camera frame (before single-field
            processing); now is the monotonic time in seconds; context is a
            zero-argument callable returning the text to attach; colour is the
            BGR frame the device delivered, or None; interlaced says the frame
            holds two fields.
        Returns: None.
        """
        signature = frame[::REPEAT_STEP, ::REPEAT_STEP].tobytes()
        if signature == self.signature:
            self.repeats += 1
        self.signature = signature
        self.frames += 1
        if self.pending is not None:
            previous, previous_colour = self.pending
            self.pending = None
            if previous.shape == frame.shape:
                self.log.write(self._line(previous, frame, previous_colour,
                                          interlaced, context))
            self.next_check = now + self.interval
            self.frames = self.repeats = 0
            return
        if self.next_check is None or now >= self.next_check:
            self.pending = (frame.copy(), colour)

    def _line(self, previous, frame, colour, interlaced, context):
        """Format one frame-check line from a pair of consecutive frames.

        Args: previous and frame are consecutive raw frames; colour is the
            delivered BGR frame of the first one, or None; interlaced says the
            frames hold two fields; context is the engine's context callable.
        Returns: The text of the line.
        """
        parts = pair_facts(previous, frame, colour, interlaced)
        parts.insert(2, f"repeated frames {self.repeats} of {self.frames}")
        if self.view is not None:
            mode, low, high = self.view
            parts.append(f"view {mode}" if low is None
                         else f"view {mode} input {low}–{high}")
        return "Frame check: " + "; ".join(parts) + f"; {context()}"


def pair_facts(previous, frame, colour=None, interlaced=False):
    """Describe a raw frame and its change to the next one, as log phrases.

    Shared by the frame-check line and the diagnostic capture's analysis.

    Args: previous and frame are consecutive raw 2-D uint8 frames of one
        shape; colour is the delivered BGR frame of the first one, or None;
        interlaced says the frames hold two fields.
    Returns: A list of phrases: levels, change to the next frame, speckle,
        detail across/down (when the crop has detail), colour (when given)
        and field mismatch (when interlaced).
    """
    levels = level_stats(previous)
    change = change_stats(previous, frame)
    parts = [
        f"levels p1/p50/p99 {levels['p1']}/{levels['p50']}/{levels['p99']}, "
        f"{levels['black']:.1f}% at or below {VIDEO_BLACK}, "
        f"{levels['white']:.1f}% at or above {VIDEO_WHITE} "
        f"({levels['zero']:.1f}% at 0, {levels['full']:.1f}% at 255)",
        f"change to next frame median {change['median']} p95 {change['p95']} "
        f"max {change['max']}"]
    crop, x, y = brightest_crop(previous)
    speckle = speckle_stats(crop)
    if speckle is None:
        parts.append(f"speckle at ({x}, {y}): flat")
    else:
        parts.append(f"speckle at ({x}, {y}) contrast {speckle['contrast']:.2f}, "
                     f"size {speckle['size_x']:.1f} × {speckle['size_y']:.1f} px "
                     "(across × down)")
    ratios = detail_ratios(crop)
    if ratios is not None:
        parts.append(f"detail across/down {ratios['low']:.2f} at "
                     f"{LOW_BAND[0]:g}–{LOW_BAND[1]:g} c/px, "
                     f"{ratios['subcarrier']:.2f} at "
                     f"{SUBCARRIER_BAND[0]:g}–{SUBCARRIER_BAND[1]:g} c/px")
    if colour is not None:
        blue, red = colour_spread(colour)
        parts.append(f"colour B−G {blue:.2f} R−G {red:.2f}")
    if interlaced:
        mismatch = field_mismatch(previous)
        parts.append("field mismatch " + ("n/a" if mismatch is None
                                          else f"{mismatch:.2f}"))
    return parts


@dataclasses.dataclass(frozen=True)
class Burst:
    """Consecutive raw frames the engine collected for a diagnostic capture.

    Attributes: frames are the raw 2-D uint8 frames, before single-field
        mode, in order; colours are the BGR frames the device delivered for
        the first few of them (empty for sources without colour);
        interlaced says the frames hold two fields; camera lists the
        source's description lines; details are a video device's (label,
        text) pairs (empty for other sources); settings is the engine's
        Settings at the time; screen is the displayed image (palette and
        gamma applied) or None; demo marks synthetic frames.
    """

    frames: tuple
    colours: tuple
    interlaced: bool
    camera: tuple
    details: tuple
    settings: object
    screen: object
    demo: bool
