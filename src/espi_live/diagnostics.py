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

Everything here is GUI-free and camera-free. Each monitor is told the time by
its caller, never reads a clock, and writes only WARNING lines for episodes
(rate-limited) and DEBUG lines for background detail.
"""

import collections
import statistics

import cv2

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
    return (f"mode={settings.mode} noise_reduction={settings.noise_reduction:g} "
            f"average={averaging} smoothing={smoothing} "
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
