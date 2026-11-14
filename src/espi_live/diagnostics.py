"""Frame-quality diagnostics for the acquisition loop.

Intermittent blur or lost detail for a few seconds can come from the camera
link (late or incomplete frames), from the sensor or optics, from the temporal
filters, or from the display. These monitors write the evidence to the log so
a short episode can be told apart afterwards:

- FrameGapMonitor: gaps between good frames and bursts of incomplete frames.
- StreamCounterMonitor: the transport layer's own lost/dropped frame counters.
- SharpnessMonitor: a cheap sharpness value, measured on the camera frame and
  again on the frame that is displayed, with the start and end of every drop.
- settings_summary: the settings in force, attached to every episode line
  and written in full when streaming starts and when a preset is applied.
- SettingsChangeLog: one line per changed setting once its value has
  settled, so dragging a slider writes the final value, not every step.

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
SETTLE_S = 0.75            # a changed setting is logged once unchanged this long


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


def on_off(flag):
    """Spell a switch the way the log writes it.

    Args: flag is any truthy or falsy value.
    Returns: "on" or "off".
    """
    return "on" if flag else "off"


def settings_summary(settings, interlaced=False):
    """Describe every setting that decides how a frame is filtered and shown.

    The first fields keep their names and order so older logs read the same;
    later ones are appended after them.

    Args: settings is an engine.Settings (any object with its fields);
        interlaced is True when the source delivers interlaced frames, the
        only case in which single_field changes the image and is listed.
    Returns: One line of text for the log.
    """
    averaging = f"{settings.average}{' rolling' if settings.rolling else ' block'}"
    smoothing = (f"{settings.smoothing_method} {settings.smoothing}"
                 if settings.smoothing > 1 else "off")
    fields = (f"ref_refresh={settings.ref_refresh_s:g}s "
              f"stretch={on_off(settings.scale_range)} "
              f"clip={settings.clip_percent:g}% gamma={settings.gamma:g} "
              f"palette={settings.palette} invert={on_off(settings.invert)} ")
    if interlaced:
        fields += f"single_field={on_off(settings.single_field)} "
    return (f"mode={settings.mode} noise_reduction={settings.noise_reduction:g} "
            f"average={averaging} smoothing={smoothing} "
            f"exposure={settings.exposure_us:g}us gain={settings.gain_db:g}dB "
            f"fps={settings.fps:g} clahe={on_off(settings.clahe)} "
            f"{fields}flip={on_off(settings.flip_horizontal)}")


def change_texts(settings):
    """Describe each setting the change log follows, as it would be logged.

    View and Fields are left out: the engine logs those at once, because
    they reset the processing.

    Args: settings is an engine.Settings (any object with its fields).
    Returns: A dict from the line's label to its value text; two settings
        with the same texts look the same on screen.
    """
    return {
        "Exposure": f"{settings.exposure_us:g} us",
        "Gain": f"{settings.gain_db:g} dB",
        "Frame rate": f"{settings.fps:g} fps",
        "Noise reduction": (f"{settings.noise_reduction:g}"
                            if settings.noise_reduction > 0 else "off"),
        "Averaging": (f"{settings.average} frames, "
                      f"{'rolling' if settings.rolling else 'block'}"
                      if settings.average > 1 else "off"),
        "Reference auto-refresh": (f"every {settings.ref_refresh_s:.1f} s"
                                   if settings.ref_refresh_s > 0 else "off"),
        "Speckle filter": (f"{settings.smoothing_method} {settings.smoothing} px"
                           if settings.smoothing > 1 else "off"),
        "Stretch": on_off(settings.scale_range),
        "Stretch clip": f"{settings.clip_percent:g}%",
        "Invert": on_off(settings.invert),
        "Flip horizontal": on_off(settings.flip_horizontal),
        "Palette": settings.palette,
        "Gamma": f"{settings.gamma:g}",
        "CLAHE": on_off(settings.clahe),
    }


class SettingsChangeLog:
    """Log each changed setting once, after its value has settled.

    A slider sends every step of a drag; logging each one buried the log
    (twenty lines per drag). A change is written only when its value has
    stayed the same for settle_s, or at once by flush() when something is
    about to depend on it (a saved frame, a recording).
    """

    def __init__(self, log, settings, settle_s=SETTLE_S):
        """Start from settings that need no line.

        Args: log is an AppLog or PrefixedLog; settings is the starting
            engine.Settings; settle_s is how long a value must stay unchanged.
        Returns: None.
        """
        self.log = log
        self.settle_s = settle_s
        self.logged = change_texts(settings)
        self.pending = {}  # label -> (value text, time it took that value)

    def observe(self, settings, now):
        """Compare the settings with what was logged, writing settled changes.

        Args: settings is the engine.Settings in force; now is a monotonic
            time in seconds.
        Returns: None.
        """
        for label, text in change_texts(settings).items():
            waiting = self.pending.get(label)
            if text == self.logged[label]:
                self.pending.pop(label, None)  # changed back: nothing to say
            elif waiting is None or waiting[0] != text:
                self.pending[label] = (text, now)  # still moving: restart
            elif now - waiting[1] >= self.settle_s:
                self._write(label, text)

    def flush(self):
        """Write every change still waiting to settle.

        Args: None.
        Returns: None.
        """
        for label, (text, _since) in list(self.pending.items()):
            self._write(label, text)

    def mark_logged(self, settings):
        """Treat settings as already logged, after a full settings line.

        Args: settings is the engine.Settings that line described.
        Returns: None; changes still waiting are dropped.
        """
        self.logged = change_texts(settings)
        self.pending.clear()

    def _write(self, label, text):
        """Write one change and remember it as logged.

        Args: label and text are an entry of change_texts.
        Returns: None.
        """
        self.log.write(f"{label}: {text}")
        self.logged[label] = text
        self.pending.pop(label, None)


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
