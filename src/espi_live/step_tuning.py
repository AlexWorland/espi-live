"""Experimental auto-tune of the piezo triangle for a steady difference image.

A steady ramp of the reference piezo changes the reference phase by the same
step between successive frames. The Previous view's difference is strongest,
and least sensitive to small changes of that step (piezo hysteresis, frame
timing), when the step is pi, half a wavelength of path: that is where the
flicker those changes cause is smallest. The step is measured from the
frames themselves, with no calibration.

For each pixel, I = a + b cos(psi + phi) with a random speckle phase psi, so
the mean squared difference of frames k apart is
D(k) = C (1 - cos(k step)) + N, with N the camera noise. Their ratio
q = D(2) / D(1) = 2 (1 + cos step) (plus noise) falls to its least, the
noise alone, at a step of pi: frames two apart are then a whole 2 pi apart.

StepMonitor works on the engine's worker thread (feed) and is read from the
front end (stats). StepTuner is pure logic: given the monitor's statistics it
hill-climbs a factor on the piezo frequency until q stops improving, then
holds and watches. q and the step are the same whichever side of pi the step
is, so the direction is found by trying. Nothing here imports a GUI or
touches a device.
"""

import collections
import math
import statistics
import threading
from dataclasses import dataclass

SUBSAMPLE = 4  # every 4th pixel each way: plenty of speckles, little work
HISTORY = 600  # frame triples kept (20 s at 30 fps)
SETTLE_S = 1.0  # frames this soon after a frequency change are not measured
MEASURE_S = 3.0  # each setting is measured for at least this long
MIN_SAMPLES = 20  # and over at least this many frame triples
COARSE_DEG = 150.0  # a measured step below this is corrected by probe and jump
PROBE = 0.05  # relative change that shows which side of pi the step is on
MAX_PROBE = 0.2
MIN_CHANGE_DEG = 2.0  # a probe moving the step less than this is repeated, larger
FINE_STEP = 0.05  # first relative change of the hill-climb near pi
MIN_STEP = 0.01  # below this, the factor is held (the step is then near pi)
BETTER = 0.9  # a try is kept when q falls below this fraction of the best
RETUNE_Q = 0.25  # while holding, q this far above its best (about 150°) restarts
MAX_JUMP = 2.0  # one move changes the factor by at most this ratio
MIN_SCALE, MAX_SCALE = 0.25, 4.0  # the factor never leaves this range


@dataclass(frozen=True)
class StepStats:
    """What the frames since a moment say about the piezo's step.

    q is (D(2) - N) / (D(1) - N) from the medians of D(1) and D(2) (the
    median drops the frames at the triangle's turning points) and the noise
    N; step_deg the step it implies, folded into 0-180° (a step and 360°
    minus it look the same); flicker the variation of D(1) from frame to
    frame (standard deviation over mean); d1 the median D(1);
    samples how many frame triples were measured.
    """

    q: float
    step_deg: float
    flicker: float
    samples: int
    d1: float


def step_from_ratio(q):
    """Work out the phase step a D(2)/D(1) ratio implies.

    Args: q is D(2)/D(1).
    Returns: The step in degrees, 0-180 (180 at q = 0); camera noise makes it
        read low near 180°.
    """
    cosine = min(1.0, max(-1.0, q / 2.0 - 1.0))
    return math.degrees(math.acos(cosine))


class StepMonitor:
    """Measure D(1) and D(2) from successive frames, while switched on."""

    def __init__(self):
        """Start switched off with no frames.

        Args: None.
        Returns: None.
        """
        self._lock = threading.Lock()
        self._on = False
        self._frames = collections.deque(maxlen=2)  # the two frames before this one
        self._samples = collections.deque(maxlen=HISTORY)  # (time, d1, d2)

    @property
    def on(self):
        """Tell whether frames are being measured.

        Args: None.
        Returns: True while switched on.
        """
        return self._on

    def switch(self, on):
        """Start or stop measuring; either way the frames so far are dropped.

        Args: on is True to measure.
        Returns: None.
        """
        with self._lock:
            self._on = bool(on)
            self._frames.clear()
            self._samples.clear()

    def feed(self, frame, now):
        """Measure one frame against the two before it.

        Args: frame is a 2-D camera image (any numeric type); now is its time
            in seconds.
        Returns: None. Does nothing while switched off; a frame of another
            size starts again.
        """
        if not self._on:
            return
        small = frame[::SUBSAMPLE, ::SUBSAMPLE].astype("float32")
        with self._lock:
            if self._frames and self._frames[-1].shape != small.shape:
                self._frames.clear()
            if len(self._frames) == 2:
                d1 = float(((small - self._frames[1]) ** 2).mean())
                d2 = float(((small - self._frames[0]) ** 2).mean())
                self._samples.append((now, d1, d2))
            self._frames.append(small)

    def stats(self, since, noise=0.0):
        """Summarise the frames measured from a moment on.

        Args: since is a time in seconds (same clock as feed); noise is the
            camera's D(1) with the piezo still, taken off D(1) and D(2).
        Returns: StepStats, or None when no frame triple has been measured
            since then.
        """
        with self._lock:
            chosen = [(d1, d2) for when, d1, d2 in self._samples if when >= since]
        if not chosen:
            return None
        d1s = [d1 for d1, _ in chosen]
        d1 = statistics.median(d1s)
        d2 = statistics.median(d2 for _, d2 in chosen)
        signal = d1 - noise
        q = (d2 - noise) / signal if signal > 0 else 2.0  # all noise: no step to see
        mean = statistics.fmean(d1s)
        flicker = statistics.pstdev(d1s) / mean if mean > 0 else 0.0  # repeated frames
        return StepStats(q, step_from_ratio(q), flicker, len(chosen), d1)


class StepTuner:
    """Find and hold the piezo frequency factor that gives a step of pi.

    Far from pi (the measured step under COARSE_DEG), a small probe shows
    which side of pi the step is on (raising the frequency raises the step
    below pi and lowers how it looks above pi), and one jump then sets the
    factor for pi. Near pi, a hill-climb in small, halving steps finds the
    least D(2)/D(1), which camera noise does not move. The factor is then
    held and watched; a step that drifts well away starts it all again.

    The front end calls changed() whenever the piezo frequency is applied,
    and update() regularly; update() returns a new factor to apply, or None.
    """

    def __init__(self):
        """Start with a factor of 1, about to measure it.

        Args: None.
        Returns: None.
        """
        self.scale = 1.0
        self.state = "noise"  # then "measuring", "probing", "trying", "holding"; or "lost"
        self.noise = 0.0  # D(1) with the piezo held still
        self.best = None  # q at the factor kept
        self.kept = 1.0  # the factor kept while probing or trying another
        self.seen = 0.0  # the step measured at the kept factor (degrees)
        self.probe = PROBE
        self.step = FINE_STEP
        self.direction = 1
        self.since = None  # when the newest frequency was applied
        self.last = None  # the newest StepStats
        self.jump = None  # (factor, step seen, other side's step) before the last jump

    @property
    def wants_hold(self):
        """Tell whether the piezo should hold still (while the noise is measured).

        Args: None.
        Returns: True in the "noise" state.
        """
        return self.state == "noise"

    def changed(self, now):
        """Note that the piezo frequency was just applied.

        Args: now is the time in seconds.
        Returns: None.
        """
        self.since = now

    def _move(self, scale, state, now):
        """Go to a new factor (within limits) and measure it.

        A jump is at most a factor of MAX_JUMP from the kept factor.

        Args: scale is the factor wanted; state is the state to measure it
            in; now is the time in seconds.
        Returns: The factor to apply: 1 (the computed frequency) and the
            "lost" state when the factor would leave MIN_SCALE-MAX_SCALE.
        """
        scale = min(self.kept * MAX_JUMP, max(self.kept / MAX_JUMP, scale))
        if not MIN_SCALE <= scale <= MAX_SCALE:  # running away: give up safely
            self.scale, self.state = 1.0, "lost"
            return self.scale
        self.scale = scale
        self.state = state
        self.since = now
        return self.scale

    def update(self, now, monitor):
        """Take the newest measurement and decide the next factor.

        Args: now is the time in seconds; monitor is the StepMonitor.
        Returns: A new factor to apply, or None to keep the present one.
        """
        if self.state == "lost":
            return None
        if self.since is None:
            self.since = now
        start = self.since + SETTLE_S
        stats = monitor.stats(start, self.noise)
        if stats is None or now - start < MEASURE_S or stats.samples < MIN_SAMPLES:
            return None
        if self.state == "noise":
            self.noise = stats.d1
            self.state, self.since = "measuring", now
            return self.scale  # back to the triangle
        self.last = stats
        if self.state == "measuring":
            jump, self.jump = self.jump, None
            if jump is not None and stats.step_deg + MIN_CHANGE_DEG < jump[1]:
                # The jump made it worse: the probe read the side wrong.
                self.kept = jump[0]
                return self._move(jump[0] * 180.0 / jump[2], "measuring", now)
            self.best, self.kept, self.seen = stats.q, self.scale, stats.step_deg
            if stats.step_deg < COARSE_DEG:
                return self._move(self.kept * (1 + self.probe), "probing", now)
            self.step, self.direction = FINE_STEP, 1
            return self._move(self.kept * (1 + self.step), "trying", now)
        if self.state == "probing":
            change = stats.step_deg - self.seen
            if abs(change) < MIN_CHANGE_DEG:  # lost in the noise: probe further
                self.probe = min(MAX_PROBE, self.probe * 2)
                return self._move(self.kept * (1 + self.probe), "probing", now)
            self.probe = PROBE
            seen = min(179.0, max(1.0, self.seen))  # 0° would jump without limit
            step, other = (seen, 360.0 - seen) if change > 0 else (360.0 - seen, seen)
            self.jump = (self.kept, self.seen, other)
            return self._move(self.kept * 180.0 / step, "measuring", now)
        if self.state == "trying":
            if stats.q < self.best * BETTER:
                self.best, self.kept = stats.q, self.scale
                return self._move(self.kept * (1 + self.direction * self.step), "trying",
                                  now)
            self.direction = -self.direction
            self.step /= 2
            if self.step < MIN_STEP:
                return self._move(self.kept, "holding", now)
            return self._move(self.kept * (1 + self.direction * self.step), "trying", now)
        # holding: watch, and start again if the step has drifted well away
        if stats.q > self.best + RETUNE_Q:
            self.state, self.since = "measuring", now
        else:
            self.best = min(self.best, stats.q)
        return None

    def describe(self):
        """Say what the tuner is doing, for the DAQ window.

        Args: None.
        Returns: One sentence.
        """
        if self.state == "noise":
            return "Auto-tune: piezo held still to measure the camera noise…"
        if self.state == "lost":
            return ("Auto-tune could not find the step and went back to the computed "
                    "frequency. Keep the object still, check that the piezo moves, and "
                    "turn Auto-tune off and on to try again.")
        if self.last is None:
            return "Auto-tune: measuring the step from the frames…"
        s = self.last
        doing = {"measuring": "measuring", "probing": f"probing ×{self.scale:.3f}",
                 "trying": f"trying ×{self.scale:.3f}",
                 "holding": f"holding ×{self.scale:.3f}"}[self.state]
        return (f"Auto-tune: {doing}; step about {s.step_deg:.0f}° per frame "
                f"(D2/D1 {s.q:.2f}), flicker {s.flicker:.0%}.")
