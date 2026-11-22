"""Acquisition and ESPI processing on a dedicated worker thread.

The engine knows nothing about any GUI toolkit. A front end:

1. creates an Engine with an AppLog, then calls open_camera() or use_demo()
   on its own thread before start(), so it can ask the user about demo mode
   (or open_on_start() to open the camera on the worker instead);
2. calls start(); from then on every PySpin call happens on the worker thread;
3. polls latest_packet() / status() / events from a UI timer, and sends
   update_settings(...) and request(...) from widget handlers;
4. calls stop() and wait() at shutdown.

The worker publishes the newest FramePacket and drops older ones, so a slow
display never delays acquisition (the camera also uses NewestOnly buffering).

With several cameras, each one gets its own Engine (see channels.py), so each
has its own settings, reference frame, averaging and recording.
"""

import ctypes
import os
import queue
import threading
import time
from dataclasses import dataclass, replace

import cv2
import numpy as np

from espi_live import error_reports, sources
from espi_live.pixel_square import (SQUARE_INTERVAL_S, SquarePacket,
                                   square_area, square_values)
from espi_live.diagnostics import (FrameGapMonitor, SharpnessMonitor,
                                   StreamCounterMonitor, settings_summary)
from espi_live.camera_reports import capture_report
from espi_live.camera import CameraSession
from espi_live.capture_models import adapter_for
from espi_live.processing import (DEFAULT_DIFFERENCE_GAIN,
                                  ESPIProcessor, MAX_FRAMES_BACK,
                                  NoiseCalibration, NoiseFloorMap, PeakHold,
                                  RangeTracker, apply_display, apply_gain,
                                  apply_levels, denoise_strength, display_lut,
                                  even_out_brightness, local_normalize,
                                  raw_difference, reject_outliers,
                                  single_field, smooth_speckle, stretch_levels,
                                  subtract_floor, watermark_demo)
from espi_live.simulation import DEFAULT_OBJECT, DemoSource
from espi_live.recording import (AVIRecorder, recording_path, save_frame,
                                 save_frame_set)

CAMERA_TIMEOUT_MS = 100
SETTING_INTERVAL_S = 0.1
READBACK_INTERVAL_S = 1.0
RANGE_INTERVAL_S = 0.5
RECORDER_FINISH_TIMEOUT_S = 5.0
PREDECESSOR_TIMEOUT_S = 10.0  # open_on_start waits this long for the old engine
TIMING_LOG_INTERVAL_S = 5.0  # DEBUG frame-timing lines at most this often
# The reference comparison window is refreshed at most this often, so three
# extra full-size images do not slow a 6 MP camera's 30 fps.
COMPARE_INTERVAL_S = 0.1
# The view name Raw difference frames carry, in packets and recording names.
RAW_DIFFERENCE = "RawDiff"


@dataclass
class Settings:
    """User-adjustable acquisition, measurement and display settings."""

    mode: str = "Raw"
    exposure_us: float = 8000.0
    gain_db: float = 0.0
    fps: float = 30.0
    noise_reduction: float = 0.0  # 0–95 slider; 0 = off. See denoise_strength.
    average: int = 1
    rolling: bool = False         # running average instead of one image per N frames
    ref_refresh_s: float = 0.0    # recapture the Reference every N s; 0 = never
    ref_frames: int = 1           # camera frames averaged into each reference
    smoothing: int = 0            # speckle filter width in pixels; 0 = off
    smoothing_method: str = "Gaussian"
    frames_back: int = 1          # Previous view compares with the frame N back
    outlier_filter: int = 0       # IMAQ LowPass-style width in px, per frame; 0 = off
    scale_range: bool = True      # True: Auto stretch; False: Fixed gain
    difference_gain: float = DEFAULT_DIFFERENCE_GAIN  # Fixed gain multiplier
    gain_before_average: bool = False  # LabVIEW order: gain each frame, then average
    average_multiplier: float = 1.0    # scales the average in LabVIEW order
    clip_percent: float = 0.0     # stretch ignores this % of pixels at each end
    min_range: int = 0            # sensitivity floor: stretch never spans fewer levels; 0 = off
    threshold: int = 0            # raw differences below this many levels become 0; 0 = off
    steady_range_s: float = 0.0   # the stretch range shrinks over this many s; 0 = per frame
    even_brightness: bool = False  # divide out the illumination profile before averaging
    peak_hold_s: float = 0.0      # show each pixel's maximum over the last N s; 0 = off
    normalize_window: int = 0     # local normalization window in px (replaces stretch); 0 = off
    invert: bool = False
    flip_horizontal: bool = False  # mirror left-right (flip across the vertical axis)
    gamma: float = 1.0
    brightness: int = 0           # grey levels added to every displayed pixel; 0 = off
    display_gain: float = 1.0     # every displayed pixel multiplied by this; 1 = off
    contrast: float = 0.0         # S-curve contrast strength, 0–100 %; 0 = off
    record: bool = False
    single_field: bool = False    # interlaced sources: even rows only, line-doubled
    auto_reference: bool = False  # Reference view captures one when it has none
    average_frames_first: bool = False  # Average N averages camera frames, then subtracts
    floor_subtract: bool = False  # take the measured per-pixel noise floor off each difference
    raw_difference: bool = False  # |live - previous| raw frames; every other step off
    demo_object: str = DEFAULT_OBJECT  # what the simulated drumhead does (demo only)


# The processing of the LabVIEW ESPI 1.5 VI with its saved front-panel
# defaults (the known-good flow for the Sony XC-ST50, docs/labview/README.md):
# a reference grabbed at the start and never refreshed (BY FRAME off), full
# frames, each difference multiplied by Constant = 42 and saturated, Low Pass
# Filter off, Average off (block averaging, Multiplier 1, when turned on), no
# wait between grabs, and nothing the VI lacks (noise
# reduction, noise threshold, beam-brightness correction, speckle filter, peak
# hold, local normalization, stretch, invert, gamma, display brightness, gain
# and contrast). The Auto-stretch-only sensitivity floor and steady
# range are left as they are, since Fixed gain ignores them. Exposure, gain,
# frame rate, recording and the image flip are left alone.
LABVIEW_FLOW = {
    "mode": "Reference",
    "noise_reduction": 0.0,
    "frames_back": 1,
    "outlier_filter": 0,
    "average": 1,
    "rolling": False,
    "ref_refresh_s": 0.0,
    "smoothing": 0,
    "threshold": 0,
    "even_brightness": False,
    "peak_hold_s": 0.0,
    "normalize_window": 0,
    "scale_range": False,
    "difference_gain": DEFAULT_DIFFERENCE_GAIN,
    "gain_before_average": True,
    "average_multiplier": 1.0,
    "invert": False,
    "gamma": 1.0,
    "brightness": 0,
    "display_gain": 1.0,
    "contrast": 0.0,
    "single_field": False,
    "auto_reference": True,
    "average_frames_first": False,
    "floor_subtract": False,
}


def is_analog(info):
    """Tell whether a camera is an analog camera behind a capture adapter.

    Args: info is a CameraInfo, or None for a demo or not-yet-chosen source.
    Returns: True for a recognised capture adapter (the Sony XC-ST50 through
        a Hauppauge USB-Live2).
    """
    return (info is not None and info.kind == "capture"
            and adapter_for(info.vid, info.pid, info.model) is not None)


# Video-capture devices start with this sensitivity floor: Auto stretch never
# spans fewer grey levels, so their few levels of noise stay dark.
ANALOG_MIN_RANGE = 30


def own_settings(info):
    """Choose a camera's own starting settings, without the LabVIEW flow.

    An analog camera uses Fixed gain, as the LabVIEW ESPI 1.5 VI showed it:
    its few levels of capture noise stay dark instead of being stretched to
    full contrast. Other cameras and demo sources use Auto stretch. Every
    video-capture device gets the ANALOG_MIN_RANGE sensitivity floor for when
    Auto stretch is chosen. These are what turning the LabVIEW flow off
    returns to.

    Args: info is a CameraInfo, or None for a demo source.
    Returns: A new Settings.
    """
    capture = info is not None and info.kind == "capture"
    return Settings(scale_range=not is_analog(info),
                    min_range=ANALOG_MIN_RANGE if capture else 0)


def default_settings(info):
    """Choose the settings a camera starts with.

    An analog camera starts in the LabVIEW flow (LABVIEW_FLOW on top of its
    own settings), the known-good processing for the Sony XC-ST50. Other
    cameras and demo sources start with their own settings.

    Args: info is a CameraInfo, or None for a demo source.
    Returns: A new Settings.
    """
    settings = own_settings(info)
    return replace(settings, **LABVIEW_FLOW) if is_analog(info) else settings


@dataclass(frozen=True)
class FramePacket:
    """One processed frame ready for display, saving or recording.

    image is a 2-D uint8 array that is never modified after publication.
    gamma is the display tone still to apply (the display is always
    grayscale).
    """

    sequence: int
    image: np.ndarray
    gamma: float
    mode: str
    demo: bool

    def render(self):
        """Produce the exact colors shown on screen, on the CPU.

        Args: None.
        Returns: A grayscale uint8 image for saving or recording.
        """
        return apply_display(self.image, self.gamma)


@dataclass(frozen=True)
class ComparePacket:
    """The baseline, the current frame and their difference, in step.

    The baseline is the stored reference (Reference and Raw views) or the
    earlier frame that Previous view subtracts. Each image is a FramePacket,
    so render() gives exactly what the comparison window shows. baseline and
    difference are None until a baseline of the current frame size exists
    (and, in Previous view, while the average is still filling).
    """

    sequence: int
    current: FramePacket
    baseline: object       # FramePacket or None
    difference: object     # FramePacket or None
    baseline_info: str     # how the baseline was taken, or why there is none
    difference_info: str   # how the difference was processed
    demo: bool
    view: str = "Reference"             # "Reference" or "Previous": the baseline kind
    baseline_label: str = "Reference"   # the panel's name, e.g. "Previous (2 frames back)"


def previous_wording(s):
    """Word the Previous comparison's labels from the settings in force.

    Args: s is the Settings snapshot.
    Returns: (frames, label, averaging): "2 frames", "Previous (2 frames back)"
        and "average 4 rolling" (or "no averaging").
    """
    back = min(max(1, int(s.frames_back)), MAX_FRAMES_BACK)
    frames = f"{back} {'frame' if back == 1 else 'frames'}"
    averaging = (f"average {s.average}{' rolling' if s.rolling else ''}"
                 if s.average > 1 else "no averaging")
    return frames, f"Previous ({frames} back)", averaging


def present(image, flip, demo, tone=None):
    """Apply the last steps every published image gets: tone, flip, watermark.

    Args: image is a 2-D uint8 array; flip mirrors it left-right; demo
        stamps the dummy-mode banner (after the flip, so its text reads
        normally); tone is a display_lut table or None.
    Returns: The image to publish (the input itself when none applies).
    """
    if tone is not None:  # a LUT makes a new array; the camera frame is untouched
        image = cv2.LUT(image, tone)
    if flip:
        image = cv2.flip(image, 1)
    if demo:
        image = watermark_demo(image)
    return image


def condition_difference(difference, frame, s):
    """Run the steps before averaging: threshold, even brightness, gain, outliers.

    Args: difference is the uint8 difference image; frame is the Mono8 frame
        it came from; s is the Settings snapshot.
    Returns: (difference, labview_order): the conditioned image, and whether
        the gain was applied already (LabVIEW order: Fixed gain only), so the
        later steps must not apply it again.
    """
    if s.threshold > 0:  # raw differences below the threshold count as no change
        difference = cv2.threshold(difference, int(s.threshold) - 1, 0,
                                   cv2.THRESH_TOZERO)[1]
    if s.even_brightness:
        difference = even_out_brightness(difference, frame)
    # LabVIEW order (Fixed gain only): gain and saturate each frame,
    # then average and scale the mean by the multiplier.
    labview_order = s.gain_before_average and not s.scale_range
    if labview_order:
        difference = apply_gain(difference, s.difference_gain)
    return reject_outliers(difference, s.outlier_filter), labview_order


def floor_key(s, processor, shape):
    """Describe what a noise floor map was measured under.

    A floor map only fits images made the same way: the same frame size,
    view and baseline, camera exposure and gain, and every step before the
    subtraction. A map whose key differs from the current one is not used.

    Args: s is the Settings snapshot; processor is the ESPIProcessor (its
        reference time identifies the Reference baseline); shape is the
        frame shape.
    Returns: A hashable tuple.
    """
    labview = s.gain_before_average and not s.scale_range
    baseline = s.frames_back if s.mode == "Previous" else processor.ref_time
    return (shape, s.mode, baseline, s.exposure_us, s.gain_db, s.noise_reduction,
            s.single_field, s.threshold, s.even_brightness, s.outlier_filter,
            s.average, s.rolling, s.average_frames_first, s.smoothing,
            s.smoothing_method,
            (s.difference_gain, s.average_multiplier) if labview else None)


def finish_difference(averaged, s, measure=True, peaks=None, ranges=None,
                      now=0.0, sample=None, labview_order=False, floor=None,
                      floor_sample=None):
    """Run the steps after averaging: speckle filter, noise floor, peak hold,
    scaling, invert.

    Args: averaged is a uint8 difference image; s is the Settings snapshot;
        measure asks for the input range
        when Fixed gain or Local normalization is on (it costs a pass over
        the frame); peaks and ranges are the worker's PeakHold and
        RangeTracker, or None to leave out peak hold and the steady range
        (the comparison's single difference); now is the monotonic time;
        sample is called with the filtered image and now (the noise
        calibration); labview_order says the gain was applied before averaging;
        floor is a NoiseFloorMap result to subtract, or None; floor_sample is
        called with the filtered image and now before the floor is taken off
        (a noise floor measurement), or None.
    Returns: (image, levels): the processed image (the display gamma is still
        to apply) and (low, high, scaled_low,
        scaled_high) for the status bar (scaled_low is None under Local
        normalization), or None when not measured.
    """
    averaged = smooth_speckle(averaged, s.smoothing_method, s.smoothing)
    if floor_sample is not None:
        floor_sample(averaged, now)
    if floor is not None:
        averaged = subtract_floor(averaged, floor)
    if peaks is not None:
        averaged = peaks.update(averaged, s.peak_hold_s, now)
    if sample is not None:
        sample(averaged, now)
    levels = None
    if s.normalize_window > 0:
        image = local_normalize(averaged, s.normalize_window, s.min_range)
        if measure:
            low, high, _, _ = cv2.minMaxLoc(averaged)
            levels = (int(low), int(high), None, None)
    elif s.scale_range:
        low, high = stretch_levels(averaged, s.clip_percent)
        low, high = (ranges or RangeTracker()).update(
            low, high, s.min_range, s.steady_range_s, now)
        image = apply_levels(averaged, low, high)
        levels = (low, high, *((0, 255) if high > low else (0, 0)))
    else:
        image = (averaged if labview_order
                 else apply_gain(averaged, s.difference_gain))
        if measure:
            low, high, _, _ = cv2.minMaxLoc(averaged)
            top, bottom, _, _ = cv2.minMaxLoc(image)
            levels = (int(low), int(high), int(top), int(bottom))
    if s.invert:
        image = cv2.bitwise_not(image)
    return image, levels


def range_text(levels, s):
    """Describe finish_difference's levels for the status bar.

    Args: levels is its fourth result (not None); s is the Settings snapshot.
    Returns: Text such as "Input: 0–12    Scaled: 0–255".
    """
    low, high, scaled_low, scaled_high = levels
    if scaled_low is None:
        return f"Input: {low}–{high}    Local normalization {s.normalize_window} px"
    return f"Input: {low}–{high}    Scaled: {scaled_low}–{scaled_high}"


def reference_text(processor, now):
    """Describe the stored reference for the log and the comparison window.

    Args: processor is the ESPIProcessor; now is the monotonic time.
    Returns: Text such as "average of 8 frames, noise reduction 0, 3 s old",
        or "none yet".
    """
    if processor.ref is None:
        return "none yet"
    frames = ("1 frame" if processor.ref_frames == 1 else
              f"average of {processor.ref_frames} frames")
    return (f"{frames}, noise reduction {processor.ref_noise:g}, "
            f"{max(0.0, now - processor.ref_time):.0f} s old")


def view_hint(s, processor, floor_note=None):
    """Choose the status-bar hint for the current view and reference state.

    Args: s is the Settings snapshot; processor is the ESPIProcessor;
        floor_note is the noise floor's state (Engine._floor_note), or None.
    Returns: One short sentence (no longer than the Qt LONGEST_HINT).
    """
    capture = processor.ref_capture
    if capture is not None:
        kept = "; the old one stays in use" if processor.ref is not None else ""
        return f"Capturing reference: {capture.count}/{capture.target} frames{kept}"
    if s.raw_difference:
        return "Raw difference: live minus previous raw frame, nothing else"
    if floor_note is not None:
        return floor_note
    if s.mode == "Raw":
        return "Raw view: camera pixels, with gamma; no stretch"
    if s.mode == "Reference" and processor.ref is None:
        return "Reference: press F or Set reference to capture a baseline"
    if s.mode == "Reference" and processor.ref_noise != s.noise_reduction:
        return (f"Reference was taken at noise reduction {processor.ref_noise:g}; "
                "press F to retake")
    if s.mode == "Reference" and s.ref_refresh_s > 0:
        return f"Reference refreshes every {s.ref_refresh_s:.1f} s to cancel slow drift"
    return "ESPI difference view; min/max range is before gamma"


class WindowsPreviewTimer:
    """Temporarily request precise Windows sleeps for frame pacing."""

    def __init__(self):
        """Prepare timer state without changing the system yet.

        Args: None.
        Returns: None.
        """
        self.winmm = None
        self.active = False

    def start(self):
        """Request 1 ms sleep resolution for this Windows process.

        Args: None.
        Returns: True if Windows accepted the request, else False.
        """
        if os.name != "nt":
            return False
        try:
            self.winmm = ctypes.WinDLL("winmm")
            self.active = self.winmm.timeBeginPeriod(1) == 0
        except OSError:
            self.winmm = None
        return self.active

    def stop(self):
        """Restore the previous Windows timer-resolution request.

        Args: None.
        Returns: None.
        """
        if self.active:
            self.winmm.timeEndPeriod(1)
            self.active = False


class Engine:
    """Run camera or demo acquisition, ESPI processing and capture off the UI."""

    def __init__(self, log, serial=None, tag="", info=None):
        """Create an idle engine.

        Args: log is an AppLog or PrefixedLog (write, debug, info, warning,
            error, exception); serial selects a FLIR camera by serial number
            (None: the first camera); tag is added to saved file names to
            tell cameras apart ("" for none); info, when given, is the
            CameraInfo to open instead (a FLIR camera or a video-capture
            device).
        Returns: None.
        """
        self.log = log
        self.serial = serial
        self.tag = tag
        self.info = info
        self.camera_info = None  # CameraInfo of the opened camera
        self._open_after = None  # (predecessor engine,) when opening on the worker
        self.events = queue.SimpleQueue()
        self._settings = default_settings(info)
        self._settings_lock = threading.Lock()
        self._actions = queue.SimpleQueue()
        self._calibration = None  # a running NoiseCalibration (worker thread only)
        self._floor = None          # (floor_key, float32 map) last measured (worker only)
        self._floor_measure = None  # (floor_key or None, running NoiseFloorMap) (worker only)
        self._saves_waiting = 0  # frame saves queued or being written
        self._saves_lock = threading.Lock()
        self._packet = None
        self._compare_packet = None
        self._compare = threading.Event()  # set while a comparison window is open
        self._square_request = None  # (size, center) while the Pixels window is open
        self._square_packet = None
        self._square_cut = None  # the request the newest square was cut for (worker)
        self._square_time = 0.0
        self._packet_lock = threading.Lock()
        self._status = {
            "demo": False, "camera": "Camera values: waiting for connection",
            "range": "Input: —    Scaled: —",
            "hint": "Raw view: camera pixels, with gamma; no stretch",
            "processing": "Processing: —", "record": "Recording: off",
            "camera_fps": "Camera: — fps",
            "processed_fps": 0.0,
        }
        self._status_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.camera = None
        self.demo_source = None
        self.preview_timer = WindowsPreviewTimer()

    # ----- setup (front-end thread, before start) -----------------------

    @property
    def demo(self):
        """Report whether the engine is producing synthetic video.

        Args: None.
        Returns: True in demo mode.
        """
        return self.demo_source is not None

    def open_camera(self):
        """Open the camera chosen for this engine; call before start().

        The camera is the first FLIR camera, or the video device given as
        info when this engine belongs to a capture channel.

        Args: None.
        Returns: None. Raises if PySpin, the device or a camera is
            unavailable; the engine is then unchanged and use_demo() may be
            called.
        """
        settings = self.settings()
        attempt = (sources.make_session(
                       self.log, self.info, settings.exposure_us,
                       settings.gain_db, settings.fps)
                   if self.info is not None else
                   CameraSession(self.log, self.serial, settings.exposure_us,
                                 settings.gain_db, settings.fps))
        attempt.open()
        self._attach_camera(attempt)

    def open_on_start(self, after=None):
        """Open the camera on the worker thread once start() is called.

        Use this to add a camera while the window runs, so a slow Init does
        not freeze the display. A failure is reported as an "open_failed"
        event followed by "stopped".

        Args: after is an earlier Engine for the same camera that is still
            stopping; the worker waits for it to release the camera first.
        Returns: None.
        """
        self._open_after = (after,)

    def use_demo(self, label=None):
        """Switch the source to synthetic video; call before start().

        Args: label is optional text drawn into the pattern, so several
            synthetic cameras look different.
        Returns: None.
        """
        self.demo_source = DemoSource(label=label, log=self.log)
        self._set_status(demo=True, camera=(
            "DUMMY MODE ACTIVE — simulated ESPI signal, ideal sensor; exposure, "
            "gain and FPS are simulated"))
        self.log.write("DUMMY MODE ACTIVE: simulated ESPI signal (ideal sensor); "
                       "no camera acquired.")

    def _attach_camera(self, session):
        """Make an opened camera session the frame source.

        Args: session is an opened CameraSession or CaptureSession.
        Returns: None; publishes the model as a "camera_identity" event, the
            device limits as a "camera_limits" event and, for a video device
            the viewer does not recognise, a "camera_report" event.
        """
        self.camera = session
        self.camera_info = session.info
        self.demo_source = None
        identity = session.identity
        reporter = error_reports.current_reporter()
        if reporter is not None:
            reporter.update_context(serials=[session.info.serial,
                                            getattr(identity, "serial", "")])
        self.log.write("Camera: " + " | ".join(identity.description()))
        for warning in identity.warnings():
            self.log.warning(f"Camera note: {warning}")
        self.events.put(("camera_identity", identity))
        self._set_status(demo=False, camera=session.describe(session.read_values()))
        self.events.put(("camera_limits", session.limits()))
        if session.info.kind == "capture" and not identity.supported:
            self._report_camera(session)

    def _report_camera(self, session):
        """Queue a camera report for an unrecognised video device.

        Args: session is the opened CaptureSession.
        Returns: None; a failure to read details sends the report without them.
        """
        try:
            details = session.read_details()
        except Exception as exc:
            self.log.write(f"Could not read the device details: {exc}")
            details = ()
        self.events.put(("camera_report",
                         capture_report(session.info, session.identity, details)))

    # ----- thread control ----------------------------------------------

    def start(self):
        """Start the worker thread.

        Args: None.
        Returns: None. Raises RuntimeError if no source was chosen.
        """
        if (self.camera is None and self.demo_source is None
                and self._open_after is None):
            raise RuntimeError("Call open_camera(), open_on_start() or use_demo() "
                               "before start().")
        if self.preview_timer.start():
            self.log.write("Windows timer: 1 ms resolution while the app is open.")
        self._thread = threading.Thread(target=self._run, name="espi-engine",
                                        daemon=True)
        self._thread.start()

    def stop(self):
        """Ask the worker to finish recordings, release the camera and exit.

        Args: None.
        Returns: None; use wait() to block until it has finished.
        """
        self._stop.set()

    def wait(self, timeout):
        """Wait for the worker to exit.

        Args: timeout is the maximum wait in seconds.
        Returns: True if the worker has exited (or never started).
        """
        if self._thread is None:
            if self.camera is not None:  # opened but never started
                self.camera.close()
                self.camera = None
            self.preview_timer.stop()
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    # ----- front-end API (any thread) ------------------------------------

    def settings(self):
        """Return a snapshot of the current settings.

        Args: None.
        Returns: A Settings copy the caller may keep.
        """
        with self._settings_lock:
            return replace(self._settings)

    def update_settings(self, **changes):
        """Change one or more settings; the worker applies them next frame.

        Args: changes are Settings field names and new values.
        Returns: None. Raises TypeError for unknown fields.
        """
        with self._settings_lock:
            self._settings = replace(self._settings, **changes)
        self.log.debug(f"Settings: {changes}")

    def request(self, action, **payload):
        """Queue a one-shot command for the worker.

        Args: action is "save" (payload title, notes; optionally packet, a
            frozen display's FramePacket to save instead of the newest one,
            and frozen_at, its datetime), "save_compare" (payload packet, a
            ComparePacket, and title, notes), "reference" (captures the
            current frame, or starts averaging the next ref_frames frames),
            "reset_previous", "reset_average", "calibrate_noise" (measure a
            still scene's difference noise for 2 s, then set min_range and
            send a "noise_calibrated" event), "measure_floor" (measure each
            pixel's still-scene difference level, the map Subtract noise
            floor takes off), "switch_live" (the
            "switched_live" event then carries every detected CameraInfo) or
            "camera_command" (payload: phone-only settings such as
            lens_position, width and height, crop_width and crop_height,
            max_send_fps or refocus, sent to a source that takes them, the
            iPhone; ignored by other cameras) or "drive" (payload drive, a
            daq.Drive: what the DAQ outputs produce, for the simulated
            drumhead in demo mode; ignored by cameras).
        Returns: None.
        """
        if action in ("save", "save_compare"):
            with self._saves_lock:
                self._saves_waiting += 1
        self._actions.put((action, payload))

    def save_pending(self):
        """Report whether a frame save is queued or being written.

        Args: None.
        Returns: True until the worker has finished that save.
        """
        with self._saves_lock:
            return self._saves_waiting > 0

    def latest_packet(self):
        """Return the newest published frame.

        Args: None.
        Returns: A FramePacket, or None before the first frame.
        """
        with self._packet_lock:
            return self._packet

    def set_compare(self, on):
        """Turn the reference comparison images on or off.

        The worker builds them only while this is on, so a closed comparison
        window costs the frame loop nothing.

        Args: on is True while a comparison window shows this engine.
        Returns: None.
        """
        if on:
            self._compare.set()
        else:
            self._compare.clear()
            with self._packet_lock:
                self._compare_packet = None

    def latest_compare(self):
        """Return the newest reference comparison.

        Args: None.
        Returns: A ComparePacket, or None while comparison is off or before
            the first one.
        """
        with self._packet_lock:
            return self._compare_packet

    def set_square(self, size, center=None):
        """Ask for the pixel square, or stop asking.

        The worker cuts it only while asked, so a closed Pixels window costs
        the frame loop nothing.

        Args: size is N for an N×N square, or None to stop; center is the
            (x, y) displayed pixel to center it on, or None for the image
            center.
        Returns: None.
        """
        with self._packet_lock:
            self._square_request = None if size is None else (int(size), center)
            self._square_packet = None

    def latest_square(self):
        """Return the newest pixel square.

        Args: None.
        Returns: A SquarePacket, or None while none is asked for or before the
            first one.
        """
        with self._packet_lock:
            return self._square_packet

    def status(self):
        """Return a snapshot of the status texts and counters.

        Args: None.
        Returns: A dict with keys demo, camera, range, hint, processing,
            record, camera_fps and processed_fps.
        """
        with self._status_lock:
            return dict(self._status)

    def _set_status(self, **fields):
        """Update status fields under the lock.

        Args: fields are status keys and values.
        Returns: None.
        """
        with self._status_lock:
            self._status.update(fields)

    # ----- worker ---------------------------------------------------------

    def _run(self):
        """Worker entry point: run the frame loop, then clean up.

        Args: None.
        Returns: None; fatal errors are logged with their traceback, reported
            (error_reports) and sent as a "fatal" event whose payload is
            (message, fingerprint or None).
        """
        self._recorder = None
        self._retired = []
        try:
            if self._open_after is not None and not self._open_deferred():
                return
            self._loop()
        except Exception as exc:
            report = error_reports.caught(self.log, exc, "engine", "Stopped", fatal=True)
            self.events.put(("fatal", (f"{type(exc).__name__}: {exc}",
                                       report.fingerprint if report else None)))
        finally:
            try:
                if self._recorder is not None:
                    self._recorder.stop()
                    self._retired.append(self._recorder)
                    self._recorder = None
                deadline = time.monotonic() + RECORDER_FINISH_TIMEOUT_S
                for old in self._retired:
                    old.done.wait(max(0.0, deadline - time.monotonic()))
                    if old.done.is_set():
                        self._recording_finished(old)
                    else:
                        self.log.warning(
                            f"Recording did not finish within 5 seconds: {old.path}")
                self._retired = []
            finally:
                if self.camera is not None:
                    self.camera.close()
                    self.camera = None
                self.preview_timer.stop()
                self.events.put(("stopped", None))

    def _open_deferred(self):
        """Open the camera requested by open_on_start(), on the worker thread.

        Args: None.
        Returns: True when the camera is open; False after an "open_failed"
            event (the worker then stops).
        """
        (after,) = self._open_after
        self._open_after = None
        if after is not None and not after.wait(PREDECESSOR_TIMEOUT_S):
            message = "the camera is still in use by its previous session"
            self.log.warning(f"Could not open the camera: {message}")
        else:
            try:
                self.open_camera()
                return True
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                error_reports.caught(self.log, exc, "camera_open", "Could not open the camera")
        self.events.put(("open_failed", message))
        return False

    def _loop(self):
        """Grab, process, publish and record frames until stop() is called.

        Args: None.
        Returns: None.
        """
        processor = ESPIProcessor()
        ranges = RangeTracker()
        peaks = PeakHold()
        applied = self.settings()  # the camera was opened at these values
        last_setting_time = 0.0
        last_mode = applied.mode
        record_signature = None
        resulting_fps = None
        last_range_time = 0.0
        last_rolling = applied.rolling
        last_refresh = applied.ref_refresh_s
        last_single_field = applied.single_field
        last_noise = applied.noise_reduction
        last_ref_frames = applied.ref_frames
        last_frames_first = applied.average_frames_first
        last_shape = None
        last_raw_difference = applied.raw_difference
        raw_previous = None  # the last unprocessed frame, for Raw difference
        last_compare = 0.0
        last_readback = time.monotonic()
        last_error_log = 0.0
        last_valid = time.monotonic()
        incomplete = 0
        window_start = time.monotonic()
        window_frames = 0
        window_work = 0.0
        last_timing_log = 0.0
        sequence = 0
        frame = None
        demo_next = time.monotonic()
        gaps = FrameGapMonitor(self.log)
        counters = StreamCounterMonitor(self.log)
        camera_sharpness = SharpnessMonitor(self.log, "camera frame")
        shown_sharpness = SharpnessMonitor(self.log, "displayed frame")
        s = applied

        def context():
            """Describe the settings and rates in force, for diagnostic lines.

            Args: None.
            Returns: One line of text.
            """
            rate = "n/a" if resulting_fps is None else f"{resulting_fps:.1f}"
            return f"{settings_summary(s)} camera_fps={rate}"
        self.log.write("Streaming. Raw shows camera pixels with gamma only; "
                       "Previous and Reference show ESPI differences.")

        while not self._stop.is_set():
            s = self.settings()
            self._reap_recorders()
            signature = ((RAW_DIFFERENCE, 1, False) if s.raw_difference else
                         (s.mode, s.average if s.mode != "Raw" else 1,
                          s.rolling and not s.average_frames_first))
            recorder = self._recorder
            if recorder is not None and (
                    not s.record or signature != record_signature
                    or recorder.done.is_set()):
                if recorder.error is not None or recorder.done.is_set():
                    self.update_settings(record=False)
                    self.events.put(("record_off", None))
                recorder.stop()
                self._retired.append(recorder)
                self._recorder = None
                record_signature = None
                self._set_status(record="Recording: finishing file")
            if s.raw_difference != last_raw_difference:
                # The processing state was not fed meanwhile: start it over.
                raw_previous = None
                processor.reset_avg()
                processor.reset_frame_average()
                processor.reset_previous()
                ranges.reset()
                peaks.reset()
                self.log.write("Raw difference on: |live - previous| of raw frames, "
                               "every other processing step off"
                               if s.raw_difference else "Raw difference off")
                last_raw_difference = s.raw_difference
            if s.mode != last_mode:
                processor.reset_avg()
                processor.reset_frame_average()
                ranges.reset()
                peaks.reset()
                if s.mode == "Previous":
                    processor.reset_previous()
                self.log.write(f"View: {s.mode}")
                last_mode = s.mode
            if s.rolling != last_rolling:
                processor.reset_avg()
                last_rolling = s.rolling
            if s.average_frames_first != last_frames_first:
                processor.reset_avg()  # the difference now compares other frames
                processor.reset_frame_average()
                processor.reset_previous()
                peaks.reset()
                self.log.write("Averaging: camera frames, then the difference"
                               if s.average_frames_first else
                               "Averaging: differences")
                last_frames_first = s.average_frames_first
            if s.single_field != last_single_field:
                processor.reset_avg()  # field geometry changed: start over
                processor.reset_frame_average()
                peaks.reset()
                processor.reset_previous()
                processor.clear_reference()
                self.log.write("Fields: single field (even rows, line-doubled)"
                               if s.single_field else "Fields: full interlaced frames")
                last_single_field = s.single_field
            if s.ref_refresh_s != last_refresh or s.ref_frames != last_ref_frames:
                if s.ref_refresh_s != last_refresh:
                    self.log.write(f"Reference auto-refresh: every {s.ref_refresh_s:.1f} s"
                                   if s.ref_refresh_s > 0 else "Reference auto-refresh: off")
                if s.ref_frames != last_ref_frames:
                    self.log.write(f"Reference frames: {s.ref_frames}")
                self._note_refresh_overlap(s, resulting_fps)
                last_refresh, last_ref_frames = s.ref_refresh_s, s.ref_frames
            if s.noise_reduction != last_noise:
                self._noise_changed(processor, last_noise, s.noise_reduction)
                last_noise = s.noise_reduction
            self._set_status(hint=view_hint(
                s, processor, self._floor_note(s, processor, last_shape)))

            work_started = time.monotonic()
            if self.demo:
                if s.demo_object != self.demo_source.object:
                    self.demo_source.set_object(s.demo_object)
                    self.log.write(f"Simulated object: {s.demo_object}")
                frame = self.demo_source.next_frame(s.exposure_us, s.gain_db, s.fps)
            else:
                grabbed, state = self.camera.grab(CAMERA_TIMEOUT_MS)
                if state != "ok":
                    if state == "incomplete":
                        incomplete += 1
                        gaps.incomplete(time.monotonic())
                        if incomplete % 30 == 1:
                            self.log.warning(f"Incomplete frames: {incomplete}")
                    elif time.monotonic() - last_error_log >= 5.0:
                        self.log.warning(f"Camera timeout or acquisition error: {grabbed}")
                        last_error_log = time.monotonic()
                    if time.monotonic() - last_valid >= 1.0:
                        self._set_status(processing="Waiting for camera frame",
                                         processed_fps=0.0)
                    processor, frame = self._handle_actions(processor, None)
                    continue
                frame = grabbed
            if frame.ndim != 2 or frame.dtype != np.uint8:
                raise RuntimeError("Expected a Mono8 image; check camera PixelFormat.")
            if (s.single_field and not s.raw_difference and
                    self.camera is not None and self.camera.interlaced):
                frame = single_field(frame)

            now = time.monotonic()
            gaps.frame_ok(now, 1.0 / max(1.0, resulting_fps or s.fps), context)
            camera_sharpness.observe(frame, now, context)
            if not s.raw_difference:
                frame = processor.denoise(frame, denoise_strength(s.noise_reduction))
            now = time.monotonic()
            last_valid = now
            if not self.demo:
                if now - last_readback >= READBACK_INTERVAL_S:
                    values = self.camera.read_values()
                    resulting_fps = values["resulting"]
                    take = getattr(self.camera, "take_settings_update", None)
                    changed = take() if take is not None else None
                    if changed:  # set on the phone itself, or clamped by it
                        self.events.put(("camera_settings", changed))
                    counters.update(self.camera.read_stream_counters(), context)
                    self._set_status(
                        camera=self.camera.describe(values),
                        camera_fps=("Camera: n/a" if resulting_fps is None else
                                    f"Camera: {resulting_fps:.1f} fps"))
                    last_readback = now
                if (now - last_setting_time >= SETTING_INTERVAL_S and
                        (s.exposure_us, s.gain_db, s.fps) !=
                        (applied.exposure_us, applied.gain_db, applied.fps)):
                    if s.exposure_us != applied.exposure_us:
                        self.camera.set_float("ExposureTime", s.exposure_us)
                    if s.gain_db != applied.gain_db:
                        self.camera.set_float("Gain", s.gain_db)
                    if s.fps != applied.fps:
                        self.camera.set_float("AcquisitionFrameRate", s.fps)
                    applied = s
                    last_setting_time = now

            reference_view = s.mode == "Reference" and not s.raw_difference
            if reference_view and s.auto_reference and processor.ref is None:
                processor.set_reference(frame, now)  # as the LabVIEW VI did at start
                processor.reset_avg()
                self.log.write("Captured reference frame (automatic).")
            if reference_view and s.ref_refresh_s > 0 and (
                    processor.ref_capture is None) and (
                    processor.ref is None or
                    now - processor.ref_time >= s.ref_refresh_s):
                if s.ref_frames <= 1:
                    processor.set_reference(frame, now, s.noise_reduction)
                    if not s.rolling:  # keep each block average on one baseline
                        processor.reset_avg()
                else:  # the old reference stays in use while the new one builds
                    processor.start_reference(s.ref_frames, s.noise_reduction,
                                              now, manual=False)
            if processor.ref_capture is not None:
                self._reference_frame(processor, frame, s)

            update_range = now - last_range_time >= RANGE_INTERVAL_S
            image = None
            measured = frame  # the frame the difference uses (an average, or frame)
            gamma = s.gamma  # every view shows the display gamma ...
            earlier = raw_previous
            if s.raw_difference:
                raw_previous = frame.copy()  # outlives this iteration's frame
                image = raw_difference(frame, earlier)
                gamma = 1.0  # ... except Raw difference, which applies nothing
                self._cut_square(frame if image is None else image, image is not None,
                                 "Raw difference", False, now)
                self._drop_calibration("Raw difference")
                if update_range:
                    self._set_status(range="Raw difference: |live - previous|, "
                                     "no gain, filters or scaling")
                    last_range_time = now
            elif s.mode == "Raw":
                image = frame  # gamma curve only; Raw skips stretch
                self._cut_square(frame, True, "Raw", s.flip_horizontal, now)
                self._drop_calibration("Raw view")
                if update_range:
                    self._set_status(range="Raw Mono8: no brightness scaling")
                    last_range_time = now
            else:
                # Average frames first: the camera frames are block-averaged
                # and the mean is subtracted, so Average N lowers the noise
                # itself instead of averaging rectified noise into a floor.
                measured = (processor.average_frames(frame, s.average)
                            if s.average_frames_first else frame)
                averaged = None
                if measured is not None:
                    last_shape = measured.shape
                    unconditioned = processor.difference(measured, s.mode,
                                                         s.frames_back)
                    self._cut_square(unconditioned, (
                        processor.previous is not None if s.mode == "Previous" else
                        processor.ref is not None and
                        processor.ref.shape == measured.shape),
                        s.mode, s.flip_horizontal, now)
                    difference, labview_order = condition_difference(
                        unconditioned, measured, s)
                    count = 1 if s.average_frames_first else s.average
                    scale = None
                    if labview_order:
                        scale = s.average_multiplier
                        if count > 1 and not s.rolling:
                            # The VI's loop stops after iteration N, so it sums N + 1
                            # frames and divides by N: one frame more, (N+1)/N brighter.
                            count += 1
                            scale = s.average_multiplier * count / s.average
                    averaged = (processor.rolling_average(difference, count, scale)
                                if s.rolling else
                                processor.average(difference, count, scale))
                if averaged is not None:
                    floor, floor_sample = self._floor_use(s, processor, measured.shape)
                    image, levels = finish_difference(
                        averaged, s, update_range, peaks=peaks,
                        ranges=ranges, now=now, sample=self._sample_noise,
                        labview_order=labview_order, floor=floor,
                        floor_sample=floor_sample)
                    if update_range:
                        self._set_status(range=range_text(levels, s))
                        last_range_time = now

            if image is not None:
                shown_sharpness.observe(image, now, context)
                if s.raw_difference:  # no flip or display tone either
                    image = present(image, False, self.demo)
                else:  # every view; before the watermark so its text reads normally
                    image = present(image, s.flip_horizontal, self.demo,
                                    display_lut(s.display_gain, s.contrast, s.brightness))
                sequence += 1
                packet = FramePacket(sequence, image, gamma,
                                     signature[0], self.demo)
                with self._packet_lock:
                    self._packet = packet
                record_signature = self._feed_recorder(
                    s, signature, record_signature, packet, resulting_fps)
                if (self._compare.is_set() and
                        now - last_compare >= COMPARE_INTERVAL_S):
                    if s.raw_difference:
                        self._publish_raw_compare(frame, earlier, packet)
                    else:
                        self._publish_compare(processor, measured, s, packet, now)
                    last_compare = now
            elif (s.mode == "Previous" and self._compare.is_set() and
                    now - last_compare >= COMPARE_INTERVAL_S and
                    self._publish_compare_filling(processor, frame, s, sequence)):
                last_compare = now  # the average is still filling
            window_frames += 1
            window_work += time.monotonic() - work_started

            elapsed = time.monotonic() - window_start
            if elapsed >= 1.0:
                if self.demo:  # no sensor; the synthetic source runs at the request
                    self._set_status(camera_fps=f"Camera: {s.fps:.1f} fps (simulated)")
                processing = (f"Processed: {window_frames / elapsed:.1f} fps    "
                              f"Work: {1000 * window_work / max(1, window_frames):.1f}"
                              " ms/frame")
                self._set_status(processed_fps=window_frames / elapsed,
                                 processing=processing)
                if time.monotonic() - last_timing_log >= TIMING_LOG_INTERVAL_S:
                    self.log.debug(f"Frame timing: {processing}")
                    last_timing_log = time.monotonic()
                window_start = time.monotonic()
                window_frames = 0
                window_work = 0.0

            processor, frame = self._handle_actions(processor, frame)
            if self.demo:  # pace synthetic video; the camera paces itself
                # A fixed schedule keeps sleep overshoot from accumulating.
                interval = 1.0 / max(1.0, s.fps)
                demo_next += interval
                now = time.monotonic()
                if demo_next < now - interval:
                    demo_next = now  # fell behind; restart the schedule
                self._stop.wait(max(0.0, demo_next - now))

    def _reference_frame(self, processor, frame, s):
        """Add a frame to the reference capture in progress; log when it ends.

        Args: processor is the ESPIProcessor holding the capture; frame is
            the frame after single-field and noise reduction; s is the
            Settings snapshot.
        Returns: None.
        """
        done = processor.add_reference_frame(frame)
        if done is None:
            return
        if done.manual or not s.rolling:  # keep each block average on one baseline
            processor.reset_avg()
        if done.manual:
            self.log.write(f"Captured reference frame (average of {done.count}) "
                           f"at noise reduction {done.noise_reduction:g}.")

    def _note_refresh_overlap(self, s, resulting_fps):
        """Log when an averaged reference takes longer than the refresh time.

        Args: s is the Settings snapshot; resulting_fps is the camera's
            measured rate, or None (the requested rate is used).
        Returns: None.
        """
        if s.ref_refresh_s <= 0 or s.ref_frames <= 1:
            return
        seconds = s.ref_frames / max(1.0, resulting_fps or s.fps)
        if seconds > s.ref_refresh_s:
            self.log.write(
                f"Reference note: averaging {s.ref_frames} frames takes about "
                f"{seconds:.1f} s, longer than the {s.ref_refresh_s:.1f} s refresh; "
                "references are captured back to back.")

    def _noise_changed(self, processor, old, new):
        """React to a Noise reduction change while a reference exists.

        A capture in progress is cancelled, so frames taken at two settings
        are never mixed. A stored reference is kept, but the log says it was
        taken at another setting (once, when the settings first differ).

        Args: processor is the ESPIProcessor; old and new are the slider values.
        Returns: None.
        """
        capture = processor.ref_capture
        if capture is not None:
            processor.ref_capture = None
            self.log.write("Reference capture cancelled: noise reduction changed "
                           "while it was running. " + (
                               "Press F to start again." if capture.manual else
                               "The next refresh starts a new one."))
        elif processor.ref is not None and processor.ref_noise == old:
            self.log.write(f"Noise reduction is now {new:g}, but the reference was "
                           f"taken at {old:g}. Press F to retake it.")

    def _cut_square(self, image, ready, view, flip, now):
        """Publish the pixel square while the Pixels window asks for one.

        Args: image is the frame's raw absolute difference (the frame itself
            in Raw view), before the threshold, gain, averaging and scaling;
            ready is False while there is no baseline to subtract (image then
            only gives the size); view names what is measured (the Settings
            mode, or "Raw difference"); flip is whether the display mirrors
            the image; now is the monotonic time.
        Returns: None. A new SquarePacket is published at most every
            SQUARE_INTERVAL_S, and at once when the request changes.
        """
        request = self._square_request  # (size, center); one read, any thread set it
        if request is None:
            return
        if request is self._square_cut and now - self._square_time < SQUARE_INTERVAL_S:
            return
        size, center = request
        left, top, size = square_area(center, size, image.shape)
        values = square_values(image, left, top, size, flip) if ready else None
        packet = SquarePacket(values, view, left, top, size, self.demo)
        with self._packet_lock:
            if self._square_request is request:  # not turned off meanwhile
                self._square_packet = packet
        self._square_cut, self._square_time = request, now

    def _publish_compare(self, processor, frame, s, packet, now):
        """Publish the baseline, the current frame and their difference.

        Args: processor is the ESPIProcessor; frame is this iteration's frame
            (after single-field and noise reduction); s is the Settings
            snapshot; packet is the FramePacket just published; now is the
            monotonic time.
        Returns: None. In Reference and Previous views the difference is the
            published packet itself (with its averaging) and, in Previous
            view, the baseline is the earlier frame that difference
            subtracted. In Raw view the baseline is the reference and the
            difference is one difference from it, processed like Reference
            view but without averaging.
        """
        flip = s.flip_horizontal
        tone = display_lut(s.display_gain, s.contrast, s.brightness)
        current = FramePacket(packet.sequence, present(frame, flip, self.demo, tone),
                              s.gamma, "Current", self.demo)
        if s.mode == "Previous":
            compare = self._previous_compare(processor, s, packet.sequence, packet,
                                             current, flip, tone)
            with self._packet_lock:
                self._compare_packet = compare
            return
        ref = processor.ref
        info = reference_text(processor, now)
        capture = processor.ref_capture
        if capture is not None:
            info = f"capturing {capture.count}/{capture.target}; in use: {info}"
        reference = difference = None
        if s.mode == "Reference":
            averaging = (f"average {s.average}{' rolling' if s.rolling else ''}"
                         if s.average > 1 else "no averaging")
            detail = f"the Reference view, {averaging}"
        else:
            detail = f"one difference, no averaging (the view is {s.mode})"
        if ref is not None and ref.shape == frame.shape:
            reference = FramePacket(packet.sequence, present(ref, flip, self.demo, tone),
                                    s.gamma, "Reference frame", self.demo)
            if s.mode == "Reference":
                difference = packet
            else:
                single, labview_order = condition_difference(
                    cv2.absdiff(frame, ref), frame, s)
                image, _ = finish_difference(
                    single, s, measure=False, now=now,
                    labview_order=labview_order)
                difference = FramePacket(packet.sequence,
                                         present(image, flip, self.demo, tone),
                                         s.gamma, "Reference", self.demo)
        compare = ComparePacket(packet.sequence, current, reference, difference,
                                info, detail, self.demo)
        with self._packet_lock:
            self._compare_packet = compare

    def _publish_raw_compare(self, frame, earlier, packet):
        """Publish the two raw frames Raw difference subtracts, and their difference.

        Args: frame is the live raw frame; earlier is the raw frame before
            it; packet is the Raw difference FramePacket just published.
        Returns: None.
        """
        def raw(image, name):
            """Wrap one unprocessed frame for a comparison panel.

            Args: image is a 2-D uint8 frame; name is the panel's mode text.
            Returns: A FramePacket (gamma 1, demo watermark only).
            """
            return FramePacket(packet.sequence, present(image, False, self.demo),
                               1.0, name, self.demo)
        compare = ComparePacket(
            packet.sequence, raw(frame, "Current"), raw(earlier, "Previous frame"),
            packet, "the raw frame before the live one, unprocessed",
            "Raw difference: |live - previous|, no processing", self.demo,
            "Previous", "Previous raw frame")
        with self._packet_lock:
            self._compare_packet = compare

    def _drop_calibration(self, view):
        """Cancel a running noise calibration or floor measurement in a view
        that has no difference to sample.

        Args: view names the view in force, for the warning.
        Returns: None.
        """
        if self._calibration is not None:
            self._calibration = None
            self.log.warning(f"Noise calibration needs the Previous or Reference "
                             f"view, not {view}; switch views and try again.")
        if self._floor_measure is not None:
            self._floor_measure = None
            self.log.warning(f"Measuring the noise floor needs the Previous or "
                             f"Reference view, not {view}; switch views and try again.")

    def _publish_compare_filling(self, processor, frame, s, sequence):
        """Publish the Previous comparison while its first average is filling.

        Block averaging produces no difference until N frames have been
        summed, so the window would stay empty. This publishes the live
        previous and current frames with no difference and a count, but only
        until a complete set exists under the current Frames back and Average
        settings: after that the last complete set stays on screen, still in
        step, until the next average is done.

        Args: processor is the ESPIProcessor; frame is this iteration's frame
            (after single-field and noise reduction); s is the Settings
            snapshot; sequence is the newest published frame's number.
        Returns: True when a packet was published.
        """
        newest = self.latest_compare()
        _, label, averaging = previous_wording(s)
        if (newest is not None and newest.view == "Previous" and
                newest.difference is not None and newest.baseline_label == label and
                newest.difference_info == f"the Previous view, {averaging}"):
            return False  # a complete set under these settings
        flip = s.flip_horizontal
        tone = display_lut(s.display_gain, s.contrast, s.brightness)
        current = FramePacket(sequence, present(frame, flip, self.demo, tone),
                              s.gamma, "Current", self.demo)
        compare = self._previous_compare(processor, s, sequence, None, current,
                                         flip, tone)
        with self._packet_lock:
            self._compare_packet = compare
        return True

    def _previous_compare(self, processor, s, sequence, packet, current, flip, tone):
        """Build the Previous view's comparison from what its difference used.

        The previous panel is the exact frame the processor subtracted in
        this iteration (ESPIProcessor.previous), which already went through
        single field and noise reduction like the current frame.

        Args: processor is the ESPIProcessor; s is the Settings snapshot;
            sequence numbers the images; packet is the Previous view's
            published FramePacket, or None while the average is still
            filling; current is the Current panel's FramePacket; flip and
            tone are the display flip and display_lut table.
        Returns: The ComparePacket. The difference is the published packet
            itself (with its averaging); it and the previous frame are None
            while no earlier frame exists, and the difference is None while
            the average fills.
        """
        frames, label, averaging = previous_wording(s)
        previous = processor.previous
        baseline = difference = None
        if previous is None:
            info = f"none yet: the view needs {frames} of history"
        else:
            baseline = FramePacket(sequence, present(previous, flip, self.demo, tone),
                                   s.gamma, "Previous frame", self.demo)
            info = f"the frame the view subtracts, {frames} back"
            if packet is not None:
                difference = packet
        if packet is None:
            target = processor.avg_target or s.average
            detail = (f"averaging: {processor.avg_count} of {target} frames; the "
                      "difference appears when the average is complete")
        else:
            detail = f"the Previous view, {averaging}"
        return ComparePacket(sequence, current, baseline, difference, info, detail,
                             self.demo, "Previous", label)

    def _feed_recorder(self, s, signature, record_signature, packet, resulting_fps):
        """Start a recording or submit the newest frame to the active one.

        Args: s is the settings snapshot; signature is (mode, average,
            rolling);
            record_signature is the active recording's signature; packet is
            the newest frame; resulting_fps is the camera's measured rate.
        Returns: The (possibly new) active recording signature.
        """
        if not s.record:
            return record_signature
        if self._recorder is not None:
            self._recorder.submit(packet.render())
            return record_signature
        source_fps = resulting_fps or s.fps
        frames_per_image = 1 if signature[2] else signature[1]  # rolling: every frame
        fps = min(30.0, max(1.0, source_fps / frames_per_image))
        try:
            path = recording_path(signature[0], self.demo, self.tag)
            self._recorder = AVIRecorder(path, packet.render(), fps)
        except OSError as exc:
            self.update_settings(record=False)
            self.events.put(("record_off", None))
            error_reports.caught(self.log, exc, "recorder", "Recording not started")
            return None
        self._set_status(record=f"Recording: {fps:.1f} fps")
        self.log.write(f"Recording AVI: {path} at {fps:.1f} fps")
        return signature

    def _sample_noise(self, image, now):
        """Feed a running noise calibration and finish it when its time is up.

        Args: image is the processed difference frame, before the stretch;
            now is the monotonic time.
        Returns: None. On completion sets min_range, logs the result and
            sends a "noise_calibrated" event carrying the new floor.
        """
        calibration = self._calibration
        if calibration is None:
            return
        calibration.add(image)
        if not calibration.done(now):
            return
        self._calibration = None
        noise, floor = calibration.result()
        self.update_settings(min_range=floor)
        self.log.write(f"Noise calibration: still-scene differences peak at "
                       f"{noise:.1f} grey levels; sensitivity floor set to {floor}.")
        self.events.put(("noise_calibrated", floor))

    def _floor_note(self, s, processor, shape):
        """Describe the noise floor's state for the status-bar hint.

        Args: s is the Settings snapshot; processor is the ESPIProcessor;
            shape is the last difference frame's shape, or None.
        Returns: A short sentence while a measurement runs, or while Subtract
            noise floor is on without a map for the current settings;
            otherwise None (Raw view and Raw difference included).
        """
        if s.mode == "Raw" or s.raw_difference:
            return None
        if self._floor_measure is not None:
            running = self._floor_measure[1]
            return (f"Measuring noise floor: {running.count}/{running.min_images} "
                    "images; keep the object still")
        if s.floor_subtract and (self._floor is None or
                                 self._floor[0] != floor_key(s, processor, shape)):
            return "No noise floor for these settings: press Measure noise floor"
        return None

    def _floor_use(self, s, processor, shape):
        """Choose the floor map to subtract and the measurement to feed.

        Args: s is the Settings snapshot; processor is the ESPIProcessor;
            shape is the difference frame's shape.
        Returns: (floor, floor_sample) for finish_difference: the stored map
            when Subtract noise floor is on and it was measured under the
            current settings, else None; and a callback feeding a running
            measurement, else None.
        """
        key = floor_key(s, processor, shape)
        floor = None
        if s.floor_subtract and self._floor is not None and self._floor[0] == key:
            floor = self._floor[1]
        sample = None
        if self._floor_measure is not None:
            def sample(image, now):
                """Feed this frame to the running floor measurement.

                Args: image is the filtered difference; now is the monotonic time.
                Returns: None.
                """
                self._sample_floor(image, key, now)
        return floor, sample

    def _sample_floor(self, image, key, now):
        """Feed a running noise floor measurement and store the map when done.

        A change of settings, view or baseline while it runs cancels it, so a
        map never mixes images made two ways.

        Args: image is the filtered difference before the floor is taken
            off; key is its floor_key; now is the monotonic time.
        Returns: None.
        """
        started, running = self._floor_measure
        if started is None:
            started = key
        elif key != started:
            self._floor_measure = None
            self.log.write("Noise floor measurement cancelled: the settings changed "
                           "while it ran. Press Measure noise floor again.")
            return
        self._floor_measure = (started, running)
        running.add(image)
        if not running.done(now):
            return
        self._floor_measure = None
        floor = running.result()
        self._floor = (key, floor)
        self.log.write(f"Noise floor measured from {running.count} images: "
                       f"median {float(np.median(floor)):.1f}, highest "
                       f"{float(floor.max()):.1f} grey levels.")

    def _reap_recorders(self):
        """Report and forget recordings whose writer threads have finished.

        Args: None.
        Returns: None.
        """
        for old in self._retired[:]:
            if old.done.is_set():
                self._recording_finished(old)
                self._retired.remove(old)
                if self._recorder is None:
                    self._set_status(record="Recording: off")

    def _recording_finished(self, recorder):
        """Log a finished recording; a failed one is logged and, unless it is
        a full or read-only disk, reported.

        Args: recorder is an AVIRecorder whose thread has finished.
        Returns: None.
        """
        self.log.write(recorder.summary())
        if recorder.error is not None:
            error_reports.caught(self.log, recorder.error, "recorder", "Recording error")

    def _handle_actions(self, processor, frame):
        """Run queued one-shot commands on the worker thread.

        Args: processor is the current ESPIProcessor; frame is the newest
            camera/demo frame or None when no frame arrived.
        Returns: (processor, frame), replaced after a live-capture switch.
        """
        while True:
            try:
                action, payload = self._actions.get_nowait()
            except queue.Empty:
                return processor, frame
            if action == "save":
                try:
                    packet = payload.get("packet")  # a frozen display's packet
                    if packet is None:
                        packet = self.latest_packet()
                    if packet is None:
                        self.log.write("No current display frame is ready to save.")
                    else:
                        save_frame(packet.render(), payload.get("title", ""),
                                   payload.get("notes", ""), packet.demo, self.log,
                                   self.tag, frozen_at=payload.get("frozen_at"))
                finally:
                    with self._saves_lock:
                        self._saves_waiting -= 1
            elif action == "reset_previous":
                processor.reset_previous()
                self.log.write("Reset previous frame.")
            elif action == "reset_average":
                processor.reset_avg()
                self.log.write("Reset averaging.")
            elif action == "save_compare":
                try:
                    self._save_compare(payload)
                finally:
                    with self._saves_lock:
                        self._saves_waiting -= 1
            elif action == "calibrate_noise":
                self._calibration = NoiseCalibration(time.monotonic())
                self.log.write("Noise calibration: keep the object still for 2 s.")
            elif action == "measure_floor":
                self._floor_measure = (None, NoiseFloorMap(time.monotonic()))
                self.log.write("Measuring the noise floor: keep the object still "
                               "for at least 2 s.")
            elif action == "reference":
                s = self.settings()
                if s.ref_frames > 1:
                    processor.start_reference(s.ref_frames, s.noise_reduction,
                                              time.monotonic())
                    self.log.write(f"Capturing reference: averaging the next "
                                   f"{s.ref_frames} frames.")
                elif frame is None:
                    self.log.write("No frame available yet for a reference.")
                else:
                    processor.ref_capture = None
                    processor.set_reference(frame, time.monotonic(), s.noise_reduction)
                    processor.reset_avg()
                    self.log.write("Captured reference frame at noise reduction "
                                   f"{s.noise_reduction:g}.")
            elif action == "switch_live" and self.demo:
                processor, frame = self._switch_live(processor, frame)
            elif action == "camera_command":
                self._camera_command(payload)
            elif action == "drive" and self.demo:
                self.demo_source.set_drive(payload["drive"])

    def _camera_command(self, fields):
        """Send phone-only settings to a source that takes them.

        Args: fields are the command keys and values.
        Returns: None; a send failure is logged, and sources without
            send_command (FLIR, capture, demo) ignore the command.
        """
        send = getattr(self.camera, "send_command", None)
        if send is None:
            return
        try:
            send(**fields)
        except OSError as exc:
            self.log.write(f"Could not send {fields} to the camera: {exc}")

    def _save_compare(self, payload):
        """Save a comparison's images as PNGs plus one notes file.

        The files are named for the baseline: reference or previous.

        Args: payload holds packet (a ComparePacket), title and notes.
        Returns: None; the outcome is logged.
        """
        compare = payload["packet"]
        name = compare.view.lower()
        images = [(part_name, part.render()) for part_name, part in (
            (name, compare.baseline), ("current", compare.current),
            ("difference", compare.difference)) if part is not None]
        if compare.baseline is None:
            self.log.write(f"No {'reference' if name == 'reference' else 'previous frame'}"
                           " yet: saving the current frame only.")
        save_frame_set(images, payload.get("title", ""), payload.get("notes", ""),
                       compare.demo, self.log, self.tag,
                       details=(f"{compare.baseline_label}: {compare.baseline_info}",
                                f"Difference: {compare.difference_info}"))

    def _switch_live(self, processor, frame):
        """Try to replace synthetic video with the first FLIR camera.

        Args: processor and frame are the current demo-mode state.
        Returns: (processor, frame); fresh state on success, unchanged on
            failure so the synthetic stream keeps running.
        """
        self.log.write("Searching for a camera to switch to live capture...")
        try:
            settings = self.settings()
            attempt = sources.open_first_live(
                self.log, settings.exposure_us, settings.gain_db, settings.fps)
        except Exception as exc:
            error_reports.caught(self.log, exc, "switch_live", "Still in dummy mode")
            self.events.put(("switch_failed", str(exc)))
            return processor, frame
        if self._recorder is not None:
            self._recorder.stop()
            self._retired.append(self._recorder)
            self._recorder = None
        self.update_settings(record=False, mode="Raw")
        with self._packet_lock:
            self._packet = None
        self._attach_camera(attempt)
        self._set_status(record="Recording: off",
                         range="Raw Mono8: no brightness scaling")
        # enumerate here so every PySpin call stays on this thread
        self.events.put(("switched_live", sources.detect_cameras(self.log)))
        self.log.write("LIVE CAPTURE ACTIVE: camera connected; demo ended.")
        self.log.write("Recording stopped for the source change; restart it if needed.")
        return ESPIProcessor(), None
