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
import traceback
from dataclasses import dataclass, replace

import cv2
import numpy as np

from espi_live.camera import CameraSession, format_camera_values, list_cameras
from espi_live.processing import (DemoSource, ESPIProcessor, apply_palette,
                                  gamma_lut, smooth_speckle,
                                  stretch_contrast, watermark_demo)
from espi_live.recording import AVIRecorder, recording_path, save_frame

CAMERA_TIMEOUT_MS = 100
SETTING_INTERVAL_S = 0.1
READBACK_INTERVAL_S = 1.0
RANGE_INTERVAL_S = 0.5
RECORDER_FINISH_TIMEOUT_S = 5.0
PREDECESSOR_TIMEOUT_S = 10.0  # open_on_start waits this long for the old engine


@dataclass
class Settings:
    """User-adjustable acquisition, measurement and display settings."""

    mode: str = "Raw"
    exposure_us: float = 8000.0
    gain_db: float = 0.0
    fps: float = 30.0
    noise_reduction: float = 0.0  # % of history blended into each camera frame; 0 = off
    average: int = 1
    rolling: bool = False         # running average instead of one image per N frames
    ref_refresh_s: float = 0.0    # recapture the Reference every N s; 0 = never
    smoothing: int = 0            # speckle filter width in pixels; 0 = off
    smoothing_method: str = "Gaussian"
    scale_range: bool = True
    clip_percent: float = 0.0     # stretch ignores this % of pixels at each end
    invert: bool = False
    palette: str = "Gray"
    gamma: float = 1.0
    clahe: bool = True
    record: bool = False


@dataclass(frozen=True)
class FramePacket:
    """One processed frame ready for display, saving or recording.

    image is a 2-D uint8 array that is never modified after publication.
    palette and gamma are the color steps still to apply; Raw frames carry
    ("Gray", 1.0), and CLAHE frames already include gamma.
    """

    sequence: int
    image: np.ndarray
    palette: str
    gamma: float
    mode: str
    demo: bool

    def render(self):
        """Produce the exact colors shown on screen, on the CPU.

        Args: None.
        Returns: A grayscale or BGR uint8 image for saving or recording.
        """
        return apply_palette(self.image, self.palette, self.gamma)


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

    def __init__(self, log, serial=None, tag=""):
        """Create an idle engine.

        Args: log is an AppLog (or any object with write(message)); serial
            selects a camera by serial number (None: the first camera); tag
            is added to saved file names to tell cameras apart ("" for none).
        Returns: None.
        """
        self.log = log
        self.serial = serial
        self.tag = tag
        self.camera_info = None  # CameraInfo of the opened camera
        self._open_after = None  # (predecessor engine,) when opening on the worker
        self.events = queue.SimpleQueue()
        self._settings = Settings()
        self._settings_lock = threading.Lock()
        self._actions = queue.SimpleQueue()
        self._packet = None
        self._packet_lock = threading.Lock()
        self._status = {
            "demo": False, "camera": "Camera values: waiting for connection",
            "range": "Input: —    Scaled: —",
            "hint": "Raw view: direct Mono8 positioning",
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
        """Try to open the first FLIR camera; call before start().

        Args: None.
        Returns: None. Raises if PySpin or a camera is unavailable; the
            engine is then unchanged and use_demo() may be called.
        """
        attempt = CameraSession(self.log, self.serial)
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
        self.demo_source = DemoSource(label=label)
        self._set_status(demo=True, camera=(
            "DUMMY MODE ACTIVE — exposure, gain, and FPS are simulated"))
        self.log.write("DUMMY MODE ACTIVE: synthetic video; no camera acquired.")

    def _attach_camera(self, session):
        """Make an opened camera session the frame source.

        Args: session is an opened CameraSession.
        Returns: None; publishes the model as a "camera_identity" event and
            the device limits as a "camera_limits" event.
        """
        self.camera = session
        self.camera_info = session.info
        self.demo_source = None
        identity = session.identity
        self.log.write("Camera: " + " | ".join(identity.description()))
        for warning in identity.warnings():
            self.log.write(f"Camera note: {warning}")
        self.events.put(("camera_identity", identity))
        self._set_status(demo=False, camera=format_camera_values(
            session.read_values()))
        self.events.put(("camera_limits", session.limits()))

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

    def request(self, action, **payload):
        """Queue a one-shot command for the worker.

        Args: action is "save" (payload title, notes; optionally packet, a
            frozen display's FramePacket to save instead of the newest one,
            and frozen_at, its datetime), "reference", "reset_previous",
            "reset_average" or "switch_live" (the "switched_live" event then
            carries every detected CameraInfo).
        Returns: None.
        """
        self._actions.put((action, payload))

    def latest_packet(self):
        """Return the newest published frame.

        Args: None.
        Returns: A FramePacket, or None before the first frame.
        """
        with self._packet_lock:
            return self._packet

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
        Returns: None; fatal errors are logged and sent as a "fatal" event.
        """
        self._recorder = None
        self._retired = []
        try:
            if self._open_after is not None and not self._open_deferred():
                return
            self._loop()
        except Exception as exc:
            self.log.write(f"Stopped: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            self.events.put(("fatal", f"{type(exc).__name__}: {exc}"))
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
                        self.log.write(old.summary())
                        if old.error is not None:
                            self.log.write(f"Recording error: {old.error}")
                    else:
                        self.log.write(
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
        else:
            try:
                self.open_camera()
                return True
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
        self.log.write(f"Could not open the camera: {message}")
        self.events.put(("open_failed", message))
        return False

    def _loop(self):
        """Grab, process, publish and record frames until stop() is called.

        Args: None.
        Returns: None.
        """
        processor = ESPIProcessor()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        applied = self.settings()  # camera starts at these defaults
        last_setting_time = 0.0
        last_mode = applied.mode
        record_signature = None
        resulting_fps = None
        last_range_time = 0.0
        last_rolling = applied.rolling
        last_refresh = applied.ref_refresh_s
        last_readback = time.monotonic()
        last_error_log = 0.0
        last_valid = time.monotonic()
        incomplete = 0
        window_start = time.monotonic()
        window_frames = 0
        window_work = 0.0
        sequence = 0
        frame = None
        demo_next = time.monotonic()
        self.log.write("Streaming. Raw shows direct Mono8 pixels; Previous and "
                       "Reference show ESPI differences.")

        while not self._stop.is_set():
            s = self.settings()
            self._reap_recorders()
            signature = (s.mode, s.average if s.mode != "Raw" else 1, s.rolling)
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
            if s.mode != last_mode:
                processor.reset_avg()
                if s.mode == "Previous":
                    processor.prev = None
                self.log.write(f"View: {s.mode}")
                last_mode = s.mode
            if s.rolling != last_rolling:
                processor.reset_avg()
                last_rolling = s.rolling
            if s.ref_refresh_s != last_refresh:
                self.log.write(f"Reference auto-refresh: every {s.ref_refresh_s:.1f} s"
                               if s.ref_refresh_s > 0 else "Reference auto-refresh: off")
                last_refresh = s.ref_refresh_s
            self._set_status(hint=(
                "Raw view: direct Mono8 positioning"
                if s.mode == "Raw" else
                "Reference: press F or Set reference to capture a baseline"
                if s.mode == "Reference" and processor.ref is None else
                f"Reference refreshes every {s.ref_refresh_s:.1f} s to cancel slow drift"
                if s.mode == "Reference" and s.ref_refresh_s > 0 else
                "ESPI difference view; min/max range is before gamma and palette"))

            work_started = time.monotonic()
            if self.demo:
                frame = self.demo_source.next_frame(
                    s.exposure_us, s.gain_db, work_started)
            else:
                grabbed, state = self.camera.grab(CAMERA_TIMEOUT_MS)
                if state != "ok":
                    if state == "incomplete":
                        incomplete += 1
                        if incomplete % 30 == 1:
                            self.log.write(f"Incomplete frames: {incomplete}")
                    elif time.monotonic() - last_error_log >= 5.0:
                        self.log.write(f"Camera timeout or acquisition error: {grabbed}")
                        last_error_log = time.monotonic()
                    if time.monotonic() - last_valid >= 1.0:
                        self._set_status(processing="Waiting for camera frame",
                                         processed_fps=0.0)
                    processor, frame = self._handle_actions(processor, None)
                    continue
                frame = grabbed
            if frame.ndim != 2 or frame.dtype != np.uint8:
                raise RuntimeError("Expected a Mono8 image; check camera PixelFormat.")

            frame = processor.denoise(frame, s.noise_reduction / 100.0)
            now = time.monotonic()
            last_valid = now
            if not self.demo:
                if now - last_readback >= READBACK_INTERVAL_S:
                    values = self.camera.read_values()
                    resulting_fps = values["resulting"]
                    self._set_status(
                        camera=format_camera_values(values),
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

            if s.mode == "Reference" and s.ref_refresh_s > 0 and (
                    processor.ref is None or
                    now - processor.ref_time >= s.ref_refresh_s):
                processor.set_reference(frame, now)
                if not s.rolling:  # keep each block average on one baseline
                    processor.reset_avg()

            update_range = now - last_range_time >= RANGE_INTERVAL_S
            image = None
            palette, gamma = "Gray", 1.0
            if s.mode == "Raw":
                image = frame
                if update_range:
                    self._set_status(range="Raw Mono8: no brightness scaling")
                    last_range_time = now
            else:
                difference = processor.difference(frame, s.mode)
                averaged = (processor.rolling_average(difference, s.average)
                            if s.rolling else processor.average(difference, s.average))
                if averaged is not None:
                    averaged = smooth_speckle(averaged, s.smoothing_method, s.smoothing)
                    if s.scale_range:
                        image, low, high = stretch_contrast(averaged, s.clip_percent)
                        scaled = (0, 255) if high > low else (0, 0)
                    else:
                        image = averaged
                        low = high = scaled = None
                        if update_range:
                            low, high, _, _ = cv2.minMaxLoc(averaged)
                            low, high = int(low), int(high)
                            scaled = (low, high)
                    if update_range:
                        self._set_status(range=(
                            f"Input: {low}–{high}    Scaled: {scaled[0]}–{scaled[1]}"))
                        last_range_time = now
                    if s.invert:
                        image = cv2.bitwise_not(image)
                    palette, gamma = s.palette, s.gamma
                    if s.clahe and s.palette != "Gray":
                        # CLAHE sits between gamma and the palette, so do both
                        # here; the display then applies only the palette.
                        lut = gamma_lut(s.gamma)
                        if lut is not None:
                            image = cv2.LUT(image, lut)
                        image = clahe.apply(image)
                        gamma = 1.0

            if image is not None:
                if self.demo:
                    image = watermark_demo(image)
                sequence += 1
                packet = FramePacket(sequence, image, palette, gamma,
                                     s.mode, self.demo)
                with self._packet_lock:
                    self._packet = packet
                record_signature = self._feed_recorder(
                    s, signature, record_signature, packet, resulting_fps)
            window_frames += 1
            window_work += time.monotonic() - work_started

            elapsed = time.monotonic() - window_start
            if elapsed >= 1.0:
                if self.demo:  # no sensor; the synthetic source runs at the request
                    self._set_status(camera_fps=f"Camera: {s.fps:.1f} fps (simulated)")
                self._set_status(
                    processed_fps=window_frames / elapsed,
                    processing=(f"Processed: {window_frames / elapsed:.1f} fps    "
                                f"Work: {1000 * window_work / max(1, window_frames):.1f}"
                                " ms/frame"))
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
            path = recording_path(s.mode, self.demo, self.tag)
            self._recorder = AVIRecorder(path, packet.render(), fps)
        except OSError as exc:
            self.update_settings(record=False)
            self.events.put(("record_off", None))
            self.log.write(str(exc))
            return None
        self._set_status(record=f"Recording: {fps:.1f} fps")
        self.log.write(f"Recording AVI: {path} at {fps:.1f} fps")
        return signature

    def _reap_recorders(self):
        """Report and forget recordings whose writer threads have finished.

        Args: None.
        Returns: None.
        """
        for old in self._retired[:]:
            if old.done.is_set():
                self.log.write(old.summary())
                if old.error is not None:
                    self.log.write(f"Recording error: {old.error}")
                self._retired.remove(old)
                if self._recorder is None:
                    self._set_status(record="Recording: off")

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
                packet = payload.get("packet")  # a frozen display's packet
                if packet is None:
                    packet = self.latest_packet()
                if packet is None:
                    self.log.write("No current display frame is ready to save.")
                else:
                    save_frame(packet.render(), payload.get("title", ""),
                               payload.get("notes", ""), packet.demo, self.log,
                               self.tag, frozen_at=payload.get("frozen_at"))
            elif action == "reset_previous":
                processor.prev = None
                self.log.write("Reset previous frame.")
            elif action == "reset_average":
                processor.reset_avg()
                self.log.write("Reset averaging.")
            elif action == "reference":
                if frame is None:
                    self.log.write("No frame available yet for a reference.")
                else:
                    processor.set_reference(frame, time.monotonic())
                    processor.reset_avg()
                    self.log.write("Captured reference frame.")
            elif action == "switch_live" and self.demo:
                processor, frame = self._switch_live(processor, frame)

    def _switch_live(self, processor, frame):
        """Try to replace synthetic video with the first FLIR camera.

        Args: processor and frame are the current demo-mode state.
        Returns: (processor, frame); fresh state on success, unchanged on
            failure so the synthetic stream keeps running.
        """
        self.log.write("Searching for a FLIR camera to switch to live capture...")
        attempt = CameraSession(self.log)
        try:
            attempt.open()
        except Exception as exc:
            self.log.write(f"Still in dummy mode: {exc}")
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
        try:  # enumerate here so every PySpin call stays on this thread
            cameras = list_cameras(self.log)
        except Exception as exc:
            self.log.write(f"Could not list the other cameras: {exc}")
            cameras = []
        self.events.put(("switched_live", cameras))
        self.log.write("LIVE CAPTURE ACTIVE: camera connected; demo ended.")
        self.log.write("Recording stopped for the source change; restart it if needed.")
        return ESPIProcessor(), None
