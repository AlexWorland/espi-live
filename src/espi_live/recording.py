"""AVI recording and still-frame saving shared by every front end."""

import math
import os
import threading
import time

import cv2

from espi_live import error_reports, updater
from espi_live.paths import output_folder, timestamp


class AVIRecorder:
    """Write a constant-rate AVI on a worker using the newest displayed frame."""

    def __init__(self, path, frame, fps, codec="MJPG", color=True, timeline=True):
        """Open an AVI and start its timing-controlled writer thread.

        Args: path is the target AVI filename; frame is the first full-size
            display frame; fps is the selected output frame rate; codec is
            the four-character OpenCV codec ("MJPG", or "FFV1" for lossless);
            color False writes 2-D frames as a single channel instead of
            converting them to BGR; timeline False writes every submitted
            frame exactly once, never repeating one to hold a constant rate
            (for measurement data), so frames the encoder cannot keep up
            with are dropped and counted instead.
        Returns: None. Raises OSError if OpenCV cannot open the file.
        """
        height, width = frame.shape[:2]
        self.path = path
        self.fps = fps
        self.color = color
        self.timeline = timeline
        self._wake = threading.Event()
        self.writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*codec), fps,
            (width, height), color)
        if not self.writer.isOpened():
            self.writer.release()
            raise OSError(f"Could not open AVI recording: {path}")
        self.lock = threading.Lock()
        self.frame = frame
        self.sequence = 1
        self.frames_written = 0
        self.skipped_preview_frames = 0
        self.late_intervals = 0
        self.error = None
        self.stop_time = None
        self.stop_event = threading.Event()
        self.done = threading.Event()
        self.thread = threading.Thread(
            target=self._run if timeline else self._run_exact, daemon=True)
        try:
            self.thread.start()
        except RuntimeError:
            self.writer.release()
            raise

    def submit(self, frame):
        """Replace the pending recording image with the newest preview frame.

        Args: frame is an immutable grayscale or BGR display image.
        Returns: None. An older, unwritten preview frame may be skipped.
        """
        with self.lock:
            self.frame = frame
            self.sequence += 1
        self._wake.set()

    def stop(self):
        """Request a clean writer-thread stop without blocking the preview.

        Args: None.
        Returns: None; completion is signaled by the done event.
        """
        if self.stop_time is None:
            self.stop_time = time.monotonic()
            self.stop_event.set()
            self._wake.set()

    def _write_latest(self, previous_sequence):
        """Encode the latest available preview frame once.

        Args: previous_sequence is the last frame sequence already sampled.
        Returns: The sequence number sampled for this AVI frame.
        """
        with self.lock:
            frame = self.frame
            sequence = self.sequence
        if sequence > previous_sequence + 1:
            self.skipped_preview_frames += sequence - previous_sequence - 1
        output = (frame if frame.ndim == 3 or not self.color else
                  cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        self.writer.write(output)
        self.frames_written += 1
        return sequence

    def _run(self):
        """Write latest frames on the AVI's constant-rate timeline.

        Args: None; the thread reads submitted frames and stop_event.
        Returns: None; sets done and records any encoder error.
        """
        interval = 1.0 / self.fps
        next_tick = time.monotonic()
        start_time = next_tick
        previous_sequence = 0
        try:
            while True:
                previous_sequence = self._write_latest(previous_sequence)
                next_tick += interval
                delay = next_tick - time.monotonic()
                if delay < 0:
                    self.late_intervals += 1
                if self.stop_event.wait(max(0, delay)):
                    final_frames = max(
                        1, math.ceil((self.stop_time - start_time) * self.fps))
                    while self.frames_written < final_frames:
                        previous_sequence = self._write_latest(previous_sequence)
                    break
        except Exception as exc:
            self.error = exc
        finally:
            try:
                self.writer.release()
            finally:
                self.done.set()

    def _run_exact(self):
        """Write each submitted frame once, with no constant-rate timeline.

        Args: None; the thread reads submitted frames and stop_event.
        Returns: None; sets done and records any encoder error.
        """
        written = 0
        try:
            while True:
                self._wake.clear()
                with self.lock:
                    sequence = self.sequence
                if sequence > written:
                    written = self._write_latest(written)
                elif self.stop_event.is_set():
                    break
                else:
                    self._wake.wait(0.5)
        except Exception as exc:
            self.error = exc
        finally:
            try:
                self.writer.release()
            finally:
                self.done.set()

    def summary(self):
        """Summarize finalized recording frames and timing diagnostics.

        Args: None.
        Returns: A human-readable recording summary string.
        """
        if not self.timeline:
            return (f"Saved AVI: {self.path} ({self.frames_written} frames, "
                    f"{self.skipped_preview_frames} camera frames dropped because "
                    f"the encoder was behind; container rate {self.fps:.1f} fps)")
        return (f"Saved AVI: {self.path} ({self.frames_written} frames at "
                f"{self.fps:.1f} fps; {self.skipped_preview_frames} preview "
                f"frames skipped, {self.late_intervals} late encoder intervals)")


def safe_name(text):
    """Reduce text to characters that are safe in file names on Windows.

    Args: text is any user or camera text.
    Returns: Letters, digits, "-" and "_" only (others become "_"), at most
        80 characters, without leading or trailing "_".
    """
    return "".join(char if char.isalnum() or char in "-_" else "_"
                   for char in text.strip()).strip("_")[:80]


WINDOW_FPS = 10.0


class RecordingSession:
    """One Record press: a timestamped folder holding every stream's AVI."""

    def __init__(self, demo, tag="", details=""):
        """Create the session folder and its session.txt.

        Args: demo marks synthetic video (folder gets the dummy_ prefix); tag
            names the camera (for example its serial number) when several
            are used; details is extra text for session.txt, such as the
            starting settings.
        Returns: None. Raises OSError if the folder or file cannot be created.
        """
        prefix = "dummy_" if demo else ""
        camera = f"_{safe_name(tag)}" if safe_name(tag) else ""
        stamp = timestamp()
        self.folder = os.path.join(output_folder("recordings"),
                                   f"{prefix}espi_{stamp}{camera}")
        os.makedirs(self.folder, exist_ok=True)
        self._processed_count = 0
        with open(os.path.join(self.folder, "session.txt"), "w",
                  encoding="utf-8") as info:
            info.write(f"Source: {'Demo' if demo else 'Camera'}"
                       f"{f' {tag}' if tag else ''}\n"
                       f"Started: {stamp}\n"
                       f"Build: {self._build_text()}\n{details}\n")

    @staticmethod
    def _build_text():
        """Describe the running build for session.txt.

        Args: None.
        Returns: "<number> (<commit>)" for a release build, else "source checkout".
        """
        build = updater.current_build()
        if not build:
            return "source checkout"
        return f"{build} ({updater.build_value('COMMIT', '')})"

    def note(self, text):
        """Append one line to session.txt.

        Args: text is the line to add.
        Returns: None. Raises OSError if the file cannot be written.
        """
        with open(os.path.join(self.folder, "session.txt"), "a",
                  encoding="utf-8") as info:
            info.write(f"{text}\n")

    def open_raw(self, frame, fps, log):
        """Start raw.avi: lossless single-channel FFV1, or MJPG if FFV1 fails.

        Args: frame is the first raw Mono8 frame; fps is the output rate; log
            has a write(message) method for the fallback notice.
        Returns: The AVIRecorder. Raises OSError if neither codec can open.
        """
        path = os.path.join(self.folder, "raw.avi")
        try:
            recorder = AVIRecorder(path, frame, fps, codec="FFV1", color=False,
                                   timeline=False)
            codec = "FFV1 (lossless)"
        except OSError:
            log.write("FFV1 lossless recording is unavailable in this OpenCV "
                      "build; raw.avi is MJPG (lossy).")
            recorder = AVIRecorder(path, frame, fps, timeline=False)
            codec = "MJPG (lossy fallback)"
        self.note(f"Raw codec: {codec}\nRaw rate: {fps:.1f} fps")
        return recorder

    def open_processed(self, mode, frame, fps):
        """Start the next numbered processed file, such as processed_01_raw.avi.

        Args: mode is the view mode name; frame is the first display frame;
            fps is the output rate.
        Returns: The AVIRecorder. Raises OSError if OpenCV cannot open the file.
        """
        self._processed_count += 1
        name = f"processed_{self._processed_count:02d}_{mode.lower()}.avi"
        return AVIRecorder(os.path.join(self.folder, name), frame, fps)


def open_window_recorder(folder, frame, fps=WINDOW_FPS):
    """Start window.avi (MJPG) in a session folder.

    Args: folder is the RecordingSession folder; frame is the first BGR window
        grab; fps is the output rate.
    Returns: The AVIRecorder. Raises OSError if OpenCV cannot open the file.
    """
    return AVIRecorder(os.path.join(folder, "window.avi"), frame, fps)


def save_frame(display, title, notes, demo, log, tag="", frozen_at=None):
    """Save a displayed frame as PNG plus an optional title/notes sidecar.

    Args: display is the grayscale or BGR image exactly as shown; title and
        notes are optional user text; demo marks synthetic video; log is an
        AppLog or PrefixedLog; tag names the camera when several are used;
        frozen_at is the datetime the display was frozen on this frame, or
        None for a live frame (the file name carries the save time).
    Returns: The PNG path when saved, otherwise None (the reason is logged).
    """
    title = title.strip()
    notes = notes.strip()
    safe_title = safe_name(title)
    suffix = f"_{safe_title}" if safe_title else ""
    prefix = "dummy_" if demo else ""
    camera = f"{safe_name(tag)}_" if safe_name(tag) else ""
    try:
        folder = output_folder("frames")
        path = os.path.join(folder, f"{prefix}frame_{camera}{timestamp()}{suffix}.png")
        if not cv2.imwrite(path, display):
            log.warning(f"Could not save: {path}")
            return None
    except (OSError, cv2.error) as exc:
        error_reports.caught(log, exc, "save", "Could not save frame")
        return None
    log.write(f"Saved: {path}")
    if title or notes:
        info_path = os.path.splitext(path)[0] + ".txt"
        frozen = f"Frozen at: {frozen_at:%Y-%m-%d %H:%M:%S}\n" if frozen_at else ""
        try:
            with open(info_path, "w", encoding="utf-8") as info:
                info.write(
                    f"Frame: {os.path.basename(path)}\n"
                    f"Source: {'Demo' if demo else 'Camera'}"
                    f"{f' {tag}' if tag else ''}\n"
                    f"{frozen}"
                    f"Title: {title}\n\nNotes:\n{notes}\n")
            log.write(f"Saved frame info: {info_path}")
        except OSError as exc:
            log.warning(f"Frame saved, but could not save info: {exc}")
    return path
