"""AVI recording and still-frame saving shared by every front end."""

import math
import os
import threading
import time

import cv2

from espi_live.paths import output_folder, timestamp


class AVIRecorder:
    """Write a constant-rate AVI on a worker using the newest displayed frame."""

    def __init__(self, path, frame, fps):
        """Open an AVI and start its timing-controlled writer thread.

        Args: path is the target AVI filename; frame is the first full-size
            display frame; fps is the selected output frame rate.
        Returns: None. Raises OSError if OpenCV cannot open the file.
        """
        height, width = frame.shape[:2]
        self.path = path
        self.fps = fps
        self.writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"MJPG"), fps,
            (width, height), True)
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
        self.thread = threading.Thread(target=self._run, daemon=True)
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

    def stop(self):
        """Request a clean writer-thread stop without blocking the preview.

        Args: None.
        Returns: None; completion is signaled by the done event.
        """
        if self.stop_time is None:
            self.stop_time = time.monotonic()
            self.stop_event.set()

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
        output = (frame if frame.ndim == 3 else
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

    def summary(self):
        """Summarize finalized recording frames and timing diagnostics.

        Args: None.
        Returns: A human-readable recording summary string.
        """
        return (f"Saved AVI: {self.path} ({self.frames_written} frames at "
                f"{self.fps:.1f} fps; {self.skipped_preview_frames} preview "
                f"frames skipped, {self.late_intervals} late encoder intervals)")


def recording_path(mode, demo):
    """Choose a new AVI path under output/recordings.

    Args: mode is the view mode name; demo marks synthetic video.
    Returns: An absolute .avi path. Raises OSError if the folder is unusable.
    """
    prefix = "dummy_" if demo else ""
    return os.path.join(output_folder("recordings"),
                        f"{prefix}espi_{mode.lower()}_{timestamp()}.avi")


def save_frame(display, title, notes, demo, log):
    """Save a displayed frame as PNG plus an optional title/notes sidecar.

    Args: display is the grayscale or BGR image exactly as shown; title and
        notes are optional user text; demo marks synthetic video; log has a
        write(message) method.
    Returns: The PNG path when saved, otherwise None (the reason is logged).
    """
    title = title.strip()
    notes = notes.strip()
    # Limit the title to safe filename characters on Windows.
    safe_title = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in title
    ).strip("_")[:80]
    suffix = f"_{safe_title}" if safe_title else ""
    prefix = "dummy_" if demo else ""
    try:
        folder = output_folder("frames")
        path = os.path.join(folder, f"{prefix}frame_{timestamp()}{suffix}.png")
        if not cv2.imwrite(path, display):
            log.write(f"Could not save: {path}")
            return None
    except (OSError, cv2.error) as exc:
        log.write(f"Could not save frame: {exc}")
        return None
    log.write(f"Saved: {path}")
    if title or notes:
        info_path = os.path.splitext(path)[0] + ".txt"
        try:
            with open(info_path, "w", encoding="utf-8") as info:
                info.write(
                    f"Frame: {os.path.basename(path)}\n"
                    f"Source: {'Demo' if demo else 'Camera'}\n"
                    f"Title: {title}\n\nNotes:\n{notes}\n")
            log.write(f"Saved frame info: {info_path}")
        except OSError as exc:
            log.write(f"Frame saved, but could not save info: {exc}")
    return path
