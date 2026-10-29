"""Read video devices through OpenCV as a camera source.

This is how the viewer uses an analog camera such as the Sony XC-ST50: its
composite video goes into a capture adapter (a Hauppauge USB-Live2), which
the operating system exposes as a video device. CaptureSession mirrors
camera.CameraSession (open, grab, set_float, limits, read_values, describe,
close, identity, info), so the engine, processing, recording and the front
end treat both alike. Exposure, gain and frame rate are set on the analog
camera itself and cannot be changed here.

Devices are listed with the optional cv2_enumerate_cameras package, imported
lazily: OpenCV alone cannot report device names or USB IDs. No GUI imports.
"""

import sys
import time

import cv2
import numpy as np

from .camera import CameraInfo
from .capture_models import CaptureIdentity, adapter_for, fourcc_text

# OpenCV properties copied into camera reports: (label, cv2 attribute name).
DETAIL_PROPERTIES = (
    ("Channel", "CAP_PROP_CHANNEL"), ("Brightness", "CAP_PROP_BRIGHTNESS"),
    ("Contrast", "CAP_PROP_CONTRAST"), ("Saturation", "CAP_PROP_SATURATION"),
    ("Hue", "CAP_PROP_HUE"), ("Gain", "CAP_PROP_GAIN"),
    ("Exposure", "CAP_PROP_EXPOSURE"), ("Convert RGB", "CAP_PROP_CONVERT_RGB"),
    ("Format", "CAP_PROP_FORMAT"), ("Mode", "CAP_PROP_MODE"),
    ("Buffer size", "CAP_PROP_BUFFERSIZE"),
)


def capture_backend():
    """Choose the OpenCV capture backend for this platform.

    Args: None.
    Returns: CAP_DSHOW on Windows (it selects composite input on analog
        adapters), CAP_V4L2 on Linux, CAP_AVFOUNDATION elsewhere.
    """
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_AVFOUNDATION


def backend_name(backend):
    """Name an OpenCV backend for display.

    Args: backend is an OpenCV backend number.
    Returns: Its name, or the number as text when OpenCV does not know it.
    """
    try:
        return cv2.videoio_registry.getBackendName(backend)
    except cv2.error:
        return str(backend)


def load_enumerator():
    """Import cv2_enumerate_cameras on first use.

    Args: None.
    Returns: The module, or None when it is not installed.
    """
    try:
        import cv2_enumerate_cameras
    except ImportError:
        return None
    return cv2_enumerate_cameras


def list_capture_devices(log):
    """Enumerate the video devices without opening them.

    Args: log has a write(message) method.
    Returns: A list of capture CameraInfo, with serials "video 1", "video
        2", ... in enumeration order; empty when listing is unavailable.
    """
    enumerator = load_enumerator()
    if enumerator is None:
        log.write("Video devices are not listed: cv2_enumerate_cameras is not installed.")
        return []
    try:
        devices = enumerator.enumerate_cameras(capture_backend())
    except Exception as exc:
        log.write(f"Could not list video devices: {exc}")
        return []
    return [CameraInfo(f"video {position}", device.name or "", kind="capture",
                       index=device.index, backend=device.backend, vid=device.vid,
                       pid=device.pid, path=device.path or "")
            for position, device in enumerate(devices, start=1)]


def to_mono8(frame):
    """Convert a captured frame to the viewer's Mono8 format.

    Args: frame is what VideoCapture.read() returned.
    Returns: A 2-D uint8 array: Mono8 unchanged, 16-bit shifted to its top
        byte, BGR reduced to its luma. Raises ValueError for other layouts.
    """
    if frame.ndim == 2 and frame.dtype == np.uint8:
        return frame
    if frame.ndim == 2 and frame.dtype == np.uint16:
        return (frame >> 8).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"cannot convert a {frame.shape} {frame.dtype} frame to Mono8")


class CaptureSession:
    """Own an OpenCV video device and its acquisition lifecycle."""

    def __init__(self, log, info):
        """Prepare to open one enumerated video device.

        Args: log has a write(message) method; info is the device's capture
            CameraInfo.
        Returns: None.
        """
        self.log = log
        self.info = info
        self.cap = None
        self.identity = CaptureIdentity(name=info.model, path=info.path,
                                        vid=info.vid, pid=info.pid)
        self.interlaced = False
        self.first_frame = None  # Mono8 copy of the first frame, for reports
        self.raw_layout = ""     # shape and type of that frame as delivered
        self._frames = 0
        self._since = None

    def open(self):
        """Open the device, request a recognised adapter's format, read a frame.

        Args: None.
        Returns: None; self.identity describes the delivered video. Raises
            RuntimeError naming the device when it does not open or sends no
            video; the device is released first.
        """
        adapter = adapter_for(self.info.vid, self.info.pid, self.info.model)
        self.cap = cv2.VideoCapture(self.info.index, self.info.backend)
        try:
            if not self.cap.isOpened():
                raise RuntimeError("the device did not open")
            if adapter:
                self.cap.set(cv2.CAP_PROP_CHANNEL, adapter.channel)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, adapter.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, adapter.height)
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise RuntimeError("no video arrived")
            self.first_frame = to_mono8(frame)
        except Exception as exc:
            self.close()
            raise RuntimeError(
                f"Could not open {self.info.label}: {exc}. Close WinTV or Hauppauge "
                "Capture if they are running, and check the video cable.") from exc
        self.raw_layout = " × ".join(str(size) for size in frame.shape) + f", {frame.dtype}"
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        height, width = self.first_frame.shape
        self.identity = CaptureIdentity(
            name=self.info.model, backend=backend_name(self.info.backend),
            path=self.info.path, vid=self.info.vid, pid=self.info.pid,
            width=width, height=height,
            fourcc=fourcc_text(self.cap.get(cv2.CAP_PROP_FOURCC)),
            fps=fps if fps > 0 else 0.0, adapter=adapter)
        self.interlaced = self.identity.interlaced
        self._frames, self._since = 0, time.monotonic()

    def grab(self, timeout_ms):
        """Wait for the next frame and convert it to Mono8.

        Args: timeout_ms is accepted for CameraSession compatibility; OpenCV's
            read() blocks for about one frame and cannot be bounded.
        Returns: (frame, "ok") with a 2-D uint8 array, or (exception,
            "error") when the device sent nothing or an unusable frame.
        """
        try:
            ok, frame = self.cap.read()
        except cv2.error as exc:
            return exc, "error"
        if not ok or frame is None:
            return RuntimeError("the capture device returned no frame"), "error"
        try:
            image = to_mono8(frame)
        except ValueError as exc:
            return exc, "error"
        self._frames += 1
        return image, "ok"

    def set_float(self, name, value):
        """Ignore a camera-setting change: the analog camera sets its own.

        Args: name and value are the requested node and value.
        Returns: None.
        """
        return None

    def limits(self):
        """Report adjustable settings; a video device has none here.

        Args: None.
        Returns: An empty dict.
        """
        return {}

    def read_values(self):
        """Measure the capture rate since the previous call.

        Args: None.
        Returns: A dict like CameraSession.read_values(): exposure, gain and
            requested are None; resulting is frames per second, or None
            before any frame arrived.
        """
        now = time.monotonic()
        elapsed = now - self._since
        rate = self._frames / elapsed if self._frames and elapsed > 0 else None
        self._frames, self._since = 0, now
        return {"exposure": None, "gain": None, "requested": None, "resulting": rate}

    def describe(self, values):
        """Format a read_values() result for the status line.

        Args: values is the dict returned by read_values().
        Returns: Status text naming the capture size and measured rate.
        """
        identity = self.identity
        kind = "Analog capture" if identity.supported else "Video capture"
        rate = values["resulting"]
        measured = f"{rate:.1f} fps measured" if rate else "measuring frame rate"
        where = "set on the camera" if identity.supported else "not adjustable here"
        return (f"{kind} {identity.width} × {identity.height}: {measured}; "
                f"exposure, gain and frame rate are {where}")

    def read_details(self):
        """Collect what the device reports, for a camera report.

        Args: None.
        Returns: A tuple of (label, text) pairs.
        """
        identity, frame = self.identity, self.first_frame
        usb = (f"{identity.vid:04x}:{identity.pid:04x}"
               if identity.vid is not None and identity.pid is not None
               else "not reported")
        rows = [("Device name", identity.name or "not reported"),
                ("Device path", identity.path or "not reported"),
                ("USB ID", usb), ("Backend", identity.backend),
                ("OpenCV", cv2.__version__),
                ("Frame size", f"{identity.width} × {identity.height}"),
                ("Reported FPS", f"{identity.fps:g}"),
                ("FOURCC", identity.fourcc or "unknown")]
        rows += [(label, f"{self.cap.get(getattr(cv2, name)):g}")
                 for label, name in DETAIL_PROPERTIES]
        rows += [("First frame (raw)", self.raw_layout),
                 ("First frame min / mean / max",
                  f"{frame.min()} / {frame.mean():.1f} / {frame.max()}")]
        return tuple(rows)

    def close(self):
        """Release the device; safe to call more than once.

        Args: None.
        Returns: None; a release error is written to the log.
        """
        if self.cap is None:
            return
        try:
            self.cap.release()
        except Exception as exc:
            self.log.write(f"Capture release error: {exc}")
        self.cap = None
