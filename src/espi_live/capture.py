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

Luma path. A recognised adapter (the USB-Live2) sends packed YUY2 (Y0 U Y1 V),
so its brightness is every other byte. OpenCV's DirectShow backend cannot hand
that over: for a YUY2 source it always asks DirectShow's Sample Grabber for
RGB24 (cap_dshow.cpp, OpenCV 5.0.0; CAP_PROP_CONVERT_RGB only affects Y16), and
taking the grey back with COLOR_BGR2GRAY adds rounding and chroma-noise error of
about ±1 grey level (up to ±4 where a channel clips). So the session reads the
Y bytes directly where a backend can deliver them:

- Windows: Media Foundation (CAP_MSMF) with CAP_PROP_CONVERT_RGB 0 returns the
  device's native sample unchanged, as one row of bytes (cap_msmf.cpp). The
  device is first opened through DirectShow as before, which selects the
  composite input (Media Foundation has no input selection) and gives a
  reference frame; Media Foundation is used only when it delivers YUY2 of the
  requested size whose brightness matches that reference.
- Linux: V4L2 with CAP_PROP_CONVERT_RGB 0 returns YUYV as (height, width, 2).

Grey-level scale: decoders send video-range luma (black 16, white 235), and the
YUV-to-RGB conversion of the BGR path expands it to 0-255 (BT.601). The raw Y
is expanded the same way, through a lookup table (EXPAND_TABLE), so the levels
the processing sees (for example the LabVIEW flow's Difference gain 42) keep
their scale; values above 235 saturate at 255 as before. Any problem keeps the
BGR path, and ESPI_ANALOG_LUMA=bgr forces it.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .camera import CameraInfo
from .capture_models import CaptureIdentity, adapter_for, fourcc_text, is_built_in

# Video-range luma (ITU-R BT.601): the codes a decoder uses for black and white.
VIDEO_BLACK, VIDEO_WHITE = 16, 235
# Packed 4:2:2 formats whose even bytes are the luma (YUY2 on Windows, YUYV on Linux).
YUY2_FOURCCS = frozenset({"YUY2", "YUYV"})
# Largest difference between the mean brightness of the raw luma (expanded) and
# of the BGR path's reference frame for the raw path to be used. A right
# conversion differs by about one level; a wrong input, format or range by more.
RAW_MATCH_TOLERANCE = 6.0
RAW_READ_ATTEMPTS = 5   # reads to wait for the first frame after a re-open
LUMA_SETTING = "ESPI_ANALOG_LUMA"  # "bgr" forces the BGR path


def video_range_table():
    """Build the table that expands video-range luma to full range.

    Args: None.
    Returns: A 256-entry uint8 array: round((Y - 16) x 255 / 219), clipped to
        0..255, which is what BT.601 YUV-to-RGB gives for a grey pixel.
    """
    levels = (np.arange(256, dtype=np.float64) - VIDEO_BLACK) * 255.0 / (VIDEO_WHITE - VIDEO_BLACK)
    return np.clip(np.floor(levels + 0.5), 0, 255).astype(np.uint8)


EXPAND_TABLE = video_range_table()


def expand_video_range(luma):
    """Map raw video-range luma onto the BGR path's 0..255 scale.

    Args: luma is a 2-D uint8 array of decoder Y codes.
    Returns: A new contiguous 2-D uint8 array.
    """
    return EXPAND_TABLE[luma]


def describe_layout(frame):
    """Describe a delivered frame's shape and type for logs and reports.

    Args: frame is a NumPy array.
    Returns: Text like "480 × 720 × 3, uint8".
    """
    return " × ".join(str(size) for size in frame.shape) + f", {frame.dtype}"


def yuy2_luma(frame, width, height):
    """Take the Y bytes out of a packed YUY2/YUYV frame of a known size.

    Backends deliver it as (height, width, 2) (V4L2), as one row of bytes
    (Media Foundation without conversion; rows may be padded to a stride), or
    as (height, 2 x width).

    Args: frame is what VideoCapture.read() returned; width and height are
        the frame size the device reports.
    Returns: A 2-D (height, width) uint8 view of the luma, or None when the
        frame is not certainly YUY2 of that size.
    """
    if frame.dtype != np.uint8 or width <= 0 or height <= 0:
        return None
    if frame.shape == (height, width, 2):
        return frame[:, :, 0]
    if frame.ndim == 2 and frame.shape == (height, 2 * width):
        return frame[:, 0::2]
    if frame.ndim == 1 or (frame.ndim == 2 and frame.shape[0] == 1):
        data = frame.reshape(-1)
        if data.size % height:
            return None
        stride = data.size // height
        if stride < 2 * width or stride > 4 * width:
            return None
        return data.reshape(height, stride)[:, 0:2 * width:2]
    return None


def luma_setting():
    """Read the luma-path override from the environment.

    Args: None.
    Returns: "bgr" when ESPI_ANALOG_LUMA asks for the BGR path, else "auto".
    """
    return "bgr" if os.environ.get(LUMA_SETTING, "").strip().lower() == "bgr" else "auto"


def raw_route(info):
    """Choose how raw luma can be read for a device, if at all.

    Args: info is the device's capture CameraInfo.
    Returns: "msmf" for a DirectShow device on Windows, "v4l2" for a V4L2
        device, or "" (BGR path only).
    """
    if info.backend == cv2.CAP_V4L2:
        return "v4l2"
    if info.backend == cv2.CAP_DSHOW and sys.platform.startswith("win"):
        return "msmf"
    return ""


def device_instance(path):
    """Reduce a device path to the part DirectShow and Media Foundation share.

    The two list one device under different interface classes, so their
    paths differ after the last "#".

    Args: path is a device path or symbolic link.
    Returns: The lower-case path up to its last "#", or "" when it has none.
    """
    lowered = (path or "").lower()
    return lowered.rsplit("#", 1)[0] if "#" in lowered else ""


def find_msmf_index(info):
    """Find a DirectShow device's index in Media Foundation's device list.

    The device path is compared first; the USB ID only when it is unique.

    Args: info is the device's capture CameraInfo (path and USB ID).
    Returns: The Media Foundation index. Raises LookupError naming why the
        device cannot be matched.
    """
    enumerator, problem = import_enumerator()
    if enumerator is None:
        raise LookupError(f"cv2_enumerate_cameras {problem}")
    devices = list(enumerator.enumerate_cameras(cv2.CAP_MSMF))
    wanted = device_instance(info.path)
    if wanted:
        for device in devices:
            if device_instance(device.path) == wanted:
                return device.index
    same_id = [device for device in devices
               if info.vid is not None and (device.vid, device.pid) == (info.vid, info.pid)]
    if len(same_id) == 1:
        return same_id[0].index
    raise LookupError("Media Foundation does not list this device"
                      if not same_id else "several devices share its USB ID")

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


_dll_directory_handles = []  # kept so the added search folders stay registered


def bundled_dll_directories():
    """Find the `*.libs` folders of a frozen build.

    Wheels built with delvewheel (cv2_enumerate_cameras, pandas, ...) keep their
    renamed runtime DLLs in a `<package>.libs` folder next to the package and
    look for them only there. PyInstaller may keep a single copy of a DLL that
    several packages share, under one of those folders, so a package can be
    left without its own.

    Args: None.
    Returns: The `*.libs` folders inside the frozen program's library folder;
        empty outside a frozen Windows build.
    """
    root = getattr(sys, "_MEIPASS", None)
    if not root or not hasattr(os, "add_dll_directory"):
        return []
    return sorted(path for path in Path(root).glob("*.libs") if path.is_dir())


def import_enumerator():
    """Import cv2_enumerate_cameras on first use.

    An ImportError that is not "the package is missing" (for example "DLL load
    failed" from its native module) is reported as such, and in a frozen build
    the import is retried once with every bundled `*.libs` folder on the DLL
    search path. The DLL names carry a content hash, so any copy is the right one.

    Args: None.
    Returns: (module, problem): the module and "", or None and a short reason
        for the log ("is not installed", or "could not be loaded: ...").
    """
    try:
        import cv2_enumerate_cameras
        return cv2_enumerate_cameras, ""
    except ImportError as exc:
        error = exc
    folders = bundled_dll_directories()
    if folders:
        for folder in folders:
            _dll_directory_handles.append(os.add_dll_directory(str(folder)))
        try:
            import cv2_enumerate_cameras
            return cv2_enumerate_cameras, ""
        except ImportError as exc:
            error = exc
    if isinstance(error, ModuleNotFoundError) and error.name == "cv2_enumerate_cameras":
        return None, "is not installed"
    return None, f"could not be loaded: {error}"


def list_capture_devices(log):
    """Enumerate the video devices without opening them.

    A laptop's built-in camera is left out (and logged), so it is never
    listed, opened or reported.

    Args: log has a write(message) method.
    Returns: A list of capture CameraInfo, with serials "video 1", "video
        2", ... in enumeration order of the devices kept; empty when listing
        is unavailable.
    """
    enumerator, problem = import_enumerator()
    if enumerator is None:
        log.write(f"Video devices are not listed: cv2_enumerate_cameras {problem}.")
        return []
    try:
        devices = enumerator.enumerate_cameras(capture_backend())
    except Exception as exc:
        log.write(f"Could not list video devices: {exc}")
        return []
    kept = []
    for device in devices:
        if is_built_in(device.vid, device.pid, device.name):
            usb = (f" (USB {device.vid:04x}:{device.pid:04x})"
                   if device.vid is not None and device.pid is not None else "")
            log.write(f"Ignoring the built-in camera {device.name or 'video device'}{usb}.")
        else:
            kept.append(device)
    return [CameraInfo(f"video {position}", device.name or "", kind="capture",
                       index=device.index, backend=device.backend, vid=device.vid,
                       pid=device.pid, path=device.path or "")
            for position, device in enumerate(kept, start=1)]


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
        self.luma_path = "BGR → grey"  # how frames become Mono8 (see the module docstring)
        self.backend = info.backend     # the OpenCV backend actually in use
        self._raw_size = None    # (width, height) while raw luma is read
        self._frames = 0
        self._since = None

    def open(self):
        """Open the device, request a recognised adapter's format, read a frame.

        A recognised adapter then gets the raw-luma path when its backend can
        deliver it (see the module docstring); any problem there keeps the
        BGR path, which is logged once either way.

        Args: None.
        Returns: None; self.identity describes the delivered video. Raises
            RuntimeError naming the device when it does not open or sends no
            video; the device is released first.
        """
        adapter = adapter_for(self.info.vid, self.info.pid, self.info.model)
        frame = self._open_bgr(adapter)
        if adapter:
            frame = self._choose_luma_path(adapter, frame)
        self.raw_layout = describe_layout(frame)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        height, width = self.first_frame.shape
        self.identity = CaptureIdentity(
            name=self.info.model, backend=backend_name(self.backend),
            path=self.info.path, vid=self.info.vid, pid=self.info.pid,
            width=width, height=height,
            fourcc=fourcc_text(self.cap.get(cv2.CAP_PROP_FOURCC)),
            fps=fps if fps > 0 else 0.0, adapter=adapter)
        self.interlaced = self.identity.interlaced
        self._frames, self._since = 0, time.monotonic()

    def _open_bgr(self, adapter):
        """Open the enumerated device as before: the BGR path.

        Args: adapter is the recognised CaptureAdapter, or None.
        Returns: The first frame as delivered; self.cap is open and
            self.first_frame holds its Mono8 copy. Raises RuntimeError naming
            the device (released first) when it does not open or sends no video.
        """
        self.backend, self._raw_size = self.info.backend, None
        self.luma_path = "BGR → grey"
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
        return frame

    def _choose_luma_path(self, adapter, bgr_frame):
        """Switch an adapter to raw luma when it checks out, else keep BGR.

        Args: adapter is the recognised CaptureAdapter; bgr_frame is the
            first frame the BGR path delivered.
        Returns: The first frame of the path in use, as delivered. Logs one
            line naming the path, the layout and both brightness means.
        """
        forced = luma_setting() == "bgr"
        route = "" if forced else raw_route(self.info)
        if not route:
            reason = (f"{LUMA_SETTING}=bgr" if forced
                      else "this backend cannot deliver raw YUY2")
            self.log.write(f"Analog luma: {self.luma_path} through "
                           f"{backend_name(self.backend)} ({reason}).")
            return bgr_frame
        reference = self.first_frame
        try:
            raw_frame, luma = (self._raw_v4l2(adapter) if route == "v4l2"
                               else self._raw_msmf(adapter))
            expanded = expand_video_range(luma)
            gap = abs(float(expanded.mean()) - float(reference.mean()))
            if gap > RAW_MATCH_TOLERANCE:
                raise ValueError(
                    f"its mean brightness {expanded.mean():.1f} is {gap:.1f} levels "
                    f"from the BGR path's {reference.mean():.1f} (input, format or range differs)")
        except Exception as exc:
            return self._keep_bgr(adapter, route, exc, bgr_frame)
        self.first_frame = expanded
        self._raw_size = (adapter.width, adapter.height)
        self.luma_path = "raw YUY2 luma, video range 16–235 expanded to 0–255"
        self.log.write(
            f"Analog luma: {self.luma_path} through {backend_name(self.backend)} "
            f"(frame {describe_layout(raw_frame)}; raw Y {int(luma.min())}–{int(luma.max())}; "
            f"mean {expanded.mean():.1f}, BGR path {reference.mean():.1f}).")
        return raw_frame

    def _raw_v4l2(self, adapter):
        """Turn off V4L2's conversion on the open device and read raw YUYV.

        Args: adapter is the recognised CaptureAdapter (its size is checked).
        Returns: (frame as delivered, 2-D luma). Raises ValueError when the
            device does not deliver YUYV of the adapter's size.
        """
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        return self._read_luma(adapter, 1)

    def _raw_msmf(self, adapter):
        """Re-open the device through Media Foundation without conversion.

        DirectShow is released first (the device serves one program at a
        time); it has already routed the composite input.

        Args: adapter is the recognised CaptureAdapter.
        Returns: (frame as delivered, 2-D luma); self.cap is the Media
            Foundation device. Raises LookupError or ValueError when the device
            is not listed, does not open or does not deliver YUY2 of the size.
        """
        index = find_msmf_index(self.info)
        self.close()
        self.backend = cv2.CAP_MSMF
        self.cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            raise ValueError("Media Foundation did not open it")
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, adapter.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, adapter.height)
        return self._read_luma(adapter, RAW_READ_ATTEMPTS)

    def _read_luma(self, adapter, attempts):
        """Read a raw frame and take its luma, checking format and size.

        Args: adapter is the recognised CaptureAdapter; attempts is how many
            reads may come back empty while the device starts.
        Returns: (frame as delivered, 2-D luma). Raises ValueError otherwise.
        """
        fourcc = fourcc_text(self.cap.get(cv2.CAP_PROP_FOURCC))
        if fourcc not in YUY2_FOURCCS:
            raise ValueError(f"it delivers {fourcc or 'an unknown format'}, not YUY2")
        size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if size != (adapter.width, adapter.height):
            raise ValueError(f"it delivers {size[0]} × {size[1]}, not "
                             f"{adapter.width} × {adapter.height}")
        for _ in range(attempts):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                break
        else:
            raise ValueError("no raw frame arrived")
        luma = yuy2_luma(frame, *size)
        if luma is None:
            raise ValueError(f"a {describe_layout(frame)} frame is not YUY2 of that size")
        return frame, luma

    def _keep_bgr(self, adapter, route, problem, bgr_frame):
        """Fall back to the BGR path after the raw path failed its checks.

        Args: adapter is the recognised CaptureAdapter; route is "msmf" or
            "v4l2"; problem is the exception that ruled the raw path out;
            bgr_frame is the BGR path's first frame.
        Returns: The first frame of the BGR path, as delivered. Raises
            RuntimeError, as open() does, when the device cannot be re-opened.
        """
        frame = bgr_frame
        if route == "v4l2":
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        elif self.backend != self.info.backend:  # DirectShow was released
            self.close()
            frame = self._open_bgr(adapter)
        where = "Media Foundation" if route == "msmf" else "V4L2"
        self.log.write(f"Analog luma: {self.luma_path} through "
                       f"{backend_name(self.backend)}; raw YUY2 through {where} "
                       f"was not used: {problem}.")
        return frame

    def grab(self, timeout_ms):
        """Wait for the next frame and convert it to Mono8.

        Args: timeout_ms is how long a failed read waits before returning, so
            an unplugged device cannot spin the caller; OpenCV's read() itself
            blocks for about one frame and cannot be bounded.
        Returns: (frame, "ok") with a 2-D uint8 array, or (exception,
            "error") when the device sent nothing or an unusable frame.
        """
        try:
            ok, frame = self.cap.read()
        except cv2.error as exc:
            time.sleep(timeout_ms / 1000.0)  # a dead driver raises at once; do not spin
            return exc, "error"
        if not ok or frame is None:
            time.sleep(timeout_ms / 1000.0)
            return RuntimeError("the capture device returned no frame"), "error"
        try:
            image = self._to_luma(frame)
        except ValueError as exc:
            return exc, "error"
        self._frames += 1
        return image, "ok"

    def _to_luma(self, frame):
        """Convert one delivered frame with the luma path chosen at open.

        Args: frame is what VideoCapture.read() returned.
        Returns: A 2-D uint8 array. Raises ValueError for an unusable frame.
        """
        if self._raw_size is None:
            return to_mono8(frame)
        luma = yuy2_luma(frame, *self._raw_size)
        if luma is None:
            raise ValueError(f"a {describe_layout(frame)} frame is not the raw YUY2 "
                             "the device delivered at first")
        return expand_video_range(luma)

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

    def read_stream_counters(self):
        """Report no transport counters: OpenCV capture exposes none.

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
        rows += [("Luma path", self.luma_path),
                 ("First frame (raw)", self.raw_layout),
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
