"""PySpin camera access: node helpers, configuration and session lifecycle.

PySpin is imported lazily (see load_pyspin) so demo mode and tests never need
the SDK. Read docs/camera/ before changing anything here.

Threading rule: a CameraSession must only be used by one thread at a time.
The engine opens it, uses it, and closes it on its worker thread. Several
sessions (one per camera, each on its own engine thread) share the PySpin
System singleton through acquire_system()/release_system(), because
System.ReleaseInstance() fails while any other camera is still referenced.
"""

import importlib
import threading
from dataclasses import dataclass

import numpy as np

from .camera_models import CameraIdentity

PySpin = None  # Imported only when a real camera is requested.

# Device-limit query names and the UI ranges they are clipped to.
CAMERA_RANGES = {
    "ExposureTime": (0.0, 50000.0),
    "Gain": (0.0, 30.0),
    "AcquisitionFrameRate": (0.0, 60.0),
}


_system_lock = threading.Lock()
_system = None
_system_module = None  # the PySpin module _system came from
_system_users = 0


@dataclass(frozen=True)
class CameraInfo:
    """Identity of one detected camera, video device or synthetic demo source.

    kind is "spinnaker" for FLIR cameras or "capture" for OpenCV video
    devices; a capture device's serial is a short "video N" id, and index,
    backend, vid, pid and path are what the device enumerator reported.
    """

    serial: str
    model: str = ""
    kind: str = "spinnaker"
    index: int | None = None
    backend: int | None = None
    vid: int | None = None
    pid: int | None = None
    path: str = ""

    @property
    def label(self):
        """Name the camera for menus and log lines.

        Args: None.
        Returns: "model (serial)", or just the serial without a model name.
        """
        return f"{self.model} ({self.serial})" if self.model else self.serial


def load_pyspin():
    """Import PySpin on first use.

    Args: None.
    Returns: The PySpin module. Raises ImportError if it is not installed.
    """
    global PySpin
    if PySpin is None:
        PySpin = importlib.import_module("PySpin")
    return PySpin


def acquire_system():
    """Take a shared reference to the PySpin System singleton.

    Args: None. PySpin must already be loaded.
    Returns: The System instance; pair every call with release_system().
    """
    global _system, _system_module, _system_users
    with _system_lock:
        if _system is None or _system_module is not PySpin:
            _system = PySpin.System.GetInstance()
            _system_module = PySpin
            _system_users = 0
        _system_users += 1
        return _system


def release_system():
    """Drop a reference from acquire_system(); the last one releases the SDK.

    Args: None.
    Returns: None. Raises whatever System.ReleaseInstance() raises; the
        reference is dropped either way.
    """
    global _system, _system_users
    with _system_lock:
        _system_users = max(0, _system_users - 1)
        if _system_users or _system is None:
            return
        system, _system = _system, None
        system.ReleaseInstance()


def camera_info(cam, index):
    """Identify a camera from its transport-layer node map (no Init needed).

    Args: cam is a PySpin camera; index is its list position, used as the
        name when the serial number cannot be read.
    Returns: A CameraInfo.
    """
    nodemap = cam.GetTLDeviceNodeMap()
    serial = read_string(nodemap, "DeviceSerialNumber") or f"camera {index + 1}"
    return CameraInfo(serial, read_string(nodemap, "DeviceModelName"))


def list_cameras(log):
    """Enumerate the connected FLIR cameras without opening them.

    Args: log has a write(message) method for cleanup errors.
    Returns: A list of CameraInfo in SDK order. Raises ImportError without
        PySpin, or the SDK's error when enumeration fails.
    """
    load_pyspin()
    system = acquire_system()
    cam_list = None
    try:
        cam_list = system.GetCameras()
        return [camera_info(cam_list.GetByIndex(index), index)
                for index in range(cam_list.GetSize())]
    finally:
        try:
            if cam_list is not None:
                cam_list.Clear()
        finally:
            try:
                release_system()
            except Exception as exc:
                log.write(f"System cleanup error: {exc}")


def set_enum(nodemap, name, entry_name):
    """Set an available, writable GenICam enumeration.

    Args: nodemap is the camera or stream map; name identifies the node;
        entry_name identifies the requested enum entry.
    Returns: True if set, or False when unsupported or read-only.
    """
    node = PySpin.CEnumerationPtr(nodemap.GetNode(name))
    if not PySpin.IsWritable(node):
        return False
    entry = PySpin.CEnumEntryPtr(node.GetEntryByName(entry_name))
    if PySpin.IsReadable(entry):
        node.SetIntValue(entry.GetValue())
        return True
    return False


def set_bool(nodemap, name, value):
    """Set an available, writable GenICam boolean.

    Args: nodemap is the camera map; name identifies the node; value is the
        requested boolean state.
    Returns: True if set, or False when read-only.
    """
    node = PySpin.CBooleanPtr(nodemap.GetNode(name))
    if PySpin.IsWritable(node):
        node.SetValue(bool(value))
        return True
    return False


def set_float(nodemap, name, value):
    """Clamp and set an available GenICam floating-point value.

    Args: nodemap is the camera map; name identifies the node; value is the
        requested number in that node's native units.
    Returns: True if set, or False when unsupported or read-only.
    """
    node = PySpin.CFloatPtr(nodemap.GetNode(name))
    if PySpin.IsReadable(node) and PySpin.IsWritable(node):
        node.SetValue(float(np.clip(value, node.GetMin(), node.GetMax())))
        return True
    return False


def readable_float_node(nodemap, name):
    """Find a readable floating-point node for limits or live readback.

    Args: nodemap is a GenICam map; name identifies the requested node.
    Returns: The PySpin float node, or None when it cannot be read.
    """
    try:
        node = PySpin.CFloatPtr(nodemap.GetNode(name))
        return node if PySpin.IsReadable(node) else None
    except PySpin.SpinnakerException:
        return None


def try_setting(log, nodemap, setter, name, value):
    """Apply a camera setting while reporting PySpin failures.

    Args: log has a write(message) method; nodemap owns the node; setter is a
        setting function; name and value identify the requested setting.
    Returns: None.
    """
    try:
        if not setter(nodemap, name, value):
            log.write(f"Camera setting unavailable or read-only: {name}")
    except PySpin.SpinnakerException as exc:
        log.write(f"Could not set {name}: {exc}")


def read_string(nodemap, name):
    """Read a GenICam string node, tolerating absent or unreadable nodes.

    Args: nodemap is a GenICam map; name identifies the node.
    Returns: The stripped value, or "" when it cannot be read.
    """
    try:
        node = PySpin.CStringPtr(nodemap.GetNode(name))
        return str(node.GetValue()).strip() if PySpin.IsReadable(node) else ""
    except PySpin.SpinnakerException:
        return ""


def read_int(nodemap, name):
    """Read a GenICam integer node, tolerating absent or unreadable nodes.

    Args: nodemap is a GenICam map; name identifies the node.
    Returns: The value, or 0 when it cannot be read.
    """
    try:
        node = PySpin.CIntegerPtr(nodemap.GetNode(name))
        return int(node.GetValue()) if PySpin.IsReadable(node) else 0
    except PySpin.SpinnakerException:
        return 0


def read_identity(cam):
    """Ask an initialized camera what model it is.

    Vendor, model and serial come from the transport-layer device map (as in
    the SDK's Enumeration example), falling back to the device map; sensor,
    firmware and full-sensor size come from the device map.

    Args: cam is an initialized PySpin camera.
    Returns: A CameraIdentity; fields the camera does not report are empty.
    """
    device = cam.GetNodeMap()
    maps = (cam.GetTLDeviceNodeMap(), device)

    def first(name):
        """Read a string from the first map that has it.

        Args: name identifies the node.
        Returns: The value, or "".
        """
        return next((value for value in (read_string(m, name) for m in maps)
                     if value), "")

    return CameraIdentity(
        vendor=first("DeviceVendorName"), model=first("DeviceModelName"),
        serial=first("DeviceSerialNumber"),
        firmware=read_string(device, "DeviceFirmwareVersion"),
        sensor=read_string(device, "SensorDescription"),
        width=read_int(device, "SensorWidth") or read_int(device, "WidthMax"),
        height=read_int(device, "SensorHeight") or read_int(device, "HeightMax"))


def configure_camera(cam, log, exposure_us=8000.0, gain_db=0.0, fps=30.0):
    """Configure Mono8 continuous preview at a starting exposure, gain and rate.

    Args: cam is an initialized PySpin camera; log reports unsupported writes;
        exposure_us, gain_db and fps are written to ExposureTime, Gain and
        AcquisitionFrameRate. Omitted, they are the viewer defaults
        (8000 µs, 0 dB, 30 fps).
    Returns: The camera's GenICam node map.
    """
    nodemap = cam.GetNodeMap()
    for name, value in (("PixelFormat", "Mono8"),
                        ("AcquisitionMode", "Continuous"),
                        ("ExposureAuto", "Off"), ("GainAuto", "Off")):
        try_setting(log, nodemap, set_enum, name, value)
    try_setting(log, nodemap, set_float, "ExposureTime", exposure_us)
    try_setting(log, nodemap, set_float, "Gain", gain_db)
    try_setting(log, nodemap, set_bool, "AcquisitionFrameRateEnable", True)
    try_setting(log, nodemap, set_float, "AcquisitionFrameRate", fps)
    try_setting(log, cam.GetTLStreamNodeMap(), set_enum,
                "StreamBufferHandlingMode", "NewestOnly")
    return nodemap


class CameraSession:
    """Own a PySpin camera and its complete acquisition lifecycle."""

    interlaced = False  # FLIR cameras deliver progressive frames

    def __init__(self, log, serial=None, exposure_us=8000.0, gain_db=0.0, fps=30.0):
        """Initialize empty camera handles for a connection attempt.

        Args: log has a write(message) method for diagnostics; serial picks
            a camera by DeviceSerialNumber, or None for the first camera;
            exposure_us, gain_db and fps are applied when the camera opens
            (8000 µs, 0 dB and 30 fps when omitted).
        Returns: None.
        """
        self.log = log
        self.serial = serial
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.fps = fps
        self.info = None
        self.system = None
        self.cam_list = None
        self.cam = None
        self.nodemap = None
        self.identity = CameraIdentity()
        self.initialized = False
        self.acquiring = False

    def open(self):
        """Detect, initialize, configure, and start the selected FLIR camera.

        Args: None.
        Returns: None; self.info identifies the opened camera. Raises when
            PySpin or the camera is unavailable; any partially opened
            resources are released first.
        """
        load_pyspin()
        try:
            self.system = acquire_system()
            self.cam_list = self.system.GetCameras()
            if self.cam_list.GetSize() < 1:
                raise RuntimeError("No FLIR/Spinnaker cameras detected.")
            for index in range(self.cam_list.GetSize()):
                cam = self.cam_list.GetByIndex(index)
                info = camera_info(cam, index)
                if self.serial is None or info.serial == self.serial:
                    self.cam, self.info = cam, info
                    break
            else:
                raise RuntimeError(f"Camera {self.serial} is no longer connected.")
            self.cam.Init()
            self.initialized = True
            self.identity = read_identity(self.cam)
            self.nodemap = configure_camera(
                self.cam, self.log, self.exposure_us, self.gain_db, self.fps)
            self.cam.BeginAcquisition()
            self.acquiring = True
        except Exception:
            self.close()
            raise

    def grab(self, timeout_ms):
        """Wait for the next complete Mono8 frame.

        Args: timeout_ms bounds the wait so the caller stays responsive.
        Returns: (frame, status) where status is "ok" with a NumPy copy that
            outlives the SDK buffer, or "incomplete"/"error" with frame None.
            For "error" the frame slot holds the exception instead.
        """
        try:
            image = self.cam.GetNextImage(timeout_ms)
        except PySpin.SpinnakerException as exc:
            return exc, "error"
        try:
            if image.IsIncomplete():
                return None, "incomplete"
            return image.GetNDArray().copy(), "ok"
        finally:
            image.Release()

    def set_float(self, name, value):
        """Write a clamped float setting, enabling frame-rate control if needed.

        Args: name is "ExposureTime", "Gain" or "AcquisitionFrameRate";
            value is in the node's native units.
        Returns: None; failures are logged.
        """
        if name == "AcquisitionFrameRate":
            try_setting(self.log, self.nodemap, set_bool,
                        "AcquisitionFrameRateEnable", True)
        try_setting(self.log, self.nodemap, set_float, name, value)

    def limits(self):
        """Read device limits for the adjustable settings.

        Args: None.
        Returns: {node name: (low, high)} clipped to CAMERA_RANGES; nodes that
            cannot be read are omitted.
        """
        result = {}
        for name, (lower, upper) in CAMERA_RANGES.items():
            node = readable_float_node(self.nodemap, name)
            if node is None:
                continue
            try:
                low = max(lower, node.GetMin())
                high = max(low, min(upper, node.GetMax()))
            except PySpin.SpinnakerException:
                continue
            result[name] = (low, high)
        return result

    def read_values(self):
        """Read applied settings and the camera's resulting frame rate.

        Args: None.
        Returns: A dict with keys exposure, gain, requested, resulting; each
            value is a float or None when unavailable.
        """
        values = {}
        for key, name in (("exposure", "ExposureTime"), ("gain", "Gain"),
                          ("requested", "AcquisitionFrameRate"),
                          ("resulting", "AcquisitionResultingFrameRate")):
            node = readable_float_node(self.nodemap, name)
            try:
                values[key] = node.GetValue() if node is not None else None
            except PySpin.SpinnakerException:
                values[key] = None
        return values

    def describe(self, values):
        """Format a read_values() result for the status line.

        Args: values is the dict returned by read_values().
        Returns: The status text.
        """
        return format_camera_values(values)

    def close(self):
        """Release acquisition, camera, list, and system in SDK-safe order.

        Args: None.
        Returns: None; cleanup errors are written to the log.
        """
        if self.acquiring:
            try:
                self.cam.EndAcquisition()
            except Exception as exc:
                self.log.write(f"EndAcquisition error: {exc}")
            self.acquiring = False
        if self.initialized:
            try:
                self.cam.DeInit()
            except Exception as exc:
                self.log.write(f"DeInit error: {exc}")
            self.initialized = False
        self.cam = None
        self.nodemap = None
        if self.cam_list is not None:
            try:
                self.cam_list.Clear()
            except Exception as exc:
                self.log.write(f"Camera-list cleanup error: {exc}")
            self.cam_list = None
        if self.system is not None:
            try:
                release_system()
            except Exception as exc:
                self.log.write(f"System cleanup error: {exc}")
            self.system = None


def format_camera_values(values):
    """Format a read_values() result for a status line.

    Args: values is the dict returned by CameraSession.read_values().
    Returns: Human-readable text; unavailable readings appear as n/a.
    """
    def reading(value, unit, digits):
        """Format one reading.

        Args: value is a float or None; unit is a suffix; digits is precision.
        Returns: Formatted text or n/a.
        """
        return f"{value:.{digits}f}{unit}" if value is not None else "n/a"

    return (f"Applied: {reading(values['exposure'], ' µs', 0)}, "
            f"{reading(values['gain'], ' dB', 1)}; "
            f"FPS set {reading(values['requested'], '', 1)}, "
            f"resulting {reading(values['resulting'], '', 1)}")
