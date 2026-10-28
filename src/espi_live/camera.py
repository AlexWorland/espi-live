"""PySpin camera access: node helpers, configuration and session lifecycle.

PySpin is imported lazily (see load_pyspin) so demo mode and tests never need
the SDK. Read docs/camera/ before changing anything here.

Threading rule: a CameraSession must only be used by one thread at a time.
The engine opens it, uses it, and closes it on its worker thread.
"""

import importlib

import numpy as np

PySpin = None  # Imported only when a real camera is requested.

# Device-limit query names and the UI ranges they are clipped to.
CAMERA_RANGES = {
    "ExposureTime": (0.0, 50000.0),
    "Gain": (0.0, 30.0),
    "AcquisitionFrameRate": (0.0, 60.0),
}


def load_pyspin():
    """Import PySpin on first use.

    Args: None.
    Returns: The PySpin module. Raises ImportError if it is not installed.
    """
    global PySpin
    if PySpin is None:
        PySpin = importlib.import_module("PySpin")
    return PySpin


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


def configure_camera(cam, log):
    """Configure the camera for the existing Mono8 continuous-preview defaults.

    Args: cam is an initialized PySpin camera; log reports unsupported writes.
    Returns: The camera's GenICam node map.
    """
    nodemap = cam.GetNodeMap()
    for name, value in (("PixelFormat", "Mono8"),
                        ("AcquisitionMode", "Continuous"),
                        ("ExposureAuto", "Off"), ("GainAuto", "Off")):
        try_setting(log, nodemap, set_enum, name, value)
    try_setting(log, nodemap, set_float, "ExposureTime", 8000.0)
    try_setting(log, nodemap, set_float, "Gain", 0.0)
    try_setting(log, nodemap, set_bool, "AcquisitionFrameRateEnable", True)
    try_setting(log, nodemap, set_float, "AcquisitionFrameRate", 30.0)
    try_setting(log, cam.GetTLStreamNodeMap(), set_enum,
                "StreamBufferHandlingMode", "NewestOnly")
    return nodemap


class CameraSession:
    """Own a PySpin camera and its complete acquisition lifecycle."""

    def __init__(self, log):
        """Initialize empty camera handles for a connection attempt.

        Args: log has a write(message) method for diagnostics.
        Returns: None.
        """
        self.log = log
        self.system = None
        self.cam_list = None
        self.cam = None
        self.nodemap = None
        self.initialized = False
        self.acquiring = False

    def open(self):
        """Detect, initialize, configure, and start the first FLIR camera.

        Args: None.
        Returns: None. Raises when PySpin or a usable camera is unavailable;
            any partially opened resources are released first.
        """
        load_pyspin()
        try:
            self.system = PySpin.System.GetInstance()
            self.cam_list = self.system.GetCameras()
            if self.cam_list.GetSize() < 1:
                raise RuntimeError("No FLIR/Spinnaker cameras detected.")
            self.cam = self.cam_list.GetByIndex(0)
            self.cam.Init()
            self.initialized = True
            self.nodemap = configure_camera(self.cam, self.log)
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
                self.system.ReleaseInstance()
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
