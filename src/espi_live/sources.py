"""Choose between FLIR cameras and video-capture devices.

The engine and the channels ask this module which cameras exist, which ones
the viewer supports, which may be opened without being asked, and which
session class opens each. FLIR/Spinnaker cameras come first; video devices
follow, and only recognised capture adapters (such as the Hauppauge
USB-Live2 carrying a Sony XC-ST50) are opened automatically, so a webcam is
never turned on unless someone switches it on. No GUI imports.
"""

from . import camera, capture
from .capture_models import adapter_for


def detect_cameras(log):
    """List every FLIR camera and video device without opening any.

    Args: log has a write(message) method.
    Returns: A list of CameraInfo: FLIR cameras in SDK order, then video
        devices. A source that cannot be listed is logged and left out.
    """
    try:
        flir = camera.list_cameras(log)
    except ImportError as exc:
        log.write(f"FLIR cameras are not listed: PySpin is not available ({exc}).")
        flir = []
    except Exception as exc:
        log.write(f"Could not list FLIR cameras: {exc}")
        flir = []
    return flir + capture.list_capture_devices(log)


def is_supported(info):
    """Report whether the viewer supports a detected camera.

    Args: info is a CameraInfo.
    Returns: True for FLIR cameras and recognised capture adapters.
    """
    if info.kind != "capture":
        return True
    return adapter_for(info.vid, info.pid, info.model) is not None


def auto_open_order(infos):
    """Order the cameras startup may open without being asked.

    Args: infos is a list of CameraInfo from detect_cameras().
    Returns: FLIR cameras, then supported capture devices; unsupported
        video devices are left out.
    """
    flir = [info for info in infos if info.kind != "capture"]
    adapters = [info for info in infos if info.kind == "capture" and is_supported(info)]
    return flir + adapters


def make_session(log, info):
    """Create the session that opens one camera.

    Args: log has a write(message) method; info is the camera's CameraInfo.
    Returns: An unopened CaptureSession or CameraSession.
    """
    if info.kind == "capture":
        return capture.CaptureSession(log, info)
    return camera.CameraSession(log, info.serial)


def open_first_live(log):
    """Open the first FLIR camera, else the first recognised capture adapter.

    Used to leave demo mode. Unsupported video devices are never tried.

    Args: log has a write(message) method.
    Returns: An opened session. Raises the last failed attempt's error when
        nothing opens, so a capture adapter's actionable message (close
        WinTV / Hauppauge Capture) is the one the user sees; every failed
        attempt is logged first.
    """
    candidates = [None] + [info for info in capture.list_capture_devices(log)
                           if is_supported(info)]
    error = None
    for info in candidates:
        session = camera.CameraSession(log) if info is None else make_session(log, info)
        try:
            session.open()
            return session
        except Exception as exc:
            label = "the FLIR camera" if info is None else info.label
            log.write(f"Could not open {label}: {exc}")
            error = exc
    raise error
