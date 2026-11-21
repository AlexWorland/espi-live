"""Choose between FLIR cameras, video-capture devices and iPhones.

The engine and the channels ask this module which cameras exist, which ones
the viewer supports, which may be opened without being asked, and which
session class opens each. FLIR/Spinnaker cameras come first; video devices
follow, and only recognised capture adapters (such as the Hauppauge
USB-Live2 carrying a Sony XC-ST50) are opened automatically, so a webcam is
never turned on unless someone switches it on. iPhones plugged in over USB
(running the ESPI Camera app, see iphone.py) come last. No GUI imports.
"""

from . import camera, capture, iphone
from .capture_models import adapter_for


def detect_cameras(log):
    """List every FLIR camera and video device without opening any.

    Args: log has a write(message) method.
    Returns: A list of CameraInfo: FLIR cameras in SDK order, then video
        devices, then iPhones. A source that cannot be listed is logged and
        left out.
    """
    try:
        flir = camera.list_cameras(log)
    except ImportError as exc:
        log.write(f"FLIR cameras are not listed: PySpin is not available ({exc}).")
        flir = []
    except Exception as exc:
        log.write(f"Could not list FLIR cameras: {exc}")
        flir = []
    return flir + capture.list_capture_devices(log) + iphone.list_iphones(log)


def is_supported(info):
    """Report whether the viewer supports a detected camera.

    Args: info is a CameraInfo.
    Returns: True for FLIR cameras, iPhones and recognised capture adapters.
    """
    if info.kind != "capture":
        return True
    return adapter_for(info.vid, info.pid, info.model) is not None


def auto_open_order(infos):
    """Order the cameras startup may open without being asked.

    Args: infos is a list of CameraInfo from detect_cameras().
    Returns: FLIR cameras, then supported capture devices, then iPhones
        (each phone's main camera only: its other cameras share the one
        link, so they are opened only when chosen); unsupported video
        devices are left out.
    """
    flir = [info for info in infos if info.kind == "spinnaker"]
    adapters = [info for info in infos if info.kind == "capture" and is_supported(info)]
    phones = iphone.main_lenses([info for info in infos if info.kind == "iphone"])
    return flir + adapters + phones


def make_session(log, info, exposure_us=8000.0, gain_db=0.0, fps=30.0):
    """Create the session that opens one camera.

    Args: log has a write(message) method; info is the camera's CameraInfo;
        exposure_us, gain_db and fps are the starting settings (ignored for
        a capture device, which has nothing adjustable).
    Returns: An unopened CaptureSession, IPhoneSession or CameraSession.
    """
    if info.kind == "capture":
        return capture.CaptureSession(log, info)
    if info.kind == "iphone":
        return iphone.IPhoneSession(log, info, exposure_us, gain_db, fps)
    return camera.CameraSession(log, info.serial, exposure_us, gain_db, fps)


def open_first_live(log, exposure_us=8000.0, gain_db=0.0, fps=30.0):
    """Open the first FLIR camera, else a recognised capture adapter or iPhone.

    Used to leave demo mode. Unsupported video devices are never tried.

    Args: log has a write(message) method; exposure_us, gain_db and fps are
        written when the FLIR camera opens, so it starts at the engine's
        current settings rather than the hard-coded defaults.
    Returns: An opened session. Raises the last failed attempt's error when
        nothing opens, so a capture adapter's actionable message (close
        WinTV / Hauppauge Capture) is the one the user sees; every failed
        attempt is logged first.
    """
    candidates = ([None] + [info for info in capture.list_capture_devices(log)
                            if is_supported(info)]
                  + iphone.main_lenses(iphone.list_iphones(log)))
    error = None
    for info in candidates:
        session = (camera.CameraSession(log, None, exposure_us, gain_db, fps)
                   if info is None else
                   make_session(log, info, exposure_us, gain_db, fps))
        try:
            session.open()
            return session
        except Exception as exc:
            label = "the FLIR camera" if info is None else info.label
            log.write(f"Could not open {label}: {exc}")
            error = exc
    raise error
