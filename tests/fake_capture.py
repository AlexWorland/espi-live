"""In-memory stand-ins for OpenCV video capture and device enumeration.

Only the surface espi_live.capture uses is modelled: cv2.VideoCapture
(isOpened, read, get, set, release) and the cv2_enumerate_cameras module
(enumerate_cameras returning objects with index, name, path, vid, pid and
backend). A device can also deliver raw YUY2 once CAP_PROP_CONVERT_RGB is 0
(as V4L2 and Media Foundation do), in any of the layouts backends use, or
ignore the flag. No real device is ever opened.
"""

import types

import cv2
import numpy as np

DSHOW_BASE = 700  # OpenCV encodes the backend in an index's high digits
MSMF = 1400       # cv2.CAP_MSMF
YUY2_CODE = float(cv2.VideoWriter_fourcc(*"YUY2"))
YUYV_CODE = float(cv2.VideoWriter_fourcc(*"YUYV"))


class ListLog:
    """Collect log messages in memory, with their levels."""

    def __init__(self):
        """Create an empty log.

        Args: None.
        Returns: None.
        """
        self.lines = []
        self.records = []  # (level name, message, exc_info)

    def _add(self, level, message, exc_info=None):
        """Record one message.

        Args: level is the level name; message the text; exc_info as logging takes it.
        Returns: None.
        """
        self.lines.append(message)
        self.records.append((level, message, exc_info))

    def write(self, message):
        """Record one INFO message.

        Args: message is the text to keep.
        Returns: None.
        """
        self._add("INFO", message)

    def debug(self, message):
        """Record one DEBUG message.

        Args: message is the text to keep.
        Returns: None.
        """
        self._add("DEBUG", message)

    def info(self, message):
        """Record one INFO message.

        Args: message is the text to keep.
        Returns: None.
        """
        self._add("INFO", message)

    def warning(self, message):
        """Record one WARNING message.

        Args: message is the text to keep.
        Returns: None.
        """
        self._add("WARNING", message)

    def error(self, message, exc_info=None):
        """Record one ERROR message.

        Args: message is the text to keep; exc_info is kept for assertions.
        Returns: None.
        """
        self._add("ERROR", message, exc_info)

    def exception(self, message):
        """Record one ERROR message with the current exception.

        Args: message is the text to keep.
        Returns: None.
        """
        self._add("ERROR", message, True)


def bgr_frame(width=720, height=480, value=100):
    """Build a flat BGR frame as DirectShow delivers it.

    Args: width and height are the frame size; value fills every channel.
    Returns: A (height, width, 3) uint8 array.
    """
    return np.full((height, width, 3), value, dtype=np.uint8)


def interlaced_frame(width=720, height=480):
    """Build a BGR frame whose two fields differ: even rows 0, odd rows 255.

    Args: width and height are the frame size.
    Returns: A (height, width, 3) uint8 array.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[1::2] = 255
    return frame


def yuy2_frame(luma, layout="pairs", chroma=128, stride=0):
    """Pack a luma image as YUY2 bytes in one of the backends' layouts.

    Args: luma is a 2-D uint8 array; layout is "pairs" ((H, W, 2), V4L2),
        "row" ((1, N) bytes, Media Foundation without conversion) or "wide"
        ((H, 2W)); chroma fills the U and V bytes; stride, when larger than
        2W, pads each row of a "row" frame to that many bytes.
    Returns: The packed uint8 array.
    """
    height, width = luma.shape
    pairs = np.empty((height, width, 2), dtype=np.uint8)
    pairs[:, :, 0] = luma
    pairs[:, :, 1] = chroma
    if layout == "pairs":
        return pairs
    rows = pairs.reshape(height, 2 * width)
    if layout == "wide":
        return rows
    if stride > 2 * width:
        rows = np.concatenate(
            [rows, np.zeros((height, stride - 2 * width), dtype=np.uint8)], axis=1)
    return rows.reshape(1, -1)


def ramp_luma(width=720, height=480, low=40, high=200):
    """Build a horizontal video-range ramp as a decoder would send it.

    Args: width and height are the frame size; low and high are the end codes.
    Returns: A (height, width) uint8 array.
    """
    row = np.linspace(low, high, width).round().astype(np.uint8)
    return np.tile(row, (height, 1))


class FakeVideoCapture:
    """A video device that replays frames and stores property writes."""

    def __init__(self, index, backend, frames, opened, props, raw_frames=None,
                 set_error=None, fixed=()):
        """Create a device as cv2.VideoCapture(index, backend) would.

        Args: index and backend are the open arguments; frames is a list
            replayed in order (its last item repeats; None is a failed read)
            or a callable returning (ok, frame); opened is what isOpened()
            reports; props maps CAP_PROP_* numbers to values; raw_frames,
            if set, is replayed like frames while CAP_PROP_CONVERT_RGB is 0
            (None: the device ignores the flag); set_error, if set, is raised
            by every set(); fixed lists properties whose writes the device
            ignores (a size it cannot deliver).
        Returns: None.
        """
        self.index = index
        self.backend = backend
        self.frames = frames
        self.raw_frames = raw_frames
        self.opened = opened
        self.props = dict(props or {})
        self.set_error = set_error
        self.fixed = frozenset(fixed)
        self.sets = []
        self.reads = 0
        self.released = False

    def isOpened(self):
        """Report whether the device opened.

        Args: None.
        Returns: The configured flag.
        """
        return self.opened

    def read(self):
        """Return the next frame.

        Args: None.
        Returns: (ok, frame); (False, None) for a failed read.
        """
        self.reads += 1
        frames = self.frames
        if self.raw_frames is not None and self.props.get(cv2.CAP_PROP_CONVERT_RGB) == 0:
            frames = self.raw_frames
        if callable(frames):
            return frames()
        frame = frames.pop(0) if len(frames) > 1 else frames[0]
        return frame is not None, frame

    def get(self, prop):
        """Read a property.

        Args: prop is a CAP_PROP_* number.
        Returns: The stored value, or 0.0 when unset (as OpenCV does).
        """
        return self.props.get(prop, 0.0)

    def set(self, prop, value):
        """Write a property and remember the call.

        Args: prop is a CAP_PROP_* number; value is the new value.
        Returns: True. Raises the configured set_error.
        """
        if self.set_error is not None:
            raise self.set_error
        self.sets.append((prop, value))
        if prop not in self.fixed:
            self.props[prop] = value
        return True

    def release(self):
        """Release the device.

        Args: None.
        Returns: None.
        """
        self.released = True


class FakeCaptureFactory:
    """Stand-in for cv2.VideoCapture that records every device it makes."""

    def __init__(self, frames=None, opened=True, props=None, raw_frames=None,
                 backends=None, fixed=()):
        """Configure the devices to make.

        Args: frames is a list (copied for each device) or a callable, as for
            FakeVideoCapture; None replays one flat BGR frame; opened, props,
            raw_frames and fixed are passed to every device; backends maps an
            OpenCV backend number to a dict of FakeVideoCapture arguments
            (frames, opened, props, raw_frames, set_error, fixed) that replace
            these for devices opened through it.
        Returns: None.
        """
        self.frames = [bgr_frame()] if frames is None else frames
        self.opened = opened
        self.props = props
        self.raw_frames = raw_frames
        self.fixed = fixed
        self.backends = dict(backends or {})
        self.made = []

    def __call__(self, index, backend):
        """Make a device, like cv2.VideoCapture(index, backend).

        Args: index and backend are the open arguments.
        Returns: A FakeVideoCapture.
        """
        options = {"frames": self.frames, "opened": self.opened, "props": self.props,
                   "raw_frames": self.raw_frames, "fixed": self.fixed}
        options.update(self.backends.get(backend, {}))
        for key in ("frames", "raw_frames"):
            if options[key] is not None and not callable(options[key]):
                options[key] = list(options[key])
        device = FakeVideoCapture(index, backend, **options)
        self.made.append(device)
        return device


def device(index, name, path="", vid=None, pid=None, backend=DSHOW_BASE):
    """Describe one enumerated device as cv2_enumerate_cameras does.

    Args: index, name, path, vid, pid and backend are the reported fields.
    Returns: A namespace with those attributes.
    """
    return types.SimpleNamespace(index=index, name=name, path=path, vid=vid,
                                 pid=pid, backend=backend)


def usb_live2(position=0):
    """Describe an enumerated Hauppauge USB-Live2.

    Args: position is its place in the device list.
    Returns: A device namespace with the USB-Live2's USB ID.
    """
    return device(DSHOW_BASE + position, "Hauppauge USB-Live2",
                  path=f"\\\\?\\usb#vid_2040&pid_c200#{position}",
                  vid=0x2040, pid=0xC200)


def usb_live2_rev_e(backend=DSHOW_BASE, position=0, interface="65e8773d"):
    """Describe the Rev E1 USB-Live2 (2040:C220) as one backend lists it.

    DirectShow and Media Foundation list the same device instance under
    different interface classes, so only the part of the path after the
    last "#" differs.

    Args: backend is the listing backend; position is its place in the list;
        interface is the start of the interface-class GUID in its path.
    Returns: A device namespace.
    """
    path = (f"\\\\?\\usb#vid_2040&pid_c220&mi_00#6&2f0c1a3&0&0000"
            f"#{{{interface}-0000-0000-0000-000000000000}}\\global")
    return device(position, "Hauppauge USB-Live2", path=path, vid=0x2040,
                  pid=0xC220, backend=backend)


def webcam(position=0):
    """Describe an enumerated webcam that is not a recognised adapter.

    Args: position is its place in the device list.
    Returns: A device namespace with a Logitech USB ID.
    """
    return device(DSHOW_BASE + position, "Webcam C270",
                  path=f"\\\\?\\usb#vid_046d&pid_0825#{position}",
                  vid=0x046D, pid=0x0825)


def integrated_webcam(position=0, name="Integrated Webcam"):
    """Describe the Dell laptop camera reported in issue #44 (built in).

    Args: position is its place in the device list; name is the reported
        name ("Integrated IR Webcam" for the module's IR camera).
    Returns: A device namespace with the Microdia/Sonix USB ID 0c45:674c.
    """
    return device(DSHOW_BASE + position, name,
                  path=f"\\\\?\\usb#vid_0c45&pid_674c&mi_00#{position}",
                  vid=0x0C45, pid=0x674C)


class FakeEnumerator:
    """Stand-in for the cv2_enumerate_cameras module."""

    def __init__(self, devices=(), error=None):
        """Configure what enumeration returns.

        Args: devices is the list to return; error, if set, is raised instead.
        Returns: None.
        """
        self.devices = list(devices)
        self.error = error
        self.calls = []

    def enumerate_cameras(self, api):
        """List the devices for one backend.

        Args: api is the OpenCV backend asked for.
        Returns: The configured devices. Raises the configured error.
        """
        self.calls.append(api)
        if self.error is not None:
            raise self.error
        return list(self.devices)
