"""In-memory stand-ins for OpenCV video capture and device enumeration.

Only the surface espi_live.capture uses is modelled: cv2.VideoCapture
(isOpened, read, get, set, release) and the cv2_enumerate_cameras module
(enumerate_cameras returning objects with index, name, path, vid, pid and
backend). No real device is ever opened.
"""

import types

import numpy as np

DSHOW_BASE = 700  # OpenCV encodes the backend in an index's high digits


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


class FakeVideoCapture:
    """A video device that replays frames and stores property writes."""

    def __init__(self, index, backend, frames, opened, props):
        """Create a device as cv2.VideoCapture(index, backend) would.

        Args: index and backend are the open arguments; frames is a list
            replayed in order (its last item repeats; None is a failed read)
            or a callable returning (ok, frame); opened is what isOpened()
            reports; props maps CAP_PROP_* numbers to values.
        Returns: None.
        """
        self.index = index
        self.backend = backend
        self.frames = frames
        self.opened = opened
        self.props = dict(props or {})
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
        if callable(self.frames):
            return self.frames()
        frame = self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]
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
        Returns: True.
        """
        self.sets.append((prop, value))
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

    def __init__(self, frames=None, opened=True, props=None):
        """Configure the devices to make.

        Args: frames is a list (copied for each device) or a callable, as for
            FakeVideoCapture; None replays one flat BGR frame; opened and
            props are passed to every device.
        Returns: None.
        """
        self.frames = [bgr_frame()] if frames is None else frames
        self.opened = opened
        self.props = props
        self.made = []

    def __call__(self, index, backend):
        """Make a device, like cv2.VideoCapture(index, backend).

        Args: index and backend are the open arguments.
        Returns: A FakeVideoCapture.
        """
        frames = self.frames if callable(self.frames) else list(self.frames)
        device = FakeVideoCapture(index, backend, frames, self.opened, self.props)
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


def webcam(position=0):
    """Describe an enumerated webcam that is not a recognised adapter.

    Args: position is its place in the device list.
    Returns: A device namespace with a Logitech USB ID.
    """
    return device(DSHOW_BASE + position, "Webcam C270",
                  path=f"\\\\?\\usb#vid_046d&pid_0825#{position}",
                  vid=0x046D, pid=0x0825)


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
