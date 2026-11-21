"""In-memory stand-ins for usbmuxd and the ESPI Camera app's link.

FakeSocket models the socket surface usbmux.py and iphone.py use (connect,
settimeout, sendall, recv, recv_into, shutdown, close). Bytes "from the
peer" are queued with feed(); a blocked recv waits for them, so the
session's reader thread can be driven from a test. No real socket is ever
opened.
"""

import plistlib
import threading

import numpy as np

from espi_live import iphone, usbmux


class FakeSocket:
    """A scripted socket: feed() what the peer sends, read .sent for the rest."""

    def __init__(self, incoming=b"", connect_error=None, family=None, send_error=None):
        """Create a socket with optional queued input.

        Args: incoming is bytes the peer has already sent; connect_error is
            raised by connect(); family records the address family asked
            for; send_error is raised by sendall().
        Returns: None.
        """
        self.family = family
        self.incoming = bytearray(incoming)
        self.sent = bytearray()
        self.timeout = "unset"
        self.address = None
        self.connect_error = connect_error
        self.send_error = send_error
        self.closed = False
        self.eof = False
        self.shutdown_error = None
        self.close_error = None
        self.cond = threading.Condition()

    def feed(self, data):
        """Queue bytes from the peer.

        Args: data is bytes.
        Returns: None.
        """
        with self.cond:
            self.incoming += data
            self.cond.notify_all()

    def finish(self):
        """Make the peer close its side: recv returns b"" once the queue is empty.

        Args: None.
        Returns: None.
        """
        with self.cond:
            self.eof = True
            self.cond.notify_all()

    def connect(self, address):
        """Record the address, or fail.

        Args: address is what the client connected to.
        Returns: None.
        """
        self.address = address
        if self.connect_error is not None:
            raise self.connect_error

    def settimeout(self, value):
        """Record the timeout.

        Args: value is seconds or None.
        Returns: None.
        """
        self.timeout = value

    def sendall(self, data):
        """Keep sent bytes, or fail.

        Args: data is bytes.
        Returns: None.
        """
        if self.send_error is not None:
            raise self.send_error
        self.sent += data

    def _take(self, count):
        """Wait for input and take up to count bytes.

        Args: count is the most bytes to take.
        Returns: bytes, empty at end of stream. Raises OSError once closed.
        """
        with self.cond:
            while not self.incoming and not self.eof and not self.closed:
                self.cond.wait(5.0)
            if self.closed:
                raise OSError("socket closed")
            chunk = bytes(self.incoming[:count])
            del self.incoming[:count]
            return chunk

    def recv(self, count):
        """Read up to count bytes.

        Args: count is the buffer size.
        Returns: bytes; b"" at end of stream.
        """
        return self._take(count)

    def recv_into(self, view):
        """Read into a buffer.

        Args: view is a writable memoryview.
        Returns: The number of bytes written; 0 at end of stream.
        """
        chunk = self._take(len(view))
        view[:len(chunk)] = chunk
        return len(chunk)

    def shutdown(self, how):
        """Close both directions, waking a blocked reader.

        Args: how is ignored.
        Returns: None.
        """
        with self.cond:
            self.closed = True
            self.cond.notify_all()
        if self.shutdown_error is not None:
            raise self.shutdown_error

    def close(self):
        """Close the socket.

        Args: None.
        Returns: None.
        """
        with self.cond:
            self.closed = True
            self.cond.notify_all()
        if self.close_error is not None:
            raise self.close_error


class SocketFactory:
    """Replace socket.socket: hand out prepared FakeSockets in order."""

    def __init__(self, *sockets):
        """Queue the sockets to hand out.

        Args: sockets are FakeSocket objects.
        Returns: None.
        """
        self.sockets = list(sockets)
        self.families = []

    def __call__(self, family, kind):
        """Return the next prepared socket.

        Args: family and kind are what socket.socket received.
        Returns: A FakeSocket.
        """
        self.families.append(family)
        sock = self.sockets.pop(0)
        sock.family = family
        return sock


def mux_reply(payload, kind=usbmux.PLIST_TYPE):
    """Encode a usbmuxd reply.

    Args: payload is a dict (or raw bytes); kind is the header type.
    Returns: The message bytes.
    """
    body = payload if isinstance(payload, bytes) else plistlib.dumps(payload)
    return usbmux.HEADER.pack(usbmux.HEADER.size + len(body), 1, kind, 1) + body


def device_entry(device_id=3, serial="00008110-001A2B3C4D5E801E", connection="USB"):
    """Build one DeviceList entry.

    Args: device_id, serial and connection fill the entry.
    Returns: A dict as usbmuxd sends it.
    """
    return {"MessageType": "Attached", "DeviceID": device_id,
            "Properties": {"ConnectionType": connection, "DeviceID": device_id,
                           "SerialNumber": serial, "ProductID": 4776,
                           "LocationID": 0}}


def phone_settings(**overrides):
    """Build the app's settings dict.

    Args: overrides replace defaults.
    Returns: A dict.
    """
    settings = {"exposure_us": 8000.0, "iso": 32.0, "fps": 30.0, "lens_position": 0.5,
                "width": 1920, "height": 1080, "crop_width": 0, "crop_height": 0,
                "max_send_fps": 0.0}
    settings.update(overrides)
    return settings


def phone_limits(**overrides):
    """Build the app's limits dict.

    Args: overrides replace defaults.
    Returns: A dict.
    """
    limits = {"exposure_us": [14.0, 1000000.0], "iso": [32.0, 3200.0],
              "fps": [1.0, 60.0], "lens_position": [0.0, 1.0]}
    limits.update(overrides)
    return limits


def hello(**overrides):
    """Build a HELLO payload.

    Args: overrides replace top-level keys.
    Returns: A dict.
    """
    data = {"protocol_version": iphone.VERSION, "app": "ESPI Camera",
            "app_version": "0.1.0 (5)", "device_model": "iPhone16,1",
            "system_version": "27.1.0", "camera": "Back wide camera",
            "pixel_format": "mono8",
            "formats": [{"width": 1280, "height": 720, "min_fps": 1, "max_fps": 60,
                         "binned": False},
                        {"width": 1920, "height": 1080, "min_fps": 1, "max_fps": 30,
                         "binned": False}],
            "limits": phone_limits(), "settings": phone_settings()}
    data.update(overrides)
    return data


def status(**overrides):
    """Build a STATUS payload.

    Args: overrides replace top-level keys.
    Returns: A dict.
    """
    data = {"settings": phone_settings(), "limits": phone_limits(),
            "thermal": "nominal", "camera_fps": 30.0, "sent_fps": 29.5,
            "sent_frames": 100, "dropped_frames": 0, "focus_locked": True}
    data.update(overrides)
    return data


def frame_bytes(width=8, height=4, value=None, **meta):
    """Encode one FRAME message.

    Args: width and height size the image; value fills it (default a
        gradient); meta are encode_frame keyword arguments.
    Returns: The message bytes.
    """
    if value is None:
        image = (np.arange(width * height) % 256).astype(np.uint8).reshape(height, width)
    else:
        image = np.full((height, width), value, np.uint8)
    meta.setdefault("exposure_us", 8000.0)
    meta.setdefault("iso", 64.0)
    return iphone.encode_frame(image, **meta)


def commands(sock):
    """Decode every COMMAND message a FakeSocket was sent.

    Args: sock is a FakeSocket used as the link.
    Returns: A list of dicts.
    """
    data, found = bytes(sock.sent), []
    while data:
        _magic, _version, kind, length, _ = iphone.HEADER.unpack_from(data)
        payload = data[iphone.HEADER.size:iphone.HEADER.size + length]
        if kind == iphone.KIND_COMMAND:
            found.append(iphone.decode_json(payload))
        data = data[iphone.HEADER.size + length:]
    return found


def phone_socket(greeting=None):
    """Make a link socket that has already sent its HELLO.

    Args: greeting is the HELLO dict (default hello()).
    Returns: A FakeSocket.
    """
    return FakeSocket(iphone.encode_json(iphone.KIND_HELLO,
                                         hello() if greeting is None else greeting))
