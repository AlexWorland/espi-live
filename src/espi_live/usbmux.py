"""Reach a TCP port inside an iPhone app over the USB cable, through usbmuxd.

Apple's USB multiplexing daemon (usbmuxd) carries TCP connections between a
computer and an iOS device over USB. On Windows it is the "Apple Mobile
Device Service" that iTunes (from apple.com) installs, listening on
127.0.0.1:27015; on macOS it is part of the system, at /var/run/usbmuxd.
Linux uses the open-source usbmuxd at the same path. The environment
variable USBMUXD_SOCKET_ADDRESS overrides the address ("host:port", or
"UNIX:/path" / a plain path for a Unix socket), as libusbmuxd allows.

The protocol (verified against libusbmuxd, pymobiledevice3 and
appium-ios-device): every message is a 16-byte little-endian header (total
length, version 1, type 8 = plist, tag) followed by an XML plist. A
"Connect" that succeeds turns the same socket into a raw byte pipe to the
device's port; ListDevices needs its own socket. The port number is sent
byte-swapped (network order read as little-endian). Pure standard library,
no GUI imports.
"""

import os
import plistlib
import socket
import struct
import sys
from dataclasses import dataclass

WINDOWS_ADDRESS = ("127.0.0.1", 27015)
UNIX_PATH = "/var/run/usbmuxd"
HEADER = struct.Struct("<IIII")  # length (with header), version, type, tag
PLIST_VERSION = 1
PLIST_TYPE = 8
MAX_REPLY = 4 * 1024 * 1024  # a device list is a few kB; refuse anything absurd
CLIENT_NAME = "espi-live"

# "Result" Number codes from libusbmuxd's usbmuxd-proto.h.
RESULT_TEXT = {
    1: "the daemon rejected the command",
    2: "the device is no longer connected",
    3: "the port refused the connection (is the app open on the phone?)",
    4: "no such service",
    6: "the daemon does not speak this protocol version",
}


class UsbmuxError(RuntimeError):
    """The daemon is unreachable, or it refused a request."""


@dataclass(frozen=True)
class MuxDevice:
    """One iOS device the daemon reports as attached over USB.

    device_id is the daemon's handle (it changes when the cable is
    re-plugged); serial is the device's UDID, which does not.
    """

    device_id: int
    serial: str
    product_id: int | None = None
    location_id: int | None = None


def default_address():
    """Choose where the daemon listens on this computer.

    Args: None.
    Returns: A (host, port) tuple for TCP, or a str path for a Unix socket:
        USBMUXD_SOCKET_ADDRESS when set, else 127.0.0.1:27015 on Windows and
        /var/run/usbmuxd elsewhere.
    """
    override = os.environ.get("USBMUXD_SOCKET_ADDRESS", "").strip()
    if override:
        if override.upper().startswith("UNIX:"):
            return override[5:]
        host, sep, port = override.rpartition(":")
        if sep and port.isdigit():
            return host.strip("[]"), int(port)
        return override
    if sys.platform.startswith("win"):
        return WINDOWS_ADDRESS
    return UNIX_PATH


def open_socket(timeout=2.0, address=None):
    """Connect to the daemon.

    Args: timeout is the connect and read timeout in seconds; address
        overrides default_address().
    Returns: A connected socket. Raises UsbmuxError when the daemon cannot
        be reached, naming what to install.
    """
    address = default_address() if address is None else address
    if isinstance(address, tuple):
        family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
    else:
        family = socket.AF_UNIX
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(address)
    except OSError as exc:
        sock.close()
        hint = (" Install iTunes from apple.com (it adds the Apple Mobile Device "
                "Service), then plug the iPhone in again."
                if isinstance(address, tuple) else "")
        raise UsbmuxError(f"The USB device service is not reachable at "
                          f"{address}: {exc}.{hint}") from exc
    return sock


def recv_exact(sock, count):
    """Read exactly count bytes.

    Args: sock is a connected socket; count is the number of bytes.
    Returns: The bytes. Raises ConnectionError if the peer closes first.
    """
    chunks = bytearray()
    while len(chunks) < count:
        chunk = sock.recv(count - len(chunks))
        if not chunk:
            raise ConnectionError("the USB device service closed the connection")
        chunks += chunk
    return bytes(chunks)


def request_dict(message_type, **fields):
    """Build a request plist with the keys every client sends.

    Args: message_type is the request name ("ListDevices", "Connect");
        fields are extra keys.
    Returns: A dict ready for plistlib.
    """
    return {"MessageType": message_type, "ClientVersionString": CLIENT_NAME,
            "ProgName": CLIENT_NAME, "kLibUSBMuxVersion": 3, **fields}


def send_plist(sock, payload, tag=1):
    """Send one plist message.

    Args: sock is a daemon socket; payload is a dict; tag is echoed back.
    Returns: None.
    """
    body = plistlib.dumps(payload)
    sock.sendall(HEADER.pack(HEADER.size + len(body), PLIST_VERSION, PLIST_TYPE, tag)
                 + body)


def read_plist(sock):
    """Read one plist message.

    Args: sock is a daemon socket.
    Returns: The decoded dict. Raises UsbmuxError for a malformed reply.
    """
    length, _version, kind, _tag = HEADER.unpack(recv_exact(sock, HEADER.size))
    if kind != PLIST_TYPE or not HEADER.size <= length <= MAX_REPLY:
        raise UsbmuxError(f"unexpected reply from the USB device service "
                          f"(type {kind}, {length} bytes)")
    try:
        reply = plistlib.loads(recv_exact(sock, length - HEADER.size))
    except plistlib.InvalidFileException as exc:
        raise UsbmuxError(f"unreadable reply from the USB device service: {exc}") from exc
    if not isinstance(reply, dict):
        raise UsbmuxError("unexpected reply from the USB device service")
    return reply


def parse_devices(reply):
    """Pick the USB-attached devices out of a ListDevices reply.

    A phone that also syncs over Wi-Fi is listed a second time with
    ConnectionType "Network"; only the USB entry is kept.

    Args: reply is the decoded ListDevices dict.
    Returns: A list of MuxDevice in the daemon's order.
    """
    devices = []
    for entry in reply.get("DeviceList", []):
        props = entry.get("Properties", {})
        if props.get("ConnectionType", "USB") != "USB":
            continue
        devices.append(MuxDevice(
            device_id=int(entry.get("DeviceID", props.get("DeviceID", 0))),
            serial=str(props.get("SerialNumber", "")),
            product_id=props.get("ProductID"), location_id=props.get("LocationID")))
    return devices


def list_devices(timeout=2.0, address=None):
    """List the iOS devices attached over USB.

    Args: timeout is in seconds; address overrides default_address().
    Returns: A list of MuxDevice. Raises UsbmuxError when the daemon is
        unreachable or answers wrongly.
    """
    sock = open_socket(timeout, address)
    try:
        send_plist(sock, request_dict("ListDevices"))
        try:
            return parse_devices(read_plist(sock))
        except (OSError, ConnectionError) as exc:
            raise UsbmuxError(f"The USB device service did not list devices: {exc}") from exc
    finally:
        sock.close()


def swap_port(port):
    """Encode a port number the way the daemon expects it.

    Args: port is the TCP port (0-65535).
    Returns: The port with its two bytes swapped (htons on every host).
    """
    return ((port & 0xFF) << 8) | ((port >> 8) & 0xFF)


def connect(device_id, port, timeout=2.0, address=None):
    """Open a TCP connection to a port on a USB-attached device.

    Args: device_id is a MuxDevice.device_id; port is the TCP port the app
        listens on; timeout is in seconds; address overrides
        default_address().
    Returns: A socket that is now a raw pipe to the device's port (its
        timeout is cleared). Raises UsbmuxError when the daemon refuses,
        with a readable reason.
    """
    sock = open_socket(timeout, address)
    try:
        send_plist(sock, request_dict("Connect", DeviceID=device_id,
                                      PortNumber=swap_port(port)))
        try:
            reply = read_plist(sock)
        except (OSError, ConnectionError) as exc:
            raise UsbmuxError(f"The USB device service did not answer: {exc}") from exc
        number = reply.get("Number")
        if reply.get("MessageType") != "Result" or number != 0:
            reason = RESULT_TEXT.get(number, f"error {number}")
            raise UsbmuxError(f"Could not reach port {port} on the iPhone: {reason}")
    except BaseException:
        sock.close()
        raise
    sock.settimeout(None)
    return sock
