"""An iPhone camera over the USB cable, through the ESPI Camera iOS app.

The ESPI Camera app (ios/ESPICamera) runs on the phone, locks the camera's
exposure, ISO, focus and white balance, and serves each frame's 8-bit luma
plane uncompressed on TCP port PORT of the phone's loopback interface. The
computer reaches that port over USB through usbmuxd (see usbmux.py), so no
Wi-Fi or network setup is involved.

Link protocol, version 1 (all integers little-endian; docs/iphone/README.md
has the full description):

- Every message is a 16-byte header (magic b"ESPC", version, kind, payload
  length, reserved) followed by the payload.
- The phone sends HELLO (JSON: device, app, the back camera in use ("lens")
  and every back camera it has ("lenses": id and name), formats, limits,
  settings) as soon as the computer connects, then FRAME messages (a 48-byte
  FRAME_META plus width × height Mono8 bytes) and a STATUS (JSON) every
  second.
- In RAW mode a FRAME carries one colour plane of the sensor's Bayer data
  instead: pixel format MONO16, and after FRAME_META a 16-byte RAW_META (black
  level, white level, colour, pattern) and then width × height little-endian
  16-bit values the phone's image pipeline has not touched. decode_frame turns
  them into a deep frame (float32, 8-bit-equivalent grey levels: black level
  0, white level 255, fractions kept), which the engine carries through its
  difference and averaging steps before rounding to 8 bits for display.
- The computer sends COMMAND (JSON with any of exposure_us, iso, fps,
  lens_position, width, height, crop_width, crop_height, max_send_fps,
  refocus, lens, raw, raw_channel); the phone applies it and answers with a
  STATUS. A "lens" command switches the camera and is answered with a new
  HELLO, because the formats and limits belong to the camera.

IPhoneSession has the CameraSession interface (open, grab, set_float,
limits, read_values, read_stream_counters, describe, close, identity,
info), so the engine treats an iPhone like any other camera: its exposure
slider sets the phone's exposure in µs, its gain slider the phone's ISO (0 dB
is the lowest ISO) and its frame-rate slider the phone's frame rate. The
phone-only settings (focus, frame size, crop, send-rate cap) go through
send_command(), and take_settings_update() hands back what changed on the
phone, so every control works from the computer and both screens agree. A
reader thread owned by the session receives frames, so grab() honours its
timeout; a lost link is re-opened from grab(). A phone with several back
cameras is listed once per camera (list_iphones), each a CameraInfo whose
lens the session asks the app for. No GUI imports.
"""

import json
import math
import struct
import threading
import time
from dataclasses import dataclass, replace

import numpy as np

from . import usbmux
from .camera import CAMERA_RANGES, CameraInfo
from .processing import FULL_SCALE

PORT = 47820
MAGIC = b"ESPC"
VERSION = 1
HEADER = struct.Struct("<4sHHII")  # magic, version, kind, payload length, reserved
# width, height, stride, pixel format, sequence, timestamp (ns), exposure (µs),
# ISO, lens position, reserved
FRAME_META = struct.Struct("<IIIIQqfffI")
KIND_HELLO, KIND_STATUS, KIND_FRAME, KIND_COMMAND, KIND_ERROR = 1, 2, 3, 4, 5
PIXEL_MONO8 = 1
PIXEL_MONO16 = 2   # RAW mode: one Bayer colour plane, 16-bit values, RAW_META follows
# black level, white level (sensor counts), colour (RAW_CHANNELS index), pattern
RAW_META = struct.Struct("<ffII")
RAW_CHANNELS = ("red", "green", "blue")
RAW_PATTERNS = ("RGGB", "GRBG", "GBRG", "BGGR")
MAX_JSON = 1 << 20
MAX_FRAME = 64 << 20  # 48 MP of Mono8 with room to spare

HELLO_TIMEOUT_S = 3.0      # the app greets at once; longer means it is not there
PROBE_TIMEOUT_S = 1.5      # listing asks a phone for its cameras; longer means skip it
LENS_TIMEOUT_S = 5.0       # the app re-introduces itself after switching camera
MAIN_LENS = "wide"         # the camera a phone opens with, listed first
STALL_S = 5.0              # the app sends a status every second; silence = lost link
RECONNECT_INTERVAL_S = 2.0
QUIET_S = 1.5              # ignore the phone's echo of a setting the viewer just sent
CROP_SIZES = (0, 2048, 1024, 512)       # 0 = full frame; square centre crops
SEND_CAPS = (0.0, 30.0, 15.0, 10.0, 5.0)  # 0 = every frame USB can carry
# Phone setting (COMMAND key) -> decimals kept when reporting it to the viewer.
RAW_SETTINGS = ("raw", "raw_channel")  # COMMAND keys that are not numbers
SETTING_DECIMALS = {"exposure_us": 1, "gain_db": 2, "fps": 2, "lens_position": 3,
                    "width": 0, "height": 0, "crop_width": 0, "crop_height": 0,
                    "max_send_fps": 2}
THERMAL_WARNINGS = ("serious", "critical")

OPEN_HINT = ("Open the ESPI Camera app on the iPhone and keep it on screen, "
             "unlock the phone, and tap Trust if it asks about this computer")


class LinkError(RuntimeError):
    """The phone sent something this version cannot read."""


# What each phone's app last said its back cameras are, by UDID. Listing fills
# it by asking a phone that has not been heard from; every HELLO refreshes it.
# An empty tuple means an app that predates camera choice.
_lens_cache = {}


@dataclass(frozen=True)
class Lens:
    """One back camera of a phone: its id and the name shown in menus."""

    id: str
    name: str


@dataclass(frozen=True)
class FrameMeta:
    """What the phone reports about one frame."""

    width: int
    height: int
    sequence: int
    timestamp_ns: int
    exposure_us: float
    iso: float
    lens_position: float
    raw: bool = False        # a RAW plane (deep frame), not the video luma
    channel: int = 0         # RAW_CHANNELS index of the plane
    black_level: float = 0.0
    white_level: float = 0.0


@dataclass(frozen=True)
class IPhoneIdentity:
    """The phone and camera behind an iPhone source, from its HELLO message.

    Same surface as CameraIdentity: short_name, description, warnings, plus
    supported and interlaced like a capture device's identity.
    """

    serial: str = ""
    device_model: str = ""
    system_version: str = ""
    app_version: str = ""
    camera: str = ""
    width: int = 0
    height: int = 0
    formats: tuple = ()  # (width, height, max fps) the phone can send
    supported: bool = True
    interlaced: bool = False
    raw: bool = False        # the phone sends Bayer RAW planes, not video luma
    raw_channel: str = "red"

    def short_name(self):
        """Name the source for the badge and window title.

        Args: None.
        Returns: "iPhone" plus the model identifier when known.
        """
        return f"iPhone {self.device_model}".strip() if self.device_model else "iPhone"

    def description(self):
        """Describe the phone in a few lines.

        Args: None.
        Returns: A list of text lines; empty before the phone answered.
        """
        if not self.device_model:
            return []
        lines = [f"{self.short_name()} · iOS {self.system_version} · "
                 f"ESPI Camera {self.app_version}"]
        if self.raw:  # the plane is half the sensor's size; the frame reports it
            lines.append(f"{self.camera or 'Camera'}: Bayer RAW {self.raw_channel} "
                         "plane, 16-bit, no image processing, uncompressed over USB")
        elif self.width and self.height:
            lines.append(f"{self.camera or 'Camera'}: {self.width} × {self.height} "
                         "Mono8 (luma), uncompressed over USB")
        if self.serial:
            lines.append(f"Device: {self.serial}")
        return lines

    def warnings(self):
        """List what makes this source differ from a scientific camera.

        Args: None.
        Returns: A list of notes.
        """
        if not self.device_model:
            return []
        if self.raw:
            return ["Phone camera, RAW mode: rolling shutter and one colour plane "
                    "of the colour filter; the phone's image pipeline is bypassed, "
                    "so the noise is the sensor's own. Check results against the "
                    "FLIR camera"]
        return ["Phone camera: rolling shutter, colour filter and the phone's "
                "own noise reduction; check results against the FLIR camera"]


def encode_message(kind, payload=b""):
    """Frame one message.

    Args: kind is a KIND_* number; payload is bytes.
    Returns: The header followed by the payload.
    """
    return HEADER.pack(MAGIC, VERSION, kind, len(payload), 0) + bytes(payload)


def encode_json(kind, data):
    """Frame one JSON message.

    Args: kind is a KIND_* number; data is a JSON-serialisable value.
    Returns: The message bytes.
    """
    return encode_message(kind, json.dumps(data, separators=(",", ":")).encode())


def encode_frame(image, sequence=0, timestamp_ns=0, exposure_us=0.0, iso=0.0,
                 lens_position=0.0):
    """Frame one Mono8 image the way the phone sends it (used by tests and tools).

    Args: image is a 2-D uint8 array; the other values fill FRAME_META.
    Returns: The FRAME message bytes.
    """
    height, width = image.shape
    meta = FRAME_META.pack(width, height, width, PIXEL_MONO8, sequence, timestamp_ns,
                           exposure_us, iso, lens_position, 0)
    return encode_message(KIND_FRAME, meta + np.ascontiguousarray(image).tobytes())


def encode_raw_frame(plane, black_level, white_level, channel=0, pattern=0, sequence=0,
                     timestamp_ns=0, exposure_us=0.0, iso=0.0, lens_position=0.0):
    """Frame one RAW colour plane the way the phone sends it (tests and tools).

    Args: plane is a 2-D uint16 array of sensor counts; black_level and
        white_level are the counts for black and full scale; channel indexes
        RAW_CHANNELS and pattern RAW_PATTERNS; the rest fill FRAME_META.
    Returns: The FRAME message bytes.
    """
    height, width = plane.shape
    meta = FRAME_META.pack(width, height, width, PIXEL_MONO16, sequence, timestamp_ns,
                           exposure_us, iso, lens_position, 0)
    raw = RAW_META.pack(black_level, white_level, channel, pattern)
    body = np.ascontiguousarray(plane, dtype="<u2").tobytes()
    return encode_message(KIND_FRAME, meta + raw + body)


def recv_into_exact(sock, buffer):
    """Fill a buffer from a socket.

    Args: sock is a connected socket; buffer is a writable bytearray.
    Returns: None. Raises ConnectionError if the phone closes the link first.
    """
    view = memoryview(buffer)
    while view:
        count = sock.recv_into(view)
        if not count:
            raise ConnectionError("the iPhone closed the connection")
        view = view[count:]


def read_message(sock):
    """Read one message.

    Args: sock is the link socket.
    Returns: (kind, payload) with payload a bytearray. Raises LinkError for
        a message this version cannot read, ConnectionError or OSError when
        the link fails.
    """
    header = bytearray(HEADER.size)
    recv_into_exact(sock, header)
    magic, version, kind, length, _reserved = HEADER.unpack(header)
    if magic != MAGIC:
        raise LinkError("the phone sent data that is not from the ESPI Camera app")
    if version != VERSION:
        raise LinkError(f"the ESPI Camera app speaks link version {version}; this "
                        f"viewer speaks {VERSION}; update the app or the viewer")
    limit = MAX_FRAME if kind == KIND_FRAME else MAX_JSON
    if length > limit:
        raise LinkError(f"message of {length} bytes is too large")
    payload = bytearray(length)
    recv_into_exact(sock, payload)
    return kind, payload


def decode_frame(payload):
    """Unpack a FRAME payload.

    Args: payload is the bytearray from read_message().
    Returns: (FrameMeta, image). For Mono8, image is a writable 2-D uint8 array
        that shares the payload's memory. For a RAW plane (Mono16) it is a
        float32 array in 8-bit-equivalent grey levels: (count - black level)
        scaled so the white level is 255, negative below black. Raises
        LinkError when the payload is malformed.
    """
    if len(payload) < FRAME_META.size:
        raise LinkError("frame header is truncated")
    (width, height, stride, pixel_format, sequence, timestamp_ns, exposure_us, iso,
     lens_position, _reserved) = FRAME_META.unpack_from(payload)
    if pixel_format == PIXEL_MONO16:
        return decode_raw_frame(payload, width, height, stride, sequence, timestamp_ns,
                                exposure_us, iso, lens_position)
    if pixel_format != PIXEL_MONO8:
        raise LinkError(f"pixel format {pixel_format} is not Mono8")
    if stride < width or len(payload) != FRAME_META.size + stride * height:
        raise LinkError(f"a {width} × {height} frame arrived with "
                        f"{len(payload) - FRAME_META.size} bytes")
    image = np.frombuffer(payload, np.uint8, stride * height, FRAME_META.size)
    image = image.reshape(height, stride)
    if stride != width:  # drop row padding; the engine expects contiguous rows
        image = np.ascontiguousarray(image[:, :width])
    meta = FrameMeta(width, height, sequence, timestamp_ns, exposure_us, iso,
                     lens_position)
    return meta, image


def decode_raw_frame(payload, width, height, stride, sequence, timestamp_ns,
                     exposure_us, iso, lens_position):
    """Unpack a RAW colour plane into a deep frame.

    Args: payload is the FRAME payload; the other arguments are FRAME_META's
        fields (stride counts 16-bit values).
    Returns: (FrameMeta, image) with image a float32 array of 8-bit-equivalent
        grey levels. Raises LinkError when the payload is malformed.
    """
    start = FRAME_META.size + RAW_META.size
    if len(payload) < start:
        raise LinkError("RAW frame header is truncated")
    black, white, channel, _pattern = RAW_META.unpack_from(payload, FRAME_META.size)
    if stride < width or len(payload) != start + 2 * stride * height:
        raise LinkError(f"a {width} × {height} RAW frame arrived with "
                        f"{len(payload) - start} bytes")
    span = float(white) - float(black)
    if not span > 0 or channel >= len(RAW_CHANNELS):
        raise LinkError(f"RAW frame has black level {black:g}, white level "
                        f"{white:g}, colour {channel}")
    counts = np.frombuffer(payload, "<u2", stride * height, start).reshape(height, stride)
    image = np.subtract(counts[:, :width], np.float32(black), dtype=np.float32)
    image *= np.float32(FULL_SCALE / span)
    meta = FrameMeta(width, height, sequence, timestamp_ns, exposure_us, iso,
                     lens_position, True, int(channel), float(black), float(white))
    return meta, image


def decode_json(payload):
    """Decode a JSON payload.

    Args: payload is bytes.
    Returns: The decoded dict. Raises LinkError when it is not a JSON object.
    """
    try:
        data = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise LinkError(f"unreadable message from the phone: {exc}") from exc
    if not isinstance(data, dict):
        raise LinkError("unexpected message from the phone")
    return data


def gain_to_iso(gain_db, min_iso):
    """Turn the gain slider's dB into the phone's ISO.

    Args: gain_db is the requested gain; min_iso is the phone's lowest ISO.
    Returns: The ISO: min_iso at 0 dB, ten times it at 20 dB.
    """
    return min_iso * 10.0 ** (gain_db / 20.0)


def iso_to_gain(iso, min_iso):
    """Turn the phone's ISO into the gain slider's dB.

    Args: iso is the ISO; min_iso is the phone's lowest ISO.
    Returns: The gain in dB, 0 at min_iso; None when either is not positive.
    """
    if iso <= 0 or min_iso <= 0:
        return None
    return 20.0 * math.log10(iso / min_iso)


def clip_range(pair, name):
    """Clip a phone limit to the range the viewer's slider shows.

    Args: pair is [low, high] from the phone; name is a CAMERA_RANGES key.
    Returns: A (low, high) tuple inside CAMERA_RANGES[name], or None when
        the pair is missing or empty after clipping.
    """
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None
    floor, ceiling = CAMERA_RANGES[name]
    low, high = max(float(pair[0]), floor), min(float(pair[1]), ceiling)
    return (low, high) if low < high else None


def parse_lenses(hello):
    """Read the back cameras a HELLO offers.

    Args: hello is the decoded HELLO dict.
    Returns: A tuple of Lens in the app's order (narrowest zoom first); empty
        when the app predates camera choice.
    """
    entries = hello.get("lenses")
    if not isinstance(entries, list):
        return ()
    return tuple(Lens(str(entry["id"]), str(entry.get("name") or entry["id"]))
                 for entry in entries if isinstance(entry, dict) and entry.get("id"))


def probe_lenses(device):
    """Ask a phone's app which back cameras it has, then hang up.

    The app serves one link at a time and a new connection replaces the old
    one, so this is only for a phone nobody is using.

    Args: device is a usbmux.MuxDevice.
    Returns: A tuple of Lens (see parse_lenses), or None when the app did not
        answer (not open, phone locked, or an unreadable greeting).
    """
    sock = None
    try:
        sock = usbmux.connect(device.device_id, PORT, timeout=PROBE_TIMEOUT_S)
        sock.settimeout(PROBE_TIMEOUT_S)
        kind, payload = read_message(sock)
        return parse_lenses(decode_json(payload)) if kind == KIND_HELLO else None
    except (OSError, usbmux.UsbmuxError, LinkError):
        return None
    finally:
        if sock is not None:
            shut_down(sock)


def lens_infos(udid, lenses):
    """Make the camera-list entries for one phone.

    The main (wide) camera keeps the phone's original serial, so a camera
    saved before phones listed every lens keeps its settings.

    Args: udid is the phone's UDID; lenses is a tuple of Lens (empty when the
        app has not said).
    Returns: A list of CameraInfo, main camera first. One entry named just
        "iPhone (USB)" for a phone with no more than one camera; its lens is
        empty until the app has said which it is.
    """
    base = f"iphone-{udid[-8:].lower()}"
    if len(lenses) < 2:
        return [CameraInfo(base, "iPhone (USB)", kind="iphone", path=udid,
                           lens=lenses[0].id if lenses else "")]
    ordered = sorted(lenses, key=lambda lens: lens.id != MAIN_LENS)  # stable
    return [CameraInfo(base if lens.id == MAIN_LENS
                       else f"{base}-{lens.id.replace('_', '-')}",
                       f"iPhone {lens.name} (USB)", kind="iphone", path=udid,
                       lens=lens.id)
            for lens in ordered]


def list_iphones(log):
    """List the iPhones (and iPads) plugged in over USB, without opening any.

    A phone whose app has not yet said which back cameras it has is asked
    once (see probe_lenses); one that does not answer is listed as a single
    entry, so it still appears and opens its default camera.

    Args: log has a write(message) method.
    Returns: A list of CameraInfo of kind "iphone" whose path is the
        device's UDID, one per back camera of each phone; empty when the USB
        device service is unavailable (logged once per call).
    """
    try:
        devices = usbmux.list_devices(timeout=1.0)
    except usbmux.UsbmuxError as exc:
        log.write(f"iPhones are not listed: {exc}")
        return []
    infos = []
    for device in devices:
        lenses = _lens_cache.get(device.serial)
        if lenses is None:
            lenses = probe_lenses(device)
            if lenses is not None:
                _lens_cache[device.serial] = lenses
        infos += lens_infos(device.serial, lenses or ())
    return infos


def main_lenses(infos):
    """Keep one camera per phone: the first listed, which is the main one.

    Args: infos is a list of CameraInfo from list_iphones().
    Returns: The list with every phone's other cameras removed. Two cameras
        of one phone cannot stream at once, so only one is ever opened
        without being asked.
    """
    seen, kept = set(), []
    for info in infos:
        if info.path not in seen:
            seen.add(info.path)
            kept.append(info)
    return kept


def usb_connector(info, port=PORT):
    """Open the link to the ESPI Camera app on a USB-attached iPhone.

    Args: info is the phone's CameraInfo (path is its UDID); port is the
        app's port.
    Returns: A connected socket. Raises RuntimeError when the phone is not
        attached, usbmux.UsbmuxError when it refuses.
    """
    for device in usbmux.list_devices():
        if device.serial == info.path:
            return usbmux.connect(device.device_id, port)
    raise RuntimeError("the iPhone is not plugged in over USB")


class IPhoneSession:
    """Own the link to the ESPI Camera app and the thread that reads it."""

    interlaced = False

    def __init__(self, log, info, exposure_us=8000.0, gain_db=0.0, fps=30.0,
                 connector=usb_connector):
        """Prepare to open one iPhone.

        Args: log has write and warning methods; info is the phone's
            CameraInfo; exposure_us, gain_db and fps are the starting
            settings sent once the app answers; connector(info) returns a
            connected socket (tests pass a fake).
        Returns: None.
        """
        self.log = log
        self.info = info
        self.connector = connector
        self.requested = {"exposure_us": float(exposure_us), "gain_db": float(gain_db),
                          "fps": float(fps)}
        self.raw_request = {}  # "raw" and "raw_channel" the viewer chose, resent on reconnect
        self.identity = IPhoneIdentity(serial=info.path)
        self.hello = {}
        self.status = {}
        self._cond = threading.Condition()
        self._send_lock = threading.Lock()
        self._sock = None
        self._reader = None
        self._frame = None
        self._meta = None
        self._frame_seq = 0
        self._taken_seq = 0
        self._error = None
        self._closing = False
        self._last_message = 0.0
        self._last_attempt = 0.0
        self._requested_at = {}  # COMMAND key -> when the viewer last sent it
        self._reported = {}      # settings last handed out by take_settings_update
        self._thermal = "nominal"
        self._frames = 0
        self._bytes = 0
        self._byte_rate = 0.0
        self._since = time.monotonic()

    # ----- lifecycle ----------------------------------------------------

    def open(self):
        """Connect, read the app's HELLO and send the starting settings.

        Args: None.
        Returns: None; self.identity describes the phone. Raises
            RuntimeError naming the phone and what to do; nothing stays open.
        """
        try:
            self._connect()
        except Exception as exc:
            self.close()
            raise RuntimeError(f"Could not open {self.info.label}: {exc}. "
                               f"{OPEN_HINT}.") from exc
        self.log.write(f"iPhone link open: {self._summary()}")

    def _connect(self):
        """Open the socket, greet, start the reader thread.

        Args: None.
        Returns: None. Raises on any failure, leaving the socket closed.
        """
        sock = self.connector(self.info)
        try:
            sock.settimeout(HELLO_TIMEOUT_S)
            kind, payload = read_message(sock)
            if kind != KIND_HELLO:
                raise LinkError("the app did not introduce itself")
            hello = decode_json(payload)
            if hello.get("protocol_version") != VERSION:
                raise LinkError(f"the app speaks link version "
                                f"{hello.get('protocol_version')}; update it")
            sock.settimeout(None)
        except BaseException:
            sock.close()
            raise
        with self._cond:
            self._sock, self._error, self._closing = sock, None, False
            self._apply_hello(hello)
            self._last_message = time.monotonic()
            self._frames, self._bytes, self._since = 0, 0, time.monotonic()
        now = time.monotonic()
        for key in ("exposure_us", "gain_db", "fps"):
            self._requested_at[key] = now
        self._reported = {}  # a new link reports every setting again
        self._send(self._starting_command())
        self._reader = threading.Thread(target=self._read_loop, args=(sock,),
                                        name="espi-iphone", daemon=True)
        self._reader.start()
        self._await_lens()

    def _await_lens(self):
        """Wait for the app to switch to this session's back camera.

        The app answers a lens command with a new HELLO describing that
        camera, so identity, formats and limits are the camera's own before
        the session opens.

        Args: None.
        Returns: None. Closes the link and raises LinkError when the app does
            not report the camera in LENS_TIMEOUT_S, so frames from another
            camera are never shown under this one's name.
        """
        wanted = self.info.lens
        if not wanted:
            return
        deadline = time.monotonic() + LENS_TIMEOUT_S
        with self._cond:
            while self.hello.get("lens") != wanted and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            switched = self.hello.get("lens") == wanted
        if not switched:
            self.close()
            raise LinkError(f"the iPhone did not switch to its {wanted} camera")

    def _starting_command(self):
        """Build the command that sends every requested setting.

        Args: None.
        Returns: A command dict.
        """
        command = {"exposure_us": self.requested["exposure_us"],
                   "fps": self.requested["fps"]}
        if self.info.lens:
            command["lens"] = self.info.lens
        iso = self._iso_for(self.requested["gain_db"])
        if iso is not None:
            command["iso"] = iso
        command.update(self.raw_request)
        return command

    def close(self):
        """Close the link and stop the reader; safe to call more than once.

        Args: None.
        Returns: None.
        """
        with self._cond:
            self._closing = True
            sock, self._sock = self._sock, None
            self._cond.notify_all()
        if sock is not None:
            shut_down(sock)
        reader, self._reader = self._reader, None
        if reader is not None and reader is not threading.current_thread():
            reader.join(1.0)

    # ----- reader thread --------------------------------------------------

    def _read_loop(self, sock):
        """Receive messages until the link closes (runs on the reader thread).

        Args: sock is the link socket this thread owns for reading.
        Returns: None; a failure is stored for grab() to report.
        """
        try:
            while True:
                kind, payload = read_message(sock)
                self._handle(kind, payload)
        except Exception as exc:
            with self._cond:
                if self._sock is sock and not self._closing:
                    self._error = exc
                self._cond.notify_all()

    def _handle(self, kind, payload):
        """Act on one message from the phone.

        Args: kind is the KIND_* number; payload the bytearray.
        Returns: None. Raises LinkError for malformed content.
        """
        now = time.monotonic()
        if kind == KIND_FRAME:
            meta, image = decode_frame(payload)
            with self._cond:
                self._frame, self._meta = image, meta
                self._frame_seq += 1
                self._frames += 1
                self._bytes += len(payload)
                self._last_message = now
                self._cond.notify_all()
            return
        with self._cond:
            self._last_message = now
        if kind == KIND_STATUS:
            self._apply_status(decode_json(payload))
        elif kind == KIND_HELLO:
            with self._cond:
                self._apply_hello(decode_json(payload))
                self._cond.notify_all()
        elif kind == KIND_ERROR:
            message = decode_json(payload).get("message", "unknown error")
            self.log.warning(f"iPhone reports: {message}")
        # other kinds come from a newer app and are ignored

    def _apply_hello(self, hello):
        """Keep a HELLO message and update the identity (lock held).

        Args: hello is the decoded dict.
        Returns: None.
        """
        self.hello = hello
        _lens_cache[self.info.path] = parse_lenses(hello)
        settings = hello.get("settings", {})
        formats = tuple((int(entry.get("width", 0)), int(entry.get("height", 0)),
                         float(entry.get("max_fps", 0)))
                        for entry in hello.get("formats", []) if isinstance(entry, dict))
        self.identity = IPhoneIdentity(
            serial=self.info.path, device_model=str(hello.get("device_model", "")),
            system_version=str(hello.get("system_version", "")),
            app_version=str(hello.get("app_version", "")),
            camera=str(hello.get("camera", "")),
            width=int(settings.get("crop_width") or settings.get("width") or 0),
            height=int(settings.get("crop_height") or settings.get("height") or 0),
            formats=formats, raw=bool(settings.get("raw", False)),
            raw_channel=str(settings.get("raw_channel", "red")))

    def _apply_status(self, status):
        """Keep a STATUS message and log what changed for the worse.

        Args: status is the decoded dict.
        Returns: None.
        """
        thermal = str(status.get("thermal", "nominal"))
        with self._cond:
            previous = self.status
            self.status = status
            changed_thermal = thermal != self._thermal
            self._thermal = thermal
            phone = status.get("settings") or {}
            raw, channel = bool(phone.get("raw", False)), str(phone.get("raw_channel", "red"))
            if (raw, channel) != (self.identity.raw, self.identity.raw_channel):
                self.identity = replace(self.identity, raw=raw, raw_channel=channel)
        if changed_thermal and thermal in THERMAL_WARNINGS:
            self.log.warning(f"The iPhone is getting hot ({thermal}); it may lower its "
                             "frame rate. Give it air or lower the frame rate.")
        interrupted = status.get("interrupted")
        if interrupted and interrupted != previous.get("interrupted"):
            self.log.warning(f"The iPhone camera is interrupted: {interrupted}")
        error = status.get("error")
        if error and error != previous.get("error"):
            self.log.warning(f"iPhone reports: {error}")

    # ----- engine interface (engine worker thread) ------------------------

    def grab(self, timeout_ms):
        """Wait for the newest frame.

        Args: timeout_ms is the longest wait for a frame.
        Returns: (frame, "ok") with a 2-D uint8 array, or (exception,
            "error") on a timeout or a lost link; a lost link is re-opened at
            most every RECONNECT_INTERVAL_S.
        """
        deadline = time.monotonic() + timeout_ms / 1000.0
        with self._cond:
            while (self._frame_seq == self._taken_seq and self._error is None
                   and self._sock is not None):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            if self._frame_seq != self._taken_seq:
                self._taken_seq = self._frame_seq
                return self._frame, "ok"
            error = self._error
            connected = self._sock is not None
            silent = time.monotonic() - self._last_message
        if error is None and connected and silent < STALL_S:
            return TimeoutError("no frame from the iPhone yet"), "error"
        if error is None:
            error = ConnectionError("no data from the iPhone for "
                                    f"{silent:.0f} s" if connected else
                                    "the iPhone link is closed")
        self._recover(error, timeout_ms)
        return error, "error"

    def _recover(self, error, timeout_ms):
        """Re-open a lost link, at most every RECONNECT_INTERVAL_S.

        Args: error is what ended the link; timeout_ms is slept instead when
            it is not yet time to retry, so a dead link cannot spin the caller.
        Returns: None.
        """
        now = time.monotonic()
        if now - self._last_attempt < RECONNECT_INTERVAL_S:
            time.sleep(timeout_ms / 1000.0)
            return
        self._last_attempt = now
        self.log.warning(f"iPhone link lost ({error}); reconnecting")
        self.close()
        try:
            self._connect()
        except Exception as exc:
            self.log.write(f"iPhone not reachable yet: {exc}. {OPEN_HINT}.")
            return
        self.log.write(f"iPhone link open again: {self._summary()}")

    def set_float(self, name, value):
        """Send a camera-setting change to the phone.

        Args: name is "ExposureTime" (µs), "Gain" (dB above the lowest ISO)
            or "AcquisitionFrameRate" (fps); value is the new value. Other
            names are ignored.
        Returns: None; a send failure is logged (grab() notices a lost link).
        """
        value = float(value)
        key = {"ExposureTime": "exposure_us", "Gain": "gain_db",
               "AcquisitionFrameRate": "fps"}.get(name)
        if key is None:
            return
        self.requested[key] = value
        self._requested_at[key] = time.monotonic()
        if key == "gain_db":
            iso = self._iso_for(value)
            if iso is None:
                return
            command = {"iso": iso}
        else:
            command = {key: value}
        try:
            self._send(command)
        except OSError as exc:
            self.log.write(f"Could not send {name} to the iPhone: {exc}")

    def send_command(self, **fields):
        """Send a phone-only setting to the app (focus, size, crop, send cap).

        Args: fields are COMMAND keys and values, for example
            lens_position=0.4, width=1920 and height=1080, crop_width and
            crop_height, max_send_fps, refocus=True, raw=True (Bayer RAW
            planes instead of video luma) or raw_channel="green".
        Returns: None. Raises OSError when the link fails.
        """
        now = time.monotonic()
        for key in fields:
            if key != "refocus":  # a refocus should report the new lens position
                self._requested_at[key] = now
        self.raw_request.update({key: fields[key] for key in RAW_SETTINGS if key in fields})
        self._send(fields)

    def take_settings_update(self):
        """Hand back the phone settings that changed since the last call.

        A setting the viewer sent within QUIET_S is left out, so the phone's
        echo of an older value cannot pull a control back while it is being
        dragged; once quiet, the phone's value (clamped, or changed on the
        phone) is reported.

        Args: None.
        Returns: A dict of changed settings in the viewer's units
            (exposure_us, gain_db, fps, lens_position, width, height,
            crop_width, crop_height, max_send_fps, and raw and raw_channel as
            the phone has them), or None when nothing changed or the phone
            has not reported yet.
        """
        with self._cond:
            phone = dict(self.status.get("settings") or self.hello.get("settings") or {})
        if not phone:
            return None
        low = self._min_iso()
        iso = phone.get("iso")
        current = {key: phone.get(key) for key in SETTING_DECIMALS}
        current["gain_db"] = (iso_to_gain(float(iso), low)
                              if iso is not None and low else None)
        if current.get("lens_position") is not None and current["lens_position"] < 0:
            current["lens_position"] = None  # not focused yet
        now = time.monotonic()
        changes = {}
        for key, value in current.items():
            if value is None or now - self._requested_at.get(key, -QUIET_S) < QUIET_S:
                continue
            value = round(float(value), SETTING_DECIMALS[key])
            if SETTING_DECIMALS[key] == 0:
                value = int(value)
            if self._reported.get(key) != value:
                changes[key] = value
        for key in RAW_SETTINGS:
            value = phone.get(key)
            if (value is None or now - self._requested_at.get(key, -QUIET_S) < QUIET_S
                    or self._reported.get(key) == value):
                continue
            changes[key] = value
        self._reported.update(changes)
        return changes or None

    def _send(self, command):
        """Send one COMMAND message, if the link is open.

        Args: command is a dict.
        Returns: None. Raises OSError when the link fails.
        """
        with self._cond:
            sock = self._sock
        if sock is None:
            return
        with self._send_lock:
            sock.sendall(encode_json(KIND_COMMAND, command))

    def _min_iso(self):
        """Read the phone's lowest ISO.

        Args: None.
        Returns: A positive float, or None before the phone reported it.
        """
        limits = self.status.get("limits") or self.hello.get("limits") or {}
        iso = limits.get("iso")
        if isinstance(iso, (list, tuple)) and iso and float(iso[0]) > 0:
            return float(iso[0])
        return None

    def _iso_for(self, gain_db):
        """Convert a gain to the phone's ISO.

        Args: gain_db is the requested gain.
        Returns: The ISO, or None before the phone reported its lowest ISO.
        """
        low = self._min_iso()
        return None if low is None else gain_to_iso(gain_db, low)

    def limits(self):
        """Report the phone's ranges in the viewer's units.

        Args: None.
        Returns: {"ExposureTime": (low, high) µs, "Gain": (0, high) dB,
            "AcquisitionFrameRate": (low, high)}, each clipped to
            CAMERA_RANGES; a range the phone did not report is left out.
        """
        with self._cond:
            phone = self.status.get("limits") or self.hello.get("limits") or {}
        result = {}
        exposure = clip_range(phone.get("exposure_us"), "ExposureTime")
        if exposure:
            result["ExposureTime"] = exposure
        iso = phone.get("iso")
        if isinstance(iso, (list, tuple)) and len(iso) == 2:
            top = iso_to_gain(float(iso[1]), float(iso[0]))
            gain = clip_range([0.0, top], "Gain") if top is not None else None
            if gain:
                result["Gain"] = gain
        fps = clip_range(phone.get("fps"), "AcquisitionFrameRate")
        if fps:
            result["AcquisitionFrameRate"] = fps
        return result

    def read_stream_counters(self):
        """Report no transport counters: the phone skips frames by design.

        The app drops a frame whenever the USB link is still busy with the
        previous one, so a rising count is normal; describe() shows it.

        Args: None.
        Returns: An empty dict.
        """
        return {}

    def read_values(self):
        """Report the applied settings and the received frame rate.

        Args: None.
        Returns: A dict like CameraSession.read_values(): exposure (µs) and
            gain (dB) from the newest frame, requested fps, and resulting =
            frames received per second since the previous call (None before
            any arrived).
        """
        now = time.monotonic()
        with self._cond:
            frames, meta, count = self._frames, self._meta, self._bytes
            elapsed = now - self._since
            self._byte_rate = count / elapsed if elapsed > 0 else 0.0
            self._frames, self._bytes, self._since = 0, 0, now
        low = self._min_iso()
        return {"exposure": meta.exposure_us if meta else None,
                "gain": (iso_to_gain(meta.iso, low) if meta and low else None),
                "requested": self.requested["fps"],
                "resulting": frames / elapsed if frames and elapsed > 0 else None}

    def describe(self, values):
        """Format a read_values() result for the status line.

        Args: values is the dict returned by read_values().
        Returns: Status text naming the phone, the frame size and rates.
        """
        with self._cond:
            meta, status, connected = self._meta, self.status, self._sock is not None
            byte_rate = self._byte_rate
        if not connected:
            return f"iPhone link lost: reconnecting. {OPEN_HINT}"
        rate = values["resulting"]
        size = f"{meta.width} × {meta.height}" if meta else "waiting for frames"
        if meta and meta.raw:
            size += f" RAW {RAW_CHANNELS[meta.channel]}"
        received = (f"{rate:.1f} fps received ({byte_rate / 1e6:.0f} MB/s)" if rate
                    else "no frames yet")
        parts = [f"{self.identity.short_name()} {size} over USB: {received}"]
        camera_fps = status.get("camera_fps")
        if camera_fps:
            parts.append(f"camera {float(camera_fps):.1f} fps")
        dropped = status.get("dropped_frames")
        if dropped:
            parts.append(f"{int(dropped)} skipped on the phone while USB was busy")
        if meta and meta.raw:
            parts.append(f"black {meta.black_level:g}, white {meta.white_level:g} counts")
        if values["exposure"] is not None and meta:
            gain = values["gain"]
            parts.append(f"exposure {values['exposure']:.0f} µs, ISO {meta.iso:.0f}"
                         + (f" ({gain:.1f} dB)" if gain is not None else ""))
        if status.get("thermal") in THERMAL_WARNINGS:
            parts.append(f"phone thermal state {status['thermal']}")
        return "; ".join(parts)

    def _summary(self):
        """Summarise the phone for log lines.

        Args: None.
        Returns: Short text.
        """
        lines = self.identity.description()
        return " | ".join(lines) if lines else self.info.label


def shut_down(sock):
    """Shut a socket down and close it, ignoring errors.

    Shutting down first wakes a thread blocked in recv on every platform.

    Args: sock is a socket.
    Returns: None.
    """
    try:
        sock.shutdown(2)  # socket.SHUT_RDWR
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass
