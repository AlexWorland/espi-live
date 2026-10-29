"""Recognise analog video-capture adapters and describe capture devices.

Pure data and text: no OpenCV capture, PySpin or GUI imports, so both the
capture source and the tests share it. An analog camera such as the Sony
XC-ST50 cannot identify itself; what the viewer can recognise is the
adapter that digitizes its composite video (for example a Hauppauge
USB-Live2), by USB vendor/product ID or, failing that, by name.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CaptureAdapter:
    """What the capture source needs to know about a recognised adapter."""

    name: str
    interlaced: bool
    width: int     # native frame size to request (NTSC)
    height: int
    channel: int = 0  # OpenCV CAP_PROP_CHANNEL: 0 composite, 1 S-Video


# Adapters tested with this viewer, keyed by USB (vendor ID, product ID).
ADAPTERS = {
    (0x2040, 0xC200): CaptureAdapter("Hauppauge USB-Live2", True, 720, 480, 0),
}
# Lower-case name fragments for drivers that do not report a USB ID.
ADAPTER_NAMES = (
    ("usb-live2", (0x2040, 0xC200)),
    ("usb live2", (0x2040, 0xC200)),
    ("usb live 2", (0x2040, 0xC200)),
)
# Pixel formats whose compression destroys speckle detail.
COMPRESSED_FOURCCS = frozenset({"MJPG", "H264", "H265", "HEVC", "AVC1", "X264",
                                "MP4V"})


def adapter_for(vid, pid, name):
    """Look up a recognised capture adapter.

    Args: vid and pid are the USB vendor and product IDs, or None when the
        driver does not report them; name is the device name.
    Returns: The CaptureAdapter, or None for an unrecognised device.
    """
    if vid is not None and pid is not None and (vid, pid) in ADAPTERS:
        return ADAPTERS[(vid, pid)]
    lowered = (name or "").lower()
    for fragment, usb_id in ADAPTER_NAMES:
        if fragment in lowered:
            return ADAPTERS[usb_id]
    return None


def fourcc_text(code):
    """Decode an OpenCV FOURCC property value.

    Args: code is the number from CAP_PROP_FOURCC (int or float), or None.
    Returns: The four-character code, or "" when there is none.
    """
    try:
        value = int(code)
    except (TypeError, ValueError):
        return ""
    if value <= 0:
        return ""
    text = "".join(chr((value >> (8 * shift)) & 0xFF) for shift in range(4))
    return text if all(" " < char < "\x7f" for char in text) else ""


@dataclass(frozen=True)
class CaptureIdentity:
    """What an opened capture device reports (empty fields when unknown)."""

    name: str = ""
    backend: str = ""
    path: str = ""
    vid: int = None
    pid: int = None
    width: int = 0
    height: int = 0
    fourcc: str = ""
    fps: float = 0.0
    adapter: CaptureAdapter = None

    @property
    def supported(self):
        """Report whether the device is a recognised capture adapter.

        Args: None.
        Returns: True for a recognised adapter.
        """
        return self.adapter is not None

    @property
    def interlaced(self):
        """Report whether frames hold two interlaced fields.

        Args: None.
        Returns: True for recognised interlaced adapters; unknown devices are
            treated as progressive.
        """
        return bool(self.adapter and self.adapter.interlaced)

    def short_name(self):
        """Name the device compactly for titles and badges.

        Args: None.
        Returns: The adapter name, else the device name, else "Video device".
        """
        if self.adapter:
            return self.adapter.name
        return self.name or "Video device"

    def description(self):
        """Describe the device in a few human-readable lines.

        Args: None.
        Returns: A list of strings: name and kind, frame format, and USB ID
            and backend where known.
        """
        kind = "analog capture" if self.adapter else "video capture"
        lines = [f"{self.short_name()} ({kind})"]
        if self.width and self.height:
            video = (f"Video: {self.width} × {self.height}, "
                     f"{self.fourcc or 'unknown format'}")
            if self.fps:
                video += f", {self.fps:.2f} fps"
            if self.interlaced:
                video += ", interlaced"
            lines.append(video)
        ids = []
        if self.vid is not None and self.pid is not None:
            ids.append(f"USB {self.vid:04x}:{self.pid:04x}")
        if self.backend:
            ids.append(self.backend)
        if ids:
            lines.append(" · ".join(ids))
        return lines

    def warnings(self):
        """List reasons this source may not give usable ESPI data.

        Args: None.
        Returns: A list of warning strings, most important first.
        """
        notes = []
        if not self.supported:
            notes.append(f"{self.name or 'This video device'} is not yet supported: "
                         "it is shown as delivered and reported to GitHub so it "
                         "can be added.")
        if self.fourcc in COMPRESSED_FOURCCS:
            notes.append(f"The device sends {self.fourcc}-compressed video; "
                         "compression damages speckle, so this source is not "
                         "suitable for ESPI measurements.")
        adapter = self.adapter
        if adapter and self.width and self.height and (
                (self.width, self.height) != (adapter.width, adapter.height)):
            mismatch = (f"The driver delivers {self.width} × {self.height} instead "
                        f"of {adapter.width} × {adapter.height}")
            if self.height == 576:
                notes.append(mismatch + ": it is set to PAL, but an EIA camera such "
                             "as the XC-ST50 needs NTSC (set it in WinTV or "
                             "Hauppauge Capture).")
            else:
                notes.append(mismatch + ": the image may be rescaled, which blurs "
                             "the speckle.")
        if self.interlaced:
            notes.append("Interlaced analog camera: exposure, gain and frame rate "
                         "are set on the camera, not here; single-field mode "
                         "removes the 1/60 s offset between the two fields but "
                         "halves vertical detail.")
        return notes
