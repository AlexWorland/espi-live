"""Identify the connected FLIR/Teledyne camera and describe it for display.

Pure data and text: no PySpin or GUI imports, so both front ends and the
tests share it. camera.read_identity() fills a CameraIdentity from the
device's GenICam nodes; this module interprets the reported model name.

Teledyne part numbers follow FAMILY-INTERFACE-SENSOR[-MOUNT], for example
BFS-U3-63S4M-C: Blackfly S, USB3, 6.3 MP, Sony, optical-format code 4,
monochrome, C-mount. Cameras report the name without the mount suffix,
usually after the family name ("Blackfly S BFS-U3-63S4M").
"""

import re
from collections import namedtuple
from dataclasses import dataclass

FAMILIES = {
    "BFS": "Blackfly S", "BFLY": "Blackfly", "GS3": "Grasshopper3",
    "GS2": "Grasshopper2", "CM3": "Chameleon3", "CM": "Chameleon",
    "ORX": "Oryx", "FFY": "Firefly", "FL3": "Flea3",
}
INTERFACES = {
    "U3": "USB3", "GE": "GigE", "PGE": "PoE GigE", "10G": "10GigE",
    "10GS": "10GigE",
}
SENSOR_MAKERS = {"S": "Sony", "Y": "onsemi", "M": "Micron"}
SPECTRA = {"M": "mono", "C": "colour", "S": "SWIR"}

PART_PATTERN = re.compile(
    r"\b([A-Z][A-Z0-9]{1,4})-(U3|PGE|GE|10GS|10G)-(\d{2,3})([A-Z])(\d)([A-Z]?)\b")

PartNumber = namedtuple(
    "PartNumber", "text family interface megapixels sensor_maker spectrum")

# Models whose sensor and shutter were checked against Teledyne/distributor
# specifications (see docs/camera/supported-models.md). Frame rates are not
# stored: published figures disagree, so the camera's own limits are used.
KNOWN_MODELS = {
    "BFS-U3-63S4M": ("Sony IMX178", "rolling shutter with global reset"),
    "BFS-U3-200S6M": ("Sony IMX183", "rolling shutter with global reset"),
    "BFS-U3-51S5M": ("Sony IMX250", "global shutter"),
    "BFS-U3-50S5M": ("Sony IMX264", "global shutter"),
    "BFS-U3-23S6M": ("Sony IMX249", "global shutter"),
    "BFS-U3-16S2M": ("Sony IMX273", "global shutter"),
    "BFS-U3-04S2M": ("Sony IMX287", "global shutter"),
    "BFS-U3-123S6M": ("Sony IMX253", "global shutter"),
}
REFERENCE_MODEL = "BFS-U3-63S4M"  # the camera this viewer was built and tuned on


def parse_part_number(text):
    """Find and decode a Teledyne/FLIR part number in a model name.

    Args: text is a reported model name such as "Blackfly S BFS-U3-63S4M".
    Returns: A PartNumber, or None when no part number is recognised.
    """
    match = PART_PATTERN.search((text or "").upper())
    if match is None:
        return None
    family, interface, resolution, maker, _optics, spectrum = match.groups()
    return PartNumber(
        text=match.group(0), family=FAMILIES.get(family, family),
        interface=INTERFACES[interface], megapixels=int(resolution) / 10,
        sensor_maker=SENSOR_MAKERS.get(maker, maker),
        spectrum=SPECTRA.get(spectrum, ""))


@dataclass(frozen=True)
class CameraIdentity:
    """What the connected camera reports about itself (empty when unknown)."""

    vendor: str = ""
    model: str = ""
    serial: str = ""
    firmware: str = ""
    sensor: str = ""
    width: int = 0
    height: int = 0

    @property
    def part(self):
        """Decode the part number from the reported model name.

        Args: None.
        Returns: A PartNumber, or None.
        """
        return parse_part_number(self.model)

    @property
    def known(self):
        """Look up curated sensor facts for this model.

        Args: None.
        Returns: (sensor, shutter) from KNOWN_MODELS, or None.
        """
        part = self.part
        return KNOWN_MODELS.get(part.text) if part else None

    def short_name(self):
        """Name the camera compactly for titles and badges.

        Args: None.
        Returns: The part number, else the reported model, else "Camera".
        """
        part = self.part
        return part.text if part else (self.model or "Camera")

    def description(self):
        """Describe the camera in a few human-readable lines.

        Args: None.
        Returns: A list of strings: model, sensor and resolution, and serial
            and firmware where reported.
        """
        part, known = self.part, self.known
        heading = self.model or "Unknown camera"
        if part and part.family not in heading:
            heading = f"{part.family} {heading}"
        if self.vendor:
            heading += f" ({self.vendor})"
        lines = [heading]

        sensor = self.sensor or (known[0] if known else "")
        details = [sensor] if sensor else []
        if known:
            details.append(known[1])
        if self.width and self.height:
            details.append(f"{self.width} × {self.height} "
                           f"({self.width * self.height / 1e6:.1f} MP)")
        if part and part.spectrum:
            details.append(part.spectrum)
        if part:
            details.append(part.interface)
        if details:
            lines.append("Sensor: " + ", ".join(details))

        ids = []
        if self.serial:
            ids.append(f"S/N {self.serial}")
        if self.firmware:
            ids.append(f"firmware {self.firmware}")
        if ids:
            lines.append(" · ".join(ids))
        return lines

    def warnings(self):
        """List reasons this camera may behave differently from the reference.

        Args: None.
        Returns: A list of warning strings; empty for the reference model.
        """
        part = self.part
        notes = []
        if part and part.spectrum == "colour":
            notes.append("Colour camera: ESPI needs monochrome speckle; Mono8 "
                         "from a colour sensor is interpolated from the Bayer "
                         "pattern, which blurs fine fringes.")
        if part is None:
            notes.append("Camera model not recognised; using the limits it "
                         "reports.")
        elif part.text != REFERENCE_MODEL and self.known is None:
            notes.append(f"{part.text} has not been tested with this viewer; "
                         "using the limits it reports.")
        return notes
