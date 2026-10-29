"""Tests for camera-model detection and description (no SDK needed)."""

import unittest

from espi_live.camera_models import CameraIdentity, parse_part_number


class ParsePartNumberTests(unittest.TestCase):
    """Decode Teledyne/FLIR part numbers from reported model names."""

    def test_reference_model(self):
        """The Blackfly S the viewer was built for decodes fully.

        Args: None.
        Returns: None.
        """
        part = parse_part_number("Blackfly S BFS-U3-63S4M")
        self.assertEqual(part.text, "BFS-U3-63S4M")
        self.assertEqual(part.family, "Blackfly S")
        self.assertEqual(part.interface, "USB3")
        self.assertAlmostEqual(part.megapixels, 6.3)
        self.assertEqual(part.sensor_maker, "Sony")
        self.assertEqual(part.spectrum, "mono")

    def test_other_families_and_interfaces(self):
        """GigE, three-digit resolution, onsemi and colour codes decode.

        Args: None.
        Returns: None.
        """
        part = parse_part_number("Grasshopper3 gs3-pge-123s6c-c")
        self.assertEqual((part.text, part.family, part.interface),
                         ("GS3-PGE-123S6C", "Grasshopper3", "PoE GigE"))
        self.assertAlmostEqual(part.megapixels, 12.3)
        self.assertEqual(part.spectrum, "colour")
        chameleon = parse_part_number("Chameleon3 CM3-U3-13Y3M")
        self.assertEqual(chameleon.sensor_maker, "onsemi")

    def test_unknown_codes_pass_through(self):
        """Unlisted family, maker and spectrum codes are kept or left blank.

        Args: None.
        Returns: None.
        """
        part = parse_part_number("XYZ-10G-50E5")
        self.assertEqual((part.family, part.interface, part.sensor_maker,
                          part.spectrum), ("XYZ", "10GigE", "E", ""))

    def test_no_part_number(self):
        """Names without a part number, and None, are not recognised.

        Args: None.
        Returns: None.
        """
        self.assertIsNone(parse_part_number("Some USB camera"))
        self.assertIsNone(parse_part_number(None))


class CameraIdentityTests(unittest.TestCase):
    """Names, descriptions and warnings shown in the UI."""

    def test_reference_model_description(self):
        """The reference camera is fully described and has no warnings.

        Args: None.
        Returns: None.
        """
        identity = CameraIdentity(
            vendor="FLIR", model="Blackfly S BFS-U3-63S4M", serial="123",
            firmware="1.0", width=3072, height=2048)
        self.assertEqual(identity.short_name(), "BFS-U3-63S4M")
        self.assertEqual(identity.description(), [
            "Blackfly S BFS-U3-63S4M (FLIR)",
            "Sensor: Sony IMX178, rolling shutter with global reset, "
            "3072 × 2048 (6.3 MP), mono, USB3",
            "S/N 123 · firmware 1.0"])
        self.assertEqual(identity.warnings(), [])

    def test_reported_sensor_wins_and_family_is_prefixed(self):
        """SensorDescription replaces the table; a bare part gets its family.

        Args: None.
        Returns: None.
        """
        identity = CameraIdentity(model="BFS-U3-51S5M", sensor="IMX250 mono")
        self.assertEqual(identity.description(), [
            "Blackfly S BFS-U3-51S5M",
            "Sensor: IMX250 mono, global shutter, mono, USB3"])
        self.assertEqual(identity.warnings(), [])

    def test_untested_colour_model_warns(self):
        """A colour model outside the table gets both warnings.

        Args: None.
        Returns: None.
        """
        identity = CameraIdentity(model="Oryx ORX-10G-71S7C")
        self.assertIsNone(identity.known)
        self.assertEqual(identity.description(),
                         ["Oryx ORX-10G-71S7C", "Sensor: colour, 10GigE"])
        warnings = identity.warnings()
        self.assertEqual(len(warnings), 2)
        self.assertTrue(warnings[0].startswith("Colour camera"))
        self.assertIn("ORX-10G-71S7C has not been tested", warnings[1])

    def test_unrecognised_and_empty_identity(self):
        """Without a part number the reported name, or a generic one, is used.

        Args: None.
        Returns: None.
        """
        other = CameraIdentity(model="Mystery cam")
        self.assertEqual(other.short_name(), "Mystery cam")
        self.assertEqual(other.description(), ["Mystery cam"])
        self.assertIn("not recognised", other.warnings()[0])
        empty = CameraIdentity()
        self.assertEqual(empty.short_name(), "Camera")
        self.assertEqual(empty.description(), ["Unknown camera"])


if __name__ == "__main__":
    unittest.main()
