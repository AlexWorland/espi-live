"""Tests for the analog capture-adapter table and capture identity (no device)."""

import unittest

import cv2

from espi_live.capture_models import (ADAPTERS, CaptureIdentity, adapter_for,
                                      fourcc_text)

USB_LIVE2 = ADAPTERS[(0x2040, 0xC200)]


def usb_live2(width=720, height=480, fourcc="YUY2"):
    """Build the identity a USB-Live2 reports after opening.

    Args: width and height are the delivered frame size; fourcc is the
        negotiated pixel format.
    Returns: A CaptureIdentity.
    """
    return CaptureIdentity(name="Hauppauge USB-Live2", backend="DSHOW",
                           path="usb#vid_2040&pid_c200", vid=0x2040, pid=0xC200,
                           width=width, height=height, fourcc=fourcc, fps=29.97,
                           adapter=USB_LIVE2)


class AdapterTests(unittest.TestCase):
    """Recognise capture adapters by USB ID or name."""

    def test_usb_live2_by_usb_id_and_by_name(self):
        """The USB-Live2 is found from its USB ID, or from its name alone.

        Args: None.
        Returns: None.
        """
        self.assertIs(adapter_for(0x2040, 0xC200, ""), USB_LIVE2)
        self.assertIs(adapter_for(None, None, "Hauppauge USB-Live2 Video"), USB_LIVE2)
        self.assertIs(adapter_for(None, None, "USB Live 2"), USB_LIVE2)
        self.assertEqual((USB_LIVE2.width, USB_LIVE2.height, USB_LIVE2.channel),
                         (720, 480, 0))
        self.assertTrue(USB_LIVE2.interlaced)

    def test_unknown_device_has_no_adapter(self):
        """A device with no USB ID and an unknown name is not recognised.

        Args: None.
        Returns: None.
        """
        self.assertIsNone(adapter_for(None, None, "OBS Virtual Camera"))
        self.assertIsNone(adapter_for(0x046D, 0x0825, "Webcam C270"))

    def test_fourcc_text(self):
        """FOURCC codes decode to text; 0 and junk decode to nothing.

        Args: None.
        Returns: None.
        """
        self.assertEqual(fourcc_text(cv2.VideoWriter_fourcc(*"YUY2")), "YUY2")
        self.assertEqual(fourcc_text(float(cv2.VideoWriter_fourcc(*"MJPG"))), "MJPG")
        self.assertEqual(fourcc_text(0), "")
        self.assertEqual(fourcc_text(-1), "")
        self.assertEqual(fourcc_text(None), "")
        self.assertEqual(fourcc_text(3), "")  # control characters, not a code


class CaptureIdentityTests(unittest.TestCase):
    """Describe a capture device the way the window shows cameras."""

    def test_recognised_identity(self):
        """A USB-Live2 at its native size only carries the analog note.

        Args: None.
        Returns: None.
        """
        identity = usb_live2()
        self.assertTrue(identity.supported)
        self.assertTrue(identity.interlaced)
        self.assertEqual(identity.short_name(), "Hauppauge USB-Live2")
        text = "\n".join(identity.description())
        self.assertIn("720 × 480", text)
        self.assertIn("interlaced", text)
        self.assertIn("YUY2", text)
        self.assertIn("USB 2040:c200", text)
        warnings = identity.warnings()
        self.assertEqual(len(warnings), 1)
        self.assertIn("set on the camera", warnings[0])
        self.assertIn("single-field", warnings[0])

    def test_pal_warning(self):
        """A USB-Live2 left in PAL is called out.

        Args: None.
        Returns: None.
        """
        warnings = usb_live2(720, 576).warnings()
        self.assertTrue(any("720 × 576" in note and "PAL" in note for note in warnings))

    def test_rescaled_warning(self):
        """A size other than the native one is reported as rescaled.

        Args: None.
        Returns: None.
        """
        warnings = usb_live2(640, 480).warnings()
        self.assertTrue(any("640 × 480" in note and "rescaled" in note
                            for note in warnings))

    def test_unsupported_and_compressed_warnings(self):
        """An unknown webcam sending MJPEG is unsupported and lossy.

        Args: None.
        Returns: None.
        """
        identity = CaptureIdentity(name="Webcam", fourcc="MJPG", width=640, height=480)
        self.assertFalse(identity.supported)
        self.assertFalse(identity.interlaced)
        warnings = identity.warnings()
        self.assertTrue(warnings[0].startswith("Webcam is not yet supported"))
        self.assertTrue(any("MJPG" in note and "not suitable for ESPI" in note
                            for note in warnings))
        self.assertFalse(any("set on the camera" in note for note in warnings))

    def test_names_and_description_fallbacks(self):
        """Empty identities still name themselves and describe what is known.

        Args: None.
        Returns: None.
        """
        empty = CaptureIdentity()
        self.assertEqual(empty.short_name(), "Video device")
        self.assertEqual(empty.description(), ["Video device (video capture)"])
        self.assertTrue(empty.warnings()[0].startswith("This video device is not yet"))
        named = CaptureIdentity(name="Webcam", width=1280, height=720, fps=0.0)
        self.assertEqual(named.short_name(), "Webcam")
        self.assertEqual(named.description(),
                         ["Webcam (video capture)", "Video: 1280 × 720, unknown format"])
