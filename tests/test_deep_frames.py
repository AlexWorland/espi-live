"""Tests for deep (float32) frames: the processing steps and the iPhone's RAW mode."""

import time
import unittest

import numpy as np

from espi_live import iphone, processing
from espi_live.camera import CameraInfo
from fake_iphone import hello, phone_settings, phone_socket, status
from list_log import ListLog

UDID = "00008110-001A2B3C4D5E801E"
BLACK, WHITE = 528.0, 4095.0


def deep(values):
    """Make a deep frame from nested lists of grey levels.

    Args: values is a nested list (or array) of 8-bit-equivalent levels.
    Returns: A float32 array.
    """
    return np.array(values, dtype=processing.DEEP_DTYPE)


def wait_until(predicate, timeout=5.0):
    """Poll a condition until it is true or a timeout passes.

    Args: predicate is a zero-argument callable; timeout is in seconds.
    Returns: The final value of predicate().
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def raw_levels(counts):
    """Convert sensor counts to grey levels as the viewer should.

    Args: counts is an array of sensor counts.
    Returns: A float64 array: black at 0, the white level at 255.
    """
    return (np.asarray(counts, np.float64) - BLACK) * 255.0 / (WHITE - BLACK)


class DeepConversionTests(unittest.TestCase):
    """to_uint8 and saturate."""

    def test_to_uint8_rounds_and_saturates_only_at_the_end(self):
        """A deep frame becomes uint8 with negatives at 0 and fractions rounded.

        Args: None.
        Returns: None.
        """
        image = deep([[-3.2, 0.4, 0.6, 254.6, 300.0]])
        result = processing.to_uint8(image)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.tolist(), [[0, 0, 1, 255, 255]])
        already = np.array([[7]], np.uint8)
        self.assertIs(processing.to_uint8(already), already)

    def test_saturate_follows_the_frames_type(self):
        """uint8 frames give rounded uint8; deep ones keep fractions, capped at 255.

        Args: None.
        Returns: None.
        """
        rounded = processing.saturate(deep([[300.0, 2.6]]), np.zeros((1, 2), np.uint8))
        self.assertEqual((rounded.dtype, rounded.tolist()), (np.uint8, [[255, 3]]))
        values = deep([[300.0, -2.5, 0.25]])
        capped = processing.saturate(values, values)
        self.assertIs(capped, values)  # in place
        self.assertEqual(capped.dtype, np.float32)
        self.assertEqual(capped.tolist(), [[255.0, -2.5, 0.25]])


class DeepLevelsTests(unittest.TestCase):
    """Stretch levels, levels mapping and the range tracker on deep frames."""

    def test_stretch_levels_keep_fractions(self):
        """Exact, clipped and flat deep images give float levels.

        Args: None.
        Returns: None.
        """
        ramp = deep(np.linspace(0.0, 1.0, 1000).reshape(10, 100))
        low, high = processing.stretch_levels(ramp, 0)
        self.assertEqual((low, high), (0.0, 1.0))
        low, high = processing.stretch_levels(ramp, 5.0)
        self.assertIsInstance(low, float)
        self.assertAlmostEqual(low, 0.05, delta=0.003)
        self.assertAlmostEqual(high, 0.95, delta=0.003)
        flat = deep(np.full((4, 4), 0.375))
        self.assertEqual(processing.stretch_levels(flat, 5.0), (0.375, 0.375))

    def test_apply_levels_maps_fractional_levels(self):
        """Fractions of a grey level between low and high spread over 0..255.

        Args: None.
        Returns: None.
        """
        result = processing.apply_levels(deep([[0.1, 0.25, 0.375, 0.75, 2.0]]), 0.25, 0.75)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.tolist(), [[0, 0, 64, 255, 255]])

    def test_range_tracker_fine_keeps_floats(self):
        """fine=True returns float levels, widened by the floor when one is set.

        Args: None.
        Returns: None.
        """
        tracker = processing.RangeTracker()
        self.assertEqual(tracker.update(0.1, 0.3, 0, 0, 0.0, fine=True), (0.1, 0.3))
        low, high = tracker.update(0.1, 0.3, 1.0, 0, 1.0, fine=True)
        self.assertEqual(low, 0.1)
        self.assertAlmostEqual(high, 1.1)
        self.assertEqual(tracker.update(0.1, 0.3, 1.0, 0, 2.0), (0, 1))  # coarse rounds

    def test_noise_calibration_allows_a_floor_under_one_level(self):
        """A RAW still scene's noise under one grey level suggests floor 0.

        Args: None.
        Returns: None.
        """
        rng = np.random.default_rng(1)
        calibration = processing.NoiseCalibration(0.0)
        for _ in range(3):
            calibration.add(deep(rng.uniform(0.0, 0.3, (32, 32))))
        noise, floor = calibration.result()
        self.assertLess(noise, 0.5)
        self.assertGreater(noise, 0.2)
        self.assertEqual(floor, 0)
        coarse = processing.NoiseCalibration(0.0)
        coarse.add(np.zeros((32, 32), np.uint8))
        self.assertEqual(coarse.result(), (0.0, 1))  # 8-bit frames keep floor >= 1


class DeepProcessingTests(unittest.TestCase):
    """Gain, reference, averaging and noise reduction on deep frames."""

    def test_apply_gain_keeps_fractions_and_negatives(self):
        """A fixed gain multiplies a deep frame, capping only at the top.

        Args: None.
        Returns: None.
        """
        result = processing.apply_gain(deep([[0.25, 100.0, -1.0]]), 3.0)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.tolist(), [[0.75, 255.0, -3.0]])

    def test_reference_capture_mean_is_deep(self):
        """Averaged reference frames keep the exact float mean.

        Args: None.
        Returns: None.
        """
        capture = processing.ReferenceCapture(2, 0.0, 0.0, True)
        self.assertIsNone(capture.add(deep([[0.25, -1.0]])))
        mean = capture.add(deep([[0.5, 0.0]]))
        self.assertEqual(mean.dtype, np.float32)
        self.assertEqual(mean.tolist(), [[0.375, -0.5]])

    def test_average_frames_before_difference_is_deep(self):
        """The block mean of camera frames keeps fractions for deep frames.

        Args: None.
        Returns: None.
        """
        espi = processing.ESPIProcessor()
        self.assertIsNone(espi.average_frames(deep([[1.0, -0.5]]), 2))
        mean = espi.average_frames(deep([[1.5, 0.0]]), 2)
        self.assertEqual(mean.dtype, np.float32)
        self.assertEqual(mean.tolist(), [[1.25, -0.25]])
        self.assertIsNone(espi.frame_accum)  # the sum starts over

    def test_denoise_hands_out_a_copy(self):
        """Noise reduction blends deep frames and returns a copy of the running sum.

        Args: None.
        Returns: None.
        """
        espi = processing.ESPIProcessor()
        first = deep([[0.0, 1.0]])
        self.assertIs(espi.denoise(first, 0.5), first)
        result = espi.denoise(deep([[1.0, 0.0]]), 0.5)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, [[0.5, 0.5]])
        self.assertIsNot(result, espi.denoised)
        espi.denoise(deep([[1.0, 1.0]]), 0.5)
        np.testing.assert_allclose(result, [[0.5, 0.5]])  # unchanged by later frames

    def test_average_with_scale_is_deep_and_saturates(self):
        """The LabVIEW sum x multiplier / N keeps fractions and caps at 255.

        Args: None.
        Returns: None.
        """
        espi = processing.ESPIProcessor()
        self.assertIsNone(espi.average(deep([[0.25, 200.0]]), 2, scale=3.0))
        result = espi.average(deep([[0.5, 200.0]]), 2, scale=3.0)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.tolist(), [[1.125, 255.0]])

    def test_rolling_average_is_deep(self):
        """The running average keeps fractions; a scale saturates at 255.

        Args: None.
        Returns: None.
        """
        espi = processing.ESPIProcessor()
        first = espi.rolling_average(deep([[0.25, 1.0]]), 2)
        self.assertEqual(first.dtype, np.float32)
        self.assertEqual(first.tolist(), [[0.25, 1.0]])
        second = espi.rolling_average(deep([[0.75, 1.0]]), 2)
        self.assertEqual(second.tolist(), [[0.5, 1.0]])
        scaled = espi.rolling_average(deep([[0.5, 1.0]]), 2, scale=400.0)
        self.assertEqual(scaled.tolist(), [[200.0, 255.0]])


class RawProtocolTests(unittest.TestCase):
    """RAW frames on the link and the identity in RAW mode."""

    def test_raw_frame_round_trip(self):
        """A RAW plane decodes to float grey levels, negative below black.

        Args: None.
        Returns: None.
        """
        counts = np.array([[0, 528, 600], [2311, 4095, 5000]], np.uint16)
        data = iphone.encode_raw_frame(counts, BLACK, WHITE, channel=1, pattern=2,
                                       sequence=7, timestamp_ns=99, exposure_us=500.0,
                                       iso=64.0, lens_position=0.25)
        header = iphone.HEADER.unpack_from(data)
        self.assertEqual(header[2], iphone.KIND_FRAME)
        meta, image = iphone.decode_frame(bytearray(data[iphone.HEADER.size:]))
        self.assertEqual(meta, iphone.FrameMeta(3, 2, 7, 99, 500.0, 64.0, 0.25, True, 1,
                                                BLACK, WHITE))
        self.assertEqual((image.dtype, image.shape), (np.float32, (2, 3)))
        np.testing.assert_allclose(image, raw_levels(counts), rtol=1e-5)
        self.assertLess(image[0, 0], 0)
        self.assertEqual(image[0, 1], 0.0)
        self.assertAlmostEqual(float(image[1, 1]), 255.0, places=3)
        self.assertNotEqual(float(image[0, 2]), round(float(image[0, 2])))  # fraction kept

    def test_raw_frame_padded_rows(self):
        """Row padding in a RAW plane is dropped.

        Args: None.
        Returns: None.
        """
        counts = np.array([[600, 700, 1], [800, 900, 1]], np.uint16)  # last column pads
        meta = iphone.FRAME_META.pack(2, 2, 3, iphone.PIXEL_MONO16, 0, 0, 0.0, 0.0, 0.0, 0)
        payload = bytearray(meta + iphone.RAW_META.pack(BLACK, WHITE, 0, 0)
                            + counts.astype("<u2").tobytes())
        decoded, image = iphone.decode_frame(payload)
        self.assertEqual((decoded.width, decoded.height, image.shape), (2, 2, (2, 2)))
        np.testing.assert_allclose(image, raw_levels(counts[:, :2]), rtol=1e-5)

    def test_bad_raw_frames(self):
        """Truncated, mis-sized and impossible RAW frames raise LinkError.

        Args: None.
        Returns: None.
        """
        counts = np.full((2, 2), 1000, np.uint16)
        good = iphone.encode_raw_frame(counts, BLACK, WHITE)[iphone.HEADER.size:]
        meta = good[:iphone.FRAME_META.size]
        cases = {
            "RAW frame header is truncated": meta + b"\0" * 4,
            "RAW frame arrived with": good[:-2],
            "black level 600, white level 500": (
                meta + iphone.RAW_META.pack(600.0, 500.0, 0, 0)
                + good[iphone.FRAME_META.size + iphone.RAW_META.size:]),
            "colour 3": (meta + iphone.RAW_META.pack(BLACK, WHITE, 3, 0)
                         + good[iphone.FRAME_META.size + iphone.RAW_META.size:]),
        }
        for message, payload in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(iphone.LinkError, message):
                    iphone.decode_frame(bytearray(payload))

    def test_identity_in_raw_mode(self):
        """RAW mode names the colour plane and warns that the pipeline is bypassed.

        Args: None.
        Returns: None.
        """
        identity = iphone.IPhoneIdentity(serial=UDID, device_model="iPhone16,1",
                                         system_version="27.1", app_version="0.1 (5)",
                                         camera="Back", width=1920, height=1080,
                                         raw=True, raw_channel="green")
        lines = identity.description()
        self.assertEqual(lines[1], "Back: Bayer RAW green plane, 16-bit, no image "
                                   "processing, uncompressed over USB")
        self.assertNotIn("1920 × 1080", " ".join(lines))
        warning = identity.warnings()[0]
        self.assertIn("RAW mode", warning)
        self.assertIn("pipeline is bypassed", warning)
        unnamed = iphone.IPhoneIdentity(device_model="iPhone16,1", raw=True)
        self.assertEqual(unnamed.description()[1].split(":")[0], "Camera")


class RawSessionTests(unittest.TestCase):
    """An IPhoneSession whose phone switches to RAW mode."""

    def setUp(self):
        """Open a session over a scripted link.

        Args: None.
        Returns: None.
        """
        self.log = ListLog()
        info = CameraInfo("iphone-4d5e801e", "iPhone (USB)", kind="iphone", path=UDID)
        self.sock = phone_socket(hello())
        self.session = iphone.IPhoneSession(self.log, info, 8000.0, 0.0, 25.0,
                                            connector=lambda _info: self.sock)
        self.addCleanup(self.session.close)
        iphone._lens_cache.clear()
        self.addCleanup(iphone._lens_cache.clear)
        self.session.open()

    def test_status_switches_identity_and_reports_settings(self):
        """A STATUS in RAW mode updates the identity and reports raw settings once.

        Args: None.
        Returns: None.
        """
        self.assertFalse(self.session.identity.raw)
        self.session.take_settings_update()  # the HELLO's own values
        self.sock.feed(iphone.encode_json(iphone.KIND_STATUS, status(
            settings=phone_settings(raw=True, raw_channel="green"))))
        self.assertTrue(wait_until(lambda: self.session.identity.raw))
        self.assertEqual(self.session.identity.raw_channel, "green")
        self.assertEqual(self.session.identity.device_model, "iPhone16,1")  # kept
        changes = self.session.take_settings_update()
        self.assertEqual((changes["raw"], changes["raw_channel"]), (True, "green"))
        self.assertIsNone(self.session.take_settings_update())  # reported once
        self.session.send_command(raw=False)  # the viewer's own change is quiet
        self.sock.feed(iphone.encode_json(iphone.KIND_STATUS, status(
            settings=phone_settings(raw=True, raw_channel="blue"))))
        self.assertTrue(wait_until(lambda: self.session.identity.raw_channel == "blue"))
        self.assertEqual(self.session.take_settings_update(), {"raw_channel": "blue"})

    def test_raw_frames_reach_grab_and_the_status_line(self):
        """RAW frames arrive deep, and the status line names the plane and levels.

        Args: None.
        Returns: None.
        """
        counts = np.full((4, 8), 2311, np.uint16)
        self.sock.feed(iphone.encode_raw_frame(counts, BLACK, WHITE, channel=0,
                                               exposure_us=8000.0, iso=64.0))
        self.assertTrue(wait_until(lambda: self.session._frame_seq == 1))
        frame, state = self.session.grab(500)
        self.assertEqual((state, frame.dtype, frame.shape), ("ok", np.float32, (4, 8)))
        self.assertTrue(processing.is_deep(frame))
        np.testing.assert_allclose(frame, raw_levels(counts), rtol=1e-5)
        text = self.session.describe(self.session.read_values())
        self.assertIn("8 × 4 RAW red over USB", text)
        self.assertIn("black 528, white 4095 counts", text)


if __name__ == "__main__":
    unittest.main()
