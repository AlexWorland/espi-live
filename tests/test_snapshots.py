"""Tests for the scheduled-capture file helpers (no camera, network or real output folder)."""

import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from espi_live import snapshots
from espi_live.camera import CameraInfo
from espi_live.engine import Settings

SERIAL = "21234567"


@dataclass
class FakeChannel:
    """The channel attributes the eligibility rule reads."""

    info: CameraInfo
    active: bool = True
    demo: bool = False
    supported: bool = True
    frozen: object = None
    demo_source: bool = False


def flir(**changes):
    """Build an eligible FLIR channel.

    Args: changes are FakeChannel fields to override.
    Returns: A FakeChannel.
    """
    return FakeChannel(CameraInfo(SERIAL, "Blackfly S BFS-U3-63S4M"), **changes)


class EligibleTests(unittest.TestCase):
    """Which channels may upload."""

    def test_active_flir_channel_is_eligible(self):
        """A processing FLIR camera uploads.

        Args: None.
        Returns: None.
        """
        self.assertTrue(snapshots.eligible(flir()))

    def test_supported_capture_adapter_is_eligible(self):
        """A recognised capture adapter uploads.

        Args: None.
        Returns: None.
        """
        channel = FakeChannel(CameraInfo("video 0", "USB-Live2", kind="capture"))
        self.assertTrue(snapshots.eligible(channel))

    def test_a_demo_channel_that_has_just_gone_live_is_not_eligible(self):
        """Until the window updates the channel, its demo info must not be trusted.

        Args: None.
        Returns: None.
        """
        channel = FakeChannel(CameraInfo("DEMO1", "Synthetic camera"), demo_source=True)
        self.assertFalse(channel.demo)  # the engine already switched
        self.assertFalse(snapshots.eligible(channel))

    def test_everything_else_is_not(self):
        """Demo, stopped, iPhone, unsupported capture and frozen channels never upload.

        Args: None.
        Returns: None.
        """
        phone = FakeChannel(CameraInfo("udid", "iPhone", kind="iphone"))
        unsupported = FakeChannel(CameraInfo("video 1", "Webcam", kind="capture"),
                                  supported=False)
        for channel in (flir(demo=True), flir(active=False), phone, unsupported,
                        flir(frozen=object())):
            self.assertFalse(snapshots.eligible(channel))


class NamingTests(unittest.TestCase):
    """Camera labels and remote paths."""

    def test_label_has_model_and_index_but_not_the_serial(self):
        """The label names the model and position only.

        Args: None.
        Returns: None.
        """
        label = snapshots.camera_label(CameraInfo(SERIAL, "Blackfly S BFS/U3 63S4M"), 2)
        self.assertIn("Blackfly", label)
        self.assertTrue(label.endswith("-2"))
        self.assertNotIn(SERIAL, label)
        self.assertNotIn("/", label)

    def test_label_without_a_model_is_just_the_index(self):
        """A camera with no model name still gets a label.

        Args: None.
        Returns: None.
        """
        self.assertEqual(snapshots.camera_label(CameraInfo(SERIAL), 0), "camera-0")

    def test_remote_stem(self):
        """The stem is dated, timed and carries the machine and camera.

        Args: None.
        Returns: None.
        """
        stem = snapshots.remote_stem(datetime(2026, 11, 21, 9, 30, 5), "ab12cd34", "FLIR-0")
        self.assertEqual(stem, "captures/2026-11-21/093005-ab12cd34-FLIR-0")


class ImageTests(unittest.TestCase):
    """8-bit conversion and size fitting."""

    def test_fit_width_shrinks_big_frames_and_keeps_small_ones(self):
        """Frames wider than the limit are scaled down with their aspect kept.

        Args: None.
        Returns: None.
        """
        wide = np.zeros((1536, 2048), np.uint8)
        small, scale = snapshots.fit_width(wide, 1024)
        self.assertEqual(small.shape, (768, 1024))
        self.assertEqual(scale, 0.5)
        same, scale = snapshots.fit_width(small, 1024)
        self.assertIs(same, small)
        self.assertEqual(scale, 1.0)

    def test_uint8_passes_through(self):
        """An 8-bit frame is returned as it is.

        Args: None.
        Returns: None.
        """
        image = np.arange(12, dtype=np.uint8).reshape(3, 4)
        out, rescaled = snapshots.to_uint8(image)
        self.assertIs(out, image)
        self.assertFalse(rescaled)

    def test_float_frame_is_min_max_scaled(self):
        """A deep float frame becomes 0..255 and is flagged.

        Args: None.
        Returns: None.
        """
        image = np.array([[0.5, 1.5], [2.5, 4.5]], dtype=np.float32)
        out, rescaled = snapshots.to_uint8(image)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual((int(out.min()), int(out.max())), (0, 255))
        self.assertTrue(rescaled)

    def test_flat_float_frame_does_not_divide_by_zero(self):
        """A constant deep frame becomes all zero.

        Args: None.
        Returns: None.
        """
        out, rescaled = snapshots.to_uint8(np.full((4, 4), 3.0, dtype=np.float32))
        self.assertEqual(int(out.max()), 0)
        self.assertTrue(rescaled)

    def test_png_within_downscales_noise_to_fit(self):
        """A noisy image over the budget comes back smaller and under it.

        Args: None.
        Returns: None.
        """
        noise = np.random.default_rng(1).integers(0, 256, (1500, 2000), dtype=np.uint8)
        data, scale = snapshots.png_within(noise, 600_000)
        self.assertLessEqual(len(data), 600_000)
        self.assertLess(scale, 1.0)
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(decoded.dtype, np.uint8)

    def test_png_within_keeps_a_smooth_image_whole(self):
        """An image that already fits is not scaled.

        Args: None.
        Returns: None.
        """
        data, scale = snapshots.png_within(np.zeros((100, 100), np.uint8), 100_000)
        self.assertEqual(scale, 1.0)
        self.assertGreater(len(data), 0)

    def test_png_within_gives_up_when_impossible(self):
        """No scale of at least 0.25 fits a tiny budget.

        Args: None.
        Returns: None.
        """
        noise = np.random.default_rng(2).integers(0, 256, (600, 800), dtype=np.uint8)
        self.assertIsNone(snapshots.png_within(noise, 50))


def write_avi(path, frames=30, size=(320, 240)):
    """Write a short noisy MJPG clip.

    Args: path is the file; frames is how many; size is (width, height).
    Returns: None.
    """
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 15.0, size, True)
    rng = np.random.default_rng(3)
    for _ in range(frames):
        writer.write(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
    writer.release()


class ShrinkClipTests(unittest.TestCase):
    """Re-encoding a clip smaller."""

    def setUp(self):
        """Make a temporary folder.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "clip.avi")
        write_avi(self.path)

    def tearDown(self):
        """Remove the folder.

        Args: None.
        Returns: None.
        """
        self.tmp.cleanup()

    def test_a_clip_that_fits_is_left_alone(self):
        """No re-encode when the file is already small enough.

        Args: None.
        Returns: None.
        """
        size = os.path.getsize(self.path)
        self.assertEqual(snapshots.shrink_clip(self.path, size + 1), 1.0)
        self.assertEqual(os.path.getsize(self.path), size)

    def test_a_big_clip_is_shrunk_under_the_cap(self):
        """A clip over the cap is re-encoded to fit.

        Args: None.
        Returns: None.
        """
        cap = os.path.getsize(self.path) // 2
        scale = snapshots.shrink_clip(self.path, cap)
        self.assertIsNotNone(scale)
        self.assertLess(scale, 1.0)
        self.assertLessEqual(os.path.getsize(self.path), cap)

    def test_an_impossible_cap_deletes_the_file(self):
        """A clip that cannot fit is removed.

        Args: None.
        Returns: None.
        """
        self.assertIsNone(snapshots.shrink_clip(self.path, 100))
        self.assertFalse(os.path.exists(self.path))

    def test_a_missing_file_gives_none(self):
        """An unreadable clip is dropped quietly.

        Args: None.
        Returns: None.
        """
        self.assertIsNone(snapshots.shrink_clip(os.path.join(self.tmp.name, "gone.avi"), 10))

    def test_an_unwritable_re_encode_gives_none(self):
        """A clip whose frames cannot be read back is deleted.

        Args: None.
        Returns: None.
        """
        with open(self.path, "wb") as handle:
            handle.write(b"not a video")
        self.assertIsNone(snapshots.shrink_clip(self.path, 5))
        self.assertFalse(os.path.exists(self.path))


class SidecarTests(unittest.TestCase):
    """The JSON that travels with a capture."""

    def test_sidecar_is_json_with_settings_and_no_serial(self):
        """The sidecar serializes, carries settings and shapes, and hides the serial.

        Args: None.
        Returns: None.
        """
        info = CameraInfo(SERIAL, "Blackfly S BFS-U3-63S4M")
        data = snapshots.sidecar(Settings(), info, "Reference", (2064, 3088), (2064, 3088),
                                 1.0, 0.75, True, 12)
        text = json.dumps(data)
        self.assertNotIn(SERIAL, text)
        self.assertEqual(data["view"], "Reference")
        self.assertEqual(data["model"], "Blackfly S BFS-U3-63S4M")
        self.assertEqual(data["build"], 12)
        self.assertEqual(data["raw"]["shape"], [2064, 3088])
        self.assertEqual(data["processed"]["scale"], 0.75)
        self.assertTrue(data["raw"]["rescaled"])
        self.assertEqual(data["settings"]["fps"], 30.0)


if __name__ == "__main__":
    unittest.main()
