"""Camera- and network-free checks for espi_live.diagnostic_capture."""

import base64
import datetime
import io
import json
import os
import tempfile
import time
import unittest
import zipfile
from unittest import mock

import numpy as np

from espi_live import diagnostic_capture as dc
from espi_live import log_uploads
from espi_live.diagnostics import Burst
from espi_live.engine import Settings
from espi_live.feedback import FeedbackError

WHEN = datetime.datetime(2026, 11, 14, 15, 30, 12)


def frames(count=4, height=48, width=64, seed=0):
    """Build consecutive noisy frames.

    Args: count is the number of frames; height and width their size; seed
        selects the pattern.
    Returns: A tuple of uint8 arrays.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, (height, width), dtype=np.uint8)
    return tuple(np.clip(base.astype(int) + rng.integers(-2, 3, base.shape), 0, 255)
                 .astype(np.uint8) for _ in range(count))


def burst(**changes):
    """Build a Burst like the one an analog camera's engine posts.

    Args: changes override fields.
    Returns: A Burst.
    """
    raw = frames()
    values = dict(frames=raw, colours=(np.dstack([raw[0]] * 3),), interlaced=True,
                  camera=("Hauppauge USB-Live2 (analog capture) serial 21234567",),
                  details=(("Device name", "Hauppauge USB-Live2"),
                           ("Serial", "21234567")),
                  settings=Settings(), screen=np.zeros((48, 64), np.uint8), demo=False)
    values.update(changes)
    return Burst(**values)


def scrub(text):
    """Stand in for the log scrubber: hide one serial.

    Args: text is the text to scrub.
    Returns: The text with "21234567" replaced.
    """
    return text.replace("21234567", "<serial>")


def unzip(data):
    """Open a package.

    Args: data is the zip bytes.
    Returns: A ZipFile.
    """
    return zipfile.ZipFile(io.BytesIO(data))


class FakeRequest:
    """Record GitHub calls; optionally fail."""

    def __init__(self, error=None):
        """Create the fake.

        Args: error is a FeedbackError to raise, or None.
        Returns: None.
        """
        self.calls = []
        self.error = error

    def __call__(self, method, path, token, payload=None):
        """Record a call.

        Args: method, path, token and payload as github_request takes them.
        Returns: An empty reply. Raises the configured error.
        """
        self.calls.append((method, path, token, payload))
        if self.error is not None:
            raise self.error
        return {}


class FrameHelperTests(unittest.TestCase):
    """Region, cut, shrink, PNG encoding and names."""

    def test_small_frames_are_kept_whole_and_large_ones_cut(self):
        """An analog frame is whole; a 6 MP frame keeps its brightest square.

        Args: None.
        Returns: None.
        """
        small = np.zeros((480, 720), np.uint8)
        self.assertIsNone(dc.region_for(small))
        self.assertIs(dc.cut(small, None), small)
        large = np.zeros((2048, 3072), np.uint8)
        large[1000:1600, 2000:2600] = 200
        region = dc.region_for(large)
        self.assertEqual(region[2:], (dc.CROP_SIDE, dc.CROP_SIDE))
        self.assertEqual(dc.cut(large, region).shape, (dc.CROP_SIDE, dc.CROP_SIDE))
        self.assertGreater(dc.cut(large, region).mean(), 50)

    def test_shrink_keeps_aspect_and_leaves_small_images(self):
        """Wide images shrink to PREVIEW_WIDTH; narrow ones are returned as is.

        Args: None.
        Returns: None.
        """
        wide = np.zeros((2048, 3072), np.uint8)
        self.assertEqual(dc.shrink(wide).shape, (1067, dc.PREVIEW_WIDTH))
        narrow = np.zeros((10, 20, 3), np.uint8)
        self.assertIs(dc.shrink(narrow), narrow)

    def test_png_reports_an_encoding_failure(self):
        """OpenCV refusing to encode raises ValueError.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(dc.cv2, "imencode", return_value=(False, None)):
            with self.assertRaises(ValueError):
                dc.png(np.zeros((2, 2), np.uint8))

    def test_names_mark_dummy_mode(self):
        """Local and remote names carry the time, machine and dummy marker.

        Args: None.
        Returns: None.
        """
        self.assertEqual(dc.package_name(WHEN, False), "diagnostics_20261114_153012.zip")
        self.assertEqual(dc.package_name(WHEN, True), "dummy_diagnostics_20261114_153012.zip")
        self.assertEqual(dc.remote_path(WHEN, "ab12cd34", False),
                         "diagnostics/2026-11-14/153012-ab12cd34.zip")
        self.assertEqual(dc.remote_path(WHEN, "ab12cd34", True),
                         "diagnostics/2026-11-14/153012-ab12cd34-dummy.zip")


class PackageTests(unittest.TestCase):
    """Variants, analysis and the zip."""

    def test_variants_cover_the_views_and_single_field(self):
        """Progressive sources get six images; interlaced ones two more.

        Args: None.
        Returns: None.
        """
        raw = frames()
        names = [name for name, _, _ in dc.variants(raw, False)]
        self.assertEqual(names, [
            "previous_difference_x8.png", "previous_stretch_clip0.png",
            "previous_stretch_clip3.5.png", "previous_average3_stretch_clip1.png",
            "previous_average3_gaussian7_stretch_clip1.png",
            "reference_first_vs_last_stretch_clip1.png"])
        interlaced = dc.variants(raw, True)
        self.assertEqual([name for name, _, _ in interlaced[-2:]],
                         ["single_field_raw.png",
                          "single_field_previous_stretch_clip3.5.png"])
        for _, image, note in interlaced:
            self.assertEqual((image.dtype, image.shape), (np.uint8, (48, 64)))
            self.assertTrue(note)
        x8 = dc.variants(raw, False)[0][1]
        self.assertEqual(x8.max() % dc.DIFFERENCE_GAIN, 0)

    def test_analysis_measures_each_pair_and_scrubs(self):
        """Every pair is described; repeats, ranges and colour are named.

        Args: None.
        Returns: None.
        """
        raw = frames()
        text = dc.analysis_text(burst(frames=raw + (raw[-1],)), None, scrub)
        self.assertIn("<serial>", text)
        self.assertNotIn("21234567", text)
        self.assertIn("Frames: 5 consecutive, 64 × 48, kept: whole frame", text)
        self.assertIn("Repeated frames: 1 of 4 pairs", text)
        self.assertIn("Frames 0→1: levels", text)
        self.assertIn("colour B−G 0.00 R−G 0.00", text.splitlines()[7])
        self.assertNotIn("colour", text.splitlines()[8])
        self.assertIn("field mismatch", text)
        self.assertIn("Previous view input range at clip 3.5 %:", text)
        cut_text = dc.analysis_text(burst(), (8, 16, 32, 32), scrub)
        self.assertIn("kept: region x 8, y 16, 32 × 32", cut_text)

    def test_package_holds_frames_views_texts_and_manifest(self):
        """An analog capture keeps whole frames, colour, screen, log and details.

        Args: None.
        Returns: None.
        """
        data = dc.build_package(burst(), "run 21234567\n", scrub, WHEN)
        archive = unzip(data)
        names = archive.namelist()
        for name in ("raw/frame_00.png", "raw/frame_03.png", "colour/frame_00.png",
                     "views/screen.png", "views/single_field_raw.png",
                     "analysis.txt", "device.txt", "log.txt", "manifest.json"):
            self.assertIn(name, names)
        self.assertNotIn("raw/full_frame_00_preview.png", names)
        self.assertEqual(archive.read("log.txt").decode(), "run <serial>\n")
        self.assertIn("Serial: <serial>", archive.read("device.txt").decode())
        manifest = json.loads(archive.read("manifest.json"))
        self.assertEqual(manifest["created"], "2026-11-14T15:30:12")
        self.assertFalse(manifest["demo"])
        self.assertEqual(manifest["camera"], ["Hauppauge USB-Live2 (analog capture) "
                                              "serial <serial>"])
        self.assertEqual(manifest["frame_size"], [64, 48])
        self.assertIsNone(manifest["region"])
        self.assertEqual(manifest["settings"]["mode"], "Raw")
        self.assertIn("raw/frame_00.png", manifest["files"])
        self.assertEqual(archive.getinfo("raw/frame_00.png").compress_type,
                         zipfile.ZIP_STORED)
        self.assertEqual(archive.getinfo("log.txt").compress_type, zipfile.ZIP_DEFLATED)

    def test_large_frames_are_cut_with_a_preview_and_optional_parts_left_out(self):
        """A multi-megapixel capture keeps a square plus a preview; no extras.

        Args: None.
        Returns: None.
        """
        raw = frames(count=2, height=1300, width=1300)
        data = dc.build_package(burst(frames=raw, colours=(), interlaced=False,
                                      details=(), screen=None), "", scrub, WHEN)
        archive = unzip(data)
        names = archive.namelist()
        self.assertIn("raw/full_frame_00_preview.png", names)
        for name in ("colour/frame_00.png", "views/screen.png", "device.txt", "log.txt",
                     "views/single_field_raw.png"):
            self.assertNotIn(name, names)
        manifest = json.loads(archive.read("manifest.json"))
        self.assertEqual(manifest["region"][2:], [dc.CROP_SIDE, dc.CROP_SIDE])


class SendTests(unittest.TestCase):
    """Saving, reading the log, uploading and the result."""

    def setUp(self):
        """Use a temporary logs folder.

        Args: None.
        Returns: None.
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = tmp.name
        self.log_path = os.path.join(self.folder, "espi_20261114_153000_1.log")
        with open(self.log_path, "w", encoding="utf-8") as handle:
            handle.write("camera 21234567 opened\n")

    def send(self, token="secret", request=None, serials=("21234567",), **changes):
        """Send a capture of the default burst.

        Args: token and request are passed to send(); serials are this run's
            serials; changes override Burst fields.
        Returns: The Result.
        """
        return dc.send(burst(**changes), self.folder, self.log_path, lambda: list(serials),
                       WHEN, token=token, request=request or FakeRequest())

    def test_sent_capture_is_committed_and_kept(self):
        """The zip goes to the logs repository as a new file and is saved locally.

        Args: None.
        Returns: None.
        """
        request = FakeRequest()
        result = self.send(request=request)
        self.assertEqual(result.outcome, "sent")
        machine = log_uploads.machine_id(self.folder)
        self.assertEqual(result.remote, f"diagnostics/2026-11-14/153012-{machine}.zip")
        self.assertEqual(os.path.basename(result.local), "diagnostics_20261114_153012.zip")
        (method, path, token, payload), = request.calls
        self.assertEqual((method, token), ("PUT", "secret"))
        self.assertEqual(path, f"/repos/{log_uploads.repository()}/contents/{result.remote}")
        self.assertEqual(payload["branch"], log_uploads.BRANCH)
        self.assertNotIn("sha", payload)
        sent = base64.b64decode(payload["content"])
        with open(result.local, "rb") as handle:
            self.assertEqual(handle.read(), sent)
        self.assertEqual(unzip(sent).read("log.txt").decode(),
                         "camera <serial> opened\n")

    def test_no_token_saves_only(self):
        """A copy without the logs token keeps the capture and sends nothing.

        Args: None.
        Returns: None.
        """
        request = FakeRequest()
        with mock.patch.object(log_uploads, "logs_token", return_value=""):
            result = self.send(token=None, request=request)
        self.assertEqual((result.outcome, result.error), ("no_token", ""))
        self.assertTrue(os.path.exists(result.local))
        self.assertEqual(request.calls, [])

    def test_failures_keep_the_saved_copy(self):
        """A GitHub error or an oversized package is reported; the copy stays.

        Args: None.
        Returns: None.
        """
        result = self.send(request=FakeRequest(FeedbackError("HTTP 422")))
        self.assertEqual((result.outcome, result.error), ("failed", "HTTP 422"))
        self.assertTrue(os.path.exists(result.local))
        with mock.patch.object(dc, "MAX_PACKAGE_BYTES", 10):
            result = self.send()
        self.assertEqual(result.outcome, "failed")
        self.assertIn("MB that can be sent", result.error)

    def test_an_unsaved_copy_is_reported(self):
        """When the copy cannot be written, the result says so.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(dc, "save_local", side_effect=OSError("disk full")):
            sent = self.send()
            unsent = self.send(token="")
        self.assertEqual((sent.outcome, sent.local), ("sent", ""))
        self.assertEqual(unsent.outcome, "no_token")
        self.assertEqual(unsent.error, "could not save a copy: disk full")

    def test_read_log_handles_missing_logs(self):
        """No log path or an unreadable one gives empty text.

        Args: None.
        Returns: None.
        """
        self.assertEqual(dc.read_log(None), "")
        self.assertEqual(dc.read_log(os.path.join(self.folder, "missing.log")), "")
        self.assertIn("opened", dc.read_log(self.log_path))


class WorkerTests(unittest.TestCase):
    """The background worker the window polls."""

    def wait_idle(self, worker):
        """Wait for the worker thread to finish.

        Args: worker is the DiagnosticWorker.
        Returns: None.
        """
        for _ in range(500):
            if not worker.busy:
                return
            time.sleep(0.01)

    def test_worker_posts_the_result_and_refuses_overlap(self):
        """One capture at a time; the result arrives as ("done", Result).

        Args: None.
        Returns: None.
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        worker = dc.DiagnosticWorker(tmp.name, None, list, token="")
        with mock.patch.object(dc, "send", side_effect=lambda *args: time.sleep(0.2)
                               or dc.Result("no_token", local="x")):
            self.assertTrue(worker.start(burst()))
            self.assertFalse(worker.start(burst()))
            kind, result = worker.events.get(timeout=5)
        self.assertEqual((kind, result.outcome), ("done", "no_token"))
        self.wait_idle(worker)
        self.assertFalse(worker.busy)

    def test_worker_uses_now_and_real_send(self):
        """Without a time the capture is stamped now and really packaged.

        Args: None.
        Returns: None.
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        worker = dc.DiagnosticWorker(tmp.name, None, list, token="")
        worker.start(burst())
        _, result = worker.events.get(timeout=10)
        self.assertEqual(result.outcome, "no_token")
        self.assertTrue(os.path.exists(result.local))

    def test_unexpected_failure_posts_and_reraises(self):
        """A bug while sending posts a failure and reaches threading.excepthook.

        Args: None.
        Returns: None.
        """
        worker = dc.DiagnosticWorker("unused", None, list, token="")
        with mock.patch.object(dc, "send", side_effect=RuntimeError("boom")), \
                mock.patch.object(dc.threading, "excepthook") as hook:
            worker.start(burst())
            _, result = worker.events.get(timeout=5)
            self.wait_idle(worker)
        self.assertEqual((result.outcome, result.error), ("failed", "RuntimeError: boom"))
        hook.assert_called_once()


if __name__ == "__main__":
    unittest.main()
