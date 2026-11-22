"""Tests for the capture ledger, queue and uploader (no network: a fake GitHub call)."""

import base64
import json
import os
import tempfile
import unittest

from espi_live import capture_uploads
from espi_live.capture_uploads import (CLIP_RESERVE_BYTES, DAILY_BYTES, CaptureLedger,
                                       CaptureQueue, CaptureUploader)
from espi_live.feedback import FeedbackError

DAY = 86400.0
NOON = 1_800_000_000.0  # a fixed instant; the tests only add whole hours to it


class Clock:
    """A settable clock."""

    def __init__(self, now=NOON):
        """Start the clock.

        Args: now is the first reading in seconds since the epoch.
        Returns: None.
        """
        self.now = now

    def __call__(self):
        """Read the clock.

        Args: None.
        Returns: The current reading.
        """
        return self.now


class TempFolderCase(unittest.TestCase):
    """A test with a temporary folder."""

    def setUp(self):
        """Make the folder.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = self.tmp.name

    def tearDown(self):
        """Remove the folder.

        Args: None.
        Returns: None.
        """
        self.tmp.cleanup()


class LedgerTests(TempFolderCase):
    """Schedule and budget bookkeeping."""

    def setUp(self):
        """Make a ledger on a fake clock.

        Args: None.
        Returns: None.
        """
        super().setUp()
        self.clock = Clock()
        self.ledger = CaptureLedger(self.folder, self.clock)

    def test_due_first_time_then_not_until_interval(self):
        """A camera is due at first, then only after the interval.

        Args: None.
        Returns: None.
        """
        self.assertTrue(self.ledger.due("cam-0", "still", 1800))
        self.ledger.mark("cam-0", "still")
        self.assertFalse(self.ledger.due("cam-0", "still", 1800))
        self.assertTrue(self.ledger.due("cam-1", "still", 1800))
        self.assertTrue(self.ledger.due("cam-0", "clip", 1800))
        self.clock.now += 1800
        self.assertTrue(self.ledger.due("cam-0", "still", 1800))

    def test_schedule_survives_a_restart(self):
        """A new ledger over the same folder remembers the last capture.

        Args: None.
        Returns: None.
        """
        self.ledger.mark("cam-0", "still")
        again = CaptureLedger(self.folder, self.clock)
        self.assertFalse(again.due("cam-0", "still", 1800))

    def test_corrupt_schedule_is_due_once_without_burst(self):
        """A garbled file makes cameras due, and marking fixes it.

        Args: None.
        Returns: None.
        """
        with open(os.path.join(self.folder, "schedule.json"), "w") as handle:
            handle.write("{not json")
        ledger = CaptureLedger(self.folder, self.clock)
        self.assertTrue(ledger.due("cam-0", "still", 1800))
        ledger.mark("cam-0", "still")
        self.assertFalse(ledger.due("cam-0", "still", 1800))
        with open(os.path.join(self.folder, "schedule.json")) as handle:
            self.assertIsInstance(json.load(handle), dict)

    def test_wrong_shaped_schedule_is_due_once(self):
        """A schedule that is valid JSON but not a dict is ignored.

        Args: None.
        Returns: None.
        """
        with open(os.path.join(self.folder, "schedule.json"), "w") as handle:
            json.dump([1, 2], handle)
        self.assertTrue(CaptureLedger(self.folder, self.clock).due("c", "still", 10))

    def test_future_timestamp_is_due_once(self):
        """A last-capture time in the future (clock change) does not silence a camera.

        Args: None.
        Returns: None.
        """
        self.ledger.mark("cam-0", "still")
        self.clock.now -= 5 * DAY
        self.assertTrue(self.ledger.due("cam-0", "still", 1800))
        self.ledger.mark("cam-0", "still")
        self.assertFalse(self.ledger.due("cam-0", "still", 1800))

    def test_unwritable_folder_keeps_the_schedule_in_memory(self):
        """When the file cannot be written, marking still works for this run.

        Args: None.
        Returns: None.
        """
        blocked = os.path.join(self.folder, "file")
        with open(blocked, "w"):
            pass
        ledger = CaptureLedger(os.path.join(blocked, "inside"), self.clock)
        ledger.mark("cam-0", "still")
        self.assertFalse(ledger.due("cam-0", "still", 1800))

    def test_still_cannot_spend_clip_reserve(self):
        """Stills stop at the daily budget minus the clip reserve; clips may use the rest.

        Args: None.
        Returns: None.
        """
        room = DAILY_BYTES - CLIP_RESERVE_BYTES
        self.assertTrue(self.ledger.afford("still", room))
        self.ledger.spend("still", room)
        self.assertFalse(self.ledger.afford("still", 1))
        self.assertTrue(self.ledger.afford("clip", CLIP_RESERVE_BYTES))
        self.ledger.spend("clip", CLIP_RESERVE_BYTES)
        self.assertFalse(self.ledger.afford("clip", 1))

    def test_clips_may_use_unspent_still_room(self):
        """With no stills taken, a clip can use the whole daily budget.

        Args: None.
        Returns: None.
        """
        self.assertTrue(self.ledger.afford("clip", DAILY_BYTES))
        self.assertFalse(self.ledger.afford("clip", DAILY_BYTES + 1))

    def test_budget_resets_next_day(self):
        """A new calendar day starts with an empty budget.

        Args: None.
        Returns: None.
        """
        self.ledger.spend("clip", DAILY_BYTES)
        self.assertFalse(self.ledger.afford("clip", 1))
        self.clock.now += 2 * DAY
        self.assertTrue(self.ledger.afford("clip", DAILY_BYTES))

    def test_spent_bytes_survive_a_restart(self):
        """Today's spending is remembered across restarts.

        Args: None.
        Returns: None.
        """
        self.ledger.spend("clip", DAILY_BYTES)
        self.assertFalse(CaptureLedger(self.folder, self.clock).afford("clip", 1))


class QueueTests(TempFolderCase):
    """The on-disk queue."""

    STEM = "captures/2026-11-21/093005-ab12cd34-FLIR-0"

    def test_group_is_added_whole_and_read_back(self):
        """A group's files come back under their suffixes.

        Args: None.
        Returns: None.
        """
        queue = CaptureQueue(self.folder, 1_000_000)
        self.assertTrue(queue.add(self.STEM, {"-raw.png": b"a", ".json": b"{}"}))
        self.assertEqual(queue.groups(), [self.STEM])
        self.assertEqual(queue.read(self.STEM), {"-raw.png": b"a", ".json": b"{}"})
        queue.remove(self.STEM)
        self.assertEqual(queue.groups(), [])

    def test_oldest_groups_dropped_over_max_bytes(self):
        """The queue sheds its oldest groups when it grows past the cap.

        Args: None.
        Returns: None.
        """
        queue = CaptureQueue(self.folder, 25)
        stems = [f"captures/2026-11-21/09300{n}-m-cam-0" for n in range(3)]
        for stem in stems:
            self.assertTrue(queue.add(stem, {".png": b"x" * 10}))
        self.assertEqual(queue.groups(), stems[1:])

    def test_a_group_bigger_than_the_cap_is_refused(self):
        """A group that alone exceeds the cap is dropped and reported.

        Args: None.
        Returns: None.
        """
        queue = CaptureQueue(self.folder, 5)
        self.assertFalse(queue.add(self.STEM, {".png": b"x" * 10}))
        self.assertEqual(queue.groups(), [])

    def test_leftover_temp_folder_removed(self):
        """A half-written group from a crash never shows up and is cleaned.

        Args: None.
        Returns: None.
        """
        pending = os.path.join(self.folder, "pending")
        os.makedirs(os.path.join(pending, ".tmp-crashed"))
        with open(os.path.join(pending, ".tmp-crashed", "x.png"), "wb") as handle:
            handle.write(b"half")
        queue = CaptureQueue(self.folder, 1000)
        self.assertEqual(queue.groups(), [])
        self.assertFalse(os.path.exists(os.path.join(pending, ".tmp-crashed")))

    def test_add_returns_false_when_folder_unwritable(self):
        """A queue folder that cannot be created drops the capture quietly.

        Args: None.
        Returns: None.
        """
        blocked = os.path.join(self.folder, "file")
        with open(blocked, "w"):
            pass
        queue = CaptureQueue(os.path.join(blocked, "inside"), 1000)
        self.assertFalse(queue.add(self.STEM, {".png": b"x"}))
        self.assertEqual(queue.groups(), [])

    def test_a_stray_file_in_the_queue_counts_as_empty(self):
        """Something that is not a group folder does not break trimming.

        Args: None.
        Returns: None.
        """
        queue = CaptureQueue(self.folder, 1000)
        with open(os.path.join(queue.root, "captures~stray"), "w") as handle:
            handle.write("x")
        self.assertTrue(queue.add(self.STEM, {".png": b"x"}))

    def test_a_write_that_fails_midway_leaves_nothing_behind(self):
        """If the final rename fails, the temporary folder is cleaned up.

        Args: None.
        Returns: None.
        """
        queue = CaptureQueue(self.folder, 1000)
        with open(queue._folder(self.STEM), "w") as handle:  # a file where the group goes
            handle.write("in the way")
        self.assertFalse(queue.add(self.STEM, {".png": b"x"}))
        leftovers = [name for name in os.listdir(queue.root) if name.startswith(".tmp-")]
        self.assertEqual(leftovers, [])

    def test_read_of_a_missing_group_is_empty(self):
        """Reading a group that is gone gives nothing.

        Args: None.
        Returns: None.
        """
        self.assertEqual(CaptureQueue(self.folder, 10).read(self.STEM), {})


class FakeContents:
    """A stand-in for the GitHub Contents API that records PUTs and can fail."""

    def __init__(self, errors=None):
        """Create it.

        Args: errors is a list of FeedbackError (or None for success) answered
            one per call, then success.
        Returns: None.
        """
        self.errors = list(errors or [])
        self.calls = []

    def __call__(self, method, path, token, payload=None):
        """Answer one call.

        Args: method, path, token and payload as feedback.github_request takes them.
        Returns: A success reply; raises the next queued error.
        """
        self.calls.append((method, path, token, payload))
        error = self.errors.pop(0) if self.errors else None
        if error is not None:
            raise error
        return {"content": {"sha": "abc"}}

    def paths(self):
        """List the repository paths written.

        Args: None.
        Returns: The path after /contents/ of each call.
        """
        return [call[1].split("/contents/", 1)[1] for call in self.calls]


class UploaderTests(TempFolderCase):
    """Uploading the queue."""

    STEM = "captures/2026-11-21/093005-ab12cd34-FLIR-0"

    def setUp(self):
        """Make a queue with one group and a clock.

        Args: None.
        Returns: None.
        """
        super().setUp()
        self.clock = Clock()
        self.queue = CaptureQueue(self.folder, 1_000_000)
        self.queue.add(self.STEM, {"-raw.png": b"RAW", "-processed.png": b"PROC",
                                   ".json": b"{}"})

    def uploader(self, errors=None, token="tok"):
        """Build an uploader over a fake GitHub.

        Args: errors are queued FeedbackErrors; token is the token to use.
        Returns: The uploader; the fake is its .request.
        """
        return CaptureUploader(self.queue, FakeContents(errors), token, self.clock)

    def test_uploads_group_and_deletes_it(self):
        """Every file of a group is PUT with its remote path, then the group is removed.

        Args: None.
        Returns: None.
        """
        up = self.uploader()
        self.assertEqual(up.send_pending(), 1)
        self.assertEqual(sorted(up.request.paths()),
                         sorted(self.STEM + suffix for suffix in
                                ("-raw.png", "-processed.png", ".json")))
        call = up.request.calls[0]
        self.assertEqual(call[0], "PUT")
        self.assertEqual(call[2], "tok")
        self.assertEqual(call[3]["branch"], "main")
        self.assertIn("message", call[3])
        self.assertEqual(base64.b64decode(call[3]["content"]),
                         {"-raw.png": b"RAW", "-processed.png": b"PROC",
                          ".json": b"{}"}[call[1].split(self.STEM)[1]])
        self.assertEqual(self.queue.groups(), [])

    def test_422_already_exists_counts_as_done(self):
        """A file uploaded before a lost reply is treated as sent.

        Args: None.
        Returns: None.
        """
        up = self.uploader([FeedbackError("exists", 422)])
        self.assertEqual(up.send_pending(), 1)
        self.assertEqual(self.queue.groups(), [])

    def test_403_backs_off_and_retries_later(self):
        """A rate limit stops the pass, then the group is retried after the delay.

        Args: None.
        Returns: None.
        """
        up = self.uploader([FeedbackError("limit", 403)])
        self.assertEqual(up.send_pending(), 0)
        self.assertEqual(self.queue.groups(), [self.STEM])
        calls = len(up.request.calls)
        self.clock.now += 30
        self.assertEqual(up.send_pending(), 0)
        self.assertEqual(len(up.request.calls), calls)  # still waiting
        self.clock.now += 40
        self.assertEqual(up.send_pending(), 1)
        self.assertEqual(self.queue.groups(), [])

    def test_backoff_doubles_up_to_an_hour(self):
        """Repeated failures wait 1 minute, then 2, and never more than 1 hour.

        Args: None.
        Returns: None.
        """
        up = self.uploader([FeedbackError("busy", 409)] * 12)
        waits = []
        for _ in range(12):
            before = self.clock.now
            up.send_pending()
            waits.append(up.retry_at - before)
            self.clock.now = up.retry_at
        self.assertEqual(waits[:3], [60, 120, 240])
        self.assertEqual(max(waits), 3600)

    def test_network_error_backs_off(self):
        """An error with no HTTP status is retried later too.

        Args: None.
        Returns: None.
        """
        up = self.uploader([FeedbackError("cannot reach GitHub")])
        self.assertEqual(up.send_pending(), 0)
        self.assertFalse(up.stopped)
        self.assertEqual(self.queue.groups(), [self.STEM])

    def test_401_stops_for_the_run(self):
        """A rejected token ends uploading until restart.

        Args: None.
        Returns: None.
        """
        up = self.uploader([FeedbackError("rejected", 401)])
        self.assertEqual(up.send_pending(), 0)
        self.assertTrue(up.stopped)
        calls = len(up.request.calls)
        self.clock.now += DAY
        self.assertEqual(up.send_pending(), 0)
        self.assertEqual(len(up.request.calls), calls)

    def test_404_means_the_repository_is_gone_and_stops(self):
        """A missing repository (or no access) ends uploading for the run.

        Args: None.
        Returns: None.
        """
        up = self.uploader([FeedbackError("missing", 404)])
        self.assertEqual(up.send_pending(), 0)
        self.assertTrue(up.stopped)

    def test_no_token_sends_nothing(self):
        """Without a token nothing is requested and the queue is untouched.

        Args: None.
        Returns: None.
        """
        up = self.uploader(token="")
        self.assertEqual(up.send_pending(), 0)
        self.assertEqual(up.request.calls, [])
        self.assertEqual(self.queue.groups(), [self.STEM])

    def test_leftover_group_from_earlier_run_is_uploaded_once(self):
        """A new uploader over an old queue sends the leftover and then has nothing.

        Args: None.
        Returns: None.
        """
        again = CaptureUploader(CaptureQueue(self.folder, 1_000_000),
                                FakeContents(), "tok", self.clock)
        self.assertEqual(again.send_pending(), 1)
        self.assertEqual(again.send_pending(), 0)
        self.assertEqual(len(again.request.calls), 3)

    def test_token_and_repository_come_from_the_environment(self):
        """The environment variables choose the token and repository.

        Args: None.
        Returns: None.
        """
        old = {name: os.environ.get(name) for name in
               ("ESPI_CAPTURES_TOKEN", "ESPI_CAPTURES_REPO")}
        os.environ["ESPI_CAPTURES_TOKEN"] = "from-env"
        os.environ["ESPI_CAPTURES_REPO"] = "me/captures"
        try:
            self.assertEqual(capture_uploads.captures_token(), "from-env")
            self.assertEqual(capture_uploads.repository(), "me/captures")
            up = CaptureUploader(self.queue, FakeContents(), None, self.clock)
            self.assertEqual(up.token(), "from-env")
            up.send_pending()
            self.assertIn("/repos/me/captures/contents/", up.request.calls[0][1])
        finally:
            for name, value in old.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_default_repository(self):
        """Without settings the private captures repository is used.

        Args: None.
        Returns: None.
        """
        os.environ.pop("ESPI_CAPTURES_REPO", None)
        self.assertEqual(capture_uploads.repository(), capture_uploads.DEFAULT_REPO)
        self.assertEqual(capture_uploads.DEFAULT_REPO, "AlexWorland/espi-live-captures")


if __name__ == "__main__":
    unittest.main()
