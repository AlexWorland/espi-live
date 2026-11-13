"""Tests for the automatic log uploads (no network: a fake GitHub call)."""

import base64
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from espi_live import log_uploads
from espi_live.feedback import FeedbackError

LOG_A = "espi_20261113_142530_000001.log"
LOG_B = "espi_20261114_090000_000002.log"


class FakeContents:
    """A stand-in for the GitHub Contents API that records every call."""

    def __init__(self, remote=None, fail_put=None, fail_get=None):
        """Create it.

        Args: remote maps a repository path to the sha of a file already
            there; fail_put/fail_get are FeedbackErrors to raise instead.
        Returns: None.
        """
        self.remote = dict(remote or {})
        self.fail_put = fail_put
        self.fail_get = fail_get
        self.calls = []

    def __call__(self, method, path, token, payload=None):
        """Answer one call.

        Args: method, path, token and payload as feedback.github_request takes them.
        Returns: The decoded reply; raises FeedbackError like the real call.
        """
        self.calls.append((method, path, token, payload))
        file_path = path.split("/contents/", 1)[1].split("?", 1)[0]
        if method == "GET":
            if self.fail_get:
                raise self.fail_get
            if file_path.endswith(".log"):
                raise FeedbackError("too_large", 403)  # GitHub refuses big files here
            entries = [{"name": name.rsplit("/", 1)[1], "type": "file", "sha": sha}
                       for name, sha in self.remote.items()
                       if name.rsplit("/", 1)[0] == file_path]
            if not entries:
                raise FeedbackError("not found", 404)
            return entries + [{"name": "other.log", "type": "file", "sha": "zzz"}]
        if self.fail_put:
            raise self.fail_put
        self.remote[file_path] = f"sha-{len(self.calls)}"
        return {"content": {"sha": self.remote[file_path]}}

    def puts(self):
        """List the PUT calls.

        Args: None.
        Returns: (path, payload) pairs.
        """
        return [(path, payload) for method, path, _, payload in self.calls if method == "PUT"]


SERIAL = "21234567"


def serials():
    """Stand in for ErrorReporter.serials.

    Args: None.
    Returns: This run's camera serial numbers.
    """
    return (SERIAL,)


class LogUploadTest(unittest.TestCase):
    """The uploader against a temporary logs folder and a fake GitHub."""

    def setUp(self):
        """Create the logs folder.

        Args: None.
        Returns: None.
        """
        self.temp = tempfile.TemporaryDirectory()
        self.folder = self.temp.name
        self.addCleanup(self.temp.cleanup)

    def write_log(self, name, text, mode="w"):
        """Write (or append to) a run log.

        Args: name is the file name; text the content; mode "w" or "a".
        Returns: None.
        """
        with open(os.path.join(self.folder, name), mode, encoding="utf-8") as handle:
            handle.write(text)

    def uploader(self, fake, token="t"):
        """Build an uploader for the temporary folder.

        Args: fake is the FakeContents; token the token.
        Returns: A LogUploader.
        """
        return log_uploads.LogUploader(self.folder, serials, request=fake, token=token)

    def machine(self):
        """Read this folder's machine ID.

        Args: None.
        Returns: The ID.
        """
        return log_uploads.machine_id(self.folder)

    def test_remote_path_layout(self):
        """Run logs land under logs/<date>/<time>-<machine>.log.

        Args: None.
        Returns: None.
        """
        self.assertEqual(log_uploads.remote_path(LOG_A, "ab12cd34"),
                         "logs/2026-11-13/142530-ab12cd34.log")
        self.assertIsNone(log_uploads.remote_path("notes.log", "x"))

    def test_machine_id_is_stable(self):
        """The ID is created once and read back; an unwritable folder gives "unknown".

        Args: None.
        Returns: None.
        """
        first = self.machine()
        self.assertRegex(first, r"^[0-9a-f]{8}$")
        self.assertEqual(self.machine(), first)
        missing = os.path.join(self.folder, "missing", "deeper")
        self.assertEqual(log_uploads.machine_id(missing), "unknown")

    def test_uploads_new_logs_scrubbed(self):
        """Each new log is sent once, scrubbed, to main, without a sha.

        Args: None.
        Returns: None.
        """
        self.write_log(LOG_A, f"hello {SERIAL}\n")
        fake = FakeContents()
        self.assertEqual(self.uploader(fake).send_pending(), 1)
        [(path, payload)] = fake.puts()
        self.assertTrue(path.endswith(f"/contents/logs/2026-11-13/142530-{self.machine()}.log"))
        self.assertIn("/repos/AlexWorland/espi-live-logs/", path)
        self.assertEqual(base64.b64decode(payload["content"]).decode(), "hello <serial>\n")
        self.assertEqual(payload["branch"], "main")
        self.assertNotIn("sha", payload)
        with open(os.path.join(self.folder, log_uploads.RECORD_NAME), encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)[LOG_A]["size"], len(f"hello {SERIAL}\n"))

    def test_unchanged_log_is_not_sent_again(self):
        """A second pass with nothing new makes no calls.

        Args: None.
        Returns: None.
        """
        self.write_log(LOG_A, "one\n")
        fake = FakeContents()
        self.uploader(fake).send_pending()
        fake.calls.clear()
        self.assertEqual(self.uploader(fake).send_pending(), 0)
        self.assertEqual(fake.calls, [])

    def test_grown_log_is_uploaded_again(self):
        """A log that grew (the current run) is sent again, over the same file.

        Args: None.
        Returns: None.
        """
        self.write_log(LOG_A, "one\n")
        fake = FakeContents()
        uploader = self.uploader(fake)
        uploader.send_pending()
        sha = next(iter(fake.remote.values()))
        self.write_log(LOG_A, "two\n", mode="a")
        self.assertEqual(uploader.send_pending(), 1)
        path, payload = fake.puts()[-1]
        self.assertEqual(payload["sha"], sha)
        self.assertEqual(base64.b64decode(payload["content"]).decode(), "one\ntwo\n")

    def test_existing_remote_file_is_updated_with_its_sha(self):
        """With no local record, the remote file's sha is looked up and sent.

        Args: None.
        Returns: None.
        """
        self.write_log(LOG_A, "one\n")
        fake = FakeContents({log_uploads.remote_path(LOG_A, self.machine()): "abc"})
        self.uploader(fake).send_pending()
        self.assertEqual(fake.puts()[0][1]["sha"], "abc")

    def test_failure_stops_pass_and_records_nothing(self):
        """A rejected token or a lost network stops the pass and records nothing.

        Args: None.
        Returns: None.
        """
        self.write_log(LOG_A, "one\n")
        self.write_log(LOG_B, "two\n")
        for fake in (FakeContents(fail_put=FeedbackError("revoked", 401)),
                     FakeContents(fail_get=FeedbackError("offline"))):
            self.assertEqual(self.uploader(fake).send_pending(), 0)
            self.assertEqual(len([c for c in fake.calls if c[0] == "GET"]), 1)
            self.assertFalse(os.path.exists(os.path.join(self.folder, log_uploads.RECORD_NAME)))
        with mock.patch.object(log_uploads, "read_for_upload", side_effect=OSError("locked")):
            self.assertEqual(self.uploader(FakeContents()).send_pending(), 0)

    def test_tokenless_sends_nothing(self):
        """Without a token (source checkouts), nothing is called.

        Args: None.
        Returns: None.
        """
        self.write_log(LOG_A, "one\n")
        fake = FakeContents()
        self.assertEqual(self.uploader(fake, token="").send_pending(), 0)
        self.assertEqual(fake.calls, [])

    def test_at_most_ten_per_pass(self):
        """A large backlog is sent a few files per pass.

        Args: None.
        Returns: None.
        """
        for index in range(12):
            self.write_log(f"espi_20261113_1000{index:02d}_000000.log", "x\n")
        fake = FakeContents()
        self.assertEqual(self.uploader(fake).send_pending(), log_uploads.MAX_PER_PASS)
        self.assertEqual(len(fake.puts()), 10)

    def test_odd_files_and_a_bad_record_are_ignored(self):
        """Files that are not run logs are skipped; an unreadable record means "nothing sent".

        Args: None.
        Returns: None.
        """
        self.write_log("espi_notes.log", "not a run\n")
        with open(os.path.join(self.folder, log_uploads.RECORD_NAME), "w") as handle:
            handle.write("{broken")
        fake = FakeContents()
        self.assertEqual(self.uploader(fake).send_pending(), 0)
        self.assertEqual(fake.calls, [])

    def test_log_removed_during_pass_is_skipped(self):
        """A log pruned by another copy between listing and reading is skipped.

        Args: None.
        Returns: None.
        """
        self.write_log(LOG_A, "one\n")
        fake = FakeContents()
        with mock.patch.object(log_uploads.os.path, "getsize", side_effect=FileNotFoundError):
            self.assertEqual(self.uploader(fake).send_pending(), 0)
        self.assertEqual(fake.calls, [])

    def test_serials_seen_before_are_scrubbed_later(self):
        """A serial seen in an earlier run is hidden even when that camera is absent.

        Args: None.
        Returns: None.
        """
        self.write_log(LOG_A, "one\n")
        self.uploader(FakeContents()).send_pending()
        self.write_log(LOG_B, f"Serial {SERIAL} and bare {SERIAL}\n")
        fake = FakeContents()
        log_uploads.LogUploader(self.folder, lambda: (), request=fake, token="t").send_pending()
        text = base64.b64decode(fake.puts()[0][1]["content"]).decode()
        self.assertNotIn(SERIAL, text)
        with open(os.path.join(self.folder, log_uploads.SERIALS_NAME), "w") as handle:
            handle.write("{broken")
        self.assertEqual(log_uploads.known_serials(self.folder), [])
        with mock.patch.object(log_uploads, "write_json", side_effect=OSError("read-only")):
            self.assertEqual(log_uploads.remember_serials(self.folder, ["123", ""]), ["123"])

    def test_camera_labels_and_prefixes_are_scrubbed_without_knowing_the_serial(self):
        """Labels "Model (serial)" and "[serial] " line prefixes never leave the PC.

        Args: None.
        Returns: None.
        """
        scrub = log_uploads.make_log_scrubber(())
        text = scrub("10:00:00 INFO [MainThread] [99887766] Displaying Blackfly S (99887766)\n"
                     "exposure 8000 at 10:00:01 (frame 1234)\n")
        self.assertNotIn("99887766", text)
        self.assertIn("Blackfly S (<serial>)", text)
        self.assertIn("[MainThread] [<serial>] Displaying", text)
        self.assertIn("exposure 8000 at 10:00:01 (frame 1234)", text)

    def test_scrubbing_runs_in_line_chunks(self):
        """Large text is scrubbed a chunk of whole lines at a time, giving other threads the GIL.

        Args: None.
        Returns: None.
        """
        calls = []

        def scrub(text):
            """Record each chunk.

            Args: text is one chunk.
            Returns: The chunk upper-cased.
            """
            calls.append(text)
            return text.upper()

        text = "".join(f"line {index}\n" for index in range(3000))
        self.assertEqual(log_uploads.scrub_in_chunks(text, scrub, chunk_chars=1000), text.upper())
        self.assertGreater(len(calls), 10)
        self.assertTrue(all(chunk.endswith("\n") for chunk in calls))
        self.assertEqual(log_uploads.scrub_in_chunks("", scrub), "")

    def test_cap_keeps_head_and_tail_and_survives_bad_bytes(self):
        """A long log keeps its start and its end; bad bytes never raise.

        Args: None.
        Returns: None.
        """
        path = os.path.join(self.folder, LOG_A)
        with open(path, "wb") as handle:
            handle.write(b"HEAD\xff" + "é".encode() * 50 + b"TAIL")
        text, size = log_uploads.read_for_upload(path, max_bytes=40, head_bytes=6)
        self.assertEqual(size, 109)
        self.assertTrue(text.startswith("HEAD"))
        self.assertTrue(text.endswith("TAIL"))
        self.assertIn("bytes left out", text)
        whole, _ = log_uploads.read_for_upload(path)
        self.assertNotIn("left out", whole)

    def test_token_and_repository_sources(self):
        """The environment overrides the token and the repository.

        Args: None.
        Returns: None.
        """
        with mock.patch.dict(os.environ, {log_uploads.TOKEN_ENV: "env-token",
                                          log_uploads.REPO_ENV: "me/logs"}):
            self.assertEqual(log_uploads.logs_token(), "env-token")
            self.assertEqual(log_uploads.repository(), "me/logs")
        with mock.patch.dict(os.environ, {log_uploads.REPO_ENV: ""}):
            self.assertEqual(log_uploads.repository(), log_uploads.DEFAULT_REPO)
        with mock.patch.object(log_uploads, "logs_token", return_value="baked"):
            uploader = log_uploads.LogUploader(self.folder, serials)
            self.assertEqual(uploader.token(), "baked")


class FakeUploader:
    """Counts passes; the first one fails."""

    def __init__(self):
        """Create it.

        Args: None.
        Returns: None.
        """
        self.passes = 0
        self.ran = threading.Event()

    def send_pending(self):
        """Count a pass; fail on the first.

        Args: None.
        Returns: 0.
        """
        self.passes += 1
        if self.passes == 1:
            raise RuntimeError("boom")
        self.ran.set()
        return 0


class LogUploadWorkerTest(unittest.TestCase):
    """The background thread."""

    def test_worker_runs_at_start_and_on_kick_and_survives_errors(self):
        """It runs at start, again when kicked after an error, and stops.

        Args: None.
        Returns: None.
        """
        uploader = FakeUploader()
        worker = log_uploads.LogUploadWorker(uploader, interval_s=60)
        with self.assertLogs("espi_live.log_uploads", "DEBUG"):
            worker.start()
            for _ in range(200):
                if uploader.passes:
                    break
                threading.Event().wait(0.01)
            worker.kick()
            self.assertTrue(uploader.ran.wait(2))
        worker.stop()
        worker.thread.join(2)
        self.assertFalse(worker.thread.is_alive())
        self.assertEqual(worker.thread.name, "log-uploads")
        self.assertGreaterEqual(uploader.passes, 2)


if __name__ == "__main__":
    unittest.main()
