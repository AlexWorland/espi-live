"""Network-free checks for the downloaded source copy: download, unpacking,
comparison with the original, patches and the background worker.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import io
import os
import shutil
import subprocess
import tempfile
import time
import unittest
import urllib.error
import zipfile
from unittest import mock

from espi_live import paths, source_code, updater
from source_fixtures import (COMMIT, FILES, OTHER_COMMIT, FakeResponse, make_copy,
                             prebuilt, write, zip_bytes)


def http_error(code):
    """Build an HTTPError with a status code.

    Args: code is the HTTP status.
    Returns: The urllib.error.HTTPError.
    """
    error = urllib.error.HTTPError("https://x.invalid", code, "msg", {}, io.BytesIO())
    error.close()
    return error


class TempCase(unittest.TestCase):
    """A test with its own temporary program folder."""

    def setUp(self):
        """Create the temporary folder.

        Args: None.
        Returns: None.
        """
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)


class LocationTests(TempCase):
    """Where the source lives and what a copy records."""

    def test_checkout_is_its_own_source(self):
        """From source the checkout root is the source folder.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(updater, "install_target", return_value=None):
            self.assertTrue(source_code.is_checkout())
            self.assertEqual(source_code.source_folder(), paths.PROJECT_ROOT)

    def test_prebuilt_copy_keeps_source_beside_the_program(self):
        """A prebuilt copy uses espi-live-source in its folder.

        Args: None.
        Returns: None.
        """
        with prebuilt(self.root):
            self.assertFalse(source_code.is_checkout())
            self.assertEqual(source_code.source_folder(),
                             os.path.join(self.root, "espi-live-source"))
            self.assertEqual(source_code.wanted_commit(), COMMIT)

    def test_state_and_sent_records_tolerate_damage(self):
        """Missing, damaged or malformed bookkeeping reads as empty.

        Args: None.
        Returns: None.
        """
        self.assertIsNone(source_code.read_state(self.root))
        self.assertEqual(source_code.sent_records(self.root), [])
        os.makedirs(os.path.join(self.root, ".espi"))
        write(self.root, ".espi/source.json", "{not json")
        self.assertIsNone(source_code.read_state(self.root))
        write(self.root, ".espi/source.json", '{"commit": ""}')
        self.assertIsNone(source_code.read_state(self.root))
        write(self.root, ".espi/sent.json", '{"url": "x"}')
        self.assertEqual(source_code.sent_records(self.root), [])
        write(self.root, ".espi/sent.json", '[{"url": "u", "patch": "p"}, {"url": "only"}, 3]')
        self.assertEqual(source_code.sent_records(self.root), [{"url": "u", "patch": "p"}])

    def test_record_sent_appends(self):
        """Each send is added to the list with a time.

        Args: None.
        Returns: None.
        """
        folder = make_copy(self.root)
        source_code.record_sent(folder, "https://x/1", "aa")
        source_code.record_sent(folder, "https://x/2", "bb")
        records = source_code.sent_records(folder)
        self.assertEqual([(r["url"], r["patch"]) for r in records],
                         [("https://x/1", "aa"), ("https://x/2", "bb")])
        self.assertTrue(records[0]["time"])


class ArchiveTests(TempCase):
    """Reading GitHub's source zip."""

    def open_zip(self, data):
        """Open zip bytes for reading.

        Args: data is the zip file's bytes.
        Returns: An open zipfile.ZipFile.
        """
        bundle = zipfile.ZipFile(io.BytesIO(data))
        self.addCleanup(bundle.close)
        return bundle

    def test_members_drop_the_top_folder_folders_and_links(self):
        """Paths lose GitHub's top folder; folders and symlinks are skipped.

        Args: None.
        Returns: None.
        """
        link = zipfile.ZipInfo("owner-repo-0123456/link")
        link.external_attr = 0o120777 << 16
        members = source_code.archive_members(self.open_zip(zip_bytes(extra=[(link, b"x")])))
        self.assertEqual(sorted(members), sorted(FILES))

    def test_members_refuse_names_that_escape(self):
        """Absolute, parent and drive-letter names are refused.

        Args: None.
        Returns: None.
        """
        for name in ("top/../evil", "top//abs", "top/c:/evil", "top/a\\b", "loose"):
            with self.subTest(name=name), self.assertRaisesRegex(
                    source_code.SourceError, "unsafe name"):
                source_code.archive_members(self.open_zip(zip_bytes({}, extra=[(name, b"x")])))

    def test_extract_writes_files_and_checks_the_commit(self):
        """Files are unpacked; a zip made from another commit is refused.

        Args: None.
        Returns: None.
        """
        archive = os.path.join(self.root, "a.zip")
        with open(archive, "wb") as handle:
            handle.write(zip_bytes())
        target = os.path.join(self.root, "out")
        source_code.extract(archive, target, COMMIT)
        with open(os.path.join(target, "scripts", "windows", "start.bat"), "rb") as handle:
            self.assertEqual(handle.read(), FILES["scripts/windows/start.bat"])
        with self.assertRaisesRegex(source_code.SourceError, "source of 0123456, not fedcba9"):
            source_code.extract(archive, target, OTHER_COMMIT)
        with open(archive, "wb") as handle:
            handle.write(zip_bytes(commit=OTHER_COMMIT))
        with self.assertRaisesRegex(source_code.SourceError, "source of fedcba9, not 0123456"):
            source_code.extract(archive, target, COMMIT)

    def test_extract_accepts_a_zip_without_comment(self):
        """A zip without a commit comment is trusted to be the one asked for.

        Args: None.
        Returns: None.
        """
        archive = os.path.join(self.root, "a.zip")
        with open(archive, "wb") as handle:
            handle.write(zip_bytes({"a.txt": b"a\n"}, commit=""))
        source_code.extract(archive, os.path.join(self.root, "out"), COMMIT)
        self.assertTrue(os.path.exists(os.path.join(self.root, "out", "a.txt")))

    def test_extract_reports_a_damaged_zip(self):
        """A file that is not a zip raises SourceError.

        Args: None.
        Returns: None.
        """
        archive = os.path.join(self.root, "a.zip")
        with open(archive, "wb") as handle:
            handle.write(b"not a zip")
        with self.assertRaisesRegex(source_code.SourceError, "cannot unpack"):
            source_code.extract(archive, os.path.join(self.root, "out"), COMMIT)


class FetchTests(TempCase):
    """Downloading the zip."""

    def test_fetch_writes_the_download(self):
        """The zipball URL of the commit is fetched with the read token.

        Args: None.
        Returns: None.
        """
        path = os.path.join(self.root, "a.zip")
        with mock.patch.object(updater, "update_token", return_value="tok"), \
                mock.patch.object(updater, "open_url",
                                  return_value=FakeResponse(b"zip data")) as opened:
            source_code.fetch("owner/repo", COMMIT, path)
        opened.assert_called_once_with(
            f"https://api.github.com/repos/owner/repo/zipball/{COMMIT}", "tok",
            "application/vnd.github+json")
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(), b"zip data")

    def test_fetch_stops_an_oversized_download(self):
        """A download over the size limit is abandoned.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(source_code, "MAX_DOWNLOAD_BYTES", 4), \
                mock.patch.object(updater, "open_url", return_value=FakeResponse(b"12345")), \
                self.assertRaisesRegex(source_code.SourceError, "unexpectedly large"):
            source_code.fetch("owner/repo", COMMIT, os.path.join(self.root, "a.zip"))

    def test_fetch_explains_http_and_network_errors(self):
        """404, other HTTP codes and network failures get plain messages.

        Args: None.
        Returns: None.
        """
        cases = ((http_error(404), "no source for commit 0123456"),
                 (http_error(500), "HTTP 500"),
                 (urllib.error.URLError("offline"), "cannot reach GitHub: offline"))
        for error, message in cases:
            with self.subTest(message=message), \
                    mock.patch.object(updater, "open_url", side_effect=error), \
                    self.assertRaisesRegex(source_code.SourceError, message):
                source_code.fetch("owner/repo", COMMIT, os.path.join(self.root, "a.zip"))


class DownloadTests(TempCase):
    """download_source and replace_folder."""

    def fake_fetch(self, commit=COMMIT, files=None):
        """Patch fetch to write a source zip instead of downloading.

        Args: commit is the zip's comment; files are its contents.
        Returns: The active patcher's mock.
        """
        def fetch(repo, wanted, path):
            """Write the zip.

            Args: repo and wanted are ignored; path is the file to write.
            Returns: None.
            """
            with open(path, "wb") as handle:
                handle.write(zip_bytes(files, commit))

        patcher = mock.patch.object(source_code, "fetch", side_effect=fetch)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_download_creates_the_copy_and_cleans_up(self):
        """A download leaves the files, the original zip and the state, and
        removes leftovers of an interrupted download.

        Args: None.
        Returns: None.
        """
        self.fake_fetch()
        os.makedirs(os.path.join(self.root, ".espi-source-999", "junk"))
        with prebuilt(self.root):
            folder = source_code.download_source()
        self.assertEqual(sorted(os.listdir(self.root)), ["espi-live-source"])
        self.assertEqual(source_code.read_state(folder)["commit"], COMMIT)
        self.assertEqual(source_code.read_state(folder)["build"], 7)
        self.assertTrue(os.path.exists(os.path.join(folder, ".espi", "download.zip")))
        self.assertEqual(source_code.compare(folder), [])

    def test_download_replaces_an_older_copy(self):
        """An older copy is swapped out for the new commit's.

        Args: None.
        Returns: None.
        """
        old = make_copy(self.root, {"old.txt": b"old\n"}, OTHER_COMMIT)
        self.fake_fetch()
        with prebuilt(self.root):
            source_code.download_source(old)
        self.assertFalse(os.path.exists(os.path.join(old, "old.txt")))
        self.assertEqual(source_code.read_state(old)["commit"], COMMIT)

    def test_download_needs_build_facts(self):
        """A build without commit or repository cannot download.

        Args: None.
        Returns: None.
        """
        with prebuilt(self.root, commit=""), \
                self.assertRaisesRegex(source_code.SourceError, "does not record"):
            source_code.download_source()

    def test_download_reports_an_unwritable_folder(self):
        """A program folder that cannot be written raises SourceError.

        Args: None.
        Returns: None.
        """
        with prebuilt(self.root), \
                mock.patch.object(source_code.os, "makedirs", side_effect=PermissionError("ro")), \
                self.assertRaisesRegex(source_code.SourceError, "cannot save the source code"):
            source_code.download_source()

    def test_failed_swap_restores_the_old_copy(self):
        """If the new copy cannot be moved in, the old one is put back.

        Args: None.
        Returns: None.
        """
        old = make_copy(self.root, {"old.txt": b"old\n"}, OTHER_COMMIT)
        new = os.path.join(self.root, "new")
        os.makedirs(new)
        staging = os.path.join(self.root, "staging")
        os.makedirs(staging)
        real_replace = os.replace

        def replace(source, destination):
            """Fail only the move of the new copy.

            Args: source and destination are os.replace's arguments.
            Returns: None.
            """
            if source == new:
                raise PermissionError("in use")
            real_replace(source, destination)

        with mock.patch.object(source_code.os, "replace", side_effect=replace), \
                self.assertRaisesRegex(source_code.SourceError, "close programs using it"):
            source_code.replace_folder(new, old, staging)
        self.assertTrue(os.path.exists(os.path.join(old, "old.txt")))

    def test_failed_swap_without_an_old_copy(self):
        """With no old copy, a failed move just raises.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(source_code.os, "replace", side_effect=PermissionError("x")), \
                self.assertRaises(source_code.SourceError):
            source_code.replace_folder(os.path.join(self.root, "new"),
                                       os.path.join(self.root, "dest"), self.root)


class CompareTests(TempCase):
    """Finding changes and writing patches."""

    def test_ignored_files(self):
        """Bookkeeping, generated, cache and output files are never changes.

        Args: None.
        Returns: None.
        """
        for name in (".espi/source.json", "AGENTS.override.md", "CHANGE_REQUEST.md",
                     "src/espi_live/__pycache__/x.pyc", "a.pyc", "output/frames/a.png",
                     ".git/HEAD", "docs/.DS_Store"):
            self.assertTrue(source_code.is_ignored(name), name)
        for name in ("AGENTS.md", "src/espi_live/engine.py", "docs/CHANGE_REQUEST.md.txt"):
            self.assertFalse(source_code.is_ignored(name), name)

    def test_working_files_skip_ignored_folders_and_links(self):
        """Ignored folders are not walked and symlinks are left out.

        Args: None.
        Returns: None.
        """
        write(self.root, "a.txt", "a")
        write(self.root, "sub/b.txt", "b")
        write(self.root, "output/c.txt", "c")
        write(self.root, "AGENTS.override.md", "guide")
        try:
            os.symlink(os.path.join(self.root, "a.txt"), os.path.join(self.root, "link.txt"))
        except (OSError, NotImplementedError):
            pass  # no symlinks on this system
        self.assertEqual(sorted(source_code.working_files(self.root)), ["a.txt", "sub/b.txt"])

    def test_text_detection_and_line_endings(self):
        """Text is UTF-8 without NUL; line endings follow the original.

        Args: None.
        Returns: None.
        """
        self.assertTrue(source_code.is_text(b"plain\n"))
        self.assertFalse(source_code.is_text(b"a\0b"))
        self.assertFalse(source_code.is_text(b"\xff\xfe"))
        match = source_code.match_line_endings
        self.assertEqual(match(b"a\nb\n", b"a\r\nc\r\n"), b"a\nc\n")
        self.assertEqual(match(b"a\r\nb\r\n", b"a\nc\r\n"), b"a\r\nc\r\n")
        self.assertEqual(match(b"a\r\nb\n", b"x\ny\r\n"), b"x\ny\r\n")

    def test_compare_finds_added_modified_and_deleted_files(self):
        """Edits, new files and deletions are listed in path order; a pure
        line-ending rewrite is not a change.

        Args: None.
        Returns: None.
        """
        folder = make_copy(self.root)
        write(folder, "README.md", "# ESPI live\n\nA live viewer.\n")
        write(folder, "scripts/windows/start.bat", "@echo off\necho start\n")
        write(folder, "docs/new.md", "new")
        os.unlink(os.path.join(folder, "src", "espi_live", "engine.py"))
        write(folder, "CHANGE_REQUEST.md", "Title")
        found = [(path, source_code.change_kind(old, new))
                 for path, old, new in source_code.compare(folder)]
        self.assertEqual(found, [("README.md", "modified"), ("docs/new.md", "added"),
                                 ("src/espi_live/engine.py", "deleted")])

    def test_compare_reports_unreadable_files(self):
        """A missing original or an unreadable file raises SourceError.

        Args: None.
        Returns: None.
        """
        with self.assertRaisesRegex(source_code.SourceError, "original download"):
            source_code.compare(self.root)
        folder = make_copy(self.root)
        real_open = open

        def failing_open(path, *args, **kwargs):
            """Fail to open the README only.

            Args: path and the rest are open()'s arguments.
            Returns: The opened file.
            """
            if str(path).endswith("README.md"):
                raise PermissionError("locked")
            return real_open(path, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=failing_open), \
                self.assertRaisesRegex(source_code.SourceError, "cannot read README.md"):
            source_code.compare(folder)

    def test_split_lines_matches_git(self):
        """Lines split only at LF; a missing final line break is kept.

        Args: None.
        Returns: None.
        """
        self.assertEqual(source_code.split_lines("a\fb\nc"), ["a\fb\n", "c"])
        self.assertEqual(source_code.split_lines("a\n"), ["a\n"])
        self.assertEqual(source_code.split_lines(""), [])

    def test_make_patch_writes_git_headers_and_skips_binaries(self):
        """Each kind of change gets git's headers; binary files are left out.

        Args: None.
        Returns: None.
        """
        changes = [("a.txt", None, b"new"), ("b.txt", b"gone\n", None),
                   ("c.txt", b"one\n", b"two\n"), ("d.png", b"\0", b"\0\1")]
        patch, skipped = source_code.make_patch(changes)
        self.assertEqual(skipped, ["d.png"])
        self.assertEqual(patch, (
            "diff --git a/a.txt b/a.txt\nnew file mode 100644\n--- /dev/null\n+++ b/a.txt\n"
            "@@ -0,0 +1 @@\n+new\n\\ No newline at end of file\n"
            "diff --git a/b.txt b/b.txt\ndeleted file mode 100644\n--- a/b.txt\n+++ /dev/null\n"
            "@@ -1 +0,0 @@\n-gone\n"
            "diff --git a/c.txt b/c.txt\n--- a/c.txt\n+++ b/c.txt\n@@ -1 +1 @@\n-one\n+two\n"))
        self.assertEqual(len(source_code.patch_digest(patch)), 64)

    def test_patch_applies_with_git(self):
        """git apply accepts the patch and reproduces the edited files.

        Args: None.
        Returns: None.
        """
        git = shutil.which("git")
        if not git:
            self.skipTest("git is not installed")
        folder = make_copy(self.root)
        write(folder, "README.md", "# ESPI live\n\nA viewer\x0cwith a form feed.")
        write(folder, "scripts/windows/start.bat", "@echo off\r\necho go\r\n")
        write(folder, "new folder/new file.md", "hello\n")
        os.unlink(os.path.join(folder, "src", "espi_live", "engine.py"))
        patch, _ = source_code.make_patch(source_code.compare(folder))
        clean = os.path.join(self.root, "clean")
        with zipfile.ZipFile(source_code.bookkeeping(folder, "download.zip")) as bundle:
            for relative, info in source_code.archive_members(bundle).items():
                write(clean, relative, bundle.read(info))
        patch_file = os.path.join(self.root, "change.patch")
        with open(patch_file, "w", encoding="utf-8", newline="") as handle:
            handle.write(patch)
        subprocess.run([git, "init", "-q"], cwd=clean, check=True)
        subprocess.run([git, "apply", patch_file], cwd=clean, check=True,
                       capture_output=True)
        for relative in ("README.md", "scripts/windows/start.bat", "new folder/new file.md"):
            with open(os.path.join(clean, *relative.split("/")), "rb") as a, \
                    open(os.path.join(folder, *relative.split("/")), "rb") as b:
                self.assertEqual(a.read(), b.read(), relative)
        self.assertFalse(os.path.exists(os.path.join(clean, "src", "espi_live", "engine.py")))

    def test_unsent_changes(self):
        """Changes count as unsent until their patch is recorded as sent;
        binary changes always count.

        Args: None.
        Returns: None.
        """
        folder = make_copy(self.root)
        self.assertFalse(source_code.has_unsent_changes(folder))
        write(folder, "README.md", "changed\n")
        self.assertTrue(source_code.has_unsent_changes(folder))
        patch, _ = source_code.make_patch(source_code.compare(folder))
        source_code.record_sent(folder, "https://x/1", source_code.patch_digest(patch))
        self.assertFalse(source_code.has_unsent_changes(folder))
        write(folder, "docs/logo.png", b"\0changed")
        self.assertTrue(source_code.has_unsent_changes(folder))


class CheckTests(TempCase):
    """The startup decision and prepare()."""

    def test_checkout_is_ready(self):
        """A source checkout never downloads.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(updater, "install_target", return_value=None):
            self.assertEqual(source_code.check(), ("ready", ""))
            self.assertEqual(source_code.prepare(), (paths.PROJECT_ROOT, ""))

    def test_build_without_facts_is_unavailable(self):
        """A prebuilt copy without a commit cannot have its source.

        Args: None.
        Returns: None.
        """
        with prebuilt(self.root, repo=""):
            self.assertEqual(source_code.check()[0], "unavailable")
            with self.assertRaisesRegex(source_code.SourceError, "does not record"):
                source_code.prepare()

    def test_foreign_folder_is_left_alone(self):
        """A folder of that name that ESPI live did not make is not replaced.

        Args: None.
        Returns: None.
        """
        os.makedirs(os.path.join(self.root, "espi-live-source"))
        with prebuilt(self.root):
            action, detail = source_code.check()
        self.assertEqual(action, "unavailable")
        self.assertIn("rename it", detail)

    def test_missing_copy_is_downloaded(self):
        """With no copy, prepare downloads one and says so.

        Args: None.
        Returns: None.
        """
        with prebuilt(self.root), mock.patch.object(source_code, "download_source") as download:
            self.assertEqual(source_code.check()[0], "download")
            folder, note = source_code.prepare()
        download.assert_called_once_with(folder)
        self.assertIn("Downloaded the source code of commit 0123456", note)

    def test_current_copy_is_ready(self):
        """A copy of this build's commit is used as it is.

        Args: None.
        Returns: None.
        """
        make_copy(self.root)
        with prebuilt(self.root):
            self.assertEqual(source_code.check(), ("ready", ""))

    def test_older_copy_is_replaced_unless_it_holds_unsent_changes(self):
        """After an update an unchanged or damaged copy is replaced, a copy
        with unsent changes is kept.

        Args: None.
        Returns: None.
        """
        folder = make_copy(self.root, commit=OTHER_COMMIT)
        with prebuilt(self.root):
            self.assertEqual(source_code.check()[0], "download")
            write(folder, "README.md", "edited\n")
            action, note = source_code.check()
            self.assertEqual(action, "ready")
            self.assertIn("Kept the source code of commit fedcba9", note)
            os.unlink(source_code.bookkeeping(folder, "download.zip"))
            self.assertEqual(source_code.check()[0], "download")


class WorkerTests(unittest.TestCase):
    """SourceWorker's thread and events."""

    def run_worker(self, **patch):
        """Run one worker job with prepare() patched.

        Args: patch holds the mock.patch keyword (return_value or side_effect).
        Returns: The posted event.
        """
        worker = source_code.SourceWorker()
        with mock.patch.object(source_code, "prepare", **patch):
            self.assertTrue(worker.start())
            event = worker.events.get(timeout=5)
        deadline = time.monotonic() + 5
        while worker.busy and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(worker.busy)
        return event

    def test_ready_and_failed_events(self):
        """Success posts ("ready", (folder, note)); failure posts its message.

        Args: None.
        Returns: None.
        """
        self.assertEqual(self.run_worker(return_value=("/f", "")), ("ready", ("/f", "")))
        self.assertEqual(self.run_worker(side_effect=source_code.SourceError("offline")),
                         ("failed", "offline"))

    def test_busy_worker_refuses_a_second_start(self):
        """start() returns False while a job runs.

        Args: None.
        Returns: None.
        """
        worker = source_code.SourceWorker()
        worker.busy = True
        self.assertFalse(worker.start())


if __name__ == "__main__":
    unittest.main()
