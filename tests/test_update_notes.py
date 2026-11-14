"""Network-free checks for the release notes shown after an update.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import io
import json
import os
import tempfile
import types
import unittest
from unittest import mock

from espi_live import updater



def ci_block(build, kind="Release", source="`main`"):
    """Build the block CI adds to a release's notes.

    Args: build is the build number; kind and source fill its first line.
    Returns: The Markdown text.
    """
    return (f"{kind} from {source}, commit `abc1234`, build {build}. "
            "Build log: https://x.invalid/run\n\n"
            "- **Windows** (`espi-live-windows-x64.zip`) includes PySpin.\n"
            "- Installed copies update themselves.\n\n\n")


GENERATED = ("<!-- Release notes generated using configuration in .github/release.yml -->\n\n"
             "## What's Changed\n### Features\n* Search the user guide by @someone in "
             "https://x.invalid/pull/48\n\n\n**Full Changelog**: https://x.invalid/compare")


def note(build, body="Notes", url="https://x.invalid/r"):
    """Build a ReleaseNote.

    Args: build is the build number; body and url are its text and page.
    Returns: An updater.ReleaseNote.
    """
    return updater.ReleaseNote(build, f"Build {build}", body, url)


class ReleaseParsingTests(unittest.TestCase):
    """Tags, boilerplate and choosing which releases an update covers."""

    def test_build_number_comes_from_the_ci_line_or_an_old_tag(self):
        """The CI line names the build; old releases were tagged build-N.

        Args: None.
        Returns: None.
        """
        body = ("Typed\r\n\r\n" + ci_block(31, source="tag `v0.10.0`")).replace("\n", "\r\n")
        self.assertEqual(updater.release_build({"tag_name": "v0.10.0", "body": body}), 31)
        self.assertEqual(updater.release_build({"tag_name": "build-42", "body": None}), 42)
        self.assertIsNone(updater.release_build({"tag_name": "v1.0", "body": "Notes"}))
        self.assertIsNone(updater.release_build({}))

    def test_ci_block_is_dropped_and_generated_notes_kept(self):
        """Typed and generated notes stay; the CI block and comments go.

        Args: None.
        Returns: None.
        """
        summary = updater.release_summary("Fixed X\n\n" + ci_block(31) + GENERATED)
        self.assertTrue(summary.startswith("Fixed X\n\n## What's Changed"))
        self.assertIn("**Full Changelog**", summary)
        self.assertNotIn("Build log", summary)
        self.assertNotIn("<!--", summary)
        pre = ci_block(5, kind="Pre-release (not offered to installed copies)")
        self.assertEqual(updater.release_summary(pre), "")
        old = "Release from `main`, commit `abc1234`. Build log: https://x\n\n- **Windows**"
        self.assertEqual(updater.release_summary(old), "")
        self.assertEqual(updater.release_summary("Only notes"), "Only notes")

    def test_collect_keeps_full_releases_in_range_newest_first(self):
        """Builds after this one up to the new one, without drafts or pre-releases.

        Args: None.
        Returns: None.
        """
        releases = [
            {"tag_name": "build-5", "name": "Five", "body": "5", "html_url": "u5"},
            {"tag_name": "v0.2.0", "name": "", "body": ci_block(7)},
            {"tag_name": "v0.1.1-pre.6", "prerelease": True, "body": ci_block(6)},
            {"tag_name": "build-8", "draft": True},
            {"tag_name": "build-3", "name": "Three"},
            {"tag_name": "build-9"},
            {"tag_name": "nightly"},
            {"body": "no tag", "name": "Lost"},
        ]
        notes = updater.collect_notes(releases, 3, 7)
        self.assertEqual(notes, (updater.ReleaseNote(7, "v0.2.0", ci_block(7), ""),
                                 updater.ReleaseNote(5, "Five", "5", "u5")))


class FetchNotesTests(unittest.TestCase):
    """fetch_notes falls back to the newest release."""

    LATEST = {"tag_name": "v0.9.0", "name": "", "body": "Latest"}

    def fetch(self, payload):
        """Run fetch_notes with open_url serving one JSON payload.

        Args: payload is the release list to return.
        Returns: The tuple of ReleaseNote.
        """
        body = json.dumps(payload).encode()
        with mock.patch.object(updater, "open_url",
                               side_effect=lambda *args: io.BytesIO(body)):
            return updater.fetch_notes("o/r", "", self.LATEST, 3, 9)

    def test_no_matching_release_uses_the_newest(self):
        """A list without usable releases falls back to the newest release.

        Args: None.
        Returns: None.
        """
        self.assertEqual(self.fetch([]), (updater.ReleaseNote(9, "v0.9.0", "Latest", ""),))

    def test_unexpected_payload_uses_the_newest(self):
        """A list of the wrong shape is ignored rather than failing the check.

        Args: None.
        Returns: None.
        """
        self.assertEqual(self.fetch({"message": "odd"})[0].body, "Latest")


class WhatsNewTests(unittest.TestCase):
    """Rendering the notes."""

    def test_markdown_lists_each_release_with_its_link(self):
        """Each release gets a heading, its notes and a link to its page.

        Args: None.
        Returns: None.
        """
        text = updater.WhatsNew(1, 3, (note(3, "Big fix\n\n" + ci_block(3)),
                                       updater.ReleaseNote(2, "", "", ""))).markdown()
        self.assertIn("# Build 3\n\nBig fix\n\n[Release page](https://x.invalid/r)", text)
        self.assertIn("# Build 2\n\n*No notes were written for this release.*", text)
        self.assertNotIn("Build log", text)
        self.assertNotIn("[Release page]()", text)

    def test_markdown_without_notes_points_to_releases(self):
        """With nothing captured the dialog still says something useful.

        Args: None.
        Returns: None.
        """
        self.assertIn("Releases page", updater.WhatsNew(1, 2, ()).markdown())


class NotesFileTests(unittest.TestCase):
    """Saving notes with a download, moving them on install, reading them once."""

    def setUp(self):
        """Create an install folder with a running build 3 and a staged build 4.

        The staged build is a program folder: the executable and its
        _internal-4 libraries.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.target = os.path.join(self.tmp.name, "espi")
        self.staged = os.path.join(self.tmp.name, "stage", "espi")
        os.makedirs(os.path.join(self.staged, updater.CONTENTS_PREFIX + "4"))
        for path in (self.target, os.path.join(self.staged, "espi")):
            open(path, "w").close()
        self.info = updater.UpdateInfo(4, "c" * 10, "a.tar.gz", "u", "0" * 64, 1,
                                       notes=(note(4, "Shiny"),))
        self.whats_new_path = os.path.join(self.tmp.name, updater.WHATS_NEW_FILE)
        build = mock.patch.object(updater, "_build_info",
                                  types.SimpleNamespace(BUILD=3))
        build.start()
        self.addCleanup(build.stop)

    def set_build(self, number):
        """Pretend to be another build from now on.

        Args: number is the build number.
        Returns: None.
        """
        updater._build_info.BUILD = number

    def test_notes_survive_install_and_show_once(self):
        """Notes saved with the download are read once by the new build.

        Args: None.
        Returns: None.
        """
        updater.save_notes(self.staged + updater.NOTES_SUFFIX, self.info)
        updater.install_update(self.staged, target=self.target)
        self.assertTrue(os.path.exists(self.whats_new_path))
        self.set_build(4)
        whats_new = updater.take_whats_new(self.target)
        self.assertEqual(whats_new, updater.WhatsNew(3, 4, (note(4, "Shiny"),)))
        self.assertFalse(os.path.exists(self.whats_new_path))
        self.assertIsNone(updater.take_whats_new(self.target))

    def test_install_without_saved_notes_still_succeeds(self):
        """A missing notes file never blocks the swap.

        Args: None.
        Returns: None.
        """
        updater.install_update(self.staged, target=self.target)
        self.assertFalse(os.path.exists(self.whats_new_path))

    def test_unwritable_notes_are_skipped(self):
        """A failed notes write is ignored.

        Args: None.
        Returns: None.
        """
        updater.save_notes(os.path.join(self.tmp.name, "missing", "n.json"), self.info)
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "missing")))

    def write(self, text):
        """Write the what's-new file directly.

        Args: text is the file content.
        Returns: None.
        """
        with open(self.whats_new_path, "w", encoding="utf-8") as handle:
            handle.write(text)

    def test_notes_for_another_build_are_discarded(self):
        """If this is not the build the update installed, nothing is shown.

        Args: None.
        Returns: None.
        """
        self.write(json.dumps({"from_build": 1, "to_build": 9, "notes": []}))
        self.assertIsNone(updater.take_whats_new(self.target))
        self.assertFalse(os.path.exists(self.whats_new_path))

    def test_unreadable_or_malformed_notes_are_discarded(self):
        """Broken JSON or missing fields give None and remove the file.

        Args: None.
        Returns: None.
        """
        for text in ("not json", json.dumps({"from_build": 1}),
                     json.dumps({"from_build": 1, "to_build": 3, "notes": [{}]})):
            self.write(text)
            self.assertIsNone(updater.take_whats_new(self.target))
            self.assertFalse(os.path.exists(self.whats_new_path))

    def test_undeletable_file_is_still_read(self):
        """If the file cannot be removed its notes are still returned.

        Args: None.
        Returns: None.
        """
        self.write(json.dumps({"from_build": 2, "to_build": 3, "notes": []}))
        with mock.patch.object(updater.os, "remove", side_effect=OSError("busy")):
            self.assertEqual(updater.take_whats_new(self.target),
                             updater.WhatsNew(2, 3, ()))

    def test_source_checkout_and_missing_file_give_none(self):
        """Running from source, or with no file, shows nothing.

        Args: None.
        Returns: None.
        """
        self.assertIsNone(updater.take_whats_new())
        self.assertIsNone(updater.take_whats_new(self.target))


if __name__ == "__main__":
    unittest.main()
