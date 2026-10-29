"""Network-free checks for the self-updater (fake releases in a temp folder).

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import hashlib
import io
import os
import sys
import tarfile
import tempfile
import unittest
import zipfile
from unittest import mock

from espi_live import updater

ASSET = "espi-live-linux-x64.tar.gz"


def add_file(tar, name, payload, mode=0o644):
    """Add one file to an open tar archive.

    Args: tar is the TarFile; name is the entry name; payload is bytes; mode
        is the file mode.
    Returns: None.
    """
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = mode
    tar.addfile(info, io.BytesIO(payload))


def make_program_tar(path, build, payload):
    """Write a .tar.gz laid out like CI's: espi-live/ with the executable and
    _internal-<build>/.

    Args: path is the archive path; build is the build number; payload is the
        executable's bytes.
    Returns: The archive bytes.
    """
    with tarfile.open(path, "w:gz") as tar:
        add_file(tar, "espi-live/espi-live", payload, mode=0o755)
        add_file(tar, f"espi-live/_internal-{build}/library", f"lib {build}".encode())
    with open(path, "rb") as handle:
        return handle.read()


def make_info(data, build=7):
    """Describe a fake published build.

    Args: data is the archive bytes; build is its build number.
    Returns: An UpdateInfo whose checksum matches data.
    """
    return updater.UpdateInfo(build=build, commit="abcdef1234", asset_name=ASSET,
                              asset_url="https://example.invalid/asset",
                              sha256=hashlib.sha256(data).hexdigest(), size=len(data))


def running_from(library):
    """Patch sys so the running build looks like it loaded library.

    Args: library is the _internal-<build> folder (PyInstaller's sys._MEIPASS).
    Returns: The active mock.patch context manager.
    """
    return mock.patch.object(sys, "_MEIPASS", library, create=True)


class SelectUpdateTests(unittest.TestCase):
    """Deciding whether a published build replaces this one."""

    def setUp(self):
        """Create a manifest and asset list for build 5.

        Args: None.
        Returns: None.
        """
        self.manifest = {"build": 5, "commit": "c0ffee", "archives": {ASSET: "AB" * 32}}
        self.assets = {ASSET: {"url": "https://api/asset/1", "size": 123}}

    def test_newer_build_is_offered(self):
        """A higher build number yields an UpdateInfo for this platform.

        Args: None.
        Returns: None.
        """
        info = updater.select_update(self.manifest, self.assets, 4, ASSET)
        self.assertEqual((info.build, info.size, info.asset_url), (5, 123, "https://api/asset/1"))
        self.assertEqual(info.sha256, "ab" * 32)

    def test_same_or_older_build_is_ignored(self):
        """Equal or lower build numbers never update (no downgrades).

        Args: None.
        Returns: None.
        """
        self.assertIsNone(updater.select_update(self.manifest, self.assets, 5, ASSET))
        self.assertIsNone(updater.select_update(self.manifest, self.assets, 9, ASSET))

    def test_missing_platform_archive_is_an_error(self):
        """A newer build without this platform's archive is reported.

        Args: None.
        Returns: None.
        """
        with self.assertRaises(updater.UpdateError):
            updater.select_update(self.manifest, {}, 1, ASSET)

    def test_sums_for_single_file_builds_are_not_used(self):
        """Only "archives" counts: "sha256" is what single-file builds read.

        Args: None.
        Returns: None.
        """
        manifest = {"build": 5, "commit": "c0ffee", "sha256": {ASSET: "AB" * 32}}
        with self.assertRaises(updater.UpdateError):
            updater.select_update(manifest, self.assets, 1, ASSET)


class InstallTests(unittest.TestCase):
    """Download, verify, swap and clean up against a fake installed copy."""

    def setUp(self):
        """Create a program folder holding build 6 and an archive of build 7.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        # Two levels down, so the folders cleanup searches for old single-file
        # copies (the two above the program folder) stay inside the temp tree.
        self.folder = os.path.join(self.tmp.name, "lab", "espi-live")
        self.target = os.path.join(self.folder, "espi-live")
        self.old_library = os.path.join(self.folder, "_internal-6")
        self.new_library = os.path.join(self.folder, "_internal-7")
        os.makedirs(self.old_library)
        with open(self.target, "wb") as handle:
            handle.write(b"old build")
        self.archive_bytes = make_program_tar(os.path.join(self.tmp.name, "src.tar.gz"),
                                              7, b"new build")

    def tearDown(self):
        """Delete the temporary folder.

        Args: None.
        Returns: None.
        """
        self.tmp.cleanup()

    def fake_open(self, data):
        """Patch open_url to serve bytes instead of contacting GitHub.

        Args: data is the response body.
        Returns: The active mock.patch context manager.
        """
        return mock.patch.object(updater, "open_url",
                                 side_effect=lambda *args: io.BytesIO(data))

    def read_target(self):
        """Read the installed executable.

        Args: None.
        Returns: Its bytes.
        """
        with open(self.target, "rb") as handle:
            return handle.read()

    def test_download_install_and_cleanup(self):
        """The new build moves in beside the running one; the next start
        deletes the old executable and library folder.

        Args: None.
        Returns: None.
        """
        seen = []
        with self.fake_open(self.archive_bytes):
            staged = updater.download_update(make_info(self.archive_bytes),
                                             lambda done, total: seen.append(done),
                                             target=self.target)
        self.assertEqual(seen[-1], len(self.archive_bytes))
        with running_from(self.old_library):
            updater.install_update(staged, target=self.target)
        self.assertEqual(self.read_target(), b"new build")
        self.assertTrue(os.access(self.target, os.X_OK))
        self.assertTrue(os.path.exists(self.target + updater.OLD_SUFFIX))
        self.assertTrue(os.path.isdir(self.old_library))  # the running build's
        with open(os.path.join(self.new_library, "library"), "rb") as handle:
            self.assertEqual(handle.read(), b"lib 7")
        self.assertFalse(any(name.startswith(updater.STAGING_PREFIX)
                             for name in os.listdir(self.folder)))
        messages = []
        with running_from(self.new_library):  # the next start runs build 7
            self.assertTrue(updater.cleanup_previous_update(self.target,
                                                            report=messages.append))
        self.assertEqual(sorted(os.listdir(self.folder)), ["_internal-7", "espi-live"])
        self.assertEqual(messages, ["Removed the previous build: espi-live.old, _internal-6"])

    def test_checksum_mismatch_leaves_program_untouched(self):
        """A corrupt download raises and does not touch the installed copy.

        Args: None.
        Returns: None.
        """
        info = make_info(b"something else")
        with self.fake_open(self.archive_bytes), self.assertRaises(updater.UpdateError):
            updater.download_update(info, target=self.target)
        self.assertEqual(self.read_target(), b"old build")

    def test_failed_install_leaves_program_untouched(self):
        """If there is nothing to move in, the installed copy stays as it was.

        Args: None.
        Returns: None.
        """
        missing = os.path.join(self.tmp.name, "does-not-exist")
        with running_from(self.old_library), self.assertRaises(updater.UpdateError):
            updater.install_update(missing, target=self.target)
        self.assertEqual(self.read_target(), b"old build")
        self.assertEqual(sorted(os.listdir(self.folder)), ["_internal-6", "espi-live"])

    def test_zip_with_bundle_folder_unpacks_to_one_entry(self):
        """A zipped .app-style folder unpacks to that folder.

        Args: None.
        Returns: None.
        """
        archive = os.path.join(self.tmp.name, "bundle.zip")
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("ESPI Live.app/Contents/Info.plist", "<plist/>")
        entry = updater.unpack(archive, os.path.join(self.tmp.name, "out"))
        self.assertEqual(os.path.basename(entry), "ESPI Live.app")
        self.assertTrue(os.path.isfile(os.path.join(entry, "Contents", "Info.plist")))


class SourceCheckoutTests(unittest.TestCase):
    """A source checkout never tries to replace itself."""

    def test_source_checkout_cannot_update(self):
        """Unfrozen runs report why updating is off and raise on check.

        Args: None.
        Returns: None.
        """
        self.assertIsNone(updater.install_target())
        self.assertIn("source", updater.unavailable_reason())
        with self.assertRaises(updater.UpdateError):
            updater.check_for_update()


if __name__ == "__main__":
    unittest.main()
