"""More network-free checks for the self-updater: build info, tokens, HTTP,
archive handling, install edge cases, relaunch and the background worker.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
import types
import unittest
import urllib.error
import zipfile
from unittest import mock

from espi_live import updater

ASSET = "espi-live-linux-x64.tar.gz"
BUILD_INFO = types.SimpleNamespace(BUILD=3, COMMIT="0123456789", REPO="owner/repo",
                                   ASSET=ASSET, TOKEN="baked")


def http_error(code):
    """Build an HTTPError with a status code.

    Args: code is the HTTP status.
    Returns: The urllib.error.HTTPError.
    """
    return urllib.error.HTTPError("https://x.invalid", code, "msg", {}, io.BytesIO())


def frozen(exe, library=None):
    """Patch sys so the program looks like a frozen build at exe.

    Args: exe is the fake executable path; library is the _internal-<build>
        folder it loaded (PyInstaller's sys._MEIPASS), if any.
    Returns: The active mock.patch.multiple context manager.
    """
    values = {"frozen": True, "executable": exe}
    if library:
        values["_MEIPASS"] = library
    return mock.patch.multiple(sys, create=True, **values)


class BuildInfoTests(unittest.TestCase):
    """Reading the CI-generated build facts."""

    def test_label_shortens_commit(self):
        """label shows build number and a 7-character commit.

        Args: None.
        Returns: None.
        """
        info = updater.UpdateInfo(9, "abcdef1234", ASSET, "u", "d", 1)
        self.assertEqual(info.label(), "build 9 (abcdef1)")

    def test_source_checkout_has_no_build_number(self):
        """Without _build_info the build is 0 and described as source.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(updater, "_build_info", None):
            self.assertEqual(updater.current_build(), 0)
            self.assertEqual(updater.build_value("X", "dflt"), "dflt")
            self.assertIn("source checkout", updater.describe_build())

    def test_frozen_build_is_described_from_build_info(self):
        """With _build_info present the description names build and commit.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(updater, "_build_info", BUILD_INFO):
            self.assertEqual(updater.current_build(), 3)
            self.assertEqual(updater.describe_build(),
                             f"ESPI live build 3 (0123456, {ASSET})")


class InstallTargetTests(unittest.TestCase):
    """Locating what an update replaces and whether it can run."""

    def test_frozen_executable_is_the_target(self):
        """A frozen build replaces its own executable.

        Args: None.
        Returns: None.
        """
        exe = os.path.join(os.sep, "apps", "espi")
        with frozen(exe):
            self.assertEqual(updater.install_target(), exe)

    def test_macos_bundle_is_the_target(self):
        """Inside a .app the whole bundle is replaced.

        Args: None.
        Returns: None.
        """
        bundle = os.path.join(os.sep, "apps", "ESPI.app")
        with frozen(os.path.join(bundle, "Contents", "MacOS", "espi")):
            self.assertEqual(updater.install_target(), bundle)

    def test_unavailable_reasons(self):
        """Source runs and builds without repo info report why.

        Args: None.
        Returns: None.
        """
        self.assertIn("source", updater.unavailable_reason())
        with frozen("/apps/espi"):
            with mock.patch.object(updater, "_build_info", None):
                self.assertIn("no update information", updater.unavailable_reason())
            with mock.patch.object(updater, "_build_info", BUILD_INFO):
                self.assertIsNone(updater.unavailable_reason())


class TokenTests(unittest.TestCase):
    """Choosing the GitHub token."""

    def setUp(self):
        """Use a temp install folder and a clean environment.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.exe = os.path.join(self.tmp.name, "espi")
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(updater.TOKEN_ENV, None)
        info = mock.patch.object(updater, "_build_info", BUILD_INFO)
        info.start()
        self.addCleanup(info.stop)

    def test_environment_token_wins(self):
        """ESPI_UPDATE_TOKEN overrides everything else.

        Args: None.
        Returns: None.
        """
        os.environ[updater.TOKEN_ENV] = "  fromenv \n"
        self.assertEqual(updater.update_token(), "fromenv")

    def test_token_file_beside_program_overrides_baked_token(self):
        """update-token.txt next to the program replaces the baked token.

        Args: None.
        Returns: None.
        """
        with open(os.path.join(self.tmp.name, updater.TOKEN_FILE), "w") as handle:
            handle.write("fromfile\n")
        with frozen(self.exe):
            self.assertEqual(updater.update_token(), "fromfile")

    def test_empty_or_missing_token_file_falls_back_to_baked(self):
        """A missing or blank token file uses the baked-in token.

        Args: None.
        Returns: None.
        """
        with frozen(self.exe):
            self.assertEqual(updater.update_token(), "baked")
            with open(os.path.join(self.tmp.name, updater.TOKEN_FILE), "w") as handle:
                handle.write("  ")
            self.assertEqual(updater.update_token(), "baked")

    def test_source_run_uses_baked_token(self):
        """Without an install target the baked token is returned.

        Args: None.
        Returns: None.
        """
        self.assertEqual(updater.update_token(), "baked")


class NetworkHelperTests(unittest.TestCase):
    """TLS context, request building and error messages."""

    def test_ssl_context_without_certifi_uses_defaults(self):
        """A missing certifi falls back to the default context.

        Args: None.
        Returns: None.
        """
        with mock.patch.dict(sys.modules, {"certifi": None}), \
                mock.patch.object(updater.ssl, "create_default_context",
                                  return_value="ctx") as create:
            self.assertEqual(updater.ssl_context(), "ctx")
        create.assert_called_once_with()

    def test_ssl_context_with_certifi_uses_its_bundle(self):
        """A bundled certifi supplies the CA file.

        Args: None.
        Returns: None.
        """
        fake = types.SimpleNamespace(where=lambda: "/ca.pem")
        with mock.patch.dict(sys.modules, {"certifi": fake}), \
                mock.patch.object(updater.ssl, "create_default_context",
                                  return_value="ctx") as create:
            self.assertEqual(updater.ssl_context(), "ctx")
        create.assert_called_once_with(cafile="/ca.pem")

    def test_open_url_sends_token_in_unredirected_header(self):
        """The token and API version never travel on redirects.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(updater, "ssl_context", return_value="ctx"), \
                mock.patch.object(updater.urllib.request, "urlopen",
                                  return_value="resp") as urlopen:
            self.assertEqual(updater.open_url("https://a.invalid/x", "tok", "app/json"),
                             "resp")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.unredirected_hdrs["Authorization"], "Bearer tok")
        self.assertIn("X-github-api-version", request.unredirected_hdrs)
        self.assertEqual(request.get_header("Accept"), "app/json")
        self.assertNotIn("Authorization", request.headers)

    def test_open_url_without_token_sends_no_authorization(self):
        """Anonymous requests omit the Authorization header.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(updater, "ssl_context", return_value="ctx"), \
                mock.patch.object(updater.urllib.request, "urlopen") as urlopen:
            updater.open_url("https://a.invalid/x", "", "a")
        self.assertNotIn("Authorization", urlopen.call_args.args[0].unredirected_hdrs)

    def test_http_errors_are_explained(self):
        """Known status codes get specific messages; others show the code.

        Args: None.
        Returns: None.
        """
        self.assertIn("token was rejected", str(updater.explain_http_error(http_error(401))))
        self.assertIn("no published build", str(updater.explain_http_error(http_error(404))))
        self.assertIn("rate limit", str(updater.explain_http_error(http_error(403))))
        self.assertIn("HTTP 500", str(updater.explain_http_error(http_error(500))))

    def test_network_errors_are_explained(self):
        """A URLError says GitHub cannot be reached.

        Args: None.
        Returns: None.
        """
        error = updater.explain_http_error(urllib.error.URLError("offline"))
        self.assertIn("cannot reach GitHub: offline", str(error))
        self.assertIsInstance(error, updater.UpdateError)


class CheckForUpdateTests(unittest.TestCase):
    """check_for_update against fake release payloads."""

    def setUp(self):
        """Pretend to be a frozen build 3 with repo information.

        Args: None.
        Returns: None.
        """
        for patcher in (frozen("/apps/espi"),
                        mock.patch.object(updater, "_build_info", BUILD_INFO),
                        mock.patch.object(updater, "update_token", return_value="")):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.release = {"assets": [
            {"name": updater.MANIFEST_NAME, "url": "https://api/manifest", "size": 1},
            {"name": ASSET, "url": "https://api/asset", "size": 50}]}
        self.manifest = {"build": 8, "commit": "feedface00", "archives": {ASSET: "AA" * 32}}

    def fake_responses(self, *payloads):
        """Patch open_url to return JSON payloads in order.

        Args: payloads are the objects (or exceptions) each call yields.
        Returns: The active mock.patch context manager.
        """
        def respond(*args):
            """Return the next fake response.

            Args: args are the open_url arguments, ignored.
            Returns: A file-like JSON body.
            """
            item = next(items)
            if isinstance(item, Exception):
                raise item
            return io.BytesIO(item if isinstance(item, bytes)
                              else json.dumps(item).encode())
        items = iter(payloads)
        return mock.patch.object(updater, "open_url", side_effect=respond)

    def test_newer_release_is_offered(self):
        """A newer manifest yields an UpdateInfo with the asset URL.

        Args: None.
        Returns: None.
        """
        with self.fake_responses(self.release, self.manifest):
            info = updater.check_for_update()
        self.assertEqual((info.build, info.asset_url, info.size),
                         (8, "https://api/asset", 50))

    def test_same_build_returns_none(self):
        """An equal build number means no update.

        Args: None.
        Returns: None.
        """
        self.manifest["build"] = 3
        with self.fake_responses(self.release, self.manifest):
            self.assertIsNone(updater.check_for_update())

    def test_missing_manifest_asset_is_an_error(self):
        """A release without build-info.json is reported.

        Args: None.
        Returns: None.
        """
        self.release["assets"].pop(0)
        with self.fake_responses(self.release), self.assertRaises(updater.UpdateError) as ctx:
            updater.check_for_update()
        self.assertIn(updater.MANIFEST_NAME, str(ctx.exception))

    def test_missing_platform_asset_is_an_error(self):
        """A newer build without this platform's archive raises.

        Args: None.
        Returns: None.
        """
        self.release["assets"].pop(1)
        with self.fake_responses(self.release, self.manifest), \
                self.assertRaises(updater.UpdateError):
            updater.check_for_update()

    def test_http_error_becomes_update_error(self):
        """An HTTP failure is translated to an UpdateError.

        Args: None.
        Returns: None.
        """
        with self.fake_responses(http_error(401)), self.assertRaises(updater.UpdateError) as ctx:
            updater.check_for_update()
        self.assertIn("token was rejected", str(ctx.exception))

    def test_socket_error_becomes_update_error(self):
        """A plain OSError (timeout, reset) is translated as unreachable.

        Args: None.
        Returns: None.
        """
        with self.fake_responses(TimeoutError("slow")), \
                self.assertRaises(updater.UpdateError) as ctx:
            updater.check_for_update()
        self.assertIn("cannot reach GitHub", str(ctx.exception))

    def test_invalid_json_becomes_update_error(self):
        """Unparseable release data is reported as unreadable.

        Args: None.
        Returns: None.
        """
        with self.fake_responses(b"not json"), self.assertRaises(updater.UpdateError) as ctx:
            updater.check_for_update()
        self.assertIn("unreadable release information", str(ctx.exception))

    def test_build_without_update_information_is_refused(self):
        """A build with no repo/asset raises before any request.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(updater, "_build_info", None), \
                mock.patch.object(updater, "open_url") as opened, \
                self.assertRaises(updater.UpdateError):
            updater.check_for_update()
        opened.assert_not_called()


class FileHelperTests(unittest.TestCase):
    """remove_path and unpack."""

    def setUp(self):
        """Create a temp folder.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def path(self, name):
        """Join a name onto the temp folder.

        Args: name is a relative file name.
        Returns: The absolute path.
        """
        return os.path.join(self.root, name)

    def make_tar(self, name, entries):
        """Write a .tar.gz with files.

        Args: name is the archive name; entries are top-level file names.
        Returns: The archive path.
        """
        archive = self.path(name)
        with tarfile.open(archive, "w:gz") as tar:
            for entry in entries:
                info = tarfile.TarInfo(entry)
                info.size = 1
                tar.addfile(info, io.BytesIO(b"x"))
        return archive

    def test_remove_path_handles_file_folder_link_and_missing(self):
        """Files, folders and symlinks are removed; missing paths are ignored.

        Args: None.
        Returns: None.
        """
        folder = self.path("dir")
        os.makedirs(os.path.join(folder, "sub"))
        updater.remove_path(folder)
        self.assertFalse(os.path.exists(folder))
        file = self.path("f")
        open(file, "w").close()
        updater.remove_path(file)
        self.assertFalse(os.path.exists(file))
        os.makedirs(self.path("real"))
        os.symlink(self.path("real"), self.path("link"))
        updater.remove_path(self.path("link"))
        self.assertFalse(os.path.lexists(self.path("link")))
        self.assertTrue(os.path.isdir(self.path("real")))
        updater.remove_path(self.path("nothing"))

    def test_tar_unpacks_to_single_entry(self):
        """A one-file tar returns that entry.

        Args: None.
        Returns: None.
        """
        entry = updater.unpack(self.make_tar("a.tar.gz", ["prog"]), self.path("out"))
        self.assertEqual(os.path.basename(entry), "prog")

    def test_tar_without_data_filter_still_unpacks(self):
        """Older Pythons lacking tarfile.data_filter extract without a filter.

        Args: None.
        Returns: None.
        """
        archive = self.make_tar("a.tar.gz", ["prog"])
        fake = types.SimpleNamespace(open=tarfile.open, TarError=tarfile.TarError)
        with mock.patch.object(updater, "tarfile", fake):
            entry = updater.unpack(archive, self.path("out"))
        self.assertTrue(os.path.isfile(entry))

    def test_archive_with_several_entries_is_rejected(self):
        """Anything but one top-level entry is an error.

        Args: None.
        Returns: None.
        """
        with self.assertRaises(updater.UpdateError):
            updater.unpack(self.make_tar("a.tar.gz", ["a", "b"]), self.path("out"))

    def test_macosx_metadata_folder_is_ignored(self):
        """__MACOSX junk beside the program does not count as an entry.

        Args: None.
        Returns: None.
        """
        archive = self.path("a.zip")
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("prog", "x")
            bundle.writestr("__MACOSX/._prog", "x")
        with mock.patch.object(updater.sys, "platform", "linux"):
            entry = updater.unpack(archive, self.path("out"))
        self.assertEqual(os.path.basename(entry), "prog")

    def test_corrupt_tar_is_an_update_error(self):
        """A damaged tar raises UpdateError.

        Args: None.
        Returns: None.
        """
        bad = self.path("bad.tar.gz")
        with open(bad, "wb") as handle:
            handle.write(b"junk")
        with self.assertRaises(updater.UpdateError):
            updater.unpack(bad, self.path("out"))

    def test_corrupt_zip_is_an_update_error(self):
        """A damaged zip raises UpdateError off macOS.

        Args: None.
        Returns: None.
        """
        bad = self.path("bad.zip")
        with open(bad, "wb") as handle:
            handle.write(b"junk")
        with mock.patch.object(updater.sys, "platform", "win32"), \
                self.assertRaises(updater.UpdateError):
            updater.unpack(bad, self.path("out"))

    def test_macos_zip_goes_through_ditto(self):
        """On macOS zips are extracted by ditto.

        Args: None.
        Returns: None.
        """
        def fake_ditto(command, **kwargs):
            """Pretend ditto extracted one entry.

            Args: command is the argv list; kwargs are ignored.
            Returns: None.
            """
            open(os.path.join(command[-1], "ESPI.app"), "w").close()

        with mock.patch.object(updater.sys, "platform", "darwin"), \
                mock.patch.object(updater.subprocess, "run",
                                  side_effect=fake_ditto) as run:
            entry = updater.unpack(self.path("a.zip"), self.path("out"))
        self.assertEqual(run.call_args.args[0][:3], ["ditto", "-x", "-k"])
        self.assertEqual(os.path.basename(entry), "ESPI.app")

    def test_ditto_failure_is_an_update_error(self):
        """A failing ditto is reported as an unpack error.

        Args: None.
        Returns: None.
        """
        failure = subprocess.CalledProcessError(1, "ditto")
        with mock.patch.object(updater.sys, "platform", "darwin"), \
                mock.patch.object(updater.subprocess, "run", side_effect=failure), \
                self.assertRaises(updater.UpdateError):
            updater.unpack(self.path("a.zip"), self.path("out"))


class DownloadTests(unittest.TestCase):
    """download_update edge cases."""

    def setUp(self):
        """Create an install folder with a fake program.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.target = os.path.join(self.tmp.name, "espi")
        open(self.target, "w").close()
        self.info = updater.UpdateInfo(4, "c" * 10, ASSET, "https://x.invalid/a",
                                       "0" * 64, 3)

    def test_source_run_cannot_download(self):
        """Without an install target download refuses.

        Args: None.
        Returns: None.
        """
        with self.assertRaises(updater.UpdateError):
            updater.download_update(self.info)

    def test_default_target_comes_from_install_target(self):
        """With no explicit target the frozen executable is used.

        Args: None.
        Returns: None.
        """
        with frozen(self.target), mock.patch.object(updater, "update_token", return_value=""), \
                mock.patch.object(updater, "open_url",
                                  side_effect=urllib.error.URLError("down")), \
                self.assertRaises(updater.UpdateError):
            updater.download_update(self.info)
        self.assertTrue(os.path.isdir(os.path.join(self.tmp.name,
                                                   f"{updater.STAGING_PREFIX}4")))

    def test_unwritable_folder_is_reported(self):
        """Staging failures tell the user to move the program.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(updater, "remove_path", side_effect=OSError("ro")), \
                self.assertRaises(updater.UpdateError) as ctx:
            updater.download_update(self.info, target=self.target)
        self.assertIn("cannot write beside the program", str(ctx.exception))

    def test_network_failure_is_explained(self):
        """A URLError during download becomes an UpdateError.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(updater, "update_token", return_value=""), \
                mock.patch.object(updater, "open_url", side_effect=http_error(404)), \
                self.assertRaises(updater.UpdateError) as ctx:
            updater.download_update(self.info, target=self.target)
        self.assertIn("no published build", str(ctx.exception))

    def test_disk_failure_during_download_is_reported(self):
        """An OSError while reading or writing is a download failure.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(updater, "update_token", return_value=""), \
                mock.patch.object(updater, "open_url", side_effect=OSError("disk")), \
                self.assertRaises(updater.UpdateError) as ctx:
            updater.download_update(self.info, target=self.target)
        self.assertIn("download failed: disk", str(ctx.exception))

    def test_corrupt_download_is_rejected_without_progress_callback(self):
        """A checksum mismatch raises even when no progress callback is given.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(updater, "update_token", return_value=""), \
                mock.patch.object(updater, "open_url",
                                  side_effect=lambda *a: io.BytesIO(b"abc")), \
                self.assertRaises(updater.UpdateError) as ctx:
            updater.download_update(self.info, target=self.target)
        self.assertIn("SHA-256 mismatch", str(ctx.exception))


class InstallEdgeTests(unittest.TestCase):
    """install_update and cleanup_previous_update edge cases."""

    def setUp(self):
        """Create an install folder.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def write(self, path, text):
        """Write a small text file, creating its folder.

        Args: path is the file; text is its content.
        Returns: path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            handle.write(text)
        return path

    def read(self, path):
        """Read a small text file.

        Args: path is the file.
        Returns: Its text.
        """
        with open(path) as handle:
            return handle.read()

    def install_folder(self, build=6):
        """Create an installed program folder that is running build.

        Args: build is the running build's number.
        Returns: (executable path, library folder path).
        """
        target = self.write(os.path.join(self.root, "espi"), "old")
        library = os.path.join(self.root, f"_internal-{build}")
        self.write(os.path.join(library, "lib"), f"lib {build}")
        return target, library

    def staged_folder(self, *names):
        """Create an unpacked program folder: files, plus folders for
        _internal-* names.

        Args: names are its top-level entries.
        Returns: The folder path.
        """
        updater.remove_path(os.path.join(self.root, "stage"))
        staged = os.path.join(self.root, "stage", "espi")
        os.makedirs(staged)
        for name in names:
            if updater.is_contents_folder(name):
                self.write(os.path.join(staged, name, "lib"), f"new {name}")
            else:
                self.write(os.path.join(staged, name), "new")
        return staged

    def test_source_run_cannot_install(self):
        """Without an install target install refuses.

        Args: None.
        Returns: None.
        """
        with self.assertRaises(updater.UpdateError):
            updater.install_update("/nowhere")

    def test_default_target_comes_from_install_target(self):
        """Without an explicit target the frozen executable is replaced.

        Args: None.
        Returns: None.
        """
        target, library = self.install_folder()
        staged = self.staged_folder("espi", "_internal-7")
        with frozen(target, library):
            self.assertEqual(updater.install_update(staged), target)
        self.assertEqual(self.read(target), "new")
        self.assertEqual(self.read(os.path.join(self.root, "_internal-7", "lib")),
                         "new _internal-7")

    def test_update_must_hold_the_program_and_one_library_folder(self):
        """Anything but the executable plus one _internal-<build> is refused
        before the installed copy is touched.

        Args: None.
        Returns: None.
        """
        layouts = (("_internal-7",), ("espi",), ("espi", "_internal-7", "README.txt"),
                   ("espi", "_internal-7", "_internal-8"), ("espi", "_internal-x"))
        for names in layouts:
            with self.subTest(names=names):
                target, library = self.install_folder()
                with frozen(target, library), self.assertRaises(updater.UpdateError) as ctx:
                    updater.install_update(self.staged_folder(*names))
                self.assertIn("unexpected update contents", str(ctx.exception))
                self.assertEqual(sorted(os.listdir(self.root)),
                                 ["_internal-6", "espi", "stage"])

    def test_single_file_update_is_refused(self):
        """A one-file archive (the old layout) is not installed over a folder build.

        Args: None.
        Returns: None.
        """
        target, library = self.install_folder()
        staged = self.write(os.path.join(self.root, "stage", "espi"), "one file")
        with frozen(target, library), self.assertRaises(updater.UpdateError):
            updater.install_update(staged)
        self.assertEqual(self.read(target), "old")

    def test_running_library_folder_is_never_replaced(self):
        """An update whose library folder has the running build's name is refused.

        Args: None.
        Returns: None.
        """
        target, library = self.install_folder(build=7)
        with frozen(target, library), self.assertRaises(updater.UpdateError) as ctx:
            updater.install_update(self.staged_folder("espi", "_internal-7"))
        self.assertIn("running build uses", str(ctx.exception))
        self.assertEqual(self.read(os.path.join(library, "lib")), "lib 7")
        self.assertEqual(self.read(target), "old")

    def test_library_left_by_an_interrupted_update_is_replaced(self):
        """A stale _internal-<new build> from an earlier attempt is overwritten.

        Args: None.
        Returns: None.
        """
        target, library = self.install_folder()
        self.write(os.path.join(self.root, "_internal-7", "junk"), "partial")
        with frozen(target, library):
            updater.install_update(self.staged_folder("espi", "_internal-7"))
        self.assertEqual(os.listdir(os.path.join(self.root, "_internal-7")), ["lib"])

    def test_library_move_failure_leaves_program_untouched(self):
        """If the library folder cannot be moved in, nothing is swapped.

        Args: None.
        Returns: None.
        """
        target, library = self.install_folder()
        staged = self.staged_folder("espi", "_internal-7")
        with frozen(target, library), \
                mock.patch.object(updater, "move", side_effect=OSError("locked")), \
                self.assertRaises(updater.UpdateError) as ctx:
            updater.install_update(staged)
        self.assertIn("cannot move _internal-7 into place: locked", str(ctx.exception))
        self.assertEqual(self.read(target), "old")

    def test_failed_program_swap_removes_the_new_library(self):
        """If the executable cannot be swapped, the moved-in library goes again.

        Args: None.
        Returns: None.
        """
        target, library = self.install_folder()
        staged = self.staged_folder("espi", "_internal-7")
        with frozen(target, library), \
                mock.patch.object(updater, "replace_entry",
                                  side_effect=updater.UpdateError("busy")), \
                self.assertRaises(updater.UpdateError):
            updater.install_update(staged)
        self.assertEqual(sorted(os.listdir(self.root)), ["_internal-6", "espi", "stage"])

    def test_failed_swap_puts_the_old_entry_back(self):
        """If the new entry cannot be moved in, the current one is restored.

        Args: None.
        Returns: None.
        """
        target = self.write(os.path.join(self.root, "espi"), "old")
        with self.assertRaises(updater.UpdateError) as ctx:
            updater.replace_entry(os.path.join(self.root, "missing"), target)
        self.assertIn("cannot install the new build", str(ctx.exception))
        self.assertEqual(self.read(target), "old")
        self.assertFalse(os.path.exists(target + updater.OLD_SUFFIX))

    def test_bundle_directory_is_swapped_and_old_one_kept(self):
        """A .app folder is renamed to .old and the new folder moved in.

        Args: None.
        Returns: None.
        """
        target = os.path.join(self.root, "ESPI.app")
        staged = os.path.join(self.root, "stage", "ESPI.app")
        os.makedirs(target)
        os.makedirs(staged)
        updater.install_update(staged, target=target)
        self.assertTrue(os.path.isdir(target + updater.OLD_SUFFIX))
        self.assertTrue(os.path.isdir(target))

    def test_stale_backup_is_replaced(self):
        """An older *.old is deleted before the new backup is made.

        Args: None.
        Returns: None.
        """
        target, library = self.install_folder()
        self.write(target + updater.OLD_SUFFIX, "stale")
        with frozen(target, library):
            updater.install_update(self.staged_folder("espi", "_internal-7"))
        self.assertEqual(self.read(target + updater.OLD_SUFFIX), "old")

    def test_unrenamable_target_is_reported(self):
        """If the running copy cannot be renamed the swap is refused.

        Args: None.
        Returns: None.
        """
        target = self.write(os.path.join(self.root, "espi"), "old")
        with mock.patch.object(updater.os, "rename", side_effect=OSError("busy")), \
                self.assertRaises(updater.UpdateError) as ctx:
            updater.replace_entry("/staged", target)
        self.assertIn("close other running copies", str(ctx.exception))

    def test_cleanup_without_target_does_nothing(self):
        """In a source run cleanup is a no-op.

        Args: None.
        Returns: None.
        """
        self.assertIsNone(updater.cleanup_previous_update())

    def test_cleanup_uses_install_target_by_default(self):
        """The frozen executable's leftovers are removed.

        Args: None.
        Returns: None.
        """
        target = os.path.join(self.root, "espi")
        old = target + updater.OLD_SUFFIX
        open(old, "w").close()
        with frozen(target):
            updater.cleanup_previous_update()
        self.assertFalse(os.path.exists(old))

    def test_cleanup_removes_staging_but_can_keep_backup(self):
        """keep_backup leaves *.old but still removes staging folders.

        Args: None.
        Returns: None.
        """
        target = os.path.join(self.root, "espi")
        old = target + updater.OLD_SUFFIX
        staging = os.path.join(self.root, f"{updater.STAGING_PREFIX}1")
        open(old, "w").close()
        os.makedirs(staging)
        updater.cleanup_previous_update(target, keep_backup=True)
        self.assertTrue(os.path.exists(old))
        self.assertFalse(os.path.exists(staging))

    def test_cleanup_survives_unlistable_folder(self):
        """A missing install folder is ignored.

        Args: None.
        Returns: None.
        """
        updater.cleanup_previous_update(os.path.join(self.root, "gone", "espi"))

    def test_cleanup_survives_undeletable_leftover(self):
        """A leftover that cannot be deleted is skipped quietly.

        Args: None.
        Returns: None.
        """
        target = os.path.join(self.root, "espi")
        open(target + updater.OLD_SUFFIX, "w").close()
        with mock.patch.object(updater, "remove_path", side_effect=OSError("in use")):
            updater.cleanup_previous_update(target)

    def test_old_library_waits_until_old_program_is_deleted(self):
        """While *.old cannot be deleted (on Windows: it still runs), the old
        build's library folder stays; a later start removes both.

        Args: None.
        Returns: None.
        """
        target, old_library = self.install_folder()
        new_library = os.path.join(self.root, "_internal-7")
        os.makedirs(new_library)
        self.write(target + updater.OLD_SUFFIX, "old program")
        real_remove = updater.remove_path

        def remove(path):
            """Refuse to delete the old program, as Windows does while it runs.

            Args: path is what cleanup deletes.
            Returns: None. Raises PermissionError for the *.old program.
            """
            if path.endswith(updater.OLD_SUFFIX):
                raise PermissionError("in use")
            real_remove(path)

        with frozen(target, new_library), \
                mock.patch.object(updater, "remove_path", side_effect=remove):
            updater.cleanup_previous_update()
        self.assertTrue(os.path.isdir(old_library))
        with frozen(target, new_library):
            updater.cleanup_previous_update()
        self.assertEqual(sorted(os.listdir(self.root)), ["_internal-7", "espi"])

    def test_library_folders_stay_unless_running_from_one_here(self):
        """Source runs and builds whose libraries are elsewhere (a .app) never
        delete _internal-* folders.

        Args: None.
        Returns: None.
        """
        target, library = self.install_folder()
        elsewhere = os.path.join(self.root, "ESPI.app", "Contents", "Frameworks")
        for running in (None, elsewhere):
            with self.subTest(running=running), frozen(target, running):
                updater.cleanup_previous_update()
                self.assertTrue(os.path.isdir(library))


class MoveTests(unittest.TestCase):
    """Renames retry on Windows, where freshly unpacked files are briefly locked."""

    def test_windows_retries_until_the_rename_works(self):
        """A PermissionError on Windows is retried.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(updater.os, "name", "nt"), \
                mock.patch.object(updater, "RENAME_WAIT_S", 0), \
                mock.patch.object(updater.os, "rename",
                                  side_effect=[PermissionError("scanning"), None]) as rename:
            updater.move("a", "b")
        self.assertEqual(rename.call_count, 2)

    def test_windows_gives_up_after_the_last_attempt(self):
        """A rename that stays locked raises after RENAME_ATTEMPTS tries.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(updater.os, "name", "nt"), \
                mock.patch.object(updater, "RENAME_WAIT_S", 0), \
                mock.patch.object(updater.os, "rename",
                                  side_effect=PermissionError("locked")) as rename, \
                self.assertRaises(PermissionError):
            updater.move("a", "b")
        self.assertEqual(rename.call_count, updater.RENAME_ATTEMPTS)

    def test_other_systems_fail_at_once(self):
        """Elsewhere a PermissionError is raised without retrying.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(updater.os, "name", "posix"), \
                mock.patch.object(updater.os, "rename",
                                  side_effect=PermissionError("denied")) as rename, \
                self.assertRaises(PermissionError):
            updater.move("a", "b")
        self.assertEqual(rename.call_count, 1)


class RelaunchTests(unittest.TestCase):
    """Starting the installed build."""

    def test_posix_launch_detaches_session(self):
        """On POSIX the new process gets its own session and reset env.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(updater.subprocess, "Popen") as popen:
            updater.relaunch(["--x"], target="/apps/espi")
        self.assertEqual(popen.call_args.args[0], ["/apps/espi", "--x"])
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(popen.call_args.kwargs["env"]["PYINSTALLER_RESET_ENVIRONMENT"], "1")

    def test_default_target_comes_from_install_target(self):
        """Without a target the frozen executable is relaunched.

        Args: None.
        Returns: None.
        """
        with frozen("/apps/espi"), mock.patch.object(updater.subprocess, "Popen") as popen:
            updater.relaunch([])
        self.assertEqual(popen.call_args.args[0], ["/apps/espi"])

    def test_app_bundle_launches_inner_executable(self):
        """A .app target launches Contents/MacOS/<running executable name>.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(updater.sys, "executable", "/x/y/espi-bin"), \
                mock.patch.object(updater.subprocess, "Popen") as popen:
            updater.relaunch([], target="/apps/ESPI.app")
        self.assertEqual(popen.call_args.args[0],
                         [os.path.join("/apps/ESPI.app", "Contents", "MacOS", "espi-bin")])

    def test_windows_launch_opens_new_console(self):
        """On Windows the new process gets a new console window.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(updater.os, "name", "nt"), \
                mock.patch.object(updater.subprocess, "CREATE_NEW_CONSOLE", 16,
                                  create=True), \
                mock.patch.object(updater.subprocess, "Popen") as popen:
            updater.relaunch(["a"], target="C:/espi.exe")
        self.assertEqual(popen.call_args.kwargs["creationflags"], 16)
        self.assertNotIn("start_new_session", popen.call_args.kwargs)


class UpdateWorkerTests(unittest.TestCase):
    """The background thread front ends poll."""

    def test_check_posts_result(self):
        """A successful check posts ("checked", result).

        Args: None.
        Returns: None.
        """
        worker = updater.UpdateWorker()
        with mock.patch.object(updater, "check_for_update", return_value="info"):
            self.assertTrue(worker.check())
            self.assertEqual(worker.events.get(timeout=5), ("checked", "info"))

    def test_check_failure_posts_message_and_clears_busy(self):
        """A failing check posts ("check_failed", text) and frees the worker.

        Args: None.
        Returns: None.
        """
        worker = updater.UpdateWorker()
        with mock.patch.object(updater, "check_for_update",
                               side_effect=updater.UpdateError("nope")):
            worker.check()
            self.assertEqual(worker.events.get(timeout=5), ("check_failed", "nope"))
        threading.Event().wait(0.05)
        self.assertFalse(worker.busy)

    def test_second_job_is_refused_while_busy(self):
        """A job started while another runs returns False.

        Args: None.
        Returns: None.
        """
        worker = updater.UpdateWorker()
        release = threading.Event()
        with mock.patch.object(updater, "check_for_update",
                               side_effect=lambda: release.wait(5)):
            self.assertTrue(worker.check())
            self.assertFalse(worker.check())
            release.set()
            worker.events.get(timeout=5)

    def test_download_reports_progress_and_result(self):
        """download updates progress and posts the staged path.

        Args: None.
        Returns: None.
        """
        worker = updater.UpdateWorker()
        info = updater.UpdateInfo(2, "c" * 10, ASSET, "u", "d", 100)

        def fake_download(passed, report):
            """Report progress then return a staged path.

            Args: passed is the UpdateInfo; report is the progress callback.
            Returns: A fake staged path.
            """
            report(40, 100)
            return "/staged"

        with mock.patch.object(updater, "download_update", side_effect=fake_download):
            self.assertTrue(worker.download(info))
            self.assertEqual(worker.events.get(timeout=5), ("downloaded", "/staged"))
        self.assertEqual(worker.progress, (40, 100))

    def test_download_failure_posts_message(self):
        """A failed download posts ("download_failed", text).

        Args: None.
        Returns: None.
        """
        worker = updater.UpdateWorker()
        info = updater.UpdateInfo(2, "c" * 10, ASSET, "u", "d", 100)
        with mock.patch.object(updater, "download_update",
                               side_effect=updater.UpdateError("bad")):
            worker.download(info)
            self.assertEqual(worker.events.get(timeout=5), ("download_failed", "bad"))
        self.assertEqual(worker.progress, (0, 100))


if __name__ == "__main__":
    unittest.main()
