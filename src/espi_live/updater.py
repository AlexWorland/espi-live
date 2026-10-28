"""Self-update of the prebuilt binaries from the newest GitHub release.

Only PyInstaller builds update themselves; a source checkout updates with git.
CI writes ``_build_info.py`` before freezing (build number, commit, repository,
the archive name for this platform and a read-only token for the private
repository) and publishes ``build-info.json`` with each archive's SHA-256 next
to the archives. Only the newest full release is used: GitHub's "latest"
release excludes pre-releases and drafts, so test builds are never offered. An update downloads the archive beside the installed copy,
verifies it, unpacks it, then renames the running executable (or the .app
bundle) to ``*.old`` and moves the new one into its place: a running Windows
executable can be renamed but not overwritten. ``*.old`` is removed on the
next start. No GUI imports: the front ends drive this module.
"""

import hashlib
import json
import os
import queue
import shutil
import ssl
import subprocess
import sys
import tarfile
import threading
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass

try:
    from espi_live import _build_info
except ImportError:  # source checkout: CI generates this module
    _build_info = None

API_ROOT = "https://api.github.com"
MANIFEST_NAME = "build-info.json"
TOKEN_FILE = "update-token.txt"
TOKEN_ENV = "ESPI_UPDATE_TOKEN"
OLD_SUFFIX = ".old"
STAGING_PREFIX = ".espi-update-"
TIMEOUT_S = 20
CHUNK_BYTES = 1 << 20


class UpdateError(Exception):
    """An update check, download or install that could not complete."""


@dataclass(frozen=True)
class UpdateInfo:
    """A newer published build for this platform."""

    build: int
    commit: str
    asset_name: str
    asset_url: str
    sha256: str
    size: int

    def label(self):
        """Describe the build for messages.

        Args: None.
        Returns: A string such as "build 42 (abc1234)".
        """
        return f"build {self.build} ({self.commit[:7]})"


def build_value(name, default):
    """Read one baked-in build fact.

    Args: name is the _build_info attribute; default is used from source.
    Returns: The value, or default when it is missing.
    """
    return getattr(_build_info, name, default)


def current_build():
    """Return this copy's CI build number.

    Args: None.
    Returns: The build number, or 0 for a source checkout.
    """
    return int(build_value("BUILD", 0))


def describe_build():
    """Describe this copy for --version and the About box.

    Args: None.
    Returns: A one-line description.
    """
    if _build_info is None:
        return "ESPI live (source checkout, no build number)"
    return (f"ESPI live build {current_build()} "
            f"({build_value('COMMIT', '')[:7]}, {build_value('ASSET', '?')})")


def install_target():
    """Find what an update replaces.

    Args: None.
    Returns: The executable path, the .app bundle path on macOS, or None when
        running from source.
    """
    if not getattr(sys, "frozen", False):
        return None
    exe = os.path.abspath(sys.executable)
    folder = os.path.dirname(exe)
    if folder.endswith(os.path.join(".app", "Contents", "MacOS")):
        return os.path.dirname(os.path.dirname(folder))
    return exe


def unavailable_reason():
    """Explain why this copy cannot update itself.

    Args: None.
    Returns: None when updating is possible, otherwise a short reason.
    """
    if install_target() is None:
        return "running from source; update with git pull"
    if not build_value("REPO", "") or not build_value("ASSET", ""):
        return "this build has no update information"
    return None


def update_token():
    """Choose the GitHub token for the private repository.

    A token in the environment or in update-token.txt beside the installed
    copy replaces the baked-in one, so an expired token can be renewed without
    a reinstall.

    Args: None.
    Returns: The token string, or "" for anonymous access.
    """
    token = os.environ.get(TOKEN_ENV, "").strip()
    if token:
        return token
    target = install_target()
    if target:
        try:
            with open(os.path.join(os.path.dirname(target), TOKEN_FILE),
                      encoding="utf-8") as handle:
                token = handle.read().strip()
        except OSError:
            token = ""
        if token:
            return token
    return build_value("TOKEN", "")


def ssl_context():
    """Build a TLS context that works in frozen builds.

    Args: None.
    Returns: An ssl.SSLContext using certifi's CA bundle when it is bundled.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def open_url(url, token, accept):
    """Open a GitHub API URL.

    The token goes in an unredirected header: asset downloads redirect to a
    signed storage URL that must not receive it.

    Args: url is the address; token is "" or a GitHub token; accept is the
        Accept header value.
    Returns: The open response. Raises urllib.error.URLError on failure.
    """
    request = urllib.request.Request(
        url, headers={"Accept": accept, "User-Agent": "espi-live-updater"})
    request.add_unredirected_header("X-GitHub-Api-Version", "2022-11-28")
    if token:
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(request, timeout=TIMEOUT_S, context=ssl_context())


def explain_http_error(exc):
    """Turn a GitHub HTTP or network error into an UpdateError message.

    Args: exc is the urllib error.
    Returns: An UpdateError to raise.
    """
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 401:
            return UpdateError("the update token was rejected (expired or revoked); "
                               f"put a new one in {TOKEN_FILE} beside the program")
        if exc.code == 404:
            return UpdateError("no published build was found, or the update token "
                               "cannot read the repository")
        if exc.code == 403:
            return UpdateError("GitHub refused the request (rate limit or token "
                               "permissions); try again later")
        return UpdateError(f"GitHub returned HTTP {exc.code}")
    return UpdateError(f"cannot reach GitHub: {getattr(exc, 'reason', exc)}")


def select_update(manifest, assets, local_build, asset_name):
    """Decide whether a published build should replace this one.

    Args: manifest is the parsed build-info.json; assets maps release asset
        names to their GitHub API records; local_build is this copy's build
        number; asset_name is this platform's archive name.
    Returns: An UpdateInfo when the published build is newer, else None.
        Raises UpdateError when it is newer but has no usable archive.
    """
    remote_build = int(manifest.get("build", 0))
    if remote_build <= local_build:
        return None
    digest = manifest.get("sha256", {}).get(asset_name)
    asset = assets.get(asset_name)
    if not digest or asset is None:
        raise UpdateError(f"build {remote_build} has no {asset_name}")
    return UpdateInfo(build=remote_build, commit=str(manifest.get("commit", "")),
                      asset_name=asset_name, asset_url=asset["url"],
                      sha256=digest.lower(), size=int(asset.get("size", 0)))


def check_for_update():
    """Ask GitHub whether a newer build exists for this platform.

    Args: None.
    Returns: An UpdateInfo, or None when this copy is current. Raises
        UpdateError when updating is unavailable or GitHub cannot be read.
    """
    reason = unavailable_reason()
    if reason:
        raise UpdateError(reason)
    token = update_token()
    repo = build_value("REPO", "")
    try:
        with open_url(f"{API_ROOT}/repos/{repo}/releases/latest",
                      token, "application/vnd.github+json") as response:
            release = json.load(response)
        assets = {asset["name"]: asset for asset in release.get("assets", [])}
        if MANIFEST_NAME not in assets:
            raise UpdateError(f"the newest release has no {MANIFEST_NAME}")
        with open_url(assets[MANIFEST_NAME]["url"], token,
                      "application/octet-stream") as response:
            manifest = json.load(response)
    except (urllib.error.URLError, OSError) as exc:
        raise explain_http_error(exc) from exc
    except ValueError as exc:
        raise UpdateError(f"unreadable release information: {exc}") from exc
    return select_update(manifest, assets, current_build(), build_value("ASSET", ""))


def remove_path(path):
    """Delete a file or folder if it exists.

    Args: path is the file or folder.
    Returns: None. Raises OSError if it exists but cannot be removed.
    """
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.remove(path)


def unpack(archive, folder):
    """Extract a release archive and find the program inside it.

    Zip files on macOS go through ditto, which keeps the symlinks and
    permissions a .app bundle needs (zipfile drops both).

    Args: archive is a .zip or .tar.gz path; folder is an empty destination.
    Returns: The path of the archive's single top-level entry. Raises
        UpdateError if the archive is unreadable or holds something else.
    """
    os.makedirs(folder, exist_ok=True)
    try:
        if archive.endswith(".tar.gz"):
            with tarfile.open(archive) as tar:
                if hasattr(tarfile, "data_filter"):
                    tar.extractall(folder, filter="data")
                else:
                    tar.extractall(folder)
        elif sys.platform == "darwin":
            subprocess.run(["ditto", "-x", "-k", archive, folder], check=True,
                           capture_output=True)
        else:
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(folder)
    except (OSError, tarfile.TarError, zipfile.BadZipFile,
            subprocess.CalledProcessError) as exc:
        raise UpdateError(f"cannot unpack {os.path.basename(archive)}: {exc}") from exc
    entries = [name for name in os.listdir(folder) if name != "__MACOSX"]
    if len(entries) != 1:
        raise UpdateError(f"unexpected archive contents: {entries}")
    return os.path.join(folder, entries[0])


def download_update(info, progress=None, target=None):
    """Download, verify and unpack a build beside the installed copy.

    Args: info is the UpdateInfo; progress is an optional callable taking
        (bytes_done, bytes_total) from the download thread; target overrides
        install_target() (tests).
    Returns: The path of the unpacked program, ready for install_update().
        Raises UpdateError on network, disk or checksum failure.
    """
    target = target or install_target()
    if target is None:
        raise UpdateError("running from source; update with git pull")
    staging = os.path.join(os.path.dirname(target), f"{STAGING_PREFIX}{info.build}")
    try:
        remove_path(staging)
        os.makedirs(staging)
    except OSError as exc:
        raise UpdateError(f"cannot write beside the program ({exc}); "
                          "move it to a folder you can write to") from exc
    archive = os.path.join(staging, info.asset_name)
    digest = hashlib.sha256()
    done = 0
    try:
        with open_url(info.asset_url, update_token(), "application/octet-stream") as response, \
                open(archive, "wb") as out:
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress:
                    progress(done, info.size)
    except urllib.error.URLError as exc:
        raise explain_http_error(exc) from exc
    except OSError as exc:
        raise UpdateError(f"download failed: {exc}") from exc
    if digest.hexdigest() != info.sha256:
        raise UpdateError("the download is corrupt (SHA-256 mismatch); try again")
    return unpack(archive, os.path.join(staging, "unpacked"))


def install_update(staged, target=None):
    """Swap the unpacked build into place, keeping the old one as *.old.

    Args: staged is the path download_update() returned; target overrides
        install_target() (tests).
    Returns: The installed path. Raises UpdateError if the swap fails (the
        old copy is put back).
    """
    target = target or install_target()
    if target is None:
        raise UpdateError("running from source; update with git pull")
    backup = target + OLD_SUFFIX
    try:
        remove_path(backup)
        os.rename(target, backup)
    except OSError as exc:
        raise UpdateError(f"cannot replace {target} ({exc}); close other "
                          "running copies and try again") from exc
    try:
        os.rename(staged, target)
    except OSError as exc:
        os.rename(backup, target)
        raise UpdateError(f"cannot install the new build: {exc}") from exc
    if os.path.isfile(target) and os.name != "nt":
        os.chmod(target, 0o755)
    cleanup_previous_update(target, keep_backup=True)
    return target


def cleanup_previous_update(target=None, keep_backup=False):
    """Remove leftovers of an earlier update (best effort).

    Args: target overrides install_target() (tests); keep_backup leaves the
        *.old copy, which is still running right after an install.
    Returns: None.
    """
    target = target or install_target()
    if target is None:
        return
    folder = os.path.dirname(target)
    leftovers = [] if keep_backup else [target + OLD_SUFFIX]
    try:
        leftovers += [os.path.join(folder, name) for name in os.listdir(folder)
                      if name.startswith(STAGING_PREFIX)]
    except OSError:
        return
    for path in leftovers:
        try:
            remove_path(path)
        except OSError:
            pass  # still in use (Windows); retried on the next start


def relaunch(args, target=None):
    """Start the installed build as a new, independent process.

    Call after the engine has released the camera. PYINSTALLER_RESET_ENVIRONMENT
    stops the new onefile build reusing this process's extraction folder.

    Args: args are the command-line arguments to pass on; target overrides
        install_target() (tests).
    Returns: None. Raises OSError if the program cannot start.
    """
    target = target or install_target()
    exe = target
    if target.endswith(".app"):
        exe = os.path.join(target, "Contents", "MacOS", os.path.basename(sys.executable))
    env = dict(os.environ, PYINSTALLER_RESET_ENVIRONMENT="1")
    if os.name == "nt":
        subprocess.Popen([exe, *args], env=env, close_fds=True,
                         creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        subprocess.Popen([exe, *args], env=env, close_fds=True, start_new_session=True)


class UpdateWorker:
    """Run checks and downloads on a background thread for a front end.

    The front end polls ``events`` for ("checked", UpdateInfo or None),
    ("check_failed", message), ("downloaded", staged_path) and
    ("download_failed", message), and reads ``progress`` for a bar.
    """

    def __init__(self):
        """Create an idle worker.

        Args: None.
        Returns: None.
        """
        self.events = queue.Queue()
        self.progress = (0, 0)
        self.busy = False

    def _run(self, job, ok_kind, fail_kind):
        """Run one job on a daemon thread and report the outcome.

        Args: job is a callable returning the result; ok_kind and fail_kind
            are the event names for success and failure.
        Returns: False if a job is already running, else True.
        """
        if self.busy:
            return False
        self.busy = True

        def work():
            """Thread body: run the job and post its event.

            Args: None.
            Returns: None.
            """
            try:
                self.events.put((ok_kind, job()))
            except Exception as exc:
                self.events.put((fail_kind, str(exc)))
            finally:
                self.busy = False

        threading.Thread(target=work, name="espi-updater", daemon=True).start()
        return True

    def check(self):
        """Start an update check.

        Args: None.
        Returns: True if the check started.
        """
        return self._run(check_for_update, "checked", "check_failed")

    def download(self, info):
        """Start downloading and unpacking a build.

        Args: info is the UpdateInfo to fetch.
        Returns: True if the download started.
        """
        self.progress = (0, info.size)

        def report(done, total):
            """Record download progress for the UI.

            Args: done and total are byte counts.
            Returns: None.
            """
            self.progress = (done, total)

        return self._run(lambda: download_update(info, report),
                         "downloaded", "download_failed")
