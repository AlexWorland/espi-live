"""Self-update of the prebuilt binaries from the newest GitHub release.

Only PyInstaller builds update themselves; a source checkout updates with git.
CI writes ``_build_info.py`` before freezing (build number, commit, repository,
the archive name for this platform and an optional read-only token for the
repository) and publishes ``build-info.json`` with each archive's SHA-256 next
to the archives. Only the newest full release is used: GitHub's "latest"
release excludes pre-releases and drafts, so test builds are never offered.

An update downloads the archive beside the installed copy, verifies it and
unpacks it. On Windows and Linux the program is a folder holding the
executable and ``_internal-<build>``, that build's libraries. Windows lets a
running program's executable and loaded libraries be renamed but not
overwritten or deleted, so the new build's ``_internal-<build>`` is moved in
beside the running one, which is left alone, and only the executable is
swapped: the running one is renamed to ``*.old``. On macOS the whole .app
bundle is swapped that way. When the new build starts (StartupCleanup), it
deletes ``*.old`` and, once that is gone (proof on Windows that the old build
has exited), the old build's ``_internal-*`` folder, retrying in the
background while the old build is still closing; it also deletes old
single-file copies (build 6 and earlier) just above its folder. No GUI
imports: the front ends drive this module.
"""

import glob
import hashlib
import json
import os
import queue
import re
import shutil
import ssl
import struct
import subprocess
import sys
import tarfile
import threading
import time
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
# Each build's libraries, beside its executable (PyInstaller's
# --contents-directory, set in build-binaries.yml).
CONTENTS_PREFIX = "_internal-"
CONTENTS_PATTERN = re.compile(re.escape(CONTENTS_PREFIX) + "[0-9]+")
RENAME_ATTEMPTS = 10
RENAME_WAIT_S = 0.5
# Startup cleanup retries while the previous build is still exiting.
CLEANUP_ATTEMPTS = 30
CLEANUP_WAIT_S = 2.0
# PyInstaller's archive format: a single-file program ends with an archive
# whose trailer ("cookie") locates its table of contents.
PYINSTALLER_MAGIC = b"MEI\014\013\012\013\016"
PYINSTALLER_COOKIE = struct.Struct("!8sIIii64s")
PYINSTALLER_ENTRY = struct.Struct("!IIIIBc")
PYINSTALLER_TAIL_BYTES = 1 << 20
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
    Returns: The executable path (its build's _internal-<build> folder is
        beside it), the .app bundle path on macOS, or None when running from
        source.
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


def read_token(env_name, file_name, build_name):
    """Choose a GitHub token from the environment, a file or the build.

    A token in the environment or in a file beside the installed copy
    replaces the baked-in one, so an expired token can be renewed without a
    reinstall.

    Args: env_name is the environment variable; file_name is the file beside
        the installed copy; build_name is the _build_info attribute.
    Returns: The token string, or "" when none is set.
    """
    token = os.environ.get(env_name, "").strip()
    if token:
        return token
    target = install_target()
    if target:
        try:
            with open(os.path.join(os.path.dirname(target), file_name),
                      encoding="utf-8") as handle:
                token = handle.read().strip()
        except OSError:
            token = ""
        if token:
            return token
    return build_value(build_name, "")


def update_token():
    """Choose the optional GitHub token for reading releases.

    Args: None.
    Returns: The token string, or "" for anonymous access.
    """
    return read_token(TOKEN_ENV, TOKEN_FILE, "TOKEN")


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
    # The single-file builds (build 6 and earlier) read the sums from "sha256",
    # which newer manifests leave out: those builds would put the new program
    # folder where their executable was, so they must not find an archive.
    digest = manifest.get("archives", {}).get(asset_name)
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
    Returns: The path of the archive's single top-level entry (the program
        folder, or the .app bundle on macOS). Raises UpdateError if the
        archive is unreadable or holds something else.
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


def discard(path):
    """Delete a file or folder if possible (best effort).

    Args: path is the file or folder.
    Returns: True when nothing is left at path.
    """
    try:
        remove_path(path)
    except OSError:
        pass  # still in use (Windows); retried on the next start
    return not os.path.lexists(path)


def remove_all(paths):
    """Delete those of some files or folders that exist (best effort).

    Args: paths are files or folders.
    Returns: (the paths removed, True when none of them is left).
    """
    removed = [path for path in paths if os.path.lexists(path) and discard(path)]
    return removed, not any(os.path.lexists(path) for path in paths)


def move(source, destination):
    """Rename a file or folder, retrying for a few seconds on Windows.

    Antivirus and search indexing briefly open freshly unpacked files, and
    while they do, Windows can refuse to rename them or the folder they are in.

    Args: source and destination are paths on the same drive.
    Returns: None. Raises OSError when the last attempt fails.
    """
    for attempt in range(RENAME_ATTEMPTS):
        try:
            os.rename(source, destination)
            return
        except PermissionError:
            if os.name != "nt" or attempt == RENAME_ATTEMPTS - 1:
                raise
            time.sleep(RENAME_WAIT_S)


def same_path(first, second):
    """Tell whether two paths name the same place.

    Args: first and second are paths.
    Returns: True when they are equal once made absolute (ignoring case on
        Windows).
    """
    return (os.path.normcase(os.path.abspath(first))
            == os.path.normcase(os.path.abspath(second)))


def is_contents_folder(name):
    """Tell whether a name is a build's library folder.

    Args: name is a file or folder name.
    Returns: True for names such as "_internal-12".
    """
    return CONTENTS_PATTERN.fullmatch(name) is not None


def running_contents():
    """Find the library folder the running build loaded.

    Args: None.
    Returns: PyInstaller's sys._MEIPASS as an absolute path (the
        _internal-<build> folder of a Windows or Linux build), or None from
        source.
    """
    folder = getattr(sys, "_MEIPASS", None)
    return os.path.abspath(folder) if folder else None


def is_single_file_copy(path):
    """Tell whether a file is a single-file build of this program.

    Builds up to 6 were single-file PyInstaller programs: an executable that
    ends with an archive whose table of contents lists every library as a
    binary (type code b), PySide6's among them. A folder build's executable
    lists none, and other files have no such archive.

    Args: path is the file to inspect.
    Returns: True for a single-file build of ESPI live.
    """
    if not os.path.isfile(path) or os.path.islink(path):
        return False
    try:
        with open(path, "rb") as handle:
            start = max(0, handle.seek(0, os.SEEK_END) - PYINSTALLER_TAIL_BYTES)
            handle.seek(start)
            tail = handle.read()
            found = tail.rfind(PYINSTALLER_MAGIC)
            if found < 0 or found + PYINSTALLER_COOKIE.size > len(tail):
                return False
            _, length, toc_offset, toc_length, _, _ = PYINSTALLER_COOKIE.unpack_from(tail, found)
            toc_start = start + found + PYINSTALLER_COOKIE.size - length + toc_offset
            if toc_start < 0:
                return False
            handle.seek(toc_start)
            toc = handle.read(toc_length)
    except OSError:
        return False
    position = 0
    while position + PYINSTALLER_ENTRY.size <= len(toc):
        size, _, _, _, _, kind = PYINSTALLER_ENTRY.unpack_from(toc, position)
        if size < PYINSTALLER_ENTRY.size:
            return False
        name = toc[position + PYINSTALLER_ENTRY.size:position + size]
        if kind == b"b" and name.startswith(b"PySide6"):
            return True
        position += size
    return False


def single_file_copies(target):
    """Find old single-file copies of this program beside its folder.

    A single-file build (build 6 and earlier) cannot update itself to the
    folder layout, so the new build is unpacked by hand, usually where the old
    one was: the old executable then sits in the folder above the program
    folder or, after Windows' "Extract all" (which adds a folder named after
    the archive), in the one above that. Its update backup (*.old) and
    staging folders go with it; its output/ folder is never included.

    Args: target is the running executable.
    Returns: The paths to delete.
    """
    name = os.path.basename(target)
    above = os.path.dirname(os.path.dirname(target))
    found = []
    for folder in dict.fromkeys((above, os.path.dirname(above))):
        old = os.path.join(folder, name)
        if is_single_file_copy(old):
            found.append(old)
            found += glob.glob(os.path.join(glob.escape(folder), STAGING_PREFIX + "*"))
        if is_single_file_copy(old + OLD_SUFFIX):
            found.append(old + OLD_SUFFIX)
    return found


def replace_entry(new, target):
    """Swap a file or folder into place, keeping the current one as *.old.

    A running Windows executable can be renamed but not overwritten, so the
    current one is renamed out of the way first.

    Args: new is the replacement; target is the path it takes.
    Returns: None. Raises UpdateError if the swap fails (target is put back).
    """
    backup = target + OLD_SUFFIX
    try:
        remove_path(backup)
        move(target, backup)
    except OSError as exc:
        raise UpdateError(f"cannot replace {target} ({exc}); close other "
                          "running copies and try again") from exc
    try:
        move(new, target)
    except OSError as exc:
        move(backup, target)
        raise UpdateError(f"cannot install the new build: {exc}") from exc


def move_in_library(staged, target):
    """Move a new build's _internal-<build> folder in beside the running one.

    The unpacked program folder must hold exactly the executable and one
    library folder (what CI packs), and that folder must not be the running
    build's own.

    Args: staged is the unpacked program folder; target is the installed
        executable.
    Returns: The path of the library folder now beside target. Raises
        UpdateError if the update holds anything else or the move fails.
    """
    program = os.path.basename(target)
    if os.path.isdir(staged):
        names = sorted(os.listdir(staged))
    else:
        names = [os.path.basename(staged)]
    libraries = [name for name in names if is_contents_folder(name)]
    if len(libraries) != 1 or sorted([program, *libraries]) != names:
        raise UpdateError(f"unexpected update contents {names}; expected {program} "
                          f"and one {CONTENTS_PREFIX}<build> folder")
    library = os.path.join(os.path.dirname(target), libraries[0])
    own = running_contents()
    if own and same_path(library, own):
        raise UpdateError(f"the update would replace {libraries[0]}, which the "
                          "running build uses")
    try:
        remove_path(library)  # left over from an interrupted update
        move(os.path.join(staged, libraries[0]), library)
    except OSError as exc:
        raise UpdateError(f"cannot move {libraries[0]} into place: {exc}") from exc
    return library


def install_update(staged, target=None):
    """Put the unpacked build in place, keeping the running one as *.old.

    For an executable, the new build's library folder is moved in beside the
    running build's first and the executable is swapped second; the old
    library folder stays until the next start. A .app bundle is swapped whole.

    Args: staged is the path download_update() returned; target overrides
        install_target() (tests).
    Returns: The installed path. Raises UpdateError if the swap fails (the
        running copy is left as it was).
    """
    target = target or install_target()
    if target is None:
        raise UpdateError("running from source; update with git pull")
    if target.endswith(".app"):
        replace_entry(staged, target)
    else:
        library = move_in_library(staged, target)
        try:
            replace_entry(os.path.join(staged, os.path.basename(target)), target)
        except UpdateError:
            discard(library)
            raise
        if os.name != "nt":
            os.chmod(target, 0o755)
    cleanup_previous_update(target, keep_backup=True)
    return target


def cleanup_previous_update(target=None, keep_backup=False, report=None):
    """Remove what earlier builds left behind (best effort).

    Staging folders go first, then *.old. Old library folders go only once
    *.old is gone: Windows cannot delete an executable while it runs, so a
    build that is still shutting down keeps its libraries for now. A folder
    build also removes old single-file copies beside its folder
    (single_file_copies). Recordings, frames, logs and token files are never
    touched.

    Args: target overrides install_target() (tests); keep_backup leaves *.old
        and every library folder, which are in use right after an install;
        report, if given, is called with a message for each thing removed.
    Returns: True when nothing that should go is left, False while something
        is still in use.
    """
    target = target or install_target()
    if target is None:
        return True
    folder = os.path.dirname(target)
    try:
        names = os.listdir(folder)
    except OSError:
        return True
    _, done = remove_all([os.path.join(folder, name) for name in names
                          if name.startswith(STAGING_PREFIX)])
    if keep_backup:
        return done
    removed, backup_gone = remove_all([target + OLD_SUFFIX])
    copies = []
    own = running_contents()
    if own is not None and same_path(os.path.dirname(own), folder):
        if backup_gone:
            libraries, done_libraries = remove_all([
                os.path.join(folder, name) for name in names
                if is_contents_folder(name) and not same_path(os.path.join(folder, name), own)])
            removed += libraries
            done = done and done_libraries
        copies, done_copies = remove_all(single_file_copies(target))
        done = done and done_copies
    if report and removed:
        report("Removed the previous build: "
               + ", ".join(os.path.basename(path) for path in removed))
    for path in copies:
        output = os.path.join(os.path.dirname(path), "output")
        if report and not os.path.basename(path).startswith(STAGING_PREFIX):
            report(f"Removed the old single-file copy {path}"
                   + (f"; its recordings stay in {output}" if os.path.isdir(output) else ""))
    return done and backup_gone


def relaunch(args, target=None):
    """Start the installed build as a new, independent process.

    Call after the engine has released the camera. PYINSTALLER_RESET_ENVIRONMENT
    makes the new build start as a separate program instead of inheriting this
    process's PyInstaller state.

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


class StartupCleanup:
    """Remove what earlier builds left behind, retrying while it is in use.

    Right after an update the new build starts while the old one is still
    exiting, and Windows cannot delete a running program, so the first pass
    often leaves *.old and the old library folder behind; a daemon thread
    retries for about a minute. Call stop() before installing an update, so
    a cleanup never runs during an install.
    """

    def __init__(self, report, target=None):
        """Prepare a cleanup; nothing is removed until start().

        Args: report is called with a message for each thing removed, from
            any thread; target overrides install_target() (tests).
        Returns: None.
        """
        self.report = report
        self.target = target
        self.stopping = threading.Event()
        self.thread = None

    def start(self):
        """Clean up now, and keep retrying in the background if something is in use.

        Args: None.
        Returns: True when a background retry was started.
        """
        if cleanup_previous_update(self.target, report=self.report):
            return False
        self.thread = threading.Thread(target=self._retry, name="espi-cleanup", daemon=True)
        self.thread.start()
        return True

    def _retry(self):
        """Thread body: retry until nothing is left, stop() is called or time runs out.

        Args: None.
        Returns: None.
        """
        for _ in range(CLEANUP_ATTEMPTS):
            if self.stopping.wait(CLEANUP_WAIT_S):
                return
            if cleanup_previous_update(self.target, report=self.report):
                return
        self.report("Some files of an earlier build are still in use; "
                    "they are removed at the next start")

    def stop(self):
        """Stop retrying, after a pass in progress has finished.

        Args: None.
        Returns: None.
        """
        self.stopping.set()
        if self.thread is not None:
            self.thread.join()


class UpdateWorker:
    """Run checks and downloads on a background thread for a front end.

    The front end polls ``events`` for ("checked", UpdateInfo or None),
    ("check_failed", message), ("downloaded", staged_path) and
    ("download_failed", message), and reads ``progress`` for a bar. An
    UpdateError (network, disk or GitHub) only fails the job; any other
    exception is a bug, so after posting the failure it is re-raised for
    threading.excepthook to log and report.
    """

    def __init__(self):
        """Create an idle worker.

        Args: None.
        Returns: None.
        """
        self.events = queue.Queue()
        self.progress = (0, 0)
        self.busy = False
        self.downloading = False

    def _run(self, job, ok_kind, fail_kind, downloading=False):
        """Run one job on a daemon thread and report the outcome.

        Args: job is a callable returning the result; ok_kind and fail_kind
            are the event names for success and failure; downloading marks a
            download, so the UI can show progress instead of a check.
        Returns: False if a job is already running, else True.
        """
        if self.busy:
            return False
        self.busy = True
        self.downloading = downloading

        def work():
            """Thread body: run the job and post its event.

            Args: None.
            Returns: None.
            """
            try:
                self.events.put((ok_kind, job()))
            except UpdateError as exc:
                self.events.put((fail_kind, str(exc)))
            except Exception as exc:
                self.events.put((fail_kind, f"{type(exc).__name__}: {exc}"))
                raise
            finally:
                self.downloading = False
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
                         "downloaded", "download_failed", downloading=True)
