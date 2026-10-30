"""Keep a copy of this build's source code beside a prebuilt program.

A prebuilt copy knows the commit it was built from (``_build_info.COMMIT``).
At startup ``SourceWorker`` checks that ``espi-live-source/`` beside the
program holds that commit's source and, if not, downloads GitHub's zip of the
commit (``/repos/{repo}/zipball/{commit}``, about 300 KB). The zip stays in
the copy as ``.espi/download.zip``: the untouched original that ``compare``
and ``make_patch`` measure the folder against, so a change made there (by
Codex, see ``codex.py``) can be sent for review (``changes.py``). A copy with
changes that were never sent is kept, even after an update to a newer build.
A source checkout is its own source and downloads nothing. No GUI imports:
the front ends drive this module.
"""

import difflib
import glob
import hashlib
import json
import os
import queue
import shutil
import threading
import time
import urllib.error
import zipfile

from espi_live import paths, updater

SOURCE_DIR = "espi-live-source"
STATE_DIR = ".espi"  # bookkeeping inside the copy; never part of a change
BASELINE_NAME = "download.zip"
STATE_NAME = "source.json"
SENT_NAME = "sent.json"
STAGING_PREFIX = ".espi-source-"
GUIDE_FILE = "AGENTS.override.md"  # written for Codex by codex.write_guide
REQUEST_FILE = "CHANGE_REQUEST.md"  # written by Codex before --send-changes
MAX_DOWNLOAD_BYTES = 64 << 20
# Never part of a change: bookkeeping, files written for Codex, caches and
# run-time output.
IGNORED_NAMES = {STATE_DIR, ".git", ".codex", "__pycache__", ".DS_Store", "Thumbs.db",
                 ".venv", "venv", "output", "build", "dist", ".pytest_cache",
                 ".mypy_cache"}
IGNORED_SUFFIXES = (".pyc", ".pyo")
GENERATED_FILES = {GUIDE_FILE, REQUEST_FILE}


class SourceError(Exception):
    """A source download, or a comparison with it, that could not complete."""


def is_checkout():
    """Tell whether this copy runs from a source checkout.

    Args: None.
    Returns: True from source, False in a prebuilt program.
    """
    return updater.install_target() is None


def source_folder():
    """Locate the source code of the running program.

    Args: None.
    Returns: The checkout root from source; ``espi-live-source`` beside the
        program (beside the .app on macOS) in a prebuilt copy.
    """
    if is_checkout():
        return paths.PROJECT_ROOT
    return os.path.join(paths.PROJECT_ROOT, SOURCE_DIR)


def wanted_commit():
    """Name the commit whose source this copy should hold.

    Args: None.
    Returns: The build's full commit SHA, or "" when the build lacks one.
    """
    return updater.build_value("COMMIT", "")


def bookkeeping(folder, name):
    """Build the path of one bookkeeping file inside a copy.

    Args: folder is the source copy; name is the file name.
    Returns: The path under the copy's ``.espi`` folder.
    """
    return os.path.join(folder, STATE_DIR, name)


def read_json(path, default):
    """Read a small JSON file, tolerating a missing or damaged one.

    Args: path is the file; default is returned when it cannot be read.
    Returns: The decoded value, or default.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def write_json(path, value):
    """Write a small JSON file atomically.

    Args: path is the file; value is JSON-serializable.
    Returns: None. Raises OSError on failure.
    """
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def read_state(folder):
    """Read what a downloaded copy holds.

    Args: folder is the source copy.
    Returns: A dict with at least "commit", or None when the folder is not a
        copy this module downloaded.
    """
    state = read_json(bookkeeping(folder, STATE_NAME), None)
    if isinstance(state, dict) and state.get("commit"):
        return state
    return None


def sent_records(folder):
    """List the changes already sent from a copy.

    Args: folder is the source copy.
    Returns: A list of dicts with "url", "patch" (its digest) and "time".
    """
    records = read_json(bookkeeping(folder, SENT_NAME), [])
    return [r for r in records if isinstance(r, dict) and "url" in r and "patch" in r] \
        if isinstance(records, list) else []


def record_sent(folder, url, digest):
    """Remember that a change was sent, so it is not sent twice.

    Args: folder is the source copy; url is the issue that received it;
        digest is patch_digest() of the patch.
    Returns: None. Raises OSError if the record cannot be written.
    """
    records = sent_records(folder)
    records.append({"url": url, "patch": digest,
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    write_json(bookkeeping(folder, SENT_NAME), records)


def archive_members(bundle):
    """Map each file in a GitHub source zip to its path in the project.

    GitHub puts everything under one top-level folder (owner-repo-sha7/),
    which is dropped; symbolic links are left out.

    Args: bundle is an open zipfile.ZipFile.
    Returns: A dict of relative path (with "/") to ZipInfo. Raises
        SourceError for a name that could escape the folder.
    """
    members = {}
    for info in bundle.infolist():
        if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000:
            continue
        relative = info.filename.partition("/")[2]
        parts = relative.split("/")
        if not relative or "\\" in relative or ":" in relative or \
                any(part in ("", ".", "..") for part in parts):
            raise SourceError(f"unsafe name in the source download: {info.filename!r}")
        members[relative] = info
    return members


def explain_download_error(exc, commit):
    """Turn an HTTP or network error into a SourceError message.

    Args: exc is the urllib or OS error; commit is the requested commit.
    Returns: A SourceError to raise.
    """
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 404:
            return SourceError(f"GitHub has no source for commit {commit[:7]}")
        return SourceError(f"GitHub returned HTTP {exc.code}")
    return SourceError(f"cannot reach GitHub: {getattr(exc, 'reason', exc)}")


def fetch(repo, commit, path):
    """Download GitHub's zip of one commit.

    Args: repo is "owner/name"; commit is the full SHA; path is the file to
        write.
    Returns: None. Raises SourceError on failure.
    """
    url = f"{updater.API_ROOT}/repos/{repo}/zipball/{commit}"
    try:
        with updater.open_url(url, updater.update_token(),
                              "application/vnd.github+json") as response, \
                open(path, "wb") as out:
            size = 0
            while True:
                chunk = response.read(updater.CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise SourceError("the source download is unexpectedly large")
                out.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        raise explain_download_error(exc, commit) from exc


def extract(archive, folder, commit):
    """Unpack a source zip, checking that it holds the expected commit.

    GitHub's zip comment is the commit SHA it was made from.

    Args: archive is the zip path; folder is the empty destination; commit
        is the expected full SHA.
    Returns: None. Raises SourceError if the zip is unreadable, unsafe or
        made from another commit.
    """
    try:
        with zipfile.ZipFile(archive) as bundle:
            made_from = bundle.comment.decode("ascii", "replace").strip()
            if made_from and made_from != commit:
                raise SourceError(f"GitHub sent the source of {made_from[:7]}, "
                                  f"not {commit[:7]}")
            for relative, info in archive_members(bundle).items():
                target = os.path.join(folder, *relative.split("/"))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with bundle.open(info) as source, open(target, "wb") as out:
                    shutil.copyfileobj(source, out)
    except (OSError, zipfile.BadZipFile) as exc:
        raise SourceError(f"cannot unpack the source download: {exc}") from exc


def replace_folder(new, folder, staging):
    """Put a freshly unpacked copy in place of the old one, if any.

    Args: new is the unpacked copy; folder is where it belongs; staging is
        a scratch folder on the same drive for the old copy.
    Returns: None. Raises SourceError if the folder cannot be swapped (for
        example while another program has a file in it open on Windows).
    """
    old = os.path.join(staging, "previous")
    try:
        if os.path.exists(folder):
            os.replace(folder, old)
        try:
            os.replace(new, folder)
        except OSError:
            if os.path.exists(old):
                os.replace(old, folder)
            raise
    except OSError as exc:
        raise SourceError(f"cannot replace {folder}: {exc}; close programs using it "
                          "(such as Codex) and start ESPI live again") from exc


def download_source(folder=None):
    """Download this build's source, replacing an older copy.

    Args: folder is the destination; None uses source_folder().
    Returns: The folder path. Raises SourceError on failure.
    """
    folder = folder or source_folder()
    commit, repo = wanted_commit(), updater.build_value("REPO", "")
    if not commit or not repo:
        raise SourceError("this build does not record the commit it was built from")
    parent = os.path.dirname(os.path.abspath(folder))
    for leftover in glob.glob(os.path.join(parent, STAGING_PREFIX + "*")):
        shutil.rmtree(leftover, ignore_errors=True)  # an interrupted download
    staging = os.path.join(parent, f"{STAGING_PREFIX}{os.getpid()}")
    try:
        os.makedirs(os.path.join(staging, "new", STATE_DIR))
        new = os.path.join(staging, "new")
        archive = bookkeeping(new, BASELINE_NAME)
        fetch(repo, commit, archive)
        extract(archive, new, commit)
        write_json(bookkeeping(new, STATE_NAME),
                   {"commit": commit, "repo": repo, "build": updater.current_build(),
                    "downloaded": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        replace_folder(new, folder, staging)
    except OSError as exc:
        raise SourceError(f"cannot save the source code beside the program: {exc}") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return folder


def is_ignored(relative):
    """Tell whether a file is never part of a change.

    Args: relative is the path inside the copy, with "/".
    Returns: True for bookkeeping, generated, cache and output files.
    """
    return (relative in GENERATED_FILES or relative.endswith(IGNORED_SUFFIXES)
            or any(part in IGNORED_NAMES for part in relative.split("/")))


def working_files(folder):
    """List the files in a copy that could be part of a change.

    Args: folder is the source copy.
    Returns: A dict of relative path (with "/") to absolute path.
    """
    files = {}
    for root, dirs, names in os.walk(folder):
        dirs[:] = [name for name in dirs if name not in IGNORED_NAMES]
        prefix = os.path.relpath(root, folder).replace(os.sep, "/")
        for name in names:
            relative = name if prefix == "." else f"{prefix}/{name}"
            path = os.path.join(root, name)
            if not is_ignored(relative) and not os.path.islink(path):
                files[relative] = path
    return files


def is_text(data):
    """Tell whether file contents can go into a text patch.

    Args: data is the file's bytes.
    Returns: True for UTF-8 without NUL bytes.
    """
    if b"\0" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def match_line_endings(old, new):
    """Give edited text the line endings of the original.

    Editors (and Codex on Windows) may rewrite every line ending, which would
    make the whole file look changed.

    Args: old and new are the original and current bytes of a text file.
    Returns: new with LF endings when old has only LF, with CRLF endings
        when old has only CRLF, otherwise unchanged.
    """
    if b"\r\n" not in old:
        return new.replace(b"\r\n", b"\n")
    if old.count(b"\n") == old.count(b"\r\n"):
        return new.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return new


def compare(folder):
    """Find the files that differ from the original download.

    Args: folder is a downloaded source copy.
    Returns: A sorted list of (relative path, original bytes or None, current
        bytes or None). Raises SourceError if the original is unreadable.
    """
    try:
        with zipfile.ZipFile(bookkeeping(folder, BASELINE_NAME)) as bundle:
            original = {relative: bundle.read(info)
                        for relative, info in archive_members(bundle).items()
                        if not is_ignored(relative)}
    except (OSError, zipfile.BadZipFile) as exc:
        raise SourceError(f"cannot read the original download: {exc}") from exc
    current = working_files(folder)
    changes = []
    for relative in sorted(set(original) | set(current)):
        old = original.get(relative)
        new = None
        if relative in current:
            try:
                with open(current[relative], "rb") as handle:
                    new = handle.read()
            except OSError as exc:
                raise SourceError(f"cannot read {relative}: {exc}") from exc
        if old is not None and new is not None and is_text(old) and is_text(new):
            new = match_line_endings(old, new)
        if old != new:
            changes.append((relative, old, new))
    return changes


def change_kind(old, new):
    """Name the kind of change to one file.

    Args: old and new are the original and current bytes, or None.
    Returns: "added", "deleted" or "modified".
    """
    if old is None:
        return "added"
    return "deleted" if new is None else "modified"


def split_lines(text):
    """Split text into lines the way git does (only at LF).

    Args: text is the file contents.
    Returns: A list of lines with their "\\n"; the last one lacks it when
        the file does not end with a line break.
    """
    lines = text.split("\n")
    tail = lines.pop()
    return [line + "\n" for line in lines] + ([tail] if tail else [])


def make_patch(changes):
    """Write changes as a git-style unified diff that ``git apply`` accepts.

    Args: changes is the list compare() returns.
    Returns: (patch text, list of paths left out because they are binary).
    """
    pieces, skipped = [], []
    for relative, old, new in changes:
        if any(data is not None and not is_text(data) for data in (old, new)):
            skipped.append(relative)
            continue
        kind = change_kind(old, new)
        pieces.append(f"diff --git a/{relative} b/{relative}\n")
        if kind != "modified":
            pieces.append(f"{'new' if kind == 'added' else 'deleted'} file mode 100644\n")
        before = split_lines(old.decode("utf-8")) if old is not None else []
        after = split_lines(new.decode("utf-8")) if new is not None else []
        for line in difflib.unified_diff(
                before, after, "/dev/null" if old is None else f"a/{relative}",
                "/dev/null" if new is None else f"b/{relative}"):
            pieces.append(line if line.endswith("\n")
                          else line + "\n\\ No newline at end of file\n")
    return "".join(pieces), skipped


def patch_digest(patch):
    """Fingerprint a patch, to recognize one that was already sent.

    Args: patch is the patch text.
    Returns: The SHA-256 hex digest.
    """
    return hashlib.sha256(patch.encode("utf-8")).hexdigest()


def has_unsent_changes(folder):
    """Tell whether replacing a copy would lose work.

    Args: folder is a downloaded source copy.
    Returns: True when it differs from its download in a way that was never
        sent (binary changes, which cannot be sent, always count).
    """
    patch, skipped = make_patch(compare(folder))
    sent = {record["patch"] for record in sent_records(folder)}
    return bool(skipped) or (bool(patch) and patch_digest(patch) not in sent)


def check(folder=None):
    """Decide whether the source copy can be used as it is.

    Args: folder is the copy; None uses source_folder().
    Returns: (action, detail): ("ready", note) when it can be opened (note
        is "" or a line for the log), ("download", reason) when this build's
        source must be fetched, or ("unavailable", reason).
    """
    folder = folder or source_folder()
    if is_checkout():
        return "ready", ""
    commit = wanted_commit()
    if not commit or not updater.build_value("REPO", ""):
        return "unavailable", "this build does not record the commit it was built from"
    state = read_state(folder)
    if state is None:
        if os.path.exists(folder):
            return "unavailable", (f"{folder} was not made by ESPI live; rename it so "
                                   "the source code can be downloaded there")
        return "download", "the source code has not been downloaded yet"
    if state["commit"] == commit:
        return "ready", ""
    try:
        keep = has_unsent_changes(folder)
    except SourceError:
        keep = False  # a damaged copy: replace it
    if keep:
        return "ready", (f"Kept the source code of commit {state['commit'][:7]} in {folder}: "
                         "it holds changes that were not sent to the developers")
    return "download", f"the source code is of commit {state['commit'][:7]}, not this build's"


def prepare(folder=None):
    """Make sure the source copy is there, downloading it when needed.

    Args: folder is the copy; None uses source_folder().
    Returns: (folder, note): note is "" or a line for the log. Raises
        SourceError when the source cannot be provided.
    """
    folder = folder or source_folder()
    action, detail = check(folder)
    if action == "unavailable":
        raise SourceError(detail)
    if action == "download":
        download_source(folder)
        return folder, f"Downloaded the source code of commit {wanted_commit()[:7]} to {folder}"
    return folder, detail


class SourceWorker:
    """Check for, and download, the source copy on a background thread.

    The front end polls ``events`` for ("ready", (folder, note)) and
    ("failed", message).
    """

    def __init__(self):
        """Create an idle worker.

        Args: None.
        Returns: None.
        """
        self.events = queue.Queue()
        self.busy = False

    def start(self):
        """Start checking (and if needed downloading) the source copy.

        Args: None.
        Returns: False if a check is already running, else True.
        """
        if self.busy:
            return False
        self.busy = True

        def work():
            """Thread body: prepare the copy and post its event.

            Args: None.
            Returns: None.
            """
            try:
                self.events.put(("ready", prepare()))
            except Exception as exc:
                self.events.put(("failed", str(exc)))
            finally:
                self.busy = False

        threading.Thread(target=work, name="espi-source", daemon=True).start()
        return True
