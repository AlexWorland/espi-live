"""Upload run logs to the developer's private logs repository.

Installed builds send every run log in ``output/logs/`` to the private
repository ``AlexWorland/espi-live-logs`` as
``logs/<YYYY-MM-DD>/<HHMMSS>-<machine>.log`` (the run's start, from the log
file name; ``<machine>`` is a random ID kept in ``machine-id.txt``, so one PC
can be followed across runs without its host name). A pass runs at startup
and every INTERVAL_S: it sends logs that were never sent or have grown since
(the current run, or a run that crashed before its last pass), updating the
same file in place. Nothing runs at shutdown; the next start sends the rest.

Builds embed ``LOGS_TOKEN``, a fine-grained token with only Contents: read &
write on that one repository. Tokens in the public binaries are extractable,
so whoever extracts it could read the uploaded logs, though never touch the
code; that is why every log is scrubbed first (make_log_scrubber: the error
reporter's rules, every camera serial this install has seen, kept in
``known-serials.json``, and camera labels and line prefixes that carry a
serial), and why each upload is capped at MAX_BYTES (the start, with the
startup header, plus the newest part). Scrubbing runs a chunk of lines at a
time so the camera and UI threads get the interpreter between chunks. The
remote file's ``sha`` comes from its folder's listing, which works for files
of any size (the file endpoint refuses files over 1 MB in JSON).

Uploads are silent: failures are logged at DEBUG and never reported, and the
worker never blocks another thread. They follow the error-report switches
(``--no-error-reports``, ``ESPI_ERROR_REPORTS`` and the Help menu). No GUI
imports: the front end drives this module.
"""

import base64
import json
import logging
import os
import re
import secrets
import threading

from espi_live import feedback, updater
from espi_live.error_reports import make_scrubber, write_json
from espi_live.feedback import FeedbackError
from espi_live.logs import run_logs

logger = logging.getLogger(__name__)

DEFAULT_REPO = "AlexWorland/espi-live-logs"
REPO_ENV = "ESPI_LOGS_REPO"
TOKEN_ENV = "ESPI_LOGS_TOKEN"
TOKEN_FILE = "logs-token.txt"
BRANCH = "main"
RECORD_NAME = "log_uploads.json"
MACHINE_FILE = "machine-id.txt"
SERIALS_NAME = "known-serials.json"
CHUNK_CHARS = 64 * 1024
INTERVAL_S = 15 * 60
MAX_BYTES = 5 * 1024 * 1024
HEAD_BYTES = 256 * 1024
MAX_PER_PASS = 10
WORKER_THREAD = "log-uploads"
RUN_NAME = re.compile(r"^espi_(\d{4})(\d{2})(\d{2})_(\d{6})_\d+\.log$")
# CameraInfo.label "Model (21234567)" and the "[21234567] " prefix of
# several-camera log lines, for serials this install has not recorded yet.
LABEL_SERIAL = re.compile(r"\(\d{6,10}\)")
PREFIX_SERIAL = re.compile(r"(\] )\[\d{6,10}\](?= )")


def logs_token():
    """Choose the token that may write to the logs repository.

    Args: None.
    Returns: The token, or "" when this copy cannot upload.
    """
    return updater.read_token(TOKEN_ENV, TOKEN_FILE, "LOGS_TOKEN")


def repository():
    """Name the repository that receives the logs.

    Args: None.
    Returns: "owner/name" from ESPI_LOGS_REPO, the build, or DEFAULT_REPO.
    """
    return (os.environ.get(REPO_ENV, "").strip()
            or updater.build_value("LOGS_REPO", "") or DEFAULT_REPO)


def machine_id(folder):
    """Read this computer's random ID, creating it the first time.

    Args: folder is the logs folder that keeps the ID.
    Returns: Eight hex digits, or "unknown" when it cannot be stored.
    """
    path = os.path.join(folder, MACHINE_FILE)
    try:
        with open(path, encoding="utf-8") as handle:
            value = handle.read().strip()
        if re.fullmatch(r"[0-9a-f]{8}", value):
            return value
    except OSError:
        pass
    value = secrets.token_hex(4)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(value + "\n")
    except OSError:
        return "unknown"
    return value


def known_serials(folder):
    """Read the camera serial numbers this install has seen.

    Args: folder is the logs folder.
    Returns: A sorted list; empty when the file is missing or unreadable.
    """
    try:
        with open(os.path.join(folder, SERIALS_NAME), encoding="utf-8") as handle:
            values = json.load(handle)
    except (OSError, ValueError):
        return []
    return sorted({str(value) for value in values if value}) if isinstance(values, list) else []


def remember_serials(folder, serials):
    """Add this run's camera serial numbers to the ones seen before.

    Args: folder is the logs folder; serials are this run's serial numbers.
    Returns: Every serial known, sorted (also when the file cannot be written).
    """
    known = known_serials(folder)
    merged = sorted(set(known) | {str(serial) for serial in serials if serial})
    if merged != known:
        try:
            write_json(os.path.join(folder, SERIALS_NAME), merged)
        except OSError:
            pass
    return merged


def make_log_scrubber(serials):
    """Build the scrubbing function for run logs.

    Args: serials are every camera serial number to hide.
    Returns: A function taking text and returning it with the error
        reporter's personal details and every camera serial replaced.
    """
    scrub = make_scrubber(serials)

    def scrub_log(text):
        """Apply the rules.

        Args: text is log text.
        Returns: The scrubbed text.
        """
        text = LABEL_SERIAL.sub("(<serial>)", scrub(text))
        return PREFIX_SERIAL.sub(r"\1[<serial>]", text)

    return scrub_log


def scrub_in_chunks(text, scrub, chunk_chars=CHUNK_CHARS):
    """Scrub a large text a chunk of whole lines at a time.

    One regular-expression pass over megabytes holds the interpreter lock
    for most of a second; chunks let the camera and UI threads run between them.

    Args: text is the log text; scrub is the scrubbing function; chunk_chars
        is the rough chunk size.
    Returns: The scrubbed text.
    """
    parts, chunk, size = [], [], 0
    for line in text.splitlines(keepends=True):
        chunk.append(line)
        size += len(line)
        if size >= chunk_chars:
            parts.append(scrub("".join(chunk)))
            chunk, size = [], 0
    if chunk:
        parts.append(scrub("".join(chunk)))
    return "".join(parts)


def remote_path(log_name, machine):
    """Place a run log in the logs repository.

    Args: log_name is a run log file name such as
        "espi_20261113_142530_000001.log"; machine is the machine ID.
    Returns: "logs/2026-11-13/142530-<machine>.log", or None for other names.
    """
    match = RUN_NAME.match(log_name)
    if not match:
        return None
    year, month, day, clock = match.groups()
    return f"logs/{year}-{month}-{day}/{clock}-{machine}.log"


def read_for_upload(path, max_bytes=MAX_BYTES, head_bytes=HEAD_BYTES):
    """Read a log, keeping its start and its newest part when it is too long.

    Args: path is the log file; max_bytes caps the upload; head_bytes is how
        much of the start (the startup header) to keep when capping.
    Returns: (text, size): the text to upload and how many bytes were read.
        Raises OSError when the file cannot be read.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    size = len(data)
    if size > max_bytes:
        left_out = size - max_bytes
        data = (data[:head_bytes] + f"\n[... {left_out} bytes left out ...]\n".encode()
                + data[size - (max_bytes - head_bytes):])
    return data.decode("utf-8", errors="replace"), size


class LogUploader:
    """Send the logs folder's new or grown run logs, remembering what was sent."""

    def __init__(self, folder, serials, enabled, request=feedback.github_request, token=None):
        """Create the uploader.

        Args: folder is the logs folder; serials returns this run's camera
            serial numbers (ErrorReporter.serials); enabled returns whether
            uploads are on; request is the GitHub call; token overrides
            logs_token() (tests).
        Returns: None.
        """
        self.folder = folder
        self.serials = serials
        self.enabled = enabled
        self.request = request
        self._token = token

    def token(self):
        """Choose the token for this pass.

        Args: None.
        Returns: The token, or "" when there is none.
        """
        return logs_token() if self._token is None else self._token

    def _record_path(self):
        """Locate the record of what was sent.

        Args: None.
        Returns: The record file path.
        """
        return os.path.join(self.folder, RECORD_NAME)

    def _read_record(self):
        """Read what earlier passes sent.

        Args: None.
        Returns: {log name: {"size": bytes sent, "path": remote path}}; empty
            when the record is missing or unreadable.
        """
        try:
            with open(self._record_path(), encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError):
            return {}
        return record if isinstance(record, dict) else {}

    def send_pending(self):
        """Upload the run logs that are new or have grown since they were sent.

        Args: None.
        Returns: How many logs were uploaded. Stops quietly at the first failure.
        """
        token = self.token()
        if not self.enabled() or not token:
            return 0
        record = self._read_record()
        machine = machine_id(self.folder)
        scrub = make_log_scrubber(remember_serials(self.folder, self.serials()))
        sent = 0
        for name in run_logs(self.folder):
            if sent >= MAX_PER_PASS:
                break
            path = remote_path(name, machine)
            local = os.path.join(self.folder, name)
            try:
                size = os.path.getsize(local)
            except OSError:
                continue
            if path is None or record.get(name, {}).get("size") == size:
                continue
            if not self.enabled():
                break
            try:
                size = self._upload(local, path, token, scrub)
                record[name] = {"size": size, "path": path}
                write_json(self._record_path(), record)
            except (FeedbackError, OSError) as exc:
                logger.debug("Log upload stopped at %s: %s", name, exc)
                break
            sent += 1
            logger.debug("Uploaded %s as %s", name, path)
        return sent

    def _remote_sha(self, path, token):
        """Find the sha of a file already in the logs repository.

        Lists the file's folder: the listing names every file's sha whatever
        its size, where the file endpoint refuses files over 1 MB.

        Args: path is the file's place in the repository; token the logs token.
        Returns: The sha, or None when the file (or its folder) is not there.
            Raises FeedbackError on other failures.
        """
        folder, name = path.rsplit("/", 1)
        try:
            entries = self.request(
                "GET", f"/repos/{repository()}/contents/{folder}?ref={BRANCH}", token)
        except FeedbackError as exc:
            if exc.status == 404:
                return None
            raise
        for entry in entries:
            if entry.get("name") == name:
                return entry.get("sha")
        return None

    def _upload(self, local, path, token, scrub):
        """Create or update one log in the logs repository.

        Args: local is the log file; path its place in the repository; token
            the logs token; scrub the scrubbing function.
        Returns: How many bytes of the file were sent. Raises FeedbackError
            or OSError on failure.
        """
        text, size = read_for_upload(local)
        url = f"/repos/{repository()}/contents/{path}"
        payload = {"message": f"Log {path}", "branch": BRANCH,
                   "content": base64.b64encode(
                       scrub_in_chunks(text, scrub).encode("utf-8")).decode("ascii")}
        sha = self._remote_sha(path, token)
        if sha:
            payload["sha"] = sha
        self.request("PUT", url, token, payload)
        return size


class LogUploadWorker:
    """Run upload passes on a background thread: at start, on kick, and every interval."""

    def __init__(self, uploader, interval_s=INTERVAL_S):
        """Create an idle worker.

        Args: uploader is the LogUploader; interval_s is the time between passes.
        Returns: None.
        """
        self.uploader = uploader
        self.interval_s = interval_s
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.thread = None

    def start(self):
        """Start the thread; the first pass runs at once.

        Args: None.
        Returns: None.
        """
        self.wake.set()
        self.thread = threading.Thread(target=self._run, name=WORKER_THREAD, daemon=True)
        self.thread.start()

    def kick(self):
        """Ask for a pass now (for example after uploads were turned back on).

        Args: None.
        Returns: None.
        """
        self.wake.set()

    def stop(self):
        """Ask the thread to exit after the upload in progress (it is not joined).

        Args: None.
        Returns: None.
        """
        self.stopping.set()
        self.wake.set()

    def _run(self):
        """Thread body: wait for a kick or the interval, then run a pass.

        Args: None.
        Returns: None.
        """
        while True:
            self.wake.wait(self.interval_s)
            self.wake.clear()
            if self.stopping.is_set():
                return
            try:
                self.uploader.send_pending()
            except Exception:
                logger.debug("A log upload pass failed", exc_info=True)
