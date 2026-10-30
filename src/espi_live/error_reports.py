"""Report unexpected errors as deduplicated GitHub issues.

Every capture point calls ``capture`` (or ``caught``, which also logs):
uncaught exceptions on the main thread (``sys.excepthook``, which also sees
Qt slots), on other threads (``threading.excepthook``) and in finalizers
(``sys.unraisablehook``); the engine worker stopping (its "fatal" event);
recorder failures; Qt critical and fatal messages; and native crashes, whose
faulthandler file from the previous run becomes a report at the next start.
Expected operating conditions (EXPECTED) are logged at WARNING and never
reported.

Capture never blocks: it writes the report to
``output/logs/pending_reports/<fingerprint>.json`` and returns. An
ErrorReportWorker thread sends pending reports at startup and after each
capture, so a crash that ends the process is still sent on the next launch.

Each error has a deterministic fingerprint, ``ESPI-ERR-v1-<12 hex>``: the
first 12 hex digits of SHA-256 over ``v1|<type>|<cause type>|<frames>[|code]``.
Frames are the innermost 8 in-app ``module:function`` names, so the same bug
gives the same ID on every machine, install path and run; line numbers,
message text, thread names, timestamps and errno are left out on purpose.
Changing what is hashed means a new version prefix; never change what v1
hashes. Before filing, the worker lists the repository's ``auto-error``
issues (the list endpoint, which unlike search is strongly consistent) and
matches the ``[ESPI-ERR-v1-...]`` title prefix exactly; see ErrorReporter.deliver
for what it does with an existing issue.

The repository is public, so everything sent is scrubbed (scrub): home
folder, user and host names, PyInstaller folders, the program and output
folders, saved file names, camera serials, e-mail addresses and token-like
strings. Reports use the Issues-only FEEDBACK_TOKEN; without it the lookup
runs anonymously and the user is offered a prefilled new-issue page. No GUI
imports: the front end drives this module.
"""

import dataclasses
import getpass
import hashlib
import json
import logging
import os
import platform
import queue
import re
import socket
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from espi_live import feedback, paths, updater
from espi_live.logs import CRASH_SUFFIX, LOG_SUFFIX, library_versions

logger = logging.getLogger(__name__)

FINGERPRINT_PREFIX = "ESPI-ERR-v1-"
FINGERPRINT_DIGITS = 12
PACKAGE = "espi_live"
IN_APP_FRAMES = 8
OTHER_FRAMES = 3
# List, dict and set comprehensions have their own frame before Python 3.12
# (PEP 709 inlined them), so they are skipped to keep fingerprints the same
# on the 3.10 and 3.12 builds.
INLINED_FRAMES = {"<listcomp>", "<dictcomp>", "<setcomp>"}
REPORT_LABEL = "auto-error"
BUG_LABEL = "bug"
AREA_LABELS = {"camera": "area: camera", "camera_models": "area: camera",
               "qt_app": "area: ui", "app": "area: ui", "updater": "area: updates"}
ENABLE_ENV = "ESPI_ERROR_REPORTS"
SETTING_KEY = "errorReports"  # QSettings key behind Help > Send error reports automatically
PENDING_FOLDER = "pending_reports"
LEDGER_NAME = "error_reports.json"
MAX_PENDING = 20
PENDING_MAX_AGE_S = 30 * 24 * 3600
MAX_DELIVERIES_PER_RUN = 5
MAX_PAGES = 20
BODY_LIMIT = 60000  # GitHub allows 65,536 characters; keep some headroom
MESSAGE_LIMIT = 2000
BROWSER_URL_LIMIT = 7000
WORKER_THREAD = "espi-error-reports"
FOOTER = "_Filed automatically by ESPI live's error reporter._"
TURN_OFF_HINT = ("Turn automatic error reports off with Help > Send error reports "
                 "automatically, --no-error-reports or ESPI_ERROR_REPORTS=0.")

# Expected operating conditions: logged at WARNING, never reported. Keys are
# capture points; values are the exception type names (matched against the
# whole class hierarchy) that are expected there.
CAMERA_ERRORS = ("PySpin.SpinnakerException", "RuntimeError", "ImportError", "OSError")
EXPECTED = {
    "camera_startup": CAMERA_ERRORS,  # no camera at startup
    "camera_open": CAMERA_ERRORS,     # camera still in use or unplugged ("open_failed")
    "switch_live": CAMERA_ERRORS,     # dummy mode could not switch ("switch_failed")
    "update": ("espi_live.updater.UpdateError",),        # network, disk or GitHub errors
    "feedback": ("espi_live.feedback.FeedbackError",),   # network or GitHub errors
    "save": ("OSError",),             # saving a frame to a full or read-only disk
    "recorder": ("OSError",),         # recording to a full or read-only disk
}
CAPTURE_POINTS = {
    "main_thread": "uncaught exception on the main thread",
    "thread": "uncaught exception on a background thread",
    "unraisable": "exception in a finalizer",
    "engine": "engine worker stopped",
    "recorder": "recording thread failed",
    "qt_message": "Qt critical or fatal message",
    "native_crash": "native crash in the previous run",
}
POSIX_SIGNALS = {"segmentation fault": "SIGSEGV", "floating-point exception": "SIGFPE",
                 "floating point exception": "SIGFPE", "aborted": "SIGABRT",
                 "bus error": "SIGBUS", "illegal instruction": "SIGILL"}

_reporter = None  # the ErrorReporter that capture() uses; set by install_hooks()


class ReportError(Exception):
    """A report that could not be delivered now; it stays pending."""


# ----- fingerprint -------------------------------------------------------------


def path_parts(filename):
    """Split a path on both Windows and POSIX separators.

    Args: filename is a path from a traceback or a crash file.
    Returns: The non-empty path components.
    """
    return [part for part in re.split(r"[\\/]+", filename or "") if part]


def in_app_module(filename):
    """Name the espi_live module a source file belongs to, wherever it is installed.

    The module is the path after the last ``espi_live`` component, so
    ``C:\\Users\\lab\\ESPI\\_internal\\espi_live\\engine.py``,
    ``/tmp/_MEIxxxx/espi_live/engine.py`` and ``src/espi_live/engine.py``
    all give ``espi_live.engine``.

    Args: filename is the frame's source path.
    Returns: The dotted module name, or None for files outside the package.
    """
    parts = path_parts(filename)
    starts = [index for index, part in enumerate(parts) if part == PACKAGE]
    if not starts or not parts[-1].endswith(".py") or starts[-1] == len(parts) - 1:
        return None
    names = parts[starts[-1] + 1:]
    names[-1] = names[-1][:-3]
    if names[-1] == "__init__":
        names.pop()
    return ".".join([PACKAGE, *names])


def collapse(names):
    """Merge consecutive repeats (recursion) into one entry.

    Args: names is a list of frame names.
    Returns: The list without consecutive duplicates.
    """
    merged = []
    for name in names:
        if not merged or merged[-1] != name:
            merged.append(name)
    return merged


def frame_signature(frames):
    """Describe a stack by the code it ran, not by where it is installed.

    Args: frames is a list of (filename, function) pairs, outermost first.
    Returns: The innermost IN_APP_FRAMES in-app frames as "module:function"
        joined by ">", or with no in-app frame the innermost OTHER_FRAMES as
        "file name:function"; "" for an empty stack.
    """
    frames = [(name, function) for name, function in frames
              if function not in INLINED_FRAMES]
    in_app = [f"{in_app_module(name)}:{function}" for name, function in frames
              if in_app_module(name)]
    if in_app:
        return ">".join(collapse(in_app)[-IN_APP_FRAMES:])
    other = [f"{(path_parts(name) or ['?'])[-1]}:{function}" for name, function in frames]
    return ">".join(collapse(other)[-OTHER_FRAMES:])


def type_name(cls):
    """Name an exception class the same way in every build.

    Args: cls is the exception class.
    Returns: "ValueError" for builtins, "PySpin.SpinnakerException" for the
        SDK (whatever submodule defines it), otherwise "module.QualName".
    """
    module = getattr(cls, "__module__", "") or ""
    name = getattr(cls, "__qualname__", cls.__name__)
    if module in ("builtins", ""):
        return name
    if module.split(".")[0] == "PySpin":
        module = "PySpin"
    return f"{module}.{name}"


def stable_code(exc):
    """Find an error code that separates different failures with the same stack.

    Args: exc is the exception.
    Returns: A SpinnakerException's error code as text, or "".
    """
    if not type_name(type(exc)).startswith("PySpin."):
        return ""
    code = getattr(exc, "errorcode", None)
    return "" if code is None else str(code)


def canonical_string(kind, cause, frames, code=""):
    """Build the text a fingerprint hashes.

    Args: kind is the exception type name (or "native:SIGSEGV", "qt:critical");
        cause is the type name of __cause__, or ""; frames is the
        frame_signature; code is an optional stable code.
    Returns: "v1|<kind>|<cause>|<frames>" plus "|<code>" when there is one.
    """
    text = f"v1|{kind}|{cause}|{frames}"
    return f"{text}|{code}" if code else text


def fingerprint_of(canonical):
    """Hash a canonical string into an error ID.

    Args: canonical is the canonical_string.
    Returns: "ESPI-ERR-v1-" plus the first 12 hex digits of its SHA-256.
    """
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return FINGERPRINT_PREFIX + digest[:FINGERPRINT_DIGITS]


def exception_frames(exc):
    """List the frames an exception's traceback passed through.

    Args: exc is the exception.
    Returns: (filename, function) pairs, outermost first.
    """
    return [(frame.f_code.co_filename, frame.f_code.co_name)
            for frame, _ in traceback.walk_tb(exc.__traceback__)]


def exception_canonical(exc):
    """Build the canonical string for an exception.

    Args: exc is the exception (with its traceback).
    Returns: The canonical_string.
    """
    cause = type_name(type(exc.__cause__)) if exc.__cause__ is not None else ""
    return canonical_string(type_name(type(exc)), cause,
                            frame_signature(exception_frames(exc)), stable_code(exc))


def fingerprint(exc):
    """Give an exception its deterministic error ID.

    Args: exc is the exception (with its traceback).
    Returns: "ESPI-ERR-v1-<12 hex digits>".
    """
    return fingerprint_of(exception_canonical(exc))


def message_signature(message):
    """Reduce a Qt message to its fixed wording, for the fingerprint.

    Numbers, addresses, quoted names and paths vary between machines and runs,
    so they are replaced by placeholders.

    Args: message is the Qt message text.
    Returns: At most 120 characters of normalized text.
    """
    text = re.sub(r"\"[^\"]*\"|'[^']*'", '""', message)
    text = re.sub(r"\S*[\\/]\S*", "<path>", text)
    text = re.sub(r"0x[0-9a-fA-F]+", "0x", text)
    text = re.sub(r"\d+", "#", text)
    return " ".join(text.split())[:120]


def parse_crash(text):
    """Read a faulthandler crash dump.

    Args: text is the content of a .crash file.
    Returns: (kind, frames) with kind such as "native:SIGSEGV" and frames as
        (filename, function) pairs of the crashing thread, outermost first;
        None when the text is not a crash.
    """
    found = re.search(r"^Fatal Python error: (.+)$", text, re.M)
    windows = re.search(r"^Windows fatal exception: (.+)$", text, re.M)
    if windows:
        name = windows.group(1)
    elif found:
        name = found.group(1).split(":")[0].strip()
        name = POSIX_SIGNALS.get(name.lower(), name)
    else:
        return None
    kind = "native:" + (re.sub(r"\W+", "_", name).strip("_").upper() or "UNKNOWN")
    block = (re.search(r"^Current thread [^\n]*:\n((?:[ \t][^\n]*\n?)*)", text, re.M)
             or re.search(r"^(?:Thread|Stack)[^\n]*\(most recent call first\):\n"
                          r"((?:[ \t][^\n]*\n?)*)", text, re.M))
    lines = block.group(1).splitlines() if block else []
    frames = [match.groups() for match in
              (re.match(r'\s+File "(.*)", line \d+ in (\S+)', line) for line in lines)
              if match]
    return kind, frames[::-1]


# ----- scrubbing ---------------------------------------------------------------


GENERIC_NAMES = {"root", "admin", "administrator", "user", "runner", "espi", "live", "lab"}
TOKEN_PATTERN = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,})")
BEARER_PATTERN = re.compile(r"(?i)\b(bearer|token)(\s+|=)[A-Za-z0-9_.\-]{16,}")
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
BUNDLE_PATTERN = re.compile(r"(?:~|[A-Za-z]:)?(?:[\\/][^\\/\s\"'<>|]+)*[\\/]_MEI\w+")
SAVED_FILE_PATTERN = re.compile(r"((?:frames|recordings)[\\/])[^\s\"'<>|]+")
SERIAL_PATTERN = re.compile(r"(?i)((?:serial(?: number)?|S/N)\s*[:=#]?\s*)([A-Za-z0-9-]{4,})")


def personal_names():
    """Map this machine's user and host names to their placeholders.

    Args: None.
    Returns: {name: "<user>" or "<host>"} for names of 3+ characters that are
        not generic words.
    """
    users = {os.environ.get(key, "") for key in ("USERNAME", "USER", "LOGNAME")}
    users.add(os.path.basename(os.path.expanduser("~")))
    try:
        users.add(getpass.getuser())
    except Exception:  # no user database entry (some containers)
        pass
    hosts = {socket.gethostname(), platform.node()}
    hosts |= {host.split(".")[0] for host in hosts}
    names = {name: "<user>" for name in users}
    names.update({name: "<host>" for name in hosts})
    return {name: label for name, label in names.items()
            if len(name) >= 3 and name.lower() not in GENERIC_NAMES}


def folder_replacements():
    """List the folders that are replaced by placeholders, longest first.

    Args: None.
    Returns: (path, placeholder) pairs.
    """
    folders = [(paths.OUTPUT_DIR, "<output>"), (paths.PROJECT_ROOT, "<program>"),
               (getattr(sys, "_MEIPASS", ""), "<bundle>"),
               (os.path.expanduser("~"), "~")]
    return sorted(((folder, label) for folder, label in folders
                   if folder and len(folder) > 3), key=lambda pair: -len(pair[0]))


def make_scrubber(serials=()):
    """Build the scrubbing rules for this machine once.

    Args: serials are camera serial numbers to hide.
    Returns: A function taking text and returning it with folders, names,
        serials, e-mail addresses and token-like strings replaced by
        placeholders.
    """
    rules = []
    for folder, label in folder_replacements():
        for variant in sorted({folder, folder.replace("\\", "/"), folder.replace("/", "\\")}):
            rules.append((re.compile(re.escape(variant), re.IGNORECASE), label))
    rules += [(BUNDLE_PATTERN, "<bundle>"), (SAVED_FILE_PATTERN, r"\1<file>"),
              (TOKEN_PATTERN, "<token>"), (BEARER_PATTERN, r"\1\2<token>"),
              (EMAIL_PATTERN, "<email>")]
    for serial in sorted({str(serial) for serial in serials if serial}, key=len, reverse=True):
        rules.append((re.compile(rf"(?<!\w){re.escape(serial)}(?!\w)"), "<serial>"))
    rules.append((SERIAL_PATTERN, r"\1<serial>"))
    for name, label in sorted(personal_names().items(), key=lambda item: -len(item[0])):
        rules.append((re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE), label))

    def scrub_text(text):
        """Apply the rules.

        Args: text is any report text.
        Returns: The scrubbed text.
        """
        for pattern, replacement in rules:
            text = pattern.sub(lambda match, replacement=replacement:
                               match.expand(replacement), text)
        return text

    return scrub_text


def scrub(text, serials=()):
    """Remove personal and machine-specific details from public report text.

    Args: text is any report text; serials are camera serial numbers to hide.
    Returns: The scrubbed text (see make_scrubber).
    """
    return make_scrubber(serials)(text)


def display_path(filename):
    """Shorten a traceback path to what matters for debugging.

    Args: filename is the frame's source path.
    Returns: "espi_live/engine.py" for the package, the part after
        site-packages for libraries, otherwise the file name.
    """
    module = in_app_module(filename)
    if module:
        return module.replace(".", "/") + (
            "/__init__.py" if path_parts(filename)[-1] == "__init__.py" else ".py")
    parts = path_parts(filename)
    if "site-packages" in parts:
        return "/".join(parts[len(parts) - parts[::-1].index("site-packages"):])
    return parts[-1] if parts else filename


def relative_traceback(text):
    """Rewrite every 'File "..."' path in a traceback with display_path.

    Args: text is traceback or faulthandler text.
    Returns: The text with shortened paths.
    """
    return re.sub(r'File "([^"]+)"',
                  lambda match: f'File "{display_path(match.group(1))}"', text)


def fence(text):
    """Make text safe inside a Markdown code block.

    Args: text is the block content.
    Returns: The text with triple backticks broken up.
    """
    return text.replace("```", "'''")


# ----- reports -----------------------------------------------------------------


@dataclasses.dataclass
class ErrorReport:
    """One captured error, as saved in pending_reports/ and sent to GitHub."""

    fingerprint: str
    canonical: str
    title: str
    error: str
    where: str
    fatal: bool
    area: str
    body: str
    seen: str
    build: int
    commit: str
    committed_at: str
    created: float

    @property
    def labels(self):
        """List the labels for a new issue.

        Args: None.
        Returns: ["bug", "auto-error"] plus the area label, if any.
        """
        return [BUG_LABEL, REPORT_LABEL] + ([self.area] if self.area else [])


@dataclasses.dataclass(frozen=True)
class Outcome:
    """What happened to a report, for the log and the UI."""

    fingerprint: str
    kind: str  # created, commented, reopened, already, fixed, declined, browser, saved
    number: int = 0
    url: str = ""
    fatal: bool = False
    message: str = ""


def outcome_text(outcome):
    """Describe a report's outcome in one line for the user.

    Args: outcome is an Outcome, or None while the report is being sent.
    Returns: Plain text such as "Reported as #123".
    """
    if outcome is None:
        return "Reporting this error…"
    return {
        "created": f"Reported as #{outcome.number}",
        "commented": f"Already reported as #{outcome.number}",
        "already": f"Already reported as #{outcome.number}",
        "reopened": f"Reported again as #{outcome.number} (reopened)",
        "fixed": "Fixed in a newer build; update to get the fix",
        "declined": f"Already known as #{outcome.number}",
        "browser": "Not sent automatically: report it in your browser",
        "saved": "Report saved; it will be sent next time",
    }.get(outcome.kind, outcome.kind)


def is_expected(exc, where):
    """Tell whether an exception is an expected operating condition.

    Args: exc is the exception; where is the capture point name.
    Returns: True when EXPECTED lists the exception's type (or a base class)
        for that capture point.
    """
    names = EXPECTED.get(where, ())
    return any(type_name(cls) in names for cls in type(exc).__mro__)


def parse_time(text):
    """Read an ISO 8601 time from GitHub or git.

    Args: text is a string such as "2026-10-29T23:21:01Z", or "".
    Returns: An aware datetime, or None when text is empty or unreadable.
    """
    try:
        value = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def resolve_enabled(cli_off=False, setting=True, frozen=None):
    """Decide whether reports are sent, and why not when something forces them off.

    Release builds report by default; source checkouts only with
    ESPI_ERROR_REPORTS=1, since developers see the traceback anyway.

    Args: cli_off is --no-error-reports; setting is the user's Help menu
        choice; frozen marks a PyInstaller build (None: detect).
    Returns: (enabled, reason) where reason explains a forced "off" that the
        menu cannot change, else None.
    """
    if frozen is None:
        frozen = bool(getattr(sys, "frozen", False))
    value = os.environ.get(ENABLE_ENV, "").strip()
    if cli_off:
        return False, "turned off with --no-error-reports"
    if value == "0":
        return False, "turned off with ESPI_ERROR_REPORTS=0"
    if value != "1" and not frozen:
        return False, "off in source checkouts; set ESPI_ERROR_REPORTS=1 to send them"
    return bool(setting), None


def write_json(path, data):
    """Write a JSON file atomically (temporary file, then rename).

    Args: path is the destination; data is JSON-serializable.
    Returns: None. Raises OSError on failure.
    """
    handle, temporary = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=1)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise


class GitHub:
    """The few issue endpoints the reporter needs."""

    def __init__(self, repo, token):
        """Create a client.

        Args: repo is "owner/name"; token is "" for anonymous reads.
        Returns: None.
        """
        self.repo = repo
        self.token = token

    def call(self, method, path, payload=None, query=None):
        """Make one API request.

        Args: method is the HTTP verb; path follows /repos/<repo>; payload is
            a JSON body or None; query is a dict of URL parameters or None.
        Returns: The decoded JSON reply. Raises ReportError on failure.
        """
        url = f"{updater.API_ROOT}/repos/{self.repo}{path}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        headers = {"Accept": "application/vnd.github+json",
                   "User-Agent": "espi-live-error-reports",
                   "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=updater.TIMEOUT_S,
                                        context=updater.ssl_context()) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise ReportError(f"GitHub returned HTTP {exc.code} for {method} {path}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise ReportError(f"cannot reach GitHub: {getattr(exc, 'reason', exc)}") from exc
        except ValueError as exc:
            raise ReportError(f"unreadable reply from GitHub: {exc}") from exc

    def reports(self):
        """List every issue labelled auto-error, open or closed.

        Args: None.
        Returns: Issue dicts (pull requests left out).
        """
        issues = []
        for page in range(1, MAX_PAGES + 1):
            batch = self.call("GET", "/issues", query={
                "labels": REPORT_LABEL, "state": "all", "per_page": 100, "page": page})
            issues += [issue for issue in batch if "pull_request" not in issue]
            if len(batch) < 100:
                break
        return issues

    def matching(self, fingerprint):
        """Find the issues filed for a fingerprint.

        Args: fingerprint is the error ID.
        Returns: Matching issue dicts, lowest number first.
        """
        prefix = f"[{fingerprint}]"
        return sorted((issue for issue in self.reports()
                       if issue.get("title", "").startswith(prefix)),
                      key=lambda issue: issue["number"])

    def create(self, report):
        """Open an issue for a report.

        Args: report is the ErrorReport.
        Returns: The new issue dict.
        """
        return self.call("POST", "/issues", {"title": report.title, "body": report.body,
                                             "labels": report.labels})

    def comment(self, number, body):
        """Comment on an issue.

        Args: number is the issue number; body is the Markdown text.
        Returns: The comment dict.
        """
        return self.call("POST", f"/issues/{number}/comments", {"body": body})

    def edit(self, number, **fields):
        """Change an issue's state.

        Args: number is the issue number; fields are the PATCH body.
        Returns: The updated issue dict.
        """
        return self.call("PATCH", f"/issues/{number}", fields)


class ErrorReporter:
    """Capture errors into pending reports and deliver them to GitHub."""

    def __init__(self, enabled=False, folder=None, log_tail=None, uptime=None):
        """Create a reporter.

        Args: enabled sends reports (see resolve_enabled); folder is the logs
            folder (None: output/logs); log_tail returns the recent log lines;
            uptime returns the run's age in seconds.
        Returns: None.
        """
        self.enabled = enabled
        self.folder = folder
        self.log_tail = log_tail or (lambda: [])
        self.uptime = uptime or (lambda: 0.0)
        self.context = {}  # front_end, display, source, cameras, serials, spinnaker
        self.listeners = []  # called with each new report (the worker's kick)
        self.lock = threading.Lock()
        self.reports = {}  # fingerprint -> ErrorReport captured this run
        self.delivered = set()  # fingerprints sent (or tried) this run
        self.announced = False
        self.local = threading.local()

    # ----- capture (any thread) --------------------------------------------------

    def update_context(self, **fields):
        """Record facts about the session for later reports.

        Args: fields are context keys: front_end, display, source ("camera
            <model>" or "dummy mode"), cameras (count), serials (to scrub),
            spinnaker (SDK version).
        Returns: None.
        """
        with self.lock:
            self.context.update(fields)

    def capture(self, exc, where, fatal=False):
        """Turn an unexpected exception into a pending report.

        Args: exc is the exception (with its traceback); where is the capture
            point; fatal marks an error that stopped a camera or the program.
        Returns: The ErrorReport (also for a repeat of an error already
            captured this run), or None when reporting is off, the error is
            expected, or the reporter itself failed (that is only logged).
        """
        def describe():
            """Describe the exception for _capture.

            Args: None.
            Returns: (canonical, error line, traceback pieces, frames).
            """
            error = f"{type_name(type(exc))}: {exc}".rstrip(": ")
            return (exception_canonical(exc), error,
                    traceback.format_exception(type(exc), exc, exc.__traceback__),
                    exception_frames(exc))

        if is_expected(exc, where):
            return None
        return self._capture(describe, where, fatal)

    def capture_message(self, level, message, frames, stack_text=""):
        """Turn a Qt critical or fatal message into a pending report.

        The fingerprint uses the message's fixed wording (message_signature)
        as its stable code, since a Qt message has no exception type.

        Args: level is "critical" or "fatal"; message is Qt's text; frames
            are the Python (filename, function) pairs active when Qt
            complained, outermost first; stack_text is that stack, formatted.
        Returns: The ErrorReport or None (see capture).
        """
        def describe():
            """Describe the message for _capture.

            Args: None.
            Returns: (canonical, error line, traceback pieces, frames).
            """
            signature = message_signature(self.scrubber()(message))
            canonical = canonical_string(f"qt:{level}", "", frame_signature(frames), signature)
            return canonical, f"Qt {level}: {message}", [stack_text] if stack_text else [], frames

        return self._capture(describe, "qt_message", level == "fatal")

    def capture_crash(self, text, log_lines=None):
        """Turn a native crash dump from an earlier run into a pending report.

        Args: text is the faulthandler output; log_lines are that run's last
            log lines (None: this run's).
        Returns: The ErrorReport, or None when text holds no crash (see capture).
        """
        parsed = parse_crash(text)
        if parsed is None:
            return None
        kind, frames = parsed

        def describe():
            """Describe the crash for _capture.

            Args: None.
            Returns: (canonical, error line, traceback pieces, frames).
            """
            return (canonical_string(kind, "", frame_signature(frames)),
                    f"Native crash ({kind[len('native:'):]})", [text], frames)

        return self._capture(describe, "native_crash", True, log_lines)

    def scrubber(self):
        """Build the scrubbing function for the current session.

        Args: None.
        Returns: A make_scrubber function hiding the session's camera serials.
        """
        with self.lock:
            serials = tuple(self.context.get("serials", ()))
        return make_scrubber(serials)

    def _capture(self, describe, where, fatal, log_lines=None):
        """Build, save and announce a report once per fingerprint and run.

        Args: describe returns (canonical, error, traceback pieces, frames);
            where and fatal describe the capture; log_lines override the log
            tail.
        Returns: The ErrorReport, or None (see capture).
        """
        if not self.enabled or getattr(self.local, "busy", False):
            return None
        self.local.busy = True
        try:
            canonical, error, lines, _ = describe()
            key = fingerprint_of(canonical)
            with self.lock:
                if key in self.reports:
                    return self.reports[key]
            report = self.build(canonical, error, lines, where, fatal,
                                self.log_tail() if log_lines is None else log_lines)
            self.save_pending(report)
            with self.lock:
                self.reports[key] = report
            for listener in list(self.listeners):
                listener(report)
            return report
        except Exception:
            logger.exception("The error reporter failed; nothing was reported")
            return None
        finally:
            self.local.busy = False

    def build(self, canonical, error, lines, where, fatal, log_lines):
        """Assemble a report: title, scrubbed body and the "seen again" note.

        Args: canonical is the canonical_string; error the one-line error;
            lines the traceback pieces; where and fatal describe the capture;
            log_lines are the log lines to attach.
        Returns: An ErrorReport.
        """
        with self.lock:
            context = dict(self.context)
        clean = self.scrubber()
        key = fingerprint_of(canonical)
        kind = canonical.split("|")[1]
        signature = canonical.split("|")[3]
        place = signature.split(">")[-1] if signature else "native code"
        module = place.split(":")[0]
        area = AREA_LABELS.get(module[len(PACKAGE) + 1:], "") if module.startswith(
            PACKAGE + ".") else ""
        error = clean(error)[:MESSAGE_LIMIT]
        build = updater.current_build()
        commit = updater.build_value("COMMIT", "")
        committed_at = updater.build_value("COMMITTED_AT", "")
        source = context.get("source", "")
        facts = [
            f"- Build: {updater.describe_build()}"
            + (f", committed {committed_at}" if committed_at else ""),
            f"- Platform: {platform.platform()} ({platform.machine()})",
            "- Libraries: " + ", ".join(self._libraries(context)),
            f"- Front end: {context.get('front_end', '?')}"
            + (f", display {context['display']}" if context.get("display") else ""),
            f"- Running with: {source or 'unknown'}",
            f"- Cameras: {context.get('cameras', '?')}",
            f"- Thread: {threading.current_thread().name}",
            f"- Captured by: {CAPTURE_POINTS.get(where, where)}"
            + (" (fatal)" if fatal else ""),
            f"- Uptime: {int(self.uptime())} s"]
        traceback_parts = [clean(relative_traceback(part)) for part in lines]
        log_lines = [clean(line) for line in log_lines]
        body = self.body(key, canonical, error, facts, traceback_parts, log_lines)
        seen = (f"Seen again in build {build} ({commit[:7] or 'source'}) on "
                f"{platform.system()} {platform.release()} ({platform.machine()}), "
                f"{context.get('front_end', '?')} {context.get('display', '')}".rstrip()
                + f", {source or 'unknown source'}:\n\n```text\n{fence(error)}\n```")
        return ErrorReport(
            fingerprint=key, canonical=canonical,
            title=f"[{key}] {kind} in {place}", error=error, where=where, fatal=fatal,
            area=area, body=body, seen=seen, build=build, commit=commit,
            committed_at=committed_at, created=time.time())

    @staticmethod
    def _libraries(context):
        """List library versions for a report.

        Args: context is the session context.
        Returns: Version strings, with Spinnaker's when known.
        """
        versions = library_versions()
        if context.get("spinnaker"):
            versions.append(f"Spinnaker {context['spinnaker']}")
        return versions

    @staticmethod
    def body(key, canonical, error, facts, traceback_parts, log_lines):
        """Write the issue body, trimmed below GitHub's size limit.

        The log is trimmed first (oldest lines), then the traceback's outer
        frames.

        Args: key is the fingerprint; canonical its canonical string; error
            the one-line error; facts the context bullets; traceback_parts
            the scrubbed traceback pieces; log_lines the scrubbed log tail.
        Returns: The Markdown body.
        """
        log_lines = list(log_lines)
        parts = list(traceback_parts)
        trimmed = False

        def render():
            """Render the body from the current parts.

            Args: None.
            Returns: The Markdown text.
            """
            frames = "".join(parts)
            return "\n".join([
                "```text", fence(error), "```", "",
                f"**Error ID:** `{key}`  ",
                f"SHA-256 of `{fence(canonical)}`, first {FINGERPRINT_DIGITS} hex digits.",
                "", "### Traceback", "", "```text", fence(frames.rstrip("\n")), "```", "",
                "### Context", "", *facts, "",
                f"### Last {len(log_lines)} log lines", "", "```text",
                fence("\n".join(log_lines)), "```", "", FOOTER])

        text = render()
        while len(text) > BODY_LIMIT and log_lines:
            log_lines = log_lines[max(1, len(log_lines) // 4):]
            text = render()
        while len(text) > BODY_LIMIT:
            frame = next((index for index, part in enumerate(parts)
                          if part.startswith("  File ")), None)
            if frame is None:  # one long piece, such as a crash dump
                share = BODY_LIMIT // (2 * len(parts))
                parts = [part[:share] + "\n... (trimmed)\n" for part in parts]
                return render()
            parts[frame] = "" if trimmed else "  ... (outer frames trimmed)\n"
            trimmed = True
            text = render()
        return text

    # ----- pending reports and the ledger ------------------------------------------

    def logs_folder(self):
        """Find the logs folder.

        Args: None.
        Returns: The folder path. Raises OSError if it cannot be created.
        """
        return self.folder or paths.output_folder("logs")

    def pending_folder(self):
        """Find (and create) the pending-reports folder.

        Args: None.
        Returns: The folder path. Raises OSError if it cannot be created.
        """
        folder = os.path.join(self.logs_folder(), PENDING_FOLDER)
        os.makedirs(folder, exist_ok=True)
        return folder

    def save_pending(self, report):
        """Write a report to pending_reports/<fingerprint>.json atomically.

        Args: report is the ErrorReport.
        Returns: None. Raises OSError on failure.
        """
        folder = self.pending_folder()
        write_json(os.path.join(folder, f"{report.fingerprint}.json"),
                   dataclasses.asdict(report))
        self.prune_pending()

    def prune_pending(self, now=None):
        """Drop pending reports older than 30 days or beyond the newest 20.

        Args: now is the current time.time() (tests).
        Returns: None.
        """
        folder = self.pending_folder()
        now = time.time() if now is None else now
        entries = []
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            try:
                entries.append((os.path.getmtime(path), path))
            except OSError:
                continue
        entries.sort(reverse=True)
        for index, (modified, path) in enumerate(entries):
            if index >= MAX_PENDING or now - modified > PENDING_MAX_AGE_S:
                try:
                    os.remove(path)
                except OSError:
                    pass

    def load_pending(self):
        """Read the pending reports, oldest first; unreadable files are dropped.

        Args: None.
        Returns: A list of (path, ErrorReport).
        """
        folder = self.pending_folder()
        found = []
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(folder, name)
            try:
                with open(path, encoding="utf-8") as handle:
                    found.append((path, ErrorReport(**json.load(handle))))
            except (OSError, ValueError, TypeError) as exc:
                logger.warning(f"Dropping unreadable error report {name}: {exc}")
                try:
                    os.remove(path)
                except OSError:
                    pass
        return sorted(found, key=lambda item: item[1].created)

    def load_ledger(self):
        """Read the record of what this install has reported.

        Args: None.
        Returns: {"reports": {fingerprint: {"issue", "url", "builds"}}}.
        """
        try:
            with open(os.path.join(self.logs_folder(), LEDGER_NAME), encoding="utf-8") as handle:
                ledger = json.load(handle)
        except (OSError, ValueError):
            ledger = {}
        if not isinstance(ledger, dict) or not isinstance(ledger.get("reports"), dict):
            ledger = {"reports": {}}
        return ledger

    def save_ledger(self, ledger):
        """Write the ledger atomically (best effort).

        Args: ledger is the dict from load_ledger.
        Returns: None.
        """
        try:
            write_json(os.path.join(self.logs_folder(), LEDGER_NAME), ledger)
        except OSError as exc:
            logger.warning(f"Could not save the error-report ledger: {exc}")

    # ----- native crashes --------------------------------------------------------

    def collect_native_crashes(self, own_crash_path=None):
        """Report crash files that earlier runs left behind.

        A non-empty .crash file means a run died in native code. Its dump is
        appended to that run's log (so the log is complete) and the file is
        deleted; when reporting is on it also becomes a pending report.

        Args: own_crash_path is this run's crash file, which is skipped.
        Returns: The ErrorReports created.
        """
        try:
            folder = self.logs_folder()
            names = sorted(os.listdir(folder))
        except OSError:
            return []
        reports = []
        for name in names:
            path = os.path.join(folder, name)
            if not name.endswith(CRASH_SUFFIX) or (
                    own_crash_path and os.path.abspath(path) == os.path.abspath(own_crash_path)):
                continue
            try:
                with open(path, encoding="utf-8", errors="replace") as handle:
                    text = handle.read()
            except OSError:
                continue
            if not text.strip():
                continue  # no crash, or a copy that is still running
            log_path = path[:-len(CRASH_SUFFIX)] + LOG_SUFFIX
            log_lines = None
            try:
                with open(log_path, encoding="utf-8", errors="replace") as handle:
                    log_lines = handle.read().splitlines()[-40:]
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write("\n--- Native crash (faulthandler) ---\n" + text)
                os.remove(path)
            except OSError as exc:
                logger.warning(f"Could not file the crash dump {name}: {exc}")
            logger.warning(f"The run logged in {os.path.basename(log_path)} crashed "
                           "in native code")
            report = self.capture_crash(text, log_lines or [])
            if report is not None:
                reports.append(report)
        return reports

    # ----- delivery (worker thread) ----------------------------------------------------

    def send_pending(self, post):
        """Deliver pending reports, within the per-run limits.

        Args: post is called with ("outcome", Outcome) for each report handled.
        Returns: None.
        """
        if not self.enabled:
            return
        pending = self.load_pending()
        if not pending:
            return
        token = feedback.feedback_token()
        github = GitHub(feedback.repository(), token)
        ledger = self.load_ledger()
        for path, report in pending:
            if report.fingerprint in self.delivered:
                continue
            if len(self.delivered) >= MAX_DELIVERIES_PER_RUN:
                break
            self.delivered.add(report.fingerprint)
            try:
                outcome = self.deliver(report, github, ledger)
            except ReportError as exc:
                logger.warning(f"Error report {report.fingerprint} not sent: {exc}; "
                               "it will be sent next time")
                outcome = Outcome(report.fingerprint, "saved", fatal=report.fatal,
                                  message=str(exc))
            else:
                self.save_ledger(ledger)
                try:
                    os.remove(path)
                except OSError:
                    pass
                self._log_outcome(report, outcome)
            post(("outcome", outcome))

    def _log_outcome(self, report, outcome):
        """Log what was done with a report; the first post also says how to opt out.

        Args: report is the ErrorReport; outcome its Outcome.
        Returns: None.
        """
        if outcome.kind == "browser":
            logger.info(f"Error {report.fingerprint} was not reported: this copy has "
                        f"no feedback token. Report it in a browser: {outcome.url}")
            return
        logger.info(f"Error {report.fingerprint}: {outcome_text(outcome)}"
                    + (f" ({outcome.url})" if outcome.url else ""))
        if outcome.kind in ("created", "commented", "reopened") and not self.announced:
            self.announced = True
            logger.info(f"Sent an automatic error report to {outcome.url}. {TURN_OFF_HINT}")

    def deliver(self, report, github, ledger):
        """Look the report up on GitHub and file, comment, reopen or skip it.

        | Existing issue (lowest number with the fingerprint) | Action |
        | None | create it (no token: offer the browser) |
        | Open | one "seen again" comment per build from this install |
        | Closed as completed, build committed after closed_at | reopen + comment |
        | Closed as completed, older build | nothing: fixed in a newer build |
        | Closed as not planned or duplicate | nothing |

        Args: report is the ErrorReport; github is the GitHub client; ledger
            is the install's record, updated in place.
        Returns: An Outcome. Raises ReportError when GitHub cannot be used.
        """
        entry = ledger["reports"].setdefault(report.fingerprint, {"builds": []})
        matches = github.matching(report.fingerprint)
        if not matches:
            if not github.token:
                return Outcome(report.fingerprint, "browser", url=browser_url(report),
                               fatal=report.fatal)
            issue = self._settle_race(github, report, github.create(report))
            entry.update(issue=issue["number"], url=issue["html_url"])
            entry["builds"] = sorted(set(entry["builds"]) | {report.build})
            return Outcome(report.fingerprint, "created", issue["number"],
                           issue["html_url"], report.fatal)
        issue = matches[0]
        number, url = issue["number"], issue["html_url"]
        entry.update(issue=number, url=url)

        def outcome(kind):
            """Build an Outcome for this issue.

            Args: kind is the outcome kind.
            Returns: The Outcome.
            """
            return Outcome(report.fingerprint, kind, number, url, report.fatal)

        if issue.get("state") == "open":
            if report.build in entry["builds"] or not github.token:
                return outcome("already")
            github.comment(number, report.seen)
            entry["builds"] = sorted(set(entry["builds"]) | {report.build})
            return outcome("commented")
        if (issue.get("state_reason") or "completed") != "completed":
            return outcome("declined")
        committed, closed = parse_time(report.committed_at), parse_time(issue.get("closed_at") or "")
        if committed is None or closed is None or committed <= closed:
            return outcome("fixed")
        if not github.token:
            return Outcome(report.fingerprint, "browser", number, url, report.fatal)
        github.edit(number, state="open", state_reason="reopened")
        github.comment(number, f"Seen again in build {report.build}, built after this was "
                               f"closed.\n\n{report.seen}")
        entry["builds"] = sorted(set(entry["builds"]) | {report.build})
        return outcome("reopened")

    @staticmethod
    def _settle_race(github, report, created):
        """Close a just-created issue if another copy filed the same error first.

        Two machines can both find no issue and both create one. Listing
        again, every copy keeps the lowest-numbered open issue and closes its
        own if it is not that one, so they converge on the same issue.

        Args: github is the client; report the ErrorReport; created the issue
            just created.
        Returns: The issue to use (the lowest-numbered open one).
        """
        open_matches = [issue for issue in github.matching(report.fingerprint)
                        if issue.get("state") == "open"]
        if not open_matches or open_matches[0]["number"] == created["number"]:
            return created
        first = open_matches[0]
        try:
            github.comment(created["number"], f"Duplicate of #{first['number']}")
            github.edit(created["number"], state="closed", state_reason="duplicate",
                        duplicate_issue_id=first["id"])
        except ReportError as exc:  # the next lookup still settles on the first one
            logger.warning(f"Could not close #{created['number']} as a duplicate of "
                           f"#{first['number']}: {exc}")
        return first


def browser_url(report):
    """Build a prefilled new-issue page for a report, for copies without a token.

    The body loses the log and as much of the traceback as needed to keep the
    URL short enough for browsers.

    Args: report is the ErrorReport.
    Returns: The https URL.
    """
    head = report.body.split("### Last ")[0].rstrip() + "\n\n" + FOOTER

    def url(body):
        """Encode one candidate URL.

        Args: body is the issue body.
        Returns: The URL.
        """
        return (f"https://github.com/{feedback.repository()}/issues/new?"
                + urllib.parse.urlencode({"title": report.title, "body": body,
                                          "labels": ",".join(report.labels)}))

    body = head
    while len(url(body)) > BROWSER_URL_LIMIT and len(body) > 200:
        body = body[:len(body) // 2].rstrip() + "\n…(trimmed)\n\n" + FOOTER
    return url(body)


class ErrorReportWorker:
    """Send pending reports on a background thread for a front end.

    The front end polls ``events`` for ("outcome", Outcome); the worker never
    touches the GUI. An exception inside the reporter is logged, never reported.
    """

    def __init__(self, reporter):
        """Create an idle worker.

        Args: reporter is the ErrorReporter.
        Returns: None.
        """
        self.reporter = reporter
        self.events = queue.Queue()
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.thread = None

    def start(self):
        """Start the thread and send what earlier runs left pending.

        Args: None.
        Returns: None.
        """
        self.wake.set()
        self.thread = threading.Thread(target=self._run, name=WORKER_THREAD, daemon=True)
        self.thread.start()

    def kick(self, report=None):
        """Ask the thread to send pending reports now.

        Args: report is the report just captured (unused; listener signature).
        Returns: None.
        """
        self.wake.set()

    def stop(self):
        """Ask the thread to exit after the report in progress (it is not joined).

        Args: None.
        Returns: None.
        """
        self.stopping.set()
        self.wake.set()

    def _run(self):
        """Thread body: wait for a kick, then deliver pending reports.

        Args: None.
        Returns: None.
        """
        while True:
            self.wake.wait()
            self.wake.clear()
            if self.stopping.is_set():
                return
            try:
                self.reporter.send_pending(self.events.put)
            except Exception:
                logger.exception("The error reporter failed; reports stay pending")


def current_reporter():
    """Return the reporter capture() uses.

    Args: None.
    Returns: The installed ErrorReporter, or None.
    """
    return _reporter


def capture(exc, where, fatal=False):
    """Report an unexpected exception through the installed reporter.

    Args: exc is the exception; where is the capture point; fatal marks an
        error that stopped a camera or the program.
    Returns: The ErrorReport, or None (no reporter, reporting off, expected).
    """
    return _reporter.capture(exc, where, fatal) if _reporter is not None else None


def caught(log, exc, where, message, fatal=False):
    """Log a caught exception at the right level and report it if unexpected.

    Args: log has warning() and error() (an AppLog or PrefixedLog); exc is the
        exception; where is the capture point; message starts the log line.
    Returns: The ErrorReport, or None for expected conditions (logged at
        WARNING) and when nothing is reported.
    """
    if is_expected(exc, where):
        log.warning(f"{message}: {exc}")
        return None
    log.error(f"{message}: {type(exc).__name__}: {exc}", exc_info=exc)
    return capture(exc, where, fatal)


class Hooks:
    """Python's process-wide error hooks, pointed at a reporter."""

    def __init__(self, reporter):
        """Prepare the hooks; nothing changes until install().

        Args: reporter is the ErrorReporter.
        Returns: None.
        """
        self.reporter = reporter
        self.saved = None

    def install(self):
        """Replace sys.excepthook, threading.excepthook and sys.unraisablehook.

        Args: None.
        Returns: This object, for uninstall().
        """
        global _reporter
        self.saved = (sys.excepthook, threading.excepthook, sys.unraisablehook, _reporter)
        sys.excepthook = self.excepthook
        threading.excepthook = self.thread_excepthook
        sys.unraisablehook = self.unraisablehook
        _reporter = self.reporter
        return self

    def uninstall(self):
        """Put the previous hooks back.

        Args: None.
        Returns: None.
        """
        global _reporter
        if self.saved is not None:
            sys.excepthook, threading.excepthook, sys.unraisablehook, _reporter = self.saved
            self.saved = None

    def excepthook(self, exc_type, exc, tb):
        """Log and report an uncaught main-thread exception (including Qt slots).

        Args: exc_type, exc and tb describe the exception.
        Returns: None.
        """
        if issubclass(exc_type, KeyboardInterrupt):
            self.saved[0](exc_type, exc, tb)
            return
        if exc.__traceback__ is None:
            exc = exc.with_traceback(tb)
        logger.error(f"Unexpected error: {exc_type.__name__}: {exc}",
                     exc_info=(exc_type, exc, tb))
        self.reporter.capture(exc, "main_thread")

    def thread_excepthook(self, args):
        """Log and report an uncaught exception on another thread.

        Args: args is threading.ExceptHookArgs.
        Returns: None.
        """
        if issubclass(args.exc_type, SystemExit):
            return
        name = args.thread.name if args.thread is not None else "?"
        logger.error(f"Unexpected error in thread {name}: {args.exc_type.__name__}: "
                     f"{args.exc_value}",
                     exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        if name != WORKER_THREAD and args.exc_value is not None:
            self.reporter.capture(args.exc_value, "thread")

    def unraisablehook(self, args):
        """Log and report an exception Python could not raise (finalizers, __del__).

        Args: args is the sys.UnraisableHookArgs.
        Returns: None.
        """
        message = args.err_msg or "Exception ignored"
        logger.error(f"{message}: {args.object!r}",
                     exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        if args.exc_value is not None:
            self.reporter.capture(args.exc_value, "unraisable")


def install_hooks(reporter):
    """Point Python's error hooks and capture() at a reporter.

    Args: reporter is the ErrorReporter.
    Returns: The installed Hooks; call uninstall() to restore the previous ones.
    """
    return Hooks(reporter).install()
