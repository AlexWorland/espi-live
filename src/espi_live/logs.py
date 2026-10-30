"""The run log: levels, the per-run file, console mirroring and crash files.

AppLog is built on the standard ``logging`` module. While it runs, the
``espi_live`` logger (and so every module's ``logging.getLogger(__name__)``)
writes ``HH:MM:SS LEVEL [thread] message`` lines to three destinations: the
per-run file ``output/logs/espi_<timestamp>.log``, the queue the GUI log dock
drains, and the console. ``AppLog.write(message)`` is an INFO shim for the
older call sites and PrefixedLog. stdout and stderr stay mirrored into the
same file and queue, because PySpin and other C libraries print directly.

Native crashes (a segfault in PySpin, OpenCV or the GL driver) cannot be
caught in Python, so faulthandler writes them to ``espi_<timestamp>.crash``;
a crash file left non-empty by a run that never closed becomes an error report
on the next start (see error_reports.collect_native_crashes). Only the newest
KEEP_LOGS run logs are kept. No GUI imports: the front end drives this module.
"""

import collections
import faulthandler
import logging
import os
import platform
import queue
import sys
import threading
import time

from espi_live import updater
from espi_live.paths import output_folder, timestamp

LOGGER_NAME = "espi_live"
LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING}
LEVEL_ENV = "ESPI_LOG_LEVEL"
DEFAULT_LEVEL = "info"
LINE_FORMAT = "%(asctime)s %(levelname)s [%(threadName)s] %(message)s"
TIME_FORMAT = "%H:%M:%S"
LOG_PREFIX = "espi_"
LOG_SUFFIX = ".log"
CRASH_SUFFIX = ".crash"
KEEP_LOGS = 50
TAIL_LINES = 40  # recent lines kept in memory for error reports
WATCHDOG_S = 10.0  # dump every thread's stack when the UI has not ticked for this long


class RunLock:
    """An OS-held lock that marks a run active until it closes or dies."""

    def __init__(self, log_path):
        """Prepare a lock beside a run log.

        Args: log_path is the run's .log file path.
        Returns: None.
        """
        self.path = log_path + ".lock"
        self.handle = None

    def acquire(self):
        """Try to own the lock without waiting for another process.

        Args: None.
        Returns: True on success; False if busy or inaccessible.
        """
        try:
            self.handle = open(self.path, "a+b")
            if os.name == "nt":
                import msvcrt
                self.handle.write(b"\0")
                self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.close()
            return False
        return True

    def close(self):
        """Release the lock, leaving its marker available for later probes.

        Args: None.
        Returns: None.
        """
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def run_is_active(log_path):
    """Check a run's lock, conservatively protecting inaccessible markers.

    Args: log_path is the run's .log file path.
    Returns: True if another process holds the lock or it cannot be checked.
    """
    if not os.path.exists(log_path + ".lock"):
        return False  # older versions did not create markers
    lock = RunLock(log_path)
    try:
        return not lock.acquire()
    finally:
        lock.close()


def resolve_level(name=None):
    """Choose the log level from an option, ESPI_LOG_LEVEL or the default.

    Args: name is the --log-level value or None.
    Returns: A logging level number (DEBUG, INFO or WARNING); unknown names
        fall through to the next source.
    """
    for candidate in (name, os.environ.get(LEVEL_ENV, "")):
        key = (candidate or "").strip().lower()
        if key in LEVELS:
            return LEVELS[key]
    return LEVELS[DEFAULT_LEVEL]


def library_versions():
    """Name the versions of the libraries this run has loaded.

    Only modules that are already imported are listed, so this never loads
    PySpin or Qt as a side effect.

    Args: None.
    Returns: A list such as ["Python 3.10.11", "NumPy 2.2.6", "OpenCV 4.11.0"].
    """
    versions = [f"Python {platform.python_version()}"]
    for module, name in (("numpy", "NumPy"), ("cv2", "OpenCV"),
                         ("PySide6", "PySide6"), ("PySpin", "PySpin")):
        version = getattr(sys.modules.get(module), "__version__", None)
        if version:
            versions.append(f"{name} {version}")
    return versions


def header_lines(front_end, level):
    """Describe this copy and its platform for the top of every log.

    Args: front_end names the user interface (for example "Qt"); level is
        the logging level number in use.
    Returns: A list of lines.
    """
    committed = updater.build_value("COMMITTED_AT", "")
    return [updater.describe_build() + (f", committed {committed}" if committed else ""),
            f"Platform: {platform.platform()} ({platform.machine()})",
            "Libraries: " + ", ".join(library_versions()),
            f"Front end: {front_end}; log level {logging.getLevelName(level)}"]


def run_logs(folder):
    """List the run logs in a folder, oldest first.

    Args: folder is the logs folder.
    Returns: File names such as "espi_20261029_101500_000000.log"; the
        timestamp in the name sorts them by start time.
    """
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted(name for name in names
                  if name.startswith(LOG_PREFIX) and name.endswith(LOG_SUFFIX))


def prune_logs(folder, keep=KEEP_LOGS):
    """Delete all but the newest run logs and their crash files (best effort).

    Args: folder is the logs folder; keep is how many run logs to keep.
    Returns: The number of run logs removed.
    """
    names = run_logs(folder)
    removed = 0
    for name in names[:max(0, len(names) - keep)]:
        stem = os.path.join(folder, name[:-len(LOG_SUFFIX)])
        if run_is_active(stem + LOG_SUFFIX):
            continue
        for path in (stem + LOG_SUFFIX, stem + CRASH_SUFFIX, stem + LOG_SUFFIX + ".lock"):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError:
                continue  # in use by another running copy; next time
        removed += 1
    return removed


class LogSink:
    """The run's text destinations behind one lock: file, GUI queue and tail."""

    def __init__(self, log_file=None):
        """Create a sink.

        Args: log_file is an open text file, or None for no file.
        Returns: None.
        """
        self.lock = threading.RLock()
        self.log_file = log_file
        self.queue = queue.SimpleQueue()
        self.tail = collections.deque(maxlen=TAIL_LINES)
        self.partial = ""

    def put(self, text):
        """Send text to the log file, the GUI queue and the recent-lines tail.

        Args: text is any chunk of output, with or without newlines.
        Returns: None. A failing log file is dropped, not raised.
        """
        if not text:
            return
        with self.lock:
            if self.log_file is not None:
                try:
                    self.log_file.write(text)
                except (OSError, ValueError):
                    self.log_file = None
            self.queue.put(text)
            lines = (self.partial + text).split("\n")
            self.partial = lines.pop()
            self.tail.extend(lines)

    def flush(self):
        """Flush the log file.

        Args: None.
        Returns: None.
        """
        with self.lock:
            if self.log_file is not None:
                try:
                    self.log_file.flush()
                except (OSError, ValueError):
                    self.log_file = None

    def recent(self):
        """Return the newest lines written.

        Args: None.
        Returns: Up to TAIL_LINES lines, oldest first.
        """
        with self.lock:
            lines = list(self.tail) + ([self.partial] if self.partial else [])
        return lines[-TAIL_LINES:]


class MirrorStream:
    """Copy Python console output to the console and the log sink."""

    def __init__(self, original, sink):
        """Prepare a console mirror.

        Args: original is the real stdout or stderr (None in windowed
            builds); sink is the LogSink that also receives the text.
        Returns: None.
        """
        self.original = original
        self.sink = sink

    def write(self, value):
        """Forward a text chunk to the console and the sink.

        Args: value is the text to write.
        Returns: Number of characters accepted.
        """
        with self.sink.lock:
            if self.original is not None:
                self.original.write(value)
            self.sink.put(value)
        return len(value)

    def flush(self):
        """Flush the underlying console and log file.

        Args: None.
        Returns: None.
        """
        with self.sink.lock:
            if self.original is not None:
                self.original.flush()
            self.sink.flush()

    def isatty(self):
        """Report whether the underlying console is interactive.

        Args: None.
        Returns: The original stream's terminal flag.
        """
        return self.original.isatty() if self.original is not None else False

    def __getattr__(self, name):
        """Expose other original-stream attributes to console users.

        Args: name is the requested attribute name.
        Returns: The attribute from the underlying stream.
        """
        if self.original is None:
            raise AttributeError(name)
        return getattr(self.original, name)


class SinkHandler(logging.Handler):
    """Write formatted records to the console and the log sink.

    It shares the sink's lock with the stdout/stderr mirrors, so a log line
    and printed text never interleave in the file.
    """

    def __init__(self, sink, console):
        """Create the handler.

        Args: sink is the LogSink; console is the real stdout, or None.
        Returns: None.
        """
        super().__init__()
        self.lock = sink.lock
        self.sink = sink
        self.console = console
        self.setFormatter(logging.Formatter(LINE_FORMAT, TIME_FORMAT))

    def emit(self, record):
        """Format one record and write it everywhere.

        Args: record is the logging.LogRecord.
        Returns: None.
        """
        try:
            text = self.format(record) + "\n"
        except Exception:
            self.handleError(record)
            return
        if self.console is not None:
            try:
                self.console.write(text)
                self.console.flush()
            except (OSError, ValueError):
                self.console = None
        self.sink.put(text)


class AppLog:
    """Own the run log: levels, mirroring, the log and crash files, GUI text."""

    def __init__(self, level=None):
        """Create an inactive log; call start() to begin.

        Args: level is a --log-level name ("debug", "info", "warning") or
            None for ESPI_LOG_LEVEL or the default.
        Returns: None.
        """
        self.level = resolve_level(level)
        self.logger = logging.getLogger(LOGGER_NAME)
        self.sink = LogSink()
        self.handler = None
        self.original = None
        self.saved = None  # logger and warning state to restore on close
        self.log_file = None
        self.crash_file = None
        self.crash_path = None
        self.faulthandler_was_enabled = False
        self.started = time.monotonic()
        self.run_lock = None

    @property
    def path(self):
        """The log file path.

        Args: None.
        Returns: The path, or None when no log file is open.
        """
        return self.log_file.name if self.log_file is not None else None

    def start(self, crash_file=True):
        """Open the run's log file, mirror the console and attach the handler.

        Args: crash_file also opens espi_<timestamp>.crash and points
            faulthandler at it, for native crashes.
        Returns: The log file path, or None if no file could be opened.
        """
        self.started = time.monotonic()
        stamp = timestamp()
        try:
            folder = output_folder("logs")
            log_path = os.path.join(folder, f"{LOG_PREFIX}{stamp}{LOG_SUFFIX}")
            self.run_lock = RunLock(log_path)
            if not self.run_lock.acquire():
                raise OSError("could not lock this run's log")
            self.log_file = open(log_path,
                                 "w", encoding="utf-8", buffering=1)
            prune_logs(folder, KEEP_LOGS)
            if crash_file:
                self._start_crash_file(os.path.join(folder, f"{LOG_PREFIX}{stamp}{CRASH_SUFFIX}"))
        except OSError:
            pass  # log to the console and the GUI only
        self.sink.log_file = self.log_file
        self.original = (sys.stdout, sys.stderr)
        sys.stdout = MirrorStream(self.original[0], self.sink)
        sys.stderr = MirrorStream(self.original[1], self.sink)
        self.handler = SinkHandler(self.sink, self.original[0])
        warnings_logger = logging.getLogger("py.warnings")
        self.saved = (self.logger.level, self.logger.propagate, warnings_logger.propagate)
        self.logger.setLevel(self.level)
        self.logger.propagate = False
        self.logger.addHandler(self.handler)
        warnings_logger.addHandler(self.handler)
        warnings_logger.propagate = False
        logging.captureWarnings(True)
        return self.path

    def _start_crash_file(self, path):
        """Open the crash file and send faulthandler's output to it.

        Args: path is the crash file to create.
        Returns: None. Raises OSError if the file cannot be opened.
        """
        self.crash_file = open(path, "w", encoding="utf-8")
        self.crash_path = path
        self.faulthandler_was_enabled = faulthandler.is_enabled()
        faulthandler.enable(self.crash_file, all_threads=True)

    def debug(self, message):
        """Log a DEBUG message.

        Args: message is the text. Safe from any thread.
        Returns: None.
        """
        self.logger.debug(message)

    def info(self, message):
        """Log an INFO message.

        Args: message is the text. Safe from any thread.
        Returns: None.
        """
        self.logger.info(message)

    def warning(self, message):
        """Log a WARNING message.

        Args: message is the text. Safe from any thread.
        Returns: None.
        """
        self.logger.warning(message)

    def error(self, message, exc_info=None):
        """Log an ERROR message, optionally with a traceback.

        Args: message is the text; exc_info is an exception, True for the one
            being handled, or None for no traceback.
        Returns: None.
        """
        self.logger.error(message, exc_info=exc_info)

    def exception(self, message):
        """Log an ERROR message with the traceback of the exception being handled.

        Args: message is the text; call from an except block.
        Returns: None.
        """
        self.logger.error(message, exc_info=True)

    def write(self, message):
        """Log an INFO message (the original API, kept for every call site).

        Args: message is the application message text. Safe from any thread.
        Returns: None.
        """
        self.logger.info(message)

    def drain(self):
        """Collect text written since the last call, for a GUI log view.

        Args: None.
        Returns: The pending text, possibly empty.
        """
        chunks = []
        while True:
            try:
                chunks.append(self.sink.queue.get_nowait())
            except queue.Empty:
                return "".join(chunks)

    def recent_lines(self):
        """Return the newest log lines, for error reports.

        Args: None.
        Returns: Up to TAIL_LINES lines, oldest first.
        """
        return self.sink.recent()

    def uptime(self):
        """Report how long the log has been running.

        Args: None.
        Returns: Seconds since start() (or since the log was created).
        """
        return time.monotonic() - self.started

    def watchdog_tick(self):
        """Re-arm the stall watchdog; call regularly from the UI thread.

        If the UI thread does not call this again within WATCHDOG_S, faulthandler
        dumps every thread's stack into the log file (the program keeps running).

        Args: None.
        Returns: None.
        """
        if self.log_file is not None:
            faulthandler.dump_traceback_later(WATCHDOG_S, repeat=False, file=self.log_file)

    def close(self):
        """Detach the handler, restore the console and close the files.

        A crash file that faulthandler wrote to while the program kept running
        (a native exception it survived) is copied into the log and deleted,
        so only a crash file left by a run that never closed is reported.

        Args: None.
        Returns: None.
        """
        if self.log_file is not None:
            faulthandler.cancel_dump_traceback_later()
        if self.crash_file is not None:
            faulthandler.disable()
            if self.faulthandler_was_enabled and sys.__stderr__ is not None:
                faulthandler.enable(sys.__stderr__)
            self.crash_file.close()
            self.crash_file = None
            self._fold_crash_file()
        if self.handler is not None:
            logging.captureWarnings(False)
            warnings_logger = logging.getLogger("py.warnings")
            warnings_logger.removeHandler(self.handler)
            self.logger.removeHandler(self.handler)
            self.logger.setLevel(self.saved[0])
            self.logger.propagate = self.saved[1]
            warnings_logger.propagate = self.saved[2]
            self.handler = None
        if self.original is not None:
            sys.stdout, sys.stderr = self.original
            self.original = None
        if self.log_file is not None:
            self.sink.log_file = None
            self.log_file.close()
            self.log_file = None
        if self.run_lock is not None:
            self.run_lock.close()
            self.run_lock = None

    def _fold_crash_file(self):
        """Move what faulthandler wrote during a run that survived into the log.

        Args: None.
        Returns: None.
        """
        try:
            with open(self.crash_path, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
            if text.strip():
                self.warning("faulthandler reported a native exception that the "
                             "program survived:\n" + text.rstrip())
            os.remove(self.crash_path)
        except OSError:
            pass  # left for the next start to report or prune
