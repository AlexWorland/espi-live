"""Camera-free checks for the run log: levels, mirroring, retention, crash files.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import faulthandler
import io
import logging
import os
import sys
import tempfile
import threading
import unittest
import warnings
from unittest import mock

import numpy

from espi_live import logs, paths


class FailingFile:
    """A file-like object whose writes and flushes raise OSError."""

    def write(self, value):
        """Always fail.

        Args: value is ignored.
        Returns: Never returns; raises OSError.
        """
        raise OSError("disk full")

    def flush(self):
        """Always fail.

        Args: None.
        Returns: Never returns; raises OSError.
        """
        raise OSError("disk full")


def drain(sink):
    """Collect everything a sink queued.

    Args: sink is a LogSink.
    Returns: The queued text.
    """
    chunks = []
    while not sink.queue.empty():
        chunks.append(sink.queue.get_nowait())
    return "".join(chunks)


class LevelTests(unittest.TestCase):
    """--log-level, ESPI_LOG_LEVEL and the default."""

    def test_option_wins_over_environment(self):
        """An explicit level beats ESPI_LOG_LEVEL.

        Args: None.
        Returns: None.
        """
        with mock.patch.dict(os.environ, {logs.LEVEL_ENV: "warning"}):
            self.assertEqual(logs.resolve_level("debug"), logging.DEBUG)
            self.assertEqual(logs.resolve_level(None), logging.WARNING)

    def test_default_and_unknown_names(self):
        """Unknown names fall through to the default INFO.

        Args: None.
        Returns: None.
        """
        with mock.patch.dict(os.environ, {logs.LEVEL_ENV: "loud"}):
            self.assertEqual(logs.resolve_level("chatty"), logging.INFO)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(logs.resolve_level(None), logging.INFO)
        self.assertEqual(logs.resolve_level(" Warning "), logging.WARNING)


class HeaderTests(unittest.TestCase):
    """The lines every log starts with."""

    def test_header_names_build_platform_libraries_and_front_end(self):
        """Build, commit time, platform, libraries and the front end are listed.

        Args: None.
        Returns: None.
        """
        info = mock.Mock(BUILD=42, COMMIT="abcdef1234", ASSET="espi.zip",
                         COMMITTED_AT="2026-10-29T10:00:00+00:00")
        fake_spin = mock.Mock(__version__="4.2.0.88")
        with mock.patch.object(logs.updater, "_build_info", info), \
                mock.patch.dict(sys.modules, {"PySpin": fake_spin}):
            lines = logs.header_lines("Qt", logging.DEBUG)
        self.assertEqual(lines[0], "ESPI live build 42 (abcdef1, espi.zip), "
                                   "committed 2026-10-29T10:00:00+00:00")
        self.assertTrue(lines[1].startswith("Platform: "))
        self.assertIn("Python ", lines[2])
        self.assertIn(f"NumPy {numpy.__version__}", lines[2])
        self.assertIn("PySpin 4.2.0.88", lines[2])
        self.assertEqual(lines[3], "Front end: Qt; log level DEBUG")

    def test_source_checkout_has_no_commit_time(self):
        """A source checkout says so and shows no commit time.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(logs.updater, "_build_info", None):
            self.assertEqual(logs.header_lines("Qt", logging.INFO)[0],
                             "ESPI live (source checkout, no build number)")


class PruneTests(unittest.TestCase):
    """Only the newest run logs are kept."""

    def setUp(self):
        """Create a temporary logs folder.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = self.tmp.name

    def touch(self, name):
        """Create an empty file in the folder.

        Args: name is the file name.
        Returns: The path.
        """
        path = os.path.join(self.folder, name)
        open(path, "w").close()
        return path

    def test_keeps_the_newest_logs_and_removes_old_crash_files(self):
        """Older logs and their crash files go; other files stay.

        Args: None.
        Returns: None.
        """
        for index in range(5):
            self.touch(f"espi_2026010{index}_000000_000000.log")
        self.touch("espi_20260100_000000_000000.crash")
        self.touch("error_reports.json")
        self.assertEqual(logs.prune_logs(self.folder, keep=3), 2)
        self.assertEqual(sorted(os.listdir(self.folder)), [
            "error_reports.json", "espi_20260102_000000_000000.log",
            "espi_20260103_000000_000000.log", "espi_20260104_000000_000000.log"])

    def test_files_in_use_are_left_for_next_time(self):
        """A log another copy still has open is skipped, not an error.

        Args: None.
        Returns: None.
        """
        self.touch("espi_20260101_000000_000000.log")
        self.touch("espi_20260102_000000_000000.log")
        with mock.patch.object(logs.os, "remove", side_effect=PermissionError("in use")):
            self.assertEqual(logs.prune_logs(self.folder, keep=1), 1)
        self.assertEqual(len(os.listdir(self.folder)), 2)

    def test_missing_folder_lists_nothing(self):
        """A folder that does not exist has no run logs.

        Args: None.
        Returns: None.
        """
        self.assertEqual(logs.run_logs(os.path.join(self.folder, "missing")), [])


class LogSinkTests(unittest.TestCase):
    """The shared destinations behind one lock."""

    def test_put_reaches_file_queue_and_tail(self):
        """Text goes to the file, the GUI queue and the recent-lines tail.

        Args: None.
        Returns: None.
        """
        file = io.StringIO()
        sink = logs.LogSink(file)
        sink.put("one\ntw")
        sink.put("o\nthree")
        sink.put("")
        self.assertEqual(file.getvalue(), "one\ntwo\nthree")
        self.assertEqual(drain(sink), "one\ntwo\nthree")
        self.assertEqual(sink.recent(), ["one", "two", "three"])

    def test_tail_keeps_only_the_newest_lines(self):
        """The tail holds at most TAIL_LINES lines.

        Args: None.
        Returns: None.
        """
        sink = logs.LogSink()
        sink.put("".join(f"line {index}\n" for index in range(100)))
        recent = sink.recent()
        self.assertEqual(len(recent), logs.TAIL_LINES)
        self.assertEqual(recent[-1], "line 99")

    def test_failing_file_is_dropped(self):
        """Write and flush errors drop the file but keep the queue.

        Args: None.
        Returns: None.
        """
        sink = logs.LogSink(FailingFile())
        sink.put("a")
        self.assertIsNone(sink.log_file)
        self.assertEqual(drain(sink), "a")
        sink.log_file = FailingFile()
        sink.flush()
        self.assertIsNone(sink.log_file)
        sink.flush()  # nothing attached: harmless

    def test_flush_reaches_the_file(self):
        """flush forwards to the log file.

        Args: None.
        Returns: None.
        """
        file = mock.Mock()
        logs.LogSink(file).flush()
        file.flush.assert_called_once_with()


class MirrorStreamTests(unittest.TestCase):
    """Printed text is copied to the console and the sink."""

    def test_write_reaches_console_and_sink(self):
        """One write is delivered to the console and the sink.

        Args: None.
        Returns: None.
        """
        console, file = io.StringIO(), io.StringIO()
        sink = logs.LogSink(file)
        stream = logs.MirrorStream(console, sink)
        self.assertEqual(stream.write("hello"), 5)
        self.assertEqual((console.getvalue(), file.getvalue()), ("hello", "hello"))
        self.assertEqual(drain(sink), "hello")

    def test_write_without_console_still_reaches_sink(self):
        """A missing console (windowed build) does not stop mirroring.

        Args: None.
        Returns: None.
        """
        sink = logs.LogSink()
        logs.MirrorStream(None, sink).write("x")
        self.assertEqual(drain(sink), "x")

    def test_flush_forwards_to_console_and_file(self):
        """flush reaches both console and file.

        Args: None.
        Returns: None.
        """
        console, file = mock.Mock(), mock.Mock()
        logs.MirrorStream(console, logs.LogSink(file)).flush()
        console.flush.assert_called_once_with()
        file.flush.assert_called_once_with()
        logs.MirrorStream(None, logs.LogSink()).flush()

    def test_isatty_and_other_attributes(self):
        """isatty and unknown attributes come from the console, if any.

        Args: None.
        Returns: None.
        """
        sink = logs.LogSink()
        self.assertFalse(logs.MirrorStream(io.StringIO(), sink).isatty())
        self.assertFalse(logs.MirrorStream(None, sink).isatty())
        self.assertEqual(logs.MirrorStream(mock.Mock(encoding="utf-8"), sink).encoding, "utf-8")
        with self.assertRaises(AttributeError):
            logs.MirrorStream(None, sink).encoding


class SinkHandlerTests(unittest.TestCase):
    """Formatted records reach the console and the sink."""

    def record(self, message="hello", level=logging.INFO):
        """Build a log record.

        Args: message and level describe it.
        Returns: The LogRecord.
        """
        return logging.LogRecord("espi_live.test", level, __file__, 1, message, None, None)

    def test_line_format_has_time_level_and_thread(self):
        """Lines read "HH:MM:SS LEVEL [thread] message".

        Args: None.
        Returns: None.
        """
        console, sink = io.StringIO(), logs.LogSink()
        logs.SinkHandler(sink, console).handle(self.record("ready", logging.WARNING))
        line = console.getvalue()
        self.assertRegex(line, r"^\d\d:\d\d:\d\d WARNING \[MainThread\] ready\n$")
        self.assertEqual(drain(sink), line)

    def test_failing_console_is_dropped(self):
        """A console error stops console output but not the sink.

        Args: None.
        Returns: None.
        """
        sink = logs.LogSink()
        handler = logs.SinkHandler(sink, FailingFile())
        handler.handle(self.record())
        self.assertIsNone(handler.console)
        self.assertIn("hello", drain(sink))

    def test_format_errors_go_to_handle_error(self):
        """A record that cannot be formatted is passed to handleError.

        Args: None.
        Returns: None.
        """
        sink = logs.LogSink()
        handler = logs.SinkHandler(sink, None)
        bad = logging.LogRecord("x", logging.INFO, __file__, 1, "%d", ("text",), None)
        with mock.patch.object(handler, "handleError") as handle_error:
            handler.handle(bad)
        handle_error.assert_called_once_with(bad)
        self.assertEqual(drain(sink), "")


class AppLogTests(unittest.TestCase):
    """The per-run log file, levels, GUI text and crash file."""

    def setUp(self):
        """Redirect output/ to a temp folder and remember the console and hooks.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(paths, "OUTPUT_DIR", self.tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.real_streams = (sys.stdout, sys.stderr)
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        self.streams = (sys.stdout, sys.stderr)
        self.faulthandler_enabled = faulthandler.is_enabled()
        self.logger = logging.getLogger(logs.LOGGER_NAME)
        self.logger_state = (self.logger.level, self.logger.propagate, list(self.logger.handlers))
        self.log = logs.AppLog("debug")

    def tearDown(self):
        """Restore the console, logger and faulthandler; delete the temp folder.

        Args: None.
        Returns: None.
        """
        self.log.close()
        sys.stdout, sys.stderr = self.real_streams
        self.assertEqual((self.logger.level, self.logger.propagate, self.logger.handlers),
                         self.logger_state)
        self.assertEqual(faulthandler.is_enabled(), self.faulthandler_enabled)
        self.tmp.cleanup()

    def folder(self):
        """Name the temporary logs folder.

        Args: None.
        Returns: The path.
        """
        return os.path.join(self.tmp.name, "logs")

    def test_start_mirrors_output_and_logs_with_levels(self):
        """Messages at every level and printed text land in the file and drain().

        Args: None.
        Returns: None.
        """
        path = self.log.start()
        self.assertEqual(self.log.path, path)
        self.log.write("hello log")
        self.log.debug("detail")
        self.log.info("info line")
        self.log.warning("careful")
        logging.getLogger("espi_live.engine").warning("module logger")
        print("plain print")
        text = self.log.drain()
        self.assertRegex(text, r"\d\d:\d\d:\d\d INFO \[MainThread\] hello log")
        self.assertIn("DEBUG [MainThread] detail", text)
        self.assertIn("WARNING [MainThread] careful", text)
        self.assertIn("module logger", text)
        self.assertEqual(self.log.drain(), "")
        self.assertEqual(self.log.recent_lines()[-1], "plain print")
        self.log.close()
        with open(path, encoding="utf-8") as handle:
            written = handle.read()
        self.assertIn("hello log", written)
        self.assertIn("plain print", written)
        self.assertTrue(path.startswith(self.tmp.name))

    def test_level_filters_messages(self):
        """At WARNING, INFO and DEBUG messages are dropped.

        Args: None.
        Returns: None.
        """
        self.log = logs.AppLog("warning")
        self.log.start()
        self.log.info("quiet")
        self.log.warning("loud")
        text = self.log.drain()
        self.assertNotIn("quiet", text)
        self.assertIn("loud", text)

    def test_error_and_exception_include_the_traceback(self):
        """error(exc_info=...) and exception() log the traceback.

        Args: None.
        Returns: None.
        """
        self.log.start()
        try:
            raise ValueError("bad value")
        except ValueError as exc:
            self.log.exception("Handled")
            self.log.error("Stored", exc_info=exc)
        text = self.log.drain()
        self.assertIn("ERROR [MainThread] Handled", text)
        self.assertIn("ERROR [MainThread] Stored", text)
        self.assertEqual(text.count("ValueError: bad value"), 2)
        self.assertIn("Traceback (most recent call last)", text)

    def test_python_warnings_are_logged(self):
        """warnings.warn lands in the log at WARNING while it runs.

        Args: None.
        Returns: None.
        """
        self.log.start()
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn("deprecated thing")
        self.assertIn("deprecated thing", self.log.drain())

    def test_close_restores_console_streams(self):
        """close puts the original stdout and stderr back.

        Args: None.
        Returns: None.
        """
        self.log.start()
        self.assertIsInstance(sys.stdout, logs.MirrorStream)
        self.log.close()
        self.assertEqual((sys.stdout, sys.stderr), self.streams)

    def test_close_before_start_is_harmless(self):
        """close on an inactive log does nothing.

        Args: None.
        Returns: None.
        """
        self.log.close()

    def test_start_without_log_file_returns_none(self):
        """If the log cannot be opened, mirroring and logging still work.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(logs, "output_folder", side_effect=OSError("ro")):
            self.assertIsNone(self.log.start())
        print("still mirrored")
        self.log.watchdog_tick()  # no file: nothing to arm
        self.assertIn("still mirrored", self.log.drain())
        self.assertFalse(os.listdir(self.tmp.name))

    def test_start_prunes_old_logs(self):
        """Starting keeps only the newest KEEP_LOGS run logs, this one included.

        Args: None.
        Returns: None.
        """
        os.makedirs(self.folder())
        for index in range(3):
            open(os.path.join(self.folder(), f"espi_2000010{index}_000000_000000.log"),
                 "w").close()
        with mock.patch.object(logs, "KEEP_LOGS", 2):
            path = self.log.start()
        self.assertEqual(logs.run_logs(self.folder()),
                         ["espi_20000102_000000_000000.log", os.path.basename(path)])

    def test_crash_file_is_enabled_and_removed_when_empty(self):
        """faulthandler writes to the run's .crash file, deleted on a clean close.

        Args: None.
        Returns: None.
        """
        self.log.start()
        crash = self.log.crash_path
        self.assertTrue(crash.endswith(logs.CRASH_SUFFIX))
        self.assertTrue(faulthandler.is_enabled())
        self.log.close()
        self.assertFalse(os.path.exists(crash))

    def test_survived_native_exception_is_folded_into_the_log(self):
        """Crash-file text from a run that closed normally moves into its log.

        Args: None.
        Returns: None.
        """
        path = self.log.start()
        crash = self.log.crash_path
        with open(crash, "a", encoding="utf-8") as handle:
            handle.write("Windows fatal exception: code 0xe0000001\n")
        self.log.close()
        self.assertFalse(os.path.exists(crash))
        with open(path, encoding="utf-8") as handle:
            self.assertIn("survived:\nWindows fatal exception: code 0xe0000001",
                          handle.read())

    def test_unreadable_crash_file_is_left_alone(self):
        """A crash file that cannot be read stays for the next start.

        Args: None.
        Returns: None.
        """
        self.log.start()
        crash = self.log.crash_path
        real_open = open

        def failing_open(name, *args, **kwargs):
            """Refuse to open the crash file.

            Args: name, args and kwargs as open() takes them.
            Returns: The opened file for any other name.
            """
            if name == crash:
                raise PermissionError("locked")
            return real_open(name, *args, **kwargs)

        with mock.patch("builtins.open", failing_open):
            self.log.close()
        self.assertTrue(os.path.exists(crash))

    def test_start_without_crash_file(self):
        """crash_file=False leaves faulthandler alone.

        Args: None.
        Returns: None.
        """
        self.log.start(crash_file=False)
        self.assertIsNone(self.log.crash_path)
        self.assertEqual(faulthandler.is_enabled(), self.faulthandler_enabled)

    def test_previously_enabled_faulthandler_is_restored(self):
        """If faulthandler was already on, close turns it back on for stderr.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(logs.faulthandler, "is_enabled", return_value=True), \
                mock.patch.object(logs.faulthandler, "enable") as enable, \
                mock.patch.object(logs.faulthandler, "disable") as disable:
            self.log.start()
            self.log.close()
        disable.assert_called_once_with()
        self.assertEqual(enable.call_args_list[-1], mock.call(sys.__stderr__))

    def test_watchdog_arms_a_traceback_dump_into_the_log(self):
        """watchdog_tick re-arms faulthandler's timer; close cancels it.

        Args: None.
        Returns: None.
        """
        self.log.start()
        with mock.patch.object(logs.faulthandler, "dump_traceback_later") as later, \
                mock.patch.object(logs.faulthandler, "cancel_dump_traceback_later") as cancel:
            self.log.watchdog_tick()
            later.assert_called_once_with(logs.WATCHDOG_S, repeat=False,
                                          file=self.log.log_file)
            self.log.close()
        cancel.assert_called_once_with()

    def test_uptime_counts_from_start(self):
        """uptime is the time since start().

        Args: None.
        Returns: None.
        """
        with mock.patch.object(logs.time, "monotonic", side_effect=[100.0, 107.5]):
            self.log.start(crash_file=False)
            self.assertEqual(self.log.uptime(), 7.5)

    def test_writes_from_threads_are_whole_lines(self):
        """Messages from several threads never interleave within a line.

        Args: None.
        Returns: None.
        """
        self.log.start()

        def chatter(name):
            """Write many lines.

            Args: name tells the threads apart.
            Returns: None.
            """
            for index in range(200):
                self.log.write(f"{name}-{index}")

        threads = [threading.Thread(target=chatter, args=(f"t{n}",)) for n in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        lines = [line for line in self.log.drain().splitlines() if " INFO " in line]
        self.assertEqual(len(lines), 800)
        self.assertTrue(all(line.split("] ", 1)[1].count("-") == 1 for line in lines))


if __name__ == "__main__":
    unittest.main()
