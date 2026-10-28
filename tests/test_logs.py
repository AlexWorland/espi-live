"""Camera-free checks for the run log and console mirroring.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import io
import os
import queue
import sys
import tempfile
import threading
import unittest
from unittest import mock

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


class MirrorStreamTests(unittest.TestCase):
    """Writes are copied to console, file and queue."""

    def make(self, original, log_file):
        """Build a stream with a fresh queue.

        Args: original is the console stand-in; log_file is a file or None.
        Returns: The MirrorStream.
        """
        self.queue = queue.SimpleQueue()
        return logs.MirrorStream(original, self.queue, log_file, threading.Lock())

    def test_write_reaches_console_file_and_queue(self):
        """One write is delivered to all three destinations.

        Args: None.
        Returns: None.
        """
        console, file = io.StringIO(), io.StringIO()
        stream = self.make(console, file)
        self.assertEqual(stream.write("hello"), 5)
        self.assertEqual((console.getvalue(), file.getvalue()), ("hello", "hello"))
        self.assertEqual(self.queue.get_nowait(), "hello")

    def test_empty_write_is_not_queued(self):
        """An empty write does not wake the GUI queue.

        Args: None.
        Returns: None.
        """
        stream = self.make(io.StringIO(), None)
        stream.write("")
        self.assertTrue(self.queue.empty())

    def test_write_without_console_still_queues(self):
        """A missing console (windowed build) does not stop mirroring.

        Args: None.
        Returns: None.
        """
        stream = self.make(None, None)
        stream.write("x")
        self.assertEqual(self.queue.get_nowait(), "x")

    def test_failing_log_file_is_dropped_on_write(self):
        """A write error disables the log file but keeps the console.

        Args: None.
        Returns: None.
        """
        console = io.StringIO()
        stream = self.make(console, FailingFile())
        stream.write("a")
        self.assertIsNone(stream.log_file)
        self.assertEqual(console.getvalue(), "a")

    def test_flush_forwards_to_console_and_file(self):
        """flush reaches both console and file.

        Args: None.
        Returns: None.
        """
        console, file = mock.Mock(), mock.Mock()
        self.make(console, file).flush()
        console.flush.assert_called_once_with()
        file.flush.assert_called_once_with()

    def test_flush_without_console_or_file_is_harmless(self):
        """flush with nothing attached does nothing.

        Args: None.
        Returns: None.
        """
        self.make(None, None).flush()

    def test_failing_log_file_is_dropped_on_flush(self):
        """A flush error disables the log file.

        Args: None.
        Returns: None.
        """
        stream = self.make(None, FailingFile())
        stream.flush()
        self.assertIsNone(stream.log_file)

    def test_isatty_reflects_console(self):
        """isatty reports the console flag, or False without a console.

        Args: None.
        Returns: None.
        """
        self.assertFalse(self.make(io.StringIO(), None).isatty())
        self.assertFalse(self.make(None, None).isatty())

    def test_other_attributes_come_from_console(self):
        """Unknown attributes are looked up on the console stream.

        Args: None.
        Returns: None.
        """
        self.assertEqual(self.make(mock.Mock(encoding="utf-8"), None).encoding, "utf-8")

    def test_missing_attribute_without_console_raises(self):
        """Without a console, unknown attributes raise AttributeError.

        Args: None.
        Returns: None.
        """
        with self.assertRaises(AttributeError):
            self.make(None, None).encoding


class AppLogTests(unittest.TestCase):
    """The per-run log file and GUI text."""

    def setUp(self):
        """Redirect output/ to a temp folder and remember the console.

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
        self.log = logs.AppLog()

    def tearDown(self):
        """Restore the console and delete the temp folder.

        Args: None.
        Returns: None.
        """
        self.log.close()
        sys.stdout, sys.stderr = self.real_streams
        self.tmp.cleanup()

    def test_start_mirrors_output_into_file_and_drain(self):
        """Printed text lands in the log file and in drain().

        Args: None.
        Returns: None.
        """
        path = self.log.start()
        self.log.write("hello log")
        print("plain print")
        self.assertIn("hello log", self.log.drain())
        self.assertEqual(self.log.drain(), "")
        self.log.close()
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("hello log", text)
        self.assertIn("plain print", text)
        self.assertTrue(path.startswith(self.tmp.name))

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
        """If the log cannot be opened, mirroring still starts.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(logs, "output_folder", side_effect=OSError("ro")):
            self.assertIsNone(self.log.start())
        print("still mirrored")
        self.assertIn("still mirrored", self.log.drain())
        self.assertFalse(os.listdir(self.tmp.name))


if __name__ == "__main__":
    unittest.main()
