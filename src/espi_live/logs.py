"""Console mirroring and the per-run log file shared by every front end."""

import os
import queue
import sys
import threading
from datetime import datetime

from espi_live.paths import output_folder, timestamp


class MirrorStream:
    """Copy Python console output to the GUI queue and an optional log file."""

    def __init__(self, original, output_queue, log_file, lock):
        """Prepare a console mirror.

        Args: original is stdout or stderr; output_queue receives GUI text;
            log_file is an optional text file; lock serializes writes.
        Returns: None.
        """
        self.original = original
        self.output_queue = output_queue
        self.log_file = log_file
        self.lock = lock

    def write(self, value):
        """Forward a text chunk to console, log file, and GUI queue.

        Args: value is the text to write.
        Returns: Number of characters accepted.
        """
        with self.lock:
            if self.original is not None:
                self.original.write(value)
            if self.log_file is not None:
                try:
                    self.log_file.write(value)
                except OSError:
                    self.log_file = None
            if value:
                self.output_queue.put(value)
        return len(value)

    def flush(self):
        """Flush the underlying console and log file.

        Args: None.
        Returns: None.
        """
        with self.lock:
            if self.original is not None:
                self.original.flush()
            if self.log_file is not None:
                try:
                    self.log_file.flush()
                except OSError:
                    self.log_file = None

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


class AppLog:
    """Own the run log: stdout/stderr mirroring, the log file and GUI text."""

    def __init__(self):
        """Create an inactive log; call start() to begin mirroring.

        Args: None.
        Returns: None.
        """
        self.output_queue = queue.SimpleQueue()
        self.log_file = None
        self.original = None

    def start(self):
        """Open output/logs/espi_<timestamp>.log and mirror the console into it.

        Args: None.
        Returns: The log file path, or None if no file could be opened.
        """
        try:
            path = os.path.join(output_folder("logs"), f"espi_{timestamp()}.log")
            self.log_file = open(path, "w", encoding="utf-8", buffering=1)
        except OSError:
            self.log_file = None
        self.original = (sys.stdout, sys.stderr)
        lock = threading.Lock()
        sys.stdout = MirrorStream(self.original[0], self.output_queue,
                                  self.log_file, lock)
        sys.stderr = MirrorStream(self.original[1], self.output_queue,
                                  self.log_file, lock)
        return self.log_file.name if self.log_file is not None else None

    def write(self, message):
        """Write a timestamped application message to every log destination.

        Args: message is the application message text. Safe from any thread.
        Returns: None.
        """
        print(f"{datetime.now():%H:%M:%S}  {message}", flush=True)

    def drain(self):
        """Collect text written since the last call, for a GUI log view.

        Args: None.
        Returns: The pending text, possibly empty.
        """
        chunks = []
        while True:
            try:
                chunks.append(self.output_queue.get_nowait())
            except queue.Empty:
                return "".join(chunks)

    def close(self):
        """Restore the console streams and close the log file.

        Args: None.
        Returns: None.
        """
        if self.original is not None:
            sys.stdout, sys.stderr = self.original
            self.original = None
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
