"""An in-memory stand-in for AppLog and PrefixedLog, shared by the tests."""


class ListLog:
    """Collect log messages in memory, with their levels."""

    def __init__(self):
        """Create an empty log.

        Args: None.
        Returns: None.
        """
        self.lines = []
        self.records = []  # (level name, message, exc_info)

    def _add(self, level, message, exc_info=None):
        """Record one message.

        Args: level is the level name; message the text; exc_info as logging takes it.
        Returns: None.
        """
        self.lines.append(message)
        self.records.append((level, message, exc_info))

    def write(self, message):
        """Record one INFO message.

        Args: message is the text to keep.
        Returns: None.
        """
        self._add("INFO", message)

    def debug(self, message):
        """Record one DEBUG message.

        Args: message is the text to keep.
        Returns: None.
        """
        self._add("DEBUG", message)

    def info(self, message):
        """Record one INFO message.

        Args: message is the text to keep.
        Returns: None.
        """
        self._add("INFO", message)

    def warning(self, message):
        """Record one WARNING message.

        Args: message is the text to keep.
        Returns: None.
        """
        self._add("WARNING", message)

    def error(self, message, exc_info=None):
        """Record one ERROR message.

        Args: message is the text to keep; exc_info is kept for assertions.
        Returns: None.
        """
        self._add("ERROR", message, exc_info)

    def exception(self, message):
        """Record one ERROR message with the current exception.

        Args: message is the text to keep.
        Returns: None.
        """
        self._add("ERROR", message, True)

    def has(self, text):
        """Check whether any logged line contains text.

        Args: text is the substring to look for.
        Returns: True if some line contains it.
        """
        return any(text in line for line in self.lines)

    def at(self, level):
        """List the messages logged at one level.

        Args: level is a level name such as "WARNING".
        Returns: The messages, oldest first.
        """
        return [message for name, message, _ in self.records if name == level]

