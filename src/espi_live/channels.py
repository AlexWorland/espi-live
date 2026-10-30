"""One processing channel per camera, so several cameras can run at once.

A Channel is a camera (or a synthetic demo source) plus the Engine that
processes it while "processing" is on. Each channel keeps its own Settings,
device limits and, while running, its own reference frame, averaging and
recording. Turning processing off stops the engine (finishing any recording
and releasing the camera) but keeps the settings, so turning it back on
resumes with the same exposure, frame rate, mode and display choices.

Nothing here imports a GUI toolkit; the Qt front end owns a list of channels
and shows one of them.
"""

from dataclasses import asdict, replace

from espi_live import error_reports
from espi_live.camera import CameraInfo, list_cameras
from espi_live.engine import Engine, Settings


class PrefixedLog:
    """Write to a shared log with a camera name in front of every line."""

    def __init__(self, log, prefix):
        """Wrap a log.

        Args: log is an AppLog (write, debug, info, warning, error,
            exception); prefix is prepended to each message (for example
            "[12345678] "); the window changes it when cameras are added or
            removed.
        Returns: None.
        """
        self.log = log
        self.prefix = prefix

    def write(self, message):
        """Write one prefixed INFO message.

        Args: message is the text to log.
        Returns: None.
        """
        self.log.write(f"{self.prefix}{message}")

    def debug(self, message):
        """Write one prefixed DEBUG message.

        Args: message is the text to log.
        Returns: None.
        """
        self.log.debug(f"{self.prefix}{message}")

    def info(self, message):
        """Write one prefixed INFO message.

        Args: message is the text to log.
        Returns: None.
        """
        self.log.info(f"{self.prefix}{message}")

    def warning(self, message):
        """Write one prefixed WARNING message.

        Args: message is the text to log.
        Returns: None.
        """
        self.log.warning(f"{self.prefix}{message}")

    def error(self, message, exc_info=None):
        """Write one prefixed ERROR message, optionally with a traceback.

        Args: message is the text to log; exc_info is an exception, True for
            the one being handled, or None.
        Returns: None.
        """
        self.log.error(f"{self.prefix}{message}", exc_info=exc_info)

    def exception(self, message):
        """Write one prefixed ERROR message with the current exception's traceback.

        Args: message is the text to log; call from an except block.
        Returns: None.
        """
        self.log.exception(f"{self.prefix}{message}")


def demo_infos(count):
    """Name the synthetic cameras used in demo mode.

    Args: count is how many synthetic cameras to create (at least 1).
    Returns: A list of CameraInfo with serials DEMO1, DEMO2, ...
    """
    return [CameraInfo(f"DEMO{index + 1}", "Synthetic camera")
            for index in range(max(1, count))]


class Channel:
    """A camera and the engine that processes it while processing is on."""

    def __init__(self, log, info, demo=False, engine=None):
        """Create a channel; it starts idle unless an engine is given.

        Args: log is the shared AppLog; info is the CameraInfo; demo makes
            the source synthetic; engine adopts an Engine that was already
            set up (opened or demo).
        Returns: None.
        """
        self.info = info
        self.demo_source = demo
        self.log = PrefixedLog(log, "")  # the window names cameras when several
        self.engine = engine
        self.settings = engine.settings() if engine is not None else Settings()
        self.limits = {}  # last "camera_limits" payload: {node name: (low, high)}
        self.identity = None  # last "camera_identity" payload (CameraIdentity)
        self.frozen = None  # FramePacket held on screen while the display is frozen
        self.frozen_at = None  # datetime the display was frozen
        self.retired = []  # stopped engines that may still be finishing

    @property
    def label(self):
        """Name the channel for menus.

        Args: None.
        Returns: The camera's label.
        """
        return self.info.label

    @property
    def active(self):
        """Report whether the channel is processing.

        Args: None.
        Returns: True while an engine is running or about to run.
        """
        return self.engine is not None

    @property
    def demo(self):
        """Report whether the channel shows synthetic video.

        Args: None.
        Returns: True for a demo source.
        """
        if self.engine is not None:
            return self.engine.demo
        return self.demo_source

    def _new_engine(self):
        """Build an engine for this channel with the saved settings.

        Args: None.
        Returns: An Engine whose settings match self.settings, recording off.
        """
        engine = Engine(self.log, serial=None if self.demo_source else self.info.serial,
                        tag=self.info.serial)
        engine.update_settings(**asdict(replace(self.settings, record=False)))
        return engine

    def open(self):
        """Open the camera (or demo source) now, on the calling thread.

        Used at startup, before the window exists. The engine is not
        started; call engine.start() afterwards.

        Args: None.
        Returns: None. Raises when the camera cannot be opened; the channel
            then stays idle.
        """
        engine = self._new_engine()
        if self.demo_source:
            engine.use_demo(self.info.serial)
        else:
            engine.open_camera()
        self.engine = engine

    def start(self):
        """Turn processing on; a camera is opened on the engine's own thread.

        Args: None.
        Returns: None. An open failure arrives later as an "open_failed"
            engine event.
        """
        if self.engine is not None:
            return
        engine = self._new_engine()
        if self.demo_source:
            engine.use_demo(self.info.serial)
        else:
            engine.open_on_start(self.retired[-1] if self.retired else None)
        self.engine = engine
        engine.start()

    def stop(self):
        """Turn processing off, keeping the settings for the next start.

        The engine finishes its recording and releases the camera in the
        background; wait() waits for that. A frozen frame is dropped.

        Args: None.
        Returns: None.
        """
        if self.engine is None:
            return
        self.frozen = self.frozen_at = None
        self.settings = replace(self.engine.settings(), record=False)
        self.engine.stop()
        self.retired.append(self.engine)
        self.engine = None

    def engines(self):
        """List every engine this channel owns, running or stopping.

        Args: None.
        Returns: A list of Engine objects.
        """
        return self.retired + ([self.engine] if self.engine is not None else [])

    def wait(self, timeout):
        """Wait for stopped engines to finish and forget the ones that did.

        Args: timeout is the maximum wait in seconds for each engine.
        Returns: True when every stopped engine has finished.
        """
        self.retired = [engine for engine in self.retired if not engine.wait(timeout)]
        return not self.retired


def open_camera_channels(log):
    """Detect the cameras and open the first one that works, for startup.

    Args: log is the shared AppLog.
    Returns: A list of Channel, one per camera, with the first openable one
        active (opened, not started). Raises when there is no camera or none
        of them opens.
    """
    channels = [Channel(log, info) for info in list_cameras(log)]
    if not channels:
        raise RuntimeError("No FLIR/Spinnaker cameras detected.")
    if len(channels) > 1:
        log.write(f"{len(channels)} cameras detected: "
                  + ", ".join(channel.label for channel in channels))
    error = None
    for channel in channels:
        try:
            channel.open()
            return channels
        except Exception as exc:
            error_reports.caught(log, exc, "camera_open", f"Could not open {channel.label}")
            error = exc
    raise error


def demo_channels(log, count):
    """Create the synthetic cameras for demo mode; the first one is opened.

    Args: log is the shared AppLog; count is the number of synthetic cameras.
    Returns: A list of Channel.
    """
    channels = [Channel(log, info, demo=True) for info in demo_infos(count)]
    channels[0].open()
    return channels
