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

from espi_live import error_reports, sources
from espi_live.camera import CameraInfo
from espi_live.daq import Drive
from espi_live.engine import Engine
from espi_live.engine import default_settings as source_settings
from espi_live.profiles import Profile, default_profile, profile_key, summary


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


# Sensitivity floor (grey levels) analog capture starts with. The analog chain
# (cable, digitizer, YUV conversion) leaves about 15 levels of frame-to-frame
# noise in a still scene, measured with a Sony XC-ST50 on a USB-Live2; twice
# that keeps the noise dark. FLIR cameras start at 0, the original behaviour.
def default_settings(info, demo=False):
    """Choose the starting settings for a camera.

    Args: info is the camera's CameraInfo; demo is True for a synthetic source.
    Returns: A Settings: an analog camera starts in the LabVIEW flow, and
        every capture device's own settings carry the ANALOG_MIN_RANGE
        sensitivity floor (see engine.default_settings and own_settings).
    """
    return source_settings(None if demo else info)


class Channel:
    """A camera and the engine that processes it while processing is on."""

    def __init__(self, log, info, demo=False, engine=None, profile=None):
        """Create a channel; it starts idle unless an engine is given.

        Args: log is the shared AppLog; info is the CameraInfo; demo makes
            the source synthetic; engine adopts an Engine that was already
            set up (opened or demo); profile is the camera's saved Profile,
            which replaces its defaults (None: start from the defaults, or
            from the given engine's settings).
        Returns: None.
        """
        self.info = info
        reporter = error_reports.current_reporter()
        if reporter is not None:
            reporter.update_context(serials=[info.serial])
        self.demo_source = demo
        self.log = PrefixedLog(log, "")  # the window names cameras when several
        self.engine = engine
        self.settings = (engine.settings() if engine is not None
                         else default_settings(info, demo))
        self.limits = {}  # last "camera_limits" payload: {node name: (low, high)}
        self.identity = None  # last "camera_identity" payload (CameraIdentity)
        self.camera_settings = None  # phone settings from "camera_settings" events
        self.report = None  # last "camera_report" payload (camera_reports.Report)
        self.frozen = None  # FramePacket held on screen while the display is frozen
        self.frozen_at = None  # datetime the display was frozen
        # The user's settings while the LabVIEW flow is on; analog cameras
        # start in the flow.
        self.labview_saved = default_profile(info, demo).labview_saved
        self.retired = []  # stopped engines that may still be finishing
        self.drive = Drive()  # the DAQ outputs, for a demo channel's simulated drumhead
        if profile is not None:
            self.apply_profile(profile)

    @property
    def label(self):
        """Name the channel for menus.

        Args: None.
        Returns: The camera's label.
        """
        return self.info.label

    @property
    def profile_key(self):
        """Name this camera's saved profile in the settings store.

        Args: None.
        Returns: profiles.profile_key for the camera; a synthetic camera
            keeps its own "demo" namespace until it switches to live capture.
        """
        return profile_key(self.info, bool(self.demo_source))

    def default_profile(self):
        """Give the profile this camera has with nothing saved.

        Args: None.
        Returns: A Profile (see profiles.default_profile).
        """
        return default_profile(self.info, bool(self.demo_source))

    def profile(self):
        """Take a snapshot of this camera's controls.

        Args: None.
        Returns: A Profile of the running engine's settings (or the kept
            ones while processing is off) and the LabVIEW-flow state;
            recording is never part of it.
        """
        settings = self.engine.settings() if self.engine is not None else self.settings
        saved = None if self.labview_saved is None else dict(self.labview_saved)
        return Profile(replace(settings, record=False), saved)

    def apply_profile(self, profile):
        """Replace this camera's settings and LabVIEW-flow state.

        Only this channel changes; a running engine gets the new values on
        its next frame, and a recording keeps running.

        Args: profile is the Profile to use.
        Returns: None.
        """
        values = asdict(profile.settings)
        values.pop("record")
        self.labview_saved = (None if profile.labview_saved is None
                              else dict(profile.labview_saved))
        if self.engine is not None:
            self.engine.update_settings(**values)
        self.settings = replace(self.settings, **values)

    @property
    def active(self):
        """Report whether the channel is processing.

        Args: None.
        Returns: True while an engine is running or about to run.
        """
        return self.engine is not None

    @property
    def supported(self):
        """Report whether the viewer supports this channel's camera.

        Args: None.
        Returns: True for demo sources, FLIR cameras and recognised capture
            adapters; False for other video devices.
        """
        return self.demo or sources.is_supported(self.info)

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
        demo = bool(self.demo_source)
        engine = Engine(self.log, serial=None if demo else self.info.serial,
                        tag=self.info.serial, info=None if demo else self.info)
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
            engine.request("drive", drive=self.drive)
        else:
            engine.open_camera()
        self.engine = engine

    def start(self, after=None):
        """Turn processing on; a camera is opened on the engine's own thread.

        Args: after is the Engine of another channel that is releasing the
            same device (see shares_device), which the new engine waits for;
            by default it waits for this channel's own last engine.
        Returns: None. An open failure arrives later as an "open_failed"
            engine event.
        """
        if self.engine is not None:
            return
        engine = self._new_engine()
        if self.demo_source:
            engine.use_demo(self.info.serial)
            engine.request("drive", drive=self.drive)
        else:
            engine.open_on_start(after or (self.retired[-1] if self.retired else None))
        self.engine = engine
        engine.start()

    def set_drive(self, drive):
        """Follow what the NI DAQ's outputs produce.

        A demo channel's simulated drumhead reacts to it (see
        simulation.DemoSource); a camera channel only keeps it.

        Args: drive is a daq.Drive (Drive() when nothing is driven).
        Returns: None; a running demo engine is sent it at once, and every
            demo engine started later starts with it.
        """
        self.drive = drive
        if self.engine is not None and self.demo:
            self.engine.request("drive", drive=drive)

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

    def shares_device(self, other):
        """Report whether another channel is the same phone on a different lens.

        An iPhone's app serves one link at a time, so two of its back cameras
        cannot stream together; opening one ends the other.

        Args: other is another Channel.
        Returns: True when both are cameras of the same iPhone.
        """
        return (self is not other and not self.demo_source and not other.demo_source
                and self.info.kind == other.info.kind == "iphone"
                and self.info.path == other.info.path)

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


def restore_channel(log, info, demo=False, profiles=None):
    """Create a channel with the camera's saved profile, if it has one.

    Args: log is the shared AppLog; info is the CameraInfo; demo makes the
        source synthetic; profiles is a profiles.ProfileStore, or None to
        start from the defaults.
    Returns: An idle Channel. A restored camera is logged with its camera
        values, since it no longer starts from the defaults.
    """
    channel = Channel(log, info, demo=demo)
    profile = (None if profiles is None else
               profiles.load_profile(channel.profile_key, channel.default_profile()))
    if profile is not None:
        channel.apply_profile(profile)
        log.write(f"{info.label}: saved settings restored ({summary(profile)}).")
    return channel


def open_camera_channels(log, profiles=None):
    """Detect the cameras and open the first one that works, for startup.

    FLIR cameras are tried first, then recognised capture adapters; other
    video devices are listed but never opened here.

    Args: log is the shared AppLog; profiles is a profiles.ProfileStore
        whose saved profiles replace each camera's defaults, or None.
    Returns: A list of Channel, one per camera, with the first openable one
        active (opened, not started). Raises when no camera may be opened
        or none of them opens.
    """
    infos = sources.detect_cameras(log)
    channels = [restore_channel(log, info, profiles=profiles) for info in infos]
    candidates = [channels[infos.index(info)] for info in sources.auto_open_order(infos)]
    if not candidates:
        raise RuntimeError("No FLIR camera or recognised capture adapter detected.")
    if len(channels) > 1:
        log.write(f"{len(channels)} cameras detected: "
                  + ", ".join(channel.label for channel in channels))
    error = None
    for channel in candidates:
        try:
            channel.open()
            return channels
        except Exception as exc:
            error_reports.caught(log, exc, "camera_open", f"Could not open {channel.label}")
            error = exc
    raise error


def demo_channels(log, count, profiles=None):
    """Create the synthetic cameras for demo mode; the first one is opened.

    Args: log is the shared AppLog; count is the number of synthetic
        cameras; profiles is a profiles.ProfileStore whose demo profiles
        replace the defaults, or None.
    Returns: A list of Channel.
    """
    channels = [restore_channel(log, info, demo=True, profiles=profiles)
                for info in demo_infos(count)]
    channels[0].open()
    return channels
