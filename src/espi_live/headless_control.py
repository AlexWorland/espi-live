"""Live control of a headless run: drive the NI DAQ and the cameras from a script.

``espi_live --headless --control`` reads one JSON command per line on stdin and
answers each with one JSON line on stdout, so a test harness (or a person at a
terminal) can do everything the NI DAQ window and the camera controls do without
a display. Log text goes to stderr and the log file only; stdout carries nothing
but replies and events.

A command is ``{"cmd": "<name>", "id": <optional, echoed back>, ...arguments}``.
A reply is ``{"ok": true, "id": ..., ...result}`` or ``{"ok": false, "error":
"<text>", "problems": [...]}``. A DAQ fault arrives on its own as
``{"event": "fault", "reason": "<text>"}``.

Commands (cameras are chosen with an optional ``camera``: a position, a label or
a serial number; the default is the first running camera)::

    status                              cameras, DAQ, stepper, faults seen
    daq_scan                            list the NI devices (SIMULATED is always there)
    daq_connect {device}                connect; every output goes to 0 V
    daq_arm {roles?}                    record the safety checklist ("generator",
                                        "piezo"; default both assigned roles);
                                        needs --daq-checklist-done, and a real
                                        device takes no drive signal without it
    daq_apply {settings}                change DaqSettings fields and write them
    daq_zero                            every output to 0 V
    daq_self_test                       step each analog output and read it back
    piezo_step {volts}                  hold the piezo at a level (the Phase view's call)
    phase_attach / phase_detach         give the piezo to a camera's Phase view, or take it away
    set_camera {settings, camera?}      change camera settings (checked like a saved profile)
    save {camera?, title?, notes?}      save the newest frame
    reference {camera?}                 capture the reference
    wait {seconds}                      let frames flow (keeps the DAQ watchdog fed)
    quit                                zero the outputs, stop and finish recordings

Closing stdin counts as ``quit``. Every way out (quit, end of input, a signal,
an error) sets the outputs to 0 V. The DAQ's own safety rules are untouched: the
checklist gate, the safety limits, the readback self-check and the watchdog all
live in ``daq.DaqController``. No GUI imports here.
"""

import json
import math
import queue
import sys
import threading
import time
from dataclasses import asdict, replace

from espi_live import daq, phase, profiles
from espi_live.daq import DaqSettings

POLL_S = 0.1  # how often the loop feeds the DAQ watchdog while idle
DAQ_TIMEOUT_S = 10.0  # longest wait for the DAQ thread to answer one command
SELF_TEST_TIMEOUT_S = 30.0
SAVE_TIMEOUT_S = 30.0
MAX_WAIT_S = 3600.0
CLOSE_TIMEOUT_S = 5.0
MAX_FAULTS = 20  # faults remembered for status
ROLE_NAMES = {"generator": daq.GENERATOR_ROLE, "piezo": daq.PIEZO_ROLE}


class CommandError(Exception):
    """A command that cannot be carried out; the session carries on."""

    def __init__(self, message, problems=None):
        """Describe the failure.

        Args: message is the sentence for the reply's "error"; problems is a
            list of sentences (the DAQ's own rejection reasons) or None.
        Returns: None.
        """
        super().__init__(message)
        self.problems = problems


def line_writer(stream):
    """Make the function that writes one JSON line to a stream.

    Args: stream is a text stream (the real stdout).
    Returns: A function taking a dict; it writes it as one line and flushes.
        Values that JSON cannot hold are written as text.
    """
    def write(message):
        """Write one message as a line.

        Args: message is a dict.
        Returns: None.
        """
        stream.write(json.dumps(message, default=str) + "\n")
        stream.flush()
    return write


def divert_console(log):
    """Keep stdout for replies: send log lines and printed text to stderr.

    Args: log is a started AppLog (its console handler and print mirror write
        to the real stdout until this is called).
    Returns: The real stdout, for line_writer(). AppLog.close() restores
        sys.stdout.
    """
    stdout, stderr = log.original
    log.handler.console = stderr
    sys.stdout = sys.stderr
    return stdout


class ControlSession:
    """The commands of a headless control run, against the cameras and the DAQ.

    Everything runs on the calling thread, one command at a time; the DAQ's
    own thread is reached only through its controller.
    """

    def __init__(self, channels, log, daq_controller=None, allow_arm=False, emit=None,
                 pump=None):
        """Create a session.

        Args: channels is the Channel list (it may be empty: the DAQ commands
            still work); log is the AppLog; daq_controller is a
            daq.DaqController (default: a new one); allow_arm lets daq_arm
            record the safety checklist; emit takes the dicts for unsolicited
            event lines; pump is called on every tick to handle the engines'
            events.
        Returns: None.
        """
        self.channels = channels
        self.log = log
        self.daq = daq_controller or daq.DaqController(log)
        self.allow_arm = allow_arm
        self.emit = emit or (lambda message: None)
        self.pump = pump or (lambda: None)
        self.settings = DaqSettings()  # what daq_apply last wrote: the base of the next
        self.stepper = None
        self.stepper_channel = None
        self.faults = []
        self.stopping = False
        self.closed = False

    # ----- running ---------------------------------------------------------

    def handle_line(self, line):
        """Run one input line.

        Args: line is a line of text.
        Returns: The reply dict, or None for a blank line.
        """
        if not line.strip():
            return None
        try:
            command = json.loads(line)
        except ValueError as exc:
            return {"ok": False, "error": f"not valid JSON: {exc}"}
        return self.execute(command)

    def execute(self, command):
        """Run one command.

        Args: command is a dict with a "cmd" name (see the module docstring).
        Returns: The reply dict. A command that fails, or a bug in one, is
            answered with ok false and the session carries on.
        """
        reply = {}
        if isinstance(command, dict) and "id" in command:
            reply["id"] = command["id"]
        try:
            name = command.get("cmd") if isinstance(command, dict) else None
            if not isinstance(name, str):
                raise CommandError('a command is a JSON object with a "cmd" name')
            handler = getattr(self, f"cmd_{name}", None) if name.isidentifier() else None
            if handler is None:
                raise CommandError(f"unknown command {name!r}")
            result = handler(command)
            reply = {"ok": True, **reply, **result}
        except CommandError as exc:
            reply = {"ok": False, **reply, "error": str(exc)}
            if exc.problems:
                reply["problems"] = list(exc.problems)
        except Exception as exc:  # a bug in one command must not end the session
            self.log.exception(f"Control command failed: {exc}")
            reply = {"ok": False, **reply, "error": f"{type(exc).__name__}: {exc}"}
        return reply

    def tick(self):
        """Do the housekeeping between commands.

        Feeds the DAQ watchdog, handles the DAQ's events (a fault is sent as an
        event line) and the engines' events.

        Args: None.
        Returns: None.
        """
        self.daq.heartbeat()
        while True:
            try:
                kind, payload = self.daq.events.get_nowait()
            except queue.Empty:
                break
            self._absorb(kind, payload)
        self.pump()

    def close(self):
        """Take the piezo away from the cameras and set the DAQ outputs to 0 V.

        Args: None.
        Returns: None. Safe to call more than once.
        """
        if self.closed:
            return
        self.closed = True
        self._attach(None)
        if not self.daq.shutdown(CLOSE_TIMEOUT_S):
            self.log.warning("NI DAQ: the device thread did not finish within "
                             f"{CLOSE_TIMEOUT_S:g} s.")

    # ----- DAQ events --------------------------------------------------------

    def _absorb(self, kind, payload):
        """Follow one DAQ event: keep the cameras' view of the outputs right.

        Args: kind and payload are the controller's event.
        Returns: None. A fault or failure is remembered and sent as an event line.
        """
        if kind == "connected":
            self._attach(None)
            self._set_drive(daq.Drive())
        elif kind == "applied":
            self._set_drive(payload.drive)
            self._refresh_stepper()
        elif kind in ("zeroed", "failed", "closed", "fault"):
            self._set_drive(daq.Drive())
            self._attach(None)
            if kind in ("failed", "fault"):
                self.faults = (self.faults + [f"{kind}: {payload}"])[-MAX_FAULTS:]
                if kind == "fault":
                    self.emit({"event": "fault", "reason": payload})

    def _await(self, kinds, timeout=DAQ_TIMEOUT_S):
        """Wait for one of the DAQ's answers, feeding its watchdog meanwhile.

        Args: kinds is a tuple of event kinds that end the wait; timeout is in
            seconds.
        Returns: (kind, payload) of the event that ended it. Raises
            CommandError when the DAQ reports a failure or does not answer.
        """
        deadline = time.monotonic() + timeout
        while True:
            self.daq.heartbeat()
            try:
                kind, payload = self.daq.events.get(timeout=POLL_S)
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise CommandError(f"the DAQ did not answer within {timeout:g} s") \
                        from None
                continue
            self._absorb(kind, payload)
            if kind in kinds:
                return kind, payload
            if kind == "failed":
                raise CommandError(f"the DAQ failed: {payload}")

    def _set_drive(self, drive):
        """Tell every camera what the outputs produce (a demo camera follows it).

        Args: drive is a daq.Drive.
        Returns: None.
        """
        for channel in self.channels:
            channel.set_drive(drive)

    def _attach(self, stepper, channel=None):
        """Give the piezo to one camera's engine and no other, or take it away.

        Args: stepper is a phase.PiezoStepper or None; channel is the Channel
            that steps it.
        Returns: None.
        """
        self.stepper = stepper
        self.stepper_channel = channel if stepper is not None else None
        for other in self.channels:
            if other.engine is not None:
                other.engine.set_phase_stepper(stepper if other is channel else None)

    def _refresh_stepper(self):
        """After new DAQ settings, step with them (or stop if no piezo is assigned).

        Args: None.
        Returns: None.
        """
        if self.stepper is None:
            return
        applied = self.daq.status()["settings"]
        device = self.daq.status()["device"]
        if applied is None or applied.piezo_output == daq.OFF or device is None:
            self._attach(None)
        else:
            self._attach(phase.PiezoStepper(self.daq, applied, device.simulated),
                         self.stepper_channel)

    # ----- helpers ------------------------------------------------------------

    def _device(self):
        """Give the connected device.

        Args: None.
        Returns: The daq.DeviceInfo. Raises CommandError when none is connected.
        """
        device = self.daq.status()["device"]
        if device is None:
            raise CommandError("connect to a NI device first (daq_connect)")
        return device

    def _channel(self, command):
        """Pick the camera a command is about.

        Args: command is the command dict; its "camera" is a position, a label
            or a serial number, or missing for the first running camera.
        Returns: The Channel. Raises CommandError when there is none, or it is
            not running.
        """
        wanted = command.get("camera")
        if wanted is None:
            running = [channel for channel in self.channels if channel.engine is not None]
            if not running:
                raise CommandError("no camera is running")
            return running[0]
        channel = None
        if isinstance(wanted, int) and not isinstance(wanted, bool):
            if 0 <= wanted < len(self.channels):
                channel = self.channels[wanted]
        else:
            for candidate in self.channels:
                if wanted in (candidate.label, candidate.info.serial):
                    channel = candidate
        if channel is None:
            raise CommandError(f"no camera {wanted!r}")
        if channel.engine is None:
            raise CommandError(f"{channel.label} is not running")
        return channel

    def _merged(self, values):
        """Lay DAQ setting changes over the settings last applied.

        Args: values is a dict of DaqSettings fields.
        Returns: The new daq.DaqSettings. Raises CommandError for an unknown
            name or a value of the wrong type.
        """
        if not isinstance(values, dict):
            raise CommandError('"settings" must be an object of DaqSettings fields')
        known = self.settings.to_dict()
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise CommandError(f"unknown DAQ settings: {', '.join(unknown)}")
        merged = DaqSettings.from_dict({**known, **values})
        wrong = sorted(name for name, value in values.items()
                       if getattr(merged, name) != value
                       or isinstance(value, bool) != isinstance(getattr(merged, name), bool))
        if wrong:
            raise CommandError(f"wrong type or non-finite value for: {', '.join(wrong)}")
        return merged

    @staticmethod
    def _volts(value):
        """Read a voltage argument.

        Args: value is what the command held.
        Returns: The voltage as a float. Raises CommandError when it is not a
            finite number.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value):
            raise CommandError('"volts" must be a finite number')
        return float(value)

    @staticmethod
    def _device_dict(info):
        """Describe a device for a reply.

        Args: info is a daq.DeviceInfo.
        Returns: A dict.
        """
        return {"name": info.name, "model": info.model, "serial": info.serial,
                "label": info.label, "simulated": info.simulated}

    # ----- commands -----------------------------------------------------------

    def cmd_status(self, command):
        """Report the cameras, the DAQ and what the session has seen.

        Args: command is the command dict.
        Returns: The reply fields.
        """
        cameras = []
        for index, channel in enumerate(self.channels):
            engine = channel.engine
            cameras.append({
                "index": index, "label": channel.label, "serial": channel.info.serial,
                "demo": channel.demo, "running": engine is not None,
                "status": engine.status() if engine is not None else {},
                "settings": asdict(engine.settings()) if engine is not None else {}})
        state = self.daq.status()
        return {"cameras": cameras,
                "daq": {"device": self._device_dict(state["device"])
                        if state["device"] is not None else None,
                        "active": state["active"], "summary": state["summary"],
                        "settings": state["settings"].to_dict()
                        if state["settings"] is not None else None,
                        "pending_settings": self.settings.to_dict(),
                        "checklist_done": self.daq.checklist_done(),
                        "allow_arm": self.allow_arm},
                "stepper": self.stepper is not None, "faults": list(self.faults)}

    def cmd_daq_scan(self, command):
        """List the NI devices.

        Args: command is the command dict.
        Returns: The reply fields: devices and reason (why none could be listed).
        """
        self.daq.request("scan")
        _, (devices, reason) = self._await(("devices",))
        return {"devices": [self._device_dict(info) for info in devices] +
                [self._device_dict(daq.SIMULATED_DEVICE)], "reason": reason}

    def cmd_daq_connect(self, command):
        """Connect to a device; every output goes to 0 V.

        Args: command is the command dict with "device" (a DAQmx name or SIMULATED).
        Returns: The reply fields: the device and its outputs.
        """
        name = command.get("device")
        if not isinstance(name, str) or not name:
            raise CommandError('"device" must be a device name (or "Simulated")')
        self.daq.request("connect", device=name)
        _, outputs = self._await(("connected",))
        self.settings = replace(self.settings, device=outputs.device.name,
                                generator_on=False, piezo_on=False)
        return {"device": self._device_dict(outputs.device), "analog": list(outputs.analog),
                "counters": list(outputs.counters)}

    def cmd_daq_arm(self, command):
        """Record that the safety checklist was ticked for this wiring.

        Args: command is the command dict; "roles" is a list of "generator" and
            "piezo" (default: every role assigned in the current settings).
        Returns: The reply fields: what is now covered. Raises CommandError
            unless the run was started with --daq-checklist-done.
        """
        if not self.allow_arm:
            raise CommandError(
                "daq_arm needs --daq-checklist-done on the command line: that flag "
                "says someone has ticked the NI DAQ safety checklist for this wiring")
        device = self._device()
        roles = command.get("roles", list(ROLE_NAMES))
        if not isinstance(roles, list) or not roles or \
                any(role not in ROLE_NAMES for role in roles):
            raise CommandError('"roles" must be a list of "generator" and/or "piezo"')
        s = self.settings
        covered = replace(
            s, generator_on="generator" in roles and s.generator_output != daq.OFF,
            piezo_on="piezo" in roles and s.piezo_output != daq.OFF)
        self.daq.arm(covered)
        return {"device": device.name, "simulated": device.simulated,
                "covered": sorted("/".join(filter(None, role))
                                  for role in daq.driven_roles(covered))}

    def cmd_daq_apply(self, command):
        """Change DAQ settings and write them to the device.

        Args: command is the command dict; "settings" holds DaqSettings fields,
            laid over the ones last applied (the first time, the defaults).
        Returns: The reply fields: what the outputs now do. A rejection is an
            error whose "problems" are the DAQ's reasons.
        """
        device = self._device()
        merged = replace(self._merged(command.get("settings", {})), device=device.name)
        self.daq.request("apply", settings=merged)
        kind, payload = self._await(("applied", "rejected"))
        if kind == "rejected":
            raise CommandError("the DAQ rejected the settings", payload)
        self.settings = merged
        return {"summary": payload.summary, "active": daq.drives(merged),
                "simulated": device.simulated, "drive": asdict(payload.drive)}

    def cmd_daq_zero(self, command):
        """Set every output to 0 V.

        Args: command is the command dict.
        Returns: The reply fields (none).
        """
        self._device()
        self.daq.request("zero")
        self._await(("zeroed",))
        self.settings = replace(self.settings, generator_on=False, piezo_on=False)
        return {}

    def cmd_daq_self_test(self, command):
        """Step each analog output and read it back.

        Args: command is the command dict.
        Returns: The reply fields: passed and the result lines.
        """
        self._device()
        self.daq.request("self_test")
        _, (passed, lines) = self._await(("self_test",), SELF_TEST_TIMEOUT_S)
        self.settings = replace(self.settings, generator_on=False, piezo_on=False)
        return {"passed": passed, "lines": lines}

    def cmd_piezo_step(self, command):
        """Hold the piezo at a level, as a phase step does.

        Args: command is the command dict with "volts".
        Returns: The reply fields: the level and what was written. A level or a
            checklist the DAQ refuses is an error.
        """
        volts = self._volts(command.get("volts"))
        self._device()
        try:
            written = self.daq.move_piezo(self.settings, volts, DAQ_TIMEOUT_S)
        except daq.PiezoMoveError as exc:
            raise CommandError(str(exc)) from None
        return {"volts": volts, "summary": written.summary}

    def cmd_phase_attach(self, command):
        """Give the piezo to a camera's Phase view.

        Args: command is the command dict; "camera" picks the camera.
        Returns: The reply fields: the camera, now in the Phase view.
        """
        channel = self._channel(command)
        device = self._device()
        applied = self.daq.status()["settings"]
        base = applied or replace(self.settings, device=device.name, generator_on=False,
                                  piezo_on=False)
        if base.piezo_output == daq.OFF:
            raise CommandError("assign the piezo driver to an analog output first "
                               "(daq_apply with piezo_output)")
        if not self.daq.is_armed(replace(base, piezo_on=True)):
            raise CommandError(daq.CHECKLIST_REQUIRED)
        self._attach(phase.PiezoStepper(self.daq, base, device.simulated), channel)
        channel.engine.update_settings(mode="Phase")
        return {"camera": channel.label, "mode": "Phase"}

    def cmd_phase_detach(self, command):
        """Take the piezo away from the cameras.

        Args: command is the command dict.
        Returns: The reply fields: whether a stepper was attached.
        """
        was = self.stepper is not None
        self._attach(None)
        return {"detached": was}

    def cmd_set_camera(self, command):
        """Change a camera's settings, checked like a saved profile.

        Args: command is the command dict; "settings" holds Settings fields;
            "camera" picks the camera.
        Returns: The reply fields: the camera and the values applied. Nothing
            is changed when any name or value is wrong.
        """
        channel = self._channel(command)
        values = command.get("settings")
        if not isinstance(values, dict) or not values:
            raise CommandError('"settings" must be an object of camera settings')
        unknown = sorted(set(values) - set(profiles.PERSISTED_FIELDS))
        if unknown:
            raise CommandError(f"unknown camera settings: {', '.join(unknown)}")
        cleaned, problems = {}, []
        for name, value in values.items():
            try:
                cleaned[name] = profiles.clean_value(name, value)
            except ValueError as exc:
                problems.append(f"{name} = {value!r} is {exc}")
        if problems:
            raise CommandError("camera settings rejected", problems)
        current = channel.profile()
        channel.apply_profile(replace(current, settings=replace(current.settings, **cleaned)))
        return {"camera": channel.label, "applied": cleaned}

    def cmd_save(self, command):
        """Save the newest frame and wait until it is written.

        Args: command is the command dict; "camera", "title" and "notes" are optional.
        Returns: The reply fields: the camera.
        """
        channel = self._channel(command)
        channel.engine.request("save", title=str(command.get("title", "headless")),
                               notes=str(command.get("notes", "Headless control")))
        deadline = time.monotonic() + SAVE_TIMEOUT_S
        while channel.engine.save_pending():
            self.tick()
            if time.monotonic() >= deadline:
                raise CommandError(f"the save did not finish within {SAVE_TIMEOUT_S:g} s")
            time.sleep(POLL_S)
        return {"camera": channel.label}

    def cmd_reference(self, command):
        """Capture the reference frame.

        Args: command is the command dict; "camera" is optional.
        Returns: The reply fields: the camera.
        """
        channel = self._channel(command)
        channel.engine.request("reference")
        return {"camera": channel.label}

    def cmd_wait(self, command):
        """Let time pass with the housekeeping running.

        Args: command is the command dict with "seconds" (0 to an hour).
        Returns: The reply fields: how long it waited.
        """
        seconds = command.get("seconds")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) \
                or not 0 <= seconds <= MAX_WAIT_S:
            raise CommandError(f'"seconds" must be a number from 0 to {MAX_WAIT_S:g}')
        end = time.monotonic() + seconds
        while not self.stopping:
            remaining = end - time.monotonic()
            if remaining <= 0:
                break
            self.tick()
            time.sleep(min(POLL_S, remaining))
        return {"waited": float(seconds)}

    def cmd_quit(self, command):
        """End the session: the caller zeroes the outputs and stops the cameras.

        Args: command is the command dict.
        Returns: The reply fields (none).
        """
        self.stopping = True
        return {}


def serve(session, source, write, stop=None, duration=0.0):
    """Read commands until quit, end of input, a stop request or the duration.

    Input is read on its own thread, so the loop keeps feeding the DAQ
    watchdog and handling events while it waits for a line.

    Args: session is the ControlSession; source is an iterable of lines (stdin);
        write is the function that writes one reply dict (line_writer());
        stop is a threading.Event that ends the loop (the signal handler's),
        or None; duration is the longest run in seconds, 0 for no limit.
    Returns: None.
    """
    session.emit = write
    lines = queue.SimpleQueue()

    def read():
        """Pass the input lines on, then say the input is closed.

        Args: None.
        Returns: None.
        """
        try:
            for line in source:
                lines.put(line)
        except (OSError, ValueError):
            pass
        lines.put(None)

    threading.Thread(target=read, name="espi-control-input", daemon=True).start()
    begun = time.monotonic()
    while not session.stopping and not (stop is not None and stop.is_set()):
        if duration and time.monotonic() - begun >= duration:
            session.log.write(f"Duration of {duration:g} s reached.")
            return
        try:
            line = lines.get(timeout=POLL_S)
        except queue.Empty:
            session.tick()
            continue
        if line is None:
            session.log.write("Control input closed: stopping.")
            return
        reply = session.handle_line(line)
        if reply is not None:
            write(reply)
        session.tick()
