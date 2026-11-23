"""Remote control of a running app: a localhost HTTP server and a client.

The Qt front end starts a small HTTP server on 127.0.0.1 (default port
8364, then the next free ports) so a terminal — human or LLM agent — can
query and change the running app's settings:

    python -m espi_live control status
    python -m espi_live control set exposure_us=5000 mode=Reference
    python -m espi_live control run reference
    curl http://127.0.0.1:8364/          # the API describes itself

The server binds the loopback interface only, so nothing outside the
machine can reach it, and it never opens a camera or the network: it only
talks to engines that are already running. (`--headless --control` is the
scripting interface for headless runs; this one drives the windowed app.)

Requests are handled in two layers. ControlService (here, GUI-free and
fully unit-testable) routes, validates and clamps; a backend supplied by
the front end does the work. Engine calls (update_settings, request,
status) are thread-safe by design and run on the server thread; anything
that touches window widgets is marshaled to the GUI thread by the backend.

The NI DAQ (the frequency generator and piezo driver on the NI USB device)
has its own pair of endpoints: GET /daq reports the device, its settings,
the safety checklist and the wiring; POST /daq runs the same actions the
NI DAQ window offers (scan, connect, apply, zero, self-test, piezo step,
Phase-view attach, auto-tune). DAQ settings are DaqSettings fields; every
change is logged and the window's controls follow, like camera changes.

GET /update reports the app's build and update state; POST /update checks
for, downloads or installs a newer build, or restarts into a downloaded
one. Control-triggered updates never open a dialog. An install closes the
window after every engine has stopped, and the installer then relaunches
the app, as the window's own Restart button does.

The module also holds the terminal client used by `espi_live control ...`,
which discovers the server through output/control.json or the default
port range.
"""

import argparse
import http.server
import json
import math
import os
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, fields

from espi_live import daq, paths, profiles
from espi_live.engine import Settings
from espi_live.updater import describe_build

DEFAULT_PORT = 8364
PORT_SCAN = 10  # try DEFAULT_PORT .. DEFAULT_PORT + PORT_SCAN - 1
MAX_BODY_BYTES = 1_000_000
CALL_TIMEOUT_S = 10.0  # GUI-thread round-trip limit for control requests
INFO_FILE_NAME = "control.json"
DEFAULT_LOG_LINES = 200
MAX_LOG_LINES = 5000

# Every Settings field and how to accept it: ("type", options-or-None).
# Types and choices come from profiles, so the API cannot drift from the
# controls: FIELD_TYPES from the defaults, CHOICES for the drop-downs.
_DEFAULTS = Settings()
SETTING_TYPES = {
    _field.name: (("enum", tuple(profiles.CHOICES[_field.name]))
                  if _field.name in profiles.CHOICES
                  else (type(getattr(_DEFAULTS, _field.name)).__name__, None))
    for _field in fields(Settings)
}
_BOOL_WORDS = {"true": True, "1": True, "on": True, "yes": True,
               "false": False, "0": False, "off": False, "no": False}

# Every DaqSettings field and its type name, for POST /daq's "settings".
_DAQ_DEFAULTS = daq.DaqSettings()
DAQ_SETTING_TYPES = {_field.name: type(getattr(_DAQ_DEFAULTS, _field.name)).__name__
                     for _field in fields(daq.DaqSettings)}

# The DAQ actions POST /daq runs: description, then each argument with its
# validator ("text", "settings", "number", "flag" or "camera") and whether
# it is required.
DAQ_ACTIONS = {
    "scan": ("list the NI devices (plus the simulated one)", {}),
    "connect": ("connect to a device by DAQmx name, or \"Simulated\"; every "
                "output goes to 0 V",
                {"device": ("text", True)}),
    "apply": ("change DAQ settings (DaqSettings fields) and write them to "
              "the device",
              {"settings": ("settings", True)}),
    "zero": ("set every output to 0 V", {}),
    "self_test": ("step each analog output to 0.5 V and read it back; switch "
                  "the amplifiers off first", {}),
    "step": ("hold the piezo at a level, in volts at the DAQ",
             {"volts": ("number", True)}),
    "attach": ("give the piezo to a camera's Phase view",
               {"camera": ("camera", False)}),
    "detach": ("take the piezo away from the Phase view", {}),
    "checklist": ("show the NI DAQ safety checklist on the app's screen for "
                  "these settings; ticking it arms the device and applies",
                  {"settings": ("settings", False)}),
    "tune": ("start or stop the experimental piezo auto-tune",
             {"on": ("flag", True)}),
}

# The update actions POST /update runs: description per action name.
UPDATE_ACTIONS = {
    "check": "look for a newer build on GitHub (no download)",
    "download": "download the newer build found by the last check",
    "install": ("download the newer build if needed, then close the app and "
                "restart into it; refused while a recording or frame save runs"),
    "restart": "restart into a build that is already downloaded",
}

COMMANDS = {
    "save": ("save the displayed frame as a PNG in output/frames (the frozen "
             "frame when the display is frozen)",
             {"title": "frame title", "notes": "frame notes"}),
    "reference": ("capture the newest frame as the measurement reference", {}),
    "reset_previous": ("forget the stored previous frame", {}),
    "reset_average": ("restart block averaging from an empty sum", {}),
    "switch_live": ("in demo mode, switch to the first real camera (runs in "
                    "the background)", {}),
    "display": ("show a different camera's video", {}),
    "processing_on": ("start processing a camera that is off", {}),
    "processing_off": ("stop processing a camera (its settings are kept)", {}),
}


class ControlError(Exception):
    """A request the control server refuses; .status is the HTTP status."""

    def __init__(self, message, status=400):
        """Remember the message and the HTTP status to answer with.

        Args: message is the user-facing reason; status is the HTTP code.
        Returns: None.
        """
        super().__init__(message)
        self.status = status


def coerce_setting(name, value):
    """Coerce one JSON-decoded value to a Settings field's type.

    Args: name is a Settings field name; value is the decoded JSON value.
    Returns: The coerced value. Raises ControlError (HTTP 400) when the
        value does not fit the field.
    """
    kind, options = SETTING_TYPES[name]
    if kind == "enum":
        if isinstance(value, str) and value in options:
            return value
        raise ControlError(f"{name} must be one of {', '.join(options)}")
    if kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in _BOOL_WORDS:
            return _BOOL_WORDS[value.strip().lower()]
        raise ControlError(f"{name} must be true or false")
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ControlError(f"{name} must be a whole number")
        try:
            number = float(value)
        except ValueError:
            raise ControlError(f"{name} must be a whole number") from None
        if not number.is_integer():
            raise ControlError(f"{name} must be a whole number")
        return int(number)
    if kind == "str":
        if isinstance(value, str):
            return value
        raise ControlError(f"{name} must be text")
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ControlError(f"{name} must be a number")
    try:
        number = float(value)
    except ValueError:
        raise ControlError(f"{name} must be a number") from None
    if not math.isfinite(number):
        raise ControlError(f"{name} must be a finite number")
    return number


def clamp_setting(name, value):
    """Keep a coerced value inside the range its control accepts.

    Args: name is a Settings field name; value is the coerced value.
    Returns: The value clamped to profiles.RANGES, or the value unchanged
        when the field has no range (booleans, enums, record).
    """
    if name not in profiles.RANGES:
        return value
    low, high = profiles.RANGES[name]
    return min(max(value, low), high)


def coerce_daq_setting(name, value):
    """Coerce one JSON-decoded value to a DaqSettings field's type.

    Args: name is a DaqSettings field name; value is the decoded JSON value.
    Returns: The coerced value. Raises ControlError (HTTP 400) when the
        value does not fit the field. Ranges and choices are checked later,
        by daq.problems on the device's own terms.
    """
    kind = DAQ_SETTING_TYPES[name]
    if kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in _BOOL_WORDS:
            return _BOOL_WORDS[value.strip().lower()]
        raise ControlError(f"{name} must be true or false")
    if kind == "str":
        if isinstance(value, str) and value:
            return value
        raise ControlError(f"{name} must be text")
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ControlError(f"{name} must be a number")
    try:
        number = float(value)
    except ValueError:
        raise ControlError(f"{name} must be a number") from None
    if not math.isfinite(number):
        raise ControlError(f"{name} must be a finite number")
    return number


def coerce_daq_settings(values):
    """Coerce a dict of DaqSettings changes.

    Args: values is the decoded "settings" object of a DAQ action.
    Returns: A dict of coerced values. Raises ControlError (HTTP 400) for
        an unknown field name or a value of the wrong type.
    """
    if not isinstance(values, dict):
        raise ControlError('"settings" must be an object of DAQ settings')
    unknown = sorted(set(values) - set(DAQ_SETTING_TYPES))
    if unknown:
        raise ControlError(f"unknown DAQ setting {unknown[0]!r}; names: "
                           + ", ".join(DAQ_SETTING_TYPES))
    return {name: coerce_daq_setting(name, value)
            for name, value in values.items()}


def _coerce_daq_argument(kind, value, action):
    """Coerce one DAQ action argument.

    Args: kind is the validator name from DAQ_ACTIONS; value is the decoded
        JSON value; action is the action's name, for error messages.
    Returns: The coerced value. Raises ControlError (HTTP 400) when the
        value does not fit.
    """
    if kind == "text":
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ControlError(f"{action} needs device as text, for example "
                           '"Simulated"')
    if kind == "settings":
        return coerce_daq_settings(value)
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ControlError(f"{action} needs volts as a finite number")
        try:
            number = float(value)
        except ValueError:
            raise ControlError(f"{action} needs volts as a finite number") from None
        if not math.isfinite(number):
            raise ControlError(f"{action} needs volts as a finite number")
        return number
    if kind == "flag":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in _BOOL_WORDS:
            return _BOOL_WORDS[value.strip().lower()]
        raise ControlError(f"{action} needs on as true or false")
    return value  # "camera": any JSON value; _resolve() reports bad ones


def tail(path, lines):
    """Read the last lines of a text file.

    Args: path is the file to read (None or missing: empty result); lines
        is how many lines to keep.
    Returns: A list of lines without their newlines, oldest first.
    """
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8", errors="replace") as file:
            all_lines = file.readlines()
    except OSError:
        return []
    return [line.rstrip("\n") for line in all_lines[-lines:]] if lines > 0 else []


def info_path():
    """Return the discovery file path that maps a running app to its port.

    Args: None.
    Returns: An absolute path (output/control.json).
    """
    return os.path.join(paths.OUTPUT_DIR, INFO_FILE_NAME)


def write_info(path, port):
    """Write the discovery file for a running server.

    Args: path is the file to write; port is the bound port.
    Returns: None. Failures are ignored: control keeps working without the
        file (clients fall back to scanning the default ports).
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump({"port": port, "pid": os.getpid()}, file)
    except OSError:
        pass


def clear_info(path):
    """Remove the discovery file.

    Args: path is the file to remove.
    Returns: None. A missing file is fine.
    """
    try:
        os.remove(path)
    except OSError:
        pass


class Job:
    """A function to run on another thread, with a place for its result.

    The control server's thread hands these to the GUI thread through the
    front end's caller (qt_app.MainThreadCaller).
    """

    def __init__(self, fn):
        """Wrap a function.

        Args: fn is a callable with no arguments.
        Returns: None.
        """
        self.fn = fn
        self.done = threading.Event()
        self.result = None
        self.error = None


class ControlService:
    """Route control requests to the running app through a backend.

    The backend exposes the app's channels; it raises ControlError for
    requests it refuses. See qt_app.ControlBackend for the Qt front end's
    implementation.
    """

    def __init__(self, backend):
        """Wrap a backend.

        Args: backend is the app adapter (see the module docstring).
        Returns: None.
        """
        self.backend = backend

    def handle(self, method, path, query, body):
        """Handle one request.

        Args: method is "GET" or "POST"; path is the URL path without the
            query; query maps parameter names to their first value; body is
            the decoded JSON body (None for GET).
        Returns: (http status, JSON-serializable payload).
        """
        try:
            route = path.rstrip("/") or "/"
            if method == "GET":
                if route == "/":
                    return 200, self._overview()
                if route == "/status":
                    return 200, self._status()
                if route == "/settings":
                    return 200, self._get_settings(query)
                if route == "/log":
                    return 200, self._log(query)
                if route == "/daq":
                    return 200, self._daq()
                if route == "/update":
                    return 200, {"ok": True, **self.backend.update_state()}
            elif method == "POST":
                if route == "/settings":
                    return 200, self._post_settings(body)
                if route == "/command":
                    return 200, self._post_command(body)
                if route == "/daq":
                    return 200, self._post_daq(body)
                if route == "/update":
                    return 200, self._post_update(body)
            else:
                raise ControlError(f"{method} is not supported here", 405)
            raise ControlError(f"unknown address {route}; GET / lists the API",
                               404)
        except ControlError as exc:
            return exc.status, {"error": str(exc)}
        except Exception as exc:  # unexpected: report it, keep serving
            return 500, {"error": f"{type(exc).__name__}: {exc}"}

    def _overview(self):
        """Describe the API, the settings and the commands.

        Args: None.
        Returns: The payload for GET /.
        """
        settings = {}
        for name, (kind, options) in SETTING_TYPES.items():
            entry = {"type": kind}
            if options is not None:
                entry["options"] = list(options)
            if name in profiles.RANGES:
                entry["range"] = list(profiles.RANGES[name])
            settings[name] = entry
        return {
            "app": "espi_live",
            "build": describe_build(),
            "pid": os.getpid(),
            "endpoints": {
                "GET /": "this description",
                "GET /status": "every camera: identity, settings, limits, status",
                "GET /settings?camera=N": ("one camera's settings (default: "
                                           "the displayed one)"),
                "POST /settings": ('{"camera": N, "exposure_us": 5000} applies'
                                   " settings; camera is optional"),
                "POST /command": ('{"command": "reference", "camera": N} runs a'
                                  " one-shot command"),
                "GET /daq": ("the NI DAQ: device, settings, checklist, wiring "
                             "and what the outputs produce"),
                "POST /daq": ('{"action": "apply", "settings": {"piezo_on": '
                              'true}} runs a DAQ action'),
                "GET /update": "the app's build, the newest known build and the update state",
                "POST /update": ('{"action": "install"} checks, downloads, or '
                                 "restarts into a newer build; actions: "
                                 + ", ".join(UPDATE_ACTIONS)),
                "GET /log?lines=200": "the last lines of the run log",
            },
            "settings": settings,
            "commands": {name: text for name, (text, _) in COMMANDS.items()},
            "daq": {
                "settings": DAQ_SETTING_TYPES,
                "actions": {name: text for name, (text, _) in DAQ_ACTIONS.items()},
            },
        }

    def _status(self):
        """Compose the full status of every channel.

        Args: None.
        Returns: The payload for GET /status.
        """
        channels = self.backend.channels()
        return {
            "app": "espi_live",
            "build": describe_build(),
            "pid": os.getpid(),
            "current": next((channel["index"] for channel in channels
                             if channel["displayed"]), None),
            "channels": [self._channel_detail(channel) for channel in channels],
            "daq": self.backend.daq_summary(),
        }

    def _channel_detail(self, base):
        """Add settings, limits and status to one channel description.

        Args: base is one entry from backend.channels().
        Returns: The channel's entry for GET /status.
        """
        index = base["index"]
        return {**base,
                "settings": self.backend.settings(index),
                "limits": self.backend.limits(index),
                "status": self.backend.status(index)}

    def _resolve(self, spec):
        """Find a channel by index, serial or label.

        Args: spec is None (the displayed camera), an index, or a serial or
            label string (case-insensitive).
        Returns: The backend's channel dict. Raises ControlError (404) when
            nothing matches.
        """
        channels = self.backend.channels()
        if spec is None:
            displayed = [channel for channel in channels if channel["displayed"]]
            if displayed:
                return displayed[0]
        elif isinstance(spec, int):
            if 0 <= spec < len(channels):
                return channels[spec]
        else:
            text = str(spec).strip().lower()
            for key in ("serial", "label"):
                matches = [channel for channel in channels
                           if str(channel[key]).lower() == text]
                if matches:
                    return matches[0]
            try:
                index = int(text)
            except ValueError:
                index = -1
            if 0 <= index < len(channels):
                return channels[index]
        names = ", ".join(f'{channel["index"]}:{channel["serial"]}'
                          for channel in channels) or "none"
        raise ControlError(f"no camera matches {spec!r}; cameras: {names}", 404)

    def _get_settings(self, query):
        """Return one camera's settings.

        Args: query may carry "camera".
        Returns: The payload for GET /settings.
        """
        index = self._resolve(query.get("camera"))["index"]
        return {"ok": True, "camera": index,
                "settings": self.backend.settings(index)}

    def _log(self, query):
        """Return the tail of the run log.

        Args: query may carry "lines".
        Returns: The payload for GET /log.
        """
        try:
            lines = int(query.get("lines", str(DEFAULT_LOG_LINES)))
        except ValueError:
            raise ControlError("lines must be a whole number") from None
        if not 1 <= lines <= MAX_LOG_LINES:
            raise ControlError(f"lines must be 1..{MAX_LOG_LINES}")
        return {"ok": True, "lines": self.backend.log_tail(lines)}

    def _post_settings(self, body):
        """Apply settings to one camera.

        Args: body maps Settings field names to values, with an optional
            "camera" key selecting the target.
        Returns: The payload for POST /settings.
        """
        if not isinstance(body, dict):
            raise ControlError("the body must be a JSON object")
        changes = dict(body)
        index = self._resolve(changes.pop("camera", None))["index"]
        unknown = [name for name in changes if name not in SETTING_TYPES]
        if unknown:
            raise ControlError(f"unknown setting {unknown[0]!r}; names: "
                               + ", ".join(SETTING_TYPES))
        if not changes:
            raise ControlError("no settings were given; GET / lists the names")
        applied = {name: clamp_setting(name, coerce_setting(name, value))
                   for name, value in changes.items()}
        settings = self.backend.apply(index, applied)
        return {"ok": True, "camera": index, "applied": applied,
                "settings": settings}

    def _post_command(self, body):
        """Run a one-shot command on one camera.

        Args: body carries "command", an optional "camera", and the
            command's payload (for example title and notes for save).
        Returns: The payload for POST /command.
        """
        if not isinstance(body, dict):
            raise ControlError("the body must be a JSON object")
        payload = dict(body)
        name = payload.pop("command", None)
        if name not in COMMANDS:
            raise ControlError(f"unknown command {name!r}; commands: "
                               + ", ".join(sorted(COMMANDS)))
        allowed = COMMANDS[name][1]
        extra = {key: value for key, value in payload.items() if key != "camera"}
        for key, value in extra.items():
            if key not in allowed:
                raise ControlError(
                    f"{name} does not take {key!r}; accepted: "
                    + (", ".join(sorted(allowed)) or "nothing"))
            if not isinstance(value, str):
                raise ControlError(f"{key} must be a string")
        index = self._resolve(payload.get("camera"))["index"]
        result = self.backend.command(index, name, extra)
        return {"ok": True, "camera": index, "command": name, "result": result}

    def _daq(self):
        """Report the NI DAQ's state.

        Args: None.
        Returns: The payload for GET /daq.
        """
        return {"ok": True, **self.backend.daq()}

    def _post_daq(self, body):
        """Run one NI DAQ action.

        Args: body carries "action", that action's arguments, and for attach
            an optional "camera" selecting the target (a position, a serial
            or a label; default the displayed one).
        Returns: The payload for POST /daq.
        """
        if not isinstance(body, dict):
            raise ControlError("the body must be a JSON object")
        payload = dict(body)
        name = payload.pop("action", None)
        if name not in DAQ_ACTIONS:
            raise ControlError(f"unknown DAQ action {name!r}; actions: "
                               + ", ".join(sorted(DAQ_ACTIONS)))
        if "camera" in payload and name != "attach":
            raise ControlError("only attach takes camera")
        arguments = {}
        for key, (kind, required) in DAQ_ACTIONS[name][1].items():
            if key not in payload:
                if required:
                    raise ControlError(f"{name} needs {key!r}; GET / describes "
                                       "the DAQ actions")
                continue
            arguments[key] = _coerce_daq_argument(kind, payload[key], name)
        if name == "attach":
            arguments["camera_index"] = self._resolve(payload.get("camera"))["index"]
        result = self.backend.daq_action(name, **arguments)
        return {"ok": True, "action": name, "result": result}

    def _post_update(self, body):
        """Run one update action.

        Args: body carries "action", one of UPDATE_ACTIONS.
        Returns: The payload for POST /update.
        """
        if not isinstance(body, dict):
            raise ControlError("the body must be a JSON object")
        name = body.get("action")
        if name not in UPDATE_ACTIONS:
            raise ControlError(f"unknown update action {name!r}; actions: "
                               + ", ".join(sorted(UPDATE_ACTIONS)))
        result = self.backend.update_action(name)
        return {"ok": True, "action": name, "result": result}


class ControlHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """Translate HTTP traffic into ControlService calls.

    The service and the log ride on the server object (see ControlServer).
    """

    server_version = "espi_live-control"

    def do_GET(self):
        """Answer a GET request.

        Args: None.
        Returns: None; the response is written to the client.
        """
        self._handle("GET", None)

    def do_POST(self):
        """Answer a POST request.

        Args: None.
        Returns: None; the response is written to the client.
        """
        try:
            body = self._read_body()
        except ControlError as exc:
            self._send(exc.status, {"error": str(exc)})
            return
        self._handle("POST", body)

    def _handle(self, method, body):
        """Run the service for one request and send the answer.

        Args: method is "GET" or "POST"; body is the decoded JSON body or
            None.
        Returns: None.
        """
        parts = urllib.parse.urlsplit(self.path)
        query = {key: values[0] for key, values
                 in urllib.parse.parse_qs(parts.query).items()}
        try:
            code, payload = self.server.service.handle(method, parts.path,
                                                       query, body)
        except ControlError as exc:  # defensive: the service maps its own
            code, payload = exc.status, {"error": str(exc)}
        self._note(method, parts.path, code, payload)
        self._send(code, payload)

    def _note(self, method, path, code, payload):
        """Log what matters to the app log: changes and refusals.

        Args: method, path and code describe the request; payload is the
            answer.
        Returns: None.
        """
        log = getattr(self.server, "log", None)
        if log is None:
            return
        if code >= 400:
            log.write(f"Control: {method} {path} refused ({code}): "
                      f"{payload.get('error', '')}")
        elif method == "POST":
            log.write(f"Control: {method} {path}")

    def _read_body(self):
        """Read and decode a POST body.

        Args: None.
        Returns: The decoded JSON value. Raises ControlError when the body
            is missing, too large or not JSON.
        """
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            raise ControlError("invalid Content-Length header") from None
        if length > MAX_BODY_BYTES:
            raise ControlError("the request body is too large", 413)
        data = self.rfile.read(length) if length > 0 else b""
        if not data:
            raise ControlError('a JSON body is required, for example '
                               '{"exposure_us": 5000}')
        try:
            return json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise ControlError("the body is not valid JSON") from None

    def _send(self, code, payload):
        """Write one JSON response.

        Args: code is the HTTP status; payload is JSON-serializable.
        Returns: None. A client that hung up is ignored.
        """
        data = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):  # noqa: A002 - stdlib API
        """Stay quiet; _note logs what matters to the app log.

        Args: format and args are the standard-library log call.
        Returns: None.
        """
        del format, args


class ControlServer:
    """A small HTTP server on 127.0.0.1 that fronts a ControlService."""

    def __init__(self, service, port=None, log=None, info_file=None):
        """Prepare the server; call start().

        Args: service is the ControlService; port is a fixed port (None:
            try the default range); log receives request notes (any object
            with write); info_file is the discovery file to write (None:
            info_path()).
        Returns: None.
        """
        self.service = service
        self.requested_port = port
        self.log = log
        self.info_file = info_path() if info_file is None else info_file
        self.httpd = None
        self.thread = None
        self._port = None

    @property
    def port(self):
        """Report the bound port.

        Args: None.
        Returns: The port number, or None before a successful start().
        """
        return self._port

    def start(self):
        """Bind a port and serve in a daemon thread.

        Args: None.
        Returns: True when serving; False when every candidate port was
            busy (the app keeps running without remote control).
        """
        if self.requested_port is not None:
            candidates = [self.requested_port]
        else:
            candidates = list(range(DEFAULT_PORT, DEFAULT_PORT + PORT_SCAN))
        for port in candidates:
            httpd = self._bind(port)
            if httpd is not None:
                break
        else:
            return False
        httpd.service = self.service
        httpd.log = self.log
        self.httpd = httpd
        self._port = httpd.server_address[1]
        self.thread = threading.Thread(target=httpd.serve_forever,
                                       kwargs={"poll_interval": 0.2},
                                       name="espi-control", daemon=True)
        self.thread.start()
        write_info(self.info_file, self._port)
        return True

    def _bind(self, port):
        """Try to bind one loopback port.

        Args: port is the port number.
        Returns: The ThreadingHTTPServer, or None when the port is busy.
        """
        try:
            return http.server.ThreadingHTTPServer(
                ("127.0.0.1", port), ControlHTTPRequestHandler)
        except OSError:
            return None

    def stop(self):
        """Stop serving, close the socket and remove the discovery file.

        Args: None.
        Returns: None. Safe to call twice or before start().
        """
        if self.httpd is None:
            return
        httpd, self.httpd = self.httpd, None
        httpd.shutdown()
        httpd.server_close()
        self.thread.join(5.0)
        self.thread = None
        clear_info(self.info_file)


class ControlClientError(Exception):
    """A terminal request that failed; the message is user-facing."""


# Loopback requests must never go through a configured HTTP proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def find_port(path=None):
    """Read the port from the discovery file.

    Args: path is the file to read (None: info_path()).
    Returns: The port, or None when the file is missing or unreadable.
    """
    try:
        with open(path or info_path(), encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and isinstance(data.get("port"), int):
        return data["port"]
    return None


def request(port, method, path, body=None, timeout=5.0):
    """Send one control request over loopback HTTP.

    Args: port is the server port; method is "GET" or "POST"; path starts
        with "/"; body is a JSON-serializable object or None; timeout is
        the socket timeout in seconds.
    Returns: (http status, decoded JSON payload). Raises ControlClientError
        with a user-facing message on any failure.
    """
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"} if body is not None else {},
        method=method)
    try:
        with _OPENER.open(req, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            payload = {}
        raise ControlClientError(payload.get("error", f"HTTP {exc.code}")) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ControlClientError(
            f"could not reach http://127.0.0.1:{port} ({exc}); is ESPI live "
            "running with remote control on?") from None
    except (UnicodeDecodeError, ValueError):
        raise ControlClientError("the app sent a response that is not JSON") from None


def discover(info_file=None):
    """Find the port of a running app.

    Args: info_file is the discovery file to try first (None: info_path()).
    Returns: The port. Raises ControlClientError when nothing answers.
    """
    candidates = []
    port = find_port(info_file)
    if port is not None:
        candidates.append(port)
    candidates += [candidate
                   for candidate in range(DEFAULT_PORT, DEFAULT_PORT + PORT_SCAN)
                   if candidate not in candidates]
    for candidate in candidates:
        try:
            _, payload = request(candidate, "GET", "/", timeout=1.0)
        except ControlClientError:
            continue
        if isinstance(payload, dict) and payload.get("app") == "espi_live":
            return candidate
    raise ControlClientError(
        "no running ESPI live app found; start the app (it serves "
        f"http://127.0.0.1:{DEFAULT_PORT}/ by default) or pass --port")


def _change(text):
    """Parse one KEY=VALUE argument.

    Args: text is the raw command-line argument.
    Returns: A (key, value) pair with the value still as text (the server
        coerces it). Raises argparse.ArgumentTypeError without a separator.
    """
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    return key, value


def _camera_query(spec):
    """Build the query string that selects a camera.

    Args: spec is the --camera value (None: the displayed camera).
    Returns: "" or "?camera=<quoted spec>".
    """
    return "" if spec is None else f"?camera={urllib.parse.quote(str(spec))}"


def add_parser(parser):
    """Add the terminal client's actions to cli.py's control subparser.

    Args: parser is the "control" argparse subparser.
    Returns: None.
    """
    camera_help = ("camera by index, serial or label (default: the displayed "
                   "one)")
    parser.add_argument("--port", type=int, default=None,
                        help="control port (default: discover a running app)")
    actions = parser.add_subparsers(dest="action", required=True,
                                    metavar="ACTION")
    status = actions.add_parser(
        "status", help="summarize the app and every camera")
    status.add_argument("--json", action="store_true", help="print raw JSON")
    status.add_argument("--port", type=int, default=None, help=argparse.SUPPRESS)
    status.set_defaults(handler=_run_status)
    cameras = actions.add_parser("cameras", help="list the cameras")
    cameras.add_argument("--json", action="store_true", help="print raw JSON")
    cameras.add_argument("--port", type=int, default=None, help=argparse.SUPPRESS)
    cameras.set_defaults(handler=_run_cameras)
    show = actions.add_parser("get", help="show settings (all, or named keys)")
    show.add_argument("keys", nargs="*", metavar="KEY",
                      help="setting names to show (default: all)")
    show.add_argument("--camera", default=None, help=camera_help)
    show.add_argument("--json", action="store_true", help="print raw JSON")
    show.add_argument("--port", type=int, default=None, help=argparse.SUPPRESS)
    show.set_defaults(handler=_run_get)
    change = actions.add_parser(
        "set", help="change settings, e.g. set exposure_us=5000 mode=Reference")
    change.add_argument("changes", nargs="+", type=_change, metavar="KEY=VALUE")
    change.add_argument("--camera", default=None, help=camera_help)
    change.add_argument("--json", action="store_true", help="print raw JSON")
    change.add_argument("--port", type=int, default=None, help=argparse.SUPPRESS)
    change.set_defaults(handler=_run_set)
    command = actions.add_parser(
        "run", help="run a one-shot command, e.g. run save title=Band notes=...")
    command.add_argument("name", metavar="COMMAND", choices=sorted(COMMANDS),
                         help="the one-shot command (run shows the list)")
    command.add_argument("payload", nargs="*", type=_change,
                         metavar="KEY=VALUE",
                         help="command arguments (title, notes)")
    command.add_argument("--camera", default=None, help=camera_help)
    command.add_argument("--json", action="store_true", help="print raw JSON")
    command.add_argument("--port", type=int, default=None, help=argparse.SUPPRESS)
    command.set_defaults(handler=_run_command)
    log = actions.add_parser("log", help="show the run log")
    log.add_argument("--lines", type=int, default=DEFAULT_LOG_LINES,
                     help=f"how many lines (default {DEFAULT_LOG_LINES})")
    log.add_argument("--port", type=int, default=None, help=argparse.SUPPRESS)
    log.set_defaults(handler=_run_log)
    _add_daq_parser(actions)
    update = actions.add_parser(
        "update", help="show the update state, or check, download, install or restart")
    update.add_argument("update_action", nargs="?", default=None,
                        choices=sorted(UPDATE_ACTIONS), metavar="ACTION",
                        help="check, download, install or restart (default: show the state)")
    update.add_argument("--json", action="store_true", help="print raw JSON")
    update.add_argument("--port", type=int, default=None, help=argparse.SUPPRESS)
    update.set_defaults(handler=_run_update)


def _daq_flags(parser):
    """Add --json and --port to one daq subcommand parser.

    Args: parser is the parser to add the flags to (a subparser of daq, or
        daq itself for the bare status form).
    Returns: None.
    """
    parser.add_argument("--json", action="store_true", help="print raw JSON")
    parser.add_argument("--port", type=int, default=None, help=argparse.SUPPRESS)


def _add_daq_parser(actions):
    """Add the "daq" action and its subcommands to cli.py's control parser.

    Args: actions is the control parser's action subparsers.
    Returns: None.
    """
    daq_parser = actions.add_parser(
        "daq", help="NI DAQ: frequency generator and piezo (status by itself)")
    _daq_flags(daq_parser)
    daq_parser.set_defaults(handler=_run_daq, daq_verb="")
    verbs = daq_parser.add_subparsers(dest="daq_verb", metavar="DAQ ACTION")

    def verb(name, help_text, body):
        """Add one daq subcommand.

        Args: name is the subcommand; help_text is its help; body is the
            POST /daq body template ({"action": name} plus fixed fields).
        Returns: The subparser.
        """
        parser = verbs.add_parser(name, help=help_text)
        _daq_flags(parser)
        parser.set_defaults(handler=_run_daq, daq_verb=name, body=body)
        return parser

    verb("scan", "list the NI devices (plus the simulated one)",
         {"action": "scan"})
    connect = verb("connect", "connect to a device; every output goes to 0 V",
                   {"action": "connect"})
    connect.add_argument("device", metavar="DEVICE",
                         help='a DAQmx name ("Dev1") or "Simulated"')
    apply_command = verb("apply", "change DAQ settings and write them",
                         {"action": "apply"})
    apply_command.add_argument("changes", nargs="*", type=_change,
                               metavar="KEY=VALUE",
                               help="DaqSettings fields (control daq shows them)")
    verb("zero", "set every output to 0 V", {"action": "zero"})
    verb("self-test", "step each analog output and read it back",
         {"action": "self_test"})
    step = verb("step", "hold the piezo at a level, in volts at the DAQ",
                {"action": "step"})
    step.add_argument("volts", type=float, metavar="VOLTS",
                      help="the level to hold (at most the piezo safety limit)")
    attach = verb("attach", "give the piezo to a camera's Phase view",
                  {"action": "attach"})
    attach.add_argument("camera", nargs="?", default=None, metavar="CAMERA",
                        help="camera by index, serial or label (default: the "
                             "displayed one)")
    verb("detach", "take the piezo away from the Phase view",
         {"action": "detach"})
    verb("checklist", "show the safety checklist on the app's screen",
         {"action": "checklist"})
    tune = verb("tune", "start or stop the experimental piezo auto-tune",
                {"action": "tune"})
    tune.add_argument("state", choices=("on", "off"), metavar="on|off")


def run_cli(args):
    """Run one control action from cli.py's parsed arguments.

    Args: args is the parsed namespace; args.handler is the action.
    Returns: The process exit code (0 ok, 1 failed).
    """
    try:
        port = args.port if args.port is not None else discover()
        return args.handler(args, port)
    except ControlClientError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run_status(args, port):
    """Print the status summary.

    Args: args carries --json; port is the server port.
    Returns: The exit code, 0.
    """
    _, payload = request(port, "GET", "/status")
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print(f"{payload['build']}  pid {payload['pid']}  "
          f"http://127.0.0.1:{port}/")
    for channel in payload["channels"]:
        marks = [mark for mark, on in (("displayed", channel["displayed"]),
                                       ("demo", channel["demo"]),
                                       ("off", not channel["active"])) if on]
        print(f"[{channel['index']}] {channel['serial']}"
              + (f"  ({', '.join(marks)})" if marks else "")
              + f"  {channel['status'].get('camera_fps', '')}")
    return 0


def _run_cameras(args, port):
    """Print the camera list.

    Args: args carries --json; port is the server port.
    Returns: The exit code, 0.
    """
    _, payload = request(port, "GET", "/status")
    if args.json:
        print(json.dumps(payload["channels"], indent=2))
        return 0
    for channel in payload["channels"]:
        state = "displayed" if channel["displayed"] else (
            "active" if channel["active"] else "off")
        print(f"[{channel['index']}] {channel['label']}  {state}"
              + ("  demo" if channel["demo"] else ""))
    return 0


def _run_get(args, port):
    """Print one camera's settings.

    Args: args carries --camera, --json and the keys; port is the server
        port.
    Returns: The exit code, 0, or 1 for unknown key names.
    """
    _, payload = request(port, "GET", "/settings" + _camera_query(args.camera))
    settings = payload["settings"]
    unknown = [key for key in args.keys if key not in settings]
    if unknown:
        raise ControlClientError(
            f"unknown settings: {', '.join(unknown)}; GET / lists the names")
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    for key in (args.keys or list(settings)):
        print(f"{key} = {settings[key]}")
    return 0


def _run_set(args, port):
    """Apply settings and print the resulting values.

    Args: args carries the KEY=VALUE pairs, --camera and --json; port is
        the server port.
    Returns: The exit code, 0, or 1 when the server refused.
    """
    body = dict(args.changes)
    if args.camera is not None:
        body["camera"] = args.camera
    _, payload = request(port, "POST", "/settings", body)
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    for key, value in payload["applied"].items():
        print(f"{key} = {value}")
    return 0


def _run_update(args, port):
    """Show the update state, or run one update action.

    Args: args carries the optional update_action and --json; port is the
        server port.
    Returns: The exit code, 0, or 1 when the server refused.
    """
    if args.update_action is None:
        _, payload = request(port, "GET", "/update")
    else:
        _, payload = request(port, "POST", "/update", {"action": args.update_action})
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    if args.update_action is not None:
        print(f"update {args.update_action}: started "
              f"{payload['result']['started']}")
        return 0
    print(f"build {payload['current']}; newest known: "
          f"{payload['label'] or 'none'}")
    if payload["unavailable"]:
        print(f"updating unavailable: {payload['unavailable']}")
    print(f"downloaded: {'yes' if payload['downloaded'] else 'no'}"
          f"  busy: {'yes' if payload['busy'] else 'no'}"
          f"  recording: {'yes' if payload['recording'] else 'no'}")
    return 0


def _run_command(args, port):
    """Run a one-shot command and print its result.

    Args: args carries the command name, its KEY=VALUE payload, --camera
        and --json; port is the server port.
    Returns: The exit code, 0, or 1 when the server refused.
    """
    body = {"command": args.name}
    body.update(dict(args.payload))
    if args.camera is not None:
        body["camera"] = args.camera
    _, payload = request(port, "POST", "/command", body)
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    result = payload.get("result") or {}
    if result:
        print("; ".join(f"{key}: {value}" for key, value in result.items()))
    else:
        print(f"ran {payload['command']} on camera {payload['camera']}")
    return 0


def _run_log(args, port):
    """Print the run log's tail.

    Args: args carries --lines; port is the server port.
    Returns: The exit code, 0.
    """
    _, payload = request(port, "GET", f"/log?lines={args.lines}")
    for line in payload["lines"]:
        print(line)
    return 0


def _run_daq(args, port):
    """Run one NI DAQ action, or show the DAQ's state by itself.

    Args: args carries daq_verb, an optional body template, the verb's own
        arguments and --json; port is the server port.
    Returns: The exit code, 0, or 1 when the server refused.
    """
    if not args.daq_verb:
        _, payload = request(port, "GET", "/daq")
    else:
        body = dict(getattr(args, "body", {}) or {})
        if getattr(args, "device", None) is not None:
            body["device"] = args.device
        if hasattr(args, "changes"):  # apply: no pairs pushes the saved settings
            body["settings"] = dict(args.changes or {})
        if getattr(args, "volts", None) is not None:
            body["volts"] = args.volts
        if getattr(args, "camera", None) is not None:
            body["camera"] = args.camera
        if args.daq_verb == "tune":
            body["on"] = args.state == "on"
        _, payload = request(port, "POST", "/daq", body)
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    if args.daq_verb:
        result = payload.get("result") or {}
        if result:
            print("; ".join(f"{key}: {value}" for key, value in result.items()))
        else:
            print(f"ran {payload['action']}")
        return 0
    daq_status = payload
    device = daq_status.get("device")
    if not daq_status.get("connected"):
        print("NI DAQ: not connected. "
              "`espi_live control daq connect Simulated` tries the controls "
              "without hardware.")
        return 0
    print(f"NI DAQ: {device['label']}"
          + ("" if daq_status.get("active") else "  (idle)"))
    print(f"Outputs: {daq_status.get('summary', '')}")
    print("Safety checklist: "
          + ("ticked off for this connection" if daq_status.get("checklist_done")
             else "not ticked off; apply needs it (daq checklist)"))
    applied = daq_status.get("settings")
    if applied:
        for key, value in applied.items():
            print(f"{key} = {value}")
    else:
        print("settings: none applied since connecting or zeroing")
    tuning = daq_status.get("tuning") or {}
    if tuning.get("on") or tuning.get("note"):
        print(f"auto-tune: {'on' if tuning.get('on') else 'off'}"
              + (f" ({tuning['note']})" if tuning.get("note") else ""))
    return 0
