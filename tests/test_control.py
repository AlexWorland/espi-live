"""Camera-free checks for the remote-control server and terminal client.

No camera, external network, real output folder or running app is touched:
the backend is a fake, the HTTP handler is driven with byte streams instead
of a socket, the server binds through a stub (plus one real loopback round
trip on an OS-chosen port), and the client's opener is replaced.

Run from the project root:  PYTHONPATH=src python -m unittest tests.test_control
"""

import argparse
import contextlib
import http.server
import io
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
from types import SimpleNamespace
from unittest import mock

from espi_live import control
from espi_live.engine import Settings


class FakeBackend:
    """In-memory stand-in for the app adapter (qt_app.ControlBackend)."""

    def __init__(self, count=2):
        """Create count channels; the first is displayed, active and demo.

        Args: count is how many channels to fake.
        Returns: None.
        """
        self.channels_data = [
            {"index": index, "label": f"CH{index}", "serial": f"S{index}",
             "demo": index == 0, "active": index == 0, "displayed": index == 0}
            for index in range(count)]
        self.settings_data = [Settings() for _ in range(count)]
        self.limits_data = ([{"ExposureTime": (1.0, 100.0)}]
                            + [{} for _ in range(count - 1)])
        self.status_data = ([{"camera_fps": "Camera: 30.0 fps"}]
                            + [{} for _ in range(count - 1)])
        self.applied = []  # (index, changes) passed to apply()
        self.commands = []  # (index, name, payload) passed to command()
        self.log_lines = ["one", "two", "three"]
        self.fail = None  # exception raised by every call when set

    def _check(self):
        """Raise the configured failure, if any.

        Args: None.
        Returns: None.
        """
        if self.fail is not None:
            raise self.fail

    def channels(self):
        """Return the channel descriptions.

        Args: None.
        Returns: A fresh list of dicts.
        """
        self._check()
        return [dict(channel) for channel in self.channels_data]

    def settings(self, index):
        """Return one channel's settings dict.

        Args: index is the channel's position.
        Returns: The settings as a plain dict.
        """
        self._check()
        settings = self.settings_data[index]
        return {name: getattr(settings, name) for name in control.SETTING_TYPES}

    def limits(self, index):
        """Return one channel's limits.

        Args: index is the channel's position.
        Returns: {node name: [low, high]}.
        """
        self._check()
        return {name: list(bounds)
                for name, bounds in self.limits_data[index].items()}

    def status(self, index):
        """Return one channel's status dict.

        Args: index is the channel's position.
        Returns: A copy of the status dict.
        """
        self._check()
        return dict(self.status_data[index])

    def apply(self, index, changes):
        """Record and apply settings.

        Args: index is the channel's position; changes are coerced values.
        Returns: The resulting settings dict.
        """
        self._check()
        self.applied.append((index, dict(changes)))
        for name, value in changes.items():
            setattr(self.settings_data[index], name, value)
        return self.settings(index)

    def command(self, index, name, payload):
        """Record a command.

        Args: index is the channel's position; name is the command;
            payload is its arguments.
        Returns: A small result dict.
        """
        self._check()
        self.commands.append((index, name, dict(payload)))
        return {"requested": name}

    def log_tail(self, lines):
        """Return the last log lines.

        Args: lines is how many to return.
        Returns: The tail of self.log_lines.
        """
        self._check()
        return self.log_lines[-lines:]


def response_of(raw):
    """Split a handler's raw HTTP response into status and payload.

    Args: raw is the bytes written to the fake wfile.
    Returns: (status code, decoded JSON payload).
    """
    head, _, body = raw.partition(b"\r\n\r\n")
    return int(head.split(b" ")[1]), json.loads(body.decode())


def run_handler(method, path, body=None, service=None, log=None):
    """Drive one request through the HTTP handler without a socket.

    Args: method is "GET" or "POST"; path is the request path; body is the
        JSON-serializable request body (None for none); service is the
        ControlService on the fake server; log receives request notes.
    Returns: (status code, decoded JSON payload).
    """
    data = b"" if body is None else json.dumps(body).encode()
    return run_raw(method, path, data, service=service, log=log)


def run_raw(method, path, data, service=None, log=None):
    """Drive one request with a raw body through the HTTP handler.

    Args: method and path describe the request; data is the exact body
        bytes (b"" for none); service and log ride on the fake server.
    Returns: (status code, decoded JSON payload).
    """
    handler = object.__new__(control.ControlHTTPRequestHandler)
    handler.rfile = io.BytesIO(data)
    handler.wfile = io.BytesIO()
    handler.headers = {"Content-Length": str(len(data))}
    handler.command = method
    handler.path = path
    handler.requestline = path
    handler.request_version = "HTTP/1.1"
    handler.server = SimpleNamespace(service=service, log=log)
    if method == "GET":
        handler.do_GET()
    else:
        handler.do_POST()
    return response_of(handler.wfile.getvalue())


class ServiceTestCase(unittest.TestCase):
    """Shared setup: a fake backend and a fixed build description."""

    def setUp(self):
        """Create the service under test.

        Args: None.
        Returns: None.
        """
        self.backend = FakeBackend()
        self.service = control.ControlService(self.backend)
        patcher = mock.patch.object(control, "describe_build",
                                    return_value="build 7")
        self.describe = patcher.start()
        self.addCleanup(patcher.stop)


class CoercionTests(unittest.TestCase):
    """Value coercion for settings sent as JSON."""

    def test_schema_covers_every_settings_field(self):
        """SETTING_TYPES names exactly the Settings fields.

        Args: None.
        Returns: None.
        """
        self.assertEqual(set(control.SETTING_TYPES), set(Settings.__dataclass_fields__))

    def test_enums_accept_their_options_and_reject_others(self):
        """Enum fields take option names and refuse everything else.

        Args: None.
        Returns: None.
        """
        self.assertEqual(control.coerce_setting("mode", "Reference"), "Reference")
        self.assertEqual(control.coerce_setting("demo_object", "Still"), "Still")
        with self.assertRaises(control.ControlError) as raised:
            control.coerce_setting("mode", "Rainbow")
        self.assertIn("must be one of", str(raised.exception))
        self.assertEqual(raised.exception.status, 400)

    def test_text_fields_take_strings_only(self):
        """A plain text field accepts a string and refuses anything else.

        Args: None.
        Returns: None.
        """
        saved = dict(control.SETTING_TYPES)
        control.SETTING_TYPES["nickname"] = ("str", None)
        try:
            self.assertEqual(control.coerce_setting("nickname", "lab bench"),
                             "lab bench")
            with self.assertRaises(control.ControlError):
                control.coerce_setting("nickname", 5)
        finally:
            control.SETTING_TYPES.clear()
            control.SETTING_TYPES.update(saved)

    def test_bools_accept_words_numbers_and_reject_garbage(self):
        """Bool fields take true/false words, 0/1, and real bools.

        Args: None.
        Returns: None.
        """
        for value, expected in (("true", True), ("ON", True), (1, True), (True, True),
                                ("no", False), (0, False), (False, False)):
            self.assertIs(control.coerce_setting("invert", value), expected)
        with self.assertRaises(control.ControlError):
            control.coerce_setting("invert", "maybe")

    def test_ints_accept_whole_numbers_and_reject_fractions(self):
        """Int fields take ints, whole floats and numeric strings.

        Args: None.
        Returns: None.
        """
        self.assertEqual(control.coerce_setting("average", 4), 4)
        self.assertEqual(control.coerce_setting("average", 4.0), 4)
        self.assertEqual(control.coerce_setting("average", "8"), 8)
        for value in (4.5, "four", None, True):
            with self.assertRaises(control.ControlError):
                control.coerce_setting("average", value)

    def test_floats_accept_numbers_and_reject_garbage(self):
        """Float fields take numbers, numeric strings and reject junk.

        Args: None.
        Returns: None.
        """
        self.assertEqual(control.coerce_setting("exposure_us", 5000), 5000.0)
        self.assertEqual(control.coerce_setting("exposure_us", "0.5"), 0.5)
        for value in ("bright", None, True):
            with self.assertRaises(control.ControlError):
                control.coerce_setting("exposure_us", value)
        with self.assertRaises(control.ControlError):
            control.coerce_setting("gamma", float("nan"))


class ClampingTests(ServiceTestCase):
    """Values are clamped to the ranges the controls accept."""

    def test_clamp_setting_uses_the_profile_ranges(self):
        """Numbers land inside profiles.RANGES; unlisted fields pass through.

        Args: None.
        Returns: None.
        """
        self.assertEqual(control.clamp_setting("average", 100), 60)
        self.assertEqual(control.clamp_setting("average", 0), 1)
        self.assertEqual(control.clamp_setting("average", 8), 8)
        self.assertEqual(control.clamp_setting("gamma", 0.0), 0.1)
        self.assertIs(control.clamp_setting("rolling", True), True)
        self.assertEqual(control.clamp_setting("mode", "Raw"), "Raw")

    def test_post_settings_clamps_to_the_ranges(self):
        """POST /settings reports the values actually applied.

        Args: None.
        Returns: None.
        """
        code, payload = self.service.handle("POST", "/settings", {},
                                            {"average": 100, "gamma": 99})
        self.assertEqual(code, 200)
        self.assertEqual(payload["applied"], {"average": 60, "gamma": 3.0})
        self.assertEqual(self.backend.applied,
                         [(0, {"average": 60, "gamma": 3.0})])


class OverviewTests(ServiceTestCase):
    """GET / describes the API."""

    def test_overview_lists_settings_commands_and_endpoints(self):
        """The overview names every setting, command and endpoint.

        Args: None.
        Returns: None.
        """
        code, payload = self.service.handle("GET", "/", {}, None)
        self.assertEqual(code, 200)
        self.assertEqual(payload["app"], "espi_live")
        self.assertEqual(payload["build"], "build 7")
        self.assertEqual(payload["pid"], os.getpid())
        self.assertEqual(set(payload["settings"]), set(control.SETTING_TYPES))
        self.assertEqual(payload["settings"]["mode"],
                         {"type": "enum",
                          "options": ["Raw", "Previous", "Reference", "Phase"]})
        self.assertEqual(payload["settings"]["exposure_us"],
                         {"type": "float", "range": [0.0, 50000.0]})
        self.assertEqual(payload["settings"]["rolling"], {"type": "bool"})
        self.assertEqual(set(payload["commands"]), set(control.COMMANDS))
        self.assertIn("GET /status", payload["endpoints"])


class StatusTests(ServiceTestCase):
    """GET /status reports every channel."""

    def test_status_details_every_channel(self):
        """Each channel carries settings, limits and status.

        Args: None.
        Returns: None.
        """
        code, payload = self.service.handle("GET", "/status", {}, None)
        self.assertEqual(code, 200)
        self.assertEqual(payload["current"], 0)
        self.assertEqual(len(payload["channels"]), 2)
        first = payload["channels"][0]
        self.assertEqual(first["serial"], "S0")
        self.assertTrue(first["displayed"])
        self.assertEqual(first["settings"]["mode"], "Raw")
        self.assertEqual(first["limits"], {"ExposureTime": [1.0, 100.0]})
        self.assertEqual(first["status"], {"camera_fps": "Camera: 30.0 fps"})


class ResolveTests(ServiceTestCase):
    """Camera selection by index, serial, label or default."""

    def test_default_is_the_displayed_camera(self):
        """With no spec the displayed camera answers.

        Args: None.
        Returns: None.
        """
        code, payload = self.service.handle("GET", "/settings", {}, None)
        self.assertEqual(code, 200)
        self.assertEqual(payload["camera"], 0)

    def test_selection_by_index_serial_and_label(self):
        """Index, serial and label (case-insensitive) all resolve.

        Args: None.
        Returns: None.
        """
        for spec, expected in ((1, 1), ("s1", 1), ("CH1", 1), ("ch1", 1), ("0", 0)):
            code, payload = self.service.handle("GET", "/settings",
                                                {"camera": spec}, None)
            self.assertEqual((code, payload["camera"]), (200, expected), spec)

    def test_unknown_camera_lists_the_choices(self):
        """A bad spec is refused with the available cameras named.

        Args: None.
        Returns: None.
        """
        code, payload = self.service.handle("GET", "/settings",
                                            {"camera": "nope"}, None)
        self.assertEqual(code, 404)
        self.assertIn("0:S0, 1:S1", payload["error"])

    def test_out_of_range_index_is_refused(self):
        """An integer index outside the list is refused.

        Args: None.
        Returns: None.
        """
        code, payload = self.service.handle("GET", "/settings",
                                            {"camera": 9}, None)
        self.assertEqual(code, 404)

    def test_no_channels_at_all(self):
        """An app with no channels still answers with a clear error.

        Args: None.
        Returns: None.
        """
        backend = FakeBackend(count=0)
        service = control.ControlService(backend)
        code, payload = service.handle("GET", "/settings", {}, None)
        self.assertEqual(code, 404)
        self.assertIn("cameras: none", payload["error"])


class PostSettingsTests(ServiceTestCase):
    """POST /settings applies coerced values."""

    def test_settings_are_coerced_and_applied(self):
        """String values are coerced and reach the backend.

        Args: None.
        Returns: None.
        """
        code, payload = self.service.handle("POST", "/settings", {}, {
            "exposure_us": "5000", "mode": "Reference", "record": "true"})
        self.assertEqual(code, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["applied"],
                         {"exposure_us": 5000.0, "mode": "Reference",
                          "record": True})
        self.assertEqual(self.backend.applied, [(0, payload["applied"])])
        self.assertEqual(payload["settings"]["exposure_us"], 5000.0)

    def test_camera_key_selects_the_target(self):
        """The camera key chooses the channel and is not applied.

        Args: None.
        Returns: None.
        """
        code, payload = self.service.handle("POST", "/settings", {},
                                            {"camera": 1, "gain_db": 5})
        self.assertEqual(code, 200)
        self.assertEqual(self.backend.applied, [(1, {"gain_db": 5.0})])

    def test_unknown_key_lists_the_valid_names(self):
        """An unknown setting name is refused with guidance.

        Args: None.
        Returns: None.
        """
        code, payload = self.service.handle("POST", "/settings", {},
                                            {"saturation": 5})
        self.assertEqual(code, 400)
        self.assertIn("exposure_us", payload["error"])

    def test_bad_value_is_refused(self):
        """A value that cannot be coerced is refused.

        Args: None.
        Returns: None.
        """
        code, payload = self.service.handle("POST", "/settings", {},
                                            {"average": "lots"})
        self.assertEqual(code, 400)
        self.assertIn("average", payload["error"])

    def test_empty_change_set_is_refused(self):
        """A body with no settings is refused.

        Args: None.
        Returns: None.
        """
        code, payload = self.service.handle("POST", "/settings", {},
                                            {"camera": 0})
        self.assertEqual(code, 400)
        self.assertIn("no settings", payload["error"])

    def test_body_must_be_an_object(self):
        """A JSON list or string is refused.

        Args: None.
        Returns: None.
        """
        for body in ([1, 2], "exposure_us=5", 5):
            code, payload = self.service.handle("POST", "/settings", {}, body)
            self.assertEqual(code, 400, body)
            self.assertIn("JSON object", payload["error"])

    def test_backend_refusals_keep_their_status(self):
        """ControlError from the backend maps to its HTTP status.

        Args: None.
        Returns: None.
        """
        self.backend.fail = control.ControlError("camera 1 is off", 409)
        code, payload = self.service.handle("POST", "/settings", {},
                                            {"exposure_us": 5000})
        self.assertEqual(code, 409)
        self.assertEqual(payload["error"], "camera 1 is off")

    def test_unexpected_backend_errors_report_500(self):
        """Any other backend exception becomes a 500 with the reason.

        Args: None.
        Returns: None.
        """
        self.backend.fail = ValueError("boom")
        code, payload = self.service.handle("POST", "/settings", {},
                                            {"exposure_us": 5000})
        self.assertEqual(code, 500)
        self.assertIn("ValueError: boom", payload["error"])


class PostCommandTests(ServiceTestCase):
    """POST /command runs one-shot commands."""

    def test_commands_reach_the_backend(self):
        """Every command name routes with its payload.

        Args: None.
        Returns: None.
        """
        for name, payload in (("save", {"title": "T", "notes": "N"}),
                              ("reference", {}), ("reset_average", {}),
                              ("display", {}), ("processing_on", {})):
            code, response = self.service.handle("POST", "/command", {},
                                                 {"command": name, **payload})
            self.assertEqual(code, 200, name)
            self.assertTrue(response["ok"])
        self.assertEqual([entry[1] for entry in self.backend.commands],
                         ["save", "reference", "reset_average", "display",
                          "processing_on"])
        self.assertEqual(self.backend.commands[0],
                         (0, "save", {"title": "T", "notes": "N"}))

    def test_unknown_command_lists_the_choices(self):
        """An unknown command name is refused with the list.

        Args: None.
        Returns: None.
        """
        code, payload = self.service.handle("POST", "/command", {},
                                            {"command": "explode"})
        self.assertEqual(code, 400)
        self.assertIn("display", payload["error"])

    def test_missing_command_is_refused(self):
        """A body without a command is refused.

        Args: None.
        Returns: None.
        """
        code, payload = self.service.handle("POST", "/command", {},
                                            {"camera": 0})
        self.assertEqual(code, 400)
        self.assertIn("unknown command", payload["error"])

    def test_save_rejects_unexpected_payload_keys(self):
        """Payload keys a command does not take are refused.

        Args: None.
        Returns: None.
        """
        code, payload = self.service.handle("POST", "/command", {},
                                            {"command": "reference", "title": "T"})
        self.assertEqual(code, 400)
        self.assertIn("does not take", payload["error"])

    def test_save_rejects_non_string_payload(self):
        """save's title and notes must be strings.

        Args: None.
        Returns: None.
        """
        code, payload = self.service.handle("POST", "/command", {},
                                            {"command": "save", "title": 5})
        self.assertEqual(code, 400)
        self.assertIn("title", payload["error"])

    def test_body_must_be_an_object(self):
        """A JSON list is refused.

        Args: None.
        Returns: None.
        """
        code, payload = self.service.handle("POST", "/command", {}, ["save"])
        self.assertEqual(code, 400)
        self.assertIn("JSON object", payload["error"])

    def test_camera_selects_the_target(self):
        """The camera key routes the command to that channel.

        Args: None.
        Returns: None.
        """
        code, payload = self.service.handle("POST", "/command", {},
                                            {"command": "reference", "camera": "S1"})
        self.assertEqual(code, 200)
        self.assertEqual(payload["camera"], 1)
        self.assertEqual(self.backend.commands, [(1, "reference", {})])


class LogTests(ServiceTestCase):
    """GET /log returns the run log tail."""

    def test_log_defaults_and_lines(self):
        """The default returns the tail; ?lines=N controls the length.

        Args: None.
        Returns: None.
        """
        code, payload = self.service.handle("GET", "/log", {}, None)
        self.assertEqual((code, payload["lines"]), (200, ["one", "two", "three"]))
        code, payload = self.service.handle("GET", "/log", {"lines": "1"}, None)
        self.assertEqual(payload["lines"], ["three"])

    def test_bad_lines_are_refused(self):
        """Non-numeric and out-of-range line counts are refused.

        Args: None.
        Returns: None.
        """
        for value in ("lots", "0", "-1", str(control.MAX_LOG_LINES + 1)):
            code, payload = self.service.handle("GET", "/log",
                                                {"lines": value}, None)
            self.assertEqual(code, 400, value)


class RoutingTests(ServiceTestCase):
    """Method and path dispatch."""

    def test_unknown_path_is_404(self):
        """An unknown address suggests GET /.

        Args: None.
        Returns: None.
        """
        code, payload = self.service.handle("GET", "/nope", {}, None)
        self.assertEqual(code, 404)
        self.assertIn("GET /", payload["error"])

    def test_unknown_method_is_405(self):
        """Methods other than GET and POST are refused.

        Args: None.
        Returns: None.
        """
        code, payload = self.service.handle("PUT", "/settings", {}, None)
        self.assertEqual(code, 405)

    def test_trailing_slashes_are_tolerated(self):
        """/status/ answers like /status.

        Args: None.
        Returns: None.
        """
        code, payload = self.service.handle("GET", "/status/", {}, None)
        self.assertEqual(code, 200)
        self.assertEqual(payload["app"], "espi_live")


class HandlerTests(ServiceTestCase):
    """The HTTP adapter, driven with byte streams instead of a socket."""

    def setUp(self):
        """Extend the service setup with a log recorder.

        Args: None.
        Returns: None.
        """
        super().setUp()
        self.log = SimpleNamespace(lines=[])

        def write(message):
            self.log.lines.append(message)

        self.log.write = write

    def test_get_overview(self):
        """A GET reaches the service and answers JSON.

        Args: None.
        Returns: None.
        """
        code, payload = run_handler("GET", "/", service=self.service)
        self.assertEqual(code, 200)
        self.assertEqual(payload["app"], "espi_live")

    def test_get_with_query_string(self):
        """Query parameters reach the service.

        Args: None.
        Returns: None.
        """
        code, payload = run_handler("GET", "/settings?camera=1",
                                    service=self.service)
        self.assertEqual((code, payload["camera"]), (200, 1))

    def test_post_settings(self):
        """A POST body reaches the service.

        Args: None.
        Returns: None.
        """
        code, payload = run_handler("POST", "/settings",
                                    {"exposure_us": 5000},
                                    service=self.service, log=self.log)
        self.assertEqual(code, 200)
        self.assertEqual(payload["applied"], {"exposure_us": 5000.0})
        self.assertTrue(any("POST /settings" in line for line in self.log.lines))

    def test_post_with_broken_body_is_refused(self):
        """Non-JSON, empty and oversized bodies are refused.

        Args: None.
        Returns: None.
        """
        code, payload = run_raw("POST", "/settings", b"{not json",
                                service=self.service)
        self.assertEqual(code, 400)
        self.assertIn("not valid JSON", payload["error"])
        code, payload = run_raw("POST", "/settings", b"", service=self.service)
        self.assertEqual(code, 400)
        self.assertIn("JSON body is required", payload["error"])
        code, payload = run_raw("POST", "/settings",
                                b"x" * (control.MAX_BODY_BYTES + 1),
                                service=self.service)
        self.assertEqual(code, 413)

    def test_post_with_bad_content_length_is_refused(self):
        """A non-numeric Content-Length is refused.

        Args: None.
        Returns: None.
        """
        handler = object.__new__(control.ControlHTTPRequestHandler)
        handler.rfile = io.BytesIO(b"{}")
        handler.wfile = io.BytesIO()
        handler.headers = {"Content-Length": "lots"}
        handler.command = "POST"
        handler.path = "/settings"
        handler.requestline = "/settings"
        handler.request_version = "HTTP/1.1"
        handler.server = SimpleNamespace(service=self.service, log=self.log)
        handler.do_POST()
        code, payload = response_of(handler.wfile.getvalue())
        self.assertEqual(code, 400)
        self.assertIn("Content-Length", payload["error"])

    def test_service_refusals_pass_through(self):
        """ControlError status and message reach the client.

        Args: None.
        Returns: None.
        """
        code, payload = run_handler("GET", "/nope", service=self.service)
        self.assertEqual(code, 404)
        self.assertIn("unknown address", payload["error"])

    def test_notes_go_to_the_log(self):
        """Refused requests and changes are noted in the app log.

        Args: None.
        Returns: None.
        """
        run_handler("GET", "/nope", service=self.service, log=self.log)
        run_handler("POST", "/command", {"command": "reference"},
                    service=self.service, log=self.log)
        self.assertIn("refused (404)", self.log.lines[0])
        self.assertIn("POST /command", self.log.lines[1])

    def test_notes_without_a_log_are_dropped(self):
        """A server without a log stays silent.

        Args: None.
        Returns: None.
        """
        code, _ = run_handler("GET", "/nope", service=self.service, log=None)
        self.assertEqual(code, 404)

    def test_hung_up_client_is_ignored(self):
        """A write error while answering does not raise.

        Args: None.
        Returns: None.
        """

        class Broken(io.BytesIO):
            """A stream whose writes fail, as a disconnected socket would."""

            def write(self, data):
                """Refuse every write.

                Args: data is ignored.
                Returns: None. Raises BrokenPipeError.
                """
                raise BrokenPipeError()

        handler = object.__new__(control.ControlHTTPRequestHandler)
        handler.rfile = io.BytesIO(b"")
        handler.wfile = Broken()
        handler.headers = {"Content-Length": "0"}
        handler.command = "GET"
        handler.path = "/"
        handler.requestline = "/"
        handler.request_version = "HTTP/1.1"
        handler.server = SimpleNamespace(service=self.service, log=None)
        handler.do_GET()  # must not raise

    def test_handler_error_fallback(self):
        """A ControlError escaping the service still maps to a response.

        Args: None.
        Returns: None.
        """
        service = mock.MagicMock()
        service.handle.side_effect = control.ControlError("late", 418)
        code, payload = run_handler("GET", "/", service=service)
        self.assertEqual(code, 418)
        self.assertEqual(payload["error"], "late")


class FakeHTTPD:
    """Stand-in for ThreadingHTTPServer: records the lifecycle calls."""

    def __init__(self, port):
        """Remember the port it was asked to bind.

        Args: port is the requested port number.
        Returns: None.
        """
        self.server_address = ("127.0.0.1", port)
        self.served = 0
        self.closed = 0
        self.service = None
        self.log = None

    def serve_forever(self, poll_interval=0.5):
        """Count one serve call and return immediately.

        Args: poll_interval is ignored.
        Returns: None.
        """
        self.served += 1

    def shutdown(self):
        """Count one shutdown call.

        Args: None.
        Returns: None.
        """
        self.closed += 1

    def server_close(self):
        """Count one close call.

        Args: None.
        Returns: None.
        """
        self.closed += 1


class ServerTests(unittest.TestCase):
    """Server lifecycle with a stubbed bind."""

    def setUp(self):
        """Prepare a temp discovery file and a bound-port record.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.info_file = os.path.join(self.tmp.name, "control.json")
        self.bound = []

    def fake_bind(self, result):
        """Return a _bind replacement that records its port.

        Args: result is the FakeHTTPD to return, or None to refuse.
        Returns: A _bind-compatible callable.
        """

        def bind(port):
            self.bound.append(port)
            return result

        return bind

    def test_start_binds_writes_the_info_file_and_serves(self):
        """A successful start serves on a thread and writes the file.

        Args: None.
        Returns: None.
        """
        httpd = FakeHTTPD(8364)
        server = control.ControlServer(control.ControlService(FakeBackend()),
                                       log=None, info_file=self.info_file)
        with mock.patch.object(server, "_bind", self.fake_bind(httpd)):
            self.assertTrue(server.start())
        self.assertEqual(server.port, 8364)
        self.assertEqual(self.bound, [8364])
        self.assertEqual(httpd.service.app if hasattr(httpd.service, "app")
                         else None, None)  # service attached
        self.assertIsNotNone(httpd.service)
        with open(self.info_file, encoding="utf-8") as file:
            self.assertEqual(json.load(file)["port"], 8364)
        deadline = time.monotonic() + 5.0
        while httpd.served == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(httpd.served, 1)
        server.stop()

    def test_start_scans_ports_until_one_is_free(self):
        """Busy ports are skipped in order.

        Args: None.
        Returns: None.
        """
        httpd = FakeHTTPD(8365)
        server = control.ControlServer(control.ControlService(FakeBackend()),
                                       info_file=self.info_file)
        counter = {"n": 0}

        def bind(port):
            counter["n"] += 1
            return httpd if counter["n"] == 2 else None

        with mock.patch.object(server, "_bind", bind):
            self.assertTrue(server.start())
        self.assertEqual(server.port, 8365)
        server.stop()

    def test_start_with_a_fixed_port_tries_only_that_port(self):
        """A requested port is the only candidate.

        Args: None.
        Returns: None.
        """
        server = control.ControlServer(control.ControlService(FakeBackend()),
                                       port=9000, info_file=self.info_file)
        with mock.patch.object(server, "_bind", self.fake_bind(None)):
            self.assertFalse(server.start())
        self.assertEqual(self.bound, [9000])
        self.assertIsNone(server.port)
        self.assertFalse(os.path.exists(self.info_file))

    def test_start_when_every_port_is_busy(self):
        """No free port leaves the app running without control.

        Args: None.
        Returns: None.
        """
        server = control.ControlServer(control.ControlService(FakeBackend()),
                                       info_file=self.info_file)
        with mock.patch.object(server, "_bind", self.fake_bind(None)):
            self.assertFalse(server.start())
        self.assertFalse(os.path.exists(self.info_file))

    def test_bind_reports_busy_ports_as_none(self):
        """An OSError while binding becomes None.

        Args: None.
        Returns: None.
        """
        server = control.ControlServer(control.ControlService(FakeBackend()))
        with mock.patch.object(http.server, "ThreadingHTTPServer",
                               side_effect=OSError("busy")):
            self.assertIsNone(server._bind(8364))

    def test_stop_is_safe_and_removes_the_info_file(self):
        """stop() shuts down, closes and clears the file; twice is fine.

        Args: None.
        Returns: None.
        """
        httpd = FakeHTTPD(8364)
        server = control.ControlServer(control.ControlService(FakeBackend()),
                                       info_file=self.info_file)
        with mock.patch.object(server, "_bind", self.fake_bind(httpd)):
            server.start()
        control.write_info(self.info_file, 8364)
        server.stop()
        self.assertEqual(httpd.closed, 2)
        self.assertFalse(os.path.exists(self.info_file))
        server.stop()  # second stop and a stop before start are no-ops
        self.assertEqual(httpd.closed, 2)

    def test_stop_before_start(self):
        """Stopping a server that never started changes nothing.

        Args: None.
        Returns: None.
        """
        server = control.ControlServer(control.ControlService(FakeBackend()))
        server.stop()


class FileHelpersTests(unittest.TestCase):
    """The discovery file and log-tail helpers."""

    def setUp(self):
        """Create a temp folder.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_info_path_lands_in_the_output_folder(self):
        """The default discovery file is output/control.json.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(control.paths, "OUTPUT_DIR", "/somewhere"):
            self.assertEqual(control.info_path(),
                             os.path.join("/somewhere", "control.json"))

    def test_write_and_clear_info(self):
        """write_info records the port and pid; clear_info removes it.

        Args: None.
        Returns: None.
        """
        path = os.path.join(self.tmp.name, "control.json")
        control.write_info(path, 8999)
        with open(path, encoding="utf-8") as file:
            info = json.load(file)
        self.assertEqual(info, {"port": 8999, "pid": os.getpid()})
        control.clear_info(path)
        self.assertFalse(os.path.exists(path))

    def test_write_info_survives_an_unwritable_folder(self):
        """A failed write is ignored.

        Args: None.
        Returns: None.
        """
        blocker = os.path.join(self.tmp.name, "file")
        with open(blocker, "w", encoding="utf-8") as file:
            file.write("x")
        control.write_info(os.path.join(blocker, "control.json"), 8999)  # no raise

    def test_clear_info_ignores_a_missing_file(self):
        """Removing a file that is not there is fine.

        Args: None.
        Returns: None.
        """
        control.clear_info(os.path.join(self.tmp.name, "absent.json"))  # no raise

    def test_tail_survives_an_unreadable_file(self):
        """A file that exists but cannot be opened yields no lines.

        Args: None.
        Returns: None.
        """
        path = os.path.join(self.tmp.name, "locked.log")
        with open(path, "w", encoding="utf-8") as file:
            file.write("secret\n")
        os.chmod(path, 0)
        self.addCleanup(os.chmod, path, 0o644)
        self.assertEqual(control.tail(path, 5), [])

    def test_find_port_reads_the_file(self):
        """find_port returns the recorded port, or None when unusable.

        Args: None.
        Returns: None.
        """
        path = os.path.join(self.tmp.name, "control.json")
        self.assertIsNone(control.find_port(path))
        control.write_info(path, 8999)
        self.assertEqual(control.find_port(path), 8999)
        with open(path, "w", encoding="utf-8") as file:
            file.write("{not json")
        self.assertIsNone(control.find_port(path))
        with open(path, "w", encoding="utf-8") as file:
            json.dump({"port": "nope"}, file)
        self.assertIsNone(control.find_port(path))

    def test_tail_reads_the_last_lines(self):
        """tail returns the newest lines without newlines.

        Args: None.
        Returns: None.
        """
        path = os.path.join(self.tmp.name, "run.log")
        with open(path, "w", encoding="utf-8") as file:
            file.write("a\nb\nc\n")
        self.assertEqual(control.tail(path, 2), ["b", "c"])
        self.assertEqual(control.tail(path, 10), ["a", "b", "c"])
        self.assertEqual(control.tail(path, 0), [])
        self.assertEqual(control.tail(os.path.join(self.tmp.name, "absent"), 5), [])
        self.assertEqual(control.tail(None, 5), [])


class FakeResponse:
    """A urllib response stand-in with context-manager behavior."""

    def __init__(self, status, data):
        """Wrap a status and a body.

        Args: status is the HTTP code; data is the raw body bytes.
        Returns: None.
        """
        self.status = status
        self._data = data

    def read(self):
        """Return the body.

        Args: None.
        Returns: The stored bytes.
        """
        return self._data

    def __enter__(self):
        """Enter the with block.

        Args: None.
        Returns: self.
        """
        return self

    def __exit__(self, *exc):
        """Exit the with block.

        Args: exc is the exception information.
        Returns: False.
        """
        return False


class RequestTests(unittest.TestCase):
    """The client's HTTP request function, with a fake opener."""

    def setUp(self):
        """Replace the opener with a recording mock.

        Args: None.
        Returns: None.
        """
        self.opener = mock.MagicMock()
        patcher = mock.patch.object(control, "_OPENER", self.opener)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_get_returns_status_and_payload(self):
        """A GET returns the decoded payload and sends no body headers.

        Args: None.
        Returns: None.
        """
        self.opener.open.return_value = FakeResponse(200, b'{"app": "espi_live"}')
        code, payload = control.request(8364, "GET", "/status")
        self.assertEqual((code, payload), (200, {"app": "espi_live"}))
        request = self.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8364/status")
        self.assertIsNone(request.data)
        self.assertEqual(request.get_method(), "GET")

    def test_post_sends_json(self):
        """A POST sends the JSON body with the right method.

        Args: None.
        Returns: None.
        """
        self.opener.open.return_value = FakeResponse(200, b'{"ok": true}')
        code, payload = control.request(8364, "POST", "/settings",
                                        {"exposure_us": 5000})
        self.assertEqual((code, payload), (200, {"ok": True}))
        request = self.opener.open.call_args.args[0]
        self.assertEqual(json.loads(request.data.decode()), {"exposure_us": 5000})
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.headers.get("Content-type"),
                         "application/json")

    def test_http_error_carries_the_server_message(self):
        """An error response surfaces the server's error text.

        Args: None.
        Returns: None.
        """
        self.opener.open.side_effect = urllib.error.HTTPError(
            "http://127.0.0.1:8364/", 409, "Conflict", {},
            io.BytesIO(b'{"error": "camera 1 is off"}'))
        with self.assertRaises(control.ControlClientError) as raised:
            control.request(8364, "POST", "/settings", {"gain_db": 5})
        self.assertEqual(str(raised.exception), "camera 1 is off")

    def test_http_error_without_json_says_the_code(self):
        """An error response without JSON falls back to the HTTP code.

        Args: None.
        Returns: None.
        """
        self.opener.open.side_effect = urllib.error.HTTPError(
            "http://127.0.0.1:8364/", 500, "Oops", {}, io.BytesIO(b"oops"))
        with self.assertRaises(control.ControlClientError) as raised:
            control.request(8364, "GET", "/")
        self.assertEqual(str(raised.exception), "HTTP 500")

    def test_connection_failure_explains_itself(self):
        """An unreachable app produces a hint to start it.

        Args: None.
        Returns: None.
        """
        self.opener.open.side_effect = urllib.error.URLError("refused")
        with self.assertRaises(control.ControlClientError) as raised:
            control.request(8364, "GET", "/")
        self.assertIn("could not reach http://127.0.0.1:8364", str(raised.exception))
        self.assertIn("is ESPI live running", str(raised.exception))

    def test_non_json_success_is_refused(self):
        """A non-JSON response is reported instead of crashing.

        Args: None.
        Returns: None.
        """
        self.opener.open.return_value = FakeResponse(200, b"<html>")
        with self.assertRaises(control.ControlClientError) as raised:
            control.request(8364, "GET", "/")
        self.assertIn("not JSON", str(raised.exception))


class DiscoverTests(unittest.TestCase):
    """Finding the running app's port."""

    def setUp(self):
        """Prepare a temp folder and a stubbed request.

        Args: None.
        Returns: None.
        """
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.info_file = os.path.join(self.tmp.name, "control.json")

    def test_discover_uses_the_info_file_first(self):
        """The recorded port is probed before the default range.

        Args: None.
        Returns: None.
        """
        control.write_info(self.info_file, 8999)
        probed = []

        def fake_request(port, method, path, body=None, timeout=5.0):
            probed.append(port)
            if port == 8999:
                return 200, {"app": "espi_live"}
            raise control.ControlClientError("nope")

        with mock.patch.object(control, "request", side_effect=fake_request):
            self.assertEqual(control.discover(self.info_file), 8999)
        self.assertEqual(probed[0], 8999)

    def test_discover_scans_when_the_file_is_stale(self):
        """A dead recorded port falls through to the default range.

        Args: None.
        Returns: None.
        """
        control.write_info(self.info_file, 8999)

        def fake_request(port, method, path, body=None, timeout=5.0):
            if port == control.DEFAULT_PORT + 2:
                return 200, {"app": "espi_live"}
            raise control.ControlClientError("nope")

        with mock.patch.object(control, "request", side_effect=fake_request):
            self.assertEqual(control.discover(self.info_file),
                             control.DEFAULT_PORT + 2)

    def test_discover_fails_with_guidance(self):
        """Nothing answering raises with how to fix it.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(control, "request",
                               side_effect=control.ControlClientError("nope")):
            with self.assertRaises(control.ControlClientError) as raised:
                control.discover(self.info_file)
        self.assertIn("no running ESPI live app", str(raised.exception))
        self.assertIn("--port", str(raised.exception))

    def test_discover_ignores_a_foreign_service(self):
        """A port that answers with something else is skipped.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(control, "request",
                               return_value=(200, {"app": "other"})):
            with self.assertRaises(control.ControlClientError):
                control.discover(self.info_file)


def parse_client(argv):
    """Parse client arguments the way cli.py wires them.

    Args: argv is the argument list after "control".
    Returns: The parsed namespace.
    """
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="subcommand")
    control.add_parser(sub.add_parser("control"))
    return parser.parse_args(argv)


class ClientActionTests(unittest.TestCase):
    """The terminal actions, with the server answers faked."""

    def setUp(self):
        """Stub discovery and requests, and capture stdout.

        Args: None.
        Returns: None.
        """
        self.discover = mock.patch.object(control, "discover", return_value=8364)
        self.discover.start()
        self.addCleanup(self.discover.stop)
        self.request = mock.patch.object(control, "request")
        self.requested = self.request.start()
        self.addCleanup(self.request.stop)
        self.stdout = io.StringIO()
        patcher = mock.patch.object(control.sys, "stdout", self.stdout)
        patcher.start()
        self.addCleanup(patcher.stop)

    def output(self):
        """Return the captured stdout.

        Args: None.
        Returns: The text printed so far.
        """
        return self.stdout.getvalue()

    def test_status_plain_and_json(self):
        """status prints a compact list, or raw JSON with --json.

        Args: None.
        Returns: None.
        """
        self.requested.return_value = (200, {
            "app": "espi_live", "build": "build 7", "pid": 4242, "current": 0,
            "channels": [
                {"index": 0, "serial": "S0", "label": "CH0", "demo": True,
                 "active": True, "displayed": True,
                 "status": {"camera_fps": "Camera: 30.0 fps"}},
                {"index": 1, "serial": "S1", "label": "CH1", "demo": False,
                 "active": False, "displayed": False, "status": {}}]})
        self.assertEqual(control.run_cli(parse_client(["control", "status"])), 0)
        self.assertIn("build 7", self.output())
        self.assertIn("[0] S0  (displayed, demo)  Camera: 30.0 fps", self.output())
        self.assertIn("[1] S1  (off)", self.output())
        self.stdout.seek(0)
        self.stdout.truncate(0)
        self.assertEqual(
            control.run_cli(parse_client(["control", "status", "--json"])), 0)
        self.assertIn('"current"', self.output())

    def test_cameras_plain_and_json(self):
        """cameras lists each camera and its state.

        Args: None.
        Returns: None.
        """
        self.requested.return_value = (200, {"channels": [
            {"index": 0, "serial": "S0", "label": "CH0", "demo": True,
             "active": True, "displayed": True, "status": {}}]})
        self.assertEqual(control.run_cli(parse_client(["control", "cameras"])), 0)
        self.assertIn("[0] CH0  displayed  demo", self.output())
        self.stdout.seek(0)
        self.stdout.truncate(0)
        self.assertEqual(
            control.run_cli(parse_client(["control", "cameras", "--json"])), 0)
        self.assertIn('"serial"', self.output())

    def test_get_prints_values(self):
        """get prints every setting, or just the named ones.

        Args: None.
        Returns: None.
        """
        settings = {"mode": "Raw", "exposure_us": 8000.0, "average": 1}
        self.requested.return_value = (200, {"camera": 0, "settings": settings})
        self.assertEqual(
            control.run_cli(parse_client(["control", "get", "mode"])), 0)
        self.assertEqual(self.output(), "mode = Raw\n")
        self.stdout.seek(0)
        self.stdout.truncate(0)
        self.assertEqual(control.run_cli(parse_client(["control", "get"])), 0)
        self.assertIn("exposure_us = 8000.0", self.output())
        self.stdout.seek(0)
        self.stdout.truncate(0)
        self.assertEqual(
            control.run_cli(parse_client(["control", "get", "--camera", "S1"])), 0)
        self.assertEqual(self.requested.call_args.args[2], "/settings?camera=S1")

    def test_get_prints_raw_json_when_asked(self):
        """get --json prints the full response.

        Args: None.
        Returns: None.
        """
        self.requested.return_value = (200, {"camera": 0,
                                             "settings": {"mode": "Raw"}})
        self.assertEqual(
            control.run_cli(parse_client(["control", "get", "--json"])), 0)
        self.assertIn('"camera": 0', self.output())

    def test_get_with_unknown_key_fails(self):
        """An unknown setting name exits 1 with guidance.

        Args: None.
        Returns: None.
        """
        self.requested.return_value = (200, {"camera": 0,
                                             "settings": {"mode": "Raw"}})
        stderr = io.StringIO()
        with mock.patch.object(control.sys, "stderr", stderr):
            code = control.run_cli(parse_client(["control", "get", "saturation"]))
        self.assertEqual(code, 1)
        self.assertIn("unknown settings: saturation", stderr.getvalue())

    def test_set_sends_pairs_and_prints_results(self):
        """set sends KEY=VALUE pairs and prints the applied values.

        Args: None.
        Returns: None.
        """
        self.requested.return_value = (200, {"camera": 0,
                                             "applied": {"exposure_us": 5000.0},
                                             "settings": {}})
        code = control.run_cli(
            parse_client(["control", "set", "exposure_us=5000", "--camera", "1"]))
        self.assertEqual(code, 0)
        body = self.requested.call_args.args[3]
        self.assertEqual(body, {"exposure_us": "5000", "camera": "1"})
        self.assertEqual(self.requested.call_args.args[1], "POST")
        self.assertIn("exposure_us = 5000.0", self.output())
        self.stdout.seek(0)
        self.stdout.truncate(0)
        self.assertEqual(
            control.run_cli(parse_client(["control", "set", "mode=Raw",
                                          "--json"])), 0)
        self.assertIn('"applied"', self.output())

    def test_set_rejects_bare_words(self):
        """A value without = is an argument error (exit 2).

        Args: None.
        Returns: None.
        """
        with self.assertRaises(SystemExit) as raised:
            parse_client(["control", "set", "exposure_us"])
        self.assertEqual(raised.exception.code, 2)

    def test_run_sends_the_command(self):
        """run posts the command name and payload.

        Args: None.
        Returns: None.
        """
        self.requested.return_value = (200, {"camera": 0, "command": "save",
                                             "result": {"requested": "save"}})
        code = control.run_cli(
            parse_client(["control", "run", "save", "title=Band", "--camera", "0"]))
        self.assertEqual(code, 0)
        self.assertEqual(self.requested.call_args.args[3],
                         {"command": "save", "title": "Band", "camera": "0"})
        self.assertIn("requested: save", self.output())

    def test_run_prints_a_confirmation_without_a_result(self):
        """A command with an empty result confirms what ran.

        Args: None.
        Returns: None.
        """
        self.requested.return_value = (200, {"camera": 1, "command": "reference",
                                             "result": {}})
        self.assertEqual(
            control.run_cli(parse_client(["control", "run", "reference"])), 0)
        self.assertIn("ran reference on camera 1", self.output())

    def test_run_prints_raw_json_when_asked(self):
        """run --json prints the full response.

        Args: None.
        Returns: None.
        """
        self.requested.return_value = (200, {"camera": 0, "command": "save",
                                             "result": {"requested": "save"}})
        self.assertEqual(
            control.run_cli(parse_client(["control", "run", "save", "--json"])), 0)
        self.assertIn('"requested"', self.output())

    def test_run_rejects_unknown_commands(self):
        """An unknown command name is an argument error (exit 2).

        Args: None.
        Returns: None.
        """
        with self.assertRaises(SystemExit) as raised:
            parse_client(["control", "run", "explode"])
        self.assertEqual(raised.exception.code, 2)

    def test_log_prints_lines(self):
        """log prints the returned lines.

        Args: None.
        Returns: None.
        """
        self.requested.return_value = (200, {"lines": ["12:00:00  Streaming",
                                                       "12:00:01  Control: GET"]})
        self.assertEqual(control.run_cli(parse_client(["control", "log"])), 0)
        self.assertEqual(self.requested.call_args.args[2], "/log?lines=200")
        self.assertIn("12:00:01  Control: GET", self.output())

    def test_explicit_port_skips_discovery(self):
        """--port is used directly.

        Args: None.
        Returns: None.
        """
        self.requested.return_value = (200, {"channels": [], "build": "b",
                                             "pid": 1})
        with mock.patch.object(control, "discover",
                               side_effect=AssertionError("should not run")):
            code = control.run_cli(
                parse_client(["control", "status", "--port", "9000"]))
        self.assertEqual(code, 0)
        self.assertEqual(self.requested.call_args.args[0], 9000)

    def test_discovery_failure_reports_and_exits_1(self):
        """When no app answers, the client prints the reason on stderr.

        Args: None.
        Returns: None.
        """
        self.discover.stop()
        stderr = io.StringIO()
        with mock.patch.object(control, "discover", autospec=True) as discover:
            discover.side_effect = control.ControlClientError("no running app")
            with mock.patch.object(control.sys, "stderr", stderr):
                code = control.run_cli(parse_client(["control", "status"]))
        self.assertEqual(code, 1)
        self.assertIn("error: no running app", stderr.getvalue())

    def test_request_failure_reports_and_exits_1(self):
        """A server refusal exits 1 with the message on stderr.

        Args: None.
        Returns: None.
        """
        self.requested.side_effect = control.ControlClientError("camera 1 is off")
        stderr = io.StringIO()
        with mock.patch.object(control.sys, "stderr", stderr):
            code = control.run_cli(
                parse_client(["control", "set", "gain_db=5", "--camera", "1"]))
        self.assertEqual(code, 1)
        self.assertIn("error: camera 1 is off", stderr.getvalue())


class CameraQueryTests(unittest.TestCase):
    """The camera query-string helper."""

    def test_none_gives_no_query(self):
        """No camera spec means no query string.

        Args: None.
        Returns: None.
        """
        self.assertEqual(control._camera_query(None), "")

    def test_specs_are_quoted(self):
        """Indexes and labels are quoted into the query.

        Args: None.
        Returns: None.
        """
        self.assertEqual(control._camera_query(0), "?camera=0")
        self.assertEqual(control._camera_query("Blackfly S (123)"),
                         "?camera=Blackfly%20S%20%28123%29")


class LoopbackTests(unittest.TestCase):
    """One real round trip on 127.0.0.1 — the server, handler and client."""

    def test_full_round_trip_on_an_os_chosen_port(self):
        """A live server answers GET and POST on the loopback interface.

        Args: None.
        Returns: None.
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        backend = FakeBackend()
        server = control.ControlServer(control.ControlService(backend), port=0,
                                       info_file=os.path.join(tmp.name,
                                                              "control.json"))
        self.assertTrue(server.start())
        self.addCleanup(server.stop)
        self.assertNotEqual(server.port, 0)
        code, payload = control.request(server.port, "GET", "/status", timeout=5.0)
        self.assertEqual(code, 200)
        self.assertEqual(payload["app"], "espi_live")
        self.assertEqual(len(payload["channels"]), 2)
        code, payload = control.request(server.port, "POST", "/settings",
                                        {"exposure_us": "5000"}, timeout=5.0)
        self.assertEqual(code, 200)
        self.assertEqual(payload["settings"]["exposure_us"], 5000.0)
        self.assertEqual(backend.applied, [(0, {"exposure_us": 5000.0})])
        code, payload = control.request(server.port, "POST", "/command",
                                        {"command": "reference"}, timeout=5.0)
        self.assertEqual(code, 200)
        self.assertEqual(backend.commands, [(0, "reference", {})])

    def test_info_file_points_clients_at_the_live_port(self):
        """A second process finds the server through the discovery file.

        Args: None.
        Returns: None.
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        info_file = os.path.join(tmp.name, "control.json")
        server = control.ControlServer(control.ControlService(FakeBackend()),
                                       port=0, info_file=info_file)
        self.assertTrue(server.start())
        self.addCleanup(server.stop)
        self.assertEqual(control.find_port(info_file), server.port)
        self.assertEqual(control.discover(info_file), server.port)

    def test_a_busy_port_is_skipped(self):
        """A second server moves to the next port when the first is taken.

        Args: None.
        Returns: None.
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        first = control.ControlServer(control.ControlService(FakeBackend()),
                                      port=0,
                                      info_file=os.path.join(tmp.name, "a.json"))
        self.assertTrue(first.start())
        self.addCleanup(first.stop)
        second = control.ControlServer(
            control.ControlService(FakeBackend()),
            port=first.port,
            info_file=os.path.join(tmp.name, "b.json"))
        self.assertFalse(second.start())  # a fixed busy port is refused
        third = control.ControlServer(control.ControlService(FakeBackend()),
                                      info_file=os.path.join(tmp.name, "c.json"))
        self.assertTrue(third.start())
        self.addCleanup(third.stop)
        self.assertNotIn(third.port, (first.port, second.requested_port))


class JobTests(unittest.TestCase):
    """The GUI-thread job envelope."""

    def test_job_starts_pending(self):
        """A fresh job is not done and holds no result.

        Args: None.
        Returns: None.
        """
        job = control.Job(lambda: 3)
        self.assertFalse(job.done.is_set())
        self.assertIsNone(job.result)
        self.assertIsNone(job.error)
        self.assertEqual(job.fn(), 3)
