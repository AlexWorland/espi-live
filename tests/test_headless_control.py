"""Checks for espi_live.headless_control: driving the NI DAQ and cameras from JSON.

Run from the project root:  PYTHONPATH=src:tests python -m unittest tests.test_headless_control

No NI hardware, camera, display or real output folder is used: cameras are demo
sources, the simulated DAQ device drives nothing, and pyLabLib's real NIDAQ runs
on the in-memory nidaqmx (fake_nidaqmx) for the checklist and level checks.
"""

import contextlib
import io
import json
import queue
import sys
import tempfile
import threading
import time
import types
import unittest
from datetime import datetime
from unittest import mock

import fake_nidaqmx
from espi_live import channels, daq, headless, headless_control, paths
from espi_live.headless_control import CommandError, ControlSession
from list_log import ListLog
from test_engine_camera import wait_until


class BlockingInput:
    """A stand-in for stdin that yields nothing until released, then ends."""

    def __init__(self):
        """Create an input that is blocked.

        Args: None.
        Returns: None.
        """
        self.release = threading.Event()

    def __iter__(self):
        """Iterate over the lines (there are none).

        Args: None.
        Returns: The input itself.
        """
        return self

    def __next__(self):
        """Block until released, then end the input.

        Args: None.
        Returns: Never returns a line; raises StopIteration.
        """
        self.release.wait(5)
        raise StopIteration


class SessionCase(unittest.TestCase):
    """Base class: one running demo camera, a DAQ controller and a session."""

    def setUp(self):
        """Redirect outputs, start a demo camera and make the session.

        Args: None.
        Returns: None.
        """
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.object(paths, "OUTPUT_DIR", tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.log = ListLog()
        self.channels = channels.demo_channels(self.log, 2)
        self.channels[0].engine.start()
        self.addCleanup(self.stop_cameras)
        self.events = []
        self.controller = daq.DaqController(self.log)
        self.session = ControlSession(self.channels, self.log, self.controller,
                                      allow_arm=True, emit=self.events.append)
        # As in a control run: the session hears the engines' events too.
        self.failures = []
        self.session.pump = lambda: headless.pump_events(
            self.channels, self.log, self.failures, self.session.note_event)
        self.addCleanup(self.session.close)

    def stop_cameras(self):
        """Stop every engine the test started.

        Args: None.
        Returns: None.
        """
        for channel in self.channels:
            channel.stop()
        for channel in self.channels:
            channel.wait(5)

    def run_command(self, name, **arguments):
        """Run one command and return its reply.

        Args: name is the command; arguments are its fields.
        Returns: The reply dict.
        """
        return self.session.execute({"cmd": name, **arguments})

    def good(self, name, **arguments):
        """Run a command that must succeed.

        Args: name is the command; arguments are its fields.
        Returns: The reply dict.
        """
        reply = self.run_command(name, **arguments)
        self.assertTrue(reply["ok"], reply)
        return reply

    def bad(self, name, text, **arguments):
        """Run a command that must fail with a message containing text.

        Args: name is the command; text is what the error must contain;
            arguments are its fields.
        Returns: The reply dict.
        """
        reply = self.run_command(name, **arguments)
        self.assertFalse(reply["ok"], reply)
        self.assertIn(text, reply["error"])
        return reply

    def connect_simulated(self):
        """Connect the simulated NI device.

        Args: None.
        Returns: The reply dict.
        """
        return self.good("daq_connect", device=daq.SIMULATED)

    def wait_daq(self, predicate, timeout=5):
        """Poll a DAQ condition while keeping the DAQ's watchdog fed.

        The window feeds the watchdog from its timer; a test does it here
        with session.tick() between the polls, or the self-check stops the
        outputs after WATCHDOG_S of silence and the poll sees 0 V.

        Args: predicate is a zero-argument callable; timeout is in seconds.
        Returns: True when predicate() holds, False at the timeout.
        """
        def polled():
            """One poll after the housekeeping."""
            self.session.tick()
            return predicate()
        return wait_until(polled, timeout)

    @staticmethod
    def triangle(**extra):
        """The DAQ settings of a frame-rate-following piezo triangle.

        Args: extra are DaqSettings fields to change.
        Returns: A dict for daq_apply's "settings".
        """
        settings = {"piezo_mode": daq.TRIANGLE, "piezo_on": True, "piezo_output": "ao1",
                    "piezo_bottom_v": 1.0, "piezo_top_v": 10.0, "piezo_limit_v": 10.0,
                    "piezo_scale": daq.MANUAL, "piezo_um_per_v": 0.1,
                    "piezo_gain": 15.0, "piezo_angle_deg": 45.0,
                    "piezo_frequency_hz": 0.25, "piezo_follow_fps": True}
        settings.update(extra)
        return settings


class ProtocolTests(SessionCase):
    """Lines in, replies out."""

    def test_blank_line_gets_no_reply(self):
        """A blank line is skipped.

        Args: None.
        Returns: None.
        """
        self.assertIsNone(self.session.handle_line("  \n"))

    def test_byte_order_mark_is_ignored(self):
        """PowerShell 5.1 starts piped text with U+FEFF: the first command still runs.

        Args: None.
        Returns: None.
        """
        reply = self.session.handle_line('\ufeff{"cmd": "status"}\n')
        self.assertTrue(reply["ok"], reply)
        self.assertIn("cameras", reply)
        self.assertTrue(self.session.handle_line('{"cmd": "status"}')["ok"])  # none: as before
        self.assertIn("not valid JSON", self.session.handle_line("\ufeff{nope")["error"])

    def test_line_with_only_a_byte_order_mark_is_blank(self):
        """A BOM, with or without whitespace, is a blank line: no reply.

        Args: None.
        Returns: None.
        """
        for line in ("\ufeff", "\ufeff\n", "\ufeff  \r\n"):
            self.assertIsNone(self.session.handle_line(line), repr(line))

    def test_bad_json_and_bad_shapes_are_errors_not_crashes(self):
        """Malformed input is answered and the session carries on.

        Args: None.
        Returns: None.
        """
        self.assertFalse(self.session.handle_line("{nope")["ok"])
        self.assertIn("not valid JSON", self.session.handle_line("{nope")["error"])
        self.assertIn('"cmd"', self.session.handle_line("[1]")["error"])
        self.assertIn('"cmd"', self.session.handle_line('{"cmd": 3}')["error"])
        self.assertIn("unknown command", self.session.handle_line('{"cmd": "nope"}')["error"])
        self.assertIn("unknown command", self.session.handle_line('{"cmd": "a b"}')["error"])
        self.assertTrue(self.session.handle_line('{"cmd": "status"}')["ok"])

    def test_the_id_is_echoed_on_success_and_failure(self):
        """A caller can match replies to commands.

        Args: None.
        Returns: None.
        """
        self.assertEqual(self.session.execute({"cmd": "status", "id": 7})["id"], 7)
        self.assertEqual(self.session.execute({"cmd": "nope", "id": "x"})["id"], "x")

    def test_a_bug_in_a_command_is_reported_and_logged(self):
        """An unexpected exception becomes an error reply.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(self.session, "cmd_status", side_effect=KeyError("boom")):
            reply = self.session.execute({"cmd": "status"})
        self.assertFalse(reply["ok"])
        self.assertIn("KeyError", reply["error"])
        self.assertTrue(self.log.has("Control command failed"))

    def test_problems_come_back_as_a_list(self):
        """A DAQ rejection carries its reasons.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        reply = self.bad("daq_apply", "rejected", settings={"piezo_on": True,
                                                            "piezo_voltage_v": 5.0})
        self.assertTrue(reply["problems"])

    def test_line_writer_writes_json_lines(self):
        """Replies are one JSON object per line, flushed.

        Args: None.
        Returns: None.
        """
        stream = io.StringIO()
        headless_control.line_writer(stream)({"ok": True, "when": datetime(2026, 11, 22)})
        self.assertEqual(json.loads(stream.getvalue()),
                         {"ok": True, "when": "2026-11-22 00:00:00"})
        self.assertTrue(stream.getvalue().endswith("\n"))

    def test_divert_console_keeps_stdout_for_replies(self):
        """Log lines and printed text move to stderr; replies get the real stdout.

        Args: None.
        Returns: None.
        """
        out, err = io.StringIO(), io.StringIO()
        log = types.SimpleNamespace(original=(out, err),
                                    handler=types.SimpleNamespace(console=out))
        with mock.patch.object(sys, "stdout"), mock.patch.object(sys, "stderr", err):
            self.assertIs(headless_control.divert_console(log), out)
            self.assertIs(sys.stdout, err)
        self.assertIs(log.handler.console, err)


class StatusTests(SessionCase):
    """What status reports."""

    def test_status_describes_cameras_and_daq(self):
        """Cameras, the DAQ and the session's own state are in one reply.

        Args: None.
        Returns: None.
        """
        reply = self.good("status")
        self.assertEqual(len(reply["cameras"]), 2)
        self.assertTrue(reply["cameras"][0]["running"])
        self.assertFalse(reply["cameras"][1]["running"])
        self.assertEqual(reply["cameras"][1]["settings"], {})
        self.assertIn("mode", reply["cameras"][0]["settings"])
        self.assertIsNone(reply["daq"]["device"])
        self.assertTrue(reply["daq"]["allow_arm"])
        self.assertFalse(reply["stepper"])
        self.connect_simulated()
        self.good("daq_apply", settings={"piezo_on": True, "piezo_voltage_v": 0.2})
        state = self.good("status")["daq"]
        self.assertTrue(state["device"]["simulated"])
        self.assertTrue(state["active"])
        self.assertEqual(state["settings"]["piezo_voltage_v"], 0.2)
        self.assertEqual(state["pending_settings"]["piezo_voltage_v"], 0.2)


class ScanAndConnectTests(SessionCase):
    """Finding and connecting to devices."""

    def test_scan_always_lists_the_simulated_device(self):
        """Without the NI driver only the simulated device is offered.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(daq, "list_devices", return_value=([], "no driver")):
            reply = self.good("daq_scan")
        self.assertEqual([d["name"] for d in reply["devices"]], [daq.SIMULATED])
        self.assertEqual(reply["reason"], "no driver")

    def test_scan_lists_a_real_device(self):
        """A device the driver finds comes first.

        Args: None.
        Returns: None.
        """
        fake_nidaqmx.install(self)
        names = [d["name"] for d in self.good("daq_scan")["devices"]]
        self.assertEqual(names, ["Dev1", daq.SIMULATED])

    def test_scan_marks_an_ni_max_simulated_device(self):
        """A device NI MAX simulates says so in its label and in ni_simulated.

        Args: None.
        Returns: None.
        """
        fake_nidaqmx.install(self, devices=("Dev1", "SimDev1"), simulated=("SimDev1",))
        real, ni_max, stand_in = self.good("daq_scan")["devices"]
        self.assertEqual((real["simulated"], real["ni_simulated"]), (False, False))
        self.assertEqual((ni_max["simulated"], ni_max["ni_simulated"]), (False, True))
        self.assertIn("NI MAX simulation, no hardware", ni_max["label"])
        self.assertNotIn("NI MAX", real["label"])
        self.assertEqual((stand_in["simulated"], stand_in["ni_simulated"]), (True, False))

    def test_connect_needs_a_device_name(self):
        """A missing or empty name is refused.

        Args: None.
        Returns: None.
        """
        self.bad("daq_connect", "device name")
        self.bad("daq_connect", "device name", device="")

    def test_connect_simulated_reports_the_outputs(self):
        """Connecting the stand-in lists its outputs and starts everything off.

        Args: None.
        Returns: None.
        """
        reply = self.connect_simulated()
        self.assertTrue(reply["device"]["simulated"])
        self.assertEqual(reply["analog"], ["ao0", "ao1"])
        self.assertIn("ctr0", reply["counters"])
        self.assertFalse(self.session.settings.piezo_on)
        self.assertEqual(self.session.settings.device, daq.SIMULATED)

    def test_connect_failure_is_an_error_and_is_remembered(self):
        """A device that cannot be opened is reported, not swallowed.

        Args: None.
        Returns: None.
        """
        fake_nidaqmx.install(self)
        self.bad("daq_connect", "DAQ failed", device="Dev9")
        self.assertTrue(self.session.faults[0].startswith("failed:"))

    def test_commands_need_a_connected_device(self):
        """Every DAQ command says to connect first.

        Args: None.
        Returns: None.
        """
        for name, arguments in (("daq_apply", {}), ("daq_zero", {}), ("daq_self_test", {}),
                                ("daq_arm", {}), ("piezo_step", {"volts": 0.1}),
                                ("phase_attach", {})):
            with self.subTest(name):
                self.bad(name, "connect to a NI device first", **arguments)

    def test_a_timeout_is_an_error(self):
        """A DAQ that never answers does not hang the session.

        Args: None.
        Returns: None.
        """
        silent = types.SimpleNamespace(events=queue.SimpleQueue(), heartbeat=lambda: None,
                                       request=lambda *a, **k: None)
        session = ControlSession([], self.log, silent)
        with mock.patch.object(headless_control, "DAQ_TIMEOUT_S", 0.2), \
                mock.patch.object(headless_control, "POLL_S", 0.01):
            reply = session.execute({"cmd": "daq_scan"})
        self.assertFalse(reply["ok"])
        self.assertIn("did not answer", reply["error"])


class ApplyTests(SessionCase):
    """Writing DAQ settings."""

    def test_partial_changes_build_on_the_last_ones(self):
        """Fields left out keep what was applied before.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        first = self.good("daq_apply", settings={"generator_on": True,
                                                 "generator_frequency_hz": 500.0})
        self.assertTrue(first["active"])
        self.assertTrue(first["simulated"])
        self.good("daq_apply", settings={"piezo_on": True, "piezo_voltage_v": 0.1})
        written = self.controller.status()["settings"]
        self.assertTrue(written.generator_on)
        self.assertEqual(written.generator_frequency_hz, 500.0)
        self.assertEqual(written.device, daq.SIMULATED)

    def test_unknown_and_mistyped_fields_are_refused(self):
        """A typo must not silently leave a default in place.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.bad("daq_apply", "unknown DAQ settings: piezo_volts", settings={"piezo_volts": 1})
        self.bad("daq_apply", "wrong type", settings={"piezo_voltage_v": "high"})
        self.bad("daq_apply", "wrong type", settings={"piezo_on": 1})
        self.bad("daq_apply", "wrong type", settings={"piezo_limit_v": True})
        self.bad("daq_apply", "wrong type", settings={"generator_frequency_hz": float("nan")})
        self.bad("daq_apply", "must be an object", settings=[1])
        self.assertFalse(self.controller.status()["active"])

    def test_a_rule_the_daq_enforces_is_a_rejection(self):
        """The safety limit and the 0 V floor come back as problems.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        too_high = self.bad("daq_apply", "rejected",
                            settings={"piezo_on": True, "piezo_voltage_v": 5.0})
        self.assertIn("limit", " ".join(too_high["problems"]))
        self.bad("daq_apply", "rejected", settings={"piezo_on": True, "piezo_voltage_v": -0.5})
        self.assertEqual(self.session.settings.piezo_voltage_v, 0.0)

    def test_zero_switches_everything_off(self):
        """After a zero, the next partial apply does not bring the drive back.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings={"generator_on": True, "piezo_on": True,
                                         "piezo_voltage_v": 0.3})
        self.good("daq_zero")
        self.assertFalse(self.controller.status()["active"])
        self.assertFalse(self.session.settings.piezo_on)
        reply = self.good("daq_apply", settings={"piezo_voltage_v": 0.1})
        self.assertFalse(reply["active"])

    def test_self_test_reports_the_result(self):
        """The simulated device passes; the result lines come back.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings={"piezo_on": True, "piezo_voltage_v": 0.3})
        reply = self.good("daq_self_test")
        self.assertTrue(reply["passed"])
        self.assertTrue(reply["lines"])
        self.assertFalse(self.session.settings.piezo_on)

    def test_a_demo_camera_follows_the_applied_drive(self):
        """The window's forwarding of the applied drive to demo cameras is kept.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        reply = self.good("daq_apply", settings={"generator_on": True,
                                                 "generator_frequency_hz": 628.0,
                                                 "generator_output": "ao0"})
        self.assertEqual(reply["drive"]["waveform"], "sine")
        self.assertEqual(self.channels[0].drive.frequency_hz, reply["drive"]["frequency_hz"])
        self.good("daq_zero")
        self.assertFalse(self.channels[0].drive.active)


class ChecklistTests(SessionCase):
    """The safety checklist gate on a real (fake-nidaqmx) device."""

    def setUp(self):
        """Put the fake NI hardware behind the controller and connect Dev1.

        Args: None.
        Returns: None.
        """
        super().setUp()
        self.hardware = fake_nidaqmx.install(self)
        self.good("daq_connect", device="Dev1")

    def test_arming_needs_the_command_line_flag(self):
        """Without --daq-checklist-done the session cannot attest the checklist.

        Args: None.
        Returns: None.
        """
        self.session.allow_arm = False
        self.bad("daq_arm", "--daq-checklist-done")
        self.bad("daq_apply", "rejected", settings={"piezo_on": True, "piezo_voltage_v": 0.2})
        self.assertEqual(self.hardware.ao, {"ao0": 0.0, "ao1": 0.0})  # still at 0 V

    def test_a_real_device_takes_no_drive_before_arming(self):
        """The controller's own gate refuses, and nothing reaches the output.

        Args: None.
        Returns: None.
        """
        reply = self.bad("daq_apply", "rejected",
                         settings={"piezo_on": True, "piezo_voltage_v": 0.2})
        self.assertIn("checklist", " ".join(reply["problems"]))
        self.bad("piezo_step", "checklist", volts=0.2)
        self.assertEqual(float(self.hardware.ao["ao1"]), 0.0)

    def test_arming_covers_only_the_roles_named(self):
        """Arming the piezo does not arm the generator.

        Args: None.
        Returns: None.
        """
        reply = self.good("daq_arm", roles=["piezo"])
        self.assertEqual(reply["covered"], ["Piezo driver/ao1"])
        self.assertFalse(reply["simulated"])
        self.good("daq_apply", settings={"piezo_on": True, "piezo_voltage_v": 0.25})
        self.assertEqual(float(self.hardware.ao["ao1"]), 0.25)
        self.bad("daq_apply", "rejected", settings={"generator_on": True})

    def test_arming_everything_by_default(self):
        """Without roles, every assigned role is covered.

        Args: None.
        Returns: None.
        """
        reply = self.good("daq_arm")
        self.assertEqual(reply["covered"], ["Frequency generator/ao0", "Piezo driver/ao1"])
        self.good("daq_apply", settings={"generator_on": True, "piezo_on": True})

    def test_roles_must_be_known_names(self):
        """Typos in roles are refused rather than arming nothing.

        Args: None.
        Returns: None.
        """
        self.bad("daq_arm", "roles", roles=["piezo", "shaker"])
        self.bad("daq_arm", "roles", roles=[])
        self.bad("daq_arm", "roles", roles="piezo")

    def test_an_unassigned_role_is_not_armed(self):
        """A role with no output has nothing to cover.

        Args: None.
        Returns: None.
        """
        self.good("daq_apply", settings={"piezo_output": daq.OFF})
        reply = self.good("daq_arm")
        self.assertEqual(reply["covered"], ["Frequency generator/ao0"])

    def test_a_new_connection_needs_the_checklist_again(self):
        """Reconnecting forgets the attestation, as in the window.

        Args: None.
        Returns: None.
        """
        self.good("daq_arm")
        self.good("daq_connect", device="Dev1")
        self.bad("daq_apply", "rejected", settings={"piezo_on": True})

    def test_piezo_steps_reach_the_output(self):
        """piezo_step holds the piezo at each level once armed.

        Args: None.
        Returns: None.
        """
        self.good("daq_arm", roles=["piezo"])
        self.good("daq_apply", settings={"piezo_limit_v": 1.0})
        for volts in (0.1, 0.35):
            reply = self.good("piezo_step", volts=volts)
            self.assertEqual(reply["volts"], volts)
            self.assertEqual(float(self.hardware.ao["ao1"]), volts)

    def test_piezo_step_refuses_bad_levels(self):
        """Non-numbers and levels over the safety limit are errors.

        Args: None.
        Returns: None.
        """
        self.good("daq_arm", roles=["piezo"])
        self.bad("piezo_step", "finite number")
        self.bad("piezo_step", "finite number", volts="0.1")
        self.bad("piezo_step", "finite number", volts=True)
        self.bad("piezo_step", "finite number", volts=float("inf"))
        self.bad("piezo_step", "limit", volts=3.0)

    def test_self_test_on_the_real_backend(self):
        """The readback self-test runs against the fake device.

        Args: None.
        Returns: None.
        """
        self.assertTrue(self.good("daq_self_test")["passed"])


class PhaseTests(SessionCase):
    """Handing the piezo to a camera's Phase view."""

    def attached(self):
        """Tell which engines hold a stepper.

        Args: None.
        Returns: The list of channels whose engine has one.
        """
        return [c for c in self.channels
                if c.engine is not None and c.engine._stepper is not None]

    def test_attach_gives_the_piezo_to_one_camera_in_phase_view(self):
        """The camera switches to the Phase view and only it steps.

        Args: None.
        Returns: None.
        """
        self.channels[1].start()
        self.connect_simulated()
        reply = self.good("phase_attach", camera=1)
        self.assertEqual(reply["camera"], self.channels[1].label)
        self.assertEqual(self.attached(), [self.channels[1]])
        self.assertEqual(self.channels[1].engine.settings().mode, "Phase")
        self.assertTrue(self.good("status")["stepper"])
        self.good("phase_attach")
        self.assertEqual(self.attached(), [self.channels[0]])

    def test_the_demo_camera_phase_view_follows_piezo_steps(self):
        """With the simulated piezo, steps move the demo interferometer.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings={"piezo_step_v": 0.05})
        self.good("phase_attach")
        self.assertTrue(wait_until(
            lambda: self.channels[0].engine.status().get("processed_fps") not in (None, "?"),
            timeout=10))
        self.assertTrue(wait_until(lambda: self.controller.status()["settings"] is not None
                                   and self.controller.status()["settings"].piezo_on,
                                   timeout=10))

    def test_attach_needs_a_piezo_and_the_checklist(self):
        """No piezo output, or an unarmed real device, is refused.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings={"piezo_output": daq.OFF})
        self.bad("phase_attach", "assign the piezo")
        fake_nidaqmx.install(self)
        self.good("daq_connect", device="Dev1")
        self.good("daq_apply", settings={"piezo_output": "ao1"})
        self.bad("phase_attach", "checklist")
        self.assertEqual(self.attached(), [])

    def test_applying_new_settings_keeps_stepping_with_them(self):
        """A later apply refreshes the stepper, and removing the piezo detaches it.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("phase_attach")
        first = self.channels[0].engine._stepper
        self.good("daq_apply", settings={"piezo_step_v": 0.07})
        second = self.channels[0].engine._stepper
        self.assertIsNot(first, second)
        self.assertEqual(second.step_v, 0.07)
        self.good("daq_apply", settings={"piezo_output": daq.OFF})
        self.assertIsNone(self.channels[0].engine._stepper)
        self.assertFalse(self.session.stepper)

    def test_zero_connect_and_fault_take_the_piezo_away(self):
        """Anything that sets the outputs to 0 V stops the stepping.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        for how in ("zero", "connect"):
            self.good("phase_attach")
            self.assertTrue(self.attached())
            if how == "zero":
                self.good("daq_zero")
            else:
                self.connect_simulated()
            self.assertEqual(self.attached(), [])

    def test_a_fault_is_an_event_and_stops_the_stepping(self):
        """A DAQ self-check fault arrives unasked and detaches the piezo.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("phase_attach")
        self.controller.events.put(("fault", "AO 1 read 0.9 V"))
        self.session.tick()
        self.assertEqual(self.events, [{"event": "fault", "reason": "AO 1 read 0.9 V"}])
        self.assertEqual(self.attached(), [])
        self.assertEqual(self.good("status")["faults"], ["fault: AO 1 read 0.9 V"])

    def test_faults_are_remembered_up_to_a_limit(self):
        """The list of faults stays short.

        Args: None.
        Returns: None.
        """
        for number in range(headless_control.MAX_FAULTS + 5):
            self.controller.events.put(("failed", str(number)))
        self.session.tick()
        self.assertEqual(len(self.session.faults), headless_control.MAX_FAULTS)
        self.assertEqual(self.session.faults[-1], f"failed: {headless_control.MAX_FAULTS + 4}")

    def test_detach(self):
        """phase_detach removes the stepper and says whether there was one.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("phase_attach")
        self.assertTrue(self.good("phase_detach")["detached"])
        self.assertFalse(self.good("phase_detach")["detached"])
        self.assertEqual(self.attached(), [])


class StubTuner:
    """A stand-in for step_tuning.StepTuner that reports what a test sets."""

    def __init__(self, state="measuring", scale=1.0, wants_hold=False):
        """Make a tuner in a state.

        Args: state is its state name; scale is its factor on the frequency;
            wants_hold is True while it measures the noise.
        Returns: None.
        """
        self.state = state
        self.scale = scale
        self.wants_hold = wants_hold
        self.noise = 0.0
        self.changes = []

    def update(self, now, monitor):
        """Take a measurement (the stub reports a state set on it).

        Args: now is the time; monitor is the engine's StepMonitor.
        Returns: None.
        """
        if getattr(self, "next_state", None) is not None:
            self.state, self.next_state = self.next_state, None
        return None

    def changed(self, now):
        """Note that the frequency was applied.

        Args: now is the time.
        Returns: None.
        """
        self.changes.append(now)

    def describe(self):
        """Say what the tuner is doing.

        Args: None.
        Returns: One sentence.
        """
        return f"Auto-tune: stub {self.state} ×{self.scale:.3f}."


class FollowTests(SessionCase):
    """Setting a frame-rate-following triangle's frequency from the camera."""

    def applied(self):
        """The settings the DAQ has written.

        Args: None.
        Returns: daq.DaqSettings, or None before the first apply.
        """
        return self.controller.status()["settings"]

    def test_follow_sets_the_frequency_the_formula_gives(self):
        """The reply and the DAQ carry the frequency for the camera's rate.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings=self.triangle(piezo_frequency_hz=0.1))
        self.session._received_fps = lambda channel: 30.0
        reply = self.good("daq_follow")
        expected, _ = daq.follow_frequency(self.session.settings, 30.0)
        self.assertTrue(reply["changed"])
        self.assertEqual(reply["fps"], 30.0)
        self.assertAlmostEqual(reply["frequency_hz"], expected, places=9)
        self.assertAlmostEqual(expected, 0.2486, places=4)
        self.assertAlmostEqual(self.applied().piezo_frequency_hz, expected, places=9)

    def test_a_settled_triangle_is_left_alone(self):
        """A rate that asks for the same frequency again sends nothing.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings=self.triangle())
        self.session._received_fps = lambda channel: 30.0
        self.good("daq_follow")
        again = self.good("daq_follow")
        self.assertFalse(again["changed"])
        self.assertIn("Follows the camera", again["text"])

    def test_follow_needs_an_applied_triangle_that_follows(self):
        """Nothing applied, a hold, or Follow off are refused with the reason.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.bad("daq_follow", "apply the piezo settings first")
        for settings in (self.triangle(piezo_follow_fps=False),
                         self.triangle(piezo_mode=daq.HOLD),
                         self.triangle(piezo_output=daq.OFF, piezo_on=False)):
            self.good("daq_apply", settings=settings)
            self.bad("daq_follow", "does not follow the camera")

    def test_follow_waits_for_frames(self):
        """Before any frame arrives there is no rate to follow.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings=self.triangle())
        self.session._received_fps = lambda channel: 0.0
        self.bad("daq_follow", "Waiting for frames")

    def test_follow_refuses_a_frequency_outside_the_range(self):
        """A calibrated scale that would need an impossible rate says so.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings=self.triangle(
            piezo_scale=daq.CALIBRATED, piezo_quarter_wave_v=1e-5))
        self.session._received_fps = lambda channel: 30.0
        self.bad("daq_follow", "outside")

    def test_follow_needs_the_checklist_for_a_drive_signal(self):
        """An unarmed real device takes no new frequency.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings=self.triangle(piezo_frequency_hz=0.1))
        self.session._received_fps = lambda channel: 30.0
        with mock.patch.object(self.session.daq, "is_armed", return_value=False):
            self.bad("daq_follow", "checklist")

    def test_tick_follows_the_frame_rate_by_itself(self):
        """A batch run has no timer: tick() keeps the triangle in step.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings=self.triangle(piezo_frequency_hz=0.1))
        self.session._received_fps = lambda channel: 30.0
        self.session.tick()
        expected, _ = daq.follow_frequency(self.session.settings, 30.0)
        self.assertTrue(self.wait_daq(
            lambda: self.applied() is not None
            and abs(self.applied().piezo_frequency_hz - expected) < 1e-9))

    def test_the_rejected_follow_apply_is_an_error(self):
        """A settings rejection on the way through comes back with its reasons.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings=self.triangle(piezo_frequency_hz=0.1))
        self.session._received_fps = lambda channel: 30.0
        with mock.patch.object(daq, "problems", return_value=["No today."]):
            reply = self.bad("daq_follow", "rejected the settings")
        self.assertEqual(reply["problems"], ["No today."])

    def test_tick_leaves_the_triangle_alone_when_it_cannot_follow(self):
        """No frames, an unarmed device or no camera: the frequency stays put.

        Args: None.
        Returns: None.
        """
        self.session.tick()  # nothing applied yet
        self.connect_simulated()
        self.good("daq_apply", settings=self.triangle())
        self.session._received_fps = lambda channel: 0.0
        self.session.tick()  # no frames yet
        self.session._received_fps = lambda channel: 30.0
        with mock.patch.object(self.session.daq, "is_armed", return_value=False):
            self.session.tick()  # the checklist does not cover it
        self.channels[0].stop()
        self.assertTrue(wait_until(lambda: self.channels[0].engine is None, timeout=10))
        self.session.tick()  # no camera
        self.assertAlmostEqual(self.controller.status()["settings"].piezo_frequency_hz,
                               0.25, places=9)

    def test_tick_does_not_follow_while_auto_tune_runs(self):
        """Auto-tune follows the frame rate itself, so tick() leaves it alone.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings=self.triangle())
        self.good("piezo_tune")
        with mock.patch.object(self.session, "_follow_once") as follow:
            self.session.tick()
        follow.assert_not_called()


class PiezoTuneTests(SessionCase):
    """The experimental auto-tune over the JSON protocol."""

    def test_tune_needs_an_applied_triangle(self):
        """Nothing applied, a hold or an unassigned piezo are refused.

        Args: None.
        Returns: None.
        """
        self.bad("piezo_tune", "needs an applied piezo triangle")
        self.connect_simulated()
        for settings in (self.triangle(piezo_mode=daq.HOLD),
                         self.triangle(piezo_on=False, piezo_output=daq.OFF)):
            self.good("daq_apply", settings=settings)
            self.bad("piezo_tune", "needs an applied piezo triangle")

    def test_tune_validates_and_cannot_start_twice(self):
        """The flag must be a boolean, and one run at a time.

        Args: None.
        Returns: None.
        """
        self.bad("piezo_tune", '"on" must be true or false', on="yes")
        self.bad("piezo_tune", "not running", on=False)
        self.connect_simulated()
        self.good("daq_apply", settings=self.triangle())
        self.good("piezo_tune")
        self.bad("piezo_tune", "already running")

    def test_tune_starts_measures_and_stops_keeping_the_frequency(self):
        """Starting switches the monitor on; stopping keeps what it found.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings=self.triangle())
        reply = self.good("piezo_tune")
        self.assertEqual(reply["state"], "noise")
        self.assertTrue(self.channels[0].engine.step_monitor.on)
        self.assertEqual(self.good("status")["tuning"]["camera"], self.channels[0].label)
        stopped = self.good("piezo_tune", on=False)
        self.assertFalse(self.channels[0].engine.step_monitor.on)
        self.assertIsNone(self.session.tuner)
        self.assertIsNone(self.good("status")["tuning"])
        self.assertAlmostEqual(stopped["frequency_hz"], 0.25, places=9)
        self.assertEqual(stopped["camera"], self.channels[0].label)

    def test_tune_measures_only_the_chosen_camera(self):
        """Only one engine measures, so the numbers belong to one camera.

        Args: None.
        Returns: None.
        """
        self.channels[1].start()
        self.connect_simulated()
        self.good("daq_apply", settings=self.triangle())
        self.good("piezo_tune", camera=1)
        self.assertTrue(self.channels[1].engine.step_monitor.on)
        self.assertFalse(self.channels[0].engine.step_monitor.on)

    def test_tune_holds_the_piezo_then_applies_the_tuned_frequency(self):
        """The noise hold and each new factor reach the DAQ, with events.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings=self.triangle())
        self.good("piezo_tune")
        stub = StubTuner(state="noise", wants_hold=True)
        self.session.tuner = stub
        self.session.tick()
        self.assertTrue(self.wait_daq(
            lambda: self.controller.status()["settings"] is not None
            and self.controller.status()["settings"].piezo_mode == daq.HOLD))
        self.assertAlmostEqual(self.controller.status()["settings"].piezo_voltage_v, 5.5)
        stub.state, stub.wants_hold, stub.scale = "measuring", False, 1.2
        self.session.tick()
        self.assertTrue(self.wait_daq(
            lambda: self.controller.status()["settings"] is not None
            and abs(self.controller.status()["settings"].piezo_frequency_hz - 0.3) < 1e-9))
        self.assertEqual(len(stub.changes), 2)
        self.session.tick()  # nothing new: no second apply
        self.assertEqual(len(stub.changes), 2)
        self.assertTrue(any(event.get("event") == "tune" for event in self.events))

    def test_tune_with_follow_uses_the_frame_rate(self):
        """Follow on: the factor is applied to the frequency the frames need.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings=self.triangle())
        self.session._received_fps = lambda channel: 30.0
        self.good("piezo_tune")
        self.session.tuner = StubTuner(scale=2.0)
        self.session.tick()
        expected, _ = daq.follow_frequency(self.session.tune_base, 30.0)
        self.assertTrue(self.wait_daq(
            lambda: self.controller.status()["settings"] is not None
            and abs(self.controller.status()["settings"].piezo_frequency_hz
                    - 2 * expected) < 1e-9))

    def test_tune_needs_the_checklist_for_a_real_device(self):
        """A device the checklist does not cover takes no auto-tune.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings=self.triangle())
        with mock.patch.object(self.session.daq, "is_armed", return_value=False):
            self.bad("piezo_tune", "checklist")
        self.assertIsNone(self.session.tuner)

    def test_tune_logs_and_emits_a_state_change(self):
        """Reaching holding or lost is logged and sent as a tune event.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings=self.triangle())
        self.good("piezo_tune")
        stub = StubTuner(state="measuring", scale=1.5)
        self.session.tuner = stub
        self.session.tick()
        self.assertFalse(any(event.get("state") == "holding" for event in self.events))
        stub.next_state = "holding"
        self.session.tick()
        self.assertTrue(any(event.get("event") == "tune"
                            and event.get("state") == "holding" for event in self.events))
        self.assertTrue(self.log.has("Auto-tune (experimental):"))

    def test_tune_stops_when_the_phase_view_takes_the_piezo(self):
        """The Phase view steps the piezo itself, so auto-tune gives way.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings=self.triangle())
        self.good("piezo_tune")
        self.good("set_camera", settings={"mode": "Phase"})
        self.assertTrue(wait_until(
            lambda: self.channels[0].engine.settings().mode == "Phase", timeout=10))
        self.session.tick()
        self.assertIsNone(self.session.tuner)
        self.assertTrue(self.log.has("Phase view steps the piezo"))

    def test_tune_stops_when_the_camera_stops(self):
        """A camera that goes away ends the tuning with a reason.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings=self.triangle())
        self.good("piezo_tune")
        self.channels[0].stop()
        self.assertTrue(wait_until(lambda: self.channels[0].engine is None, timeout=10))
        self.session.tick()
        self.assertIsNone(self.session.tuner)
        self.assertTrue(self.log.has("the camera stopped"))

    def test_tune_stops_when_the_checklist_does_not_cover_the_outputs(self):
        """Settings the checklist no longer covers stop it rather than run on.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings=self.triangle())
        self.good("piezo_tune")
        self.session.tuner = StubTuner(scale=1.3)
        with mock.patch.object(self.session.daq, "is_armed", return_value=False):
            self.session.tick()
        self.assertIsNone(self.session.tuner)
        self.assertTrue(self.log.has("safety checklist does not cover"))

    def test_a_fault_stops_the_tuning_too(self):
        """A DAQ fault stops the stepping and the tuner, and says so.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings=self.triangle())
        self.good("piezo_tune")
        self.controller.events.put(("fault", "AO 1 read 0.9 V"))
        self.session.tick()
        self.assertIsNone(self.session.tuner)
        self.assertFalse(self.channels[0].engine.step_monitor.on)
        self.assertTrue(self.log.has("the DAQ reported fault"))

    def test_close_stops_tuning(self):
        """Closing the session measures no more.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings=self.triangle())
        self.good("piezo_tune")
        self.session.close()
        self.assertIsNone(self.session.tuner)
        self.assertFalse(self.channels[0].engine.step_monitor.on)


class PhaseCalibrateTests(SessionCase):
    """Measuring the piezo's phase step over the JSON protocol."""

    def test_calibrate_needs_the_piezo_attached_to_that_camera(self):
        """Calibrating needs the stepper, on the camera it is asked about.

        Args: None.
        Returns: None.
        """
        self.bad("phase_calibrate", "attach the piezo")
        self.channels[1].start()
        self.connect_simulated()
        self.good("daq_apply", settings={"piezo_step_v": 0.05})
        self.good("phase_attach")
        self.bad("phase_calibrate", "attach the piezo", camera=1)

    def test_calibrate_measures_the_demo_piezo(self):
        """The demo's 8π rad/V step is measured and corrected towards 90°.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings={"piezo_step_v": 0.05, "piezo_on": True})
        self.good("phase_attach")
        reply = self.good("phase_calibrate")
        self.assertGreater(reply["degrees"], 40)
        self.assertLess(reply["degrees"], 110)
        self.assertAlmostEqual(reply["step_v"], 0.0625, delta=0.03)
        self.assertAlmostEqual(self.session.stepper.step_v, reply["step_v"])
        self.assertAlmostEqual(self.session.settings.piezo_quarter_wave_v, reply["step_v"])
        self.assertAlmostEqual(self.session.settings.piezo_step_v, reply["step_v"])
        self.assertEqual(self.channels[0].engine.settings().mode, "Phase")

    def test_calibrate_reports_a_step_too_far_from_90(self):
        """A step outside the trusted range is an error, not a new voltage.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings={"piezo_step_v": 0.05})
        self.good("phase_attach")
        self.channels[0].engine.events.put(("phase_calibration_failed", (12.0, 1)))
        self.bad("phase_calibrate", "too far from 90")

    def test_calibrate_reports_a_phase_view_that_stopped(self):
        """A piezo that could not be moved ends the wait with its reason.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings={"piezo_step_v": 0.05})
        self.good("phase_attach")
        self.channels[0].engine.events.put(("phase_stopped", "the piezo could not move"))
        self.bad("phase_calibrate", "Phase view stopped")

    def test_waiting_for_an_event_times_out(self):
        """A calibration that never answers does not hang the session.

        Args: None.
        Returns: None.
        """
        with self.assertRaises(CommandError):
            self.session._await_engine(("never",), 0.05)

    def test_engine_events_are_kept_up_to_a_limit(self):
        """Only the newest events are remembered.

        Args: None.
        Returns: None.
        """
        for number in range(headless_control.MAX_EVENTS + 3):
            self.session.note_event("x", number, self.channels[0])
        self.assertEqual(len(self.session.seen_events), headless_control.MAX_EVENTS)
        self.assertEqual(self.session.seen_events[-1][1], headless_control.MAX_EVENTS + 2)

    def test_a_camera_that_never_started_has_no_frame_rate(self):
        """A stopped channel reads 0 fps rather than failing.

        Args: None.
        Returns: None.
        """
        self.assertEqual(self.session._received_fps(self.channels[1]), 0.0)


class ConfigDaqTests(SessionCase):
    """The config file's "daq" block."""

    def test_configure_connects_arms_and_applies(self):
        """The block connects, records the checklist, then switches the outputs on.

        Args: None.
        Returns: None.
        """
        self.session.configure_daq({"device": daq.SIMULATED, "arm": True,
                                    "settings": self.triangle()})
        applied = self.controller.status()["settings"]
        self.assertEqual(applied.piezo_mode, daq.TRIANGLE)
        self.assertTrue(applied.piezo_on)
        self.assertTrue(self.session.daq.checklist_done())

    def test_configure_can_start_auto_tune(self):
        """A block with "tune" starts the tuner on the first running camera.

        Args: None.
        Returns: None.
        """
        self.session.configure_daq({"device": daq.SIMULATED, "arm": True,
                                    "settings": self.triangle(), "tune": True})
        self.assertIsNotNone(self.session.tuner)
        self.assertTrue(self.channels[0].engine.step_monitor.on)

    def test_configure_arms_without_settings(self):
        """A block with only a device and "arm" still records the checklist.

        Args: None.
        Returns: None.
        """
        self.session.configure_daq({"device": daq.SIMULATED, "arm": True})
        self.assertTrue(self.session.daq.checklist_done())

    def test_an_empty_block_does_nothing(self):
        """No block, or no device, leaves the DAQ alone.

        Args: None.
        Returns: None.
        """
        for block in ({}, {"settings": self.triangle()}, {"device": ""}):
            self.session.configure_daq(block)
        self.assertIsNone(self.controller.status()["device"])

    def test_a_malformed_block_is_refused(self):
        """Every wrong shape names what it wanted.

        Args: None.
        Returns: None.
        """
        for block, text in (({"device": "Simulated", "tune": "yes"}, "true or false"),
                            ({"device": "Simulated", "settings": []}, "daq.settings"),
                            ({"device": 3}, "daq.device"),
                            ({"device": "Simulated", "extra": 1}, "unknown daq keys"),
                            (["Simulated"], "must be an object")):
            with self.assertRaises(CommandError) as caught:
                self.session.configure_daq(block)
            self.assertIn(text, str(caught.exception))

    def test_a_drive_signal_without_the_checklist_is_refused(self):
        """Arming is what lets the block switch a real output on.

        Args: None.
        Returns: None.
        """
        fake_nidaqmx.install(self)
        with self.assertRaises(CommandError) as caught:
            self.session.configure_daq({"device": "Dev1", "settings": self.triangle()})
        self.assertTrue(caught.exception.problems)


class CameraCommandTests(SessionCase):
    """Camera settings, saving and the reference."""

    def test_set_camera_applies_checked_values(self):
        """Values are cleaned like a saved profile's and reach the engine.

        Args: None.
        Returns: None.
        """
        reply = self.good("set_camera", settings={"exposure_us": 4000, "mode": "Previous"})
        self.assertEqual(reply["applied"], {"exposure_us": 4000.0, "mode": "Previous"})
        settings = self.channels[0].engine.settings()
        self.assertEqual((settings.exposure_us, settings.mode), (4000.0, "Previous"))

    def test_set_camera_changes_nothing_when_a_value_is_wrong(self):
        """One bad value rejects the whole command.

        Args: None.
        Returns: None.
        """
        before = self.channels[0].engine.settings()
        reply = self.bad("set_camera", "rejected",
                         settings={"exposure_us": 4000, "mode": "Sideways"})
        self.assertIn("mode", reply["problems"][0])
        self.assertEqual(self.channels[0].engine.settings(), before)
        self.bad("set_camera", "unknown camera settings: speed", settings={"speed": 1})
        self.bad("set_camera", "object", settings={})
        self.bad("set_camera", "object")

    def test_cameras_are_picked_by_position_label_or_serial(self):
        """Every way of naming a camera, and the ways that fail.

        Args: None.
        Returns: None.
        """
        self.channels[1].start()
        first, second = self.channels
        for wanted, expected in ((0, first), (1, second), (second.label, second),
                                 (first.info.serial, first)):
            with self.subTest(wanted):
                self.assertEqual(self.good("reference", camera=wanted)["camera"],
                                 expected.label)
        self.bad("reference", "no camera 5", camera=5)
        self.bad("reference", "no camera 'zzz'", camera="zzz")
        self.bad("reference", "no camera True", camera=True)

    def test_a_stopped_camera_cannot_be_used(self):
        """A camera that is not running is named in the error.

        Args: None.
        Returns: None.
        """
        self.bad("reference", "not running", camera=1)
        self.channels[0].stop()
        self.channels[0].wait(5)
        self.bad("reference", "no camera is running")

    def test_save_waits_for_the_file(self):
        """A save is finished before the reply, and the file is in frames/.

        Args: None.
        Returns: None.
        """
        self.assertTrue(wait_until(lambda: self.channels[0].engine.latest_packet() is not None,
                                   timeout=10))
        self.good("save", title="t", notes="n")
        self.assertFalse(self.channels[0].engine.save_pending())

    def test_save_that_never_finishes_is_an_error(self):
        """A stuck save does not hang the session.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(headless_control, "SAVE_TIMEOUT_S", 0.1), \
                mock.patch.object(headless_control, "POLL_S", 0.01), \
                mock.patch.object(self.channels[0].engine, "save_pending",
                                  return_value=True), \
                mock.patch.object(self.channels[0].engine, "request"):
            self.bad("save", "did not finish")


class WaitAndCloseTests(SessionCase):
    """Waiting, quitting and shutting down."""

    def test_wait_validates_and_waits(self):
        """Seconds must be a sensible number; the wait takes about that long.

        Args: None.
        Returns: None.
        """
        self.bad("wait", "seconds")
        self.bad("wait", "seconds", seconds="1")
        self.bad("wait", "seconds", seconds=-1)
        self.bad("wait", "seconds", seconds=True)
        self.bad("wait", "seconds", seconds=headless_control.MAX_WAIT_S + 1)
        self.assertEqual(self.good("wait", seconds=0.15)["waited"], 0.15)

    def test_wait_keeps_the_watchdog_fed(self):
        """The DAQ heartbeat is called while waiting.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(self.controller, "heartbeat") as beat:
            self.good("wait", seconds=0.2)
        self.assertGreaterEqual(beat.call_count, 1)

    def test_quit_stops_a_wait(self):
        """quit makes the session stop and wait() return early.

        Args: None.
        Returns: None.
        """
        self.session.stopping = True
        self.assertEqual(self.good("wait", seconds=30)["waited"], 30.0)
        self.assertEqual(self.good("quit"), {"ok": True})
        self.assertTrue(self.session.stopping)

    def test_close_zeroes_the_outputs_and_is_idempotent(self):
        """Closing leaves the simulated outputs off and detaches the piezo.

        Args: None.
        Returns: None.
        """
        self.connect_simulated()
        self.good("daq_apply", settings={"piezo_on": True, "piezo_voltage_v": 0.3})
        self.good("phase_attach")
        self.session.close()
        self.assertIsNone(self.controller.status()["device"])
        self.assertFalse(self.controller.status()["active"])
        self.assertIsNone(self.channels[0].engine._stepper)
        self.session.close()

    def test_close_says_when_the_daq_thread_hangs(self):
        """A DAQ thread that will not finish is logged.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(self.controller, "shutdown", return_value=False):
            self.session.close()
        self.assertTrue(self.log.has("did not finish"))


class ServeTests(SessionCase):
    """The stdin loop."""

    def serve(self, text, **options):
        """Serve commands from text and return the lines written.

        Args: text is the input; options go to headless_control.serve.
        Returns: The list of reply dicts.
        """
        replies = []
        headless_control.serve(self.session, io.StringIO(text), replies.append, **options)
        return replies

    def test_replies_come_back_in_order_until_quit(self):
        """Each line gets one reply; nothing after quit runs.

        Args: None.
        Returns: None.
        """
        text = '{"cmd":"status","id":1}\n\nnot json\n{"cmd":"quit","id":2}\n{"cmd":"status"}\n'
        replies = self.serve(text)
        self.assertEqual([r.get("id") for r in replies], [1, None, 2])
        self.assertEqual([r["ok"] for r in replies], [True, False, True])

    def test_piped_text_with_a_byte_order_mark_runs_every_command(self):
        """PowerShell's piped text (BOM, then the lines) answers the first command too.

        Args: None.
        Returns: None.
        """
        replies = self.serve('\ufeff{"cmd":"status","id":1}\r\n{"cmd":"status","id":2}\r\n')
        self.assertEqual([(r["ok"], r["id"]) for r in replies], [(True, 1), (True, 2)])

    def test_end_of_input_stops_the_loop(self):
        """Closed stdin counts as quit.

        Args: None.
        Returns: None.
        """
        self.assertEqual(len(self.serve('{"cmd":"status"}\n')), 1)
        self.assertTrue(self.log.has("input closed"))

    def test_a_read_error_counts_as_end_of_input(self):
        """A broken pipe ends the session instead of hanging it.

        Args: None.
        Returns: None.
        """
        def broken():
            """Yield one line, then fail.

            Args: None.
            Returns: A generator.
            """
            yield '{"cmd":"status"}\n'
            raise OSError("pipe closed")

        replies = []
        headless_control.serve(self.session, broken(), replies.append)
        self.assertEqual(len(replies), 1)

    def test_a_stop_request_ends_the_loop_and_idle_ticks_run(self):
        """A signal's stop event wins over waiting for input.

        Args: None.
        Returns: None.
        """
        stop, source = threading.Event(), BlockingInput()
        self.addCleanup(source.release.set)
        ticks = []

        def pump():
            """Count the tick, and ask to stop on the second one.

            Args: None.
            Returns: None.
            """
            ticks.append(1)
            if len(ticks) == 2:
                stop.set()

        self.session.pump = pump
        headless_control.serve(self.session, source, lambda reply: None, stop=stop)
        self.assertEqual(len(ticks), 2)

    def test_the_duration_ends_the_loop(self):
        """--duration still applies in a control run.

        Args: None.
        Returns: None.
        """
        source = BlockingInput()
        self.addCleanup(source.release.set)
        headless_control.serve(self.session, source, lambda reply: None, duration=0.2)
        self.assertTrue(self.log.has("Duration of 0.2 s reached"))

    def test_events_are_written_to_the_same_stream(self):
        """A fault between commands is sent as an event line.

        Args: None.
        Returns: None.
        """
        self.controller.events.put(("fault", "x"))
        replies = self.serve('{"cmd":"status"}\n')
        self.assertIn({"event": "fault", "reason": "x"}, replies)


class HeadlessRunTests(unittest.TestCase):
    """headless.run with --control, end to end on a demo camera."""

    def run_control(self, text, **options):
        """Run a control session in this process with a demo camera.

        Args: text is the stdin text; options go to headless.run.
        Returns: (exit code, stdout text, stderr text).
        """
        out, err = io.StringIO(), io.StringIO()
        config = dict(headless.DEFAULTS, demo=True)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with mock.patch.object(paths, "OUTPUT_DIR", tmp.name), \
                mock.patch.object(sys, "stdin", io.StringIO(text)), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = headless.run(config, control=True, **options)
        return code, out.getvalue(), err.getvalue()

    def test_stdout_holds_only_replies(self):
        """Log lines never reach stdout, and the outputs end at 0 V.

        Args: None.
        Returns: None.
        """
        commands = [{"cmd": "daq_connect", "device": daq.SIMULATED},
                    {"cmd": "daq_apply", "settings": {"piezo_on": True,
                                                      "piezo_voltage_v": 0.2}},
                    {"cmd": "phase_attach"},
                    {"cmd": "wait", "seconds": 0.3},
                    {"cmd": "quit"}]
        code, out, err = self.run_control("".join(json.dumps(c) + "\n" for c in commands))
        self.assertEqual(code, 0)
        replies = [json.loads(line) for line in out.splitlines()]
        self.assertEqual(len(replies), len(commands))
        self.assertTrue(all(reply["ok"] for reply in replies), replies)
        self.assertIn("Control: reading JSON commands", err)

    def test_a_control_run_works_without_a_camera(self):
        """No camera is not fatal: the DAQ commands still work, and nothing uploads.

        Args: None.
        Returns: None.
        """
        out, err = io.StringIO(), io.StringIO()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        text = '{"cmd":"daq_connect","device":"Simulated"}\n{"cmd":"status"}\n'
        with mock.patch.object(paths, "OUTPUT_DIR", tmp.name), \
                mock.patch.object(sys, "stdin", io.StringIO(text)), \
                mock.patch.object(headless, "open_channels", return_value=[]), \
                mock.patch.object(headless, "upload_outputs") as upload, \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = headless.run(dict(headless.DEFAULTS), control=True)
        self.assertEqual(code, 0)
        upload.assert_not_called()
        replies = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(replies[1]["cameras"], [])
        self.assertIn("without a camera", err.getvalue())

    def test_checklist_flag_allows_arming(self):
        """--daq-checklist-done is what makes daq_arm available.

        Args: None.
        Returns: None.
        """
        text = '{"cmd":"daq_connect","device":"Simulated"}\n{"cmd":"daq_arm"}\n'
        _, out, _ = self.run_control(text)
        self.assertFalse(json.loads(out.splitlines()[1])["ok"])
        _, out, _ = self.run_control(text, checklist_done=True)
        self.assertTrue(json.loads(out.splitlines()[1])["ok"])

    def test_a_daq_block_drives_the_outputs_in_a_batch_run(self):
        """The config file's "daq" block connects, arms and applies with no stdin.

        Args: None.
        Returns: None.
        """
        out, err = io.StringIO(), io.StringIO()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        config = dict(headless.DEFAULTS, demo=True, duration_s=0.4,
                      daq={"device": daq.SIMULATED, "arm": True,
                           "settings": {"piezo_mode": daq.TRIANGLE, "piezo_on": True,
                                        "piezo_bottom_v": 1.0, "piezo_top_v": 10.0,
                                        "piezo_limit_v": 10.0,
                                        "piezo_scale": daq.MANUAL,
                                        "piezo_frequency_hz": 0.25,
                                        "piezo_follow_fps": True}})
        with mock.patch.object(paths, "OUTPUT_DIR", tmp.name), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = headless.run(config, checklist_done=True)
        self.assertEqual(code, 0)
        text = out.getvalue() + err.getvalue()  # a batch run logs to stdout
        self.assertIn("NI DAQ: connected", text)
        self.assertIn("triangle", text)

    def test_a_malformed_daq_block_stops_the_run(self):
        """A block the DAQ cannot use is an error, not a silent run.

        Args: None.
        Returns: None.
        """
        out, err = io.StringIO(), io.StringIO()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        config = dict(headless.DEFAULTS, demo=True, duration_s=0.2,
                      daq={"device": daq.SIMULATED,
                           "settings": {"piezo_mode": daq.TRIANGLE, "piezo_on": True,
                                        "piezo_bottom_v": 10.0, "piezo_top_v": 1.0}})
        with mock.patch.object(paths, "OUTPUT_DIR", tmp.name), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = headless.run(config)
        self.assertEqual(code, 1)
        text = out.getvalue() + err.getvalue()
        self.assertIn("Headless run failed", text)
        self.assertIn("device released", text)  # the refused block still closed the DAQ


if __name__ == "__main__":
    unittest.main()
