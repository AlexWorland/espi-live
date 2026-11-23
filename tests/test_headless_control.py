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


class ProtocolTests(SessionCase):
    """Lines in, replies out."""

    def test_blank_line_gets_no_reply(self):
        """A blank line is skipped.

        Args: None.
        Returns: None.
        """
        self.assertIsNone(self.session.handle_line("  \n"))

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


if __name__ == "__main__":
    unittest.main()
