"""Camera-free checks for the NI DAQ window (offscreen platform, demo engine).

Run from the project root:
    PYTHONPATH=src QT_QPA_PLATFORM=offscreen python -m unittest tests.test_qt_daq

The window's DaqController runs on simulated backends; no NI hardware,
driver, network or real settings store is touched.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import time
import unittest
from unittest import mock

from espi_live import daq, qt_app
from test_qt_app import APP, FakeSettings, WindowTestCase

REAL = daq.DeviceInfo("Dev1", "USB-6251", "1A2B3C4D")


class RealLikeBackend(daq.SimulatedBackend):
    """A simulated backend that reports itself as a real USB-6251."""

    simulated = False

    def open(self, device):
        """Pretend to connect to Dev1.

        Args: device is ignored.
        Returns: DeviceOutputs for a USB-6251 named Dev1.
        """
        self.outputs = daq.DeviceOutputs(REAL)
        return self.outputs


def backend_for(device):
    """Hand out backends for the tests' controller.

    Args: device is a DAQmx name or daq.SIMULATED.
    Returns: SimulatedBackend for the simulated device, else RealLikeBackend.
    """
    return daq.SimulatedBackend() if device == daq.SIMULATED else RealLikeBackend()


class DaqWindowTestCase(WindowTestCase):
    """A window whose DAQ controller uses test backends and a fixed device list."""

    devices = ([REAL], None)

    def setUp(self):
        """Build the window, then patch the device list and the backends.

        Args: None.
        Returns: None.
        """
        super().setUp()
        patchers = (mock.patch.object(qt_app.daq, "list_devices",
                                      side_effect=lambda: self.devices),
                    mock.patch.object(qt_app.daq, "default_backend", backend_for))
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self._shutdown_daq)

    def _shutdown_daq(self):
        """Stop the DAQ thread if a test opened the window.

        Args: None.
        Returns: None.
        """
        if self.window.daq is not None:
            self.window.daq.shutdown(5.0)

    def open_dialog(self):
        """Open the DAQ window and wait until it has connected.

        Args: None.
        Returns: The DaqDialog.
        """
        self.window._show_daq_dialog()
        dialog = self.window.daq_dialog
        self.settle(dialog, lambda: dialog.outputs is not None)
        return dialog

    def settle(self, dialog, predicate, timeout=5.0):
        """Poll the dialog until a condition holds.

        Args: dialog is the DaqDialog; predicate is a zero-argument callable;
            timeout is in seconds.
        Returns: None. Fails the test when the condition never holds.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            APP.processEvents()
            dialog.poll()
            if predicate():
                return
            time.sleep(0.01)
        self.fail("the DAQ window never reached the expected state: device "
                  f"{dialog.device_status.text()!r}, outputs {dialog.state_label.text()!r}, "
                  f"problems {dialog.problem_label.text()!r}")


class DaqMenuTests(DaqWindowTestCase):
    """The Devices menu and the window's lifetime."""

    def test_menu_opens_one_window_and_controller(self):
        """The menu action builds the controller and window once.

        Args: None.
        Returns: None.
        """
        self.assertIsNone(self.window.daq)
        self.assertIn("NI DAQ", self.window.daq_action.text())
        self.assertTrue(self.window.daq_action.toolTip())
        dialog = self.open_dialog()
        controller = self.window.daq
        self.window._show_daq_dialog()
        self.assertIs(self.window.daq_dialog, dialog)
        self.assertIs(self.window.daq, controller)
        self.assertIn("Connected to Dev1: USB-6251", dialog.device_status.text())
        self.assertEqual(dialog.device_status.styleSheet(), "")

    def test_closing_the_window_zeroes_the_outputs(self):
        """Closing ESPI live shuts the DAQ down first.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        dialog.generator_on.setChecked(True)
        dialog.apply()
        self.settle(dialog, lambda: "sine" in dialog.state_label.text())
        self.window.close()
        self.assertTrue(self.log.has("NI DAQ: outputs at 0 V, device released"))

    def test_slow_shutdown_is_logged(self):
        """A DAQ that does not stop in time leaves a warning.

        Args: None.
        Returns: None.
        """
        self.window.daq = mock.Mock()
        self.window.daq.shutdown.return_value = False
        self.window.close()
        self.assertTrue(self.log.has("did not confirm 0 V"))
        self.window.daq = None


class DaqDialogTests(DaqWindowTestCase):
    """The form: assignment, enabling, preview, apply and the replies."""

    def test_assignments_enable_the_right_controls(self):
        """Counters take a terminal and no volts; piezo modes enable their fields.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        generator_outputs = [dialog.generator_output.itemData(i)
                             for i in range(dialog.generator_output.count())]
        self.assertEqual(generator_outputs, [daq.OFF, "ao0", "ao1", "ctr0", "ctr1"])
        self.assertEqual(dialog.generator_output.itemText(3), "Counter 0 (square, 5 V TTL)")
        self.assertFalse(dialog.generator_terminal.isEnabled())
        dialog._set_combo(dialog.generator_output, "ctr0")
        self.assertTrue(dialog.generator_terminal.isEnabled())
        self.assertFalse(dialog.amplitude.isEnabled())
        self.assertIn("square, on PFI 4", dialog.generator_result.text())
        dialog._set_combo(dialog.generator_output, daq.OFF)
        self.assertFalse(dialog.frequency.isEnabled())
        self.assertEqual(dialog.generator_result.text(), "")
        self.assertTrue(dialog.step_up.isEnabled())
        self.assertFalse(dialog.piezo_phase.isEnabled())
        dialog._set_combo(dialog.piezo_mode, daq.SINE)
        self.assertTrue(dialog.piezo_phase.isEnabled())
        self.assertFalse(dialog.step_up.isEnabled())
        self.assertFalse(dialog.piezo_bottom.isEnabled())
        dialog._set_combo(dialog.piezo_mode, daq.TRIANGLE)
        for widget in (dialog.piezo_bottom, dialog.piezo_top, dialog.piezo_frequency):
            self.assertTrue(widget.isEnabled())
        self.assertFalse(dialog.piezo_voltage.isEnabled())
        self.assertFalse(dialog.piezo_phase.isEnabled())
        self.assertFalse(dialog.step_up.isEnabled())
        dialog._set_combo(dialog.piezo_output, daq.OFF)
        self.assertFalse(dialog.piezo_on.isEnabled())
        self.assertFalse(dialog.piezo_amplitude.isEnabled())
        self.assertFalse(dialog.piezo_bottom.isEnabled())
        self.assertEqual(dialog.piezo_result.text(), "")

    def test_triangle_for_the_thor(self):
        """The THOR's 0-10 V triangle is previewed, refused beside a sine, and applied.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        self.assertEqual(dialog.piezo_voltage.minimum(), 0.0)
        self.assertEqual(dialog.piezo_bottom.minimum(), 0.0)
        self.assertEqual(dialog.piezo_top.maximum(), daq.MAX_VOLTS)
        dialog._set_combo(dialog.piezo_mode, daq.TRIANGLE)
        dialog.piezo_limit.setValue(10.0)
        dialog.piezo_bottom.setValue(0.5)
        dialog.piezo_top.setValue(9.5)
        dialog.piezo_frequency.setValue(0.281)
        self.assertIn("0.281001 Hz triangle from 0.5 V to 9.5 V", dialog.piezo_result.text())
        self.assertIn("2000 samples per cycle", dialog.piezo_result.text())
        self.assertTrue(dialog.apply_button.isEnabled())
        dialog.piezo_on.setChecked(True)
        dialog.generator_on.setChecked(True)
        self.assertIn("share one sample clock", dialog.problem_label.text())
        self.assertEqual(dialog.piezo_result.text(), "")
        self.assertFalse(dialog.apply_button.isEnabled())
        dialog.generator_on.setChecked(False)
        s = dialog.settings()
        self.assertEqual((s.piezo_mode, s.piezo_bottom_v, s.piezo_top_v, s.piezo_frequency_hz),
                         (daq.TRIANGLE, 0.5, 9.5, 0.281))
        dialog.load(daq.DaqSettings())
        self.assertEqual(dialog.piezo_mode.currentData(), daq.HOLD)
        self.assertEqual((dialog.piezo_bottom.value(), dialog.piezo_top.value(),
                          dialog.piezo_frequency.value()), (0.0, 1.0, 0.25))

    def test_where_to_plug_in(self):
        """The help window marks the sockets in use and follows the form.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        self.assertIsNone(dialog.wiring)
        dialog.wiring_button.click()
        wiring = dialog.wiring
        self.assertTrue(wiring.isVisible())
        self.assertEqual(wiring.windowTitle(), qt_app.WIRING_TITLE)
        orange, blue = (qt_app.ROLE_COLORS[role] for role in (daq.PIEZO_ROLE,
                                                               daq.GENERATOR_ROLE))
        self.assertEqual(wiring.panel.marks, {"ao1": (1, orange), "ao0": (2, blue)})
        self.assertIn("1. AO 1", wiring.legend.text())
        self.assertIn(daq.PIEZO_AMPLIFIER, wiring.legend.text())
        self.assertIn("Connected: Dev1: USB-6251", wiring.device_note.text())
        # It follows the form while open, including a screw-terminal PFI.
        dialog._set_combo(dialog.generator_output, "ctr0")
        self.assertEqual(wiring.panel.marks["pfi4"], (2, blue))
        dialog._set_combo(dialog.generator_terminal, "pfi12")
        self.assertNotIn("pfi12", wiring.panel.marks)
        self.assertIn("screw terminal, not drawn", wiring.legend.text())
        dialog._set_combo(dialog.generator_output, daq.OFF)
        dialog._set_combo(dialog.piezo_output, daq.OFF)
        self.assertEqual(wiring.panel.marks, {})
        self.assertIn("Nothing to plug in yet", wiring.legend.text())
        # The drawing paints at any size and keeps its proportions.
        dialog._set_combo(dialog.piezo_output, "ao1")
        wiring.panel.resize(700, 340)
        self.assertFalse(wiring.panel.grab().isNull())
        self.assertTrue(wiring.panel.hasHeightForWidth())
        self.assertEqual(wiring.panel.heightForWidth(940), 450)
        # A second press reuses the window.
        wiring.hide()
        dialog.wiring_button.click()
        self.assertIs(dialog.wiring, wiring)
        self.assertTrue(wiring.isVisible())

    def test_where_to_plug_in_names_the_device(self):
        """The note says when there is no device, a simulated one or another model.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        dialog.show_wiring()
        settings = daq.DaqSettings()
        dialog.wiring.show_settings(settings, None)
        self.assertIn("simulated device has no sockets", dialog.wiring.device_note.text())
        dialog.wiring.show_settings(settings, daq.SIMULATED_DEVICE)
        self.assertIn("simulated device has no sockets", dialog.wiring.device_note.text())
        dialog.wiring.show_settings(settings, daq.DeviceInfo("Dev2", "USB-6009"))
        self.assertIn("sockets may differ", dialog.wiring.device_note.text())

    def test_problems_block_apply(self):
        """A clash or an unsafe level is explained and disables Apply.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        self.assertTrue(dialog.apply_button.isEnabled())
        self.assertIn("250 samples per cycle", dialog.generator_result.text())
        dialog._set_combo(dialog.piezo_output, "ao0")
        self.assertIn("both on AO 0", dialog.problem_label.text())
        self.assertFalse(dialog.apply_button.isEnabled())
        self.assertEqual(dialog.generator_result.text(), "")

    def test_apply_saves_and_reports(self):
        """Apply stores the settings and shows what the device produces.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        dialog.generator_on.setChecked(True)
        dialog.piezo_on.setChecked(True)
        dialog.piezo_voltage.setValue(0.25)
        dialog.apply()
        self.settle(dialog, lambda: "holds 0.25 V" in dialog.state_label.text())
        self.assertTrue(dialog.state_label.text().startswith("Outputs: "))
        saved = json.loads(FakeSettings.store[qt_app.DAQ_SETTINGS_KEY])
        self.assertEqual(saved["device"], "Dev1")
        self.assertTrue(saved["generator_on"])
        self.assertEqual(self.window.daq.status()["active"], True)

    def test_step_moves_the_piezo(self):
        """A step changes the held voltage, and applies it when the piezo is on.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        dialog.piezo_step.setValue(0.05)
        with mock.patch.object(dialog, "apply") as apply:
            dialog.step(1)
            apply.assert_not_called()
            dialog.piezo_on.setChecked(True)
            dialog.step(-1)
            apply.assert_called_once()
        self.assertAlmostEqual(dialog.piezo_voltage.value(), 0.0)

    def test_replies_update_the_form(self):
        """Rejections, zeroing and failures are shown, and switch the outputs off.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        events = self.window.daq.events
        events.put(("rejected", ["One.", "Two."]))
        dialog.poll()
        self.assertEqual(dialog.problem_label.text(), "One.\nTwo.")
        self.assertEqual(dialog.state_label.text(), "Outputs: not changed.")
        dialog.generator_on.setChecked(True)
        dialog.piezo_on.setChecked(True)
        events.put(("zeroed", None))
        events.put(("closed", None))
        dialog.poll()
        self.assertFalse(dialog.generator_on.isChecked())
        self.assertEqual(dialog.state_label.text(), "Outputs: all at 0 V.")
        dialog.generator_on.setChecked(True)
        events.put(("failed", "DeviceError: unplugged"))
        dialog.poll()
        self.assertIn("unplugged", dialog.problem_label.text())
        self.assertFalse(dialog.generator_on.isChecked())
        self.assertEqual(dialog.state_label.text(), "Outputs: stopped after an error.")

    def test_zero_button_reaches_the_device(self):
        """All outputs to 0 V asks the controller and reports back.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        dialog.zero_button.click()
        self.settle(dialog, lambda: dialog.state_label.text() == "Outputs: all at 0 V.")

    def test_hide_and_show(self):
        """Hiding stops polling; showing again does not rescan.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        dialog.reject()
        self.assertFalse(dialog.timer.isActive())
        with mock.patch.object(dialog, "scan") as scan:
            self.window._show_daq_dialog()
            scan.assert_not_called()
        self.assertTrue(dialog.timer.isActive())

    def test_rescan_and_device_choice(self):
        """Rescan lists devices again; choosing the simulated one says so.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        dialog.rescan_button.click()
        self.assertEqual(dialog.device_status.text(), "Looking for NI devices…")
        # Wait for the rescan's own list and connection, not the one from opening:
        # a late list would reconnect to Dev1 after the simulated device is chosen.
        self.settle(dialog, lambda: dialog.outputs is not None
                    and dialog.device_status.text() != "Looking for NI devices…")
        dialog.device_box.setCurrentIndex(dialog.device_box.findData(daq.SIMULATED))
        dialog._connect_selected()
        self.settle(dialog, lambda: dialog.outputs is not None
                    and dialog.outputs.device.simulated)
        self.assertIn("nothing is driven", dialog.device_status.text())
        self.assertIn("background:#8b0000", dialog.device_status.styleSheet())
        dialog.generator_on.setChecked(True)
        dialog.apply()
        self.settle(dialog, lambda: dialog.state_label.text().startswith("Simulated"))
        dialog.device_box.clear()
        dialog._connect_selected()  # nothing chosen: nothing happens

    def test_saved_settings_are_restored(self):
        """Stored settings fill the form; unreadable ones give the defaults.

        Args: None.
        Returns: None.
        """
        stored = daq.DaqSettings(device="Dev1", generator_output="ctr1",
                                 generator_terminal="pfi6", generator_frequency_hz=440.0,
                                 piezo_mode=daq.SINE, piezo_phase_deg=30.0)
        FakeSettings.store[qt_app.DAQ_SETTINGS_KEY] = json.dumps(stored.to_dict())
        self.assertEqual(qt_app.DaqDialog._saved_settings(), stored)
        dialog = self.open_dialog()
        self.assertEqual(dialog.generator_output.currentData(), "ctr1")
        self.assertEqual(dialog.generator_terminal.currentData(), "pfi6")
        self.assertEqual(dialog.frequency.value(), 440.0)
        self.assertEqual(dialog.piezo_mode.currentData(), daq.SINE)
        FakeSettings.store[qt_app.DAQ_SETTINGS_KEY] = "{not json"
        self.assertEqual(qt_app.DaqDialog._saved_settings(), daq.DaqSettings())


class DriveHandOffTests(DaqWindowTestCase):
    """What the DAQ outputs produce reaches the simulated drumhead."""

    def test_applied_drive_reaches_the_window_and_channels(self):
        """Apply hands the plan's drive to the window and every channel.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        self.assertEqual(self.window.daq_drive, daq.Drive())
        dialog.frequency.setValue(1000.0)
        dialog.generator_on.setChecked(True)
        dialog.apply()
        self.settle(dialog, lambda: self.window.daq_drive.waveform == "sine")
        self.assertAlmostEqual(self.window.daq_drive.frequency_hz, 1000.0, delta=0.01)
        self.assertIs(self.window.current.drive, self.window.daq_drive)

    def test_zero_and_failure_switch_the_drive_off(self):
        """All outputs to 0 V, or a DAQ error, leaves the drum undriven.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        on = daq.Drive("sine", 1000.0, 0.1)
        self.window.set_daq_drive(on)
        self.window.daq.events.put(("zeroed", None))
        dialog.poll()
        self.assertEqual(self.window.daq_drive, daq.Drive())
        self.window.set_daq_drive(on)
        self.window.daq.events.put(("failed", "DaqError: gone"))
        dialog.poll()
        self.assertEqual(self.window.daq_drive, daq.Drive())

    def test_hidden_dialog_still_hands_off(self):
        """The window drains the DAQ's events while its window is closed.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        dialog.hide()
        self.window.set_daq_drive(daq.Drive("sine", 1000.0, 0.1))
        self.window.daq.events.put(("failed", "DaqError: gone"))
        self.window._poll_status()
        self.assertEqual(self.window.daq_drive, daq.Drive())

    def test_hidden_dialog_leaves_device_lists_for_later(self):
        """Closed, the DAQ window connects to nothing; it picks the list up when shown.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        dialog.hide()
        with mock.patch.object(dialog, "_connect_selected") as connect:
            self.window.daq.events.put(("devices", ([], None)))
            self.window._poll_status()
            connect.assert_not_called()
            dialog.show()
            dialog.poll()
            connect.assert_called_once_with()


class NoDeviceTests(DaqWindowTestCase):
    """Without NI support or devices, only the simulated device is offered."""

    devices = ([], "NI-DAQmx runs on Windows and Linux only.")

    def test_reason_is_shown_with_the_simulated_device(self):
        """The reason stays in view next to the simulated device's warning.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        self.assertEqual(dialog.device_box.count(), 1)
        self.assertTrue(dialog.outputs.device.simulated)
        text = dialog.device_status.text()
        self.assertIn("Windows and Linux only", text)
        self.assertIn("nothing is driven", text)

    def test_no_devices_found(self):
        """With support but no device, the window says to plug one in.

        Args: None.
        Returns: None.
        """
        self.devices = ([], None)
        dialog = self.open_dialog()
        self.assertIn("No NI device found", dialog.device_status.text())


class HelperTests(unittest.TestCase):
    """Small helpers of the DAQ window."""

    def test_choice_text_and_spin(self):
        """Generator choices name their signal; number boxes take their limits.

        Args: None.
        Returns: None.
        """
        self.assertEqual(qt_app.generator_choice_text(daq.OFF), "Off (not assigned)")
        self.assertEqual(qt_app.generator_choice_text("ao1"), "AO 1 (sine)")
        spin = qt_app.daq_spin(-1.0, 2.0, 3, " V", 0.5)
        self.assertEqual((spin.minimum(), spin.maximum(), spin.decimals()), (-1.0, 2.0, 3))
        self.assertEqual((spin.suffix(), spin.singleStep()), (" V", 0.5))


if __name__ == "__main__":
    unittest.main()


class PhaseWindowTests(DaqWindowTestCase):
    """The window follows the piezo: Phase view, steps, calibration."""

    def pump(self, predicate, timeout=5.0):
        """Run the window's DAQ events until a condition holds.

        Args: predicate is a zero-argument callable; timeout is in seconds.
        Returns: None. Fails the test when the condition never holds.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            APP.processEvents()
            self.window.poll_daq()
            if predicate():
                return
            time.sleep(0.01)
        self.fail("the window never reached the expected state")

    def save(self, **fields):
        """Store DAQ settings as Apply would have saved them.

        Args: fields are DaqSettings values.
        Returns: The DaqSettings stored.
        """
        settings = daq.DaqSettings(**fields)
        FakeSettings.store[qt_app.DAQ_SETTINGS_KEY] = json.dumps(settings.to_dict())
        return settings

    def test_startup_reconnects_the_saved_piezo_and_opens_the_phase_view(self):
        """A saved device with a piezo connects at startup and selects Phase.

        Args: None.
        Returns: None.
        """
        w = self.window
        self.assertFalse(w.phase_mode_action.isVisible())
        self.assertFalse(w.calibrate_phase_button.isEnabled())
        self.assertIn("not connected", w.phase_state.text())
        self.save(device="Dev1", piezo_output="ao1", piezo_voltage_v=0.2, piezo_step_v=0.06,
                  generator_on=True)
        w._auto_connect_daq()
        self.assertTrue(self.log.has("looking for Dev1"))
        self.pump(lambda: w.phase_stepper is not None)
        stepper = w.phase_stepper
        self.assertIs(self.engine.phase_stepper, stepper)
        self.assertFalse(stepper.simulated)
        self.assertFalse(stepper.settings.generator_on)  # all outputs are at 0 V now
        self.assertEqual(self.engine.settings().mode, "Phase")
        self.assertTrue(w.phase_mode_action.isVisible())
        self.assertTrue(w.calibrate_phase_button.isEnabled())
        self.assertIn("AO 1, from 0.2 V in steps of 0.06 V.", w.phase_state.text())
        self.assertNotIn("Simulated", w.phase_state.text())
        self.assertTrue(self.log.has("Phase view on (phase-shifting ESPI)"))
        w.poll_daq()  # nothing waiting: returns quietly

    def test_startup_leaves_what_it_cannot_use(self):
        """Nothing saved, no piezo, a simulated device for a real camera, a missing device.

        Args: None.
        Returns: None.
        """
        w = self.window
        w._auto_connect_daq()
        self.assertIsNone(w.daq)
        w.poll_daq()  # no controller yet: nothing to do
        self.save(device="Dev1", piezo_output=daq.OFF)
        w._auto_connect_daq()
        self.assertIsNone(w.daq)
        self.save(device="", piezo_output="ao1")
        w._auto_connect_daq()
        self.assertIsNone(w.daq)
        self.save(device=daq.SIMULATED, piezo_output="ao1")
        with mock.patch.object(w, "current", mock.Mock(demo=False)):
            w._auto_connect_daq()
        self.assertIsNone(w.daq)
        self.save(device="Dev9", piezo_output="ao1")
        w._auto_connect_daq()
        self.pump(lambda: self.log.has("Dev9 was not found"))
        self.assertIsNone(w.phase_stepper)

    def test_simulated_piezo_for_dummy_mode(self):
        """The saved simulated device reconnects for a synthetic camera.

        Args: None.
        Returns: None.
        """
        self.save(device=daq.SIMULATED, piezo_output="ao1")
        self.window._auto_connect_daq()
        self.pump(lambda: self.window.phase_stepper is not None)
        self.assertTrue(self.window.phase_stepper.simulated)
        self.assertIn("Simulated device: nothing is driven.", self.window.phase_state.text())

    def test_opening_the_daq_window_alone_never_drives_the_piezo(self):
        """Without settings saved by Apply, connecting attaches no piezo.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        self.assertIsNone(self.window.phase_stepper)
        self.assertNotEqual(self.engine.settings().mode, "Phase")
        dialog.piezo_output.setCurrentIndex(dialog.piezo_output.findData(daq.OFF))
        dialog.apply()
        self.settle(dialog, lambda: "Outputs:" in dialog.state_label.text()
                    and "applying" not in dialog.state_label.text())
        self.assertIsNone(self.window.phase_stepper)

    def test_apply_attaches_the_piezo_and_zero_takes_it_away(self):
        """Apply with a piezo gives the Phase view a stepper; 0 V removes it.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        dialog.apply()  # the defaults assign the piezo to AO 1
        self.settle(dialog, lambda: self.window.phase_stepper is not None)
        self.assertEqual(self.engine.settings().mode, "Phase")
        self.engine.update_settings(mode="Reference")  # moved away by hand
        dialog.apply()
        self.settle(dialog, lambda: self.window.phase_stepper is not None
                    and "applying" not in dialog.state_label.text())
        self.assertEqual(self.engine.settings().mode, "Reference")  # not forced back
        dialog.zero_button.click()
        self.settle(dialog, lambda: self.window.phase_stepper is None)
        self.assertTrue(self.log.has("piezo stepping stopped"))
        self.assertIsNone(self.engine.phase_stepper)

    def test_calibration_and_stop_events(self):
        """A calibrated step is saved and used; failures and stops are reported.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.calibrate_phase()
        self.dialogs.information.assert_called_once()
        self.assertEqual(self.queued(), [])
        stepper = qt_app.phase.PiezoStepper(mock.Mock(), daq.DaqSettings(piezo_step_v=0.05))
        w._attach_stepper(stepper)
        self.engine.update_settings(mode="Reference")
        w.calibrate_phase()
        self.assertEqual(self.engine.settings().mode, "Phase")
        self.assertIn(("calibrate_phase", {}), self.queued())
        w._handle_event(w.current, "phase_calibrated", (72.0, 0.0625))
        self.assertEqual(w.phase_stepper.step_v, 0.0625)
        self.assertIs(self.engine.phase_stepper, w.phase_stepper)
        saved = json.loads(FakeSettings.store[qt_app.DAQ_SETTINGS_KEY])
        self.assertEqual(saved["piezo_step_v"], 0.0625)
        self.assertIn("now 0.0625 V per step", w.statusBar().currentMessage())
        w._show_daq_dialog()
        w._handle_event(w.current, "phase_calibrated", (88.0, 0.07))
        self.assertEqual(w.daq_dialog.piezo_step.value(), 0.07)
        w._handle_event(w.current, "phase_calibration_failed", 12.0)
        self.assertIn("12° per step", w.statusBar().currentMessage())
        w._handle_event(w.current, "phase_stopped", "gone")
        self.assertIsNone(w.phase_stepper)
        self.assertIn("Phase view stopped: gone", w.statusBar().currentMessage())
        w._handle_event(w.current, "phase_calibrated", (90.0, 0.08))  # no piezo now
        self.assertIsNone(w.phase_stepper)
        w._handle_event(mock.Mock(), "phase_stopped", "elsewhere")  # not displayed

    def test_phase_controls_reach_the_engine(self):
        """Steps, filter and settle are sent and shown per camera.

        Args: None.
        Returns: None.
        """
        w = self.window
        self.assertEqual(w.phase_steps.currentData(), 5)
        w.phase_steps.setCurrentIndex(w.phase_steps.findData(4))
        w.phase_filter.spin.setValue(5)
        w.phase_settle.spin.setValue(3)
        s = self.engine.settings()
        self.assertEqual((s.phase_steps, s.phase_filter, s.phase_settle_frames), (4, 5, 3))
        w._load_settings(daq.replace(s, phase_steps=5, phase_filter=11,
                                     phase_settle_frames=1), {})
        self.assertEqual((w.phase_steps.currentData(), w.phase_filter.spin.value(),
                          w.phase_settle.spin.value()), (5, 11, 1))
        self.assertEqual(self.engine.settings().phase_steps, 4)  # not sent back
