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
        self.fail("the DAQ window never reached the expected state")


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
        dialog._set_combo(dialog.piezo_output, daq.OFF)
        self.assertFalse(dialog.piezo_on.isEnabled())
        self.assertFalse(dialog.piezo_amplitude.isEnabled())

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
        self.settle(dialog, lambda: dialog.outputs is not None)
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
