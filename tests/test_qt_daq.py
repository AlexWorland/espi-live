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

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

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
                                      side_effect=lambda *args: self.devices),
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

    def allow(self, dialog):
        """Tick off the safety checklist the last Apply opened, allowing drive signals.

        Args: dialog is the DaqDialog.
        Returns: None.
        """
        checklist = dialog.checklist
        self.assertIsNotNone(checklist)
        for box in checklist.boxes:
            box.setChecked(True)
        checklist.allow_button.click()

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
        self.allow(dialog)
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

    def test_window_fits_the_screen_and_keeps_buttons_in_view(self):
        """A tall form scrolls instead of running off a short screen.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        room = dialog.screen().availableGeometry()
        self.assertLessEqual(dialog.height(), room.height() * 9 // 10)
        self.assertLessEqual(dialog.width(), room.width() * 9 // 10)
        self.assertIs(dialog.scroll.widget(), dialog.body)
        self.assertIs(dialog.apply_button.parentWidget().parentWidget(), dialog)
        self.assertTrue(dialog.sized)
        # A size the user chose is kept; one taller than the screen shrinks.
        dialog.resize(dialog.width(), room.height() * 2)
        dialog.fit_to_screen()
        self.assertLessEqual(dialog.height(), room.height() * 9 // 10)
        dialog.resize(dialog.width() - 10, 300)
        dialog.fit_to_screen()
        self.assertEqual(dialog.height(), 300)

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

    def test_triangle_follows_the_frame_rate(self):
        """Follow sets the frequency from the frame rate, by calibration or typed numbers.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        fps = mock.patch.object(self.window, "received_fps", return_value=30.0)
        fps.start()
        self.addCleanup(fps.stop)
        for widget in (dialog.piezo_follow, dialog.piezo_phase_per_frame,
                       dialog.piezo_scale, dialog.piezo_um_per_v):
            self.assertFalse(widget.isEnabled())  # hold mode
        dialog._set_combo(dialog.piezo_mode, daq.TRIANGLE)
        dialog.piezo_limit.setValue(10.0)
        dialog.piezo_bottom.setValue(0.5)
        dialog.piezo_top.setValue(9.5)
        dialog.piezo_frequency.setValue(0.281)
        self.assertTrue(dialog.piezo_follow.isEnabled())
        self.assertFalse(dialog.piezo_phase_per_frame.isEnabled())
        self.assertEqual(dialog.piezo_follow_result.text(), "")
        dialog.piezo_follow.setChecked(True)  # no calibration yet
        self.assertFalse(dialog.piezo_frequency.isEnabled())
        self.assertTrue(dialog.piezo_phase_per_frame.isEnabled())
        self.assertFalse(dialog.piezo_um_per_v.isEnabled())
        self.assertIn("Calibrate phase step has not measured", dialog.problem_label.text())
        self.assertEqual(dialog.piezo_follow_result.text(), "")  # said once, above
        self.assertFalse(dialog.apply_button.isEnabled())
        self.assertEqual(dialog.settings().piezo_frequency_hz, 0.281)  # typed
        dialog.set_quarter_wave(0.075)
        self.assertEqual(dialog.piezo_step.value(), 0.075)
        self.assertAlmostEqual(dialog.settings().piezo_frequency_hz, 0.25)
        self.assertAlmostEqual(dialog.piezo_frequency.value(), 0.25)
        self.assertIn("0.25 Hz for 180° per frame at 30.0 fps (calibrated 0.075 V per 90°)",
                      dialog.piezo_follow_result.text())
        self.assertTrue(dialog.apply_button.isEnabled())
        dialog._set_combo(dialog.piezo_scale, daq.MANUAL)
        for widget in (dialog.piezo_um_per_v, dialog.piezo_gain, dialog.piezo_angle):
            self.assertTrue(widget.isEnabled())
        dialog.piezo_angle.setValue(45.0)
        self.assertAlmostEqual(dialog.settings().piezo_frequency_hz, 0.2486, places=4)
        self.assertIn("manual 0.1 µm/V × 15 at 45°, 632.8 nm",
                      dialog.piezo_follow_result.text())
        self.window.received_fps.return_value = 20.0
        dialog.follow_frame_rate()
        self.assertIn("at 20.0 fps", dialog.piezo_follow_result.text())
        self.window.received_fps.return_value = 0.0
        dialog.follow_frame_rate()
        self.assertIn("Waiting for frames", dialog.piezo_follow_result.text())
        s = dialog.settings()
        self.assertAlmostEqual(s.piezo_frequency_hz, 0.1657, places=4)  # the last one shown
        self.assertEqual((s.piezo_follow_fps, s.piezo_scale, s.piezo_angle_deg,
                          s.piezo_quarter_wave_v), (True, daq.MANUAL, 45.0, 0.075))
        dialog.load(s)
        self.assertEqual(dialog._typed_settings(), s)
        dialog.piezo_follow.setChecked(False)
        dialog.follow_frame_rate()  # not following: nothing to show
        self.assertEqual(dialog.piezo_follow_result.text(), "")

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
        self.assertEqual(wiring.panel.marks["pfi12"], (2, blue))  # its screw-terminal row
        self.assertIn("PFI 12 (screw terminal)", wiring.legend.text())
        dialog._set_combo(dialog.generator_output, daq.OFF)
        dialog._set_combo(dialog.piezo_output, daq.OFF)
        self.assertEqual(wiring.panel.marks, {})
        self.assertIn("Nothing to plug in yet", wiring.legend.text())
        # The drawing paints at any size and keeps its proportions.
        dialog._set_combo(dialog.piezo_output, "ao1")
        dialog._set_combo(dialog.generator_output, "ctr0")  # a marked screw row is drawn
        wiring.panel.resize(400, 640)
        self.assertFalse(wiring.panel.grab().isNull())
        self.assertTrue(wiring.panel.hasHeightForWidth())
        self.assertEqual(wiring.panel.heightForWidth(1270), 2010)
        # A socket the USB-6251 BNC does not have is listed but not marked.
        wiring.show_settings(daq.DaqSettings(piezo_output="ao2", generator_output=daq.OFF))
        self.assertEqual(wiring.panel.marks, {})
        self.assertIn("AO 2 (not on the USB-6251 BNC)", wiring.legend.text())
        # A second press reuses the window.
        wiring.hide()
        dialog.wiring_button.click()
        self.assertIs(dialog.wiring, wiring)
        self.assertTrue(wiring.isVisible())

    def test_panel_follows_the_ni_figure(self):
        """Every socket the device prints has a place: BNCs and the PFI screw rows.

        Args: None.
        Returns: None.
        """
        places = qt_app.panel_positions()
        self.assertEqual(set(places), set(daq.BNC_TERMINALS))
        self.assertEqual(places["pfi0"][2], "PFI 0/P1.0")
        self.assertEqual(len(qt_app.SCREW_ROWS), 30)
        self.assertEqual(sorted(qt_app.SCREW_TERMINALS),
                         sorted(f"pfi{index}" for index in range(8, 16)))
        self.assertEqual([qt_app.panel_place(name) for name in ("ao1", "pfi12", "ao2")],
                         ["bnc", "screw", ""])

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

    def test_safety_checklist_before_the_first_drive(self):
        """Apply opens the checklist; Allow stays off until all are ticked; once per connection.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        self.assertIn("shown before a drive signal", dialog.safety_label.text())
        dialog.apply()  # nothing switched on: no checklist needed
        self.settle(dialog, lambda: "off" in dialog.state_label.text())
        self.assertIsNone(dialog.checklist)
        dialog.piezo_on.setChecked(True)
        dialog.apply()
        checklist = dialog.checklist
        self.assertTrue(checklist.isVisible())
        self.assertEqual(len(checklist.boxes), 5)  # the generator is off: no check for it
        self.assertFalse(checklist.allow_button.isEnabled())
        for box in checklist.boxes[:-1]:
            box.setChecked(True)
        self.assertFalse(checklist.allow_button.isEnabled())
        checklist.labels[-1].mouseReleaseEvent(
            QMouseEvent(QEvent.MouseButtonRelease, QPointF(1, 1), QPointF(1, 1),
                        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        self.assertTrue(checklist.boxes[-1].isChecked())  # clicking the sentence ticks it
        self.assertTrue(checklist.allow_button.isEnabled())
        checklist.allow_button.click()
        self.settle(dialog, lambda: "holds" in dialog.state_label.text())
        self.assertIn("ticked off for what is switched on", dialog.safety_label.text())
        self.assertTrue(self.log.has("safety checklist ticked off for Dev1: USB-6251"))
        dialog.apply()  # already ticked for this connection
        self.assertIs(dialog.checklist, checklist)
        dialog.piezo_on.setChecked(False)  # nothing on: still ticked for this connection
        self.assertIn("ticked off for what is switched on", dialog.safety_label.text())
        dialog.piezo_on.setChecked(True)
        dialog.generator_on.setChecked(True)  # a role not yet covered asks again
        self.assertIn("shown before a drive signal", dialog.safety_label.text())
        dialog.apply()
        self.assertIsNot(dialog.checklist, checklist)
        self.assertEqual(len(dialog.checklist.boxes), 6)
        self.allow(dialog)
        self.settle(dialog, lambda: "sine" in dialog.state_label.text())
        checklist = dialog.checklist
        dialog.generator_on.setChecked(False)
        # Reassigning the piezo asks again; cancelling sends nothing.
        dialog._set_combo(dialog.generator_output, daq.OFF)
        dialog._set_combo(dialog.piezo_output, "ao0")
        dialog.apply()
        self.assertIsNot(dialog.checklist, checklist)
        again = dialog.checklist
        dialog.apply()  # a second Apply replaces the open checklist
        self.assertFalse(again.isVisible())
        dialog.checklist.reject()
        self.assertEqual(self.window.daq.status()["settings"].piezo_output, "ao1")

    def test_simulated_device_needs_no_checklist(self):
        """The simulated device drives nothing, so it never asks.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        dialog.device_box.setCurrentIndex(dialog.device_box.findData(daq.SIMULATED))
        dialog._connect_selected()
        self.settle(dialog, lambda: dialog.outputs is not None
                    and dialog.outputs.device.simulated)
        self.assertEqual(dialog.safety_label.text(), "")
        dialog.piezo_on.setChecked(True)
        dialog.apply()
        self.assertIsNone(dialog.checklist)
        self.settle(dialog, lambda: dialog.state_label.text().startswith("Simulated"))

    def test_self_test_button(self):
        """Test outputs asks first, then shows the result; a fault is explained.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        self.assertTrue(dialog.test_button.isEnabled())
        dialog.test_button.click()  # answered No
        self.assertEqual(dialog.test_label.text(), "")
        self.dialogs.question.return_value = self.dialogs.Yes
        dialog.test_button.click()
        self.settle(dialog, lambda: "Self-test passed" in dialog.test_label.text())
        self.assertIn("AO 1 at 0.5 V: OK", dialog.test_label.text())
        self.assertIn("after the self-test", dialog.state_label.text())
        dialog.handle("self_test", (False, ["AO 1 at 0.5 V: FAILED, AO 1 read 0.8 V"]))
        self.assertIn("Self-test FAILED", dialog.test_label.text())
        dialog.piezo_on.setChecked(True)
        dialog.handle("fault", "AO 1 read 3 V where 0.2 V was written")
        self.assertFalse(dialog.piezo_on.isChecked())
        self.assertIn("Self-check: AO 1 read 3 V", dialog.problem_label.text())
        self.assertIn("stopped by the self-check", dialog.state_label.text())
        dialog._connect_selected()  # nothing chosen changes nothing; no device: disabled
        dialog.connected(None)
        self.assertFalse(dialog.test_button.isEnabled())

    def test_window_beats_and_follows_faults(self):
        """Polling sends the heartbeat; a fault drops the drive and is logged.

        Args: None.
        Returns: None.
        """
        self.open_dialog()
        daq_controller = self.window.daq
        with mock.patch.object(daq_controller, "heartbeat") as beat:
            self.window.poll_daq()
        beat.assert_called_once_with()
        self.window.set_daq_drive(daq.Drive("sine", 1000.0, 0.1))
        self.window._daq_event("fault", "the window stopped responding for 3.0 s")
        self.assertEqual(self.window.daq_drive, daq.Drive())
        self.assertTrue(self.log.has("NI DAQ self-check: the window stopped responding"))

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
        self.allow(dialog)
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
        self.allow(dialog)
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


class NiMaxSimulatedDeviceTests(DaqWindowTestCase):
    """A device created as simulated in NI MAX is listed, and marked as such."""

    devices = ([REAL, daq.DeviceInfo("SimDev1", "NI Simulated DAQ Device", "0",
                                     ni_simulated=True)], None)

    def test_device_list_marks_the_ni_max_simulation(self):
        """The combo says which entry is NI MAX's simulation, and which is hardware.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        texts = [dialog.device_box.itemText(i) for i in range(dialog.device_box.count())]
        self.assertEqual(texts, [
            "Dev1: USB-6251 (1A2B3C4D)",
            "SimDev1: NI Simulated DAQ Device (NI MAX simulation, no hardware)",
            daq.SIMULATED_DEVICE.label])
        self.assertEqual(dialog.device_box.itemData(1), "SimDev1")


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
        # A real device takes no drive signal before the checklist: no stepping yet.
        self.pump(lambda: self.log.has("waits for the safety checklist"))
        self.assertIsNone(w.phase_stepper)
        saved = qt_app.DaqDialog._saved_settings()
        w.daq.arm(daq.replace(saved, piezo_on=True))  # the checklist covered the piezo
        w._daq_event("connected", daq.DeviceOutputs(REAL))
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

    def test_calibration_is_kept_for_the_follow_scale(self):
        """Calibrate phase step's volts are saved as the volts per 90°.

        Args: None.
        Returns: None.
        """
        w = self.window
        w._attach_stepper(qt_app.phase.PiezoStepper(mock.Mock(),
                                                    daq.DaqSettings(piezo_step_v=0.05)))
        w._handle_event(w.current, "phase_calibrated", (72.0, 0.0625))
        saved = json.loads(FakeSettings.store[qt_app.DAQ_SETTINGS_KEY])
        self.assertEqual(saved["piezo_quarter_wave_v"], 0.0625)
        w._show_daq_dialog()
        w._handle_event(w.current, "phase_calibrated", (88.0, 0.07))
        self.assertEqual(w.daq_dialog.quarter_wave_v, 0.07)

    def test_the_window_keeps_the_triangle_on_the_frame_rate(self):
        """A following triangle is re-applied when the frame rate moves it enough.

        Args: None.
        Returns: None.
        """
        w = self.window
        self.assertGreaterEqual(w.received_fps(), 0.0)
        w._follow_frame_rate()  # no DAQ yet
        applied = daq.DaqSettings(piezo_on=True, piezo_mode=daq.TRIANGLE,
                                  piezo_bottom_v=0.5, piezo_top_v=9.5, piezo_limit_v=10.0,
                                  piezo_frequency_hz=0.25, piezo_follow_fps=True,
                                  piezo_quarter_wave_v=0.075)
        controller = mock.Mock()
        controller.status.return_value = {"settings": None}
        controller.is_armed.return_value = True
        w.daq = controller
        fps = mock.patch.object(w, "received_fps", return_value=20.0)
        fps.start()
        self.addCleanup(fps.stop)
        w._follow_frame_rate()  # nothing applied
        controller.status.return_value = {"settings": daq.replace(applied, piezo_on=False)}
        w._follow_frame_rate()  # piezo off
        controller.request.assert_not_called()
        controller.status.return_value = {"settings": applied}
        w._follow_frame_rate()
        (action,), payload = controller.request.call_args
        self.assertEqual(action, "apply")
        self.assertAlmostEqual(payload["settings"].piezo_frequency_hz, 20.0 * 0.075 / 9.0)
        self.assertTrue(self.log.has("follows the camera's 20.0 fps: 0.1667 Hz"))
        w._follow_frame_rate()  # the change is on its way: not sent twice
        self.assertEqual(controller.request.call_count, 1)
        w._daq_event("rejected", ["no"])
        self.assertIsNone(w.piezo_follow_hz)
        controller.is_armed.return_value = False
        w._follow_frame_rate()  # not allowed to drive: nothing sent
        self.assertEqual(controller.request.call_count, 1)
        controller.is_armed.return_value = True
        w.daq_dialog = mock.Mock()
        w.daq_dialog.isVisible.return_value = True
        w._follow_frame_rate()
        w.daq_dialog.follow_frame_rate.assert_called_once()
        self.assertEqual(controller.request.call_count, 2)
        w.daq_dialog = None
        w.daq = None

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


class AutoTuneTests(DaqWindowTestCase):
    """The experimental auto-tune of the piezo triangle and the debug overlay."""

    TRIANGLE = daq.DaqSettings(piezo_on=True, piezo_mode=daq.TRIANGLE, piezo_bottom_v=0.5,
                               piezo_top_v=9.5, piezo_limit_v=10.0, piezo_frequency_hz=0.25)

    def controller(self, settings=TRIANGLE):
        """Give the window a stand-in DAQ controller with settings applied.

        Args: settings is the daq.DaqSettings it reports as applied.
        Returns: The mock controller.
        """
        controller = mock.Mock()
        controller.status.return_value = {"settings": settings, "device": REAL}
        controller.is_armed.return_value = True
        self.window.daq = controller
        self.addCleanup(setattr, self.window, "daq", None)
        return controller

    def sent(self, controller):
        """List the settings the window applied.

        Args: controller is the mock controller.
        Returns: A list of daq.DaqSettings.
        """
        return [call.kwargs["settings"] for call in controller.request.call_args_list
                if call.args == ("apply",)]

    def test_needs_an_applied_triangle(self):
        """Without a DAQ, or with a held piezo, it explains and stays off.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.set_auto_tune(True)
        self.assertIsNone(w.piezo_tuner)
        self.assertIn("needs a piezo triangle", w.tune_text())
        self.controller(daq.replace(self.TRIANGLE, piezo_mode=daq.HOLD))
        w.set_auto_tune(True)
        self.assertIsNone(w.piezo_tuner)
        w.set_auto_tune(False)  # already off: nothing happens
        self.assertIsNone(w.piezo_tuner)

    def test_holds_for_the_noise_then_tunes_and_keeps_the_result(self):
        """On: the piezo is held at the centre, then the triangle is tuned; off keeps it.

        Args: None.
        Returns: None.
        """
        w = self.window
        controller = self.controller()
        w.set_auto_tune(True)
        tuner = w.piezo_tuner
        self.assertIsNotNone(tuner)
        self.assertTrue(w.engine.step_monitor.on)
        self.assertTrue(self.log.has("Auto-tune (experimental): on"))
        self.assertIn("held still", w.tune_text())
        w._follow_frame_rate()  # runs the tuner
        hold = self.sent(controller)[-1]
        self.assertEqual((hold.piezo_mode, hold.piezo_voltage_v), (daq.HOLD, 5.0))
        w._run_tuner()  # nothing new to send
        self.assertEqual(len(self.sent(controller)), 1)
        tuner.state, tuner.scale = "measuring", 0.8  # the noise was measured
        w._run_tuner()
        tuned = self.sent(controller)[-1]
        self.assertEqual(tuned.piezo_mode, daq.TRIANGLE)
        self.assertAlmostEqual(tuned.piezo_frequency_hz, 0.2)
        self.assertTrue(self.log.has("Auto-tune (experimental): 0.2 Hz (×0.800)"))
        tuner.last = qt_app.step_tuning.StepStats(0.05, 176.0, 0.02, 90, 300.0)
        with mock.patch.object(tuner, "update",
                               side_effect=lambda *_: setattr(tuner, "state", "holding")):
            w._run_tuner()
        self.assertTrue(self.log.has("Auto-tune (experimental): Auto-tune: holding ×0.800; "
                                     "step about 176° per frame"))
        w.set_auto_tune(False)  # by hand: the tuned frequency stays, nothing new to send
        self.assertEqual(len(self.sent(controller)), 2)
        self.assertTrue(self.log.has("off; the triangle keeps 0.2 Hz"))
        self.assertFalse(w.engine.step_monitor.on)
        self.assertEqual(w.tune_text(), "")

    def test_turning_off_during_the_noise_hold_puts_the_triangle_back(self):
        """Stopped by hand while held, the triangle is applied again.

        Args: None.
        Returns: None.
        """
        w = self.window
        controller = self.controller()
        w.set_auto_tune(True)
        w._run_tuner()
        w.set_auto_tune(False)
        self.assertEqual(self.sent(controller)[-1], self.TRIANGLE)

    def test_follow_and_frequency_limits(self):
        """With Follow on the frame rate sets the base; the result stays in range.

        Args: None.
        Returns: None.
        """
        w = self.window
        follow = daq.replace(self.TRIANGLE, piezo_follow_fps=True, piezo_quarter_wave_v=0.075)
        self.controller(follow)
        w.set_auto_tune(True)
        with mock.patch.object(w, "received_fps", return_value=20.0):
            self.assertAlmostEqual(w._tune_target(1.0, False).piezo_frequency_hz,
                                   20.0 * 0.075 / 9.0)
        with mock.patch.object(w, "received_fps", return_value=0.0):
            self.assertEqual(w._tune_target(1.0, False).piezo_frequency_hz, 0.25)
        w.tune_settings = daq.replace(self.TRIANGLE, piezo_frequency_hz=900.0)
        self.assertEqual(w._tune_target(4.0, False).piezo_frequency_hz,
                         daq.MAX_TRIANGLE_HZ)

    def test_what_stops_it(self):
        """A camera switch, the Phase view, the checklist or the DAQ stop it.

        Args: None.
        Returns: None.
        """
        w = self.window
        controller = self.controller()
        w.set_auto_tune(True)
        w.tune_engine = mock.Mock()
        w._run_tuner()
        self.assertIn("the displayed camera changed", w.tune_text())
        w.set_auto_tune(True)
        self.engine.update_settings(mode="Phase")
        w._run_tuner()
        self.assertIn("Phase view steps the piezo", w.tune_text())
        self.engine.update_settings(mode="Previous")
        w.set_auto_tune(True)
        controller.is_armed.return_value = False
        w._run_tuner()
        self.assertIn("safety checklist", w.tune_text())
        self.assertFalse(w._send_tuned(self.TRIANGLE))
        controller.is_armed.return_value = True
        for kind, words in (("zeroed", "outputs were zeroed"),
                            ("rejected", "refused a setting")):
            w.set_auto_tune(True)
            w._daq_event(kind, ["no"] if kind == "rejected" else None)
            self.assertIsNone(w.piezo_tuner)
            self.assertIn(words, w.tune_text())
        self.assertTrue(self.log.has("Auto-tune (experimental): stopped, the NI DAQ"))

    def test_dialog_button_follows_the_tuner(self):
        """The DAQ window's button starts it, shows it, and Apply stops it.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        w = self.window
        self.assertFalse(dialog.auto_tune_button.isEnabled())  # hold mode in the form
        dialog._set_combo(dialog.piezo_mode, daq.TRIANGLE)
        self.assertTrue(dialog.auto_tune_button.isEnabled())
        dialog.auto_tune_button.click()  # nothing applied yet
        self.assertFalse(dialog.auto_tune_button.isChecked())
        self.assertIn("needs a piezo triangle", dialog.piezo_tune_result.text())
        real = w.daq
        self.controller()
        dialog.auto_tune_button.click()
        self.assertTrue(dialog.auto_tune_button.isChecked())
        self.assertIn("held still", dialog.piezo_tune_result.text())
        w.daq = real
        dialog.apply()  # new settings replace what it tunes
        self.assertIsNone(w.piezo_tuner)
        self.assertFalse(dialog.auto_tune_button.isChecked())
        self.assertIn("new settings were applied", dialog.piezo_tune_result.text())

    def test_debug_overlay(self):
        """The overlay draws the frame rate, piezo, formula, step and tuner lines.

        Args: None.
        Returns: None.
        """
        w = self.window
        self.assertFalse(w.piezo_debug.isVisible())
        self.assertIn("Piezo debug overlay", w.piezo_debug_action.text())
        w.piezo_debug_action.trigger()
        self.assertTrue(w.piezo_debug_action.isChecked())
        self.assertFalse(w.piezo_debug.isHidden())
        self.assertTrue(w.engine.step_monitor.on)
        text = w.piezo_debug.text()
        self.assertIn("camera   ", text)
        self.assertIn("fps received", text)
        self.assertIn("piezo    not driven", text)
        self.assertIn("step     waiting for frames", text)
        controller = self.controller(daq.replace(self.TRIANGLE, piezo_follow_fps=True,
                                                 piezo_quarter_wave_v=0.075))
        monitor = w.engine.step_monitor
        for n in range(5):
            monitor.feed(qt_app.np.full((8, 8), float(n % 2) * 10), time.monotonic())
        text = w.piezo_debug_text()
        self.assertIn("triangle 0.5–9.5 V at 0.25 Hz", text)
        self.assertIn("formula  ", text)
        self.assertIn("(noise not taken off)", text)
        self.assertIn("flicker", text)
        w.set_auto_tune(True)
        w.piezo_tuner.noise = 1.0
        text = w.piezo_debug_text()
        self.assertNotIn("noise not taken off", text)
        self.assertIn("tune     noise ×1.000", text)
        for settings, words in ((daq.replace(self.TRIANGLE, piezo_mode=daq.HOLD,
                                             piezo_voltage_v=2.0), "held at 2 V"),
                                (daq.replace(self.TRIANGLE, piezo_mode=daq.SINE,
                                             piezo_amplitude_v=0.3), "sine 0.3 V peak")):
            controller.status.return_value["settings"] = settings
            self.assertIn(words, w.piezo_debug_text())
        w.piezo_debug_action.trigger()
        self.assertTrue(w.piezo_debug.isHidden())
