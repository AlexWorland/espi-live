"""Checks for the NI DAQ outputs (frequency generator and piezo driver).

Run from the project root:
    PYTHONPATH=src python -m unittest tests.test_daq

No NI hardware or driver is needed: pyLabLib's real NIDAQ class runs on an
in-memory nidaqmx (fake_nidaqmx), and the controller is checked with the
simulated backend.
"""

import math
import queue
import time
import types
import unittest
from unittest import mock

import numpy as np
from pylablib.core.devio.base import DeviceError

import fake_nidaqmx
from espi_live import daq, error_reports
from list_log import ListLog


def wait_event(controller, kind, timeout=5.0):
    """Wait for one kind of controller event, collecting the others.

    Args: controller is the DaqController; kind is the event kind wanted;
        timeout is in seconds.
    Returns: (payload, list of the (kind, payload) events seen before it).
        Fails the test when it does not arrive.
    """
    seen = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            event = controller.events.get(timeout=0.05)
        except queue.Empty:
            continue
        if event[0] == kind:
            return event[1], seen
        seen.append(event)
    raise AssertionError(f"no {kind!r} event; saw {seen}")


class LabelTests(unittest.TestCase):
    """Names for outputs, terminals and devices."""

    def test_output_and_terminal_labels(self):
        """Outputs and terminals read as on the front panel.

        Args: None.
        Returns: None.
        """
        self.assertEqual(daq.output_label(daq.OFF), "Off")
        self.assertEqual(daq.output_label("ao1"), "AO 1")
        self.assertEqual(daq.output_label("ctr0"), "Counter 0")
        self.assertEqual(daq.output_label("port0"), "PORT0")
        self.assertEqual(daq.terminal_label("pfi12"), "PFI 12")
        self.assertEqual(daq.terminal_label("rtsi0"), "RTSI0")

    def test_device_labels(self):
        """Real devices show name, model and serial; the simulated one says so.

        Args: None.
        Returns: None.
        """
        self.assertEqual(daq.DeviceInfo("Dev1", "USB-6251", "01A2").label,
                         "Dev1: USB-6251 (01A2)")
        self.assertEqual(daq.DeviceInfo("Dev2", "USB-6251").label, "Dev2: USB-6251")
        self.assertFalse(daq.DeviceInfo("Dev1", "USB-6251").simulated)
        self.assertTrue(daq.SIMULATED_DEVICE.simulated)
        self.assertIn("no hardware", daq.SIMULATED_DEVICE.label)

    def test_ni_max_simulated_device_is_marked_but_not_the_stand_in(self):
        """An NI MAX simulated device says so in its label, yet is not `simulated`.

        `simulated` is what lets a device skip the safety checklist, so it stays
        False for a device NI-DAQmx merely reports as simulated.

        Args: None.
        Returns: None.
        """
        ni_max = daq.DeviceInfo("SimDev1", "NI Simulated DAQ Device", "0", ni_simulated=True)
        self.assertEqual(ni_max.label,
                         "SimDev1: NI Simulated DAQ Device (NI MAX simulation, no hardware)")
        self.assertFalse(ni_max.simulated)
        self.assertFalse(daq.DeviceInfo("Dev1", "USB-6251").ni_simulated)
        self.assertFalse(daq.SIMULATED_DEVICE.ni_simulated)


class WiringTests(unittest.TestCase):
    """Which socket each role uses, and where its cable goes."""

    def test_socket_labels(self):
        """Every socket is named as printed on the device.

        Args: None.
        Returns: None.
        """
        self.assertEqual([daq.socket_label(name) for name in
                          ("ai7", "ao1", "apfi0", "pfi12", "user2", "ctr0")],
                         ["AI 7", "AO 1", "APFI 0", "PFI 12", "USER 2", "Counter 0"])
        self.assertEqual(len(daq.BNC_TERMINALS), 8 + 2 + 1 + 8 + 2)

    def test_piezo_and_sine_generator(self):
        """The default roles: AO 1 to the MDT694A first, then AO 0.

        Args: None.
        Returns: None.
        """
        piezo, generator = daq.wiring(daq.DaqSettings(generator_limit_v=2.5))
        self.assertEqual((piezo.terminal, piezo.role, piezo.target, piezo.label),
                         ("ao1", daq.PIEZO_ROLE, daq.PIEZO_AMPLIFIER, "AO 1"))
        self.assertIn("0–10 V", piezo.note)
        self.assertTrue(piezo.bnc)
        self.assertEqual((generator.terminal, generator.role), ("ao0", daq.GENERATOR_ROLE))
        self.assertIn("2.5 V safety limit", generator.note)

    def test_counter_terminals(self):
        """A counter's cable goes on its PFI terminal; PFI 8-15 have no BNC.

        Args: None.
        Returns: None.
        """
        (square,) = daq.wiring(daq.DaqSettings(piezo_output=daq.OFF, generator_output="ctr0",
                                               generator_terminal="pfi4"))
        self.assertEqual((square.terminal, square.label, square.bnc), ("pfi4", "PFI 4", True))
        self.assertEqual(square.note, "0–5 V square wave (TTL).")
        (screw,) = daq.wiring(daq.DaqSettings(piezo_output=daq.OFF, generator_output="ctr1",
                                              generator_terminal="pfi12"))
        self.assertFalse(screw.bnc)
        self.assertIn("PFI 12 is a screw terminal", screw.note)

    def test_nothing_assigned(self):
        """No roles, no cables.

        Args: None.
        Returns: None.
        """
        self.assertEqual(daq.wiring(daq.DaqSettings(piezo_output=daq.OFF,
                                                    generator_output=daq.OFF)), [])


class LoadingTests(unittest.TestCase):
    """The lazy import and the reasons NI devices can be unavailable."""

    def setUp(self):
        """Forget any loaded module for the test.

        Args: None.
        Returns: None.
        """
        patcher = mock.patch.object(daq, "_NI", None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_load_ni_imports_once(self):
        """The module is imported on first use and cached.

        Args: None.
        Returns: None.
        """
        module = types.SimpleNamespace(daq=types.SimpleNamespace(nidaqmx=object()))
        with mock.patch.object(daq.importlib, "import_module",
                               return_value=module) as importer:
            self.assertIs(daq.load_ni(), module)
            self.assertIs(daq.load_ni(), module)
        importer.assert_called_once_with("pylablib.devices.NI")

    def test_load_ni_needs_nidaqmx(self):
        """pyLabLib without nidaqmx counts as not installed.

        Args: None.
        Returns: None.
        """
        module = types.SimpleNamespace(daq=types.SimpleNamespace(nidaqmx=None))
        with mock.patch.object(daq.importlib, "import_module", return_value=module):
            with self.assertRaises(ImportError):
                daq.load_ni()

    def test_support_text(self):
        """--check's line says available only when both packages import.

        Args: None.
        Returns: None.
        """
        ready = types.SimpleNamespace(daq=types.SimpleNamespace(nidaqmx=object()))
        missing = types.SimpleNamespace(daq=types.SimpleNamespace(nidaqmx=None))
        for result, word in ((ready, "available"), (missing, "not installed"),
                             (ImportError("no pylablib"), "not installed")):
            option = ({"side_effect": result} if isinstance(result, Exception)
                      else {"return_value": result})
            with mock.patch.object(daq.importlib, "import_module", **option):
                self.assertTrue(daq.support_text().endswith(word))
        self.assertIsNone(daq._NI)  # nothing cached

    def test_unavailable_reasons(self):
        """Each failure explains what to do.

        Args: None.
        Returns: None.
        """
        self.assertIn("pip install", daq.unavailable_reason(ImportError("x")))
        not_found = type("DaqNotFoundError", (Exception,), {})
        self.assertIn("driver", daq.unavailable_reason(not_found("x")))
        unsupported = type("DaqNotSupportedError", (Exception,), {})
        self.assertIn("Windows and Linux", daq.unavailable_reason(unsupported("x")))
        self.assertIn("boom", daq.unavailable_reason(RuntimeError("boom")))

    def test_list_devices_reports_missing_support(self):
        """Without pyLabLib there are no devices, and a reason.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(daq, "load_ni", side_effect=ImportError("no module")):
            devices, reason = daq.list_devices()
        self.assertEqual(devices, [])
        self.assertIn("not installed", reason)


class SettingsTests(unittest.TestCase):
    """Saving and restoring DaqSettings."""

    def test_round_trip(self):
        """to_dict and from_dict restore every value.

        Args: None.
        Returns: None.
        """
        settings = daq.DaqSettings(device="Dev1", generator_output="ctr1",
                                   generator_on=True, generator_frequency_hz=250.0,
                                   piezo_mode=daq.SINE, piezo_phase_deg=45.0)
        self.assertEqual(daq.DaqSettings.from_dict(settings.to_dict()), settings)
        triangle = daq.DaqSettings(piezo_mode=daq.TRIANGLE, piezo_bottom_v=0.5,
                                   piezo_top_v=9.5, piezo_frequency_hz=0.281)
        self.assertEqual(daq.DaqSettings.from_dict(triangle.to_dict()), triangle)
        older = daq.DaqSettings.from_dict({"piezo_mode": daq.HOLD})  # before triangles
        self.assertEqual(older.piezo_frequency_hz, daq.DaqSettings().piezo_frequency_hz)

    def test_bad_values_keep_defaults(self):
        """Unknown keys, wrong types and non-finite numbers are ignored.

        Args: None.
        Returns: None.
        """
        default = daq.DaqSettings()
        self.assertEqual(daq.DaqSettings.from_dict("not a dict"), default)
        restored = daq.DaqSettings.from_dict({
            "unknown": 1, "generator_on": 1, "piezo_on": True,
            "generator_frequency_hz": float("nan"), "generator_amplitude_v": True,
            "piezo_voltage_v": 2, "device": 7, "piezo_mode": daq.SINE})
        self.assertEqual(restored.generator_on, default.generator_on)
        self.assertTrue(restored.piezo_on)
        self.assertEqual(restored.generator_frequency_hz, default.generator_frequency_hz)
        self.assertEqual(restored.generator_amplitude_v, default.generator_amplitude_v)
        self.assertEqual(restored.piezo_voltage_v, 2.0)
        self.assertIsInstance(restored.piezo_voltage_v, float)
        self.assertEqual(restored.device, default.device)
        self.assertEqual(restored.piezo_mode, daq.SINE)


class FollowFrequencyTests(unittest.TestCase):
    """The piezo triangle's frequency worked out from the camera's frame rate."""

    def triangle(self, **fields):
        """Build triangle settings that follow the frame rate.

        Args: fields override DaqSettings values.
        Returns: DaqSettings: 0.5-9.5 V triangle, calibrated at 0.075 V per 90°.
        """
        values = dict(piezo_mode=daq.TRIANGLE, piezo_bottom_v=0.5, piezo_top_v=9.5,
                      piezo_limit_v=10.0, piezo_follow_fps=True,
                      piezo_quarter_wave_v=0.075)
        values.update(fields)
        return daq.DaqSettings(**values)

    def test_calibrated_scale(self):
        """f = fps x V90 / (top - bottom) for 180° per frame, and scales with fps.

        Args: None.
        Returns: None.
        """
        s = self.triangle()
        self.assertAlmostEqual(daq.phase_per_volt(s)[0], (math.pi / 2) / 0.075)
        self.assertAlmostEqual(daq.follow_frequency(s, 30.0)[0], 30.0 * 0.075 / 9.0)
        self.assertAlmostEqual(daq.follow_frequency(s, 20.0)[0], 20.0 * 0.075 / 9.0)
        quarter = daq.replace(s, piezo_phase_per_frame_deg=90.0)
        self.assertAlmostEqual(daq.follow_frequency(quarter, 30.0)[0], 0.125)
        self.assertEqual(daq.follow_frequency(s, 30.0)[1], "")

    def test_manual_scale_at_the_hene_wavelength(self):
        """Travel x gain x cos(angle) at 632.8 nm gives the phase per volt.

        Args: None.
        Returns: None.
        """
        self.assertEqual(daq.DaqSettings().piezo_angle_deg, 45.0)  # the lab's fold mirror
        s = self.triangle(piezo_scale=daq.MANUAL, piezo_um_per_v=0.1, piezo_gain=15.0,
                          piezo_angle_deg=0.0)
        per_volt = 4 * math.pi * 1500.0 / 632.8
        self.assertEqual(daq.HENE_WAVELENGTH_NM, 632.8)
        self.assertAlmostEqual(daq.phase_per_volt(s)[0], per_volt)
        self.assertAlmostEqual(daq.follow_frequency(s, 30.0)[0],
                               math.pi * 30.0 / (2 * 9.0 * per_volt))
        tilted = daq.replace(s, piezo_angle_deg=45.0)
        self.assertAlmostEqual(daq.follow_frequency(tilted, 30.0)[0],
                               daq.follow_frequency(s, 30.0)[0] / math.cos(math.pi / 4))
        self.assertAlmostEqual(daq.follow_frequency(tilted, 30.0)[0], 0.2486, places=4)

    def test_what_stops_it(self):
        """A missing calibration, bad numbers or no frame rate give a reason instead.

        Args: None.
        Returns: None.
        """
        cases = (
            (self.triangle(piezo_quarter_wave_v=0.0), 30.0, "Calibrate phase step"),
            (self.triangle(piezo_scale="odd"), 30.0, "Unknown piezo scale"),
            (self.triangle(piezo_scale=daq.MANUAL, piezo_gain=0.0), 30.0,
             "must be above 0"),
            (self.triangle(piezo_scale=daq.MANUAL, piezo_um_per_v=0.0), 30.0,
             "must be above 0"),
            (self.triangle(piezo_scale=daq.MANUAL, piezo_angle_deg=85.0), 30.0,
             "angle of incidence"),
            (self.triangle(piezo_phase_per_frame_deg=0.0), 30.0, "phase per frame"),
            (self.triangle(piezo_phase_per_frame_deg=400.0), 30.0, "phase per frame"),
            (self.triangle(piezo_top_v=0.5), 30.0, "top must be above"),
            (self.triangle(), 0.0, "Waiting for frames"),
            (self.triangle(), None, "Waiting for frames"))
        for settings, fps, words in cases:
            frequency, reason = daq.follow_frequency(settings, fps)
            self.assertIsNone(frequency)
            self.assertIn(words, reason)
            self.assertEqual(daq.follow_text(settings, fps), reason)

    def test_text(self):
        """The window's line names the frequency, the frame rate and the scale.

        Args: None.
        Returns: None.
        """
        self.assertEqual(daq.follow_text(self.triangle(), 30.0),
                         "Follows the camera: 0.25 Hz for 180° per frame at 30.0 fps "
                         "(calibrated 0.075 V per 90°).")
        manual = self.triangle(piezo_scale=daq.MANUAL, piezo_angle_deg=45.0)
        self.assertIn("manual 0.1 µm/V × 15 at 45°, 632.8 nm", daq.follow_text(manual, 20.0))
        fast = self.triangle(piezo_quarter_wave_v=5.0, piezo_top_v=0.51)
        self.assertIn("outside 0.01–1,000 Hz", daq.follow_text(fast, 30.0))

    def test_followed_only_when_it_moves_enough(self):
        """A change is made for a following triangle, past the tolerance, in range.

        Args: None.
        Returns: None.
        """
        s = self.triangle(piezo_output="ao1", piezo_frequency_hz=0.25)
        self.assertIsNone(daq.followed(s, 30.0))  # already right
        self.assertIsNone(daq.followed(s, 30.0 * (1 + daq.FOLLOW_TOLERANCE / 2)))
        moved = daq.followed(s, 20.0)
        self.assertAlmostEqual(moved.piezo_frequency_hz, 20.0 * 0.075 / 9.0)
        self.assertEqual(daq.replace(moved, piezo_frequency_hz=0.25), s)
        self.assertIsNotNone(daq.followed(daq.replace(s, piezo_frequency_hz=0.0), 30.0))
        for other in (daq.replace(s, piezo_follow_fps=False),
                      daq.replace(s, piezo_mode=daq.HOLD),
                      daq.replace(s, piezo_output=daq.OFF)):
            self.assertIsNone(daq.followed(other, 20.0))
        self.assertIsNone(daq.followed(s, 0.0))  # no frame rate yet
        fast = daq.replace(s, piezo_quarter_wave_v=5.0, piezo_top_v=0.51)
        self.assertIsNone(daq.followed(fast, 30.0))  # beyond the triangle's range

    def test_problems_and_storage(self):
        """Following needs a scale; the new values are saved and restored.

        Args: None.
        Returns: None.
        """
        found = daq.problems(self.triangle(piezo_quarter_wave_v=0.0))
        self.assertTrue(any("Calibrate phase step" in line for line in found))
        self.assertEqual(daq.problems(self.triangle()), [])
        flat = daq.problems(self.triangle(piezo_quarter_wave_v=0.0, piezo_top_v=0.5))
        self.assertEqual(flat, ["The piezo triangle's top must be above its bottom."])
        manual = self.triangle(piezo_scale=daq.MANUAL, piezo_um_per_v=0.09,
                               piezo_gain=10.0, piezo_angle_deg=45.0,
                               piezo_phase_per_frame_deg=120.0)
        self.assertEqual(daq.DaqSettings.from_dict(manual.to_dict()), manual)
        older = daq.DaqSettings.from_dict({"piezo_mode": daq.TRIANGLE})
        self.assertFalse(older.piezo_follow_fps)
        self.assertEqual(older.piezo_quarter_wave_v, 0.0)


class TimingTests(unittest.TestCase):
    """Sample rate and samples per cycle for the sine."""

    def test_exact_frequencies(self):
        """Round frequencies come out exactly, at a modest rate.

        Args: None.
        Returns: None.
        """
        self.assertEqual(daq.choose_timing(1000.0, 2), (250_000.0, 250, 1000.0))
        rate, samples, actual = daq.choose_timing(0.1, 1)
        self.assertEqual((samples, actual), (2000, 0.1))
        self.assertAlmostEqual(rate, 200.0)
        rate, samples, actual = daq.choose_timing(100_000.0, 2)
        self.assertEqual((rate, samples, actual), (2_000_000.0, 20, 100_000.0))

    def test_awkward_frequency_keeps_many_samples(self):
        """A frequency within 0.01 % gets many samples per cycle.

        Args: None.
        Returns: None.
        """
        rate, samples, actual = daq.choose_timing(1234.5, 2)
        self.assertGreaterEqual(samples, 64)
        self.assertLessEqual(abs(actual - 1234.5) / 1234.5, daq.CLOSE_ENOUGH)
        divisor = daq.TIMEBASE_HZ / rate
        self.assertAlmostEqual(divisor, round(divisor))
        self.assertAlmostEqual(actual, rate / samples)

    def test_unreachable_frequency_gets_the_closest(self):
        """When nothing is within 0.01 %, the smallest error wins.

        Args: None.
        Returns: None.
        """
        rate, samples, actual = daq.choose_timing(77_000.0, 2)
        self.assertLessEqual(rate, daq.MAX_AO_RATE[2])
        self.assertLess(abs(actual - 77_000.0) / 77_000.0, 0.002)

    def test_rate_limit_override_and_channel_count(self):
        """A lower device limit and the channel count bound the rate.

        Args: None.
        Returns: None.
        """
        rate, samples, _ = daq.choose_timing(10_000.0, 1, max_rate=200_000.0)
        self.assertLessEqual(rate, 200_000.0)
        self.assertEqual(daq.choose_timing(50_000.0, 5)[0], 2_000_000.0)
        self.assertLessEqual(daq.choose_timing(100_000.0, 1)[0], daq.MAX_AO_RATE[1])

    def test_square_frequency(self):
        """The counter's half periods are whole timebase ticks.

        Args: None.
        Returns: None.
        """
        self.assertEqual(daq.square_frequency(1000.0), 1000.0)
        self.assertAlmostEqual(daq.square_frequency(1234.5), 1234.5, delta=0.1)
        self.assertEqual(daq.square_frequency(1e9), daq.COUNTER_TIMEBASE_HZ / 4)


class ProblemTests(unittest.TestCase):
    """Validation of output assignments and levels."""

    def check(self, needle, **changes):
        """Expect one problem mentioning a phrase.

        Args: needle is text the problem must contain; changes are
            DaqSettings fields.
        Returns: None.
        """
        found = daq.problems(daq.DaqSettings(**changes))
        self.assertTrue(any(needle in item for item in found), found)

    def test_defaults_are_valid(self):
        """The default settings can be applied.

        Args: None.
        Returns: None.
        """
        self.assertEqual(daq.problems(daq.DaqSettings()), [])

    def test_each_problem(self):
        """Every rule names what is wrong.

        Args: None.
        Returns: None.
        """
        self.check("no AO 7", generator_output="ao7")
        self.check("needs an analog output", piezo_output="ctr0")
        self.check("both on AO 0", generator_output="ao0", piezo_output="ao0")
        self.check("safety limit", generator_limit_v=0.0)
        self.check("safety limit", piezo_limit_v=11.0)
        self.check("PFI terminal", generator_output="ctr0", generator_terminal="pfi99")
        self.check("between", generator_frequency_hz=200_000.0)
        self.check("between", generator_frequency_hz=0.01)
        self.check("above its 1 V limit", generator_amplitude_v=0.8, generator_offset_v=0.5)
        self.check("piezo driver would reach", piezo_voltage_v=2.0)
        self.check("piezo driver would reach", piezo_mode=daq.SINE, piezo_voltage_v=0.5,
                   piezo_amplitude_v=0.6)
        self.check("Unknown piezo mode", piezo_mode="ramp")
        self.check("go down to -0.1 V", piezo_voltage_v=-0.1)
        self.check("go down to -0.2 V", piezo_mode=daq.SINE, piezo_voltage_v=0.1,
                   piezo_amplitude_v=0.3)
        self.check("go down to -0.5 V", piezo_mode=daq.TRIANGLE, piezo_bottom_v=-0.5)
        self.check("piezo driver would reach 2 V", piezo_mode=daq.TRIANGLE,
                   piezo_top_v=2.0)
        self.check("top must be above its bottom", piezo_mode=daq.TRIANGLE,
                   piezo_bottom_v=0.5, piezo_top_v=0.5)
        self.check("triangle's frequency must be between", piezo_mode=daq.TRIANGLE,
                   piezo_frequency_hz=0.001)
        self.check("triangle's frequency must be between", piezo_mode=daq.TRIANGLE,
                   piezo_frequency_hz=5000.0)
        self.check("share one sample clock", piezo_mode=daq.TRIANGLE, piezo_on=True,
                   generator_on=True)
        self.check("follows the frequency generator", piezo_mode=daq.SINE, piezo_on=True)
        self.check("follows the frequency generator", piezo_mode=daq.SINE, piezo_on=True,
                   generator_on=True, generator_output="ctr0")

    def test_triangle_beside_a_counter_or_an_idle_generator(self):
        """A triangle runs while the generator is a counter, off or unassigned.

        Args: None.
        Returns: None.
        """
        triangle = dict(piezo_mode=daq.TRIANGLE, piezo_on=True, piezo_bottom_v=0.0,
                        piezo_top_v=1.0)
        for generator in (dict(generator_output="ctr0", generator_on=True),
                          dict(generator_on=False), dict(generator_output=daq.OFF)):
            self.assertEqual(daq.problems(daq.DaqSettings(**generator, **triangle)), [])
        # Unassigned, the piezo's levels are not checked.
        self.assertEqual(daq.problems(daq.DaqSettings(piezo_output=daq.OFF,
                                                      piezo_voltage_v=-1.0)), [])

    def test_piezo_range(self):
        """Each mode's lowest and highest level.

        Args: None.
        Returns: None.
        """
        s = daq.DaqSettings(piezo_voltage_v=0.4, piezo_amplitude_v=-0.1,
                            piezo_bottom_v=0.9, piezo_top_v=0.2)
        self.assertEqual(daq.piezo_range(s), (0.4, 0.4))
        low, high = daq.piezo_range(daq.replace(s, piezo_mode=daq.SINE))
        self.assertAlmostEqual(low, 0.3)
        self.assertAlmostEqual(high, 0.5)
        self.assertEqual(daq.piezo_range(daq.replace(s, piezo_mode=daq.TRIANGLE)), (0.2, 0.9))
        self.assertEqual(daq.piezo_range(daq.replace(s, piezo_mode="ramp")), (0.4, 0.4))

    def test_square_wave_allows_high_frequencies_and_ignores_volts(self):
        """A counter goes to 1 MHz and its amplitude does not matter.

        Args: None.
        Returns: None.
        """
        settings = daq.DaqSettings(generator_output="ctr1", generator_frequency_hz=500_000.0,
                                   generator_amplitude_v=9.0)
        self.assertEqual(daq.problems(settings), [])

    def test_unassigned_generator_skips_frequency(self):
        """An unassigned generator's frequency is not checked.

        Args: None.
        Returns: None.
        """
        settings = daq.DaqSettings(generator_output=daq.OFF, generator_frequency_hz=0.0)
        self.assertEqual(daq.problems(settings), [])


class PlanTests(unittest.TestCase):
    """What plan() writes for each combination."""

    def test_invalid_settings_raise(self):
        """plan() refuses settings with problems.

        Args: None.
        Returns: None.
        """
        with self.assertRaises(ValueError):
            daq.plan(daq.DaqSettings(piezo_voltage_v=5.0))

    def test_sine_with_held_piezo(self):
        """A sine on AO 0 and a held level on AO 1 share one clock.

        Args: None.
        Returns: None.
        """
        settings = daq.DaqSettings(generator_on=True, generator_amplitude_v=0.5,
                                   generator_offset_v=0.1, piezo_on=True,
                                   piezo_voltage_v=0.25)
        p = daq.plan(settings)
        self.assertEqual(p.analog, ("ao0", "ao1"))
        self.assertEqual((p.rate, p.samples_per_cycle, p.frequency_hz),
                         (250_000.0, 250, 1000.0))
        self.assertEqual(p.buffer_samples % p.samples_per_cycle, 0)
        self.assertGreaterEqual(p.buffer_samples, p.rate * daq.MIN_WRITE_S)
        wave = p.values["ao0"]
        self.assertAlmostEqual(wave.max(), 0.6, places=3)
        self.assertAlmostEqual(wave.min(), -0.4, places=3)
        self.assertAlmostEqual(wave[0], 0.1)
        self.assertEqual(p.values["ao1"], 0.25)
        self.assertIsNone(p.counter)
        self.assertIn("holds 0.25 V", p.summary)

    def test_piezo_sine_follows_the_generator(self):
        """The piezo sine has the generator's period and the set phase.

        Args: None.
        Returns: None.
        """
        settings = daq.DaqSettings(generator_on=True, piezo_on=True, piezo_mode=daq.SINE,
                                   piezo_voltage_v=0.5, piezo_amplitude_v=0.3,
                                   piezo_phase_deg=90.0)
        p = daq.plan(settings)
        piezo = p.values["ao1"]
        self.assertEqual(len(piezo), p.samples_per_cycle)
        self.assertAlmostEqual(piezo[0], 0.8)
        self.assertEqual(p.piezo_frequency_hz, 0.0)
        self.assertIn("at 90°", p.summary)

    def test_triangle_wave_shape(self):
        """The triangle starts at its bottom, peaks half-way and is symmetric.

        Args: None.
        Returns: None.
        """
        wave = daq.triangle_wave(1.0, 9.0, 8)
        np.testing.assert_allclose(wave, [1, 3, 5, 7, 9, 7, 5, 3])

    def test_piezo_triangle_on_its_own_clock(self):
        """The THOR's slow ramp: a positive triangle at its own frequency on AO 1.

        Args: None.
        Returns: None.
        """
        settings = daq.DaqSettings(piezo_on=True, piezo_mode=daq.TRIANGLE,
                                   piezo_bottom_v=0.5, piezo_top_v=9.5,
                                   piezo_frequency_hz=0.281, piezo_limit_v=10.0)
        p = daq.plan(settings)
        self.assertEqual(p.analog, ("ao0", "ao1"))
        self.assertEqual(p.values["ao0"], 0.0)  # the generator, off
        wave = p.values["ao1"]
        self.assertEqual(len(wave), p.samples_per_cycle)
        self.assertEqual(p.buffer_samples % p.samples_per_cycle, 0)
        self.assertEqual(wave[0], 0.5)
        self.assertAlmostEqual(wave.max(), 9.5)
        self.assertGreaterEqual(wave.min(), 0.0)
        self.assertLessEqual(abs(p.piezo_frequency_hz - 0.281) / 0.281, daq.CLOSE_ENOUGH)
        self.assertAlmostEqual(p.piezo_frequency_hz, p.rate / p.samples_per_cycle)
        self.assertEqual(p.frequency_hz, 0.0)
        self.assertIn("piezo AO 1 triangle 0.5–9.5 V", p.summary)
        self.assertIn("generator AO 0 off", p.summary)
        alone = daq.plan(daq.replace(settings, generator_output=daq.OFF))
        self.assertEqual(alone.analog, ("ao1",))

    def test_square_wave_and_off_outputs(self):
        """A counter generator needs no AO clock; an off piezo sits at 0 V.

        Args: None.
        Returns: None.
        """
        settings = daq.DaqSettings(generator_output="ctr0", generator_terminal="pfi2",
                                   generator_on=True, generator_frequency_hz=500.0)
        p = daq.plan(settings)
        self.assertEqual(p.analog, ("ao1",))
        self.assertEqual(p.rate, 0.0)
        self.assertEqual(p.values, {"ao1": 0.0})
        self.assertEqual(p.counter, daq.CounterPlan("ctr0", "pfi2", 0.001, 0.001))
        self.assertEqual(p.frequency_hz, 500.0)
        self.assertIn("on PFI 2", p.summary)
        self.assertIn("piezo AO 1 off (0 V)", p.summary)

    def test_generator_off_and_nothing_assigned(self):
        """Off roles are listed; with no roles the plan says so.

        Args: None.
        Returns: None.
        """
        p = daq.plan(daq.DaqSettings())
        self.assertIn("generator AO 0 off", p.summary)
        self.assertEqual(p.values, {"ao0": 0.0, "ao1": 0.0})
        empty = daq.plan(daq.DaqSettings(generator_output=daq.OFF, piezo_output=daq.OFF))
        self.assertEqual(empty.analog, ())
        self.assertEqual(empty.summary, "no outputs assigned")


class DriveTests(unittest.TestCase):
    """Each output plan says what the outputs produce, for the ESPI simulation."""

    SIMULATED_LABEL = "Simulated device (no hardware)"

    def test_sine_generator_and_held_piezo(self):
        """A sine and a held piezo level are described with the real frequency.

        Args: None.
        Returns: None.
        """
        p = daq.plan(daq.DaqSettings(generator_on=True, generator_frequency_hz=1000,
                                     generator_amplitude_v=0.1, piezo_on=True,
                                     piezo_voltage_v=0.05))
        self.assertEqual(p.drive, daq.Drive("sine", p.frequency_hz, 0.1, 0.0, daq.HOLD,
                                            0.05, 0.0, 0.0, self.SIMULATED_LABEL, True))
        self.assertTrue(p.drive.active)

    def test_square_counter(self):
        """A counter's square wave has the counter's frequency and no amplitude.

        Args: None.
        Returns: None.
        """
        s = daq.DaqSettings(generator_output="ctr0", generator_on=True,
                            generator_frequency_hz=777.0, generator_amplitude_v=0.3)
        drive = daq.plan(s).drive
        self.assertEqual((drive.waveform, drive.frequency_hz, drive.amplitude_v),
                         ("square", daq.square_frequency(777.0), 0.0))
        self.assertEqual(drive.piezo_mode, "")

    def test_sine_piezo(self):
        """A piezo sine keeps its centre, amplitude and phase.

        Args: None.
        Returns: None.
        """
        s = daq.DaqSettings(generator_on=True, piezo_on=True, piezo_mode=daq.SINE,
                            piezo_voltage_v=0.3, piezo_amplitude_v=0.2,
                            piezo_phase_deg=90.0)
        drive = daq.plan(s).drive
        self.assertEqual((drive.piezo_mode, drive.piezo_v, drive.piezo_amplitude_v,
                          drive.piezo_phase_deg, drive.piezo_frequency_hz),
                         (daq.SINE, 0.3, 0.2, 90.0, 0.0))

    def test_triangle_piezo(self):
        """A piezo triangle is described by its centre, half its span and its frequency.

        Args: None.
        Returns: None.
        """
        s = daq.DaqSettings(generator_output="ctr0", generator_on=True, piezo_on=True,
                            piezo_mode=daq.TRIANGLE, piezo_bottom_v=0.1, piezo_top_v=0.9,
                            piezo_frequency_hz=0.5)
        p = daq.plan(s)
        drive = p.drive
        self.assertEqual(drive.piezo_mode, daq.TRIANGLE)
        self.assertAlmostEqual(drive.piezo_v, 0.5)
        self.assertAlmostEqual(drive.piezo_amplitude_v, 0.4)
        self.assertEqual(drive.piezo_frequency_hz, p.piezo_frequency_hz)
        self.assertAlmostEqual(drive.piezo_frequency_hz, 0.5)
        self.assertEqual(drive.waveform, "square")

    def test_off_is_inactive(self):
        """Outputs that are off describe no drive.

        Args: None.
        Returns: None.
        """
        drive = daq.plan(daq.DaqSettings()).drive
        self.assertFalse(drive.active)
        self.assertEqual((drive.waveform, drive.piezo_mode), ("", ""))
        self.assertFalse(daq.Drive().active)
        unassigned = daq.plan(daq.DaqSettings(generator_output=daq.OFF, piezo_output=daq.OFF,
                                              generator_on=True, piezo_on=True)).drive
        self.assertFalse(unassigned.active)

    def test_real_device_label(self):
        """A real device is named and not marked simulated.

        Args: None.
        Returns: None.
        """
        outputs = daq.DeviceOutputs(daq.DeviceInfo("Dev1", "USB-6251"))
        drive = daq.plan(daq.DaqSettings(piezo_on=True, piezo_voltage_v=0.2), outputs).drive
        self.assertEqual((drive.device, drive.simulated), ("Dev1: USB-6251", False))
        self.assertEqual((drive.piezo_mode, drive.piezo_v), (daq.HOLD, 0.2))
        self.assertTrue(drive.active)


class SimulatedBackendTests(unittest.TestCase):
    """The stand-in device."""

    def test_records_without_driving(self):
        """It reports USB-6251 outputs and remembers what it was given.

        Args: None.
        Returns: None.
        """
        backend = daq.SimulatedBackend()
        outputs = backend.open("anything")
        self.assertTrue(outputs.device.simulated)
        self.assertEqual(outputs.analog, daq.AO_OUTPUTS)
        p = daq.plan(daq.DaqSettings(generator_on=True))
        backend.apply(p)
        self.assertIs(backend.applied, p)
        backend.zero()
        self.assertIsNone(backend.applied)
        self.assertEqual(backend.zeroed, 1)
        backend.close()
        self.assertIsNone(backend.outputs)
        self.assertIsInstance(daq.default_backend(daq.SIMULATED), daq.SimulatedBackend)
        self.assertIsInstance(daq.default_backend("Dev1"), daq.PylablibBackend)


class PylablibBackendTests(unittest.TestCase):
    """The real pyLabLib NIDAQ on an in-memory nidaqmx."""

    def setUp(self):
        """Install the fake nidaqmx and connect to Dev1.

        Args: None.
        Returns: None.
        """
        self.hardware = fake_nidaqmx.install(self)
        self.backend = daq.PylablibBackend()
        self.outputs = self.backend.open("Dev1")

    def apply(self, **changes):
        """Plan and apply settings.

        Args: changes are DaqSettings fields.
        Returns: The OutputPlan.
        """
        p = daq.plan(daq.DaqSettings(device="Dev1", **changes), self.outputs)
        self.backend.apply(p)
        return p

    def test_open_reads_the_device(self):
        """Model, serial, channels and PFI terminals come from the device.

        Args: None.
        Returns: None.
        """
        info = self.outputs.device
        self.assertEqual((info.name, info.model, info.serial), ("Dev1", "USB-6251", "1A2B3C4D"))
        self.assertEqual(self.outputs.analog, ("ao0", "ao1"))
        self.assertEqual(self.outputs.counters, ("ctr0", "ctr1"))
        self.assertEqual(self.outputs.terminals, daq.PFI_TERMINALS)
        devices, reason = daq.list_devices()
        self.assertIsNone(reason)
        self.assertEqual(devices, [info])
        self.assertFalse(info.ni_simulated)

    def test_list_devices_reads_the_ni_max_simulation_flag(self):
        """list_devices and a connection mark a device nidaqmx reports as simulated.

        Args: None.
        Returns: None.
        """
        fake_nidaqmx.install(self, devices=("Dev1", "SimDev1"), simulated=("SimDev1",))
        devices, reason = daq.list_devices()
        self.assertIsNone(reason)
        self.assertEqual([(d.name, d.ni_simulated, d.simulated) for d in devices],
                         [("Dev1", False, False), ("SimDev1", True, False)])
        self.assertIn("NI MAX simulation", devices[1].label)
        backend = daq.PylablibBackend()
        self.addCleanup(backend.close)
        self.assertTrue(backend.open("SimDev1").device.ni_simulated)
        self.assertFalse(self.outputs.device.ni_simulated)  # Dev1, connected in setUp

    def test_unreadable_simulation_flag_means_hardware(self):
        """When nidaqmx cannot say, the device is listed as hardware and DEBUG notes why.

        Args: None.
        Returns: None.
        """
        log = ListLog()
        self.hardware.fail["is_simulated"] = fake_nidaqmx.daq_error('driver too old')
        devices, reason = daq.list_devices(log)
        self.assertIsNone(reason)
        self.assertEqual([d.ni_simulated for d in devices], [False])
        self.assertEqual([level for level, _, _ in log.records], ["DEBUG"])
        self.assertTrue(log.has("could not read whether Dev1 is simulated"))
        self.hardware.fail["is_simulated"] = fake_nidaqmx.daq_error('driver too old')
        self.assertFalse(daq.read_ni_simulated(daq.load_ni(), "Dev1"))  # no log to write to

    def test_sine_and_held_level_reach_the_outputs(self):
        """The generator's sine and the piezo's level are generated together.

        Args: None.
        Returns: None.
        """
        p = self.apply(generator_on=True, generator_amplitude_v=0.5, piezo_on=True,
                       piezo_voltage_v=0.25)
        self.assertTrue(self.hardware.ao_running)
        self.assertEqual(self.hardware.ao_rate, p.rate)
        wave = self.hardware.ao["ao0"]
        self.assertEqual(len(wave) % p.samples_per_cycle, 0)
        self.assertAlmostEqual(wave.max(), 0.5, places=3)
        np.testing.assert_allclose(self.hardware.ao["ao1"], 0.25)

    def test_triangle_reaches_the_piezo_output(self):
        """The triangle is streamed on its own clock while a counter keeps running.

        Args: None.
        Returns: None.
        """
        p = self.apply(generator_output="ctr0", generator_on=True, piezo_on=True,
                       piezo_mode=daq.TRIANGLE, piezo_bottom_v=0.2, piezo_top_v=0.8,
                       piezo_frequency_hz=2.0)
        self.assertTrue(self.hardware.ao_running)
        self.assertEqual(self.hardware.ao_rate, p.rate)
        wave = self.hardware.ao["ao1"]
        self.assertEqual(len(wave) % p.samples_per_cycle, 0)
        self.assertAlmostEqual(wave.min(), 0.2)
        self.assertAlmostEqual(wave.max(), 0.8)

    def test_held_levels_stop_the_clock(self):
        """With only constant levels, the AO task runs on demand.

        Args: None.
        Returns: None.
        """
        self.apply(generator_on=True)
        self.apply(generator_on=False, piezo_on=True, piezo_voltage_v=0.3)
        self.assertFalse(self.hardware.ao_running)
        self.assertEqual(self.hardware.ao, {"ao0": 0.0, "ao1": 0.3})

    def test_reassigning_reconnects_at_zero(self):
        """A change of AO channels zeroes the old ones and reconnects.

        Args: None.
        Returns: None.
        """
        self.apply(piezo_on=True, piezo_voltage_v=0.4)
        self.apply(generator_output=daq.OFF, piezo_output="ao0", piezo_on=True,
                   piezo_voltage_v=0.1)
        self.assertEqual(self.backend.analog, ("ao0",))
        self.assertEqual(self.hardware.ao["ao1"], 0.0)
        self.assertEqual(self.hardware.ao["ao0"], 0.1)

    def test_square_wave_start_change_move_and_stop(self):
        """The counter starts, retunes, moves terminal and stops.

        Args: None.
        Returns: None.
        """
        self.apply(generator_output="ctr0", generator_terminal="pfi2", generator_on=True,
                   generator_frequency_hz=1000.0)
        counter = self.hardware.counters["ctr0"]
        self.assertEqual((counter["terminal"], counter["running"]), ("pfi2", True))
        self.assertAlmostEqual(counter["high"], 0.0005)
        self.apply(generator_output="ctr0", generator_terminal="pfi2", generator_on=True,
                   generator_frequency_hz=250.0)
        self.assertAlmostEqual(self.hardware.counters["ctr0"]["high"], 0.002)
        self.apply(generator_output="ctr0", generator_terminal="pfi5", generator_on=True)
        self.assertEqual(self.hardware.counters["ctr0"]["terminal"], "pfi5")
        self.assertEqual(self.hardware.tristated, ["/Dev1/pfi2"])
        self.apply(generator_output="ctr0", generator_terminal="pfi5", generator_on=False)
        self.assertFalse(self.hardware.counters["ctr0"]["running"])
        self.apply(generator_output="ctr0", generator_terminal="pfi5", generator_on=False)

    def test_zero_and_close(self):
        """Zero sets every output to 0 V; close does too, then releases.

        Args: None.
        Returns: None.
        """
        self.apply(generator_output="ctr1", generator_on=True, piezo_on=True,
                   piezo_voltage_v=0.3)
        self.backend.zero()
        self.assertEqual(self.hardware.ao, {"ao1": 0.0})
        self.assertFalse(self.hardware.counters["ctr1"]["running"])
        self.apply(piezo_on=True, piezo_voltage_v=0.3)
        self.backend.close()
        self.assertEqual(self.hardware.ao["ao1"], 0.0)
        self.assertIsNone(self.backend.nidaq)
        self.backend.zero()  # nothing open: no effect
        self.backend.close()

    def test_close_releases_even_when_zero_fails(self):
        """A failure while zeroing still closes the device.

        Args: None.
        Returns: None.
        """
        self.apply(piezo_on=True, piezo_voltage_v=0.3)
        self.hardware.fail["write"] = fake_nidaqmx.daq_error("Device removed", -50405)
        with self.assertRaises(Exception):
            self.backend.close()
        self.assertIsNone(self.backend.nidaq)

    def test_missing_device_raises_an_expected_error(self):
        """Opening an absent device fails with an error the reporter expects.

        Args: None.
        Returns: None.
        """
        with self.assertRaises(Exception) as caught:
            daq.PylablibBackend().open("Dev9")
        self.assertTrue(error_reports.is_expected(caught.exception, "daq"))


class ExpectedErrorTests(unittest.TestCase):
    """Which DAQ errors are operating conditions rather than bugs."""

    def test_daq_errors_are_expected(self):
        """Missing packages, NI errors and device errors are expected; bugs are not.

        Args: None.
        Returns: None.
        """
        self.assertTrue(error_reports.is_expected(ImportError("x"), "daq"))
        self.assertTrue(error_reports.is_expected(fake_nidaqmx.daq_error(), "daq"))
        self.assertTrue(error_reports.is_expected(DeviceError("x"), "daq"))
        self.assertFalse(error_reports.is_expected(ValueError("x"), "daq"))


class FailingBackend(daq.SimulatedBackend):
    """A simulated backend whose operations fail on request."""

    def __init__(self, fail_apply=None, fail_zero=None, fail_close=None):
        """Remember which operations fail.

        Args: fail_apply, fail_zero and fail_close are exceptions to raise.
        Returns: None.
        """
        super().__init__()
        self.fail_apply, self.fail_zero, self.fail_close = fail_apply, fail_zero, fail_close

    def apply(self, output_plan):
        """Fail, or record the plan.

        Args: output_plan is the OutputPlan.
        Returns: None.
        """
        if self.fail_apply:
            raise self.fail_apply
        super().apply(output_plan)

    def zero(self):
        """Fail, or record the zeroing.

        Args: None.
        Returns: None.
        """
        if self.fail_zero:
            raise self.fail_zero
        super().zero()

    def close(self):
        """Fail, or disconnect.

        Args: None.
        Returns: None.
        """
        if self.fail_close:
            raise self.fail_close
        super().close()


class ControllerTests(unittest.TestCase):
    """DaqController's worker thread, events and status."""

    def setUp(self):
        """Make a controller on simulated backends.

        Args: None.
        Returns: None.
        """
        self.log = ListLog()
        self.backends = []

        def factory(device):
            """Hand out the next prepared backend, or a simulated one.

            Args: device is ignored.
            Returns: A backend.
            """
            return self.backends.pop(0) if self.backends else daq.SimulatedBackend()

        self.controller = daq.DaqController(self.log, factory)
        self.addCleanup(self.controller.shutdown, 5.0)

    def connect(self):
        """Connect to the simulated device.

        Args: None.
        Returns: The DeviceOutputs.
        """
        self.controller.request("connect", device=daq.SIMULATED)
        return wait_event(self.controller, "connected")[0]

    def test_scan_reports_devices_and_reasons(self):
        """A scan lists devices and logs why none could be listed.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(daq, "list_devices", return_value=([], "No driver.")):
            self.controller.request("scan")
            devices, reason = wait_event(self.controller, "devices")[0]
        self.assertEqual((devices, reason), ([], "No driver."))
        self.assertTrue(self.log.has("NI DAQ: No driver."))
        with mock.patch.object(daq, "list_devices",
                               return_value=([daq.DeviceInfo("Dev1", "USB-6251")], None)):
            self.controller.request("scan")
            self.assertEqual(len(wait_event(self.controller, "devices")[0][0]), 1)

    def test_connect_apply_zero_and_close(self):
        """The normal sequence reports each step and keeps the status current.

        Args: None.
        Returns: None.
        """
        self.assertEqual(self.controller.status()["summary"], "Not connected")
        outputs = self.connect()
        self.assertTrue(outputs.device.simulated)
        settings = daq.DaqSettings(generator_on=True)
        self.controller.request("apply", settings=settings)
        applied = wait_event(self.controller, "applied")[0]
        self.assertEqual(applied.frequency_hz, 1000.0)
        self.assertTrue(self.controller.status()["active"])
        self.assertTrue(self.log.has("NI DAQ applied: SIMULATED, nothing driven"))
        self.controller.request("zero")
        wait_event(self.controller, "zeroed")
        self.assertFalse(self.controller.status()["active"])
        self.connect()  # reconnecting releases the previous device first
        self.assertTrue(self.controller.shutdown(5.0))
        wait_event(self.controller, "closed")
        self.assertIsNone(self.controller.status()["device"])

    def test_apply_needs_a_device_and_valid_settings(self):
        """Applying unconnected or invalid settings is refused, not reported.

        Args: None.
        Returns: None.
        """
        self.controller.request("apply", settings=daq.DaqSettings())
        self.assertEqual(wait_event(self.controller, "rejected")[0],
                         ["Connect to a device first."])
        self.controller.request("zero")
        wait_event(self.controller, "zeroed")
        self.connect()
        self.controller.request("apply", settings=daq.DaqSettings(piezo_voltage_v=9.0))
        self.assertIn("above its", wait_event(self.controller, "rejected")[0][0])

    def test_failure_reports_and_zeroes(self):
        """A device error is logged as expected, and the outputs set to 0 V.

        Args: None.
        Returns: None.
        """
        backend = FailingBackend(fail_apply=DeviceError("unplugged"))
        self.backends.append(backend)
        self.connect()
        self.controller.request("apply", settings=daq.DaqSettings(generator_on=True))
        message = wait_event(self.controller, "failed")[0]
        self.assertIn("unplugged", message)
        self.assertEqual(backend.zeroed, 1)
        self.assertTrue(self.log.has("NI DAQ apply failed: unplugged"))
        self.assertEqual(self.controller.status()["summary"], "Stopped after an error")

    def test_move_piezo_holds_a_level_without_events(self):
        """A phase step writes the level, records the settings and sends no event.

        Args: None.
        Returns: None.
        """
        with self.assertRaisesRegex(daq.PiezoMoveError, "not connected"):
            self.controller.move_piezo(daq.DaqSettings(), 0.1, 5.0)
        self.connect()
        base = daq.DaqSettings(piezo_voltage_v=0.0, generator_output=daq.OFF)
        written = self.controller.move_piezo(base, 0.25, 5.0)
        self.assertEqual(written.values["ao1"], 0.25)
        status = self.controller.status()
        self.assertTrue(status["active"])
        self.assertEqual((status["settings"].piezo_voltage_v, status["settings"].piezo_on,
                          status["settings"].piezo_mode), (0.25, True, daq.HOLD))
        self.assertTrue(self.controller.events.empty())
        self.assertTrue(self.log.has("NI DAQ phase step: piezo AO 1 holds 0.25 V"))
        with self.assertRaisesRegex(daq.PiezoMoveError, "above its"):
            self.controller.move_piezo(base, 5.0, 5.0)  # past the 1 V safety limit
        self.controller.request("zero")
        wait_event(self.controller, "zeroed")
        self.assertIsNone(self.controller.status()["settings"])

    def test_move_piezo_times_out_and_reports_device_failures(self):
        """No answer in time, or a failing device, raises PiezoMoveError.

        Args: None.
        Returns: None.
        """
        idle = daq.DaqController(self.log)
        with mock.patch.object(idle, "request"):  # the worker never answers
            with self.assertRaisesRegex(daq.PiezoMoveError, "within 0.05 s"):
                idle.move_piezo(daq.DaqSettings(), 0.1, 0.05)
        backend = FailingBackend(fail_apply=DeviceError("unplugged"))
        self.backends.append(backend)
        self.connect()
        with self.assertRaisesRegex(daq.PiezoMoveError, "unplugged"):
            self.controller.move_piezo(daq.DaqSettings(generator_output=daq.OFF), 0.1, 5.0)
        self.assertIn("unplugged", wait_event(self.controller, "failed")[0])
        self.assertEqual(backend.zeroed, 1)

    def test_unexpected_failure_is_reported(self):
        """A bug is logged at ERROR through the error reporter.

        Args: None.
        Returns: None.
        """
        self.backends.append(FailingBackend(fail_apply=ValueError("bug")))
        self.connect()
        with mock.patch.object(error_reports, "capture") as capture:
            self.controller.request("apply", settings=daq.DaqSettings())
            wait_event(self.controller, "failed")
        capture.assert_called_once()
        self.assertEqual(capture.call_args[0][1], "daq")

    def test_failure_with_a_dead_device_releases_it(self):
        """When zeroing also fails, the device is closed and forgotten.

        Args: None.
        Returns: None.
        """
        backend = FailingBackend(fail_apply=DeviceError("gone"), fail_zero=DeviceError("gone"),
                                 fail_close=DeviceError("gone"))
        self.backends.append(backend)
        self.connect()
        self.controller.request("apply", settings=daq.DaqSettings())
        wait_event(self.controller, "failed")
        self.assertTrue(self.log.has("could not set the outputs to 0 V"))
        self.assertTrue(self.log.has("could not release the device"))
        self.assertIsNone(self.controller.status()["device"])
        self.controller.request("apply", settings=daq.DaqSettings())
        wait_event(self.controller, "rejected")
        self.controller.request("zero")  # no device: still answers
        wait_event(self.controller, "zeroed")

    def test_failure_without_a_device_only_reports(self):
        """A failing scan leaves nothing to make safe.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(daq, "list_devices", side_effect=OSError("busy")):
            self.controller.request("scan")
            wait_event(self.controller, "failed")
        self.assertTrue(self.log.has("NI DAQ scan failed: busy"))

    def test_close_without_a_thread(self):
        """Closing before any work answers at once and starts no thread.

        Args: None.
        Returns: None.
        """
        controller = daq.DaqController(self.log)
        self.assertTrue(controller.shutdown(1.0))
        self.assertEqual(controller.events.get_nowait(), ("closed", None))
        self.assertIsNone(controller._thread)

    def test_default_backend_factory(self):
        """Without a factory, the simulated name gets the simulated backend.

        Args: None.
        Returns: None.
        """
        controller = daq.DaqController(self.log)
        self.addCleanup(controller.shutdown, 5.0)
        controller.request("connect", device=daq.SIMULATED)
        self.assertTrue(wait_event(controller, "connected")[0].device.simulated)
        self.assertEqual(controller.status()["device"].name, daq.SIMULATED)


if __name__ == "__main__":
    unittest.main()


class FakeClock:
    """A clock the tests move by hand."""

    def __init__(self):
        """Start at zero.

        Args: None.
        Returns: None.
        """
        self.now = 0.0

    def __call__(self):
        """Read the clock.

        Args: None.
        Returns: Seconds.
        """
        return self.now


class RealLikeBackend(daq.SimulatedBackend):
    """A simulated backend that reports itself as a real USB-6251 (it needs the checklist)."""

    simulated = False

    def open(self, device):
        """Pretend to connect to Dev1.

        Args: device is ignored.
        Returns: DeviceOutputs for a USB-6251 named Dev1.
        """
        self.outputs = daq.DeviceOutputs(daq.DeviceInfo("Dev1", "USB-6251"))
        return self.outputs


class SafetyRuleTests(unittest.TestCase):
    """The GUI-free pieces of the checklist and the readback check."""

    def test_checklist_follows_what_is_switched_on(self):
        """Only roles switched on get their checks; the watcher check is always asked.

        Args: None.
        Returns: None.
        """
        both = daq.DaqSettings(piezo_limit_v=10.0, piezo_on=True, generator_on=True)
        items = daq.checklist_items(both)
        self.assertEqual(len(items), 6)
        self.assertIn("OUTPUT ADJ knob", items[0])
        self.assertIn("AO 1 cable goes to the MDT694A's EXT INPUT", items[1])
        self.assertIn("about 150 V at the piezo", items[3])
        self.assertIn("AO 0 cable", items[4])
        counter = daq.checklist_items(daq.DaqSettings(generator_output="ctr0",
                                                      generator_terminal="pfi4",
                                                      generator_on=True))
        self.assertEqual(len(counter), 2)  # the piezo is assigned but off: no piezo checks
        self.assertIn("PFI 4 cable", counter[0])
        piezo_only = daq.checklist_items(daq.DaqSettings(piezo_on=True))
        self.assertEqual(len(piezo_only), 5)
        self.assertNotIn("generator", " ".join(piezo_only))
        alone = daq.checklist_items(daq.DaqSettings(piezo_output=daq.OFF,
                                                    generator_output=daq.OFF,
                                                    piezo_on=True, generator_on=True))
        self.assertEqual(len(alone), 1)
        self.assertIn("All outputs to 0 V", alone[0])

    def test_drives_and_driven_roles(self):
        """Only an assigned, switched-on role drives; the terminal counts for counters.

        Args: None.
        Returns: None.
        """
        self.assertFalse(daq.drives(daq.DaqSettings()))
        self.assertTrue(daq.drives(daq.DaqSettings(piezo_on=True)))
        self.assertTrue(daq.drives(daq.DaqSettings(generator_on=True)))
        self.assertFalse(daq.drives(daq.DaqSettings(piezo_on=True, piezo_output=daq.OFF)))
        self.assertEqual(daq.driven_roles(daq.DaqSettings()), frozenset())
        self.assertEqual(daq.driven_roles(daq.DaqSettings(piezo_on=True, generator_on=True,
                                                          generator_output="ctr1",
                                                          generator_terminal="pfi2")),
                         {(daq.PIEZO_ROLE, "ao1", ""), (daq.GENERATOR_ROLE, "ctr1", "pfi2")})

    def test_expected_levels_and_readback(self):
        """Readings within the written range pass; others name the channel and both values.

        Args: None.
        Returns: None.
        """
        plan = daq.plan(daq.DaqSettings(piezo_on=True, piezo_mode=daq.TRIANGLE,
                                        piezo_bottom_v=0.5, piezo_top_v=0.9))
        expected = daq.expected_levels(plan)
        self.assertEqual(expected["ao0"], (0.0, 0.0))
        self.assertAlmostEqual(expected["ao1"][1], 0.9)
        good = {"ao0": np.zeros(5), "ao1": np.linspace(0.5, 0.9, 5)}
        self.assertEqual(daq.readback_problem(expected, good), "")
        high = dict(good, ao1=np.full(5, 3.0))
        self.assertEqual(daq.readback_problem(expected, high),
                         "AO 1 read 3 V where 0.5–0.9 V was written")
        noisy = dict(good, ao0=np.array([-0.5, 0.2]))
        self.assertEqual(daq.readback_problem(expected, noisy),
                         "AO 0 read -0.5–0.2 V where 0 V was written")
        self.assertIn("could not be read back", daq.readback_problem(expected, {}))


class BackendReadbackTests(unittest.TestCase):
    """Zeroing everything, holding levels and reading the outputs back."""

    def test_simulated_backend_reads_what_it_holds(self):
        """The stand-in reads back its levels and waveforms, plus any injected shift.

        Args: None.
        Returns: None.
        """
        backend = daq.SimulatedBackend()
        backend.open(daq.SIMULATED)
        backend.apply(daq.plan(daq.DaqSettings(piezo_on=True, piezo_voltage_v=0.4)))
        readings = backend.read_outputs(["ao0", "ao1"])
        self.assertEqual(len(readings["ao1"]), daq.READBACK_SAMPLES)
        np.testing.assert_allclose(readings["ao1"], 0.4)
        backend.set_levels({"ao0": 0.5})
        backend.shift["ao0"] = 0.25
        np.testing.assert_allclose(backend.read_outputs(["ao0"])["ao0"], 0.75)
        backend.zero_all()
        self.assertEqual((backend.zeroed_all, backend.zeroed), (1, 0))
        np.testing.assert_allclose(backend.read_outputs(["ao1"])["ao1"], 0.0)

    def test_pylablib_backend_zeroes_everything_and_reads_back(self):
        """On the fake nidaqmx: both AOs to 0 V, levels held, internal channels read.

        Args: None.
        Returns: None.
        """
        hardware = fake_nidaqmx.install(self)
        backend = daq.PylablibBackend()
        backend.open("Dev1")
        hardware.ao = {"ao0": 3.0, "ao1": 7.0}  # left by a crash
        backend.zero_all()
        self.assertEqual(hardware.ao, {"ao0": 0.0, "ao1": 0.0})
        backend.set_levels({"ao1": 0.5})
        hardware.readback_shift["ao1"] = 0.01
        readings = backend.read_outputs(["ao0", "ao1"])
        np.testing.assert_allclose(readings["ao0"], 0.0)
        np.testing.assert_allclose(readings["ao1"], 0.51)
        self.assertEqual(len(backend.read_outputs(["ao1"])["ao1"]), daq.READBACK_SAMPLES)
        self.assertEqual(hardware.reads, 2)
        # A waveform reads back within its range; a running counter is stopped.
        p = daq.plan(daq.DaqSettings(device="Dev1", generator_output="ctr0",
                                     generator_on=True, piezo_on=True,
                                     piezo_mode=daq.TRIANGLE, piezo_bottom_v=0.2,
                                     piezo_top_v=0.8), backend.outputs)
        backend.apply(p)
        wave = backend.read_outputs(["ao1"])["ao1"]
        self.assertGreaterEqual(wave.min(), 0.2)
        self.assertLessEqual(wave.max(), 0.8)
        backend.zero_all()
        self.assertFalse(hardware.counters["ctr0"]["running"])
        self.assertEqual(hardware.ao, {"ao0": 0.0, "ao1": 0.0})


class SelfCheckTests(unittest.TestCase):
    """The controller's checklist gate, readback, watchdog and self-test."""

    def setUp(self):
        """Make a controller with a hand-moved clock.

        Args: None.
        Returns: None.
        """
        self.log = ListLog()
        self.clock = FakeClock()
        self.backend = RealLikeBackend()
        self.controller = daq.DaqController(self.log, lambda device: self.backend,
                                            clock=self.clock)
        self.addCleanup(self.controller.shutdown, 5.0)

    def connect(self):
        """Connect to the real-like device.

        Args: None.
        Returns: The DeviceOutputs.
        """
        self.controller.request("connect", device="Dev1")
        return wait_event(self.controller, "connected")[0]

    def test_connecting_sets_every_output_to_zero(self):
        """Whatever was left on, a connection starts with every AO written to 0 V.

        Args: None.
        Returns: None.
        """
        self.connect()
        self.assertEqual(self.backend.zeroed_all, 1)

    def test_real_device_needs_the_checklist(self):
        """No drive signal before arm(); a reassignment or reconnection asks again.

        Args: None.
        Returns: None.
        """
        settings = daq.DaqSettings(piezo_on=True, piezo_voltage_v=0.2)
        self.assertFalse(self.controller.is_armed(settings))  # no device yet
        self.assertFalse(self.controller.checklist_done())
        self.connect()
        self.controller.request("apply", settings=settings)
        self.assertEqual(wait_event(self.controller, "rejected")[0], [daq.CHECKLIST_REQUIRED])
        with self.assertRaisesRegex(daq.PiezoMoveError, "safety checklist"):
            self.controller.move_piezo(settings, 0.3, 5.0)
        self.controller.request("apply", settings=daq.DaqSettings())  # nothing on: fine
        wait_event(self.controller, "applied")
        self.controller.arm(settings)
        self.assertTrue(self.controller.is_armed(settings))
        self.controller.request("apply", settings=settings)
        wait_event(self.controller, "applied")
        self.controller.move_piezo(settings, 0.3, 5.0)
        moved = daq.replace(settings, piezo_output="ao0", generator_output=daq.OFF)
        self.assertFalse(self.controller.is_armed(moved))  # another socket: ask again
        both = daq.replace(settings, generator_on=True)
        self.assertFalse(self.controller.is_armed(both))  # the generator was not covered
        self.controller.arm(both)
        self.assertTrue(self.controller.is_armed(both))
        self.assertTrue(self.controller.is_armed(settings))  # earlier ticks still count
        self.assertTrue(self.controller.checklist_done())
        self.connect()
        self.assertFalse(self.controller.is_armed(settings))
        self.assertFalse(self.controller.checklist_done())

    def test_ni_max_simulated_device_still_needs_the_checklist(self):
        """A device created as simulated in NI MAX is gated like hardware.

        Only the app's own SIMULATED stand-in skips the checklist.

        Args: None.
        Returns: None.
        """
        fake_nidaqmx.install(self, devices=("SimDev1",), simulated=("SimDev1",))
        controller = daq.DaqController(self.log)
        self.addCleanup(controller.shutdown, 5.0)
        controller.request("connect", device="SimDev1")
        outputs = wait_event(controller, "connected")[0]
        self.assertTrue(outputs.device.ni_simulated)
        self.assertFalse(outputs.device.simulated)
        self.assertTrue(controller.status()["device"].ni_simulated)
        settings = daq.DaqSettings(device="SimDev1", piezo_on=True, piezo_voltage_v=0.2)
        self.assertFalse(controller.is_armed(settings))
        controller.request("apply", settings=settings)
        self.assertEqual(wait_event(controller, "rejected")[0], [daq.CHECKLIST_REQUIRED])
        with self.assertRaisesRegex(daq.PiezoMoveError, "safety checklist"):
            controller.move_piezo(settings, 0.3, 5.0)
        controller.arm(settings)
        self.assertTrue(controller.is_armed(settings))
        controller.request("apply", settings=settings)
        wait_event(controller, "applied")

    def test_readback_fault_sets_everything_to_zero_once(self):
        """An output that reads back wrong stops everything, and is reported once.

        Args: None.
        Returns: None.
        """
        self.connect()
        settings = daq.DaqSettings(piezo_on=True, piezo_voltage_v=0.2)
        self.controller.arm(settings)
        self.controller.request("apply", settings=settings)
        wait_event(self.controller, "applied")
        self.controller.request("check")  # all well: nothing happens
        self.backend.shift["ao1"] = 1.0
        self.controller.request("check")
        reason = wait_event(self.controller, "fault")[0]
        self.assertEqual(reason, "AO 1 read 1.2 V where 0.2 V was written")
        self.assertEqual(self.backend.zeroed_all, 2)
        status = self.controller.status()
        self.assertFalse(status["active"])
        self.assertIn("Stopped by the self-check", status["summary"])
        self.assertTrue(self.log.has("NI DAQ self-check: AO 1 read"))
        self.controller.request("check")  # still shifted, but reported only once
        self.controller.request("scan")
        with mock.patch.object(daq, "list_devices", return_value=([], None)):
            _, seen = wait_event(self.controller, "devices")
        self.assertNotIn("fault", [kind for kind, _ in seen])

    def test_watchdog_needs_a_heartbeat_and_an_active_output(self):
        """A silent window stops active outputs; without heartbeats there is no watchdog.

        Args: None.
        Returns: None.
        """
        self.connect()
        settings = daq.DaqSettings(generator_on=True)
        self.controller.arm(settings)
        self.controller.request("apply", settings=settings)
        wait_event(self.controller, "applied")
        self.clock.now = 10.0
        self.controller.request("check")  # never had a heartbeat: no watchdog
        self.controller.heartbeat()
        self.clock.now = 11.0
        self.controller.request("check")  # 1 s quiet: fine
        self.clock.now = 13.5
        self.controller.request("check")
        reason = wait_event(self.controller, "fault")[0]
        self.assertEqual(reason, "the window stopped responding for 3.5 s")

    def test_failing_check_is_reported_and_zeroes(self):
        """A device error during the readback is reported like any other.

        Args: None.
        Returns: None.
        """
        self.connect()
        with mock.patch.object(self.backend, "read_outputs",
                               side_effect=fake_nidaqmx.daq_error("gone", -50405)):
            self.controller.request("check")
            self.assertIn("gone", wait_event(self.controller, "failed")[0])
        self.assertTrue(self.log.has("NI DAQ self-check failed"))

    def test_self_test(self):
        """Each output is stepped and read back; a wrong reading fails the test.

        Args: None.
        Returns: None.
        """
        self.controller.request("self_test")
        self.assertEqual(wait_event(self.controller, "self_test")[0],
                         (False, ["Connect to a device first."]))
        self.connect()
        self.controller.request("self_test")
        passed, lines = wait_event(self.controller, "self_test")[0]
        self.assertTrue(passed)
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("AO 0 at 0.5 V: OK"))
        self.assertTrue(lines[2].startswith("every output at 0 V: OK"))
        self.assertEqual(self.controller.status()["summary"],
                         "Self-test passed; all outputs at 0 V")
        self.backend.shift["ao1"] = 0.3
        self.controller.request("self_test")
        passed, lines = wait_event(self.controller, "self_test")[0]
        self.assertFalse(passed)
        self.assertIn("FAILED, AO 1 read", lines[0])
        self.assertTrue(self.log.has("NI DAQ self-test FAILED"))
