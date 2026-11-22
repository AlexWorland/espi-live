"""NI DAQ outputs: a frequency generator and a piezo driver on one NI device.

The frequency generator excites the specimen (a sine on an analog output, or
a square TTL wave from a counter on a PFI terminal). The piezo driver sets the
reference piezo's voltage: a held level for manual phase steps, a sine locked
to the generator's frequency and phase, or a triangle at its own frequency
(the slow ramp a function generator used to feed the piezo amplifier). Each
role is assigned to its own output, and every voltage is clamped to a limit
the user sets for the amplifier behind it. The piezo driver never goes below
0 V: the piezo amplifier (a Thorlabs MDT694A) takes 0-10 V on its EXT INPUT,
and a negative input would drive the piezo below 0 V.

pyLabLib's NIDAQ class does the device work on top of NI's ``nidaqmx``
package; both are imported lazily, so the program runs without them. The
NI-DAQmx driver itself is a separate NI install (Windows and Linux only).
A simulated device, clearly named as such, stands in when no hardware is
available. This module is GUI-free and camera-free: DaqController runs every
device call on its own worker thread, and the front end talks to it through
request(), status() and the events queue, like the camera Engine.

See docs/notes/ni-usb-6251-integration.md for the hardware research.
"""

import importlib
import math
import queue
import threading
import time
from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np

from espi_live import error_reports

AO_OUTPUTS = ("ao0", "ao1")
COUNTER_OUTPUTS = ("ctr0", "ctr1")
OFF = ""  # an unassigned role
PFI_TERMINALS = tuple(f"pfi{index}" for index in range(16))
SIMULATED = "Simulated"  # device name of the stand-in that drives no hardware
HOLD, SINE, TRIANGLE = "hold", "sine", "triangle"  # piezo modes
PIEZO_MODES = (HOLD, SINE, TRIANGLE)
PIEZO_FLOOR_V = 0.0  # the piezo amplifier's input takes 0-10 V: never below this

TIMEBASE_HZ = 20e6  # AO sample clock timebase; the rate is this divided by a whole number
COUNTER_TIMEBASE_HZ = 80e6  # fastest counter timebase (estimates the square wave's frequency)
# USB-6251 AO update limits: one channel, two channels (per channel).
MAX_AO_RATE = {1: TIMEBASE_HZ / 7, 2: 2_000_000.0}
MIN_SAMPLES_PER_CYCLE = 16
MAX_SAMPLES_PER_CYCLE = 2000
PREFERRED_MAX_RATE = 250_000.0  # S/s; higher only when a cycle would get under 64 samples
PREFERRED_SAMPLES = 64
CLOSE_ENOUGH = 1e-4  # relative frequency error accepted (the clock itself is 50 ppm)
MIN_FREQUENCY_HZ = 0.1
MAX_SINE_HZ = 100_000.0
MAX_SQUARE_HZ = 1_000_000.0
MIN_TRIANGLE_HZ = 0.01
MAX_TRIANGLE_HZ = 1000.0
MAX_VOLTS = 10.0  # the AO range
MIN_WRITE_S = 0.001  # pyLabLib writes at least this much waveform per start

# The USB-6251 BNC's sockets, as printed on its panel. PFI 8-15, P0.0-7, +5 V
# and D GND are on its screw terminals only.
BNC_GROUPS = (("ANALOG INPUTS", tuple(f"ai{index}" for index in range(8))),
              ("ANALOG OUTPUTS", ("ao0", "ao1")),
              ("APFI", ("apfi0",)),
              ("PFI (DIGITAL / TIMING)", tuple(f"pfi{index}" for index in range(8))),
              ("USER", ("user1", "user2")))
BNC_TERMINALS = tuple(name for _, names in BNC_GROUPS for name in names)
PIEZO_AMPLIFIER = "Thorlabs MDT694A EXT INPUT"
PIEZO_ROLE, GENERATOR_ROLE = "Piezo driver", "Frequency generator"

# Self-checks. The USB-6251 measures its own analog outputs through the
# internal channels "_ao0_vs_aognd" and "_ao1_vs_aognd", with no wiring.
READBACK_SAMPLES = 50
READBACK_RATE = 10_000.0  # S/s: 5 ms per check
READBACK_TOLERANCE_V = 0.1  # how far a reading may be from what was written
CHECK_INTERVAL_S = 1.0  # readback and watchdog period while connected
WATCHDOG_S = 2.0  # outputs go to 0 V when the window has not polled for this long
SELF_TEST_V = 0.5  # the level each output is stepped to by the self-test
CHECKLIST_REQUIRED = ("Tick off the safety checklist in the NI DAQ window first: no drive "
                      "signal goes to this device until it is done for this connection.")

INSTALL_HINT = 'pip install pylablib-lightweight "nidaqmx>=1.4,<1.8"'
_NI = None  # pylablib.devices.NI once loaded


class PiezoMoveError(RuntimeError):
    """The piezo could not be moved to a requested level."""


def load_ni():
    """Import pyLabLib's NI module (and through it nidaqmx) on first use.

    Args: None.
    Returns: The pylablib.devices.NI module. Raises ImportError when pyLabLib
        or nidaqmx is not installed.
    """
    global _NI
    if _NI is None:
        module = importlib.import_module("pylablib.devices.NI")
        if getattr(module.daq, "nidaqmx", None) is None:
            raise ImportError("nidaqmx is not installed")
        _NI = module
    return _NI


def support_text():
    """Say whether NI DAQ support is installed, for --check.

    Imports pyLabLib's NI module (so a build missing it is caught) but does
    not cache it and lists no devices.

    Args: None.
    Returns: One line of text.
    """
    try:
        module = importlib.import_module("pylablib.devices.NI")
        ready = getattr(module.daq, "nidaqmx", None) is not None
    except Exception:  # missing or broken install: reported, not raised
        ready = False
    return "NI DAQ: pyLabLib and nidaqmx " + ("available" if ready else "not installed")


def unavailable_reason(exc):
    """Explain in one sentence why NI devices cannot be used.

    Args: exc is the exception raised while loading or listing devices.
    Returns: Text for the user, naming what to install where it helps.
    """
    name = type(exc).__name__
    if isinstance(exc, ImportError):
        return f"NI DAQ support is not installed ({INSTALL_HINT})."
    if name == "DaqNotFoundError":
        return "The NI-DAQmx driver is not installed. Install it from ni.com."
    if name == "DaqNotSupportedError":
        return "NI-DAQmx runs on Windows and Linux only."
    return f"NI devices could not be listed: {exc}"


@dataclass(frozen=True)
class Connection:
    """One cable to plug in: a DAQ socket, the role it serves and where it goes."""

    terminal: str  # "ao1", "pfi4", ...
    role: str  # PIEZO_ROLE or GENERATOR_ROLE
    target: str  # what the other end of the cable plugs into
    note: str = ""

    @property
    def bnc(self):
        """Tell whether the socket is a BNC on the USB-6251 BNC.

        Args: None.
        Returns: True for a BNC, False for a screw terminal or another device's output.
        """
        return self.terminal in BNC_TERMINALS

    @property
    def label(self):
        """Name the socket as printed on the device.

        Args: None.
        Returns: Text such as "AO 1", "PFI 4" or "USER 1".
        """
        return socket_label(self.terminal)


def socket_label(name):
    """Name any socket as printed on the device: letters, a space, the number.

    Args: name is a socket such as "ai0", "apfi0", "user2" or "ctr1".
    Returns: Text such as "AI 0", "APFI 0", "USER 2"; counters as output_label
        names them ("Counter 1").
    """
    if name.startswith("ctr"):
        return output_label(name)
    stem = name.rstrip("0123456789")
    return f"{stem.upper()} {name[len(stem):]}".strip()


def wiring(settings):
    """List the cables the settings need, for the "Where to plug in" window.

    Args: settings is DaqSettings.
    Returns: A list of Connection, the piezo driver first; empty when no
        role is assigned.
    """
    s = settings
    found = []
    if s.piezo_output != OFF:
        found.append(Connection(
            s.piezo_output, PIEZO_ROLE, PIEZO_AMPLIFIER,
            f"0–{MAX_VOLTS:g} V in; the amplifier adds its OUTPUT ADJ knob, so turn "
            "the knob to zero. Its OUTPUT stays on the piezo."))
    if s.generator_output.startswith("ctr"):
        note = "0–5 V square wave (TTL)."
        if s.generator_terminal not in BNC_TERMINALS:
            note += (f" {terminal_label(s.generator_terminal)} is a screw terminal: "
                     "choose PFI 0–7 for a BNC socket.")
        found.append(Connection(s.generator_terminal, GENERATOR_ROLE,
                                "the specimen drive's input (shaker or tone amplifier)",
                                note))
    elif s.generator_output != OFF:
        found.append(Connection(s.generator_output, GENERATOR_ROLE,
                                "the specimen drive's input (shaker or tone amplifier)",
                                f"Sine, at most the {s.generator_limit_v:g} V safety limit."))
    return found


MDT694A_GAIN = 15.0  # V/V at its 150 V limit


def checklist_items(settings):
    """List what the user must confirm before drive signals reach a real device.

    Args: settings is DaqSettings: only the roles it switches on get their
        checks (an output that stays off is held at 0 V).
    Returns: A list of sentences, one checkbox each; the last is always asked.
    """
    s = settings
    items = []
    if s.piezo_on and s.piezo_output != OFF:
        piezo = socket_label(s.piezo_output)
        items += [
            "The MDT694A's OUTPUT ADJ knob is turned to zero (it adds to the DAQ's "
            "voltage).",
            f"Only the DAQ's {piezo} cable goes to the MDT694A's EXT INPUT: the Wavetek "
            "is unplugged from it.",
            "The MDT694A's OUTPUT goes only to the piezo, never to the DAQ (it reaches "
            "150 V).",
            f"The piezo's safety limit here ({s.piezo_limit_v:g} V at the DAQ, about "
            f"{s.piezo_limit_v * MDT694A_GAIN:.0f} V at the piezo) and the MDT694A's "
            "VOLT LIMIT suit this piezo.",
        ]
    if s.generator_on and s.generator_output != OFF:
        where = socket_label(s.generator_terminal if s.generator_output.startswith("ctr")
                             else s.generator_output)
        items.append(f"The generator's {where} cable goes to the specimen drive's input, "
                     f"which accepts its {s.generator_limit_v:g} V safety limit.")
    items.append("Someone is watching the setup and knows that “All outputs to 0 V” "
                 "stops everything.")
    return items


def output_label(output):
    """Name an output the way the device's front panel does.

    Args: output is "ao0", "ctr1" and so on, or OFF.
    Returns: Text such as "AO 0" or "Counter 1"; "Off" for OFF.
    """
    if output == OFF:
        return "Off"
    if output.startswith("ao"):
        return f"AO {output[2:]}"
    if output.startswith("ctr"):
        return f"Counter {output[3:]}"
    return output.upper()


def terminal_label(terminal):
    """Name a PFI terminal as printed on the device.

    Args: terminal is a name such as "pfi4".
    Returns: Text such as "PFI 4".
    """
    return f"PFI {terminal[3:]}" if terminal.startswith("pfi") else terminal.upper()


@dataclass(frozen=True)
class DeviceInfo:
    """One NI device: its DAQmx name, model and serial number."""

    name: str
    model: str
    serial: str = ""

    @property
    def simulated(self):
        """Tell whether this is the stand-in that drives no hardware.

        Args: None.
        Returns: True for the simulated device.
        """
        return self.name == SIMULATED

    @property
    def label(self):
        """Describe the device for a list.

        Args: None.
        Returns: Text such as "Dev1: USB-6251 (01A2B3C4)".
        """
        if self.simulated:
            return "Simulated device (no hardware)"
        serial = f" ({self.serial})" if self.serial else ""
        return f"{self.name}: {self.model}{serial}"


SIMULATED_DEVICE = DeviceInfo(SIMULATED, "Simulated NI DAQ")


@dataclass(frozen=True)
class DeviceOutputs:
    """What a connected device offers: analog outputs, counters and PFI lines."""

    device: DeviceInfo
    analog: tuple = AO_OUTPUTS
    counters: tuple = COUNTER_OUTPUTS
    terminals: tuple = PFI_TERMINALS


@dataclass
class DaqSettings:
    """Output assignments and levels for the frequency generator and the piezo driver.

    Voltages are at the DAQ terminal, before any amplifier. An assigned
    output that is switched off is held at 0 V.
    """

    device: str = ""  # DAQmx name ("Dev1"), SIMULATED, or "" for the first found
    generator_output: str = "ao0"  # an AO (sine) or a counter (square TTL), or OFF
    generator_terminal: str = "pfi4"  # PFI line for a counter output
    generator_on: bool = False
    generator_frequency_hz: float = 1000.0
    generator_amplitude_v: float = 0.1  # peak
    generator_offset_v: float = 0.0
    generator_limit_v: float = 1.0  # largest |V| the amplifier may receive
    piezo_output: str = "ao1"  # an AO, or OFF
    piezo_on: bool = False
    piezo_mode: str = HOLD
    piezo_voltage_v: float = 0.0  # the held level, or the sine's centre
    piezo_amplitude_v: float = 0.0  # sine peak
    piezo_phase_deg: float = 0.0  # sine phase relative to the generator
    piezo_bottom_v: float = 0.0  # triangle's lowest level
    piezo_top_v: float = 1.0  # triangle's highest level
    piezo_frequency_hz: float = 0.25  # triangle frequency, on its own clock
    piezo_step_v: float = 0.05  # what one Step button press adds
    piezo_limit_v: float = 1.0

    def to_dict(self):
        """Describe the settings for storage.

        Args: None.
        Returns: A dict of plain values.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Rebuild settings from storage, ignoring unknown or mistyped values.

        Args: data is a dict as to_dict() wrote it (possibly from an older
            or newer build), or anything else.
        Returns: DaqSettings; values that do not fit keep their defaults.
        """
        settings = cls()
        if not isinstance(data, dict):
            return settings
        for item in fields(cls):
            value = data.get(item.name)
            default = getattr(settings, item.name)
            if isinstance(default, bool):
                if isinstance(value, bool):
                    settings = replace(settings, **{item.name: value})
            elif isinstance(default, float):
                if isinstance(value, (int, float)) and not isinstance(value, bool) \
                        and math.isfinite(value):
                    settings = replace(settings, **{item.name: float(value)})
            elif isinstance(value, str):
                settings = replace(settings, **{item.name: value})
        return settings


@dataclass(frozen=True)
class CounterPlan:
    """A continuous square wave from one counter on one PFI terminal."""

    counter: str
    terminal: str
    high_s: float
    low_s: float


@dataclass(frozen=True)
class Drive:
    """What the outputs produce, for the simulated ESPI signal in dummy mode.

    The generator's waveform is "sine", "square" or "" (off or unassigned),
    at the frequency the device really produces; the piezo's mode is HOLD,
    SINE, TRIANGLE or "" (off or unassigned), with piezo_v its held level or
    its sine's or triangle's centre, piezo_amplitude_v the swing either side
    of it, and piezo_frequency_hz the triangle's real frequency. device is
    the label of the device that applied it. Drive() is "nothing driven".
    """

    waveform: str = ""
    frequency_hz: float = 0.0
    amplitude_v: float = 0.0
    offset_v: float = 0.0
    piezo_mode: str = ""
    piezo_v: float = 0.0
    piezo_amplitude_v: float = 0.0
    piezo_phase_deg: float = 0.0
    device: str = ""
    simulated: bool = False
    piezo_frequency_hz: float = 0.0

    @property
    def active(self):
        """Tell whether the generator or the piezo is producing anything.

        Args: None.
        Returns: True when either is on.
        """
        return bool(self.waveform or self.piezo_mode)


@dataclass(frozen=True)
class OutputPlan:
    """Exactly what to write to the device, worked out from DaqSettings."""

    analog: tuple = ()  # AO channels in use, in order
    rate: float = 0.0  # AO sample rate; 0 holds constant levels
    samples_per_cycle: int = 0
    buffer_samples: int = 0  # per channel, a whole number of cycles
    values: dict = field(default_factory=dict)  # channel -> float or 1-D array
    counter: CounterPlan = None
    frequency_hz: float = 0.0  # what the generator really produces (0 when off)
    piezo_frequency_hz: float = 0.0  # what the piezo triangle really produces (0 otherwise)
    summary: str = ""
    drive: Drive = Drive()  # the same, described for the ESPI simulation


def choose_timing(frequency, channels, max_rate=None):
    """Pick an AO sample rate and samples per cycle that land on a frequency.

    The rate is the 20 MHz timebase divided by a whole number, and each cycle
    is a whole number of samples, so the frequency is quantised. The most
    samples per cycle whose frequency error is within 0.01 % wins (the
    clock itself is accurate to 50 ppm); otherwise the smallest error.

    The rate stays at or below 250 kS/s, rising only as far as needed for 64
    samples per cycle at high frequencies: the waveform is streamed over USB,
    which the camera shares.

    Args: frequency is the wanted sine frequency in Hz; channels is how many
        AO channels share the clock (1 or 2); max_rate overrides the device's
        highest per-channel update rate.
    Returns: (rate in S/s, samples per cycle, actual frequency in Hz).
    """
    device_limit = max_rate or MAX_AO_RATE[min(max(channels, 1), 2)]
    limit = min(device_limit, max(PREFERRED_MAX_RATE, PREFERRED_SAMPLES * frequency))
    divisor_min = math.ceil(TIMEBASE_HZ / limit - 1e-9)
    best = None
    for samples in range(MAX_SAMPLES_PER_CYCLE, MIN_SAMPLES_PER_CYCLE - 1, -1):
        divisor = max(divisor_min, round(TIMEBASE_HZ / (frequency * samples)))
        actual = TIMEBASE_HZ / (divisor * samples)
        error = abs(actual - frequency) / frequency
        if error <= CLOSE_ENOUGH:
            return TIMEBASE_HZ / divisor, samples, actual
        if best is None or error < best[0]:
            best = (error, divisor, samples, actual)
    _, divisor, samples, actual = best
    return TIMEBASE_HZ / divisor, samples, actual


def square_frequency(frequency):
    """Estimate the frequency a counter really produces for a square wave.

    Args: frequency is the wanted frequency in Hz.
    Returns: The frequency with each half period rounded to whole ticks of
        the 80 MHz timebase (NI-DAQmx picks the timebase; this is close).
    """
    ticks = max(2, round(COUNTER_TIMEBASE_HZ / (2 * frequency)))
    return COUNTER_TIMEBASE_HZ / (2 * ticks)


def triangle_wave(bottom, top, samples):
    """Build one cycle of a symmetric triangle that starts at its bottom.

    Starting at the bottom means the first sample after a held level is the
    lowest one, never a jump to the top.

    Args: bottom and top are the lowest and highest volts; samples is the
        number of samples in the cycle.
    Returns: A 1-D float array of samples: bottom, rising to top half-way,
        falling back.
    """
    position = np.arange(samples) / samples
    return bottom + (top - bottom) * (1.0 - np.abs(2.0 * position - 1.0))


def piezo_range(settings):
    """Work out the lowest and highest voltage the piezo driver produces.

    Args: settings is DaqSettings (an unknown mode counts as a held level).
    Returns: (lowest, highest) in volts at the DAQ.
    """
    s = settings
    if s.piezo_mode == SINE:
        swing = abs(s.piezo_amplitude_v)
        return s.piezo_voltage_v - swing, s.piezo_voltage_v + swing
    if s.piezo_mode == TRIANGLE:
        return min(s.piezo_bottom_v, s.piezo_top_v), max(s.piezo_bottom_v, s.piezo_top_v)
    return s.piezo_voltage_v, s.piezo_voltage_v


def drives(settings):
    """Tell whether settings send any drive signal.

    Args: settings is DaqSettings.
    Returns: True when the generator or the piezo is switched on and assigned.
    """
    s = settings
    return (s.generator_on and s.generator_output != OFF) or (
        s.piezo_on and s.piezo_output != OFF)


def driven_roles(settings):
    """Name each role the settings switch on, with the socket it drives.

    Args: settings is DaqSettings.
    Returns: A frozenset of (PIEZO_ROLE, AO channel) and (GENERATOR_ROLE,
        output, PFI terminal or ""); the safety checklist must cover each one.
    """
    s = settings
    roles = set()
    if s.piezo_on and s.piezo_output != OFF:
        roles.add((PIEZO_ROLE, s.piezo_output, ""))
    if s.generator_on and s.generator_output != OFF:
        terminal = s.generator_terminal if s.generator_output.startswith("ctr") else ""
        roles.add((GENERATOR_ROLE, s.generator_output, terminal))
    return frozenset(roles)


def expected_levels(output_plan):
    """Work out the range each written AO channel may read back in.

    Args: output_plan is the OutputPlan written.
    Returns: A dict from channel to (lowest, highest) volts.
    """
    found = {}
    for channel in output_plan.analog:
        values = np.asarray(output_plan.values.get(channel, 0.0), dtype=float)
        found[channel] = (float(values.min()), float(values.max()))
    return found


def readback_problem(expected, readings):
    """Compare readings of the outputs with what was written.

    Args: expected maps a channel to (lowest, highest) volts; readings maps
        a channel to its read samples.
    Returns: A sentence describing the first channel out of range, or "".
    """
    for channel, (low, high) in expected.items():
        samples = np.asarray(readings.get(channel, ()), dtype=float)
        if not samples.size:
            return f"{output_label(channel)} could not be read back"
        lowest, highest = float(samples.min()), float(samples.max())
        if lowest < low - READBACK_TOLERANCE_V or highest > high + READBACK_TOLERANCE_V:
            wanted = f"{low:.3g} V" if low == high else f"{low:.3g}–{high:.3g} V"
            seen = f"{lowest:.3g} V" if abs(highest - lowest) < 1e-3 \
                else f"{lowest:.3g}–{highest:.3g} V"
            return f"{output_label(channel)} read {seen} where {wanted} was written"
    return ""


def _buffer_samples(rate, samples):
    """Size an AO buffer: whole cycles, at least MIN_WRITE_S of waveform.

    Args: rate is the sample rate; samples is the number per cycle.
    Returns: Samples per channel.
    """
    return math.ceil(max(rate * MIN_WRITE_S, samples) / samples) * samples


def problems(settings, outputs=None):
    """List what is wrong with the settings, in words for the user.

    Args: settings is DaqSettings; outputs is the connected device's
        DeviceOutputs, or None to check against the USB-6251's outputs.
    Returns: A list of sentences; empty when the settings can be applied.
    """
    outputs = outputs or DeviceOutputs(SIMULATED_DEVICE)
    s = settings
    found = []
    generator_outputs = (OFF,) + outputs.analog + outputs.counters
    if s.generator_output not in generator_outputs:
        found.append(f"This device has no {output_label(s.generator_output)} "
                     "for the frequency generator.")
    if s.piezo_output not in (OFF,) + outputs.analog:
        found.append(f"The piezo driver needs an analog output; "
                     f"{output_label(s.piezo_output)} is not one on this device.")
    if s.piezo_output != OFF and s.piezo_output == s.generator_output:
        found.append(f"The frequency generator and the piezo driver are both on "
                     f"{output_label(s.piezo_output)}. Give each its own output.")
    for name, limit in (("frequency generator", s.generator_limit_v),
                        ("piezo driver", s.piezo_limit_v)):
        if not 0 < limit <= MAX_VOLTS:
            found.append(f"The {name}'s safety limit must be above 0 V and at most "
                         f"{MAX_VOLTS:g} V.")
    counter = s.generator_output in outputs.counters
    if counter and s.generator_terminal not in outputs.terminals:
        found.append(f"Choose a PFI terminal for {output_label(s.generator_output)}.")
    top = MAX_SQUARE_HZ if counter else MAX_SINE_HZ
    if s.generator_output != OFF and not MIN_FREQUENCY_HZ <= s.generator_frequency_hz <= top:
        found.append(f"The frequency must be between {MIN_FREQUENCY_HZ:g} Hz and "
                     f"{top:,.0f} Hz.")
    if s.generator_output in outputs.analog:
        peak = abs(s.generator_offset_v) + abs(s.generator_amplitude_v)
        if peak > s.generator_limit_v:
            found.append(f"The generator would reach {peak:g} V, above its "
                         f"{s.generator_limit_v:g} V limit.")
    if s.piezo_output != OFF:
        lowest, highest = piezo_range(s)
        peak = max(abs(lowest), abs(highest))
        if peak > s.piezo_limit_v:
            found.append(f"The piezo driver would reach {peak:g} V, above its "
                         f"{s.piezo_limit_v:g} V limit.")
        if lowest < PIEZO_FLOOR_V:
            found.append(f"The piezo driver would go down to {lowest:g} V. The piezo "
                         f"amplifier's input takes 0–{MAX_VOLTS:g} V (a negative input "
                         "drives the piezo below 0 V): keep every level at 0 V or above.")
        found.extend(_piezo_mode_problems(s, outputs))
    return found


def _piezo_mode_problems(settings, outputs):
    """Check the rules of the piezo driver's mode.

    Args: settings is DaqSettings with the piezo assigned; outputs is the
        device's DeviceOutputs.
    Returns: A list of sentences; empty when the mode can run.
    """
    s = settings
    sine_generator = s.generator_on and s.generator_output in outputs.analog
    if s.piezo_mode not in PIEZO_MODES:
        return [f"Unknown piezo mode {s.piezo_mode!r}."]
    if s.piezo_mode == SINE and s.piezo_on and not sine_generator:
        return ["A piezo sine follows the frequency generator: put the "
                "generator on an analog output and switch it on."]
    if s.piezo_mode != TRIANGLE:
        return []
    found = []
    if s.piezo_top_v <= s.piezo_bottom_v:
        found.append("The piezo triangle's top must be above its bottom.")
    if not MIN_TRIANGLE_HZ <= s.piezo_frequency_hz <= MAX_TRIANGLE_HZ:
        found.append(f"The piezo triangle's frequency must be between "
                     f"{MIN_TRIANGLE_HZ:g} Hz and {MAX_TRIANGLE_HZ:,.0f} Hz.")
    if s.piezo_on and sine_generator:
        found.append("A piezo triangle runs at its own frequency, but both analog "
                     "outputs share one sample clock: switch the frequency generator "
                     "off, or put it on a counter output.")
    return found


def plan(settings, outputs=None):
    """Work out the device writes for valid settings.

    Args: settings is DaqSettings that problems() accepts; outputs is the
        connected device's DeviceOutputs (None: the USB-6251's).
    Returns: An OutputPlan. Raises ValueError when problems() finds any.
    """
    outputs = outputs or DeviceOutputs(SIMULATED_DEVICE)
    found = problems(settings, outputs)
    if found:
        raise ValueError(" ".join(found))
    s = settings
    analog = tuple(channel for channel in outputs.analog
                   if channel in (s.generator_output, s.piezo_output))
    sine = s.generator_on and s.generator_output in outputs.analog
    values = {channel: 0.0 for channel in analog}
    rate, samples, frequency, buffer_samples = 0.0, 0, 0.0, 0
    piezo_frequency = 0.0
    parts = []
    if sine:
        rate, samples, frequency = choose_timing(s.generator_frequency_hz, len(analog))
        phase = 2 * math.pi * np.arange(samples) / samples
        values[s.generator_output] = (s.generator_offset_v
                                      + s.generator_amplitude_v * np.sin(phase))
        buffer_samples = _buffer_samples(rate, samples)
        parts.append(f"generator {output_label(s.generator_output)} sine "
                     f"{frequency:.6g} Hz (asked {s.generator_frequency_hz:g}), "
                     f"{s.generator_amplitude_v:g} V peak, offset "
                     f"{s.generator_offset_v:g} V; {samples} samples per cycle at "
                     f"{rate:,.0f} S/s")
    counter = None
    if s.generator_on and s.generator_output in outputs.counters:
        half = 0.5 / s.generator_frequency_hz
        counter = CounterPlan(s.generator_output, s.generator_terminal, half, half)
        frequency = square_frequency(s.generator_frequency_hz)
        parts.append(f"generator {output_label(s.generator_output)} square "
                     f"{frequency:.6g} Hz (asked {s.generator_frequency_hz:g}) on "
                     f"{terminal_label(s.generator_terminal)}")
    if not s.generator_on and s.generator_output != OFF:
        parts.append(f"generator {output_label(s.generator_output)} off")
    if s.piezo_output != OFF:
        if not s.piezo_on:
            parts.append(f"piezo {output_label(s.piezo_output)} off (0 V)")
        elif s.piezo_mode == SINE:
            shift = math.radians(s.piezo_phase_deg)
            phase = 2 * math.pi * np.arange(samples) / samples + shift
            values[s.piezo_output] = (s.piezo_voltage_v
                                      + s.piezo_amplitude_v * np.sin(phase))
            parts.append(f"piezo {output_label(s.piezo_output)} sine "
                         f"{s.piezo_amplitude_v:g} V peak at "
                         f"{s.piezo_phase_deg:g}°, centre {s.piezo_voltage_v:g} V")
        elif s.piezo_mode == TRIANGLE:
            rate, samples, piezo_frequency = choose_timing(s.piezo_frequency_hz, len(analog))
            values[s.piezo_output] = triangle_wave(s.piezo_bottom_v, s.piezo_top_v, samples)
            buffer_samples = _buffer_samples(rate, samples)
            parts.append(f"piezo {output_label(s.piezo_output)} triangle "
                         f"{s.piezo_bottom_v:g}–{s.piezo_top_v:g} V at "
                         f"{piezo_frequency:.6g} Hz (asked {s.piezo_frequency_hz:g}); "
                         f"{samples} samples per cycle at {rate:,.0f} S/s")
        else:
            values[s.piezo_output] = float(s.piezo_voltage_v)
            parts.append(f"piezo {output_label(s.piezo_output)} holds "
                         f"{s.piezo_voltage_v:g} V")
    summary = "; ".join(parts) if parts else "no outputs assigned"
    return OutputPlan(analog, rate, samples, buffer_samples, values, counter,
                      frequency, piezo_frequency, summary,
                      describe_drive(s, outputs, frequency, piezo_frequency))


def describe_drive(settings, outputs, frequency, piezo_frequency=0.0):
    """Describe what valid settings make the outputs produce.

    Args: settings is the DaqSettings plan() accepted; outputs is the
        device's DeviceOutputs; frequency is the generator's real frequency
        (0 when it is off); piezo_frequency is the piezo triangle's real
        frequency (0 for other modes).
    Returns: A Drive.
    """
    s = settings
    waveform, amplitude = "", 0.0
    if s.generator_on and s.generator_output in outputs.analog:
        waveform, amplitude = "sine", s.generator_amplitude_v
    elif s.generator_on and s.generator_output in outputs.counters:
        waveform = "square"
    mode = s.piezo_mode if s.piezo_on and s.piezo_output != OFF else ""
    sine = mode == SINE
    centre, swing = (s.piezo_voltage_v if mode else 0.0,
                     s.piezo_amplitude_v if sine else 0.0)
    if mode == TRIANGLE:
        centre = (s.piezo_bottom_v + s.piezo_top_v) / 2
        swing = (s.piezo_top_v - s.piezo_bottom_v) / 2
    return Drive(waveform, frequency if waveform else 0.0, amplitude,
                 s.generator_offset_v if waveform == "sine" else 0.0, mode,
                 centre, swing, s.piezo_phase_deg if sine else 0.0,
                 outputs.device.label, outputs.device.simulated,
                 piezo_frequency if mode == TRIANGLE else 0.0)


class SimulatedBackend:
    """A stand-in device that records what it would write and drives nothing."""

    simulated = True

    def __init__(self):
        """Start closed, with nothing written.

        Args: None.
        Returns: None.
        """
        self.outputs = None
        self.applied = None  # the last OutputPlan
        self.zeroed = 0  # how many times the outputs in use were set to 0 V
        self.zeroed_all = 0  # how many times every output of the device was
        self.levels = {}  # channel -> the value or waveform it holds
        self.shift = {}  # channel -> volts added to its readback (fault injection in tests)

    def open(self, device):
        """Pretend to connect.

        Args: device is ignored (always the simulated device).
        Returns: DeviceOutputs with the USB-6251's outputs.
        """
        self.outputs = DeviceOutputs(SIMULATED_DEVICE)
        return self.outputs

    def apply(self, output_plan):
        """Remember the plan.

        Args: output_plan is the OutputPlan.
        Returns: None.
        """
        self.applied = output_plan
        self.levels.update(output_plan.values)

    def set_levels(self, levels):
        """Pretend to hold constant levels.

        Args: levels maps an AO channel to volts.
        Returns: None.
        """
        self.levels.update(levels)

    def read_outputs(self, channels):
        """Read back what each output holds (exactly, plus any injected shift).

        Args: channels are AO channel names.
        Returns: A dict from channel to an array of READBACK_SAMPLES volts.
        """
        found = {}
        for channel in channels:
            held = np.asarray(self.levels.get(channel, 0.0), dtype=float).ravel()
            found[channel] = np.resize(held, READBACK_SAMPLES) + self.shift.get(channel, 0.0)
        return found

    def zero(self):
        """Pretend to set every output to 0 V and stop the counter.

        Args: None.
        Returns: None.
        """
        self.applied = None
        self.levels = {}
        self.zeroed += 1

    def zero_all(self):
        """Pretend to set every analog output of the device to 0 V.

        Args: None.
        Returns: None.
        """
        self.applied = None
        self.levels = {}
        self.zeroed_all += 1

    def close(self):
        """Pretend to disconnect.

        Args: None.
        Returns: None.
        """
        self.outputs = None


class PylablibBackend:
    """Drive a real NI device through pyLabLib's NIDAQ class.

    Use from one thread only (DaqController's worker). AO channels cannot be
    removed from a pyLabLib task, so a change of assignment reconnects.
    """

    simulated = False
    GENERATOR = "generator"  # pyLabLib's name for the counter output

    def __init__(self):
        """Start disconnected.

        Args: None.
        Returns: None.
        """
        self.device = None  # the DAQmx name
        self.nidaq = None
        self.outputs = None
        self.analog = ()  # AO channels added to the current NIDAQ
        self.counter = None  # (counter, terminal) of the pulse output

    def open(self, device):
        """Connect and read what the device offers.

        Args: device is the DAQmx device name, such as "Dev1".
        Returns: DeviceOutputs. Raises pyLabLib or NI-DAQmx errors.
        """
        ni = load_ni()
        self.device = device
        self.nidaq = ni.NIDAQ(device)
        info = self.nidaq.get_device_info()
        dev = self.nidaq.dev
        self.outputs = DeviceOutputs(
            DeviceInfo(info.name, info.model, info.serial_number),
            analog=tuple(_short(name) for name in dev.ao_physical_chans.channel_names),
            # Counters only: the USB-6251 also lists "freqout" here, which
            # cannot make the generator's square wave.
            counters=tuple(name for name in map(_short, dev.co_physical_chans.channel_names)
                           if name.startswith("ctr")),
            terminals=tuple(name for name in (_short(t) for t in dev.terminals)
                            if name.startswith("pfi")))
        return self.outputs

    def apply(self, output_plan):
        """Write the plan: AO levels or waveforms, then the counter output.

        Args: output_plan is the OutputPlan.
        Returns: None.
        """
        p = output_plan
        if p.analog != self.analog:
            self._reconnect(p.analog)
        if p.analog:
            names = list(p.analog)
            # Hold the first samples while the clock changes, then start the
            # waveforms from the top; constants keep the task on demand.
            self.nidaq.setup_voltage_output_clock(rate=0)
            self.nidaq.set_voltage_outputs(names, [p.values[n] for n in names])
            if p.rate:
                self.nidaq.setup_voltage_output_clock(
                    rate=p.rate, continuous=True, samps_per_chan=p.buffer_samples,
                    autoloop=True)
        self._apply_counter(p.counter)

    def _apply_counter(self, counter):
        """Start, change or stop the square wave.

        Args: counter is a CounterPlan, or None to stop it.
        Returns: None.
        """
        if counter is None:
            if self.counter is not None:
                self.nidaq.stop_pulse_output(self.GENERATOR)
            return
        where = (counter.counter, counter.terminal)
        if where != self.counter:
            self.nidaq.add_pulse_output(self.GENERATOR, counter.counter, counter.terminal,
                                        kind="time", on=counter.high_s,
                                        off=counter.low_s, continuous=True)
            self.counter = where
        else:
            self.nidaq.set_pulse_output(self.GENERATOR, on=counter.high_s,
                                        off=counter.low_s, restart=False)
        self.nidaq.start_pulse_output(self.GENERATOR)

    def _reconnect(self, analog):
        """Open a fresh NIDAQ with exactly these AO channels.

        Args: analog is the tuple of AO channel names.
        Returns: None.
        """
        self.zero()
        self.nidaq.close()
        self.nidaq = load_ni().NIDAQ(self.device)
        self.counter = None
        for channel in analog:
            self.nidaq.add_voltage_output(channel, channel, rng=(-MAX_VOLTS, MAX_VOLTS))
        self.analog = tuple(analog)

    def zero_all(self):
        """Set every analog output of the device to 0 V, not only those in use.

        Used on connecting, so a level left by a crash or another program is
        cleared before anything else is written.

        Args: None.
        Returns: None.
        """
        if self.counter is not None:
            self.nidaq.stop_pulse_output(self.GENERATOR)
        self.set_levels({channel: 0.0 for channel in self.outputs.analog})

    def set_levels(self, levels):
        """Hold constant levels on every analog output (unlisted ones at 0 V).

        Args: levels maps an AO channel to volts.
        Returns: None.
        """
        channels = tuple(self.outputs.analog)
        if self.analog != channels:
            self._reconnect(channels)
        self.nidaq.setup_voltage_output_clock(rate=0)
        self.nidaq.set_voltage_outputs(list(channels),
                                       [float(levels.get(c, 0.0)) for c in channels])

    def read_outputs(self, channels):
        """Measure the analog outputs through the device's internal channels.

        M Series devices read each AO against AO GND on "_aoN_vs_aognd"
        through their own ADC, so no cable is needed.

        Args: channels are AO channel names.
        Returns: A dict from channel to an array of READBACK_SAMPLES volts.
        """
        nidaqmx = load_ni().daq.nidaqmx
        task = nidaqmx.Task()
        try:
            for channel in channels:
                task.ai_channels.add_ai_voltage_chan(f"{self.device}/_{channel}_vs_aognd",
                                                     min_val=-MAX_VOLTS, max_val=MAX_VOLTS)
            task.timing.cfg_samp_clk_timing(
                READBACK_RATE, sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=READBACK_SAMPLES)
            data = np.asarray(task.read(number_of_samples_per_channel=READBACK_SAMPLES,
                                        timeout=1.0), dtype=float)
        finally:
            task.close()
        rows = data.reshape(len(channels), -1)
        return dict(zip(channels, rows))

    def zero(self):
        """Set every AO channel to 0 V and stop the counter output.

        Args: None.
        Returns: None.
        """
        if self.nidaq is None:
            return
        if self.analog:
            self.nidaq.setup_voltage_output_clock(rate=0)
            self.nidaq.set_voltage_outputs(list(self.analog), [0.0] * len(self.analog))
        if self.counter is not None:
            self.nidaq.stop_pulse_output(self.GENERATOR)

    def close(self):
        """Set the outputs to 0 V and disconnect (M Series AO keeps its last value).

        Args: None.
        Returns: None.
        """
        if self.nidaq is None:
            return
        try:
            self.zero()
        finally:
            self.nidaq.close()
            self.nidaq = None
            self.analog = ()
            self.counter = None


def _short(name):
    """Strip the device from a channel or terminal name.

    Args: name is a name such as "Dev1/ao0" or "/Dev1/PFI4".
    Returns: The lower-case last part: "ao0", "pfi4".
    """
    return name.strip("/").split("/")[-1].lower()


def list_devices():
    """List the NI devices NI-DAQmx can see.

    Args: None.
    Returns: (devices, reason): a list of DeviceInfo, and None, or the text
        returned by unavailable_reason() when devices cannot be listed.
    """
    try:
        found = load_ni().list_nidaqmx_devices()
    except Exception as exc:  # missing package, driver or platform support
        return [], unavailable_reason(exc)
    return [DeviceInfo(d.name, d.model, d.serial_number) for d in found], None


class DaqController:
    """Own the NI device on a worker thread and apply settings to it.

    request() queues work: "scan" (list devices), "connect" (device: a
    DAQmx name or SIMULATED), "apply" (settings: DaqSettings), "zero" (every
    output to 0 V), "self_test" (step each AO to SELF_TEST_V and back,
    reading it back) and "close" (0 V, disconnect, end the thread). Results
    arrive on the events queue as (kind, payload): "devices" ((list of
    DeviceInfo, reason or None)), "connected" (DeviceOutputs), "applied"
    (OutputPlan), "rejected" (list of problems), "zeroed", "failed" (text),
    "fault" (text: a self-check set every output to 0 V), "self_test"
    ((passed, list of lines)) and "closed". move_piezo() is the one blocking
    call: the camera engine uses it to step the piezo between frames for
    phase-shifting ESPI.

    Self-checks, while a device is connected: every AO is set to 0 V on
    connecting; about once a second the outputs are read back through the
    device's internal channels and compared with what was written; and when
    heartbeat() has been called (by the window) but not for WATCHDOG_S while
    an output is on, everything goes to 0 V. A real device takes no drive
    signal until arm() has recorded the safety checklist for this connection
    and output assignment.
    """

    def __init__(self, log, backend_factory=None, clock=time.monotonic):
        """Create an idle controller; the thread starts with the first request.

        Args: log is an AppLog or PrefixedLog; backend_factory, given a device
            name, returns the backend for it (default: SimulatedBackend for
            SIMULATED, otherwise PylablibBackend); clock gives seconds (tests
            pass their own).
        Returns: None.
        """
        self.log = log
        self.backend_factory = backend_factory or default_backend
        self.clock = clock
        self._beat = None  # when the window last called heartbeat()
        self._armed = (None, frozenset())  # device name, roles the checklist covered
        self.expected = {}  # worker thread only: channel -> (lowest, highest) volts
        self._faulted = False  # worker thread only: a self-check fault is reported once
        self.events = queue.SimpleQueue()
        self._commands = queue.SimpleQueue()
        self._thread = None
        self._lock = threading.Lock()
        self.backend = None  # worker thread only
        self.outputs = None  # worker thread only
        self._status = {"device": None, "active": False, "summary": "Not connected",
                        "settings": None}

    def request(self, action, **payload):
        """Queue work for the device thread, starting it if needed.

        Args: action is "scan", "connect", "apply", "zero" or "close";
            payload is device= for connect and settings= for apply.
        Returns: None.
        """
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                if action == "close":
                    self.events.put(("closed", None))
                    return
                self._thread = threading.Thread(target=self._run, name="espi-daq",
                                                daemon=True)
                self._thread.start()
            self._commands.put((action, payload))

    def heartbeat(self):
        """Tell the controller the window is still responding.

        Once called, a window that stops calling it for WATCHDOG_S while an
        output is on gets every output set to 0 V.

        Args: None.
        Returns: None.
        """
        self._beat = self.clock()

    def arm(self, settings):
        """Record that the safety checklist was ticked off for the roles switched on.

        Each role it covered (and the socket it drives) stays allowed for the
        connected device until the next connection.

        Args: settings is the DaqSettings the checklist was shown for.
        Returns: None.
        """
        with self._lock:
            device = self._status["device"]
            name = device.name if device else None
            covered = self._armed[1] if self._armed[0] == name else frozenset()
            self._armed = (name, covered | driven_roles(settings))

    def is_armed(self, settings):
        """Tell whether drive signals may go to the device with these settings.

        Args: settings is DaqSettings.
        Returns: True for the simulated device, or when the checklist covered
            every role these settings switch on (with its socket) for the
            connected device.
        """
        with self._lock:
            device = self._status["device"]
            if device is not None and device.simulated:
                return True
            return device is not None and self._armed[0] == device.name and \
                driven_roles(settings) <= self._armed[1]

    def checklist_done(self):
        """Tell whether the checklist was ticked off for anything on this connection.

        Args: None.
        Returns: True once arm() covered at least one role of the connected device.
        """
        with self._lock:
            device = self._status["device"]
            return device is not None and self._armed[0] == device.name and \
                bool(self._armed[1])

    def status(self):
        """Report the device and what it is producing.

        Args: None.
        Returns: A dict copy: device (DeviceInfo or None), active (True while
            any output is on), summary (text), settings (the DaqSettings last
            written to the device, or None since connecting or zeroing).
        """
        with self._lock:
            return dict(self._status)

    def move_piezo(self, settings, volts, timeout):
        """Hold the piezo at a level and wait until the device has it.

        The piezo is switched on in hold mode at volts; every other output
        keeps the values in settings. Unlike request("apply"), nothing is
        put on the events queue, so stepping does not flood the DAQ window.

        Args: settings is DaqSettings (the generator part is written as is);
            volts is the piezo level at the DAQ; timeout is the longest wait
            in seconds.
        Returns: The OutputPlan written. Raises PiezoMoveError when there is
            no device, the level breaks a rule (the safety limit), the device
            fails or it does not answer within timeout.
        """
        reply = queue.SimpleQueue()
        self.request("piezo", settings=replace(settings, piezo_on=True, piezo_mode=HOLD,
                                               piezo_voltage_v=float(volts)),
                     reply=reply)
        try:
            ok, result = reply.get(timeout=timeout)
        except queue.Empty:
            raise PiezoMoveError(f"the DAQ did not apply {volts:g} V within "
                                 f"{timeout:g} s") from None
        if not ok:
            raise PiezoMoveError(result)
        return result

    def shutdown(self, timeout):
        """Set every output to 0 V, disconnect and wait for the thread.

        Args: timeout is the longest wait in seconds.
        Returns: True when the thread has finished (or never started).
        """
        self.request("close")
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def _set_status(self, **changes):
        """Update the status the front end reads.

        Args: changes are status keys and values.
        Returns: None.
        """
        with self._lock:
            self._status.update(changes)

    def _run(self):
        """Run queued work until "close".

        Args: None.
        Returns: None.
        """
        next_check = self.clock() + CHECK_INTERVAL_S
        while True:
            try:
                action, payload = self._commands.get(
                    timeout=max(0.0, next_check - self.clock()))
            except queue.Empty:
                action, payload = None, {}
            if action == "close":
                self._handle("close", self._disconnect)
                self.events.put(("closed", None))
                return
            if action is not None:
                name = "self-check" if action == "check" else action
                self._handle(name, getattr(self, f"_do_{action}"), **payload)
            if self.clock() >= next_check:  # also between busy phase steps
                next_check = self.clock() + CHECK_INTERVAL_S
                self._handle("self-check", self._do_check)

    def _handle(self, name, work, **payload):
        """Run one piece of work; on any failure report it and go to 0 V.

        Args: name names the work in the log; work is a bound method;
            payload its keyword arguments.
        Returns: None.
        """
        try:
            work(**payload)
        except Exception as exc:  # report, make the outputs safe, carry on
            error_reports.caught(self.log, exc, "daq", f"NI DAQ {name} failed")
            self._make_safe()
            self.events.put(("failed", f"{type(exc).__name__}: {exc}"))

    def _do_check(self):
        """Run the self-checks: the freeze watchdog, then the output readback.

        Args: None.
        Returns: None.
        """
        if self.backend is None or self._faulted:
            return
        beat = self._beat
        if beat is not None and self.status()["active"]:
            silent = self.clock() - beat
            if silent > WATCHDOG_S:
                self._fault(f"the window stopped responding for {silent:.1f} s")
                return
        if self.expected:
            problem = readback_problem(self.expected,
                                       self.backend.read_outputs(list(self.expected)))
            if problem:
                self._fault(problem)

    def _fault(self, reason):
        """Set every output to 0 V after a failed self-check, and say why once.

        Args: reason is a sentence fragment naming what was wrong.
        Returns: None.
        """
        self._faulted = True
        self.log.warning(f"NI DAQ self-check: {reason}. Every output was set to 0 V.")
        self.backend.zero_all()
        self.expected = {channel: (0.0, 0.0) for channel in self.outputs.analog}
        self._set_status(active=False, summary=f"Stopped by the self-check: {reason}",
                         settings=None)
        self.events.put(("fault", reason))

    def _do_scan(self):
        """List devices.

        Args: None.
        Returns: None.
        """
        devices, reason = list_devices()
        if reason:
            self.log.warning(f"NI DAQ: {reason}")
        self.events.put(("devices", (devices, reason)))

    def _do_connect(self, device):
        """Connect to a device, disconnecting (at 0 V) from any other.

        Args: device is a DAQmx name or SIMULATED.
        Returns: None.
        """
        self._disconnect()
        backend = self.backend_factory(device)
        self.outputs = backend.open(device)
        self.backend = backend
        with self._lock:
            self._armed = (None, frozenset())  # the checklist is ticked again per connection
        backend.zero_all()  # clear anything a crash or another program left on
        self.expected = {channel: (0.0, 0.0) for channel in self.outputs.analog}
        self._faulted = False
        info = self.outputs.device
        self.log.info(f"NI DAQ: connected to {info.label}; outputs "
                      f"{', '.join(self.outputs.analog + self.outputs.counters)}")
        self._set_status(device=info, active=False, summary="Connected, all outputs at 0 V")
        self.events.put(("connected", self.outputs))

    def _do_apply(self, settings):
        """Validate, plan and write settings.

        Args: settings is DaqSettings.
        Returns: None.
        """
        if self.backend is None:
            self.events.put(("rejected", ["Connect to a device first."]))
            return
        found = problems(settings, self.outputs)
        if not found and drives(settings) and not self.is_armed(settings):
            found = [CHECKLIST_REQUIRED]
        if found:
            self.events.put(("rejected", found))
            return
        output_plan = self._write(settings)
        prefix = "SIMULATED, nothing driven: " if self.backend.simulated else ""
        self.log.info(f"NI DAQ applied: {prefix}{output_plan.summary}")
        self.events.put(("applied", output_plan))

    def _write(self, settings):
        """Plan valid settings, write them and record them as applied.

        Args: settings is DaqSettings that problems() accepts.
        Returns: The OutputPlan written.
        """
        output_plan = plan(settings, self.outputs)
        self.backend.apply(output_plan)
        self.expected = expected_levels(output_plan)
        self._faulted = False
        active = (settings.generator_on and settings.generator_output != OFF) or (
            settings.piezo_on and settings.piezo_output != OFF)
        self._set_status(active=active, summary=output_plan.summary, settings=settings)
        return output_plan

    def _do_piezo(self, settings, reply):
        """Write one phase-step level and answer the waiting move_piezo().

        Args: settings is DaqSettings with the piezo level to hold; reply is
            the queue move_piezo() waits on, given (True, OutputPlan) or
            (False, reason).
        Returns: None. A device failure is answered, then raised for _run.
        """
        if self.backend is None:
            reply.put((False, "the NI DAQ is not connected"))
            return
        found = problems(settings, self.outputs)
        if not found and not self.is_armed(settings):
            found = [CHECKLIST_REQUIRED]
        if found:
            reply.put((False, " ".join(found)))
            return
        try:
            output_plan = self._write(settings)
        except Exception as exc:
            reply.put((False, f"{type(exc).__name__}: {exc}"))
            raise
        self.log.debug(f"NI DAQ phase step: {output_plan.summary}")
        reply.put((True, output_plan))

    def _do_zero(self):
        """Set every output to 0 V.

        Args: None.
        Returns: None.
        """
        if self.backend is not None:
            self.backend.zero()
            self.expected = {channel: (0.0, 0.0) for channel in self.expected}
            self.log.info("NI DAQ: every output set to 0 V")
        self._set_status(active=False, summary="All outputs at 0 V", settings=None)
        self.events.put(("zeroed", None))

    def _do_self_test(self):
        """Step each analog output to SELF_TEST_V and back to 0 V, reading it back.

        Every output is left at 0 V afterwards. Needs no checklist: the
        window asks for the piezo amplifier to be off first.

        Args: None.
        Returns: None; the result is a "self_test" event (passed, lines).
        """
        if self.backend is None:
            self.events.put(("self_test", (False, ["Connect to a device first."])))
            return
        channels = list(self.outputs.analog)
        lines, passed = [], True
        for channel in channels + [None]:  # None: every output back at 0 V
            levels = {c: (SELF_TEST_V if c == channel else 0.0) for c in channels}
            self.backend.set_levels(levels)
            readings = self.backend.read_outputs(channels)
            problem = readback_problem({c: (v, v) for c, v in levels.items()}, readings)
            step = f"{output_label(channel)} at {SELF_TEST_V:g} V" if channel \
                else "every output at 0 V"
            means = ", ".join(f"{output_label(c)} {float(np.mean(readings[c])):.3f} V"
                              for c in channels)
            lines.append(f"{step}: {'FAILED, ' + problem if problem else 'OK'} ({means})")
            passed = passed and not problem
        self.backend.zero_all()
        self.expected = {channel: (0.0, 0.0) for channel in channels}
        self._faulted = False
        self._set_status(active=False, settings=None,
                         summary="Self-test " + ("passed" if passed else "FAILED")
                         + "; all outputs at 0 V")
        self.log.info(f"NI DAQ self-test {'passed' if passed else 'FAILED'}: "
                      + "; ".join(lines))
        self.events.put(("self_test", (passed, lines)))

    def _disconnect(self):
        """Set the outputs to 0 V and release the device, if one is open.

        Args: None.
        Returns: None.
        """
        backend, self.backend = self.backend, None
        self.outputs = None
        self.expected = {}
        self._set_status(device=None, active=False, summary="Not connected", settings=None)
        if backend is not None:
            backend.close()
            self.log.info("NI DAQ: outputs at 0 V, device released")

    def _make_safe(self):
        """After a failure, try to leave every output at 0 V.

        Args: None.
        Returns: None. A second failure is logged and the device released.
        """
        self._set_status(active=False, summary="Stopped after an error", settings=None)
        self.expected = {}
        if self.backend is None:
            return
        try:
            self.backend.zero()
        except Exception as exc:  # the device may be gone; release what is left
            self.log.warning(f"NI DAQ: could not set the outputs to 0 V: {exc}")
            try:
                self.backend.close()
            except Exception as close_exc:
                self.log.warning(f"NI DAQ: could not release the device: {close_exc}")
            self.backend = None
            self.outputs = None
            self._set_status(device=None)


def default_backend(device):
    """Choose the backend for a device name.

    Args: device is a DAQmx name or SIMULATED.
    Returns: SimulatedBackend for SIMULATED, otherwise PylablibBackend.
    """
    return SimulatedBackend() if device == SIMULATED else PylablibBackend()
