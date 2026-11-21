"""NI DAQ outputs: a frequency generator and a piezo driver on one NI device.

The frequency generator excites the specimen (a sine on an analog output, or
a square TTL wave from a counter on a PFI terminal). The piezo driver sets the
reference piezo's voltage: a held level for manual phase steps, or a sine
locked to the generator's frequency and phase. Each role is assigned to its
own output, and every voltage is clamped to a limit the user sets for the
amplifier behind it.

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
from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np

from espi_live import error_reports

AO_OUTPUTS = ("ao0", "ao1")
COUNTER_OUTPUTS = ("ctr0", "ctr1")
OFF = ""  # an unassigned role
PFI_TERMINALS = tuple(f"pfi{index}" for index in range(16))
SIMULATED = "Simulated"  # device name of the stand-in that drives no hardware
HOLD, SINE = "hold", "sine"  # piezo modes
PIEZO_MODES = (HOLD, SINE)

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
MAX_VOLTS = 10.0  # the AO range
MIN_WRITE_S = 0.001  # pyLabLib writes at least this much waveform per start

INSTALL_HINT = 'pip install pylablib-lightweight "nidaqmx>=1.4,<1.8"'
_NI = None  # pylablib.devices.NI once loaded


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
class OutputPlan:
    """Exactly what to write to the device, worked out from DaqSettings."""

    analog: tuple = ()  # AO channels in use, in order
    rate: float = 0.0  # AO sample rate; 0 holds constant levels
    samples_per_cycle: int = 0
    buffer_samples: int = 0  # per channel, a whole number of cycles
    values: dict = field(default_factory=dict)  # channel -> float or 1-D array
    counter: CounterPlan = None
    frequency_hz: float = 0.0  # what the generator really produces (0 when off)
    summary: str = ""


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
        swing = abs(s.piezo_amplitude_v) if s.piezo_mode == SINE else 0.0
        peak = abs(s.piezo_voltage_v) + swing
        if peak > s.piezo_limit_v:
            found.append(f"The piezo driver would reach {peak:g} V, above its "
                         f"{s.piezo_limit_v:g} V limit.")
        if s.piezo_mode not in PIEZO_MODES:
            found.append(f"Unknown piezo mode {s.piezo_mode!r}.")
        elif s.piezo_mode == SINE and s.piezo_on and not (
                s.generator_on and s.generator_output in outputs.analog):
            found.append("A piezo sine follows the frequency generator: put the "
                         "generator on an analog output and switch it on.")
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
    parts = []
    if sine:
        rate, samples, frequency = choose_timing(s.generator_frequency_hz, len(analog))
        phase = 2 * math.pi * np.arange(samples) / samples
        values[s.generator_output] = (s.generator_offset_v
                                      + s.generator_amplitude_v * np.sin(phase))
        cycles = math.ceil(max(rate * MIN_WRITE_S, samples) / samples)
        buffer_samples = cycles * samples
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
        else:
            values[s.piezo_output] = float(s.piezo_voltage_v)
            parts.append(f"piezo {output_label(s.piezo_output)} holds "
                         f"{s.piezo_voltage_v:g} V")
    summary = "; ".join(parts) if parts else "no outputs assigned"
    return OutputPlan(analog, rate, samples, buffer_samples, values, counter,
                      frequency, summary)


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
        self.zeroed = 0  # how many times every output was set to 0 V

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

    def zero(self):
        """Pretend to set every output to 0 V and stop the counter.

        Args: None.
        Returns: None.
        """
        self.applied = None
        self.zeroed += 1

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
            counters=tuple(_short(name) for name in dev.co_physical_chans.channel_names),
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
    output to 0 V) and "close" (0 V, disconnect, end the thread). Results
    arrive on the events queue as (kind, payload): "devices" ((list of
    DeviceInfo, reason or None)), "connected" (DeviceOutputs), "applied"
    (OutputPlan), "rejected" (list of problems), "zeroed", "failed" (text)
    and "closed".
    """

    def __init__(self, log, backend_factory=None):
        """Create an idle controller; the thread starts with the first request.

        Args: log is an AppLog or PrefixedLog; backend_factory, given a device
            name, returns the backend for it (default: SimulatedBackend for
            SIMULATED, otherwise PylablibBackend).
        Returns: None.
        """
        self.log = log
        self.backend_factory = backend_factory or default_backend
        self.events = queue.SimpleQueue()
        self._commands = queue.SimpleQueue()
        self._thread = None
        self._lock = threading.Lock()
        self.backend = None  # worker thread only
        self.outputs = None  # worker thread only
        self._status = {"device": None, "active": False, "summary": "Not connected"}

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

    def status(self):
        """Report the device and what it is producing.

        Args: None.
        Returns: A dict copy: device (DeviceInfo or None), active (True while
            any output is on), summary (text).
        """
        with self._lock:
            return dict(self._status)

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
        while True:
            action, payload = self._commands.get()
            try:
                if action == "close":
                    self._disconnect()
                    self.events.put(("closed", None))
                    return
                getattr(self, f"_do_{action}")(**payload)
            except Exception as exc:  # report, make the outputs safe, carry on
                error_reports.caught(self.log, exc, "daq", f"NI DAQ {action} failed")
                self._make_safe()
                self.events.put(("failed", f"{type(exc).__name__}: {exc}"))

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
        if found:
            self.events.put(("rejected", found))
            return
        output_plan = plan(settings, self.outputs)
        self.backend.apply(output_plan)
        active = (settings.generator_on and settings.generator_output != OFF) or (
            settings.piezo_on and settings.piezo_output != OFF)
        prefix = "SIMULATED, nothing driven: " if self.backend.simulated else ""
        self.log.info(f"NI DAQ applied: {prefix}{output_plan.summary}")
        self._set_status(active=active, summary=output_plan.summary)
        self.events.put(("applied", output_plan))

    def _do_zero(self):
        """Set every output to 0 V.

        Args: None.
        Returns: None.
        """
        if self.backend is not None:
            self.backend.zero()
            self.log.info("NI DAQ: every output set to 0 V")
        self._set_status(active=False, summary="All outputs at 0 V")
        self.events.put(("zeroed", None))

    def _disconnect(self):
        """Set the outputs to 0 V and release the device, if one is open.

        Args: None.
        Returns: None.
        """
        backend, self.backend = self.backend, None
        self.outputs = None
        self._set_status(device=None, active=False, summary="Not connected")
        if backend is not None:
            backend.close()
            self.log.info("NI DAQ: outputs at 0 V, device released")

    def _make_safe(self):
        """After a failure, try to leave every output at 0 V.

        Args: None.
        Returns: None. A second failure is logged and the device released.
        """
        self._set_status(active=False, summary="Stopped after an error")
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
