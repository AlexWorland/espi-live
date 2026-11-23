"""In-memory stand-in for NI's nidaqmx package, under the real pyLabLib NIDAQ.

Only the surface pyLabLib's NIDAQ uses for analog voltage outputs and
counter pulse outputs is modelled, plus the finite AI read of the internal
"_aoN_vs_aognd" channels the self-check uses: Task (AO, CO and AI channels,
timing, output stream, write, read, start, stop, close), system.Device, system.System and the real
nidaqmx constants and errors. A FakeHardware records what reaches each
output, so tests can check the volts and waveforms a device would produce.
Install it with install(), which patches pyLabLib's module-level nidaqmx.
"""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import nidaqmx.constants as constants
import nidaqmx.errors as errors
from pylablib.devices import NI

from espi_live import daq

SampleTimingType = constants.SampleTimingType
AcquisitionType = constants.AcquisitionType


class FakeHardware:
    """What the device's outputs are doing, and failures to inject."""

    def __init__(self):
        """Start with every output idle.

        Args: None.
        Returns: None.
        """
        self.ao = {}  # "ao0" -> last value (float) or waveform (array) written
        self.ao_running = False  # an AO waveform task is generating
        self.ao_rate = 0.0
        self.counters = {}  # "ctr0" -> dict(terminal, high, low, running)
        self.tristated = []  # terminals released by pyLabLib
        self.fail = {}  # operation name -> exception to raise once
        self.closed_tasks = 0
        self.readback_shift = {}  # "ao1" -> volts added to what the AO reads back as
        self.reads = 0  # AI reads of the outputs

    def check(self, operation):
        """Raise an injected failure for an operation, once.

        Args: operation is a name such as "write" or "start".
        Returns: None.
        """
        exc = self.fail.pop(operation, None)
        if exc is not None:
            raise exc


def daq_error(message="Device not found", code=-200220):
    """Build a real nidaqmx DaqError.

    Args: message and code are the error text and NI error code.
    Returns: An nidaqmx.errors.DaqError.
    """
    return errors.DaqError(message, code)


def _short(name):
    """Strip the device from a physical channel or terminal name.

    Args: name is "/Dev1/ao0" or "/Dev1/pfi4".
    Returns: "ao0", "pfi4".
    """
    return name.strip("/").split("/")[-1].lower()


class FakeAOChannel:
    """One analog output channel in a task."""

    def __init__(self, physical, name):
        """Remember the channel.

        Args: physical is "/Dev1/ao0"; name is the channel's name.
        Returns: None.
        """
        self.physical_channel = SimpleNamespace(name=physical)
        self.name = name


class FakeChannels(list):
    """A task's channel collection (AO and CO add methods)."""

    def __init__(self, task):
        """Start empty.

        Args: task is the owning FakeTask.
        Returns: None.
        """
        super().__init__()
        self.task = task

    def add_ao_voltage_chan(self, physical_channel, name_to_assign_to_channel="",
                            min_val=-10.0, max_val=10.0):
        """Add an AO channel.

        Args: physical_channel, name_to_assign_to_channel, min_val and max_val
            as in nidaqmx.
        Returns: The channel.
        """
        self.task.hardware.check("add_ao")
        channel = FakeAOChannel(physical_channel, name_to_assign_to_channel)
        self.append(channel)
        return channel

    def add_ai_voltage_chan(self, physical_channel, name_to_assign_to_channel="",
                            terminal_config=None, min_val=-5.0, max_val=5.0):
        """Add an AI channel; only the internal "_aoN_vs_aognd" ones read anything.

        Args: physical_channel such as "Dev1/_ao1_vs_aognd"; the rest as in nidaqmx.
        Returns: The channel.
        """
        self.task.hardware.check("add_ai")
        channel = FakeAOChannel(physical_channel, name_to_assign_to_channel)
        self.append(channel)
        return channel

    def add_co_pulse_chan_time(self, counter, name_to_assign_to_channel="", units=None,
                               idle_state=None, initial_delay=0.0, low_time=0.01,
                               high_time=0.01):
        """Add a counter pulse channel timed in seconds.

        Args: counter is "/Dev1/ctr0"; low_time and high_time are seconds;
            the rest as in nidaqmx.
        Returns: The channel.
        """
        self.task.hardware.check("add_co")
        channel = SimpleNamespace(physical_channel=SimpleNamespace(name=counter),
                                  co_pulse_term="", co_pulse_high_time=high_time,
                                  co_pulse_low_time=low_time)
        self.append(channel)
        return channel


class FakeTiming:
    """A task's timing: on demand until a sample clock is configured."""

    def __init__(self):
        """Start on demand.

        Args: None.
        Returns: None.
        """
        self.samp_timing_type = SampleTimingType.ON_DEMAND
        self.samp_clk_rate = 0.0
        self.samp_quant_samp_mode = AcquisitionType.CONTINUOUS
        self.samp_quant_samp_per_chan = 0

    def cfg_samp_clk_timing(self, rate, source="", active_edge=None,
                            sample_mode=AcquisitionType.FINITE, samps_per_chan=1000):
        """Use the sample clock.

        Args: rate, source, sample_mode and samps_per_chan as in nidaqmx.
        Returns: None.
        """
        self.samp_timing_type = SampleTimingType.SAMPLE_CLOCK
        self.samp_clk_rate = float(rate)
        self.samp_quant_samp_mode = sample_mode
        self.samp_quant_samp_per_chan = samps_per_chan

    def cfg_implicit_timing(self, sample_mode=AcquisitionType.FINITE, samps_per_chan=1000):
        """Use implicit timing (counter outputs).

        Args: sample_mode and samps_per_chan as in nidaqmx.
        Returns: None.
        """
        self.samp_timing_type = SampleTimingType.IMPLICIT
        self.samp_quant_samp_mode = sample_mode
        self.samp_quant_samp_per_chan = samps_per_chan


class FakeTask:
    """An NI-DAQmx task that writes into FakeHardware."""

    hardware = None  # set by install()

    def __init__(self, new_task_name=""):
        """Create an empty task.

        Args: new_task_name is ignored.
        Returns: None.
        """
        self.hardware.check("task")
        self.ao_channels = FakeChannels(self)
        self.co_channels = FakeChannels(self)
        self.ai_channels = FakeChannels(self)
        self.di_channels = []
        self.timing = FakeTiming()
        self.out_stream = SimpleNamespace(regen_mode=None)
        self.running = False

    def write(self, data, auto_start=False, timeout=10.0):
        """Send data to the AO channels.

        Args: data is one value per channel, or one row of samples per
            channel; auto_start starts a clocked task.
        Returns: The number of samples written per channel.
        """
        self.hardware.check("write")
        data = np.asarray(data, dtype=float)
        rows = data.reshape(len(self.ao_channels), -1) if data.ndim else data.reshape(1, 1)
        for channel, row in zip(self.ao_channels, rows):
            key = _short(channel.physical_channel.name)
            self.hardware.ao[key] = float(row[0]) if row.size == 1 else row.copy()
        if auto_start:
            self.start()
        return rows.shape[1]

    def read(self, number_of_samples_per_channel=1, timeout=10.0):
        """Read the AI channels: each internal channel reads its AO's output.

        Args: number_of_samples_per_channel and timeout as in nidaqmx.
        Returns: A list of samples for one channel, else one list per channel.
        """
        self.hardware.check("read")
        self.hardware.reads += 1
        rows = []
        for channel in self.ai_channels:
            name = _short(channel.physical_channel.name)  # "_ao1_vs_aognd"
            output = name.strip("_").split("_")[0]
            held = np.asarray(self.hardware.ao.get(output, 0.0), dtype=float).ravel()
            samples = np.resize(held, number_of_samples_per_channel)
            rows.append(list(samples + self.hardware.readback_shift.get(output, 0.0)))
        return rows[0] if len(rows) == 1 else rows

    def _counter(self):
        """Find this task's counter record.

        Args: None.
        Returns: The FakeHardware counters entry, or None for other tasks.
        """
        if not self.co_channels:
            return None
        key = _short(self.co_channels[0].physical_channel.name)
        return self.hardware.counters.setdefault(key, {})

    def start(self):
        """Start the task.

        Args: None.
        Returns: None.
        """
        self.hardware.check("start")
        self.running = True
        counter = self._counter()
        if counter is not None:
            channel = self.co_channels[0]
            counter.update(terminal=_short(channel.co_pulse_term or ""),
                           high=channel.co_pulse_high_time,
                           low=channel.co_pulse_low_time, running=True)
        elif self.ao_channels and self.timing.samp_timing_type != SampleTimingType.ON_DEMAND:
            self.hardware.ao_running = True
            self.hardware.ao_rate = self.timing.samp_clk_rate

    def stop(self):
        """Stop the task; analog outputs keep their last value.

        Args: None.
        Returns: None.
        """
        self.hardware.check("stop")
        self.running = False
        counter = self._counter()
        if counter is not None:
            counter["running"] = False
        elif self.ao_channels:
            self.hardware.ao_running = False

    def is_task_done(self):
        """Report whether the task is idle.

        Args: None.
        Returns: True when not running.
        """
        return not self.running

    def close(self):
        """Release the task (stopping it first).

        Args: None.
        Returns: None.
        """
        self.hardware.check("close")
        if self.running:
            self.stop()
        self.hardware.closed_tasks += 1


class FakeDevice:
    """An NI device as nidaqmx.system.Device describes it."""

    def __init__(self, name, product_type="USB-6251", serial=0x1A2B3C4D, simulated=False,
                 hardware=None):
        """Describe a USB-6251.

        Args: name is the DAQmx name; product_type and serial describe it;
            simulated is what is_simulated reports (a device created as
            simulated in NI MAX); hardware is the FakeHardware whose injected
            "is_simulated" failure the property raises, or None.
        Returns: None.
        """
        self.name = name
        self._simulated = simulated
        self._hardware = hardware
        self.product_type = product_type
        self.dev_serial_num = serial
        self.ao_physical_chans = SimpleNamespace(
            channel_names=[f"{name}/ao0", f"{name}/ao1"])
        self.co_physical_chans = SimpleNamespace(
            # The real USB-6251 also lists its frequency output here.
            channel_names=[f"{name}/ctr0", f"{name}/ctr1", f"{name}/freqout"])
        self.terminals = ([f"/{name}/PFI{index}" for index in range(16)]
                          + [f"/{name}/20MHzTimebase", f"/{name}/ao/SampleClock"])

    @property
    def is_simulated(self):
        """Report whether the device is an NI MAX simulated one, as nidaqmx does.

        Args: None.
        Returns: True for a simulated device. Raises an injected "is_simulated" failure.
        """
        if self._hardware is not None:
            self._hardware.check("is_simulated")
        return self._simulated


def make_module(hardware, devices=("Dev1",), simulated=()):
    """Build the fake nidaqmx module.

    Args: hardware is the FakeHardware; devices are the DAQmx names present;
        simulated are the names among them that report is_simulated.
    Returns: A namespace with Task, system, constants, errors and DaqError.
    """
    task = type("Task", (FakeTask,), {"hardware": hardware})
    found = {name: FakeDevice(name, simulated=name in simulated, hardware=hardware)
             for name in devices}

    def device(name):
        """Look a device up, as nidaqmx.system.Device does.

        Args: name is the DAQmx name.
        Returns: The FakeDevice. Raises DaqError when it is absent.
        """
        hardware.check("device")
        if name not in found:
            raise daq_error(f"Device {name} not found")
        return found[name]

    def system():
        """Describe the local system, as nidaqmx.system.System() does.

        Args: None.
        Returns: A namespace with devices and tristate_output_term.
        """
        hardware.check("system")
        return SimpleNamespace(devices=list(found.values()),
                               tristate_output_term=hardware.tristated.append)

    return SimpleNamespace(Task=task, system=SimpleNamespace(Device=device, System=system),
                           constants=constants, errors=errors, DaqError=errors.DaqError)


def install(test, devices=("Dev1",), simulated=()):
    """Run pyLabLib's real NIDAQ on the fake for the rest of a test.

    Args: test is the unittest.TestCase (its cleanup undoes the patches);
        devices are the DAQmx names present; simulated are the names among
        them that nidaqmx reports as simulated (NI MAX simulated devices).
    Returns: The FakeHardware.
    """
    hardware = FakeHardware()
    patchers = (mock.patch.object(NI.daq, "nidaqmx", make_module(hardware, devices, simulated)),
                mock.patch.object(daq, "_NI", NI))
    for patcher in patchers:
        patcher.start()
        test.addCleanup(patcher.stop)
    return hardware
