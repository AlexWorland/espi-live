# Driving the specimen and the reference piezo from an NI USB-6251, in step with the camera

Research notes and a proposed design, written 2026-11-21. No application code
has been written. The code snippets are sketches of the `nidaqmx` calls and have
not been run on hardware.

**How sure each claim is.** Every hardware or API claim carries a source
number (`[S1]` …; see [Sources](#sources)). Claims marked **(unverified)**
come from forum posts, from a different model in the same family, or from
my own inference. Check them at the bench before relying on them. ni.com was
down for scheduled maintenance on the day of this research, so the NI
specification and user manual were read from NI's own PDFs hosted elsewhere
(the document numbers are given) rather than from ni.com.

---

## 1. Recommendation

1. **Make the DAQ the master clock for the Blackfly (strategy a).** Use one
   NI-DAQmx AO task with both channels: AO 0 drives the shaker or PZT
   amplifier, and AO 1 drives the reference-piezo amplifier. One AO task is
   all an M Series device can run hardware-timed [S5], and because both
   channels share the sample clock, the drive-to-piezo phase is exact. Clock
   the task from the internal 20 MHz timebase. Counter 0 generates the camera
   frame trigger as a continuous pulse train. Its timebase is the same 20 MHz
   clock, and its start trigger is the first edge of `ao/SampleClock`, so
   every frame starts at the same sample index of the drive waveform. Route
   the counter output to a PFI that has a front-panel BNC (PFI 0–7), and
   from there to the Blackfly's opto-isolated **Line 0** with
   `TriggerSelector=FrameStart`, `TriggerSource=Line0`,
   `TriggerActivation=RisingEdge`. Choose the frame period as a whole number
   M of drive periods (f_frame = f_drive / M). Use `ExposureMode=Timed`, or
   `TriggerWidth` when the exposure itself must be an exact whole number of
   drive cycles. Return the camera's **ExposureActive** on Line 1 to a PFI
   so the DAQ can count the frames that were actually exposed.
2. **Treat the analog XC-ST50 separately (strategy b, camera as master).**
   Its external trigger shutter and HD/VD sync are usable from 5 V logic.
   The USB-Live2 stick, however, gives the program no frame identity or
   trigger timing. Lock the DAQ to the camera's own HD/VD outputs instead,
   for time-averaged work.
3. **Use software timing (strategy c) only as a fallback for time-averaged
   ESPI.** The waveform stays hardware-timed, but nothing ties frames to
   drive phase. Expect about 1 ms of USB call latency plus Windows
   scheduling jitter. That is too coarse for stroboscopic or
   frame-synchronised phase stepping.
4. **Amplifiers are required.** The USB-6251's AO delivers ±10 V at only
   ±5 mA [S1]. A shaker needs a power amplifier, and a PZT or reference
   piezo needs a piezo amplifier. Never connect a transducer straight to AO.
5. **Software.** Python 3.10 on the camera PC needs `nidaqmx>=1.4,<1.8`.
   1.7.0 is current and supports 3.10, and the unreleased 1.8.0 drops 3.10
   [S9]. It also needs the NI-DAQmx driver. Put the integration in a
   GUI-free `daq.py` whose `DaqController` runs its **own worker thread**
   (section 6). Plans are computed by a pure, testable function. `nidaqmx`
   is imported lazily, and the tests use an in-memory fake.

Frequency limits in one line: AO updates at up to 2.86 MS/s on one channel
or 2.00 MS/s per channel on two [S1]. A clean sine (at least 20 samples per
cycle, plus a simple low-pass filter) is practical up to about 100 kHz
**(inference)**. Any drive frequency can be locked to the frames by choosing
f_frame = f_drive / M.

---

## 2. USB-6251 capabilities that matter

Sources: NI 6251 Device Specifications 375213C-01 (Jun 2016) [S1], and the
M Series User Manual 371022L-01 (Jul 2016) [S2].

### Analog output

| Item | Value | Source |
|---|---|---|
| Channels | 2 (AO 0, AO 1), 16-bit | [S1] |
| Maximum update rate | 2.86 MS/s with 1 channel; 2.00 MS/s **per channel** with 2 | [S1] |
| Ranges | ±10 V, ±5 V, ±external reference on APFI 0 | [S1] |
| Output current | ±5 mA; output impedance 0.2 Ω | [S1] |
| Slew rate / settling | 20 V/µs; 2 µs full-scale step to 1 LSB | [S1] |
| Glitch at midscale | 10 mV for 1 µs | [S1] |
| FIFO | **8,191 samples shared among the channels used** (about 4,095 per channel with both) | [S1] |
| Waveform modes | non-periodic; periodic regeneration from the onboard FIFO; periodic regeneration from the host buffer, including dynamic update | [S1] |
| USB transfer | "USB Signal Stream", or programmed I/O; 4 streams shared by AI/AO/DI/DO/counters | [S1] |
| Timing | 50 ns resolution; 50 ppm accuracy | [S1] |
| **Power-on glitch** | **1.5 V peak for 1.5 s**; on the USB screw-terminal model the AO level is undefined until USB configuration finishes | [S1] |
| AO sample clock sources | AO timebase divided down (20 MHz or 100 kHz timebase, PFI, …), **Counter n Internal Output**, PFI 0–15 | [S2] ch. 5 |
| AO start trigger sources | software, PFI 0–15, `ai/StartTrigger`, `ai/ReferenceTrigger`, analog comparison, and "several internal signals" (see Device Routes in MAX) | [S2] ch. 5 |
| AO sample-clock timebase | can be taken from a PFI but **cannot be exported** | [S2] ch. 5 |

The three regeneration modes, as NI defines them [S2]:

- **Host-buffer regeneration** (the default). The driver keeps copying the
  PC buffer to the FIFO, and you can write into the buffer while the task
  runs.
- **FIFO regeneration** (`ao_use_only_on_brd_mem = True`). The whole buffer
  must fit in the FIFO, and after the start nothing crosses USB.
- **Non-regeneration.** The program must keep the buffer fed, or the task
  stops with an underflow error.

### Counters, PFI and DIO

| Item | Value | Source |
|---|---|---|
| Counters | 2 × 32-bit; internal timebases 80 MHz, 20 MHz and 100 kHz (50 ppm); external source up to 20 MHz | [S1] |
| Output modes | pulse, pulse train **with dynamic updates**, frequency division, equivalent-time sampling. **A finite pulse train uses both counters.** | [S1], [S2] ch. 7 |
| Counter source routes | 80/20/0.1 MHz timebases, PFI 0–15, the other counter's TC or Gate. **The AO sample clock is not listed.** | [S2] ch. 7 |
| Counter gate (start trigger) routes | PFI 0–15, `ai/StartTrigger`, `ai/SampleClock`, **`ao/SampleClock`**, `do/SampleClock`, … **`ao/StartTrigger` is not listed.** | [S2] ch. 7 |
| Default counter terminals | CTR 0: SRC PFI 8, GATE PFI 9, **OUT PFI 12**. CTR 1: SRC PFI 3, GATE PFI 4, OUT PFI 13. FREQ OUT PFI 14. Any counter output can be routed to any PFI. | [S2] App. A |
| Frequency generator | a separate square wave: 10 MHz or 100 kHz ÷ 1–16. Too coarse for this work. | [S1] |
| PFI/DIO | 24 lines: P0.0–7 (hardware-timed DIO), PFI 0–15 (static or timing). 50 kΩ pull-downs. Inputs: VIH ≥ 2.2 V, VIL ≤ 0.8 V, 5.25 V maximum. Outputs: 5 V logic, ±16 mA on PFI and ±24 mA on P0. Debounce filters 125 ns / 6.425 µs / 2.56 ms. | [S1] |
| DO waveform | no timing engine of its own; it can be clocked from `ao/SampleClock` or a counter. Regeneration from the 2,047-sample FIFO at up to 10 MHz; streaming over USB at up to 1 MHz. | [S1] |
| Bus | USB 2.0 Hi-Speed. **No RTSI, and no PLL** (the PLL is PCI/PXI only), so the board cannot lock its timebase to an external reference clock | [S1] |
| Power | **External 11–30 VDC, 20 W supply.** The board is not bus-powered. | [S1] |

**USB-6251 BNC front panel** ([S2] Fig. A-16): BNCs for AI 0–7, AO 0, AO 1,
APFI 0, **PFI 0–7**, and two user BNCs (USER 1/2) wired to screw terminals.
PFI 8–15, P0.0–7, +5 V and D GND are on **screw terminals only**. The default
`CTR 0 OUT` (PFI 12) therefore has no BNC. Either route the counter to
PFI 0–7 in software (`co_pulse_term = "/Dev1/PFI4"`), or jumper PFI 12 to
USER 1 at the screw block.

### Cleanest sine, and why amplifiers are needed

- With f_s up to 2 MS/s per channel and a 16-bit DAC, a regenerated sine of
  200 or more samples per cycle is clean up to about 10 kHz. At 20 samples
  per cycle with a first- or second-order RC filter, it is clean up to about
  100 kHz **(inference from [S1]: settling 2 µs, slew 20 V/µs, glitch
  10 mV·µs)**. A shaker or a stacked PZT filters the image frequencies
  mechanically. At high frequencies, an external reconstruction filter is
  cheap insurance.
- ±5 mA and ±10 V [S1] cannot drive a shaker (a few ohms) or a piezo (µF of
  capacitance at tens to hundreds of volts). **Both need an external
  amplifier.** Into a 50 Ω input, AO would current-limit at about 0.25 V.
  Into a typical amplifier input of 10 kΩ or more it is fine.
- **Piezo current** is I ≈ C·dV/dt. For a sine that is I_pk = 2π·f·C·V_pk,
  and the amplifier's own output capacitance adds to C [S20], [S21]. For a
  1.8 µF stack **(example value; check the lab's piezo)**:
  - holding λ/8 (0.8 V on a 100 nm/V stack) at 10 kHz needs about 45 mA;
  - the same at 1 kHz needs about 4.5 mA;
  - a 0.8 V phase step in 1 ms needs about 1.4 mA.

  Current limits are usually reached before slew limits on capacitive loads
  [S22].
- **Resolution is not a concern.** With an amplifier gain of 15 and a
  100 nm/V stack, one AO LSB on the ±10 V range (0.305 mV) is about 0.46 nm
  of mirror motion.

---

## 3. Drivers and Python support

- **NI-DAQmx on Windows 11.**
  - The NI-DAQmx 2023 Q2 readme lists Windows 11 and 10 as supported, and
    no 32-bit Windows [S6].
  - The NI 6251 over USB appears in NI-DAQmx device tables, for example the
    19.1 readme [S7].
  - An NI obsolescence page still offers the USB-6251 as a *replacement*
    product [S8].
  - **(unverified)** I could not confirm that the newest NI-DAQmx release
    (2025/2026) still supports the USB-6251, because ni.com was in
    maintenance. On the camera PC, check NI's Hardware and OS Compatibility
    page (`ni.com/r/hw-support`), then confirm that NI MAX lists the device
    and its self-test passes.
- **`nidaqmx` Python package** [S9]:
  - **1.7.0** (2026-10-26) is the newest release. It declares
    `requires_python >=3.9` and lists Python 3.10 as supported.
  - The repository's CHANGELOG already has an unreleased **1.8.0: "Removed
    support for Python 3.9 and 3.10"**, and the master README says
    "CPython 3.11+".
  - The camera PC runs Python 3.10, so pin
    `python -m pip install "nidaqmx>=1.4,<1.8"`. pip would also pick 1.7.x
    by itself once 1.8 declares 3.11+, but the pin makes it explicit.
  - The driver is separate. Install it with `python -m nidaqmx installdriver`
    (which downloads NI's installer) or from NI's download page [S9].
  - Without the driver, importing `nidaqmx` works, but the first call raises
    `nidaqmx.errors.DaqNotFoundError` [S9 source, `errors.py`/`_lib.py`].
- **PyInstaller.** espi-live imports optional packages lazily, so PyInstaller
  will not find `nidaqmx` unless it is declared as a hidden import, or added
  with `--collect-all nidaqmx` together with its dependencies (`hightime`,
  `nitypes`, `python-decouple`, `tzlocal`, `requests`, `click`, …) [S9]. The
  NI-DAQmx DLL (`nicaiu.dll`) comes from the installed driver and is not
  bundled.

### Code patterns (sketches against nidaqmx 1.7.0; names checked in its source [S10])

**Continuous regenerated two-channel sine (drive plus piezo), with one shared
clock:**

```python
import numpy as np, nidaqmx
from nidaqmx.constants import AcquisitionType, RegenerationMode

fs = 200_000.0                  # 20 MHz / 100: exact
n = 200                         # one 1 kHz cycle; any whole number of cycles works
t = np.arange(n) / fs
drive = 1.0 * np.sin(2 * np.pi * 1000 * t)
piezo = 0.4 * np.sin(2 * np.pi * 1000 * t + np.deg2rad(30))   # locked phase offset
ao = nidaqmx.Task("espi-ao")
ao.ao_channels.add_ao_voltage_chan("Dev1/ao0:1", min_val=-10, max_val=10)
ao.timing.cfg_samp_clk_timing(fs, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=n)
ao.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
for ch in ao.ao_channels:
    ch.ao_use_only_on_brd_mem = True      # FIFO regeneration: 2 × 200 ≤ 8191
ao.write(np.vstack([drive, piezo]), auto_start=False)
actual_fs = ao.timing.samp_clk_rate       # log the coerced rate
```

**Camera trigger from counter 0, phase-locked to the AO samples:**

```python
from nidaqmx.constants import Level, Edge
co = nidaqmx.Task("espi-cam-trigger")
# 33 drive periods per frame = 33 × 200 samples × 100 ticks = 660,000 ticks = 33.000 ms
high = 160_000                           # 8.000 ms = 8 drive cycles (TriggerWidth mode)
ch = co.co_channels.add_co_pulse_chan_ticks(
    "Dev1/ctr0", source_terminal="/Dev1/20MHzTimebase",
    idle_state=Level.LOW, initial_delay=2, high_ticks=high, low_ticks=660_000 - high)
ch.co_pulse_term = "/Dev1/PFI4"          # a PFI that has a BNC
co.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
co.triggers.start_trigger.cfg_dig_edge_start_trig("/Dev1/ao/SampleClock", Edge.RISING)
co.start()                               # arm first
ao.start()                               # its first sample-clock edge starts the counter
```

- The counter's gate accepts `ao/SampleClock` according to [S2], but the
  *Some of these options may not be available in some driver software* note
  applies. **Check this route in NI MAX → Device Routes.**
- Fallback, with only documented routes: create a one-channel AI task. Set
  **both** the AO and the counter start triggers to `/Dev1/ai/StartTrigger`
  (both are listed in [S2]), arm them, then start the AI task last.
- Both clocks come from the same onboard oscillator, so frame and drive
  never drift apart. The offset between them is a fixed number of ticks:
  measure it once and log it.

**Phase-continuous frequency change (counter-clocked AO):**

```python
from nidaqmx.types import CtrFreq
clk = nidaqmx.Task("espi-ao-clock")
clk.co_channels.add_co_pulse_chan_freq("Dev1/ctr1", freq=200_000.0, duty_cycle=0.5)
clk.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
ao.timing.cfg_samp_clk_timing(200_000.0, source="/Dev1/Ctr1InternalOutput",
                              sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=n)
# … start ao, then clk; later:
clk.write(CtrFreq(freq=210_000.0, duty_cycle=0.5))   # f_drive = f_clk / n, no phase jump
```

- The counter "pulse train with dynamic updates" is listed in [S1], and NI
  documents on-the-fly pulse-train frequency changes [S12]. The new period
  takes effect at the next pulse.
- The cost is counter 1, plus a frequency step of 80 MHz / ticks. With this
  scheme, counter 0 is still free for the camera, but the camera period no
  longer contains a whole number of drive periods. Use it for *sweeping and
  searching*, not for locked measurements.

### What a glitch-free frequency change really takes

| Method | Glitch | Phase lock to frames | Notes |
|---|---|---|---|
| Stop, write a new buffer, start | output holds its last value through the gap; phase restarts | kept (the counter is re-armed too) | simplest. Ramp the amplitude to 0 over about 20 ms first, so the shaker is not kicked. A few camera frames are lost. |
| Write into a running host-buffer regeneration | can tear at the write position unless the write is aligned to a period and the waveforms are phase-continuous | kept | NI forum reports that new data is ignored or torn in regeneration mode [S13] **(unverified)** |
| Non-regenerating stream with a software DDS | none, if the stream is never starved | kept (fixed f_s) | needs a writer thread. USB latency of buffer size ÷ f_s; risk of underflow (section 7) |
| Counter-clocked AO with `CtrFreq` updates | none | lost while sweeping | best for resonance searches |
| Change `samp_clk_rate` while running | — | — | a forum reply says "many boards" allow it **(unverified for M Series)** [S13] |

---

## 4. Synchronization strategies

### (a) DAQ as master (recommended for the Blackfly)

Wiring: counter output → Blackfly **Line 0** (opto input, pin 2), and DAQ
D GND → **Opto GND** (pin 5). The camera's GPIO is a 6-pin Hirose
HR10A-7R-6PB; the mating plug is an HR10A-7P-6S(73) [S14].

| Pin | Line | Function |
|---|---|---|
| 1 | 3 | +12 V power in / non-isolated input |
| 2 | 0 | opto input |
| 3 | 2 | non-isolated I/O / 3.3 V out |
| 4 | 1 | opto output |
| 5 | — | Opto GND (not camera ground) |
| 6 | — | camera GND |

Camera nodes [S15]. Configure them on the engine thread, in the order given in
`docs/camera/acquisition-and-performance.md`:

1. `TriggerMode=Off`.
2. `TriggerSelector=FrameStart`, `TriggerSource=Line0`,
   `TriggerActivation=RisingEdge`.
3. `TriggerOverlap=ReadOut`. Enumeration values: Off, ReadOut,
   PreviousFrame.
4. `ExposureMode`:
   - `Timed`: the exposure is `ExposureTime`;
   - `TriggerWidth`: the exposure follows the high time of the pulse.
5. `TriggerMode=On`.
6. Optionally `TriggerDelay` (µs) to trim the phase.
7. For verification: `LineSelector=Line1`, `LineMode=Output`,
   `LineSource=ExposureActive`. Other `LineSource` choices are
   FrameTriggerWait and Counter/User outputs.

`AcquisitionFrameRateEnable` should probably be turned off while triggered,
so the internal rate limiter cannot cap or reject triggers **(inference;
check on the camera)**.

- **Pros.**
  - Exact, drift-free phase between frames, drive and piezo, from a single
    oscillator.
  - A whole number of drive cycles per exposure can be enforced in
    hardware with `TriggerWidth` (high_ticks = n·T_drive).
  - Stroboscopic gating and phase stepping fall out of the same clock.
  - The app keeps its normal grab loop.
- **Cons.**
  - The camera produces no frames until the DAQ runs, so the engine sees
    100 ms grab timeouts ([`engine.py`](../../src/espi_live/engine.py),
    `CAMERA_TIMEOUT_MS`).
  - Two devices must be started and stopped in the right order.
  - The opto input adds latency: 13.4 µs L→H and 9.2 µs H→L at 5 V
    (family data, [S17]). In `TriggerWidth` mode, the 4 µs difference
    between the two shortens the exposure by about 0.08 cycle at 20 kHz.
    Non-isolated Line 2 (about 1 µs) avoids this, at the cost of tying the
    grounds together.

**Electrical check for a PFI driving Line 0:**

- NI PFI outputs are 5 V logic with 16 mA of drive [S1]. The Blackfly
  family's opto input wants 2.6–30 V for a high, ≤1.4 V for a low, and
  draws 3.5–7 mA [S16], [S17].
- The family pages show the opto input with its own current control, since
  they specify a current range over a 2.6–30 V input range. **No external
  resistor or buffer should be needed (verify against the 63S4 drawing).**
- FLIR's warning: connect Opto GND first, before applying voltage to a
  GPIO line [S16].

**Caveat.** The detailed electrical and timing tables are from the
**BFS-U3-200S6** family pages [S16], [S17]. The BFS-U3-63S4 pages that FLIR
publishes give the pinout and node reference [S14], [S15], but I did not
find a 63S4-specific electrical table online. The board-level BFS-U3-63S4-BD2
table [S18] is a different, non-isolated design. Check the 63S4's
Installation Guide before wiring.

**ExposureActive back to the DAQ.** Line 1 is an open-collector opto output:
pull it up with 1 kΩ to the DAQ's +5 V, and tie Opto GND to D GND. FLIR's
reference values are 4.2 mA and a 0.87 V low level, with propagation delays
of 36 µs (L→H) and 18 µs (H→L) [S16], [S19]. Count its edges on counter 1,
or read them as a DI change event, to prove that triggers sent equal frames
exposed.

### (b) Camera as master

- **Blackfly.** Run the camera free at its own rate, and feed ExposureActive
  (Line 1) into a PFI as the AO start trigger.
  - M Series AO start triggers are not retriggerable. The manual describes
    retriggering only for counter outputs [S2] **(absence inferred; check
    the `retriggerable` property on the device)**. One start edge
    therefore aligns only the first frame: after that, the camera crystal
    and the DAQ's 50 ppm timebase drift apart (up to about 100 ppm, or
    0.1 cycle/s at 1 kHz).
  - Using ExposureActive as the AO *sample clock* gives only one sample per
    frame, which is useless for the drive.
  - **Verdict:** usable only for time-averaged work, where drift does not
    matter.
- **XC-ST50** (this is where camera-as-master makes sense):
  - Set the HD/VD switch to INT. The camera then *outputs* HD (pin 6,
    15.734 kHz) and VD (pin 7) on the 12-pin connector [S23], and the
    DC-700 brings them out on BNCs.
  - Feed HD to a PFI as the **AO sample clock**. The drive is then
    f_HD / N, for example N = 16 gives 983.4 Hz, locked to the camera's
    own crystal.
  - Use VD as the AO start trigger.
  - **(unverified)** I found no published HD/VD output level, only the
    2–5 Vp-p input level. Measure it with a scope against the PFI's
    VIH = 2.2 V. Also check that the 75 Ω termination switch is set as
    the manual requires for output.

### (c) Software-timed fallback

- The drive is still a hardware-timed, regenerated AO waveform. The program
  only starts, stops and retunes it.
- Frames are tagged after the fact, using the host time of arrival or,
  better, Blackfly chunk data (FrameID and timestamp).
- **Jitter.** A software-timed single-point call on USB DAQ costs about
  0.5–1 ms **(unverified third-party measurement [S25]; NI only says that
  external buses always have more latency than PCI [S24])**. Windows sleep
  granularity is 1 ms at best; the app already sets 1 ms timer resolution
  (`WindowsPreviewTimer`). Camera transfer and processing latency add
  several ms more.
- **Usable for:**
  - time-averaged ESPI, which needs no phase;
  - slow phase stepping (≤10 fps), if one frame is discarded after each
    step.
- **Not usable for:** stroboscopic ESPI or 30 fps stepping.

---

## 5. ESPI timing

### Time-averaged (sinusoidal drive, continuous laser)

- Theory: the fringe function is J0(K·a), and the subtraction or rectified
  image shows J0² [S27], [S28].
- Lokberg and Rosvold show that this holds only when the exposure contains
  many vibration cycles. With few cycles, the fringe function depends on the
  number of cycles and on the starting phase in the cycle [S30].
- Rules:
  - Make Te much longer than T_drive. If Te·f_drive = n is small, make it
    a whole number.
  - When n is not a whole number, the starting phase walks from frame to
    frame at the beat between f_drive and the frame rate. The pattern then
    flickers, and Previous-mode differences show phantom fringes.
  - DAQ-as-master removes the walk entirely.
- **At the app defaults (30 fps, Te = 8000 µs, T_frame = 33.333 ms):**

| f_drive | cycles per exposure n | f_drive / f_frame (unlocked) | Effect |
|---|---|---|---|
| 125 Hz | 1.0 | 4.167 | n = 1: valid only because it is a whole number; poor J0² |
| 500 Hz | 4.0 | 16.667 | whole number. Exposure-start phase repeats every 3 frames (0.667) |
| 1.0 kHz | 8.0 | 33.333 | whole number; 3-frame phase pattern |
| 1.1 kHz | 8.8 | 36.667 | **non-integer**: 0.8 extra cycle, about a 10 % residue, flickering every 3 frames |
| 5.0 kHz | 40.0 | 166.667 | fine even when not a whole number (residue ≤ 1/40) |
| 12.5 kHz | 100 | 416.667 | fine |

  - A whole number of cycles in 8 ms means f_drive must be a multiple of
    **125 Hz**. Otherwise, snap Te to round(n)/f_drive, or use
    `TriggerWidth` with high_ticks = n·T_drive·20 MHz.
  - Locked frame rates near 30 fps: f_drive = 1 kHz with M = 33 →
    **30.303 fps**, and M = 34 → 29.412 fps. A 3 kHz drive with M = 100
    gives exactly 30.000 fps. Any f_drive that is a multiple of 30 Hz allows
    exactly 30 fps.
  - Counter period at 1 kHz, M = 33, with 200 samples per cycle
    (f_s = 200 kS/s = 20 MHz / 100): 660,000 ticks = 33.000 ms exactly.
    `TriggerWidth` with 160,000 high ticks gives 8.000 ms, which is exactly
    8 cycles.

### Stroboscopic

- The illumination (or reference) is pulsed once per drive cycle at a chosen
  phase φ. This "freezes" the deformation, so ordinary subtraction or
  phase-stepping analysis applies [S31], [S32].
- Rules:
  - Pulse width τ ≤ T_drive / 10, which keeps the smear under ±18°.
  - The exposure gathers n = Te·f_drive pulses.
  - φ must be fixed relative to the drive, so the gate must come from the
    same clock as AO.
- The USB-6251 has no free counter left once counter 0 triggers the camera
  and counter 1 counts ExposureActive. A **DO waveform on P0.x clocked by
  `ao/SampleClock`** (a listed route [S1]) gives a gate at the AO's sample
  resolution, phase-exact by construction. Wire P0.0 to USER 2 for a BNC,
  then to the AOM driver's TTL input.
- **Rolling shutter.** In the Blackfly (IMX178, rolling shutter with global
  reset [S3]), rows expose at staggered times, so a CW-laser phase sequence
  would differ row to row. A strobe makes every row see the same phases,
  because light only exists at φ.

### Phase stepping (for example 4 steps of π/2 per frame)

- Theory: four frames at reference phases 0, π/2, π, 3π/2 give the phase by
  arctangent [S33], [S34].
- **Mirror step.** In reflection at near-normal incidence, the mirror moves
  λ/8 per π/2: **79.1 nm at 632.8 nm, 66.5 nm at 532 nm**. With 100 nm/V,
  that is about 0.8 V at the amplifier output, or about 53 mV at AO with a
  gain of 15.
- The step must happen while **no row is exposing**. FLIR: with rolling
  shutter and global reset, all lines start together, but each line ends
  later than the one before, delayed by the line readout [S3].
- **Blackfly gap per frame (estimate):**
  - Full-frame readout ≈ 1/59 s = 16.9 ms with the 10-bit ADC, inferred
    from the 59 fps maximum. Line time ≈ 8.3 µs **(inference from [S4])**.
  - At 30 fps and Te = 8 ms: 33.3 − (16.9 + 8.0) ≈ **8.4 ms** left in which
    to step and settle the piezo.
  - With the **14-bit ADC** the full-frame maximum is 30 fps [S4]. Readout
    then takes about 33 ms and **there is no gap at 30 fps.** Fixes: select
    the 10-bit ADC, reduce the ROI height, or lower the frame rate (at 20
    fps the gap is about 25 ms).
  - Read `AdcBitDepth` and `SensorShutterMode` on the camera; I do not know
    their current values.
- Synchronisation:
  - With DAQ-as-master, the piezo staircase is part of the AO 1 buffer,
    one level per frame period, aligned to the trigger counter. The buffer
    must contain a whole number of both drive cycles and step sequences:
    length = lcm(...). For example, 4 frames × 33 cycles × 200 samples =
    26,400 samples per channel. That exceeds the FIFO, so use host-buffer
    regeneration, which streams at 2 × 200 kS/s over USB.
  - Frame identity matters. The engine uses `StreamBufferHandlingMode =
    NewestOnly` (`camera.configure_camera`), which silently drops frames
    when processing lags. That would scramble the step order. Phase-step
    capture needs OldestFirst plus ChunkFrameID, or a per-frame marker.

### Reference-beam modulation at the drive frequency (Høgmoen and Løkberg)

- Modulate the reference mirror at f_drive with amplitude a_r and phase θ
  relative to the drive. The fringe function becomes J0(K·|a_obj − a_r·e^{iθ}|).
  - The zero-order (bright) fringe moves to where the object matches the
    reference.
  - Small amplitudes become measurable.
  - Sweeping θ maps the vibration phase [S29].
  [S26], [S27], [S28]
- The USB-6251 suits this well: AO 0 and AO 1 share one sample clock, so θ is
  exact. Compute it analytically in the buffer, which gives sub-sample
  resolution.
- **Offset (beat) variant.** Piezo at f_drive + Δ (Δ ≈ 0.5–2 Hz) makes the
  fringes pulse live. It needs Δ·Te ≪ 1 (1 Hz × 8 ms = 0.008 cycle, fine)
  and f_frame ≫ Δ. The buffer must hold a whole number of cycles of both
  frequencies, so its length is 1/Δ: 200,000 samples per channel for
  Δ = 1 Hz at 200 kS/s. That needs host-buffer regeneration.

---

## 6. Proposed espi-live integration

### Module and threading

Add a GUI-free and camera-free `src/espi_live/daq.py`. Its parts:

- `load_nidaqmx()`: a lazy import in the style of `camera.load_pyspin()`.
- `DaqSettings`: a dataclass.
- `plan_timing(settings, camera) -> DaqPlan | list[Problem]`: pure, with
  no `nidaqmx` calls.
- `DaqController`: the worker.
- `SimulatedDaq`: for demo mode.
- `list_daq_devices()`.

**`DaqController` runs its own worker thread (`espi-daq`) and is not owned
by the `Engine`.** Reasons:

1. **One DAQ serves the whole rig, while an `Engine` belongs to one
   `Channel` (camera)** (AGENTS.md, "Several cameras"). Engines are replaced
   on `switch_live` and when cameras are turned on or off. Owning the drive
   from an engine would stop the shaker every time the displayed camera
   changes.
2. **The engine loop is frame-paced.** Restarting a task, rewriting a buffer
   or recovering from an error can block for tens of ms over USB
   **(unverified figure; measure it)**, and the engine must not stall
   inside `grab`.
3. **The threading rule stays simple.** The rule "one thread per device
   session" is the same one `CameraSession` follows: every `nidaqmx` call
   happens on `espi-daq`, and every PySpin call stays on the engine thread.
   The DAQ never touches the camera.
4. **The API mirrors the engine's:** `update_settings(**fields)`,
   `request(action, **payload)` (`start`, `stop`, `retune`, `zero`,
   `self_test`), `status()`, and an `events` queue (`daq_started`,
   `daq_stopped`, `daq_failed`, `daq_counts`, `daq_missing`). The Qt window
   drains it the same way it drains `Engine.events`.

### Camera side, inside the engine's rules

- Add a `CameraSession.configure_trigger(source, activation, exposure_mode,
  overlap)` method and a `trigger_off()` method, written with the existing
  `try_setting`/`set_enum` guards.
- Reach them through a new engine action, `request("camera_trigger",
  on=..., ...)`. The engine answers with `trigger_applied` or
  `trigger_failed`, so PySpin is touched only on the engine thread.
- While triggered, the engine's grab-timeout status should read
  **"Waiting for DAQ trigger"** rather than logging timeout warnings every
  5 s. `CaptureSession` (XC-ST50) has no trigger API; its sync is
  camera-as-master.

**Start order (DAQ as master):**

1. UI → `engine.request("camera_trigger", on=True)`.
2. Wait for `trigger_applied`.
3. UI → `daq.request("start", plan)`: the counter is armed, then AO starts,
   which also starts the counter.

**Stop order:**

1. `engine.request("camera_trigger", on=False)`, so the camera free-runs
   again.
2. `daq.request("stop")`.

On a DAQ error, the window does the same two steps in the same order.

### Settings (`DaqSettings`)

| Field | Meaning |
|---|---|
| `device` | DAQmx device name, `"Dev1"` |
| `sync_mode` | `off`, `daq_master`, `camera_master`, `software` |
| `drive_on`, `drive_freq_hz`, `drive_amp_v`, `drive_offset_v` | AO 0 waveform, in DAQ volts (before the amplifier) |
| `drive_limit_v` | user-set ceiling for \|AO 0\|, below ±10 V, chosen for the amplifier and transducer |
| `piezo_mode` | `off`, `dc`, `step` (N-step per frame), `sine_locked` (reference modulation at f_drive), `sine_offset` (f_drive + Δ) |
| `piezo_offset_v`, `piezo_amp_v`, `piezo_phase_deg`, `piezo_offset_hz`, `piezo_limit_v` | AO 1 waveform and its limit |
| `piezo_steps`, `piezo_volts_per_pi` | step count, and the calibration (V at AO for a π phase change) |
| `frames_per_cycle_group` (M) | f_frame = f_drive / M in locked modes |
| `exposure_policy` | `camera` (leave ExposureTime alone), `whole_cycles` (snap Te to n/f), `trigger_width` |
| `strobe_on`, `strobe_duty`, `strobe_phase_deg`, `strobe_line` | DO gate for an AOM |
| `trigger_term`, `camera_line` | `"/Dev1/PFI4"`, `"Line0"` (or `"Line2"`) |
| `ramp_s` | amplitude ramp on start, stop and retune (default 0.02 s) |

### Validation (`plan_timing`)

`plan_timing` returns the coerced plan, or a list of problems. The UI shows
the exact values that will be applied.

- **AO rate.**
  - f_s = 20 MHz / d, with d a whole number. f_s ≤ 2.0 MS/s with both
    channels (2.86 MS/s with one) [S1].
  - Samples per drive cycle: at least 10, warning below 20.
- **Buffer.**
  - It holds a whole number of drive cycles and, in `step` and
    `sine_offset` modes, a whole number of step sequences or beat periods.
  - It must fit in 8,191 samples shared, for FIFO regeneration. Otherwise
    use host regeneration, after checking that f_s × channels stays within
    USB streaming.
- **Locked frame rate.**
  - f_frame = f_drive / M, so the frame period is M·N·d ticks. It must stay
    within the camera's frame-rate range (1–59 fps [S4]).
  - It must respect the readout: with `TriggerOverlap=Off`,
    Te + readout < T_frame.
- **Exposure.**
  - `whole_cycles`: Te = round(Te·f)/f, with n ≥ 1. Warn below 5 and
    advise 50 or more.
  - Changing ExposureTime away from the 8000 µs default must be shown to
    the user and logged, as AGENTS.md requires.
- **Phase stepping.**
  - The step-and-settle time must be ≤ T_frame − (readout + Te). Use the
    camera's `AdcBitDepth`, ROI height and exposure; refuse when there is
    no gap.
  - The step size comes from `piezo_volts_per_pi`.
- **Strobe.** duty ≤ 0.1; the gate phase resolution is 1/N of a cycle.
- **Limits.**
  - \|offset\| + amp ≤ `*_limit_v`, never just ±10 V.
  - f_drive ≤ an amplifier bandwidth the user enters.
  - Piezo current estimate 2π·f·C·V ≤ the amplifier limit, when C and
    gains are entered.

### Missing package, missing device, demo mode and tests

- **`nidaqmx` not installed:** the panel shows "NI-DAQmx support not
  installed (`pip install "nidaqmx>=1.4,<1.8"`)".
- **`DaqNotFoundError`:** "NI-DAQmx driver not installed".
- **No device listed:** "No NI DAQ found". In every case, the app runs
  exactly as today.
- **`--check`** reports the package and driver versions, and the devices.
- **Demo mode** uses `SimulatedDaq`. It runs the same plan and logs what it
  would apply, and its status reads **"DAQ: simulated, no hardware"**, so it
  can never pass for a real DAQ.
  - Done (2026-11-21): `simulation.DemoSource`'s **Follow DAQ** object renders
    the exposure-averaged fringes of a simulated drumhead from the applied
    drive (generator frequency and amplitude, piezo level or sine); see
    `docs/notes/simulated-espi-signal-design.md`.
- **Tests.** Add `tests/fake_nidaqmx.py`, an in-memory stand-in installed
  into `sys.modules` the way `fake_pyspin.py` is. It needs:
  - `nidaqmx.Task` with `ao_channels`, `co_channels`, `do_channels`,
    `timing` (rate coercion to 20 MHz / d), `triggers.start_trigger`,
    `out_stream.regen_mode`, `write`, `start`, `stop` and `close`;
  - `nidaqmx.system.System.local().devices`;
  - `nidaqmx.constants`;
  - `nidaqmx.errors.DaqError`, `DaqNotFoundError`;
  - injectable failures.

  `plan_timing` gets table tests built from the numbers in section 5. The
  suite stays camera-free, network-free and at 100 % line coverage.

### Logging and safe shutdown

- **Logging.** One INFO line per apply, with the *coerced* values read
  back from DAQmx: `samp_clk_rate`, the counter high and low ticks, the
  resulting frame period, Te, n, M, buffer length and regeneration mode,
  amplitudes, θ, and the routing terminals. Counter 1's ExposureActive
  count is compared with the triggers sent every second, and a mismatch is
  a WARNING. A DAQ error logs once, at ERROR, through the error reporter.
  Expected states ("no DAQ found") are WARNING only and go on the tested
  `EXPECTED` allowlist.
- **Shutdown.** Stop, error, window close, `fatal` and `stopped` all run
  the same sequence:
  1. ramp AO to 0 over `ramp_s` (when the task is healthy);
  2. `stop()` the tasks;
  3. `close()` them;
  4. **open a fresh on-demand AO task and write 0.0 V to both channels**.

  M Series AO holds the last generated value after a stop.
  `ao_idle_output_behavior` exists in the API, but is likely not supported
  on M Series **(unverified)**.
- **Exit and update.** Register this zeroing in the shutdown that
  `qt_app.finish_update` waits on, so an update never relaunches with the
  amplifiers driven.
- **Power-on glitch.** The DAQ itself glitches 1.5 V for 1.5 s at power-on
  [S1]. The written lab procedure must be: **power the DAQ, start the app,
  then switch on the amplifiers. Switch the amplifiers off first.**

---

## 7. Pitfalls

- **Counter budget.** There are only two counters [S1].
  - Camera trigger plus ExposureActive counting uses both, which leaves no
    counter for a counter-clocked AO sweep or a finite pulse train (a
    finite pulse train needs both [S2]).
  - Use the DO-on-`ao/SampleClock` route for strobes and step markers.
- **One hardware-timed AO task per device [S5].** Drive and piezo must be in
  one task with one sample clock. NI says "most devices" also allow a
  software-timed task alongside **(unverified on the 6251)**, but keeping
  AO 1 in the same task is the design anyway.
- **FIFO and underflow.**
  - Two channels share 8,191 samples [S1].
  - Host-buffer regeneration and non-regeneration depend on USB Signal
    Stream. A busy USB bus, or the Blackfly's own traffic on the same root
    hub at up to about 371 MB/s at full rate
    (`docs/camera/acquisition-and-performance.md`), can starve a
    non-regenerating stream.
  - Put the DAQ and the camera on different USB host controllers. Prefer
    FIFO regeneration.
- **Frequency-change glitches.** See the table in section 3. Always ramp the
  amplitude, and never jump a shaker from full amplitude to a held DC level.
- **The default counter output has no BNC.** PFI 12 is screw-terminal only
  on the BNC model [S2]. Route the counter to PFI 0–7.
- **Ground loops.**
  - The DAQ and the camera share the PC's USB ground, and the amplifiers
    add mains earth through the AO BNC shields.
  - Use the opto input (Line 0) so the camera's ground stays separate.
  - Use amplifiers with differential or floating inputs, or a ground-lift
    or isolation transformer on the drive line.
  - Bond everything at one star point. Hum at 60 Hz or 120 Hz on the drive
    shows up as fringe "breathing".
- **Piezo capacitance and amplifier limits.** These are current-limited
  before they are slew-limited [S22]. Size the amplifier with
  I = 2π·f·C·V_pk, adding the amplifier's own output capacitance [S21].
  Stacked piezos with a mirror have mechanical resonances (often a few kHz);
  stay well below them for stepping.
- **Rolling shutter.** Phase stepping needs a gap with no row exposing; see
  section 5. The 14-bit ADC removes the gap at 30 fps.
- **Frame drops.** `NewestOnly` drops frames, which breaks step order and
  frame counting. Use FrameID chunks for measurement modes.
- **The opto input's delay is asymmetric:** 13.4/9.2 µs (family data
  [S17]). It biases `TriggerWidth` exposures by a few µs. Use Line 2 when
  that matters.
- **The DAQ needs its 11–30 V supply [S1]**, and the AO power-on glitch (1.5
  V) is passed on through any amplifier that is switched on.
- **PyInstaller and lazy import.** `nidaqmx` must be declared as a hidden
  import or collected in `build-binaries.yml` and `quick-release.yml`. The
  driver itself is a separate NI install on the camera PC.
- **The XC-ST50 through the USB-Live2:**
  - The stick has no frame or trigger metadata, and DirectShow's latency is
    unknown, so the program cannot match captured frames to triggers. That
    rules out per-frame phase stepping with this camera unless a marker
    (for example one dark or blank step) is detected in the images.
  - The camera's Restart/Reset mode produces irregular VD and non-standard
    video, which the decoder may not lock to **(unverified)**.
  - The camera is non-interlaced under external sync [S23], which the
    decoder may also mishandle **(unverified)**.
  - Prefer trigger-shutter **Mode 1** with continuous internal VD [S23]: the
    video timing stays standard, and only the exposure moves.

---

## 8. Cabling and parts list

**Blackfly (strategy a), USB-6251 BNC:**

| From | To | Cable or part | Notes |
|---|---|---|---|
| AO 0 BNC | drive amplifier input (shaker power amp or PZT amp) | 50 Ω BNC | set `drive_limit_v` to the amplifier's rated input |
| AO 1 BNC | reference-piezo amplifier input | 50 Ω BNC | low-noise piezo amp; current per section 2 |
| PFI 4 BNC (counter 0 output, routed) | Blackfly pin 2 (Line 0, opto in) | BNC to flying lead → Hirose 6-pin GPIO cable | shield/D GND → pin 5 (Opto GND). Connect Opto GND first [S16] |
| Blackfly pin 4 (Line 1, ExposureActive) | PFI 0 BNC (counter 1 source/gate, or DI) | same GPIO cable | 1 kΩ pull-up from pin 4 to DAQ +5 V (screw terminal); pin 5 to D GND |
| P0.0 (screw) → USER 2 BNC | AOM driver TTL gate | BNC | strobe only; DO clocked by `ao/SampleClock` |
| DAQ power jack | 11–30 VDC, 20 W supply | NI adapter | required [S1] |
| DAQ USB | PC port on a different host controller from the camera | USB 2.0 | |

**XC-ST50 (strategy b), through the DC-700:**

| From | To | Notes |
|---|---|---|
| DC-700 HD BNC (camera HD/VD switch on INT) | PFI 1 BNC (AO sample clock) | f_drive = 15,734.26 Hz / N. Check the output level ≥ 2.2 V |
| DC-700 VD BNC | PFI 2 BNC (AO start trigger) | |
| PFI 4 BNC (optional trigger shutter) | DC-700 TRIG BNC (12-pin pin 11) | + polarity: 2–5 V high, 0–0.6 V low, ≥10 kΩ input, width 2 µs–1/4 s; Te = width + 97 µs (EIA) [S23] |

**Parts to have:**

- The NI USB-6251 (BNC or screw-terminal) and its power supply.
- BNC cables.
- A Teledyne FLIR 6-pin GPIO cable (Hirose HR10A-7P-6S(73) plug [S14]) with
  flying leads, and a small screw/BNC breakout box.
- A 1 kΩ resistor.
- A shaker or PZT exciter **power amplifier**, matched to the transducer's
  impedance and bandwidth.
- A **piezo amplifier** for the reference mirror. Its gain, output range,
  current and bandwidth come from the section 2 formula.
- Optionally an AOM and driver, for stroboscopic work.
- Optionally a passive RC reconstruction filter for drive frequencies above
  about 20 kHz.
- Ground-lift or isolation parts if hum appears.

---

## 9. Open questions for the lab

1. Which USB-6251 variant is on hand: **BNC**, screw-terminal, mass
   termination or OEM? Is its **power supply** there?
2. Which **amplifiers** does the lab own, for the shaker or PZT exciter and
   for the reference piezo? I need their gain, input range and impedance,
   whether the input is single-ended or differential, the output voltage
   and current, and the bandwidth.
3. **The specimen exciter:**
   - Is it an electrodynamic shaker or a PZT? Give its impedance and
     capacitance and its usable frequency range.
   - Which frequencies matter most: below about 500 Hz, 1–10 kHz, or
     above?
4. **The reference piezo:**
   - Model, stroke per volt, capacitance and mechanical resonance with the
     mirror fitted.
   - The beam geometry (angle of incidence) and the **laser wavelength**.
     These set the λ/8 step voltage.
   - Is there a phase-step mirror in the rig at all? This is still open
     from PR #63.
5. Which ESPI modes are wanted first: time-averaged (live J0²),
   reference-modulated (amplitude and phase), stroboscopic (does the lab
   have an AOM or a modulatable laser?), or phase-stepped?
6. **The Blackfly:**
   - What do `AdcBitDepth`, `SensorShutterMode` and `TriggerOverlap` read
     on the camera?
   - Is a GPIO cable on hand?
   - Is it acceptable to change the frame rate to f_drive / M (for example
     30.303 fps) in locked modes, and to change the 8000 µs exposure to a
     whole number of drive cycles?
7. **The XC-ST50:**
   - Are the DC-700's HD/VD/TRIG BNCs accessible?
   - What do HD and VD measure on a scope when the camera outputs them?
   - Which trigger-shutter or DIP settings are currently set?
8. **The camera PC:** which NI-DAQmx version is installed (if any)? Does NI
   MAX see the device, and does it pass self-test? Are the camera and the
   DAQ on separate USB controllers?
9. **Safety:** what are the maximum safe AO voltages for each amplifier and
   transducer combination? These become `drive_limit_v` and
   `piezo_limit_v`.

---

## Sources

NI hardware and drivers

- [S1] NI 6251 Device Specifications, 375213C-01, June 2016 (NI document; copy hosted by Farnell): <https://www.farnell.com/datasheets/3184444.pdf>. Product page: <https://www.ni.com/en/support/model.usb-6251.html>
- [S2] NI M Series User Manual, 371022L-01, July 2016. Used: ch. 5 (AO timing, regeneration), ch. 7 (counter routing, default pins), ch. 3 (USER 1/2), App. A Fig. A-16 (USB-6251 BNC panel), Table A-4 (default counter pins). Canonical: <https://ni.com/docs/en-US/bundle/pci-pcie-pxi-pxie-usb-62xx-features/resource/m-series-user-manual.pdf>; copy read: <https://www.apexwaves.com/pdf/manuals/TBX-68/m-series-user-manual.pdf>
- [S5] NI KB, "Error -50103 When Using NI-DAQmx" (one hardware-timed AO task per E/M/S/X device): <https://digital.ni.com/public.nsf/allkb/25EFE6592CFD36538625818C004D2437>
- [S6] NI-DAQmx 2023 Q2 Readme (supported OS, Windows 11): <https://ni.com/pdf/manuals/ni-daqmx-2023-q2.html>
- [S7] NI-DAQmx 19.1 Readme (NI 6251 USB in the device tables): <https://download.ni.com/support/manuals/374768ac.html>
- [S8] NI, Obsolescence Planning for Several Legacy Product Families: <https://www.ni.com/en/support/documentation/supplemental/17/obsolescence-planning-for-several-legacy-product-families.html>
- [S11] NI, Analog Output Regeneration in NI-DAQmx: <https://ni.com/en/support/documentation/supplemental/06/analog-output-regeneration-in-ni-daqmx.html>
- [S12] NI KB, Generate a Continuous Digital Pulse Train With On The Fly Frequency Control: <https://digital.ni.com/public.nsf/allkb/46982EDD6D95E9A2862577A300450162>
- [S13] NI forum, dynamically setting AO frequency and amplitude (regeneration; on-the-fly rate change; not authoritative): <https://forums.ni.com/t5/Multifunction-DAQ/9264-dynamically-set-frequency-and-amplitude/td-p/3838453>
- [S24] NI, Signal Streaming on external buses (USB latency is higher than PCI): <https://www.ni.com/tutorial/4636/en/>
- [S25] Third-party measurement of USB DAQmx call overhead (about 0.6–0.84 ms per call; **unverified**): <https://industrialmonitordirect.com/ja/blogs/knowledgebase/reducing-labview-daqmx-loop-time-below-1ms-for-force-control>

Python

- [S9] `nidaqmx` on PyPI (1.7.0, 2026-10-26, requires Python ≥ 3.9, 3.10 classifier): <https://pypi.org/project/nidaqmx/> and <https://pypi.org/pypi/nidaqmx/1.7.0/json>. Repository README (CPython 3.11+ on master; `installdriver`): <https://github.com/ni/nidaqmx-python>. CHANGELOG (unreleased 1.8.0 removes Python 3.9 and 3.10): <https://github.com/ni/nidaqmx-python/blob/master/CHANGELOG.md>
- [S10] `nidaqmx-python` 1.7.0 source and examples (`add_co_pulse_chan_ticks`, `co_pulse_term`, `ao_use_only_on_brd_mem`, `regen_mode`, `cfg_dig_edge_start_trig`, `CtrFreq`, `errors.DaqNotFoundError`; `examples/analog_out/cont_gen_voltage_wfm_int_clk.py`, `examples/counter_out/cont_gen_dig_pulse_train*.py`): <https://github.com/ni/nidaqmx-python/tree/1.7.0>

Teledyne FLIR Blackfly S

- [S3] Sensor shutter types: rolling shutter with global reset, strobe windows (BFS-U3-23S6 page, family text): <https://softwareservices.flir.com/BFS-U3-23S6/latest/40-Installation/Readout.htm>. Knowledge base: <https://flir.com/support-center/iis/machine-vision/knowledge-base/key-differences-between-rolling-shutter-and-frame-global-shutter>
- [S4] BFS-U3-63S4 specification and frame-rate table (59 fps at 10-bit ADC, 30 fps at 14-bit; 1 opto in and 1 opto out; 1 bidirectional and 1 input non-isolated): <https://softwareservices.flir.com/BFS-U3-63S4/latest/Model/spec.html>
- [S14] BFS-U3-63S4 Digital IO Control (6-pin Hirose pinout, LineSource values): <https://softwareservices.flir.com/BFS-U3-63S4/latest/Model/public/DigitalIOControl.html>
- [S15] BFS-U3-63S4 Acquisition Control (TriggerSelector, Source, Activation, Overlap, Delay, ExposureMode Timed/TriggerWidth, SensorShutterMode Rolling/GlobalReset): <https://softwareservices.flir.com/BFS-U3-63S4/latest/Model/public/AcquisitionControl.html>
- [S16] Blackfly S (BFS-U3-200S6) GPIO Electrical Characteristics (opto input 0–30 V, output 24 V / 25 mA, Opto GND warning, pull-up table): <https://softwareservices.flir.com/BFS-U3-200S6/latest/Family/ElectricalGPIO.htm>. Pin table with input thresholds and currents: <https://softwareservices.flir.com/BFS-U3-200S6/latest/Family/InputOutputControl.htm>
- [S17] Blackfly S (BFS-U3-200S6) Input Timing Characteristics: <https://softwareservices.flir.com/BFS-U3-200S6/latest/Family/TimingInput.htm>
- [S18] BFS-U3-63S4-BD2 (board-level) GPIO unit test (a different, non-isolated connector; context only): <https://softwareservices.flir.com/BFS-U3-63S4-BD2/latest/50-Quality/GPIOTest.htm>
- [S19] Blackfly S (BFS-U3-200S6) Output Timing Characteristics: <https://softwareservices.flir.com/BFS-U3-200S6/latest/Family/TimingOutput.htm>

Sony XC-ST50

- [S23] Sony XC-ST30/ST50/ST51 User's Guide: 12-pin assignments by mode, HD/VD 2–5 Vp-p, ±1 % HD tolerance, trigger pulse levels and widths, Mode 1/Mode 2, Te = width + 97 µs, non-interlaced under external sync: <https://www.saphir.fr/pdf/Technical_product_documentation/Sony_XC-ST_User_Manual.pdf>. Sony's copy: <https://www.sony.jp/camera-module/download/manual/past_data/XC-ST30_ST30CE_ST50_ST50CE_ST51_ST51CE_UGJ_V2.pdf>

Piezo and amplifier

- [S20] PiezoDrive, piezo current calculators (I ≈ C·dV/dt; I_max = V_pp·π·C·f): <https://www.piezodrive.com/calculators/>
- [S21] Advanced Energy (Trek), "HV amplifiers and capacitive loads" (I_peak = (C_load + C_int)·π·f·V_pp): <https://www.advancedenergy.com/getmedia/edf34b8d-964e-4bd8-9445-5efb5898cb44/en-hv-amplifier-capacitive-loads.pdf>
- [S22] Apex Microtechnology AN25, driving capacitive loads (slew = 2π·f·V_pk): <https://www.apexanalog.com/resources/appnotes/an25u.pdf>

ESPI literature

- [S26] O. J. Løkberg and K. Høgmoen, "Use of modulated reference wave in electronic speckle pattern interferometry," J. Phys. E 9, 847 (1976).
- [S27] K. Høgmoen and O. J. Løkberg, "Detection and measurement of small vibrations using electronic speckle pattern interferometry," Appl. Opt. 16(7), 1869–1875 (1977), doi:10.1364/AO.16.001869
- [S28] K. Høgmoen and H. M. Pedersen, "Measurement of small vibrations using electronic speckle pattern interferometry: theory," J. Opt. Soc. Am. 67(11), 1578–1583 (1977)
- [S29] O. J. Løkberg and K. Høgmoen, "Vibration phase mapping using electronic speckle pattern interferometry," Appl. Opt. 15(11), 2701 (1976)
- [S30] G. Å. Rosvold and O. J. Løkberg, "Effect and use of exposure control in vibration analysis using TV holography," Appl. Opt. 32(5), 684 (1993), doi:10.1364/AO.32.000684
- [S31] J. D. Valera, J. D. C. Jones et al. (1994), "Whole-field vibration phase measurement with electronic speckle pattern interferometry (ESPI)" (stroboscopic ESPI): <https://researchportal.hw.ac.uk/en/publications/whole-field-vibration-phase-measurement-with-electronic-speckle-p-2/>
- [S32] "Vibration modal analysis using stroboscopic digital speckle pattern interferometry," Proc. SPIE 1554 (1991) (CW laser chopped by an AOM): <https://ebooks.spiedigitallibrary.org/conference-proceedings-of-spie/1554/0000/Vibration-modal-analysis-using-stroboscopic-digital-speckle-pattern-interferometry/10.1117/12.49540.full>
- [S33] K. Creath, "Phase-shifting speckle interferometry," Appl. Opt. 24(18), 3053–3058 (1985), doi:10.1364/AO.24.003053
- [S34] S. Nakadate and H. Saito, "Fringe scanning speckle-pattern interferometry," Appl. Opt. 24, 2172 (1985), doi:10.1364/AO.24.002172 (four frames at 0, π/2, π, 3π/2)
- In the repo: `docs/camera/sony-xc-st50-usb-live2.md` (interline-transfer modes, Opt. Eng. 38(9), 1999), `docs/camera/acquisition-and-performance.md` (trigger configuration order), and `docs/labview/README.md`.
