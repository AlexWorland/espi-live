# Driving the reference piezos' amplifiers (Thorlabs MDT694A) from the NI USB-6251

How the lab's piezo chain is wired, what each instrument accepts, and how to
wire the NI DAQ so ESPI live drives the piezos — one Thorlabs amplifier or
two (for example the front and the back of a sample, one camera each).
Written 2026-11-22 from bench photos and the instruments' documentation;
the one-Thor chain was the original; two-Thor support was added 2026-11-23.
The sources are listed at the end.

## One Thor or two

The NI USB-6251 BNC has exactly **two analog outputs**, AO 0 and AO 1. Each
Thorlabs MDT694A needs one of them, so the choice is:

- **One Thor (Thor 1)** — the original chain. AO 1 drives it; AO 0 stays
  free for the frequency generator (a sine for the specimen's shaker).
- **Two Thors (Thor 1 and Thor 2)** — Thor 1 keeps AO 1 and Thor 2 takes
  AO 0. The frequency generator then has no analog output left: switch it
  off, or put it on a **counter** (a 0–5 V square wave on a PFI terminal),
  which the app supports.

Both amplifiers are identical Thorlabs MDT694As, so every number below
applies to either. In the app, "Piezo driver (Thor 1)" is the first driver
and "Piezo driver (Thor 2)" the second; each is paired to a camera in the
DAQ window ("Thor 1 steps camera" / "Thor 2 steps camera"), and each keeps
its own settings, safety limit and calibration.

### One Thor: what to plug in

```
NI USB-6251                        Thorlabs MDT694A #1
┌──────────────┐   BNC cable   ┌──────────────────────┐
│          AO 1├──────────────>│ EXT INPUT (0–10 V)   │
│          AO 0│  (free: the   │                      │
│              │   generator)  │ OUTPUT (0–150 V)     │──> piezo 1 (reference
└──────────────┘               └──────────────────────┘    mirror, camera 1's beam)
```

### Two Thors: what to plug in

```
NI USB-6251                        Thorlabs MDT694A            camera beams
┌──────────────┐   BNC cable   ┌──────────────────────┐
│          AO 1├──────────────>│ #1 EXT INPUT (0–10 V)│   front of the sample
│              │               │ #1 OUTPUT (0–150 V)  │──> piezo 1 ──> camera 1
│          AO 0├──────────────>│ #2 EXT INPUT (0–10 V)│   back of the sample
│              │               │ #2 OUTPUT (0–150 V)  │──> piezo 2 ──> camera 2
│  PFI 0 (BNC) │<-- counter:   └──────────────────────┘
│              │    the frequency generator's square wave,
└──────────────┘    when the generator is wanted at all
```

Only the two BNC cables AO 1 → EXT INPUT and AO 0 → EXT INPUT are new
work for the second Thor: one 50 Ω BNC cable each, exactly like the first.
The MDT694A's OUTPUT never goes to the DAQ (up to 150 V).

| NI USB-6251 | Instrument | Notes |
|-------------|------------|-------|
| **AO 1** BNC | MDT694A **#1 EXT INPUT** | Piezo driver (Thor 1)'s default output |
| **AO 0** BNC | MDT694A **#2 EXT INPUT** (one-Thor setups: the generator's sine) | Piezo driver (Thor 2)'s default output |
| PFI 0–7 BNC | (two-Thor setups) the specimen drive | The generator as a counter's 0–5 V square wave |
| — | each MDT694A **OUTPUT** | **Never** to the DAQ: up to 150 V |

## What each instrument accepts

### Thorlabs MDT694A

From the MDT694 operating manual (2000; it does not mention the A's front
75/100/150 V limit LEDs, so treat the numbers as close, not certain) and the
MDT693B/694B manual for the limit settings:

| Item | Value |
|------|-------|
| Remote control | No USB. Optional RS-232 (rear DB9, only with the interface card fitted) |
| EXT INPUT | **0 to 10 V**, 10 kΩ, about 40 kHz −3 dB (150 V output, no load), 10 V maximum |
| Gain | **15 V/V** (±5 %) at the 150 V limit; the B model's manual gives 10 V/V at 100 V and 7.5 V/V at 75 V |
| Summing | Output ≈ 15 × EXT INPUT **+ the OUTPUT ADJ knob** (the knob acts as an offset) |
| Negative input | Not clamped: the manual warns a negative input can drive the output to about −15 V. Keep the input at 0 V or above |
| Output | 0–150 V, 60 mA, 50 Ω, < 1.5 mV RMS noise |
| Monitor | Front display only; no analog monitor output |

### Wavetek Model 19

From the Model 19 datasheet:

| Item | Value |
|------|-------|
| Remote control | **None** (no GPIB, RS-232 or USB) |
| MAIN OUT | 50 Ω; sine, triangle, square, DC; 20 Vp-p open circuit, 10 Vp-p into 50 Ω; offset ±10 V open circuit (front knob, detent at 0) |
| AUX OUT | 600 Ω, 0 to 5 V logic square at the main frequency (drives CMOS and 2 TTL loads) |
| SWEEP OUT | 600 Ω, 0 to 6 V triangle |
| AM/SWEEP IN | 10 kΩ. **External sweep: 0 to 3 V for a 1000:1 frequency change** within the selected range. AM: 2 Vp-p for 50 % modulation, DC to 100 kHz |
| Frequency | 2 mHz to 2 MHz in seven ranges; triangle linearity 99 % to 200 kHz |

Amplitude, offset, waveform and range are front-panel controls only. A DAQ
could set the Wavetek's frequency through AM/SWEEP IN and read its output, but
not its amplitude or offset.

### NI USB-6251 (BNC)

See `ni-usb-6251-integration.md`. What matters here: AO 0 and AO 1 on BNCs,
±10 V, ±5 mA, 0.2 Ω, 16-bit, one shared sample clock; AI 0–7 and PFI 0–7 on
BNCs. The MDT694A's 10 kΩ input draws 1 mA at 10 V, well inside the AO's
±5 mA. At power-on the AO may glitch to 1.5 V for 1.5 s, which the MDT694A
would turn into about 22 V on the piezo.

## History: the Wavetek came first

Until 2026-11-22 a Wavetek Model 19 made the piezo's ramp by hand:

```
Wavetek Model 19  MAIN OUT ──BNC──> Thorlabs MDT694A  EXT INPUT (0–10 V)
                                    Thorlabs MDT694A  OUTPUT ──> piezo
```

- **Wavetek Model 19** (2 MHz sweep/function generator), photographed showing
  **0.281 Hz**, symmetry knob centred.
- **Thorlabs MDT694A** (single-channel open-loop piezo controller), **150 V**
  limit LED lit, display 32.4 V at the moment of the photo (one point on the
  ramp).
- **Scope** (Tektronix TBS1102B) on the Wavetek's output: a symmetric
  triangle, period about 3.6 s, from about +1 V to about +10.5 V — "just
  under 10 V bottom to top". The DC offset keeps it positive, because the
  MDT694A must never get a negative input (below).

The DAQ's AO makes the same positive triangle at a frequency set exactly
from software, and also the held levels and steps the Phase view needs, so
the Wavetek left the piezo chain. It has no remote interface; keeping it
would leave amplitude and offset on its knobs.

## Settings in ESPI live

**Devices → Frequency generator and piezo (NI DAQ)…**, per piezo driver:

- **Safety limit**: at most **10 V** (the MDT694A's input range), for each
  driver separately.
- **Mode**: **Triangle at its own frequency** for the slow ramp, or **Hold a
  voltage** for the Phase view's phase steps. To reproduce the Wavetek's
  ramp: **Bottom** 0.5 V, **Top** 9.5 V, **Frequency** 0.281 Hz. With the
  OUTPUT ADJ knob at zero, that is about 7.5 V to 142.5 V on the piezo.
- **Thor 1 / Thor 2 steps camera**: which camera's Phase view that driver
  steps. "Follow the displayed camera" keeps the original one-camera
  behaviour; pick a camera per driver to phase-step two cameras at once.
- The program refuses any piezo level below 0 V (`daq.PIEZO_FLOOR_V`) in
  every mode, phase steps included, and any level above that driver's
  safety limit.
- The two analog outputs share one sample clock. A triangle needs the
  frequency generator off or on a counter. With **both** drivers running
  triangles they must agree on one frequency (the app says so when they
  do not); held levels and phase steps — what the Phase view uses — run
  independently on the two outputs.
- A sine follows the generator's analog output. That is possible for Thor 1
  (generator on AO 0, sine on AO 1) or for Thor 2 alone (generator on
  AO 0, sine on AO 1, Thor 1 off) — but not for both drivers at once,
  and not together with a triangle.

### Safe order at the bench

One Thor:

1. With the MDT694A off, connect AO 1 to its EXT INPUT; turn OUTPUT ADJ to zero.
2. Plug in the DAQ and start ESPI live; connect the device in the DAQ window.
3. Switch the MDT694A on, then apply the piezo settings.
4. Before unplugging the DAQ: **All outputs to 0 V**, then switch the MDT694A
   off. The DAQ's outputs are undefined for a moment while it powers up.

Two Thors: the same order, one amplifier at a time —

1. With both MDT694As off, connect AO 1 → #1 EXT INPUT and AO 0 → #2 EXT
   INPUT; turn both OUTPUT ADJ knobs to zero.
2. Plug in the DAQ and start ESPI live; connect the device in the DAQ window.
3. Switch **one** MDT694A on, apply that driver's settings with only its
   "Piezo on" ticked, and check its piezo behaves; then switch the second
   MDT694A on and apply the second driver.
4. Before unplugging the DAQ: **All outputs to 0 V**, then switch both
   MDT694As off.

Keep Top × 15 + the knob's voltage at or below each MDT694A's lit limit
(150 V), and set each driver's safety limit in the app to what its own
amplifier and piezo tolerate. The safety checklist is asked per switched-on
driver and socket, so wiring the second Thor means ticking its checks too
before its drive signals start.

## Later: the Wavetek alongside the DAQ

If the Wavetek is wanted again (for example as the tone source), the DAQ can
work with it without taking its knobs away:

- MAIN OUT → a BNC tee → AI 0, to record the waveform it really makes;
- AUX OUT (0–5 V logic) → PFI 0, to count its frequency or lock the camera to it;
- AO 0 → AM/SWEEP IN, to set its frequency (0–3 V for 1000:1 within the range,
  with the external sweep selected).

None of this is implemented.

## Sources

- Thorlabs MDT694 operating manual, ManualsLib:
  <https://manualslib.com/manual/3398358/Thorlabs-Mdt694.html?page=4> (pages 3–8)
- MIT wiki, Thorlabs MDT694A: <https://wikis.mit.edu/confluence/x/wQWQCg>
- Thorlabs MDT693B/694B manual (limit settings, rear EXT input):
  <https://manualshive.com/en/manual/thorlabs/mdt693b/1110769?page=11>
- Wavetek Model 19 datasheet (Jefferson Lab):
  <https://hallaweb.jlab.org/tech/Detectors/public_html/manuals/data_sheets-manuals/W/wavetek/w19/19.pdf>
- University of Toronto, Wavetek 19 specifications:
  <https://faraday.physics.utoronto.ca/specs/wavetek19.html> and
  <https://ugdemos.physics.utoronto.ca/db/apparatus/function-generator-wavetek-model-19>
- NI USB-6251: `ni-usb-6251-integration.md` and
  <https://www.ni.com/en-us/support/model.usb-6251.html>
