# Driving the reference piezo's amplifier (Thorlabs MDT694A) from the NI USB-6251

How the lab's piezo chain was wired, what each instrument accepts, and how to
wire the NI DAQ so ESPI live drives the piezo. Written 2026-11-22 from bench
photos and the instruments' documentation; the sources are listed at the end.

## The chain before the DAQ

```
Wavetek Model 19  MAIN OUT ──BNC──> Thorlabs MDT694A  EXT INPUT (0–10 V)
                                    Thorlabs MDT694A  OUTPUT ──> piezo
```

- **Wavetek Model 19** (2 MHz sweep/function generator), photographed showing
  **0.281 Hz**, symmetry knob centred.
- **Thorlabs MDT694A** (single-channel open-loop piezo controller), **150 V**
  limit LED lit, display 32.4 V at the moment of the photo (one point on the
  ramp).
- **Scope** (Tektronix TBS1102B, CH1 DC coupled, 1X probe, 5 V/div, 1 s/div,
  scan mode) on the Wavetek's output: a symmetric triangle with a period of
  about 3.6 s, from about +1 V up to about +10.5 V read off the screen. The
  person at the bench measured it as just under 10 V from bottom to top. The
  DC offset keeps it positive, because the MDT694A must never get a negative
  input (below).

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

## Decision: the DAQ drives the MDT694A directly

The DAQ's AO makes the same positive triangle the Wavetek made, at a
frequency set exactly from software, and also the held levels and steps
the Phase view needs. Taking the Wavetek out of the chain gives full control
from ESPI live; keeping it would leave amplitude and offset on its knobs.

```
NI USB-6251  AO 1 BNC ──BNC──> MDT694A EXT INPUT
             MDT694A OUTPUT ──> piezo            (unchanged)
NI USB-6251  AO 0 BNC          free: the frequency generator / tone, later
Wavetek Model 19               out of the piezo chain
```

| NI USB-6251 | Instrument | Notes |
|-------------|------------|-------|
| **AO 1** BNC | MDT694A **EXT INPUT** | The piezo driver's default output |
| AO 0 BNC | (later) the tone / specimen drive | The frequency generator's default output |
| — | MDT694A **OUTPUT** | **Never** to the DAQ: up to 150 V |

### Settings in ESPI live

**Devices → Frequency generator and piezo (NI DAQ)…**, piezo driver:

- **Safety limit**: at most **10 V** (the MDT694A's input range).
- **Mode**: **Triangle at its own frequency**. To reproduce the Wavetek's ramp:
  **Bottom** 0.5 V, **Top** 9.5 V, **Frequency** 0.281 Hz. With the OUTPUT
  ADJ knob at zero, that is about 7.5 V to 142.5 V on the piezo.
- The program refuses any piezo level below 0 V (`daq.PIEZO_FLOOR_V`) in
  every mode, phase steps included, and any level above the safety limit.
- A triangle runs on the AO sample clock that both analog outputs share, so
  it needs the frequency generator off or on a counter (a square wave on a
  PFI terminal). A sine on AO 0 together with a triangle on AO 1 would need
  one buffer holding whole cycles of both, which is not supported.

### Safe order at the bench

1. With the MDT694A off, connect AO 1 to its EXT INPUT; turn OUTPUT ADJ to zero.
2. Plug in the DAQ and start ESPI live; connect the device in the DAQ window.
3. Switch the MDT694A on, then apply the piezo settings.
4. Before unplugging the DAQ: **All outputs to 0 V**, then switch the MDT694A
   off. The DAQ's outputs are undefined for a moment while it powers up.

Keep Top × 15 + the knob's voltage at or below the MDT694A's lit limit
(150 V).

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
