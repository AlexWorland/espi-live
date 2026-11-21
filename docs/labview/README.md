# LabVIEW ESPI 1.5 (known-good flow for the Sony XC-ST50)

`ESPI 1.5 modded.vi` is the LabVIEW program the lab used with the analog
Sony XC-ST50 before this viewer existed. The camera fed an NI frame grabber
directly (NI-IMAQ, interface `img0`), with no USB capture adapter. It is the
reference for how the analog camera should look. It is published here with
permission.

- `ESPI 1.5 modded.vi`: the VI as received. Saved in LabVIEW 2013 (13.0);
  needs NI Vision Development Module and NI-IMAQ (`imaqhl.llb`, the legacy
  frame-grabber driver, not IMAQdx). The block diagram has no password.
- `espi-1.5-block-diagram.png`: its block diagram, from Rand Worland's
  slide *ESPI LabVIEW VI 2025* (24 November 2025).

The description below was decoded from the VI with
[pylabview](https://github.com/mefistotelis/pylabview) (front-panel and
block-diagram heaps, link tables) and checked against the diagram image.

## The flow

The diagram is a two-frame sequence.

**Frame 0: the reference.** Open the grabber (`IMAQ Init "img0"`), create a
buffer with the **Image Type** control, grab one image, close the session.
That image is the Reference baseline; it is never recaptured while the VI
runs.

**Frame 1: the live loop.**

1. Open the grabber again and create `current image` (8 bits), `last image`,
   `final image` and `Average Image`. A file dialog asks for an AVI path and
   `IMAQ AVI Create` opens it at 30 fps, whether or not **Save AVI** is on.
2. An outer while loop holds an inner loop, one pass per frame:
   - Grab, then wait **milliseconds to wait** (`Wait (ms)`).
   - `IMAQ Absolute Difference` of the frame against the startup reference,
     or, with **BY FRAME** on, against the previous pass's frame. A sequence
     makes the difference run before `IMAQ Copy` replaces the previous frame.
     Because of the wait, "previous" is the frame from one loop pass ago, not
     the camera's preceding frame.
   - `IMAQ Multiply` by **Constant** (SGL, default 42). On 8-bit images the
     product saturates at 255: this is a fixed gain, not a stretch. At 42 any
     difference of 7 grey levels or more is white.
   - **Low Pass Filter** (optional): `IMAQ LowPass` with a constant
     *Size & Tolerance* cluster that reads `1, 1, 0` (1 × 1 kernel, 0 %), so
     as configured it changes nothing.
   - **Average** off: draw the frame with the **Palette** in window 0 ("ESPI")
     and, with **Show ESPI 2**, window 5 ("ESPI 2"); with **Save AVI** on, write
     it to the AVI.
   - **Average** on: `IMAQ ImageToArray`, convert to DBL and add to a
     shift-register sum.
   - The inner loop stops when its iteration counter i equals **NUMBER OF
     FRAMES**, or (with Average on) equals **Number of frames to average** N,
     or on **stop** or an error. The test runs after the frame is added, so a
     block holds N + 1 frames. With N above NUMBER OF FRAMES (30) the loop
     stops first, which is why the control is labelled "< 31".
3. With **Average** on: sum × **Multiplier** ÷ N (not N + 1, so the result is
   (N+1)/N too bright), `To Unsigned Byte` (rounds and saturates),
   `IMAQ ArrayToImage`, draw in window 4 and write to the AVI. The next outer
   pass starts a fresh sum, so this is a block average.
4. **Save Frame** ends the outer loop. The VI then closes the grabber, runs
   the error handler, closes the AVI, and writes the last image to a JPEG
   (its file dialog has no data dependency, so it probably appears at start).

Frames are 640 × 480 (the X and Y Size constants).

### Saved front-panel defaults

Read from the `DefaultData` of each control in the front-panel heap (a
control without one defaults to zero or off):

| Control | Default |
|---|---|
| Constant | 42.0 |
| Multiplier | 1.0 |
| Number of frames to average | 30 |
| NUMBER OF FRAMES | 30 |
| Show ESPI 2 | on |
| BY FRAME, Average, Low Pass Filter, Save AVI, Save Frame, stop | off |
| milliseconds to wait | 0 |
| Palette | 0 (Gray) |
| Image Type | 0 (8 bits) |

The operator may have changed any of these while the VI ran; these are the
values it opens with.

### Not used

A *Waveform Array To Dynamic* Express VI (labelled "DAQ Assistant2") feeds a
*Tone Measurements* Express VI (amplitude and frequency, about 10 Hz ± 5 %),
whose result goes to a `Format Into String "%4.1f"` with an unwired output.
It computes nothing that is shown or saved.

### Not in the VI

The grabber's own settings (field or frame mode, gain, offset, lookup table)
were in the NI MAX camera file for `img0`. The VI's default-data block
(`DFDS`) is undocumented and pylabview cannot parse it for this file (it
fails on the IMAQ image type), but the front-panel defaults above do not
depend on it.

## How it maps to the app

| LabVIEW ESPI 1.5 | ESPI live |
|---|---|
| Reference grabbed at start | Reference view; **LabVIEW flow** captures one when turned on |
| BY FRAME | Previous view; **Frames back** stands in for the wait between grabs |
| `IMAQ Multiply` by Constant (42) | Scaling **Fixed gain**, **Difference gain** (42 by default) |
| `IMAQ LowPass` | **Outlier filter** (NI's Lowpass rule; off by default, as the VI's 1 × 1 setting was in effect) |
| Average, Number of frames to average | **Average N** (block) |
| Gain before the sum; N + 1 frames × Multiplier ÷ N to U8 | **Gain before averaging (LabVIEW order)**, **Average multiplier** (also sums N + 1 frames and divides by N) |
| Full frames from the grabber | **Single field** off (the default) |
| Palette | None: the viewer's display is always grayscale, the VI's setting (Gray) |
| Save AVI | Record AVI |
| Save Frame (ends the run, writes a JPEG) | Save frame (keeps running, writes a PNG) |

The **LabVIEW flow** toggle in the ESPI measurement section switches the
displayed camera to this processing with the saved defaults above and back to
the user's own settings. It is on by default for the Sony XC-ST50. It sets
Reference view and captures a reference at once (and whenever Reference view
has none), full frames, Fixed gain 42 applied to each frame, the outlier
filter off, Average N 1, Average multiplier 1, Frames back 1, and turns
off everything the VI does not have (noise reduction, noise threshold,
beam-brightness correction, speckle filter, peak hold, local normalization,
stretch, invert, gamma, display brightness, gain and contrast). Exposure, gain, frame rate, recording and the image
flip are not changed.

What it cannot reproduce:

- The capture hardware (below), and 640 × 480 frames (the USB-Live2 gives
  720 × 480).
- The VI's windows: the second "ESPI 2" window, and the separate window for
  averages. The app shows one image.
- **Save Frame** ending the run with a JPEG, the AVI file created on every
  run, and averaged frames written to it whether or not Save AVI is on. The
  app records and saves only when asked.
- The VI's reference is captured once per run; the app also captures one
  when Reference view has none (after switching fields, for example).
- NUMBER OF FRAMES only restarts the VI's loop (and caps averaging at 30
  frames); it has no visible effect, so it has no control here.

The capture hardware still differs. The NI grabber digitised the monochrome
video directly; the Hauppauge USB-Live2 is a colour NTSC decoder (chroma
filtering, YUY2, then grey), which can remove speckle-sized detail near the
3.58 MHz colour subcarrier (about 3.8 pixels per cycle at its sampling rate).
No processing setting undoes that. See
`docs/camera/sony-xc-st50-usb-live2.md`.
