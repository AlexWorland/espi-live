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
   - `IMAQ Multiply` by **Constant** (SGL). On 8-bit images the product
     saturates at 255: this is a fixed gain, not a stretch.
   - **Low Pass Filter** (optional): `IMAQ LowPass` with a constant
     *Size & Tolerance* cluster that reads `1, 1, 0` (1 × 1 kernel, 0 %), so
     as configured it changes nothing.
   - **Average** off: draw the frame with the **Palette** in window 0 ("ESPI")
     and, with **Show ESPI 2**, window 5 ("ESPI 2"); with **Save AVI** on, write
     it to the AVI.
   - **Average** on: `IMAQ ImageToArray`, convert to DBL and add to a
     shift-register sum.
   - The inner loop ends after **NUMBER OF FRAMES** passes, when
     **Number of frames to average** frames are summed, or on **stop** or an
     error.
3. With **Average** on: sum × **Multiplier** ÷ N, `To Unsigned Byte` (rounds
   and saturates), `IMAQ ArrayToImage`, draw in window 4 and write to the AVI.
4. **Save Frame** ends the outer loop. The VI then closes the grabber, runs
   the error handler, closes the AVI, and writes the last image to a JPEG
   (its file dialog has no data dependency, so it probably appears at start).

Front-panel controls: Image Type, Palette, Constant, Multiplier, Number of
frames to average (< 31), NUMBER OF FRAMES, milliseconds to wait, BY FRAME,
Average, Low Pass Filter, Show ESPI 2, Save AVI, Save Frame, stop, Selected
Path. Frames are 640 × 480 (the X and Y Size constants).

### Not used

A *Waveform Array To Dynamic* Express VI (labelled "DAQ Assistant2") feeds a
*Tone Measurements* Express VI (amplitude and frequency, about 10 Hz ± 5 %),
whose result goes to a `Format Into String "%4.1f"` with an unwired output.
It computes nothing that is shown or saved.

### Not recovered

The default values of **Constant**, **Multiplier** and **milliseconds to
wait** sit in the VI's default-data block (`DFDS`), which NI does not
document and pylabview cannot parse for this file (it fails on the IMAQ image
type). The grabber's own settings (field or frame mode, gain, offset, lookup
table) were in the NI MAX camera file for `img0`, not in the VI.

## How it maps to the app

| LabVIEW ESPI 1.5 | ESPI live |
|---|---|
| Reference grabbed at start | Reference view; **LabVIEW flow** captures one when turned on |
| BY FRAME | Previous view; **Frames back** stands in for the wait between grabs |
| `IMAQ Multiply` by Constant | Scaling **Fixed gain**, **Difference gain** |
| `IMAQ LowPass` | **Outlier filter** (NI's Lowpass rule; off by default, as the VI's 1 × 1 setting was in effect) |
| Average, Number of frames to average | **Average N** (block) |
| Gain before the sum, × Multiplier ÷ N to U8 | **Gain before averaging (LabVIEW order)**, **Average multiplier** |
| Full frames from the grabber | **Single field** off (the default) |
| Palette | Palette |
| Save AVI | Record AVI |
| Save Frame (ends the run, writes a JPEG) | Save frame (keeps running, writes a PNG) |

The **LabVIEW flow** button in the ESPI measurement section switches the
displayed camera to this processing and back to the user's own settings.

The capture hardware still differs. The NI grabber digitised the monochrome
video directly; the Hauppauge USB-Live2 is a colour NTSC decoder (chroma
filtering, YUY2, then grey), which can remove speckle-sized detail near the
3.58 MHz colour subcarrier (about 3.8 pixels per cycle at its sampling rate).
No processing setting undoes that. See
`docs/camera/sony-xc-st50-usb-live2.md`.
