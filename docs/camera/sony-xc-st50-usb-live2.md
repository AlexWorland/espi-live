# Sony XC-ST50 through a Hauppauge USB-Live2

Researched 2026-10-29 for issue #23. The viewer reads this camera through
`src/espi_live/capture.py` (OpenCV) and recognises the adapter in
`src/espi_live/capture_models.py`.

## The chain

```
XC-ST50 ──12-pin──► DC-700 (power, sync breakout) ──BNC VIDEO OUT──► BNC→RCA ──► USB-Live2 composite in ──USB 2/3──► camera PC
```

- **Camera:** Sony XC-ST50, a 1/2" interline-transfer CCD, monochrome, 768 × 494
  pixels (EIA). The CE model is PAL: 752 × 582. It runs 2:1 interlaced at 59.94
  fields/s (30 frames/s); it is non-interlaced only with external sync. Frame
  or field integration can be selected. Gain is AGC, manual or fixed. The
  shutter is set by the rear DIP switch. Power is +12 V DC (10.5–15 V, 2.0 W).
  Nothing about the camera can be controlled from software.
- **Power/sync adapter:** a Sony DC-700 gives 13 V DC, which is within the
  camera's range, and breaks out VIDEO, HD, VD, trigger and WEN to BNC jacks.
  It does not digitize anything.
- **Capture adapter:** Hauppauge USB-Live2 (USB ID `2040:c200`). USB 2.0,
  composite and S-Video in, 720 × 480 for NTSC (720 × 576 for PAL), 4:2:2
  sampling. Hauppauge provides the Windows 11 driver. Linux uses the kernel's
  `cx231xx` driver. There is no macOS driver.

## Setting up the camera PC

1. Install Hauppauge's USB-Live2 driver and plug the stick in.
2. Connect the DC-700's **VIDEO OUT** BNC to the stick's yellow composite
   input. Use a BNC-to-RCA adapter or cable.
3. Make sure the driver is set to **NTSC** for the XC-ST50 (EIA). If
   WinTV / Hauppauge Capture was ever set to PAL, the viewer shows a
   720 × 576 warning. Set it back to NTSC there.
4. **Close WinTV and Hauppauge Capture** before starting the viewer. The
   device serves one program at a time, and opening it fails while they
   run.
5. For ESPI, set the camera's **field integration** switch if it is
   accessible (see below).

The prebuilt release includes `cv2_enumerate_cameras`, which lists video
devices with their names and USB IDs. Source checkouts need
`pip install -r requirements.txt` (the setup script installs it). Without
the package, no video devices are listed and the log says why. `--check`
reports whether it is present.

## How the viewer uses it

- The USB-Live2 appears in the camera list as "Hauppauge USB-Live2 (video N)"
  next to any Blackfly, and works with every processing, saving and
  recording feature. At startup a Blackfly is opened first; with none
  connected, the USB-Live2 is. **Switch to live capture** in demo mode tries
  the same order.
- **Format:** the viewer selects the composite input (`CAP_PROP_CHANNEL` 0)
  and requests 720 × 480. On Windows it uses DirectShow, which converts the
  stick's YUV to BGR. The viewer takes the luma back with `COLOR_BGR2GRAY`,
  which is within about ±1 grey level of the original. OpenCV's DirectShow
  backend cannot deliver raw YUY2.
- **Exposure, gain, frame rate:** the camera's switches set them. Those
  controls are disabled for this source, and the status line shows the
  measured capture rate and the frame size.
- **Single field** (on by default): each frame holds two fields exposed
  1/60 s apart. Any motion between them shows up as line-to-line noise in
  the difference images. Single-field mode keeps the even rows and repeats
  each one, so the frame stays 720 × 480 and saved frames and recordings
  match the screen. This halves the vertical detail. Changing the setting
  clears the reference and the averages. Studies of interline-transfer
  cameras in subtraction ESPI favour non-interlaced field and frame modes
  (Optical Engineering 38(9), 1999). The camera's own field-integration
  setting helps too.
- **Aspect:** NTSC pixels are not square (720 × 480 carries a 4:3 picture),
  so images look about 12 % too wide. They are not resampled, because
  resampling would blur the speckle.

## What to expect for ESPI

Analog capture is coarser than the Blackfly. The image is VGA class,
interlaced, digitized to 8 bits after an analog cable run, and has more noise
and timing jitter than a USB3 Vision camera. Fringes will be visible but less
sharp, and averaging matters more. Keep the camera and sample still between
fields, or use single-field mode.

## Unrecognised video devices

Every video device is listed. A device that is not a recognised adapter
(a webcam, a different capture stick) is marked "not yet supported" and is
never opened automatically. When someone switches one on, it works as
delivered. The viewer files a GitHub issue labelled `camera-support` with
what the device reports: name, USB ID, backend, OpenCV version, format,
properties and first-frame statistics. Only one issue is filed per device
type across all PCs. Copies without a feedback token save the report under
`output/logs/camera-report-*.md` and show a **Report this camera** button.
A device that sends compressed video (MJPG, H.264) gets a warning, because
compression destroys speckle.

## Sources

- Sony XC-ST50/ST50CE manual (specifications, scanning, power):
  <https://www.manualslib.com/manual/905590/Sony-Xc-St50.html?page=5>
- Sony XC-ST user guide: <https://www.sony.jp/camera-module/download/manual/past_data/XC-ST30_ST30CE_ST50_ST50CE_ST51_ST51CE_UGJ_V2.pdf>
- Sony DC-700 manual: <https://www.sony.jp/camera-module/download/manual/past_data/DC-700J_700UC_700CE.pdf>
- Hauppauge USB-Live2: <https://hauppauge.com/pages/products/data_usblive2.html>
- Linux `cx231xx` support (2040:c200): <https://linuxtv.org/wiki/index.php/Hauppauge_USB-Live-2>
- OpenCV DirectShow backend (composite default, `CAP_PROP_CHANNEL`):
  `modules/videoio/src/cap_dshow.cpp`
- `cv2_enumerate_cameras`: <https://pypi.org/project/cv2-enumerate-cameras/>
- "Operation modes of an interline-transfer camera in pulsed subtraction
  video speckle interferometry", Optical Engineering 38(9), 1999,
  doi:10.1117/1.602214
