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
- **Capture adapter:** Hauppauge USB-Live2. USB 2.0, composite and S-Video
  in, 720 × 480 for NTSC (720 × 576 for PAL), 4:2:2 sampling (YUY2: bytes
  Y0 U Y1 V, so the brightness is every other byte). Hauppauge provides the
  Windows 11 driver; there is no macOS driver. Two hardware revisions exist,
  and the viewer recognises both:
  - `2040:c200` (model 122000 and earlier): a Conexant CX23102 bridge, run by
    Linux's `cx231xx` driver.
  - `2040:c220` (Rev E1, model 122020; the camera PC's stick): an Empia
    em2828X bridge with an embedded video decoder. Linux support for it is
    new: the 2026 linux-media series "em28xx: Add empia EM2828X chip" adds it
    to `em28xx`, and it is not in older kernels. Check with `lsusb`.

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
the package, no video devices are listed and the log says why (not installed,
or installed but its native module could not load, with the error). `--check`
reports whether it is available, and the release build fails its smoke test
when it is not.

## How the viewer uses it

- The USB-Live2 appears in the camera list as "Hauppauge USB-Live2 (video N)"
  next to any Blackfly, and works with every processing, saving and
  recording feature. At startup a Blackfly is opened first; with none
  connected, the USB-Live2 is. **Switch to live capture** in demo mode tries
  the same order.
- **Format:** the viewer selects the composite input (`CAP_PROP_CHANNEL` 0)
  and requests 720 × 480. It then reads the stick's **raw YUY2 luma** where it
  can, instead of a colour conversion (see "Luma path" below). The log says
  which path it uses (`Analog luma: ...`), and so does a camera report.
- **Exposure, gain, frame rate:** the camera's switches set them. Those
  controls are disabled for this source, and the status line shows the
  measured capture rate and the frame size.
- **Full frames** (the default): both interlaced fields, as the LabVIEW
  ESPI 1.5 VI used them. The fields are exposed 1/60 s apart, but Reference
  and Previous differences compare rows woven the same way, so the offset
  only shows when the scene changes between the two fields (a transient or
  uneven drive), as faint line-to-line texture. **Single field** keeps the
  even rows and repeats each one, so the frame stays 720 × 480 and saved
  frames and recordings match the screen; it halves the vertical detail.
  Changing the setting clears the reference and the averages. Studies of
  interline-transfer cameras in subtraction ESPI favour non-interlaced field
  and frame modes (Optical Engineering 38(9), 1999). The camera's own
  field-integration setting helps too.
- **LabVIEW flow** (on by default for this camera): the LabVIEW ESPI 1.5 VI's
  processing with its saved defaults: Reference view with an automatic
  reference, the difference multiplied by **Difference gain** 42 (the VI's
  Constant) and saturated at 255, no averaging or filters. At 42× the
  capture noise (3–5 grey levels) shows as mid-grey; lower the gain if it
  hides the fringes. Turning the toggle off gives this camera's own settings,
  which also use Fixed gain: Auto stretch would expand the noise to
  full-contrast grain. See `docs/labview/README.md`.
- **Aspect:** NTSC pixels are not square (720 × 480 carries a 4:3 picture),
  so images look about 12 % too wide. They are not resampled, because
  resampling would blur the speckle.

## Luma path

The stick sends YUY2. Earlier versions converted it to BGR (DirectShow on
Windows, OpenCV's `COLOR_YUV2BGR_YUYV` on Linux) and took the grey back with
`COLOR_BGR2GRAY`. The two roundings add about ±1 grey level of error, more
(up to ±4 in a test with chroma noise of 6 codes standard deviation) where
noise in the colour bytes makes a channel clip. The Fixed-gain flow multiplies
differences by 42, so that error was visible. Now:

- **Windows:** OpenCV's DirectShow backend cannot deliver YUY2: for a YUY2
  source it always asks DirectShow's Sample Grabber for RGB24, and
  `CAP_PROP_CONVERT_RGB` changes only Y16 sources (`cap_dshow.cpp`, OpenCV
  5.0.0). Its Media Foundation backend can: with `CAP_PROP_CONVERT_RGB` 0 it
  returns the device's native sample unchanged, as one row of bytes, and no
  video processor (so no deinterlacing) is inserted (`cap_msmf.cpp`). Media
  Foundation cannot select an input, so the viewer opens the stick through
  DirectShow first, as before, to route the composite input and take a
  reference frame, releases it, and re-opens it through Media Foundation
  (matched by device path, else by a unique USB ID). Opening takes about a
  second longer.
- **Linux:** V4L2 with `CAP_PROP_CONVERT_RGB` 0 returns YUYV as
  720 × 480 × 2 on the same device.
- **Checks:** the raw path is used only when the device reports YUY2/YUYV at
  720 × 480, the frame has a YUY2 layout of that size (`capture.yuy2_luma`
  accepts H × W × 2, one byte row with or without row padding, and
  H × 2W), and the mean brightness of the first raw frame is within 6 grey
  levels of the BGR path's reference frame (a wrong input, format or scale
  shows up there). Anything else, including any error, keeps the BGR path,
  exactly as before. Setting the environment variable
  `ESPI_ANALOG_LUMA=bgr` forces the BGR path.
- **Grey-level scale:** decoders send video-range luma (black 16, white 235).
  The BGR conversions expand that to 0–255 (BT.601: `1.164 × (Y − 16)`;
  OpenCV's conversion matches `round((Y − 16) × 255 / 219)` within one level
  for every code). The viewer expands the raw Y with that same table
  (`capture.EXPAND_TABLE`), so the levels the processing sees, and the
  Difference gain 42 of the LabVIEW flow, keep the scale they had. Like
  before, codes above 235 saturate at 255 and codes below 16 read 0; the
  expansion leaves a one-code gap about every sixth level, as the conversion
  did. What changes is only the conversion error. DirectShow's own
  YUY2-to-RGB24 range handling could not be confirmed from documentation;
  `scripts/check_analog_luma.py` measures it (below), and if it turned out
  not to expand, the brightness check above would not catch a mid-grey
  scene, so run the script once.

### Checking it on the camera PC

With ESPI Live, WinTV and Hauppauge Capture closed, the camera on and a still
scene with light and dark areas in view:

```
python scripts\check_analog_luma.py
```

It prints the BGR path's statistics, the raw path's layout, FOURCC and size,
the raw Y range and the share outside 16–235, a same-frame comparison of the
expanded raw Y with OpenCV's YUY2 → BGR → grey, both paths side by side with a
verdict on the scale, and the path ESPI Live itself chooses. `--list` lists
the devices; `--device N` picks one.

## What to expect for ESPI

Analog capture is coarser than the Blackfly. The image is VGA class,
interlaced, digitized to 8 bits after an analog cable run, and has more noise
and timing jitter than a USB3 Vision camera. Fringes will be visible but less
sharp, and averaging matters more. Keep the camera and sample still between
fields, or use single-field mode.

The known-good LabVIEW flow used an NI frame grabber, which digitised the
monochrome video directly. The USB-Live2 is a colour NTSC decoder, and its
chroma filtering can remove speckle-sized detail near the 3.58 MHz colour
subcarrier (about 3.8 pixels per cycle at 720 pixels per line). Processing
cannot restore that detail.

## Unrecognised video devices

Every video device except a laptop's built-in camera is listed (see
`supported-models.md`). A device that is not a recognised adapter
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
- OpenCV 5.0.0 DirectShow backend (composite default, `CAP_PROP_CHANNEL`,
  RGB24 for YUY2): `modules/videoio/src/cap_dshow.cpp`; Media Foundation
  (`CAP_PROP_CONVERT_RGB` 0 returns the native sample):
  `modules/videoio/src/cap_msmf.cpp`; V4L2: `modules/videoio/src/cap_v4l.cpp`
- YUY2 layout: <https://learn.microsoft.com/windows/win32/directshow/yuv-video-subtypes>
- Studio range (16–235) for BT.601/BT.709:
  <https://learn.microsoft.com/windows-hardware/drivers/display/yuv-format-ranges>
- em2828X USB-Live2 revision for Linux ("em28xx: Add empia EM2828X chip",
  linux-media, March 2026): <https://ratatoskr.run/linux-media/2026/03/8466352/t>
- `cv2_enumerate_cameras`: <https://pypi.org/project/cv2-enumerate-cameras/>
- "Operation modes of an interline-transfer camera in pulsed subtraction
  video speckle interferometry", Optical Engineering 38(9), 1999,
  doi:10.1117/1.602214
