# Analog camera support: Sony XC-ST50 through a Hauppauge USB-Live2

Design for issue #23 (and #15, closed as its duplicate). Status: draft for
review; nothing is built yet.

## Intent

The lab's second camera is a **Sony XC-ST50**, an analog monochrome CCD
camera. Its composite video reaches the camera PC through a **Hauppauge
USB-Live2** USB capture stick (it replaced an older NI PCIe frame grabber).
The owner wants the viewer to use it like the Blackfly: pick it in the camera
list, see it live, and use Raw / Previous / Reference, averaging, saving and
recording on it, **alongside** the Blackfly in the same session.

While doing that, the viewer should also learn about video devices it does
not know: when someone switches on an unrecognised video device, the viewer
files a GitHub issue with everything it can read about it (the camera-report
idea from a parked design for FLIR cameras, folded in here for capture
devices; the FLIR part follows in a later PR).

## Confirmed facts

- **XC-ST50** (Sony manual): 1/2" interline CCD, 768 × 494 (EIA; the CE/PAL
  model is 752 × 582), 2:1 interlaced (non-interlaced only with external
  sync), **frame or field integration selectable**, gain AGC/manual/fixed,
  shutter by rear DIP switch, +12 V DC (10.5–15 V, 2.0 W). No software
  control of the camera exists. EIA is assumed; a PAL camera would show up as
  720 × 576 and the report/log will make that visible.
- **USB-Live2:** USB 2.0, composite and S-Video in, 720 × 480 (NTSC) /
  720 × 576 (PAL), 4:2:2 sampling. USB ID `2040:c200`. Hauppauge ships a
  Windows 11 driver; Linux has the kernel `cx231xx` driver; there is no
  macOS driver.
- **Camera PC:** Windows 11, the same PC as the Blackfly, prebuilt release
  (carries the Issues-only `FEEDBACK_TOKEN`).
- **OpenCV** (already a dependency, 5.0): the DirectShow backend selects
  composite by default (`CAP_PROP_CHANNEL` 0 = composite, 1 = S-Video) and
  returns BGR for YUY2 devices; `CAP_PROP_CONVERT_RGB=0` only yields raw data
  for single-byte/Y16 formats, so raw YUY2 is not available there.
  `COLOR_BGR2GRAY` recovers the luma within about ±1 level.
- OpenCV cannot list device names or USB IDs; `cv2_enumerate_cameras`
  (MIT, prebuilt Windows wheels; on macOS it pulls in `pyobjc`) can, and
  gives the index to pass to `cv2.VideoCapture(index, backend)`.

## Goals

- The USB-Live2 appears in the camera list next to any Blackfly and works
  with every existing processing, saving and recording feature, per camera.
- Single-field / full-frame control for interlaced sources, **single field
  by default** (owner's choice).
- Exposure, gain and frame-rate controls are clearly not adjustable for this
  source; the measured capture rate is shown.
- Unrecognised video devices are listed, marked "not yet supported", and
  reported to GitHub once per device type when someone switches one on.

## Non-goals

- NI frame grabbers / NI-IMAQ (no longer used), other vendor SDKs, a generic
  "any analog grabber" layer beyond OpenCV.
- macOS capture support (no USB-Live2 driver); the code just finds no such
  device there.
- Setting the video standard (NTSC/PAL), brightness/contrast (proc amp) or
  crossbar beyond OpenCV's channel property. The Hauppauge driver keeps its
  own standard setting.
- The FLIR per-model profiles and FLIR camera reports (parked design; a later
  PR that reuses this one's report machinery).
- Pixel-aspect correction (NTSC pixels are not square; see Decided).

## Anti-goals

- Anything that changes Blackfly behaviour or its defaults (8000 µs, 0 dB,
  30 fps).
- Opening a webcam or other video device the user did not ask for (startup
  only auto-opens Blackflys and recognised capture adapters).
- Filing reports for webcams that were merely detected.
- Saved frames or recordings that differ from what is on screen (an existing
  test enforces `FramePacket.render()` == display).
- Lossy or resampled data silently passed off as camera data.
- A network call or a blocking dialog on the startup path or the engine
  thread; demo mode filing reports; tests opening devices or the network.

## Constraints (from the repo)

- The engine talks to its source through `open`, `grab(timeout_ms) →
  (frame, status)`, `set_float`, `limits`, `read_values`, `close`,
  `identity`, `info`; only the source's owning thread touches it.
- Source, processing and report code stay free of GUI imports; display colour
  stays only in `processing.apply_palette`.
- Optional platform pieces are imported lazily and guarded (as PySpin is);
  `--demo` and `--check` never open a device.
- 100 % line coverage, `Args: / Returns:` docstrings, `CHANGELOG.md`,
  README and `docs/camera/` kept accurate, PR screenshots of every new UI
  state, no committed media.
- Dependencies live in `requirements.txt`; the Windows build job and
  `scripts/windows/setup_espi_live.ps1` list packages explicitly and must
  gain the new one.

## How it works (decided)

### 1. A capture source next to the Spinnaker one

- A new module holds the capture source; it mirrors `CameraSession`'s
  interface so `Engine`, `Channel`, processing, recording and the window
  stay unchanged apart from choosing the source.
- Backend: DirectShow on Windows, V4L2 on Linux (AVFoundation where it
  exists, for development only).
- Frame size: OpenCV's DirectShow backend opens at 640 × 480 unless told
  otherwise, which would rescale the USB-Live2. The source requests the
  recognised adapter's native size from the table (720 × 480), reads back
  what the device actually delivers, and shows that. A size other than the
  requested one gets a "rescaled by the driver" warning; 720 × 576 also means
  the driver is set to PAL. Unrecognised devices keep their default size.
- Frames: read with `VideoCapture.read()`; converted to Mono8 with
  `COLOR_BGR2GRAY` (or used as-is when already single-channel); a failed read
  is an error status, as a Spinnaker grab error is.
- `limits()` reports nothing adjustable; `set_float` does nothing;
  `read_values()` reports the measured capture rate only.
- `CameraInfo` gains the source kind so channels, the engine and "Switch to
  live capture" open the right session; a stable device id (the
  enumerator's device path, else "video N") takes the serial's place for
  labels and de-duplication.

### 2. Detection and "recognised"

- Detection lists Spinnaker cameras (as today) **and** video devices
  (`cv2_enumerate_cameras`). Either list failing (no PySpin, enumerator
  missing) only empties that list and is logged.
- A table of recognised capture adapters keyed by USB ID, starting with the
  USB-Live2 (`2040:c200`), holds what the source needs (interlaced, input
  channel composite). Name matching ("USB-Live2" / "USB Live2") is the
  fallback when the enumerator gives no USB ID.
- Startup opens the first Blackfly, else the first recognised adapter, else
  offers demo mode (today's flow). Unrecognised video devices are never
  opened automatically. "Switch to live capture" follows the same order.
- Demo mode stays as today: synthetic and real cameras never share the
  list. Real devices (video devices included) appear once a real camera is
  live, as other Blackflys do today (my call).
- An unrecognised video device that someone switches on becomes a normal
  live camera (the owner's rule for unsupported cameras: open with standard
  settings and a visible note), marked "not yet supported". If it delivers
  compressed video (FOURCC MJPG, H.264 and the like) the camera info warns
  that the data is lossy and unsuitable for ESPI measurements.
- The camera list shows every device; unrecognised video devices carry
  "· not yet supported".

### 3. Interlace

- New per-camera setting: **single field** (on by default) / full frame,
  shown only for interlaced sources.
- Single field keeps the even rows and line-doubles them back to full height
  (720 × 480), applied as a pure processing step before differencing, so
  display, saved frames and recordings are identical. Toggling it resets
  Previous, Reference and averaging, as a mode change does.
- The docs recommend setting the camera's field-integration switch.

### 4. Controls and identity for an analog camera

- Exposure, gain and FPS controls are disabled for this source with a
  tooltip saying the camera's switches set them; the status line shows the
  measured capture rate and frame size instead of "Applied: …".
- Camera info names the adapter (for example "Hauppauge USB-Live2, analog
  capture, 720 × 480, interlaced") and notes: exposure/gain/rate are set on
  the camera; single-field mode halves vertical detail; analog capture is
  noisier than the Blackfly, so fringes are coarser.
- Pixels are kept as delivered (720 × 480, non-square), so images look about
  12 % wide; resampling would blur the speckle. Documented, not corrected.

### 5. Camera reports for unrecognised video devices

- Trigger: an unrecognised video device is **switched on** (opened); never
  on mere detection, never for recognised adapters, Blackflys or demo mode.
- Gathered on the engine thread after open: name, USB ID, device path,
  backend, OpenCV version, negotiated width/height/fps/FOURCC, channel, the
  values OpenCV reports for its common capture properties, and first-frame
  facts (shape, type, min/mean/max). Plus build and platform lines.
- Filing: label `camera-support` (created in the repository once, before
  release), title `Camera report: <device name> (<USB ID>)`, filed with the
  existing Issues-only `FEEDBACK_TOKEN` through the same GitHub code as
  `feedback.py` (shared, not copied).
- One issue per device type across all PCs: before filing, list the
  repository's `camera-support` issues (open and closed, skipping pull
  requests) and file nothing if one already belongs to the same key (exact
  title or a hidden marker in the body); within one run a key is handled at
  most once.
- A background worker (like `feedback.FeedbackWorker`) does the network
  work; the window polls it. Offline or GitHub errors are logged; the camera
  keeps working; nothing is stored, and the report is simply tried again the
  next time that device is switched on in a later run (my call).
- What the person sees: the camera info gets a status line ("Reporting this
  camera to GitHub…" / "Reported to GitHub as <link>" / "Already reported as
  <link>" / "Report not sent: <reason>"); the log records the same. No modal
  dialogs.
- No token (source checkouts): nothing is posted and no browser opens; the
  duplicate check still runs anonymously (the repository is public). If no
  issue exists, the full report is saved under `output/logs/` and a
  **Report this camera** button opens a prefilled new-issue page with the
  short facts (a URL cannot carry the full report).
- The report machinery is written once, generic over the report's key,
  title and body, so the later FLIR PR only adds the Spinnaker report.

### 6. Dependency and packaging

- `cv2_enumerate_cameras` added to `requirements.txt` with a marker that
  skips macOS (avoids `pyobjc`; no USB-Live2 there), and to the Windows
  build job's `pip_deps` and `setup_espi_live.ps1`. The frozen builds must
  include it.
- `--check` reports whether video-device enumeration is available.

### 7. Docs

- `docs/camera/` gains an XC-ST50 + USB-Live2 page: wiring (BNC → RCA
  composite), power (DC-700 13 V is within the camera's 10.5–15 V),
  field-integration switch, NTSC, the Hauppauge driver, "close WinTV /
  Hauppauge Capture first" (the device serves one program at a time), the
  interlace and aspect notes, and the ESPI caveats. `supported-models.md`,
  README and CHANGELOG updated.

## Left to the builder

- Module, class and field names; how `CameraInfo` carries the kind and
  device id; whether the capture identity is a `CameraIdentity` variant or a
  sibling class with the same methods the window uses.
- Exact wording of labels, tooltips, notes and status lines.
- How the report body is laid out and trimmed (limits: body under 65 536
  characters, prefilled URL ≤ 8 000).
- The grab timeout strategy (`read()` blocks for about one frame).

## Risks

- The prebuilt release's capture path will first run for real on the lab
  PC; the tests use a fake `VideoCapture` and enumerator.
- If the Hauppauge driver exposes the device without a USB ID through the
  enumerator, recognition falls back to the name.
- Hauppauge's driver may hold a PAL/NTSC setting from earlier use; a
  720 × 576 frame size is the tell.
