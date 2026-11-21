# On-phone difference view in ESPI Camera (design)

## Intent

ESPI Camera (`ios/ESPICamera/`) is a pass-through today: it captures the back
camera with every automatic adjustment locked and sends 8-bit luma frames over
USB to ESPI Live, which does all the ESPI processing. The goal is a
**standalone** mode: the iPhone alone, with no computer or cable, shows the
live speckle difference `|image(n) − image(n−1)|` as a moving image, so
fringes can be seen at the bench with just the phone.

It is deliberately thin. It is the desktop's *Raw difference*
(`processing.raw_difference`) and nothing else, computed on the **rawest
image data the phone can deliver**.

## Goals, non-goals, anti-goals

- Goal: a full-screen view of `|live − previous|` that updates continuously
  from consecutive camera images.
- Goal: use the rawest data the phone has, found by inspecting the camera's
  capabilities (pixel formats), never by device model. "Rawest" beats
  "smoothest": slow but raw is preferred to fast but processed. In
  particular, nothing blended across frames by the phone's temporal noise
  reduction is used while an unblended source exists.
- Goal: no colour work. Colour is discarded at the source: only the red
  photosites of the Bayer mosaic are used (the lab laser is a 633 nm HeNe),
  or only the Y plane on the luma fallback.
- Goal: the screen always says which pipeline is in use (for example
  "Bayer RAW stills · red photosites · 14-bit · 2016 × 1512 · 6.8 fps").
- Non-goals: gain, averaging, smoothing, stretch, thresholds, a captured
  reference frame, recording or saving, colour palettes, a laser-colour
  setting, Bayer *video* (ProRes RAW), any change to ESPI Live on the PC.
- Anti-goal: anything that changes the raw values before subtraction
  (demosaicing, tone curves, noise reduction we add).
- Anti-goal: a processed pipeline shown as if it were raw, or a silent
  fallback. The pipeline label is always visible.
- Anti-goal: breaking the existing USB streaming mode.

## Constraints

- iOS 17+, iPhone only, Swift 6 with complete concurrency checking, XcodeGen
  (`project.yml`). The `.xcodeproj` is generated and git-ignored.
- The test phone is an **iPhone 16 Pro Max**. The simulator has no camera.
- All camera work stays on `CameraStreamer`'s single serial queue (the reason
  it can be `@unchecked Sendable`). New capture code follows the same rule.
- Exposure, ISO, white balance (unity gains) and focus stay locked exactly as
  the streaming mode locks them today.

## What each phone offers (researched 2026-11-21)

| Pipeline | Where | Notes |
|---|---|---|
| Bayer video: `AVCaptureVideoDataOutput` delivering `kCVPixelFormatType_96VersatileBayerPacked12` (the ProRes RAW path) | iPhone 17 Pro / Pro Max, iOS 26+ | 12-bit packed mosaic, up to 60 fps, but reportedly still with white balance and **temporal noise reduction** applied. Not used (see Approaches). |
| Bayer RAW stills: `AVCapturePhotoOutput` with a Bayer `rawPhotoPixelFormatType` | the 16 Pro Max and most iPhones since the 7 | Unprocessed sensor mosaic (usually 14-bit, in 16-bit words), binned 12 MP. Not a video API: back-to-back captures reach roughly 5–10 fps with uneven spacing. |
| 10-bit luma video: `x420` (`420YpCbCr10BiPlanarFullRange`), plane 0 | most recent iPhones | Processed, with built-in noise reduction that cannot be turned off. |
| 8-bit luma video: `420f`, plane 0 | every iPhone | What the streaming mode uses today. |

Sources: Apple Developer Forums threads on ProRes RAW and
`96VersatileBayerPacked12` in `AVCaptureVideoDataOutput`, Apple's ProRes and
ProRAW support pages, and the `AVCapturePhotoOutput` reference.

## Approaches

1. **Rawest video stream, chosen automatically.** Bayer video, then 10-bit
   luma, then 8-bit. Smooth on every phone, but on the 16 Pro Max it is
   processed 10-bit luma.
2. **Approach 1 plus a manual "Bayer stills" mode.** Both kinds can be
   compared on the 16 Pro Max, at the cost of two capture paths and a mode
   choice.
3. **Always the rawest data, even if slow (chosen by the owner).** On the
   16 Pro Max this is a stream of RAW stills at a few frames per second.

The owner chose 3: rawness matters more than frame rate, because the phone's
video noise reduction blends neighbouring frames and softens the frame-to-frame
speckle change that ESPI measures. For the same reason the owner chose stills
**ahead of** Bayer video on phones that have both (17 Pros). Bayer video is
reportedly temporally noise-reduced too, so it is left out entirely, along
with its undocumented packed-12-bit unpacking, which could not be tested on
the test phone.

The owner also chose to use **only the red photosites**, fixed and with no
setting, because the lab laser is a 633 nm HeNe, and green and blue
photosites would contribute mostly noise.

## Design

### Pipeline selection

On entering the difference view, the app picks the first pipeline that the
camera supports, in this order:

1. **Bayer RAW stills.** `AVCapturePhotoOutput.availableRawPhotoPixelFormatTypes`
   contains a type for which `AVCapturePhotoOutput.isBayerRAWPixelFormat(_:)`
   is true. Apple ProRAW is excluded, because it is processed. To make RAW
   available, the device's active format may need to change (see Formats and
   settings).
2. **10-bit luma video** (`x420`) when the video data output offers it.
3. **8-bit luma video** (`420f`).

The selection logic is a pure function over a small "capabilities" value (the
offered RAW and video pixel formats and sizes), so it can be unit-tested
without a camera. The chosen pipeline, its bit depth and its size form the
label shown on screen.

### Removing colour

- **Bayer:** no demosaicing. The kernel reads only the red photosite of each
  2 × 2 cell, giving a red-only grey image at half the width and half the
  height (12 MP → about 2016 × 1512). Where the red site sits in the cell
  comes from the RAW pixel format's CFA pattern (`RGGB`, `GRBG`, `GBRG` or
  `BGGR`, encoded in the `OSType`, for example
  `kCVPixelFormatType_14Bayer_RGGB`). An unknown pattern is an error shown on
  screen, never a guess.
- **Luma fallback:** read plane 0 (Y) only. The CbCr plane is never read.
  Luma cannot be restricted to red without colour work, so the fallback label
  says "luma (all colours)".

### The difference

- For each pixel: `d = |cur − prev|`, computed in the source's own integer
  levels. `prev` is the immediately previous image from the same pipeline. The
  black level cancels in the subtraction.
- The first image after entering the view, or after any change of pipeline,
  size or camera setting, has no `prev` and shows black. The previous-image
  buffer is reset at that point.
- **Display scaling: the source's full scale maps to display white.** For an
  N-bit source the displayed grey is `d × 255 / (2^N − 1)`. For 8-bit luma
  this is exactly the desktop's Raw difference. For 10- and 14-bit it shows
  the same fraction of full scale that a Mono8 camera, such as the FLIR, would
  show for the same change. It is a fixed bit-depth conversion, not a gain or
  stretch. (A 1 level = 1 grey mapping was considered and rejected: 14-bit
  photon noise alone would saturate most of the frame to white.)
- The computation runs on the GPU in a Metal compute kernel, reading the camera
  `CVPixelBuffer`s through a `CVMetalTextureCache` (no CPU copies). The
  previous image is kept as a GPU texture of the native bit depth (`r16Uint` or
  `r8Unorm`).
- The result is drawn aspect-fit into an `MTKView` with linear filtering
  (display only), rotated to portrait the same way `CameraPreview` rotates
  today (90°).

### Bayer RAW stills cadence

- Captures run back to back with at most one in flight. The next one is
  requested when the previous delivers its photo. The settings ask for speed
  (`photoQualityPrioritization = .speed`) and the largest binned Bayer size
  available.
- Device-level custom exposure, ISO, locked white balance and locked focus
  apply to every still, so frames are comparable. If the session
  reconfiguration resets any of them, they are re-applied before the first
  capture.
- Shutter sound: use `isShutterSoundSuppressionEnabled` where
  `isShutterSoundSuppressionSupported` (iOS 18+). Where it is unsupported,
  the phone clicks on each capture. Do not use private workarounds.
- The label shows the achieved rate and the time between the two images being
  compared.

### Formats and settings in the difference view

- The pipeline chooses its own capture format: for stills, a device format
  that offers Bayer RAW at the largest binned size. The streaming **Size** and
  **Frame rate** settings do not apply, and those pickers are hidden while the
  view is on.
- **Exposure, ISO and focus** (lens position and "Autofocus once, then lock")
  stay adjustable from a compact control strip in the difference view. A change
  is applied without rebuilding the pipeline and resets `prev`.
- Commands from ESPI Live while the view is on: exposure, ISO and focus are
  applied live. Size, crop, frame rate and send-rate changes are stored and
  take effect when the view is left.
- STATUS reports the streaming settings, as it does now, plus the
  `interrupted` text below.

### Modes and the USB link

- A **Difference view** button on the phone's screen switches between the
  existing streaming screen and the full-screen difference view, with an
  overlay for the pipeline label, the rate and the thermal state, the control
  strip, and a button to leave.
- While the view is on, USB frame sending pauses, so only one pipeline runs at
  a time. STATUS messages continue with `interrupted` set to "difference view
  on the phone". ESPI Live (`src/espi_live/iphone.py`) already logs this and
  treats the missing frames as a timeout, not a lost link, so there is no
  protocol change and no PC change.
- Leaving the view restores the streaming configuration (format, outputs,
  pixel format, stored settings) and frame sending.
- The streaming mode itself is unchanged.

### Code layout (proposed)

- `Sources/RawPipeline.swift`: the pipeline enum, the capabilities value, the
  pure selection function, the CFA-pattern decoding and the labels.
- `Sources/DifferenceRenderer.swift` and `Sources/Difference.metal`: the
  texture cache, the compute kernels (red-site Bayer and luma) and the
  `MTKView` drawing.
- `Sources/DifferenceView.swift`: the SwiftUI screen, overlay and control
  strip.
- `CameraStreamer.swift` gains the mode switch and the RAW stills loop, all on
  its queue. It may be split if it grows too large.
- `Tests/`: the selection function, CFA decoding, a CPU reference of the
  difference and the display scaling, and the Metal kernels checked against
  that reference on synthetic `CVPixelBuffer`s for each CFA pattern and bit
  depth (Metal runs in the simulator).

### Documentation

`ios/README.md` (layout table), `docs/iphone/README.md` (a section on the
difference view, the pipeline order and the red-only choice) and a
`CHANGELOG.md` entry.

## Known risks

- **Frame rate on the 16 Pro Max is unknown until measured.** It is expected
  to be about 5–10 fps, unevenly spaced. Consecutive differences span
  different intervals, and this is shown, not hidden.
- **Heat.** Continuous RAW capture will warm the phone. The thermal state is
  shown, and iOS may throttle or interrupt the session.
- **Shutter clicks** on iOS 17 or wherever suppression is unsupported.
- **Red-only is fixed.** A green laser would need a code change. Accepted by
  the owner.

## Revision after PRs #82 and #83 (same day)

While this was being built, `main` gained a per-back-camera selection (#82)
and a streaming Bayer RAW mode (#83) with its own RAW format search,
Bayer-layout table, one-in-flight stills loop, and black/white levels from the
DNG metadata. With the owner's agreement the difference view drives that
RAW loop instead of a second one, uses the selected back camera, and takes
display white from the photo's white minus black level (falling back to
`2^14 − 1`). Everything else above stands.

## Left to the builder

- How the device format for RAW is chosen and restored, and whether exposure
  locks need re-applying after reconfiguration.
- The kernel structure (one kernel per source or a shared one).
- The overlay and control-strip styling.
- Whether `CameraStreamer` is split into smaller types.

## Calls made by the designer (for the owner to check)

- Display scaling maps the source's full scale to white (see The difference).
- The streaming Size and Frame-rate settings are hidden in the difference
  view; exposure, ISO and focus remain adjustable there.
- The PC's size and rate commands are deferred while the view is on.
- Display uses linear filtering when fitting to the screen.
