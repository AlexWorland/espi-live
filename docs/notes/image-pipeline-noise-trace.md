# Image pipeline noise trace (BFS-U3-63S4M, Mono8)

Where noise can enter or be changed between the sensor and a pixel on screen or
in a saved file. Written from a read of `src/espi_live/` at `main` commit
`3a703ac`, plus camera-free numeric experiments that call the real
`espi_live.processing` and `espi_live.engine` functions. **No camera was run.**
Each claim is tagged:

- **[code]** read directly from this repository (file:line given).
- **[exp]** measured by feeding synthetic frames through the repository's code:
  a static scene at level 100 plus independent Gaussian temporal noise of a
  chosen σ, rounded to `uint8` like the camera's Mono8 output.
- **[vendor]** from Teledyne/FLIR documentation (links at the end).
- **[derived]** arithmetic from vendor numbers, with the assumptions stated.
- **[unknown]** not determinable without the camera; section 6 says how to find out.

This covers the FLIR Blackfly path. Other sources join the same pipeline at
stage 3, with different stages 0–2:

- The analog XC-ST50 (`capture.py`, see `docs/camera/sony-xc-st50-usb-live2.md`):
  `capture.to_mono8` (`capture.py:150-163`) shifts 16-bit frames to their top
  byte or converts BGR to luma.
- The iPhone in **Video** mode: 8-bit luma that the phone's own pipeline has
  already denoised, sharpened and tone-mapped.
- The iPhone in **Bayer RAW** mode: one colour plane, decoded into a *deep*
  frame (`iphone.py:283-340`). That is float32 on the 8-bit scale, with black
  at 0, white at 255 and fractions kept. Every stage below keeps float32 for a
  deep frame, and only `to_uint8` (`processing.py:76-85`), called from
  `present` and the final scaling, rounds it.

The Blackfly sends Mono8, so for it every stage below runs on `uint8`.

## 1. The pipeline at a glance

| # | Stage | Data (Blackfly) | Effect on noise |
|---|-------|-----------------|-----------------|
| 0 | Photons → electrons → 14-bit ADC (in the camera) | analog → 14-bit integer | Sets the physical floor: shot noise dominates (section 2) |
| 1 | Camera image pipeline → Mono8 (in the camera) | 8-bit integer | Quantization (0.29 DN8 rms). Gamma, black level, sharpening and binning are **not set by this app** [unknown] |
| 2 | USB3 Vision → `GetNextImage` → `GetNDArray().copy()` | `uint8`, shape `(H, W)` | None added: no host-side conversion exists [code] |
| 3 | Optional camera-feed noise reduction (EMA) | `float32` → rounded `uint8` | Lowers temporal noise and collapses Previous differences toward zero. Off by default |
| 4 | Raw view stops here. Raw difference: `absdiff` and stop. Previous / Reference: optional frame block-average, then `absdiff` | `uint8` | Rectifies noise (positive mean). Reference freezes the baseline's own noise |
| 5 | Threshold, even brightness, LabVIEW-order gain, outlier filter (all off by default) | `uint8` | See section 4 |
| 6 | Block or rolling average of differences | `float32` accumulate → `uint8` | Averages temporal noise, then truncates (block) or rounds (rolling) back to 8 bits |
| 7 | Speckle smoothing, noise-floor subtraction, peak hold (off by default) | `uint8` | Blur. Floor subtraction hides changes below the floor. Peak hold keeps the noise maximum |
| 8 | Local normalization, Auto stretch (FLIR default) or Fixed gain (analog default) | `uint8` | Auto stretch rescales whatever noise exists to the full range |
| 9 | Invert, display LUT, flip, demo watermark, display gamma | `uint8` | Tone only. Identity at the defaults |
| 10 | GPU/CPU display; save PNG / record AVI | R8 texture; PNG; MJPG | Fit-to-window averages pixels. PNG is lossless. **MJPG is lossy** |

The defaults matter. A FLIR camera with nothing saved starts from
`own_settings(info)` (`engine.py:162-177`): mode Raw, Auto stretch with 0 %
clip and no sensitivity floor, and every optional stage off. But **profiles
persist per camera** (`profiles.py`), so the camera PC runs whatever settings
were last left there. The run log attaches `settings_summary`
(`diagnostics.py:73-96`) to its diagnostic lines; check there what was actually
on.

## 2. Stage 0: what the sensor physically produces

The sensor is a Sony IMX178 with 2.4 µm pixels and a 14-bit ADC [vendor]. EMVA
1288 values for the mono model, measured in a 16-bit pixel format with the ISP
disabled [vendor]:

- read noise 2.45 e⁻ (10.58 DN on the report's 16-bit scale)
- saturation capacity 14 181 e⁻
- gain 0.23 e⁻/ADU
- dynamic range 73.63 dB (12.23 bits)

Converted to Mono8 counts (DN8) **[derived]**. Assumptions: gain 0 dB (the page
does not state the test gain), and Mono8 is the top 8 bits of the 16-bit scale,
so 1 DN8 ≈ 257 ADU ≈ 59 e⁻. Whether the camera rounds or truncates when it drops
bits is **[unknown]**.

| Noise source | e⁻ | DN8 (1σ) |
|--------------|----|----------|
| Read noise | 2.45 | **0.04** |
| 8-bit quantization (uniform, 1/√12) | ~17 | **0.29** |
| Shot noise at level 20 / 100 / 200 / 240 | √(59·L) | **0.58 / 1.30 / 1.84 / 2.0** |

Shot noise is √(L/59) DN8 at level L. For any reasonably lit pixel it is
**4 to 7 times quantization and about 30 times read noise**, so a temporal σ of
1 to 2 DN8 in a well-exposed Raw frame is what a perfect sensor gives at this
bit depth. Noise only counts as excess if it is clearly above √(L/59).

A lens-capped dark frame should be nearly flat: std ≲ 0.3 DN8 once the
black-level offset is accounted for. A noisier dark frame points at something
other than photons.

Quantization adds in quadrature, σ² = σ_analog² + 1/12, once the analog noise
is above about half a count [exp]:

| True σ (DN8) | Measured | Predicted √(σ² + 1/12) |
|--------------|----------|------------------------|
| 0.5 | 0.570 | 0.577 |
| 1.0 | 1.041 | 1.041 |
| 2.0 | 2.020 | 2.021 |

Below half a count it stops behaving like noise. A true σ of 0.2 measured 0.11:
most pixels round to the same level every frame, and the error becomes a fixed
staircase.

## 3. Stages 1–2: how frames are requested and what format they arrive in

Request path, in order [code]:

1. `CameraSession.open` (`camera.py:362-393`): `Init()`, `read_identity`,
   `configure_camera`, `BeginAcquisition()`.
2. `configure_camera` (`camera.py:311-331`) writes only:
   - `PixelFormat=Mono8`, `AcquisitionMode=Continuous`
   - `ExposureAuto=Off`, `GainAuto=Off`
   - `ExposureTime`, `Gain`, `AcquisitionFrameRateEnable=True`,
     `AcquisitionFrameRate` (from the profile; 8000 µs, 0 dB and 30 fps by
     default)
   - the stream's `StreamBufferHandlingMode=NewestOnly`

   A write to an unreadable or read-only node is skipped with a log line.
3. `grab` (`camera.py:395-412`): `GetNextImage(100 ms)`; incomplete frames are
   discarded; otherwise `image.GetNDArray().copy()`, then `Release()`.
4. `engine.py:1018-1020` only checks that the frame is 2-D and either `uint8`
   or float32.

What that means for noise:

- **No host-side conversion.** Nothing in `src/` calls `ImageProcessor` or
  `Convert` [code, grepped]. The array is the camera's Mono8 payload byte for
  byte, so everything that turns 14 bits into 8 happens inside the camera.
- **Dropped frames are visible.** `NewestOnly` drops frames when the host is
  slow, which loses data but adds no noise. `read_stream_counters`
  (`camera.py:464-483`) and `StreamCounterMonitor` log a warning when the lost
  or dropped counters rise.
- **Camera settings the app never writes keep whatever the camera holds**: its
  default user set, or whatever was last saved to it. The code never loads a
  user set.
  - Not touched anywhere in `src/` [code, grepped]: `AdcBitDepth`,
    `BlackLevel`, `GammaEnable`/`Gamma`, `SharpnessEnable`/`Sharpness`,
    `LUTEnable`, defect correction, the binning nodes.
  - Their defaults for this model are **[unknown]**. The Teledyne datasheet
    lists gamma, LUT and sharpness as on-camera processing without defaults.
- **Possible 2×2 binning in Mono8.**
  - The BFS-U3-63S4 release notes list known issue F-2702, "Mono8 uses 2x2
    binning by default", for firmware 1907.3.1.0 (notes dated 11/16/2022). They
    do not say it was fixed.
  - The size of grabbed frames is still never logged, only `SensorWidth` and
    `SensorHeight` (`camera.py:307-308`), so the logs cannot rule it out.
  - [unknown]: low prior, but cheap to check.

## 4. Stages 3–10: what each step does to noise

The order is set by the worker loop (`engine.py:1018-1150`),
`condition_difference` (`engine.py:273-292`), `finish_difference`
(`engine.py:334-387`) and `present` (`engine.py:254-270`).

**Raw view is bit-exact camera output at the defaults** [code]. The frame goes
`grab → present → FramePacket` (`engine.py:1093-1099, 1141-1150`). Every step is
the identity when noise reduction is 0, display gain 1, contrast 0, brightness
0 and gamma 1. Single field applies only to interlaced sources (`interlaced` is
False for FLIR, `camera.py:337`). Any noise in Raw view, apart from display
scaling, was already in the camera's frame.

**Raw difference** (`Settings.raw_difference`, the toolbar toggle,
`engine.py:1082-1092`) skips noise reduction and every later step. It shows
`|live − previous|` in camera counts, with no gain, stretch or tone. It is the
cleanest in-app view of frame-to-frame noise.

**Stage 3, noise reduction** (`processing.py:838-863`): an EMA in float32,
rounded back to `uint8`, applied before differencing. The slider maps through
`denoise_strength` (`processing.py:531-555`).

A Previous difference then shrinks by the history weight w, because
y_t − y_{t−1} = (1 − w)(x_t − y_{t−1}) [derived]. A real change shrinks in the
same proportion, so this suppresses changes along with noise. At σ = 1.3 [exp]:

| Slider | History weight | Previous diff mean | Zero pixels |
|--------|----------------|--------------------|-------------|
| 0 | 0.00 | 1.47 | 21 % |
| 10 | 0.67 | 0.40 | 61 % |
| 30 | 0.86 | 0.14 | 86 % |
| 95 | 0.95 | 0.03 | 97 % |

**Stage 4, difference** (`processing.py:930-961`): `cv2.absdiff` discards the
sign, so noise becomes half-normal with mean 2σ/√π ≈ 1.13σ [exp: 1.126 at
σ = 1].

- Every Previous frame appears in two differences, so adjacent differences are
  correlated (0.22 in the experiments).
- Reference mode subtracts one stored frame, so that frame's own noise is
  frozen into every difference.

**Averaging order: where the noise floor comes from** (`average`,
`processing.py:963-1004`; `average_frames`, `processing.py:801-836`;
`ReferenceCapture`, `processing.py:707-752`). Static scene, σ = 1.3, N = 64
[exp]. Std is of the difference before scaling. Correlation is between two
independent results: a static pattern correlates, fresh noise does not.

| Setup | Mean | Std | Zero pixels | Correlation |
|-------|------|-----|-------------|-------------|
| Previous, average the differences (default) | 1.00 | 0.07 | 0 % | 0.00 |
| Previous, average frames first | 0.00 | 0.07 | 100 % | shared block only |
| Reference, average the differences, 1-frame reference (default) | 1.09 | 0.65 | 13 % | 0.69 |
| Reference, average frames first, 1-frame reference | 1.02 | 0.87 | 30 % | **1.00** |
| Reference, average frames first, 64-frame reference | 0.00 | 0.07 | 100 % | — |

- **Averaging the differences (the default) can never go below a floor.** The
  rectified pedestal of about 1 count remains. In Reference mode it plateaus at
  std 0.65 as a frozen grain that looks like fixed-pattern noise.
- **Average camera frames first** removes the pedestal. In Reference mode it
  does so only if the reference is averaged too (`ref_frames` = N). With a
  one-frame reference it is *worse* than the default: the result is exactly the
  reference's noise, perfectly static (correlation 1.00).

**Re-rounding to 8 bits after averaging discards precision** [exp]. Shot noise
of about 1.3 counts acts as dither, so a float mean of 8-bit frames resolves
changes far smaller than one count. For `uint8` input, `average_frames`
(`processing.py:833`) and `ReferenceCapture` (`processing.py:752`) round their
means back to `uint8`.

The test: a patch raised by a known step. The block mean of N frames
(`average_frames`) and the N-frame reference (`ReferenceCapture`) come from the
repository's code. The measured change is the signed mean of block mean minus
reference over the patch, less the same over the rest of the frame:

| True step | N | Mono8 as today (`uint8`) | Mono8 cast to float32 (deep path) | 14-bit-like float32 |
|-----------|---|--------------------------|-----------------------------------|---------------------|
| 0.1 | 16 | 0.069 | 0.100 | 0.102 |
| 0.1 | 64 | **0.007** | 0.100 | 0.099 |
| 0.3 | 16 | 0.241 | 0.303 | 0.309 |
| 0.3 | 64 | **0.106** | 0.297 | 0.302 |

On today's `uint8` path, more averaging makes things *worse*. Once the block
mean's noise falls below the rounding step the dither is gone, and small
changes snap to the nearest level (a 0.6 step came out as 0.71 at N = 64).

The same Mono8 frames recover the change almost exactly once they go through
the existing deep path. The camera's 8-bit cut costs nothing here, because the
shot noise dithers it.

The difference average re-quantizes as well:

- Without a scale it truncates (`processing.py:1002`): mean 0.747 against 1.128
  expected at N = 4, σ = 1.
- With a scale (LabVIEW order), and in the rolling average, it rounds
  (`processing.py:995, 1029`): mean 1.131.

**Stage 5, conditioning** (`engine.py:273-292`, all off by default):

- The threshold zeroes differences below N counts.
- `even_out_brightness` (`processing.py:404-432`) multiplies by up to 6×
  toward the beam edge, which amplifies the noise there by the same factor.
- LabVIEW-order gain saturates before averaging.
- `reject_outliers` (`processing.py:471-498`) pulls isolated spikes to the
  local mean.

**Stage 7, floor subtraction and peak hold.**

- `NoiseFloorMap` (`processing.py:630-690`) measures each pixel's still-scene
  mean plus one sd, and `subtract_floor` (`processing.py:693-704`) removes it.
  At σ = 1.3 the floor averaged 2.57 counts, and a still scene went to mean
  0.16 with 11 % of pixels non-zero [exp]. The cost is that a real 3-count
  change (3.08 measured) dropped to about 1.0 count. Changes below the floor
  disappear.
- `PeakHold` (`processing.py:312-375`) keeps each pixel's maximum, so it shows
  the noise *peak*, which is several σ.

**Stage 8, scaling** (`engine.py:367-384`). The FLIR default is Auto stretch
(`stretch_levels` → `RangeTracker` → `apply_levels`,
`processing.py:163-258`) with no clip and no sensitivity floor. It maps the
observed range to 0–255, so **the noise floor always looks full-contrast
however quiet the system is**. Previous view, single frame [exp]:

| σ (DN8) | Range | Displayed mean / std |
|---------|-------|----------------------|
| 0.3 | 0–2 | 23 / 51 |
| 0.5 | 0–4 | 35 / 38 |
| 1.0 | 0–7 | 41 / 34 |
| 2.0 | 0–15 | 38 / 30 |

- The sensitivity floor (`min_range`) fixes this, and analog cameras get 30 by
  default (`engine.py:159`). At σ = 1, a floor of 10 lowered the displayed
  mean to 29 and a floor of 30 to 9 [exp].
- Fixed gain (the analog default, ×42) and local normalization
  (`processing.py:378-401`, floored at `min_range`/4) do not follow the
  frame's range.

Do not judge sensor noise from a stretched difference view with `min_range` 0.

**Stage 9, tone** (`display_lut`, `processing.py:570-596`; `apply_display`,
`processing.py:599-613`): lookup tables on `uint8`, the identity at default
settings. Palettes and CLAHE are gone; the display is always grayscale.

**Stage 10, display, saving and recording.**

- **Display.** The view uploads an `R8` texture with a `LINEAR_MIPMAP_LINEAR`
  minify filter (`qt_app.py:1354, 1389-1394`). Fit-to-window minifies a
  3072×2048 frame about 3×, which averages neighbouring pixels: 1.53 DN8 of
  noise became 0.55 after a 3× area resize [exp]. Zoomed in, or in a saved
  file, the same frame looks about 3× noisier.
- **Files hold the 8-bit display image, for deep sources too**, because
  `present` rounds (`engine.py:254-270`). Stills go to lossless PNG
  (`recording.py:172, 196-222`).
- **Recordings.** AVI is MJPG with no quality set (`recording.py:27-29`), with
  gray expanded to BGR (`recording.py:81-83`). On noise-like content the JPEG
  error is comparable to the noise itself: rms 1.37 DN at quality 95, 1.56 at
  90 and 1.72 at 75, on a frame with total std 1.73. [exp, using
  `cv2.imencode` as a proxy; the `VideoWriter` default quality was not
  tested.] Recordings are not a place to measure noise.

## 5. In-app tools for noise work

- **Raw view** with every optional stage off: bit-exact camera frames.
- **Raw difference** toggle: `|live − previous|` in camera counts, nothing else.
- **Pixels window** (`pixel_square.py`): an N×N square's values with min, mean
  and max, refreshed every 0.1 s. In Raw difference it is cut from the raw
  difference, in Raw from the level, and in Previous/Reference from the
  unconditioned difference.
- **Noise calibration** (`NoiseCalibration`, `processing.py:261-309`): the
  still-scene difference's 99.5th-percentile level and a suggested sensitivity
  floor.
- **Stream-counter warnings** in the log: lost, dropped or incomplete frames.

## 6. How to separate optical from digital sources

All of these use Raw view or Raw difference with noise reduction 0, and all
need camera time.

1. **Log what the camera is actually doing.** Read `PixelFormat`,
   `AdcBitDepth`, `Width`/`Height`, binning, `GammaEnable`/`Gamma`,
   `BlackLevel`, `SharpnessEnable`, `LUTEnable`, `DeviceFirmwareVersion` and
   `DeviceTemperature` (SpinView shows them). This turns most **[unknown]**
   items above into facts.
2. **Lens-capped dark stack** at the measurement exposure and gain: per-pixel
   temporal std should be ≲ 0.3 DN8. More than that is electronic or pipeline
   noise, not optics.
3. **Static scene, temporal variance against level** (a photon-transfer check):
   variance in DN8² should rise roughly as L/59. On that line the noise is
   photon-limited; above it there is excess.
4. **Temporal against spatial.** With a truly static scene, per-pixel temporal
   std is noise. Stable pixel-to-pixel variation is speckle (signal) or fixed
   pattern. ESPI speckle has exponential intensity statistics, so a grainy Raw
   frame is expected and is not noise by itself.
5. **Optical culprits show up only with light.** Switch the laser on and off at
   fixed exposure, then the vibration isolation, then the enclosure or airflow.
   Temporal noise that follows those while the dark stack stays flat is optical
   or mechanical.

## 7. What this means for higher precision on the Blackfly

The deep (float32) pipeline from the iPhone RAW mode already carries
sub-count precision from the frame to the final scaling. For the Blackfly there
are two ways to use it:

**A. Cast Mono8 frames to float32 at grab.**

- **Gain:** averaged measurements (frames first, an averaged reference, or
  averaged differences) recover changes far below one count, as well as
  14-bit-like input does (the table in section 4).
- **What it does not help:** single, unaveraged frames.
- **Changes:** no camera, bandwidth or protocol change.
- **Costs, untested:**
  - Memory: four times per frame. The Previous view's `frames_back` history of
    up to 30 full frames grows from about 190 MB to 750 MB.
  - CPU: float work at 6.3 MP and 30 fps. The RAW mode runs 12 MP at only 5 to
    10 fps.
- **Possible subset:** promote only the averaged means (`average_frames` and
  `ReferenceCapture`). `difference` returns zeros when the frame and the
  baseline differ in type (`processing.py:930-961`), so both sides must be
  promoted together.

**B. `Mono16` from the camera into the deep path.**

- **Gain:** removes the camera's 8-bit cut. It also helps single frames and dim
  scenes where shot noise is below about half a count.
- **Changes:**
  - `configure_camera` and the camera's own processing nodes (gamma,
    sharpness, black level), which the app does not set today.
  - A grab-side conversion to 8-bit-scale float32, as `iphone.decode_raw_frame`
    does.
  - `fake_pyspin.py`, which models Mono8 only.
- **Costs:**
  - USB load doubles: 3072 × 2048 × 2 bytes at 30 fps is about 377 MB/s, at the
    USB3 ceiling.
  - Everything in option A.

**Either way, files stay 8-bit.** Saved frames and recordings hold the 8-bit
display image, so nothing saves the deep data today.

## Sources

- BFS-U3-63S4 EMVA 1288 report: <https://softwareservices.flir.com/BFS-U3-63S4/latest/EMVA/EMVA.html>
- BFS-U3-63S4 release notes (F-2702): <https://softwareservices.flir.com/BFS-U3-63S4/latest/Model/release_notes.html>
- In this repository: `docs/camera/blackfly-bfs-u3-63s4m-c.md`, `docs/camera/acquisition-and-performance.md`, `docs/camera/sony-xc-st50-usb-live2.md`, `docs/iphone/README.md`.
