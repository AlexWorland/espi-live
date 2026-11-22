# ESPI noise compensation and USB-Live2 capture: research notes

Web research done on 2026-11-21, plus a read of `src/espi_live/` at `main`
commit `22c2f26`. **No camera was run.** The sources were search summaries and
abstracts, not full papers, so method rankings are the authors' own claims.
Sources are listed at the end of each part.

---

## Part 1: Compensating for sensor noise in ESPI

Every camera, digital or analog, adds noise. In ESPI the camera's own noise is
usually the smaller problem: the dominant "noise" is the speckle itself, which
is random and multiplies the fringe pattern. The literature handles camera
noise and speckle noise at different places in the pipeline.

### 1. Fixed-pattern noise (hot pixels, pixel-to-pixel offsets)

- **Standard fix:** dark-frame subtraction and flat-field division
  (corrected = (image − dark) / flat, dark subtracted first). Dark frames must
  match exposure and gain, and should themselves be averages of several frames.
- **In ESPI it mostly cancels.** A fixed offset is identical in the live frame
  and the reference, so `cv2.absdiff(live, ref)` = `|live − ref|` removes it.
  Dark frames add nothing to Reference or Previous views.
- **Flat-field (gain) variation** scales differences slightly and does not
  cancel, but illumination non-uniformity is a far larger effect. The app's
  **Even brightness** option (`even_out_brightness`) targets that.

### 2. Temporal sensor noise (shot, read, quantisation; analog capture noise)

- **Standard fix:** average N frames. The signal stays and random noise falls
  as 1/√N. Averaging cannot remove fixed-pattern noise.
- **Where the average goes matters in ESPI:**
  - **Before the subtraction:** the noise itself shrinks by √N, and the
    difference of two cleaner frames has a lower rectified floor. In the app:
    **Average camera frames first** (`average_frames_first`,
    `ESPIProcessor.average_frames`) and, for the baseline, **Reference frames**
    (`ref_frames`, `ReferenceCapture`). Without a multi-frame reference, the
    reference's own noise is frozen into every difference.
  - **After the subtraction:** averaging `|noise₁ − noise₂|`, which is never
    negative, lowers flicker but not the grey floor it averages to. This is
    the app's default **Average** of differences.
- A digital-holography study found averaging alone raises SNR, but subtraction
  and fringe stability were needed to reach the shot-noise limit.
- **Cost:** time resolution. N frames at 30 fps leaves 30/N independent images
  per second.
- The app also has an EMA **Noise reduction** on the camera feed
  (`processor.denoise`), which trades a motion trail for lower temporal noise.

### 3. Speckle noise (the main ESPI noise)

Fringes are speckle-correlation patterns, so each fringe is broken into
grains. This is where the ESPI literature spends its effort.

- **Classic subtraction ESPI chain:** high-pass filter the video, subtract the
  stored reference, then full-wave rectify or square before display.
  High-pass filtering removes slowly varying intensity and "significantly
  enhance[s] the fringe contrast". Self-interference of the object and
  reference beams leaves residual noise. The app's absdiff followed by fixed
  gain is the rectifying form. No source compared squaring with rectifying
  directly.
- **Speckle averaging (Montgomery and Bergquist):** average many
  interferograms whose speckle patterns are *decorrelated* for much higher
  fringe contrast. Camera frames of the same speckle do not qualify: the
  speckle has to change between frames (moved diffuser, changed illumination
  angle) while the deformation stays the same. This is an optics change.
- **Spatial filters on the fringe image:** median, Gaussian, bilateral
  (edge-preserving, tends to over-smooth), non-local means (Wielgus and
  Patorski), wavelet and wave-atom thresholding, oriented PDE filters. Xu et
  al. (2018) call coherence-enhancing diffusion and the windowed Fourier
  transform "possibly the most effective" at present. Few methods model the
  multiplicative nature of speckle. In the app: **Speckle filter**
  (`smooth_speckle`, Gaussian or Median; median capped at 5 px for speed) and
  the **outlier filter** (`reject_outliers`, the LabVIEW low-pass rule).
- **Phase-shifting ESPI with phase-map filtering (the main modern approach):**
  1. Step a reference piezo through known phase shifts.
  2. Compute the wrapped phase.
  3. Filter it:
     - **Sine/cosine filtering** (Aebischer and Waldner 1999): smooth sin φ
       and cos φ separately, then recombine. Simple, struggles with dense
       fringes. Fu et al. (2008) add fringe-orientation-adaptive windows.
     - **Windowed Fourier transform (Kemao):** threshold local frequency
       coefficients, keep phase. Preserves detail; sensitive to the
       threshold (small keeps noise, large removes signal). Fast and
       non-threshold variants exist.
  - The app already has a piezo driver (`daq.py`: "hold" mode for manual
    phase steps, "sine" mode). Phase calculation and phase filtering do not
    exist. Whether the lab has a calibrated phase-step mirror is unconfirmed.
- **Deep learning:** U-Net and residual CNNs (FPD-CNN, V-Net, SDCNN) denoise
  fringe patterns well. Nearly all train on simulated speckle because
  labelled real ESPI data is scarce. A research project, not a quick feature.

### Suggested order for the app

1. **Use what exists, and measure:** Average camera frames first plus a
   multi-frame reference, compared with today's flow at the same gain. The
   textbook √N fix; no code needed.
2. **Better spatial speckle filtering** after the difference (sine/cosine-style
   or orientation-aware): single-frame, software only.
3. **Phase-shifting ESPI:** the largest gain the literature reports. Needs
   calibrated piezo phase steps and new processing: phase calculation, phase
   filtering, wrapped-phase display.
4. **Skip for now:** dark frames and flat fields (they cancel or barely matter
   in differences), and deep learning.

### Key points

- Subtraction cancels additive, fixed errors (offsets, dark current) for free.
  Multiplicative ones (gain, range expansion) pass straight through to the
  fixed gain.
- The same √N average gives a lower floor before `absdiff` than after it,
  because afterwards you average values that are never negative.
- More frames of the same speckle don't remove speckle. Only decorrelated
  speckle, phase-domain filtering or spatial filters do.

### Sources (Part 1)

- [Kulkarni & Rastogi, "Fringe denoising algorithms: A review" (EPFL)](https://infoscience.epfl.ch/entities/publication/011df971-be4e-4020-b6ab-508ca6a5b686)
- [Xu et al., variational decomposition filter vs WFT and coherence-enhancing diffusion (Applied Optics)](https://www.osapublishing.org/ao/upcoming_pdf.cfm?id=285919)
- [ESSPI fringe-denoising families (core.ac.uk)](https://core.ac.uk/download/288361354.pdf)
- [Bianco et al., speckle reduction in digital holography (Light: Sci. & Appl. 2018)](https://www.doi.org/10.1038/S41377-018-0050-9)
- [Electronic speckle pattern interferometry (Wikipedia)](https://en.wikipedia.org/wiki/Electronic_speckle_pattern_interferometry)
- [Contrast enhancement of ESPI vibration patterns by speckle averaging (SPIE)](https://nanolithography.spiedigitallibrary.org/conference-proceedings-of-spie/599/0000/Contrast-Enhancement-Of-ESPI-Vibration-Patterns-By-Speckle-Averaging-In/10.1117/12.952376.full)
- [Iowa State thesis on ESPI processing](https://dr.lib.iastate.edu/entities/publication/7910f599-180e-4eef-8bea-6b4f50946580/full)
- [Vibration observation techniques for DSPI (high-pass, rectification, squaring)](https://typeset.io/pdf/vibration-observation-techniques-for-digital-speckle-pattern-56te440i0i.pdf)
- [Time-average subtraction method in ESPI](https://www.bohrium.com/en/paper-details/time-average-subraction-method-in-electronic-speckle-pattern-interferometry/812446095337586693-587)
- [Pulsed image-addition ESPI contrast patent (US6362873)](https://patents.google.com/patent/US6362873)
- [Fu et al., fringe-contoured-window sine/cosine filter](https://journal.hep.com.cn/foe/EN/10.1007/s12200-008-0044-y)
- [Kemao & Seah, 2-D windowed Fourier frames (Optical Engineering 2005)](https://nanophotonics.spiedigitallibrary.org/journals/optical-engineering/volume-44/issue-07/075601/Two-dimensional-windowed-Fourier-frames-for-noise-reduction-in-fringe/10.1117/1.1948107.full)
- [Shulev & Gotchev, sliding-window transform for DSPI (SPIE 2014)](https://neurophotonics.spiedigitallibrary.org/conference-proceedings-of-spie/9019/90190N/A-sliding-window-transform-domain-technique-for-denoising-of-DSPI/10.1117/12.2048139.full)
- [FPD-CNN (arXiv 1901.00361)](https://arxiv.org/abs/1901.00361v3)
- [Reyes-Figueroa & Rivera, V-Net (arXiv 1906.06224)](https://arxiv.org/abs/1906.06224v1)
- [Crespo et al., CNN for ESPI visibility (2023)](https://www.researching.cn/ArticlePdf/m00116/2023/19/1/2023015.pdf)
- [SDCNN for shearography (Applied Sciences)](https://mdpi.com/2076-3417/14/10/4135/html)
- [Dark-frame subtraction (Wikipedia)](https://en.wikipedia.org/wiki/Dark-frame_subtraction)
- [Flat field and dark frame corrections (CFHT)](https://www.cfht.hawaii.edu/~baril/Pyxis/Help/flatdarkfield.html)
- [Denoising by averaging (MIT computational photography)](https://people.csail.mit.edu/fredo/comp-photo-book/10-multiple-exposure-imaging-01-denoising-by-averaging.html)
- [Emil Martinec, noise in digital sensors](https://photonstophotos.net/Emil%20Martinec/noise-p4.html)

### What was implemented (2026-11-21)

Items 1–3 above are now the default pipeline; item 4 stands.

- **Defaults (item 1):** every camera starts with Average camera frames
  first, Average N 4, a 16-frame automatic reference, and the speckle filter
  below (`engine.DEFAULT_*`). The LabVIEW flow is a toggle, off by default.
- **Oriented speckle filter (item 2):** sine/cosine filtering applies to a
  *phase* map, so for subtraction fringes (intensity, no phase) the
  orientation-guided family is the fit. The literature's best performers
  (coherence-enhancing diffusion, windowed Fourier transform) are iterative
  or transform-based and too slow for 30 fps at 6 MP in Python, so
  `processing.oriented_smooth` is a one-pass approximation: a light
  isotropic blur (σ = width/8), then a long Gaussian along the fringe
  (σ = width/2, four directions blended by the doubled-angle orientation of
  the structure tensor, measured on a 1/4-size copy), blended into a plain
  Gaussian where the coherence is low.
  - Measured on simulated |I₂ − I₁| speckle fringes (correlation with the
    true |sin(Δφ/2)| envelope, equal kernel width): Oriented ≥ Gaussian in
    23 of 24 cases (fringe spacings 12–80 px, two angles, widths 9–13); the
    gap is largest for dense fringes (20 px spacing, width 13: 0.844 against
    0.757) and smallest for broad ones (80 px: 0.904 against 0.900). A first
    design with a thin line kernel only lost to the Gaussian, because it
    averaged too few pixels.
  - Cost: ~45 ms per 6 MP frame (~7 ms at 720 × 480). A diagonal kernel
    through `cv2.filter2D` alone took ~60 ms; shifted adds take ~6 ms.
- **Phase-shifting ESPI (item 3):** `espi_live.phase`, the engine's Phase
  view and `DaqController.move_piezo`.
  - Algorithms checked on synthetic interferograms: both recover a
    deformation exactly at a 90° step; at a 75° step the 4-step result is
    off by 7.5° rms and Hariharan's 5-step by 1.0°, and the 5-step frames
    measure the step itself (75.0°), which Calibrate phase step uses. In
    dummy mode with the simulated DAQ, calibration measured 71.4° for a true
    72° and proposed 0.0630 V for a true 0.0625 V.
  - The display maps no change to mid-grey: mapping 0 to black made a still
    area flicker between black and white from rounding around 0.
  - Safety: the piezo is attached only after Apply has saved settings that
    assign it (the DAQ window's defaults name AO 1, so "assigned" alone
    would have started stepping on merely opening the window), and every
    level is checked against the piezo's safety limit before it is written.
  - Not verified on hardware: the piezo's settle time (the Settle control,
    default 2 frames, is a guess to check), its linearity over five steps
    (hysteresis can make the steps unequal; 5-step tolerates moderate
    error), and that the lab's piezo mirror exists and is wired.

### Does any of this help with the USB-Live2's own artifacts?

The stick decodes composite video into YUY2 (uncompressed, so no compression
artifacts). What its decoding adds, and what each step does about it:

| Live2 artifact | Kind | Helped by |
|----------------|------|-----------|
| ADC quantization and analog cable noise | Random per frame | Averaging frames first (√N), the averaged reference, Phase view's per-step averaging |
| Horizontal line jitter (sync timing) | Random per frame, sub-pixel | Averaging frames first; partly, the Oriented filter |
| Dot crawl / cross-colour from the comb filter reading fine speckle as colour | Alternates frame to frame | Averaging frames first (opposite-phase frames cancel); the Oriented filter |
| Chroma notch removing detail near 3.58 MHz (~3.8 px per cycle) | Fixed, removes signal | Nothing in software: that detail never reaches the PC. Its fixed part cancels in differences, but the lost speckle contrast stays lost |
| BGR round trip in OpenCV (±1 grey level) | Fixed per pixel value | The raw-YUY2 change on `feat/analog-raw-yuy2-capture` |
| Interlaced fields 1/60 s apart | Motion between fields | Single field mode; the Phase view cancels anything common to all five steps |

Phase-shifting also cancels every fixed additive pattern (each phase term is
a difference of frames) and its modulation-weighted filter lets dark or
weakly modulated speckles, where the stick's filtering removed contrast,
count least. The hardware lever (an NI frame grabber digitising the
monochrome video directly) is still the only fix for the lost detail.

---

## Part 2: How the difference step works (`cv2.absdiff`)

`ESPIProcessor.difference` (`processing.py`) computes `cv2.absdiff(frame,
ref)` in Reference view (the analog default) or against the frame
`frames_back` ago in Previous view. `raw_difference` does the same between
consecutive raw frames.

| live | reference | `cv2.absdiff` | NumPy `live - ref` (uint8) |
|------|-----------|---------------|----------------------------|
| 120 | 100 | 20 | 20 |
| 100 | 120 | 20 | 236 (wraps) |
| 0 | 255 | 255 | 1 |

- No wraparound; the true distance per pixel.
- Inputs must match in shape and dtype. Until a same-size reference exists the
  code returns zeros; in Previous view a size or type change clears history.
- Works on float32 (deep) frames, keeping fractions.
- The sign is lost, which costs nothing for ESPI: dark fringes are where the
  speckle re-correlates (path phase change 2πn), whatever the direction.
- **Rectified noise floor:** for a still scene `|noise₁ − noise₂| ≥ 0`, so the
  difference has a positive mean. On the analog camera this is a few grey
  levels, and the LabVIEW flow's ×42 fixed gain (`apply_gain`,
  `cv2.convertScaleAbs(alpha=42)`, saturating) maps 1 → 42, 3 → 126, ≥ 7 → 255.

**Relevance to raw YUY2 capture:** an offset common to both frames (black at
16 in video-range luma) cancels in the difference. A scale does not: if the
current BGR path expands 16–235 to 0–255, differences there are about
255/219 ≈ 1.16× larger than in raw Y. Raw Y would then come out about 14 %
dimmer after ×42 (roughly gain 49 to match), on top of losing the ±1 grey-level
conversion error.

---

## Part 3: The Hauppauge USB-Live2 capture stick

### Two hardware versions

| Model | USB ID | Bridge / decoder | Linux |
|-------|--------|------------------|-------|
| Old, 122000 | `2040:C200` | Conexant CX23102 | `cx231xx` (kernel card list, card 9) |
| New, 122020 (Rev E1) | `2040:C220` | Empia em2828X with embedded video decoder | 2026 linux-media patch series (merge status unknown) and a community GPL DKMS driver |

- The camera PC's stick reports `2040:C220` with driver `hcwembda_x64.inf`, so
  it is the new Empia model.
- `docs/camera/sony-xc-st50-usb-live2.md` says Linux uses `cx231xx` for this
  stick; that is true only of the old model.
- Patch series "em28xx: Add empia EM2828X chip" (Bradford Love, March 2026):
  the new revision "switches from cx231xx usb bridge to Empia em2828X bridge";
  the chip "contains an embedded video decoder"; composite, S-Video and analog
  audio inputs; decoder register writes per input. No merge or ack seen.
- Community driver `usblive2-e1-linux7`: DKMS build of em28xx, based on Linux
  6.16 plus Hauppauge mediatree E1 changes; verified on one Linux Mint machine
  (kernel 7.0) for composite and S-Video at NTSC 720 × 480 YUYV; raw VBI
  disabled to fix a bottom-scanline defect; "the stock driver may leave this
  E1 revision's video interface unbound"; GPL-2.0.

### Is there a Windows API other than OpenCV?

- **No public one.** Hauppauge's FAQ: "for industrial uses, there is the WinTV
  SDK which would allow you to write a custom Windows application" with
  USB-Live2 video. Its SDK text: "provided under NDA and can be requested at
  techsupport@hauppauge.com or sdk@hauppauge.co.uk". The described features
  are TV-tuner oriented (channels, OSD, recording). No raw-frame API is
  described.
- The stick exposes a standard Windows capture interface: OpenCV's DirectShow
  backend opens it, and Hauppauge's instructions show a DirectShow crossbar.
- **Realistic improvement:** read the raw YUY2 and take its Y plane, skipping
  OpenCV's YUV→BGR conversion and our BGR→gray (about ±1 grey level). Routes:
  - OpenCV with `CAP_PROP_CONVERT_RGB = 0` (forum reports conflict on whether
    DirectShow honours it; test against the pinned OpenCV version);
  - a Media Foundation source reader (`MFVideoFormat_YUY2`); Microsoft
    recommends Media Foundation over DirectShow for new code;
  - a DirectShow Sample Grabber, or an FFmpeg pipe.
- **Limit:** the stick's colour decoder has already filtered the picture,
  including near the 3.58 MHz colour subcarrier. Raw YUY2 removes only the
  extra conversion round trip.
- Work on this is in progress on branch `feat/analog-raw-yuy2-capture`.

### Could we write our own driver?

Technically yes; not worth it.

- **Kernel driver:** 64-bit Windows loads only signed kernel drivers
  (attestation signing for test use, WHQL/HLK for release). It would replace
  a working Hauppauge driver for no new capability.
- **User-mode driver:** WinUSB supports isochronous transfers from desktop
  apps since Windows 8.1 (bind with Zadig). You would still need chip and
  decoder initialisation (the em28xx patch is public documentation, but GPL),
  em28xx packet framing and field assembly, and recovery from dropped packets
  (isochronous transfers are not retried). One forum thread reports failing
  to read raw isochronous data from a different composite capture card this
  way; its outcome is unknown.
- **Gain:** access to individual fields and no "close WinTV first" conflict.
  Not a way around the on-chip colour decoder; whether the chip has a raw
  sample mode is unknown.
- **If image quality is the goal, change the hardware.** The known-good
  LabVIEW flow used an NI frame grabber that digitised the monochrome video
  directly.

### Sources (Part 3)

- [Hauppauge support: USB-Live2](https://hauppauge.com/pages/support/support_usblive2.html)
- [Hauppauge support index (SDK section, drivers)](https://hauppauge.com/pages/support/support.html)
- [LinuxTV: Hauppauge USB-Live-2](https://linuxtv.org/wiki/index.php/Hauppauge_USB-Live-2) (search result only; the page blocked automated fetching)
- [Linux cx231xx card list](https://docs.kernel.org/6.8/admin-guide/media/cx231xx-cardlist.html)
- [linux-media patch: em28xx: Add empia EM2828X chip](https://ratatoskr.run/linux-media/2026/03/8466352/t)
- [USB-Live2 Rev E1 community Linux driver](https://github.com/NursePandaz/usblive2-e1-linux7)
- [Fix USB-Live2 not detected in the Windows build (PR #84)](https://github.com/AlexWorland/espi-live/pull/84)
- [OpenCV forum: fetching RGBA with VideoCapture](https://forum.opencv.org/t/fetching-rgba-format-from-camera-with-videocapture/16896)
- [Microsoft: YUV video subtypes](https://learn.microsoft.com/en-ie/Windows/Win32/directshow/yuv-video-subtypes)
- [Microsoft: WinUSB isochronous transfers](https://learn.microsoft.com/en-us/windows-hardware/drivers/usbcon/getting-set-up-to-use-windows-devices-usb)
- [OSR forum: WinUSB can not read RAW data from a composite capture card](https://community.osr.com/t/winusb-can-not-read-raw-data-from-generic-usb-composite-cap-card-device/59418)
- [Microsoft: driver signing offerings](https://learn.microsoft.com/en-us/windows-hardware/drivers/dashboard/driver-signing-offerings)
