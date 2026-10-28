# BFS-U3-63S4M-C camera context

## Identity and core capabilities

- Model: Blackfly S BFS-U3-63S4M-C. `M` denotes the monochrome model; it is a
  USB3 Vision camera, not a GigE camera.
- Sensor: Sony IMX178 CMOS, 1/1.8-inch, 2.4 µm pixels.
- Full resolution: 3072 x 2048 (6.3 MP). Lens mount: C-mount.
- Readout: rolling shutter with global reset. Do not describe it as a global
  shutter camera or assume all rows expose simultaneously.
- Full-frame rated maximum is 59 FPS for Mono8. The documented minimum is 1 FPS.
- Supported monochrome pixel formats: `Mono8`, `Mono10Packed`, `Mono10p`,
  `Mono12Packed`, `Mono12p`, and `Mono16`. Pixel-format symbol spelling should
  be checked on the connected camera before use.
- ADC modes are 10-bit and 14-bit. The full sensor precision is not retained by
  `Mono8`; use a higher-bit-depth format for quantitative work where the rest
  of the processing/storage pipeline supports it.
- Documented ranges: exposure 8 µs to 30 s; gain 0.0 to 47.9943 dB. Always read
  `GetMin()`/`GetMax()` and clamp at runtime instead of hard-coding these values.
- Supported acquisition modes: Continuous, SingleFrame, MultiFrame. ROI and
  pixel binning are supported.

## Relevant GenICam features

Use the connected camera node map as the source of truth and guard every node
with `PySpin.IsReadable` / `PySpin.IsWritable`.

- `PixelFormat`, `Width`, `Height`, `OffsetX`, `OffsetY`: image format / ROI.
  Stop acquisition before changing them unless the device documentation says the
  node is writable while streaming. Set offsets to valid values after dimension
  changes; increments and maximums are camera-provided.
- `ExposureAuto`, `ExposureTime`, `GainAuto`, `Gain`: exposure and gain. Set an
  auto feature to `Off` before taking manual control, and report the change to
  the user because it materially changes acquisition behavior.
- `AcquisitionFrameRateEnable`, `AcquisitionFrameRate`,
  `AcquisitionResultingFrameRate`: manual requested rate and observed rate.
  Exposure longer than a frame period can prevent the resulting rate from
  reaching the requested rate.
- `TriggerSelector`, `TriggerMode`, `TriggerSource`, `TriggerActivation`,
  `TriggerDelay`, `TriggerSoftware`: trigger configuration. Set `TriggerMode`
  Off first, configure source/activation, then set it On; restore Off during
  cleanup when the program owns trigger setup.
- `UserSetSelector`, `UserSetLoad`, `UserSetSave`, `UserSetDefault`: persistent
  user sets. Saving or making a user set the boot default changes camera state
  beyond the current process, so do that only with explicit user direction.

## Sources

- Teledyne model performance specification: <https://softwareservices.flir.com/BFS-U3-63S4/latest/Model/spec.html>
- Teledyne product/resources page: <https://www.teledynevisionsolutions.com/en-gb/products/blackfly-s-usb3/?model=BFS-U3-63S4M-C&segment=iis&vertical=machine+vision>
- Model user-set reference: <https://softwareservices.flir.com/BFS-U3-63S4/latest/Model/public/UserSetControl.html>

