# Acquisition, triggering, and performance context

## Default choices for this project

- For the maintained ESPI application, keep the current Mono8 pipeline unless a
  task explicitly calls for higher bit depth. Any new 10/12/16-bit path needs an
  end-to-end review: NumPy dtype, difference/normalization arithmetic, display,
  file format, and bandwidth all change.
- Use `AcquisitionMode=Continuous` for interactive preview. When manually
  limiting rate, enable `AcquisitionFrameRateEnable` before setting
  `AcquisitionFrameRate`, then observe `AcquisitionResultingFrameRate` when
  available rather than assuming the requested rate is reached.
- Full resolution Mono8 is about 6.29 MB/frame (3072 x 2048). At 59 FPS that is
  roughly 371 MB/s of image payload before USB/protocol overhead. If latency or
  dropped frames matter, reduce ROI, reduce bit depth only if acceptable, or
  reduce the rate/exposure; profile before making defaults more aggressive.
- A USB3 camera can build latency if the display/processing loop cannot keep up.
  The stream node map offers `StreamBufferHandlingMode`; `NewestOnly` is usually
  the appropriate starting policy for a *live preview* where recency matters
  more than processing every frame. It intentionally drops queued frames. For
  recording or measurement, choose a policy based on data-loss requirements and
  explicitly surface/drop-count behavior.

## ROI and pixel-format changes

Stop acquisition before reconfiguring pixel format, binning, ROI dimensions, or
offsets. Read `GetMin()`, `GetMax()`, and `GetInc()` for every dimension and
offset; width/height changes affect permitted offsets. Apply a safe order such
as offsets to a valid baseline, then dimensions, then final offsets, with all
values aligned to their increments. Recreate downstream arrays/recorders when
frame shape, channel count, or dtype changes.

## Triggering

Trigger mode changes capture semantics: a trigger-enabled camera produces frames
only when its configured trigger occurs. To configure safely:

1. Set `TriggerMode` Off.
2. Select the desired `TriggerSelector` when exposed, then configure
   `TriggerSource`, `TriggerActivation`, `TriggerDelay`, and overlap as needed.
3. Set `TriggerMode` On.
4. For software triggering, execute the writable `TriggerSoftware` command once
   per desired frame after acquisition has begun.
5. On teardown, return TriggerMode to Off if this program changed it.

Do not turn triggering on in a preview application casually: `GetNextImage()`
will time out until actual triggers arrive. Hardware line voltage/pinout and
opto-isolation requirements are physical-camera documentation matters; verify
the model technical reference before wiring anything.

## Diagnostics before changing defaults

- Record camera model, serial, firmware, and `system.GetLibraryVersion()`.
- Inspect node availability and current values with the local `NodeMapInfo.py`
  examples or SpinView; do not assume a node exists because another Blackfly
  model supports it.
- Log incomplete image status and timeouts. Never treat an incomplete image as a
  valid measurement frame.
- Test changes with camera hardware only after the user authorizes acquisition.

## Sources

- Buffer behavior and stream settings: <https://softwareservices.flir.com/Spinnaker/4.4/examples/python/BufferHandling.html>
- Acquisition/trigger semantics: <https://softwareservices.flir.com/BFS-U3-63S4-BD2/latest/Model/public/AcquisitionControl.html>
- BFS-U3-63S4M performance table: <https://softwareservices.flir.com/BFS-U3-63S4/latest/Model/spec.html>

