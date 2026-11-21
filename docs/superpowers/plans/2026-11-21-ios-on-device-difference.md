# On-phone difference view Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone full-screen `|image(n) − image(n−1)|` view in ESPI Camera, computed on the GPU from the rawest data the phone has (Bayer RAW stills, red photosites only; otherwise luma video).

**Architecture:** A pure `RawPipeline` model chooses the source from the camera's pixel formats and owns every per-source constant. A camera-free `DifferenceEngine` runs one Metal compute kernel (`raw_difference`) that subtracts consecutive images and writes an `r8Unorm` texture. An `MTKView` draws that texture. `CameraStreamer` gains a difference mode with a RAW-stills loop and a luma path, pausing USB frames while it is on.

**Tech Stack:** Swift 6 (complete concurrency), SwiftUI, AVFoundation, Metal/MetalKit, CoreVideo, XCTest, XcodeGen.

**Spec:** `docs/superpowers/specs/2026-11-21-ios-on-device-difference-design.md`

## Global Constraints

- iOS 17.0 deployment target, iPhone only, `SWIFT_VERSION 6.0`, `SWIFT_STRICT_CONCURRENCY complete`. No new package dependencies.
- All camera and AVFoundation work runs on `CameraStreamer`'s serial `queue`.
- Pipeline order: Bayer RAW stills (known CFA pattern, Apple ProRAW excluded) → 10-bit luma (`x420`) → 8-bit luma (`420f`). Bayer video is not used.
- Bayer: red photosite of each 2 × 2 cell only, no demosaic. Luma: plane 0 only.
- Difference in native levels; display grey = `(d * 255 + (maxLevel >> 1)) / maxLevel`, with `maxLevel = 2^bits − 1`. For 8-bit this is `d`.
- First image after entering, a reset, or a size change shows black.
- The pipeline label is always on screen, e.g. `Bayer RAW stills · red photosites · 14-bit · 2016 × 1512 · 6.8 fps`.
- STATUS `interrupted` = `difference view on the phone` while the view is on; no protocol change.
- Exposure, ISO and focus adjustable in the view (apply live, reset previous). Size, crop, fps and send-rate changes are stored and applied on leaving.
- Shutter suppression via `isShutterSoundSuppressionEnabled` only when supported (iOS 18+); no private workarounds.
- Repo rules: do not commit unless the owner asks (commit steps below are checkpoints only). Keep the Python suite untouched.

## Review Focus

1. **The previous pixel is brighter than the current one.** The result must be `prev − cur`, not an unsigned wrap to a huge value. Test: `testEngineAbsoluteWhenPreviousBrighter` (Task 2).
2. **The phone offers only a Bayer RAW type whose CFA pattern isn't known** (for example the versatile `'bp16'`). The app falls back to luma, and the label says why. Test: `testUnknownBayerFallsBackToLumaWithNote` (Task 1).
3. **Odd mosaic dimensions.** The output is `floor(w/2) × floor(h/2)`, with no read past the edge. Tests: `testOutputSizeOddMosaic` (Task 1) and `testEngineOddMosaic` (Task 2).
4. **The image size changes between two images**, for example on a pipeline switch. The app resets, the first image is black, and nothing crashes. Test: `testEngineSizeChangeResets` (Task 2).
5. **The view is left and re-entered.** `reset()` makes the first image black again rather than diffing against a stale image. Test: `testEngineResetShowsBlack` (Task 2).

Run every test with:

```sh
cd ios/ESPICamera && xcodegen generate && xcodebuild -project ESPICamera.xcodeproj \
  -scheme ESPICamera -destination 'platform=iOS Simulator,name=iPhone 17' test
```

Device-code compile check (no signing):

```sh
cd ios/ESPICamera && xcodebuild -project ESPICamera.xcodeproj -scheme ESPICamera \
  -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build
```

---

### Task 1: Pipeline model and selection

**Files:**
- Create: `ios/ESPICamera/Sources/RawPipeline.swift`
- Test: `ios/ESPICamera/Tests/RawPipelineTests.swift`

**Interfaces:**
- Produces:
  - `enum CFAPattern: String, Sendable { case rggb, grbg, gbrg, bggr }`, with `var redOffset: (x: Int, y: Int)` returning rggb (0,0), grbg (1,0), gbrg (0,1), bggr (1,1).
  - `enum RawFormats { static func bayer(_ type: OSType) -> (pattern: CFAPattern, bits: Int)?; static func fourCC(_ type: OSType) -> String }`. `bayer` knows `kCVPixelFormatType_14Bayer_RGGB/GRBG/GBRG/BGGR` (14 bits) and returns nil otherwise.
  - `enum RawPipeline: Sendable, Equatable { case bayerStills(format: OSType, pattern: CFAPattern, bits: Int); case luma(format: OSType, bits: Int) }`, with:
    - `bits: Int`, `maxLevel: UInt32`
    - `shift: UInt32`: 6 for `x420` (10 bits in the MSBs), 0 otherwise
    - `step: UInt32`: 2 for Bayer, 1 for luma
    - `offset: (x: UInt32, y: UInt32)`: the red offset for Bayer, (0,0) for luma
    - `sourcePixelFormat: MTLPixelFormat`: `.r16Uint` when the source word is 16-bit (Bayer, `x420`), `.r8Uint` for `420f`
    - `outputSize(width: Int, height: Int) -> (width: Int, height: Int)`: halved and floored for Bayer
    - `label(width: Int, height: Int, fps: Double) -> String`, where width and height are the output size
  - `struct PipelineCapabilities: Sendable, Equatable { var rawPhotoFormats: [OSType]; var videoFormats: [OSType] }`. `rawPhotoFormats` holds only types that `AVCapturePhotoOutput.isBayerRAWPixelFormat` accepts. `videoFormats` holds the device formats' `mediaSubType`s.
  - `struct PipelineChoice: Sendable, Equatable { var pipeline: RawPipeline?; var note: String? }`
  - `func choosePipeline(_ caps: PipelineCapabilities) -> PipelineChoice`

- [ ] **Step 1: Write the failing tests** in `RawPipelineTests` (`@testable import ESPICamera`):
  - `testRedOffsets`: the four offsets above.
  - `testBayerTypes`: `RawFormats.bayer(kCVPixelFormatType_14Bayer_GRBG)` returns `(.grbg, 14)`; `RawFormats.bayer(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)` returns nil; `fourCC(kCVPixelFormatType_14Bayer_RGGB) == "rgg4"`.
  - `testPrefersBayerStills`: caps `raw [rgg4]`, `video [420f, x420]` → `.bayerStills(format: rgg4, pattern: .rggb, bits: 14)`, note nil.
  - `testUnknownBayerFallsBackToLumaWithNote`: caps `raw ['bp16']`, `video [420f, x420]` → `.luma(format: x420, bits: 10)`, and the note contains `bp16`.
  - `testTenBitBeforeEightBit`: no raw, `video [420f, x420]` → `.luma(x420, 10)`; `video [420f]` → `.luma(420f, 8)`.
  - `testNothingUsable`: empty caps → `pipeline == nil` and a non-nil note.
  - `testDerivedConstants`: Bayer 14 gives `maxLevel 16383`, shift 0, step 2, `.r16Uint`. `x420` gives maxLevel 1023, shift 6, step 1, `.r16Uint`. `420f` gives maxLevel 255, shift 0, step 1, `.r8Uint`.
  - `testOutputSizeOddMosaic`: Bayer `outputSize(width: 5, height: 3) == (2, 1)`; luma `(5, 3) == (5, 3)`.
  - `testLabels`: Bayer `label(width: 2016, height: 1512, fps: 6.84) == "Bayer RAW stills · red photosites · 14-bit · 2016 × 1512 · 6.8 fps"`. Luma `x420` `label(width: 1920, height: 1080, fps: 30) == "Luma video (all colours) · 10-bit · 1920 × 1080 · 30.0 fps"`.
- [ ] **Step 2: Run the tests.** They fail with "cannot find 'RawPipeline' in scope".
- [ ] **Step 3: Implement `RawPipeline.swift`** with the interfaces above. The note for an unknown Bayer type is `"Bayer RAW format '<fourCC>' is not supported yet; showing luma"`. The no-format note is `"No usable camera format"`.
- [ ] **Step 4: Run the tests.** All `RawPipelineTests` and the existing `LinkProtocolTests` pass.
- [ ] **Step 5: Checkpoint.** Do not commit unless asked.

### Task 2: Metal difference engine

**Files:**
- Create: `ios/ESPICamera/Sources/Difference.metal` (`kernel raw_difference`)
- Create: `ios/ESPICamera/Sources/DifferenceEngine.swift`
- Test: `ios/ESPICamera/Tests/DifferenceReference.swift` (the CPU reference, test target only)
- Test: `ios/ESPICamera/Tests/DifferenceEngineTests.swift`

**Interfaces:**
- Consumes: `RawPipeline` (Task 1), including `step`, `offset`, `shift`, `maxLevel`, `sourcePixelFormat` and `outputSize`.
- Produces:
  - `struct DiffParams { var offsetX, offsetY, step, shift, maxLevel, hasPrevious: UInt32 }`. It is mirrored exactly as `struct DiffParams { uint offsetX, offsetY, step, shift, maxLevel, hasPrevious; }` in Metal.
  - `final class DifferenceEngine: @unchecked Sendable`:
    - `init?(device: MTLDevice? = MTLCreateSystemDefaultDevice())`
    - `let device: MTLDevice`
    - `func reset()`
    - `func process(_ buffer: CVPixelBuffer, pipeline: RawPipeline) throws -> MTLTexture`: synchronous (waits for completion). It returns the `r8Unorm` output, which is black when there is no previous image.
    - `var latest: MTLTexture? { get }`: lock-protected, read by the display on the main thread.
  - `enum DifferenceError: Error { case texture(String) }`
- Kernel: for each output `gid`:
  1. `src = gid * step + offset`
  2. `cur = source.read(src).r >> shift`
  3. `d = hasPrevious ? abs(int(cur) − int(previous.read(gid).r)) : 0`
  4. `previous.write(cur, gid)`
  5. `out.write((d * 255 + (maxLevel >> 1)) / maxLevel / 255.0, gid)`

  Use `texture2d<ushort, access::read>` for the source and `texture2d<ushort, access::read_write>` (`r16Uint`) for the previous image.
- Source texture: plane 0. Use `CVMetalTextureCacheCreateTextureFromImage` with `pipeline.sourcePixelFormat`. If that fails (for example with a buffer that is not IOSurface-backed), fall back to `makeTexture` plus `replace(region:)` from the locked base address of plane 0, honouring bytes-per-row.
- Reset rule: `hasPrevious` is 0 after `reset()`, and whenever the output size or the pipeline differs from the last call. The previous and output textures are then reallocated.

- [ ] **Step 1: Write `DifferenceReference.swift`** (test helpers, not product code):
  - `func makeBuffer16(_ values: [UInt16], width: Int, height: Int) -> CVPixelBuffer` (`kCVPixelFormatType_OneComponent16`, IOSurface-backed)
  - `func makeBuffer8(_ values: [UInt8], width: Int, height: Int) -> CVPixelBuffer` (`kCVPixelFormatType_OneComponent8`)
  - `func makeYUV(_ y: [UInt16] or [UInt8], width:, height:, format:)` for `420f`/`x420` with a filled CbCr plane
  - `func referenceDifference(cur: [UInt16], prev: [UInt16]?, width: Int, height: Int, pipeline: RawPipeline) -> [UInt8]`, applying the same sampling, shift and rounding formula as the kernel
  - `func readR8(_ texture: MTLTexture) -> [UInt8]`
- [ ] **Step 2: Write the failing tests** in `DifferenceEngineTests`. Each test does `try XCTSkipIf(DifferenceEngine() == nil)`, and the engine tests compare against `referenceDifference`.
  - `testReferenceScaling`, on the CPU reference alone: 8-bit `d` maps to `d`; 14-bit 16383 → 255, 8192 → 128, 32 → 0, 33 → 1; `x420` `(1023 << 6)` vs 0 → 255.
  - `testEngineFirstImageBlack`: the first `process` returns all zeros.
  - `testEngineMatchesReferenceForEachPattern`: for each `CFAPattern`, two 8 × 6 mosaics with distinct values at every site give a 4 × 3 output equal to the reference. This proves only red sites are read.
  - `testEngineAbsoluteWhenPreviousBrighter`: 8-bit, prev 200 and cur 50 give 150.
  - `testEngineLumaTenBitIgnoresChromaAndLowBits`: `x420` with garbage in the low 6 bits and in CbCr gives output from the top 10 bits of Y only.
  - `testEngineOddMosaic`: a 5 × 3 Bayer gives a 2 × 1 output and no crash.
  - `testEngineSizeChangeResets`: 8 × 6, then 10 × 6 gives black.
  - `testEngineResetShowsBlack`: process, process, `reset()`, process gives black.
  - `testLatestIsLastOutput`: after `process`, `latest` returns the same texture.
- [ ] **Step 3: Run the tests.** They fail with "cannot find 'DifferenceEngine' in scope".
- [ ] **Step 4: Implement `Difference.metal` and `DifferenceEngine.swift`.** Load the kernel from `device.makeDefaultLibrary()`. XcodeGen compiles `.metal` files under `Sources` automatically.
- [ ] **Step 5: Run the tests.** All pass in the simulator.
- [ ] **Step 6: Checkpoint.**

### Task 3: Difference screen

**Files:**
- Create: `ios/ESPICamera/Sources/DifferenceView.swift`: `DifferenceView`, `DifferenceMetalView: UIViewRepresentable` and `DifferenceDisplay: NSObject, MTKViewDelegate`
- Modify: `ios/ESPICamera/Sources/Difference.metal`: add `vertex display_vertex` and `fragment display_fragment`
- Modify: `ios/ESPICamera/Sources/AppModel.swift`, `ContentView.swift`
- Modify: `ios/ESPICamera/Sources/CameraStreamer.swift`: snapshot fields only
- Test: `ios/ESPICamera/Tests/RawPipelineTests.swift`: the display transform

**Interfaces:**
- Consumes: `DifferenceEngine.latest` and `.device` (Task 2), and `RawPipeline.label` (Task 1).
- Produces:
  - `StreamerSnapshot` gains `differenceOn: Bool = false`, `pipelineLabel: String? = nil`, `pipelineNote: String? = nil` and `differenceIntervalMs: Double? = nil`.
  - `CameraStreamer` exposes `let engine: DifferenceEngine?`, created in `init`.
  - `CameraStreamer.setDifferenceView(_ on: Bool)` is declared here as a stub that only records the flag and publishes. Task 4 fills it in.
  - `AppModel.setDifferenceView(_ on: Bool)` forwards to the streamer.
  - `func displayQuad(texture: (w: Int, h: Int), view: (w: Int, h: Int)) -> SIMD2<Float>`: the quad's half-extent in clip space for an aspect fit after a 90° rotation.
- Drawing: an `MTKView` at 30 fps redraws `engine.latest` on a full-screen quad. It is rotated 90° (portrait, as `CameraPreview` does), aspect-fit using `displayQuad`, sampled with a linear `MTLSamplerState`, and drawn black when `latest` is nil.
- The screen is a `.fullScreenCover` from `ContentView`, opened by a **Difference view** button above the preview. It contains:
  - top: the label, the note and the interval (`"Δt 142 ms"`), plus a thermal warning in the same wording as `StatusCard`;
  - bottom: an exposure slider (log, the same ranges as `ControlsForm`), an ISO slider, a lens slider, an **Autofocus once, then lock** button and a **Done** button;
  - each slider sending its `LinkCommand` on release, as `ControlsForm` does. Move the shared slider helpers (`logBinding`, `logRange`, `exposureRange`, `range`) to file-level functions so both screens use them.

- [ ] **Step 1: Write the failing test** `testDisplayQuadAspectFit`:
  - texture (2016, 1512) in view (1320, 2868): after rotation the image is 1512 wide × 2016 tall, so the result is `(1.0, 1320/1512 * 2016 / 2868)` within 1e-4;
  - texture (100, 100) in view (100, 200) gives `(1.0, 0.5)`.
- [ ] **Step 2: Run it.** It fails because `displayQuad` is undefined.
- [ ] **Step 3: Implement** `displayQuad`, the shaders, `DifferenceDisplay`, `DifferenceView`, the snapshot fields, the `setDifferenceView` stub and the wiring in `AppModel` and `ContentView`.
- [ ] **Step 4: Run the tests and the device-code compile check.** Both pass.
- [ ] **Step 5: Checkpoint.**

### Task 4: Camera difference mode

**Files:**
- Modify: `ios/ESPICamera/Sources/CameraStreamer.swift`, with an extension in a new `ios/ESPICamera/Sources/CameraStreamer+Difference.swift` that holds the stills loop and photo delegate
- Modify: `ios/ESPICamera/project.yml` only if needed (no change expected)

**Interfaces:**
- Consumes: `choosePipeline`, `PipelineCapabilities`, `RawPipeline`, `RawFormats` (Task 1); `DifferenceEngine.process` and `reset` (Task 2); the snapshot fields and `setDifferenceView` (Task 3).
- Produces: a working `setDifferenceView(_:)`.
- Split `applyCameraSettings()` into:
  - `applyStreamingFormat()`: the current format, frame-rate and fallback code;
  - `applyLocks()`: HDR, GDC, low-light, colour space, tone mapping, zoom, custom exposure/ISO with limits from the *current* `activeFormat`, unity white balance, focus lock, stabilization and iOS 27 noise reduction.

  `applyCameraSettings()` calls both. Streaming behaviour is unchanged.
- **Entering** (on `queue`):
  1. Set `differenceOn`, stop sending frames, `engine.reset()`.
  2. Build the capabilities. Read the video formats from `camera.formats` mediaSubTypes (`420f`/`x420`). For RAW, set `session.sessionPreset = .photo`, add `photoOutput: AVCapturePhotoOutput` once, and read `availableRawPhotoPixelFormatTypes` filtered by `isBayerRAWPixelFormat`.
  3. Run `choosePipeline`.
  4. For Bayer stills, set `photoOutput.maxPhotoDimensions` to the largest `activeFormat.supportedMaxPhotoDimensions` entry with `width × height ≤ 12_600_000`, call `applyLocks()`, then start the loop. For luma, restore `.inputPriority`, activate the device format with the chosen subtype at `settings.width × settings.height` (otherwise the largest of that subtype ≤ 1920 × 1080), set `output.videoSettings` to that subtype, call `applyLocks()`, and feed `captureOutput` frames to the engine instead of USB.
  5. Publish the label and the note.
- **Stills loop:** at most one capture in flight.
  - Settings: `AVCapturePhotoSettings(rawPixelFormatType:)`, `photoQualityPrioritization = .speed`, `maxPhotoDimensions` as above, and `isShutterSoundSuppressionEnabled = true` when iOS 18+ and `photoOutput.isShutterSoundSuppressionSupported`.
  - In `didFinishProcessingPhoto`, hop to `queue` (wrap the `CVPixelBuffer` in an `@unchecked Sendable` box), call `engine.process(photo.pixelBuffer, pipeline:)`, update the rate and Δt, then request the next capture if the view is still on.
  - A capture error shows in `lastError` and retries after 0.5 s.
- **Commands while on:** exposure, ISO and lens call `applyLocks()` then `engine.reset()`; `refocus` runs as now and resets once the lock lands. Size, crop, fps and `maxSendFps` update `settings` only.
- **STATUS:** `interrupted` sends `interrupted ?? (differenceOn ? "difference view on the phone" : nil)`.
- **Leaving:** stop the loop (a capture finishing later is dropped), restore `.inputPriority` and `output.videoSettings` = `420f`, call `applyCameraSettings()` (which applies the stored settings), `engine.reset()`, then resume sending.
- **Thermal and interruptions:** as now. An interruption stops the stills loop, and `interruptionEnded` restarts it if the view is on.

- [ ] **Step 1: Make the `applyCameraSettings()` split.** Run the device-code compile check and the tests; streaming is unchanged.
- [ ] **Step 2: Implement entering, leaving and the luma path.** Run the compile check.
- [ ] **Step 3: Implement the stills loop and photo delegate.** Run the compile check and the full test suite; both pass.
- [ ] **Step 4: On-device check (the owner, iPhone 16 Pro Max):**
  - Enter the view. The label reads `Bayer RAW stills · red photosites · 14-bit · …`.
  - A still scene is near black, and touching the object shows bright speckle change.
  - Changing exposure shows one black frame, then the difference again.
  - ESPI Live shows "difference view on the phone" in its log.
  - Leaving restores USB frames.
  - Record the measured fps.
- [ ] **Step 5: Checkpoint.**

### Task 5: Documentation

**Files:**
- Modify: `ios/README.md` (the layout table: `RawPipeline.swift`, `DifferenceEngine.swift`, `Difference.metal`, `DifferenceView.swift`, `CameraStreamer+Difference.swift`, the new tests)
- Modify: `docs/iphone/README.md`: a "Difference view on the phone" section covering the pipeline order, red photosites only, the display scaling, USB pausing, shutter sound and expected fps
- Modify: `CHANGELOG.md`: one user-visible entry

- [ ] **Step 1: Write the docs.**
- [ ] **Step 2: Run the full test command and the device-code compile check one last time.** Both pass.
- [ ] **Step 3: Checkpoint.** Report to the owner, including that the on-device check in Task 4 Step 4 is theirs.
