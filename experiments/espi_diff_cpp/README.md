# espi_diff: a C++ subtraction-ESPI viewer

A from-scratch prototype that does one thing: open a camera, subtract each
frame from the previous one (or from a reference frame), and show the absolute
difference live. It shares no code with the Python application and uses no
Python, NumPy or OpenCV. The image processing is a few dozen lines of plain
C++ written here.

| Needs | For |
|-------|-----|
| A C++17 compiler and CMake 3.21+ | everything |
| [SDL3](https://www.libsdl.org/) 3.2+ | the window, and video capture devices |
| FLIR Spinnaker SDK (C++ development files) | FLIR Blackfly cameras; optional |

## What it does

```
camera thread                                   main (UI) thread
-------------                                   ----------------
grab frame (Spinnaker / SDL camera / synthetic)
copy it out of the driver's buffer
|frame - base| per pixel   (base = previous frame or the reference)
publish the newest result  --------------------> map through the gain table to grey
                                                 draw into the window, letterboxed
```

- **Difference** (`src/difference.cpp`): `out = max(a, b) - min(a, b)` per
  byte, which never wraps and needs no 16-bit intermediate. The compiler turns
  it into SIMD byte instructions: 0.16 ms for a 3072 × 2048 frame on an Apple
  M5 Max, plus 1.6 ms for the display mapping. A 30 fps frame period is 33 ms.
- **Modes**: *previous* (each frame minus the one before it: shows what is
  moving) and *reference* (each frame minus a captured frame: shows the
  deformation since then). The difference runs on the camera thread for every
  frame in order, so previous mode always subtracts consecutive frames even if
  the display drops some.
- **Display** (`src/display.cpp`): a 256-entry table maps each difference to a
  grey level after multiplying by the gain (1 to 128, saturating at white).
  The window shows only the newest image, so a slow display never builds
  latency.
- **Capture devices** (`src/capture_source.cpp`, `src/luma.cpp`): SDL3's
  camera API (Media Foundation on Windows, AVFoundation on macOS, V4L2 on
  Linux). It picks a YUV format and copies out only the luma bytes, with no
  colour conversion. Values stay in video range (black ≈ 16, white ≈ 235),
  so differences are about 14 % smaller than after a range expansion. Raise
  the gain to compensate.
- **Synthetic source** (`src/synthetic_source.cpp`, `--demo`): a simulated
  speckle interferometer. Each pixel follows `I = I_o + I_r + 2√(I_o I_r)
  cos(ψ + δ)`, with random speckle intensity and phase in 2-pixel grains and
  a Gaussian dome that bulges to six fringes and back every 12 s. Demo frames
  carry a red "DEMO: SYNTHETIC DATA, NOT A CAMERA" banner.

## What it changes on a FLIR camera

On opening a Blackfly it prints each change:

- `AcquisitionMode` Continuous, `PixelFormat` Mono8.
- `ExposureAuto` and `GainAuto` **Off**, then `ExposureTime` 8000 µs and
  `Gain` 0 dB (the project defaults). With auto exposure or gain, the
  brightness would change between the two frames being subtracted and flood
  the difference.
- `AcquisitionFrameRateEnable` on, `AcquisitionFrameRate` 30 fps.
- Stream `StreamBufferHandlingMode` NewestOnly, so the view stays live
  instead of working through a queue.

Each node is checked for read/write access first and numeric values are
clamped to the camera's limits. A node the camera lacks is reported and left
alone. Incomplete images are skipped, and every buffer goes back to the SDK
in all cases. Nothing is saved to the camera's user sets.

## Use

```
espi_diff                  # first FLIR camera
espi_diff --serial 1234    # a particular FLIR camera
espi_diff --list           # FLIR cameras and capture devices, with indices
espi_diff --device 0       # capture device 0, e.g. the USB-Live2 stick
espi_diff --demo           # no hardware
```

Options: `--gain G` (initial gain), `--reference` (start in reference mode),
`--frames N` (quit after N displayed images) and `--screenshot FILE.bmp`
(with `--frames`: save the final view).

| Key | Action |
|-----|--------|
| R or Space | capture a reference (the next frame) and switch to reference mode |
| P | previous-frame mode |
| Up / Down (or + / −) | double / halve the gain |
| F | fullscreen |
| H | hide or show the key help |
| Esc or Q | quit |

The status line shows the mode, gain, the camera's delivered frame rate and
the display rate.

## Build

### Windows camera PC

1. Visual Studio 2022 (or Build Tools) with "Desktop development with C++",
   which includes CMake.
2. SDL3 through [vcpkg](https://vcpkg.io): `vcpkg install sdl3:x64-windows`.
   Or download `SDL3-devel-3.x.x-VC.zip` from the SDL releases page and pass
   `-DSDL3_DIR=<unzipped>/cmake` instead of the toolchain file.
3. The Spinnaker SDK installed with its development (C++) files. CMake looks
   in `C:\Program Files\Teledyne\Spinnaker` and
   `C:\Program Files\FLIR Systems\Spinnaker`; otherwise pass
   `-DSPINNAKER_ROOT=<install folder>`.

From a "Developer PowerShell for VS 2022", in the repository root:

```
cmake -S experiments/espi_diff_cpp -B build/espi_diff -DESPI_DIFF_SPINNAKER=ON ^
      -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
cmake --build build/espi_diff --config Release
ctest --test-dir build/espi_diff -C Release
build\espi_diff\Release\espi_diff.exe
```

`-DESPI_DIFF_SPINNAKER=ON` makes a missing SDK a build error instead of a
build without FLIR support. SDL3.dll is copied next to the executable, and the
Spinnaker DLLs are on `PATH` after the SDK installer runs.

### macOS and Linux

```
brew install sdl3 cmake          # or your distribution's SDL3 package
cmake -S experiments/espi_diff_cpp -B build/espi_diff
cmake --build build/espi_diff -j
ctest --test-dir build/espi_diff
build/espi_diff/espi_diff --demo
```

FLIR support is built automatically when Spinnaker is installed
(`/usr/local/include/spinnaker` on macOS, `/opt/spinnaker` on Linux).
`-DESPI_DIFF_BUILD_VIEWER=OFF` builds only the tests, which need neither SDL
nor Spinnaker.

## Tests

`tests/core_tests.cpp` covers the difference (all 65,536 byte pairs), both
modes, reference capture and replacement, size changes, the gain table and
padded texture rows, luma extraction from YUY2, UYVY and NV12 with row
padding, the synthetic interferometer (fringes where the dome moved, noise
only where it did not), and the processing thread (delivery, mode switch,
error reporting). It opens no camera and no window.

## Limits

- Mono8 only. The BFS-U3-63S4M-C can send Mono12/Mono16 for more precision,
  which would need 16-bit frames end to end.
- No saving, recording, averaging, filtering or contrast stretch: this is the
  minimal pipeline.
- Capture devices: SDL cannot choose a capture stick's input. If the USB-Live2
  shows black, select Composite once in WinTV, close WinTV, and start again.
  NTSC pixels are not square, so the picture looks about 12 % too wide.
- On macOS the first `--device` run asks for camera permission.
