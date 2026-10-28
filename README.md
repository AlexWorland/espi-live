# ESPI live viewer

A live viewer for a FLIR Blackfly (BFS-U3-63S4M-C) Mono8 camera used for ESPI (electronic speckle pattern interferometry) measurements. Python, PySpin/Spinnaker, OpenCV and Qt (PySide6), with GPU-accelerated display; runs on Windows, macOS and Linux.

## Quick start

```sh
# Any machine (no camera or PySpin needed): synthetic video
./run.sh --demo

# Camera PC (Windows, Python 3.10 with PySpin): double-click
scripts/windows/SETUP AND START ESPI.bat

# Legacy Tk interface, if needed
./run.sh --demo --ui tk
```

Manually: `PYTHONPATH=src python -m espi_live [--demo]` (on Windows, `set PYTHONPATH=src` first). Run it with an interpreter that has the packages in `requirements.txt` (OpenCV, NumPy, Pillow, PySide6); the camera runtime also needs PySpin, normally `C:\Python310\PySpin\python.exe`. Without PySide6 the app falls back to the Tk interface. `python -m espi_live --check` loads everything without opening a window; `python -m unittest discover -s tests` runs the camera-free tests (both with `PYTHONPATH=src`).

## Project layout

```
src/espi_live/        the application: engine (acquisition/processing thread), Qt UI, legacy Tk UI
tests/                camera-free unit tests
scripts/
  windows/            camera-PC setup and launch helpers (.bat, .ps1)
  bash/               Bash equivalents
  camera_live_control.py   standalone live camera-control helper
experiments/          candidate, test and older working scripts
docs/
  camera/             curated PySpin / BFS-U3-63S4M-C / acquisition research notes
  notes/              development notes and reports
third_party/          vendor material, never edited
  flir-spinnaker-4.2/   SDK installer zip, PySpin wheel, docs, official Examples (matches the camera PC)
  spinnaker-4.4/        4.4.0.246 installers for macOS and Linux
  (the SDK zip, wheels, dmg and tarballs are stored in Git LFS)
.github/workflows/    build-binaries.yml: PyInstaller binaries for Windows, Linux and macOS on PR merge
reference/            preserved earlier scripts (saba-originals/), read-only
research/             physics background: talks, notebooks, papers, earlier code (local only, git-ignored)
archive/              superseded app versions and old scripts, kept for history
output/               logs, frames and recordings written at run time (git-ignored)
CHANGELOG.md  AGENTS.md  CLAUDE.md  requirements.txt  run.sh
```

`.venv` (Python 3.14, the packages in `requirements.txt`) is only for camera-free work on a Mac. There is no macOS PySpin wheel, so the camera path runs only on the Windows camera PC (SDK 4.2).

## Prebuilt binaries

When a pull request merges into `main`, `.github/workflows/build-binaries.yml` builds self-contained PyInstaller binaries and publishes them to the rolling **Latest build** pre-release (and as run artifacts). Run it manually from the Actions tab to build any branch.

| File | Camera support |
|------|----------------|
| `espi-live-windows-x64.zip` | PySpin 4.2 bundled (Python 3.10 build) |
| `espi-live-linux-x64.tar.gz` | PySpin 4.4 bundled (Ubuntu 24.04 build) |
| `espi-live-macos-arm64.zip` | `--demo` only (no macOS PySpin wheel); unsigned, run `xattr -cr "ESPI Live.app"` after unzipping |

Output goes to `output/` next to the executable (next to the `.app` on macOS). The CI builds only check that each binary starts and loads its modules (`--help`, `--check`); the bundled camera path has not been exercised against a real camera. A fresh clone needs `git lfs install` to get the SDK files.

### Automatic updates

Prebuilt copies update themselves from the **Latest build** release. At startup the Qt interface checks GitHub in the background; when a newer build exists, a blue **Update available** button appears in the status bar (and in the log). Clicking it downloads the build while measurement continues, verifies its SHA-256, then offers to restart: the app finishes any recording, releases the camera, swaps in the new build and starts it with the same options. Choosing not to restart installs it when you close the app. **Help > Check for updates…** checks on demand; **Help > About** shows the build number.

- `espi-live --update` updates without opening a window (any interface, including `--ui tk`); `--version` prints the build; `--no-update-check` skips the startup check.
- The program's folder must be writable: the new build is unpacked beside it, and the old copy is kept as `*.old` until the next start.
- Builds are ordered by the workflow run number, so a copy never "updates" to an older build. Running from source never updates itself; use `git pull`.
- The repository is private, so CI bakes a read-only GitHub token into each binary from the `UPDATE_TOKEN` repository secret: a fine-grained personal access token limited to this repository with **Contents: read-only**. Anyone with a binary can extract that token and read the repository. When it expires, installed copies report "the update token was rejected"; create a new token, update the secret, and either install the next build by hand or put the new token in `update-token.txt` beside the program (or the `ESPI_UPDATE_TOKEN` environment variable), which overrides the baked-in one.

## Using the viewer

One window holds the image, a toolbar with the view mode, Record, Save frame, Set reference and the reset buttons, a **Controls** panel for camera, measurement and display settings, and a **Log** panel; both panels can be docked on any side, floated as separate windows, or hidden from the **View** menu, and the layout is remembered. The status bar shows the current view's hint, the source (Camera or DUMMY MODE), the recording state, the measurement range, processing and display rates, and the value of the pixel under the cursor.

Camera acquisition and ESPI processing run on their own thread, so moving, resizing or interacting with the window never pauses a measurement or a recording. The image is drawn by the GPU (OpenGL 3.3 through Qt): scaling, gamma and the color palette are applied in a shader. On machines without suitable OpenGL (for example some remote-desktop sessions) the app logs "Display: CPU fallback" and draws the same image on the CPU.

- **Image navigation**: the toolbar's **Zoom** slider (drag, click or scroll it) zooms from fit-to-window up to 64× and shows the zoom as a percentage, where 100% is the whole image fitting the window; **Fit** (`0`) and **1:1** (`1`) jump to those levels. The mouse wheel or trackpad on the image zooms around the cursor, and dragging pans.
- **View**: **Raw** shows direct Mono8 pixels for positioning (no ESPI processing, scaling, gamma or palette). **Previous** shows frame-to-frame differences; **Reference** shows differences against a frame captured with `F` or **Set reference**. **Average N** averages that many difference frames to reduce speckle noise: one image per N frames, or, with **Rolling average**, a running average that still updates every frame. **Ref. refresh** recaptures the Reference every N seconds so slow drift (air currents, heat) cancels; 0 keeps one reference until you press `F`.
- **Fringe enhancement**: **Speckle filter** (Gaussian, or edge-keeping Median capped at 5 px to keep the frame rate) with **Filter width** in pixels blurs away pixel-scale speckle; keep it well below the fringe spacing (0 = off). **Invert** swaps bright and dark fringes. **Stretch clip** lets that percentage of the darkest and brightest pixels saturate so a few stray speckles don't flatten the contrast (0 = exact min/max).
- **Stretch to 0–255** maps each measurement frame's darkest and brightest difference pixels (after the clip) to 0 and 255; the status bar shows the range before and after scaling (before gamma and palette). A constant difference image maps to all zeros when scaling is on.
- **Camera** sliders show direct values with exact-entry boxes and adapt to the camera's readable limits; the panel shows applied exposure/gain and requested/resulting FPS. Camera bandwidth and exposure determine the attainable frame rate.
- **Capture**: **Record AVI** writes the displayed image on a steady timeline (intermediate frames are skipped when acquisition outruns encoding; the last frame is repeated when it is slower); changing view mode or averaging starts a new file, and the log reports skipped frames and late encoder intervals. **Save frame** writes exactly what is on screen; an optional **Frame title** is added to the PNG's filename, and a title or notes also write a matching `.txt` file.
- **Shortcuts** (when not typing in a text field): `S` save, `F` set reference, `R` reset last, `A` reset average, `Q`/Esc quit (Esc in a text field just leaves the field).

Output goes to `output/logs/`, `output/frames/` and `output/recordings/`.

The legacy Tk interface (`--ui tk`) has the same controls in three separate windows, without zoom or the separate processing thread.

### Demo mode

If no FLIR camera is detected (or PySpin is unavailable), startup offers **dummy/demo mode**, and `--demo` starts it directly without touching a camera. It opens the same windows with a moving synthetic grayscale pattern; exposure, gain and FPS are simulated. A persistent **DUMMY MODE ACTIVE** banner appears in the viewer and in saved frames and recordings, which also get `dummy_` filename prefixes. **Switch to live capture** retries camera detection without restarting: if no camera is available, demo continues; if one connects, measurement baselines reset to Raw view and any demo recording stops so synthetic and live frames never share a file. Demo mode is for interface and processing checks, not camera-performance measurements.

### Windows notes

The app temporarily requests 1 ms timer resolution while running so demo pacing tracks the requested FPS; this can increase power use. In the legacy Tk interface only, dragging a native window border pauses the preview and measurement processing until the drag ends.

## Camera-PC setup

Double-click `scripts/windows/SETUP AND START ESPI.bat`. It checks for Python 3.10, OpenCV, NumPy, Pillow, PySide6, Tkinter, PySpin and the FLIR Spinnaker SDK, and asks before installing or launching anything. After it installs or repairs a dependency, close any installers and run the same file once more; a clean validation run launches the app. The individual helpers (`install_spinnaker_sdk.bat`, `install_pyspin.bat`, `run_pyspin_example.bat`) remain for troubleshooting. Point VS Code at the Python 3.10 interpreter that contains PySpin.

## Other variants

`experiments/espi_live_bfs_u3_200s6m.py` is a standalone full copy of the viewer adapted for the 20 MP BFS-U3-200S6M-C (15 fps default, 1000 ms image timeout). It predates the package layout and has not been reconciled with the package.

See [CHANGELOG.md](CHANGELOG.md) for history and [AGENTS.md](AGENTS.md) for contributor and agent guidance.
