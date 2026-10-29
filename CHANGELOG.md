# Changelog

## 2026-10-29 — Releases are approved by publishing a draft

### Previous state

- Releases were started by hand with **Actions > Build release > Run workflow** and typed-in notes; each was tagged `build-N` after the workflow run number.

### Current state

- A new `release-drafter.yml` workflow keeps a draft release up to date after every merge into `main`: the next `vX.Y.Z` tag (minor for `enhancement`, major for `major`, otherwise patch) and notes made from the merged pull requests. Publishing the draft starts `build-binaries.yml`, which tests (100% coverage gate), builds and attaches the Windows, Linux and macOS archives. Until the archives are attached the release is held as a draft, so installed copies never see a release without `build-info.json`; if a test or build fails it stays a draft.
- Running **Build release** by hand still works for test builds of any branch and still creates `build-N` releases. Installed copies are unaffected: they order builds by the build number in `build-info.json`, not by tag.

## 2026-10-29 — Tk interface removed

### Previous state

- Besides the Qt interface, the app had a legacy Tk interface (`--ui tk`, also used automatically when PySide6 was missing). It was the only user of Pillow, which had 41 open security advisories at the pinned 12.3.0; the fix, Pillow 13, no longer supports the camera PC's Python 3.10.

### Current state

- The Qt interface is the only one. `--ui` is gone; without PySide6 the program exits with a message. Pillow is no longer a dependency of the app or the builds, which clears those advisories on every platform.

## 2026-10-29 — Public source, private SDK

### Previous state

- The repository was private, so a release could only be downloaded by people with a GitHub account and access to it, and every build carried a read-only token to fetch updates.
- The FLIR Spinnaker SDK (installers, PySpin wheels, docs, Examples) was committed under `third_party/` through Git LFS.

### Current state

- The source is public under the GPL-3.0 (`LICENSE`), so anyone can download a build from the Releases page without an account, and installed copies update without a token (`UPDATE_TOKEN` is optional).
- The SDK lives in the private `espi-live-sdk` repository. CI checks it out into `third_party/` with a read-only deploy key (`SDK_DEPLOY_KEY`), and lab members clone it there; `third_party/` is git-ignored. It, and the preserved `reference/` scripts, were removed from this repository's whole history.

## 2026-10-28 — Several cameras at once

### Previous state

- The app used the first FLIR camera it found; other connected cameras were ignored. There was one set of settings, one reference frame and one recording.

### Current state

- With more than one camera connected, a **Cameras** section in the Controls panel has a **Display** drop-down for the camera on screen and a **Process** checkbox per camera. Checked cameras keep acquiring, processing and recording in the background while another is displayed; each row shows its frame rate and `REC`.
- Each camera has its own settings (exposure, gain, frame rate, noise reduction, view mode, reference, averaging, fringe and display options, Record). Switching the display loads that camera's settings into the controls; unchecking Process stops the camera but keeps its settings.
- Recordings and saved frames include the camera serial number in their filenames (`espi_raw_<serial>_<time>.avi`, `frame_<serial>_<time>.png`), and the frame-notes file names the camera. Demo files use `DEMO1`, `DEMO2`, ...
- With several cameras, log lines start with the camera serial, and an error in one camera turns only that camera off instead of closing the app.
- `--demo-cameras N` simulates N cameras in demo mode. **Switch to live capture** now lists every connected camera afterwards.
- Defaults are unchanged: exposure 8000 µs, gain 0 dB, 30 fps; at startup only the first camera that opens is processed. The legacy Tk interface still uses one camera.

## 2026-10-28 — Noise reduction slider for the camera feed

### Previous state

- Noise filtering (Average N, Speckle filter) only applied to the Previous and Reference difference images; the Raw view and the frames used as references kept the camera's full sensor noise.

### Current state

- A **Noise reduction** slider in the Camera controls (Qt and Tk) blends each incoming camera frame with the previous ones, 0–95% (default 0 = off, so existing behavior is unchanged). It removes random sensor noise in every view and in captured references, recordings and saved frames, and keeps the speckle pattern ESPI needs. High values trail motion and weaken Previous-view fringes.

## 2026-10-28 — Automatic camera model detection

### Previous state

- The viewer assumed a BFS-U3-63S4M-C: the window and badge said only "ESPI live" / "Camera" (Tk: "Blackfly ESPI"), and nothing reported which camera was actually connected.

### Current state

- On connection the app reads the camera's vendor, model, serial number, firmware, sensor and full-sensor size and shows the model in the window title, the status-bar badge (hover for details) and the Camera controls; the Tk viewer title names it too. The log records the description.
- Other FLIR/Teledyne models are recognised from their part number. Colour cameras and models not yet tested with the viewer get a note in the log and controls; exposure, gain and frame rate still follow the limits the camera reports. Defaults (8000 µs, 0 dB, 30 fps) are unchanged.
- New notes on similar Blackfly S models and the part-number scheme: `docs/camera/supported-models.md`.

## 2026-10-28 — Feature-request button with optional GitHub Copilot enrichment

### Previous state

- There was no way to send a feature request from the app; requests had to go to the developers separately.

### Current state

- A **Request feature** button in the status bar (and **Help > Request a feature…**) files the request as a GitHub issue labelled `feature-request`, in the background, with the build, platform and video source attached. A dialog links to the new issue; a failed send can be edited and retried, or opened in a browser.
- **Enrich with GitHub Copilot** adds the `enrich` label, and a new workflow (`enrich-feature-request.yml`) has GitHub Copilot append a structured spec (summary, motivation, proposed behavior, acceptance criteria, open questions) to the issue.
- Builds embed an Issues-only token from the new `FEEDBACK_TOKEN` secret (overridable with `feedback-token.txt` or `ESPI_FEEDBACK_TOKEN`); copies without one open a prefilled GitHub page instead. Enrichment needs the `COPILOT_PAT` secret.

## 2026-10-28 — Full test coverage

### Previous state

- The tests covered processing, the demo engine and part of the updater (25% of lines); the Qt and Tk front ends, camera code, CLI and logging were untested.

### Current state

- The camera-free test suite covers every line of `src/espi_live/` (about 430 tests, ~12 s), using an in-memory PySpin, offscreen Qt and withdrawn Tk windows; `.coveragerc` fails the report below 100%.
- The **Build release** workflow runs this suite first (Linux, Python 3.12); a failing test or coverage below 100% stops the run before any binary is built or released.
- Fixed: zooming with the anchor at the image's top-left corner (point 0,0) zoomed about the window center instead.

## 2026-10-28 — Automatic updates and manual releases for the prebuilt binaries

### Previous state

- Every merge into `main` rebuilt the binaries and replaced a rolling "latest" pre-release. Prebuilt binaries had no build number and no way to update: getting a newer build meant downloading and unpacking the release by hand.

### Current state

- Builds are now released by hand (**Actions > Build release > Run workflow**, with optional notes) as versioned releases **Build N**; merging into `main` no longer builds. A **Pre-release** option (always on for branches other than `main`) publishes test builds that installed copies are not offered.
- Prebuilt copies check the newest full release at startup (in the background) and show an **Update available** button in the status bar. Clicking it downloads and verifies the new build while measurement continues, then offers to restart; restarting finishes any recording and releases the camera before the new build starts with the same options. Otherwise the update installs when the app closes. **Help > Check for updates…** and **Help > About** were added.
- New command-line options: `--update` (install the newest build without a window), `--version`, and `--no-update-check`.
- CI stamps each binary with the workflow run number and publishes `build-info.json` (build number, commit, SHA-256 of each archive) with the release. Because the repository is private, builds embed a read-only token from the `UPDATE_TOKEN` secret; `update-token.txt` beside the program or `ESPI_UPDATE_TOKEN` overrides it.
- Running from source is unchanged and never updates itself.

## 2026-10-28 — Zoom slider; Exposure dock removed

### Previous state

- Zoom was available only through the mouse wheel, double-click (toggle 4×) and the `0`/`1` keys, with no indication of the current zoom. A raw-exposure histogram dock sat beside the log.

### Current state

- The toolbar has a **Zoom** slider (drag, click or scroll; logarithmic, so each step is the same factor), a percentage readout (100% = whole image fits the window), and **Fit** / **1:1** buttons. Wheel zoom and drag-to-pan on the image still work and keep the slider in sync. Double-click zoom is removed.
- The Exposure dock and its statistics are removed. "Switch to live capture" moved from the toolbar to under the dummy-mode banner, which now also says that exposure, gain and FPS are simulated, so the toolbar fits a 1280 px window.

## 2026-10-28 — ESPI fringe tools: speckle filter, rolling average, reference refresh, exposure histogram

### Previous state

- Measurement images were the absolute difference, a block average (one image per N frames, so N = 10 showed 3 fps at 30 fps), and an exact min/max stretch that a few bright speckles could flatten. There was no spatial speckle filtering, the Reference stayed fixed until `F`, and nothing showed raw exposure or saturation.

### Current state

- **Speckle filter** (Gaussian or Median) with a width in pixels, applied after averaging. **Rolling average** keeps the display at the camera rate. **Ref. refresh** recaptures the Reference every N seconds to cancel slow drift. **Stretch clip** ignores a percentage of outlier pixels when stretching, and **Invert** swaps bright and dark fringes. Saved frames and recordings include all of these, as before.
- The shortcut legend moved to the image tooltip and the View menu, and the dummy-mode banner is one line, so the Controls panel still fits without scrolling (minimum window height about 750 px).
- Defaults leave the processing unchanged (filter 0, rolling off, refresh 0, clip 0, invert off). Recording with rolling averaging records at the source rate. New tests cover the stretch, rolling average, filters and the combined pipeline in the engine.

## 2026-10-28 — Qt layout: toolbar actions and an aligned controls panel

### Previous state

- In the Qt UI, Save frame, Set reference, Reset and Record sat at the bottom of a scrolling Controls dock, out of view on laptop-height windows. Each settings group had its own framed form, so sliders were narrow and value boxes didn't line up. The Log dock ran under the Controls dock and shortened it.

### Current state

- A fixed toolbar above the image holds the view mode (Raw/Previous/Reference), Record AVI, Save frame, Set reference, Reset last and Reset average, plus "Switch to live capture" in dummy mode. These are always visible.
- The Controls dock is one three-column grid (label, slider, value) with plain section headings instead of group frames. It keeps the full window height and no longer scrolls: every control is always visible, and the window can't be made shorter than the panel. The Log dock spans only the image. Frame title/notes are under "Saved frame details".
- The view hint and recording state moved to the status bar. The status bar also shows the camera frame rate (the camera's `AcquisitionResultingFrameRate`, read once per second; "(simulated)" in dummy mode) ahead of the processed and displayed rates, so any drop between camera, app and screen is visible. Window layouts saved by the previous version are ignored once, so the new arrangement appears.
- No change to defaults, processing, recording or shortcuts.

## 2026-10-28 — Qt interface with GPU display and a separate processing thread

### Previous state

- Acquisition, ESPI processing, drawing and UI events all shared one Tk loop on the main thread, so any UI stall (including dragging a window border on Windows) paused measurement and recording. Each frame was resized, colored and copied into Tk on the CPU (~14 ms per full-resolution frame on a fast Mac), and a 60 fps demo request reached only 40–45 fps.
- The UI was three separate Tk windows with a scrolling control panel.

### Current state

- New default front end (`--ui qt`, PySide6): one window with the image, dockable/floatable Controls and Log panels, a status bar with source badge, range, processing and display rates, and a pixel-value readout. Mouse-wheel zoom up to 64× with sharp pixels, drag to pan, `0` fit and `1` actual size. Single-key shortcuts are unchanged and are ignored while typing in a text field. Window layout is remembered.
- GPU display: frames are uploaded as a Mono8 texture and a shader applies gamma and the palette through a 256-entry table; scaling and zoom are GPU work. The table comes from the same function that colors saved frames and recordings, so files match the screen exactly. Without OpenGL 3.3 (for example some remote-desktop sessions) the app uses a CPU-drawn view.
- New `Engine` worker thread owns the camera (all PySpin calls after startup), processing, recording and saving; the UI only reads the newest frame and sends settings/commands. Code is split into `paths`, `processing`, `camera`, `recording`, `logs`, `engine`, `qt_app`, `cli`; the Tk front end (`--ui tk`, also used automatically when PySide6 is missing) now uses the same modules.
- Faster averaging (`cv2.accumulate` and an in-place divide) with bit-identical results; demo pacing uses a fixed schedule and now reaches the requested 30 or 60 fps.
- Removed from the Qt UI: the Display wait slider and Fast Raw preview (both existed to work around CPU drawing cost) and the Text size slider (Qt follows the system font size). They remain in the Tk UI.
- `requirements.txt`, the Windows setup script and the binary workflow now include PySide6. `python -m espi_live --check` loads every module without a window, and CI runs it on each binary. New camera-free tests in `tests/`.
- Defaults are unchanged (8000 µs, 0 dB, 30 fps, Raw view, same recording behavior). Verified on macOS (Apple M5 Max): tests, both front ends in demo mode, GPU and CPU views, and a locally built PyInstaller `.app`. Not verified: Windows and Linux (including their OpenGL drivers), the live-camera path through the new engine thread, and the CI workflow changes.

## 2026-10-28 — Binary builds on PR merge

### Previous state

- The viewer ran only from source (`python -m espi_live`). The Spinnaker SDK zip, wheels and installers were git-ignored and existed only on one disk.

### Current state

- `.github/workflows/build-binaries.yml` builds PyInstaller binaries for Windows x64 (PySpin 4.2, Python 3.10), Linux x64 (PySpin 4.4, Ubuntu 24.04) and macOS arm64 (demo only) when a PR merges into `main` or on manual dispatch. Each binary is smoke-tested with `--help`; results are uploaded as artifacts and to the rolling `latest` pre-release.
- SDK files under `third_party/` are now tracked in Git LFS (`.gitattributes`; `.gitignore` no longer excludes them). CI fetches and caches only the files each platform needs.
- Frozen builds write `output/` next to the executable (next to the `.app` on macOS) instead of inside PyInstaller's temporary extraction folder.
- Not verified: the Windows and Linux builds (only the macOS build was run locally), the bundled PySpin on either platform, and live-camera use from a frozen binary.

## 2026-10-28 — Project reorganization

### Previous state

- The viewer existed as several diverging copies (top-level, `FlirPythonScratch/active-project/`, and a stale duplicate tree under `ESPI/`). Neither maintained copy was a superset: the top-level one added frame title/notes and safer frame saving but had dropped the shortcut legend and pinned capture buttons.
- The scratch project, SDK installers, physics archive and app files were mixed at the top level.

### Current state

- One application at `src/espi_live/app.py`, run with `python -m espi_live` (`run.sh` and the Windows setup script set `PYTHONPATH=src`). It combines both copies: frame title/notes and error-handled frame saving, plus the shortcut legend and pinned capture/quit buttons. Run-time output now goes to `<project>/output/` regardless of the working directory.
- Layout split into `src/`, `scripts/`, `experiments/`, `docs/`, `third_party/`, `reference/`, `research/` and `archive/`. Installer and example launchers were updated for the new paths; `third_party/flir-spinnaker-4.2/` follows the layout the newer `install_*.bat` already expected.
- Byte-identical duplicates (the `ESPI/FlirPythonScratch/` mirror, `espi_live copy.py`, and two stray copies) were deleted after checksum verification. The older two app copies are kept in `archive/legacy/`.
- `experiments/espi_live_bfs_u3_200s6m.py` (newest file, Oct 17) is a copy adapted for the 20 MP BFS-U3-200S6M-C with a 15 fps default; it is not merged. Defaults for the BFS-U3-63S4M-C (8000 µs, 0 dB, 30 fps) are unchanged.
- Not verified on the camera PC: the Windows `.bat`/`.ps1` path changes and the pinned-button layout with the added frame-notes fields. Demo mode was started on macOS and ran without errors.

## 2026-10-10 — ESPI live viewer usability and performance

### Previous state

- The OpenCV control sliders had small fixed labels and used scaled numbers such as 300 for 30 fps and 100 for gamma 1.0. The controls could cover the image.
- The viewer only showed ESPI differences. Brightness scaling happened automatically and was difficult to verify.
- AVI used a fixed 30 fps header regardless of preview rate or averaging. Recording and image processing shared the UI loop, and the log window showed only selected application messages.

### Current state

- A separate, resizable control panel has adjustable text, grouped settings, direct-unit sliders with precise entry fields, named palettes, capture buttons, device-limit-aware ranges, and camera readbacks. A separate log window mirrors Python console output and saves a text log.
- Raw Mono8 positioning view is the starting mode. ESPI scaling can be switched on or off, and input/output brightness ranges appear in the panel. The resizable video window preserves image proportions.
- Preview work is reduced, controls are serviced once per preview cycle, and the panel reports displayed FPS and processing time. AVI writing runs on a worker using a steady timeline; mode or averaging changes start a new file. The recording log reports skipped preview frames and late encoder intervals.
- Frame saves no longer use an image from a previous view mode. Every function now has purpose, input, and output documentation.

The camera's exposure, gain, and requested frame-rate defaults remain 8000 µs, 0 dB, and 30 fps. Live hardware performance and window behavior still require checking on the camera PC.

## 2026-10-10 — Camera-free demo startup

### Previous state

- Startup stopped with an error when no camera was connected or PySpin was missing, leaving the viewer unavailable for interface testing.

### Current state

- Startup offers a dummy mode when camera detection fails; `--demo` starts it directly without accessing a camera. The normal controls, log, Raw/measurement views, saving, and recording work with a moving synthetic Mono8 pattern.
- The viewer and saved media carry a prominent **DUMMY MODE ACTIVE** banner, and demo exports use `dummy_` filenames. Exposure, gain, and FPS are clearly labeled as simulated.
- A live UI check found fatal Windows crashes when OpenCV's viewer event loop and Tk windows were resized. The viewer now uses the same Tk event loop as the controls and log, with Pillow for aspect-correct frame display. Demo startup also avoids importing PySpin, the initial control/log split gives the log more room on high-DPI screens, and capture/quit buttons remain anchored beneath the scrollable settings.
- The control panel now shows each keyboard shortcut beside its action and in a compact viewer-shortcut legend.
- Minimum window sizes keep the shortcut legend and essential actions visible. Interactive Windows border dragging can still pause preview until the drag ends; this is now documented as a measurement/recording caveat.
- A highlighted note above the settings explains that more controls are below and can be reached by scrolling or making the control window taller; the initial split gives the controls slightly more height.
- Startup now measures the controls' requested height and opens the panel tall enough to show all settings when the screen permits. On wide screens, the log opens in its own column beside the controls instead of occupying the space below them.
- Demo mode now offers **Switch to live capture**. A failed retry keeps the synthetic stream running; a successful connection clears demo labels and measurement baselines, returns to Raw view, and stops any demo recording before live frames arrive.
- Windows preview pacing now uses an absolute frame deadline and requests 1 ms timer resolution only while the app runs, then releases it at shutdown. This avoids the measured default 15–16 ms sleep granularity; the higher timer resolution may use more power.
- **Fast Raw preview** now limits only Tk display pixels to reduce paint time; saving and recording still use full-frame images. It can be turned off when a larger on-screen image matters more than refresh rate.
- A remaining ~21 fps ceiling at a 30 fps demo setting was traced to Tk painting after the frame deadline. The preview now services Tk inside the same frame budget and sleeps only for the remaining time, with no redundant event pass. On this PC, the camera-free demo measured roughly 28–31 fps at a 30 fps request. A 60 fps request measured roughly 40–45 fps because Tk image conversion and painting still take about 20–25 ms combined; live-camera throughput has not been tested without hardware.
