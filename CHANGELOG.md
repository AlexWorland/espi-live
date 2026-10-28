# Changelog

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
