# AGENTS.md

Guidance for AI coding agents and developers working in this repository. See `README.md` for usage and the full layout.

## Overview

A live viewer for a FLIR Blackfly (BFS-U3-63S4M-C) Mono8 camera used for ESPI measurements, written in Python with PySpin/Spinnaker, OpenCV, NumPy and PySide6 (Qt), with a legacy Tk front end. It must stay cross-platform (Windows camera PC, macOS and Linux). The camera runtime is a Windows PC (Python 3.10 with PySpin, SDK 4.2); the macOS checkout is for editing and camera-free `--demo` runs.

## Layout

| Path | Contents | Edit? |
|------|----------|-------|
| `src/espi_live/` | The application package (see Architecture). `__main__.py` → `cli.py` is the entry point. | Yes: single source of truth |
| `tests/` | Camera-free `unittest` checks for processing and the engine (demo source) | Yes |
| `scripts/windows/`, `scripts/bash/` | Camera-PC setup and launch helpers; `scripts/camera_live_control.py` is a standalone camera-control helper | Yes; keep their paths current when files move |
| `experiments/` | Tentative work and older scripts, including `espi_live_bfs_u3_200s6m.py` (a 20 MP-camera variant of the viewer, not reconciled with the package) | Yes |
| `docs/camera/` | Curated PySpin, BFS-U3-63S4M-C and acquisition notes | Yes |
| `docs/notes/` | Development notes and reports | Yes |
| `third_party/` | FLIR Spinnaker 4.2 (camera-PC SDK, wheel, docs, Examples) and 4.4 installers; large files are in Git LFS | Never |
| `.github/workflows/` | `build-binaries.yml`: PyInstaller builds for Windows/Linux/macOS when a PR merges into main | Yes |
| `reference/saba-originals/` | Preserved earlier scripts | No |
| `research/` | Physics talks, notebooks, papers, earlier code (not source to build). Local only: git-ignored because it holds third-party and personal documents | No |
| `archive/` | Superseded app versions, old scripts, a scratch-project zip | No |
| `output/` | Run-time logs, frames, recordings (git-ignored) | Never commit |

`espi_live.py` used to exist as several diverging copies. They were merged into the package; the originals are kept in `archive/legacy/`.

## Commands

```sh
./run.sh --demo                                    # Qt UI, synthetic video, no camera or PySpin
./run.sh                                           # try a real camera; offers demo mode if none found
./run.sh --demo --ui tk                            # legacy Tk UI
PYTHONPATH=src python -m espi_live --check         # import everything (Qt, engine, PySpin if present), no window
PYTHONPATH=src python -m unittest discover -s tests   # camera-free tests
python -m py_compile src/espi_live/*.py            # required check after edits (no display or camera needed)
```

- `.venv` (Python 3.14) holds numpy, opencv-python, pillow and PySide6 from `requirements.txt`. PySpin is imported lazily by `camera.load_pyspin()`.
- `--ui auto` (default) uses Qt when PySide6 imports, otherwise Tk.
- Windows camera PC: `scripts/windows/SETUP AND START ESPI.bat` validates and installs dependencies (including PySide6), then launches the app with `PYTHONPATH=src`.
- Do not run camera-acquisition code without the user's direction; the camera may be absent or in use. `--demo` and `--check` never open a camera.
- Outputs go to `<project>/output/{logs,frames,recordings}/` (`paths.OUTPUT_DIR`, independent of the working directory; next to the executable in PyInstaller builds); demo files are prefixed `dummy_`.

## Architecture (`src/espi_live/`)

| Module | Role |
|--------|------|
| `paths.py` | `PROJECT_ROOT`/`OUTPUT_DIR` (frozen-build aware), `timestamp()`, `output_folder()` |
| `processing.py` | `ESPIProcessor` (difference, averaging), `normalize_min_max`, `apply_palette` (the one definition of display color), `palette_rgb_table` (same colors as a GPU table), `DemoSource`, `watermark_demo` |
| `camera.py` | PySpin node helpers (`set_enum`/`set_bool`/`set_float`/`try_setting`), `configure_camera`, `CameraSession` (open/grab/set_float/limits/read_values/close) |
| `recording.py` | `AVIRecorder` (own thread, constant-rate timeline), `recording_path`, `save_frame` (PNG + title/notes sidecar) |
| `logs.py` | `AppLog`: per-run log file and stdout/stderr mirroring (`MirrorStream`) into a queue the UI drains |
| `engine.py` | `Engine`: worker thread that owns the source, processing, recording and saving; `Settings`, `FramePacket` |
| `qt_app.py` | Default front end: `MainWindow` (dockable controls/log, status bar), `GLImageView` (GPU), `CpuImageView` (fallback) |
| `app.py` | Legacy Tk front end (`--ui tk`): single-threaded loop, uses the shared modules |
| `cli.py` | Argument parsing, front-end choice, `--check` |

**Engine threading.** The front end chooses the source on its own thread before `start()` (`open_camera()` or `use_demo()`, so it can ask about demo mode); after that, every PySpin call (grab, node writes, readback, close, live switch) happens on the engine's worker thread. A `CameraSession` must only ever be used by one thread at a time. The UI talks to the engine only through `update_settings(**fields)`, `request(action, **payload)` (`save`, `reference`, `reset_previous`, `reset_average`, `switch_live`), `latest_packet()`, `status()` and the `events` queue (`camera_limits`, `record_off`, `switched_live`, `switch_failed`, `fatal`, `stopped`). Never touch Qt or Tk objects from the worker.

**Frame flow.** Each worker iteration: grab (`CameraSession.grab(100)` or `DemoSource.next_frame` on a fixed schedule) → apply changed camera settings (rate-limited to 0.1 s; readback every 1 s) → Raw passes the frame through; Previous/Reference run `difference` → `average` → optional min/max scaling → (CLAHE path: gamma + CLAHE on the CPU) → demo watermark → publish an immutable `FramePacket` (newest wins; older ones are dropped) → feed the recorder → handle queued actions. A mode change resets averaging; `average` returns `None` until N frames accumulate.

**Display.** A `FramePacket` holds the Mono8 image plus the palette and gamma still to apply. `GLImageView` uploads it as an R8 texture (with mipmaps) and a fragment shader maps it through a 256-entry RGB table from `palette_rgb_table`; scaling, zoom and pan are GPU work. Saving and recording call `FramePacket.render()` → `apply_palette`, so files match the screen exactly (a test enforces this). `CpuImageView` is used when an OpenGL 3.3 core context is unavailable (for example some remote-desktop sessions).

**Demo mode.** `DemoSource` simulates exposure/gain via brightness; `watermark_demo` stamps "DUMMY MODE ACTIVE" into every published frame, so it appears on screen, in saved frames and in recordings, and the Qt UI also shows a red banner, title and status badge. Demo data must never be mistakable for camera data. The "Switch to live capture" retry must leave the synthetic stream running on failure, and on success reset measurement state and stop any demo recording.

## Rules

- Before changing PySpin, camera lifecycle, node-map or image-buffer code, read `docs/camera/README.md` and the files it points to: `spinnaker-pyspin.md` always, `blackfly-bfs-u3-63s4m-c.md` for model-specific settings, `acquisition-and-performance.md` for streaming, ROI, pixel format, frame rate, preview latency, recording or triggering.
- Verify a node is readable/writable before writing it and clamp numeric values to device limits (`set_enum`/`set_bool`/`set_float` wrapped by `try_setting` do this). Release every `GetNextImage` result in a `finally` (`CameraSession.grab` does), and `.copy()` `GetNDArray()` data that outlives the image. Preserve safe lifecycle handling on errors.
- Don't change default exposure (8000 µs), gain (0 dB), frame rate (30 fps) or recording behavior without calling out the impact. Don't assume the displayed rate equals the camera's acquisition rate.
- Keep processing, camera and recording code free of GUI imports so both front ends and the tests share it. Keep display color defined only in `processing.apply_palette`.
- Keep it cross-platform: no platform-only GUI SDKs; guard OS-specific calls (as `WindowsPreviewTimer` does). The GPU path needs only OpenGL 3.3 core through Qt; keep the CPU fallback working.
- Don't reintroduce OpenCV HighGUI windows: mixing them with another toolkit caused fatal Windows crashes. In the legacy Tk UI, keep painting inside each frame's deadline and don't add a second Tk event pass per frame (that capped a 30 fps demo at ~21 fps).
- Every function needs a docstring in the existing `Args: ... / Returns: ...` format.
- Put tentative work in `experiments/`; promote it into `src/espi_live/` only when it becomes the intended maintained implementation. Use descriptive, purpose-based filenames; date-based names belong in `archive/` or `reference/`.
- Store recordings under `output/recordings/` and still frames under `output/frames/`. Don't commit generated media.
- Keep Windows and Bash helpers in `scripts/` and update their paths whenever a referenced file moves (they resolve `third_party/` and `src/` relative to the repo root).
- Don't add a virtual-environment or dependency-management system (pyproject, lockfiles) unless asked.
- Keep `README.md` accurate when layout, entry points or setup change, and add a `CHANGELOG.md` entry for user-visible changes.
- After edits, run `python -m py_compile` on the changed files and the tests.

## Git

- `.gitignore` excludes `.venv/`, `__pycache__/`, `.DS_Store`, `.omc/`, `output/`, `build/`, `dist/`, `research/` (never push it), Office lock files, and large binaries (`*.avi`, `*.zip`, `*.whl`, `*.dmg`, `*.tar.gz`, `*.pptx`). Don't force-add them. The exception is `third_party/`: its SDK zips, wheels, dmg and tarballs are tracked through Git LFS (`.gitattributes`); keep that LFS tracking when adding SDK files, because GitHub rejects plain files over 100 MB.
- CI fetches only the LFS files each platform needs (`git lfs pull --include`) and caches them, since LFS bandwidth is metered. Keep that behavior when editing `build-binaries.yml`.
- Don't commit unless asked.
