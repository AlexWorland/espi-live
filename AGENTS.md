# AGENTS.md

Guidance for AI coding agents and developers working in this repository. See `README.md` for usage and the full layout.

## Overview

A live viewer for a FLIR Blackfly (BFS-U3-63S4M-C) Mono8 camera used for ESPI measurements, written in Python with PySpin/Spinnaker, OpenCV, Pillow and Tk. There is no test suite or linter config. The camera runtime is a Windows PC (Python 3.10 with PySpin, SDK 4.2); the macOS checkout is for editing and camera-free `--demo` runs.

## Layout

| Path | Contents | Edit? |
|------|----------|-------|
| `src/espi_live/app.py` | The whole application. `__main__.py` is the `python -m espi_live` entry point. | Yes: this is the single source of truth |
| `scripts/windows/`, `scripts/bash/` | Camera-PC setup and launch helpers; `scripts/camera_live_control.py` is a standalone camera-control helper | Yes; keep their paths current when files move |
| `experiments/` | Tentative work and older scripts, including `espi_live_bfs_u3_200s6m.py` (a 20 MP-camera variant of the viewer, not reconciled with `app.py`) | Yes |
| `docs/camera/` | Curated PySpin, BFS-U3-63S4M-C and acquisition notes | Yes |
| `docs/notes/` | Development notes and reports | Yes |
| `third_party/` | FLIR Spinnaker 4.2 (camera-PC SDK, wheel, docs, Examples) and 4.4 installers | Never |
| `reference/saba-originals/` | Preserved earlier scripts | No |
| `research/` | Physics talks, notebooks, papers, earlier code (not source to build). Local only: git-ignored because it holds third-party and personal documents | No |
| `archive/` | Superseded app versions, old scripts, a scratch-project zip | No |
| `output/` | Run-time logs, frames, recordings (git-ignored) | Never commit |

`espi_live.py` used to exist as several diverging copies. They were merged into `src/espi_live/app.py`; the originals are in `archive/legacy/` and in git history (see the commit "Initial import…").

## Commands

```sh
./run.sh --demo                                    # synthetic video in .venv, no camera or PySpin
./run.sh                                           # try a real camera; offers demo mode if none found
PYTHONPATH=src python -m espi_live --demo          # same without run.sh
python -m py_compile src/espi_live/app.py          # required check after edits (no display or camera needed)
```

- `.venv` (Python 3.14) holds only numpy, opencv-python and pillow from `requirements.txt`. PySpin is imported lazily in `CameraSession.open()`.
- Windows camera PC: `scripts/windows/SETUP AND START ESPI.bat` validates and installs dependencies, then launches the app with `PYTHONPATH=src`.
- Do not run camera-acquisition code without the user's direction; the camera may be absent or in use. `--demo` never touches PySpin.
- Outputs go to `<project>/output/{logs,frames,recordings}/` (resolved from `PROJECT_ROOT` in `app.py`, independent of the working directory); demo files are prefixed `dummy_`.

## Architecture (`src/espi_live/app.py`)

Everything runs in `main()` as one polling loop on the main thread, with no acquisition thread. Each iteration grabs a frame (`GetNextImage(100)` or `DemoSource.next_frame`), applies pending camera setting changes (rate-limited to 0.1 s), runs `ESPIProcessor.difference` → `.average` → `make_display`, paints via `Viewer.show`, feeds `AVIRecorder`, then `wait_for_input` pumps Tk once and sleeps out the rest of the frame interval. Tk is only serviced from this loop, so every `continue` path (camera timeout, incomplete frame) must still call `controls.update()` or the UI freezes.

- **`Controls`** owns the Tk root, control panel and log window. UI state lives in `tk.Variable`s read each frame (`mode`, `avg`, `gamma`, `palette`, `frame_title`, ...). Buttons and viewer shortcuts push one-letter actions (`s` save, `r` reset previous, `a` reset averaging, `f` capture reference, `l` demo→live) onto `controls.actions`, popped once per iteration; `q`/Esc call `controls.stop()`.
- **Modes**: `Raw` (direct pixels, optional "fast preview" that limits only painted pixels), `Previous`, `Reference`. A mode change resets averaging. `ESPIProcessor.average` returns `None` until `count` difference frames accumulate, so `display` can be `None` for several iterations.
- **`AVIRecorder`** has its own thread on a constant-rate timeline and writes the latest submitted display frame, so it may skip or repeat preview frames. The loop starts a new file when mode or averaging changes; finished recorders move to `retired_recorders` and are reaped in the loop and in `finally`.
- **`MirrorStream`** replaces `sys.stdout`/`sys.stderr` to copy output to the console, the per-run log file and the GUI log. Use `controls.write()` for logging.
- **`CameraSession`** wraps the PySpin lifecycle. `close()` tears down in SDK order (EndAcquisition → DeInit → cam_list.Clear → ReleaseInstance) and must run on every exit path. Code touching `PySpin.*` is only reachable on the camera path (the global is `None` until the first real-camera attempt).
- **Demo mode**: `DemoSource` simulates exposure/gain via brightness; `watermark_demo` stamps "DUMMY MODE ACTIVE" on every displayed, saved and recorded frame. Demo data must never be mistakable for camera data (controls, viewer, logs, exported media). The "Switch to live capture" retry must leave the synthetic stream running on failure, and on success reset measurement state and stop any demo recording.

## Rules

- Before changing PySpin, camera lifecycle, node-map or image-buffer code, read `docs/camera/README.md` and the files it points to: `spinnaker-pyspin.md` always, `blackfly-bfs-u3-63s4m-c.md` for model-specific settings, `acquisition-and-performance.md` for streaming, ROI, pixel format, frame rate, preview latency, recording or triggering.
- Verify a node is readable/writable before writing it and clamp numeric values to device limits (`set_enum`/`set_bool`/`set_float` wrapped by `try_setting` do this). Release every `GetNextImage` result in a `finally`, and `.copy()` `GetNDArray()` data that outlives the image. Preserve safe lifecycle handling on errors.
- Don't change default exposure (8000 µs), gain (0 dB), frame rate (30 fps) or recording behavior without calling out the impact. Don't assume the Raw-view display rate equals the camera's acquisition rate.
- Keep the Tk-only GUI (separate viewer, controls and log windows, documented shortcuts). Don't reintroduce OpenCV HighGUI windows alongside Tk: resizing mixed-toolkit windows caused fatal Windows crashes. Keep Tk painting inside each preview frame's deadline and don't add a second Tk event pass per frame: servicing events after the deadline previously capped a 30 fps demo at ~21 fps.
- Every function needs a docstring in the existing `Args: ... / Returns: ...` format.
- Put tentative work in `experiments/`; promote it into `src/espi_live/` only when it becomes the intended maintained implementation. Use descriptive, purpose-based filenames; date-based names belong in `archive/` or `reference/`.
- Store recordings under `output/recordings/` and still frames under `output/frames/`. Don't commit generated media.
- Keep Windows and Bash helpers in `scripts/` and update their paths whenever a referenced file moves (they resolve `third_party/` and `src/` relative to the repo root).
- Don't add a virtual-environment or dependency-management system (pyproject, lockfiles) unless asked.
- Keep `README.md` accurate when layout, entry points or setup change, and add a `CHANGELOG.md` entry for user-visible changes.
- After edits, run `python -m py_compile` on the changed files.

## Git

- `.gitignore` excludes `.venv/`, `__pycache__/`, `.DS_Store`, `.omc/`, `output/`, `research/` (never push it), Office lock files, and large binaries (`third_party/spinnaker-4.4/`, `*.avi`, `*.zip`, `*.whl`, `*.dmg`, `*.tar.gz`, `*.pptx`). Don't force-add them; note that ignored files such as the SDK zip and wheel in `third_party/flir-spinnaker-4.2/` exist only on disk.
- Don't commit unless asked.
