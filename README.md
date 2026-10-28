# ESPI live viewer

A live viewer for a FLIR Blackfly (BFS-U3-63S4M-C) Mono8 camera used for ESPI (electronic speckle pattern interferometry) measurements. Python, PySpin/Spinnaker, OpenCV, Pillow and Tk.

## Quick start

```sh
# Any machine (no camera or PySpin needed): synthetic video
./run.sh --demo

# Camera PC (Windows, Python 3.10 with PySpin): double-click
scripts/windows/SETUP AND START ESPI.bat
```

Manually: `PYTHONPATH=src python -m espi_live [--demo]` (on Windows, `set PYTHONPATH=src` first). Run it with an interpreter that has OpenCV, NumPy, Pillow and Tkinter; the camera runtime also needs PySpin, normally `C:\Python310\PySpin\python.exe`. After editing, check syntax with `python -m py_compile src/espi_live/app.py` (needs no display or camera).

## Project layout

```
src/espi_live/        the application (app.py, `python -m espi_live` entry point)
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
  spinnaker-4.4/        4.4.0.246 installers for macOS and Linux (git-ignored, large)
reference/            preserved earlier scripts (saba-originals/), read-only
research/             physics background: talks, notebooks, papers, earlier code (local only, git-ignored)
archive/              superseded app versions and old scripts, kept for history
output/               logs, frames and recordings written at run time (git-ignored)
CHANGELOG.md  AGENTS.md  CLAUDE.md  requirements.txt  run.sh
```

`.venv` (Python 3.14, numpy + opencv-python + pillow from `requirements.txt`) is only for camera-free work on a Mac. There is no macOS PySpin wheel, so the camera path runs only on the Windows camera PC (SDK 4.2).

## Using the viewer

The viewer opens separate Tk viewer, control and log windows. The control panel has a Text size slider; the panels start to the right of the image and can be moved independently. The video keeps its aspect ratio when its window is resized. The viewer uses Pillow (`PIL.ImageTk`) to display frames without mixing two GUI event loops.

The program starts in **Raw** for direct Mono8 positioning video; this mode skips ESPI processing, brightness scaling, gamma, palette and averaging, and uses a 1 ms display wait. **Fast Raw preview** limits only the number of pixels painted in the viewer; saved frames and AVI still use the full frame. Turn it off to fill the viewer at maximum display size. Camera sliders show direct values (for example 5.0 dB and 30.0 fps), have exact-value entry boxes, and adapt to readable camera limits. The panel separately shows applied exposure/gain and requested/resulting camera FPS when those nodes are available. Camera bandwidth and exposure determine the attainable frame rate.

Choose **Previous** for frame-to-frame differences or **Reference** for differences against a frame captured with `f` or **Set reference**. In either measurement mode, **Scale min/max to 0–255** maps that frame's darkest and brightest difference pixels to 0 and 255; the panel shows the input range and the range after scaling, before gamma or palette. To check grayscale scaling directly, choose the **Gray** palette and gamma 1.0, then toggle scaling. A constant difference image maps to all zeros when scaling is on.

Focus the image window for `q`/Esc (quit), `s` (save displayed frame), `r` (reset previous frame), `a` (reset averaging) and `f` (capture reference); the same shortcuts are listed in the control panel. **Frame title** and **Frame notes** are optional: a title is added to the saved PNG's filename, and a title or notes also write a matching `.txt` file next to it.

The panel also shows displayed preview FPS and average application work per input frame. This is preview throughput, not the camera's `AcquisitionResultingFrameRate`. The Wait slider can slow measurement preview; UI input is checked once per preview cycle, so a long wait also delays controls and Quit by up to that interval. AVI encoding runs on a separate worker using the most recent preview image on a steady timeline: when preview outruns encoding, intermediate images are skipped; when it is slower, the last image is repeated to keep playback aligned with real time. The final log reports skipped preview frames and late encoder intervals. Changing view mode or averaging starts a new AVI file.

Output goes to `output/logs/`, `output/frames/` and `output/recordings/`.

### Demo mode

If no FLIR camera is detected (or PySpin is unavailable), startup offers **dummy/demo mode**, and `--demo` starts it directly without touching a camera. It opens the same windows with a moving synthetic grayscale pattern; exposure, gain and FPS are simulated. A persistent **DUMMY MODE ACTIVE** banner appears in the viewer and in saved frames and recordings, which also get `dummy_` filename prefixes. **Switch to live capture** retries camera detection without restarting: if no camera is available, demo continues; if one connects, measurement baselines reset to Raw view and any demo recording stops so synthetic and live frames never share a file. Demo mode is for interface and processing checks, not camera-performance measurements.

### Windows notes

The app temporarily requests 1 ms timer resolution while running so short preview waits track the requested FPS; this can increase power use. Dragging a native window border can briefly pause the Tk preview (and measurement processing) until the drag ends, and a recording may repeat its latest frame during the pause; avoid resizing while collecting measurements.

## Camera-PC setup

Double-click `scripts/windows/SETUP AND START ESPI.bat`. It checks for Python 3.10, OpenCV, NumPy, Pillow, Tkinter, PySpin and the FLIR Spinnaker SDK, and asks before installing or launching anything. After it installs or repairs a dependency, close any installers and run the same file once more; a clean validation run launches the app. The individual helpers (`install_spinnaker_sdk.bat`, `install_pyspin.bat`, `run_pyspin_example.bat`) remain for troubleshooting. Point VS Code at the Python 3.10 interpreter that contains PySpin.

## Other variants

`experiments/espi_live_bfs_u3_200s6m.py` is a standalone full copy of the viewer adapted for the 20 MP BFS-U3-200S6M-C (15 fps default, 1000 ms image timeout). It predates the package layout and has not been reconciled with `src/espi_live/app.py`.

See [CHANGELOG.md](CHANGELOG.md) for history and [AGENTS.md](AGENTS.md) for contributor and agent guidance.
