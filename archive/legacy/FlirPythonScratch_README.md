# FLIR PySpin Scratchpad

This project is organized into separate working, reference, vendor, and setup areas so current development is easy to find without losing prior material.

## Start here

The current ESPI program is [`active-project/espi_live.py`](active-project/espi_live.py). Open that file in VS Code and run it with the Python 3.10 interpreter that has PySpin installed. It opens separate Tk viewer, control, and log windows. The control panel has a Text size slider; the panels start to the right of the image and can be moved independently. The video keeps its aspect ratio when its window is resized. The viewer uses Pillow (`PIL.ImageTk`) to display frames without mixing two GUI event loops.

The program starts in **Raw** for direct Mono8 positioning video; this mode skips ESPI processing, brightness scaling, gamma, palette, and averaging, and uses a 1 ms display wait. **Fast Raw preview** limits only the number of pixels painted in the viewer for better responsiveness; saved frames and AVI still use the full frame, and the image remains aspect-correct. Turn it off to fill the viewer at maximum display size. Camera sliders show direct values (for example, 5.0 dB and 30.0 fps), have exact-value entry boxes, and adapt to readable camera limits. The panel separately shows applied exposure/gain and requested/resulting camera FPS when those nodes are available. Use the FPS control to request a higher rate when positioning; camera bandwidth and exposure still determine the attainable rate.

If no FLIR camera is detected (or PySpin is unavailable), startup offers **dummy/demo mode**. It opens the same controls, log, and video windows with a moving synthetic grayscale test pattern; exposure, gain, and FPS controls are simulated. A persistent **DUMMY MODE ACTIVE** banner appears in the viewer and in saved frames/recordings, which also receive `dummy_` filename prefixes. To start demo mode directly without touching a camera, run `python active-project/espi_live.py --demo` with an interpreter that has OpenCV, NumPy, Pillow, and Tkinter installed. In demo mode, **Switch to live capture** retries camera detection without restarting. If no camera is available, demo continues; if connection succeeds, measurement baselines reset to Raw view and any demo recording stops so synthetic and live frames never share a file. Demo mode is for interface and processing checks, not camera-performance measurements.

Choose **Previous** for frame-to-frame differences or **Reference** for differences against a frame captured with `f` or **Set reference**. In either measurement mode, **Scale min/max to 0–255** maps that frame's darkest and brightest difference pixels to 0 and 255. The panel shows the input range and range after scaling, before gamma or palette. To check the displayed grayscale scaling directly, choose the **Gray** palette and gamma 1.0, then toggle scaling. A constant difference image maps to all zeros when scaling is on. The controls also offer named palettes and buttons for saving and resetting frames.

The control panel also shows displayed preview FPS and average application work per input frame (image copy, processing, and display submission). This is a preview throughput measurement, not the camera's `AcquisitionResultingFrameRate`. The Wait slider can slow measurement preview; UI input is checked once per preview cycle, so a long wait also delays controls and Quit by up to that interval. AVI encoding runs on a separate worker, using the most recent preview image on a steady output timeline. When preview runs faster than AVI encoding, intermediate preview images can be skipped in the recording; when it runs slower, the last image is repeated to keep playback duration aligned with real time. The final log reports skipped preview frames and late encoder intervals. Changing view mode or averaging starts a separate AVI file.

On Windows, the app temporarily requests 1 ms timer resolution while the viewer is running so short preview waits track the requested FPS more closely; it releases that request at shutdown. This can increase power use while the app is open. The displayed FPS remains the source of truth for achieved preview rate.

On Windows, dragging a native window border can briefly pause the Tk preview until the drag ends. The app resumes without crashing, but acquisition and measurement processing in the UI loop pause too; avoid resizing while collecting measurements. A recording may repeat its latest frame during that pause.

Focus the image window for `q`/Esc (quit), `s` (save displayed frame), `r` (reset previous frame), `a` (reset averaging), and `f` (capture reference). The same shortcuts and their meanings are shown beside the controls' action buttons. The log window can be reopened with **Show log** and mirrors Python stdout and stderr. Logs are also saved under `active-project/output/logs/`; recordings go to `active-project/output/recordings/` and still frames to `active-project/output/frames/`. See [CHANGELOG.md](CHANGELOG.md) for a short history of the changes.

## Folder layout

- `active-project/` — the only place for active development.
  - `espi_live.py` — current main ESPI live viewer (previously `Dec-17-Python.py`).
  - `camera_live_control.py` — live camera-control helper.
  - `experiments/` — candidate, test, and older working scripts.
  - `output/` — recordings, frames, and Blackfly output.
  - `notes/` — development notes and reports.
- `saba-originals/` — preserved Saba scripts for reference. `for-my-camera/` contains camera-specific variants; `not-working/` holds known non-working scripts.
- `vendor-flir-spinnaker/` — unmodified FLIR/Spinnaker wheel, SDK archive, official examples, and documentation. Treat this as third-party reference material.
- `agent-context/` — maintained local research context for the BFS-U3-63S4M-C
  camera and the Spinnaker/PySpin SDK. See `agent-context/README.md` before
  planning camera-code changes.
- `setup-and-tools/` — Windows and Bash setup/launch helpers.

## Setup and quick validation

For a new Windows computer, double-click `setup-and-tools/windows/SETUP AND START ESPI.bat`. It checks for Python 3.10, OpenCV, NumPy, Pillow, Tkinter, PySpin, and the FLIR Spinnaker SDK. It asks before installing or launching anything. After it installs or repairs a dependency, close any installers and run the same file once more; a clean validation run launches `active-project/espi_live.py`.

The individual helpers remain available when troubleshooting: `install_spinnaker_sdk.bat`, `install_pyspin.bat`, and `run_pyspin_example.bat`. Configure VS Code to use the Python 3.10 interpreter containing PySpin, normally `C:\Python310\PySpin\python.exe`.

The Bash equivalents live in `setup-and-tools/bash/`.

## Working convention

Add new work only under `active-project/`. Keep experiments there until they become the maintained program. Do not modify `saba-originals/` or `vendor-flir-spinnaker/` unless you deliberately want to update the preserved reference material.
