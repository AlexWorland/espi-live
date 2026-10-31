# ESPI live viewer

A live viewer for a FLIR Blackfly (BFS-U3-63S4M-C) Mono8 camera used for ESPI (electronic speckle pattern interferometry) measurements. Python, PySpin/Spinnaker, OpenCV and Qt (PySide6), with GPU-accelerated display; runs on Windows, macOS and Linux.

The connected camera's model is detected automatically (from its GenICam identity nodes) and shown in the window title, status badge and Camera controls. Other FLIR/Teledyne models (other Blackfly S sensors, Grasshopper, Chameleon, Oryx) are named and run with the limits they report, with a note in the log when a model is untested or colour; see `docs/camera/supported-models.md`.

The analog **Sony XC-ST50** works too, through a **Hauppauge USB-Live2** capture stick, alongside a Blackfly or on its own; see [Analog camera](#analog-camera-sony-xc-st50) below.

## Quick start

```sh
# Any machine (no camera or PySpin needed): synthetic video
./run.sh --demo

# Camera PC (Windows, Python 3.10 with PySpin): double-click
scripts/windows/SETUP AND START ESPI.bat
```

Manually: `PYTHONPATH=src python -m espi_live [--demo]` (on Windows, `set PYTHONPATH=src` first). Run it with an interpreter that has the packages in `requirements.txt` (OpenCV, NumPy, PySide6); the camera runtime also needs PySpin, normally `C:\Python310\PySpin\python.exe`. PySide6 is required; without it the app exits with a message. `python -m espi_live --check` loads everything without opening a window; `python -m unittest discover -s tests` runs the camera-free tests (both with `PYTHONPATH=src`). For line coverage (kept at 100%), `pip install coverage`, then `QT_QPA_PLATFORM=offscreen python -m coverage run -m unittest discover -s tests` and `python -m coverage report`.

## Project layout

```
src/espi_live/        the application: engine (acquisition/processing thread), Qt UI
tests/                camera-free unit tests
scripts/
  windows/            camera-PC setup and launch helpers (.bat, .ps1)
  bash/               Bash equivalents
  camera_live_control.py   standalone live camera-control helper
experiments/          candidate, test and older working scripts
docs/
  camera/             curated PySpin / BFS-U3-63S4M-C / other-model / acquisition / XC-ST50 + USB-Live2 notes
  notes/              development notes and reports
third_party/          not in this repository: clone of the private espi-live-sdk (FLIR Spinnaker SDK), git-ignored
.github/workflows/    build-binaries.yml: tests, builds and releases PyInstaller binaries after every merge into main;
                      release-drafter.yml: regenerates the draft release (next version, notes) after a release or by hand;
                      enrich-feature-request.yml: Copilot enrichment of feature-request issues;
                      codex-change.yml: turns changes sent from Open in Codex into pull requests;
                      add-error-to-project.yml: auto-error issues onto the roadmap project
research/             physics background: talks, notebooks, papers, earlier code (local only, git-ignored)
archive/              superseded app versions and old scripts, kept for history
output/               logs, frames and recordings written at run time (git-ignored)
CHANGELOG.md  AGENTS.md  CLAUDE.md  requirements.txt  run.sh
```

The FLIR Spinnaker SDK (installers, PySpin wheels, docs and official Examples) is not part of this public repository. Lab members with access clone the private SDK repository into `third_party/`, where the scripts and docs expect it: `git clone git@github.com:AlexWorland/espi-live-sdk.git third_party`.

`.venv` (Python 3.14, the packages in `requirements.txt`) is only for camera-free work on a Mac. There is no macOS PySpin wheel, so the camera path runs only on the Windows camera PC (SDK 4.2).

## Prebuilt binaries

Download the newest build from **https://github.com/AlexWorland/espi-live/releases/latest**; no GitHub account is needed. The Windows and Linux archives each hold an `espi-live` folder: run `espi-live.exe` (`espi-live` on Linux) inside it, and keep the folder together, because the program loads its libraries from the `_internal-N` folder beside it (N is the build number). Nothing is unpacked at startup, unlike the single-file builds up to build 6, which extracted everything to a temporary folder on every start.

Every merge into `main` is released:

1. `.github/workflows/build-binaries.yml` (**Release** in Actions) runs after every merge. It first rewrites the **draft release** with release-drafter: the pull requests merged since the last release become the draft's notes, and their labels pick the next version tag (`vX.Y.Z`): `enhancement` raises the minor version, `major` the major version, anything else the patch version; `skip-changelog` leaves a pull request out. Label a pull request before merging it.
2. It runs the test suite on Linux, builds self-contained PyInstaller binaries for Windows, Linux and macOS, and creates the release with that tag, the binaries attached and GitHub's generated release notes (the pull requests merged since the previous release, grouped by `.github/release.yml` with the same labels as the drafter). Installed copies are offered it at once. If a test fails or line coverage of `src/espi_live/` is below 100%, or a build fails, nothing is released, and the next merge's release includes this one's changes.
3. A merge that lands while a release is building waits for it. When several wait, GitHub keeps only the newest run, and its release covers all of them.

The draft release (on **Releases**) previews the next version. Publishing it by hand still works: it creates the tag on `main` and starts the same workflow, which turns the release back into a draft while it works and publishes it again with the binaries attached (tick **Set as a pre-release** for a test build that installed copies are not offered). If that build fails the release stays a draft: delete its tag (`git push origin --delete vX.Y.Z`), fix the problem and publish again.

A release can also be started by hand: open **Actions > Release > Run workflow**, pick the branch and run it. Untick **Publish** to only test and build the branch: the archives stay on the run's page as artifacts and nothing is released. Otherwise the release is tagged with a version: the one typed into **Version** (`vX.Y.Z`), or else the version the draft proposes, or else the next patch version after the newest release; a version whose tag already exists is skipped. When the branch is not `main` or **Pre-release** is ticked it is a pre-release tagged `vX.Y.Z-pre.N` (N is the workflow run number), which installed copies are not offered and which leaves `vX.Y.Z` for the real release. After a full release the draft is refreshed to propose the next version. Its notes are GitHub's generated release notes, below anything typed into **Notes**.

| File | Camera support |
|------|----------------|
| `espi-live-windows-x64.zip` | PySpin 4.2 bundled (Python 3.10 build) |
| `espi-live-linux-x64.tar.gz` | PySpin 4.4 bundled (Ubuntu 24.04 build) |
| `espi-live-macos-arm64.zip` | `--demo` only (no macOS PySpin wheel); unsigned, run `xattr -cr "ESPI Live.app"` after unzipping |

Output goes to `output/` next to the executable, inside the `espi-live` folder (next to the `.app` on macOS). The CI builds only check that each binary starts and loads its modules (`--help`, `--check`); the bundled camera path has not been exercised against a real camera. The release workflow checks out the private SDK repository with a read-only deploy key (the `SDK_DEPLOY_KEY` secret); without it, only the macOS (demo) build succeeds.

### Automatic updates

Prebuilt copies update themselves to the newest full release (pre-releases are skipped). The Qt interface checks GitHub in the background at startup and again about every three hours (a little earlier or later, so many cameras do not ask at the same moment). A check waits while a recording or a frame save is in progress, and a failed check is only written to the log. When a newer build exists, a blue **Install update** button appears in the status bar (the versions are in its tooltip, and the log notes the build); the status bar mentions it once, and later checks that find the same build leave the button as it is. Clicking it downloads the build while measurement continues, verifies its SHA-256, then the button reads **Restart to update**: the app finishes any recording, releases the camera, swaps in the new build and starts it with the same options. Choosing not to restart installs it when you close the app. The **Check for updates** button in the status bar (or **Help > Check for updates…**) checks on demand and reports the result in a dialog; it is greyed out while a check runs, and the blue button takes its place once a newer build is found. **Help > About** shows the build number.

- `espi-live --update` updates without opening a window; `--version` prints the build; `--no-update-check` skips the startup check, the later checks, and the startup download of the source code (see [Open in Codex](#open-in-codex)).
- The program's folder must be writable: the new build is unpacked there, its `_internal-N` folder is moved in beside the running build's, and the executable is swapped, the running one being kept as `espi-live.exe.old` (Windows cannot overwrite or delete a running program or the libraries it has loaded). When the new build starts it deletes the old executable and then the old `_internal-N` folder by itself; while the old build is still closing it retries in the background for about a minute, and anything still in use then goes at the next start. `output/` and the token files are never touched, and the log lists what was removed.
- Copies that are a single file (build 6 and earlier) cannot update themselves to the folder layout: their check reports `build N has no espi-live-windows-x64.zip` (or the Linux archive). Download a current release once by hand and unpack it where the old copy is. When the new build starts, it deletes an old single-file `espi-live.exe` (`espi-live` on Linux) in the folder above its own folder or the one above that (where Windows' **Extract all** puts it), with that copy's update leftovers. It deletes only files it recognizes as single-file builds of ESPI live, and the old copy's `output/` folder (recordings, frames, logs) stays where it is.
- Builds are ordered by the workflow run number, so a copy never "updates" to an older build. Running from source never updates itself; use `git pull`.
- The repository is public, so installed copies read releases without a token. The optional `UPDATE_TOKEN` repository secret (a fine-grained token with **Contents: read-only**) is baked in when set and only raises GitHub's API rate limit; `update-token.txt` beside the program or the `ESPI_UPDATE_TOKEN` environment variable overrides it.

### Feature requests

The **Request feature** button in the status bar (also **Help > Request a feature…**) asks for a title, an optional description and an optional name, then files a GitHub issue labelled `feature-request` and `enrich` in this repository in the background. The issue includes the build, platform and video source (camera or dummy mode). **Submit** files it from the app. When it is sent, a dialog says the request was submitted; if sending fails, you can edit and retry. The description is optional: leave it blank and the request is written from the title, or write one and that text is kept and expanded.

- The `enrich` label starts `.github/workflows/enrich-feature-request.yml`, which asks GitHub Copilot (through `actions/ai-inference`) for a structured spec (summary, motivation, proposed behavior, acceptance criteria, open questions) and appends it below the original text. The label is then replaced with `enriched`. Maintainers can add `enrich` to any issue by hand to get the same result. The system prompt is `.github/prompts/enrich-feature-request.txt`.
- Builds file issues with a token from the `FEEDBACK_TOKEN` repository secret: a fine-grained personal access token limited to this repository with **Issues: read and write** only. The release build stops if that secret is empty, so a published binary always contains it (`--check` says `feedback token: present` and never prints the token). It is separate from `UPDATE_TOKEN`, so it can be revoked without breaking updates; anyone can extract it from a public binary, so the worst case is unwanted issues (and, through the `enrich` label, Copilot runs on them). `feedback-token.txt` beside the program or the `ESPI_FEEDBACK_TOKEN` environment variable overrides it. A copy without a token cannot file the request and says so.
- Enrichment needs the `COPILOT_PAT` repository secret: a fine-grained personal access token from an account with a Copilot plan, with the **Copilot Requests** permission. Without it, the workflow comments on the issue that enrichment is not configured.

### Logs and automatic error reports

The run log is `output/logs/espi_<timestamp>.log` (the newest 50 run logs and any older active runs are kept; inactive older logs and their crash files are deleted at the next start). Every line reads `HH:MM:SS LEVEL [thread] message`, and the first lines record the build and commit time, the platform, the loaded library versions (Python, NumPy, OpenCV, PySide6, PySpin/Spinnaker), the front end and the log level. `--log-level {debug,info,warning}` or the `ESPI_LOG_LEVEL` environment variable raises the detail level (`debug` adds camera node writes, settings changes and frame timing). **Help > Open log folder** opens the folder in the file manager. When the program dies in native code (a crash inside PySpin, OpenCV or the GL driver), `faulthandler` writes `espi_<timestamp>.crash`, which the next start reports, appends to that run's log and deletes (the report includes the sanitized crash stack, but the previous run's log stays local because its personal identifiers may differ); a UI stall longer than about 10 seconds dumps every thread's stack into the log while the program keeps running.

Unexpected errors — an uncaught exception on any thread, an engine or recorder failure, a Qt critical message, or the crash file of the previous run — are saved to `output/logs/pending_reports/` and sent automatically as GitHub issues labelled `auto-error`. Each error gets a deterministic ID (`ESPI-ERR-v1-…`, computed from the exception type and the code path it crossed, printed in the issue title and body), so the same bug is reported once no matter which machine, install path or run hit it: an existing issue gets one "seen again" comment per build, an issue closed as fixed and hit again in a newer build is reopened, and wontfix issues stay quiet. Reports are public, so they are scrubbed: no home folder, user or host name, camera serial number, saved file name, e-mail address or token-like string survives, and the body carries the build, platform, library versions, the scrubbed last log lines and the traceback. Expected conditions (no camera, camera busy, network failures, a full disk) are only logged, never reported.

- Release builds report by default; source checkouts don't, unless `ESPI_ERROR_REPORTS=1` is set. `--no-error-reports`, `ESPI_ERROR_REPORTS=0` or unticking **Help > Send error reports automatically** turns reporting off, and the log says how the first time a report is sent. Copies without the feedback token offer a **Report in browser…** link instead of filing silently.
- Create the `auto-error` label in the repository before enabling automatic reports. Automatic reports are added to the **ESPI Live roadmap** project by `.github/workflows/add-error-to-project.yml`, which needs the `PROJECT_PAT` repository secret (a token with **projects: write** for this repository; the app's issue token cannot reach projects).

### Open in Codex

Prebuilt copies keep the source code they were built from in `espi-live-source/` next to the program (next to the `.app` on macOS). At startup the program checks that the folder holds its own build's commit and, if not, downloads it from GitHub in the background (about 300 KB); `.espi/download.zip` inside it keeps the original download. After an update the folder is replaced with the new build's source, unless it holds changes that were never sent, which are kept.

The **Open in Codex** button in the status bar (also **Help > Ask Codex about the source code…**) opens OpenAI's Codex app (also part of the ChatGPT desktop app) on that folder, so anyone curious about how ESPI live works can ask questions that are answered from the real code. It reads **Getting source…** while the source downloads first. The first time, a dialog explains what happens and links to the Codex download page; if no app handles Codex links, a dialog says so.

- Before opening, the program writes `AGENTS.override.md` into the folder; Codex reads it instead of `AGENTS.md`. It tells Codex that the person is not a developer: answer from the code in plain words; for a change, follow `AGENTS.md`'s rules, explain the change and ask whether they are happy with it; and only after a clear yes, write `CHANGE_REQUEST.md` (title and description) and run `espi-live --send-changes`, never git.
- `--send-changes` compares the folder with the original download (text files only; caches, `output/` and the files written for Codex are left out) and files a GitHub issue labelled `codex-change` with the description, the build, the base commit, a readable diff and the exact patch (gzip and base64). It uses the same Issues-only `FEEDBACK_TOKEN` as feature requests: no credential that can write code is in the program. Sending the same changes again only prints the earlier link; later changes go out as a new issue that includes the earlier ones.
- `.github/workflows/codex-change.yml` turns such issues (filed by the owner's token, so from the owner, a member or a collaborator) into a pull request: it reads the issue with `python -m espi_live.changes extract`, applies the patch to a new `codex/issue-<n>` branch from the base commit, opens a pull request labelled `codex-change` and closes the issue with a link. Patches that touch `.github/` or reach outside the project are refused, and a patch that does not apply is reported on the issue. The workflow needs **Settings > Actions > General > Allow GitHub Actions to create and approve pull requests**.
- Prebuilt copies cannot run the tests (they have no Python), so these pull requests arrive untested; review them like any outside contribution. Running from source, **Open in Codex** opens the checkout itself with the developers' `AGENTS.md`, and changes go through git as usual.

## Using the viewer

One window holds the image, a toolbar with the view mode, Record, Freeze, Save frame, Set reference and the reset buttons, a **Controls** panel for camera, measurement and display settings, and a **Log** panel; both panels can be docked on any side, floated as separate windows, or hidden from the **View** menu, and the layout is remembered. The status bar shows a rotating **Did you know** tip, the current view's hint, FROZEN while the display is frozen, the Check for updates, Request feature and Open in Codex buttons, a separator, then the source (Camera or DUMMY MODE), the recording state, the measurement range, processing and display rates, and the value of the pixel under the cursor. The tip changes about every 12 seconds, starting on a random one; click it for the next. Right-click it, or uncheck **View > Help tips**, to turn the tips off, and that choice is remembered. The badges keep their place: a narrow window drops the tip before it crowds the dummy-mode badge, and shows it again, shortened with an ellipsis if needed, once there is room. The first launch opens wide enough for the whole tip. **Help > User Guide** (also **F1**) opens the illustrated offline [user guide](src/espi_live/user_guide.html) in the default browser.

Camera acquisition and ESPI processing run on their own thread, so moving, resizing or interacting with the window never pauses a measurement or a recording. The image is drawn by the GPU (OpenGL 3.3 through Qt): scaling, gamma and the color palette are applied in a shader. On machines without suitable OpenGL (for example some remote-desktop sessions) the app logs "Display: CPU fallback" and draws the same image on the CPU.

- **Freeze** (`Space`, toolbar or **View > Freeze display**): holds the image on screen so it can be inspected (zoom, pan, pixel readout) without saving anything. Only the display stops: the camera keeps acquiring, and averaging, reference refresh and recording carry on, so the status-bar rates stay live. An amber **FROZEN at** *time* bar over the image and a **FROZEN** badge in the status bar mark the held image; press `Space` again to return to the live view. **Save frame** saves the held image, and its `.txt` file records when it was frozen. **Set reference** still uses the live camera frame. With several cameras, each camera keeps its own frozen frame; **Switch to live capture** or turning a camera's processing off drops it.
- **Image navigation**: the toolbar's **Zoom** slider (drag, click or scroll it) zooms from fit-to-window up to 64× and shows the zoom as a percentage, where 100% is the whole image fitting the window; **Fit** (`0`) and **1:1** (`1`) jump to those levels. The mouse wheel or trackpad on the image zooms around the cursor, and dragging pans.
- **Noise reduction** (Camera section): blends each new camera frame with the frames before it, in Raw, Previous and Reference (0 = off, 95 = strongest). It settles flickering sensor noise and leaves the speckle grain alone, because the fringes come from that grain. A still speckle image therefore looks almost the same at every setting. What changes is how much the picture boils, and a trail behind anything that moves. The number is how far the control is turned up, not the share of the previous frame kept: 10 is already about three frames of memory, and 95 is about twenty (the same cap as before). It used to read the number as that share, so most of the slider did almost nothing. **Stretch to 0–255** then paints a quieter difference across the full brightness, so the fringe picture can look unchanged even though the status bar's Input range got narrower; turn stretch off to see the amplitude. It is separate from **Average N** (which smooths the fringe picture) and the **Speckle filter** (which blurs the grain). High settings trail motion and soften **Previous**-view fringes, which are the change from one frame to the next. A Reference captured with it on is itself less noisy.
- **View**: **Raw** shows camera pixels for positioning (no ESPI processing, stretch or palette; **Gamma** still applies). **Previous** shows frame-to-frame differences; **Reference** shows differences against a frame captured with `F` or **Set reference**. **Average N** averages that many difference frames to reduce speckle noise: one image per N frames, or, with **Rolling average**, a running average that still updates every frame. **Ref. refresh** recaptures the Reference every N seconds so slow drift (air currents, heat) cancels; 0 keeps one reference until you press `F`.
- **Fringe enhancement**: **Speckle filter** (Gaussian, or edge-keeping Median capped at 5 px to keep the frame rate) with **Filter width** in pixels blurs away pixel-scale speckle; keep it well below the fringe spacing (0 = off). **Invert** swaps bright and dark fringes. **Stretch clip** lets that percentage of the darkest and brightest pixels saturate so a few stray speckles don't flatten the contrast (0 = exact min/max).
- **Stretch to 0–255** maps each measurement frame's darkest and brightest difference pixels (after the clip) to 0 and 255; the status bar shows the range before and after scaling (before gamma and palette). A constant difference image maps to all zeros when scaling is on. **Gamma** (Display section, its own control) is not a second stretch: it is a curve applied afterwards, in every view, that shifts mid-tones. Below 1.0 brightens dark fringes; above 1.0 darkens bright ones. With a good stretch, leave it at 1.0.
- **Camera** sliders show direct values with exact-entry boxes and adapt to the camera's readable limits; the panel shows applied exposure/gain and requested/resulting FPS. Camera bandwidth and exposure determine the attainable frame rate.
- **Capture**: **Record AVI** writes the displayed image on a steady timeline (intermediate frames are skipped when acquisition outruns encoding; the last frame is repeated when it is slower); changing view mode or averaging starts a new file, and the log reports skipped frames and late encoder intervals. **Save frame** writes exactly what is on screen; an optional **Frame title** is added to the PNG's filename, and a title or notes also write a matching `.txt` file.
- **Shortcuts** (when not typing in a text field): `Space` freeze/resume, `S` save, `F` set reference, `R` reset last, `A` reset average, `Q`/Esc quit (Esc in a text field just leaves the field). The same list is in the View menu.
- **Tooltips**: hovering a slider, button, menu item or status readout explains what it does. A slider also shows its current value and unit, and a control with a shortcut names the key. Tips wait a moment so they do not flicker as the pointer moves, and they open beside the control so they do not cover the live image. The image itself has no tooltip.

### Which settings should I use?

Hover a processing control for what it does, when to use it, and a starting value. The same guide is under **Help > Which settings should I use?**. Two buttons at the top of **ESPI measurement** apply a starting point and leave exposure, gain, frame rate and recording alone:

- **Live preview** — aim the camera and see whether fringes form. Previous view, averaging off, noise reduction 0, speckle filter off, reference refresh 0, gray display. Switch to **Raw** to frame the object.
- **Measurement** — a still object against one reference. Reference view (press `F` once the object is still), an 8-frame block average (one clean image every 8 frames), Median speckle filter at 3 px, reference refresh 2 s, stretch on with a 0.5% clip, gray display. CLAHE stays on for when you switch to a color palette.

**Ref. refresh** trades drift cancellation against seeing slow motion: 2 s removes air and heat drift, and also any real motion slower than that. Use 0 and press `F` for one fixed before/after.

Noise reduction, averaging and the speckle filter are different steps. Noise reduction blends camera frames before the difference and keeps the speckle (it trails motion). Average N blends difference frames (block: one image per N frames; rolling: full frame rate, with a lag). The speckle filter blurs each difference image; keep its width well below the fringe spacing.

Processing order for Previous and Reference views: noise reduction → difference → average → speckle filter → stretch → invert → gamma/CLAHE → palette. Raw applies noise reduction and display gamma, without stretch or palette.

Output goes to `output/logs/`, `output/frames/` and `output/recordings/`. Recordings and saved frames carry the camera's serial number in the filename (for example `espi_raw_12345678_<time>.avi`).

### Several cameras

When more than one FLIR camera is connected, a **Cameras** section appears at the top of the Controls panel. **Display** picks the camera shown in the window. Each camera also has a **Process** checkbox: a checked camera keeps acquiring, processing and (when Record is on for it) recording while another camera is displayed. The displayed camera always processes, so its checkbox is locked on; choosing an unchecked camera in **Display** turns it on. Each row shows that camera's processed frame rate and `REC` while it records.

Each camera has its own settings: exposure, gain, frame rate, noise reduction, view mode, reference frame, averaging, fringe and display options, and its own Record state. The controls, toolbar and shortcuts always act on the displayed camera. Unchecking **Process** stops that camera (finishing its recording and releasing it) but keeps its settings for the next time; its reference frame and running average start fresh. At startup only the first camera that opens is processed. Log lines from each camera start with its serial number, and an error in one camera turns only that camera off.

Several 6.3 MP cameras streaming at once can exceed one USB 3 controller's bandwidth; watch each row's frame rate, and lower the frame rate or put the cameras on separate controllers if it drops. `--demo --demo-cameras 3` simulates three cameras for trying this without hardware.

### Analog camera (Sony XC-ST50)

The XC-ST50's composite video goes through a Hauppauge USB-Live2 into the PC, and the stick appears in the camera list as **Hauppauge USB-Live2 (video N)**. Close WinTV and Hauppauge Capture first: the stick serves one program at a time. Exposure, gain and frame rate are set by the camera's own switches, so those controls are disabled for it, and the status line shows the measured capture rate. **Single field (interlaced camera)**, on by default, shows only one of the two interlaced fields (repeated to full height), which keeps the 1/60 s between fields from blurring the fringes at the cost of half the vertical detail. Images look about 12 % wide because NTSC pixels are not square; they are not resampled. Expect coarser fringes than with the Blackfly. Wiring, driver settings and details: `docs/camera/sony-xc-st50-usb-live2.md`.

Other video devices (webcams, other capture sticks) are listed as **not yet supported** and are never opened unless you choose one. When you switch one on, it runs as delivered, and the viewer files a GitHub issue (label `camera-support`) with what the device reports, once per device type. A copy without a feedback token saves the report to `output/logs/camera-report-*.md` and shows a **Report this camera** button instead.

### Demo mode

If no FLIR camera or recognised capture adapter is detected (or PySpin is unavailable and no adapter is connected), startup offers **dummy/demo mode**, and `--demo` starts it directly without touching a camera. It opens the same windows with a moving synthetic grayscale pattern; exposure, gain and FPS are simulated. A persistent **DUMMY MODE ACTIVE** banner appears in the viewer and in saved frames and recordings, which also get `dummy_` filename prefixes. **Switch to live capture** retries camera detection without restarting: if no camera is available, demo continues; if one connects, measurement baselines reset to Raw view and any demo recording stops so synthetic and live frames never share a file. Demo mode is for interface and processing checks, not camera-performance measurements.

### Windows notes

The app temporarily requests 1 ms timer resolution while running so demo pacing tracks the requested FPS; this can increase power use.

## Camera-PC setup

Double-click `scripts/windows/SETUP AND START ESPI.bat`. It checks for Python 3.10, OpenCV, NumPy, PySide6, PySpin and the FLIR Spinnaker SDK, and asks before installing or launching anything. After it installs or repairs a dependency, close any installers and run the same file once more; a clean validation run launches the app. The individual helpers (`install_spinnaker_sdk.bat`, `install_pyspin.bat`, `run_pyspin_example.bat`) remain for troubleshooting. Point VS Code at the Python 3.10 interpreter that contains PySpin.

## Other variants

`experiments/espi_live_bfs_u3_200s6m.py` is a standalone full copy of the viewer adapted for the 20 MP BFS-U3-200S6M-C (15 fps default, 1000 ms image timeout). It predates the package layout and has not been reconciled with the package.

See [CHANGELOG.md](CHANGELOG.md) for history and [AGENTS.md](AGENTS.md) for contributor and agent guidance.

## License

ESPI live is free software under the GNU General Public License v3.0; see `LICENSE`. The FLIR Spinnaker SDK and PySpin are Teledyne FLIR's proprietary software under their own license and are not covered by it.
