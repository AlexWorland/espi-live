# Changelog

## 2026-11-23 — A second piezo driver (Thor 2), and two camera streams at once

### Previous state

- The NI DAQ window drove one reference piezo (Thor 1, on AO 1) plus the frequency generator on the only other analog output, and the Phase view's piezo stepped whichever camera was displayed. The window showed one camera at a time.

### Current state

- **Two piezo drivers.** The DAQ window now has **Piezo driver (Thor 1)** and **Piezo driver (Thor 2)**, each with its own output (Thor 1 defaults to AO 1, Thor 2 is assignable to AO 0), mode, voltage, step, safety limit, follow-frame-rate and calibration. The USB-6251 BNC has only two analog outputs, so with both drivers in use the frequency generator must be off or on a counter (a PFI square wave, already supported); two triangles share the one sample clock, so they must agree on a frequency (the window says so). Held levels and the Phase view's steps run independently on the two outputs. Each driver's phase steps never move the other driver's level.
- **Piezo↔camera pairing.** "Thor 1 steps camera" / "Thor 2 steps camera" in the DAQ window pair each driver to a camera for the Phase view, so two cameras can phase-shift at once (for example the front and the back of a sample). "Follow the displayed camera" (the default) keeps the original behaviour; each camera keeps its own calibrated step voltage.
- **Two streams at once** (View menu, needs two processing cameras): **Split two streams** shows two panes side by side, each named, and clicking a pane makes that camera the one the controls edit; **Overlay two streams** blends both cameras' displayed images into one (half and half, scaled to the displayed camera's size when they differ, dummy-mode watermark kept); **Second stream window…** opens a separate window showing another camera's live stream with its own zoom and pan.
- **Where to plug in…** and the wiring notes (`docs/notes/piezo-thorlabs-mdt694a-wiring.md`) now show one-Thor and two-Thor options with a drawing of each chain, the panel sockets and the bench order. The safety checklist lists each switched-on driver separately.
- Defaults are unchanged: Thor 2 is off until assigned, and the display stays one stream until split, overlay or the second window is chosen.

## 2026-11-23 — DAQ and NI USB controls in the remote-control API

### Previous state

- The control server (`python -m espi_live control …`) could drive the cameras only; the frequency generator and piezo driver on the NI USB device could be changed only in the app's NI DAQ window, so a terminal or agent had to reach for the GUI.

### Current state

- `GET /daq` reports the NI device, the settings last applied, the saved ones, the safety checklist's state, the cables the settings need and auto-tune's state; `GET /status` now includes a DAQ summary. `POST /daq` runs every action the NI DAQ window offers: `scan`, `connect`, `apply` (any DaqSettings field, merged over the last applied or saved settings and saved like Apply does), `zero`, `self_test`, `step` (hold the piezo at a level), `attach`/`detach` (the Phase view's piezo, on any camera), `checklist` (shows the safety checklist on the app's screen; ticking it arms the device and applies) and `tune` (auto-tune on/off). The checklist gate is untouched: no drive signal reaches a real device until it is ticked off for the connection.
- The terminal client wraps it all: `control daq` shows the state, and `control daq scan|connect DEVICE|apply KEY=VALUE…|zero|self-test|step VOLTS|attach [CAMERA]|detach|checklist|tune on|off` run the actions.
- Changes made through the API are logged with a `Control:` line naming exactly what changed, the NI DAQ window's form follows the new values (device, settings, checkboxes and drop-downs), and demo cameras follow the outputs. DAQ requests coordinate with the window's event polling, so nothing is lost while a terminal command waits for the device's answer.

## 2026-11-23 — Frequency generator and piezo window fits the screen

### Previous state

- The Frequency generator and piezo (NI DAQ) window opened at a fixed size, so on a short screen its lower controls and the Apply and other buttons ran off the bottom.

### Current state

- The window sizes itself to its contents but never beyond 90% of the screen's usable height or width. A taller form scrolls, while the note and the buttons (Apply, All outputs to 0 V, Test outputs, Where to plug in, Close) stay in view below it. A size you choose is kept; reopening only shrinks a window that has outgrown the screen.

## 2026-11-23 — Remote control of the running app from the terminal

### Previous state

- Once the windowed app was running, every setting change had to go through its window; there was no way to query or adjust it from a terminal, so scripted or agent-driven experiments had to drive the GUI by hand. (`--headless --control` drove headless runs only.)

### Current state

- While the Qt app runs it serves a control API on `http://127.0.0.1:8364/` (loopback only; the next free ports are tried if 8364 is taken, and the chosen port is written to `output/control.json`). `GET /` describes the whole API — every setting with its type, choices and range, and every command; `GET /status` reports every camera's settings, limits and status; `POST /settings` changes them; `POST /command` runs one-shot commands (`save` with optional title/notes, `reference`, `reset_previous`, `reset_average`, `switch_live`, `display`, `processing_on`, `processing_off`); `GET /log` returns the run log's tail.
- A built-in terminal client covers the same ground: `python -m espi_live control status|cameras|get|set|run|log`, with `--camera` to pick a camera by index, serial or label and `--json` for raw output. `save` saves the frozen frame when the display is frozen, like the Save button.
- Settings changes apply through the running engines and the window's controls follow along; the change is saved to the camera's profile like a hand-made one. Camera values are clamped to the device's reported limits, other values to their controls' ranges. Start the app with `--control-port N` to fix the port or `--no-control` to turn the server off. The API is unauthenticated and bound to 127.0.0.1, so nothing outside the machine can reach it.

## 2026-11-23 — Headless runs can drive the NI DAQ: the config file's "daq" block, and follow, tune and calibrate over the control protocol

### Previous state

- A batch run (`--headless --config`) could not drive the NI DAQ at all: no connect, no apply, no piezo. Only a `--control` run could, one JSON command at a time, and even then the two closed loops stayed out of reach: the **Follow the camera's frame rate** checkbox was accepted into the settings but never acted on (the window's timer did that), **Auto-tune** was a button in the DAQ window only, and **Calibrate phase step** was a menu click that no `--control` command reached.

### Current state

- **The `"daq"` config block** drives the DAQ from a batch run's config file: `{"daq": {"device": "Dev1", "settings": {…DaqSettings fields…}, "arm": true, "tune": true}}`. It connects, applies the settings **with the outputs switched off first** (so the safety checklist can be recorded for the sockets the settings use), arms (only with `--daq-checklist-done`, the same gate as before) and then switches the outputs on. `"tune": true` also starts the experimental auto-tune on the first running camera. A block the DAQ refuses stops the run with its reason instead of running blind, and the session is closed either way.
- **`tick` runs the window's job in a batch run**: while the run lasts, a `piezo_follow_fps` triangle is kept at the frequency the frames need (re-applied only past the 3% tolerance, each change restarting the triangle) and the auto-tune keeps measuring and applying, with no window and no timer.
- **New `--control` commands**, one JSON line each:
  - `daq_follow {camera?}` — set a frame-rate-following triangle's frequency from the camera's measured frame rate now (the window's timer in one shot). The reply carries the frame rate, the frequency and the formula's sentence.
  - `piezo_tune {on?, camera?}` — start or stop the experimental auto-tune headless. While it runs it is driven from `tick` and reports `{"event": "tune", state, scale, text}` lines; `status` carries a `tuning` object. Stopping with `on: false` keeps the tuned frequency, as in the window.
  - `phase_calibrate {camera?}` — the **Calibrate phase step** measurement over the protocol: it switches the camera to the Phase view, waits for the five 5-step sets and replies with the step measured in degrees and the new volts per step, kept for the next `daq_apply`. A step too far from 90° or a Phase view that stops is an error with the reason.
- **Auto-tune stops on the same events as in the window**: a camera that stops, the Phase view, a new DAQ connection, a zero, a failure, a fault or a checklist that no longer covers the outputs — and the DAQ fault path also detaches the stepper and sends `Drive()` to the cameras, as before.
- Defaults, exposure, gain, frame rate and the safety rules are unchanged; the checklist gate stays in `DaqController`, and every way out still sets the outputs to 0 V.

## 2026-11-22 — Experimental auto-tune of the piezo triangle, and a piezo debug overlay

### Previous state

- The piezo triangle's frequency came from the formula (Follow the camera's frame rate) or was typed in. Nothing checked, from the image, whether the reference phase really moved by 180° between frames; an open-loop piezo's travel, the mirror angle or the frame timing could leave it off.

### Current state

- **Auto-tune (experimental)** in the DAQ window's piezo section tunes the applied triangle's frequency from the camera frames for a 180° step per frame, where the Previous-view difference is strongest and least sensitive to small changes of the step (so it flickers least), and keeps it there. It holds the piezo still at the triangle's centre for a few seconds to measure the camera noise, compares frames one and two apart (at 180° frames two apart match: D(2)/D(1) falls to its least), finds the side of 180° with a 5% trial change, jumps close and finishes with small steps. Each change restarts the triangle and is logged. It stops by itself on a camera switch, the Phase view, new settings or the outputs stopping, and if it cannot find the step it returns to the computed frequency and says so. Turning it off keeps the tuned frequency. In simulation (turning points, ±10% step wobble, camera noise) it settled within 6% of 180° from steps between 0.5× and 1.8× of it.
- **View → Piezo debug overlay** shows, in a strip along the top of the image area (covering neither the image nor the dummy-mode watermark), the frames received per second, the triangle applied and the formula's frequency, the step measured from the frames (D2/D1, D1, noise, flicker) and the tuner's state. On screen only: saved frames and recordings never show it.
- Defaults are unchanged: both are off until turned on.

## 2026-11-22 — The piezo triangle can follow the camera's frame rate

### Previous state

- The NI DAQ piezo triangle ran at the frequency typed into the DAQ window. Changing the camera's frame rate (for example from 30 to 20 fps) changed how far the reference phase moved between frames, so the frequency had to be worked out and retyped by hand.

### Current state

- **Follow the camera's frame rate** (Triangle mode, **Devices → Frequency generator and piezo (NI DAQ)…**) sets the triangle's frequency from the rate frames arrive from the displayed camera, so the reference phase moves by **Phase per frame** (default 180°, half a wavelength of path) between successive frames. It is set again whenever the frame rate moves the frequency by more than 3%; each change restarts the triangle at its bottom and is logged.
- **Piezo scale**: **Calibrated** uses the volts per 90° measured by **Calibrate phase step** (now also saved for this: frequency = frame rate × volts per 90° ÷ (Top − Bottom) at 180°); **Manual** uses **Piezo travel** (default 0.1 µm/V, about a Thorlabs PE4's 15 µm at 150 V), **Amplifier gain** (default 15, the MDT694A at 150 V) and **Mirror angle** (default 45°, the lab's) at the HeNe wavelength, 632.8 nm. With the defaults, Manual gives 0.249 Hz at 30 fps and 0.166 Hz at 20 fps for a 0.5–9.5 V triangle.
- Following needs a scale: with Calibrated and no calibration yet, the window says so and Apply stays off. The triangle's **Frequency** box shows the followed value and takes four decimals.
- Defaults, exposure, gain and frame rate are unchanged; following is off until ticked. A Calibrate phase step result from an earlier build is not on record for Calibrated: run it again.

## 2026-11-22 — Headless control accepts a byte-order mark; NI MAX simulated DAQ devices are marked

### Previous state

- `espi-live.exe --headless --control` answered the first command piped from PowerShell 5.1 with `not valid JSON: Expecting value: line 1 column 1 (char 0)`, because PowerShell starts piped text with a byte-order mark (U+FEFF) and the line reached `json.loads` with it.
- A device created as simulated in NI MAX (for example `SimDev1`, "NI Simulated DAQ Device") looked exactly like hardware in the DAQ window and in the headless replies.

### Current state

- A byte-order mark at the start of an input line is ignored, and a line holding only one (and whitespace) counts as blank.
- The DAQ window's device list and the headless `daq_scan`, `daq_connect` and `status` replies label an NI MAX simulated device "(NI MAX simulation, no hardware)", and the headless device dicts carry `ni_simulated` next to `simulated`. The safety gate is unchanged: only the app's own `Simulated` device skips the checklist, so an NI MAX simulated device still takes no drive signal until its checklist is recorded.

## 2026-11-22 — Drive the NI DAQ and cameras from a script (`--headless --control`)

### Previous state

- Headless runs could not touch the NI DAQ or the Phase view, so testing the generator, the piezo or phase stepping meant clicking through Devices → NI DAQ.

### Current state

- `espi_live --headless --control` reads one JSON command per line on stdin and answers with one JSON line on stdout (logs go to stderr and the log file only). Commands: `status`, `daq_scan`, `daq_connect`, `daq_arm`, `daq_apply`, `daq_zero`, `daq_self_test`, `piezo_step`, `phase_attach`/`phase_detach`, `set_camera`, `save`, `reference`, `wait`, `quit`; they are listed in `src/espi_live/headless_control.py`. It works with `--demo` and the simulated DAQ device, and it starts even with no camera.
- The safety rules are unchanged: a real device takes no drive signal until the checklist is recorded, which a control run can do only with `--daq-checklist-done` and the `daq_arm` command (the simulated device needs neither). Every way out (`quit`, closed stdin, Ctrl+C/SIGTERM, an error) sets the outputs to 0 V; a DAQ self-check fault arrives as an `{"event": "fault"}` line. A control run never uploads.
- `headless_control.py` has tests and counts toward the 100% coverage gate.

## 2026-11-22 — The Windows build can reach the NI DAQ again

### Previous state

- The prebuilt Windows program (v0.17.0, build 62) could not use a real NI DAQ: `espi-live.exe --check` said `NI DAQ: pyLabLib and nidaqmx not installed`, and the DAQ window offered only the simulated device. nidaqmx reads its own version from its package metadata when it is imported, and the build bundled the code without that metadata, so the import failed and pyLabLib treated nidaqmx as missing.

### Current state

- The build bundles the package metadata of nidaqmx and its dependencies, so the program finds a connected USB-6251 (or any NI-DAQmx device) again.
- The build's smoke test fails when the program cannot import pyLabLib and nidaqmx, so a build without NI DAQ support is never published. The Quick release smoke test also gained the video-device enumerator check that Release already had.

## 2026-11-22 — Headless batch runs (`--headless`)

### Previous state

- The program needed a window and PySide6 to run at all, so an unattended recording meant leaving the viewer open.

### Current state

- `espi_live --headless --config run.json --duration 600 --record` opens the cameras, applies the config's settings, records an AVI per camera, optionally saves a PNG every `--save-interval` seconds, and exits after the duration or on Ctrl+C / SIGTERM, finishing the recordings first.
- When it ends, the run's recordings and frames are copied to the private captures repository (same repository and token as the scheduled uploads: `ESPI_CAPTURES_TOKEN` or `captures-token.txt`), shrunk in a copy when over their size budget (stills 3 MB, clips 20 MB). `--no-upload` keeps them local; demo runs never upload.
- No DAQ or Phase-view control in headless runs. Has no tests yet.

## 2026-11-22 — The NI DAQ drives the piezo amplifier with a triangle, never below 0 V

### Previous state

- In the lab, a Wavetek Model 19 function generator fed a triangle (about 0.28 Hz, +1 V to +10 V) into the Thorlabs MDT694A piezo amplifier's EXT INPUT. The program could not make that ramp: the DAQ window's piezo driver only held a level or followed the frequency generator with a sine.
- The piezo driver accepted negative levels down to its safety limit. The MDT694A takes 0–10 V, and a negative input drives the piezo below 0 V.

### Current state

- **Triangle at its own frequency**, a new piezo mode in **Devices → Frequency generator and piezo (NI DAQ)…**: **Bottom**, **Top** and **Frequency** (0.01–1000 Hz). The ramp starts at the bottom and is built from whole samples of the 20 MHz clock; the line under the controls says the frequency the device really produces. Both analog outputs share one sample clock, so the triangle needs the frequency generator off or on a counter, and the window says so.
- The piezo driver never goes below 0 V in any mode, phase steps included. The Voltage, Bottom and Top boxes start at 0 V, and settings or steps that would go lower are refused in words.
- **Where to plug in…** at the bottom of the DAQ window opens a drawing of the USB-6251 BNC's front panel, laid out after NI's own front-panel figure (PFI BNCs at the top, APFI 0, the analog outputs, the analog inputs with their FS/GS switches, USER 1/2 and the 30-way screw-terminal strip), with the sockets the current settings use marked and numbered, and beside it where each cable goes, plus the safety notes. It follows the form while open; a counter on PFI 8–15 is marked on its screw-terminal row, since those have no BNC.
- **Safety checklist before any drive signal.** The first Apply that switches an output on for a real NI device opens a checklist (MDT694A knob at zero, only the DAQ's cable on its EXT INPUT with the Wavetek unplugged, its OUTPUT never to the DAQ, the safety limit and VOLT LIMIT suit the piezo, the generator's cable and limit, someone watching). It lists checks only for the outputs being switched on (the piezo's when **Piezo on** is ticked, the generator's when **Generator on** is), plus the watcher check. **Allow drive signals** stays off until every box is ticked. Each role is allowed once per connection; switching on another role, or moving one to another socket, asks again for it. The simulated device never asks. The gate is in the DAQ controller, so Phase-view stepping from saved settings also waits for it: at startup the Phase view no longer starts stepping a real piezo by itself.
- **Self-checks.** Every analog output is set to 0 V the moment a device connects (clearing anything a crash or another program left on). About once a second the outputs are read back through the USB-6251's internal `_ao0_vs_aognd`/`_ao1_vs_aognd` channels (no wiring); a reading outside what was written (±0.1 V) sets every output to 0 V and says which output read what. If the window stops responding for 2 s while an output is on, the DAQ thread sets everything to 0 V by itself. **Test outputs…** steps each output to 0.5 V and back and checks the readback (asked first: switch the MDT694A off).
- Wiring: the DAQ's **AO 1** BNC goes to the MDT694A's **EXT INPUT**; the Wavetek is no longer needed in the piezo chain. Set the piezo's safety limit to 10 V at most. `docs/notes/piezo-thorlabs-mdt694a-wiring.md` has the instruments' specifications, the scope reading of the old ramp and the wiring.
- The real USB-6251 also lists its `freqout` channel among its counter outputs; the DAQ window offered it as a generator output named "FREQOUT (sine)", which pyLabLib cannot use for the square wave. Only `ctr` counters are listed now.
- Dummy mode does not sweep the simulated reference mirror with a DAQ triangle: the image keeps the triangle's centre, and the **Follow DAQ** note says so.

## 2026-11-22 — The iPhone difference view says why it has no image, and falls back to luma video

### Previous state

- When the Bayer RAW stills never produced an image, the difference view stayed on a black screen headed "Choosing the camera source…" with nothing to act on: the still loop re-polled silently every 50 ms, and the only way out was to close the view.

### Current state

- Before the first image the header says which source is running, how many stills were asked for, returned and discarded, and what is holding the next still up (session not running, camera interrupted, previous still not back, photo connection missing, disabled or inactive, or the Bayer format no longer offered, with the formats that are).
- If no Bayer still produces an image within 6 s, the view switches to luma video by itself and shows an orange note with the reason.
- A **Source** switch (Rawest (Bayer RAW) / Luma video) in the view forces either one. The view starts on the rawest source each time it opens.
- Not confirmed on a phone: why the Bayer stills stalled on the reporter's iPhone. The new header text and note name it on the next run.

## 2026-11-22 — The mouse wheel no longer changes drop-downs

### Previous state

- Sliders and number boxes ignored the wheel, but every drop-down (camera, smoothing, scaling, phase steps, the iPhone options, the simulated object) still stepped to its next entry on a wheel notch. Scrolling the Controls panel with the pointer over one silently changed a setting.

### Current state

- All drop-downs ignore the wheel, so the Controls panel scrolls instead, and take focus only by a click or Tab. Click, the open list and the arrow keys still choose an entry.
- A test sends wheel notches to every drop-down, slider and number box in the window, so a plain one added later fails.

## 2026-11-21 — Always-on capture uploads

### Previous state

- Only run logs and error reports were sent to the developer. Images and clips left the PC only when someone sent them by hand.

### Current state

- While a real camera (FLIR or a supported capture adapter; never demo, never an iPhone) is processing, the program saves and uploads, per camera and without any prompt or switch: a **raw and a processed still** every 30 minutes and a **raw and a processed 5 s clip** every 6 hours, to the private `AlexWorland/espi-live-captures` repository. Each capture has a `.json` sidecar with the view, camera model and every processing setting, so it can be re-run offline. No camera serial number appears in names or sidecars.
- Limits: 60 MB a day per PC (30 MB of it kept for clips), 3 MB per still file and 7 MB per clip file (larger ones are downscaled, never lossy-compressed), a 200 MB on-disk queue (`output/captures/pending/`) that drops its oldest captures. A clip is skipped while you record or when fewer than 2 processed frames a second are published; a half pair is never uploaded.
- The user guide says that images and clips are sent. `--check` reports whether the captures token is present; the startup log says `Capture uploads: on/off`.
- Engine: new `set_raw_tap` / `latest_raw`, the camera frame before processing, copied only while a capture needs it.
- To turn it on: create the private repository `AlexWorland/espi-live-captures`, issue a fine-grained token with **Contents: read & write** on that repository only, and store it as the `CAPTURES_TOKEN` secret. Accepted risk: the token baked into public builds can read every uploaded image (images cannot be scrubbed); `CaptureUploader.put` is the one place to change for a write-only relay.
- Design and plan: `docs/superpowers/specs/2026-11-21-always-on-capture-design.md`, `docs/superpowers/plans/2026-11-21-always-on-capture.md`.

## 2026-11-21 — Noise pipeline by default, and phase-shifting ESPI

### Previous state

- Every camera started with Average N 1, a one-frame reference and the speckle filter off. Averaging, when turned on, averaged the difference images, whose camera noise is always positive, so it settled into a grey floor instead of averaging out.
- The analog Sony XC-ST50 started in the LabVIEW flow (a one-frame automatic reference, Fixed gain 42×, no averaging or filters).
- The speckle filter was Gaussian or Median; a width large enough to quiet the speckle also blurred dense fringes away.
- The NI DAQ could step the reference piezo only by hand (Step −/+), and only subtraction fringes (|live − reference|) could be shown.

### Current state

- **Default noise pipeline** for every camera, analog included: **Average camera frames, then subtract** on with **Average N** 4 (camera and capture noise fall by √4 = 2 before the difference), **Ref. frames** 16 with a reference that Reference view captures by itself, and the new **Oriented** speckle filter at 11 px. The full frame rate needs Average N 1. A camera with a saved profile keeps the values it was left with; **Reset camera to defaults…** brings the pipeline in. The run log's settings summary now says whether camera frames or differences are averaged, and the reference frame count.
- The analog camera starts in Reference view with Fixed gain 42× and the pipeline; the **LabVIEW flow** is a toggle, off by default (it also turns the averaged reference off now). A camera PC whose analog profile was saved in the LabVIEW flow keeps it.
- **Oriented** speckle filter: smooths along the local fringe direction (measured from the structure tensor) instead of across it, and acts like a Gaussian where no direction stands out. On simulated subtraction fringes it matched or beat a Gaussian of the same width at every fringe spacing tried (correlation with the true fringes 0.844 against 0.757 at a 20 px spacing); about 45 ms on a 6 MP frame.
- **Phase view** (phase-shifting ESPI): with the piezo driver applied in the NI DAQ window, the engine steps the piezo between frames (Average N frames per step, the **Settle** frames after each move dropped) and shows each pixel's phase change since the phase reference, wrapped to one fringe with its sign: mid-grey is no change, black to white 2π. **5-step (Hariharan)** (default) tolerates a wrong step (1° phase error at a 75° step in simulation, against 7.5° for **4-step**) and measures the step made; the change is filtered as sine and cosine (**Phase filter**, 9 px). **Calibrate phase step** measures the step over five sets and saves the voltage per step for 90°. When that device connects (at startup or from the DAQ window) the displayed camera switches to the Phase view and its toolbar button appears; opening the DAQ window alone never drives the piezo, and every step stays inside the piezo's safety limit. In dummy mode with the simulated device, the synthetic camera becomes a speckle interferometer that follows the simulated piezo.
- The comparison window names frame-first averaging correctly ("4 camera frames averaged before the difference").
- Research and measurements: `docs/notes/espi-noise-and-capture-research.md`.

## 2026-11-21 — Dummy mode simulates an ESPI signal

### Previous state

- Dummy mode (`--demo`) showed a moving test pattern (a gradient, a checkerboard and a bouncing circle). It exercised the plumbing, but the ESPI processing meant nothing on it: no speckle, no fringes.

### Current state

- **Dummy mode is a simulated ESPI signal**: an ideal camera (no sensor noise; brightness proportional to exposure × gain; 8-bit, clipping at 255) looking at a speckle interferometer whose specimen is a **drumhead**, an ideal circular membrane clamped at its rim. Previous and Reference views, averaging, gain or stretch, the LabVIEW flow and the noise floor now run on a signal whose correct result is known, to check the processing by eye.
- A **Simulated object** control (Camera section, demo cameras only, saved per demo camera) chooses **Still**, **Slow deformation** (the default: pressed in to 8 fringes and released every 12 s), **Vibration** (drum mode (1, 1) at 1000 Hz) or **Follow DAQ** (the NI DAQ window's generator drives the drum's resonant modes; the Simulated device works). A note under it says what drives the drum.
- The DAQ piezo moves the simulated reference mirror in every case (1500 nm per DAQ volt; the default 0.05 V step is about 85°), and its sine modulates the reference in Follow DAQ.
- A **Reference piezo (triangle, decorrelation)** checkbox (demo only, off by default, saved per demo camera) ramps the reference mirror with a slow triangle wave (1 Hz, 1.2 µm peak to peak), as on the bench: the reference phase changes about 90° per frame at 30 fps, so in Previous view a vibration shows |J0| fringes with a **white nodal line** instead of a black image.
- Vibration is averaged exactly over each exposure: a whole number of drive cycles per exposure gives steady J0 fringes, a fractional number flickers as on a real camera.
- Each demo camera has its own speckle pattern. The **DUMMY MODE ACTIVE** watermark, banner and `dummy_` file names are unchanged.

## 2026-11-21 — Analog camera read as raw YUY2 brightness

### Previous state

- The Sony XC-ST50 through the Hauppauge USB-Live2 was read as colour: on Windows DirectShow turned the stick's YUY2 into BGR (OpenCV's DirectShow backend cannot hand over YUY2), on Linux OpenCV did, and the viewer then turned BGR back into grey. The two roundings added about ±1 grey level of error per pixel, more where noise in the colour bytes made a channel clip. The LabVIEW flow's Fixed gain multiplies differences by 42, so that error showed.
- `docs/camera/sony-xc-st50-usb-live2.md` said Linux runs the stick with `cx231xx`; that is only the older `2040:C200` model.

### Current state

- For a recognised adapter, the viewer reads the brightness bytes of the YUY2 video directly: on Windows through Media Foundation with conversion off (after DirectShow has selected the composite input, so opening takes about a second longer), on Linux through V4L2 with conversion off.
- **Grey-level scale unchanged:** the stick sends video-range brightness (black 16, white 235), and the colour path expanded it to 0–255. The raw bytes are expanded with the same formula, so levels, the Difference gain 42 and saved profiles keep their meaning; only the conversion error is gone. As before, values above 235 read 255.
- The raw path is used only when the device delivers YUY2 at 720 × 480 and its first frame's mean brightness is within 6 grey levels of the colour path's; otherwise, or on any error, the viewer uses the colour path exactly as before. The log line `Analog luma: ...` and camera reports say which path is in use. Setting `ESPI_ANALOG_LUMA=bgr` forces the colour path.
- `scripts/check_analog_luma.py` (run on the camera PC with ESPI Live closed) shows the delivered format and layout, the raw brightness range, a same-frame comparison with the colour conversion, whether the colour path really had the expanded scale, and which path ESPI Live chooses.
- The XC-ST50 notes describe both USB-Live2 revisions: `2040:C200` (Conexant CX23102, Linux `cx231xx`) and the Rev E1 `2040:C220` (Empia em2828X, Linux support only from the 2026 `em28xx` patches).

## 2026-11-21 — Difference view on the iPhone

### Previous state

- The ESPI Camera app only sent the camera image to ESPI Live over USB; seeing fringes needed the computer.

### Current state

- A **Difference view** button on the phone shows `|image(n) − image(n−1)|` full screen, computed on the phone's GPU with no computer or cable, and no gain, averaging or filtering.
- It uses the rawest data the selected back camera has: back-to-back Bayer RAW stills (red photosites only, for the 633 nm laser; about 5–10 a second, with the photo's white minus black level as white), otherwise 10-bit, then 8-bit luma video. The source, its bit depth, size and rate are always shown on screen.
- Exposure, ISO and focus can be changed in the view. While it is on, no frames go to the computer and ESPI Live is told why; **Done** restores streaming.

## 2026-11-21 — USB-Live2 found again in the Windows build

### Previous state

- The Windows build (v0.14.8) listed no video devices, so with the Hauppauge USB-Live2 plugged in the app said "No FLIR camera or recognised capture adapter detected". The log only said `cv2_enumerate_cameras is not installed`. The package was in the build, but its native module could not load: the runtime DLL it needs sits in a `cv2_enumerate_cameras.libs` folder, which PyInstaller left out (it kept one copy of the shared DLL, under `pandas.libs`).
- The adapter table knew only USB ID `2040:C200`; a USB-Live2 on Windows 11 reports `2040:C220`, so it matched by name only.

### Current state

- When `cv2_enumerate_cameras` fails to import in the built program, the app adds the build's `*.libs` folders to the DLL search path and tries once more, so it finds the DLL wherever PyInstaller put it.
- The log and `--check` now give the real reason: `is not installed`, or `could not be loaded: <the ImportError>`.
- The release workflow's smoke test fails unless `--check` reports `cv2_enumerate_cameras available`, so a build that cannot list video devices is no longer published.
- `2040:C220` is recognised as a Hauppauge USB-Live2, next to `2040:C200`.

## 2026-11-21 — iPhone Bayer RAW mode

### Previous state

- The iPhone sent the video pipeline's 8-bit luma only, so the phone's own noise reduction, sharpening and tone mapping were already in every frame, and the whole processing chain worked on 8-bit values.

### Current state

- The iPhone section has a **Capture** choice: **Video** (as before) or **Bayer RAW**. RAW takes Bayer RAW stills (about 5–10 per second, binned 12 MP) with the phone's image pipeline bypassed, keeps one colour plane (**RAW colour**: red for a 633 nm laser, green for 532 nm, or blue; two green sites are averaged) and sends it as 16-bit values with the black and white levels. The ESPI Camera app has the same two choices on its own screen. The app must be updated from TestFlight; an older app ignores the new commands and stays on video.
- RAW frames are *deep*: float32 grey levels on the 8-bit scale (black level 0, white level 255, fractions kept). Noise reduction, frame averaging, reference capture, difference, averaging, Speckle filter, outlier rejection, noise floor, peak hold and the Auto stretch all keep that precision, and the picture is rounded to 8 bits only at the final scaling. Auto stretch and the status bar's Input range therefore work on differences far below one 8-bit level (the range shows fractions such as `0.07–0.31`). Calibrate noise suggests a floor of 0 when a RAW still scene's noise is under one level. Fixed gain has the same units as for 8-bit cameras, so a dim RAW difference needs a large Difference gain; Auto stretch is the usual choice.
- The link protocol is unchanged for video: a RAW frame uses pixel format Mono16 and a 16-byte extension after the frame metadata (see `docs/iphone/README.md`).
- Limits: rolling shutter, uneven frame timing and a lower frame rate make RAW unsuitable for pulsed lasers or phase-stepped capture. The RAW path has not been run on a phone yet.

## 2026-11-21 — iPhone back cameras listed one by one

### Previous state

- A plugged-in iPhone was one camera, **iPhone (USB)**, and the ESPI Camera app always used its wide back camera. The ultra wide and telephoto cameras could not be used.

### Current state

- **Each back camera of an iPhone is its own entry in the Display drop-down**: an iPhone 16 Pro Max lists **iPhone Wide 1× (USB)**, **iPhone Ultra Wide 0.5× (USB)** and **iPhone Telephoto 5× (USB)**; a phone with one back camera still shows **iPhone (USB)**. Choosing a lens switches the phone to that camera and reads its formats and limits again.
- The phone streams **one camera at a time**: choosing another lens of the same phone turns the lens that was showing off (settings kept, logged), and the Process boxes of the phone's other lenses are unavailable. Only the wide camera opens at startup or on **Switch to live capture**.
- Each lens has its own saved profile; the wide camera keeps its old serial, so its saved settings carry over.
- **Needs the new ESPI Camera app build** (TestFlight): the app reports its back cameras in HELLO and switches on a new `lens` command. With an older app the phone is listed as before, as one **iPhone (USB)**.

## 2026-11-21 — Pixel values window

### Previous state

- The only numeric readout was the pixel under the pointer, and it showed the displayed value, after gain, averaging and scaling, so it could not show whether anything had actually moved.

### Current state

- A **Pixels (P)** toolbar button opens a window with a square of the image enlarged: 1×1, 2×2, 4×4, 8×8, 16×16, 32×32 or 64×64 camera pixels (8×8 at first; the choice is remembered). Each pixel is a cell with its integer value written on it, shaded from black (0) to white (the largest in the square), and the square's position, min, mean and max are shown above it.
- In Previous and Reference view the values are the raw absolute difference in camera levels (0–255), taken before the noise threshold, Difference gain, averaging and Scaling, so display settings do not change them: 0 means the pixel did not change, and movement raises it. With averaging before the difference on, they are the difference of the averaged frames. In Raw difference they are that view's |live − previous|; Raw view shows camera levels.
- The square starts at the image center, is outlined in yellow on the main image, and moves to wherever you click the image. It refreshes ten times a second, has its own **Freeze**, and follows the displayed camera. The outline is not part of saved frames or recordings.

## 2026-11-21 — Per-camera profiles and saved presets

### Previous state

- Each camera had its own settings only while the program ran; every start began from the defaults again (exposure 8000 µs, gain 0 dB, 30 fps; the LabVIEW flow for the analog camera).
- Showing a camera whose exposure, gain or frame-rate range was narrower than the value still on screen could send the previous camera's value, clamped, to the newly shown camera.
- After **Switch to live capture**, the live camera kept the values set on the synthetic camera.
- Only the three built-in processing presets existed, and they never hold exposure, gain or frame rate; there was no way to keep your own setup.

### Current state

- **Each camera keeps its own profile between runs**: exposure, gain, frame rate, the view and every processing and display control, including the LabVIEW flow's state. It is saved about a second after it stops changing, on a camera switch and on close, under the camera's serial number (FLIR), USB ID and name (capture devices) or demo name, and restored before the camera opens. **A camera with a saved profile now starts from its last values instead of the defaults**, and the log says so with its exposure, gain, frame rate and view. A camera with nothing saved still starts from exactly the defaults above, and exposure, gain and frame rate are still clamped to the camera's limits. Recording is never saved.
- **Reset camera to defaults…** (Camera section, with a confirmation) puts the displayed camera back to its defaults and clears its saved profile.
- **Presets** (Camera section, also in the simple interface): **Save as…** names and stores the displayed camera's settings (an existing name asks before it is replaced), **Apply** puts the chosen preset on the displayed camera only (a preset from another camera model fits exposure, gain and frame rate into this camera's limits and logs each change, then `Preset applied: <name> to <camera>`), and **Delete** removes one after a confirmation. The built-in Live preview, Measurement and Peak hold buttons are unchanged and not listed there.
- With several cameras, the Camera section names the camera being edited (**Settings of** *camera*).
- Fixed: showing another camera no longer sends it the previous camera's clamped exposure, gain or frame rate, and the live camera after **Switch to live capture** gets its own saved profile or its defaults instead of the demo's values.
- Older stored values that no longer exist (such as `palette` and `clahe`) are ignored, and a damaged value falls back to the default with a log line.

## 2026-11-21 — Color palettes and CLAHE removed

### Previous state

- The Display section offered a **Palette** list (Gray, Turbo, Jet, Hot, Viridis, Plasma, Inferno) and a **CLAHE (color palettes only)** checkbox. CLAHE did nothing while the palette was Gray, and the Measurement preset turned it on.

### Current state

- The display is always grayscale, with the display gamma. **Palette** and **CLAHE** are gone from the controls, the presets, the guide and the settings, and the diagnostics line no longer reports CLAHE.
- Saved PNGs and AVI recordings are unchanged: they hold the same gray values as the screen, written as before. `processing.apply_palette` and `palette_rgb_table` became `apply_display(image, gamma)` and `display_rgb_table(gamma)`.
- The guide and help text now give the display order as brightness, gain and contrast, then gamma, which is what the program does.
## 2026-11-21 — Compare button is shown in the Simple UI

### Previous state

- The Simple UI hid the **Compare (C)** toolbar button, so the window with the Previous, Current and Difference images (or the Reference, Current and Difference ones) could only be opened with the `C` key until **Simple UI** was turned off.

### Current state

- **Compare (C)** is shown in the Simple UI, next to **Set reference (F)**, and opens the same window. Its tooltip now describes both views: in Previous view the earlier frame being subtracted, the current frame and their difference; in Reference view the stored reference, the current frame and their difference.

## 2026-11-21 — Quieter difference images: frame averaging and a noise floor

### Previous state

- A still scene in Previous or Reference view showed TV-like grain, worst on the analog Sony XC-ST50: a frame difference is never zero, because camera noise differs in every frame. Fixed gain 42× turns 1–5 grey levels of noise into mid-grey and white.
- **Average N** averaged the difference images. Each difference is an absolute value, so that only smoothed the grain into an even grey floor; it never brought a still scene to black.
- **Noise threshold** could hide the grain, but it cut faint fringes along with it.

### Current state

- **Average camera frames, then subtract** (ESPI measurement, off by default): Average N averages the camera frames and the averages are subtracted, which lowers the noise itself by √N. A steady change, such as time-averaged fringes, stays. One image appears per N frames; Rolling average does not apply.
- **Subtract noise floor** with **Measure noise floor (object still)** (Fringe enhancement, off by default): the measurement records each pixel's still-scene difference level for at least 2 s and 16 images. With the box ticked, that level is taken off every difference after averaging and the speckle filter, so a still scene goes black. Faint changes that last across the average still show. The map is per pixel because noise grows with brightness. It only applies under the settings it was measured with; otherwise the status bar asks for a new measurement.
- The LabVIEW flow and the presets turn both off.

## 2026-11-21 — Raw difference: live minus previous, with no processing

### Previous state

- Every difference view went through the processing chain: noise reduction, single field, filters, averaging, Fixed gain or Auto stretch, palette, gamma and display tone. On the analog camera path there was no way to see the plain difference between two frames as captured.

### Current state

- A **Raw difference** toggle on the toolbar shows |live − previous| of the raw camera frames and nothing else: every processing step and display setting is bypassed while it is on. The button turns red and the view buttons grey out; turning it off returns to your view and settings unchanged.
- The camera's own exposure and gain are not touched (the analog camera sets those on its switches). Recordings made in it are named `espi_rawdiff_*.avi`, and the comparison window shows the two raw frames that were subtracted.

## 2026-11-21 — Simple UI button

### Previous state

- The simple interface (only the basic controls) was fixed at build time by `SIMPLE_UI` in `qt_app.py`; showing every control meant editing the source.

### Current state

- A **Simple UI** button at the left of the toolbar switches between the basic controls and every control, while the program runs. Nothing is rebuilt or reset: settings, collapsed sections and the camera keep their state, and the hidden controls keep working in both modes. **View > Help tips** and the rotating tip return with the full interface.
- The choice is remembered for the next run. A new install still starts in the simple interface (`SIMPLE_UI` is now only that default).

## 2026-11-21 — iPhone camera over USB

### Previous state

- Only FLIR cameras and the analog Sony XC-ST50 (through a USB-Live2) could feed the viewer.

### Current state

- An iPhone running the new **ESPI Camera** app (`ios/ESPICamera/`, distributed through TestFlight as ESPI-Live-Camera) is a camera over its USB cable. Frames are the camera's 8-bit luma, sent uncompressed. When USB is busy the phone skips a frame rather than compressing it.
- The phone's exposure, ISO, focus and white balance are locked. HDR, low-light boost, distortion correction and stabilization are off.
- On Windows, iTunes from apple.com provides the USB connection (Apple Mobile Device Service); macOS has it built in. No network is involved.
- The phone appears in the camera list as **iPhone (USB)**, after any FLIR camera or capture adapter. It is opened at startup only when neither is present, and **Switch to live capture** finds it.
- Everything is controlled from ESPI Live:
  - **Exposure** sets the phone's exposure in µs.
  - **Gain** sets its ISO: 0 dB is the lowest ISO, +20 dB is ten times that.
  - A new **iPhone** section (shown only for a phone) sets frame rate, frame size, a centre crop, a cap on frames sent, the locked focus, and **Autofocus once, then lock**.
  - Changes made on the phone's own screen are reflected back.
- A lost link (the app left the screen, the cable was pulled) is re-opened automatically every 2 s.
- The phone is a rolling-shutter colour camera with noise reduction that cannot be turned off, so the viewer marks it as a qualitative source. Details are in `docs/iphone/README.md`.

## 2026-11-21 — The camera drop-down is always shown

### Previous state

- The **Cameras** section at the top of Controls, with the **Display** drop-down, was hidden unless two or more cameras were detected. With a single camera there was no camera list in the window, so it looked as if the camera switcher had been removed.

### Current state

- The **Display** drop-down is always shown. With one camera it lists that camera, so the window names the camera in use.
- The **Process** note and the per-camera checkboxes (with their rate and `REC` readouts) still appear only when more than one camera is detected, since a single camera has nothing to run in the background.
- Nothing else changes: choosing a camera in **Display** still shows it and starts its processing if it was off.

## 2026-11-21 — Every merge into main publishes a quick release

### Previous state

- Merging into `main` released nothing: **Quick release** (and **Release**) had to be started by hand, and merges piled up in the draft release.

### Current state

- **Quick release** also runs on every push to `main` (every merged pull request) while the project is in active development. It is the same untested build as before, published as a full release, so installed copies are offered each merge: the tag is the next patch version and the notes are GitHub's generated notes.
- If merges land faster than a build finishes, GitHub keeps only the newest waiting run, so one release can cover several merges. Running it by hand, with the **Pre-release** and **Publish** options, works as before, and **Release** from `main` is still the tested release.
- Deleting the `push:` trigger in `quick-release.yml` restores hand-started releases only.

## 2026-11-21 — Frequency generator and piezo driver on an NI DAQ

### Previous state

- ESPI live controlled only cameras. The specimen's drive signal and the reference piezo's voltage came from separate instruments, set by hand.

### Current state

- **Devices → Frequency generator and piezo (NI DAQ)…** runs both from an NI multifunction DAQ such as the USB-6251. Each role is assigned its own output: the frequency generator to **AO 0** or **AO 1** (a sine with amplitude and offset) or to **Counter 0** or **Counter 1** (a 0–5 V square wave on a chosen **PFI** terminal), and the piezo driver to an analog output. A role can be left **Off**, and an assigned output that is switched off is held at 0 V. Two roles cannot share one output.
- The piezo **holds a voltage**, with **Step −** and **Step +** to move it by a set amount, or follows the generator with a **sine at the generator's frequency** with its own amplitude, centre and **phase**. Both analog outputs run on one sample clock, so that phase is exact.
- Each role has a **Safety limit** (1 V by default) that no setting may exceed; problems are explained in the window and Apply stays disabled until they are fixed. The window shows the frequency the device will really produce (the sine is built from whole samples of a 20 MHz clock), and the log records every setting applied.
- **All outputs to 0 V** stops everything. The outputs also go to 0 V after a device error and when ESPI live closes, before the cameras are released. Closing the window leaves them running.
- Without NI support (the `pylablib-lightweight` and `nidaqmx` packages, now in `requirements.txt` and the Windows setup script, and NI's NI-DAQmx driver, Windows and Linux only), the window offers only a **Simulated device (no hardware)**, marked in red, and says what is missing. `--check` reports whether NI DAQ support is installed. A missing driver or an unplugged device is logged as a warning, not reported as an error.
- Frames are not synchronised to the drive yet; `docs/notes/ni-usb-6251-integration.md` describes how that could be done.

## 2026-11-21 — Quick release is offered to installed copies

### Previous state

- **Quick release** always made a pre-release tagged `vX.Y.Z-dev.N`, which installed copies are never offered, so a quick build had to be downloaded and unpacked by hand.
- Each workflow numbered its builds with its own run number. The updater orders builds by that number, so a quick build could not simply be made a full release: its number (for example 3) was far below the installed copies' (the newest release was build 33), and no copy would have updated to it. A copy installed from one workflow's build could also rank above the other workflow's next build and never be offered it.

### Current state

- **Actions > Quick release** publishes like **Release**: a full release tagged `vX.Y.Z` (the version the draft proposes, else the next patch), marked latest, with GitHub's generated notes. Installed copies are offered it through **Check for updates** and the status-bar button, and the draft release is refreshed afterwards. The tests are still skipped, and the notes say so.
- A new **Pre-release** checkbox keeps the old behavior: a `vX.Y.Z-dev.N` pre-release that installed copies are not offered. **Publish** still builds without releasing.
- The `version` job of **Release** and **Quick release** now picks the build number: the run number, or one above the newest build any published release's notes name, whichever is higher. Both workflows share one sequence, so every build, tested or not, is newer than the ones before it. Pre-release tags (`-pre.N`, `-dev.N`) use this number, so they skip ahead of the old run numbers. Both workflows also share a concurrency group, so a run waits for the other's.
- Installed copies, `build-info.json` and the release notes' build line are unchanged.

## 2026-11-21 — Compare works in Previous view

### Previous state

- **Compare** (`C`) always showed the stored reference, even in Previous view, which does not use one. The window gave no way to see the two frames Previous view actually subtracts.

### Current state

- In Previous view the Compare window shows **Previous (N frames back)**, the **Current frame** and the **Difference**, live and together. The previous panel is the exact frame the difference subtracted (the processor keeps it as `ESPIProcessor.previous`), after single field and noise reduction like the current frame, so Frames back above 1 is shown correctly. The difference is the main window's image, averaging included.
- With Average N above 1 the window shows the live previous and current frames and a count ("averaging: 3 of 10 frames") until the first average completes; after that all three images update together each time one does. Without an earlier frame yet the panels say so.
- **Freeze** holds all three, **Save all three** writes `previous`, `current` and `difference` PNGs with the notes file naming how many frames back, and **Set reference** is greyed out in Previous view. Changing the view while the window is open switches between the reference and Previous panels. Reference and Raw view behave as before.
- `ComparePacket.reference` and `reference_info` are now `baseline` and `baseline_info`, with `view` and `baseline_label` added.

## 2026-11-21 — The mouse wheel scrolls the panel, not the sliders

### Previous state

- Scrolling the Controls panel with the mouse wheel changed any slider or number box under the pointer, so exposure or gain could change by accident. The toolbar's Zoom slider also zoomed on a wheel notch.

### Current state

- The mouse wheel over a slider or a number box in the Controls panel scrolls the panel and never changes the value. Dragging, clicking, the arrow keys and typing a number still change it, and a wheel no longer moves keyboard focus into a number box.
- The toolbar's **Zoom** slider ignores the wheel too. Zoom is still set by dragging, clicking, the arrow keys, the **Fit** and **1:1** buttons, and the wheel or trackpad on the image.

## 2026-11-21 — Basic controls only, and a quick release

### Previous state

- The controls panel and toolbar showed every setting at once (frame rate, noise reduction, averaging, the fringe-enhancement and display tools, presets, Freeze, Compare, the reset buttons, the zoom bar and the interface size), which overwhelmed new users.
- A release always ran the full test suite and the native update tests first.

### Current state

- Only the basics are rendered: the Raw/Previous/Reference views, Record AVI, Save frame, Set reference, the camera picker, the camera's model and status, Exposure and Gain. Everything else is created but not shown (`SIMPLE_UI` in `qt_app.py`; set it to `False` to render every control again). The shortcuts (S, F, R, A, C, 0, 1, Space), mouse-wheel zoom and the View menu still work, and the hidden settings keep their defaults. This is temporary while the interface is redesigned.
- **Actions > Quick release** (`quick-release.yml`) builds the Windows program without running the tests, for quick development builds. It only makes a pre-release tagged `vX.Y.Z-dev.N`, which installed copies are never offered. Use **Release** for a tested release.

## 2026-11-15 — Peak hold preset for the analog camera

### Previous state

- Seeing the drum's vibration pattern on the Sony XC-ST50 (through the USB-Live2) meant setting about ten controls by hand: Previous view, a noise threshold, the speckle filter, peak hold, steady range and contrast. Pressing **Live preview** or **Measurement** afterwards left peak hold and the other fringe-enhancement controls on, because those presets only set the controls they had always set.

### Current state

- A **Peak hold** button sits beside **Live preview** and **Measurement** (top of **ESPI measurement**, and in **Help > Which settings should I use?**). It selects Previous view, noise threshold 6 levels, a Median speckle filter at 3 px, a 1 s peak hold window, stretch with a 0.5% clip and a 1 s steady range, and S-curve contrast 30%, with noise reduction, averaging, reference refresh, local normalization and CLAHE off. The threshold and filter run before peak hold, so the noise is cut before peak hold can keep it.
- It leaves the **Sensitivity floor** as it is: analog cameras start at 30, and **Calibrate noise** (drive off) measures your own. Exposure, gain, frame rate and recording are unchanged too.
- Peak hold uses Auto stretch, Previous compared with the frame just before (frames back 1) and no outlier filter, like the other presets; none of the three touches the **LabVIEW flow** toggle.
- Every preset now sets the fringe-enhancement and display controls it does not use back to off (threshold, steady range, even brightness, peak hold, local normalization, brightness, gain, S-curve contrast), so nothing carries over from the last preset.

## 2026-11-15 — Sensitivity, peak hold and display tone controls (PR #61)

### Previous state

- With the analog XC-ST50, Previous and Reference looked like full-contrast grain: a still scene's few grey levels of noise were stretched to 0–255, and the stretch range jumped from frame to frame.
- A driven drum barely showed in Previous view, because each frame's exposure averages many vibration cycles.
- The Controls panel was one long column.

### Current state

- **Sensitivity floor** (grey levels): Auto stretch and Local normalization never span fewer, so noise stays dark. Video-capture devices start at 30; other sources at 0, as before. **Calibrate noise (object still)** measures a still scene for 2 s and sets the floor to twice its peak.
- **Noise threshold**: raw differences below it become 0 right after the subtraction.
- **Steady range**: Auto stretch's range shrinks only gradually, so the contrast stops pumping.
- **Peak hold window**: each pixel shows its largest difference within the last N seconds, which makes a steady vibration build up in Previous view.
- **Local normalization** and **Even out beam brightness** (both off by default).
- **Display**: **Brightness**, **Gain** and **S-curve contrast** in one lookup table, in every view, before the palette; saved frames and recordings match.
- Each Controls section collapses from its heading, and the state is remembered.
- Everything new is off by default except the analog sensitivity floor.

## 2026-11-15 — The LabVIEW flow for the Sony XC-ST50

### Previous state

- The analog Sony XC-ST50 showed one interlaced field doubled to full height, which threw away half of the rows.
- Previous and Reference were always scaled by **Stretch to 0–255**, or not at all. With the XC-ST50's 3–5 grey levels of capture noise, the stretch turned a still scene into full-contrast grain.
- Previous always compared with the frame just before. Smoothing (Gaussian or Median) ran only after averaging.
- The LabVIEW program the lab used with this camera (ESPI 1.5) was not in the repository.

### Current state

- **Full frames by default** for interlaced cameras, as the LabVIEW VI used them. **Single field** is still there, now off.
- **Scaling** chooses **Fixed gain** (each difference × **Difference gain**, 1–64, saturating at 255, as the VI's IMAQ Multiply did) or **Auto stretch** (the old stretch, with Stretch clip). The default Difference gain is 42×, the VI's saved Constant. The XC-ST50's own settings use Fixed gain; the Blackfly and demo mode keep Auto stretch.
- **Gain before averaging (LabVIEW order)** applies the fixed gain to every frame and averages those, then multiplies the mean by **Average multiplier**, as the VI did. Off, the raw differences are averaged and the gain is applied once.
- **Frames back** (ESPI measurement): Previous compares with the frame N back, so slow changes show instead of cancelling. The VI got the same effect by waiting between grabs.
- **Outlier filter** (Fringe enhancement): on every difference frame before averaging, a pixel further from its neighbourhood's mean than the neighbourhood's spread becomes that mean (NI Vision's Lowpass rule, from the VI). Off by default.
- **LabVIEW flow** (a toggle beside the presets) switches the displayed camera to the VI's processing with its saved defaults (decoded from the VI: Constant 42, Multiplier 1, BY FRAME, Average and Low Pass Filter off, no wait) and captures a reference, as the VI did at start; Reference view then captures one whenever it has none. With averaging on it sums N + 1 frames and divides by N, as the VI's loop does. **It is on by default for the XC-ST50.** Turning it off restores your own settings. Exposure, gain, frame rate, recording and the flip are left alone.
- The log's settings line names the scaling, frames back and outlier filter.
- `docs/labview/` holds the VI, its block diagram and a decoded description of its flow, published with permission.
- **Note for the Blackfly:** nothing changes until you pick Fixed gain or LabVIEW flow. A setting of the old "Stretch to 0–255" off now means Fixed gain; at a Difference gain of 1 it shows the raw difference, as before.

## 2026-11-14 — Compare the reference, and average it over several frames

### Previous state

- The reference was the single frame on hand when `F` (Set reference) was pressed. With Noise reduction at 0 that frame's sensor noise was frozen into every later Reference difference, and Average N, which averages differences, could not remove it. On the analog XC-ST50, whose differences are only a few grey levels, that grain could be most of the picture.
- Nobody could see the reference: it lived only inside the processing, and Save frame saved the difference. A bad reference (taken mid-drift, or before Noise reduction was raised) could not be told apart from a noisy measurement.
- Pressing `F` and then raising Noise reduction compared smoothed frames with a noisy reference, leaving a fixed grain pattern, and the log did not show the order of the two.

### Current state

- **Compare** (toolbar, `C`) opens a window with three live images of the displayed camera side by side: the stored **Reference**, the **Current frame** (both as Raw view shows them: gamma, display tone and flip applied) and the processed **Difference** (in Reference view exactly the main image; in other views one difference from the reference with the same filters, without averaging). A line under each says how the reference was taken (frames, noise reduction, age) and how the difference was made; without a reference the panels say so. It has its own **Freeze**, **Set reference (F)** and **Save all three** (three PNGs exactly as shown, with one notes file carrying the title, notes and how the reference was taken). The images are made only while the window is open, at most 10 a second.
- **Ref. frames** (ESPI measurement, 1–60, default 1) makes `F` average the next N frames into the reference, at the same point as before (after single field and noise reduction). The status bar counts them ("Capturing reference: 7/30 frames") and the old reference stays in use until the new one is complete; the log says "Captured reference frame (average of N)". 1 behaves exactly as before. **Ref. refresh** averages Ref. frames frames on each refresh, counted from the start of the last capture; when averaging outlasts the refresh, references are captured back to back and the log says so. A view change keeps a capture going; changing Single field or Noise reduction cancels it.
- The log now records the noise reduction each reference was taken at. When Noise reduction changes while a reference is held, the log and the Reference view's hint say so ("Reference was taken at noise reduction 0; press F to retake"); the reference is not retaken by itself.

## 2026-11-14 — Interface size slider

### Previous state

- Text, buttons, sliders, panels and the status bar had one fixed size. On a 1280 × 720 laptop or a high-resolution screen the window was too big or too small, and the only remedy was the operating system's display scaling, which needs a restart of the program.

### Current state

- **Interface size** (Controls panel, Display section) scales the whole window while it runs: text, buttons, sliders, check boxes, icons, docks, tooltips, dialogs and the status bar. It runs from 70% to 200% in 5% steps; 100% is the look it had before. Drag the slider and release (it applies on release, so the slider does not move away from the pointer), type a percentage, or press **Reset interface size to 100%**.
- **View → Interface size** has **Larger**, **Smaller** and **Reset to 100%**, on `Ctrl++` (or `Ctrl+=`), `Ctrl+-` and `Ctrl+0` (`Cmd` on a Mac). The plain `0` and `1` keys still fit the image and show it at actual size.
- The size is remembered and applied before the window opens the next time. The controls panel and the log keep their share of the window when the size changes.
- A size that would make the window's smallest size larger than the screen (200% on a small screen, or a resolution change while the program runs) is lowered until it fits, and the log says so. 100% and smaller are never changed.
- The image is not scaled: it still fills the room the panels leave. Status-bar wrapping is unchanged; at larger sizes it simply wraps onto more rows.
- On the Mac the menu bar belongs to the system and keeps its size; the camera PC (Windows) scales it with the rest.

## 2026-11-14 — The status bar wraps instead of crowding

### Previous state

- Every status-bar item shared one row. On a laptop-width window the readouts squeezed the "Did you know" tip down to "Did yo…", cut the camera name to "Hauppauge USB…" and left no room for the view hint.

### Current state

- The status bar adapts to the window width. When everything fits it is one row, as before. When it does not, the readouts and buttons wrap onto further rows beneath the tip and hint, so none is shortened or hidden. Each row is only as tall as it needs to be: at 1280 × 720 the bar is two compact rows (44 px, against 27 px for one), and a 1920-pixel-wide window still fits it all on one row.
- The view hint ("Raw view: camera pixels, …") always shows its whole sentence. The tip takes the rest of its row and is shortened with an ellipsis only when little is left.
- The source badge widens to fit a long camera name ("Hauppauge USB-Live2 …") instead of cutting it.
- Temporary messages (for example "Checking GitHub for a newer build…") still replace the tip and hint while the readouts stay visible.

## 2026-11-14 — Release notes after an update

- The first start after an update shows a **What's new** window with the notes from the GitHub Releases page for every full release the update moved past, newest first, each with a link to its release page. The download and platform notes every release repeats are left out.
- The notes are fetched with the update check and saved beside the program (`espi-live-whats-new.json`) when the update installs, so the window works offline and appears only once, also after `espi-live --update`. Updating from a build older than this one installs normally but shows no notes, since that build does not save them.

## 2026-11-14 — The window title shows the version

- The main window's title now starts with the release version, for example `ESPI live v0.10.0 — BFS-U3-63S4M` or `ESPI live v0.10.0 — DUMMY MODE`. The version is the release's tag: the Release workflow bakes the tag chosen by its `version` job (`vX.Y.Z`, or `vX.Y.Z-pre.N` for a pre-release) into the build as `VERSION`.
- **Help > About**, `--version`, the log header and error reports name the version before the build number. A source checkout, and a build made before this change, shows `dev` in the title and leaves the version out of the description. Updates are still ordered by build number only.

## 2026-11-14 — Open in Codex uses the Codex command-line tool

### Previous state

- **Open in Codex** opened a `codex://` link for the Codex app. On a computer without the app, Windows reported the link as handled and showed its own "Get an app to open this 'codex' link" box, so the program's explanation never appeared.

### Current state

- The program looks for the `codex` command. If it is installed, **Open in Codex** (and **Help > Ask Codex about the source code…**) opens it in a new terminal window in the `espi-live-source` folder (a console window on Windows, Terminal on macOS, the desktop's terminal on Linux). If not, a dialog says Codex is missing and its **How to install Codex…** button opens the install page. A terminal that cannot start is reported separately.
- The first-use dialog now asks for the Codex command-line tool and a ChatGPT account (sign-in happens the first time `codex` runs), instead of the Codex app.

## 2026-11-14 — Releases are started by hand again

### Previous state

- Every merge into `main` started the **Release** workflow, which tested, built and published a release, so each merge was offered to installed copies at once and several quick merges meant several releases.

### Current state

- Merging into `main` releases nothing. `release-drafter.yml` runs after every merge and keeps the draft release up to date (next version and notes from the merged pull requests' labels), so merges stack up in one draft.
- Run **Actions > Release > Run workflow** from `main` when the changes are ready: it tests, builds and publishes one release covering everything merged since the last one. Everything else about the workflow is unchanged: the **Version**, **Notes**, **Pre-release** and **Publish** inputs, and publishing the draft by hand.

## 2026-11-13 — Run logs are sent to the developer automatically

- Release builds now upload their run logs, without any prompt or message, to the developer's **private** repository `espi-live-logs`, so a problem can be looked into from the history of runs without asking anyone to find and send log files. Each run's log appears as `logs/<date>/<time>-<machine>.log`; `<machine>` is a random ID, not the computer's name.
- Logs are sent at startup (earlier runs not yet sent, including ones that crashed) and every 15 minutes while the program runs. They are scrubbed exactly like error reports (no user or host names, folders, camera serials, e-mail addresses or tokens) and capped at 5 MB each.
- There is no switch for log uploads; the error-report switches (**Help > Send error reports automatically**, `--no-error-reports`, `ESPI_ERROR_REPORTS=0`) affect error reports only. The user guide mentions that logs are sent. `--check` now says whether the build has a logs token.
- Builds embed a new `LOGS_TOKEN` secret that can only write to the logs repository. A build without it still succeeds, with a warning, and does not upload.

## 2026-11-13 — Search the user guide

- The user guide (**Help → User Guide**, **F1**) has a search box at the top of the page that stays in view while you scroll. Press `/` to jump to it; the browser's own Ctrl+F / Cmd+F still works as before.
- Matches are highlighted as you type (two or more characters, upper and lower case alike, including phrases that run across a line break or bold text). The counter shows "3 of 17"; **Enter** and **Shift+Enter** (or the ↑ ↓ buttons) step to the next and previous match and scroll it into view, opening a collapsed troubleshooting answer when the match is inside one. **Esc** clears the search and restores the page exactly as it was. A search with no results says **No matches**.
- The contents list shows only the sections that contain the search, each with its number of matches. This replaces the sidebar's "Filter sections…" box, which only looked at section titles.

## 2026-11-13 — Release builds are Windows-only

### Previous state

- The **Release** workflow built and published Windows, Linux and macOS binaries, and the **Update end-to-end tests** workflow ran the native update tests on all three.

### Current state

- Only the Windows binary is built and published, and the native update tests run on Windows only. The Linux and macOS jobs are removed from `build-binaries.yml` and `test-updates.yml` because those platforms are not functional yet and the only user is on Windows. The camera-free test suite that gates a release still runs on Linux; it builds nothing. Restore the Linux and macOS matrix entries and steps from git history to bring them back.

## 2026-11-13 — The laptop's built-in camera is left out

- A laptop's built-in camera is no longer listed. Before, the Dell laptop's **Integrated Webcam** (USB `0c45:674c`) appeared as "not yet supported", and switching it on filed a camera report (#44). Built-in cameras are never ESPI sources, so they are now left out of the camera list: they are not shown, opened or reported, and the log says which were skipped (`Ignoring the built-in camera …`).
- A camera counts as built-in when its USB ID is a known laptop module (so far the Dell/Microdia `0c45:674c`, including its IR camera) or its name says so: "Integrated Webcam", "Integrated Camera", "Integrated IR …", "IR Webcam", "IR Camera" or "FaceTime". USB webcams, other capture sticks and the Hauppauge USB-Live2 are listed as before.
- Video devices are numbered (`video 1`, `video 2`, …) without the skipped cameras, so on a laptop the capture stick may now be `video 1` where it used to be `video 2`.

## 2026-11-13 — The window fits the screen, and the status bar has room for its readouts

- The window now fits the screen it is on, whatever its resolution or display scaling: on a laptop the whole image and the controls are visible instead of running under the taskbar and off the right edge. It refits when the resolution, scaling or taskbar changes and when it moves to another monitor; a saved size from a larger monitor shrinks to fit. A maximized window is left alone.
- The Controls panel scrolls when the screen is too short for every control, rather than making the window taller than the screen.
- Wrapped notes in the Controls panel (such as the Noise reduction note) show every line; before, they could be cut off and overlap the controls above them.
- **Check for updates**, **Request feature** and **Open in Codex** in the status bar are now small icon buttons (circular arrow, light bulb, `</>`), named in their tooltips, so the camera, recording, range and frame-rate readouts are no longer cut off. While one is busy it is greyed out, its tooltip says what it is doing, and the status bar says so briefly. An available update still shows as the labelled blue button.

## 2026-11-13 — Flip horizontally

- New **Flip horizontally** button in the Display section mirrors the image left to right (a flip across the vertical axis) in every view. Saved frames and recordings are flipped the same way; the dummy-mode watermark stays readable. Off by default, so existing output is unchanged. It is a per-camera setting.

## 2026-11-13 — Log evidence for intermittent blur and lost detail

- The log now records what is behind a few seconds of blurred or reduced-resolution video. `Frame gap` and `Incomplete-frame burst` lines show late or incomplete frames from the camera link, `Camera stream counters rose` shows frames the camera reported lost (where the model exposes those counters), and `Sharpness drop` / `Sharpness recovered` lines give the start, duration and depth of each drop for both the camera frame and the displayed frame. Every line carries the mode, noise reduction, averaging, smoothing, exposure, gain, frame rate and measured camera rate in force.
- `--log-level debug` adds a `Display:` line whenever the view size, zoom or image size changes, and update checks, source-copy downloads and error-report sends log when they start and finish, so they can be matched to an episode.
- Nothing about acquisition, processing or display changed; the sharpness check measures a shrunk copy a few times per second.

## 2026-10-31 — Verify updates through download, restart and cleanup

- Added real-worker Qt and CLI integration tests against a local release service, covering automatic and manual checks, download consent, corrupt and interrupted downloads, retries and preservation of recordings and tokens.
- Added native PyInstaller build-to-build tests for Windows, Linux and macOS, including periodic discovery, restart now or later, `--update`, argument preservation, multiple-engine shutdown and old-build cleanup. They run on pull requests and gate release builds.
- If an engine cannot stop during shutdown, the app now leaves the running installation untouched and logs that the update was not installed. Previously the shutdown check only prevented relaunch after the files had already been replaced.

## 2026-10-30 — A restarted camera keeps its exposure, gain and frame rate

### Previous state

- Turning a camera off and on again started it at 8000 µs, 0 dB and 30 fps, while the sliders still showed the values saved for that camera. It stayed there until one of those three settings was moved.

### Current state

- A camera that is turned back on starts at the exposure, gain and frame rate its sliders show. A new camera still starts at 8000 µs, 0 dB and 30 fps.

## 2026-10-30 — Leveled logs and automatic error reports

### Previous state

- The log was untyped text: `log.write(...)` lines with no levels, no startup header and no pruning of old log files. Tracebacks reached the log only from the engine's crash path and the front ends; exceptions in Qt slots, the recorder or background workers were lost. Qt warnings and criticals were not logged at all, and a native crash (inside PySpin, OpenCV or the GL driver) left nothing behind. Learning about a failure required someone at the camera PC to send the log file.

### Current state

- Review fixes: scrub every public report field, register camera serials before startup failures, and keep historical crash logs local. Analog camera identities no longer interrupt status polling. Opt-out stops queued delivery and subsequent GitHub writes, expired pending reports are dropped at startup, and held OS locks protect active logs and crash files from retention. Qt aborts retain native diagnostics when no report was saved; tests pin literal v1 fingerprints and restore exception hooks correctly.

- The log is built on the standard `logging` module: `HH:MM:SS LEVEL [thread] message` lines in the per-run file, the GUI log panel and the console, with a startup header (build and commit time, platform, library versions, front end, level). `--log-level {debug,info,warning}` or `ESPI_LOG_LEVEL` raises the detail (`debug` adds camera node writes, settings changes and frame timing), the newest 50 run logs are kept, `Help > Open log folder` opens the folder, warnings from Python and Qt are logged, `faulthandler` writes `espi_<timestamp>.crash` for native crashes (reported by the next start), and a UI stall longer than ~10 s dumps every thread's stack into the log.
- Unexpected errors — uncaught exceptions on any thread (including Qt slots), engine and recorder failures, Qt critical messages and the previous run's crash file — are reported automatically as GitHub issues labelled `auto-error`. Each error carries a deterministic ID (`ESPI-ERR-v1-…`) from its exception type and code path, so the same bug is reported once across machines, install paths and runs: the existing issue gets one "seen again" comment per build, a regression in a newer build reopens it, wontfix issues stay quiet. Every report is scrubbed of home folders, user and host names, camera serials, saved-file names, e-mail addresses and token-like strings. Release builds report by default; `--no-error-reports`, `ESPI_ERROR_REPORTS=0` or **Help > Send error reports automatically** turns it off, and copies without the token get a **Report in browser…** link instead. The "ESPI stopped" dialog names the issue, and non-fatal errors show a status-bar message with the link. Issues are added to the **ESPI Live roadmap** project by a workflow using the new `PROJECT_PAT` secret.

## 2026-10-30 — Status-bar actions look like buttons

### Previous state

- **Check for updates** and **Request feature** were flat, so they read as status text like the frame-rate readouts. While a check or a request was in progress they were only grey text. The blue **Install update** button was a flat rectangle with no hover or pressed state.

### Current state

- **Check for updates**, **Request feature** and **Open in Codex** are bordered buttons with padding, a background distinct from the status bar, and hover, pressed, focus and disabled states. The pointer changes to a hand over them.
- **Check for updates** has a refresh icon and **Request feature** a lightbulb. The blue **Install update** button (download icon) stays the primary action, more prominent than the others, including while it reads **Downloading update…**.
- A vertical line separates the buttons from the readouts. Busy labels (**Checking for updates…**, **Sending request…**, **Downloading update…**) stay disabled buttons and do not shift the readouts. The source badge keeps its full text (**DUMMY MODE**, or the camera name); the numeric readouts shorten first if the window is narrow.

## 2026-10-30 — Guidance for the processing controls

### Previous state

- Speckle filter, filter width, Average N, rolling average, reference refresh and noise reduction had tooltips that described the mechanism, or none (Gamma, Stretch to 0–255, Palette, CLAHE). Nothing said when to use a control or what to try first.

### Current state

- Each of those controls now says what it does, when to use it, and a starting value. **Help > Which settings should I use?** repeats that advice, including the processing order: noise reduction → difference → average → speckle filter → stretch → invert → gamma/CLAHE → palette.
- **Live preview** and **Measurement** in the ESPI measurement section apply those starting points. They do not change exposure (8000 µs), gain (0 dB), frame rate (30 fps) or recording.
- The remembered window layout resets once, so the taller controls panel is not clipped by an older saved size.

## 2026-10-30 — Gamma in every view, and how it differs from stretch

### Previous state

- **Gamma** sat in the Display section with no explanation. In **Raw** (the view the program opens in) the slider did nothing: those frames were always drawn at gamma 1.0. **Previous** and **Reference** did apply it.
- **Stretch to 0–255** is a straight remap of a measurement frame's range. With a good stretch, gamma's mid-tone shift is easy to mistake for the same control, or to miss entirely.

### Current state

- Gamma stays its own control in Display (not an advanced setting). Its tooltip, and the Stretch checkbox's, say what each one does: stretch sets contrast, gamma is the curve after that. Below 1.0 brightens dark fringes; above 1.0 darkens bright ones; 1.0 leaves mid-tones alone.
- Raw, Previous and Reference all use the gamma slider. Raw still skips stretch and the color palette, so positioning pixels stay gray. The default is still 1.0, so an unchanged setup looks the same. Saved frames and recordings follow the screen.

## 2026-10-30 — Noise reduction works across the slider, and says so

### Previous state

- **Noise reduction** read its 0–95 slider as the share of the previous frame to keep. Twenty was about one frame of memory, so most of the travel did almost nothing, and the speckle grain (which the control keeps on purpose) made a still image look unchanged. It was easy to take it for the Speckle filter or Average N, and **Stretch to 0–255** repaints a quieter difference at full brightness, which hides the change again.

### Current state

- The same slider, still 0 = off and 95 = strongest, now means how far the control is turned up. Each step adds the same amount of memory, so 10 is about three frames and 95 is still the old cap of about twenty. The default stays 0. The speckle grain is still kept.
- The hover tip, which also shows the current value, and a note under the slider say what to look for (less flicker, a trail when something moves) and what will not change (a still speckle image; stretch hides a quieter difference), and that this is not the Speckle filter or Average N.

## 2026-10-30 — Built-in user guide under Help

### Previous state

- The UI had no in-application user guide. Control meanings, measurement steps, shortcuts, camera handling and troubleshooting were spread across the README and tooltips.

### Current state

- New `src/espi_live/user_guide.html`, a self-contained offline page bundled with the package. It covers the toolbar, view modes, camera settings, noise reduction, averaging and rolling average, reference refresh, fringe enhancement, display, capture and saved frames, status bar, log, several cameras, the analog camera, defaults and limits, common operations, shortcuts, output files, troubleshooting, updates, feature requests and Codex.
- **Help → User Guide** (**F1**) opens the page in the default browser; `--check` fails when the guide is missing from a binary. The page has linked contents, a section filter, collapsible troubleshooting answers, light/dark themes, print/PDF support and in-page cross-references, with no network access or external assets.
- `guide.py` resolves the guide from source or a PyInstaller build (including a macOS `.app`'s `Contents/Resources`) and builds an error page if it cannot be read. The release workflow ships `user_guide.html` with `--add-data`, and camera-free tests check the content, anchors, print rules, frozen paths and error page.

## 2026-10-30 — Feature requests are filed by the app

### Previous state

- The request form had an **Enrich with GitHub Copilot** checkbox, and a blank description was refused.
- A copy without a feedback token, and a failed send, offered **Open in browser** instead of filing the issue.

### Current state

- There is no enrichment checkbox. Every request is labelled `enrich`. A blank description is filed from the title, and Copilot writes the spec; a description you write is kept and expanded.
- **Submit** files the issue from the app. When it is sent, the dialog says the request was submitted. A failed send can be edited and retried. The app does not open a browser page for it.
- A published build always contains the feedback token. The release build stops if that secret is missing, and `--check` says whether this copy has one. It never prints the token.

## 2026-10-30 — Status-bar help tips

### Previous state

- Hovering a control shows what it does, and the shortcuts are listed in the View menu, but nothing in the window points them out. A new user can miss both.

### Current state

- A **Did you know** tip on the left of the status bar rotates about every 12 seconds and starts on a random tip, so it is not the same sentence every launch. Click the tip for the next one. The camera, recording and dummy-mode badges keep their width; the tip uses only the room left after them, and a narrow window shortens it with an ellipsis (or hides it) rather than covering **DUMMY MODE**.
- Right-click the tip, or uncheck **View > Help tips**, to turn them off. The choice is remembered. The timer runs on the window's thread and does not touch the camera.
- The tips cover hovering for a description, `S` and `F`, stretch clip and gamma, rolling and block average, Request feature, the dockable panels, switching cameras, and zoom and pan.
- A first launch opens at 2100×900, wide enough for a whole tip beside the badges. A layout saved by an older version is kept.

## 2026-10-30 — Hover tooltips on every control

### Previous state

- Some controls explained themselves (noise reduction, averaging, the speckle filter, stretch clip, invert) and others did not: gamma, stretch to 0–255, palette, CLAHE, exposure, gain, frame rate, the view-mode buttons and several camera controls.
- Slider tips did not show the value under the pointer, and a shortcut was not always named on the control that uses it.
- The shortcut list was a tooltip on the live image, so it covered the measurement.

### Current state

- Hovering any slider, spin box, checkbox, combo box, button, text field, menu item or status readout shows what it does, and, where it helps, when to use it and a typical value. Sliders include the current value and unit (`Now: 8000 µs`). A control with a shortcut names it (`Shortcut: S`).
- Tips wait 600 ms, including when the pointer moves from one control to the next, so they do not flash. They open beside the control and never over the live image; the image itself has no tooltip. The shortcut list is in the View menu.
- Disabled controls still show their tip, so an analog camera's exposure, gain and frame rate still say they are set on the camera, and dummy mode still says those three are simulated.
- The old Tk window is not part of the application any more (`--ui tk` is gone), so this is the Qt window only.

## 2026-10-30 — Updates are noticed while the app stays open

### Previous state

- A prebuilt copy looked for a newer build once, at startup. A camera PC left running for days never noticed a build released after it opened.
- The status-bar button read **Update available:** followed by the version.

### Current state

- While the app is open it checks again about every three hours, a little earlier or later each time. It skips a check that would overlap one already running, a download, a recording or a frame save. `--no-update-check` turns the later checks off too. Running from source still does not update itself.
- When a newer build is found the button reads **Install update** (the versions are in the tooltip, such as "Build 42 → 45") and the status bar mentions it once. Finding that same build again leaves the button as it is. A check that fails only goes to the log, and an update already found stays on the button. Downloading and restarting work as before: the new build is installed only after the camera is released.
## 2026-10-29 — Analog camera support (Sony XC-ST50 via USB-Live2)

### Previous state

- Only FLIR/Spinnaker cameras could be used. The lab's second camera, an analog Sony XC-ST50 connected through a Hauppauge USB-Live2 capture stick, was not seen at all.
- A camera the viewer did not know was only mentioned in the log.

### Current state

- **XC-ST50 through a USB-Live2:**
  - The stick appears in the camera list as **Hauppauge USB-Live2 (video N)**, next to any Blackfly or on its own, and works with every view, averaging, saving and recording feature.
  - Startup opens a Blackfly first; with none connected it opens the USB-Live2. **Switch to live capture** in demo mode follows the same order.
- **Fixed controls:** exposure, gain and frame rate are set on the analog camera itself, so those controls are disabled for it. The status line shows the capture size and the measured frame rate. The viewer selects the composite input and asks for 720 × 480; a different delivered size (for example PAL 720 × 576) gets a warning.
- **Single field (interlaced camera)**, on by default, shows one field of each interlaced frame repeated to full height, so the 1/60 s between fields cannot blur the fringes; it halves the vertical detail. Changing it clears the reference and the averages.
- **Other video devices** (webcams, other capture sticks) are listed as **not yet supported** and are never opened unless chosen. Compressed video gets an "unsuitable for ESPI" warning.
- **Camera reports:** switching on an unrecognised video device files a GitHub issue labelled `camera-support` with what the device reports, once per device type across all PCs. The status appears under the camera information. Copies without a feedback token save the report to `output/logs/camera-report-*.md` and offer **Report this camera** instead.
- **Packaging:** builds include `cv2_enumerate_cameras` to list video devices by name and USB ID, and `--check` reports whether it is present.

## 2026-10-29 — Open in Codex

### Previous state

- Someone curious about how ESPI live works had to find the repository and read Python, and could suggest a change only in words, through **Request feature**.

### Current state

- Prebuilt copies download the source code of their own build into `espi-live-source/` next to the program, checked at every start (skipped with `--no-update-check`). A copy holding unsent changes is kept across updates.
- An **Open in Codex** button in the status bar (and **Help > Ask Codex about the source code…**) opens OpenAI's Codex app on that folder, with instructions that make Codex answer from the real code in plain words. A first-use dialog explains it and links to the Codex download.
- When the user asks Codex for a change and approves it, Codex runs the new `--send-changes` option, which files the change as a `codex-change` GitHub issue; the new `codex-change.yml` workflow applies it to a `codex/issue-<n>` branch and opens a pull request for the developers to review. The program still holds only an Issues token.

## 2026-10-29 — Freeze the displayed image without saving it

### Previous state

- The only way to hold an image for inspection was **Save frame** (`S`), which wrote a PNG (and, with a title or notes, a `.txt` file) to `output/frames/`.

### Current state

- **Freeze** (`Space`, a toolbar button, or **View > Freeze display**) holds the image on screen so it can be zoomed, panned and read with the pixel readout, without writing anything. Only the display stops: the camera keeps acquiring, and averaging, reference refresh and recording carry on. Press `Space` again to return to the live image.
- A frozen image is marked by an amber **FROZEN at** *time* bar over the image, a **FROZEN** status-bar badge and the highlighted toolbar button; the displayed rate reads "frozen". **Save frame** saves the held image, and its notes file records when it was frozen. **Set reference** still uses the live camera frame.
- With several cameras, each camera keeps its own frozen frame (the camera list shows `frozen`); turning a camera's processing off, or **Switch to live capture** from dummy mode, drops it. Defaults (exposure 8000 µs, gain 0 dB, 30 fps) and recording are unchanged.

## 2026-10-29 — Fixed-position status bar

### Previous state

- Each status-bar readout was as wide as its text, so a longer value (a high frame rate, a long range) or a button changing its label shifted every item to its left.

### Current state

- Every status-bar item has a fixed width sized for its widest expected text, so no value moves another item. Text that still does not fit is shortened with an ellipsis. **Check for updates** and **Update available** share one fixed slot, so swapping them moves nothing.

## 2026-10-29 — Check for updates button

### Previous state

- Checking for a newer build on demand meant finding **Help > Check for updates…**; the status bar showed nothing about updates unless the startup check had already found a newer build.

### Current state

- A **Check for updates** button in the status bar, next to **Request feature**, checks GitHub on demand like the Help menu item and reports the result in a dialog. It reads **Checking for updates…** and is greyed out while a check runs, including the automatic one at startup, and gives way to the blue **Update available** button once a newer build is found.

## 2026-10-29 — Builds ship as a folder; every merge into main is released

### Previous state

- The Windows and Linux builds were one self-extracting file (`espi-live.exe`, `espi-live`) that unpacked its libraries (about 390 MB) to a temporary folder on every start.
- An update downloaded the new file and swapped it for the running one.
- Merging into `main` released nothing: a release came from publishing the draft release or from **Actions > Release > Run workflow**, and every run of that workflow published a release, so it could not test a branch's builds without releasing them.

### Current state

- The Windows and Linux archives hold an `espi-live` folder: the executable plus `_internal-N`, the libraries of build N. The program starts without unpacking anything. Keep the folder together; `output/` is still written next to the executable, now inside that folder. The macOS app was already a folder (`.app`) and is unchanged.
- An update moves the new build's `_internal-N` folder in beside the running one and swaps only the executable, because Windows cannot overwrite or delete a running program's libraries. When the new build starts, it deletes the old executable (`espi-live.exe.old`) and, once that is gone, the old `_internal-N` folder by itself, retrying in the background for about a minute while the old build is still closing.
- Copies that are a single file (build 6 and earlier) cannot update themselves to the folder layout: their update check reports `build N has no espi-live-windows-x64.zip`. Download a current release once by hand and unpack it where the old copy is: the new build then deletes the old single-file copy when it starts (it looks in the folder above its own and the one above that, and deletes only files it recognizes as single-file builds of ESPI live), keeping that copy's `output/` folder. `build-info.json` now lists the checksums under `archives`; the `sha256` key those copies read is left out, so they never put the new folder where their executable was.
- Everything the startup cleanup removes is written to the log.
- Every merge into `main` builds and publishes a release, which installed copies are offered at once. Its version comes from the merged pull requests' labels (the release workflow rewrites the draft release first, so it counts the merge just made) and its notes from GitHub's generated release notes. A merge that lands while a release is building is released after it, together with any others that waited. `release-drafter.yml` no longer runs on merges; the release workflow updates the draft itself.
- **Actions > Release > Run workflow** has a **Publish** checkbox: untick it to test and build a branch without releasing anything; the archives stay on the run's page as artifacts.

## 2026-10-29 — Releases started by hand get version tags

### Previous state

- A release started with **Actions > Release > Run workflow** was named **Build N** and tagged `build-N` after the workflow run number, while releases published from the draft were tagged `vX.Y.Z`.

### Current state

- Every release is tagged with a version. A hand-started run uses the new **Version** field (`vX.Y.Z`), or else the version the draft release proposes, or else the next patch version after the newest release, and skips any version whose tag already exists. The release is named after its tag.
- Pre-releases (runs from a branch other than `main`, or with **Pre-release** ticked) are tagged `vX.Y.Z-pre.N`, so test builds don't use up the version number.
- After a full release started by hand, the draft release is refreshed so it proposes the following version.
- Installed copies are unaffected: they still order builds by the build number in `build-info.json`, not by tag, so they update from `build-N` releases to the new ones as usual.

## 2026-10-29 — Linux build installs the Spinnaker SDK again

### Previous state

- The Linux build failed while installing the Spinnaker 4.4 packages (FLIR's installer stopped at its license and analytics questions, which CI cannot answer), so no release could be published: the workflow only publishes when all three builds succeed.

### Current state

- The workflow records the license acceptance up front and installs without naming a user, so the installer skips the analytics question and sends no install analytics.
- It also installs the two runtime pieces PySpin needs that the FLIR packages leave out: `spinupdate` (`libSpinUpdate`) and FFmpeg (`libavcodec60`, `libavformat60`, `libavutil58`, `libswscale7`, which `libSpinVideo` links against without declaring them). Before that the Linux binary failed its PySpin import check, so Windows, Linux and macOS now all build and the release is published with all three archives.

## 2026-10-29 — Release workflow renamed, generated notes, pip cache

### Previous state

- The workflow that tests, builds and attaches the binaries to a release was listed in GitHub Actions as **Build release**.
- Hand-started `build-N` releases had only the notes typed in when starting the run.
- Every test and build run downloaded all Python dependencies from scratch.

### Current state

- It is listed as **Release**. Publishing the drafted release still starts it, and **Actions > Release > Run workflow** still makes a `build-N` test build. The file is still `build-binaries.yml`, so the run number that installed copies use to order builds keeps counting.
- Hand-started `build-N` releases now carry GitHub's generated release notes (pull requests merged since the previous release), below any notes typed in. The new `.github/release.yml` groups them by the same labels as the drafter (Features, Fixes, Documentation, Dependencies; `skip-changelog` leaves a pull request out), and the **Generate release notes** button on the Releases page uses it too.
- The test and build jobs cache pip downloads, so a run no longer downloads PySide6, OpenCV and the other dependencies again each time.

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
