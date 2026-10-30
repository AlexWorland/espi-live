# Analog Capture (XC-ST50 via USB-Live2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the Sony XC-ST50 through a Hauppauge USB-Live2 as a camera next to the Blackfly. Single-field mode is the default. An unrecognised video device files a GitHub camera report when it is switched on.

**Architecture:**
- **Sources:** a pure adapter table (`capture_models.py`) and an OpenCV capture source (`capture.py`) with the `CameraSession` interface sit behind a small dispatcher (`sources.py`) that the engine and channels use.
- **Interlace:** single-field handling is a pure processing step selected by a per-camera setting.
- **Reports:** `camera_reports.py` formats reports and files them from a background worker. The Qt window wires events, controls and a status line.

**Tech Stack:** Python 3.10 (camera PC) / 3.14 (`.venv`), OpenCV 5.0 (`opencv-python`), `cv2_enumerate_cameras` 1.4.0, PySide6, `unittest` + `coverage`, GitHub REST API via `urllib`.

**Spec:** `docs/notes/analog-capture-xc-st50-design.md`. Read it first.

## Global Constraints

- **Where to work:** `~/git-repos/espi-live-worktrees/analog-capture-xc-st50` (branch `analog-capture-xc-st50`, public history), never `~/espi_live` or its T3 worktrees.
- **Interpreter:** `PY=~/espi_live/.venv/bin/python`.
- **Tests:** `PYTHONPATH=src QT_QPA_PLATFORM=offscreen $PY -m unittest …`.
- **After every task:**
  - `$PY -m py_compile src/espi_live/*.py tests/*.py` passes.
  - The full suite under coverage reports **100 %** (`$PY -m coverage run -m unittest discover -s tests && $PY -m coverage report`; the baseline is 457 tests). No `# pragma: no cover`.
- **Docstrings:** every new function or method has one in the `Args: … / Returns: …` format.
- **No GUI imports** in `capture_models.py`, `capture.py`, `sources.py` or `camera_reports.py`. Only `camera_reports.py` (and `feedback.py`) do network I/O, and never on the engine thread or the startup path.
- **Blackfly behaviour unchanged:** defaults stay 8000 µs, 0 dB, 30 fps.
- **USB-Live2:** USB ID `0x2040:0xC200`, 720 × 480, interlaced, `CAP_PROP_CHANNEL` 0 (composite).
- **Backends:** `CAP_DSHOW` on Windows, `CAP_V4L2` on Linux, `CAP_AVFOUNDATION` on macOS.
- **Reports:** label `camera-support`; title `Camera report: <key>`; hidden marker `<!-- espi-live camera report: <key> -->`; body ≤ 60 000 characters; prefilled URL ≤ 8 000 characters.
- **Tests never** open a real camera or video device (fake `VideoCapture` and enumerator), touch the network (patch `urllib.request.urlopen`), write to the real `output/` (patch `paths.OUTPUT_DIR`), or block on a dialog.
- **Commits:** AGENTS.md says don't commit unless asked. Each task's commit step runs only once the owner has approved commits.

## Review Focus

1. **USB-Live2 left in PAL by WinTV** (720 × 576 frames) → a warning names PAL, and single-field mode still works on 576 rows and on odd heights. Tests in Tasks 1 and 4.
2. **Device unplugged mid-run** (`read()` keeps returning False) → the engine keeps running, shows "Waiting for camera frame" and logs at most every 5 s; nothing crashes. Test in Task 3.
3. **WinTV or Hauppauge Capture holds the device** (not opened, or the first read fails) → opening fails with a message that names the device and says to close WinTV / Hauppauge Capture, and the channel stays off. Test in Task 2.
4. **Two USB-Live2 sticks, or two devices with the same name** → two channels with distinct ids, and still only one report per key. Tests in Tasks 2 and 6.
5. **A device with no USB ID and an unknown name** (for example "OBS Virtual Camera") → listed as unrecognised and never auto-opened; its report key falls back to the name. Tests in Tasks 1 and 3.

---

### Task 1: Adapter table and capture identity

**Files:**
- Create: `src/espi_live/capture_models.py`
- Test: `tests/test_capture_models.py`

**Interfaces:**
- Produces:
  - `CaptureAdapter(name: str, interlaced: bool, width: int, height: int, channel: int = 0)`, frozen.
  - `ADAPTERS: dict[tuple[int, int], CaptureAdapter]`, holding `(0x2040, 0xC200)` → `CaptureAdapter("Hauppauge USB-Live2", True, 720, 480, 0)`.
  - `adapter_for(vid: int | None, pid: int | None, name: str) -> CaptureAdapter | None`. The USB ID is tried first; otherwise the lower-cased name is matched against `"usb-live2"`, `"usb live2"` and `"usb live 2"`.
  - `COMPRESSED_FOURCCS = frozenset({"MJPG", "H264", "H265", "HEVC", "AVC1", "X264", "MP4V"})`.
  - `fourcc_text(code) -> str`: a 4-character code, or `""` for 0 and invalid input.
  - `CaptureIdentity(name="", backend="", path="", vid=None, pid=None, width=0, height=0, fourcc="", fps=0.0, adapter=None)`, frozen. It has `supported`, `interlaced`, `short_name()`, `description() -> list[str]` and `warnings() -> list[str]`, the same surface the window already uses on `CameraIdentity`.

- [ ] **Step 1: Write the failing tests:**
  - `test_usb_live2_by_usb_id_and_by_name`: `adapter_for(0x2040, 0xC200, "")` and `adapter_for(None, None, "Hauppauge USB-Live2 Video")` both return the USB-Live2 entry.
  - `test_unknown_device_has_no_adapter` (Review Focus 5): `adapter_for(None, None, "OBS Virtual Camera") is None`.
  - `test_fourcc_text`: `fourcc_text(cv2.VideoWriter_fourcc(*"YUY2")) == "YUY2"`; `fourcc_text(0) == ""`.
  - `test_recognised_identity`: a USB-Live2 identity at 720 × 480 YUY2 is supported and interlaced. `short_name() == "Hauppauge USB-Live2"`. `description()` mentions `"720 × 480"` and `"interlaced"`. `warnings()` holds only the analog note, which mentions `"set on the camera"` and `"single-field"`.
  - `test_pal_warning` (Review Focus 1): the same identity at 720 × 576 has a warning containing `"720 × 576"` and `"PAL"`.
  - `test_rescaled_warning`: at 640 × 480, a warning contains `"rescaled"`.
  - `test_unsupported_and_compressed_warnings`: `CaptureIdentity(name="Webcam", fourcc="MJPG")` is not supported; its warnings include `"Webcam is not yet supported"` and one containing `"MJPG"` and `"not suitable for ESPI"`; `interlaced` is False.
- [ ] **Step 2: Run them and confirm they fail.** Run: `PYTHONPATH=src $PY -m unittest tests.test_capture_models -v`. Expected: ImportError.
- [ ] **Step 3: Implement** the interfaces above. Warnings come in this order: unsupported, compressed, size mismatch, analog note (the analog note is for interlaced devices only).
- [ ] **Step 4: Run them and confirm they pass,** plus the full suite under coverage.
- [ ] **Step 5: Commit** (once approved): `git add src/espi_live/capture_models.py tests/test_capture_models.py && git commit -m "Describe analog capture adapters, starting with the Hauppauge USB-Live2"`.

### Task 2: OpenCV capture source

**Files:**
- Create: `src/espi_live/capture.py`, `tests/fake_capture.py`
- Modify: `src/espi_live/camera.py` (`CameraInfo` fields; `CameraSession.interlaced = False` and `describe(values)`)
- Test: `tests/test_capture.py`, `tests/test_camera.py`

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - `CameraInfo(serial, model="", kind="spinnaker", index=None, backend=None, vid=None, pid=None)`. For a capture device, `serial` holds the device path, or `f"video {index}"` when there is no path.
  - `CameraSession.describe(values) -> str` returns `format_camera_values(values)`.
  - In `capture`:
    - `capture_backend() -> int`, chosen by `sys.platform`.
    - `load_enumerator()`: the `cv2_enumerate_cameras` module, or `None` on ImportError.
    - `list_capture_devices(log) -> list[CameraInfo]`: `enumerate_cameras(capture_backend())`, with `kind="capture"`. A missing enumerator or an exception gives `[]` and one log line.
    - `to_mono8(frame) -> np.ndarray`: 2-D uint8 is returned unchanged; 3-channel goes through `COLOR_BGR2GRAY`; 2-D uint16 is shifted `>> 8`; anything else raises `ValueError`.
  - `CaptureSession(log, info)`:
    - `open()`:
      1. `cv2.VideoCapture(info.index, info.backend)`.
      2. For a recognised adapter, set `CAP_PROP_CHANNEL`, `CAP_PROP_FRAME_WIDTH` and `CAP_PROP_FRAME_HEIGHT` from the adapter.
      3. Read one frame. The identity takes its size from that frame's shape and fills `fourcc` and `fps` from `cap.get`.
      4. If the device didn't open, or the first read fails: release it and raise `RuntimeError(f"Could not open {info.label}: … close WinTV or Hauppauge Capture if they are running")`.
    - `grab(timeout_ms) -> (frame, "ok") | (exception, "error")`. `read()` blocks for about one frame, so `timeout_ms` is ignored and the docstring says so.
    - `set_float(name, value)` does nothing; `limits() -> {}`.
    - `read_values()` returns `{"exposure": None, "gain": None, "requested": None, "resulting": <fps measured since the previous call, or None>}`.
    - `describe(values) -> str`, for example `"Analog capture 720 × 480: 29.9 fps measured; exposure, gain and frame rate are set on the camera"`.
    - `read_details() -> tuple[tuple[str, str], ...]`:
      - device name, path, USB ID, backend name (`cv2.videoio_registry.getBackendName`) and `cv2.__version__`;
      - the negotiated width, height, fps, FOURCC and channel;
      - `cap.get` for BRIGHTNESS, CONTRAST, SATURATION, HUE, GAIN, EXPOSURE, CONVERT_RGB, FORMAT, MODE and BUFFERSIZE;
      - the first frame's shape, dtype, min, mean and max.
    - `close()` is idempotent and logs release errors.
    - Attributes `info`, `identity` and `interlaced`.
- `tests/fake_capture.py`:
  - `FakeVideoCapture(frames, opened=True, props=None)` with `read`, `get`, `set` (recorded), `isOpened` and `release`. `frames` is a list, or a callable that returns `(ok, frame)`.
  - `fake_devices(*items)`: an enumerator namespace whose `enumerate_cameras(api)` returns `SimpleNamespace(index, name, path, vid, pid, backend)` items.

- [ ] **Step 1: Write the failing tests** (`capture.cv2.VideoCapture` and `capture.load_enumerator` patched):
  - `test_list_capture_devices`: two devices with the same name and different paths (Review Focus 4) give two `CameraInfo` with `kind == "capture"`, distinct `serial` values, and `index`, `backend`, `vid` and `pid` carried over. A device with no path gets `serial == "video 3"` for index 3.
  - `test_list_without_enumerator`: `load_enumerator` returns None → `[]` and one log line containing `"cv2_enumerate_cameras"`.
  - `test_open_usb_live2_requests_native_size`: the recorded `set` calls include `(CAP_PROP_CHANNEL, 0)`, `(CAP_PROP_FRAME_WIDTH, 720)` and `(CAP_PROP_FRAME_HEIGHT, 480)`; `session.interlaced` is True.
  - `test_open_unknown_device_sets_nothing`: no `set` calls; `interlaced` is False.
  - `test_open_failure_names_device_and_wintv` (Review Focus 3): `opened=False`, and separately a failing first read → `RuntimeError` whose message contains the device label and `"WinTV"`; `release` was called.
  - `test_grab_converts_bgr_to_mono8`: a 3-channel frame comes back 2-D uint8 with `"ok"`. A failed read gives `"error"`.
  - `test_to_mono8_variants`: uint16 is shifted; 2-D uint8 is unchanged; a 4-D input raises `ValueError`.
  - `test_read_values_measures_rate`: with patched `time.monotonic`, 30 grabs in 1.0 s give `resulting ≈ 30.0`; the first call with no frames gives `None`.
  - `test_read_details_lists_device_facts`: contains `("Backend", …)`, `("FOURCC", "YUY2")` and the first-frame mean.
  - `test_camera_session_describe` (in `test_camera.py`): equals `format_camera_values`.
- [ ] **Step 2: Run them and confirm they fail.** Run: `PYTHONPATH=src $PY -m unittest tests.test_capture tests.test_camera -v`.
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run them and confirm they pass,** plus the full suite under coverage.
- [ ] **Step 5: Commit** (once approved): `git add src/espi_live/capture.py tests/fake_capture.py tests/test_capture.py && git commit -am "Read analog video through OpenCV as a camera source"`.

### Task 3: Choosing the source in engine, channels and live switch

**Files:**
- Create: `src/espi_live/sources.py`
- Modify: `src/espi_live/engine.py` (`Engine.__init__` gains `info=None`, `open_camera`, `_attach_camera`, the loop's readback, `_switch_live`), `src/espi_live/channels.py` (`_new_engine`, `Channel.supported`, `open_camera_channels`)
- Test: `tests/test_sources.py`, `tests/test_engine_camera.py`, `tests/test_channels.py`

**Interfaces:**
- Consumes: Task 2.
- Produces:
  - `sources.detect_cameras(log) -> list[CameraInfo]`: Spinnaker cameras first, then capture devices. An ImportError or any error from `camera.list_cameras` is logged and empties only that list.
  - `sources.make_session(log, info)`: a `CaptureSession` for `kind == "capture"`, else `CameraSession(log, info.serial)`.
  - `sources.is_supported(info) -> bool`: True for Spinnaker; for capture, `adapter_for(info.vid, info.pid, info.model) is not None`.
  - `sources.auto_open_order(infos) -> list[CameraInfo]`: Spinnaker in order, then supported capture devices. Unsupported devices are left out.
  - `Engine(log, serial=None, tag=None, info=None)`: `open_camera()` uses `make_session` when `info` is given. Status lines use `session.describe(values)`.
  - `_switch_live`: tries `CameraSession(log)`, then each supported capture device, logging each failed attempt. When all fail it sends `switch_failed` with the last error — the plan's original "first error" would hide the capture adapter's actionable message (close WinTV / Hauppauge Capture) behind the generic FLIR one. The `switched_live` payload is `detect_cameras(log)`.
  - `Channel.supported -> bool` (True for demo channels); `_new_engine` passes `info`.
  - `open_camera_channels(log)`: a channel for every detected camera. Candidates are tried in `auto_open_order`. With no candidates it raises `RuntimeError("No FLIR camera or recognised capture adapter detected.")`.

- [ ] **Step 1: Write the failing tests:**
  - `test_detect_merges_and_survives_missing_pyspin`: `list_cameras` raises ImportError → only capture devices come back, and the log mentions PySpin.
  - `test_auto_open_order_skips_unknown` (Review Focus 5): [webcam, USB-Live2, Blackfly] → [Blackfly, USB-Live2].
  - `test_startup_opens_usb_live2_without_blackfly`: no Spinnaker cameras, a webcam and a USB-Live2 (fakes) → the USB-Live2 channel is active, and the webcam channel exists but is inactive.
  - `test_startup_with_only_webcam_raises`: raises the RuntimeError above, and the webcam was never opened (`FakeVideoCapture` was not constructed).
  - `test_engine_streams_from_capture`: an engine with a USB-Live2 `info` publishes packets, and its status camera text starts with `"Analog capture"`.
  - `test_unplugged_device_keeps_engine_running` (Review Focus 2): the fake returns `(False, None)` forever after 3 frames → after 1.5 s the engine is still alive, status `processing == "Waiting for camera frame"`, and at most one "acquisition error" log line per 5 s.
  - `test_switch_live_falls_back_to_usb_live2`: a demo engine, no Spinnaker cameras, a USB-Live2 present → `switched_live` arrives and the engine is no longer in demo mode.
  - `test_channel_supported`: true for a Blackfly, a USB-Live2 and a demo channel; false for a webcam.
- [ ] **Step 2: Run them and confirm they fail.** Run: `PYTHONPATH=src $PY -m unittest tests.test_sources tests.test_engine_camera tests.test_channels -v`.
- [ ] **Step 3: Implement.** Update the existing tests that assert the old "No FLIR/Spinnaker cameras detected." text or `Engine(…, serial=…)` construction.
- [ ] **Step 4: Run them and confirm they pass,** plus the full suite under coverage and `PYTHONPATH=src $PY -m espi_live --check`.
- [ ] **Step 5: Commit** (once approved): `git add src/espi_live/sources.py tests/test_sources.py && git commit -am "Open Blackfly cameras and recognised capture adapters through one source chooser"`.

### Task 4: Single-field mode

**Files:**
- Modify: `src/espi_live/processing.py` (`single_field`), `src/espi_live/engine.py` (`Settings.single_field`, the loop)
- Test: `tests/test_processing.py`, `tests/test_engine_camera.py`

**Interfaces:**
- Produces:
  - `processing.single_field(frame) -> np.ndarray`: the even rows, each repeated twice and cropped to the input height. Returns a new array.
  - `Settings.single_field: bool = True`, applied only when `self.camera is not None and self.camera.interlaced`.
  - Changing `single_field` runs `processor.reset_avg()` and sets `processor.prev = None` and `processor.ref = None`. It logs `"Fields: single field (even rows, line-doubled)"` or `"Fields: full interlaced frames"`.

- [ ] **Step 1: Write the failing tests:**
  - `test_single_field_doubles_even_rows`: rows `[0,1,2,3]` → rows `[0,0,2,2]`; shape is kept for heights 480, 576 and 481 (Review Focus 1); the input is not modified.
  - `test_engine_applies_single_field_to_interlaced_source`: a fake capture with even rows 0 and odd rows 255 → the Raw packet is all 0 by default; after `update_settings(single_field=False)` it alternates.
  - `test_single_field_ignored_for_blackfly`: a Spinnaker fake frame passes through unchanged with `single_field=True`.
  - `test_toggle_resets_reference`: in Reference mode with a reference set, toggling `single_field` → `processor.ref is None`, observed as the "press F" hint in `status()["hint"]`.
- [ ] **Step 2: Run them and confirm they fail.** Run: `PYTHONPATH=src $PY -m unittest tests.test_processing tests.test_engine_camera -v`.
- [ ] **Step 3: Implement** the interfaces above. The step runs right after the Mono8 check, before `denoise`.
- [ ] **Step 4: Run them and confirm they pass,** plus the full suite under coverage.
- [ ] **Step 5: Commit** (once approved): `git commit -am "Show interlaced analog cameras one field at a time by default"`.

### Task 5: Camera report text and GitHub calls

**Files:**
- Create: `src/espi_live/camera_reports.py`
- Modify: `src/espi_live/feedback.py` (`github_request`, `environment_lines`), `src/espi_live/engine.py` (`_attach_camera` sends `camera_report`)
- Test: `tests/test_camera_reports.py`, `tests/test_feedback.py`, `tests/test_engine_camera.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `feedback.environment_lines() -> list[str]`: the Build and Platform bullets that `context_lines` now starts with.
  - `feedback.github_request(method: str, path: str, token: str, payload: dict | None = None) -> object`: parsed JSON. With an empty token, no Authorization header is sent. Errors are raised as `FeedbackError` via `explain_http_error`. `submit` uses it.
  - In `camera_reports`:
    - Constants: `LABEL`, `MAX_BODY = 60000`, `MAX_URL = 8000`.
    - `Report(key: str, title: str, body: str, summary: str)`, frozen.
    - `marker(key) -> str`.
    - `cell(text) -> str`: escapes `|` and backticks; newlines become spaces.
    - `fit(lines: list[str], limit: int) -> str`: drops table rows from the end and adds `| … | N more rows not shown |`.
    - `capture_report(info, identity, details) -> Report`. The key is `f"{name} ({vid:04x}:{pid:04x})"`, else the name, else `info.serial`. The body holds an intro, the `identity.description()` lines, a details table, `environment_lines()` and the marker. `summary` is the description plus the USB ID and backend.
    - `find_report(key, token) -> dict | None`: `GET /repos/{repo}/issues?labels=camera-support&state=all&per_page=100`. Skips items with `"pull_request"`. Matches the marker, or a title exactly equal to `"Camera report: " + key`.
    - `file_report(report, token) -> dict`: POSTs with `labels: ["camera-support"]`.
    - `browser_url(report) -> str`: title plus summary, at most `MAX_URL` characters.
    - `save_report(report) -> str`: writes `output/logs/camera-report-<slug>.md`, overwriting, and returns the path.
  - Engine event `("camera_report", Report)`: sent by `_attach_camera` only for `kind == "capture"` sessions whose identity is not supported. If `read_details()` fails, the error is logged and an empty details tuple is used.

- [ ] **Step 1: Write the failing tests** (`urlopen` patched, `OUTPUT_DIR` set to a temp dir):
  - `test_report_key_and_title`: with a USB ID → `"Webcam (046d:0825)"`; without one → `"OBS Virtual Camera"` (Review Focus 5); with no name → the serial.
  - `test_markdown_breaking_values_escaped`: a detail value `"a|b\n```c"` gives one intact row containing `a\|b`.
  - `test_oversized_details_trimmed`: 5 000 rows of 100 characters → `len(body) <= MAX_BODY`, and the body contains `"more rows not shown"`.
  - `test_find_report_ignores_prs_and_lookalikes`: a PR with the exact title, an issue `"Camera report: Webcam (046d:0825) 2"` and the real issue → the real one; an empty listing → `None`.
  - `test_anonymous_request_has_no_auth_header` and `test_file_report_posts_label`.
  - `test_browser_url_short`: `len <= MAX_URL` and it contains `labels=camera-support`.
  - `test_save_report_overwrites`.
  - `test_engine_reports_only_unrecognised_capture`: a webcam session → one `camera_report` event whose key starts with the webcam name. The USB-Live2, a Blackfly fake and demo mode → none.
  - `test_feedback` stays green after the `github_request` / `environment_lines` refactor.
- [ ] **Step 2: Run them and confirm they fail.** Run: `PYTHONPATH=src $PY -m unittest tests.test_camera_reports tests.test_feedback tests.test_engine_camera -v`.
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run them and confirm they pass,** plus the full suite under coverage.
- [ ] **Step 5: Commit** (once approved): `git add src/espi_live/camera_reports.py tests/test_camera_reports.py && git commit -am "Format camera reports for unrecognised video devices and file them on GitHub"`.

### Task 6: Background report worker

**Files:**
- Modify: `src/espi_live/camera_reports.py`
- Test: `tests/test_camera_reports.py`

**Interfaces:**
- Consumes: Task 5; `feedback.feedback_token()`.
- Produces:
  - `ReportState(url="", created=False, path="", error="")`, frozen.
  - `CameraReportWorker()` with `.events: queue.Queue`.
    - `.submit(report) -> bool` returns False when the key was already submitted this run; that set is touched only by the calling thread.
    - Jobs run in order on one daemon thread named `espi-camera-report`, started on first use.
    - Each job puts `("state", (key, ReportState))`.
  - Job rules:
    - Found → `url` is set.
    - Not found, with a token → `file_report`, then `url` and `created=True`.
    - Not found, without a token → `path = save_report(report)`.
    - `FeedbackError` → `error=str(exc)`, and without a token the report is still saved to `path`.
  - `status_html(state) -> str` (HTML-escaped). In priority order:
    1. `created` → `Reported to GitHub as <a href="URL">URL</a>.`
    2. `url` → `Already reported as <a href="URL">URL</a>.`
    3. `error` → `Report not sent: {error}.`
    4. `path` → `This copy cannot file reports on GitHub; the report is saved to {path}.`
    5. otherwise → `Reporting this camera to GitHub…`
  - `needs_button(state) -> bool`: `not state.url and bool(state.error or state.path)`.

- [ ] **Step 1: Write the failing tests** (GitHub functions and `feedback_token` patched; wait on `events` with a timeout):
  - `test_same_key_twice_files_once` (Review Focus 4): the second `submit` returns False; `file_report` is called once; one event.
  - `test_existing_issue_not_refiled`.
  - `test_no_token_saves_and_never_posts`: no POST; the state has `path`; the file exists in the temp `OUTPUT_DIR`.
  - `test_error_state`: `find_report` raises `FeedbackError("cannot reach GitHub: offline")` → `state.error` equals that text.
  - `test_status_html_and_button`: one assertion per priority case, plus escaping (`"<b>"` → `&lt;b&gt;`).
- [ ] **Step 2: Run them and confirm they fail.** Run: `PYTHONPATH=src $PY -m unittest tests.test_camera_reports -v`.
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run them and confirm they pass,** plus the full suite under coverage.
- [ ] **Step 5: Commit** (once approved): `git commit -am "File each camera report once, in the background"`.

### Task 7: Window: capture controls, list marker, report status

**Files:**
- Modify: `src/espi_live/qt_app.py`; `src/espi_live/channels.py` (`Channel.report = None`)
- Test: `tests/test_qt_app.py`

**Interfaces:**
- Consumes:
  - `Channel.supported` and the `info.kind` field (Task 3);
  - `Settings.single_field` (Task 4);
  - the `camera_report` event and `Report` (Task 5);
  - the worker, `ReportState`, `status_html` and `needs_button` (Task 6).
- Produces:
  - Attributes: `MainWindow.reports`, `MainWindow.report_states: dict[str, ReportState]`, `self.single_field` (QCheckBox `"Single field (interlaced camera)"`), `self.report_label` (rich text, opens external links) and `self.report_button` (`"Report this camera"`).
  - New methods: `_sync_source_controls()`, `_handle_report_events()`, `_show_report()`, `_open_report_in_browser()`.
- Behavior:
  - **Capture channel displayed:**
    - The exposure, gain and FPS controls are disabled, with the tooltip `"Set on the camera itself; not adjustable over analog video"`.
    - `single_field` is visible when `getattr(channel.identity, "interlaced", False)` is true; it shows `s.single_field` (from `_load_settings`), and toggling it calls `update_settings(single_field=…)`.
    - Demo and Blackfly channels keep today's control behavior, and `single_field` stays hidden.
  - **Camera list:** combo items for unsupported channels read `f"{label} · not yet supported"`.
  - **Report events:**
    - `camera_report` → `channel.report = payload` and `reports.submit(payload)`.
    - Each state event is stored, logged (URL, path or error), and refreshes `_show_report()`.
  - **Status line and button:**
    - `report_label` is visible when the displayed channel has a report and is not in demo mode; it shows `status_html(...)`.
    - `report_button` is visible when `needs_button(...)` is true. It opens `browser_url(report)` via `QDesktopServices.openUrl`; if that fails, `QMessageBox.warning(self, "Report this camera", "Could not open a web browser.")`.
  - **When it all runs:** `_sync_source_controls()` and `_show_report()` run from `_set_demo` and whenever the displayed channel changes; `_handle_report_events()` runs in `_poll_status`.

- [ ] **Step 1: Write the failing tests** (the worker replaced by a recording fake; `OUTPUT_DIR` patched):
  - `test_capture_channel_disables_camera_controls`: with a USB-Live2 channel displayed, `exposure`, `gain` and `fps` are disabled with the tooltip, and `single_field` is visible and checked. With a Blackfly channel displayed, they are enabled and `single_field` is hidden.
  - `test_single_field_checkbox_updates_engine`.
  - `test_unsupported_marker_in_camera_list`: only the webcam's combo text ends with `"· not yet supported"`.
  - `test_camera_report_event_submits_and_shows_state`: the event → `submit` is called. A state with a URL → `report_label` contains the URL, the button is hidden, and the log has the URL. A state with a `path` → the button is visible.
  - `test_report_button_opens_browser`: `openUrl` True → called with `labels=camera-support`; False → `QMessageBox.warning` (patched) is called.
  - `test_demo_hides_report_and_capture_controls`.
- [ ] **Step 2: Run them and confirm they fail.** Run: `PYTHONPATH=src QT_QPA_PLATFORM=offscreen $PY -m unittest tests.test_qt_app -v`.
- [ ] **Step 3: Implement** the behavior above.
- [ ] **Step 4: Run them and confirm they pass,** plus the full suite under coverage and `PYTHONPATH=src $PY -m espi_live --check`.
- [ ] **Step 5: Commit** (once approved): `git commit -am "Show analog cameras' fixed controls, field toggle and camera-report status"`.

### Task 8: Dependency, packaging, check, docs, PR

**Files:**
- Modify: `requirements.txt`, `.github/workflows/build-binaries.yml`, `scripts/windows/setup_espi_live.ps1`, `src/espi_live/cli.py` (`self_check`), `tests/test_cli.py`, `docs/camera/supported-models.md`, `docs/camera/README.md`, `README.md`, `AGENTS.md`, `CHANGELOG.md`
- Create: `docs/camera/sony-xc-st50-usb-live2.md`

- [ ] **Step 1: Packaging.**
  - `requirements.txt`: `cv2_enumerate_cameras==1.4.0; platform_system != "Darwin"`.
  - Windows `pip_deps` in the build job: append `cv2_enumerate_cameras==1.4.0`.
  - PyInstaller args for Windows and Linux: add `--collect-all cv2_enumerate_cameras`, next to the existing PySpin args.
  - `setup_espi_live.ps1`: install and validate the package like the others.
- [ ] **Step 2: `--check`.** Write `test_cli.py::test_check_reports_video_enumeration`: with the module importable, the output contains `"Video devices: cv2_enumerate_cameras"`; when it isn't, the output contains `"not installed"` and the exit code is still 0. Implement it in `self_check` without enumerating devices.
- [ ] **Step 3: Docs.**
  - `sony-xc-st50-usb-live2.md` covers:
    - wiring: BNC → RCA composite, DC-700 at 13 V (within 10.5–15 V);
    - the camera's field-integration switch, NTSC and the Hauppauge driver;
    - closing WinTV / Hauppauge Capture first;
    - single-field mode, the ~12 % wide image, and the ESPI caveats.
  - Link it from `docs/camera/README.md` and `supported-models.md`.
  - `README.md`: supported cameras.
  - `AGENTS.md`: add rows for `capture_models.py`, `capture.py`, `sources.py` and `camera_reports.py`, and add `camera_report` to the engine events.
  - `CHANGELOG.md`: a `## 2026-10-29 — Analog camera support (Sony XC-ST50 via USB-Live2)` entry in the Previous state / Current state format.
- [ ] **Step 4: Verify:** `$PY -m py_compile src/espi_live/*.py tests/*.py`, then the full suite under coverage at 100 %.
- [ ] **Step 5: Owner-approved outward steps only.**
  - Run `gh label create camera-support --repo AlexWorland/espi-live --color 1D76DB --description "Camera models and video devices reported by the app"`.
  - Commit and push.
  - Capture screenshots offscreen from the real widgets: the USB-Live2 controls (disabled, field toggle), the camera list marker, each report status state with and without the button, and the browser-failure warning. Push them to `pr-screenshots` under `pr-<n>/`.
  - Open the PR. It closes #23, embeds the screenshots and ends with the Claude Code attribution line.
