# Always-on capture uploads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** While a real camera is processing, periodically save a raw and a processed still, and less often a raw and a processed short clip, per camera, and upload them silently to a private GitHub repository.

**Architecture:** A small raw-frame tap on `Engine` exposes the camera's unprocessed frame. A GUI-free `snapshots.py` builds the files (naming, size fitting, sidecar). A GUI-free `capture_uploads.py` holds the schedule/budget ledger, a bounded on-disk queue, an uploader (Contents API, behind one `put` seam) and a scheduler thread that reads channels through a provider callback. `qt_app.Reporting` starts it beside the log uploads.

**Tech Stack:** Python 3.10 syntax, NumPy, OpenCV (PNG/AVI via `AVIRecorder`), `unittest`, GitHub Contents API through `feedback.github_request`.

**Spec:** `docs/superpowers/specs/2026-11-21-always-on-capture-design.md` (the spec names one module `capture_uploads.py`; this plan splits file-building into `snapshots.py` so each file stays focused).

## Global Constraints

- New modules are GUI- and camera-free; the front end drives them. No Qt or Tk objects touched from worker threads.
- 100% line coverage of `src/espi_live/`, no `# pragma: no cover`; tests never use a camera, the network, the real `output/` folder, or block on a dialog.
- Python 3.10 syntax; every function has a docstring in the `Args: ... / Returns: ...` format.
- Defaults (constants in `capture_uploads.py`): `STILL_INTERVAL_S = 1800`, `CLIP_INTERVAL_S = 21600`, `CLIP_SECONDS = 5`, `MAX_STILL_BYTES = 3 * 1024 * 1024`, `MAX_CLIP_BYTES = 7 * 1024 * 1024`, `DAILY_BYTES = 60 * 1024 * 1024`, `CLIP_RESERVE_BYTES = 30 * 1024 * 1024`, `MIN_PACKET_RATE = 2.0`, backoff 1 minute doubling to 1 hour, scheduler wake about every 60 s.
- Repository `AlexWorland/espi-live-captures`; env `ESPI_CAPTURES_REPO` / `ESPI_CAPTURES_TOKEN`; token file `captures-token.txt`; build values `CAPTURES_TOKEN` and `CAPTURES_REPO`; remote path `captures/<YYYY-MM-DD>/<HHMMSS>-<machine>-<camera>-<raw|processed>.<png|avi>` plus one `.json` sidecar per capture; `<machine>` is `log_uploads.machine_id`; no camera serial numbers anywhere in names or sidecars.
- Only FLIR (`info.kind == "spinnaker"`) and supported capture adapters (`info.kind == "capture"` and `channel.supported`) upload. Never demo, iPhone, unsupported video devices, stopped/errored channels, or a frozen display.
- No per-install switch. Failures log at DEBUG only and are never shown or reported. Do not change default exposure, gain, frame rate or the user's recording behaviour. Commit steps run only once the owner has asked for commits (AGENTS.md: don't commit unless asked); otherwise leave the work uncommitted.
- Test command for every task: `PYTHONPATH=src:tests QT_QPA_PLATFORM=offscreen ~/espi_live/.venv/bin/python -m unittest discover -s tests -p '<file>'` (worktrees have no `.venv`).

## Review Focus

- A channel stops, errors or disconnects in the middle of a clip: both partial clip files are deleted and no half pair is queued or uploaded (Task 4).
- The output folder is unwritable or the disk is full: the capture is dropped, the worker thread survives, the camera and UI are untouched (Tasks 3, 4).
- `schedule.json` is corrupt, empty, or has timestamps in the future (clock change): each camera becomes due once, with no burst of captures and no permanent silence (Task 3).
- Deep (`float32`) raw frames and all-black or blank frames: raw files are valid 8-bit PNGs with the scaling recorded in the sidecar; a blank frame is skipped (Tasks 2, 4).
- The app exits with captures still queued, or an upload succeeded but the reply was lost: the next start uploads the leftovers once, and a "file already exists" reply (422) counts as done (Task 3).

---

### Task 1: Raw-frame tap on the Engine

**Files:**
- Modify: `src/espi_live/engine.py` (add `RawFrame`, `Engine.set_raw_tap`, `Engine.latest_raw`; store in `_loop` right after the Mono8/deep dtype check and `single_field` step, before `processor.denoise`)
- Test: `tests/test_engine.py`

**Interfaces:**
- Produces: `RawFrame` frozen dataclass `(sequence: int, image: np.ndarray)` (image never modified after publication); `Engine.set_raw_tap(on: bool) -> None`; `Engine.latest_raw() -> RawFrame | None`.

- [ ] **Step 1: Write failing tests** in the existing demo-engine style of `tests/test_engine.py`: `test_raw_tap_off_stores_nothing` (`latest_raw()` is `None` after frames flow), `test_raw_tap_on_publishes_copy_before_processing` (the array is not the published packet's image object, mutating it does not change the packet, `sequence` increases between two reads), `test_raw_tap_off_clears_stored_frame`.
- [ ] **Step 2: Run** `... -p 'test_engine.py'`; expect the new tests to fail with `AttributeError`.
- [ ] **Step 3: Implement** the three items above. The tap is a `threading.Event`; when set, the loop stores `RawFrame(next sequence, frame.copy())` under `_packet_lock`; when not set the loop does no copy.
- [ ] **Step 4: Run** `test_engine.py` and `test_engine_camera.py`; expect PASS.
- [ ] **Step 5: Commit** `Add a raw-frame tap to the engine`.

### Task 2: `snapshots.py`: eligibility, names, sidecar, size fitting

**Files:**
- Create: `src/espi_live/snapshots.py`
- Test: `tests/test_snapshots.py`

**Interfaces:**
- Consumes: `Channel` attributes `active`, `demo`, `supported`, `info`, `frozen`.
- Produces (exact):
  - `eligible(channel) -> bool`
  - `camera_label(info, index: int) -> str` (short safe model label plus `index`, never the serial; `index` is the channel's position in the channel list)
  - `remote_stem(when: datetime, machine: str, camera: str) -> str` returning `captures/YYYY-MM-DD/HHMMSS-<machine>-<camera>`
  - `to_uint8(image) -> tuple[np.ndarray, bool]` (the bool says it was rescaled, for float32 deep frames)
  - `png_within(image, max_bytes: int) -> tuple[bytes, float] | None`: lossless PNG bytes and the scale applied; downscales with `cv2.INTER_AREA` in 0.75 steps; `None` if it would need a scale below 0.25
  - `shrink_clip(path: str, max_bytes: int) -> float | None`: re-encode MJPG smaller in place (frame size first, then drop frames) until it fits; returns the scale used, or `None` after deleting the file when it cannot fit
  - `sidecar(settings, info, view: str, scale_raw: float, scale_processed: float, rescaled_raw: bool, build: int) -> dict` (JSON-able: the full `Settings` as a dict, model, resolution, view, scales, build; no serial)

- [ ] **Step 1: Write failing tests**, one per function. `eligible`: true for an active FLIR channel; false for demo, stopped (`active` False), `kind == "iphone"`, unsupported `kind == "capture"`, and a channel with `frozen` set. `camera_label`: contains the model and the index, does not contain `info.serial`. `remote_stem`: exactly `captures/2026-11-21/093005-ab12cd34-FLIR-0`-shaped output for a fixed datetime. `to_uint8`: uint8 passes through unchanged with `False`; float32 is min/max scaled to uint8 with `True`; an all-equal float32 image does not divide by zero. `png_within`: a noisy 2000x1500 uint8 image with `max_bytes=300_000` returns `<= 300_000` bytes and a scale below 1; a smooth image returns scale 1.0; an impossible budget returns `None`. `shrink_clip`: write a short MJPG AVI with `cv2.VideoWriter`, shrink to half its size, assert the file is under the cap; an impossible cap deletes the file and returns `None`. `sidecar`: `json.dumps` works and the serial string is absent.
- [ ] **Step 2: Run** `... -p 'test_snapshots.py'`; expect import failure.
- [ ] **Step 3: Implement** the functions above with the signatures given.
- [ ] **Step 4: Run** the file; expect PASS and no uncovered lines in `snapshots.py`.
- [ ] **Step 5: Commit** `Add snapshot naming, fitting and sidecar helpers`.

### Task 3: Ledger, queue and uploader in `capture_uploads.py`

**Files:**
- Create: `src/espi_live/capture_uploads.py` (constants from Global Constraints, `captures_token()`, `repository()` built like `log_uploads.logs_token`/`repository` using `updater.read_token("ESPI_CAPTURES_TOKEN", "captures-token.txt", "CAPTURES_TOKEN")`)
- Test: `tests/test_capture_uploads.py`

**Interfaces:**
- Consumes: `feedback.github_request(method, path, token, payload=None)` and `feedback.FeedbackError(message, status)`; `error_reports.write_json`.
- Produces (exact):
  - `CaptureLedger(folder: str, clock=time.time)`: `due(camera: str, kind: str, interval_s: float) -> bool`; `mark(camera: str, kind: str) -> None`; `afford(kind: str, nbytes: int) -> bool` (kind is `"still"` or `"clip"`; stills may not use the clip reserve; a day is the local calendar day); `spend(kind: str, nbytes: int) -> None`. State in `<folder>/schedule.json`, written with `write_json`. Missing, corrupt or future-dated entries make a camera due once and are then rewritten.
  - `CaptureQueue(folder: str, max_bytes: int)`: `add(stem: str, files: dict[str, bytes]) -> bool` (writes into a temporary folder and renames it so a half group is never visible; returns `False` and drops the group on `OSError`); `groups() -> list[str]` oldest first; `read(stem) -> dict[str, bytes]`; `remove(stem) -> None`; the queue deletes the oldest groups when over `max_bytes`; leftover temporary folders are removed at construction.
  - `CaptureUploader(queue, request=feedback.github_request, token=None, clock=time.time)`: `put(path: str, data: bytes) -> None` (Contents API PUT creating the file, `message` `"Capture <path>"`, branch `main`); `send_pending() -> int` uploads groups oldest first, deleting a group after every file in it succeeded; error handling: `401` or a `404` on the repository itself sets `stopped` for the run; `403`, `409`, `422 "already exists"` (counts as done), `5xx` and network errors back off `1 min` doubling to `1 hour` and are retried later; nothing is raised.

- [ ] **Step 1: Write failing tests** with a fake request like `FakeContents` in `tests/test_log_uploads.py`. Ledger: `test_due_first_time_then_not_until_interval`, `test_corrupt_schedule_is_due_once_without_burst`, `test_future_timestamp_is_due_once`, `test_still_cannot_spend_clip_reserve`, `test_budget_resets_next_day`. Queue: `test_group_is_added_whole_or_not_at_all`, `test_oldest_groups_dropped_over_max_bytes`, `test_leftover_temp_folder_removed`, `test_add_returns_false_when_folder_unwritable` (point it at a path under a regular file). Uploader: `test_uploads_group_and_deletes_it`, `test_422_already_exists_counts_as_done`, `test_403_backs_off_and_retries_later` (advance the injected `clock`), `test_401_stops_for_the_run`, `test_404_on_repo_stops_but_404_elsewhere_does_not` , `test_no_token_sends_nothing`, `test_leftover_group_from_earlier_run_is_uploaded_once`.
- [ ] **Step 2: Run** `... -p 'test_capture_uploads.py'`; expect import failure.
- [ ] **Step 3: Implement** the three classes and helpers with the signatures given. Mirror `log_uploads.py` style for tokens and request wiring; the base64 payload is `{"message", "branch", "content"}`.
- [ ] **Step 4: Run** the file; expect PASS.
- [ ] **Step 5: Commit** `Add the capture ledger, queue and uploader`.

### Task 4: Scheduler worker

**Files:**
- Modify: `src/espi_live/capture_uploads.py` (add `CaptureWorker`)
- Test: `tests/test_capture_worker.py`

**Interfaces:**
- Consumes: Task 1 (`Engine.set_raw_tap`, `latest_raw`), Task 2 (`snapshots.*`), Task 3 (`CaptureLedger`, `CaptureQueue`, `CaptureUploader`), `recording.AVIRecorder(path, frame, fps)` with `submit(frame)`, `stop()`, `done`; `Engine.latest_packet()`, `status()["processed_fps"]` and `["camera_fps"]`, `settings().record`; `log_uploads.machine_id(folder)`.
- Produces: `CaptureWorker(channels, folder, uploader, ledger, queue, machine, build=0, interval_s=60.0)` where `channels` is a zero-argument callable returning the current channel list; `start()`, `kick()`, `stop()`, `run_once() -> int` (captures queued this pass; used by tests and the thread body).

- [ ] **Step 1: Write failing tests** with fake channels and a fake engine (a stub with `latest_packet`, `latest_raw`, `set_raw_tap`, `status`, `settings`, `demo`) and a fake `AVIRecorder` where useful: `test_still_pair_is_queued_with_sidecar`, `test_raw_and_processed_clip_pair_are_queued_together`, `test_ineligible_channels_are_skipped` (demo, stopped, frozen, iPhone), `test_blank_frame_is_skipped`, `test_clip_skipped_when_user_is_recording_but_still_taken`, `test_clip_pair_skipped_when_processed_rate_below_two`, `test_channel_stopping_mid_clip_deletes_both_partials_and_queues_nothing`, `test_over_budget_drops_whole_pair_never_half`, `test_unwritable_folder_drops_capture_and_worker_survives`, `test_raw_tap_is_switched_off_again_after_each_capture`, `test_worker_thread_survives_an_exception_in_a_pass`, `test_float32_raw_frame_saved_as_valid_png_and_sidecar_records_rescale`.
- [ ] **Step 2: Run** `... -p 'test_capture_worker.py'`; expect `AttributeError`.
- [ ] **Step 3: Implement** `CaptureWorker`. Per pass, for each eligible channel and in this order: take a still pair if `ledger.due(..., STILL_INTERVAL_S)` and the budget allows; take a clip pair if `due(..., CLIP_INTERVAL_S)`, the user is not recording and the processed packet rate is at least `MIN_PACKET_RATE`. The raw tap is on only while a capture needs it (wait up to 5 s for the first raw frame). A clip runs on this worker thread for `CLIP_SECONDS`, feeding two `AVIRecorder`s from `latest_raw()` and `latest_packet().render()` at the camera/processed rate; it aborts and deletes both files when the channel stops being eligible. Fit every file with `snapshots.png_within`/`shrink_clip`, spend the budget only for a complete pair, `queue.add` the pair with its sidecar, then `uploader.send_pending()`. The thread body catches `Exception` per pass and logs at DEBUG, like `LogUploadWorker`.
- [ ] **Step 4: Run** `test_capture_worker.py` and `test_capture_uploads.py`; expect PASS.
- [ ] **Step 5: Commit** `Schedule raw and processed stills and clips per camera`.

### Task 5: Wire it into the Qt front end and startup log

**Files:**
- Modify: `src/espi_live/qt_app.py` (`Reporting.__init__`, `start`, `start_uploads` near line 247-270 and the call at about line 7221; wherever `reporting.uploads.stop()` is called on close), `src/espi_live/cli.py` (mirror what line ~103 does for `logs_token` in the status output)
- Test: `tests/test_qt_app.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `CaptureWorker`, `capture_uploads.captures_token()`.
- Produces: `Reporting.start_uploads(channels)` where `channels` is a zero-argument callable (the window passes `lambda: window.channels`); `Reporting.captures` is the `CaptureWorker`.

- [ ] **Step 1: Write failing tests** beside the existing `Reporting`/log-upload tests: `test_start_uploads_starts_capture_worker_with_token`, `test_no_capture_worker_runs_without_token`, `test_startup_log_says_capture_uploads_on_or_off` (`"Capture uploads: on"` / `"Capture uploads: off (no captures token in this copy)"`), `test_close_stops_capture_worker`; a CLI test that the token status line mirrors the logs one.
- [ ] **Step 2: Run** the two test files; expect the new tests to fail.
- [ ] **Step 3: Implement** the wiring. The Tk front end is left alone.
- [ ] **Step 4: Run** `test_qt_app.py` and `test_cli.py`; expect PASS.
- [ ] **Step 5: Commit** `Start capture uploads with the log uploads`.

### Task 6: Release workflows, docs and the full gate

**Files:**
- Modify: `.github/workflows/build-binaries.yml` and `.github/workflows/quick-release.yml` (add `CAPTURES_TOKEN: ${{ secrets.CAPTURES_TOKEN }}` beside `LOGS_TOKEN`, write it into `_build_info`, warn `::warning::CAPTURES_TOKEN is empty; this build will not upload captures` when empty, extend the summary string), `AGENTS.md` (module table rows for `snapshots.py` and `capture_uploads.py`, the tokens rule naming `CAPTURES_TOKEN`: Contents read and write on `espi-live-captures` only, one line in the frame-flow or engine section about the raw tap), `README.md`, `CHANGELOG.md` (entry in the existing dated Previous state / Current state format), `src/espi_live/user_guide.html` (one sentence: images and short clips are sent), and the guide test that pins the text if one exists
- Test: `tests/test_guide.py` (if it pins guide text)

- [ ] **Step 1: Edit** the files above. Manual owner steps, listed in the CHANGELOG entry and the PR description: create the private repo `AlexWorland/espi-live-captures`, issue a fine-grained token with Contents read and write on that repository only, store it as the `CAPTURES_TOKEN` secret.
- [ ] **Step 2: Run** `python -m py_compile src/espi_live/*.py`; expect no output.
- [ ] **Step 3: Run** `PYTHONPATH=src QT_QPA_PLATFORM=offscreen ~/espi_live/.venv/bin/python -m coverage run -m unittest discover -s tests && ~/espi_live/.venv/bin/python -m coverage report`; expect all tests pass and coverage `100%` (the report fails below it).
- [ ] **Step 4: Run** `python -c "import yaml,sys; [yaml.safe_load(open(f)) for f in sys.argv[1:]]" .github/workflows/*.yml`; expect no error.
- [ ] **Step 5: Commit** `Document and release capture uploads`.

---

## Self-review

- **Spec coverage:** raw and processed stills and clips (Tasks 1, 2, 4); schedule persistence (3); queue bound and daily/clip-reserve budgets (3); remote path and sidecar without serials (2, Global Constraints); source filter, demo, frozen and recording rules (2, 4); error mapping and backoff (3); token, repo and build wiring (3, 6); Qt wiring and disclosure (5, 6); docs and tests (all). The `put(path, data)` seam for a later relay is `CaptureUploader.put`.
- **Types:** `RawFrame` (Task 1) is consumed in Task 4; `snapshots` signatures in Task 2 match the calls in Task 4; `CaptureWorker.channels` is a callable in Tasks 4 and 5.
- **Open choices left to the implementer:** the downscale ladder detail, the AVI re-encode loop, the queue's default `max_bytes` (suggest 200 MB), and the sidecar field order.
