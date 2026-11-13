# Automatic Log Uploads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Installed builds silently upload scrubbed run logs to the private `AlexWorland/espi-live-logs` repository.

**Architecture:** A GUI-free module `log_uploads.py` holds a `LogUploader` (finds logs that are new or grew, scrubs, caps, PUTs them through the GitHub Contents API, records what it sent) and a `LogUploadWorker` thread that runs a pass at start and every 15 minutes. `qt_app.Reporting` owns the worker and shares the error reporter's on/off state.

**Tech Stack:** Python 3.10-compatible stdlib (`urllib`, `threading`, `json`, `base64`), PySide6 only for the menu text, `unittest`.

**Spec:** `docs/superpowers/specs/2026-11-13-log-uploads-design.md`

## Global Constraints

- Destination repo: `AlexWorland/espi-live-logs` (`LOGS_REPO`), branch `main`.
- Remote path: `logs/<YYYY-MM-DD>/<HHMMSS>-<machine>.log`, date/time from the run log's name `espi_YYYYMMDD_HHMMSS_ffffff.log`.
- Machine ID: 8 lowercase hex digits, random, stored in `<logs folder>/machine-id.txt`.
- Interval 15 minutes (`INTERVAL_S = 900`); cap 5 MB (`MAX_BYTES = 5 * 1024 * 1024`), head kept `HEAD_BYTES = 256 * 1024`.
- Token: `updater.read_token("ESPI_LOGS_TOKEN", "logs-token.txt", "LOGS_TOKEN")`.
- Everything uploaded passes through the error reporter's scrubber.
- Failures: `logger.debug` only; never `error_reports.capture`, never UI.
- No GUI imports in `log_uploads.py`; docstrings in `Args:/Returns:` form; 3.10 syntax; 100% line coverage, no pragmas; tests never touch the network or the real `output/`.

## Review Focus

1. A log file that is still being written (the current run) uploads its partial content and re-uploads later when it grew; it is never marked fully sent on a short read. → Task 2 `test_grown_log_is_uploaded_again`.
2. The remote file already exists but the local record is gone (record deleted, reinstall): the PUT must carry the remote `sha`, not fail forever. → Task 2 `test_existing_remote_file_is_updated_with_its_sha`.
3. Network down / token revoked (401) / repo missing (404): the pass stops quietly, nothing is recorded as sent, the next pass retries. → Task 2 `test_failure_stops_pass_and_records_nothing`.
4. A log with non-UTF-8 bytes or a cut through a multi-byte character at the cap boundary must not raise. → Task 2 `test_cap_keeps_head_and_tail_and_survives_bad_bytes`.
5. Turning reports off in the Help menu stops uploads within the current pass's next file. → Task 3 `test_menu_switch_turns_uploads_off`.

---

### Task 1: Let the shared GitHub call do PUT and expose the HTTP status

**Files:**
- Modify: `src/espi_live/feedback.py` (`FeedbackError`, `explain_http_error`, `github_request` docstring)
- Test: `tests/test_feedback.py`

**Interfaces:**
- Produces: `FeedbackError.status` (int HTTP code, or `None` for network/decoding errors); `github_request("PUT", path, token, payload)` works (it already passes `method` through; document it).

- [ ] **Step 1: Write failing tests** `test_http_error_keeps_status` (a 404 `HTTPError` → `explain_http_error(exc).status == 404`) and `test_network_error_has_no_status` (`URLError` → `.status is None`).
- [ ] **Step 2: Run** `PYTHONPATH=src python -m unittest tests.test_feedback -v` → the two new tests FAIL.
- [ ] **Step 3: Implement** `FeedbackError.__init__(self, message, status=None)`; `explain_http_error` passes `exc.code` for `HTTPError`. Update `github_request`'s docstring to "GET, POST, PATCH or PUT".
- [ ] **Step 4: Run** the same command → PASS.
- [ ] **Step 5: Commit** `Keep the HTTP status on GitHub request errors`.

### Task 2: `log_uploads.py`: uploader and worker

**Files:**
- Create: `src/espi_live/log_uploads.py`
- Test: `tests/test_log_uploads.py`

**Interfaces:**
- Consumes: `feedback.github_request`, `feedback.FeedbackError.status` (Task 1); `logs.run_logs(folder)`, `logs.LOG_SUFFIX`; `updater.read_token`, `updater.build_value`; `error_reports.write_json`.
- Produces:
  - `logs_token() -> str`; `repository() -> str` (env `ESPI_LOGS_REPO`, else build `LOGS_REPO`, else `"AlexWorland/espi-live-logs"`).
  - `machine_id(folder) -> str`: reads or creates `machine-id.txt` (`secrets.token_hex(4)`); returns `"unknown"` if it cannot write.
  - `remote_path(log_name, machine) -> str | None`: `None` for names that do not match `espi_YYYYMMDD_HHMMSS_*.log`.
  - `read_for_upload(path, max_bytes=MAX_BYTES, head_bytes=HEAD_BYTES) -> (text, size)`: `size` is the byte length read; beyond the cap returns head + `"\n[... N bytes left out ...]\n"` + tail; decodes with `errors="replace"`.
  - `class LogUploader(folder, scrubber, enabled, request=feedback.github_request, token=None)`: `enabled` is a zero-argument callable; `scrubber` a zero-argument callable returning a text→text function (pass `ErrorReporter.scrubber`); `send_pending() -> int` (files uploaded). Record file `<folder>/log_uploads.json`: `{log_name: {"size": int, "path": str}}`. A pass: return 0 if not `enabled()` or no token; for each run log (oldest first) whose size differs from the record, at most `MAX_PER_PASS = 10`: GET `/repos/{repo}/contents/{path}` for the `sha` (404 → none), PUT `{"message": "Log <path>", "content": base64, "branch": "main", "sha"?}`, then record `size`. Re-check `enabled()` before each file. Any `FeedbackError` or `OSError`: `logger.debug`, stop the pass.
  - `class LogUploadWorker(uploader, interval_s=INTERVAL_S)`: `start()`, `kick()`, `stop()`; thread name `"log-uploads"`, daemon; loop `wake.wait(interval_s)` → `send_pending()`; any exception is `logger.debug`'d, never raised.

- [ ] **Step 1: Write failing tests** in `tests/test_log_uploads.py`, with a fake `request(method, path, token, payload)` recording calls and a temporary folder holding fake run logs:
  - `test_remote_path_layout`: `remote_path("espi_20261113_142530_000001.log", "ab12cd34") == "logs/2026-11-13/142530-ab12cd34.log"`; `remote_path("notes.log", "x") is None`.
  - `test_machine_id_is_stable`: two calls return the same 8-hex string; unwritable folder → `"unknown"`.
  - `test_uploads_new_logs_scrubbed`: one PUT per log, decoded content passes through the scrubber (`"secret"` → `"<x>"`), `branch == "main"`, no `sha` when GET is 404; record holds the size.
  - `test_unchanged_log_is_not_sent_again`: a second pass makes no calls.
  - `test_grown_log_is_uploaded_again`: append to a log → second pass PUTs it with the `sha` the GET returned.
  - `test_existing_remote_file_is_updated_with_its_sha`: no record, GET returns `{"sha": "abc"}` → PUT payload `sha == "abc"`.
  - `test_failure_stops_pass_and_records_nothing`: PUT raises `FeedbackError("x", 401)` → returns 0, no record written, later logs not tried.
  - `test_off_or_tokenless_sends_nothing`: `enabled=lambda: False` → no calls; `token=""` → no calls.
  - `test_switch_off_mid_pass`: `enabled` flips False after the first file → only one PUT.
  - `test_at_most_ten_per_pass`: 12 logs → 10 PUTs.
  - `test_cap_keeps_head_and_tail_and_survives_bad_bytes`: small `max_bytes`/`head_bytes`, content with `b"\xff"` and a split UTF-8 character → no exception, text starts with the head, ends with the tail, contains `"bytes left out"`.
  - `test_worker_runs_at_start_and_on_kick_and_survives_errors`: a fake uploader whose `send_pending` raises once, then counts; `start()` + `kick()` + `stop()` and `thread.join(1)` → called at least twice, thread ended.
- [ ] **Step 2: Run** `PYTHONPATH=src python -m unittest tests.test_log_uploads -v` → FAIL (module missing).
- [ ] **Step 3: Implement** `src/espi_live/log_uploads.py` with the interfaces above, a module docstring explaining the private repo, the token's scope (Contents on that repo only) and why everything is scrubbed.
- [ ] **Step 4: Run** the same command → PASS; `python -m py_compile src/espi_live/log_uploads.py`.
- [ ] **Step 5: Commit** `Upload run logs to the private logs repository`.

### Task 3: Wire it into the app

**Files:**
- Modify: `src/espi_live/qt_app.py` (`Reporting.__init__/start/choose/stop`, `reports_action` text and tooltip)
- Modify: `src/espi_live/cli.py` (`--check` names `logs token: present|absent`)
- Test: `tests/test_qt_app.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `log_uploads.LogUploader`, `log_uploads.LogUploadWorker`, `log_uploads.logs_token` (Task 2).
- Produces: `Reporting.uploads` (the `LogUploadWorker`); the uploader's `enabled` is `lambda: self.reporter.enabled`, its scrubber `self.reporter.scrubber`, its folder `paths.output_folder("logs")` resolved in `start()` (tests patch it).

- [ ] **Step 1: Write failing tests**: `test_reporting_starts_and_stops_log_uploads` (after `start`, `reporting.uploads.thread` is alive; after `stop`, `stopping` is set), `test_menu_switch_turns_uploads_off` (`choose(False)` → `uploader.enabled()` is False; `choose(True)` kicks the worker), `test_reports_action_names_logs` (menu text `"Send error reports and logs automatically"`), and in `test_cli.py` `--check` output contains `"logs token: "`. Patch the worker's uploader `request` so nothing reaches the network.
- [ ] **Step 2: Run** `PYTHONPATH=src QT_QPA_PLATFORM=offscreen python -m unittest tests.test_qt_app tests.test_cli -v` → new tests FAIL.
- [ ] **Step 3: Implement.** `Reporting.start` logs `"Log uploads: on"`/`"off (...)"` like error reports. Tooltip: `"Unexpected errors are filed as public GitHub issues, and run logs are sent to the developer, without your user name, folders or camera serial numbers"`.
- [ ] **Step 4: Run** the same command → PASS.
- [ ] **Step 5: Commit** `Send logs automatically with the error-report switch`.

### Task 4: Build, docs and full verification

**Files:**
- Modify: `.github/workflows/build-binaries.yml` ("Write build info": `LOGS_TOKEN: ${{ secrets.LOGS_TOKEN }}` env; `values["LOGS_TOKEN"]`; `values["LOGS_REPO"] = "AlexWorland/espi-live-logs"`; empty → `print("::warning::LOGS_TOKEN is empty; this build will not upload logs")`; extend the build-info print and the generated docstring to name `log_uploads.py`)
- Modify: `src/espi_live/user_guide.html` (one sentence where error reports are described), `README.md`, `AGENTS.md` (module table row, the tokens rule naming `LOGS_TOKEN` Contents: read & write on `espi-live-logs` only), `CHANGELOG.md`

- [ ] **Step 1:** Make the edits above.
- [ ] **Step 2: Run** `docker run --rm -v "$PWD":/repo -w /repo rhysd/actionlint:latest` → no output.
- [ ] **Step 3: Run** `python -m py_compile src/espi_live/*.py tests/*.py` and `PYTHONPATH=src QT_QPA_PLATFORM=offscreen python -m coverage run -m unittest discover -s tests && python -m coverage report` → all pass, `TOTAL ... 100%`.
- [ ] **Step 4: Live check (one file):** `ESPI_LOGS_TOKEN=$(cat <token file>)` and a temporary folder with one fake run log, call `LogUploader(...).send_pending()` once → the file appears under `logs/` in `espi-live-logs`; delete it afterwards.
- [ ] **Step 5: Commit** `Bake the logs token into builds and document log uploads`; open the PR labelled `enhancement`, with a screenshot of the renamed Help-menu item pushed to `pr-screenshots`.
