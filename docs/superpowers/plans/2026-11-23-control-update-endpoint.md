# Control-server update endpoint Implementation Plan

> **For agentic workers:** executed natively in one session (tasks share interfaces). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the running app's control server check for, download and install a newer build, and restart into it, through `GET /update` and `POST /update` plus `espi_live control update`.

**Architecture:** `control.py` routes the endpoint and stays GUI-free (`ControlService` validates, `ControlBackend` in `qt_app.py` does the work). The backend marshals window work to the GUI thread with `MainThreadCaller`. Control-triggered updates never open a dialog: the window tracks the request in `update_remote` ("download" or "install") and only closes itself for "install", reusing the existing `relaunch_after_exit` → `finish_update` path, so the camera is released before the new copy starts.

**Tech Stack:** Python 3.14 stdlib http.server, PySide6 (offscreen in tests), unittest + coverage (100% line gate).

**Spec:** the approved short design in the conversation of 2026-11-23 (endpoint shape: dedicated `/update`; recording in progress: refuse with 409).

## Global Constraints

- Coverage of `src/espi_live/` stays at 100% (`.coveragerc`); no `# pragma: no cover`.
- Every new function has an `Args:` / `Returns:` docstring.
- Tests never open a camera, touch the network, write to `output/` or block on a dialog.
- `python -m py_compile` on each changed file; full suite must pass.
- No commit unless asked. CHANGELOG entry and README endpoint list updated.

## Review Focus

1. Install requested while a recording runs → must return 409 and must not close the window. Test: `test_install_refused_while_recording`.
2. Download completes after recording started → must not close; the build stays pending with no dialog. Test: `test_downloaded_while_recording_waits_for_restart`.
3. Check or download requested while the updater is busy → 409, not a second job. Test: `test_check_refused_while_busy`.
4. Source run (no install target) → every action 409 with the reason. Test: `test_actions_refused_when_updates_unavailable`.
5. Control-triggered download failure or no-update result → no modal dialog left open. Test: `test_remote_download_failure_shows_no_dialog`.

---

### Task 1: Route `GET /update` and `POST /update` (control.py)

**Files:**
- Modify: `src/espi_live/control.py` (module docstring, `UPDATE_ACTIONS`, `ControlService.handle`, `_post_update`, `_overview`)
- Test: `tests/test_control.py`

**Interfaces:**
- Consumes: none.
- Produces: `UPDATE_ACTIONS: dict[str, str]` (name → description for `check`, `download`, `install`); backend methods `update_state() -> dict` and `update_action(name: str) -> dict`; `ControlService._post_update(body) -> dict`.

- [ ] **Step 1: Write failing tests** in `tests/test_control.py`: `UpdateTests(ServiceTestCase)` with
  - `test_get_update_returns_backend_state`: `FakeBackend.update_state` returns `{"available": False}`; `service.handle("GET", "/update", {}, None)` → `(200, {"ok": True, "available": False})`.
  - `test_post_update_runs_the_action`: body `{"action": "check"}` → `(200, {"ok": True, "action": "check", "result": {...}})` and `backend.update_calls == ["check"]`.
  - `test_unknown_action_lists_the_choices`: `{"action": "reboot"}` → 400, message contains `check, download, install`.
  - `test_body_must_be_an_object`: body `[]` → 400.
  - `test_overview_lists_update`: `GET /` endpoints contains `"GET /update"` and `"POST /update"`.
  - Add to `FakeBackend`: `update_calls = []`, `update_state()`, `update_action(name)` (appends, returns `{"requested": name}`), both honoring `self.fail`.

- [ ] **Step 2: Run to verify failure:** `PYTHONPATH=src python -m unittest tests.test_control.UpdateTests -v` → FAIL (unknown address 404).

- [ ] **Step 3: Implement.** Add `UPDATE_ACTIONS` next to `DAQ_ACTIONS`. In `handle`, GET branch: `if route == "/update": return 200, {"ok": True, **self.backend.update_state()}`; POST branch: `if route == "/update": return 200, self._post_update(body)`. `_post_update` mirrors `_post_daq`: require dict, pop `action`, reject unknown with `ControlError("unknown update action …; actions: check, download, install")`, return `{"ok": True, "action": name, "result": self.backend.update_action(name)}`. Add `/update` lines to `_overview` endpoints and the module docstring.

- [ ] **Step 4: Run tests:** the same command → PASS; then `python -m py_compile src/espi_live/control.py`.

- [ ] **Step 5: Commit** — skipped (not requested).

### Task 2: Window update logic on the GUI thread (qt_app.py)

**Files:**
- Modify: `src/espi_live/qt_app.py` (`ControlBackend.update_state`, `ControlBackend.update_action`; `MainWindow.__init__` adds `self.update_remote = None`; `MainWindow._control_update(name)`, `MainWindow._control_update_state()`; `MainWindow._handle_update_events` branches; `MainWindow._control_restart()`)
- Test: `tests/test_qt_app.py`

**Interfaces:**
- Consumes: Task 1 backend contract.
- Produces: `update_state() -> dict` with keys `current` (int), `latest` (int or None), `label` (str or None), `downloaded` (bool), `restart_pending` (bool), `busy` (bool), `downloading` (bool), `progress` ([done, total]), `unavailable` (str or None), `recording` (bool). `update_action(name) -> dict` returns `{"started": "check" | "download" | "install" | "restart"}`.

Rules (GUI thread):
- `ControlBackend.update_action` first raises `ControlError(f"updating is unavailable: {reason}", 409)` when `updater.unavailable_reason()` is set.
- `check`: if `updates.busy` → 409 "an update check or download is already running". Else `update_remote` unchanged, `updates.check()`, return `started: "check"`.
- `download`: if `pending_update` → 409 "the update is already downloaded; use install". If `update_info is None` → 409 "no update is known; run check first". If `updates.busy` → 409. Else `update_remote = "download"`, `updates.download(update_info)`, return `started: "download"`.
- `install`: if `pending_update` → if recording (`_capture_busy()`) → 409 "a recording or frame save is running; stop it first"; else `restart` (below). If `updates.busy` → 409. If recording → 409 (same message). Else `update_remote = "install"`; if `update_info` → `updates.download(update_info)` and `started: "install"`; else `updates.check()` and `started: "install"` (the checked event then starts the download).
- `restart`: `relaunch_after_exit = True`, `call_async(self.close)`, return `started: "restart"`.
- `_handle_update_events` changes (keep the existing manual/dialog paths unchanged when `update_remote is None`):
  - `checked` with a build: if `update_remote == "install"` → `updates.download(payload)`; the existing `_show_available_update` still runs.
  - `checked` with `None` and `update_remote` set → log, `update_remote = None`.
  - `check_failed` / `download_failed`: when `update_remote` is set, log only (no `QMessageBox`), then `update_remote = None`.
  - `downloaded`: `pending_update = payload` and the existing log/button text run. If `update_remote == "install"` and not `_capture_busy()` → `relaunch_after_exit = True` and `self.close()`. If `update_remote` is set otherwise → log "Update downloaded; restart to install when ready" and no `_offer_restart`. Then `update_remote = None`.
- `_control_update_state` reads `updater.current_build()`, `update_info`, `pending_update`, `updates.busy/downloading/progress`, `_capture_busy()`.

- [ ] **Step 1: Write failing tests** in `tests/test_qt_app.py`, using the existing `FakeUpdateWorker` and `demo_window` helpers (patch `updater.unavailable_reason` to None; `_capture_busy` patched for recording cases):
  - `test_check_starts_without_a_dialog`
  - `test_check_refused_while_busy` → `ControlError` status 409
  - `test_download_needs_a_known_update` → 409 when `update_info` is None
  - `test_install_downloads_then_restarts`: feed `("checked", info)` then `("downloaded", path)` through the event queue; assert `relaunch_after_exit` True, `close` called, `QMessageBox.question` not called.
  - `test_install_refused_while_recording` → 409, `relaunch_after_exit` False.
  - `test_downloaded_while_recording_waits_for_restart`: `_capture_busy` True during the `downloaded` event → no close, `pending_update` set, no dialog.
  - `test_remote_download_failure_shows_no_dialog`: `("download_failed", "x")` with `update_remote="download"` → `QMessageBox.warning` not called.
  - `test_actions_refused_when_updates_unavailable` → 409 with reason.
  - `test_restart_closes_the_window`: `restart` action sets `relaunch_after_exit` and calls close via caller.

- [ ] **Step 2: Run to verify failure:** `QT_QPA_PLATFORM=offscreen PYTHONPATH=src python -m unittest tests.test_qt_app -k "Update or update" -v` → FAIL.

- [ ] **Step 3: Implement** per the rules above. Add `update_remote` next to `update_manual` in `MainWindow.__init__`.

- [ ] **Step 4: Run tests** → PASS. `python -m py_compile src/espi_live/qt_app.py`.

- [ ] **Step 5: Commit** — skipped.

### Task 3: `espi_live control update` (CLI)

**Files:**
- Modify: `src/espi_live/control.py` (`add_parser` adds `update` subparser; `_run_update`)
- Test: `tests/test_control.py`, `tests/test_cli.py` if the subcommand is wired there

**Interfaces:**
- Consumes: Task 1 endpoint.
- Produces: `_run_update(args, port) -> None`.

Behavior: `espi_live control update` prints the state (`GET /update`) as one readable line per field, or raw JSON with `--json`. `espi_live control update check|download|install|restart` POSTs `{"action": …}` and prints the result. `--port` follows the other actions.

- [ ] **Step 1: Failing tests:** `test_update_status_prints_state` and `test_update_action_posts_the_action`, both with `request` patched (see existing `_run_status`/`_run_daq` tests for the pattern).
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** the subparser (`update` with optional positional `action` choices `check download install restart`, default None) and `_run_update`.
- [ ] **Step 4: Run → PASS;** `python -m py_compile src/espi_live/control.py`.
- [ ] **Step 5: Commit** — skipped.

### Task 4: Docs and full gate

**Files:** Modify `README.md` (control endpoints list), `CHANGELOG.md` (Unreleased: "Control server can check, download and install updates and restart the app").

- [ ] **Step 1:** Add the entries.
- [ ] **Step 2:** `PYTHONPATH=src QT_QPA_PLATFORM=offscreen python -m coverage run -m unittest discover -s tests && python -m coverage report` → all tests pass, 100%.
- [ ] **Step 3:** `python -m py_compile src/espi_live/*.py`.
