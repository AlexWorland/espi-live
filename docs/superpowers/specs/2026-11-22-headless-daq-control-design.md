# Headless NI DAQ control (design)

Status: approved and built. Size: a small-to-medium change (one new module, small edits to `headless.py` and `cli.py`, new tests).

## Intent

`--headless` can open cameras and record, but it cannot touch the NI USB DAQ. To test the generator, the piezo and the Phase view from code, someone has to click through Devices → NI DAQ. The goal is to let a script, a test harness or an agent configure and drive the DAQ (and change camera settings) in a running headless process, and read back what happened.

## Goals

- Drive every DAQ function the window offers, headless: scan, connect (real or `SIMULATED`), apply settings, zero, self-test, piezo steps, attach/detach the Phase-view stepper.
- Change camera settings, save a frame, take a reference, mid-run, with the validation a saved profile gets.
- Read results back: replies, `status`, and DAQ events (`fault`, `rejected`, `self_test` results) as structured JSON.
- Testable with no hardware: the dispatcher is unit-tested with the in-memory `fake_nidaqmx` and `SIMULATED`, and stays inside the 100% coverage gate.

## Non-goals

- No network listener, no remote control, no authentication scheme.
- No change to the GUI, the DAQ safety rules, or the existing batch behaviour of `--headless` (config file, `--duration`, uploads).
- No new scripting language: a command is one JSON object.
- Phase *processing* is unchanged; headless only attaches the existing `phase.PiezoStepper` to an engine.

## Anti-goals (failure even if it "works")

- A drive signal reaching a real device without the safety checklist having been explicitly attested.
- Outputs left driven after a crash, EOF on stdin, Ctrl+C or any error.
- Log lines corrupting the reply stream, so a harness can't parse replies.

## Approaches considered

- **A. Dispatcher + stdin/stdout JSON lines (chosen).** Portable, no port or firewall prompt, not reachable from another machine, drives the real `--headless` process; the dispatcher is also called directly in unit tests.
- B. Dispatcher + local TCP socket: allows reconnecting clients, but opens a port beside a piezo amplifier and adds a firewall prompt on the camera PC. Not needed.
- C. Python library only: simplest, but not usable from a shell or another language and doesn't exercise the real process.

## Design

**Modules.** New `src/espi_live/headless_control.py` (no GUI imports): `Controller`-style class `ControlSession(channels, daq_controller, log, config)` with `execute(command: dict) -> dict`, plus `serve(session, stdin, out)` which reads lines, calls `execute`, writes one reply line. `headless.py` builds the session when `--control` is given and runs `serve` in place of the timed `loop` (the camera event pump still runs on the main thread between commands). `cli.py` gains `--control` and `--daq-checklist-done`.

**Wire format.** One JSON object per line in, one per line out. A command is `{"cmd": "<name>", "id": <optional, echoed>, ...args}`. A reply is `{"id": ..., "ok": true, ...result}` or `{"id": ..., "ok": false, "error": "<text>", "problems": [...]}`. Replies go to the process's *original* stdout; `AppLog` replaces `sys.stdout`/`sys.stderr` with mirror streams, so the transport keeps the original handle from before `AppLog.start()`. Logs stay on stderr and in the log file.

**Commands.**

| Command | Effect |
|---------|--------|
| `status` | Cameras (label, running, fps, settings), DAQ (`DaqController.status()`, armed roles, stepper attached), pending DAQ events |
| `daq_scan` | `request("scan")`; reply lists devices (SIMULATED always included) |
| `daq_connect {device}` | `request("connect")`, waits for `connected`/`failed`; zero on connect is the controller's own |
| `daq_arm {roles?}` | Records the checklist (`controller.arm`) for the named roles (default: those switched on in the current settings). Refused unless the process was started with `--daq-checklist-done`; not needed for `SIMULATED` |
| `daq_apply {settings}` | `DaqSettings.from_dict` + `daq.problems`; a rejection returns the problems; otherwise waits for `applied`/`rejected`/`failed` |
| `daq_zero`, `daq_self_test` | As the window's buttons; reply carries the result lines |
| `piezo_step {volts}` | `DaqController.move_piezo` (the same call phase stepping uses) |
| `phase_attach` / `phase_detach` | Builds/removes `phase.PiezoStepper` from the last applied settings and gives it to the chosen camera's engine only (same rules as `MainWindow._attach_stepper`); refused when the piezo is unarmed or unassigned |
| `set_camera {settings, camera?}` | `headless.apply_settings` (profile validation) |
| `save`, `reference` | Engine requests, as the window's actions |
| `wait {seconds}` | Sleeps while pumping events (so a harness can let frames flow) |
| `quit` | Normal shutdown |

**Safety.** The gate stays in `DaqController`; headless adds no bypass. `--daq-checklist-done` only makes the `daq_arm` command available; without it a real device answers drive requests with `CHECKLIST_REQUIRED`. The controller's own self-checks (0 V on connect, readback, watchdog) stay on; the session calls `daq.heartbeat()` from its loop, as the window does. Every exit path (`quit`, EOF, SIGINT/SIGTERM, exception) calls `DaqController.shutdown`, which zeroes first. The simulated device stays visibly marked in `status`.

**Failure handling.** A malformed line, unknown command or bad argument returns `ok: false` and the session keeps running. A DAQ `fault` event is sent as an unsolicited `{"event": "fault", ...}` line and the session reports it in later `status` replies.

**Tests (`tests/test_headless_control.py`).** Dispatcher against `SIMULATED` and the fake-nidaqmx real backend (arm refusal and acceptance, apply rejection, piezo step, phase attach with a demo channel, fault event), the stdin loop with `io.StringIO` (EOF zeroes outputs, bad JSON survives), and one subprocess test of `python -m espi_live --headless --demo --control` over a pipe. `headless_control.py` is inside the coverage gate; `headless.py` stays omitted unless its new lines are cheap to cover.

**Docs.** `README.md` headless paragraph (replace "The NI DAQ and Phase view are not driven headless"), `AGENTS.md` architecture row for `headless_control.py` and the `cli.py` options, a `CHANGELOG.md` entry.

## My calls (not discussed; change any you disagree with)

- `--control` is its own flag rather than a config key, so a stray config file can't open a control channel.
- Commands run one at a time on the main thread; no concurrency between commands.
- `daq_arm` takes roles from the current settings by default, because the checklist covers "the roles switched on" and a harness shouldn't have to know the role names.
- `--duration` still applies in control mode if given; otherwise the session runs until `quit` or EOF.
- A control run starts even when no camera opens (the DAQ commands do not need one), so exit code 2 does not apply to it.
- A control run never uploads, whatever the config says, so a test session cannot publish captures by accident.
- stdout is diverted to replies from the first log line (the AppLog console handler and printed text go to stderr), so the reply stream stays clean.
- `set_camera` rejects the whole command when any name or value is wrong (no partial application and no silent fallback to defaults, unlike loading a saved profile).
- `phase_attach` switches the camera to the Phase view, as the window does when a piezo arrives; `phase_detach` leaves the view alone.

## Left to the builder

Exact reply field names beyond those above, timeouts for waiting on DAQ events, and the wording of error text.
