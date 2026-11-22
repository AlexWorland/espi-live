# Simulated ESPI Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace dummy mode's test pattern with an ideal-sensor simulation of an ESPI interferogram whose specimen is still, slowly deforms, vibrates, or follows the NI DAQ window's outputs.

**Architecture:** A new GUI-free module `src/espi_live/simulation.py` holds the optics, the plate and the exact exposure-average lookup table (`FringeTable`), plus the `DemoSource` class the engine already uses. `daq.plan()` attaches a frozen `daq.Drive` (what the outputs produce) to its `OutputPlan`. The window passes it to every `Channel`, and each demo channel's engine receives it through `request("drive", ...)`. A demo-only **Simulated object** combo writes `Settings.demo_object`.

**Tech Stack:** Python 3.10–3.14, NumPy, OpenCV, PySide6, `unittest` + `coverage`.

**Spec:** `docs/notes/simulated-espi-signal-design.md` (read it first; section numbers below refer to it).

## Global Constraints

- **Amendment (2026-11-21, user request): the specimen is a drumhead, not a plate.** Spec §5.2 is binding: a circular membrane, diameter 88 % of the shorter side, modes J_m(j_mn r/R)·cos(m(θ − 30°)) with j_mn ≤ 30, (1, 1) = 1000 Hz, fundamental (0, 1) = 627.6 Hz, drive pad at R/3 and 30°. `DemoSource.drum_mask` replaces `plate_mask`. Tasks 1–4 were done on the plate and then converted. Where a later task names a plate frequency, use the drum's: 628 / 1000 / 1340 / 1441 / 1665 / 1831 Hz for (0, 1), (1, 1), (2, 1), (0, 2), (3, 1), (1, 2).

- Python: `~/espi_live/.venv/bin/python` (the worktree has no `.venv`). Single modules run with `PYTHONPATH=src:tests`.
- Full suite: `PYTHONPATH=src:tests QT_QPA_PLATFORM=offscreen ~/espi_live/.venv/bin/python -m coverage run -m unittest discover -s tests && ~/espi_live/.venv/bin/python -m coverage report` must pass at **100 % line coverage**, with no `# pragma: no cover`.
- After each edit: `~/espi_live/.venv/bin/python -m py_compile <changed files>`.
- Every function and method has a docstring in the `Args: ... / Returns: ...` format.
- `simulation.py`, `daq.py`, `engine.py`, `channels.py` and `profiles.py` import no Qt and no PySpin. NumPy and OpenCV only; there's no SciPy.
- Defaults stay: exposure 8000 µs, gain 0 dB, 30 fps. Recording behaviour is unchanged.
- Dummy indicators stay: the `DUMMY MODE ACTIVE` watermark (`processing.watermark_demo`, applied in `engine.present`), the red banner, the title, the badge and the `dummy_` prefix.
- Tests never open a camera, touch the network, write to the real `output/` folder or block on a dialog.
- Constants (spec §5; the plate values in the first version are superseded by the drum amendment above): λ = 632.8 nm; K = 4π/λ rad/nm; plate = centred 88 % × 88 % of the frame, simply supported, modes 1 ≤ m, n ≤ 15, physical aspect 1.5 for the frequencies whatever the frame size; f₁₁ = 520 Hz (so f₂₁ = 1000 Hz); Q = 50; drive point (0.37, 0.41) in plate coordinates; piezo 1500 nm per DAQ volt; slow cycle 12 s (5 s raised-cosine ramp up to 8 fringes = 8·λ/2 nm, 1 s hold, 5 s ramp down, 1 s still); built-in vibration = mode (2, 1) at 1000 Hz with peak K·|Z| = 16 rad; Follow-DAQ compliance set so 0.1 V sine at f₁₁ gives a peak K·|Z| of 16 rad; square TTL = fundamental (4/π)·2.5 V; speckle ≈ 3 px; spot centre ≈ 64 grey levels at 8000 µs and 0 dB, under 1 % of plate pixels at 255.
- Accuracy (spec §5.5): a simulated frame before rounding is within 1 grey level of a brute-force time integral.
- Budget (spec §5.7): ≤ 8 ms per frame at 960×640 on the dev Mac, ≤ 150 ms per table rebuild.
- Commits: AGENTS.md says "Don't commit unless asked". Run the commit steps only if the user has asked for commits or a PR; otherwise leave the work uncommitted and skip them. Never push from a branch with pre-split history (`git log origin/main..HEAD` must list only this work).

## Review Focus

1. **Exposure slider dragged in Follow DAQ or Vibration** (n changes on every value): frames keep coming at the requested rate. The table is rebuilt at most once per frame, and only when n or u_max really changed. Test in Task 4.
2. **Drive frequency where n = f·Tₑ < 0.25 or u_max is huge** (e.g. 10 V at resonance): no crash and no NaN. F is still within its accuracy where the table covers it, and u beyond the table's range is clamped to its last entry with one log warning. Test in Task 1 (bounds) and Task 4 (no NaN).
3. **DAQ fails or the window closes while the DAQ dialog is hidden**: the demo object stops vibrating (`Drive()`), because the dialog's timer stops on hide. Test in Task 7.
4. **Demo → live switch while Follow DAQ is selected**: the live engine ignores `drive`, the combo hides, and the live camera's profile doesn't gain the demo's object (its own default). Test in Task 6 (engine ignores) and Task 7 (hidden; profile).
5. **Several demo cameras** (`--demo-cameras 2`): different speckle per label, one DAQ drive reaching both, each with its own object choice. Test in Task 2 (seeds) and Task 6 (channels).

---

### Task 1: Exact exposure average table (`FringeTable`)

**Files:**
- Create: `src/espi_live/simulation.py` (module docstring, the §5 constants and `FringeTable` only)
- Test: `tests/test_simulation.py`

**Interfaces:**
- Produces: `simulation.K_PER_NM: float`; `class FringeTable(u_max: float, cycles: float)`, with attributes `u_max` and `cycles` and the method `values(u: ndarray, beta: ndarray) -> ndarray[complex64]` (same shape as u). It is F(u, β; n) = (1/Tₑ)·∫ exp(i·u·sin(ωτ + β)) dτ over n = `cycles` periods, for 0 ≤ u ≤ u_max (larger u clamped), β any real (taken mod 2π).
- Algorithm (spec §5.5): u grid × θ grid; FFT of `exp(1j*u*sin(theta))` along θ gives J_k(u); multiply by `exp(1j*pi*k*n) * sinc(pi*k*n)` (`np.sinc` is normalised: use `np.sinc(k*n)`); inverse FFT gives F on the β grid. The θ/β grid needs ≥ 2·(u_max + 32) points against aliasing. Short exposures (n ≪ 1, where F ≈ e^(iu·sin β) varies at rate u in β) need it much finer for interpolation: with linear interpolation, about 40·u_max points. The u grid can be coarse (|∂F/∂u| ≤ 1; a linear step of about 0.15 rad suffices). The builder chooses the grid sizes and the interpolation to meet the tolerances below.

- [ ] **Step 1: Write the failing tests** in `tests/test_simulation.py`, class `FringeTableTests`. A helper `brute(u, beta, n, samples=20000)` returns `np.exp(1j*u*np.sin(2*np.pi*n*s + beta)).mean()` with `s = (np.arange(samples) + 0.5) / samples`, and `j0(u)` = `np.cos(u*np.sin(t)).mean()` over `t = (np.arange(20000) + 0.5) * np.pi / 20000`.
  - `test_whole_cycles_give_j0`: for `FringeTable(30, 8)`, `values` at u = [0, 2.404826, 5.520078, 10, 16, 29.9] and any β matches `j0(u)` within 3e-3, and the two zeros are within 3e-3 of 0.
  - `test_fractional_cycles_match_brute_force`: for n in (0.01, 0.25, 1.0, 1.3, 8.7) and a grid of u in [0, 30] by β in [0, 2π), `abs(values - brute) <= 3e-3`.
  - `test_beta_is_periodic`: `values(u, b)` equals `values(u, b + 2π)` and `values(u, b - 6π)` within 1e-6.
  - `test_zero_amplitude_is_one`: `values(0, β) == 1` within 1e-6 for n = 0.3.
  - `test_u_beyond_range_is_clamped`: `values([u_max * 2], [0])` equals `values([u_max], [0])` and is finite.
- [ ] **Step 2: Run** `PYTHONPATH=src:tests ~/espi_live/.venv/bin/python -m unittest test_simulation -v`. Expected: an ImportError on `espi_live.simulation`.
- [ ] **Step 3: Implement** the constants block and `FringeTable` in `simulation.py`.
- [ ] **Step 4: Run** the same command. Expected: all `FringeTableTests` pass. Also time `FringeTable(60, 8.7)` construction by hand (≤ 150 ms) and note it in the task report.
- [ ] **Step 5: Commit** (only if commits were asked for): `git add src/espi_live/simulation.py tests/test_simulation.py && git commit -m "Add the exact exposure-average fringe table for the ESPI simulation"`

### Task 2: Scene, plate, sensor; Still and Slow deformation

**Files:**
- Modify: `src/espi_live/simulation.py`
- Modify: `src/espi_live/processing.py` (delete `DemoSource`; drop the `time` import if unused; keep `watermark_demo`)
- Modify: `tests/test_processing.py` (move `DemoTests.test_demo_frames_have_requested_shape` to `tests/test_simulation.py`; the watermark tests stay)
- Test: `tests/test_simulation.py`

**Interfaces:**
- Consumes: Task 1 constants and `FringeTable` (not yet used for Still or deformation).
- Produces:
  - `SIMULATED_OBJECTS = ("Still", "Slow deformation", "Vibration", "Follow DAQ")` and `DEFAULT_OBJECT = "Slow deformation"`.
  - `deformation_nm(t_s: float) -> float`: the 12 s cycle (0 at t = 0, rising; 8·λ/2 from 5 to 6 s; 0 from 11 to 12 s; periodic).
  - `class DemoSource(width=960, height=640, label=None, log=None)` (`log` is an AppLog-like object with `warning`, or None), with attributes:
    - `width`, `height`;
    - `object` (the current `SIMULATED_OBJECTS` name, `DEFAULT_OBJECT` at creation);
    - `background` and `modulation` (float32 H×W, in grey levels at 8000 µs and 0 dB: B·s and M·s of spec §5.1);
    - `speckle_phase` (float32, radians);
    - `plate_mask` (bool);
    - `load_shape` (float32, peak 1, 0 outside the plate);
    - `clock_s` (start time of the next frame, 0.0 at creation).
  - Methods:
    - `set_object(name: str) -> None` raises `ValueError` for a name not in `SIMULATED_OBJECTS`; choosing the current object does nothing.
    - `next_frame(exposure_us: float, gain_db: float, fps: float) -> ndarray[uint8]` renders the exposure starting at `clock_s`, then advances `clock_s` by `1 / max(fps, 1.0)`.
  - The seed is `zlib.crc32(label.encode())` (`label=None` → `"DEMO"`). The load shape is the 15×15 point-load series Σ S_mn(drive)·S_mn(x, y)/((m²/a² + n²/1)²) (a = 1.5), evaluated separably and normalised to a peak of 1.

- [ ] **Step 1: Write the failing tests** in `tests/test_simulation.py`, class `SceneTests` (use `DemoSource(96, 64, "DEMO1")` unless the test says otherwise):
  - `test_frames_have_requested_shape`: `next_frame(8000, 0, 30)` has shape (64, 96) and dtype uint8 (this replaces the processing test).
  - `test_same_label_repeats_and_labels_differ`: two `DemoSource(96, 64, "DEMO1")` give equal first frames; `"DEMO2"` gives a different one.
  - `test_speckle_statistics`: on a `DemoSource(480, 320)`, `background.mean()` over the middle 20 % × 20 % of the frame is 64 ± 10; over the plate, the object speckle's contrast (std/mean of `(background - reference level)`, or of I_o if the source keeps it) is between 0.8 and 1.2, and the fraction of a Still frame at 255 is < 0.01.
  - `test_sensor_is_linear_and_clips`: with `set_object("Still")`, a frame at 4000 µs equals `np.clip(np.rint(I * 0.5), 0, 255)` and one at 8000 µs with 6.0206 dB equals `np.clip(np.rint(I * 2.0), 0, 255)`, where `I = background + modulation * np.cos(speckle_phase)`; each is within 1 level everywhere.
  - `test_still_frames_are_identical`: Still, 10 consecutive frames are equal.
  - `test_deformation_cycle`: `deformation_nm(0) == 0`; `deformation_nm(5.5) == deformation_nm(5.0) == 8 * 632.8 / 2`; `deformation_nm(11.5) == 0`; `deformation_nm(14.5) == deformation_nm(2.5)`; it rises strictly on (0, 5).
  - `test_deformation_never_moves_outside_the_plate`: Slow deformation, frames at clock 0 and at 5.5 s (`clock_s` set directly) are equal wherever `~plate_mask`, and differ on the plate.
  - `test_deformation_frame_matches_the_phase`: at `clock_s = 5.5`, the frame is within 1 level of `np.clip(np.rint(background + modulation * np.cos(speckle_phase + K_PER_NM * deformation_nm(5.5) * load_shape)), 0, 255)`.
  - `test_load_peaks_near_the_drive_point`: argmax of `load_shape` is within 10 % of the plate size of (0.37, 0.41) in plate coordinates.
  - `test_clock_advances_by_frame_period`: three frames at 30 fps leave `clock_s` at 0.1 within 1e-9; `fps=0` advances by 1.0.
  - `test_unknown_object_is_rejected`: `set_object("Wobble")` raises `ValueError`.
- [ ] **Step 2: Run** `PYTHONPATH=src:tests ~/espi_live/.venv/bin/python -m unittest test_simulation -v`. Expected: the new tests fail (no `DemoSource` in simulation).
- [ ] **Step 3: Implement** in `simulation.py`: the speckle field (complex Gaussian noise × circular pupil in the FFT domain, radius giving ≈ 3 px grains, normalised so ⟨|E|²⟩ = 1), the Gaussian spot (about ¼ of the peak at the plate corners), the uniform reference beam equal to the object's mean at the spot centre, the plate mask and sine bases, `load_shape`, `deformation_nm`, `set_object` and `next_frame` (Still/Slow deformation only; Vibration and Follow DAQ render as Still until Task 4). Compute `cos(speckle_phase + phase)` directly. Delete `DemoSource` from `processing.py` and update `processing`'s module docstring if it names it.
- [ ] **Step 4: Run** `test_simulation` and `test_processing`. Expected: pass. (The engine still imports `DemoSource` from processing; Task 5 fixes that, so don't run the full suite yet.)
- [ ] **Step 5: Commit** (if asked): `git add src/espi_live/simulation.py src/espi_live/processing.py tests/test_simulation.py tests/test_processing.py && git commit -m "Simulate a speckle interferogram on an ideal sensor for dummy mode"`

### Task 3: `daq.Drive` on every output plan

**Files:**
- Modify: `src/espi_live/daq.py` (`Drive`, `OutputPlan.drive`, `plan()`)
- Test: `tests/test_daq.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) class Drive` with the fields:
  - `waveform: str = ""` ("sine", "square" or "" when the generator is off or unassigned);
  - `frequency_hz: float = 0.0` (what the device really produces);
  - `amplitude_v: float = 0.0` (sine peak; 0 for a square);
  - `offset_v: float = 0.0`;
  - `piezo_mode: str = ""` (`HOLD`, `SINE` or "" when off or unassigned);
  - `piezo_v: float = 0.0` (held level or sine centre);
  - `piezo_amplitude_v: float = 0.0`;
  - `piezo_phase_deg: float = 0.0`;
  - `device: str = ""` (the `DeviceInfo.label` that applied it);
  - `simulated: bool = False`.

  The property `active -> bool` is True when either part is on. `OutputPlan` gains the last field `drive: Drive = Drive()`. `plan()` fills it from the settings and `outputs.device`.

- [ ] **Step 1: Write the failing tests** in `tests/test_daq.py`, class `DriveTests`:
  - `test_sine_generator_and_held_piezo`: with `DaqSettings(generator_on=True, generator_frequency_hz=1000, generator_amplitude_v=0.1, piezo_on=True, piezo_voltage_v=0.05)`, `plan(...).drive` is `Drive("sine", plan.frequency_hz, 0.1, 0.0, HOLD, 0.05, 0.0, 0.0, "Simulated NI DAQ", True)`.
  - `test_square_counter`: `generator_output="ctr0"` gives waveform "square", `frequency_hz == square_frequency(...)` and `amplitude_v == 0`.
  - `test_sine_piezo`: `piezo_mode=SINE, piezo_amplitude_v=0.2, piezo_phase_deg=90` copies those values.
  - `test_off_is_inactive`: default settings give `drive.active is False` and `waveform == ""`; `Drive().active` is False.
  - `test_real_device_label`: with `DeviceOutputs(DeviceInfo("Dev1", "USB-6251"))`, `drive.device` is that label and `simulated` is False (build the `DeviceInfo` as the existing tests do).
- [ ] **Step 2: Run** `PYTHONPATH=src:tests ~/espi_live/.venv/bin/python -m unittest test_daq -v`. Expected: the DriveTests fail.
- [ ] **Step 3: Implement** `Drive` and fill it in `plan()`.
- [ ] **Step 4: Run** `test_daq`, `test_qt_daq`. Expected: pass.
- [ ] **Step 5: Commit** (if asked): `git commit -am "Describe what the DAQ outputs produce on each output plan"`

### Task 4: Vibration and Follow DAQ in the source

**Files:**
- Modify: `src/espi_live/simulation.py`
- Test: `tests/test_simulation.py`

**Interfaces:**
- Consumes: `FringeTable`, `daq.Drive`, `daq.HOLD`, `daq.SINE`.
- Produces:
  - `DemoSource.set_drive(drive: daq.Drive) -> None`.
  - `DemoSource.harmonic_path_nm() -> tuple[float, ndarray | None]`: `(frequency_hz, Z)` with Z complex64 H×W in nm (W − P, spec §5.4), or `(0.0, None)` when nothing vibrates.
  - `DemoSource.static_piezo_nm() -> float`: 1500·held level, or 1500·sine centre (0 when the piezo is off).
  - `mode_frequency_hz(m: int, n: int) -> float`.
  - `object_note(name: str, drive: daq.Drive) -> str` (the text under the combo, spec §6):
    - Still: "Nothing moves";
    - Slow deformation: mentions "12 s";
    - Vibration: "mode (2, 1)" and "1000 Hz";
    - Follow DAQ with an active drive: the device label, e.g. "1000 Hz sine, 0.10 V" or "square", and the piezo text "piezo holds 0.05 V" or "piezo sine 0.20 V at 90°";
    - Follow DAQ with no drive: contains "no DAQ drive" and "Devices → NI DAQ…".
    - A held piezo level is listed in every object.
- Rules:
  - The held piezo level acts in every object; piezo sine acts only in Follow DAQ (its centre acts in all).
  - Vibration: Z = the (2, 1) mode shape scaled so max K·|Z| = 16, at 1000 Hz.
  - Follow DAQ: W = A·Σ C_mn·H_mn(f)·S_mn, with H_mn = 1/(1 − (f/f_mn)² + i·f/(Q·f_mn)) and C_mn = C₀·S_mn(drive)/(f_mn/f₁₁)²; A = `amplitude_v` (sine) or (4/π)·2.5 (square); P = 1500·`piezo_amplitude_v`·e^(i·phase).
  - Frame: B + M·Re(exp(i(ψ + K·d − K·p_static))·F(K|Z|, arg Z + 2π·f·t₀; n = f·Tₑ)).
  - Rebuild the `FringeTable` only when n or the needed u_max change (cache it). Rebuild W only when the drive or object changes.

- [ ] **Step 1: Write the failing tests** in `tests/test_simulation.py`, class `VibrationTests` (frames 96×64 unless noted; `brute_frame(source, exposure_us, t0)` averages 4000 instants of `background + modulation*cos(speckle_phase - K*p_static + K*Im(Z*exp(2j*pi*f*t)))` over t ∈ [t₀, t₀ + Tₑ], times `exposure_us / 8000`, using `Z` from `harmonic_path_nm()` and `p_static` from `static_piezo_nm()`):
  - `test_mode_frequencies`: `mode_frequency_hz(1, 1) == 520`, `(2, 1) == 1000` within 1e-9, `(1, 2) == 520 * (1/2.25 + 4) / (1/2.25 + 1)`.
  - `test_vibration_has_a_still_nodal_line`: Vibration, `harmonic_path_nm()` gives f = 1000, `max(K*|Z|) == 16` within 1e-3, and |Z| < 1e-3·max on the plate's vertical centre column and outside the plate.
  - `test_vibration_matches_brute_force_whole_and_fractional`: Vibration, at 8000 µs and 8700 µs and t₀ ∈ {0, 0.0123}, the frame is within 1 level of `np.clip(np.rint(brute_frame(...)), 0, 255)`.
  - `test_whole_cycles_do_not_flicker`: Vibration at 8000 µs, 30 fps, consecutive frames are equal. At 8700 µs, they differ.
  - `test_follow_daq_resonance`: Follow DAQ with `Drive("sine", 520.0, 0.1)` gives max K·|Z| = 16 within 2 %; at 700 Hz it is below 30 % of that; `Drive()` gives `(0.0, None)` and frames equal to Still.
  - `test_follow_daq_square_uses_the_fundamental`: `Drive("square", 520.0)` equals a sine of amplitude (4/π)·2.5 V.
  - `test_piezo_hold_shifts_the_phase_in_every_object`: for Still and Vibration, `Drive(piezo_mode=HOLD, piezo_v=0.05)` gives `static_piezo_nm() == 75.0`, and the Still frame equals the Task 2 formula with phase −K·75.
  - `test_piezo_sine_only_in_follow_daq`: `Drive("sine", 1000, 0.1, piezo_mode=SINE, piezo_v=0.01, piezo_amplitude_v=0.02, piezo_phase_deg=30)`: in Vibration, Z equals the built-in Z (no P) and the static piezo is 15 nm; in Follow DAQ, Z = W − 30·e^(iπ/6) (1500·0.02 = 30 nm).
  - `test_reference_modulation_cancels_where_matched`: Follow DAQ; choose piezo amplitude and phase equal to W at one plate pixel; |Z| there < 1e-3 nm.
  - `test_drag_rebuilds_table_once_per_change`: patch `simulation.FringeTable` with a counting wrapper; frames at exposures 8000, 8000, 8100, 8100 build it twice.
  - `test_strong_drive_stays_finite`: Follow DAQ at 520 Hz, 10 V: the frame has no NaN and is uint8; the warning about u beyond the table range is logged at most once (the source takes an optional `log=None` and writes to `log.warning`).
  - `test_object_note_texts`: the cases in Interfaces above.
- [ ] **Step 2: Run** `test_simulation`. Expected: the VibrationTests fail.
- [ ] **Step 3: Implement** `set_drive`, `harmonic_path_nm`, `static_piezo_nm`, `mode_frequency_hz`, `object_note`, the cached W and table, and the harmonic branch of `next_frame`.
- [ ] **Step 4: Run** `test_simulation`. Expected: pass. Time `next_frame` at 960×640 for each object (Vibration and Follow DAQ after the first frame, at a whole and a fractional n) and report the numbers. The target is ≤ 8 ms on the dev Mac. A fractional-n frame needs a 2-D interpolated lookup and may run over. If it does, report the number and keep the accuracy (the user put accuracy first) rather than coarsening the table; anything under about 15 ms still leaves the engine most of the 33 ms frame.
- [ ] **Step 5: Commit** (if asked): `git commit -am "Simulate time-averaged vibration and the DAQ drive and piezo"`

### Task 5: Engine and profiles

**Files:**
- Modify: `src/espi_live/engine.py` (imports, `Settings.demo_object`, `use_demo`, the demo branch of `_run`, `_handle_actions`, the `request` docstring)
- Modify: `src/espi_live/profiles.py` (`CHOICES["demo_object"] = SIMULATED_OBJECTS`)
- Modify: tests that replace `next_frame` (`tests/test_engine.py`, `tests/test_reference.py`, `tests/test_engine_camera.py::test_demo_schedule_restarts_when_processing_falls_behind`): the third positional argument is now `fps`; rename the parameter, and keep the behaviour.
- Test: `tests/test_engine.py`, `tests/test_profiles.py`

**Interfaces:**
- Consumes: `simulation.DemoSource`, `DEFAULT_OBJECT`, `SIMULATED_OBJECTS`; `daq.Drive`.
- Produces:
  - `Settings.demo_object: str = DEFAULT_OBJECT`, placed after `raw_difference`.
  - `Engine.request("drive", drive=Drive)`: in demo mode it calls `demo_source.set_drive(drive)`; otherwise it is ignored.
  - Each demo frame calls `demo_source.set_object(s.demo_object)`, then `next_frame(s.exposure_us, s.gain_db, s.fps)`. It logs `"Simulated object: <name>"` when the name changes.
  - `use_demo(label)` builds `DemoSource(label=label, log=self.log)`. Its status camera text is "DUMMY MODE ACTIVE — simulated ESPI signal, ideal sensor; exposure, gain and FPS are simulated", and its log line is "DUMMY MODE ACTIVE: simulated ESPI signal (ideal sensor); no camera acquired."

- [ ] **Step 1: Write the failing tests:**
  - In `tests/test_engine.py`, class `SimulatedObjectEngineTests`:
    - `test_demo_object_reaches_the_source` sets `update_settings(demo_object="Still")` and waits until `engine.demo_source.object == "Still"` and the log has "Simulated object: Still".
    - `test_drive_action_reaches_a_demo_source` sends `request("drive", drive=Drive("sine", 1000.0, 0.1))` with object Follow DAQ, and waits until `demo_source.harmonic_path_nm()[0] == 1000.0`.
    - `test_drive_is_ignored_without_demo` uses the fake camera engine from `tests/test_engine_camera.py`'s `CameraEngineCase`: the request does nothing and raises nothing.
  - In `tests/test_profiles.py`: `test_demo_object_choices` checks that `CHOICES["demo_object"] == SIMULATED_OBJECTS` and that a stored `"Wobble"` loads as `DEFAULT_OBJECT` (the existing every-text-has-choices test covers the rest).
- [ ] **Step 2: Run** `test_engine`, `test_profiles`. Expected: the new tests fail.
- [ ] **Step 3: Implement** as in Interfaces. Import `DemoSource` from `espi_live.simulation` (so `engine.DemoSource` stays patchable), and drop it from the processing import.
- [ ] **Step 4: Run** the **full suite with coverage** (Global Constraints). Expected: every test passes. Fix any existing test that relied on the old pattern (for example a nonzero Previous difference in the first frames: set `demo_object` or `clock_s` explicitly in that test, and don't change the simulation to suit it). Coverage may be < 100 % only in `qt_app.py` lines not yet written; anything else is fixed now.
- [ ] **Step 5: Commit** (if asked): `git commit -am "Feed the engine's demo source the simulated object and the DAQ drive"`

### Task 6: Channels forward the drive

**Files:**
- Modify: `src/espi_live/channels.py` (`Channel.__init__`, `start`, the open-now path around line 247, and a new `set_drive`)
- Test: `tests/test_channels.py`

**Interfaces:**
- Consumes: `Engine.request("drive", drive=...)`; `daq.Drive`.
- Produces:
  - `Channel.drive: daq.Drive` (default `Drive()`).
  - `Channel.set_drive(drive: daq.Drive) -> None` stores it and, when the channel is a demo with an engine, sends it.
  - Every new demo engine (`start`, open-now) gets `request("drive", drive=self.drive)` before `engine.start()`.

- [ ] **Step 1: Write the failing tests** in `tests/test_channels.py`, class `DriveForwardingTests` (use a `mock.Mock` engine where possible):
  - `test_set_drive_reaches_a_running_demo_engine`.
  - `test_new_demo_engine_gets_the_current_drive`: `set_drive` while stopped, then `start()`; the new engine's first request is the drive.
  - `test_live_channel_keeps_but_does_not_send`.
  - `test_two_demo_channels_each_get_it`.
- [ ] **Step 2: Run** `test_channels`. Expected: fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `test_channels`, `test_engine`. Expected: pass.
- [ ] **Step 5: Commit** (if asked): `git commit -am "Pass the DAQ drive to every demo channel"`

### Task 7: Simulated object control and DAQ hand-off in the window

**Files:**
- Modify: `src/espi_live/qt_app.py`:
  - Camera section, after `add(self.fps, "Requested FPS")`;
  - `_load_settings` combos;
  - `_sync_source_controls`;
  - `DaqDialog.poll` and `connected`;
  - `MainWindow.__init__` (`self.daq_drive = daq.Drive()`);
  - a new `set_daq_drive`;
  - `_poll_status` (poll a hidden DaqDialog).
- Test: `tests/test_qt_app.py` (control, profile), `tests/test_qt_daq.py` (hand-off)

**Interfaces:**
- Consumes: `SIMULATED_OBJECTS`, `object_note`, `daq.Drive`, `Channel.set_drive`.
- Produces:
  - `self.simulated_box` is a full-width `QWidget` added with `add(self.simulated_box, basic=True)`. It holds a "Simulated object" `QLabel`, `self.simulated_object` (a `QComboBox` with `SIMULATED_OBJECTS`, tooltip naming every item) and `self.simulated_note` (a word-wrapped `QLabel` with a tooltip).
  - The combo writes `self.engine.update_settings(demo_object=text)` and refreshes the note.
  - `_load_settings` sets it with signals blocked.
  - `_sync_source_controls` shows `simulated_box` only when `self.current.demo`.
  - `MainWindow.set_daq_drive(drive: daq.Drive) -> None` stores `self.daq_drive`, calls `channel.set_drive(drive)` on every channel, and refreshes the note.
  - `DaqDialog.poll` calls `self.window.set_daq_drive(payload.drive)` on "applied", and `set_daq_drive(daq.Drive())` on "zeroed" and "failed". `connected(None)` (disconnect) also calls it with `daq.Drive()`.
  - `_poll_status` calls `self.daq_dialog.poll()` when the dialog exists and is hidden.
  - On close, the DAQ shutdown path needs nothing new: the engines stop.

- [ ] **Step 1: Write the failing tests:**
  - In `tests/test_qt_app.py`, a new class `SimulatedObjectTests(WindowTestCase)`:
    - `test_control_shows_for_demo_and_hides_for_live` (use the existing multi-camera or live-switch fixtures).
    - `test_switch_live_hides_control_and_keeps_live_default`: in a demo window with Follow DAQ selected, a successful `switched_live` (as the existing switch-live tests drive it) hides `simulated_box`, and the live channel's settings have `demo_object == DEFAULT_OBJECT` (its own default or saved profile, never the demo's value).
    - `test_choice_writes_only_the_displayed_engine` (two demo channels, as in `MultiCameraTests`).
    - `test_choice_is_saved_in_the_demo_profile_and_restored`, in `ProfileWindowTests` style with `FakeSettings`.
    - `test_note_follows_object_and_drive`: the note equals `object_note(text, window.daq_drive)` after a change and after `set_daq_drive`.
    - `test_shown_in_simple_ui`.
  - The existing `test_controls_dock_rejects_a_control_without_a_tooltip` must still pass.
  - In `tests/test_qt_daq.py`, a new class `DriveHandOffTests(DaqWindowTestCase)`:
    - `test_applied_drive_reaches_the_window_and_channels` (connect Simulated, apply a sine at 1000 Hz; wait for `window.daq_drive.frequency_hz == 1000.0` within the device's rounding; `window.current.drive` is the same object);
    - `test_zero_and_failure_switch_the_drive_off`;
    - `test_hidden_dialog_still_hands_off` (hide the dialog, inject a "failed" event into `controller.events`, call `window._poll_status()`, and check `daq_drive` is `Drive()`).
- [ ] **Step 2: Run** `QT_QPA_PLATFORM=offscreen PYTHONPATH=src:tests ~/espi_live/.venv/bin/python -m unittest test_qt_app.SimulatedObjectTests test_qt_daq -v`. Expected: the new tests fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the **full suite with coverage**. Expected: everything passes, 100 %.
- [ ] **Step 5: Commit** (if asked): `git commit -am "Add the Simulated object control and hand the DAQ drive to demo cameras"`

### Task 8: Documentation

**Files:**
- Modify:
  - `README.md` (the dummy-mode text);
  - `src/espi_live/user_guide.html` (the `#start-demo` section: the simulation, the four objects, the drum's mode frequencies 628 / 1000 / 1340 / 1441 / 1665 / 1831 Hz for (0,1), (1,1), (2,1), (0,2), (3,1), (1,2), the piezo 1500 nm/V and 0.05 V ≈ 85°, the by-eye checks of spec §7, and the Simulated object control in the controls reference);
  - `AGENTS.md` (the layout and architecture rows: `simulation.py`; `processing.py` no longer lists `DemoSource`; the `daq.py` row's `Drive`; the Demo-mode paragraph; the Engine-threading list gains `drive`; the Frame-flow `DemoSource.next_frame` wording);
  - `CHANGELOG.md` (a new top entry dated 2026-11-21);
  - `docs/notes/ni-usb-6251-integration.md` (the "(future)" line now points to `simulation.py`).
- Test: `tests/test_guide.py`

- [ ] **Step 1: Write the failing test:** in `tests/test_guide.py`, extend `test_guide_covers_every_control_and_operation`'s phrase list with "Simulated object", "Slow deformation", "Follow DAQ" and "ideal sensor".
- [ ] **Step 2: Run** `test_guide`. Expected: fail.
- [ ] **Step 3: Write the docs.**
- [ ] **Step 4: Run** `test_guide` and the full suite with coverage. Expected: pass, 100 %.
- [ ] **Step 5: Commit** (if asked): `git commit -am "Document the simulated ESPI signal in dummy mode"`

### Task 9: Verify by eye and hand over

**Files:** none in the repo. Screenshots go to the scratchpad, and to `pr-screenshots/pr-<number>/` only if a PR is opened.

- [ ] **Step 1:** Run `PYTHONPATH=src ~/espi_live/.venv/bin/python -m espi_live --check`. Expected: exit 0.
- [ ] **Step 2:** Run an offscreen script (a scratchpad file, not in the repo) that builds a demo `MainWindow` with `QT_QPA_PLATFORM=offscreen` and saves `widget.grab()` PNGs. Cover:
  - the control in each object (including Simple UI);
  - Raw, Reference against a Still reference, and Previous for each object;
  - Vibration at 8000 µs and at 8700 µs;
  - Follow DAQ at 628 Hz and 1340 Hz with the DAQ window;
  - a live (fake) camera with the control hidden.
  
  Look at each image and check it against spec §7.
- [ ] **Step 3:** Report the per-frame timings from Tasks 1 and 4 and the full-suite result (test count, coverage).
- [ ] **Step 4 (only if a PR was asked for):** push the branch after checking `git log origin/main..HEAD` and `git merge-tree --write-tree HEAD origin/main`. Then open the PR with `gh pr create`, push the screenshots to the `pr-screenshots` branch and embed them with one-line captions.
