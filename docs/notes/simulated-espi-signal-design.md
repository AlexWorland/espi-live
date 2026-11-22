# Simulated ESPI signal for dummy mode: design

Status: draft for approval (2026-11-21). Branch `feat/simulated-espi-signal`, based on public `origin/main` (00e37e9).

## 1. Intent

Dummy mode (`--demo`, and the "use demo mode?" fallback) currently shows a test pattern: a gradient, a checkerboard, a bouncing circle and a caption (`processing.DemoSource`). It exercises the plumbing, but none of the ESPI processing means anything on it.

Replace it with a **simulated ESPI signal**: what an ideal camera would record when looking at a speckle interferometer whose specimen deforms or vibrates. The real processing chain (Previous/Reference difference, averaging, fixed gain or stretch, LabVIEW flow, noise floor, peak hold and the rest) then runs on frames whose correct outcome is known from the physics. The sensor is **theoretically perfectly clean**: no shot, read, dark or fixed-pattern noise. What you see is the optics, the 8-bit rounding and clipping at 255, and nothing else.

The user checks the results **by eye in the app**. There is no ground-truth overlay and no pipeline-versus-truth test suite (see non-goals).

## 2. Goals, non-goals, anti-goals

**Goals**

- A demo frame is the exposure-averaged intensity of a speckle interferogram: an object speckle field plus a smooth reference beam, with an optical phase that follows the specimen's out-of-plane motion.
- Four object behaviours, chosen per demo camera: **Still**, **Slow deformation**, **Vibration** (built-in drive) and **Follow DAQ** (the NI DAQ window's generator excites the specimen).
- The DAQ piezo moves the simulated reference mirror. Held levels step the reference phase in every behaviour. The piezo's sine mode modulates the reference in Follow DAQ.
- Exposure time is physical. Brightness is proportional to exposure × gain, and the time average over the exposure is exact for any drive frequency: few-cycle exposures flicker and walk in phase exactly as the DAQ notes predict (docs/notes/ni-usb-6251-integration.md §5).
- Each demo camera has its own speckle pattern.
- At least 30 fps at the default 960×640 on the development Mac, leaving the engine its processing time.

**Non-goals**

- Sensor noise of any kind, or a noise slider. "Perfectly clean" is the point.
- Ground truth in the UI (an "Expected" panel, expected pixel values) and automated tests that compare the processed output with the analytic answer. The simulation's own unit tests are still required (100 % coverage rule).
- In-plane motion and speckle decorrelation, air turbulence, laser phase drift, rolling shutter, interlacing, colour (iPhone Bayer), stroboscopic illumination, ring-up and ring-down transients, and square-wave harmonics.
- Controls for amplitude or speed beyond the object choice. The DAQ window already sets the frequency and amplitude in Follow DAQ.
- Command-line options for the simulated object.

**Anti-goals** (failures even if the simulation works)

- Demo data that could pass for camera data. The frames now look realistic, so every existing dummy-mode indicator stays: the `DUMMY MODE ACTIVE` watermark burned into every published frame, the red banner, the title, the status badge and the `dummy_` file prefix.
- Any change to live-camera code paths, or to what the DAQ drives. Forwarding the DAQ state to the simulation only reads it.
- Demo mode below the requested frame rate on the dev machine because of the simulation.

## 3. Approaches considered

1. **Exact exposure average through lookup tables** (chosen). The sensor records ⟨exp(iΔφ(t))⟩ over the exposure. For a sinusoidal motion that average depends only on the local phase amplitude u, the phase β at exposure start, and the number of cycles per exposure n = f·Tₑ. Tabulate it whenever the drive or the exposure changes, then look it up per pixel each frame. It's exact for any n, and piezo steps and sine modulation fall out of the same formula. Measured cost of the core operations: about 3.5 ms per frame at 960×640, plus about 40 ms once per drive change.
2. **Long-exposure shortcut.** cos(K·Δd) for deformation and J0(K·|a|) for vibration, as if every exposure held infinitely many cycles. Simpler and about 2 ms per frame, but exposure time and drive frequency would stop interacting: a 125 Hz drive at 8 ms (one cycle) would look perfect, so DAQ-timing mistakes would be invisible in the demo.
3. **Brute-force sub-exposures.** Average N rendered instants per exposure. Obviously correct, but large amplitudes need hundreds of instants per frame to avoid aliasing, far too slow at 30 fps (16 instants already take 11 ms here).

The user handed this choice over ("whatever is easiest and most accurate"). Approach 1 is the accurate one, and its extra work is contained in one table builder.

## 4. Decided with the user

- All three behaviours: slow deformation, vibration, and DAQ-driven, plus a Still state.
- A demo-only **Simulated object** control (Still / Slow deformation / Vibration / Follow DAQ), saved with that demo camera's profile. Vibration has a built-in drive, so no DAQ setup is needed for a quick test.
- The DAQ piezo moves the reference mirror in every case.
- Checking is by eye in the app.

## 5. The model

All of this lives in a new GUI-free module, `src/espi_live/simulation.py`, with no PySpin and no Qt (the processing-module rule).

### 5.1 Optics and sensor

- Wavelength λ = 632.8 nm (HeNe). Illumination and viewing are along the surface normal, so the phase sensitivity is K = 4π/λ for the specimen, and the same for the reflecting piezo mirror.
- **Object beam:** a fully developed speckle field E = low-pass-filtered complex Gaussian noise (a circular pupil in the FFT domain), sized for a mean speckle diameter of about 3 px and normalised to ⟨|E|²⟩ = 1. I_o = |E|² · G(x, y), where G is a centred Gaussian laser spot that falls to half its peak at the drum's rim. ψ = arg E. The seed is a stable hash of the camera label (e.g. `zlib.crc32`, not Python's per-process `hash`), so DEMO1 and DEMO2 differ but each repeats exactly from run to run.
- **Reference beam:** smooth and uniform, I_r equal to the object's mean intensity at the spot centre (the best fringe visibility there).
- **Intensity at an instant:** I = I_o + I_r + 2√(I_o I_r) cos(ψ + K·(w − p)), where w is the specimen's out-of-plane displacement and p the piezo mirror's displacement.
- **Ideal sensor:** value = round(clip(s · (Tₑ / 8000 µs) · 10^(gain/20) · ⟨I⟩_exposure, 0, 255)), where s is set so the spot centre averages about 64 grey levels at the defaults (8000 µs, 0 dB), with under 1 % of pixels at 255. It's linear with no floor (the old demo clamped its brightness factor to 0.1–2.5): a very short exposure is honestly dark.
- Precompute once per source: B = I_o + I_r, M = 2√(I_o I_r) (float32) and ψ.

### 5.2 Specimen: a drumhead

*Changed on 2026-11-21 at the user's request ("simulate a drumhead: not the actual materials, but a circular plane"). The first version used a rectangular plate.*

- An ideal circular membrane clamped at its rim, centred in the frame, with a diameter of 88 % of the frame's shorter side. There is no material model: no tension, density or thickness. Outside the rim the mount is rigid (w = 0) but still speckled, so the rim is a visible zero-motion circle in every behaviour.
- Mode (m, n) has m nodal diameters and n nodal circles (the rim included). Its shape is J_m(j_mn·r/R)·cos(m(θ − θ_d)), and its frequency f_mn = 1000 Hz · j_mn / j₁₁, where j_mn is the n-th zero of J_m. So (1, 1) is exactly 1000 Hz, and the fundamental (0, 1) is 627.6 Hz. Then come (2, 1) 1340 Hz, (0, 2) 1441 Hz, (3, 1) 1665 Hz, (1, 2) 1831 Hz and so on: not harmonic, as on a real drum. Every mode with j_mn ≤ 30 (109 modes) has Q = 50.
- **Drive pad** at r = R/3, angle θ_d = 30° (counter-clockwise from +x, y up). A drive on that axis excites only the cos(m(θ − θ_d)) member of each degenerate pair, so the sin members are left out exactly. Each mode's static share is J_m(j_mn/3)/(N_mn·j_mn²), where N_mn = (2π if m = 0, else π)·J_{m+1}(j_mn)²/2. It is tapered by exp(−(j_mn/14)²) for the pad's size (a true point load is logarithmically singular on a membrane). At 0 Hz the mode sum is the pad's dent, which is also the Slow-deformation load. Static and dynamic behaviour are one consistent drum.
- **Bessel functions without SciPy:** one cached FFT table of e^(ix·sin t) gives J_0..J_40 on 0 ≤ x ≤ 34 (step 0.002, error < 1e-6). The zeros are read from its sign changes, searched from x = m, because below that a high order is ~1e-20 and rounding noise flips its sign.
- **Polar evaluation:** for each m, the modes' radial profiles are summed on a 2048-point radius grid, read at every pixel's radius and multiplied by cos(mθ), built by recurrence. Only pixels on the drum are computed. A full complex amplitude map costs about 48 ms (once per drive change); building a source costs 60–120 ms.

### 5.3 Behaviours

| Object | Static displacement d(x, y, t) | Harmonic motion | Piezo |
|---|---|---|---|
| Still | 0 | none | hold level |
| Slow deformation | D(t)·Ŝ_load | none | hold level |
| Vibration | 0 | drum mode (1, 1) at its resonance, 1000 Hz, peak K·a = 16 rad (5 J0 fringes) | hold level |
| Follow DAQ | 0 | generator → mode sum (5.4) | hold level, or sine (5.4) |

- **Slow deformation:** Ŝ_load is the pad's dent, normalised to a peak of 1. D(t) cycles every 12 s: a smooth (raised-cosine) ramp from 0 to 8 fringes' worth (8·λ/2 ≈ 2.53 µm) over 5 s, a 1 s hold, a 5 s ramp back down, then 1 s still. Reference view shows fringes appearing and multiplying. Previous view shows faint rate fringes (the peak rate is about 26 nm per frame at 30 fps, about 0.5 rad) and is exactly black during the hold and the still second.
- The vibration is in steady state at once. The usual workflow, as on the bench: choose Still (or switch the generator off), set the reference, then start the vibration.

### 5.4 DAQ coupling

- **Generator.** Sine on an AO: V(t) = A·sin(ωt). The drum's complex amplitude is W = A·C₀·Σ_mn (static share)_mn·H_mn(f)·shape_mn, with H_mn(f) = 1/(1 − (f/f_mn)² + i·f/(Q·f_mn)). C₀ is set so 0.1 V at the fundamental, f(0, 1), peaks at 16 rad; higher modes need more volts, as on a real drum. A square TTL wave from a counter counts as its fundamental, amplitude (4/π)·2.5 V, and the harmonics and DC are ignored. The generator's DC offset is ignored (a constant displacement; invisible in Previous view).
- **Piezo:** 1500 nm per DAQ volt (the lab's 100 nm/V stack behind a gain-15 amplifier, docs/notes/ni-usb-6251-integration.md §5). A held level V_h gives a constant reference phase K·1500·V_h; the default 0.05 V step is about 85°, close to π/2. In sine mode, p(t) = 1500·(V_c + V_a·sin(ωt + θ)): V_c is a held phase, and the harmonic part becomes the reference phasor P = 1500·V_a·e^(iθ). Piezo sine applies only in Follow DAQ (it is locked to the generator); in the other behaviours only its centre or held level acts.
- **Net harmonic path difference:** Z(x, y) = W − P (nm, complex), so Δφ_h(t) = K·|Z|·sin(ωt + arg Z). Reference modulation moves the zero-order fringe to where W = P, as Høgmoen and Løkberg describe.
- **Which DAQ:** the object follows whatever the DAQ window has applied, whether the Simulated device or real hardware. The status text says which. When nothing is connected or applied, or the outputs are zeroed, or the DAQ fails or closes, the drive is "off": no generator, piezo at 0 V.

### 5.4b Reference piezo for decorrelation (triangle)

*Added on 2026-11-21 at the user's request: "the piezo is on the reference beam and driven at a low triangle wave frequency. It's for decorrelation. The two-zone (1, 1) image should have a white line down the middle, not a dark line."*

- A demo-only checkbox, **Reference piezo (triangle, decorrelation)**: `Settings.demo_piezo`, off by default and saved per demo camera. It works in every object and adds to the DAQ piezo.
- The reference mirror moves in a triangle wave: PIEZO_TRIANGLE_HZ = 1 Hz, PIEZO_TRIANGLE_NM = 1200 nm peak to peak, so the phase spans K·1200 ≈ 23.8 rad, about 91° per frame at 30 fps. *These are my calls; the lab's generator settings were not given.*
- **Exposure average:** the reference beam is uniform, so the ramp contributes one complex number per frame, R = ⟨e^(−iφ_tri(t))⟩ over the exposure. It is exact in closed form for each linear piece (a sinc), across any number of turning points. The frame is B + M·Re(e^(i(ψ + K·d − K·p_static))·F·R). The ramp (tens of rad/s) is far slower than any vibration (kHz), so its cross terms with the k ≠ 0 vibration terms are below 1 %. A brute-force test holds the result to 1 grey level, including exposures that straddle a turning point.
- **Result:** consecutive frames differ by Δφ, so the Previous difference is 2M·|J0(K|Z|)|·|sin(Δφ/2)|·|sin(…)|. Nodes are bright (a white nodal line) and J0 zeros are dark; the still mount is bright too. Without the piezo, a steady whole-cycle vibration's Previous view is exactly black.

### 5.5 Exposure average (the core formula)

Let the exposure start at t₀ and last Tₑ, with n = f·Tₑ cycles. Everything except the harmonic term is constant within an exposure:

⟨I⟩ = B + M·Re( e^(i(ψ + K·d − K·p_static)) · F(u, β; n) ),  with u = K·|Z|, β = arg Z + ω·t₀,

F(u, β; n) = (1/Tₑ)·∫ exp(i·u·sin(ωτ + β)) dτ over the exposure = Σ_k J_k(u)·e^(ik(β + πn))·sinc(πkn)  (Jacobi–Anger).

- A whole number of cycles gives F = J0(u) exactly. Otherwise the k ≠ 0 terms (weights ≤ 1/(πkn)) make the frame depend on β, which walks by 2π·f/fps per frame: the flicker and phantom fringes of the DAQ notes.
- **Tabulate F per (drive, exposure) change:** for each u on a grid, sample e^(iu·sin θ) on a θ grid, FFT to get the J_k(u) (exact up to aliasing; the θ grid needs more than 2(u_max + 30) points), multiply by e^(iπkn)·sinc(πkn), inverse FFT, and read F(u, β) on the β grid. There's no Bessel library: numpy only.
- **Per pixel, per drive change:** u(x, y) and arg Z(x, y) as table coordinates. **Per frame:** shift β by ω·t₀, look up F, compute cos and sin of (ψ + K·d − K·p_static) directly (cheap, and exact), and combine.
- **Accuracy target:** the result before rounding is within 1 grey level of a brute-force time integral (tens of thousands of samples) everywhere, for any n ≥ 0.25 and u up to the table's range. The grid sizes and interpolation (nearest or linear in each axis) are the builder's choice, constrained by this target and the frame budget.
- Without harmonic motion, F = 1 and the table is skipped.
- **Deliberately not integrated:** the slow deformation's change during one exposure. At most 6 nm, a 0.13 rad ramp, which changes the modulation by under 0.1 % (below one grey level).

### 5.6 Clock

The simulation keeps its own sensor clock: frame k starts at t₀ = k / fps (the requested rate), independent of the engine's wall-clock jitter. A free-running camera's crystal clock is what makes the real phase walk regular, which sleep jitter would randomise. The slow-deformation cycle runs on the same clock (it slows if the engine falls behind, which is fine for simulated time). `next_frame` therefore needs the frame rate. Changing its signature from `(exposure_us, gain_db, now)` is acceptable; the tests that replace it are updated with it.

### 5.7 Performance budget

On the dev Mac at 960×640: at most 8 ms per frame in every behaviour, and at most 150 ms for a table rebuild (on the engine worker, only when the drive, object or exposure changes; a single slow frame is acceptable then). The per-source setup (speckle FFT, bases) happens once in `use_demo`.

## 6. Integration

- **`simulation.py`:** the source class (it replaces `processing.DemoSource`; keeping the `DemoSource` name and having `processing` stop defining it is the builder's call), the constants above, `SIMULATED_OBJECTS = ("Still", "Slow deformation", "Vibration", "Follow DAQ")` and the table builder. `watermark_demo` stays in `processing.py`.
- **`daq.py`:** a small frozen `Drive` dataclass describing what the outputs produce (generator waveform "sine", "square" or none, its actual frequency, amplitude, offset; piezo mode, level or centre, amplitude, phase). `plan()` fills an `OutputPlan.drive` field, so the existing `"applied"` event carries it. `Drive()` with no arguments means off.
- **Engine:** `Settings.demo_object: str = "Slow deformation"` (ignored by live sources; persisted like every other field; `profiles.CHOICES` gains it). There's a new `request("drive", drive=...)` action: in demo mode the worker hands it to the source, otherwise it does nothing. The worker gives the source the current object each frame (cheap when unchanged).
- **Window:** `DaqDialog.poll` reports the latest `Drive` to the window on "applied", and `Drive()` on "zeroed", "failed", "closed" and disconnect. The window keeps it and sends it to every demo channel's engine, including demo engines started later and after a channel restart. One DAQ serves every demo camera, and each camera's own object choice decides whether it listens. Forwarding must not depend on the DAQ window being open: if its poll timer stops while it is hidden, the window keeps draining the DAQ events (*builder check; my call*).
- **Control:** a **Simulated object** combo in the Camera section, shown only while the displayed channel is a demo (also in the Simple UI). It writes only `self.engine`, is shown by `_load_settings` without sending, and has a tooltip per item.
- **What the simulation is following** is shown in a note under the combo, not the status bar (*builder check; my call*). The engine's camera line is hidden in dummy mode on purpose (the red banner replaces it), and the status-bar camera readout is fixed-width. In Follow DAQ the window composes the note from the `Drive` it holds: which device ("Simulated NI DAQ" or the real one), the generator ("1000 Hz sine, 0.10 V") and the piezo ("holds 0.05 V"), or "no DAQ drive — open Devices → NI DAQ… and choose the Simulated device". For Vibration the note names the built-in drive (drum mode (1, 1), 1000 Hz). For Slow deformation it gives the 12 s cycle, and for Still it says nothing moves. A held piezo level is listed in every behaviour.
- **Unchanged:** the watermark and every dummy indicator, **Switch to live capture** (live cameras ignore `demo_object`; its control hides), recording, saving, Freeze, Raw difference, the LabVIEW flow (it doesn't touch `demo_object`), and presets (a preset saved from a demo camera carries `demo_object`, which is harmless on a live one).

## 7. What a user should see (the by-eye checks)

- **Raw:** a speckled laser spot on the drumhead (it falls to half at the rim). Vibrating regions show lower speckle contrast; nodal lines keep full contrast.
- **Still:** Previous view is exactly black, and Measure noise floor reads 0. A piezo step (the DAQ window's Step) flashes a uniform fringe-free change in Previous view and a uniform offset against a reference.
- **Slow deformation:** Reference view shows closed rings around the pad (at R/3, 30°), growing to 8 and back, always black outside the drum. A membrane's dent is logarithmic, so the rings crowd toward the pad and spread toward the rim (the reverse of a plate).
- **Vibration:** against a still reference, drum mode (1, 1): 1 − J0 bands in two lobes either side of a dark nodal diameter at 120° (perpendicular to the drive axis). Previous view stays black at 8000 µs (8 whole cycles) and flickers at, say, 8700 µs (8.7 cycles).
- **Reference piezo on, Vibration, Previous view:** |J0| fringes with a white nodal diameter between the two lobes; a rolling Average N smooths the speckle.
- **Follow DAQ:** sweeping the generator through 628 Hz (0, 1: rings), 1000 Hz (1, 1: one nodal diameter), 1340 Hz (2, 1: a nodal cross), 1441 Hz (0, 2: a nodal circle), 1665 Hz (3, 1: six lobes) … makes successive drum modes appear. A whole number of cycles per exposure keeps them steady. The piezo sine at the drive frequency moves the zero-order fringe.

## 8. Tests (camera-free, 100 % line coverage)

- `tests/test_simulation.py`: speckle statistics (mean intensity and a contrast near 1 on the drum), the per-label seed (repeatable, distinct), sensor linearity in exposure and gain and clipping at 255; Still frames identical; Slow-deformation zero outside the drum; the drum mask, the Bessel functions and zeros, the mode frequencies and the (1, 1) shape against an independent J1 and the cycle's phases; a piezo held step equal to the expected phase; the FFT table against direct numerical integration of J_k (J0 zeros at u = 2.405 and 5.520) and against brute-force exposure integration on a small frame within 1 grey level, for whole and fractional n; reference modulation cancelling where W = P; `Drive` from `plan()` for sine, square, hold, sine piezo and off.
- Engine: the `drive` action reaches a demo source and is ignored by a camera; `demo_object` changes are applied.
- Qt: control visibility (demo or live, Simple UI), its persistence in the demo profile, no cross-camera leaks, and DAQ events forwarded to demo engines (including off on zero, failure and close).
- Existing tests that run real demo frames (test_qt_app, test_engine_camera, test_channels, test_noise_floor, test_reference, test_pixel_square, test_fringe_enhancement, test_wheel, test_update_integration and others): re-run them all, and fix any that relied on the old pattern (e.g. a nonzero Previous difference in the first frames, where the deformation ramp starts at zero rate).
- The performance budget is checked by hand and reported (not as a timing test, which would be flaky in CI).

## 9. Documentation and PR

- README (dummy mode), the in-app `user_guide.html` (what the simulation is, the four objects, the mode frequencies, the DAQ mapping constants), AGENTS.md (the layout and architecture rows for `simulation.py`, the Demo-mode and Engine-threading paragraphs, the new `drive` action), CHANGELOG, and the "future" note in docs/notes/ni-usb-6251-integration.md (now done).
- PR screenshots per the repo rule: the control in each state (demo and live, Simple UI), Raw/Reference/Previous for each object, Vibration at a whole and a fractional cycle count, and Follow DAQ beside the DAQ window. They go on the `pr-screenshots` branch.
- Push only from a branch that carries public history alone (this one does: `git log origin/main..HEAD` holds only new commits), and re-check `git merge-tree` against `origin/main` before pushing.

## 10. Left to the builder

Grid sizes and interpolation for F (within §5.5's target); the exact spot width, speckle pupil and sensor scale (within §5.1's targets); whether the class keeps the `DemoSource` name; note and tooltip wording; how `next_frame` receives the frame rate; and test structure. The builder review also listed these as the builder's to settle: the table's u range and what happens beyond it, exposures under 0.25 cycles, rebuilding tables while the exposure slider is dragged (coalesce, don't rebuild on every value), whether choosing Slow deformation restarts its cycle, and presets carrying `demo_object` between demo cameras.
