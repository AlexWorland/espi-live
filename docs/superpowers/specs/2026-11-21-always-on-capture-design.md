# Always-on capture uploads (design)

## Intent

Installed copies periodically save a still image and, less often, a short video
clip from each camera view and upload them on their own, so the maintainer
collects real data from real setups (different cameras, optics, lighting,
speckle and fringe quality) to tune processing and camera support without
asking users to send files.

## Goals, non-goals, anti-goals

- Goal: while a real camera is processing, a still and an occasional short clip
  per camera reach the maintainer without anyone doing anything.
- Goal: silent in operation (no prompts, dialogs or status-bar messages), like
  the log uploads.
- Goal: the files match what the user sees (same rendering as saved frames and
  recordings).
- Non-goal: a viewer, gallery or search tool (GitHub's file browser is enough).
- Non-goal: replacing the user's own Save/Record. Scheduled captures are
  separate and never interrupt, rename or restart them.
- Non-goal: a per-install switch or consent prompt. The owner chose silent for
  every install, as with logs. (An earlier draft proposed a Help-menu switch;
  the shipped log uploads have none, so this follows them.)
- Anti-goal: anything that blocks or slows the camera, the UI or shutdown, or
  drops frames from a user's own recording.
- Anti-goal: demo (synthetic) frames in the uploads; they must never be
  mistakable for camera data.
- Anti-goal: a token in the public binaries that can write code.

## Constraints

- The repository and its binaries are public; baked-in tokens are extractable.
- Images cannot be scrubbed. A token that can read the captures repository can
  read every uploaded image (accepted below).
- New code is GUI- and camera-free (the front end drives it); 100% line
  coverage with no `# pragma: no cover`; Python 3.10 syntax; `Args:/Returns:`
  docstrings; tests never use a camera, the network or the real `output/`.
- Do not change default exposure, gain, frame rate or the user's recording
  behaviour.
- GitHub repositories are a poor bulk store: a private repo should stay well
  under a few GB. The capture rate must be low enough that the owner can leave
  this running for months.

## Approaches

1. **Private GitHub repository via the Contents API (chosen by the owner).** A
   new private repo, `AlexWorland/espi-live-captures`, written with a
   fine-grained `CAPTURES_TOKEN` (Contents: read and write on that repository
   only), reusing the request, token and background-worker patterns of
   `log_uploads.py`. No new service. Costs: the extractable token can read all
   images; video and stills grow the repo permanently; per-file size limits
   apply.
2. Releases or Git LFS in a private repo. Larger files, but the same read/write
   token exposure and metered LFS bandwidth. Rejected.
3. Write-only relay (Cloudflare Worker in front of R2). A leaked key could only
   upload. Best protection, but a service to deploy and maintain. This was
   recommended; the owner chose approach 1 instead. If the token leaks or the
   repo grows too fast, this is the migration target: keep the uploader behind
   one `put(path, data)` function so the destination can change.

## Design

- New GUI-free module `src/espi_live/capture_uploads.py` beside
  `log_uploads.py`:
  - A **scheduler worker thread** wakes about once a minute and, for each
    channel that is processing a real camera (never demo), decides whether a
    still or clip is due. Timing is per channel and remembered across restarts
    in `output/captures/schedule.json`, so restarting the app does not trigger
    a burst.
  - **Stills:** one every `STILL_INTERVAL_S` (default 30 minutes): a raw and a
    processed pair (see "Decided with the owner"), the processed one from
    `engine.latest_packet().render()`, both lossless PNG. **Clips:** one
    every `CLIP_INTERVAL_S` (default 6 hours), `CLIP_SECONDS` (default 5 s)
    long, as a raw clip and a processed clip over the same window, each written
    with its own `AVIRecorder` (the processed one fed from `latest_packet()`,
    the raw one from the engine's raw-frame hook), so neither touches the
    user's recording. If a user recording is active, take the still but skip
    the clip this round.
  - A pass skips a channel that has no new frame, is stopped, errored or
    switching live, and does nothing when the displayed frame is blank.
  - **Queue:** captures are written to `output/captures/pending/` first, then
    uploaded by a separate **uploader thread** and deleted after a successful
    upload. The queue is bounded (`MAX_QUEUE_BYTES`, oldest dropped first) so an
    offline machine cannot fill the disk.
  - **Budgets:** each file is capped at `MAX_FILE_BYTES` (default 10 MB). A
    still over the cap is downscaled (never lossy-compressed first) until it
    fits; a clip over the cap is re-encoded smaller (frame size, then fps),
    and one that still does not fit is dropped and logged at DEBUG. Each
    machine uploads at most `DAILY_BYTES` (default 40 MB) per day; beyond it,
    captures are dropped until the next day.
- **Remote path:** `captures/<YYYY-MM-DD>/<HHMMSS>-<machine>-<camera>.<png|avi>`
  (the file name also carries `-raw` or `-processed` before the extension),
  where `<machine>` is the existing random 8-hex machine ID from
  `log_uploads.machine_id` and `<camera>` is a short model label plus a stable
  per-camera index, never a serial number. A sidecar `.json` with the same
  stem records the view mode, resolution, exposure, gain, frame rate, app
  build and camera model, so the data is usable without the app.
- **Upload:** the Contents API PUT, creating files only (names are unique, so
  no `sha` lookup). Failures back off exponentially (1 min up to 1 hour), are
  logged at DEBUG only, and are never reported or shown. A 401/403/404 stops
  uploading for the rest of the run.
- **Token:** `_build_info.CAPTURES_TOKEN` from the `CAPTURES_TOKEN` secret
  (stored), `ESPI_CAPTURES_TOKEN` overrides. Missing token: uploads are off and
  the build warns, as for `LOGS_TOKEN`. Source checkouts do not upload.
- **Wiring:** `qt_app.Reporting` (or a sibling object) owns the workers and
  starts them with `start_uploads` once cameras are open. The Tk front end does
  not capture. The channel list is read through a small provider callback so
  the module never imports Qt or touches Qt objects.
- **Disclosure:** one sentence in the user guide and README that images and
  short clips are sent. The logs line in the startup log gets a matching
  "Capture uploads: on/off" line.
- **Docs and release:** README, AGENTS.md (module table row; the tokens rule
  names `CAPTURES_TOKEN`: Contents read and write on `espi-live-captures` only),
  CHANGELOG, and `build-binaries.yml` (inject the secret, warn when missing,
  like `LOGS_TOKEN`).
- **Tests:** against a fake GitHub and fake channels/engines: scheduling and
  persistence, demo and stopped channels skipped, queue bounding, daily budget,
  size capping, backoff, and that a user recording is not disturbed. No
  network, camera or real `output/`.

## Decided with the owner

- **What is sent:** every still capture and every clip sends two files: the
  **raw** camera frames (before any processing; lossless PNG for stills, an
  AVI clip for video) and the **processed** view as shown
  (`FramePacket.render()`). A capture's raw and processed files share a stem
  and one sidecar. The sidecar lists the full processing settings (mode,
  averaging, smoothing, stretch, gamma, noise pipeline, palette) and camera
  settings so a capture can be re-run offline. This needs a small engine hook
  that publishes the newest raw frame (an immutable copy, no extra work on the
  grab path when nothing is due); a raw clip is fed from that hook at the
  camera's grab rate, a processed clip from the displayed packets.

## Calls I made (from the builder check; correct any you disagree with)

- **Which sources:** only FLIR (Spinnaker) cameras and supported capture
  adapters (`channel.supported`, for example the USB-Live2). iPhone sources and
  unsupported or generic video devices never upload, because they are the most
  likely to show a person or a room. Sources without a serial get a stable
  index from their position in the saved profile, never a device name.
- **Budgets:** per machine (not per camera), 60 MB a day because clips now come
  in raw and processed pairs. Clips have a reserved 30 MB share, so stills
  cannot starve them. A raw+processed pair (still or clip) counts as one
  capture: each still file is downscaled to fit about 3 MB when needed and
  each clip file about 7 MB (the sidecar records the downscale or re-encode).
  Fewer captures when over budget, never half a pair.
- **Views that publish rarely:** the processed clip uses the displayed packet
  rate and the raw clip uses the camera's grab rate. If fewer than 2 processed
  packets a second are published (block averaging, phase-shifting), the whole
  clip pair is skipped and logged at DEBUG, because a raw clip alone would not
  match; stills are still taken. A frozen display is skipped (no new frames).
- **GitHub errors:** 401, or 404 on the repository itself, stops uploads for the
  run. 403 (including secondary rate limits), 409, 422 and 5xx back off and
  retry, since many installs write to one branch.

## Left to the builder

Exact interval, size and budget defaults (the numbers above are starting
points; check that a month of one camera stays well under the repo's size
budget), the clip encoder settings and the downscale ladder, the sidecar field
list, how the provider callback exposes channels, the thread shutdown details,
and where the captures repository is created and its token issued (a manual
owner step, like `LOGS_TOKEN`).

## Risks the owner accepted or should watch

- **Extractable read/write token.** Anyone who extracts it from a public build
  can read every uploaded image and could delete or overwrite them. Logs were
  scrubbed text; images are not. Migration path: approach 3.
- **Repo growth.** At the defaults one camera attempts about 48 still pairs and
  4 clip pairs a day, bounded by the 60 MB daily budget. Many installs add up; the owner prunes old
  `captures/<date>/` folders, or recreates the repository.
- **Privacy.** A camera can see a person, a desk or a notice board. Disclosure
  is a documented sentence, not a prompt.
