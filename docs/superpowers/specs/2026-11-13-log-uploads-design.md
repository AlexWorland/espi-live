# Automatic log uploads (design)

## Intent

Installed copies upload their run logs on their own, so the maintainer has a
dated history of every machine's runs to read when someone reports a problem,
without walking the user through sending logs.

## Goals, non-goals, anti-goals

- Goal: every run's log reaches the maintainer, including runs that crashed
  or were killed.
- Goal: silent in operation (no prompts, dialogs or status-bar messages).
- Non-goal: a viewer or search tool for the logs (GitHub's file browser is enough).
- Anti-goal: anything that makes logs public, or a token in the public
  binaries that can write code.
- Anti-goal: an upload that blocks or slows the camera, UI or shutdown.

## Constraints

- The repository and its binaries are public; baked-in tokens are extractable.
- `logs.py`/`error_reports.py` stay GUI- and camera-free; 100% line coverage;
  Python 3.10 syntax; docstrings in `Args:/Returns:` form.

## Approaches

1. **Private logs repository via the Contents API (chosen).** Private repo
   `AlexWorland/espi-live-logs`; builds embed `LOGS_TOKEN`, a fine-grained
   token with Contents: read & write on that repository only (verified: no
   access to other private repositories, writes to `espi-live` refused). A
   leaked token exposes or overwrites logs, never code.
2. Issues in the public repository with the existing `FEEDBACK_TOKEN`: no new
   setup, but every log is public. Rejected.
3. Write-only relay (Cloudflare Worker committing to a private repo): no
   GitHub token in the binary, but a service to deploy and maintain. Rejected
   for now.
4. A `logs` branch in the public repository (the original idea): public logs
   and a code-write token in public builds. Rejected.

## Design (as approved)

- New GUI-free module `src/espi_live/log_uploads.py`: an uploader plus a
  background worker thread, following `error_reports.ErrorReportWorker`.
- Remote path: `logs/<YYYY-MM-DD>/<HHMMSS>-<machine>.log`, the date and time
  of the run (from the log file name); `<machine>` is a random 8-hex ID kept
  in `output/logs/`, so one PC can be followed without its host name.
- Timing: at startup, upload every earlier run log that was never sent or has
  grown since; during the run, re-upload the current log every 15 minutes if
  it grew (same remote file, updated in place). Nothing at shutdown: the next
  start picks up the rest.
- Every log is passed through `error_reports.make_scrubber` before upload.
- Cap about 5 MB per upload: beyond it, keep the start (startup header) and
  the newest part with a marker where the middle was cut.
- Failures are logged locally at DEBUG only; never reported, never shown.
- Off switch shared with error reports: `--no-error-reports`,
  `ESPI_ERROR_REPORTS`, and the Help-menu item, renamed "Send error reports
  and logs automatically". One line in the user guide says logs are sent.
- Token: `_build_info.LOGS_TOKEN` from the `LOGS_TOKEN` secret (stored),
  `ESPI_LOGS_TOKEN` overrides. Missing token: the build warns, uploads are
  off. Source checkouts do not upload.
- Tests against a fake GitHub; README, AGENTS.md, CHANGELOG updated.

## Left to the builder

The record format of what was sent, the Contents API details (GET for the
current `sha`, PUT to create or update), the exact head/tail split, and how
`qt_app` schedules the 15-minute kick.
