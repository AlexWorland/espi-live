"""Headless batch run: open the cameras, apply settings, record or save frames, exit.

``espi_live --headless --config run.json --duration 600 --record`` runs the
same engines the window uses, with no Qt and no display. The run is controlled
by a JSON file and by signals (Ctrl+C, SIGTERM, Ctrl+Break on Windows), which
stop it cleanly: recordings are finished and cameras released.

When the run ends, the recordings and saved frames it made are copied to the
private captures repository the window's scheduled uploads already use
(``capture_uploads``: the same repository, token and Contents API). Files
that are over their size budget are shrunk in a copy first (stills are
downscaled, clips re-encoded), so the full-size originals stay in ``output/``.
Demo runs never upload.

Config keys (all optional; command-line options override them)::

    {"duration_s": 600,          # 0 or missing: run until a signal
     "record": true,             # record an AVI of each camera
     "save_interval_s": 60,      # also save a PNG every N seconds (0: off)
     "demo": false, "demo_cameras": 1,
     "all_cameras": false,       # start every detected camera, not only the first
     "upload": true,
     "upload_still_mb": 3, "upload_clip_mb": 20,
     "settings": {"mode": "Previous", "exposure_us": 8000},   # every camera
     "cameras": [{"serial": "12345678", "settings": {"gain_db": 3}}]}  # or "index"

``settings`` takes any field a saved profile does (see ``profiles.PERSISTED_FIELDS``).

With ``--control`` the run does not stop on a timer: it takes JSON commands on
stdin (NI DAQ, Phase view, camera settings, saves) and answers on stdout, see
``headless_control``. A control run starts even with no camera, never uploads,
and sets the NI outputs to 0 V on every way out. No GUI imports here.
"""

import json
import os
import queue
import shutil
import signal
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from datetime import datetime

from espi_live import (capture_uploads, headless_control, log_uploads, paths, profiles,
                       snapshots, updater)
from espi_live.channels import demo_channels, open_camera_channels
from espi_live.feedback import FeedbackError
from espi_live.logs import AppLog, header_lines
from espi_live.recording import safe_name

DEFAULTS = {"duration_s": 0, "record": False, "save_interval_s": 0, "demo": False,
            "demo_cameras": 1, "all_cameras": False, "upload": True,
            "upload_still_mb": 3, "upload_clip_mb": 20, "settings": {}, "cameras": []}
POLL_S = 0.25
STATUS_EVERY_S = 30.0
STOP_WAIT_S = 20.0
MAX_UPLOAD_FAILURES = 3


def load_config(path):
    """Read the run's JSON config file.

    Args: path is the file, or None for no file.
    Returns: The DEFAULTS with the file's values on top. Exits with a message
        when the file cannot be read or is not a JSON object.
    """
    config = dict(DEFAULTS)
    if not path:
        return config
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Cannot read config {path}: {exc}")
    if not isinstance(data, dict):
        raise SystemExit(f"Config {path} must be a JSON object.")
    unknown = sorted(set(data) - set(DEFAULTS))
    if unknown:
        print(f"Config {path}: ignoring unknown keys {', '.join(unknown)}", file=sys.stderr)
    config.update({key: value for key, value in data.items() if key in DEFAULTS})
    return config


def build_config(args):
    """Merge the config file with the command-line options.

    Args: args is the argparse namespace from cli.
    Returns: The run's config dict.
    """
    config = load_config(args.config)
    if args.duration is not None:
        config["duration_s"] = args.duration
    if args.record:
        config["record"] = True
    if args.demo:
        config["demo"] = True
    if args.demo_cameras != 1:
        config["demo_cameras"] = args.demo_cameras
    if args.save_interval is not None:
        config["save_interval_s"] = args.save_interval
    if args.no_upload:
        config["upload"] = False
    return config


def main(args):
    """Entry point for ``--headless``.

    Args: args is the argparse namespace from cli.
    Returns: The process exit code (see run()).
    """
    return run(build_config(args), args.log_level, control=args.control,
               checklist_done=args.daq_checklist_done)


def install_signals(stop, log):
    """Make Ctrl+C, SIGTERM and Ctrl+Break request a clean stop.

    Args: stop is the threading.Event the run loop watches; log is the AppLog.
    Returns: A dict of the previous handlers, for restore_signals().
    """
    def handler(number, frame):
        if stop.is_set():
            raise KeyboardInterrupt  # a second signal: stop waiting
        log.write(f"Signal {number} received: stopping, recordings will be finished.")
        stop.set()

    previous = {}
    names = ["SIGINT", "SIGTERM"] + (["SIGBREAK"] if hasattr(signal, "SIGBREAK") else [])
    for name in names:
        try:
            previous[getattr(signal, name)] = signal.signal(getattr(signal, name), handler)
        except ValueError:  # not the main thread
            pass
    return previous


def restore_signals(previous):
    """Put the earlier signal handlers back.

    Args: previous is what install_signals() returned.
    Returns: None.
    """
    for number, handler in previous.items():
        signal.signal(number, handler)


def apply_settings(channel, values, log):
    """Apply config settings to one camera, validated like a saved profile.

    Args: channel is the Channel; values is the settings dict; log is the AppLog.
    Returns: None. Unknown names are reported; wrong values fall back to the
        camera's current ones (with a log line), as when a profile loads.
    """
    unknown = sorted(set(values) - set(profiles.PERSISTED_FIELDS))
    if unknown:
        log.warning(f"{channel.label}: unknown settings ignored: {', '.join(unknown)}")
    profile = profiles.parse_profile(json.dumps({"settings": values}), channel.profile(),
                                     log, f"{channel.label} config")
    if profile is not None:
        channel.apply_profile(profile)


def configure(channels, config, log):
    """Apply the config's settings to every camera.

    Args: channels is the Channel list; config is the run config; log is the AppLog.
    Returns: None.
    """
    for index, channel in enumerate(channels):
        values = dict(config["settings"])
        for entry in config["cameras"]:
            if entry.get("serial") == channel.info.serial or entry.get("index") == index:
                values.update(entry.get("settings", {}))
        if values:
            apply_settings(channel, values, log)
        if len(channels) > 1:
            channel.log.prefix = f"[{channel.label}] "


def open_channels(config, log):
    """Open the demo cameras or detect the real ones.

    Args: config is the run config; log is the AppLog.
    Returns: A list of Channel with the first one opened; empty when no camera
        could be opened (the reason is logged).
    """
    if config["demo"]:
        return demo_channels(log, int(config["demo_cameras"]))
    try:
        return open_camera_channels(log)
    except Exception as exc:
        log.error(f"No camera could be opened: {exc}")
        return []


def start_channels(channels, config, log):
    """Start processing, and recording when asked.

    Args: channels is the Channel list; config is the run config; log is the AppLog.
    Returns: None. Only the opened camera starts unless all_cameras is set.
    """
    for channel in channels:
        if channel.active:
            channel.engine.start()
        elif config["all_cameras"] and channel.supported:
            channel.start()
        else:
            continue
        if config["record"]:
            channel.engine.update_settings(record=True)
        log.info(f"{channel.label}: running"
                 + (", recording" if config["record"] else "") + ".")


def pump_events(channels, log, failures):
    """Handle every engine's events; stop channels that ended.

    Args: channels is the Channel list; log is the AppLog; failures collects
        the labels of cameras that failed.
    Returns: How many channels are still running.
    """
    alive = 0
    for channel in channels:
        engine = channel.engine
        if engine is None:
            continue
        ended = False
        while True:
            try:
                kind, payload = engine.events.get_nowait()
            except queue.Empty:
                break
            if kind == "record_off":
                log.warning(f"{channel.label}: recording stopped by the engine (see the log).")
            elif kind in ("fatal", "open_failed"):
                message = payload[0] if isinstance(payload, tuple) else payload
                log.error(f"{channel.label}: {message}")
                failures.append(channel.label)
                ended = True
            elif kind == "stopped":
                ended = True
        if ended:
            channel.stop()
        else:
            alive += 1
    return alive


def log_status(channels, log):
    """Write one status line per running camera.

    Args: channels is the Channel list; log is the AppLog.
    Returns: None.
    """
    for channel in channels:
        if channel.engine is not None:
            status = channel.engine.status()
            log.info(f"{channel.label}: camera {status.get('camera_fps', '?')} fps, "
                     f"processed {status.get('processed_fps', '?')} fps, "
                     f"{status.get('record', 'Recording: off')}")


def loop(channels, config, stop, log, failures):
    """Run until the duration ends, a signal arrives or every camera has stopped.

    Args: channels is the Channel list; config is the run config; stop is the
        threading.Event the signal handler sets; log is the AppLog; failures
        collects the labels of cameras that failed.
    Returns: None.
    """
    duration = float(config["duration_s"] or 0)
    interval = float(config["save_interval_s"] or 0)
    begun = time.monotonic()
    next_save = begun + interval
    next_status = begun + STATUS_EVERY_S
    while not stop.is_set():
        now = time.monotonic()
        if duration and now - begun >= duration:
            log.write(f"Duration of {duration:g} s reached.")
            return
        if not pump_events(channels, log, failures):
            log.write("No camera is running any more.")
            return
        if interval and now >= next_save:
            next_save += interval
            for channel in channels:
                if channel.engine is not None:
                    channel.engine.request("save", title="headless", notes="Headless run")
        if now >= next_status:
            next_status += STATUS_EVERY_S
            log_status(channels, log)
        stop.wait(POLL_S)


def new_outputs(since):
    """List the recordings and frames this run wrote.

    Args: since is the run's start, seconds since the epoch.
    Returns: A list of (kind, path) with kind "rec" (AVI) or "frame" (PNG),
        oldest first.
    """
    found = []
    for folder, extension, kind in (("recordings", ".avi", "rec"), ("frames", ".png", "frame")):
        directory = paths.output_folder(folder)
        for name in sorted(os.listdir(directory)):
            full = os.path.join(directory, name)
            if (name.lower().endswith(extension) and not name.startswith("dummy_")
                    and os.path.getmtime(full) >= since - 1):
                found.append((kind, full))
    return found


def prepare_upload(kind, path, config, work):
    """Read a file for upload, shrunk in a copy when it is over its budget.

    Args: kind is "rec" or "frame"; path is the original; config is the run
        config; work is a scratch folder.
    Returns: The bytes to upload, or None when it cannot be made to fit
        (the original is never changed).
    """
    if kind == "frame":
        limit = int(config["upload_still_mb"] * 1024 * 1024)
        with open(path, "rb") as handle:
            data = handle.read()
        if len(data) <= limit:
            return data
        import cv2
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        fitted = None if image is None else snapshots.png_within(image, limit)
        return None if fitted is None else fitted[0]
    copy = os.path.join(work, os.path.basename(path))
    shutil.copyfile(path, copy)
    if snapshots.shrink_clip(copy, int(config["upload_clip_mb"] * 1024 * 1024)) is None:
        return None
    with open(copy, "rb") as handle:
        return handle.read()


def upload_outputs(channels, config, started, ended, settings, log):
    """Copy this run's recordings and frames to the captures repository.

    Args: channels is the Channel list; config is the run config; started and
        ended are the run's wall-clock times (seconds); settings maps each
        camera's label to its final Settings; log is the AppLog.
    Returns: True when everything that could be sent was sent (or nothing
        needed sending); False when an upload failed.
    """
    if not capture_uploads.captures_token():
        log.warning("No captures token (set ESPI_CAPTURES_TOKEN, or put captures-token.txt "
                    "beside the program): the outputs stay in the output folder.")
        return False
    machine = log_uploads.machine_id(paths.output_folder("logs"))
    uploader = capture_uploads.CaptureUploader(None)
    when = datetime.fromtimestamp(started)
    outputs = new_outputs(started)
    sent, failures, ok = 0, 0, True
    work = tempfile.mkdtemp(prefix="espi-upload-")
    try:
        for index, channel in enumerate(channels):
            camera = snapshots.camera_label(channel.info, index)
            stem = snapshots.remote_stem(when, machine, camera)
            tag = f"_{safe_name(channel.info.serial)}_"
            mine = [(kind, path) for kind, path in outputs if tag in os.path.basename(path)]
            names = []
            counts = {"rec": 0, "frame": 0}
            for kind, path in mine:
                counts[kind] += 1
                extension = ".avi" if kind == "rec" else ".png"
                remote = f"{stem}-headless-{kind}{counts[kind]}{extension}"
                if not _send(uploader, kind, path, remote, config, work, log):
                    ok, failures = False, failures + 1
                    if uploader.stopped or failures >= MAX_UPLOAD_FAILURES:
                        log.error("Upload stopped; the outputs stay in the output folder.")
                        return False
                else:
                    sent += 1
                    names.append(remote)
            if names:
                meta = {"headless": True, "build": updater.current_build(),
                        "model": channel.info.model, "duration_s": round(ended - started, 1),
                        "settings": asdict(settings[channel.label]) if channel.label in settings
                        else {}, "files": names}
                _send_bytes(uploader, json.dumps(meta, indent=1).encode("utf-8"),
                            f"{stem}-headless.json", log)
        log.info(f"Uploaded {sent} of {len(outputs)} output file(s) to "
                 f"{capture_uploads.repository()}.")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return ok


def _send(uploader, kind, path, remote, config, work, log):
    """Prepare and upload one file.

    Args: uploader is the CaptureUploader; kind, path, config and work are as
        in prepare_upload(); remote is the repository path; log is the AppLog.
    Returns: True when it is uploaded or already there; False otherwise.
    """
    try:
        data = prepare_upload(kind, path, config, work)
    except OSError as exc:
        log.warning(f"Could not read {path}: {exc}")
        return False
    if data is None:
        log.warning(f"{os.path.basename(path)} is too large to upload even when shrunk; "
                    "it stays in the output folder.")
        return False
    return _send_bytes(uploader, data, remote, log)


def _send_bytes(uploader, data, remote, log):
    """Upload bytes to one repository path.

    Args: uploader is the CaptureUploader; data is the content; remote is the
        repository path; log is the AppLog.
    Returns: True when uploaded or already there (422); False on failure
        (a 401 or 404 also sets uploader.stopped).
    """
    try:
        uploader.put(remote, data)
    except FeedbackError as exc:
        if exc.status == 422:
            return True
        log.error(f"Upload of {remote} failed: {exc}")
        if exc.status in (401, 404):
            uploader.stopped = True
        return False
    log.info(f"Uploaded {remote} ({len(data) / 1e6:.1f} MB).")
    return True


def shutdown(channels, log):
    """Stop every camera, finishing recordings, and report what is left.

    Args: channels is the Channel list; log is the AppLog.
    Returns: A dict of each camera's final Settings by label, taken before the stop.
    """
    final = {}
    for channel in channels:
        if channel.engine is not None:
            final[channel.label] = channel.engine.settings()
            channel.stop()
    for channel in channels:
        if not channel.wait(STOP_WAIT_S):
            log.warning(f"{channel.label}: did not stop within {STOP_WAIT_S:g} s.")
    return final


def run_control(channels, config, stop, log, failures, checklist_done, replies):
    """Serve JSON commands on stdin until quit, end of input, a signal or the duration.

    Args: channels is the Channel list (possibly empty); config is the run
        config; stop is the threading.Event the signal handler sets; log is the
        AppLog; failures collects the labels of cameras that failed;
        checklist_done is True when --daq-checklist-done was given; replies is
        the headless_control.line_writer() for stdout.
    Returns: None. The NI outputs are at 0 V and the DAQ thread is finished.
    """
    session = headless_control.ControlSession(
        channels, log, allow_arm=checklist_done,
        pump=lambda: pump_events(channels, log, failures))
    log.info("Control: reading JSON commands on stdin, replying on stdout.")
    try:
        headless_control.serve(session, sys.stdin, replies, stop,
                               float(config["duration_s"] or 0))
    finally:
        session.close()  # outputs to 0 V before the cameras stop


def run(config, log_level=None, control=False, checklist_done=False):
    """Run one headless batch.

    Args: config is the run config (see the module docstring); log_level is
        the --log-level name or None; control serves JSON commands on stdin
        instead of running to a timer (see headless_control); checklist_done
        lets the control session record the NI safety checklist.
    Returns: 0 on success; 1 when a camera failed or the run was cut short by
        an error; 2 when no camera could be opened (not in a control run); 3
        when uploads were wanted but did not all succeed.
    """
    config = dict(config, upload=False) if control else config
    log = AppLog(log_level)
    path = log.start()
    replies = None
    if control:  # stdout is for replies only, from the first log line on
        replies = headless_control.line_writer(headless_control.divert_console(log))
    for line in header_lines("Headless", log.level):
        log.info(line)
    if path:
        log.write(f"Log file: {path}")
    stop = threading.Event()
    previous = install_signals(stop, log)
    started = time.time()
    channels, failures, final, code = [], [], {}, 0
    try:
        channels = open_channels(config, log)
        if channels:
            configure(channels, config, log)
            start_channels(channels, config, log)
        elif control:
            log.warning("Control run: continuing without a camera (the NI DAQ commands work).")
        else:
            code = 2
        if control:
            run_control(channels, config, stop, log, failures, checklist_done, replies)
        elif channels:
            loop(channels, config, stop, log, failures)
    except KeyboardInterrupt:
        log.write("Interrupted.")
    except Exception as exc:
        log.exception(f"Headless run failed: {exc}")
        code = 1
    finally:
        final = shutdown(channels, log)
        restore_signals(previous)
    ended = time.time()
    if failures:
        code = 1
    if code != 2 and config["upload"] and not config["demo"]:
        try:
            if not upload_outputs(channels, config, started, ended, final, log):
                code = code or 3
        except Exception as exc:
            log.exception(f"Upload failed: {exc}")
            code = code or 3
    elif config["upload"] and config["demo"]:
        log.info("Demo run: nothing is uploaded.")
    log.write(f"Headless run finished (exit code {code}).")
    log.close()
    return code
