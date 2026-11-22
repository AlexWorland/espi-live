"""Scheduled camera captures, uploaded to the developer's private captures repository.

While a real camera is processing, installed builds periodically save a raw
and a processed still, and less often a raw and a processed short clip, per
camera, and send them to the private repository ``AlexWorland/espi-live-captures``
as ``captures/<YYYY-MM-DD>/<HHMMSS>-<machine>-<camera>-<raw|processed>.<png|avi>``
plus one ``.json`` sidecar. Pieces here:

- ``CaptureLedger``: when each camera is next due, and the per-machine daily
  byte budget (stills may not use the clip reserve).
- ``CaptureQueue``: captures wait on disk, a group (the pair or pairs plus the
  sidecar) at a time, until they are uploaded; the oldest are dropped when the
  queue grows past its cap.
- ``CaptureUploader``: PUTs the queued files through the GitHub Contents API.
  ``CaptureUploader.put`` is the one place that talks to the destination, so
  the write-only relay the design names as the later option can replace it.
- ``CaptureWorker``: the scheduler thread. Each pass looks at every eligible
  channel (see ``snapshots.eligible``), takes a raw and a processed still and,
  less often, a raw and a processed clip when they are due and the budget
  allows, queues them, then asks the uploader to send the queue.

The token baked into public builds can read every uploaded image: images,
unlike logs, cannot be scrubbed. That risk was accepted by the owner (see
docs/superpowers/specs/2026-11-21-always-on-capture-design.md). Uploads are
silent and have no switch; failures are logged at DEBUG and never shown or
reported. No GUI imports: the front end drives this module.
"""

import base64
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from datetime import datetime

from espi_live import feedback, snapshots, updater
from espi_live.error_reports import write_json
from espi_live.feedback import FeedbackError
from espi_live.recording import AVIRecorder

logger = logging.getLogger(__name__)

DEFAULT_REPO = "AlexWorland/espi-live-captures"
REPO_ENV = "ESPI_CAPTURES_REPO"
TOKEN_ENV = "ESPI_CAPTURES_TOKEN"
TOKEN_FILE = "captures-token.txt"
BRANCH = "main"
SCHEDULE_NAME = "schedule.json"
PENDING_NAME = "pending"
TEMP_PREFIX = ".tmp-"
STILL_INTERVAL_S = 30 * 60
CLIP_INTERVAL_S = 6 * 60 * 60
CLIP_SECONDS = 5
MAX_STILL_BYTES = 3 * 1024 * 1024
MAX_CLIP_BYTES = 7 * 1024 * 1024
DAILY_BYTES = 60 * 1024 * 1024
CLIP_RESERVE_BYTES = 30 * 1024 * 1024
MAX_QUEUE_BYTES = 200 * 1024 * 1024
MIN_PACKET_RATE = 2.0
BACKOFF_START_S = 60.0
BACKOFF_MAX_S = 3600.0
WORKER_INTERVAL_S = 60.0
RAW_WAIT_S = 5.0
STALL_S = 1.0
CLIP_MAX_WIDTH = 1024
CLIP_MAX_FPS = 10.0
WORK_NAME = "work"
WORKER_THREAD = "capture-uploads"
KINDS = ("still", "clip")


def captures_token():
    """Choose the token that may write to the captures repository.

    Args: None.
    Returns: The token, or "" when this copy cannot upload captures.
    """
    return updater.read_token(TOKEN_ENV, TOKEN_FILE, "CAPTURES_TOKEN")


def repository():
    """Name the repository that receives the captures.

    Args: None.
    Returns: "owner/name" from ESPI_CAPTURES_REPO, the build, or DEFAULT_REPO.
    """
    return (os.environ.get(REPO_ENV, "").strip()
            or updater.build_value("CAPTURES_REPO", "") or DEFAULT_REPO)


class CaptureLedger:
    """Remember when each camera was last captured and what has been spent today."""

    def __init__(self, folder, clock=time.time):
        """Load the saved schedule (a missing or damaged file starts empty).

        Args: folder is the captures folder; clock returns seconds since the
            epoch (tests replace it).
        Returns: None.
        """
        self.folder = folder
        self.clock = clock
        self.last = {}
        self.day = ""
        self.spent = {kind: 0 for kind in KINDS}
        self._load()

    def _path(self):
        """Locate the schedule file.

        Args: None.
        Returns: The path of schedule.json.
        """
        return os.path.join(self.folder, SCHEDULE_NAME)

    def _load(self):
        """Read schedule.json, ignoring anything that is not the expected shape.

        Args: None.
        Returns: None.
        """
        try:
            with open(self._path(), encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        last = data.get("last")
        if isinstance(last, dict):
            self.last = {str(key): value for key, value in last.items()
                         if isinstance(value, (int, float))}
        spent = data.get("spent")
        if isinstance(data.get("day"), str) and isinstance(spent, dict):
            self.day = data["day"]
            for kind in KINDS:
                value = spent.get(kind)
                self.spent[kind] = value if isinstance(value, int) and value >= 0 else 0

    def _save(self):
        """Write the schedule; a folder that cannot be written keeps it in memory.

        Args: None.
        Returns: None.
        """
        try:
            os.makedirs(self.folder, exist_ok=True)
            write_json(self._path(), {"last": self.last, "day": self.day,
                                      "spent": self.spent})
        except OSError as exc:
            logger.debug("Capture schedule not saved: %s", exc)

    def _today(self):
        """Roll the spending over to today.

        Args: None.
        Returns: None.
        """
        today = datetime.fromtimestamp(self.clock()).strftime("%Y-%m-%d")
        if today != self.day:
            self.day = today
            self.spent = {kind: 0 for kind in KINDS}

    def due(self, camera, kind, interval_s):
        """Say whether a camera's next still or clip is due.

        Args: camera is the camera label; kind is "still" or "clip"; interval_s
            is the time between captures of that kind.
        Returns: True when it was never captured, when its saved time is in
            the future (the clock was set back), or when the interval passed.
        """
        last = self.last.get(f"{camera}|{kind}")
        now = self.clock()
        return last is None or last > now or now - last >= interval_s

    def mark(self, camera, kind):
        """Record that a camera was just captured.

        Args: camera is the camera label; kind is "still" or "clip".
        Returns: None.
        """
        self.last[f"{camera}|{kind}"] = self.clock()
        self._save()

    def afford(self, kind, nbytes):
        """Say whether today's budget has room for a capture.

        Args: kind is "still" or "clip"; nbytes is the capture's total size.
        Returns: True when the daily budget allows it; stills may not use the
            bytes reserved for clips.
        """
        self._today()
        if sum(self.spent.values()) + nbytes > DAILY_BYTES:
            return False
        return kind != "still" or self.spent["still"] + nbytes <= DAILY_BYTES - CLIP_RESERVE_BYTES

    def spend(self, kind, nbytes):
        """Charge a finished capture to today's budget.

        Args: kind is "still" or "clip"; nbytes is the capture's total size.
        Returns: None.
        """
        self._today()
        self.spent[kind] += nbytes
        self._save()


class CaptureQueue:
    """Captures waiting on disk for upload, one group (pair or pairs plus sidecar) at a time."""

    def __init__(self, folder, max_bytes):
        """Open the queue, deleting groups a crash left half-written.

        Args: folder is the captures folder; max_bytes caps the queue's size.
        Returns: None.
        """
        self.root = os.path.join(folder, PENDING_NAME)
        self.max_bytes = max_bytes
        try:
            os.makedirs(self.root, exist_ok=True)
            for name in os.listdir(self.root):
                if name.startswith(TEMP_PREFIX):
                    shutil.rmtree(os.path.join(self.root, name), ignore_errors=True)
        except OSError as exc:
            logger.debug("Capture queue not opened: %s", exc)

    def _folder(self, stem):
        """Locate a group's folder.

        Args: stem is the group's remote path before the extension.
        Returns: The folder path ("/" in the stem becomes "~").
        """
        return os.path.join(self.root, stem.replace("/", "~"))

    def groups(self):
        """List the queued groups.

        Args: None.
        Returns: Their stems, oldest first (stems start with the date and time).
        """
        try:
            names = os.listdir(self.root)
        except OSError:
            return []
        return [name.replace("~", "/") for name in sorted(names)
                if not name.startswith(TEMP_PREFIX)]

    def _size(self, stem):
        """Add up a group's file sizes.

        Args: stem is the group.
        Returns: The size in bytes (0 for a missing group).
        """
        folder = self._folder(stem)
        try:
            return sum(os.path.getsize(os.path.join(folder, name))
                       for name in os.listdir(folder))
        except OSError:
            return 0

    def add(self, stem, files):
        """Queue a group, all of it or none of it.

        Args: stem is the remote path before the extension; files maps each
            file's suffix (such as "-raw.png" or ".json") to its bytes.
        Returns: True when queued; False when the disk refused it or the group
            alone is bigger than the queue's cap.
        """
        temporary = None
        try:
            temporary = tempfile.mkdtemp(dir=self.root, prefix=TEMP_PREFIX)
            for suffix, data in files.items():
                with open(os.path.join(temporary, suffix), "wb") as handle:
                    handle.write(data)
            final = self._folder(stem)
            shutil.rmtree(final, ignore_errors=True)
            os.rename(temporary, final)
        except OSError as exc:
            logger.debug("Capture not queued: %s", exc)
            if temporary:
                shutil.rmtree(temporary, ignore_errors=True)
            return False
        groups = self.groups()
        total = sum(self._size(name) for name in groups)
        while total > self.max_bytes and groups:
            oldest = groups.pop(0)
            total -= self._size(oldest)
            self.remove(oldest)
        return stem in groups

    def read(self, stem):
        """Read a queued group.

        Args: stem is the group.
        Returns: {suffix: bytes}; empty when the group is gone or unreadable.
        """
        folder = self._folder(stem)
        try:
            files = {}
            for name in os.listdir(folder):
                with open(os.path.join(folder, name), "rb") as handle:
                    files[name] = handle.read()
            return files
        except OSError:
            return {}

    def remove(self, stem):
        """Delete a group.

        Args: stem is the group.
        Returns: None.
        """
        shutil.rmtree(self._folder(stem), ignore_errors=True)


class CaptureUploader:
    """Send queued captures to the captures repository, backing off on trouble."""

    def __init__(self, queue, request=feedback.github_request, token=None, clock=time.time):
        """Create the uploader.

        Args: queue is the CaptureQueue; request is the GitHub call; token
            overrides captures_token() (tests); clock returns seconds.
        Returns: None.
        """
        self.queue = queue
        self.request = request
        self._token = token
        self.clock = clock
        self.stopped = False
        self.retry_at = 0.0
        self.delay = BACKOFF_START_S

    def token(self):
        """Choose the token for this pass.

        Args: None.
        Returns: The token, or "" when there is none.
        """
        return captures_token() if self._token is None else self._token

    def put(self, path, data):
        """Create one file in the captures repository.

        Args: path is the file's place in the repository; data is its bytes.
        Returns: None. Raises FeedbackError on failure (422 when it exists).
        """
        payload = {"message": f"Capture {path}", "branch": BRANCH,
                   "content": base64.b64encode(data).decode("ascii")}
        self.request("PUT", f"/repos/{repository()}/contents/{path}", self.token(), payload)

    def _back_off(self):
        """Wait before the next pass, doubling each time up to an hour.

        Args: None.
        Returns: None.
        """
        self.retry_at = self.clock() + self.delay
        self.delay = min(self.delay * 2, BACKOFF_MAX_S)

    def send_pending(self):
        """Upload the queued groups, oldest first.

        Args: None.
        Returns: How many groups were uploaded and removed. Never raises: a
            401 or 404 stops uploads for the run; any other failure backs off.
        """
        if self.stopped or not self.token() or self.clock() < self.retry_at:
            return 0
        sent = 0
        for stem in self.queue.groups():
            files = self.queue.read(stem)
            for suffix in sorted(files):
                try:
                    self.put(stem + suffix, files[suffix])
                except FeedbackError as exc:
                    if exc.status == 422:  # the file is already there (a lost reply)
                        continue
                    logger.debug("Capture upload stopped at %s: %s", stem, exc)
                    if exc.status in (401, 404):
                        self.stopped = True
                    else:
                        self._back_off()
                    return sent
            self.queue.remove(stem)
            self.delay = BACKOFF_START_S
            sent += 1
        return sent


class CaptureWorker:
    """Take the due stills and clips on a background thread and hand them to the uploader."""

    def __init__(self, channels, folder, uploader, ledger, queue, machine, build=0,
                 interval_s=WORKER_INTERVAL_S, *, clip_seconds=CLIP_SECONDS,
                 max_still_bytes=MAX_STILL_BYTES, max_clip_bytes=MAX_CLIP_BYTES,
                 sleep=time.sleep, monotonic=time.monotonic, now=datetime.now):
        """Create an idle worker.

        Args: channels is a zero-argument callable returning the current
            channel list (so the worker never imports the front end); folder is
            the captures folder; uploader, ledger and queue are the Task 3
            objects; machine is the machine ID; build is the app build number;
            interval_s is the time between passes. The keywords are for tests:
            the clip length, the per-file size caps, and the clock, sleep and
            wall-time functions.
        Returns: None.
        """
        self.channels = channels
        self.folder = folder
        self.uploader = uploader
        self.ledger = ledger
        self.queue = queue
        self.machine = machine
        self.build = build
        self.interval_s = interval_s
        self.clip_seconds = clip_seconds
        self.max_still_bytes = max_still_bytes
        self.max_clip_bytes = max_clip_bytes
        self.sleep = sleep
        self.monotonic = monotonic
        self.now = now
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.thread = None
        shutil.rmtree(os.path.join(folder, WORK_NAME), ignore_errors=True)

    def start(self):
        """Start the thread; the first pass runs at once.

        Args: None.
        Returns: None.
        """
        self.wake.set()
        self.thread = threading.Thread(target=self._run, name=WORKER_THREAD, daemon=True)
        self.thread.start()

    def kick(self):
        """Ask for a pass now.

        Args: None.
        Returns: None.
        """
        self.wake.set()

    def stop(self):
        """Ask the thread to exit after the pass in progress (it is not joined).

        Args: None.
        Returns: None.
        """
        self.stopping.set()
        self.wake.set()

    def _run(self):
        """Thread body: wait for a kick or the interval, then run a pass.

        Args: None.
        Returns: None.
        """
        while True:
            self.wake.wait(self.interval_s)
            self.wake.clear()
            if self.stopping.is_set():
                return
            try:
                self.run_once()
            except Exception:
                logger.debug("A capture pass failed", exc_info=True)

    def run_once(self):
        """Take every capture that is due, queue it, and upload the queue.

        Args: None.
        Returns: How many captures (still pairs and clip pairs) were queued.
        """
        queued = 0
        for index, channel in enumerate(self.channels()):
            if not snapshots.eligible(channel):
                continue
            camera = snapshots.camera_label(channel.info, index)
            for kind, interval, action in (("still", STILL_INTERVAL_S, self._still),
                                           ("clip", CLIP_INTERVAL_S, self._clip)):
                if not self.ledger.due(camera, kind, interval):
                    continue
                try:
                    queued += 1 if action(channel, camera) else 0
                except Exception:
                    logger.debug("Capture of %s failed", camera, exc_info=True)
        self.uploader.send_pending()
        return queued

    def _wait_raw(self, engine):
        """Wait for the engine's first raw frame after the tap is turned on.

        Args: engine is the channel's engine, its tap already on.
        Returns: The RawFrame, or None when none arrived within RAW_WAIT_S or
            the frame is blank (a constant image: the camera delivers nothing).
        """
        end = self.monotonic() + RAW_WAIT_S
        while True:
            raw = engine.latest_raw()
            if raw is not None:
                blank = raw.image.max() == raw.image.min()
                return None if blank else raw
            if self.monotonic() >= end:
                return None
            self.sleep(0.05)

    def _queue(self, kind, camera, files, total):
        """Charge the budget and queue a finished capture, whole.

        Args: kind is "still" or "clip"; camera is the label; files maps
            suffixes to bytes; total is their size. The camera is marked
            whether the capture was kept or dropped, so a full budget or disk
            is not retried every pass.
        Returns: True when the capture was queued.
        """
        stem = snapshots.remote_stem(self.now(), self.machine, camera)
        ok = self.ledger.afford(kind, total) and self.queue.add(stem, files)
        if ok:
            self.ledger.spend(kind, total)
        self.ledger.mark(camera, kind)
        return ok

    def _still(self, channel, camera):
        """Take a raw and a processed still.

        Args: channel is an eligible Channel; camera is its label.
        Returns: True when a pair was queued. A pair that cannot fit its size
            or budget is dropped whole and the camera waits its next turn;
            having no frame yet leaves it due.
        """
        engine = channel.engine
        if engine.latest_packet() is None:
            return False
        engine.set_raw_tap(True)
        try:
            raw = self._wait_raw(engine)
        finally:
            engine.set_raw_tap(False)
        packet = engine.latest_packet()
        if raw is None or packet is None:
            return False
        raw8, rescaled = snapshots.to_uint8(raw.image)
        processed = packet.render()
        raw_png = snapshots.png_within(raw8, self.max_still_bytes)
        processed_png = snapshots.png_within(processed, self.max_still_bytes)
        if raw_png is None or processed_png is None:
            self.ledger.mark(camera, "still")
            return False
        meta = json.dumps(snapshots.sidecar(
            engine.settings(), channel.info, packet.mode, raw8.shape, processed.shape,
            raw_png[1], processed_png[1], rescaled, self.build)).encode("utf-8")
        files = {"-raw.png": raw_png[0], "-processed.png": processed_png[0], ".json": meta}
        if not snapshots.eligible(channel):  # it may have stopped or switched meanwhile
            return False
        return self._queue("still", camera, files, sum(len(data) for data in files.values()))

    def _clip(self, channel, camera):
        """Take a raw and a processed clip over the same few seconds.

        Args: channel is an eligible Channel; camera is its label.
        Returns: True when a pair was queued. Skipped (and left due) while the
            user records, when fewer than MIN_PACKET_RATE processed frames a
            second are published, or when the channel stops mid-clip (both
            partial files are deleted).
        """
        engine = channel.engine
        rate = engine.status().get("processed_fps", 0.0)
        if (engine.settings().record or rate < MIN_PACKET_RATE
                or engine.latest_packet() is None):
            return False
        if not self.ledger.afford("clip", 1):
            self.ledger.mark(camera, "clip")
            return False
        work = os.path.join(self.folder, WORK_NAME)
        os.makedirs(work, exist_ok=True)
        folder = tempfile.mkdtemp(dir=work, prefix="clip-")
        try:
            return self._record_clip(channel, camera, folder, min(rate, CLIP_MAX_FPS))
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def _record_clip(self, channel, camera, folder, fps):
        """Record both clips into a work folder and queue them.

        Frames are shrunk to CLIP_MAX_WIDTH before they are encoded, so a 6 MP
        camera does not need two full-size encoders beside the live engine.

        Args: channel is an eligible Channel; camera is its label; folder is
            the clip's own work folder; fps is the clip frame rate.
        Returns: True when the pair was queued. False when the clip was
            abandoned (the channel stopped, the camera stalled, the user began
            recording: it stays due) or dropped (an encoder failed or never
            finished, or the clips would not fit: the camera waits its turn).
        """
        engine = channel.engine
        raw_path = os.path.join(folder, "raw.avi")
        processed_path = os.path.join(folder, "processed.avi")
        recorders = []
        engine.set_raw_tap(True)
        try:
            raw = self._wait_raw(engine)
            if raw is None:
                return False
            raw8, rescaled = snapshots.to_uint8(raw.image)
            packet = engine.latest_packet()
            first = packet.render()
            small_raw, raw_shrink = snapshots.fit_width(raw8, CLIP_MAX_WIDTH)
            small_first, processed_shrink = snapshots.fit_width(first, CLIP_MAX_WIDTH)
            recorders.append(AVIRecorder(raw_path, small_raw, fps))
            recorders.append(AVIRecorder(processed_path, small_first, fps))
            completed = self._feed(channel, recorders, fps, raw.sequence)
        finally:
            engine.set_raw_tap(False)
            for recorder in recorders:
                recorder.stop()
            finished = all([recorder.done.wait(RAW_WAIT_S) for recorder in recorders])
        if not completed:
            return False
        if not finished or any(recorder.error for recorder in recorders):
            self.ledger.mark(camera, "clip")
            return False
        raw_scale = snapshots.shrink_clip(raw_path, self.max_clip_bytes)
        processed_scale = None if raw_scale is None else \
            snapshots.shrink_clip(processed_path, self.max_clip_bytes)
        if raw_scale is None or processed_scale is None:
            self.ledger.mark(camera, "clip")
            return False
        with open(raw_path, "rb") as handle:
            raw_bytes = handle.read()
        with open(processed_path, "rb") as handle:
            processed_bytes = handle.read()
        meta = json.dumps(snapshots.sidecar(
            engine.settings(), channel.info, packet.mode, raw8.shape, first.shape,
            raw_shrink * raw_scale, processed_shrink * processed_scale, rescaled,
            self.build)).encode("utf-8")
        files = {"-raw.avi": raw_bytes, "-processed.avi": processed_bytes, ".json": meta}
        if not snapshots.eligible(channel):
            return False
        return self._queue("clip", camera, files, sum(len(data) for data in files.values()))

    def _feed(self, channel, recorders, fps, sequence):
        """Feed the two recorders the newest frames for clip_seconds.

        Args: channel is the channel being captured; recorders are the raw and
            processed AVIRecorders; fps is the clip frame rate; sequence is the
            raw frame the clip started from.
        Returns: True when the whole clip ran; False when it was abandoned: the
            channel stopped being eligible, the user started recording, or the
            camera delivered no new raw frame for STALL_S.
        """
        engine = channel.engine
        started = last_moved = self.monotonic()
        end = started + self.clip_seconds
        while True:
            now = self.monotonic()
            if now >= end:
                return True
            if not snapshots.eligible(channel) or engine.settings().record:
                return False
            latest = engine.latest_raw()
            if latest is not None and latest.sequence != sequence:
                sequence, last_moved = latest.sequence, now
            elif now - last_moved >= STALL_S:
                return False
            if latest is not None:
                recorders[0].submit(snapshots.fit_width(
                    snapshots.to_uint8(latest.image)[0], CLIP_MAX_WIDTH)[0])
            packet = engine.latest_packet()
            if packet is not None:
                recorders[1].submit(snapshots.fit_width(packet.render(), CLIP_MAX_WIDTH)[0])
            self.sleep(1.0 / fps)
