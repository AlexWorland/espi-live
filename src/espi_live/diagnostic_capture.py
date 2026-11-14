"""Diagnostic captures: a burst of raw frames, packaged and sent for analysis.

Help > Send diagnostic capture asks the displayed camera's engine for a burst
of consecutive raw frames (engine.request("diagnostics"); they arrive as a
diagnostics.Burst). This module turns the burst into a zip and sends it to the
developer's private logs repository, beside the run logs that log_uploads
sends: ``diagnostics/<YYYY-MM-DD>/<HHMMSS>-<machine>.zip``.

The package holds the raw frames (losslessly, as PNG), the colour frames a
video device delivered, the image on screen, and copies of the frames
processed at several settings (Previous with and without stretch, averaging,
a speckle filter, Reference against the first frame, single field for
interlaced sources). The settings are applied to copies, so the camera, the
user's settings and the live view never change. It also holds the frame
measurements of diagnostics.pair_facts, the capture device's details, the
settings in force and this run's log. All text is scrubbed like the run logs
(log_uploads.make_log_scrubber).

A copy is always kept in ``output/logs/diagnostics/``, so a capture that
cannot be sent (no logs token in this copy, no network) can be sent by hand.
No GUI imports: the Qt front end drives DiagnosticWorker.
"""

import base64
import dataclasses
import datetime
import io
import json
import os
import queue
import threading
import zipfile

import cv2
import numpy as np

from espi_live import diagnostics, feedback, log_uploads, updater
from espi_live.feedback import FeedbackError
from espi_live.processing import single_field, smooth_speckle, stretch_contrast

BURST_FRAMES = 8           # consecutive raw frames per capture
MAX_FULL_PIXELS = 1_500_000  # larger frames are cut to the brightest CROP_SIDE square
CROP_SIDE = 1024
PREVIEW_WIDTH = 1600       # the screen image and full-frame preview are shrunk to this
DIFFERENCE_GAIN = 8        # the unstretched difference is multiplied by this
MAX_PACKAGE_BYTES = 30 * 1024 * 1024
LOCAL_FOLDER = "diagnostics"   # under output/logs
REMOTE_FOLDER = "diagnostics"  # in the logs repository
WORKER_THREAD = "espi-diagnostics"


@dataclasses.dataclass(frozen=True)
class Result:
    """What became of a diagnostic capture.

    Attributes: outcome is "sent", "no_token" (saved only: this copy cannot
        send) or "failed" (saved, but not sent); local is the saved zip's
        path, or "" when it could not be saved; remote is its path in the
        logs repository when sent; error says what went wrong.
    """

    outcome: str
    local: str = ""
    remote: str = ""
    error: str = ""


def region_for(frame):
    """Choose the part of a frame the package keeps at full resolution.

    Small frames (an analog capture) are kept whole; a multi-megapixel frame
    would make a package of tens of megabytes, so the brightest square, where
    the lit object is, is kept instead, together with a shrunk full frame.

    Args: frame is a 2-D uint8 array.
    Returns: None for the whole frame, else (x, y, width, height).
    """
    if frame.size <= MAX_FULL_PIXELS:
        return None
    crop, x, y = diagnostics.brightest_crop(frame, CROP_SIDE)
    return (x, y, crop.shape[1], crop.shape[0])


def cut(image, region):
    """Cut the kept region out of an image.

    Args: image is a 2-D or 3-D array; region is from region_for().
    Returns: The image itself for None, else the region's view.
    """
    if region is None:
        return image
    x, y, width, height = region
    return image[y:y + height, x:x + width]


def shrink(image, width=PREVIEW_WIDTH):
    """Shrink an image for a preview, keeping its aspect.

    Args: image is a 2-D or 3-D uint8 array; width is the largest width kept.
    Returns: The image itself when it is narrow enough, else a resized copy.
    """
    if image.shape[1] <= width:
        return image
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def variants(frames, interlaced):
    """Process copies of the frames at several settings.

    Args: frames are at least two consecutive 2-D uint8 frames of one shape;
        interlaced adds the single-field versions.
    Returns: A list of (file name, uint8 image, description) in the order
        they are packaged.
    """
    first, second, last = frames[0], frames[1], frames[-1]
    difference = cv2.absdiff(second, first)
    differences = [cv2.absdiff(after, before) for before, after in zip(frames, frames[1:])]
    average = np.mean(differences, axis=0).astype(np.uint8)
    count = len(differences)
    images = [
        ("previous_difference_x8.png",
         np.clip(difference.astype(np.uint16) * DIFFERENCE_GAIN, 0, 255).astype(np.uint8),
         f"Previous view without stretch: |frame 1 - frame 0| × {DIFFERENCE_GAIN}, "
         "so one grey level of change shows as 8"),
        ("previous_stretch_clip0.png", stretch_contrast(difference, 0.0)[0],
         "Previous view, Stretch on, clip 0 % (exact min/max)"),
        ("previous_stretch_clip3.5.png", stretch_contrast(difference, 3.5)[0],
         "Previous view, Stretch on, clip 3.5 %"),
        (f"previous_average{count}_stretch_clip1.png", stretch_contrast(average, 1.0)[0],
         f"Previous view, average of {count} differences, Stretch clip 1 %"),
        (f"previous_average{count}_gaussian7_stretch_clip1.png",
         stretch_contrast(smooth_speckle(average, "Gaussian", 7), 1.0)[0],
         f"Previous view, average of {count}, Gaussian 7 px, Stretch clip 1 %"),
        ("reference_first_vs_last_stretch_clip1.png",
         stretch_contrast(cv2.absdiff(last, first), 1.0)[0],
         f"Reference view with frame 0 as the reference, frame {len(frames) - 1} "
         "shown, Stretch clip 1 %"),
    ]
    if interlaced:
        field_first, field_second = single_field(first), single_field(second)
        images += [
            ("single_field_raw.png", field_first, "Raw frame 0 in single-field mode"),
            ("single_field_previous_stretch_clip3.5.png",
             stretch_contrast(cv2.absdiff(field_second, field_first), 3.5)[0],
             "Previous view in single-field mode, Stretch clip 3.5 %"),
        ]
    return images


def analysis_text(burst, region, scrub):
    """Write the measurements of every consecutive pair of frames.

    Args: burst is the diagnostics.Burst; region is from region_for();
        scrub is the scrubbing function.
    Returns: The text of analysis.txt.
    """
    frames = burst.frames
    repeats = sum(1 for before, after in zip(frames, frames[1:])
                  if np.array_equal(before, after))
    lines = ["Diagnostic capture analysis", ""]
    lines += [scrub(line) for line in burst.camera]
    height, width = frames[0].shape
    kept = ("whole frame" if region is None else
            f"region x {region[0]}, y {region[1]}, {region[2]} × {region[3]}")
    lines += [f"Frames: {len(frames)} consecutive, {width} × {height}, kept: {kept}",
              f"Interlaced: {'yes' if burst.interlaced else 'no'}",
              f"Repeated frames: {repeats} of {len(frames) - 1} pairs", ""]
    for index, (before, after) in enumerate(zip(frames, frames[1:])):
        colour = burst.colours[index] if index < len(burst.colours) else None
        facts = diagnostics.pair_facts(before, after, colour, burst.interlaced)
        lines.append(f"Frames {index}→{index + 1}: " + "; ".join(facts))
    for clip in (0.0, 3.5):
        _, low, high = stretch_contrast(cv2.absdiff(frames[1], frames[0]), clip)
        lines.append(f"Previous view input range at clip {clip:g} %: {low}–{high}")
    return "\n".join(lines) + "\n"


def png(image):
    """Encode an image as PNG.

    Args: image is a 2-D or 3-D uint8 array.
    Returns: The PNG bytes. Raises ValueError when OpenCV cannot encode it.
    """
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"cannot encode a {image.shape} image as PNG")
    return data.tobytes()


def build_package(burst, log_text, scrub, when):
    """Build the zip of a diagnostic capture.

    Args: burst is the diagnostics.Burst; log_text is this run's log, not
        yet scrubbed (may be ""); scrub is the scrubbing function; when is
        the capture's datetime.
    Returns: The zip as bytes.
    """
    frames = burst.frames
    region = region_for(frames[0])
    kept = [cut(frame, region) for frame in frames]
    files, notes = [], {}

    def add(name, image, note):
        """Queue one image for the zip.

        Args: name is its path in the zip; image is the uint8 array; note
            describes it in the manifest.
        Returns: None.
        """
        files.append((name, png(image)))
        notes[name] = note

    for index, frame in enumerate(kept):
        add(f"raw/frame_{index:02d}.png", frame,
            f"Raw camera frame {index} (before single-field mode), lossless")
    if region is not None:
        add("raw/full_frame_00_preview.png", shrink(frames[0]),
            "Raw frame 0, whole frame, shrunk for orientation only")
    for index, colour in enumerate(burst.colours):
        add(f"colour/frame_{index:02d}.png", cut(colour, region),
            f"Frame {index} as the device delivered it (BGR); a monochrome "
            "camera should be grey")
    if burst.screen is not None:
        add("views/screen.png", shrink(burst.screen),
            "The image on screen when the capture finished, with the user's settings")
    for name, image, note in variants(kept, burst.interlaced):
        add(f"views/{name}", image, note)
    texts = [("analysis.txt", analysis_text(burst, region, scrub))]
    if burst.details:
        texts.append(("device.txt", "\n".join(
            scrub(f"{label}: {text}") for label, text in burst.details) + "\n"))
    if log_text:
        texts.append(("log.txt", log_uploads.scrub_in_chunks(log_text, scrub)))
    manifest = {
        "created": when.isoformat(timespec="seconds"),
        "build": updater.describe_build(),
        "demo": burst.demo,
        "camera": [scrub(line) for line in burst.camera],
        "interlaced": burst.interlaced,
        "frame_size": [int(frames[0].shape[1]), int(frames[0].shape[0])],
        "region": list(region) if region is not None else None,
        "settings": dataclasses.asdict(burst.settings),
        "files": notes,
    }
    texts.append(("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False)))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files:  # PNG is already compressed
            archive.writestr(name, data, compress_type=zipfile.ZIP_STORED)
        for name, text in texts:
            archive.writestr(name, text.encode("utf-8"), compress_type=zipfile.ZIP_DEFLATED)
    return buffer.getvalue()


def package_name(when, demo):
    """Name a saved capture.

    Args: when is the capture's datetime; demo marks synthetic frames.
    Returns: A file name such as "diagnostics_20261114_153012.zip", prefixed
        "dummy_" for dummy mode.
    """
    return f"{'dummy_' if demo else ''}diagnostics_{when:%Y%m%d_%H%M%S}.zip"


def remote_path(when, machine, demo):
    """Place a capture in the logs repository.

    Args: when is the capture's datetime; machine is the machine ID; demo
        marks synthetic frames.
    Returns: "diagnostics/2026-11-14/153012-<machine>.zip" (with
        "-dummy" before ".zip" for dummy mode).
    """
    return (f"{REMOTE_FOLDER}/{when:%Y-%m-%d}/{when:%H%M%S}-{machine}"
            f"{'-dummy' if demo else ''}.zip")


def save_local(data, folder, name):
    """Keep a copy of the package.

    Args: data is the zip; folder is the logs folder; name is the file name.
    Returns: The saved path. Raises OSError when it cannot be written.
    """
    target = os.path.join(folder, LOCAL_FOLDER)
    os.makedirs(target, exist_ok=True)
    path = os.path.join(target, name)
    with open(path, "wb") as handle:
        handle.write(data)
    return path


def upload(data, path, token, request=feedback.github_request):
    """Commit the package to the logs repository as a new file.

    Args: data is the zip; path is its place in the repository; token is the
        logs token; request is the GitHub call.
    Returns: None. Raises FeedbackError on failure.
    """
    request("PUT", f"/repos/{log_uploads.repository()}/contents/{path}", token,
            {"message": f"Diagnostic capture {path}", "branch": log_uploads.BRANCH,
             "content": base64.b64encode(data).decode("ascii")})


def read_log(path):
    """Read this run's log for the package.

    Args: path is the run log, or None when there is none.
    Returns: The text (capped as for log uploads), or "" when it cannot be read.
    """
    if not path:
        return ""
    try:
        return log_uploads.read_for_upload(path)[0]
    except OSError:
        return ""


def send(burst, folder, log_path, serials, when, token=None, request=feedback.github_request):
    """Package a capture, keep a copy and send it.

    Args: burst is the diagnostics.Burst; folder is the logs folder;
        log_path is this run's log file or None; serials returns this run's
        camera serial numbers; when is the capture's datetime; token
        overrides log_uploads.logs_token() (tests); request is the GitHub call.
    Returns: A Result.
    """
    scrub = log_uploads.make_log_scrubber(log_uploads.remember_serials(folder, serials()))
    data = build_package(burst, read_log(log_path), scrub, when)
    try:
        local = save_local(data, folder, package_name(when, burst.demo))
    except OSError as exc:
        local, saving_error = "", f"could not save a copy: {exc}"
    else:
        saving_error = ""
    token = log_uploads.logs_token() if token is None else token
    if not token:
        return Result("no_token", local=local, error=saving_error)
    if len(data) > MAX_PACKAGE_BYTES:
        return Result("failed", local=local, error=(
            f"the capture is {len(data) / 1e6:.0f} MB, more than the "
            f"{MAX_PACKAGE_BYTES / 1e6:.0f} MB that can be sent"))
    path = remote_path(when, log_uploads.machine_id(folder), burst.demo)
    try:
        upload(data, path, token, request)
    except FeedbackError as exc:
        return Result("failed", local=local, error=str(exc))
    return Result("sent", local=local, remote=path)


class DiagnosticWorker:
    """Package and send a capture on a background thread the front end polls.

    The front end reads ``events`` for ("done", Result). An exception other
    than the expected ones is a bug: it is posted as a failure and re-raised
    for threading.excepthook to log and report.
    """

    def __init__(self, folder, log_path, serials, token=None,
                 request=feedback.github_request):
        """Create an idle worker.

        Args: folder is the logs folder; log_path is this run's log file or
            None; serials returns this run's camera serial numbers; token and
            request are passed to send() (tests).
        Returns: None.
        """
        self.folder = folder
        self.log_path = log_path
        self.serials = serials
        self.token = token
        self.request = request
        self.events = queue.Queue()
        self.busy = False

    def start(self, burst, when=None):
        """Start packaging and sending a capture.

        Args: burst is the diagnostics.Burst; when is the capture's datetime
            (now when None).
        Returns: False if a capture is already being sent, else True.
        """
        if self.busy:
            return False
        self.busy = True
        when = when or datetime.datetime.now()

        def work():
            """Thread body: send the capture and post the result.

            Args: None.
            Returns: None.
            """
            try:
                self.events.put(("done", send(burst, self.folder, self.log_path,
                                              self.serials, when, self.token,
                                              self.request)))
            except Exception as exc:
                self.events.put(("done", Result("failed",
                                                error=f"{type(exc).__name__}: {exc}")))
                raise
            finally:
                self.busy = False

        threading.Thread(target=work, name=WORKER_THREAD, daemon=True).start()
        return True
