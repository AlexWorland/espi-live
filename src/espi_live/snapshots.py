"""File helpers for scheduled captures: who may upload, names, size fitting, sidecars.

Scheduled captures send a raw and a processed still (PNG) and clip (AVI) per
camera. This module holds the parts that need neither a camera nor the
network, so ``capture_uploads`` and its tests share them. Nothing here
imports a GUI toolkit, and no name or sidecar carries a camera serial number.
"""

import os
import tempfile
from dataclasses import asdict

import cv2
import numpy as np

from espi_live.recording import safe_name

MIN_SCALE = 0.25
SCALE_STEP = 0.75
# (frame-size scale, keep every Nth frame) tried in order to fit a clip.
CLIP_LADDER = ((0.75, 1), (0.5, 1), (0.5, 2), (0.35, 2), (0.25, 2))


def eligible(channel):
    """Decide whether a channel's camera may be captured and uploaded.

    Args: channel is a Channel (or anything with active, demo, supported, info
        and frozen).
    Returns: True for a processing FLIR camera or supported capture adapter
        whose display is live; False for demo sources, stopped channels,
        iPhones, unsupported video devices and a frozen display.
    """
    # demo_source stays set until the window handles "switched_live", while the
    # engine already says the source is live: until then the channel's info is
    # still the synthetic camera's, so the channel is not captured.
    if (not channel.active or channel.demo or getattr(channel, "demo_source", False)
            or channel.frozen is not None):
        return False
    kind = channel.info.kind
    return kind == "spinnaker" or (kind == "capture" and bool(channel.supported))


def camera_label(info, index):
    """Name a camera for file names without its serial number.

    Args: info is the camera's CameraInfo; index is its position in the
        channel list.
    Returns: "<model>-<index>" with file-safe characters, or "camera-<index>"
        when the camera reports no model.
    """
    model = safe_name(info.model)[:24].strip("_")
    return f"{model or 'camera'}-{index}"


def remote_stem(when, machine, camera):
    """Place one capture in the captures repository, before the extension.

    Args: when is the capture's local datetime; machine is the machine ID;
        camera is the camera_label.
    Returns: "captures/YYYY-MM-DD/HHMMSS-<machine>-<camera>".
    """
    return f"captures/{when:%Y-%m-%d}/{when:%H%M%S}-{machine}-{camera}"


def to_uint8(image):
    """Make a frame 8-bit for PNG and AVI files.

    Args: image is a 2-D uint8 or deep float32 frame.
    Returns: (image8, rescaled): uint8 frames come back unchanged with False;
        float frames are min/max scaled to 0..255 (all zero when constant)
        with True.
    """
    if image.dtype == np.uint8:
        return image, False
    low, high = float(image.min()), float(image.max())
    if high <= low:
        return np.zeros(image.shape, np.uint8), True
    scaled = (image.astype(np.float32) - low) * (255.0 / (high - low))
    return scaled.astype(np.uint8), True


def fit_width(image, max_width):
    """Shrink a frame so it is at most a given width, keeping its aspect.

    Args: image is a 2-D or BGR image; max_width is the widest allowed.
    Returns: (image, scale): the image itself and 1.0 when it already fits,
        else an INTER_AREA copy and the scale applied.
    """
    width = image.shape[1]
    if width <= max_width:
        return image, 1.0
    scale = max_width / width
    return cv2.resize(image, (max_width, max(1, round(image.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA), scale


def png_within(image, max_bytes):
    """Encode a lossless PNG that fits a byte budget, downscaling if needed.

    Args: image is a 2-D uint8 image; max_bytes is the file size limit.
    Returns: (png_bytes, scale), where scale is 1.0 when nothing was
        downscaled; None when it would need a scale below MIN_SCALE. Pixels are
        never lossy-compressed: only INTER_AREA downscaling reduces size.
    """
    scale = 1.0
    while scale >= MIN_SCALE:
        if scale == 1.0:
            candidate = image
        else:
            height, width = image.shape[:2]
            candidate = cv2.resize(image, (max(1, round(width * scale)),
                                           max(1, round(height * scale))),
                                   interpolation=cv2.INTER_AREA)
        ok, data = cv2.imencode(".png", candidate)
        if ok and len(data) <= max_bytes:
            return data.tobytes(), scale
        scale *= SCALE_STEP
    return None


def _reencode(path, target, scale, keep_every):
    """Write a smaller copy of a clip.

    Args: path is the source AVI; target the copy's path; scale is the frame
        size factor; keep_every keeps every Nth frame (the rate drops with it).
    Returns: True when frames were written.
    """
    source = cv2.VideoCapture(path)
    writer = None
    written = 0
    try:
        fps = (source.get(cv2.CAP_PROP_FPS) or 15.0) / keep_every
        index = 0
        while True:
            ok, frame = source.read()
            if not ok:
                break
            index += 1
            if (index - 1) % keep_every:
                continue
            if scale != 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(target, cv2.VideoWriter_fourcc(*"MJPG"),
                                         fps, (width, height), True)
            writer.write(frame)
            written += 1
    finally:
        source.release()
        if writer is not None:
            writer.release()
    return written > 0


def discard(path):
    """Delete a capture file, ignoring one that is already gone.

    Args: path is the file.
    Returns: None.
    """
    try:
        os.unlink(path)
    except OSError:
        pass


def shrink_clip(path, max_bytes):
    """Re-encode an AVI in place until it fits a byte budget.

    Args: path is the clip; max_bytes is the size limit.
    Returns: The frame-size scale used (1.0 when the clip already fit), or
        None after deleting the file because it could not be made to fit (or
        could not be read).
    """
    try:
        if os.path.getsize(path) <= max_bytes:
            return 1.0
        handle, target = tempfile.mkstemp(dir=os.path.dirname(path) or ".",
                                          prefix=".shrink_", suffix=".avi")
        os.close(handle)
    except OSError:
        return None
    try:
        for scale, keep_every in CLIP_LADDER:
            if _reencode(path, target, scale, keep_every) and \
                    os.path.getsize(target) <= max_bytes:
                os.replace(target, path)
                return scale
        discard(path)
        return None
    finally:
        discard(target)


def sidecar(settings, info, view, raw_shape, processed_shape, scale_raw,
            scale_processed, rescaled_raw, build):
    """Describe a capture so it can be re-run offline.

    Args: settings is the channel's engine Settings; info is its CameraInfo;
        view is the view mode name; raw_shape and processed_shape are the
        images' (height, width) before fitting; scale_raw and scale_processed
        are the downscale or re-encode factors applied; rescaled_raw is True
        when a deep raw frame was min/max scaled to 8 bits; build is the app
        build number.
    Returns: A JSON-serializable dict with no camera serial number.
    """
    return {
        "view": view,
        "model": info.model,
        "build": build,
        "settings": asdict(settings),
        "raw": {"shape": list(raw_shape), "scale": scale_raw, "rescaled": bool(rescaled_raw)},
        "processed": {"shape": list(processed_shape), "scale": scale_processed},
    }
