#!/usr/bin/env python3
"""Check how the analog capture stick's brightness reaches the viewer.

Run this on the camera PC with the USB-Live2 plugged in, the XC-ST50 running
and ESPI Live, WinTV and Hauppauge Capture closed (the stick serves one
program at a time). Keep the scene still while it runs (about 10 s). It opens
the stick, so never run it from the test suite.

    python scripts/check_analog_luma.py            # the first USB-Live2 found
    python scripts/check_analog_luma.py --list     # list video devices only
    python scripts/check_analog_luma.py --device 2 --frames 60

It prints, for the same device:

1. The BGR path (what earlier versions always used): DirectShow on Windows,
   V4L2 on Linux, with the stick's YUV converted to BGR and then to grey.
2. The raw path: Media Foundation (Windows) or V4L2 (Linux) with
   CAP_PROP_CONVERT_RGB 0: the delivered layout, FOURCC and size, the raw Y
   range and how much of it lies outside video range (16-235), and, on the
   same frames, the raw Y expanded to 0-255 against OpenCV's own YUY2 -> BGR
   -> grey (mean and largest difference).
3. Both paths side by side (different frames of the still scene): mean,
   spread and percentiles of the BGR path's grey, the expanded raw Y and the
   unexpanded raw Y, to show which grey-level scale the BGR path really has.
4. Which path ESPI Live itself chooses (CaptureSession.open), with its log line.

Only the standard library, NumPy and OpenCV are needed, plus the package in
src/ (found relative to this file) and cv2_enumerate_cameras for device names.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from espi_live import capture  # noqa: E402
from espi_live.capture_models import adapter_for, fourcc_text  # noqa: E402


class PrintLog:
    """Print log lines with a prefix, like the viewer's log."""

    def write(self, message):
        """Print one log line.

        Args: message is the text.
        Returns: None.
        """
        print(f"  [log] {message}")


def stats(name, image):
    """Print the brightness statistics of a grey image.

    Args: name labels the line; image is a 2-D array (or stacked frames).
    Returns: A dict with mean, sd and the 1st and 99th percentiles.
    """
    values = np.asarray(image, dtype=np.float64)
    low, high = np.percentile(values, [1, 99])
    result = {"mean": values.mean(), "sd": values.std(), "p1": low, "p99": high}
    print(f"  {name:<28} min {values.min():5.0f}  p1 {low:6.1f}  mean {values.mean():6.1f}"
          f"  sd {values.std():5.1f}  p99 {high:6.1f}  max {values.max():5.0f}")
    return result


def yuy2_pairs(frame, width, height):
    """Rearrange a raw YUY2 frame as (height, width, 2): luma, then chroma.

    Args: frame is a raw frame in any layout capture.yuy2_luma accepts;
        width and height are its size.
    Returns: A (height, width, 2) uint8 array, or None when it is not YUY2.
    """
    if capture.yuy2_luma(frame, width, height) is None:
        return None
    if frame.shape == (height, width, 2):
        return frame
    data = frame.reshape(-1)
    rows = data.reshape(height, data.size // height)[:, :2 * width]
    return np.ascontiguousarray(rows).reshape(height, width, 2)


def read_frames(cap, count, convert):
    """Read frames, skipping empty reads.

    Args: cap is an open cv2.VideoCapture; count is how many to keep;
        convert turns each frame into what is kept, or returns None to stop.
    Returns: (kept list, first frame as delivered or None).
    """
    kept, first = [], None
    for _ in range(count * 3):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if first is None:
            first = frame
        item = convert(frame)
        if item is None:
            break
        kept.append(item)
        if len(kept) == count:
            break
    return kept, first


def bgr_path(info, adapter, count):
    """Measure the BGR path: the stick's own backend, converted to grey.

    Args: info is the device's CameraInfo; adapter its CaptureAdapter;
        count is the number of frames.
    Returns: The stacked grey frames, or None when nothing arrived.
    """
    print(f"\n1. BGR path through {capture.backend_name(info.backend)}")
    cap = cv2.VideoCapture(info.index, info.backend)
    try:
        if not cap.isOpened():
            print("  The device did not open. Close ESPI Live, WinTV and Hauppauge Capture.")
            return None
        cap.set(cv2.CAP_PROP_CHANNEL, adapter.channel)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, adapter.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, adapter.height)
        frames, first = read_frames(cap, count, capture.to_mono8)
        if not frames:
            print("  No video arrived.")
            return None
        print(f"  FOURCC {fourcc_text(cap.get(cv2.CAP_PROP_FOURCC)) or 'unknown'}, "
              f"first frame {capture.describe_layout(first)}, {len(frames)} frames")
        grey = np.stack(frames)
        stats("BGR -> grey", grey)
        return grey
    finally:
        cap.release()


def open_raw(info, adapter):
    """Open the device the way the raw path does, conversion off.

    Args: info is the device's CameraInfo; adapter its CaptureAdapter.
    Returns: (cv2.VideoCapture, backend) or (None, reason).
    """
    route = capture.raw_route(info)
    if route == "msmf":
        try:
            index = capture.find_msmf_index(info)
        except LookupError as exc:
            return None, str(exc)
        print(f"  Media Foundation lists it at index {index}")
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    elif route == "v4l2":
        cap = cv2.VideoCapture(info.index, info.backend)
        cap.set(cv2.CAP_PROP_CHANNEL, adapter.channel)
    else:
        return None, "this platform has no raw route (Windows or Linux only)"
    if not cap.isOpened():
        cap.release()
        return None, "the device did not open"
    accepted = cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, adapter.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, adapter.height)
    print(f"  set(CONVERT_RGB, 0) returned {accepted}; it now reads "
          f"{cap.get(cv2.CAP_PROP_CONVERT_RGB):g}")
    return cap, cv2.CAP_MSMF if route == "msmf" else info.backend


def raw_path(info, adapter, count):
    """Measure the raw path and compare it with OpenCV's conversion of the same frames.

    Args: info is the device's CameraInfo; adapter its CaptureAdapter;
        count is the number of frames.
    Returns: The stacked raw luma frames, or None when the raw path fails.
    """
    print("\n2. Raw path")
    cap, backend = open_raw(info, adapter)
    if cap is None:
        print(f"  Not available: {backend}.")
        return None
    try:
        fourcc = fourcc_text(cap.get(cv2.CAP_PROP_FOURCC))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Backend {capture.backend_name(backend)}, FOURCC {fourcc or 'unknown'}, "
              f"size {width} x {height}")

        def luma_of(frame):
            """Take one frame's luma when it is YUY2 of the reported size.

            Args: frame is what read() returned.
            Returns: A 2-D uint8 copy, or None.
            """
            luma = capture.yuy2_luma(frame, width, height)
            return None if luma is None else luma.copy()

        frames, first = read_frames(cap, count, luma_of)
        if first is None:
            print("  No raw frame arrived.")
            return None
        print(f"  First frame {capture.describe_layout(first)}")
        if fourcc not in capture.YUY2_FOURCCS or not frames:
            print("  Not YUY2 of that size: the backend ignored CONVERT_RGB 0 or "
                  "sends another format. ESPI Live keeps the BGR path.")
            return None
        raw = np.stack(frames)
        print(f"  {len(frames)} frames; raw Y codes {int(raw.min())}-{int(raw.max())}; "
              f"below 16: {100 * np.mean(raw < capture.VIDEO_BLACK):.2f} %, "
              f"above 235: {100 * np.mean(raw > capture.VIDEO_WHITE):.2f} %")
        histogram = np.bincount(raw.reshape(-1), minlength=256)
        used = np.nonzero(histogram)[0]
        print(f"  Histogram: {len(used)} distinct codes between {used[0]} and {used[-1]}")
        expanded = capture.expand_video_range(raw)
        ok, sample = cap.read()
        pairs = yuy2_pairs(sample, width, height) if ok and sample is not None else None
        if pairs is not None:
            old = cv2.cvtColor(cv2.cvtColor(pairs, cv2.COLOR_YUV2BGR_YUY2), cv2.COLOR_BGR2GRAY)
            new = capture.expand_video_range(pairs[:, :, 0])
            differences = np.abs(old.astype(int) - new.astype(int))
            print(f"  Same frame, OpenCV YUY2 -> BGR -> grey vs expanded raw Y: "
                  f"mean |difference| {differences.mean():.3f}, largest {differences.max()}, "
                  f"pixels off by more than 1: {100 * np.mean(differences > 1):.2f} %")
            chroma = pairs[:, :, 1].astype(np.float64)
            print(f"  Chroma bytes (a monochrome camera should sit near 128): "
                  f"mean {chroma.mean():.1f}, sd {chroma.std():.1f}")
        stats("raw Y expanded", expanded)
        return raw
    finally:
        cap.release()


def compare(grey, raw):
    """Say which grey-level scale the BGR path has, from both paths' statistics.

    Args: grey is the BGR path's stacked frames; raw the raw luma frames.
    Returns: None.
    """
    print("\n3. Both paths (different frames of the still scene)")
    bgr = stats("BGR -> grey", grey)
    expanded = stats("raw Y expanded (viewer)", capture.expand_video_range(raw))
    plain = stats("raw Y unexpanded", raw)
    for name, other in (("expanded", expanded), ("unexpanded", plain)):
        ratio = bgr["sd"] / other["sd"] if other["sd"] else float("nan")
        print(f"  BGR vs {name:<10}: mean gap {bgr['mean'] - other['mean']:+6.1f}, "
              f"spread ratio {ratio:5.3f} (1.000 = same scale)")
    # The spread tells the scales apart (1.164 between them); the mean barely
    # does at mid-grey, because both scales cross near level 110.
    if not bgr["sd"] or not plain["sd"] or bgr["sd"] < 3:
        print("  The scene has too little contrast to tell the scales apart; aim the "
              "camera at something with light and dark areas and run again.")
    elif abs(np.log(bgr["sd"] / expanded["sd"])) <= abs(np.log(bgr["sd"] / plain["sd"])):
        print("  The BGR path matches the expanded raw Y: the viewer keeps the earlier "
              "grey-level scale.")
    else:
        print("  The BGR path matches the UNEXPANDED raw Y: the expansion in "
              "capture.EXPAND_TABLE would change the scale here. Report this, and run "
              "ESPI Live with ESPI_ANALOG_LUMA=bgr until it is fixed.")


def viewer_choice(info):
    """Open the device as ESPI Live does and report the luma path it chooses.

    Args: info is the device's CameraInfo.
    Returns: None.
    """
    print("\n4. What ESPI Live chooses")
    session = capture.CaptureSession(PrintLog(), info)
    try:
        session.open()
    except RuntimeError as exc:
        print(f"  Open failed: {exc}")
        return
    try:
        frame, state = session.grab(100)
        print(f"  Luma path: {session.luma_path}; backend {session.identity.backend}; "
              f"first frame {session.raw_layout}")
        if state == "ok":
            print(f"  Delivered frame {frame.shape} {frame.dtype}, mean {frame.mean():.1f}")
        else:
            print(f"  Grab failed: {frame}")
    finally:
        session.close()


def main(argv=None):
    """Run the check.

    Args: argv is the argument list (default: sys.argv[1:]).
    Returns: The process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--list", action="store_true", help="list video devices and stop")
    parser.add_argument("--device", type=int, default=0,
                        help="device number from --list (default: the first USB-Live2)")
    parser.add_argument("--frames", type=int, default=30, help="frames per path (default 30)")
    args = parser.parse_args(argv)

    print(f"OpenCV {cv2.__version__}, NumPy {np.__version__}, {sys.platform}")
    devices = capture.list_capture_devices(PrintLog())
    for number, info in enumerate(devices, start=1):
        known = "recognised adapter" if adapter_for(info.vid, info.pid, info.model) else ""
        usb = f"{info.vid:04x}:{info.pid:04x}" if info.vid is not None else "no USB ID"
        print(f"  {number}: {info.model or 'video device'} ({usb}) {known}")
    if args.list:
        return 0
    if args.device:
        chosen = devices[args.device - 1] if 0 < args.device <= len(devices) else None
    else:
        chosen = next((info for info in devices
                       if adapter_for(info.vid, info.pid, info.model)), None)
    if chosen is None:
        print("No USB-Live2 (or the chosen device) was found.")
        return 1
    adapter = adapter_for(chosen.vid, chosen.pid, chosen.model)
    if adapter is None:
        print("The chosen device is not a recognised adapter; ESPI Live uses it as BGR.")
        return 1
    print(f"Checking {chosen.label}. Keep the scene still.")
    grey = bgr_path(chosen, adapter, args.frames)
    raw = raw_path(chosen, adapter, args.frames)
    if grey is not None and raw is not None:
        compare(grey, raw)
    viewer_choice(chosen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
