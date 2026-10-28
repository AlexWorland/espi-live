"""
blackfly_espi_preview_diff_simple.py

FLIR Blackfly (Spinnaker / PySpin) script with:

- Low-latency camera setup (NewestOnly buffer mode, fast normalization, optional flush)
- Live preview (raw frames)
- Frame-to-frame subtraction (current - previous)
- Software ROI (user-selectable via 'r')
- Averaged still-image capture ('s')
- AVI recording of difference frames only ('v')

Deliberately omitted:
- Trigger/tone control
- ESPI phase stepping
- Wrapped phase / modulation / pseudocolor
- OpenCV trackbar controls (exposure/gain set in code)
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import PySpin


class BlackflyCamera:
    """
    Minimal Blackfly wrapper with low-latency stream settings.

    Uses:
    - StreamBufferHandlingMode = NewestOnly
    - Manual exposure and gain
    - Optional (commented) hardware ROI
    """

    def __init__(self, serial=None, exposure_us=3000.0, gain_db=0.0):
        self.streaming = False

        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        if self.cam_list.GetSize() == 0:
            self.cam_list.Clear()
            self.system.ReleaseInstance()
            raise RuntimeError("No Spinnaker/PySpin cameras detected.")

        if serial is None:
            self.cam = self.cam_list[0]
        else:
            self.cam = self.cam_list.GetBySerial(serial)

        self.cam.Init()

        # ---------- low-latency stream setup ----------
        s_nodemap = self.cam.GetTLStreamNodeMap()
        handling_mode = PySpin.CEnumerationPtr(s_nodemap.GetNode("StreamBufferHandlingMode"))
        if PySpin.IsReadable(handling_mode) and PySpin.IsWritable(handling_mode):
            newest_only = handling_mode.GetEntryByName("NewestOnly")
            if PySpin.IsReadable(newest_only):
                handling_mode.SetIntValue(newest_only.GetValue())

        # ---------- acquisition and exposure/gain ----------
        nodemap = self.cam.GetNodeMap()

        # Acquisition mode: Continuous
        acq_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        if PySpin.IsReadable(acq_mode) and PySpin.IsWritable(acq_mode):
            cont_entry = acq_mode.GetEntryByName("Continuous")
            acq_mode.SetIntValue(cont_entry.GetValue())

        # Disable auto exposure/gain
        exp_auto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
        if PySpin.IsReadable(exp_auto) and PySpin.IsWritable(exp_auto):
            off_entry = exp_auto.GetEntryByName("Off")
            exp_auto.SetIntValue(off_entry.GetValue())

        gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
        if PySpin.IsReadable(gain_auto) and PySpin.IsWritable(gain_auto):
            off_entry = gain_auto.GetEntryByName("Off")
            gain_auto.SetIntValue(off_entry.GetValue())

        # Cache exposure and gain limits
        exp_node = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
        if PySpin.IsReadable(exp_node) and PySpin.IsWritable(exp_node):
            self.exp_min = exp_node.GetMin()
            self.exp_max = exp_node.GetMax()
        else:
            self.exp_min, self.exp_max = 100.0, 100000.0

        gain_node = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
        if PySpin.IsReadable(gain_node) and PySpin.IsWritable(gain_node):
            self.gain_min = gain_node.GetMin()
            self.gain_max = gain_node.GetMax()
        else:
            self.gain_min, self.gain_max = 0.0, 20.0

        self.exp_node = exp_node
        self.gain_node = gain_node

        # Set initial exposure and gain (clamped to limits)
        self.set_exposure_us(exposure_us)
        self.set_gain_db(gain_db)

        # (Optional) hardware ROI – uncomment and adjust if desired:
        # width_node = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
        # height_node = PySpin.CIntegerPtr(nodemap.GetNode("Height"))
        # offsetx_node = PySpin.CIntegerPtr(nodemap.GetNode("OffsetX"))
        # offsety_node = PySpin.CIntegerPtr(nodemap.GetNode("OffsetY"))
        # if (PySpin.IsWritable(width_node) and PySpin.IsWritable(height_node) and
        #     PySpin.IsWritable(offsetx_node) and PySpin.IsWritable(offsety_node)):
        #     width_node.SetValue(min(width_node.GetMax(), 1024))
        #     height_node.SetValue(min(height_node.GetMax(), 1024))
        #     offsetx_node.SetValue(0)
        #     offsety_node.SetValue(0)

        self.height = self.cam.Height.GetValue()
        self.width = self.cam.Width.GetValue()

    # ---------- camera control ----------

    def set_exposure_us(self, exposure_us):
        if self.exp_node is not None and PySpin.IsWritable(self.exp_node):
            val = float(np.clip(exposure_us, self.exp_min, self.exp_max))
            self.exp_node.SetValue(val)

    def set_gain_db(self, gain_db):
        if self.gain_node is not None and PySpin.IsWritable(self.gain_node):
            val = float(np.clip(gain_db, self.gain_min, self.gain_max))
            self.gain_node.SetValue(val)

    def start(self):
        if self.streaming:
            return
        try:
            self.cam.BeginAcquisition()
            self.streaming = True
        except PySpin.SpinnakerException as e:
            raise RuntimeError(f"BeginAcquisition failed: {e}") from e

    def stop(self):
        if not self.streaming:
            return
        try:
            self.cam.EndAcquisition()
        except PySpin.SpinnakerException:
            pass
        finally:
            self.streaming = False

    def close(self):
        self.stop()
        try:
            self.cam.DeInit()
        except PySpin.SpinnakerException:
            pass
        del self.cam
        try:
            self.cam_list.Clear()
        except PySpin.SpinnakerException:
            pass
        try:
            self.system.ReleaseInstance()
        except PySpin.SpinnakerException:
            pass

    def grab_frame(self, timeout_ms=100):
        """
        Grab one frame with timeout. If the camera is not delivering frames,
        this will raise a RuntimeError instead of blocking forever.
        """
        try:
            img = self.cam.GetNextImage(timeout_ms)
        except PySpin.SpinnakerException as e:
            raise RuntimeError(f"GetNextImage timeout or error: {e}") from e

        if img.IsIncomplete():
            status = img.GetImageStatus()
            img.Release()
            raise RuntimeError(f"Incomplete image: status {status}")

        if img.GetPixelFormat() != PySpin.PixelFormat_Mono8:
            conv = img.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
            data = conv.GetData().reshape(self.height, self.width)
        else:
            data = img.GetData().reshape(self.height, self.width)

        img.Release()
        return data.astype(np.float32)

    def grab_average(self, n_frames=8, timeout_ms=100):
        """
        Grab and average multiple frames (for saving stills).
        """
        acc = np.zeros((self.height, self.width), dtype=np.float32)
        for _ in range(n_frames):
            acc += self.grab_frame(timeout_ms=timeout_ms)
        return acc / float(n_frames)


# ---------- utility functions ----------


def normalize_fast_to_u8(img):
    """
    Fast normalization using min/max (no percentiles).
    Used for low-latency preview.
    """
    mn = float(img.min())
    mx = float(img.max())
    if mx <= mn:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - mn) * (255.0 / (mx - mn))
    return out.astype(np.uint8)


def normalize_to_u8(img, clip_percent=1.0):
    """
    Slower normalization using percentiles.
    Used for saved still images (better contrast).
    """
    low, high = np.percentile(img, [clip_percent, 100.0 - clip_percent])
    img_c = np.clip(img, low, high)
    img_n = (img_c - low) / (high - low + 1e-9)
    return (255.0 * img_n).astype(np.uint8)


def create_video_writer(path, width, height, fps=30.0):
    """
    Create an OpenCV VideoWriter for AVI output (grayscale).
    """
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height), isColor=False)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    return writer


# ---------- preview + diff + ROI + AVI ----------


def preview_loop(cam,
                 out_dir,
                 n_avg=8,
                 video_fps=30.0,
                 flush_extra_frames=1):
    """
    Live preview with:

    - Raw preview window
    - Difference window (current - previous)
    - 'r' : select ROI (software crop)
    - 's' : save averaged still (ROI or full frame)
    - 'v' : start/stop AVI recording (diff frames only)
    - 'q' : quit

    Low-latency:
    - NewestOnly stream mode
    - Fast min/max normalization
    - Optional extra frame flush per loop
    """
    print("Preview running.")
    print("Keys:")
    print("  's' : save an averaged still image (ROI if selected)")
    print("  'v' : start/stop AVI recording (difference frames only)")
    print("  'r' : select ROI (software cropping)")
    print("  'q' : quit")

    roi = None           # None -> full frame, otherwise (x, y, w, h)
    prev_frame = None    # For frame subtraction
    writer = None        # cv2.VideoWriter (None when not recording)
    recording = False
    capture_index = 0

    while True:
        # Base frame; on timeout, skip and keep UI alive
        try:
            frame = cam.grab_frame(timeout_ms=150)
        except RuntimeError as e:
            print("Frame grab error (skipping frame):", e)
            continue

        # Optional: flush extra frames to stay near real-time
        for _ in range(flush_extra_frames):
            try:
                frame = cam.grab_frame(timeout_ms=1)
            except RuntimeError:
                break

        # ROI crop if defined
        if roi is not None:
            x, y, w, h = roi
            frame_roi = frame[y:y + h, x:x + w]
        else:
            frame_roi = frame

        # Frame subtraction
        diff = None
        if prev_frame is not None:
            if roi is not None:
                prev_roi = prev_frame[y:y + h, x:x + w]
            else:
                prev_roi = prev_frame
            diff = frame_roi - prev_roi

        prev_frame = frame.copy()

        # Display
        raw_u8 = normalize_fast_to_u8(frame_roi)
        cv2.imshow("Blackfly Raw (ROI)", raw_u8)

        if diff is not None:
            diff_u8 = normalize_fast_to_u8(diff)
            cv2.imshow("Blackfly Diff (ROI)", diff_u8)

        # Recording (difference frames only)
        if recording and diff is not None:
            frame_to_write = normalize_fast_to_u8(diff)
            writer.write(frame_to_write)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            print(f"Capturing {n_avg} frames for averaging...")
            avg = cam.grab_average(n_frames=n_avg, timeout_ms=150)
            if roi is not None:
                x, y, w, h = roi
                avg = avg[y:y + h, x:x + w]
            avg_u8 = normalize_to_u8(avg)
            filename = out_dir / f"still_avg_{capture_index:03d}.png"
            cv2.imwrite(str(filename), avg_u8)
            print(f"Saved still image: {filename}")
            capture_index += 1

        elif key == ord('v'):
            if not recording:
                # Start recording
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                avi_path = out_dir / f"blackfly_diff_{timestamp}.avi"

                if roi is not None:
                    x, y, w, h = roi
                    width, height = w, h
                else:
                    width, height = cam.width, cam.height

                writer = create_video_writer(avi_path, width, height, fps=video_fps)
                recording = True
                print(f"Started recording AVI (difference frames): {avi_path}")
            else:
                # Stop recording
                recording = False
                if writer is not None:
                    writer.release()
                    writer = None
                print("Stopped recording AVI.")

        elif key == ord('r'):
            print("Select ROI in 'Blackfly Raw (ROI)' window, then press Enter or Space.")
            print("Press 'c' to cancel.")
            r = cv2.selectROI("Blackfly Raw (ROI)", raw_u8, fromCenter=False, showCrosshair=True)
            if r != (0, 0, 0, 0):
         q       x, y, w, h = map(int, r)
                roi = (x, y, w, h)
                print(f"ROI set to x={x}, y={y}, w={w}, h={h}")
            else:
                print("ROI selection cancelled or invalid.")

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main():
    # User parameters
    serial = None           # or your camera serial as string
    exposure_us = 3000.0    # lower → less inherent latency
    gain_db = 0.0
    n_avg = 8               # frames for still averaging
    video_fps = 30.0
    flush_extra_frames = 1  # 0, 1, or 2

    out_dir = Path("blackfly_output")
    out_dir.mkdir(exist_ok=True)

    cam = BlackflyCamera(serial=serial,
                         exposure_us=exposure_us,
                         gain_db=gain_db)

    try:
        cam.start()
        print("Streaming. Press 'q' to quit.")
        preview_loop(cam,
                     out_dir=out_dir,
                     n_avg=n_avg,
                     video_fps=video_fps,
                     flush_extra_frames=flush_extra_frames)
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
