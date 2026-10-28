"""
q

FLIR Blackfly (Spinnaker / PySpin) script with:

- Camera initialization
- Live preview (raw and difference frames)
- Low-latency settings (NewestOnly buffer mode, fast normalization, optional flush)
- Frame-to-frame subtraction (current - previous)
- Software ROI (user-selectable via 'r')
- Averaged still-image capture ('s')
- AVI recording of difference frames only ('v')
- UI controls for exposure and gain (OpenCV trackbars)

Deliberately omitted:
- Trigger/tone control
- ESPI phase stepping
- Wrapped phase / modulation / pseudocolor
"""

import time
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import PySpin


# Global camera handle for trackbar callbacks
_cam_for_ui = None


class BlackflyCamera:
    """
    Simple wrapper around a FLIR Blackfly camera using PySpin.

    Responsibilities:
    - Connect to the first camera (or one by serial)
    - Configure basic exposure and gain
    - Set low-latency stream mode
    - Start/stop acquisition safely
    - Grab single frames or averaged frames
    - Provide methods to adjust exposure/gain at runtime
    """

    def __init__(self, serial=None, exposure_us=3000.0, gain_db=0.0):
        """
        Initialize system and camera.

        Parameters
        ----------
        serial : str or None
            Camera serial number to select a specific camera.
            If None, the first detected camera is used.
        exposure_us : float
            Initial exposure time in microseconds.
        gain_db : float
            Initial gain value in dB.
        """
        self.streaming = False

        # Get the global PySpin system object
        self.system = PySpin.System.GetInstance()

        # Enumerate all connected cameras
        self.cam_list = self.system.GetCameras()
        if self.cam_list.GetSize() == 0:
            self.cam_list.Clear()
            self.system.ReleaseInstance()
            raise RuntimeError("No Spinnaker/PySpin cameras detected.")

        # Select camera by serial or just take the first one
        if serial is None:
            self.cam = self.cam_list[0]
        else:
            self.cam = self.cam_list.GetBySerial(serial)

        # Initialize the camera
        self.cam.Init()

        # ------------------ low-latency stream setup ------------------
        s_nodemap = self.cam.GetTLStreamNodeMap()
        handling_mode = PySpin.CEnumerationPtr(s_nodemap.GetNode("StreamBufferHandlingMode"))
        if PySpin.IsReadable(handling_mode) and PySpin.IsWritable(handling_mode):
            newest_only = handling_mode.GetEntryByName("NewestOnly")
            if PySpin.IsReadable(newest_only):
                handling_mode.SetIntValue(newest_only.GetValue())
        # Optional: manual buffer count (uncomment if desired)
        # buffer_count_mode = PySpin.CEnumerationPtr(s_nodemap.GetNode("StreamBufferCountMode"))
        # if PySpin.IsReadable(buffer_count_mode) and PySpin.IsWritable(buffer_count_mode):
        #     manual = buffer_count_mode.GetEntryByName("Manual")
        #     if PySpin.IsReadable(manual):
        #         buffer_count_mode.SetIntValue(manual.GetValue())
        #         buffer_count = PySpin.CIntegerPtr(s_nodemap.GetNode("StreamBufferCountManual"))
        #         if PySpin.IsWritable(buffer_count):
        #             buffer_count.SetValue(4)

        # ------------------ acquisition and exposure/gain ------------------
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

        # (Optional) hardware ROI – uncomment and adjust if you want a smaller frame:
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

        # Cache image dimensions for reshaping image data
        self.height = self.cam.Height.GetValue()
        self.width = self.cam.Width.GetValue()

    # ----------------- camera control methods -----------------

    def set_exposure_us(self, exposure_us):
        """Set exposure in microseconds (clamped to camera limits)."""
        if self.exp_node is not None and PySpin.IsWritable(self.exp_node):
            val = float(np.clip(exposure_us, self.exp_min, self.exp_max))
            self.exp_node.SetValue(val)

    def set_gain_db(self, gain_db):
        """Set analog gain in dB (clamped to camera limits)."""
        if self.gain_node is not None and PySpin.IsWritable(self.gain_node):
            val = float(np.clip(gain_db, self.gain_min, self.gain_max))
            self.gain_node.SetValue(val)

    def exposure_from_slider(self, slider_val, slider_max=100):
        """Map integer slider value [0, slider_max] to exposure range [exp_min, exp_max]."""
        alpha = slider_val / float(slider_max)
        exposure_us = self.exp_min + alpha * (self.exp_max - self.exp_min)
        self.set_exposure_us(exposure_us)

    def gain_from_slider(self, slider_val, slider_max=100):
        """Map integer slider value [0, slider_max] to gain range [gain_min, gain_max]."""
        alpha = slider_val / float(slider_max)
        gain_db = self.gain_min + alpha * (self.gain_max - self.gain_min)
        self.set_gain_db(gain_db)

    # ----------------- acquisition methods -----------------

    def start(self):
        """Begin continuous acquisition."""
        if self.streaming:
            return
        try:
            self.cam.BeginAcquisition()
            self.streaming = True
        except PySpin.SpinnakerException as e:
            msg = (
                f"BeginAcquisition failed: {e}\n"
                "Check USB cable, port, and power settings; try unplug/replug the camera "
                "or a different USB3 port."
            )
            raise RuntimeError(msg) from e

    def stop(self):
        """Stop continuous acquisition if it was started."""
        if not self.streaming:
            return
        try:
            self.cam.EndAcquisition()
        except PySpin.SpinnakerException:
            # Ignore 'not streaming' errors on shutdown
            pass
        finally:
            self.streaming = False

    def close(self):
        """
        Deinitialize the camera and release system resources.
        """
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
        Grab a single Mono8 frame as a float32 NumPy array, with timeout.

        Parameters
        ----------
        timeout_ms : int
            Milliseconds to wait for a frame.

        Returns
        -------
        frame : np.ndarray (float32)
            2D array of shape (height, width) with intensity values.
        """
        try:
            img = self.cam.GetNextImage(timeout_ms)
        except PySpin.SpinnakerException as e:
            raise RuntimeError(f"GetNextImage timeout or error: {e}") from e

        # Check for incomplete images
        if img.IsIncomplete():
            status = img.GetImageStatus()
            img.Release()
            raise RuntimeError(f"Incomplete image: status {status}")

        # Convert to Mono8 if camera is in another pixel format
        if img.GetPixelFormat() != PySpin.PixelFormat_Mono8:
            conv = img.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
            data = conv.GetData().reshape(self.height, self.width)
        else:
            data = img.GetData().reshape(self.height, self.width)

        img.Release()

        return data.astype(np.float32)

    def grab_average(self, n_frames=8, timeout_ms=100):
        """
        Grab and average multiple frames.

        This corresponds to the "NUMBER OF FRAMES" averaging in your LabVIEW VI.

        Parameters
        ----------
        n_frames : int
            Number of frames to average.
        timeout_ms : int
            Timeout per frame in milliseconds.

        Returns
        -------
        avg : np.ndarray (float32)
            Averaged image.
        """
        acc = np.zeros((self.height, self.width), dtype=np.float32)
        for _ in range(n_frames):
            acc += self.grab_frame(timeout_ms=timeout_ms)
        return acc / float(n_frames)


# ------------- utility functions -------------


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
    Slower normalization using percentile clipping.
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


# ------------- trackbar callbacks -------------


def on_exposure_trackbar(val):
    """Trackbar callback to update exposure."""
    global _cam_for_ui
    if _cam_for_ui is not None:
        _cam_for_ui.exposure_from_slider(val)


def on_gain_trackbar(val):
    """Trackbar callback to update gain."""
    global _cam_for_ui
    if _cam_for_ui is not None:
        _cam_for_ui.gain_from_slider(val)


# ------------- preview and control loop -------------


def preview_loop(cam,
                 out_dir,
                 n_avg=8,
                 video_fps=30.0,
                 flush_extra_frames=1):
    """
    Live preview loop using OpenCV.

    Features:
    - Real-time preview of raw frames (optionally cropped by ROI).
    - Frame subtraction: diff = current_frame - previous_frame (ROI or full frame).
    - 'r' to select ROI (software cropping).
    - 's' to save an averaged still (ROI or full frame).
    - 'v' to start/stop AVI recording (difference frames only, ROI or full frame).
    - 'q' to quit.

    Low-latency tweaks:
    - StreamBufferHandlingMode = NewestOnly
    - Fast min/max normalization
    - Optional flush of extra frames per loop (flush_extra_frames)
    """
    print("Preview running.")
    print("Keys:")
    print("  's' : save an averaged still image (ROI if selected)")
    print("  'v' : start/stop AVI recording (difference frames only)")
    print("  'r' : select ROI (software cropping)")
    print("  'q' : quit")

    roi = None           # None -> full frame, otherwise (x, y, w, h)
    prev_frame = None    # For frame subtraction (full frame)
    writer = None        # cv2.VideoWriter (None when not recording)
    recording = False
    capture_index = 0    # For numbering still image files

    while True:
        # Grab a frame; on timeout, skip this iteration instead of exiting
        try:
            frame = cam.grab_frame(timeout_ms=150)  # you can raise to 200–250 if needed
        except RuntimeError as e:
            print("Frame grab error (skipping frame):", e)
            continue

        # Optionally flush extra frames to keep latency low
        for _ in range(flush_extra_frames):
            try:
                frame = cam.grab_frame(timeout_ms=1)
            except RuntimeError:
                # If we can't get an extra frame quickly, just use the last good one
                break

        # Apply ROI crop if defined
        if roi is not None:
            x, y, w, h = roi
            frame_roi = frame[y:y + h, x:x + w]
        else:
            frame_roi = frame

        # Compute difference to previous frame, if available
        diff = None
        if prev_frame is not None:
            if roi is not None:
                prev_roi = prev_frame[y:y + h, x:x + w]
            else:
                prev_roi = prev_frame
            diff = frame_roi - prev_roi

        prev_frame = frame.copy()

        # Fast normalization for display
        raw_u8 = normalize_fast_to_u8(frame_roi)
        cv2.imshow("Blackfly Raw (ROI)", raw_u8)

        if diff is not None:
            diff_u8 = normalize_fast_to_u8(diff)
            cv2.imshow("Blackfly Diff (ROI)", diff_u8)

        # If recording, write ONLY difference frames (if diff available)
        if recording and diff is not None:
            frame_to_write = normalize_fast_to_u8(diff)
            writer.write(frame_to_write)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Quit preview
            break

        elif key == ord('s'):
            # Save an averaged still image (ROI if selected)
            print(f"Capturing {n_avg} frames for averaging...")
            avg = cam.grab_average(n_frames=n_avg, timeout_ms=150)

            if roi is not None:
                x, y, w, h = roi
                avg = avg[y:y + h, x:x + w]

            # Use percentile-based normalization for nicer saved images
            avg_u8 = normalize_to_u8(avg)
            filename = out_dir / f"still_avg_{capture_index:03d}.png"
            cv2.imwrite(str(filename), avg_u8)
            print(f"Saved still image: {filename}")
            capture_index += 1

        elif key == ord('v'):
            # Toggle video recording on/off (difference frames only)
            if not recording:
                # Start recording
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                avi_path = out_dir / f"blackfly_diff_{timestamp}.avi"

                if roi is not None:
                    x, y, w, h = roi
                    width, height = w, h
                else:
                    width, height = cam.width, cam.height

                writer = create_video_writer(
                    avi_path,
                    width=width,
                    height=height,
                    fps=video_fps
                )
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
            # Select ROI interactively on the latest raw image
            print("Select ROI in the 'Blackfly Raw (ROI)' window, then press Enter or Space.")
            print("Press 'c' to cancel.")
            r = cv2.selectROI("Blackfly Raw (ROI)", raw_u8, fromCenter=False, showCrosshair=True)
            if r != (0, 0, 0, 0):
                x, y, w, h = map(int, r)
                roi = (x, y, w, h)
                print(f"ROI set to x={x}, y={y}, w={w}, h={h}")
            else:
                print("ROI selection cancelled or invalid.")

    # Clean up windows and writer (if still open)
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main():
    """
    Main entry point.

    - Creates output directory
    - Connects to the camera
    - Creates UI window with exposure/gain controls
    - Starts acquisition
    - Runs preview loop (with ROI, frame subtraction, AVI recording)
    - Cleans up resources on exit
    """
    global _cam_for_ui

    # User-adjustable parameters
    serial = None           # Set to camera serial string to select a specific unit
    exposure_us = 3000.0    # Initial exposure time in microseconds
    gain_db = 0.0           # Initial gain in dB
    n_avg = 8               # Frames to average when saving stills
    video_fps = 30.0        # AVI frame rate (metadata only)
    flush_extra_frames = 1  # 0, 1, or 2 – higher = lower latency, more CPU
    out_dir = Path("blackfly_output")

    out_dir.mkdir(exist_ok=True)

    # Initialize camera
    cam = BlackflyCamera(serial=serial,
                         exposure_us=exposure_us,
                         gain_db=gain_db)
    _cam_for_ui = cam  # expose to trackbar callbacks

    try:
        # Start streaming first; if this fails, do not create UI
        cam.start()

        # Create control window with exposure/gain sliders
        cv2.namedWindow("Controls", cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Exposure", "Controls", 50, 100, on_exposure_trackbar)
        cv2.createTrackbar("Gain", "Controls", 0, 100, on_gain_trackbar)

        # Enter the main preview loop
        preview_loop(cam,
                     out_dir=out_dir,
                     n_avg=n_avg,
                     video_fps=video_fps,
                     flush_extra_frames=flush_extra_frames)

    except RuntimeError as e:
        print(e)
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
