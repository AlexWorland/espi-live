"""
blackfly_diff_avi.py

FLIR Blackfly (Spinnaker / PySpin) script with:

- Camera initialization
- Live preview (OpenCV)
- Frame-to-frame subtraction (current - previous)
- Option to save averaged still images
- Option to record an AVI movie

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


class BlackflyCamera:
    """
    Simple wrapper around a FLIR Blackfly camera using PySpin.

    Responsibilities:
    - Connect to the first camera (or one by serial)
    - Configure basic exposure and gain
    - Start/stop acquisition
    - Grab single frames or averaged frames
    """

    def __init__(self, serial=None, exposure_us=5000.0, gain_db=0.0):
        """
        Initialize system and camera.

        Parameters
        ----------
        serial : str or None
            Camera serial number to select a specific camera.
            If None, the first detected camera is used.
        exposure_us : float
            Exposure time in microseconds.
        gain_db : float
            Gain value in dB.
        """
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

        # Set acquisition mode to continuous
        node = self.cam.AcquisitionMode.GetEntry(PySpin.AcquisitionMode_Continuous)
        self.cam.AcquisitionMode.SetIntValue(node.GetValue())

        # Disable auto exposure and auto gain to get manual control
        if self.cam.ExposureAuto.GetAccessMode() == PySpin.RW:
            self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        if self.cam.GainAuto.GetAccessMode() == PySpin.RW:
            self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)

        # Set manual exposure time (bounded by camera limits)
        if self.cam.ExposureTime.GetAccessMode() == PySpin.RW:
            min_exp = self.cam.ExposureTime.GetMin()
            max_exp = self.cam.ExposureTime.GetMax()
            self.cam.ExposureTime.SetValue(
                np.clip(exposure_us, min_exp, max_exp)
            )

        # Set manual gain (bounded by camera limits)
        if self.cam.Gain.GetAccessMode() == PySpin.RW:
            min_gain = self.cam.Gain.GetMin()
            max_gain = self.cam.Gain.GetMax()
            self.cam.Gain.SetValue(
                np.clip(gain_db, min_gain, max_gain)
            )

        # Cache image dimensions for reshaping image data
        self.height = self.cam.Height.GetValue()
        self.width = self.cam.Width.GetValue()

    def start(self):
        """Begin continuous acquisition."""
        self.cam.BeginAcquisition()

    def stop(self):
        """Stop continuous acquisition."""
        self.cam.EndAcquisition()

    def close(self):
        """
        Deinitialize the camera and release system resources.

        Call this at the end of your program to clean up.
        """
        self.cam.DeInit()
        del self.cam
        self.cam_list.Clear()
        self.system.ReleaseInstance()

    def grab_frame(self, sleep_ms=0.0):
        """
        Grab a single Mono8 frame as a float32 NumPy array.

        Parameters
        ----------
        sleep_ms : float
            Optional delay before grabbing, in milliseconds.

        Returns
        -------
        frame : np.ndarray (float32)
            2D array of shape (height, width) with intensity values.
        """
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

        img = self.cam.GetNextImage()

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

        # Return as float32 for easier math later
        return data.astype(np.float32)

    def grab_average(self, n_frames=8, sleep_ms=0.0):
        """
        Grab and average multiple frames.

        This corresponds to the "NUMBER OF FRAMES" averaging in your LabVIEW VI.

        Parameters
        ----------
        n_frames : int
            Number of frames to average.
        sleep_ms : float
            Delay between frames, in milliseconds.

        Returns
        -------
        avg : np.ndarray (float32)
            Averaged image.
        """
        acc = np.zeros((self.height, self.width), dtype=np.float32)
        for _ in range(n_frames):
            acc += self.grab_frame(sleep_ms)
        return acc / float(n_frames)


def normalize_to_u8(img, clip_percent=1.0):
    """
    Normalize a float image to 0–255 uint8 for display/saving.

    Uses percentile clipping to reduce the influence of outliers.
    """
    low, high = np.percentile(img, [clip_percent, 100.0 - clip_percent])
    img_c = np.clip(img, low, high)
    img_n = (img_c - low) / (high - low + 1e-9)
    return (255.0 * img_n).astype(np.uint8)


def create_video_writer(path, width, height, fps=30.0, is_color=False):
    """
    Create an OpenCV VideoWriter for AVI output.

    Parameters
    ----------
    path : str or Path
        Path to the AVI file.
    width, height : int
        Frame dimensions.
    fps : float
        Frames per second.
    is_color : bool
        Whether frames are color (3-channel) or grayscale (1-channel).

    Returns
    -------
    writer : cv2.VideoWriter
    """
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height), isColor=is_color)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    return writer


def preview_loop(cam,
                 out_dir,
                 n_avg=8,
                 sleep_ms=0.0,
                 record_diff=True,
                 preview_diff=True,
                 video_fps=30.0):
    """
    Live preview loop using OpenCV.

    Features:
    - Real-time preview of raw frames and (optionally) difference frames.
    - Frame subtraction: diff = current_frame - previous_frame.
    - Press 's' to save an averaged still image to PNG.
    - Press 'v' to start/stop AVI recording.
      (Records either raw or diff frames, depending on record_diff flag.)
    - Press 'q' to quit.

    Parameters
    ----------
    cam : BlackflyCamera
        Initialized camera object.
    out_dir : pathlib.Path
        Directory where captured images and videos will be saved.
    n_avg : int
        Number of frames to average when saving stills.
    sleep_ms : float
        Delay between frames.
    record_diff : bool
        If True, the AVI records difference frames.
        If False, the AVI records raw frames.
    preview_diff : bool
        If True, show a separate window with the difference image.
    video_fps : float
        Frame rate used for AVI file metadata.
    """
    print("Preview running.")
    print("Keys:")
    print("  's' : save an averaged still image")
    print("  'v' : start/stop AVI recording")
    print("  'q' : quit")

    prev_frame = None      # For frame subtraction
    writer = None          # cv2.VideoWriter (None when not recording)
    recording = False
    capture_index = 0      # For numbering still image files

    while True:
        # Grab current frame
        frame = cam.grab_frame(sleep_ms)

        # Compute difference to previous frame, if available
        diff = None
        if prev_frame is not None:
            # Simple difference; you can also take abs() if you prefer
            diff = frame - prev_frame

        prev_frame = frame.copy()

        # Normalize for display
        raw_u8 = normalize_to_u8(frame)
        cv2.imshow("Blackfly Raw", raw_u8)

        if preview_diff and diff is not None:
            diff_u8 = normalize_to_u8(diff)
            cv2.imshow("Blackfly Diff", diff_u8)

        # If recording, choose what to write (raw or diff)
        if recording:
            if record_diff and diff is not None:
                frame_to_write = normalize_to_u8(diff)
            else:
                frame_to_write = raw_u8

            # For grayscale, VideoWriter expects single-channel or 3-channel.
            # Here we pass single-channel frames, having isColor=False.
            writer.write(frame_to_write)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Quit preview
            break

        elif key == ord('s'):
            # Save an averaged still image
            print(f"Capturing {n_avg} frames for averaging...")
            avg = cam.grab_average(n_frames=n_avg, sleep_ms=sleep_ms)
            avg_u8 = normalize_to_u8(avg)
            filename = out_dir / f"still_avg_{capture_index:03d}.png"
            cv2.imwrite(str(filename), avg_u8)
            print(f"Saved still image: {filename}")
            capture_index += 1

        elif key == ord('v'):
            # Toggle video recording on/off
            if not recording:
                # Start recording
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                mode_str = "diff" if record_diff else "raw"
                avi_path = out_dir / f"blackfly_{mode_str}_{timestamp}.avi"
                writer = create_video_writer(
                    avi_path,
                    width=cam.width,
                    height=cam.height,
                    fps=video_fps,
                    is_color=False
                )
                recording = True
                print(f"Started recording AVI: {avi_path}")
            else:
                # Stop recording
                recording = False
                if writer is not None:
                    writer.release()
                    writer = None
                print("Stopped recording AVI.")

    # Clean up windows and writer (if still open)
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main():
    """
    Main entry point.

    - Creates output directory
    - Connects to the camera
    - Starts acquisition
    - Runs preview loop with frame subtraction and AVI option
    - Cleans up resources on exit
    """
    # User-adjustable parameters
    serial = None           # Set to camera serial string to select a specific unit
    exposure_us = 5000.0    # Exposure time in microseconds
    gain_db = 0.0           # Gain in dB
    n_avg = 8               # Frames to average when saving stills
    interframe_ms = 0.0     # Delay between frames
    record_diff = True      # If True: AVI records diff frames; False: raw frames
    preview_diff = True     # If True: show separate window with diff image
    video_fps = 30.0        # AVI frame rate (metadata only)
    out_dir = Path("blackfly_output")

    out_dir.mkdir(exist_ok=True)

    cam = BlackflyCamera(serial=serial,
                         exposure_us=exposure_us,
                         gain_db=gain_db)

    try:
        cam.start()
        preview_loop(cam,
                     out_dir=out_dir,
                     n_avg=n_avg,
                     sleep_ms=interframe_ms,
                     record_diff=record_diff,
                     preview_diff=preview_diff,
                     video_fps=video_fps)
    finally:
        cam.stop()
        cam.close()


if __name__ == "__main__":
    main()
