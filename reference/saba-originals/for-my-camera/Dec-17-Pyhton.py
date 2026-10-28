"""
Blackfly ESPI live viewer (BFS-U3-63S4M, Mono8) with two ESPI modes:

1) Frame-to-frame ESPI: abs(current - last)
2) Reference ESPI:      abs(current - reference)

Controls (OpenCV window must be focused):
  q  quit
  s  save displayed frame (PNG)
  r  reset "last frame" (frame-to-frame mode)
  a  reset averaging accumulator
  f  capture reference frame (reference mode uses this)

Trackbars:
  Exposure us
  Gain x10 dB
  FPS x10
  Avg N (1=off)
  Palette (0=gray, 1-6=colormaps)
  Gamma x10 (100=1.0, lower=darker, higher=brighter)
  ESPI mode (0=prev, 1=ref)
  Wait ms
  Record AVI (0/1)
"""

import os
import time
from datetime import datetime

import cv2
import numpy as np
import PySpin


def ts():
    """
    Generate a timestamp string in the format YYYYMMDD_HHMMSS_microseconds.
    
    Returns:
        str: Timestamp string suitable for use in filenames.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def ensure_dir(path: str) -> str:
    """
    Create a directory if it doesn't exist.
    
    Args:
        path: Directory path to create.
        
    Returns:
        str: The same path that was passed in.
    """
    os.makedirs(path, exist_ok=True)
    return path


def np_from_imageptr_mono8(img: PySpin.ImagePtr) -> np.ndarray:
    """
    ImagePtr (Mono8) -> numpy uint8 image (H,W), copied so buffer can be released.
    Optimized to reduce memory allocations.
    """
    w = img.GetWidth()
    h = img.GetHeight()
    data = img.GetData()
    # Use frombuffer for faster conversion, then copy to ensure we own the memory
    return np.frombuffer(data, dtype=np.uint8, count=w*h).reshape(h, w).copy()


def set_enum(nodemap, node_name: str, entry_name: str) -> None:
    """
    Set an enumeration node value in the camera nodemap.
    
    Args:
        nodemap: PySpin nodemap object.
        node_name: Name of the enumeration node to set.
        entry_name: Name of the enumeration entry to set.
        
    Returns:
        None: Silently returns if the node is not readable/writable or entry not found.
    """
    node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
    if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
        return
    entry = node.GetEntryByName(entry_name)
    if not PySpin.IsReadable(entry):
        return
    node.SetIntValue(entry.GetValue())


def set_bool(nodemap, node_name: str, value: bool) -> None:
    """
    Set a boolean node value in the camera nodemap.
    
    Args:
        nodemap: PySpin nodemap object.
        node_name: Name of the boolean node to set.
        value: Boolean value to set.
        
    Returns:
        None: Silently returns if the node is not writable.
    """
    node = PySpin.CBooleanPtr(nodemap.GetNode(node_name))
    if PySpin.IsWritable(node):
        node.SetValue(bool(value))


def set_float(nodemap, node_name: str, value: float) -> None:
    """
    Set a float node value in the camera nodemap, clamping to valid range.
    
    Args:
        nodemap: PySpin nodemap object.
        node_name: Name of the float node to set.
        value: Float value to set (will be clamped to node's min/max range).
        
    Returns:
        None: Silently returns if the node is not writable.
    """
    node = PySpin.CFloatPtr(nodemap.GetNode(node_name))
    if PySpin.IsWritable(node):
        vmin = node.GetMin()
        vmax = node.GetMax()
        node.SetValue(float(np.clip(value, vmin, vmax)))


class ESPIProcessor:
    def __init__(self):
        """
        Initialize the ESPI processor with empty state for previous frame, reference frame,
        and averaging accumulator.
        """
        self.prev = None
        self.ref = None
        self.avg_accum = None
        self.avg_count = 0

    def reset_prev(self):
        """
        Reset the previous frame buffer used for frame-to-frame ESPI mode.
        Next frame will become the new baseline.
        """
        self.prev = None

    def reset_avg(self):
        """
        Reset the temporal averaging accumulator and counter.
        Clears accumulated frames for averaging calculation.
        """
        self.avg_accum = None
        self.avg_count = 0

    def set_reference(self, frame8: np.ndarray):
        """
        Set the reference frame for reference ESPI mode.
        
        Args:
            frame8: 8-bit grayscale numpy array to use as the reference frame.
        """
        self.ref = frame8.copy()

    def diff_prev(self, curr8: np.ndarray) -> np.ndarray:
        """
        Compute frame-to-frame ESPI difference: absolute difference between current and previous frame.
        
        Args:
            curr8: Current 8-bit grayscale frame.
            
        Returns:
            np.ndarray: Absolute difference image (uint8). Returns zeros if no previous frame exists.
        """
        if self.prev is None:
            self.prev = curr8.copy()
            return np.zeros_like(curr8)
        diff = cv2.absdiff(curr8, self.prev)
        self.prev = curr8.copy()
        return diff

    def diff_ref(self, curr8: np.ndarray) -> np.ndarray:
        """
        Compute reference ESPI difference: absolute difference between current frame and reference frame.
        
        Args:
            curr8: Current 8-bit grayscale frame.
            
        Returns:
            np.ndarray: Absolute difference image (uint8). Returns zeros if no reference frame has been set.
        """
        if self.ref is None:
            # No reference captured yet -> show zeros
            return np.zeros_like(curr8)
        diff = cv2.absdiff(curr8, self.ref)
        return diff

    def avg_over_n(self, img8: np.ndarray, n: int) -> np.ndarray:
        """
        Compute temporal average over N frames. Accumulates frames and returns running average.
        Resets accumulator when N frames have been accumulated.
        Applies fast min-max normalization for low-latency display when averaging is off,
        or percentile-based normalization when averaging is enabled.
        
        Args:
            img8: 8-bit grayscale frame to add to the average.
            n: Number of frames to average over. If n <= 1, returns normalized input.
            
        Returns:
            np.ndarray: Normalized averaged 8-bit grayscale image. Returns normalized input if n <= 1.
        """
        if n <= 1:
            self.reset_avg()
            # Fast min-max normalization (much faster than percentile for real-time)
            img_min = img8.min()
            img_max = img8.max()
            if img_max > img_min:
                # Use cv2.normalize for optimized normalization
                return cv2.normalize(img8, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            else:
                return img8

        if self.avg_accum is None:
            self.avg_accum = img8.astype(np.float32)
            self.avg_count = 1
        else:
            self.avg_accum += img8.astype(np.float32)
            self.avg_count += 1

        out = self.avg_accum / float(self.avg_count)
        out8 = np.clip(out, 0, 255).astype(np.uint8)

        # Fast min-max normalization for averaged frames (percentiles too slow for real-time)
        # For averaged frames, min-max is usually sufficient and much faster
        out_min = out8.min()
        out_max = out8.max()
        if out_max > out_min:
            out8 = cv2.normalize(out8, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        if self.avg_count >= n:
            self.reset_avg()

        return out8


def main():
    """
    Main function that initializes the FLIR camera, sets up the ESPI viewer with OpenCV,
    and runs the live processing loop. Handles camera acquisition, ESPI processing,
    UI controls, and video recording.
    """
    out_dir = ensure_dir(os.path.join(os.getcwd(), "espi_output"))

    # -------------------------
    # PySpin init
    # -------------------------
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() < 1:
        cam_list.Clear()
        system.ReleaseInstance()
        raise RuntimeError("No FLIR/Spinnaker cameras detected.")

    cam = cam_list.GetByIndex(0)
    cam.Init()
    nodemap = cam.GetNodeMap()

    # Stable ESPI settings (Mono8, manual exposure/gain, continuous)
    try:
        set_enum(nodemap, "PixelFormat", "Mono8")
    except PySpin.SpinnakerException:
        pass
    try:
        set_enum(nodemap, "AcquisitionMode", "Continuous")
    except PySpin.SpinnakerException:
        pass
    try:
        set_enum(nodemap, "ExposureAuto", "Off")
    except PySpin.SpinnakerException:
        pass
    try:
        set_enum(nodemap, "GainAuto", "Off")
    except PySpin.SpinnakerException:
        pass

    # Start values (adjust live with sliders)
    try:
        set_float(nodemap, "ExposureTime", 8000.0)  # us
    except PySpin.SpinnakerException:
        pass
    try:
        set_float(nodemap, "Gain", 0.0)  # dB
    except PySpin.SpinnakerException:
        pass

    # Reduce lag: newest-only buffer handling
    try:
        s_node_map = cam.GetTLStreamNodeMap()
        set_enum(s_node_map, "StreamBufferHandlingMode", "NewestOnly")
    except PySpin.SpinnakerException:
        pass

    # Enable frame rate control if supported
    try:
        set_bool(nodemap, "AcquisitionFrameRateEnable", True)
        set_float(nodemap, "AcquisitionFrameRate", 30.0)
    except PySpin.SpinnakerException:
        pass

    # -------------------------
    # OpenCV UI
    # -------------------------
    win = "Blackfly ESPI (prev/ref)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def _noop(_):
        """
        No-op callback function for OpenCV trackbars (trackbars don't require callbacks).
        
        Args:
            _: Trackbar position value (unused).
        """
        return

    cv2.createTrackbar("Exposure us", win, 8000, 50000, _noop)
    cv2.createTrackbar("Gain x10 dB", win, 0, 300, _noop)          # 0..30.0 dB
    cv2.createTrackbar("FPS x10", win, 300, 600, _noop)            # 30.0..60.0 fps (clamped)
    cv2.createTrackbar("Avg N (1=off)", win, 1, 60, _noop)
    cv2.createTrackbar("Palette (0=gray,1-6)", win, 0, 6, _noop)  # 0=gray, 1=TURBO, 2=JET, 3=HOT, 4=VIRIDIS, 5=PLASMA, 6=INFERNO
    cv2.createTrackbar("Gamma x10", win, 100, 300, _noop)          # 10-300, 100=gamma 1.0
    cv2.createTrackbar("ESPI mode (0=prev,1=ref)", win, 0, 1, _noop)
    cv2.createTrackbar("Wait ms", win, 1, 200, _noop)  # Default 1ms for minimal delay
    cv2.createTrackbar("Record AVI (0/1)", win, 0, 1, _noop)

    proc = ESPIProcessor()
    writer = None
    last_param_write = 0.0
    last_gamma_val = -1.0
    gamma_lut = None

    cam.BeginAcquisition()

    try:
        print("Keys: q quit | s save | r reset last | a reset avg | f set reference")
        while True:
            img = cam.GetNextImage(100)  # Reduced timeout for faster frame skipping
            try:
                if img.IsIncomplete():
                    continue
                frame8 = np_from_imageptr_mono8(img)
            finally:
                img.Release()

            # Write camera params at ~5 Hz (avoid hammering nodemap)
            now = time.time()
            if now - last_param_write > 0.20:
                last_param_write = now
                exp_us = float(cv2.getTrackbarPos("Exposure us", win))
                gain_db = float(cv2.getTrackbarPos("Gain x10 dB", win)) / 10.0
                fps = float(cv2.getTrackbarPos("FPS x10", win)) / 10.0

                try:
                    set_float(nodemap, "ExposureTime", exp_us)
                except PySpin.SpinnakerException:
                    pass
                try:
                    set_float(nodemap, "Gain", gain_db)
                except PySpin.SpinnakerException:
                    pass
                try:
                    set_bool(nodemap, "AcquisitionFrameRateEnable", True)
                    set_float(nodemap, "AcquisitionFrameRate", fps)
                except PySpin.SpinnakerException:
                    pass

            # ESPI mode selection (maps to LabVIEW "reference vs last frame" logic)
            mode_ref = cv2.getTrackbarPos("ESPI mode (0=prev,1=ref)", win) == 1
            if mode_ref:
                espi8 = proc.diff_ref(frame8)
            else:
                espi8 = proc.diff_prev(frame8)

            # Temporal averaging (maps to LabVIEW averaging block)
            avg_n = int(cv2.getTrackbarPos("Avg N (1=off)", win))
            disp8 = proc.avg_over_n(espi8, avg_n)

            # Apply gamma correction for contrast adjustment (cached lookup table)
            gamma_val = float(cv2.getTrackbarPos("Gamma x10", win)) / 100.0  # 0.1 to 3.0
            if gamma_val != last_gamma_val:
                # Rebuild lookup table only when gamma changes
                if gamma_val != 1.0:
                    inv_gamma = 1.0 / gamma_val
                    gamma_lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                else:
                    gamma_lut = None
                last_gamma_val = gamma_val
            
            if gamma_lut is not None:
                disp8_gamma = cv2.LUT(disp8, gamma_lut)
            else:
                disp8_gamma = disp8

            # Apply colormap palette
            # 0=grayscale, 1=TURBO, 2=JET, 3=HOT, 4=VIRIDIS, 5=PLASMA, 6=INFERNO
            pal = cv2.getTrackbarPos("Palette (0=gray,1-6)", win)
            if pal == 0:
                disp = cv2.cvtColor(disp8_gamma, cv2.COLOR_GRAY2BGR)
            elif pal == 1:
                disp = cv2.applyColorMap(disp8_gamma, cv2.COLORMAP_TURBO)  # Blue-cyan-green-yellow-red
            elif pal == 2:
                disp = cv2.applyColorMap(disp8_gamma, cv2.COLORMAP_JET)    # Blue-cyan-yellow-red
            elif pal == 3:
                disp = cv2.applyColorMap(disp8_gamma, cv2.COLORMAP_HOT)    # Black-red-yellow-white
            elif pal == 4:
                disp = cv2.applyColorMap(disp8_gamma, cv2.COLORMAP_VIRIDIS) # Dark blue-green-yellow
            elif pal == 5:
                disp = cv2.applyColorMap(disp8_gamma, cv2.COLORMAP_PLASMA)  # Purple-pink-yellow
            elif pal == 6:
                disp = cv2.applyColorMap(disp8_gamma, cv2.COLORMAP_INFERNO) # Black-red-yellow
            else:
                disp = cv2.cvtColor(disp8_gamma, cv2.COLOR_GRAY2BGR)

            # AVI recording (writes displayed frame)
            rec = cv2.getTrackbarPos("Record AVI (0/1)", win) == 1
            if rec and writer is None:
                h, w = disp.shape[:2]
                avi_path = os.path.join(out_dir, f"espi_{ts()}.avi")
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(avi_path, fourcc, 30.0, (w, h), True)
                print(f"Recording AVI: {avi_path}")
            if (not rec) and writer is not None:
                writer.release()
                writer = None
                print("Stopped recording.")
            if writer is not None:
                writer.write(disp)

            cv2.imshow(win, disp)

            wait_ms = max(int(cv2.getTrackbarPos("Wait ms", win)), 1)
            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                path = os.path.join(out_dir, f"frame_{ts()}.png")
                cv2.imwrite(path, disp)
                print(f"Saved: {path}")
            elif key == ord("r"):
                proc.reset_prev()
                print("Reset last frame.")
            elif key == ord("a"):
                proc.reset_avg()
                print("Reset averaging.")
            elif key == ord("f"):
                proc.set_reference(frame8)
                print("Captured reference frame.")

    finally:
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass

        try:
            cam.EndAcquisition()
        except PySpin.SpinnakerException:
            pass
        try:
            cam.DeInit()
        except PySpin.SpinnakerException:
            pass

        cam_list.Clear()
        system.ReleaseInstance()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
