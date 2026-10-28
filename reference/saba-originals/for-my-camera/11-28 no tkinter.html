"""
ESPI Drumhead Imaging — High-Speed OpenCV-Only Version
FLIR Blackfly + PySpin + OpenCV

Goals:
- Minimize lag.
- Use only OpenCV windows (no Tkinter, no PIL).
- Keep logic simple and fast.

Features:
- Live camera view.
- Time-averaged ESPI-style processing:
    * mode 0: RAW (live speckle)
    * mode 1: DIFF (|current - reference|)
    * mode 2: DIFF+EQ (|current - reference| with histogram equalization)
- Keyboard controls:
    - 'r' : capture reference (drum at rest)
    - 's' : save current processed frame (PNG)
    - space : start/stop recording processed view (AVI)
    - 'q' or ESC : quit
- Trackbars (in main window):
    - "mode" (0–2): Raw / Diff / Diff+EQ
    - "contrast x10" (1–30): diff contrast scaling
    - "side_by_side" (0/1): show Raw vs Processed in second window
    - "exp_ms" (1–50): exposure time in ms (optional, may not exist on all cams)
    - "gain_dB" (0–24): gain in dB (optional)

Notes:
- This assumes camera pixel format is Mono8.
- ROI reduction is optional (commented; you can enable it to reduce load).
"""

import cv2
import numpy as np
import PySpin


# ------------------- Camera helpers ------------------- #

def get_camera():
    """
    Connect to the first available PySpin camera and return:
    - system handle
    - camera object
    - nodemap
    """
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        raise RuntimeError("No PySpin-compatible cameras detected.")

    cam = cam_list[0]
    cam.Init()
    nodemap = cam.GetNodeMap()
    cam_list.Clear()
    return system, cam, nodemap


def set_enum(nodemap, name, value):
    node = PySpin.CEnumerationPtr(nodemap.GetNode(name))
    if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
        return
    entry = node.GetEntryByName(value)
    if PySpin.IsReadable(entry):
        node.SetIntValue(entry.GetValue())


def set_float(nodemap, name, val):
    node = PySpin.CFloatPtr(nodemap.GetNode(name))
    if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
        return
    val = float(val)
    val = max(node.GetMin(), min(node.GetMax(), val))
    node.SetValue(val)


def set_bool(nodemap, name, val):
    node = PySpin.CBooleanPtr(nodemap.GetNode(name))
    if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
        return
    node.SetValue(bool(val))


def get_mono8_array(image):
    """
    Convert a PySpin ImagePtr to a 2D uint8 NumPy array.
    Assumes the camera is already set to Mono8 pixel format.
    """
    w = image.GetWidth()
    h = image.GetHeight()
    data = image.GetData()
    arr = np.array(data, dtype=np.uint8).reshape((h, w))
    return arr


def configure_camera_for_speed(cam, nodemap):
    """
    Configure camera for ESPI and speed.
    """
    set_enum(nodemap, "PixelFormat", "Mono8")
    set_enum(nodemap, "ExposureAuto", "Off")
    set_enum(nodemap, "GainAuto", "Off")
    set_enum(nodemap, "AcquisitionMode", "Continuous")

    # Try to set buffer handling to "NewestOnly" to avoid lag from queued frames
    try:
        s_node_map = cam.GetTLStreamNodeMap()
        handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode("StreamBufferHandlingMode"))
        if PySpin.IsReadable(handling_mode) and PySpin.IsWritable(handling_mode):
            newest_only = handling_mode.GetEntryByName("NewestOnly")
            if PySpin.IsReadable(newest_only):
                handling_mode.SetIntValue(newest_only.GetValue())
    except Exception:
        pass

    # Try to boost frame rate (if exposed)
    try:
        set_bool(nodemap, "AcquisitionFrameRateEnable", True)
        set_float(nodemap, "AcquisitionFrameRate", 60.0)
    except Exception:
        pass

    # Exposure and gain defaults (will be controlled by trackbars)
    set_float(nodemap, "ExposureTime", 3000)  # 3 ms
    set_float(nodemap, "Gain", 6)

    # Optional: reduce ROI to cut data rate (enable if needed)
    # try:
    #     width_node = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
    #     height_node = PySpin.CIntegerPtr(nodemap.GetNode("Height"))
    #     width_node.SetValue(min(800, width_node.GetMax()))
    #     height_node.SetValue(min(600, height_node.GetMax()))
    # except Exception:
    #     pass


# ------------------- Main ESPI viewer ------------------- #

def main():
    system = None
    cam = None

    try:
        system, cam, nodemap = get_camera()
        configure_camera_for_speed(cam, nodemap)
        cam.BeginAcquisition()

        # State
        reference = None
        last_processed = None

        # Recording
        is_recording = False
        video_writer = None
        video_path = None
        video_fps = 30.0

        # OpenCV UI
        cv2.namedWindow("ESPI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ESPI", 960, 720)

        # Trackbars
        # mode: 0=Raw, 1=Diff, 2=Diff+EQ
        cv2.createTrackbar("mode", "ESPI", 2, 2, lambda v: None)
        # contrast scaling factor / 10
        cv2.createTrackbar("contrast x10", "ESPI", 10, 30, lambda v: None)
        # side-by-side toggle
        cv2.createTrackbar("side_by_side", "ESPI", 0, 1, lambda v: None)
        # exposure (ms) and gain (dB) — optional controls
        cv2.createTrackbar("exp_ms", "ESPI", 3, 50, lambda v: None)
        cv2.createTrackbar("gain_dB", "ESPI", 6, 24, lambda v: None)

        # Remember last applied exposure/gain to avoid hammering camera
        last_exp_ms = 3
        last_gain_db = 6

        print("Controls:")
        print("  r      : capture reference (drum at rest)")
        print("  s      : save current processed frame (PNG)")
        print("  space  : start/stop recording processed view (AVI)")
        print("  q or ESC : quit")

        while True:
            # Acquire frame (short timeout to avoid blocking)
            try:
                image = cam.GetNextImage(100)  # ms
            except PySpin.SpinnakerException:
                continue

            if image.IsIncomplete():
                image.Release()
                continue

            frame = get_mono8_array(image)
            image.Release()

            # Read trackbar settings
            mode = cv2.getTrackbarPos("mode", "ESPI")
            contrast_raw = cv2.getTrackbarPos("contrast x10", "ESPI")
            side_by_side = cv2.getTrackbarPos("side_by_side", "ESPI")
            exp_ms = cv2.getTrackbarPos("exp_ms", "ESPI")
            gain_db = cv2.getTrackbarPos("gain_dB", "ESPI")

            # Clamp exposure/gain
            if exp_ms < 1:
                exp_ms = 1
                cv2.setTrackbarPos("exp_ms", "ESPI", 1)
            if gain_db < 0:
                gain_db = 0
                cv2.setTrackbarPos("gain_dB", "ESPI", 0)

            # Apply exposure/gain only if changed
            if exp_ms != last_exp_ms:
                try:
                    set_float(nodemap, "ExposureTime", exp_ms * 1000.0)
                    last_exp_ms = exp_ms
                except Exception:
                    pass

            if gain_db != last_gain_db:
                try:
                    set_float(nodemap, "Gain", gain_db)
                    last_gain_db = gain_db
                except Exception:
                    pass

            # Process according to mode
            if mode == 0 or reference is None:
                processed = frame
            else:
                diff = cv2.absdiff(frame, reference)
                scale_factor = max(0.1, contrast_raw / 10.0)
                diff_scaled = np.clip(diff.astype(np.float32) * scale_factor,
                                      0, 255).astype(np.uint8)
                if mode == 1:
                    processed = diff_scaled
                else:
                    processed = cv2.equalizeHist(diff_scaled)

            last_processed = processed

            # Recording
            if is_recording and video_writer is not None:
                video_writer.write(processed)

            # Display
            cv2.imshow("ESPI", processed)

            if side_by_side:
                if frame.shape != processed.shape:
                    proc_resized = cv2.resize(processed,
                                              (frame.shape[1], frame.shape[0]),
                                              interpolation=cv2.INTER_AREA)
                else:
                    proc_resized = processed
                combined = np.hstack([frame, proc_resized])
                cv2.imshow("Raw_vs_Processed", combined)
            else:
                try:
                    cv2.destroyWindow("Raw_vs_Processed")
                except cv2.error:
                    pass

            # Keyboard handling
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):  # 'q' or ESC
                break
            elif key == ord('r'):
                reference = frame.copy()
                print("Reference captured.")
            elif key == ord('s'):
                if last_processed is not None:
                    import time
                    fname = time.strftime("espi_frame_%Y%m%d_%H%M%S.png")
                    cv2.imwrite(fname, last_processed)
                    print(f"Saved {fname}")
            elif key == ord(' '):  # space toggles recording
                if not is_recording:
                    if last_processed is None:
                        print("No frame yet for recording size; wait a moment.")
                    else:
                        import time
                        fname = time.strftime("espi_record_%Y%m%d_%H%M%S.avi")
                        h, w = last_processed.shape
                        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                        video_writer = cv2.VideoWriter(fname, fourcc,
                                                       video_fps, (w, h),
                                                       isColor=False)
                        if not video_writer.isOpened():
                            print("Could not open VideoWriter.")
                            video_writer = None
                        else:
                            is_recording = True
                            video_path = fname
                            print(f"Recording started: {fname}")
                else:
                    is_recording = False
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                        print(f"Recording stopped: {video_path}")
                        video_path = None

        # Cleanup
        if is_recording and video_writer is not None:
            video_writer.release()

        cam.EndAcquisition()
        cam.DeInit()
        system.ReleaseInstance()
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", e)
        try:
            if cam is not None:
                cam.EndAcquisition()
        except Exception:
            pass
        try:
            if cam is not None:
                cam.DeInit()
        except Exception:
            pass
        try:
            if system is not None:
                system.ReleaseInstance()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
