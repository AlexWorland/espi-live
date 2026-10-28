"""
blackfly_espi_minimal.py

Goal:
- Force the Blackfly into Mono8, continuous acquisition, minimal buffering.
- Show:
    1) RAW speckle
    2) Consecutive diff
    3) Reference diff
- Print a checksum so we KNOW frames are changing.

Keys:
  q : quit
  r : grab new reference (object at rest)
"""

import PySpin
import numpy as np
import cv2


def to_gray_float(image):
    """Convert PySpin image to float32 grayscale numpy array."""
    # Convert to 8-bit mono if not already
    img_mono = image.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
    data = img_mono.GetNDArray()
    gray = data.astype(np.float32)
    return gray


def main():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        print("No cameras found.")
        cam_list.Clear()
        system.ReleaseInstance()
        return

    cam = cam_list[0]
    cam.Init()

    # ---------------- Camera configuration: force simple, ESPI-friendly mode -------------
    nodemap = cam.GetNodeMap()

    # AcquisitionMode = Continuous
    node_acq = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
    node_acq_cont = node_acq.GetEntryByName("Continuous")
    node_acq.SetIntValue(node_acq_cont.GetValue())

    # PixelFormat = Mono8
    node_pix = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
    node_pix_mono8 = node_pix.GetEntryByName("Mono8")
    node_pix.SetIntValue(node_pix_mono8.GetValue())

    # Exposure: manual, modest brightness (adjust as needed)
    try:
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed)
        cam.ExposureTime.SetValue(5000.0)  # 5000 us = 5 ms as a starting point
    except PySpin.SpinnakerException:
        pass

    # Gain: modest
    try:
        cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        cam.Gain.SetValue(5.0)
    except PySpin.SpinnakerException:
        pass

    # Frame rate (if controllable)
    try:
        cam.AcquisitionFrameRateEnable.SetValue(True)
        cam.AcquisitionFrameRate.SetValue(30.0)
    except PySpin.SpinnakerException:
        pass

    # Stream buffer handling: NewestOnly to avoid stale frames
    s_node_map = cam.GetTLStreamNodeMap()
    handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode("StreamBufferHandlingMode"))
    if PySpin.IsReadable(handling_mode) and PySpin.IsWritable(handling_mode):
        newest_only = handling_mode.GetEntryByName("NewestOnly")
        handling_mode.SetIntValue(newest_only.GetValue())

    # Start acquisition
    cam.BeginAcquisition()

    # Grab one frame to initialize
    image = cam.GetNextImage(1000)
    if image.IsIncomplete():
        print("Initial image incomplete, aborting.")
        image.Release()
        cam.EndAcquisition()
        cam.DeInit()
        cam_list.Clear()
        system.ReleaseInstance()
        return

    gray = to_gray_float(image)
    image.Release()

    ref_frame = gray.copy()
    prev_frame = gray.copy()

    print("Running minimal ESPI test. q=quit, r=new reference.")
    while True:
        image = cam.GetNextImage(1000)
        if image.IsIncomplete():
            image.Release()
            print("Incomplete frame, skipping.")
            continue

        gray = to_gray_float(image)
        image.Release()

        # Quick checksum: should change if frames are changing
        checksum = int(np.sum(gray) % 10000)
        print("checksum:", checksum)

        # RAW display
        raw_8u = np.clip(gray, 0, 255).astype(np.uint8)
        cv2.imshow("RAW speckle", raw_8u)

        # Consecutive diff
        diff_con = np.abs(gray - prev_frame)
        prev_frame[:] = gray
        consec_8u = np.clip(diff_con * 10.0, 0, 255).astype(np.uint8)
        cv2.imshow("Consecutive |I(t)-I(t-1)|", consec_8u)

        # Reference diff
        diff_ref = np.abs(gray - ref_frame)
        ref_8u = np.clip(diff_ref * 10.0, 0, 255).astype(np.uint8)
        cv2.imshow("Reference |I(t)-I_ref|", ref_8u)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            ref_frame[:] = gray
            print("New reference captured.")

    cam.EndAcquisition()
    cam.DeInit()
    cam_list.Clear()
    system.ReleaseInstance()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
