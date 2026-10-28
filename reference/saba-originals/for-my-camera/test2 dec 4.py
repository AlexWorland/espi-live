"""
blackfly_espi_minimal.py

Minimal ESPI-style test for a monochrome Teledyne FLIR Blackfly camera
using the PySpin API directly.

Windows:
  - "RAW speckle"                 : live image
  - "Consecutive |I(t)-I(t-1)|"   : frame-to-frame difference
  - "Reference |I(t)-I_ref|"      : difference from stored reference frame

Keys:
  - q : quit
  - r : capture new reference (object must be at rest)

This script assumes a monochrome camera; no color / Bayer handling is used.
"""

import PySpin
import numpy as np
import cv2


def to_gray_float(image):
    """
    Convert a PySpin ImagePtr to a float32 2D array.

    For a monochrome camera, this uses the ImageProcessor to convert to Mono8
    (8-bit grayscale) and then casts to float32 for safe arithmetic.
    """
    processor = PySpin.ImageProcessor()
    img_mono = processor.Convert(image, PySpin.PixelFormat_Mono8)
    data = img_mono.GetNDArray()  # uint8 array
    return data.astype(np.float32)


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

    # ---------------- Camera configuration ----------------
    nodemap = cam.GetNodeMap()

    # AcquisitionMode = Continuous
    try:
        node_acq = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        node_acq_cont = node_acq.GetEntryByName("Continuous")
        node_acq.SetIntValue(node_acq_cont.GetValue())
    except PySpin.SpinnakerException:
        pass

    # PixelFormat = Mono8 (monochrome)
    try:
        node_pix = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
        node_pix_mono8 = node_pix.GetEntryByName("Mono8")
        node_pix.SetIntValue(node_pix_mono8.GetValue())
    except PySpin.SpinnakerException:
        pass

    # Exposure: manual, timed, moderate exposure (adjust as needed)
    try:
        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed)
        cam.ExposureTime.SetValue(5000.0)  # microseconds; try 2000–10000 as needed
    except PySpin.SpinnakerException:
        pass

    # Gain: manual, modest
    try:
        cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        cam.Gain.SetValue(5.0)  # dB; adjust if too dark/bright
    except PySpin.SpinnakerException:
        pass

    # Frame rate (if supported)
    try:
        cam.AcquisitionFrameRateEnable.SetValue(True)
        cam.AcquisitionFrameRate.SetValue(30.0)
    except PySpin.SpinnakerException:
        pass

    # Stream buffer handling: NewestOnly to reduce lag and stale frames
    try:
        s_node_map = cam.GetTLStreamNodeMap()
        handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode("StreamBufferHandlingMode"))
        if PySpin.IsReadable(handling_mode) and PySpin.IsWritable(handling_mode):
            newest_only = handling_mode.GetEntryByName("NewestOnly")
            handling_mode.SetIntValue(newest_only.GetValue())
    except PySpin.SpinnakerException:
        pass

    # ---------------- Start acquisition ----------------
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

    print("Running minimal ESPI test.")
    print("q = quit, r = new reference (object at rest)")

    while True:
        image = cam.GetNextImage(1000)
        if image.IsIncomplete():
            image.Release()
            print("Incomplete frame, skipping.")
            continue

        gray = to_gray_float(image)
        image.Release()

        # Simple checksum to confirm frames are changing
        checksum = int(np.sum(gray) % 10000)
        print("checksum:", checksum)

        # RAW display
        raw_8u = np.clip(gray, 0, 255).astype(np.uint8)
        cv2.imshow("RAW speckle", raw_8u)

        # Consecutive difference: |I(t) - I(t-1)|
        diff_con = np.abs(gray - prev_frame)
        prev_frame[:] = gray
        consec_8u = np.clip(diff_con * 10.0, 0, 255).astype(np.uint8)  # gain=10
        cv2.imshow("Consecutive |I(t)-I(t-1)|", consec_8u)

        # Reference difference: |I(t) - I_ref|
        diff_ref = np.abs(gray - ref_frame)
        ref_8u = np.clip(diff_ref * 10.0, 0, 255).astype(np.uint8)  # gain=10
        cv2.imshow("Reference |I(t)-I_ref|", ref_8u)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            ref_frame[:] = gray
            print("New reference captured.")

    # ---------------- Cleanup ----------------
    cam.EndAcquisition()
    cam.DeInit()
    cam_list.Clear()
    system.ReleaseInstance()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

