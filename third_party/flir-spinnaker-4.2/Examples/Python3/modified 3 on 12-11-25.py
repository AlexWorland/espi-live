import os
import sys
import time

import numpy as np
import PySpin
import cv2

continue_recording = True


def configure_camera(nodemap, exposure_us=20000.0, gain_db=5.0, fps=30.0):
    """
    Put camera into manual mode and set initial exposure, gain, and frame rate.

    Parameters
    ----------
    nodemap : PySpin.INodeMap
        Camera nodemap.
    exposure_us : float
        Desired exposure time in microseconds.
    gain_db : float
        Desired gain in dB.
    fps : float
        Desired frame rate in frames per second.
    """

    # -------- Exposure (manual) --------
    try:
        node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
        if PySpin.IsReadable(node_exposure_auto) and PySpin.IsWritable(node_exposure_auto):
            entry_off = PySpin.CEnumEntryPtr(node_exposure_auto.GetEntryByName("Off"))
            node_exposure_auto.SetIntValue(entry_off.GetValue())

        node_exposure_time = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
        if PySpin.IsReadable(node_exposure_time) and PySpin.IsWritable(node_exposure_time):
            exp_min = node_exposure_time.GetMin()
            exp_max = node_exposure_time.GetMax()
            exposure_us = max(exp_min, min(exp_max, exposure_us))
            node_exposure_time.SetValue(exposure_us)
            print(
                f"Initial exposure set to {node_exposure_time.GetValue():.1f} us "
                f"(range {exp_min:.1f}–{exp_max:.1f})"
            )
    except PySpin.SpinnakerException as ex:
        print("Exposure configuration error:", ex)

    # -------- Gain (manual) --------
    try:
        node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
        if PySpin.IsReadable(node_gain_auto) and PySpin.IsWritable(node_gain_auto):
            entry_off = PySpin.CEnumEntryPtr(node_gain_auto.GetEntryByName("Off"))
            node_gain_auto.SetIntValue(entry_off.GetValue())

        node_gain = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
        if PySpin.IsReadable(node_gain) and PySpin.IsWritable(node_gain):
            g_min = node_gain.GetMin()
            g_max = node_gain.GetMax()
            gain_db = max(g_min, min(g_max, gain_db))
            node_gain.SetValue(gain_db)
            print(
                f"Initial gain set to {node_gain.GetValue():.2f} dB "
                f"(range {g_min:.2f}–{g_max:.2f})"
            )
    except PySpin.SpinnakerException as ex:
        print("Gain configuration error:", ex)

    # -------- Frame rate (manual) --------
    try:
        # Some models need this boolean to enable manual frame rate
        node_fps_enable = PySpin.CBooleanPtr(
            nodemap.GetNode("AcquisitionFrameRateEnabled")
        )
        if PySpin.IsReadable(node_fps_enable) and PySpin.IsWritable(node_fps_enable):
            node_fps_enable.SetValue(True)

        # Turn off frame-rate auto, if present
        node_fps_auto = PySpin.CEnumerationPtr(
            nodemap.GetNode("AcquisitionFrameRateAuto")
        )
        if (
            node_fps_auto is not None
            and PySpin.IsReadable(node_fps_auto)
            and PySpin.IsWritable(node_fps_auto)
        ):
            entry_off = PySpin.CEnumEntryPtr(node_fps_auto.GetEntryByName("Off"))
            node_fps_auto.SetIntValue(entry_off.GetValue())

        node_fps = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
        if PySpin.IsReadable(node_fps) and PySpin.IsWritable(node_fps):
            fps_min = node_fps.GetMin()
            fps_max = node_fps.GetMax()
            fps = max(fps_min, min(fps_max, fps))
            node_fps.SetValue(fps)
            print(
                f"Initial frame rate set to {node_fps.GetValue():.2f} fps "
                f"(range {fps_min:.2f}–{fps_max:.2f})"
            )
    except PySpin.SpinnakerException as ex:
        print("Frame-rate configuration error:", ex)


def get_float_node_and_range(nodemap, name):
    """
    Convenience to fetch a writable float node and its min/max.
    Returns (node, (min, max)) or (None, (None, None)) if unavailable.
    """
    try:
        node = PySpin.CFloatPtr(nodemap.GetNode(name))
        if PySpin.IsReadable(node) and PySpin.IsWritable(node):
            return node, (node.GetMin(), node.GetMax())
    except PySpin.SpinnakerException:
        pass
    return None, (None, None)


def acquire_and_display_images(cam, nodemap, nodemap_tldevice):
    global continue_recording

    # Stream buffer mode: NewestOnly to reduce lag
    try:
        sNodemap = cam.GetTLStreamNodeMap()
        node_bufferhandling_mode = PySpin.CEnumerationPtr(
            sNodemap.GetNode("StreamBufferHandlingMode")
        )
        node_newestonly = node_bufferhandling_mode.GetEntryByName("NewestOnly")
        node_bufferhandling_mode.SetIntValue(node_newestonly.GetValue())
        print("Stream buffer mode set to NewestOnly.")
    except PySpin.SpinnakerException as ex:
        print("Could not set stream buffer mode:", ex)

    print("*** IMAGE ACQUISITION ***")

    try:
        # Acquisition mode = Continuous
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName(
            "Continuous"
        )
        node_acquisition_mode.SetIntValue(node_acquisition_mode_continuous.GetValue())
        print("Acquisition mode set to Continuous.")

        # Initial manual configuration
        configure_camera(nodemap)

        # Nodes for live control
        exposure_node, (exp_min, exp_max) = get_float_node_and_range(
            nodemap, "ExposureTime"
        )
        gain_node, (g_min, g_max) = get_float_node_and_range(nodemap, "Gain")
        fps_node, (fps_min, fps_max) = get_float_node_and_range(
            nodemap, "AcquisitionFrameRate"
        )

        if exposure_node is not None:
            print(f"Live exposure start: {exposure_node.GetValue():.1f} us")
        if gain_node is not None:
            print(f"Live gain start: {gain_node.GetValue():.2f} dB")
        if fps_node is not None:
            print(f"Live frame rate start: {fps_node.GetValue():.2f} fps")

        # Start grabbing
        cam.BeginAcquisition()
        print("Acquiring images...")
        print(
            "Controls: q/ESC/Enter=quit, s=save, "
            "e/E=exp +/-10%, g/G=gain +/-1 dB, f/F=fps +/-5%"
        )

        window_name = "Blackfly ESPI Live"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        save_dir = "captures"
        os.makedirs(save_dir, exist_ok=True)

        previous_frame = None

        try:
            while continue_recording:
                # Check for key press
                key = cv2.waitKey(1) & 0xFF

                # Quit
                if key in (ord("q"), ord("Q"), 27, 13):
                    print("Stopping acquisition...")
                    continue_recording = False
                    break

                # -------- Live exposure control --------
                if exposure_node is not None:
                    if key == ord("e"):  # exposure up by 10%
                        try:
                            new_exp = exposure_node.GetValue() * 1.10
                            new_exp = max(exp_min, min(exp_max, new_exp))
                            exposure_node.SetValue(new_exp)
                            print(f"Exposure = {new_exp:.1f} us")
                        except PySpin.SpinnakerException as ex:
                            print("Error changing exposure:", ex)
                    elif key == ord("E"):  # exposure down by 10%
                        try:
                            new_exp = exposure_node.GetValue() * 0.90
                            new_exp = max(exp_min, min(exp_max, new_exp))
                            exposure_node.SetValue(new_exp)
                            print(f"Exposure = {new_exp:.1f} us")
                        except PySpin.SpinnakerException as ex:
                            print("Error changing exposure:", ex)

                # -------- Live gain control --------
                if gain_node is not None:
                    if key == ord("g"):  # gain up by 1 dB
                        try:
                            new_gain = gain_node.GetValue() + 1.0
                            new_gain = max(g_min, min(g_max, new_gain))
                            gain_node.SetValue(new_gain)
                            print(f"Gain = {new_gain:.2f} dB")
                        except PySpin.SpinnakerException as ex:
                            print("Error changing gain:", ex)
                    elif key == ord("G"):  # gain down by 1 dB
                        try:
                            new_gain = gain_node.GetValue() - 1.0
                            new_gain = max(g_min, min(g_max, new_gain))
                            gain_node.SetValue(new_gain)
                            print(f"Gain = {new_gain:.2f} dB")
                        except PySpin.SpinnakerException as ex:
                            print("Error changing gain:", ex)

                # -------- Live frame-rate control --------
                if fps_node is not None:
                    if key == ord("f"):  # fps up by 5%
                        try:
                            new_fps = fps_node.GetValue() * 1.05
                            new_fps = max(fps_min, min(fps_max, new_fps))
                            fps_node.SetValue(new_fps)
                            print(f"Frame rate = {new_fps:.2f} fps")
                        except PySpin.SpinnakerException as ex:
                            print("Error changing frame rate:", ex)
                    elif key == ord("F"):  # fps down by 5%
                        try:
                            new_fps = fps_node.GetValue() * 0.95
                            new_fps = max(fps_min, min(fps_max, new_fps))
                            fps_node.SetValue(new_fps)
                            print(f"Frame rate = {new_fps:.2f} fps")
                        except PySpin.SpinnakerException as ex:
                            print("Error changing frame rate:", ex)

                # Grab image with timeout
                try:
                    image_result = cam.GetNextImage(1000)
                except PySpin.SpinnakerException as ex:
                    print("Error getting image:", ex)
                    break

                if not image_result.IsIncomplete():
                    frame = image_result.GetNDArray()

                    # ESPI: subtract consecutive frames
                    if previous_frame is not None:
                        if frame.dtype == np.uint16:
                            diff = (
                                frame.astype(np.int32)
                                - previous_frame.astype(np.int32)
                            )
                        else:
                            diff = (
                                frame.astype(np.int16)
                                - previous_frame.astype(np.int16)
                            )

                        diff_abs = np.abs(diff)
                        if diff_abs.size == 0:
                            display_img = np.zeros_like(frame, dtype=np.uint8)
                        else:
                            diff_max = diff_abs.max()
                            if diff_max <= 0:
                                display_img = np.zeros_like(frame, dtype=np.uint8)
                            else:
                                display_img = (
                                    diff_abs / diff_max * 255.0
                                ).astype(np.uint8)
                    else:
                        # First frame: simple scaling if needed
                        if frame.dtype == np.uint16:
                            display_img = (frame / 256).astype(np.uint8)
                        else:
                            display_img = frame

                    previous_frame = frame.copy()

                    cv2.imshow(window_name, display_img)

                    # Save displayed frame
                    if key in (ord("s"), ord("S")):
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"capture_{timestamp}.png"
                        full_path = os.path.join(save_dir, filename)
                        cv2.imwrite(full_path, display_img)
                        print(f"Saved frame to {full_path}")

                image_result.Release()

        except KeyboardInterrupt:
            print("\nInterrupted by user (Ctrl+C). Stopping acquisition...")
            continue_recording = False

        cv2.destroyAllWindows()
        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print("Acquisition error:", ex)
        return False

    return True


def run_single_camera(cam):
    try:
        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        cam.Init()
        nodemap = cam.GetNodeMap()
        result = acquire_and_display_images(cam, nodemap, nodemap_tldevice)
        cam.DeInit()
        return result
    except PySpin.SpinnakerException as ex:
        print("Camera error:", ex)
        return False


def main():
    global continue_recording
    continue_recording = True  # reset if script is re-run in same process

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    num = cam_list.GetSize()

    if num == 0:
        print("No cameras detected.")
        cam_list.Clear()
        system.ReleaseInstance()
        return False

    for i, cam in enumerate(cam_list):
        print(f"Running example for camera {i}...")
        run_single_camera(cam)

    del cam
    cam_list.Clear()
    system.ReleaseInstance()
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
