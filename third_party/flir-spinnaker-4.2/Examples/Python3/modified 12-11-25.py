import os
import sys
import time

import numpy as np
import PySpin
import cv2

continue_recording = True


def configure_camera(nodemap):
    """
    Put camera into manual mode and set:
      - exposure time
      - gain
      - frame rate
    Adjust the numeric values below as desired.
    """

    # ----- Exposure (manual) -----
    try:
        node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
        if PySpin.IsReadable(node_exposure_auto) and PySpin.IsWritable(node_exposure_auto):
            entry_off = PySpin.CEnumEntryPtr(node_exposure_auto.GetEntryByName("Off"))
            node_exposure_auto.SetIntValue(entry_off.GetValue())

        node_exposure_time = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
        if PySpin.IsReadable(node_exposure_time) and PySpin.IsWritable(node_exposure_time):
            # Desired exposure in microseconds (example: 20000 us = 20 ms)
            desired_exposure = 20000.0

            # Clamp to camera's allowed range
            exp_min = node_exposure_time.GetMin()
            exp_max = node_exposure_time.GetMax()
            desired_exposure = max(exp_min, min(exp_max, desired_exposure))
            node_exposure_time.SetValue(desired_exposure)
            print(f"Exposure set to {desired_exposure:.1f} us "
                  f"(range {exp_min:.1f}–{exp_max:.1f})")

    except PySpin.SpinnakerException as ex:
        print("Exposure configuration error:", ex)

    # ----- Gain (manual) -----
    try:
        node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
        if PySpin.IsReadable(node_gain_auto) and PySpin.IsWritable(node_gain_auto):
            entry_off = PySpin.CEnumEntryPtr(node_gain_auto.GetEntryByName("Off"))
            node_gain_auto.SetIntValue(entry_off.GetValue())

        node_gain = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
        if PySpin.IsReadable(node_gain) and PySpin.IsWritable(node_gain):
            # Desired gain in dB (example: 5 dB)
            desired_gain = 5.0

            g_min = node_gain.GetMin()
            g_max = node_gain.GetMax()
            desired_gain = max(g_min, min(g_max, desired_gain))
            node_gain.SetValue(desired_gain)
            print(f"Gain set to {desired_gain:.2f} dB (range {g_min:.2f}–{g_max:.2f})")

    except PySpin.SpinnakerException as ex:
        print("Gain configuration error:", ex)

    # ----- Frame rate (manual) -----
    try:
        # Some cameras require enabling manual frame rate
        node_acq_frame_rate_enable = PySpin.CBooleanPtr(
            nodemap.GetNode("AcquisitionFrameRateEnabled")
        )
        if PySpin.IsReadable(node_acq_frame_rate_enable) and PySpin.IsWritable(node_acq_frame_rate_enable):
            node_acq_frame_rate_enable.SetValue(True)

        # Turn off frame-rate auto, if present
        node_acq_frame_rate_auto = PySpin.CEnumerationPtr(
            nodemap.GetNode("AcquisitionFrameRateAuto")
        )
        if node_acq_frame_rate_auto is not None and PySpin.IsReadable(node_acq_frame_rate_auto) and PySpin.IsWritable(node_acq_frame_rate_auto):
            entry_off = PySpin.CEnumEntryPtr(node_acq_frame_rate_auto.GetEntryByName("Off"))
            node_acq_frame_rate_auto.SetIntValue(entry_off.GetValue())

        node_acq_frame_rate = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
        if PySpin.IsReadable(node_acq_frame_rate) and PySpin.IsWritable(node_acq_frame_rate):
            # Desired frame rate in Hz (example: 30 fps)
            desired_fps = 30.0

            fps_min = node_acq_frame_rate.GetMin()
            fps_max = node_acq_frame_rate.GetMax()
            desired_fps = max(fps_min, min(fps_max, desired_fps))
            node_acq_frame_rate.SetValue(desired_fps)
            print(f"Frame rate set to {desired_fps:.2f} fps (range {fps_min:.2f}–{fps_max:.2f})")

    except PySpin.SpinnakerException as ex:
        print("Frame-rate configuration error:", ex)


def acquire_and_display_images(cam, nodemap, nodemap_tldevice):
    global continue_recording

    # Set stream buffer handling mode = NewestOnly
    sNodemap = cam.GetTLStreamNodeMap()
    node_bufferhandling_mode = PySpin.CEnumerationPtr(
        sNodemap.GetNode('StreamBufferHandlingMode')
    )
    node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
    node_bufferhandling_mode.SetIntValue(node_newestonly.GetValue())

    print('Stream buffer mode set to NewestOnly.')
    print('*** IMAGE ACQUISITION ***')

    try:
        # Acquisition mode = Continuous
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        node_acquisition_mode.SetIntValue(node_acquisition_mode_continuous.GetValue())
        print('Acquisition mode set to Continuous.')

        # Put camera into manual mode and set exposure/gain/fps
        configure_camera(nodemap)

        # Start grabbing
        cam.BeginAcquisition()
        print('Acquiring images...')
        print('Press "q" in the OpenCV window to quit, or Ctrl+C in terminal to force stop')

        # Create OpenCV window
        window_name = "Blackfly Live View"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Create a folder for saved frames
        save_dir = "captures"
        os.makedirs(save_dir, exist_ok=True)

        saved_image_counter = 0  # optional counter

        # Initialize previous frame for ESPI
        previous_frame = None

        try:
            while continue_recording:
                try:
                    # Check for key press first (even before getting image)
                    key = cv2.waitKey(1) & 0xFF

                    # Quit with q, Q, ESC, or Enter
                    if key in (ord('q'), ord('Q'), 27, 13):
                        print("Stopping acquisition...")
                        continue_recording = False
                        break

                    # Grab an image with timeout
                    image_result = cam.GetNextImage(1000)

                    if not image_result.IsIncomplete():
                        frame = image_result.GetNDArray()

                        # ESPI: Subtract consecutive frames
                        if previous_frame is not None:
                            if frame.dtype == 'uint16':
                                diff = frame.astype('int32') - previous_frame.astype('int32')
                                diff_abs = np.abs(diff)
                                diff_max = diff_abs.max() if diff_abs.max() > 0 else 1
                                display_img = ((diff_abs / diff_max) * 255).astype('uint8')
                            else:
                                diff = frame.astype('int16') - previous_frame.astype('int16')
                                diff_abs = np.abs(diff)
                                diff_max = diff_abs.max() if diff_abs.max() > 0 else 1
                                display_img = ((diff_abs / diff_max) * 255).astype('uint8')
                        else:
                            if frame.dtype == 'uint16':
                                display_img = (frame / 256).astype('uint8')
                            else:
                                display_img = frame

                        previous_frame = frame.copy()

                        cv2.imshow(window_name, display_img)

                        # SAVE FRAME when pressing 's'
                        if key in (ord('s'), ord('S')):
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"capture_{timestamp}.png"
                            full_path = os.path.join(save_dir, filename)

                            cv2.imwrite(full_path, display_img)
                            print("\n*** IMAGE SAVED SUCCESSFULLY ***")
                            print(f"File: {full_path}")
                            print("Press 's' again to save another frame, or 'q' to quit\n")

                    image_result.Release()

                except PySpin.SpinnakerException as ex:
                    print("Error during acquisition:", ex)
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user (Ctrl+C). Stopping acquisition...")
            continue_recording = False

        cv2.destroyAllWindows()
        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print("Error:", ex)
        return False

    return True


def run_single_camera(cam):
    try:
        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        cam.Init()
        nodemap = cam.GetNodeMap()

        # Manual configuration occurs inside acquire_and_display_images via configure_camera()
        result = acquire_and_display_images(cam, nodemap, nodemap_tldevice)

        cam.DeInit()
        return result
    except PySpin.SpinnakerException as ex:
        print("Error:", ex)
        return False


def main():
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
