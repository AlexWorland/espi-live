import os
import sys
import time

import numpy as np
import PySpin
import cv2

continue_recording = True


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
                    if key == ord('q') or key == ord('Q') or key == 27 or key == 13:
                        print("Stopping acquisition...")
                        continue_recording = False
                        break

                    # Grab an image with timeout
                    image_result = cam.GetNextImage(1000)

                    if not image_result.IsIncomplete():
                        frame = image_result.GetNDArray()

                        # ESPI: Subtract consecutive frames
                        if previous_frame is not None:
                            # Calculate difference (handle both uint8 and uint16)
                            if frame.dtype == 'uint16':
                                # Convert to signed int for proper subtraction
                                diff = frame.astype('int32') - previous_frame.astype('int32')
                                # Take absolute value and scale for display
                                diff_abs = np.abs(diff)
                                # Normalize to 0-255 range for display
                                diff_max = diff_abs.max() if diff_abs.max() > 0 else 1
                                display_img = ((diff_abs / diff_max) * 255).astype('uint8')
                            else:
                                # For uint8, convert to signed int for subtraction
                                diff = frame.astype('int16') - previous_frame.astype('int16')
                                # Take absolute value and normalize
                                diff_abs = np.abs(diff)
                                diff_max = diff_abs.max() if diff_abs.max() > 0 else 1
                                display_img = ((diff_abs / diff_max) * 255).astype('uint8')
                        else:
                            # First frame: just convert for display
                            if frame.dtype == 'uint16':
                                display_img = (frame / 256).astype('uint8')
                            else:
                                display_img = frame

                        # Store current frame as previous for next iteration
                        previous_frame = frame.copy()

                        # Show ESPI difference frame
                        cv2.imshow(window_name, display_img)

                        # SAVE FRAME when pressing 's'
                        if key == ord('s') or key == ord('S'):
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"capture_{timestamp}.png"
                            full_path = os.path.join(save_dir, filename)

                            # Save the displayed frame
                            cv2.imwrite(full_path, display_img)
                            print(f"\n*** IMAGE SAVED SUCCESSFULLY ***")
                            print(f"File: {full_path}")
                            print(f"Press 's' again to save another frame, or 'q' to quit\n")

                    # Release the image back to buffer pool
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
