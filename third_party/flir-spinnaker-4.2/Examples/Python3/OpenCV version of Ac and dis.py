import os
import sys
import time

import PySpin
import cv2

# Global flag to control acquisition loop
continue_recording = True


def acquire_and_display_images(cam, nodemap, nodemap_tldevice):
    """
    Continuously acquire images from the camera and display them using OpenCV.
    """

    global continue_recording

    # Get TL stream node map (for buffer handling mode)
    sNodemap = cam.GetTLStreamNodeMap()

    # Set stream buffer handling mode to NewestOnly (avoids growing lag)
    node_bufferhandling_mode = PySpin.CEnumerationPtr(
        sNodemap.GetNode('StreamBufferHandlingMode')
    )
    if not PySpin.IsReadable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
        print('Unable to set stream buffer handling mode. Aborting...')
        return False

    node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
    if not PySpin.IsReadable(node_newestonly):
        print('Unable to get NewestOnly entry. Aborting...')
        return False

    node_newestonly_mode = node_newestonly.GetValue()
    node_bufferhandling_mode.SetIntValue(node_newestonly_mode)
    print('Stream buffer handling mode set to NewestOnly.')

    print('*** IMAGE ACQUISITION ***\n')

    try:
        # Set acquisition mode to Continuous
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous. Aborting...')
            return False

        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsReadable(node_acquisition_mode_continuous):
            print('Unable to get Continuous acquisition mode. Aborting...')
            return False

        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
        print('Acquisition mode set to Continuous.')

        # Begin acquisition
        cam.BeginAcquisition()
        print('Acquiring images...')

        # Device serial (optional, for logging)
        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('Device serial number: %s' % device_serial_number)

        print("Press 'q' or Enter (in the OpenCV window) to stop.")

        # Create OpenCV window once
        window_name = "Blackfly Live View"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while continue_recording:
            try:
                # Get next image (timeout in ms)
                image_result = cam.GetNextImage(1000)

                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...'
                          % image_result.GetImageStatus())
                else:
                    # Get image as NumPy array
                    image_data = image_result.GetNDArray()

                    # If 16-bit, scale to 8-bit for display
                    if image_data.dtype == 'uint16':
                        display_img = (image_data / 256).astype('uint8')
                    else:
                        display_img = image_data

                    # For mono images, show directly (OpenCV treats single-channel as grayscale)
                    cv2.imshow(window_name, display_img)

                    # waitKey processes GUI events and returns key pressed
                    key = cv2.waitKey(1) & 0xFF
                    # 'q' or Enter key (ASCII 13) to quit
                    if key == ord('q') or key == 13:
                        print('Stopping acquisition...')
                        continue_recording = False

                # Release image to avoid filling the buffer
                image_result.Release()

            except PySpin.SpinnakerException as ex:
                print('Error during acquisition: %s' % ex)
                break

        # Clean up OpenCV window
        cv2.destroyAllWindows()

        # End acquisition
        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return True


def run_single_camera(cam):
    """
    Initialize camera, run acquisition, then deinitialize.
    """
    result = True

    try:
        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        # Acquire images
        result &= acquire_and_display_images(cam, nodemap, nodemap_tldevice)

        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


def main():
    """
    Entry point: connect to system, run on each detected camera.
    """
    result = True

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get library version
    version = system.GetLibraryVersion()
    print('Spinnaker library version: %d.%d.%d.%d'
          % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras
    cam_list = system.GetCameras()
    num_cameras = cam_list.GetSize()
    print('Number of cameras detected: %d' % num_cameras)

    if num_cameras == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        print('No cameras detected.')
        input('Done! Press Enter to exit...')
        return False

    # Run example on each camera
    for i, cam in enumerate(cam_list):
        print('Running example for camera %d...' % i)
        result &= run_single_camera(cam)
        print('Camera %d example complete.\n' % i)

    # Release reference to last camera
    del cam

    # Clear camera list and release system
    cam_list.Clear()
    system.ReleaseInstance()

    input('Done! Press Enter to exit...')
    return result


if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
