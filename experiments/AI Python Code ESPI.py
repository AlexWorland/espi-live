import PySpin
import numpy as np
import cv2

def acquire_image_pyspin(serial_number=None):
    """
    Acquires a single image from a FLIR Blackfly camera using PySpin.
    
    Args:
        serial_number (str, optional): The serial number of the camera to use. 
                                        If None, uses the first available camera.
    Returns:
        numpy.ndarray: The acquired image as a NumPy array, or None if acquisition fails.
    """
    
    # 1. Get system instance
    system = PySpin.System.GetInstance()
    
    # 2. Get camera list
    cam_list = system.GetCameras()
    
    if cam_list.GetSize() == 0:
        print("No cameras found. Please check connections and Spinnaker drivers.")
        system.ReleaseInstance()
        return None
    
    if serial_number:
        # Get specific camera by serial number
        cam = cam_list.GetBySerial(serial_number)
    else:
        # Get the first camera
        cam = cam_list.GetByIndex(0)
        
    if not PySpin(cam, PySpin.CPlusPlusCamera):
        print(f"Camera {serial_number if serial_number else 0} is not a valid camera object.")
        return None

    print(f"Using camera {cam.GetDeviceInfo().GetSerialNumber()}")

    try:
        # 3. Initialize camera
        cam.Init()
        
        # Configure camera for continuous acquisition
        cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        cam.BeginAcquisition()
        
        # Optional: Set pixel format to Mono16 for raw data (common in scientific applications)
        # cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)

        # 4. Acquire image
        image_result = cam.GetNextImage(PySpin.EventTimeoutSource_Any)
        
        if image_result.IsValid():
            # Convert to numpy array
            image_data = image_result.GetData()
            width = image_result.GetWidth()
            height = image_result.GetHeight()
            
            # Reshape the 1D array to a 2D image (adjust channels if using color)
            # This example assumes a monochrome camera (1 channel) for ESPI.
            image_np = image_data.reshape((height, width))
            print(f"Image acquired: {width}x{height}")

            # Release the image
            image_result.Release()
            
            return image_np
        else:
            print("Image not valid")
            return None

    except PySpin.SpinnakerException as e:
        print(f"An error occurred: {e}")
        return None
        
    finally:
        # 5. Deinitialize and release camera
        if cam.IsCapturing():
            cam.EndAcquisition()
        cam.DeInit()
        del cam
        system.ReleaseInstance()
        del system

if __name__ == '__main__':
    # You might need to change the serial number
    image = acquire_image_pyspin(serial_number='YOUR_CAMERA_SERIAL_NUMBER') 
    
    if image is not None:
        # Example using OpenCV to save the image
        # Note: OpenCV's imwrite can handle 16-bit images directly
        cv2.imwrite("captured_espi_frame.png", image)
        print("Image saved as captured_espi_frame.png")

