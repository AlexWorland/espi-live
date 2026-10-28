"""Live camera control tool.

This script mirrors FLIR's official acquisition example while layering on an
educational OpenCV UI. It connects to the PySpin system, detects connected
FLIR/Spinnaker cameras, and for each one it:

1. Initializes the transport layer stream so the camera can communicate.
2. Disables auto exposure/gain to give full manual control.
3. Starts a continuous acquisition loop.
4. Displays live frames alongside sliders for exposure, gain, and frame rate.
5. Reacts to slider changes immediately by writing values back to the camera.

Press 'q' while the OpenCV window is focused to end the live view for the
current camera, after which the script cleans up the camera and system handles.
The design mimics the official tutorial so novice users can recognize the same
setup/teardown pattern while experimenting interactively.
"""

import platform  # Detect the operating system so we can choose the right stream mode.
import sys  # Provide a clean KeyboardInterrupt exit path like the official example.

import cv2  # Offer a simple UI layer (window + sliders) for real-time camera adjustments.
import PySpin  # Core Spinnaker SDK used for enumerating, configuring, and streaming from FLIR cameras.


WINDOW_NAME = "Camera Live Control"


class StreamMode:
    STREAM_MODE_TELEDYNE_GIGE_VISION = 0
    STREAM_MODE_PGRLWF = 1
    STREAM_MODE_SOCKET = 2


_SYSTEM_NAME = platform.system()
if _SYSTEM_NAME == "Windows":
    CHOSEN_STREAMMODE = StreamMode.STREAM_MODE_TELEDYNE_GIGE_VISION
elif _SYSTEM_NAME in ("Linux", "Darwin"):
    CHOSEN_STREAMMODE = StreamMode.STREAM_MODE_SOCKET
else:
    CHOSEN_STREAMMODE = StreamMode.STREAM_MODE_SOCKET


def _set_stream_mode(camera):
    # Pick the safest stream mode for the OS so the camera talks to the PC.
    modes = {
        StreamMode.STREAM_MODE_TELEDYNE_GIGE_VISION: "TeledyneGigeVision",
        StreamMode.STREAM_MODE_PGRLWF: "LWF",
        StreamMode.STREAM_MODE_SOCKET: "Socket",
    }
    stream_name = modes.get(CHOSEN_STREAMMODE, "Socket")

    try:
        nodemap_tlstream = camera.GetTLStreamNodeMap()
        node_stream_mode = PySpin.CEnumerationPtr(nodemap_tlstream.GetNode("StreamMode"))
        if not PySpin.IsReadable(node_stream_mode) or not PySpin.IsWritable(node_stream_mode):
            return
        node_entry = PySpin.CEnumEntryPtr(node_stream_mode.GetEntryByName(stream_name))
        if not PySpin.IsReadable(node_entry):
            return
        node_stream_mode.SetIntValue(node_entry.GetValue())
    except PySpin.SpinnakerException:
        pass


def _set_if_writable(node, value):
    # Try to set the camera node, but only if it actually allows writes.
    if node is None:
        return
    try:
        if node.GetAccessMode() == PySpin.RW:
            node.SetValue(value)
    except PySpin.SpinnakerException:
        pass


def _disable_auto_features(camera):
    # Turn off auto exposure/gain so the sliders have full control.
    for attr, target in (
        ("ExposureAuto", PySpin.ExposureAuto_Off),
        ("GainAuto", PySpin.GainAuto_Off),
    ):
        node = getattr(camera, attr, None)
        if node is not None:
            _set_if_writable(node, target)


def _set_trackbars(window, nodes):
    # Build the OpenCV sliders using the camera's current settings.
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(
        "Exposure (µs)",
        window,
        min(int(nodes["exposure"].GetValue()), nodes["exp_max"]),
        nodes["exp_max"],
        lambda v: None,
    )
    cv2.createTrackbar(
        "Gain (dB)",
        window,
        min(int(nodes["gain"].GetValue()), nodes["gain_max"]),
        nodes["gain_max"],
        lambda v: None,
    )
    cv2.createTrackbar(
        "FrameRate (fps)",
        window,
        min(int(nodes["framerate"].GetValue()), nodes["framerate_max"]),
        nodes["framerate_max"],
        lambda v: None,
    )


def _read_trackbar(window, name):
    # Read the latest value of a slider.
    return cv2.getTrackbarPos(name, window)


def _grab_frame(camera, processor):
    # Grab the next frame, convert it for OpenCV, then free camera buffers.
    image = camera.GetNextImage(1000)
    if image.IsIncomplete():
        image.Release()
        return None

    converted = processor.Convert(image, PySpin.PixelFormat_BGR8)
    frame = converted.GetNDArray()
    converted.Release()
    image.Release()
    return frame


def _build_node_config(camera):
    # Collect the nodes we tweak (exposure, gain, fps) along with safe limits.
    exp_node = camera.ExposureTime
    gain_node = camera.Gain
    framerate_node = camera.AcquisitionFrameRate

    exp_max = max(min(int(exp_node.GetMax()), 1000000), 1)
    gain_max = max(min(int(gain_node.GetMax()), 60), 1)
    framerate_max = max(min(int(framerate_node.GetMax()), 120), 1)

    nodes = {
        "exposure": exp_node,
        "gain": gain_node,
        "framerate": framerate_node,
        "exp_max": exp_max,
        "gain_max": gain_max,
        "framerate_max": framerate_max,
    }

    return nodes


def _update_settings(window, nodes):
    # Write the current slider values back to the camera every loop.
    expo = max(_read_trackbar(window, "Exposure (µs)"), 1)
    gain = _read_trackbar(window, "Gain (dB)")
    fps = max(_read_trackbar(window, "FrameRate (fps)"), 1)

    _set_if_writable(nodes["exposure"], float(expo))
    _set_if_writable(nodes["gain"], float(gain))
    _set_if_writable(nodes["framerate"], float(fps))


def run_live_view(camera):
    # Start continuous capture and show frames until the user presses 'q'.
    camera.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    fps_enable = getattr(camera, "AcquisitionFrameRateEnable", None)
    if fps_enable is not None:
        _set_if_writable(fps_enable, True)

    nodes = _build_node_config(camera)
    _set_trackbars(WINDOW_NAME, nodes)

    processor = PySpin.ImageProcessor()
    processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

    camera.BeginAcquisition()
    try:
        while True:
            frame = _grab_frame(camera, processor)
            if frame is None:
                continue

            _update_settings(WINDOW_NAME, nodes)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.EndAcquisition()
        cv2.destroyAllWindows()


def _run_single_camera(camera):
    # Init the camera, run the live view, and always clean up afterward.
    try:
        camera.Init()
        _set_stream_mode(camera)
        _disable_auto_features(camera)
        run_live_view(camera)
    finally:
        camera.DeInit()


def main():
    # Match the FLIR example flow: init system, handle each camera, release.
    system = PySpin.System.GetInstance()
    version = system.GetLibraryVersion()
    print(
        "Library version: %d.%d.%d.%d"
        % (version.major, version.minor, version.type, version.build)
    )

    cam_list = system.GetCameras()
    num_cameras = cam_list.GetSize()
    print("Number of cameras detected: %d" % num_cameras)

    if num_cameras == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        print("No cameras detected.")
        return

    camera = None
    try:
        for index in range(num_cameras):
            camera = cam_list.GetByIndex(index)
            print("Starting live view for camera %d..." % index)
            _run_single_camera(camera)
            print("Camera %d live view ended.\n" % index)
            del camera
            camera = None
    finally:
        if camera is not None:
            del camera
        cam_list.Clear()
        system.ReleaseInstance()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
