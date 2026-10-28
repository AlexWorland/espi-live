class BlackflyCamera:
    """
    Simple wrapper around a FLIR Blackfly camera using PySpin.

    Responsibilities:
    - Connect to the first camera (or one by serial)
    - Configure basic exposure and gain
    - Start/stop acquisition
    - Grab single frames or averaged frames
    - Provide methods to adjust exposure/gain at runtime
    """

    def __init__(self, serial=None, exposure_us=5000.0, gain_db=0.0):
        # Track whether streaming has successfully started
        self.streaming = False

        # Get the global PySpin system object
        self.system = PySpin.System.GetInstance()

        # Enumerate all connected cameras
        self.cam_list = self.system.GetCameras()
        if self.cam_list.GetSize() == 0:
            self.cam_list.Clear()
            self.system.ReleaseInstance()
            raise RuntimeError("No Spinnaker/PySpin cameras detected.")

        # Select camera by serial or just take the first one
        if serial is None:
            self.cam = self.cam_list[0]
        else:
            self.cam = self.cam_list.GetBySerial(serial)

        # Initialize the camera
        self.cam.Init()

        # Set acquisition mode to continuous
        node = self.cam.AcquisitionMode.GetEntry(PySpin.AcquisitionMode_Continuous)
        self.cam.AcquisitionMode.SetIntValue(node.GetValue())

        # Disable auto exposure and auto gain to get manual control
        if self.cam.ExposureAuto.GetAccessMode() == PySpin.RW:
            self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        if self.cam.GainAuto.GetAccessMode() == PySpin.RW:
            self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)

        # Cache exposure/gain limits for later UI mapping
        if self.cam.ExposureTime.GetAccessMode() == PySpin.RW:
            self.exp_min = self.cam.ExposureTime.GetMin()
            self.exp_max = self.cam.ExposureTime.GetMax()
        else:
            self.exp_min = 100.0
            self.exp_max = 100000.0

        if self.cam.Gain.GetAccessMode() == PySpin.RW:
            self.gain_min = self.cam.Gain.GetMin()
            self.gain_max = self.cam.Gain.GetMax()
        else:
            self.gain_min = 0.0
            self.gain_max = 20.0

        # Set initial exposure and gain (clamped to limits)
        self.set_exposure_us(exposure_us)
        self.set_gain_db(gain_db)

        # Cache image dimensions for reshaping image data
        self.height = self.cam.Height.GetValue()
        self.width = self.cam.Width.GetValue()

    # ----------------- camera control methods -----------------

    def set_exposure_us(self, exposure_us):
        """Set exposure in microseconds (clamped to camera limits)."""
        if self.cam.ExposureTime.GetAccessMode() == PySpin.RW:
            value = float(np.clip(exposure_us, self.exp_min, self.exp_max))
            self.cam.ExposureTime.SetValue(value)

    def set_gain_db(self, gain_db):
        """Set analog gain in dB (clamped to camera limits)."""
        if self.cam.Gain.GetAccessMode() == PySpin.RW:
            value = float(np.clip(gain_db, self.gain_min, self.gain_max))
            self.cam.Gain.SetValue(value)

    def exposure_from_slider(self, slider_val, slider_max=100):
        """Map integer slider value [0, slider_max] to exposure range [exp_min, exp_max]."""
        alpha = slider_val / float(slider_max)
        exposure_us = self.exp_min + alpha * (self.exp_max - self.exp_min)
        self.set_exposure_us(exposure_us)

    def gain_from_slider(self, slider_val, slider_max=100):
        """Map integer slider value [0, slider_max] to gain range [gain_min, gain_max]."""
        alpha = slider_val / float(slider_max)
        gain_db = self.gain_min + alpha * (self.gain_max - self.gain_min)
        self.set_gain_db(gain_db)

    # ----------------- acquisition methods -----------------

    def start(self):
        """Begin continuous acquisition."""
        if self.streaming:
            return
        try:
            self.cam.BeginAcquisition()
            self.streaming = True
        except PySpin.SpinnakerException as e:
            # Do not mark streaming; re-raise with clearer context
            msg = (
                f"BeginAcquisition failed: {e}\n"
                "Check USB cable, port, and power settings; try unplug/replug camera "
                "or a different USB3 port."
            )
            raise RuntimeError(msg) from e

    def stop(self):
        """Stop continuous acquisition if it was started."""
        if not self.streaming:
            return
        try:
            self.cam.EndAcquisition()
        except PySpin.SpinnakerException:
            # Ignore "has not started streaming" errors and similar on shutdown
            pass
        finally:
            self.streaming = False

    def close(self):
        """
        Deinitialize the camera and release system resources.

        Call this at the end of your program to clean up.
        """
        # Ensure acquisition is stopped
        self.stop()
        self.cam.DeInit()
        del self.cam
        self.cam_list.Clear()
        self.system.ReleaseInstanc