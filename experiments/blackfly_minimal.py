"""
blackfly_minimal_preview.py

Very simple FLIR Blackfly test:

- Initialize camera
- Start acquisition
- Grab frames with timeout
- Show live preview
- 'q' to quit

No ROI, no frame subtraction, no AVI, no trackbars.
This is just to confirm that streaming works cleanly.
"""

import time

import numpy as np
import cv2
import PySpin


class BlackflyCamera:
    def __init__(self, serial=None, exposure_us=5000.0, gain_db=0.0):
        self.streaming = False

        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        if self.cam_list.GetSize() == 0:
            self.cam_list.Clear()
            self.system.ReleaseInstance()
            raise RuntimeError("No Spinnaker/PySpin cameras detected.")

        if serial is None:
            self.cam = self.cam_list[0]
        else:
            self.cam = self.cam_list.GetBySerial(serial)

        self.cam.Init()

        # Continuous acquisition
        node = self.cam.AcquisitionMode.GetEntry(PySpin.AcquisitionMode_Continuous)
        self.cam.AcquisitionMode.SetIntValue(node.GetValue())

        # Manual exposure/gain
        if self.cam.ExposureAuto.GetAccessMode() == PySpin.RW:
            self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        if self.cam.GainAuto.GetAccessMode() == PySpin.RW:
            self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)

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

        self.set_exposure_us(exposure_us)
        self.set_gain_db(gain_db)

        self.height = self.cam.Height.GetValue()
        self.width = self.cam.Width.GetValue()

    def set_exposure_us(self, exposure_us):
        if self.cam.ExposureTime.GetAccessMode() == PySpin.RW:
            val = float(np.clip(exposure_us, self.exp_min, self.exp_max))
            self.cam.ExposureTime.SetValue(val)

    def set_gain_db(self, gain_db):
        if self.cam.Gain.GetAccessMode() == PySpin.RW:
            val = float(np.clip(gain_db, self.gain_min, self.gain_max))
            self.cam.Gain.SetValue(val)

    def start(self):
        if self.streaming:
            return
        try:
            self.cam.BeginAcquisition()
            self.streaming = True
        except PySpin.SpinnakerException as e:
            raise RuntimeError(f"BeginAcquisition failed: {e}") from e

    def stop(self):
        if not self.streaming:
            return
        try:
            self.cam.EndAcquisition()
        except PySpin.SpinnakerException:
            pass
        finally:
            self.streaming = False

    def close(self):
        self.stop()
        self.cam.DeInit()
        del self.cam
        self.cam_list.Clear()
        self.system.ReleaseInstance()

    def grab_frame(self, timeout_ms=1000):
        """
        Grab one frame with a timeout. If the camera is not delivering frames,
        this will raise a RuntimeError instead of blocking forever.
        """
        try:
            img = self.cam.GetNextImage(timeout_ms)
        except PySpin.SpinnakerException as e:
            raise RuntimeError(f"GetNextImage timeout or error: {e}") from e

        if img.IsIncomplete():
            status = img.GetImageStatus()
            img.Release()
            raise RuntimeError(f"Incomplete image: status {status}")

        if img.GetPixelFormat() != PySpin.PixelFormat_Mono8:
            conv = img.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
            data = conv.GetData().reshape(self.height, self.width)
        else:
            data = img.GetData().reshape(self.height, self.width)

        img.Release()
        return data.astype(np.float32)


def normalize_to_u8(img, clip_percent=1.0):
    low, high = np.percentile(img, [clip_percent, 100.0 - clip_percent])
    img_c = np.clip(img, low, high)
    img_n = (img_c - low) / (high - low + 1e-9)
    return (255.0 * img_n).astype(np.uint8)


def main():
    serial = None          # or your camera serial as a string
    exposure_us = 5000.0
    gain_db = 0.0

    cam = BlackflyCamera(serial=serial,
                         exposure_us=exposure_us,
                         gain_db=gain_db)

    try:
        cam.start()
        print("Streaming. Press 'q' in the preview window to quit.")

        while True:
            try:
                frame = cam.grab_frame(timeout_ms=1000)
            except RuntimeError as e:
                print(e)
                break

            u8 = normalize_to_u8(frame)
            cv2.imshow("Blackfly Minimal Preview", u8)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
