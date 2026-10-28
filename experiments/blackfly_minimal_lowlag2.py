"""
blackfly_minimal_lowlag2.py

Minimal FLIR Blackfly preview with more aggressive low-latency tweaks:

- Stream buffer handling mode = NewestOnly (drop old frames)
- Cheap per-frame normalization (no percentiles)
- Optional "flush" of extra frames per loop to stay near real-time
"""

import numpy as np
import cv2
import PySpin


class BlackflyCamera:
    def __init__(self, serial=None, exposure_us=3000.0, gain_db=0.0):
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

        # Stream node map for low-latency buffer handling
        s_nodemap = self.cam.GetTLStreamNodeMap()
        handling_mode = PySpin.CEnumerationPtr(s_nodemap.GetNode("StreamBufferHandlingMode"))
        if PySpin.IsReadable(handling_mode) and PySpin.IsWritable(handling_mode):
            newest_only = handling_mode.GetEntryByName("NewestOnly")
            if PySpin.IsReadable(newest_only):
                handling_mode.SetIntValue(newest_only.GetValue())

        # Acquisition mode: Continuous
        nodemap = self.cam.GetNodeMap()
        acq_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        if PySpin.IsReadable(acq_mode) and PySpin.IsWritable(acq_mode):
            cont_entry = acq_mode.GetEntryByName("Continuous")
            acq_mode.SetIntValue(cont_entry.GetValue())

        # Disable auto exposure/gain
        exp_auto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
        if PySpin.IsReadable(exp_auto) and PySpin.IsWritable(exp_auto):
            off_entry = exp_auto.GetEntryByName("Off")
            exp_auto.SetIntValue(off_entry.GetValue())

        gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
        if PySpin.IsReadable(gain_auto) and PySpin.IsWritable(gain_auto):
            off_entry = gain_auto.GetEntryByName("Off")
            gain_auto.SetIntValue(off_entry.GetValue())

        # Cache exposure and gain limits
        exp_node = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
        if PySpin.IsReadable(exp_node) and PySpin.IsWritable(exp_node):
            self.exp_min = exp_node.GetMin()
            self.exp_max = exp_node.GetMax()
        else:
            self.exp_min, self.exp_max = 100.0, 100000.0

        gain_node = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
        if PySpin.IsReadable(gain_node) and PySpin.IsWritable(gain_node):
            self.gain_min = gain_node.GetMin()
            self.gain_max = gain_node.GetMax()
        else:
            self.gain_min, self.gain_max = 0.0, 20.0

        self.exp_node = exp_node
        self.gain_node = gain_node
        self.set_exposure_us(exposure_us)
        self.set_gain_db(gain_db)

        # (Optional) Hardware ROI – uncomment and adjust if you want smaller frames:
        # width_node = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
        # height_node = PySpin.CIntegerPtr(nodemap.GetNode("Height"))
        # offsetx_node = PySpin.CIntegerPtr(nodemap.GetNode("OffsetX"))
        # offsety_node = PySpin.CIntegerPtr(nodemap.GetNode("OffsetY"))
        # if (PySpin.IsWritable(width_node) and PySpin.IsWritable(height_node) and
        #     PySpin.IsWritable(offsetx_node) and PySpin.IsWritable(offsety_node)):
        #     width_node.SetValue(min(width_node.GetMax(), 1024))
        #     height_node.SetValue(min(height_node.GetMax(), 1024))
        #     offsetx_node.SetValue(0)
        #     offsety_node.SetValue(0)

        self.height = self.cam.Height.GetValue()
        self.width = self.cam.Width.GetValue()

    def set_exposure_us(self, exposure_us):
        if self.exp_node is not None and PySpin.IsWritable(self.exp_node):
            val = float(np.clip(exposure_us, self.exp_min, self.exp_max))
            self.exp_node.SetValue(val)

    def set_gain_db(self, gain_db):
        if self.gain_node is not None and PySpin.IsWritable(self.gain_node):
            val = float(np.clip(gain_db, self.gain_min, self.gain_max))
            self.gain_node.SetValue(val)

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
        try:
            self.cam.DeInit()
        except PySpin.SpinnakerException:
            pass
        del self.cam
        try:
            self.cam_list.Clear()
        except PySpin.SpinnakerException:
            pass
        try:
            self.system.ReleaseInstance()
        except PySpin.SpinnakerException:
            pass

    def grab_frame(self, timeout_ms=100):
        """
        Grab one frame with timeout. If the camera is not delivering frames,
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


def normalize_fast_to_u8(img):
    """
    Fast normalization using min/max (no percentiles).
    """
    mn = float(img.min())
    mx = float(img.max())
    if mx <= mn:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - mn) * (255.0 / (mx - mn))
    return out.astype(np.uint8)


def main():
    serial = None          # or your camera serial string
    exposure_us = 3000.0   # lower exposure → lower inherent latency
    gain_db = 0.0

    # How aggressive to be in flushing old frames:
    # 0 = just one frame per loop; 1 or 2 = discard extra frames so you stay near real-time
    flush_extra_frames = 1

    cam = BlackflyCamera(serial=serial,
                         exposure_us=exposure_us,
                         gain_db=gain_db)

    try:
        cam.start()
        print("Streaming (NewestOnly, fast normalize). Press 'q' in the preview window to quit.")

        while True:
            try:
                frame = cam.grab_frame(timeout_ms=100)
            except RuntimeError as e:
                print(e)
                break

            # Optionally flush extra frames to keep latency down
            for _ in range(flush_extra_frames):
                try:
                    frame = cam.grab_frame(timeout_ms=1)
                except RuntimeError:
                    break

            u8 = normalize_fast_to_u8(frame)
            cv2.imshow("Blackfly Low-Lag Preview", u8)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
