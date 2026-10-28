import os

import PySpin
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from PIL import Image


def rescale_intensity(img, min_val, max_val):
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val + 1e-9) * 255
    return img.astype(np.uint8)


def get_camera():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        raise RuntimeError("No PySpin-compatible cameras detected.")

    cam = cam_list[0]
    cam.Init()
    cam_list.Clear()
    return system, cam


def set_enum_value(nodemap, node_name, entry_name):
    node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
    if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
        return
    entry = PySpin.CEnumEntryPtr(node.GetEntryByName(entry_name))
    if not PySpin.IsReadable(entry):
        return
    node.SetIntValue(entry.GetValue())


def clamp_value(node, value):
    return max(node.GetMin(), min(node.GetMax(), value))


def apply_static_config(nodemap):
    set_enum_value(nodemap, "PixelFormat", "Mono8")
    set_enum_value(nodemap, "ExposureAuto", "Off")
    set_enum_value(nodemap, "GainAuto", "Off")


def apply_gain_exposure(nodemap, gain_db, exposure_seconds):
    gain_node = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
    if PySpin.IsReadable(gain_node) and PySpin.IsWritable(gain_node):
        gain_node.SetValue(clamp_value(gain_node, gain_db))

    exposure_node = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
    if PySpin.IsReadable(exposure_node) and PySpin.IsWritable(exposure_node):
        exposure_us = exposure_seconds * 1e6
        exposure_node.SetValue(clamp_value(exposure_node, exposure_us))


def configure_continuous_acquisition(nodemap):
    node_acq_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
    if PySpin.IsReadable(node_acq_mode) and PySpin.IsWritable(node_acq_mode):
        entry_continuous = node_acq_mode.GetEntryByName("Continuous")
        if PySpin.IsReadable(entry_continuous):
            node_acq_mode.SetIntValue(entry_continuous.GetValue())


def snap_sequence(cam, num_images, timeout_ms=1000):
    frames = []
    for _ in range(num_images):
        image = cam.GetNextImage(timeout_ms)
        try:
            if image.IsIncomplete():
                print(f"Discarded incomplete image: {image.GetImageStatus()}")
                continue
            frames.append(image.GetNDArray().copy())
        finally:
            image.Release()
    return [np.squeeze(f) for f in frames]


def save_tiff(img, filename):
    img_output = Image.fromarray(np.array(img, dtype=np.float32))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f"{filename}.tiff")
    img_output.save(output_path, format="TIFF", mode="F")
    print("saved", output_path)


def main():
    gain = 1.0
    num_sequence = 20
    live_avg_frames = 1
    exposure_time_sec = 0.25

    system, cam = get_camera()
    nodemap = cam.GetNodeMap()
    apply_static_config(nodemap)
    configure_continuous_acquisition(nodemap)
    apply_gain_exposure(nodemap, gain, exposure_time_sec)

    try:
        output_name = input("Type in run name and press Enter...")
        input("Press Enter to collect still sequence...")

        cam.BeginAcquisition()
        still_sequence = snap_sequence(cam, num_sequence)
        cam.EndAcquisition()

        if not still_sequence:
            raise RuntimeError("Failed to capture still sequence.")

        still_average = np.mean(still_sequence, axis=0)

        input("Press Enter to start live subtraction view...")

        plt.ion()
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)
        diff_display = np.zeros_like(still_average, dtype=np.uint8)
        img_disp = ax.imshow(diff_display, cmap="gray", vmin=0, vmax=255)
        ax.set_axis_off()

        ax_min = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor="lightgoldenrodyellow")
        ax_max = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor="lightgoldenrodyellow")
        s_min = widgets.Slider(ax_min, "Min", 0, 254, valinit=0, valstep=1)
        s_max = widgets.Slider(ax_max, "Max", 1, 255, valinit=10, valstep=1)

        contrast = {"min": 0, "max": 10}

        def update_contrast(_=None):
            min_val = s_min.val
            max_val = s_max.val
            if min_val < max_val:
                contrast["min"] = min_val
                contrast["max"] = max_val

        s_min.on_changed(update_contrast)
        s_max.on_changed(update_contrast)

        cam.BeginAcquisition()
        try:
            image_num = 0
            prefix = output_name.strip() or "run"
            while plt.fignum_exists(fig.number):
                frames = snap_sequence(cam, live_avg_frames)
                if not frames:
                    continue
                moving_average = np.mean(frames, axis=0)
                diff_img = np.abs(moving_average - still_average)

                save_tiff(diff_img, f"{prefix}_image{image_num}")
                image_num = (image_num + 1) % 5

                display_img = rescale_intensity(diff_img, contrast["min"], contrast["max"])
                img_disp.set_data(display_img)
                fig.canvas.draw_idle()
                plt.pause(0.5)
        except KeyboardInterrupt:
            print("Live view stopped by keyboard interrupt.")
        finally:
            try:
                cam.EndAcquisition()
            except PySpin.SpinnakerException:
                pass

        print("Live view stopped (window closed).")
    finally:
        cam.DeInit()
        del cam
        system.ReleaseInstance()


if __name__ == "__main__":
    main()
