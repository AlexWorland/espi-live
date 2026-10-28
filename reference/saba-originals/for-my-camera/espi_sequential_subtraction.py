import functools
import logging
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import PySpin


def log_timing(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logging.info("%s took %.3f ms", func.__name__, elapsed_ms)

    return wrapper


@log_timing
def rescale_intensity(img, min_val, max_val):
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val + 1e-9) * 255
    return img.astype(np.uint8)


@log_timing
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


@log_timing
def set_enum_value(nodemap, node_name, entry_name):
    node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
    if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
        return
    entry = PySpin.CEnumEntryPtr(node.GetEntryByName(entry_name))
    if not PySpin.IsReadable(entry):
        return
    node.SetIntValue(entry.GetValue())


@log_timing
def clamp_value(node, value):
    return max(node.GetMin(), min(node.GetMax(), value))


@log_timing
def apply_static_config(nodemap):
    set_enum_value(nodemap, "PixelFormat", "Mono8")
    set_enum_value(nodemap, "ExposureAuto", "Off")
    set_enum_value(nodemap, "GainAuto", "Off")


@log_timing
def apply_gain_exposure(nodemap, gain_db, exposure_seconds):
    gain_node = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
    if PySpin.IsReadable(gain_node) and PySpin.IsWritable(gain_node):
        gain_node.SetValue(clamp_value(gain_node, gain_db))

    exposure_node = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
    if PySpin.IsReadable(exposure_node) and PySpin.IsWritable(exposure_node):
        exposure_us = exposure_seconds * 1e6
        exposure_node.SetValue(clamp_value(exposure_node, exposure_us))


@log_timing
def apply_frame_rate(nodemap, fps):
    """
    Request a target acquisition frame rate if the camera allows it.
    Some cameras need AcquisitionFrameRateEnable toggled before setting the value.
    """
    try:
        enable_node = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnable"))
        if PySpin.IsWritable(enable_node):
            enable_node.SetValue(True)

        fps_node = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
        if PySpin.IsReadable(fps_node) and PySpin.IsWritable(fps_node):
            fps_node.SetValue(clamp_value(fps_node, fps))
            logging.info("Requested frame rate: %.2f fps (camera clamp min=%.2f max=%.2f)", fps_node.GetValue(), fps_node.GetMin(), fps_node.GetMax())
        else:
            logging.warning("Camera does not expose writable AcquisitionFrameRate")
    except PySpin.SpinnakerException as exc:
        logging.warning("Unable to set frame rate: %s", exc)


@log_timing
def clamp_exposure_for_fps(exposure_seconds, fps):
    """
    Keep exposure under the frame period to prioritize target FPS.
    Uses a 90% safety margin to leave time for transfer/draw.
    """
    if fps <= 0:
        return exposure_seconds
    max_exposure = 0.9 / fps
    return min(exposure_seconds, max_exposure)


@log_timing
def apply_stream_settings(cam):
    """
    Favor freshest frames to keep latency low.
    """
    try:
        s_node_map = cam.GetTLStreamNodeMap()
        handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode("StreamBufferHandlingMode"))
        if PySpin.IsWritable(handling_mode):
            newest_only = handling_mode.GetEntryByName("NewestOnly")
            if PySpin.IsReadable(newest_only):
                handling_mode.SetIntValue(newest_only.GetValue())
                logging.info("StreamBufferHandlingMode set to NewestOnly")
    except PySpin.SpinnakerException as exc:
        logging.warning("Unable to set stream handling mode: %s", exc)


@log_timing
def grab_frame(cam, timeout_ms=1000):
    image = cam.GetNextImage(timeout_ms)
    try:
        if image.IsIncomplete():
            print(f"Discarded incomplete image: {image.GetImageStatus()}")
            return None
        return np.squeeze(image.GetNDArray().copy())
    finally:
        image.Release()


@log_timing
def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    # defaults = {"min": 0, "max": 20, "gain": 6.0, "exposure": 0.15}
    defaults = {"min": 0, "max": 20, "gain": 9.0, "exposure": 0.05, "fps": 30.0}
    defaults["exposure"] = clamp_exposure_for_fps(defaults["exposure"], defaults["fps"])

    system, cam = get_camera()
    nodemap = cam.GetNodeMap()
    apply_static_config(nodemap)
    apply_gain_exposure(nodemap, defaults["gain"], defaults["exposure"])
    apply_frame_rate(nodemap, defaults["fps"])
    apply_stream_settings(cam)

    # Configure acquisition mode once
    node_acq_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
    if PySpin.IsReadable(node_acq_mode) and PySpin.IsWritable(node_acq_mode):
        entry_continuous = node_acq_mode.GetEntryByName("Continuous")
        if PySpin.IsReadable(entry_continuous):
            node_acq_mode.SetIntValue(entry_continuous.GetValue())

    plt.ion()
    fig, ax = plt.subplots()
    plt.show(block=False)
    plt.subplots_adjust(left=0.25, bottom=0.40)
    # Match the camera's native resolution (2048 x 3072) for the initial display buffer
    disp = ax.imshow(np.zeros((2048, 3072)), cmap="gray", vmin=0, vmax=255)
    ax.set_axis_off()

    ax_min = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    ax_max = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    ax_gain = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    ax_exp = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    ax_fps = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor="lightgoldenrodyellow")

    s_min = widgets.Slider(ax_min, "Min", 0, 254, valinit=defaults["min"], valstep=1)
    s_max = widgets.Slider(ax_max, "Max", 1, 255, valinit=defaults["max"], valstep=1)
    s_gain = widgets.Slider(ax_gain, "Gain", 0, 12.0, valinit=defaults["gain"], valstep=0.1)
    s_exp = widgets.Slider(
        ax_exp, "Exposure (s)", 0.01, 0.25, valinit=defaults["exposure"], valstep=0.01
    )
    s_fps = widgets.Slider(ax_fps, "FPS", 1.0, 120.0, valinit=defaults["fps"], valstep=1.0)

    settings = dict(defaults)

    @log_timing
    def update_settings(_=None):
        settings["min"] = s_min.val
        settings["max"] = s_max.val
        if settings["min"] >= settings["max"]:
            settings["max"] = settings["min"] + 1
            s_max.set_val(settings["max"])

        settings["fps"] = s_fps.val
        gain = s_gain.val
        # Clamp exposure to respect target FPS; update slider if we clamp
        exposure = clamp_exposure_for_fps(s_exp.val, settings["fps"])
        if exposure != s_exp.val:
            s_exp.set_val(exposure)
        settings["gain"] = gain
        settings["exposure"] = exposure
        apply_frame_rate(nodemap, settings["fps"])
        apply_gain_exposure(nodemap, gain, exposure)

    for slider in (s_min, s_max, s_gain, s_exp, s_fps):
        slider.on_changed(update_settings)

    try:
        cam.BeginAcquisition()
        prev_frame = None
        logging.info("Acquisition started. Priming first frame...")
        # Prime buffer so we always subtract against the most recent image
        while prev_frame is None:
            prev_frame = grab_frame(cam)

        frame_idx = 0
        last_frame_time = time.perf_counter()

        while True:
            if not plt.fignum_exists(fig.number):
                print("Window closed, exiting")
                break
            current_frame = grab_frame(cam)
            if current_frame is None:
                continue

            now = time.perf_counter()
            frame_dt = (now - last_frame_time) * 1000.0  # ms
            last_frame_time = now

            diff = np.abs(current_frame.astype(np.float32) - prev_frame.astype(np.float32))
            diff = rescale_intensity(diff, settings["min"], settings["max"])
            disp.set_data(diff)
            # Draw + flush is faster than a longer pause and keeps events responsive
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)
            prev_frame = current_frame

            frame_idx += 1
            instantaneous_fps = 1000.0 / frame_dt if frame_dt > 0 else float("inf")
            logging.info(
                "frame %d | instantaneous_fps=%.1f | frame_dt=%.2f ms | target_fps=%.1f | gain=%.2f dB | exposure=%.3f s",
                frame_idx,
                instantaneous_fps,
                frame_dt,
                settings["fps"],
                settings["gain"],
                settings["exposure"],
            )
    except KeyboardInterrupt:
        print("Stopped live ESPI difference view.")
    finally:
        try:
            cam.EndAcquisition()
        except PySpin.SpinnakerException:
            pass
        cam.DeInit()
        del cam
        system.ReleaseInstance()


if __name__ == "__main__":
    main()
