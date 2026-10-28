import time

import PySpin
import matplotlib.pyplot as plt
import numpy as np


def main():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        raise RuntimeError("No cameras found")

    cam = cam_list[0]
    cam.Init()
    cam_list.Clear()

    # Minimal configuration: continuous mode, Mono8
    nodemap = cam.GetNodeMap()
    acq_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
    if PySpin.IsWritable(acq_mode):
        entry = acq_mode.GetEntryByName("Continuous")
        if PySpin.IsReadable(entry):
            acq_mode.SetIntValue(entry.GetValue())
    pix_fmt = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
    if PySpin.IsWritable(pix_fmt):
        mono8 = pix_fmt.GetEntryByName("Mono8")
        if PySpin.IsReadable(mono8):
            pix_fmt.SetIntValue(mono8.GetValue())

    plt.ion()
    fig, ax = plt.subplots()
    img_disp = None
    background = None
    stats = {
        "grab": [0.0, 0],
        "reshape": [0.0, 0],
        "update": [0.0, 0],
        "draw_render": [0.0, 0],
        "draw_flush": [0.0, 0],
        "draw_pause": [0.0, 0],
        "loop": [0.0, 0],
    }
    report_interval = 5.0
    next_report = time.perf_counter() + report_interval

    try:
        cam.BeginAcquisition()
        while True:
            if not plt.fignum_exists(fig.number):
                break

            loop_start = time.perf_counter()

            t0 = time.perf_counter()
            image = cam.GetNextImage()
            stats["grab"][0] += time.perf_counter() - t0
            stats["grab"][1] += 1
            if image.IsIncomplete():
                image.Release()
                continue

            t0 = time.perf_counter()
            height = image.GetHeight()
            width = image.GetWidth()
            channels = max(1, image.GetBitsPerPixel() // 8)
            frame = np.frombuffer(image.GetData(), dtype=np.uint8)
            frame = frame.reshape((height, width, channels)) if channels > 1 else frame.reshape((height, width))
            if channels > 3:  # Drop alpha/extra channels for display
                frame = frame[:, :, :3]
            if frame.ndim == 2:  # Avoid colormap path (which allocates RGBA floats)
                frame = np.repeat(frame[:, :, None], 3, axis=2)
            if frame.dtype != np.uint8:  # Ensure byte data for matplotlib
                frame = frame.astype(np.uint8, copy=False)
            stats["reshape"][0] += time.perf_counter() - t0
            stats["reshape"][1] += 1

            t0 = time.perf_counter()
            if img_disp is None:
                img_disp = ax.imshow(frame)
                img_disp.set_animated(True)
                ax.set_axis_off()
                fig.canvas.draw()  # draw once to cache background
                background = fig.canvas.copy_from_bbox(ax.bbox)
            else:
                img_disp.set_data(frame)
            stats["update"][0] += time.perf_counter() - t0
            stats["update"][1] += 1

            t0 = time.perf_counter()
            if background is not None:
                fig.canvas.restore_region(background)
                ax.draw_artist(img_disp)
                fig.canvas.blit(ax.bbox)
            else:
                fig.canvas.draw()
            stats["draw_render"][0] += time.perf_counter() - t0
            stats["draw_render"][1] += 1

            t0 = time.perf_counter()
            fig.canvas.flush_events()
            stats["draw_flush"][0] += time.perf_counter() - t0
            stats["draw_flush"][1] += 1

            t0 = time.perf_counter()
            plt.pause(0.0001)
            stats["draw_pause"][0] += time.perf_counter() - t0
            stats["draw_pause"][1] += 1

            image.Release()

            stats["loop"][0] += time.perf_counter() - loop_start
            stats["loop"][1] += 1

            now = time.perf_counter()
            if now >= next_report:
                print("\nStep timing averages (last {:.0f}s):".format(report_interval))
                print("{:<12} {:>10}".format("step", "avg ms"))
                print("-" * 22)
                for key in (
                    "grab",
                    "reshape",
                    "update",
                    "draw_render",
                    "draw_flush",
                    "draw_pause",
                    "loop",
                ):
                    total, count = stats[key]
                    avg_ms = (total / count * 1000.0) if count else 0.0
                    print("{:<12} {:>10.3f}".format(key, avg_ms))
                    stats[key] = [0.0, 0]  # reset after reporting
                next_report = now + report_interval
    except KeyboardInterrupt:
        pass
    finally:
        cam.EndAcquisition()
        cam.DeInit()
        del cam
        system.ReleaseInstance()


if __name__ == "__main__":
    main()
