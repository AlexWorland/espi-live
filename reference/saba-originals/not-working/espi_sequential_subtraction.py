import imagingcontrol4 as ic4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

def rescale_intensity(img, min_val, max_val):
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val + 1e-9) * 255
    return img.astype(np.uint8)

def snap_sequence(grabber, sink, num_images):
    buffers = sink.snap_sequence(num_images, 500000)
    images = [b.numpy_wrap().copy() for b in buffers]
    for b in buffers:
        b.release()
    return [np.squeeze(i) for i in images]

def main():
    defaults = {
        "min": 0,
        "max": 20,
        "gain": 6.0,
        "exposure": 0.15  # seconds
    }

    with ic4.Library.init_context(api_log_level=ic4.LogLevel.INFO,
                                  log_targets=ic4.LogTarget.STDERR):
        dev = ic4.DeviceEnum.devices()[0]
        grabber = ic4.Grabber(dev)
        sink = ic4.SnapSink()

        # Stream setup
        grabber.stream_setup(sink, setup_option=ic4.StreamSetupOption.ACQUISITION_START)

        # --- Set static properties once ---
        dmap = grabber.device_property_map
        dmap.set_value(ic4.PropId.PIXEL_FORMAT, ic4.PixelFormat.Mono8)
        dmap.set_value(ic4.PropId.EXPOSURE_AUTO, "Off")
        dmap.set_value(ic4.PropId.GAIN_AUTO, "Off")

        # Apply initial gain & exposure
        dmap.set_value(ic4.PropId.GAIN, defaults["gain"])
        dmap.set_value(ic4.PropId.EXPOSURE_TIME, defaults["exposure"] * 1e6)

        # --- Interactive plot ---
        plt.ion()
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.35)

        # initialize with 1200x1920 blank image
        disp = ax.imshow(np.zeros((1200, 1920)), cmap='gray', vmin=0, vmax=255)
        ax.set_axis_off()

        # --- Sliders ---
        ax_min = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        ax_max = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        ax_gain = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        ax_exp  = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor='lightgoldenrodyellow')

        s_min = widgets.Slider(ax_min, 'Min', 0, 254, valinit=defaults["min"], valstep=1)
        s_max = widgets.Slider(ax_max, 'Max', 1, 255, valinit=defaults["max"], valstep=1)
        s_gain = widgets.Slider(ax_gain, 'Gain', 0, 6.9, valinit=defaults["gain"], valstep=0.1)
        s_exp  = widgets.Slider(ax_exp, 'Exposure (s)', 0.01, 0.25,
                                valinit=defaults["exposure"], valstep=0.01)

        settings = dict(defaults)

        def update_settings(val=None):
            settings["min"] = s_min.val
            settings["max"] = s_max.val
            settings["gain"] = s_gain.val
            settings["exposure"] = s_exp.val

            # Update gain & exposure on the device
            dmap.set_value(ic4.PropId.GAIN, settings["gain"])
            dmap.set_value(ic4.PropId.EXPOSURE_TIME, settings["exposure"] * 1e6)

        for s in [s_min, s_max, s_gain, s_exp]:
            s.on_changed(update_settings)

        try:
            while True:
                # Break if the figure window has been closed
                if not plt.fignum_exists(fig.number):
                    print("Window closed, exiting…")
                    break

                # Grab a sequence of 2 frames
                frames = snap_sequence(grabber, sink, 2)
                if len(frames) < 2:
                    continue
                f1, f2 = frames

                # Absolute difference
                diff = np.abs(f2.astype(np.float32) - f1.astype(np.float32))

                # Rescale contrast
                diff = rescale_intensity(diff, settings["min"], settings["max"])

                disp.set_data(diff)
                fig.canvas.draw_idle()
                plt.pause(0.05)

        except KeyboardInterrupt:
            print("Stopped live ESPI difference view.")

        # --- Explicit cleanup before context exits ---
        grabber.stream_stop()
        grabber.device_close()
        del sink, grabber, dev, dmap  # ensure cleanup while library is still alive

if __name__ == "__main__":
    main()
