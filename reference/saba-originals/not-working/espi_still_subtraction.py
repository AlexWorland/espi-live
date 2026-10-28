import imagingcontrol4 as ic4
import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.widgets as widgets
import os
from PIL import Image
from datetime import datetime


def rescale_intensity(img, min_val, max_val):
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val + 1e-5) * 255
    return img.astype(np.uint8)


def plot_image_with_min_max_sliders(img):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    disp_img = ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.set_axis_off()

    ax_min = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_max = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')

    s_min = widgets.Slider(ax_min, 'Min', 0, 254, valinit=0, valstep=1)
    s_max = widgets.Slider(ax_max, 'Max', 1, 255, valinit=10, valstep=1)  # default now 10

    def update(val):
        min_val = s_min.val
        max_val = s_max.val
        if min_val >= max_val:
            return
        new_img = rescale_intensity(img, min_val, max_val)
        disp_img.set_data(new_img)
        fig.canvas.draw_idle()

    s_min.on_changed(update)
    s_max.on_changed(update)

    plt.show()


def snap_frame(grabber, sink, gain, exposure_time):
    grabber.device_property_map.set_value(ic4.PropId.EXPOSURE_AUTO, "Off")
    grabber.device_property_map.set_value(ic4.PropId.EXPOSURE_TIME, exposure_time)
    grabber.device_property_map.set_value(ic4.PropId.GAIN_AUTO, "Off")
    grabber.device_property_map.set_value(ic4.PropId.GAIN, gain)

    buffer = sink.snap_single(500000)
    image_array = buffer.numpy_wrap().copy()
    buffer.release()
    return image_array


def snap_sequence(grabber, sink, num_images, gain, exposure_time):
    grabber.device_property_map.set_value(ic4.PropId.EXPOSURE_AUTO, "Off")
    grabber.device_property_map.set_value(ic4.PropId.EXPOSURE_TIME, exposure_time)
    grabber.device_property_map.set_value(ic4.PropId.GAIN_AUTO, "Off")
    grabber.device_property_map.set_value(ic4.PropId.GAIN, gain)

    buffers = sink.snap_sequence(num_images, 500000)
    images = [b.numpy_wrap().copy() for b in buffers]
    for b in buffers:
        b.release()
    return images


def save_tiff(img, filename):
    img_output = Image.fromarray(img)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, filename + ".tiff")
    img_output.save(output_path, format='TIFF', mode='F')
    print("saved", output_path)


def main():
    gain = 1.0
    num_sequence = 20
    live_avg_frames = 1  # how many frames to average for live view
    exposure_time = 250000.0  # microseconds

    with ic4.Library.init_context(api_log_level=ic4.LogLevel.INFO, log_targets=ic4.LogTarget.STDERR):
        # initialize camera 
        dev = ic4.DeviceEnum.devices()[0]

        # prompt user to name experimental run
        output_name = input("Type in run name and press Enter...")

        # collect stills
        input("Press Enter to collect still sequence...")

        grabber1 = ic4.Grabber(dev)
        sink1 = ic4.SnapSink()
        grabber1.stream_setup(sink1, setup_option=ic4.StreamSetupOption.ACQUISITION_START)

        still_sequence = snap_sequence(grabber1, sink1, num_sequence, gain, exposure_time)
        still_sequence = [np.squeeze(i) for i in still_sequence]

        grabber1.stream_stop()
        grabber1.device_close()
        del sink1
        del grabber1

        still_average = np.mean(still_sequence, axis=0)

        # live subtraction
        input("Press Enter to start live subtraction view...")
        grabber2 = ic4.Grabber(dev)
        sink2 = ic4.SnapSink()
        grabber2.stream_setup(sink2, setup_option=ic4.StreamSetupOption.ACQUISITION_START)

        # Set up interactive plot with sliders
        plt.ion()
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)
        diff_display = np.zeros_like(still_average, dtype=np.uint8)
        img_disp = ax.imshow(diff_display, cmap='gray', vmin=0, vmax=255)
        ax.set_axis_off()

        # Sliders for contrast adjustment
        ax_min = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        ax_max = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        s_min = widgets.Slider(ax_min, 'Min', 0, 254, valinit=0, valstep=1)
        s_max = widgets.Slider(ax_max, 'Max', 1, 255, valinit=10, valstep=1)  # default now 10

        # update function for sliders
        contrast = {'min': 0, 'max': 10}  # match slider defaults
        def update_contrast(val):
            min_val = s_min.val
            max_val = s_max.val
            if min_val < max_val:
                contrast['min'] = min_val
                contrast['max'] = max_val
        s_min.on_changed(update_contrast)
        s_max.on_changed(update_contrast)

        try:
            image_num = 0
            while plt.fignum_exists(fig.number):  # loop runs until figure is closed
                # Grab multiple frames and average
                frames = [np.squeeze(snap_frame(grabber2, sink2, gain, exposure_time)) for _ in range(live_avg_frames)]
                moving_average = np.mean(frames, axis=0)

                # Subtract still image
                diff_img = np.abs(moving_average - still_average)

                save_tiff(diff_img, 'image' + str(image_num))
                image_num += 1
                image_num = image_num%5

                # Apply current contrast settings
                diff_img = rescale_intensity(diff_img, contrast['min'], contrast['max'])

                img_disp.set_data(diff_img)
                fig.canvas.draw_idle()
                plt.pause(0.5)

        except KeyboardInterrupt:
            print("Live view stopped by keyboard interrupt.")

        print("Live view stopped (window closed).")

        grabber2.stream_stop()
        grabber2.device_close()
        del sink2
        del grabber2
        del dev


if __name__ == "__main__":
    main()
