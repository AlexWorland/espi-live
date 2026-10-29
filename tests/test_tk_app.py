"""Camera-free checks for the legacy Tk front end (``espi_live.app``).

Real (withdrawn) Tk windows are driven directly; the camera, the recorder, the
log file, dialogs and the clock are replaced by fakes, so nothing opens a
camera or blocks on user input.

Run from the project root:  PYTHONPATH=src python -m unittest tests.test_tk_app
"""

import contextlib
import io
import math
import os
import queue
import re
import runpy
import tempfile
import tkinter as tk
import unittest
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from espi_live import app
from espi_live.camera_models import CameraIdentity

REAL_TK = tk.Tk
REAL_CONTROLS = app.Controls


def withdrawn_tk(*args, **kwargs):
    """Create a Tk root that is hidden immediately.

    Args: Forwarded to tkinter.Tk.
    Returns: The withdrawn root window.
    """
    root = REAL_TK(*args, **kwargs)
    root.withdraw()
    return root


def build_controls():
    """Create real Controls with every window hidden.

    Args: None.
    Returns: A Controls instance; the caller must close() it.
    """
    with mock.patch.object(tk, "Tk", withdrawn_tk):
        controls = REAL_CONTROLS()
    controls.log_window.withdraw()
    return controls


def destroy_quietly(controls):
    """Destroy a Controls root unless it is already gone.

    Args: controls is a Controls instance.
    Returns: None.
    """
    try:
        controls.root.destroy()
    except tk.TclError:
        pass


def entry_for(scale):
    """Find the exact-value entry that shares a slider's row.

    Args: scale is a Tk Scale built by Controls._scale.
    Returns: The Tk Entry beside it.
    """
    return [w for w in scale.master.winfo_children()
            if isinstance(w, tk.Entry)][0]


def type_into(entry, text):
    """Replace an entry's text and commit it by leaving the field.

    Args: entry is a Tk Entry; text is the string to type.
    Returns: None.
    """
    entry.delete(0, "end")
    entry.insert(0, text)
    entry.event_generate("<FocusOut>")


def mono_frame(value=100):
    """Build a small Mono8 test frame with a horizontal gradient.

    Args: value is added to the gradient (kept inside 0..255).
    Returns: A 64x96 uint8 array.
    """
    return np.clip(np.tile(np.arange(96), (64, 1)) + (value - 100),
                   0, 255).astype(np.uint8)


class ControlsTests(unittest.TestCase):
    """Control panel widgets, settings, status text and log window."""

    def setUp(self):
        """Create hidden controls.

        Args: None.
        Returns: None.
        """
        self.controls = build_controls()
        self.addCleanup(destroy_quietly, self.controls)

    def send_wheel(self, widget, delta):
        """Deliver a mouse-wheel event to the app-wide wheel binding.

        Args: widget is the widget under the pointer; delta is the wheel delta.
        Returns: None.
        """
        script = self.controls.root.bind_all("<MouseWheel>")
        command = re.search(r"\[(\w+) ", script)[1]
        args = ["0"] * 19
        args[14] = str(widget)
        args[18] = str(delta)
        self.controls.root.tk.call(command, *args)

    def test_defaults_match_documented_settings(self):
        """Fresh controls start in Raw mode with the documented defaults.

        Args: None.
        Returns: None.
        """
        controls = self.controls
        self.assertTrue(controls.running)
        self.assertEqual(controls.mode.get(), "Raw")
        self.assertEqual(controls.exposure.get(), 8000)
        self.assertEqual(controls.gain.get(), 0.0)
        self.assertEqual(controls.fps.get(), 30.0)
        self.assertEqual(controls.record_text.get(), "Recording: off")
        self.assertGreaterEqual(controls.viewer_width, 320)

    def test_queue_action_reaches_action_queue(self):
        """Queued shortcut letters come out in order.

        Args: None.
        Returns: None.
        """
        self.controls.queue_action("s")
        self.controls.queue_action("f")
        self.assertEqual(self.controls.actions.get_nowait(), "s")
        self.assertEqual(self.controls.actions.get_nowait(), "f")

    def test_capture_buttons_queue_their_shortcuts(self):
        """The live-capture button queues the 'l' shortcut.

        Args: None.
        Returns: None.
        """
        self.controls.switch_live_button.invoke()
        self.assertEqual(self.controls.actions.get_nowait(), "l")

    def test_quit_button_stops_the_loop(self):
        """The Quit button clears the running flag.

        Args: None.
        Returns: None.
        """
        self.controls.quit_button.invoke()
        self.assertFalse(self.controls.running)

    def test_stop_clears_running_flag(self):
        """stop() asks the acquisition loop to end.

        Args: None.
        Returns: None.
        """
        self.controls.stop()
        self.assertFalse(self.controls.running)

    def test_status_labels_rewrap_when_root_resizes(self):
        """A root resize re-wraps status labels, never below 180 px.

        Args: None.
        Returns: None.
        """
        controls = self.controls
        controls._resize_labels(SimpleNamespace(widget=controls.root, width=500))
        self.assertEqual(int(controls.labels[0].cget("wraplength")), 465)
        self.assertEqual(int(controls.shortcut_label.cget("wraplength")), 465)
        controls._resize_labels(SimpleNamespace(widget=controls.root, width=100))
        self.assertEqual(int(controls.labels[0].cget("wraplength")), 180)

    def test_status_labels_ignore_child_resize(self):
        """A resize of some other widget leaves label wrapping alone.

        Args: None.
        Returns: None.
        """
        controls = self.controls
        before = int(controls.labels[0].cget("wraplength"))
        controls._resize_labels(
            SimpleNamespace(widget=controls.log_window, width=900))
        self.assertEqual(int(controls.labels[0].cget("wraplength")), before)

    def test_configure_event_rewraps_labels(self):
        """The bound <Configure> handler is wired to the root window.

        Args: None.
        Returns: None.
        """
        self.controls.root.event_generate("<Configure>", width=500)
        self.assertEqual(
            int(self.controls.labels[0].cget("wraplength")), 465)

    def test_text_size_slider_changes_font(self):
        """Moving the Text size slider resizes the shared font.

        Args: None.
        Returns: None.
        """
        controls = self.controls
        slider = [w for w in controls.root.winfo_children()
                  if isinstance(w, tk.Scale)][0]
        controls.root.tk.call(slider.cget("command"), "20.0")
        self.assertEqual(int(controls.font.cget("size")), 20)

    def test_slider_value_is_mirrored_in_entry(self):
        """Changing a slider variable updates its exact-value entry.

        Args: None.
        Returns: None.
        """
        controls = self.controls
        controls.exposure.set(1234)
        controls.gain.set(2.5)
        self.assertEqual(entry_for(controls.exposure_scale).get(), "1234")
        self.assertEqual(entry_for(controls.gain_scale).get(), "2.5")

    def test_typed_value_is_applied_to_slider(self):
        """A typed number inside the range moves the slider.

        Args: None.
        Returns: None.
        """
        type_into(entry_for(self.controls.exposure_scale), "1500")
        self.assertEqual(self.controls.exposure.get(), 1500)

    def test_typed_value_is_clamped_to_slider_range(self):
        """A typed number outside the range is limited to the slider ends.

        Args: None.
        Returns: None.
        """
        type_into(entry_for(self.controls.exposure_scale), "99999999")
        self.assertEqual(self.controls.exposure.get(), 50000)
        type_into(entry_for(self.controls.exposure_scale), "-5")
        self.assertEqual(self.controls.exposure.get(), 0)

    def test_typed_non_finite_value_is_replaced(self):
        """NaN text is rejected and the entry shows the current value.

        Args: None.
        Returns: None.
        """
        entry = entry_for(self.controls.exposure_scale)
        type_into(entry, "nan")
        self.assertEqual(entry.get(), "8000")
        self.assertEqual(self.controls.exposure.get(), 8000)

    def test_typed_garbage_is_replaced(self):
        """Non-numeric text is rejected and the entry shows the current value.

        Args: None.
        Returns: None.
        """
        entry = entry_for(self.controls.gain_scale)
        type_into(entry, "abc")
        self.assertEqual(entry.get(), "0.0")

    def test_mouse_wheel_over_sliders_scrolls_the_list(self):
        """The wheel scrolls the controls when the pointer is over them.

        Args: None.
        Returns: None.
        """
        canvas = self.controls.sliders.master
        with mock.patch.object(canvas, "yview_scroll") as scroll:
            self.send_wheel(self.controls.sliders, -120)
        scroll.assert_called_once_with(1, "units")

    def test_mouse_wheel_elsewhere_does_not_scroll(self):
        """The wheel is ignored over widgets outside the slider list.

        Args: None.
        Returns: None.
        """
        canvas = self.controls.sliders.master
        with mock.patch.object(canvas, "yview_scroll") as scroll:
            self.send_wheel(self.controls.shortcut_label, -120)
            self.send_wheel(self.controls.root, -120)
        scroll.assert_not_called()

    def test_slider_container_follows_layout_events(self):
        """Layout events update the scroll region and slider width.

        Args: None.
        Returns: None.
        """
        canvas = self.controls.sliders.master
        self.controls.sliders.event_generate("<Configure>", width=300, height=200)
        self.assertNotEqual(str(canvas.cget("scrollregion")), "")
        canvas.event_generate("<Configure>", width=321, height=100)
        item = canvas.find_all()[0]
        self.assertEqual(int(float(canvas.itemcget(item, "width"))), 321)

    def test_demo_state_toggles_switch_button(self):
        """The live-capture button is packed only in demo mode.

        Args: None.
        Returns: None.
        """
        button = self.controls.switch_live_button
        self.controls.set_demo_state(True)
        self.assertEqual(button.winfo_manager(), "pack")
        self.controls.set_demo_state(False)
        self.assertEqual(button.winfo_manager(), "")

    def test_camera_limits_clamp_sliders_and_values(self):
        """Device limits narrow the sliders and pull stale values inside.

        Args: None.
        Returns: None.
        """
        controls = self.controls
        controls.apply_camera_limits({
            "ExposureTime": (10, 1000),
            "Gain": (0.05, 20.0),
            "AcquisitionFrameRate": (1.0, 120.0),
        })
        self.assertEqual(float(controls.exposure_scale.cget("to")), 1000)
        self.assertEqual(controls.exposure.get(), 1000)
        self.assertAlmostEqual(float(controls.gain_scale.cget("from")), 0.1)
        self.assertAlmostEqual(controls.gain.get(), 0.1)
        self.assertAlmostEqual(float(controls.fps_scale.cget("to")), 120.0)
        self.assertEqual(controls.fps.get(), 30.0)

    def test_camera_limits_keep_value_at_or_above_rounded_lower_bound(self):
        """A 0.1 dB step rounding error still leaves gain at the device minimum.

        The 0.3 dB limit rounds up to 0.30000000000000004, just above the
        0.3 that Tk's own clamping stores, so the value is set again.

        Args: None.
        Returns: None.
        """
        controls = self.controls
        controls.apply_camera_limits({"Gain": (0.3, 20.0)})
        self.assertAlmostEqual(controls.gain.get(), 0.3)
        self.assertAlmostEqual(float(controls.gain_scale.cget("from")), 0.3)

    def test_camera_limits_skip_unsupported_nodes(self):
        """Nodes missing from the limits keep their UI range.

        Args: None.
        Returns: None.
        """
        controls = self.controls
        controls.apply_camera_limits({})
        self.assertEqual(float(controls.exposure_scale.cget("to")), 50000)

    def test_camera_values_update_status_and_rate(self):
        """Readback shows applied settings and remembers the resulting rate.

        Args: None.
        Returns: None.
        """
        self.controls.update_camera_values({
            "exposure": 8000.0, "gain": 0.0, "requested": 30.0,
            "resulting": 29.5})
        self.assertEqual(self.controls.resulting_fps, 29.5)
        self.assertIn("resulting 29.5", self.controls.camera_text.get())

    def test_wide_screen_puts_log_window_beside_controls(self):
        """On a wide screen the log sits left of the control panel.

        Args: None.
        Returns: None.
        """
        controls = self.controls
        with mock.patch.object(controls.root, "winfo_screenwidth",
                               return_value=1920), \
                mock.patch.object(controls.root, "winfo_screenheight",
                                  return_value=1080):
            controls._place_windows()
        control_x = int(re.search(r"\+(\d+)\+", controls.root.geometry())[1])
        log_x = int(re.search(r"\+(\d+)\+", controls.log_window.geometry())[1])
        self.assertLess(log_x, control_x)
        self.assertEqual(controls.viewer_width, max(320, log_x - 20))

    def test_medium_screen_stacks_log_under_controls(self):
        """On a mid-size screen the log sits under the control panel.

        Args: None.
        Returns: None.
        """
        controls = self.controls
        with mock.patch.object(controls.root, "winfo_screenwidth",
                               return_value=1200), \
                mock.patch.object(controls.root, "winfo_screenheight",
                                  return_value=900):
            controls._place_windows()
        control_x = int(re.search(r"\+(\d+)\+", controls.root.geometry())[1])
        log_x = int(re.search(r"\+(\d+)\+", controls.log_window.geometry())[1])
        self.assertEqual(log_x, control_x)
        self.assertEqual(controls.viewer_width, max(320, control_x - 20))

    def test_small_screen_uses_narrow_controls(self):
        """On a small screen the controls use a third of the width.

        Args: None.
        Returns: None.
        """
        controls = self.controls
        with mock.patch.object(controls.root, "winfo_screenwidth",
                               return_value=900), \
                mock.patch.object(controls.root, "winfo_screenheight",
                                  return_value=700):
            controls._place_windows()
        control_x = int(re.search(r"\+(\d+)\+", controls.root.geometry())[1])
        self.assertEqual(control_x, 900 - 320 - 10)
        self.assertEqual(controls.viewer_width, control_x - 20)

    def test_update_services_tk_and_mirrors_output(self):
        """update() drains queued console text into the log window.

        Args: None.
        Returns: None.
        """
        self.controls.output_queue.put("hello log\n")
        self.controls.update()
        self.assertIn("hello log", self.controls.log_text.get("1.0", "end"))

    def test_update_does_nothing_after_stop(self):
        """update() leaves queued text alone once the app is stopping.

        Args: None.
        Returns: None.
        """
        self.controls.stop()
        self.controls.output_queue.put("pending\n")
        self.controls.update()
        self.assertNotIn("pending", self.controls.log_text.get("1.0", "end"))

    def test_write_prints_timestamped_line(self):
        """write() prints a time-stamped message and mirrors queued text.

        Args: None.
        Returns: None.
        """
        out = io.StringIO()
        self.controls.output_queue.put("from queue\n")
        with contextlib.redirect_stdout(out):
            self.controls.write("hello")
        self.assertRegex(out.getvalue(), r"^\d\d:\d\d:\d\d  hello\n$")
        self.assertIn("from queue", self.controls.log_text.get("1.0", "end"))

    def test_drain_with_nothing_queued_is_a_no_op(self):
        """An empty queue leaves the log untouched.

        Args: None.
        Returns: None.
        """
        self.controls.drain_output()
        self.assertEqual(self.controls.log_text.get("1.0", "end"), "\n")
        self.assertEqual(self.controls.log_lines, 0)

    def test_drain_trims_log_beyond_500_lines(self):
        """The log window drops its oldest 100 lines past 500.

        Args: None.
        Returns: None.
        """
        self.controls.output_queue.put("".join(
            f"line {i}\n" for i in range(501)))
        self.controls.drain_output()
        text = self.controls.log_text.get("1.0", "end")
        self.assertNotIn("line 0\n", text)
        self.assertIn("line 100\n", text)
        self.assertEqual(self.controls.log_lines, 401)

    def test_drain_survives_destroyed_log_widget(self):
        """A Tcl error while appending to the log is swallowed.

        Args: None.
        Returns: None.
        """
        self.controls.output_queue.put("text\n")
        with mock.patch.object(self.controls.log_text, "configure",
                               side_effect=tk.TclError("gone")):
            self.controls.drain_output()
        self.assertTrue(self.controls.output_queue.empty())

    def test_close_destroys_windows(self):
        """close() destroys the control and log windows.

        Args: None.
        Returns: None.
        """
        self.controls.close()
        with self.assertRaises(tk.TclError):
            self.controls.root.winfo_exists()
            self.controls.root.update()


class MakeDisplayTests(unittest.TestCase):
    """Brightness scaling, gamma and palette selection."""

    def setUp(self):
        """Create hidden controls and a gamma cache.

        Args: None.
        Returns: None.
        """
        self.controls = build_controls()
        self.addCleanup(destroy_quietly, self.controls)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.cache = [-1, None]
        self.image = mono_frame(160)

    def display(self, update_range=True, clahe=None):
        """Run make_display with the shared fixtures.

        Args: update_range requests a range readout; clahe overrides the enhancer.
        Returns: The display image.
        """
        return app.make_display(self.image, self.controls,
                                clahe or self.clahe, self.cache, update_range)

    def test_scaled_gray_stretches_to_full_range(self):
        """Min/max scaling with the Gray palette returns a 0-255 image.

        Args: None.
        Returns: None.
        """
        result = self.display()
        self.assertEqual(result.ndim, 2)
        self.assertEqual((int(result.min()), int(result.max())), (0, 255))
        self.assertEqual(self.controls.range_text.get(),
                         "Input: 60–155    Scaled: 0–255")

    def test_flat_image_reports_zero_scaled_range(self):
        """A constant image cannot be stretched and reports 0-0.

        Args: None.
        Returns: None.
        """
        self.image = np.full((8, 8), 5, dtype=np.uint8)
        self.display()
        self.assertEqual(self.controls.range_text.get(),
                         "Input: 5–5    Scaled: 0–0")

    def test_unscaled_image_is_returned_unchanged_with_raw_range(self):
        """Without scaling the image passes through and its range is shown.

        Args: None.
        Returns: None.
        """
        self.controls.scale_range.set(False)
        result = self.display()
        np.testing.assert_array_equal(result, self.image)
        self.assertEqual(self.controls.range_text.get(),
                         "Input: 60–155    Scaled: 60–155")

    def test_range_text_untouched_when_no_update_requested(self):
        """With update_range False the status readout keeps its old text.

        Args: None.
        Returns: None.
        """
        self.controls.scale_range.set(False)
        before = self.controls.range_text.get()
        self.display(update_range=False)
        self.assertEqual(self.controls.range_text.get(), before)

    def test_gamma_applies_lookup_table_and_is_cached(self):
        """A non-unity gamma builds a table once and brightens midtones.

        Args: None.
        Returns: None.
        """
        self.controls.gamma.set(2.0)
        result = self.display()
        self.assertEqual(self.cache[0], 2.0)
        table = self.cache[1]
        self.assertEqual(int(table[0]), 0)
        self.assertEqual(int(table[255]), 255)
        self.assertEqual(int(table[64]), int((64 / 255.0) ** 0.5 * 255))
        normalized = app.normalize_min_max(self.image)[0]
        np.testing.assert_array_equal(result, cv2.LUT(normalized, table))
        self.display()
        self.assertIs(self.cache[1], table)

    def test_unity_gamma_clears_lookup_table(self):
        """Returning gamma to 1.0 drops the cached table.

        Args: None.
        Returns: None.
        """
        self.cache[:] = [2.0, np.arange(256, dtype=np.uint8)]
        self.display()
        self.assertEqual(self.cache[0], 1.0)
        self.assertIsNone(self.cache[1])

    def test_color_palette_with_clahe_returns_bgr(self):
        """A color palette enhances with CLAHE and returns a BGR image.

        Args: None.
        Returns: None.
        """
        self.controls.palette.set("Turbo")
        clahe = mock.Mock(wraps=self.clahe)
        result = self.display(clahe=clahe)
        self.assertEqual(result.shape, self.image.shape + (3,))
        clahe.apply.assert_called_once()

    def test_color_palette_without_clahe_skips_enhancement(self):
        """Turning CLAHE off leaves the scaled image alone before coloring.

        Args: None.
        Returns: None.
        """
        self.controls.palette.set("Jet")
        self.controls.clahe.set(False)
        clahe = mock.Mock()
        result = self.display(clahe=clahe)
        clahe.apply.assert_not_called()
        expected = cv2.applyColorMap(app.normalize_min_max(self.image)[0],
                                     app.COLOR_MAPS["Jet"])
        np.testing.assert_array_equal(result, expected)


class FitDimensionsTests(unittest.TestCase):
    """Aspect-preserving viewport fitting."""

    def test_wide_image_is_limited_by_viewport_width(self):
        """A wide image fills the width and keeps its aspect ratio.

        Args: None.
        Returns: None.
        """
        self.assertEqual(app.fit_dimensions(200, 100, 100, 100), (100, 50))

    def test_tall_image_is_limited_by_viewport_height(self):
        """A tall image fills the height and keeps its aspect ratio.

        Args: None.
        Returns: None.
        """
        self.assertEqual(app.fit_dimensions(100, 200, 100, 100), (50, 100))

    def test_tiny_result_is_never_below_one_pixel(self):
        """Extreme aspect ratios still give at least one pixel per side.

        Args: None.
        Returns: None.
        """
        self.assertEqual(app.fit_dimensions(1000, 1, 10, 10), (10, 1))
        self.assertEqual(app.fit_dimensions(1000, 1, 1, 1), (1, 1))


class ViewerTests(unittest.TestCase):
    """The separate image window."""

    def setUp(self):
        """Create hidden controls and a hidden viewer.

        Args: None.
        Returns: None.
        """
        self.controls = build_controls()
        self.addCleanup(destroy_quietly, self.controls)
        self.viewer = app.Viewer(self.controls, "Test viewer")
        self.viewer.window.withdraw()

    def size_label(self, width, height):
        """Pretend the image label has a given on-screen size.

        Args: width and height are pixel sizes.
        Returns: None.
        """
        patcher = mock.patch.multiple(
            self.viewer.label,
            winfo_width=mock.Mock(return_value=width),
            winfo_height=mock.Mock(return_value=height))
        patcher.start()
        self.addCleanup(patcher.stop)

    def key(self, keysym):
        """Send a key event to the viewer handler.

        Args: keysym is the Tk key symbol name.
        Returns: None.
        """
        self.viewer._on_key(SimpleNamespace(keysym=keysym))

    def test_viewer_window_is_titled_and_sized(self):
        """The viewer uses the given title and a 320x240 minimum size.

        Args: None.
        Returns: None.
        """
        self.assertEqual(self.viewer.window.title(), "Test viewer")
        self.assertEqual(self.viewer.window.minsize(), (320, 240))

    def test_q_and_escape_stop_the_app(self):
        """Q and Escape ask the loop to stop.

        Args: None.
        Returns: None.
        """
        self.key("Q")
        self.assertFalse(self.controls.running)
        self.controls.running = True
        self.key("Escape")
        self.assertFalse(self.controls.running)

    def test_capture_keys_queue_actions(self):
        """S, R, A and F queue their capture actions.

        Args: None.
        Returns: None.
        """
        for letter in "sraf":
            self.key(letter.upper())
        queued = [self.controls.actions.get_nowait() for _ in range(4)]
        self.assertEqual(queued, list("sraf"))

    def test_other_keys_are_ignored(self):
        """Unrelated keys neither stop the app nor queue anything.

        Args: None.
        Returns: None.
        """
        self.key("x")
        self.assertTrue(self.controls.running)
        self.assertTrue(self.controls.actions.empty())

    def test_close_key_binding_and_window_close_stop_app(self):
        """The window close button is wired to stop().

        Args: None.
        Returns: None.
        """
        self.assertNotEqual(self.viewer.window.protocol("WM_DELETE_WINDOW"), "")

    def test_show_matching_size_pastes_into_existing_photo(self):
        """A second frame of the same size reuses the Tk image.

        Args: None.
        Returns: None.
        """
        self.size_label(100, 100)
        frame = np.zeros((100, 100), dtype=np.uint8)
        self.viewer.show(frame)
        photo = self.viewer.photo
        self.viewer.show(frame + 7)
        self.assertIs(self.viewer.photo, photo)
        self.assertEqual((photo.width(), photo.height()), (100, 100))

    def test_show_new_size_creates_new_photo(self):
        """A resized viewport allocates a new Tk image.

        Args: None.
        Returns: None.
        """
        self.size_label(100, 100)
        frame = np.zeros((100, 100), dtype=np.uint8)
        self.viewer.show(frame)
        first = self.viewer.photo
        self.size_label(50, 50)
        self.viewer.show(frame)
        self.assertIsNot(self.viewer.photo, first)
        self.assertEqual(self.viewer.photo.width(), 50)

    def test_show_upscales_color_frames_to_rgb(self):
        """A small BGR frame is enlarged to fit and shown as RGB.

        Args: None.
        Returns: None.
        """
        self.size_label(40, 40)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[..., 0] = 255
        with mock.patch.object(app.cv2, "resize",
                               wraps=app.cv2.resize) as resize:
            self.viewer.show(frame)
        self.assertEqual(resize.call_args.args[1], (40, 40))
        self.assertEqual(resize.call_args.kwargs["interpolation"],
                         cv2.INTER_LINEAR)
        self.assertEqual((self.viewer.photo.width(),
                          self.viewer.photo.height()), (40, 40))

    def test_show_downscales_with_area_interpolation(self):
        """A frame larger than the viewport is shrunk with INTER_AREA.

        Args: None.
        Returns: None.
        """
        self.size_label(50, 50)
        frame = np.zeros((200, 200), dtype=np.uint8)
        with mock.patch.object(app.cv2, "resize",
                               wraps=app.cv2.resize) as resize:
            self.viewer.show(frame)
        self.assertEqual(resize.call_args.kwargs["interpolation"],
                         cv2.INTER_AREA)

    def test_fast_preview_caps_pixel_count(self):
        """Fast Raw preview limits the painted size to the pixel budget.

        Args: None.
        Returns: None.
        """
        self.size_label(2000, 1500)
        frame = np.zeros((1500, 2000), dtype=np.uint8)
        self.viewer.show(frame, fast=True)
        photo = self.viewer.photo
        self.assertLessEqual(photo.width() * photo.height(),
                             app.FAST_PREVIEW_MAX_PIXELS)
        self.assertLess(photo.width(), 2000)

    def test_normal_preview_keeps_full_viewport_size(self):
        """Without fast mode a large viewport is painted at full size.

        Args: None.
        Returns: None.
        """
        self.size_label(2000, 1500)
        frame = np.zeros((1500, 2000), dtype=np.uint8)
        self.viewer.show(frame, fast=False)
        self.assertEqual(self.viewer.photo.width(), 2000)

    def test_close_destroys_window(self):
        """close() destroys the viewer window.

        Args: None.
        Returns: None.
        """
        self.viewer.close()
        self.assertEqual(self.viewer.window.winfo_exists(), 0)


class AnnounceCameraTests(unittest.TestCase):
    """Logging the detected camera in the Tk front end."""

    def test_logs_model_and_notes_and_names_viewer(self):
        """The model, its notes and a model-specific viewer title are produced.

        Args: None.
        Returns: None.
        """
        controls = mock.Mock()
        title = app.announce_camera(
            controls, CameraIdentity(model="Blackfly S BFS-U3-31S4C"))
        self.assertEqual(title, "BFS-U3-31S4C ESPI")
        lines = [call.args[0] for call in controls.write.call_args_list]
        self.assertTrue(lines[0].startswith("Camera: Blackfly S BFS-U3-31S4C"))
        self.assertTrue(lines[1].startswith("Camera note: Colour camera"))


class WaitForInputTests(unittest.TestCase):
    """Frame pacing between Tk service passes."""

    def test_sleeps_only_the_remaining_interval(self):
        """The wait services Tk once and sleeps what is left.

        Args: None.
        Returns: None.
        """
        controls = mock.Mock()
        clock = FakeClock()
        clock.now = 10.0
        sleeper = SimpleNamespace(monotonic=clock.monotonic, sleep=mock.Mock())
        with mock.patch.object(app, "time", sleeper):
            key = app.wait_for_input(controls, 10.25)
        self.assertEqual(key, 0xFF)
        controls.update.assert_called_once_with()
        sleeper.sleep.assert_called_once_with(0.25)

    def test_no_sleep_after_the_deadline(self):
        """A missed deadline returns immediately.

        Args: None.
        Returns: None.
        """
        controls = mock.Mock()
        clock = FakeClock()
        clock.now = 10.0
        sleeper = SimpleNamespace(monotonic=clock.monotonic, sleep=mock.Mock())
        with mock.patch.object(app, "time", sleeper):
            key = app.wait_for_input(controls, 9.0)
        self.assertEqual(key, 0xFF)
        sleeper.sleep.assert_not_called()


class FakeClock:
    """A manually advanced monotonic clock."""

    def __init__(self):
        """Start at an arbitrary positive time.

        Args: None.
        Returns: None.
        """
        self.now = 1000.0

    def monotonic(self):
        """Read the fake time.

        Args: None.
        Returns: The current fake monotonic time in seconds.
        """
        return self.now

    def sleep(self, seconds):
        """Advance the fake time instead of sleeping.

        Args: seconds is the requested sleep.
        Returns: None.
        """
        self.now += seconds


class FakeDone:
    """Stand-in for a recorder's ``done`` threading.Event."""

    def __init__(self, clock):
        """Create an unfinished event.

        Args: clock is the FakeClock advanced by wait().
        Returns: None.
        """
        self.clock = clock
        self.finished = False

    def is_set(self):
        """Report whether the recorder has finished.

        Args: None.
        Returns: True once finished.
        """
        return self.finished

    def wait(self, timeout):
        """Advance the fake clock instead of blocking.

        Args: timeout is the requested wait in seconds.
        Returns: The finished flag.
        """
        self.clock.now += timeout
        return self.finished


class FakeRecorder:
    """Stand-in for AVIRecorder that never touches the disk."""

    def __init__(self, factory, path, frame, fps):
        """Remember the arguments.

        Args: factory is the RecorderFactory; path, frame and fps are the
            AVIRecorder arguments.
        Returns: None.
        """
        self.factory = factory
        self.path = path
        self.frame = frame
        self.fps = fps
        self.submitted = []
        self.stopped = False
        self.error = None
        self.done = FakeDone(factory.clock)

    def submit(self, frame):
        """Record a submitted preview frame.

        Args: frame is the display image.
        Returns: None.
        """
        self.submitted.append(frame)

    def stop(self):
        """Mark the recorder stopped, finishing it when the factory allows.

        Args: None.
        Returns: None.
        """
        self.stopped = True
        if self.factory.finish_on_stop:
            self.done.finished = True

    def summary(self):
        """Describe the recording.

        Args: None.
        Returns: A fixed summary text.
        """
        return f"Saved AVI: fake {os.path.basename(self.path)}"


class RecorderFactory:
    """Callable replacing the AVIRecorder class."""

    def __init__(self, clock):
        """Create a factory that finishes recorders when they stop.

        Args: clock is the FakeClock shared with the recorders.
        Returns: None.
        """
        self.clock = clock
        self.finish_on_stop = True
        self.created = []

    def __call__(self, path, frame, fps):
        """Build a fake recorder.

        Args: path, frame and fps match AVIRecorder.
        Returns: The new FakeRecorder.
        """
        recorder = FakeRecorder(self, path, frame, fps)
        self.created.append(recorder)
        return recorder


class FakeCamera:
    """Stand-in for CameraSession that never touches PySpin."""

    def __init__(self, factory, controls, open_error):
        """Remember the factory and the error open() should raise.

        Args: factory is the CameraFactory; controls is the log sink;
            open_error is an exception to raise from open(), or None.
        Returns: None.
        """
        self.factory = factory
        self.controls = controls
        self.open_error = open_error
        self.identity = CameraIdentity(model="Blackfly S BFS-U3-63S4M")
        self.closed = False
        self.grab_calls = 0
        self.read_calls = 0
        self.set_calls = []

    def open(self):
        """Fail or succeed as scripted.

        Args: None.
        Returns: None.
        """
        if self.open_error is not None:
            raise self.open_error

    def limits(self):
        """Report device limits.

        Args: None.
        Returns: A limits dict for exposure and gain.
        """
        return {"ExposureTime": (10, 20000), "Gain": (0.0, 24.0)}

    def read_values(self):
        """Report applied settings.

        Args: None.
        Returns: A read_values() style dict.
        """
        self.read_calls += 1
        return {"exposure": 8000.0, "gain": 0.0, "requested": 30.0,
                "resulting": 25.0}

    def grab(self, timeout_ms):
        """Return the next scripted result or a good Mono8 frame.

        Args: timeout_ms is the requested timeout.
        Returns: A (frame, state) tuple.
        """
        self.grab_calls += 1
        assert timeout_ms == 100
        if self.factory.grab_script:
            return self.factory.grab_script.pop(0)()
        return mono_frame(), "ok"

    def set_float(self, name, value):
        """Record a node write.

        Args: name is the node; value is the number written.
        Returns: None.
        """
        self.set_calls.append((name, value))

    def close(self):
        """Mark the camera closed.

        Args: None.
        Returns: None.
        """
        self.closed = True


class CameraFactory:
    """Callable replacing the CameraSession class."""

    def __init__(self):
        """Create a factory whose cameras open successfully.

        Args: None.
        Returns: None.
        """
        self.cameras = []
        self.open_errors = []
        self.grab_script = []

    def __call__(self, controls):
        """Build a fake camera, consuming one scripted open error if any.

        Args: controls is the log sink.
        Returns: The new FakeCamera.
        """
        error = self.open_errors.pop(0) if self.open_errors else None
        camera = FakeCamera(self, controls, error)
        self.cameras.append(camera)
        return camera


class FakeAppLog:
    """Stand-in for AppLog that writes no file and mirrors nothing."""

    def __init__(self, path):
        """Remember the path start() reports.

        Args: path is the log path, or None for no file.
        Returns: None.
        """
        self.path = path
        self.output_queue = queue.SimpleQueue()
        self.closed = False

    def start(self):
        """Pretend to open a log file.

        Args: None.
        Returns: The configured path.
        """
        return self.path

    def close(self):
        """Record that the log was closed.

        Args: None.
        Returns: None.
        """
        self.closed = True


class FakeViewer(app.Viewer):
    """Real viewer window that is hidden and records what it is shown."""

    instances = []
    close_error = None

    def __init__(self, controls, title):
        """Create the hidden window and register this viewer.

        Args: controls and title are the Viewer arguments.
        Returns: None.
        """
        super().__init__(controls, title)
        self.window.withdraw()
        self.shown = []
        FakeViewer.instances.append(self)

    def show(self, frame, fast=False):
        """Record the frame instead of painting it.

        Args: frame is the display image; fast is the fast-preview flag.
        Returns: None.
        """
        self.shown.append((frame, fast))

    def close(self):
        """Close the window, or raise the configured Tcl error.

        Args: None.
        Returns: None.
        """
        if FakeViewer.close_error is not None:
            raise FakeViewer.close_error
        super().close()


def set_var(controls, name, value):
    """Set one of the controls' Tk variables.

    Args: controls is the Controls; name is the attribute; value the new value.
    Returns: None.
    """
    getattr(controls, name).set(value)


def press(controls, action):
    """Queue a capture shortcut as if the user pressed it.

    Args: controls is the Controls; action is the shortcut letter.
    Returns: None.
    """
    controls.queue_action(action)


def return_key(_controls, key):
    """Return a key code from the scripted wait.

    Args: _controls is unused; key is the key code to report.
    Returns: key.
    """
    return key


def type_title_and_notes(controls, title, notes):
    """Fill in the frame title and notes fields.

    Args: controls is the Controls; title and notes are the texts to enter.
    Returns: None.
    """
    controls.frame_title.set(title)
    controls.frame_notes.insert("1.0", notes)


def snapshot(controls, store, name, getter):
    """Store a value read from the controls.

    Args: controls is the Controls; store is a dict; name is the key to fill;
        getter is a callable taking the controls.
    Returns: None.
    """
    store[name] = getter(controls)


def do_all(controls, *steps):
    """Run several step callables in order.

    Args: controls is the Controls; steps are callables taking the controls.
    Returns: None.
    """
    for step in steps:
        step(controls)


def step(func, *args):
    """Wrap a helper and its arguments into a script step.

    Args: func is called as func(controls, *args); args are its extra arguments.
    Returns: A callable taking the controls and returning func's result.
    """
    def run(controls):
        """Run the wrapped helper.

        Args: controls is the Controls instance.
        Returns: The helper's return value.
        """
        return func(controls, *args)
    return run


def idle(_controls):
    """A script step that does nothing.

    Args: _controls is unused.
    Returns: None.
    """


def note_var(controls, store, key, name):
    """Store the current value of one of the controls' Tk variables.

    Args: controls is the Controls; store is a dict; key is the entry to fill;
        name is the variable attribute.
    Returns: None.
    """
    store[key] = getattr(controls, name).get()


def root_title(controls):
    """Read the control window title.

    Args: controls is the Controls.
    Returns: The title string.
    """
    return controls.root.title()


def log_title(controls):
    """Read the log window title.

    Args: controls is the Controls.
    Returns: The title string.
    """
    return controls.log_window.title()


def switch_button_manager(controls):
    """Tell whether the live-capture button is packed.

    Args: controls is the Controls.
    Returns: "pack" when visible, else an empty string.
    """
    return controls.switch_live_button.winfo_manager()


def exposure_limit(controls):
    """Read the exposure slider's upper bound.

    Args: controls is the Controls.
    Returns: The bound as a float.
    """
    return float(controls.exposure_scale.cget("to"))


def float_frame():
    """Build a non-Mono8 grab result.

    Args: None.
    Returns: A (float32 frame, "ok") tuple.
    """
    return np.zeros((4, 4), dtype=np.float32), "ok"


def color_frame():
    """Build a three-channel grab result.

    Args: None.
    Returns: A (BGR uint8 frame, "ok") tuple.
    """
    return np.zeros((4, 4, 3), dtype=np.uint8), "ok"


class MainLoopTests(unittest.TestCase):
    """The acquisition loop in main(), driven by scripted steps."""

    def setUp(self):
        """Install fakes for the clock, camera, recorder, dialogs and log.

        Args: None.
        Returns: None.
        """
        self.clock = FakeClock()
        self.cameras = CameraFactory()
        self.recorders = RecorderFactory(self.clock)
        self.app_log = FakeAppLog("fake.log")
        self.timer = mock.Mock()
        self.timer.start.return_value = False
        self.dialogs = mock.Mock()
        self.dialogs.askyesno.return_value = True
        self.save = mock.Mock()
        self.recording_path = mock.Mock(side_effect=self.fake_recording_path)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.steps = []
        self.step_index = 0
        self.dt = 0.6
        self.controls = None
        FakeViewer.instances = []
        FakeViewer.close_error = None
        self.addCleanup(setattr, FakeViewer, "close_error", None)

    def fake_recording_path(self, mode, demo):
        """Pick a path inside the temporary folder.

        Args: mode is the view name; demo marks synthetic video.
        Returns: A path string (nothing is created).
        """
        return os.path.join(self.tmp.name, f"{mode}_{demo}.avi")

    def make_controls(self):
        """Create hidden controls and remember them.

        Args: None.
        Returns: The new Controls.
        """
        self.controls = build_controls()
        self.addCleanup(destroy_quietly, self.controls)
        return self.controls

    def fake_wait(self, controls, deadline):
        """Advance the clock, run the next scripted step, then end the script.

        Args: controls is the Controls; deadline is the ignored frame deadline.
        Returns: 0xFF, or the key code a step returned.
        """
        self.clock.now += self.dt
        index = self.step_index
        self.step_index += 1
        if index >= len(self.steps):
            controls.stop()
            return 0xFF
        result = self.steps[index](controls)
        return 0xFF if result is None else result

    def run_main(self, steps=(), force_demo=True, dt=0.6, exits=False):
        """Run main() against the fakes.

        Args: steps are callables run after each frame; force_demo is passed to
            main(); dt is the clock advance per frame; exits expects a
            SystemExit(1) from a fatal error.
        Returns: None; the console text is left in self.output.
        """
        self.steps = list(steps)
        self.dt = dt
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(out))
            stack.enter_context(contextlib.redirect_stderr(out))
            for name, value in (
                    ("wait_for_input", self.fake_wait),
                    ("time", SimpleNamespace(monotonic=self.clock.monotonic,
                                             sleep=self.clock.sleep)),
                    ("Controls", self.make_controls),
                    ("AppLog", mock.Mock(return_value=self.app_log)),
                    ("WindowsPreviewTimer",
                     mock.Mock(return_value=self.timer)),
                    ("CameraSession", self.cameras),
                    ("messagebox", self.dialogs),
                    ("Viewer", FakeViewer),
                    ("recording_path", self.recording_path),
                    ("AVIRecorder", self.recorders),
                    ("save_frame", self.save)):
                stack.enter_context(mock.patch.object(app, name, value))
            if exits:
                with self.assertRaises(SystemExit) as caught:
                    app.main(force_demo)
                self.assertEqual(caught.exception.code, 1)
            else:
                app.main(force_demo)
        self.output = out.getvalue()

    def test_forced_demo_marks_every_window_as_dummy(self):
        """Forced demo mode never asks, and labels windows and status.

        Args: None.
        Returns: None.
        """
        seen = {}
        self.run_main([
            step(snapshot, seen, "root", root_title),
            step(snapshot, seen, "log", log_title),
            step(note_var, seen, "camera", "camera_text"),
            step(snapshot, seen, "button", switch_button_manager),
        ])
        self.assertEqual(seen["root"], "ESPI controls — DUMMY MODE")
        self.assertEqual(seen["log"], "ESPI log — DUMMY MODE")
        self.assertIn("DUMMY MODE ACTIVE", seen["camera"])
        self.assertEqual(seen["button"], "pack")
        self.assertEqual(FakeViewer.instances[0].shown[0][0].shape, (640, 960))
        self.dialogs.askyesno.assert_not_called()
        self.assertEqual(self.cameras.cameras, [])
        self.assertIn("DUMMY MODE ACTIVE: synthetic video", self.output)
        self.assertIn("Streaming.", self.output)
        self.assertTrue(self.app_log.closed)
        self.timer.stop.assert_called_once_with()

    def test_demo_frames_carry_watermark(self):
        """Frames shown in demo mode are watermarked as dummy data.

        Args: None.
        Returns: None.
        """
        self.run_main()
        shown = FakeViewer.instances[0].shown[0][0]
        plain = app.DemoSource().base
        self.assertFalse(np.array_equal(shown[:40], plain[:40]))

    def test_log_file_path_is_reported(self):
        """The log file location is written when a file was opened.

        Args: None.
        Returns: None.
        """
        self.run_main()
        self.assertIn("Log file: fake.log", self.output)

    def test_missing_log_file_is_not_reported(self):
        """No log message appears when the log file could not be opened.

        Args: None.
        Returns: None.
        """
        self.app_log.path = None
        self.run_main()
        self.assertNotIn("Log file:", self.output)

    def test_windows_timer_start_is_reported(self):
        """A granted 1 ms timer request is announced.

        Args: None.
        Returns: None.
        """
        self.timer.start.return_value = True
        self.run_main()
        self.assertIn("Windows preview timer", self.output)

    def test_raw_view_uses_fast_preview_setting(self):
        """Raw frames follow the fast-preview checkbox.

        Args: None.
        Returns: None.
        """
        self.run_main([step(set_var, "fast_preview", False)])
        shown = FakeViewer.instances[0].shown
        self.assertEqual([fast for _, fast in shown[:2]], [True, False])

    def test_processed_views_never_use_fast_preview(self):
        """Previous view paints at full quality even with fast preview on.

        Args: None.
        Returns: None.
        """
        self.run_main([step(set_var, "mode", "Previous"), idle])
        shown = FakeViewer.instances[0].shown
        self.assertTrue(shown[0][1])
        self.assertFalse(shown[1][1])

    def test_raw_status_text_and_speed_readout(self):
        """Raw view reports no scaling and the displayed frame rate.

        Args: None.
        Returns: None.
        """
        seen = {}
        self.run_main([
            idle, idle,
            step(note_var, seen, "range", "range_text"),
            step(note_var, seen, "speed", "speed_text"),
        ])
        self.assertEqual(seen["range"], "Raw Mono8: no brightness scaling")
        self.assertRegex(seen["speed"],
                         r"Displayed: [\d.]+ fps    App work: [\d.]+ ms")

    def test_view_change_resets_state_and_updates_hint(self):
        """Switching views logs the change and explains the Reference view.

        Args: None.
        Returns: None.
        """
        seen = {}
        self.run_main([
            step(note_var, seen, "raw", "hint_text"),
            step(set_var, "mode", "Previous"),
            step(note_var, seen, "previous", "hint_text"),
            step(set_var, "mode", "Reference"),
            step(note_var, seen, "reference", "hint_text"),
        ])
        self.assertIn("Raw view", seen["raw"])
        self.assertIn("ESPI difference view", seen["previous"])
        self.assertIn("press f", seen["reference"])
        self.assertIn("View: Previous", self.output)
        self.assertIn("View: Reference", self.output)

    def test_reference_hint_clears_after_capture(self):
        """Capturing a reference changes the hint to the difference text.

        Args: None.
        Returns: None.
        """
        seen = {}
        self.run_main([
            step(set_var, "mode", "Reference"),
            step(press, "f"),
            idle,
            step(note_var, seen, "hint", "hint_text"),
        ])
        self.assertIn("Captured reference frame.", self.output)
        self.assertIn("ESPI difference view", seen["hint"])

    def test_averaging_withholds_display_until_enough_frames(self):
        """With Average N set, nothing is shown until N frames accumulate.

        Args: None.
        Returns: None.
        """
        self.run_main([
            step(set_var, "mode", "Previous"),
            step(set_var, "avg", 3),
            idle, idle, idle, idle,
        ])
        shown = FakeViewer.instances[0].shown
        # frame 0 (Raw), then Previous frames: two withheld, the rest shown
        self.assertGreaterEqual(len(shown), 2)
        self.assertLess(len(shown), 6)

    def test_save_key_saves_current_display_with_title_and_notes(self):
        """S saves the last shown frame with the typed title and notes.

        Args: None.
        Returns: None.
        """
        self.run_main([
            step(type_title_and_notes, "Fringe A", "some notes"),
            step(press, "s"),
        ])
        self.save.assert_called_once()
        display, title, notes, demo, log = self.save.call_args.args
        self.assertEqual(display.shape, (640, 960))
        self.assertEqual((title, notes, demo), ("Fringe A", "some notes", True))
        self.assertIs(log, self.controls)

    def test_save_key_without_display_reports_nothing_ready(self):
        """S before any frame is displayed logs a message and saves nothing.

        Args: None.
        Returns: None.
        """
        self.run_main([
            step(do_all, step(set_var, "mode", "Previous"),
                 step(set_var, "avg", 5)),
            step(press, "s"),
        ])
        self.save.assert_not_called()
        self.assertIn("No current display frame is ready to save.", self.output)

    def test_reset_keys_log_their_action(self):
        """R and A reset previous-frame and averaging state.

        Args: None.
        Returns: None.
        """
        self.run_main([step(press, "r"), step(press, "a")])
        self.assertIn("Reset previous frame.", self.output)
        self.assertIn("Reset averaging.", self.output)

    def test_quit_action_stops_the_loop(self):
        """A queued q stops the loop before the script ends.

        Args: None.
        Returns: None.
        """
        self.run_main([step(press, "q"), step(press, "s")])
        self.save.assert_not_called()
        self.assertFalse(self.controls.running)

    def test_escape_key_code_stops_the_loop(self):
        """Key code 27 (Escape) stops the loop.

        Args: None.
        Returns: None.
        """
        self.run_main([step(return_key, 27), step(press, "s")])
        self.save.assert_not_called()

    def test_live_key_is_ignored_when_not_in_demo(self):
        """The switch-to-live shortcut does nothing with a camera attached.

        Args: None.
        Returns: None.
        """
        self.run_main([step(press, "l")], force_demo=False)
        self.assertEqual(len(self.cameras.cameras), 1)
        self.assertNotIn("Searching for a FLIR camera", self.output)

    def test_recording_lifecycle(self):
        """Ticking Record starts a recorder; unticking finishes the file.

        Args: None.
        Returns: None.
        """
        texts = {}
        self.run_main([
            step(set_var, "record", True),
            step(note_var, texts, 1, "record_text"),
            step(set_var, "record", False),
            step(note_var, texts, 2, "record_text"),
            step(note_var, texts, 3, "record_text"),
        ])
        recorder = self.recorders.created[0]
        self.assertEqual(texts, {1: "Recording: 30.0 fps",
                                 2: "Recording: finishing file",
                                 3: "Recording: off"})
        self.assertEqual(recorder.fps, 30.0)
        self.assertEqual(len(recorder.submitted), 1)
        self.assertTrue(recorder.stopped)
        self.assertIn("Recording AVI:", self.output)
        self.assertIn("Saved AVI: fake", self.output)

    def test_recording_rate_is_divided_by_average_count(self):
        """The AVI rate is the source rate over the averaging count.

        Args: None.
        Returns: None.
        """
        self.run_main([
            step(do_all, step(set_var, "mode", "Previous"),
                 step(set_var, "avg", 2), step(set_var, "record", True)),
            idle, idle, idle,
        ])
        self.assertEqual(self.recorders.created[0].fps, 15.0)

    def test_recorder_error_unticks_record_and_is_reported(self):
        """A failed recorder is stopped, Record unticks, the error is logged.

        Args: None.
        Returns: None.
        """
        def fail(_controls):
            """Make the live recorder report a finished failure.

            Args: _controls is unused.
            Returns: None.
            """
            recorder = self.recorders.created[0]
            recorder.error = "disk full"
            recorder.done.finished = True

        self.run_main([step(set_var, "record", True), fail,
                       idle, idle])
        self.assertFalse(self.controls.record.get())
        self.assertIn("Recording error: disk full", self.output)
        self.assertEqual(len(self.recorders.created), 1)

    def test_view_change_restarts_recording_in_new_file(self):
        """Changing the view finishes the file and starts a new one.

        Args: None.
        Returns: None.
        """
        self.run_main([
            step(set_var, "record", True),
            step(set_var, "mode", "Previous"),
            idle,
        ])
        first, second = self.recorders.created
        self.assertTrue(first.stopped)
        self.assertEqual(second.path,
                         self.fake_recording_path("Previous", True))
        self.assertEqual(self.recording_path.call_args_list[0].args,
                         ("Raw", True))

    def test_recording_path_failure_unticks_record(self):
        """If no file can be created, Record unticks and the reason is logged.

        Args: None.
        Returns: None.
        """
        self.recording_path.side_effect = OSError("no space left")
        self.run_main([step(set_var, "record", True), idle])
        self.assertFalse(self.controls.record.get())
        self.assertIn("no space left", self.output)
        self.assertEqual(self.recorders.created, [])

    def test_shutdown_waits_then_reports_stuck_recorder(self):
        """A recorder that never finishes is abandoned after five seconds.

        Args: None.
        Returns: None.
        """
        self.recorders.finish_on_stop = False
        self.run_main([step(set_var, "record", True), idle])
        recorder = self.recorders.created[0]
        self.assertTrue(recorder.stopped)
        self.assertIn("Recording did not finish within 5 seconds", self.output)
        self.assertGreaterEqual(self.clock.now - 1000.0, 5.0)

    def test_shutdown_reports_summary_and_error_of_finished_recorder(self):
        """A recorder finishing at shutdown has its summary and error logged.

        Args: None.
        Returns: None.
        """
        def poison(_controls):
            """Give the live recorder an error to report at shutdown.

            Args: _controls is unused.
            Returns: None.
            """
            self.recorders.created[0].error = "late failure"

        self.run_main([step(set_var, "record", True), poison])
        self.assertIn("Saved AVI: fake", self.output)
        self.assertIn("Recording error: late failure", self.output)

    def test_unticked_recorder_that_never_finishes_stays_retired(self):
        """A retired recorder not yet done is kept until shutdown.

        Args: None.
        Returns: None.
        """
        self.recorders.finish_on_stop = False
        self.run_main([
            step(set_var, "record", True),
            step(set_var, "record", False),
            idle, idle,
        ])
        self.assertNotIn("Saved AVI", self.output)
        self.assertIn("Recording did not finish within 5 seconds", self.output)

    def test_camera_frames_apply_setting_changes_and_read_back(self):
        """Slider changes are written to the camera and values are read back.

        Args: None.
        Returns: None.
        """
        seen = {}
        self.run_main([
            step(note_var, seen, "camera", "camera_text"),
            step(do_all, step(set_var, "exposure", 9000),
                 step(set_var, "gain", 1.5), step(set_var, "fps", 20.0)),
            idle,
        ], force_demo=False, dt=1.2)
        camera = self.cameras.cameras[0]
        self.assertEqual(camera.set_calls, [("ExposureTime", 9000),
                                            ("Gain", 1.5),
                                            ("AcquisitionFrameRate", 20.0)])
        self.assertTrue(seen["camera"].startswith("Applied:"))
        self.assertGreaterEqual(camera.read_calls, 3)
        self.assertTrue(camera.closed)
        self.assertEqual(self.controls.resulting_fps, 25.0)

    def test_camera_limits_are_applied_at_startup(self):
        """Device limits narrow the sliders as soon as the camera opens.

        Args: None.
        Returns: None.
        """
        seen = {}
        self.run_main([step(snapshot, seen, "to", exposure_limit)], force_demo=False)
        self.assertEqual(seen["to"], 20000.0)

    def test_recording_uses_resulting_camera_rate(self):
        """With a camera, the AVI rate follows the measured acquisition rate.

        Args: None.
        Returns: None.
        """
        self.run_main([step(set_var, "record", True), idle],
                      force_demo=False)
        self.assertEqual(self.recorders.created[0].fps, 25.0)
        self.assertEqual(self.recording_path.call_args.args, ("Raw", False))

    def test_incomplete_and_timeout_frames_are_reported_sparingly(self):
        """Bad grabs are logged at a limited rate and flagged in the status.

        Args: None.
        Returns: None.
        """
        seen = {}

        def incomplete():
            """Return an incomplete grab two seconds later.

            Args: None.
            Returns: A (None, "incomplete") grab result.
            """
            self.clock.now += 2.0
            return None, "incomplete"

        def timeout():
            """Return a timeout grab.

            Args: None.
            Returns: A ("boom", "timeout") grab result.
            """
            return "boom", "timeout"

        def timeout_and_look():
            """Record the status text, then time out again.

            Args: None.
            Returns: A ("boom", "timeout") grab result.
            """
            seen["speed"] = self.controls.speed_text.get()
            return "boom", "timeout"

        self.cameras.grab_script = [incomplete, incomplete, timeout,
                                    timeout_and_look, timeout_and_look]
        self.run_main(force_demo=False)
        self.assertEqual(self.output.count("Incomplete frames:"), 1)
        self.assertEqual(
            self.output.count("Camera timeout or acquisition error: boom"), 1)
        self.assertEqual(seen["speed"],
                         "Displayed: 0 fps    Waiting for camera frame")

    def test_non_mono8_frame_is_fatal(self):
        """A float frame stops the app with a clear message and exit code 1.

        Args: None.
        Returns: None.
        """
        self.cameras.grab_script = [float_frame]
        self.run_main(force_demo=False, exits=True)
        self.assertIn("Stopped: RuntimeError: Expected a Mono8 image",
                      self.output)
        self.assertTrue(self.cameras.cameras[0].closed)
        self.assertTrue(self.app_log.closed)

    def test_color_frame_is_fatal(self):
        """A three-channel frame is rejected like any non-Mono8 frame.

        Args: None.
        Returns: None.
        """
        self.cameras.grab_script = [color_frame]
        self.run_main(force_demo=False, exits=True)
        self.assertIn("Expected a Mono8 image", self.output)

    def test_missing_camera_can_fall_back_to_demo(self):
        """When no camera opens, accepting the prompt starts demo mode.

        Args: None.
        Returns: None.
        """
        self.cameras.open_errors = [RuntimeError("no camera")]
        seen = {}
        self.run_main([step(snapshot, seen, "title", root_title)], force_demo=False)
        self.assertIn("Camera startup unavailable: no camera", self.output)
        self.dialogs.askyesno.assert_called_once()
        self.assertIs(self.dialogs.askyesno.call_args.kwargs["parent"],
                      self.controls.root)
        self.assertIn("DUMMY MODE", seen["title"])
        self.assertEqual(self.cameras.cameras[0].grab_calls, 0)

    def test_missing_camera_and_declined_demo_exits_cleanly(self):
        """Declining demo mode logs a message and shuts down without a viewer.

        Args: None.
        Returns: None.
        """
        self.cameras.open_errors = [RuntimeError("no camera")]
        self.dialogs.askyesno.return_value = False
        self.run_main(force_demo=False)
        self.assertIn("No camera detected; demo mode declined.", self.output)
        self.assertEqual(FakeViewer.instances, [])
        self.assertTrue(self.app_log.closed)
        self.timer.stop.assert_called_once_with()

    def test_viewer_cleanup_error_is_logged_and_shutdown_continues(self):
        """A Tcl error closing the viewer is logged, not raised.

        Args: None.
        Returns: None.
        """
        FakeViewer.close_error = tk.TclError("window gone")
        self.run_main()
        self.assertIn("Window cleanup error: window gone", self.output)
        self.assertTrue(self.app_log.closed)

    def test_switch_to_live_replaces_demo_state(self):
        """Switching to live capture resets the UI and starts using the camera.

        Args: None.
        Returns: None.
        """
        seen = {}

        def look(controls):
            """Record the UI state after the switch.

            Args: controls is the Controls.
            Returns: None.
            """
            seen.update(
                root=controls.root.title(), log=controls.log_window.title(),
                viewer=FakeViewer.instances[0].window.title(),
                mode=controls.mode.get(), record=controls.record_text.get(),
                button=controls.switch_live_button.winfo_manager(),
                limit=float(controls.exposure_scale.cget("to")),
                range=controls.range_text.get())

        self.run_main([step(set_var, "mode", "Previous"),
                       step(press, "l"), look])
        camera = self.cameras.cameras[0]
        self.assertEqual(seen, {
            "root": "ESPI controls", "log": "ESPI log",
            "viewer": "BFS-U3-63S4M ESPI", "mode": "Raw",
            "record": "Recording: off", "button": "", "limit": 20000.0,
            "range": "Raw Mono8: no brightness scaling"})
        self.assertGreaterEqual(camera.grab_calls, 1)
        self.assertTrue(camera.closed)
        self.assertIn("Searching for a FLIR camera", self.output)
        self.assertIn("LIVE CAPTURE ACTIVE", self.output)
        self.assertIn("Recording stopped for the source change", self.output)

    def test_switch_to_live_stops_active_demo_recording(self):
        """A demo recording is finished when the source changes to live.

        Args: None.
        Returns: None.
        """
        self.run_main([step(set_var, "record", True), step(press, "l"),
                       idle])
        recorder = self.recorders.created[0]
        self.assertTrue(recorder.stopped)
        self.assertFalse(self.controls.record.get())
        self.assertEqual(len(self.recorders.created), 1)
        self.assertIn("Saved AVI: fake", self.output)

    def test_failed_switch_keeps_demo_running(self):
        """If no camera opens, demo mode continues and the user is told.

        Args: None.
        Returns: None.
        """
        self.cameras.open_errors = [RuntimeError("camera busy")]
        self.run_main([step(press, "l"), idle])
        self.assertIn("Still in dummy mode: camera busy", self.output)
        self.dialogs.showwarning.assert_called_once()
        self.assertIn("camera busy", self.dialogs.showwarning.call_args.args[1])
        self.assertEqual(self.cameras.cameras[0].grab_calls, 0)
        self.assertGreaterEqual(len(FakeViewer.instances[0].shown), 3)


class EntryPointTests(unittest.TestCase):
    """The thin command-line wrappers."""

    def test_cli_delegates_to_package_entry_point(self):
        """app.cli forwards its arguments to espi_live.cli.cli.

        Args: None.
        Returns: None.
        """
        with mock.patch("espi_live.cli.cli") as run_cli:
            app.cli(["--demo"])
        run_cli.assert_called_once_with(["--demo"])

    def test_running_module_as_script_calls_cli(self):
        """Executing app.py as __main__ runs the command-line entry point.

        Args: None.
        Returns: None.
        """
        with mock.patch("espi_live.cli.cli") as run_cli:
            runpy.run_path(app.__file__, run_name="__main__")
        run_cli.assert_called_once_with(None)


if __name__ == "__main__":
    unittest.main()
