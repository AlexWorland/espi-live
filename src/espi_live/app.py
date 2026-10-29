"""Live Blackfly Mono8 preview and ESPI measurement viewer.

Legacy Tk front end (``--ui tk``); the default is the Qt front end in
qt_app.py.
The camera image stays in a separate Tk viewer. A resizable Tk control panel has a
font-size slider, and a separate log window shows application messages. Focus
the image window for shortcuts: q/Esc quit, s save, r reset previous frame,
a reset averaging, and f capture a reference frame.
"""

import math
import queue
import time
import traceback
import tkinter as tk
from datetime import datetime
from functools import partial
from tkinter import font as tkfont
from tkinter import messagebox
from tkinter import scrolledtext

import cv2
import numpy as np
from PIL import Image, ImageTk

from espi_live.camera import CameraSession, format_camera_values
from espi_live.engine import WindowsPreviewTimer
from espi_live.logs import AppLog
from espi_live.processing import (COLOR_MAPS, DemoSource, ESPIProcessor,
                                  normalize_min_max, watermark_demo)
from espi_live.recording import AVIRecorder, recording_path, save_frame

FAST_PREVIEW_MAX_PIXELS = 1_500_000


class Controls:
    """Own Tk settings, command buttons, live status, and the log window."""

    def __init__(self):
        """Build the control panel and independent log window.

        Args: None.
        Returns: None.
        """
        self.root = tk.Tk()
        self.root.title("ESPI controls")
        self.root.geometry("420x760")
        self.root.minsize(420, 540)
        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        self.font = tkfont.Font(family="Arial", size=13)

        self.log_window = tk.Toplevel(self.root)
        self.log_window.title("ESPI log")
        self.log_window.geometry("620x300")
        self.log_window.minsize(280, 180)
        self.log_window.protocol("WM_DELETE_WINDOW", self.log_window.withdraw)
        self.log_text = scrolledtext.ScrolledText(
            self.log_window, state="disabled", wrap="word", font=self.font
        )
        self.log_text.pack(fill="both", expand=True)
        self.output_queue = queue.SimpleQueue()
        self.actions = queue.SimpleQueue()

        self.mode = tk.StringVar(value="Raw")
        self.scale_range = tk.BooleanVar(value=True)
        self.exposure = tk.IntVar(value=8000)
        self.gain = tk.DoubleVar(value=0.0)
        self.fps = tk.DoubleVar(value=30.0)
        self.avg = tk.IntVar(value=1)
        self.palette = tk.StringVar(value="Gray")
        self.gamma = tk.DoubleVar(value=1.0)
        self.wait_ms = tk.IntVar(value=10)
        self.clahe = tk.BooleanVar(value=True)
        self.fast_preview = tk.BooleanVar(value=True)
        self.record = tk.BooleanVar(value=False)
        self.frame_title = tk.StringVar(value="")
        self.font_size = tk.IntVar(value=13)
        self.range_text = tk.StringVar(value="Input: —    Scaled: —")
        self.speed_text = tk.StringVar(
            value="Displayed: — fps    App work: — ms/input frame")
        self.camera_text = tk.StringVar(value="Camera values: waiting for connection")
        self.hint_text = tk.StringVar(value="Raw view: direct Mono8 positioning")
        self.record_text = tk.StringVar(value="Recording: off")
        self.resulting_fps = None
        self.log_lines = 0
        self.labels = []

        top = tk.LabelFrame(self.root, text="View", font=self.font)
        top.pack(fill="x", padx=10, pady=5)
        for name in ("Raw", "Previous", "Reference"):
            tk.Radiobutton(top, text=name, variable=self.mode, value=name,
                           font=self.font).pack(side="left")
        self._label(self.root, textvariable=self.hint_text)
        self._label(self.root, textvariable=self.range_text)
        self._label(self.root, textvariable=self.speed_text)
        self._label(self.root, textvariable=self.camera_text)

        def set_font_size(value):
            """Apply a text-size slider event.

            Args: value is the Tk slider's numeric string.
            Returns: None.
            """
            self.font.configure(size=int(float(value)))

        tk.Scale(
            self.root, label="Text size", variable=self.font_size,
            from_=10, to=24, orient="horizontal", length=200,
            font=self.font, command=set_font_size
        ).pack(fill="x", padx=10)

        resize_hint = tk.Label(
            self.root,
            text="More settings below ↓  Scroll this section or drag a window "
                 "edge to make the controls taller.",
            bg="#e4f2ff", anchor="w", justify="left", wraplength=390,
            font=self.font
        )
        resize_hint.pack(fill="x", padx=10, pady=(2, 5))
        self.labels.append(resize_hint)

        body = tk.Frame(self.root)
        self.settings_body = body
        canvas = tk.Canvas(body, highlightthickness=0)
        scrollbar = tk.Scrollbar(
            body, orient="vertical", command=canvas.yview, width=18)
        self.sliders = tk.Frame(canvas)

        def update_scroll_region(_event):
            """Refresh slider scroll bounds after a layout event.

            Args: _event is the unused Tk configure event.
            Returns: None.
            """
            canvas.configure(scrollregion=canvas.bbox("all"))

        self.sliders.bind("<Configure>", update_scroll_region)
        slider_window = canvas.create_window((0, 0), window=self.sliders, anchor="nw")

        def resize_slider_region(event):
            """Match the slider container width to the canvas.

            Args: event is the Tk configure event containing the new width.
            Returns: None.
            """
            canvas.itemconfigure(slider_window, width=event.width)

        canvas.bind("<Configure>", resize_slider_region)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def scroll_controls(event):
            """Scroll the controls only when the pointer is in their list.

            Args: event supplies the Windows mouse-wheel delta and widget.
            Returns: None.
            """
            widget = event.widget
            while widget is not None and widget is not self.root:
                if widget is self.sliders or widget is canvas:
                    canvas.yview_scroll(-int(event.delta / 120), "units")
                    return
                widget = widget.master

        self.root.bind_all("<MouseWheel>", scroll_controls)

        camera_group = tk.LabelFrame(self.sliders, text="Camera", font=self.font)
        camera_group.pack(fill="x", padx=8, pady=5)
        self.exposure_scale = self._scale(
            camera_group, "Exposure (µs)", self.exposure, 0, 50000)
        self.gain_scale = self._scale(
            camera_group, "Gain (dB)", self.gain, 0.0, 30.0, resolution=0.1)
        self.fps_scale = self._scale(
            camera_group, "Requested FPS", self.fps, 0.0, 60.0, resolution=0.1)

        measurement_group = tk.LabelFrame(
            self.sliders, text="ESPI measurement", font=self.font)
        measurement_group.pack(fill="x", padx=8, pady=5)
        self._scale(measurement_group, "Average N (1 = off)", self.avg, 1, 60)
        tk.Checkbutton(
            measurement_group, text="Scale min/max to 0–255",
            variable=self.scale_range, font=self.font
        ).pack(anchor="w", padx=8)

        display_group = tk.LabelFrame(self.sliders, text="Display", font=self.font)
        display_group.pack(fill="x", padx=8, pady=5)
        palette_row = tk.Frame(display_group)
        palette_row.pack(fill="x", padx=8, pady=3)
        tk.Label(palette_row, text="Palette", font=self.font).pack(side="left")
        palette_menu = tk.OptionMenu(
            palette_row, self.palette, "Gray", *COLOR_MAPS.keys())
        palette_menu.configure(font=self.font)
        palette_menu["menu"].configure(font=self.font)
        palette_menu.pack(side="right", fill="x", expand=True)
        self._scale(display_group, "Gamma", self.gamma, 0.1, 3.0, resolution=0.1)
        tk.Checkbutton(
            display_group, text="CLAHE (color palettes only)",
            variable=self.clahe, font=self.font
        ).pack(anchor="w", padx=8)
        tk.Checkbutton(
            display_group, text="Fast Raw preview (full-size captures)",
            variable=self.fast_preview, font=self.font
        ).pack(anchor="w", padx=8)
        self._scale(display_group, "Display wait (ms; Raw uses 1)",
                    self.wait_ms, 1, 200)

        capture = tk.LabelFrame(self.root, text="Capture", font=self.font)
        tk.Checkbutton(capture, text="Record AVI", variable=self.record,
                       font=self.font).pack(side="left")
        tk.Label(capture, textvariable=self.record_text,
                 font=self.font).pack(side="left", padx=8)
        buttons = tk.Frame(self.root)
        tk.Label(buttons, text="Frame title (optional)", font=self.font,
                 anchor="w").grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Entry(buttons, textvariable=self.frame_title, font=self.font).grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=2)
        tk.Label(buttons, text="Frame notes (optional)", font=self.font,
                 anchor="w").grid(row=2, column=0, columnspan=2, sticky="ew")
        self.frame_notes = tk.Text(buttons, height=2, width=24, wrap="word",
                                   font=self.font)
        self.frame_notes.grid(row=3, column=0, columnspan=2,
                              sticky="ew", padx=2, pady=(0, 4))
        for index, (label, action) in enumerate((
                ("Save frame (S)", "s"), ("Set reference (F)", "f"),
                ("Reset last (R)", "r"), ("Reset average (A)", "a"))):
            tk.Button(buttons, text=label, font=self.font,
                      command=partial(self.queue_action, action)).grid(
                          row=4 + index // 2, column=index % 2,
                          sticky="ew", padx=2, pady=2)
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        self.shortcut_label = tk.Label(
            buttons,
            text="Viewer shortcuts: S save · F set reference · R reset last · "
                 "A reset average · Q/Esc quit",
            font=self.font, justify="left", wraplength=390
        )
        self.shortcut_label.grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        footer = tk.Frame(self.root)
        tk.Button(footer, text="Show log", command=self.log_window.deiconify,
                  font=self.font).pack(side="left")
        self.quit_button = tk.Button(
            footer, text="Quit (Q/Esc)", command=self.stop, font=self.font)
        self.quit_button.pack(side="right")
        self.switch_live_button = tk.Button(
            footer, text="Switch to live capture",
            command=partial(self.queue_action, "l"), font=self.font)
        footer.pack(side="bottom", fill="x", padx=10, pady=5)
        buttons.pack(side="bottom", fill="x", padx=10, pady=5)
        capture.pack(side="bottom", fill="x", padx=10, pady=5)
        body.pack(side="top", fill="both", expand=True)
        self.root.bind("<Configure>", self._resize_labels)
        self.root.update_idletasks()
        self._place_windows()

    def _label(self, parent, text=None, textvariable=None):
        """Create a wrapping status label in a parent widget.

        Args: parent owns the label; text is fixed wording; textvariable is
            optional live Tk text.
        Returns: The created Tk label.
        """
        label = tk.Label(
            parent, text=text, textvariable=textvariable, anchor="w",
            justify="left", wraplength=390, font=self.font
        )
        label.pack(fill="x", padx=10)
        self.labels.append(label)
        return label

    def _resize_labels(self, event):
        """Keep status text wrapped when the control window is resized.

        Args: event is the Tk resize event.
        Returns: None.
        """
        if event.widget is self.root:
            for label in self.labels:
                label.configure(wraplength=max(180, event.width - 35))
            self.shortcut_label.configure(wraplength=max(180, event.width - 35))

    def _scale(self, parent, label, variable, low, high, resolution=1):
        """Build a slider and an exact-value entry that share a setting.

        Args: parent owns the row; label names the setting; variable stores its
            value; low/high bound it; resolution is its step size.
        Returns: The created Tk Scale for later device-limit updates.
        """
        row = tk.Frame(parent)
        row.pack(fill="x", padx=8, pady=2)
        tk.Label(row, text=label, font=self.font, anchor="w").pack(fill="x")
        value_row = tk.Frame(row)
        value_row.pack(fill="x")
        scale = tk.Scale(
            value_row, variable=variable, from_=low, to=high,
            orient="horizontal", resolution=resolution, length=170,
            showvalue=False, font=self.font
        )
        scale.pack(side="left", fill="x", expand=True)
        digits = 0 if resolution == 1 else 1
        entry_text = tk.StringVar()

        def reflect_value(*_):
            """Copy a slider value into the entry field.

            Args: *_ contains unused Tk variable-trace arguments.
            Returns: None.
            """
            entry_text.set(f"{variable.get():.{digits}f}")

        def commit_value(_=None):
            """Validate and apply a typed value to the slider.

            Args: _ is an optional, unused Tk key or focus event.
            Returns: None; invalid text is replaced by the current value.
            """
            try:
                value = float(entry_text.get())
                if not math.isfinite(value):
                    raise ValueError("Value must be finite")
                value = max(float(scale.cget("from")),
                            min(value, float(scale.cget("to"))))
                scale.set(value)
            except ValueError:
                reflect_value()

        variable.trace_add("write", reflect_value)
        reflect_value()
        entry = tk.Entry(value_row, textvariable=entry_text, width=8,
                         font=self.font, justify="right")
        entry.pack(side="right")
        entry.bind("<Return>", commit_value)
        entry.bind("<FocusOut>", commit_value)
        return scale

    def queue_action(self, action):
        """Queue a capture-button shortcut for the frame loop.

        Args: action is one of the documented one-letter shortcut keys.
        Returns: None.
        """
        self.actions.put(action)

    def set_demo_state(self, active):
        """Show the live-capture handoff only for synthetic video.

        Args: active is True while dummy mode is running.
        Returns: None.
        """
        if active:
            self.switch_live_button.pack(side="right", padx=6)
        else:
            self.switch_live_button.pack_forget()

    def apply_camera_limits(self, limits):
        """Clamp camera sliders to readable device limits.

        Args: limits is CameraSession.limits(): {node name: (low, high)}.
        Returns: None. Unsupported nodes retain their current UI range.
        """
        for name, scale, variable, step in (
            ("ExposureTime", self.exposure_scale, self.exposure, 1),
            ("Gain", self.gain_scale, self.gain, 0.1),
            ("AcquisitionFrameRate", self.fps_scale, self.fps, 0.1),
        ):
            if name not in limits:
                continue
            lower, upper = limits[name]
            low = math.ceil(lower / step) * step
            high = max(low, math.floor(upper / step) * step)
            scale.configure(from_=low, to=high)
            if not low <= variable.get() <= high:
                scale.set(max(low, min(variable.get(), high)))

    def update_camera_values(self, values):
        """Show applied camera settings and resulting acquisition rate.

        Args: values is CameraSession.read_values().
        Returns: None; unavailable readings appear as n/a.
        """
        self.resulting_fps = values["resulting"]
        self.camera_text.set(format_camera_values(values))

    def _place_windows(self):
        """Fit all controls when possible and arrange the three windows.

        Args: None.
        Returns: None.
        """
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        control_w = (min(500, max(420, screen_w // 6)) if screen_w >= 1000
                     else max(320, screen_w // 3))
        fixed_h = self.root.winfo_reqheight() - self.settings_body.winfo_reqheight()
        desired_h = fixed_h + self.sliders.winfo_reqheight() + 20
        control_h = min(max(540, screen_h - 80), max(540, desired_h))
        control_x = max(0, screen_w - control_w - 10)
        self.root.geometry(f"{control_w}x{control_h}+{control_x}+0")
        self.root.update_idletasks()
        if screen_w >= 1500:
            log_w = min(540, max(360, screen_w // 4))
            log_x = max(0, control_x - log_w - 10)
            log_h = min(650, max(300, screen_h - 120))
            self.log_window.geometry(f"{log_w}x{log_h}+{log_x}+0")
            self.viewer_width = max(320, log_x - 20)
        else:
            log_h = max(180, screen_h - control_h - 70)
            self.log_window.geometry(
                f"{control_w}x{log_h}+{control_x}+{control_h}")
            self.viewer_width = max(320, control_x - 20)

    def update(self):
        """Service pending Tk events and mirror queued console output.

        Args: None.
        Returns: None.
        """
        if self.running:
            self.root.update()
            self.drain_output()

    def stop(self):
        """Request that the acquisition loop stop.

        Args: None.
        Returns: None.
        """
        self.running = False

    def write(self, message):
        """Write a timestamped application message to both log destinations.

        Args: message is the application message text.
        Returns: None.
        """
        line = f"{datetime.now():%H:%M:%S}  {message}"
        print(line, flush=True)
        self.drain_output()

    def drain_output(self):
        """Append queued stdout and stderr text to the log window.

        Args: None.
        Returns: None.
        """
        chunks = []
        while True:
            try:
                chunks.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return
        output = "".join(chunks)
        try:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", output)
            self.log_lines += output.count("\n")
            if self.log_lines > 500:
                self.log_text.delete("1.0", "101.0")
                self.log_lines -= 100
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        except tk.TclError:
            pass

    def close(self):
        """Destroy the control and log windows at shutdown.

        Args: None.
        Returns: None.
        """
        self.root.destroy()


def make_display(measurement, controls, clahe, gamma_cache, update_range):
    """Apply ESPI brightness scaling and display color controls.

    Args: measurement is an averaged uint8 difference image; controls supplies
        live settings; clahe is the OpenCV enhancer; gamma_cache stores the
        last lookup table; update_range requests a new status readout.
    Returns: A grayscale or BGR NumPy image for preview and capture.
    """
    if controls.scale_range.get():
        scaled, low, high = normalize_min_max(measurement)
        scaled_low, scaled_high = (0, 255) if high > low else (0, 0)
    else:
        scaled = measurement
        if update_range:
            low, high, _, _ = cv2.minMaxLoc(measurement)
            low, high = int(low), int(high)
            scaled_low, scaled_high = low, high
    if update_range:
        controls.range_text.set(
            f"Input: {low}–{high}    Scaled: {scaled_low}–{scaled_high}")

    gamma = controls.gamma.get()
    if gamma != gamma_cache[0]:
        gamma_cache[0] = gamma
        gamma_cache[1] = None if gamma == 1.0 else np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
            dtype=np.uint8
        )
    if gamma_cache[1] is not None:
        scaled = cv2.LUT(scaled, gamma_cache[1])

    palette = controls.palette.get()
    if palette == "Gray":
        return scaled
    if controls.clahe.get():
        scaled = clahe.apply(scaled)
    return cv2.applyColorMap(scaled, COLOR_MAPS[palette])


def fit_dimensions(source_width, source_height, width, height):
    """Calculate an aspect-preserving size within a viewport.

    Args: source_width/source_height are the image dimensions; width/height
        are the available viewport dimensions, all positive integers.
    Returns: The largest integer (fitted_width, fitted_height) inside the
        viewport without altering the source's aspect ratio beyond rounding.
    """
    scale = min(width / source_width, height / source_height)
    return (max(1, min(width, round(source_width * scale))),
            max(1, min(height, round(source_height * scale))))


class Viewer:
    """Display aspect-correct video in Tk's shared GUI event loop."""

    def __init__(self, controls, title):
        """Create a separate viewer and bind its keyboard shortcuts.

        Args: controls owns the Tk root and action queue; title labels the viewer.
        Returns: None.
        """
        self.controls = controls
        self.window = tk.Toplevel(controls.root)
        self.window.title(title)
        self.window.minsize(320, 240)
        self.window.protocol("WM_DELETE_WINDOW", controls.stop)
        self.label = tk.Label(self.window, bg="black", bd=0)
        self.label.pack(fill="both", expand=True)
        self.photo = None
        self.window.bind("<Key>", self._on_key)
        screen_h = controls.root.winfo_screenheight()
        available_w = controls.viewer_width
        self.window.geometry(f"{available_w}x{max(240, screen_h - 120)}+0+0")

    def _on_key(self, event):
        """Route viewer shortcuts to the existing capture actions.

        Args: event is a Tk key event from the focused viewer.
        Returns: None.
        """
        key = event.keysym.lower()
        if key in ("q", "escape"):
            self.controls.stop()
        elif key in ("s", "r", "a", "f"):
            self.controls.queue_action(key)

    def show(self, frame, fast=False):
        """Fit and draw one grayscale or BGR frame without distortion.

        Args: frame is the current display image as a uint8 NumPy array; fast
            caps preview pixels to reduce Tk paint time in Raw view.
        Returns: None; the Tk image is retained for the next paint event.
        """
        width = max(1, self.label.winfo_width())
        height = max(1, self.label.winfo_height())
        source_h, source_w = frame.shape[:2]
        fitted_w, fitted_h = fit_dimensions(source_w, source_h, width, height)
        if fast and fitted_w * fitted_h > FAST_PREVIEW_MAX_PIXELS:
            scale = math.sqrt(FAST_PREVIEW_MAX_PIXELS / (fitted_w * fitted_h))
            fitted_w = max(1, int(fitted_w * scale))
            fitted_h = max(1, int(fitted_h * scale))
        if fitted_w != source_w or fitted_h != source_h:
            shrinking = fitted_w < source_w or fitted_h < source_h
            interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (fitted_w, fitted_h),
                               interpolation=interpolation)
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        picture = Image.fromarray(frame)
        if (self.photo is None or self.photo.width() != fitted_w or
                self.photo.height() != fitted_h):
            self.photo = ImageTk.PhotoImage(picture, master=self.window)
            self.label.configure(image=self.photo)
        else:
            self.photo.paste(picture)

    def close(self):
        """Destroy the viewer during application shutdown.

        Args: None.
        Returns: None.
        """
        self.window.destroy()


def announce_camera(controls, identity):
    """Log the detected camera model and any notes about it.

    Args: controls has a write(message) method; identity is the opened
        session's CameraIdentity.
    Returns: The viewer window title naming the model.
    """
    controls.write("Camera: " + " | ".join(identity.description()))
    for warning in identity.warnings():
        controls.write(f"Camera note: {warning}")
    return f"{identity.short_name()} ESPI"


def wait_for_input(controls, deadline):
    """Service Tk once, then sleep only the remainder of the frame interval.

    Args: controls owns Tk events; deadline is a monotonic target time.
    Returns: The no-key sentinel 255; viewer shortcuts enter the action queue.
    """
    controls.update()
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)
    return 0xFF


def main(force_demo=False):
    """Run camera or demo frames through the viewer and clean up safely.

    Args: force_demo skips camera detection and starts synthetic video directly.
    Returns: None. Fatal errors are logged and exit with a nonzero status.
    """
    controls = Controls()
    app_log = AppLog()
    app_log.output_queue = controls.output_queue
    log_path = app_log.start()
    camera = None
    preview_timer = WindowsPreviewTimer()
    recorder = None
    retired_recorders = []
    record_signature = None
    viewer = None
    image_window = "Blackfly ESPI"
    demo = force_demo
    demo_source = None
    try:
        if log_path:
            controls.write(f"Log file: {log_path}")
        if not demo:
            attempt = CameraSession(controls)
            try:
                attempt.open()
            except Exception as exc:
                controls.write(f"Camera startup unavailable: {exc}")
                demo = True
            else:
                camera = attempt
                image_window = announce_camera(controls, camera.identity)
                controls.apply_camera_limits(camera.limits())
                controls.update_camera_values(camera.read_values())
        if demo and not force_demo:
            controls.root.update_idletasks()
            if not messagebox.askyesno(
                    "No camera detected",
                    "No FLIR camera is available. Start dummy/demo mode with a "
                    "synthetic example pattern?",
                    parent=controls.root):
                controls.write("No camera detected; demo mode declined.")
                return
        if demo:
            demo_source = DemoSource()
            controls.root.title("ESPI controls — DUMMY MODE")
            controls.log_window.title("ESPI log — DUMMY MODE")
            controls.set_demo_state(True)
            controls.camera_text.set(
                "DUMMY MODE ACTIVE — exposure, gain, and FPS are simulated")
            controls.write("DUMMY MODE ACTIVE: synthetic video; no camera acquired.")
            image_window = "Blackfly ESPI - DUMMY MODE"

        viewer = Viewer(controls, image_window)
        if preview_timer.start():
            controls.write("Windows preview timer: 1 ms resolution while app is open.")

        processor = ESPIProcessor()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gamma_cache = [-1, None]
        last_settings = (8000, 0.0, 30.0)
        last_setting_time = 0.0
        last_mode = controls.mode.get()
        last_display = None
        last_status_time = 0.0
        last_camera_read_time = time.monotonic()
        speed_start_time = time.monotonic()
        displayed_frames = 0
        processed_frames = 0
        work_time_ms = 0.0
        incomplete_count = 0
        last_camera_error_time = 0.0
        last_valid_frame_time = time.monotonic()
        controls.write("Streaming. Focus the image for q/Esc, s, r, a, f shortcuts.")
        controls.write("Raw shows direct Mono8 pixels; Previous and Reference show ESPI differences.")

        while controls.running:
            for old in retired_recorders[:]:
                if old.done.is_set():
                    controls.write(old.summary())
                    if old.error is not None:
                        controls.write(f"Recording error: {old.error}")
                    retired_recorders.remove(old)
                    if recorder is None and not controls.record.get():
                        controls.record_text.set("Recording: off")
            mode = controls.mode.get()
            signature = (mode, controls.avg.get() if mode != "Raw" else 1)
            if recorder is not None and (
                    not controls.record.get() or signature != record_signature
                    or recorder.done.is_set()):
                if recorder.error is not None:
                    controls.record.set(False)
                recorder.stop()
                retired_recorders.append(recorder)
                recorder = None
                record_signature = None
                controls.record_text.set("Recording: finishing file")
            if mode != last_mode:
                processor.reset_avg()
                if mode == "Previous":
                    processor.prev = None
                last_display = None
                controls.write(f"View: {mode}")
                last_mode = mode
            hint = (
                "Raw view: direct Mono8 positioning; fast preview affects display only"
                if mode == "Raw" else
                "Reference: press f or Set reference to capture a baseline"
                if mode == "Reference" and processor.ref is None else
                "ESPI difference view; min/max range is before gamma and palette")
            if controls.hint_text.get() != hint:
                controls.hint_text.set(hint)

            work_started = time.monotonic()
            if demo:
                frame = demo_source.next_frame(
                    controls.exposure.get(), controls.gain.get(), work_started)
            else:
                frame, state = camera.grab(100)
                if state != "ok":
                    if state == "incomplete":
                        incomplete_count += 1
                        if incomplete_count % 30 == 1:
                            controls.write(f"Incomplete frames: {incomplete_count}")
                    elif time.monotonic() - last_camera_error_time >= 5.0:
                        controls.write(f"Camera timeout or acquisition error: {frame}")
                        last_camera_error_time = time.monotonic()
                    if (time.monotonic() - last_valid_frame_time >= 1.0 and
                            controls.speed_text.get() !=
                            "Displayed: 0 fps    Waiting for camera frame"):
                        controls.speed_text.set(
                            "Displayed: 0 fps    Waiting for camera frame")
                    controls.update()
                    continue
            if frame.ndim != 2 or frame.dtype != np.uint8:
                raise RuntimeError("Expected a Mono8 image; check camera PixelFormat.")

            now = time.monotonic()
            last_valid_frame_time = now
            if not demo and now - last_camera_read_time >= 1.0:
                controls.update_camera_values(camera.read_values())
                last_camera_read_time = now
            settings = (controls.exposure.get(), controls.gain.get(),
                        controls.fps.get())
            if not demo and settings != last_settings and now - last_setting_time >= 0.1:
                if settings[0] != last_settings[0]:
                    camera.set_float("ExposureTime", settings[0])
                if settings[1] != last_settings[1]:
                    camera.set_float("Gain", settings[1])
                if settings[2] != last_settings[2]:
                    camera.set_float("AcquisitionFrameRate", settings[2])
                last_settings = settings
                last_setting_time = now

            update_range = now - last_status_time >= 0.5
            if mode == "Raw":
                display = frame
                if update_range:
                    controls.range_text.set("Raw Mono8: no brightness scaling")
                    last_status_time = now
            else:
                difference = processor.difference(frame, mode)
                averaged = processor.average(difference, controls.avg.get())
                display = (make_display(averaged, controls, clahe,
                                        gamma_cache, update_range)
                           if averaged is not None else None)
                if display is not None and update_range:
                    last_status_time = now

            if display is not None:
                if demo:
                    display = watermark_demo(display)
                last_display = display
                if controls.record.get() and recorder is None:
                    source_fps = controls.resulting_fps or controls.fps.get()
                    recording_fps = min(30.0, max(1.0, source_fps / signature[1]))
                    try:
                        path = recording_path(mode, demo)
                        recorder = AVIRecorder(path, display, recording_fps)
                    except OSError as exc:
                        controls.record.set(False)
                        controls.write(str(exc))
                    else:
                        record_signature = signature
                        controls.record_text.set(
                            f"Recording: {recording_fps:.1f} fps")
                        controls.write(f"Recording AVI: {path} at {recording_fps:.1f} fps")
                elif recorder is not None:
                    recorder.submit(display)
                viewer.show(display, fast=mode == "Raw" and
                            controls.fast_preview.get())
                displayed_frames += 1
            processed_frames += 1
            work_time_ms += (time.monotonic() - work_started) * 1000.0

            elapsed = time.monotonic() - speed_start_time
            if elapsed >= 1.0:
                controls.speed_text.set(
                    f"Displayed: {displayed_frames / elapsed:.1f} fps    "
                    f"App work: {work_time_ms / processed_frames:.1f} ms/input frame")
                speed_start_time = time.monotonic()
                displayed_frames = 0
                processed_frames = 0
                work_time_ms = 0.0

            wait_ms = 1 if mode == "Raw" else max(1, controls.wait_ms.get())
            deadline = time.monotonic() + wait_ms / 1000.0
            if demo:
                deadline = max(deadline, work_started +
                               1.0 / max(1.0, controls.fps.get()))
            key = wait_for_input(controls, deadline)
            if key == 0xFF:
                try:
                    key = ord(controls.actions.get_nowait())
                except queue.Empty:
                    pass
            if key in (ord("q"), 27):
                controls.stop()
            elif key == ord("s"):
                if last_display is None:
                    controls.write("No current display frame is ready to save.")
                else:
                    save_frame(last_display, controls.frame_title.get(),
                               controls.frame_notes.get("1.0", "end-1c"),
                               demo, controls)
            elif key == ord("r"):
                processor.prev = None
                controls.write("Reset previous frame.")
            elif key == ord("a"):
                processor.reset_avg()
                controls.write("Reset averaging.")
            elif key == ord("f"):
                processor.ref = frame.copy()
                processor.reset_avg()
                controls.write("Captured reference frame.")
            elif key == ord("l") and demo:
                controls.write("Searching for a FLIR camera to switch to live capture...")
                attempt = CameraSession(controls)
                try:
                    attempt.open()
                except Exception as exc:
                    controls.write(f"Still in dummy mode: {exc}")
                    messagebox.showwarning(
                        "Camera unavailable",
                        f"Could not start live capture: {exc}\n\n"
                        "Dummy mode will continue.", parent=controls.root)
                else:
                    if recorder is not None:
                        recorder.stop()
                        retired_recorders.append(recorder)
                        recorder = None
                        record_signature = None
                    controls.record.set(False)
                    controls.record_text.set("Recording: off")
                    camera = attempt
                    controls.apply_camera_limits(camera.limits())
                    controls.update_camera_values(camera.read_values())
                    demo = False
                    demo_source = None
                    processor = ESPIProcessor()
                    controls.mode.set("Raw")
                    last_mode = "Raw"
                    last_display = None
                    last_settings = (controls.exposure.get(), controls.gain.get(),
                                     controls.fps.get())
                    last_setting_time = time.monotonic()
                    last_camera_read_time = time.monotonic()
                    controls.range_text.set("Raw Mono8: no brightness scaling")
                    controls.root.title("ESPI controls")
                    controls.log_window.title("ESPI log")
                    viewer.window.title(announce_camera(controls, camera.identity))
                    controls.set_demo_state(False)
                    controls.write("LIVE CAPTURE ACTIVE: camera connected; demo ended.")
                    controls.write("Recording stopped for the source change; restart it if needed.")
    except Exception as exc:
        controls.write(f"Stopped: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        controls.drain_output()
        raise SystemExit(1) from None
    finally:
        try:
            if recorder is not None:
                recorder.stop()
                retired_recorders.append(recorder)
            deadline = time.monotonic() + 5.0
            for old in retired_recorders:
                while not old.done.is_set() and time.monotonic() < deadline:
                    old.done.wait(0.05)
                    controls.update()
                if old.done.is_set():
                    controls.write(old.summary())
                    if old.error is not None:
                        controls.write(f"Recording error: {old.error}")
                else:
                    controls.write(f"Recording did not finish within 5 seconds: {old.path}")
            if viewer is not None:
                try:
                    viewer.close()
                except tk.TclError as exc:
                    controls.write(f"Window cleanup error: {exc}")
            if camera is not None:
                camera.close()
            preview_timer.stop()
        finally:
            controls.drain_output()
            app_log.close()
            controls.close()


def cli(argv=None):
    """Run the command-line entry point (see espi_live.cli).

    Args: argv is an optional list of argument strings.
    Returns: None.
    """
    from espi_live.cli import cli as run_cli
    run_cli(argv)


if __name__ == "__main__":
    cli()
