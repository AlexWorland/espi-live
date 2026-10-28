"""
ESPI Drumhead Imaging GUI
FLIR Blackfly + PySpin + OpenCV + Tkinter

Features:
- Live camera view.
- Time-averaged ESPI-style processing with real-time subtraction:
    * RAW (live speckle)
    * DIFF (|current - reference|)
    * DIFF+EQ (|current - reference| + histogram equalization)
- GUI controls:
    * Buttons:
        - Capture Reference
        - Start Recording / Stop Recording (video of processed view)
        - Save Current Frame (PNG)
        - Quit
    * Sliders:
        - Exposure (ms)
        - Gain (dB)
        - Contrast scale for diff image
    * Dropdown:
        - Display Mode: Raw / Diff / Diff+EQ
    * Checkbox:
        - Show “Raw vs Processed” side-by-side window
- Large preview embedded in Tkinter (no OpenCV GUI windows).

Requirements:
    pip install opencv-python numpy pillow
    (PySpin / Spinnaker SDK installed from Teledyne)
"""

import time
import cv2
import numpy as np
import PySpin
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk


# ------------------- Camera helpers ------------------- #

def get_camera():
    """
    Connect to the first available PySpin camera and return:
    - system handle
    - camera object
    - nodemap
    """
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        raise RuntimeError("No PySpin-compatible cameras detected.")

    cam = cam_list[0]
    cam.Init()
    nodemap = cam.GetNodeMap()
    cam_list.Clear()
    return system, cam, nodemap


def set_enum(nodemap, name, value):
    """
    Set an enumeration node (e.g. PixelFormat, ExposureAuto).
    Safely no-ops if node or entry is not accessible.
    """
    node = PySpin.CEnumerationPtr(nodemap.GetNode(name))
    if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
        return
    entry = node.GetEntryByName(value)
    if PySpin.IsReadable(entry):
        node.SetIntValue(entry.GetValue())


def set_float(nodemap, name, val):
    """
    Set a float node (e.g. ExposureTime, Gain, AcquisitionFrameRate),
    clipping the value into the allowed range.
    """
    node = PySpin.CFloatPtr(nodemap.GetNode(name))
    if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
        return
    val = float(val)
    val = max(node.GetMin(), min(node.GetMax(), val))
    node.SetValue(val)


def get_mono8_array(image):
    """
    Convert a PySpin ImagePtr to a 2D uint8 NumPy array (Mono8).
    """
    img_conv = image.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
    w = img_conv.GetWidth()
    h = img_conv.GetHeight()
    data = img_conv.GetData()
    arr = np.array(data, dtype=np.uint8).reshape((h, w))
    return arr


# ------------------- ESPI Application ------------------- #

class ESPIApp:
    """
    Main ESPI GUI application.
    Handles:
    - camera acquisition
    - real-time ESPI processing
    - Tkinter UI
    - frame saving and video recording
    """

    DISPLAY_MODES = ["Raw", "Diff", "Diff+EQ"]

    def __init__(self, root):
        self.root = root
        self.root.title("ESPI Drumhead Imaging")

        # Connect camera
        self.system, self.cam, self.nodemap = get_camera()
        self.configure_camera()

        # Start acquisition
        self.cam.BeginAcquisition()

        # State
        self.reference = None          # reference frame (drum at rest)
        self.last_raw = None           # last raw frame (Mono8)
        self.last_processed = None     # last processed frame (Mono8)
        self.running = True

        # Recording state
        self.is_recording = False
        self.video_writer = None
        self.video_path = None
        self.video_fps = 30.0

        # Tkinter image holders (avoid garbage collection)
        self.tk_main_image = None
        self.tk_side_image = None

        # Side-by-side window
        self.side_window = None

        # Build GUI
        self.build_gui()

        # Kick off update loop
        self.update_loop()

    # ------------- camera configuration ------------ #

    def configure_camera(self):
        """
        Basic ESPI-friendly configuration for the camera.
        """
        set_enum(self.nodemap, "PixelFormat", "Mono8")
        set_enum(self.nodemap, "ExposureAuto", "Off")
        set_enum(self.nodemap, "GainAuto", "Off")
        set_enum(self.nodemap, "AcquisitionMode", "Continuous")
        set_float(self.nodemap, "ExposureTime", 3000)  # 3 ms
        set_float(self.nodemap, "Gain", 6)

    # ---------------- GUI construction -------------- #

    def build_gui(self):
        """
        Build the main Tkinter interface:
        - Buttons
        - Sliders
        - Display mode dropdown
        - Side-by-side toggle
        - Main preview label
        """
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Control row
        controls = ttk.Frame(main_frame)
        controls.pack(side="top", fill="x")

        # Buttons
        ttk.Button(controls, text="Capture Reference",
                   command=self.capture_reference).grid(row=0, column=0, padx=4, pady=4)
        ttk.Button(controls, text="Save Current Frame",
                   command=self.save_current_frame).grid(row=0, column=1, padx=4, pady=4)
        self.record_btn = ttk.Button(controls, text="Start Recording",
                                     command=self.toggle_recording)
        self.record_btn.grid(row=0, column=2, padx=4, pady=4)
        ttk.Button(controls, text="Quit", command=self.quit).grid(row=0, column=3, padx=4, pady=4)

        # Exposure slider
        ttk.Label(controls, text="Exposure (ms)").grid(row=1, column=0, sticky="w")
        self.exp_var = tk.DoubleVar(value=3.0)
        exp_scale = ttk.Scale(controls, from_=1, to=50,
                              variable=self.exp_var,
                              command=self.on_exposure_change)
        exp_scale.grid(row=1, column=1, columnspan=3, sticky="ew", padx=4)

        # Gain slider
        ttk.Label(controls, text="Gain (dB)").grid(row=2, column=0, sticky="w")
        self.gain_var = tk.DoubleVar(value=6.0)
        gain_scale = ttk.Scale(controls, from_=0, to=24,
                               variable=self.gain_var,
                               command=self.on_gain_change)
        gain_scale.grid(row=2, column=1, columnspan=3, sticky="ew", padx=4)

        # Contrast scale slider for diff image
        ttk.Label(controls, text="Diff Contrast Scale").grid(row=3, column=0, sticky="w")
        self.contrast_var = tk.DoubleVar(value=10.0)  # factor / 10
        contrast_scale = ttk.Scale(controls, from_=1, to=30,
                                   variable=self.contrast_var)
        contrast_scale.grid(row=3, column=1, columnspan=3, sticky="ew", padx=4)

        # Display mode dropdown
        mode_frame = ttk.Frame(main_frame)
        mode_frame.pack(side="top", fill="x", pady=(4, 4))

        ttk.Label(mode_frame, text="Display Mode:").pack(side="left")
        self.display_mode_var = tk.StringVar(value=self.DISPLAY_MODES[2])  # default: Diff+EQ
        mode_combo = ttk.Combobox(mode_frame,
                                  textvariable=self.display_mode_var,
                                  values=self.DISPLAY_MODES,
                                  state="readonly",
                                  width=10)
        mode_combo.pack(side="left", padx=4)

        # Side-by-side toggle
        self.side_by_side_var = tk.BooleanVar(value=False)
        side_check = ttk.Checkbutton(mode_frame,
                                     text="Show Raw vs Processed",
                                     variable=self.side_by_side_var,
                                     command=self.on_side_toggle)
        side_check.pack(side="left", padx=8)

        # Main preview area
        preview_frame = ttk.Frame(main_frame)
        preview_frame.pack(side="top", fill="both", expand=True, pady=(8, 0))

        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(fill="both", expand=True)

    # ---------------- GUI callbacks ---------------- #

    def capture_reference(self):
        """
        Store current raw frame as reference (drum at rest).
        """
        if self.last_raw is not None:
            self.reference = self.last_raw.copy()
            print("Reference frame captured.")

    def save_current_frame(self):
        """
        Save the currently displayed processed frame as PNG.
        """
        if self.last_processed is None:
            print("No processed frame to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png")])
        if path:
            cv2.imwrite(path, self.last_processed)
            print(f"Saved {path}")

    def toggle_recording(self):
        """
        Start or stop video recording of the processed view.
        """
        if not self.is_recording:
            # Start recording
            if self.last_processed is None:
                print("No frame yet to determine size; wait a moment and try again.")
                return

            path = filedialog.asksaveasfilename(defaultextension=".avi",
                                                filetypes=[("AVI", "*.avi")],
                                                title="Save recording as...")
            if not path:
                return

            h, w = self.last_processed.shape
            # FourCC for MJPG; write grayscale as 1-channel
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.video_writer = cv2.VideoWriter(path, fourcc, self.video_fps, (w, h), isColor=False)

            if not self.video_writer.isOpened():
                print("Could not open VideoWriter.")
                self.video_writer = None
                return

            self.video_path = path
            self.is_recording = True
            self.record_btn.config(text="Stop Recording")
            print(f"Recording started: {path}")

        else:
            # Stop recording
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.record_btn.config(text="Start Recording")
            print(f"Recording stopped: {self.video_path}")
            self.video_path = None

    def quit(self):
        """
        Stop acquisition and close the application.
        """
        self.running = False

    def on_exposure_change(self, value):
        """
        Update camera exposure (ms slider -> µs node value).
        """
        try:
            ms = float(value)
            set_float(self.nodemap, "ExposureTime", ms * 1000.0)
        except Exception:
            pass

    def on_gain_change(self, value):
        """
        Update camera gain (dB).
        """
        try:
            gain = float(value)
            set_float(self.nodemap, "Gain", gain)
        except Exception:
            pass

    def on_side_toggle(self):
        """
        Show or hide the side-by-side window.
        """
        if self.side_by_side_var.get():
            if self.side_window is None or not self.side_window.winfo_exists():
                self.side_window = tk.Toplevel(self.root)
                self.side_window.title("Raw vs Processed")
                self.side_label = ttk.Label(self.side_window)
                self.side_label.pack(fill="both", expand=True)
        else:
            if self.side_window is not None and self.side_window.winfo_exists():
                self.side_window.destroy()
                self.side_window = None

    # ---------------- Frame processing ---------------- #

    def process_frame(self, frame):
        """
        Given raw Mono8 frame, compute processed frame according to:
        - display mode
        - reference frame
        - contrast scale
        Returns processed Mono8 frame.
        """
        mode = self.display_mode_var.get()
        if mode == "Raw" or self.reference is None:
            # Display raw if:
            # - Raw mode selected
            # - OR no reference yet (regardless of mode)
            return frame

        diff = cv2.absdiff(frame, self.reference)

        # Apply contrast scaling: factor = contrast_var / 10
        scale_factor = max(0.1, self.contrast_var.get() / 10.0)
        diff_scaled = np.clip(diff.astype(np.float32) * scale_factor, 0, 255).astype(np.uint8)

        if mode == "Diff":
            return diff_scaled
        elif mode == "Diff+EQ":
            return cv2.equalizeHist(diff_scaled)
        else:
            # Fallback
            return frame

    # ---------------- Display helpers ---------------- #

    @staticmethod
    def cv2_to_tk_image(gray_frame, max_size=(960, 720)):
        """
        Convert a Mono8 OpenCV frame to a Tkinter PhotoImage.
        Optionally resize to fit max_size while preserving aspect ratio.
        """
        h, w = gray_frame.shape
        # Compute scale factor
        scale = min(max_size[0] / w, max_size[1] / h, 1.0)
        if scale != 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(gray_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = gray_frame

        # Convert to PIL Image then to PhotoImage
        pil_img = Image.fromarray(resized)
        return ImageTk.PhotoImage(image=pil_img)

    def update_main_preview(self):
        """
        Update the main preview Tk label with the latest processed frame.
        """
        if self.last_processed is None:
            return
        self.tk_main_image = self.cv2_to_tk_image(self.last_processed)
        self.preview_label.configure(image=self.tk_main_image)

    def update_side_preview(self):
        """
        Update the side-by-side window (raw vs processed).
        """
        if (self.side_window is None or
                not self.side_window.winfo_exists() or
                self.last_raw is None or
                self.last_processed is None):
            return

        # Stack raw and processed horizontally
        raw = self.last_raw
        proc = self.last_processed

        # Make them the same size if needed
        if raw.shape != proc.shape:
            proc = cv2.resize(proc, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_AREA)

        combined = np.hstack([raw, proc])
        self.tk_side_image = self.cv2_to_tk_image(combined, max_size=(1600, 720))
        self.side_label.configure(image=self.tk_side_image)

    # ---------------- Main update loop ---------------- #

    def update_loop(self):
        """
        Periodic acquisition + processing loop, scheduled via Tkinter .after().
        """
        if not self.running:
            # Clean up camera resources and quit
            try:
                self.cam.EndAcquisition()
            except Exception:
                pass
            try:
                self.cam.DeInit()
            except Exception:
                pass
            try:
                self.system.ReleaseInstance()
            except Exception:
                pass
            self.root.destroy()
            return

        try:
            image = self.cam.GetNextImage(1000)  # 1 ms timeout
            if not image.IsIncomplete():
                raw = get_mono8_array(image)
                self.last_raw = raw
                proc = self.process_frame(raw)
                self.last_processed = proc

                # Recording
                if self.is_recording and self.video_writer is not None:
                    self.video_writer.write(proc)

                # Update preview(s)
                self.update_main_preview()
                self.update_side_preview()

            image.Release()

        except PySpin.SpinnakerException:
            # Ignore transient acquisition errors
            pass

        # Schedule next iteration
        self.root.after(1, self.update_loop)


# -------------------- entry point -------------------- #

if __name__ == "__main__":
    root = tk.Tk()
    app = ESPIApp(root)
    root.mainloop()
