"""
Fast ESPI Drumhead Imaging GUI
FLIR Blackfly + PySpin + OpenCV + Tkinter

Changes vs previous version:
- Tkinter used only for controls (buttons, sliders, dropdown, checkbox).
- All image display is via OpenCV windows (no PIL/ImageTk).
- This removes heavy conversions and greatly reduces lag.

Features:
- Live camera view.
- Time-averaged ESPI-style processing (Raw / Diff / Diff+EQ).
- Buttons:
    - Capture Reference
    - Start/Stop Recording
    - Save Current Frame (PNG)
    - Quit
- Sliders:
    - Exposure (ms)
    - Gain (dB)
    - Diff contrast scale
- Display mode dropdown.
- Checkbox: show Raw vs Processed side-by-side in a second OpenCV window.

Requirements:
    pip install opencv-python numpy
    (PySpin / Spinnaker SDK installed from Teledyne)
"""

import cv2
import numpy as np
import PySpin
import tkinter as tk
from tkinter import ttk, filedialog


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
    node = PySpin.CEnumerationPtr(nodemap.GetNode(name))
    if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
        return
    entry = node.GetEntryByName(value)
    if PySpin.IsReadable(entry):
        node.SetIntValue(entry.GetValue())


def set_float(nodemap, name, val):
    node = PySpin.CFloatPtr(nodemap.GetNode(name))
    if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
        return
    val = float(val)
    val = max(node.GetMin(), min(node.GetMax(), val))
    node.SetValue(val)


def get_mono8_array(image):
    """
    Convert a PySpin ImagePtr to a 2D uint8 NumPy array.
    Assumes the camera is already set to Mono8 pixel format.
    """
    w = image.GetWidth()
    h = image.GetHeight()
    data = image.GetData()
    arr = np.array(data, dtype=np.uint8).reshape((h, w))
    return arr


# ------------------- ESPI Application ------------------- #

class ESPIApp:
    DISPLAY_MODES = ["Raw", "Diff", "Diff+EQ"]

    def __init__(self, root):
        self.root = root
        self.root.title("ESPI Drumhead Imaging (Fast OpenCV Display)")

        # Connect camera
        self.system, self.cam, self.nodemap = get_camera()
        self.configure_camera()

        # Start acquisition
        self.cam.BeginAcquisition()

        # State
        self.reference = None
        self.last_raw = None
        self.last_processed = None
        self.running = True

        # Recording
        self.is_recording = False
        self.video_writer = None
        self.video_path = None
        self.video_fps = 30.0

        # Build GUI
        self.build_gui()

        # Create OpenCV windows
        cv2.namedWindow("ESPI Processed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ESPI Processed", 960, 720)

        # Side-by-side window flag
        self.side_window_active = False

        # Start update loop
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

        # Optional: try to set higher frame rate (if supported)
        try:
            set_enum(self.nodemap, "AcquisitionFrameRateEnable", "True")
            set_float(self.nodemap, "AcquisitionFrameRate", 60.0)
        except Exception:
            pass

        set_float(self.nodemap, "ExposureTime", 3000)  # 3 ms
        set_float(self.nodemap, "Gain", 6)

        # Optional: reduce resolution (ROI) for less data per frame (uncomment if desired)
        # try:
        #     width_node = PySpin.CIntegerPtr(self.nodemap.GetNode("Width"))
        #     height_node = PySpin.CIntegerPtr(self.nodemap.GetNode("Height"))
        #     width_node.SetValue(min(800, width_node.GetMax()))
        #     height_node.SetValue(min(600, height_node.GetMax()))
        # except Exception:
        #     pass

    # ---------------- GUI construction -------------- #

    def build_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=8, pady=8, fill="both", expand=True)

        controls = ttk.Frame(main_frame)
        controls.pack(side="top", fill="x")

        ttk.Button(controls, text="Capture Reference",
                   command=self.capture_reference).grid(row=0, column=0, padx=4, pady=4)
        ttk.Button(controls, text="Save Current Frame",
                   command=self.save_current_frame).grid(row=0, column=1, padx=4, pady=4)

        self.record_btn = ttk.Button(controls, text="Start Recording",
                                     command=self.toggle_recording)
        self.record_btn.grid(row=0, column=2, padx=4, pady=4)

        ttk.Button(controls, text="Quit",
                   command=self.quit).grid(row=0, column=3, padx=4, pady=4)

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

        # Contrast scale
        ttk.Label(controls, text="Diff Contrast Scale").grid(row=3, column=0, sticky="w")
        self.contrast_var = tk.DoubleVar(value=10.0)  # factor / 10
        contrast_scale = ttk.Scale(controls, from_=1, to=30,
                                   variable=self.contrast_var)
        contrast_scale.grid(row=3, column=1, columnspan=3, sticky="ew", padx=4)

        # Display mode + side-by-side
        mode_frame = ttk.Frame(main_frame)
        mode_frame.pack(side="top", fill="x", pady=(4, 0))

        ttk.Label(mode_frame, text="Display Mode:").pack(side="left")
        self.display_mode_var = tk.StringVar(value=self.DISPLAY_MODES[2])
        mode_combo = ttk.Combobox(mode_frame,
                                  textvariable=self.display_mode_var,
                                  values=self.DISPLAY_MODES,
                                  state="readonly",
                                  width=10)
        mode_combo.pack(side="left", padx=4)

        self.side_by_side_var = tk.BooleanVar(value=False)
        side_check = ttk.Checkbutton(mode_frame,
                                     text="Raw vs Processed window",
                                     variable=self.side_by_side_var)
        side_check.pack(side="left", padx=8)

    # ---------------- GUI callbacks ---------------- #

    def capture_reference(self):
        if self.last_raw is not None:
            self.reference = self.last_raw.copy()
            print("Reference frame captured.")

    def save_current_frame(self):
        if self.last_processed is None:
            print("No processed frame to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png")])
        if path:
            cv2.imwrite(path, self.last_processed)
            print(f"Saved {path}")

    def toggle_recording(self):
        if not self.is_recording:
            if self.last_processed is None:
                print("No frame yet to determine size; try again in a moment.")
                return
            path = filedialog.asksaveasfilename(defaultextension=".avi",
                                                filetypes=[("AVI", "*.avi")],
                                                title="Save recording as...")
            if not path:
                return
            h, w = self.last_processed.shape
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.video_writer = cv2.VideoWriter(path, fourcc,
                                                self.video_fps, (w, h),
                                                isColor=False)
            if not self.video_writer.isOpened():
                print("Could not open VideoWriter.")
                self.video_writer = None
                return
            self.video_path = path
            self.is_recording = True
            self.record_btn.config(text="Stop Recording")
            print(f"Recording started: {path}")
        else:
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.record_btn.config(text="Start Recording")
            print(f"Recording stopped: {self.video_path}")
            self.video_path = None

    def quit(self):
        self.running = False

    def on_exposure_change(self, value):
        try:
            ms = float(value)
            set_float(self.nodemap, "ExposureTime", ms * 1000.0)
        except Exception:
            pass

    def on_gain_change(self, value):
        try:
            gain = float(value)
            set_float(self.nodemap, "Gain", gain)
        except Exception:
            pass

    # ---------------- Frame processing ---------------- #

    def process_frame(self, frame):
        mode = self.display_mode_var.get()
        if mode == "Raw" or self.reference is None:
            return frame

        diff = cv2.absdiff(frame, self.reference)
        scale_factor = max(0.1, self.contrast_var.get() / 10.0)
        diff_scaled = np.clip(diff.astype(np.float32) * scale_factor,
                              0, 255).astype(np.uint8)

        if mode == "Diff":
            return diff_scaled
        elif mode == "Diff+EQ":
            return cv2.equalizeHist(diff_scaled)
        else:
            return frame

    # ---------------- Main update loop ---------------- #

    def update_loop(self):
        if not self.running:
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
            cv2.destroyAllWindows()
            self.root.destroy()
            return

        try:
            image = self.cam.GetNextImage(1000)
            if not image.IsIncomplete():
                raw = get_mono8_array(image)
                self.last_raw = raw
                proc = self.process_frame(raw)
                self.last_processed = proc

                # Recording
                if self.is_recording and self.video_writer is not None:
                    self.video_writer.write(proc)

                # Show processed image
                cv2.imshow("ESPI Processed", proc)

                # Optional raw vs processed
                if self.side_by_side_var.get():
                    if raw.shape != proc.shape:
                        proc_resized = cv2.resize(proc, (raw.shape[1], raw.shape[0]),
                                                  interpolation=cv2.INTER_AREA)
                    else:
                        proc_resized = proc
                    combined = np.hstack([raw, proc_resized])
                    cv2.imshow("Raw vs Processed", combined)
                else:
                    # Close side window if not needed
                    cv2.destroyWindow("Raw vs Processed")

                cv2.waitKey(1)

            image.Release()

        except PySpin.SpinnakerException:
            pass

        # Schedule next iteration (small delay; Tkinter handles GUI)
        self.root.after(1, self.update_loop)


# -------------------- entry point -------------------- #

if __name__ == "__main__":
    root = tk.Tk()
    app = ESPIApp(root)
    root.mainloop()
