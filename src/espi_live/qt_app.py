"""Cross-platform PySide6 front end with GPU image display.

The window is one QMainWindow: the image in the center, a dockable controls
panel and log, and a status bar. Docks can be floated as separate windows.

Display path: the engine publishes a Mono8 FramePacket; the GPU view uploads
it as a single-channel texture, and a fragment shader applies the display
gamma through a 256-entry gray lookup table built by processing.display_rgb_table
(the same function saved frames use). Scaling to the window is done by the
GPU with mipmaps. Machines without OpenGL 3.3 (for example some remote-desktop
sessions) fall back to a CPU-drawn view with identical controls.

Several cameras: the window owns one Channel per detected camera. A drop-down
picks the camera on screen, and a "process" checkbox per camera keeps others
acquiring, processing and recording in the background. Every control acts on
the displayed camera's own engine and settings. Those settings are the
camera's profile, saved between runs through profiles.ProfileStore, and the
Camera section's Presets row saves and applies named presets.
"""

import faulthandler
import html
import json
import logging
import math
import os
import queue
import random
import sys
import textwrap
import threading
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime

import cv2
import numpy as np
from PySide6.QtCore import (QCoreApplication, QEvent, QObject, QPoint, QPointF, QRect, QRectF,
                            QSettings, QSize, Qt, QtMsgType, QTimer, QUrl, Signal,
                            qInstallMessageHandler)
from PySide6.QtGui import (QAction, QColor, QDesktopServices, QFont, QFontMetrics, QIcon, QImage,
                           QKeySequence, QOffscreenSurface, QOpenGLContext, QPainter,
                           QPainterPath, QPalette, QPen, QPixmap, QSurfaceFormat, QVector4D)
from PySide6.QtOpenGL import (QOpenGLShader, QOpenGLShaderProgram,
                              QOpenGLVertexArrayObject)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (QAbstractButton, QAbstractSpinBox, QApplication, QButtonGroup,
                               QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDockWidget, QDoubleSpinBox, QFormLayout, QFrame, QGridLayout,
                               QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLayout,
                               QLineEdit, QMainWindow,
                               QMessageBox,
                               QPlainTextEdit, QProxyStyle, QPushButton, QScrollArea,
                               QSizePolicy, QSlider, QStackedWidget, QStyle, QTextBrowser,
                               QTextEdit, QToolBar, QToolButton, QToolTip,
                               QVBoxLayout, QWidget)

from espi_live import (camera, camera_reports, capture_uploads, codex, control,
                       daq, error_reports, feedback, iphone, log_uploads, paths,
                       phase, profiles, simulation, source_code, step_tuning,
                       tips, ui_scale, updater)
from espi_live.channels import demo_channels, open_camera_channels, restore_channel
from espi_live.engine import LABVIEW_FLOW, FramePacket
from espi_live.guide import guide_path
from espi_live.logs import AppLog, header_lines
from espi_live.pixel_square import DEFAULT_SQUARE_SIZE, SQUARE_SIZES
from espi_live.processing import (DEFAULT_DIFFERENCE_GAIN, MAX_DENOISE_PERCENT,
                                  MAX_FRAMES_BACK, MAX_REFERENCE_FRAMES,
                                  SCALING_MODES, SMOOTHING_METHODS,
                                  VIEW_MODES, display_rgb_table)

# Wide enough that a full help tip and the DUMMY MODE badge both fit. A saved
# layout is kept; a narrower window keeps the badge and elides the tip. Either
# size shrinks to the screen when shown (MainWindow.fit_to_screen).
DEFAULT_WIDTH = 2100
DEFAULT_HEIGHT = 900
QWIDGETSIZE_MAX = 16777215  # Qt's "no maximum size" (not exported by PySide6)

ANALOG_CONTROL_TIP = "Set on the camera itself; not adjustable over analog video"
IPHONE_GAIN_TIP = "iPhone: sets the phone's ISO; 0 dB is its lowest ISO, +20 dB ten times that"
IPHONE_CONTROL_TIP = "iPhone: sent to the ESPI Camera app over USB"
PHONE_FRAME_RATES = (1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 24.0, 25.0, 30.0, 60.0)
# Gamma stays its own Display control: it is a curve, not a second stretch.
GAMMA_TIP = (
    "Shifts mid-tones after Scaling (Fixed gain or Auto stretch), in Raw, "
    "Previous and Reference. Scaling is a straight remap of the difference "
    "values (contrast); gamma is a curve on top of that. Below 1.0 darkens mid-tones, "
    "above 1.0 brightens them. With a good stretch, start at 1.0."
)
SCALING_TIP = (
    "How difference values become screen brightness. Fixed gain multiplies "
    "every difference by Difference gain and shows anything above 255 as "
    "white, as the LabVIEW VI did: a still scene's few levels of noise stay "
    "dark and the brightness does not pump. Auto stretch maps each frame's "
    "darkest and brightest pixels, after Stretch clip, to 0 and 255, which "
    "fills the display but also blows a quiet frame's noise up to full "
    "contrast. Both are straight remaps (contrast), not gamma. The status bar "
    "shows the input and scaled ranges. Analog cameras start in Fixed gain, "
    "others in Auto stretch. Previous and Reference only; Raw is not scaled."
)
GL_TEXTURE_2D = 0x0DE1
GL_R8 = 0x8229
GL_RED = 0x1903
GL_RGB = 0x1907
GL_RGB8 = 0x8051
GL_UNSIGNED_BYTE = 0x1401
GL_TEXTURE_MIN_FILTER = 0x2801
GL_TEXTURE_MAG_FILTER = 0x2800
GL_TEXTURE_WRAP_S = 0x2802
GL_TEXTURE_WRAP_T = 0x2803
GL_CLAMP_TO_EDGE = 0x812F
GL_NEAREST = 0x2600
GL_LINEAR_MIPMAP_LINEAR = 0x2703
GL_UNPACK_ALIGNMENT = 0x0CF5
GL_TEXTURE0 = 0x84C0
GL_TRIANGLE_STRIP = 0x0005
GL_COLOR_BUFFER_BIT = 0x4000

VERTEX_SHADER = """#version 330 core
uniform vec4 rect;  // left, top, right, bottom in normalized device coordinates
out vec2 uv;
void main() {
    vec2 corner = vec2(gl_VertexID & 1, gl_VertexID >> 1);
    uv = corner;  // texture row 0 is the image's top row
    gl_Position = vec4(mix(rect.x, rect.z, corner.x),
                       mix(rect.y, rect.w, corner.y), 0.0, 1.0);
}
"""
FRAGMENT_SHADER = """#version 330 core
in vec2 uv;
out vec4 color;
uniform sampler2D image;  // R8 Mono8 measurement
uniform sampler2D lut;    // 256x1 RGB: display gamma
void main() {
    int index = int(texture(image, uv).r * 255.0 + 0.5);
    color = vec4(texelFetch(lut, ivec2(index, 0), 0).rgb, 1.0);
}
"""
MAX_ZOOM = 64.0
ZOOM_TICKS = 1000  # zoom slider resolution; position maps logarithmically to 1..MAX_ZOOM
CODEX_LABEL = "Open in Codex"
CODEX_BUSY_LABEL = "Getting source…"
# Icon-only status-bar actions: the label is the accessible name and leads the
# tooltip, and the busy label replaces it while the action runs.
CHECK_LABEL = "Check for updates"
CHECK_BUSY_LABEL = "Checking for updates…"
FEATURE_LABEL = "Request feature"
FEATURE_BUSY_LABEL = "Sending request…"
CHECK_TIP = "Check GitHub for a newer build of ESPI live."
FEATURE_TIP = "Send a feature request to the developers as a GitHub issue."
CODEX_TIP = "Ask OpenAI's Codex how ESPI live works, using this build's real source code."
# Simple UI: only the basic controls are rendered. The rest are still created
# (so settings, presets, shortcuts and tests keep working) but are not shown.
# SIMPLE_UI is the default for a fresh install; the toolbar's Simple UI button
# switches it while the window runs and the choice is remembered
# (SIMPLE_UI_SETTING_KEY).
SIMPLE_UI = True
SIMPLE_UI_SETTING_KEY = "ui/simple"
SIMPLE_UI_TIP = ("Show only the basic controls. Turn it off to show every control, "
                 "section and toolbar button. Your settings are kept either way.")
LAYOUT_VERSION = 5  # bump when docks/toolbars change so stale saved layouts are ignored
# A camera PC can stay open for days. Recheck well inside GitHub's rate limit.
UPDATE_CHECK_INTERVAL_MS = 3 * 60 * 60 * 1000
UPDATE_CHECK_JITTER_MS = 20 * 60 * 1000
TEXT_INPUTS = (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QComboBox)
# Control ranges before a camera reports its limits (CAMERA_RANGES in camera.py).
DEFAULT_LIMITS = {"ExposureTime": (0, 50000), "Gain": (0, 30),
                  "AcquisitionFrameRate": (0, 60)}
REPORT_STATUS_MS = 60000  # how long an error-report message stays in the status bar
QT_LEVELS = {QtMsgType.QtDebugMsg: logging.DEBUG, QtMsgType.QtInfoMsg: logging.INFO,
             QtMsgType.QtWarningMsg: logging.WARNING, QtMsgType.QtCriticalMsg: logging.ERROR,
             QtMsgType.QtFatalMsg: logging.ERROR}
qt_logger = logging.getLogger("espi_live.qt")


def handle_qt_message(kind, context, message):
    """Log a Qt message; report critical and fatal ones (qInstallMessageHandler).

    Qt calls this from any thread, so it only logs and captures.

    Args: kind is the QtMsgType; context is the QMessageLogContext; message
        is the text.
    Returns: None.
    """
    category = getattr(context, "category", "") or ""
    where = f"Qt ({category})" if category not in ("", "default") else "Qt"
    qt_logger.log(QT_LEVELS.get(kind, logging.WARNING), f"{where}: {message}")
    if kind not in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
        return
    fatal = kind == QtMsgType.QtFatalMsg
    reporter = error_reports.current_reporter()
    report = None
    if reporter is not None:
        stack = traceback.extract_stack()[:-1]
        report = reporter.capture_message("fatal" if fatal else "critical", message,
                                 [(frame.filename, frame.name) for frame in stack],
                                 "".join(traceback.format_list(stack)))
    if fatal and report is not None:
        faulthandler.disable()  # Qt aborts next; this report already covers it


def setting_on(value):
    """Read a QSettings flag, which some backends store as text.

    Args: value is what QSettings returned.
    Returns: False for False, 0, "false" or "0"; otherwise True.
    """
    return str(value).strip().lower() not in ("false", "0")


@dataclass
class ControlRow:
    """One row of the controls grid that the simple UI can take out and put back.

    Attributes: widget is the control; label is its name, or None for a
        full-width row; grid is the section grid and row its row number there;
        name is the QLabel in column 0 while the row is placed; placed is False
        while the widget is parked in the hidden holder.
    """

    widget: object
    label: object
    grid: object
    row: int
    name: object = None
    placed: bool = False


class Reporting:
    """This run's error reporter, Python hooks, Qt message handler, worker and log uploads."""

    def __init__(self, log, send_reports=True):
        """Create the reporter from the switches; nothing is hooked until start().

        Args: log is the AppLog; send_reports is False for --no-error-reports.
        Returns: None.
        """
        self.cli_off = not send_reports
        settings = QSettings("ESPI", "espi_live")
        self.chosen = setting_on(settings.value(error_reports.SETTING_KEY, True))
        enabled, self.forced = error_reports.resolve_enabled(self.cli_off, self.chosen)
        self.reporter = error_reports.ErrorReporter(enabled, log_tail=log.recent_lines,
                                                    uptime=log.uptime)
        self.reporter.update_context(front_end="Qt")
        self.worker = error_reports.ErrorReportWorker(self.reporter)
        self.reporter.listeners.append(self.worker.kick)
        # Log uploads have no switch: every copy with the logs token sends them.
        self.uploads = log_uploads.LogUploadWorker(log_uploads.LogUploader(
            paths.output_folder("logs"), self.reporter.serials))
        self.captures = None  # the CaptureWorker, built by start_uploads when a token exists
        self.hooks = None
        self.previous_handler = None

    def start(self, log):
        """Install the hooks, report earlier native crashes and start sending.

        Args: log is the AppLog.
        Returns: None.
        """
        self.hooks = error_reports.install_hooks(self.reporter)
        self.previous_handler = qInstallMessageHandler(handle_qt_message)
        log.info("Error reports: " + ("on" if self.reporter.enabled else
                                      f"off ({self.forced or 'turned off in the Help menu'})"))
        log.info("Log uploads: " + ("on" if log_uploads.logs_token()
                                    else "off (no logs token in this copy)"))
        log.info("Capture uploads: " + ("on" if capture_uploads.captures_token()
                                        else "off (no captures token in this copy)"))
        self.reporter.collect_native_crashes(log.crash_path)
        self.worker.start()

    def start_uploads(self, channels):
        """Start the log and capture uploads, once the cameras are open and their serials known.

        Capture uploads, like log uploads, have no switch: a copy with the
        captures token sends scheduled stills and clips from its real cameras.

        Args: channels is a zero-argument callable returning the window's
            current channel list.
        Returns: None.
        """
        self.uploads.start()
        if not capture_uploads.captures_token():
            return
        try:
            folder = paths.output_folder("captures")
            logs_folder = paths.output_folder("logs")
        except OSError as exc:  # a silent feature must never stop the instrument
            logging.getLogger(__name__).debug("Capture uploads off: %s", exc)
            return
        queue = capture_uploads.CaptureQueue(folder, capture_uploads.MAX_QUEUE_BYTES)
        self.captures = capture_uploads.CaptureWorker(
            channels, folder, capture_uploads.CaptureUploader(queue),
            capture_uploads.CaptureLedger(folder), queue,
            log_uploads.machine_id(logs_folder), updater.current_build())
        self.captures.start()

    def choose(self, on, log):
        """Apply the Help menu choice.

        Args: on is the new choice; log is the AppLog.
        Returns: None.
        """
        self.chosen = on
        QSettings("ESPI", "espi_live").setValue(error_reports.SETTING_KEY, on)
        self.reporter.enabled, self.forced = error_reports.resolve_enabled(self.cli_off, on)
        log.info(f"Error reports: {'on' if self.reporter.enabled else 'off'}")
        if self.reporter.enabled:
            self.worker.kick()

    def stop(self):
        """Restore the hooks and the Qt message handler and stop the worker.

        Args: None.
        Returns: None.
        """
        if self.hooks is not None:
            qInstallMessageHandler(self.previous_handler)
            self.hooks.uninstall()
            self.hooks = None
        self.worker.stop()
        self.uploads.stop()
        if self.captures is not None:
            self.captures.stop()


def describe_session(channels, display):
    """Summarize what this run uses, for the log.

    Args: channels is the list of Channel; display describes the image view.
    Returns: One line.
    """
    if all(channel.demo for channel in channels):
        source = f"dummy mode, {len(channels)} synthetic camera(s)"
    else:
        source = f"{len(channels)} camera(s): " + ", ".join(
            channel.label + (" (processing)" if channel.active else "") for channel in channels)
    spinnaker = camera.spinnaker_version()
    return (f"Session: {source}; display {display}"
            + (f"; Spinnaker {spinnaker}" if spinnaker else ""))


# Amber marks a frozen (held, not live) image; red is reserved for dummy mode.
FROZEN_STYLE = "background:#e0a000;color:black;font-weight:bold;"
PIEZO_DEBUG_STYLE = ("background:#101010;color:#7cfc00;font-family:Menlo,Consolas,"
                     "'DejaVu Sans Mono','Courier New',monospace;padding:4px 8px;")
PIEZO_DEBUG_WINDOW_S = 2.0  # the overlay's step figures cover the last this many seconds
# The pixel square's outline on the main image: a bright line with a dark
# edge reads on every palette. A square smaller than BOX_MIN_SIZE screen
# pixels is outlined at that size around its center, so a 1×1 square shows.
BOX_COLOR = QColor(255, 212, 0)
BOX_EDGE = QColor(0, 0, 0, 200)
BOX_MIN_SIZE = 8
# A press that moves less than this many screen pixels is a click, not a pan.
CLICK_SLOP = 4
# The longest view hint the engine reports (the status bar measures it), and the
# least width the tip keeps beside it. The hint is information and gets its whole
# sentence; the tip takes what is left and is shortened with an ellipsis.
LONGEST_HINT = "Reference was taken at noise reduction 100; press F to retake"
# QSettings key prefix; "<prefix><section heading>" is true while that
# Controls section is collapsed.
SECTION_SETTING_PREFIX = "controls/collapsed/"
# Each camera's profile is written once it has not changed for this long
# (profiles.ProfileStore.save_when_settled), so a slider drag writes once.
PROFILE_SAVE_MS = 1000
PRESET_NAME_MAX = 60  # characters in a saved preset's name
PROFILE_TIP = (
    "Every control in this panel changes only this camera. Each camera keeps "
    "its own settings, saved between runs; pick another camera under Display "
    "to edit its settings."
)
PRESET_COMBO_TIP = (
    "Your saved presets: a camera's exposure, gain, frame rate and every "
    "processing and display setting, including the LabVIEW flow. Pick one, "
    "then press Apply to use it on the displayed camera."
)
PRESET_APPLY_TIP = (
    "Use the chosen preset on the displayed camera only. Exposure, gain and "
    "frame rate are fitted to this camera's limits, and the log lists any "
    "value that had to change. The camera keeps these settings next time."
)
PRESET_SAVE_TIP = (
    "Save the displayed camera's current settings as a named preset, which "
    "any camera can apply later. A name already in use is replaced after you "
    "confirm."
)
PRESET_DELETE_TIP = "Remove the chosen saved preset. Cameras keep their settings."
RESET_CAMERA_TIP = (
    "Put every control of the displayed camera back to its starting values "
    "(exposure 8000 µs, gain 0 dB, 30 fps on a FLIR camera, and the default "
    "noise pipeline: Average N 4 on camera frames, 16 reference frames, "
    "Oriented speckle filter) and clear its saved settings, after you "
    "confirm. Other cameras and saved presets do not change."
)
MESSAGE_TIP_WIDTH = 120
# Previous and Reference views run in this order. Raw skips difference processing but still applies display gamma.
PIPELINE = ("noise reduction → (frame average) → difference (N frames back) → "
            "noise threshold → beam brightness → outlier filter → average → "
            "speckle filter → noise floor → peak hold → fixed gain, stretch or "
            "local normalization → invert → brightness, gain and contrast → gamma. "
            "Phase view: piezo phase steps → wrapped phase → minus the reference "
            "phase → sine/cosine filter")
# Fringe-enhancement and display-tone values every preset sets unless it
# overrides them, so a preset never inherits peak hold or a threshold from the
# last one. The sensitivity floor is not here: it is per camera (analog capture
# starts at 30, Calibrate noise measures it) and no preset changes it.
FRINGE_TOOLS_OFF = {
    "average_frames_first": False,
    "floor_subtract": False,
    "threshold": 0,
    "steady_range_s": 0.0,
    "even_brightness": False,
    "peak_hold_s": 0.0,
    "normalize_window": 0,
    "brightness": 0,
    "display_gain": 1.0,
    "contrast": 0.0,
}
# Opt-in processing setups. Exposure, gain, frame rate and recording stay put.
PROCESSING_PRESETS = {
    "Live preview": {
        **FRINGE_TOOLS_OFF,
        "mode": "Previous",
        "noise_reduction": 0.0,
        "average": 1,
        "rolling": False,
        "ref_refresh_s": 0.0,
        "smoothing": 0,
        "smoothing_method": "Gaussian",
        "frames_back": 1,
        "outlier_filter": 0,
        "scale_range": True,
        "gain_before_average": False,
        "clip_percent": 0.0,
        "invert": False,
        "gamma": 1.0,
    },
    "Measurement": {
        **FRINGE_TOOLS_OFF,
        "mode": "Reference",
        "noise_reduction": 0.0,
        "average": 8,
        "rolling": False,
        "ref_refresh_s": 2.0,
        "smoothing": 3,
        "smoothing_method": "Median",
        "frames_back": 1,
        "outlier_filter": 0,
        "scale_range": True,
        "gain_before_average": False,
        "clip_percent": 0.5,
        "invert": False,
        "gamma": 1.0,
    },
    # A vibrating object seen through a noisy analog camera (Sony XC-ST50 on
    # a USB-Live2): each pixel keeps its largest frame-to-frame change for 1 s,
    # so a steadily driven drum's ring pattern builds up. The threshold and
    # Median filter run before peak hold, which would otherwise ratchet noise
    # and hot pixels up to their maximum.
    "Peak hold": {
        **FRINGE_TOOLS_OFF,
        "mode": "Previous",
        "noise_reduction": 0.0,
        "average": 1,
        "rolling": False,
        "ref_refresh_s": 0.0,
        "smoothing": 3,
        "smoothing_method": "Median",
        "frames_back": 1,
        "outlier_filter": 0,
        "scale_range": True,
        "gain_before_average": False,
        "clip_percent": 0.5,
        "invert": False,
        "gamma": 1.0,
        "threshold": 6,
        "steady_range_s": 1.0,
        "peak_hold_s": 1.0,
        "contrast": 30.0,
    },
}
RAW_DIFFERENCE_TIP = (
    "Show only the difference between two raw camera frames: each pixel is "
    "|live - previous| in the camera's own grey levels. Every other step is "
    "bypassed while it is on: noise reduction, single field, filters, "
    "averaging, difference gain, stretch, gamma, flip and display "
    "tone. The view buttons and your settings are kept and come back when "
    "you turn it off. Exposure and gain on the camera itself are unchanged.")
# The Raw difference button stands out while it overrides every setting.
RAW_DIFFERENCE_STYLE = "background:#c03030;color:white;font-weight:bold;"
LABVIEW_FLOW_TIP = (
    "Switch between your own settings and the LabVIEW ESPI 1.5 flow: Reference "
    "view with a reference captured now (and again whenever Reference view has "
    "none), full frames, Fixed gain applied to each frame before averaging, and "
    "no noise reduction, speckle filter or invert. Difference gain, "
    "exposure, gain, frame rate and recording stay as they are. Off by "
    "default: every camera starts with the noise pipeline (camera frames "
    "averaged before the difference, an averaged reference, the Oriented "
    "speckle filter), which the VI did not have. Turning it off puts back "
    "your own settings; changes made while it is on are dropped."
)
PRESET_TIPS = {
    "Live preview": "Watch fringes form while you align the setup: Previous view, "
                    "no averaging, no noise reduction, speckle filter off. "
                    "Does not change exposure, gain, frame rate or recording.",
    "Measurement": "A still object against one reference: Reference view, 8-frame "
                   "block average, Median filter 3 px, reference refresh 2 s. "
                   "Does not change exposure, gain, frame rate or recording.",
    "Peak hold": "Vibration fringes through a noisy analog camera: Previous "
                 "view, noise threshold 6, Median filter 3 px, peak hold 1 s, "
                 "steady range 1 s, S-curve contrast 30%. Keeps your sensitivity "
                 "floor. Does not change exposure, gain, frame rate or recording.",
}
SETTINGS_GUIDE = f"""A typical ESPI measurement starts with one of the three presets. None of them changes exposure, gain, frame rate or recording, and each turns off the fringe-enhancement and display controls it does not use.

Live preview — while you aim the camera and check that fringes form.
  View: Previous. Average N: 1 (off). Rolling average: off.
  Noise reduction: 0. Speckle filter: off. Ref. refresh: 0.
  Frames back: 1. Outlier filter: off.
  Scaling: Auto stretch. Gamma: 1.0.
  Switch the view to Raw to frame the object. Raw skips difference processing but still applies display gamma.

Measurement — a still object, compared with one reference.
  View: Reference. Press F (or Set reference) once the object is still.
  Average N: 8 with Rolling average off: one clean image every 8 frames.
  Speckle filter: Median, width 3 px, well below the fringe spacing.
  Ref. frames: 1 for a quiet camera; 8–30 for a noisy or analog one, so the reference's own noise is not frozen into every difference. Press C to see the reference beside the current frame and the difference.
  Ref. refresh: 2 s. That cancels slow drift (air, heat). Motion slower than the refresh disappears; use 0 and press F when you need one fixed before/after.
  Noise reduction: 0. Raise it only when the camera is noisy and the object is still, and set it before pressing F: the reference keeps the setting it was taken at, and a change afterwards leaves a frozen grain until you press F again.
  Scaling: Auto stretch, clip 0.5% so a few speckles don't flatten the contrast.
  Gamma: 1.0.

Peak hold — a vibrating object through a noisy analog camera (Sony XC-ST50 on the USB-Live2).
  View: Previous. Each pixel shows its largest frame-to-frame change over the last second, so a steadily driven object's ring pattern builds up instead of flickering.
  Noise threshold: 6 levels. Speckle filter: Median, 3 px. Both run before peak hold, which would otherwise keep the loudest noise and hot pixels.
  Peak hold window: 1 s. Longer fills in fainter rings but trails changes in the drive.
  Scaling: Auto stretch, clip 0.5%, steady range 1 s so the contrast does not pump.
  Sensitivity floor: unchanged. Analog cameras start at 30; press Calibrate noise with the drive off to measure your own.
  S-curve contrast: 30%. Gamma 1.0. Noise reduction, averaging and Ref. refresh: off.

Processing order (Previous and Reference views):
  {PIPELINE}

Raw view applies noise reduction and display gamma, without scaling.

Analog camera (Sony XC-ST50): it starts in Reference view with Fixed gain 42× (Auto stretch would blow the camera's few grey levels of noise up to full-contrast grain) and the noise pipeline every camera gets: four frames averaged before the difference, a 16-frame automatic reference and the Oriented speckle filter. The LabVIEW flow toggle gives the LabVIEW ESPI 1.5 VI's processing with its saved defaults instead: a one-frame automatic reference, Fixed gain 42× (the VI's Constant) on each frame, no averaging or filters. Lower Difference gain when noise shows as grey grain. Gain before averaging (LabVIEW order) and Average multiplier reproduce the VI's averaging maths, including its N + 1 frames divided by N.

Three controls that are easy to mix up:
  Noise reduction blends camera frames over time, before the difference. It removes sensor noise and keeps the speckle. It trails motion and weakens Previous-view fringes.
  Average N blends difference frames. It reduces speckle noise in the fringes. Off (block): one image per N frames. On (rolling): the full frame rate, with a lag of about N frames.
  The speckle filter blurs each difference image in space. Gaussian softens edges at any width. Median keeps fringe edges and stops at 5 px. Keep the width well below the fringe spacing or the fringes disappear.

Hover any control for the same advice: what it does, when to use it, and a starting value.
"""

# Status-bar actions. palette() follows the light or dark theme at startup.
# The blue update button stays the primary action, distinct from the red badge.
STATUS_BUTTON_STYLE = """
QPushButton {
    border: 1px solid palette(mid);
    border-radius: 4px;
    padding: 2px 8px;
    background: palette(base);
    color: palette(button-text);
}
QPushButton:hover {
    border-color: palette(highlight);
    background: palette(midlight);
}
QPushButton:pressed {
    background: palette(mid);
    border-color: palette(dark);
}
QPushButton:focus {
    border-color: palette(highlight);
}
QPushButton:disabled {
    color: palette(placeholder-text);
    background: palette(window);
    border: 1px solid palette(mid);
}
"""
UPDATE_BUTTON_STYLE = """
QPushButton {
    border: 1px solid #163a6b;
    border-radius: 4px;
    padding: 2px 8px;
    background: #1d4f91;
    color: white;
    font-weight: bold;
}
QPushButton:hover {
    background: #2a64b0;
    border-color: #102848;
}
QPushButton:pressed {
    background: #163a6b;
    border-color: #0c1e36;
}
QPushButton:focus {
    border-color: #8eb4e8;
}
QPushButton:disabled {
    background: #8aa4c4;
    color: #f4f7fb;
    border: 1px solid #6d88a8;
    font-weight: bold;
}
"""


def _paint_status_icon(kind, color, size=16):
    """Draw a monochrome icon for a status-bar action.

    The drawing is laid out on a 16-unit square and scaled to the size.

    Args: kind is "refresh", "download", "idea" or "code"; color is a QColor;
        size is the side in pixels.
    Returns: A QPixmap with a transparent background.
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    try:
        painter.scale(size / 16, size / 16)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(color, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                   Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if kind == "refresh":
            painter.drawArc(2, 2, 12, 12, 40 * 16, 280 * 16)
            painter.setBrush(color)
            arrow = QPainterPath()
            arrow.moveTo(7, 1)
            arrow.lineTo(13, 1)
            arrow.lineTo(13, 7)
            painter.drawPath(arrow)
        elif kind == "download":
            painter.drawLine(8, 2, 8, 9)
            painter.drawLine(4, 6, 8, 11)
            painter.drawLine(12, 6, 8, 11)
            painter.drawLine(3, 13, 13, 13)
        elif kind == "idea":
            painter.drawEllipse(4, 1, 8, 8)
            painter.drawLine(6, 9, 6, 12)
            painter.drawLine(10, 9, 10, 12)
            painter.drawLine(6, 12, 10, 12)
            painter.drawLine(7, 14, 9, 14)
        elif kind == "code":  # </>
            painter.drawPolyline([QPointF(5, 4), QPointF(1.5, 8), QPointF(5, 12)])
            painter.drawPolyline([QPointF(11, 4), QPointF(14.5, 8), QPointF(11, 12)])
            painter.drawLine(QPointF(9.5, 2.5), QPointF(6.5, 13.5))
    finally:
        painter.end()
    return pixmap


def _status_icon(kind, color, size=16):
    """Build a status-bar icon, with a dimmer copy for the disabled state.

    Args: kind is "refresh", "download", "idea" or "code"; color is a QColor;
        size is the side in pixels.
    Returns: A QIcon.
    """
    icon = QIcon()
    icon.addPixmap(_paint_status_icon(kind, color, size), QIcon.Mode.Normal, QIcon.State.Off)
    dimmed = QColor(color)
    dimmed.setAlpha(150)
    icon.addPixmap(_paint_status_icon(kind, dimmed, size), QIcon.Mode.Disabled,
                   QIcon.State.Off)
    return icon


def _arm_status_button(button, style, texts, icon=None):
    """Give a status-bar control button chrome, a pointer and a fixed width.

    The width fits every label the button shows, so a busy label cannot push
    the readouts sideways. The arguments are kept on the button, so the same
    call can be repeated when the interface size changes (the style sheet's
    lengths, the icon and the width all follow it).

    Args: button is a QPushButton; style is its style sheet, written for 100%;
        texts are the labels; icon is (kind, color) for _status_icon, or None.
    Returns: The fixed width in pixels.
    """
    button.arm = (style, texts, icon)
    button.setFlat(False)
    button.setCursor(Qt.PointingHandCursor)
    button.setFocusPolicy(Qt.StrongFocus)
    button.setStyleSheet(ui_scale.scale_stylesheet(style, ui_scale.percent()))
    if icon is not None:
        side = ui_scale.px(16)
        button.setIcon(_status_icon(*icon, size=side))
        button.setIconSize(QSize(side, side))
    current = button.text()
    width = 0
    for text in texts:
        button.setText(text)
        width = max(width, button.sizeHint().width())
    button.setText(current)
    button.setFixedWidth(width)
    return width


def _set_icon_action(button, label, tip):
    """Name an icon-only status-bar button and explain it.

    The label is the button's accessible name and the tooltip's first line,
    so the action is still named in words although the button shows none.

    Args: button is a QPushButton; label names what it does now (an action,
        or what is running); tip is the explanation.
    Returns: None.
    """
    button.setAccessibleName(label)
    set_tooltip(button, f"{label}: {tip}")


# Long enough that a moving pointer does not flash a tip; short enough to wait for.
TOOLTIP_DELAY_MS = 600
TOOLTIP_FALL_ASLEEP_MS = 0  # the next control waits too, so sweeping does not pop tips
TOOLTIP_WRAP = 52  # characters; keeps a tip about as wide as the controls dock
TOOLTIP_MARGIN = 2
TOOLTIP_CHROME = 18  # Qt's tooltip frame, so the estimated size is not smaller than the real one
SIMULATED_TIP = "Simulated in dummy mode"
SIMULATED_OBJECT_TIP = (
    "What the simulated drumhead does in dummy mode. The video is an ideal "
    "camera's view of an ESPI setup: speckle, a reference beam and an ideal "
    "sensor with no noise, so every processing step works on a known signal. "
    "Still: nothing moves; a still scene's differences are exactly zero. "
    "Slow deformation: the drum is pressed in to 8 fringes and released every "
    "12 s. Vibration: drum mode (1, 1) at 1000 Hz; set the reference on Still "
    "first. Follow DAQ: the NI DAQ window's generator drives the drum (Devices "
    "→ NI DAQ…, the Simulated device works). The DAQ piezo moves the reference "
    "mirror in every case. Only shown for demo cameras.")
SIMULATED_NOTE_TIP = "What the simulated drumhead is doing now, and what drives it."
SIMULATED_PIEZO_TIP = (
    "Ramp the simulated reference mirror with a slow triangle wave (1 Hz, 1.2 µm "
    "peak to peak), as a piezo on the reference beam does on the bench. The "
    "reference phase then changes between frames (about 90° at 30 fps), which "
    "decorrelates the speckle: in Previous view a vibration's still nodal lines "
    "turn bright and its J0 fringes dark, instead of a dark nodal line. Adds to "
    "the DAQ piezo. Off by default; demo cameras only.")
ZOOM_SLIDER_TIP = ("Drag, click or scroll to zoom. The mouse wheel and "
                   "drag-to-pan on the image still work.")
MODE_TIPS = {
    "Raw": ("Raw: the camera's Mono8 image, for aiming and focusing. "
            "Gamma applies; no difference or stretch."),
    "Previous": ("Previous: the difference from the last frame, so fringes show "
                 "what just changed. Shortcut: R restarts that difference."),
    "Reference": ("Reference: the difference from a stored frame. Shortcut: F "
                  "stores the live frame (or the average of Ref. frames frames). "
                  "Ref. refresh can recapture it on a timer so slow drift "
                  "cancels. C shows it beside the current frame."),
    "Phase": ("Phase: phase-shifting ESPI. The NI DAQ steps the reference "
              "piezo between frames and each pixel's phase change since the "
              "phase reference is shown with its sign (subtraction fringes "
              "have none): mid-grey is no change, black to white one fringe. "
              "Needs the piezo driver applied in Devices > NI DAQ; the camera "
              "then starts in this view. F takes a new phase reference."),
}


def format_tooltip(*parts):
    """Wrap tooltip paragraphs so the popup stays about as wide as the controls.

    Args: parts are paragraphs; blank ones are dropped. A newline inside a
        part starts a new paragraph, and a short part such as "Now: 8000 µs"
        stays on its own line.
    Returns: Plain text with newlines.
    """
    lines = []
    for part in parts:
        for paragraph in part.split("\n"):
            paragraph = paragraph.strip()
            if paragraph:
                lines.extend(textwrap.wrap(paragraph, width=TOOLTIP_WRAP))
    return "\n".join(lines)


def set_tooltip(target, *parts):
    """Explain a control. Disabled widgets still show the tip.

    Args: target is a QWidget or QAction; parts are paragraphs.
    Returns: None.
    """
    target.setToolTip(format_tooltip(*parts))
    if isinstance(target, QWidget):
        target.setAttribute(Qt.WA_AlwaysShowToolTips, True)


class TooltipDelayStyle(QProxyStyle):
    """Pause before a tooltip, including when the pointer moves to the next control."""

    def styleHint(self, hint, option=None, widget=None, returnData=None):
        """Return the hover delays for tooltips and delegate every other hint.

        Args: hint is a QStyle.StyleHint; option, widget and returnData are
            the Qt arguments for this hint.
        Returns: Milliseconds, or the base style's answer.
        """
        if hint == QStyle.SH_ToolTip_WakeUpDelay:
            return TOOLTIP_DELAY_MS
        if hint == QStyle.SH_ToolTip_FallAsleepDelay:
            return TOOLTIP_FALL_ASLEEP_MS
        return super().styleHint(hint, option, widget, returnData)


# The style's own pixel sizes that grow with the interface size: layout margins and
# spacing, check and radio indicators, slider handles, scroll bars, icons, docks and
# toolbars. Frame and border widths are left alone, so lines stay crisp.
SCALED_METRICS = frozenset((
    QStyle.PM_LayoutLeftMargin, QStyle.PM_LayoutTopMargin, QStyle.PM_LayoutRightMargin,
    QStyle.PM_LayoutBottomMargin, QStyle.PM_LayoutHorizontalSpacing,
    QStyle.PM_LayoutVerticalSpacing, QStyle.PM_ButtonMargin,
    QStyle.PM_IndicatorWidth, QStyle.PM_IndicatorHeight,
    QStyle.PM_ExclusiveIndicatorWidth, QStyle.PM_ExclusiveIndicatorHeight,
    QStyle.PM_SliderThickness, QStyle.PM_SliderLength, QStyle.PM_SliderControlThickness,
    QStyle.PM_ScrollBarExtent, QStyle.PM_ScrollBarSliderMin,
    QStyle.PM_SmallIconSize, QStyle.PM_ButtonIconSize, QStyle.PM_ToolBarIconSize,
    QStyle.PM_LargeIconSize, QStyle.PM_DockWidgetSeparatorExtent,
    QStyle.PM_DockWidgetTitleMargin, QStyle.PM_DockWidgetTitleBarButtonMargin,
    QStyle.PM_SplitterWidth, QStyle.PM_ToolBarHandleExtent,
    QStyle.PM_ToolBarSeparatorExtent, QStyle.PM_ToolBarItemSpacing,
    QStyle.PM_ToolBarItemMargin))


class ScaledStyle(TooltipDelayStyle):
    """The application style: the tooltip delay, and chrome sized by the interface size.

    The metrics in SCALED_METRICS are multiplied by ``ui_scale.factor()``, read at
    every call, so check boxes, slider handles, icons, margins and docks follow the
    font when the interface size changes.
    """

    def pixelMetric(self, metric, option=None, widget=None):
        """Return a pixel size, scaled for the metrics in SCALED_METRICS.

        Args: metric is a QStyle.PixelMetric; option and widget are the Qt
            arguments for this metric.
        Returns: The size in pixels (negative "use the default" values pass through).
        """
        size = super().pixelMetric(metric, option, widget)
        if metric in SCALED_METRICS and size > 0:
            return ui_scale.scaled(size, ui_scale.percent())
        return size


def install_tooltip_delay():
    """Install the application style (tooltip delay, scaled chrome) once for this process.

    Args: None. A QApplication must already exist.
    Returns: None.
    """
    app = QApplication.instance()
    if isinstance(app.style(), TooltipDelayStyle):
        return
    app.setStyle(ScaledStyle())


_base_font = None  # the application font before any scaling, taken on first use
_font_scaled = False  # True while the application font has been replaced by a scaled one


def base_font():
    """Return the application font as it was before the interface size changed it.

    Args: None. A QApplication must already exist.
    Returns: A QFont (a copy).
    """
    global _base_font
    if _base_font is None:
        _base_font = QFont(QApplication.font())
    return QFont(_base_font)


def scaled_font(font, percent):
    """Scale a font's size.

    Args: font is the QFont at 100%; percent is the interface size.
    Returns: A new QFont, sized in points when the original is, else in pixels.
    """
    result = QFont(font)
    if font.pointSizeF() > 0:
        result.setPointSizeF(font.pointSizeF() * percent / 100)
    else:
        result.setPixelSize(ui_scale.scaled(font.pixelSize(), percent))
    return result


def nudged_font(font):
    """Return a font that looks the same but differs from the given one.

    Qt draws a dock's title bar with a stale font after the application font
    changes, unless the dock holds an explicit font that is not equal to the
    application's. A hundredth of a point (or one pixel, for pixel-sized fonts)
    is invisible and forces the title to be drawn in the right size.

    Args: font is a QFont, normally the application font.
    Returns: A new QFont.
    """
    result = QFont(font)
    if font.pointSizeF() > 0:
        result.setPointSizeF(font.pointSizeF() + 0.01)
    else:
        result.setPixelSize(font.pixelSize() + 1)
    return result


def apply_app_scale(percent):
    """Make the whole application follow an interface size, without restarting.

    Sets the shared value, the application font (every widget, menu, dialog and
    tooltip that has no font of its own size follows it), and replaces the
    application style so Qt re-polishes every widget with the new metrics. When it
    returns every widget already has the new font. The caller still re-applies the
    pixel sizes it fixed itself.

    Args: percent is the wanted interface size (clamped to the allowed range).
    Returns: The percentage now in force.
    """
    global _font_scaled
    font = base_font()  # before anything changes
    changed = ui_scale.percent() != ui_scale.clamp_percent(percent)
    percent = ui_scale.set_percent(percent)
    # At 100% the platform's own fonts stay in force (setFont would replace its
    # per-widget-class ones), unless a larger or smaller size has to be undone.
    if percent != ui_scale.DEFAULT_PERCENT or _font_scaled:
        QApplication.setFont(scaled_font(font, percent))
        _font_scaled = percent != ui_scale.DEFAULT_PERCENT
    if changed:
        QApplication.setStyle(ScaledStyle())  # Qt polishes every widget again
    else:
        install_tooltip_delay()
    # Qt hands the new font to existing widgets through posted events; deliver
    # them now, so the caller measures widgets that already have it.
    QCoreApplication.sendPostedEvents()
    return percent


def tooltip_size(text):
    """Estimate the rectangle Qt will give a tooltip.

    Args: text is the tooltip, which may contain newlines.
    Returns: A QSize in pixels.
    """
    metrics = QFontMetrics(QToolTip.font())
    lines = text.split("\n")
    width = max(metrics.horizontalAdvance(line) for line in lines) + TOOLTIP_CHROME
    height = metrics.lineSpacing() * len(lines) + TOOLTIP_CHROME
    return QSize(width, height)


def image_global_rect(image):
    """Return the live image's global rectangle, or None when it is not shown.

    Args: image is the image view, or None.
    Returns: A QRect, or None.
    """
    if image is None or not image.isVisible():
        return None
    return QRect(image.mapToGlobal(QPoint(0, 0)), image.size())


def _screen_rect(widget, screen):
    """Choose the desktop rectangle a tooltip must stay inside.

    Args: widget is the control; screen is an explicit QRect, or None to use
        the widget's screen.
    Returns: A QRect.
    """
    if screen is not None:
        return screen
    widget_screen = widget.screen()
    if widget_screen is None:
        return QRect(0, 0, 4096, 4096)
    return widget_screen.availableGeometry()


def fit_rect(rect, available):
    """Shrink and move a window rectangle so it lies inside the available area.

    Args: rect is the window's outer rectangle (title bar and borders
        included); available is the screen's usable area (no taskbar or dock).
    Returns: A QRect inside available, or rect itself when it already fits.
    """
    width = min(rect.width(), available.width())
    height = min(rect.height(), available.height())
    x = min(max(rect.x(), available.left()), available.right() + 1 - width)
    y = min(max(rect.y(), available.top()), available.bottom() + 1 - height)
    return QRect(x, y, width, height)


def _candidate_points(origin, widget_size, tip_size, blocked):
    """List tooltip positions, nearest to the control first.

    When the control sits on the image, later positions are outside that image.

    Args: origin is the control's global top-left; widget_size is the
        control's size; tip_size is the tooltip size; blocked is the image
        QRect, or None.
    Returns: A list of QPoint.
    """
    points = [
        QPoint(origin.x(), origin.y() + widget_size.height() + TOOLTIP_MARGIN),
        QPoint(origin.x(), origin.y() - tip_size.height() - TOOLTIP_MARGIN),
        QPoint(origin.x() + widget_size.width() + TOOLTIP_MARGIN, origin.y()),
        QPoint(origin.x() - tip_size.width() - TOOLTIP_MARGIN, origin.y()),
    ]
    if blocked is not None:
        points.extend((
            QPoint(blocked.left(), blocked.bottom() + TOOLTIP_MARGIN),
            QPoint(blocked.left(), blocked.top() - tip_size.height() - TOOLTIP_MARGIN),
            QPoint(blocked.right() + TOOLTIP_MARGIN, blocked.top()),
            QPoint(blocked.left() - tip_size.width() - TOOLTIP_MARGIN, blocked.top()),
        ))
    return points


def _fits(pos, size, screen, blocked):
    """Report whether a tooltip is fully on screen and off the live image.

    Args: pos is the global top-left; size is the tooltip size; screen is
        the available QRect; blocked is the image QRect, or None.
    Returns: True when the rectangle can be used.
    """
    rect = QRect(pos, size)
    if not screen.contains(rect):
        return False
    return blocked is None or not blocked.intersects(rect)


def _clamp_point(pos, size, screen):
    """Move a tooltip so it lies inside the screen.

    Args: pos is the preferred global top-left; size is the tooltip size;
        screen is the available QRect.
    Returns: A QPoint.
    """
    x = min(max(pos.x(), screen.left()), max(screen.left(), screen.right() - size.width() + 1))
    y = min(max(pos.y(), screen.top()), max(screen.top(), screen.bottom() - size.height() + 1))
    return QPoint(x, y)


def _nudge_on_screen(pos, size, screen, blocked):
    """Slide a tip onto the screen without moving it onto the image.

    Args: pos is the preferred global top-left; size is the tooltip size;
        screen is the available QRect; blocked is the image QRect, or None.
    Returns: A QPoint, or None when the slide would cover the image or the
        tip still cannot fit.
    """
    if blocked is not None and blocked.intersects(QRect(pos, size)):
        return None
    nudged = _clamp_point(pos, size, screen)
    rect = QRect(nudged, size)
    if not screen.contains(rect):
        return None
    if blocked is not None and blocked.intersects(rect):
        return None
    return nudged


def tooltip_position(widget, text, image=None, screen=None):
    """Choose where a tooltip opens so it does not cover the live image.

    The image view itself never takes a tooltip: a popup there would hide the
    measurement. Qt still waits TOOLTIP_DELAY_MS before asking for a tip, and
    the tip is not modal, so it does not block the controls.

    Args: widget is the control under the pointer; text is the tip; image is
        the image view to keep clear, or None; screen is the available
        desktop rectangle, or None to use the widget's screen.
    Returns: A QPoint in global coordinates.
    """
    size = tooltip_size(text)
    screen = _screen_rect(widget, screen)
    origin = widget.mapToGlobal(QPoint(0, 0))
    blocked = image_global_rect(image)
    points = _candidate_points(origin, widget.size(), size, blocked)
    for pos in points:
        if _fits(pos, size, screen, blocked):
            return pos
    for pos in points:
        nudged = _nudge_on_screen(pos, size, screen, blocked)
        if nudged is not None:
            return nudged
    # Covering the measurement is worse than a tip that runs off the screen.
    if blocked is not None:
        for pos in points:
            if not QRect(pos, size).intersects(blocked):
                return pos
    return _clamp_point(points[0], size, screen)


def request_gl_format():
    """Ask Qt for an OpenGL 3.3 core context (macOS supplies 4.1 core).

    Args: None. Must run before QApplication is created.
    Returns: None.
    """
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)


def opengl_available():
    """Check that an OpenGL 3.3+ core context can be created and used.

    Args: None. Requires a QApplication.
    Returns: (True, description) when usable, else (False, reason).
    """
    context = QOpenGLContext()
    if not context.create():
        return False, "no OpenGL context"
    surface = QOffscreenSurface()
    surface.setFormat(context.format())
    surface.create()
    if not context.makeCurrent(surface):
        return False, "OpenGL context could not be made current"
    try:
        fmt = context.format()
        version = (fmt.majorVersion(), fmt.minorVersion())
        functions = context.functions()
        renderer = functions.glGetString(0x1F01) or "unknown renderer"
        if version < (3, 3):
            return False, f"OpenGL {version[0]}.{version[1]} on {renderer} (3.3 needed)"
        return True, f"OpenGL {version[0]}.{version[1]} on {renderer}"
    finally:
        context.doneCurrent()


class ViewNavigation:
    """Fit, zoom and pan shared by the GPU and CPU image views.

    zoom is relative to fit-to-window (1.0); center is the image point shown
    at the widget center, or None to center the image.
    """

    def init_navigation(self):
        """Reset navigation state; call from the widget's __init__.

        Args: None.
        Returns: None.
        """
        self.zoom = 1.0
        self.center = None
        self.image_size = None  # (width, height)
        self.drag_origin = None
        self.press_point = None  # where the left button went down, for clicks
        self.box = None  # the outlined pixel square (left, top, right, bottom), or None
        self.setMouseTracking(True)

    def display_rect(self):
        """Compute where the image is drawn in widget coordinates.

        Args: None.
        Returns: (x, y, width, height, scale) in logical pixels, or None.
        """
        if self.image_size is None:
            return None
        image_w, image_h = self.image_size
        widget_w, widget_h = max(1, self.width()), max(1, self.height())
        scale = min(widget_w / image_w, widget_h / image_h) * self.zoom
        center_x, center_y = self.center or (image_w / 2, image_h / 2)
        x = widget_w / 2 - center_x * scale
        y = widget_h / 2 - center_y * scale
        return x, y, image_w * scale, image_h * scale, scale

    def widget_to_image(self, pos):
        """Map a widget position to image pixel coordinates.

        Args: pos is a QPointF in widget coordinates.
        Returns: (x, y) floats in image pixels, or None without an image.
        """
        rect = self.display_rect()
        if rect is None:
            return None
        x, y, _, _, scale = rect
        return (pos.x() - x) / scale, (pos.y() - y) / scale

    def set_zoom(self, zoom, anchor=None):
        """Zoom while keeping the image point under anchor fixed.

        Args: zoom is relative to fit (clamped to 1..MAX_ZOOM); anchor is a
            widget QPointF, or None for the widget center.
        Returns: None.
        """
        if self.image_size is None:
            return
        zoom = max(1.0, min(MAX_ZOOM, zoom))
        if anchor is None:
            anchor = QPointF(self.width() / 2, self.height() / 2)
        image_point = self.widget_to_image(anchor)
        self.zoom = zoom
        if zoom == 1.0:
            self.center = None
        else:
            scale = self.display_rect()[4]
            self.center = (image_point[0] - (anchor.x() - self.width() / 2) / scale,
                           image_point[1] - (anchor.y() - self.height() / 2) / scale)
            self._clamp_center()
        self.update()
        self.zoomChanged.emit(self.zoom)

    def fit(self):
        """Show the whole image.

        Args: None.
        Returns: None.
        """
        self.zoom, self.center = 1.0, None
        self.update()
        self.zoomChanged.emit(self.zoom)

    def actual_size(self):
        """Show one image pixel per screen pixel.

        Args: None.
        Returns: None.
        """
        if self.image_size is None:
            return
        image_w, image_h = self.image_size
        fit_scale = min(self.width() / image_w, self.height() / image_h)
        ratio = self.devicePixelRatioF()
        self.set_zoom(1.0 / (fit_scale * ratio))

    def _clamp_center(self):
        """Keep the view center inside the image.

        Args: None.
        Returns: None.
        """
        image_w, image_h = self.image_size
        x, y = self.center
        self.center = (min(max(x, 0.0), image_w), min(max(y, 0.0), image_h))

    def wheelEvent(self, event):
        """Zoom around the cursor with the mouse wheel or trackpad.

        Args: event is the Qt wheel event.
        Returns: None.
        """
        steps = event.angleDelta().y() / 120.0
        if steps:
            self.set_zoom(self.zoom * (1.25 ** steps), event.position())
        event.accept()

    def mousePressEvent(self, event):
        """Start panning with the left button.

        Args: event is the Qt mouse event.
        Returns: None.
        """
        if event.button() == Qt.LeftButton:
            self.press_point = event.position()
        if event.button() == Qt.LeftButton and self.zoom > 1.0:
            self.drag_origin = event.position()
            self.setCursor(Qt.ClosedHandCursor)
        self.setFocus()

    def mouseMoveEvent(self, event):
        """Pan while dragging and report the pixel under the cursor.

        Args: event is the Qt mouse event.
        Returns: None.
        """
        if self.drag_origin is not None and self.center is not None:
            scale = self.display_rect()[4]
            delta = event.position() - self.drag_origin
            self.drag_origin = event.position()
            self.center = (self.center[0] - delta.x() / scale,
                           self.center[1] - delta.y() / scale)
            self._clamp_center()
            self.update()
        point = self.widget_to_image(event.position())
        if point is not None:
            self.hovered.emit(int(point[0]), int(point[1]))

    def mouseReleaseEvent(self, event):
        """Finish panning, or report a click on the image.

        A left press and release that hardly moved is a click: imageClicked
        reports the image pixel under it (when there is one).

        Args: event is the Qt mouse event.
        Returns: None.
        """
        press, self.press_point = self.press_point, None
        if press is not None and event.button() == Qt.LeftButton and \
                (event.position() - press).manhattanLength() < CLICK_SLOP:
            point = self.widget_to_image(event.position())
            if point is not None and 0 <= point[0] < self.image_size[0] and \
                    0 <= point[1] < self.image_size[1]:
                self.imageClicked.emit(int(point[0]), int(point[1]))
        self.drag_origin = None
        self.unsetCursor()

    def set_box(self, box):
        """Outline a square on the image, or none.

        Args: box is (left, top, right, bottom) in image pixels, right and
            bottom exclusive, or None.
        Returns: None.
        """
        if box != self.box:
            self.box = box
            self.update()

    def paint_box(self, painter):
        """Draw the outline over the image.

        Args: painter is an active QPainter on this widget.
        Returns: None.
        """
        rect = self.display_rect()
        if self.box is None or rect is None:
            return
        x, y, _, _, scale = rect
        left, top, right, bottom = self.box
        area = QRectF(x + left * scale, y + top * scale,
                      (right - left) * scale, (bottom - top) * scale)
        if area.width() < BOX_MIN_SIZE:
            area = QRectF(area.center().x() - BOX_MIN_SIZE / 2,
                          area.center().y() - BOX_MIN_SIZE / 2,
                          BOX_MIN_SIZE, BOX_MIN_SIZE)
        painter.setPen(QPen(BOX_EDGE, 3))
        painter.drawRect(area)
        painter.setPen(QPen(BOX_COLOR, 1))
        painter.drawRect(area)

    def leaveEvent(self, event):
        """Clear the pixel readout when the cursor leaves the image.

        Args: event is the Qt leave event.
        Returns: None.
        """
        self.hovered.emit(-1, -1)


class GLImageView(ViewNavigation, QOpenGLWidget):
    """Draw Mono8 frames with GPU scaling and gamma."""

    hovered = Signal(int, int)
    zoomChanged = Signal(float)  # zoom relative to fit-to-window
    imageClicked = Signal(int, int)  # the image pixel clicked (not dragged)

    def __init__(self, parent=None):
        """Create the view; GL resources are made in initializeGL.

        Args: parent is the owning widget.
        Returns: None.
        """
        super().__init__(parent)
        self.init_navigation()
        self.pending = None  # (packet, rgb_table, table_key)
        self.texture_shape = None
        self.table_key = None
        self.program = None
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)

    def show_packet(self, packet, table, table_key):
        """Queue a frame and its color table for the next paint.

        Args: packet is a FramePacket; table is a (256, 3) RGB array; table_key
            identifies the table so unchanged tables are not re-uploaded.
        Returns: None.
        """
        self.pending = (packet, table, table_key)
        height, width = packet.image.shape
        if self.image_size != (width, height):
            self.image_size = (width, height)
            self.zoom, self.center = 1.0, None
            self.zoomChanged.emit(self.zoom)
        self.update()

    def clear(self):
        """Show no image until the next frame (used when the camera changes).

        Args: None.
        Returns: None.
        """
        self.pending = None
        self.texture_shape = None
        self.image_size = None
        self.update()

    def initializeGL(self):
        """Compile the shaders and create the image and lookup-table textures.

        Args: None.
        Returns: None. Raises RuntimeError if the shaders cannot be built.
        """
        gl = self.context().functions()
        self.program = QOpenGLShaderProgram(self)
        if not (self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, VERTEX_SHADER)
                and self.program.addShaderFromSourceCode(QOpenGLShader.Fragment,
                                                         FRAGMENT_SHADER)
                and self.program.link()):
            raise RuntimeError(f"Shader build failed: {self.program.log()}")
        self.vao = QOpenGLVertexArrayObject(self)
        self.vao.create()
        names = np.zeros(2, dtype=np.uint32)
        gl.glGenTextures(2, names)
        self.image_texture, self.lut_texture = int(names[0]), int(names[1])
        for texture, minify, magnify in (
                (self.image_texture, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST),
                (self.lut_texture, GL_NEAREST, GL_NEAREST)):
            gl.glBindTexture(GL_TEXTURE_2D, texture)
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minify)
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magnify)
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        self.texture_shape = None
        self.table_key = None
        self.context().aboutToBeDestroyed.connect(self._release_gl)

    def _release_gl(self):
        """Delete GL textures before the context goes away.

        Args: None.
        Returns: None.
        """
        self.makeCurrent()
        names = np.array([self.image_texture, self.lut_texture], dtype=np.uint32)
        self.context().functions().glDeleteTextures(2, names)
        self.doneCurrent()

    def _upload(self, gl):
        """Send the pending frame and (if changed) color table to the GPU.

        Args: gl is the context's QOpenGLFunctions.
        Returns: None.
        """
        packet, table, table_key = self.pending
        self.pending = None
        image = np.ascontiguousarray(packet.image)
        height, width = image.shape
        gl.glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        gl.glBindTexture(GL_TEXTURE_2D, self.image_texture)
        if self.texture_shape != (height, width):
            gl.glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0,
                            GL_RED, GL_UNSIGNED_BYTE, image.tobytes())
            self.texture_shape = (height, width)
        else:
            gl.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                               GL_RED, GL_UNSIGNED_BYTE, memoryview(image))
        gl.glGenerateMipmap(GL_TEXTURE_2D)
        if table_key != self.table_key:
            rgb = np.ascontiguousarray(table, dtype=np.uint8)
            gl.glBindTexture(GL_TEXTURE_2D, self.lut_texture)
            gl.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 256, 1, 0,
                            GL_RGB, GL_UNSIGNED_BYTE, rgb.tobytes())
            self.table_key = table_key

    def paintGL(self):
        """Draw the newest frame centered, fitted, or zoomed.

        Args: None.
        Returns: None.
        """
        gl = self.context().functions()
        background = self.palette().color(self.backgroundRole())
        gl.glClearColor(background.redF() * 0.5, background.greenF() * 0.5,
                        background.blueF() * 0.5, 1.0)
        gl.glClear(GL_COLOR_BUFFER_BIT)
        if self.pending is not None:
            self._upload(gl)
        rect = self.display_rect()
        if rect is None or self.texture_shape is None:
            return
        x, y, width, height, _ = rect
        widget_w, widget_h = max(1, self.width()), max(1, self.height())
        self.program.bind()
        self.vao.bind()
        self.program.setUniformValue(self.program.uniformLocation("rect"), QVector4D(
            2 * x / widget_w - 1, 1 - 2 * y / widget_h,
            2 * (x + width) / widget_w - 1, 1 - 2 * (y + height) / widget_h))
        gl.glActiveTexture(GL_TEXTURE0)
        gl.glBindTexture(GL_TEXTURE_2D, self.image_texture)
        gl.glActiveTexture(GL_TEXTURE0 + 1)
        gl.glBindTexture(GL_TEXTURE_2D, self.lut_texture)
        gl.glActiveTexture(GL_TEXTURE0)
        self.program.setUniformValue1i(self.program.uniformLocation("image"), 0)
        self.program.setUniformValue1i(self.program.uniformLocation("lut"), 1)
        gl.glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        self.vao.release()
        self.program.release()
        if self.box is not None:
            painter = QPainter(self)  # after the GL drawing, on top of it
            self.paint_box(painter)
            painter.end()


class CpuImageView(ViewNavigation, QWidget):
    """Fallback view: resize and colorize on the CPU, draw with QPainter."""

    hovered = Signal(int, int)
    zoomChanged = Signal(float)  # zoom relative to fit-to-window
    imageClicked = Signal(int, int)  # the image pixel clicked (not dragged)

    def __init__(self, parent=None):
        """Create the fallback view.

        Args: parent is the owning widget.
        Returns: None.
        """
        super().__init__(parent)
        self.init_navigation()
        self.packet = None
        self.table = None
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_OpaquePaintEvent)

    def show_packet(self, packet, table, table_key):
        """Keep the newest frame and repaint.

        Args: packet is a FramePacket; table is a (256, 3) RGB array;
            table_key is unused here.
        Returns: None.
        """
        self.packet, self.table = packet, table
        height, width = packet.image.shape
        if self.image_size != (width, height):
            self.image_size = (width, height)
            self.zoom, self.center = 1.0, None
            self.zoomChanged.emit(self.zoom)
        self.update()

    def clear(self):
        """Show no image until the next frame (used when the camera changes).

        Args: None.
        Returns: None.
        """
        self.packet = None
        self.image_size = None
        self.update()

    def paintEvent(self, event):
        """Crop to the visible area, resize, apply the color table and draw.

        Args: event is the Qt paint event.
        Returns: None.
        """
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().color(self.backgroundRole()).darker(200))
        rect = self.display_rect()
        if rect is None or self.packet is None:
            return
        x, y, width, height, scale = rect
        ratio = self.devicePixelRatioF()
        image = self.packet.image
        # Crop to what is visible so zooming in stays cheap.
        left = max(0, int(-x / scale))
        top = max(0, int(-y / scale))
        right = min(image.shape[1], int((self.width() - x) / scale) + 2)
        bottom = min(image.shape[0], int((self.height() - y) / scale) + 2)
        if right <= left or bottom <= top:
            return
        crop = image[top:bottom, left:right]
        target_w = max(1, round((right - left) * scale * ratio))
        target_h = max(1, round((bottom - top) * scale * ratio))
        shrinking = target_w < crop.shape[1]
        resized = cv2.resize(crop, (target_w, target_h), interpolation=(
            cv2.INTER_AREA if shrinking else cv2.INTER_NEAREST))
        rgb = np.ascontiguousarray(self.table[resized])
        qimage = QImage(rgb.data, target_w, target_h, 3 * target_w,
                        QImage.Format_RGB888)
        qimage.setDevicePixelRatio(ratio)
        painter.drawImage(QPointF(x + left * scale, y + top * scale), qimage)
        self.paint_box(painter)


class WheelIgnoringSlider(QSlider):
    """A slider that leaves the mouse wheel to the widgets around it.

    A wheel notch over the slider is ignored, so it reaches the parent and a
    scrolling panel scrolls instead of the value changing. Dragging, clicking
    and the keyboard still set the value.
    """

    def wheelEvent(self, event):
        """Ignore the wheel so the parent widget gets it.

        Args: event is the Qt wheel event.
        Returns: None.
        """
        event.ignore()


class WheelIgnoringSpinBox(QDoubleSpinBox):
    """A number box that leaves the mouse wheel to the widgets around it.

    It also takes focus only by a click or Tab: Qt's default is to take focus
    on a wheel event, which would move the keyboard focus while scrolling.
    """

    def __init__(self):
        """Create the number box with click-and-Tab focus only.

        Args: None.
        Returns: None.
        """
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        """Ignore the wheel so the parent widget gets it.

        Args: event is the Qt wheel event.
        Returns: None.
        """
        event.ignore()


class WheelIgnoringComboBox(QComboBox):
    """A drop-down that leaves the mouse wheel to the widgets around it.

    Qt's default steps to the next entry on a wheel notch over a closed
    drop-down, which silently changes a setting while the Controls panel is
    being scrolled. The wheel is ignored here so the panel scrolls instead; a
    click, the keyboard and the open list still choose an entry. Like the
    number box it takes focus only by a click or Tab, not from a wheel.
    """

    def __init__(self):
        """Create the drop-down with click-and-Tab focus only.

        Args: None.
        Returns: None.
        """
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        """Ignore the wheel so the parent widget gets it.

        Args: event is the Qt wheel event.
        Returns: None.
        """
        event.ignore()


class SliderSpin(QWidget):
    """A slider and exact-entry spin box sharing one numeric value."""

    valueChanged = Signal(float)

    def __init__(self, low, high, step, decimals, value, suffix=""):
        """Build the paired controls.

        Args: low/high bound the value; step is the increment; decimals is
            the spin-box precision; value is the initial value; suffix is a
            unit label.
        Returns: None.
        """
        super().__init__()
        self.step = step
        self._description = ""
        self._note = ""
        self.slider = WheelIgnoringSlider(Qt.Horizontal)
        self.spin = WheelIgnoringSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setSuffix(suffix)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(ui_scale.px(104))  # equal widths keep value columns aligned
        for widget in (self, self.slider, self.spin):
            widget.setAttribute(Qt.WA_AlwaysShowToolTips, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        self.set_range(low, high)
        self.set_value(value)
        self.slider.valueChanged.connect(
            lambda ticks: self.spin.setValue(ticks * self.step))
        self.spin.valueChanged.connect(self._spin_changed)

    def rescale(self):
        """Resize the spin box for the interface size now in force.

        Args: None.
        Returns: None.
        """
        self.spin.setFixedWidth(ui_scale.px(104))

    def set_range(self, low, high):
        """Change the allowed range, clamping the current value.

        Args: low and high are the new bounds.
        Returns: None.
        """
        self.spin.setRange(low, high)
        self.slider.setRange(round(low / self.step), round(high / self.step))

    def set_value(self, value):
        """Set the value without emitting valueChanged.

        Args: value is the new number (clamped to the range).
        Returns: None.
        """
        self.spin.blockSignals(True)
        self.slider.blockSignals(True)
        self.spin.setValue(value)
        self.slider.setValue(round(self.spin.value() / self.step))
        self.spin.blockSignals(False)
        self.slider.blockSignals(False)
        self._apply_tooltip()

    def set_description(self, text):
        """Say what this control does. The live value is added to the tooltip.

        Args: text explains the control, including a typical value where one helps.
        Returns: None.
        """
        self._description = text
        self._apply_tooltip()

    def set_note(self, text):
        """Add a situation line, such as dummy mode or an analog camera.

        Args: text is the extra line, or "" to remove it.
        Returns: None.
        """
        self._note = text
        self._apply_tooltip()

    def _value_text(self):
        """Format the current value with its unit.

        Args: None.
        Returns: A string such as "8000 µs" or "1.0".
        """
        value = self.spin.value()
        if self.spin.decimals() == 0:
            number = str(int(round(value)))
        else:
            number = f"{value:.{self.spin.decimals()}f}"
        return f"{number}{self.spin.suffix()}"

    def _apply_tooltip(self):
        """Rebuild the tooltip from the description, the live value and any note.

        Args: None.
        Returns: None.
        """
        parts = []
        if self._description:
            parts.append(self._description)
            parts.append(f"Now: {self._value_text()}")
        if self._note:
            parts.append(self._note)
        text = format_tooltip(*parts)
        super().setToolTip(text)
        self.slider.setToolTip(text)
        self.spin.setToolTip(text)

    def value(self):
        """Return the current value.

        Args: None.
        Returns: The spin box's float value.
        """
        return self.spin.value()

    def _spin_changed(self, value):
        """Mirror a spin-box change on the slider and announce it.

        Args: value is the new spin-box value.
        Returns: None.
        """
        self.slider.blockSignals(True)
        self.slider.setValue(round(value / self.step))
        self.slider.blockSignals(False)
        self._apply_tooltip()
        self.valueChanged.emit(value)


class ShortcutFilter(QObject):
    """Route single-key shortcuts unless the user is typing in a field."""

    def __init__(self, window):
        """Attach to a main window.

        Args: window is the MainWindow whose handle_key receives keys.
        Returns: None.
        """
        super().__init__(window)
        self.window = window

    def eventFilter(self, watched, event):
        """Intercept key presses for the main window.

        Args: watched is the receiving object; event is the Qt event.
        Returns: True when the key was handled as a shortcut.
        """
        if event.type() != QEvent.KeyPress or event.modifiers() & (
                Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier):
            return False
        focus = QApplication.focusWidget()
        if focus is not None and focus.window() is not self.window:
            return False
        if isinstance(focus, TEXT_INPUTS):
            if event.key() == Qt.Key_Escape:
                focus.clearFocus()  # Esc leaves a text field instead of quitting
                return True
            return False
        return self.window.handle_key(event.key())


class TooltipFilter(QObject):
    """Open control tooltips beside the control, never over the live image."""

    def __init__(self, window):
        """Attach to a main window.

        Args: window is the MainWindow whose image must stay visible.
        Returns: None.
        """
        super().__init__(window)
        self.window = window

    def eventFilter(self, watched, event):
        """Place a tooltip off the image. The image itself gets none.

        Qt sends this event only after the hover delay, and the tip is not
        modal, so it does not block the controls under it.

        Args: watched is the object under the pointer; event is the Qt event.
        Returns: True when this filter handled the event.
        """
        if event.type() != QEvent.ToolTip or not isinstance(watched, QWidget):
            return False
        if watched.window() is not self.window:
            return False
        if watched is self.window.view:
            QToolTip.hideText()
            return True
        text = watched.toolTip()
        if not text:
            return False
        QToolTip.showText(tooltip_position(watched, text, self.window.view), text, watched)
        return True


class FeatureRequestDialog(QDialog):
    """Collect a feature request: title, description and name."""

    def __init__(self, parent, request=None):
        """Build the form, prefilled from an earlier attempt or saved name.

        Args: parent is the main window; request is an optional
            FeatureRequest whose text is restored (after a failed send).
        Returns: None.
        """
        super().__init__(parent)
        self.setWindowTitle("Request a feature")
        self.resize(ui_scale.px(560), ui_scale.px(420))
        settings = QSettings("ESPI", "espi_live")
        self.title = QLineEdit(request.title if request else "")
        self.title.setMaxLength(feedback.MAX_TITLE)
        self.title.setPlaceholderText("Short summary, e.g. Export the reference frame")
        set_tooltip(self.title, "Short summary of the change you want. Required.")
        self.description = QPlainTextEdit(request.description if request else "")
        self.description.setPlaceholderText(
            "What would you like ESPI live to do, and why? Leave this blank "
            "to have it written from the title.")
        set_tooltip(self.description,
                    "What ESPI live should do. Optional. A blank description is "
                    "written up from the title. Text you write is kept and expanded.")
        self.submitter = QLineEdit(request.submitter if request else
                                   settings.value("feedback/name", "", str))
        self.submitter.setPlaceholderText("Optional, so we can follow up")
        set_tooltip(self.submitter,
                    "Optional name so the developers can follow up. Remembered on "
                    "this computer.")
        form = QFormLayout()
        form.addRow("Title", self.title)
        form.addRow("Description", self.description)
        form.addRow("Your name", self.submitter)
        note = QLabel(f"Filed as a GitHub issue in {feedback.repository()}, with "
                      "this build's version and platform. A blank description is "
                      "written from the title; text you write is kept and expanded.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        send = buttons.button(QDialogButtonBox.Ok)
        send.setText("Submit")
        set_tooltip(send, "File this request as a GitHub issue.")
        set_tooltip(buttons.button(QDialogButtonBox.Cancel), "Close without submitting.")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form, 1)
        layout.addWidget(note)
        layout.addWidget(buttons)

    def request(self, source):
        """Read the form.

        Args: source describes the video source ("camera" or "dummy mode").
        Returns: A FeatureRequest.
        """
        return feedback.FeatureRequest(
            title=self.title.text(), description=self.description.toPlainText(),
            submitter=self.submitter.text(), source=source)

    def accept(self):
        """Close only when the request has a title; remember the name.

        Args: None.
        Returns: None.
        """
        try:
            feedback.validate(self.request(""))
        except feedback.FeedbackError as exc:
            QMessageBox.warning(self, "Request a feature", f"Please {exc}.")
            return
        settings = QSettings("ESPI", "espi_live")
        settings.setValue("feedback/name", self.submitter.text().strip())
        super().accept()


class SettingsGuideDialog(QDialog):
    """Explain the processing controls and offer the starting presets."""

    def __init__(self, parent):
        """Build the guide text and the buttons that apply a preset.

        Args: parent is the main window; its apply_preset handles the buttons.
        Returns: None.
        """
        super().__init__(parent)
        self.setWindowTitle("Which settings should I use?")
        self.resize(ui_scale.px(680), ui_scale.px(640))
        self.guide = QPlainTextEdit(SETTINGS_GUIDE)
        self.guide.setReadOnly(True)
        self.guide.setObjectName("settingsGuide")
        buttons = QDialogButtonBox()
        for name in PROCESSING_PRESETS:
            apply = buttons.addButton(f"Apply {name}", QDialogButtonBox.ActionRole)
            apply.clicked.connect(
                lambda _checked=False, name=name: parent.apply_preset(name))
        close = buttons.addButton(QDialogButtonBox.Close)
        close.clicked.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.guide, 1)
        layout.addWidget(buttons)


# The three images of the comparison window: packet attribute, heading and
# tooltip. The first is the baseline (the reference, or in Previous view the
# earlier frame); COMPARE_BASELINES names it per view.
COMPARE_PANELS = (
    ("baseline", "Reference", ""),
    ("current", "Current frame",
     "The camera frame being compared, after single field and noise reduction, "
     "the same point the baseline is taken at. Gamma only."),
    ("difference", "Difference",
     "The processed difference between the two. In Reference and Previous views "
     "it is exactly the main window's image, averaging included; in Raw view it "
     "is one difference from the reference with the same filters, without "
     "averaging."),
)
# Per view: the baseline panel's tooltip, the text when there is none, and the
# Set reference button's tooltip.
COMPARE_BASELINES = {
    "Reference": (
        "The stored reference frame, as Raw view would show it: gamma only, no "
        "stretch. F or Set reference retakes it; Ref. frames averages "
        "several frames into it.",
        "No reference yet.\n\nPress Set reference (F) to capture one; "
        "Ref. frames sets how many frames it averages.",
        "Retake the displayed camera's reference, averaging Ref. frames frames."),
    "Previous": (
        "The earlier frame that Previous view subtracted to make the difference "
        "(Frames back sets how far), after single field and noise reduction. "
        "Gamma only.",
        "No previous frame yet.\n\nIt appears once the camera has delivered "
        "enough frames to look back; Frames back sets how many.",
        "Not used in Previous view: the earlier frame is taken automatically."),
}
NO_REFERENCE_TEXT = COMPARE_BASELINES["Reference"][1]
AVERAGING_TEXT = ("Averaging...\n\nThe difference appears when the average is "
                  "complete; Average N sets how many it sums.")
COMPARE_POLL_MS = 50


class ReferenceCompareDialog(QDialog):
    """Show the displayed camera's baseline, current frame and difference live.

    The baseline is the stored reference, or in Previous view the earlier
    frame that view subtracts. The images come from the engine's
    ComparePacket, so all three belong to the same moment. The engine builds
    them only while this window is open.
    """

    def __init__(self, window):
        """Build the three panels and the buttons.

        Args: window is the MainWindow; its engine (the displayed camera) is
            the source, and its frame title and notes go with a save.
        Returns: None.
        """
        super().__init__(window)
        self.window = window
        self.engine = None  # the engine currently asked for comparisons
        self.shown = None  # the ComparePacket on screen (Freeze keeps it)
        self.view = "Reference"  # the view whose baseline the panels are named for
        self.tables = {}
        self.setWindowTitle("Reference comparison")
        self.resize(ui_scale.px(1200), ui_scale.px(480))
        grid = QGridLayout()
        self.views, self.stacks, self.messages, self.captions = {}, {}, {}, {}
        self.titles = {}
        for column, (name, heading, tip) in enumerate(COMPARE_PANELS):
            title = QLabel(heading)
            font = title.font()
            font.setBold(True)
            title.setFont(font)
            if tip:  # the baseline's tooltip depends on the view: set_view
                set_tooltip(title, tip)
            view = CpuImageView()
            view.setMinimumSize(ui_scale.px(200), ui_scale.px(150))
            message = QLabel(NO_REFERENCE_TEXT)
            message.setWordWrap(True)
            message.setAlignment(Qt.AlignCenter)
            message.setFrameShape(QFrame.StyledPanel)
            stack = QStackedWidget()
            stack.addWidget(view)
            stack.addWidget(message)
            caption = QLabel()
            caption.setWordWrap(True)
            grid.addWidget(title, 0, column)
            grid.addWidget(stack, 1, column)
            grid.addWidget(caption, 2, column, Qt.AlignTop)
            grid.setColumnStretch(column, 1)
            self.views[name], self.stacks[name] = view, stack
            self.messages[name], self.captions[name] = message, caption
            self.titles[name] = title
        grid.setRowStretch(1, 1)
        self.demo_label = QLabel("DUMMY MODE — synthetic video, not camera data")
        self.demo_label.setAlignment(Qt.AlignCenter)
        self.demo_label.setStyleSheet(
            f"background:#8b0000;color:white;font-weight:bold;"
            f"padding:{ui_scale.px(4)}px;border-radius:{ui_scale.px(4)}px;")
        self.demo_label.setVisible(False)
        buttons = QDialogButtonBox()
        self.reference_button = buttons.addButton("Set reference (F)",
                                                  QDialogButtonBox.ActionRole)
        self.freeze_button = buttons.addButton("Freeze", QDialogButtonBox.ActionRole)
        self.freeze_button.setCheckable(True)
        self.freeze_button.setStyleSheet(f"QPushButton:checked {{{FROZEN_STYLE}}}")
        set_tooltip(self.freeze_button,
                    "Hold these three images still to compare them. The camera "
                    "keeps running; the main window stays live.")
        self.save_button = buttons.addButton("Save all three", QDialogButtonBox.ActionRole)
        set_tooltip(self.save_button,
                    "Save the three images as shown (PNG) plus one notes file with "
                    "the main window's title and notes and how the reference or "
                    "previous frame was taken.")
        close = buttons.addButton(QDialogButtonBox.Close)
        for button in buttons.buttons():
            button.setAutoDefault(False)  # Enter must not press Save or Close
        self.reference_button.clicked.connect(
            lambda: self.window.engine.request("reference"))
        self.freeze_button.toggled.connect(self.set_frozen)
        self.save_button.clicked.connect(self.save)
        close.clicked.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.demo_label)
        layout.addLayout(grid, 1)
        layout.addWidget(buttons)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.set_view("Reference", "Reference")

    def showEvent(self, event):
        """Ask the displayed camera for comparisons while the window is open.

        Args: event is the Qt show event.
        Returns: None.
        """
        super().showEvent(event)
        self._follow(self.window.engine)
        self.timer.start(COMPARE_POLL_MS)

    def hideEvent(self, event):
        """Stop the comparisons, so a closed window costs the camera nothing.

        Args: event is the Qt hide event.
        Returns: None.
        """
        super().hideEvent(event)
        self.timer.stop()
        self._follow(None)
        self.freeze_button.setChecked(False)

    def _follow(self, engine):
        """Move the comparison request to another engine.

        Args: engine is the Engine to compare from, or None for none.
        Returns: None.
        """
        if engine is self.engine:
            return
        if self.engine is not None:
            self.engine.set_compare(False)
        self.engine = engine
        self.shown = None
        if engine is not None:
            engine.set_compare(True)
            settings = engine.settings()
            if settings.mode == "Previous":
                back = settings.frames_back
                self.set_view("Previous", f"Previous ({back} "
                              f"{'frame' if back == 1 else 'frames'} back)")
            else:
                self.set_view("Reference", "Reference")

    def poll(self):
        """Show the newest comparison of the displayed camera.

        Follows the main window to another camera; does nothing while frozen.

        Args: None.
        Returns: None.
        """
        if self.window.engine is not self.engine:
            self.freeze_button.setChecked(False)
            self._follow(self.window.engine)
        if self.freeze_button.isChecked():
            return
        packet = self.engine.latest_compare()
        if packet is None or packet is self.shown:
            return
        self.show_compare(packet)

    def set_view(self, view, label):
        """Name the panels for a view, and enable Set reference only where it applies.

        Args: view is "Reference" (also Raw view) or "Previous"; label is the
            baseline panel's heading, for example "Previous (2 frames back)".
        Returns: None.
        """
        self.view = view
        tip, _, button_tip = COMPARE_BASELINES[view]
        self.titles["baseline"].setText(label)
        set_tooltip(self.titles["baseline"], tip)
        self.setWindowTitle(f"{view} comparison")
        self.reference_button.setEnabled(view == "Reference")
        set_tooltip(self.reference_button, button_tip)
        self.messages["baseline"].setText(COMPARE_BASELINES[view][1])
        self.messages["difference"].setText(COMPARE_BASELINES[view][1])

    def show_compare(self, packet):
        """Put a ComparePacket's three images and captions on screen.

        Args: packet is the ComparePacket.
        Returns: None.
        """
        self.shown = packet
        self.set_view(packet.view, packet.baseline_label)
        self.demo_label.setVisible(packet.demo)
        self.setWindowTitle(f"{packet.view} comparison" + (" — DUMMY MODE" if packet.demo
                                                          else ""))
        if packet.view == "Previous" and packet.baseline is not None:
            self.messages["difference"].setText(AVERAGING_TEXT)
        captions = {"baseline": packet.baseline_info,
                    "current": "live, after single field and noise reduction",
                    "difference": packet.difference_info}
        for name, _, _ in COMPARE_PANELS:
            part = getattr(packet, name)
            self.captions[name].setText(captions[name])
            if part is None:
                self.stacks[name].setCurrentWidget(self.messages[name])
                continue
            key = round(part.gamma, 4)
            if key not in self.tables:
                self.tables[key] = display_rgb_table(part.gamma)
            self.views[name].show_packet(part, self.tables[key], key)
            self.stacks[name].setCurrentWidget(self.views[name])

    def set_frozen(self, on):
        """Hold the shown images still, or follow the camera again.

        Args: on is the Freeze button state.
        Returns: None.
        """
        self.window.log.write(f"{self.view} comparison frozen." if on else
                              f"{self.view} comparison live again.")

    def save(self):
        """Save the three shown images through the engine.

        Args: None.
        Returns: None.
        """
        if self.shown is None:
            self.window.log.write("No comparison on screen to save yet.")
            return
        self.engine.request("save_compare", packet=self.shown,
                            title=self.window.frame_title.text(),
                            notes=self.window.frame_notes.toPlainText())


PIXELS_POLL_MS = 100
PIXELS_SIZE_SETTING = "pixels/size"
# Numbers are written on a cell at least this many screen pixels wide; the
# hint says to enlarge the window when the square's cells are smaller.
MIN_NUMBER_CELL = 16
PIXELS_HINT = (
    "Each cell is one camera pixel, enlarged, with its value written on it. "
    "In Previous and Reference view the value is the raw difference in camera "
    "levels (0–255), before the noise threshold, Difference gain, averaging and "
    "Scaling: 0 means that pixel did not change, and movement raises it. In Raw "
    "view it is the camera level. Cells are shaded from black (0) to white (the "
    "largest value in the square). Click the main image to move the square; "
    "enlarge this window if a large square's numbers are hidden."
)


class PixelGrid(QWidget):
    """Draw a pixel square enlarged, each cell shaded and numbered."""

    def __init__(self, parent=None):
        """Create an empty grid.

        Args: parent is the owning widget.
        Returns: None.
        """
        super().__init__(parent)
        self.packet = None
        self.message = "Waiting for the camera…"
        self.setMinimumSize(200, 200)
        self.setAttribute(Qt.WA_OpaquePaintEvent)

    def set_packet(self, packet):
        """Show a square, or none.

        Args: packet is a SquarePacket, or None to show the message.
        Returns: None.
        """
        self.packet = packet
        self.update()

    def cell_size(self):
        """Measure one cell on screen.

        Args: None.
        Returns: The cell's width in logical pixels, or 0 without a square.
        """
        if self.packet is None or self.packet.values is None:
            return 0
        return min(self.width(), self.height()) / self.packet.size

    def paintEvent(self, event):
        """Shade and number every cell, or show why there is nothing yet.

        Args: event is the Qt paint event.
        Returns: None.
        """
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().color(QPalette.ColorRole.Window))
        packet = self.packet
        if packet is None or packet.values is None:
            text = self.message if packet is None else packet.summary().split(" · ")[-1]
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, text)
            return
        values = np.asarray(packet.values)
        size, cell = packet.size, self.cell_size()
        left = (self.width() - cell * size) / 2
        top = (self.height() - cell * size) / 2
        peak = int(values.max())
        shades = (np.rint(values.astype(np.float32) * (255.0 / peak)) if peak else
                  np.zeros(values.shape, np.float32)).astype(np.uint8)
        numbers = cell >= MIN_NUMBER_CELL
        if numbers:
            font = painter.font()
            font.setPixelSize(max(6, int(cell * 0.36)))
            painter.setFont(font)
        painter.setPen(QPen(QColor(90, 90, 90), 1) if cell >= 6 else Qt.NoPen)
        for row in range(size):
            for column in range(size):
                shade = int(shades[row, column])
                area = QRectF(left + column * cell, top + row * cell, cell, cell)
                painter.setBrush(QColor(shade, shade, shade))
                painter.drawRect(area)
                if numbers:
                    pen = painter.pen()
                    painter.setPen(Qt.white if shade < 140 else Qt.black)
                    painter.drawText(area, Qt.AlignCenter, str(int(values[row, column])))
                    painter.setPen(pen)


class PixelSquareDialog(QDialog):
    """Show an N×N square of the displayed camera's raw difference, numbered.

    The engine cuts the square only while this window is open. A click on
    the main image moves it there; it starts at the image center.
    """

    def __init__(self, window):
        """Build the size choice, Freeze, the summary and the grid.

        Args: window is the MainWindow; its engine (the displayed camera) is
            the source.
        Returns: None.
        """
        super().__init__(window)
        self.window = window
        self.engine = None  # the engine currently asked for squares
        self.shown = None  # the SquarePacket on screen (Freeze keeps it)
        self.center = None  # displayed (x, y) the square is centered on; None = middle
        saved = QSettings("ESPI", "espi_live").value(PIXELS_SIZE_SETTING,
                                                      DEFAULT_SQUARE_SIZE)
        try:
            self.size = int(saved)
        except (TypeError, ValueError):
            self.size = DEFAULT_SQUARE_SIZE
        if self.size not in SQUARE_SIZES:
            self.size = DEFAULT_SQUARE_SIZE
        self.setWindowTitle("Pixel values")
        self.resize(ui_scale.px(640), ui_scale.px(720))
        self.demo_label = QLabel("DUMMY MODE — synthetic video, not camera data")
        self.demo_label.setAlignment(Qt.AlignCenter)
        self.demo_label.setStyleSheet(
            f"background:#8b0000;color:white;font-weight:bold;"
            f"padding:{ui_scale.px(4)}px;border-radius:{ui_scale.px(4)}px;")
        self.demo_label.setVisible(False)
        self.size_combo = WheelIgnoringComboBox()
        for size in SQUARE_SIZES:
            self.size_combo.addItem(f"{size} × {size} pixels", size)
        self.size_combo.setCurrentIndex(SQUARE_SIZES.index(self.size))
        set_tooltip(self.size_combo, "How many camera pixels the square spans on each side.")
        self.size_combo.currentIndexChanged.connect(self.set_size)
        self.freeze_button = QPushButton("Freeze")
        self.freeze_button.setCheckable(True)
        self.freeze_button.setStyleSheet(f"QPushButton:checked {{{FROZEN_STYLE}}}")
        set_tooltip(self.freeze_button,
                    "Hold these values still to read them. The camera keeps running; "
                    "the main window stays live.")
        self.freeze_button.toggled.connect(self.set_frozen)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Square"))
        controls.addWidget(self.size_combo)
        controls.addStretch(1)
        controls.addWidget(self.freeze_button)
        self.summary = QLabel("Waiting for the camera…")
        self.summary.setWordWrap(True)
        set_tooltip(self.summary, "Where the square is on the image, and the smallest, "
                    "mean and largest value in it.")
        self.grid = PixelGrid()
        hint = QLabel(PIXELS_HINT)
        hint.setWordWrap(True)
        buttons = QDialogButtonBox()
        close = buttons.addButton(QDialogButtonBox.Close)
        close.setAutoDefault(False)
        close.clicked.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.demo_label)
        layout.addLayout(controls)
        layout.addWidget(self.summary)
        layout.addWidget(self.grid, 1)
        layout.addWidget(hint)
        layout.addWidget(buttons)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)

    def showEvent(self, event):
        """Ask the displayed camera for squares while the window is open.

        Args: event is the Qt show event.
        Returns: None.
        """
        super().showEvent(event)
        self._follow(self.window.engine)
        self.timer.start(PIXELS_POLL_MS)

    def hideEvent(self, event):
        """Stop the squares and remove the outline from the main image.

        Args: event is the Qt hide event.
        Returns: None.
        """
        super().hideEvent(event)
        self.timer.stop()
        self._follow(None)
        self.freeze_button.setChecked(False)

    def _follow(self, engine):
        """Move the square request to another engine.

        Args: engine is the Engine to read from, or None for none.
        Returns: None.
        """
        if self.engine is not None:
            self.engine.set_square(None)
        self.engine = engine
        self.shown = None
        self.grid.set_packet(None)
        self.window.view.set_box(None)
        self._request()

    def _request(self):
        """Send the current size and center to the engine.

        Args: None.
        Returns: None.
        """
        if self.engine is not None:
            self.engine.set_square(self.size, self.center)

    def set_size(self, index):
        """Use another square size, and remember it.

        Args: index is the size choice's index into SQUARE_SIZES.
        Returns: None.
        """
        self.size = SQUARE_SIZES[index]
        QSettings("ESPI", "espi_live").setValue(PIXELS_SIZE_SETTING, self.size)
        self._request()

    def move_to(self, x, y):
        """Center the square on a displayed pixel.

        Args: x and y are the image pixel clicked on the main image.
        Returns: None.
        """
        self.center = (x, y)
        self._request()
        self.freeze_button.setChecked(False)

    def poll(self):
        """Show the displayed camera's newest square.

        Follows the main window to another camera; does nothing while frozen.

        Args: None.
        Returns: None.
        """
        if self.window.engine is not self.engine:
            self.freeze_button.setChecked(False)
            self._follow(self.window.engine)
        if self.freeze_button.isChecked():
            return
        packet = self.engine.latest_square()
        if packet is None or packet is self.shown:
            return
        self.show_square(packet)

    def show_square(self, packet):
        """Put a square on screen and outline it on the main image.

        Args: packet is the SquarePacket.
        Returns: None.
        """
        self.shown = packet
        self.grid.set_packet(packet)
        self.summary.setText(packet.summary())
        self.demo_label.setVisible(packet.demo)
        self.setWindowTitle("Pixel values" + (" — DUMMY MODE" if packet.demo else ""))
        self.window.view.set_box(packet.box())

    def set_frozen(self, on):
        """Hold the shown values still, or follow the camera again.

        Args: on is the Freeze button state.
        Returns: None.
        """
        self.window.log.write("Pixel values frozen." if on else "Pixel values live again.")


DAQ_POLL_MS = 100
DAQ_SHUTDOWN_S = 5.0  # longest wait for the DAQ to reach 0 V at exit
DAQ_SETTINGS_KEY = "daq/settings"
DAQ_TITLE = "Frequency generator and piezo (NI DAQ)"
GENERATOR_KINDS = {"ao": "sine", "ctr": "square, 5 V TTL"}
PIEZO_MODE_TEXT = ((daq.HOLD, "Hold a voltage"),
                   (daq.SINE, "Sine at the generator's frequency"),
                   (daq.TRIANGLE, "Triangle at its own frequency"))
PIEZO_SCALE_TEXT = ((daq.CALIBRATED, "Calibrated (Calibrate phase step)"),
                    (daq.MANUAL, "Manual (typed in)"))
PHASE_STEP_TEXT = ((5, "5-step (Hariharan)"), (4, "4-step"))
PHASE_MESSAGE_MS = 10000  # how long phase-step notices stay in the status bar


def daq_spin(minimum, maximum, decimals, suffix, step):
    """Build a number box for the DAQ window.

    Args: minimum and maximum bound the value; decimals and step set its
        resolution; suffix is the unit shown after the number.
    Returns: A WheelIgnoringSpinBox.
    """
    spin = WheelIgnoringSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setSuffix(suffix)
    spin.setKeyboardTracking(False)
    return spin


def fit_label(label):
    """Give a word-wrapped label the height its text needs, or hide it when empty.

    A wrapped label in a QFormLayout keeps the height of its first text, so
    a longer one is cut off, and an empty one still takes a row.

    Args: label is a word-wrapped QLabel.
    Returns: None.
    """
    label.setVisible(bool(label.text()))
    label.setMinimumHeight(label.heightForWidth(max(1, label.width()))
                           if label.text() else 0)


def generator_choice_text(output):
    """Describe a frequency-generator output choice.

    Args: output is "ao0", "ctr1" and so on, or daq.OFF.
    Returns: Text such as "AO 0 (sine)" or "Counter 1 (square, 5 V TTL)".
    """
    if output == daq.OFF:
        return "Off (not assigned)"
    kind = GENERATOR_KINDS["ctr" if output.startswith("ctr") else "ao"]
    return f"{daq.output_label(output)} ({kind})"


class DaqDialog(QDialog):
    """Assign NI DAQ outputs to the frequency generator and the piezo driver.

    The window edits DaqSettings and sends them to the window's
    DaqController, which owns the device on its own thread. Closing the
    window leaves the outputs running; they go to 0 V with "All outputs to
    0 V", on an error, and when ESPI live closes.
    """

    def __init__(self, window, controller):
        """Build the form from the saved settings.

        Args: window is the MainWindow; controller is its daq.DaqController.
        Returns: None.
        """
        super().__init__(window)
        self.window = window
        self.controller = controller
        self.outputs = None  # DeviceOutputs of the connected device
        self.scanned = False
        self.scan_note = ""  # why no real device is listed, from the last scan
        self.setWindowTitle(DAQ_TITLE)
        self.sized = False  # the first show fits the window to its contents and the screen
        saved = self._saved_settings()
        self.quarter_wave = {1: 0.0, 2: 0.0}  # volts per 90° from Calibrate phase step

        self.device_box = WheelIgnoringComboBox()
        set_tooltip(self.device_box,
                    "The NI device to drive. The simulated device drives nothing: "
                    "use it to try the controls without hardware.")
        self.rescan_button = QPushButton("Rescan")
        set_tooltip(self.rescan_button, "Look for NI devices again (after plugging one in).")
        self.device_status = QLabel("Looking for NI devices…")
        self.device_status.setWordWrap(True)
        device_row = QHBoxLayout()
        device_row.addWidget(self.device_box, 1)
        device_row.addWidget(self.rescan_button)

        self.generator_output = WheelIgnoringComboBox()
        set_tooltip(self.generator_output,
                    "Which output produces the frequency generator's signal. An analog "
                    "output gives a sine for the shaker or PZT amplifier; a counter gives a "
                    "5 V square wave on a PFI terminal.")
        self.generator_terminal = WheelIgnoringComboBox()
        set_tooltip(self.generator_terminal,
                    "The PFI terminal a counter's square wave comes out on. On the "
                    "USB-6251 BNC, PFI 0–7 have BNC sockets.")
        self.piezo_widgets = {index: self._build_piezo_widgets(index)
                              for index in daq.PIEZOS}
        # Driver 1 keeps the historic attribute names (piezo_on, ...); driver
        # 2 gets the same names with a piezo2_ prefix.
        special = {"step_down": ("step_down", "piezo2_step_down"),
                   "step_up": ("step_up", "piezo2_step_up"),
                   "follow_result": ("piezo_follow_result", "piezo2_follow_result"),
                   "result": ("piezo_result", "piezo2_result")}
        for index in daq.PIEZOS:
            for name, widget in self.piezo_widgets[index].items():
                attribute = special[name][index - 1] if name in special \
                    else f"piezo{'2' if index == 2 else ''}_{name}"
                setattr(self, attribute, widget)
        self.piezo_camera_box = self._camera_chooser(
            1, "Which camera's Phase view this driver steps. None follows the "
               "displayed camera, as it always has.")
        self.piezo2_camera_box = self._camera_chooser(
            2, "Which camera's Phase view the second driver steps. Pick another "
               "camera than Thor 1's to phase-step two cameras at once.")
        generator_row = QHBoxLayout()
        generator_row.addWidget(self.generator_output, 1)
        generator_row.addWidget(self.generator_terminal)
        assign = QGroupBox("Assign outputs")
        assign_form = QFormLayout(assign)
        assign_form.addRow("Frequency generator", generator_row)
        for index in daq.PIEZOS:
            assign_form.addRow(daq.piezo_driver_label(index),
                               self.piezo_widgets[index]["output"])
        assign_form.addRow("Thor 1 steps camera", self.piezo_camera_box)
        assign_form.addRow("Thor 2 steps camera", self.piezo2_camera_box)

        self.generator_on = QCheckBox("Generator on")
        set_tooltip(self.generator_on,
                    "Produce the signal after Apply. Off holds the output at 0 V.")
        self.frequency = daq_spin(daq.MIN_FREQUENCY_HZ, daq.MAX_SQUARE_HZ, 3, " Hz", 10.0)
        set_tooltip(self.frequency,
                    f"Drive frequency. Sine: up to {daq.MAX_SINE_HZ:,.0f} Hz; square: up "
                    f"to {daq.MAX_SQUARE_HZ:,.0f} Hz. The device rounds it slightly; "
                    "the value it produces is shown below.")
        self.amplitude = daq_spin(0.0, daq.MAX_VOLTS, 4, " V", 0.01)
        set_tooltip(self.amplitude, "Sine amplitude (peak) at the DAQ, before the amplifier.")
        self.offset = daq_spin(-daq.MAX_VOLTS, daq.MAX_VOLTS, 4, " V", 0.01)
        set_tooltip(self.offset, "DC offset added to the sine.")
        self.generator_limit = daq_spin(0.01, daq.MAX_VOLTS, 2, " V", 0.1)
        set_tooltip(self.generator_limit,
                    "Safety limit: the generator never exceeds this many volts either "
                    "way. Set it to the amplifier's largest safe input.")
        self.generator_result = QLabel()
        self.generator_result.setWordWrap(True)
        generator = QGroupBox("Frequency generator")
        generator_form = QFormLayout(generator)
        generator_form.addRow(self.generator_on)
        generator_form.addRow("Frequency", self.frequency)
        generator_form.addRow("Amplitude", self.amplitude)
        generator_form.addRow("Offset", self.offset)
        generator_form.addRow("Safety limit", self.generator_limit)
        generator_form.addRow(self.generator_result)

        self.piezo_groups = [self._build_piezo_group(index) for index in daq.PIEZOS]

        self.problem_label = QLabel()
        self.problem_label.setWordWrap(True)
        self.problem_label.setStyleSheet("color:#c00000;")
        self.state_label = QLabel("Outputs: not connected")
        self.state_label.setWordWrap(True)
        self.safety_label = QLabel()
        self.safety_label.setWordWrap(True)
        self.test_label = QLabel()
        self.test_label.setWordWrap(True)
        self.test_label.setTextFormat(Qt.RichText)
        note = QLabel("Volts are at the DAQ, before your amplifiers. Closing this window "
                      "leaves the outputs running; they go to 0 V when ESPI live closes.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        buttons = QDialogButtonBox()
        self.apply_button = buttons.addButton("Apply", QDialogButtonBox.ApplyRole)
        set_tooltip(self.apply_button, "Send these settings to the device.")
        self.zero_button = buttons.addButton("All outputs to 0 V",
                                             QDialogButtonBox.ActionRole)
        set_tooltip(self.zero_button,
                    "Stop the generator and set every assigned output to 0 V.")
        self.test_button = buttons.addButton("Test outputs…", QDialogButtonBox.ActionRole)
        set_tooltip(self.test_button,
                    "Self-test: step each analog output to "
                    f"{daq.SELF_TEST_V:g} V and back, and read it back through the "
                    "device's internal channels (no wiring). Switch the MDT694A off first.")
        self.wiring_button = buttons.addButton("Where to plug in…",
                                               QDialogButtonBox.HelpRole)
        set_tooltip(self.wiring_button,
                    "Show a drawing of the NI USB-6251 with the sockets these settings "
                    "use, and where each cable goes.")
        close = buttons.addButton(QDialogButtonBox.Close)
        set_tooltip(close, "Hide this window. The outputs keep running.")
        for button in buttons.buttons():
            button.setAutoDefault(False)  # Enter in a number box must not apply

        # The settings scroll; the note and the buttons stay in view below them,
        # so Apply and the other buttons never fall off a small screen.
        self.body = QWidget()
        body_layout = QVBoxLayout(self.body)
        body_layout.addLayout(device_row)
        body_layout.addWidget(self.device_status)
        body_layout.addWidget(assign)
        body_layout.addWidget(generator)
        for group in self.piezo_groups:
            body_layout.addWidget(group)
        body_layout.addWidget(self.problem_label)
        body_layout.addWidget(self.state_label)
        body_layout.addWidget(self.safety_label)
        body_layout.addWidget(self.test_label)
        body_layout.addStretch(1)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidget(self.body)
        self.footer = [note, buttons]
        layout = QVBoxLayout(self)
        layout.addWidget(self.scroll, 1)
        layout.addWidget(note)
        layout.addWidget(buttons)

        self.rescan_button.clicked.connect(self.scan)
        self.device_box.activated.connect(self._connect_selected)
        self.apply_button.clicked.connect(self.apply)
        self.zero_button.clicked.connect(lambda: self.controller.request("zero"))
        self.wiring = None  # the "Where to plug in" window, built on first use
        self.wiring_button.clicked.connect(self.show_wiring)
        self.checklist = None  # the safety checklist, while it is open
        self.test_button.clicked.connect(self.run_self_test)
        close.clicked.connect(self.reject)
        self._fill_outputs(daq.DeviceOutputs(daq.SIMULATED_DEVICE))
        self.load(saved)
        for box in (self.generator_output, self.generator_terminal,
                    self.piezo_camera_box, self.piezo2_camera_box):
            box.currentIndexChanged.connect(self.refresh)
        for spin in (self.frequency, self.amplitude, self.offset, self.generator_limit):
            spin.valueChanged.connect(self.refresh)
        self.generator_on.toggled.connect(self.refresh)
        for index in daq.PIEZOS:
            w = self.piezo_widgets[index]
            for box in (w["output"], w["mode"], w["scale"]):
                box.currentIndexChanged.connect(self.refresh)
            for spin in (w["voltage"], w["amplitude"], w["phase"], w["bottom"],
                         w["top"], w["frequency"], w["step"], w["limit"],
                         w["phase_per_frame"], w["um_per_v"], w["gain"], w["angle"]):
                spin.valueChanged.connect(self.refresh)
            for check in (w["on"], w["follow"]):
                check.toggled.connect(self.refresh)
        self.connected(None)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)

    @staticmethod
    def _saved_settings():
        """Read the settings saved on this computer.

        Args: None.
        Returns: daq.DaqSettings (the defaults when none are saved or the
            saved text is unreadable).
        """
        text = QSettings("ESPI", "espi_live").value(DAQ_SETTINGS_KEY, "")
        try:
            data = json.loads(text) if text else {}
        except (TypeError, ValueError):
            data = {}
        return daq.DaqSettings.from_dict(data)

    @property
    def quarter_wave_v(self):
        """The first driver's volts per 90° from Calibrate phase step.

        Args: None.
        Returns: Volts, 0.0 when nothing was calibrated.
        """
        return self.quarter_wave[1]

    @quarter_wave_v.setter
    def quarter_wave_v(self, volts):
        """Record the first driver's volts per 90°.

        Args: volts is the new voltage per 90° step.
        Returns: None.
        """
        self.quarter_wave[1] = volts

    def _camera_chooser(self, index, tip):
        """Build the camera chooser of one piezo driver.

        Args: index is the driver, 1 or 2; tip is the chooser's tooltip.
        Returns: A WheelIgnoringComboBox whose data is a channel's profile
            key, or "" for "follow the displayed camera".
        """
        box = WheelIgnoringComboBox()
        set_tooltip(box, tip)
        box.currentIndexChanged.connect(self.refresh)
        return box

    def _build_piezo_widgets(self, index):
        """Build one piezo driver's controls.

        Args: index is the driver, 1 or 2 (Thor 1 or Thor 2).
        Returns: A dict from a short name ("output", "on", "mode", "voltage",
            "amplitude", "phase", "bottom", "top", "frequency", "follow",
            "phase_per_frame", "scale", "um_per_v", "gain", "angle",
            "follow_result", "step", "step_down", "step_up", "limit",
            "result", and for driver 1 also "auto_tune" and "tune_result")
            to the widget; _build_piezo_group lays it out.
        """
        which = "" if index == 1 else "second "
        w = {}
        w["output"] = box = WheelIgnoringComboBox()
        set_tooltip(box,
                    f"Which analog output drives the {which}reference piezo's "
                    "amplifier.")
        w["on"] = check = QCheckBox("Piezo on")
        set_tooltip(check, "Drive the piezo after Apply. Off holds it at 0 V.")
        w["mode"] = mode = WheelIgnoringComboBox()
        for mode_name, text in PIEZO_MODE_TEXT:
            mode.addItem(text, mode_name)
        set_tooltip(mode,
                    "Hold a voltage (step it by hand for phase steps), follow the "
                    "generator with a sine at the same frequency and a set phase, or "
                    "sweep a triangle between two voltages at its own frequency (the "
                    "ramp a function generator would feed the piezo amplifier).")
        w["voltage"] = daq_spin(daq.PIEZO_FLOOR_V, daq.MAX_VOLTS, 4, " V", 0.01)
        set_tooltip(w["voltage"],
                    "The held voltage, or the centre of the sine, at the DAQ.")
        w["amplitude"] = daq_spin(0.0, daq.MAX_VOLTS, 4, " V", 0.01)
        set_tooltip(w["amplitude"], "Sine amplitude (peak). Sine mode only.")
        w["phase"] = daq_spin(-360.0, 360.0, 1, "°", 5.0)
        set_tooltip(w["phase"],
                    "Phase of the piezo sine relative to the generator. Sine mode only.")
        w["bottom"] = daq_spin(daq.PIEZO_FLOOR_V, daq.MAX_VOLTS, 4, " V", 0.01)
        set_tooltip(w["bottom"],
                    "The triangle's lowest voltage (its DC offset). At 0 V or above, "
                    "so the piezo amplifier's input never goes negative. Triangle mode "
                    "only.")
        w["top"] = daq_spin(daq.PIEZO_FLOOR_V, daq.MAX_VOLTS, 4, " V", 0.01)
        set_tooltip(w["top"],
                    "The triangle's highest voltage, at most the safety limit. "
                    "Triangle mode only.")
        w["frequency"] = daq_spin(daq.MIN_TRIANGLE_HZ, daq.MAX_TRIANGLE_HZ, 4,
                                  " Hz", 0.01)
        set_tooltip(w["frequency"],
                    "How many times a second the triangle goes up and down. The device "
                    "rounds it slightly; the value it produces is shown below. Needs "
                    "the frequency generator off or on a counter, because both analog "
                    "outputs share one sample clock. Triangle mode only.")
        w["follow"] = QCheckBox("Follow the camera's frame rate")
        set_tooltip(w["follow"],
                    "Set the triangle's frequency from the rate frames arrive from the "
                    "camera this driver steps, so the reference phase changes by the "
                    "same amount (Phase per frame) between successive frames. It is "
                    "set again whenever the frame rate moves the frequency by more "
                    f"than {daq.FOLLOW_TOLERANCE:.0%}. Triangle mode only.")
        w["phase_per_frame"] = daq_spin(1.0, daq.MAX_PHASE_PER_FRAME_DEG, 1, "°", 10.0)
        set_tooltip(w["phase_per_frame"],
                    "How far the reference phase moves from one frame to the next. 180° "
                    "(half a wavelength of path) gives the strongest Previous-view "
                    "difference; with Frames back N, use 180° / N.")
        w["scale"] = scale = WheelIgnoringComboBox()
        for scale_name, text in PIEZO_SCALE_TEXT:
            scale.addItem(text, scale_name)
        set_tooltip(scale,
                    "Where the piezo's phase per volt comes from. Calibrated uses the "
                    "voltage per 90° step that Calibrate phase step measured (the mirror "
                    "angle and piezo are then included). Manual works it out from the "
                    "numbers below for the HeNe laser "
                    f"({daq.HENE_WAVELENGTH_NM:g} nm).")
        w["um_per_v"] = daq_spin(0.0001, 10.0, 4, " µm/V", 0.01)
        set_tooltip(w["um_per_v"],
                    "Manual: how far the piezo moves per volt across it. A Thorlabs PE4 "
                    "moves about 15 µm at 150 V, about 0.1 µm/V (open loop, approximate).")
        w["gain"] = daq_spin(0.1, 100.0, 2, " ×", 0.5)
        set_tooltip(w["gain"],
                    "Manual: the piezo amplifier's gain from the DAQ to the piezo (15 for "
                    "the Thorlabs MDT694A at its 150 V limit).")
        w["angle"] = daq_spin(0.0, daq.MAX_MIRROR_ANGLE_DEG, 1, "°", 5.0)
        set_tooltip(w["angle"],
                    "Manual: the angle of incidence of the reference beam on the piezo "
                    "mirror (0° straight on; 45° in the lab's setup). The path changes by "
                    "twice the mirror's travel times the cosine of this angle.")
        w["follow_result"] = label = QLabel()
        label.setWordWrap(True)
        if index == 1:  # auto-tune belongs to the first driver's group only
            self.auto_tune_button = tune = QPushButton("Auto-tune (experimental)")
            tune.setCheckable(True)
            set_tooltip(tune,
                        "Experimental. Tune the applied triangle's frequency from the "
                        "camera frames for a 180° step per frame, the steadiest "
                        "Previous-view image, and keep it there. The piezo first holds "
                        "still for a few seconds to measure the camera noise. Keep the "
                        "object still while it tunes. Turning it off keeps the tuned "
                        "frequency (with Follow on, the computed one returns when the "
                        "frame rate moves).")
            tune.toggled.connect(self.window.set_auto_tune)
            self.piezo_tune_result = label = QLabel()
            label.setWordWrap(True)
        w["step"] = daq_spin(0.0001, daq.MAX_VOLTS, 4, " V", 0.01)
        set_tooltip(w["step"],
                    "How far one Step press moves the held voltage (for example the "
                    "voltage of a quarter-wave phase step).")
        w["step_down"] = down = QPushButton("Step −")
        set_tooltip(down, "Lower the held voltage by one step and apply it.")
        w["step_up"] = up = QPushButton("Step +")
        set_tooltip(up, "Raise the held voltage by one step and apply it.")
        down.clicked.connect(lambda: self.step(-1, index))
        up.clicked.connect(lambda: self.step(1, index))
        row = QHBoxLayout()
        row.addWidget(w["step"], 1)
        row.addWidget(down)
        row.addWidget(up)
        w["step_row"] = row
        w["limit"] = daq_spin(0.01, daq.MAX_VOLTS, 2, " V", 0.1)
        set_tooltip(w["limit"],
                    "Safety limit: the piezo driver stays between 0 V and this many "
                    "volts. Set it to the piezo amplifier's largest safe input (10 V "
                    "for the Thorlabs MDT694A's EXT INPUT).")
        w["result"] = label = QLabel()
        label.setWordWrap(True)
        return w

    def _build_piezo_group(self, index):
        """Lay one piezo driver's controls out in a titled group.

        Args: index is the driver, 1 or 2, naming the widgets built by
            _build_piezo_widgets.
        Returns: A QGroupBox.
        """
        w = self.piezo_widgets[index]
        group = QGroupBox(daq.piezo_driver_label(index))
        form = QFormLayout(group)
        form.addRow(w["on"])
        form.addRow("Mode", w["mode"])
        form.addRow("Voltage", w["voltage"])
        form.addRow("Amplitude", w["amplitude"])
        form.addRow("Phase", w["phase"])
        form.addRow("Bottom", w["bottom"])
        form.addRow("Top", w["top"])
        form.addRow("Frequency", w["frequency"])
        form.addRow(w["follow"])
        form.addRow("Phase per frame", w["phase_per_frame"])
        form.addRow("Piezo scale", w["scale"])
        form.addRow("Piezo travel", w["um_per_v"])
        form.addRow("Amplifier gain", w["gain"])
        form.addRow("Mirror angle", w["angle"])
        form.addRow(w["follow_result"])
        if index == 1:
            form.addRow(self.auto_tune_button)
            form.addRow(self.piezo_tune_result)
        form.addRow("Step", w["step_row"])
        form.addRow("Safety limit", w["limit"])
        form.addRow(w["result"])
        return group

    def showEvent(self, event):
        """Follow the controller while the window is open; scan the first time.

        Args: event is the Qt show event.
        Returns: None.
        """
        super().showEvent(event)
        self.timer.start(DAQ_POLL_MS)
        if not self.scanned:
            self.scan()
        self.fit_to_screen()

    def fit_to_screen(self):
        """Size the window to show every control, but never taller than the screen.

        The first call picks the size that fits the whole form (or 90% of the
        screen's usable height when the form is taller, and then the form
        scrolls). Later calls only shrink a window that has outgrown the
        screen, so a size the user chose is kept.

        Args: None.
        Returns: None.
        """
        room = self.screen().availableGeometry()
        margins = self.layout().contentsMargins()
        gap = self.layout().spacing()
        footer_height = sum(widget.sizeHint().height() + gap for widget in self.footer)
        wanted_width = (self.body.sizeHint().width() + self.scroll.verticalScrollBar()
                        .sizeHint().width() + margins.left() + margins.right())
        wanted_height = (self.body.sizeHint().height() + footer_height
                         + margins.top() + margins.bottom())
        if self.sized:
            wanted_width, wanted_height = self.width(), self.height()
        else:
            wanted_width = max(wanted_width, ui_scale.px(620))
        self.sized = True
        self.resize(min(wanted_width, room.width() * 9 // 10),
                    min(wanted_height, room.height() * 9 // 10))

    def hideEvent(self, event):
        """Stop following the controller while hidden.

        Args: event is the Qt hide event.
        Returns: None.
        """
        self.timer.stop()
        super().hideEvent(event)

    def scan(self):
        """Ask for the list of NI devices.

        Args: None.
        Returns: None.
        """
        self.scanned = True
        self.device_status.setText("Looking for NI devices…")
        self.controller.request("scan")

    def _set_combo(self, box, data):
        """Select the entry with this data, when present.

        Args: box is a QComboBox; data is the wanted entry's data.
        Returns: True when it was found.
        """
        index = box.findData(data)
        if index >= 0:
            box.setCurrentIndex(index)
        return index >= 0

    def _fill_outputs(self, outputs):
        """List a device's outputs in the assignment boxes, keeping choices.

        Args: outputs is the device's daq.DeviceOutputs.
        Returns: None.
        """
        choices = ((self.generator_output,
                    (daq.OFF,) + outputs.analog + outputs.counters, generator_choice_text),
                   (self.generator_terminal, outputs.terminals, daq.terminal_label))
        for index in daq.PIEZOS:
            choices += ((self.piezo_widgets[index]["output"], (daq.OFF,) + outputs.analog,
                         lambda output: "Off (not assigned)" if output == daq.OFF
                         else daq.output_label(output)),)
        for box, values, text in choices:
            current = box.currentData()
            box.blockSignals(True)
            box.clear()
            for value in values:
                box.addItem(text(value), value)
            if not self._set_combo(box, current):
                box.setCurrentIndex(0)
            box.blockSignals(False)

    def load(self, settings):
        """Show settings in the form.

        Args: settings is daq.DaqSettings.
        Returns: None.
        """
        s = settings
        self._set_combo(self.generator_output, s.generator_output)
        self._set_combo(self.generator_terminal, s.generator_terminal)
        self._set_combo(self.piezo_camera_box, s.piezo_camera)
        self._set_combo(self.piezo2_camera_box, s.piezo2_camera)
        self.generator_on.setChecked(s.generator_on)
        self.frequency.setValue(s.generator_frequency_hz)
        self.amplitude.setValue(s.generator_amplitude_v)
        self.offset.setValue(s.generator_offset_v)
        self.generator_limit.setValue(s.generator_limit_v)
        for index in daq.PIEZOS:
            w = self.piezo_widgets[index]
            self._set_combo(w["output"], daq.pget(s, index, "output"))
            self._set_combo(w["mode"], daq.pget(s, index, "mode"))
            w["on"].setChecked(daq.pget(s, index, "on"))
            w["voltage"].setValue(daq.pget(s, index, "voltage_v"))
            w["amplitude"].setValue(daq.pget(s, index, "amplitude_v"))
            w["phase"].setValue(daq.pget(s, index, "phase_deg"))
            w["bottom"].setValue(daq.pget(s, index, "bottom_v"))
            w["top"].setValue(daq.pget(s, index, "top_v"))
            w["frequency"].setValue(daq.pget(s, index, "frequency_hz"))
            w["step"].setValue(daq.pget(s, index, "step_v"))
            w["limit"].setValue(daq.pget(s, index, "limit_v"))
            w["follow"].setChecked(daq.pget(s, index, "follow_fps"))
            w["phase_per_frame"].setValue(daq.pget(s, index, "phase_per_frame_deg"))
            self._set_combo(w["scale"], daq.pget(s, index, "scale"))
            w["um_per_v"].setValue(daq.pget(s, index, "um_per_v"))
            w["gain"].setValue(daq.pget(s, index, "gain"))
            w["angle"].setValue(daq.pget(s, index, "angle_deg"))
            self.quarter_wave[index] = daq.pget(s, index, "quarter_wave_v")
        self.wanted_device = s.device

    def settings(self):
        """Read the form.

        With a driver's Follow the camera's frame rate on, its triangle's
        frequency is the one its camera's frame rate needs, once measured
        and within the triangle's range; otherwise the typed one.

        Args: None.
        Returns: daq.DaqSettings.
        """
        typed = self._typed_settings()
        for index in daq.PIEZOS:
            if not daq.pget(typed, index, "follow_fps"):
                continue
            fps = self.window.received_fps_for(daq.pget(typed, index, "camera"))
            frequency, _ = daq.follow_frequency(typed, fps, index)
            if frequency is not None and daq.MIN_TRIANGLE_HZ <= frequency <= daq.MAX_TRIANGLE_HZ:
                typed = daq.pset(typed, index, frequency_hz=frequency)
        return typed

    def _typed_settings(self):
        """Read the form as typed.

        Args: None.
        Returns: daq.DaqSettings.
        """
        values = dict(
            device=self.device_box.currentData() or "",
            generator_output=self.generator_output.currentData() or daq.OFF,
            generator_terminal=self.generator_terminal.currentData() or "",
            generator_on=self.generator_on.isChecked(),
            generator_frequency_hz=self.frequency.value(),
            generator_amplitude_v=self.amplitude.value(),
            generator_offset_v=self.offset.value(),
            generator_limit_v=self.generator_limit.value())
        for index in daq.PIEZOS:
            w = self.piezo_widgets[index]
            values.update({
                daq.pfield(index, "output"): w["output"].currentData() or daq.OFF,
                daq.pfield(index, "on"): w["on"].isChecked(),
                daq.pfield(index, "mode"): w["mode"].currentData(),
                daq.pfield(index, "voltage_v"): w["voltage"].value(),
                daq.pfield(index, "amplitude_v"): w["amplitude"].value(),
                daq.pfield(index, "phase_deg"): w["phase"].value(),
                daq.pfield(index, "bottom_v"): w["bottom"].value(),
                daq.pfield(index, "top_v"): w["top"].value(),
                daq.pfield(index, "frequency_hz"): w["frequency"].value(),
                daq.pfield(index, "step_v"): w["step"].value(),
                daq.pfield(index, "limit_v"): w["limit"].value(),
                daq.pfield(index, "follow_fps"): w["follow"].isChecked(),
                daq.pfield(index, "phase_per_frame_deg"): w["phase_per_frame"].value(),
                daq.pfield(index, "scale"): w["scale"].currentData(),
                daq.pfield(index, "quarter_wave_v"): self.quarter_wave[index],
                daq.pfield(index, "um_per_v"): w["um_per_v"].value(),
                daq.pfield(index, "gain"): w["gain"].value(),
                daq.pfield(index, "angle_deg"): w["angle"].value(),
                daq.pfield(index, "camera"): self._camera_choice(index)})
        return daq.DaqSettings(**values)

    def _camera_choice(self, index):
        """The camera one driver steps, as its profile key.

        Args: index is the driver, 1 or 2.
        Returns: The chosen channel's profile key, or "" to follow the
            displayed camera.
        """
        box = self.piezo_camera_box if index == 1 else self.piezo2_camera_box
        return box.currentData() or ""

    def refresh(self, *_):
        """Enable the controls that apply, and preview or explain the result.

        Args: _ are the signal's values (ignored).
        Returns: None.
        """
        s = self.settings()
        outputs = self.outputs or daq.DeviceOutputs(daq.SIMULATED_DEVICE)
        counter = s.generator_output in outputs.counters
        sine_generator = s.generator_output in outputs.analog
        self.generator_terminal.setEnabled(counter)
        for widget in (self.amplitude, self.offset):
            widget.setEnabled(sine_generator)
        has_generator = s.generator_output != daq.OFF
        for widget in (self.generator_on, self.frequency, self.generator_limit):
            widget.setEnabled(has_generator)
        for index in daq.PIEZOS:
            self._refresh_piezo(s, index, outputs)
        self._refresh_camera_choosers()
        found = daq.problems(s, outputs)
        self.problem_label.setText("\n".join(found))
        self.piezo_widgets[1]["follow_result"].setText(
            "" if found else self._follow_text(s, 1))
        self.piezo_widgets[2]["follow_result"].setText(
            "" if found else self._follow_text(s, 2))
        fit_label(self.piezo_widgets[1]["follow_result"])
        fit_label(self.piezo_widgets[2]["follow_result"])
        self.show_auto_tune()
        self.apply_button.setEnabled(self.outputs is not None and not found)
        self.test_button.setEnabled(self.outputs is not None)
        self.safety_label.setText(self._safety_text(s))
        self.generator_result.setText(self._preview(s, outputs, found))
        for index in daq.PIEZOS:
            self.piezo_widgets[index]["result"].setText(
                self._piezo_preview(s, index, outputs, found))
        if self.wiring is not None:
            self.wiring.show_settings(s, self.outputs and self.outputs.device)

    def _refresh_piezo(self, s, index, outputs):
        """Enable one piezo driver's controls and show its follow frequency.

        Args: s is the form's daq.DaqSettings; index is the driver, 1 or 2;
            outputs is the connected device's daq.DeviceOutputs (the
            simulated one when nothing is connected).
        Returns: None.
        """
        w = self.piezo_widgets[index]
        has_piezo = daq.pget(s, index, "output") != daq.OFF
        sine_piezo = has_piezo and daq.pget(s, index, "mode") == daq.SINE
        triangle_piezo = has_piezo and daq.pget(s, index, "mode") == daq.TRIANGLE
        for widget in (w["on"], w["mode"], w["limit"]):
            widget.setEnabled(has_piezo)
        w["voltage"].setEnabled(has_piezo and not triangle_piezo)
        for widget in (w["amplitude"], w["phase"]):
            widget.setEnabled(sine_piezo)
        for widget in (w["bottom"], w["top"], w["follow"]):
            widget.setEnabled(triangle_piezo)
        following = triangle_piezo and daq.pget(s, index, "follow_fps")
        w["frequency"].setEnabled(triangle_piezo and not following)
        for widget in (w["phase_per_frame"], w["scale"]):
            widget.setEnabled(following)
        for widget in (w["um_per_v"], w["gain"], w["angle"]):
            widget.setEnabled(following and daq.pget(s, index, "scale") == daq.MANUAL)
        if following:
            w["frequency"].blockSignals(True)
            w["frequency"].setValue(daq.pget(s, index, "frequency_hz"))
            w["frequency"].blockSignals(False)
        for widget in (w["step"], w["step_down"], w["step_up"]):
            widget.setEnabled(has_piezo and daq.pget(s, index, "mode") == daq.HOLD)

    def _refresh_camera_choosers(self):
        """Fill the camera choosers from the window's channels and keep them apart.

        Each driver may step any open camera, but both drivers cannot step
        the same one: choosing a camera for one driver drops it from the
        other's list.

        Args: None.
        Returns: None.
        """
        choices = [("", "Follow the displayed camera")]
        choices += [(channel.profile_key, channel.label)
                    for channel in self.window.channels if channel.supported]
        chosen = {index: self._camera_choice(index) for index in daq.PIEZOS}
        for index in daq.PIEZOS:
            box = self.piezo_camera_box if index == 1 else self.piezo2_camera_box
            other = 2 if index == 1 else 1
            box.blockSignals(True)
            box.clear()
            for key, label in choices:
                if key and key == chosen[other]:
                    continue  # one camera per driver
                box.addItem(label, key)
            if not self._set_combo(box, chosen[index]):
                box.setCurrentIndex(0)
            box.blockSignals(False)

    def show_wiring(self):
        """Open the "Where to plug in" window for the settings in the form.

        It follows the form while it stays open.

        Args: None.
        Returns: None.
        """
        if self.wiring is None:
            self.wiring = DaqWiringDialog(self)
        self.wiring.show_settings(self.settings(), self.outputs and self.outputs.device)
        self.wiring.show()
        self.wiring.raise_()

    @staticmethod
    def _preview(settings, outputs, found):
        """Say what the generator will produce once applied.

        Args: settings is daq.DaqSettings; outputs the device's outputs;
            found is the list of problems.
        Returns: One line of text ("" when there is nothing to say).
        """
        if found or settings.generator_output == daq.OFF:
            return ""
        if settings.generator_output in outputs.counters:
            actual = daq.square_frequency(settings.generator_frequency_hz)
            return (f"Produces about {actual:.6g} Hz, 0–5 V square, on "
                    f"{daq.terminal_label(settings.generator_terminal)}.")
        channels = len({settings.generator_output,
                        *[daq.pget(settings, index, "output") for index in daq.PIEZOS]}
                       - {daq.OFF})
        rate, samples, actual = daq.choose_timing(settings.generator_frequency_hz, channels)
        return (f"Produces {actual:.6g} Hz: {samples} samples per cycle at "
                f"{rate:,.0f} samples/s.")

    def _follow_text(self, settings, index):
        """Say what frequency a triangle follows, and the scale behind it.

        Args: settings is the form's daq.DaqSettings; index is the piezo
            driver, 1 or 2.
        Returns: One line of text ("" when that triangle does not follow the
            frame rate, and the calibration on record otherwise).
        """
        s = settings
        if not daq.piezo_assigned(s, index) or daq.pget(s, index, "mode") != daq.TRIANGLE:
            return ""
        if not daq.pget(s, index, "follow_fps"):
            return ""
        fps = self.window.received_fps_for(daq.pget(s, index, "camera"))
        return daq.follow_text(s, fps, index)

    def show_auto_tune(self):
        """Show whether auto-tune runs and what it is doing.

        Args: None.
        Returns: None.
        """
        tune = self.auto_tune_button
        tuning = self.window.piezo_tuner is not None
        tune.setEnabled(tuning or (
            self.outputs is not None and self.piezo_widgets[1]["output"].currentData()
            != daq.OFF and self.piezo_widgets[1]["mode"].currentData() == daq.TRIANGLE))
        tune.blockSignals(True)
        tune.setChecked(tuning)
        tune.blockSignals(False)
        self.piezo_tune_result.setText(self.window.tune_text())
        fit_label(self.piezo_tune_result)

    def follow_frame_rate(self):
        """Show the frequency the newest frame rates need, while following them.

        Args: None.
        Returns: None.
        """
        for index in daq.PIEZOS:
            w = self.piezo_widgets[index]
            if w["follow"].isChecked() and w["follow"].isEnabled():
                self.refresh()
                return

    def set_quarter_wave(self, volts, index=1):
        """Take a new Calibrate phase step result for the Calibrated scale.

        Args: volts is the DAQ voltage per 90° step; index is the driver it
            was measured on, 1 or 2.
        Returns: None.
        """
        self.quarter_wave[index] = volts
        self.piezo_widgets[index]["step"].setValue(volts)
        self.refresh()

    @staticmethod
    def _piezo_preview(settings, index, outputs, found):
        """Say what a piezo triangle will produce once applied.

        Args: settings is daq.DaqSettings; index is the driver, 1 or 2;
            outputs the device's outputs; found is the list of problems.
        Returns: One line of text ("" outside triangle mode or with problems).
        """
        s = settings
        if found or not daq.piezo_assigned(s, index) \
                or daq.pget(s, index, "mode") != daq.TRIANGLE:
            return ""
        channels = len({s.generator_output,
                        *[daq.pget(s, i, "output") for i in daq.PIEZOS]}
                       & set(outputs.analog))
        rate, samples, actual = daq.choose_timing(
            daq.pget(s, index, "frequency_hz"), channels)
        which = "" if index == 1 else "second "
        return (f"Produces a {which}{actual:.6g} Hz triangle from "
                f"{daq.pget(s, index, 'bottom_v'):g} V to "
                f"{daq.pget(s, index, 'top_v'):g} V: {samples} samples per cycle at "
                f"{rate:,.0f} samples/s.")

    def _connect_selected(self, *_):
        """Connect to the device chosen in the list.

        Args: _ is the activated index (ignored).
        Returns: None.
        """
        device = self.device_box.currentData()
        if device:
            self.outputs = None
            self.connected(None)
            self.device_status.setText("Connecting…")
            self.controller.request("connect", device=device)

    def apply(self):
        """Save the settings and send them to the device.

        Args: None.
        Returns: None.
        """
        if self.window.piezo_tuner is not None:  # new settings replace what it tunes
            self.window.set_auto_tune(False, "new settings were applied")
        settings = self.settings()
        QSettings("ESPI", "espi_live").setValue(DAQ_SETTINGS_KEY,
                                                json.dumps(settings.to_dict()))
        if daq.drives(settings) and not self.controller.is_armed(settings):
            self._ask_checklist(settings)
            return
        self._send(settings)

    def _send(self, settings):
        """Send settings to the device.

        Args: settings is daq.DaqSettings.
        Returns: None.
        """
        self.state_label.setText("Outputs: applying…")
        self.controller.request("apply", settings=settings)

    def _ask_checklist(self, settings):
        """Show the safety checklist; once it is ticked off, apply the settings.

        Args: settings is the daq.DaqSettings waiting to be applied.
        Returns: None.
        """
        if self.checklist is not None:
            self.checklist.close()
        self.checklist = DaqChecklistDialog(self, settings)
        self.checklist.accepted.connect(lambda: self._checklist_done(settings))
        self.checklist.open()

    def _checklist_done(self, settings):
        """Record the ticked checklist with the controller and apply.

        Args: settings is the daq.DaqSettings the checklist was shown for.
        Returns: None.
        """
        self.controller.arm(settings)
        self.window.log.write("NI DAQ: safety checklist ticked off for "
                              f"{self.outputs.device.label if self.outputs else 'the device'}.")
        self.refresh()
        self._send(settings)

    def _safety_text(self, settings):
        """Say where the safety checklist stands for the connected device.

        Args: settings is the form's daq.DaqSettings.
        Returns: One line ("" with no device or the simulated one).
        """
        if self.outputs is None or self.outputs.device.simulated:
            return ""
        if daq.drives(settings) and self.controller.is_armed(settings) or (
                not daq.drives(settings) and self.controller.checklist_done()):
            return "Safety checklist: ticked off for what is switched on."
        return ("Safety checklist: shown before a drive signal to this device, for the "
                "outputs switched on (once per connection, and again for an output "
                "switched on or reassigned later).")

    def run_self_test(self):
        """Ask for the amplifiers to be off, then run the output self-test.

        Args: None.
        Returns: None.
        """
        answer = QMessageBox.question(
            self, "Test outputs",
            f"Each analog output goes to {daq.SELF_TEST_V:g} V for a moment and back to "
            "0 V, read back through the device's own internal channels.\n\n"
            "Switch the MDT694A off (or unplug its EXT INPUT) and the specimen drive "
            "before you start. Run the test?")
        if answer != QMessageBox.Yes:
            return
        self._switch_off()
        self.test_label.setText("Testing outputs…")
        self.controller.request("self_test")

    def step(self, direction, index=1):
        """Move a held piezo voltage by one step, and apply it if the piezo is on.

        Args: direction is +1 or -1; index is the driver to step, 1 or 2.
        Returns: None.
        """
        w = self.piezo_widgets[index]
        w["voltage"].setValue(w["voltage"].value() + direction * w["step"].value())
        if w["on"].isChecked() and self.apply_button.isEnabled():
            self.apply()

    def connected(self, outputs):
        """Show the connected device (or none).

        Args: outputs is the device's daq.DeviceOutputs, or None.
        Returns: None; a new connection (or none) starts at 0 V, so the
            demo cameras' drive is switched off.
        """
        self.outputs = outputs
        self.window.set_daq_drive(daq.Drive())
        if outputs is not None:
            self._fill_outputs(outputs)
            self._set_combo(self.device_box, outputs.device.name)
            info = outputs.device
            if info.simulated:
                note = f"{self.scan_note} " if self.scan_note else ""
                self.device_status.setText(f"{note}Simulated device: nothing is driven. "
                                           "Use it to try the controls.")
                # Styled like dummy mode, so simulated output is never taken for real.
                self.device_status.setStyleSheet(
                    "background:#8b0000;color:white;font-weight:bold;"
                    f"padding:{ui_scale.px(4)}px;border-radius:{ui_scale.px(4)}px;")
            else:
                self.device_status.setText(f"Connected to {info.label}.")
                self.device_status.setStyleSheet("")
            self.state_label.setText("Outputs: all at 0 V until you apply.")
        self.refresh()

    def poll(self, drive_only=False):
        """Handle what the controller has reported.

        The window drains the controller's events (it also follows the
        drive for the demo cameras and the piezo for the Phase view) and
        passes each one to handle().

        Args: drive_only is True to keep a device list, which would connect to
            a device, for when the window is shown (the window does this by
            itself while this window is closed).
        Returns: None.
        """
        self.window.poll_daq(defer_devices=drive_only)

    def handle(self, kind, payload):
        """Show one controller event in the form.

        Args: kind and payload are the event.
        Returns: None.
        """
        if kind == "devices":
            self._show_devices(*payload)
        elif kind == "connected":
            self.connected(payload)
        elif kind == "applied":
            prefix = ("Simulated, nothing driven: "
                      if self.outputs is not None and self.outputs.device.simulated
                      else "Outputs: ")
            self.state_label.setText(prefix + payload.summary)
            applied = self.controller.status()["settings"]
            if applied is not None and applied.to_dict() != self.settings().to_dict():
                # The settings came from somewhere else (the control server, the
                # frame-rate follower, auto-tune): show them, as if typed here.
                self.load(applied)
                self.refresh()
        elif kind == "rejected":
            self.problem_label.setText("\n".join(payload))
            self.state_label.setText("Outputs: not changed.")
        elif kind == "zeroed":
            self._switch_off()
            self.state_label.setText("Outputs: all at 0 V.")
        elif kind == "fault":
            self._switch_off()
            self.problem_label.setText(
                f"Self-check: {payload}. Every output was set to 0 V; find the cause "
                "before switching anything on again.")
            self.state_label.setText("Outputs: stopped by the self-check.")
        elif kind == "self_test":
            passed, lines = payload
            colour = "#2b8a3e" if passed else "#c00000"
            heading = "Self-test passed" if passed else "Self-test FAILED"
            self.test_label.setText(f'<span style="color:{colour}"><b>{heading}.</b><br>'
                                    + "<br>".join(html.escape(line) for line in lines)
                                    + "</span>")
            self.state_label.setText("Outputs: all at 0 V after the self-test.")
        elif kind == "failed":
            self._switch_off()
            self.problem_label.setText(
                f"The DAQ reported an error: {payload}. The outputs were set to "
                "0 V where the device allowed it.")
            self.state_label.setText("Outputs: stopped after an error.")

    def _switch_off(self):
        """Untick every On box, to match outputs at 0 V.

        Args: None.
        Returns: None.
        """
        self.generator_on.setChecked(False)
        for index in daq.PIEZOS:
            self.piezo_widgets[index]["on"].setChecked(False)

    def _show_devices(self, devices, reason):
        """List the devices found (plus the simulated one) and connect.

        Args: devices is a list of daq.DeviceInfo; reason explains why none
            could be listed, or is None.
        Returns: None.
        """
        self.scan_note = reason or ("" if devices else
                                    "No NI device found: plug it in and press Rescan.")
        self.device_box.clear()
        for info in devices + [daq.SIMULATED_DEVICE]:
            self.device_box.addItem(info.label, info.name)
        if not self._set_combo(self.device_box, self.wanted_device):
            self.device_box.setCurrentIndex(0)
        self._connect_selected()

WIRING_TITLE = "Where to plug in (NI USB-6251)"
ROLE_COLORS = {daq.PIEZO_ROLE: "#d9480f", daq.GENERATOR_ROLE: "#1c7ed6"}
# The drawing follows NI's "NI USB-6251 BNC Front Panel and Pinout" figure
# (NI 6251 Specifications), in its own units; it is scaled to the widget.
PANEL_SIZE = (1270.0, 2010.0)
PANEL_TOP = 40.0  # below the blue end cap
BNC_RADIUS = 64.0
PANEL_INK, PANEL_FACE, PANEL_CAP = "#24405f", "#f6f6f2", "#2f5d9b"
# Boxes printed on the panel: (title, x, y, width, height).
PANEL_BOXES = (("DIGITAL AND TIMING I/O", 12, 88, 750, 372),
               ("APFI", 12, 470, 310, 206),
               ("ANALOG OUTPUT", 12, 686, 482, 205),
               ("ANALOG INPUT", 12, 902, 750, 918),
               ("POWER (+5 V)", 772, 88, 168, 108),
               ("DIGITAL AND TIMING I/O", 772, 208, 486, 1612))
# The screw terminals from top to bottom: (label as printed, socket name or "").
SCREW_ROWS = (("USER 1", "user1"), ("USER 2", "user2"), ("D GND", ""), ("+5 V", ""),
              ("D GND", ""), ("P0.0", ""), ("P0.1", ""), ("P0.2", ""), ("P0.3", ""),
              ("D GND", ""), ("P0.4", ""), ("P0.5", ""), ("P0.6", ""), ("P0.7", ""),
              ("D GND", ""), ("PFI 8/P2.0", "pfi8"), ("PFI 9/P2.1", "pfi9"),
              ("PFI 10/P2.2", "pfi10"), ("PFI 11/P2.3", "pfi11"), ("D GND", ""),
              ("PFI 12/P2.4", "pfi12"), ("PFI 13/P2.5", "pfi13"),
              ("PFI 14/P2.6", "pfi14"), ("PFI 15/P2.7", "pfi15"), ("D GND", ""),
              ("D GND", ""), ("AI GND", ""), ("AI SENSE", ""), ("NC", ""), ("CHS GND", ""))
SCREW_TOP, SCREW_PITCH = 572.0, 38.66  # first row's centre and the row spacing
SCREW_TERMINALS = {name: index for index, (_, name) in enumerate(SCREW_ROWS)
                   if name.startswith("pfi")}


def panel_positions():
    """Place every BNC socket of the USB-6251 BNC in the drawing.

    Args: None.
    Returns: A dict from socket name ("ai0", "ao1", "pfi4", ...) to
        (centre x, centre y, label as printed) in drawing units.
    """
    columns = (168, 340, 512, 684)
    places = {}
    for index in range(4):
        x = columns[index]
        places[f"pfi{index}"] = (x, 213, f"PFI {index}/P1.{index}")
        places[f"pfi{index + 4}"] = (x, 380, f"PFI {index + 4}/P1.{index + 4}")
        places[f"ai{index}"] = (x, 1027, f"AI {index}")
        places[f"ai{index + 4}"] = (x, 1243, f"AI {index + 4}")
    places.update(apfi0=(168, 597, "APFI 0"), ao0=(168, 812, "AO 0"),
                  ao1=(340, 812, "AO 1"), user1=(938, 380, "USER 1"),
                  user2=(1112, 380, "USER 2"))
    return places


def panel_place(terminal):
    """Say where a socket is on the USB-6251 BNC.

    Args: terminal is a socket name such as "ao1" or "pfi12".
    Returns: "bnc", "screw" (PFI 8-15 on the terminal strip) or "" when
        the USB-6251 BNC has no such socket.
    """
    if terminal in panel_positions():
        return "bnc"
    return "screw" if terminal in SCREW_TERMINALS else ""


class DaqPanelView(QWidget):
    """A drawing of the USB-6251 BNC's front panel with the sockets to use marked.

    It follows NI's front-panel figure: the PFI BNCs at the top, APFI 0,
    the analog outputs, the analog inputs with their FS/GS switches, and
    on the right USER 1/2 and the screw-terminal strip. Each socket in use
    gets its role's colour and the number of its line in the legend.
    """

    def __init__(self, parent=None):
        """Start with nothing marked.

        Args: parent is the owning widget.
        Returns: None.
        """
        super().__init__(parent)
        self.marks = {}  # socket -> (number, colour)
        self.setMinimumSize(ui_scale.px(290), ui_scale.px(459))
        policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def set_connections(self, connections):
        """Mark the sockets a list of connections uses.

        Args: connections is a list of daq.Connection, numbered from 1 in order;
            sockets this panel does not have are left out.
        Returns: None.
        """
        self.marks = {item.terminal: (number, ROLE_COLORS[item.role])
                      for number, item in enumerate(connections, 1)
                      if panel_place(item.terminal)}
        self.update()

    def heightForWidth(self, width):
        """Keep the drawing's proportions.

        Args: width is the offered width in pixels.
        Returns: The matching height.
        """
        return round(width * PANEL_SIZE[1] / PANEL_SIZE[0])

    def hasHeightForWidth(self):
        """Tell layouts that the height follows the width.

        Args: None.
        Returns: True.
        """
        return True

    @staticmethod
    def _font(base, size, bold=False):
        """Make a font in drawing units.

        Args: base is the widget's font; size is the pixel size in drawing
            units; bold makes it bold.
        Returns: A QFont.
        """
        font = QFont(base)
        font.setPixelSize(size)
        font.setBold(bold)
        return font

    def paintEvent(self, event):
        """Draw the panel, its printed boxes and labels, the sockets and the marks.

        Args: event is the Qt paint event.
        Returns: None.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        scale = min(self.width() / PANEL_SIZE[0], self.height() / PANEL_SIZE[1])
        painter.translate((self.width() - PANEL_SIZE[0] * scale) / 2,
                          (self.height() - PANEL_SIZE[1] * scale) / 2)
        painter.scale(scale, scale)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(PANEL_CAP))
        painter.drawRoundedRect(QRectF(0, 0, PANEL_SIZE[0], PANEL_SIZE[1]), 40, 40)
        painter.setBrush(QColor(PANEL_FACE))
        painter.setPen(QPen(QColor(PANEL_INK), 3))
        painter.drawRect(QRectF(0, PANEL_TOP, PANEL_SIZE[0], PANEL_SIZE[1] - 2 * PANEL_TOP))
        painter.translate(0, PANEL_TOP)
        self._draw_printing(painter)
        for name, (x, y, label) in panel_positions().items():
            self._draw_bnc(painter, name, x, y, label)
        self._draw_terminals(painter)
        painter.end()

    def _draw_printing(self, painter):
        """Draw the boxes, titles, switches, model text and logo printed on the panel.

        Args: painter is the active QPainter, in drawing units.
        Returns: None.
        """
        base = self.font()
        for title, x, y, w, h in PANEL_BOXES:
            painter.setPen(QPen(QColor(PANEL_INK), 3))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(QRectF(x, y, w, h), 8, 8)
            painter.setFont(self._font(base, 19, True))
            painter.drawText(QRectF(x, y + 6, w, 26), Qt.AlignHCenter, title)
        painter.setBrush(QColor("#9fd59a"))
        painter.drawEllipse(QPointF(856, 158), 17, 17)  # the power light
        for x, y in ((47, 242), (1224, 246), (47, 1785), (1224, 1785)):
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(x, y), 23, 23)  # mounting screws
        painter.setFont(self._font(base, 16))
        for x in (168, 340, 512, 684):
            for y in (1128, 1344):  # the FS/GS switch under each AI
                painter.drawRect(QRectF(x - 34, y - 14, 68, 28))
                painter.drawText(QRectF(x - 74, y - 12, 36, 24), Qt.AlignRight, "FS")
                painter.drawText(QRectF(x + 40, y - 12, 36, 24), Qt.AlignLeft, "GS")
        painter.drawText(QRectF(60, 1500, 300, 24), Qt.AlignHCenter, "Floating Source (FS)")
        painter.drawText(QRectF(390, 1500, 300, 24), Qt.AlignHCenter,
                         "Ground Ref. Source (GS)")
        painter.drawText(QRectF(560, 552, 80, 50), Qt.AlignCenter, "⚠")
        painter.setFont(self._font(base, 24, True))
        painter.drawText(QRectF(940, 104, 318, 30), Qt.AlignRight, "NI USB-6251")
        painter.setFont(self._font(base, 17))
        painter.drawText(QRectF(900, 136, 358, 70), Qt.AlignRight,
                         "8 Inputs, 16-bit, 1.25 MS/s\nMultifunction I/O with\n"
                         "Correlated Digital I/O for USB")
        for x in (864, 1038):  # the write-on name boxes under USER 1/2
            painter.drawRect(QRectF(x, 478, 148, 48))
            painter.drawText(QRectF(x, 452, 148, 24), Qt.AlignHCenter, "NAME")
        painter.setPen(QPen(QColor(PANEL_INK), 2))
        painter.drawPolyline([QPointF(876, 388), QPointF(832, 428), QPointF(832, 572),
                              QPointF(878, 572)])
        painter.drawPolyline([QPointF(1174, 388), QPointF(1215, 428), QPointF(1215, 608),
                              QPointF(1110, 608)])
        painter.setFont(self._font(base, 44, True))
        painter.setPen(QColor(PANEL_INK))
        painter.drawText(QRectF(0, 1835, PANEL_SIZE[0], 100), Qt.AlignCenter,
                         "NATIONAL INSTRUMENTS")

    def _draw_bnc(self, painter, name, x, y, label):
        """Draw one BNC socket with its printed name, and its mark when in use.

        Args: painter is the active QPainter; name is the socket ("ao1");
            x and y are its centre; label is the name printed above it.
        Returns: None.
        """
        mark = self.marks.get(name)
        centre = QPointF(x, y)
        if mark:
            painter.setPen(QPen(QColor(mark[1]), 14))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(centre, BNC_RADIUS + 12, BNC_RADIUS + 12)
        painter.setPen(QPen(QColor(PANEL_INK), 3))
        painter.setBrush(QColor("#c9cdd2"))
        painter.drawEllipse(centre, BNC_RADIUS, BNC_RADIUS)
        painter.setBrush(QColor("#a7adb4"))
        painter.drawEllipse(centre, BNC_RADIUS * 0.78, BNC_RADIUS * 0.78)
        painter.setBrush(QColor("#2a2f35"))
        painter.drawEllipse(centre, BNC_RADIUS * 0.47, BNC_RADIUS * 0.47)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#d8b45a"))
        painter.drawEllipse(centre, 9, 9)
        font = self._font(self.font(), 17, mark is not None)
        painter.setFont(font)
        # As on NI's figure: the name sits just above the socket, under the box title.
        box = QRectF(x - 110, y - 93, 220, 26)
        if mark:  # a tag in the role's colour, so the ring cannot hide the name
            width = QFontMetrics(font).horizontalAdvance(label) + 16
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(mark[1]))
            painter.drawRoundedRect(QRectF(x - width / 2, box.y(), width, box.height()), 6, 6)
        painter.setPen(QColor("white") if mark else QColor(PANEL_INK))
        painter.drawText(box, Qt.AlignCenter, label)
        if mark:
            self._draw_badge(painter, QPointF(x + BNC_RADIUS * 0.9, y - BNC_RADIUS * 0.9),
                             mark)

    def _draw_badge(self, painter, centre, mark):
        """Draw a numbered badge in a role's colour.

        Args: painter is the active QPainter; centre is a QPointF; mark is
            (number, colour).
        Returns: None.
        """
        painter.setPen(QPen(QColor("white"), 4))
        painter.setBrush(QColor(mark[1]))
        painter.drawEllipse(centre, 28, 28)
        painter.setFont(self._font(self.font(), 30, True))
        painter.drawText(QRectF(centre.x() - 28, centre.y() - 28, 56, 56), Qt.AlignCenter,
                         str(mark[0]))

    def _draw_terminals(self, painter):
        """Draw the screw-terminal strip with its printed labels and any marks.

        Args: painter is the active QPainter, in drawing units.
        Returns: None.
        """
        top = SCREW_TOP - SCREW_PITCH / 2
        painter.setPen(QPen(QColor(PANEL_INK), 3))
        painter.setBrush(QColor("#2f3338"))
        painter.drawRect(QRectF(878, top - 6, 150, SCREW_PITCH * len(SCREW_ROWS) + 30))
        for index, (label, name) in enumerate(SCREW_ROWS):
            y = SCREW_TOP + index * SCREW_PITCH
            mark = self.marks.get(name) if name else None
            if mark:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(mark[1]))
                painter.drawRect(QRectF(880, y - SCREW_PITCH / 2, 146, SCREW_PITCH))
            painter.setPen(QPen(QColor("#9aa1a8"), 2))
            painter.setBrush(QColor("#4a4f55"))
            painter.drawRect(QRectF(902, y - 14, 50, 28))  # the wire opening
            painter.setBrush(QColor("#b9bec4"))
            painter.drawEllipse(QPointF(976, y), 13, 13)  # the screw
            painter.drawLine(QPointF(967, y - 9), QPointF(985, y + 9))
            painter.setFont(self._font(self.font(), 17, mark is not None))
            painter.setPen(QColor(mark[1]) if mark else QColor(PANEL_INK))
            painter.drawText(QRectF(1045, y - 13, 210, 26), Qt.AlignLeft | Qt.AlignVCenter,
                             label)
            if mark:
                self._draw_badge(painter, QPointF(848, y), mark)


class CheckLabel(QLabel):
    """A wrapping label that ticks its checkbox when clicked, as a checkbox's own text does."""

    def __init__(self, text, box):
        """Show text beside a checkbox.

        Args: text is the sentence; box is the QCheckBox it belongs to.
        Returns: None.
        """
        super().__init__(text)
        self.box = box
        self.setWordWrap(True)
        self.setBuddy(box)

    def mouseReleaseEvent(self, event):
        """Toggle the checkbox on a click.

        Args: event is the Qt mouse event.
        Returns: None.
        """
        self.box.toggle()
        super().mouseReleaseEvent(event)


class DaqChecklistDialog(QDialog):
    """The safety checklist the user ticks off before drive signals reach a real device.

    "Allow drive signals" stays disabled until every box is ticked. It is
    shown once per connection, and again when the outputs are reassigned.
    """

    def __init__(self, parent, settings):
        """List the checks for these settings' output assignment.

        Args: parent is the DaqDialog; settings is the daq.DaqSettings about
            to be applied.
        Returns: None.
        """
        super().__init__(parent)
        self.setWindowTitle("Before the DAQ drives anything")
        self.resize(ui_scale.px(520), ui_scale.px(380))
        intro = QLabel("Tick off each check for the outputs you are switching on. Until "
                       "you do, ESPI live sends no drive signal to this device; this is "
                       "asked once per connection, and again for an output switched on or "
                       "reassigned later.")
        intro.setWordWrap(True)
        self.boxes, self.labels = [], []
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        for text in daq.checklist_items(settings):  # long sentences wrap beside each box
            box = QCheckBox()
            box.toggled.connect(self._update)
            label = CheckLabel(text, box)
            row = QHBoxLayout()
            row.addWidget(box, 0, Qt.AlignTop)
            row.addWidget(label, 1)
            layout.addLayout(row)
            self.boxes.append(box)
            self.labels.append(label)
        layout.addStretch(1)
        buttons = QDialogButtonBox()
        self.allow_button = buttons.addButton("Allow drive signals",
                                              QDialogButtonBox.AcceptRole)
        cancel = buttons.addButton(QDialogButtonBox.Cancel)
        for button in buttons.buttons():
            button.setAutoDefault(False)
        buttons.accepted.connect(self.accept)
        cancel.clicked.connect(self.reject)
        layout.addWidget(buttons)
        self._update()

    def _update(self, *_):
        """Enable "Allow drive signals" once every box is ticked.

        Args: _ is the toggled state (ignored).
        Returns: None.
        """
        self.allow_button.setEnabled(all(box.isChecked() for box in self.boxes))


class DaqWiringDialog(QDialog):
    """Show where each cable goes, for the DAQ window's current settings."""

    def __init__(self, parent):
        """Build the drawing, the numbered list and the safety notes.

        Args: parent is the DaqDialog.
        Returns: None.
        """
        super().__init__(parent)
        self.setWindowTitle(WIRING_TITLE)
        room = parent.screen().availableGeometry()  # a large interface size must fit
        self.resize(min(ui_scale.px(920), room.width() * 9 // 10),
                    min(ui_scale.px(720), room.height() * 9 // 10))
        self.panel = DaqPanelView(self)
        self.legend = QLabel()
        self.legend.setWordWrap(True)
        self.legend.setTextFormat(Qt.RichText)
        self.legend.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.device_note = QLabel()
        self.device_note.setWordWrap(True)
        self.device_note.setStyleSheet("color: gray;")
        source = QLabel("Drawn from NI's “NI USB-6251 BNC Front Panel and Pinout” figure.")
        source.setWordWrap(True)
        source.setStyleSheet("color: gray;")
        safety = QLabel(
            "<b>Never connect an MDT694A's OUTPUT (up to 150 V) to the DAQ.</b> "
            "Switch each MDT694A on after ESPI live is running and the device is "
            "connected, and off before you unplug the DAQ: its outputs are undefined "
            "for a moment while it powers up. The Wavetek function generator is not "
            "needed in the piezo chain: the DAQ makes the triangle.")
        safety.setWordWrap(True)
        safety.setStyleSheet("color:#c00000;")
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        side = QVBoxLayout()
        side.addWidget(self.legend)
        side.addWidget(self.device_note)
        side.addWidget(safety)
        side.addStretch(1)
        side.addWidget(source)
        side.addWidget(buttons)
        layout = QHBoxLayout(self)
        layout.addWidget(self.panel, 3)
        layout.addLayout(side, 2)

    def show_settings(self, settings, device=None):
        """Mark the sockets and list the cables for these settings.

        Args: settings is daq.DaqSettings; device is the connected
            daq.DeviceInfo, or None.
        Returns: None.
        """
        connections = daq.wiring(settings)
        self.panel.set_connections(connections)
        lines = []
        for number, item in enumerate(connections, 1):
            colour = ROLE_COLORS[item.role]
            where = {"bnc": "", "screw": " (screw terminal)"}.get(
                panel_place(item.terminal), " (not on the USB-6251 BNC)")
            lines.append(f'<p><span style="color:{colour}; font-weight:bold;">'
                         f"{number}. {item.label}{where}</span> → {item.target} "
                         f"<i>({item.role.lower()})</i><br>{item.note}</p>")
        assigned = [daq.pget(settings, index, "output") for index in daq.PIEZOS
                    if daq.piezo_assigned(settings, index)]
        if len(assigned) > 1:
            lines.append("<p><b>Two Thorlabs amplifiers:</b> Thor 1 is on "
                         f"{daq.output_label(assigned[0])} and Thor 2 on "
                         f"{daq.output_label(assigned[1])}. The frequency generator "
                         "then needs a counter output (or must be off): the "
                         "USB-6251 BNC has only two analog outputs.</p>")
        self.legend.setText("".join(lines) or
                            "<p>Nothing to plug in yet: assign a piezo driver or the "
                            "frequency generator to an output.</p>")
        model = device.model if device is not None else ""
        if device is None or device.simulated:
            self.device_note.setText("Drawn for the USB-6251 BNC. The simulated device "
                                     "has no sockets: this is where the real one's are.")
        elif "6251" in model:
            self.device_note.setText(f"Connected: {device.label}.")
        else:
            self.device_note.setText(f"Connected: {device.label}. The drawing shows the "
                                     "USB-6251 BNC; this device's sockets may differ.")


class StatusLayout(QLayout):
    """Lay out the status bar in one row when it fits, and in more when not.

    The first item is the message area (tip, view hint, notices). It always
    keeps at least ``message_min`` of the first row and takes whatever the
    other items leave. The other items follow in order, as many as fit on that
    row beside it; the rest flow onto further rows, each as tall as its own
    tallest item. Rows are right-aligned. Nothing is ever cut short to make
    everything fit on one row, and a row only costs the height it needs. The
    ``separator`` is dropped when it would begin or end a row.
    """

    GAP = 6  # between items on a row
    ROW_GAP = 2  # between rows

    def __init__(self, parent=None, message_min=0):
        """Start an empty layout.

        Args: parent is the widget that owns it; message_min is the least
            width the message area may have on a shared row.
        Returns: None.
        """
        super().__init__(parent)
        self._items = []
        self.message_min = message_min
        self.separator = None  # a widget dropped at the edge of a row

    def addItem(self, item):
        """Append a layout item (QLayout calls this from addWidget).

        Args: item is the QLayoutItem.
        Returns: None.
        """
        self._items.append(item)

    def count(self):
        """Count the items, hidden ones included.

        Args: None.
        Returns: The number of items.
        """
        return len(self._items)

    def itemAt(self, index):
        """Return one item.

        Args: index is its position.
        Returns: The QLayoutItem, or None past the end.
        """
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        """Remove and return one item.

        Args: index is its position.
        Returns: The QLayoutItem, or None past the end.
        """
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def hasHeightForWidth(self):
        """Say that the height depends on the width (the number of rows does).

        Args: None.
        Returns: True.
        """
        return True

    def heightForWidth(self, width):
        """Return the height the items need at a width.

        Args: width is the available width in pixels.
        Returns: The total height in pixels.
        """
        return self._plan(width)[1]

    def expandingDirections(self):
        """Say that the layout takes no more room than it is given.

        Args: None.
        Returns: No orientations.
        """
        return Qt.Orientation(0)

    def sizeHint(self):
        """Return the size that puts every item on one row.

        Args: None.
        Returns: A QSize.
        """
        items = [item for item in self._items if not item.isEmpty()]
        width = sum(item.sizeHint().width() for item in items) + self.GAP * (len(items) - 1)
        return QSize(width, self._row_height(items))

    def minimumSize(self):
        """Return the least size: any width, because rows can multiply.

        Args: None.
        Returns: A QSize one row tall.
        """
        return QSize(0, self._row_height([item for item in self._items
                                          if not item.isEmpty()]))

    def setGeometry(self, rect):
        """Place the items and make the host exactly as tall as the rows.

        The status bar sizes itself from its widgets' fixed heights, not from
        heightForWidth, so the host's height is set here.

        Args: rect is the area given to the layout.
        Returns: None.
        """
        super().setGeometry(rect)
        placed, height = self._plan(rect.width())
        for item, cell in placed:
            item.setGeometry(cell.translated(rect.topLeft()))
        host = self.parentWidget()
        if host.minimumHeight() != height or host.maximumHeight() != height:
            host.setFixedHeight(height)

    @staticmethod
    def _row_height(items):
        """Return the tallest item's height.

        Args: items are the visible QLayoutItems.
        Returns: The height in pixels (0 for no items).
        """
        return max((item.sizeHint().height() for item in items), default=0)

    def _plan(self, width):
        """Decide where every visible item goes at a width.

        Args: width is the available width in pixels.
        Returns: A list of (QLayoutItem, QRect relative to the layout), and
            the total height.
        """
        message = self._items[0]
        rest = [item for item in self._items[1:] if not item.isEmpty()]
        widths = [min(item.sizeHint().width(), width) for item in rest]
        heights = [item.sizeHint().height() for item in rest]
        # Fill rows left to right. The first row starts with the message area's
        # least width already used, so readouts only share it when they fit.
        rows, used = [[]], self.message_min
        for index in range(len(rest)):
            gap = self.GAP if used else 0
            wide = widths[index]
            if rest[index].widget() is self.separator:
                after = widths[index + 1] if index + 1 < len(rest) else None
                if not rows[-1] or after is None or \
                        used + gap + wide + self.GAP + after > width:
                    rows[-1].append((index, 0))  # would sit at a row's edge: no room
                    continue
            elif used + gap + wide > width and (rows[-1] or len(rows) == 1):
                rows.append([])
                used, gap = 0, 0
            rows[-1].append((index, wide))
            used += gap + wide
        placed, y = [], 0
        for number, row in enumerate(rows):
            shown = [(index, wide) for index, wide in row if wide]
            high = max((heights[index] for index, _ in shown), default=0)
            if number == 0:
                high = max(high, message.sizeHint().height())
            right = width - (sum(wide for _, wide in shown) + self.GAP * max(len(shown) - 1, 0))
            if number == 0:  # the message area takes what the readouts leave
                taken = right - self.GAP if shown else width
                placed.append((message, QRect(0, 0, taken, high)))
            x = right
            for index, wide in row:
                # A dropped item (width 0) is parked past the right edge, where
                # the host clips it: widgets with a fixed width cannot be squeezed to 0.
                left = x if wide else width
                placed.append((rest[index], QRect(left, y + (high - heights[index]) // 2,
                                                  wide, heights[index])))
                x += wide + (self.GAP if wide else 0)
            y += high + self.ROW_GAP
        return placed, y - self.ROW_GAP


class StatusStrip(QWidget):
    """The status bar's whole content: messages on the left, readouts and buttons.

    It sits in the QStatusBar as one permanent widget and wraps onto more rows
    when the window is too narrow for everything on one row (see StatusLayout).
    It also shows the status bar's temporary messages (QStatusBar.showMessage),
    which the bar itself would otherwise draw under this widget.
    """

    def __init__(self, message_min):
        """Create the strip with its message area.

        Args: message_min is the least width the message area may have beside
            the readouts; below that the readouts move to their own rows.
        Returns: None.
        """
        super().__init__()
        self.setAutoFillBackground(True)  # hides what the bar draws under it
        self.flow = StatusLayout(self, message_min)
        self.flow.setContentsMargins(0, 0, 0, 0)
        self.notices = QWidget()  # error report, tip and view hint
        self.notices_row = QHBoxLayout(self.notices)
        self.notices_row.setContentsMargins(0, 0, 0, 0)
        self.notices_row.setSpacing(StatusLayout.GAP)
        self.message_label = QLabel()  # a temporary message replaces the notices
        self.message_label.setVisible(False)
        self.message_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.message_area = QWidget()
        area = QHBoxLayout(self.message_area)
        area.setContentsMargins(0, 0, 0, 0)
        area.addWidget(self.notices)
        area.addWidget(self.message_label)
        self.flow.addWidget(self.message_area)

    def add_notice(self, widget, stretch=0):
        """Add a widget to the message area.

        Args: widget is the QWidget; stretch is its share of spare width.
        Returns: None.
        """
        self.notices_row.addWidget(widget, stretch)

    def add_item(self, widget):
        """Add a readout or button after the message area.

        Args: widget is the QWidget.
        Returns: None.
        """
        self.flow.addWidget(widget)

    def show_message(self, text):
        """Show a temporary status-bar message, or bring the notices back.

        Args: text is the message; "" ends it (QStatusBar.messageChanged).
        Returns: None.
        """
        self.message_label.setText(text)
        self.message_label.setVisible(bool(text))
        self.notices.setVisible(not text)


class FixedLabel(QLabel):
    """A status-bar label whose width never depends on its text.

    Its preferred width is that of the widest of a few sample strings; longer
    text is shortened with an ellipsis instead of growing the label, so the
    items beside it stay where they are. Only a window too narrow for the whole
    bar makes it narrower, and then by the same amount whatever it shows.
    """

    def __init__(self, *samples):
        """Reserve room for the samples.

        Args: samples are the widest strings the label should show in full.
        Returns: None.
        """
        super().__init__()
        self._full = ""
        self._hold = False  # when True, the bar may not shrink this label
        self._grow = False  # when True, text wider than the samples widens the label
        self._samples = samples
        self._sampled = self._reserved = 0
        self._measure()
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)

    def _measure(self):
        """Reserve room for the samples in the label's current font.

        Args: None.
        Returns: None.
        """
        self._sampled = max(self.fontMetrics().horizontalAdvance(text)
                            for text in self._samples) + 2
        self._reserved = self._sampled

    def changeEvent(self, event):
        """Reserve room again when the font changes (the interface size did).

        Args: event is the QEvent.
        Returns: None.
        """
        super().changeEvent(event)
        if event.type() == QEvent.FontChange and hasattr(self, "_samples"):
            self._measure()
            self.updateGeometry()
            self.setText(self._full)

    def hold_full_text(self):
        """Keep the reserved width, so the samples are never shortened.

        Args: None.
        Returns: None.
        """
        self._hold = True

    def grow_to_text(self):
        """Widen the label for text wider than the samples, instead of shortening it.

        The width is then that of the widest sample or the current text,
        whichever is larger, and changes only when the text does.

        Args: None.
        Returns: None.
        """
        self._grow = True

    def sizeHint(self):
        """Return the reserved width, whatever the text.

        Args: None.
        Returns: A QSize.
        """
        return QSize(self._reserved, super().sizeHint().height())

    def minimumSizeHint(self):
        """Allow shrinking to nothing when the window is too narrow.

        A label that holds its text keeps the reserved width instead.

        Args: None.
        Returns: A QSize.
        """
        if self._hold:
            return self.sizeHint()
        return QSize(0, super().minimumSizeHint().height())

    def text(self):
        """Return the full text, not the shortened display.

        Args: None.
        Returns: The string last given to setText.
        """
        return self._full

    def setText(self, text):
        """Show text, shortened with an ellipsis when it exceeds the width.

        Args: text is the string to show.
        Returns: None.
        """
        self._full = text
        if self._grow:
            reserved = max(self._sampled, self.fontMetrics().horizontalAdvance(text) + 2)
            if reserved != self._reserved:
                self._reserved = reserved
                self.updateGeometry()
        super().setText(self.fontMetrics().elidedText(text, Qt.ElideRight, self.width()))

    def resizeEvent(self, event):
        """Shorten the text again for the new width.

        Args: event is the QResizeEvent.
        Returns: None.
        """
        super().resizeEvent(event)
        self.setText(self._full)


class TipLabel(FixedLabel):
    """The help tip. A left click shows the next one; a right click hides it."""

    clicked = Signal()
    dismissed = Signal()

    def __init__(self, *samples):
        """Reserve room for the samples and show that the tip is clickable.

        Args: samples are the widest strings the label should show in full.
        Returns: None.
        """
        super().__init__(*samples)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        """A left click asks for the next tip; a right click hides them.

        Args: event is the Qt mouse event.
        Returns: None.
        """
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        elif event.button() == Qt.RightButton:
            self.dismissed.emit()


class WhatsNewDialog(QDialog):
    """Show the release notes of the update that was just installed."""

    def __init__(self, parent, whats_new):
        """Build a read-only view of the notes with links that open in a browser.

        Args: parent is the main window; whats_new is the updater.WhatsNew.
        Returns: None.
        """
        super().__init__(parent)
        self.setWindowTitle("ESPI live was updated")
        self.resize(640, 540)
        heading = QLabel(f"<b>Updated to build {whats_new.to_build}</b> "
                         f"(from build {whats_new.from_build}). What's new:")
        self.notes = QTextBrowser()
        self.notes.setOpenExternalLinks(True)
        self.notes.setMarkdown(whats_new.markdown())
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(self.notes, 1)
        layout.addWidget(buttons)


class MainThreadCaller(QObject):
    """Run functions on the GUI thread and wait for the result.

    The control server's thread uses this to touch window code: emitting a
    signal from another thread queues the call on the GUI event loop, and
    the caller blocks until the GUI thread has run it.
    """

    _submit = Signal(object)

    def __init__(self, parent=None):
        """Attach to the GUI thread.

        Args: parent is an object owned by the GUI thread.
        Returns: None.
        """
        super().__init__(parent)
        self._submit.connect(self._run)

    def call(self, fn, timeout=control.CALL_TIMEOUT_S):
        """Run fn on the GUI thread and return its result.

        Args: fn is a callable with no arguments; timeout is the maximum
            wait in seconds when calling from another thread.
        Returns: Whatever fn returned. Raises the call's exception, or
            RuntimeError when the GUI thread did not answer in time.
        """
        job = control.Job(fn)
        if self.thread().isCurrentThread():
            self._run(job)
        else:
            self._submit.emit(job)
            if not job.done.wait(timeout):
                raise RuntimeError("the app did not respond in time; is it busy?")
        if job.error is not None:
            raise job.error
        return job.result

    def call_async(self, fn):
        """Queue fn on the GUI thread without waiting.

        Args: fn is a callable with no arguments.
        Returns: None.
        """
        self._submit.emit(control.Job(fn))

    def _run(self, job):
        """Run one job and release its caller.

        Args: job is the control.Job emitted by call() or call_async().
        Returns: None.
        """
        try:
            job.result = job.fn()
        except Exception as exc:  # reported to the waiting control request
            job.error = exc
        finally:
            job.done.set()


class ControlBackend:
    """Answer control-server requests from the window's channels.

    The methods run on the control server's thread. Engine calls
    (update_settings, request, status) are thread-safe by design and run
    directly; anything that touches widgets goes through the caller to the
    GUI thread, which refreshes the controls afterwards.
    """

    def __init__(self, window, caller):
        """Bind to a window.

        Args: window is the MainWindow; caller runs closures on the GUI
            thread.
        Returns: None.
        """
        self.window = window
        self.caller = caller

    def _channels(self):
        """Snapshot the window's channel list.

        Args: None.
        Returns: A list copy, safe to index while the window rebuilds the
            list. Raises ControlError (503) when the app is closing.
        """
        if self.window.closing:
            raise control.ControlError("the app is closing", 503)
        return list(self.window.channels)

    def _channel(self, index):
        """Return one channel.

        Args: index is the channel's position in the window's list.
        Returns: The Channel. Raises ControlError (404) when the index is
            out of range.
        """
        channels = self._channels()
        if not isinstance(index, int) or not 0 <= index < len(channels):
            names = ", ".join(f"{position}: {channel.label}"
                              for position, channel in enumerate(channels))
            raise control.ControlError(
                f"no camera at index {index!r}; cameras: {names or 'none'}", 404)
        return channels[index]

    def channels(self):
        """Describe every channel.

        Args: None.
        Returns: A list of {index, label, serial, demo, active, displayed}.
        """
        return [{"index": index, "label": channel.label,
                 "serial": channel.info.serial, "demo": channel.demo,
                 "active": channel.active,
                 "displayed": channel is self.window.current}
                for index, channel in enumerate(self._channels())]

    def settings(self, index):
        """Return one channel's settings.

        Args: index is the channel's position.
        Returns: The Settings dict.
        """
        channel = self._channel(index)
        engine = channel.engine
        return asdict(engine.settings() if engine is not None
                      else channel.settings)

    def limits(self, index):
        """Return one channel's device limits.

        Args: index is the channel's position.
        Returns: {node name: [low, high]} (lists, for JSON).
        """
        return {name: list(bounds)
                for name, bounds in self._channel(index).limits.items()}

    def status(self, index):
        """Return one channel's engine status.

        Args: index is the channel's position.
        Returns: The engine's status dict, or {"processing": "off"} when
            the channel is not processing.
        """
        engine = self._channel(index).engine
        return engine.status() if engine is not None else {"processing": "off"}

    def apply(self, index, changes):
        """Change one channel's settings.

        Args: index is the channel's position; changes maps Settings field
            names to coerced, range-clamped values.
        Returns: The resulting settings dict. Raises ControlError (409)
            when the channel is not processing.
        """
        channel = self._channel(index)
        engine = channel.engine
        if engine is None:
            raise control.ControlError(
                f"{channel.label} is not processing; run command "
                "processing_on first", 409)
        engine.update_settings(**self._clamp(channel, changes))
        self.caller.call_async(lambda: self.window._control_sync(channel))
        return asdict(engine.settings())

    def _clamp(self, channel, changes):
        """Keep camera values inside the device's reported or default range.

        Args: channel is the target Channel; changes is the coerced change
            dict.
        Returns: The change dict with the camera fields clamped.
        """
        limits = {**DEFAULT_LIMITS, **channel.limits}
        for field, (node, _, _) in profiles.CAMERA_FIELDS.items():
            if field in changes:
                low, high = limits[node]
                changes[field] = min(max(changes[field], low), high)
        return changes

    def command(self, index, name, payload):
        """Run a one-shot command on one channel.

        Args: index is the channel's position; name is a COMMANDS key;
            payload holds validated string arguments (title, notes).
        Returns: A small result dict for the response.
        """
        channel = self._channel(index)
        if name == "display":
            if channel is not self.window.current:
                self.caller.call(lambda: self.window.show_channel(channel))
            return {"displayed": self.window.current.label}
        if name == "processing_on":
            if not channel.active:
                self.caller.call(lambda: self.window._set_processing(channel, True))
            return {"active": channel.active}
        if name == "processing_off":
            if channel is self.window.current:
                raise control.ControlError(
                    f"{channel.label} is displayed; display another camera "
                    "first", 409)
            if channel.active:
                self.caller.call(lambda: self.window._set_processing(channel, False))
            return {"active": channel.active}
        engine = channel.engine
        if engine is None:
            raise control.ControlError(
                f"{channel.label} is not processing; run command "
                "processing_on first", 409)
        if name == "switch_live" and not engine.demo:
            raise control.ControlError(
                "the camera is already live; switch_live applies in demo "
                "mode only", 409)
        if name == "save" and channel.frozen is not None:
            # A frozen display saves its held frame, as the Save button does.
            packet, frozen_at = self.caller.call(
                lambda: (channel.frozen, channel.frozen_at))
            payload = {**payload, "packet": packet, "frozen_at": frozen_at}
        engine.request(name, **payload)
        result = {"requested": name}
        if name == "save" and channel.frozen is not None:
            result["frozen"] = True
        return result

    def log_tail(self, lines):
        """Return the last lines of the run log.

        Args: lines is how many to return.
        Returns: A list of lines (oldest first), possibly empty.
        """
        return control.tail(getattr(self.window.log, "path", None), lines)
class StreamWindow(QWidget):
    """A second window showing one camera's live stream, beside the main one.

    It draws the channel's newest processed frame on its own timer; the main
    window's controls and status bar stay untouched. Closing it only closes
    the window.
    """

    POLL_MS = 33  # about one frame at 30 fps

    def __init__(self, main):
        """Build the window and start polling.

        Args: main is the MainWindow that opened it (for the log, the GL or
            CPU view choice and to be told when this window closes).
        Returns: None.
        """
        super().__init__(None)  # its own top-level window, not a dialog
        self.main = main
        self.channel = None
        self.sequence = None
        self.setWindowTitle("Second stream")
        self.resize(ui_scale.px(480), ui_scale.px(360))
        self.caption = QLabel()
        self.caption.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.caption.setStyleSheet("color:white;background:rgba(0,0,0,55%);"
                                   f"padding:2px 8px;border-radius:3px;"
                                   f"font-size:{ui_scale.px(11)}px;")
        self.caption.setVisible(False)
        self.view = GLImageView() if main.using_gl else CpuImageView()
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.view, 0, 0)
        layout.addWidget(self.caption, 0, 0, Qt.AlignTop | Qt.AlignLeft)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(self.POLL_MS)

    def set_channel(self, channel):
        """Show another camera's stream.

        Args: channel is a Channel with a running engine, or None to show
            nothing.
        Returns: None.
        """
        if channel is self.channel:
            return
        self.channel = channel
        self.sequence = None
        self.view.clear()
        label = f"{channel.label} (second stream)" if channel is not None \
            else "Second stream"
        self.setWindowTitle(label)
        self.caption.setText(channel.label if channel is not None else "")
        self.caption.setVisible(channel is not None)

    def poll(self):
        """Draw the channel's newest frame.

        Args: None.
        Returns: None.
        """
        packet = self.channel.engine.latest_packet() \
            if self.channel is not None and self.channel.engine is not None else None
        if packet is None or packet.sequence == self.sequence:
            return
        self.sequence = packet.sequence
        table = display_rgb_table(packet.gamma)
        self.view.show_packet(packet, table, round(packet.gamma, 4))

    def closeEvent(self, event):
        """Tell the main window this window is gone.

        Args: event is the Qt close event.
        Returns: None.
        """
        self.main.stream_window = None
        super().closeEvent(event)

    # ----- NI DAQ ------------------------------------------------------------

    DAQ_TIMEOUT_S = 10.0  # longest wait for the DAQ thread to answer one action
    SELF_TEST_TIMEOUT_S = 30.0  # the self-test steps every analog output and reads it back

    def _daq_controller(self):
        """Return the window's DAQ controller, creating it on the GUI thread.

        Args: None.
        Returns: The daq.DaqController. Raises ControlError (503) when the
            app is closing.
        """
        if self.window.closing:
            raise control.ControlError("the app is closing", 503)
        return self.caller.call(self.window._ensure_daq)

    def _daq_state(self):
        """Snapshot the controller, its status and the saved settings.

        Args: None.
        Returns: (controller, status dict, saved DaqSettings); the status
            dict is controller.status()'s (device, active, summary, settings)
            and the saved settings come from the GUI thread's QSettings.
        """
        controller = self._daq_controller()
        saved = self.caller.call(DaqDialog._saved_settings)
        return controller, controller.status(), saved

    def _daq_base(self, state, saved):
        """The settings the next DAQ apply or step starts from.

        Args: state is controller.status()'s dict; saved is the saved
            DaqSettings.
        Returns: The settings last written to the device, or the saved ones
            with the device filled in and both roles off (nothing drives
            from a bare connection).
        """
        if state["settings"] is not None:
            return state["settings"]
        device = state["device"]
        return replace(saved, device=device.name if device is not None else "",
                       generator_on=False, piezo_on=False)

    @staticmethod
    def _daq_device_dict(info):
        """Describe a device for a reply.

        Args: info is a daq.DeviceInfo.
        Returns: A dict.
        """
        return {"name": info.name, "model": info.model, "serial": info.serial,
                "label": info.label, "simulated": info.simulated,
                "ni_simulated": info.ni_simulated}

    @staticmethod
    def _daq_connection(connection):
        """Describe one cable of daq.wiring for a reply.

        Args: connection is a daq.Connection.
        Returns: A dict.
        """
        return {"terminal": connection.terminal, "label": connection.label,
                "role": connection.role, "target": connection.target,
                "note": connection.note, "bnc": connection.bnc}

    def daq(self):
        """Describe the NI DAQ for GET /daq.

        Args: None.
        Returns: A dict: the device, its state, the settings last applied,
            the saved ones, the checklist, the wiring and auto-tune.
        """
        controller, state, saved = self._daq_state()
        device = state["device"]
        base = self._daq_base(state, saved)
        tuning, note = self.caller.call(
            lambda: (self.window.piezo_tuner is not None, self.window.tune_text()))
        return {
            "connected": device is not None,
            "device": self._daq_device_dict(device) if device is not None else None,
            "active": state["active"],
            "summary": state["summary"],
            "settings": state["settings"].to_dict() if state["settings"] is not None
            else None,
            "saved": saved.to_dict(),
            "checklist_done": controller.checklist_done(),
            "problems": daq.problems(base),
            "wiring": [self._daq_connection(item) for item in daq.wiring(base)],
            "tuning": {"on": tuning, "note": note},
        }

    def daq_summary(self):
        """Summarize the NI DAQ for GET /status.

        Args: None.
        Returns: A small dict, or {"connected": False} when no device is
            connected.
        """
        controller, state, _ = self._daq_state()
        device = state["device"]
        return {"connected": device is not None,
                "device": device.label if device is not None else None,
                "active": state["active"], "summary": state["summary"],
                "checklist_done": controller.checklist_done()}

    def daq_action(self, action, **arguments):
        """Run one DAQ action for POST /daq.

        Args: action is a control.DAQ_ACTIONS key; arguments are its
            coerced values (device, settings, volts, camera_index, on).
        Returns: The action's result dict. Raises ControlError when the
            action cannot be carried out.
        """
        runners = {"scan": self._daq_scan, "connect": self._daq_connect,
                   "apply": self._daq_apply, "zero": self._daq_zero,
                   "self_test": self._daq_self_test, "step": self._daq_step,
                   "attach": self._daq_attach, "detach": self._daq_detach,
                   "checklist": self._daq_checklist, "tune": self._daq_tune}
        return runners[action](**arguments)

    def _dispatch_daq_event(self, kind, payload):
        """Run one DAQ event's window side effects from the control server.

        Runs each part on the GUI thread, exactly as poll_daq would: the
        window follows the piezo and the demo cameras' drive, and an open,
        visible DAQ window shows the event. Unlike poll_daq, a device list
        is never kept back: the control request itself is waiting for it.

        Args: kind and payload are the controller's event.
        Returns: None.
        """
        def run():
            """Dispatch one event on the GUI thread."""
            self.window._daq_event(kind, payload)
            dialog = self.window.daq_dialog
            if dialog is not None and dialog.isVisible():
                dialog.handle(kind, payload)

        self.caller.call(run)

    def _daq_flush(self, controller):
        """Dispatch the events queued after an answer, without waiting.

        Args: controller is the daq.DaqController.
        Returns: None.
        """
        pending = []
        while True:
            try:
                pending.append(controller.events.get_nowait())
            except queue.Empty:
                break
        for kind, payload in pending:
            self._dispatch_daq_event(kind, payload)

    def _daq_roundtrip(self, action, kinds, timeout, **payload):
        """Send one DAQ request and wait for its answer, dispatching the events.

        Runs on the control server's thread. It holds the window's
        daq_events_lock, so poll_daq leaves the events queue alone; every
        event's side effects still run, on the GUI thread.

        Args: action is a DaqController.request action; kinds are the event
            kinds that end the wait; timeout is in seconds; payload goes to
            request().
        Returns: (kind, payload) of the event that ended the wait. Raises
            ControlError (504) when the DAQ does not answer in time, and
            ControlError when it reports a failure.
        """
        controller = self._daq_controller()
        with self.window.daq_events_lock:
            controller.request(action, **payload)
            deadline = time.monotonic() + timeout
            while True:
                controller.heartbeat()
                try:
                    kind, event = controller.events.get(timeout=0.1)
                except queue.Empty:
                    if time.monotonic() >= deadline:
                        raise control.ControlError(
                            f"the DAQ did not answer within {timeout:g} s", 504) \
                            from None
                    continue
                self._dispatch_daq_event(kind, event)
                if kind in kinds:
                    self._daq_flush(controller)
                    return kind, event
                if kind == "failed":
                    raise control.ControlError(f"the DAQ failed: {event}") \
                        from None

    def _daq_scan(self):
        """List the NI devices.

        Args: None.
        Returns: The reply fields: devices (with the simulated one last) and
            reason (why none could be listed).
        """
        _, (devices, reason) = self._daq_roundtrip("scan", ("devices",),
                                                   self.DAQ_TIMEOUT_S)
        names = ", ".join(info.name for info in devices) or "none"
        self.window.log.write(f"Control: NI DAQ scan: {names}"
                              + (f"; {reason}" if reason else ""))
        return {"devices": [self._daq_device_dict(info) for info in devices]
                + [self._daq_device_dict(daq.SIMULATED_DEVICE)], "reason": reason}

    def _daq_connect(self, name):
        """Connect to a device; every output goes to 0 V.

        Args: name is a DAQmx device name or "Simulated".
        Returns: The reply fields: the device and its outputs.
        """
        _, outputs = self._daq_roundtrip("connect", ("connected",),
                                         self.DAQ_TIMEOUT_S, device=name)
        self.window.log.write(f"Control: NI DAQ connected to {outputs.device.label}; "
                              "every output is at 0 V.")
        return {"device": self._daq_device_dict(outputs.device),
                "analog": list(outputs.analog), "counters": list(outputs.counters)}

    @staticmethod
    def _daq_changed_text(base, merged):
        """Describe what a DAQ apply changes, for the log.

        Args: base is the DaqSettings the changes start from; merged is
            what will be applied.
        Returns: Text such as "piezo_on: off → on, piezo_mode: hold →
            triangle (2 changed)", or "no settings changed".
        """
        before, after = base.to_dict(), merged.to_dict()
        changed = [name for name in after if after[name] != before[name]]
        if not changed:
            return "no settings changed"
        text = ", ".join(f"{name}: {before[name]} → {after[name]}"
                         for name in changed)
        return f"{text} ({len(changed)} changed)"

    def _daq_apply(self, changes):
        """Change the DAQ settings and write them to the device.

        Args: changes is a dict of DaqSettings fields, laid over the
            settings last applied (or the saved ones).
        Returns: The reply fields: what the outputs now do. Raises
            ControlError for a rejected setting or a checklist not yet
            ticked off.
        """
        controller, state, saved = self._daq_state()
        device = state["device"]
        if device is None:
            raise control.ControlError(
                "connect to a NI device first (the connect action)", 409)
        base = self._daq_base(state, saved)
        candidate = replace(base, **changes)
        merged = replace(candidate, device=device.name)
        found = daq.problems(merged)
        if found:
            raise control.ControlError("the DAQ refused the settings: "
                                       + " ".join(found), 400)
        changed = self._daq_changed_text(base, merged)
        if daq.drives(merged) and not controller.is_armed(merged):
            raise control.ControlError(
                daq.CHECKLIST_REQUIRED + ' Run the "checklist" action to show '
                "it on the app's screen.", 409)
        if self.window.piezo_tuner is not None:  # new settings replace what it tunes
            self.caller.call(lambda: self.window.set_auto_tune(
                False, "new settings were applied"))
        # Saved first, as the DAQ window's Apply does: the settings are what
        # you want even while the checklist is still waiting.
        self.caller.call(lambda: QSettings("ESPI", "espi_live").setValue(
            DAQ_SETTINGS_KEY, json.dumps(merged.to_dict())))
        kind, plan = self._daq_roundtrip("apply", ("applied", "rejected"),
                                         self.DAQ_TIMEOUT_S, settings=merged)
        if kind == "rejected":
            raise control.ControlError("the DAQ rejected the settings: "
                                       + " ".join(plan), 400)
        self.window.log.write(f"Control: NI DAQ apply: {changed}. Now: {plan.summary}.")
        return {"summary": plan.summary, "active": daq.drives(merged),
                "settings": merged.to_dict(), "drive": asdict(plan.drive),
                "frequency_hz": plan.frequency_hz,
                "piezo_frequency_hz": plan.piezo_frequency_hz}

    def _daq_zero(self):
        """Set every output to 0 V.

        Args: None.
        Returns: The reply fields (none). Raises ControlError (409) when no
            device is connected.
        """
        _, state, _ = self._daq_state()
        if state["device"] is None:
            raise control.ControlError(
                "no NI device is connected (the connect action)", 409)
        self._daq_roundtrip("zero", ("zeroed",), self.DAQ_TIMEOUT_S)
        self.window.log.write("Control: NI DAQ: every output set to 0 V.")
        return {}

    def _daq_self_test(self):
        """Step each analog output and read it back.

        Args: None.
        Returns: The reply fields: passed and the result lines. Raises
            ControlError (409) when no device is connected.
        """
        _, state, _ = self._daq_state()
        if state["device"] is None:
            raise control.ControlError(
                "no NI device is connected (the connect action)", 409)
        _, (passed, lines) = self._daq_roundtrip("self_test", ("self_test",),
                                                 self.SELF_TEST_TIMEOUT_S)
        outcome = "passed." if passed else "FAILED: " + "; ".join(lines)
        self.window.log.write(f"Control: NI DAQ self-test: {outcome}")
        return {"passed": passed, "lines": lines}

    def _daq_step(self, volts):
        """Hold the piezo at a level, as a phase step does.

        Args: volts is the level at the DAQ.
        Returns: The reply fields: the level and what was written. Raises
            ControlError when no piezo is assigned or the DAQ refuses the
            level.
        """
        controller, state, saved = self._daq_state()
        device = state["device"]
        if device is None:
            raise control.ControlError(
                "connect to a NI device first (the connect action)", 409)
        base = self._daq_base(state, saved)
        if base.piezo_output == daq.OFF:
            raise control.ControlError(
                "assign the piezo driver to an analog output first (apply "
                "with piezo_output)", 400)
        moved = replace(base, piezo_on=True, piezo_mode=daq.HOLD,
                        piezo_voltage_v=float(volts))
        try:
            written = controller.move_piezo(moved, volts, self.DAQ_TIMEOUT_S)
        except daq.PiezoMoveError as exc:
            raise control.ControlError(str(exc), 400) from None
        self._daq_flush(controller)
        self.caller.call_async(lambda: self.window.set_daq_drive(written.drive))
        self.caller.call_async(lambda: self.window._control_daq_form(moved))
        self.window.log.write(f"Control: NI DAQ: piezo held at {float(volts):g} V.")
        return {"volts": float(volts), "summary": written.summary}

    def _daq_attach(self, camera_index):
        """Give the piezo to a camera's Phase view.

        Args: camera_index is the channel's position (the service resolved
            it; the displayed camera by default).
        Returns: The reply fields: the camera, now in the Phase view.
            Raises ControlError when no device is connected, the piezo is
            unassigned or the checklist does not cover it.
        """
        controller, state, saved = self._daq_state()
        device = state["device"]
        if device is None:
            raise control.ControlError(
                "connect to a NI device first (the connect action)", 409)
        base = self._daq_base(state, saved)
        if base.piezo_output == daq.OFF:
            raise control.ControlError(
                "assign the piezo driver to an analog output first (apply "
                "with piezo_output)", 400)
        if not controller.is_armed(replace(base, piezo_on=True)):
            raise control.ControlError(
                daq.CHECKLIST_REQUIRED + ' Run the "checklist" action to show '
                "it on the app's screen.", 409)
        label = self.caller.call(lambda: self.window._control_daq_attach(
            camera_index, base, device.simulated))
        self.window.log.write(f"Control: NI DAQ: the piezo now drives camera "
                              f"{label}'s Phase view.")
        return {"camera": label, "mode": "Phase"}

    def _daq_detach(self):
        """Take the piezo away from the Phase view.

        Args: None.
        Returns: The reply fields: whether a stepper was attached.
        """
        had = self.caller.call(lambda: self.window.phase_stepper is not None)
        self.caller.call(lambda: self.window._attach_stepper(None))
        self.window.log.write("Control: NI DAQ: the piezo no longer drives the "
                              "Phase view.")
        return {"detached": had}

    def _daq_checklist(self, changes):
        """Show the safety checklist on the app's screen for these settings.

        Args: changes is a dict of DaqSettings fields laid over the settings
            the next apply would start from, or None for those as they are.
        Returns: The reply fields: shown, the checklist's items and the
            settings they are about. Raises ControlError (409) when no
            device is connected.
        """
        _, state, saved = self._daq_state()
        device = state["device"]
        if device is None:
            raise control.ControlError(
                "connect to a NI device first (the connect action)", 409)
        base = self._daq_base(state, saved)
        merged = replace(base, **(changes or {}), device=device.name)
        items = daq.checklist_items(merged)
        self.caller.call(lambda: self.window._control_daq_checklist(merged))
        self.window.log.write("Control: NI DAQ: the safety checklist was shown on "
                              "the app's screen; ticking it arms the device and "
                              "applies the settings.")
        return {"shown": True, "items": items, "settings": merged.to_dict()}

    def _daq_tune(self, on):
        """Start or stop the experimental piezo auto-tune.

        Args: on is True to start, False to stop.
        Returns: The reply fields: tuning and note (what it is doing, or why
            it cannot).
        """
        self.caller.call(lambda: self.window.set_auto_tune(on))
        tuning, note = self.caller.call(
            lambda: (self.window.piezo_tuner is not None, self.window.tune_text()))
        self.window.log.write("Control: NI DAQ auto-tune "
                              + ("started." if on else "stopped.")
                              + (f" {note}" if note else ""))
        return {"tuning": tuning, "note": note}


class MainWindow(QMainWindow):
    """Viewer, controls, log and status for the displayed camera's engine."""

    def __init__(self, channels, log, use_gl, check_updates=True, reporting=None,
                 profile_store=None):
        """Build the window and start polling the engines.

        Args: channels is a list of Channel, at least one of them active
            (opened, not yet started); the first active one is displayed;
            log is the AppLog; use_gl selects the GPU view instead of the CPU
            fallback; check_updates looks for a newer prebuilt binary at
            startup and again every few hours; reporting is the run's
            Reporting, or None; and the window downloads this build's source
            code if it is missing (otherwise that waits for Open in Codex).
            False skips every update check, including the later ones.
            profile_store is the profiles.ProfileStore the channels were
            restored from (None: one over the app's QSettings); each
            camera's profile and the named presets are saved there.
        Returns: None.
        """
        super().__init__()
        install_tooltip_delay()
        self._size_rules = []  # (setter, 100% sizes): pixel sizes this window fixes itself
        self._style_rules = []  # (widget, style sheet written for 100%)
        apply_app_scale(ui_scale.parse_percent(
            QSettings("ESPI", "espi_live").value(ui_scale.SETTING_KEY, ui_scale.DEFAULT_PERCENT)))
        self.setAttribute(Qt.WA_AlwaysShowToolTips, True)
        self.channels = list(channels)
        self.current = next(channel for channel in self.channels if channel.active)
        self.log = log
        self.profiles = (profile_store if profile_store is not None else
                         profiles.ProfileStore(QSettings("ESPI", "espi_live"), log))
        self.reporting = reporting
        self.report_outcomes = {}  # fingerprint -> latest error_reports.Outcome
        self.fatal_boxes = {}  # fingerprint -> (open "ESPI stopped" box, its message)
        self.simple_ui = setting_on(
            QSettings("ESPI", "espi_live").value(SIMPLE_UI_SETTING_KEY, SIMPLE_UI))
        self._advanced_actions = []  # toolbar actions the simple UI hides
        self._advanced_sections = []  # (heading button, body) the simple UI hides
        self._advanced_rows = []  # _ControlRow entries the simple UI parks
        self.closing = False
        self.updates = updater.UpdateWorker()
        self.update_info = None
        self.announced_build = None  # build number last named in the status bar
        self.update_manual = False  # report "up to date" and errors in dialogs
        self.pending_update = None  # unpacked build to install on exit
        self._periodic_checks = False
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._periodic_update_check)
        self.relaunch_after_exit = False
        self.feedback = feedback.FeedbackWorker()
        self.reports = camera_reports.CameraReportWorker()
        self.report_states = {}  # report key -> camera_reports.ReportState
        self.source = source_code.SourceWorker()  # this build's source, for Codex
        self.source_folder = None  # set once the source copy is ready
        self.codex_when_ready = False  # open Codex when the download finishes
        self.retired_channels = []  # demo channels replaced by real cameras
        self.drawn_sequence = 0
        self.shown_packet = None  # the packet on screen; Freeze holds this one
        self.compare_dialog = None  # ReferenceCompareDialog, built on first use
        self.pixels_dialog = None  # PixelSquareDialog, built on first use
        self.daq = None  # daq.DaqController, made when the DAQ window first opens
        self.piezo_follow_hz = {}  # driver -> a frame-rate frequency change sent, not yet applied
        self.piezo_tuner = None  # step_tuning.StepTuner while auto-tune runs
        self.tune_index = None  # the driver auto-tune tunes, 1 or 2
        self.tune_engine = None  # the engine whose frames it measures
        self.tune_settings = None  # the applied triangle it tunes
        self.tune_sent = None  # (DaqSettings, factor, held) last sent by it
        self.tune_note = ""  # why auto-tune last stopped
        self.daq_drive = daq.Drive()  # what its outputs produce, for demo cameras
        self.daq_dialog = None  # DaqDialog, built on first use
        self.daq_auto = None  # device name startup connects to once the scan ends
        self.daq_events_lock = threading.Lock()  # the control server drains events too
        self.phase_steppers = {1: None, 2: None}  # daq.PiezoStepper per driver
        self.stepper_claims = {}  # Channel -> driver index whose stepper it runs
        self.piezo_pairing = {1: "", 2: ""}  # driver -> camera profile key
        self.phase_stepper = None  # the displayed camera's stepper (from phase_steppers)
        self.view_state = None  # last logged (view size, zoom, image size)
        self.drawn_frames = 0
        self.fps_window_start = time.monotonic()
        self.display_fps = 0.0
        self.hover = (-1, -1)
        self.using_gl = use_gl
        self.view = GLImageView() if use_gl else CpuImageView()
        self.view.hovered.connect(self._set_hover)
        self.view.imageClicked.connect(self._image_clicked)
        self.second_view = GLImageView() if use_gl else CpuImageView()
        self.second_view.hovered.connect(self._set_hover)
        self.second_view.imageClicked.connect(self._second_clicked)
        self.second_view.setVisible(False)
        # Small captions name each pane in split view (on screen only).
        self.view_caption = self._pane_caption()
        self.second_caption = self._pane_caption()
        # The frozen banner floats over the top of the image (same grid cell);
        # mouse events pass through it so zoom, pan and the readout still work.
        self.frozen_banner = QLabel()
        self.frozen_banner.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._styled(self.frozen_banner,
                     FROZEN_STYLE + "padding:4px 12px;margin-top:8px;border-radius:4px;")
        self.frozen_banner.setVisible(False)
        # The piezo debug readout is a strip along the top of the image area,
        # so it never covers the image or dummy mode's watermark; it is on
        # screen only, never in saved frames or recordings.
        self.piezo_debug = QLabel()
        self._styled(self.piezo_debug, PIEZO_DEBUG_STYLE)
        self.piezo_debug.setVisible(False)
        center = QWidget()
        stack = QGridLayout(center)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(0)
        stack.addWidget(self.piezo_debug, 0, 0, 1, 2)
        stack.addWidget(self.view, 1, 0)
        stack.addWidget(self.second_view, 1, 1)
        stack.addWidget(self.view_caption, 1, 0, Qt.AlignTop | Qt.AlignLeft)
        stack.addWidget(self.second_caption, 1, 1, Qt.AlignTop | Qt.AlignLeft)
        stack.addWidget(self.frozen_banner, 1, 0, Qt.AlignTop | Qt.AlignHCenter)
        stack.setColumnStretch(0, 1)
        stack.setColumnStretch(1, 1)
        stack.setRowStretch(1, 1)
        self.setCentralWidget(center)
        self.display_mode = "single"  # or "split" or "overlay": how two streams show
        self.second_channel = None  # the Channel the second pane (or window) shows
        self.second_sequence = None  # (sequences) last drawn for the second stream
        self.stream_window = None  # the second stream window, built on first use
        # The controls dock takes the full window height; the log sits under
        # the image only.
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
        self._build_toolbar()
        self._build_controls()
        self._build_log()
        self._build_status_bar()
        self._build_menus()
        self._sync_scale_controls()
        self._rebuild_cameras()
        self._set_demo(self.current.demo)
        self._track_profiles()
        self.profile_timer = QTimer(self)
        self.profile_timer.setTimerType(Qt.CoarseTimer)
        self.profile_timer.timeout.connect(self._save_profiles)
        self.profile_timer.start(PROFILE_SAVE_MS)
        self.shortcut_filter = ShortcutFilter(self)
        self.tooltip_filter = TooltipFilter(self)
        app = QApplication.instance()
        app.installEventFilter(self.shortcut_filter)
        app.installEventFilter(self.tooltip_filter)

        self._following_screens = False
        self._watched_screen = None
        settings = QSettings("ESPI", "espi_live")
        if not self.restoreGeometry(settings.value("geometry", b"")):
            self.resize(DEFAULT_WIDTH, DEFAULT_HEIGHT)
        self.restoreState(settings.value("windowState", b""), LAYOUT_VERSION)

        self.frame_timer = QTimer(self)
        self.frame_timer.setTimerType(Qt.PreciseTimer)
        self.frame_timer.timeout.connect(self._poll_frame)
        self.frame_timer.start(4)
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._poll_status)
        self.status_timer.start(200)
        # Coarse, and only while tips are showing: nothing per frame, and the
        # slot only changes a label. It never talks to an engine.
        self.tip_timer = QTimer(self)
        self.tip_timer.setTimerType(Qt.CoarseTimer)
        self.tip_timer.setInterval(tips.ROTATE_MS)
        self.tip_timer.timeout.connect(self._rotate_tip)
        if not self.tips.hidden:
            self.tip_timer.start()
        if check_updates and updater.unavailable_reason() is None:
            self._periodic_checks = True
            self.updates.check()
            self._schedule_update_check()
        if check_updates:
            self.source.start()  # downloads this build's source if it is missing
        self._auto_connect_daq()
        self.whats_new = updater.take_whats_new()
        if self.whats_new is not None:
            # After the window is on screen, so the notes appear in front of it.
            QTimer.singleShot(0, self._show_whats_new)

    def showEvent(self, event):
        """Fit the window to its screen and follow that screen from now on.

        Args: event is the QShowEvent.
        Returns: None.
        """
        super().showEvent(event)
        handle = self.windowHandle()
        if not self._following_screens:
            self._following_screens = True
            handle.screenChanged.connect(self._watch_screen)
        self._watch_screen(handle.screen())

    def _watch_screen(self, screen):
        """Refit when the window moves to a screen or that screen's area changes.

        A resolution or scaling change, a moved taskbar or a different monitor
        all change the usable area.

        Args: screen is the QScreen the window is now on.
        Returns: None.
        """
        if self._watched_screen is not None:
            self._watched_screen.availableGeometryChanged.disconnect(self._fit_window)
        self._watched_screen = screen
        screen.availableGeometryChanged.connect(self._fit_window)
        self._fit_window()

    def fit_to_screen(self, *_):
        """Shrink and move the window so all of it is on its screen.

        A saved size from a larger monitor, or a resolution change while the
        program runs, would otherwise leave the image and controls partly off
        screen or under the taskbar. A maximized or full-screen window already
        fits and is left alone.

        Args: _ ignores the new area that availableGeometryChanged passes.
        Returns: None.
        """
        if self.isMaximized() or self.isFullScreen():
            return
        frame = self.frameGeometry()
        inner = self.geometry()
        fitted = fit_rect(frame, self.screen().availableGeometry())
        if fitted == frame:
            return
        # The title bar and borders keep their size; only the inside shrinks.
        self.setGeometry(fitted.x() + inner.x() - frame.x(),
                         fitted.y() + inner.y() - frame.y(),
                         fitted.width() - (frame.width() - inner.width()),
                         fitted.height() - (frame.height() - inner.height()))

    def _sized(self, setter, *lengths):
        """Set a pixel size now, and again whenever the interface size changes.

        Only for widgets that live as long as the window.

        Args: setter is the bound Qt setter (for example widget.setFixedWidth);
            lengths are the arguments it takes, written for 100%.
        Returns: None.
        """
        self._size_rules.append((setter, lengths))
        setter(*[ui_scale.px(length) for length in lengths])

    def _styled(self, widget, style):
        """Style a widget with lengths that follow the interface size.

        Only for widgets that live as long as the window.

        Args: widget is the QWidget; style is its style sheet, written for 100%
            (1px borders stay 1px).
        Returns: None.
        """
        self._style_rules.append((widget, style))
        widget.setStyleSheet(ui_scale.scale_stylesheet(style, ui_scale.percent()))

    def set_ui_scale(self, percent):
        """Resize the whole interface (text, controls, docks, status bar) live.

        The size is remembered for the next run. The controls dock and the log
        keep their share of the window. A size above 100% that would make the
        window's smallest size larger than its screen is stepped down until it
        fits, so the window never leaves the screen.

        Args: percent is the wanted interface size, clamped to the allowed range.
        Returns: None.
        """
        before = ui_scale.percent()
        self._apply_scale(ui_scale.clamp_percent(percent))
        self._limit_scale_to_screen()
        after = ui_scale.percent()
        if after != before:
            QSettings("ESPI", "espi_live").setValue(ui_scale.SETTING_KEY, after)
            self.log.write(f"Interface size: {after}%")
        self.fit_to_screen()
        self._sync_scale_controls()

    def _apply_scale(self, percent):
        """Apply an interface size to Qt, the fixed pixel sizes and the docks.

        Args: percent is an allowed interface size; the current one is a no-op.
        Returns: None.
        """
        before = ui_scale.percent()
        if percent == before:
            return
        ratio = percent / before
        controls_width, log_height = self.controls_dock.width(), self.log_dock.height()
        apply_app_scale(percent)
        self._rescale_widgets()
        self.resizeDocks([self.controls_dock], [round(controls_width * ratio)], Qt.Horizontal)
        self.resizeDocks([self.log_dock], [round(log_height * ratio)], Qt.Vertical)
        self.layout().activate()

    def _fits_screen(self):
        """Say whether the window's smallest size, frame included, fits its screen.

        Args: None.
        Returns: True when it does.
        """
        area = self.screen().availableGeometry()
        smallest = self.minimumSizeHint()
        frame, inner = self.frameGeometry(), self.geometry()
        return (smallest.width() + frame.width() - inner.width() <= area.width()
                and smallest.height() + frame.height() - inner.height() <= area.height())

    def _limit_scale_to_screen(self):
        """Step an interface size above 100% down until the window fits its screen.

        Controls and docks scroll or shrink, but the text and the toolbar have a
        smallest size; on a small screen a large interface size would push the
        window past the edge. 100% and below are never changed here.

        Args: None.
        Returns: None.
        """
        while ui_scale.percent() > ui_scale.DEFAULT_PERCENT and not self._fits_screen():
            self._apply_scale(ui_scale.smaller(ui_scale.percent()))

    def _fit_window(self, *_):
        """Fit the window, and an interface size above 100%, to the screen.

        Args: _ ignores the new area that availableGeometryChanged passes.
        Returns: None.
        """
        before = ui_scale.percent()
        self._limit_scale_to_screen()
        if ui_scale.percent() != before:
            QSettings("ESPI", "espi_live").setValue(ui_scale.SETTING_KEY, ui_scale.percent())
            self.log.write(f"Interface size limited to {ui_scale.percent()}% to fit the screen")
            self._sync_scale_controls()
        self.fit_to_screen()

    def _rescale_widgets(self):
        """Re-apply every pixel size the window fixed itself, for the new size.

        Qt re-measures anything laid out from the font by itself; this covers
        the lengths written down in pixels.

        Args: None.
        Returns: None.
        """
        percent = ui_scale.percent()
        for setter, lengths in self._size_rules:
            setter(*[ui_scale.scaled(length, percent) for length in lengths])
        for widget, style in self._style_rules:
            widget.setStyleSheet(ui_scale.scale_stylesheet(style, percent))
        for row in self.findChildren(SliderSpin):
            row.rescale()
        for button in self._status_buttons():
            _arm_status_button(button, *button.arm)
        for dock in (self.controls_dock, self.log_dock):
            dock.setFont(nudged_font(QApplication.font()))  # else its title keeps the old size
        self._align_sections()
        self._fit_controls_width()
        self._fit_status_bar()

    def _sync_scale_controls(self):
        """Show the interface size in the slider, Reset button and View menu.

        Args: None.
        Returns: None.
        """
        percent = ui_scale.percent()
        self.ui_scale_row.set_value(percent)
        self.ui_scale_reset.setEnabled(percent != ui_scale.DEFAULT_PERCENT)
        self.scale_larger_action.setEnabled(percent < ui_scale.MAX_PERCENT)
        self.scale_smaller_action.setEnabled(percent > ui_scale.MIN_PERCENT)
        self.scale_reset_action.setEnabled(percent != ui_scale.DEFAULT_PERCENT)

    def _scale_slider_changed(self, value):
        """Apply a slider or spin-box change, except while the handle is being dragged.

        The slider itself resizes with the interface, so applying mid-drag would
        move it away from the pointer; the release applies the final value.

        Args: value is the slider's percentage.
        Returns: None.
        """
        if not self.ui_scale_row.slider.isSliderDown():
            self.set_ui_scale(value)

    @property
    def camera_identity(self):
        """The displayed camera's detected model.

        Args: None.
        Returns: A CameraIdentity, or None before the camera reported one.
        """
        return self.current.identity

    @camera_identity.setter
    def camera_identity(self, identity):
        """Set the displayed camera's detected model.

        Args: identity is a CameraIdentity or None.
        Returns: None.
        """
        self.current.identity = identity

    @property
    def engine(self):
        """The displayed camera's engine; every control acts on it.

        Args: None.
        Returns: An Engine.
        """
        return self.current.engine

    # ----- construction ---------------------------------------------------

    def _build_toolbar(self):
        """Create the always-visible toolbar: view mode, recording and actions.

        Args: None.
        Returns: None.
        """
        s = self.engine.settings()
        bar = QToolBar("Actions", self)
        bar.setObjectName("actions")
        bar.setMovable(False)
        bar.setAttribute(Qt.WA_AlwaysShowToolTips, True)
        self.addToolBar(Qt.TopToolBarArea, bar)
        self.actions_bar = bar

        # First, so a narrow window never pushes it into the overflow menu: it
        # is the only way back from the simple UI to the full one.
        self.simple_ui_button = QPushButton("Simple UI")
        self.simple_ui_button.setCheckable(True)
        self.simple_ui_button.setChecked(self.simple_ui)
        set_tooltip(self.simple_ui_button, SIMPLE_UI_TIP)
        self.simple_ui_button.toggled.connect(self.set_simple_ui)
        bar.addWidget(self.simple_ui_button)
        bar.addSeparator()

        self.mode_group = QButtonGroup(self)
        for mode in VIEW_MODES:
            button = QPushButton(mode)
            button.setCheckable(True)
            button.setChecked(mode == s.mode)
            set_tooltip(button, MODE_TIPS[mode])
            self.mode_group.addButton(button)
            action = bar.addWidget(button)
            if mode == "Phase":  # shown only with a piezo (see _show_phase_state)
                self.phase_mode_action = action
        self.mode_group.buttonClicked.connect(
            lambda button: self.engine.update_settings(mode=button.text()))
        self.raw_difference_button = QPushButton("Raw difference")
        self.raw_difference_button.setCheckable(True)
        self.raw_difference_button.setChecked(s.raw_difference)
        set_tooltip(self.raw_difference_button, RAW_DIFFERENCE_TIP)
        self.raw_difference_button.setStyleSheet(
            f"QPushButton:checked {{{RAW_DIFFERENCE_STYLE}}}")
        self.raw_difference_button.toggled.connect(self.set_raw_difference)
        bar.addWidget(self.raw_difference_button)
        self._sync_raw_difference(s.raw_difference)
        bar.addSeparator()

        self.record_button = QPushButton("Record AVI")
        self.record_button.setCheckable(True)
        set_tooltip(self.record_button,
                    "Write the displayed image to an AVI at a steady rate. Changing "
                    "the view or the averaging starts a new file. Frames are skipped "
                    "when encoding falls behind, and the last frame is repeated when "
                    "it is slower.")
        self.record_button.toggled.connect(
            lambda on: self.engine.update_settings(record=on))
        bar.addWidget(self.record_button)
        bar.addSeparator()
        self.freeze_button = QPushButton("Freeze (Space)")
        self.freeze_button.setCheckable(True)
        set_tooltip(self.freeze_button,
                    "Hold the displayed image to inspect it (zoom, pan, read pixel "
                    "values) without saving anything. The camera keeps acquiring, "
                    "processing and recording. Shortcut: Space. S saves the held image.")
        self.freeze_button.setStyleSheet(f"QPushButton:checked {{{FROZEN_STYLE}}}")
        self.freeze_button.toggled.connect(self.set_frozen)
        self._mark_advanced(bar.addWidget(self.freeze_button))
        for label, tip, handler, basic in (
                ("Save frame (S)",
                 "Save the displayed frame as a PNG, plus a notes file when the title "
                 "or notes box is filled in. Shortcut: S.",
                 self.save_frame, True),
                ("Set reference (F)",
                 "Use the current live frame as the Reference view's base, or the "
                 "average of the next Ref. frames frames. Shortcut: F.",
                 lambda: self.engine.request("reference"), True),
                ("Compare (C)",
                 "Open a window with three live images side by side. In Previous "
                 "view: the earlier frame being subtracted, the current frame and "
                 "their difference. In Reference view: the stored reference, the "
                 "current frame and their difference. It has its own Freeze and "
                 "saves all three. Shortcut: C.",
                 self.show_compare, True),
                ("Pixels (P)",
                 "Open a window with a small square of the image (1×1 up to 64×64 "
                 "pixels) enlarged, each pixel's value written on it: the raw "
                 "difference in camera levels, so 0 means no change. Click the "
                 "image to move the square. Shortcut: P.",
                 self.show_pixels, True),
                ("Reset last (R)",
                 "Restart the Previous view's difference from the next frame. Shortcut: R.",
                 lambda: self.engine.request("reset_previous"), False),
                ("Reset average (A)",
                 "Discard the frames averaged so far and start the average again. "
                 "Shortcut: A.",
                 lambda: self.engine.request("reset_average"), False)):
            button = QPushButton(label)
            set_tooltip(button, tip)
            button.clicked.connect(handler)
            action = bar.addWidget(button)
            if not basic:
                self._mark_advanced(action)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)

        self.zoom_slider = WheelIgnoringSlider(Qt.Horizontal)
        self.zoom_slider.setRange(0, ZOOM_TICKS)
        # The wheel no longer moves this slider. Arrow keys still step a third
        # of a 1.25x zoom step, the size a wheel notch used to scroll in three.
        self.zoom_slider.setSingleStep(round(
            ZOOM_TICKS * math.log(1.25) / math.log(MAX_ZOOM) / 3))
        self.zoom_slider.setPageStep(ZOOM_TICKS // 6)
        self._sized(self.zoom_slider.setFixedWidth, 130)
        self.zoom_slider.setAttribute(Qt.WA_AlwaysShowToolTips, True)
        self.zoom_slider.valueChanged.connect(
            lambda ticks: self.view.set_zoom(MAX_ZOOM ** (ticks / ZOOM_TICKS)))
        self.zoom_label = QLabel()
        self._sized(self.zoom_label.setMinimumWidth, 44)
        set_tooltip(self.zoom_label, "100% means the whole image fits the window.")
        self.view.zoomChanged.connect(self._show_zoom)
        fit_button = QPushButton("Fit")
        set_tooltip(fit_button, "Show the whole image. Shortcut: 0.")
        fit_button.clicked.connect(self.view.fit)
        actual_button = QPushButton("1:1")
        set_tooltip(actual_button, "One camera pixel per screen pixel. Shortcut: 1.")
        actual_button.clicked.connect(self.view.actual_size)
        zoom_icon = QLabel("Zoom")
        self._sized(zoom_icon.setContentsMargins, 0, 0, 4, 0)
        set_tooltip(zoom_icon, "Zoom from the whole image (100%) up to 64 times.")
        for widget in (zoom_icon, self.zoom_slider, self.zoom_label, fit_button,
                       actual_button):
            # Wheel zoom, the 0 and 1 keys and the View menu remain.
            self._mark_advanced(bar.addWidget(widget))
        self._show_zoom(self.view.zoom)

    def _mark_advanced(self, action):
        """Register a toolbar item the simple UI hides, and show or hide it now.

        Args: action is the QAction QToolBar.addWidget returned.
        Returns: None.
        """
        self._advanced_actions.append(action)
        action.setVisible(not self.simple_ui)

    def set_simple_ui(self, on):
        """Switch the simple UI on or off while the window runs.

        The choice is remembered for the next run. Hidden controls keep their
        values and wiring, so nothing about the measurement changes.

        Args: on is True to render only the basic controls.
        Returns: None.
        """
        self.simple_ui = bool(on)
        QSettings("ESPI", "espi_live").setValue(SIMPLE_UI_SETTING_KEY, self.simple_ui)
        for action in self._advanced_actions:
            action.setVisible(not self.simple_ui)
        for toggle, body in self._advanced_sections:
            toggle.setVisible(not self.simple_ui)
            body.setVisible(not self.simple_ui and toggle.isChecked())
        for entry in self._advanced_rows:
            if self.simple_ui:
                self._park_row(entry)
            else:
                self._place_row(entry)
        self.tips_action.setVisible(not self.simple_ui)
        self._show_tip()
        self._align_sections()
        self._fit_controls_width()
        # Rows put back are shown by queued events, so measure once they are.
        QTimer.singleShot(0, self._remeasure_controls)
        self.log.write("Simple UI on: only the basic controls are shown."
                       if self.simple_ui else "Simple UI off: every control is shown.")

    def _remeasure_controls(self):
        """Make the controls scroll area re-measure its panel.

        It does not notice rows that came or went in one step, and would squeeze
        the rows into the visible height instead of scrolling.

        Args: None.
        Returns: None.
        """
        self.controls_panel.layout().activate()
        self.controls_scroll.setWidgetResizable(True)

    def _place_row(self, entry):
        """Put a controls row into its section's grid, with its label.

        Args: entry is a ControlRow; placing a row already placed does nothing.
        Returns: None.
        """
        if entry.placed:
            return
        entry.placed = True
        if entry.label is None:
            entry.grid.addWidget(entry.widget, entry.row, 0, 1, 2)
            return
        entry.name = QLabel(entry.label)
        entry.name.setToolTip(entry.widget.toolTip())
        entry.grid.addWidget(entry.name, entry.row, 0)
        entry.grid.addWidget(entry.widget, entry.row, 1)
        self._section_names.append(entry.name)

    def _park_row(self, entry):
        """Take a controls row out of its grid into the hidden holder.

        The widget keeps its own show/hide state, so placing it again restores
        the row as it was.

        Args: entry is a ControlRow; parking a row already parked does nothing.
        Returns: None.
        """
        if not entry.placed:
            return
        entry.placed = False
        entry.grid.removeWidget(entry.widget)
        entry.widget.setParent(self.hidden_controls)
        if entry.name is not None:
            entry.grid.removeWidget(entry.name)
            self._section_names.remove(entry.name)
            entry.name.deleteLater()
            entry.name = None

    def _build_controls(self):
        """Create the controls dock as one aligned grid of settings.

        Args: None.
        Returns: None.
        """
        s = self.engine.settings()
        panel = QWidget()
        panel.setAttribute(Qt.WA_AlwaysShowToolTips, True)
        column = QVBoxLayout(panel)
        self._sized(column.setContentsMargins, 10, 8, 10, 8)

        # The Display drop-down is always shown (it names the active camera);
        # the per-camera Process rows only appear with more than one camera.
        self.cameras_box = QWidget()
        cameras = QVBoxLayout(self.cameras_box)
        self._sized(cameras.setContentsMargins, 0, 0, 0, 6)
        title = QLabel("Cameras")
        title.setStyleSheet("font-weight:bold;")
        cameras.addWidget(title)
        display_row = QHBoxLayout()
        display_row.addWidget(QLabel("Display"))
        self.camera_combo = WheelIgnoringComboBox()
        set_tooltip(self.camera_combo,
                    "Camera shown in the window. The controls below change this "
                    "camera's settings.")
        self.camera_combo.currentIndexChanged.connect(self._camera_chosen)
        display_row.addWidget(self.camera_combo, 1)
        cameras.addLayout(display_row)
        self.process_note = QLabel(
            "Process (keeps acquiring and recording when not displayed)")
        self.process_note.setWordWrap(True)
        self.process_note.setStyleSheet("color: palette(placeholder-text);")
        cameras.addWidget(self.process_note)
        self.camera_rows = QGridLayout()
        self.camera_rows.setColumnStretch(0, 1)
        self._sized(self.camera_rows.setVerticalSpacing, 2)
        cameras.addLayout(self.camera_rows)
        self.camera_checks = []
        self.camera_states = []
        column.addWidget(self.cameras_box)

        # In dummy mode the banner also replaces the camera readback line,
        # which would only say the same thing.
        self.demo_banner = QLabel(
            "DUMMY MODE — synthetic video; exposure, gain and FPS are simulated")
        self.demo_banner.setWordWrap(True)
        self.demo_banner.setAlignment(Qt.AlignCenter)
        self._styled(self.demo_banner,
                     "background:#8b0000;color:white;font-weight:bold;padding:6px;"
                     "border-radius:4px;")
        column.addWidget(self.demo_banner)
        self.switch_live_button = QPushButton("Switch to live capture")
        set_tooltip(self.switch_live_button,
                    "Look again for a FLIR camera or a recognised capture device and "
                    "leave synthetic video. Dummy mode stays on if nothing is found. "
                    "A demo recording stops first so synthetic frames are not written "
                    "into a live file.")
        self.switch_live_button.clicked.connect(
            lambda: self.engine.request("switch_live"))
        column.addWidget(self.switch_live_button)

        # Each section is a heading button over its own grid, so a section can
        # be collapsed without touching the visibility its widgets manage
        # themselves. _align_sections gives every grid the same label column,
        # so labels, sliders and values still line up across sections.
        # Controls outside the basic set are parked in a hidden widget while the
        # simple UI is on: created and wired up, but not rendered.
        self.hidden_controls = QWidget(panel)
        self.hidden_controls.setVisible(False)
        self._section_grids = []
        self._section_names = []  # the label in each labeled row, for alignment
        self._section_toggles = {}  # heading text -> its toggle button
        grid = None
        row = 0

        def heading(text, basic=False):
            """Start a collapsible section.

            Args: text is the section name; clicking it hides or shows the
                section, and the choice is remembered between runs; basic is
                False for a section the simple UI leaves out.
            Returns: None.
            """
            nonlocal grid, row
            toggle = QToolButton()
            toggle.setText(text)
            toggle.setCheckable(True)
            toggle.setChecked(True)
            toggle.setArrowType(Qt.DownArrow)
            toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            toggle.setAutoRaise(True)
            set_tooltip(toggle, f"Show or hide the {text} controls.")
            self._styled(toggle, "QToolButton { font-weight:bold; border:none; "
                         f"padding-top:{0 if not self._section_grids else 8}px; }}")
            body = QWidget()
            grid = QGridLayout(body)
            grid.setContentsMargins(0, 0, 0, 0)
            self._sized(grid.setHorizontalSpacing, 8)
            self._sized(grid.setVerticalSpacing, 4)
            grid.setColumnStretch(1, 1)
            row = 0
            column.addWidget(toggle)
            column.addWidget(body)
            self._section_grids.append(grid)
            self._section_toggles[text] = toggle

            def show_section(on, body=body, toggle=toggle, text=text, basic=basic):
                """Show or hide one section and remember the choice.

                Args: on is True to show the section.
                Returns: None.
                """
                body.setVisible(on and (basic or not self.simple_ui))
                toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
                QSettings("ESPI", "espi_live").setValue(
                    f"{SECTION_SETTING_PREFIX}{text}", not on)
            toggle.toggled.connect(show_section)
            if not basic:
                self._advanced_sections.append((toggle, body))
                toggle.setVisible(not self.simple_ui)
                body.setVisible(not self.simple_ui)

        def add(widget, label=None, basic=False):
            """Add a labeled setting, or a full-width widget when unlabeled.

            Args: widget is the control; label is its name, if any; basic is
                False for a control the simple UI leaves out.
            Returns: None.
            """
            nonlocal row
            entry = ControlRow(widget, label, grid, row)
            row += 1
            if basic:
                self._place_row(entry)
                return
            self._advanced_rows.append(entry)
            if self.simple_ui:
                widget.setParent(self.hidden_controls)
            else:
                self._place_row(entry)

        def note(label, basic=False):
            """Style a secondary, wrapping status line and add it full width.

            Args: label is the QLabel to style; basic is False for a line the
                simple UI leaves out.
            Returns: None.
            """
            label.setWordWrap(True)
            label.setStyleSheet("color: palette(placeholder-text);")
            add(label, basic=basic)

        heading("Camera", basic=True)
        # Whose settings the panel edits; shown when there are several cameras.
        self.profile_label = QLabel()
        self.profile_label.setWordWrap(True)
        self.profile_label.setStyleSheet("font-weight:bold;")
        set_tooltip(self.profile_label, PROFILE_TIP)
        add(self.profile_label, basic=True)
        self.model_label = QLabel()  # detected model; hidden in dummy mode
        self.model_label.setWordWrap(True)
        self.model_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        add(self.model_label, basic=True)
        self.report_label = QLabel()  # camera-report status for unrecognised devices
        self.report_label.setWordWrap(True)
        self.report_label.setTextFormat(Qt.RichText)
        self.report_label.setOpenExternalLinks(True)
        self.report_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        set_tooltip(self.report_label,
                    "Whether a report for this unrecognised camera was filed, and a "
                    "link to the issue when one was.")
        self.report_label.hide()
        add(self.report_label, basic=True)
        self.report_button = QPushButton("Report this camera")
        set_tooltip(self.report_button,
                    "Open GitHub's new-issue page with this camera's details. This "
                    "copy cannot file reports itself.")
        self.report_button.clicked.connect(self._open_report_in_browser)
        self.report_button.hide()
        add(self.report_button, basic=True)
        self.exposure = SliderSpin(*DEFAULT_LIMITS["ExposureTime"], 1, 0,
                                   s.exposure_us, " µs")
        self.exposure.set_description(
            "How long each frame collects light. Raise it when the speckle field "
            "is too dim; lower it when motion blurs or the frame rate falls. A "
            "typical starting point is 8000 µs.")
        self.gain = SliderSpin(*DEFAULT_LIMITS["Gain"], 0.1, 1, s.gain_db, " dB")
        self.gain.set_description(
            "Sensor gain. Raise it only when exposure is already as long as the "
            "frame rate allows: it brightens the image and adds noise. A typical "
            "value is 0 dB.")
        self.fps = SliderSpin(*DEFAULT_LIMITS["AcquisitionFrameRate"], 0.1, 1,
                              s.fps, " fps")
        self.fps.set_description(
            "Requested acquisition rate. The camera may run slower when exposure "
            "or bandwidth does not allow the request; the status bar shows the "
            "rate it actually reaches. A typical value is 30 fps.")
        self.exposure.valueChanged.connect(
            lambda v: self.engine.update_settings(exposure_us=v))
        self.gain.valueChanged.connect(lambda v: self.engine.update_settings(gain_db=v))
        self.fps.valueChanged.connect(lambda v: self.engine.update_settings(fps=v))
        add(self.exposure, "Exposure", basic=True)
        add(self.gain, "Gain", basic=True)
        add(self.fps, "Requested FPS")
        self.simulated_object = WheelIgnoringComboBox()
        self.simulated_object.addItems(simulation.SIMULATED_OBJECTS)
        self.simulated_object.setCurrentText(s.demo_object)
        set_tooltip(self.simulated_object, SIMULATED_OBJECT_TIP)
        self.simulated_object.currentTextChanged.connect(self._choose_simulated_object)
        self.simulated_note = QLabel()
        self.simulated_note.setWordWrap(True)
        self.simulated_note.setStyleSheet("color: palette(placeholder-text);")
        set_tooltip(self.simulated_note, SIMULATED_NOTE_TIP)
        simulated_name = QLabel("Simulated object")
        set_tooltip(simulated_name, SIMULATED_OBJECT_TIP)
        self.simulated_box = QWidget()
        simulated_grid = QGridLayout(self.simulated_box)
        simulated_grid.setContentsMargins(0, 0, 0, 0)
        self._sized(simulated_grid.setHorizontalSpacing, 8)
        self._sized(simulated_grid.setVerticalSpacing, 4)
        simulated_grid.setColumnStretch(1, 1)
        simulated_grid.addWidget(simulated_name, 0, 0)
        simulated_grid.addWidget(self.simulated_object, 0, 1)
        simulated_grid.addWidget(self.simulated_note, 1, 0, 1, 2)
        self.simulated_piezo = QCheckBox("Reference piezo (triangle, decorrelation)")
        self.simulated_piezo.setChecked(s.demo_piezo)
        set_tooltip(self.simulated_piezo, SIMULATED_PIEZO_TIP)
        self.simulated_piezo.toggled.connect(self._choose_simulated_piezo)
        simulated_grid.addWidget(self.simulated_piezo, 2, 0, 1, 2)
        self._show_simulated_note()
        add(self.simulated_box, basic=True)
        self.single_field = QCheckBox("Single field (interlaced camera)")
        set_tooltip(self.single_field,
                    "Show only the even rows of each interlaced frame, doubled to full "
                    "height: removes the 1/60 s offset between the two fields at half "
                    "the vertical detail. Reference and Previous differences compare "
                    "rows woven the same way, so the offset rarely matters for ESPI. "
                    "Changing it clears the reference and the averages. Off by "
                    "default: full frames, as the LabVIEW VI used.")
        self.single_field.setChecked(s.single_field)
        self.single_field.toggled.connect(
            lambda on: self.engine.update_settings(single_field=on))
        self.single_field.hide()
        add(self.single_field)
        self.noise_reduction = SliderSpin(0, MAX_DENOISE_PERCENT, 1, 0,
                                          s.noise_reduction, " %")
        self.noise_reduction.set_description(
            "Settle flickering sensor noise in Raw, Previous and Reference by "
            "blending each camera frame with the ones before it. The speckle "
            "grain stays, so a still speckle image looks almost the same: "
            "watch for less boiling and a trail when something moves. Auto "
            "stretch repaints a quieter difference at full brightness; judge "
            "it with Fixed gain, or by the status bar's Input range. This is "
            "not the Speckle filter (that blurs the grain) or Average N (that "
            "smooths the fringe picture). 0 = off. The number is how far the "
            "control is turned up, not the share of the previous frame kept: "
            "low values already blend several frames, and "
            f"{MAX_DENOISE_PERCENT} trails motion the most, which also softens "
            "Previous-view fringes. Start at 0.")
        self.noise_reduction.valueChanged.connect(
            lambda v: self.engine.update_settings(noise_reduction=v))
        add(self.noise_reduction, "Noise reduction")
        self.noise_note = QLabel(
            "Grain stays. Look for less flicker and a motion trail, not a "
            "smoother still image. Auto stretch hides a quieter difference; "
            "Fixed gain shows it. Not the Speckle filter or Average N.")
        note(self.noise_note)
        self.camera_label = QLabel()
        note(self.camera_label, basic=True)
        self._add_profile_controls(add)

        heading("iPhone", basic=True)
        self.phone_section = (self._section_toggles["iPhone"],
                              self._section_grids[-1].parentWidget())
        self._build_phone_controls(add)

        heading("ESPI measurement")
        self._add_preset_buttons(add)
        self.average = SliderSpin(1, 60, 1, 0, s.average)
        self.average.set_description(
            "How many difference frames are combined to quiet speckle in the "
            "fringes. Use it in Previous or Reference view when one difference "
            "is too speckled to read. 1 turns it off and keeps the full frame "
            "rate. With Average camera frames on (the default) the camera "
            "frames are averaged before the difference instead, which lowers "
            "camera and capture noise by the square root of N; in the Phase "
            "view each phase step averages N frames. Starts at 4; use 1 while "
            "aligning.")
        self.average.valueChanged.connect(
            lambda v: self.engine.update_settings(average=int(v)))
        add(self.average, "Average N")
        self.rolling = QCheckBox("Rolling average (updates every frame)")
        set_tooltip(self.rolling,
            "Off: a block average, one image every N frames, then a fresh start. "
            "Use that when each result should stand alone. On: a running average "
            "that keeps the full frame rate but lags behind motion by about N "
            "frames. Start off for a measurement; turn it on to watch a slow "
            "change continuously.")
        self.rolling.setChecked(s.rolling)
        self.rolling.toggled.connect(lambda on: self.engine.update_settings(rolling=on))
        add(self.rolling)
        self.frames_first = QCheckBox("Average camera frames, then subtract")
        set_tooltip(self.frames_first,
            "Off: Average N averages the difference images. Each difference is "
            "an absolute value, so camera noise never averages to zero; it "
            "settles into a grey floor. On: Average N averages the camera "
            "frames themselves (one image every N frames) and subtracts the "
            "averages, which cuts the noise by the square root of N while a "
            "steady change, such as time-averaged fringes, stays. In Previous "
            "view, Frames back then counts averages. Rolling average does not "
            "apply. Pair it with Ref. frames of N or more in Reference view, so "
            "the reference is as quiet. On by default (the noise pipeline), "
            "with Average N 4 and 16 reference frames; turn it off for the "
            "full frame rate or a rolling average.")
        self.frames_first.setChecked(s.average_frames_first)
        self.frames_first.toggled.connect(self._frames_first_changed)
        add(self.frames_first)
        self.rolling.setEnabled(not s.average_frames_first)
        self.ref_frames = SliderSpin(1, MAX_REFERENCE_FRAMES, 1, 0, s.ref_frames)
        self.ref_frames.set_description(
            "How many camera frames F (Set reference) averages into the reference. "
            "One frame keeps its sensor noise, which then shows in every Reference "
            "difference as a frozen grain that Average N cannot remove; averaging "
            "several frames quiets it. The old reference stays in use until the new "
            "one is complete, and the status bar counts the frames. Use 1 for a fast "
            "camera with little noise; try 8–30 for a noisy or analog camera, with "
            "the object still. Starts at 16 (about half a second at 30 fps); "
            "Reference view captures one automatically when it has none.")
        self.ref_frames.valueChanged.connect(
            lambda v: self.engine.update_settings(ref_frames=int(v)))
        add(self.ref_frames, "Ref. frames")
        self.ref_refresh = SliderSpin(0, 10, 0.1, 1, s.ref_refresh_s, " s")
        self.ref_refresh.set_description(
            "In Reference view, recaptures the baseline this often so slow drift "
            "(air, heat) cancels out of the fringes. A shorter time hides real "
            "motion that is slower than the refresh. 0 keeps one reference until "
            "you press F. Start at 0 for a single before/after; try 2 s when the "
            "setup drifts and the change you care about is faster than that.")
        self.ref_refresh.valueChanged.connect(
            lambda v: self.engine.update_settings(ref_refresh_s=v))
        add(self.ref_refresh, "Ref. refresh")
        self.frames_back = SliderSpin(1, MAX_FRAMES_BACK, 1, 0, s.frames_back)
        self.frames_back.set_description(
            "In Previous view, which earlier frame each one is compared with: 1 "
            "is the frame just before. A larger number spans more time, so "
            "slow changes show up instead of cancelling, as the LabVIEW VI did "
            "by waiting between grabs. Each step back keeps one more frame in "
            "memory. Start at 1.")
        self.frames_back.valueChanged.connect(
            lambda v: self.engine.update_settings(frames_back=int(v)))
        add(self.frames_back, "Frames back")

        heading("Phase shifting")
        self._build_phase_controls(add, note, s)

        heading("Fringe enhancement")
        self.outlier_filter = SliderSpin(0, 15, 1, 0, s.outlier_filter, " px")
        self.outlier_filter.set_description(
            "Pulls isolated bright or dark pixels back to their neighbourhood's "
            "mean, on every difference frame before averaging (NI Vision's "
            "Lowpass filter, from the LabVIEW VI). A pixel is replaced only when "
            "it is further from the mean than the neighbourhood's spread, so "
            "fringes and smooth areas stay untouched. 0 turns it off. Start at 0; "
            "try 3 px when single hot pixels sparkle.")
        self.outlier_filter.valueChanged.connect(
            lambda v: self.engine.update_settings(outlier_filter=int(v)))
        add(self.outlier_filter, "Outlier filter")
        self.smoothing_combo = WheelIgnoringComboBox()
        self.smoothing_combo.addItems(SMOOTHING_METHODS)
        self.smoothing_combo.setCurrentText(s.smoothing_method)
        set_tooltip(self.smoothing_combo,
            "Blurs each difference image so pixel-scale speckle fades and broader "
            "fringes stand out. Oriented (the default) measures which way the "
            "fringes run and smooths along them, not across them, so dense "
            "fringes survive a width that would blur them away with Gaussian; "
            "where no direction shows it acts like Gaussian. Gaussian softens "
            "edges and accepts any width. Median keeps fringe edges sharper and "
            "is capped at 5 px so the frame rate holds. Start with Oriented.")
        self.smoothing_combo.currentTextChanged.connect(
            lambda name: self.engine.update_settings(smoothing_method=name))
        self.smoothing = SliderSpin(0, 31, 1, 0, s.smoothing, " px")
        self.smoothing.set_description(
            "Width of the speckle filter, in pixels. Keep it below the fringe "
            "spacing or the fringes blur away with the speckle (Oriented "
            "tolerates more, because it smooths along the fringes). 0 turns "
            "the filter off. Starts at 11 px; lower it if dense fringes fade.")
        self.smoothing.valueChanged.connect(
            lambda v: self.engine.update_settings(smoothing=int(v)))
        add(self.smoothing_combo, "Speckle filter")
        add(self.smoothing, "Filter width")
        self.scaling = WheelIgnoringComboBox()
        self.scaling.addItems(SCALING_MODES)
        self.scaling.setCurrentText(SCALING_MODES[1] if s.scale_range else SCALING_MODES[0])
        set_tooltip(self.scaling, SCALING_TIP,
                    "Gamma is a separate curve applied afterwards and replaces "
                    "neither. Start with Fixed gain on an analog camera, Auto "
                    "stretch on the Blackfly.")
        self.scaling.currentTextChanged.connect(self._scaling_changed)
        add(self.scaling, "Scaling")
        self.difference_gain = SliderSpin(1, 64, 0.5, 1, s.difference_gain, " ×")
        self.difference_gain.set_description(
            "Fixed gain: each difference is multiplied by this and anything "
            "above 255 shows white, as in the LabVIEW VI (IMAQ Multiply by "
            "Constant). At the VI's default of 42, any difference of 7 grey "
            "levels or more is white. Lower it when noise shows as grey grain; "
            f"raise it when fringes are dim. Start at {DEFAULT_DIFFERENCE_GAIN:g}.")
        self.difference_gain.valueChanged.connect(
            lambda v: self.engine.update_settings(difference_gain=v))
        add(self.difference_gain, "Difference gain")
        self.gain_before_average = QCheckBox("Gain before averaging (LabVIEW order)")
        set_tooltip(self.gain_before_average,
            "Fixed gain only. On: the gain is applied to every difference frame "
            "(saturating at 255) and the outlier filter runs on that; Average N "
            "then averages those frames and multiplies the mean by the Average "
            "multiplier, as the LabVIEW VI did (it sums N + 1 frames and divides "
            "by N, so each block is one frame longer and (N+1)/N brighter). Off: "
            "the raw differences are "
            "averaged and the gain is applied once at the end. Start off.")
        self.gain_before_average.setChecked(s.gain_before_average)
        self.gain_before_average.toggled.connect(self._gain_order_changed)
        add(self.gain_before_average)
        self.average_multiplier = SliderSpin(0.1, 10, 0.1, 1, s.average_multiplier, " ×")
        self.average_multiplier.set_description(
            "With Gain before averaging on and Average N above 1, the mean is "
            "multiplied by this before it is shown (the LabVIEW VI's Multiplier). "
            "As in the VI, N + 1 frames are summed and divided by N. Start at 1.")
        self.average_multiplier.valueChanged.connect(
            lambda v: self.engine.update_settings(average_multiplier=v))
        add(self.average_multiplier, "Average multiplier")
        self.invert = QCheckBox("Invert")
        set_tooltip(self.invert,
            "Swaps bright and dark fringes. Use it when the pattern is easier to "
            "read the other way around. Start off.")
        self.invert.setChecked(s.invert)
        self.invert.toggled.connect(lambda on: self.engine.update_settings(invert=on))
        toggles = QWidget()
        toggles_row = QHBoxLayout(toggles)
        toggles_row.setContentsMargins(0, 0, 0, 0)
        toggles_row.addWidget(self.invert)
        toggles_row.addStretch(1)
        add(toggles)
        self.clip = SliderSpin(0, 10, 0.1, 1, s.clip_percent, " %")
        self.clip.set_description(
            "With Auto stretch, lets this share of the darkest and brightest pixels "
            "saturate so a few stray speckles don't flatten the contrast. 0 uses "
            "the exact minimum and maximum. Start at 0; try 0.5% when speckles "
            "wash the image out.")
        self.clip.valueChanged.connect(
            lambda v: self.engine.update_settings(clip_percent=v))
        add(self.clip, "Stretch clip")
        self.min_range = SliderSpin(0, 128, 1, 0, s.min_range, " lv")
        self.min_range.set_description(
            "Sensitivity floor: with Auto stretch (or Local normalization), white "
            "is never fewer than this "
            "many grey levels of difference. A still scene's camera noise (a few "
            "levels) then stays dark instead of filling the screen, while real "
            "fringes, which change far more, still use the full range. 0 = off. "
            "Use Calibrate noise with the object still to set it.")
        self.min_range.valueChanged.connect(
            lambda v: self.engine.update_settings(min_range=int(v)))
        add(self.min_range, "Sensitivity floor")
        self.threshold = SliderSpin(0, 64, 1, 0, s.threshold, " lv")
        self.threshold.set_description(
            "Noise threshold: right after the subtraction, any pixel whose "
            "difference is below this many grey levels is set to 0 (no change), "
            "before averaging, smoothing, peak hold and the stretch. Hides camera "
            "noise; too high also hides faint fringes. 0 = off. Try 3–8.")
        self.threshold.valueChanged.connect(
            lambda v: self.engine.update_settings(threshold=int(v)))
        add(self.threshold, "Noise threshold")
        self.calibrate_button = QPushButton("Calibrate noise (object still)")
        set_tooltip(self.calibrate_button,
            "In Previous or Reference view, with the object at rest, measures "
            "the difference noise for 2 s and sets the Sensitivity floor to "
            "twice its peak.")
        self.calibrate_button.clicked.connect(
            lambda: self.engine.request("calibrate_noise"))
        add(self.calibrate_button)
        self.floor_subtract = QCheckBox("Subtract noise floor")
        set_tooltip(self.floor_subtract,
            "Takes a measured still-scene level off every pixel after "
            "averaging and the speckle filter, so a still scene shows black "
            "instead of grey grain. Each pixel gets its own level, because "
            "noise grows with brightness. It needs a measurement: with the "
            "object still, press Measure noise floor. The map only fits the "
            "settings it was measured with; after a change (view, reference, "
            "exposure, averaging, filters) the status bar asks for a new one. "
            "Unlike Noise threshold, it keeps faint changes that last across "
            "the average. Start off; use it with Average N 4 or more.")
        self.floor_subtract.setChecked(s.floor_subtract)
        self.floor_subtract.toggled.connect(
            lambda on: self.engine.update_settings(floor_subtract=on))
        add(self.floor_subtract)
        self.floor_button = QPushButton("Measure noise floor (object still)")
        set_tooltip(self.floor_button,
            "In Previous or Reference view, with the object at rest, records "
            "each pixel's difference level for at least 2 s and 16 images. "
            "Subtract noise floor then takes it off. Measure again after "
            "changing the view, the reference or the processing.")
        self.floor_button.clicked.connect(
            lambda: self.engine.request("measure_floor"))
        add(self.floor_button)
        self.steady_range = SliderSpin(0, 3, 0.1, 1, s.steady_range_s, " s")
        self.steady_range.set_description(
            "With Auto stretch, lets the stretch range shrink only over this many "
            "seconds, so the contrast stops pumping from frame to frame. A range "
            "that grows (new fringes) is followed at once. 0 = recompute every "
            "frame. Try 0.5 s.")
        self.steady_range.valueChanged.connect(
            lambda v: self.engine.update_settings(steady_range_s=v))
        add(self.steady_range, "Steady range")
        self.even_brightness = QCheckBox("Even out beam brightness")
        set_tooltip(self.even_brightness,
            "Divides each difference by the local illumination, so the bright "
            "middle of the laser spot no longer dominates and fringes near its "
            "edge stay visible. Unlit areas go black. Start off.")
        self.even_brightness.setChecked(s.even_brightness)
        self.even_brightness.toggled.connect(
            lambda on: self.engine.update_settings(even_brightness=on))
        add(self.even_brightness)
        self.peak_hold = SliderSpin(0, 10, 0.1, 1, s.peak_hold_s, " s")
        self.peak_hold.set_description(
            "Shows each pixel's largest difference in the last this-many seconds "
            "(a sliding window: older peaks drop out). In Previous view a "
            "vibration only shows in the frames where the motion happens to "
            "change; peak hold keeps those moments on screen so the pattern "
            "builds up. 0 = off. Try 3 s.")
        self.peak_hold.valueChanged.connect(
            lambda v: self.engine.update_settings(peak_hold_s=v))
        add(self.peak_hold, "Peak hold window")
        self.normalize_window = SliderSpin(0, 255, 1, 0, s.normalize_window, " px")
        self.normalize_window.set_description(
            "Local normalization, in place of Stretch: each pixel is compared with "
            "the average and spread of the difference within this window, so faint "
            "and strong fringes get the same contrast everywhere. Mid-grey is the "
            "local average. Use a window wider than the fringe spacing; the "
            "Sensitivity floor still keeps camera noise from being boosted. "
            "0 = off. Try 41–81 px.")
        self.normalize_window.valueChanged.connect(
            lambda v: self.engine.update_settings(normalize_window=int(v)))
        add(self.normalize_window, "Local normalization")
        self._sync_scaling_controls(s.scale_range, s.gain_before_average)

        heading("Display")
        self.flip_button = QPushButton("Flip horizontally")
        self.flip_button.setCheckable(True)
        set_tooltip(self.flip_button,
            "Mirrors the image left to right (a flip across the vertical axis) in "
            "every view. Saved frames and recordings are flipped the same way. Use it "
            "when the optics reverse the scene. Start off.")
        self.flip_button.setChecked(s.flip_horizontal)
        self.flip_button.toggled.connect(
            lambda on: self.engine.update_settings(flip_horizontal=on))
        add(self.flip_button)
        self.gamma = SliderSpin(0.1, 3.0, 0.1, 1, s.gamma)
        self.gamma.set_description(GAMMA_TIP + " Start at 1.0.")
        self.gamma.valueChanged.connect(lambda v: self.engine.update_settings(gamma=v))
        self.brightness = SliderSpin(0, 128, 1, 0, s.brightness, " lv")
        self.brightness.set_description(
            "Adds this many grey levels to every pixel in every view, so the whole "
            "image is brighter. Values near white stay white, and black becomes "
            "grey, so contrast drops a little. Saved frames and recordings match "
            "the screen. 0 = off. Try 30–60.")
        self.brightness.valueChanged.connect(
            lambda v: self.engine.update_settings(brightness=int(v)))
        self.display_gain = SliderSpin(1, 8, 0.1, 1, s.display_gain, " ×")
        self.display_gain.set_description(
            "Multiplies every pixel, so bright areas get brighter while black stays "
            "black. Pixels pushed past white saturate. 1 = off. Try 2–3.")
        self.display_gain.valueChanged.connect(
            lambda v: self.engine.update_settings(display_gain=v))
        self.contrast = SliderSpin(0, 100, 1, 0, s.contrast, " %")
        self.contrast.set_description(
            "S-curve contrast: darkens values below mid-grey and brightens those "
            "above it, keeping black and white fixed, so bright features stand out "
            "from a dim background. Very faint features can be pushed toward "
            "black. 0 = off. Try 30–60%.")
        self.contrast.valueChanged.connect(
            lambda v: self.engine.update_settings(contrast=v))
        add(self.gamma, "Gamma")
        add(self.brightness, "Brightness")
        add(self.display_gain, "Gain")
        add(self.contrast, "S-curve contrast")
        self.ui_scale_row = SliderSpin(
            ui_scale.MIN_PERCENT, ui_scale.MAX_PERCENT, ui_scale.STEP_PERCENT, 0,
            ui_scale.percent(), "%")
        self.ui_scale_row.set_description(
            "Size of the whole window: text, buttons, controls, docks and the "
            "status bar. Drag, then release to apply; it takes effect at once and "
            "is remembered. Make it smaller on a small or high-resolution screen, "
            "larger when text is hard to read. 100% is the standard size. "
            "Shortcuts: Ctrl++, Ctrl+- and Ctrl+0.")
        self.ui_scale_row.valueChanged.connect(self._scale_slider_changed)
        self.ui_scale_row.slider.sliderReleased.connect(
            lambda: self.set_ui_scale(self.ui_scale_row.value()))
        add(self.ui_scale_row, "Interface size")
        self.ui_scale_reset = QPushButton(f"Reset interface size to {ui_scale.DEFAULT_PERCENT}%")
        set_tooltip(self.ui_scale_reset,
                    "Return the window to its standard size. Shortcut: Ctrl+0.")
        self.ui_scale_reset.clicked.connect(
            lambda: self.set_ui_scale(ui_scale.DEFAULT_PERCENT))
        add(self.ui_scale_reset)

        heading("Saved frame details")
        self.frame_title = QLineEdit()
        self.frame_title.setPlaceholderText("Title (optional)")
        set_tooltip(self.frame_title,
                    "Optional words added to the saved PNG's filename. Shortcut: S "
                    "saves the frame.")
        self.frame_notes = QPlainTextEdit()
        self.frame_notes.setPlaceholderText("Notes (optional)")
        set_tooltip(self.frame_notes,
                    "Optional notes written beside the PNG when you save. Leave the "
                    "title and the notes blank to save the image only. Shortcut: S.")
        self._sized(self.frame_notes.setFixedHeight, 48)
        add(self.frame_title)
        add(self.frame_notes)

        column.addStretch(1)

        # The panel scrolls vertically when the screen is too short for it, so
        # it never makes the window taller than the screen. The scroll area
        # also gives wrapped notes their full height, which a dock alone does
        # not. Its width leaves room for the scroll bar, so nothing is cut off
        # sideways.
        self.controls_panel = panel
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidget(panel)
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QFrame.NoFrame)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._align_sections()
        self._fit_controls_width()  # measured with every section open
        saved = QSettings("ESPI", "espi_live")
        for text, toggle in self._section_toggles.items():
            if str(saved.value(f"{SECTION_SETTING_PREFIX}{text}", False)).lower() == "true":
                toggle.setChecked(False)
        self.controls_dock = QDockWidget("Controls", self)
        self.controls_dock.setObjectName("controls")
        self.controls_dock.setWidget(self.controls_scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.controls_dock)
        self._tip_dock_buttons(self.controls_dock)

    def _align_sections(self):
        """Give every controls section the same label-column width.

        Runs at build time and whenever the interface size changes, so the
        sliders of all sections start at the same x position.

        Args: None.
        Returns: None.
        """
        width = max((name.sizeHint().width() for name in self._section_names), default=0)
        for grid in self._section_grids:
            grid.setColumnMinimumWidth(0, width)

    def _fit_controls_width(self):
        """Give the controls panel the width its widest row needs, plus the scroll bar.

        Runs at build time and whenever the interface size changes.

        Args: None.
        Returns: None.
        """
        panel = self.controls_panel
        panel.setMinimumWidth(ui_scale.px(320))
        self.controls_scroll.setMinimumWidth(
            panel.minimumSizeHint().expandedTo(panel.minimumSize()).width()
            + self.controls_scroll.verticalScrollBar().sizeHint().width())

    def _add_profile_controls(self, add):
        """Add the saved-preset row and the reset button to the Camera section.

        Args: add places a widget on the next controls-grid row.
        Returns: None.
        """
        self.preset_combo = WheelIgnoringComboBox()
        self.preset_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.preset_combo.setMinimumContentsLength(12)
        set_tooltip(self.preset_combo, PRESET_COMBO_TIP)
        self.preset_combo.currentIndexChanged.connect(self._sync_preset_buttons)
        add(self.preset_combo, "Presets", basic=True)
        buttons = QWidget()
        row = QHBoxLayout(buttons)
        row.setContentsMargins(0, 0, 0, 0)
        self.preset_apply_button = QPushButton("Apply")
        self.preset_save_button = QPushButton("Save as…")
        self.preset_delete_button = QPushButton("Delete")
        for button, tip, handler in (
                (self.preset_apply_button, PRESET_APPLY_TIP, self.apply_saved_preset),
                (self.preset_save_button, PRESET_SAVE_TIP, self.save_preset_as),
                (self.preset_delete_button, PRESET_DELETE_TIP, self.delete_preset)):
            set_tooltip(button, tip)
            button.clicked.connect(lambda _checked=False, handler=handler: handler())
            row.addWidget(button)
        add(buttons, basic=True)
        self.reset_camera_button = QPushButton("Reset camera to defaults…")
        set_tooltip(self.reset_camera_button, RESET_CAMERA_TIP)
        self.reset_camera_button.clicked.connect(self.reset_camera)
        add(self.reset_camera_button, basic=True)
        self._refresh_presets()

    def _refresh_presets(self, select=None):
        """Fill the preset drop-down from the store.

        Args: select is the name to choose (None: keep the current choice
            while it still exists, otherwise choose nothing).
        Returns: None.
        """
        combo = self.preset_combo
        wanted = combo.currentText() if select is None else select
        entries = self.profiles.presets()
        combo.blockSignals(True)
        combo.clear()
        for index, entry in enumerate(entries):
            combo.addItem(entry.name)
            combo.setItemData(index, format_tooltip(
                f"Saved from {entry.source_model or 'an unknown camera'}.",
                PRESET_COMBO_TIP), Qt.ToolTipRole)
        combo.setCurrentIndex(combo.findText(wanted) if wanted else -1)
        combo.setPlaceholderText("Choose a preset" if entries else "No saved presets")
        combo.blockSignals(False)
        self._sync_preset_buttons()

    def _sync_preset_buttons(self, *_):
        """Enable Apply and Delete only while a preset is chosen.

        Args: any arguments of the drop-down's signal are ignored.
        Returns: None.
        """
        chosen = self.preset_combo.currentIndex() >= 0
        self.preset_apply_button.setEnabled(chosen)
        self.preset_delete_button.setEnabled(chosen)

    @staticmethod
    def _source_model(channel):
        """Name a camera's model for a preset saved from it.

        Args: channel is a Channel.
        Returns: The detected model's short name, else the listed model or serial.
        """
        if channel.identity is not None:
            return channel.identity.short_name()
        return channel.info.model or channel.info.serial

    def apply_saved_preset(self, name=None):
        """Apply a saved preset to the displayed camera, and keep it as its profile.

        Camera values are clamped to this camera's limits; each clamp, and
        the result, is logged. No other camera changes.

        Args: name is the preset (None: the one chosen in the drop-down).
        Returns: None.
        """
        name = self.preset_combo.currentText() if name is None else name
        if not name:
            return
        channel = self.current
        profile = self.profiles.load_preset(name, channel.default_profile())
        if profile is None:
            self.log.write(f"Preset {name} is no longer saved.")
            self._refresh_presets()
            return
        settings, notes = profiles.clamp_to_limits(
            profile.settings, {**DEFAULT_LIMITS, **channel.limits})
        for note in notes:
            self.log.write(f"Preset {name}: {note}")
        channel.apply_profile(replace(profile, settings=settings))
        self._load_settings(self.engine.settings(), channel.limits)
        self.profiles.save_profile(channel.profile_key, channel.profile())
        self.log.write(f"Preset applied: {name} to {channel.label} "
                       f"({profiles.summary(channel.profile())})")

    def save_preset_as(self):
        """Ask for a name and save the displayed camera's settings as a preset.

        An existing name (letter case aside) is replaced only after a
        confirmation.

        Args: None.
        Returns: None.
        """
        channel = self.current
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Save preset")
        dialog.setLabelText(f"Save the settings of {channel.label} as:")
        dialog.setTextValue(self.preset_combo.currentText())
        dialog.setOkButtonText("Save")
        dialog.findChild(QLineEdit).setMaxLength(PRESET_NAME_MAX)
        accepted = dialog.exec() == QDialog.Accepted
        name = dialog.textValue().strip()
        dialog.deleteLater()
        if not accepted:
            return
        if not name:
            self.log.write("Preset not saved: it needs a name.")
            return
        existing = self.profiles.find_preset(name)
        if existing is not None and QMessageBox.question(
                self, "Replace preset?",
                f"A preset named “{existing.name}” already exists. Replace it with "
                f"the current settings of {channel.label}?") != QMessageBox.Yes:
            return
        self.profiles.save_preset(name, channel.profile(), self._source_model(channel))
        self._refresh_presets(select=name)
        self.log.write(f"Preset saved: {name} from {channel.label} "
                       f"({profiles.summary(channel.profile())})")

    def delete_preset(self):
        """Delete the chosen saved preset after a confirmation.

        Args: None.
        Returns: None.
        """
        name = self.preset_combo.currentText()
        if not name:
            return
        if QMessageBox.question(
                self, "Delete preset?",
                f"Delete the preset “{name}”? Every camera keeps its current "
                "settings; only the saved preset is removed.") != QMessageBox.Yes:
            return
        self.profiles.delete_preset(name)
        self._refresh_presets()
        self.log.write(f"Preset deleted: {name}")

    def reset_camera(self):
        """Put the displayed camera back to its defaults after a confirmation.

        Its saved profile is removed; other cameras and the presets stay.

        Args: None.
        Returns: None.
        """
        channel = self.current
        defaults = channel.default_profile()
        if QMessageBox.question(
                self, "Reset camera to defaults?",
                f"Put every control of {channel.label} back to its starting values "
                f"({profiles.summary(defaults)}) and clear its saved settings? "
                "Other cameras and saved presets do not change.") != QMessageBox.Yes:
            return
        channel.apply_profile(defaults)
        self._load_settings(self.engine.settings(), channel.limits)
        self.profiles.forget_profile(channel.profile_key, channel.profile())
        self.log.write(f"Camera reset to defaults: {channel.label} "
                       f"({profiles.summary(defaults)})")

    def _track_profiles(self):
        """Count every camera's settings now in force as saved.

        Args: None.
        Returns: None.
        """
        for channel in self.channels:
            self.profiles.track(channel.profile_key, channel.profile())

    def _save_profiles(self, settle=True):
        """Save every camera's profile that changed.

        Args: settle is True (the timer) to wait until a profile stops
            changing; False (closing) writes every change at once.
        Returns: None.
        """
        save = self.profiles.save_when_settled if settle else self.profiles.save_profile
        for channel in self.channels:
            save(channel.profile_key, channel.profile())
    def _build_phone_controls(self, add):
        """Create the iPhone's own controls, shown only while an iPhone is displayed.

        Exposure, gain (the phone's ISO) and frame rate use the Camera
        section's sliders; these are the settings only the phone has. Every
        one is sent to the ESPI Camera app, so nothing needs changing on the
        phone itself.

        Args: add is _build_controls' add(widget, label, basic) helper.
        Returns: None.
        """
        self.phone_capture = WheelIgnoringComboBox()
        self.phone_capture.addItem("Video (the phone's processed luma)", False)
        self.phone_capture.addItem("Bayer RAW (sensor data)", True)
        set_tooltip(self.phone_capture,
                    "Video: the phone's own video pipeline, with its noise reduction "
                    "and tone mapping, at the frame rate below. Bayer RAW: one colour "
                    "plane of the sensor's 10-bit data, with that pipeline bypassed: "
                    "noisier pixels, but the noise is the sensor's own and the speckle "
                    "is not smoothed. It is taken as still photos, so expect about "
                    "5-10 frames per second, at half the sensor's width and height; "
                    "frame rate, size and crop below then do not apply.")
        self.phone_capture.activated.connect(
            lambda index: self._phone_command(raw=self.phone_capture.itemData(index)))
        add(self.phone_capture, "Capture", basic=True)
        self.phone_raw_channel = WheelIgnoringComboBox()
        for channel, text in (("red", "Red (633 nm laser)"),
                              ("green", "Green (532 nm laser)"), ("blue", "Blue")):
            self.phone_raw_channel.addItem(text, channel)
        set_tooltip(self.phone_raw_channel,
                    "Which of the sensor's colour sites RAW mode keeps. Match the "
                    "laser: a red laser lights mainly the red sites (a quarter of the "
                    "pixels) and a green one the green sites (both are averaged). The "
                    "other sites carry little signal.")
        self.phone_raw_channel.activated.connect(
            lambda index: self._phone_command(
                raw_channel=self.phone_raw_channel.itemData(index)))
        add(self.phone_raw_channel, "RAW colour", basic=True)
        self.phone_fps = WheelIgnoringComboBox()
        set_tooltip(self.phone_fps,
                    "Frame rate the iPhone captures at. A USB 2 iPhone carries about "
                    "30 MB/s: 1920 × 1080 at about 15 fps, or a smaller crop at 30 fps.")
        self.phone_fps.activated.connect(self._phone_fps_chosen)
        add(self.phone_fps, "Frame rate", basic=True)
        self.phone_size = WheelIgnoringComboBox()
        set_tooltip(self.phone_size,
                    "Frame size the iPhone camera captures. Larger frames show finer "
                    "speckle but need more USB bandwidth.")
        self.phone_size.activated.connect(self._phone_size_chosen)
        add(self.phone_size, "Frame size", basic=True)
        self.phone_crop = WheelIgnoringComboBox()
        for size in iphone.CROP_SIZES:
            self.phone_crop.addItem("Full frame" if size == 0 else f"{size} × {size} centre",
                                    size)
        set_tooltip(self.phone_crop,
                    "Send only the centre of each frame, at full resolution: less "
                    "data, so a higher frame rate over USB.")
        self.phone_crop.activated.connect(
            lambda index: self._phone_command(crop_width=self.phone_crop.itemData(index),
                                              crop_height=self.phone_crop.itemData(index)))
        add(self.phone_crop, "Crop", basic=True)
        self.phone_send_cap = WheelIgnoringComboBox()
        for cap in iphone.SEND_CAPS:
            self.phone_send_cap.addItem("Every frame" if cap == 0 else f"{cap:g} fps", cap)
        set_tooltip(self.phone_send_cap,
                    "Send at most this many frames per second, even when the camera "
                    "captures more. The phone always skips a frame rather than "
                    "compress or queue it.")
        self.phone_send_cap.activated.connect(
            lambda index: self._phone_command(
                max_send_fps=self.phone_send_cap.itemData(index)))
        add(self.phone_send_cap, "Send at most", basic=True)
        self.phone_focus = SliderSpin(0, 1, 0.005, 3, 0.5)
        self.phone_focus.set_description(
            "Lens position, locked: 0 is the nearest focus, 1 the farthest. "
            "Autofocus sets it once and locks it again.")
        self.phone_focus.valueChanged.connect(
            lambda v: self._phone_command(lens_position=v))
        add(self.phone_focus, "Focus", basic=True)
        self.phone_autofocus = QPushButton("Autofocus once, then lock")
        set_tooltip(self.phone_autofocus,
                    "Focus on the centre of the image, then lock the lens there so "
                    "focus cannot drift during a measurement.")
        self.phone_autofocus.clicked.connect(lambda: self._phone_command(refocus=True))
        add(self.phone_autofocus, basic=True)
        for widget in self.phone_section:
            widget.setVisible(False)

    def _phone_command(self, **fields):
        """Send phone-only settings to the displayed iPhone.

        Args: fields are COMMAND keys and values (see iphone.IPhoneSession).
        Returns: None.
        """
        if "raw" in fields:  # a new capture mode starts the measurement over
            self.engine.request("reset_average")
            self.engine.request("reset_previous")
        self.engine.request("camera_command", **fields)

    def _phone_fps_chosen(self, index):
        """Use a frame rate picked in the iPhone section.

        Args: index is the drop-down position.
        Returns: None; the Camera section's frame-rate slider follows.
        """
        fps = self.phone_fps.itemData(index)
        self.fps.set_value(fps)
        self.engine.update_settings(fps=fps)

    def _phone_size_chosen(self, index):
        """Use a frame size picked in the iPhone section.

        Args: index is the drop-down position.
        Returns: None.
        """
        width, height = (int(part) for part in self.phone_size.itemData(index).split("x"))
        self._phone_command(width=width, height=height)

    def _sync_phone_controls(self, channel):
        """Show the iPhone section for an iPhone and fill it from what it reported.

        Args: channel is the displayed Channel.
        Returns: None.
        """
        phone = not channel.demo and channel.info.kind == "iphone"
        toggle, body = self.phone_section
        toggle.setVisible(phone)
        body.setVisible(phone and toggle.isChecked())  # keep a collapsed section collapsed
        if not phone:
            return
        identity = channel.identity
        formats = getattr(identity, "formats", ()) or ()
        settings = channel.camera_settings or {}
        self.phone_size.clear()
        for width, height, max_fps in formats:
            # "WxH" text, not a tuple: findData cannot match Python tuples.
            self.phone_size.addItem(f"{width} × {height} (up to {max_fps:g} fps)",
                                    f"{width}x{height}")
        top = max([entry[2] for entry in formats] + [30.0])
        self.phone_fps.clear()
        for fps in PHONE_FRAME_RATES:
            if fps <= top:
                self.phone_fps.addItem(f"{fps:g} fps", fps)
        self._show_phone_settings({"fps": self.engine.settings().fps, **settings})

    def _show_phone_settings(self, settings):
        """Set the iPhone controls to the phone's values, without sending them back.

        Args: settings maps take_settings_update() keys to values; missing
            keys leave their control unchanged.
        Returns: None.
        """
        def select(combo, data):
            """Pick the item whose data matches, if there is one.

            Args: combo is a QComboBox; data is the item data to find.
            Returns: None.
            """
            index = combo.findData(data)
            if index >= 0:
                combo.setCurrentIndex(index)

        if "raw" in settings:
            select(self.phone_capture, bool(settings["raw"]))
        if "raw_channel" in settings:
            select(self.phone_raw_channel, settings["raw_channel"])
        if "width" in settings and "height" in settings:
            select(self.phone_size, f"{settings['width']}x{settings['height']}")
        if "crop_width" in settings:
            select(self.phone_crop, settings["crop_width"])
        if "max_send_fps" in settings:
            select(self.phone_send_cap, float(settings["max_send_fps"]))
        if "fps" in settings:
            fps = float(settings["fps"])
            if self.phone_fps.findData(fps) < 0:
                self.phone_fps.addItem(f"{fps:g} fps", fps)
            select(self.phone_fps, fps)
        if "lens_position" in settings:
            self.phone_focus.set_value(settings["lens_position"])

    def _phone_settings_changed(self, channel, changed):
        """Follow settings the phone reports (changed on the phone, or clamped).

        Args: channel is the iPhone's Channel; changed is the dict from a
            "camera_settings" event.
        Returns: None; the camera sliders and the engine settings follow
            exposure, gain and frame rate, so the two screens agree.
        """
        channel.camera_settings = {**(channel.camera_settings or {}), **changed}
        engine_fields = {field: changed[key] for key, field in (
            ("exposure_us", "exposure_us"), ("gain_db", "gain_db"), ("fps", "fps"))
            if key in changed}
        if engine_fields and channel.engine is not None:
            channel.engine.update_settings(**engine_fields)
        if channel is not self.current:
            return
        for key, widget in (("exposure_us", self.exposure), ("gain_db", self.gain),
                            ("fps", self.fps)):
            if key in changed:
                widget.set_value(changed[key])
        self._show_phone_settings(changed)

    def _add_preset_buttons(self, add):
        """Add one button per processing preset, and the LabVIEW flow toggle, to the controls grid.

        Args: add places a widget on the next controls-grid row.
        Returns: None.
        """
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        for name, tip in PRESET_TIPS.items():
            button = QPushButton(name)
            set_tooltip(button, tip)
            button.clicked.connect(
                lambda _checked=False, name=name: self.apply_preset(name))
            row.addWidget(button)
        add(row_widget)
        # Its own row: four buttons side by side made the panel (and so the
        # window) too wide for a small screen.
        self.labview_button = QPushButton("LabVIEW flow")
        self.labview_button.setCheckable(True)
        set_tooltip(self.labview_button, LABVIEW_FLOW_TIP)
        self.labview_button.setChecked(self.current.labview_saved is not None)
        self.labview_button.toggled.connect(self.set_labview_flow)
        add(self.labview_button)

    def set_raw_difference(self, on):
        """Turn the displayed camera's Raw difference view on or off.

        Args: on is True to show |live - previous| of raw frames only.
        Returns: None.
        """
        self.engine.update_settings(raw_difference=on)
        self._sync_raw_difference(on)

    def _sync_raw_difference(self, on):
        """Grey out the view buttons while Raw difference overrides them.

        Args: on is the Raw difference state.
        Returns: None.
        """
        for button in self.mode_group.buttons():
            button.setEnabled(not on)

    def set_labview_flow(self, on):
        """Swap the displayed camera between the user's and the LabVIEW settings.

        Turning it on keeps the user's values of every LABVIEW_FLOW setting on
        the channel, applies the LabVIEW flow and captures a reference, as the
        VI did when it started. Turning it off restores the kept values.

        Args: on is True to use the LabVIEW flow.
        Returns: None.
        """
        channel = self.current
        if on and channel.labview_saved is None:
            settings = asdict(self.engine.settings())
            channel.labview_saved = {key: settings[key] for key in LABVIEW_FLOW}
            self.engine.update_settings(**LABVIEW_FLOW)
            self.engine.request("reference")
            self.log.write("LabVIEW flow on: your settings are kept and come back "
                           "when you turn it off.")
        elif not on and channel.labview_saved is not None:
            self.engine.update_settings(**channel.labview_saved)
            channel.labview_saved = None
            self.log.write("LabVIEW flow off: your settings are back.")
        self._load_settings(self.engine.settings(), channel.limits)

    def apply_preset(self, name):
        """Apply a processing preset to the displayed camera.

        Exposure, gain, frame rate, recording and the single-field option are
        left as they are. The controls update to match.

        Args: name is a key of PROCESSING_PRESETS.
        Returns: None.
        """
        self.engine.update_settings(**PROCESSING_PRESETS[name])
        self._load_settings(self.engine.settings(), self.current.limits)
        self.log.write(f"Preset applied: {name}")

    def _show_settings_guide(self):
        """Open the guide to the processing controls and the presets.

        Args: None.
        Returns: None.
        """
        SettingsGuideDialog(self).exec()

    def _build_log(self):
        """Create the log dock.

        Args: None.
        Returns: None.
        """
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        set_tooltip(self.log_view, "Messages from this run, including the camera, "
                    "recording and updates. The same text is written to the log file.")
        self.log_view.setMaximumBlockCount(2000)
        font = self.log_view.font()
        font.setStyleHint(font.StyleHint.Monospace)
        font.setFamily("Menlo" if sys.platform == "darwin" else "Consolas")
        self.log_view.setFont(font)
        self.log_dock = QDockWidget("Log", self)
        self.log_dock.setObjectName("log")
        self.log_dock.setWidget(self.log_view)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self._tip_dock_buttons(self.log_dock)
        self.resizeDocks([self.log_dock], [ui_scale.px(120)], Qt.Vertical)

    def _tip_dock_buttons(self, dock):
        """Explain the float and close buttons Qt puts on a dock title.

        Args: dock is a QDockWidget.
        Returns: None.
        """
        tips = {
            "qt_dockwidget_floatbutton":
                "Float this panel in its own window, or dock it again.",
            "qt_dockwidget_closebutton":
                "Hide this panel. The View menu shows it again.",
        }
        for button in dock.findChildren(QAbstractButton):
            tip = tips.get(button.objectName())
            if tip:
                set_tooltip(button, tip)

    def _build_status_bar(self):
        """Create the status-bar readouts and action buttons.

        Args: None.
        Returns: None.
        """
        bar = self.statusBar()
        bar.setAttribute(Qt.WA_AlwaysShowToolTips, True)
        # One strip holds every item: it keeps them on one row while they fit
        # and wraps them onto more rows when the window is narrower, so nothing
        # is cut short. The strip is the bar's only widget, and it also shows
        # the bar's temporary messages.
        self.status_strip = StatusStrip(message_min=0)
        bar.addPermanentWidget(self.status_strip, 1)
        bar.messageChanged.connect(self.status_strip.show_message)
        self.error_report_label = QLabel()  # error-report results, with the issue link
        self.error_report_label.setTextFormat(Qt.RichText)
        self.error_report_label.setOpenExternalLinks(True)
        self.error_report_label.setVisible(False)
        self.error_report_label.setStyleSheet("color:#b00020;")
        self.status_strip.add_notice(self.error_report_label)
        self.error_report_timer = QTimer(self)
        self.error_report_timer.setSingleShot(True)
        self.error_report_timer.timeout.connect(self.error_report_label.hide)
        settings = QSettings("ESPI", "espi_live")
        self.tips = tips.TipRotator(
            hidden=bool(settings.value("tips/hidden", False, bool)))
        # The hint keeps its whole sentence; the tip is Ignored, with no
        # minimum, and fills what the hint leaves (capped at the longest tip),
        # shortened with an ellipsis when that is little. Right-click and
        # View > Help tips dismiss it.
        self.tip_label = TipLabel(*tips.TIPS)
        self.tip_label.clicked.connect(self._rotate_tip)
        self.tip_label.dismissed.connect(lambda: self._set_tips_visible(False))
        self.tip_label.setMinimumWidth(0)
        self.tip_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.hint_label = QLabel()  # the current view's one-line explanation
        self.hint_label.setMinimumWidth(0)
        self.hint_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        set_tooltip(self.hint_label, "What the current view is showing.")
        self.status_strip.add_notice(self.tip_label, 1)  # left of the hint, as before
        self.status_strip.add_notice(self.hint_label)
        self._show_tip()
        # First item after the message area: showing it only narrows the hint
        # on a shared row, and takes its place in the flow on a wrapped one.
        self.frozen_badge = QLabel("  FROZEN  ")
        self.frozen_badge.setStyleSheet(FROZEN_STYLE)
        set_tooltip(self.frozen_badge,
                    "The image on screen is held; the camera keeps running. "
                    "Shortcut: Space returns to live.")
        self.frozen_badge.setVisible(False)
        self.status_strip.add_item(self.frozen_badge)
        # Every readout has a fixed width, so no value can move its neighbours.
        self.source_badge = FixedLabel("  DUMMY MODE  ", "  BFS-U3-200S6M-C  ")
        self.source_badge.hold_full_text()  # never "DUMM…"; the readouts ellipsize first
        self.source_badge.grow_to_text()  # a long camera name widens it
        self.record_label = FixedLabel("Recording: finishing file", "Recording: 999.9 fps")
        set_tooltip(self.record_label,
                    "Whether this camera is recording, and the rate the file is "
                    "being written.")
        self.range_status = FixedLabel("Input: 255–255    Scaled: 255–255",
                                       "Raw Mono8: no brightness scaling")
        set_tooltip(self.range_status,
                    "Brightness before and after Scaling (Fixed gain or Auto "
                    "stretch), measured before gamma. Raw view is "
                    "not scaled.")
        self.camera_fps_status = FixedLabel(  # sensor rate, then what the app keeps up with
            "Camera: 999.9 fps (simulated)")
        set_tooltip(self.camera_fps_status,
                    "Camera: the camera's resulting acquisition rate "
                    "(AcquisitionResultingFrameRate).",
                    "Processed: frames the app received and processed.",
                    "Displayed: frames drawn on screen.")
        self.speed_status = FixedLabel(
            "Processed: 999.9 fps    Work: 99.9 ms/frame    Displayed: 999.9 fps")
        set_tooltip(self.speed_status,
                    "Processed: frames the app kept up with. Work: time spent on "
                    "each frame. Displayed: frames drawn, or frozen while the "
                    "display is held.")
        self.pixel_status = FixedLabel("x 3088, y 2064: 255")
        set_tooltip(self.pixel_status,
                    "Brightness from 0 to 255 of the image pixel under the pointer.")
        # Check for updates, Request feature and Open in Codex are icon-only,
        # so the readouts keep their room on a laptop screen; each is named in
        # its tooltip and accessible name. They are real buttons (border, icon,
        # pointer), split from the readouts by a rule. The blue update button
        # keeps its words: it is the one action to act on, and it replaces
        # the check button in the same slot once a newer build is found.
        self.update_button = QPushButton()  # shown when a newer build exists
        self.update_button.setVisible(False)
        set_tooltip(self.update_button, "Download the newer build of ESPI live.")
        self.update_button.clicked.connect(self._update_clicked)
        self.check_button = QPushButton()  # until a newer build is found
        _set_icon_action(self.check_button, CHECK_LABEL, CHECK_TIP)
        self.check_button.clicked.connect(self._check_updates_now)
        ink = self.check_button.palette().color(QPalette.ColorRole.ButtonText)
        _arm_status_button(
            self.update_button, UPDATE_BUTTON_STYLE,
            ("Install update", "Downloading update… 100%",
             "Restart to update"),
            ("download", QColor(Qt.white)))
        _arm_status_button(self.check_button, STATUS_BUTTON_STYLE, ("",),
                           ("refresh", ink))
        self.update_slot = QWidget()
        slot = QGridLayout(self.update_slot)
        slot.setContentsMargins(0, 0, 0, 0)
        for button in (self.update_button, self.check_button):
            slot.addWidget(button, 0, 0)  # same cell; the slot fits the visible one
        self.status_strip.add_item(self.update_slot)
        self.feature_button = QPushButton()
        _set_icon_action(self.feature_button, FEATURE_LABEL, FEATURE_TIP)
        self.feature_button.clicked.connect(lambda: self._request_feature())
        _arm_status_button(self.feature_button, STATUS_BUTTON_STYLE, ("",),
                           ("idea", ink))
        self.status_strip.add_item(self.feature_button)
        self.codex_button = QPushButton()
        _set_icon_action(self.codex_button, CODEX_LABEL, CODEX_TIP)
        self.codex_button.clicked.connect(self._open_codex)
        _arm_status_button(self.codex_button, STATUS_BUTTON_STYLE, ("",),
                           ("code", ink))
        self.status_strip.add_item(self.codex_button)
        self.status_actions_rule = QFrame()
        self.status_actions_rule.setFrameShape(QFrame.Shape.VLine)
        self.status_actions_rule.setFrameShadow(QFrame.Shadow.Sunken)
        self._sized(self.status_actions_rule.setFixedWidth, 10)
        self.status_strip.add_item(self.status_actions_rule)
        self.status_strip.flow.separator = self.status_actions_rule
        for widget in (self.source_badge, self.record_label, self.range_status,
                       self.camera_fps_status, self.speed_status, self.pixel_status):
            self.status_strip.add_item(widget)
        self._fit_status_bar()

    def _status_buttons(self):
        """List the status bar's four action buttons.

        Args: None.
        Returns: A tuple of QPushButton.
        """
        return (self.update_button, self.check_button, self.feature_button,
                self.codex_button)

    def _fit_status_bar(self):
        """Measure the status bar's items in the current font.

        The message area keeps the longest hint and a little of the tip beside
        the readouts (with less room the readouts move to their own rows), the
        tip is capped at its longest sentence, and the four buttons share the
        tallest one's height. Runs at build time and whenever the interface
        size changes.

        Args: None.
        Returns: None.
        """
        self.tip_label.setMaximumWidth(self.tip_label.sizeHint().width())
        self.status_strip.flow.message_min = (
            self.hint_label.fontMetrics().horizontalAdvance(LONGEST_HINT)
            + ui_scale.px(MESSAGE_TIP_WIDTH))
        buttons = self._status_buttons()
        for button in buttons:
            button.setMinimumHeight(0)
            button.setMaximumHeight(QWIDGETSIZE_MAX)
        height = max(button.sizeHint().height() for button in buttons)
        for button in buttons:
            button.setFixedHeight(height)
        self.status_strip.flow.invalidate()

    def _build_menus(self):
        """Create the View menu (docks, zoom and freeze) and the Help menu.

        Args: None.
        Returns: None.
        """
        self.menuBar().setAttribute(Qt.WA_AlwaysShowToolTips, True)
        menu = self.menuBar().addMenu("&View")
        set_tooltip(self.controls_dock.toggleViewAction(), "Show or hide the Controls panel.")
        set_tooltip(self.log_dock.toggleViewAction(), "Show or hide the Log panel.")
        set_tooltip(self.actions_bar.toggleViewAction(), "Show or hide the toolbar.")
        menu.addAction(self.controls_dock.toggleViewAction())
        menu.addAction(self.log_dock.toggleViewAction())
        menu.addAction(self.actions_bar.toggleViewAction())
        self.tips_action = QAction("Help tips", self)
        self.tips_action.setCheckable(True)
        self.tips_action.setChecked(not self.tips.hidden)
        set_tooltip(self.tips_action, "Show rotating help tips in the status bar.")
        self.tips_action.triggered.connect(self._set_tips_visible)
        self.tips_action.setVisible(not self.simple_ui)
        menu.addAction(self.tips_action)
        scale_menu = menu.addMenu("Interface size")
        self.scale_larger_action = QAction("Larger", self)
        self.scale_larger_action.setShortcuts(
            [QKeySequence(QKeySequence.ZoomIn), QKeySequence("Ctrl+=")])
        set_tooltip(self.scale_larger_action,
                    "Make the text and controls larger. Shortcut: Ctrl++.")
        self.scale_larger_action.triggered.connect(
            lambda: self.set_ui_scale(ui_scale.larger(ui_scale.percent())))
        self.scale_smaller_action = QAction("Smaller", self)
        self.scale_smaller_action.setShortcut(QKeySequence(QKeySequence.ZoomOut))
        set_tooltip(self.scale_smaller_action,
                    "Make the text and controls smaller. Shortcut: Ctrl+-.")
        self.scale_smaller_action.triggered.connect(
            lambda: self.set_ui_scale(ui_scale.smaller(ui_scale.percent())))
        self.scale_reset_action = QAction(f"Reset to {ui_scale.DEFAULT_PERCENT}%", self)
        self.scale_reset_action.setShortcut(QKeySequence("Ctrl+0"))
        set_tooltip(self.scale_reset_action,
                    "Return the window to its standard size. Shortcut: Ctrl+0.")
        self.scale_reset_action.triggered.connect(
            lambda: self.set_ui_scale(ui_scale.DEFAULT_PERCENT))
        for action in (self.scale_larger_action, self.scale_smaller_action,
                       self.scale_reset_action):
            scale_menu.addAction(action)
        menu.addSeparator()
        for text, key, tip, handler in (
                ("Fit to window", "0", "Show the whole image. Shortcut: 0.", self.view.fit),
                ("Actual size", "1",
                 "One camera pixel per screen pixel. Shortcut: 1.", self.view.actual_size)):
            action = QAction(text, self)
            action.setShortcut(QKeySequence(key))
            action.setShortcutContext(Qt.WidgetShortcut)  # keys go via handle_key
            set_tooltip(action, tip)
            action.triggered.connect(handler)
            menu.addAction(action)
        menu.addSeparator()
        self.freeze_action = QAction("Freeze display", self)
        self.freeze_action.setCheckable(True)
        self.freeze_action.setShortcut(QKeySequence("Space"))
        self.freeze_action.setShortcutContext(Qt.WidgetShortcut)  # via handle_key
        set_tooltip(self.freeze_action,
                    "Hold the displayed image without saving it. The camera keeps "
                    "running. Shortcut: Space.")
        self.freeze_action.triggered.connect(self.set_frozen)
        menu.addAction(self.freeze_action)
        self.piezo_debug_action = QAction("Piezo debug overlay", self)
        self.piezo_debug_action.setCheckable(True)
        set_tooltip(self.piezo_debug_action,
                    "Show the piezo's numbers in a strip above the image: the frame rate "
                    "received, the triangle applied and computed, the phase step per "
                    "frame measured from the frames, flicker and auto-tune. On screen "
                    "only: saved frames and recordings never show it.")
        self.piezo_debug_action.triggered.connect(self._show_piezo_debug)
        menu.addAction(self.piezo_debug_action)
        menu.addSeparator()
        self.split_action = QAction("Split two streams", self)
        self.split_action.setCheckable(True)
        set_tooltip(self.split_action,
                    "Show two cameras side by side. Click a pane to make that camera "
                    "the one the controls edit. Needs two processing cameras.")
        self.split_action.triggered.connect(
            lambda on: self.set_display_mode("split" if on else "single"))
        menu.addAction(self.split_action)
        self.overlay_action = QAction("Overlay two streams", self)
        self.overlay_action.setCheckable(True)
        set_tooltip(self.overlay_action,
                    "Blend two cameras into one image (their average grey level), for "
                    "example the front and back of a sample. Needs two processing "
                    "cameras.")
        self.overlay_action.triggered.connect(
            lambda on: self.set_display_mode("overlay" if on else "single"))
        menu.addAction(self.overlay_action)
        self.stream_window_action = QAction("Second stream window…", self)
        set_tooltip(self.stream_window_action,
                    "Open another window showing a second camera's live stream, while "
                    "this window keeps its own. Needs two processing cameras.")
        self.stream_window_action.triggered.connect(self.open_stream_window)
        menu.addAction(self.stream_window_action)
        self._sync_stream_actions()
        menu.addSeparator()
        # The live image has no tooltip: a popup there would hide the measurement.
        # The shortcut list lives here instead.
        hint = menu.addAction("Wheel to zoom, drag to pan, Q or Esc to quit")
        hint.setEnabled(False)  # a reminder, not a command
        set_tooltip(hint,
                    "Wheel: zoom. Drag: pan. 0: fit. 1: actual size. Space: freeze. "
                    "S: save. F: set reference. C: compare with the reference. "
                    "P: pixel values (click the image to move the square). "
                    "R: reset last. A: reset average. "
                    "Q or Esc: quit.")

        devices_menu = self.menuBar().addMenu("&Devices")
        devices_menu.setToolTipsVisible(True)
        self.daq_action = devices_menu.addAction(f"{DAQ_TITLE}…")
        set_tooltip(self.daq_action,
                    "Assign an NI DAQ's outputs to the frequency generator (specimen "
                    "drive) and the piezo driver, and run them.")
        self.daq_action.triggered.connect(self._show_daq_dialog)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.setToolTipsVisible(True)
        guide = help_menu.addAction("Which settings should I use?")
        set_tooltip(guide, "Explain the processing controls and apply a starting preset.")
        guide.triggered.connect(self._show_settings_guide)
        check = help_menu.addAction("Check for updates…")
        set_tooltip(check, "Check GitHub for a newer build of ESPI live.")
        check.triggered.connect(self._check_updates_now)
        request = help_menu.addAction("Request a feature…")
        set_tooltip(request, "Send a feature request to the developers as a GitHub issue.")
        request.triggered.connect(lambda: self._request_feature())
        ask = help_menu.addAction("Ask Codex about the source code…")
        set_tooltip(ask, "Ask OpenAI's Codex how ESPI live works, using this build's "
                    "real source code.")
        ask.triggered.connect(self._open_codex)
        help_menu.addSeparator()
        self.reports_action = help_menu.addAction("Send error reports automatically")
        self.reports_action.setCheckable(True)
        self._sync_reports_action()
        self.reports_action.toggled.connect(self._choose_error_reports)
        logs = help_menu.addAction("Open log folder")
        logs.triggered.connect(self._open_log_folder)
        help_menu.addSeparator()
        guide = help_menu.addAction("User Guide")
        guide.setShortcut(QKeySequence.HelpContents)
        set_tooltip(guide, "Open the illustrated user guide in your browser. "
                    "Shortcut: F1.")
        guide.triggered.connect(self.show_guide)
        help_menu.addSeparator()
        about = help_menu.addAction("About ESPI live")
        set_tooltip(about, "Show the version and build of this copy.")
        about.triggered.connect(lambda: QMessageBox.about(
            self, "About ESPI live", updater.describe_build()))

    def show_guide(self):
        """Open the bundled HTML user guide in the default browser.

        Args: None.
        Returns: None; reports the path when the browser cannot be started.
        """
        path = guide_path()
        if not os.path.isfile(path):
            QMessageBox.warning(
                self, "User guide missing",
                f"The user guide could not be found at:\n{path}")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(path)):
            QMessageBox.information(
                self, "User guide",
                "The default browser could not be opened. Open this file "
                f"manually:\n{path}")

    # ----- cameras -------------------------------------------------------------

    def _rebuild_cameras(self):
        """Fill the camera drop-down and process checkboxes from self.channels.

        The drop-down is shown even for a single camera; the process checkboxes
        and their note only matter when there is more than one.

        Args: None.
        Returns: None.
        """
        several = len(self.channels) > 1
        for channel in self.channels:
            channel.log.prefix = f"[{channel.info.serial}] " if several else ""
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        self.camera_combo.addItems([
            channel.label if channel.supported else f"{channel.label} · not yet supported"
            for channel in self.channels])
        self.camera_combo.blockSignals(False)
        for widget in self.camera_checks + self.camera_states:
            self.camera_rows.removeWidget(widget)
            widget.deleteLater()
        self.camera_checks, self.camera_states = [], []
        for row, channel in enumerate(self.channels):
            check = QCheckBox(channel.info.serial)
            set_tooltip(check,
                        f"{channel.label}: acquire, process and (when Record is on) "
                        "record this camera even while another one is displayed. The "
                        "displayed camera always processes.")
            check.toggled.connect(
                lambda on, channel=channel: self._set_processing(channel, on))
            state = QLabel()
            state.setStyleSheet("color: palette(placeholder-text);")
            set_tooltip(state, "This camera's processed frame rate, REC while it "
                        "records, and frozen while its display is held.")
            check.setVisible(several)
            state.setVisible(several)
            self.camera_rows.addWidget(check, row, 0)
            self.camera_rows.addWidget(state, row, 1)
            self.camera_checks.append(check)
            self.camera_states.append(state)
        self.process_note.setVisible(several)
        self.profile_label.setVisible(several)
        self._sync_cameras()
        self._stream_follow()

    def _sync_cameras(self):
        """Show which camera is displayed and which ones are processing.

        The displayed camera always processes, so its checkbox is locked on.

        Args: None.
        Returns: None.
        """
        self.camera_combo.blockSignals(True)
        self.camera_combo.setCurrentIndex(self.channels.index(self.current))
        self.camera_combo.blockSignals(False)
        self.profile_label.setText(f"Settings of {self.current.label}")
        for channel, check in zip(self.channels, self.camera_checks):
            check.blockSignals(True)
            check.setChecked(channel.active)
            check.blockSignals(False)
            check.setEnabled(channel is not self.current
                             and not channel.shares_device(self.current))

    def _camera_chosen(self, index):
        """Display the camera picked in the drop-down.

        Args: index is the drop-down position (one per channel).
        Returns: None.
        """
        if 0 <= index < len(self.channels):
            self.show_channel(self.channels[index])

    def show_channel(self, channel):
        """Display a camera, starting its processing if it was off.

        Args: channel is one of self.channels.
        Returns: None.
        """
        if channel is self.current:
            return
        # The camera left behind keeps what was set on it, saved now.
        self.profiles.save_profile(self.current.profile_key, self.current.profile())
        if not channel.active:
            self.log.write(f"Processing on: {channel.label}")
            channel.start(self._release_shared_device(channel))
        self.current = channel
        self._attach_steppers()  # each camera keeps its own driver's stepper
        self.drawn_sequence = 0
        self.shown_packet = None
        self.view.clear()
        if channel.frozen is not None:  # its held frame comes back at once
            self._draw(channel.frozen)
        self._show_frozen()
        self._load_settings(self.engine.settings(), channel.limits)
        self._set_demo(channel.demo)
        self._sync_cameras()
        self._stream_follow()
        self.log.write(f"Displaying {channel.label}")

    def _release_shared_device(self, channel):
        """Stop the other cameras of the phone a channel is about to open.

        An iPhone streams one back camera at a time, so choosing another
        lens of the same phone turns the one it was showing off (its
        settings are kept).

        Args: channel is the channel about to start.
        Returns: The Engine the new one should wait for, or None when no
            other camera shares its device.
        """
        after = None
        for other in self.channels:
            if other.active and other.shares_device(channel):
                self.log.write(f"Processing off: {other.label} "
                               f"(the phone streams one camera at a time)")
                other.stop()
                after = other.retired[-1]
        return after

    def _set_processing(self, channel, on):
        """Start or stop a camera that is not displayed.

        Args: channel is one of self.channels; on is the checkbox state.
        Returns: None.
        """
        if on and not channel.active:
            if channel.shares_device(self.current):
                self._sync_cameras()  # the displayed lens of this phone keeps it
                return
            self.log.write(f"Processing on: {channel.label}")
            channel.start(self._release_shared_device(channel))
        elif not on and channel.active and channel is not self.current:
            self.log.write(f"Processing off: {channel.label} (settings kept)")
            channel.stop()
        self._sync_cameras()

    def _load_settings(self, s, limits):
        """Show a camera's settings and limits without sending them back.

        Args: s is that camera's Settings; limits is its {node name: (low,
            high)}, missing names using DEFAULT_LIMITS.
        Returns: None.
        """
        # Quietly: a slider that clamps the value still on screen (another
        # camera's) would otherwise send it to this camera.
        self._apply_limits({**DEFAULT_LIMITS, **limits}, quiet=True)
        for widget, value in ((self.exposure, s.exposure_us), (self.gain, s.gain_db),
                              (self.fps, s.fps), (self.noise_reduction, s.noise_reduction),
                              (self.average, s.average),
                              (self.ref_frames, s.ref_frames),
                              (self.ref_refresh, s.ref_refresh_s),
                              (self.smoothing, s.smoothing), (self.clip, s.clip_percent),
                              (self.gamma, s.gamma), (self.frames_back, s.frames_back),
                              (self.outlier_filter, s.outlier_filter),
                              (self.difference_gain, s.difference_gain),
                              (self.average_multiplier, s.average_multiplier),
                              (self.min_range, s.min_range),
                              (self.threshold, s.threshold),
                              (self.steady_range, s.steady_range_s),
                              (self.peak_hold, s.peak_hold_s),
                              (self.normalize_window, s.normalize_window),
                              (self.brightness, s.brightness),
                              (self.display_gain, s.display_gain),
                              (self.contrast, s.contrast),
                              (self.phase_filter, s.phase_filter),
                              (self.phase_settle, s.phase_settle_frames)):
            widget.set_value(value)
        self.phase_steps.blockSignals(True)
        self.phase_steps.setCurrentIndex(self.phase_steps.findData(s.phase_steps))
        self.phase_steps.blockSignals(False)
        toggles = ((self.rolling, s.rolling),
                   (self.gain_before_average, s.gain_before_average),
                   (self.invert, s.invert), (self.flip_button, s.flip_horizontal),
                   (self.even_brightness, s.even_brightness),
                   (self.frames_first, s.average_frames_first),
                   (self.floor_subtract, s.floor_subtract),
                   (self.record_button, s.record), (self.single_field, s.single_field),
                   (self.raw_difference_button, s.raw_difference),
                   (self.simulated_piezo, s.demo_piezo))
        combos = ((self.smoothing_combo, s.smoothing_method),
                  (self.simulated_object, s.demo_object),
                  (self.scaling, SCALING_MODES[1] if s.scale_range else SCALING_MODES[0]))
        for widget, _ in toggles + combos:
            widget.blockSignals(True)
        for widget, on in toggles:
            widget.setChecked(on)
        for widget, text in combos:
            widget.setCurrentText(text)
        for widget, _ in toggles + combos:
            widget.blockSignals(False)
        self._show_simulated_note()
        for button in self.mode_group.buttons():
            button.setChecked(button.text() == s.mode)
        self._sync_raw_difference(s.raw_difference)
        self._sync_scaling_controls(s.scale_range, s.gain_before_average)
        self.rolling.setEnabled(not s.average_frames_first)
        self.labview_button.blockSignals(True)
        self.labview_button.setChecked(self.current is not None and
                                       self.current.labview_saved is not None)
        self.labview_button.blockSignals(False)
        self._show_phase_state()

    def _scaling_changed(self, name):
        """Send a Scaling choice to the engine and match the related controls.

        Args: name is a SCALING_MODES entry.
        Returns: None.
        """
        stretch = name == SCALING_MODES[1]
        self.engine.update_settings(scale_range=stretch)
        self._sync_scaling_controls(stretch, self.gain_before_average.isChecked())

    def _frames_first_changed(self, on):
        """Send the Average camera frames choice and match Rolling average.

        Args: on is the checkbox state.
        Returns: None.
        """
        self.engine.update_settings(average_frames_first=on)
        self.rolling.setEnabled(not on)

    def _gain_order_changed(self, on):
        """Send the Gain before averaging choice and match the multiplier.

        Args: on is the checkbox state.
        Returns: None.
        """
        self.engine.update_settings(gain_before_average=on)
        self._sync_scaling_controls(
            self.scaling.currentText() == SCALING_MODES[1], on)

    def _sync_scaling_controls(self, stretch, before_average):
        """Enable only the controls that the chosen scaling uses.

        Args: stretch is True for Auto stretch; before_average is the Gain
            before averaging choice.
        Returns: None.
        """
        self.clip.setEnabled(stretch)
        self.steady_range.setEnabled(stretch)
        self.difference_gain.setEnabled(not stretch)
        self.gain_before_average.setEnabled(not stretch)
        self.average_multiplier.setEnabled(not stretch and before_average)

    def _apply_limits(self, limits, quiet=False):
        """Set the camera sliders to device limits.

        Args: limits is {node name: (low, high)} for the names to change;
            quiet keeps a value the new range clamps from being sent to the
            engine (used while another camera's values are still shown).
        Returns: None.
        """
        for name, widget in (("ExposureTime", self.exposure), ("Gain", self.gain),
                             ("AcquisitionFrameRate", self.fps)):
            if name in limits:
                widget.blockSignals(quiet)
                widget.set_range(*limits[name])
                widget.blockSignals(False)

    def _control_sync(self, channel):
        """Refresh the controls after the control server changed a camera.

        Runs on the GUI thread (scheduled by the control backend).

        Args: channel is the Channel whose settings or state changed.
        Returns: None.
        """
        if channel not in self.channels:
            return
        if channel is self.current and channel.engine is not None:
            self._load_settings(self.engine.settings(), channel.limits)
        self._sync_cameras()

    def _camera_state(self, channel):
        """Summarize a camera for its row in the camera list.

        Args: channel is one of self.channels.
        Returns: Short text such as "displayed · 29.8 fps · REC · frozen",
            or "off".
        """
        if not channel.active:
            return "off"
        shown = "displayed · " if channel is self.current else ""
        recording = " · REC" if channel.engine.settings().record else ""
        frozen = " · frozen" if channel.frozen is not None else ""
        return (f"{shown}{channel.engine.status()['processed_fps']:.1f} fps"
                f"{recording}{frozen}")

    def _adopt_live_cameras(self, channel, cameras):
        """After a demo channel switched to a camera, list the real cameras.

        Other synthetic cameras are stopped and removed; the other detected
        cameras are added with processing off.

        Args: channel is the channel that went live; cameras is the list of
            CameraInfo the engine found (may be empty).
        Returns: None.
        """
        for other in self.channels:  # the demo cameras' own profiles, as they were
            if other.demo_source:
                self.profiles.save_profile(other.profile_key, other.profile())
        channel.demo_source = False
        if channel.engine.camera_info is not None:
            channel.info = channel.engine.camera_info
            channel.engine.tag = channel.info.serial
        for other in self.channels:
            if other is not channel and other.demo_source:
                other.stop()
                self.retired_channels.append(other)
        # Dummy-mode values never carry over to a real camera: it gets its own
        # saved profile, or its defaults.
        defaults = channel.default_profile()
        saved = self.profiles.load_profile(channel.profile_key, defaults)
        channel.apply_profile(saved or defaults)
        self.log.write(f"{channel.label}: " + (
            f"saved settings restored ({profiles.summary(saved)})" if saved else
            f"starts from its defaults ({profiles.summary(defaults)})"))
        self.channels = [channel] + [
            restore_channel(self.log, info, profiles=self.profiles) for info in cameras
            if info.serial != channel.info.serial]
        self._track_profiles()
        self._rebuild_cameras()

    def wait_for_engines(self, timeout):
        """Wait until every camera has been released.

        Args: timeout is the maximum wait in seconds per engine.
        Returns: True when every engine has stopped.
        """
        return all([engine.wait(timeout) for engine in self._all_engines()])

    def _all_engines(self):
        """List every engine the window started, running or stopping.

        Args: None.
        Returns: A list of Engine objects.
        """
        return [engine for channel in self.channels + self.retired_channels
                for engine in channel.engines()]

    # ----- behavior ----------------------------------------------------------

    def _set_demo(self, demo):
        """Show or hide every dummy-mode indicator.

        Args: demo is True while the displayed camera uses synthetic video.
        Returns: None.
        """
        identity = None if demo else self.camera_identity
        name = identity.short_name() if identity else "Camera"
        lines = identity.description() if identity else []
        several = len(self.channels) > 1
        app = updater.app_title()
        if demo:
            title = f"{app} — {self.current.label}" if several else app
            self.setWindowTitle(f"{title} — DUMMY MODE")
        elif identity:  # the serial tells same-model cameras apart
            serial = f" ({self.current.info.serial})" if several else ""
            self.setWindowTitle(f"{app} — {name}{serial}")
        else:
            self.setWindowTitle(f"{app} — {self.current.label}" if several else app)
        self.demo_banner.setVisible(demo)
        self.switch_live_button.setVisible(demo)
        self.camera_label.setVisible(not demo)
        self.model_label.setVisible(bool(lines))
        self.model_label.setText("\n".join(
            lines + [f"⚠ {note}" for note in identity.warnings()]) if lines else "")
        self.source_badge.setText("  DUMMY MODE  " if demo else f"  {name}  ")
        self.source_badge.setToolTip(
            "Synthetic video: no camera data" if demo else "\n".join(lines))
        self.source_badge.setStyleSheet(
            "background:#8b0000;color:white;font-weight:bold;" if demo else
            "background:#1f6f3f;color:white;font-weight:bold;")
        self._sync_source_controls()
        self._show_report()

    def _sync_source_controls(self):
        """Fit the camera controls to the displayed source.

        An analog camera behind a capture adapter sets its own exposure, gain
        and frame rate, so those controls are disabled for it; an iPhone's
        are sent to its app, the gain as ISO; interlaced sources also get
        the single-field toggle.

        Args: None.
        Returns: None.
        """
        channel = self.current
        analog = not channel.demo and channel.info.kind == "capture"
        phone = not channel.demo and channel.info.kind == "iphone"
        if analog:
            note = ANALOG_CONTROL_TIP
        elif phone:
            note = IPHONE_CONTROL_TIP
        elif not self.demo_banner.isHidden():
            # isHidden, not isVisible: the banner is flagged shown before the
            # window itself is on screen, which is when this first runs.
            note = SIMULATED_TIP
        else:
            note = ""
        for widget in (self.exposure, self.gain, self.fps):
            widget.setEnabled(not analog)
            widget.set_note(IPHONE_GAIN_TIP if phone and widget is self.gain else note)
        self._sync_phone_controls(channel)
        self.simulated_box.setVisible(channel.demo)
        self.single_field.setVisible(
            not channel.demo and getattr(channel.identity, "interlaced", False))

    def _show_report(self):
        """Show the displayed camera's report status, if it has a report.

        Args: None.
        Returns: None.
        """
        report = None if self.current.demo else self.current.report
        self.report_label.setVisible(report is not None)
        self.report_button.setVisible(False)
        if report is None:
            return
        state = self.report_states.get(report.key, camera_reports.ReportState())
        self.report_label.setText(camera_reports.status_html(state))
        self.report_button.setVisible(camera_reports.needs_button(state))

    def _handle_report_events(self):
        """Record camera-report results from the background worker.

        Args: None.
        Returns: None.
        """
        while True:
            try:
                _, (key, state) = self.reports.events.get_nowait()
            except Exception:
                return
            self.report_states[key] = state
            parts = (state.url, state.error and f"not sent ({state.error})",
                     state.path and f"saved to {state.path}")
            self.log.write(f"Camera report for {key}: "
                           + "; ".join(part for part in parts if part))
            self._show_report()

    def _open_report_in_browser(self):
        """Open GitHub's new-issue page prefilled with the displayed camera's report.

        Args: None.
        Returns: None.
        """
        url = camera_reports.browser_url(self.current.report)
        if not QDesktopServices.openUrl(QUrl(url)):
            QMessageBox.warning(self, "Report this camera", "Could not open a web browser.")

    def handle_key(self, key):
        """Act on a single-key shortcut.

        Args: key is a Qt.Key value.
        Returns: True if the key was a shortcut.
        """
        actions = {
            Qt.Key_Space: lambda: self.set_frozen(self.current.frozen is None),
            Qt.Key_S: self.save_frame,
            Qt.Key_F: lambda: self.engine.request("reference"),
            Qt.Key_C: self.show_compare,
            Qt.Key_P: self.show_pixels,
            Qt.Key_R: lambda: self.engine.request("reset_previous"),
            Qt.Key_A: lambda: self.engine.request("reset_average"),
            Qt.Key_Q: self.close,
            Qt.Key_Escape: self.close,
            Qt.Key_0: self.view.fit,
            Qt.Key_1: self.view.actual_size,
        }
        handler = actions.get(key)
        if handler is None:
            return False
        handler()
        return True

    def set_frozen(self, on):
        """Freeze the displayed camera's image on screen, or return to live.

        Only the display holds still: the engine keeps acquiring, processing
        and recording. The held frame belongs to the displayed camera.

        Args: on is True to hold the frame now on screen, False to resume.
        Returns: None.
        """
        channel = self.current
        if on and channel.frozen is None:
            if self.shown_packet is None:
                self.log.write("No frame on screen to freeze yet.")
            else:
                channel.frozen, channel.frozen_at = self.shown_packet, datetime.now()
                self.log.write(f"Display frozen at {channel.frozen_at:%H:%M:%S}; "
                               "the camera keeps running.")
        elif not on and channel.frozen is not None:
            channel.frozen = channel.frozen_at = None
            self.log.write("Display live again.")
        self._show_frozen()

    def show_compare(self):
        """Open (or raise) the reference comparison window for the displayed camera.

        Args: None.
        Returns: None.
        """
        if self.compare_dialog is None:
            self.compare_dialog = ReferenceCompareDialog(self)
            self.compare_dialog.finished.connect(
                lambda _: self.log.write("Reference comparison closed."))
        if not self.compare_dialog.isVisible():
            self.log.write(f"Reference comparison opened for {self.current.label}.")
        self.compare_dialog.show()
        self.compare_dialog.raise_()
        self.compare_dialog.activateWindow()

    def show_pixels(self):
        """Open (or raise) the pixel values window for the displayed camera.

        Args: None.
        Returns: None.
        """
        if self.pixels_dialog is None:
            self.pixels_dialog = PixelSquareDialog(self)
            self.pixels_dialog.finished.connect(
                lambda _: self.log.write("Pixel values closed."))
        if not self.pixels_dialog.isVisible():
            self.log.write(f"Pixel values opened for {self.current.label}.")
        self.pixels_dialog.show()
        self.pixels_dialog.raise_()
        self.pixels_dialog.activateWindow()

    def _image_clicked(self, x, y):
        """Move the pixel square to a clicked pixel while its window is open.

        Args: x and y are the image pixel clicked.
        Returns: None.
        """
        if self.pixels_dialog is not None and self.pixels_dialog.isVisible():
            self.pixels_dialog.move_to(x, y)

    def _ensure_daq(self):
        """Create the DAQ controller on first use.

        Its events are read by poll_daq, from _poll_status (and from the DAQ
        window's own timer while it is open).

        Args: None.
        Returns: The daq.DaqController.
        """
        if self.daq is None:
            self.daq = daq.DaqController(self.log)
        return self.daq

    def _auto_connect_daq(self):
        """At startup, reconnect the NI DAQ saved with a piezo, for the Phase view.

        Connecting sets every output to 0 V; nothing is driven until the
        Phase view steps the piezo. A saved simulated device is reconnected
        only for a synthetic camera, since it cannot move a real camera's
        reference beam.

        Args: None.
        Returns: None.
        """
        saved = self._saved_piezo()
        if saved is None or not saved.device:
            return
        if saved.device == daq.SIMULATED and not self.current.demo:
            return
        self.daq_auto = saved.device
        self.log.write(f"NI DAQ: looking for {saved.device}; its piezo drives the "
                       "Phase view.")
        self._ensure_daq().request("scan")

    @staticmethod
    def _saved_piezo():
        """The DAQ settings saved by Apply, when they assign a piezo driver.

        A piezo counts as configured only after Apply has saved settings that
        assign it: the DAQ window's defaults name AO 1 for the first driver,
        and merely opening that window must never start driving anything.

        Args: None.
        Returns: daq.DaqSettings, or None when nothing is saved or neither
            driver is assigned.
        """
        if not QSettings("ESPI", "espi_live").value(DAQ_SETTINGS_KEY, ""):
            return None
        saved = DaqDialog._saved_settings()
        return saved if any(daq.piezo_assigned(saved, index)
                            for index in daq.PIEZOS) else None

    def received_fps_for(self, key):
        """The rate frames arrive from a camera, chosen by its profile key.

        Args: key is the camera's profile key, or "" for the displayed one.
        Returns: Frames per second over the last second (0.0 before any
            arrive, or when the camera is not open).
        """
        if not key:
            return self.received_fps()
        channel = self._channel_by_key(key)
        if channel is None or channel.engine is None:
            return 0.0
        return float(channel.engine.status().get("processed_fps") or 0.0)

    def received_fps(self):
        """The rate frames arrive from the displayed camera.

        Args: None.
        Returns: Frames per second over the last second (0.0 before any arrive).
        """
        return float(self.engine.status().get("processed_fps") or 0.0)

    def _channel_by_key(self, key):
        """Find a channel by its profile key.

        Args: key is a profiles.profile_key value, or "".
        Returns: The Channel, or None.
        """
        return next((channel for channel in self.channels
                     if channel.profile_key == key), None)

    def _driver_of(self, channel):
        """Which piezo driver a camera is paired to.

        Args: channel is one of self.channels.
        Returns: The driver index, 1 or 2, whose pairing names this camera;
            1 when nothing pairs it (the historical default: Thor 1 follows
            the displayed camera).
        """
        for index in (2, 1):
            if self.piezo_pairing.get(index) == channel.profile_key:
                return index
        return 1

    def poll_daq(self, defer_devices=False):
        """Handle the DAQ controller's events, for the window and the DAQ window.

        The control server drains the same queue while a DAQ request of its
        own is waiting for an answer (it holds daq_events_lock and dispatches
        each event exactly as this method would), so the drain here is
        skipped while it holds the lock.

        Args: defer_devices keeps a device list for the DAQ window until it
            is shown (as it is kept while that window is closed), because the
            window connects to a device when it gets one.
        Returns: None; nothing is read once the window is closing (the DAQ is
            already shut down then).
        """
        if self.daq is None or self.closing:
            return
        self.daq.heartbeat()  # the DAQ zeroes the outputs if this stops while they are on
        if not self.daq_events_lock.acquire(blocking=False):
            return
        try:
            self._drain_daq_events(defer_devices)
        finally:
            self.daq_events_lock.release()

    def _drain_daq_events(self, defer_devices):
        """Read every queued DAQ event and dispatch it.

        Runs on the GUI thread with daq_events_lock held.

        Args: defer_devices keeps a device list for the DAQ window until it
            is shown, because the window connects to a device when it gets
            one.
        Returns: None.
        """
        later = []
        while True:
            try:
                kind, payload = self.daq.events.get_nowait()
            except queue.Empty:
                break
            dialog = self.daq_dialog
            if kind == "devices" and dialog is not None and (
                    defer_devices or not dialog.isVisible()):
                later.append((kind, payload))
                continue
            self._daq_event(kind, payload)
            if dialog is not None:
                dialog.handle(kind, payload)
        for event in later:
            self.daq.events.put(event)

    def _control_daq_form(self, settings):
        """Refresh the NI DAQ window's form after the control server changed it.

        Runs on the GUI thread (scheduled by the control backend). The
        applied event already updates the form while the window is open;
        this covers paths that send no event, such as a held piezo level.

        Args: settings is the DaqSettings to show in the form.
        Returns: None.
        """
        dialog = self.daq_dialog
        if dialog is None:
            return
        dialog.load(settings)
        dialog.refresh()

    def _control_daq_checklist(self, settings):
        """Show the safety checklist for settings the control server wants applied.

        Runs on the GUI thread. The DAQ window gets it through its own
        checklist flow when it is open; otherwise a checklist opens over the
        main window, and ticking it arms the device and applies the settings.

        Args: settings is the daq.DaqSettings the checklist is shown for.
        Returns: None.
        """
        dialog = self.daq_dialog
        if dialog is not None:
            dialog.load(settings)
            dialog._ask_checklist(settings)
            return
        box = DaqChecklistDialog(self, settings)

        def allowed():
            """Arm the device and apply the settings (the checklist's Allow)."""
            self.daq.arm(settings)
            self.log.write("NI DAQ: safety checklist ticked off for "
                           f"{settings.device or 'the device'} (from the control server).")
            self.daq.request("apply", settings=settings)

        box.accepted.connect(allowed)
        box.open()

    def _control_daq_attach(self, index, settings, simulated):
        """Display a camera and give its Phase view the piezo (control server).

        Runs on the GUI thread. A camera other than the displayed one is
        displayed first, as a person would.

        Args: index is the channel's position (out of range: the displayed
            one); settings is the DaqSettings the stepper steps with;
            simulated says whether the device drives no hardware.
        Returns: The camera's label.
        """
        channel = self.current
        if isinstance(index, int) and 0 <= index < len(self.channels):
            channel = self.channels[index]
        if channel is not self.current:
            self.show_channel(channel)
        self._attach_stepper(phase.PiezoStepper(self.daq, settings, simulated))
        return channel.label

    def _follow_frame_rate(self):
        """Keep piezo triangles that follow a frame rate at the frequency it needs.

        Each driver follows the frame rate of the camera it is paired to (the
        displayed one when nothing is paired). The frequency is set again
        only when the frame rate moves it by more than daq.FOLLOW_TOLERANCE,
        since each change restarts the triangle.

        Args: None.
        Returns: None.
        """
        if self.daq is None or self.closing:
            return
        if self.piezo_tuner is not None:  # auto-tune follows the frame rate itself
            self._run_tuner()
            return
        applied = self.daq.status()["settings"]
        if applied is None:
            return
        wanted = applied
        changes = {}
        for index in daq.PIEZOS:
            if not daq.pget(applied, index, "on") or \
                    not daq.pget(applied, index, "follow_fps"):
                continue
            pending = self.piezo_follow_hz.get(index)
            if pending is not None:  # a change is on its way
                wanted = daq.pset(wanted, index, frequency_hz=pending)
                continue
            fps = self.received_fps_for(daq.pget(applied, index, "camera"))
            settings = daq.followed(applied, fps, index)
            if settings is None:
                continue
            wanted = daq.pset(wanted, index,
                              frequency_hz=daq.pget(settings, index, "frequency_hz"))
            changes[index] = (fps, daq.pget(settings, index, "frequency_hz"),
                              daq.pget(settings, index, "phase_per_frame_deg"))
        if self.daq_dialog is not None and self.daq_dialog.isVisible():
            self.daq_dialog.follow_frame_rate()
        if not changes or not self.daq.is_armed(wanted):
            return
        for index, (fps, frequency, per_frame) in changes.items():
            which = ("the piezo triangle follows the camera's" if index == 1
                     else "the second piezo's triangle follows its camera's")
            self.log.write(f"NI DAQ: {which} {fps:.1f} fps: "
                           f"{frequency:.4g} Hz for {per_frame:g}° per frame.")
            self.piezo_follow_hz[index] = frequency
        self.daq.request("apply", settings=wanted)

    def set_auto_tune(self, on, reason=""):
        """Start or stop the experimental auto-tune of the piezo triangle.

        It needs an applied triangle with the piezo on. Stopping by hand keeps
        the tuned frequency (and puts the triangle back after the noise
        hold); stopping for a reason (the DAQ stopped, settings changed) sends
        nothing.

        Args: on is True to start; reason says why it stops, "" when the
            user turned it off.
        Returns: None.
        """
        if on and self.piezo_tuner is None:
            applied = self.daq.status()["settings"] if self.daq is not None else None
            index = self._driver_of(self.current)
            if applied is None or not daq.pget(applied, index, "on") or \
                    not daq.piezo_assigned(applied, index) or \
                    daq.pget(applied, index, "mode") != daq.TRIANGLE:
                self.tune_note = ("Auto-tune needs a piezo triangle: choose Triangle, tick "
                                  "Piezo on and Apply first.")
            else:
                self.piezo_tuner = step_tuning.StepTuner()
                self.tune_index = index
                self.tune_engine, self.tune_settings = self.engine, applied
                self.tune_sent, self.tune_note = None, ""
                which = "" if index == 1 else "Thor 2's "
                self.log.write("Auto-tune (experimental): on. The piezo holds still to "
                               "measure the camera noise, then the triangle's frequency is "
                               f"tuned for a 180° step per frame on {which}output. Keep "
                               "the object still.")
        elif not on and self.piezo_tuner is not None:
            tuner, self.piezo_tuner = self.piezo_tuner, None
            index, self.tune_index = self.tune_index, None
            wanted = self._tune_target(tuner.scale, False)
            self.tune_note = f"Auto-tune stopped: {reason}." if reason else ""
            if reason:
                self.log.write(f"Auto-tune (experimental): stopped, {reason}.")
            else:
                self.log.write(f"Auto-tune (experimental): off; the triangle keeps "
                               f"{daq.pget(wanted, index, 'frequency_hz'):.4g} Hz "
                               f"(×{tuner.scale:.3f}).")
                if self.tune_sent is None or self.tune_sent[0] != wanted:
                    self._send_tuned(wanted)
        self._sync_step_monitor()
        if self.daq_dialog is not None:
            self.daq_dialog.show_auto_tune()

    def tune_text(self):
        """Say what auto-tune is doing, or why it last stopped.

        Args: None.
        Returns: One sentence, or "".
        """
        if self.piezo_tuner is not None:
            return self.piezo_tuner.describe()
        return self.tune_note

    def _tune_target(self, scale, hold):
        """Work out the piezo settings auto-tune wants now.

        Args: scale is the tuner's factor on the frequency; hold is True while
            it measures the noise with the piezo still.
        Returns: daq.DaqSettings: the tuned triangle (its frequency follows
            the frame rate when Follow is on), or the piezo held at the
            triangle's centre.
        """
        index = self.tune_index or 1
        base = self.tune_settings
        if hold:
            return daq.pset(base, index, mode=daq.HOLD,
                            voltage_v=(daq.pget(base, index, "bottom_v")
                                       + daq.pget(base, index, "top_v")) / 2)
        frequency = daq.pget(base, index, "frequency_hz")
        if daq.pget(base, index, "follow_fps"):
            computed, _ = daq.follow_frequency(base, self.received_fps(), index)
            frequency = computed or frequency
        frequency = min(daq.MAX_TRIANGLE_HZ, max(daq.MIN_TRIANGLE_HZ, frequency * scale))
        return daq.pset(base, index, frequency_hz=frequency)

    def _send_tuned(self, settings):
        """Apply settings auto-tune chose, if the safety checklist allows them.

        Args: settings is daq.DaqSettings.
        Returns: True when they were sent.
        """
        if not self.daq.is_armed(settings):
            return False
        self.daq.request("apply", settings=settings)
        return True

    def _run_tuner(self):
        """Measure, let the tuner decide, and apply what it wants.

        Args: None.
        Returns: None.
        """
        tuner = self.piezo_tuner
        if self.engine is not self.tune_engine:
            self.set_auto_tune(False, "the displayed camera changed")
            return
        if self.engine.settings().mode == "Phase":
            self.set_auto_tune(False, "the Phase view steps the piezo itself; use the "
                               "Previous view")
            return
        now = time.monotonic()
        before = tuner.state
        tuner.update(now, self.tune_engine.step_monitor)
        if tuner.state != before and tuner.state in ("holding", "lost"):
            self.log.write(f"Auto-tune (experimental): {tuner.describe()}")
        wanted = self._tune_target(tuner.scale, tuner.wants_hold)
        sent = self.tune_sent
        key = (tuner.scale, tuner.wants_hold)
        index = self.tune_index or 1
        if sent is not None and (sent[1], sent[2]) == key:
            old = daq.pget(sent[0], index, "frequency_hz")
            if abs(daq.pget(wanted, index, "frequency_hz") - old) <= \
                    daq.FOLLOW_TOLERANCE * old:
                return  # nothing new (a small frame-rate drift is left alone)
        if not self._send_tuned(wanted):
            self.set_auto_tune(False, "the safety checklist does not cover these outputs")
            return
        self.tune_sent = (wanted, tuner.scale, tuner.wants_hold)
        tuner.changed(now)
        if not tuner.wants_hold:
            self.log.write(f"Auto-tune (experimental): "
                           f"{daq.pget(wanted, index, 'frequency_hz'):.4g} Hz "
                           f"(×{tuner.scale:.3f}).")

    def _sync_step_monitor(self):
        """Measure the step only on the displayed camera, and only when needed.

        Args: None.
        Returns: None.
        """
        wanted = self.piezo_tuner is not None or self.piezo_debug_action.isChecked()
        for channel in self.channels:
            engine = channel.engine
            if engine is not None:
                on = wanted and channel is self.current
                if engine.step_monitor.on != on:
                    engine.step_monitor.switch(on)

    def _show_piezo_debug(self, *_):
        """Draw or hide the piezo debug overlay over the image.

        Args: _ is the menu action's checked state (ignored; the action is read).
        Returns: None.
        """
        on = self.piezo_debug_action.isChecked()
        self.piezo_debug.setVisible(on)
        if on:
            self._sync_step_monitor()
            self.piezo_debug.setText(self.piezo_debug_text())

    def piezo_debug_text(self):
        """Build the overlay's lines from the DAQ, the frames and auto-tune.

        Args: None.
        Returns: Text of a few lines.
        """
        fps = self.received_fps()
        lines = [f"camera   {fps:.1f} fps received"]
        applied = self.daq.status()["settings"] if self.daq is not None else None
        if applied is None:
            lines.append("piezo    not driven")
        for index in daq.PIEZOS:
            if applied is None:
                break
            name = "piezo" if index == 1 else "piezo2"
            if not daq.pget(applied, index, "on") or not daq.piezo_assigned(applied, index):
                if daq.piezo_assigned(applied, index) or index == 1:
                    lines.append(f"{name}   not driven")
                continue
            mode = daq.pget(applied, index, "mode")
            driver_fps = self.received_fps_for(daq.pget(applied, index, "camera")) \
                if daq.pget(applied, index, "camera") else fps
            if mode == daq.TRIANGLE:
                lines.append(f"{name}   triangle {daq.pget(applied, index, 'bottom_v'):g}–"
                             f"{daq.pget(applied, index, 'top_v'):g} V at "
                             f"{daq.pget(applied, index, 'frequency_hz'):.4g} Hz")
                computed, reason = daq.follow_frequency(applied, driver_fps, index)
                lines.append("formula  " + (
                    f"{computed:.4g} Hz for "
                    f"{daq.pget(applied, index, 'phase_per_frame_deg'):g}°/frame"
                    if computed is not None else reason))
            elif mode == daq.HOLD:
                lines.append(f"{name}   held at {daq.pget(applied, index, 'voltage_v'):g} V")
            else:
                lines.append(f"{name}   sine {daq.pget(applied, index, 'amplitude_v'):g} V peak")
        tuner = self.piezo_tuner
        noise = tuner.noise if tuner is not None else 0.0
        stats = self.engine.step_monitor.stats(time.monotonic() - PIEZO_DEBUG_WINDOW_S,
                                               noise)
        if stats is None:
            lines.append("step     waiting for frames")
        else:
            measured = "" if noise else " (noise not taken off)"
            lines.append(f"step     {stats.step_deg:.0f}°/frame  D2/D1 {stats.q:.2f}"
                         f"{measured}")
            lines.append(f"D1       {stats.d1:.4g}  noise {noise:.4g}  flicker "
                         f"{stats.flicker:.0%}")
        if tuner is not None:  # short, so the overlay stays clear of the watermark
            lines.append(f"tune     {tuner.state} ×{tuner.scale:.3f}")
        return "\n".join(lines)

    def _daq_event(self, kind, payload):
        """Follow the piezos: attach them to their paired cameras, or take them away.

        A device whose saved settings assign a piezo driver makes the Phase
        view that driver's paired camera's view (the displayed camera when
        nothing is paired, as it always has been). Applying settings in the
        DAQ window updates the steps; 0 V, an error or a disconnect stops
        them.

        Args: kind and payload are the controller's event.
        Returns: None.
        """
        if kind in ("applied", "rejected", "zeroed", "failed", "closed", "fault"):
            self.piezo_follow_hz = {}  # frequency changes sent have been answered
        if kind in ("rejected", "zeroed", "failed", "closed", "fault") and \
                self.piezo_tuner is not None:
            self.set_auto_tune(False, f"the NI DAQ outputs were {kind}"
                               if kind != "rejected" else "the NI DAQ refused a setting")
        if kind == "devices" and self.daq_auto is not None and self.daq_dialog is None:
            wanted, self.daq_auto = self.daq_auto, None
            names = [info.name for info in payload[0]] + [daq.SIMULATED]
            if wanted in names:
                self.daq.request("connect", device=wanted)
            else:
                self.log.write(f"NI DAQ: {wanted} was not found, so the Phase view "
                               "has no piezo. Plug it in and open Devices > NI DAQ.")
        elif kind == "connected":
            self.set_daq_drive(daq.Drive())  # a new connection starts at 0 V
            saved = self._saved_piezo()
            if saved is None:
                self._take_steppers()
                return
            # Everything is at 0 V after connecting, so the generator and both
            # drivers stay off until a step or an Apply turns one on.
            base = replace(saved, device=payload.device.name, generator_on=False)
            off_both = daq.pset(daq.pset(base, 1, on=False), 2, on=False)
            simulated = payload.device.simulated
            for index in daq.PIEZOS:
                stepper = None
                if daq.piezo_assigned(base, index) and \
                        self.daq.is_armed(daq.pset(off_both, index, on=True)):
                    stepper = phase.PiezoStepper(self.daq, off_both, simulated, index)
                self.phase_steppers[index] = stepper
            self._read_pairing(base)
            if not any(self.phase_steppers.values()):
                self._take_steppers()
                self.log.write("Phase view: the NI DAQ piezo waits for the safety "
                               "checklist (Devices > NI DAQ…, then Apply with the "
                               "piezo on).")
                return
            self._attach_steppers(self._arrived())
        elif kind == "applied":
            self.set_daq_drive(payload.drive)
            applied = self.daq.status()["settings"]
            device = self.daq.status()["device"]
            simulated = device is not None and device.simulated
            for index in daq.PIEZOS:
                self.phase_steppers[index] = (
                    phase.PiezoStepper(self.daq, applied, simulated, index)
                    if applied is not None and daq.piezo_assigned(applied, index)
                    else None)
            self._read_pairing(applied)
            self._attach_steppers(self._arrived())
        elif kind in ("zeroed", "failed", "closed", "fault"):
            if kind == "fault":
                self.log.write(f"NI DAQ self-check: {payload}. Every output was set to 0 V.")
            self.set_daq_drive(daq.Drive())
            if any(self.phase_steppers.values()):
                self.log.write("Phase view: piezo stepping stopped (the NI DAQ outputs "
                               "are at 0 V or disconnected).")
                self._take_steppers()

    def _read_pairing(self, settings):
        """Record which camera each driver steps, from DaqSettings.

        Args: settings is daq.DaqSettings (the applied or saved ones), or
            None to forget the pairing.
        Returns: None.
        """
        self.piezo_pairing = {index: (daq.pget(settings, index, "camera")
                                      if settings is not None else "")
                              for index in daq.PIEZOS}

    def _take_steppers(self):
        """Remove every piezo stepper from every engine.

        Args: None.
        Returns: None.
        """
        self.phase_steppers = {index: None for index in daq.PIEZOS}
        self._attach_steppers()

    def _arrived(self):
        """Name the drivers whose stepper is new since the last attach.

        Args: None.
        Returns: A frozenset of driver indexes whose stepper object changed
            (a new attach), so the Phase view may start for its camera.
        """
        arrived = frozenset(index for index in daq.PIEZOS
                            if self.phase_steppers.get(index) is not None
                            and getattr(self, "_attached_steppers", {}).get(index)
                            is not self.phase_steppers.get(index))
        self._attached_steppers = dict(self.phase_steppers)
        return arrived

    def _attach_stepper(self, stepper):
        """Attach one driver's stepper, or take every stepper away.

        The historical single-stepper entry point: a stepper attaches where
        its pairing (or, with none, the displayed camera) says; None clears
        every driver.

        Args: stepper is a phase.PiezoStepper, or None.
        Returns: None.
        """
        if stepper is None:
            self._take_steppers()
            return
        self.phase_steppers[stepper.index] = stepper
        self._attach_steppers(self._arrived())

    def _attach_steppers(self, arrivals=frozenset()):
        """Give each paired camera's engine its driver's stepper, and only that.

        A driver whose pairing names a camera attaches there; a camera two
        drivers name goes to Thor 1. When no driver is paired at all, Thor 1
        follows the displayed camera, as it always has. A stepper that just
        arrived for the displayed camera makes the Phase view its view.

        Args: arrivals is a frozenset of driver indexes whose stepper is new
            (from _arrived), for the Phase-view auto-selection.
        Returns: None.
        """
        previous = self.stepper_claims
        claims = {}
        paired_any = any(self.piezo_pairing.values())
        for index in daq.PIEZOS:
            stepper = self.phase_steppers.get(index)
            if stepper is None:
                continue
            channel = self._channel_by_key(self.piezo_pairing.get(index, ""))
            if channel is not None and channel not in claims:
                claims[channel] = index
            elif channel is None and not paired_any and self.current not in claims:
                claims[self.current] = index
        self.stepper_claims = claims
        for channel in self.channels:
            engine = channel.engine
            if engine is None:
                continue
            index = claims.get(channel)
            engine.set_phase_stepper(
                self.phase_steppers.get(index) if index is not None else None)
        self.phase_stepper = (self.phase_steppers.get(claims[self.current])
                              if self.current in claims else None)
        if self.current in claims and previous.get(self.current) != \
                claims[self.current] and claims[self.current] in arrivals:
            self.engine.update_settings(mode="Phase")
            self._load_settings(self.engine.settings(), self.current.limits)
            stepper = self.phase_stepper
            self.log.write(f"NI DAQ piezo on {daq.output_label(daq.pget(stepper.settings, claims[self.current], 'output'))} "
                           f"(Thor {claims[self.current]}): Phase view on "
                           "(phase-shifting ESPI).")
        self._show_phase_state()

    def _show_phase_state(self):
        """Describe the displayed camera's piezo under Phase shifting and enable Calibrate.

        Args: None.
        Returns: None.
        """
        stepper = self.phase_stepper
        self.calibrate_phase_button.setEnabled(stepper is not None)
        # The Phase view button takes toolbar room only when it can work, or
        # while the displayed camera is in that view (so it can be left).
        self.phase_mode_action.setVisible(
            stepper is not None or self.engine.settings().mode == "Phase")
        if stepper is None:
            self.phase_state.setText(
                "Piezo: not connected. Assign a piezo driver in Devices > NI DAQ "
                "(and pair it to this camera there); the Phase view then starts "
                "by itself.")
            return
        s = stepper.settings
        simulated = " Simulated device: nothing is driven." if stepper.simulated else ""
        thor = "" if stepper.index == 1 else " (Thor 2)"
        self.phase_state.setText(
            f"Piezo{thor}: {daq.output_label(daq.pget(s, stepper.index, 'output'))}, "
            f"from {daq.pget(s, stepper.index, 'voltage_v'):g} V "
            f"in steps of {daq.pget(s, stepper.index, 'step_v'):.4g} V.{simulated}")

    def calibrate_phase(self):
        """Measure the piezo's phase step and set the voltage per step.

        Args: None.
        Returns: None; the result arrives as a "phase_calibrated" event.
        """
        if self.phase_stepper is None:
            QMessageBox.information(
                self, "Calibrate phase step",
                "Connect the NI DAQ and assign the piezo driver first "
                "(Devices > NI DAQ).")
            return
        self.engine.update_settings(mode="Phase")
        self._load_settings(self.engine.settings(), self.current.limits)
        self.engine.request("calibrate_phase")

    def _save_piezo_step(self, volts, index=1):
        """Keep a calibrated voltage per step in the saved DAQ settings.

        Args: volts is the new voltage per phase step; index is the driver it
            was measured on, 1 or 2.
        Returns: None.
        """
        saved = daq.pset(DaqDialog._saved_settings(), index, step_v=volts,
                         quarter_wave_v=volts)
        QSettings("ESPI", "espi_live").setValue(DAQ_SETTINGS_KEY,
                                                json.dumps(saved.to_dict()))
        if self.daq_dialog is not None:
            self.daq_dialog.set_quarter_wave(volts, index)

    def _build_phase_controls(self, add, note, s):
        """Create the Phase shifting controls.

        Args: add and note place widgets in the controls grid (see
            _build_controls); s is the displayed camera's Settings.
        Returns: None.
        """
        self.phase_steps = WheelIgnoringComboBox()
        for steps, text in PHASE_STEP_TEXT:
            self.phase_steps.addItem(text, steps)
        self.phase_steps.setCurrentIndex(self.phase_steps.findData(s.phase_steps))
        set_tooltip(self.phase_steps,
            "How many phase steps make one phase map. 5-step (Hariharan) "
            "tolerates a piezo step that is not exactly a quarter wave and "
            "measures the step it made (shown in the status bar). 4-step is "
            "faster but turns a wrong step into ripple across the fringes. "
            "Start with 5-step.")
        self.phase_steps.currentIndexChanged.connect(
            lambda _index: self.engine.update_settings(
                phase_steps=int(self.phase_steps.currentData())))
        add(self.phase_steps, "Phase steps")
        self.phase_filter = SliderSpin(0, phase.MAX_PHASE_FILTER, 1, 0, s.phase_filter,
                                       " px")
        self.phase_filter.set_description(
            "Width of the sine/cosine filter on the phase change. It smooths "
            "speckle noise without blurring across the jump from black to "
            "white, and weights strongly interfering speckles most. 0 turns it "
            "off. Keep it below the fringe spacing. Starts at 9 px.")
        self.phase_filter.valueChanged.connect(
            lambda v: self.engine.update_settings(phase_filter=int(v)))
        add(self.phase_filter, "Phase filter")
        self.phase_settle = SliderSpin(0, phase.MAX_SETTLE_FRAMES, 1, 0,
                                       s.phase_settle_frames, " fr")
        self.phase_settle.set_description(
            "Camera frames thrown away after each piezo step, because they were "
            "exposed while the piezo moved or settled. Raise it if the measured "
            "step wanders or the phase map is noisy right after a step; a slow "
            "piezo amplifier or a capture stick that buffers frames needs more. "
            "Starts at 2.")
        self.phase_settle.valueChanged.connect(
            lambda v: self.engine.update_settings(phase_settle_frames=int(v)))
        add(self.phase_settle, "Settle")
        self.calibrate_phase_button = QPushButton("Calibrate phase step")
        set_tooltip(self.calibrate_phase_button,
            "Measure the phase step the piezo makes over five 5-step sets, with "
            "the object still, then set the NI DAQ's voltage per step so that "
            "each step is a quarter wave (90°). The new voltage is saved with "
            "the DAQ settings and a new phase reference is taken.")
        self.calibrate_phase_button.clicked.connect(self.calibrate_phase)
        add(self.calibrate_phase_button)
        self.phase_state = QLabel()
        note(self.phase_state)
        self._show_phase_state()

    def _show_daq_dialog(self):
        """Open the NI DAQ window, creating its controller on first use.

        Args: None.
        Returns: None.
        """
        self._ensure_daq()
        if self.daq_dialog is None:
            self.daq_dialog = DaqDialog(self, self.daq)
        self.daq_dialog.show()
        self.daq_dialog.raise_()
        self.daq_dialog.activateWindow()

    def _show_frozen(self):
        """Show or hide every frozen-display indicator for the displayed camera.

        Args: None.
        Returns: None.
        """
        frozen = self.current.frozen is not None
        if frozen:
            self.frozen_banner.setText(
                f"FROZEN at {self.current.frozen_at:%H:%M:%S} — not live · "
                "Space resumes")
        self.frozen_banner.setVisible(frozen)
        self.frozen_badge.setVisible(frozen)
        for widget in (self.freeze_button, self.freeze_action):
            widget.blockSignals(True)
            widget.setChecked(frozen)
            widget.blockSignals(False)

    def save_frame(self):
        """Ask the engine to save the displayed frame with the title and notes.

        While the display is frozen, the held frame is saved.

        Args: None.
        Returns: None.
        """
        frozen = {} if self.current.frozen is None else {
            "packet": self.current.frozen, "frozen_at": self.current.frozen_at}
        self.engine.request("save", title=self.frame_title.text(),
                            notes=self.frame_notes.toPlainText(), **frozen)

    def _show_zoom(self, zoom):
        """Sync the zoom slider and percentage with the view.

        Args: zoom is the view's zoom relative to fit-to-window.
        Returns: None.
        """
        self.zoom_slider.blockSignals(True)  # don't feed the value back to the view
        self.zoom_slider.setValue(round(ZOOM_TICKS * math.log(zoom) / math.log(MAX_ZOOM)))
        self.zoom_slider.blockSignals(False)
        percent = f"{100 * zoom:.0f}%"
        self.zoom_label.setText(percent)
        self.zoom_slider.setToolTip(format_tooltip(ZOOM_SLIDER_TIP, f"Now: {percent}"))

    def _set_hover(self, x, y):
        """Remember the image pixel under the cursor for the readout.

        Args: x and y are image coordinates, or -1 when outside.
        Returns: None.
        """
        self.hover = (x, y)

    def _poll_frame(self):
        """Draw the newest engine frame if it has not been drawn yet.

        Nothing is drawn while the displayed camera is frozen. In split view
        the second pane is drawn here too; in overlay mode the blend of the
        two streams replaces the single image.

        Args: None.
        Returns: None.
        """
        if self.current.frozen is not None:
            return
        if self.display_mode == "overlay":
            self._draw_overlay()
            return
        packet = self.engine.latest_packet()
        if packet is None or packet.sequence == self.drawn_sequence:
            return
        self.drawn_sequence = packet.sequence
        self._draw(packet)
        self.drawn_frames += 1
        if self.display_mode == "split":
            self._draw_second()

    def _draw(self, packet):
        """Put a frame on screen, rebuilding the color table only when needed.

        Args: packet is the FramePacket to show.
        Returns: None.
        """
        key = round(packet.gamma, 4)
        if key != getattr(self, "table_key", None):
            self.table = display_rgb_table(packet.gamma)
            self.table_key = key
        self.view.show_packet(packet, self.table, key)
        self.shown_packet = packet

    # ----- two streams ---------------------------------------------------------

    def _pane_caption(self):
        """Build a small caption naming a pane's camera.

        Args: None.
        Returns: A QLabel that never takes mouse events and starts hidden.
        """
        label = QLabel()
        label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._styled(label, "color:white;background:rgba(0,0,0,55%);"
                     "padding:2px 8px;border-radius:3px;")
        label.setVisible(False)
        return label

    def _second_clicked(self, x, y):
        """Make the second pane's camera the displayed one.

        Args: x and y are the image pixel clicked.
        Returns: None.
        """
        if self.display_mode == "split" and self.second_channel is not None:
            self.show_channel(self.second_channel)
        self._image_clicked(x, y)

    def _active_channels(self):
        """List the channels that are processing.

        Args: None.
        Returns: A list of Channel.
        """
        return [channel for channel in self.channels if channel.active]

    def _sync_stream_actions(self):
        """Enable the two-stream actions and check the split/overlay ones.

        Args: None.
        Returns: None.
        """
        two = len(self._active_channels()) >= 2
        self.split_action.setEnabled(two)
        self.overlay_action.setEnabled(two)
        self.stream_window_action.setEnabled(two)
        self.split_action.blockSignals(True)
        self.overlay_action.blockSignals(True)
        self.split_action.setChecked(self.display_mode == "split")
        self.overlay_action.setChecked(self.display_mode == "overlay")
        self.split_action.blockSignals(False)
        self.overlay_action.blockSignals(False)
        self._show_captions()

    def set_display_mode(self, mode):
        """Show one stream, two split panes, or two streams overlaid.

        Args: mode is "single", "split" or "overlay".
        Returns: None.
        """
        if mode == self.display_mode:
            return
        if mode != "single" and len(self._active_channels()) < 2:
            return
        previous, self.display_mode = self.display_mode, mode
        self.second_channel = self._pick_second_channel()
        self.second_sequence = None
        self.drawn_sequence = 0
        self.second_view.setVisible(mode == "split")
        if mode == "single":
            if previous != "single":
                self.log.write("Hiding the second stream.")
            self.second_view.clear()
        elif self.second_channel is not None:
            self.log.write(f"{'Showing' if mode == 'split' else 'Overlaying'} two "
                           f"streams: {self.current.label} and "
                           f"{self.second_channel.label}.")
        self._sync_stream_actions()
        self._show_captions()

    def _pick_second_channel(self):
        """Choose the channel the second pane (or window) shows.

        Args: None.
        Returns: The first active channel that is not the displayed one, or
            None when there is no other.
        """
        return next((channel for channel in self._active_channels()
                     if channel is not self.current), None)

    def _show_captions(self):
        """Name each pane's camera, in split view only.

        Args: None.
        Returns: None.
        """
        show = self.display_mode == "split" and self.second_channel is not None
        self.view_caption.setText(self.current.label if show else "")
        self.view_caption.setVisible(show)
        self.second_caption.setText(self.second_channel.label if show else "")
        self.second_caption.setVisible(show)

    def _stream_follow(self):
        """Re-point the second stream after cameras opened, closed or switched.

        Args: None.
        Returns: None.
        """
        if self.display_mode != "single" or self.stream_window is not None:
            self.second_channel = self._pick_second_channel()
        self._sync_stream_actions()
        if self.stream_window is not None:
            self.stream_window.set_channel(self.second_channel)

    def _draw_second(self):
        """Draw the second pane's newest frame.

        Args: None.
        Returns: None.
        """
        packet = self.second_channel.engine.latest_packet() \
            if self.second_channel is not None and self.second_channel.engine \
            is not None else None
        if packet is None or packet.sequence == self.second_sequence:
            return
        self.second_sequence = packet.sequence
        key = round(packet.gamma, 4)
        table = display_rgb_table(packet.gamma)
        self.second_view.show_packet(packet, table, key)

    def _blend_packets(self, first, second):
        """Blend two cameras' packets into one, for the overlay display.

        The two images are shown exactly as each would be alone (display
        tone applied), then averaged. A different frame size is scaled to
        the displayed camera's. Dummy mode wins: a blend involving a demo
        stream keeps the dummy-mode watermark.

        Args: first and second are FramePacket.
        Returns: A FramePacket with gamma 1 (already display-applied) and
            the displayed camera's mode and sequence.
        """
        image = first.render().astype(np.float32)
        other = second.render()
        if other.shape != image.shape:
            other = cv2.resize(other, (image.shape[1], image.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        blended = np.round((image + other) / 2.0).astype(np.uint8)
        return FramePacket(first.sequence, blended, 1.0, first.mode,
                           first.demo or second.demo)

    def _draw_overlay(self):
        """Draw the blend of the two streams into the main view.

        Args: None.
        Returns: None.
        """
        packets = []
        for channel in (self.current, self.second_channel):
            packet = channel.engine.latest_packet() if channel is not None and \
                channel.engine is not None else None
            packets.append(packet)
        if packets[0] is None or packets[1] is None:
            return
        key = (packets[0].sequence, packets[1].sequence)
        if key == self.second_sequence:
            return
        self.second_sequence = key
        blended = self._blend_packets(packets[0], packets[1])
        table = display_rgb_table(1.0)
        self.view.show_packet(blended, table, 1.0)
        self.drawn_frames += 1

    def open_stream_window(self):
        """Open a second window showing another camera's live stream.

        Args: None.
        Returns: None.
        """
        if self.stream_window is None:
            self.stream_window = StreamWindow(self)
        self.second_channel = self.second_channel or self._pick_second_channel()
        self.stream_window.set_channel(self.second_channel)
        self.stream_window.show()
        self.stream_window.raise_()

    def _log_view_state(self):
        """Log, at DEBUG, each change of what size the image is drawn at.

        A frame shown smaller than its sensor size is filtered down by the
        display, which can look like lost resolution; this records when the
        view size, zoom or image size changed so that can be ruled in or out.

        Args: None.
        Returns: None.
        """
        packet = self.shown_packet
        if packet is None:
            return
        rect = self.view.display_rect()
        if rect is None:
            return
        height, width = packet.image.shape[:2]
        key = (self.view.width(), self.view.height(), round(self.view.zoom, 3),
               width, height)
        if key == self.view_state:
            return
        self.view_state = key
        self.log.debug(
            f"Display: view {key[0]}x{key[1]} px, zoom {key[2]:g}, image "
            f"{width}x{height}, drawn at {rect[4]:.2f}x "
            f"({type(self.view).__name__})")

    def _show_tip(self):
        """Put the current tip in the status bar, or hide it.

        Args: None.
        Returns: None.
        """
        text = self.tips.current()
        # The tips advertise controls the simple UI does not render.
        shown = not self.tips.hidden and not self.simple_ui
        self.tip_label.setText(text)
        if text:
            set_tooltip(self.tip_label, text,
                        "Click for the next tip. Right-click to hide.")
        else:
            set_tooltip(self.tip_label)
        self.tip_label.setVisible(shown)
        action = getattr(self, "tips_action", None)
        if action is not None:
            action.setChecked(shown)

    def _rotate_tip(self):
        """Show the next tip and wait the full interval again.

        Shared by the timer and a click. A timeout that arrives after the
        tips were hidden must not bring them back.

        Args: None.
        Returns: None.
        """
        if self.tips.hidden:
            return
        self.tips.advance()
        self._show_tip()
        self.tip_timer.start()

    def _set_tips_visible(self, visible):
        """Show or hide the tips and remember the choice.

        Args: visible is True to show tips.
        Returns: None.
        """
        if visible:
            self.tips.restore()
            self.tip_timer.start()
        else:
            self.tips.dismiss()
            self.tip_timer.stop()
        self._show_tip()
        QSettings("ESPI", "espi_live").setValue("tips/hidden", self.tips.hidden)

    def _choose_simulated_object(self, name):
        """Send the chosen simulated object to the displayed camera.

        Args: name is a simulation.SIMULATED_OBJECTS name.
        Returns: None.
        """
        self.engine.update_settings(demo_object=name)
        self._show_simulated_note()

    def _choose_simulated_piezo(self, on):
        """Turn the displayed demo camera's reference piezo triangle on or off.

        Args: on is the checkbox state.
        Returns: None.
        """
        self.engine.update_settings(demo_piezo=on)
        self._show_simulated_note()

    def _show_simulated_note(self):
        """Say under the control what the simulated drumhead is doing.

        Args: None.
        Returns: None.
        """
        self.simulated_note.setText(simulation.object_note(
            self.simulated_object.currentText(), self.daq_drive,
            piezo=self.simulated_piezo.isChecked()))

    def set_daq_drive(self, drive):
        """Follow what the NI DAQ outputs produce, for every demo camera.

        Each camera follows the piezo driver it is paired to (Thor 1 when
        nothing is paired).

        Args: drive is a daq.Drive (daq.Drive() when nothing is driven).
        Returns: None; every channel gets its driver's view of it (demo
            engines use it).
        """
        self.daq_drive = drive
        for channel in self.channels:
            channel.set_drive(daq.drive_for(drive, self._driver_of(channel)))
        self._show_simulated_note()

    def _poll_status(self):
        """Refresh status text, the log, and handle engine events.

        Args: None.
        Returns: None.
        """
        text = self.log.drain()
        if text:
            self.log_view.appendPlainText(text.rstrip("\n"))
        self.poll_daq()  # the drive for demo cameras and the piezo for the Phase view
        self._follow_frame_rate()
        self._sync_step_monitor()
        self._show_piezo_debug()
        now = time.monotonic()
        if now - self.fps_window_start >= 1.0:
            self.display_fps = self.drawn_frames / (now - self.fps_window_start)
            self.drawn_frames = 0
            self.fps_window_start = now
        self._log_view_state()
        status = self.engine.status()
        self.camera_label.setText(status["camera"])
        self.hint_label.setText(status["hint"])
        self.record_label.setText(status["record"])
        self.range_status.setText(status["range"])
        self.camera_fps_status.setText(status["camera_fps"])
        frozen = self.current.frozen
        displayed = "frozen" if frozen is not None else f"{self.display_fps:.1f} fps"
        self.speed_status.setText(f"{status['processing']}    Displayed: {displayed}")
        packet = frozen if frozen is not None else self.engine.latest_packet()
        x, y = self.hover
        if packet is not None and 0 <= y < packet.image.shape[0] and \
                0 <= x < packet.image.shape[1]:
            self.pixel_status.setText(f"x {x}, y {y}: {int(packet.image[y, x])}")
        else:
            self.pixel_status.setText("")
        for channel, state in zip(self.channels, self.camera_states):
            state.setText(self._camera_state(channel))
        self.log.watchdog_tick()
        self._handle_events()
        self._handle_update_events()
        self._handle_feedback_events()
        self._handle_report_events()
        self._handle_source_events()
        self._update_report_context()
        self._handle_error_report_events()

    # ----- error reports -------------------------------------------------------

    def _sync_reports_action(self):
        """Show the error-report choice and whether a switch forces it off.

        Args: None.
        Returns: None.
        """
        action = self.reports_action
        action.blockSignals(True)
        if self.reporting is None:
            action.setChecked(False)
            action.setEnabled(False)
        else:
            action.setChecked(self.reporting.reporter.enabled)
            action.setEnabled(self.reporting.forced is None)
            action.setToolTip(
                f"Off: {self.reporting.forced}" if self.reporting.forced else
                "Unexpected errors are filed as public GitHub issues, without your "
                "user name, folders or camera serial numbers")
        action.blockSignals(False)

    def _choose_error_reports(self, on):
        """Turn automatic error reports on or off from the Help menu.

        Args: on is the new state.
        Returns: None.
        """
        self.reporting.choose(on, self.log)
        self._sync_reports_action()

    def _open_log_folder(self):
        """Open the folder that holds the run logs.

        Args: None.
        Returns: None.
        """
        try:
            folder = (os.path.dirname(self.log.path) if self.log.path
                      else paths.output_folder("logs"))
        except OSError as exc:
            QMessageBox.warning(self, "Log folder", f"The log folder is unavailable: {exc}")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(folder)):
            QMessageBox.information(self, "Log folder", f"The logs are in {folder}")

    def _update_report_context(self):
        """Tell the reporter what this session shows, for later reports.

        Args: None.
        Returns: None.
        """
        if self.reporting is None:
            return
        identity = self.current.identity
        if self.current.demo:
            source = "dummy mode"
        else:
            source = "camera " + (identity.short_name() if identity
                                  else self.current.info.model or "(model unknown)")
        serials = [channel.info.serial for channel in self.channels]
        serials += [getattr(channel.identity, "serial", "") for channel in self.channels
                    if getattr(channel.identity, "serial", "")]
        self.reporting.reporter.update_context(
            source=source, cameras=len(self.channels), serials=serials,
            spinnaker=camera.spinnaker_version())

    def _handle_error_report_events(self):
        """Show what happened to error reports: in an open "ESPI stopped"
        dialog for that error, otherwise in the status bar.

        Args: None.
        Returns: None.
        """
        if self.reporting is None:
            return
        while True:
            try:
                _, outcome = self.reporting.worker.events.get_nowait()
            except queue.Empty:
                return
            self.report_outcomes[outcome.fingerprint] = outcome
            if outcome.fingerprint in self.fatal_boxes:
                box, message = self.fatal_boxes[outcome.fingerprint]
                box.setText(self._fatal_text(message, outcome.fingerprint))
            else:
                self._show_report_status(outcome)

    @staticmethod
    def _outcome_html(outcome):
        """Describe an outcome as rich text with the issue link.

        Args: outcome is an error_reports.Outcome, or None while sending.
        Returns: HTML text.
        """
        text = html.escape(error_reports.outcome_text(outcome))
        if outcome is None or not outcome.url:
            return text
        url = html.escape(outcome.url, quote=True)
        if outcome.kind == "browser":
            return f'Not sent automatically: <a href="{url}">Report in browser…</a>'
        if outcome.number:
            number = f"#{outcome.number}"
            return text.replace(number, f'<a href="{url}">{number}</a>', 1)
        return text

    def _fatal_text(self, message, fingerprint):
        """Build the "ESPI stopped" text, with the report's state when there is one.

        Args: message is the error; fingerprint is its error ID or None.
        Returns: HTML text.
        """
        text = html.escape(message)
        if fingerprint is None:
            return text
        outcome = self.report_outcomes.get(fingerprint)
        return (f"{text}<br><br>{self._outcome_html(outcome)}"
                f"<br><small>Error ID {html.escape(fingerprint)}</small>")

    def _show_fatal(self, message, fingerprint):
        """Show the "ESPI stopped" dialog; its report line updates while it is open.

        Args: message is the error; fingerprint is its error ID or None.
        Returns: None.
        """
        box = QMessageBox(QMessageBox.Critical, "ESPI stopped",
                          self._fatal_text(message, fingerprint), parent=self)
        box.setTextFormat(Qt.RichText)
        box.setTextInteractionFlags(Qt.TextBrowserInteraction)
        if fingerprint is not None:
            self.fatal_boxes[fingerprint] = (box, message)
        try:
            box.exec()
        finally:
            self.fatal_boxes.pop(fingerprint, None)

    def _show_report_status(self, outcome):
        """Put a report outcome in the status bar for a while (no dialog).

        Args: outcome is an error_reports.Outcome.
        Returns: None.
        """
        if outcome.kind == "declined":
            return  # the developers chose not to fix it; the log has the details
        if outcome.kind == "fixed":
            text = "An error occurred that a newer build fixes: use Check for updates"
        elif outcome.kind == "saved":
            text = "An unexpected error was saved; it will be reported next time"
        else:
            text = f"Unexpected error: {self._outcome_html(outcome)}"
        self.error_report_label.setText(text)
        self.error_report_label.setToolTip(f"Error ID {outcome.fingerprint}")
        self.error_report_label.setVisible(True)
        self.error_report_timer.start(REPORT_STATUS_MS)

    # ----- feature requests --------------------------------------------------

    def _request_feature(self, previous=None):
        """Ask for a feature request and file it in the background.

        Args: previous is a FeatureRequest to restore after a failed send.
        Returns: None.
        """
        if self.feedback.busy:
            self.statusBar().showMessage("A feature request is still being sent…", 3000)
            return
        dialog = FeatureRequestDialog(self, previous)
        if dialog.exec() != QDialog.Accepted:
            return
        request = dialog.request("dummy mode" if self.engine.demo else "camera")
        self.feedback.submit(request)
        self.feature_button.setEnabled(False)
        _set_icon_action(self.feature_button, FEATURE_BUSY_LABEL, FEATURE_TIP)
        self.statusBar().showMessage("Sending the feature request…", 5000)
        self.log.write(f"Sending feature request: {request.title.strip()}")

    def _handle_feedback_events(self):
        """Report feature requests that were sent or failed.

        Args: None.
        Returns: None.
        """
        while True:
            try:
                kind, (request, payload) = self.feedback.events.get_nowait()
            except queue.Empty:
                return
            self.feature_button.setEnabled(True)
            _set_icon_action(self.feature_button, FEATURE_LABEL, FEATURE_TIP)
            if kind == "sent":
                self.log.write(f"Feature request filed: {payload}")
                box = QMessageBox(QMessageBox.Information, "Request submitted",
                                  "Your request was submitted.", parent=self)
                box.exec()
            else:
                self.log.warning(f"Feature request failed: {payload}")
                box = QMessageBox(QMessageBox.Warning, "Request not sent",
                                  f"The request could not be sent: {payload}.", parent=self)
                retry = box.addButton("Edit and retry", QMessageBox.AcceptRole)
                box.addButton(QMessageBox.Cancel)
                box.exec()
                if box.clickedButton() is retry:
                    self._request_feature(request)

    # ----- source code and Codex ---------------------------------------------

    def _open_codex(self):
        """Open the Codex app on this build's source code.

        Downloads the source first when it is not there yet (the startup
        download failed or is still running), then opens Codex once it is.

        Args: None.
        Returns: None.
        """
        if self.source_folder is None:
            self.codex_when_ready = True
            self.codex_button.setEnabled(False)
            _set_icon_action(self.codex_button, CODEX_BUSY_LABEL, CODEX_TIP)
            if self.source.start():
                self.statusBar().showMessage("Getting the source code for Codex…", 5000)
            else:
                self.statusBar().showMessage("Still downloading the source code…", 3000)
            return
        settings = QSettings("ESPI", "espi_live")
        if not settings.value("codex/intro_shown", False, bool):
            if not self._codex_intro():
                return
            settings.setValue("codex/intro_shown", True)
        try:
            codex.write_guide(self.source_folder)
        except OSError as exc:
            QMessageBox.warning(self, "Open in Codex",
                                "Could not prepare the source code for Codex in "
                                f"{self.source_folder}: {exc.strerror or exc}.")
            return
        try:
            started = codex.open_cli(self.source_folder)
        except OSError as exc:
            QMessageBox.warning(self, "Open in Codex",
                                "Could not start Codex in "
                                f"{self.source_folder}: {exc.strerror or exc}.")
            return
        if started:
            self.log.write(f"Opened Codex on {self.source_folder}")
            self.statusBar().showMessage("Opening Codex with ESPI live's source code…", 5000)
            return
        box = QMessageBox(QMessageBox.Warning, "Codex not found",
                          "Codex is not installed on this computer (the program looks for "
                          f"the \"{codex.CLI_NAME}\" command). Install the Codex command-line "
                          "tool, open a new terminal and run \"codex\" once to sign in with "
                          "your ChatGPT account, then press Open in Codex again.", parent=self)
        get = box.addButton("How to install Codex…", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Close)
        box.exec()
        if box.clickedButton() is get:
            QDesktopServices.openUrl(QUrl(codex.INSTALL_URL))

    def _codex_intro(self):
        """Explain Open in Codex the first time it is used.

        Args: None.
        Returns: True to go on and open Codex, False to stop.
        """
        box = QMessageBox(QMessageBox.Information, "Open in Codex",
                          "Codex, OpenAI's coding assistant, opens in a terminal window with "
                          "the source code of this copy of ESPI live, so you can ask it how "
                          "the program works."
                          "<br><br>If you ask it to change something, it shows you the "
                          "change and asks whether you are happy with it. If you are, it "
                          "sends the change to the ESPI live developers to review. Your "
                          "installed program stays as it is.<br><br>You need the Codex "
                          "command-line tool (the \"codex\" command) and a ChatGPT account."
                          f"<br><br>Source code: {html.escape(self.source_folder)}",
                          parent=self)
        box.setTextFormat(Qt.RichText)
        go = box.addButton("Open Codex", QMessageBox.AcceptRole)
        get = box.addButton("How to install Codex…", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        if box.clickedButton() is get:
            QDesktopServices.openUrl(QUrl(codex.INSTALL_URL))
        return box.clickedButton() is go

    def _handle_source_events(self):
        """Note when the source copy is ready, or why it is not.

        Args: None.
        Returns: None.
        """
        while True:
            try:
                kind, payload = self.source.events.get_nowait()
            except Exception:
                return
            wanted, self.codex_when_ready = self.codex_when_ready, False
            self.codex_button.setEnabled(True)
            _set_icon_action(self.codex_button, CODEX_LABEL, CODEX_TIP)
            if kind == "ready":
                self.source_folder, note = payload
                if note:
                    self.log.write(note)
                if wanted:
                    self._open_codex()
            else:
                self.log.write(f"Source code not available: {payload}")
                if wanted:
                    QMessageBox.warning(self, "Open in Codex",
                                        "The source code could not be downloaded: "
                                        f"{html.escape(payload)}."
                                        "<br><br>Check the internet connection and try again.")

    # ----- updates -----------------------------------------------------------

    def _show_whats_new(self):
        """Show the release notes of the update this start follows.

        Args: None.
        Returns: None.
        """
        self.log.write(f"Updated from build {self.whats_new.from_build} to "
                       f"{self.whats_new.to_build}")
        WhatsNewDialog(self, self.whats_new).exec()

    def _schedule_update_check(self):
        """Arm the next update check a few hours from now, with a little jitter.

        Args: None.
        Returns: None.
        """
        if self.closing or not self._periodic_checks:
            return
        spread = random.randint(-UPDATE_CHECK_JITTER_MS, UPDATE_CHECK_JITTER_MS)
        self.update_timer.start(UPDATE_CHECK_INTERVAL_MS + spread)

    def _capture_busy(self):
        """Report whether a recording or a frame save should wait out this check.

        Args: None.
        Returns: True while any camera is recording, finishing a recording,
            or still saving a frame.
        """
        for engine in self._all_engines():
            if engine.settings().record:
                return True
            if engine.status().get("record") != "Recording: off":
                return True
            if engine.save_pending():
                return True
        return False

    def _periodic_update_check(self):
        """Check for a newer build, unless one is already running or capture is busy.

        A skipped tick still schedules the next one. The check is not manual,
        so a failure is only written to the log. Nothing is scheduled once
        the window is closing.

        Args: None.
        Returns: None.
        """
        if self.closing or not self._periodic_checks:
            return
        if not self.updates.busy and self.pending_update is None and not self._capture_busy():
            self.updates.check()
        self._schedule_update_check()

    def _show_available_update(self, info):
        """Point the status-bar button at a newer build.

        The button reads "Install update"; the version numbers stay in the
        tooltip ("Build 42 → 45"). The first time a build number is seen, a
        brief status message is shown. A later check that finds the same
        build leaves the button as it is and does not repeat the message.

        Args: info is the UpdateInfo.
        Returns: None.
        """
        self.update_info = info
        self.update_button.setText("Install update")
        set_tooltip(self.update_button,
                    f"Build {updater.current_build()} → {info.build}.",
                    "Download the newer build of ESPI live.")
        self.update_button.setVisible(True)
        self.update_button.setEnabled(True)
        if self.announced_build != info.build:
            self.announced_build = info.build
            self.statusBar().showMessage(f"Update available: {info.label()}", 10000)

    def _check_updates_now(self):
        """Check for a newer build on request and report the result in a dialog.

        Args: None.
        Returns: None.
        """
        reason = updater.unavailable_reason()
        if reason:
            QMessageBox.information(self, "Updates", f"Automatic updates are off: {reason}.")
        elif self.pending_update:
            self._offer_restart()
        elif self.updates.busy:
            self.statusBar().showMessage("An update check or download is already running…", 3000)
        else:
            self.update_manual = True
            self.updates.check()
            self.statusBar().showMessage("Checking GitHub for a newer build…", 3000)

    def _update_clicked(self):
        """Download the available build, or offer to restart once it is ready.

        Args: None.
        Returns: None.
        """
        if self.pending_update:
            self._offer_restart()
            return
        if self.update_info is None or self.updates.busy:
            return
        answer = QMessageBox.question(
            self, "Update available",
            f"ESPI live {self.update_info.label()} is available "
            f"(this is build {updater.current_build()}).\n\n"
            f"Download it now ({self.update_info.size / 1e6:.0f} MB)? "
            "Measurement and recording continue while it downloads.")
        if answer == QMessageBox.Yes and self.updates.download(self.update_info):
            self.update_button.setEnabled(False)
            self.log.write(f"Downloading update {self.update_info.label()}")

    def _offer_restart(self):
        """Ask to restart into the downloaded build now or on exit.

        Args: None.
        Returns: None.
        """
        answer = QMessageBox.question(
            self, "Update ready",
            "The update is downloaded. Restart ESPI live now to install it?\n\n"
            "A recording in progress is finished and saved first, and the camera "
            "is released before the new version starts. If you choose No, the "
            "update installs when you close ESPI live.")
        if answer == QMessageBox.Yes:
            self.relaunch_after_exit = True
            self.close()

    def _handle_update_events(self):
        """Show update check and download results.

        Args: None.
        Returns: None.
        """
        if (self.updates.busy and self.updates.downloading and self.update_info is not None
                and not self.pending_update):
            done, total = self.updates.progress
            percent = f" {100 * done / total:.0f}%" if total else ""
            self.update_button.setText(f"Downloading update…{percent}")
        while True:
            try:
                kind, payload = self.updates.events.get_nowait()
            except queue.Empty:
                break
            manual, self.update_manual = self.update_manual, False
            if kind == "checked" and payload is None:
                self.log.write(f"Up to date ({updater.describe_build()})")
                if manual:
                    QMessageBox.information(self, "Updates", "ESPI live is up to date.")
            elif kind == "checked":
                self.log.write(f"Update available: {payload.label()}")
                self._show_available_update(payload)
                if manual:
                    self._update_clicked()
            elif kind == "check_failed":
                self.log.warning(f"Update check failed: {payload}")
                if manual:
                    QMessageBox.warning(self, "Updates", f"Update check failed: {payload}")
            elif kind == "downloaded":
                self.pending_update = payload
                self.log.write("Update downloaded; it installs when ESPI live restarts")
                self.update_button.setText("Restart to update")
                set_tooltip(self.update_button,
                            "Restart now to install the downloaded build, or close ESPI live "
                            "and the update installs on exit.")
                self.update_button.setEnabled(True)
                self._offer_restart()
            elif kind == "download_failed":
                self.log.warning(f"Update download failed: {payload}")
                self._show_available_update(self.update_info)
                QMessageBox.warning(self, "Update failed", f"Download failed: {payload}")
        self._sync_check_button()

    def _sync_check_button(self):
        """Show on the Check for updates button what the update worker is doing.

        Called every status poll, after update events are handled.

        Args: None.
        Returns: None.
        """
        found = self.update_info is not None or bool(self.pending_update)
        checking = self.updates.busy and not found  # downloads use the same worker
        self.check_button.setVisible(not found)  # the blue update button takes over
        if self.check_button.isEnabled() == checking:  # only on a change: this runs every poll
            self.check_button.setEnabled(not checking)
            _set_icon_action(self.check_button,
                             CHECK_BUSY_LABEL if checking else CHECK_LABEL, CHECK_TIP)

    def _handle_events(self):
        """React to every camera's engine events.

        Args: None.
        Returns: None.
        """
        for channel in list(self.channels):
            engine = channel.engine
            while engine is not None and channel.engine is engine:
                try:
                    kind, payload = engine.events.get_nowait()
                except queue.Empty:
                    break
                self._handle_event(channel, kind, payload)

    def _handle_event(self, channel, kind, payload):
        """React to one engine event (model, limits, record stop, live switch, errors).

        Args: channel is the Channel whose engine sent it; kind and payload
            are the event.
        Returns: None.
        """
        displayed = channel is self.current
        name = f"{channel.label}: " if len(self.channels) > 1 else ""
        if kind == "camera_identity":
            channel.identity = payload
            if displayed:
                self._set_demo(channel.demo)
        elif kind == "camera_report":
            channel.report = payload
            self.reports.submit(payload)
            if displayed:
                self._show_report()
        elif kind == "camera_limits":
            channel.limits = payload
            if displayed:
                self._apply_limits(payload)
        elif kind == "camera_settings":
            self._phone_settings_changed(channel, payload)
        elif kind == "noise_calibrated" and displayed:
            self.min_range.set_value(payload)
        elif kind == "phase_calibrated":
            degrees, volts, index = payload
            self._save_piezo_step(volts, index)
            stepper = self.phase_steppers.get(index)
            if stepper is not None:
                self.phase_steppers[index] = stepper.with_step(volts)
                self._attach_steppers()
            which = "" if index == 1 else " (Thor 2)"
            self.statusBar().showMessage(
                f"Phase step calibrated{which}: {degrees:.0f}° measured; now "
                f"{volts:.4g} V per step.", PHASE_MESSAGE_MS)
        elif kind == "phase_calibration_failed":
            degrees, index = payload
            which = "" if index == 1 else " (Thor 2)"
            self.statusBar().showMessage(
                f"Phase step calibration failed{which}: {degrees:.0f}° per step "
                "measured (see the log).", PHASE_MESSAGE_MS)
        elif kind == "phase_stopped":
            index = self.stepper_claims.get(channel)
            if index is not None:
                self.phase_steppers[index] = None
                self._attach_steppers()
            self.statusBar().showMessage(f"{name}Phase view stopped: {payload}",
                                         PHASE_MESSAGE_MS)
        elif kind == "record_off" and displayed:
            self.record_button.blockSignals(True)
            self.record_button.setChecked(False)
            self.record_button.blockSignals(False)
        elif kind == "switched_live":
            # A held demo frame must not outlive the dummy-mode indicators.
            channel.frozen = channel.frozen_at = None
            self._show_frozen()
            self._adopt_live_cameras(channel, payload or [])
            if displayed:
                self._load_settings(channel.engine.settings(), channel.limits)
            self._set_demo(False)
            self._attach_stepper(self.phase_stepper)
        elif kind == "switch_failed":
            QMessageBox.warning(self, "Camera unavailable",
                                f"Could not start live capture: {payload}\n\n"
                                "Dummy mode will continue.")
        elif kind == "open_failed":
            QMessageBox.warning(self, "Camera unavailable",
                                f"Could not start {channel.label}: {payload}")
            self._channel_ended(channel)
        elif kind == "fatal":
            message, fingerprint = payload
            self._show_fatal(f"{name}{message}", fingerprint)
            self._channel_ended(channel)
        elif kind == "stopped" and not self.closing:
            self._channel_ended(channel)

    def _channel_ended(self, channel):
        """Handle a camera whose engine stopped on its own.

        With other cameras still running, the window stays open and shows
        one of them; when it was the last one, the window closes.

        Args: channel is the Channel whose engine stopped or failed.
        Returns: None.
        """
        others = [other for other in self.channels
                  if other.active and other is not channel]
        if not others:
            self.close()
            return
        channel.stop()
        if channel is self.current:
            self.show_channel(others[0])
        else:
            self._sync_cameras()
            self._stream_follow()

    def closeEvent(self, event):
        """Stop the engine (finishing recordings) and save the window layout.

        Args: event is the Qt close event.
        Returns: None.
        """
        if not self.closing:
            self.closing = True
            self.tip_timer.stop()
            self.update_timer.stop()
            if self.stream_window is not None:
                self.stream_window.close()
            if self.daq is not None:  # amplifiers must not be left driven
                self.statusBar().showMessage("Setting the NI DAQ outputs to 0 V…")
                if not self.daq.shutdown(DAQ_SHUTDOWN_S):
                    self.log.warning("NI DAQ: the device did not confirm 0 V on its "
                                     f"outputs within {DAQ_SHUTDOWN_S:g} s.")
            self.statusBar().showMessage("Finishing recordings and releasing the camera…")
            for engine in self._all_engines():
                engine.stop()
            deadline = time.monotonic() + 8.0
            while not self.wait_for_engines(0.05) and time.monotonic() < deadline:
                QApplication.processEvents()
            self._poll_status()
            self.profile_timer.stop()
            self._save_profiles(settle=False)
            self.profiles.sync()
            settings = QSettings("ESPI", "espi_live")
            settings.setValue("geometry", self.saveGeometry())
            settings.setValue("windowState", self.saveState(LAYOUT_VERSION))
        app = QApplication.instance()
        app.removeEventFilter(self.shortcut_filter)
        app.removeEventFilter(self.tooltip_filter)
        event.accept()


def finish_update(window, log):
    """Install a downloaded build after the window closed, and restart into it.

    Waits for every engine to stop before changing files, so the camera is
    free for the new copy. A shutdown timeout leaves the installation alone.

    Args: window is the closed MainWindow (or None); log is the AppLog.
    Returns: None.
    """
    if window is None or not window.pending_update:
        return
    if not window.wait_for_engines(5.0):
        log.warning("The camera was not released in time; the update was not installed. "
                    "Close any running copies and try the update again.")
        return
    try:
        path = updater.install_update(window.pending_update)
        log.write(f"Installed update: {path}")
        if window.relaunch_after_exit:
            updater.relaunch(sys.argv[1:])
            log.write("Restarting into the new build")
    except (updater.UpdateError, OSError) as exc:
        log.warning(f"Update could not be installed: {exc}")


def start_control(window, log, port=None):
    """Start the localhost control server for a window.

    Args: window is the MainWindow to control; log is the AppLog; port is
        a fixed port (None: pick the first free port from the default
        range).
    Returns: The ControlServer, or None when no port was free.
    """
    service = control.ControlService(ControlBackend(window, MainThreadCaller(window)))
    server = control.ControlServer(service, port=port, log=log)
    if not server.start():
        log.write("Remote control unavailable: every port "
                  f"{control.DEFAULT_PORT}–{control.DEFAULT_PORT + control.PORT_SCAN - 1}"
                  " is busy")
        return None
    log.write(f"Remote control: http://127.0.0.1:{server.port}/ "
              "(GET / describes the API; `espi_live control` is the client)")
    return server


def run(force_demo=False, check_updates=True, demo_cameras=1, log_level=None,
        send_reports=True, control_port=None, no_control=False):
    """Start the Qt front end.

    Args: force_demo skips camera detection and starts synthetic video;
        check_updates looks for a newer prebuilt binary (at startup and
        every few hours) and downloads this build's source code (if missing)
        at startup;
        demo_cameras is how many synthetic cameras demo mode simulates;
        log_level is the --log-level name or None; send_reports is False
        for --no-error-reports; control_port is the remote-control server's
        port (None: pick a free port from the default range); no_control
        skips the server.
    Returns: The process exit code.
    """
    request_gl_format()
    QApplication.setApplicationName("ESPI live")
    QApplication.setOrganizationName("ESPI")
    app = QApplication.instance() or QApplication(sys.argv)
    log = AppLog(log_level)
    path = log.start()
    for line in header_lines("Qt", log.level):
        log.info(line)
    reporting = Reporting(log, send_reports)
    cleanup = updater.StartupCleanup(log.write)
    window = None
    channels = []
    server = None
    try:
        if path:
            log.write(f"Log file: {path}")
        reporting.start(log)
        cleanup.start()
        # Each camera's saved profile is restored before its camera opens.
        store = profiles.ProfileStore(QSettings("ESPI", "espi_live"), log)
        if not force_demo:
            try:
                channels = open_camera_channels(log, store)
            except Exception as exc:
                error_reports.caught(log, exc, "camera_startup", "Camera startup unavailable")
                answer = QMessageBox.question(
                    None, "No camera detected",
                    "No FLIR camera or recognised capture adapter is available. "
                    "Start dummy/demo mode with a simulated ESPI signal (a "
                    "simulated drumhead on an ideal sensor)?")
                if answer != QMessageBox.Yes:
                    log.write("No camera detected; demo mode declined.")
                    return 0
        if not channels:
            channels = demo_channels(log, demo_cameras, store)
        use_gl, description = opengl_available()
        display = f"GPU ({description})" if use_gl else f"CPU fallback ({description})"
        log.write(f"Display: {display}")
        log.info(describe_session(channels, display))
        reporting.reporter.update_context(display=display)
        window = MainWindow(channels, log, use_gl, check_updates, reporting, store)
        window.show()
        if not no_control:
            server = start_control(window, log, control_port)
        for channel in channels:
            if channel.active:
                channel.engine.start()
        # After the cameras opened, so their serials are scrubbed.
        reporting.start_uploads(lambda: window.channels)
        return app.exec()
    except Exception as exc:  # the hooks are restored below, before Python's own hook runs
        error_reports.caught(log, exc, "main_thread", "ESPI live stopped", fatal=True)
        raise
    finally:
        if server is not None:
            server.stop()
        if window is None:
            for channel in channels:
                if channel.active:
                    channel.engine.stop()
                    channel.engine.wait(8.0)
        cleanup.stop()
        finish_update(window, log)
        reporting.stop()
        log.close()
