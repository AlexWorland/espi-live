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
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime

import cv2
import numpy as np
from PySide6.QtCore import (QCoreApplication, QEvent, QObject, QPoint, QPointF, QRect,
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

from espi_live import (camera, camera_reports, codex, daq, error_reports, feedback, iphone,
                       log_uploads, paths, profiles, source_code, tips, ui_scale, updater)
from espi_live.channels import demo_channels, open_camera_channels, restore_channel
from espi_live.engine import LABVIEW_FLOW
from espi_live.guide import guide_path
from espi_live.logs import AppLog, header_lines
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
        self.reporter.collect_native_crashes(log.crash_path)
        self.worker.start()

    def start_uploads(self):
        """Start the log uploads, once the cameras are open and their serials known.

        Args: None.
        Returns: None.
        """
        self.uploads.start()

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
    "(exposure 8000 µs, gain 0 dB, 30 fps on a FLIR camera; the LabVIEW flow "
    "on an analog one) and clear its saved settings, after you confirm. Other "
    "cameras and saved presets do not change."
)
MESSAGE_TIP_WIDTH = 120
# Previous and Reference views run in this order. Raw skips difference processing but still applies display gamma.
PIPELINE = ("noise reduction → (frame average) → difference (N frames back) → "
            "noise threshold → beam brightness → outlier filter → average → "
            "speckle filter → noise floor → peak hold → fixed gain, stretch or "
            "local normalization → invert → brightness, gain and contrast → gamma")
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
    "exposure, gain, frame rate and recording stay as they are. On by default "
    "for an analog camera. Turning it off puts back your own settings; changes "
    "made while it is on are dropped."
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

Analog camera (Sony XC-ST50): it starts in the LabVIEW flow, the LabVIEW ESPI 1.5 VI's processing with its saved defaults: Reference view with an automatic reference, Fixed gain 42× (the VI's Constant) on each frame, no averaging or filters. Turn LabVIEW flow off for your own settings, which also use Fixed gain: Auto stretch would blow the camera's few grey levels of noise up to full-contrast grain. Lower Difference gain when noise shows as grey grain. Gain before averaging (LabVIEW order) and Average multiplier reproduce the VI's averaging maths, including its N + 1 frames divided by N.

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
        """Finish panning.

        Args: event is the Qt mouse event.
        Returns: None.
        """
        self.drag_origin = None
        self.unsetCursor()

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


class CpuImageView(ViewNavigation, QWidget):
    """Fallback view: resize and colorize on the CPU, draw with QPainter."""

    hovered = Signal(int, int)
    zoomChanged = Signal(float)  # zoom relative to fit-to-window

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


DAQ_POLL_MS = 100
DAQ_SHUTDOWN_S = 5.0  # longest wait for the DAQ to reach 0 V at exit
DAQ_SETTINGS_KEY = "daq/settings"
DAQ_TITLE = "Frequency generator and piezo (NI DAQ)"
GENERATOR_KINDS = {"ao": "sine", "ctr": "square, 5 V TTL"}
PIEZO_MODE_TEXT = ((daq.HOLD, "Hold a voltage"),
                   (daq.SINE, "Sine at the generator's frequency"))


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
        self.resize(ui_scale.px(620), ui_scale.px(640))
        saved = self._saved_settings()

        self.device_box = QComboBox()
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

        self.generator_output = QComboBox()
        set_tooltip(self.generator_output,
                    "Which output produces the frequency generator's signal. An analog "
                    "output gives a sine for the shaker or PZT amplifier; a counter gives a "
                    "5 V square wave on a PFI terminal.")
        self.generator_terminal = QComboBox()
        set_tooltip(self.generator_terminal,
                    "The PFI terminal a counter's square wave comes out on. On the "
                    "USB-6251 BNC, PFI 0–7 have BNC sockets.")
        self.piezo_output = QComboBox()
        set_tooltip(self.piezo_output,
                    "Which analog output drives the reference piezo's amplifier.")
        generator_row = QHBoxLayout()
        generator_row.addWidget(self.generator_output, 1)
        generator_row.addWidget(self.generator_terminal)
        assign = QGroupBox("Assign outputs")
        assign_form = QFormLayout(assign)
        assign_form.addRow("Frequency generator", generator_row)
        assign_form.addRow("Piezo driver", self.piezo_output)

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

        self.piezo_on = QCheckBox("Piezo on")
        set_tooltip(self.piezo_on, "Drive the piezo after Apply. Off holds it at 0 V.")
        self.piezo_mode = QComboBox()
        for mode, text in PIEZO_MODE_TEXT:
            self.piezo_mode.addItem(text, mode)
        set_tooltip(self.piezo_mode,
                    "Hold a voltage (step it by hand for phase steps), or follow the "
                    "generator with a sine at the same frequency and a set phase.")
        self.piezo_voltage = daq_spin(-daq.MAX_VOLTS, daq.MAX_VOLTS, 4, " V", 0.01)
        set_tooltip(self.piezo_voltage,
                    "The held voltage, or the centre of the sine, at the DAQ.")
        self.piezo_amplitude = daq_spin(0.0, daq.MAX_VOLTS, 4, " V", 0.01)
        set_tooltip(self.piezo_amplitude, "Sine amplitude (peak). Sine mode only.")
        self.piezo_phase = daq_spin(-360.0, 360.0, 1, "°", 5.0)
        set_tooltip(self.piezo_phase,
                    "Phase of the piezo sine relative to the generator. Sine mode only.")
        self.piezo_step = daq_spin(0.0001, daq.MAX_VOLTS, 4, " V", 0.01)
        set_tooltip(self.piezo_step,
                    "How far one Step press moves the held voltage (for example the "
                    "voltage of a quarter-wave phase step).")
        self.step_down = QPushButton("Step −")
        set_tooltip(self.step_down, "Lower the held voltage by one step and apply it.")
        self.step_up = QPushButton("Step +")
        set_tooltip(self.step_up, "Raise the held voltage by one step and apply it.")
        step_row = QHBoxLayout()
        step_row.addWidget(self.piezo_step, 1)
        step_row.addWidget(self.step_down)
        step_row.addWidget(self.step_up)
        self.piezo_limit = daq_spin(0.01, daq.MAX_VOLTS, 2, " V", 0.1)
        set_tooltip(self.piezo_limit,
                    "Safety limit: the piezo driver never exceeds this many volts either "
                    "way. Set it to the piezo amplifier's largest safe input.")
        piezo = QGroupBox("Piezo driver")
        piezo_form = QFormLayout(piezo)
        piezo_form.addRow(self.piezo_on)
        piezo_form.addRow("Mode", self.piezo_mode)
        piezo_form.addRow("Voltage", self.piezo_voltage)
        piezo_form.addRow("Amplitude", self.piezo_amplitude)
        piezo_form.addRow("Phase", self.piezo_phase)
        piezo_form.addRow("Step", step_row)
        piezo_form.addRow("Safety limit", self.piezo_limit)

        self.problem_label = QLabel()
        self.problem_label.setWordWrap(True)
        self.problem_label.setStyleSheet("color:#c00000;")
        self.state_label = QLabel("Outputs: not connected")
        self.state_label.setWordWrap(True)
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
        close = buttons.addButton(QDialogButtonBox.Close)
        set_tooltip(close, "Hide this window. The outputs keep running.")
        for button in buttons.buttons():
            button.setAutoDefault(False)  # Enter in a number box must not apply

        layout = QVBoxLayout(self)
        layout.addLayout(device_row)
        layout.addWidget(self.device_status)
        layout.addWidget(assign)
        layout.addWidget(generator)
        layout.addWidget(piezo)
        layout.addWidget(self.problem_label)
        layout.addWidget(self.state_label)
        layout.addStretch(1)
        layout.addWidget(note)
        layout.addWidget(buttons)

        self.rescan_button.clicked.connect(self.scan)
        self.device_box.activated.connect(self._connect_selected)
        self.apply_button.clicked.connect(self.apply)
        self.zero_button.clicked.connect(lambda: self.controller.request("zero"))
        self.step_down.clicked.connect(lambda: self.step(-1))
        self.step_up.clicked.connect(lambda: self.step(1))
        close.clicked.connect(self.reject)
        self._fill_outputs(daq.DeviceOutputs(daq.SIMULATED_DEVICE))
        self.load(saved)
        for box in (self.generator_output, self.generator_terminal, self.piezo_output,
                    self.piezo_mode):
            box.currentIndexChanged.connect(self.refresh)
        for spin in (self.frequency, self.amplitude, self.offset, self.generator_limit,
                     self.piezo_voltage, self.piezo_amplitude, self.piezo_phase,
                     self.piezo_step, self.piezo_limit):
            spin.valueChanged.connect(self.refresh)
        for check in (self.generator_on, self.piezo_on):
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

    def showEvent(self, event):
        """Follow the controller while the window is open; scan the first time.

        Args: event is the Qt show event.
        Returns: None.
        """
        super().showEvent(event)
        self.timer.start(DAQ_POLL_MS)
        if not self.scanned:
            self.scan()

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
                   (self.piezo_output, (daq.OFF,) + outputs.analog,
                    lambda output: "Off (not assigned)" if output == daq.OFF
                    else daq.output_label(output)),
                   (self.generator_terminal, outputs.terminals, daq.terminal_label))
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
        self._set_combo(self.piezo_output, s.piezo_output)
        self._set_combo(self.piezo_mode, s.piezo_mode)
        self.generator_on.setChecked(s.generator_on)
        self.frequency.setValue(s.generator_frequency_hz)
        self.amplitude.setValue(s.generator_amplitude_v)
        self.offset.setValue(s.generator_offset_v)
        self.generator_limit.setValue(s.generator_limit_v)
        self.piezo_on.setChecked(s.piezo_on)
        self.piezo_voltage.setValue(s.piezo_voltage_v)
        self.piezo_amplitude.setValue(s.piezo_amplitude_v)
        self.piezo_phase.setValue(s.piezo_phase_deg)
        self.piezo_step.setValue(s.piezo_step_v)
        self.piezo_limit.setValue(s.piezo_limit_v)
        self.wanted_device = s.device

    def settings(self):
        """Read the form.

        Args: None.
        Returns: daq.DaqSettings.
        """
        return daq.DaqSettings(
            device=self.device_box.currentData() or "",
            generator_output=self.generator_output.currentData() or daq.OFF,
            generator_terminal=self.generator_terminal.currentData() or "",
            generator_on=self.generator_on.isChecked(),
            generator_frequency_hz=self.frequency.value(),
            generator_amplitude_v=self.amplitude.value(),
            generator_offset_v=self.offset.value(),
            generator_limit_v=self.generator_limit.value(),
            piezo_output=self.piezo_output.currentData() or daq.OFF,
            piezo_on=self.piezo_on.isChecked(),
            piezo_mode=self.piezo_mode.currentData(),
            piezo_voltage_v=self.piezo_voltage.value(),
            piezo_amplitude_v=self.piezo_amplitude.value(),
            piezo_phase_deg=self.piezo_phase.value(),
            piezo_step_v=self.piezo_step.value(),
            piezo_limit_v=self.piezo_limit.value())

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
        has_piezo = s.piezo_output != daq.OFF
        sine_piezo = has_piezo and s.piezo_mode == daq.SINE
        for widget in (self.piezo_on, self.piezo_mode, self.piezo_voltage,
                       self.piezo_limit):
            widget.setEnabled(has_piezo)
        for widget in (self.piezo_amplitude, self.piezo_phase):
            widget.setEnabled(sine_piezo)
        for widget in (self.piezo_step, self.step_down, self.step_up):
            widget.setEnabled(has_piezo and s.piezo_mode == daq.HOLD)
        found = daq.problems(s, outputs)
        self.problem_label.setText("\n".join(found))
        self.apply_button.setEnabled(self.outputs is not None and not found)
        self.generator_result.setText(self._preview(s, outputs, found))

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
        channels = len({settings.generator_output, settings.piezo_output} - {daq.OFF})
        rate, samples, actual = daq.choose_timing(settings.generator_frequency_hz, channels)
        return (f"Produces {actual:.6g} Hz: {samples} samples per cycle at "
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
        settings = self.settings()
        QSettings("ESPI", "espi_live").setValue(DAQ_SETTINGS_KEY,
                                                json.dumps(settings.to_dict()))
        self.state_label.setText("Outputs: applying…")
        self.controller.request("apply", settings=settings)

    def step(self, direction):
        """Move the held piezo voltage by one step, and apply it if the piezo is on.

        Args: direction is +1 or -1.
        Returns: None.
        """
        self.piezo_voltage.setValue(self.piezo_voltage.value()
                                    + direction * self.piezo_step.value())
        if self.piezo_on.isChecked() and self.apply_button.isEnabled():
            self.apply()

    def connected(self, outputs):
        """Show the connected device (or none).

        Args: outputs is the device's daq.DeviceOutputs, or None.
        Returns: None.
        """
        self.outputs = outputs
        if outputs is not None:
            self._fill_outputs(outputs)
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

    def poll(self):
        """Handle what the controller has reported.

        Args: None.
        Returns: None.
        """
        while True:
            try:
                kind, payload = self.controller.events.get_nowait()
            except queue.Empty:
                return
            if kind == "devices":
                self._show_devices(*payload)
            elif kind == "connected":
                self.connected(payload)
            elif kind == "applied":
                prefix = ("Simulated, nothing driven: "
                          if self.outputs is not None and self.outputs.device.simulated
                          else "Outputs: ")
                self.state_label.setText(prefix + payload.summary)
            elif kind == "rejected":
                self.problem_label.setText("\n".join(payload))
                self.state_label.setText("Outputs: not changed.")
            elif kind == "zeroed":
                self._switch_off()
                self.state_label.setText("Outputs: all at 0 V.")
            elif kind == "failed":
                self._switch_off()
                self.problem_label.setText(
                    f"The DAQ reported an error: {payload}. The outputs were set to "
                    "0 V where the device allowed it.")
                self.state_label.setText("Outputs: stopped after an error.")

    def _switch_off(self):
        """Untick both On boxes, to match outputs at 0 V.

        Args: None.
        Returns: None.
        """
        self.generator_on.setChecked(False)
        self.piezo_on.setChecked(False)

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
        self.daq = None  # daq.DaqController, made when the DAQ window first opens
        self.daq_dialog = None  # DaqDialog, built on first use
        self.view_state = None  # last logged (view size, zoom, image size)
        self.drawn_frames = 0
        self.fps_window_start = time.monotonic()
        self.display_fps = 0.0
        self.hover = (-1, -1)
        self.view = GLImageView() if use_gl else CpuImageView()
        self.view.hovered.connect(self._set_hover)
        # The frozen banner floats over the top of the image (same grid cell);
        # mouse events pass through it so zoom, pan and the readout still work.
        self.frozen_banner = QLabel()
        self.frozen_banner.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._styled(self.frozen_banner,
                     FROZEN_STYLE + "padding:4px 12px;margin-top:8px;border-radius:4px;")
        self.frozen_banner.setVisible(False)
        center = QWidget()
        stack = QGridLayout(center)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.addWidget(self.view, 0, 0)
        stack.addWidget(self.frozen_banner, 0, 0, Qt.AlignTop | Qt.AlignHCenter)
        self.setCentralWidget(center)
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
            bar.addWidget(button)
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
        self.camera_combo = QComboBox()
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
            "rate. Start at 8 for a measurement; use 1 while aligning.")
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
            "the reference is as quiet. Start off; try it with Average N 4–8 on "
            "a noisy or analog camera.")
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
            "the object still. Start at 1.")
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
        self.smoothing_combo = QComboBox()
        self.smoothing_combo.addItems(SMOOTHING_METHODS)
        self.smoothing_combo.setCurrentText(s.smoothing_method)
        set_tooltip(self.smoothing_combo,
            "Blurs each difference image so pixel-scale speckle fades and broader "
            "fringes stand out. Gaussian softens edges and accepts any width. "
            "Median keeps fringe edges sharper and is capped at 5 px so the frame "
            "rate holds. Use Median when the fringe edges matter; Gaussian for a "
            "softer look. Start with Median.")
        self.smoothing_combo.currentTextChanged.connect(
            lambda name: self.engine.update_settings(smoothing_method=name))
        self.smoothing = SliderSpin(0, 31, 1, 0, s.smoothing, " px")
        self.smoothing.set_description(
            "Width of the speckle filter, in pixels. Keep it well below the "
            "fringe spacing or the fringes blur away with the speckle. 0 turns "
            "the filter off. Start at 3 px and raise it only while the fringes "
            "stay sharp.")
        self.smoothing.valueChanged.connect(
            lambda v: self.engine.update_settings(smoothing=int(v)))
        add(self.smoothing_combo, "Speckle filter")
        add(self.smoothing, "Filter width")
        self.scaling = QComboBox()
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
        self.preset_combo = QComboBox()
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
        self.phone_fps = QComboBox()
        set_tooltip(self.phone_fps,
                    "Frame rate the iPhone captures at. A USB 2 iPhone carries about "
                    "30 MB/s: 1920 × 1080 at about 15 fps, or a smaller crop at 30 fps.")
        self.phone_fps.activated.connect(self._phone_fps_chosen)
        add(self.phone_fps, "Frame rate", basic=True)
        self.phone_size = QComboBox()
        set_tooltip(self.phone_size,
                    "Frame size the iPhone camera captures. Larger frames show finer "
                    "speckle but need more USB bandwidth.")
        self.phone_size.activated.connect(self._phone_size_chosen)
        add(self.phone_size, "Frame size", basic=True)
        self.phone_crop = QComboBox()
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
        self.phone_send_cap = QComboBox()
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
        menu.addSeparator()
        # The live image has no tooltip: a popup there would hide the measurement.
        # The shortcut list lives here instead.
        hint = menu.addAction("Wheel to zoom, drag to pan, Q or Esc to quit")
        hint.setEnabled(False)  # a reminder, not a command
        set_tooltip(hint,
                    "Wheel: zoom. Drag: pan. 0: fit. 1: actual size. Space: freeze. "
                    "S: save. F: set reference. C: compare with the reference. "
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
            check.setEnabled(channel is not self.current)

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
            channel.start()
        self.current = channel
        self.drawn_sequence = 0
        self.shown_packet = None
        self.view.clear()
        if channel.frozen is not None:  # its held frame comes back at once
            self._draw(channel.frozen)
        self._show_frozen()
        self._load_settings(self.engine.settings(), channel.limits)
        self._set_demo(channel.demo)
        self._sync_cameras()
        self.log.write(f"Displaying {channel.label}")

    def _set_processing(self, channel, on):
        """Start or stop a camera that is not displayed.

        Args: channel is one of self.channels; on is the checkbox state.
        Returns: None.
        """
        if on and not channel.active:
            self.log.write(f"Processing on: {channel.label}")
            channel.start()
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
                              (self.contrast, s.contrast)):
            widget.set_value(value)
        toggles = ((self.rolling, s.rolling),
                   (self.gain_before_average, s.gain_before_average),
                   (self.invert, s.invert), (self.flip_button, s.flip_horizontal),
                   (self.even_brightness, s.even_brightness),
                   (self.frames_first, s.average_frames_first),
                   (self.floor_subtract, s.floor_subtract),
                   (self.record_button, s.record), (self.single_field, s.single_field),
                   (self.raw_difference_button, s.raw_difference))
        combos = ((self.smoothing_combo, s.smoothing_method),
                  (self.scaling, SCALING_MODES[1] if s.scale_range else SCALING_MODES[0]))
        for widget, _ in toggles + combos:
            widget.blockSignals(True)
        for widget, on in toggles:
            widget.setChecked(on)
        for widget, text in combos:
            widget.setCurrentText(text)
        for widget, _ in toggles + combos:
            widget.blockSignals(False)
        for button in self.mode_group.buttons():
            button.setChecked(button.text() == s.mode)
        self._sync_raw_difference(s.raw_difference)
        self._sync_scaling_controls(s.scale_range, s.gain_before_average)
        self.rolling.setEnabled(not s.average_frames_first)
        self.labview_button.blockSignals(True)
        self.labview_button.setChecked(self.current is not None and
                                       self.current.labview_saved is not None)
        self.labview_button.blockSignals(False)

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

    def _show_daq_dialog(self):
        """Open the NI DAQ window, creating its controller on first use.

        Args: None.
        Returns: None.
        """
        if self.daq is None:
            self.daq = daq.DaqController(self.log)
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

        Nothing is drawn while the displayed camera is frozen.

        Args: None.
        Returns: None.
        """
        if self.current.frozen is not None:
            return
        packet = self.engine.latest_packet()
        if packet is None or packet.sequence == self.drawn_sequence:
            return
        self.drawn_sequence = packet.sequence
        self._draw(packet)
        self.drawn_frames += 1

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

    def _poll_status(self):
        """Refresh status text, the log, and handle engine events.

        Args: None.
        Returns: None.
        """
        text = self.log.drain()
        if text:
            self.log_view.appendPlainText(text.rstrip("\n"))
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

    def closeEvent(self, event):
        """Stop the engine (finishing recordings) and save the window layout.

        Args: event is the Qt close event.
        Returns: None.
        """
        if not self.closing:
            self.closing = True
            self.tip_timer.stop()
            self.update_timer.stop()
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


def run(force_demo=False, check_updates=True, demo_cameras=1, log_level=None,
        send_reports=True):
    """Start the Qt front end.

    Args: force_demo skips camera detection and starts synthetic video;
        check_updates looks for a newer prebuilt binary (at startup and
        every few hours) and downloads this build's source code (if missing)
        at startup;
        demo_cameras is how many synthetic cameras demo mode simulates;
        log_level is the --log-level name or None; send_reports is False
        for --no-error-reports.
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
                    "Start dummy/demo mode with a synthetic example pattern?")
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
        for channel in channels:
            if channel.active:
                channel.engine.start()
        reporting.start_uploads()  # after the cameras opened, so their serials are scrubbed
        return app.exec()
    except Exception as exc:  # the hooks are restored below, before Python's own hook runs
        error_reports.caught(log, exc, "main_thread", "ESPI live stopped", fatal=True)
        raise
    finally:
        if window is None:
            for channel in channels:
                if channel.active:
                    channel.engine.stop()
                    channel.engine.wait(8.0)
        cleanup.stop()
        finish_update(window, log)
        reporting.stop()
        log.close()
