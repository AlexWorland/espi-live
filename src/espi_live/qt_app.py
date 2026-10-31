"""Cross-platform PySide6 front end with GPU image display.

The window is one QMainWindow: the image in the center, a dockable controls
panel and log, and a status bar. Docks can be floated as separate windows.

Display path: the engine publishes a Mono8 FramePacket; the GPU view uploads
it as a single-channel texture, and a fragment shader applies gamma and the
palette through a 256-entry lookup table built by processing.palette_rgb_table
(the same function saved frames use). Scaling to the window is done by the
GPU with mipmaps. Machines without OpenGL 3.3 (for example some remote-desktop
sessions) fall back to a CPU-drawn view with identical controls.

Several cameras: the window owns one Channel per detected camera. A drop-down
picks the camera on screen, and a "process" checkbox per camera keeps others
acquiring, processing and recording in the background. Every control acts on
the displayed camera's own engine and settings.
"""

import faulthandler
import html
import logging
import math
import os
import queue
import random
import sys
import textwrap
import time
import traceback
from datetime import datetime

import cv2
import numpy as np
from PySide6.QtCore import (QEvent, QObject, QPoint, QPointF, QRect, QSettings, QSize,
                            Qt, QtMsgType, QTimer, QUrl, Signal, qInstallMessageHandler)
from PySide6.QtGui import (QAction, QColor, QDesktopServices, QFontMetrics, QIcon, QImage,
                           QKeySequence, QOffscreenSurface, QOpenGLContext, QPainter,
                           QPainterPath, QPalette, QPen, QPixmap, QSurfaceFormat, QVector4D)
from PySide6.QtOpenGL import (QOpenGLShader, QOpenGLShaderProgram,
                              QOpenGLVertexArrayObject)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (QAbstractButton, QAbstractSpinBox, QApplication, QButtonGroup,
                               QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDockWidget, QDoubleSpinBox, QFormLayout, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProxyStyle, QPushButton, QSizePolicy, QSlider,
                               QStyle, QTextEdit, QToolBar, QToolTip, QVBoxLayout, QWidget)

from espi_live import camera, camera_reports, codex, error_reports, feedback, paths, source_code, tips, updater
from espi_live.channels import Channel, demo_channels, open_camera_channels
from espi_live.logs import AppLog, header_lines
from espi_live.processing import (MAX_DENOISE_PERCENT, PALETTES, SMOOTHING_METHODS,
                                  VIEW_MODES, palette_rgb_table)

# Wide enough that a full help tip and the DUMMY MODE badge both fit. A saved
# layout is left alone; a narrower window keeps the badge and elides the tip.
DEFAULT_WIDTH = 2100
DEFAULT_HEIGHT = 900

ANALOG_CONTROL_TIP = "Set on the camera itself; not adjustable over analog video"
# Gamma stays its own Display control: it is a curve, not a second stretch.
GAMMA_TIP = (
    "Shifts mid-tones after Stretch to 0–255, in Raw, Previous and Reference. "
    "Stretch is a straight remap of this frame's dark-to-bright range "
    "(contrast); gamma is a curve on top of that. Below 1.0 darkens mid-tones, "
    "above 1.0 brightens them. With a good stretch, start at 1.0."
)
STRETCH_TIP = (
    "Map each difference frame's darkest and brightest pixels, after Stretch "
    "clip, to 0 and 255 so fringes fill the display. This is a straight "
    "remap (contrast), not gamma. Off keeps the difference values as they "
    "are. The status bar shows both ranges. On by default. Previous and "
    "Reference only; Raw is not stretched."
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
uniform sampler2D lut;    // 256x1 RGB: gamma + palette
void main() {
    int index = int(texture(image, uv).r * 255.0 + 0.5);
    color = vec4(texelFetch(lut, ivec2(index, 0), 0).rgb, 1.0);
}
"""
MAX_ZOOM = 64.0
ZOOM_TICKS = 1000  # zoom slider resolution; position maps logarithmically to 1..MAX_ZOOM
CODEX_LABEL = "Open in Codex"
CODEX_BUSY_LABEL = "Getting source…"
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


class Reporting:
    """This run's error reporter, Python hooks, Qt message handler and worker."""

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
        self.reporter.collect_native_crashes(log.crash_path)
        self.worker.start()

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
# Previous and Reference views run in this order. Raw skips difference processing but still applies display gamma.
PIPELINE = ("noise reduction → difference → average → speckle filter → "
            "stretch → invert → gamma/CLAHE → palette")
# Opt-in processing setups. Exposure, gain, frame rate and recording stay put.
PROCESSING_PRESETS = {
    "Live preview": {
        "mode": "Previous",
        "noise_reduction": 0.0,
        "average": 1,
        "rolling": False,
        "ref_refresh_s": 0.0,
        "smoothing": 0,
        "smoothing_method": "Gaussian",
        "scale_range": True,
        "clip_percent": 0.0,
        "invert": False,
        "palette": "Gray",
        "gamma": 1.0,
        "clahe": False,
    },
    "Measurement": {
        "mode": "Reference",
        "noise_reduction": 0.0,
        "average": 8,
        "rolling": False,
        "ref_refresh_s": 2.0,
        "smoothing": 3,
        "smoothing_method": "Median",
        "scale_range": True,
        "clip_percent": 0.5,
        "invert": False,
        "palette": "Gray",
        "gamma": 1.0,
        "clahe": True,
    },
}
PRESET_TIPS = {
    "Live preview": "Watch fringes form while you align the setup: Previous view, "
                    "no averaging, no noise reduction, speckle filter off. "
                    "Does not change exposure, gain, frame rate or recording.",
    "Measurement": "A still object against one reference: Reference view, 8-frame "
                   "block average, Median filter 3 px, reference refresh 2 s. "
                   "Does not change exposure, gain, frame rate or recording.",
}
SETTINGS_GUIDE = f"""A typical ESPI measurement starts with one of the two presets. Neither changes exposure, gain, frame rate or recording.

Live preview — while you aim the camera and check that fringes form.
  View: Previous. Average N: 1 (off). Rolling average: off.
  Noise reduction: 0. Speckle filter: off. Ref. refresh: 0.
  Stretch to 0–255: on. Palette: Gray. Gamma: 1.0. CLAHE: off.
  Switch the view to Raw to frame the object. Raw skips difference processing but still applies display gamma.

Measurement — a still object, compared with one reference.
  View: Reference. Press F (or Set reference) once the object is still.
  Average N: 8 with Rolling average off: one clean image every 8 frames.
  Speckle filter: Median, width 3 px, well below the fringe spacing.
  Ref. refresh: 2 s. That cancels slow drift (air, heat). Motion slower than the refresh disappears; use 0 and press F when you need one fixed before/after.
  Noise reduction: 0. Raise it only when the camera is noisy and the object is still.
  Stretch to 0–255: on, clip 0.5% so a few speckles don't flatten the contrast.
  Palette: Gray, gamma 1.0. CLAHE is on but only affects a color palette; switch to Turbo if faint fringes are hard to see.

Processing order (Previous and Reference views):
  {PIPELINE}

Raw view applies noise reduction and display gamma, without stretch or palette.

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


def _paint_status_icon(kind, color):
    """Draw a 16px monochrome icon for a status-bar action.

    Args: kind is "refresh", "download" or "idea"; color is a QColor.
    Returns: A QPixmap with a transparent background.
    """
    pixmap = QPixmap(16, 16)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    try:
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
    finally:
        painter.end()
    return pixmap


def _status_icon(kind, color):
    """Build a status-bar icon, with a dimmer copy for the disabled state.

    Args: kind is "refresh", "download" or "idea"; color is a QColor.
    Returns: A QIcon.
    """
    icon = QIcon()
    icon.addPixmap(_paint_status_icon(kind, color), QIcon.Mode.Normal, QIcon.State.Off)
    dimmed = QColor(color)
    dimmed.setAlpha(150)
    icon.addPixmap(_paint_status_icon(kind, dimmed), QIcon.Mode.Disabled, QIcon.State.Off)
    return icon


def _arm_status_button(button, style, texts, icon=None):
    """Give a status-bar control button chrome, a pointer and a fixed width.

    The width fits every label the button shows, so a busy label cannot push
    the readouts sideways.

    Args: button is a QPushButton; style is its style sheet; texts are the
        labels; icon is a QIcon, or None.
    Returns: The fixed width in pixels.
    """
    button.setFlat(False)
    button.setCursor(Qt.PointingHandCursor)
    button.setFocusPolicy(Qt.StrongFocus)
    button.setStyleSheet(style)
    if icon is not None:
        button.setIcon(icon)
        button.setIconSize(QSize(16, 16))
    current = button.text()
    width = 0
    for text in texts:
        button.setText(text)
        width = max(width, button.sizeHint().width())
    button.setText(current)
    button.setFixedWidth(width)
    return width


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
            "Gamma applies; no difference, stretch or palette."),
    "Previous": ("Previous: the difference from the last frame, so fringes show "
                 "what just changed. Shortcut: R restarts that difference."),
    "Reference": ("Reference: the difference from a stored frame. Shortcut: F "
                  "stores the live frame. Ref. refresh can recapture it on a "
                  "timer so slow drift cancels."),
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


def install_tooltip_delay():
    """Apply the tooltip delay once for this process.

    Args: None. A QApplication must already exist.
    Returns: None.
    """
    app = QApplication.instance()
    if isinstance(app.style(), TooltipDelayStyle):
        return
    app.setStyle(TooltipDelayStyle())


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
    """Draw Mono8 frames with GPU scaling, gamma and palette."""

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
        """Compile the shaders and create the image and palette textures.

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
        self.slider = QSlider(Qt.Horizontal)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setSuffix(suffix)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(104)  # equal widths keep every value column aligned
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
        self.resize(560, 420)
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
    """Explain the processing controls and offer the two starting presets."""

    def __init__(self, parent):
        """Build the guide text and the buttons that apply a preset.

        Args: parent is the main window; its apply_preset handles the buttons.
        Returns: None.
        """
        super().__init__(parent)
        self.setWindowTitle("Which settings should I use?")
        self.resize(680, 640)
        self.guide = QPlainTextEdit(SETTINGS_GUIDE)
        self.guide.setReadOnly(True)
        self.guide.setObjectName("settingsGuide")
        buttons = QDialogButtonBox()
        live = buttons.addButton("Apply Live preview", QDialogButtonBox.ActionRole)
        measure = buttons.addButton("Apply Measurement", QDialogButtonBox.ActionRole)
        close = buttons.addButton(QDialogButtonBox.Close)
        live.clicked.connect(lambda: parent.apply_preset("Live preview"))
        measure.clicked.connect(lambda: parent.apply_preset("Measurement"))
        close.clicked.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.guide, 1)
        layout.addWidget(buttons)


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
        self._reserved = max(self.fontMetrics().horizontalAdvance(text)
                             for text in samples) + 2
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)

    def hold_full_text(self):
        """Keep the reserved width, so the samples are never shortened.

        Args: None.
        Returns: None.
        """
        self._hold = True

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


class MainWindow(QMainWindow):
    """Viewer, controls, log and status for the displayed camera's engine."""

    def __init__(self, channels, log, use_gl, check_updates=True, reporting=None):
        """Build the window and start polling the engines.

        Args: channels is a list of Channel, at least one of them active
            (opened, not yet started); the first active one is displayed;
            log is the AppLog; use_gl selects the GPU view instead of the CPU
            fallback; check_updates looks for a newer prebuilt binary at
            startup and again every few hours; reporting is the run's
            Reporting, or None; and the window downloads this build's source
            code if it is missing (otherwise that waits for Open in Codex).
            False skips every update check, including the later ones.
        Returns: None.
        """
        super().__init__()
        install_tooltip_delay()
        self.setAttribute(Qt.WA_AlwaysShowToolTips, True)
        self.channels = list(channels)
        self.current = next(channel for channel in self.channels if channel.active)
        self.log = log
        self.reporting = reporting
        self.report_outcomes = {}  # fingerprint -> latest error_reports.Outcome
        self.fatal_boxes = {}  # fingerprint -> (open "ESPI stopped" box, its message)
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
        self.frozen_banner.setStyleSheet(
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
        self._rebuild_cameras()
        self._set_demo(self.current.demo)
        self.shortcut_filter = ShortcutFilter(self)
        self.tooltip_filter = TooltipFilter(self)
        app = QApplication.instance()
        app.installEventFilter(self.shortcut_filter)
        app.installEventFilter(self.tooltip_filter)

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
        bar.addWidget(self.freeze_button)
        for label, tip, handler in (
                ("Save frame (S)",
                 "Save the displayed frame as a PNG, plus a notes file when the title "
                 "or notes box is filled in. Shortcut: S.",
                 self.save_frame),
                ("Set reference (F)",
                 "Use the current live frame as the Reference view's base. Shortcut: F.",
                 lambda: self.engine.request("reference")),
                ("Reset last (R)",
                 "Restart the Previous view's difference from the next frame. Shortcut: R.",
                 lambda: self.engine.request("reset_previous")),
                ("Reset average (A)",
                 "Discard the frames averaged so far and start the average again. "
                 "Shortcut: A.",
                 lambda: self.engine.request("reset_average"))):
            button = QPushButton(label)
            set_tooltip(button, tip)
            button.clicked.connect(handler)
            bar.addWidget(button)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(0, ZOOM_TICKS)
        # A wheel notch scrolls three single steps; make that one 1.25x step,
        # matching wheel zoom on the image.
        self.zoom_slider.setSingleStep(round(
            ZOOM_TICKS * math.log(1.25) / math.log(MAX_ZOOM) / 3))
        self.zoom_slider.setPageStep(ZOOM_TICKS // 6)
        self.zoom_slider.setFixedWidth(130)
        self.zoom_slider.setAttribute(Qt.WA_AlwaysShowToolTips, True)
        self.zoom_slider.valueChanged.connect(
            lambda ticks: self.view.set_zoom(MAX_ZOOM ** (ticks / ZOOM_TICKS)))
        self.zoom_label = QLabel()
        self.zoom_label.setMinimumWidth(44)
        set_tooltip(self.zoom_label, "100% means the whole image fits the window.")
        self.view.zoomChanged.connect(self._show_zoom)
        fit_button = QPushButton("Fit")
        set_tooltip(fit_button, "Show the whole image. Shortcut: 0.")
        fit_button.clicked.connect(self.view.fit)
        actual_button = QPushButton("1:1")
        set_tooltip(actual_button, "One camera pixel per screen pixel. Shortcut: 1.")
        actual_button.clicked.connect(self.view.actual_size)
        zoom_icon = QLabel("Zoom")
        zoom_icon.setContentsMargins(0, 0, 4, 0)
        set_tooltip(zoom_icon, "Zoom from the whole image (100%) up to 64 times.")
        bar.addWidget(zoom_icon)
        bar.addWidget(self.zoom_slider)
        bar.addWidget(self.zoom_label)
        bar.addWidget(fit_button)
        bar.addWidget(actual_button)
        self._show_zoom(self.view.zoom)

    def _build_controls(self):
        """Create the controls dock as one aligned grid of settings.

        Args: None.
        Returns: None.
        """
        s = self.engine.settings()
        panel = QWidget()
        panel.setAttribute(Qt.WA_AlwaysShowToolTips, True)
        column = QVBoxLayout(panel)
        column.setContentsMargins(10, 8, 10, 8)

        # Shown only when there is more than one camera.
        self.cameras_box = QWidget()
        cameras = QVBoxLayout(self.cameras_box)
        cameras.setContentsMargins(0, 0, 0, 6)
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
        process_note = QLabel("Process (keeps acquiring and recording when not displayed)")
        process_note.setWordWrap(True)
        process_note.setStyleSheet("color: palette(placeholder-text);")
        cameras.addWidget(process_note)
        self.camera_rows = QGridLayout()
        self.camera_rows.setColumnStretch(0, 1)
        self.camera_rows.setVerticalSpacing(2)
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
        self.demo_banner.setStyleSheet(
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

        # One grid for every section keeps labels, sliders and values in the
        # same three columns; headings replace group-box frames.
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        column.addLayout(grid)
        row = 0

        def heading(text):
            """Add a section heading row.

            Args: text is the section name.
            Returns: None.
            """
            nonlocal row
            label = QLabel(text)
            label.setStyleSheet("font-weight:bold;")
            label.setContentsMargins(0, 0 if row == 0 else 10, 0, 2)
            grid.addWidget(label, row, 0, 1, 2)
            row += 1

        def add(widget, label=None):
            """Add a labeled setting, or a full-width widget when unlabeled.

            Args: widget is the control; label is its name, if any.
            Returns: None.
            """
            nonlocal row
            if label is None:
                grid.addWidget(widget, row, 0, 1, 2)
            else:
                name = QLabel(label)
                name.setToolTip(widget.toolTip())
                grid.addWidget(name, row, 0)
                grid.addWidget(widget, row, 1)
            row += 1

        def note(label):
            """Style a secondary, wrapping status line and add it full width.

            Args: label is the QLabel to style.
            Returns: None.
            """
            label.setWordWrap(True)
            label.setStyleSheet("color: palette(placeholder-text);")
            add(label)

        heading("Camera")
        self.model_label = QLabel()  # detected model; hidden in dummy mode
        self.model_label.setWordWrap(True)
        self.model_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        add(self.model_label)
        self.report_label = QLabel()  # camera-report status for unrecognised devices
        self.report_label.setWordWrap(True)
        self.report_label.setTextFormat(Qt.RichText)
        self.report_label.setOpenExternalLinks(True)
        self.report_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        set_tooltip(self.report_label,
                    "Whether a report for this unrecognised camera was filed, and a "
                    "link to the issue when one was.")
        self.report_label.hide()
        add(self.report_label)
        self.report_button = QPushButton("Report this camera")
        set_tooltip(self.report_button,
                    "Open GitHub's new-issue page with this camera's details. This "
                    "copy cannot file reports itself.")
        self.report_button.clicked.connect(self._open_report_in_browser)
        self.report_button.hide()
        add(self.report_button)
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
        add(self.exposure, "Exposure")
        add(self.gain, "Gain")
        add(self.fps, "Requested FPS")
        self.single_field = QCheckBox("Single field (interlaced camera)")
        set_tooltip(self.single_field,
                    "Show only the even rows of each interlaced frame, doubled to full "
                    "height: removes the 1/60 s offset between the two fields (cleaner "
                    "fringes) at half the vertical detail. Changing it clears the "
                    "reference and the averages. On by default for an interlaced camera.")
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
            "watch for less boiling and a trail when something moves. Stretch "
            "to 0–255 repaints a quieter difference at full brightness; judge "
            "it with stretch off, or by the status bar's Input range. This is "
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
            "smoother still image. Stretch hides a quieter difference. Not "
            "the Speckle filter or Average N.")
        note(self.noise_note)
        self.camera_label = QLabel()
        note(self.camera_label)

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

        heading("Fringe enhancement")
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
        self.scale_range = QCheckBox("Stretch to 0–255")
        set_tooltip(self.scale_range, STRETCH_TIP,
                    "Gamma is a separate curve applied afterwards and does not "
                    "replace this stretch. Start on.")
        self.scale_range.setChecked(s.scale_range)
        self.scale_range.toggled.connect(
            lambda on: self.engine.update_settings(scale_range=on))
        self.invert = QCheckBox("Invert")
        set_tooltip(self.invert,
            "Swaps bright and dark fringes. Use it when the pattern is easier to "
            "read the other way around. Start off.")
        self.invert.setChecked(s.invert)
        self.invert.toggled.connect(lambda on: self.engine.update_settings(invert=on))
        toggles = QWidget()
        toggles_row = QHBoxLayout(toggles)
        toggles_row.setContentsMargins(0, 0, 0, 0)
        toggles_row.addWidget(self.scale_range)
        toggles_row.addWidget(self.invert)
        toggles_row.addStretch(1)
        add(toggles)
        self.clip = SliderSpin(0, 10, 0.1, 1, s.clip_percent, " %")
        self.clip.set_description(
            "With Stretch on, lets this share of the darkest and brightest pixels "
            "saturate so a few stray speckles don't flatten the contrast. 0 uses "
            "the exact minimum and maximum. Start at 0; try 0.5% when speckles "
            "wash the image out.")
        self.clip.valueChanged.connect(
            lambda v: self.engine.update_settings(clip_percent=v))
        add(self.clip, "Stretch clip")

        heading("Display")
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(PALETTES)
        self.palette_combo.setCurrentText(s.palette)
        set_tooltip(self.palette_combo,
            "Colors the stretched image. Gray shows brightness and is what you "
            "want for a measurement. A color palette (Turbo, Jet and the others) "
            "makes small steps easier to see; pair it with CLAHE. Start with Gray.")
        self.palette_combo.currentTextChanged.connect(
            lambda name: self.engine.update_settings(palette=name))
        self.gamma = SliderSpin(0.1, 3.0, 0.1, 1, s.gamma)
        self.gamma.set_description(GAMMA_TIP + " Start at 1.0.")
        self.gamma.valueChanged.connect(lambda v: self.engine.update_settings(gamma=v))
        self.clahe = QCheckBox("CLAHE (color palettes only)")
        set_tooltip(self.clahe,
            "Boosts local contrast so faint fringes show up in a color palette. "
            "It does nothing while the palette is Gray. Turn it on when a color "
            "palette still looks flat; leave it off when the gray values must "
            "stay as they are. Start on with a color palette, and off with Gray.")
        self.clahe.setChecked(s.clahe)
        self.clahe.toggled.connect(lambda on: self.engine.update_settings(clahe=on))
        add(self.palette_combo, "Palette")
        add(self.gamma, "Gamma")
        add(self.clahe)

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
        self.frame_notes.setFixedHeight(48)
        add(self.frame_title)
        add(self.frame_notes)

        column.addStretch(1)

        # No scroll area: every control is always on screen, and the window's
        # minimum height grows to fit the panel instead.
        panel.setMinimumWidth(320)
        self.controls_dock = QDockWidget("Controls", self)
        self.controls_dock.setObjectName("controls")
        self.controls_dock.setWidget(panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.controls_dock)
        self._tip_dock_buttons(self.controls_dock)

    def _add_preset_buttons(self, add):
        """Add the Live preview and Measurement buttons to the controls grid.

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
        """Open the guide to the processing controls and the two presets.

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
        self.resizeDocks([self.log_dock], [120], Qt.Vertical)

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
        self.error_report_label = QLabel()  # error-report results, with the issue link
        self.error_report_label.setTextFormat(Qt.RichText)
        self.error_report_label.setOpenExternalLinks(True)
        self.error_report_label.setVisible(False)
        self.error_report_label.setStyleSheet("color:#b00020;")
        bar.addWidget(self.error_report_label)
        self.error_report_timer = QTimer(self)
        self.error_report_timer.setSingleShot(True)
        self.error_report_timer.timeout.connect(self.error_report_label.hide)
        settings = QSettings("ESPI", "espi_live")
        self.tips = tips.TipRotator(
            hidden=bool(settings.value("tips/hidden", False, bool)))
        # Ignored, with no minimum, so the badges keep the width they had
        # before the tip existed. Stretch 20 against the view hint's 1 fills
        # the sentence first; the cap is the longest tip, and the hint gets
        # whatever is left. Right-click and View > Help tips dismiss it.
        self.tip_label = TipLabel(*tips.TIPS)
        self.tip_label.clicked.connect(self._rotate_tip)
        self.tip_label.dismissed.connect(lambda: self._set_tips_visible(False))
        self.tip_label.setMinimumWidth(0)
        self.tip_label.setMaximumWidth(self.tip_label.sizeHint().width())
        self.tip_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        bar.addWidget(self.tip_label, 20)
        self.hint_label = QLabel()  # takes the space the tip does not use
        self.hint_label.setMinimumWidth(0)
        self.hint_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        set_tooltip(self.hint_label, "What the current view is showing.")
        bar.addWidget(self.hint_label, 1)
        self._show_tip()
        # Leftmost permanent item: showing it only narrows the hint, so it moves
        # nothing else.
        self.frozen_badge = QLabel("  FROZEN  ")
        self.frozen_badge.setStyleSheet(FROZEN_STYLE)
        set_tooltip(self.frozen_badge,
                    "The image on screen is held; the camera keeps running. "
                    "Shortcut: Space returns to live.")
        self.frozen_badge.setVisible(False)
        bar.addPermanentWidget(self.frozen_badge)
        # Every readout has a fixed width, so no value can move its neighbours.
        self.source_badge = FixedLabel("  DUMMY MODE  ", "  BFS-U3-200S6M-C  ")
        self.source_badge.hold_full_text()  # never "DUMM…"; the readouts ellipsize first
        self.record_label = FixedLabel("Recording: finishing file", "Recording: 999.9 fps")
        set_tooltip(self.record_label,
                    "Whether this camera is recording, and the rate the file is "
                    "being written.")
        self.range_status = FixedLabel("Input: 255–255    Scaled: 255–255",
                                       "Raw Mono8: no brightness scaling")
        set_tooltip(self.range_status,
                    "Brightness before and after stretch to 0–255, measured before "
                    "gamma and the palette. Raw view is not scaled.")
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
        # The two update buttons are never shown together, so they share one
        # fixed-width slot that stays put whichever is visible. They are real
        # buttons (border, icon, pointer), split from the readouts by a rule.
        self.update_button = QPushButton()  # shown when a newer build exists
        self.update_button.setVisible(False)
        set_tooltip(self.update_button, "Download the newer build of ESPI live.")
        self.update_button.clicked.connect(self._update_clicked)
        self.check_button = QPushButton("Check for updates")  # until a newer build is found
        set_tooltip(self.check_button, "Check GitHub for a newer build of ESPI live.")
        self.check_button.clicked.connect(self._check_updates_now)
        ink = self.check_button.palette().color(QPalette.ColorRole.ButtonText)
        update_width = _arm_status_button(
            self.update_button, UPDATE_BUTTON_STYLE,
            ("Install update", "Downloading update… 100%",
             "Restart to update"),
            _status_icon("download", QColor(Qt.white)))
        check_width = _arm_status_button(
            self.check_button, STATUS_BUTTON_STYLE,
            ("Check for updates", "Checking for updates…"),
            _status_icon("refresh", ink))
        slot_width = max(update_width, check_width)
        self.update_slot = QWidget()
        slot = QGridLayout(self.update_slot)
        slot.setContentsMargins(0, 0, 0, 0)
        for button in (self.update_button, self.check_button):
            button.setFixedWidth(slot_width)
            slot.addWidget(button, 0, 0)  # same cell: they overlap
        self.update_slot.setFixedWidth(slot_width)
        bar.addPermanentWidget(self.update_slot)
        self.feature_button = QPushButton("Request feature")
        set_tooltip(self.feature_button,
                    "Send a feature request to the developers as a GitHub issue.")
        self.feature_button.clicked.connect(lambda: self._request_feature())
        _arm_status_button(self.feature_button, STATUS_BUTTON_STYLE,
                           ("Request feature", "Sending request…"),
                           _status_icon("idea", ink))
        bar.addPermanentWidget(self.feature_button)
        self.codex_button = QPushButton(CODEX_LABEL)
        set_tooltip(self.codex_button,
                    "Ask OpenAI's Codex how ESPI live works, using this build's "
                    "real source code.")
        self.codex_button.clicked.connect(self._open_codex)
        _arm_status_button(self.codex_button, STATUS_BUTTON_STYLE,
                           (CODEX_LABEL, CODEX_BUSY_LABEL))
        bar.addPermanentWidget(self.codex_button)
        self.status_actions_rule = QFrame()
        self.status_actions_rule.setFrameShape(QFrame.Shape.VLine)
        self.status_actions_rule.setFrameShadow(QFrame.Shadow.Sunken)
        self.status_actions_rule.setFixedWidth(10)
        bar.addPermanentWidget(self.status_actions_rule)
        buttons = (self.update_button, self.check_button, self.feature_button,
                   self.codex_button)
        height = max(button.sizeHint().height() for button in buttons)
        for button in buttons:
            button.setFixedHeight(height)
        for widget in (self.source_badge, self.record_label, self.range_status,
                       self.camera_fps_status, self.speed_status, self.pixel_status):
            bar.addPermanentWidget(widget)

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
        menu.addAction(self.tips_action)
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
                    "S: save. F: set reference. R: reset last. A: reset average. "
                    "Q or Esc: quit.")

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
        about = help_menu.addAction("About ESPI live")
        set_tooltip(about, "Show the version and build of this copy.")
        about.triggered.connect(lambda: QMessageBox.about(
            self, "About ESPI live", updater.describe_build()))

    # ----- cameras -------------------------------------------------------------

    def _rebuild_cameras(self):
        """Fill the camera drop-down and process checkboxes from self.channels.

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
            self.camera_rows.addWidget(check, row, 0)
            self.camera_rows.addWidget(state, row, 1)
            self.camera_checks.append(check)
            self.camera_states.append(state)
        self.cameras_box.setVisible(several)
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
        self._apply_limits({**DEFAULT_LIMITS, **limits})
        for widget, value in ((self.exposure, s.exposure_us), (self.gain, s.gain_db),
                              (self.fps, s.fps), (self.noise_reduction, s.noise_reduction),
                              (self.average, s.average),
                              (self.ref_refresh, s.ref_refresh_s),
                              (self.smoothing, s.smoothing), (self.clip, s.clip_percent),
                              (self.gamma, s.gamma)):
            widget.set_value(value)
        toggles = ((self.rolling, s.rolling), (self.scale_range, s.scale_range),
                   (self.invert, s.invert), (self.clahe, s.clahe),
                   (self.record_button, s.record), (self.single_field, s.single_field))
        combos = ((self.smoothing_combo, s.smoothing_method),
                  (self.palette_combo, s.palette))
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

    def _apply_limits(self, limits):
        """Set the camera sliders to device limits.

        Args: limits is {node name: (low, high)} for the names to change.
        Returns: None.
        """
        for name, widget in (("ExposureTime", self.exposure), ("Gain", self.gain),
                             ("AcquisitionFrameRate", self.fps)):
            if name in limits:
                widget.set_range(*limits[name])

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
        channel.demo_source = False
        if channel.engine.camera_info is not None:
            channel.info = channel.engine.camera_info
            channel.engine.tag = channel.info.serial
        for other in self.channels:
            if other is not channel and other.demo_source:
                other.stop()
                self.retired_channels.append(other)
        self.channels = [channel] + [
            Channel(self.log, info) for info in cameras
            if info.serial != channel.info.serial]
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
        if demo:
            title = f"ESPI live — {self.current.label}" if several else "ESPI live"
            self.setWindowTitle(f"{title} — DUMMY MODE")
        elif identity:  # the serial tells same-model cameras apart
            serial = f" ({self.current.info.serial})" if several else ""
            self.setWindowTitle(f"ESPI live — {name}{serial}")
        else:
            self.setWindowTitle(f"ESPI live — {self.current.label}" if several
                                else "ESPI live")
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
        and frame rate, so those controls are disabled for it; interlaced
        sources also get the single-field toggle.

        Args: None.
        Returns: None.
        """
        channel = self.current
        analog = not channel.demo and channel.info.kind == "capture"
        if analog:
            note = ANALOG_CONTROL_TIP
        elif not self.demo_banner.isHidden():
            # isHidden, not isVisible: the banner is flagged shown before the
            # window itself is on screen, which is when this first runs.
            note = SIMULATED_TIP
        else:
            note = ""
        for widget in (self.exposure, self.gain, self.fps):
            widget.setEnabled(not analog)
            widget.set_note(note)
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
        key = (packet.palette, round(packet.gamma, 4))
        if key != getattr(self, "table_key", None):
            self.table = palette_rgb_table(packet.palette, packet.gamma)
            self.table_key = key
        self.view.show_packet(packet, self.table, key)
        self.shown_packet = packet

    def _show_tip(self):
        """Put the current tip in the status bar, or hide it.

        Args: None.
        Returns: None.
        """
        text = self.tips.current()
        shown = not self.tips.hidden
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
        self.feature_button.setText("Sending request…")
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
            self.feature_button.setText("Request feature")
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
            self.codex_button.setText(CODEX_BUSY_LABEL)
            if not self.source.start():
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
        if QDesktopServices.openUrl(QUrl.fromEncoded(
                codex.deep_link(self.source_folder).encode("ascii"))):
            self.log.write(f"Opened Codex on {self.source_folder}")
            self.statusBar().showMessage("Opening Codex with ESPI live's source code…", 5000)
            return
        box = QMessageBox(QMessageBox.Warning, "Codex not found",
                          "Codex did not open. Install the Codex app (or the ChatGPT "
                          "desktop app, which includes it), sign in, then press "
                          "Open in Codex again.", parent=self)
        get = box.addButton("Get Codex…", QMessageBox.ActionRole)
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
                          "Codex, OpenAI's coding assistant, opens with the source code of "
                          "this copy of ESPI live, so you can ask it how the program works."
                          "<br><br>If you ask it to change something, it shows you the "
                          "change and asks whether you are happy with it. If you are, it "
                          "sends the change to the ESPI live developers to review. Your "
                          "installed program stays as it is.<br><br>You need the Codex "
                          "app (or the ChatGPT desktop app) and a ChatGPT account."
                          f"<br><br>Source code: {html.escape(self.source_folder)}",
                          parent=self)
        box.setTextFormat(Qt.RichText)
        go = box.addButton("Open Codex", QMessageBox.AcceptRole)
        get = box.addButton("Get Codex…", QMessageBox.ActionRole)
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
            self.codex_button.setText(CODEX_LABEL)
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
        self.check_button.setEnabled(not checking)
        self.check_button.setText("Checking for updates…" if checking else "Check for updates")

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
        elif kind == "record_off" and displayed:
            self.record_button.blockSignals(True)
            self.record_button.setChecked(False)
            self.record_button.blockSignals(False)
        elif kind == "switched_live":
            # A held demo frame must not outlive the dummy-mode indicators.
            channel.frozen = channel.frozen_at = None
            self._show_frozen()
            self._adopt_live_cameras(channel, payload or [])
            for button in self.mode_group.buttons():
                button.setChecked(button.text() == "Raw")
            self.record_button.blockSignals(True)
            self.record_button.setChecked(False)
            self.record_button.blockSignals(False)
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
            self.statusBar().showMessage("Finishing recordings and releasing the camera…")
            for engine in self._all_engines():
                engine.stop()
            deadline = time.monotonic() + 8.0
            while not self.wait_for_engines(0.05) and time.monotonic() < deadline:
                QApplication.processEvents()
            self._poll_status()
            settings = QSettings("ESPI", "espi_live")
            settings.setValue("geometry", self.saveGeometry())
            settings.setValue("windowState", self.saveState(LAYOUT_VERSION))
        app = QApplication.instance()
        app.removeEventFilter(self.shortcut_filter)
        app.removeEventFilter(self.tooltip_filter)
        event.accept()


def finish_update(window, log):
    """Install a downloaded build after the window closed, and restart into it.

    Runs once the engine has stopped, so the camera is free for the new copy.

    Args: window is the closed MainWindow (or None); log is the AppLog.
    Returns: None.
    """
    if window is None or not window.pending_update:
        return
    try:
        path = updater.install_update(window.pending_update)
        log.write(f"Installed update: {path}")
        if window.relaunch_after_exit and not window.wait_for_engines(5.0):
            log.write("The camera was not released in time; start ESPI live again "
                      "to use the new build")
        elif window.relaunch_after_exit:
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
        if not force_demo:
            try:
                channels = open_camera_channels(log)
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
            channels = demo_channels(log, demo_cameras)
        use_gl, description = opengl_available()
        display = f"GPU ({description})" if use_gl else f"CPU fallback ({description})"
        log.write(f"Display: {display}")
        log.info(describe_session(channels, display))
        reporting.reporter.update_context(display=display)
        window = MainWindow(channels, log, use_gl, check_updates, reporting)
        window.show()
        for channel in channels:
            if channel.active:
                channel.engine.start()
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
