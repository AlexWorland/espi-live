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

import math
import sys
import time

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QObject, QPointF, QSettings, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (QAction, QDesktopServices, QFontMetrics, QImage, QKeySequence, QOffscreenSurface,
                           QOpenGLContext, QPainter, QSurfaceFormat, QVector4D)
from PySide6.QtOpenGL import (QOpenGLShader, QOpenGLShaderProgram,
                              QOpenGLVertexArrayObject)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QButtonGroup,
                               QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDockWidget, QDoubleSpinBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QMessageBox, QPlainTextEdit,
                               QPushButton, QSizePolicy, QSlider,
                               QTextEdit, QToolBar, QVBoxLayout, QWidget)

from espi_live import feedback, updater
from espi_live.channels import Channel, demo_channels, open_camera_channels
from espi_live.logs import AppLog
from espi_live.processing import (PALETTES, SMOOTHING_METHODS, VIEW_MODES,
                                  palette_rgb_table)

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
LAYOUT_VERSION = 4  # bump when docks/toolbars change so stale saved layouts are ignored
TEXT_INPUTS = (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QComboBox)
# Control ranges before a camera reports its limits (CAMERA_RANGES in camera.py).
DEFAULT_LIMITS = {"ExposureTime": (0, 50000), "Gain": (0, 30),
                  "AcquisitionFrameRate": (0, 60)}


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
        self.slider = QSlider(Qt.Horizontal)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setSuffix(suffix)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(104)  # equal widths keep every value column aligned
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


class FeatureRequestDialog(QDialog):
    """Collect a feature request: title, description, name and enrichment."""

    def __init__(self, parent, request=None):
        """Build the form, prefilled from an earlier attempt or saved name.

        Args: parent is the main window; request is an optional
            FeatureRequest whose text is restored (after a failed send).
        Returns: None.
        """
        super().__init__(parent)
        self.setWindowTitle("Request a feature")
        self.resize(560, 460)
        settings = QSettings("ESPI", "espi_live")
        self.title = QLineEdit(request.title if request else "")
        self.title.setMaxLength(feedback.MAX_TITLE)
        self.title.setPlaceholderText("Short summary, e.g. Export the reference frame")
        self.description = QPlainTextEdit(request.description if request else "")
        self.description.setPlaceholderText(
            "What would you like ESPI live to do, and why? What are you "
            "measuring when you need it?")
        self.submitter = QLineEdit(request.submitter if request else
                                   settings.value("feedback/name", "", str))
        self.submitter.setPlaceholderText("Optional, so we can follow up")
        self.enrich = QCheckBox("Enrich with GitHub Copilot")
        self.enrich.setToolTip(
            "After the issue is filed, GitHub Copilot adds a structured version "
            "(summary, proposed behavior, acceptance criteria) below your text.")
        self.enrich.setChecked(request.enrich if request else
                               settings.value("feedback/enrich", True, bool))
        form = QFormLayout()
        form.addRow("Title", self.title)
        form.addRow("Description", self.description)
        form.addRow("Your name", self.submitter)
        form.addRow("", self.enrich)
        note = QLabel(f"Filed as a GitHub issue in {feedback.repository()}, with "
                      "this build's version and platform.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(
            "Send" if feedback.feedback_token() else "Open in browser…")
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
            enrich=self.enrich.isChecked(), submitter=self.submitter.text(),
            source=source)

    def accept(self):
        """Close only when the request is complete; remember name and choice.

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
        settings.setValue("feedback/enrich", self.enrich.isChecked())
        super().accept()


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
        self._reserved = max(self.fontMetrics().horizontalAdvance(text)
                             for text in samples) + 2
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)

    def sizeHint(self):
        """Return the reserved width, whatever the text.

        Args: None.
        Returns: A QSize.
        """
        return QSize(self._reserved, super().sizeHint().height())

    def minimumSizeHint(self):
        """Allow shrinking to nothing when the window is too narrow.

        Args: None.
        Returns: A QSize.
        """
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


class MainWindow(QMainWindow):
    """Viewer, controls, log and status for the displayed camera's engine."""

    def __init__(self, channels, log, use_gl, check_updates=True):
        """Build the window and start polling the engines.

        Args: channels is a list of Channel, at least one of them active
            (opened, not yet started); the first active one is displayed;
            log is the AppLog; use_gl selects the GPU view instead of the CPU
            fallback; check_updates looks for a newer prebuilt binary at
            startup.
        Returns: None.
        """
        super().__init__()
        self.channels = list(channels)
        self.current = next(channel for channel in self.channels if channel.active)
        self.log = log
        self.closing = False
        self.updates = updater.UpdateWorker()
        self.update_info = None
        self.update_manual = False  # report "up to date" and errors in dialogs
        self.pending_update = None  # unpacked build to install on exit
        self.relaunch_after_exit = False
        self.feedback = feedback.FeedbackWorker()
        self.retired_channels = []  # demo channels replaced by real cameras
        self.drawn_sequence = 0
        self.drawn_frames = 0
        self.fps_window_start = time.monotonic()
        self.display_fps = 0.0
        self.hover = (-1, -1)
        self.view = GLImageView() if use_gl else CpuImageView()
        self.view.hovered.connect(self._set_hover)
        self.setCentralWidget(self.view)
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
        QApplication.instance().installEventFilter(self.shortcut_filter)

        settings = QSettings("ESPI", "espi_live")
        if not self.restoreGeometry(settings.value("geometry", b"")):
            self.resize(1400, 900)
        self.restoreState(settings.value("windowState", b""), LAYOUT_VERSION)

        self.frame_timer = QTimer(self)
        self.frame_timer.setTimerType(Qt.PreciseTimer)
        self.frame_timer.timeout.connect(self._poll_frame)
        self.frame_timer.start(4)
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._poll_status)
        self.status_timer.start(200)
        if check_updates and updater.unavailable_reason() is None:
            self.updates.check()

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
        self.addToolBar(Qt.TopToolBarArea, bar)
        self.actions_bar = bar

        self.mode_group = QButtonGroup(self)
        for mode in VIEW_MODES:
            button = QPushButton(mode)
            button.setCheckable(True)
            button.setChecked(mode == s.mode)
            self.mode_group.addButton(button)
            bar.addWidget(button)
        self.mode_group.buttonClicked.connect(
            lambda button: self.engine.update_settings(mode=button.text()))
        bar.addSeparator()

        self.record_button = QPushButton("Record AVI")
        self.record_button.setCheckable(True)
        self.record_button.toggled.connect(
            lambda on: self.engine.update_settings(record=on))
        bar.addWidget(self.record_button)
        bar.addSeparator()
        for label, tip, handler in (
                ("Save frame (S)", "Save the displayed frame with its title and notes",
                 self.save_frame),
                ("Set reference (F)", "Use the current frame as the Reference view's base",
                 lambda: self.engine.request("reference")),
                ("Reset last (R)", "Restart the Previous view's difference",
                 lambda: self.engine.request("reset_previous")),
                ("Reset average (A)", "Discard the frames averaged so far",
                 lambda: self.engine.request("reset_average"))):
            button = QPushButton(label)
            button.setToolTip(tip)
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
        self.zoom_slider.setToolTip("Drag, click or scroll to zoom; the mouse "
                                    "wheel and drag-to-pan on the image still work")
        self.zoom_slider.valueChanged.connect(
            lambda ticks: self.view.set_zoom(MAX_ZOOM ** (ticks / ZOOM_TICKS)))
        self.zoom_label = QLabel()
        self.zoom_label.setMinimumWidth(44)
        self.zoom_label.setToolTip("100% = whole image fits the window")
        self.view.zoomChanged.connect(self._show_zoom)
        fit_button = QPushButton("Fit")
        fit_button.setToolTip("Show the whole image (0)")
        fit_button.clicked.connect(self.view.fit)
        actual_button = QPushButton("1:1")
        actual_button.setToolTip("One camera pixel per screen pixel (1)")
        actual_button.clicked.connect(self.view.actual_size)
        zoom_icon = QLabel("Zoom")
        zoom_icon.setContentsMargins(0, 0, 4, 0)
        bar.addWidget(zoom_icon)
        bar.addWidget(self.zoom_slider)
        bar.addWidget(self.zoom_label)
        bar.addWidget(fit_button)
        bar.addWidget(actual_button)

    def _build_controls(self):
        """Create the controls dock as one aligned grid of settings.

        Args: None.
        Returns: None.
        """
        s = self.engine.settings()
        panel = QWidget()
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
        self.camera_combo.setToolTip("Camera shown in the window; the controls "
                                     "below change this camera's settings")
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
                grid.addWidget(QLabel(label), row, 0)
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
        self.exposure = SliderSpin(*DEFAULT_LIMITS["ExposureTime"], 1, 0,
                                   s.exposure_us, " µs")
        self.gain = SliderSpin(*DEFAULT_LIMITS["Gain"], 0.1, 1, s.gain_db, " dB")
        self.fps = SliderSpin(*DEFAULT_LIMITS["AcquisitionFrameRate"], 0.1, 1,
                              s.fps, " fps")
        self.exposure.valueChanged.connect(
            lambda v: self.engine.update_settings(exposure_us=v))
        self.gain.valueChanged.connect(lambda v: self.engine.update_settings(gain_db=v))
        self.fps.valueChanged.connect(lambda v: self.engine.update_settings(fps=v))
        add(self.exposure, "Exposure")
        add(self.gain, "Gain")
        add(self.fps, "Requested FPS")
        self.noise_reduction = SliderSpin(0, 95, 1, 0, s.noise_reduction, " %")
        self.noise_reduction.setToolTip(
            "Blend each camera frame with the previous ones to remove sensor "
            "noise in every view; speckle is kept. Higher values smooth more "
            "but trail motion and weaken Previous-view fringes. 0 = off.")
        self.noise_reduction.valueChanged.connect(
            lambda v: self.engine.update_settings(noise_reduction=v))
        add(self.noise_reduction, "Noise reduction")
        self.camera_label = QLabel()
        note(self.camera_label)

        heading("ESPI measurement")
        self.average = SliderSpin(1, 60, 1, 0, s.average)
        self.average.setToolTip("Frames averaged to reduce speckle noise; 1 turns averaging off")
        self.average.valueChanged.connect(
            lambda v: self.engine.update_settings(average=int(v)))
        add(self.average, "Average N")
        self.rolling = QCheckBox("Rolling average (updates every frame)")
        self.rolling.setToolTip("Off: one image per N frames. On: a running average "
                                "that keeps the camera's frame rate.")
        self.rolling.setChecked(s.rolling)
        self.rolling.toggled.connect(lambda on: self.engine.update_settings(rolling=on))
        add(self.rolling)
        self.ref_refresh = SliderSpin(0, 10, 0.1, 1, s.ref_refresh_s, " s")
        self.ref_refresh.setToolTip(
            "Reference view: recapture the reference this often so slow drift "
            "(air, heat) cancels out. 0 keeps one reference until you press F.")
        self.ref_refresh.valueChanged.connect(
            lambda v: self.engine.update_settings(ref_refresh_s=v))
        add(self.ref_refresh, "Ref. refresh")

        heading("Fringe enhancement")
        self.smoothing_combo = QComboBox()
        self.smoothing_combo.addItems(SMOOTHING_METHODS)
        self.smoothing_combo.setCurrentText(s.smoothing_method)
        self.smoothing_combo.setToolTip("Gaussian: any width. Median keeps fringe "
                                        "edges sharper and is capped at 5 px.")
        self.smoothing_combo.currentTextChanged.connect(
            lambda name: self.engine.update_settings(smoothing_method=name))
        self.smoothing = SliderSpin(0, 31, 1, 0, s.smoothing, " px")
        self.smoothing.setToolTip("Blur width that removes pixel-scale speckle; "
                                  "keep it well below the fringe spacing. 0 = off.")
        self.smoothing.valueChanged.connect(
            lambda v: self.engine.update_settings(smoothing=int(v)))
        add(self.smoothing_combo, "Speckle filter")
        add(self.smoothing, "Filter width")
        self.scale_range = QCheckBox("Stretch to 0–255")
        self.scale_range.setChecked(s.scale_range)
        self.scale_range.toggled.connect(
            lambda on: self.engine.update_settings(scale_range=on))
        self.invert = QCheckBox("Invert")
        self.invert.setToolTip("Swap bright and dark fringes")
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
        self.clip.setToolTip("Stretch: let this share of the darkest and brightest "
                             "pixels saturate so stray speckles don't flatten the "
                             "contrast. 0 = exact min/max.")
        self.clip.valueChanged.connect(
            lambda v: self.engine.update_settings(clip_percent=v))
        add(self.clip, "Stretch clip")

        heading("Display")
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(PALETTES)
        self.palette_combo.setCurrentText(s.palette)
        self.palette_combo.currentTextChanged.connect(
            lambda name: self.engine.update_settings(palette=name))
        self.gamma = SliderSpin(0.1, 3.0, 0.1, 1, s.gamma)
        self.gamma.valueChanged.connect(lambda v: self.engine.update_settings(gamma=v))
        self.clahe = QCheckBox("CLAHE (color palettes only)")
        self.clahe.setChecked(s.clahe)
        self.clahe.toggled.connect(lambda on: self.engine.update_settings(clahe=on))
        add(self.palette_combo, "Palette")
        add(self.gamma, "Gamma")
        add(self.clahe)

        heading("Saved frame details")
        self.frame_title = QLineEdit()
        self.frame_title.setPlaceholderText("Title (optional)")
        self.frame_notes = QPlainTextEdit()
        self.frame_notes.setPlaceholderText("Notes (optional)")
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

    def _build_log(self):
        """Create the log dock.

        Args: None.
        Returns: None.
        """
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        font = self.log_view.font()
        font.setStyleHint(font.StyleHint.Monospace)
        font.setFamily("Menlo" if sys.platform == "darwin" else "Consolas")
        self.log_view.setFont(font)
        self.log_dock = QDockWidget("Log", self)
        self.log_dock.setObjectName("log")
        self.log_dock.setWidget(self.log_view)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.resizeDocks([self.log_dock], [120], Qt.Vertical)

    def _build_status_bar(self):
        """Create the status-bar readouts.

        Args: None.
        Returns: None.
        """
        bar = self.statusBar()
        self.hint_label = QLabel()  # left side; elides first when space runs out
        self.hint_label.setMinimumWidth(0)
        self.hint_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        bar.addWidget(self.hint_label, 1)
        # Every readout has a fixed width, so no value can move its neighbours.
        self.source_badge = FixedLabel("  DUMMY MODE  ", "  BFS-U3-200S6M-C  ")
        self.record_label = FixedLabel("Recording: finishing file", "Recording: 999.9 fps")
        self.range_status = FixedLabel("Input: 255–255    Scaled: 255–255",
                                       "Raw Mono8: no brightness scaling")
        self.camera_fps_status = FixedLabel(  # sensor rate, then what the app keeps up with
            "Camera: 999.9 fps (simulated)")
        self.camera_fps_status.setToolTip(
            "Camera: the camera's resulting acquisition rate (AcquisitionResultingFrameRate)\n"
            "Processed: frames the app received and processed\n"
            "Displayed: frames drawn on screen")
        self.speed_status = FixedLabel(
            "Processed: 999.9 fps    Work: 99.9 ms/frame    Displayed: 999.9 fps")
        self.pixel_status = FixedLabel("x 3088, y 2064: 255")
        # The two update buttons are never shown together, so they share one
        # fixed-width slot that stays put whichever is visible.
        self.update_button = QPushButton()  # shown when a newer build exists
        self.update_button.setVisible(False)
        self.update_button.setStyleSheet("background:#1d4f91;color:white;font-weight:bold;")
        self.update_button.clicked.connect(self._update_clicked)
        self.check_button = QPushButton("Check for updates")  # until a newer build is found
        self.check_button.setFlat(True)
        self.check_button.setToolTip("Check GitHub for a newer build of ESPI live")
        self.check_button.clicked.connect(self._check_updates_now)
        bold = self.update_button.font()
        bold.setBold(True)
        metrics = QFontMetrics(bold)
        slot_width = max(metrics.horizontalAdvance(text) for text in (
            "Update available: build 9999 (abcdef0)", "Downloading update… 100%",
            "Restart to update", "Check for updates", "Checking for updates…")) + 24
        self.update_slot = QWidget()
        slot = QGridLayout(self.update_slot)
        slot.setContentsMargins(0, 0, 0, 0)
        for button in (self.update_button, self.check_button):
            button.setFixedWidth(slot_width)
            slot.addWidget(button, 0, 0)  # same cell: they overlap
        self.update_slot.setFixedWidth(slot_width)
        bar.addPermanentWidget(self.update_slot)
        self.feature_button = QPushButton("Request feature")
        self.feature_button.setFlat(True)
        self.feature_button.setToolTip("Send a feature request to the developers (GitHub issue)")
        self.feature_button.clicked.connect(lambda: self._request_feature())
        bar.addPermanentWidget(self.feature_button)
        for widget in (self.source_badge, self.record_label, self.range_status,
                       self.camera_fps_status, self.speed_status, self.pixel_status):
            bar.addPermanentWidget(widget)

    def _build_menus(self):
        """Create the View menu (docks and zoom).

        Args: None.
        Returns: None.
        """
        menu = self.menuBar().addMenu("&View")
        menu.addAction(self.controls_dock.toggleViewAction())
        menu.addAction(self.log_dock.toggleViewAction())
        menu.addAction(self.actions_bar.toggleViewAction())
        menu.addSeparator()
        for text, key, handler in (("Fit to window", "0", self.view.fit),
                                   ("Actual size", "1", self.view.actual_size)):
            action = QAction(text, self)
            action.setShortcut(QKeySequence(key))
            action.setShortcutContext(Qt.WidgetShortcut)  # keys go via handle_key
            action.triggered.connect(handler)
            menu.addAction(action)
        menu.addSeparator()
        hint = menu.addAction("Wheel to zoom, drag to pan, Q or Esc to quit")
        hint.setEnabled(False)  # a reminder, not a command
        self.view.setToolTip("Wheel: zoom · drag: pan · 0: fit · 1: actual size\n"
                             "S save · F set reference · R reset last · "
                             "A reset average · Q/Esc quit")

        help_menu = self.menuBar().addMenu("&Help")
        check = help_menu.addAction("Check for updates…")
        check.triggered.connect(self._check_updates_now)
        request = help_menu.addAction("Request a feature…")
        request.triggered.connect(lambda: self._request_feature())
        about = help_menu.addAction("About ESPI live")
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
        self.camera_combo.addItems([channel.label for channel in self.channels])
        self.camera_combo.blockSignals(False)
        for widget in self.camera_checks + self.camera_states:
            self.camera_rows.removeWidget(widget)
            widget.deleteLater()
        self.camera_checks, self.camera_states = [], []
        for row, channel in enumerate(self.channels):
            check = QCheckBox(channel.info.serial)
            check.setToolTip(f"{channel.label}: acquire, process and (when Record is "
                             "on) record this camera even while another one is "
                             "displayed. The displayed camera always processes.")
            check.toggled.connect(
                lambda on, channel=channel: self._set_processing(channel, on))
            state = QLabel()
            state.setStyleSheet("color: palette(placeholder-text);")
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
        self.view.clear()
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
                   (self.record_button, s.record))
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
        Returns: Short text such as "displayed · 29.8 fps · REC", or "off".
        """
        if not channel.active:
            return "off"
        shown = "displayed · " if channel is self.current else ""
        recording = " · REC" if channel.engine.settings().record else ""
        return f"{shown}{channel.engine.status()['processed_fps']:.1f} fps{recording}"

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
        for widget in (self.exposure, self.gain, self.fps):
            widget.setToolTip("Simulated in dummy mode" if demo else "")

    def handle_key(self, key):
        """Act on a single-key shortcut.

        Args: key is a Qt.Key value.
        Returns: True if the key was a shortcut.
        """
        actions = {
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

    def save_frame(self):
        """Ask the engine to save the displayed frame with the title and notes.

        Args: None.
        Returns: None.
        """
        self.engine.request("save", title=self.frame_title.text(),
                            notes=self.frame_notes.toPlainText())

    def _show_zoom(self, zoom):
        """Sync the zoom slider and percentage with the view.

        Args: zoom is the view's zoom relative to fit-to-window.
        Returns: None.
        """
        self.zoom_slider.blockSignals(True)  # don't feed the value back to the view
        self.zoom_slider.setValue(round(ZOOM_TICKS * math.log(zoom) / math.log(MAX_ZOOM)))
        self.zoom_slider.blockSignals(False)
        self.zoom_label.setText(f"{100 * zoom:.0f}%")

    def _set_hover(self, x, y):
        """Remember the image pixel under the cursor for the readout.

        Args: x and y are image coordinates, or -1 when outside.
        Returns: None.
        """
        self.hover = (x, y)

    def _poll_frame(self):
        """Draw the newest engine frame if it has not been drawn yet.

        Args: None.
        Returns: None.
        """
        packet = self.engine.latest_packet()
        if packet is None or packet.sequence == self.drawn_sequence:
            return
        self.drawn_sequence = packet.sequence
        key = (packet.palette, round(packet.gamma, 4))
        if key != getattr(self, "table_key", None):
            self.table = palette_rgb_table(packet.palette, packet.gamma)
            self.table_key = key
        self.view.show_packet(packet, self.table, key)
        self.drawn_frames += 1

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
        self.speed_status.setText(
            f"{status['processing']}    Displayed: {self.display_fps:.1f} fps")
        packet = self.engine.latest_packet()
        x, y = self.hover
        if packet is not None and 0 <= y < packet.image.shape[0] and \
                0 <= x < packet.image.shape[1]:
            self.pixel_status.setText(f"x {x}, y {y}: {int(packet.image[y, x])}")
        else:
            self.pixel_status.setText("")
        for channel, state in zip(self.channels, self.camera_states):
            state.setText(self._camera_state(channel))
        self._handle_events()
        self._handle_update_events()
        self._handle_feedback_events()

    # ----- feature requests --------------------------------------------------

    def _request_feature(self, previous=None):
        """Ask for a feature request and send it in the background.

        Without a feedback token the request opens as a prefilled GitHub page.

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
        if not feedback.feedback_token():
            self._open_request_in_browser(request)
            return
        self.feedback.submit(request)
        self.feature_button.setEnabled(False)
        self.feature_button.setText("Sending request…")
        self.log.write(f"Sending feature request: {request.title.strip()}")

    def _open_request_in_browser(self, request):
        """Open GitHub's new-issue page prefilled with the request.

        Args: request is the FeatureRequest.
        Returns: None.
        """
        if not QDesktopServices.openUrl(QUrl(feedback.browser_url(request))):
            QMessageBox.warning(self, "Request a feature",
                                "Could not open a web browser.")

    def _handle_feedback_events(self):
        """Report feature requests that were sent or failed.

        Args: None.
        Returns: None.
        """
        while True:
            try:
                kind, (request, payload) = self.feedback.events.get_nowait()
            except Exception:
                return
            self.feature_button.setEnabled(True)
            self.feature_button.setText("Request feature")
            if kind == "sent":
                self.log.write(f"Feature request filed: {payload}")
                extra = ("<br><br>GitHub Copilot will add a structured version to "
                         "the issue in a minute or two." if request.enrich else "")
                box = QMessageBox(QMessageBox.Information, "Request sent",
                                  f"Thank you. Your request is filed as "
                                  f'<a href="{payload}">{payload}</a>.{extra}', parent=self)
                box.setTextFormat(Qt.RichText)
                box.setTextInteractionFlags(Qt.TextBrowserInteraction)
                box.exec()
            else:
                self.log.write(f"Feature request failed: {payload}")
                box = QMessageBox(QMessageBox.Warning, "Request not sent",
                                  f"The request could not be sent: {payload}.", parent=self)
                retry = box.addButton("Edit and retry", QMessageBox.AcceptRole)
                browser = box.addButton("Open in browser", QMessageBox.ActionRole)
                box.addButton(QMessageBox.Cancel)
                box.exec()
                if box.clickedButton() is retry:
                    self._request_feature(request)
                elif box.clickedButton() is browser:
                    self._open_request_in_browser(request)

    # ----- updates -----------------------------------------------------------

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
        if self.updates.busy and self.update_info is not None and not self.pending_update:
            done, total = self.updates.progress
            percent = f" {100 * done / total:.0f}%" if total else ""
            self.update_button.setText(f"Downloading update…{percent}")
        while True:
            try:
                kind, payload = self.updates.events.get_nowait()
            except Exception:
                break
            manual, self.update_manual = self.update_manual, False
            if kind == "checked" and payload is None:
                self.log.write(f"Up to date ({updater.describe_build()})")
                if manual:
                    QMessageBox.information(self, "Updates", "ESPI live is up to date.")
            elif kind == "checked":
                self.update_info = payload
                self.log.write(f"Update available: {payload.label()}")
                self.update_button.setText(f"Update available: {payload.label()}")
                self.update_button.setToolTip("Download the new build")
                self.update_button.setVisible(True)
                if manual:
                    self._update_clicked()
            elif kind == "check_failed":
                self.log.write(f"Update check failed: {payload}")
                if manual:
                    QMessageBox.warning(self, "Updates", f"Update check failed: {payload}")
            elif kind == "downloaded":
                self.pending_update = payload
                self.log.write("Update downloaded; it installs when ESPI live restarts")
                self.update_button.setText("Restart to update")
                self.update_button.setToolTip("Restart now, or the update installs on exit")
                self.update_button.setEnabled(True)
                self._offer_restart()
            elif kind == "download_failed":
                self.log.write(f"Update download failed: {payload}")
                self.update_button.setText(f"Update available: {self.update_info.label()}")
                self.update_button.setEnabled(True)
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
                except Exception:
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
        elif kind == "camera_limits":
            channel.limits = payload
            if displayed:
                self._apply_limits(payload)
        elif kind == "record_off" and displayed:
            self.record_button.blockSignals(True)
            self.record_button.setChecked(False)
            self.record_button.blockSignals(False)
        elif kind == "switched_live":
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
            QMessageBox.critical(self, "ESPI stopped", f"{name}{payload}")
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
        QApplication.instance().removeEventFilter(self.shortcut_filter)
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
        log.write(f"Update could not be installed: {exc}")


def run(force_demo=False, check_updates=True, demo_cameras=1):
    """Start the Qt front end.

    Args: force_demo skips camera detection and starts synthetic video;
        check_updates looks for a newer prebuilt binary at startup;
        demo_cameras is how many synthetic cameras demo mode simulates.
    Returns: The process exit code.
    """
    request_gl_format()
    QApplication.setApplicationName("ESPI live")
    QApplication.setOrganizationName("ESPI")
    app = QApplication.instance() or QApplication(sys.argv)
    log = AppLog()
    path = log.start()
    cleanup = updater.StartupCleanup(log.write)
    window = None
    channels = []
    try:
        if path:
            log.write(f"Log file: {path}")
        cleanup.start()
        if not force_demo:
            try:
                channels = open_camera_channels(log)
            except Exception as exc:
                log.write(f"Camera startup unavailable: {exc}")
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
        log.write(f"Display: GPU ({description})" if use_gl else
                  f"Display: CPU fallback ({description})")
        window = MainWindow(channels, log, use_gl, check_updates)
        window.show()
        for channel in channels:
            if channel.active:
                channel.engine.start()
        return app.exec()
    finally:
        if window is None:
            for channel in channels:
                if channel.active:
                    channel.engine.stop()
                    channel.engine.wait(8.0)
        cleanup.stop()
        finish_update(window, log)
        log.close()
