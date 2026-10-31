"""Camera-free checks for the Qt front end (offscreen platform, demo engine).

Run from the project root:
    PYTHONPATH=src QT_QPA_PLATFORM=offscreen python -m unittest tests.test_qt_app

No camera, network, real dialog or real settings store is touched: the engine
uses its demo source, the updater worker is replaced by a fake, QSettings is
replaced by a dictionary and every QMessageBox call is patched.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import logging
import queue
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QHelpEvent, QKeyEvent, QSurfaceFormat
from PySide6.QtWidgets import (QAbstractButton, QAbstractSlider, QAbstractSpinBox, QApplication,
                               QComboBox, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QStyle,
                               QTextEdit, QWidget)

from espi_live import camera, camera_reports, capture, feedback, paths, qt_app, sources, tips, updater
from fake_capture import FakeCaptureFactory, FakeEnumerator, interlaced_frame, usb_live2, webcam
from fake_pyspin import FakeCamera, FakeSystem, make_pyspin
from espi_live.camera import CameraInfo
from espi_live.camera_models import CameraIdentity
from espi_live.channels import Channel, demo_channels
from espi_live.engine import Engine, FramePacket
from list_log import ListLog

APP = QApplication.instance() or QApplication(sys.argv)

FINGERPRINT_UI = "ESPI-ERR-v1-" + "b" * 12


class FakeLog(ListLog):
    """In-memory stand-in for AppLog."""

    level = logging.INFO
    path = None
    crash_path = None

    def __init__(self):
        """Create an empty log.

        Args: None.
        Returns: None.
        """
        super().__init__()
        self.pending = []
        self.ticks = 0

    def _add(self, level, message, exc_info=None):
        """Record one message for assertions and for drain().

        Args: level, message and exc_info as ListLog takes them.
        Returns: None.
        """
        super()._add(level, message, exc_info)
        self.pending.append(message + "\n")

    def drain(self):
        """Return and clear the text written since the last call.

        Args: None.
        Returns: The pending text, possibly empty.
        """
        text, self.pending = "".join(self.pending), []
        return text

    def recent_lines(self):
        """Return the newest lines, as AppLog does for error reports.

        Args: None.
        Returns: The last 40 lines.
        """
        return self.lines[-40:]

    def uptime(self):
        """Pretend the run started a minute ago.

        Args: None.
        Returns: 60.0.
        """
        return 60.0

    def watchdog_tick(self):
        """Count UI ticks instead of arming faulthandler.

        Args: None.
        Returns: None.
        """
        self.ticks += 1

    def start(self):
        """Pretend to open a log file.

        Args: None.
        Returns: A fake path.
        """
        return "fake.log"

    def close(self):
        """Pretend to close the log.

        Args: None.
        Returns: None.
        """
        self.lines.append("closed")


class FakeSettings:
    """Dictionary-backed QSettings replacement so the user's store is untouched."""

    store = {}

    def __init__(self, *args):
        """Accept and ignore the organization and application names.

        Args: args are ignored.
        Returns: None.
        """

    def value(self, key, default=None, kind=None):
        """Read a stored value.

        Args: key is the setting name; default is returned when unset; kind
            is QSettings' optional type argument (ignored).
        Returns: The stored value or default.
        """
        return self.store.get(key, default)

    def setValue(self, key, value):
        """Store a value.

        Args: key is the setting name; value is what to keep.
        Returns: None.
        """
        self.store[key] = value


class FakeUpdateWorker:
    """Replacement for updater.UpdateWorker that never starts a thread."""

    def __init__(self):
        """Create an idle fake worker.

        Args: None.
        Returns: None.
        """
        self.events = queue.Queue()
        self.progress = (0, 0)
        self.busy = False
        self.downloading = False
        self.checks = 0
        self.downloads = []
        self.download_result = True

    def check(self):
        """Count a requested check.

        Args: None.
        Returns: True.
        """
        self.checks += 1
        return True

    def download(self, info):
        """Record a requested download.

        Args: info is the UpdateInfo to fetch.
        Returns: The configured download_result.
        """
        self.downloads.append(info)
        return self.download_result


def make_dialogs(answer=None):
    """Build a MagicMock standing in for the QMessageBox class.

    Args: answer is the value question() returns (default: No).
    Returns: A MagicMock whose Yes/No constants match QMessageBox.
    """
    dialogs = mock.MagicMock()
    dialogs.Yes = qt_app.QMessageBox.Yes
    dialogs.No = qt_app.QMessageBox.No
    dialogs.question.return_value = dialogs.No if answer is None else answer
    return dialogs


def make_packet(sequence=1, width=64, height=48, palette="Gray", gamma=1.0):
    """Build a FramePacket with a horizontal ramp image.

    Args: sequence, width, height and palette/gamma describe the frame.
    Returns: A FramePacket.
    """
    image = np.tile(np.arange(width, dtype=np.uint8) * 3, (height, 1))
    return FramePacket(sequence, image, palette, gamma, "Raw", True)


def wait_for(predicate, timeout=5.0):
    """Poll a condition until it is true or a timeout passes.

    Args: predicate is a zero-argument callable; timeout is in seconds.
    Returns: The final value of predicate().
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.01)
    return predicate()


def demo_window(engine, log, use_gl=False, check_updates=False):
    """Build a MainWindow around one demo engine.

    Args: engine is a demo Engine (not started); log is the FakeLog; use_gl
        and check_updates are passed to MainWindow.
    Returns: The MainWindow.
    """
    channel = Channel(log, CameraInfo("DEMO1", "Synthetic camera"), demo=True,
                      engine=engine)
    return qt_app.MainWindow([channel], log, use_gl, check_updates)


def make_info(size=5_000_000):
    """Build an UpdateInfo for update-flow checks.

    Args: size is the download size in bytes.
    Returns: An updater.UpdateInfo.
    """
    return updater.UpdateInfo(build=99, commit="abcdef1234567", asset_name="a.zip",
                              asset_url="https://example.invalid/a.zip",
                              sha256="0" * 64, size=size)


class OpenGlProbeTests(unittest.TestCase):
    """request_gl_format and opengl_available."""

    def test_request_gl_format_sets_core_33_default(self):
        """The default surface format asks for OpenGL 3.3 core with vsync.

        Args: None.
        Returns: None.
        """
        previous = QSurfaceFormat.defaultFormat()
        self.addCleanup(QSurfaceFormat.setDefaultFormat, previous)
        qt_app.request_gl_format()
        fmt = QSurfaceFormat.defaultFormat()
        self.assertEqual((fmt.majorVersion(), fmt.minorVersion()), (3, 3))
        self.assertEqual(fmt.profile(), QSurfaceFormat.CoreProfile)
        self.assertEqual(fmt.swapInterval(), 1)

    def probe(self, create=True, current=True, version=(3, 3), renderer="Mock GPU"):
        """Run opengl_available against a fake context.

        Args: create/current are the results of create()/makeCurrent();
            version is the context version; renderer is glGetString's result.
        Returns: (result, context) from opengl_available and the fake context.
        """
        context = mock.MagicMock()
        context.create.return_value = create
        context.makeCurrent.return_value = current
        context.format.return_value.majorVersion.return_value = version[0]
        context.format.return_value.minorVersion.return_value = version[1]
        context.functions.return_value.glGetString.return_value = renderer
        with mock.patch.object(qt_app, "QOpenGLContext", return_value=context), \
                mock.patch.object(qt_app, "QOffscreenSurface"):
            return qt_app.opengl_available(), context

    def test_reports_missing_context(self):
        """A context that cannot be created is reported as unavailable.

        Args: None.
        Returns: None.
        """
        result, _ = self.probe(create=False)
        self.assertEqual(result, (False, "no OpenGL context"))

    def test_reports_context_that_cannot_be_made_current(self):
        """A context that will not become current is unavailable.

        Args: None.
        Returns: None.
        """
        result, _ = self.probe(current=False)
        self.assertFalse(result[0])
        self.assertIn("current", result[1])

    def test_rejects_old_opengl_and_releases_context(self):
        """OpenGL below 3.3 is rejected and the context is released.

        Args: None.
        Returns: None.
        """
        result, context = self.probe(version=(2, 1))
        self.assertEqual(result, (False, "OpenGL 2.1 on Mock GPU (3.3 needed)"))
        context.doneCurrent.assert_called_once()

    def test_accepts_modern_opengl(self):
        """OpenGL 4.1 core is accepted with a description.

        Args: None.
        Returns: None.
        """
        result, _ = self.probe(version=(4, 1))
        self.assertEqual(result, (True, "OpenGL 4.1 on Mock GPU"))

    def test_unknown_renderer_name_is_substituted(self):
        """An empty renderer string is described as unknown.

        Args: None.
        Returns: None.
        """
        result, _ = self.probe(renderer="")
        self.assertEqual(result, (True, "OpenGL 3.3 on unknown renderer"))

    def test_real_probe_returns_a_pair_without_raising(self):
        """The unpatched probe returns (bool, str) even without OpenGL.

        Args: None.
        Returns: None.
        """
        ok, text = qt_app.opengl_available()
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(text, str)


class NavigationTests(unittest.TestCase):
    """Fit, zoom and pan shared by both views, driven through CpuImageView."""

    def setUp(self):
        """Create a 400x300 CPU view showing a 2000x1500 image.

        Args: None.
        Returns: None.
        """
        self.view = qt_app.CpuImageView()
        self.view.resize(400, 300)
        self.zooms = []
        self.hovers = []
        self.view.zoomChanged.connect(self.zooms.append)
        self.view.hovered.connect(lambda x, y: self.hovers.append((x, y)))

    def tearDown(self):
        """Delete the view.

        Args: None.
        Returns: None.
        """
        self.view.deleteLater()

    def load_image(self):
        """Give the view a large image so it must shrink to fit.

        Args: None.
        Returns: None.
        """
        self.view.show_packet(make_packet(width=2000, height=1500),
                              np.zeros((256, 3), np.uint8), None)
        self.zooms.clear()

    def test_geometry_helpers_return_none_without_image(self):
        """Rect and coordinate mapping are None before any frame arrives.

        Args: None.
        Returns: None.
        """
        self.assertIsNone(self.view.display_rect())
        self.assertIsNone(self.view.widget_to_image(QPointF(1, 1)))

    def test_zoom_and_actual_size_ignored_without_image(self):
        """Zoom requests before a frame do nothing and emit nothing.

        Args: None.
        Returns: None.
        """
        self.view.set_zoom(4.0)
        self.view.actual_size()
        self.assertEqual(self.view.zoom, 1.0)
        self.assertEqual(self.zooms, [])

    def test_fit_rect_centers_image_in_widget(self):
        """At zoom 1 the image is scaled to fit and centered.

        Args: None.
        Returns: None.
        """
        self.load_image()
        x, y, width, height, scale = self.view.display_rect()
        self.assertAlmostEqual(scale, 0.2)
        self.assertEqual((x, y, width, height), (0.0, 0.0, 400.0, 300.0))

    def test_widget_to_image_maps_center_to_image_center(self):
        """The widget center maps to the middle of the image.

        Args: None.
        Returns: None.
        """
        self.load_image()
        px, py = self.view.widget_to_image(QPointF(200, 150))
        self.assertAlmostEqual(px, 1000.0)
        self.assertAlmostEqual(py, 750.0)

    def test_set_zoom_keeps_anchor_point_fixed(self):
        """Zooming about the top-left corner keeps image point (0, 0) there.

        Args: None.
        Returns: None.
        """
        self.load_image()
        self.view.set_zoom(2.0, QPointF(0, 0))
        self.assertEqual(self.view.zoom, 2.0)
        self.assertAlmostEqual(self.view.center[0], 500.0)
        self.assertAlmostEqual(self.view.center[1], 375.0)
        self.assertEqual(self.zooms, [2.0])

    def test_set_zoom_defaults_to_widget_center_anchor(self):
        """Without an anchor the zoom stays centered on the image.

        Args: None.
        Returns: None.
        """
        self.load_image()
        self.view.set_zoom(3.0)
        self.assertAlmostEqual(self.view.center[0], 1000.0)
        self.assertAlmostEqual(self.view.center[1], 750.0)

    def test_set_zoom_clamps_to_limits_and_resets_center_at_fit(self):
        """Zoom is clamped to 1..MAX_ZOOM and zoom 1 recenters the image.

        Args: None.
        Returns: None.
        """
        self.load_image()
        self.view.set_zoom(10_000.0)
        self.assertEqual(self.view.zoom, qt_app.MAX_ZOOM)
        self.view.set_zoom(0.01)
        self.assertEqual(self.view.zoom, 1.0)
        self.assertIsNone(self.view.center)

    def test_set_zoom_clamps_center_inside_image(self):
        """Zooming about a point outside the image keeps the center inside it.

        Args: None.
        Returns: None.
        """
        self.load_image()
        self.view.set_zoom(2.0, QPointF(-5000, -5000))
        self.assertEqual(self.view.center, (0.0, 0.0))

    def test_fit_restores_whole_image(self):
        """fit() clears zoom and center and announces zoom 1.

        Args: None.
        Returns: None.
        """
        self.load_image()
        self.view.set_zoom(4.0, QPointF(10, 10))
        self.zooms.clear()
        self.view.fit()
        self.assertEqual((self.view.zoom, self.view.center), (1.0, None))
        self.assertEqual(self.zooms, [1.0])

    def test_actual_size_shows_one_image_pixel_per_screen_pixel(self):
        """1:1 zoom makes the display scale equal 1 / devicePixelRatio.

        Args: None.
        Returns: None.
        """
        self.load_image()
        self.view.actual_size()
        scale = self.view.display_rect()[4]
        self.assertAlmostEqual(scale, 1.0 / self.view.devicePixelRatioF())

    def test_show_packet_resets_zoom_only_when_size_changes(self):
        """A frame of the same size keeps the zoom; a new size resets it.

        Args: None.
        Returns: None.
        """
        self.load_image()
        self.view.set_zoom(4.0)
        self.view.show_packet(make_packet(width=2000, height=1500),
                              np.zeros((256, 3), np.uint8), None)
        self.assertEqual(self.view.zoom, 4.0)
        self.view.show_packet(make_packet(width=10, height=10),
                              np.zeros((256, 3), np.uint8), None)
        self.assertEqual(self.view.zoom, 1.0)

    def wheel(self, delta):
        """Build a stand-in wheel event.

        Args: delta is the vertical angle delta in eighths of a degree.
        Returns: A mock event positioned at the widget center.
        """
        event = mock.MagicMock()
        event.angleDelta.return_value.y.return_value = delta
        event.position.return_value = QPointF(200, 150)
        return event

    def test_wheel_up_zooms_in_by_one_step(self):
        """One wheel notch multiplies zoom by 1.25 and accepts the event.

        Args: None.
        Returns: None.
        """
        self.load_image()
        event = self.wheel(120)
        self.view.wheelEvent(event)
        self.assertAlmostEqual(self.view.zoom, 1.25)
        event.accept.assert_called_once()

    def test_wheel_without_delta_does_not_zoom(self):
        """A zero-delta wheel event leaves the zoom alone.

        Args: None.
        Returns: None.
        """
        self.load_image()
        event = self.wheel(0)
        self.view.wheelEvent(event)
        self.assertEqual(self.view.zoom, 1.0)
        event.accept.assert_called_once()

    def mouse(self, x, y, button=Qt.LeftButton):
        """Build a stand-in mouse event.

        Args: x and y are the widget position; button is the Qt button.
        Returns: A mock event.
        """
        event = mock.MagicMock()
        event.position.return_value = QPointF(x, y)
        event.button.return_value = button
        return event

    def test_press_starts_pan_only_when_zoomed_in(self):
        """The left button begins a drag only above zoom 1.

        Args: None.
        Returns: None.
        """
        self.load_image()
        self.view.mousePressEvent(self.mouse(10, 10))
        self.assertIsNone(self.view.drag_origin)
        self.view.set_zoom(2.0)
        self.view.mousePressEvent(self.mouse(10, 10, Qt.RightButton))
        self.assertIsNone(self.view.drag_origin)
        self.view.mousePressEvent(self.mouse(10, 10))
        self.assertEqual(self.view.drag_origin, QPointF(10, 10))

    def test_drag_pans_the_view_center(self):
        """Dragging right by 40 px moves the center left by 40/scale pixels.

        Args: None.
        Returns: None.
        """
        self.load_image()
        self.view.set_zoom(2.0)
        start = self.view.center
        self.view.mousePressEvent(self.mouse(100, 100))
        self.view.mouseMoveEvent(self.mouse(140, 100))
        self.assertAlmostEqual(self.view.center[0], start[0] - 40 / 0.4)
        self.assertAlmostEqual(self.view.center[1], start[1])

    def test_move_reports_hovered_pixel_and_release_ends_pan(self):
        """Moving reports the image pixel; release clears the drag origin.

        Args: None.
        Returns: None.
        """
        self.load_image()
        self.view.mouseMoveEvent(self.mouse(200, 150))
        self.assertEqual(self.hovers[-1], (1000, 750))
        self.view.drag_origin = QPointF(1, 1)
        self.view.mouseReleaseEvent(self.mouse(0, 0))
        self.assertIsNone(self.view.drag_origin)

    def test_move_without_image_reports_nothing(self):
        """Without an image no hover coordinates are emitted.

        Args: None.
        Returns: None.
        """
        self.view.mouseMoveEvent(self.mouse(5, 5))
        self.assertEqual(self.hovers, [])

    def test_leave_clears_hover_readout(self):
        """Leaving the widget emits (-1, -1).

        Args: None.
        Returns: None.
        """
        self.view.leaveEvent(mock.MagicMock())
        self.assertEqual(self.hovers, [(-1, -1)])


class CpuImageViewPaintTests(unittest.TestCase):
    """CpuImageView.paintEvent branches, rendered offscreen with grab()."""

    def setUp(self):
        """Create a 400x300 view with a color table.

        Args: None.
        Returns: None.
        """
        self.view = qt_app.CpuImageView()
        self.view.resize(400, 300)
        self.table = np.stack([np.arange(256, dtype=np.uint8)] * 3, axis=1)

    def tearDown(self):
        """Delete the view.

        Args: None.
        Returns: None.
        """
        self.view.deleteLater()

    def center_pixel(self):
        """Render the view and read its center pixel's red channel.

        Args: None.
        Returns: The red value 0-255.
        """
        image = self.view.grab().toImage()
        return image.pixelColor(image.width() // 2, image.height() // 2).red()

    def test_paints_background_without_a_frame(self):
        """With no frame the widget is just the darkened background.

        Args: None.
        Returns: None.
        """
        self.assertEqual(self.view.grab().size().width(), 400)

    def test_shrinks_large_image_through_color_table(self):
        """A large image is area-resized and mapped through the table.

        Args: None.
        Returns: None.
        """
        image = np.full((1500, 2000), 200, np.uint8)
        packet = FramePacket(1, image, "Gray", 1.0, "Raw", True)
        self.view.show_packet(packet, self.table, None)
        self.assertEqual(self.center_pixel(), 200)

    def test_zoomed_in_small_image_uses_nearest_neighbor(self):
        """A small image enlarged on screen is drawn with the table colors.

        Args: None.
        Returns: None.
        """
        image = np.full((20, 20), 90, np.uint8)
        packet = FramePacket(1, image, "Gray", 1.0, "Raw", True)
        self.view.show_packet(packet, self.table, None)
        self.view.set_zoom(3.0)
        self.assertEqual(self.center_pixel(), 90)

    def test_empty_visible_crop_draws_nothing_extra(self):
        """A view center far outside the image leaves only the background.

        Args: None.
        Returns: None.
        """
        image = np.full((100, 100), 255, np.uint8)
        packet = FramePacket(1, image, "Gray", 1.0, "Raw", True)
        self.view.show_packet(packet, self.table, None)
        self.view.zoom, self.view.center = 2.0, (-100000.0, -100000.0)
        self.assertLess(self.center_pixel(), 255)


class ViewClearTests(unittest.TestCase):
    """clear() on both views, used when the displayed camera changes."""

    def test_both_views_forget_the_image(self):
        """After clear() neither view has an image to draw.

        Args: None.
        Returns: None.
        """
        for view in (qt_app.CpuImageView(), qt_app.GLImageView()):
            view.show_packet(make_packet(), np.zeros((256, 3), np.uint8), None)
            view.clear()
            self.assertIsNone(view.image_size)
            self.assertIsNone(view.display_rect())
            view.deleteLater()


class GLImageViewTests(unittest.TestCase):
    """GLImageView driven with a mocked GL context (none exists offscreen)."""

    def setUp(self):
        """Create a view whose context() returns a recording fake.

        Args: None.
        Returns: None.
        """
        self.view = qt_app.GLImageView()
        self.view.resize(400, 300)
        self.gl = mock.MagicMock()
        self.context = mock.MagicMock()
        self.context.functions.return_value = self.gl
        self.view.context = mock.MagicMock(return_value=self.context)
        self.view.makeCurrent = mock.MagicMock()
        self.view.doneCurrent = mock.MagicMock()
        self.table = np.zeros((256, 3), np.uint8)

    def tearDown(self):
        """Delete the view.

        Args: None.
        Returns: None.
        """
        self.view.deleteLater()

    def init_gl(self, link=True):
        """Run initializeGL with patched shader and VAO classes.

        Args: link is whether the shader program links.
        Returns: The fake QOpenGLShaderProgram instance.
        """
        program = mock.MagicMock()
        program.addShaderFromSourceCode.return_value = True
        program.link.return_value = link
        program.log.return_value = "boom"

        def fill(count, names):
            """Fill the texture-name array like glGenTextures.

            Args: count is the number of names; names is the output array.
            Returns: None.
            """
            names[:] = [7, 9]

        self.gl.glGenTextures.side_effect = fill
        with mock.patch.object(qt_app, "QOpenGLShaderProgram", return_value=program), \
                mock.patch.object(qt_app, "QOpenGLVertexArrayObject"):
            self.view.initializeGL()
        return program

    def test_initialize_builds_program_and_two_textures(self):
        """initializeGL compiles shaders, creates textures and hooks cleanup.

        Args: None.
        Returns: None.
        """
        program = self.init_gl()
        self.assertIs(self.view.program, program)
        self.assertEqual((self.view.image_texture, self.view.lut_texture), (7, 9))
        self.assertEqual(program.addShaderFromSourceCode.call_count, 2)
        self.assertEqual(self.gl.glTexParameteri.call_count, 8)
        self.context.aboutToBeDestroyed.connect.assert_called_once_with(
            self.view._release_gl)

    def test_initialize_raises_when_shader_does_not_link(self):
        """A shader link failure raises RuntimeError with the GL log.

        Args: None.
        Returns: None.
        """
        with self.assertRaisesRegex(RuntimeError, "Shader build failed: boom"):
            self.init_gl(link=False)

    def test_release_deletes_both_textures(self):
        """Cleanup deletes the image and LUT textures with the context current.

        Args: None.
        Returns: None.
        """
        self.view.image_texture, self.view.lut_texture = 7, 9
        self.view._release_gl()
        count, names = self.gl.glDeleteTextures.call_args.args
        self.assertEqual((count, list(names)), (2, [7, 9]))
        self.view.makeCurrent.assert_called_once()
        self.view.doneCurrent.assert_called_once()

    def test_show_packet_queues_frame_and_sets_size(self):
        """show_packet stores the pending frame and the image size.

        Args: None.
        Returns: None.
        """
        packet = make_packet(width=64, height=48)
        self.view.show_packet(packet, self.table, "k")
        self.assertEqual(self.view.image_size, (64, 48))
        self.assertEqual(self.view.pending[2], "k")

    def prepare_upload(self):
        """Give the view texture ids and clear upload state.

        Args: None.
        Returns: None.
        """
        self.view.image_texture, self.view.lut_texture = 7, 9
        self.view.texture_shape = None
        self.view.table_key = None

    def test_first_upload_allocates_texture_and_table(self):
        """The first frame allocates the image texture and uploads the LUT.

        Args: None.
        Returns: None.
        """
        self.prepare_upload()
        self.view.pending = (make_packet(), self.table, "k")
        self.view._upload(self.gl)
        self.assertEqual(self.gl.glTexImage2D.call_count, 2)
        self.gl.glTexSubImage2D.assert_not_called()
        self.assertEqual(self.view.texture_shape, (48, 64))
        self.assertEqual(self.view.table_key, "k")
        self.assertIsNone(self.view.pending)

    def test_repeat_upload_updates_in_place_and_skips_same_table(self):
        """Same-size frames use glTexSubImage2D and an unchanged LUT is kept.

        Args: None.
        Returns: None.
        """
        self.prepare_upload()
        self.view.pending = (make_packet(), self.table, "k")
        self.view._upload(self.gl)
        self.gl.reset_mock()
        self.view.pending = (make_packet(2), self.table, "k")
        self.view._upload(self.gl)
        self.gl.glTexImage2D.assert_not_called()
        self.gl.glTexSubImage2D.assert_called_once()

    def paint_ready_view(self):
        """Initialize GL and queue a frame ready to paint.

        Args: None.
        Returns: None.
        """
        self.init_gl()
        self.view.show_packet(make_packet(), self.table, "k")

    def test_paint_uploads_and_draws_a_quad(self):
        """paintGL uploads the pending frame and draws a triangle strip.

        Args: None.
        Returns: None.
        """
        self.paint_ready_view()
        self.view.paintGL()
        self.gl.glDrawArrays.assert_called_once_with(qt_app.GL_TRIANGLE_STRIP, 0, 4)
        self.view.program.bind.assert_called_once()
        self.view.program.release.assert_called_once()

    def test_paint_without_image_only_clears(self):
        """With no image size paintGL clears the background and returns.

        Args: None.
        Returns: None.
        """
        self.init_gl()
        self.view.paintGL()
        self.gl.glClear.assert_called_once()
        self.gl.glDrawArrays.assert_not_called()

    def test_paint_without_texture_does_not_draw(self):
        """A known image size but no uploaded texture draws nothing.

        Args: None.
        Returns: None.
        """
        self.init_gl()
        self.view.image_size = (64, 48)
        self.view.paintGL()
        self.gl.glDrawArrays.assert_not_called()


class SliderSpinTests(unittest.TestCase):
    """SliderSpin keeps its slider and spin box in step."""

    def setUp(self):
        """Create a 0..10 control in steps of 0.5.

        Args: None.
        Returns: None.
        """
        self.control = qt_app.SliderSpin(0, 10, 0.5, 1, 2.0, " x")
        self.emitted = []
        self.control.valueChanged.connect(self.emitted.append)

    def tearDown(self):
        """Delete the control.

        Args: None.
        Returns: None.
        """
        self.control.deleteLater()

    def test_initial_value_is_shown_on_both_widgets(self):
        """The starting value appears on the spin box and the slider.

        Args: None.
        Returns: None.
        """
        self.assertEqual(self.control.value(), 2.0)
        self.assertEqual(self.control.slider.value(), 4)

    def test_slider_move_updates_spin_and_announces(self):
        """Moving the slider sets the spin box and emits valueChanged.

        Args: None.
        Returns: None.
        """
        self.control.slider.setValue(8)
        self.assertEqual(self.control.value(), 4.0)
        self.assertEqual(self.emitted, [4.0])

    def test_spin_change_updates_slider_and_announces(self):
        """Typing a value moves the slider and emits valueChanged.

        Args: None.
        Returns: None.
        """
        self.control.spin.setValue(7.5)
        self.assertEqual(self.control.slider.value(), 15)
        self.assertEqual(self.emitted, [7.5])

    def test_set_value_is_silent_and_clamped(self):
        """set_value emits nothing and clamps to the range.

        Args: None.
        Returns: None.
        """
        self.control.set_value(99)
        self.assertEqual(self.control.value(), 10.0)
        self.assertEqual(self.control.slider.value(), 20)
        self.assertEqual(self.emitted, [])

    def test_set_range_clamps_current_value(self):
        """Shrinking the range pulls the value inside it.

        Args: None.
        Returns: None.
        """
        self.control.set_range(0, 1)
        self.assertEqual(self.control.value(), 1.0)
        self.assertEqual(self.control.slider.maximum(), 2)

    def test_tooltip_tracks_the_value_and_an_optional_note(self):
        """The tip names the control, the live value and a situation line.

        Args: None.
        Returns: None.
        """
        self.control.set_description("Scale factor. A typical value is 2.")
        self.assertIn("Now: 2.0 x", self.control.toolTip())
        self.assertEqual(self.control.slider.toolTip(), self.control.toolTip())
        self.assertEqual(self.control.spin.toolTip(), self.control.toolTip())
        self.control.set_note("Simulated in dummy mode")
        self.assertIn("Simulated in dummy mode", self.control.toolTip())
        self.control.slider.setValue(8)
        self.assertIn("Now: 4.0 x", self.control.toolTip())
        self.control.set_note("")
        self.assertNotIn("Simulated", self.control.toolTip())

    def test_integer_value_is_shown_without_decimals(self):
        """A whole-number control does not show a fractional part.

        Args: None.
        Returns: None.
        """
        control = qt_app.SliderSpin(0, 10, 1, 0, 3, " px")
        self.addCleanup(control.deleteLater)
        control.set_description("Width.")
        self.assertIn("Now: 3 px", control.toolTip())


class WindowTestCase(unittest.TestCase):
    """Base class: a MainWindow on a demo engine with every side effect faked."""

    use_gl = False

    def setUp(self):
        """Patch settings, dialogs and the updater, then build the window.

        Args: None.
        Returns: None.
        """
        FakeSettings.store = {}
        self.dialogs = make_dialogs()
        for patcher in (
                mock.patch.object(qt_app, "QSettings", FakeSettings),
                mock.patch.object(qt_app, "QMessageBox", self.dialogs),
                mock.patch.object(qt_app.updater, "UpdateWorker", FakeUpdateWorker),
                mock.patch.object(qt_app.updater, "unavailable_reason",
                                  return_value="disabled in tests")):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.log = FakeLog()
        self.engine = Engine(self.log)
        self.engine.use_demo()
        self.window = demo_window(self.engine, self.log, self.use_gl)
        self.addCleanup(self.cleanup_window)

    def cleanup_window(self):
        """Stop timers, close the window and stop the engine.

        Args: None.
        Returns: None.
        """
        self.window.frame_timer.stop()
        self.window.status_timer.stop()
        self.window.tip_timer.stop()
        self.window.close()
        self.engine.stop()
        self.engine.wait(5.0)
        self.window.deleteLater()

    def queued(self):
        """Drain the engine's pending one-shot actions.

        Args: None.
        Returns: A list of (action, payload) tuples.
        """
        items = []
        while True:
            try:
                items.append(self.engine._actions.get_nowait())
            except queue.Empty:
                return items


class MainWindowBuildTests(WindowTestCase):
    """Construction, controls wiring and demo indicators."""

    def test_demo_engine_shows_dummy_indicators(self):
        """Demo mode titles the window and shows banner, badge and switch button.

        Args: None.
        Returns: None.
        """
        self.assertIn("DUMMY MODE", self.window.windowTitle())
        self.assertFalse(self.window.demo_banner.isHidden())
        self.assertFalse(self.window.switch_live_button.isHidden())
        self.assertTrue(self.window.camera_label.isHidden())
        self.assertIn("DUMMY", self.window.source_badge.text())
        self.assertIn("Simulated in dummy mode", self.window.exposure.toolTip())
        self.assertIn("Now: 8000 µs", self.window.exposure.toolTip())

    def test_set_demo_false_switches_to_camera_indicators(self):
        """Leaving demo mode hides every dummy indicator.

        Args: None.
        Returns: None.
        """
        self.window._set_demo(False)
        self.assertEqual(self.window.windowTitle(), "ESPI live")
        self.assertTrue(self.window.demo_banner.isHidden())
        self.assertTrue(self.window.switch_live_button.isHidden())
        self.assertFalse(self.window.camera_label.isHidden())
        self.assertIn("Camera", self.window.source_badge.text())
        tip = self.window.exposure.toolTip()
        self.assertNotIn("Simulated", tip)
        self.assertIn("collects light", tip)
        self.assertIn("Now: 8000 µs", tip)

    def test_first_run_uses_default_window_size(self):
        """Without saved geometry the window opens at the default size.

        Args: None.
        Returns: None.
        """
        self.assertEqual((self.window.width(), self.window.height()),
                         (qt_app.DEFAULT_WIDTH, qt_app.DEFAULT_HEIGHT))

    def test_cpu_view_is_used_when_gl_not_requested(self):
        """use_gl=False builds the CPU fallback view.

        Args: None.
        Returns: None.
        """
        self.assertIsInstance(self.window.view, qt_app.CpuImageView)

    def test_startup_check_runs_when_updates_are_available(self):
        """With updates available and enabled the worker starts a check.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(qt_app.updater, "unavailable_reason", return_value=None):
            window = demo_window(self.engine, self.log, check_updates=True)
        self.addCleanup(window.deleteLater)
        window.frame_timer.stop()
        window.status_timer.stop()
        window.tip_timer.stop()
        self.assertEqual(window.updates.checks, 1)
        self.assertTrue(window.update_timer.isActive())
        span = abs(window.update_timer.interval() - qt_app.UPDATE_CHECK_INTERVAL_MS)
        self.assertLessEqual(span, qt_app.UPDATE_CHECK_JITTER_MS)
        window.close()

    def test_mode_button_changes_engine_mode(self):
        """Clicking a view-mode button updates the engine's mode.

        Args: None.
        Returns: None.
        """
        button = next(b for b in self.window.mode_group.buttons() if b.text() != "Raw")
        button.click()
        self.assertEqual(self.engine.settings().mode, button.text())

    def test_record_button_toggles_engine_recording(self):
        """Toggling the record button sets the engine's record flag.

        Args: None.
        Returns: None.
        """
        self.window.record_button.setChecked(True)
        self.assertTrue(self.engine.settings().record)

    def test_toolbar_buttons_queue_engine_requests(self):
        """Reference, reset and switch-live buttons queue their actions.

        Args: None.
        Returns: None.
        """
        buttons = {b.text(): b for b in self.window.actions_bar.findChildren(
            qt_app.QPushButton)}
        for label in ("Set reference (F)", "Reset last (R)", "Reset average (A)"):
            buttons[label].click()
        self.window.switch_live_button.click()
        self.assertEqual([a for a, _ in self.queued()],
                         ["reference", "reset_previous", "reset_average", "switch_live"])

    def test_save_frame_sends_title_and_notes(self):
        """Saving passes the title and notes text to the engine.

        Args: None.
        Returns: None.
        """
        self.window.frame_title.setText("Fringes")
        self.window.frame_notes.setPlainText("run 3")
        self.window.save_frame()
        self.assertEqual(self.queued(), [("save", {"title": "Fringes", "notes": "run 3"})])

    def test_sliders_update_engine_settings(self):
        """Each slider-and-spin control writes its engine setting.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.exposure.spin.setValue(1234)
        w.gain.spin.setValue(3.5)
        w.fps.spin.setValue(12.5)
        w.noise_reduction.spin.setValue(60)
        w.average.spin.setValue(5)
        w.ref_refresh.spin.setValue(2.5)
        w.smoothing.spin.setValue(3)
        w.clip.spin.setValue(1.5)
        w.gamma.spin.setValue(1.7)
        s = self.engine.settings()
        self.assertEqual((s.exposure_us, s.gain_db, s.fps), (1234, 3.5, 12.5))
        self.assertEqual(s.noise_reduction, 60)
        tip = " ".join(w.noise_reduction.toolTip().split())
        self.assertIn("grain stays", tip)
        self.assertIn("Speckle filter", tip)
        self.assertIn("Average N", tip)
        self.assertIn("Stretch", tip)
        self.assertIn("Now: 60 %", tip)
        note = w.noise_note.text()
        self.assertIn("Grain stays", note)
        self.assertIn("Speckle filter", note)
        self.assertIn("Average N", note)
        self.assertEqual((s.average, s.ref_refresh_s, s.smoothing), (5, 2.5, 3))
        self.assertEqual((s.clip_percent, s.gamma), (1.5, 1.7))

    def test_gamma_stays_its_own_control_and_explains_stretch(self):
        """Gamma is a visible Display slider, with a tip that contrasts stretch.

        Args: None.
        Returns: None.
        """
        self.window.show()
        gamma = self.window.gamma
        self.assertTrue(gamma.isVisible())
        self.assertFalse(gamma.isHidden())
        tip = gamma.toolTip()
        self.assertIn("Stretch", qt_app.GAMMA_TIP)
        self.assertIn("Stretch", tip)
        self.assertIn("1.0", tip)
        self.assertIn("Now: 1.0", tip)
        self.assertEqual(gamma.slider.toolTip(), tip)
        self.assertEqual(gamma.spin.toolTip(), tip)
        labels = [label for label in self.window.controls_dock.findChildren(QLabel)
                  if label.text() == "Gamma"]
        self.assertEqual([label.toolTip() for label in labels], [tip])
        stretch = self.window.scale_range.toolTip()
        self.assertIn("Raw", stretch)
        self.assertIn("gamma", stretch.lower())
        self.assertNotIn("advanced", tip.lower())

    def test_checkboxes_and_combos_update_engine_settings(self):
        """Check boxes and combo boxes write their engine settings.

        Args: None.
        Returns: None.
        """
        w = self.window
        before = self.engine.settings()
        w.rolling.setChecked(not before.rolling)
        w.scale_range.setChecked(not before.scale_range)
        w.invert.setChecked(not before.invert)
        w.clahe.setChecked(not before.clahe)
        w.smoothing_combo.setCurrentIndex(w.smoothing_combo.count() - 1)
        w.palette_combo.setCurrentIndex(w.palette_combo.count() - 1)
        after = self.engine.settings()
        self.assertEqual(after.rolling, not before.rolling)
        self.assertEqual(after.scale_range, not before.scale_range)
        self.assertEqual(after.invert, not before.invert)
        self.assertEqual(after.clahe, not before.clahe)
        self.assertEqual(after.smoothing_method, w.smoothing_combo.currentText())
        self.assertEqual(after.palette, w.palette_combo.currentText())

    def test_zoom_slider_and_buttons_drive_the_view(self):
        """The zoom slider zooms the view and Fit and 1:1 buttons navigate.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.view.show_packet(make_packet(width=2000, height=1500),
                           np.zeros((256, 3), np.uint8), None)
        w.zoom_slider.setValue(qt_app.ZOOM_TICKS // 2)
        self.assertAlmostEqual(w.view.zoom, qt_app.MAX_ZOOM ** 0.5)
        buttons = {b.text(): b for b in w.actions_bar.findChildren(qt_app.QPushButton)}
        buttons["Fit"].click()
        self.assertEqual(w.view.zoom, 1.0)
        buttons["1:1"].click()
        self.assertGreater(w.view.zoom, 1.0)

    def test_view_zoom_updates_slider_and_label_without_feedback(self):
        """View zoom changes move the slider and label, not the view again.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.view.show_packet(make_packet(width=2000, height=1500),
                           np.zeros((256, 3), np.uint8), None)
        w.view.set_zoom(8.0)
        self.assertEqual(w.zoom_label.text(), "800%")
        self.assertEqual(w.zoom_slider.value(), qt_app.ZOOM_TICKS // 2)
        self.assertEqual(w.view.zoom, 8.0)

    def test_view_menu_actions_navigate(self):
        """The View menu's fit and actual-size actions call the view.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.view.show_packet(make_packet(width=2000, height=1500),
                           np.zeros((256, 3), np.uint8), None)
        actions = {a.text(): a for a in w.menuBar().actions()[0].menu().actions()}
        actions["Actual size"].trigger()
        self.assertGreater(w.view.zoom, 1.0)
        actions["Fit to window"].trigger()
        self.assertEqual(w.view.zoom, 1.0)

    def test_about_action_shows_build_description(self):
        """Help > About opens the about box with the build description.

        Args: None.
        Returns: None.
        """
        help_menu = self.window.menuBar().actions()[1].menu()
        next(a for a in help_menu.actions() if a.text().startswith("About")).trigger()
        self.dialogs.about.assert_called_once()
        self.assertEqual(self.dialogs.about.call_args.args[2], updater.describe_build())

    def test_guide_action_opens_the_html_file(self):
        """Help > User Guide hands the bundled HTML file to the browser.

        Args: None.
        Returns: None.
        """
        help_menu = self.window.menuBar().actions()[1].menu()
        guide = next(a for a in help_menu.actions() if a.text() == "User Guide")
        path = os.path.join(os.path.dirname(qt_app.__file__), "user_guide.html")
        with mock.patch.object(qt_app, "guide_path", return_value=path), \
                mock.patch.object(qt_app.QDesktopServices, "openUrl",
                                  return_value=True) as open_url:
            guide.trigger()
        self.assertEqual(open_url.call_count, 1)
        self.assertTrue(open_url.call_args.args[0].toLocalFile().endswith(
            "user_guide.html"))

    def test_guide_action_reports_a_missing_file(self):
        """Help > User Guide warns when the bundled file is absent.

        Args: None.
        Returns: None.
        """
        help_menu = self.window.menuBar().actions()[1].menu()
        guide = next(a for a in help_menu.actions() if a.text() == "User Guide")
        with mock.patch.object(qt_app, "guide_path",
                               return_value="/no/such/user_guide.html"):
            guide.trigger()
        self.dialogs.warning.assert_called_once()
        self.assertIn("could not be found", self.dialogs.warning.call_args.args[2])

    def test_guide_action_reports_an_unopenable_browser(self):
        """Help > User Guide gives the path when no browser starts.

        Args: None.
        Returns: None.
        """
        help_menu = self.window.menuBar().actions()[1].menu()
        guide = next(a for a in help_menu.actions() if a.text() == "User Guide")
        path = os.path.join(os.path.dirname(qt_app.__file__), "user_guide.html")
        with mock.patch.object(qt_app, "guide_path", return_value=path), \
                mock.patch.object(qt_app.QDesktopServices, "openUrl",
                                  return_value=False):
            guide.trigger()
        self.dialogs.information.assert_called_once()
        self.assertIn(path, self.dialogs.information.call_args.args[2])

    def test_hover_signal_is_remembered(self):
        """The view's hover signal is stored for the pixel readout.

        Args: None.
        Returns: None.
        """
        self.window.view.hovered.emit(3, 4)
        self.assertEqual(self.window.hover, (3, 4))


def _dock_controls_missing_tooltips(dock):
    """List interactive controls in a dock that have no tooltip.

    The spin box's internal editor is part of the spin box, so it is not
    listed on its own. A new slider, button, combo or text field with no
    tooltip fails the suite.

    Args: dock is the controls QDockWidget.
    Returns: Widgets that still need a tooltip.
    """
    kinds = (QAbstractButton, QAbstractSlider, QAbstractSpinBox, QComboBox,
             QLineEdit, QPlainTextEdit, QTextEdit)
    containers = (QAbstractSpinBox, QComboBox, QPlainTextEdit, QTextEdit)
    missing = []
    for widget in dock.findChildren(QWidget):
        if not isinstance(widget, kinds):
            continue
        parent = widget.parentWidget()
        inside = False
        while parent is not None:
            if isinstance(parent, containers):
                inside = True
                break
            parent = parent.parentWidget()
        if inside:
            continue
        if not widget.toolTip().strip():
            missing.append(widget)
    return missing


class SettingsGuideTests(WindowTestCase):
    """Tooltips, the settings guide and the two processing presets."""

    def test_processing_controls_explain_use_and_a_starting_value(self):
        """Each processing control says what it does and where to start.

        Args: None.
        Returns: None.
        """
        w = self.window
        expected = {
            w.noise_reduction: ("sensor noise", "Start at 0"),
            w.average: ("difference frames", "Start at 8"),
            w.rolling: ("one image", "frame rate", "lag"),
            w.ref_refresh: ("drift", "Start at 0"),
            w.smoothing_combo: ("Gaussian", "Median", "5 px"),
            w.smoothing: ("fringe spacing", "Start at 3"),
            w.scale_range: ("0 and 255", "Start on"),
            w.invert: ("bright and dark", "Start off"),
            w.clip: ("speckles", "Start at 0"),
            w.palette_combo: ("Start with Gray",),
            w.gamma: ("Start at 1.0",),
            w.clahe: ("color palette", "Gray"),
        }
        for widget, phrases in expected.items():
            tip = " ".join(widget.toolTip().split())
            for phrase in phrases:
                self.assertIn(phrase, tip)

    def test_presets_apply_processing_and_leave_the_camera_alone(self):
        """Live preview and Measurement set processing, not exposure or recording.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.exposure.spin.setValue(1234)
        w.gain.spin.setValue(2.5)
        w.fps.spin.setValue(15)
        w.record_button.setChecked(True)
        w.single_field.setChecked(False)
        buttons = {b.text(): b for b in w.controls_dock.findChildren(qt_app.QPushButton)}
        buttons["Measurement"].click()
        s = self.engine.settings()
        self.assertEqual(s.mode, "Reference")
        self.assertEqual((s.average, s.rolling, s.ref_refresh_s), (8, False, 2.0))
        self.assertEqual((s.smoothing, s.smoothing_method), (3, "Median"))
        self.assertEqual((s.noise_reduction, s.clip_percent, s.clahe), (0.0, 0.5, True))
        self.assertEqual((s.exposure_us, s.gain_db, s.fps, s.record), (1234, 2.5, 15, True))
        self.assertFalse(s.single_field)
        self.assertEqual(w.average.value(), 8)
        self.assertTrue(w.mode_group.checkedButton().text() == "Reference")
        buttons["Live preview"].click()
        s = self.engine.settings()
        self.assertEqual(qt_app.PROCESSING_PRESETS["Live preview"]["mode"], s.mode)
        self.assertEqual((s.average, s.smoothing, s.clahe, s.ref_refresh_s), (1, 0, False, 0.0))
        self.assertEqual((s.exposure_us, s.gain_db, s.fps, s.record), (1234, 2.5, 15, True))
        self.assertEqual(w.mode_group.checkedButton().text(), "Previous")
        self.assertIn("Preset applied: Live preview", self.log.lines)
        self.assertIn("exposure", buttons["Live preview"].toolTip())

    def test_guide_lists_the_pipeline_and_applies_a_preset(self):
        """The Help guide matches the README and its buttons apply a preset.

        Args: None.
        Returns: None.
        """
        with open(os.path.join(paths.PROJECT_ROOT, "README.md"), encoding="utf-8") as handle:
            readme = handle.read()
        self.assertIn(qt_app.PIPELINE, readme)
        self.assertIn(qt_app.PIPELINE, qt_app.SETTINGS_GUIDE)
        self.assertIn("display gamma", qt_app.SETTINGS_GUIDE)
        self.assertIn("Below 1.0 darkens mid-tones", " ".join(self.window.gamma.toolTip().split()))
        help_menu = self.window.menuBar().actions()[-1].menu()
        entry = next(a for a in help_menu.actions()
                     if a.text() == "Which settings should I use?")
        with mock.patch.object(qt_app.SettingsGuideDialog, "exec", return_value=0) as shown:
            entry.trigger()
        shown.assert_called_once()
        dialog = qt_app.SettingsGuideDialog(self.window)
        self.assertEqual(dialog.guide.toPlainText(), qt_app.SETTINGS_GUIDE)
        buttons = {b.text(): b for b in dialog.findChildren(qt_app.QPushButton)}
        buttons["Apply Measurement"].click()
        self.assertEqual(self.engine.settings().mode, "Reference")
        self.assertEqual(self.engine.settings().average, 8)
        buttons["Apply Live preview"].click()
        self.assertEqual(self.engine.settings().mode, "Previous")
        buttons["Close"].click()
        self.assertFalse(dialog.isVisible())


class TooltipCoverageTests(WindowTestCase):
    """Every control explains itself, and tips stay off the live image."""

    def test_controls_dock_rejects_a_control_without_a_tooltip(self):
        """A controls-dock slider, button, combo or text field must have a tip.

        Args: None.
        Returns: None.
        """
        missing = _dock_controls_missing_tooltips(self.window.controls_dock)
        self.assertEqual([type(widget).__name__ for widget in missing], [])

    def test_sliders_include_the_live_value_and_shortcuts(self):
        """Slider tips name the current value; shortcut controls name the key.

        Args: None.
        Returns: None.
        """
        self.assertIn("Now: 1.0", self.window.gamma.toolTip())
        self.assertIn("1.0", self.window.gamma.toolTip())
        self.window.gamma.set_value(2.5)
        self.assertIn("Now: 2.5", self.window.gamma.toolTip())
        self.assertIn("Now: 8000 µs", self.window.exposure.toolTip())
        self.assertIn("Now: 100%", self.window.zoom_slider.toolTip())
        self.window._show_zoom(8)
        self.assertIn("Now: 800%", self.window.zoom_slider.toolTip())
        self.assertIn("Shortcut: S", self.window.frame_title.toolTip())
        self.assertIn("Shortcut: Space", self.window.freeze_button.toolTip())
        for button in self.window.mode_group.buttons():
            self.assertIn(button.text(), button.toolTip())
        self.assertIn("AVI", self.window.record_button.toolTip())
        self.assertEqual(self.window.view.toolTip(), "")

    def test_menus_and_status_badges_have_tooltips(self):
        """Menu actions and status readouts say what they show.

        Args: None.
        Returns: None.
        """
        for name in ("hint_label", "frozen_badge", "source_badge", "record_label",
                     "range_status", "camera_fps_status", "speed_status", "pixel_status",
                     "update_button", "check_button", "feature_button", "codex_button"):
            self.assertTrue(getattr(self.window, name).toolTip().strip(), name)
        for top in self.window.menuBar().actions():
            for action in top.menu().actions():
                if not action.isSeparator():
                    self.assertTrue(action.toolTip().strip(), action.text())
        self.assertIn("Space", self.window.freeze_action.toolTip())
        self.assertTrue(self.window.record_button.testAttribute(Qt.WA_AlwaysShowToolTips))
        self.assertTrue(self.window.testAttribute(Qt.WA_AlwaysShowToolTips))

    def test_tooltip_delay_is_installed_once(self):
        """Tips wait briefly, and a second install does not replace the style.

        Args: None.
        Returns: None.
        """
        style = QApplication.style()
        self.assertIsInstance(style, qt_app.TooltipDelayStyle)
        self.assertEqual(style.styleHint(QStyle.SH_ToolTip_WakeUpDelay), qt_app.TOOLTIP_DELAY_MS)
        self.assertEqual(style.styleHint(QStyle.SH_ToolTip_FallAsleepDelay),
                         qt_app.TOOLTIP_FALL_ASLEEP_MS)
        self.assertIsInstance(style.styleHint(QStyle.SH_EtchDisabledText), int)
        qt_app.install_tooltip_delay()
        self.assertIs(QApplication.style(), style)

    def _tip_event(self, widget):
        """Build the hover event Qt sends after the tooltip delay.

        Args: widget is the control under the pointer.
        Returns: A QHelpEvent.
        """
        local = QPoint(1, 1)
        return QHelpEvent(QEvent.ToolTip, local, widget.mapToGlobal(local))

    def test_a_control_tooltip_is_placed_off_the_image(self):
        """The filter shows the tip, and the rectangle misses the image.

        Args: None.
        Returns: None.
        """
        self.window.show()
        QApplication.processEvents()
        slider = self.window.gamma.slider
        with mock.patch.object(qt_app.QToolTip, "showText") as show:
            self.assertTrue(self.window.tooltip_filter.eventFilter(
                slider, self._tip_event(slider)))
        pos, text, widget = show.call_args.args
        self.assertIs(widget, slider)
        self.assertIn("Now:", text)
        view = self.window.view
        view_rect = QRect(view.mapToGlobal(QPoint(0, 0)), view.size())
        self.assertGreater(view_rect.width(), 0)
        self.assertFalse(view_rect.intersects(QRect(pos, qt_app.tooltip_size(text))))

    def test_the_image_never_shows_a_tooltip(self):
        """A tip on the image view is swallowed so the picture stays visible.

        Args: None.
        Returns: None.
        """
        self.window.view.setToolTip("would cover the fringes")
        with mock.patch.object(qt_app.QToolTip, "showText") as show, \
                mock.patch.object(qt_app.QToolTip, "hideText") as hide:
            self.assertTrue(self.window.tooltip_filter.eventFilter(
                self.window.view, self._tip_event(self.window.view)))
        show.assert_not_called()
        hide.assert_called_once()

    def test_filter_ignores_other_events_windows_and_empty_tips(self):
        """Only a tip on one of this window's controls is handled.

        Args: None.
        Returns: None.
        """
        filt = self.window.tooltip_filter
        self.assertFalse(filt.eventFilter(self.window.gamma, QEvent(QEvent.Leave)))
        self.assertFalse(filt.eventFilter(QObject(), self._tip_event(self.window.gamma)))
        outsider = QPushButton("no")
        outsider.setToolTip("elsewhere")
        self.addCleanup(outsider.deleteLater)
        self.assertFalse(filt.eventFilter(outsider, self._tip_event(outsider)))
        self.assertFalse(filt.eventFilter(
            self.window.demo_banner, self._tip_event(self.window.demo_banner)))


class TooltipPlacementTests(unittest.TestCase):
    """Tooltip geometry stays on screen and off the image."""

    def setUp(self):
        """A host, an image and a control, in one coordinate system.

        Args: None.
        Returns: None.
        """
        self.screen = QRect(-10000, -10000, 40000, 40000)
        self.host = QWidget()
        self.host.setGeometry(0, 0, 2000, 1400)
        self.image = QWidget(self.host)
        self.image.setGeometry(200, 80, 900, 700)
        self.control = QPushButton("Gamma", self.host)
        self.control.setGeometry(1200, 200, 140, 28)
        self.control.setToolTip("Brighten midtones")
        self.host.show()

    def tearDown(self):
        """Delete the host.

        Args: None.
        Returns: None.
        """
        self.host.deleteLater()

    def _image_rect(self):
        """Return the image in global coordinates.

        Args: None.
        Returns: A QRect.
        """
        return QRect(self.image.mapToGlobal(QPoint(0, 0)), self.image.size())

    def test_a_tip_beside_the_image_opens_under_the_control(self):
        """A control clear of the image keeps the nearest position.

        Args: None.
        Returns: None.
        """
        pos = qt_app.tooltip_position(self.control, "Hi", self.image, self.screen)
        origin = self.control.mapToGlobal(QPoint(0, 0))
        self.assertEqual(pos, QPoint(origin.x(), origin.y() + self.control.height()
                                     + qt_app.TOOLTIP_MARGIN))
        self.assertFalse(self._image_rect().intersects(
            QRect(pos, qt_app.tooltip_size("Hi"))))

    def test_a_tip_on_the_image_moves_outside_it(self):
        """Every side of a control sitting on the image is rejected.

        Args: None.
        Returns: None.
        """
        self.control.setGeometry(500, 400, 40, 24)
        text = "Brightness"
        pos = qt_app.tooltip_position(self.control, text, self.image, self.screen)
        self.assertFalse(self._image_rect().intersects(
            QRect(pos, qt_app.tooltip_size(text))))

    def test_a_hidden_image_does_not_move_the_tip(self):
        """No visible image means the tip opens just below the control.

        Args: None.
        Returns: None.
        """
        self.control.setGeometry(500, 400, 40, 24)
        self.image.hide()
        pos = qt_app.tooltip_position(self.control, "Hi", self.image, self.screen)
        origin = self.control.mapToGlobal(QPoint(0, 0))
        self.assertEqual(pos.y(), origin.y() + self.control.height() + qt_app.TOOLTIP_MARGIN)

    def test_missing_image_uses_the_first_candidate(self):
        """None for the image is the same as a hidden one.

        Args: None.
        Returns: None.
        """
        pos = qt_app.tooltip_position(self.control, "Hi", None, self.screen)
        origin = self.control.mapToGlobal(QPoint(0, 0))
        self.assertEqual(pos, QPoint(origin.x(), origin.y() + self.control.height()
                                     + qt_app.TOOLTIP_MARGIN))

    def test_a_wide_tip_slides_onto_the_screen_beside_the_image(self):
        """A tip that hangs off the screen moves sideways, still clear of the image.

        Args: None.
        Returns: None.
        """
        origin = self.control.mapToGlobal(QPoint(0, 0))
        text = "word " * 30
        size = qt_app.tooltip_size(text)
        # The unnudged tip hangs off the right; sliding it left still clears the image.
        screen = QRect(origin.x() - 200, origin.y() - 20, size.width() + 160, 400)
        pos = qt_app.tooltip_position(self.control, text, self.image, screen)
        rect = QRect(pos, size)
        self.assertTrue(screen.contains(rect))
        self.assertFalse(self._image_rect().intersects(rect))
        self.assertLess(pos.x(), origin.x())

    def test_a_tip_may_leave_the_screen_to_spare_the_image(self):
        """An off-screen tip is kept when the on-screen one would cover the image.

        Args: None.
        Returns: None.
        """
        image = self._image_rect()
        screen = QRect(image.x() + 10, image.y() + 10, max(1, image.width() - 20),
                       max(1, image.height() - 20))
        text = "Brighten the midtones of a difference image"
        pos = qt_app.tooltip_position(self.control, text, self.image, screen)
        self.assertFalse(self._image_rect().intersects(
            QRect(pos, qt_app.tooltip_size(text))))
        self.assertFalse(screen.contains(QRect(pos, qt_app.tooltip_size(text))))

    def test_a_tiny_screen_clamps_the_tip(self):
        """When nothing fits, the tip is pushed to the screen's top-left.

        Args: None.
        Returns: None.
        """
        screen = QRect(0, 0, 20, 20)
        text = "one\ntwo\nthree\nfour"
        pos = qt_app.tooltip_position(self.control, text, None, screen)
        self.assertEqual(pos, QPoint(screen.left(), screen.top()))

    def test_no_widget_screen_uses_a_fallback_desktop(self):
        """A widget with no screen still gets a tip below it.

        Args: None.
        Returns: None.
        """

        class NoScreen(QPushButton):
            """A button whose screen() is missing."""

            def screen(self):
                """Report that no screen is attached.

                Args: None.
                Returns: None.
                """
                return None

        button = NoScreen("n", self.host)
        button.setGeometry(10, 10, 40, 20)
        button.show()
        pos = qt_app.tooltip_position(button, "Hi")
        origin = button.mapToGlobal(QPoint(0, 0))
        self.assertEqual(pos, QPoint(origin.x(), origin.y() + button.height()
                                     + qt_app.TOOLTIP_MARGIN))

    def test_the_widget_screen_is_used_when_none_is_passed(self):
        """Omitting the screen reads it from the widget.

        Args: None.
        Returns: None.
        """
        self.assertIsNotNone(self.control.screen())
        pos = qt_app.tooltip_position(self.control, "Hi")
        origin = self.control.mapToGlobal(QPoint(0, 0))
        self.assertEqual(pos.y(), origin.y() + self.control.height() + qt_app.TOOLTIP_MARGIN)

    def test_format_tooltip_wraps_and_drops_blank_parts(self):
        """Long text wraps; a blank part adds no line.

        Args: None.
        Returns: None.
        """
        self.assertEqual(qt_app.format_tooltip(), "")
        self.assertEqual(qt_app.format_tooltip("  ", ""), "")
        wrapped = qt_app.format_tooltip("word " * 40)
        self.assertGreater(wrapped.count("\n"), 0)
        self.assertLessEqual(max(len(line) for line in wrapped.split("\n")), qt_app.TOOLTIP_WRAP)
        self.assertGreater(qt_app.tooltip_size(wrapped).height(),
                           qt_app.tooltip_size("word").height())


class MainWindowGlTests(WindowTestCase):
    """The GPU view can be selected."""

    use_gl = True

    def test_gl_view_is_used_and_accepts_frames(self):
        """use_gl=True builds GLImageView and frames reach it.

        Args: None.
        Returns: None.
        """
        self.assertIsInstance(self.window.view, qt_app.GLImageView)
        self.engine._packet = make_packet()
        self.window._poll_frame()
        self.assertEqual(self.window.view.image_size, (64, 48))


class KeyHandlingTests(WindowTestCase):
    """handle_key and ShortcutFilter."""

    def test_action_keys_queue_engine_requests(self):
        """S, F, R and A queue save, reference, reset_previous, reset_average.

        Args: None.
        Returns: None.
        """
        for key in (Qt.Key_S, Qt.Key_F, Qt.Key_R, Qt.Key_A):
            self.assertTrue(self.window.handle_key(key))
        self.assertEqual([a for a, _ in self.queued()],
                         ["save", "reference", "reset_previous", "reset_average"])

    def test_quit_keys_close_the_window(self):
        """Q and Escape close the window.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(self.window, "close") as close:
            self.assertTrue(self.window.handle_key(Qt.Key_Q))
            self.assertTrue(self.window.handle_key(Qt.Key_Escape))
        self.assertEqual(close.call_count, 2)

    def test_zoom_keys_navigate(self):
        """0 fits and 1 shows actual size.

        Args: None.
        Returns: None.
        """
        self.window.view.show_packet(make_packet(width=2000, height=1500),
                                     np.zeros((256, 3), np.uint8), None)
        self.assertTrue(self.window.handle_key(Qt.Key_1))
        self.assertGreater(self.window.view.zoom, 1.0)
        self.assertTrue(self.window.handle_key(Qt.Key_0))
        self.assertEqual(self.window.view.zoom, 1.0)

    def test_unknown_key_is_not_handled(self):
        """Other keys are left alone.

        Args: None.
        Returns: None.
        """
        self.assertFalse(self.window.handle_key(Qt.Key_Z))

    def key_event(self, key, modifiers=Qt.NoModifier):
        """Build a key-press event.

        Args: key is a Qt.Key; modifiers are the keyboard modifiers.
        Returns: A QKeyEvent.
        """
        return QKeyEvent(QEvent.KeyPress, key, modifiers)

    def filter(self, event, focus):
        """Run the shortcut filter with a given focus widget.

        Args: event is the Qt event; focus is the widget QApplication reports.
        Returns: The filter's boolean result.
        """
        with mock.patch.object(QApplication, "focusWidget", return_value=focus):
            return self.window.shortcut_filter.eventFilter(None, event)

    def test_filter_ignores_non_key_events(self):
        """Events other than key presses pass through.

        Args: None.
        Returns: None.
        """
        self.assertFalse(self.filter(QEvent(QEvent.User), None))

    def test_filter_ignores_modified_keys(self):
        """Ctrl-modified keys are not shortcuts.

        Args: None.
        Returns: None.
        """
        event = self.key_event(Qt.Key_S, Qt.ControlModifier)
        self.assertFalse(self.filter(event, None))
        self.assertEqual(self.queued(), [])

    def test_filter_ignores_focus_in_other_windows(self):
        """Keys typed in another window are not shortcuts.

        Args: None.
        Returns: None.
        """
        other = QLineEdit()
        self.addCleanup(other.deleteLater)
        self.assertFalse(self.filter(self.key_event(Qt.Key_S), other))
        self.assertEqual(self.queued(), [])

    def test_filter_handles_shortcut_without_focus(self):
        """With no focus widget a shortcut key is consumed.

        Args: None.
        Returns: None.
        """
        self.assertTrue(self.filter(self.key_event(Qt.Key_F), None))
        self.assertEqual([a for a, _ in self.queued()], ["reference"])

    def test_filter_lets_text_fields_receive_typing(self):
        """Typing in a text field is not treated as a shortcut.

        Args: None.
        Returns: None.
        """
        self.assertFalse(self.filter(self.key_event(Qt.Key_S), self.window.frame_title))
        self.assertEqual(self.queued(), [])

    def test_escape_in_text_field_leaves_the_field(self):
        """Esc in a text field clears its focus instead of quitting.

        Args: None.
        Returns: None.
        """
        field = mock.MagicMock(spec=QLineEdit)
        field.window.return_value = self.window
        self.assertTrue(self.filter(self.key_event(Qt.Key_Escape), field))
        field.clearFocus.assert_called_once()


class PollingTests(WindowTestCase):
    """Frame and status polling."""

    def test_poll_frame_ignores_missing_and_stale_packets(self):
        """No packet, or an already-drawn sequence, draws nothing.

        Args: None.
        Returns: None.
        """
        self.window._poll_frame()
        self.assertEqual(self.window.drawn_frames, 0)
        self.engine._packet = make_packet(sequence=0)
        self.window._poll_frame()
        self.assertEqual(self.window.drawn_frames, 0)

    def test_poll_frame_draws_new_packet_and_caches_table(self):
        """A new frame is drawn; the color table is rebuilt only on change.

        Args: None.
        Returns: None.
        """
        self.engine._packet = make_packet(sequence=1)
        self.window._poll_frame()
        first_table = self.window.table
        self.engine._packet = make_packet(sequence=2)
        self.window._poll_frame()
        self.assertEqual(self.window.drawn_frames, 2)
        self.assertIs(self.window.table, first_table)
        self.assertEqual(self.window.view.image_size, (64, 48))
        self.engine._packet = make_packet(sequence=3, palette="Jet")
        self.window._poll_frame()
        self.assertIsNot(self.window.table, first_table)

    def test_poll_status_shows_log_text_and_engine_status(self):
        """Status polling appends log text and mirrors engine status strings.

        Args: None.
        Returns: None.
        """
        self.log.write("hello log")
        self.window.log_view.clear()
        self.window._poll_status()
        self.assertIn("hello log", self.window.log_view.toPlainText())
        status = self.engine.status()
        self.assertEqual(self.window.record_label.text(), status["record"])
        self.assertEqual(self.window.hint_label.text(), status["hint"])
        self.assertIn("Displayed:", self.window.speed_status.text())

    def test_status_bar_items_keep_their_positions(self):
        """No readout value, however long, moves another status-bar item.

        Args: None.
        Returns: None.
        """
        window = self.window
        window.show()
        QApplication.processEvents()
        items = (window.update_slot, window.feature_button, window.codex_button,
                 window.status_actions_rule, window.source_badge,
                 window.record_label, window.range_status, window.camera_fps_status,
                 window.speed_status, window.pixel_status)

        def geometry():
            QApplication.processEvents()
            return [(w.x(), w.width()) for w in items]

        before = geometry()
        window.camera_fps_status.setText("Camera: 1000000000.0 fps (simulated)")
        window.speed_status.setText("Processed: 1000000000.0 fps    Work: 0.0 ms/frame"
                                    "    Displayed: 1000000000.0 fps")
        window.record_label.setText("Recording: 1000000000.0 fps")
        window.range_status.setText("")
        window.pixel_status.setText("x 123456, y 123456: 255")
        self.assertEqual(geometry(), before)
        window.update_button.setText("Update available: build 9999 (abcdef0)")
        window.check_button.setVisible(False)
        window.update_button.setVisible(True)
        self.assertEqual(geometry(), before)
        window.check_button.setText("Checking for updates…")
        window.check_button.setVisible(True)
        window.update_button.setVisible(False)
        self.assertEqual(geometry(), before)
        window._rotate_tip()  # a longer or shorter tip must not shove the badges
        self.assertEqual(geometry(), before)
        window._rotate_tip()  # a longer or shorter tip must not shove the badges
        self.assertEqual(geometry(), before)
        window._rotate_tip()  # a longer or shorter tip must not shove the badges
        self.assertEqual(geometry(), before)

    def test_status_actions_look_like_buttons(self):
        """Idle, checking, update, sending and disabled states stay buttons.

        Args: None.
        Returns: None.
        """
        window = self.window
        window.show()
        QApplication.processEvents()
        for button in (window.check_button, window.feature_button, window.codex_button):
            self.assert_button_chrome(button, primary=False)
        self.assertFalse(window.check_button.icon().isNull())
        self.assertFalse(window.feature_button.icon().isNull())
        self.assertTrue(window.codex_button.icon().isNull())
        self.assertEqual(window.check_button.toolTip(),
                         qt_app.format_tooltip("Check GitHub for a newer build of ESPI live."))
        self.assertEqual(window.feature_button.toolTip(),
                         qt_app.format_tooltip(
                             "Send a feature request to the developers as a GitHub issue."))
        self.assertLess(window.codex_button.x(), window.status_actions_rule.x())
        self.assertLess(window.status_actions_rule.x(), window.source_badge.x())
        self.assertEqual(QLabel.text(window.source_badge), "  DUMMY MODE  ")
        self.assertGreaterEqual(window.source_badge.width(),
                                window.source_badge.sizeHint().width())
        self.assertEqual(window.status_actions_rule.frameShape(), qt_app.QFrame.Shape.VLine)
        window.check_button.setFocus(Qt.OtherFocusReason)
        self.assertTrue(window.check_button.hasFocus())

        window.updates.busy = True
        window._sync_check_button()
        self.assertFalse(window.check_button.isEnabled())
        self.assertEqual(window.check_button.text(), "Checking for updates…")
        self.assert_button_chrome(window.check_button, primary=False)
        self.assert_has_frame(window.check_button)

        window.updates.busy = False
        window.update_info = make_info()
        window.update_button.setText("Install update")
        window.update_button.setVisible(True)
        window._sync_check_button()
        self.assertTrue(window.check_button.isHidden())
        self.assertFalse(window.update_button.isHidden())
        self.assert_button_chrome(window.update_button, primary=True)
        self.assert_has_frame(window.update_button)
        self.assertLessEqual(window.update_button.sizeHint().width(),
                             window.update_button.width())

        window.update_button.setEnabled(False)
        window.update_button.setText("Downloading update… 25%")
        self.assertFalse(window.update_button.isEnabled())
        self.assert_button_chrome(window.update_button, primary=True)
        self.assert_has_frame(window.update_button)
        self.assertLessEqual(window.update_button.sizeHint().width(),
                             window.update_button.width())

        window.feature_button.setEnabled(False)
        window.feature_button.setText("Sending request…")
        self.assert_button_chrome(window.feature_button, primary=False)
        self.assert_has_frame(window.feature_button)
        self.assertLessEqual(window.feature_button.sizeHint().width(),
                             window.feature_button.width())
        self.assertEqual(window.feature_button.toolTip(),
                         qt_app.format_tooltip(
                             "Send a feature request to the developers as a GitHub issue."))

    def assert_button_chrome(self, button, primary):
        """Check the border, pointer, focus and state rules of one button.

        Args: button is a QPushButton; primary is True for the blue update button.
        Returns: None.
        """
        sheet = button.styleSheet()
        self.assertFalse(button.isFlat())
        self.assertEqual(button.cursor().shape(), Qt.PointingHandCursor)
        self.assertEqual(button.focusPolicy(), Qt.StrongFocus)
        for state in (":hover", ":pressed", ":focus", ":disabled"):
            self.assertIn(state, sheet)
        self.assertIn("border-radius", sheet)
        if primary:
            self.assertIn("#1d4f91", sheet)
            self.assertIn("font-weight: bold", sheet)
        else:
            self.assertNotIn("#1d4f91", sheet)
            self.assertIn("palette(base)", sheet)

    def assert_has_frame(self, button):
        """Compare the border with text-free padding at any display scale.

        Args: button is a visible QPushButton.
        Returns: None.
        """
        image = button.grab().toImage()
        mid_x = max(1, image.width() // 2)
        # QImage coordinates are physical pixels: a 1px border occupies only
        # row 0 at 1x, but rows 0 and 1 at 2x. Sample the padding before text.
        fill = image.pixelColor(mid_x, int(2 * image.devicePixelRatio()))
        edge = image.pixelColor(mid_x, 0)
        self.assertNotEqual((fill.red(), fill.green(), fill.blue()),
                            (edge.red(), edge.green(), edge.blue()))

    def test_fixed_label_keeps_full_text_but_shows_an_ellipsis(self):
        """A label too narrow for its text shows "…" yet still reports the full string.

        Args: None.
        Returns: None.
        """
        label = qt_app.FixedLabel("Camera: 9.9 fps")
        label.resize(label.sizeHint())
        hint = label.sizeHint().width()
        label.setText("Camera: 1000000000.0 fps")
        self.assertEqual(label.text(), "Camera: 1000000000.0 fps")
        self.assertTrue(QLabel.text(label).endswith("…"))
        self.assertEqual(label.sizeHint().width(), hint)  # the text never changes it
        label.setText("Camera: 1.0 fps")
        self.assertEqual(QLabel.text(label), "Camera: 1.0 fps")

    def test_fixed_label_shortens_again_when_the_window_narrows(self):
        """Shrinking the label re-elides its text; it may shrink to nothing.

        Args: None.
        Returns: None.
        """
        label = qt_app.FixedLabel("Camera: 999.9 fps")
        label.resize(label.sizeHint())
        label.show()  # a hidden widget gets its resize events only when shown
        self.addCleanup(label.close)
        label.setText("Camera: 30.0 fps")
        self.assertEqual(QLabel.text(label), "Camera: 30.0 fps")
        label.resize(30, label.height())
        self.assertTrue(QLabel.text(label).endswith("…"))
        self.assertEqual(label.text(), "Camera: 30.0 fps")
        self.assertEqual(label.minimumSizeHint().width(), 0)

    def test_poll_status_computes_display_fps_after_a_second(self):
        """After a second the displayed fps is frames over elapsed time.

        Args: None.
        Returns: None.
        """
        self.window.drawn_frames = 30
        self.window.fps_window_start = time.monotonic() - 2.0
        self.window._poll_status()
        self.assertAlmostEqual(self.window.display_fps, 15.0, delta=0.5)
        self.assertEqual(self.window.drawn_frames, 0)

    def test_poll_status_reads_pixel_value_under_cursor(self):
        """The pixel readout shows the value at the hovered pixel.

        Args: None.
        Returns: None.
        """
        packet = make_packet()
        self.engine._packet = packet
        self.window.hover = (10, 5)
        self.window._poll_status()
        self.assertEqual(self.window.pixel_status.text(),
                         f"x 10, y 5: {int(packet.image[5, 10])}")

    def test_poll_status_clears_pixel_readout_outside_image(self):
        """Hovering outside the image (or with no frame) clears the readout.

        Args: None.
        Returns: None.
        """
        self.window.pixel_status.setText("stale")
        self.window._poll_status()
        self.assertEqual(self.window.pixel_status.text(), "")
        self.engine._packet = make_packet()
        self.window.hover = (-1, -1)
        self.window.pixel_status.setText("stale")
        self.window._poll_status()
        self.assertEqual(self.window.pixel_status.text(), "")


class EngineEventTests(WindowTestCase):
    """Reaction to engine events."""

    def test_camera_limits_set_slider_ranges(self):
        """camera_limits updates the ranges of the controls it names.

        Args: None.
        Returns: None.
        """
        self.engine.events.put(("camera_limits", {
            "ExposureTime": (10, 20000), "Gain": (0, 12)}))
        self.window._handle_events()
        self.assertEqual(self.window.exposure.spin.maximum(), 20000)
        self.assertEqual(self.window.gain.spin.maximum(), 12)
        self.assertEqual(self.window.fps.spin.maximum(), 60)

    def test_camera_identity_names_the_model(self):
        """camera_identity puts the model in the title, badge and controls.

        Args: None.
        Returns: None.
        """
        source, self.engine.demo_source = self.engine.demo_source, None
        self.addCleanup(setattr, self.engine, "demo_source", source)
        self.engine.events.put(("camera_identity", CameraIdentity(
            vendor="FLIR", model="Blackfly S BFS-U3-70S7C", serial="42")))
        self.window._handle_events()
        self.assertEqual(self.window.windowTitle(), "ESPI live — BFS-U3-70S7C")
        self.assertEqual(self.window.source_badge.text(), "  BFS-U3-70S7C  ")
        self.assertIn("S/N 42", self.window.source_badge.toolTip())
        self.assertFalse(self.window.model_label.isHidden())
        self.assertIn("⚠ Colour camera", self.window.model_label.text())

    def test_camera_identity_is_hidden_in_dummy_mode(self):
        """A known camera is not advertised while synthetic video runs.

        Args: None.
        Returns: None.
        """
        self.window.camera_identity = CameraIdentity(model="BFS-U3-63S4M")
        self.window._set_demo(True)
        self.assertIn("DUMMY MODE", self.window.windowTitle())
        self.assertTrue(self.window.model_label.isHidden())
        self.assertEqual(self.window.source_badge.toolTip(),
                         "Synthetic video: no camera data")

    def test_camera_limits_set_frame_rate_range(self):
        """camera_limits can also set the frame-rate range.

        Args: None.
        Returns: None.
        """
        self.engine.events.put(("camera_limits", {"AcquisitionFrameRate": (1, 25)}))
        self.window._handle_events()
        self.assertEqual(self.window.fps.spin.maximum(), 25)

    def test_record_off_unchecks_button_without_touching_engine(self):
        """record_off clears the button silently.

        Args: None.
        Returns: None.
        """
        self.window.record_button.setChecked(True)
        self.engine.update_settings(record=True)
        self.engine.events.put(("record_off", None))
        self.window._handle_events()
        self.assertFalse(self.window.record_button.isChecked())
        self.assertTrue(self.engine.settings().record)

    def test_switched_live_selects_raw_and_leaves_demo(self):
        """switched_live selects Raw, unchecks record and hides demo indicators.

        Args: None.
        Returns: None.
        """
        self.window.record_button.setChecked(True)
        self.engine.events.put(("switched_live", None))
        self.window._handle_events()
        checked = [b.text() for b in self.window.mode_group.buttons() if b.isChecked()]
        self.assertEqual(checked, ["Raw"])
        self.assertFalse(self.window.record_button.isChecked())
        self.assertEqual(self.window.windowTitle(), "ESPI live")

    def test_switch_failed_warns_and_keeps_demo(self):
        """switch_failed warns the user and stays in demo mode.

        Args: None.
        Returns: None.
        """
        self.engine.events.put(("switch_failed", "no camera"))
        self.window._handle_events()
        self.dialogs.warning.assert_called_once()
        self.assertIn("no camera", self.dialogs.warning.call_args.args[2])
        self.assertIn("DUMMY", self.window.windowTitle())

    def test_fatal_shows_error_and_closes(self):
        """fatal shows a critical dialog and closes the window.

        Args: None.
        Returns: None.
        """
        self.engine.events.put(("fatal", ("kaboom", None)))
        with mock.patch.object(self.window, "close") as close:
            self.window._handle_events()
        self.assertEqual(self.dialogs.call_args.args[1:], ("ESPI stopped", "kaboom"))
        self.dialogs.return_value.exec.assert_called_once()
        close.assert_called_once()

    def test_stopped_closes_unless_already_closing(self):
        """stopped closes the window once, and is ignored while closing.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(self.window, "close") as close:
            self.engine.events.put(("stopped", None))
            self.window._handle_events()
            self.window.closing = True
            self.engine.events.put(("stopped", None))
            self.window._handle_events()
        close.assert_called_once()


class MultiCameraTests(unittest.TestCase):
    """Several cameras: the drop-down, process checkboxes and per-camera settings."""

    def setUp(self):
        """Build a window over three synthetic cameras (only the first open).

        Args: None.
        Returns: None.
        """
        FakeSettings.store = {}
        self.dialogs = make_dialogs()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for patcher in (
                mock.patch.object(qt_app, "QSettings", FakeSettings),
                mock.patch.object(qt_app, "QMessageBox", self.dialogs),
                mock.patch.object(qt_app.updater, "UpdateWorker", FakeUpdateWorker),
                mock.patch.object(qt_app.updater, "unavailable_reason",
                                  return_value="disabled in tests"),
                mock.patch.object(paths, "OUTPUT_DIR", tmp.name)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.log = FakeLog()
        self.channels = demo_channels(self.log, 3)
        self.window = qt_app.MainWindow(self.channels, self.log, False, False)
        self.window.frame_timer.stop()
        self.window.status_timer.stop()
        self.window.tip_timer.stop()
        self.addCleanup(self.cleanup_window)

    def cleanup_window(self):
        """Close the window, which stops and waits for every engine.

        Args: None.
        Returns: None.
        """
        self.window.close()
        self.window.deleteLater()

    def test_single_camera_hides_the_camera_list(self):
        """With one camera there is no drop-down and no log prefix.

        Args: None.
        Returns: None.
        """
        engine = Engine(self.log)
        engine.use_demo()
        window = demo_window(engine, self.log)
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        self.assertTrue(window.cameras_box.isHidden())
        self.assertEqual(window.windowTitle(), "ESPI live — DUMMY MODE")

    def test_cameras_are_listed_with_the_displayed_one_locked_on(self):
        """The drop-down lists every camera; the shown one must keep processing.

        Args: None.
        Returns: None.
        """
        window = self.window
        self.assertFalse(window.cameras_box.isHidden())
        self.assertEqual(window.camera_combo.count(), 3)
        self.assertEqual([c.isChecked() for c in window.camera_checks], [True, False, False])
        self.assertEqual([c.isEnabled() for c in window.camera_checks], [False, True, True])
        self.assertIn("(DEMO1) — DUMMY MODE", window.windowTitle())
        self.assertEqual(self.channels[1].log.prefix, "[DEMO2] ")

    def test_process_checkbox_starts_and_stops_a_background_camera(self):
        """Checking starts the camera's engine; unchecking stops it, keeping settings.

        Args: None.
        Returns: None.
        """
        second = self.channels[1]
        self.window.camera_checks[1].setChecked(True)
        self.assertTrue(second.active)
        self.assertTrue(wait_for(lambda: second.engine.latest_packet() is not None))
        second.engine.update_settings(gain_db=6.0)
        self.window.camera_checks[1].setChecked(False)
        self.assertFalse(second.active)
        self.assertEqual(second.settings.gain_db, 6.0)
        self.assertIn("Processing off: Synthetic camera (DEMO2) (settings kept)",
                      self.log.lines)

    def test_the_displayed_camera_cannot_be_turned_off(self):
        """A request to stop the displayed camera is ignored.

        Args: None.
        Returns: None.
        """
        self.window._set_processing(self.channels[0], False)
        self.assertTrue(self.channels[0].active)
        self.assertTrue(self.window.camera_checks[0].isChecked())

    def test_choosing_a_camera_shows_its_own_settings(self):
        """The controls switch to the chosen camera's settings and limits.

        Args: None.
        Returns: None.
        """
        window, second = self.window, self.channels[1]
        second.settings = replace(second.settings, exposure_us=2500.0, mode="Previous",
                                  palette="Jet", rolling=True, noise_reduction=40.0)
        second.limits = {"Gain": (0.0, 12.0)}
        window.exposure.set_value(4000.0)
        window.exposure.valueChanged.emit(4000.0)
        window.camera_combo.setCurrentIndex(1)
        self.assertIs(window.current, second)
        self.assertTrue(second.active)
        self.assertEqual(window.exposure.value(), 2500.0)
        self.assertEqual(window.noise_reduction.value(), 40.0)
        self.assertEqual(window.gain.spin.maximum(), 12.0)
        self.assertEqual(window.palette_combo.currentText(), "Jet")
        self.assertTrue(window.rolling.isChecked())
        checked = [b.text() for b in window.mode_group.buttons() if b.isChecked()]
        self.assertEqual(checked, ["Previous"])
        self.assertEqual([c.isEnabled() for c in window.camera_checks], [True, False, True])
        window.gain.valueChanged.emit(5.0)  # controls now act on the second camera
        self.assertEqual(second.engine.settings().gain_db, 5.0)
        self.assertEqual(self.channels[0].engine.settings().gain_db, 0.0)
        self.assertEqual(self.channels[0].engine.settings().exposure_us, 4000.0)
        window.camera_combo.setCurrentIndex(0)
        self.assertEqual(window.exposure.value(), 4000.0)
        self.assertEqual(window.gain.spin.maximum(), 30.0)

    def test_choosing_the_same_or_no_camera_does_nothing(self):
        """Re-selecting the shown camera or an empty index changes nothing.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(self.window, "_load_settings") as load:
            self.window._camera_chosen(-1)
            self.window.show_channel(self.channels[0])
        load.assert_not_called()

    def test_status_column_shows_display_rate_and_recording(self):
        """Each row shows displayed, processed fps, REC, or off.

        Args: None.
        Returns: None.
        """
        self.window.camera_checks[1].setChecked(True)
        self.channels[1].engine.update_settings(record=True)
        self.window._poll_status()
        states = [label.text() for label in self.window.camera_states]
        self.assertTrue(states[0].startswith("displayed · "))
        self.assertTrue(states[1].endswith(" · REC"))
        self.assertEqual(states[2], "off")

    def test_freeze_belongs_to_the_displayed_camera(self):
        """Each camera keeps its own frozen frame across display switches.

        Args: None.
        Returns: None.
        """
        window, first = self.window, self.channels[0]
        frozen = make_packet(sequence=1)
        first.engine._packet = frozen
        window._poll_frame()
        window.set_frozen(True)
        window._poll_status()
        self.assertTrue(window.camera_states[0].text().endswith(" · frozen"))
        window.camera_combo.setCurrentIndex(1)  # the second camera is live
        self.assertIsNone(window.current.frozen)
        self.assertIsNone(window.view.packet)
        self.assertTrue(window.frozen_banner.isHidden())
        self.assertFalse(window.freeze_button.isChecked())
        window._poll_status()
        self.assertTrue(window.camera_states[0].text().endswith(" · frozen"))
        self.assertFalse(window.camera_states[1].text().endswith(" · frozen"))
        window.camera_combo.setCurrentIndex(0)  # back: the held frame is shown at once
        self.assertIs(window.view.packet, frozen)
        self.assertFalse(window.frozen_banner.isHidden())
        self.assertTrue(window.freeze_button.isChecked())
        first.engine._packet = make_packet(sequence=2)
        window._poll_frame()
        self.assertIs(window.view.packet, frozen)

    def test_turning_off_a_frozen_background_camera_drops_its_frame(self):
        """A camera whose processing is turned off loses its frozen frame.

        Args: None.
        Returns: None.
        """
        window, first = self.window, self.channels[0]
        first.engine._packet = make_packet(sequence=1)
        window._poll_frame()
        window.set_frozen(True)
        window.camera_combo.setCurrentIndex(1)
        window.camera_checks[0].setChecked(False)
        self.assertFalse(first.active)
        self.assertIsNone(first.frozen)


    def test_background_events_do_not_change_the_shown_controls(self):
        """Limits are stored and record_off ignored for a camera not shown.

        Args: None.
        Returns: None.
        """
        window, second = self.window, self.channels[1]
        window.camera_checks[1].setChecked(True)
        window.record_button.setChecked(True)
        second.engine.events.put(("camera_limits", {"Gain": (0.0, 9.0)}))
        second.engine.events.put(("record_off", None))
        window._handle_events()
        self.assertEqual(second.limits, {"Gain": (0.0, 9.0)})
        self.assertEqual(window.gain.spin.maximum(), 30.0)
        self.assertTrue(window.record_button.isChecked())

    def test_each_camera_keeps_its_own_model(self):
        """A background camera's model is stored; the title follows the display.

        Args: None.
        Returns: None.
        """
        window, first, second = self.window, self.channels[0], self.channels[1]
        window.camera_checks[1].setChecked(True)
        for channel in (first, second):  # pretend both are real cameras
            source, channel.engine.demo_source = channel.engine.demo_source, None
            self.addCleanup(setattr, channel.engine, "demo_source", source)
        first.engine.events.put(("camera_identity", CameraIdentity(
            model="Blackfly S BFS-U3-63S4M", serial="DEMO1")))
        second.engine.events.put(("camera_identity", CameraIdentity(
            model="Blackfly S BFS-U3-70S7C", serial="DEMO2")))
        window._handle_events()
        self.assertEqual(window.windowTitle(), "ESPI live — BFS-U3-63S4M (DEMO1)")
        self.assertEqual(second.identity.model, "Blackfly S BFS-U3-70S7C")
        window.camera_combo.setCurrentIndex(1)
        self.assertEqual(window.windowTitle(), "ESPI live — BFS-U3-70S7C (DEMO2)")
        self.assertIn("⚠ Colour camera", window.model_label.text())

    def test_failed_open_of_the_shown_camera_returns_to_another(self):
        """open_failed warns, turns that camera off and shows a running one.

        Args: None.
        Returns: None.
        """
        window = self.window
        window.camera_combo.setCurrentIndex(2)
        third = self.channels[2]
        third.engine.events.put(("open_failed", "busy"))
        window._handle_events()
        self.dialogs.warning.assert_called_once()
        self.assertIn("busy", self.dialogs.warning.call_args.args[2])
        self.assertFalse(third.active)
        self.assertIs(window.current, self.channels[0])
        self.assertEqual(window.camera_combo.currentIndex(), 0)

    def test_fatal_error_in_a_background_camera_keeps_the_window(self):
        """A background camera's error names it and turns it off only.

        Args: None.
        Returns: None.
        """
        window, second = self.window, self.channels[1]
        window.camera_checks[1].setChecked(True)
        second.engine.events.put(("fatal", ("kaboom", None)))
        with mock.patch.object(window, "close") as close:
            window._handle_events()
        close.assert_not_called()
        self.assertIn("Synthetic camera (DEMO2): kaboom", self.dialogs.call_args.args[2])
        self.assertFalse(second.active)
        self.assertFalse(window.camera_checks[1].isChecked())
        self.assertIs(window.current, self.channels[0])

    def test_unexpected_stop_of_the_shown_camera_shows_another(self):
        """If the shown camera stops while another runs, that one is shown.

        Args: None.
        Returns: None.
        """
        window = self.window
        window.camera_checks[1].setChecked(True)
        self.channels[0].engine.events.put(("stopped", None))
        window._handle_events()
        self.assertIs(window.current, self.channels[1])
        self.assertFalse(self.channels[0].active)

    def test_going_live_replaces_synthetic_cameras_with_real_ones(self):
        """switched_live keeps the shown channel and lists the real cameras.

        Args: None.
        Returns: None.
        """
        window = self.window
        window.camera_checks[1].setChecked(True)
        demo_two = self.channels[1]
        shown = self.channels[0]
        shown.engine.camera_info = CameraInfo("111", "BFS")
        cameras = [CameraInfo("111", "BFS"), CameraInfo("222", "BFS")]
        shown.engine.events.put(("switched_live", cameras))
        window._handle_events()
        self.assertEqual([c.info.serial for c in window.channels], ["111", "222"])
        self.assertIs(window.channels[0], shown)
        self.assertEqual(shown.engine.tag, "111")
        self.assertFalse(window.channels[1].active)
        self.assertFalse(demo_two.active)
        self.assertIn(demo_two, window.retired_channels)
        self.assertEqual(window.camera_combo.itemText(1), "BFS (222)")
        self.assertIn("ESPI live — BFS (111)", window.windowTitle())

    def test_close_stops_every_engine(self):
        """Closing stops the running and the retired engines.

        Args: None.
        Returns: None.
        """
        self.window.camera_checks[1].setChecked(True)
        self.window.camera_checks[1].setChecked(False)
        self.window.camera_checks[2].setChecked(True)
        engines = self.window._all_engines()
        self.assertEqual(len(engines), 3)
        self.window.close()
        self.assertTrue(all(engine._stop.is_set() for engine in engines))
        self.assertTrue(self.window.wait_for_engines(5.0))


class UpdateFlowTests(WindowTestCase):
    """Update checks, downloads and restart offers, with no network."""

    def test_manual_check_reports_unavailable_updates(self):
        """When updates are off the reason is shown in a dialog.

        Args: None.
        Returns: None.
        """
        self.window._check_updates_now()
        self.dialogs.information.assert_called_once()
        self.assertIn("disabled in tests", self.dialogs.information.call_args.args[2])

    def test_manual_check_with_pending_update_offers_restart(self):
        """A downloaded update turns the manual check into a restart offer.

        Args: None.
        Returns: None.
        """
        self.window.pending_update = "staged"
        with mock.patch.object(qt_app.updater, "unavailable_reason", return_value=None):
            self.window._check_updates_now()
        self.dialogs.question.assert_called_once()
        self.assertEqual(self.dialogs.question.call_args.args[1], "Update ready")

    def test_manual_check_while_busy_only_shows_status(self):
        """A busy worker is not started again.

        Args: None.
        Returns: None.
        """
        self.window.updates.busy = True
        with mock.patch.object(qt_app.updater, "unavailable_reason", return_value=None):
            self.window._check_updates_now()
        self.assertEqual(self.window.updates.checks, 0)
        self.assertIn("already running", self.window.statusBar().currentMessage())

    def test_manual_check_starts_a_check_in_manual_mode(self):
        """An idle manual check starts the worker and reports results in dialogs.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(qt_app.updater, "unavailable_reason", return_value=None):
            self.window._check_updates_now()
        self.assertEqual(self.window.updates.checks, 1)
        self.assertTrue(self.window.update_manual)

    def test_update_click_with_pending_update_offers_restart(self):
        """Clicking the button after a download offers the restart.

        Args: None.
        Returns: None.
        """
        self.window.pending_update = "staged"
        self.window.update_button.click()
        self.assertEqual(self.dialogs.question.call_args.args[1], "Update ready")

    def test_update_click_without_info_or_while_busy_does_nothing(self):
        """No known update, or a busy worker, means no dialog.

        Args: None.
        Returns: None.
        """
        self.window._update_clicked()
        self.window.update_info = make_info()
        self.window.updates.busy = True
        self.window._update_clicked()
        self.dialogs.question.assert_not_called()

    def test_update_click_yes_starts_download(self):
        """Confirming the dialog starts the download and disables the button.

        Args: None.
        Returns: None.
        """
        self.dialogs.question.return_value = self.dialogs.Yes
        self.window.update_info = make_info()
        self.window._update_clicked()
        self.assertEqual(len(self.window.updates.downloads), 1)
        self.assertFalse(self.window.update_button.isEnabled())
        self.assertTrue(any("Downloading update" in line for line in self.log.lines))

    def test_update_click_no_does_not_download(self):
        """Declining the dialog downloads nothing.

        Args: None.
        Returns: None.
        """
        self.window.update_info = make_info()
        self.window._update_clicked()
        self.assertEqual(self.window.updates.downloads, [])

    def test_update_click_yes_with_busy_download_keeps_button(self):
        """If the worker refuses the download the button stays enabled.

        Args: None.
        Returns: None.
        """
        self.dialogs.question.return_value = self.dialogs.Yes
        self.window.updates.download_result = False
        self.window.update_info = make_info()
        self.window._update_clicked()
        self.assertTrue(self.window.update_button.isEnabled())

    def test_offer_restart_yes_closes_and_marks_relaunch(self):
        """Choosing Yes flags a relaunch and closes the window.

        Args: None.
        Returns: None.
        """
        self.dialogs.question.return_value = self.dialogs.Yes
        with mock.patch.object(self.window, "close") as close:
            self.window._offer_restart()
        self.assertTrue(self.window.relaunch_after_exit)
        close.assert_called_once()

    def test_offer_restart_no_keeps_running(self):
        """Choosing No leaves the window open and no relaunch flagged.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(self.window, "close") as close:
            self.window._offer_restart()
        self.assertFalse(self.window.relaunch_after_exit)
        close.assert_not_called()

    def post(self, kind, payload):
        """Queue an updater event and process it.

        Args: kind and payload are the event name and data.
        Returns: None.
        """
        self.window.updates.events.put((kind, payload))
        self.window._handle_update_events()

    def test_up_to_date_is_logged_and_silent_unless_manual(self):
        """An up-to-date result logs, and shows a dialog only when manual.

        Args: None.
        Returns: None.
        """
        self.post("checked", None)
        self.dialogs.information.assert_not_called()
        self.window.update_manual = True
        self.post("checked", None)
        self.dialogs.information.assert_called_once()
        self.assertFalse(self.window.update_manual)
        self.assertTrue(any(line.startswith("Up to date") for line in self.log.lines))

    def test_new_build_shows_update_button(self):
        """A newer build becomes Install update, with versions in the tooltip.

        The status bar mentions it once. A later check that finds the same
        build keeps the button and does not repeat the message.

        Args: None.
        Returns: None.
        """
        info = make_info()
        with mock.patch.object(qt_app.updater, "current_build", return_value=42):
            self.post("checked", info)
        self.assertIs(self.window.update_info, info)
        self.assertFalse(self.window.update_button.isHidden())
        self.assertEqual(self.window.update_button.text(), "Install update")
        self.assertIn("Build 42 → 99", self.window.update_button.toolTip())
        self.assertIn("Download", self.window.update_button.toolTip())
        self.assertEqual(self.window.statusBar().currentMessage(),
                         f"Update available: {info.label()}")
        self.dialogs.question.assert_not_called()
        self.window.statusBar().clearMessage()
        self.post("checked", info)
        self.assertEqual(self.window.statusBar().currentMessage(), "")
        self.assertEqual(self.window.update_button.text(), "Install update")

    def test_new_build_found_by_manual_check_prompts_download(self):
        """A manual check that finds a build goes straight to the download prompt.

        Args: None.
        Returns: None.
        """
        self.window.update_manual = True
        self.post("checked", make_info())
        self.dialogs.question.assert_called_once()

    def test_failed_check_is_logged_and_warned_only_when_manual(self):
        """A failed check warns the user only for manual checks.

        Args: None.
        Returns: None.
        """
        self.post("check_failed", "offline")
        self.dialogs.warning.assert_not_called()
        self.window.update_manual = True
        self.post("check_failed", "offline")
        self.dialogs.warning.assert_called_once()
        self.assertTrue(any("offline" in line for line in self.log.lines))

    def test_downloaded_build_is_staged_and_restart_offered(self):
        """A finished download is staged and the restart is offered.

        Args: None.
        Returns: None.
        """
        self.post("downloaded", "staged-path")
        self.assertEqual(self.window.pending_update, "staged-path")
        self.assertEqual(self.window.update_button.text(), "Restart to update")
        self.assertTrue(self.window.update_button.isEnabled())
        self.assertEqual(self.dialogs.question.call_args.args[1], "Update ready")

    def test_failed_download_restores_button_and_warns(self):
        """A failed download re-enables the button and warns.

        Args: None.
        Returns: None.
        """
        self.window.update_info = make_info()
        self.window.update_button.setEnabled(False)
        self.post("download_failed", "disk full")
        self.assertTrue(self.window.update_button.isEnabled())
        self.assertEqual(self.window.update_button.text(), "Install update")
        self.assertIn("Build 0 → 99", self.window.update_button.toolTip())
        self.dialogs.warning.assert_called_once()

    def test_progress_is_shown_while_downloading(self):
        """Download progress appears as a percentage on the button.

        Args: None.
        Returns: None.
        """
        self.window.update_info = make_info()
        self.window.updates.busy = True
        self.window.updates.downloading = True
        self.window.updates.progress = (25, 100)
        self.window._handle_update_events()
        self.assertEqual(self.window.update_button.text(), "Downloading update… 25%")
        self.window.updates.progress = (0, 0)
        self.window._handle_update_events()
        self.assertEqual(self.window.update_button.text(), "Downloading update…")

    def test_check_button_runs_a_manual_check(self):
        """The status-bar button checks like Help > Check for updates….

        Args: None.
        Returns: None.
        """
        with mock.patch.object(qt_app.updater, "unavailable_reason", return_value=None):
            self.window.check_button.click()
        self.assertEqual(self.window.updates.checks, 1)
        self.assertTrue(self.window.update_manual)

    def test_check_button_greys_out_while_a_check_runs(self):
        """A running check (startup or manual) disables the button until it ends.

        Args: None.
        Returns: None.
        """
        for kind, payload in (("checked", None), ("check_failed", "offline")):
            with self.subTest(kind=kind):
                self.window.updates.busy = True
                self.window._handle_update_events()
                self.assertFalse(self.window.check_button.isEnabled())
                self.assertIn("Checking", self.window.check_button.text())
                self.window.updates.busy = False
                self.post(kind, payload)
                self.assertTrue(self.window.check_button.isEnabled())
                self.assertEqual(self.window.check_button.text(), "Check for updates")

    def test_check_button_gives_way_to_a_found_build(self):
        """Once a newer build is known only the blue button shows, through its download.

        Args: None.
        Returns: None.
        """
        self.post("checked", make_info())
        self.assertTrue(self.window.check_button.isHidden())
        self.assertFalse(self.window.update_button.isHidden())
        self.window.updates.busy = True  # downloads run on the same worker
        self.window._handle_update_events()
        self.assertTrue(self.window.check_button.isHidden())
        self.window.updates.busy = False
        self.post("downloaded", "staged-path")
        self.assertTrue(self.window.check_button.isHidden())

    def checking_window(self):
        """Build a window whose periodic update check is armed.

        Args: None.
        Returns: The MainWindow. The startup check has already run once.
        """
        with mock.patch.object(qt_app.updater, "unavailable_reason", return_value=None):
            window = demo_window(self.engine, self.log, check_updates=True)
        self.addCleanup(window.frame_timer.stop)
        self.addCleanup(window.status_timer.stop)
        self.addCleanup(window.tip_timer.stop)
        self.addCleanup(window.update_timer.stop)
        self.addCleanup(window.close)
        self.addCleanup(window.deleteLater)
        return window

    def test_no_update_check_disables_the_periodic_timer(self):
        """A window opened with checks off never arms the timer or checks.

        Args: None.
        Returns: None.
        """
        self.assertFalse(self.window._periodic_checks)
        self.assertFalse(self.window.update_timer.isActive())
        self.window.update_timer.timeout.emit()
        self.window._schedule_update_check()
        self.assertFalse(self.window.update_timer.isActive())
        self.assertEqual(self.window.updates.checks, 0)

    def test_unavailable_updates_do_not_start_the_periodic_timer(self):
        """Source checkouts keep today's behavior: no check and no timer.

        Args: None.
        Returns: None.
        """
        window = demo_window(self.engine, self.log, check_updates=True)
        self.addCleanup(window.frame_timer.stop)
        self.addCleanup(window.status_timer.stop)
        self.addCleanup(window.tip_timer.stop)
        self.addCleanup(window.close)
        self.addCleanup(window.deleteLater)
        self.assertFalse(window._periodic_checks)
        self.assertFalse(window.update_timer.isActive())
        self.assertEqual(window.updates.checks, 0)

    def test_periodic_timer_fires_a_silent_check(self):
        """When the timer fires, another check starts and is not manual.

        Args: None.
        Returns: None.
        """
        window = self.checking_window()
        window.updates.checks = 0
        window.update_timer.timeout.emit()
        self.assertEqual(window.updates.checks, 1)
        self.assertFalse(window.update_manual)
        self.assertTrue(window.update_timer.isActive())

    def test_periodic_check_skips_while_busy_recording_or_saving(self):
        """A tick waits while a check, download, recording or save is in progress.

        A downloaded build waiting to install is left alone too, so Restart
        to update is not replaced. A skipped tick still schedules the next one.

        Args: None.
        Returns: None.
        """
        window = self.checking_window()
        window.updates.checks = 0
        window.update_timer.stop()
        window.updates.busy = True
        window._periodic_update_check()
        self.assertEqual(window.updates.checks, 0)
        self.assertTrue(window.update_timer.isActive())
        window.updates.busy = False
        window.pending_update = "staged"
        window.updates.checks = 0
        window._periodic_update_check()
        self.assertEqual(window.updates.checks, 0)
        window.pending_update = None
        self.engine.update_settings(record=True)
        window.updates.checks = 0
        window._periodic_update_check()
        self.assertEqual(window.updates.checks, 0)
        self.engine.update_settings(record=False)
        self.engine._set_status(record="Recording: finishing file")
        window.updates.checks = 0
        window._periodic_update_check()
        self.assertEqual(window.updates.checks, 0)
        self.engine._set_status(record="Recording: off")
        self.engine.request("save", title="", notes="")
        window.updates.checks = 0
        window._periodic_update_check()
        self.assertEqual(window.updates.checks, 0)
        self.assertTrue(self.engine.save_pending())

    def test_periodic_check_stops_when_the_window_is_closing(self):
        """Closing does not start a check or arm another one.

        Args: None.
        Returns: None.
        """
        window = self.checking_window()
        window.updates.checks = 0
        window.closing = True
        window.update_timer.stop()
        window._periodic_update_check()
        self.assertEqual(window.updates.checks, 0)
        self.assertFalse(window.update_timer.isActive())

    def test_failed_check_keeps_a_found_update_and_stays_silent(self):
        """A later failed check leaves Install update up and shows no dialog.

        Args: None.
        Returns: None.
        """
        self.post("checked", make_info())
        self.window.statusBar().clearMessage()
        self.post("check_failed", "offline")
        self.assertEqual(self.window.update_button.text(), "Install update")
        self.assertFalse(self.window.update_button.isHidden())
        self.assertIn("Build 0 → 99", self.window.update_button.toolTip())
        self.dialogs.warning.assert_not_called()
        self.assertEqual(self.window.statusBar().currentMessage(), "")
        self.assertTrue(any("offline" in line for line in self.log.lines))

    def test_recheck_does_not_relabel_the_button_as_a_download(self):
        """A check that follows a found build keeps the Install update label.

        Args: None.
        Returns: None.
        """
        self.post("checked", make_info())
        self.window.updates.busy = True
        self.window.updates.downloading = False
        self.window._handle_update_events()
        self.assertEqual(self.window.update_button.text(), "Install update")


class CloseTests(WindowTestCase):
    """closeEvent and a real engine round trip."""

    def test_close_stops_engine_and_saves_layout(self):
        """Closing stops the engine, shows a message and stores the layout.

        Args: None.
        Returns: None.
        """
        self.window.close()
        self.assertTrue(self.window.closing)
        self.assertTrue(self.engine._stop.is_set())
        self.assertIn("geometry", FakeSettings.store)
        self.assertIn("windowState", FakeSettings.store)

    def test_close_waits_until_engine_has_finished(self):
        """Closing keeps pumping events while the engine is still stopping.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(self.engine, "wait", side_effect=[False, True]) as wait:
            self.window.close()
        self.assertEqual(wait.call_count, 2)

    def test_second_close_does_not_stop_engine_again(self):
        """A repeated close skips the shutdown work.

        Args: None.
        Returns: None.
        """
        self.window.close()
        with mock.patch.object(self.engine, "stop") as stop:
            self.window.close()
        stop.assert_not_called()

    def test_saved_geometry_is_restored_by_next_window(self):
        """A new window restores the geometry saved by the previous one.

        Args: None.
        Returns: None.
        """
        self.window.resize(1000, 700)
        self.window.close()
        engine = Engine(self.log)
        engine.use_demo()
        window = demo_window(engine, self.log)
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.status_timer.stop)
        self.addCleanup(window.frame_timer.stop)
        self.addCleanup(window.tip_timer.stop)
        self.addCleanup(window.close)
        self.assertEqual(window.height(), 700)  # not the default height

    def test_running_demo_engine_feeds_the_view(self):
        """A started demo engine publishes frames that the window draws.

        Args: None.
        Returns: None.
        """
        self.engine.start()
        deadline = time.monotonic() + 5.0
        while self.engine.latest_packet() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        self.window._poll_frame()
        self.assertEqual(self.window.drawn_frames, 1)
        self.assertIsNotNone(self.window.view.image_size)
        self.window.close()
        self.assertTrue(self.engine.wait(5.0))


class FinishUpdateTests(unittest.TestCase):
    """finish_update installs a staged build after the window closed."""

    def make_window(self, relaunch, wait_result=True):
        """Build a stand-in closed window with a staged update.

        Args: relaunch is relaunch_after_exit; wait_result is engine.wait's result.
        Returns: A SimpleNamespace window.
        """
        return SimpleNamespace(pending_update="staged", relaunch_after_exit=relaunch,
                               wait_for_engines=mock.MagicMock(return_value=wait_result))

    def test_nothing_happens_without_window_or_pending_update(self):
        """No window, or nothing staged, installs nothing.

        Args: None.
        Returns: None.
        """
        log = FakeLog()
        with mock.patch.object(qt_app.updater, "install_update") as install:
            qt_app.finish_update(None, log)
            qt_app.finish_update(SimpleNamespace(pending_update=None), log)
        install.assert_not_called()

    def test_installs_without_relaunch_when_not_requested(self):
        """Installing on exit does not relaunch unless the user chose Yes.

        Args: None.
        Returns: None.
        """
        log, window = FakeLog(), self.make_window(False)
        with mock.patch.object(qt_app.updater, "install_update",
                               return_value="/app") as install, \
                mock.patch.object(qt_app.updater, "relaunch") as relaunch:
            qt_app.finish_update(window, log)
        install.assert_called_once_with("staged")
        relaunch.assert_not_called()
        self.assertIn("Installed update: /app", log.lines)

    def test_relaunches_once_camera_is_released(self):
        """A requested relaunch happens after the engine has stopped.

        Args: None.
        Returns: None.
        """
        log, window = FakeLog(), self.make_window(True)
        with mock.patch.object(qt_app.updater, "install_update", return_value="/app"), \
                mock.patch.object(qt_app.updater, "relaunch") as relaunch:
            qt_app.finish_update(window, log)
        relaunch.assert_called_once_with(sys.argv[1:])
        self.assertIn("Restarting into the new build", log.lines)

    def test_does_not_relaunch_while_camera_is_still_open(self):
        """If the engine did not stop in time the new build is not started.

        Args: None.
        Returns: None.
        """
        log, window = FakeLog(), self.make_window(True, wait_result=False)
        with mock.patch.object(qt_app.updater, "install_update", return_value="/app"), \
                mock.patch.object(qt_app.updater, "relaunch") as relaunch:
            qt_app.finish_update(window, log)
        relaunch.assert_not_called()
        self.assertTrue(any("not released in time" in line for line in log.lines))

    def test_install_errors_are_logged(self):
        """UpdateError and OSError during install are logged, not raised.

        Args: None.
        Returns: None.
        """
        for error in (updater.UpdateError("bad zip"), OSError("denied")):
            log = FakeLog()
            with mock.patch.object(qt_app.updater, "install_update", side_effect=error):
                qt_app.finish_update(self.make_window(False), log)
            self.assertTrue(any("could not be installed" in line for line in log.lines))


class ShortcutFilterConstructionTests(unittest.TestCase):
    """ShortcutFilter keeps a reference to its window."""

    def test_filter_remembers_window(self):
        """The filter stores the window it routes keys to.

        Args: None.
        Returns: None.
        """
        window = QLineEdit()
        self.addCleanup(window.deleteLater)
        self.assertIs(qt_app.ShortcutFilter(window).window, window)


class RunTests(unittest.TestCase):
    """run() startup logic with the channels, window and dialogs faked."""

    def setUp(self):
        """Patch everything run() touches that would block or hit hardware.

        Args: None.
        Returns: None.
        """
        self.log = FakeLog()
        self.engine = mock.MagicMock()
        self.channel = mock.MagicMock(active=True, engine=self.engine)
        self.idle = mock.MagicMock(active=False)
        self.cameras = mock.MagicMock(return_value=[self.channel, self.idle])
        self.demos = mock.MagicMock(return_value=[self.channel])
        self.window = mock.MagicMock()
        self.dialogs = make_dialogs(qt_app.QMessageBox.Yes)
        self.opengl = mock.MagicMock(return_value=(True, "OpenGL 4.1 on X"))
        self.finish = mock.MagicMock()
        self.main_window = mock.MagicMock(return_value=self.window)
        self.cleanup = mock.MagicMock()
        self.startup_cleanup = mock.MagicMock(return_value=self.cleanup)
        self.reporting = mock.MagicMock()
        self.reporting_class = mock.MagicMock(return_value=self.reporting)
        for patcher in (
                mock.patch.object(qt_app, "Reporting", self.reporting_class),
                mock.patch.object(qt_app.updater, "StartupCleanup", self.startup_cleanup),
                mock.patch.object(qt_app, "request_gl_format"),
                mock.patch.object(qt_app, "AppLog", return_value=self.log),
                mock.patch.object(qt_app, "open_camera_channels", self.cameras),
                mock.patch.object(qt_app, "demo_channels", self.demos),
                mock.patch.object(qt_app, "QMessageBox", self.dialogs),
                mock.patch.object(qt_app, "opengl_available", self.opengl),
                mock.patch.object(qt_app, "MainWindow", self.main_window),
                mock.patch.object(qt_app, "finish_update", self.finish),
                mock.patch.object(qt_app.QApplication, "exec", return_value=7)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_forced_demo_uses_demo_source_and_gpu_view(self):
        """force_demo skips the camera, builds the GPU window and returns exec's code.

        Args: None.
        Returns: None.
        """
        code = qt_app.run(force_demo=True, check_updates=False, demo_cameras=2)
        self.assertEqual(code, 7)
        self.cameras.assert_not_called()
        self.demos.assert_called_once_with(self.log, 2)
        self.assertEqual(self.main_window.call_args.args[0], [self.channel])
        self.assertEqual(self.main_window.call_args.args[2:], (True, False, self.reporting))
        self.window.show.assert_called_once()
        self.engine.start.assert_called_once()
        self.finish.assert_called_once_with(self.window, self.log)
        self.assertIn("closed", self.log.lines)
        self.assertIn("Display: GPU (OpenGL 4.1 on X)", self.log.lines)

    def test_cpu_fallback_is_logged_when_opengl_is_missing(self):
        """Without OpenGL the CPU view is chosen and the reason logged.

        Args: None.
        Returns: None.
        """
        self.opengl.return_value = (False, "no OpenGL context")
        qt_app.run(force_demo=True)
        self.assertEqual(self.main_window.call_args.args[2], False)
        self.assertIn("Display: CPU fallback (no OpenGL context)", self.log.lines)

    def test_cameras_are_used_when_one_opens(self):
        """Detected cameras are shown; only the opened one is started.

        Args: None.
        Returns: None.
        """
        qt_app.run()
        self.cameras.assert_called_once_with(self.log)
        self.demos.assert_not_called()
        self.assertEqual(self.main_window.call_args.args[0], [self.channel, self.idle])
        self.engine.start.assert_called_once()
        self.idle.engine.start.assert_not_called()

    def test_missing_camera_with_accepted_prompt_falls_back_to_demo(self):
        """No camera plus Yes at the prompt starts demo mode.

        Args: None.
        Returns: None.
        """
        self.cameras.side_effect = RuntimeError("no camera")
        self.assertEqual(qt_app.run(), 7)
        self.dialogs.question.assert_called_once()
        self.demos.assert_called_once_with(self.log, 1)
        self.assertTrue(any("Camera startup unavailable" in line
                            for line in self.log.lines))

    def test_missing_camera_with_declined_prompt_exits_cleanly(self):
        """No camera plus No at the prompt exits 0 without a window.

        Args: None.
        Returns: None.
        """
        self.cameras.side_effect = RuntimeError("no camera")
        self.dialogs.question.return_value = self.dialogs.No
        self.assertEqual(qt_app.run(), 0)
        self.main_window.assert_not_called()
        self.finish.assert_called_once_with(None, self.log)

    def test_opened_engines_are_released_when_the_window_fails(self):
        """If building the window raises, opened engines are stopped and waited for.

        Args: None.
        Returns: None.
        """
        self.main_window.side_effect = RuntimeError("no window")
        with self.assertRaises(RuntimeError):
            qt_app.run()
        self.engine.stop.assert_called_once()
        self.engine.wait.assert_called_once_with(8.0)
        self.idle.engine.stop.assert_not_called()
        self.finish.assert_called_once_with(None, self.log)

    def test_startup_cleanup_logs_and_stops_before_an_update_installs(self):
        """Old copies are cleaned up with messages to the log, and the cleanup
        stops before finish_update can install a new build.

        Args: None.
        Returns: None.
        """
        order = mock.MagicMock()
        order.attach_mock(self.cleanup.stop, "stop")
        order.attach_mock(self.finish, "finish")
        qt_app.run(force_demo=True)
        self.startup_cleanup.assert_called_once_with(self.log.write)
        self.cleanup.start.assert_called_once_with()
        self.assertEqual([name for name, _, _ in order.mock_calls], ["stop", "finish"])

    def test_log_file_path_is_reported_when_available(self):
        """The log file path from AppLog.start is written to the log.

        Args: None.
        Returns: None.
        """
        qt_app.run(force_demo=True)
        self.assertIn("Log file: fake.log", self.log.lines)

    def test_missing_log_file_is_not_reported(self):
        """If no log file could be opened no path line is written.

        Args: None.
        Returns: None.
        """
        self.log.start = lambda: None
        qt_app.run(force_demo=True)
        self.assertFalse(any(line.startswith("Log file:") for line in self.log.lines))


class FeatureRequestTests(WindowTestCase):
    """The Request feature dialog, sending and result reporting, with no network."""

    REQUEST = feedback.FeatureRequest(title="Export reference", description="As PNG",
                                      submitter="Sam", source="camera")

    def setUp(self):
        """Build the window with a fake token and a worker that never starts a thread.

        Args: None.
        Returns: None.
        """
        super().setUp()
        patcher = mock.patch.object(qt_app.feedback, "feedback_token", return_value="t")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.window.feedback = mock.MagicMock(busy=False, events=queue.Queue())

    def test_entry_points_exist(self):
        """The status bar button and the Help menu entry both open the request.

        Args: None.
        Returns: None.
        """
        help_menu = self.window.menuBar().actions()[-1].menu()
        entry = next(a for a in help_menu.actions() if a.text() == "Request a feature…")
        with mock.patch.object(self.window, "_request_feature") as request:
            entry.trigger()
            self.window.feature_button.click()
        self.assertEqual(request.call_count, 2)

    def test_dialog_restores_a_request_and_reads_it_back(self):
        """A previous request refills the form, and Submit files it.

        Args: None.
        Returns: None.
        """
        dialog = qt_app.FeatureRequestDialog(self.window, self.REQUEST)
        self.assertEqual(dialog.request("camera"), self.REQUEST)
        self.assertFalse(dialog.findChildren(qt_app.QCheckBox))
        send = dialog.findChild(qt_app.QDialogButtonBox).button(qt_app.QDialogButtonBox.Ok)
        self.assertEqual(send.text(), "Submit")
        self.assertIn("GitHub issue", send.toolTip())
        self.assertIn("Required", dialog.title.toolTip())
        self.assertIn("blank", dialog.description.toolTip())
        self.assertIn("follow up", dialog.submitter.toolTip())

    def test_dialog_defaults_come_from_settings_without_a_token(self):
        """A fresh form uses the saved name and still offers Submit.

        Args: None.
        Returns: None.
        """
        FakeSettings.store.update({"feedback/name": "Kim"})
        with mock.patch.object(qt_app.feedback, "feedback_token", return_value=""):
            dialog = qt_app.FeatureRequestDialog(self.window)
        self.assertEqual(dialog.submitter.text(), "Kim")
        send = dialog.findChild(qt_app.QDialogButtonBox).button(qt_app.QDialogButtonBox.Ok)
        self.assertEqual(send.text(), "Submit")
        cancel = dialog.findChild(qt_app.QDialogButtonBox).button(qt_app.QDialogButtonBox.Cancel)
        self.assertIn("without submitting", cancel.toolTip())

    def test_accept_refuses_a_blank_title(self):
        """A blank title warns and keeps the dialog open.

        Args: None.
        Returns: None.
        """
        dialog = qt_app.FeatureRequestDialog(self.window)
        dialog.description.setPlainText("A description")
        dialog.accept()
        self.dialogs.warning.assert_called_once()
        self.assertIn("title", self.dialogs.warning.call_args.args[2])
        self.assertEqual(dialog.result(), 0)

    def test_accept_allows_a_blank_description(self):
        """A title alone is enough; enrichment writes the description.

        Args: None.
        Returns: None.
        """
        dialog = qt_app.FeatureRequestDialog(self.window)
        dialog.title.setText("Only a title")
        dialog.accept()
        self.dialogs.warning.assert_not_called()
        self.assertEqual(dialog.result(), qt_app.QDialog.Accepted)
        self.assertEqual(dialog.request("camera").description, "")

    def test_accept_remembers_the_name(self):
        """A complete request closes the dialog and saves the name.

        Args: None.
        Returns: None.
        """
        dialog = qt_app.FeatureRequestDialog(self.window, self.REQUEST)
        dialog.accept()
        self.assertEqual(dialog.result(), qt_app.QDialog.Accepted)
        self.assertEqual(FakeSettings.store["feedback/name"], "Sam")
        self.assertNotIn("feedback/enrich", FakeSettings.store)

    def run_request(self, result):
        """Open the request flow with the dialog's exec() returning result.

        Args: result is the dialog code exec() returns.
        Returns: None.
        """
        with mock.patch.object(qt_app.FeatureRequestDialog, "exec", return_value=result), \
                mock.patch.object(qt_app.FeatureRequestDialog, "request",
                                  return_value=self.REQUEST):
            self.window._request_feature()

    def test_request_is_sent_in_the_background(self):
        """An accepted dialog submits through the worker and shows a busy button.

        Args: None.
        Returns: None.
        """
        self.run_request(qt_app.QDialog.Accepted)
        self.window.feedback.submit.assert_called_once_with(self.REQUEST)
        self.assertFalse(self.window.feature_button.isEnabled())
        self.assertEqual(self.window.feature_button.text(), "Sending request…")
        self.assertIn("Sending feature request: Export reference", self.log.lines)

    def test_cancel_sends_nothing(self):
        """A cancelled dialog submits nothing.

        Args: None.
        Returns: None.
        """
        self.run_request(qt_app.QDialog.Rejected)
        self.window.feedback.submit.assert_not_called()

    def test_busy_worker_only_shows_status(self):
        """A second request while one is sending is refused with a status message.

        Args: None.
        Returns: None.
        """
        self.window.feedback.busy = True
        self.window._request_feature()
        self.assertIn("still being sent", self.window.statusBar().currentMessage())

    def test_without_token_the_app_still_files_the_request(self):
        """A copy without a token still submits; it does not open a browser.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(qt_app.feedback, "feedback_token", return_value=""), \
                mock.patch.object(qt_app.QDesktopServices, "openUrl") as open_url:
            self.run_request(qt_app.QDialog.Accepted)
        self.window.feedback.submit.assert_called_once_with(self.REQUEST)
        open_url.assert_not_called()

    def test_sent_request_confirms_submission(self):
        """A filed request restores the button and says it was submitted.

        Args: None.
        Returns: None.
        """
        self.window.feature_button.setEnabled(False)
        url = "https://github.com/o/r/issues/9"
        self.window.feedback.events.put(("sent", (self.REQUEST, url)))
        self.window._handle_feedback_events()
        self.assertTrue(self.window.feature_button.isEnabled())
        self.assertEqual(self.window.feature_button.text(), "Request feature")
        self.assertIn(f"Feature request filed: {url}", self.log.lines)
        self.assertEqual(self.dialogs.call_args.args[2], "Your request was submitted.")
        self.dialogs.return_value.exec.assert_called_once()

    def failed_choice(self, choice):
        """Report a failed send and click one of the offered buttons.

        Args: choice is "retry" or "cancel".
        Returns: The _request_feature mock.
        """
        box = self.dialogs.return_value
        buttons = {"retry": object(), "cancel": object()}
        box.addButton.side_effect = [buttons["retry"], buttons["cancel"]]
        box.clickedButton.return_value = buttons[choice]
        self.window.feedback.events.put(("failed", (self.REQUEST, "offline")))
        with mock.patch.object(self.window, "_request_feature") as retry:
            self.window._handle_feedback_events()
        self.assertIn("Feature request failed: offline", self.log.lines)
        return retry

    def test_failed_request_can_be_retried(self):
        """Edit and retry reopens the dialog with the same request.

        Args: None.
        Returns: None.
        """
        retry = self.failed_choice("retry")
        retry.assert_called_once_with(self.REQUEST)

    def test_failed_request_can_be_dismissed(self):
        """Cancel does nothing more.

        Args: None.
        Returns: None.
        """
        retry = self.failed_choice("cancel")
        retry.assert_not_called()


class FakeReportWorker:
    """Stand-in for CameraReportWorker that records reports and never files."""

    def __init__(self):
        """Create an idle worker.

        Args: None.
        Returns: None.
        """
        self.events = queue.Queue()
        self.submitted = []

    def submit(self, report):
        """Record a report.

        Args: report is a camera_reports.Report.
        Returns: True.
        """
        self.submitted.append(report)
        return True


ANALOG_TIP = "Set on the camera itself; not adjustable over analog video"


class CaptureWindowTests(unittest.TestCase):
    """A Blackfly, a USB-Live2 (displayed) and a webcam in one window."""

    def setUp(self):
        """Build the window over fake cameras and fake video devices.

        Args: None.
        Returns: None.
        """
        FakeSettings.store = {}
        self.dialogs = make_dialogs()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        system = FakeSystem([FakeCamera(serial="1", model="Blackfly S BFS-U3-63S4M")])
        for patcher in (
                mock.patch.object(qt_app, "QSettings", FakeSettings),
                mock.patch.object(qt_app, "QMessageBox", self.dialogs),
                mock.patch.object(qt_app.updater, "UpdateWorker", FakeUpdateWorker),
                mock.patch.object(qt_app.updater, "unavailable_reason",
                                  return_value="disabled in tests"),
                mock.patch.object(qt_app.camera_reports, "CameraReportWorker",
                                  FakeReportWorker),
                mock.patch.object(paths, "OUTPUT_DIR", tmp.name),
                mock.patch.object(camera, "PySpin", make_pyspin(system)),
                mock.patch.object(capture, "load_enumerator",
                                  return_value=FakeEnumerator([usb_live2(0), webcam(1)])),
                mock.patch.object(capture.cv2, "VideoCapture",
                                  FakeCaptureFactory(frames=[interlaced_frame()]))):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.log = FakeLog()
        self.channels = [Channel(self.log, info) for info in sources.detect_cameras(self.log)]
        self.flir, self.usb, self.webcam = self.channels
        self.usb.open()
        self.window = qt_app.MainWindow(self.channels, self.log, False, False)
        self.window.frame_timer.stop()
        self.window.status_timer.stop()
        self.window.tip_timer.stop()
        self.addCleanup(self.cleanup_window)
        self.window._handle_events()  # identity and limits queued by the open

    def cleanup_window(self):
        """Close the window, which stops and waits for every engine.

        Args: None.
        Returns: None.
        """
        self.window.close()
        self.window.deleteLater()

    def show_webcam_report(self):
        """Display the webcam and wait until its engine sends a camera report.

        Args: None.
        Returns: The Report the window submitted.
        """
        window = self.window
        window.show_channel(self.webcam)

        def submitted():
            """Process engine events and look for the submitted report.

            Args: None.
            Returns: True once a report was submitted.
            """
            window._handle_events()
            return bool(window.reports.submitted)

        self.assertTrue(wait_for(submitted))
        return window.reports.submitted[0]

    def test_capture_channel_disables_camera_controls(self):
        """Analog cameras get fixed controls and a field toggle; FLIR ones do not.

        Args: None.
        Returns: None.
        """
        window = self.window
        self.assertIs(window.current, self.usb)
        for widget in (window.exposure, window.gain, window.fps):
            self.assertFalse(widget.isEnabled())
            self.assertIn(ANALOG_TIP, " ".join(widget.toolTip().split()))
            self.assertIn("Now:", widget.toolTip())
        self.assertFalse(window.single_field.isHidden())
        self.assertTrue(window.single_field.isChecked())
        self.assertTrue(window.report_label.isHidden())
        window.show_channel(self.flir)
        for widget in (window.exposure, window.gain, window.fps):
            self.assertTrue(widget.isEnabled())
            self.assertNotIn("analog video", widget.toolTip())
            self.assertNotIn("Simulated", widget.toolTip())
            self.assertIn("Now:", widget.toolTip())
        self.assertTrue(window.single_field.isHidden())

    def test_single_field_checkbox_updates_engine(self):
        """Unticking the field toggle switches the engine to full frames.

        Args: None.
        Returns: None.
        """
        self.window.single_field.setChecked(False)
        self.assertFalse(self.usb.engine.settings().single_field)
        self.window.single_field.setChecked(True)
        self.assertTrue(self.usb.engine.settings().single_field)

    def test_unsupported_marker_in_camera_list(self):
        """Only the unrecognised webcam is marked in the camera list.

        Args: None.
        Returns: None.
        """
        combo = self.window.camera_combo
        self.assertEqual([combo.itemText(i) for i in range(combo.count())],
                         ["Blackfly S BFS-U3-63S4M (1)", "Hauppauge USB-Live2 (video 1)",
                          "Webcam C270 (video 2) · not yet supported"])

    def test_camera_report_event_submits_and_shows_state(self):
        """Switching the webcam on files a report; its state shows under the camera.

        Args: None.
        Returns: None.
        """
        window = self.window
        report = self.show_webcam_report()
        self.assertTrue(report.key.startswith("Webcam C270"))
        self.assertIs(self.webcam.report, report)
        self.assertFalse(window.report_label.isHidden())
        self.assertIn("Reporting this camera", window.report_label.text())
        url = "https://github.com/owner/repo/issues/3"
        window.reports.events.put(("state", (report.key, camera_reports.ReportState(
            url=url, created=True))))
        window._poll_status()
        self.assertIn(url, window.report_label.text())
        self.assertTrue(window.report_button.isHidden())
        self.assertTrue(any(url in line for line in self.log.lines))
        window.reports.events.put(("state", (report.key, camera_reports.ReportState(
            path="out/logs/camera-report.md"))))
        window._handle_report_events()
        self.assertIn("saved to out/logs/camera-report.md", window.report_label.text())
        self.assertFalse(window.report_button.isHidden())
        window.show_channel(self.usb)
        self.assertTrue(window.report_label.isHidden())
        self.assertTrue(window.report_button.isHidden())

    def test_report_button_opens_browser(self):
        """The button opens the prefilled page, or warns when no browser opens.

        Args: None.
        Returns: None.
        """
        window = self.window
        report = self.show_webcam_report()
        window.reports.events.put(("state", (report.key, camera_reports.ReportState(
            error="offline"))))
        window._handle_report_events()
        with mock.patch.object(qt_app.QDesktopServices, "openUrl",
                               return_value=True) as open_url:
            window.report_button.click()
        self.assertIn("labels=camera-support", open_url.call_args[0][0].toString())
        self.dialogs.warning.assert_not_called()
        with mock.patch.object(qt_app.QDesktopServices, "openUrl", return_value=False):
            window.report_button.click()
        self.dialogs.warning.assert_called_once_with(
            window, "Report this camera", "Could not open a web browser.")


class DemoCaptureControlsTests(WindowTestCase):
    """Dummy mode shows neither analog controls nor camera reports."""

    def test_demo_hides_report_and_capture_controls(self):
        """The field toggle and report line stay hidden; sliders stay simulated.

        Args: None.
        Returns: None.
        """
        window = self.window
        self.assertTrue(window.single_field.isHidden())
        self.assertTrue(window.report_label.isHidden())
        self.assertTrue(window.report_button.isHidden())
        self.assertTrue(window.exposure.isEnabled())
        self.assertIn("Simulated in dummy mode", window.exposure.toolTip())
        self.assertIn("Now: 8000 µs", window.exposure.toolTip())


class QtMessageHandlerTests(unittest.TestCase):
    """handle_qt_message: Qt's own messages logged and reported."""

    def setUp(self):
        """Install a mock reporter for capture_message.

        Args: None.
        Returns: None.
        """
        self.reporter = mock.MagicMock()
        patcher = mock.patch.object(qt_app.error_reports, "_reporter", self.reporter)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_info_and_warning_messages_are_only_logged(self):
        """Warnings land in the log with their category; nothing is reported.

        Args: None.
        Returns: None.
        """
        with self.assertLogs("espi_live.qt", "WARNING") as captured:
            qt_app.handle_qt_message(qt_app.QtMsgType.QtWarningMsg,
                                     SimpleNamespace(category="qt.opengl"), "slow shader")
        self.assertIn("Qt (qt.opengl): slow shader", captured.output[0])
        with self.assertLogs("espi_live.qt", "INFO") as captured:
            qt_app.handle_qt_message(qt_app.QtMsgType.QtInfoMsg, None, "hello")
        self.assertIn("Qt: hello", captured.output[0])
        self.reporter.capture_message.assert_not_called()

    def test_critical_and_fatal_messages_are_reported(self):
        """Criticals report the message with the active Python stack.

        Args: None.
        Returns: None.
        """
        with self.assertLogs("espi_live.qt", "ERROR"):
            qt_app.handle_qt_message(qt_app.QtMsgType.QtCriticalMsg,
                                     SimpleNamespace(category="default"), "context lost")
        call = self.reporter.capture_message.call_args
        self.assertEqual(call.args[0], "critical")
        self.assertEqual(call.args[1], "context lost")
        self.assertIn("handle_qt_message", call.args[3])

    def test_fatal_messages_also_drop_the_crash_handler(self):
        """A fatal Qt message covers the coming abort, so faulthandler steps aside.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(qt_app, "faulthandler") as faulthandler, \
                self.assertLogs("espi_live.qt", "ERROR"):
            qt_app.handle_qt_message(qt_app.QtMsgType.QtFatalMsg, None, "aborting")
        self.assertEqual(self.reporter.capture_message.call_args.args[0], "fatal")
        faulthandler.disable.assert_called_once()


    def test_fatal_keeps_native_diagnostics_without_a_saved_report(self):
        """Disabled or failed reporting preserves faulthandler for the abort.

        Args: None.
        Returns: None.
        """
        self.reporter.capture_message.return_value = None
        with mock.patch.object(qt_app, "faulthandler") as handler, \
                self.assertLogs("espi_live.qt", "ERROR"):
            qt_app.handle_qt_message(qt_app.QtMsgType.QtFatalMsg, None, "aborting")
            with mock.patch.object(qt_app.error_reports, "_reporter", None):
                qt_app.handle_qt_message(qt_app.QtMsgType.QtFatalMsg, None, "aborting")
        handler.disable.assert_not_called()


class SettingOnTests(unittest.TestCase):
    """Reading QSettings flags stored as text."""

    def test_false_spellings(self):
        """False, 0 and their text forms are off.

        Args: None.
        Returns: None.
        """
        for value in (False, 0, "false", "0", " FALSE "):
            self.assertFalse(qt_app.setting_on(value), value)

    def test_everything_else_is_on(self):
        """True, numbers and unknown text are on.

        Args: None.
        Returns: None.
        """
        for value in (True, 1, "yes", None):
            self.assertTrue(qt_app.setting_on(value), value)


class ReportingClassTests(unittest.TestCase):
    """The run's Reporting: switches, hooks, Qt handler and worker."""

    def setUp(self):
        """Patch QSettings with the fake store and clear the level environment.

        Args: None.
        Returns: None.
        """
        FakeSettings.store = {}
        patcher = mock.patch.object(qt_app, "QSettings", FakeSettings)
        patcher.start()
        self.addCleanup(patcher.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        outputs = mock.patch.object(paths, "OUTPUT_DIR", tmp.name)
        outputs.start()
        self.addCleanup(outputs.stop)
        self.log = FakeLog()

    def reporting(self):
        """Build a Reporting for these tests.

        Args: None.
        Returns: The Reporting.
        """
        reporting = qt_app.Reporting(self.log)
        self.addCleanup(reporting.stop)
        return reporting

    def test_init_reads_the_saved_choice(self):
        """The stored choice decides the reporter's initial state.

        Args: None.
        Returns: None.
        """
        with mock.patch.dict(os.environ, {qt_app.error_reports.ENABLE_ENV: ""}):
            off = self.reporting()
            self.assertFalse(off.reporter.enabled)
            self.assertIn("source checkouts", off.forced)
        FakeSettings.store[qt_app.error_reports.SETTING_KEY] = "false"
        with mock.patch.dict(os.environ, {qt_app.error_reports.ENABLE_ENV: "1"}):
            self.assertFalse(self.reporting().reporter.enabled)
            FakeSettings.store.clear()
            self.assertTrue(self.reporting().reporter.enabled)
        self.assertEqual(off.reporter.context.get("front_end"), "Qt")
        self.assertIn(off.worker.kick, off.reporter.listeners)

    def test_start_installs_hooks_and_drains_old_crashes(self):
        """Starting hooks Python and Qt, logs the state and sends old crashes.

        Args: None.
        Returns: None.
        """
        reporting = self.reporting()
        handler = mock.MagicMock(return_value="PREVIOUS")
        with mock.patch.dict(os.environ, {qt_app.error_reports.ENABLE_ENV: ""}), \
                mock.patch.object(qt_app, "qInstallMessageHandler", handler), \
                mock.patch.object(reporting.reporter, "collect_native_crashes") as collect, \
                mock.patch.object(reporting.worker, "start") as start:
            try:
                reporting.start(self.log)
                self.assertIn("Error reports: off", self.log.lines[-1])
                collect.assert_called_once_with(None)
                start.assert_called_once()
                handler.assert_called_once_with(qt_app.handle_qt_message)
                self.assertEqual(reporting.previous_handler, "PREVIOUS")
                self.assertIsNotNone(reporting.hooks)
            finally:
                reporting.stop()
        handler.assert_called_with("PREVIOUS")
        self.assertIsNone(reporting.hooks)
        self.assertIsNone(qt_app.error_reports.current_reporter())

    def test_choose_records_the_choice_and_kicks(self):
        """The Help-menu choice is saved, applied and kicks the worker.

        Args: None.
        Returns: None.
        """
        with mock.patch.dict(os.environ, {qt_app.error_reports.ENABLE_ENV: "1"}):
            reporting = self.reporting()
            with mock.patch.object(reporting.worker, "kick") as kick:
                reporting.choose(False, self.log)
                self.assertFalse(FakeSettings.store[qt_app.error_reports.SETTING_KEY])
                self.assertFalse(reporting.reporter.enabled)
                self.assertIn("Error reports: off", self.log.lines[-1])
                kick.assert_not_called()
                reporting.choose(True, self.log)
                self.assertTrue(reporting.reporter.enabled)
                kick.assert_called_once()

    def test_stop_restores_the_previous_handler(self):
        """Stopping uninstalls the hooks and the Qt message handler.

        Args: None.
        Returns: None.
        """
        reporting = self.reporting()
        before = (sys.excepthook, threading.excepthook, sys.unraisablehook)
        with mock.patch.object(reporting.worker, "start"):
            reporting.start(self.log)
        self.assertNotEqual(sys.excepthook, before[0])
        reporting.stop()
        self.assertEqual((sys.excepthook, threading.excepthook, sys.unraisablehook), before)
        self.assertIsNone(qt_app.error_reports.current_reporter())
        with mock.patch.object(reporting.worker, "stop") as stop:
            reporting.stop()
        stop.assert_called_once()


class SessionDescriptionTests(unittest.TestCase):
    """describe_session's Spinnaker note."""

    def test_the_sdk_version_is_appended_when_known(self):
        """A loaded Spinnaker shows up in the session line.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(qt_app.camera, "spinnaker_version",
                               return_value="4.2.0.88"):
            line = qt_app.describe_session(
                [mock.MagicMock(demo=True, label="Synthetic camera (DEMO1)", active=True)],
                "GPU")
        self.assertIn("dummy mode, 1 synthetic camera(s)", line)
        self.assertIn("display GPU; Spinnaker 4.2.0.88", line)

    def test_mixed_channels_are_listed(self):
        """A camera channel is named, with its processing state.

        Args: None.
        Returns: None.
        """
        camera = mock.MagicMock(demo=False, label="[12345678] BFS-U3-63S4M", active=True)
        with mock.patch.object(qt_app.camera, "spinnaker_version", return_value=None):
            line = qt_app.describe_session([camera], "CPU")
        self.assertEqual(line, "Session: 1 camera(s): [12345678] BFS-U3-63S4M (processing)"
                               "; display CPU")


class ErrorReportUITests(WindowTestCase):
    """The window's error-report menu, status and dialogs."""

    def reporting(self, enabled=True, forced=None):
        """Build a reporting stand-in with a working event queue.

        Args: enabled and forced describe the reporter state.
        Returns: A MagicMock reporting.
        """
        reporting = mock.MagicMock(reporter=mock.MagicMock(enabled=enabled), forced=forced)
        reporting.worker = mock.MagicMock(events=queue.Queue())
        return reporting

    def test_the_action_mirrors_the_reporter(self):
        """The checkable action shows the state and explains itself.

        Args: None.
        Returns: None.
        """
        self.window.reporting = self.reporting()
        self.window._sync_reports_action()
        self.assertTrue(self.window.reports_action.isChecked())
        self.assertTrue(self.window.reports_action.isEnabled())
        self.assertIn("public GitHub issues", self.window.reports_action.toolTip())

    def test_a_forced_off_shows_the_reason(self):
        """When a switch forces reporting off, the action is locked.

        Args: None.
        Returns: None.
        """
        self.window.reporting = self.reporting(enabled=False,
                                               forced="turned off with --no-error-reports")
        self.window._sync_reports_action()
        self.assertFalse(self.window.reports_action.isChecked())
        self.assertFalse(self.window.reports_action.isEnabled())
        self.assertEqual(self.window.reports_action.toolTip(),
                         "Off: turned off with --no-error-reports")

    def test_the_menu_choice_reaches_reporting(self):
        """Toggling the action calls Reporting.choose and refreshes the menu.

        Args: None.
        Returns: None.
        """
        reporting = self.reporting()
        self.window.reporting = reporting
        self.window._choose_error_reports(True)
        reporting.choose.assert_called_once_with(True, self.window.log)
        self.assertTrue(self.window.reports_action.isChecked())

    def test_open_log_folder_opens_the_log_directory(self):
        """The log's folder opens in the desktop file manager.

        Args: None.
        Returns: None.
        """
        self.window.log.path = "/x/runs/espi_1.log"
        with mock.patch.object(qt_app, "QDesktopServices") as desktop:
            desktop.openUrl.return_value = True
            self.window._open_log_folder()
        self.assertEqual(desktop.openUrl.call_args.args[0].path(), "/x/runs")

    def test_open_log_folder_without_a_log_file(self):
        """Without a log file the default logs folder is opened.

        Args: None.
        Returns: None.
        """
        self.window.log.path = None
        with mock.patch.object(qt_app.paths, "output_folder", return_value="/fake/logs"), \
                mock.patch.object(qt_app, "QDesktopServices") as desktop:
            desktop.openUrl.return_value = True
            self.window._open_log_folder()
        self.assertEqual(desktop.openUrl.call_args.args[0].path(), "/fake/logs")

    def test_open_log_folder_falls_back_to_a_message(self):
        """If the desktop refuses, the path is shown in a dialog.

        Args: None.
        Returns: None.
        """
        self.window.log.path = "/x/runs/espi_1.log"
        with mock.patch.object(qt_app, "QDesktopServices") as desktop:
            desktop.openUrl.return_value = False
            self.window._open_log_folder()
        self.assertIn("/x/runs", self.dialogs.information.call_args.args[2])

    def test_open_log_folder_survives_an_unavailable_folder(self):
        """A failing folder lookup becomes a warning dialog.

        Args: None.
        Returns: None.
        """
        self.window.log.path = None
        with mock.patch.object(qt_app.paths, "output_folder",
                               side_effect=OSError("gone")):
            self.window._open_log_folder()
        self.assertIn("unavailable", self.dialogs.warning.call_args.args[2])

    def test_report_context_describes_demo_mode(self):
        """Demo runs say so and list the demo serials.

        Args: None.
        Returns: None.
        """
        reporting = self.reporting()
        self.window.reporting = reporting
        with mock.patch.object(qt_app.camera, "spinnaker_version", return_value=None):
            self.window._update_report_context()
        call = reporting.reporter.update_context.call_args
        self.assertEqual(call.kwargs["source"], "dummy mode")
        self.assertEqual(call.kwargs["cameras"], 1)
        self.assertEqual(call.kwargs["serials"], ["DEMO1"])

    def test_report_context_describes_the_camera(self):
        """A camera run names the model and scrubs its serials.

        Args: None.
        Returns: None.
        """
        reporting = self.reporting()
        self.window.reporting = reporting
        camera = mock.MagicMock(demo=False, identity=None)
        camera.info = SimpleNamespace(serial="SN-1", model="")
        original, original_channels = self.window.current, self.window.channels
        self.addCleanup(setattr, self.window, "current", original)
        self.addCleanup(setattr, self.window, "channels", original_channels)
        self.window.current = camera
        self.window.channels = [camera]
        with mock.patch.object(qt_app.camera, "spinnaker_version", return_value="4.2"):
            self.window._update_report_context()
        call = reporting.reporter.update_context.call_args
        self.assertEqual(call.kwargs["source"], "camera (model unknown)")
        self.assertEqual(call.kwargs["serials"], ["SN-1"])
        self.assertEqual(call.kwargs["spinnaker"], "4.2")

        camera.identity = SimpleNamespace(short_name=lambda: "BFS-U3-63S4M", serial="SN-1")
        with mock.patch.object(qt_app.camera, "spinnaker_version", return_value="4.2"):
            self.window._update_report_context()
        call = reporting.reporter.update_context.call_args
        self.assertEqual(
            call.kwargs["source"], "camera BFS-U3-63S4M")
        self.assertEqual(call.kwargs["serials"], ["SN-1", "SN-1"])

    def test_report_context_accepts_analog_capture_identity(self):
        """An analog identity has no hardware serial and must not break polling.

        Args: None.
        Returns: None.
        """
        from espi_live.capture_models import CaptureIdentity
        reporting = self.reporting()
        self.window.reporting = reporting
        channel = SimpleNamespace(demo=False, identity=CaptureIdentity(name="USB-Live2"),
                                  info=CameraInfo("video 0", "USB-Live2", kind="capture"))
        with mock.patch.object(self.window, "current", channel), \
                mock.patch.object(self.window, "channels", [channel]):
            self.window._update_report_context()
        context = reporting.reporter.update_context.call_args.kwargs
        self.assertEqual(context["serials"], ["video 0"])
        self.assertEqual(context["source"], "camera USB-Live2")

    def test_report_outcomes_reach_the_status_bar(self):
        """A delivered report shows its outcome with the issue link.

        Args: None.
        Returns: None.
        """
        reporting = self.reporting()
        self.window.reporting = reporting
        outcome = qt_app.error_reports.Outcome(FINGERPRINT_UI, "created", 5,
                                               "https://github.com/owner/repo/issues/5")
        reporting.worker.events.put(("outcome", outcome))
        with mock.patch.object(self.window.error_report_timer, "start") as timer:
            self.window._handle_error_report_events()
        self.assertEqual(self.window.report_outcomes[FINGERPRINT_UI], outcome)
        self.assertIn("Reported as <a href=", self.window.error_report_label.text())
        timer.assert_called_once()
        self.assertIn(FINGERPRINT_UI, self.window.error_report_label.toolTip())

    def test_report_outcomes_fill_an_open_fatal_dialog(self):
        """While the "ESPI stopped" dialog is open its line is updated in place.

        Args: None.
        Returns: None.
        """
        reporting = self.reporting()
        self.window.reporting = reporting
        box = mock.MagicMock()
        self.window.fatal_boxes[FINGERPRINT_UI] = (box, "kaboom")
        outcome = qt_app.error_reports.Outcome(FINGERPRINT_UI, "created", 5,
                                               "https://github.com/owner/repo/issues/5")
        reporting.worker.events.put(("outcome", outcome))
        self.window._handle_error_report_events()
        box.setText.assert_called_once()
        text = box.setText.call_args.args[0]
        self.assertIn("kaboom", text)
        self.assertIn("Reported as", text)
        self.assertIn(FINGERPRINT_UI, text)
        self.assertFalse(self.window.error_report_label.isVisibleTo(self.window))

    def test_outcome_html_links_the_issue(self):
        """The outcome line becomes a link when there is an issue.

        Args: None.
        Returns: None.
        """
        html = qt_app.MainWindow._outcome_html
        self.assertEqual(html(None), "Reporting this error…")
        plain = qt_app.error_reports.Outcome(FINGERPRINT_UI, "saved")
        self.assertEqual(html(plain), "Report saved; it will be sent next time")
        linked = qt_app.error_reports.Outcome(FINGERPRINT_UI, "fixed", url="https://x/y")
        self.assertEqual(html(linked), "Fixed in a newer build; update to get the fix")
        browser = qt_app.error_reports.Outcome(FINGERPRINT_UI, "browser",
                                               url="https://x/y?a=b&c=d")
        self.assertEqual(html(browser),
                         'Not sent automatically: <a href="https://x/y?a=b&amp;c=d">'
                         'Report in browser…</a>')
        created = qt_app.error_reports.Outcome(FINGERPRINT_UI, "created", 5,
                                               "https://github.com/o/r/issues/5")
        self.assertEqual(html(created),
                         'Reported as <a href="https://github.com/o/r/issues/5">#5</a>')

    def test_fatal_text_gains_the_report_line(self):
        """The stopped-dialog text carries the report state and the error ID.

        Args: None.
        Returns: None.
        """
        self.assertEqual(self.window._fatal_text("kaboom", None), "kaboom")
        outcome = qt_app.error_reports.Outcome(FINGERPRINT_UI, "saved")
        self.window.report_outcomes[FINGERPRINT_UI] = outcome
        text = self.window._fatal_text("kaboom", FINGERPRINT_UI)
        self.assertTrue(text.startswith("kaboom<br><br>Report saved"))
        self.assertIn(f"Error ID {FINGERPRINT_UI}", text)

    def test_show_fatal_registers_its_dialog(self):
        """While a fatal dialog is open its fingerprint is tracked, then freed.

        Args: None.
        Returns: None.
        """
        seen = {}

        def fake_exec():
            """Record what the window tracked while the dialog was open.

            Args: None.
            Returns: None.
            """
            seen.update(self.window.fatal_boxes)

        self.dialogs.return_value.exec.side_effect = fake_exec
        self.window._show_fatal("kaboom", FINGERPRINT_UI)
        self.assertEqual(seen[FINGERPRINT_UI][1], "kaboom")
        self.assertEqual(self.window.fatal_boxes, {})
        self.assertEqual(self.dialogs.call_args.args[1:],
                         ("ESPI stopped", self.window._fatal_text("kaboom", FINGERPRINT_UI)))

    def test_report_status_variants(self):
        """Each outcome kind shows the right status-bar text.

        Args: None.
        Returns: None.
        """
        cases = {
            "declined": None,
            "fixed": "Check for updates",
            "saved": "will be reported next time",
            "created": "Unexpected error: Reported as"}
        for kind, needle in cases.items():
            outcome = qt_app.error_reports.Outcome(FINGERPRINT_UI, kind, 5,
                                                   "https://github.com/o/r/issues/5")
            self.window.error_report_label.setVisible(False)
            with mock.patch.object(self.window.error_report_timer, "start") as timer:
                self.window._show_report_status(outcome)
            if needle is None:
                self.assertFalse(self.window.error_report_label.isVisibleTo(self.window))
                timer.assert_not_called()
                continue
            self.assertIn(needle, self.window.error_report_label.text())
            timer.assert_called_once()


class FreezeTests(WindowTestCase):
    """Freezing the displayed image without saving it (issue #14)."""

    def show(self, packet):
        """Publish a packet from the engine and let the window draw it.

        Args: packet is the FramePacket to publish.
        Returns: None.
        """
        self.engine._packet = packet
        self.window._poll_frame()

    def freeze_action(self):
        """Find View > Freeze display.

        Args: None.
        Returns: The QAction.
        """
        view_menu = self.window.menuBar().actions()[0].menu()
        return next(a for a in view_menu.actions() if a.text() == "Freeze display")

    def freeze_on_first_frame(self):
        """Draw frame 1, freeze it, then publish a newer frame 2.

        Args: None.
        Returns: The frozen FramePacket.
        """
        frozen = make_packet(sequence=1)
        self.show(frozen)
        self.window.freeze_button.click()
        self.show(make_packet(sequence=2, palette="Jet"))
        return frozen

    def test_freeze_holds_the_shown_frame_while_the_engine_continues(self):
        """Newer engine frames are not drawn while the display is frozen.

        Args: None.
        Returns: None.
        """
        frozen = self.freeze_on_first_frame()
        self.assertIs(self.window.view.packet, frozen)
        self.assertIs(self.window.current.frozen, frozen)
        self.assertEqual(self.window.drawn_frames, 1)
        self.assertTrue(self.log.lines[-1].startswith("Display frozen"))

    def test_unfreeze_returns_to_the_newest_frame(self):
        """Resuming draws the engine's current frame on the next poll.

        Args: None.
        Returns: None.
        """
        self.freeze_on_first_frame()
        self.window.freeze_button.click()
        self.assertIsNone(self.window.current.frozen)
        self.window._poll_frame()
        self.assertEqual(self.window.view.packet.sequence, 2)
        self.assertEqual(self.log.lines[-1], "Display live again.")

    def test_freeze_without_a_frame_logs_and_stays_live(self):
        """Freezing before anything was drawn does nothing but explain why.

        Args: None.
        Returns: None.
        """
        self.window.freeze_button.click()
        self.assertIsNone(self.window.current.frozen)
        self.assertFalse(self.window.freeze_button.isChecked())
        self.assertTrue(self.window.frozen_banner.isHidden())
        self.assertEqual(self.log.lines[-1], "No frame on screen to freeze yet.")

    def test_repeated_freeze_or_resume_changes_nothing(self):
        """Freezing a frozen view, or resuming a live one, is a no-op.

        Args: None.
        Returns: None.
        """
        frozen = self.freeze_on_first_frame()
        lines = len(self.log.lines)
        self.window.set_frozen(True)
        self.assertIs(self.window.current.frozen, frozen)
        self.window.set_frozen(False)
        self.window.set_frozen(False)
        self.assertEqual(len(self.log.lines), lines + 1)

    def test_frozen_indicators_show_and_hide(self):
        """Banner, badge, button and menu item all mark a frozen view.

        Args: None.
        Returns: None.
        """
        window = self.window
        action = self.freeze_action()
        for widget in (window.frozen_banner, window.frozen_badge):
            self.assertTrue(widget.isHidden())
        self.freeze_on_first_frame()
        when = window.current.frozen_at.strftime("%H:%M:%S")
        self.assertFalse(window.frozen_banner.isHidden())
        self.assertIn(f"FROZEN at {when}", window.frozen_banner.text())
        self.assertIn("Space", window.frozen_banner.text())
        self.assertFalse(window.frozen_badge.isHidden())
        self.assertTrue(window.freeze_button.isChecked())
        self.assertTrue(action.isChecked())
        window.set_frozen(False)
        for widget in (window.frozen_banner, window.frozen_badge):
            self.assertTrue(widget.isHidden())
        self.assertFalse(window.freeze_button.isChecked())
        self.assertFalse(action.isChecked())

    def test_frozen_badge_moves_no_other_status_item(self):
        """Showing or hiding the FROZEN badge leaves every other item in place.

        The badge takes its room from the hint text, so the window is made wide
        enough for the hint to have some.

        Args: None.
        Returns: None.
        """
        window = self.window
        window.resize(2400, 900)
        window.show()
        items = (window.update_slot, window.feature_button, window.source_badge,
                 window.record_label, window.range_status, window.camera_fps_status,
                 window.speed_status, window.pixel_status)

        def geometry():
            QApplication.processEvents()
            return [(w.x(), w.width()) for w in items]

        before = geometry()
        self.freeze_on_first_frame()
        self.assertFalse(window.frozen_badge.isHidden())
        self.assertEqual(geometry(), before)
        window.set_frozen(False)
        self.assertEqual(geometry(), before)

    def test_banner_lets_mouse_reach_the_image(self):
        """The banner over the image does not block zoom, pan or the readout.

        Args: None.
        Returns: None.
        """
        self.assertTrue(self.window.frozen_banner.testAttribute(
            Qt.WA_TransparentForMouseEvents))

    def test_space_key_toggles_freeze(self):
        """Space freezes and resumes the display.

        Args: None.
        Returns: None.
        """
        self.show(make_packet(sequence=1))
        self.assertTrue(self.window.handle_key(Qt.Key_Space))
        self.assertIsNotNone(self.window.current.frozen)
        self.assertTrue(self.window.handle_key(Qt.Key_Space))
        self.assertIsNone(self.window.current.frozen)

    def test_view_menu_item_toggles_freeze(self):
        """View > Freeze display freezes and resumes.

        Args: None.
        Returns: None.
        """
        self.show(make_packet(sequence=1))
        action = self.freeze_action()
        self.assertEqual(action.shortcut().toString(), "Space")
        action.trigger()
        self.assertIsNotNone(self.window.current.frozen)
        action.trigger()
        self.assertIsNone(self.window.current.frozen)

    def test_save_while_frozen_sends_the_frozen_frame(self):
        """Save writes the held frame and its freeze time, not the live one.

        Args: None.
        Returns: None.
        """
        frozen = self.freeze_on_first_frame()
        self.window.frame_title.setText("Held")
        self.window.save_frame()
        self.assertEqual(self.queued(), [("save", {
            "title": "Held", "notes": "", "packet": frozen,
            "frozen_at": self.window.current.frozen_at})])

    def test_pixel_readout_reads_the_frozen_frame(self):
        """The hover readout reports the frozen image, not the live one.

        Args: None.
        Returns: None.
        """
        frozen = self.freeze_on_first_frame()
        self.engine._packet = FramePacket(3, np.zeros((48, 64), np.uint8), "Gray",
                                          1.0, "Raw", True)
        self.window.hover = (10, 5)
        self.window._poll_status()
        self.assertEqual(self.window.pixel_status.text(),
                         f"x 10, y 5: {int(frozen.image[5, 10])}")

    def test_displayed_rate_reads_frozen(self):
        """The displayed-frame rate says frozen instead of a number.

        Args: None.
        Returns: None.
        """
        self.freeze_on_first_frame()
        self.window._poll_status()
        self.assertIn("Displayed: frozen", self.window.speed_status.text())

    def test_switch_to_live_drops_a_frozen_demo_frame(self):
        """A held demo frame never outlives the dummy-mode indicators.

        Args: None.
        Returns: None.
        """
        self.freeze_on_first_frame()
        self.engine.events.put(("switched_live", None))
        self.window._handle_events()
        self.assertIsNone(self.window.current.frozen)
        self.assertTrue(self.window.frozen_banner.isHidden())
        self.assertFalse(self.window.freeze_button.isChecked())


class CodexTests(WindowTestCase):
    """Open in Codex: getting the source, the first-use dialog and opening the
    app, with the source worker, settings and dialogs faked."""

    FOLDER = os.path.join(tempfile.gettempdir(), "espi-live-source")

    def setUp(self):
        """Build the window with a source worker that never starts a thread.

        Args: None.
        Returns: None.
        """
        super().setUp()
        self.window.source = mock.MagicMock(busy=False, events=queue.Queue())
        self.window.source.start.return_value = True
        patcher = mock.patch.object(qt_app.codex, "write_guide", return_value=None)
        self.write_guide = patcher.start()
        self.addCleanup(patcher.stop)

    def ready(self, note=""):
        """Deliver a "source ready" event.

        Args: note is the event's log line.
        Returns: None.
        """
        self.window.source.events.put(("ready", (self.FOLDER, note)))
        self.window._handle_source_events()

    def buttons(self, choice, count):
        """Make the next message box report one of its added buttons as clicked.

        Args: choice is the index of the clicked button; count is how many
            buttons the box adds.
        Returns: None.
        """
        box = self.dialogs.return_value
        made = [object() for _ in range(count)]
        box.addButton.side_effect = made
        box.clickedButton.return_value = made[choice]

    def test_entry_points_exist(self):
        """The status bar button and the Help menu entry both open Codex.

        Args: None.
        Returns: None.
        """
        help_menu = self.window.menuBar().actions()[-1].menu()
        entry = next(a for a in help_menu.actions()
                     if a.text() == "Ask Codex about the source code…")
        with mock.patch.object(self.window, "_open_codex") as open_codex:
            entry.trigger()
            self.window.codex_button.click()
        self.assertEqual(open_codex.call_count, 2)
        self.assertEqual(self.window.codex_button.text(), "Open in Codex")

    def test_startup_gets_the_source_only_with_update_checks(self):
        """The startup check runs unless GitHub is off limits at startup.

        Args: None.
        Returns: None.
        """
        worker = mock.MagicMock()
        with mock.patch.object(qt_app.source_code, "SourceWorker", return_value=worker):
            for check_updates in (False, True):
                engine = Engine(self.log)
                engine.use_demo()
                window = demo_window(engine, self.log, check_updates=check_updates)
                window.frame_timer.stop()
                window.status_timer.stop()
                window.tip_timer.stop()
                window.deleteLater()
        worker.start.assert_called_once_with()

    def test_click_before_the_source_is_ready_downloads_then_opens(self):
        """Without the source the button waits for it, then opens Codex.

        Args: None.
        Returns: None.
        """
        self.window._open_codex()
        self.assertFalse(self.window.codex_button.isEnabled())
        self.assertEqual(self.window.codex_button.text(), "Getting source…")
        self.window.source.start.assert_called_once_with()
        FakeSettings.store["codex/intro_shown"] = True
        with mock.patch.object(qt_app.QDesktopServices, "openUrl",
                               return_value=True) as open_url:
            self.ready("Downloaded the source code")
        self.assertTrue(self.window.codex_button.isEnabled())
        self.assertEqual(self.window.codex_button.text(), "Open in Codex")
        self.assertIn("Downloaded the source code", self.log.lines)
        self.write_guide.assert_called_once_with(self.FOLDER)
        url = open_url.call_args.args[0]
        self.assertEqual(url.toEncoded().data().decode(), qt_app.codex.deep_link(self.FOLDER))
        self.assertIn(f"Opened Codex on {self.FOLDER}", self.log.lines)
        self.assertIn("Opening Codex", self.window.statusBar().currentMessage())

    def test_ready_source_at_startup_opens_nothing(self):
        """The startup event only records the folder.

        Args: None.
        Returns: None.
        """
        before = list(self.log.lines)
        with mock.patch.object(qt_app.QDesktopServices, "openUrl") as open_url:
            self.ready()
        self.assertEqual(self.window.source_folder, self.FOLDER)
        open_url.assert_not_called()
        self.assertEqual(self.log.lines, before)

    def test_click_while_downloading_waits(self):
        """A click during the startup download opens Codex when it finishes.

        Args: None.
        Returns: None.
        """
        self.window.source.start.return_value = False
        self.window._open_codex()
        self.assertTrue(self.window.codex_when_ready)
        self.assertIn("Still downloading", self.window.statusBar().currentMessage())

    def test_failed_download(self):
        """A failure is logged, and reported in a dialog only after a click.

        Args: None.
        Returns: None.
        """
        self.window.source.events.put(("failed", "offline"))
        self.window._handle_source_events()
        self.assertIn("Source code not available: offline", self.log.lines)
        self.dialogs.warning.assert_not_called()
        self.window._open_codex()
        self.window.source.events.put(("failed", "R&D <offline>"))
        self.window._handle_source_events()
        self.dialogs.warning.assert_called_once()
        self.assertIn("could not be downloaded: R&amp;D &lt;offline&gt;",
                      self.dialogs.warning.call_args.args[2])
        self.assertTrue(self.window.codex_button.isEnabled())
        self.assertIsNone(self.window.source_folder)

    def test_first_use_explains_and_remembers(self):
        """The first open asks first; cancelling opens nothing and asks again.

        Args: None.
        Returns: None.
        """
        self.ready()
        with mock.patch.object(qt_app.QDesktopServices, "openUrl",
                               return_value=True) as open_url:
            with mock.patch.object(self.window, "_codex_intro", return_value=False):
                self.window._open_codex()
            open_url.assert_not_called()
            self.assertNotIn("codex/intro_shown", FakeSettings.store)
            with mock.patch.object(self.window, "_codex_intro", return_value=True) as intro:
                self.window._open_codex()
                self.window._open_codex()
        intro.assert_called_once_with()
        self.assertTrue(FakeSettings.store["codex/intro_shown"])
        self.assertEqual(open_url.call_count, 2)

    def test_intro_buttons(self):
        """Open Codex goes on; Get Codex opens the download page; Cancel stops.

        Args: None.
        Returns: None.
        """
        self.ready()
        self.buttons(0, 3)
        self.assertTrue(self.window._codex_intro())
        self.assertIn(self.FOLDER, self.dialogs.call_args.args[2])
        self.buttons(2, 3)
        self.assertFalse(self.window._codex_intro())
        self.buttons(1, 3)
        with mock.patch.object(qt_app.QDesktopServices, "openUrl") as open_url:
            self.assertFalse(self.window._codex_intro())
        self.assertEqual(open_url.call_args.args[0].toString(), qt_app.codex.INSTALL_URL)

    def test_guide_that_cannot_be_written_warns(self):
        """A read-only source folder is reported and Codex is not opened.

        Args: None.
        Returns: None.
        """
        self.ready()
        FakeSettings.store["codex/intro_shown"] = True
        self.write_guide.side_effect = PermissionError("read-only")
        with mock.patch.object(qt_app.QDesktopServices, "openUrl") as open_url:
            self.window._open_codex()
        open_url.assert_not_called()
        self.assertIn("read-only", self.dialogs.warning.call_args.args[2])

    def test_missing_codex_offers_the_download(self):
        """When no app takes the link, a dialog offers the download page.

        Args: None.
        Returns: None.
        """
        self.ready()
        FakeSettings.store["codex/intro_shown"] = True
        for choice, opened in ((1, 1), (0, 2)):
            self.buttons(choice, 2)
            with mock.patch.object(qt_app.QDesktopServices, "openUrl",
                                   return_value=False) as open_url:
                self.window._open_codex()
            self.assertEqual(open_url.call_count, opened)
            self.assertEqual(self.dialogs.call_args.args[1], "Codex not found")
        self.assertEqual(open_url.call_args.args[0].toString(), qt_app.codex.INSTALL_URL)


class HelpTipTests(WindowTestCase):
    """Rotating status-bar tips (issue #21)."""

    def test_a_tip_is_showing_and_the_timer_is_slow(self):
        """The window opens on one tip, and the timer is not per-frame.

        Args: None.
        Returns: None.
        """
        window = self.window
        self.assertIn(window.tip_label.text(), tips.TIPS)
        self.assertFalse(window.tip_label.isHidden())
        self.assertTrue(window.tips_action.isChecked())
        self.assertEqual(window.tip_timer.interval(), tips.ROTATE_MS)
        self.assertEqual(window.tip_timer.timerType(), Qt.CoarseTimer)
        self.assertTrue(window.tip_timer.isActive())
        self.assertIn(window.tip_label.text(), window.tip_label.toolTip())
        self.assertIn("Right-click to hide.", window.tip_label.toolTip())

    def test_click_and_timer_advance_and_the_last_tip_wraps(self):
        """A click and a timeout both move to the next tip, and the end wraps.

        Args: None.
        Returns: None.
        """
        window = self.window
        window._poll_status()
        hint = window.hint_label.text()
        self.assertEqual(hint, self.engine.status()["hint"])
        index = window.tips.index
        event = mock.MagicMock()
        event.button.return_value = Qt.LeftButton
        window.tip_label.mousePressEvent(event)
        self.assertEqual(window.tips.index, (index + 1) % len(tips.TIPS))
        self.assertEqual(window.tip_label.text(), tips.TIPS[window.tips.index])
        self.assertIn("Click for the next tip.", window.tip_label.toolTip())
        self.assertEqual(window.hint_label.text(), hint)
        window.tips.index = len(tips.TIPS) - 1
        window.tip_timer.timeout.emit()
        self.assertEqual(window.tips.index, 0)
        self.assertEqual(window.tip_label.text(), tips.TIPS[0])
        self.assertEqual(window.hint_label.text(), hint)

    def test_narrow_tip_elides_but_keeps_the_full_text(self):
        """A tip squeezed to a few pixels shows an ellipsis and reports the whole string.

        Args: None.
        Returns: None.
        """
        label = qt_app.TipLabel(*tips.TIPS)
        self.addCleanup(label.close)
        label.setText(tips.TIPS[0])
        label.resize(30, 20)
        label.setText(tips.TIPS[0])
        self.assertEqual(label.text(), tips.TIPS[0])
        self.assertTrue(QLabel.text(label).endswith("…"))
        label.resize(label.sizeHint().width(), 20)
        label.setText(tips.TIPS[0])
        self.assertEqual(QLabel.text(label), tips.TIPS[0])

    def test_hiding_persists_and_a_late_timeout_stays_hidden(self):
        """Hide remembers the choice; the next window stays quiet until View > Help tips.

        Args: None.
        Returns: None.
        """
        window = self.window
        index = window.tips.index
        event = mock.MagicMock()
        event.button.return_value = Qt.RightButton
        window.tip_label.mousePressEvent(event)
        self.assertTrue(window.tip_label.isHidden())
        self.assertFalse(window.tips_action.isChecked())
        self.assertFalse(window.tip_timer.isActive())
        self.assertEqual(window.tip_label.toolTip(), "")
        self.assertTrue(FakeSettings.store["tips/hidden"])
        window.tip_timer.timeout.emit()
        self.assertEqual(window.tips.index, index)
        window.close()

        engine = Engine(self.log)
        engine.use_demo()
        again = demo_window(engine, self.log)
        self.addCleanup(again.deleteLater)
        self.addCleanup(again.status_timer.stop)
        self.addCleanup(again.frame_timer.stop)
        self.addCleanup(again.tip_timer.stop)
        self.addCleanup(again.close)
        self.assertTrue(again.tip_label.isHidden())
        self.assertFalse(again.tips_action.isChecked())
        self.assertFalse(again.tip_timer.isActive())
        self.assertTrue(again.tips.hidden)
        again.tips_action.trigger()
        self.assertFalse(again.tip_label.isHidden())
        self.assertIn(again.tip_label.text(), tips.TIPS)
        self.assertFalse(FakeSettings.store["tips/hidden"])
        self.assertTrue(again.tip_timer.isActive())

    def test_view_menu_hides_tips_without_moving_the_dummy_badge(self):
        """Unchecking Help tips hides them, and the dummy-mode badge stays whole.

        Args: None.
        Returns: None.
        """
        window = self.window
        window.show()
        QApplication.processEvents()
        self.assertEqual(QLabel.text(window.source_badge), "  DUMMY MODE  ")
        for tip in tips.TIPS:
            window.tip_label.setText(tip)
            self.assertEqual(QLabel.text(window.tip_label), tip)
        before = window.source_badge.geometry().getRect()
        window.resize(1400, 900)  # a saved narrow layout keeps the badge
        QApplication.processEvents()
        self.assertIn("DUMMY", QLabel.text(window.source_badge))
        window.resize(qt_app.DEFAULT_WIDTH, qt_app.DEFAULT_HEIGHT)
        QApplication.processEvents()
        window.tips_action.trigger()
        QApplication.processEvents()
        self.assertTrue(window.tip_label.isHidden())
        self.assertTrue(FakeSettings.store["tips/hidden"])
        self.assertEqual(window.source_badge.geometry().getRect(), before)


if __name__ == "__main__":
    unittest.main()
