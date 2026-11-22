"""Camera-free checks for the Qt front end (offscreen platform, demo engine).

Run from the project root:
    PYTHONPATH=src QT_QPA_PLATFORM=offscreen python -m unittest tests.test_qt_app

No camera, network, real dialog or real settings store is touched: the engine
uses its demo source, the updater worker is replaced by a fake, QSettings is
replaced by a dictionary and every QMessageBox call is patched.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import logging
import queue
import sys
import tempfile
import threading
import time
import types
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, QSize, Qt
from PySide6.QtGui import QFont, QHelpEvent, QKeyEvent, QKeySequence, QScreen, QSurfaceFormat
from PySide6.QtWidgets import (QAbstractButton, QAbstractScrollArea, QAbstractSlider,
                               QAbstractSpinBox, QApplication, QComboBox, QLabel, QLineEdit,
                               QPlainTextEdit, QPushButton, QStyle, QTextEdit, QToolTip, QWidget)

from espi_live import (camera, camera_reports, capture, feedback, iphone, paths, qt_app,
                       sources, tips,
                       ui_scale, updater)
from fake_capture import (FakeCaptureFactory, FakeEnumerator, integrated_webcam,
                          interlaced_frame, usb_live2, webcam)
from fake_iphone import commands, lens_socket, phone_socket
from fake_pyspin import FakeCamera, FakeSystem, make_pyspin
from espi_live.camera import CameraInfo
from espi_live.camera_models import CameraIdentity
from espi_live.channels import Channel, demo_channels
from espi_live.engine import ComparePacket, Engine, FramePacket
from espi_live.pixel_square import SquarePacket
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
    syncs = 0

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

    def remove(self, key):
        """Forget a value.

        Args: key is the setting name.
        Returns: None.
        """
        self.store.pop(key, None)

    def allKeys(self):
        """List every stored key.

        Args: None.
        Returns: A list of setting names.
        """
        return list(self.store)

    def sync(self):
        """Count a request to write the store to disk.

        Args: None.
        Returns: None.
        """
        FakeSettings.syncs += 1


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


def make_packet(sequence=1, width=64, height=48, gamma=1.0):
    """Build a FramePacket with a horizontal ramp image.

    Args: sequence, width, height and gamma describe the frame.
    Returns: A FramePacket.
    """
    image = np.tile(np.arange(width, dtype=np.uint8) * 3, (height, 1))
    return FramePacket(sequence, image, gamma, "Raw", True)


def wait_for(predicate, timeout=5.0):
    """Poll a condition until it is true or a timeout passes.

    Args: predicate is a zero-argument callable; timeout is in seconds.
    Returns: The final value of predicate().
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.01)
    return predicate()


def demo_window(engine, log, use_gl=False, check_updates=False, simple_ui=False):
    """Build a MainWindow around one demo engine.

    Most tests exercise every control, so the simple UI (which only renders the
    basic ones) is off unless a test asks for it.

    Args: engine is a demo Engine (not started); log is the FakeLog; use_gl
        and check_updates are passed to MainWindow; simple_ui is the value of
        qt_app.SIMPLE_UI while the window is built.
    Returns: The MainWindow.
    """
    channel = Channel(log, CameraInfo("DEMO1", "Synthetic camera"), demo=True,
                      engine=engine)
    with mock.patch.object(qt_app, "SIMPLE_UI", simple_ui):
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
        packet = FramePacket(1, image, 1.0, "Raw", True)
        self.view.show_packet(packet, self.table, None)
        self.assertEqual(self.center_pixel(), 200)

    def test_zoomed_in_small_image_uses_nearest_neighbor(self):
        """A small image enlarged on screen is drawn with the table colors.

        Args: None.
        Returns: None.
        """
        image = np.full((20, 20), 90, np.uint8)
        packet = FramePacket(1, image, 1.0, "Raw", True)
        self.view.show_packet(packet, self.table, None)
        self.view.set_zoom(3.0)
        self.assertEqual(self.center_pixel(), 90)

    def test_empty_visible_crop_draws_nothing_extra(self):
        """A view center far outside the image leaves only the background.

        Args: None.
        Returns: None.
        """
        image = np.full((100, 100), 255, np.uint8)
        packet = FramePacket(1, image, 1.0, "Raw", True)
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
    simple_ui = False

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
        self.window = demo_window(self.engine, self.log, self.use_gl,
                                  simple_ui=self.simple_ui)
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
        self.assertEqual(self.window.windowTitle(), updater.app_title())
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

    def test_first_start_after_update_shows_release_notes(self):
        """Notes left by an update open in a dialog once the window is up.

        Args: None.
        Returns: None.
        """
        notes = updater.WhatsNew(4, 6, (
            updater.ReleaseNote(6, "Build 6", "Faster [recording](https://x.invalid/r)",
                                "https://x.invalid/6"),))
        shown = []

        def fake_exec(dialog):
            """Record the dialog instead of blocking.

            Args: dialog is the WhatsNewDialog.
            Returns: 0.
            """
            shown.append((dialog.windowTitle(), dialog.notes.toPlainText(),
                          dialog.notes.openExternalLinks()))
            return 0

        with mock.patch.object(qt_app.updater, "take_whats_new", return_value=notes):
            window = demo_window(self.engine, self.log)
        self.addCleanup(window.deleteLater)
        window.frame_timer.stop()
        window.status_timer.stop()
        self.assertEqual(shown, [])  # waits for the event loop
        with mock.patch.object(qt_app.WhatsNewDialog, "exec", fake_exec):
            APP.processEvents()
        window.close()
        self.assertEqual(len(shown), 1)
        title, text, external = shown[0]
        self.assertEqual(title, "ESPI live was updated")
        self.assertIn("Faster recording", text)
        self.assertTrue(external)
        self.assertIn("Updated from build 4 to 6", self.log.lines)

    def test_normal_start_shows_no_release_notes(self):
        """Without notes from an update no dialog is scheduled.

        Args: None.
        Returns: None.
        """
        self.assertIsNone(self.window.whats_new)

    def test_mode_button_changes_engine_mode(self):
        """Clicking a view-mode button updates the engine's mode.

        Args: None.
        Returns: None.
        """
        button = next(b for b in self.window.mode_group.buttons() if b.text() != "Raw")
        button.click()
        self.assertEqual(self.engine.settings().mode, button.text())

    def test_raw_difference_button_overrides_the_view(self):
        """Raw difference turns on in the engine and greys out the view buttons.

        Loading settings (for example on a camera switch) keeps the button
        and the view buttons in step; turning it off gives the views back.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.raw_difference_button.setChecked(True)
        self.assertTrue(self.engine.settings().raw_difference)
        self.assertFalse(any(b.isEnabled() for b in w.mode_group.buttons()))
        self.engine.update_settings(raw_difference=False)
        w._load_settings(self.engine.settings(), w.current.limits)
        self.assertFalse(w.raw_difference_button.isChecked())
        self.assertTrue(all(b.isEnabled() for b in w.mode_group.buttons()))
        self.assertIn("|live - previous|", w.raw_difference_button.toolTip())

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
        self.assertIn("Auto stretch", tip)
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
        self.assertIn("Scaling", qt_app.GAMMA_TIP)
        self.assertIn("Scaling", tip)
        self.assertIn("1.0", tip)
        self.assertIn("Now: 1.0", tip)
        self.assertEqual(gamma.slider.toolTip(), tip)
        self.assertEqual(gamma.spin.toolTip(), tip)
        labels = [label for label in self.window.controls_dock.findChildren(QLabel)
                  if label.text() == "Gamma"]
        self.assertEqual([label.toolTip() for label in labels], [tip])
        stretch = self.window.scaling.toolTip()
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
        w.scaling.setCurrentText("Fixed gain" if before.scale_range else "Auto stretch")
        w.gain_before_average.setChecked(not before.gain_before_average)
        w.invert.setChecked(not before.invert)
        w.flip_button.setChecked(not before.flip_horizontal)
        w.smoothing_combo.setCurrentIndex(w.smoothing_combo.count() - 1)
        after = self.engine.settings()
        self.assertEqual(after.rolling, not before.rolling)
        self.assertEqual(after.scale_range, not before.scale_range)
        self.assertEqual(after.gain_before_average, not before.gain_before_average)
        self.assertEqual(after.invert, not before.invert)
        self.assertEqual(after.flip_horizontal, not before.flip_horizontal)
        self.assertEqual(after.smoothing_method, w.smoothing_combo.currentText())

    def test_fringe_and_display_controls_update_engine(self):
        """Sensitivity, threshold, steady range, peak hold, normalization, tone.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.min_range.spin.setValue(12)
        w.threshold.spin.setValue(4)
        w.steady_range.spin.setValue(0.5)
        w.even_brightness.setChecked(True)
        w.peak_hold.spin.setValue(3)
        w.normalize_window.spin.setValue(41)
        w.brightness.spin.setValue(6)
        w.display_gain.spin.setValue(1.5)
        w.contrast.spin.setValue(20)
        s = self.engine.settings()
        self.assertEqual((s.min_range, s.threshold, s.steady_range_s), (12, 4, 0.5))
        self.assertEqual((s.even_brightness, s.peak_hold_s, s.normalize_window),
                         (True, 3, 41))
        self.assertEqual((s.brightness, s.display_gain, s.contrast), (6, 1.5, 20))
        self.queued()
        w.calibrate_button.click()
        self.assertIn(("calibrate_noise", {}), self.queued())
        self.engine.events.put(("noise_calibrated", 17))
        w._handle_events()
        self.assertEqual(w.min_range.value(), 17)
        w.scaling.setCurrentText("Fixed gain")
        self.assertFalse(w.steady_range.isEnabled())
        self.assertTrue(w.min_range.isEnabled())

    def test_sections_collapse_and_are_remembered(self):
        """A collapsed section hides its controls and stays collapsed next time.

        Args: None.
        Returns: None.
        """
        toggle = self.window._section_toggles["Display"]
        toggle.setChecked(False)
        self.assertTrue(self.window.gamma.isHidden() or not self.window.gamma.isVisible())
        self.assertTrue(FakeSettings.store[qt_app.SECTION_SETTING_PREFIX + "Display"])
        again = demo_window(self.engine, self.log, self.use_gl)
        self.addCleanup(again.deleteLater)
        self.assertFalse(again._section_toggles["Display"].isChecked())
        self.assertTrue(again._section_toggles["Camera"].isChecked())
        toggle.setChecked(True)
        self.assertFalse(FakeSettings.store[qt_app.SECTION_SETTING_PREFIX + "Display"])

    def test_labview_controls_update_engine_and_enable_each_other(self):
        """The new sliders write settings; scaling enables only what it uses.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.frames_back.spin.setValue(4)
        w.outlier_filter.spin.setValue(3)
        w.scaling.setCurrentText("Fixed gain")
        w.difference_gain.spin.setValue(12)
        self.assertFalse(w.average_multiplier.isEnabled())
        w.gain_before_average.setChecked(True)
        self.assertTrue(w.average_multiplier.isEnabled())
        w.average_multiplier.spin.setValue(2.5)
        s = self.engine.settings()
        self.assertEqual((s.frames_back, s.outlier_filter), (4, 3))
        self.assertEqual((s.scale_range, s.difference_gain), (False, 12))
        self.assertEqual((s.gain_before_average, s.average_multiplier), (True, 2.5))
        self.assertFalse(w.clip.isEnabled())
        w.scaling.setCurrentText("Auto stretch")
        self.assertTrue(w.clip.isEnabled())
        for widget in (w.difference_gain, w.gain_before_average, w.average_multiplier):
            self.assertFalse(widget.isEnabled())

    def test_labview_flow_button_swaps_settings_and_restores_them(self):
        """On applies the LabVIEW flow and a reference; off restores the user's.

        Args: None.
        Returns: None.
        """
        w = self.window
        self.engine.update_settings(mode="Previous", noise_reduction=40, smoothing=5,
                                    gamma=1.4, difference_gain=12, exposure_us=3000)
        w._load_settings(self.engine.settings(), w.current.limits)
        with mock.patch.object(self.engine, "request") as request:
            w.labview_button.setChecked(True)
        request.assert_called_once_with("reference")
        s = self.engine.settings()
        for key, value in qt_app.LABVIEW_FLOW.items():
            self.assertEqual(getattr(s, key), value, key)
        self.assertEqual((s.difference_gain, s.exposure_us), (42, 3000))
        self.assertEqual(w.scaling.currentText(), "Fixed gain")
        self.assertIn("LabVIEW flow on", " ".join(self.log.lines))
        w.set_labview_flow(True)  # already on: nothing more is kept
        w.smoothing.spin.setValue(7)  # a change made in the LabVIEW flow is dropped
        w.labview_button.setChecked(False)
        s = self.engine.settings()
        self.assertEqual((s.mode, s.noise_reduction, s.smoothing, s.gamma),
                         ("Previous", 40, 5, 1.4))
        self.assertEqual(s.difference_gain, 12)
        self.assertIsNone(w.current.labview_saved)
        self.assertEqual(w.mode_group.checkedButton().text(), "Previous")
        self.assertIn("LabVIEW flow off", " ".join(self.log.lines))
        w.set_labview_flow(False)  # already off: settings stay
        self.assertEqual(self.engine.settings().mode, "Previous")

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
        help_menu = self.window.menuBar().actions()[-1].menu()
        next(a for a in help_menu.actions() if a.text().startswith("About")).trigger()
        self.dialogs.about.assert_called_once()
        self.assertEqual(self.dialogs.about.call_args.args[2], updater.describe_build())

    def test_guide_action_opens_the_html_file(self):
        """Help > User Guide hands the bundled HTML file to the browser.

        Args: None.
        Returns: None.
        """
        help_menu = self.window.menuBar().actions()[-1].menu()
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
        help_menu = self.window.menuBar().actions()[-1].menu()
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
        help_menu = self.window.menuBar().actions()[-1].menu()
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

    The spin box's internal editor is part of the spin box, and a scroll
    area's scroll bars are part of that area, so neither is listed on its own. A new slider, button, combo or text field with no
    tooltip fails the suite.

    Args: dock is the controls QDockWidget.
    Returns: Widgets that still need a tooltip.
    """
    kinds = (QAbstractButton, QAbstractSlider, QAbstractSpinBox, QComboBox,
             QLineEdit, QPlainTextEdit, QTextEdit)
    containers = (QAbstractSpinBox, QComboBox, QAbstractScrollArea)
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


class SimpleUiTests(WindowTestCase):
    """The temporary basic-controls-only build renders just the essentials."""

    simple_ui = True

    def test_only_basic_controls_are_rendered(self):
        """Basic controls are visible; every other control exists but is hidden.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.show()
        for shown in (w.exposure, w.gain, w.record_button, *w.mode_group.buttons()):
            self.assertTrue(shown.isVisibleTo(w), shown)
        for hidden in (w.fps, w.noise_reduction, w.average, w.rolling, w.ref_refresh,
                       w.frames_back, w.smoothing_combo, w.smoothing, w.scaling,
                       w.invert, w.clip, w.gamma, w.brightness,
                       w.flip_button, w.labview_button, w.ui_scale_row,
                       w.frame_title, w.frame_notes, w.zoom_slider, w.zoom_label):
            self.assertFalse(hidden.isVisibleTo(w), hidden)
        for toggle in w._section_toggles.values():
            self.assertEqual(toggle.isVisibleTo(w), toggle.text() == "Camera",
                             toggle.text())
        buttons = {b.text(): b for b in w.actions_bar.findChildren(qt_app.QPushButton)}
        for label in ("Save frame (S)", "Set reference (F)"):
            self.assertTrue(buttons[label].isVisibleTo(w), label)
        # The basic toolbar is wider than this test window, so Qt may move the
        # last button into its overflow menu: check the toolbar item, not the widget.
        compare = next(a for a in w.actions_bar.actions()
                       if w.actions_bar.widgetForAction(a) is buttons["Compare (C)"])
        self.assertTrue(compare.isVisible())
        self.assertNotIn(compare, w._advanced_actions)
        for label in ("Reset last (R)", "Reset average (A)", "Fit",
                      "1:1", "Freeze (Space)"):
            self.assertFalse(buttons[label].isVisibleTo(w), label)

    def test_placing_or_parking_a_row_twice_does_nothing(self):
        """A row already parked stays parked; one already placed stays placed.

        Args: None.
        Returns: None.
        """
        w = self.window
        entry = w._advanced_rows[0]
        self.assertFalse(entry.placed)
        w._park_row(entry)
        self.assertFalse(entry.placed)
        w._place_row(entry)
        name = entry.name
        w._place_row(entry)
        self.assertTrue(entry.placed)
        self.assertIs(entry.name, name)

    def test_help_tips_stay_hidden(self):
        """Tips advertise hidden controls, so neither the tip nor its menu item shows.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.show()
        w._show_tip()
        self.assertFalse(w.tip_label.isVisibleTo(w))
        self.assertFalse(w.tips_action.isVisible())

    def test_toggle_button_shows_every_control_and_back(self):
        """The Simple UI button renders the hidden controls, then hides them again.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.show()
        button = w.simple_ui_button
        self.assertTrue(button.isChecked())
        self.assertTrue(button.isVisibleTo(w))
        self.assertFalse(w.tips_action.isVisible())
        button.click()
        QApplication.processEvents()
        self.assertFalse(w.simple_ui)
        self.assertTrue(button.isVisibleTo(w))
        # The panel scrolls at its full height instead of squeezing its rows.
        self.assertGreaterEqual(w.controls_panel.height(),
                                w.controls_panel.minimumSizeHint().height())
        for shown in (w.exposure, w.gain, w.fps, w.noise_reduction, w.average,
                      w.scaling, w.gamma, w.labview_button, w.frame_title):
            self.assertTrue(shown.isVisibleTo(w), shown)
        self.assertFalse(w.single_field.isVisibleTo(w))  # hides itself unless needed
        for toggle in w._section_toggles.values():
            if toggle.text() != "iPhone":  # shown only for a phone camera
                self.assertTrue(toggle.isVisibleTo(w), toggle.text())
        # The window is too narrow for every toolbar item, so Qt moves the last
        # ones into its overflow menu; check the actions, not the widgets.
        self.assertEqual(len(w._advanced_actions), 8)
        self.assertTrue(all(a.isVisible() for a in w._advanced_actions))
        self.assertTrue(w.tips_action.isVisible())
        self.assertIn(("ui/simple", False), FakeSettings.store.items())
        button.click()
        QApplication.processEvents()
        self.assertTrue(w.simple_ui)
        for hidden in (w.fps, w.noise_reduction, w.scaling, w.frame_title,
                       w.zoom_slider):
            self.assertFalse(hidden.isVisibleTo(w), hidden)
        self.assertFalse(any(a.isVisible() for a in w._advanced_actions))
        for toggle in w._section_toggles.values():
            if toggle.text() != "iPhone":
                self.assertEqual(toggle.isVisibleTo(w), toggle.text() == "Camera",
                                 toggle.text())
        self.assertFalse(w.tips_action.isVisible())
        self.assertIs(FakeSettings.store["ui/simple"], True)

    def test_toggle_keeps_values_and_collapsed_sections(self):
        """Switching modes changes nothing about the settings or collapsed sections.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.show()
        w.noise_reduction.set_value(10)
        w.noise_reduction.valueChanged.emit(10)
        w.simple_ui_button.setChecked(False)
        w._section_toggles["Display"].setChecked(False)
        QApplication.processEvents()
        self.assertFalse(w.gamma.isVisibleTo(w))
        self.assertEqual(w.noise_reduction.value(), 10)
        w.simple_ui_button.setChecked(True)
        w.simple_ui_button.setChecked(False)
        QApplication.processEvents()
        self.assertEqual(self.engine.settings().noise_reduction, 10)
        self.assertEqual(w.noise_reduction.value(), 10)
        self.assertFalse(w.gamma.isVisibleTo(w))  # still collapsed
        self.assertTrue(w.noise_reduction.isVisibleTo(w))
        w._section_toggles["Display"].setChecked(True)
        QApplication.processEvents()
        self.assertTrue(w.gamma.isVisibleTo(w))

    def test_toggle_does_not_duplicate_labels(self):
        """Each full-UI toggle makes one label per labeled row, not an extra copy.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.show()
        simple_labels = len(w._section_names)
        w.simple_ui_button.setChecked(False)
        full_labels = len(w._section_names)
        self.assertGreater(full_labels, simple_labels)
        w.simple_ui_button.setChecked(True)
        w.simple_ui_button.setChecked(False)
        self.assertEqual(len(w._section_names), full_labels)
        w.simple_ui_button.setChecked(True)
        self.assertEqual(len(w._section_names), simple_labels)

    def test_saved_choice_wins_over_the_default(self):
        """A window built after the choice was saved starts in that mode.

        Args: None.
        Returns: None.
        """
        FakeSettings.store["ui/simple"] = "false"  # QSettings reads booleans as text
        engine = Engine(self.log)
        engine.use_demo()
        self.addCleanup(engine.wait, 5.0)
        self.addCleanup(engine.stop)
        window = demo_window(engine, self.log, simple_ui=True)
        self.addCleanup(window.close)
        self.assertFalse(window.simple_ui)
        self.assertFalse(window.simple_ui_button.isChecked())
        window.show()
        self.assertTrue(window.noise_reduction.isVisibleTo(window))

    def test_hidden_controls_still_drive_the_engine(self):
        """A hidden control keeps its wiring, so presets and shortcuts still work.

        Args: None.
        Returns: None.
        """
        self.window.noise_reduction.set_value(10)
        self.window.noise_reduction.valueChanged.emit(10)
        self.assertEqual(self.engine.settings().noise_reduction, 10)
        self.assertTrue(self.window.handle_key(Qt.Key_R))


class SettingsGuideTests(WindowTestCase):
    """Tooltips, the settings guide and the processing presets."""

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
            w.scaling: ("Fixed gain", "Auto stretch", "0 and 255", "Start with Fixed gain"),
            w.difference_gain: ("255", "Start at 42"),
            w.gain_before_average: ("LabVIEW", "Start off"),
            w.average_multiplier: ("Multiplier", "Start at 1"),
            w.frames_back: ("Previous view", "Start at 1"),
            w.outlier_filter: ("neighbourhood", "Start at 0"),
            w.labview_button: ("LabVIEW ESPI 1.5", "Fixed gain", "your own settings",
                               "analog camera"),
            w.invert: ("bright and dark", "Start off"),
            w.flip_button: ("left to right", "Start off"),
            w.clip: ("speckles", "Start at 0"),
            w.gamma: ("Start at 1.0",),
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
        self.assertEqual((s.noise_reduction, s.clip_percent), (0.0, 0.5))
        self.assertEqual((s.exposure_us, s.gain_db, s.fps, s.record), (1234, 2.5, 15, True))
        self.assertFalse(s.single_field)
        self.assertEqual(w.average.value(), 8)
        self.assertTrue(w.mode_group.checkedButton().text() == "Reference")
        buttons["Live preview"].click()
        s = self.engine.settings()
        self.assertEqual(qt_app.PROCESSING_PRESETS["Live preview"]["mode"], s.mode)
        self.assertEqual((s.average, s.smoothing, s.ref_refresh_s), (1, 0, 0.0))
        self.assertEqual((s.exposure_us, s.gain_db, s.fps, s.record), (1234, 2.5, 15, True))
        self.assertEqual(w.mode_group.checkedButton().text(), "Previous")
        self.assertIn("Preset applied: Live preview", self.log.lines)
        self.assertIn("exposure", buttons["Live preview"].toolTip())

    def test_peak_hold_preset_cuts_noise_first_and_keeps_the_floor(self):
        """Peak hold turns on the analog fringe tools; other presets turn them off.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.min_range.spin.setValue(42)  # a calibrated floor the preset must keep
        w.exposure.spin.setValue(1234)
        buttons = {b.text(): b for b in w.controls_dock.findChildren(qt_app.QPushButton)}
        buttons["Peak hold"].click()
        s = self.engine.settings()
        self.assertEqual(s.mode, "Previous")
        self.assertEqual((s.peak_hold_s, s.threshold, s.steady_range_s), (1.0, 6, 1.0))
        self.assertEqual((s.smoothing, s.smoothing_method), (3, "Median"))
        self.assertEqual((s.scale_range, s.clip_percent, s.contrast), (True, 0.5, 30.0))
        self.assertEqual((s.noise_reduction, s.average, s.ref_refresh_s), (0.0, 1, 0.0))
        self.assertEqual((s.normalize_window, s.even_brightness), (0, False))
        self.assertEqual((s.min_range, s.exposure_us), (42, 1234))
        self.assertEqual((w.peak_hold.value(), w.threshold.value()), (1.0, 6))
        self.assertIn("Preset applied: Peak hold", self.log.lines)
        tip = " ".join(buttons["Peak hold"].toolTip().split())
        for phrase in ("peak hold 1 s", "sensitivity floor", "exposure"):
            self.assertIn(phrase, tip)
        for name in ("Live preview", "Measurement"):
            buttons["Peak hold"].click()
            buttons[name].click()
            s = self.engine.settings()
            for field, off in qt_app.FRINGE_TOOLS_OFF.items():
                self.assertEqual(getattr(s, field), off, f"{name} left {field} on")
            self.assertEqual(s.min_range, 42)
        self.assertEqual(w.peak_hold.value(), 0.0)

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
        buttons["Apply Peak hold"].click()
        self.assertEqual(self.engine.settings().peak_hold_s, 1.0)
        self.assertIn("Peak hold", qt_app.SETTINGS_GUIDE)
        buttons["Close"].click()
        self.assertFalse(dialog.isVisible())


class FringeControlsTests(WindowTestCase):
    """Collapsible Controls sections and the Calibrate noise result."""

    def test_collapsed_section_is_remembered_for_the_next_window(self):
        """Collapsing a heading hides it, saves it, and a new window starts collapsed.

        Args: None.
        Returns: None.
        """
        toggle = self.window._section_toggles["Fringe enhancement"]
        toggle.setChecked(False)
        self.assertEqual(toggle.arrowType(), qt_app.Qt.RightArrow)
        key = f"{qt_app.SECTION_SETTING_PREFIX}Fringe enhancement"
        self.assertIs(FakeSettings.store[key], True)
        toggle.setChecked(True)
        self.assertIs(FakeSettings.store[key], False)
        self.assertEqual(toggle.arrowType(), qt_app.Qt.DownArrow)
        FakeSettings.store[key] = "true"  # QSettings reads booleans back as text
        window = demo_window(self.engine, self.log)
        self.addCleanup(window.deleteLater)
        for timer in (window.frame_timer, window.status_timer, window.tip_timer):
            timer.stop()
        self.assertFalse(window._section_toggles["Fringe enhancement"].isChecked())
        self.assertTrue(window._section_toggles["Display"].isChecked())
        window.close()

    def test_frame_averaging_and_noise_floor_controls(self):
        """The two checkboxes send their settings; the button asks for a map.

        Args: None.
        Returns: None.
        """
        w = self.window
        self.assertFalse(w.frames_first.isChecked())
        self.assertTrue(w.rolling.isEnabled())
        w.frames_first.setChecked(True)
        self.assertTrue(self.engine.settings().average_frames_first)
        self.assertFalse(w.rolling.isEnabled())  # block averages only
        w.frames_first.setChecked(False)
        self.assertTrue(w.rolling.isEnabled())
        w.floor_subtract.setChecked(True)
        self.assertTrue(self.engine.settings().floor_subtract)
        self.queued()
        w.floor_button.click()
        self.assertIn(("measure_floor", {}), self.queued())
        w._load_settings(replace(self.engine.settings(), average_frames_first=True,
                                 floor_subtract=False), {})
        self.assertTrue(w.frames_first.isChecked())
        self.assertFalse(w.floor_subtract.isChecked())
        self.assertFalse(w.rolling.isEnabled())
        self.assertTrue(self.engine.settings().floor_subtract)  # not sent back

    def test_noise_calibration_result_updates_the_floor_slider(self):
        """noise_calibrated shows the measured floor without sending it back.

        Args: None.
        Returns: None.
        """
        self.engine.events.put(("noise_calibrated", 17))
        self.window._handle_events()
        self.assertEqual(self.window.min_range.value(), 17)
        self.assertEqual(self.engine.settings().min_range, 0)  # the engine set its own


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


class ReferenceCompareTests(WindowTestCase):
    """The reference comparison window and the Ref. frames control."""

    def compare(self, sequence=1, reference=True, demo=False):
        """Build a ComparePacket with small images.

        Args: sequence is its number; reference adds a reference and a
            difference; demo marks it synthetic.
        Returns: The ComparePacket.
        """
        image = np.full((40, 60), 80, dtype=np.uint8)
        part = FramePacket(sequence, image, 1.0, "Current", demo)
        diff = FramePacket(sequence, image, 1.0, "Reference", demo)
        return ComparePacket(sequence, part, part if reference else None,
                             diff if reference else None,
                             "1 frame, noise reduction 0, 2 s old" if reference
                             else "none yet", "the Reference view, no averaging", demo)

    def previous_compare(self, sequence=1, back=2, baseline=True, difference=True):
        """Build a Previous view ComparePacket with small images.

        Args: sequence is its number; back is the frames back; baseline and
            difference say which of those panels exist.
        Returns: The ComparePacket.
        """
        image = np.full((40, 60), 80, dtype=np.uint8)
        frames = f"{back} {'frame' if back == 1 else 'frames'}"
        part = FramePacket(sequence, image, 1.0, "Current", False)
        diff = FramePacket(sequence, image, 1.0, "Previous", False)
        return ComparePacket(
            sequence, part, part if baseline else None, diff if difference else None,
            f"the frame the view subtracts, {frames} back" if baseline else "none yet",
            "the Previous view, no averaging" if difference else "averaging: 1 of 4 frames",
            False, "Previous", f"Previous ({frames} back)")

    def open_dialog(self):
        """Open the comparison window with the C key.

        Args: None.
        Returns: The ReferenceCompareDialog.
        """
        self.assertTrue(self.window.handle_key(Qt.Key_C))
        dialog = self.window.compare_dialog
        self.addCleanup(dialog.hide)
        return dialog

    def test_toolbar_button_and_key_open_one_window(self):
        """Compare (C) opens the window once and asks the engine for comparisons.

        Args: None.
        Returns: None.
        """
        buttons = [b for b in self.window.actions_bar.findChildren(QPushButton)
                   if b.text() == "Compare (C)"]
        self.assertEqual(len(buttons), 1)
        self.assertIn("side by side", buttons[0].toolTip())
        self.assertIn("Previous view", buttons[0].toolTip())
        dialog = self.open_dialog()
        self.assertTrue(dialog.isVisible())
        self.assertTrue(self.engine._compare.is_set())
        buttons[0].click()
        self.assertIs(self.window.compare_dialog, dialog)
        self.assertEqual(sum("Reference comparison opened" in line
                             for line in self.log.lines), 1)
        dialog.reject()
        self.assertFalse(self.engine._compare.is_set())
        self.assertTrue(any("Reference comparison closed" in line for line in self.log.lines))

    def test_panels_show_images_or_the_no_reference_message(self):
        """With a reference all three images show; without one a message does.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        self.assertIsNone(self.engine.latest_compare())
        dialog.poll()  # nothing yet
        self.assertIsNone(dialog.shown)
        self.engine._compare_packet = self.compare(1, reference=False)
        dialog.poll()
        self.assertIs(dialog.stacks["baseline"].currentWidget(), dialog.messages["baseline"])
        self.assertIs(dialog.stacks["difference"].currentWidget(),
                      dialog.messages["difference"])
        self.assertIs(dialog.stacks["current"].currentWidget(), dialog.views["current"])
        self.assertIn("Press Set reference (F)", dialog.messages["baseline"].text())
        self.assertEqual(dialog.captions["baseline"].text(), "none yet")
        packet = self.compare(2, demo=True)
        self.engine._compare_packet = packet
        dialog.poll()
        for name in ("baseline", "current", "difference"):
            self.assertIs(dialog.stacks[name].currentWidget(), dialog.views[name])
        self.assertIs(dialog.views["difference"].packet, packet.difference)
        self.assertIn("2 s old", dialog.captions["baseline"].text())
        self.assertIn("DUMMY MODE", dialog.windowTitle())
        self.assertFalse(dialog.demo_label.isHidden())
        with mock.patch.object(dialog, "show_compare") as show:
            dialog.poll()  # same sequence: not redrawn
        show.assert_not_called()

    def test_freeze_holds_and_save_sends_the_shown_set(self):
        """Freeze stops updates; Save sends the shown packet with title and notes.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        dialog.save_button.click()
        self.assertTrue(any("No comparison on screen" in line for line in self.log.lines))
        first = self.compare(1)
        self.engine._compare_packet = first
        dialog.poll()
        dialog.freeze_button.setChecked(True)
        self.engine._compare_packet = self.compare(2)
        dialog.poll()
        self.assertIs(dialog.shown, first)
        self.window.frame_title.setText("plate A")
        self.window.frame_notes.setPlainText("before load")
        dialog.save_button.click()
        dialog.reference_button.click()
        (save, payload), (reference, _) = self.queued()
        self.assertEqual((save, reference), ("save_compare", "reference"))
        self.assertEqual((payload["packet"], payload["title"], payload["notes"]),
                         (first, "plate A", "before load"))
        dialog.freeze_button.setChecked(False)
        dialog.poll()
        self.assertEqual(dialog.shown.sequence, 2)
        self.assertTrue(any("comparison frozen" in line for line in self.log.lines))
        self.assertTrue(any("comparison live again" in line for line in self.log.lines))

    def test_previous_view_names_the_panels_and_disables_set_reference(self):
        """In Previous view the panels read Previous / Current / Difference.

        Args: None.
        Returns: None.
        """
        self.engine.update_settings(mode="Previous", frames_back=3)
        dialog = self.open_dialog()
        self.assertEqual(dialog.titles["baseline"].text(), "Previous (3 frames back)")
        self.assertEqual(dialog.windowTitle(), "Previous comparison")
        self.assertFalse(dialog.reference_button.isEnabled())
        self.assertIn("Not used in Previous view", dialog.reference_button.toolTip())
        self.assertIn("earlier frame", dialog.titles["baseline"].toolTip())
        self.assertIn("No previous frame yet", dialog.messages["baseline"].text())
        self.engine._compare_packet = self.previous_compare(1, back=1)
        dialog.poll()
        self.assertEqual(dialog.titles["baseline"].text(), "Previous (1 frame back)")
        self.assertEqual(dialog.titles["current"].text(), "Current frame")
        self.assertEqual(dialog.titles["difference"].text(), "Difference")
        self.assertEqual(dialog.captions["baseline"].text(),
                         "the frame the view subtracts, 1 frame back")
        self.assertEqual(dialog.captions["difference"].text(),
                         "the Previous view, no averaging")
        for name in ("baseline", "current", "difference"):
            self.assertIs(dialog.stacks[name].currentWidget(), dialog.views[name])
        self.assertFalse(dialog.reference_button.isEnabled())

    def test_previous_view_without_a_baseline_or_difference_shows_messages(self):
        """No earlier frame, or an average still filling, gets a message per panel.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        self.engine._compare_packet = self.previous_compare(1, baseline=False,
                                                            difference=False)
        dialog.poll()
        self.assertIs(dialog.stacks["baseline"].currentWidget(), dialog.messages["baseline"])
        self.assertIn("No previous frame yet", dialog.messages["difference"].text())
        self.engine._compare_packet = self.previous_compare(2, difference=False)
        dialog.poll()
        self.assertIs(dialog.stacks["baseline"].currentWidget(), dialog.views["baseline"])
        self.assertIs(dialog.stacks["difference"].currentWidget(),
                      dialog.messages["difference"])
        self.assertIn("Averaging", dialog.messages["difference"].text())
        self.assertEqual(dialog.captions["difference"].text(), "averaging: 1 of 4 frames")

    def test_switching_views_switches_the_panels_and_set_reference(self):
        """Reference and Previous packets rename the panels and toggle Set reference.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        self.assertTrue(dialog.reference_button.isEnabled())
        self.engine._compare_packet = self.previous_compare(1)
        dialog.poll()
        self.assertEqual(dialog.titles["baseline"].text(), "Previous (2 frames back)")
        self.assertFalse(dialog.reference_button.isEnabled())
        self.engine._compare_packet = self.compare(2)
        dialog.poll()
        self.assertEqual(dialog.titles["baseline"].text(), "Reference")
        self.assertEqual(dialog.windowTitle(), "Reference comparison")
        self.assertTrue(dialog.reference_button.isEnabled())
        self.assertIn("Press Set reference (F)", dialog.messages["baseline"].text())
        self.assertIn("stored reference frame", dialog.titles["baseline"].toolTip())

    def test_freeze_holds_all_three_previous_panels(self):
        """Freeze keeps the Previous set; Save sends it; unfreezing resumes.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        first = self.previous_compare(1)
        self.engine._compare_packet = first
        dialog.poll()
        dialog.freeze_button.setChecked(True)
        newer = self.previous_compare(2, back=4)
        self.engine._compare_packet = newer
        dialog.poll()
        self.assertIs(dialog.shown, first)
        for name in ("baseline", "current", "difference"):
            self.assertIs(dialog.views[name].packet, getattr(first, name))
        self.assertEqual(dialog.titles["baseline"].text(), "Previous (2 frames back)")
        dialog.save_button.click()
        (save, payload), = self.queued()
        self.assertEqual((save, payload["packet"]), ("save_compare", first))
        dialog.freeze_button.setChecked(False)
        dialog.poll()
        self.assertIs(dialog.shown, newer)
        self.assertEqual(dialog.titles["baseline"].text(), "Previous (4 frames back)")
        self.assertTrue(any("Previous comparison frozen" in line for line in self.log.lines))
        self.assertTrue(any("Previous comparison live again" in line
                            for line in self.log.lines))

    def test_window_follows_the_displayed_camera(self):
        """When the displayed camera changes, the request moves with it.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        other = Engine(self.log)
        dialog._follow(other)
        self.assertFalse(self.engine._compare.is_set())
        self.assertTrue(other._compare.is_set())
        dialog.freeze_button.setChecked(True)
        dialog.poll()  # the window still displays self.engine
        self.assertIs(dialog.engine, self.engine)
        self.assertFalse(other._compare.is_set())
        self.assertFalse(dialog.freeze_button.isChecked())
        dialog._follow(self.engine)  # already there: nothing changes
        self.assertTrue(self.engine._compare.is_set())

    def test_reference_frames_control(self):
        """Ref. frames sends whole numbers and shows each camera's value.

        Args: None.
        Returns: None.
        """
        self.window.ref_frames.spin.setValue(12)
        self.assertEqual(self.engine.settings().ref_frames, 12)
        self.assertIn("Start at 1", self.window.ref_frames.toolTip())
        self.window._load_settings(replace(self.engine.settings(), ref_frames=3), {})
        self.assertEqual(self.window.ref_frames.spin.value(), 3)
        self.assertEqual(self.engine.settings().ref_frames, 12)  # not sent back


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
        self.engine._packet = make_packet(sequence=3, gamma=2.0)
        self.window._poll_frame()
        self.assertIsNot(self.window.table, first_table)

    def test_view_state_is_logged_once_per_change(self):
        """The drawn size is logged at DEBUG when the view or image changes.

        Args: None.
        Returns: None.
        """
        self.window._log_view_state()  # nothing drawn yet
        self.assertFalse(self.log.has("Display:"))
        self.engine._packet = make_packet(sequence=1)
        self.window._poll_frame()
        self.window._log_view_state()
        self.window._log_view_state()
        lines = [line for line in self.log.lines if line.startswith("Display: view")]
        self.assertEqual(len(lines), 1)
        self.assertIn("image 64x48", lines[0])
        self.window.view.zoom = 2.0
        self.window._log_view_state()
        lines = [line for line in self.log.lines if line.startswith("Display: view")]
        self.assertEqual(len(lines), 2)
        self.assertIn("zoom 2", lines[1])
        self.window.view.image_size = None  # no drawing rectangle
        self.window.view.zoom = 3.0
        self.window._log_view_state()
        self.assertEqual(len([line for line in self.log.lines
                              if line.startswith("Display: view")]), 2)

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

        Checking for updates and the update button's own states move nothing
        either. Only finding a newer build does, once, when the labelled blue
        button replaces the icon-only check button.

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
            return [w.geometry().getRect() for w in items]

        before = geometry()
        window.camera_fps_status.setText("Camera: 1000000000.0 fps (simulated)")
        window.speed_status.setText("Processed: 1000000000.0 fps    Work: 0.0 ms/frame"
                                    "    Displayed: 1000000000.0 fps")
        window.record_label.setText("Recording: 1000000000.0 fps")
        window.range_status.setText("")
        window.pixel_status.setText("x 123456, y 123456: 255")
        self.assertEqual(geometry(), before)
        window.updates.busy = True
        window._sync_check_button()  # checking: the icon button only greys out
        self.assertEqual(geometry(), before)
        window.updates.busy = False
        window._sync_check_button()
        self.assertEqual(geometry(), before)
        window.update_button.setText("Install update")
        window.check_button.setVisible(False)
        window.update_button.setVisible(True)
        found = geometry()
        self.assertNotEqual(found, before)  # the labelled button needs room once
        for text in ("Downloading update… 7%", "Downloading update… 100%",
                     "Restart to update"):
            window.update_button.setText(text)
            self.assertEqual(geometry(), found)
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
        window.resize(1600, 800)  # wide enough that no status item wraps or is dropped
        QApplication.processEvents()
        names = {window.check_button: "Check for updates",
                 window.feature_button: "Request feature",
                 window.codex_button: "Open in Codex"}
        for button, name in names.items():
            self.assert_button_chrome(button, primary=False)
            # Icon-only and compact, but still named for tooltips and screen readers.
            self.assertFalse(button.icon().isNull(), name)
            self.assertEqual(button.text(), "", name)
            self.assertLessEqual(button.width(), 2 * button.height(), name)
            self.assertEqual(button.accessibleName(), name)
            self.assertTrue(button.toolTip().startswith(name + ":"), name)
        self.assertEqual(window.check_button.toolTip(),
                         qt_app.format_tooltip(
                             "Check for updates: Check GitHub for a newer build of ESPI live."))
        self.assertEqual(window.feature_button.toolTip(),
                         qt_app.format_tooltip(
                             "Request feature: Send a feature request to the developers "
                             "as a GitHub issue."))
        # Reading order (row, then left to right): the buttons, the rule, then the source.
        # The window is wide, so they share one row; a narrow one wraps the source down.
        def reading_order(widget):
            """Place a status item in reading order: its row, then its x.

            Args: widget is a status-bar item.
            Returns: (row, x), where the row is its vertical centre in button heights.
            """
            row_height = window.codex_button.height()
            return (widget.geometry().center().y() // row_height, widget.x())

        self.assertLess(reading_order(window.codex_button),
                        reading_order(window.status_actions_rule))
        self.assertLess(reading_order(window.status_actions_rule),
                        reading_order(window.source_badge))
        self.assertEqual(QLabel.text(window.source_badge), "  DUMMY MODE  ")
        self.assertGreaterEqual(window.source_badge.width(),
                                window.source_badge.sizeHint().width())
        self.assertEqual(window.status_actions_rule.frameShape(), qt_app.QFrame.Shape.VLine)
        window.check_button.setFocus(Qt.OtherFocusReason)
        self.assertTrue(window.check_button.hasFocus())

        window.updates.busy = True
        window._sync_check_button()
        self.assertFalse(window.check_button.isEnabled())
        self.assertEqual(window.check_button.accessibleName(), "Checking for updates…")
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
        qt_app._set_icon_action(window.feature_button, qt_app.FEATURE_BUSY_LABEL,
                                qt_app.FEATURE_TIP)
        self.assert_button_chrome(window.feature_button, primary=False)
        self.assert_has_frame(window.feature_button)
        self.assertLessEqual(window.feature_button.sizeHint().width(),
                             window.feature_button.width())
        self.assertEqual(window.feature_button.toolTip(),
                         qt_app.format_tooltip(
                             "Sending request…: Send a feature request to the developers "
                             "as a GitHub issue."))

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

    def test_window_title_carries_the_release_version(self):
        """A build made by CI shows its release tag first in every title form.

        Args: None.
        Returns: None.
        """
        tagged = types.SimpleNamespace(VERSION="v0.10.0")
        with mock.patch.object(updater, "_build_info", tagged):
            self.window._set_demo(True)
            self.assertEqual(self.window.windowTitle(), "ESPI live v0.10.0 — DUMMY MODE")
            self.window._set_demo(False)
            self.assertTrue(self.window.windowTitle().startswith("ESPI live v0.10.0"))

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
        self.assertEqual(self.window.windowTitle(),
                         f"{updater.app_title()} — BFS-U3-70S7C")
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

        The dummy-mode values stay behind: the live camera starts from its
        defaults (it has no saved profile here).

        Args: None.
        Returns: None.
        """
        self.window.record_button.setChecked(True)
        self.window.gain.spin.setValue(7.0)
        self.window.mode_group.buttons()[1].click()
        self.engine.update_settings(record=False, mode="Raw")  # as _switch_live does
        self.engine.events.put(("switched_live", None))
        self.window._handle_events()
        checked = [b.text() for b in self.window.mode_group.buttons() if b.isChecked()]
        self.assertEqual(checked, ["Raw"])
        self.assertFalse(self.window.record_button.isChecked())
        self.assertEqual(self.window.windowTitle(), updater.app_title())
        self.assertEqual(self.engine.settings().gain_db, 0.0)
        self.assertEqual(self.window.gain.value(), 0.0)
        self.assertTrue(self.log.has("starts from its defaults"))
        demo = json.loads(FakeSettings.store["profiles/demo/DEMO1"])  # kept apart
        self.assertEqual(demo["settings"]["gain_db"], 7.0)

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


def box(width, height):
    """Make a widget of a fixed size, to stand in for a status-bar item.

    Args: width and height are its size in pixels.
    Returns: A QWidget.
    """
    widget = QWidget()
    widget.setFixedSize(width, height)
    return widget


class StatusLayoutTests(unittest.TestCase):
    """StatusLayout: one row when everything fits, more rows when it does not."""

    def build(self, sizes, message_min=100):
        """Make a host with a 50x10 message area and items of the given sizes.

        Args: sizes lists (width, height) of the items after the message;
            message_min is the layout's least message width.
        Returns: The StatusLayout; self.host, self.message and self.items
            hold the widgets.
        """
        self.host = QWidget()
        self.addCleanup(self.host.deleteLater)
        flow = qt_app.StatusLayout(self.host, message_min)
        self.message = QWidget()  # at least 50x10, and free to grow
        self.message.setMinimumSize(50, 10)
        flow.addWidget(self.message)
        self.items = [box(*size) for size in sizes]
        for item in self.items:
            flow.addWidget(item)
        return flow

    def test_items_share_one_row_when_they_fit(self):
        """The message area takes what the items leave; the items sit at the right.

        Args: None.
        Returns: None.
        """
        flow = self.build([(60, 20)] * 3)
        self.assertEqual(flow.heightForWidth(400), 20)
        flow.setGeometry(QRect(0, 0, 400, 20))
        self.assertEqual(self.message.geometry().getRect(), (0, 0, 202, 20))
        self.assertEqual([item.geometry().getRect() for item in self.items],
                         [(208, 0, 60, 20), (274, 0, 60, 20), (340, 0, 60, 20)])
        self.assertEqual(self.host.height(), 20)

    def test_items_centre_vertically_in_their_row(self):
        """A shorter item sits in the middle of its row.

        Args: None.
        Returns: None.
        """
        flow = self.build([(60, 20), (60, 10)])
        flow.setGeometry(QRect(0, 0, 400, 20))
        self.assertEqual(self.items[1].geometry().y(), 5)

    def test_items_that_do_not_fit_move_to_more_rows(self):
        """Too narrow: the message keeps the first row, the items wrap beneath it.

        Each row is as tall as its tallest item, and the rows are right-aligned.

        Args: None.
        Returns: None.
        """
        flow = self.build([(60, 20)] * 3)
        self.assertEqual(flow.heightForWidth(150), 10 + 2 + 20 + 2 + 20)
        flow.setGeometry(QRect(0, 0, 150, 54))
        self.assertEqual(self.message.geometry().getRect(), (0, 0, 150, 10))
        self.assertEqual([item.geometry().getRect() for item in self.items],
                         [(24, 12, 60, 20), (90, 12, 60, 20), (90, 34, 60, 20)])
        self.assertEqual(self.host.height(), 54)
        flow.setGeometry(QRect(0, 0, 400, 54))  # wider again: back to one row
        self.assertEqual(self.host.height(), 20)

    def test_the_first_row_is_shared_when_the_message_keeps_its_minimum(self):
        """Items that fit beside the message's minimum stay on its row.

        Args: None.
        Returns: None.
        """
        flow = self.build([(60, 20)] * 3, message_min=100)
        flow.setGeometry(QRect(0, 0, 290, 0))  # two items fit beside the 100 px minimum
        self.assertEqual(self.message.geometry().getRect(), (0, 0, 158, 20))
        self.assertEqual([item.geometry().y() for item in self.items], [0, 0, 22])

    def test_hidden_items_take_no_room(self):
        """A hidden item leaves no gap, and showing it again restores its room.

        Args: None.
        Returns: None.
        """
        flow = self.build([(60, 20)] * 3)
        self.items[1].hide()
        flow.setGeometry(QRect(0, 0, 400, 0))
        self.assertEqual([item.geometry().x() for item in (self.items[0], self.items[2])],
                         [274, 340])
        self.assertEqual(flow.sizeHint().width(), 50 + 2 * 60 + 2 * 6)
        self.items[1].show()
        self.assertEqual(flow.sizeHint().width(), 50 + 3 * 60 + 3 * 6)

    def test_the_separator_is_dropped_at_the_edge_of_a_row(self):
        """The rule never begins or ends a row; it is parked past the right edge.

        Args: None.
        Returns: None.
        """
        flow = self.build([(60, 20), (60, 20), (10, 20), (60, 20)])
        flow.separator = self.items[2]
        flow.setGeometry(QRect(0, 0, 140, 0))  # a and b fill the row, then the rule
        self.assertGreaterEqual(self.items[2].x(), 140)
        self.assertEqual(self.items[3].x(), 80)  # c wrapped, right-aligned alone
        self.assertEqual(self.items[3].y(), self.items[0].y() + 22)
        flow.setGeometry(QRect(0, 0, 400, 0))  # roomy: the rule sits between b and c
        self.assertLess(self.items[2].x(), 400)
        self.assertEqual(self.items[2].width(), 10)

    def test_the_separator_is_dropped_first_and_last(self):
        """A rule with nothing before it, or nothing after it, is not shown.

        Args: None.
        Returns: None.
        """
        flow = self.build([(10, 20), (60, 20)])
        flow.separator = self.items[0]
        flow.setGeometry(QRect(0, 0, 300, 0))
        self.assertGreaterEqual(self.items[0].x(), 300)
        flow = self.build([(60, 20), (10, 20)])
        flow.separator = self.items[1]
        flow.setGeometry(QRect(0, 0, 300, 0))
        self.assertGreaterEqual(self.items[1].x(), 300)

    def test_a_lone_message_gets_every_row_width(self):
        """With no other items the message area fills the strip.

        Args: None.
        Returns: None.
        """
        flow = self.build([])
        flow.setGeometry(QRect(0, 0, 300, 0))
        self.assertEqual(self.message.geometry().getRect(), (0, 0, 300, 10))
        self.assertEqual(self.host.height(), 10)

    def test_size_hints_and_item_access(self):
        """The one-row size, the any-width minimum and the QLayout item protocol.

        Args: None.
        Returns: None.
        """
        flow = self.build([(60, 20)] * 3)
        self.assertEqual(flow.sizeHint(), QSize(50 + 3 * 60 + 3 * 6, 20))
        self.assertEqual(flow.minimumSize(), QSize(0, 20))
        self.assertTrue(flow.hasHeightForWidth())
        self.assertEqual(flow.expandingDirections(), Qt.Orientation(0))
        self.assertEqual(flow.count(), 4)
        self.assertIs(flow.itemAt(1).widget(), self.items[0])
        self.assertIsNone(flow.itemAt(9))
        self.assertIsNone(flow.itemAt(-1))
        self.assertIsNone(flow.takeAt(9))
        self.assertIs(flow.takeAt(1).widget(), self.items[0])
        self.assertEqual(flow.count(), 3)


class StatusBarLayoutTests(WindowTestCase):
    """The window's status bar wraps its readouts instead of cutting them short."""

    WIDEST = {
        "camera_fps_status": "Camera: 999.9 fps (simulated)",
        "speed_status": "Processed: 999.9 fps    Work: 99.9 ms/frame    Displayed: 999.9 fps",
        "record_label": "Recording: 999.9 fps",
        "range_status": "Input: 255–255    Scaled: 255–255",
        "pixel_status": "x 3088, y 2064: 255",
        "source_badge": "  BFS-U3-200S6M-C  ",
    }

    def settle(self, width):
        """Resize the window and let the layout, and the bar's height, settle.

        Args: width is the window width in pixels.
        Returns: None.
        """
        self.window.resize(width, 700)
        for _ in range(3):
            QApplication.processEvents()

    def readouts(self):
        """List the widgets the strip lays out after its message area.

        Args: None.
        Returns: The visible items, in layout order.
        """
        flow = self.window.status_strip.flow
        items = [flow.itemAt(i).widget() for i in range(1, flow.count())]
        return [item for item in items if not item.isHidden()]

    def test_every_readout_shows_in_full_at_any_width(self):
        """Wide or narrow, nothing is shortened, clipped or drawn over another item.

        Args: None.
        Returns: None.
        """
        window = self.window
        window.show()
        for name, text in self.WIDEST.items():
            getattr(window, name).setText(text)
        window.hint_label.setText(qt_app.LONGEST_HINT)
        for width in (2400, 1600, 1280, 900, 700):
            self.settle(width)
            strip = window.status_strip
            seen = []
            for item in self.readouts():
                rect = item.geometry()
                if item is window.status_actions_rule and rect.x() >= strip.width():
                    continue  # dropped at a row's edge
                self.assertGreaterEqual(rect.left(), 0, (width, item))
                self.assertLessEqual(rect.right(), strip.width() - 1, (width, item))
                self.assertFalse(any(rect.intersects(other) for other in seen), (width, item))
                seen.append(rect)
            for name in self.WIDEST:
                label = getattr(window, name)
                self.assertEqual(QLabel.text(label), label.text(), (width, name))
            self.assertGreaterEqual(window.hint_label.width(),
                                    window.hint_label.sizeHint().width(), width)

    def test_the_bar_is_one_row_when_wide_and_taller_when_narrow(self):
        """Extra rows cost height only when the window is too narrow for one.

        Args: None.
        Returns: None.
        """
        window = self.window
        window.show()
        heights = {}
        for width in (700, 900, 1280, 2400):
            self.settle(width)
            strip = window.status_strip
            heights[width] = strip.height()
            self.assertGreaterEqual(window.statusBar().height(), strip.height())
            self.assertEqual(strip.height(), strip.flow.heightForWidth(strip.width()))
        self.assertEqual(heights[2400], window.status_strip.flow.sizeHint().height())
        self.assertLessEqual(heights[1280], heights[900])
        self.assertLessEqual(heights[900], heights[700])
        self.assertLess(heights[2400], heights[1280])

    def test_the_hint_and_readouts_do_not_crowd_each_other_when_narrow(self):
        """A narrow window moves the readouts below the message instead of squeezing it.

        Args: None.
        Returns: None.
        """
        window = self.window
        window.show()
        window.hint_label.setText(qt_app.LONGEST_HINT)
        self.settle(900)
        strip = window.status_strip
        self.assertGreater(strip.height(), strip.flow.sizeHint().height())
        self.assertGreaterEqual(window.hint_label.width(),
                                window.hint_label.fontMetrics().horizontalAdvance(
                                    qt_app.LONGEST_HINT))
        hint_top = window.hint_label.mapTo(strip, QPoint(0, 0)).y()
        self.assertGreater(window.pixel_status.y(), hint_top)  # a row below the hint

    def test_a_temporary_message_replaces_the_notices_but_not_the_readouts(self):
        """showMessage shows its text in the strip; the readouts stay on screen.

        Args: None.
        Returns: None.
        """
        window = self.window
        window.show()
        QApplication.processEvents()
        strip = window.status_strip
        self.assertTrue(window.hint_label.isVisible())
        self.assertFalse(strip.message_label.isVisible())
        window.statusBar().showMessage("Checking GitHub for a newer build…")
        QApplication.processEvents()
        self.assertTrue(strip.message_label.isVisible())
        self.assertEqual(strip.message_label.text(), "Checking GitHub for a newer build…")
        self.assertFalse(window.hint_label.isVisible())
        self.assertTrue(window.source_badge.isVisible())
        window.statusBar().clearMessage()
        QApplication.processEvents()
        self.assertFalse(strip.message_label.isVisible())
        self.assertTrue(window.hint_label.isVisible())

    def test_a_long_camera_name_widens_the_source_badge(self):
        """The badge shows the whole name, and goes back to its usual width after.

        Args: None.
        Returns: None.
        """
        window = self.window
        window.show()
        usual = window.source_badge.sizeHint().width()
        window.source_badge.setText("  Hauppauge USB-Live2 capture stick  ")
        QApplication.processEvents()
        self.assertGreater(window.source_badge.sizeHint().width(), usual)
        self.assertEqual(QLabel.text(window.source_badge),
                         "  Hauppauge USB-Live2 capture stick  ")
        window.source_badge.setText("  DUMMY MODE  ")
        self.assertEqual(window.source_badge.sizeHint().width(), usual)


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

    def test_single_camera_keeps_the_dropdown_but_not_the_process_rows(self):
        """With one camera the drop-down names it; there is nothing to process
        in the background and no log prefix.

        Args: None.
        Returns: None.
        """
        engine = Engine(self.log)
        engine.use_demo()
        window = demo_window(engine, self.log)
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        self.assertFalse(window.cameras_box.isHidden())
        self.assertFalse(window.camera_combo.isHidden())
        self.assertEqual(window.camera_combo.count(), 1)
        self.assertEqual(window.camera_combo.itemText(0), "Synthetic camera (DEMO1)")
        self.assertTrue(window.process_note.isHidden())
        self.assertTrue(all(w.isHidden() for w in window.camera_checks + window.camera_states))
        self.assertEqual(window.current.log.prefix, "")
        self.assertEqual(window.windowTitle(), f"{updater.app_title()} — DUMMY MODE")

    def test_cameras_are_listed_with_the_displayed_one_locked_on(self):
        """The drop-down lists every camera; the shown one must keep processing.

        Args: None.
        Returns: None.
        """
        window = self.window
        self.assertFalse(window.cameras_box.isHidden())
        self.assertFalse(window.process_note.isHidden())
        self.assertFalse(any(w.isHidden() for w in window.camera_checks + window.camera_states))
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
                                  gamma=2.0, rolling=True, noise_reduction=40.0)
        second.limits = {"Gain": (0.0, 12.0)}
        window.exposure.set_value(4000.0)
        window.exposure.valueChanged.emit(4000.0)
        window.camera_combo.setCurrentIndex(1)
        self.assertIs(window.current, second)
        self.assertTrue(second.active)
        self.assertEqual(window.exposure.value(), 2500.0)
        self.assertEqual(window.noise_reduction.value(), 40.0)
        self.assertEqual(window.gain.spin.maximum(), 12.0)
        self.assertEqual(window.gamma.value(), 2.0)
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
        # Pretend both are real cameras without touching the engines: their
        # worker threads are running, and one that finds no source mid-frame
        # dies with a fatal event, which made this test depend on timing.
        patcher = mock.patch.object(Channel, "demo", new_callable=mock.PropertyMock,
                                    return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        first.engine.events.put(("camera_identity", CameraIdentity(
            model="Blackfly S BFS-U3-63S4M", serial="DEMO1")))
        second.engine.events.put(("camera_identity", CameraIdentity(
            model="Blackfly S BFS-U3-70S7C", serial="DEMO2")))
        window._handle_events()
        self.assertEqual(window.windowTitle(),
                         f"{updater.app_title()} — BFS-U3-63S4M (DEMO1)")
        self.assertEqual(second.identity.model, "Blackfly S BFS-U3-70S7C")
        window.camera_combo.setCurrentIndex(1)
        self.assertEqual(window.windowTitle(),
                         f"{updater.app_title()} — BFS-U3-70S7C (DEMO2)")
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
        self.assertIn(f"{updater.app_title()} — BFS (111)", window.windowTitle())

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


KEY1 = "profiles/demo/DEMO1"
KEY2 = "profiles/demo/DEMO2"


def stored(key):
    """Decode one JSON entry of the fake settings store.

    Args: key is the entry's name.
    Returns: The decoded object.
    """
    return json.loads(FakeSettings.store[key])


class ProfileWindowTests(unittest.TestCase):
    """Each camera's own controls, saved between runs, and the saved presets."""

    simple_ui = False

    def setUp(self):
        """Build a window over two synthetic cameras on an empty fake store.

        Args: None.
        Returns: None.
        """
        FakeSettings.store = {}
        FakeSettings.syncs = 0
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
        self.prompts = []  # (title, label, preset text) of each name prompt
        self.window = self.make_window()

    def make_window(self, count=2):
        """Start the program again: cameras restored from the store, then the window.

        Args: count is how many synthetic cameras to create.
        Returns: The MainWindow; self.channels holds its channels.
        """
        store = qt_app.profiles.ProfileStore(FakeSettings(), self.log)
        self.channels = demo_channels(self.log, count, store)
        with mock.patch.object(qt_app, "SIMPLE_UI", self.simple_ui):
            window = qt_app.MainWindow(self.channels, self.log, False, False,
                                       profile_store=store)
        for timer in (window.frame_timer, window.status_timer, window.tip_timer,
                      window.profile_timer):
            timer.stop()
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        return window

    def display(self, index):
        """Pick a camera in the Display drop-down.

        Args: index is its position.
        Returns: That Channel.
        """
        self.window.camera_combo.setCurrentIndex(index)
        self.assertIs(self.window.current, self.channels[index])
        return self.channels[index]

    def checked_mode(self):
        """Name the view mode button that is checked.

        Args: None.
        Returns: "Raw", "Previous" or "Reference".
        """
        return next(b.text() for b in self.window.mode_group.buttons() if b.isChecked())

    def name_prompt(self, name, accept=True):
        """Answer the preset-name prompt instead of showing it.

        Args: name is the text typed; accept presses Save (True) or Cancel.
        Returns: A patcher for QInputDialog.exec.
        """
        def fake_exec(dialog):
            """Record the prompt, type the name and close it.

            Args: dialog is the QInputDialog.
            Returns: Accepted or Rejected.
            """
            self.prompts.append((dialog.windowTitle(), dialog.labelText(),
                                 dialog.textValue()))
            dialog.setTextValue(name)
            return qt_app.QDialog.Accepted if accept else qt_app.QDialog.Rejected
        return mock.patch.object(qt_app.QInputDialog, "exec", fake_exec)

    def save_preset(self, name, answer=None):
        """Save the displayed camera's settings as a preset.

        Args: name is the typed name; answer is the reply to a replace question.
        Returns: None.
        """
        if answer is not None:
            self.dialogs.question.return_value = answer
        with self.name_prompt(name):
            self.window.preset_save_button.click()

    # ----- isolation ------------------------------------------------------

    def test_changes_stay_with_their_camera_both_ways(self):
        """Camera 1's changes never reach camera 2, and camera 2's never reach 1.

        Args: None.
        Returns: None.
        """
        w = self.window
        first, second = self.channels
        w.exposure.spin.setValue(12000)
        w.gain.spin.setValue(6.0)
        w.noise_reduction.spin.setValue(20)
        w.mode_group.buttons()[2].click()
        w.invert.setChecked(True)
        w.scaling.setCurrentText(qt_app.SCALING_MODES[0])
        w.smoothing_combo.setCurrentText("Median")
        self.display(1)
        self.assertEqual(second.profile(), second.default_profile())
        self.assertEqual((w.exposure.value(), w.gain.value(), w.noise_reduction.value()),
                         (8000, 0, 0))
        self.assertEqual(self.checked_mode(), "Raw")
        self.assertFalse(w.invert.isChecked())
        self.assertEqual(w.scaling.currentText(), qt_app.SCALING_MODES[1])
        self.assertIn("Settings of Synthetic camera (DEMO2)", w.profile_label.text())
        w.labview_button.setChecked(True)
        w.gain.spin.setValue(12.0)
        w.flip_button.setChecked(True)
        self.display(0)
        s = first.profile().settings
        self.assertEqual((s.exposure_us, s.gain_db, s.noise_reduction, s.mode, s.invert,
                          s.scale_range, s.smoothing_method, s.flip_horizontal),
                         (12000, 6.0, 20, "Reference", True, False, "Median", False))
        self.assertIsNone(first.labview_saved)
        self.assertFalse(w.labview_button.isChecked())
        self.assertEqual((w.exposure.value(), w.gain.value()), (12000, 6.0))
        self.assertEqual(self.checked_mode(), "Reference")
        s = second.profile().settings
        self.assertEqual((s.gain_db, s.flip_horizontal, s.exposure_us), (12.0, True, 8000))
        self.assertIsNotNone(second.labview_saved)
        self.display(1)
        self.assertTrue(w.labview_button.isChecked())
        self.assertEqual(w.gain.value(), 12.0)

    def test_another_cameras_limits_never_send_it_this_cameras_values(self):
        """A narrower range clamps the shown value quietly instead of sending it.

        Args: None.
        Returns: None.
        """
        w = self.window
        first, second = self.channels
        second.limits = {"ExposureTime": (100.0, 20000.0), "Gain": (0.0, 10.0)}
        w.exposure.spin.setValue(40000)
        w.gain.spin.setValue(20.0)
        self.display(1)
        self.assertEqual(w.exposure.spin.maximum(), 20000.0)
        self.assertEqual((second.profile().settings.exposure_us,
                          second.profile().settings.gain_db), (8000.0, 0.0))
        w.gain.spin.setValue(9.0)
        first.limits = {"Gain": (0.0, 5.0)}
        self.display(0)
        self.assertEqual(second.profile().settings.gain_db, 9.0)
        self.assertEqual(first.profile().settings.exposure_us, 40000.0)

    def test_limits_arriving_for_the_shown_camera_still_clamp_it(self):
        """The camera's own limits event clamps and sends its own value, as before.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.gain.spin.setValue(20.0)
        self.channels[0].engine.events.put(("camera_limits", {"Gain": (0.0, 10.0)}))
        w._handle_events()
        self.assertEqual(self.channels[0].profile().settings.gain_db, 10.0)
        self.assertEqual(self.channels[1].profile().settings.gain_db, 0.0)

    # ----- persistence ------------------------------------------------------

    def test_nothing_is_written_until_something_changes(self):
        """A camera left at its defaults writes nothing, even on close.

        Args: None.
        Returns: None.
        """
        self.assertEqual(self.window.profile_timer.interval(), qt_app.PROFILE_SAVE_MS)
        self.window._save_profiles()
        self.window._save_profiles()
        self.window.close()
        self.assertEqual([key for key in FakeSettings.store if key.startswith("profiles/")], [])

    def test_profiles_are_written_once_settled_and_on_close(self):
        """The timer writes a profile after it stops changing; closing writes the rest.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.gain.spin.setValue(3.0)
        w.record_button.setChecked(True)
        w._save_profiles()
        self.assertNotIn(KEY1, FakeSettings.store)  # still changing, as far as it knows
        w._save_profiles()
        self.assertEqual(stored(KEY1)["settings"]["gain_db"], 3.0)
        self.assertNotIn("record", stored(KEY1)["settings"])
        self.assertNotIn(KEY2, FakeSettings.store)
        w.gain.spin.setValue(4.0)
        w.close()
        self.assertEqual(stored(KEY1)["settings"]["gain_db"], 4.0)
        self.assertEqual(FakeSettings.syncs, 1)

    def test_switching_saves_the_camera_left_behind(self):
        """Picking another camera writes the one just edited at once.

        Args: None.
        Returns: None.
        """
        self.window.exposure.spin.setValue(9000)
        self.display(1)
        self.assertEqual(stored(KEY1)["settings"]["exposure_us"], 9000.0)
        self.assertNotIn(KEY2, FakeSettings.store)

    def test_each_camera_comes_back_with_its_own_settings_after_a_restart(self):
        """A new window on the same store restores both cameras, never recording.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.exposure.spin.setValue(15000)
        w.mode_group.buttons()[1].click()
        w.labview_button.setChecked(True)
        w.record_button.setChecked(True)
        self.display(1)
        w.gain.spin.setValue(11.0)
        w.gamma.spin.setValue(2.0)
        w.close()
        restarted = self.make_window()
        first, second = self.channels
        s = first.engine.settings()
        self.assertEqual((s.exposure_us, s.mode, s.record), (15000.0, "Reference", False))
        self.assertTrue(restarted.labview_button.isChecked())
        self.assertFalse(restarted.record_button.isChecked())
        self.assertEqual(restarted.exposure.value(), 15000)
        self.assertEqual((second.settings.gain_db, second.settings.gamma), (11.0, 2.0))
        self.assertTrue(self.log.has("Synthetic camera (DEMO1): saved settings restored "
                                     "(exposure 15000 µs, gain 0 dB, 30 fps, Reference "
                                     "view, LabVIEW flow)."))
        restarted.labview_button.setChecked(False)  # back to the settings kept for it
        self.assertEqual(first.engine.settings().mode, "Previous")
        restarted.camera_combo.setCurrentIndex(1)
        self.assertEqual((restarted.gain.value(), restarted.gamma.value()), (11.0, 2.0))

    def test_a_real_camera_gets_its_own_profile_after_going_live(self):
        """Switching to live capture saves the demo profile and restores the camera's.

        Args: None.
        Returns: None.
        """
        w = self.window
        shown = self.channels[0]
        saved = replace(qt_app.profiles.default_profile(CameraInfo("111", "BFS")).settings,
                        exposure_us=5000.0, mode="Previous")
        FakeSettings.store["profiles/flir/111"] = qt_app.profiles.Profile(saved).dumps()
        w.gain.spin.setValue(8.0)
        shown.engine.update_settings(record=False, mode="Raw")  # as _switch_live does
        shown.engine.camera_info = CameraInfo("111", "BFS")
        shown.engine.events.put(("switched_live", [CameraInfo("111", "BFS"),
                                                   CameraInfo("222", "BFS")]))
        w._handle_events()
        s = shown.engine.settings()
        self.assertEqual((s.exposure_us, s.gain_db, s.mode), (5000.0, 0.0, "Previous"))
        self.assertEqual((w.exposure.value(), self.checked_mode()), (5000, "Previous"))
        self.assertEqual(stored(KEY1)["settings"]["gain_db"], 8.0)
        self.assertTrue(self.log.has("BFS (111): saved settings restored"))
        self.assertEqual(w.channels[1].profile_key, "profiles/flir/222")
        w.gain.spin.setValue(2.0)
        w.close()
        self.assertEqual(stored("profiles/flir/111")["settings"]["gain_db"], 2.0)
        self.assertNotIn("profiles/flir/222", FakeSettings.store)

    # ----- presets ------------------------------------------------------------

    def test_save_as_lists_the_preset(self):
        """Save as asks for a name, stores the settings and selects the preset.

        Args: None.
        Returns: None.
        """
        w = self.window
        self.assertEqual(w.preset_combo.count(), 0)
        self.assertEqual(w.preset_combo.placeholderText(), "No saved presets")
        self.assertFalse(w.preset_apply_button.isEnabled())
        self.assertFalse(w.preset_delete_button.isEnabled())
        w.gain.spin.setValue(5.0)
        self.save_preset("  Bright  ")
        self.assertEqual(self.prompts, [("Save preset",
                                         "Save the settings of Synthetic camera (DEMO1) as:",
                                         "")])
        self.assertEqual(w.preset_combo.currentText(), "Bright")
        self.assertTrue(w.preset_apply_button.isEnabled())
        data = stored("presets/bright")
        self.assertEqual((data["name"], data["source_model"], data["settings"]["gain_db"]),
                         ("Bright", "Synthetic camera", 5.0))
        self.assertIn("Saved from Synthetic camera.",
                      w.preset_combo.itemData(0, Qt.ToolTipRole))
        self.assertTrue(self.log.has("Preset saved: Bright from Synthetic camera (DEMO1)"))
        self.dialogs.question.assert_not_called()
        identity = CameraIdentity(model="BFS-U3-63S4M")  # a detected model names the source
        self.channels[0].identity = identity
        self.save_preset("Detected")
        self.assertEqual(stored("presets/detected")["source_model"], identity.short_name())

    def test_cancel_or_a_blank_name_saves_nothing(self):
        """Cancel leaves the store alone; a blank name is refused.

        Args: None.
        Returns: None.
        """
        with self.name_prompt("Kept", accept=False):
            self.window.save_preset_as()
        self.save_preset("   ")
        self.assertFalse(any(key.startswith("presets/") for key in FakeSettings.store))
        self.assertTrue(self.log.has("Preset not saved: it needs a name."))

    def test_an_existing_name_is_replaced_only_after_confirmation(self):
        """Saving over a name asks first; No keeps the old preset, Yes replaces it.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.gain.spin.setValue(1.0)
        self.save_preset("Bright")
        w.gain.spin.setValue(2.0)
        self.save_preset("BRIGHT", answer=self.dialogs.No)
        title, text = self.dialogs.question.call_args.args[1:3]
        self.assertEqual(title, "Replace preset?")
        self.assertIn("“Bright” already exists", text)
        self.assertEqual(stored("presets/bright")["settings"]["gain_db"], 1.0)
        self.assertEqual(self.prompts[-1][2], "Bright")  # the chosen name is offered
        self.save_preset("BRIGHT", answer=self.dialogs.Yes)
        self.assertEqual(stored("presets/bright")["settings"]["gain_db"], 2.0)
        self.assertEqual([w.preset_combo.itemText(i) for i in range(w.preset_combo.count())],
                         ["BRIGHT"])

    def test_apply_changes_only_the_displayed_camera_and_keeps_it(self):
        """A preset saved on camera 1 and applied to camera 2 leaves camera 1 alone.

        Args: None.
        Returns: None.
        """
        w = self.window
        first, second = self.channels
        w.exposure.spin.setValue(10000)
        w.gain.spin.setValue(5.0)
        w.mode_group.buttons()[1].click()
        w.average.spin.setValue(8)
        self.save_preset("Mine")
        w.gain.spin.setValue(1.0)  # camera 1 moves on
        self.display(1)
        w.preset_apply_button.click()
        s = second.engine.settings()
        self.assertEqual((s.exposure_us, s.gain_db, s.mode, s.average),
                         (10000.0, 5.0, "Previous", 8))
        self.assertEqual((w.gain.value(), self.checked_mode()), (5.0, "Previous"))
        self.assertEqual(first.profile().settings.gain_db, 1.0)
        self.assertEqual(stored(KEY2)["settings"]["gain_db"], 5.0)  # kept at once
        self.assertTrue(self.log.has("Preset applied: Mine to Synthetic camera (DEMO2) "
                                     "(exposure 10000 µs, gain 5 dB, 30 fps, Previous view)"))

    def test_a_preset_from_another_model_is_clamped_to_this_camera(self):
        """Processing values apply in full; camera values are fitted and logged.

        Args: None.
        Returns: None.
        """
        w = self.window
        base = qt_app.profiles.default_profile(CameraInfo("9", "BFS-U3-200S6M")).settings
        preset = qt_app.profiles.Profile(replace(
            base, exposure_us=45000.0, gain_db=27.0, fps=12.0, mode="Reference",
            average=6, smoothing=3, smoothing_method="Median", contrast=40.0))
        FakeSettings.store["presets/wide"] = json.dumps(
            {**preset.to_data(), "name": "Wide", "source_model": "Blackfly S BFS-U3-200S6M"})
        self.channels[0].limits = {"ExposureTime": (50.0, 30000.0), "Gain": (0.0, 24.0)}
        w._refresh_presets(select="Wide")
        self.assertIn("Saved from Blackfly S BFS-U3-200S6M.",
                      w.preset_combo.itemData(0, Qt.ToolTipRole))
        w.apply_saved_preset()
        s = self.channels[0].engine.settings()
        self.assertEqual((s.exposure_us, s.gain_db, s.fps), (30000.0, 24.0, 12.0))
        self.assertEqual((s.mode, s.average, s.smoothing, s.smoothing_method, s.contrast),
                         ("Reference", 6, 3, "Median", 40.0))
        self.assertEqual((w.exposure.value(), w.gain.value()), (30000, 24.0))
        self.assertTrue(self.log.has(
            "Preset Wide: Exposure 45000 µs is outside this camera's range "
            "(50–30000 µs); using 30000 µs."))
        self.assertTrue(self.log.has(
            "Preset Wide: Gain 27 dB is outside this camera's range (0–24 dB); using 24 dB."))
        self.assertTrue(self.log.has("Preset applied: Wide to Synthetic camera (DEMO1)"))

    def test_a_preset_brings_its_labview_state(self):
        """A preset saved in the LabVIEW flow turns it on where it is applied.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.gamma.spin.setValue(1.5)
        w.labview_button.setChecked(True)
        self.save_preset("VI")
        self.display(1)
        w.apply_saved_preset("VI")
        self.assertTrue(w.labview_button.isChecked())
        w.labview_button.setChecked(False)
        self.assertEqual(self.channels[1].engine.settings().gamma, 1.5)

    def test_apply_without_a_choice_or_a_vanished_preset(self):
        """Nothing chosen does nothing; a preset removed meanwhile is reported.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.apply_saved_preset()
        self.save_preset("Gone")
        del FakeSettings.store["presets/gone"]
        w.apply_saved_preset()
        self.assertTrue(self.log.has("Preset Gone is no longer saved."))
        self.assertEqual(w.preset_combo.count(), 0)
        self.assertEqual(self.channels[0].profile(), self.channels[0].default_profile())

    def test_delete_asks_first(self):
        """Delete removes the preset only after Yes; cameras keep their settings.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.delete_preset()  # nothing chosen
        w.gain.spin.setValue(3.0)
        self.save_preset("Old")
        self.save_preset("Other")
        w.preset_combo.setCurrentText("Old")
        w.preset_delete_button.click()
        title, text = self.dialogs.question.call_args.args[1:3]
        self.assertEqual((title, text.split("?")[0]),
                         ("Delete preset?", "Delete the preset “Old”"))
        self.assertIn("presets/old", FakeSettings.store)
        self.dialogs.question.return_value = self.dialogs.Yes
        w.preset_delete_button.click()
        self.assertNotIn("presets/old", FakeSettings.store)
        self.assertEqual([w.preset_combo.itemText(i) for i in range(w.preset_combo.count())],
                         ["Other"])
        self.assertEqual(w.preset_combo.currentIndex(), -1)
        self.assertEqual(w.preset_combo.placeholderText(), "Choose a preset")
        self.assertFalse(w.preset_apply_button.isEnabled())
        self.assertEqual(self.channels[0].profile().settings.gain_db, 3.0)
        self.assertTrue(self.log.has("Preset deleted: Old"))

    # ----- reset ----------------------------------------------------------------

    def test_reset_returns_the_shown_camera_to_its_defaults_after_confirmation(self):
        """No keeps everything; Yes resets this camera only and forgets its profile.

        Args: None.
        Returns: None.
        """
        w = self.window
        first, second = self.channels
        w.exposure.spin.setValue(20000)
        w.labview_button.setChecked(True)
        self.display(1)
        w.gain.spin.setValue(4.0)
        self.display(0)
        self.assertIn(KEY1, FakeSettings.store)
        w.reset_camera_button.click()
        title, text = self.dialogs.question.call_args.args[1:3]
        self.assertEqual(title, "Reset camera to defaults?")
        self.assertIn("Synthetic camera (DEMO1)", text)
        self.assertIn("exposure 8000 µs, gain 0 dB, 30 fps, Raw view", text)
        self.assertEqual(first.profile().settings.exposure_us, 20000.0)
        self.dialogs.question.return_value = self.dialogs.Yes
        w.reset_camera_button.click()
        self.assertEqual(first.profile(), first.default_profile())
        self.assertEqual((w.exposure.value(), self.checked_mode()), (8000, "Raw"))
        self.assertFalse(w.labview_button.isChecked())
        self.assertNotIn(KEY1, FakeSettings.store)
        self.assertEqual(second.profile().settings.gain_db, 4.0)
        self.assertTrue(self.log.has("Camera reset to defaults: Synthetic camera (DEMO1)"))
        w.close()
        self.assertNotIn(KEY1, FakeSettings.store)  # nothing to save afterwards

    # ----- layout ---------------------------------------------------------------

    def test_every_control_range_is_a_valid_profile_range(self):
        """Nothing a control can be set to is refused when it is read back.

        Args: None.
        Returns: None.
        """
        w = self.window
        for widget, name in ((w.noise_reduction, "noise_reduction"), (w.average, "average"),
                             (w.ref_frames, "ref_frames"), (w.ref_refresh, "ref_refresh_s"),
                             (w.frames_back, "frames_back"),
                             (w.outlier_filter, "outlier_filter"),
                             (w.smoothing, "smoothing"),
                             (w.difference_gain, "difference_gain"),
                             (w.average_multiplier, "average_multiplier"),
                             (w.clip, "clip_percent"), (w.min_range, "min_range"),
                             (w.threshold, "threshold"), (w.steady_range, "steady_range_s"),
                             (w.peak_hold, "peak_hold_s"),
                             (w.normalize_window, "normalize_window"), (w.gamma, "gamma"),
                             (w.brightness, "brightness"),
                             (w.display_gain, "display_gain"), (w.contrast, "contrast"),
                             (w.exposure, "exposure_us"), (w.gain, "gain_db"),
                             (w.fps, "fps")):
            low, high = qt_app.profiles.RANGES[name]
            self.assertLessEqual(low, widget.spin.minimum(), name)
            self.assertGreaterEqual(high, widget.spin.maximum(), name)
        self.assertEqual(tuple(w.smoothing_combo.itemText(i)
                               for i in range(w.smoothing_combo.count())),
                         qt_app.profiles.CHOICES["smoothing_method"])


class SimpleUiProfileTests(ProfileWindowTests):
    """The profile and preset controls are part of the basic controls."""

    simple_ui = True

    def test_the_camera_switcher_and_preset_controls_are_shown(self):
        """With two cameras: the switcher, whose settings, presets and reset.

        Args: None.
        Returns: None.
        """
        w = self.window
        w.show()
        for widget in (w.cameras_box, w.camera_combo, w.profile_label, w.preset_combo,
                       w.preset_apply_button, w.preset_save_button,
                       w.preset_delete_button, w.reset_camera_button):
            self.assertTrue(widget.isVisibleTo(w), widget)

    def test_one_camera_hides_only_the_process_rows_and_name_line(self):
        """A single camera keeps the Display drop-down and presets, not the name line.

        Args: None.
        Returns: None.
        """
        w = self.make_window(count=1)
        w.show()
        self.assertFalse(w.process_note.isVisibleTo(w))
        self.assertFalse(w.profile_label.isVisibleTo(w))
        for widget in (w.cameras_box, w.camera_combo, w.preset_combo,
                       w.preset_save_button, w.reset_camera_button):
            self.assertTrue(widget.isVisibleTo(w), widget)


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
                self.assertEqual(self.window.check_button.accessibleName(),
                                 "Checking for updates…")
                self.assertIn("Checking for updates…", self.window.check_button.toolTip())
                self.window.updates.busy = False
                self.post(kind, payload)
                self.assertTrue(self.window.check_button.isEnabled())
                self.assertEqual(self.window.check_button.accessibleName(), "Check for updates")

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
        FakeSettings.store = {}
        for patcher in (
                mock.patch.object(qt_app, "QSettings", FakeSettings),
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
        self.demos.assert_called_once()
        store = self.demos.call_args.args[2]
        self.assertEqual(self.demos.call_args.args[:2], (self.log, 2))
        self.assertIsInstance(store, qt_app.profiles.ProfileStore)
        self.assertIsInstance(store.store, FakeSettings)  # never the real store in tests
        self.assertEqual(self.main_window.call_args.args[0], [self.channel])
        self.assertEqual(self.main_window.call_args.args[2:],
                         (True, False, self.reporting, store))
        self.window.show.assert_called_once()
        self.engine.start.assert_called_once()
        self.reporting.start_uploads.assert_called_once_with()
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
        self.assertEqual(self.cameras.call_args.args[0], self.log)
        store = self.cameras.call_args.args[1]
        self.assertIs(self.main_window.call_args.args[5], store)  # the same store saves them
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
        self.assertEqual(self.demos.call_args.args[:2], (self.log, 1))
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
        self.assertEqual(self.window.feature_button.accessibleName(), "Sending request…")
        self.assertIn("Sending request…", self.window.feature_button.toolTip())
        self.assertEqual(self.window.statusBar().currentMessage(),
                         "Sending the feature request…")
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
        self.assertEqual(self.window.feature_button.accessibleName(), "Request feature")
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


class PhoneWindowTests(unittest.TestCase):
    """A Blackfly and an iPhone (displayed, over a scripted USB link) in one window."""

    def setUp(self):
        """Build the window over a fake FLIR camera and a fake iPhone link.

        Args: None.
        Returns: None.
        """
        FakeSettings.store = {}
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        system = FakeSystem([FakeCamera(serial="1", model="Blackfly S BFS-U3-63S4M")])
        phone = CameraInfo("iphone-4d5e801e", "iPhone (USB)", kind="iphone",
                           path="00008110-001A2B3C4D5E801E")
        self.links = []
        real_make = sources.make_session

        def make(log, info, exposure_us=8000.0, gain_db=0.0, fps=30.0):
            """Give iPhones a scripted link; other cameras their real session.

            Args: as sources.make_session.
            Returns: An unopened session.
            """
            if info.kind != "iphone":
                return real_make(log, info, exposure_us, gain_db, fps)
            return iphone.IPhoneSession(log, info, exposure_us, gain_db, fps,
                                        connector=self.connect)

        for patcher in (
                mock.patch.object(qt_app, "QSettings", FakeSettings),
                mock.patch.object(qt_app, "QMessageBox", make_dialogs()),
                mock.patch.object(qt_app.updater, "UpdateWorker", FakeUpdateWorker),
                mock.patch.object(qt_app.updater, "unavailable_reason",
                                  return_value="disabled in tests"),
                mock.patch.object(paths, "OUTPUT_DIR", tmp.name),
                mock.patch.object(camera, "PySpin", make_pyspin(system)),
                mock.patch.object(capture, "import_enumerator",
                                  return_value=(FakeEnumerator([]), "")),
                mock.patch.object(iphone, "list_iphones", return_value=[phone]),
                mock.patch.object(sources, "make_session", make)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.log = FakeLog()
        self.channels = [Channel(self.log, info) for info in sources.detect_cameras(self.log)]
        self.flir, self.phone = self.channels
        self.phone.open()
        self.window = qt_app.MainWindow(self.channels, self.log, False, False)
        self.window.frame_timer.stop()
        self.window.status_timer.stop()
        self.window.tip_timer.stop()
        self.addCleanup(self.cleanup_window)
        self.window._handle_events()  # identity and limits queued by the open
        self.phone.engine.start()  # the worker sends the phone's commands

    def connect(self, info):
        """Hand out a fresh scripted link that greets like the app.

        Args: info is the phone's CameraInfo.
        Returns: A FakeSocket.
        """
        self.links.append(phone_socket())
        return self.links[-1]

    def cleanup_window(self):
        """Close the window, which stops and waits for every engine.

        Args: None.
        Returns: None.
        """
        self.window.close()
        self.window.deleteLater()

    def sent(self, command):
        """Wait until the phone was sent a command.

        Args: command is the dict expected.
        Returns: True when it arrived.
        """
        return wait_for(lambda: command in commands(self.links[-1]))

    def test_phone_section_shows_and_sends(self):
        """The iPhone section appears for the phone and every control reaches it.

        Args: None.
        Returns: None.
        """
        window = self.window
        toggle, body = window.phone_section
        self.assertIs(window.current, self.phone)
        self.assertFalse(toggle.isHidden())
        self.assertFalse(body.isHidden())
        for widget in (window.exposure, window.gain, window.fps):
            self.assertTrue(widget.isEnabled())
        self.assertIn("phone's ISO", " ".join(window.gain.toolTip().split()))
        self.assertIn("ESPI Camera app", " ".join(window.exposure.toolTip().split()))
        self.assertEqual(window.phone_size.count(), 2)
        self.assertEqual(window.phone_fps.itemData(window.phone_fps.count() - 1), 60.0)
        window.phone_crop.activated.emit(2)
        window.phone_send_cap.activated.emit(2)
        window.phone_size.activated.emit(0)
        window.phone_autofocus.click()
        window.phone_focus.spin.setValue(0.25)
        for command in ({"crop_width": 1024, "crop_height": 1024}, {"max_send_fps": 15.0},
                        {"width": 1280, "height": 720}, {"refocus": True},
                        {"lens_position": 0.25}):
            self.assertTrue(self.sent(command), command)
        window.phone_fps.activated.emit(window.phone_fps.findData(15.0))
        self.assertEqual(self.phone.engine.settings().fps, 15.0)
        self.assertEqual(window.fps.spin.value(), 15.0)  # sent with the next frame
        window.show_channel(self.flir)
        self.assertTrue(toggle.isHidden())
        self.assertTrue(body.isHidden())
        self.assertNotIn("ISO", window.gain.toolTip())

    def test_collapsed_section_stays_collapsed(self):
        """Switching cameras keeps a collapsed iPhone section collapsed.

        Args: None.
        Returns: None.
        """
        window = self.window
        toggle, body = window.phone_section
        toggle.setChecked(False)
        window.show_channel(self.flir)
        window.show_channel(self.phone)
        self.assertFalse(toggle.isHidden())
        self.assertTrue(body.isHidden())

    def test_phone_settings_follow_the_phone(self):
        """Settings changed on the phone move the controls and the engine.

        Args: None.
        Returns: None.
        """
        window = self.window
        window._handle_event(self.phone, "camera_settings", {
            "exposure_us": 1234.0, "gain_db": 6.0, "fps": 12.0, "lens_position": 0.7,
            "width": 1280, "height": 720, "crop_width": 512, "crop_height": 512,
            "max_send_fps": 10.0})
        settings = self.phone.engine.settings()
        self.assertEqual((settings.exposure_us, settings.gain_db, settings.fps),
                         (1234.0, 6.0, 12.0))
        self.assertEqual((window.exposure.spin.value(), window.gain.spin.value()),
                         (1234.0, 6.0))
        self.assertEqual(window.phone_size.currentData(), "1280x720")
        self.assertEqual(window.phone_crop.currentData(), 512)
        self.assertEqual(window.phone_send_cap.currentData(), 10.0)
        self.assertEqual(window.phone_fps.currentData(), 12.0)  # added to the list
        self.assertAlmostEqual(window.phone_focus.spin.value(), 0.7)
        window._handle_event(self.phone, "camera_settings", {"width": 1920, "height": 1080})
        self.assertEqual(window.phone_size.currentData(), "1920x1080")  # not the first item
        window.show_channel(self.flir)
        window._handle_event(self.phone, "camera_settings", {"lens_position": 0.1})
        self.assertEqual(self.phone.camera_settings["lens_position"], 0.1)
        self.assertAlmostEqual(window.phone_focus.spin.value(), 0.7)  # not displayed
        window.show_channel(self.phone)
        self.assertAlmostEqual(window.phone_focus.spin.value(), 0.1)

    def test_capture_mode_choice_starts_the_measurement_over(self):
        """Choosing Bayer RAW resets averaging and the previous frame; a colour does not.

        Args: None.
        Returns: None.
        """
        window = self.window
        engine = self.phone.engine
        self.assertFalse(any(entry.endswith("Reset averaging.") for entry in self.log.lines))
        with mock.patch.object(engine, "request", wraps=engine.request) as request:
            window.phone_capture.activated.emit(window.phone_capture.findData(True))
            self.assertEqual(request.call_args_list,
                             [mock.call("reset_average"), mock.call("reset_previous"),
                              mock.call("camera_command", raw=True)])
            self.assertTrue(self.sent({"raw": True}))
            for line in ("Reset averaging.", "Reset previous frame."):
                self.assertTrue(wait_for(lambda line=line: any(
                    entry.endswith(line) for entry in self.log.lines)), line)
            request.reset_mock()
            window.phone_raw_channel.activated.emit(
                window.phone_raw_channel.findData("green"))
            self.assertEqual(request.call_args_list,
                             [mock.call("camera_command", raw_channel="green")])
        self.assertTrue(self.sent({"raw_channel": "green"}))

    def test_capture_mode_follows_the_phone(self):
        """RAW mode and colour reported by the phone select the matching choices.

        Args: None.
        Returns: None.
        """
        window = self.window
        self.assertIs(window.phone_capture.currentData(), False)
        self.assertEqual(window.phone_raw_channel.currentData(), "red")
        window._handle_event(self.phone, "camera_settings",
                             {"raw": True, "raw_channel": "green"})
        self.assertIs(window.phone_capture.currentData(), True)
        self.assertEqual(window.phone_raw_channel.currentData(), "green")
        self.assertEqual((self.phone.camera_settings["raw"],
                          self.phone.camera_settings["raw_channel"]), (True, "green"))
        window._handle_event(self.phone, "camera_settings", {"raw": False})
        self.assertIs(window.phone_capture.currentData(), False)
        self.assertEqual(window.phone_raw_channel.currentData(), "green")  # unchanged


class PhoneLensWindowTests(unittest.TestCase):
    """One iPhone with three back cameras: one drop-down entry each, one streaming."""

    def setUp(self):
        """Build the window over a phone listed as ultra wide, wide and telephoto.

        Args: None.
        Returns: None.
        """
        FakeSettings.store = {}
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        udid = "00008110-001A2B3C4D5E801E"
        self.infos = [
            CameraInfo("iphone-4d5e801e", "iPhone Wide 1× (USB)", kind="iphone",
                       path=udid, lens="wide"),
            CameraInfo("iphone-4d5e801e-ultra-wide", "iPhone Ultra Wide 0.5× (USB)",
                       kind="iphone", path=udid, lens="ultra_wide"),
            CameraInfo("iphone-4d5e801e-telephoto", "iPhone Telephoto 5× (USB)",
                       kind="iphone", path=udid, lens="telephoto")]
        self.links = {}
        real_make = sources.make_session

        def make(log, info, exposure_us=8000.0, gain_db=0.0, fps=30.0):
            """Give the phone a scripted link that greets as the app does.

            Args: as sources.make_session.
            Returns: An unopened session.
            """
            return iphone.IPhoneSession(log, info, exposure_us, gain_db, fps,
                                        connector=self.connect)

        for patcher in (
                mock.patch.object(qt_app, "QSettings", FakeSettings),
                mock.patch.object(qt_app, "QMessageBox", make_dialogs()),
                mock.patch.object(qt_app.updater, "UpdateWorker", FakeUpdateWorker),
                mock.patch.object(qt_app.updater, "unavailable_reason",
                                  return_value="disabled in tests"),
                mock.patch.object(paths, "OUTPUT_DIR", tmp.name),
                mock.patch.object(sources, "make_session", make)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(iphone._lens_cache.clear)
        self.log = FakeLog()
        self.channels = [Channel(self.log, info) for info in self.infos]
        self.wide, self.ultra, self.tele = self.channels
        self.wide.open()
        self.window = qt_app.MainWindow(self.channels, self.log, False, False)
        self.window.frame_timer.stop()
        self.window.status_timer.stop()
        self.window.tip_timer.stop()
        self.addCleanup(self.cleanup_window)
        self.wide.engine.start()

    def connect(self, info):
        """Hand out a scripted link for the camera asked for.

        Args: info is the camera's CameraInfo.
        Returns: A FakeSocket.
        """
        self.links[info.lens] = lens_socket(info.lens)
        return self.links[info.lens]

    def cleanup_window(self):
        """Close the window, which stops and waits for every engine.

        Args: None.
        Returns: None.
        """
        self.window.close()
        self.window.deleteLater()

    def test_every_lens_is_a_camera_but_only_one_can_stream(self):
        """The drop-down names each lens; the other lenses cannot be processed
        alongside the displayed one.

        Args: None.
        Returns: None.
        """
        window = self.window
        self.assertEqual([window.camera_combo.itemText(i) for i in range(3)],
                         [info.label for info in self.infos])
        self.assertIn("Ultra Wide 0.5×", window.camera_combo.itemText(1))
        self.assertEqual([check.isChecked() for check in window.camera_checks],
                         [True, False, False])
        self.assertEqual([check.isEnabled() for check in window.camera_checks],
                         [False, False, False])

    def test_choosing_a_lens_hands_the_phone_over(self):
        """The shown lens stops, the chosen one opens and asks the app for its camera.

        Args: None.
        Returns: None.
        """
        window = self.window
        window.camera_combo.setCurrentIndex(1)
        self.assertIs(window.current, self.ultra)
        self.assertFalse(self.wide.active)
        self.assertTrue(self.ultra.active)
        self.assertIn("Processing off: iPhone Wide 1× (USB) (iphone-4d5e801e) "
                      "(the phone streams one camera at a time)", self.log.lines)
        self.assertTrue(wait_for(lambda: "ultra_wide" in self.links and any(
            command.get("lens") == "ultra_wide"
            for command in commands(self.links["ultra_wide"]))))
        self.assertTrue(wait_for(lambda: self.ultra.engine.status().get("camera")
                                 is not None))
        self.assertEqual([check.isEnabled() for check in window.camera_checks],
                         [False, False, False])
        self.assertEqual([check.isChecked() for check in window.camera_checks],
                         [False, True, False])

    def test_a_background_lens_cannot_be_processed(self):
        """Asking to process another lens of the displayed phone is refused.

        Args: None.
        Returns: None.
        """
        self.window._set_processing(self.tele, True)
        self.assertFalse(self.tele.active)
        self.assertTrue(self.wide.active)
        self.assertFalse(self.window.camera_checks[2].isChecked())


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
                mock.patch.object(capture, "import_enumerator",
                                  return_value=(FakeEnumerator(
                                      [usb_live2(0), integrated_webcam(1), webcam(2)]), "")),
                mock.patch.object(capture.cv2, "VideoCapture",
                                  FakeCaptureFactory(frames=[interlaced_frame()])),
                mock.patch.object(iphone, "list_iphones", return_value=[])):
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
        self.assertFalse(window.single_field.isChecked())
        self.assertEqual(window.scaling.currentText(), "Fixed gain")
        self.assertTrue(window.difference_gain.isEnabled())
        self.assertFalse(window.clip.isEnabled())
        self.assertTrue(window.labview_button.isChecked())  # on by default
        self.assertEqual(window.mode_group.checkedButton().text(), "Reference")
        self.assertTrue(window.report_label.isHidden())
        window.show_channel(self.flir)
        for widget in (window.exposure, window.gain, window.fps):
            self.assertTrue(widget.isEnabled())
            self.assertNotIn("analog video", widget.toolTip())
            self.assertNotIn("Simulated", widget.toolTip())
            self.assertIn("Now:", widget.toolTip())
        self.assertTrue(window.single_field.isHidden())
        self.assertEqual(window.scaling.currentText(), "Auto stretch")
        self.assertFalse(window.difference_gain.isEnabled())
        self.assertTrue(window.clip.isEnabled())
        self.assertFalse(window.labview_button.isChecked())
        window.show_channel(self.usb)
        window.labview_button.setChecked(False)  # back to the analog camera's own
        s = self.usb.engine.settings()
        self.assertEqual((s.mode, s.scale_range, s.auto_reference), ("Raw", False, False))

    def test_single_field_checkbox_updates_engine(self):
        """Unticking the field toggle switches the engine to full frames.

        Args: None.
        Returns: None.
        """
        self.window.single_field.setChecked(True)
        self.assertTrue(self.usb.engine.settings().single_field)
        self.window.single_field.setChecked(False)
        self.assertFalse(self.usb.engine.settings().single_field)

    def test_unsupported_marker_in_camera_list(self):
        """Only the unrecognised webcam is marked; the laptop's own camera is absent.

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
                self.assertIn("Error reports: off", self.log.lines[-2])
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


    def test_reporting_starts_and_stops_log_uploads(self):
        """Starting logs the upload state and starts the uploader; stopping stops it.

        Args: None.
        Returns: None.
        """
        reporting = self.reporting()
        self.assertEqual(reporting.uploads.uploader.folder,
                         os.path.join(paths.OUTPUT_DIR, "logs"))
        with mock.patch.dict(os.environ, {qt_app.error_reports.ENABLE_ENV: ""}), \
                mock.patch.object(reporting.worker, "start"), \
                mock.patch.object(reporting.uploads, "start") as start:
            reporting.start(self.log)
            start.assert_not_called()  # not before the cameras are open
            reporting.start_uploads()
        start.assert_called_once()
        reporting.stop()
        self.assertTrue(reporting.uploads.stopping.is_set())
        # Error reports turned off (or on) never changes log uploads: only the token does.
        for reports, token, expected in (("0", "t", "Log uploads: on"),
                                         ("1", "", "Log uploads: off (no logs token in this copy)")):
            with mock.patch.dict(os.environ, {qt_app.error_reports.ENABLE_ENV: reports}), \
                    mock.patch.object(qt_app.log_uploads, "logs_token", return_value=token):
                reporting = self.reporting()
                with mock.patch.object(reporting.worker, "start"), \
                        mock.patch.object(reporting.uploads, "start"):
                    reporting.start(self.log)
                reporting.stop()
            self.assertIn(expected, self.log.lines)

    def test_menu_switch_leaves_log_uploads_alone(self):
        """The Help-menu switch is for error reports only; log uploads have no switch.

        Args: None.
        Returns: None.
        """
        with mock.patch.dict(os.environ, {qt_app.error_reports.ENABLE_ENV: "1"}):
            reporting = self.reporting()
            self.assertIs(reporting.uploads.uploader.serials.__self__, reporting.reporter)
            self.assertFalse(hasattr(reporting.uploads.uploader, "enabled"))
            with mock.patch.object(reporting.worker, "kick"), \
                    mock.patch.object(reporting.uploads, "kick") as kick:
                reporting.choose(False, self.log)
                reporting.choose(True, self.log)
            kick.assert_not_called()


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
        self.assertNotIn("logs", self.window.reports_action.toolTip())
        self.assertEqual(self.window.reports_action.text(), "Send error reports automatically")

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
        self.show(make_packet(sequence=2, gamma=2.0))
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
        window.show()  # showing fits the window to the screen; widen it after
        window.resize(2400, 900)
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
        self.engine._packet = FramePacket(3, np.zeros((48, 64), np.uint8), 1.0, "Raw", True)
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
        patcher = mock.patch.object(qt_app.codex, "open_cli", return_value="/bin/codex")
        self.open_cli = patcher.start()
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
        self.assertEqual(self.window.codex_button.accessibleName(), "Open in Codex")

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
        self.assertEqual(self.window.codex_button.accessibleName(), "Getting source…")
        self.assertIn("Getting source…", self.window.codex_button.toolTip())
        self.assertEqual(self.window.statusBar().currentMessage(),
                         "Getting the source code for Codex…")
        self.window.source.start.assert_called_once_with()
        FakeSettings.store["codex/intro_shown"] = True
        self.ready("Downloaded the source code")
        self.assertTrue(self.window.codex_button.isEnabled())
        self.assertEqual(self.window.codex_button.accessibleName(), "Open in Codex")
        self.assertIn("Downloaded the source code", self.log.lines)
        self.write_guide.assert_called_once_with(self.FOLDER)
        self.open_cli.assert_called_once_with(self.FOLDER)
        self.assertIn(f"Opened Codex on {self.FOLDER}", self.log.lines)
        self.assertIn("Opening Codex", self.window.statusBar().currentMessage())

    def test_ready_source_at_startup_opens_nothing(self):
        """The startup event only records the folder.

        Args: None.
        Returns: None.
        """
        before = list(self.log.lines)
        self.ready()
        self.assertEqual(self.window.source_folder, self.FOLDER)
        self.open_cli.assert_not_called()
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
        with mock.patch.object(self.window, "_codex_intro", return_value=False):
            self.window._open_codex()
        self.open_cli.assert_not_called()
        self.assertNotIn("codex/intro_shown", FakeSettings.store)
        with mock.patch.object(self.window, "_codex_intro", return_value=True) as intro:
            self.window._open_codex()
            self.window._open_codex()
        intro.assert_called_once_with()
        self.assertTrue(FakeSettings.store["codex/intro_shown"])
        self.assertEqual(self.open_cli.call_count, 2)

    def test_intro_buttons(self):
        """Open Codex goes on; How to install opens the install page; Cancel stops.

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
        self.window._open_codex()
        self.open_cli.assert_not_called()
        self.assertIn("read-only", self.dialogs.warning.call_args.args[2])

    def test_missing_codex_offers_the_install_page(self):
        """When Codex is not installed, a dialog says so and links the install page.

        Args: None.
        Returns: None.
        """
        self.ready()
        FakeSettings.store["codex/intro_shown"] = True
        self.open_cli.return_value = None
        for choice, opened in ((1, 0), (0, 1)):
            self.buttons(choice, 2)
            with mock.patch.object(qt_app.QDesktopServices, "openUrl") as open_url:
                self.window._open_codex()
            self.assertEqual(open_url.call_count, opened)
            self.assertEqual(self.dialogs.call_args.args[1], "Codex not found")
        self.assertEqual(open_url.call_args.args[0].toString(), qt_app.codex.INSTALL_URL)
        self.assertNotIn(f"Opened Codex on {self.FOLDER}", self.log.lines)

    def test_terminal_that_cannot_start_warns(self):
        """A terminal that fails to start is reported, not mistaken for a missing Codex.

        Args: None.
        Returns: None.
        """
        self.ready()
        FakeSettings.store["codex/intro_shown"] = True
        self.open_cli.side_effect = OSError("no terminal program was found")
        self.window._open_codex()
        self.dialogs.warning.assert_called_once()
        self.assertIn("no terminal program", self.dialogs.warning.call_args.args[2])
        self.assertNotIn(f"Opened Codex on {self.FOLDER}", self.log.lines)


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
        window.show()  # showing fits the window to the screen; widen it after
        window.resize(qt_app.DEFAULT_WIDTH, qt_app.DEFAULT_HEIGHT)
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


class FitRectTests(unittest.TestCase):
    """A window rectangle is shrunk and moved onto the usable screen area."""

    def test_a_window_that_fits_is_unchanged(self):
        """A rectangle already inside the area comes back the same.

        Args: None.
        Returns: None.
        """
        rect = QRect(10, 50, 600, 400)
        self.assertEqual(qt_app.fit_rect(rect, QRect(0, 40, 1280, 672)), rect)

    def test_an_oversized_window_fills_the_area(self):
        """A window larger than the screen shrinks to the area, taskbar excluded.

        Args: None.
        Returns: None.
        """
        self.assertEqual(qt_app.fit_rect(QRect(500, -20, 2100, 900), QRect(0, 40, 1280, 672)),
                         QRect(0, 40, 1280, 672))

    def test_a_window_hanging_off_the_corner_moves_back(self):
        """A window that fits but sticks out past the bottom right moves in.

        Args: None.
        Returns: None.
        """
        self.assertEqual(qt_app.fit_rect(QRect(1000, 500, 400, 300), QRect(0, 40, 1280, 672)),
                         QRect(880, 412, 400, 300))


class ScreenFitTests(unittest.TestCase):
    """The window fits whatever screen and resolution it is shown on."""

    def setUp(self):
        """Build a demo window with no saved layout (default size 2100x900).

        Args: None.
        Returns: None.
        """
        FakeSettings.store = {}
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for patcher in (
                mock.patch.object(qt_app, "QSettings", FakeSettings),
                mock.patch.object(qt_app.updater, "UpdateWorker", FakeUpdateWorker),
                mock.patch.object(qt_app.updater, "unavailable_reason",
                                  return_value="disabled in tests"),
                mock.patch.object(paths, "OUTPUT_DIR", tmp.name)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.log = FakeLog()
        engine = Engine(self.log)
        engine.use_demo()
        self.window = demo_window(engine, self.log)
        self.window.frame_timer.stop()
        self.window.status_timer.stop()
        self.window.tip_timer.stop()
        self.addCleanup(self.window.deleteLater)
        self.addCleanup(self.window.close)

    def assert_on_screen(self, area):
        """Check that the whole window, frame included, lies inside an area.

        Args: area is the usable screen QRect.
        Returns: None.
        """
        QApplication.processEvents()
        frame = self.window.frameGeometry()
        self.assertTrue(area.contains(frame), f"{frame} is not inside {area}")

    def test_showing_fits_the_default_size_to_a_small_screen(self):
        """The 2100-pixel default shrinks to a screen narrower than that.

        Args: None.
        Returns: None.
        """
        area = self.window.screen().availableGeometry()
        self.assertGreater(qt_app.DEFAULT_WIDTH, area.width())
        self.window.show()
        self.assert_on_screen(area)

    def test_a_resolution_change_while_running_refits_the_window(self):
        """A smaller usable area (resolution, scaling or taskbar) shrinks the window.

        Args: None.
        Returns: None.
        """
        self.window.show()
        screen = self.window.screen()
        smaller = QRect(0, 30, 700, 500)
        with mock.patch.object(QScreen, "availableGeometry", return_value=smaller):
            screen.availableGeometryChanged.emit(smaller)
            self.assert_on_screen(smaller)

    def test_showing_again_watches_the_screen_once(self):
        """Hiding and showing again neither loses nor doubles the screen watch.

        Args: None.
        Returns: None.
        """
        self.window.show()
        self.window.hide()
        self.window.show()
        screen = self.window.screen()
        self.assertIs(self.window._watched_screen, screen)
        smaller = QRect(0, 0, 680, 480)
        with mock.patch.object(QScreen, "availableGeometry", return_value=smaller), \
                mock.patch.object(self.window, "setGeometry",
                                  wraps=self.window.setGeometry) as set_geometry:
            screen.availableGeometryChanged.emit(smaller)
        self.assertEqual(set_geometry.call_count, 1)

    def test_a_maximized_window_is_left_alone(self):
        """Maximized already fits the screen, so its size is not touched.

        Args: None.
        Returns: None.
        """
        self.window.show()
        self.window.resize(3000, 2000)
        with mock.patch.object(self.window, "isMaximized", return_value=True):
            self.window.fit_to_screen()
        self.assertEqual((self.window.width(), self.window.height()), (3000, 2000))

    def test_controls_scroll_instead_of_stretching_the_window(self):
        """The window can be shorter than the controls, which then scroll.

        Args: None.
        Returns: None.
        """
        panel = self.window.controls_scroll.widget()
        self.assertLess(self.window.minimumSizeHint().height(), panel.sizeHint().height())
        self.window.show()
        QApplication.processEvents()
        self.assertTrue(self.window.controls_scroll.verticalScrollBar().isVisible())

    def test_wrapped_notes_get_their_full_height(self):
        """A wrapped note gets every line it needs, so no text is clipped.

        Args: None.
        Returns: None.
        """
        self.window.show()
        QApplication.processEvents()
        note = self.window.noise_note
        self.assertGreaterEqual(note.height(), note.heightForWidth(note.width()))


BIG_SCREEN = QRect(0, 0, 3000, 2000)


class UiScaleTests(unittest.TestCase):
    """The interface-size slider, menu and shortcuts scale the whole window live."""

    def setUp(self):
        """Build a demo window on a screen big enough for every interface size.

        Args: None.
        Returns: None.
        """
        FakeSettings.store = {}
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for patcher in (
                mock.patch.object(qt_app, "QSettings", FakeSettings),
                mock.patch.object(qt_app.updater, "UpdateWorker", FakeUpdateWorker),
                mock.patch.object(qt_app.updater, "unavailable_reason",
                                  return_value="disabled in tests"),
                mock.patch.object(QScreen, "availableGeometry", return_value=BIG_SCREEN),
                mock.patch.object(paths, "OUTPUT_DIR", tmp.name)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(qt_app.apply_app_scale, ui_scale.DEFAULT_PERCENT)
        self.log = FakeLog()
        self.window = self.make_window()
        self.base_point_size = qt_app.base_font().pointSizeF()

    def make_window(self):
        """Build a demo window with its timers stopped.

        Args: None.
        Returns: The MainWindow.
        """
        engine = Engine(self.log)
        engine.use_demo()
        window = demo_window(engine, self.log)
        window.frame_timer.stop()
        window.status_timer.stop()
        window.tip_timer.stop()
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        window.show()  # Qt re-fonts a widget only once it has been shown
        QApplication.processEvents()
        return window

    def test_the_default_is_the_standard_size(self):
        """Nothing stored means 100%: the font is untouched and the controls say so.

        Args: None.
        Returns: None.
        """
        window = self.window
        self.assertEqual(ui_scale.percent(), 100)
        self.assertEqual(QApplication.font().pointSizeF(), self.base_point_size)
        self.assertEqual(window.ui_scale_row.value(), 100)
        self.assertEqual(window.zoom_slider.maximumWidth(), 130)
        self.assertEqual(window.ui_scale_row.spin.maximumWidth(), 104)
        self.assertFalse(window.ui_scale_reset.isEnabled())
        self.assertFalse(window.scale_reset_action.isEnabled())
        self.assertTrue(window.scale_larger_action.isEnabled())
        self.assertTrue(window.scale_smaller_action.isEnabled())
        self.assertEqual(FakeSettings.store, {})

    def test_the_slider_row_explains_itself_in_the_display_section(self):
        """The slider is labeled, spans the allowed range and has a tooltip.

        Args: None.
        Returns: None.
        """
        row = self.window.ui_scale_row
        self.assertEqual((row.spin.minimum(), row.spin.maximum()), (70, 200))
        self.assertEqual(row.spin.suffix(), "%")
        self.assertIn("release", row.toolTip())
        self.assertIn("Now: 100%", row.toolTip())
        labels = [label.text() for label in self.window.controls_panel.findChildren(QLabel)]
        self.assertIn("Interface size", labels)

    def test_a_larger_size_scales_text_pixel_sizes_and_icons(self):
        """150% makes every font and every fixed pixel size one and a half times larger.

        Args: None.
        Returns: None.
        """
        window = self.window
        buttons = window._status_buttons()
        before_widths = [button.maximumWidth() for button in buttons]
        window.set_ui_scale(150)
        self.assertEqual(ui_scale.percent(), 150)
        self.assertAlmostEqual(QApplication.font().pointSizeF(), self.base_point_size * 1.5)
        for widget in (window, window.log_view, window.statusBar(), window.menuBar(),
                       window.zoom_label, window.ui_scale_row.spin):
            self.assertAlmostEqual(widget.font().pointSizeF(), self.base_point_size * 1.5, 1,
                                   msg=type(widget).__name__)
        self.assertEqual(window.zoom_slider.maximumWidth(), 195)
        self.assertEqual(window.zoom_label.minimumWidth(), 66)
        self.assertEqual(window.frame_notes.maximumHeight(), 72)
        self.assertEqual(window.ui_scale_row.spin.maximumWidth(), 156)
        self.assertEqual(window.status_actions_rule.maximumWidth(), 15)
        self.assertEqual(window.controls_panel.minimumWidth(), 480)
        self.assertEqual(window.controls_panel.layout().contentsMargins().left(), 15)
        self.assertIn("padding:9px", window.demo_banner.styleSheet())
        self.assertIn("margin-top:12px", window.frozen_banner.styleSheet())
        self.assertAlmostEqual(QToolTip.font().pointSizeF(), self.base_point_size * 1.5)
        for button, before in zip(buttons, before_widths):
            self.assertGreater(button.maximumWidth(), before)
            self.assertEqual(button.iconSize(), QSize(24, 24))
        self.assertGreater(window.controls_scroll.minimumWidth(), 480)

    def test_status_bar_items_are_measured_again(self):
        """Readouts, the tip cap, the message area and the buttons follow the font.

        Args: None.
        Returns: None.
        """
        window = self.window
        readout = window.speed_status.sizeHint().width()
        tip_cap = window.tip_label.maximumWidth()
        message_min = window.status_strip.flow.message_min
        window.set_ui_scale(150)
        self.assertGreater(window.speed_status.sizeHint().width(), readout * 1.3)
        self.assertGreater(window.tip_label.maximumWidth(), tip_cap * 1.3)
        self.assertGreater(window.status_strip.flow.message_min, message_min * 1.3)
        heights = {button.maximumHeight() for button in window._status_buttons()}
        self.assertEqual(len(heights), 1)
        self.assertEqual({button.minimumHeight() for button in window._status_buttons()}, heights)

    def test_dock_titles_follow_the_size(self):
        """A dock's title is drawn from an explicit font, which Qt would otherwise keep stale.

        Args: None.
        Returns: None.
        """
        self.window.set_ui_scale(150)
        for dock in (self.window.controls_dock, self.window.log_dock):
            self.assertAlmostEqual(dock.font().pointSizeF(), self.base_point_size * 1.5, 1)
            self.assertNotEqual(dock.font(), QApplication.font())

    def test_the_controls_and_log_keep_their_share_of_the_window(self):
        """The controls dock widens and the log dock deepens with the size.

        Args: None.
        Returns: None.
        """
        window = self.window
        window.resize(2400, 1200)
        QApplication.processEvents()
        with mock.patch.object(window, "resizeDocks") as resize_docks:
            window.controls_dock.setFixedWidth(400)
            window.log_dock.setFixedHeight(100)
            window.set_ui_scale(125)
        self.assertEqual(resize_docks.call_args_list[0].args[1], [500])
        self.assertEqual(resize_docks.call_args_list[1].args[1], [125])

    def test_returning_to_100_percent_restores_the_original_sizes(self):
        """Resetting undoes every change: fonts, pixel sizes, icons and styles.

        Args: None.
        Returns: None.
        """
        window = self.window
        widths = [button.maximumWidth() for button in window._status_buttons()]
        reserved = window.range_status.sizeHint().width()
        banner = window.demo_banner.styleSheet()
        window.set_ui_scale(175)
        window.ui_scale_reset.click()
        self.assertEqual(ui_scale.percent(), 100)
        self.assertEqual(QApplication.font().pointSizeF(), self.base_point_size)
        self.assertEqual([button.maximumWidth() for button in window._status_buttons()], widths)
        self.assertEqual(window.range_status.sizeHint().width(), reserved)
        self.assertEqual(window.demo_banner.styleSheet(), banner)
        self.assertEqual(window.zoom_slider.maximumWidth(), 130)
        self.assertEqual(window._status_buttons()[1].iconSize(), QSize(16, 16))
        self.assertFalse(window.ui_scale_reset.isEnabled())

    def test_a_change_is_remembered_and_logged(self):
        """The size is stored for the next run and named in the log.

        Args: None.
        Returns: None.
        """
        self.window.set_ui_scale(125)
        self.assertEqual(FakeSettings.store[ui_scale.SETTING_KEY], 125)
        self.assertIn("Interface size: 125%", self.log.drain())
        self.window.set_ui_scale(125)  # no change: nothing more to say
        self.assertNotIn("Interface size", self.log.drain())

    def test_a_stored_size_is_applied_before_the_window_is_built(self):
        """The next run starts at the remembered size, whatever type the store holds.

        Args: None.
        Returns: None.
        """
        FakeSettings.store[ui_scale.SETTING_KEY] = "150"  # ini files hand back text
        window = self.make_window()
        self.assertEqual(ui_scale.percent(), 150)
        self.assertEqual(window.zoom_slider.maximumWidth(), 195)
        self.assertEqual(window.ui_scale_row.value(), 150)
        self.assertTrue(window.ui_scale_reset.isEnabled())
        self.assertAlmostEqual(window.log_view.font().pointSizeF(), self.base_point_size * 1.5, 1)

    def test_a_damaged_stored_size_means_100_percent(self):
        """Junk or out-of-range values do not break startup.

        Args: None.
        Returns: None.
        """
        FakeSettings.store[ui_scale.SETTING_KEY] = "huge"
        self.make_window()
        self.assertEqual(ui_scale.percent(), 100)
        FakeSettings.store[ui_scale.SETTING_KEY] = 9000
        self.make_window()
        self.assertEqual(ui_scale.percent(), ui_scale.MAX_PERCENT)

    def test_dragging_the_slider_applies_only_on_release(self):
        """Mid-drag the value shows but the window waits, so the slider stays under the pointer.

        Args: None.
        Returns: None.
        """
        row = self.window.ui_scale_row
        row.slider.setSliderDown(True)
        row.slider.setValue(30)  # 150%
        self.assertEqual(row.spin.value(), 150)
        self.assertEqual(ui_scale.percent(), 100)
        row.slider.setSliderDown(False)  # the release
        self.assertEqual(ui_scale.percent(), 150)
        self.assertEqual(row.value(), 150)

    def test_keys_wheel_clicks_and_typing_apply_at_once(self):
        """Without a drag in progress, every change takes effect immediately.

        Args: None.
        Returns: None.
        """
        row = self.window.ui_scale_row
        row.slider.setValue(25)  # 125%: a click on the groove, a key or the wheel
        self.assertEqual(ui_scale.percent(), 125)
        row.spin.setValue(90)
        self.assertEqual(ui_scale.percent(), 90)
        self.assertLess(QApplication.font().pointSizeF(), self.base_point_size)

    def test_the_reset_button_returns_to_100_percent(self):
        """The Display section's button undoes any size and then disables itself.

        Args: None.
        Returns: None.
        """
        self.window.set_ui_scale(80)
        self.assertTrue(self.window.ui_scale_reset.isEnabled())
        self.window.ui_scale_reset.click()
        self.assertEqual(ui_scale.percent(), 100)
        self.assertEqual(self.window.ui_scale_row.value(), 100)
        self.assertFalse(self.window.ui_scale_reset.isEnabled())

    def test_the_view_menu_steps_resets_and_stops_at_the_ends(self):
        """Larger, Smaller and Reset move through the presets and disable at the limits.

        Args: None.
        Returns: None.
        """
        window = self.window
        window.scale_larger_action.trigger()
        self.assertEqual(ui_scale.percent(), 110)
        window.scale_smaller_action.trigger()
        window.scale_smaller_action.trigger()
        self.assertEqual(ui_scale.percent(), 90)
        window.scale_reset_action.trigger()
        self.assertEqual(ui_scale.percent(), 100)
        self.assertFalse(window.scale_reset_action.isEnabled())
        window.set_ui_scale(ui_scale.MAX_PERCENT)
        self.assertFalse(window.scale_larger_action.isEnabled())
        self.assertTrue(window.scale_smaller_action.isEnabled())
        window.set_ui_scale(ui_scale.MIN_PERCENT)
        self.assertFalse(window.scale_smaller_action.isEnabled())
        self.assertTrue(window.scale_larger_action.isEnabled())

    def test_the_menu_is_in_the_view_menu_with_shortcuts(self):
        """Ctrl++, Ctrl+= , Ctrl+- and Ctrl+0 are bound, and each action has a tooltip.

        Args: None.
        Returns: None.
        """
        window = self.window
        view = window.menuBar().actions()[0].menu()
        names = [action.text() for action in view.actions()]
        self.assertIn("Interface size", names)
        submenu = next(action.menu() for action in view.actions()
                       if action.text() == "Interface size")
        self.assertEqual([action.text() for action in submenu.actions()],
                         ["Larger", "Smaller", "Reset to 100%"])
        self.assertIn(QKeySequence("Ctrl+="), window.scale_larger_action.shortcuts())
        self.assertIn(QKeySequence(QKeySequence.ZoomIn), window.scale_larger_action.shortcuts())
        self.assertEqual(window.scale_smaller_action.shortcut(), QKeySequence(QKeySequence.ZoomOut))
        self.assertEqual(window.scale_reset_action.shortcut(), QKeySequence("Ctrl+0"))
        for action in submenu.actions():
            self.assertTrue(action.toolTip().strip(), action.text())

    def test_the_digit_shortcuts_still_mean_fit_and_actual_size(self):
        """Ctrl+0 is the size reset; the plain 0 key still fits the image.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(self.window.view, "fit") as fit:
            self.assertTrue(self.window.handle_key(Qt.Key_0))
        fit.assert_called_once()

    def test_a_size_too_large_for_the_screen_is_stepped_down(self):
        """The window's smallest size must fit the screen, so the size backs off.

        Args: None.
        Returns: None.
        """
        window = self.window
        window.set_ui_scale(150)
        window.layout().activate()
        smallest = window.minimumSizeHint()
        tight = QRect(0, 0, smallest.width() + 20, smallest.height() + 20)  # fits 150%
        with mock.patch.object(QScreen, "availableGeometry", return_value=tight):
            window.set_ui_scale(ui_scale.MAX_PERCENT)
            self.assertLess(ui_scale.percent(), ui_scale.MAX_PERCENT)
            self.assertGreaterEqual(ui_scale.percent(), 100)
            self.assertTrue(window._fits_screen())
        self.assertEqual(window.ui_scale_row.value(), ui_scale.percent())
        self.assertEqual(FakeSettings.store[ui_scale.SETTING_KEY], ui_scale.percent())

    def test_100_percent_is_never_lowered_for_a_small_screen(self):
        """A screen too small even for 100% leaves it alone, as before this feature.

        Args: None.
        Returns: None.
        """
        window = self.window
        with mock.patch.object(QScreen, "availableGeometry",
                               return_value=QRect(0, 0, 200, 150)):
            window.set_ui_scale(125)
        self.assertEqual(ui_scale.percent(), 100)
        self.assertEqual(window.ui_scale_row.value(), 100)
        self.assertNotIn(ui_scale.SETTING_KEY, FakeSettings.store)

    def test_a_smaller_screen_while_running_lowers_a_large_size(self):
        """A resolution change refits the window and steps the interface size down.

        Args: None.
        Returns: None.
        """
        window = self.window
        window.set_ui_scale(200)
        window.layout().activate()
        smallest = window.minimumSizeHint()
        self.log.drain()
        screen = window.screen()
        smaller = QRect(0, 0, smallest.width() - 100, smallest.height() + 400)
        with mock.patch.object(QScreen, "availableGeometry", return_value=smaller):
            screen.availableGeometryChanged.emit(smaller)
            QApplication.processEvents()
            self.assertTrue(window._fits_screen())
        self.assertLess(ui_scale.percent(), 200)
        self.assertEqual(FakeSettings.store[ui_scale.SETTING_KEY], ui_scale.percent())
        self.assertIn("limited to", self.log.drain())
        self.assertEqual(window.ui_scale_row.value(), ui_scale.percent())

    def test_a_screen_change_that_still_fits_keeps_the_size(self):
        """Nothing is lowered or logged when the window still fits.

        Args: None.
        Returns: None.
        """
        window = self.window
        window.set_ui_scale(125)
        self.log.drain()
        window.screen().availableGeometryChanged.emit(BIG_SCREEN)
        self.assertEqual(ui_scale.percent(), 125)
        self.assertNotIn("limited", self.log.drain())

    def test_dialogs_open_at_the_scaled_size(self):
        """The guide and request dialogs scale their opening size with the interface.

        Args: None.
        Returns: None.
        """
        self.window.set_ui_scale(150)
        guide = qt_app.SettingsGuideDialog(self.window)
        self.addCleanup(guide.deleteLater)
        self.assertEqual((guide.width(), guide.height()), (1020, 960))
        request = qt_app.FeatureRequestDialog(self.window)
        self.addCleanup(request.deleteLater)
        self.assertEqual((request.width(), request.height()), (840, 630))

    def test_a_status_label_keeps_its_reserved_width_in_any_font(self):
        """A FixedLabel reserves room for its samples again when the font changes.

        Args: None.
        Returns: None.
        """
        label = qt_app.FixedLabel("Camera: 999.9 fps")
        label.setText("Camera: 30.0 fps")
        before = label.sizeHint().width()
        font = QFont(label.font())
        font.setPointSizeF(font.pointSizeF() * 2)
        label.setFont(font)
        self.assertGreater(label.sizeHint().width(), before * 1.6)
        self.assertEqual(label.text(), "Camera: 30.0 fps")

    def test_a_growing_label_still_grows_after_a_font_change(self):
        """A label that widens for long text keeps doing so at a new size.

        Args: None.
        Returns: None.
        """
        label = qt_app.FixedLabel("short")
        label.grow_to_text()
        label.setText("a much longer camera name than the sample")
        wide = label.sizeHint().width()
        font = QFont(label.font())
        font.setPointSizeF(font.pointSizeF() * 1.5)
        label.setFont(font)
        self.assertGreater(label.sizeHint().width(), wide * 1.3)


class ScaledStyleTests(unittest.TestCase):
    """The application style, font helpers and icons that follow the interface size."""

    def tearDown(self):
        """Return the application to the standard size.

        Args: None.
        Returns: None.
        """
        qt_app.apply_app_scale(ui_scale.DEFAULT_PERCENT)

    def test_chrome_metrics_scale_but_frames_do_not(self):
        """Indicators, handles and icons grow with the size; a frame stays a hairline.

        Args: None.
        Returns: None.
        """
        qt_app.install_tooltip_delay()
        qt_app.apply_app_scale(100)
        style = QApplication.style()
        scaled_metrics = (QStyle.PM_IndicatorWidth, QStyle.PM_SliderLength,
                          QStyle.PM_ScrollBarExtent, QStyle.PM_SmallIconSize)
        before = [style.pixelMetric(metric) for metric in scaled_metrics]
        frame = style.pixelMetric(QStyle.PM_DefaultFrameWidth)
        qt_app.apply_app_scale(200)
        style = QApplication.style()
        self.assertIsInstance(style, qt_app.ScaledStyle)
        self.assertEqual([style.pixelMetric(metric) for metric in scaled_metrics],
                         [value * 2 for value in before])
        self.assertEqual(style.pixelMetric(QStyle.PM_DefaultFrameWidth), frame)

    def test_the_tooltip_delay_survives_a_size_change(self):
        """Replacing the style keeps the hover delay.

        Args: None.
        Returns: None.
        """
        qt_app.apply_app_scale(150)
        self.assertEqual(QApplication.style().styleHint(QStyle.SH_ToolTip_WakeUpDelay),
                         qt_app.TOOLTIP_DELAY_MS)

    def test_applying_the_same_size_again_keeps_the_style(self):
        """Nothing is re-polished when the size did not change.

        Args: None.
        Returns: None.
        """
        qt_app.apply_app_scale(125)
        style = QApplication.style()
        self.assertEqual(qt_app.apply_app_scale(125), 125)
        self.assertIs(QApplication.style(), style)

    def test_the_platform_font_is_left_alone_at_100_percent(self):
        """At the standard size, applying 100% does not replace the application font.

        Args: None.
        Returns: None.
        """
        qt_app.apply_app_scale(100)
        font = QFont(QApplication.font())
        with mock.patch.object(QApplication, "setFont") as set_font:
            qt_app.apply_app_scale(100)
        set_font.assert_not_called()
        self.assertEqual(QApplication.font(), font)
        with mock.patch.object(QApplication, "setFont") as set_font:
            qt_app.apply_app_scale(125)
            qt_app.apply_app_scale(100)  # undoing a size does set the font back
        self.assertEqual(set_font.call_count, 2)

    def test_sizes_are_clamped(self):
        """An out-of-range request is limited to the allowed range.

        Args: None.
        Returns: None.
        """
        self.assertEqual(qt_app.apply_app_scale(10_000), ui_scale.MAX_PERCENT)
        self.assertEqual(qt_app.apply_app_scale(1), ui_scale.MIN_PERCENT)

    def test_scaled_font_handles_point_and_pixel_sizes(self):
        """Fonts sized in points or in pixels both scale.

        Args: None.
        Returns: None.
        """
        points = QFont("Sans Serif")
        points.setPointSizeF(10)
        self.assertAlmostEqual(qt_app.scaled_font(points, 150).pointSizeF(), 15)
        pixels = QFont("Sans Serif")
        pixels.setPixelSize(12)
        self.assertEqual(qt_app.scaled_font(pixels, 150).pixelSize(), 18)
        self.assertEqual(points.pointSizeF(), 10)  # the original is untouched

    def test_nudged_font_differs_a_little(self):
        """A nudged font is not equal to the original but is the same size to the eye.

        Args: None.
        Returns: None.
        """
        points = QFont("Sans Serif")
        points.setPointSizeF(10)
        nudged = qt_app.nudged_font(points)
        self.assertNotEqual(nudged, points)
        self.assertAlmostEqual(nudged.pointSizeF(), 10, 1)
        pixels = QFont("Sans Serif")
        pixels.setPixelSize(12)
        self.assertEqual(qt_app.nudged_font(pixels).pixelSize(), 13)

    def test_the_base_font_is_taken_once(self):
        """The unscaled font is remembered, so a scaled font is never scaled again.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(qt_app, "_base_font", None):
            first = qt_app.base_font()
            qt_app.apply_app_scale(150)
            self.assertEqual(qt_app.base_font().pointSizeF(), first.pointSizeF())
            qt_app.apply_app_scale(150)
            self.assertAlmostEqual(QApplication.font().pointSizeF(), first.pointSizeF() * 1.5)
            qt_app.apply_app_scale(100)

    def test_status_icons_are_drawn_at_the_requested_size(self):
        """Icons come out at the button's icon size, with the drawing scaled to fit.

        Args: None.
        Returns: None.
        """
        for kind in ("refresh", "download", "idea", "code"):
            pixmap = qt_app._paint_status_icon(kind, qt_app.QColor("black"), 32)
            self.assertEqual((pixmap.width(), pixmap.height()), (32, 32), kind)
            icon = qt_app._status_icon(kind, qt_app.QColor("black"), 24)
            self.assertEqual(icon.actualSize(QSize(64, 64)), QSize(24, 24), kind)
        self.assertEqual(qt_app._paint_status_icon("idea", qt_app.QColor("black")).width(), 16)



def click_event(x, y, button=Qt.LeftButton):
    """Build a stand-in mouse event for click tests.

    Args: x and y are the widget position; button is the Qt button.
    Returns: A mock event.
    """
    event = mock.MagicMock()
    event.position.return_value = QPointF(x, y)
    event.button.return_value = button
    return event


def square(values=None, view="Reference", left=4, top=6, size=2, demo=False):
    """Build a SquarePacket.

    Args: values is the array (default a 2×2 with a 7), or None for waiting;
        view, left, top, size and demo are its fields.
    Returns: The SquarePacket.
    """
    if values is None and size == 2 and view != "waiting":
        values = np.array([[0, 1], [2, 7]], np.uint8)
    return SquarePacket(values, "Reference" if view == "waiting" else view,
                        left, top, size, demo)


class ViewClickAndOutlineTests(unittest.TestCase):
    """Clicks on the image and the pixel square's outline."""

    def setUp(self):
        """Create a 400x300 CPU view showing a 2000x1500 image (scale 0.2).

        Args: None.
        Returns: None.
        """
        self.view = qt_app.CpuImageView()
        self.view.resize(400, 300)
        self.view.show_packet(make_packet(width=2000, height=1500),
                              np.zeros((256, 3), np.uint8), None)
        self.clicks = []
        self.view.imageClicked.connect(lambda x, y: self.clicks.append((x, y)))

    def tearDown(self):
        """Delete the view.

        Args: None.
        Returns: None.
        """
        self.view.deleteLater()

    def test_a_click_reports_the_image_pixel(self):
        """Press and release in place reports the pixel under the pointer.

        Args: None.
        Returns: None.
        """
        self.view.mousePressEvent(click_event(100, 60))
        self.view.mouseReleaseEvent(click_event(101, 61))
        self.assertEqual(self.clicks, [(505, 305)])

    def test_drags_other_buttons_and_outside_clicks_are_not_clicks(self):
        """A drag, a right click, a release without press or a click off the
        image reports nothing.

        Args: None.
        Returns: None.
        """
        self.view.mousePressEvent(click_event(100, 60))
        self.view.mouseReleaseEvent(click_event(140, 60))
        self.view.mousePressEvent(click_event(100, 60, Qt.RightButton))
        self.view.mouseReleaseEvent(click_event(100, 60, Qt.RightButton))
        self.view.mouseReleaseEvent(click_event(100, 60))
        self.view.set_zoom(1.0)
        self.view.resize(800, 300)  # the image no longer fills the width
        self.view.mousePressEvent(click_event(5, 100))
        self.view.mouseReleaseEvent(click_event(5, 100))
        self.assertEqual(self.clicks, [])

    def outline_pixels(self):
        """Render the view and count outline-colored pixels.

        Args: None.
        Returns: The number of pixels in BOX_COLOR.
        """
        image = self.view.grab().toImage()
        return sum(image.pixelColor(x, y) == qt_app.BOX_COLOR
                   for x in range(image.width()) for y in range(image.height()))

    def test_outline_is_drawn_and_a_one_pixel_square_still_shows(self):
        """A square is outlined; a 1×1 square gets the minimum outline.

        Args: None.
        Returns: None.
        """
        self.assertEqual(self.outline_pixels(), 0)
        self.view.set_box((500, 500, 1000, 1000))
        self.assertGreater(self.outline_pixels(), 300)
        with mock.patch.object(self.view, "update") as update:
            self.view.set_box((500, 500, 1000, 1000))
        update.assert_not_called()
        self.view.set_box((500, 500, 501, 501))
        self.assertGreaterEqual(self.outline_pixels(), 4 * (qt_app.BOX_MIN_SIZE - 1))

    def test_gl_view_overlays_the_outline_only_when_there_is_one(self):
        """paintGL opens a painter for the outline and ends it.

        Args: None.
        Returns: None.
        """
        view = qt_app.GLImageView()
        self.addCleanup(view.deleteLater)
        view.context = mock.MagicMock()
        view.program, view.vao = mock.MagicMock(), mock.MagicMock()
        view.image_texture, view.lut_texture = 7, 9
        view.show_packet(make_packet(), np.zeros((256, 3), np.uint8), "k")
        view.texture_shape, view.pending = (48, 64), None
        with mock.patch.object(qt_app, "QPainter") as painter, \
                mock.patch.object(view, "paint_box") as paint_box:
            view.paintGL()
            painter.assert_not_called()
            view.set_box((1, 1, 5, 5))
            view.paintGL()
        paint_box.assert_called_once_with(painter.return_value)
        painter.return_value.end.assert_called_once()


class PixelGridTests(unittest.TestCase):
    """PixelGrid paints shaded, numbered cells, rendered offscreen."""

    def setUp(self):
        """Create a 200x200 grid.

        Args: None.
        Returns: None.
        """
        self.grid = qt_app.PixelGrid()
        self.grid.resize(200, 200)

    def tearDown(self):
        """Delete the grid.

        Args: None.
        Returns: None.
        """
        self.grid.deleteLater()

    def color_at(self, x, y):
        """Render the grid and read one pixel.

        Args: x and y are the widget position.
        Returns: The QColor there.
        """
        return self.grid.grab().toImage().pixelColor(x, y)

    def test_without_values_there_is_no_cell(self):
        """No packet, or one waiting for a baseline, draws only the message.

        Args: None.
        Returns: None.
        """
        self.assertEqual(self.grid.cell_size(), 0)
        self.grid.grab()
        self.grid.set_packet(square(view="waiting"))
        self.assertEqual(self.grid.cell_size(), 0)
        self.grid.grab()

    def test_cells_are_shaded_relative_to_the_largest_value(self):
        """The largest value is white, 0 is black, and both carry numbers.

        Args: None.
        Returns: None.
        """
        self.grid.set_packet(square())
        self.assertEqual(self.grid.cell_size(), 100)
        self.assertEqual(self.color_at(190, 190).red(), 255)  # the 7, away from its digit
        self.assertEqual(self.color_at(10, 10).red(), 0)
        with mock.patch.object(qt_app.QPainter, "drawText") as draw:
            self.grid.grab()
        self.assertEqual(sorted(call.args[2] for call in draw.call_args_list),
                         ["0", "1", "2", "7"])

    def test_all_zero_square_is_black_and_small_cells_have_no_numbers(self):
        """A still square is all black; 64×64 in 200 px is too small for text.

        Args: None.
        Returns: None.
        """
        self.grid.set_packet(square(np.zeros((2, 2), np.uint8)))
        self.assertEqual(self.color_at(150, 150).red(), 0)
        self.grid.set_packet(square(np.zeros((64, 64), np.uint8), size=64))
        with mock.patch.object(qt_app.QPainter, "drawText") as draw:
            self.grid.grab()
        draw.assert_not_called()


class PixelSquareWindowTests(WindowTestCase):
    """The Pixels window: opening, size, moving, freezing and following."""

    def open_dialog(self):
        """Open the window with the P key.

        Args: None.
        Returns: The PixelSquareDialog.
        """
        self.assertTrue(self.window.handle_key(Qt.Key_P))
        dialog = self.window.pixels_dialog
        self.addCleanup(dialog.hide)
        return dialog

    def test_button_and_key_open_one_window_that_asks_for_squares(self):
        """Pixels (P) opens the window once; closing stops the squares.

        Args: None.
        Returns: None.
        """
        buttons = [b for b in self.window.actions_bar.findChildren(QPushButton)
                   if b.text() == "Pixels (P)"]
        self.assertEqual(len(buttons), 1)
        dialog = self.open_dialog()
        self.assertTrue(dialog.isVisible())
        self.assertEqual(self.engine._square_request, (8, None))
        buttons[0].click()
        self.assertIs(self.window.pixels_dialog, dialog)
        self.assertEqual(sum("Pixel values opened" in line for line in self.log.lines), 1)
        dialog.reject()
        self.assertIsNone(self.engine._square_request)
        self.assertIsNone(self.window.view.box)
        self.assertTrue(any("Pixel values closed" in line for line in self.log.lines))

    def test_square_is_shown_and_outlined(self):
        """A new square fills the grid and summary and outlines the image.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        dialog.poll()  # nothing yet
        self.assertIsNone(dialog.shown)
        packet = square(demo=True)
        self.engine._square_packet = packet
        dialog.poll()
        self.assertIs(dialog.grid.packet, packet)
        self.assertIn("difference min 0, mean 2.50, max 7", dialog.summary.text())
        self.assertEqual(self.window.view.box, (4, 6, 6, 8))
        self.assertIn("DUMMY MODE", dialog.windowTitle())
        self.assertFalse(dialog.demo_label.isHidden())
        dialog.poll()  # the same square again changes nothing
        self.assertIs(dialog.shown, packet)

    def test_size_choice_is_sent_and_remembered(self):
        """Choosing 32×32 asks the engine for it and is saved for next time.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        dialog.size_combo.setCurrentIndex(qt_app.SQUARE_SIZES.index(32))
        self.assertEqual(self.engine._square_request, (32, None))
        self.assertEqual(FakeSettings.store[qt_app.PIXELS_SIZE_SETTING], 32)
        reopened = qt_app.PixelSquareDialog(self.window)
        self.addCleanup(reopened.deleteLater)
        self.assertEqual(reopened.size, 32)

    def test_unreadable_or_unknown_saved_size_falls_back_to_the_default(self):
        """A saved size that is not a number or not offered uses the default.

        Args: None.
        Returns: None.
        """
        for saved in ("big", 3):
            FakeSettings.store[qt_app.PIXELS_SIZE_SETTING] = saved
            dialog = qt_app.PixelSquareDialog(self.window)
            self.addCleanup(dialog.deleteLater)
            self.assertEqual(dialog.size, qt_app.DEFAULT_SQUARE_SIZE)

    def test_clicking_the_image_moves_the_square_only_while_open(self):
        """A click moves the square while the window is open, and unfreezes.

        Args: None.
        Returns: None.
        """
        self.window._image_clicked(3, 4)  # no window yet
        dialog = self.open_dialog()
        dialog.freeze_button.setChecked(True)
        self.window.view.imageClicked.emit(30, 40)
        self.assertEqual(self.engine._square_request, (8, (30, 40)))
        self.assertFalse(dialog.freeze_button.isChecked())
        dialog.hide()
        self.window._image_clicked(50, 60)
        self.assertEqual(dialog.center, (30, 40))

    def test_freeze_holds_the_shown_values(self):
        """Frozen, a newer square is not shown; the log records both states.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        first = square()
        self.engine._square_packet = first
        dialog.poll()
        dialog.freeze_button.setChecked(True)
        self.engine._square_packet = square(left=9)
        dialog.poll()
        self.assertIs(dialog.shown, first)
        dialog.freeze_button.setChecked(False)
        self.assertEqual(self.log.lines[-2:], ["Pixel values frozen.",
                                               "Pixel values live again."])

    def test_window_follows_the_displayed_camera(self):
        """When the displayed engine changes, the request moves with it.

        Args: None.
        Returns: None.
        """
        dialog = self.open_dialog()
        other = Engine(self.log)
        other.use_demo()
        self.addCleanup(other.wait, 1.0)
        dialog.freeze_button.setChecked(True)
        with mock.patch.object(type(self.window), "engine",
                               new_callable=mock.PropertyMock, return_value=other):
            dialog.poll()
        self.assertIsNone(self.engine._square_request)
        self.assertEqual(other._square_request, (8, None))
        self.assertFalse(dialog.freeze_button.isChecked())


if __name__ == "__main__":
    unittest.main()
