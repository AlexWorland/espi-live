"""Camera-free checks for the Qt front end (offscreen platform, demo engine).

Run from the project root:
    PYTHONPATH=src QT_QPA_PLATFORM=offscreen python -m unittest tests.test_qt_app

No camera, network, real dialog or real settings store is touched: the engine
uses its demo source, the updater worker is replaced by a fake, QSettings is
replaced by a dictionary and every QMessageBox call is patched.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import queue
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QSurfaceFormat
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit

from espi_live import feedback, paths, qt_app, updater
from espi_live.camera import CameraInfo
from espi_live.camera_models import CameraIdentity
from espi_live.channels import Channel, demo_channels
from espi_live.engine import Engine, FramePacket

APP = QApplication.instance() or QApplication(sys.argv)


class FakeLog:
    """In-memory stand-in for AppLog."""

    def __init__(self):
        """Create an empty log.

        Args: None.
        Returns: None.
        """
        self.lines = []
        self.pending = []

    def write(self, message):
        """Record one message for assertions and for drain().

        Args: message is the text to keep.
        Returns: None.
        """
        self.lines.append(message)
        self.pending.append(message + "\n")

    def drain(self):
        """Return and clear the text written since the last call.

        Args: None.
        Returns: The pending text, possibly empty.
        """
        text, self.pending = "".join(self.pending), []
        return text

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
        self.assertEqual(self.window.exposure.toolTip(), "Simulated in dummy mode")

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
        self.assertEqual(self.window.exposure.toolTip(), "")

    def test_first_run_uses_default_window_size(self):
        """Without saved geometry the window opens at 1400x900.

        Args: None.
        Returns: None.
        """
        self.assertEqual((self.window.width(), self.window.height()), (1400, 900))

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
        self.assertEqual(window.updates.checks, 1)
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
        self.assertEqual((s.average, s.ref_refresh_s, s.smoothing), (5, 2.5, 3))
        self.assertEqual((s.clip_percent, s.gamma), (1.5, 1.7))

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

    def test_hover_signal_is_remembered(self):
        """The view's hover signal is stored for the pixel readout.

        Args: None.
        Returns: None.
        """
        self.window.view.hovered.emit(3, 4)
        self.assertEqual(self.window.hover, (3, 4))


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
        items = (window.update_slot, window.feature_button, window.source_badge,
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
        self.engine.events.put(("fatal", "kaboom"))
        with mock.patch.object(self.window, "close") as close:
            self.window._handle_events()
        self.dialogs.critical.assert_called_once()
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
        second.engine.events.put(("fatal", "kaboom"))
        with mock.patch.object(window, "close") as close:
            window._handle_events()
        close.assert_not_called()
        self.assertIn("Synthetic camera (DEMO2): kaboom",
                      self.dialogs.critical.call_args.args[2])
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
        """A newer build is remembered and shown on the status-bar button.

        Args: None.
        Returns: None.
        """
        info = make_info()
        self.post("checked", info)
        self.assertIs(self.window.update_info, info)
        self.assertFalse(self.window.update_button.isHidden())
        self.assertIn("build 99", self.window.update_button.text())
        self.dialogs.question.assert_not_called()

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
        self.assertIn("build 99", self.window.update_button.text())
        self.dialogs.warning.assert_called_once()

    def test_progress_is_shown_while_downloading(self):
        """Download progress appears as a percentage on the button.

        Args: None.
        Returns: None.
        """
        self.window.update_info = make_info()
        self.window.updates.busy = True
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
        self.addCleanup(window.close)
        self.assertEqual(window.height(), 700)  # not the 1400x900 default

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
        for patcher in (
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
        self.assertEqual(self.main_window.call_args.args[2:], (True, False))
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
                                      enrich=True, submitter="Sam", source="camera")

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
        """A previous request refills the form; Send is offered with a token.

        Args: None.
        Returns: None.
        """
        dialog = qt_app.FeatureRequestDialog(self.window, self.REQUEST)
        self.assertEqual(dialog.request("camera"), self.REQUEST)
        self.assertEqual(dialog.findChild(qt_app.QDialogButtonBox)
                         .button(qt_app.QDialogButtonBox.Ok).text(), "Send")

    def test_dialog_defaults_come_from_settings_without_a_token(self):
        """A fresh form uses the saved name and choice and offers the browser.

        Args: None.
        Returns: None.
        """
        FakeSettings.store.update({"feedback/name": "Kim", "feedback/enrich": False})
        with mock.patch.object(qt_app.feedback, "feedback_token", return_value=""):
            dialog = qt_app.FeatureRequestDialog(self.window)
        self.assertEqual(dialog.submitter.text(), "Kim")
        self.assertFalse(dialog.enrich.isChecked())
        self.assertEqual(dialog.findChild(qt_app.QDialogButtonBox)
                         .button(qt_app.QDialogButtonBox.Ok).text(), "Open in browser…")

    def test_accept_refuses_an_incomplete_request(self):
        """A blank description warns and keeps the dialog open.

        Args: None.
        Returns: None.
        """
        dialog = qt_app.FeatureRequestDialog(self.window)
        dialog.title.setText("Only a title")
        dialog.accept()
        self.dialogs.warning.assert_called_once()
        self.assertIn("describe", self.dialogs.warning.call_args.args[2])
        self.assertEqual(dialog.result(), 0)

    def test_accept_remembers_name_and_choice(self):
        """A complete request closes the dialog and saves name and enrichment.

        Args: None.
        Returns: None.
        """
        dialog = qt_app.FeatureRequestDialog(self.window, self.REQUEST)
        dialog.accept()
        self.assertEqual(dialog.result(), qt_app.QDialog.Accepted)
        self.assertEqual(FakeSettings.store["feedback/name"], "Sam")
        self.assertTrue(FakeSettings.store["feedback/enrich"])

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

    def test_without_token_the_browser_opens(self):
        """Copies without a token open the prefilled GitHub page instead.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(qt_app.feedback, "feedback_token", return_value=""), \
                mock.patch.object(qt_app.QDesktopServices, "openUrl",
                                  return_value=True) as open_url:
            self.run_request(qt_app.QDialog.Accepted)
        self.window.feedback.submit.assert_not_called()
        self.assertIn("/issues/new?", open_url.call_args.args[0].toString())
        self.dialogs.warning.assert_not_called()

    def test_browser_failure_warns(self):
        """A missing browser is reported.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(qt_app.QDesktopServices, "openUrl", return_value=False):
            self.window._open_request_in_browser(self.REQUEST)
        self.dialogs.warning.assert_called_once()

    def test_sent_request_links_the_issue(self):
        """A filed request restores the button, logs and links the issue.

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
        text = self.dialogs.call_args.args[2]
        self.assertIn(url, text)
        self.assertIn("Copilot", text)
        self.dialogs.return_value.exec.assert_called_once()

    def failed_choice(self, choice):
        """Report a failed send and click one of the offered buttons.

        Args: choice is "retry", "browser" or "cancel".
        Returns: (request_feature mock, open_request_in_browser mock).
        """
        box = self.dialogs.return_value
        buttons = {"retry": object(), "browser": object(), "cancel": object()}
        box.addButton.side_effect = [buttons["retry"], buttons["browser"], buttons["cancel"]]
        box.clickedButton.return_value = buttons[choice]
        self.window.feedback.events.put(("failed", (self.REQUEST, "offline")))
        with mock.patch.object(self.window, "_request_feature") as retry, \
                mock.patch.object(self.window, "_open_request_in_browser") as browser:
            self.window._handle_feedback_events()
        self.assertIn("Feature request failed: offline", self.log.lines)
        return retry, browser

    def test_failed_request_can_be_retried(self):
        """Edit and retry reopens the dialog with the same request.

        Args: None.
        Returns: None.
        """
        retry, browser = self.failed_choice("retry")
        retry.assert_called_once_with(self.REQUEST)
        browser.assert_not_called()

    def test_failed_request_can_open_in_browser(self):
        """Open in browser keeps the text without retyping.

        Args: None.
        Returns: None.
        """
        retry, browser = self.failed_choice("browser")
        browser.assert_called_once_with(self.REQUEST)
        retry.assert_not_called()

    def test_failed_request_can_be_dismissed(self):
        """Cancel does nothing more.

        Args: None.
        Returns: None.
        """
        retry, browser = self.failed_choice("cancel")
        retry.assert_not_called()
        browser.assert_not_called()


if __name__ == "__main__":
    unittest.main()
