"""Camera-free checks that the mouse wheel never changes a slider or number box.

A wheel notch over a slider or the number box beside it must reach the panel
around it (so the Controls panel scrolls), while dragging, clicking and the
keyboard still change the value. Offscreen platform, demo engine only: the
window uses the same fakes as test_qt_app (dictionary QSettings, a fake
updater, an in-memory log) and no camera, network or dialog is touched.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QSlider

from espi_live import paths, qt_app
from espi_live.channels import demo_channels
from test_qt_app import FakeLog, FakeSettings, FakeUpdateWorker, make_packet

APP = QApplication.instance() or QApplication(sys.argv)

NOTCH_UP = 120   # one wheel notch away from the user (angleDelta.y > 0)
NOTCH_DOWN = -120


def send_wheel(widget, delta, propagate=False):
    """Deliver one wheel notch to a widget with QApplication.sendEvent.

    QApplication.sendEvent does not pass an ignored wheel event to the parent
    (a plain QLabel's ignored event never reaches its scroll area), so with
    propagate=True the event is walked up the parents here, as a window does
    for a real event, until one accepts it.

    Args: widget is the receiver; delta is the angleDelta.y (120 per notch,
        positive scrolls up or zooms in); propagate walks ignored events up.
    Returns: The QWheelEvent after delivery (check isAccepted()).
    """
    local = QPointF(widget.width() / 2, widget.height() / 2)
    event = QWheelEvent(local, QPointF(widget.mapToGlobal(local.toPoint())),
                        QPoint(0, 0), QPoint(0, delta), Qt.NoButton,
                        Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(widget, event)
    target = widget.parentWidget() if propagate else None
    while target is not None and not event.isAccepted():
        event.ignore()
        QApplication.sendEvent(target, event)
        target = target.parentWidget()
    return event


def send_key(widget, key):
    """Deliver one key press to a widget with QApplication.sendEvent.

    Args: widget is the receiver; key is a Qt.Key_* value.
    Returns: None.
    """
    QApplication.sendEvent(widget, QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier))


class WheelIgnoringWidgetTests(unittest.TestCase):
    """The wheel-ignoring subclasses, tested without a parent so nothing scrolls."""

    def test_plain_slider_changes_on_wheel(self):
        """Control case: a stock QSlider does change on a wheel notch.

        Args: None.
        Returns: None.
        """
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        send_wheel(slider, NOTCH_UP)
        self.assertGreater(slider.value(), 0)

    def test_wheel_slider_keeps_value_and_ignores_event(self):
        """A wheel notch over the subclass leaves the value alone and is not accepted.

        Args: None.
        Returns: None.
        """
        slider = qt_app.WheelIgnoringSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(40)
        for delta in (NOTCH_UP, NOTCH_DOWN):
            event = send_wheel(slider, delta)
            self.assertEqual(slider.value(), 40)
            self.assertFalse(event.isAccepted())

    def test_wheel_slider_keyboard_still_steps(self):
        """Arrow keys and Page Up/Down still change the value of the subclass.

        Args: None.
        Returns: None.
        """
        slider = qt_app.WheelIgnoringSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(40)
        send_key(slider, Qt.Key_Right)
        self.assertEqual(slider.value(), 41)
        send_key(slider, Qt.Key_Left)
        self.assertEqual(slider.value(), 40)
        send_key(slider, Qt.Key_PageUp)
        self.assertEqual(slider.value(), 40 + slider.pageStep())

    def test_wheel_spin_box_keeps_value_and_ignores_event(self):
        """A wheel notch over the number box leaves its value alone and is not accepted.

        Args: None.
        Returns: None.
        """
        spin = qt_app.WheelIgnoringSpinBox()
        spin.setRange(0, 10)
        spin.setSingleStep(0.5)
        spin.setValue(2.0)
        for delta in (NOTCH_UP, NOTCH_DOWN):
            event = send_wheel(spin, delta)
            self.assertEqual(spin.value(), 2.0)
            self.assertFalse(event.isAccepted())

    def test_spin_box_takes_focus_by_click_or_tab_only(self):
        """The number box does not take keyboard focus from a wheel (StrongFocus).

        Args: None.
        Returns: None.
        """
        spin = qt_app.WheelIgnoringSpinBox()
        self.assertEqual(spin.focusPolicy(), Qt.StrongFocus)

    def test_spin_box_keyboard_still_steps(self):
        """Up and Down arrows still change the number box's value.

        Args: None.
        Returns: None.
        """
        spin = qt_app.WheelIgnoringSpinBox()
        spin.setRange(0, 10)
        spin.setSingleStep(0.5)
        spin.setValue(2.0)
        send_key(spin, Qt.Key_Up)
        self.assertEqual(spin.value(), 2.5)
        send_key(spin, Qt.Key_Down)
        self.assertEqual(spin.value(), 2.0)


class ControlsWheelTests(unittest.TestCase):
    """The real window: the Controls panel scrolls, and no value changes under the wheel."""

    def setUp(self):
        """Build the main window over one synthetic camera, shown at a short height.

        Args: None.
        Returns: None.
        """
        FakeSettings.store = {}
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for patcher in (
                mock.patch.object(qt_app, "QSettings", FakeSettings),
                mock.patch.object(qt_app, "SIMPLE_UI", False),
                mock.patch.object(qt_app.updater, "UpdateWorker", FakeUpdateWorker),
                mock.patch.object(qt_app.updater, "unavailable_reason",
                                  return_value="disabled in tests"),
                mock.patch.object(paths, "OUTPUT_DIR", tmp.name)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.log = FakeLog()
        self.channels = demo_channels(self.log, 1)
        self.window = qt_app.MainWindow(self.channels, self.log, False, False)
        self.window.frame_timer.stop()
        self.window.status_timer.stop()
        self.window.tip_timer.stop()
        self.addCleanup(self.cleanup_window)
        self.window.resize(900, 420)
        self.window.show()
        QApplication.processEvents()
        # A frame gives the image a size, so it can zoom (zoom is fixed at 1 without one).
        table = np.zeros((256, 3), np.uint8)
        self.window.view.show_packet(make_packet(width=2000, height=1500), table, "Raw")

    def cleanup_window(self):
        """Close the window, which stops and waits for every engine.

        Args: None.
        Returns: None.
        """
        self.window.close()
        self.window.deleteLater()

    def test_wheel_over_control_scrolls_panel_not_value(self):
        """A notch over a control in the Controls panel moves the panel's scroll bar.

        Args: None.
        Returns: None.
        """
        scroll = self.window.controls_scroll
        bar = scroll.verticalScrollBar()
        self.assertGreater(bar.maximum(), 0, "the Controls panel must be scrollable here")
        bar.setValue(0)
        control = self.window.gain
        value, number = control.slider.value(), control.spin.value()

        send_wheel(control.slider, NOTCH_DOWN, propagate=True)
        self.assertGreater(bar.value(), 0)
        self.assertEqual(control.slider.value(), value)
        self.assertEqual(control.spin.value(), number)

        bar.setValue(bar.maximum())
        before = bar.value()
        send_wheel(control.spin, NOTCH_UP, propagate=True)
        self.assertLess(bar.value(), before)
        self.assertEqual(control.slider.value(), value)
        self.assertEqual(control.spin.value(), number)

    def test_wheel_over_paired_spin_box_does_not_take_focus(self):
        """Every SliderSpin's number box is a WheelIgnoringSpinBox with StrongFocus.

        Args: None.
        Returns: None.
        """
        control = self.window.exposure
        self.assertIsInstance(control.spin, qt_app.WheelIgnoringSpinBox)
        self.assertIsInstance(control.slider, qt_app.WheelIgnoringSlider)
        self.assertEqual(control.spin.focusPolicy(), Qt.StrongFocus)

    def test_zoom_slider_ignores_wheel_but_keys_still_zoom(self):
        """The toolbar Zoom slider keeps its value on a wheel notch; arrow keys still zoom.

        Args: None.
        Returns: None.
        """
        zoom = self.window.zoom_slider
        view = self.window.view
        start = zoom.value()
        zoom_before = view.zoom
        send_wheel(zoom, NOTCH_UP)
        self.assertEqual(zoom.value(), start)
        self.assertEqual(view.zoom, zoom_before)

        send_key(zoom, Qt.Key_Right)
        self.assertEqual(zoom.value(), start + zoom.singleStep())

    def test_wheel_on_image_still_zooms(self):
        """The image's own wheel zoom is unchanged by the slider change.

        Args: None.
        Returns: None.
        """
        view = self.window.view
        before = view.zoom
        send_wheel(view, NOTCH_UP)
        self.assertGreater(view.zoom, before)


if __name__ == "__main__":
    unittest.main()
