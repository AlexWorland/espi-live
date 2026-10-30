"""Camera-free checks for espi_live.camera using an in-memory fake PySpin.

Run from the project root:  PYTHONPATH=src python -m unittest discover -s tests
"""

import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from espi_live import camera
from fake_pyspin import (FakeCamera, FakeImage, FakeSpinnakerException,
                         FakeSystem, make_pyspin)
from list_log import ListLog


class FakeSpinTestCase(unittest.TestCase):
    """Base class that installs a fake PySpin for the camera module."""

    def setUp(self):
        """Patch camera.PySpin with a fake wrapping one camera.

        Args: None.
        Returns: None.
        """
        self.log = ListLog()
        self.cam = FakeCamera()
        self.system = FakeSystem([self.cam])
        self.pyspin = make_pyspin(self.system)
        patcher = mock.patch.object(camera, "PySpin", self.pyspin)
        patcher.start()
        self.addCleanup(patcher.stop)


class LoadPySpinTests(unittest.TestCase):
    """Lazy import of the SDK."""

    def test_imports_module_once_and_caches_it(self):
        """The first call imports PySpin; later calls reuse the module.

        Args: None.
        Returns: None.
        """
        fake = object()
        with mock.patch.object(camera, "PySpin", None), \
                mock.patch.object(importlib, "import_module",
                                  return_value=fake) as importer:
            self.assertIs(camera.load_pyspin(), fake)
            self.assertIs(camera.load_pyspin(), fake)
        importer.assert_called_once_with("PySpin")

    def test_raises_import_error_when_sdk_missing(self):
        """A missing SDK surfaces as ImportError.

        Args: None.
        Returns: None.
        """
        with mock.patch.object(camera, "PySpin", None), \
                mock.patch.dict(sys.modules, {"PySpin": None}):
            with self.assertRaises(ImportError):
                camera.load_pyspin()


class NodeHelperTests(FakeSpinTestCase):
    """set_enum, set_bool, set_float and friends."""

    def test_set_enum_selects_entry(self):
        """A writable enum with a readable entry is set.

        Args: None.
        Returns: None.
        """
        nodemap = self.cam.GetNodeMap()
        self.assertTrue(camera.set_enum(nodemap, "PixelFormat", "Mono8"))
        self.assertEqual(nodemap.nodes["PixelFormat"].value, 17)

    def test_set_enum_refuses_read_only_node(self):
        """A read-only enum is left alone.

        Args: None.
        Returns: None.
        """
        nodemap = self.cam.GetNodeMap()
        nodemap.nodes["PixelFormat"].writable = False
        self.assertFalse(camera.set_enum(nodemap, "PixelFormat", "Mono8"))

    def test_set_enum_refuses_missing_entry(self):
        """An enum without the requested entry is not changed.

        Args: None.
        Returns: None.
        """
        nodemap = self.cam.GetNodeMap()
        self.assertFalse(camera.set_enum(nodemap, "PixelFormat", "Mono16"))
        self.assertEqual(nodemap.nodes["PixelFormat"].value, 0.0)

    def test_set_bool_writes_value(self):
        """A writable boolean takes the requested state.

        Args: None.
        Returns: None.
        """
        nodemap = self.cam.GetNodeMap()
        self.assertTrue(camera.set_bool(nodemap, "AcquisitionFrameRateEnable", 1))
        self.assertIs(nodemap.nodes["AcquisitionFrameRateEnable"].value, True)

    def test_set_bool_refuses_unavailable_node(self):
        """A missing boolean reports failure.

        Args: None.
        Returns: None.
        """
        self.assertFalse(camera.set_bool(self.cam.GetNodeMap(), "Nope", True))

    def test_set_float_clamps_to_upper_limit(self):
        """Values above the device maximum are clipped.

        Args: None.
        Returns: None.
        """
        nodemap = self.cam.GetNodeMap()
        self.assertTrue(camera.set_float(nodemap, "Gain", 99.0))
        self.assertEqual(nodemap.nodes["Gain"].value, 48.0)

    def test_set_float_clamps_to_lower_limit(self):
        """Values below the device minimum are clipped.

        Args: None.
        Returns: None.
        """
        nodemap = self.cam.GetNodeMap()
        self.assertTrue(camera.set_float(nodemap, "ExposureTime", 1.0))
        self.assertEqual(nodemap.nodes["ExposureTime"].value, 10.0)

    def test_set_float_refuses_read_only_node(self):
        """A read-only float is not written.

        Args: None.
        Returns: None.
        """
        nodemap = self.cam.GetNodeMap()
        self.assertFalse(camera.set_float(
            nodemap, "AcquisitionResultingFrameRate", 10.0))

    def test_set_float_refuses_unreadable_node(self):
        """A writable but unreadable float cannot be clamped, so it is skipped.

        Args: None.
        Returns: None.
        """
        nodemap = self.cam.GetNodeMap()
        nodemap.nodes["Gain"].readable = False
        self.assertFalse(camera.set_float(nodemap, "Gain", 1.0))

    def test_readable_float_node_returns_node(self):
        """A readable node is returned as is.

        Args: None.
        Returns: None.
        """
        nodemap = self.cam.GetNodeMap()
        self.assertIs(camera.readable_float_node(nodemap, "Gain"),
                      nodemap.nodes["Gain"])

    def test_readable_float_node_returns_none_when_unreadable(self):
        """An unreadable node yields None.

        Args: None.
        Returns: None.
        """
        self.assertIsNone(camera.readable_float_node(
            self.cam.GetNodeMap(), "Missing"))

    def test_readable_float_node_returns_none_on_sdk_error(self):
        """A SpinnakerException while looking up the node yields None.

        Args: None.
        Returns: None.
        """
        nodemap = self.cam.GetNodeMap()
        nodemap.raising.add("Gain")
        self.assertIsNone(camera.readable_float_node(nodemap, "Gain"))


class TrySettingTests(FakeSpinTestCase):
    """Error reporting around setters."""

    def test_logs_when_setting_unavailable(self):
        """A setter that returns False is reported.

        Args: None.
        Returns: None.
        """
        camera.try_setting(self.log, self.cam.GetNodeMap(), camera.set_float,
                           "Missing", 1.0)
        self.assertEqual(self.log.lines,
                         ["Camera setting unavailable or read-only: Missing"])

    def test_logs_sdk_exception(self):
        """A SpinnakerException from the setter is logged, not raised.

        Args: None.
        Returns: None.
        """
        nodemap = self.cam.GetNodeMap()
        nodemap.nodes["Gain"].fail_set = True
        camera.try_setting(self.log, nodemap, camera.set_float, "Gain", 1.0)
        self.assertEqual(self.log.lines, ["Could not set Gain: set failed"])

    def test_success_is_logged_at_debug_only(self):
        """A successful write logs one DEBUG line and no warning.

        Args: None.
        Returns: None.
        """
        camera.try_setting(self.log, self.cam.GetNodeMap(), camera.set_float,
                           "Gain", 1.0)
        self.assertEqual(self.log.records, [("DEBUG", "Camera setting Gain = 1.0", None)])


class ConfigureCameraTests(FakeSpinTestCase):
    """configure_camera defaults."""

    def test_applies_documented_defaults(self):
        """Exposure, gain, frame rate and buffering defaults are written.

        Args: None.
        Returns: None.
        """
        nodemap = camera.configure_camera(self.cam, self.log)
        self.assertIs(nodemap, self.cam.nodemap)
        self.assertEqual(nodemap.nodes["ExposureTime"].value, 8000.0)
        self.assertEqual(nodemap.nodes["Gain"].value, 0.0)
        self.assertIs(nodemap.nodes["AcquisitionFrameRateEnable"].value, True)
        self.assertEqual(nodemap.nodes["AcquisitionFrameRate"].value, 30.0)
        self.assertEqual(self.cam.stream_nodemap.nodes[
            "StreamBufferHandlingMode"].value, 1)
        self.assertEqual(self.log.at("WARNING"), [])

    def test_logs_each_unsupported_setting(self):
        """A camera without the nodes still configures without raising.

        Args: None.
        Returns: None.
        """
        self.cam.nodemap.nodes.clear()
        camera.configure_camera(self.cam, self.log)
        self.assertEqual(len(self.log.at("WARNING")), 8)


class SessionOpenTests(FakeSpinTestCase):
    """CameraSession.open lifecycle."""

    def test_open_starts_acquisition(self):
        """A healthy camera is initialized, configured and started.

        Args: None.
        Returns: None.
        """
        session = camera.CameraSession(self.log)
        session.open()
        self.assertTrue(session.initialized)
        self.assertTrue(session.acquiring)
        self.assertEqual(self.cam.calls, ["Init", "BeginAcquisition"])
        self.assertIs(session.nodemap, self.cam.nodemap)

    def test_open_without_camera_releases_system(self):
        """No detected camera raises and releases the SDK resources.

        Args: None.
        Returns: None.
        """
        self.system.camera_list.cameras.clear()
        session = camera.CameraSession(self.log)
        with self.assertRaisesRegex(RuntimeError, "No FLIR"):
            session.open()
        self.assertTrue(self.system.released)
        self.assertTrue(self.system.camera_list.cleared)
        self.assertIsNone(session.system)

    def test_open_failure_after_init_deinitializes(self):
        """A failure while starting acquisition deinitializes the camera.

        Args: None.
        Returns: None.
        """
        self.cam.fail.add("BeginAcquisition")
        session = camera.CameraSession(self.log)
        with self.assertRaisesRegex(RuntimeError, "BeginAcquisition failed"):
            session.open()
        self.assertEqual(self.cam.calls, ["Init", "BeginAcquisition", "DeInit"])
        self.assertFalse(session.initialized)


class SessionGrabTests(FakeSpinTestCase):
    """CameraSession.grab outcomes."""

    def setUp(self):
        """Open a session on the fake camera.

        Args: None.
        Returns: None.
        """
        super().setUp()
        self.session = camera.CameraSession(self.log)
        self.session.open()

    def test_ok_frame_is_a_copy_and_buffer_released(self):
        """A complete image returns detached pixels and releases the buffer.

        Args: None.
        Returns: None.
        """
        image = FakeImage(np.full((4, 5), 9, np.uint8))
        self.cam.frame_source = lambda: image
        frame, state = self.session.grab(100)
        self.assertEqual(state, "ok")
        self.assertIsNot(frame, image.array)
        np.testing.assert_array_equal(frame, image.array)
        self.assertTrue(image.released)

    def test_incomplete_image_is_reported_and_released(self):
        """A damaged buffer returns no frame but is still released.

        Args: None.
        Returns: None.
        """
        image = FakeImage(np.zeros((4, 5), np.uint8), incomplete=True)
        self.cam.frame_source = lambda: image
        self.assertEqual(self.session.grab(100), (None, "incomplete"))
        self.assertTrue(image.released)

    def test_timeout_returns_the_exception(self):
        """A SpinnakerException from GetNextImage is returned as an error.

        Args: None.
        Returns: None.
        """
        def fail():
            """Raise like a grab timeout.

            Args: None.
            Returns: None.
            """
            raise FakeSpinnakerException("timeout")
        self.cam.frame_source = fail
        exc, state = self.session.grab(100)
        self.assertEqual(state, "error")
        self.assertEqual(str(exc), "timeout")


class SessionSettingsTests(FakeSpinTestCase):
    """set_float, limits and read_values."""

    def setUp(self):
        """Open a session on the fake camera.

        Args: None.
        Returns: None.
        """
        super().setUp()
        self.session = camera.CameraSession(self.log)
        self.session.open()

    def test_set_float_writes_exposure(self):
        """A plain setting writes only that node.

        Args: None.
        Returns: None.
        """
        self.session.set_float("ExposureTime", 1234.0)
        self.assertEqual(self.cam.nodemap.nodes["ExposureTime"].value, 1234.0)

    def test_set_float_enables_frame_rate_control(self):
        """Setting the frame rate first enables frame-rate control.

        Args: None.
        Returns: None.
        """
        enable = self.cam.nodemap.nodes["AcquisitionFrameRateEnable"]
        enable.value = False
        self.session.set_float("AcquisitionFrameRate", 12.0)
        self.assertIs(enable.value, True)
        self.assertEqual(
            self.cam.nodemap.nodes["AcquisitionFrameRate"].value, 12.0)

    def test_limits_are_clipped_to_ui_ranges(self):
        """Device limits are intersected with CAMERA_RANGES.

        Args: None.
        Returns: None.
        """
        limits = self.session.limits()
        self.assertEqual(limits["ExposureTime"], (10.0, 50000.0))
        self.assertEqual(limits["Gain"], (0.0, 30.0))
        self.assertEqual(limits["AcquisitionFrameRate"], (1.0, 60.0))

    def test_limits_omit_unreadable_and_failing_nodes(self):
        """Unreadable nodes and nodes whose limits raise are left out.

        Args: None.
        Returns: None.
        """
        self.cam.nodemap.nodes["Gain"].readable = False
        self.cam.nodemap.nodes["ExposureTime"].fail_limits = True
        self.assertEqual(list(self.session.limits()), ["AcquisitionFrameRate"])

    def test_limits_high_never_below_low(self):
        """A device maximum under the UI minimum collapses to a single point.

        Args: None.
        Returns: None.
        """
        node = self.cam.nodemap.nodes["Gain"]
        node.minimum, node.maximum = 5.0, 2.0
        self.assertEqual(self.session.limits()["Gain"], (5.0, 5.0))

    def test_read_values_reports_each_node(self):
        """Readable nodes report floats; missing ones report None.

        Args: None.
        Returns: None.
        """
        self.cam.nodemap.nodes["Gain"].readable = False
        values = self.session.read_values()
        self.assertEqual(values["exposure"], 8000.0)
        self.assertIsNone(values["gain"])
        self.assertEqual(values["requested"], 30.0)
        self.assertEqual(values["resulting"], 29.5)

    def test_read_values_tolerates_read_errors(self):
        """A GetValue failure yields None for that reading.

        Args: None.
        Returns: None.
        """
        self.cam.nodemap.nodes["ExposureTime"].fail_get = True
        self.assertIsNone(self.session.read_values()["exposure"])


class SessionCloseTests(FakeSpinTestCase):
    """close() ordering and error handling."""

    def test_close_releases_everything_in_order(self):
        """Acquisition, camera, list and system are released and cleared.

        Args: None.
        Returns: None.
        """
        session = camera.CameraSession(self.log)
        session.open()
        session.close()
        self.assertEqual(self.cam.calls,
                         ["Init", "BeginAcquisition", "EndAcquisition", "DeInit"])
        self.assertTrue(self.system.camera_list.cleared)
        self.assertTrue(self.system.released)
        self.assertIsNone(session.cam)
        self.assertIsNone(session.system)
        self.assertFalse(session.acquiring or session.initialized)

    def test_close_on_unopened_session_is_a_no_op(self):
        """Closing a session that never opened does nothing.

        Args: None.
        Returns: None.
        """
        session = camera.CameraSession(self.log)
        session.close()
        self.assertEqual(self.log.lines, [])

    def test_close_logs_every_cleanup_error_and_continues(self):
        """Each failing cleanup step is logged and later steps still run.

        Args: None.
        Returns: None.
        """
        session = camera.CameraSession(self.log)
        session.open()
        self.cam.fail.update({"EndAcquisition", "DeInit"})
        self.system.camera_list.fail_clear = True
        self.system.fail_release = True
        session.close()
        self.assertEqual(self.log.at("WARNING"), [
            "EndAcquisition error: EndAcquisition failed",
            "DeInit error: DeInit failed",
            "Camera-list cleanup error: clear failed",
            "System cleanup error: release failed"])
        self.assertIsNone(session.system)


class FormatCameraValuesTests(unittest.TestCase):
    """Status-line formatting."""

    def test_formats_available_readings(self):
        """All readings appear with units.

        Args: None.
        Returns: None.
        """
        text = camera.format_camera_values(
            {"exposure": 8000.0, "gain": 1.25, "requested": 30.0,
             "resulting": 29.46})
        self.assertEqual(
            text, "Applied: 8000 µs, 1.2 dB; FPS set 30.0, resulting 29.5")

    def test_formats_missing_readings_as_na(self):
        """None readings print n/a.
        Args: None.
        Returns: None.
        """
        text = camera.format_camera_values(
            {"exposure": None, "gain": None, "requested": None,
             "resulting": None})
        self.assertEqual(
            text, "Applied: n/a, n/a; FPS set n/a, resulting n/a")


class ReadIdentityTests(FakeSpinTestCase):
    """Model detection from the camera's identity nodes."""

    def test_reads_tl_and_device_nodes(self):
        """Vendor, model and serial come from the TL map; the rest from the device.

        Args: None.
        Returns: None.
        """
        identity = camera.read_identity(self.cam)
        self.assertEqual((identity.vendor, identity.model, identity.serial),
                         ("FLIR", "Blackfly S BFS-U3-63S4M", "21234567"))
        self.assertEqual((identity.width, identity.height), (3072, 2048))
        self.assertTrue(identity.sensor.startswith("Sony IMX178"))
        self.assertEqual(identity.firmware, "1907.0.390.0")

    def test_falls_back_to_device_map_and_width_max(self):
        """Missing TL nodes and sensor sizes use the device map and WidthMax.

        Args: None.
        Returns: None.
        """
        from fake_pyspin import FakeNode
        self.cam.tl_device_nodemap.nodes.clear()
        nodes = self.cam.nodemap.nodes
        nodes["DeviceModelName"] = FakeNode("Blackfly S BFS-U3-200S6M")
        del nodes["SensorWidth"], nodes["SensorHeight"]
        nodes["WidthMax"] = FakeNode(5472)
        nodes["HeightMax"] = FakeNode(3648)
        identity = camera.read_identity(self.cam)
        self.assertEqual(identity.model, "Blackfly S BFS-U3-200S6M")
        self.assertEqual(identity.vendor, "")
        self.assertEqual((identity.width, identity.height), (5472, 3648))

    def test_node_errors_leave_fields_empty(self):
        """Nodes that raise SpinnakerException read as empty values.

        Args: None.
        Returns: None.
        """
        self.cam.tl_device_nodemap.raising.add("DeviceModelName")
        self.cam.nodemap.raising.update({"DeviceModelName", "SensorWidth",
                                         "WidthMax"})
        identity = camera.read_identity(self.cam)
        self.assertEqual((identity.model, identity.width), ("", 0))
        self.assertEqual(identity.serial, "21234567")

    def test_open_records_identity(self):
        """CameraSession.open() stores the detected model.

        Args: None.
        Returns: None.
        """
        session = camera.CameraSession(self.log)
        session.open()
        self.addCleanup(session.close)
        self.assertEqual(session.identity.short_name(), "BFS-U3-63S4M")


if __name__ == "__main__":
    unittest.main()


class SharedSystemTests(FakeSpinTestCase):
    """acquire_system/release_system and several sessions at once."""

    def test_two_sessions_share_one_system_until_the_last_closes(self):
        """The SDK is released only after every camera has been closed.

        Args: None.
        Returns: None.
        """
        second = FakeCamera(serial="87654321")
        self.system.camera_list.cameras.append(second)
        first_session = camera.CameraSession(self.log, "21234567")
        second_session = camera.CameraSession(self.log, "87654321")
        first_session.open()
        second_session.open()
        self.assertEqual(self.system.instances, 1)
        self.assertEqual(second.calls, ["Init", "BeginAcquisition"])
        first_session.close()
        self.assertEqual(self.system.releases, 0)
        second_session.close()
        self.assertEqual(self.system.releases, 1)

    def test_extra_release_is_ignored(self):
        """Releasing without a reference neither fails nor releases twice.

        Args: None.
        Returns: None.
        """
        camera.acquire_system()
        camera.release_system()
        camera.release_system()
        self.assertEqual(self.system.releases, 1)

    def test_a_new_pyspin_module_gets_a_fresh_system(self):
        """A reference held for another PySpin module is not reused.

        Args: None.
        Returns: None.
        """
        camera.acquire_system()
        other = FakeSystem([])
        with mock.patch.object(camera, "PySpin", make_pyspin(other)):
            self.assertIs(camera.acquire_system(), other)
            camera.release_system()
        self.assertEqual(other.releases, 1)


class CameraSelectionTests(FakeSpinTestCase):
    """Choosing a camera by serial number and naming it."""

    def test_open_by_serial_picks_that_camera_and_records_its_info(self):
        """The camera whose serial matches is opened, not the first one.

        Args: None.
        Returns: None.
        """
        second = FakeCamera(serial="87654321", model="Blackfly S")
        self.system.camera_list.cameras.append(second)
        session = camera.CameraSession(self.log, "87654321")
        session.open()
        self.addCleanup(session.close)
        self.assertEqual(self.cam.calls, [])
        self.assertEqual(session.info, camera.CameraInfo("87654321", "Blackfly S"))

    def test_open_with_a_missing_serial_raises_and_releases(self):
        """A camera that was unplugged is reported and the SDK released.

        Args: None.
        Returns: None.
        """
        session = camera.CameraSession(self.log, "99999999")
        with self.assertRaisesRegex(RuntimeError, "99999999 is no longer connected"):
            session.open()
        self.assertEqual(self.system.releases, 1)

    def test_labels_include_the_model_when_known(self):
        """CameraInfo.label is "model (serial)", or just the serial.

        Args: None.
        Returns: None.
        """
        self.assertEqual(camera.CameraInfo("1", "BFS").label, "BFS (1)")
        self.assertEqual(camera.CameraInfo("1").label, "1")

    def test_unreadable_serial_falls_back_to_the_list_position(self):
        """Without a readable serial the camera is named by its position.

        Args: None.
        Returns: None.
        """
        self.cam.tl_device_nodemap.nodes["DeviceSerialNumber"].readable = False
        self.cam.tl_device_nodemap.raising.add("DeviceModelName")
        self.assertEqual(camera.camera_info(self.cam, 1), camera.CameraInfo("camera 2", ""))


class ListCamerasTests(FakeSpinTestCase):
    """list_cameras enumerates without opening."""

    def test_lists_every_camera_and_releases_the_sdk(self):
        """Each camera is identified; none is initialized; everything is released.

        Args: None.
        Returns: None.
        """
        self.system.camera_list.cameras.append(FakeCamera(serial="2", model="M"))
        cameras = camera.list_cameras(self.log)
        self.assertEqual(cameras, [camera.CameraInfo("21234567", "Blackfly S BFS-U3-63S4M"),
                                   camera.CameraInfo("2", "M")])
        self.assertEqual(self.cam.calls, [])
        self.assertTrue(self.system.camera_list.cleared)
        self.assertEqual(self.system.releases, 1)

    def test_release_errors_are_logged(self):
        """A failing ReleaseInstance is logged and the list still returned.

        Args: None.
        Returns: None.
        """
        self.system.fail_release = True
        self.assertEqual(len(camera.list_cameras(self.log)), 1)
        self.assertEqual(self.log.lines, ["System cleanup error: release failed"])

    def test_enumeration_errors_propagate_after_cleanup(self):
        """If GetCameras fails the error is raised and the SDK released.

        Args: None.
        Returns: None.
        """
        self.system.GetCameras = mock.Mock(side_effect=FakeSpinnakerException("bus"))
        with self.assertRaises(FakeSpinnakerException):
            camera.list_cameras(self.log)
        self.assertEqual(self.system.releases, 1)


class LibraryVersionTests(unittest.TestCase):
    """Reading the SDK version for the log header and reports."""

    def test_the_version_is_formatted_from_the_system(self):
        """major.minor.type.build is read from the System.

        Args: None.
        Returns: None.
        """
        version = SimpleNamespace(major=4, minor=2, type=0, build=88)
        system = SimpleNamespace(GetLibraryVersion=lambda: version)
        self.assertEqual(camera.read_library_version(system), "4.2.0.88")
