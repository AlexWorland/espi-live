"""Tests for the iPhone link protocol and IPhoneSession (no phone, no network)."""

import time
import unittest
from unittest import mock

import numpy as np

from espi_live import iphone, usbmux
from espi_live.camera import CameraInfo
from fake_iphone import (FakeSocket, commands, frame_bytes, hello, lens_hello,
                         lens_socket, phone_limits, phone_settings, phone_socket,
                         status)
from list_log import ListLog

UDID = "00008110-001A2B3C4D5E801E"


def wait_until(predicate, timeout=5.0):
    """Poll a condition until it is true or a timeout passes.

    Args: predicate is a zero-argument callable; timeout is in seconds.
    Returns: The final value of predicate().
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def phone_info(lens=""):
    """Describe the test phone as list_iphones would.

    Args: lens is the back camera's id, empty for a phone that has not said.
    Returns: A CameraInfo of kind "iphone".
    """
    return CameraInfo("iphone-4d5e801e", "iPhone (USB)", kind="iphone", path=UDID,
                      lens=lens)


class ProtocolTests(unittest.TestCase):
    """Message framing, frames and JSON."""

    def test_frame_round_trip(self):
        """A frame decodes to the same pixels and metadata, writable.

        Args: None.
        Returns: None.
        """
        image = np.arange(12, dtype=np.uint8).reshape(3, 4)
        data = iphone.encode_frame(image, sequence=5, timestamp_ns=9, exposure_us=100.0,
                                   iso=50.0, lens_position=0.25)
        kind, payload = iphone.read_message(FakeSocket(data))
        meta, decoded = iphone.decode_frame(payload)
        self.assertEqual(kind, iphone.KIND_FRAME)
        np.testing.assert_array_equal(decoded, image)
        self.assertTrue(decoded.flags.writeable)
        self.assertEqual((meta.width, meta.height, meta.sequence, meta.timestamp_ns),
                         (4, 3, 5, 9))
        self.assertEqual((meta.exposure_us, meta.iso, meta.lens_position), (100.0, 50.0, 0.25))

    def test_padded_rows_are_dropped(self):
        """A stride wider than the image is cut back to contiguous rows.

        Args: None.
        Returns: None.
        """
        meta = iphone.FRAME_META.pack(2, 2, 3, iphone.PIXEL_MONO8, 0, 0, 0, 0, 0, 0)
        _, image = iphone.decode_frame(bytearray(meta + bytes([1, 2, 9, 3, 4, 9])))
        np.testing.assert_array_equal(image, [[1, 2], [3, 4]])
        self.assertTrue(image.flags.c_contiguous)

    def test_bad_frames(self):
        """Truncated, non-Mono8 and wrongly sized frames are LinkErrors.

        Args: None.
        Returns: None.
        """
        wrong_format = iphone.FRAME_META.pack(1, 1, 1, 7, 0, 0, 0, 0, 0, 0) + b"x"
        wrong_size = iphone.FRAME_META.pack(2, 2, 2, 1, 0, 0, 0, 0, 0, 0) + b"xyz"
        narrow = iphone.FRAME_META.pack(2, 1, 1, 1, 0, 0, 0, 0, 0, 0) + b"xy"
        for payload in (b"short", wrong_format, wrong_size, narrow):
            with self.subTest(payload=payload[-3:]):
                with self.assertRaises(iphone.LinkError):
                    iphone.decode_frame(bytearray(payload))

    def test_bad_headers(self):
        """Foreign data, another version and oversized messages are refused.

        Args: None.
        Returns: None.
        """
        cases = (b"HTTP" + bytes(12),
                 iphone.HEADER.pack(iphone.MAGIC, 2, 2, 0, 0),
                 iphone.HEADER.pack(iphone.MAGIC, 1, 2, iphone.MAX_JSON + 1, 0),
                 iphone.HEADER.pack(iphone.MAGIC, 1, 3, iphone.MAX_FRAME + 1, 0))
        for data in cases:
            with self.subTest(data=data[:8]):
                with self.assertRaises(iphone.LinkError):
                    iphone.read_message(FakeSocket(data))

    def test_early_close(self):
        """A link that closes mid-message raises ConnectionError.

        Args: None.
        Returns: None.
        """
        sock = FakeSocket(iphone.HEADER.pack(iphone.MAGIC, 1, 2, 10, 0) + b"abc")
        sock.finish()
        with self.assertRaises(ConnectionError):
            iphone.read_message(sock)

    def test_json(self):
        """JSON messages round-trip; bad JSON and non-objects are LinkErrors.

        Args: None.
        Returns: None.
        """
        kind, payload = iphone.read_message(FakeSocket(
            iphone.encode_json(iphone.KIND_STATUS, {"a": 1})))
        self.assertEqual((kind, iphone.decode_json(payload)), (iphone.KIND_STATUS, {"a": 1}))
        for bad in (b"\xff", b"{", b"[1]"):
            with self.assertRaises(iphone.LinkError):
                iphone.decode_json(bad)

    def test_gain_iso_and_ranges(self):
        """Gain maps to ISO in dB above the lowest ISO; ranges are clipped.

        Args: None.
        Returns: None.
        """
        self.assertAlmostEqual(iphone.gain_to_iso(20.0, 32.0), 320.0)
        self.assertAlmostEqual(iphone.iso_to_gain(320.0, 32.0), 20.0)
        self.assertIsNone(iphone.iso_to_gain(0, 32.0))
        self.assertIsNone(iphone.iso_to_gain(64.0, 0))
        self.assertEqual(iphone.clip_range([14, 1e6], "ExposureTime"), (14.0, 50000.0))
        self.assertIsNone(iphone.clip_range(None, "Gain"))
        self.assertIsNone(iphone.clip_range([5, 1], "Gain"))


class IdentityAndListingTests(unittest.TestCase):
    """Phone identity, listing and the USB connector."""

    def setUp(self):
        """Forget which cameras any phone reported earlier.

        Args: None.
        Returns: None.
        """
        iphone._lens_cache.clear()
        self.addCleanup(iphone._lens_cache.clear)

    def test_identity_text(self):
        """Before HELLO there is nothing to say; afterwards model, size and a caution.

        Args: None.
        Returns: None.
        """
        blank = iphone.IPhoneIdentity()
        self.assertEqual((blank.short_name(), blank.description(), blank.warnings()),
                         ("iPhone", [], []))
        known = iphone.IPhoneIdentity(serial=UDID, device_model="iPhone16,1",
                                      system_version="27.1", app_version="0.1 (5)",
                                      camera="Back", width=1920, height=1080)
        self.assertEqual(known.short_name(), "iPhone iPhone16,1")
        self.assertIn("1920 × 1080", " ".join(known.description()))
        self.assertIn(UDID, known.description()[-1])
        self.assertIn("rolling shutter", known.warnings()[0])
        minimal = iphone.IPhoneIdentity(device_model="iPhone16,1")
        self.assertEqual(len(minimal.description()), 1)

    def test_list_iphones(self):
        """Each USB device becomes an iphone CameraInfo; no service means none.

        Args: None.
        Returns: None.
        """
        log = ListLog()
        device = usbmux.MuxDevice(3, UDID)
        with mock.patch.object(usbmux, "list_devices", return_value=[device]), \
                mock.patch.object(iphone, "probe_lenses", return_value=None):
            infos = iphone.list_iphones(log)
        self.assertEqual(infos, [phone_info()])
        with mock.patch.object(usbmux, "list_devices",
                               side_effect=usbmux.UsbmuxError("no service")):
            self.assertEqual(iphone.list_iphones(log), [])
        self.assertIn("iPhones are not listed: no service", log.lines)

    def test_parse_lenses(self):
        """HELLO's cameras are read in order; anything unreadable is skipped.

        Args: None.
        Returns: None.
        """
        found = iphone.parse_lenses(lens_hello())
        self.assertEqual([lens.id for lens in found], ["ultra_wide", "wide", "telephoto"])
        self.assertEqual(found[0], iphone.Lens("ultra_wide", "Ultra Wide 0.5×"))
        self.assertEqual(iphone.parse_lenses(hello()), ())
        self.assertEqual(iphone.parse_lenses(hello(lenses="wide")), ())
        messy = hello(lenses=["wide", {"name": "no id"}, {"id": "wide"}])
        self.assertEqual(iphone.parse_lenses(messy), (iphone.Lens("wide", "wide"),))

    def test_lens_infos(self):
        """Each back camera is listed, the main one first under the old serial.

        Args: None.
        Returns: None.
        """
        found = iphone.parse_lenses(lens_hello())
        infos = iphone.lens_infos(UDID, found)
        self.assertEqual([info.lens for info in infos], ["wide", "ultra_wide", "telephoto"])
        self.assertEqual([info.serial for info in infos],
                         ["iphone-4d5e801e", "iphone-4d5e801e-ultra-wide",
                          "iphone-4d5e801e-telephoto"])
        self.assertEqual(infos[1].label,
                         "iPhone Ultra Wide 0.5× (USB) (iphone-4d5e801e-ultra-wide)")
        self.assertEqual({info.path for info in infos}, {UDID})
        self.assertEqual({info.kind for info in infos}, {"iphone"})
        only = iphone.lens_infos(UDID, found[1:2])
        self.assertEqual(only, [phone_info("wide")])
        self.assertEqual(iphone.lens_infos(UDID, ()), [phone_info()])

    def test_main_lenses_keep_one_camera_per_phone(self):
        """Only the first-listed camera of each phone is kept.

        Args: None.
        Returns: None.
        """
        mine = iphone.lens_infos(UDID, iphone.parse_lenses(lens_hello()))
        other = iphone.lens_infos("OTHER-PHONE-0001", iphone.parse_lenses(lens_hello()))
        kept = iphone.main_lenses(mine + other)
        self.assertEqual([(info.path, info.lens) for info in kept],
                         [(UDID, "wide"), ("OTHER-PHONE-0001", "wide")])

    def test_probe_lenses(self):
        """A running app is asked once and hung up on; anything else gives None.

        Args: None.
        Returns: None.
        """
        device = usbmux.MuxDevice(3, UDID)
        sock = phone_socket(lens_hello())
        with mock.patch.object(usbmux, "connect", return_value=sock) as connect:
            found = iphone.probe_lenses(device)
        connect.assert_called_once_with(3, iphone.PORT, timeout=iphone.PROBE_TIMEOUT_S)
        self.assertEqual(len(found), 3)
        self.assertTrue(sock.closed)
        self.assertEqual(bytes(sock.sent), b"")  # it only listens
        failures = [usbmux.UsbmuxError("refused"), OSError("timed out")]
        for failure in failures:
            with self.subTest(failure=failure), \
                    mock.patch.object(usbmux, "connect", side_effect=failure):
                self.assertIsNone(iphone.probe_lenses(device))
        for link in (FakeSocket(iphone.encode_json(iphone.KIND_STATUS, status())),
                     FakeSocket(b"not the app's greeting")):
            with self.subTest(link=bytes(link.incoming[:6])), \
                    mock.patch.object(usbmux, "connect", return_value=link):
                self.assertIsNone(iphone.probe_lenses(device))
            self.assertTrue(link.closed)

    def test_list_iphones_lists_every_lens_and_asks_a_phone_once(self):
        """A phone is asked for its cameras the first time; the answer is kept.

        Args: None.
        Returns: None.
        """
        log = ListLog()
        device = usbmux.MuxDevice(3, UDID)
        answer = iphone.parse_lenses(lens_hello())
        with mock.patch.object(usbmux, "list_devices", return_value=[device]), \
                mock.patch.object(iphone, "probe_lenses", return_value=answer) as probe:
            first = iphone.list_iphones(log)
            second = iphone.list_iphones(log)
        self.assertEqual(first, second)
        self.assertEqual([info.lens for info in first], ["wide", "ultra_wide", "telephoto"])
        probe.assert_called_once_with(device)

    def test_list_iphones_asks_again_after_no_answer(self):
        """A phone whose app was closed is listed once, and asked again next time.

        Args: None.
        Returns: None.
        """
        device = usbmux.MuxDevice(3, UDID)
        with mock.patch.object(usbmux, "list_devices", return_value=[device]), \
                mock.patch.object(iphone, "probe_lenses",
                                  side_effect=[None, iphone.parse_lenses(lens_hello())]):
            self.assertEqual(iphone.list_iphones(ListLog()), [phone_info()])
            self.assertEqual(len(iphone.list_iphones(ListLog())), 3)

    def test_usb_connector(self):
        """The phone is found by UDID, since its device id changes on replug.

        Args: None.
        Returns: None.
        """
        sock = FakeSocket()
        devices = [usbmux.MuxDevice(2, "OTHER"), usbmux.MuxDevice(7, UDID)]
        with mock.patch.object(usbmux, "list_devices", return_value=devices), \
                mock.patch.object(usbmux, "connect", return_value=sock) as connect:
            self.assertIs(iphone.usb_connector(phone_info()), sock)
        connect.assert_called_once_with(7, iphone.PORT)
        with mock.patch.object(usbmux, "list_devices", return_value=devices[:1]):
            with self.assertRaisesRegex(RuntimeError, "not plugged in"):
                iphone.usb_connector(phone_info())

    def test_shut_down_ignores_errors(self):
        """Errors while closing a dead socket are ignored.

        Args: None.
        Returns: None.
        """
        sock = FakeSocket()
        sock.shutdown_error = OSError("not connected")
        sock.close_error = OSError("bad descriptor")
        iphone.shut_down(sock)
        self.assertTrue(sock.closed)


class SessionCase(unittest.TestCase):
    """Base class: a session whose connector hands out prepared sockets."""

    def setUp(self):
        """Create the log, the socket queue and the session.

        Args: None.
        Returns: None.
        """
        self.log = ListLog()
        self.sockets = [phone_socket()]
        self.session = iphone.IPhoneSession(self.log, phone_info(), 8000.0, 6.0, 25.0,
                                            connector=self.connect)
        self.addCleanup(self.session.close)
        iphone._lens_cache.clear()
        self.addCleanup(iphone._lens_cache.clear)

    def connect(self, info):
        """Hand out the next socket, or raise a queued exception.

        Args: info is the phone's CameraInfo.
        Returns: A FakeSocket.
        """
        item = self.sockets.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def open(self):
        """Open the session and return its link socket.

        Args: None.
        Returns: The FakeSocket in use.
        """
        sock = self.sockets[0]
        self.session.open()
        return sock


class SessionOpenTests(SessionCase):
    """Opening and closing."""

    def test_open_sends_starting_settings(self):
        """After HELLO the viewer's exposure, ISO (from gain) and rate are sent.

        Args: None.
        Returns: None.
        """
        sock = self.open()
        self.assertIsNone(sock.timeout)
        command = commands(sock)[0]
        self.assertEqual((command["exposure_us"], command["fps"]), (8000.0, 25.0))
        self.assertAlmostEqual(command["iso"], iphone.gain_to_iso(6.0, 32.0))
        self.assertEqual(self.session.identity.device_model, "iPhone16,1")
        self.assertEqual(self.session.identity.formats[-1], (1920, 1080, 30.0))
        self.assertEqual((self.session.identity.width, self.session.identity.height),
                         (1920, 1080))
        self.assertTrue(any("iPhone link open" in line for line in self.log.lines))

    def test_open_asks_for_the_camera_and_waits_for_it(self):
        """A back camera is requested in the starting command and described by the
        HELLO that answers it, before open() returns.

        Args: None.
        Returns: None.
        """
        self.session = iphone.IPhoneSession(
            self.log, phone_info("ultra_wide"), 8000.0, 6.0, 25.0, connector=self.connect)
        self.addCleanup(self.session.close)
        self.sockets = [lens_socket("ultra_wide")]
        sock = self.open()
        self.assertEqual(commands(sock)[0]["lens"], "ultra_wide")
        self.assertEqual(self.session.hello["lens"], "ultra_wide")
        self.assertEqual(self.session.identity.camera, "Back Ultra Wide 0.5×")
        self.assertEqual([lens.id for lens in iphone._lens_cache[UDID]],
                         ["ultra_wide", "wide", "telephoto"])

    def test_open_with_the_camera_already_in_use_does_not_wait(self):
        """When the app is already on the camera asked for, its first HELLO is enough.

        Args: None.
        Returns: None.
        """
        self.session = iphone.IPhoneSession(
            self.log, phone_info("wide"), connector=self.connect)
        self.addCleanup(self.session.close)
        self.sockets = [lens_socket("wide")]
        started = time.monotonic()
        sock = self.open()
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(commands(sock)[0]["lens"], "wide")

    def test_open_without_a_lens_does_not_ask_for_one(self):
        """A phone that has not said which cameras it has is left on its own.

        Args: None.
        Returns: None.
        """
        sock = self.open()
        self.assertNotIn("lens", commands(sock)[0])

    def test_open_fails_when_the_phone_never_switches(self):
        """Frames from another camera are never shown under this one's name.

        Args: None.
        Returns: None.
        """
        self.session = iphone.IPhoneSession(
            self.log, phone_info("telephoto"), connector=self.connect)
        self.addCleanup(self.session.close)
        sock = phone_socket(lens_hello("wide", lenses=[{"id": "wide", "name": "Wide 1×"}]))
        self.sockets = [sock]
        with mock.patch.object(iphone, "LENS_TIMEOUT_S", 0.05), \
                self.assertRaisesRegex(RuntimeError, "did not switch to its telephoto"):
            self.session.open()
        self.assertTrue(sock.closed)

    def test_open_without_iso_limits_skips_iso(self):
        """Without the phone's ISO range the gain cannot be converted yet.

        Args: None.
        Returns: None.
        """
        self.sockets = [phone_socket(hello(limits={}))]
        sock = self.open()
        self.assertNotIn("iso", commands(sock)[0])

    def test_open_failures_close_the_socket(self):
        """No HELLO, another version or a refused connection raise RuntimeError.

        Args: None.
        Returns: None.
        """
        wrong_kind = FakeSocket(iphone.encode_json(iphone.KIND_STATUS, status()))
        old = phone_socket(hello(protocol_version=99))
        for item in (wrong_kind, old, usbmux.UsbmuxError("refused")):
            self.sockets = [item]
            with self.subTest(item=item), \
                    self.assertRaisesRegex(RuntimeError, "Open the ESPI Camera app"):
                self.session.open()
            if isinstance(item, FakeSocket):
                self.assertTrue(item.closed)

    def test_close_is_repeatable(self):
        """close() stops the reader and can be called again.

        Args: None.
        Returns: None.
        """
        sock = self.open()
        self.session.close()
        self.session.close()
        self.assertTrue(sock.closed)
        self.assertIsNone(self.session._reader)


class SessionFrameTests(SessionCase):
    """Frames, timeouts and a lost link."""

    def test_grab_returns_newest_frame(self):
        """grab() returns each new frame once, then times out.

        Args: None.
        Returns: None.
        """
        sock = self.open()
        sock.feed(frame_bytes(8, 4, value=1) + frame_bytes(8, 4, value=2))
        self.assertTrue(wait_until(lambda: self.session._frame_seq == 2))
        frame, state = self.session.grab(500)
        self.assertEqual((state, int(frame[0, 0]), frame.shape), ("ok", 2, (4, 8)))
        error, state = self.session.grab(20)
        self.assertEqual(state, "error")
        self.assertIsInstance(error, TimeoutError)

    def test_messages_update_state_and_log(self):
        """STATUS, ERROR, a second HELLO and unknown kinds are handled.

        Args: None.
        Returns: None.
        """
        sock = self.open()
        sock.feed(iphone.encode_json(iphone.KIND_STATUS, status(
            thermal="serious", interrupted="the app is in the background",
            error="Send failed")))
        sock.feed(iphone.encode_json(iphone.KIND_STATUS, status(
            thermal="serious", interrupted="the app is in the background",
            error="Send failed")))
        sock.feed(iphone.encode_json(iphone.KIND_ERROR, {"message": "Unreadable command"}))
        sock.feed(iphone.encode_json(iphone.KIND_ERROR, {}))
        sock.feed(iphone.encode_json(iphone.KIND_HELLO, hello(device_model="iPhone17,2")))
        sock.feed(iphone.encode_message(99, b"future"))
        sock.feed(frame_bytes())
        self.assertTrue(wait_until(lambda: self.session._frame_seq == 1))
        warnings = [line for level, line, _ in self.log.records if level == "WARNING"]
        self.assertEqual(sum("getting hot" in line for line in warnings), 1)
        self.assertEqual(sum("interrupted" in line for line in warnings), 1)
        self.assertEqual(sum("Send failed" in line for line in warnings), 1)
        self.assertIn("iPhone reports: Unreadable command", warnings)
        self.assertIn("iPhone reports: unknown error", warnings)
        self.assertEqual(self.session.identity.device_model, "iPhone17,2")

    def test_lost_link_reconnects(self):
        """A closed link is reported and re-opened from grab().

        Args: None.
        Returns: None.
        """
        first = self.open()
        second = phone_socket()
        self.sockets = [second]
        first.finish()
        self.assertTrue(wait_until(lambda: self.session._error is not None))
        error, state = self.session.grab(10)
        self.assertEqual(state, "error")
        self.assertIsInstance(error, ConnectionError)
        self.assertTrue(any("open again" in line for line in self.log.lines))
        second.feed(frame_bytes(value=7))
        frame, state = self.session.grab(1000)
        self.assertEqual((state, int(frame[0, 0])), ("ok", 7))
        self.assertEqual(len(commands(second)), 1)  # the settings were sent again

    def test_reconnect_failure_and_rate_limit(self):
        """A failed reconnect is logged; the next try waits RECONNECT_INTERVAL_S.

        Args: None.
        Returns: None.
        """
        first = self.open()
        self.sockets = [usbmux.UsbmuxError("refused")]
        first.finish()
        self.assertTrue(wait_until(lambda: self.session._error is not None))
        with mock.patch.object(iphone.time, "sleep") as sleep:
            self.session.grab(10)
            self.assertTrue(any("not reachable yet" in line for line in self.log.lines))
            error, state = self.session.grab(10)  # too soon: sleeps instead
        self.assertEqual(state, "error")
        self.assertIn("closed", str(error))
        sleep.assert_called_once_with(0.01)

    def test_silent_link_counts_as_lost(self):
        """No message for STALL_S drops and re-opens the link.

        Args: None.
        Returns: None.
        """
        self.open()
        self.sockets = [phone_socket()]
        self.session._last_message -= iphone.STALL_S + 1
        error, state = self.session.grab(10)
        self.assertEqual(state, "error")
        self.assertIn("no data from the iPhone", str(error))
        self.assertTrue(any("open again" in line for line in self.log.lines))


class SessionSettingsTests(SessionCase):
    """Settings sent to the phone and reported back."""

    def test_set_float_sends_commands(self):
        """Exposure, gain (as ISO) and frame rate become COMMANDs; others are ignored.

        Args: None.
        Returns: None.
        """
        sock = self.open()
        self.session.set_float("ExposureTime", 500)
        self.session.set_float("Gain", 20.0)
        self.session.set_float("AcquisitionFrameRate", 15)
        self.session.set_float("Gamma", 1.0)
        sent = commands(sock)[1:]
        self.assertEqual(sent[0], {"exposure_us": 500.0})
        self.assertAlmostEqual(sent[1]["iso"], 320.0)
        self.assertEqual(sent[2], {"fps": 15.0})
        self.assertEqual(len(sent), 3)
        self.assertEqual(self.session.requested["fps"], 15.0)

    def test_set_float_edge_cases(self):
        """Gain waits for the ISO range; a send failure is logged; closed is a no-op.

        Args: None.
        Returns: None.
        """
        self.sockets = [phone_socket(hello(limits={}))]
        sock = self.open()
        self.session.set_float("Gain", 3.0)
        self.assertEqual(len(commands(sock)), 1)  # only the starting command
        sock.send_error = OSError("broken pipe")
        self.session.set_float("ExposureTime", 100)
        self.assertTrue(any("Could not send ExposureTime" in line for line in self.log.lines))
        self.session.close()
        self.session.set_float("ExposureTime", 200)  # nothing to send to

    def test_send_command(self):
        """Phone-only settings go out as one COMMAND.

        Args: None.
        Returns: None.
        """
        sock = self.open()
        self.session.send_command(width=1280, height=720, refocus=True)
        self.assertEqual(commands(sock)[-1], {"width": 1280, "height": 720, "refocus": True})

    def test_take_settings_update(self):
        """Settings are reported once, after the viewer's own changes go quiet.

        Args: None.
        Returns: None.
        """
        self.assertIsNone(self.session.take_settings_update())  # nothing heard yet
        sock = self.open()
        first = self.session.take_settings_update()
        self.assertNotIn("exposure_us", first)  # just sent by the viewer
        self.assertEqual((first["width"], first["lens_position"], first["max_send_fps"]),
                         (1920, 0.5, 0.0))
        self.assertIsNone(self.session.take_settings_update())  # unchanged
        self.session._requested_at = {key: 0.0 for key in self.session._requested_at}
        later = self.session.take_settings_update()
        self.assertEqual((later["exposure_us"], later["gain_db"], later["fps"]),
                         (8000.0, 0.0, 30.0))
        sock.feed(iphone.encode_json(iphone.KIND_STATUS, status(
            settings=phone_settings(exposure_us=1234.0, iso=320.0, lens_position=-1))))
        self.assertTrue(wait_until(lambda: self.session.status.get("settings", {})
                                   .get("exposure_us") == 1234.0))
        changed = self.session.take_settings_update()
        self.assertEqual(changed, {"exposure_us": 1234.0, "gain_db": 20.0})
        self.session.send_command(lens_position=0.2)
        self.session._requested_at.pop("lens_position")
        self.assertIsNone(self.session.take_settings_update())

    def test_take_settings_update_without_iso(self):
        """Without an ISO the gain is left out.

        Args: None.
        Returns: None.
        """
        settings = phone_settings()
        del settings["iso"]
        self.sockets = [phone_socket(hello(settings=settings))]
        self.open()
        self.assertNotIn("gain_db", self.session.take_settings_update())

    def test_limits(self):
        """Phone limits become slider ranges clipped to the viewer's ranges.

        Args: None.
        Returns: None.
        """
        self.open()
        limits = self.session.limits()
        self.assertEqual(limits["ExposureTime"], (14.0, 50000.0))
        self.assertEqual(limits["Gain"][0], 0.0)
        self.assertAlmostEqual(limits["Gain"][1], 30.0)  # 40 dB clipped
        self.assertEqual(limits["AcquisitionFrameRate"], (1.0, 60.0))
        self.session.status = {"limits": phone_limits(iso=[0, 100], exposure_us=None,
                                                       fps=[60, 1])}
        self.assertEqual(self.session.limits(), {})
        self.session.status = {"limits": phone_limits(iso=[32])}
        self.assertNotIn("Gain", self.session.limits())

    def test_read_values_and_describe(self):
        """Values come from the newest frame; the status line names rates and drops.

        Args: None.
        Returns: None.
        """
        sock = self.open()
        empty = self.session.read_values()
        self.assertEqual((empty["exposure"], empty["gain"], empty["resulting"]),
                         (None, None, None))
        self.assertIn("no frames yet", self.session.describe(empty))
        self.assertIn("waiting for frames", self.session.describe(empty))
        sock.feed(iphone.encode_json(iphone.KIND_STATUS, status(
            dropped_frames=12, thermal="critical")))
        sock.feed(frame_bytes(iso=320.0))
        self.assertTrue(wait_until(lambda: self.session._frame_seq == 1))
        self.assertTrue(wait_until(lambda: self.session.status.get("dropped_frames") == 12))
        values = self.session.read_values()
        self.assertEqual(values["requested"], 25.0)
        self.assertAlmostEqual(values["gain"], 20.0)
        self.assertGreater(values["resulting"], 0)
        text = self.session.describe(values)
        for part in ("iPhone iPhone16,1 8 × 4 over USB", "fps received", "camera 30.0 fps",
                     "12 skipped", "exposure 8000 µs, ISO 320 (20.0 dB)",
                     "thermal state critical"):
            self.assertIn(part, text)
        self.session.status = {"limits": {}}
        self.assertNotIn("dB", self.session.describe(self.session.read_values() | {
            "exposure": 8000.0, "gain": None}))
        self.session.close()
        self.assertIn("link lost", self.session.describe(values))

    def test_read_stream_counters_is_empty(self):
        """Skipped frames are expected, so no counters are reported.

        Args: None.
        Returns: None.
        """
        self.assertEqual(self.session.read_stream_counters(), {})


if __name__ == "__main__":
    unittest.main()
