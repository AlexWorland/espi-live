"""Tests for the usbmuxd client, against scripted sockets (no daemon, no network)."""

import os
import plistlib
import socket
import unittest
from unittest import mock

from espi_live import usbmux
from fake_iphone import FakeSocket, SocketFactory, device_entry, mux_reply


class AddressTests(unittest.TestCase):
    """Where the daemon is found."""

    def address(self, override=None, platform="linux"):
        """Resolve the address with a given environment and platform.

        Args: override is USBMUXD_SOCKET_ADDRESS or None; platform is sys.platform.
        Returns: default_address().
        """
        env = {} if override is None else {"USBMUXD_SOCKET_ADDRESS": override}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(usbmux.sys, "platform", platform):
            return usbmux.default_address()

    def test_platform_defaults(self):
        """Windows uses iTunes' TCP port; others the Unix socket.

        Args: None.
        Returns: None.
        """
        self.assertEqual(self.address(platform="win32"), ("127.0.0.1", 27015))
        self.assertEqual(self.address(platform="darwin"), "/var/run/usbmuxd")

    def test_overrides(self):
        """The environment variable accepts the libusbmuxd forms.

        Args: None.
        Returns: None.
        """
        self.assertEqual(self.address("UNIX:/tmp/mux"), "/tmp/mux")
        self.assertEqual(self.address("192.168.0.2:27015"), ("192.168.0.2", 27015))
        self.assertEqual(self.address("[::1]:27015"), ("::1", 27015))
        self.assertEqual(self.address("/tmp/other"), "/tmp/other")


class SocketTests(unittest.TestCase):
    """Opening the daemon socket and framing messages."""

    def test_open_socket_families(self):
        """TCP v4, v6 and Unix addresses pick the right family and set the timeout.

        Args: None.
        Returns: None.
        """
        factory = SocketFactory(FakeSocket(), FakeSocket(), FakeSocket())
        with mock.patch.object(usbmux.socket, "socket", factory):
            first = usbmux.open_socket(1.5, ("127.0.0.1", 27015))
            usbmux.open_socket(1.0, ("::1", 27015))
            usbmux.open_socket(1.0, "/var/run/usbmuxd")
        self.assertEqual(factory.families,
                         [socket.AF_INET, socket.AF_INET6, socket.AF_UNIX])
        self.assertEqual((first.timeout, first.address), (1.5, ("127.0.0.1", 27015)))

    def test_open_socket_uses_default_address(self):
        """Without an address the platform default is used.

        Args: None.
        Returns: None.
        """
        factory = SocketFactory(FakeSocket())
        with mock.patch.object(usbmux.socket, "socket", factory), \
                mock.patch.object(usbmux, "default_address", return_value="/x"):
            self.assertEqual(usbmux.open_socket().address, "/x")

    def test_open_socket_failure_names_itunes_for_tcp(self):
        """An unreachable TCP daemon suggests installing iTunes; the socket closes.

        Args: None.
        Returns: None.
        """
        refused = FakeSocket(connect_error=ConnectionRefusedError("refused"))
        with mock.patch.object(usbmux.socket, "socket", SocketFactory(refused)):
            with self.assertRaisesRegex(usbmux.UsbmuxError, "Install iTunes"):
                usbmux.open_socket(address=("127.0.0.1", 27015))
        self.assertTrue(refused.closed)
        missing = FakeSocket(connect_error=FileNotFoundError("missing"))
        with mock.patch.object(usbmux.socket, "socket", SocketFactory(missing)):
            with self.assertRaises(usbmux.UsbmuxError) as caught:
                usbmux.open_socket(address="/var/run/usbmuxd")
        self.assertNotIn("iTunes", str(caught.exception))

    def test_recv_exact_reports_early_close(self):
        """A peer that closes mid-message raises ConnectionError.

        Args: None.
        Returns: None.
        """
        sock = FakeSocket(b"ab")
        sock.finish()
        with self.assertRaises(ConnectionError):
            usbmux.recv_exact(sock, 4)
        self.assertEqual(usbmux.recv_exact(FakeSocket(b"abcd"), 3), b"abc")

    def test_send_plist_header(self):
        """Requests carry a little-endian header and the common keys.

        Args: None.
        Returns: None.
        """
        sock = FakeSocket()
        usbmux.send_plist(sock, usbmux.request_dict("ListDevices"), tag=9)
        length, version, kind, tag = usbmux.HEADER.unpack_from(sock.sent)
        self.assertEqual((length, version, kind, tag), (len(sock.sent), 1, 8, 9))
        body = plistlib.loads(bytes(sock.sent[16:]))
        self.assertEqual(body["MessageType"], "ListDevices")
        self.assertEqual(body["ProgName"], "espi-live")
        self.assertEqual(body["kLibUSBMuxVersion"], 3)

    def test_read_plist_rejects_bad_replies(self):
        """Wrong type, absurd length, unreadable or non-dict plists are errors.

        Args: None.
        Returns: None.
        """
        bad = (mux_reply({"a": 1}, kind=1),
               usbmux.HEADER.pack(usbmux.MAX_REPLY + 1, 1, 8, 1),
               mux_reply(b"not a plist"),
               mux_reply(plistlib.dumps([1, 2])))
        for data in bad:
            with self.subTest(data=data[:20]):
                with self.assertRaises(usbmux.UsbmuxError):
                    usbmux.read_plist(FakeSocket(data))
        self.assertEqual(usbmux.read_plist(FakeSocket(mux_reply({"a": 1}))), {"a": 1})


class DeviceTests(unittest.TestCase):
    """Listing devices and connecting to a port."""

    def test_parse_devices_keeps_usb_only(self):
        """A phone also listed over Wi-Fi appears once.

        Args: None.
        Returns: None.
        """
        reply = {"DeviceList": [device_entry(3, "AAA"), device_entry(4, "AAA", "Network"),
                                {"DeviceID": 5, "Properties": {"SerialNumber": "BBB"}}]}
        devices = usbmux.parse_devices(reply)
        self.assertEqual([(d.device_id, d.serial) for d in devices],
                         [(3, "AAA"), (5, "BBB")])
        self.assertEqual(devices[0].product_id, 4776)
        self.assertEqual(usbmux.parse_devices({}), [])

    def test_list_devices(self):
        """ListDevices is sent and the reply parsed; the socket is closed.

        Args: None.
        Returns: None.
        """
        sock = FakeSocket(mux_reply({"DeviceList": [device_entry()]}))
        with mock.patch.object(usbmux.socket, "socket", SocketFactory(sock)):
            devices = usbmux.list_devices(address="/mux")
        self.assertEqual(devices[0].serial, "00008110-001A2B3C4D5E801E")
        self.assertTrue(sock.closed)
        self.assertIn(b"ListDevices", bytes(sock.sent))

    def test_list_devices_read_failure(self):
        """A daemon that hangs up is reported as UsbmuxError.

        Args: None.
        Returns: None.
        """
        sock = FakeSocket()
        sock.finish()
        with mock.patch.object(usbmux.socket, "socket", SocketFactory(sock)):
            with self.assertRaisesRegex(usbmux.UsbmuxError, "did not list"):
                usbmux.list_devices(address="/mux")

    def test_swap_port(self):
        """Ports are sent byte-swapped, as htons does on Intel and ARM.

        Args: None.
        Returns: None.
        """
        self.assertEqual(usbmux.swap_port(0x1FA4), 0xA41F)  # 8100 -> 42015
        self.assertEqual(usbmux.swap_port(47820), socket.htons(47820))

    def test_connect_success(self):
        """A zero Result leaves a raw tunnel with no timeout.

        Args: None.
        Returns: None.
        """
        sock = FakeSocket(mux_reply({"MessageType": "Result", "Number": 0}))
        with mock.patch.object(usbmux.socket, "socket", SocketFactory(sock)):
            tunnel = usbmux.connect(3, 47820, address="/mux")
        self.assertIs(tunnel, sock)
        self.assertIsNone(sock.timeout)
        request = plistlib.loads(bytes(sock.sent[16:]))
        self.assertEqual((request["MessageType"], request["DeviceID"], request["PortNumber"]),
                         ("Connect", 3, usbmux.swap_port(47820)))

    def test_connect_refused(self):
        """Refusals carry a readable reason and close the socket.

        Args: None.
        Returns: None.
        """
        for number, text in ((3, "is the app open"), (99, "error 99")):
            sock = FakeSocket(mux_reply({"MessageType": "Result", "Number": number}))
            with self.subTest(number=number), \
                    mock.patch.object(usbmux.socket, "socket", SocketFactory(sock)):
                with self.assertRaisesRegex(usbmux.UsbmuxError, text):
                    usbmux.connect(3, 47820, address="/mux")
                self.assertTrue(sock.closed)

    def test_connect_no_answer(self):
        """A daemon that hangs up during Connect is a UsbmuxError.

        Args: None.
        Returns: None.
        """
        sock = FakeSocket()
        sock.finish()
        with mock.patch.object(usbmux.socket, "socket", SocketFactory(sock)):
            with self.assertRaisesRegex(usbmux.UsbmuxError, "did not answer"):
                usbmux.connect(3, 47820, address="/mux")
        self.assertTrue(sock.closed)


if __name__ == "__main__":
    unittest.main()
