"""In-memory stand-in for PySpin so camera code runs without the SDK.

Only the surface that espi_live.camera uses is modelled: System, camera
lists, cameras (with serial numbers on the transport-layer device map), node
maps, nodes with IsReadable/IsWritable, images, the pointer-cast helpers and
SpinnakerException.
"""

import time
import types

import numpy as np


class FakeSpinnakerException(Exception):
    """Stand-in for PySpin.SpinnakerException."""


class FakeNode:
    """A GenICam node (enum, enum entry, boolean or float)."""

    def __init__(self, value=0.0, minimum=0.0, maximum=100.0, readable=True,
                 writable=True, entries=None):
        """Create a node.

        Args: value is the initial value; minimum and maximum are the float
            limits; readable and writable are the access flags; entries maps
            entry names to FakeNode enum entries for enumeration nodes.
        Returns: None.
        """
        self.value = value
        self.minimum = minimum
        self.maximum = maximum
        self.readable = readable
        self.writable = writable
        self.entries = entries or {}
        self.fail_limits = False
        self.fail_get = False
        self.fail_set = False

    def GetValue(self):
        """Return the node value, or raise when configured to fail.

        Args: None.
        Returns: The stored value.
        """
        if self.fail_get:
            raise FakeSpinnakerException("get failed")
        return self.value

    def SetValue(self, value):
        """Store a value, or raise when configured to fail.

        Args: value is the new value.
        Returns: None.
        """
        if self.fail_set:
            raise FakeSpinnakerException("set failed")
        self.value = value

    def SetIntValue(self, value):
        """Select an enumeration entry by integer value.

        Args: value is the entry's integer value.
        Returns: None.
        """
        if self.fail_set:
            raise FakeSpinnakerException("set failed")
        self.value = value

    def GetMin(self):
        """Return the lower limit, or raise when configured to fail.

        Args: None.
        Returns: The minimum.
        """
        if self.fail_limits:
            raise FakeSpinnakerException("limits failed")
        return self.minimum

    def GetMax(self):
        """Return the upper limit.

        Args: None.
        Returns: The maximum.
        """
        return self.maximum

    def GetEntryByName(self, name):
        """Look up an enumeration entry.

        Args: name is the entry name.
        Returns: The entry, or an unreadable node when it does not exist.
        """
        return self.entries.get(name) or FakeNode(readable=False, writable=False)


def enum_entry(value):
    """Build a readable enum entry node.

    Args: value is the entry's integer value.
    Returns: A FakeNode whose GetValue() is value.
    """
    return FakeNode(value=value)


class FakeNodeMap:
    """A node map that returns unavailable nodes for unknown names."""

    def __init__(self, nodes):
        """Store the nodes.

        Args: nodes maps node names to FakeNode objects.
        Returns: None.
        """
        self.nodes = dict(nodes)
        self.raising = set()

    def GetNode(self, name):
        """Look up a node.

        Args: name is the node name.
        Returns: The node, or an unreadable, unwritable placeholder. Raises
            FakeSpinnakerException for names listed in self.raising.
        """
        if name in self.raising:
            raise FakeSpinnakerException(f"no access to {name}")
        return self.nodes.get(name) or FakeNode(readable=False, writable=False)


def default_nodes():
    """Build the node set a Blackfly exposes for configure_camera.

    Args: None.
    Returns: (camera nodes dict, transport-layer stream nodes dict).
    """
    mono = {"Mono8": enum_entry(17)}
    nodes = {
        "PixelFormat": FakeNode(entries=mono),
        "AcquisitionMode": FakeNode(entries={"Continuous": enum_entry(2)}),
        "ExposureAuto": FakeNode(entries={"Off": enum_entry(0)}),
        "GainAuto": FakeNode(entries={"Off": enum_entry(0)}),
        "ExposureTime": FakeNode(8000.0, 10.0, 100000.0),
        "Gain": FakeNode(0.0, 0.0, 48.0),
        "AcquisitionFrameRateEnable": FakeNode(False),
        "AcquisitionFrameRate": FakeNode(30.0, 1.0, 200.0),
        "AcquisitionResultingFrameRate": FakeNode(29.5, 1.0, 200.0,
                                                  writable=False),
    }
    stream = {"StreamBufferHandlingMode": FakeNode(
        entries={"NewestOnly": enum_entry(1)})}
    return nodes, stream


def identity_nodes():
    """Build the device-map identity nodes a BFS-U3-63S4M reports.

    Args: None.
    Returns: A dict of read-only string and integer nodes.
    """
    return {
        "DeviceFirmwareVersion": FakeNode("1907.0.390.0", writable=False),
        "SensorDescription": FakeNode("Sony IMX178 (1/1.8\" Mono CMOS)",
                                      writable=False),
        "SensorWidth": FakeNode(3072, writable=False),
        "SensorHeight": FakeNode(2048, writable=False),
    }


class FakeImage:
    """A grabbed image buffer."""

    def __init__(self, array, incomplete=False):
        """Wrap pixel data.

        Args: array is the NumPy frame; incomplete marks a damaged buffer.
        Returns: None.
        """
        self.array = array
        self.incomplete = incomplete
        self.released = False

    def IsIncomplete(self):
        """Report whether the buffer is damaged.

        Args: None.
        Returns: True for an incomplete image.
        """
        return self.incomplete

    def GetNDArray(self):
        """Return the pixel data.

        Args: None.
        Returns: The frame array.
        """
        return self.array

    def Release(self):
        """Mark the buffer as returned to the SDK.

        Args: None.
        Returns: None.
        """
        self.released = True


def steady_frames():
    """Create a frame source that yields varying Mono8 images.

    Args: None.
    Returns: A zero-argument callable returning a FakeImage.
    """
    counter = [0]

    def make():
        """Produce the next image.

        Args: None.
        Returns: A FakeImage with a slowly changing gradient.
        """
        counter[0] += 1
        base = np.tile(np.arange(64, dtype=np.uint8), (48, 1))
        return FakeImage(base + np.uint8(counter[0] % 50))
    return make


class FakeCamera:
    """A camera whose lifecycle calls can be made to fail."""

    def __init__(self, frame_source=None, serial="21234567",
                 model="Blackfly S BFS-U3-63S4M"):
        """Create a camera with the default node set.

        Args: frame_source is a callable returning an image, or raising
            FakeSpinnakerException; None uses steady_frames(); serial and
            model fill DeviceSerialNumber and DeviceModelName.
        Returns: None.
        """
        nodes, stream = default_nodes()
        nodes.update(identity_nodes())
        self.nodemap = FakeNodeMap(nodes)
        self.stream_nodemap = FakeNodeMap(stream)
        self.tl_device_nodemap = FakeNodeMap({
            "DeviceVendorName": FakeNode("FLIR", writable=False),
            "DeviceModelName": FakeNode(model, writable=False),
            "DeviceSerialNumber": FakeNode(serial, writable=False),
        })
        self.frame_source = frame_source or steady_frames()
        self.fail = set()
        self.calls = []

    def _step(self, name):
        """Record a lifecycle call and fail it when configured.

        Args: name is the call name.
        Returns: None. Raises RuntimeError if name is in self.fail.
        """
        self.calls.append(name)
        if name in self.fail:
            raise RuntimeError(f"{name} failed")

    def Init(self):
        """Initialize the camera.

        Args: None.
        Returns: None.
        """
        self._step("Init")

    def DeInit(self):
        """Deinitialize the camera.

        Args: None.
        Returns: None.
        """
        self._step("DeInit")

    def BeginAcquisition(self):
        """Start streaming.

        Args: None.
        Returns: None.
        """
        self._step("BeginAcquisition")

    def EndAcquisition(self):
        """Stop streaming.

        Args: None.
        Returns: None.
        """
        self._step("EndAcquisition")

    def GetNodeMap(self):
        """Return the device node map.

        Args: None.
        Returns: The FakeNodeMap.
        """
        return self.nodemap

    def GetTLDeviceNodeMap(self):
        """Return the transport-layer device node map.

        Args: None.
        Returns: The FakeNodeMap with vendor, model and serial.
        """
        return self.tl_device_nodemap

    def GetTLStreamNodeMap(self):
        """Return the transport-layer stream node map.

        Args: None.
        Returns: The FakeNodeMap.
        """
        return self.stream_nodemap

    def GetNextImage(self, timeout_ms):
        """Return the next image after a short pacing delay.

        Args: timeout_ms is ignored except to bound the pacing delay.
        Returns: A FakeImage. Propagates exceptions from the frame source.
        """
        time.sleep(min(timeout_ms, 5) / 1000.0)
        return self.frame_source()


class FakeCameraList:
    """The list returned by System.GetCameras()."""

    def __init__(self, cameras):
        """Hold cameras.

        Args: cameras is a list of FakeCamera.
        Returns: None.
        """
        self.cameras = cameras
        self.fail_clear = False
        self.cleared = False

    def GetSize(self):
        """Count the cameras.

        Args: None.
        Returns: The number of cameras.
        """
        return len(self.cameras)

    def GetByIndex(self, index):
        """Fetch a camera.

        Args: index selects the camera.
        Returns: The FakeCamera.
        """
        return self.cameras[index]

    def Clear(self):
        """Release the list.

        Args: None.
        Returns: None. Raises RuntimeError when fail_clear is set.
        """
        self.cleared = True
        if self.fail_clear:
            raise RuntimeError("clear failed")


class FakeSystem:
    """The singleton PySpin system."""

    def __init__(self, cameras):
        """Create a system that sees the given cameras.

        Args: cameras is a list of FakeCamera.
        Returns: None.
        """
        self.camera_list = FakeCameraList(cameras)
        self.fail_release = False
        self.released = False
        self.instances = 0  # GetInstance calls
        self.releases = 0   # ReleaseInstance calls

    def GetCameras(self):
        """Enumerate cameras.

        Args: None.
        Returns: The FakeCameraList (shared, like the SDK's cameras).
        """
        return self.camera_list

    def ReleaseInstance(self):
        """Release the system.

        Args: None.
        Returns: None. Raises RuntimeError when fail_release is set.
        """
        self.released = True
        self.releases += 1
        if self.fail_release:
            raise RuntimeError("release failed")


def make_pyspin(system):
    """Build a fake PySpin module around one system.

    Args: system is the FakeSystem that System.GetInstance() returns.
    Returns: A namespace with the PySpin names camera.py uses.
    """
    def get_instance():
        """Count and return the system, like PySpin.System.GetInstance.

        Args: None.
        Returns: The FakeSystem.
        """
        system.instances += 1
        return system

    return types.SimpleNamespace(
        System=types.SimpleNamespace(GetInstance=get_instance),
        CEnumerationPtr=lambda node: node,
        CEnumEntryPtr=lambda node: node,
        CBooleanPtr=lambda node: node,
        CFloatPtr=lambda node: node,
        CStringPtr=lambda node: node,
        CIntegerPtr=lambda node: node,
        IsReadable=lambda node: node.readable,
        IsWritable=lambda node: node.writable,
        SpinnakerException=FakeSpinnakerException,
        __file__="/fake/PySpin.py",
    )
