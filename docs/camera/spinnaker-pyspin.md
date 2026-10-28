# Spinnaker SDK and PySpin coding context

## Model of the SDK

Spinnaker uses GenICam node maps for configuration and an acquisition engine for
streamed image buffers. PySpin exposes both the generic node-map API and
QuickSpin-style camera properties. Prefer a cached, checked node handle when a
feature must work across firmware variations; QuickSpin properties are concise
when the property is known to exist.

The relevant node maps are:

- `cam.GetNodeMap()` — camera GenICam features.
- `cam.GetTLDeviceNodeMap()` — transport-layer device identity/information.
- `cam.GetTLStreamNodeMap()` — host stream settings, including buffer handling.

## Required ownership and lifecycle

Use this ordering and make cleanup exception-safe:

1. `system = PySpin.System.GetInstance()`; then `cam_list = system.GetCameras()`.
2. Get a camera, call `cam.Init()`, and configure writable nodes.
3. Call `cam.BeginAcquisition()`.
4. Repeatedly call `cam.GetNextImage(timeout_ms)`.
5. In an inner `finally`, call `cam.EndAcquisition()` only if acquisition began.
6. In an outer `finally`, call `cam.DeInit()` only if initialization completed;
   release camera references with `del cam`; clear the camera list; finally call
   `system.ReleaseInstance()`.

Never retain a camera-owned image buffer without releasing it. Every successful
`GetNextImage()` needs `image.Release()` in a `finally`, including incomplete or
failed frames. `Image.IsIncomplete()` and/or `GetImageStatus()` must be checked
before processing. A timeout is preferable to an unbounded wait in an
interactive application.

`image.GetNDArray()` can reference SDK-managed memory. Call `.copy()` before
releasing the image if the NumPy data must outlive that image buffer. Likewise,
if a converted `Image` is used, release the converted image after making a safe
copy or completing immediate processing.

## Node-access patterns

Retrieve and cast generic nodes by type, then verify access before using them:

```python
node = PySpin.CFloatPtr(cam.GetNodeMap().GetNode("ExposureTime"))
if PySpin.IsReadable(node) and PySpin.IsWritable(node):
    value = max(node.GetMin(), min(requested_us, node.GetMax()))
    node.SetValue(value)
```

For enumerations, verify both the enumeration node and the desired entry:

```python
node = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
entry = PySpin.CEnumEntryPtr(node.GetEntryByName("Continuous"))
if PySpin.IsWritable(node) and PySpin.IsReadable(entry):
    node.SetIntValue(entry.GetValue())
```

Cache nodes outside the frame loop; repeated node-map lookups add unnecessary
per-frame work. Treat unavailable/read-only nodes as a capability difference,
not an exceptional condition, unless the requested feature is mandatory.

## Local and official references

- Start with the matching local vendor example in
  `third_party/flir-spinnaker-4.2/Examples/Python3/` (`Acquisition.py`, `Trigger.py`,
  `BufferHandling.py`, `ImageFormatControl.py`, and the `*_QuickSpin.py`
  variants).
- Python guide: <https://softwareservices.flir.com/Spinnaker/4.4/guides/python-programmer-guide.html>
- Python getting-started acquisition and buffer-lifetime guidance: <https://softwareservices.flir.com/Spinnaker/4.4/getting-started/python.html>
- Core GenICam/node-map guide: <https://softwareservices.flir.com/Spinnaker/4.4/guides/programmer-guide.html>

