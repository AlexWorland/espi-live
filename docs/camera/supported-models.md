# Other FLIR/Teledyne cameras and model detection

Researched 2026-10-28. `src/espi_live/camera_models.py` uses these notes to
identify the connected camera; `camera.read_identity()` reads the nodes.

## How the viewer identifies a camera

Every GenICam camera describes itself through read-only string nodes. The SDK's
own `Enumeration.py` example reads them from the transport-layer device map
(`cam.GetTLDeviceNodeMap()`), which is available even before `Init()`:

| Node | Map | Example (BFS-U3-63S4M) |
| --- | --- | --- |
| `DeviceVendorName` | TL device (device map fallback) | `FLIR` |
| `DeviceModelName` | TL device (device map fallback) | `Blackfly S BFS-U3-63S4M` |
| `DeviceSerialNumber` | TL device (device map fallback) | `21234567` |
| `DeviceFirmwareVersion` | device | firmware string |
| `SensorDescription` | device (Blackfly S Device Control) | sensor model string |
| `SensorWidth`/`SensorHeight` (else `WidthMax`/`HeightMax`) | device | `3072`/`2048` |

Every read is guarded (`IsReadable`, `SpinnakerException` → empty), so cameras
that lack a node still open. Older Point Grey cameras report the vendor as
`Point Grey Research`.

## Part-number scheme

Teledyne's part numbers are `FAMILY-INTERFACE-SENSOR[-MOUNT]`; e.g.
`BFS-U3-63S4M-C`:

| Segment | Meaning | Codes |
| --- | --- | --- |
| Family | product line | `BFS` Blackfly S, `BFLY` Blackfly, `GS3` Grasshopper3, `CM3` Chameleon3, `ORX` Oryx, `FFY` Firefly, `FL3` Flea3 |
| Interface | transport | `U3` USB3, `GE` GigE, `PGE` PoE GigE, `10G`/`10GS` 10GigE |
| Sensor: digits | resolution in 0.1 MP | `63` = 6.3 MP, `200` = 20 MP |
| Sensor: letter | sensor maker | `S` Sony, `Y` onsemi, `M` Micron |
| Sensor: digit | optical-format code | e.g. `4` for the 1/1.8" IMX178 (see Teledyne's table) |
| Sensor: last letter | spectrum | `M` mono, `C` colour, `S` SWIR |
| Mount suffix | lens mount (not reported by the camera) | `-C` C-mount, `-CS` CS-mount |

Position decides meaning: the `M` after the digits is the maker (Micron), the
final `M` is monochrome, and the `S` in `BFS` is part of the family name.

## Blackfly S USB3 monochrome models

The table in `KNOWN_MODELS` holds sensor and shutter only. Published frame
rates disagree between sources (51S5M: 73 vs 75 fps; 200S6M: 16 vs 18 fps), so
the viewer always uses the limits the camera reports.

| Model | Sensor | Shutter | Full resolution | Notes |
| --- | --- | --- | --- | --- |
| BFS-U3-63S4M | Sony IMX178, 1/1.8" | rolling, global reset | 3072 × 2048 | reference camera for this viewer (~59 fps) |
| BFS-U3-200S6M | Sony IMX183, 1" | rolling, global reset | 5472 × 3648 | 20 MP; ~16–18 fps; 10/12-bit ADC; gain 0–27 dB; exposure from 69 µs; see `experiments/espi_live_bfs_u3_200s6m.py` |
| BFS-U3-51S5M | Sony IMX250, 2/3" | global | 2448 × 2048 | ~73–75 fps |
| BFS-U3-50S5M | Sony IMX264, 2/3" | global | 2448 × 2048 | ~35 fps |
| BFS-U3-23S6M | Sony IMX249, 1/1.2" | global | 1920 × 1200 | ~41 fps |
| BFS-U3-16S2M | Sony IMX273, 1/2.9" | global | 1440 × 1080 | ~226 fps |
| BFS-U3-04S2M | Sony IMX287, 1/2.9" | global | 720 × 540 | ~522 fps |
| BFS-U3-123S6M | Sony IMX253, 1.1" | global | 4096 × 3000 | ~30 fps |

Other models (other Blackfly S sensors, GigE variants, Grasshopper3,
Chameleon3, Oryx) are still recognised and named from their part number; the
viewer logs that they have not been tested and relies on their reported limits.

## What changes between models for ESPI

- **Mono vs colour.** ESPI relies on per-pixel speckle; Mono8 from a colour
  sensor is interpolated from the Bayer pattern. The viewer warns for colour
  models.
- **Shutter.** The IMX178 and IMX183 are rolling-shutter sensors with global
  reset; the Pregius sensors (IMX250/264/249/273/287/253) are global shutter.
  This matters with pulsed lasers or vibrating samples.
- **Frame size and rate.** The 30 fps default is clamped to the camera's
  `AcquisitionFrameRate` maximum, so a 20 MP camera runs at its own maximum
  (about 16–18 fps). Display, saving and recording use the frame's actual size.

## Analog cameras through a capture adapter

Cameras that are not FLIR/Spinnaker cameras reach the viewer as video devices
through OpenCV. The first is the Sony XC-ST50 on a Hauppauge USB-Live2; see
`sony-xc-st50-usb-live2.md`. Recognised adapters are listed in
`capture_models.ADAPTERS`. Other video devices are listed as "not yet
supported", opened only when someone switches them on, and then reported to
GitHub as `camera-support` issues.

## Sources

- Teledyne, "Understanding Teledyne FLIR machine vision part numbers": <https://www.teledynevisionsolutions.com/support/support-center/article/iis/understanding-product-part-numbers/>
- BFS-U3-63S4M Device Control (identity nodes): <https://softwareservices.flir.com/BFS-U3-63S4/latest/Model/public/DeviceControl.html>
- BFS-U3-200S6 specification: <https://softwareservices.flir.com/BFS-U3-200S6/latest/Model/spec.html>
- BFS-U3-51S5 specification: <https://softwareservices.flir.com/BFS-U3-51S5/latest/Model/spec.html>
- Teledyne Blackfly S USB3 product pages: <https://www.teledynevisionsolutions.com/products/blackfly-s-usb3?vertical=machine+vision&segment=iis>
- Stemmer Imaging Blackfly S USB3 series table: <https://www.stemmer-imaging.com/en/series-teledyne-flir-blackfly-s-usb3>
- Edmund Optics listings (BFS-U3-63S4M-C, BFS-U3-200S6M-C, BFS-U3-51S5M-C, BFS-U3-50S5M-C)
- Vendored SDK examples: `third_party/flir-spinnaker-4.2/Examples/Python3/Enumeration.py`
