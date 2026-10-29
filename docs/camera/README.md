# Camera and Spinnaker context

This directory holds maintained, local research notes for agents working on the
active FLIR/Teledyne camera code. It is reference context, not a substitute for
querying the connected camera's GenICam node map: node availability, allowed
ranges, and SDK names can differ with firmware and installed Spinnaker version.

## Load guide

| Working on | Load |
| --- | --- |
| Any PySpin lifecycle, node-map, image-buffer, or error-handling code | `spinnaker-pyspin.md` |
| Features, settings, or limits specific to the BFS-U3-63S4M-C | `blackfly-bfs-u3-63s4m-c.md` |
| Preview latency, USB throughput, frame rate, ROI, pixel format, or triggering | `acquisition-and-performance.md` |
| Model detection, other FLIR models, or `camera_models.py` | `supported-models.md` |

For camera edits, load the first file plus the file(s) matching the changed
feature. The vendored examples in `third_party/flir-spinnaker-4.2/Examples/Python3/` are
the local implementation reference; do not modify them.

## Sources and currency

These notes were researched 2026-10-10 using Teledyne/FLIR primary sources.
The online SDK documentation referenced is Spinnaker 4.4.0.249. Use the API and
examples bundled with the installed SDK when they conflict with newer online
documentation.

