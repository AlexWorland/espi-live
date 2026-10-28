# Project guidance

## Camera research context

- Local camera/SDK research notes live in `agent-context/`. Load
  `agent-context/README.md` first when planning a camera-related change; it
  directs you to the focused files needed for PySpin, this model, and
  acquisition/performance work.
- For any edit that imports PySpin, changes camera lifecycle, accesses a node
  map, or handles image buffers, load `agent-context/spinnaker-pyspin.md`.
- Also load `agent-context/blackfly-bfs-u3-63s4m-c.md` when changing hardware
  settings or assumptions for the BFS-U3-63S4M-C, and
  `agent-context/acquisition-and-performance.md` when changing streaming, ROI,
  pixel format, frame rate, preview latency, recording, or triggering.

## Directory boundaries

- Make new or changed application code in `active-project/`.
- Treat `saba-originals/` as preserved reference material. Do not edit it unless a task explicitly calls for changing an original Saba example.
- Treat `vendor-flir-spinnaker/` as third-party material. Do not modify it.
- Keep Windows and Bash launch/setup helpers in `setup-and-tools/` and update their paths whenever a referenced file moves.

## Python environment

- The intended camera runtime is Python 3.10 with the FLIR PySpin package installed, normally at `C:\Python310\PySpin\python.exe` on Windows. `active-project/espi_live.py --demo` may run without PySpin, but still requires OpenCV, NumPy, Pillow, and Tkinter.
- Other project scripts may depend on PySpin and sometimes Matplotlib. Do not add a virtual environment or dependency-management system unless requested.
- Before declaring an active Python change complete, run an appropriate syntax check such as `python -m py_compile` on the changed file. Do not require camera hardware for a syntax-only check.

## Files and outputs

- Put active captures in `active-project/output/` and avoid committing/generated-media churn unless specifically requested.
- Use descriptive, purpose-based filenames for active code; date-based script names belong in archived/reference material.
- Keep `README.md` accurate when the project layout, entry point, or setup instructions change.
