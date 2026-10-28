# Active project guidance

## Required camera context

- Before changing camera code, load `../agent-context/README.md` and the files
  it identifies for the work at hand.
- Always load `../agent-context/spinnaker-pyspin.md` for a PySpin, camera
  lifecycle, node-map, or image-buffer change.
- Load `../agent-context/blackfly-bfs-u3-63s4m-c.md` for model-specific
  settings, and `../agent-context/acquisition-and-performance.md` for changes
  involving acquisition, ROI, format, throughput, preview latency, recording,
  or triggering.

## Entry points

- `espi_live.py` is the maintained ESPI application and the primary script to improve.
- `espi_live.py --demo` starts synthetic video without camera detection. Normal startup offers demo mode if no camera is detected or PySpin is unavailable. Keep demo clearly labeled in the controls, viewer, logs, and exported media so it cannot be mistaken for camera data.
- The demo control panel can retry camera detection with **Switch to live capture**. Preserve safe cleanup after a failed retry; on success, reset demo measurement state and stop a demo recording before live frames arrive.
- `camera_live_control.py` is a separate live camera-control helper.
- Put tentative work in `experiments/`; promote it to an active entry point only when it becomes the intended maintained implementation.

## Camera-code practices

- Preserve safe PySpin camera lifecycle handling: initialize, begin acquisition, end acquisition, deinitialize, and release system resources even on errors.
- Keep camera settings resilient across supported devices: verify PySpin node readability/writability before changing a node, and clamp configurable numeric values to device limits.
- Do not change exposure, gain, frame-rate, or recording defaults without clearly calling out the behavioral impact.
- Retain the separate Tk viewer, controls, log window, and documented keyboard shortcuts unless the task explicitly changes the interface. Do not reintroduce OpenCV HighGUI windows into the Tk event loop; resizing mixed-toolkit windows caused fatal crashes on Windows.
- Keep Tk painting inside each preview frame's deadline. Servicing events after the deadline caused a reproducible ~21 fps result at a 30 fps demo setting; avoid adding a second per-frame event pass. Do not assume the Raw-view display rate equals the camera's acquisition rate.
- Document each new or changed function's purpose, inputs, and output in its docstring.

## Outputs and verification

- Store recordings under `output/recordings/` and still frames under `output/frames/` when adding or updating output behavior.
- Do not run camera-acquisition programs without user direction; camera hardware may be unavailable or in use.
- Use `espi_live.py --demo` for camera-free GUI checks when OpenCV, NumPy, Pillow, Tkinter, and a display are available. It must not initialize PySpin or acquire camera frames; syntax checks require neither a display nor camera hardware.
- After edits, syntax-check the changed Python file with `python -m py_compile` when possible.
