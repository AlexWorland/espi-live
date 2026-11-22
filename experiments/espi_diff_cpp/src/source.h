// Where frames come from: a FLIR Blackfly, a video capture device or a
// synthetic interferometer.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "frame.h"

namespace espi {

/// What one grab produced.
enum class GrabResult {
    Ok,       ///< `out` holds a new frame
    Timeout,  ///< no frame within the timeout; try again
    Skipped,  ///< a frame arrived but is unusable (incomplete transfer); try again
    Error,    ///< the source failed; `error` says why, and it should be closed
};

/// A camera-like source of 8-bit greyscale frames.
///
/// open() and close() run on the main thread; grab() runs on the processing
/// thread. Calls never overlap: the processing thread starts after open() and
/// is joined before close().
class FrameSource {
public:
    virtual ~FrameSource() = default;

    /// Open the device and start streaming.
    ///
    /// Args:
    ///   error: receives a message on failure.
    ///
    /// Returns:
    ///   true when frames can be grabbed.
    virtual bool open(std::string& error) = 0;

    /// Wait up to `timeout_ms` for the next frame.
    ///
    /// Args:
    ///   out: receives the frame; resized as needed.
    ///   timeout_ms: longest wait in milliseconds.
    ///   error: receives a message when the result is Error.
    ///
    /// Returns:
    ///   what happened.
    virtual GrabResult grab(Frame& out, int timeout_ms, std::string& error) = 0;

    /// Stop streaming and release the device. Safe to call more than once.
    virtual void close() = 0;

    /// A human-readable name for titles and logs.
    ///
    /// Returns:
    ///   the name, for example "Blackfly S BFS-U3-63S4M (SN 12345678)".
    virtual std::string name() const = 0;

    /// Whether the frames are simulated rather than measured.
    ///
    /// Returns:
    ///   true for the synthetic source; the viewer then marks every frame as demo data.
    virtual bool synthetic() const { return false; }

    /// The SDL camera instance ID, so the viewer can tell when this device is unplugged.
    ///
    /// Returns:
    ///   the ID for a capture device, 0 for any other source.
    virtual std::uint32_t device_id() const { return 0; }
};

/// A synthetic speckle interferometer (no hardware).
///
/// Returns:
///   a source that is ready to open.
std::unique_ptr<FrameSource> make_synthetic_source();

/// Whether this build can talk to FLIR cameras (it was built with the Spinnaker SDK).
///
/// Returns:
///   true when make_blackfly_source() can return a source.
bool blackfly_supported();

/// A FLIR Blackfly (or other Spinnaker camera).
///
/// Args:
///   serial: serial number to open; empty opens the first camera found.
///
/// Returns:
///   a source that is ready to open, or nullptr when the build has no Spinnaker support.
std::unique_ptr<FrameSource> make_blackfly_source(const std::string& serial);

/// Describe every connected Spinnaker camera without opening any.
///
/// Args:
///   error: receives a message if the SDK could not be queried.
///
/// Returns:
///   one line per camera: model and serial number.
std::vector<std::string> list_blackfly_cameras(std::string& error);

/// A video capture device (USB capture stick, UVC camera) through SDL3.
/// SDL must have been initialised with SDL_INIT_CAMERA.
///
/// Args:
///   index: position in list_capture_devices().
///
/// Returns:
///   a source that is ready to open.
std::unique_ptr<FrameSource> make_capture_source(int index);

/// Describe every video capture device. SDL must have been initialised with SDL_INIT_CAMERA.
///
/// Args:
///   error: receives a message if the devices could not be listed.
///
/// Returns:
///   one line per device: index, name and its luma-capable formats.
std::vector<std::string> list_capture_devices(std::string& error);

}  // namespace espi
