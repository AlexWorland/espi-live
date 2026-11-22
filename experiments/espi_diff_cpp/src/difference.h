// Subtraction ESPI: |live - base| per pixel, where the base is the previous
// frame or a captured reference frame.
#pragma once

#include <cstddef>
#include <cstdint>

#include "frame.h"

namespace espi {

/// Which image the live frame is subtracted from.
enum class Mode {
    Previous,   ///< the frame just before it (shows what changed in one frame period)
    Reference,  ///< a frame captured on request (shows what changed since then)
};

/// Write |a - b| for each pixel. The result never wraps or saturates.
///
/// Args:
///   a: first image, `count` bytes.
///   b: second image, `count` bytes.
///   out: destination, `count` bytes; may alias `a` or `b`.
///   count: number of pixels.
void absolute_difference(const std::uint8_t* a, const std::uint8_t* b, std::uint8_t* out,
                         std::size_t count);

/// Turns a stream of camera frames into a stream of difference images.
///
/// It is not thread-safe: one thread feeds it every frame in order.
class Differencer {
public:
    /// The mode of the next difference image.
    ///
    /// Returns:
    ///   Previous or Reference.
    Mode mode() const { return mode_; }

    /// Compare each frame with the one before it from the next frame on.
    void use_previous();

    /// Make the next frame the reference and switch to Reference mode.
    void capture_reference();

    /// Feed one camera frame, in arrival order.
    ///
    /// The frame's buffer is kept as the next "previous" frame: on return
    /// `frame` holds an old buffer of unspecified size and contents, which the
    /// caller can reuse for its next grab without allocating. When the size
    /// changes, the stored frames are dropped; in Reference mode the new frame
    /// becomes the reference.
    ///
    /// Args:
    ///   frame: the newest camera frame; swapped with an internal buffer.
    ///   out: receives |frame - base| when one can be computed.
    ///
    /// Returns:
    ///   true when `out` holds a new difference image; false for the first
    ///   frame in Previous mode (nothing to subtract yet).
    bool push(Frame& frame, Frame& out);

private:
    Mode mode_ = Mode::Previous;
    bool capture_pending_ = false;
    Frame previous_;
    Frame reference_;
};

}  // namespace espi
