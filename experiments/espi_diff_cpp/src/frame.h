// One 8-bit greyscale image with packed rows.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace espi {

/// An 8-bit greyscale image. Rows are packed: the stride is the width.
struct Frame {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;

    /// Set the size and allocate storage. Pixel values are left as they are.
    ///
    /// Args:
    ///   w: width in pixels.
    ///   h: height in pixels.
    void resize(int w, int h) {
        width = w;
        height = h;
        pixels.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    }

    /// Whether another frame has the same width and height.
    ///
    /// Args:
    ///   other: the frame to compare with.
    ///
    /// Returns:
    ///   true when both dimensions match.
    bool same_size(const Frame& other) const {
        return width == other.width && height == other.height;
    }

    /// Whether the frame holds no pixels.
    ///
    /// Returns:
    ///   true when no image has been stored yet.
    bool empty() const { return pixels.empty(); }
};

}  // namespace espi
