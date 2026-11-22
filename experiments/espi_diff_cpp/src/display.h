// Difference values to screen pixels: multiply by a gain, saturate at white.
#pragma once

#include <array>
#include <cstdint>

#include "frame.h"

namespace espi {

/// Maps 8-bit difference values to opaque grey 32-bit XRGB8888 pixels.
///
/// The gain is folded into a 256-entry table, so changing it costs 256
/// multiplications instead of one per pixel.
class GreyMap {
public:
    /// Build the table for a gain.
    ///
    /// Args:
    ///   gain: brightness multiplier; values below 0 are treated as 0.
    explicit GreyMap(double gain = 1.0);

    /// Rebuild the table for a new gain.
    ///
    /// Args:
    ///   gain: brightness multiplier; values below 0 are treated as 0.
    void set_gain(double gain);

    /// The gain the table was built for.
    ///
    /// Returns:
    ///   the multiplier.
    double gain() const { return gain_; }

    /// The screen pixel for one difference value.
    ///
    /// Args:
    ///   value: a difference value 0-255.
    ///
    /// Returns:
    ///   0xFFvvvvvv, where v = min(255, round(value * gain)).
    std::uint32_t operator[](std::uint8_t value) const { return table_[value]; }

    /// Convert a whole image into a 32-bit pixel buffer, such as a locked texture.
    ///
    /// Args:
    ///   image: the difference image.
    ///   dst: destination; at least image.height rows of dst_pitch bytes.
    ///   dst_pitch: bytes from one destination row to the next (>= 4 * width).
    void convert(const Frame& image, void* dst, int dst_pitch) const;

private:
    double gain_ = 1.0;
    std::array<std::uint32_t, 256> table_{};
};

}  // namespace espi
