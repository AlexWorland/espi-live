#include "display.h"

#include <cmath>
#include <cstddef>

namespace espi {

GreyMap::GreyMap(double gain) {
    set_gain(gain);
}

void GreyMap::set_gain(double gain) {
    gain_ = gain > 0.0 ? gain : 0.0;
    for (int value = 0; value < 256; ++value) {
        const double scaled = std::round(value * gain_);
        const std::uint32_t v = scaled >= 255.0 ? 255u : static_cast<std::uint32_t>(scaled);
        table_[static_cast<std::size_t>(value)] = 0xFF000000u | (v << 16) | (v << 8) | v;
    }
}

void GreyMap::convert(const Frame& image, void* dst, int dst_pitch) const {
    auto* row_bytes = static_cast<std::uint8_t*>(dst);
    const std::uint8_t* src = image.pixels.data();
    for (int y = 0; y < image.height; ++y) {
        auto* row = reinterpret_cast<std::uint32_t*>(row_bytes + static_cast<std::ptrdiff_t>(y) * dst_pitch);
        for (int x = 0; x < image.width; ++x) {
            row[x] = table_[*src++];
        }
    }
}

}  // namespace espi
