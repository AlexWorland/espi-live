#include "luma.h"

#include <cstddef>
#include <cstring>

namespace espi {

void extract_luma(LumaLayout layout, const std::uint8_t* data, int width, int height, int pitch,
                  Frame& out) {
    out.resize(width, height);
    std::uint8_t* dst = out.pixels.data();
    const auto w = static_cast<std::size_t>(width);
    for (int y = 0; y < height; ++y) {
        const std::uint8_t* row = data + static_cast<std::ptrdiff_t>(y) * pitch;
        std::uint8_t* out_row = dst + static_cast<std::size_t>(y) * w;
        switch (layout) {
            case LumaLayout::Planar:
                std::memcpy(out_row, row, w);
                break;
            case LumaLayout::PackedEven:
                for (std::size_t x = 0; x < w; ++x) out_row[x] = row[2 * x];
                break;
            case LumaLayout::PackedOdd:
                for (std::size_t x = 0; x < w; ++x) out_row[x] = row[2 * x + 1];
                break;
        }
    }
}

}  // namespace espi
