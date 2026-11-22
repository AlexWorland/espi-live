// Brightness (luma) out of the YUV layouts that capture devices deliver.
#pragma once

#include <cstdint>

#include "frame.h"

namespace espi {

/// Where the luma bytes sit in a capture device's pixel data.
enum class LumaLayout {
    PackedEven,  ///< YUY2 / YVYU: bytes Y0 U Y1 V, luma in every even byte
    PackedOdd,   ///< UYVY: bytes U Y0 V Y1, luma in every odd byte
    Planar,      ///< NV12 / NV21 / I420 / YV12: a full-size luma plane comes first
};

/// Copy the luma of one captured image into a greyscale frame.
///
/// The values are kept as the device sent them (video range, black about 16
/// and white about 235); no colour conversion or range expansion is applied,
/// so nothing rounds the speckle before the difference.
///
/// Args:
///   layout: where the luma bytes are.
///   data: start of the image (of the luma plane for Planar).
///   width: image width in pixels.
///   height: image height in pixels.
///   pitch: bytes from one row to the next (of the luma plane for Planar).
///   out: receives a width x height greyscale frame.
void extract_luma(LumaLayout layout, const std::uint8_t* data, int width, int height, int pitch,
                  Frame& out);

}  // namespace espi
