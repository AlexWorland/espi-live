// Stand-ins for a build without the Spinnaker SDK: FLIR cameras are unavailable,
// and the viewer says so instead of failing to link.
#include "source.h"

namespace espi {

bool blackfly_supported() {
    return false;
}

std::unique_ptr<FrameSource> make_blackfly_source(const std::string& /*serial*/) {
    return nullptr;
}

std::vector<std::string> list_blackfly_cameras(std::string& error) {
    error = "this build has no FLIR support (CMake did not find the Spinnaker SDK)";
    return {};
}

}  // namespace espi
