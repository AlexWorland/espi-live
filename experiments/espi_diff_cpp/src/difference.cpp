#include "difference.h"

#include <utility>

namespace espi {

void absolute_difference(const std::uint8_t* a, const std::uint8_t* b, std::uint8_t* out,
                         std::size_t count) {
    // max - min instead of abs(int(a) - int(b)): it stays in 8 bits, so the
    // compiler turns the loop into byte-wide SIMD max/min/subtract (16 to 64
    // pixels per instruction) with no widening.
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint8_t x = a[i];
        const std::uint8_t y = b[i];
        const std::uint8_t hi = x > y ? x : y;
        const std::uint8_t lo = x > y ? y : x;
        out[i] = static_cast<std::uint8_t>(hi - lo);
    }
}

void Differencer::use_previous() {
    mode_ = Mode::Previous;
    capture_pending_ = false;
}

void Differencer::capture_reference() {
    capture_pending_ = true;
}

bool Differencer::push(Frame& frame, Frame& out) {
    if (!previous_.empty() && !previous_.same_size(frame)) {
        previous_ = Frame{};
        reference_ = Frame{};
        if (mode_ == Mode::Reference) {
            capture_pending_ = true;
        }
    }
    if (capture_pending_) {
        reference_ = frame;  // the one copy per capture; every other frame is swapped
        mode_ = Mode::Reference;
        capture_pending_ = false;
    }

    const Frame& base = mode_ == Mode::Reference ? reference_ : previous_;
    bool produced = false;
    if (!base.empty()) {
        out.resize(frame.width, frame.height);
        absolute_difference(frame.pixels.data(), base.pixels.data(), out.pixels.data(),
                            frame.pixels.size());
        produced = true;
    }
    std::swap(previous_, frame);
    return produced;
}

}  // namespace espi
