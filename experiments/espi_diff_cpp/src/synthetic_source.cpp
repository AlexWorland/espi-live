#include "synthetic_source.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <thread>

namespace espi {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kGrain = 2;               // speckle grain size in pixels
constexpr float kMeanSpeckle = 40.0f;   // mean object-beam intensity, grey levels
constexpr float kReference = 40.0f;     // reference-beam intensity, grey levels
constexpr double kMaxFringes = 6.0;     // deformation at the dome's centre at full bulge
constexpr double kCycleSeconds = 12.0;  // out and back again
constexpr float kNoise = 1.5f;          // sensor noise, +/- grey levels

}  // namespace

SyntheticSource::SyntheticSource(int width, int height, double fps, std::uint32_t seed)
    : width_(width), height_(height), fps_(fps), noise_state_(seed | 1u) {
    const auto count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    mean_.resize(count);
    modulation_.resize(count);
    speckle_phase_.resize(count);
    shape_.resize(count);

    for (int i = 0; i < kPhaseSteps; ++i) {
        cosine_[static_cast<std::size_t>(i)] = static_cast<float>(std::cos(2.0 * kPi * i / kPhaseSteps));
    }

    // One random intensity and phase per speckle grain. Fully developed
    // speckle has exponentially distributed intensity and uniform phase.
    std::mt19937 random(seed);
    std::exponential_distribution<float> intensity(1.0f / kMeanSpeckle);
    std::uniform_int_distribution<int> phase(0, kPhaseSteps - 1);
    const int grains_x = (width + kGrain - 1) / kGrain;
    const int grains_y = (height + kGrain - 1) / kGrain;
    std::vector<float> grain_intensity(static_cast<std::size_t>(grains_x) * grains_y);
    std::vector<std::uint16_t> grain_phase(grain_intensity.size());
    for (std::size_t g = 0; g < grain_intensity.size(); ++g) {
        grain_intensity[g] = intensity(random);
        grain_phase[g] = static_cast<std::uint16_t>(phase(random));
    }

    const double cx = 0.45 * width;
    const double cy = 0.50 * height;
    const double sigma = 0.22 * std::min(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const auto i = static_cast<std::size_t>(y) * width + x;
            const auto g = static_cast<std::size_t>(y / kGrain) * grains_x + x / kGrain;
            const float object = grain_intensity[g];
            mean_[i] = object + kReference;
            modulation_[i] = 2.0f * std::sqrt(object * kReference);
            speckle_phase_[i] = grain_phase[g];
            const double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            shape_[i] = static_cast<float>(std::exp(-r2 / (2.0 * sigma * sigma)));
        }
    }
}

bool SyntheticSource::open(std::string& /*error*/) {
    frame_index_ = 0;
    next_due_ = std::chrono::steady_clock::now();
    return true;
}

std::string SyntheticSource::name() const {
    return "Synthetic interferometer (DEMO, not a camera)";
}

double SyntheticSource::peak_fringes(double t_seconds) {
    return kMaxFringes * 0.5 * (1.0 - std::cos(2.0 * kPi * t_seconds / kCycleSeconds));
}

void SyntheticSource::render(double t_seconds, Frame& out) {
    out.resize(width_, height_);
    const float steps = static_cast<float>(peak_fringes(t_seconds) * kPhaseSteps);
    const std::size_t count = out.pixels.size();
    std::uint32_t state = noise_state_;
    for (std::size_t i = 0; i < count; ++i) {
        const int delta = static_cast<int>(shape_[i] * steps);
        const int index = (speckle_phase_[i] + delta) & (kPhaseSteps - 1);
        // xorshift32: cheap per-pixel noise
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        const float noise = (static_cast<float>(state & 0xFFFFu) / 65535.0f - 0.5f) * 2.0f * kNoise;
        const float value = mean_[i] + modulation_[i] * cosine_[static_cast<std::size_t>(index)] + noise;
        out.pixels[i] = static_cast<std::uint8_t>(std::clamp(value + 0.5f, 0.0f, 255.0f));
    }
    noise_state_ = state;
}

GrabResult SyntheticSource::grab(Frame& out, int timeout_ms, std::string& /*error*/) {
    using clock = std::chrono::steady_clock;
    const auto now = clock::now();
    if (now < next_due_) {
        std::this_thread::sleep_until(std::min(next_due_, now + std::chrono::milliseconds(timeout_ms)));
        if (clock::now() < next_due_) {
            return GrabResult::Timeout;
        }
    }
    render(static_cast<double>(frame_index_) / fps_, out);
    ++frame_index_;
    const auto period = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(1.0 / fps_));
    next_due_ += period;
    if (clock::now() - next_due_ > period) {
        next_due_ = clock::now();  // fell behind (debugger, sleep): don't burst to catch up
    }
    return GrabResult::Ok;
}

std::unique_ptr<FrameSource> make_synthetic_source() {
    return std::make_unique<SyntheticSource>();
}

}  // namespace espi
