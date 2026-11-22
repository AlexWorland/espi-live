// A simulated speckle interferometer, so the viewer can run without a camera.
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "source.h"

namespace espi {

/// Speckle interferometer with a smooth reference beam and an object that
/// bulges out and back (a Gaussian dome, up to six fringes, every 12 s).
///
/// Each pixel follows the two-beam interference law
///   I = I_o + I_r + 2 sqrt(I_o I_r) cos(psi + delta),
/// with a random speckle intensity I_o and phase psi per speckle grain, a
/// uniform reference intensity I_r, and the deformation phase delta. Plus
/// +/-1.5 grey levels of sensor noise, rounded and clipped to 8 bits like a
/// camera. Subtracting a reference then shows dark fringes where delta is a
/// whole number of 2 pi.
class SyntheticSource final : public FrameSource {
public:
    /// Lay out the speckle field. Deterministic for a given seed.
    ///
    /// Args:
    ///   width: frame width in pixels.
    ///   height: frame height in pixels.
    ///   fps: frame rate that grab() paces to.
    ///   seed: random seed for the speckle and the noise.
    SyntheticSource(int width = 1024, int height = 768, double fps = 30.0,
                    std::uint32_t seed = 12345);

    /// Start the frame clock.
    ///
    /// Args:
    ///   error: unused; opening cannot fail.
    ///
    /// Returns:
    ///   true.
    bool open(std::string& error) override;

    /// Wait for the next frame time and render it.
    ///
    /// Args:
    ///   out: receives the frame.
    ///   timeout_ms: longest wait in milliseconds.
    ///   error: unused.
    ///
    /// Returns:
    ///   Ok, or Timeout if the next frame is not due within the timeout.
    GrabResult grab(Frame& out, int timeout_ms, std::string& error) override;

    /// Nothing to release.
    void close() override {}

    /// The source's name.
    ///
    /// Returns:
    ///   a name that says the frames are synthetic.
    std::string name() const override;

    /// Whether the frames are simulated.
    ///
    /// Returns:
    ///   true.
    bool synthetic() const override { return true; }

    /// Render the camera image at a moment, without pacing (used by tests).
    ///
    /// Args:
    ///   t_seconds: simulation time.
    ///   out: receives the frame.
    void render(double t_seconds, Frame& out);

    /// The deformation at the dome's centre.
    ///
    /// Args:
    ///   t_seconds: simulation time.
    ///
    /// Returns:
    ///   the peak deformation phase in fringes (multiples of 2 pi).
    static double peak_fringes(double t_seconds);

private:
    static constexpr int kPhaseSteps = 4096;  // phase resolution: 2 pi / 4096

    int width_;
    int height_;
    double fps_;
    std::vector<float> mean_;                  // I_o + I_r
    std::vector<float> modulation_;            // 2 sqrt(I_o I_r)
    std::vector<std::uint16_t> speckle_phase_; // psi in phase steps
    std::vector<float> shape_;                 // deformation shape, 1 at the dome's centre
    std::array<float, kPhaseSteps> cosine_{};
    std::uint32_t noise_state_;
    std::uint64_t frame_index_ = 0;
    std::chrono::steady_clock::time_point next_due_{};
};

}  // namespace espi
