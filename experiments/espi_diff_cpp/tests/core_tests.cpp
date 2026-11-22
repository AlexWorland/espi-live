// Camera-free tests for the difference core. No framework: each CHECK that
// fails prints its line, and the exit status is the failure count.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "difference.h"
#include "display.h"
#include "frame.h"
#include "luma.h"
#include "pipeline.h"
#include "synthetic_source.h"

namespace {

int failures = 0;

#define CHECK(condition)                                                                 \
    do {                                                                                 \
        if (!(condition)) {                                                              \
            ++failures;                                                                  \
            std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK(" #condition ") failed\n"; \
        }                                                                                \
    } while (0)

/// A frame filled from a function of the pixel position.
///
/// Args:
///   w: width.
///   h: height.
///   value: maps (x, y) to a grey level.
///
/// Returns:
///   the frame.
template <typename F>
espi::Frame make_frame(int w, int h, F value) {
    espi::Frame f;
    f.resize(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) f.pixels[static_cast<std::size_t>(y) * w + x] = static_cast<std::uint8_t>(value(x, y));
    return f;
}

/// Whether `out` is |a - b| pixel for pixel.
///
/// Args:
///   out: the result to check.
///   a: first frame.
///   b: second frame.
///
/// Returns:
///   true when every pixel matches.
bool is_difference(const espi::Frame& out, const espi::Frame& a, const espi::Frame& b) {
    if (!out.same_size(a) || !out.same_size(b)) return false;
    for (std::size_t i = 0; i < out.pixels.size(); ++i) {
        if (out.pixels[i] != std::abs(int(a.pixels[i]) - int(b.pixels[i]))) return false;
    }
    return true;
}

/// Every pair of 8-bit values, so no case of the max-min trick is untested.
void test_absolute_difference_exhaustive() {
    std::vector<std::uint8_t> a(65536), b(65536), out(65536);
    for (int i = 0; i < 65536; ++i) {
        a[i] = static_cast<std::uint8_t>(i & 0xFF);
        b[i] = static_cast<std::uint8_t>(i >> 8);
    }
    espi::absolute_difference(a.data(), b.data(), out.data(), out.size());
    bool all = true;
    for (int i = 0; i < 65536; ++i) all = all && out[i] == std::abs(int(a[i]) - int(b[i]));
    CHECK(all);

    // In place (out aliases a) gives the same result.
    std::vector<std::uint8_t> in_place = a;
    espi::absolute_difference(in_place.data(), b.data(), in_place.data(), in_place.size());
    CHECK(in_place == out);
}

/// Previous mode subtracts consecutive frames and needs two frames to start.
void test_previous_mode() {
    const auto f1 = make_frame(5, 3, [](int x, int y) { return x * 10 + y; });
    const auto f2 = make_frame(5, 3, [](int x, int y) { return 200 - x * 7 - y; });
    const auto f3 = make_frame(5, 3, [](int x, int) { return 255 - x; });
    espi::Differencer d;
    espi::Frame in, out;
    in = f1;
    CHECK(!d.push(in, out));
    in = f2;
    CHECK(d.push(in, out));
    CHECK(is_difference(out, f2, f1));
    CHECK(d.mode() == espi::Mode::Previous);
    in = f3;
    CHECK(d.push(in, out));
    CHECK(is_difference(out, f3, f2));
}

/// A captured reference stays fixed while frames keep arriving, and switching
/// back to Previous uses the frame just before.
void test_reference_mode() {
    std::vector<espi::Frame> f;
    for (int k = 0; k < 6; ++k) f.push_back(make_frame(4, 4, [k](int x, int y) { return (x * 31 + y * 17 + k * 40) % 256; }));
    espi::Differencer d;
    espi::Frame in, out;
    in = f[0];
    d.push(in, out);
    d.capture_reference();
    in = f[1];
    CHECK(d.push(in, out));
    CHECK(d.mode() == espi::Mode::Reference);
    CHECK(is_difference(out, f[1], f[1]));  // the capture frame itself: all black
    in = f[2];
    CHECK(d.push(in, out));
    CHECK(is_difference(out, f[2], f[1]));
    in = f[3];
    CHECK(d.push(in, out));
    CHECK(is_difference(out, f[3], f[1]));
    d.use_previous();
    in = f[4];
    CHECK(d.push(in, out));
    CHECK(d.mode() == espi::Mode::Previous);
    CHECK(is_difference(out, f[4], f[3]));
    // A new capture replaces the old reference.
    d.capture_reference();
    in = f[5];
    d.push(in, out);
    in = f[0];
    CHECK(d.push(in, out));
    CHECK(is_difference(out, f[0], f[5]));
}

/// A size change drops the stored frames instead of reading out of bounds.
void test_size_change() {
    espi::Differencer d;
    espi::Frame in, out;
    in = make_frame(4, 4, [](int, int) { return 10; });
    d.push(in, out);
    in = make_frame(6, 2, [](int, int) { return 50; });
    CHECK(!d.push(in, out));  // Previous mode: nothing of this size to subtract yet

    d.capture_reference();
    in = make_frame(6, 2, [](int, int) { return 60; });
    CHECK(d.push(in, out));
    in = make_frame(3, 3, [](int, int) { return 90; });
    CHECK(d.push(in, out));  // Reference mode: the new-size frame becomes the reference
    CHECK(out.width == 3 && out.height == 3);
    CHECK(out.pixels == std::vector<std::uint8_t>(9, 0));
    CHECK(d.mode() == espi::Mode::Reference);
}

/// The gain table and the conversion into a padded texture.
void test_grey_map() {
    espi::GreyMap unity(1.0);
    CHECK(unity[0] == 0xFF000000u);
    CHECK(unity[128] == 0xFF808080u);
    CHECK(unity[255] == 0xFFFFFFFFu);

    espi::GreyMap four(4.0);
    CHECK(four[10] == 0xFF282828u);  // 40
    CHECK(four[63] == 0xFFFCFCFCu);  // 252
    CHECK(four[64] == 0xFFFFFFFFu);  // 256 saturates
    four.set_gain(1.5);
    CHECK(four[3] == 0xFF050505u);   // round(4.5) = 5 (half away from zero)
    CHECK(four.gain() == 1.5);

    const auto image = make_frame(3, 2, [](int x, int y) { return x + 10 * y; });
    const int pitch = 3 * 4 + 8;  // two spare words per row, like a padded texture
    std::vector<std::uint32_t> texture(static_cast<std::size_t>(pitch / 4) * 2, 0xDEADBEEFu);
    unity.convert(image, texture.data(), pitch);
    CHECK(texture[0] == 0xFF000000u);
    CHECK(texture[2] == 0xFF020202u);
    CHECK(texture[3] == 0xDEADBEEFu);  // padding untouched
    CHECK(texture[4] == 0xDEADBEEFu);
    CHECK(texture[5] == 0xFF0A0A0Au);  // row 1 starts after the padding
    CHECK(texture[7] == 0xFF0C0C0Cu);
}

/// Luma from packed and planar YUV, with row padding.
void test_extract_luma() {
    // YUY2, 4 x 2, pitch 12 (8 bytes of pixels + 4 padding): Y0 U Y1 V ...
    const std::uint8_t yuy2[] = {
        10, 1, 11, 2, 12, 3, 13, 4, 99, 99, 99, 99,
        20, 5, 21, 6, 22, 7, 23, 8, 99, 99, 99, 99,
    };
    espi::Frame out;
    espi::extract_luma(espi::LumaLayout::PackedEven, yuy2, 4, 2, 12, out);
    CHECK((out.pixels == std::vector<std::uint8_t>{10, 11, 12, 13, 20, 21, 22, 23}));

    // UYVY: luma in the odd bytes.
    espi::extract_luma(espi::LumaLayout::PackedOdd, yuy2, 4, 2, 12, out);
    CHECK((out.pixels == std::vector<std::uint8_t>{1, 2, 3, 4, 5, 6, 7, 8}));

    // NV12 luma plane, 3 x 2, pitch 4.
    const std::uint8_t nv12[] = {1, 2, 3, 0, 4, 5, 6, 0, 128, 128, 128, 128};
    espi::extract_luma(espi::LumaLayout::Planar, nv12, 3, 2, 4, out);
    CHECK((out.pixels == std::vector<std::uint8_t>{1, 2, 3, 4, 5, 6}));
    CHECK(out.width == 3 && out.height == 2);
}

/// Mean of |a - b| over a square window.
///
/// Args:
///   a: first frame.
///   b: second frame.
///   cx: window centre x.
///   cy: window centre y.
///   half: half the window size.
///
/// Returns:
///   the mean absolute difference.
double mean_difference(const espi::Frame& a, const espi::Frame& b, int cx, int cy, int half) {
    double sum = 0.0;
    int n = 0;
    for (int y = cy - half; y < cy + half; ++y)
        for (int x = cx - half; x < cx + half; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) * a.width + x;
            sum += std::abs(int(a.pixels[i]) - int(b.pixels[i]));
            ++n;
        }
    return sum / n;
}

/// The synthetic interferometer behaves like one: still frames differ only by
/// noise, deformation shows where the object moved, and the seed fixes the speckle.
void test_synthetic_source() {
    CHECK(espi::SyntheticSource::peak_fringes(0.0) == 0.0);
    CHECK(std::abs(espi::SyntheticSource::peak_fringes(6.0) - 6.0) < 1e-9);  // full bulge at half cycle

    espi::SyntheticSource a(200, 150, 30.0, 7);
    espi::SyntheticSource b(200, 150, 30.0, 7);
    espi::Frame fa, fb, still, bulged;
    a.render(1.0, fa);
    b.render(1.0, fb);
    CHECK(fa.pixels == fb.pixels);  // same seed, same image
    CHECK(fa.width == 200 && fa.height == 150);

    a.render(0.0, still);
    a.render(0.0, fb);  // same moment, new noise
    CHECK(mean_difference(still, fb, 100, 75, 20) < 2.0);

    a.render(3.0, bulged);  // three fringes at the centre
    const double centre = mean_difference(bulged, still, 90, 75, 15);  // dome centre is at 0.45 w
    // The corner farthest from the dome: deflection exp(-8) of the peak, so only noise remains.
    const double corner = mean_difference(bulged, still, 188, 140, 8);
    CHECK(centre > 15.0);
    CHECK(corner < 3.0);
}

/// The thread delivers difference images of the source's size and stops cleanly.
void test_pipeline_runs() {
    espi::SyntheticSource source(64, 48, 200.0, 3);
    std::string error;
    CHECK(source.open(error));
    espi::Pipeline pipeline(source);
    pipeline.start();
    espi::Frame out;
    espi::Mode mode = espi::Mode::Reference;
    int received = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (received < 3 && std::chrono::steady_clock::now() < deadline) {
        if (pipeline.take_latest(out, mode)) ++received;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    CHECK(received == 3);
    CHECK(out.width == 64 && out.height == 48);
    CHECK(mode == espi::Mode::Previous);

    pipeline.capture_reference();
    bool reference_seen = false;
    const auto deadline2 = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!reference_seen && std::chrono::steady_clock::now() < deadline2) {
        if (pipeline.take_latest(out, mode) && mode == espi::Mode::Reference) reference_seen = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    CHECK(reference_seen);
    pipeline.stop();
    CHECK(!pipeline.failed());
    CHECK(pipeline.stats().frames >= 4);
}

/// A source that fails on its second grab.
class FailingSource final : public espi::FrameSource {
public:
    bool open(std::string&) override { return true; }
    espi::GrabResult grab(espi::Frame& out, int, std::string& error) override {
        if (grabs_++ == 0) {
            out.resize(2, 2);
            return espi::GrabResult::Ok;
        }
        error = "cable unplugged";
        return espi::GrabResult::Error;
    }
    void close() override {}
    std::string name() const override { return "failing"; }

private:
    int grabs_ = 0;
};

/// A grab error stops the thread and is reported, not swallowed.
void test_pipeline_reports_failure() {
    FailingSource source;
    espi::Pipeline pipeline(source);
    pipeline.start();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!pipeline.failed() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(pipeline.failed());
    CHECK(pipeline.error() == "cable unplugged");
    pipeline.stop();
}

}  // namespace

int main() {
    test_absolute_difference_exhaustive();
    test_previous_mode();
    test_reference_mode();
    test_size_change();
    test_grey_map();
    test_extract_luma();
    test_synthetic_source();
    test_pipeline_runs();
    test_pipeline_reports_failure();
    if (failures == 0) std::cout << "all core tests passed\n";
    return failures;
}
