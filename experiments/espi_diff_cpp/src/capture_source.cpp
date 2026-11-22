// Video capture devices (a USB capture stick such as the Hauppauge USB-Live2,
// or any UVC camera) through SDL3's camera API: Media Foundation on Windows,
// AVFoundation on macOS, V4L2 on Linux.
#include <SDL3/SDL.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>

#include "luma.h"
#include "source.h"

namespace espi {

namespace {

/// Where the luma is in an SDL pixel format, when the format carries it directly.
///
/// Args:
///   format: the SDL pixel format.
///   layout: receives the layout.
///
/// Returns:
///   true for the YUV formats whose luma can be copied out as is.
bool luma_layout(SDL_PixelFormat format, LumaLayout& layout) {
    switch (format) {
        case SDL_PIXELFORMAT_YUY2:
        case SDL_PIXELFORMAT_YVYU:
            layout = LumaLayout::PackedEven;
            return true;
        case SDL_PIXELFORMAT_UYVY:
            layout = LumaLayout::PackedOdd;
            return true;
        case SDL_PIXELFORMAT_NV12:
        case SDL_PIXELFORMAT_NV21:
        case SDL_PIXELFORMAT_IYUV:
        case SDL_PIXELFORMAT_YV12:
            layout = LumaLayout::Planar;
            return true;
        default:
            return false;
    }
}

/// Frames per second of a spec.
///
/// Args:
///   spec: the format.
///
/// Returns:
///   the rate, or 0 when the device does not say.
double spec_fps(const SDL_CameraSpec& spec) {
    return spec.framerate_denominator > 0
               ? static_cast<double>(spec.framerate_numerator) / spec.framerate_denominator
               : 0.0;
}

/// One line describing a format, for example "YUY2 720x480 @ 29.97 fps".
///
/// Args:
///   spec: the format.
///
/// Returns:
///   the description.
std::string describe(const SDL_CameraSpec& spec) {
    char rate[32];
    SDL_snprintf(rate, sizeof rate, "%.2f", spec_fps(spec));
    const char* format = SDL_GetPixelFormatName(spec.format);
    return std::string(format ? format : "?") + " " + std::to_string(spec.width) + "x" +
           std::to_string(spec.height) + " @ " + rate + " fps";
}

/// Choose the device format to open: one that carries luma directly (so no
/// colour conversion touches the speckle), at a video rate if there is one,
/// then the largest frame, then the highest rate.
///
/// Args:
///   id: the device.
///   chosen: receives the format.
///
/// Returns:
///   true when a format was chosen; false to let SDL pick (the device lists none).
bool choose_spec(SDL_CameraID id, SDL_CameraSpec& chosen) {
    int count = 0;
    SDL_CameraSpec** specs = SDL_GetCameraSupportedFormats(id, &count);
    if (!specs) return false;
    bool found = false;
    bool best_video_rate = false;
    long long best_area = -1;
    double best_fps = -1.0;
    for (int i = 0; i < count; ++i) {
        const SDL_CameraSpec& spec = *specs[i];
        LumaLayout layout;
        if (!luma_layout(spec.format, layout)) continue;
        const bool video_rate = spec_fps(spec) >= 24.0;
        const long long area = static_cast<long long>(spec.width) * spec.height;
        const double fps = spec_fps(spec);
        const bool better = !found || (video_rate != best_video_rate ? video_rate
                                       : area != best_area           ? area > best_area
                                                                     : fps > best_fps);
        if (better) {
            chosen = spec;
            found = true;
            best_video_rate = video_rate;
            best_area = area;
            best_fps = fps;
        }
    }
    if (!found && count > 0) {
        // No luma format listed: take the first size and let SDL convert to NV12.
        chosen = *specs[0];
        chosen.format = SDL_PIXELFORMAT_NV12;
        found = true;
    }
    SDL_free(specs);
    return found;
}

/// A capture device delivering YUV video, of which only the luma is used.
class CaptureSource final : public FrameSource {
public:
    /// Remember which device to open.
    ///
    /// Args:
    ///   index: position in SDL's device list.
    explicit CaptureSource(int index) : index_(index) {}

    /// Close the device if the caller did not.
    ~CaptureSource() override { close(); }

    bool open(std::string& error) override {
        int count = 0;
        SDL_CameraID* ids = SDL_GetCameras(&count);
        if (!ids) {
            error = std::string("listing capture devices failed: ") + SDL_GetError();
            return false;
        }
        if (index_ < 0 || index_ >= count) {
            error = "no capture device " + std::to_string(index_) + " (" + std::to_string(count) +
                    " found; see --list)";
            SDL_free(ids);
            return false;
        }
        id_ = ids[index_];
        SDL_free(ids);
        const char* device_name = SDL_GetCameraName(id_);
        name_ = device_name ? device_name : "capture device";

        SDL_CameraSpec spec{};
        const bool have_spec = choose_spec(id_, spec);
        camera_ = SDL_OpenCamera(id_, have_spec ? &spec : nullptr);
        if (!camera_) {
            error = "opening " + name_ + " failed: " + SDL_GetError() +
                    " (close other programs using it, such as WinTV)";
            return false;
        }
        SDL_CameraSpec actual{};
        if (SDL_GetCameraFormat(camera_, &actual)) {
            std::cout << "[capture] " << name_ << ": " << describe(actual) << '\n';
            if (actual.format == static_cast<SDL_PixelFormat>(SDL_DEFINE_PIXELFOURCC('M', 'J', 'P', 'G'))) {
                std::cout << "[capture] warning: compressed video (MJPG) smears speckle\n";
            }
        }
        return true;
    }

    GrabResult grab(Frame& out, int timeout_ms, std::string& error) override {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        for (;;) {
            // -1 denied, 0 pending, 1 approved. Compared as int: SDL 3.2 returns
            // int here, SDL 3.4 an enum with those values.
            const int permission = static_cast<int>(SDL_GetCameraPermissionState(camera_));
            if (permission < 0) {
                error = "camera access was denied (allow it in the system privacy settings)";
                return GrabResult::Error;
            }
            if (permission == 0 && !told_pending_) {
                std::cout << "[capture] waiting for permission to use the camera\n";
                told_pending_ = true;
            }
            SDL_Surface* surface = SDL_AcquireCameraFrame(camera_, nullptr);
            if (surface) {
                const bool ok = copy_luma(surface, out, error);
                SDL_ReleaseCameraFrame(camera_, surface);
                return ok ? GrabResult::Ok : GrabResult::Error;
            }
            // SDL's camera API is non-blocking; poll at 1 ms (well under a frame period).
            if (std::chrono::steady_clock::now() >= deadline) return GrabResult::Timeout;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    void close() override {
        if (camera_) {
            SDL_CloseCamera(camera_);
            camera_ = nullptr;
        }
    }

    std::string name() const override { return name_; }

    std::uint32_t device_id() const override { return id_; }

private:
    /// Copy the luma of a delivered frame, converting first only when the
    /// format has no luma plane of its own.
    ///
    /// Args:
    ///   surface: the frame from SDL_AcquireCameraFrame.
    ///   out: receives the greyscale frame.
    ///   error: receives a message on failure.
    ///
    /// Returns:
    ///   true on success.
    bool copy_luma(SDL_Surface* surface, Frame& out, std::string& error) {
        LumaLayout layout;
        if (luma_layout(surface->format, layout)) {
            extract_luma(layout, static_cast<const std::uint8_t*>(surface->pixels), surface->w,
                         surface->h, surface->pitch, out);
            return true;
        }
        SDL_Surface* converted = SDL_ConvertSurface(surface, SDL_PIXELFORMAT_NV12);
        if (!converted) {
            error = std::string("cannot read this capture format: ") + SDL_GetError();
            return false;
        }
        extract_luma(LumaLayout::Planar, static_cast<const std::uint8_t*>(converted->pixels),
                     converted->w, converted->h, converted->pitch, out);
        SDL_DestroySurface(converted);
        return true;
    }

    int index_;
    SDL_CameraID id_ = 0;
    SDL_Camera* camera_ = nullptr;
    std::string name_;
    bool told_pending_ = false;
};

}  // namespace

std::unique_ptr<FrameSource> make_capture_source(int index) {
    return std::make_unique<CaptureSource>(index);
}

std::vector<std::string> list_capture_devices(std::string& error) {
    std::vector<std::string> lines;
    int count = 0;
    SDL_CameraID* ids = SDL_GetCameras(&count);
    if (!ids) {
        error = std::string("listing capture devices failed: ") + SDL_GetError();
        return lines;
    }
    for (int i = 0; i < count; ++i) {
        const char* device_name = SDL_GetCameraName(ids[i]);
        std::string line = "[" + std::to_string(i) + "] " + (device_name ? device_name : "?");
        const SDL_CameraPosition position = SDL_GetCameraPosition(ids[i]);
        if (position != SDL_CAMERA_POSITION_UNKNOWN) line += " (built-in)";
        SDL_CameraSpec spec{};
        if (choose_spec(ids[i], spec)) line += ": would open " + describe(spec);
        lines.push_back(line);
    }
    SDL_free(ids);
    return lines;
}

}  // namespace espi
