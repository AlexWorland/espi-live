// espi_diff: open a camera, subtract frames, show the difference live.
//
//   espi_diff                 FLIR Blackfly (first found)
//   espi_diff --serial SN     a particular FLIR camera
//   espi_diff --device N      video capture device N (see --list)
//   espi_diff --demo          synthetic interferometer, no hardware
//   espi_diff --list          list cameras and capture devices
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "difference.h"
#include "display.h"
#include "frame.h"
#include "pipeline.h"
#include "source.h"

namespace {

constexpr double kMinGain = 1.0;
constexpr double kMaxGain = 128.0;

/// Command-line choices.
struct Options {
    bool demo = false;
    bool list = false;
    int device = -1;            ///< capture device index, or -1
    std::string serial;         ///< FLIR serial number, or empty
    double gain = 1.0;          ///< initial display gain
    bool reference = false;     ///< start in Reference mode
    long long max_frames = -1;  ///< quit after this many displayed images, or -1
    std::string screenshot;     ///< BMP to save the final view to (with max_frames), or empty
};

/// Print usage to stdout.
void print_usage() {
    std::cout <<
        "espi_diff: live subtraction ESPI (|frame - previous| or |frame - reference|)\n"
        "\n"
        "  espi_diff [--serial SN]   FLIR camera (first found, or the one with this serial)\n"
        "  espi_diff --device N      video capture device N, e.g. a USB-Live2 stick\n"
        "  espi_diff --demo          synthetic interferometer (no hardware)\n"
        "  espi_diff --list          list FLIR cameras and capture devices\n"
        "\n"
        "  --gain G                  initial display gain, 1 to 128 (default 1)\n"
        "  --reference               start in Reference mode (the first frame is the reference)\n"
        "  --frames N                quit after N displayed images\n"
        "  --screenshot FILE.bmp     with --frames: save the final view as a BMP\n"
        "\n"
        "Keys: R or Space capture a reference, P previous-frame mode, Up/Down double or\n"
        "      halve the gain, F fullscreen, H hide or show the help, Esc or Q quit.\n";
}

/// Parse the command line.
///
/// Args:
///   argc: argument count.
///   argv: arguments.
///   options: receives the choices.
///
/// Returns:
///   0 to run, 1 when help was printed, 2 on a usage error.
int parse_options(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << arg << " needs " << what << "\n";
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--demo") {
            options.demo = true;
        } else if (arg == "--list") {
            options.list = true;
        } else if (arg == "--device") {
            const char* v = value("a device number");
            if (!v) return 2;
            options.device = std::atoi(v);
        } else if (arg == "--serial") {
            const char* v = value("a serial number");
            if (!v) return 2;
            options.serial = v;
        } else if (arg == "--gain") {
            const char* v = value("a number");
            if (!v) return 2;
            options.gain = std::atof(v);
            if (options.gain < kMinGain || options.gain > kMaxGain) {
                std::cerr << "--gain must be between 1 and 128\n";
                return 2;
            }
        } else if (arg == "--frames") {
            const char* v = value("a count");
            if (!v) return 2;
            options.max_frames = std::atoll(v);
        } else if (arg == "--reference") {
            options.reference = true;
        } else if (arg == "--screenshot") {
            const char* v = value("a file name");
            if (!v) return 2;
            options.screenshot = v;
        } else if (arg == "-h" || arg == "--help") {
            print_usage();
            return 1;
        } else {
            std::cerr << "unknown option " << arg << " (try --help)\n";
            return 2;
        }
    }
    const int chosen = (options.demo ? 1 : 0) + (options.device >= 0 ? 1 : 0) + (options.serial.empty() ? 0 : 1);
    if (chosen > 1) {
        std::cerr << "choose one of --demo, --device and --serial\n";
        return 2;
    }
    if (!options.screenshot.empty() && options.max_frames < 0) {
        std::cerr << "--screenshot needs --frames N\n";
        return 2;
    }
    return 0;
}

/// Print every camera and capture device.
///
/// Returns:
///   the process exit code.
int list_everything() {
    std::string error;
    std::cout << "FLIR cameras:\n";
    const auto flir = espi::list_blackfly_cameras(error);
    if (!error.empty()) std::cout << "  (" << error << ")\n";
    if (flir.empty() && error.empty()) std::cout << "  none\n";
    for (std::size_t i = 0; i < flir.size(); ++i) std::cout << "  " << flir[i] << "\n";

    std::cout << "Capture devices:\n";
    if (!SDL_Init(SDL_INIT_CAMERA)) {
        std::cout << "  (SDL camera support unavailable: " << SDL_GetError() << ")\n";
        return 1;
    }
    error.clear();
    const auto devices = espi::list_capture_devices(error);
    if (!error.empty()) std::cout << "  (" << error << ")\n";
    if (devices.empty() && error.empty()) std::cout << "  none\n";
    for (const auto& line : devices) std::cout << "  " << line << "\n";
    SDL_Quit();
    return 0;
}

/// Text with a translucent dark box behind it, in output pixels.
///
/// Args:
///   renderer: the renderer.
///   x: left edge of the text.
///   y: top edge of the text.
///   scale: text magnification (the built-in font is 8 x 8).
///   text: ASCII text.
///   color: text colour.
void draw_label(SDL_Renderer* renderer, float x, float y, float scale, const std::string& text,
                SDL_Color color) {
    const float glyph = SDL_DEBUG_TEXT_FONT_CHARACTER_SIZE * scale;
    const SDL_FRect box{x - 4 * scale, y - 3 * scale, glyph * static_cast<float>(text.size()) + 8 * scale,
                        glyph + 6 * scale};
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 170);
    SDL_RenderFillRect(renderer, &box);
    SDL_SetRenderScale(renderer, scale, scale);
    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, 255);
    SDL_RenderDebugText(renderer, x / scale, y / scale, text.c_str());
    SDL_SetRenderScale(renderer, 1.0f, 1.0f);
}

/// The largest rectangle with the image's aspect ratio centred in the output.
///
/// Args:
///   image_w: image width.
///   image_h: image height.
///   out_w: output width in pixels.
///   out_h: output height in pixels.
///
/// Returns:
///   the destination rectangle.
SDL_FRect fit(int image_w, int image_h, int out_w, int out_h) {
    const float scale_w = static_cast<float>(out_w) / image_w;
    const float scale_h = static_cast<float>(out_h) / image_h;
    const float scale = scale_w < scale_h ? scale_w : scale_h;
    const float w = image_w * scale;
    const float h = image_h * scale;
    return SDL_FRect{(out_w - w) / 2.0f, (out_h - h) / 2.0f, w, h};
}

/// Size the window to show an image as large as fits on its display.
///
/// Args:
///   window: the window.
///   image_w: image width.
///   image_h: image height.
void size_window_for(SDL_Window* window, int image_w, int image_h) {
    SDL_Rect usable{0, 0, 1280, 800};
    SDL_GetDisplayUsableBounds(SDL_GetDisplayForWindow(window), &usable);
    const SDL_FRect r = fit(image_w, image_h, usable.w * 85 / 100, usable.h * 85 / 100);
    SDL_SetWindowSize(window, static_cast<int>(r.w), static_cast<int>(r.h));
    SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
}

/// Open the chosen source.
///
/// Args:
///   options: command-line choices.
///   error: receives a message on failure.
///
/// Returns:
///   an opened source, or nullptr.
std::unique_ptr<espi::FrameSource> open_source(const Options& options, std::string& error) {
    std::unique_ptr<espi::FrameSource> source;
    if (options.demo) {
        source = espi::make_synthetic_source();
    } else if (options.device >= 0) {
        source = espi::make_capture_source(options.device);
    } else {
        source = espi::make_blackfly_source(options.serial);
        if (!source) {
            error = "this build has no FLIR support (no Spinnaker SDK at build time); "
                    "use --device N (see --list) or --demo";
            return nullptr;
        }
    }
    if (!source->open(error)) return nullptr;
    return source;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    const int parsed = parse_options(argc, argv, options);
    if (parsed == 1) return 0;
    if (parsed != 0) return parsed;
    if (options.list) return list_everything();

    const SDL_InitFlags flags = SDL_INIT_VIDEO | (options.device >= 0 ? SDL_INIT_CAMERA : 0);
    if (!SDL_Init(flags)) {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << "\n";
        return 1;
    }

    std::string error;
    std::unique_ptr<espi::FrameSource> source = open_source(options, error);
    if (!source) {
        std::cerr << "error: " << error << "\n";
        SDL_Quit();
        return 1;
    }
    const bool demo = source->synthetic();
    const std::string title = std::string("ESPI difference - ") + source->name();
    std::cout << "source: " << source->name() << "\n";

    SDL_Window* window = SDL_CreateWindow(title.c_str(), 1280, 860,
                                          SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!window) {
        std::cerr << "cannot open a window: " << SDL_GetError() << "\n";
        source->close();
        SDL_Quit();
        return 1;
    }
    // The best GPU renderer (Direct3D, Metal, OpenGL...), else plain software
    // drawing, for example in a remote-desktop session without acceleration.
    SDL_Renderer* renderer = SDL_CreateRenderer(window, nullptr);
    if (!renderer) renderer = SDL_CreateRenderer(window, SDL_SOFTWARE_RENDERER);
    if (!renderer) {
        std::cerr << "cannot draw in the window: " << SDL_GetError() << "\n";
        source->close();
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    SDL_SetRenderVSync(renderer, 1);
    std::cout << "renderer: " << SDL_GetRendererName(renderer) << "\n";

    espi::Pipeline pipeline(*source);
    if (options.reference) pipeline.capture_reference();
    pipeline.start();

    espi::GreyMap grey(options.gain);
    espi::Frame shown;
    espi::Mode shown_mode = espi::Mode::Previous;
    SDL_Texture* texture = nullptr;
    int texture_w = 0;
    int texture_h = 0;
    bool show_help = true;
    bool fullscreen = false;
    bool upload = false;  // shown changed (new image or new gain): refill the texture
    bool redraw = true;
    int exit_code = 0;
    long long displayed = 0;

    using clock = std::chrono::steady_clock;
    auto rate_since = clock::now();
    std::uint64_t frames_since = 0;
    long long displayed_since = 0;
    double camera_fps = 0.0;
    double display_fps = 0.0;

    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_EVENT_QUIT:
                    running = false;
                    break;
                case SDL_EVENT_WINDOW_RESIZED:
                case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                case SDL_EVENT_WINDOW_EXPOSED:
                    redraw = true;
                    break;
                case SDL_EVENT_CAMERA_DEVICE_DENIED:
                    std::cerr << "error: camera access was denied\n";
                    exit_code = 1;
                    running = false;
                    break;
                case SDL_EVENT_CAMERA_DEVICE_REMOVED:
                    if (event.cdevice.which == source->device_id()) {
                        std::cerr << "error: the capture device was disconnected\n";
                        exit_code = 1;
                        running = false;
                    }
                    break;
                case SDL_EVENT_KEY_DOWN:
                    switch (event.key.key) {
                        case SDLK_ESCAPE:
                        case SDLK_Q:
                            running = false;
                            break;
                        case SDLK_R:
                        case SDLK_SPACE:
                            pipeline.capture_reference();
                            break;
                        case SDLK_P:
                            pipeline.use_previous();
                            break;
                        case SDLK_UP:
                        case SDLK_EQUALS:
                        case SDLK_KP_PLUS:
                            grey.set_gain(grey.gain() * 2.0 > kMaxGain ? kMaxGain : grey.gain() * 2.0);
                            upload = redraw = true;
                            break;
                        case SDLK_DOWN:
                        case SDLK_MINUS:
                        case SDLK_KP_MINUS:
                            grey.set_gain(grey.gain() / 2.0 < kMinGain ? kMinGain : grey.gain() / 2.0);
                            upload = redraw = true;
                            break;
                        case SDLK_F:
                            fullscreen = !fullscreen;
                            SDL_SetWindowFullscreen(window, fullscreen);
                            break;
                        case SDLK_H:
                            show_help = !show_help;
                            redraw = true;
                            break;
                        default:
                            break;
                    }
                    break;
                default:
                    break;
            }
        }
        if (pipeline.failed()) {
            std::cerr << "error: " << pipeline.error() << "\n";
            exit_code = 1;
            break;
        }

        if (pipeline.take_latest(shown, shown_mode)) {
            if (texture_w != shown.width || texture_h != shown.height) {
                if (texture) SDL_DestroyTexture(texture);
                texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_XRGB8888,
                                            SDL_TEXTUREACCESS_STREAMING, shown.width, shown.height);
                if (!texture) {
                    std::cerr << "error: cannot create a " << shown.width << "x" << shown.height
                              << " texture: " << SDL_GetError() << "\n";
                    exit_code = 1;
                    break;
                }
                if (texture_w == 0) {
                    std::cout << "frames: " << shown.width << "x" << shown.height << "\n";
                    size_window_for(window, shown.width, shown.height);
                }
                texture_w = shown.width;
                texture_h = shown.height;
            }
            upload = redraw = true;
            ++displayed;
        }

        if (upload && texture) {
            void* pixels = nullptr;
            int pitch = 0;
            if (SDL_LockTexture(texture, nullptr, &pixels, &pitch)) {
                grey.convert(shown, pixels, pitch);
                SDL_UnlockTexture(texture);
            }
            upload = false;
        }

        const auto now = clock::now();
        const double elapsed = std::chrono::duration<double>(now - rate_since).count();
        if (elapsed >= 0.5) {
            const auto stats = pipeline.stats();
            camera_fps = (stats.frames - frames_since) / elapsed;
            display_fps = (displayed - displayed_since) / elapsed;
            frames_since = stats.frames;
            displayed_since = displayed;
            rate_since = now;
            redraw = true;
        }

        if (redraw) {
            int out_w = 0;
            int out_h = 0;
            SDL_GetRenderOutputSize(renderer, &out_w, &out_h);
            SDL_SetRenderDrawColor(renderer, 16, 16, 16, 255);
            SDL_RenderClear(renderer);
            if (texture) {
                const SDL_FRect dst = fit(texture_w, texture_h, out_w, out_h);
                // Magnified: nearest keeps each speckle a crisp block. Reduced: linear.
                SDL_SetTextureScaleMode(texture, dst.w >= texture_w ? SDL_SCALEMODE_NEAREST : SDL_SCALEMODE_LINEAR);
                SDL_RenderTexture(renderer, texture, nullptr, &dst);
            }

            float scale = SDL_GetWindowDisplayScale(window) * 1.5f;
            if (scale < 1.0f) scale = 1.0f;
            const float line = SDL_DEBUG_TEXT_FONT_CHARACTER_SIZE * scale + 10 * scale;
            float y = 8 * scale;
            if (demo) {
                draw_label(renderer, 8 * scale, y, scale * 1.5f, "DEMO: SYNTHETIC DATA, NOT A CAMERA",
                           SDL_Color{255, 70, 70, 255});
                y += line * 1.5f;
            }
            char status[160];
            if (texture) {
                SDL_snprintf(status, sizeof status, "%s | gain x%g | camera %.1f fps | view %.1f fps",
                             shown_mode == espi::Mode::Reference ? "|frame - reference|" : "|frame - previous|",
                             grey.gain(), camera_fps, display_fps);
            } else {
                SDL_snprintf(status, sizeof status, "waiting for frames... (camera %.1f fps)", camera_fps);
            }
            draw_label(renderer, 8 * scale, y, scale, status, SDL_Color{255, 230, 120, 255});
            if (show_help) {
                draw_label(renderer, 8 * scale, out_h - line, scale,
                           "R reference  P previous  Up/Down gain  F fullscreen  H help  Esc quit",
                           SDL_Color{200, 200, 200, 255});
            }
            const bool last = options.max_frames >= 0 && displayed >= options.max_frames;
            if (last && !options.screenshot.empty()) {
                // Read back before presenting: the back buffer is undefined afterwards.
                SDL_Surface* shot = SDL_RenderReadPixels(renderer, nullptr);
                if (shot && SDL_SaveBMP(shot, options.screenshot.c_str())) {
                    std::cout << "saved " << options.screenshot << "\n";
                } else {
                    std::cerr << "screenshot failed: " << SDL_GetError() << "\n";
                    exit_code = 1;
                }
                SDL_DestroySurface(shot);
            }
            SDL_RenderPresent(renderer);
            redraw = false;
        } else {
            SDL_Delay(1);
        }

        if (options.max_frames >= 0 && displayed >= options.max_frames) running = false;
    }

    pipeline.stop();
    const auto stats = pipeline.stats();
    std::cout << "grabbed " << stats.frames << " frames, displayed " << displayed << ", skipped "
              << stats.skipped << "\n";
    source->close();
    if (texture) SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return exit_code;
}
