// FLIR Blackfly through the Spinnaker C++ API. Compiled only when CMake finds
// the Spinnaker SDK; otherwise blackfly_unavailable.cpp provides the stubs.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>

#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"

#include "source.h"

namespace espi {

using namespace Spinnaker;
using namespace Spinnaker::GenApi;

namespace {

// The project's camera defaults (see AGENTS.md): 8000 us, 0 dB, 30 fps.
constexpr double kExposureUs = 8000.0;
constexpr double kGainDb = 0.0;
constexpr double kFrameRate = 30.0;

/// Print one configuration line, so the console says what was changed on the camera.
///
/// Args:
///   text: the line.
void note(const std::string& text) {
    std::cout << "[blackfly] " << text << '\n';
}

/// Set an enumeration node to a named entry, if both are accessible.
///
/// Args:
///   map: node map holding the node.
///   node: node name, for example "ExposureAuto".
///   entry: entry name, for example "Off".
///
/// Returns:
///   true when the value was written.
bool set_enum(INodeMap& map, const char* node, const char* entry) {
    CEnumerationPtr ptr = map.GetNode(node);
    if (!IsReadable(ptr) || !IsWritable(ptr)) return false;
    CEnumEntryPtr value = ptr->GetEntryByName(entry);
    if (!IsReadable(value)) return false;
    ptr->SetIntValue(value->GetValue());
    return true;
}

/// Set a boolean node, if it is writable.
///
/// Args:
///   map: node map holding the node.
///   node: node name.
///   value: the value to write.
///
/// Returns:
///   true when the value was written.
bool set_bool(INodeMap& map, const char* node, bool value) {
    CBooleanPtr ptr = map.GetNode(node);
    if (!IsWritable(ptr)) return false;
    ptr->SetValue(value);
    return true;
}

/// Set a float node to a value clamped to the device's limits, if it is writable.
///
/// Args:
///   map: node map holding the node.
///   node: node name.
///   value: the requested value.
///   applied: receives the value actually written.
///
/// Returns:
///   true when a value was written.
bool set_float(INodeMap& map, const char* node, double value, double& applied) {
    CFloatPtr ptr = map.GetNode(node);
    if (!IsReadable(ptr) || !IsWritable(ptr)) return false;
    const double lo = ptr->GetMin();
    const double hi = ptr->GetMax();
    applied = value < lo ? lo : (value > hi ? hi : value);
    ptr->SetValue(applied);
    return true;
}

/// Read a string node.
///
/// Args:
///   map: node map holding the node.
///   node: node name.
///
/// Returns:
///   the value, or an empty string when it is not readable.
std::string read_string(INodeMap& map, const char* node) {
    CStringPtr ptr = map.GetNode(node);
    if (!IsReadable(ptr)) return {};
    return std::string(ptr->GetValue().c_str());
}

/// A Spinnaker camera streaming Mono8.
class BlackflySource final : public FrameSource {
public:
    /// Remember which camera to open.
    ///
    /// Args:
    ///   serial: serial number, or empty for the first camera.
    explicit BlackflySource(std::string serial) : serial_(std::move(serial)) {}

    /// Release the camera if the caller did not.
    ~BlackflySource() override { close(); }

    bool open(std::string& error) override {
        try {
            system_ = System::GetInstance();
            const LibraryVersion v = system_->GetLibraryVersion();
            note("Spinnaker " + std::to_string(v.major) + "." + std::to_string(v.minor) + "." +
                 std::to_string(v.type) + "." + std::to_string(v.build));
            cameras_ = system_->GetCameras();
            if (cameras_.GetSize() == 0) {
                error = "no FLIR camera found (check the USB3 cable, and close SpinView)";
                close();
                return false;
            }
            camera_ = serial_.empty() ? cameras_.GetByIndex(0) : cameras_.GetBySerial(serial_);
            if (!camera_.IsValid()) {
                error = "no FLIR camera with serial number " + serial_;
                close();
                return false;
            }

            INodeMap& device = camera_->GetTLDeviceNodeMap();
            name_ = read_string(device, "DeviceModelName");
            const std::string sn = read_string(device, "DeviceSerialNumber");
            if (name_.empty()) name_ = "FLIR camera";
            if (!sn.empty()) name_ += " (SN " + sn + ")";
            note("opening " + name_);

            camera_->Init();
            initialized_ = true;
            configure();
            camera_->BeginAcquisition();
            streaming_ = true;
            return true;
        } catch (const Spinnaker::Exception& e) {
            error = std::string(e.what());
            close();
            return false;
        }
    }

    GrabResult grab(Frame& out, int timeout_ms, std::string& error) override {
        try {
            ImagePtr image = camera_->GetNextImage(static_cast<std::uint64_t>(timeout_ms));
            // Every buffer from GetNextImage goes back to the SDK, whatever happens below.
            struct Release {
                ImagePtr& image;
                ~Release() {
                    try {
                        if (image.IsValid()) image->Release();
                    } catch (...) {
                    }
                }
            } release{image};

            if (image->IsIncomplete()) {
                if (incomplete_++ % 100 == 0) {
                    note(std::string("incomplete image (") +
                         Image::GetImageStatusDescription(image->GetImageStatus()) + "), skipped");
                }
                return GrabResult::Skipped;
            }
            if (image->GetPixelFormat() != PixelFormat_Mono8) {
                error = "the camera is not sending Mono8 (PixelFormat could not be set)";
                return GrabResult::Error;
            }
            const int width = static_cast<int>(image->GetWidth());
            const int height = static_cast<int>(image->GetHeight());
            std::size_t stride = image->GetStride();
            if (stride < static_cast<std::size_t>(width)) stride = static_cast<std::size_t>(width);
            const auto* src = static_cast<const std::uint8_t*>(image->GetData());
            // Copy out of the SDK buffer: it is reused as soon as it is released.
            out.resize(width, height);
            for (int y = 0; y < height; ++y) {
                std::memcpy(out.pixels.data() + static_cast<std::size_t>(y) * width,
                            src + static_cast<std::size_t>(y) * stride, static_cast<std::size_t>(width));
            }
            return GrabResult::Ok;
        } catch (const Spinnaker::Exception& e) {
            if (e.GetError() == SPINNAKER_ERR_TIMEOUT) return GrabResult::Timeout;
            error = std::string(e.what());
            return GrabResult::Error;
        }
    }

    void close() override {
        // Teardown order matters: stop the stream, de-initialise, drop every
        // camera reference, clear the list, and only then release the system.
        if (streaming_) {
            try {
                camera_->EndAcquisition();
            } catch (const Spinnaker::Exception& e) {
                note(std::string("EndAcquisition: ") + e.what());
            }
            streaming_ = false;
        }
        if (initialized_) {
            try {
                camera_->DeInit();
            } catch (const Spinnaker::Exception& e) {
                note(std::string("DeInit: ") + e.what());
            }
            initialized_ = false;
        }
        camera_ = nullptr;
        cameras_.Clear();
        if (system_.IsValid()) {
            try {
                system_->ReleaseInstance();
            } catch (const Spinnaker::Exception& e) {
                note(std::string("ReleaseInstance: ") + e.what());
            }
            system_ = nullptr;
        }
    }

    std::string name() const override { return name_.empty() ? "FLIR camera" : name_; }

private:
    /// Put the camera in a known state for subtraction ESPI. Every node is
    /// optional: a missing or read-only one is reported and left alone.
    void configure() {
        INodeMap& map = camera_->GetNodeMap();
        double applied = 0.0;

        if (!set_enum(map, "AcquisitionMode", "Continuous")) note("AcquisitionMode not writable, left as is");
        if (set_enum(map, "PixelFormat", "Mono8")) {
            note("PixelFormat = Mono8");
        } else {
            note("PixelFormat not writable; frames must already be Mono8");
        }

        // Auto exposure or auto gain would change the brightness between the
        // two frames being subtracted and flood the difference, so both are
        // turned off and fixed at the project defaults.
        if (set_enum(map, "ExposureAuto", "Off")) note("ExposureAuto = Off");
        if (set_float(map, "ExposureTime", kExposureUs, applied)) {
            note("ExposureTime = " + std::to_string(applied) + " us");
        } else {
            note("ExposureTime not writable, left as is");
        }
        if (set_enum(map, "GainAuto", "Off")) note("GainAuto = Off");
        if (set_float(map, "Gain", kGainDb, applied)) {
            note("Gain = " + std::to_string(applied) + " dB");
        } else {
            note("Gain not writable, left as is");
        }

        if (set_bool(map, "AcquisitionFrameRateEnable", true) &&
            set_float(map, "AcquisitionFrameRate", kFrameRate, applied)) {
            note("AcquisitionFrameRate = " + std::to_string(applied) + " fps");
        } else {
            note("AcquisitionFrameRate not writable, the camera picks its rate");
        }

        // Live view: when processing falls behind, show the newest frame
        // instead of building latency through a queue of old ones.
        INodeMap& stream = camera_->GetTLStreamNodeMap();
        if (set_enum(stream, "StreamBufferHandlingMode", "NewestOnly")) {
            note("StreamBufferHandlingMode = NewestOnly");
        }
    }

    std::string serial_;
    std::string name_;
    SystemPtr system_;
    CameraList cameras_;
    CameraPtr camera_;
    bool initialized_ = false;
    bool streaming_ = false;
    std::uint64_t incomplete_ = 0;
};

}  // namespace

bool blackfly_supported() {
    return true;
}

std::unique_ptr<FrameSource> make_blackfly_source(const std::string& serial) {
    return std::make_unique<BlackflySource>(serial);
}

std::vector<std::string> list_blackfly_cameras(std::string& error) {
    std::vector<std::string> lines;
    SystemPtr system;
    CameraList cameras;
    try {
        system = System::GetInstance();
        cameras = system->GetCameras();
        for (unsigned int i = 0; i < cameras.GetSize(); ++i) {
            CameraPtr camera = cameras.GetByIndex(i);
            INodeMap& device = camera->GetTLDeviceNodeMap();
            lines.push_back(read_string(device, "DeviceModelName") + " (SN " +
                            read_string(device, "DeviceSerialNumber") + ")");
        }
    } catch (const Spinnaker::Exception& e) {
        error = std::string(e.what());
    }
    cameras.Clear();
    if (system.IsValid()) {
        try {
            system->ReleaseInstance();
        } catch (const Spinnaker::Exception&) {
        }
    }
    return lines;
}

}  // namespace espi
