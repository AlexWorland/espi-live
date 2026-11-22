// The processing thread: grab -> difference -> hand the newest result to the display.
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#include "difference.h"
#include "frame.h"
#include "source.h"

namespace espi {

/// Runs a FrameSource and a Differencer on their own thread.
///
/// Every camera frame is differenced in order, so Previous mode always
/// subtracts consecutive frames, however slowly the display runs. The display
/// takes only the newest difference image; older ones are overwritten.
class Pipeline {
public:
    /// Counters since start().
    struct Stats {
        std::uint64_t frames = 0;   ///< frames grabbed
        std::uint64_t skipped = 0;  ///< unusable frames (incomplete transfers)
    };

    /// Bind to an opened source. Nothing runs until start().
    ///
    /// Args:
    ///   source: an opened source; must outlive the pipeline.
    explicit Pipeline(FrameSource& source);

    /// Stop the thread if it is still running.
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    /// Start grabbing on a new thread.
    void start();

    /// Ask the thread to finish and wait for it (at most one grab timeout).
    void stop();

    /// Switch to Previous mode from the next frame.
    void use_previous();

    /// Capture the next frame as the reference and switch to Reference mode.
    void capture_reference();

    /// Take the newest difference image if one arrived since the last call.
    ///
    /// Args:
    ///   out: receives the image (its old buffer is recycled by the thread).
    ///   mode: receives the image's mode.
    ///
    /// Returns:
    ///   true when `out` and `mode` were updated.
    bool take_latest(Frame& out, Mode& mode);

    /// Snapshot of the counters.
    ///
    /// Returns:
    ///   frames grabbed and skipped so far.
    Stats stats() const;

    /// Whether the thread stopped because the source failed.
    ///
    /// Returns:
    ///   true after a grab error.
    bool failed() const { return failed_.load(); }

    /// The failure message.
    ///
    /// Returns:
    ///   what went wrong, or an empty string.
    std::string error() const;

private:
    /// Thread body: grab and difference until stop() or an error.
    void run();

    FrameSource& source_;
    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> failed_{false};
    std::atomic<bool> want_previous_{false};
    std::atomic<bool> want_reference_{false};
    std::atomic<std::uint64_t> frames_{0};
    std::atomic<std::uint64_t> skipped_{0};

    mutable std::mutex mutex_;  // guards the fields below
    Frame ready_;
    Mode ready_mode_ = Mode::Previous;
    bool fresh_ = false;
    std::string error_;
};

}  // namespace espi
