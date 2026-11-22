#include "pipeline.h"

#include <exception>
#include <utility>

namespace espi {

namespace {
constexpr int kGrabTimeoutMs = 100;  // bounds how long stop() waits
}

Pipeline::Pipeline(FrameSource& source) : source_(source) {}

Pipeline::~Pipeline() {
    stop();
}

void Pipeline::start() {
    if (thread_.joinable()) return;
    stop_ = false;
    thread_ = std::thread(&Pipeline::run, this);
}

void Pipeline::stop() {
    stop_ = true;
    if (thread_.joinable()) thread_.join();
}

void Pipeline::use_previous() {
    want_reference_ = false;
    want_previous_ = true;
}

void Pipeline::capture_reference() {
    want_previous_ = false;
    want_reference_ = true;
}

bool Pipeline::take_latest(Frame& out, Mode& mode) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!fresh_) return false;
    std::swap(out, ready_);
    mode = ready_mode_;
    fresh_ = false;
    return true;
}

Pipeline::Stats Pipeline::stats() const {
    return Stats{frames_.load(), skipped_.load()};
}

std::string Pipeline::error() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_;
}

void Pipeline::run() {
    Differencer differ;
    Frame grabbed;  // the grab buffer; after push() it holds a recycled one
    Frame work;     // the difference being built; swapped with ready_ when done
    std::string message;
    try {
        while (!stop_) {
            if (want_previous_.exchange(false)) differ.use_previous();
            if (want_reference_.exchange(false)) differ.capture_reference();

            const GrabResult result = source_.grab(grabbed, kGrabTimeoutMs, message);
            if (result == GrabResult::Timeout) continue;
            if (result == GrabResult::Skipped) {
                ++skipped_;
                continue;
            }
            if (result == GrabResult::Error) break;
            ++frames_;

            if (differ.push(grabbed, work)) {
                std::lock_guard<std::mutex> lock(mutex_);
                std::swap(work, ready_);
                ready_mode_ = differ.mode();
                fresh_ = true;
            }
        }
    } catch (const std::exception& e) {
        message = e.what();
    }
    if (!stop_) {
        std::lock_guard<std::mutex> lock(mutex_);
        error_ = message.empty() ? "the frame source stopped" : message;
        failed_ = true;
    }
}

}  // namespace espi
