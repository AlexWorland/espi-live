// Which camera data the on-phone difference view uses, and the constants the
// GPU kernel needs for it.
//
// The rawest source wins: Bayer RAW stills (unprocessed sensor mosaic, red
// photosites only, because the lab laser is a 633 nm HeNe), then 10-bit luma
// video, then 8-bit luma video. The choice is made from the pixel formats the
// camera offers, never from the device model, and is a pure function so it
// can be tested without a camera.

import CoreVideo
import Foundation
import Metal

/// Where the red photosite sits in each 2 × 2 cell of a Bayer mosaic.
enum CFAPattern: String, Sendable {
    case rggb, grbg, gbrg, bggr

    /// The red site's column and row inside its 2 × 2 cell.
    var redOffset: (x: Int, y: Int) {
        switch self {
        case .rggb: return (0, 0)
        case .grbg: return (1, 0)
        case .gbrg: return (0, 1)
        case .bggr: return (1, 1)
        }
    }
}

enum RawFormats {
    /// The CFA pattern and bit depth of a Bayer pixel format this app can read.
    static func bayer(_ type: OSType) -> (pattern: CFAPattern, bits: Int)? {
        switch type {
        case kCVPixelFormatType_14Bayer_RGGB: return (.rggb, 14)
        case kCVPixelFormatType_14Bayer_GRBG: return (.grbg, 14)
        case kCVPixelFormatType_14Bayer_GBRG: return (.gbrg, 14)
        case kCVPixelFormatType_14Bayer_BGGR: return (.bggr, 14)
        default: return nil
        }
    }

    /// A pixel format's four-character code, such as "rgg4".
    static func fourCC(_ type: OSType) -> String {
        let bytes = [24, 16, 8, 0].map { UInt8(truncatingIfNeeded: type >> $0) }
        let printable = bytes.allSatisfy { $0 >= 0x20 && $0 < 0x7F }
        return printable ? String(decoding: bytes, as: UTF8.self) : String(type)
    }
}

/// The source of the images the difference view subtracts.
enum RawPipeline: Sendable, Equatable {
    /// Back-to-back RAW stills; only the red photosites are used.
    case bayerStills(format: OSType, pattern: CFAPattern, bits: Int)
    /// The Y plane of video frames (luma mixes every colour).
    case luma(format: OSType, bits: Int)

    var bits: Int {
        switch self {
        case .bayerStills(_, _, let bits), .luma(_, let bits): return bits
        }
    }

    /// The largest value a sample can have; it maps to display white.
    var maxLevel: UInt32 { (UInt32(1) << UInt32(bits)) - 1 }

    /// Right shift that brings a sample to its own levels (x420 keeps 10 bits in the MSBs).
    var shift: UInt32 {
        if case .luma(kCVPixelFormatType_420YpCbCr10BiPlanarFullRange, _) = self { return 6 }
        return 0
    }

    /// Source pixels per output pixel along each axis.
    var step: UInt32 {
        if case .bayerStills = self { return 2 }
        return 1
    }

    /// The source pixel read for output (0, 0): the red site for Bayer.
    var offset: (x: UInt32, y: UInt32) {
        guard case .bayerStills(_, let pattern, _) = self else { return (0, 0) }
        return (UInt32(pattern.redOffset.x), UInt32(pattern.redOffset.y))
    }

    /// How plane 0 of the source buffer is read on the GPU.
    var sourcePixelFormat: MTLPixelFormat {
        if case .luma(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, _) = self { return .r8Uint }
        return .r16Uint
    }

    /// The difference image's size for a source of `width` × `height`.
    func outputSize(width: Int, height: Int) -> (width: Int, height: Int) {
        let step = Int(self.step)
        return (width / step, height / step)
    }

    /// What the screen says about this pipeline; `width` × `height` is the output size.
    func label(width: Int, height: Int, fps: Double) -> String {
        let name: String
        switch self {
        case .bayerStills: name = "Bayer RAW stills · red photosites"
        case .luma: name = "Luma video (all colours)"
        }
        return "\(name) · \(bits)-bit · \(width) × \(height) · "
            + String(format: "%.1f fps", fps)
    }
}

/// The pixel formats the camera offers.
struct PipelineCapabilities: Sendable, Equatable {
    /// Bayer RAW photo formats (those `AVCapturePhotoOutput.isBayerRAWPixelFormat` accepts).
    var rawPhotoFormats: [OSType]
    /// The `mediaSubType` of each device format.
    var videoFormats: [OSType]
}

struct PipelineChoice: Sendable, Equatable {
    var pipeline: RawPipeline?
    /// Why a rawer source was not used, or why nothing can be shown.
    var note: String?
}

/// Which source the difference view is asked to use.
enum DifferenceSource: String, Sendable, Equatable {
    /// The rawest the camera offers (Bayer RAW stills, else luma video).
    case auto
    /// Luma video only: the fallback when Bayer RAW stills do not arrive.
    case luma
}

/// Picks the rawest source the camera offers, or luma video when `source` is `.luma`.
func choosePipeline(_ caps: PipelineCapabilities,
                    source: DifferenceSource = .auto) -> PipelineChoice {
    var note: String?
    if source == .auto {
        for type in caps.rawPhotoFormats {
            if let bayer = RawFormats.bayer(type) {
                return PipelineChoice(pipeline: .bayerStills(format: type, pattern: bayer.pattern,
                                                             bits: bayer.bits))
            }
        }
        if let first = caps.rawPhotoFormats.first {
            note = "Bayer RAW format '\(RawFormats.fourCC(first))' is not supported yet; showing luma"
        }
    }
    let ten = kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
    let eight = kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
    if caps.videoFormats.contains(ten) {
        return PipelineChoice(pipeline: .luma(format: ten, bits: 10), note: note)
    }
    if caps.videoFormats.contains(eight) {
        return PipelineChoice(pipeline: .luma(format: eight, bits: 8), note: note)
    }
    return PipelineChoice(pipeline: nil, note: note ?? "No usable camera format")
}

// MARK: When Bayer RAW stills do not arrive

/// Seconds the view waits for its first Bayer still before showing luma video instead.
let bayerFirstImageTimeout = 6.0

/// Bayer stills have produced no image for `seconds` since the source started.
func bayerStalled(_ pipeline: RawPipeline?, imagesSinceStart: Int, seconds: Double) -> Bool {
    guard case .bayerStills = pipeline else { return false }
    return imagesSinceStart == 0 && seconds >= bayerFirstImageTimeout
}

/// What the photo output looks like right now, as far as taking a still is concerned.
struct RawReadiness: Sendable, Equatable {
    var sessionRunning: Bool
    var interrupted: Bool
    var stillInFlight: Bool
    /// The photo output has a video connection at all.
    var hasConnection: Bool
    var connectionEnabled: Bool
    var connectionActive: Bool
    /// The Bayer format the view chose, and what the camera offers now.
    var wantedFormat: OSType
    var offeredFormats: [OSType]
}

/// Why no still is being taken, in words for the screen; empty when one can be.
func rawBlockers(_ r: RawReadiness) -> [String] {
    var blockers: [String] = []
    if !r.sessionRunning { blockers.append("the camera session is not running") }
    if r.interrupted { blockers.append("the camera is interrupted") }
    if r.stillInFlight { blockers.append("the previous still has not come back") }
    if !r.hasConnection {
        blockers.append("the photo output has no camera connection")
    } else {
        if !r.connectionEnabled { blockers.append("the photo connection is disabled") }
        if !r.connectionActive { blockers.append("the photo connection is not active") }
    }
    if !r.offeredFormats.contains(r.wantedFormat) {
        let offered = r.offeredFormats.isEmpty
            ? "none" : r.offeredFormats.map(RawFormats.fourCC).joined(separator: ", ")
        blockers.append("the camera does not offer Bayer RAW "
                        + "'\(RawFormats.fourCC(r.wantedFormat))' now (offers: \(offered))")
    }
    return blockers
}

/// The header text before the first image: the source, how many stills were
/// asked for and came back, and what is holding the next one up.
func differenceWaitingText(pipeline: RawPipeline, asked: Int, returned: Int, discarded: Int,
                           blockers: [String]) -> String {
    var text: String
    switch pipeline {
    case .bayerStills:
        text = "Bayer RAW stills · waiting for the first image "
            + "(\(asked) asked, \(returned) returned, \(discarded) discarded)"
        if !blockers.isEmpty { text += "\nNot taking a still: " + blockers.joined(separator: "; ") }
    case .luma:
        text = "Luma video · waiting for the first image"
    }
    return text
}

/// The note shown when the view gives up on Bayer stills.
func bayerFallbackNote(seconds: Double, blockers: [String]) -> String {
    let why = blockers.isEmpty ? "no still came back" : blockers.joined(separator: "; ")
    return String(format: "Bayer RAW stills gave no image in %.0f s (", seconds) + why
        + "); showing luma video"
}
