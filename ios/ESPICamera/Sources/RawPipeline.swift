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

/// Picks the rawest source the camera offers.
func choosePipeline(_ caps: PipelineCapabilities) -> PipelineChoice {
    var note: String?
    for type in caps.rawPhotoFormats {
        if let bayer = RawFormats.bayer(type) {
            return PipelineChoice(pipeline: .bayerStills(format: type, pattern: bayer.pattern,
                                                         bits: bayer.bits))
        }
    }
    if let first = caps.rawPhotoFormats.first {
        note = "Bayer RAW format '\(RawFormats.fourCC(first))' is not supported yet; showing luma"
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
