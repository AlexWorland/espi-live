// Test helpers: synthetic camera buffers and a CPU reference of the GPU
// difference kernel. Not product code.

import CoreVideo
import Metal
@testable import ESPICamera

private var bufferAttributes: [String: Any] {
    [kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
     kCVPixelBufferMetalCompatibilityKey as String: true]
}

/// Copies rows of `bytesPerSample`-byte samples into a plane, honouring its row padding.
private func fill<T>(_ buffer: CVPixelBuffer, plane: Int?, _ values: [T], width: Int, height: Int) {
    let base: UnsafeMutableRawPointer
    let rowBytes: Int
    if let plane {
        base = CVPixelBufferGetBaseAddressOfPlane(buffer, plane)!
        rowBytes = CVPixelBufferGetBytesPerRowOfPlane(buffer, plane)
    } else {
        base = CVPixelBufferGetBaseAddress(buffer)!
        rowBytes = CVPixelBufferGetBytesPerRow(buffer)
    }
    values.withUnsafeBytes { source in
        let size = MemoryLayout<T>.size
        for row in 0 ..< height {
            (base + row * rowBytes).copyMemory(from: source.baseAddress! + row * width * size,
                                               byteCount: width * size)
        }
    }
}

private func create(_ format: OSType, _ width: Int, _ height: Int,
                    ioSurface: Bool = true) -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    let attributes = ioSurface ? bufferAttributes : [:]
    let status = CVPixelBufferCreate(nil, width, height, format,
                                     attributes as CFDictionary, &buffer)
    precondition(status == kCVReturnSuccess, "CVPixelBufferCreate \(status)")
    return buffer!
}

/// A single-plane 16-bit buffer, as a Bayer RAW still's mosaic is laid out.
/// Without an IOSurface the engine cannot wrap it and must copy it.
func makeBuffer16(_ values: [UInt16], width: Int, height: Int,
                  ioSurface: Bool = true) -> CVPixelBuffer {
    let buffer = create(kCVPixelFormatType_OneComponent16, width, height, ioSurface: ioSurface)
    CVPixelBufferLockBaseAddress(buffer, [])
    fill(buffer, plane: nil, values, width: width, height: height)
    CVPixelBufferUnlockBaseAddress(buffer, [])
    return buffer
}

/// A single-plane 8-bit buffer.
func makeBuffer8(_ values: [UInt8], width: Int, height: Int) -> CVPixelBuffer {
    let buffer = create(kCVPixelFormatType_OneComponent8, width, height)
    CVPixelBufferLockBaseAddress(buffer, [])
    fill(buffer, plane: nil, values, width: width, height: height)
    CVPixelBufferUnlockBaseAddress(buffer, [])
    return buffer
}

/// A 10-bit (x420) video buffer: `y` fills plane 0 as stored (10 bits in the
/// MSBs), and the CbCr plane is filled with `chroma`.
func makeX420(_ y: [UInt16], width: Int, height: Int, chroma: UInt16 = 0xBEEF) -> CVPixelBuffer {
    let buffer = create(kCVPixelFormatType_420YpCbCr10BiPlanarFullRange, width, height)
    CVPixelBufferLockBaseAddress(buffer, [])
    fill(buffer, plane: 0, y, width: width, height: height)
    fill(buffer, plane: 1, [UInt16](repeating: chroma, count: width * height / 2),
         width: width, height: height / 2)
    CVPixelBufferUnlockBaseAddress(buffer, [])
    return buffer
}

/// An 8-bit (420f) video buffer with `y` in plane 0 and `chroma` in CbCr.
func make420f(_ y: [UInt8], width: Int, height: Int, chroma: UInt8 = 0xAB) -> CVPixelBuffer {
    let buffer = create(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, width, height)
    CVPixelBufferLockBaseAddress(buffer, [])
    fill(buffer, plane: 0, y, width: width, height: height)
    fill(buffer, plane: 1, [UInt8](repeating: chroma, count: width * height / 2),
         width: width, height: height / 2)
    CVPixelBufferUnlockBaseAddress(buffer, [])
    return buffer
}

/// What the kernel must produce for `cur` after `prev` (nil: no previous image).
/// Both are `width` × `height` source samples as stored (before the shift).
func referenceDifference(cur: [UInt16], prev: [UInt16]?, width: Int, height: Int,
                         pipeline: RawPipeline) -> [UInt8] {
    let size = pipeline.outputSize(width: width, height: height)
    guard let prev else { return [UInt8](repeating: 0, count: size.width * size.height) }
    let step = Int(pipeline.step), shift = pipeline.shift, maxLevel = pipeline.maxLevel
    var result: [UInt8] = []
    for y in 0 ..< size.height {
        for x in 0 ..< size.width {
            let index = (y * step + Int(pipeline.offset.y)) * width + x * step
                + Int(pipeline.offset.x)
            let c = Int(UInt32(cur[index]) >> shift)
            let p = Int(UInt32(prev[index]) >> shift)
            let d = UInt32(abs(c - p))
            result.append(UInt8(min((d * 255 + (maxLevel >> 1)) / maxLevel, 255)))
        }
    }
    return result
}

/// The bytes of an r8Unorm texture, row by row.
func readR8(_ texture: MTLTexture) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: texture.width * texture.height)
    bytes.withUnsafeMutableBytes { raw in
        texture.getBytes(raw.baseAddress!, bytesPerRow: texture.width,
                         from: MTLRegionMake2D(0, 0, texture.width, texture.height),
                         mipmapLevel: 0)
    }
    return bytes
}
