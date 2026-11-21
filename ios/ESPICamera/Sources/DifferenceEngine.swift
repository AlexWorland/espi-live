// Subtracts each camera image from the one before it on the GPU and keeps the
// result as an r8Unorm texture for the screen. Camera-free, so it is tested
// in the simulator against a CPU reference (Tests/DifferenceReference.swift).
//
// process() and reset() are called from one thread at a time (the camera
// queue); `latest` may be read from any thread (the display).

import CoreVideo
import Foundation
import Metal

/// Mirrors `DiffParams` in Difference.metal.
struct DiffParams {
    var offsetX: UInt32
    var offsetY: UInt32
    var step: UInt32
    var shift: UInt32
    var maxLevel: UInt32
    var hasPrevious: UInt32
}

enum DifferenceError: Error, Equatable {
    case texture(String)
}

final class DifferenceEngine: @unchecked Sendable {
    let device: MTLDevice
    private let commands: MTLCommandQueue
    private let kernel: MTLComputePipelineState
    private var cache: CVMetalTextureCache?

    private let work = NSLock()       // guards everything down to `published`
    private var outputs: [MTLTexture] = []   // two, so the screen never reads the one being written
    private var history: [MTLTexture] = []   // [previous image, this image], swapped after each
    private var nextOutput = 0
    private var hasPrevious = false
    private var lastPipeline: RawPipeline?

    private let published = NSLock()
    private var latestTexture: MTLTexture?

    init?(device: MTLDevice? = MTLCreateSystemDefaultDevice()) {
        guard let device, let commands = device.makeCommandQueue(),
              let function = device.makeDefaultLibrary()?.makeFunction(name: "raw_difference"),
              let kernel = try? device.makeComputePipelineState(function: function) else {
            return nil
        }
        self.device = device
        self.commands = commands
        self.kernel = kernel
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
    }

    /// The newest difference image, or nil before the first.
    var latest: MTLTexture? {
        published.withLock { latestTexture }
    }

    /// Forgets the previous image, so the next one shows black, and clears the
    /// screen, so nothing from before stays up under a new label.
    func reset() {
        work.withLock { hasPrevious = false }
        published.withLock { latestTexture = nil }
    }

    /// Subtracts the previous image from `buffer` (plane 0) and returns the
    /// display-grey difference; black when there is no previous image of the
    /// same pipeline and size. `fullScale` is the difference shown as white
    /// (a RAW still's white minus black level); nil uses `pipeline.maxLevel`.
    /// Waits for the GPU.
    func process(_ buffer: CVPixelBuffer, pipeline: RawPipeline,
                 fullScale: UInt32? = nil) throws -> MTLTexture {
        work.lock()
        defer { work.unlock() }
        let planar = CVPixelBufferIsPlanar(buffer)
        let width = planar ? CVPixelBufferGetWidthOfPlane(buffer, 0) : CVPixelBufferGetWidth(buffer)
        let height = planar ? CVPixelBufferGetHeightOfPlane(buffer, 0)
            : CVPixelBufferGetHeight(buffer)
        let size = pipeline.outputSize(width: width, height: height)
        guard size.width > 0, size.height > 0 else {
            throw DifferenceError.texture("image of \(width) × \(height) is too small")
        }
        if pipeline != lastPipeline || outputs.first?.width != size.width
            || outputs.first?.height != size.height {
            try allocate(width: size.width, height: size.height)
            lastPipeline = pipeline
            hasPrevious = false
        }
        let (source, keepAlive) = try sourceTexture(buffer, pipeline: pipeline, planar: planar,
                                                    width: width, height: height)
        var params = DiffParams(offsetX: pipeline.offset.x, offsetY: pipeline.offset.y,
                                step: pipeline.step, shift: pipeline.shift,
                                maxLevel: max(fullScale ?? pipeline.maxLevel, 1),
                                hasPrevious: hasPrevious ? 1 : 0)
        let output = outputs[nextOutput]
        guard let commandBuffer = commands.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DifferenceError.texture("no GPU command buffer")
        }
        encoder.setComputePipelineState(kernel)
        encoder.setTexture(source, index: 0)
        encoder.setTexture(history[0], index: 1)
        encoder.setTexture(history[1], index: 2)
        encoder.setTexture(output, index: 3)
        encoder.setBytes(&params, length: MemoryLayout<DiffParams>.stride, index: 0)
        let group = MTLSize(width: 16, height: 16, depth: 1)
        encoder.dispatchThreadgroups(
            MTLSize(width: (size.width + 15) / 16, height: (size.height + 15) / 16, depth: 1),
            threadsPerThreadgroup: group)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        withExtendedLifetime(keepAlive) {}
        if let cache { CVMetalTextureCacheFlush(cache, 0) }
        if let error = commandBuffer.error {
            throw DifferenceError.texture("GPU: \(error.localizedDescription)")
        }
        history.swapAt(0, 1)
        hasPrevious = true
        nextOutput = 1 - nextOutput
        published.withLock { latestTexture = output }
        return output
    }

    private func allocate(width: Int, height: Int) throws {
        func make(_ format: MTLPixelFormat, _ usage: MTLTextureUsage,
                  _ storage: MTLStorageMode) throws -> MTLTexture {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: format, width: width, height: height, mipmapped: false)
            descriptor.usage = usage
            descriptor.storageMode = storage
            guard let texture = device.makeTexture(descriptor: descriptor) else {
                throw DifferenceError.texture("cannot allocate \(width) × \(height)")
            }
            return texture
        }
        outputs = try (0 ..< 2).map { _ in try make(.r8Unorm, [.shaderWrite, .shaderRead], .shared) }
        history = try (0 ..< 2).map { _ in try make(.r16Uint, [.shaderWrite, .shaderRead], .private) }
        nextOutput = 0
    }

    /// Plane 0 as a GPU texture: zero-copy through the texture cache when the
    /// buffer allows it, otherwise copied row by row. The second value must be
    /// kept alive until the GPU has finished.
    private func sourceTexture(_ buffer: CVPixelBuffer, pipeline: RawPipeline, planar: Bool,
                               width: Int, height: Int) throws -> (MTLTexture, AnyObject?) {
        if let cache {
            var wrapped: CVMetalTexture?
            let status = CVMetalTextureCacheCreateTextureFromImage(
                nil, cache, buffer, nil, pipeline.sourcePixelFormat, width, height, 0, &wrapped)
            if status == kCVReturnSuccess, let wrapped,
               let texture = CVMetalTextureGetTexture(wrapped) {
                return (texture, wrapped)
            }
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pipeline.sourcePixelFormat, width: width, height: height,
            mipmapped: false)
        descriptor.usage = .shaderRead
        descriptor.storageMode = .shared
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw DifferenceError.texture("cannot allocate the source image")
        }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let base = planar ? CVPixelBufferGetBaseAddressOfPlane(buffer, 0)
            : CVPixelBufferGetBaseAddress(buffer)
        let rowBytes = planar ? CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
            : CVPixelBufferGetBytesPerRow(buffer)
        guard let base else { throw DifferenceError.texture("camera image has no pixels") }
        let sampleBytes = pipeline.sourcePixelFormat == .r16Uint ? 2 : 1
        guard rowBytes >= width * sampleBytes else {  // a buffer of another pixel format
            throw DifferenceError.texture("camera image is not \(pipeline.bits)-bit")
        }
        texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                        withBytes: base, bytesPerRow: rowBytes)
        return (texture, nil)
    }
}
