import CoreVideo
import Metal
import XCTest
@testable import ESPICamera

/// The GPU kernel must match the CPU reference exactly.
final class DifferenceEngineTests: XCTestCase {
    private let bayerRGGB = RawPipeline.bayerStills(format: kCVPixelFormatType_14Bayer_RGGB,
                                                    pattern: .rggb, bits: 14)
    private let luma8 = RawPipeline.luma(format: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                                         bits: 8)
    private let luma10 = RawPipeline.luma(format: kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
                                          bits: 10)

    private func engine() throws -> DifferenceEngine {
        try XCTUnwrap(DifferenceEngine(), "no Metal device")
    }

    /// Every site gets a distinct 14-bit value, so reading a wrong site shows.
    private func mosaic(_ width: Int, _ height: Int, seed: Int) -> [UInt16] {
        (0 ..< width * height).map { UInt16((($0 + 1) * 977 + seed * 131) % 16384) }
    }

    func testReferenceScaling() {
        let one = RawPipeline.bayerStills(format: kCVPixelFormatType_14Bayer_RGGB,
                                          pattern: .rggb, bits: 14)
        func bayerGrey(_ d: UInt16) -> UInt8 {
            // 2 × 2 mosaic: red site at (0, 0) differs by d.
            referenceDifference(cur: [d, 0, 0, 0], prev: [0, 0, 0, 0], width: 2, height: 2,
                                pipeline: one)[0]
        }
        XCTAssertEqual(bayerGrey(16383), 255)
        XCTAssertEqual(bayerGrey(8192), 128)
        XCTAssertEqual(bayerGrey(32), 0)
        XCTAssertEqual(bayerGrey(33), 1)
        XCTAssertEqual(referenceDifference(cur: [77], prev: [0], width: 1, height: 1,
                                           pipeline: luma8), [77])
        XCTAssertEqual(referenceDifference(cur: [1023 << 6], prev: [0], width: 1, height: 1,
                                           pipeline: luma10), [255])
    }

    func testEngineFirstImageBlack() throws {
        let engine = try engine()
        let out = try engine.process(makeBuffer16(mosaic(8, 6, seed: 1), width: 8, height: 6),
                                     pipeline: bayerRGGB)
        XCTAssertEqual(out.width, 4)
        XCTAssertEqual(out.height, 3)
        XCTAssertEqual(readR8(out), [UInt8](repeating: 0, count: 12))
    }

    func testEngineMatchesReferenceForEachPattern() throws {
        for pattern in [CFAPattern.rggb, .grbg, .gbrg, .bggr] {
            let pipeline = RawPipeline.bayerStills(format: kCVPixelFormatType_14Bayer_RGGB,
                                                   pattern: pattern, bits: 14)
            let engine = try engine()
            let first = mosaic(8, 6, seed: 1), second = mosaic(8, 6, seed: 40)
            _ = try engine.process(makeBuffer16(first, width: 8, height: 6), pipeline: pipeline)
            let out = try engine.process(makeBuffer16(second, width: 8, height: 6),
                                         pipeline: pipeline)
            let expected = referenceDifference(cur: second, prev: first, width: 8, height: 6,
                                               pipeline: pipeline)
            XCTAssertEqual(readR8(out), expected, "\(pattern)")
            XCTAssertTrue(expected.contains { $0 > 0 }, "the test images must differ")
        }
    }

    func testEngineCopiesBuffersWithoutIOSurface() throws {
        let engine = try engine()
        let first = mosaic(8, 6, seed: 3), second = mosaic(8, 6, seed: 17)
        _ = try engine.process(makeBuffer16(first, width: 8, height: 6, ioSurface: false),
                               pipeline: bayerRGGB)
        let out = try engine.process(makeBuffer16(second, width: 8, height: 6, ioSurface: false),
                                     pipeline: bayerRGGB)
        XCTAssertEqual(readR8(out), referenceDifference(cur: second, prev: first, width: 8,
                                                        height: 6, pipeline: bayerRGGB))
    }

    func testEngineAbsoluteWhenPreviousBrighter() throws {
        let engine = try engine()
        _ = try engine.process(make420f([200, 200, 200, 200], width: 2, height: 2),
                               pipeline: luma8)
        let out = try engine.process(make420f([50, 200, 210, 0], width: 2, height: 2),
                                     pipeline: luma8)
        XCTAssertEqual(readR8(out), [150, 0, 10, 200])
    }

    func testEngineLumaTenBitIgnoresChromaAndLowBits() throws {
        let engine = try engine()
        let first: [UInt16] = [100 << 6 | 0x3F, 500 << 6, 1023 << 6 | 0x15, 0]
        let second: [UInt16] = [200 << 6, 500 << 6 | 0x2A, 0 | 0x3F, 1023 << 6]
        _ = try engine.process(makeX420(first, width: 2, height: 2, chroma: 0x1234),
                               pipeline: luma10)
        let out = try engine.process(makeX420(second, width: 2, height: 2, chroma: 0xFFC0),
                                     pipeline: luma10)
        XCTAssertEqual(readR8(out), referenceDifference(cur: second, prev: first, width: 2,
                                                        height: 2, pipeline: luma10))
        XCTAssertEqual(readR8(out), [25, 0, 255, 255])
    }

    func testEngineOddMosaic() throws {
        let engine = try engine()
        let pipeline = RawPipeline.bayerStills(format: kCVPixelFormatType_14Bayer_BGGR,
                                               pattern: .bggr, bits: 14)
        let first = mosaic(5, 3, seed: 2), second = mosaic(5, 3, seed: 9)
        _ = try engine.process(makeBuffer16(first, width: 5, height: 3), pipeline: pipeline)
        let out = try engine.process(makeBuffer16(second, width: 5, height: 3),
                                     pipeline: pipeline)
        XCTAssertEqual(out.width, 2)
        XCTAssertEqual(out.height, 1)
        XCTAssertEqual(readR8(out), referenceDifference(cur: second, prev: first, width: 5,
                                                        height: 3, pipeline: pipeline))
    }

    func testEngineSizeChangeResets() throws {
        let engine = try engine()
        _ = try engine.process(makeBuffer16(mosaic(8, 6, seed: 1), width: 8, height: 6),
                               pipeline: bayerRGGB)
        let out = try engine.process(makeBuffer16(mosaic(10, 6, seed: 5), width: 10, height: 6),
                                     pipeline: bayerRGGB)
        XCTAssertEqual(out.width, 5)
        XCTAssertEqual(readR8(out), [UInt8](repeating: 0, count: 15))
    }

    func testEngineResetShowsBlack() throws {
        let engine = try engine()
        _ = try engine.process(makeBuffer8([10, 20, 30, 40], width: 2, height: 2), pipeline: luma8)
        let before = try engine.process(makeBuffer8([90, 20, 30, 40], width: 2, height: 2),
                                        pipeline: luma8)
        XCTAssertEqual(readR8(before), [80, 0, 0, 0])
        engine.reset()
        let after = try engine.process(makeBuffer8([0, 0, 0, 0], width: 2, height: 2),
                                       pipeline: luma8)
        XCTAssertEqual(readR8(after), [0, 0, 0, 0])
    }

    func testLatestIsLastOutput() throws {
        let engine = try engine()
        XCTAssertNil(engine.latest)
        let out = try engine.process(makeBuffer8([1, 2, 3, 4], width: 2, height: 2),
                                     pipeline: luma8)
        XCTAssertTrue(engine.latest === out)
    }
}
