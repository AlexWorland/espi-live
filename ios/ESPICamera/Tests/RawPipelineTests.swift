import CoreVideo
import Metal
import XCTest
@testable import ESPICamera

/// The OSType for a four-character code such as "bp16".
func osType(_ code: String) -> OSType {
    code.utf8.reduce(OSType(0)) { $0 << 8 | OSType($1) }
}

/// Pipeline choice and the per-source constants the GPU kernel uses.
final class RawPipelineTests: XCTestCase {
    private let x420 = kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
    private let f420 = kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
    private let rggb = kCVPixelFormatType_14Bayer_RGGB

    func testRedOffsets() {
        XCTAssertTrue(CFAPattern.rggb.redOffset == (0, 0))
        XCTAssertTrue(CFAPattern.grbg.redOffset == (1, 0))
        XCTAssertTrue(CFAPattern.gbrg.redOffset == (0, 1))
        XCTAssertTrue(CFAPattern.bggr.redOffset == (1, 1))
    }

    func testBayerTypes() {
        let grbg = RawFormats.bayer(kCVPixelFormatType_14Bayer_GRBG)
        XCTAssertEqual(grbg?.pattern, .grbg)
        XCTAssertEqual(grbg?.bits, 14)
        XCTAssertEqual(RawFormats.bayer(kCVPixelFormatType_14Bayer_GBRG)?.pattern, .gbrg)
        XCTAssertEqual(RawFormats.bayer(kCVPixelFormatType_14Bayer_BGGR)?.pattern, .bggr)
        XCTAssertNil(RawFormats.bayer(f420))
        XCTAssertEqual(RawFormats.fourCC(rggb), "rgg4")
    }

    func testPrefersBayerStills() {
        let choice = choosePipeline(PipelineCapabilities(rawPhotoFormats: [rggb],
                                                         videoFormats: [f420, x420]))
        XCTAssertEqual(choice.pipeline, .bayerStills(format: rggb, pattern: .rggb, bits: 14))
        XCTAssertNil(choice.note)
    }

    func testUnknownBayerFallsBackToLumaWithNote() {
        let choice = choosePipeline(PipelineCapabilities(rawPhotoFormats: [osType("bp16")],
                                                         videoFormats: [f420, x420]))
        XCTAssertEqual(choice.pipeline, .luma(format: x420, bits: 10))
        XCTAssertTrue(choice.note?.contains("bp16") == true, choice.note ?? "no note")
    }

    func testTenBitBeforeEightBit() {
        XCTAssertEqual(choosePipeline(PipelineCapabilities(rawPhotoFormats: [],
                                                           videoFormats: [f420, x420])).pipeline,
                       .luma(format: x420, bits: 10))
        XCTAssertEqual(choosePipeline(PipelineCapabilities(rawPhotoFormats: [],
                                                           videoFormats: [f420])).pipeline,
                       .luma(format: f420, bits: 8))
    }

    func testNothingUsable() {
        let choice = choosePipeline(PipelineCapabilities(rawPhotoFormats: [], videoFormats: []))
        XCTAssertNil(choice.pipeline)
        XCTAssertNotNil(choice.note)
    }

    func testDerivedConstants() {
        let bayer = RawPipeline.bayerStills(format: rggb, pattern: .bggr, bits: 14)
        XCTAssertEqual(bayer.maxLevel, 16383)
        XCTAssertEqual(bayer.shift, 0)
        XCTAssertEqual(bayer.step, 2)
        XCTAssertTrue(bayer.offset == (1, 1))
        XCTAssertEqual(bayer.sourcePixelFormat, .r16Uint)

        let ten = RawPipeline.luma(format: x420, bits: 10)
        XCTAssertEqual(ten.maxLevel, 1023)
        XCTAssertEqual(ten.shift, 6)
        XCTAssertEqual(ten.step, 1)
        XCTAssertTrue(ten.offset == (0, 0))
        XCTAssertEqual(ten.sourcePixelFormat, .r16Uint)

        let eight = RawPipeline.luma(format: f420, bits: 8)
        XCTAssertEqual(eight.maxLevel, 255)
        XCTAssertEqual(eight.shift, 0)
        XCTAssertEqual(eight.step, 1)
        XCTAssertEqual(eight.sourcePixelFormat, .r8Uint)
    }

    func testOutputSizeOddMosaic() {
        let bayer = RawPipeline.bayerStills(format: rggb, pattern: .rggb, bits: 14)
        XCTAssertTrue(bayer.outputSize(width: 5, height: 3) == (2, 1))
        XCTAssertTrue(bayer.outputSize(width: 4032, height: 3024) == (2016, 1512))
        let luma = RawPipeline.luma(format: f420, bits: 8)
        XCTAssertTrue(luma.outputSize(width: 5, height: 3) == (5, 3))
    }

    func testDisplayQuadAspectFit() {
        // Landscape 2016 × 1512 turns portrait (1512 wide, 2016 tall) on a 1320 × 2868 screen.
        let tall = displayQuad(texture: (w: 2016, h: 1512), view: (w: 1320, h: 2868))
        XCTAssertEqual(tall.x, 1.0, accuracy: 1e-4)
        XCTAssertEqual(tall.y, Float(1320.0 / 1512.0 * 2016.0 / 2868.0), accuracy: 1e-4)
        let square = displayQuad(texture: (w: 100, h: 100), view: (w: 100, h: 200))
        XCTAssertEqual(square.x, 1.0, accuracy: 1e-4)
        XCTAssertEqual(square.y, 0.5, accuracy: 1e-4)
    }

    func testLabels() {
        let bayer = RawPipeline.bayerStills(format: rggb, pattern: .rggb, bits: 14)
        XCTAssertEqual(bayer.label(width: 2016, height: 1512, fps: 6.84),
                       "Bayer RAW stills · red photosites · 14-bit · 2016 × 1512 · 6.8 fps")
        XCTAssertEqual(RawPipeline.luma(format: x420, bits: 10).label(width: 1920, height: 1080,
                                                                     fps: 30),
                       "Luma video (all colours) · 10-bit · 1920 × 1080 · 30.0 fps")
    }
}
