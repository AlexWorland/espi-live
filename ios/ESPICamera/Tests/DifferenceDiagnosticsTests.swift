import CoreVideo
import XCTest
@testable import ESPICamera

/// What the difference view says while it waits for its first image, and when it
/// gives up on Bayer RAW stills and shows luma video instead.
final class DifferenceDiagnosticsTests: XCTestCase {
    private let x420 = kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
    private let rggb = kCVPixelFormatType_14Bayer_RGGB
    private let bayer = RawPipeline.bayerStills(format: kCVPixelFormatType_14Bayer_RGGB,
                                                pattern: .rggb, bits: 14)
    private let luma = RawPipeline.luma(format: kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
                                        bits: 10)

    private func ready() -> RawReadiness {
        RawReadiness(sessionRunning: true, interrupted: false, stillInFlight: false,
                     hasConnection: true, connectionEnabled: true, connectionActive: true,
                     wantedFormat: rggb, offeredFormats: [rggb])
    }

    // MARK: Forcing luma

    func testForcedLumaSkipsBayerEvenWhenOffered() {
        let caps = PipelineCapabilities(rawPhotoFormats: [rggb], videoFormats: [x420])
        let choice = choosePipeline(caps, source: .luma)
        XCTAssertEqual(choice.pipeline, .luma(format: x420, bits: 10))
        XCTAssertNil(choice.note, "choosing luma is not a problem to report")
    }

    func testAutoStillPrefersBayer() {
        let caps = PipelineCapabilities(rawPhotoFormats: [rggb], videoFormats: [x420])
        XCTAssertEqual(choosePipeline(caps, source: .auto).pipeline,
                       .bayerStills(format: rggb, pattern: .rggb, bits: 14))
    }

    func testForcedLumaWithNoVideoFormatIsNothingUsable() {
        let choice = choosePipeline(PipelineCapabilities(rawPhotoFormats: [rggb],
                                                         videoFormats: []), source: .luma)
        XCTAssertNil(choice.pipeline)
        XCTAssertNotNil(choice.note)
    }

    // MARK: When to give up on Bayer stills

    func testBayerStallsOnlyAfterTheTimeoutWithNoImage() {
        XCTAssertFalse(bayerStalled(bayer, imagesSinceStart: 0, seconds: bayerFirstImageTimeout - 0.1))
        XCTAssertTrue(bayerStalled(bayer, imagesSinceStart: 0, seconds: bayerFirstImageTimeout))
    }

    func testAnImageMeansNoStall() {
        XCTAssertFalse(bayerStalled(bayer, imagesSinceStart: 1, seconds: 600))
    }

    func testOnlyBayerStalls() {
        XCTAssertFalse(bayerStalled(luma, imagesSinceStart: 0, seconds: 600))
        XCTAssertFalse(bayerStalled(nil, imagesSinceStart: 0, seconds: 600))
    }

    // MARK: Why a still is not being taken

    func testReadyHasNoBlockers() {
        XCTAssertEqual(rawBlockers(ready()), [])
    }

    func testEachBlockerIsNamed() {
        var r = ready()
        r.sessionRunning = false
        XCTAssertEqual(rawBlockers(r), ["the camera session is not running"])
        r = ready(); r.interrupted = true
        XCTAssertEqual(rawBlockers(r), ["the camera is interrupted"])
        r = ready(); r.stillInFlight = true
        XCTAssertEqual(rawBlockers(r), ["the previous still has not come back"])
        r = ready(); r.hasConnection = false
        XCTAssertEqual(rawBlockers(r), ["the photo output has no camera connection"])
        r = ready(); r.connectionEnabled = false
        XCTAssertEqual(rawBlockers(r), ["the photo connection is disabled"])
        r = ready(); r.connectionActive = false
        XCTAssertEqual(rawBlockers(r), ["the photo connection is not active"])
    }

    func testMissingFormatListsWhatTheCameraOffers() {
        var r = ready()
        r.offeredFormats = []
        XCTAssertEqual(rawBlockers(r),
                       ["the camera does not offer Bayer RAW 'rgg4' now (offers: none)"])
        r.offeredFormats = [osType("bp16"), osType("bgg4")]
        XCTAssertEqual(rawBlockers(r),
                       ["the camera does not offer Bayer RAW 'rgg4' now (offers: bp16, bgg4)"])
    }

    func testBlockersAccumulate() {
        var r = ready()
        r.sessionRunning = false
        r.connectionActive = false
        XCTAssertEqual(rawBlockers(r).count, 2)
    }

    func testNoConnectionDoesNotAlsoReportItsFlags() {
        var r = ready()
        r.hasConnection = false
        r.connectionEnabled = false
        r.connectionActive = false
        XCTAssertEqual(rawBlockers(r), ["the photo output has no camera connection"])
    }

    // MARK: The text under the header

    func testWaitingTextCountsStillsAndNamesTheBlocker() {
        let text = differenceWaitingText(pipeline: bayer, asked: 3, returned: 1, discarded: 1,
                                         blockers: ["the camera is interrupted"])
        XCTAssertTrue(text.contains("Bayer RAW stills"), text)
        XCTAssertTrue(text.contains("3 asked"), text)
        XCTAssertTrue(text.contains("1 returned"), text)
        XCTAssertTrue(text.contains("1 discarded"), text)
        XCTAssertTrue(text.contains("the camera is interrupted"), text)
    }

    func testWaitingTextForLumaHasNoStillCounts() {
        let text = differenceWaitingText(pipeline: luma, asked: 0, returned: 0, discarded: 0,
                                         blockers: [])
        XCTAssertTrue(text.contains("Luma video"), text)
        XCTAssertFalse(text.contains("asked"), text)
    }

    func testWaitingTextWithoutBlockersSaysTheStillIsOnItsWay() {
        let text = differenceWaitingText(pipeline: bayer, asked: 1, returned: 0, discarded: 0,
                                         blockers: [])
        XCTAssertTrue(text.contains("1 asked"), text)
        XCTAssertFalse(text.contains("Not taking"), text)
    }

    func testFallbackNoteGivesTheReason() {
        let note = bayerFallbackNote(seconds: 6, blockers: ["the photo connection is not active"])
        XCTAssertTrue(note.contains("6 s"), note)
        XCTAssertTrue(note.contains("the photo connection is not active"), note)
        XCTAssertTrue(note.contains("luma"), note)
    }

    func testFallbackNoteWithoutABlockerSaysNoStillCame() {
        let note = bayerFallbackNote(seconds: 6, blockers: [])
        XCTAssertTrue(note.contains("no still came back"), note)
    }
}
