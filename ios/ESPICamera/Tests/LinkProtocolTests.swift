import XCTest
@testable import ESPICamera

/// The wire format must match src/espi_live/iphone.py byte for byte.
final class LinkProtocolTests: XCTestCase {
    func testHeaderLayout() {
        let header = Link.header(kind: .status, length: 0x0102_0304)
        XCTAssertEqual([UInt8](header), [
            0x45, 0x53, 0x50, 0x43,  // "ESPC"
            0x01, 0x00,              // version 1
            0x02, 0x00,              // kind 2 (status)
            0x04, 0x03, 0x02, 0x01,  // payload length, little-endian
            0x00, 0x00, 0x00, 0x00,  // reserved
        ])
    }

    func testFrameStartLayout() {
        var frame = Link.frameStart(width: 3, height: 2, sequence: 7, timestampNs: -1,
                                    exposureUs: 8000, iso: 50, lensPosition: 0.5)
        XCTAssertEqual(frame.count, Link.headerSize + Link.frameMetaSize)
        frame.append(contentsOf: [1, 2, 3, 4, 5, 6])
        let bytes = [UInt8](frame)
        func u32(_ at: Int) -> UInt32 {
            (0 ..< 4).reduce(UInt32(0)) { $0 | UInt32(bytes[at + $1]) << (8 * $1) }
        }
        XCTAssertEqual(u32(8), UInt32(Link.frameMetaSize + 6))  // payload length
        XCTAssertEqual(u32(16), 3)  // width
        XCTAssertEqual(u32(20), 2)  // height
        XCTAssertEqual(u32(24), 3)  // stride
        XCTAssertEqual(u32(28), Link.pixelMono8)
        XCTAssertEqual(u32(32), 7)  // sequence, low word
        XCTAssertEqual(Float(bitPattern: u32(48)), 8000)
        XCTAssertEqual(Float(bitPattern: u32(52)), 50)
        XCTAssertEqual(Float(bitPattern: u32(56)), 0.5)
        XCTAssertEqual(Array(bytes.suffix(6)), [1, 2, 3, 4, 5, 6])
    }

    func testParserSplitsAndJoinsMessages() throws {
        let command = Link.json(kind: .command, LinkCommand(exposureUs: 1234, cropWidth: 512))
        let second = Link.json(kind: .command, LinkCommand(refocus: true))
        var parser = MessageParser()
        let stream = command + second
        let first = try parser.feed(stream.prefix(5))
        XCTAssertTrue(first.isEmpty)
        let rest = try parser.feed(stream.dropFirst(5))
        XCTAssertEqual(rest.count, 2)
        XCTAssertEqual(rest[0].0, Link.Kind.command.rawValue)
        let decoded = try Link.decoder.decode(LinkCommand.self, from: rest[0].1)
        XCTAssertEqual(decoded.exposureUs, 1234)
        XCTAssertEqual(decoded.cropWidth, 512)
        XCTAssertEqual(try Link.decoder.decode(LinkCommand.self, from: rest[1].1).refocus, true)
    }

    func testParserRejectsForeignData() {
        var parser = MessageParser()
        XCTAssertThrowsError(try parser.feed(Data(repeating: 0x41, count: 16))) {
            XCTAssertEqual($0 as? LinkError, .badMagic)
        }
    }

    func testParserRejectsHugeMessage() {
        var parser = MessageParser()
        XCTAssertThrowsError(try parser.feed(Link.header(kind: .command, length: 1 << 24))) {
            XCTAssertEqual($0 as? LinkError, .tooLarge(1 << 24))
        }
    }

    func testCommandUsesSnakeCaseKeys() throws {
        let json = #"{"exposure_us":500,"max_send_fps":15,"lens_position":0.25}"#
        let command = try Link.decoder.decode(LinkCommand.self, from: Data(json.utf8))
        XCTAssertEqual(command, LinkCommand(exposureUs: 500, lensPosition: 0.25, maxSendFps: 15))
    }

    func testHelloKeys() throws {
        let hello = LinkHello(appVersion: "1", deviceModel: "iPhone16,1", systemVersion: "27.0",
                              camera: "Back", formats: [], limits: CaptureLimits(),
                              settings: CaptureSettings())
        let message = Link.json(kind: .hello, hello)
        let body = message.dropFirst(Link.headerSize)
        let object = try JSONSerialization.jsonObject(with: body) as? [String: Any]
        XCTAssertEqual(object?["protocol_version"] as? Int, 1)
        XCTAssertEqual(object?["device_model"] as? String, "iPhone16,1")
        let settings = object?["settings"] as? [String: Any]
        XCTAssertEqual(settings?["exposure_us"] as? Double, 8000)
        XCTAssertNotNil(settings?["crop_width"])
    }
}
