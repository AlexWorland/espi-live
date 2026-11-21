// The USB link protocol shared with ESPI Live (src/espi_live/iphone.py).
//
// Every message is a 16-byte little-endian header (magic "ESPC", version,
// kind, payload length, reserved) and a payload. The phone sends HELLO and
// STATUS as JSON and FRAME as a 48-byte FrameMeta followed by Mono8 rows; the
// computer sends COMMAND as JSON. In RAW mode a FRAME is instead pixel format
// Mono16: the FrameMeta, a 16-byte RawMeta (black level, white level, colour,
// pattern) and then one Bayer colour plane of little-endian 16-bit sensor
// values. Keep this file and iphone.py in step.

import Foundation

enum Link {
    static let port: UInt16 = 47820
    static let magic: [UInt8] = Array("ESPC".utf8)
    static let version: UInt16 = 1
    static let headerSize = 16
    static let frameMetaSize = 48
    static let rawMetaSize = 16
    static let maxIncoming = 1 << 20
    static let pixelMono8: UInt32 = 1
    static let pixelMono16: UInt32 = 2

    enum Kind: UInt16 {
        case hello = 1, status = 2, frame = 3, command = 4, error = 5
    }

    /// The 16-byte header for a message of `length` payload bytes.
    static func header(kind: Kind, length: Int) -> Data {
        var data = Data(capacity: headerSize)
        data.append(contentsOf: magic)
        data.appendLE(version)
        data.appendLE(kind.rawValue)
        data.appendLE(UInt32(length))
        data.appendLE(UInt32(0))
        return data
    }

    /// A complete JSON message.
    static func json<T: Encodable>(kind: Kind, _ value: T) -> Data {
        let body = (try? encoder.encode(value)) ?? Data("{}".utf8)
        var data = header(kind: kind, length: body.count)
        data.append(body)
        return data
    }

    /// The FRAME header and metadata; the caller appends width × height bytes.
    static func frameStart(width: Int, height: Int, sequence: UInt64, timestampNs: Int64,
                           exposureUs: Float, iso: Float, lensPosition: Float) -> Data {
        var data = header(kind: .frame, length: frameMetaSize + width * height)
        data.reserveCapacity(headerSize + frameMetaSize + width * height)
        data.appendLE(UInt32(width))
        data.appendLE(UInt32(height))
        data.appendLE(UInt32(width))  // stride: rows are always sent unpadded
        data.appendLE(pixelMono8)
        data.appendLE(sequence)
        data.appendLE(timestampNs)
        data.appendLE(exposureUs.bitPattern)
        data.appendLE(iso.bitPattern)
        data.appendLE(lensPosition.bitPattern)
        data.appendLE(UInt32(0))
        return data
    }

    /// The FRAME header, metadata and RAW metadata for one Bayer colour plane;
    /// the caller appends width × height little-endian UInt16 values.
    /// `channel` is 0 red, 1 green, 2 blue; `pattern` is 0 RGGB, 1 GRBG,
    /// 2 GBRG, 3 BGGR (the sensor's layout).
    static func rawFrameStart(width: Int, height: Int, sequence: UInt64, timestampNs: Int64,
                              exposureUs: Float, iso: Float, lensPosition: Float,
                              blackLevel: Float, whiteLevel: Float,
                              channel: UInt32, pattern: UInt32) -> Data {
        let length = frameMetaSize + rawMetaSize + width * height * 2
        var data = header(kind: .frame, length: length)
        data.reserveCapacity(headerSize + length)
        data.appendLE(UInt32(width))
        data.appendLE(UInt32(height))
        data.appendLE(UInt32(width))  // stride in values: rows are always sent unpadded
        data.appendLE(pixelMono16)
        data.appendLE(sequence)
        data.appendLE(timestampNs)
        data.appendLE(exposureUs.bitPattern)
        data.appendLE(iso.bitPattern)
        data.appendLE(lensPosition.bitPattern)
        data.appendLE(UInt32(0))
        data.appendLE(blackLevel.bitPattern)
        data.appendLE(whiteLevel.bitPattern)
        data.appendLE(channel)
        data.appendLE(pattern)
        return data
    }

    static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        return encoder
    }()

    static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }()
}

enum LinkError: Error, Equatable {
    case badMagic, badVersion(UInt16), tooLarge(Int)
}

/// Splits the byte stream from the computer into messages.
struct MessageParser {
    private var buffer = Data()

    /// Adds received bytes and returns every complete (kind, payload).
    mutating func feed(_ bytes: Data) throws -> [(UInt16, Data)] {
        buffer.append(bytes)
        var messages: [(UInt16, Data)] = []
        while buffer.count >= Link.headerSize {
            let start = buffer.startIndex
            let head = [UInt8](buffer[start ..< start + Link.headerSize])
            guard Array(head[0 ..< 4]) == Link.magic else { throw LinkError.badMagic }
            let version = UInt16(head[4]) | UInt16(head[5]) << 8
            guard version == Link.version else { throw LinkError.badVersion(version) }
            let kind = UInt16(head[6]) | UInt16(head[7]) << 8
            let length = Int(UInt32(head[8]) | UInt32(head[9]) << 8
                             | UInt32(head[10]) << 16 | UInt32(head[11]) << 24)
            guard length <= Link.maxIncoming else { throw LinkError.tooLarge(length) }
            guard buffer.count >= Link.headerSize + length else { break }
            let bodyStart = start + Link.headerSize
            messages.append((kind, Data(buffer[bodyStart ..< bodyStart + length])))
            buffer.removeSubrange(start ..< bodyStart + length)
        }
        return messages
    }
}

extension Data {
    mutating func appendLE<T: FixedWidthInteger>(_ value: T) {
        var little = value.littleEndian
        Swift.withUnsafeBytes(of: &little) { append(contentsOf: $0) }
    }
}

// MARK: - Messages

/// Settings the computer may change; every field is optional.
struct LinkCommand: Codable, Sendable, Equatable {
    var exposureUs: Double?
    var iso: Double?
    var fps: Double?
    var lensPosition: Double?
    var width: Int?
    var height: Int?
    var cropWidth: Int?
    var cropHeight: Int?
    var maxSendFps: Double?
    var refocus: Bool?
    var raw: Bool?
    var rawChannel: String?
}

/// What the camera is set to now.
struct CaptureSettings: Codable, Sendable, Equatable {
    var exposureUs: Double = 8000     // the FLIR camera's default exposure
    var iso: Double = 0               // 0 = the camera's lowest ISO
    var fps: Double = 30
    var lensPosition: Double = -1     // < 0 until the first focus lock
    var width = 1920
    var height = 1080
    var cropWidth = 0                 // 0 = full frame
    var cropHeight = 0
    var maxSendFps: Double = 0        // 0 = send every frame USB can carry
    var raw = false                   // Bayer RAW stills instead of the video luma
    var rawChannel = "red"            // which colour sites RAW keeps: red, green, blue
}

struct CaptureLimits: Codable, Sendable, Equatable {
    var exposureUs: [Double] = [0, 0]
    var iso: [Double] = [0, 0]
    var fps: [Double] = [1, 30]
    var lensPosition: [Double] = [0, 1]
}

struct CameraFormatInfo: Codable, Sendable, Hashable, Identifiable {
    var width: Int
    var height: Int
    var minFps: Double
    var maxFps: Double
    var binned: Bool
    var id: String { "\(width)x\(height)" }
}

struct LinkHello: Codable, Sendable {
    var protocolVersion = Int(Link.version)
    var app = "ESPI Camera"
    var appVersion: String
    var deviceModel: String
    var systemVersion: String
    var camera: String
    var pixelFormat = "mono8"
    var formats: [CameraFormatInfo]
    var limits: CaptureLimits
    var settings: CaptureSettings
}

struct LinkStatus: Codable, Sendable {
    var settings: CaptureSettings
    var limits: CaptureLimits
    var thermal: String
    var interrupted: String?
    var cameraFps: Double
    var sentFps: Double
    var sentFrames: Int
    var droppedFrames: Int
    var focusLocked: Bool
    var error: String?
}

struct LinkErrorMessage: Codable, Sendable {
    var message: String
}
