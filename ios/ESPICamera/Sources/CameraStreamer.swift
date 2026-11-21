// Captures the back camera with every automatic adjustment locked and serves
// each frame's luma plane, uncompressed, to the computer over USB. In RAW mode
// it instead takes Bayer RAW stills (the sensor's data, without the video
// pipeline's noise reduction and tone mapping), keeps one colour plane of
// each and sends it as 16-bit values.
//
// The computer reaches the TCP listener on the phone's loopback interface
// through usbmuxd (the USB cable), so no network is involved. Everything —
// session configuration, the sample-buffer delegate, the listener and
// sending — runs on one serial queue, which is what makes the class safe to
// mark @unchecked Sendable.

@preconcurrency import AVFoundation
import Foundation
import ImageIO
import Network

struct DeviceInfo: Sendable {
    var model: String
    var systemVersion: String
    var appVersion: String

    static func current() -> DeviceInfo {
        var system = utsname()
        uname(&system)
        let model = withUnsafeBytes(of: &system.machine) { raw in
            String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
        }
        let os = ProcessInfo.processInfo.operatingSystemVersion
        let info = Bundle.main.infoDictionary ?? [:]
        let version = info["CFBundleShortVersionString"] as? String ?? "?"
        let build = info["CFBundleVersion"] as? String ?? "?"
        return DeviceInfo(model: model,
                          systemVersion: "\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)",
                          appVersion: "\(version) (\(build))")
    }
}

/// What the screen shows; sent to the main actor twice a second.
struct StreamerSnapshot: Sendable {
    var listening = false
    var connected = false
    var sentFps = 0.0
    var megabytesPerSecond = 0.0
    var cameraFps = 0.0
    var sentFrames = 0
    var droppedFrames = 0
    var settings = CaptureSettings()
    var limits = CaptureLimits()
    var formats: [CameraFormatInfo] = []
    var focusLocked = false
    var thermal = "nominal"
    var interrupted: String?
    var error: String?
    var lens = ""  // the back camera in use, for example "Wide 1×"
}

/// One back camera: its id and menu name as sent to ESPI Live, and the device.
struct LensChoice {
    let id: String
    let name: String
    let device: AVCaptureDevice
}

/// A camera format that delivers Bayer RAW stills at about 12 MP.
struct RawChoice: @unchecked Sendable {
    var format: AVCaptureDevice.Format
    var dimensions: CMVideoDimensions
    var pixelFormat: OSType
}

final class CameraStreamer: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate,
                            AVCapturePhotoCaptureDelegate, @unchecked Sendable {
    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "espi.camera", qos: .userInitiated)
    private let output = AVCaptureVideoDataOutput()
    private let photoOutput = AVCapturePhotoOutput()
    private var rawChoice: RawChoice?
    private var rawSearched = false
    private var rawInFlight = false
    private var rawPending: DispatchWorkItem?
    private let deviceInfo: DeviceInfo
    private let onSnapshot: @Sendable (StreamerSnapshot) -> Void

    // Everything below is touched only on `queue`.
    private var device: AVCaptureDevice?
    private var lenses: [LensChoice] = []
    private var lensId = ""
    private var configured = false
    private var settings = CaptureSettings()
    private var limits = CaptureLimits()
    private var formats: [CameraFormatInfo] = []
    private var focusLocked = false
    private var focusWork: DispatchWorkItem?
    private var listener: NWListener?
    private var listenerRetry: DispatchWorkItem?
    private var connection: NWConnection?
    private var connectionReady = false
    private var parser = MessageParser()
    private var inFlight = false
    private var sequence: UInt64 = 0
    private var lastSend = 0.0
    private var windowStart = ProcessInfo.processInfo.systemUptime
    private var capturedInWindow = 0
    private var sentInWindow = 0
    private var bytesInWindow = 0
    private var sentFrames = 0
    private var droppedFrames = 0
    private var cameraFps = 0.0
    private var sentFps = 0.0
    private var megabytesPerSecond = 0.0
    private var interrupted: String?
    private var lastError: String?
    private var statsTimer: DispatchSourceTimer?
    private var ticks = 0
    private var observers: [NSObjectProtocol] = []

    init(deviceInfo: DeviceInfo, onSnapshot: @escaping @Sendable (StreamerSnapshot) -> Void) {
        self.deviceInfo = deviceInfo
        self.onSnapshot = onSnapshot
        super.init()
    }

    // MARK: Public API (any thread)

    /// Configures the camera on first use, then (re)starts capture and the listener.
    func start() {
        queue.async { self.startOnQueue() }
    }

    /// Applies a change from the screen or the computer.
    func apply(_ command: LinkCommand) {
        queue.async { self.applyOnQueue(command) }
    }

    // MARK: Startup

    private func startOnQueue() {
        if !configured {
            configured = true
            configureSession()
            observeNotifications()
            startStatsTimer()
            refocusOnQueue()
        }
        startListenerIfNeeded()
        if !session.isRunning && device != nil {
            session.startRunning()
        }
        publish()
    }

    private func configureSession() {
        lenses = Self.discoverLenses()
        guard let first = lenses.first(where: { $0.id == "wide" }) ?? lenses.first else {
            lastError = "No back camera found"
            return
        }
        session.beginConfiguration()
        session.sessionPreset = .inputPriority
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String:
                kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        ]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) { session.addOutput(output) }
        if session.canAddOutput(photoOutput) { session.addOutput(photoOutput) }
        session.commitConfiguration()
        selectLens(first)
    }

    /// The back cameras this phone has, narrowest first: ultra wide, wide, telephoto.
    static func discoverLenses() -> [LensChoice] {
        let kinds: [(type: AVCaptureDevice.DeviceType, id: String, name: String)] = [
            (.builtInUltraWideCamera, "ultra_wide", "Ultra Wide"),
            (.builtInWideAngleCamera, "wide", "Wide"),
            (.builtInTelephotoCamera, "telephoto", "Telephoto"),
        ]
        let found = AVCaptureDevice.DiscoverySession(
            deviceTypes: kinds.map(\.type), mediaType: .video, position: .back).devices
        let present = kinds.compactMap { kind in
            found.first { $0.deviceType == kind.type }.map { (kind: kind, device: $0) }
        }
        let main = present.first { $0.kind.id == "wide" } ?? present.first
        let mainAngle = main.map { fieldOfView($0.device) } ?? 0
        return present.map { entry in
            let zoom = entry.kind.id == "wide" ? 1.0 : LensMath.zoom(
                lensFieldOfView: fieldOfView(entry.device), mainFieldOfView: mainAngle)
            return LensChoice(id: entry.kind.id,
                              name: "\(entry.kind.name) \(LensMath.zoomText(zoom))",
                              device: entry.device)
        }
    }

    /// Horizontal field of view at 1920 × 1080, so cameras are compared at one frame size.
    private static func fieldOfView(_ camera: AVCaptureDevice) -> Double {
        let format = camera.formats.first { format in
            let size = format.formatDescription.dimensions
            return isFullRangeYUV(format) && size.width == 1920 && size.height == 1080
        } ?? camera.activeFormat
        return Double(format.videoFieldOfView)
    }

    /// Points the session at one back camera and reads that camera's formats.
    private func selectLens(_ choice: LensChoice) {
        session.beginConfiguration()
        for input in session.inputs { session.removeInput(input) }
        do {
            let input = try AVCaptureDeviceInput(device: choice.device)
            if session.canAddInput(input) { session.addInput(input) }
        } catch {
            lastError = "Camera input: \(error.localizedDescription)"
        }
        session.commitConfiguration()
        device = choice.device
        lensId = choice.id
        rawChoice = nil  // each camera has its own RAW formats
        rawSearched = false
        formats = Self.formatList(choice.device)
        applyCameraSettings()
    }

    private var lensName: String {
        lenses.first { $0.id == lensId }?.name ?? ""
    }

    /// One entry per frame size, preferring unbinned formats and then the highest rate.
    static func formatList(_ camera: AVCaptureDevice) -> [CameraFormatInfo] {
        var best: [String: CameraFormatInfo] = [:]
        for format in camera.formats where isFullRangeYUV(format) {
            let size = format.formatDescription.dimensions
            let rates = format.videoSupportedFrameRateRanges
            let entry = CameraFormatInfo(
                width: Int(size.width), height: Int(size.height),
                minFps: rates.map(\.minFrameRate).min() ?? 1,
                maxFps: rates.map(\.maxFrameRate).max() ?? 30,
                binned: format.isVideoBinned)
            if let old = best[entry.id] {
                let better = (old.binned && !entry.binned)
                    || (old.binned == entry.binned && entry.maxFps > old.maxFps)
                if !better { continue }
            }
            best[entry.id] = entry
        }
        return best.values.sorted { $0.width * $0.height < $1.width * $1.height }
    }

    private static func isFullRangeYUV(_ format: AVCaptureDevice.Format) -> Bool {
        format.formatDescription.mediaSubType.rawValue
            == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
    }

    /// The device format for the requested size: unbinned first, fast enough for the rate.
    private func chooseFormat(_ camera: AVCaptureDevice) -> AVCaptureDevice.Format? {
        let matching = camera.formats.filter { format in
            let size = format.formatDescription.dimensions
            return Self.isFullRangeYUV(format) && Int(size.width) == settings.width
                && Int(size.height) == settings.height
        }
        func maxRate(_ format: AVCaptureDevice.Format) -> Double {
            format.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0
        }
        let fastEnough = matching.filter { maxRate($0) >= settings.fps }
        let pool = fastEnough.isEmpty ? matching : fastEnough
        return pool.sorted { a, b in
            if a.isVideoBinned != b.isVideoBinned { return !a.isVideoBinned }
            return maxRate(a) > maxRate(b)
        }.first
    }

    /// The most pixels a RAW still may have: the sensor's binned 12 MP readout.
    /// (Apple only offers Bayer RAW binned; full resolution is processed ProRAW.)
    private static let rawPixelLimit = 13_000_000

    /// Finds a format whose photo output offers plain Bayer RAW (not ProRAW), and
    /// the largest photo size up to 12 MP it supports. Makes each candidate the
    /// active format to ask the photo output, so it runs once and the answer is kept.
    private func findRawFormat(_ camera: AVCaptureDevice) -> RawChoice? {
        rawSearched = true
        photoOutput.isAppleProRAWEnabled = false
        let candidates = camera.formats.compactMap { format -> (AVCaptureDevice.Format, CMVideoDimensions)? in
            let sizes = format.supportedMaxPhotoDimensions.filter {
                Int($0.width) * Int($0.height) <= Self.rawPixelLimit
            }
            guard let best = sizes.max(by: { Int($0.width) * Int($0.height)
                                             < Int($1.width) * Int($1.height) })
            else { return nil }
            return (format, best)
        }.sorted { Int($0.1.width) * Int($0.1.height) > Int($1.1.width) * Int($1.1.height) }
        for (format, dimensions) in candidates {
            do {
                try camera.lockForConfiguration()
                camera.activeFormat = format
                camera.unlockForConfiguration()
            } catch {
                continue
            }
            let plain = photoOutput.availableRawPhotoPixelFormatTypes.first {
                !AVCapturePhotoOutput.isAppleProRAWPixelFormat($0)
                    && Self.bayerLayout($0) != nil
            }
            if let plain {
                return RawChoice(format: format, dimensions: dimensions, pixelFormat: plain)
            }
        }
        return nil
    }

    // MARK: Camera settings

    /// Sets the format and locks every automatic adjustment that would change the image.
    private func applyCameraSettings() {
        guard let camera = device else { return }
        if settings.raw {
            if rawChoice == nil && !rawSearched { rawChoice = findRawFormat(camera) }
            if rawChoice == nil {
                lastError = "This camera has no Bayer RAW photo format; staying on video"
                settings.raw = false
            }
        }
        if chooseFormat(camera) == nil, let fallback = formats.first(where: {
            $0.width == 1920 && $0.height == 1080 }) ?? formats.last {
            settings.width = fallback.width
            settings.height = fallback.height
        }
        guard let format = settings.raw ? rawChoice?.format : chooseFormat(camera) else {
            lastError = "No \(settings.width) × \(settings.height) camera format"
            return
        }
        session.beginConfiguration()
        defer {
            session.commitConfiguration()
            if settings.raw, let choice = rawChoice {
                photoOutput.maxPhotoDimensions = choice.dimensions
            }
        }
        // RAW takes stills, so the video output is detached while it runs.
        if settings.raw {
            if session.outputs.contains(output) { session.removeOutput(output) }
        } else if !session.outputs.contains(output), session.canAddOutput(output) {
            session.addOutput(output)
        }
        do {
            try camera.lockForConfiguration()
            defer { camera.unlockForConfiguration() }
            if camera.activeFormat != format { camera.activeFormat = format }

            if #available(iOS 18.0, *), camera.isAutoVideoFrameRateEnabled {
                camera.isAutoVideoFrameRateEnabled = false
            }
            let rates = format.videoSupportedFrameRateRanges
            let lowRate = rates.map(\.minFrameRate).min() ?? 1
            let highRate = rates.map(\.maxFrameRate).max() ?? 30
            settings.fps = min(max(settings.fps, lowRate), highRate)
            let frame = CMTime(seconds: 1.0 / settings.fps, preferredTimescale: 1_000_000)
            camera.activeVideoMinFrameDuration = frame
            camera.activeVideoMaxFrameDuration = frame

            if format.isVideoHDRSupported {
                camera.automaticallyAdjustsVideoHDREnabled = false
                camera.isVideoHDREnabled = false
            }
            if camera.isGeometricDistortionCorrectionSupported {
                camera.isGeometricDistortionCorrectionEnabled = false
            }
            if camera.isLowLightBoostSupported {
                camera.automaticallyEnablesLowLightBoostWhenAvailable = false
            }
            if format.supportedColorSpaces.contains(.sRGB) {
                camera.activeColorSpace = .sRGB
            }
            if format.isGlobalToneMappingSupported {  // one curve per frame, not local
                camera.isGlobalToneMappingEnabled = true
            }
            camera.videoZoomFactor = 1.0

            // Exposure: always custom, never automatic.
            let minUs = format.minExposureDuration.seconds * 1e6
            let maxUs = format.maxExposureDuration.seconds * 1e6
            limits = CaptureLimits(exposureUs: [minUs, maxUs],
                                   iso: [Double(format.minISO), Double(format.maxISO)],
                                   fps: [lowRate, highRate], lensPosition: [0, 1])
            settings.exposureUs = min(max(settings.exposureUs, minUs), maxUs)
            settings.iso = settings.iso <= 0 ? Double(format.minISO)
                : min(max(settings.iso, Double(format.minISO)), Double(format.maxISO))
            if camera.isExposureModeSupported(.custom) {
                camera.setExposureModeCustom(
                    duration: CMTime(seconds: settings.exposureUs / 1e6,
                                     preferredTimescale: 1_000_000_000),
                    iso: Float(settings.iso), completionHandler: nil)
            }

            // White balance: fixed unity gains, the sensor's own response.
            if camera.isWhiteBalanceModeSupported(.locked) {
                let unity = AVCaptureDevice.WhiteBalanceGains(redGain: 1, greenGain: 1,
                                                              blueGain: 1)
                camera.setWhiteBalanceModeLocked(with: unity, completionHandler: nil)
            }

            if settings.lensPosition >= 0, camera.isLockingFocusWithCustomLensPositionSupported {
                camera.setFocusModeLocked(lensPosition: Float(min(max(settings.lensPosition, 0), 1)),
                                          completionHandler: nil)
                focusLocked = true
            }
        } catch {
            lastError = "Camera settings: \(error.localizedDescription)"
        }
        if let connection = output.connection(with: .video) {
            if connection.isVideoStabilizationSupported {
                connection.preferredVideoStabilizationMode = .off
            }
            if #available(iOS 27.0, *) {
                connection.automaticallyEnablesLowLightVideoNoiseReduction = false
                if connection.isLowLightVideoNoiseReductionSupported {
                    connection.isLowLightVideoNoiseReductionEnabled = false
                }
            }
        }
    }

    private func applyOnQueue(_ command: LinkCommand) {
        var reconfigure = false
        var switched = false
        // The camera first, so the other values in the same command are clamped to its limits.
        if let id = command.lens, id != lensId {
            if let choice = lenses.first(where: { $0.id == id }) {
                focusWork?.cancel()
                settings.lensPosition = -1  // the focus of one camera means nothing on another
                selectLens(choice)
                refocusOnQueue()
                switched = true
            } else {
                send(Link.json(kind: .error,
                               LinkErrorMessage(message: "This phone has no \(id) camera")))
            }
        }
        if let value = command.exposureUs { settings.exposureUs = value; reconfigure = true }
        if let value = command.iso { settings.iso = value; reconfigure = true }
        if let value = command.fps, value > 0 { settings.fps = value; reconfigure = true }
        if let value = command.lensPosition {
            settings.lensPosition = min(max(value, 0), 1)
            focusWork?.cancel()
            reconfigure = true
        }
        if let width = command.width, let height = command.height {
            settings.width = width
            settings.height = height
            reconfigure = true
        }
        var modeChanged = false
        if let value = command.raw, value != settings.raw {
            settings.raw = value
            modeChanged = true
            reconfigure = true
        }
        if let value = command.rawChannel, ["red", "green", "blue"].contains(value) {
            settings.rawChannel = value
        }
        if let value = command.cropWidth { settings.cropWidth = max(0, value) }
        if let value = command.cropHeight { settings.cropHeight = max(0, value) }
        if let value = command.maxSendFps { settings.maxSendFps = max(0, value) }
        if reconfigure { applyCameraSettings() }
        if modeChanged {
            rawPending?.cancel()
            if settings.raw { scheduleRawCapture(after: 0.3) }  // let the format settle
        }
        if command.refocus == true { refocusOnQueue() }
        if switched { sendHello() }  // formats and limits belong to the camera
        sendStatus()
        publish()
    }

    /// Focuses once on the centre, then locks the lens where it settled.
    private func refocusOnQueue() {
        guard let camera = device else { return }
        do {
            try camera.lockForConfiguration()
            if camera.isFocusPointOfInterestSupported {
                camera.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
            }
            if camera.isFocusModeSupported(.autoFocus) { camera.focusMode = .autoFocus }
            camera.unlockForConfiguration()
            focusLocked = false
        } catch {
            lastError = "Focus: \(error.localizedDescription)"
            return
        }
        focusWork?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.lockFocusOnQueue() }
        focusWork = work
        queue.asyncAfter(deadline: .now() + 1.5, execute: work)
    }

    private func lockFocusOnQueue() {
        guard let camera = device else { return }
        settings.lensPosition = Double(camera.lensPosition)
        do {
            try camera.lockForConfiguration()
            camera.setFocusModeLocked(lensPosition: AVCaptureDevice.currentLensPosition,
                                      completionHandler: nil)
            camera.unlockForConfiguration()
            focusLocked = true
        } catch {
            lastError = "Focus lock: \(error.localizedDescription)"
        }
        sendStatus()
        publish()
    }

    // MARK: Frames

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        if settings.raw { return }  // RAW mode takes stills instead
        capturedInWindow += 1
        guard let link = self.connection, connectionReady else { return }
        if inFlight {  // USB is still busy with the previous frame: skip, never queue
            droppedFrames += 1
            return
        }
        let now = ProcessInfo.processInfo.systemUptime
        if settings.maxSendFps > 0, now - lastSend < 1.0 / settings.maxSendFps - 0.002 {
            return
        }
        guard let pixels = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        CVPixelBufferLockBaseAddress(pixels, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixels, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(pixels, 0) else { return }
        let width = CVPixelBufferGetWidthOfPlane(pixels, 0)
        let height = CVPixelBufferGetHeightOfPlane(pixels, 0)
        let rowBytes = CVPixelBufferGetBytesPerRowOfPlane(pixels, 0)
        let cropWidth = settings.cropWidth > 0 ? min(settings.cropWidth, width) : width
        let cropHeight = settings.cropHeight > 0 ? min(settings.cropHeight, height) : height
        let left = (width - cropWidth) / 2
        let top = (height - cropHeight) / 2

        let stamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        sequence += 1
        var message = Link.frameStart(
            width: cropWidth, height: cropHeight, sequence: sequence,
            timestampNs: Int64(stamp.seconds * 1e9),
            exposureUs: Float((device?.exposureDuration.seconds ?? 0) * 1e6),
            iso: device?.iso ?? 0, lensPosition: device?.lensPosition ?? 0)
        let source = base.assumingMemoryBound(to: UInt8.self)
        if left == 0 && rowBytes == cropWidth {
            message.append(source + top * rowBytes, count: cropWidth * cropHeight)
        } else {
            for row in 0 ..< cropHeight {
                message.append(source + (top + row) * rowBytes + left, count: cropWidth)
            }
        }
        inFlight = true
        lastSend = now
        let size = message.count
        link.send(content: message, completion: .contentProcessed { [weak self] error in
            guard let self else { return }
            self.inFlight = false
            if let error { self.lastError = "Send: \(error.localizedDescription)" }
        })
        sentInWindow += 1
        sentFrames += 1
        bytesInWindow += size
    }

    // MARK: Bayer RAW stills

    /// Queues the next RAW still after `delay` seconds.
    private func scheduleRawCapture(after delay: TimeInterval = 0) {
        rawPending?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.captureRawOnQueue() }
        rawPending = work
        queue.asyncAfter(deadline: .now() + delay, execute: work)
    }

    /// Takes one RAW still when the link is ready and idle, else tries again soon.
    private func captureRawOnQueue() {
        guard settings.raw, let choice = rawChoice else { return }
        guard connectionReady, session.isRunning, !rawInFlight, !inFlight else {
            scheduleRawCapture(after: 0.05)
            return
        }
        let now = ProcessInfo.processInfo.systemUptime
        if settings.maxSendFps > 0 {
            let wait = lastSend + 1.0 / settings.maxSendFps - now
            if wait > 0.002 {
                scheduleRawCapture(after: wait)
                return
            }
        }
        let photoSettings = AVCapturePhotoSettings(rawPixelFormatType: choice.pixelFormat)
        photoSettings.maxPhotoDimensions = choice.dimensions
        photoSettings.photoQualityPrioritization = .speed  // required for a Bayer RAW still
        photoSettings.flashMode = .off
        rawInFlight = true
        photoOutput.capturePhoto(with: photoSettings, delegate: self)
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto,
                     error: Error?) {
        queue.async { self.rawPhotoFinished(photo, error: error) }
    }

    private func rawPhotoFinished(_ photo: AVCapturePhoto, error: Error?) {
        rawInFlight = false
        defer { if settings.raw { scheduleRawCapture() } }
        capturedInWindow += 1
        if let error {
            lastError = "RAW capture: \(error.localizedDescription)"
            return
        }
        guard settings.raw, let link = connection, connectionReady else { return }
        guard !inFlight else {
            droppedFrames += 1
            return
        }
        guard photo.isRawPhoto, let pixels = photo.pixelBuffer,
              let layout = Self.bayerLayout(CVPixelBufferGetPixelFormatType(pixels)) else {
            lastError = "RAW capture returned no Bayer data"
            return
        }
        let wanted: Character = ["red": "R", "green": "G", "blue": "B"][settings.rawChannel] ?? "R"
        let sites = layout.letters.indices.filter { layout.letters[$0] == wanted }
        let levels = Self.rawLevels(photo.metadata, sites: sites)
        if !levels.found {
            lastError = "RAW data has no black or white level; assuming 0 and 16383"
        }

        CVPixelBufferLockBaseAddress(pixels, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixels, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixels) else { return }
        let plane = Self.extractPlane(
            base: UnsafeRawPointer(base), rowBytes: CVPixelBufferGetBytesPerRow(pixels),
            width: CVPixelBufferGetWidth(pixels), height: CVPixelBufferGetHeight(pixels),
            sites: sites, cropWidth: settings.cropWidth, cropHeight: settings.cropHeight)

        let stamp = photo.timestamp
        sequence += 1
        var message = Link.rawFrameStart(
            width: plane.width, height: plane.height, sequence: sequence,
            timestampNs: Int64(stamp.seconds * 1e9),
            exposureUs: Float((device?.exposureDuration.seconds ?? 0) * 1e6),
            iso: device?.iso ?? 0, lensPosition: device?.lensPosition ?? 0,
            blackLevel: levels.black, whiteLevel: levels.white,
            channel: ["R": 0, "G": 1, "B": 2][wanted] ?? 0, pattern: layout.pattern)
        plane.values.withUnsafeBytes { message.append(contentsOf: $0) }  // little-endian
        inFlight = true
        lastSend = ProcessInfo.processInfo.systemUptime
        let size = message.count
        link.send(content: message, completion: .contentProcessed { [weak self] error in
            guard let self else { return }
            self.inFlight = false
            if let error { self.lastError = "Send: \(error.localizedDescription)" }
        })
        sentInWindow += 1
        sentFrames += 1
        bytesInWindow += size
    }

    /// The colour layout of a Bayer pixel format: the pattern code the link
    /// sends, and the colour at (0,0), (0,1), (1,0), (1,1) of each 2 × 2 cell.
    static func bayerLayout(_ type: OSType) -> (pattern: UInt32, letters: [Character])? {
        switch type {
        case kCVPixelFormatType_14Bayer_RGGB: return (0, ["R", "G", "G", "B"])
        case kCVPixelFormatType_14Bayer_GRBG: return (1, ["G", "R", "B", "G"])
        case kCVPixelFormatType_14Bayer_GBRG: return (2, ["G", "B", "R", "G"])
        case kCVPixelFormatType_14Bayer_BGGR: return (3, ["B", "G", "G", "R"])
        default: return nil
        }
    }

    /// Black and white levels (sensor counts) for the chosen colour sites, from
    /// the photo's DNG metadata. `found` is false when it has none.
    static func rawLevels(_ metadata: [String: Any], sites: [Int])
        -> (black: Float, white: Float, found: Bool) {
        func numbers(_ value: Any?) -> [Double] {
            if let list = value as? [Any] { return list.compactMap { ($0 as? NSNumber)?.doubleValue } }
            if let one = value as? NSNumber { return [one.doubleValue] }
            return []
        }
        let dng = metadata[kCGImagePropertyDNGDictionary as String] as? [String: Any]
        let blacks = numbers(dng?[kCGImagePropertyDNGBlackLevel as String])
        let whites = numbers(dng?[kCGImagePropertyDNGWhiteLevel as String])
        guard let white = whites.max(), !blacks.isEmpty else { return (0, 16383, false) }
        let mine = blacks.count >= 4 ? sites.map { blacks[$0] } : [blacks[0]]
        let black = mine.reduce(0, +) / Double(max(mine.count, 1))
        return (Float(black), Float(white), true)
    }

    /// Cuts one colour's samples out of a Bayer buffer: a plane half the buffer's
    /// width and height (two sites are averaged for green), optionally a
    /// centre crop of `cropWidth` × `cropHeight` plane pixels (0 = all).
    static func extractPlane(base: UnsafeRawPointer, rowBytes: Int, width: Int, height: Int,
                             sites: [Int], cropWidth: Int, cropHeight: Int)
        -> (values: [UInt16], width: Int, height: Int) {
        let planeWidth = width / 2, planeHeight = height / 2
        let outWidth = cropWidth > 0 ? min(cropWidth, planeWidth) : planeWidth
        let outHeight = cropHeight > 0 ? min(cropHeight, planeHeight) : planeHeight
        let left = (planeWidth - outWidth) / 2, top = (planeHeight - outHeight) / 2
        let first = sites.first ?? 0
        let second = sites.count > 1 ? sites[1] : nil
        let values = [UInt16](unsafeUninitializedCapacity: outWidth * outHeight) { buffer, count in
            for y in 0 ..< outHeight {
                let row = top + y
                let a = (base + (2 * row + first / 2) * rowBytes).assumingMemoryBound(to: UInt16.self)
                let aOffset = 2 * left + first % 2
                if let second {
                    let b = (base + (2 * row + second / 2) * rowBytes)
                        .assumingMemoryBound(to: UInt16.self)
                    let bOffset = 2 * left + second % 2
                    for x in 0 ..< outWidth {
                        let sum = UInt32(a[aOffset + 2 * x]) + UInt32(b[bOffset + 2 * x])
                        buffer[y * outWidth + x] = UInt16((sum + 1) / 2)
                    }
                } else {
                    for x in 0 ..< outWidth { buffer[y * outWidth + x] = a[aOffset + 2 * x] }
                }
            }
            count = outWidth * outHeight
        }
        return (values, outWidth, outHeight)
    }

    // MARK: USB listener

    private func startListenerIfNeeded() {
        guard listener == nil else { return }
        do {
            let tcp = NWProtocolTCP.Options()
            tcp.noDelay = true
            let parameters = NWParameters(tls: nil, tcp: tcp)
            parameters.requiredInterfaceType = .loopback  // usbmuxd connects from here
            parameters.allowLocalEndpointReuse = true
            let created = try NWListener(using: parameters,
                                         on: NWEndpoint.Port(rawValue: Link.port)!)
            created.newConnectionHandler = { [weak self] connection in
                self?.accept(connection)
            }
            created.stateUpdateHandler = { [weak self, weak created] state in
                guard let self, let created else { return }
                self.listenerChanged(created, state)
            }
            listener = created
            created.start(queue: queue)
        } catch {
            lastError = "USB listener: \(error.localizedDescription)"
            retryListener()
        }
    }

    private func listenerChanged(_ which: NWListener, _ state: NWListener.State) {
        guard which === listener else { return }
        switch state {
        case .failed(let error):
            lastError = "USB listener: \(error.localizedDescription)"
            which.cancel()
            listener = nil
            retryListener()
        case .cancelled:
            listener = nil
        default:
            break
        }
        publish()
    }

    private func retryListener() {
        listenerRetry?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.startListenerIfNeeded() }
        listenerRetry = work
        queue.asyncAfter(deadline: .now() + 2, execute: work)
    }

    /// A new computer connection replaces the old one.
    private func accept(_ incoming: NWConnection) {
        connection?.cancel()
        connection = incoming
        connectionReady = false
        inFlight = false
        parser = MessageParser()
        incoming.stateUpdateHandler = { [weak self, weak incoming] state in
            guard let self, let incoming else { return }
            self.connectionChanged(incoming, state)
        }
        incoming.start(queue: queue)
        receive(on: incoming)
    }

    private func connectionChanged(_ which: NWConnection, _ state: NWConnection.State) {
        guard which === connection else { return }
        switch state {
        case .ready:
            connectionReady = true
            lastError = nil
            sendHello()
            sendStatus()
            if settings.raw { scheduleRawCapture() }
        case .failed, .cancelled:
            connection = nil
            connectionReady = false
            inFlight = false
        default:
            break
        }
        publish()
    }

    private func receive(on link: NWConnection) {
        link.receive(minimumIncompleteLength: 1, maximumLength: 65536) {
            [weak self] data, _, isComplete, error in
            guard let self else { return }
            if let data, !data.isEmpty {
                do {
                    for (kind, payload) in try self.parser.feed(data) {
                        self.handle(kind: kind, payload: payload)
                    }
                } catch {
                    self.lastError = "Bad data from the computer: \(error)"
                    link.cancel()
                    return
                }
            }
            if isComplete || error != nil {
                link.cancel()
                return
            }
            self.receive(on: link)
        }
    }

    private func handle(kind: UInt16, payload: Data) {
        guard kind == Link.Kind.command.rawValue else { return }
        do {
            applyOnQueue(try Link.decoder.decode(LinkCommand.self, from: payload))
        } catch {
            send(Link.json(kind: .error, LinkErrorMessage(message: "Unreadable command: \(error)")))
        }
    }

    private func send(_ message: Data) {
        guard let link = connection, connectionReady else { return }
        link.send(content: message, completion: .idempotent)
    }

    private func sendHello() {
        send(Link.json(kind: .hello, LinkHello(
            appVersion: deviceInfo.appVersion, deviceModel: deviceInfo.model,
            systemVersion: deviceInfo.systemVersion,
            camera: lensName.isEmpty ? "Back camera" : "Back \(lensName)",
            lens: lensId, lenses: lenses.map { LensInfo(id: $0.id, name: $0.name) },
            formats: formats, limits: limits, settings: settings)))
    }

    private func sendStatus() {
        send(Link.json(kind: .status, LinkStatus(
            settings: settings, limits: limits, thermal: Self.thermalText(),
            interrupted: interrupted, cameraFps: cameraFps, sentFps: sentFps,
            sentFrames: sentFrames, droppedFrames: droppedFrames,
            focusLocked: focusLocked, error: lastError)))
    }

    // MARK: Statistics and system events

    private func startStatsTimer() {
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + 0.5, repeating: 0.5)
        timer.setEventHandler { [weak self] in self?.tick() }
        timer.resume()
        statsTimer = timer
    }

    private func tick() {
        let now = ProcessInfo.processInfo.systemUptime
        let elapsed = max(now - windowStart, 0.001)
        cameraFps = Double(capturedInWindow) / elapsed
        sentFps = Double(sentInWindow) / elapsed
        megabytesPerSecond = Double(bytesInWindow) / elapsed / 1e6
        capturedInWindow = 0
        sentInWindow = 0
        bytesInWindow = 0
        windowStart = now
        ticks += 1
        if ticks % 2 == 0 { sendStatus() }  // once a second: the computer's liveness check
        publish()
    }

    private func publish() {
        onSnapshot(StreamerSnapshot(
            listening: listener != nil, connected: connectionReady, sentFps: sentFps,
            megabytesPerSecond: megabytesPerSecond, cameraFps: cameraFps,
            sentFrames: sentFrames, droppedFrames: droppedFrames, settings: settings,
            limits: limits, formats: formats, focusLocked: focusLocked,
            thermal: Self.thermalText(), interrupted: interrupted, error: lastError,
            lens: lensName))
    }

    static func thermalText() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private func observeNotifications() {
        let center = NotificationCenter.default
        observe(center, AVCaptureSession.wasInterruptedNotification, session) { me, note in
            me.interrupted = Self.interruptionText(note)
            me.sendStatus()
        }
        observe(center, AVCaptureSession.interruptionEndedNotification, session) { me, _ in
            me.interrupted = nil
            me.sendStatus()
        }
        observe(center, AVCaptureSession.runtimeErrorNotification, session) { me, note in
            me.lastError = "Camera error: \(note.error)"
            if !me.session.isRunning { me.session.startRunning() }
        }
        observe(center, ProcessInfo.thermalStateDidChangeNotification, nil) { me, _ in
            me.sendStatus()
        }
    }

    /// Runs `work` on the capture queue for each notification, then refreshes the screen.
    private func observe(_ center: NotificationCenter, _ name: Notification.Name,
                         _ object: AnyObject?,
                         _ work: @escaping @Sendable (CameraStreamer, NoteSummary) -> Void) {
        observers.append(center.addObserver(forName: name, object: object, queue: nil) {
            [weak self] note in
            guard let me = self else { return }
            let summary = NoteSummary(note)
            me.queue.async {
                work(me, summary)
                me.publish()
            }
        })
    }

    private static func interruptionText(_ note: NoteSummary) -> String {
        guard let raw = note.reason,
              let reason = AVCaptureSession.InterruptionReason(rawValue: raw) else {
            return "interrupted"
        }
        switch reason {
        case .videoDeviceNotAvailableInBackground: return "the app is in the background"
        case .audioDeviceInUseByAnotherClient, .videoDeviceInUseByAnotherClient:
            return "another app is using the camera"
        case .videoDeviceNotAvailableWithMultipleForegroundApps:
            return "another app is on screen"
        case .videoDeviceNotAvailableDueToSystemPressure: return "the phone is too hot"
        default: return "interrupted"
        }
    }
}

/// The parts of a notification the capture queue needs (Notification is not Sendable).
struct NoteSummary: Sendable {
    var reason: Int?
    var error: String

    init(_ note: Notification) {
        reason = note.userInfo?[AVCaptureSessionInterruptionReasonKey] as? Int
        error = (note.userInfo?[AVCaptureSessionErrorKey] as? NSError)?.localizedDescription
            ?? "unknown"
    }
}
