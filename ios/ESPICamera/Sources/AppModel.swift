import AVFoundation
import Observation
import SwiftUI

/// Lets the streamer's background callback reach the model without a retain cycle.
final class WeakModel: @unchecked Sendable {
    @MainActor weak var model: AppModel?
}

@MainActor
@Observable
final class AppModel {
    var snapshot = StreamerSnapshot()
    var cameraAllowed: Bool?
    var previewOn = true

    @ObservationIgnored let streamer: CameraStreamer

    init() {
        let box = WeakModel()
        streamer = CameraStreamer(deviceInfo: .current()) { snapshot in
            Task { @MainActor in box.model?.snapshot = snapshot }
        }
        box.model = self
    }

    /// Asks for camera access once, then starts capturing and listening.
    func start() async {
        if cameraAllowed == nil {
            cameraAllowed = await AVCaptureDevice.requestAccess(for: .video)
        }
        if cameraAllowed == true { streamer.start() }
    }

    func apply(_ command: LinkCommand) {
        streamer.apply(command)
    }
}
