import SwiftUI

struct ContentView: View {
    @Bindable var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            ZStack {
                Color.black
                if model.previewOn && model.cameraAllowed == true {
                    CameraPreview(session: model.streamer.session)
                } else {
                    Text(model.cameraAllowed == false ? "Camera access is off" : "Preview off")
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxHeight: 300)
            StatusCard(snapshot: model.snapshot)
            if model.cameraAllowed == false {
                Text("Allow camera access in Settings › ESPI Camera.")
                    .padding()
            }
            ControlsForm(model: model)
        }
    }
}

struct StatusCard: View {
    let snapshot: StreamerSnapshot

    var body: some View {
        let s = snapshot.settings
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Circle().fill(snapshot.connected ? .green : .orange).frame(width: 10, height: 10)
                Text(snapshot.connected ? "Sending to the computer over USB"
                     : snapshot.listening ? "Waiting for ESPI Live on the computer"
                     : "Starting the USB link…")
                    .font(.headline)
            }
            Text(String(format: "Sent %.1f fps · %.1f MB/s · camera %.1f fps",
                        snapshot.sentFps, snapshot.megabytesPerSecond, snapshot.cameraFps))
            Text("Skipped while USB was busy: \(snapshot.droppedFrames)")
                .foregroundStyle(.secondary)
            if s.raw {
                Text("Bayer RAW · \(s.rawChannel) plane · 16-bit, no video processing")
            }
            Text(String(format: "%d × %d%@ · %.0f µs · ISO %.0f · focus %@",
                        s.width, s.height, cropText(s), s.exposureUs, s.iso,
                        snapshot.focusLocked ? String(format: "%.2f locked", s.lensPosition)
                                             : "adjusting"))
            if snapshot.thermal == "serious" || snapshot.thermal == "critical" {
                Label("Phone is \(snapshot.thermal)ly hot", systemImage: "thermometer.high")
                    .foregroundStyle(.red)
            }
            if let interrupted = snapshot.interrupted {
                Label("Camera paused: \(interrupted)", systemImage: "pause.circle")
                    .foregroundStyle(.orange)
            }
            if let error = snapshot.error {
                Label(error, systemImage: "exclamationmark.triangle").foregroundStyle(.red)
            }
        }
        .font(.footnote.monospacedDigit())
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(.thinMaterial)
    }

    private func cropText(_ s: CaptureSettings) -> String {
        s.cropWidth > 0 && s.cropHeight > 0 ? " (crop \(s.cropWidth) × \(s.cropHeight))" : ""
    }
}

struct ControlsForm: View {
    @Bindable var model: AppModel
    @State private var exposure = 8000.0
    @State private var iso = 0.0
    @State private var lens = 0.5
    @State private var editing = false

    private static let crops = [0, 2048, 1024, 512]
    private static let sendCaps: [Double] = [0, 30, 15, 10, 5]

    var body: some View {
        let snap = model.snapshot
        let s = snap.settings
        Form {
            Section {
                LabeledContent("Exposure", value: String(format: "%.0f µs", exposure))
                Slider(value: logBinding($exposure), in: logRange(exposureRange(snap.limits))) {
                    Text("Exposure")
                } onEditingChanged: { active in
                    editing = active
                    if !active { model.apply(LinkCommand(exposureUs: exposure)) }
                }
                LabeledContent("ISO", value: String(format: "%.0f", iso))
                Slider(value: $iso, in: range(snap.limits.iso, fallback: 32 ... 3200)) {
                    Text("ISO")
                } onEditingChanged: { active in
                    editing = active
                    if !active { model.apply(LinkCommand(iso: iso)) }
                }
                Picker("Frame rate", selection: Binding(
                    get: { s.fps.rounded() },
                    set: { model.apply(LinkCommand(fps: $0)) })) {
                    ForEach(frameRates(snap.limits), id: \.self) { Text("\(Int($0)) fps").tag($0) }
                }
            } header: {
                Text("Camera (locked: no auto exposure)")
            } footer: {
                Text("ESPI Live's exposure, gain (ISO) and frame-rate sliders change these too.")
            }
            Section {
                Picker("Capture", selection: Binding(
                    get: { s.raw },
                    set: { model.apply(LinkCommand(raw: $0)) })) {
                    Text("Video (processed luma)").tag(false)
                    Text("Bayer RAW (sensor data)").tag(true)
                }
                Picker("RAW colour", selection: Binding(
                    get: { s.rawChannel },
                    set: { model.apply(LinkCommand(rawChannel: $0)) })) {
                    Text("Red (633 nm laser)").tag("red")
                    Text("Green (532 nm laser)").tag("green")
                    Text("Blue").tag("blue")
                }
            } header: {
                Text("Capture mode")
            } footer: {
                Text("RAW takes stills (about 5–10 per second) and sends one colour "
                     + "plane of the sensor's data, without the phone's noise reduction.")
            }
            Section("Focus") {
                LabeledContent("Lens position", value: String(format: "%.3f", lens))
                Slider(value: $lens, in: 0 ... 1) { Text("Lens position") } onEditingChanged: { active in
                    editing = active
                    if !active { model.apply(LinkCommand(lensPosition: lens)) }
                }
                Button("Autofocus once, then lock") { model.apply(LinkCommand(refocus: true)) }
            }
            Section {
                Picker("Size", selection: Binding(
                    get: { "\(s.width)x\(s.height)" },
                    set: { id in
                        if let format = snap.formats.first(where: { $0.id == id }) {
                            model.apply(LinkCommand(width: format.width, height: format.height))
                        }
                    })) {
                    ForEach(snap.formats) { format in
                        Text("\(format.width) × \(format.height) (≤ \(Int(format.maxFps)) fps)")
                            .tag(format.id)
                    }
                }
                Picker("Centre crop", selection: Binding(
                    get: { s.cropWidth },
                    set: { model.apply(LinkCommand(cropWidth: $0, cropHeight: $0)) })) {
                    ForEach(Self.crops, id: \.self) { Text($0 == 0 ? "Full frame" : "\($0) × \($0)").tag($0) }
                }
                Picker("Send at most", selection: Binding(
                    get: { s.maxSendFps },
                    set: { model.apply(LinkCommand(maxSendFps: $0)) })) {
                    ForEach(Self.sendCaps, id: \.self) {
                        Text($0 == 0 ? "Every frame" : "\(Int($0)) fps").tag($0)
                    }
                }
            } header: {
                Text("Image sent")
            } footer: {
                Text("USB 2 iPhones carry about 30 MB/s: 1920 × 1080 at 15 fps, or a crop. "
                     + "Frames are never compressed; one is skipped when USB is busy.")
            }
            Section {
                Toggle("Show preview", isOn: $model.previewOn)
            } footer: {
                Text("Keep this app open and the phone unlocked while ESPI Live runs. "
                     + "Tap Trust when the phone asks about the computer.")
            }
        }
        .onChange(of: model.snapshot.settings) { _, new in
            guard !editing else { return }
            exposure = new.exposureUs
            iso = new.iso
            if new.lensPosition >= 0 { lens = new.lensPosition }
        }
    }

    private func exposureRange(_ limits: CaptureLimits) -> ClosedRange<Double> {
        let low = max(limits.exposureUs[0], 10)
        let high = min(max(limits.exposureUs[1], low + 1), 100_000)
        return low ... high
    }

    private func range(_ pair: [Double], fallback: ClosedRange<Double>) -> ClosedRange<Double> {
        pair.count == 2 && pair[0] < pair[1] ? pair[0] ... pair[1] : fallback
    }

    private func logRange(_ range: ClosedRange<Double>) -> ClosedRange<Double> {
        log10(range.lowerBound) ... log10(range.upperBound)
    }

    private func logBinding(_ value: Binding<Double>) -> Binding<Double> {
        Binding(get: { log10(max(value.wrappedValue, 1)) },
                set: { value.wrappedValue = pow(10, $0).rounded() })
    }

    private func frameRates(_ limits: CaptureLimits) -> [Double] {
        let top = limits.fps.count == 2 ? limits.fps[1] : 30
        let current = model.snapshot.settings.fps.rounded()
        var rates = [1.0, 2, 5, 10, 15, 20, 24, 25, 30, 60].filter { $0 <= top }
        if !rates.contains(current) { rates.append(current) }
        return rates.sorted()
    }
}
