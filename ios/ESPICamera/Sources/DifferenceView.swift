// The full-screen on-phone difference view: |image(n) - image(n-1)| from the
// rawest source the phone has, with the pipeline always named on screen and
// exposure, ISO and focus adjustable without leaving.

import MetalKit
import SwiftUI

/// Half-extent, in clip space, of a quad that shows a landscape `texture`
/// turned 90° to portrait and fitted inside `view` (both in pixels).
func displayQuad(texture: (w: Int, h: Int), view: (w: Int, h: Int)) -> SIMD2<Float> {
    guard texture.w > 0, texture.h > 0, view.w > 0, view.h > 0 else { return SIMD2(1, 1) }
    let imageWidth = Double(texture.h), imageHeight = Double(texture.w)  // turned 90°
    let scale = min(Double(view.w) / imageWidth, Double(view.h) / imageHeight)
    return SIMD2(Float(imageWidth * scale / Double(view.w)),
                 Float(imageHeight * scale / Double(view.h)))
}

struct DifferenceView: View {
    @Bindable var model: AppModel
    @State private var exposure = 8000.0
    @State private var iso = 0.0
    @State private var lens = 0.5
    @State private var editing = false

    var body: some View {
        let snap = model.snapshot
        ZStack {
            Color.black.ignoresSafeArea()
            if let engine = model.streamer.engine {
                DifferenceMetalView(engine: engine).ignoresSafeArea()
            } else {
                Text("This phone has no Metal GPU, so the difference cannot be shown.")
                    .foregroundStyle(.white)
                    .padding()
            }
            VStack(spacing: 0) {
                header(snap)
                Spacer()
                controls(snap)
            }
        }
        .preferredColorScheme(.dark)
        .onAppear { sync(snap.settings) }
        .onChange(of: model.snapshot.settings) { _, new in
            if !editing { sync(new) }
        }
    }

    private func header(_ snap: StreamerSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .top) {
                Text(snap.pipelineLabel ?? snap.differenceStatus ?? "Choosing the camera source…")
                    .font(.footnote.monospacedDigit().weight(.semibold))
                    .fixedSize(horizontal: false, vertical: true)
                Spacer()
                Button("Done") { model.setDifferenceView(false) }
                    .buttonStyle(.borderedProminent)
            }
            if let interval = snap.differenceIntervalMs {
                Text(String(format: "Δt %.0f ms between the images compared", interval))
                    .font(.caption.monospacedDigit())
            }
            if let note = snap.pipelineNote {
                Label(note, systemImage: "info.circle").foregroundStyle(.orange)
            }
            if snap.thermal == "serious" || snap.thermal == "critical" {
                Label("Phone is \(snap.thermal)ly hot", systemImage: "thermometer.high")
                    .foregroundStyle(.red)
            }
            if let interrupted = snap.interrupted {
                Label("Camera paused: \(interrupted)", systemImage: "pause.circle")
                    .foregroundStyle(.orange)
            }
            if let error = snap.error {
                Label(error, systemImage: "exclamationmark.triangle").foregroundStyle(.red)
            }
        }
        .font(.caption)
        .foregroundStyle(.white)
        .padding(10)
        .background(.black.opacity(0.6))
    }

    private func controls(_ snap: StreamerSnapshot) -> some View {
        VStack(spacing: 6) {
            sliderRow(String(format: "Exposure %.0f µs", exposure),
                      value: logBinding($exposure), in: logRange(exposureRange(snap.limits))) {
                model.apply(LinkCommand(exposureUs: exposure))
            }
            sliderRow(String(format: "ISO %.0f", iso), value: $iso,
                      in: sliderRange(snap.limits.iso, fallback: 32 ... 3200)) {
                model.apply(LinkCommand(iso: iso))
            }
            sliderRow(String(format: "Focus %.3f", lens), value: $lens, in: 0 ... 1) {
                model.apply(LinkCommand(lensPosition: lens))
            }
            Picker("Source", selection: Binding(
                get: { snap.differenceSource },
                set: { model.setDifferenceSource($0) })) {
                Text("Rawest (Bayer RAW)").tag(DifferenceSource.auto)
                Text("Luma video").tag(DifferenceSource.luma)
            }
            .pickerStyle(.segmented)
            Button("Autofocus once, then lock") { model.apply(LinkCommand(refocus: true)) }
                .buttonStyle(.bordered)
        }
        .font(.caption.monospacedDigit())
        .foregroundStyle(.white)
        .padding(10)
        .background(.black.opacity(0.6))
    }

    /// A labelled slider that sends its command when the finger lifts.
    private func sliderRow(_ title: String, value: Binding<Double>, in range: ClosedRange<Double>,
                           send: @escaping () -> Void) -> some View {
        HStack {
            Text(title).frame(width: 130, alignment: .leading)
            Slider(value: value, in: range) { Text(title) } onEditingChanged: { active in
                editing = active
                if !active { send() }
            }
        }
    }

    private func sync(_ settings: CaptureSettings) {
        exposure = settings.exposureUs
        iso = settings.iso
        if settings.lensPosition >= 0 { lens = settings.lensPosition }
    }
}

/// An MTKView that redraws the engine's newest difference image.
struct DifferenceMetalView: UIViewRepresentable {
    let engine: DifferenceEngine

    func makeCoordinator() -> DifferenceDisplay? {
        DifferenceDisplay(engine: engine)
    }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: engine.device)
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        view.preferredFramesPerSecond = 30
        view.delegate = context.coordinator
        return view
    }

    func updateUIView(_ view: MTKView, context: Context) {}
}

/// Draws `engine.latest` turned to portrait and fitted to the view.
@MainActor
final class DifferenceDisplay: NSObject, MTKViewDelegate {
    private let engine: DifferenceEngine
    private let commands: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private let sampler: MTLSamplerState

    init?(engine: DifferenceEngine) {
        let device = engine.device
        guard let commands = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary(),
              let sampler = device.makeSamplerState(descriptor: {
                  let descriptor = MTLSamplerDescriptor()
                  descriptor.minFilter = .linear
                  descriptor.magFilter = .linear
                  return descriptor
              }()) else { return nil }
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = library.makeFunction(name: "display_vertex")
        descriptor.fragmentFunction = library.makeFunction(name: "display_fragment")
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        guard let pipeline = try? device.makeRenderPipelineState(descriptor: descriptor) else {
            return nil
        }
        self.engine = engine
        self.commands = commands
        self.pipeline = pipeline
        self.sampler = sampler
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let pass = view.currentRenderPassDescriptor, let drawable = view.currentDrawable,
              let commandBuffer = commands.makeCommandBuffer(),
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { return }
        if let image = engine.latest {
            var extent = displayQuad(texture: (image.width, image.height),
                                     view: (Int(view.drawableSize.width),
                                            Int(view.drawableSize.height)))
            encoder.setRenderPipelineState(pipeline)
            encoder.setVertexBytes(&extent, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)
            encoder.setFragmentTexture(image, index: 0)
            encoder.setFragmentSamplerState(sampler, index: 0)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }
        encoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
