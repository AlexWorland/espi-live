import SwiftUI

@main
struct ESPICameraApp: App {
    @State private var model = AppModel()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
                .task { await model.start() }
                .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
                .onChange(of: scenePhase) { _, phase in
                    // iOS stops the camera in the background; pick up again on return.
                    if phase == .active { Task { await model.start() } }
                }
        }
    }
}
