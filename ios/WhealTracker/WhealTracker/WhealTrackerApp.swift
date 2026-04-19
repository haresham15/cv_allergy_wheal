import SwiftUI

@main
struct WhealTrackerApp: App {
    @StateObject private var pipeline = AnalysisPipeline()
    @StateObject private var gridModel = AllergenGridModel()
    
    @State private var isConfiguring = true

    var body: some Scene {
        WindowGroup {
            if isConfiguring {
                ConfigView(gridModel: gridModel, isConfiguring: $isConfiguring)
            } else {
                CaptureView()
                    .environmentObject(pipeline)
                    .environmentObject(gridModel)
                    .onAppear {
                        // Pass grid into pipeline
                        pipeline.gridModel = gridModel
                    }
            }
        }
    }
}
