import SwiftUI
import ARKit

struct CaptureView: View {
    @EnvironmentObject var pipeline: AnalysisPipeline
    
    var body: some View {
        ZStack {
            ARViewContainer(session: pipeline.captureManager.session)
                .edgesIgnoringSafeArea(.all)
            
            VStack {
                Spacer()
                
                if pipeline.isProcessing {
                    ProgressView("Analyzing Wheal Matrix...")
                        .padding()
                        .background(Color.black.opacity(0.7))
                        .foregroundColor(.white)
                        .cornerRadius(10)
                } else if !pipeline.latestResults.isEmpty {
                    ResultsView()
                }
                
                Button(action: {
                    if pipeline.isProcessing {
                        pipeline.stopCapture()
                    } else {
                        pipeline.startCapture()
                    }
                }) {
                    Text(pipeline.isProcessing ? "Stop Scanner" : "Start LiDAR Scan")
                        .font(.headline)
                        .foregroundColor(.white)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(pipeline.isProcessing ? Color.red : Color.blue)
                        .cornerRadius(12)
                        .padding(.horizontal, 40)
                        .padding(.bottom, 20)
                }
            }
        }
        .onAppear {
            pipeline.startCapture()
        }
        .onDisappear {
            pipeline.stopCapture()
        }
    }
}

struct ARViewContainer: UIViewRepresentable {
    var session: ARSession
    
    func makeUIView(context: Context) -> ARSCNView {
        let arView = ARSCNView(frame: .zero)
        arView.session = session // Link the preview UI tightly to our custom-configured LiDAR session
        arView.automaticallyUpdatesLighting = true
        return arView
    }
    
    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}


